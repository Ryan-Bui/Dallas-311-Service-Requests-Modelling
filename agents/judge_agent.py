# ============================================
# Agent: Reinforcement Judge
# Phase 3 — Truth Reconciliation & Graph Health
# ============================================
"""
ReinforcementJudge
------------------
Responsibility: Acts as the quality control layer for the Knowledge Graph.
Compares ExplorerAgent insights against Golden Audit data and resolves conflicts.
"""
from __future__ import annotations
import os
import json
import logging
import uuid
from typing import List, Dict, Any
from datetime import datetime

from neo4j import GraphDatabase
from langchain_core.prompts import ChatPromptTemplate
from .base_agent import BaseAgent
from inference.llm_factory import get_llm

logger = logging.getLogger(__name__)

class ReinforcementJudge(BaseAgent):
    """The 'Fact Checker' that certifies Explorer insights against City Audit rules."""

    def __init__(self) -> None:
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.pwd = os.getenv("NEO4J_PASSWORD")
        self.last_judgment_: Dict[str, Any] = {}
        
        # Initialize LLM with Fallback
        try:
            self.llm = get_llm(provider="groq", temperature=0.0)
        except Exception:
            self.llm = get_llm(provider="vertexai", temperature=0.0)

    def run(self, service_name: str) -> Dict[str, Any]:
        """Evaluate the most recent insight for a service against its Golden rules."""
        logger.info(f"[ReinforcementJudge] Evaluating recent insights for: {service_name}")
        
        # 1. Fetch data for comparison
        data = self._fetch_reconciliation_data(service_name)
        if not data["insight"]:
            logger.info(f"[ReinforcementJudge] No new insights for {service_name} to judge.")
            return {"status": "skipped", "reason": "no_insight"}

        # 2. Perform AI Judgment
        judgment = self._deliberate(data)
        
        # 3. Apply changes to the Graph
        if data["insight_id"]:
            self._apply_judgment(data["insight_id"], judgment)
        
        self.last_judgment_ = judgment
        return judgment

    def _fetch_reconciliation_data(self, service_name: str) -> Dict[str, Any]:
        """Collects the Web Insight and the Golden Rule for comparison."""
        if not all([self.uri, self.user, self.pwd]):
            return {"insight_id": None, "insight": None, "golden_rules": []}
            
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
            with driver.session() as session:
                # Get latest web insight
                insight_res = session.run("""
                    MATCH (c:DocumentChunk {type: 'Insight', service: $service})
                    RETURN c.id as id, c.content as content ORDER BY c.timestamp DESC LIMIT 1
                """, service=service_name).single()
                
                # Get related Golden data (Audit Reports)
                golden_res = session.run("""
                    MATCH (c:DocumentChunk {type: 'Insight', service: $service})-[:RELEVANT_TO]->(d:Department)
                    MATCH (d)-[:MONITORED_IN]->(a:AuditTopic)
                    RETURN a.name as topic, a.objective as objective LIMIT 3
                """, service=service_name)
                
                golden_rules = [f"Topic: {r['topic']} | Objective: {r['objective']}" for r in golden_res]
                
            driver.close()
            return {
                "insight_id": insight_res["id"] if insight_res else None,
                "insight": insight_res["content"] if insight_res else None,
                "golden_rules": golden_rules
            }
        except Exception as e:
            logger.error(f"[ReinforcementJudge] Data fetch failed: {e}")
            return {"insight_id": None, "insight": None, "golden_rules": []}

    def _deliberate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Uses LLM to evaluate conflict and assign certification."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the University Professor analyzing student projects and a strict Data Integrity Judge. "),
            ("human", """
            NEW INSIGHT: {insight}
            GOLDEN RULES: {golden_rules}
            
            TASK: Return JSON with: 'classification', 'trust_score' (0-1), 'reasoning', 'is_certified' (bool).
            """)
        ])
        
        try:
            try:
                # Primary AI attempt
                chain = prompt | self.llm
                response = chain.invoke(data)
            except Exception as e:
                # Fallback AI attempt
                if "429" in str(e) or "rate" in str(e).lower():
                    fallback_llm = get_llm(provider="vertexai", temperature=0.0)
                    chain = prompt | fallback_llm
                    response = chain.invoke(data)
                else:
                    raise e
            
            content = response.content if hasattr(response, 'content') else str(response)
            clean_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
        except Exception as e:
            logger.error(f"[ReinforcementJudge] Deliberation failed: {e}")
            return {"classification": "error", "is_certified": False, "trust_score": 0.0}

    def _apply_judgment(self, insight_id: str, judgment: Dict[str, Any]) -> None:
        """Saves the judgment result onto the node in Neo4j."""
        if not all([self.uri, self.user, self.pwd]): return
        
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
            with driver.session() as session:
                session.run("""
                    MATCH (c:DocumentChunk {id: $id})
                    SET c.is_certified = $certified,
                        c.trust_score = $score,
                        c.judgment_reason = $reason,
                        c.classification = $cls
                """, id=insight_id, certified=judgment.get("is_certified", False), 
                     score=judgment.get("trust_score", 0.1), reason=judgment.get("reasoning", "Unknown"),
                     cls=judgment.get("classification", "unclassified"))
            driver.close()
        except Exception as e:
            logger.error(f"[ReinforcementJudge] Graph update failed: {e}")

    def validate(self) -> dict:
        return {"passed": True, "issues": []}

    def report(self) -> dict:
        return self.last_judgment_ if self.last_judgment_ else {"status": "idle"}
