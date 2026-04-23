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
import logging
import uuid
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase
from dotenv import load_dotenv

from .base_agent import BaseAgent

load_dotenv()

logger = logging.getLogger(__name__)

class ReinforcementJudge(BaseAgent):
    """The 'Fact Checker' that certifies Explorer insights against City Audit rules."""

    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        # Initialize LLM for reasoning
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.0 # High precision, no creativity
        )
        
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.pwd = os.getenv("NEO4J_PASSWORD")
        self.last_judgment_: Dict[str, Any] = {}

    def run(self, service_name: str) -> Dict[str, Any]:
        """
        Evaluate the most recent insight for a service against its Golden rules.
        """
        logger.info(f"[ReinforcementJudge] Evaluating recent insights for: {service_name}")
        
        # 1. Fetch data for comparison
        data = self._fetch_reconciliation_data(service_name)
        if not data["insight"]:
            logger.info(f"[ReinforcementJudge] No new insights for {service_name} to judge.")
            return {"status": "skipped", "reason": "no_insight"}

        # 2. Perform AI Judgment
        judgment = self._deliberate(data)
        
        # 3. Apply changes to the Graph
        self._apply_judgment(data["insight_id"], judgment)
        
        self.last_judgment_ = judgment
        return judgment

    def _fetch_reconciliation_data(self, service_name: str) -> Dict[str, Any]:
        """Collects the Web Insight and the Golden Rule for comparison."""
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
        with driver.session() as session:
            # Get latest web insight
            insight_res = session.run("""
                MATCH (c:DocumentChunk {type: 'Insight', service: $service})
                RETURN c.id as id, c.content as content ORDER BY c.timestamp DESC LIMIT 1
            """, service=service_name).single()
            
            # Get related Golden data (Audit Reports) via relationship
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

    def _deliberate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Uses LLM to evaluate conflict and assign certification."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Dallas City Data Integrity Judge. Your goal is to certify new web insights against established Audit Objectives."),
            ("human", """
            NEW INSIGHT (from Web):
            {insight}
            
            ESTABLISHED AUDIT OBJECTIVES (Golden Data):
            {golden_rules}
            
            TASK:
            Compare the two. 
            - Is the New Insight a 'Correction' (overrides old info), an 'Addendum' (adds extra context), or 'Suspicious' (contradicts established rules without good evidence)?
            - Assign a Trust Score (0-1).
            
            Return ONLY JSON with keys: 'classification', 'trust_score', 'reasoning', 'is_certified' (bool).
            """)
        ])
        
        chain = prompt | self.llm
        try:
            # We use a simple wrap here, assuming standard JSON output from model or parser
            import json
            response = chain.invoke(data)
            # Basic cleanup of markdown if model wraps JSON in triple backticks
            clean_content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
        except Exception as e:
            logger.error(f"[ReinforcementJudge] Deliberation failed: {e}")
            return {"classification": "error", "is_certified": False, "trust_score": 0.0}

    def _apply_judgment(self, insight_id: str, judgment: Dict[str, Any]) -> None:
        """Saves the judgment result onto the node in Neo4j."""
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
        logger.info(f"[ReinforcementJudge] Node {insight_id} certified: {judgment.get('is_certified')}")

    def validate(self) -> dict:
        return {"passed": True, "issues": []}

    def report(self) -> dict:
        return self.last_judgment_
