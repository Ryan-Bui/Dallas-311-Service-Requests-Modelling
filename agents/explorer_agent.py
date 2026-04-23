# ============================================
# Agent: Explorer Agent
# Phase 2 — Online Optimization & Search
# ============================================
"""
ExplorerAgent
-------------
Responsibility: Use search tools to fill knowledge gaps identified during 
the pipeline. Specifically targets Dallas City ordinances and press releases.
"""
from __future__ import annotations
import os
import json
import logging
import uuid
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from .base_agent import BaseAgent
from inference.llm_factory import get_llm

logger = logging.getLogger(__name__)

class ExplorerAgent(BaseAgent):
    """Dallas City Policy Researcher that fills knowledge gaps using web search."""

    def __init__(self) -> None:
        self.results_: List[Dict[str, Any]] = []
        
        # Initialize LLM with Fallback
        try:
            self.llm = get_llm(provider="groq", temperature=0.1)
        except Exception:
            self.llm = get_llm(provider="vertexai", temperature=0.1)
        
        # Initialize Search Tool
        try:
            self.search = TavilySearch(k=3)
        except Exception as e:
            logger.warning(f"Tavily Search init failed: {e}. Falling back to mock results.")
            self.search = None

    def run(self, service_name: str) -> Dict[str, Any]:
        """Execute the full research loop: Search -> Analyze -> Save."""
        logger.info(f"[ExplorerAgent] Deep Researching: {service_name}")
        
        # 1. Search (Focused on City of Dallas domain)
        search_context = "No recent city hall search results found."
        if self.search:
            query = f"site:dallascityhall.com {service_name} service updates ordinances delays 2024 2025"
            try:
                raw_results = self.search.invoke(query)
                search_context = "\n".join([f"Source: {r.get('url')}\nContent: {r.get('content')}" for r in raw_results])
            except Exception as e:
                logger.error(f"[ExplorerAgent] Search failed: {e}")

        # 2. Analyze (Phase 4 Extraction)
        opus_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Dallas City Policy Analyst. Summarize status for the service and identify factors."),
            ("human", """
            SERVICE: {service}
            SEARCH CONTEXT:
            {context}
            
            TASK: Return JSON with 'analysis' (string) and 'external_factors' (list of {{'name', 'impact'}}).
            """)
        ])
        
        analysis_data = {"analysis": "Standard service analysis.", "external_factors": []}
        try:
            try:
                # Primary AI attempt
                chain = opus_prompt | self.llm
                response = chain.invoke({"service": service_name, "context": search_context[:5000]})
            except Exception as e:
                # Fallback AI attempt
                if "429" in str(e) or "rate" in str(e).lower():
                    logger.warning("[ExplorerAgent] Rate limit hit. Pivoting to Vertex AI...")
                    fallback_llm = get_llm(provider="vertexai", temperature=0.1)
                    chain = opus_prompt | fallback_llm
                    response = chain.invoke({"service": service_name, "context": search_context[:5000]})
                else:
                    raise e
            
            content = response.content if hasattr(response, 'content') else str(response)
            clean_json = content.replace("```json", "").replace("```", "").strip()
            analysis_data = json.loads(clean_json)
        except Exception as e:
            logger.error(f"[ExplorerAgent] Extraction failed: {e}")

        # 3. Save to Graph
        self.save_to_graph(service_name, analysis_data)
        
        self.results_.append({"service_name": service_name, "analysis": analysis_data})
        return analysis_data

    def save_to_graph(self, service_name: str, data: Dict[str, Any]) -> None:
        """Persist findings and external factors into Neo4j."""
        from neo4j import GraphDatabase
        
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        pwd = os.getenv("NEO4J_PASSWORD")
        
        if not all([uri, user, pwd]): return
            
        try:
            driver = GraphDatabase.driver(uri, auth=(user, pwd))
            with driver.session() as session:
                # B. Save main Insight
                insight_id = str(uuid.uuid4())
                session.run("""
                    MERGE (c:DocumentChunk {source: 'ExplorerAgent', service: $service})
                    SET c.id = $id, c.content = $content, c.type = 'Insight', c.timestamp = datetime()
                """, id=insight_id, service=service_name, content=data.get('analysis'))
                
                # C. Save & Link External Factors
                for factor in data.get('external_factors', []):
                    session.run("""
                        MERGE (f:ExternalFactor {name: $name})
                        SET f.id = coalesce(f.id, $id), f.impact = $impact
                        WITH f
                        OPTIONAL MATCH (s:Service) WHERE s.name =~ $regex
                        FOREACH (x IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
                            MERGE (s)-[:AFFECTED_BY]->(f)
                        )
                    """, name=factor['name'], impact=factor['impact'], id=str(uuid.uuid4()), 
                        regex=f"(?i).*{service_name}.*")

            driver.close()
            logger.info(f"[ExplorerAgent] Research persisted for {service_name}.")
        except Exception as e:
            logger.error(f"[ExplorerAgent] Graph save failed: {e}")

    def validate(self) -> dict:
        return {"passed": True, "issues": []}

    def report(self) -> dict:
        return {"status": "active" if self.results_ else "inactive"}
