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
import logging
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

from .base_agent import BaseAgent

load_dotenv()

logger = logging.getLogger(__name__)

class ExplorerAgent(BaseAgent):
    """Dallas City Policy Researcher that fills knowledge gaps using web search."""

    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.results_: List[Dict[str, Any]] = []
        
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1
        )
        
        # Initialize Search Tool (Tavily is recommended for LangChain)
        self.search = TavilySearchResults(k=3)

    # ------------------------------------------------------------------
    def run(self, service_name: str) -> Dict[str, Any]:
        """
        Search for recent updates regarding a specific city service.
        
        Parameters
        ----------
        service_name : str
            The name of the service request type (e.g., 'Street Repair').
            
        Returns
        -------
        Dict[str, Any]
            Insights found, confidence score, and raw source snippets.
        """
        logger.info(f"[ExplorerAgent] Investigating knowledge gap for: {service_name}")
        
        # 1. Generate Search Queries
        query_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Dallas City Policy Researcher."),
            ("human", f"The current Knowledge Base shows high uncertainty for the service: '{service_name}'. Generate localized search queries to find recent city ordinances, seasonal delays, or departmental reorganizations at dallascityhall.com.")
        ])
        
        # Note: In a production setting, we'd use an agentic loop. 
        # For this script, we'll focus on the primary task.
    def run(self, service_name: str) -> Dict[str, Any]:
        """Execute the full research loop: Search -> Analyze -> Save."""
        logger.info(f"[ExplorerAgent] Deep Researching: {service_name}")
        
        # 1. Search (Focused on City of Dallas domain)
        query = f"site:dallascityhall.com {service_name} service updates ordinances delays 2024 2025"
        try:
            raw_results = self.search.invoke(query)
            # Flatten search snippets
            search_context = "\n".join([f"Source: {r.get('url')}\nContent: {r.get('content')}" for r in raw_results])
        except Exception as e:
            logger.error(f"[ExplorerAgent] Search failed: {e}")
            search_context = "No recent city hall search results found."

        # 2. Analyze (Phase 4 Extraction)
        opus_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Dallas City Policy Analyst. Identify service insights and external factors (weather, seasonal, etc)."),
            ("human", """
            SERVICE: {service}
            SEARCH CONTEXT:
            {context}
            
            TASK:
            1. Summarize the latest status for {service}.
            2. Identify specific EXTERNAL FACTORS (e.g., 'Summer Heat', 'Budget Backlog').
            
            Return ONLY JSON:
            {{ 'analysis': 'string', 'external_factors': [{{'name': 'factor', 'impact': 'desc'}}] }}
            """)
        ])
        
        chain = opus_prompt | self.llm
        try:
            import json
            response = chain.invoke({"service": service_name, "context": search_context[:5000]})
            clean_json = response.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
        except Exception as e:
            logger.error(f"[ExplorerAgent] Extraction failed: {e}")
            data = {"analysis": "Standard service analysis.", "external_factors": []}

        # 3. Save to Graph
        self.save_to_graph(service_name, data)
        return data

    def save_to_graph(self, service_name: str, data: Dict[str, Any]) -> None:
        """Persist findings and external factors into Neo4j."""
        from neo4j import GraphDatabase
        import uuid
        
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        pwd = os.getenv("NEO4J_PASSWORD")
        
        if not all([uri, user, pwd]): return
            
        try:
            driver = GraphDatabase.driver(uri, auth=(user, pwd))
            with driver.session() as session:
                # A. Link to Department (Smart Linking Logic)
                dept_res = session.run("MATCH (d:Department) RETURN d.name as name")
                dept_names = [r["name"] for r in dept_res]
                
                mapping_prompt = f"Map '{service_name}' to one of these: {dept_names}. Respond ONLY with the name."
                mapping_resp = self.llm.invoke(mapping_prompt).content.strip()
                target_dept = mapping_resp if mapping_resp in dept_names else None
                
                # B. Save main Insight
                insight_id = str(uuid.uuid4())
                session.run("""
                    MERGE (c:DocumentChunk {source: 'ExplorerAgent', service: $service})
                    SET c.id = $id, c.content = $content, c.type = 'Insight', c.timestamp = datetime()
                """, id=insight_id, service=service_name, content=data.get('analysis'))
                
                if target_dept:
                    session.run("MATCH (c:DocumentChunk {id: $id}), (d:Department {name: $dept}) MERGE (c)-[:RELEVANT_TO]->(d)", 
                                id=insight_id, dept=target_dept)

                # C. Save & Link External Factors (Resilient Linking)
                for factor in data.get('external_factors', []):
                    f_id = str(uuid.uuid4())
                    # Try to link to Service FIRST, fallback to Department if missing
                    session.run("""
                        MERGE (f:ExternalFactor {name: $name})
                        SET f.id = coalesce(f.id, $id), f.impact = $impact
                        WITH f
                        // Try Service Match
                        OPTIONAL MATCH (s:Service) WHERE s.name =~ $regex
                        FOREACH (x IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
                            MERGE (s)-[:AFFECTED_BY]->(f)
                        )
                        // Fallback to Department Link if Service match failed
                        WITH f, s
                        WHERE s IS NULL AND $dept IS NOT NULL
                        OPTIONAL MATCH (d:Department {name: $dept})
                        FOREACH (y IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
                            MERGE (d)-[:AFFECTED_BY]->(f)
                        )
                    """, name=factor['name'], impact=factor['impact'], id=f_id, 
                        regex=f"(?i).*{service_name}.*", dept=target_dept)

            driver.close()
            logger.info(f"[ExplorerAgent] Pushed grounded insight and {len(data.get('external_factors', []))} factors for {service_name}.")
        except Exception as e:
            logger.error(f"[ExplorerAgent] Graph save failed: {e}")

    # ------------------------------------------------------------------
    def validate(self) -> dict:
        """Check if insights were generated."""
        if not self.results_:
            return {"passed": False, "issues": ["No investigations performed."]}
        return {"passed": True, "issues": []}

    # ------------------------------------------------------------------
    def report(self) -> dict:
        """Summary of the recent investigation."""
        if not self.results_:
            return {"status": "inactive"}
        last_result = self.results_[-1]
        return {
            "service": last_result["service_name"],
            "has_insights": "analysis" in last_result,
            "search_count": len(last_result.get("raw_search", [])) if isinstance(last_result.get("raw_search"), list) else 0
        }
