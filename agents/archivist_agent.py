import os
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

from neo4j import GraphDatabase
from langchain_core.prompts import ChatPromptTemplate
from inference.llm_factory import get_llm

logger = logging.getLogger(__name__)

class KnowledgeArchivist:
    """
    Phase 5: Semantic Compression Agent.
    Consolidates redundant nodes into Master Insights to optimize the Knowledge Graph.
    """
    def __init__(self):
        # Initialize LLM with Fallback
        try:
            self.llm = get_llm(provider="groq", temperature=0)
        except Exception:
            self.llm = get_llm(provider="vertexai", temperature=0)
            
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.pwd = os.getenv("NEO4J_PASSWORD")

    def compress_knowledge(self):
        """Perform semantic merging of ExternalFactors."""
        logger.info("[Archivist] Beginning graph compression...")
        
        if not all([self.uri, self.user, self.pwd]):
            return "Credentials missing."

        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
            with driver.session() as session:
                # 1. Fetch all External Factors
                res = session.run("MATCH (f:ExternalFactor) RETURN f.name AS name, f.impact AS impact")
                factors = list(res)
                
                if len(factors) < 2:
                    driver.close()
                    return "Not enough nodes to compress."

                # 2. Use LLM to identify duplicates
                factor_list = "\n".join([f"- {r['name']}: {r['impact']}" for r in factors])
                
                prompt = ChatPromptTemplate.from_template("""
                    Identify semantically identical Dallas City operational factors.
                    Return ONLY JSON list: [{{ 'master': 'Name', 'duplicates': ['name1'], 'impact': 'Description' }}]
                    
                    FACTORS:
                    {list}
                """)
                
                # Resilient AI Execution
                try:
                    chain = prompt | self.llm
                    resp = chain.invoke({"list": factor_list})
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        logger.warning("[Archivist] Primary AI busy, activating Vertex AI fallback...")
                        fallback_llm = get_llm(provider="vertexai", temperature=0)
                        chain = prompt | fallback_llm
                        resp = chain.invoke({"list": factor_list})
                    else:
                        raise e

                content = resp.content if hasattr(resp, 'content') else str(resp)
                clean_json = content.replace("```json", "").replace("```", "").strip()
                merges = json.loads(clean_json)

                # 3. Apply Merges in Neo4j
                for m in merges:
                    master = m['master']
                    impact = m['impact']
                    duplicates = m['duplicates']
                    if not duplicates: continue
                    
                    logger.info(f"[Archivist] Merging {duplicates} into '{master}'")
                    session.run(\"\"\"
                        MERGE (m:ExternalFactor {name: $master})
                        SET m.impact = $impact, m.is_compressed = true
                        WITH m
                        UNWIND $dups AS d_name
                        MATCH (d:ExternalFactor {name: d_name})
                        WHERE d <> m
                        MATCH (s)-[r:AFFECTED_BY]->(d)
                        MERGE (s)-[:AFFECTED_BY]->(m)
                        DETACH DELETE d
                    \"\"\", master=master, impact=impact, dups=duplicates)

            driver.close()
            logger.info("[Archivist] Compression complete.")
            return f"Successfully processed {len(merges)} master clusters."
        except Exception as e:
            logger.error(f"[Archivist] Compression failed: {e}")
            return f"Error: {e}"

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    arch = KnowledgeArchivist()
    print(arch.compress_knowledge())
