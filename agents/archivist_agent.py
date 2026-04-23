import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from neo4j import GraphDatabase
from inference.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class KnowledgeArchivist:
    """
    Phase 5: Semantic Compression Agent.
    Consolidates redundant nodes into Master Insights to optimize the Knowledge Graph.
    """
    def __init__(self):
        self.llm = get_llm(temperature=0) # Deterministic for grouping
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.pwd = os.getenv("NEO4J_PASSWORD")

    def compress_knowledge(self):
        """Perform semantic merging of ExternalFactors."""
        logger.info("[Archivist] Beginning graph compression...")
        
        if not all([self.uri, self.user, self.pwd]):
            return "Credentials missing."

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
                Review this list of Dallas City operational factors:
                {list}
                
                Identify any groups that are SEMANTICALLY IDENTICAL (e.g., 'Heat' and 'Temperature').
                For each group, provide:
                1. The 'Master Name' (the best representative name).
                2. A list of 'Duplicates' to be merged into the master.
                3. A 'Consolidated Impact' string that combines the wisdom.
                
                Return ONLY a JSON list of objects:
                [{{ 'master': 'Name', 'duplicates': ['name1', 'name2'], 'impact': 'Combined info' }}]
            """)
            
            chain = prompt | self.llm
            try:
                import json
                resp = chain.invoke({"list": factor_list})
                clean_json = resp.content.replace("```json", "").replace("```", "").strip()
                merges = json.loads(clean_json)
            except Exception as e:
                logger.error(f"[Archivist] Clustering failed: {e}")
                driver.close()
                return f"Error: {e}"

            # 3. Apply Merges in Neo4j
            for m in merges:
                master = m['master']
                impact = m['impact']
                duplicates = m['duplicates']
                
                if not duplicates: continue
                
                logger.info(f"[Archivist] Merging {duplicates} into '{master}'")
                
                # Merge logic: Create master, move edges from duplicates, delete duplicates
                session.run("""
                    MERGE (m:ExternalFactor {name: $master})
                    SET m.impact = $impact, m.is_compressed = true
                    WITH m
                    UNWIND $dups AS d_name
                    MATCH (d:ExternalFactor {name: d_name})
                    WHERE d <> m
                    MATCH (s)-[r:AFFECTED_BY]->(d)
                    MERGE (s)-[:AFFECTED_BY]->(m)
                    DETACH DELETE d
                """, master=master, impact=impact, dups=duplicates)

        driver.close()
        logger.info("[Archivist] Compression complete.")
        return f"Successfully processed {len(merges)} master clusters."

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    arch = KnowledgeArchivist()
    print(arch.compress_knowledge())
