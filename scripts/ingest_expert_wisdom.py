import os
import re
import sys
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env", override=True)

def ingest_expert_wisdom():
    file_path = ROOT / "knowledge" / "expert_metric_definitions.md"
    if not file_path.exists():
        print("Expert definitions file not found!")
        return

    content = file_path.read_text(encoding="utf-8")
    
    # Simple parser for the markdown sections
    # Looks for ## Metric: [Name] followed by Expert Insight and Operational Reality
    pattern = r"## Metric: (.*?)\n\*\*Expert Insight\*\*: (.*?)\n\*\*Operational Reality\*\*: (.*?)(?=\n##|$)"
    matches = re.findall(pattern, content, re.DOTALL)

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    
    with driver.session() as session:
        print(f"--- INGESTING EXPERT WISDOM ({len(matches)} items) ---")
        for metric, insight, reality in matches:
            metric = metric.strip()
            # Combine for the model
            combined_wisdom = f"TEAMS EXPERT ANALYSIS: {insight.strip()} | REAL-WORLD IMPACT: {reality.strip()}"
            
            session.run("""
                MERGE (w:ExpertWisdom {topic: $topic})
                SET w.content = $content, 
                    w.timestamp = datetime(),
                    w.source = 'Human Team'
            """, topic=metric, content=combined_wisdom)
            print(f"[DONE] Tagged wisdom for: {metric}")

    driver.close()
    print("\n[SUCCESS] Your team's human wisdom is now live in the Knowledge Graph!")

if __name__ == "__main__":
    ingest_expert_wisdom()
