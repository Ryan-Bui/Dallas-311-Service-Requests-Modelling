import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def clear_db():
    if not all([URI, USERNAME, PASSWORD]):
        print("Error: Required Neo4j credentials missing in .env")
        return

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    print("Clearing Neo4j Graph Database (DETACH DELETE)...")
    
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("Database wiped successfully.")

    driver.close()

if __name__ == "__main__":
    clear_db()
