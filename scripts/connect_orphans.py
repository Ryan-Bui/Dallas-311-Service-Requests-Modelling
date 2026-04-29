import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(override=True)

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def connect_orphans():
    if not all([URI, USERNAME, PASSWORD]):
        print("Error: Missing Neo4j credentials")
        return

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    
    with driver.session() as session:
        print("Linking all unmapped DocumentChunks back to their Source files...")
        result = session.run("""
            MATCH (c:DocumentChunk)
            MERGE (doc:DocumentSource {name: c.source})
            MERGE (c)-[:BELONGS_TO]->(doc)
            RETURN count(c) as count
        """)
        count = result.single()["count"]
        print(f"Successfully processed {count} DocumentChunks.")
        
    driver.close()

if __name__ == "__main__":
    connect_orphans()
