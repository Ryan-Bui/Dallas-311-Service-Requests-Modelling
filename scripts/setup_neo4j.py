import os
import sys
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from inference.embedding_factory import (
    get_embedding_dimensions,
    get_embedding_model,
    get_embedding_provider,
)

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def setup_neo4j():
    if not all([URI, USERNAME, PASSWORD]):
        print("Error: Neo4j credentials missing in .env")
        return

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    
    with driver.session() as session:
        print("Creating constraints...")
        # Unique constraints for core nodes
        session.run("CREATE CONSTRAINT service_id_unique IF NOT EXISTS FOR (s:Service) REQUIRE s.id IS UNIQUE")
        session.run("CREATE CONSTRAINT dept_id_unique IF NOT EXISTS FOR (d:Department) REQUIRE d.id IS UNIQUE")
        session.run("CREATE CONSTRAINT factor_id_unique IF NOT EXISTS FOR (f:ExternalFactor) REQUIRE f.id IS UNIQUE")
        session.run("CREATE CONSTRAINT audit_id_unique IF NOT EXISTS FOR (a:AuditTopic) REQUIRE a.id IS UNIQUE")
        session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:DocumentChunk) REQUIRE c.id IS UNIQUE")

        embedding_dimensions = get_embedding_dimensions()
        print(
            f"Creating Neo4j Vector Index ({embedding_dimensions} dimensions, "
            f"{get_embedding_provider()}:{get_embedding_model()})..."
        )
        session.run("DROP INDEX chunk_embeddings IF EXISTS")
        # Vector index for PDF chunks
        # Using Neo4j 5.x syntax
        session.run(f"""
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:DocumentChunk) ON (c.embedding)
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: {embedding_dimensions},
                `vector.similarity_function`: 'cosine'
              }}
            }}
        """)
        
        print("Neo4j Schema Setup Complete.")
    
    driver.close()

if __name__ == "__main__":
    setup_neo4j()
