import os
import uuid
import sys
from pathlib import Path
from neo4j import GraphDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(override=True)

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

KNOWLEDGE_DIR = ROOT / "knowledge"

def ingest_markdowns():
    if not all([URI, USERNAME, PASSWORD]):
        print("Error: Neo4j credentials missing in .env")
        return

    # Use the same model as the main RAG chain for consistency
    embeddings_service = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    try:
        driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
        driver.verify_connectivity()
    except Exception as e:
        print(f"Error: Could not connect to Neo4j. Please verify credentials in .env.\nDetails: {e}")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    files_to_ingest = [
        "intelligent_report 1.md",
        "intelligent_Report_filled.md"
    ]

    # Get department mapping from Neo4j for linking
    with driver.session() as session:
        result = session.run("MATCH (d:Department) RETURN d.id as id, d.name as name")
        dept_map = {record["name"]: record["id"] for record in result}
        print(f"Loaded {len(dept_map)} departments from Neo4j for tagging.")

    for filename in files_to_ingest:
        file_path = KNOWLEDGE_DIR / filename
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        print(f"Processing {filename}...")
        
        try:
            text = file_path.read_text(encoding="utf-8")
            chunks = text_splitter.split_text(text)
            print(f"  Created {len(chunks)} chunks.")

            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                embeddings = embeddings_service.embed_documents(batch_chunks)
                
                with driver.session() as session:
                    for chunk_text, embedding in zip(batch_chunks, embeddings):
                        chunk_id = str(uuid.uuid4())
                        
                        # Ingest chunk
                        session.run("""
                            MERGE (c:DocumentChunk {id: $id})
                            SET c.content = $content,
                                c.embedding = $embedding,
                                c.source = $source,
                                c.type = $type
                        """, id=chunk_id, content=chunk_text, embedding=embedding, source=filename, type="Report")
                        
                        # Detect and Link Department
                        for d_name, d_id in dept_map.items():
                            if d_name.lower() in chunk_text.lower():
                                session.run("""
                                    MATCH (c:DocumentChunk {id: $c_id}), (d:Department {id: $d_id})
                                    MERGE (c)-[:RELEVANT_TO]->(d)
                                """, c_id=chunk_id, d_id=d_id)
                                break
            print(f"  Successfully ingested {filename}.")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    driver.close()

if __name__ == "__main__":
    ingest_markdowns()
