import os
import uuid
import sys
from pathlib import Path
from pypdf import PdfReader
from neo4j import GraphDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from inference.embedding_factory import (
    create_embeddings_service,
    embed_documents_batch,
    get_embedding_dimensions,
    get_embedding_model,
    get_embedding_provider,
)

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

REPORTS_DIR = ROOT / "knowledge" / "reports"

def ingest_pdfs():
    if not all([URI, USERNAME, PASSWORD]):
        print("Error: Neo4j credentials missing in .env")
        return

    if not REPORTS_DIR.exists():
        print(f"Directory {REPORTS_DIR} not found.")
        return

    embeddings_enabled = os.getenv("SKIP_EMBEDDINGS", os.getenv("SKIP_GEMINI_EMBEDDINGS", "")).lower() not in {"1", "true", "yes"}
    embeddings_service = None
    if embeddings_enabled:
        embeddings_service = create_embeddings_service()
        print(
            f"Using {get_embedding_provider()} embeddings: "
            f"{get_embedding_model()} ({get_embedding_dimensions()} dimensions)"
        )
    
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pdf_files = sorted(REPORTS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {REPORTS_DIR}.")
        return

    # Get department mapping from Neo4j
    with driver.session() as session:
        result = session.run("MATCH (d:Department) RETURN d.id as id, d.name as name")
        dept_map = {record["name"]: record["id"] for record in result}
        print(f"Loaded {len(dept_map)} departments from Neo4j for tagging.")

    for file_path in pdf_files:
        pdf_file = file_path.name
        print(f"Ingesting {pdf_file}...")
        print(f"Processing {pdf_file}...")
        
        try:
            reader = PdfReader(file_path)
            text = "".join([page.extract_text() + "\n" for page in reader.pages])
            chunks = text_splitter.split_text(text)
            print(f"  Created {len(chunks)} chunks.")

            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                embeddings, embeddings_enabled = embed_documents_batch(
                    embeddings_service,
                    batch_chunks,
                    embeddings_enabled,
                )
                
                with driver.session() as session:
                    source_type = "Golden" if "knowledge" in pdf_file.lower() else "Report"
                    
                    for offset, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
                        chunk_index = i + offset
                        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{pdf_file}:{chunk_index}"))
                        
                        # Ingest chunk
                        session.run("""
                            MERGE (doc:DocumentSource {name: $source})
                            MERGE (c:DocumentChunk {id: $id})
                            SET c.content = $content,
                                c.embedding = $embedding,
                                c.chunk_index = $chunk_index,
                                c.source = $source,
                                c.type = $type
                            MERGE (c)-[:BELONGS_TO]->(doc)
                        """, id=chunk_id, content=chunk_text, embedding=embedding, chunk_index=chunk_index, source=pdf_file, type=source_type)
                        
                        # Detect and Link Department
                        for d_name, d_id in dept_map.items():
                            if d_name.lower() in chunk_text.lower():
                                session.run("""
                                    MATCH (c:DocumentChunk {id: $c_id}), (d:Department {id: $d_id})
                                    MERGE (c)-[:RELEVANT_TO]->(d)
                                """, c_id=chunk_id, d_id=d_id)
                                break
            print(f"  Successfully ingested {pdf_file}.")
        except Exception as e:
            print(f"  Error processing {pdf_file}: {e}")

    driver.close()

if __name__ == "__main__":
    ingest_pdfs()
