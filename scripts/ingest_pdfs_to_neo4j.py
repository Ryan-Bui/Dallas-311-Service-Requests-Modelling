import os
import uuid
from pypdf import PdfReader
from neo4j import GraphDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REPORTS_DIR = "data/reports"

def ingest_pdfs():
    if not all([URI, USERNAME, PASSWORD]):
        print("Error: Neo4j credentials missing in .env")
        return

    if not os.path.exists(REPORTS_DIR):
        print(f"Directory {REPORTS_DIR} not found.")
        return

    # Use the same model as the main RAG chain for consistency
    embeddings_service = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pdf_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDFs found in {REPORTS_DIR}.")
        return

    # Get department mapping from Neo4j
    with driver.session() as session:
        result = session.run("MATCH (d:Department) RETURN d.id as id, d.name as name")
        dept_map = {record["name"]: record["id"] for record in result}
        print(f"Loaded {len(dept_map)} departments from Neo4j for tagging.")

    for pdf_file in pdf_files:
        # Check if already ingested
        with driver.session() as session:
            count = session.run("MATCH (c:DocumentChunk {source: $file}) RETURN count(c) as count", file=pdf_file).single()["count"]
            if count > 0:
                print(f"Skipping {pdf_file} (Already ingested).")
                continue

        file_path = os.path.join(REPORTS_DIR, pdf_file)
        print(f"Processing {pdf_file}...")
        
        try:
            reader = PdfReader(file_path)
            text = "".join([page.extract_text() + "\n" for page in reader.pages])
            chunks = text_splitter.split_text(text)
            print(f"  Created {len(chunks)} chunks.")

            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                embeddings = embeddings_service.embed_documents(batch_chunks)
                
                with driver.session() as session:
                    source_type = "Golden" if "knowledge" in pdf_file.lower() else "Report"
                    
                    for chunk_text, embedding in zip(batch_chunks, embeddings):
                        chunk_id = str(uuid.uuid4())
                        
                        # Ingest chunk
                        session.run("""
                            MERGE (c:DocumentChunk {id: $id})
                            SET c.content = $content,
                                c.embedding = $embedding,
                                c.source = $source,
                                c.type = $type
                        """, id=chunk_id, content=chunk_text, embedding=embedding, source=pdf_file, type=source_type)
                        
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
