import os
import uuid
from pypdf import PdfReader
from google.cloud import spanner
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
INSTANCE_ID = os.getenv("GCP_INSTANCE_ID")
DATABASE_ID = os.getenv("GCP_DATABASE_ID")
REPORTS_DIR = "data/reports"

def ingest_pdfs():
    """Extract text from PDFs, chunk it, embed it, and store it in Spanner."""
    if not os.path.exists(REPORTS_DIR):
        print(f"Directory {REPORTS_DIR} not found. Please create it and add PDFs.")
        return

    # Initialize Vertex AI Embeddings (Consistent with retrieval)
    from langchain_google_vertexai import VertexAIEmbeddings
    embeddings_service = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=PROJECT_ID
    )
    
    # Initialize Spanner Client
    client = spanner.Client(project=PROJECT_ID)
    instance = client.instance(INSTANCE_ID)
    database = instance.database(DATABASE_ID)

    # Initialize Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    pdf_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDFs found in {REPORTS_DIR}.")
        return

    for pdf_file in pdf_files:
        # Check if file is already ingested to support resuming
        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                "SELECT COUNT(1) FROM DocumentChunks WHERE SourceFile = @file",
                params={"file": pdf_file},
                param_types={"file": spanner.param_types.STRING}
            )
            count = list(results)[0][0]
            if count > 0:
                print(f"Skipping {pdf_file} (Already ingested {count} chunks).")
                continue

        file_path = os.path.join(REPORTS_DIR, pdf_file)
        print(f"Processing {pdf_file}...")
        
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Split into chunks
            chunks = text_splitter.split_text(text)
            print(f"  Created {len(chunks)} chunks.")

            # Generate embeddings and store in batches
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                embeddings = embeddings_service.embed_documents(batch_chunks)
                
                with database.batch() as batch:
                    # Determine source type
                    source_type = "Golden" if "knowledge" in pdf_file.lower() else "Report"
                    
                    for j, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
                        batch.insert(
                            table="DocumentChunks",
                            columns=("ChunkId", "Content", "Embedding", "SourceFile", "SourceType"),
                            values=[(
                                str(uuid.uuid4()),
                                chunk_text,
                                embedding,
                                pdf_file,
                                source_type
                            )]
                        )
            print(f"  Successfully ingested {pdf_file}.")
        except Exception as e:
            print(f"  Error processing {pdf_file}: {e}")

if __name__ == "__main__":
    ingest_pdfs()
