import os
from google.cloud import spanner
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
INSTANCE_ID = os.getenv("GCP_INSTANCE_ID")
DATABASE_ID = os.getenv("GCP_DATABASE_ID")

def patch_chunks():
    client = spanner.Client(project=PROJECT_ID)
    instance = client.instance(INSTANCE_ID)
    database = instance.database(DATABASE_ID)

    # 1. Fetch Department Mapping
    print("Fetching department mapping...")
    with database.snapshot() as snapshot:
        dept_results = snapshot.execute_sql("SELECT DeptId, Name FROM Departments")
        dept_map = {row[1]: row[0] for row in dept_results}
        print(f"Loaded {len(dept_map)} departments.")

    # 2. Fetch Chunks that have NULL DeptId
    print("Identifying chunks to patch...")
    with database.snapshot() as snapshot:
        # We only need ChunkId and Content to perform the matching
        chunk_results = snapshot.execute_sql("SELECT ChunkId, Content FROM DocumentChunks WHERE DeptId IS NULL")
        all_chunks = list(chunk_results)
        print(f"Total chunks to evaluate: {len(all_chunks)}")

    # 3. Patching
    if not all_chunks:
        print("No chunks need patching.")
        return

    print("Evaluating and patching chunks...")
    patch_count = 0
    
    # We'll patch in batches to stay within transaction limits
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        
        updates = []
        for chunk_id, content in batch:
            detected_dept_id = None
            for d_name, d_id in dept_map.items():
                if d_name.lower() in content.lower():
                    detected_dept_id = d_id
                    break
            
            if detected_dept_id:
                updates.append((chunk_id, detected_dept_id))
        
        if updates:
            def write_updates(transaction):
                transaction.update(
                    table="DocumentChunks",
                    columns=("ChunkId", "DeptId"),
                    values=updates
                )
            
            database.run_in_transaction(write_updates)
            patch_count += len(updates)
            print(f"  Patched {patch_count}/{len(all_chunks)} chunks...")

    print(f"\nPatching complete. Total chunks updated with DeptId: {patch_count}")

if __name__ == "__main__":
    patch_chunks()
