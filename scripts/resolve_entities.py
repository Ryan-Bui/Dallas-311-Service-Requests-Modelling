import os
import sys
import re
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(override=True)

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def clean_entity_name(name):
    """
    Normalizes string variations to detect core semantic entity duplicates.
    Example: 'the AuthService', 'Auth Service', 'auth_service' -> 'authservice'
    """
    if not isinstance(name, str):
        return ""
    # 1. Lowercase
    name = name.lower()
    # 2. Strip standard prefixes
    name = re.sub(r"^(the |a |an )", "", name)
    # 3. Strip all non-alphanumeric characters
    name = re.sub(r"[^a-z0-9]", "", name)
    return name.strip()

def resolve_entities():
    if not all([URI, USERNAME, PASSWORD]):
        print("Error: Required Neo4j credentials missing in .env")
        return

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    print("Fetching all KnowledgeEntity nodes...")

    with driver.session() as session:
        result = session.run("MATCH (e:KnowledgeEntity) RETURN e.id as id, e.name as name")
        nodes = [{"id": record["id"], "name": record["name"]} for record in result]
        
    print(f"Loaded {len(nodes)} total nodes.")

    # Group by cleaned name
    groups = {}
    for node in nodes:
        key = clean_entity_name(node["name"])
        if not key:
            continue
        if key not in groups:
            groups[key] = []
        groups[key].append(node)

    # Filter out non-duplicates
    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"Found {len(duplicate_groups)} overlapping entity clusters to merge.")

    # For each group, merge into the canonical node
    for key, cluster in duplicate_groups.items():
        # Pick canonical: shortest name length or first alphabetical
        cluster_sorted = sorted(cluster, key=lambda x: (len(x["name"]), x["name"]))
        canonical = cluster_sorted[0]
        duplicates = cluster_sorted[1:]

        print(f"\nMerging cluster for '{key}':")
        print(f"  -> Canonical Entity: {canonical['name']} ({canonical['id']})")
        
        with driver.session() as session:
            for dup in duplicates:
                print(f"  -> Migrating relationships from duplicate: {dup['name']} ({dup['id']})")
                
                # 1. Migrate OUTGOING relationships
                session.run("""
                    MATCH (dup:KnowledgeEntity {id: $dup_id})-[r:KNOWLEDGE_REL]->(target)
                    MATCH (canon:KnowledgeEntity {id: $canon_id})
                    WHERE dup <> canon
                    MERGE (canon)-[new_r:KNOWLEDGE_REL {relation: r.relation, source: r.source}]->(target)
                    DELETE r
                """, dup_id=dup["id"], canon_id=canonical["id"])
                
                # 2. Migrate INCOMING relationships
                session.run("""
                    MATCH (source)-[r:KNOWLEDGE_REL]->(dup:KnowledgeEntity {id: $dup_id})
                    MATCH (canon:KnowledgeEntity {id: $canon_id})
                    WHERE dup <> canon
                    MERGE (source)-[new_r:KNOWLEDGE_REL {relation: r.relation, source: r.source}]->(canon)
                    DELETE r
                """, dup_id=dup["id"], canon_id=canonical["id"])
                
                # 3. Delete isolated duplicate node
                session.run("MATCH (dup:KnowledgeEntity {id: $dup_id}) DELETE dup", dup_id=dup["id"])
                
    driver.close()
    print("\n[SUCCESS] Entity Resolution execution concluded.")

if __name__ == "__main__":
    resolve_entities()
