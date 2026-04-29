#!/usr/bin/env python3
"""
Master Orchestrator for Neo4j Knowledge Graph Ingestion
========================================================

Runs all ingestion scripts in dependency order:
1. Clear old graph (optional)
2. Setup Neo4j schema and indices
3. Ingest structured knowledge (ontology, audit plans)
4. Ingest expert wisdom (metric definitions)
5. Ingest markdown/PDF documents (with embeddings)
6. Extract knowledge triples (from presentations/docs)
7. Ingest aggregated 311 operational data

Result: Clean, relationship-enriched knowledge graph
  - ~300 domain entity nodes (services, departments, factors)
  - Rich statistics on relationships (not bloat node chains)
  - Expert annotations and document grounding
  - Operational patterns from 311 data aggregates
"""

import sys
import importlib.util
from pathlib import Path

# Add repo and scripts directories to path so the orchestrator works whether it
# is launched from the repo root or from inside scripts/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

def run_ingestion(script_name, description):
    """Run a single ingestion script and report status."""
    print(f"\n{'='*70}")
    print(f"[{script_name}]")
    print(f"{description}")
    print(f"{'='*70}")
    
    try:
        module_name = script_name.replace(".py", "")
        script_path = Path(__file__).resolve().parent / script_name
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the main ingestion function (matches module naming convention)
        if module_name == "clear_neo4j":
            module.clear_db()
        elif module_name == "ingest_311_csv":
            module.ingest_311_csv()
        elif module_name == "ingest_to_neo4j":
            module.ingest_data()
        elif module_name == "ingest_markdowns":
            module.ingest_markdowns()
        elif module_name == "ingest_pdfs_to_neo4j":
            module.ingest_pdfs()
        elif module_name == "ingest_expert_wisdom":
            module.ingest_expert_wisdom()
        elif module_name == "extract_knowledge_triples":
            module.extract_triples()
        elif module_name == "setup_neo4j":
            module.setup_neo4j()
        else:
            print(f"[ERROR] Unknown ingestion script: {script_name}")
            return False
        
        print(f"\n✓ {script_name} completed successfully")
        return True
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Stopped while running {script_name}.")
        raise
    except Exception as e:
        print(f"\n✗ {script_name} failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        return False

def main():
    """Run all ingestion steps in order."""
    print("\n" + "="*70)
    print("DALLAS 311 NEO4J KNOWLEDGE GRAPH INGESTION ORCHESTRATOR")
    print("="*70)
    
    # Define ingestion pipeline
    pipeline = [
        ("clear_neo4j.py",
         "Clear stale graph data so old ServiceRequest/Location row nodes cannot survive"),

        ("setup_neo4j.py", 
         "Create Neo4j schema, constraints, and indices"),
        
        ("ingest_to_neo4j.py",
         "Ingest structured knowledge: Service Ontology + Audit Plans"),
        
        ("ingest_expert_wisdom.py",
         "Ingest expert metrics and operational definitions"),
        
        ("ingest_markdowns.py",
         "Ingest markdown reports with configured embeddings"),
        
        ("ingest_pdfs_to_neo4j.py",
         "Ingest PDF documents (budget, audit, performance reports)"),
        
        ("extract_knowledge_triples.py",
         "Extract relationship triples from presentations and documents"),
        
        ("ingest_311_csv.py",
         "Ingest aggregated 311 operational data (relationship-enriched, no SR nodes)"),
    ]
    
    results = []
    for script, description in pipeline:
        success = run_ingestion(script, description)
        results.append((script, success))
        
        if not success:
            print(f"\n[WARNING] {script} failed. Continue? (y/n): ", end="", flush=True)
            response = input().strip().lower()
            if response != "y":
                print("Stopping orchestration.")
                break

    # Print summary
    print(f"\n{'='*70}")
    print("INGESTION SUMMARY")
    print(f"{'='*70}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for script, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}  {script}")
    
    print(f"\nTotal: {successful}/{total} scripts completed successfully")
    
    if successful == total:
        print("\n🎉 Knowledge graph ingestion pipeline completed successfully!")
        print("\nNext steps:")
        print("  1. Verify graph in Neo4j: MATCH (n) RETURN count(n) as total_nodes")
        print("  2. Check relationships: MATCH (n)-[r]-(m) RETURN type(r), count(*)")
        print("  3. Run diagnostic queries to validate aggregations")
        return 0
    else:
        print(f"\n⚠️  Pipeline completed with {total - successful} failures.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user. No traceback shown.")
        sys.exit(130)
