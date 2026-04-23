import os
import pandas as pd
import uuid
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

ONTOLOGY_CSV = "data/reports/Service Audit and Performance Evaluation Ontology - Table 1.csv"
AUDIT_CSV = "data/reports/Fiscal Year 2026 Recommended Audit Work Plan Engagements - Table 1.csv"

def generate_id(name):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))

def ingest_data():
    if not all([URI, USERNAME, PASSWORD]):
        print("Error: Neo4j credentials missing in .env")
        return

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    
    # 1. Process Ontology CSV
    print(f"Reading {ONTOLOGY_CSV}...")
    df_ontology = pd.read_csv(ONTOLOGY_CSV)
    
    services = []
    departments = []
    factors = []
    ownership_rels = []
    impact_rels = []
    
    dept_map = {}
    factor_map = {}

    for _, row in df_ontology.iterrows():
        s_name = str(row['Service Type'])
        s_id = generate_id(s_name)
        
        services.append({
            "id": s_id,
            "name": s_name,
            "ert": str(row['Estimated Response Time (ERT)']),
            "sla": str(row['Target SLA %']),
            "priority": str(row['Socio-economic / Resource priority'])
        })
        
        d_name = str(row['Department'])
        if d_name not in dept_map:
            d_id = generate_id(d_name)
            dept_map[d_name] = d_id
            departments.append({"id": d_id, "name": d_name})
        
        ownership_rels.append({"s_id": s_id, "d_id": dept_map[d_name]})
        
        f_name = str(row['External Factor'])
        if f_name != "Not in source" and f_name not in factor_map:
            f_id = generate_id(f_name)
            factor_map[f_name] = f_id
            factors.append({"id": f_id, "name": f_name, "impact": str(row['Seasonality / Weather Impact'])})
        
        if f_name != "Not in source":
            impact_rels.append({"s_id": s_id, "f_id": factor_map[f_name]})

    # 2. Process Audit CSV
    print(f"Reading {AUDIT_CSV}...")
    df_audit = pd.read_csv(AUDIT_CSV)
    
    audit_topics = []
    monitored_rels = []
    
    for _, row in df_audit.iterrows():
        t_name = str(row['Topic'])
        t_id = generate_id(t_name)
        
        audit_topics.append({
            "id": t_id,
            "name": t_name,
            "objective": str(row['Preliminary Objective(s)']),
            "hours": str(row['Hours Estimate'])
        })
        
        d_raw = str(row['Department/ Division'])
        potential_depts = [d.strip() for d in d_raw.split('/')]
        if "Multiple Departments" in d_raw:
            for d_name in dept_map:
                monitored_rels.append({"d_id": dept_map[d_name], "t_id": t_id})
        else:
            for d_name in potential_depts:
                for known_dept, d_id in dept_map.items():
                    if d_name.lower() in known_dept.lower() or known_dept.lower() in d_name.lower():
                        monitored_rels.append({"d_id": d_id, "t_id": t_id})

    # 3. Batch Ingest into Neo4j
    with driver.session() as session:
        print("Ingesting Nodes...")
        session.run("UNWIND $batch AS row MERGE (s:Service {id: row.id}) SET s.name = row.name, s.ert = row.ert, s.sla = row.sla, s.priority = row.priority", batch=services)
        session.run("UNWIND $batch AS row MERGE (d:Department {id: row.id}) SET d.name = row.name", batch=departments)
        session.run("UNWIND $batch AS row MERGE (f:ExternalFactor {id: row.id}) SET f.name = row.name, f.impact = row.impact", batch=factors)
        session.run("UNWIND $batch AS row MERGE (a:AuditTopic {id: row.id}) SET a.name = row.name, a.objective = row.objective, a.hours = row.hours", batch=audit_topics)
        
        print("Ingesting Relationships...")
        session.run("UNWIND $batch AS row MATCH (s:Service {id: row.s_id}), (d:Department {id: row.d_id}) MERGE (s)-[:OWNED_BY]->(d)", batch=ownership_rels)
        session.run("UNWIND $batch AS row MATCH (s:Service {id: row.s_id}), (f:ExternalFactor {id: row.f_id}) MERGE (s)-[:AFFECTED_BY]->(f)", batch=impact_rels)
        session.run("UNWIND $batch AS row MATCH (d:Department {id: row.d_id}), (a:AuditTopic {id: row.t_id}) MERGE (d)-[:MONITORED_IN]->(a)", batch=monitored_rels)

    print("Neo4j Core Ingestion Complete.")
    driver.close()

if __name__ == "__main__":
    ingest_data()
