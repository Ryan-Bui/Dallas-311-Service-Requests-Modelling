import os
import csv
import uuid
import pandas as pd
from google.cloud import spanner
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
INSTANCE_ID = os.getenv("GCP_INSTANCE_ID")
DATABASE_ID = os.getenv("GCP_DATABASE_ID")

ONTOLOGY_CSV = "data/reports/Service Audit and Performance Evaluation Ontology - Table 1.csv"
AUDIT_CSV = "data/reports/Fiscal Year 2026 Recommended Audit Work Plan Engagements - Table 1.csv"

def generate_id(name):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))

def ingest_data():
    client = spanner.Client(project=PROJECT_ID)
    instance = client.instance(INSTANCE_ID)
    database = instance.database(DATABASE_ID)

    # 1. Process Ontology CSV
    print(f"Reading {ONTOLOGY_CSV}...")
    df_ontology = pd.read_csv(ONTOLOGY_CSV)
    
    services_data = []
    depts_data = []
    factors_data = []
    service_dept_edges = []
    service_factor_edges = []
    
    dept_map = {} # Name -> Id
    factor_map = {} # Name -> Id

    for _, row in df_ontology.iterrows():
        s_name = str(row['Service Type'])
        s_id = generate_id(s_name)
        
        services_data.append((
            s_id, 
            s_name, 
            str(row['Estimated Response Time (ERT)']), 
            str(row['Target SLA %']), 
            str(row['Socio-economic / Resource priority'])
        ))
        
        d_name = str(row['Department'])
        if d_name not in dept_map:
            d_id = generate_id(d_name)
            dept_map[d_name] = d_id
            depts_data.append((d_id, d_name))
        
        service_dept_edges.append((
            str(uuid.uuid4()), 
            s_id, 
            dept_map[d_name]
        ))
        
        f_name = str(row['External Factor'])
        if f_name != "Not in source" and f_name not in factor_map:
            f_id = generate_id(f_name)
            factor_map[f_name] = f_id
            factors_data.append((f_id, f_name, str(row['Seasonality / Weather Impact'])))
        
        if f_name != "Not in source":
            service_factor_edges.append((
                str(uuid.uuid4()), 
                s_id, 
                factor_map[f_name]
            ))

    # 2. Process Audit CSV
    print(f"Reading {AUDIT_CSV}...")
    df_audit = pd.read_csv(AUDIT_CSV)
    
    audit_topics_data = []
    dept_audit_edges = []
    
    for _, row in df_audit.iterrows():
        t_name = str(row['Topic'])
        t_id = generate_id(t_name)
        
        audit_topics_data.append((
            t_id, 
            t_name, 
            str(row['Preliminary Objective(s)']), 
            str(row['Hours Estimate'])
        ))
        
        # Link to departments (some mention Multiple Departments)
        d_raw = str(row['Department/ Division'])
        potential_depts = [d.strip() for d in d_raw.split('/')]
        if "Multiple Departments" in d_raw:
            # For now, link to all known departments if multiple
            for d_name in dept_map:
                dept_audit_edges.append((str(uuid.uuid4()), dept_map[d_name], t_id))
        else:
            for d_name in potential_depts:
                # Fuzzy match or direct match against our dept map
                for known_dept in dept_map:
                    if d_name.lower() in known_dept.lower() or known_dept.lower() in d_name.lower():
                        dept_audit_edges.append((str(uuid.uuid4()), dept_map[known_dept], t_id))

    # 3. Batch Insert into Spanner
    def write_to_spanner(transaction):
        transaction.insert_or_update("Services", columns=("ServiceId", "ServiceType", "ERT", "TargetSLA", "ResourcePriority"), values=services_data)
        transaction.insert_or_update("Departments", columns=("DeptId", "Name"), values=depts_data)
        transaction.insert_or_update("ExternalFactors", columns=("FactorId", "FactorName", "ImpactDetails"), values=factors_data)
        transaction.insert_or_update("AuditTopics", columns=("TopicId", "TopicName", "Objective", "AuditHours"), values=audit_topics_data)
        transaction.insert_or_update("ServiceOwnedByDept", columns=("RelId", "ServiceId", "DeptId"), values=service_dept_edges)
        transaction.insert_or_update("ServiceAffectedByFactor", columns=("RelId", "ServiceId", "FactorId"), values=service_factor_edges)
        transaction.insert_or_update("DeptMonitoredInAudit", columns=("RelId", "DeptId", "TopicId"), values=dept_audit_edges)

    print("Running transaction...")
    database.run_in_transaction(write_to_spanner)
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()
