#!/usr/bin/env python3
import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

with driver.session() as session:
    print("Sample RECURS_IN relationships:")
    result = session.run("""
        MATCH (st:ServiceRequestType)-[r:RECURS_IN]->(d:CityCouncilDistrict)
        RETURN st.name, r.request_volume, r.avg_hours_to_close, d.name
        LIMIT 5
    """)
    for record in result:
        print(f"  {record[0]} -> {record[3]}: volume={record[1]}, avg_hours={record[2]}")
    
    print("\nSample HANDLED_BY relationships:")
    result = session.run("""
        MATCH (st:ServiceRequestType)-[r:HANDLED_BY]->(d:Department)
        RETURN st.name, r.request_count, r.avg_hours_to_close, d.name
        LIMIT 5
    """)
    for record in result:
        print(f"  {record[0]} -> {record[3]}: count={record[1]}, avg_hours={record[2]}")

driver.close()
