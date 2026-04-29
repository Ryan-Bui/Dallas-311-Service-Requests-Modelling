#!/usr/bin/env python3
"""Clear 311-related data from Neo4j to prepare for re-ingestion."""
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")


def clear_311_data():
    if not all([URI, USERNAME, PASSWORD]):
        print("Missing Neo4j credentials")
        return

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    with driver.session() as session:
        print("Clearing 311 operational data...")

        labels = [
            "ServiceRequest",
            "Location",
            "ServiceRequestType",
            "CityCouncilDistrict",
            "Priority",
            "MethodReceived",
            "Outcome",
            "Status",
        ]

        for label in labels:
            session.run(f"MATCH (n:{label}) DETACH DELETE n")
            print(f"  Deleted {label} nodes")

    driver.close()
    print("\n311 operational data cleared. Ready for aggregated re-ingestion.")


if __name__ == "__main__":
    clear_311_data()
