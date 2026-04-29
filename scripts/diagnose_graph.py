#!/usr/bin/env python3
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")


def print_records(title, records, formatter):
    print(f"\n{title}")
    for record in records:
        print(formatter(record))


def main():
    if not all([URI, USERNAME, PASSWORD]):
        print("Missing Neo4j credentials")
        return 1

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    with driver.session() as session:
        print_records(
            "Node distribution:",
            session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(*) AS count
                ORDER BY count DESC
            """),
            lambda r: f"  {r['label']}: {r['count']}",
        )

        print_records(
            "Relationship distribution:",
            session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                ORDER BY count DESC
            """),
            lambda r: f"  {r['type']}: {r['count']}",
        )

        row_node_counts = session.run("""
            MATCH (sr:ServiceRequest)
            WITH count(sr) AS service_requests
            MATCH (loc:Location)
            RETURN service_requests, count(loc) AS locations
        """).single()
        print("\nRow-node check:")
        print(f"  ServiceRequest nodes: {row_node_counts['service_requests']}")
        print(f"  Location nodes: {row_node_counts['locations']}")

        one_offs = session.run("""
            MATCH (:ServiceRequestType)-[r:RECURS_IN]->(:CityCouncilDistrict)
            WHERE coalesce(r.request_volume, 0) < 2
            RETURN count(r) AS count
        """).single()["count"]
        print("\nOperational aggregate check:")
        print(f"  RECURS_IN edges with request_volume < 2: {one_offs}")

        doc_counts = session.run("""
            MATCH (c:DocumentChunk)
            RETURN count(c) AS chunks,
                   count { (c)-[:BELONGS_TO]->() } AS belongs_to_edges,
                   count { (c)-[:RELEVANT_TO]->() } AS relevant_to_edges
        """).single()
        print("\nDocument evidence layer:")
        print(f"  DocumentChunk nodes: {doc_counts['chunks']}")
        print(f"  BELONGS_TO edges from chunks: {doc_counts['belongs_to_edges']}")
        print(f"  RELEVANT_TO edges from chunks: {doc_counts['relevant_to_edges']}")

    driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
