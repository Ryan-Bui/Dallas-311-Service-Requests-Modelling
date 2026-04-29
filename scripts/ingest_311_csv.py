import os
import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from neo4j import GraphDatabase
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
MIN_RECURS_IN_VOLUME = int(os.getenv("MIN_RECURS_IN_VOLUME", "2"))

def parse_date(date_str):
    """Parse date string and return datetime object."""
    if not date_str or date_str.strip() == "":
        return None
    try:
        # Format: '2025 Dec 29 08:35:09 PM'
        return datetime.strptime(date_str.strip(), "%Y %b %d %I:%M:%S %p")
    except:
        return None

def calculate_hours_to_close(created_date_str, closed_date_str):
    """Calculate hours between created and closed dates."""
    created = parse_date(created_date_str)
    closed = parse_date(closed_date_str)
    if created and closed:
        delta = closed - created
        return delta.total_seconds() / 3600
    return None

def ingest_311_csv():
    """
    Aggregate 311 service request data and create relationship-enriched graph.
    Instead of creating individual ServiceRequest nodes (14k+ bloat),
    aggregate by key dimensions and store statistics on relationships.
    """
    if not all([URI, USERNAME, PASSWORD]):
        print("Missing Neo4j credentials")
        return

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    csv_path = ROOT / "data" / "uploaded" / "sampleb.csv"
    
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return

    print(f"Aggregating 311 data from {csv_path} into relationship-enriched graph...")

    # Aggregate data by key dimensions
    aggregates = {
        "by_type_district": defaultdict(lambda: {
            "count": 0, "hours": [], "outcomes": defaultdict(int), "statuses": defaultdict(int)
        }),
        "by_type_dept": defaultdict(lambda: {
            "count": 0, "hours": [], "outcomes": defaultdict(int), "priorities": defaultdict(int)
        }),
        "by_district_dept": defaultdict(lambda: {
            "count": 0, "methods": defaultdict(int), "outcomes": defaultdict(int)
        }),
        "by_method": defaultdict(lambda: {"count": 0, "avg_hours": 0}),
        "by_priority": defaultdict(lambda: {"count": 0, "outcomes": defaultdict(int)}),
        "all_dimensions": defaultdict(set)  # Track all unique values
    }

    # Read and aggregate CSV
    row_count = 0
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dept = row.get("Department", "").strip() or "Unknown"
            req_type = row.get("Service Request Type", "").strip() or "Unknown"
            district = row.get("City Council District", "").strip() or "Unknown"
            if district and not str(district).startswith("District"):
                district = f"District {district}"
            
            priority = row.get("Priority", "").strip() or "Unknown"
            outcome = row.get("Outcome", "").strip() or "Unknown"
            status = row.get("Status", "").strip() or "Unknown"
            method = row.get("Method Received Description", "").strip() or "Unknown"
            
            # Calculate hours to close
            hours_to_close = calculate_hours_to_close(
                row.get("Created Date", ""),
                row.get("Closed Date", "")
            )

            # Aggregate by ServiceType + District
            key_td = (req_type, district)
            aggregates["by_type_district"][key_td]["count"] += 1
            if hours_to_close:
                aggregates["by_type_district"][key_td]["hours"].append(hours_to_close)
            aggregates["by_type_district"][key_td]["outcomes"][outcome] += 1
            aggregates["by_type_district"][key_td]["statuses"][status] += 1

            # Aggregate by ServiceType + Department
            key_td2 = (req_type, dept)
            aggregates["by_type_dept"][key_td2]["count"] += 1
            if hours_to_close:
                aggregates["by_type_dept"][key_td2]["hours"].append(hours_to_close)
            aggregates["by_type_dept"][key_td2]["outcomes"][outcome] += 1
            aggregates["by_type_dept"][key_td2]["priorities"][priority] += 1

            # Aggregate by District + Department
            key_dd = (district, dept)
            aggregates["by_district_dept"][key_dd]["count"] += 1
            aggregates["by_district_dept"][key_dd]["methods"][method] += 1
            aggregates["by_district_dept"][key_dd]["outcomes"][outcome] += 1

            # Aggregate by Method
            aggregates["by_method"][method]["count"] += 1
            if hours_to_close:
                # Running average
                old_avg = aggregates["by_method"][method]["avg_hours"]
                n = aggregates["by_method"][method]["count"]
                aggregates["by_method"][method]["avg_hours"] = (old_avg * (n - 1) + hours_to_close) / n

            # Aggregate by Priority
            aggregates["by_priority"][priority]["count"] += 1
            aggregates["by_priority"][priority]["outcomes"][outcome] += 1

            # Track all unique values
            aggregates["all_dimensions"]["services"].add(req_type)
            aggregates["all_dimensions"]["districts"].add(district)
            aggregates["all_dimensions"]["departments"].add(dept)
            aggregates["all_dimensions"]["priorities"].add(priority)
            aggregates["all_dimensions"]["outcomes"].add(outcome)
            aggregates["all_dimensions"]["statuses"].add(status)
            aggregates["all_dimensions"]["methods"].add(method)

            row_count += 1
            if row_count % 5000 == 0:
                print(f"  Processed {row_count} records...")

    print(f"Read {row_count} records. Now ingesting aggregated relationships...")

    # Ingest into Neo4j
    with driver.session() as session:
        # Create constraints
        for label in ["Department", "ServiceRequestType", "CityCouncilDistrict", 
                      "Priority", "Outcome", "Status", "MethodReceived"]:
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")

        # 1. Create dimension nodes (batched)
        print("Creating dimension nodes...")
        
        # Batch create all Services
        services = [{"name": s} for s in aggregates["all_dimensions"]["services"]]
        session.run("UNWIND $batch AS item MERGE (s:ServiceRequestType {name: item.name})", batch=services)
        
        # Batch create all Districts
        districts = [{"name": d} for d in aggregates["all_dimensions"]["districts"]]
        session.run("UNWIND $batch AS item MERGE (d:CityCouncilDistrict {name: item.name})", batch=districts)
        
        # Batch create all Departments
        depts = [{"name": d} for d in aggregates["all_dimensions"]["departments"]]
        session.run("UNWIND $batch AS item MERGE (d:Department {name: item.name})", batch=depts)
        
        # Batch create all Priorities
        priorities = [{"name": p} for p in aggregates["all_dimensions"]["priorities"]]
        session.run("UNWIND $batch AS item MERGE (p:Priority {name: item.name})", batch=priorities)
        
        # Batch create all Outcomes
        outcomes = [{"name": o} for o in aggregates["all_dimensions"]["outcomes"]]
        session.run("UNWIND $batch AS item MERGE (o:Outcome {name: item.name})", batch=outcomes)
        
        # Batch create all Statuses
        statuses = [{"name": s} for s in aggregates["all_dimensions"]["statuses"]]
        session.run("UNWIND $batch AS item MERGE (s:Status {name: item.name})", batch=statuses)
        
        # Batch create all Methods
        methods = [{"name": m} for m in aggregates["all_dimensions"]["methods"]]
        session.run("UNWIND $batch AS item MERGE (m:MethodReceived {name: item.name})", batch=methods)

        # 2. Create aggregated relationships with statistics (batched)
        print("Creating aggregated relationships...")
        
        # ServiceType -> District (RECURS_IN with stats)
        td_rels = []
        for (service, district), stats in aggregates["by_type_district"].items():
            if stats["count"] < MIN_RECURS_IN_VOLUME:
                continue

            avg_hours = sum(stats["hours"]) / len(stats["hours"]) if stats["hours"] else 0
            max_hours = max(stats["hours"]) if stats["hours"] else 0
            median_hours = sorted(stats["hours"])[len(stats["hours"])//2] if stats["hours"] else 0
            success_rate = stats["outcomes"].get("Completed", 0) / stats["count"] if stats["count"] > 0 else 0
            
            td_rels.append({
                "service": service,
                "district": district,
                "volume": stats["count"],
                "avg_hours": round(avg_hours, 2),
                "median_hours": round(median_hours, 2),
                "max_hours": round(max_hours, 2),
                "success_rate": round(success_rate, 4),
                "outcomes": json.dumps(dict(stats["outcomes"])),
                "statuses": json.dumps(dict(stats["statuses"]))
            })
        
        session.run("""
            UNWIND $rels AS rel
            MATCH (st:ServiceRequestType {name: rel.service})
            MATCH (d:CityCouncilDistrict {name: rel.district})
            MERGE (st)-[r:RECURS_IN]->(d)
            SET r.request_volume = rel.volume,
                r.avg_hours_to_close = rel.avg_hours,
                r.median_hours_to_close = rel.median_hours,
                r.max_hours_to_close = rel.max_hours,
                r.success_rate = rel.success_rate,
                r.outcome_distribution = rel.outcomes,
                r.status_distribution = rel.statuses,
                r.last_updated = datetime()
        """, rels=td_rels)

        # ServiceType -> Department (HANDLED_BY with stats)
        td2_rels = []
        for (service, dept), stats in aggregates["by_type_dept"].items():
            avg_hours = sum(stats["hours"]) / len(stats["hours"]) if stats["hours"] else 0
            
            td2_rels.append({
                "service": service,
                "dept": dept,
                "count": stats["count"],
                "avg_hours": round(avg_hours, 2),
                "outcomes": json.dumps(dict(stats["outcomes"])),
                "priorities": json.dumps(dict(stats["priorities"]))
            })
        
        session.run("""
            UNWIND $rels AS rel
            MATCH (st:ServiceRequestType {name: rel.service})
            MATCH (d:Department {name: rel.dept})
            MERGE (st)-[r:HANDLED_BY]->(d)
            SET r.request_count = rel.count,
                r.avg_hours_to_close = rel.avg_hours,
                r.outcome_distribution = rel.outcomes,
                r.priority_distribution = rel.priorities,
                r.last_updated = datetime()
        """, rels=td2_rels)

        # District -> Department (OVERLAPS_WITH with stats)
        dd_rels = []
        for (district, dept), stats in aggregates["by_district_dept"].items():
            dd_rels.append({
                "district": district,
                "dept": dept,
                "count": stats["count"],
                "methods": json.dumps(dict(stats["methods"])),
                "outcomes": json.dumps(dict(stats["outcomes"]))
            })
        
        session.run("""
            UNWIND $rels AS rel
            MATCH (d:CityCouncilDistrict {name: rel.district})
            MATCH (dept:Department {name: rel.dept})
            MERGE (d)-[r:OVERLAPS_WITH]->(dept)
            SET r.request_volume = rel.count,
                r.method_distribution = rel.methods,
                r.outcome_distribution = rel.outcomes,
                r.last_updated = datetime()
        """, rels=dd_rels)

        # Update Method nodes with stats
        method_updates = []
        for method, stats in aggregates["by_method"].items():
            method_updates.append({
                "method": method,
                "count": stats["count"],
                "avg_hours": round(stats["avg_hours"], 2)
            })
        
        session.run("""
            UNWIND $updates AS update
            MERGE (m:MethodReceived {name: update.method})
            SET m.total_requests = update.count,
                m.avg_hours_to_close = update.avg_hours,
                m.last_updated = datetime()
        """, updates=method_updates)

        # Update Priority nodes with stats
        priority_updates = []
        for priority, stats in aggregates["by_priority"].items():
            priority_updates.append({
                "priority": priority,
                "count": stats["count"],
                "outcomes": json.dumps(dict(stats["outcomes"]))
            })
        
        session.run("""
            UNWIND $updates AS update
            MERGE (p:Priority {name: update.priority})
            SET p.request_count = update.count,
                p.outcome_distribution = update.outcomes,
                p.last_updated = datetime()
        """, updates=priority_updates)

    print(f"\n✓ Aggregated graph ingestion completed!")
    print(f"  Total records processed: {row_count}")
    print(f"  Services: {len(aggregates['all_dimensions']['services'])}")
    print(f"  Districts: {len(aggregates['all_dimensions']['districts'])}")
    print(f"  Departments: {len(aggregates['all_dimensions']['departments'])}")
    print(f"  Relationships created with enriched statistics (no bloat nodes)")
    driver.close()

if __name__ == "__main__":
    ingest_311_csv()
