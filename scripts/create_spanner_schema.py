import os
from google.cloud import spanner
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
INSTANCE_ID = os.getenv("GCP_INSTANCE_ID")
DATABASE_ID = os.getenv("GCP_DATABASE_ID")

def create_schema():
    client = spanner.Client(project=PROJECT_ID)
    instance = client.instance(INSTANCE_ID)
    database = instance.database(DATABASE_ID)

    print(f"Creating schema in {DATABASE_ID}...")

    # Define Node Tables and Edge Tables
    # We use a Property Graph approach mapping to these tables
    statements = [
        # 1. Services Node Table
        """CREATE TABLE IF NOT EXISTS Services (
            ServiceId STRING(100) NOT NULL,
            ServiceType STRING(MAX),
            ERT STRING(100),
            TargetSLA STRING(50),
            ResourcePriority STRING(MAX)
        ) PRIMARY KEY (ServiceId)""",

        # 2. Departments Node Table
        """CREATE TABLE IF NOT EXISTS Departments (
            DeptId STRING(100) NOT NULL,
            Name STRING(MAX)
        ) PRIMARY KEY (DeptId)""",

        # 3. ExternalFactors Node Table
        """CREATE TABLE IF NOT EXISTS ExternalFactors (
            FactorId STRING(100) NOT NULL,
            FactorName STRING(MAX),
            ImpactDetails STRING(MAX)
        ) PRIMARY KEY (FactorId)""",

        # 4. AuditTopics Node Table
        """CREATE TABLE IF NOT EXISTS AuditTopics (
            TopicId STRING(100) NOT NULL,
            TopicName STRING(MAX),
            Objective STRING(MAX),
            AuditHours STRING(50)
        ) PRIMARY KEY (TopicId)""",

        # 5. Relationship Tables (Edges)
        """CREATE TABLE IF NOT EXISTS ServiceOwnedByDept (
            RelId STRING(100) NOT NULL,
            ServiceId STRING(100) NOT NULL,
            DeptId STRING(100) NOT NULL,
            FOREIGN KEY (ServiceId) REFERENCES Services (ServiceId),
            FOREIGN KEY (DeptId) REFERENCES Departments (DeptId)
        ) PRIMARY KEY (RelId)""",

        """CREATE TABLE IF NOT EXISTS ServiceAffectedByFactor (
            RelId STRING(100) NOT NULL,
            ServiceId STRING(100) NOT NULL,
            FactorId STRING(100) NOT NULL,
            FOREIGN KEY (ServiceId) REFERENCES Services (ServiceId),
            FOREIGN KEY (FactorId) REFERENCES ExternalFactors (FactorId)
        ) PRIMARY KEY (RelId)""",

        """CREATE TABLE IF NOT EXISTS DeptMonitoredInAudit (
            RelId STRING(100) NOT NULL,
            DeptId STRING(100) NOT NULL,
            TopicId STRING(100) NOT NULL,
            FOREIGN KEY (DeptId) REFERENCES Departments (DeptId),
            FOREIGN KEY (TopicId) REFERENCES AuditTopics (TopicId)
        ) PRIMARY KEY (RelId)""",

        # 6. DocumentChunks Table (for Vector Search)
        # Note: We DROP and CREATE to ensure vector_length is set correctly
        "DROP INDEX IF EXISTS DocumentChunks_Embedding_Index",
        "DROP TABLE IF EXISTS DocumentChunks",
        """CREATE TABLE DocumentChunks (
            ChunkId STRING(100) NOT NULL,
            Content STRING(MAX),
            Embedding ARRAY<FLOAT64>(vector_length=>768),
            SourceFile STRING(MAX),
            DeptId STRING(100),
            SourceType STRING(50)
        ) PRIMARY KEY (ChunkId)""",

        # 7. Vector Index for semantic search
        """CREATE VECTOR INDEX DocumentChunks_Embedding_Index
            ON DocumentChunks(Embedding)
            WHERE Embedding IS NOT NULL
            OPTIONS (
                distance_type = 'COSINE',
                tree_depth = 2,
                num_leaves = 1000
            )""",

        # 8. Create the Property Graph
        # Note: Spanner Graph syntax (GQL)
        """CREATE OR REPLACE PROPERTY GRAPH Dallas311Graph
            NODE TABLES (
                Services,
                Departments,
                ExternalFactors,
                AuditTopics
            )
            EDGE TABLES (
                ServiceOwnedByDept
                    SOURCE KEY (ServiceId) REFERENCES Services (ServiceId)
                    DESTINATION KEY (DeptId) REFERENCES Departments (DeptId)
                    LABEL OwnedBy,
                ServiceAffectedByFactor
                    SOURCE KEY (ServiceId) REFERENCES Services (ServiceId)
                    DESTINATION KEY (FactorId) REFERENCES ExternalFactors (FactorId)
                    LABEL AffectedBy,
                DeptMonitoredInAudit
                    SOURCE KEY (DeptId) REFERENCES Departments (DeptId)
                    DESTINATION KEY (TopicId) REFERENCES AuditTopics (TopicId)
                    LABEL MonitoredIn
            )"""
    ]

    try:
        operation = database.update_ddl(statements)
        print("Waiting for DDL operation to complete...")
        operation.result(600)  # 10 min timeout
        print("Schema and Property Graph created successfully.")
    except Exception as e:
        print(f"Error creating schema: {e}")

if __name__ == "__main__":
    create_schema()
