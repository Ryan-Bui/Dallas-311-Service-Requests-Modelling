# Dallas 311: Expert GraphRAG Implementation

This document summarizes the transition of the Dallas 311 ML Pipeline from a basic descriptive model to an **Expert-Grade GraphRAG System** powered by Google Cloud Spanner.

---

## 🏛️ System Architecture

The system uses a **Hybrid Retrieval** model, combining structured relational logic with unstructured semantic search.

### 1. Structured Knowledge (The Graph)
- **Engine**: Google Cloud Spanner (Graph Property Graph)
- **Dialect**: GoogleSQL
- **Language**: GQL (Graph Query Language)
- **Key Entities**: 
    - `Services`: Specific 311 request types (e.g., Signal Malfunction).
    - `Departments`: Owners of the services (e.g., Transportation).
    - `AuditTopics`: Strategic goals from the City Manager’s Audit Plan.

### 2. Unstructured Knowledge (The Vector Store)
- **Engine**: Spanner Vector Search
- **Embeddings**: Google Vertex AI (`text-embedding-004`)
- **Sources**: Digital PDFs (Budgets, Fiscal Reports, Staffing Charts)
- **Retrieval**: Cosine Similarity search on document chunks.

---

## 🚀 Phases of Implementation

### Phase 5: Expert Ontology Design
- **Objective**: Move beyond "What" happened to "Why" it matters according to city policy.
- **Outcome**: Created a multi-hop relationship model linking operational data (311 requests) to strategic data (City Audits).

### Phase 6: Cloud Spanner Migration
- **Objective**: Scale the system from a local JSON mock to a production-grade managed database.
- **Outcome**: Provisioned the `mark4200` instance and implemented a schema using **SQL/PGQ**. This enabled native graph traversals (`MATCH ... RETURN`) which are far more efficient than nested JSON lookups.

### Phase 7: GraphRAG & PDF Ingestion
- **Objective**: Allow the AI to "read" city reports to understand current context (e.g., staffing shortages mentioned in a PDF).
- **Outcome**: 
    - Implemented **Vertex AI Vector Search**.
    - Created an automated ingestion pipeline (`ingest_pdfs.py`) that chunks PDFs and generates 768-dimension embeddings.
    - Upgraded the `explainability_chain` to perform **Hybrid Retrieval**: performing both a GQL graph lookup and a Vector semantic search simultaneously.

---

## 🧠 Reasoning Logic (The "Hybrid" Chain)

When a prediction is made (e.g., "Closure delay likely"), the **Explainability Agent** does the following:

1.  **Identity Retrieval**: Identifies the department and service type.
2.  **Graph Lookup**: Queries Spanner for official SLAs and Audit Focus areas for that department.
3.  **Vector Lookup**: Searches the `DocumentChunks` table for recent "Staffing" or "Budget" mentions related to that department.
4.  **Synthesis**: Injects both the **Graph (SLA rules)** and **Vector (PDF context)** into the LLM prompt.
5.  **Output**: Generates a professional narrative that cites both city policy and specific report details.

---

## 🛠️ Tech Stack Summary

- **Database**: Google Cloud Spanner (Graph + Vector)
- **AI Framework**: LangChain (LCEL)
- **Embeddings**: Vertex AI
- **LLM**: Groq / OpenAI / Vertex
- **GQL**: ISO-standard Graph Query Language
