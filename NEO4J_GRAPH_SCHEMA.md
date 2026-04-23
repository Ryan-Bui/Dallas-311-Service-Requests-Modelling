# Neo4j Graph Schema: 311 Intelligence Layer

This document defines the schema for the Neo4j Knowledge Graph used for RAG (Retrieval Augmented Generation) and ML explainability.

## 🟢 Nodes (Total: 1,730)

n
"(:Service {ert: 10 business days, name: Residential permit application first review, sla: Not in source, id: affbb8f6-c223-5c1e-95c8-e2788c98da15, priority: Equity priority areas})"
"(:Service {ert: Within SLA, name: Litter and high weed service requests, sla: 85.0% - 90.0%, id: 877fdcdb-8bad-5f3b-b3e5-75eab360384b, priority: Analyzed by Council District and City service area})"
"(:Service {ert: Not in source, name: Neighborhood Code (Rollup), sla: 96% (Responded to within ERT), id: c196e5ac-230b-59f7-b61a-dea317e3e6c0, priority: Not in source})"
"(:Service {ert: Within estimated response time, name: 311 service requests completion, sla: 96.0%, id: c11887e3-43bd-5817-949d-69eff9e4ad5c, priority: Not in source})"
"(:Service {ert: 3 days, name: Pothole repair, sla: 98.0%, id: a5dedca1-fdf8-5043-870f-8aae83f99934, priority: Not in source})"
"(:Service {ert: 120 minutes, name: Signal malfunction responses, sla: 91.0%, id: c628b539-235c-5da0-89b8-bb53d3811e59, priority: Not in source})"
"(:Service {ert: 9 minutes or less, name: EMS responses, sla: 90.0%, id: 18f0736f-4fb0-5681-8ef5-77643d85debb, priority: Not in source})"
"(:Service {ert: 5 days, name: Graffiti violations abatement, sla: 90.0%, id: 5b665a01-3714-50a5-a247-14f24d952685, priority: Not in source})"
"(:Service {ert: 5 days, name: Illegal dumping sites abatement, sla: 90.0%, id: fe992119-e22d-5aa0-bcba-ebc8af940ea9, priority: Not in source})"
"(:Service {ert: Not in source, name: Loose Aggressive Animals, sla: 90%, id: d39c31ee-b40a-5596-b46b-7a23208c5cf8, priority: Not in source})"
"(:Service {ert: 21 days, name: Service requests resolution, sla: 85.0%, id: 7a39a312-79cb-5261-a418-d0c8b7481ea1, priority: Not in source})"
"(:Service {ert: 220 seconds (Call Answering), name: 311 Queue (General), sla: 60%, id: 420a0cf6-9c9d-5699-a2d8-6144bb04c23a, priority: Not in source})"
"(:Service {ert: 90 seconds (Call Answering), name: Water related concerns (Water Queue), sla: 45%, id: b1d982fe-aac4-5568-8a8c-1a841a92f52b, priority: Not in source})"
"(:Service {ert: 90 seconds (Call Answering), name: Courts Queue, sla: 45%, id: f370b370-7f8d-5c75-85ec-0bd662ce6e4f, priority: Not in source})"
"(:Service {ert: 15 business days, name: Commercial permit application first review, sla: Not in source, id: 9fd75d42-b24d-5c4f-b51c-71ac45d7d481, priority: Not in source})"
"(:Service {ert: Not in source, name: S.M.A.R.T Summer Reading Challenge enrollments, sla: Not in source, id: c49f9677-4a04-5ae2-b739-ba44b7fb07f0, priority: Southern Dallas / Vulnerable populations})"
"(:Department {name: Development Services, id: 42baa4b9-47ef-5827-a596-ae9a12465f3e})"
"(:Department {name: Code Compliance Services, id: 8dd72552-cd9c-578d-a8b1-75b85c14b6bc})"
"(:Department {name: Code Compliance, id: 40e88a01-ebdb-58a3-83bb-c4a404541123})"
"(:Department {name: Public Works, id: c6d5acf6-713d-567f-8ac0-8dc4a8bba2c4})"
"(:Department {name: Transportation, id: 86e620ab-28b7-5c7a-8498-60e65b883f22})"
"(:Department {name: Dallas Fire-Rescue, id: 18b7edbd-998d-5436-8738-e174e9b3357a})"
"(:Department {name: MGT - Office of Homeless Solutions, id: 19f5d554-f727-54ea-8fae-ba4ac9b50881})"
"(:Department {name: 311 Customer Service Center, id: 92f70743-d7b4-57b4-9f75-d49b6c7c4821})"
"(:Department {name: Dallas Water Utilities, id: c901b3ce-9e48-558d-9e27-1440da723bbe})"

| Node Label                   | Description                                                                                      |
| :--------------------------- | :----------------------------------------------------------------------------------------------- |
| **`Service`**        | Specific types of 311 service requests (e.g., "Pothole Repair").                                 |
| **`Department`**     | City departments responsible for fulfilling service requests.                                    |
| **`AuditTopic`**     | High-level city audit categories or themes related to service performance.                       |
| **`DocumentChunk`**  | Vectorized text snippets from city documents/policies for RAG.                                   |
| **`ExternalFactor`** | Environmental or socio-economic events that impact service resolution (e.g., "Extreme Weather"). |

## 🔗 Relationships (Total: 297)

p
"(:Service {ert: 10 business days, name: Residential permit application first review, sla: Not in source, id: affbb8f6-c223-5c1e-95c8-e2788c98da15, priority: Equity priority areas})-[:OWNED_BY]->(:Department {name: Development Services, id: 42baa4b9-47ef-5827-a596-ae9a12465f3e})"
"(:Service {ert: 10 business days, name: Residential permit application first review, sla: Not in source, id: affbb8f6-c223-5c1e-95c8-e2788c98da15, priority: Equity priority areas})-[:AFFECTED_BY]->(:ExternalFactor {impact: Targets may vary based on the seasonality of the work., name: Seasonality, id: 2a3ab1bb-6c4a-565b-ae7a-3bfdf7c7c000})"
"(:Service {ert: Within SLA, name: Litter and high weed service requests, sla: 85.0% - 90.0%, id: 877fdcdb-8bad-5f3b-b3e5-75eab360384b, priority: Analyzed by Council District and City service area})-[:OWNED_BY]->(:Department {name: Code Compliance Services, id: 8dd72552-cd9c-578d-a8b1-75b85c14b6bc})"
"(:Service {ert: Within SLA, name: Litter and high weed service requests, sla: 85.0% - 90.0%, id: 877fdcdb-8bad-5f3b-b3e5-75eab360384b, priority: Analyzed by Council District and City service area})-[:AFFECTED_BY]->(:ExternalFactor {impact: Late spring & summer, name: Seasonality / Growing season, id: 3cbdd791-93e7-51a4-b16d-c730e008b405})"
"(:Service {ert: Not in source, name: Neighborhood Code (Rollup), sla: 96% (Responded to within ERT), id: c196e5ac-230b-59f7-b61a-dea317e3e6c0, priority: Not in source})-[:OWNED_BY]->(:Department {name: Code Compliance Services, id: 8dd72552-cd9c-578d-a8b1-75b85c14b6bc})"
"(:Service {ert: Within estimated response time, name: 311 service requests completion, sla: 96.0%, id: c11887e3-43bd-5817-949d-69eff9e4ad5c, priority: Not in source})-[:OWNED_BY]->(:Department {name: Code Compliance, id: 40e88a01-ebdb-58a3-83bb-c4a404541123})"
"(:Service {ert: Within estimated response time, name: 311 service requests completion, sla: 96.0%, id: c11887e3-43bd-5817-949d-69eff9e4ad5c, priority: Not in source})-[:AFFECTED_BY]->(:ExternalFactor {impact: Targets may vary based on the seasonality of the work., name: Seasonality, id: 2a3ab1bb-6c4a-565b-ae7a-3bfdf7c7c000})"
"(:Service {ert: 3 days, name: Pothole repair, sla: 98.0%, id: a5dedca1-fdf8-5043-870f-8aae83f99934, priority: Not in source})-[:OWNED_BY]->(:Department {name: Public Works, id: c6d5acf6-713d-567f-8ac0-8dc4a8bba2c4})"
"(:Service {ert: 3 days, name: Pothole repair, sla: 98.0%, id: a5dedca1-fdf8-5043-870f-8aae83f99934, priority: Not in source})-[:AFFECTED_BY]->(:ExternalFactor {impact: Targets may vary based on the seasonality of the work., name: Seasonality, id: 2a3ab1bb-6c4a-565b-ae7a-3bfdf7c7c000})"
"(:Service {ert: 120 minutes, name: Signal malfunction responses, sla: 91.0%, id: c628b539-235c-5da0-89b8-bb53d3811e59, priority: Not in source})-[:OWNED_BY]->(:Department {name: Transportation, id: 86e620ab-28b7-5c7a-8498-60e65b883f22})"
"(:Service {ert: 120 minutes, name: Signal malfunction responses, sla: 91.0%, id: c628b539-235c-5da0-89b8-bb53d3811e59, priority: Not in source})-[:AFFECTED_BY]->(:ExternalFactor {impact: Targets may vary based on the seasonality of the work., name: Seasonality, id: 2a3ab1bb-6c4a-565b-ae7a-3bfdf7c7c000})"
"(:Service {ert: 9 minutes or less, name: EMS responses, sla: 90.0%, id: 18f0736f-4fb0-5681-8ef5-77643d85debb, priority: Not in source})-[:OWNED_BY]->(:Department {name: Dallas Fire-Rescue, id: 18b7edbd-998d-5436-8738-e174e9b3357a})"
"(:Service {ert: 9 minutes or less, name: EMS responses, sla: 90.0%, id: 18f0736f-4fb0-5681-8ef5-77643d85debb, priority: Not in source})-[:AFFECTED_BY]->(:ExternalFactor {impact: Targets may vary based on the seasonality of the work., name: NFPA Standard 1710, id: 649153e8-96e7-5fa1-b3b5-b46101722b53})"
"(:Service {ert: 5 days, name: Graffiti violations abatement, sla: 90.0%, id: 5b665a01-3714-50a5-a247-14f24d952685, priority: Not in source})-[:OWNED_BY]->(:Department {name: Code Compliance, id: 40e88a01-ebdb-58a3-83bb-c4a404541123})"
"(:Service {ert: 5 days, name: Graffiti violations abatement, sla: 90.0%, id: 5b665a01-3714-50a5-a247-14f24d952685, priority: Not in source})-[:AFFECTED_BY]->(:ExternalFactor {impact: Targets may vary based on the seasonality of the work., name: Seasonality, id: 2a3ab1bb-6c4a-565b-ae7a-3bfdf7c7c000})"
"(:Service {ert: 5 days, name: Illegal dumping sites abatement, sla: 90.0%, id: fe992119-e22d-5aa0-bcba-ebc8af940ea9, priority: Not in source})-[:OWNED_BY]->(:Department {name: Code Compliance, id: 40e88a01-ebdb-58a3-83bb-c4a404541123})"
"(:Service {ert: 5 days, name: Illegal dumping sites abatement, sla: 90.0%, id: fe992119-e22d-5aa0-bcba-ebc8af940ea9, priority: Not in source})-[:AFFECTED_BY]->(:ExternalFactor {impact: Targets may vary based on the seasonality of the work., name: Seasonality, id: 2a3ab1bb-6c4a-565b-ae7a-3bfdf7c7c000})"
"(:Service {ert: Not in source, name: Loose Aggressive Animals, sla: 90%, id: d39c31ee-b40a-5596-b46b-7a23208c5cf8, priority: Not in source})-[:OWNED_BY]->(:Department {name: Code Compliance Services, id: 8dd72552-cd9c-578d-a8b1-75b85c14b6bc})"
"(:Service {ert: Not in source, name: Loose Aggressive Animals, sla: 90%, id: d39c31ee-b40a-5596-b46b-7a23208c5cf8, priority: Not in source})-[:AFFECTED_BY]->(:ExternalFactor {impact: Late spring & summer, name: Animal reproduction, id: 7b5a0ed5-d0b7-559a-85fe-a49cfaa71c93})"
"(:Service {ert: 21 days, name: Service requests resolution, sla: 85.0%, id: 7a39a312-79cb-5261-a418-d0c8b7481ea1, priority: Not in source})-[:OWNED_BY]->(:Department {name: MGT - Office of Homeless Solutions, id: 19f5d554-f727-54ea-8fae-ba4ac9b50881})"
"(:Service {ert: 21 days, name: Service requests resolution, sla: 85.0%, id: 7a39a312-79cb-5261-a418-d0c8b7481ea1, priority: Not in source})-[:AFFECTED_BY]->(:ExternalFactor {impact: Targets may vary based on the seasonality of the work., name: Seasonality, id: 2a3ab1bb-6c4a-565b-ae7a-3bfdf7c7c000})"
"(:Service {ert: 220 seconds (Call Answering), name: 311 Queue (General), sla: 60%, id: 420a0cf6-9c9d-5699-a2d8-6144bb04c23a, priority: Not in source})-[:OWNED_BY]->(:Department {name: 311 Customer Service Center, id: 92f70743-d7b4-57b4-9f75-d49b6c7c4821})"
"(:Service {ert: 220 seconds (Call Answering), name: 311 Queue (General), sla: 60%, id: 420a0cf6-9c9d-5699-a2d8-6144bb04c23a, priority: Not in source})-[:AFFECTED_BY]->(:ExternalFactor {impact: Late spring & summer, name: Mondays, Fridays, day after holidays, id: 6966aaa7-45d8-56d6-9a9f-f5bd2fba28df})"
"(:Service {ert: 90 seconds (Call Answering), name: Water related concerns (Water Queue), sla: 45%, id: b1d982fe-aac4-5568-8a8c-1a841a92f52b, priority: Not in source})-[:OWNED_BY]->(:Department {name: Dallas Water Utilities, id: c901b3ce-9e48-558d-9e27-1440da723bbe})"
"(:Service {ert: 90 seconds (Call Answering), name: Water related concerns (Water Queue), sla: 45%, id: b1d982fe-aac4-5568-8a8c-1a841a92f52b, priority: Not in source})-[:AFFECTED_BY]->(:ExternalFactor {impact: Late summer & early fall, name: Summer watering bills, id: f3a1b14f-aa33-5050-8e67-870e759d68cc})"

| Relationship Type          | Direction                           | Logic                                                                 |
| :------------------------- | :---------------------------------- | :-------------------------------------------------------------------- |
| **`OWNED_BY`**     | `(Service) -> (Department)`       | Maps which city department is accountable for a service.              |
| **`AFFECTED_BY`**  | `(Service) -> (ExternalFactor)`   | Identifies external variables that correlate with increased delays.   |
| **`MONITORED_IN`** | `(Service) -> (AuditTopic)`       | Links services to specific audit reports or oversight themes.         |
| **`RELEVANT_TO`**  | `(DocumentChunk) -> (AuditTopic)` | Bridges text chunks to high-level audit categories for multi-hop RAG. |

## 🛠️ Property Keys

- **`id`**: Unique identifier for the node.
- **`name`**: Human-readable name.
- **`content`**: Text body (for `DocumentChunk`).
- **`embedding`**: Vector representation (for semantic search).
- **`ert`**: Estimated Response Time (from city benchmarks).
- **`sla`**: Service Level Agreement thresholds (in hours).
- **`priority`**: Default priority level (1-5).
- **`impact` / `objective`**: Metadata for audit topics.
- **`hours`**: Historical average resolution timeframe.
- **`source` / `type`**: Origin or classification of the data.

---

### Implementation in Pipeline

The `inference/explainability_chain.py` uses this schema to perform **Cypher Queries** and **Vector Search** to augment ML predictions with high-level city operational context.
