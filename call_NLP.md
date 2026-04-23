# Dallas 311: Acoustic Intelligence Architecture (Call-to-Inference)
**Blueprint**: Real-Time Citizen Prediction Interface
**Target**: Citizens during 311 Service Request Ingestion

---

## 1. Overview: The "Live Forecast" Loop
Transmuting citizen audio into real-time operational predictions. This system provides an immediate **"Success Level"** and **"Estimated Resolution Probability"** directly to the caller based on their verbal description.

## 2. Pipeline Stages (Acoustic -> Semantic -> Predictive)

### Stage A: Acoustic Harvesting (STT)
- **Engine**: OpenAI Whisper or Google Speech-to-Text.
- **Input**: Raw 311 Call Audio stream.
- **Output**: Raw Text Transcript (e.g., *"I'm calling from District 7 to report a high weed violation on 123 Main St."*)

### Stage B: Semantic Extraction (NLP)
- **Task**: Named Entity Recognition (NER) & Classification.
- **Mapping Logic**:
    - **Entity**: "District 7" -> **Feature**: `City Council District` (ID: 7)
    - **Entity**: "High weed violation" -> **Feature**: `Service Request Type` (Code Concern - CCS)
    - **Entity**: "Urgent/Emergency" -> **Feature**: `Priority` (High)

### Stage C: The Predictive Handshake (ML Inference)
- **Trigger**: The NLP-extracted features are posted to the `/api/manual_infer` endpoint.
- **Model Context**: Evaluated against the **600,000-row Master Dataset**.
- **Output**: `prediction_class` (Fast/Slow) + `probability` (e.g., 0.88).

---

## 3. The "Citizen Clarity" Interface
*Purpose: Translating raw ML probability into human-centric feedback.*

| Probability Score | Citizen Message (AI Output) | Strategic Meaning |
| :--- | :--- | :--- |
| **0.85 - 1.00** | "Your request matches our 'Optimal Path' profile. Estimated resolution within 72 hours." | High resolution success probability. |
| **0.50 - 0.84** | "System indicates standard processing. Expected completion conforms to standard city ERT." | Predictable, non-outlier case. |
| **< 0.50** | "Due to high volume in your area (CCS Bottleneck), this case may require additional oversight." | Flagged as a potential 'Slow-Close' outlier. |

---

## 4. Technical Integration (Dashboard Sync)
When a citizen initiates a call, the **Dashboard** will show a "Live Ingestion" status:
1.  **Agent Orchestration**: The `Explorer Agent` pulls the latest CCS bottleneck facts from Neo4j.
2.  **Strategic Advisor**: Cross-references the citizen's keywords with **Expert Wisdom Node: 'Electronic Citation Transition'**.
3.  **Real-Time Insight**: *"Citizen is reporting a CCS violation during the system transition—expect delay notification."*

---

## 5. Metadata for Future Development
- **NLP Library**: SpaCy / HuggingFace (Transformer-based NER)
- **STT Latency Target**: < 3.0 seconds for "Live Handshake"
- **Training Requirement**: Model must be trained on **600k rows** to capture dialect and slang variations in request types.
