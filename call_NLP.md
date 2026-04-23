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

| Probability Score     | Citizen Message (AI Output)                                                                     | Strategic Meaning                            |
| :-------------------- | :---------------------------------------------------------------------------------------------- | :------------------------------------------- |
| **0.85 - 1.00** | "Your request matches our 'Optimal Path' profile. Estimated resolution within 72 hours."        | High resolution success probability.         |
| **0.50 - 0.84** | "System indicates standard processing. Expected completion conforms to standard city ERT."      | Predictable, non-outlier case.               |
| **< 0.50**      | "Due to high volume in your area (CCS Bottleneck), this case may require additional oversight." | Flagged as a potential 'Slow-Close' outlier. |

---

## 4. Hierarchical Intelligence (The Sub-Model Dispatcher)

*Purpose: Achieving hyper-granularity by training specialized models for distinct departmental behaviors.*

### Stage D: The Dispatcher Logic

1. **Input**: NLP-extracted `Department` entity.
2. **Routing**: The system dynamically loads the **"Departmental Champion"** model (e.g., if Dept='DAS', use `model_DAS.joblib`).
3. **Specialization**: Each sub-model is trained on the laboratory-grade **600k-row subset** specific to that department, capturing unique resolution SOPs.

### The Location-Service-Department Triplet

Citizens receive a likelihood score derived from the intersection of:

- **Location (District/Zone)**: Accounting for geographical resource allocation.
- **Service Type (Specific Request)**: Accounting for task complexity (e.g., Illegal Dumping vs. Pothole).
- **Departmental Efficiency**: Accounting for current staffing and transition bottlenecks (e.g., Code Compliance's electronic citation lag).

## 5. The "Citizen Clarity" Interface (Enhanced)

*Purpose: Translating sub-model probabilities into localized strategic feedback.*

| Probability Score               | Citizen Message (AI Output)                                                                             | Strategic Meaning                       |
| :------------------------------ | :------------------------------------------------------------------------------------------------------ | :-------------------------------------- |
| **Best-in-Class (DAS)**   | "Your Animal Service request in District 4 has a 94% success profile based on local dashboard trends."  | Hyper-accurate departmental prediction. |
| **Strategic Delay (CCS)** | "Our specialized Code Model identifies a potential resolve-time lag due to current system transitions." | Grounded, high-fidelity explanation.    |
| **Baseline Success**      | "Your request conforms to the standard 0.866 city-wide performance baseline."                           | Fallback to Global Model intelligence.  |

---

## 6. Technical Integration (600k Scale-Up)

- **Agent Orchestration**: `Explorer Agent` now retrieves department-specific audit facts.
- **Model Storage**: `/models/champions/` directory houses the fleet of specialized `.joblib` files.
- **Reasoning**: `Strategic Advisor` cites the specific sub-model's feature importance in its final response.
