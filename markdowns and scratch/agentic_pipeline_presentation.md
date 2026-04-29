# Dallas 311 Service Request ML Platform: Agentic Pipeline & Inference Engine

*Source material for presentation slide generation.*

---

## 1. Executive Summary

The **Dallas 311 Intelligence Platform** is a state-of-the-art, fully autonomous Machine Learning workflow designed to predict the resolution times of city service requests (e.g., potholes, stray dogs, high weeds). 

It abandons the traditional, static "Jupyter Notebook" paradigm in favor of an **Agentic Pipeline architecture**. Specialized autonomous agents handle distinct phases of the machine learning lifecycle—from data scrubbing and modeling to real-time external research and truth reconciliation via a Neo4j Knowledge Graph. 

On the production side, the system utilizes a **Multi-Threshold Inference Engine** and an **Acoustic Extractor**, allowing citizens to speak their complaints into a microphone while the backend simultaneously runs 12 different ML models to forecast closure probabilities across a 4-tier timeline.

---

## 2. The Agentic Training Pipeline (Autonomous ML Lifecycle)

The core training pipeline is entirely automated and divided into sequential agents that inherit from a standard `BaseAgent` class. They pass data forward and feature built-in garbage collection for memory optimization.

### Agent 1: DataPrepAgent
- **Role:** Handles data ingestion and initial sanitization.
- **Actions:** Loads raw historical service request data, drops completely empty columns, standardizes date formats (calculating `days_to_close` dynamically), and scrubs erroneous zip codes.
- **Output:** A clean, uniformly structured pandas DataFrame ready for advanced statistical manipulation.

### Agent 2: TransformationAgent
- **Role:** Manages class imbalances and feature scaling.
- **Actions:** Detects severe class imbalances in the dataset and applies techniques like SMOTE (Synthetic Minority Over-sampling Technique) or strategic down-sampling to ensure minority service requests aren't overshadowed by high-volume complaints.
- **Output:** A balanced dataset that prevents model bias.

### Agent 3: DiagnosticsAgent
- **Role:** Acts as the statistical gatekeeper.
- **Actions:** Performs rigorous statistical checks on the data, such as Variance Inflation Factor (VIF) analysis to detect severe multicollinearity among features. 
- **Feedback Loop:** If diagnostic thresholds are violated (e.g., VIF > 10), it can trigger a feedback loop to the TransformationAgent to drop redundant features.

### Agent 4: ModelSelectionAgent
- **Role:** The core predictive engine builder.
- **Actions:**
  - Encodes categorical features (One-Hot / Label Encoding) and standardizes numerical columns.
  - Splits data into strict Training and Testing matrices (preventing data leakage).
  - Trains three distinct baseline models: **Logistic Regression, Random Forest, and XGBoost**.
  - Evaluates models using ROC-AUC scores and selects the top-performing candidate.
- **Output:** Saves the "Best Model" artifacts (`.joblib`/`.json`), feature arrays, and scalers to disk. Aggressively clears memory (Garbage Collection) immediately after saving to prevent cloud memory limits.

### Agent 5: RegularizationAgent
- **Role:** Model interpretability and overfitting prevention.
- **Actions:** Fits Ridge, LASSO (L1), and ElasticNet models across varying alpha parameters.
- **Output:** Extracts a penalized coefficient table to identify the strongest positive and negative drivers of service resolution times, regardless of the underlying ML algorithm.

---

## 3. Intelligent Enrichment & Graph Reasoning (Phases 2 & 3)

Once the core statistical models are trained, the pipeline employs advanced LLMs and Graph databases to enrich the models with real-world strategic context.

### ExplorerAgent & ArchivistAgent (Knowledge Graph)
- **Role:** Real-time strategic enrichment.
- **Actions:** The **ExplorerAgent** searches the web (or external audit reports) for systemic bottlenecks related to specific city services (e.g., "Why are Dallas Animal Services responses delayed?").
- **Graph Storage:** The **ArchivistAgent** maps these findings into a **Neo4j Knowledge Graph**, linking Nodes (Service Types, Departments, Bottlenecks) with relationships (`IMPACTS`, `MITIGATES`), embedding the textual data using Gemini 1.5.

### ReinforcementJudge (Truth Reconciliation)
- **Role:** AI Hallucination mitigation.
- **Actions:** Uses Vector RAG (Retrieval-Augmented Generation) against the Neo4j Graph. It retrieves Ground Truth audit reports and forces a primary LLM (Groq / Llama 3) to vet its own assumptions against established facts, outputting a Confidence/Trust Score before finalizing the model's metadata.

---

## 4. The Multi-Threshold Inference Engine

The production dashboard exposes the trained models through a highly resilient, multi-tiered inference engine designed to handle real-time citizen interactions.

### Stage A: Acoustic Extraction (Speech-to-Text)
- Users record simulated 311 calls directly through their microphone.
- The audio is transmitted to the backend where **Groq's Whisper API** transcodes the audio to text in milliseconds.
- An extraction LLM parses the unstructured transcript and outputs a strict JSON payload mapping to the ML model's required schema (Service Type, Department, District, Priority, Method).

### Stage B: 12-Model Ensemble Timeline
- The legacy approach of predicting a binary "Fast vs Slow" close is abandoned.
- The engine dynamically loads **12 distinct models** (XGBoost, Random Forest, Logistic Regression tuned specifically for **24h, 48h, 72h, and 96h** thresholds) alongside their respective preprocessors.
- It runs the extracted citizen data through all 12 models simultaneously.
- **Output:** An aggregated "Ensemble Average" probability for each time block, rendered in the UI as a vibrant, 4-tier horizontal progression timeline.

### Stage C: Citizen Clarity AI (Strategic Explainability)
- The raw mathematical probabilities across the 4 thresholds are passed into the `explainability_chain.py`.
- **LLM Reasoning:** A powerful language model analyzes the multi-threshold progression (e.g., "Why does the probability spike at 72 hours but remain low at 24 hours?") and cross-references it with the Neo4j Knowledge Graph.
- **Output:** Generates a human-readable **Expert Analysis** and strategic advisory block, giving citizens realistic expectations and city officials actionable root-cause insights.
