# Intelligent Pipeline Phases: Agentic 311 Forecasting

This document outlines the transition from a static, linear ML pipeline to an **Agentic Intelligent Pipeline** powered by LangChain and Groq reasoning.

## Phase 1: Contextual Data Assessment (The "Strategic Lead")
Instead of hardcoded preprocessing, the pipeline begins with a reasoning step.
- **Action**: A sample of raw data and the schema is passed to the `ExplainabilityChain`.
- **Intelligent Direction**: The AI identifies data quality issues (e.g., "The 'Location' column has 40% noise") and outputs a **Dynamic Transformation Plan (JSON)**.
- **Benefit**: Adapts cleaning strategies (imputation, encoding) based on the specific dataset version without manual code updates.

## Phase 2: Dynamic Preprocessing Execution
The Orchestrator reads the AI's "Transformation Plan" and routes data to specialized agents.
- **Agentic Routing**: 
  - If Plan says `high_cardinality_detected`: Route to `TargetEncoderAgent`.
  - If Plan says `time_series_gap`: Route to `TemporalInterpolationAgent`.
- **Implementation**: Utilize LangChain `RunnableBranch` to switch between processing paths based on the AI's diagnostic.

## Phase 3: Intelligent Refinement Loop (Self-Correcting ML)
The pipeline no longer stops after the first training run if results are poor.
- **Action**: `ModelSelectionAgent` outputs results to the `DiagnosticsAgent`.
- **Refinement Trigger**: If `ROC-AUC < 0.75` or `Recall < 0.60`, the AI triggers a **Refinement Cycle**:
  1. **Feature Synthesis**: AI suggests new interaction terms (e.g., `Department` x `Priority`).
  2. **Data Bias Correction**: AI identifies if the model is underperforming on specific Zip Codes and requests targeted oversampling.
  3. **Auto-Retrain**: The pipeline loops back to Phase 2 with the refined features.

## Phase 4: Domain Synthesis & Knowledge Grounding (Neo4j Integration)
The final model isn't just a set of weights; it is grounded in city policy and operational relationships via a native **Neo4j Knowledge Graph**.

- **Action**: All statistical results and predictions are cross-referenced with the graph schema.
- **Node Intelligence**:
  - `Service` nodes linked to `Department` via `OWNED_BY`.
  - `AuditTopic` and `DocumentChunk` nodes (vectorized) containing RAG context.
  - `ExternalFactor` nodes showing what impacts specific services (`AFFECTED_BY`).
- **Transformation**: Raw feature importance (e.g., "ERT_days: 0.13") is joined with graph-traversed insights.
  - *Example*: "Model identifies `ERT_days` as critical in `Street Repair`. Neo4j shows this `Service` is `AFFECTED_BY` 'Weather Events' and documented in '2024 Audit Topic: Resource Allocation'."
- **Benefit**: Provides **High-Fidelity Explainability** that goes beyond numbers and into "City Logic."


---

> [!TIP]
> **Implementation Note**: The current `flask_app.py` is ready for Phase 4. To initiate Phase 3, we simply need to wrap the `_run_pipeline` call in a conditional loop that checks the `reg_result` status before marking the state as `done`.
