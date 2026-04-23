# Dallas 311 Intelligence: Expert Metric Manual
**Status**: Golden Source (Primary Grounding)
**Last Updated**: April 2026

## 1. Core Model Performance (XGBoost)
- **Primary Metric (ROC-AUC)**: 0.838 (Achieved in Second Cycle).
- **Interpretation**: A score of 0.838 indicates a "High-Fidelity" model. In the context of Dallas 311, this means we can reliably distinguish between "Routine Resolutions" and "At-Risk Escalations" with 84% accuracy.
- **Operational Reality**: Any case predicted as a "Slow Close" by this model should be flagged for supervisor review immediately.

## 2. Top Predictors (Operational Bottlenecks)
1. **Neighborhood (Council District)**: Geographical location is the #1 predictor of response time, suggesting staffing disparities or infrastructure complexity in specific districts.
2. **Service Type**: "Code Concern - CCS" and "Animal Related" service types consistently show the highest variance in Estimated Response Time (ERT).
3. **Temporal Factors**: Requests made on Friday afternoons or adjacent to city holidays show a 15% increase in resolution lag.

## 3. Knowledge Graph Strategy (Neo4j)
- **Native Vector Search**: We use a 768-dimension vector index (`chunk_embeddings`) to ground AI answers in audited city reports.
- **Expert Wisdom Nodes**: These nodes bypass standard RAG and act as "Immutable Business Rules" for the Strategic Advisor.
- **Strategic Goal**: Transition from "Reactive Reporting" to "Proactive Orchestration" by linking 311 requests directly to Council District budget allocations.

## 4. Expert Pro-Tips for City Officials
- **Refining ERT**: To improve the ROC-AUC score above 0.85, the city must integrate "Weather Impact" and "Crew Availability" data into the Neo4j Graph.
- **Bottleneck Resolution**: Reducing the resolution time for "Code Concern" cases requires automated cross-departmental "Handshake" protocols between CCS and Public Works.
