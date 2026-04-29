# Dallas 311 Intelligence: Master Strategic Report
**Author**: Ryan Bui
**Role**: Principal City Data Scientist
**Status**: DRAFT — EDA-Grounded (Awaiting Human Expert Augmentation)

---

## 1. Operational "Why" (Contextualizing the Model)

*Purpose: Use this section to explain 'Slow-Close' or multi-class prediction results based on real-world departmental constraints.*

### Regression Target: Hours to Close
The raw `hours_to_close` distribution is extremely right-skewed (skew = 12.47), driven by a small number of requests that take thousands of hours to resolve. The log-transformed distribution (skew = 0.03) is near-normal, which means **the regression model should be trained on `log(1 + hours_to_close)` and its output back-transformed for reporting**. Raw-hour predictions without this transformation will systematically underperform on outlier cases.

### Classification Target: 3-Class Slow-Close Tiers
The proposed classification scheme maps directly to observed natural breakpoints in the CDF:

| Class Label | Threshold | % of Requests |
|---|---|---|
| **Fast** (≤ 1 day) | 24h | ~40% |
| **Standard** (1 day – 1 week) | 24h – 168h | ~36% |
| **Slow** (> 1 week) | > 168h | ~24% |

At the 72h threshold (the current operational standard), 59.4% of requests close as Fast and 40.6% as Slow — the minority class fraction of ~0.41 exceeds the DiagnosticsAgent minimum of 0.25, making this a viable binary boundary if a two-class model is also needed.

### Service Type: MCC (Median 169h)
- **Operational Constraint**: MCC requests have the highest median resolution time across all priority tiers at 169 hours — more than double the 72h operational standard. Any model prediction of delay for MCC-type requests should be interpreted as structurally expected, not as a performance failure.
- **AI Grounding Note**: If the model predicts Slow-class for MCC requests, this reflects the baseline operational reality of that service type and not a data anomaly.

### Service Type: Proactive Intake (Median 197h)
- **Operational Constraint**: Proactively submitted requests take a median 197 hours to close — the slowest of all intake methods — despite representing 265,259 requests. This is likely because proactive requests are non-emergency by definition and are triaged accordingly.
- **AI Grounding Note**: Intake method = Proactive is a strong positive predictor of Slow-class. This is expected and operationally appropriate.

### Service Type: Dispatch (Median 5h, n=619,321)
- **Operational Constraint**: Dispatch is by far the largest single priority tier (619,321 requests) and resolves with a median of only 5 hours — well below the 72h threshold. This tier will dominate any model trained without stratification.
- **AI Grounding Note**: High Fast-class recall for Dispatch is expected and not indicative of model overfit. Analysts should evaluate model performance on non-Dispatch tiers separately.

---

## 2. External Factors (The "Invisible" Variables)

*Purpose: Document factors observed in the data that are NOT labeled in the CSV but influence 311 performance.*

### Temporal Seasonality
- **June Peak**: June shows the highest median resolution time across all months (approximately 50h), coinciding with elevated request volume. Summer heat and outdoor activity likely drive increased service demand across categories including animal services and street/infrastructure issues.
- **February Trough**: February shows the lowest median resolution time of any month (approximately 32h), likely reflecting reduced request volume and faster staff throughput.
- **AI Grounding Note**: A model predicting delay for June-submitted requests should be interpreted in the context of seasonal demand increases, not solely as a capacity failure.

### Weekly Timing Effects
- **Friday Submissions**: Requests submitted on Fridays show the highest median resolution time of the week (~75h), likely because weekend staffing reduces throughput and the clock runs through Saturday and Sunday before work resumes.
- **Monday Submissions**: Monday-submitted requests close fastest (~37h), reflecting full-week staff availability following submission.
- **AI Grounding Note**: Day-of-week is a legitimate and meaningful feature for the model. A Friday submission being predicted Slow is structurally reasonable.

### Staff Submission Timing
- Requests submitted around **6–7 AM** carry the highest median resolution lag of any hour (~55h), despite low submission volume at that hour. Early-morning requests may miss the morning triage window and queue behind the business-hours surge.
- Staffing limitations can also lead to request sent during atypical business hours such as 12-6am can take longer to process this can serve as a motivator for having an Agent take care of the requests.
- The highest-volume submission hours (10 AM–3 PM) have moderate and consistent resolution times (~43–48h).

### ERT Compliance Landscape
- **44.3% of requests close in under 50% of their ERT** — ahead of schedule. This is a strong positive signal that many service types have conservatively set ERTs.
- **26.5% of requests exceed 200% of their ERT** — these are the operationally critical "very late" cases the Slow-Close model is primarily designed to catch.
- Long-ERT service types (110-day, 120-day ERT assignments) are the primary contributors to the very-late cohort and should be treated as a distinct analytical segment.

### Year-over-Year Anomaly (2025)
- Median resolution time spiked sharply in 2025 relative to 2022–2024 before dropping again in 2026 (partial year data). The cause of this spike is not captured in the dataset and requires human expert annotation (see Section 3).

---

## 3. Golden Human Wisdom (The Primary Truth)

*Purpose: Add your definitive expert judgments here. The AI will prioritize these over standard RAG.*

- **Metric Interpretation — Regression**: A Mean Absolute Error (MAE) of under 24 hours on back-transformed predictions should be considered a strong result given the structural skew of the distribution. Outlier cases (requests > 1,000 hours) should be evaluated separately and not penalize the model's primary use case.
- **Metric Interpretation — 3-Class Classification**: Given the class distribution (~40% Fast / ~36% Standard / ~24% Slow), weighted F1-score is the appropriate primary evaluation metric. Recall on the Slow class should be treated as the most operationally important metric — missing a slow case is costlier than a false alarm.
- **Threshold Selection**: The 72h binary threshold is operationally meaningful (current city standard) and produces an acceptable minority class fraction (0.41). The 48h threshold produces the most statistically balanced binary split (minority fraction ~0.48) but may not correspond to a meaningful city policy boundary. 48 hours is a critical operational threshold in Dallas 311 Standard Operating Procedures (SOPs). While "SLA" (Service Level Agreement) targets vary by department, the 48-hour mark serves as the definitive boundary between Standard and Urgent response tiers.
- **2025 Resolution Spike**: Current staffing levels have not supported the city's performance goals. By early 2025, the 311 system faced significant vacancies, which meant that even if a call was answered, the dispatch to field departments (like Code Compliance or Public Works) was delayed. The city reported that it lacked the headcount required to meet decades-old response standards.
- **Outcome Label Reliability**: The data is most likely manually entered but formatted automatically and then published to the Open Data portal automatically as well.

---

## 4. Standard Operating Procedures (SOP Mapping)

*Purpose: Link predictive model tiers to actual city policy.*

- **Operational Standard — 72h**: The 72h threshold is the citywide standard resolution target as evidenced by its use as the benchmark line across all EDA visualizations. This is the primary SOP boundary the model should enforce.
- **Priority Tier SOP Alignment**:
  - Dispatch (median 5h): Likely governed by an emergency or same-day response SOP. [PLACEHOLDER: Confirm SOP reference code.]
  - CMO (median 13h): [PLACEHOLDER: Confirm SOP reference code and intended resolution window.]
  - High (median 24h): Aligns with a standard 1-business-day resolution expectation.
  - Priority / Standard / Emergency (median 68–73h): All cluster at the 72h standard, indicating these tiers are directly governed by the 72h SOP.
  - MCC (median 169h): Exceeds the 72h standard by design. [PLACEHOLDER: Confirm whether MCC has its own SOP with a longer resolution window, e.g., 7 or 10 business days.]
- **ERT Overrun Escalation**: [PLACEHOLDER: Is there an automatic escalation trigger when a request exceeds 200% of its ERT? If so, document the escalation path and responsible department so the model's Slow-class predictions can be linked to this workflow.]
- **Proactive Request Handling**: [PLACEHOLDER: Do proactively submitted requests follow a different triage SOP than phone/API submissions? Their 197h median suggests they are deprioritized. Document whether this is by policy or by default.]

---

## 5. Metadata for Neo4j (Ingestion Instructions)

*Note: Do not delete this section. The Archivist Agent uses this for indexing.*

- **Ingestion_Priority**: HIGH
- **Reasoning_Role**: PRIMARY_GROUNDING
- **Semantic_Trust_Score**: 0.95