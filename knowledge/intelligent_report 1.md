# Dallas 311 Intelligence: Master Strategic Report
**Author**: Quoc Anh Nguyen **Role**: Principal City Data Scientist **Status**: DRAFT — ML-Grounded (Awaiting Human Expert Augmentation)

---

## 1. Operational "Why" (Contextualizing the Model)

*Purpose: Use this section to explain why the model behave certain way based on real departmental constraints.*

### Classification Target: Fast vs Slow (Threshold-Based)

The model classify each request as fast (1) or slow (0) depend on how many hours it take to close. The 72-hour mark is our primary threshold, but we also test 24h, 48h, and 96h to understand how the result shift:

| Threshold  Fast (%)  Slow (%) 
| 24h          ~16%      ~84% 
| 48h          ~24%      ~76% 
| 72h          ~30%      ~70% 
| 96h          ~35%      ~65% 

At 24h, only about 1 from 6 requests is fast — very strict and not realistic for most service types. At 96h, more than one third qualify. The 72h threshold is the main operational standard we build the model around.

---

### Service Type: Code Compliance (CCS)

- **Operational Constraint**: Most CCS requests need someone to go physically to the location and inspect before anything can happen. We don't have unlimited inspectors, so just scheduling alone can take several days. Some cases like illegal dumping or hazardous material must go to outside contractor — after that we are waiting on their schedule, not ours. A good example is illegal dumping report — the city log it, flag it, follow up, but cannot force contractor to come faster.
- **AI Grounding Note**: The model consistently predict these as slow, especially at 24h and 48h thresholds. This is expected. Only small percentage of CCS requests close within one day. The model is learning the real structural delay of this service type, not making mistake.

---

### Service Type: Street & Sidewalk Repair

- **Operational Constraint**: This type come up very often — someone report cracked sidewalk but it is not dangerous. So it go behind higher priority work like potholes or flooding. Even when the repair look simple, still need inspection, approval, and crew scheduling. Crews usually group jobs by location, so individual request must wait until there is nearby work to combine with. A resident reporting broken curb on their street may wait weeks — not because nobody care, but because crew won't drive across the city for one small job.
- **AI Grounding Note**: The model predict these as slow especially at strict thresholds. This match how the city actually prioritize work. The model is not finding a problem — it is correctly describing the system behavior.

---

### Service Type: Abandoned Vehicle

- **Operational Constraint**: Abandoned vehicle cases have a mandatory legal hold period before the city can do anything. Officer can go mark the vehicle same day, but the case cannot close until that legal window pass. After that, tow company must be dispatched and their availability vary by time of day and location. So even a fast response from the officer still result in a slow close time by definition.
- **AI Grounding Note**: The model will almost never predict fast for abandoned vehicle cases — and this is correct. The legal hold alone make fast resolution structurally impossible. This is a policy constraint, not an operational failure. [PLACEHOLDER: Confirm if abandoned vehicle is a service type category in your dataset and verify it appear in the `Service Request Type` feature after top-15 filtering.]

---

### Service Type: Graffiti Removal

- **Operational Constraint**: Graffiti on city property and graffiti on private property follow completely different process. For private property, city must notify the owner first and wait for response before doing anything. This notification step alone add significant time. Also, removal crew need to match paint color and surface type — not every crew carry all materials for every job.
- **AI Grounding Note**: The model is likely learning the difference between these two case types even without explicit property-type label, because the resolution time pattern is very different between them. Cases that take very long are probably private property cases where owner notification caused delay. [PLACEHOLDER: Confirm if property type — public vs private — is distinguishable from any field in your dataset.]

---

### Service Type: Animal Services

- **Operational Constraint**: Animal services cases vary a lot in urgency. A dog bite report and a stray cat complaint can enter the same general queue, but they are completely different situations. The department has limited animal control officers and vehicles, and response depends heavily on geographic coverage and time of day.
- **AI Grounding Note**: When the model predict fast for animal services, it is likely capturing the emergency-tier cases like bite reports or injured animals. When it predict slow, it is capturing the low-priority stray or nuisance cases. Both predictions can be correct at same time for different reasons. Analysts should not evaluate this category as one single group. [PLACEHOLDER: Confirm if animal services is a separate department group in your `Department_grouped` feature or if it fall into "Other" category.]

---

## 2. External Factors (The "Invisible" Variables)

*Purpose: Document real-world conditions that are not directly in the dataset but clearly affect the results.*

### City Events and Seasonal Spikes

- **State Fair and Cotton Bowl game days**: Every October, Fair Park area see big spike in specific request types — noise complaints, trash overflow, parking problems, event-related congestion. This happen same time every year and it show up clearly in the data as a seasonal pattern. Graffiti reports near Fair Park also tend to increase around these events.
- **Summer heat**: July and August bring more requests across many categories — animal services, abandoned vehicles left in the sun, infrastructure issues — while at same time field crews work in very difficult conditions. More demand and lower throughput happen at same time.
- **AI Grounding Note**: The model capture these patterns indirectly through the `month` feature. A slow prediction for summer or October-submitted request should be understood in this context, not only as capacity failure.

---

###  The Weekend Trap

- There is a timing gap that many people don't think about — the period between when a request is submitted and when it actually reach a human who can act on it. This gap is not the same for every request. A graffiti report that come in at 11 PM on Friday will sit for almost 60 hours before a crew is even available to look at it Monday morning. An abandoned vehicle reported on Saturday morning start its mandatory legal hold period over the weekend — so by the time Monday arrive, it still need more days before tow can happen. The submission timing and the operational calendar interact in a way that multiply delay, not just add to it.
- This is different from just saying "weekends are slower." The real issue is that certain request types have multi-step processes, and if the first step get pushed by even one day, every step after it also shift. A Friday submission for a complex case is not one day slower — it can be three or four days slower because of how the process chain work.
- **AI Grounding Note**: Our model include both `day_of_week` and `hour` as features to help capture this compounding effect. The insight here is not just that timing matter — it is that timing interact differently depending on the service type. A Friday graffiti report and a Friday pothole report are not delayed by the same amount, even though both are submitted on the same day. [PLACEHOLDER: Run a cross-tabulation of `day_of_week` x `Service Request Type` against median `days_to_close` to see if the weekend delay effect is stronger for some service types than others.]

---

### Language and Communication Barriers

- Dallas has large Spanish-speaking population and the 311 system offer Spanish-language intake option. But when a request come in through Spanish-language channel, it must go through handoff from Spanish-speaking intake staff to English-speaking field team. Sometimes in this handoff, details get lost or the request description become incomplete — and this cause delay because field crew arrive without full understanding of the issue. A graffiti report with wrong address or a code compliance case with missing detail can sit unresolved much longer just because of this communication gap.
- This is not about the resident receiving lower priority. It is about a process gap that add time between intake and actual work starting.
- **AI Grounding Note**: Language of submission is not currently a labeled feature in our model. But the `Method Received Description` field in the dataset may carry indirect signal about intake channel. [PLACEHOLDER: Check if Spanish-language submissions are distinguishable from any field in the dataset. If yes, this is worth analyzing as a subgroup and possibly adding as a feature to improve model fairness.]

---

### Repeat Request Pattern

- Some addresses submit the same type of request multiple times in short period. This usually mean the first request was closed without actually fixing the problem, or the fix was only temporary. A cracked sidewalk that get patched but break again, or graffiti that get removed but come back on same wall — these create repeat requests that are often harder and slower to resolve than the original.
- **AI Grounding Note**: The model does not currently have an explicit repeat request flag as a feature. But it may be picking up some of this pattern indirectly through address or service type patterns. [PLACEHOLDER: Check if repeat request within 30-day window can be engineered from the address and service type fields in your dataset. If yes, this could be a strong additional feature.]

---

### Department-Level Behavior

- We group departments with fewer than 1,000 requests into "Other" category to reduce noise. Each remaining department show different average resolution time because they have different processes, staff size, and type of work. Code Compliance is structurally slower than Animal Services emergency cases for example — not because of poor performance, but because the process itself is more complex and involve more steps.
- **AI Grounding Note**: `Department_grouped` is one of the stronger features in the model. A slow prediction for certain department is often just reflecting that department baseline reality, not a failure signal. [PLACEHOLDER: Run `dept_compare` output from notebook and document the average and median resolution hours per department group here.]

---

## 3. Golden Human Wisdom (The Primary Truth)

*Purpose: Add definitive expert judgments here. The AI will prioritize these over standard RAG.*

- **On Model Selection**: We test three models — Logistic Regression, Random Forest, and XGBoost. XGBoost perform best overall based on ROC-AUC. Random Forest is second and also very useful because it give us feature importance. Logistic Regression serve as baseline. [PLACEHOLDER: Fill in actual accuracy and ROC-AUC scores from notebook output for all three models.]
- **On Feature Importance**: The top features from Random Forest include `Priority`, `ERT_days`, `day_of_week`, `month`, and `hour`. This make operational sense — priority and estimated response time are set at intake, so they carry strong predictive signal. Time features capture scheduling and staffing patterns. [PLACEHOLDER: Fill in actual top 10 features and importance scores from notebook output.]
- **On Slow Predictions**: A slow prediction does not mean something went wrong. An abandoned vehicle case is slow because the law require it. A graffiti case on private property is slow because the owner notification process require it. The model is learning the system structure, not judging performance.

---

## 4. Standard Operating Procedures (SOP Mapping)

*Purpose: Link model thresholds to actual city policy and operational expectations.*

- **Primary Threshold — 72h**: This is the citywide standard resolution target and the main boundary our model is built around. Cases predicted slow at this threshold are the ones most likely to need operational attention.
- **Legal Hold Cases**: For service types with mandatory waiting period before resolution — like abandoned vehicles — the model threshold should be interpreted carefully. Comparing these cases against 24h or 48h is not meaningful. The minimum realistic threshold for these cases is 72h or higher. [PLACEHOLDER: Confirm full list of service types in your dataset that have mandatory hold or notification periods before work can begin.]
- **Priority Feature Alignment**: Our model use `Priority` as a feature because the city already assign priority level at intake. Emergency cases are expected to close much faster than standard or low-priority ones. The model learn this relationship directly from data.
- **ERT Alignment**: The `ERT_days` feature come from the Estimated Response Time field. When ERT is long, the model generally predict slow — which is correct because long ERT mean the department already expect the case to take more time. Graffiti on private property or complex code compliance cases tend to have longer ERT by design.
- **Department SOP Mapping**: [PLACEHOLDER: For each major department group in your dataset, confirm the official resolution SOP and whether it align with what the model is predicting. For example, Code Compliance may have different expected resolution window than Animal Services or Street Repair.]
- **Escalation Path**: [PLACEHOLDER: Is there an automatic escalation trigger when request go past 48h or 72h without update? If yes, document the escalation path so model slow-class predictions can connect to that workflow.]

---

## 5. Metadata for Neo4j (Ingestion Instructions)

*Note: Do not delete this section. The Archivist Agent uses this for indexing.*

- **Ingestion_Priority**: HIGH
- **Reasoning_Role**: PRIMARY_GROUNDING
- **Semantic_Trust_Score**: 0.95