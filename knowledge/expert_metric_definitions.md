# DALLAS 311: EXPERT METRIC INTERPRETATIONS
*This is the "Golden Source" for AI reasoning. Add your team's human insights here.*

## Metric: Best ROC-AUC
**Expert Insight**: An ROC-AUC above 0.80 generally indicates that our data correctly identifies the "Slow Close" cases before they become a public image crisis. 
**Operational Reality**: In Dallas, a high score here means the Sanitation department is properly logging their route delays, allowing for predictable rescheduling.

## Metric: Features Selected
**Expert Insight**: We prioritize features like 'Council District' and 'Method Received' because they are the strongest proxies for departmental workload complexity.
**Operational Reality**: If 'Council District' is a top feature, it usually points to localized infrastructure imbalances that need budget reallocation, not just faster trucks.

## Metric: Accuracy
**Expert Insight**: Total accuracy is less important than "Precision" for Code Compliance. 
**Operational Reality**: We would rather be 70% accurate but have zero "False Negatives" (missed hazardous violations) than be 99% accurate on routine trash pickups.

## Metric: Precision (XGBoost)
**Expert Insight**: Precision is the "Trust Factor" for our field agents.
**Operational Reality**: High precision means when the app tells an inspector a case will be delayed, they can trust that prediction and prioritize their travel accordingly.
