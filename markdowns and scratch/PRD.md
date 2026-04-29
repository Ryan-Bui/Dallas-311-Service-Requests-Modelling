# Product Requirements Document (PRD)

## 1. Project Objective
The goal of this project is to build a machine learning pipeline that analyzes and predicts the resolution time (`days_to_close`) for 311 Service Requests. We will implement a decision tree-based model (e.g., Decision Tree Regressor, Random Forest) to understand which predictor variables significantly impact the time it takes for a service request to go from creation to closure.

## 2. Dataset Overview
- **Source**: 311 Service Requests dataset (October 1, 2020, to Present).
- **Sampling**: To manage computational load, a 10% stratified sample based on the department is utilized.

## 3. Target Variable
- **`days_to_close`**: A continuous numerical variable representing the total duration (converted to hours) between the `Created Date` and the `Closed Date` of a service request. 

## 4. Predictor Variables (Features)
Based on the preprocessing steps in the project, the following variables will be used as inputs for the decision tree model:
- **Temporal Features** (Extracted from `Created Date`):
  - `month`: Month of the year the request was created.
  - `day_of_week`: Day of the week the request was created.
  - `hour`: Hour of the day the request was created.
- **Categorical Features** (Label Encoded):
  - `Priority`: Priority level of the request.
  - `Method Received Description`: How the request was initiated (e.g., API, Phone).
  - `Department_grouped`: Grouped responsible department (rare departments are binned into 'Other').
- **Engineered Features**:
  - `City Council District`: Geographic district of the request.
  - `service_type_encoded`: Mean target encoded value of `days_to_close` grouped by `Service Request Type`.
  - `ERT_days`: Parsed numerical value (in days) extracted from the `ERT (Estimated Response Time)` string.

*(Note: Columns inducing data leakage such as `Update Date`, `Status`, `Outcome`, and `Closed Date` are explicitly removed prior to modeling).*

## 5. Methodology & Modeling
1. **Data Preprocessing**:
   - Parse dates and calculate the exact resolution time (`days_to_close`).
   - Remove irrelevant features, high cardinality/messy features (like `Lat_Long Location`), and leakage variables.
   - Extract datetime components for seasonal and operational insights.
   - Apply Label Encoding to categorical fields and Target Encoding to high-cardinality service types.
2. **Model Implementation**:
   - Train a Decision Tree Regressor (or ensemble like Random Forest/XGBoost) on the preprocessed predictor variables.
   - Evaluate the model's accuracy predicting the `days_to_close`.
   - Extract feature importances to determine which operational factors most heavily dictate service request resolution times. 
