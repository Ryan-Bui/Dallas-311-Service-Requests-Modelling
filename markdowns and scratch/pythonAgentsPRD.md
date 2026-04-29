# Product Requirements Document (PRD): Python-Based Agentic ML Pipeline

## 1. Project Overview

This project aims to build an agentic orchestration pipeline for end-to-end machine learning processes. Utilizing Python allows for seamless integration of the user's existing `.ipynb` Jupyter Notebook code snippets, vastly accelerating the development process. The pipeline will comprehensively handle **data import, validation, merger, training, and inference**. It features a central AI orchestrator to manage Specialized Agents continuously refining statistical models. Furthermore, a dedicated Inference UI with a natively integrated Python LLM wrapper will provide human-readable explainability for model data components and predictions.

## 2. Architecture & Orchestration

The system relies on an **AI Orchestrator Manager** that coordinates independent **Specialized Python Agents**. Because the ecosystem is unified in Python, process communication can be handled via native asynchronous Python libraries or agentic frameworks (e.g., LangChain, AutoGen) rather than complex IPC.

### 2.1. AI Orchestrator Manager

- **Task Planner**: The central control node that initiates the pipeline (reading e.g., 311_Service_Requests). It delegates specific data and statistical operations to the downstream agents and aggregates the final validated results.
- **Critic / Verification Agent**: Evaluates the model constraints, metrics (e.g., test accuracy, residual behavior), and sends an "Approve" or "Refine" signal back to the Task Planner to maintain high-quality models.

### 2.2. Specialized Python Agents (The Execution Nodes)

These agents will be built by lifting and refactoring existing code snippets from the user's `.ipynb` files into modular, object-oriented Python classes.

1. **Data Prep Agent**: Utilizes `pandas` to merge datasets, handle missing values, and clean raw data. It delivers sanitized DataFrame objects.
2. **Transformation Agent**: Applies `numpy` or `scipy` operations (e.g., log transformations, Box-Cox) to normalize features and fix distribution shape issues.
3. **Diagnostics Agent**: Uses `statsmodels` or `scipy.stats` to run statistical diagnostics.
   - *Feedback Loop*: Detects heteroscedasticity or non-normality and signals the Transformation Agent to "Request Re-transformation".
   - If tests pass, it greenlights the data for the next phase.
4. **Model Selection Agent**: Iterates through predictor combinations optimizing for BIC/AIC using `statsmodels` or custom `scikit-learn` pipelines.
5. **Regularization Agent**: Applies `scikit-learn` algorithms (Ridge, LASSO, ElasticNet) to treat multicollinearity and overfitting. Outputs the **Final Validated Model Report**.

## 3. Jupyter Notebook (.ipynb) Integration Strategy

To rapidly accelerate the build, the pipeline will utilize existing code:

- **Extraction**: Core logic cells from `.ipynb` notebooks (like data cleaning routines or specific LASSO configurations) will be extracted into standardized python modules (`.py`).
- **Agent Wrapping**: Extracted functions will be wrapped inside the agent class structures so the Task Planner can call them programmatically.

## 4. Explainable Inference Window & UI

Python's ecosystem allows for rapid UI development suited specifically for data science and AI applications.

### 4.1. Inference UI Frontend

- We will leverage frameworks like **Streamlit** or **Gradio** to instantly deploy an interactive Inference Window.
- Users can input new parameters dynamically, triggering the backend Python pipeline to run an inference prediction instantly.

### 4.2. Native LLM Wrapper Integration

- **Direct Library Access**: The UI will natively integrate with LLM APIs (e.g., using `openai` or `langchain` Python packages).
- **Component Explanation**: The LLM will ingest the model's coefficients, test results, and raw prediction metrics, converting them into natural language. It will explain which data components (e.g., 'driving_accuracy', 'putts_per_round') were the key drivers for the specific inference result, offering complete transparency.

## 5. Technical Next Steps

1. **Repository Setup**: Initialize the Python environment (`requirements.txt` or `conda` env) with `pandas`, `scikit-learn`, `statsmodels`, and LLM dependencies.
2. **Snippet Migration**: Begin moving the `.ipynb` code blocks into specialized Python scripts (e.g., `data_prep_agent.py`, `transformation_agent.py`).
3. **Orchestrator Logic**: Draft the Task Planner script to connect the agents sequentially and establish the feedback loops.
4. **UI Prototyping**: Stand up a basic Streamlit app to serve as the standalone Inference Window.
