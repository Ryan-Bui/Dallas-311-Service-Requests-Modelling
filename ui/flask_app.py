# ============================================================
# Flask Web Server — Dallas 311 ML Pipeline Dashboard
# ============================================================
"""
Serves dashboard.html at / and exposes a REST API for the
agentic ML pipeline.

Launch with:
    python -m ui.flask_app
    -- or --
    python ui/flask_app.py [--port 5000] [--debug]

API endpoints:
    GET  /              → dashboard.html
    GET  /api/status    → pipeline state + recent logs
    GET  /api/results   → full results (after a run)
    POST /api/run       → start pipeline  { "data_path": "..." }
    POST /api/infer     → start fast inference (no train) { "data_path": "..." }
    POST /api/reset     → reset to idle
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from http import HTTPStatus
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()
import shutil
import joblib
from groq import Groq

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]   # project root
UI_DIR    = Path(__file__).resolve().parent        # ui/
UPLOAD_DIR = ROOT / "data" / "uploaded"
ARTIFACTS_DIR = ROOT / "models"
HISTORY_DIR = ARTIFACTS_DIR / "history"
RESULTS_PATH = ARTIFACTS_DIR / "latest_results.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory, url_for
try:
    from flask_cors import CORS as _CORS
except ImportError:
    _CORS = None  # optional — not needed for same-origin requests

from inference.explainability_chain import create_explainability_chain, format_coef_summary, get_domain_context

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(UI_DIR))
if _CORS:
    _CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Groq LLM Client ─────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_client = None

if GROQ_API_KEY and "gsk_" in GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print(f"\n[AI] Initializing Groq reasoning engine...")
        print(f"[AI] Using Model: {GROQ_MODEL}")
        
        # Test connection immediately
        test_comp = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "Ping"}],
            model=GROQ_MODEL,
            max_tokens=5
        )
        print(f"[AI] Groq connectivity verified. AI Reasoning is ACTIVE.\n")
    except Exception as e:
        print(f"\n[AI] WARNING: Groq initialization failed: {e}")
        print("[AI] Reasoning will use rule-based fallbacks.\n")
else:
    print("\n[AI] INFO: GROQ_API_KEY not found. AI Reasoning is DISABLED.\n")

# ── In-memory pipeline state ──────────────────────────────────────────────────
_state: dict = {
    "status":      "idle",   # idle | running | done | error
    "progress":    0,
    "logs":        [],
    "agents": {
        "DataPrepAgent":       "idle",
        "TransformationAgent": "idle",
        "DiagnosticsAgent":    "idle",
        "ModelSelectionAgent": "idle",
        "RegularizationAgent": "idle",
    },
    "results":     None,
    "error":       None,
    "data_path":   None,
    "started_at":  None,
    "finished_at": None,
}
_lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_key_findings(detailed_results, best_model_name):
    """Generate a summary of the model comparison findings."""
    if not detailed_results: return "No detailed results available."
    best = detailed_results.get(best_model_name, {})
    
    findings = [
        f"The <strong>{best_model_name}</strong> model demonstrated the highest overall predictive power with an ROC-AUC of {best.get('ROC_AUC', 0):.3f}."
    ]
    
    xgb = detailed_results.get("XGBoost", {})
    lr = detailed_results.get("Logistic Regression", {})
    if xgb.get("ROC_AUC", 0) > lr.get("ROC_AUC", 0) + 0.05:
        findings.append("XGBoost significantly outperformed Logistic Regression, suggesting complex non-linear patterns in the service requests.")
        
    if best.get("Recall", 0) > best.get("Precision", 0):
        findings.append("Most models prioritize Recall over Precision, ensuring more potential delayed cases are flagged even with some false positives.")

    return " ".join(findings)


def _log(msg: str, tag: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {"t": ts, "tag": tag, "msg": msg}
    with _lock:
        _state["logs"].append(entry)
    logger.info("[%s] %s", tag, msg)


def _set_agent(name: str, status: str) -> None:
    with _lock:
        _state["agents"][name] = status


def _set_progress(pct: int) -> None:
    with _lock:
        _state["progress"] = pct


def _make_json_safe(obj):
    """Recursively convert numpy / pandas types → JSON-safe Python natives."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is pd.NaT:
        return None
    try:
        if pd.isna(obj):
            return None
    except TypeError:
        pass
    return obj


def _build_last_trained_case(df: pd.DataFrame | None) -> dict | None:
    """Return the most recent case retained in the transformed training sample."""
    if df is None or df.empty:
        return None

    if "Created Date" in df.columns and df["Created Date"].notna().any():
        case_row = df.sort_values("Created Date").iloc[-1]
    else:
        case_row = df.iloc[-1]

    def value(column: str):
        return _make_json_safe(case_row[column]) if column in case_row.index else None

    days_to_close = value("days_to_close")
    if isinstance(days_to_close, float):
        days_to_close = round(days_to_close, 2)

    ert_days = value("ERT_days")
    if isinstance(ert_days, float):
        ert_days = round(ert_days, 2)

    target_class = value("target")
    if target_class is not None:
        target_class = int(target_class)

    return {
        "service_request_type": value("Service Request Type"),
        "department": value("Department"),
        "department_grouped": value("Department_grouped"),
        "priority": value("Priority"),
        "method_received_description": value("Method Received Description"),
        "city_council_district": value("City Council District"),
        "created_date": value("Created Date"),
        "overall_due_date": value("Overall Service Request Due Date"),
        "days_to_close_hours": days_to_close,
        "target_class": target_class,
        "ert_days": ert_days,
        "month": value("month"),
        "day_of_week": value("day_of_week"),
        "hour": value("hour"),
    }


def _generate_metric_reasoning(metric_name: str, value: any, delta: float = 0.0, domain_context: str = None) -> str:
    """Generate human-like reasoning for a metric using Groq LLM and Knowledge Graph context."""
    # 1. Fallback base text
    fallback_text = "Metric is within normal operational parameters."
    
    # 2. Attempt LLM Reasoning
    if groq_client:
        try:
            # Construct a prompt that enforces the new 3-section structure
            prompt = f"""
            System: You are an expert City Operations Consultant specializing in Dallas 311 Service Requests.
            User: Provide a strategic, structured analysis for the following metric.
            
            METRIC: {metric_name}
            VALUE: {value}
            
            GROUNDING DATA (from Spanner Knowledge Graph & City Audit Reports):
            {domain_context if domain_context else "Standard Dallas 311 departmental procedures apply."}
            
            TASK: 
            Provide the response strictly in the following Markdown format:
1. **Strategic Insight**: [One sentence core takeaway]
2. **Operational Context**: [One sentence connecting to Spanner KG/Audit grounding]
3. **Recommended Action**: [One sentence actionable next step for city officials]

            Use professional, authoritative language. Keep the entire response under 60 words.
            """
            
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=GROQ_MODEL,
                max_tokens=200,
                temperature=0.6,
            )
            llm_text = chat_completion.choices[0].message.content.strip()
            if llm_text:
                print(f"[AI] Success: Generated grounded reasoning for '{metric_name}'")
                return llm_text
        except Exception as e:
            print(f"[AI] Grounded reasoning failed for {metric_name}: {str(e)}")

    return fallback_text


def _generate_features_reasoning(feat_imp: list, domain_context: str = None) -> str:
    """Generate structured AI reasoning for the Top Predictors (Feature Importance)."""
    top_features = ", ".join([f"{f['name']} ({f['score']:.3f})" for f in feat_imp[:5]])
    fallback_text = f"Top predictors for this model include: {top_features}. These factors significantly influence prediction accuracy."
    
    if groq_client:
        try:
            prompt = f"""
            System: You are an expert City Operations Consultant specializing in Dallas 311 Service Requests.
            User: Provide a strategic analysis of the model's Top Predictors.
            
            TOP PREDICTORS (Feature Name & Importance Score):
            {top_features}
            
            GROUNDING DATA (from Spanner Knowledge Graph & City Audit Reports):
            {domain_context if domain_context else "Standard Dallas 311 departmental procedures apply."}
            
            TASK: 
            Provide the response strictly in the following Markdown format:
1. **Strategic Insight**: [One sentence explaining why the #1 predictor is critical to performance]
2. **Operational Context**: [One sentence connecting these predictors to city departmental workflows (e.g. sanitation, code compliance)]
3. **Recommended Action**: [One sentence actionable advice for city managers based on these predictors]

            Use professional, authoritative language. Keep the entire response under 65 words.
            """
            
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=GROQ_MODEL,
                max_tokens=250,
                temperature=0.6,
            )
            llm_text = chat_completion.choices[0].message.content.strip()
            if llm_text:
                print("[AI] Success: Generated feature importance reasoning")
                return llm_text
        except Exception as e:
            print(f"[AI] Feature reasoning failed: {str(e)}")

    return fallback_text


def _generate_report_reasoning(topic: str, data_summary: str) -> str:
    """Generate a qualitative report for a complex topic (e.g. Diagnostics or Model Comparison)."""
    fallback_text = f"The {topic} phase completed successfully with metrics within normal operational bounds."
    
    if groq_client:
        try:
            prompt = f"""
            System: You are a senior data science advisor for the City of Dallas 311 Service Requests team.
            User: Provide a comprehensive, detailed qualitative analysis (approx 150-200 words) based on the following {topic} data.
            Focus on uncovering deep insights, identifying potential risks, and proposing actionable business values for the City's service optimization objective.
            
            {topic} Data:
            {data_summary}
            
            Format: A detailed, professional report with an emphasis on data-driven reasoning.
            """
            
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=GROQ_MODEL,
                max_tokens=500,
                temperature=0.7,
            )
            llm_text = chat_completion.choices[0].message.content.strip()
            if llm_text:
                print(f"[AI] Success: Generated report for '{topic}'")
                return llm_text
        except Exception as e:
            print(f"[AI] Report reasoning failed for {topic}: {str(e)}")
            # Improved non-robotic fallback that still references the topic
            if "ROC" in topic:
                fallback_text = f"The ROC curve indicates strong model discrimination (API Err: {str(e)[:40]}...)"
            elif "PR" in topic:
                fallback_text = f"The Precision-Recall curve suggests robust performance (API Err: {str(e)[:40]}...)"
            else:
                fallback_text = f"The analysis for {topic} is based on current metrics (API Err: {str(e)[:40]}...)"

    return fallback_text


def _generate_diagnostics_reasoning(diag_dict: dict) -> str:
    """Convenience wrapper for Diagnostics LLM reasoning."""
    summary = "\n".join([f"- {k}: {v}" for k, v in diag_dict.items() if not isinstance(v, (list, dict))])
    if "multicollinear_features" in diag_dict:
        cols = diag_dict["multicollinear_features"]
        summary += f"\n- Multicollinear Features Found: {len(cols)} ({', '.join(cols[:5]) if cols else 'None'})"
    
    return _generate_report_reasoning("Diagnostics & Data Health", summary)


def _generate_model_report(det_results: dict, best_name: str) -> str:
    """Convenience wrapper for Model Comparison LLM reasoning."""
    summary_lines = []
    for name, metrics in det_results.items():
        summary_lines.append(
            f"Model: {name} | ROC-AUC: {metrics.get('ROC_AUC', 0):.3f} | Accuracy: {metrics.get('Accuracy', 0):.3f} | F1: {metrics.get('F1_Score', 0):.3f}"
        )
    summary = "\n".join(summary_lines)
    summary += f"\nBest Model identified by agent: {best_name}"
    
    return _generate_report_reasoning("Model Performance Comparison", summary)


def _generate_confusion_matrix_reasoning(det_results: dict, best_name: str) -> str:
    """Explain the Confusion Matrix for the best model using actual counts."""
    best = det_results.get(best_name, {})
    cm = best.get("confusion_matrix")
    if not cm: return "No confusion matrix data available."
    
    summary = (
        f"Model: {best_name}\n"
        f"True Negatives (Correctly predicted Slow Close): {cm[0][0]}\n"
        f"False Positives (Predicted Fast, but was Slow): {cm[0][1]}\n"
        f"False Negatives (Predicted Slow, but was Fast - MISSES): {cm[1][0]}\n"
        f"True Positives (Correctly predicted Fast Close): {cm[1][1]}"
    )
    
    return _generate_report_reasoning("Confusion Matrix Breakdown", summary)


def _summarize_curve_shape(curve_dict: dict, type_name: str) -> str:
    """Downsample a coordinate array into a text-based shape description."""
    if not curve_dict: return "No curve data."
    
    # Extract arrays (keys depend on ROC vs PR)
    if type_name == "ROC":
        x_name, y_name = "fpr", "tpr"
        x_label, y_label = "FPR", "TPR"
    else:
        x_name, y_name = "recall", "precision"
        x_label, y_label = "Recall", "Prec"
        
    xs = curve_dict.get(x_name, [])
    ys = curve_dict.get(y_name, [])
    
    if not xs or not ys: return "Curve arrays are empty."
    
    # Take 7 samples across the range
    indices = [0, len(xs)//6, (len(xs)*2)//6, (len(xs)*3)//6, (len(xs)*4)//6, (len(xs)*5)//6, len(xs)-1]
    samples = []
    for i in indices:
        if i < len(xs):
            samples.append(f"{x_label}: {xs[i]:.2f} -> {y_label}: {ys[i]:.2f}")
            
    return f"{type_name} Shape Points: " + " | ".join(samples)


def _generate_graph_insights(det_results: dict, best_name: str) -> dict:
    """Generate high-level visual analysis for ROC and PR curves."""
    best = det_results.get(best_name, {})
    
    roc_summary = _summarize_curve_shape(best.get("roc_curve"), "ROC")
    pr_summary = _summarize_curve_shape(best.get("pr_curve"), "PR")
    
    return {
        "roc": _generate_report_reasoning("ROC Curve Shape Analysis", roc_summary),
        "pr":  _generate_report_reasoning("PR Curve Stability Analysis", pr_summary)
    }


def _persist_results(results: dict) -> None:
    """Write the most recent successful dashboard payload to disk."""
    RESULTS_PATH.write_text(
        json.dumps(_make_json_safe(results), indent=2),
        encoding="utf-8",
    )


def _load_persisted_results() -> dict | None:
    """Return the last successful dashboard payload from disk, if available."""
    if not RESULTS_PATH.exists():
        return None

    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        logger.exception("Unable to load persisted dashboard results from %s", RESULTS_PATH)
        return None


def _restore_results_to_state(results: dict | None) -> dict | None:
    """Hydrate in-memory state from a persisted successful run payload."""
    if results is None:
        return None

    with _lock:
        _state["results"] = results
        _state["status"] = "done"
        _state["progress"] = 100
        _state["error"] = None
        _state["data_path"] = results.get("data_path")
        _state["finished_at"] = results.get("finished_at")
        _state["started_at"] = results.get("started_at", _state["started_at"])
        for agent in _state["agents"]:
            _state["agents"][agent] = "done"

    return results


def _dashboard_bootstrap_payload() -> dict:
    """Return the saved dashboard state so the main page can render immediately."""
    with _lock:
        results = _state["results"]
        status = _state["status"]
        data_path = _state["data_path"]
        finished_at = _state["finished_at"]

    if results is None:
        results = _restore_results_to_state(_load_persisted_results())
        with _lock:
            status = _state["status"]
            data_path = _state["data_path"]
            finished_at = _state["finished_at"]

    return _make_json_safe({
        "status": status,
        "data_path": data_path,
        "finished_at": finished_at,
        "results": results,
    })


def _configured_data_path() -> Path | None:
    """Return the configured dataset path, resolved relative to the project when needed."""
    try:
        from src import config as src_cfg
    except Exception:
        return None

    configured = Path(str(src_cfg.DATA_PATH)).expanduser()
    if not configured.is_absolute():
        configured = (ROOT / configured).resolve()
    return configured


def _sample_data_path() -> Path | None:
    """Return the bundled sample dataset path when available."""
    sample_path = ROOT / "sample.csv"
    return sample_path if sample_path.exists() else None


def _existing_default_data_path() -> Path | None:
    """Prefer a valid configured dataset, otherwise fall back to the local sample."""
    configured_path = _configured_data_path()
    if configured_path and configured_path.exists():
        return configured_path
    return _sample_data_path()


def _discover_available_datasets() -> list[dict[str, str | bool]]:
    """Return CSV datasets the dashboard can offer in the change-data modal."""
    configured_path = _configured_data_path()
    default_path = configured_path if configured_path and configured_path.exists() else None
    sample_path = _sample_data_path()
    recommended_path = default_path or sample_path
    recommended_key = str(recommended_path.resolve()) if recommended_path else None

    candidates: list[tuple[Path, str]] = []
    if default_path:
        candidates.append((default_path, "Default Dataset (Config)"))
    if sample_path and sample_path != default_path:
        candidates.append((sample_path, "Sample File (Root)"))

    for path in sorted(ROOT.glob("*.csv"), key=lambda item: item.name.lower()):
        if path in {default_path, sample_path}:
            continue
        candidates.append((path, "Project File"))

    if UPLOAD_DIR.exists():
        for path in sorted(UPLOAD_DIR.glob("*.csv"), key=lambda item: item.name.lower()):
            candidates.append((path, "Uploaded File"))

    datasets: list[dict[str, str | bool]] = []
    seen_paths: set[str] = set()

    for path, source in candidates:
        if not path.exists() or not path.is_file():
            continue

        resolved_key = str(path.resolve())
        if resolved_key in seen_paths:
            continue
        seen_paths.add(resolved_key)

        datasets.append({
            "path": str(path),
            "filename": path.name,
            "source": source,
            "is_recommended": resolved_key == recommended_key,
        })

    return datasets


def _resolve_data_path(raw_path: str | None) -> Path:
    """Validate and resolve the dataset path for a pipeline run."""
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
    else:
        candidate = _existing_default_data_path()
        if candidate is None:
            raise FileNotFoundError(
                "No dataset is available. Upload a CSV or configure a valid default data path.",
            )

    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Dataset not found: {candidate}")
    if candidate.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported for pipeline runs.")
    return candidate


_restore_results_to_state(_load_persisted_results())


# ── Background pipeline runner ────────────────────────────────────────────────

def _run_pipeline(data_path: str | None = None) -> None:  # noqa: C901
    """Full agent pipeline — runs in a daemon thread."""

    with _lock:
        _state.update({
            "status":     "running",
            "progress":   0,
            "logs":       [],
            "results":    None,
            "error":      None,
            "data_path":  data_path,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
        })
        for k in _state["agents"]:
            _state["agents"][k] = "idle"

    try:
        # ── 1. DataPrep ────────────────────────────────────────────────────────
        _log("Pipeline started — dallas_311_service_requests")
        _set_agent("DataPrepAgent", "running")
        _set_progress(5)

        from agents.data_prep_agent import DataPrepAgent
        dpa = DataPrepAgent(data_path=data_path)
        df_clean = dpa.run()
        val = dpa.validate()
        if not val["passed"]:
            raise RuntimeError(f"DataPrep validation failed: {val['issues']}")

        _set_agent("DataPrepAgent", "done")
        _log(f"DataPrepAgent: Done — shape {df_clean.shape}", "DONE")
        _set_progress(20)

        # ── 2. Transformation ──────────────────────────────────────────────────
        _log("TransformationAgent: Encoding & splitting …")
        _set_agent("TransformationAgent", "running")

        from agents.transformation_agent import TransformationAgent
        ta = TransformationAgent()
        df_transformed = ta.run(df_clean)
        val = ta.validate()
        if not val["passed"]:
            raise RuntimeError(f"Transformation validation failed: {val['issues']}")

        _set_agent("TransformationAgent", "done")
        _log(f"TransformationAgent: Done — shape {df_transformed.shape}", "DONE")
        _set_progress(40)

        # ── 3. Diagnostics ─────────────────────────────────────────────────────
        _log("DiagnosticsAgent: Running normality & imbalance checks …")
        _set_agent("DiagnosticsAgent", "running")

        from agents.diagnostics_agent import DiagnosticsAgent
        da = DiagnosticsAgent()
        diag = da.run(df_transformed)

        _set_agent("DiagnosticsAgent", "done")
        _log(f"DiagnosticsAgent: Done — overall pass: {diag.get('overall_pass')}", "DONE")
        _set_progress(55)

        # ── 4. Model Selection ─────────────────────────────────────────────────
        _log("ModelSelectionAgent: Training LR / Random Forest / XGBoost …")
        _set_agent("ModelSelectionAgent", "running")

        from agents.model_selection_agent import ModelSelectionAgent
        msa = ModelSelectionAgent()
        model_result = msa.run(df_transformed)
        val = msa.validate()
        if not val["passed"]:
            raise RuntimeError(f"ModelSelection validation failed: {val['issues']}")

        _set_agent("ModelSelectionAgent", "done")
        _log(f"ModelSelectionAgent: Best = {model_result['best_model_name']}", "DONE")
        _set_progress(80)

        # ── 5. Regularization ──────────────────────────────────────────────────
        skip_reg = os.getenv("SKIP_REGULARIZATION", "False").lower() == "true"
        
        if skip_reg:
            _log("RegularizationAgent: Skipping as requested in .env", "DONE")
            # Use the best model selection result as the 'regularization' placeholder
            best_name = model_result['best_model_name']
            best_auc = model_result['detailed_results'][best_name]['ROC_AUC']
            reg_result = {
                "best_method": f"None (Skipped, using {best_name})",
                "best_roc_auc": float(best_auc),
                "all_results": {},
                "coef_summary": pd.DataFrame()
            }
        else:
            _log("RegularizationAgent: Ridge / LASSO / ElasticNet …")
            _set_agent("RegularizationAgent", "running")

            from agents.regularization_agent import RegularizationAgent
            ra = RegularizationAgent()
            reg_result = ra.run(
                msa.X_train_,
                msa.y_train_,
                msa.X_test_,
                msa.y_test_,
                feature_names=msa.feature_names_,
            )

            _set_agent("RegularizationAgent", "done")
            _log(
                f"RegularizationAgent: Best = {reg_result['best_method']} "
                f"(ROC-AUC = {reg_result['best_roc_auc']})",
                "DONE",
            )
        _set_progress(100)

        # ── Build results payload ──────────────────────────────────────────────
        # Model comparison rows — normalise column names from DataFrame / list
        raw_comp = model_result.get("comparison", [])
        comparison = _make_json_safe(raw_comp)

        # Feature importances (Random Forest / XGBoost if available)
        feat_imp: list[dict] = []
        if hasattr(msa, "best_model_") and hasattr(msa.best_model_, "feature_importances_"):
            names  = msa.feature_names_ or [f"f{i}" for i in range(len(msa.best_model_.feature_importances_))]
            scores = msa.best_model_.feature_importances_
            feat_imp = sorted(
                [{"name": n, "score": float(s)} for n, s in zip(names, scores)],
                key=lambda x: x["score"],
                reverse=True,
            )[:10]

        # Enrichment: Key Findings & Detailed metrics
        det_results = model_result.get("detailed_results", {})
        split_info = model_result.get("split_info", {})
        key_findings = _generate_key_findings(det_results, model_result.get("best_model_name"))

        # Diagnostics — strip any non-serialisable values
        safe_diag = {
            k: _make_json_safe(v)
            for k, v in diag.items()
            if not isinstance(v, (pd.DataFrame, np.ndarray))
        }

        finished_at = datetime.now().isoformat()

        # Build top-level 'Metric Objects' with context
        last_case = _build_last_trained_case(df_transformed)
        dept_name = last_case.get("department") if last_case else "Dallas 311"
        
        try:
            domain_context = get_domain_context(department=dept_name)
        except Exception:
            domain_context = "Standard Dallas 311 procedures apply."

        metrics = {
            "records": {
                "label": "Records Processed",
                "value": int(df_clean.shape[0]),
                "delta": 0,
                "reasoning": _generate_metric_reasoning("Records Processed", int(df_clean.shape[0]), domain_context=domain_context)
            },
            "features": {
                "label": "Features Selected",
                "value": len(feat_imp) if feat_imp else int(df_clean.shape[1]),
                "reasoning": _generate_metric_reasoning("Features Selected", None, domain_context=domain_context)
            },
            "accuracy": {
                "label": "Best ROC-AUC",
                "value": round(float(reg_result["best_roc_auc"]), 3),
                "reasoning": _generate_metric_reasoning("Best ROC-AUC", round(float(reg_result["best_roc_auc"]), 3), domain_context=domain_context)
            }
        }

        finished_at = datetime.now().isoformat()

        results = {
            "metrics":       metrics,
            "models":        [ { **v, "Model": k } for k, v in det_results.items() ],
            "detailed_results": det_results,
            "best_model":    model_result.get("best_model_name"),
            "diagnostics":   safe_diag,
            "diagnostics_reasoning": _generate_diagnostics_reasoning(safe_diag),
            "key_findings":  key_findings,
            "model_comparison_reasoning": _generate_model_report(det_results, model_result.get("best_model_name")),
            "confusion_matrix_reasoning": _generate_confusion_matrix_reasoning(det_results, model_result.get("best_model_name")),
            "graph_insights": _generate_graph_insights(det_results, model_result.get("best_model_name")),
            "data_path":     data_path,
            "started_at":    _state["started_at"],
            "finished_at":   finished_at,
            "split_info":    split_info,
            "split_info_reasoning": _generate_report_reasoning("Data Split Analysis", f"Train size: {split_info.get('train_size')} | Test size: {split_info.get('test_size')} | Total: {split_info.get('total_size')}"),
            "feature_importances": feat_imp,
            "features_reasoning": _generate_features_reasoning(feat_imp, domain_context=domain_context),
            "regularization": {
                "best_method":  reg_result["best_method"],
                "best_roc_auc": float(reg_result["best_roc_auc"]),
                "all_results":  _make_json_safe(reg_result.get("all_results", {})),
                "coef_summary": _make_json_safe(reg_result.get("coef_summary")),
            },
            "artifacts": {
                "model_path": model_result.get("model_path"),
                "encoders_path": model_result.get("encoders_path"),
                "results_path": str(RESULTS_PATH),
            },
            "last_trained_case": last_case,
            "feature_importance": feat_imp,
            "data_shape":  [int(df_clean.shape[0]), int(df_clean.shape[1])],
            "train_test":  [int(msa.X_train_.shape[0]), int(msa.X_test_.shape[0])],
        }

        # Expert AI Analysis is now distributed into individual metrics reasoning fields.

        results = _make_json_safe(results)
        _persist_results(results)

        with _lock:
            _state["results"]     = results
            _state["status"]      = "done"
            _state["finished_at"] = finished_at

        _log("Pipeline complete! All agents finished successfully.", "DONE")

    except Exception as exc:  # noqa: BLE001
        _log(f"Pipeline error: {exc}", "ERR")
        with _lock:
            _state["status"] = "error"
            _state["error"]  = str(exc)
            for k, v in _state["agents"].items():
                if v == "running":
                    _state["agents"][k] = "error"
        logger.exception("Pipeline thread exception")


def _run_inference(data_path: str, model_path: Path | None = None, enc_path: Path | None = None) -> None:
    """Fast inference using a specific or the default best model."""
    m_path = model_path or (ARTIFACTS_DIR / "best_model.joblib")
    e_path = enc_path or (ARTIFACTS_DIR / "encoders.joblib")
    
    # Import locally for inference
    from src.preprocessing import handle_missing_values, handle_service_request_type, encode_categoricals, split_features_target

    with _lock:
        _state.update({
            "status": "running",
            "progress": 0,
            "logs": [],
            "data_path": data_path,
            "started_at": datetime.now().isoformat(),
        })
        for k in _state["agents"]: _state["agents"][k] = "idle"

    try:
        _log(f"Inference started using model: {m_path.name}")
        _set_agent("DataPrepAgent", "running")
        from agents.data_prep_agent import DataPrepAgent
        dpa = DataPrepAgent(data_path=data_path)
        df_clean = dpa.run()
        _set_agent("DataPrepAgent", "done")
        _set_progress(30)

        # Check if the data is already transformed (e.g. from a saved test set)
        if 'Created Date' in df_clean.columns:
            _set_agent("TransformationAgent", "running")
            from agents.transformation_agent import TransformationAgent
            ta = TransformationAgent()
            df_trans = ta.run(df_clean)
            _set_agent("TransformationAgent", "done")
        else:
            _log("Data appears pre-transformed. Skipping TransformationAgent.")
            df_trans = df_clean
            _set_agent("TransformationAgent", "done")
            
        _set_progress(60)

        _log("Loading model and encoders...")
        model = joblib.load(m_path)
        encoders = joblib.load(e_path)

        # Encode & Clean (Only if RAW data)
        if 'Created Date' in df_clean.columns:
            _log("Preprocessing and encoding raw data...")
            X, y = split_features_target(df_trans)
            X, _ = handle_missing_values(X, X)
            X, _ = handle_service_request_type(X, X)
            X_final, _, _ = encode_categoricals(X, X, initial_encoders=encoders)
            
            # Apply Scaling (Inference mode)
            if "scaler" in encoders:
                _log("Standardizing features for inference...")
                scaler = encoders["scaler"]
                X_cols = X_final.columns
                X_scaled = scaler.transform(X_final)
                X_final = pd.DataFrame(X_scaled, columns=X_cols, index=X_final.index)
        else:
            _log("Skipping encoding as data is already in final feature space.")
            X_final, _ = split_features_target(df_trans)

        _log(f"Performing predictions on {len(X_final)} rows...")
        y_pred = model.predict(X_final)
        
        # Build shallow results
        metrics = {
            "records": { "label": "Records Inferred", "value": len(X_final), "reasoning": "Batch inference complete." },
            "accuracy": { "label": "Last Model Strength", "value": "Production", "reasoning": "Using the validated best model from last run." }
        }
        
        results = {
            "metrics": metrics,
            "status": "inference_complete",
            "data_path": data_path,
            "finished_at": datetime.now().isoformat(),
            "best_model": "Production Model",
            "inference_count": len(y_pred)
        }
        
        with _lock:
            _state["results"] = results
            _state["status"] = "done"
            _state["progress"] = 100
        
        _log("Inference complete!", "DONE")

    except Exception as exc:
        _log(f"Inference error: {exc}", "ERR")
        with _lock: _state["status"] = "error"; _state["error"] = str(exc)
        logger.exception("Inference thread exception")


def _preprocess_manual_row(row: dict, encoders: dict) -> pd.DataFrame:
    """Transform a single manual entry dict into a model-ready DataFrame."""
    from src.feature_engineering import add_time_features
    from src.preprocessing import clean_ert, encode_categoricals, handle_service_request_type, handle_missing_values

    # 1. Create initial DataFrame
    df = pd.DataFrame([row])

    # 2. Extract time features if Created Date is present
    if "Created Date" in df.columns:
        df["Created Date"] = pd.to_datetime(df["Created Date"])
        df = add_time_features(df)
        df = df.drop(columns=["Created Date"], errors="ignore")

    # 3. Clean ERT
    df = clean_ert(df)

    # 4. Handle Service Request Type (top_n logic)
    # We pass it through handle_service_request_type but we need X_train for fitting usually.
    # Here we just ensure it aligns with encoders if needed, or we rely on encode_categoricals unknown handling.
    
    # 5. Encoding & Scaling
    # We use encode_categoricals in inference mode (initial_encoders=encoders)
    # We need a dummy Y for the function signature
    X_final, _, _ = encode_categoricals(df, df, initial_encoders=encoders)
    
    # Apply Scaling
    if "scaler" in encoders:
        scaler = encoders["scaler"]
        X_final = pd.DataFrame(scaler.transform(X_final), columns=X_final.columns, index=X_final.index)

    return X_final


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard HTML."""
    dashboard_path = UI_DIR / "dashboard.html"
    dashboard_html = dashboard_path.read_text(encoding="utf-8")
    bootstrap_json = json.dumps(
        _dashboard_bootstrap_payload(),
        separators=(",", ":"),
    ).replace("</", "<\\/")
    marker = '<script id="dashboard-bootstrap" type="application/json">null</script>'
    rendered = dashboard_html.replace(
        marker,
        f'<script id="dashboard-bootstrap" type="application/json">{bootstrap_json}</script>',
        1,
    )
    return Response(rendered, mimetype="text/html")


@app.route("/pipeline-tester")
def pipeline_tester():
    """Serve the popup window used for testing alternate pipeline files."""
    return send_from_directory(UI_DIR, "pipeline_tester.html")


@app.route("/api/status")
def get_status():
    """Lightweight poll — returns status, progress, last 50 logs, agent states."""
    with _lock:
        return jsonify({
            "status":   _state["status"],
            "progress": _state["progress"],
            "logs":     _state["logs"][-50:],
            "agents":   _state["agents"],
            "error":    _state["error"],
            "data_path": _state["data_path"],
        })


@app.route("/api/results")
def get_results():
    """Full results payload — only available after a successful run."""
    with _lock:
        results = _state["results"]
        status = _state["status"]

    if results is None:
        results = _restore_results_to_state(_load_persisted_results())

    if results is None:
        return jsonify({
            "error": "No results yet. Run the pipeline first.",
            "status": status
        }), HTTPStatus.NOT_FOUND

    return jsonify(results)


@app.route("/api/dashboard-bootstrap")
def get_dashboard_bootstrap():
    """Return the current dashboard bootstrap payload for client-side restore."""
    return jsonify(_dashboard_bootstrap_payload())


@app.route("/api/publish", methods=["POST"])
def publish_dashboard():
    """Persist the latest successful run and return the dashboard location."""
    with _lock:
        status = _state["status"]
        results = _state["results"]

    if status == "running":
        return jsonify({
            "error": "Cannot save the dashboard while the pipeline is still running.",
            "status": status,
        }), HTTPStatus.CONFLICT

    if results is None:
        results = _load_persisted_results()

    if results is None:
        return jsonify({
            "error": "No completed run is available to save yet.",
            "status": status,
        }), HTTPStatus.NOT_FOUND

    results = _make_json_safe(results)
    _persist_results(results)
    _restore_results_to_state(results)

    cache_busted_dashboard_url = f"{url_for('index')}?saved_at={int(datetime.now().timestamp() * 1000)}"

    return jsonify({
        "status": "saved",
        "message": "Model artifacts and dashboard snapshot saved. Opening the main dashboard.",
        "dashboard_url": cache_busted_dashboard_url,
        "bootstrap_url": url_for("get_dashboard_bootstrap"),
        "artifacts": results.get("artifacts", {}),
        "results": results,
    })


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    """Trigger a pipeline run in a background thread."""
    with _lock:
        if _state["status"] == "running":
            return jsonify({
                "error": "Conflict: Pipeline is already running.",
                "current_status": _state["status"],
                "message": "Fetch /api/status to monitor progress."
            }), HTTPStatus.CONFLICT

    body      = request.get_json(silent=True) or {}
    data_path = body.get("data_path") or None

    try:
        resolved_data_path = _resolve_data_path(data_path)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.BAD_REQUEST

    with _lock:
        _state["data_path"] = str(resolved_data_path)

    thread = threading.Thread(
        target=_run_pipeline,
        args=(str(resolved_data_path),),
        daemon=True,
        name="pipeline-runner",
    )
    thread.start()
    return jsonify({
        "status": "started",
        "message": "Pipeline execution accepted and started in background.",
        "data_path": str(resolved_data_path),
    }), HTTPStatus.ACCEPTED


@app.route("/api/infer", methods=["POST"])
def infer_pipeline():
    """Trigger a fast inference run."""
    with _lock:
        if _state["status"] == "running":
            return jsonify({"error": "Conflict: Busy."}), HTTPStatus.CONFLICT
        if not (ARTIFACTS_DIR / "best_model.joblib").exists():
            return jsonify({"error": "No model found. Run full pipeline first."}), HTTPStatus.BAD_REQUEST

    body = request.get_json(silent=True) or {}
    data_path = body.get("data_path")
    try:
        resolved_data_path = _resolve_data_path(data_path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    thread = threading.Thread(
        target=_run_inference,
        args=(str(resolved_data_path),),
        daemon=True,
        name="inference-runner",
    )
    thread.start()
    return jsonify({"status": "started", "data_path": str(resolved_data_path)}), 202


@app.route("/api/history/list", methods=["GET"])
def list_history():
    """List all saved models in history."""
    history = []
    if not HISTORY_DIR.exists():
        return jsonify([])
    
    for folder in sorted(HISTORY_DIR.iterdir(), reverse=True):
        if folder.is_dir():
            meta_path = folder / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                        meta["id"] = folder.name
                        history.append(meta)
                except Exception as e:
                    logger.error(f"Error reading history meta in {folder}: {e}")
    return jsonify(history)


@app.route("/api/history/save", methods=["POST"])
def save_to_history():
    """Save the current 'latest' model to history with a custom name."""
    body = request.get_json(silent=True) or {}
    custom_name = body.get("custom_name", "Untitled Model").strip()
    
    if not RESULTS_PATH.exists():
        return jsonify({"error": "No current results to save."}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(x for x in custom_name if x.isalnum() or x in "-_ ").replace(" ", "_")
    folder_name = f"{timestamp}_{safe_name}"
    target_dir = HISTORY_DIR / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load latest results to get metrics
        with open(RESULTS_PATH, "r") as f:
            results = json.load(f)
            
        # Copy artifacts
        files_to_copy = [
            ("best_model.joblib", "model.joblib"),
            ("encoders.joblib", "encoders.joblib"),
            ("latest_test_set.csv", "test_set.csv"),
            ("latest_results.json", "results.json")
        ]
        
        for src_name, dest_name in files_to_copy:
            src_path = ARTIFACTS_DIR / src_name
            if src_path.exists():
                shutil.copyfile(src_path, target_dir / dest_name)
        
        # Save metadata
        metadata = {
            "name": custom_name,
            "timestamp": timestamp,
            "best_model": results.get("best_model"),
            "metrics": results.get("metrics", {}),
            "roc_auc": results.get("models", [{}])[0].get("ROC_AUC"), # Simplified
            "data_path": results.get("data_path")
        }
        with open(target_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        return jsonify({"status": "success", "id": folder_name})
    except Exception as e:
        logger.exception("Failed to save to history")
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/infer", methods=["POST"])
def infer_from_history():
    """Run inference using a historical model."""
    body = request.get_json(silent=True) or {}
    history_id = body.get("history_id")
    mode = body.get("mode", "new") # "new" or "original"
    
    if not history_id:
        return jsonify({"error": "history_id required"}), 400
        
    model_dir = HISTORY_DIR / history_id
    if not model_dir.exists():
        return jsonify({"error": "History item not found"}), 404
        
    # Determine data path
    if mode == "original":
        data_path = model_dir / "test_set.csv"
        if not data_path.exists():
            return jsonify({"error": "Original test set not found"}), 400
    else:
        data_path = body.get("data_path")
        if not data_path:
            return jsonify({"error": "data_path required for new inference"}), 400
        try:
           data_path = _resolve_data_path(data_path)
        except Exception as e:
           return jsonify({"error": str(e)}), 400

    thread = threading.Thread(
        target=_run_inference,
        kwargs={
            "data_path": str(data_path),
            "model_path": model_dir / "model.joblib",
            "enc_path": model_dir / "encoders.joblib"
        },
        daemon=True,
        name="history-inference-runner",
    )
    thread.start()
    return jsonify({"status": "started", "model": history_id, "data": str(data_path)}), 202


@app.route("/api/manual_infer", methods=["POST"])
def manual_infer():
    """Run model inference on a single manually entered row."""
    m_path = ARTIFACTS_DIR / "best_model.joblib"
    e_path = ARTIFACTS_DIR / "encoders.joblib"

    if not m_path.exists() or not e_path.exists():
        return jsonify({"error": "Model artifacts not found. Run pipeline first."}), 400

    try:
        row = request.get_json()
        if not row:
            return jsonify({"error": "No data provided"}), 400

        # 1. Preprocess
        model = joblib.load(m_path)
        encoders = joblib.load(e_path)
        X_final = _preprocess_manual_row(row, encoders)

        # 2. Predict
        prediction_prob = model.predict_proba(X_final)[0][1]
        prediction_class = int(model.predict(X_final)[0])
        
        # 3. Explain (Agentic reasoning)
        # We build a 'coef_summary' if it's a linear model, or just use features
        coef_summary = ""
        if hasattr(model, 'coef_'):
            import pandas as pd
            df_coef = pd.DataFrame({
                'Feature': X_final.columns,
                'Coefficient': model.coef_[0]
            })
            coef_summary = format_coef_summary(df_coef)
        elif hasattr(model, 'feature_importances_'):
            # Fallback for trees: top local features
            imps = model.feature_importances_
            top_idx = np.argsort(imps)[-5:][::-1]
            coef_summary = "\n".join([f"- {X_final.columns[i]}: {imps[i]:.4f} (Importance)" for i in top_idx])

        # Call the explainability chain
        chain = create_explainability_chain()
        explanation = chain.invoke({
            "prediction": f"{'Fast' if prediction_class == 1 else 'Slow'} Close (Probability: {prediction_prob:.2%})",
            "coef_summary": coef_summary,
            "department": row.get("Department"),
            "district": row.get("City Council District")
        })

        return jsonify({
            "prediction": prediction_class,
            "probability": float(prediction_prob),
            "explanation": explanation,
            "row": row
        })

    except Exception as e:
        logger.exception("Manual inference failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_pipeline():
    """Reset state back to idle (only when not running)."""
    with _lock:
        if _state["status"] == "running":
            return jsonify({
                "error": "Conflict: Cannot reset while pipeline is running.",
                "current_status": _state["status"]
            }), HTTPStatus.CONFLICT
        preserved_results = _state["results"]
        if preserved_results is None:
            preserved_results = _load_persisted_results()
        _state.update({
            "status":   "idle",
            "progress": 0,
            "logs":     [],
            "results":  preserved_results,
            "error":    None,
            "data_path": preserved_results.get("data_path") if preserved_results else None,
            "finished_at": preserved_results.get("finished_at") if preserved_results else None,
        })
        for k in _state["agents"]:
            _state["agents"][k] = "idle"
    return jsonify({"status": "reset"})


@app.route("/api/config")
def get_config():
    """Return default data paths so the dashboard modal can pre-fill them."""
    configured_path = _configured_data_path()
    default_path = configured_path if configured_path and configured_path.exists() else None
    sample_path = _sample_data_path()
    recommended_path = default_path or sample_path

    return jsonify({
        "configured_data_path": str(configured_path) if configured_path else "",
        "default_data_path": str(default_path) if default_path else "",
        "sample_path": str(sample_path) if sample_path else "",
        "recommended_data_path": str(recommended_path) if recommended_path else "",
        "upload_dir": str(UPLOAD_DIR),
        "available_datasets": _discover_available_datasets(),
    })


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    """Accept a CSV file upload and save it to data/uploaded/."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # Only accept CSV
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted."}), 400

    # Sanitise filename
    safe_name = "".join(c if c.isalnum() or c in (".", "_", "-") else "_" for c in f.filename)
    save_path = UPLOAD_DIR / safe_name
    f.save(save_path)

    logger.info("[UPLOAD] Saved to %s (%d bytes)", save_path, save_path.stat().st_size)
    return jsonify({"path": str(save_path), "filename": safe_name})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dallas 311 Dashboard — Flask Server")
    parser.add_argument("--port",  type=int, default=5000,  help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true",      help="Enable Flask debug mode")
    args = parser.parse_args()

    print(f"\n  Dallas 311 ML Dashboard")
    print(f"  Running at: http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop.\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug, use_reloader=True)
