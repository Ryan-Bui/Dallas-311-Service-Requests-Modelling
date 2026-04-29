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
import gc
try:
    import psutil
except ImportError:
    psutil = None
from http import HTTPStatus
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()
import warnings
warnings.filterwarnings("ignore")
import shutil
import joblib
from groq import Groq

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]   # project root
UI_DIR    = Path(__file__).resolve().parent        # ui/

# Render Persistent Disk detection
PERSISTENT_DIR = Path("/var/data")
if PERSISTENT_DIR.exists():
    STORAGE_BASE = PERSISTENT_DIR
else:
    STORAGE_BASE = ROOT

UPLOAD_DIR = STORAGE_BASE / "data" / "uploaded"
ARTIFACTS_DIR = STORAGE_BASE / "models"
HISTORY_DIR = ARTIFACTS_DIR / "history"
RESULTS_PATH = ARTIFACTS_DIR / "latest_results.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory, url_for, session
try:
    from flask_cors import CORS as _CORS
except ImportError:
    _CORS = None  # optional — not needed for same-origin requests
from dotenv import load_dotenv
load_dotenv(override=True)

from inference.explainability_chain import create_explainability_chain, format_coef_summary, get_domain_context, Neo4jChatMemory
from inference.llm_factory import get_llm

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(UI_DIR))
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dallas_311_fallback_secret")
if _CORS:
    _CORS(app)

import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = STORAGE_BASE / "userbase.db"

def init_auth_db():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            gmail_app_password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_auth_db()

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
        "ExplorerAgent":       "idle",
        "ReinforcementJudge":  "idle",
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
        # Cap logs to prevent memory leak over time
        if len(_state["logs"]) > 500:
            _state["logs"] = _state["logs"][-500:]
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
            System: You are a student who just recently studied about Dallas 311 Service Requests for an academic project.
            User: Provide an intelligent response to a common citizen's query in 100-200 words. AVoid jargon.

            
            METRIC: {metric_name}
            VALUE: {value}
            
            GROUNDING DATA (from Neo4j Knowledge Graph & City Audit Reports):
            {domain_context if domain_context else "Standard Dallas 311 departmental procedures apply."}
            
            TASK: 
            Provide the response strictly in this format:
            * **{metric_name} Analysis**: [Sentence 1: A direct insight about this value]. [Sentence 2: A contextual connection to our Neo4j Knowledge Graph or City Audits].

            Keep the entire response under 50 words. Do not use more than two sentences.
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
            
            GROUNDING DATA (from Neo4j Knowledge Graph & City Audit Reports):
            {domain_context if domain_context else "Standard Dallas 311 departmental procedures apply."}
            
            TASK: 
            Provide the response strictly in this format:
            * **Top Predictors Insight**: [Sentence 1: Explain why these features dominate the model's predictions]. [Sentence 2: Connect these predictors to the departmental bottlenecks identified in Neo4j].

            Keep the entire response under 55 words. Do not use more than two sentences.
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
        "agents": _state["agents"],
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
    """Prefer a valid configured dataset, otherwise fall back to the local small sample."""
    # Priority: 1. Manual selection (handled in route) 2. sample1.csv (small) 3. Configured path
    small_sample = ROOT / "sample1.csv"
    if small_sample.exists():
        return small_sample
    
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

    # --- Pipeline Reporting Defaults (Safety Shield) ---
    row_count, col_count = 0, 0
    train_size, test_size = 0, 0
    diag = {"overall_pass": False, "issues": ["Pipeline started"]}
    model_result = {"best_model_name": "None", "detailed_results": {}}
    reg_result = {"best_method": "Idle", "best_roc_auc": 0.0}
    feat_imp = []
    df_clean = None
    df_transformed = None

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
        
        # Capture research topic before any potential clearing
        service_col = 'Service Request Type'
        target_service = "Dallas 311"
        if service_col in df_clean.columns:
            target_service = str(df_clean[service_col].mode()[0])
            _log(f"Intelligence Target Identified: {target_service}", "DEBUG")
        
        # Log memory usage
        if psutil:
            mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            _log(f"Current Memory Usage: {mem:.1f} MB", "DEBUG")
        else:
            _log("Memory monitoring unavailable (psutil not installed).", "DEBUG")

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

        row_count = int(df_clean.shape[0])
        col_count = int(df_clean.shape[1])
        
        dpa.clear()

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

        # Cache split sizes for reporting before clearing agents
        train_size = int(msa.X_train_.shape[0]) if hasattr(msa, 'X_train_') and msa.X_train_ is not None else 0
        test_size  = int(msa.X_test_.shape[0]) if hasattr(msa, 'X_test_') and msa.X_test_ is not None else 0

        # Release transformed data copies inside agent (keep msa until after regularization)
        ta.clear()

        # ── 5. Regularization ──────────────────────────────────────────────────
        skip_reg = os.getenv("SKIP_REGULARIZATION", "False").lower() == "true"
        reg_result = {} # Default
        
        if skip_reg:
            _log("RegularizationAgent: Bypassed via .env (Idle)", "INFO")
            reg_result = {
                "best_method": f"None (Skipped)",
                "best_roc_auc": float(model_result['detailed_results'][model_result['best_model_name']]['ROC_AUC']),
                "all_results": {}, "coef_summary": pd.DataFrame()
            }
        else:
            try:
                _log("RegularizationAgent: fitting Ridge/LASSO/ElasticNet …")
                _set_agent("RegularizationAgent", "running")
                from agents.regularization_agent import RegularizationAgent
                ra = RegularizationAgent()
                reg_result = ra.run(
                    msa.X_train_, 
                    msa.y_train_,
                    msa.X_test_,
                    msa.y_test_,
                    feature_names=msa.feature_names_
                )
                _set_agent("RegularizationAgent", "done")
            except Exception as e:
                _log(f"RegularizationAgent failed: {e}", "WARNING")
                _set_agent("RegularizationAgent", "error")
                reg_result = {"best_method": "Regression Error", "best_roc_auc": 0.0, "all_results": {}, "coef_summary": pd.DataFrame()}
        
        # Now we can safely release the model selection data matrices
        msa.clear()
        _set_progress(95)

        # ── 6. Intelligent Enrichment (Phase 2) ───────────────────────────────
        _log("ExplorerAgent: Searching for real-time city updates …")
        try:
            _set_agent("ExplorerAgent", "running")
            from agents.explorer_agent import ExplorerAgent
            explorer = ExplorerAgent()
            
            _log(f"ExplorerAgent: Researching {target_service}...")

            _log(f"ExplorerAgent: Researching {target_service}...")
            explorer_res = explorer.run(target_service)
            _set_agent("ExplorerAgent", "done")
            _log(f"ExplorerAgent: Research complete and pushed to Neo4j.", "DONE")

            # Memory Release after last use of df_clean
            del df_clean
            gc.collect()
            _log("Clean data released after enrichment.", "DEBUG")

            # ── 7. Truth Reconciliation (Phase 3) ─────────────────────────────
            _log("ReinforcementJudge: Vetting new insights against Audit data …")
            _set_agent("ReinforcementJudge", "running")
            from agents.judge_agent import ReinforcementJudge
            judge = ReinforcementJudge()
            judgment = judge.run(target_service)
            _set_agent("ReinforcementJudge", "done")
            _log(f"ReinforcementJudge: Case {judgment.get('classification')} (Score: {judgment.get('trust_score')})", "DONE")
        except Exception as e:
            _set_agent("ExplorerAgent", "error")
            _set_agent("ReinforcementJudge", "error")
            _log(f"Intelligent agents skipped or failed: {e}", "WARNING")
        
        # --- ALL AGENTS COMPLETE ---
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
        
        # Memory Release after last use of df_transformed
        del df_transformed
        gc.collect()
        _log("Transformed data released before payload build.", "DEBUG")
        
        dept_name = last_case.get("department") if last_case else "Dallas 311"
        
        try:
            domain_context = get_domain_context(department=dept_name)
        except Exception:
            domain_context = "Standard Dallas 311 procedures apply."

        metrics = {
            "records": {
                "label": "Records Processed",
                "value": row_count,
                "delta": 0,
                "reasoning": "Team Wisdom: Click to see how this affects our sanitation routes."
            },
            "features": {
                "label": "Features Selected",
                "value": len(feat_imp) if feat_imp else col_count,
                "reasoning": "Team Wisdom: Click to see why these predictors matter to Dallas."
            },
            "accuracy": {
                "label": "Best ROC-AUC",
                "value": round(float(reg_result.get("best_roc_auc", 0.0)), 3),
                "reasoning": "Team Wisdom: Click to see if this model is city-ready."
            }
        }

        finished_at = datetime.now().isoformat()

        results = {
            "metrics":       metrics,
            "models":        [ { **v, "Model": k } for k, v in det_results.items() ],
            "detailed_results": det_results,
            "best_model":    model_result.get("best_model_name"),
            "diagnostics":   safe_diag,
            "diagnostics_reasoning": "Strategic Vetting: Click to ask Expert AI about data health.",
            "key_findings":  key_findings,
            "model_comparison_reasoning": "Strategic Vetting: Click to compare model architectures.",
            "confusion_matrix_reasoning": "Strategic Vetting: Click to see our error profile.",
            "graph_insights": {"roc": "Visual Analysis ready.", "pr": "Stability check ready."},
            "data_path":     data_path,
            "started_at":    _state["started_at"],
            "finished_at":   finished_at,
            "split_info":    split_info,
            "split_info_reasoning": "Strategic Vetting: Click to see training distributions.",
            "feature_importances": feat_imp,
            "features_reasoning": "Strategic Vetting: Click to see domain-specific predictors.",
            "regularization": {
                "best_method":  reg_result.get("best_method", "N/A"),
                "best_roc_auc": float(reg_result.get("best_roc_auc", 0.0)),
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
            "data_shape":  [row_count, col_count],
            "train_test":  [train_size, test_size],
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
            X, _, _ = handle_missing_values(X, X, initial_imputers=encoders.get("imputers"))
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
    import numpy as np

    # 1. Create initial DataFrame
    df = pd.DataFrame([row])

    # 2. Schema Alignment (Stage B: Robust Feature Handling)
    # Ensure all features the model expects are present (even if NaN/None)
    if "original_features" in encoders:
        # Fallback for stale artifacts: if numeric_cols missing, try to infer from df or defaults
        num_cols = encoders.get("numeric_cols", [])
        if not num_cols:
            # Heuristic: anything not in cat_cols or config labels is likely numeric
            from src.config import LABEL_ENCODE_COLUMNS
            num_cols = [c for c in encoders["original_features"] if c not in encoders.get("cat_cols", []) and c not in LABEL_ENCODE_COLUMNS]

        for col in encoders["original_features"]:
            if col not in df.columns:
                # Use np.nan for numeric, None for object/categorical to prevent isnan type errors
                df[col] = np.nan if col in num_cols else None
        
        # Ensure correct column order and DROP any extra columns not in training
        df = df[encoders["original_features"]]

    # 3. Extract time features if Created Date is present
    if "Created Date" in df.columns and pd.notnull(df["Created Date"].iloc[0]):
        df["Created Date"] = pd.to_datetime(df["Created Date"])
        df = add_time_features(df)
    
    # Always drop Created Date as it's high-cardinality leakage
    df = df.drop(columns=["Created Date"], errors="ignore")

    # 4. Clean ERT
    df = clean_ert(df)

    # 5. Stateful Imputation
    imputers = encoders.get("imputers", {})
    # We pass the same DF twice as X_train/X_test for the signature
    df, _, _ = handle_missing_values(df, df, initial_imputers=imputers)

    # 6. Service Request Type grouping
    # In manual mode, we just ensure it's matched against encoders during OHE
    
    # 7. Encoding & Scaling
    # We use encode_categoricals in inference mode (initial_encoders=encoders)
    X_final, _, _ = encode_categoricals(df, df, initial_encoders=encoders)
    
    # Apply Scaling
    if "scaler" in encoders:
        scaler = encoders["scaler"]
        # Ensure columns match scaler expectations (add missing OHE columns as 0)
        for col in scaler.feature_names_in_:
            if col not in X_final.columns:
                X_final[col] = 0
        
        # Reorder to match scaler
        X_final = X_final[scaler.feature_names_in_]
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
            "message": "No results yet. Run the pipeline to see AI intelligence.",
            "status": status,
            "metrics": {},
            "models": []
        }), 200

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
    """Run model inference on a single manually entered row across 4 time thresholds."""
    
    xgb_fallback_path = ARTIFACTS_DIR / "artifacts" / "xgb_model_500k.json"
    feat_path = ARTIFACTS_DIR / "artifacts" / "feature_names.pkl"
    enc_path = ARTIFACTS_DIR / "artifacts" / "label_encoders.pkl"
    
    if not feat_path.exists() and not xgb_fallback_path.exists():
        # Baseline fallback
        m_path = ARTIFACTS_DIR / "best_model.joblib"
        e_path = ARTIFACTS_DIR / "encoders.joblib"

        if not m_path.exists() or not e_path.exists():
            return jsonify({"error": "Model artifacts not found. Run pipeline first."}), 400

        try:
            row = request.get_json()
            if not row:
                return jsonify({"error": "No data provided"}), 400

            import joblib
            model = joblib.load(m_path)
            encoders = joblib.load(e_path)
            X_final = _preprocess_manual_row(row, encoders)

            prediction_prob = float(model.predict_proba(X_final)[0][1])
            prediction_class = int(model.predict(X_final)[0])
            
            chain = create_explainability_chain()
            explanation = chain.invoke({
                "prediction": f"{'Fast' if prediction_class == 1 else 'Slow'} Close (Probability: {prediction_prob:.2%})",
                "coef_summary": "Legacy 500k Model used.",
                "department": row.get("Department"),
                "district": row.get("City Council District")
            })

            return jsonify({
                "prediction": prediction_class,
                "probability": prediction_prob,
                "predictions": {"Baseline": {"class": prediction_class, "probability": prediction_prob}},
                "explanation": explanation,
                "row": row
            })

        except Exception as e:
            logger.exception("Baseline manual inference failed")
            return jsonify({"error": str(e)}), 500


    import xgboost as xgb
    import joblib
    import pandas as pd
    import numpy as np

    try:
        row = request.get_json()
        if not row:
            return jsonify({"error": "No data provided"}), 400

        # 1. Load Preprocessors (Global)
        feature_names = joblib.load(feat_path) if feat_path.exists() else []
        
        if enc_path.exists():
            label_encoders = joblib.load(enc_path)
        else:
            e_path = ARTIFACTS_DIR / "encoders.joblib"
            raw_enc = joblib.load(e_path)
            label_encoders = {k: v for k, v in raw_enc.items() if hasattr(v, 'classes_')}

        # 2. Preprocess row
        df = pd.DataFrame([row])

        if "Created Date" in df.columns and pd.notnull(df["Created Date"].iloc[0]):
            df["Created Date"] = pd.to_datetime(df["Created Date"])
            df["hour"] = df["Created Date"].dt.hour
            df["day_of_week"] = df["Created Date"].dt.dayofweek
            df["month"] = df["Created Date"].dt.month
        else:
            now = pd.Timestamp.now()
            df["hour"] = now.hour
            df["day_of_week"] = now.dayofweek
            df["month"] = now.month

        if 'Estimated Response Time Description' in df.columns and pd.notnull(df['Estimated Response Time Description'].iloc[0]):
            df['ERT_days'] = pd.to_numeric(df['Estimated Response Time Description'].str.extract(r'(\d+)')[0], errors='coerce')
        else:
            df['ERT_days'] = np.nan

        for col in ['Service Request Type', 'Priority', 'Method Received Description', 'Department']:
            if col in df.columns:
                df[col] = df[col].astype(str)

        for col, le in label_encoders.items():
            if col == 'Department_grouped' and 'Department' in df.columns:
                val = df['Department'].iloc[0]
                if val in le.classes_:
                    df[col] = le.transform([val])[0]
                else:
                    df[col] = le.transform(['Other'])[0] if 'Other' in le.classes_ else 0
            elif col in df.columns:
                val = df[col].iloc[0]
                if val in le.classes_:
                    df[col] = le.transform([val])[0]
                else:
                    df[col] = 0

        # 3. Multi-Threshold Inference
        thresholds = [24, 48, 72, 96]
        threshold_predictions = {}
        
        # If no specific threshold models exist, fallback to 500k singular
        has_multi = (ARTIFACTS_DIR / "artifacts" / "xgb_model_72h.json").exists()
        target_thresholds = thresholds if has_multi else ["500k"]

        coef_summary = ""
        overall_prob = 0.0
        overall_class = 0

        for hr in target_thresholds:
            # Reconstruct raw features
            if len(feature_names) > 0:
                X_raw = pd.DataFrame(0, index=[0], columns=feature_names)
                for col in df.columns:
                    if col in feature_names:
                        X_raw[col] = df[col].iloc[0]
                for col in df.select_dtypes(include=['object', 'category']).columns:
                    val = df[col].iloc[0]
                    ohe_col = f"{col}_{val}"
                    if ohe_col in feature_names:
                        X_raw[ohe_col] = 1
            else:
                X_raw = df.copy()

            # Load specific scalers
            imp_path = ARTIFACTS_DIR / "artifacts" / f"imputer_{hr}h.pkl"
            scale_path = ARTIFACTS_DIR / "artifacts" / f"scaler_{hr}h.pkl"
            if not imp_path.exists(): imp_path = ARTIFACTS_DIR / "artifacts" / "imputer.pkl"
            if not scale_path.exists(): scale_path = ARTIFACTS_DIR / "artifacts" / "scaler.pkl"

            if imp_path.exists() and scale_path.exists():
                imputer = joblib.load(imp_path)
                scaler = joblib.load(scale_path)
                
                # Coerce any remaining unencoded strings (like missing label encoders) to NaN
                # so the imputer can safely fill them with the median.
                for col in X_raw.columns:
                    X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
                        
                try:
                    X_final = pd.DataFrame(imputer.transform(X_raw), columns=feature_names)
                except Exception as imputer_error:
                    logger.warning(f"Pickled imputer failed: {imputer_error}. Using fillna(0) as fallback.")
                    X_final = X_raw.fillna(0)

                try:
                    X_final = pd.DataFrame(scaler.transform(X_final), columns=feature_names)
                except Exception as scaler_error:
                    logger.warning(f"Pickled scaler failed: {scaler_error}. Proceeding without scaling.")
            else:
                X_final = X_raw

            xgb_path = ARTIFACTS_DIR / "artifacts" / f"xgb_model_{hr}h.json"
            rf_path = ARTIFACTS_DIR / "artifacts" / f"random_forest_{hr}h.joblib"
            lr_path = ARTIFACTS_DIR / "artifacts" / f"logistic_regression_{hr}h.joblib"

            if not xgb_path.exists() and hr != "500k":
                xgb_path = ARTIFACTS_DIR / "artifacts" / f"xgb_model_{hr}.json" # fallback naming

            if not xgb_path.exists():
                continue

            model = xgb.XGBClassifier()
            model.load_model(str(xgb_path))

            preds = {}
            xgb_prob = float(model.predict_proba(X_final)[0][1])
            xgb_class = int(model.predict(X_final)[0])
            preds["XGBoost"] = {"class": xgb_class, "probability": xgb_prob}

            total_prob = xgb_prob
            model_count = 1

            if rf_path.exists():
                try:
                    rf_model = joblib.load(str(rf_path))
                    rf_prob = float(rf_model.predict_proba(X_final)[0][1])
                    preds["Random Forest"] = {"class": int(rf_model.predict(X_final)[0]), "probability": rf_prob}
                    total_prob += rf_prob
                    model_count += 1
                except Exception as e:
                    logger.warning(f"RF {hr}h missing: {e}")

            if lr_path.exists():
                try:
                    lr_model = joblib.load(str(lr_path))
                    if hasattr(lr_model, "predict_proba"):
                        lr_prob = float(lr_model.predict_proba(X_final)[0][1])
                    else:
                        lr_prob = float(lr_model.predict(X_final)[0])
                    preds["Logistic Regression"] = {"class": int(lr_model.predict(X_final)[0]), "probability": lr_prob}
                    total_prob += lr_prob
                    model_count += 1
                except Exception as e:
                    logger.warning(f"LR {hr}h missing: {e}")

            ensemble_prob = total_prob / model_count
            threshold_predictions[f"{hr}h" if hr != "500k" else "500k"] = {
                "ensemble_probability": ensemble_prob,
                "models": preds
            }
            
            if hr == 72 or hr == "500k":
                overall_prob = ensemble_prob
                overall_class = 1 if ensemble_prob >= 0.5 else 0
                if hasattr(model, 'feature_importances_') and len(feature_names) > 0:
                    imps = model.feature_importances_
                    top_idx = np.argsort(imps)[-5:][::-1]
                    coef_summary = "\n".join([f"- {feature_names[i]}: {imps[i]:.4f} (Importance)" for i in top_idx])

        # 4. Explain
        chain = create_explainability_chain()
        
        timeline_str = "\n".join([f"{k}: {v['ensemble_probability']:.2%} probability of resolution" for k, v in threshold_predictions.items()])
        
        explanation = chain.invoke({
            "prediction": f"Multi-Threshold Resolution Prediction:\n{timeline_str}",
            "coef_summary": coef_summary,
            "department": row.get("Department"),
            "district": row.get("City Council District")
        })

        return jsonify({
            "prediction": overall_class,
            "probability": overall_prob,
            "predictions": threshold_predictions.get("500k", {}).get("models", {}), # legacy support
            "threshold_predictions": threshold_predictions,
            "explanation": explanation,
            "row": row
        })

    except Exception as e:
        logger.exception("Manual inference with multi-threshold models failed")
        return jsonify({"error": f"Model error: {str(e)}"}), 500
@app.route('/api/acoustic_extraction_v2', methods=['POST'])
def acoustic_extract():
    """Stage A: Transcription & Semantic Extraction (V2.0)."""
    try:
        from agents.transcription_agent import TranscriptionAgent
        data = request.json
        transcript = data.get("transcript", "")
        
        if not transcript:
            return jsonify({"error": "No transcript provided"}), 400
            
        agent = TranscriptionAgent()
        extracted_features = agent.run(transcript)
        
        # LOGGING for your terminal
        print(f"[STAGE A] AI Extraction result: {extracted_features}")
        
        return jsonify(extracted_features)
    except Exception as e:
        print(f"[STAGE A ERROR] {e}")
        return jsonify({"debug_error": str(e)}), 500


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using Groq Whisper API (free tier, uses existing GROQ_API_KEY)."""
    try:
        from groq import Groq as GroqClient
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        filename = audio_file.filename or "recording.webm"

        client = GroqClient(api_key=os.getenv("GROQ_API_KEY"))
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en",
            temperature=0.0
        )

        return jsonify({"transcription": transcription.text.strip()})

    except Exception as e:
        logger.exception("Transcription failed")
        return jsonify({"error": str(e)}), 500


@app.route('/api/tts', methods=['POST'])
def api_tts():
    """Synthesize text using Edge-TTS (free, high-quality, unlimited)."""
    data = request.json or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    try:
        import asyncio
        import edge_tts
        import tempfile
        import os
        from flask import Response
        
        async def generate_edge_audio(text_in):
            communicate = edge_tts.Communicate(text_in, "en-US-GuyNeural")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                edge_tmp_path = tmp.name
            await communicate.save(edge_tmp_path)
            return edge_tmp_path

        edge_path = asyncio.run(generate_edge_audio(text))
        
        try:
            with open(edge_path, 'rb') as f:
                audio_bytes = f.read()
        finally:
            if os.path.exists(edge_path):
                os.remove(edge_path)
                
        return Response(audio_bytes, mimetype="audio/mpeg")
        
    except Exception as e:
        logger.exception(f"Edge-TTS synthesis failed: {e}")
        return jsonify({"error": f"TTS generation error: {e}"}), 500


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


@app.route("/api/chat", methods=["POST"])
def chat():
    """Real-time RAG Chat endpoint with Neo4j memory."""
    data = request.json
    user_query = data.get("message")
    session_id = data.get("session_id", "default_session")
    dept = data.get("department") # Optional context
    
    if not user_query:
        return jsonify({"error": "No message provided"}), 400

    try:
        memory = Neo4jChatMemory(session_id)

        # 1. Retrieve Hybrid Context (PDFs + Web Insights)
        facts = get_domain_context(department=dept, user_query=user_query)
        logger.info(f"[Chat] Fact-Grounding complete. Context size: {len(facts)} chars.")
        
        # 1b. Inject Live Dashboard Telemetry (Ground truth for CURRENT numbers)
        with _lock:
            live_results = _state.get("results")
            if not live_results:
                live_results = _load_persisted_results()
        
        telemetry = "No live telemetry available."
        if live_results:
            m = live_results.get("metrics", {})
            d = live_results.get("diagnostics", {})
            telemetry = f"""
            - DATA VOLUME: {m.get('records', {}).get('value')} total records.
            - ACCURACY (ROC-AUC): {m.get('accuracy', {}).get('value')}
            - FEATURES: {m.get('features', {}).get('value')} predictors utilized.
            - DIAGNOSTICS: Normality={d.get('is_normal')}, Balance={not d.get('is_imbalanced')}
            - ALGORITHM: {live_results.get('best_model', 'N/A')}
            """
        
        # 2. Retrieve Conversation History from Graph
        history = memory.get_history(limit=5)
        
        # 3. Use LLM to generate response
        llm = get_llm(temperature=0.7)
        from langchain_core.prompts import ChatPromptTemplate
        chat_prompt = ChatPromptTemplate.from_template("""
            System: You are an academic researcher analyzing the Dallas 311 project. 
            Rule: Speak like a normal human using simple, plain English. ABSOLUTELY AVOID consultant jargon, fluff, and overly complex terminology (e.g., NEVER use words like leverage, methodology, synergy, optimization, strategic alignment, utilize, or pinpoint).
            
            PROJECT METRICS (Live Dashboard Numbers):
            {telemetry}
            
            BACKGROUND DATA (Neo4j Graph & PDF Insights):
            {facts}

            STRICT RULES:
            - Be extremely concise. Give your answer in 2 to 3 sentences maximum.
            - Get straight to the point. Answer the question based on the PROJECT METRICS or BACKGROUND DATA above.
            - Do not be overly literal or argumentative about Neo4j source locations if the answer is clear in the telemetry.
            - If you cannot find the necessary information in either BACKGROUND DATA or PROJECT METRICS, output ONLY the exact word "NEED_EXPLORER". Do not say anything else.
            
            HUMAN QUERY: {query}
            RESPONSE:
        """)

        
        # Triple-Vault Fallback Logic
        answer = None
        try:
            # 1. Primary: Try Groq
            logger.info("[Chat] Attempting Primary Reasoning (Groq)")
            llm = get_llm(provider="groq", temperature=0.7)
            from langchain_core.output_parsers import StrOutputParser
            chain = chat_prompt | llm | StrOutputParser()
            answer = chain.invoke({"facts": facts, "history": history, "query": user_query, "telemetry": telemetry})
        except Exception as groq_err:
            logger.warning(f"[Chat] Groq unavailable, trying Secondary (Vertex AI): {groq_err}")
            try:
                # 2. Secondary: Try Google Gemini (AI Studio)
                if os.getenv("GOOGLE_API_KEY"):
                    llm = get_llm(provider="vertexai", temperature=0.7)
                    from langchain_core.output_parsers import StrOutputParser
                    chain = chat_prompt | llm | StrOutputParser()
                    answer = chain.invoke({"facts": facts, "history": history, "query": user_query, "telemetry": telemetry})
                else:
                    raise ValueError("No Google API Key found for Gemini fallback.")
            except Exception as vertex_err:
                logger.error(f"[Chat] All AI Providers failed: {vertex_err}")
                # 3. Final Fallback: Direct Knowledge Graph Facts
                answer = f"⚠️ **Service Note: Multi-Agent Cloud is currently busy.**\n\nI have retrieved the official records from our **Dallas Neo4j Knowledge Graph** for you:\n\n{facts}\n\n*Strategic AI reasoning will resume shortly.*"

        # ── 3.5 Escape Hatch: LLM requests Explorer Agent ────────────────────
        if answer and "NEED_EXPLORER" in answer:
            logger.info("[Chat] LLM requested Explorer Agent. Running deep research...")
            try:
                from agents.explorer_agent import ExplorerAgent
                explorer = ExplorerAgent()
                target = dept if dept else user_query
                explorer.run(target)
                
                # Fetch refreshed context
                facts = get_domain_context(department=dept, user_query=user_query)
                logger.info(f"[Chat] Fact-Grounding refreshed. New context size: {len(facts)} chars.")
                
                # Re-prompt without the NEED_EXPLORER escape rule
                second_prompt = ChatPromptTemplate.from_template("""
                    System: You are an academic researcher analyzing the Dallas 311 project.
                    Rule: Speak like a normal human using simple, plain English. ABSOLUTELY AVOID consultant jargon, fluff, and overly complex terminology (e.g., NEVER use words like leverage, methodology, synergy, optimization, strategic alignment, utilize, or pinpoint).
                    
                    PROJECT METRICS:
                    {telemetry}
                    
                    BACKGROUND DATA:
                    {facts}
                    
                    STRICT RULES:
                    - Be extremely concise. Give your answer in 2 to 3 sentences maximum.
                    - Do not write a long introduction, do not restate the question, and do not repeat yourself. Get straight to the point.
                    
                    HUMAN QUERY: {query}
                    RESPONSE:
                """)
                llm = get_llm(provider="groq", temperature=0.7)
                from langchain_core.output_parsers import StrOutputParser
                chain = second_prompt | llm | StrOutputParser()
                answer = chain.invoke({"facts": facts, "history": history, "query": user_query, "telemetry": telemetry})
            except Exception as exp_err:
                logger.error(f"[Chat] Explorer Agent invocation failed: {exp_err}")

        # 4. Save entire transaction back to Neo4j (if we have a result)

        # 4. Save entire transaction back to Neo4j (if we have a result)
        if answer:
            memory.add_message("human", user_query)
            memory.add_message("ai", answer)
        memory.close()

        return jsonify({
            "answer": answer,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[Chat] Fatal Orchestration Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    data = request.json or {}
    email = data.get('email')
    password = data.get('password')
    app_password = data.get('app_password', '')
    
    if not email or not password:
        return jsonify({"success": False, "error": "Email and Password are required."}), 400

    hashed = generate_password_hash(password)
    
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("INSERT INTO admins (email, password_hash, gmail_app_password) VALUES (?, ?, ?)", 
                       (email, hashed, app_password))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Admin registered successfully."})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "error": "This admin email is already registered."}), 400

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.json or {}
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"success": False, "error": "Email and Password are required."}), 400
        
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash, gmail_app_password FROM admins WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return jsonify({"success": False, "error": "Invalid credentials or user not registered."}), 401
        
    hashed_pw, app_pw = row
    if not check_password_hash(hashed_pw, password):
        return jsonify({"success": False, "error": "Invalid credentials."}), 401

    session['admin_email'] = email
    session['admin_password'] = app_pw
    return jsonify({"success": True, "message": "Logged in successfully.", "user": email})

@app.route('/api/auth/logout', methods=['POST', 'GET'])
def api_logout():
    session.pop('admin_email', None)
    session.pop('admin_password', None)
    return jsonify({"success": True, "message": "Logged out successfully."})

@app.route('/api/auth/status', methods=['GET'])
def api_auth_status():
    if 'admin_email' in session:
        return jsonify({"logged_in": True, "user": session['admin_email']})
    return jsonify({"logged_in": False})

@app.route('/api/send_email_report', methods=['POST'])
def api_send_report():
    if 'admin_email' not in session or 'admin_password' not in session:
        return jsonify({"success": False, "error": "Authentication required. Please log in first."}), 401
        
    data = request.json or {}
    recipient = data.get('recipient_email')
    
    if not recipient:
        return jsonify({"success": False, "error": "Recipient email is required."}), 400
        
    with _lock:
        live_results = _state.get("results")
        if not live_results:
            live_results = _load_persisted_results()
            
    if not live_results:
        return jsonify({"success": False, "error": "No performance data available to email."}), 400
        
    resend_api_key = os.getenv("RESEND_API_KEY")
    resend_sender = os.getenv("RESEND_FROM_EMAIL") or os.getenv("MAIL_DEFAULT_SENDER")
    if resend_api_key in {"", "your_resend_api_key", "re_xxxxxxxxx"}:
        resend_api_key = None

    if resend_api_key:
        if not resend_sender:
            return jsonify({"success": False, "error": "Resend sender not configured. Please set RESEND_FROM_EMAIL in .env."}), 400

        from agents.email_agent import send_resend_report
        success, error_message = send_resend_report(
            api_key=resend_api_key,
            sender_email=resend_sender,
            recipient_email=recipient,
            report_data=live_results
        )
    else:
        # Prioritize session credentials if logged in, fallback to .env
        if session.get('admin_email') and session.get('admin_password'):
            sender_email = session.get('admin_email')
            app_password = session.get('admin_password')
        else:
            sender_email = os.getenv("MAIL_DEFAULT_SENDER")
            app_password = os.getenv("MAIL_APP_PASSWORD")

        if not sender_email or not app_password:
            return jsonify({"success": False, "error": "Email sender not configured. Please set RESEND_API_KEY and RESEND_FROM_EMAIL, or configure MAIL_DEFAULT_SENDER and MAIL_APP_PASSWORD for Gmail SMTP."}), 400

        from agents.email_agent import send_gmail_report
        success, error_message = send_gmail_report(
            sender_email=sender_email,
            app_password=app_password,
            recipient_email=recipient,
            report_data=live_results
        )
    
    if success:
        return jsonify({"success": True, "message": f"Report sent securely to {recipient}."})
    error_detail = error_message or "Check your credentials/App Password."
    return jsonify({"success": False, "error": f"Failed to send email. {error_detail}"}), 500

@app.route('/favicon.png')
def serve_favicon():
    """Route to explicitly serve the dashboard favicon."""
    import os
    from flask import send_from_directory
    return send_from_directory('.', 'favicon.png')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dallas 311 Dashboard — Flask Server")
    parser.add_argument("--port",  type=int, default=5000,  help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true",      help="Enable Flask debug mode")
    args = parser.parse_args()

    print(f"  Press Ctrl+C to stop.\n")

    import atexit
    import signal

    def graceful_exit():
        # Prevent 'could not acquire lock' errors on Windows reloads
        import sys
        import gc
        sys.stdout.flush()
        sys.stderr.flush()
        gc.collect() 
        
    atexit.register(graceful_exit)

    try:
        app.run(host="0.0.0.0", port=args.port, debug=args.debug, use_reloader=False)
    except (SystemExit, KeyboardInterrupt):
        pass
    except Exception as e:
        print(f"[Fatal] Server Error: {e}")
