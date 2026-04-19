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
    POST /api/reset     → reset to idle
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from http import HTTPStatus
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]   # project root
UI_DIR    = Path(__file__).resolve().parent        # ui/
UPLOAD_DIR = ROOT / "data" / "uploaded"
ARTIFACTS_DIR = ROOT / "models"
RESULTS_PATH = ARTIFACTS_DIR / "latest_results.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory, url_for
try:
    from flask_cors import CORS as _CORS
except ImportError:
    _CORS = None  # optional — not needed for same-origin requests

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(UI_DIR))
if _CORS:
    _CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

        # Diagnostics — strip any non-serialisable values
        safe_diag = {
            k: _make_json_safe(v)
            for k, v in diag.items()
            if not isinstance(v, (pd.DataFrame, np.ndarray))
        }

        finished_at = datetime.now().isoformat()

        results = {
            "models":        comparison,
            "best_model":    model_result.get("best_model_name"),
            "diagnostics":   safe_diag,
            "data_path":     data_path,
            "started_at":    _state["started_at"],
            "finished_at":   finished_at,
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
            "last_trained_case": _build_last_trained_case(df_transformed),
            "feature_importance": feat_imp,
            "data_shape":  [int(df_clean.shape[0]), int(df_clean.shape[1])],
            "train_test":  [int(msa.X_train_.shape[0]), int(msa.X_test_.shape[0])],
        }

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

    app.run(host="0.0.0.0", port=args.port, debug=args.debug, use_reloader=False)
