# ============================================
# Streamlit Web App — Dallas 311 ML Pipeline
# Phase 1 UI
# ============================================
"""
Launch with:
    streamlit run ui/app.py
"""
import sys
from pathlib import Path

# Make the project root importable regardless of where streamlit is invoked from
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging
import time

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src import config

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dallas 311 ML Pipeline",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom style ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background: #0f1117; }
    .agent-badge-ok  { background:#1a9c3e; color:#fff; padding:4px 10px;
                       border-radius:6px; font-size:0.82rem; }
    .agent-badge-err { background:#c0392b; color:#fff; padding:4px 10px;
                       border-radius:6px; font-size:0.82rem; }
    .agent-badge-run { background:#e67e22; color:#fff; padding:4px 10px;
                       border-radius:6px; font-size:0.82rem; }
    .wrap-text       { white-space: pre-wrap; word-wrap: break-word; font-family: monospace;
                       background: #1e1e1e; padding: 10px; border-radius: 5px; color: #d4d4d4; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏙️ Dallas 311")
    st.caption("Agentic ML Pipeline — Phase 1")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload 311 CSV",
        type=["csv"],
        help="Upload your 311 service-requests CSV file.",
    )
    run_btn = st.button(
        "🚀  Run Full Pipeline",
        use_container_width=True,
        type="primary",
        disabled=uploaded_file is None,
    )
    if uploaded_file is None:
        st.caption("⬆️ Upload a CSV to enable the pipeline.")
    st.divider()
    st.markdown(
        "**Agents**\n"
        "1. DataPrep\n"
        "2. Transformation\n"
        "3. Diagnostics\n"
        "4. ModelSelection\n"
        "5. Regularization"
    )

# ── Session state defaults ─────────────────────────────────────────────────────
for key in ("pipeline_ran", "agent_status", "diag_report",
            "model_report", "reg_report", "msa"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_run, tab_results, tab_diag = st.tabs(
    ["🚀 Run Pipeline", "📊 Results", "🔬 Diagnostics"]
)

# ==============================================================================
# TAB 1 — RUN PIPELINE
# ==============================================================================
with tab_run:
    st.header("Pipeline Execution")

    agent_names = [
        "DataPrep Agent",
        "Transformation Agent",
        "Diagnostics Agent",
        "Model Selection Agent",
        "Regularization Agent",
    ]

    # Status placeholders
    placeholders = {}
    cols = st.columns(len(agent_names))
    for col, name in zip(cols, agent_names):
        with col:
            st.caption(name)
            placeholders[name] = st.empty()
            placeholders[name].markdown(
                '<span class="agent-badge-err">⬜ idle</span>',
                unsafe_allow_html=True,
            )

    log_box = st.empty()
    progress = st.progress(0)

    def badge(name: str, state: str) -> None:
        css = {"running": "agent-badge-run", "ok": "agent-badge-ok", "err": "agent-badge-err"}
        icon = {"running": "⏳", "ok": "✅", "err": "❌"}
        placeholders[name].markdown(
            f'<span class="{css[state]}">{icon[state]} {state}</span>',
            unsafe_allow_html=True,
        )

    if run_btn:
        logs: list[str] = []
        logging.basicConfig(level=logging.INFO)

        try:
            # ── 1. DataPrep ──────────────────────────────────────────────────
            badge("DataPrep Agent", "running")
            progress.progress(5)
            from agents.data_prep_agent import DataPrepAgent
            dpa = DataPrepAgent(data_path=uploaded_file)   # file-like object
            with st.spinner("DataPrep Agent running …"):
                df_clean = dpa.run()
            val = dpa.validate()
            if not val["passed"]:
                raise RuntimeError(f"DataPrep validation failed: {val['issues']}")
            badge("DataPrep Agent", "ok")
            logs.append(f"✅ DataPrep done — shape {df_clean.shape}")
            progress.progress(20)

            # ── 2. Transformation ────────────────────────────────────────────
            badge("Transformation Agent", "running")
            from agents.transformation_agent import TransformationAgent
            ta = TransformationAgent()
            with st.spinner("Transformation Agent running …"):
                df_transformed = ta.run(df_clean)
            val = ta.validate()
            if not val["passed"]:
                raise RuntimeError(f"Transformation validation failed: {val['issues']}")
            badge("Transformation Agent", "ok")
            logs.append(f"✅ Transformation done — shape {df_transformed.shape}")
            progress.progress(40)

            # ── 3. Diagnostics ───────────────────────────────────────────────
            badge("Diagnostics Agent", "running")
            from agents.diagnostics_agent import DiagnosticsAgent
            da = DiagnosticsAgent()
            with st.spinner("Diagnostics Agent running …"):
                diag = da.run(df_transformed)
            badge("Diagnostics Agent", "ok")
            st.session_state["diag_report"] = diag
            logs.append(f"✅ Diagnostics done — overall pass: {diag['overall_pass']}")
            progress.progress(55)

            # ── 4. Model Selection ───────────────────────────────────────────
            badge("Model Selection Agent", "running")
            from agents.model_selection_agent import ModelSelectionAgent
            msa = ModelSelectionAgent()
            with st.spinner("Model Selection Agent running (training 3 models) …"):
                model_result = msa.run(df_transformed)
            val = msa.validate()
            if not val["passed"]:
                raise RuntimeError(f"ModelSelection validation failed: {val['issues']}")
            badge("Model Selection Agent", "ok")
            st.session_state["model_report"] = model_result
            st.session_state["msa"] = msa
            logs.append(f"✅ ModelSelection done — best: {model_result['best_model_name']}")
            progress.progress(80)

            # ── 5. Regularization ────────────────────────────────────────────
            badge("Regularization Agent", "running")
            from agents.regularization_agent import RegularizationAgent
            ra = RegularizationAgent()
            with st.spinner("Regularization Agent running (Ridge / LASSO / ElasticNet) …"):
                reg_result = ra.run(
                    msa.X_train_,
                    msa.y_train_,
                    msa.X_test_,
                    msa.y_test_,
                    feature_names=msa.feature_names_,
                )
            badge("Regularization Agent", "ok")
            st.session_state["reg_report"] = reg_result
            logs.append(f"✅ Regularization done — best: {reg_result['best_method']}")
            progress.progress(100)

            st.session_state["pipeline_ran"] = True
            st.success("🎉 Pipeline complete! See the Results and Diagnostics tabs.")

        except Exception as exc:  # noqa: BLE001
            for name in agent_names:
                badge_val = placeholders[name]._provided_value
                if badge_val and "running" in str(badge_val):
                    badge(name, "err")
            # ── Stacked vertical error display ────────────────────────
            st.error("### ❌ Pipeline Error")
            with st.container(border=True):
                st.markdown("**Error message:**")
                st.markdown(f'<div class="wrap-text">{exc}</div>', unsafe_allow_html=True)
                with st.expander("🔍 Full traceback", expanded=False):
                    st.exception(exc)
            logs.append(f"❌ Error: {exc}")

        st.markdown(f'<div class="wrap-text">{"<br>".join(logs)}</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 2 — RESULTS
# ==============================================================================
with tab_results:
    st.header("Model Comparison Results")

    if not st.session_state.get("pipeline_ran"):
        st.info("Run the pipeline first (sidebar → 🚀 Run Full Pipeline).")
    else:
        model_report = st.session_state["model_report"]

        # ── ROC-AUC bar chart ────────────────────────────────────────────────
        st.subheader("ROC-AUC by Model")
        comp_df = pd.DataFrame(model_report["comparison"])
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ["#2ecc71" if m == model_report["best_model_name"] else "#3498db"
                  for m in comp_df["Model"]]
        ax.barh(comp_df["Model"], comp_df["ROC_AUC"], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("ROC-AUC")
        ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, label="random baseline")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(comp_df.sort_values("ROC_AUC", ascending=False), use_container_width=True)

        st.caption(f"Best model: **{model_report['best_model_name']}** "
                   f"saved to `models/best_model.joblib`")

        # ── Feature Importances (Random Forest if available) ─────────────────
        msa = st.session_state["msa"]
        if msa and hasattr(msa.best_model_, "feature_importances_"):
            st.subheader("Top 10 Feature Importances")
            imp = pd.Series(
                msa.best_model_.feature_importances_,
                index=msa.feature_names_ or range(len(msa.best_model_.feature_importances_)),
            ).sort_values(ascending=False).head(10)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            imp.sort_values().plot(kind="barh", ax=ax2, color="#3498db")
            ax2.set_xlabel("Importance Score")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

# ==============================================================================
# TAB 3 — DIAGNOSTICS
# ==============================================================================
with tab_diag:
    st.header("Diagnostic Report")

    if not st.session_state.get("pipeline_ran"):
        st.info("Run the pipeline first (sidebar → 🚀 Run Full Pipeline).")
    else:
        diag = st.session_state["diag_report"]
        reg  = st.session_state["reg_report"]

        # ── Normality ────────────────────────────────────────────────────────
        st.subheader("Normality of `days_to_close`")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Normality pass", "✅ Yes" if diag.get("normality_pass") else "❌ No")
        col2.metric("Skewness",       diag.get("skewness", "N/A"))
        col3.metric("Log skewness",   diag.get("log_skewness", "N/A"))
        col4.metric("Log transform?",
                    "✅ Recommended" if diag.get("log_transform_recommended") else "Not needed")

        if diag.get("normality_p") is not None:
            st.caption(f"D'Agostino K² — stat={diag['normality_stat']}, p={diag['normality_p']}")

        # ── Class Imbalance ──────────────────────────────────────────────────
        st.subheader("Class Imbalance")
        col5, col6 = st.columns(2)
        col5.metric("Minority class fraction", diag.get("class_imbalance_ratio", "N/A"))
        col6.metric("Imbalance detected",
                    "⚠️ Yes" if diag.get("class_imbalance_detected") else "✅ No")

        if diag.get("target_distribution"):
            dist = pd.Series(diag["target_distribution"]).rename("count")
            st.bar_chart(dist)

        # ── Regularization ───────────────────────────────────────────────────
        if reg:
            st.subheader("Regularization Results")
            all_res = reg.get("all_results", {})
            reg_df = pd.DataFrame(
                [{"Method": m, "Best Alpha": r["best_alpha"], "ROC-AUC": r["roc_auc"]}
                 for m, r in all_res.items()]
            )
            st.dataframe(reg_df, use_container_width=True)
            st.caption(f"Best regularizer: **{reg['best_method']}** "
                       f"(ROC-AUC = {reg['best_roc_auc']})")

            coef_df = reg.get("coef_summary")
            if coef_df is not None and not coef_df.empty:
                st.subheader("Coefficient Summary (top 15 by |coefficient|)")
                top_coef = (
                    coef_df.assign(abs_coef=coef_df["Coefficient"].abs())
                    .sort_values("abs_coef", ascending=False)
                    .drop(columns="abs_coef")
                    .head(15)
                )
                st.dataframe(top_coef, use_container_width=True)
