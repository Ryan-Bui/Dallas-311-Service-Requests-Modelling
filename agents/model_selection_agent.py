# ============================================
# Agent 4: Model Selection
# Phase 1.5 — Snippet Migration
# ============================================
"""
ModelSelectionAgent
-------------------
Supports two modes controlled by ``src.config.REGRESSION_MODE``:

  False (default) — Classification
      Trains Logistic Regression, Random Forest, XGBoost (classifiers).
      Selects the best model by ROC-AUC.

  True — Regression
      Trains Decision Tree, Random Forest, Gradient Boosting, XGBoost
      (regressors) to predict ``hours_to_close`` directly.
      Selects the best model by RMSE (lower is better).

The agent's public interface (run / validate / report) is identical in
both modes so the orchestrator does not need to change.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import (
    handle_missing_values,
    handle_service_request_type,
    encode_categoricals,
    split_features_target,
    split_train_test,
    scale_features,
)

# ── Classification trainers ───────────────────────────────────────────────────
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)

# ── Regression trainers ───────────────────────────────────────────────────────
from src.models import (
    train_decision_tree_regressor,
    train_random_forest_regressor,
    train_gradient_boosting_regressor,
    train_xgboost_regressor,
)

# ── Evaluation ────────────────────────────────────────────────────────────────
from src.evaluation import (
    evaluate_model,
    evaluate_regressor,
    compare_models,
    compare_regressors,
)

from src import config
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class ModelSelectionAgent(BaseAgent):
    """Train, rank, and persist the best ML model.

    Set ``src.config.REGRESSION_MODE = True`` to switch from
    classification (binary target) to regression (continuous hours_to_close).
    """

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.regression_mode_: bool = config.REGRESSION_MODE

        # Populated after run()
        self.best_model_      = None
        self.best_model_name_: str | None = None
        self.detailed_results_: dict | None = None
        self.comparison_df_:   pd.DataFrame | None = None
        self.encoders_: dict | None = None
        self.X_train_: pd.DataFrame | None = None
        self.y_train_  = None
        self.X_test_: pd.DataFrame | None = None
        self.y_test_   = None
        self.feature_names_: list[str] | None = None

    # ── run ───────────────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> dict:
        """Encode, split, train all models, evaluate, and save the best.

        Parameters
        ----------
        df : pd.DataFrame
            Transformed DataFrame from TransformationAgent.
            Must contain a ``target`` column (binary for classification,
            continuous for regression).

        Returns
        -------
        dict with keys:
            best_model_name, detailed_results, split_info,
            model_path, encoders_path, test_set_path,
            mode ('classification' | 'regression')
        """
        mode = "regression" if self.regression_mode_ else "classification"
        logger.info("[ModelSelectionAgent] Mode: %s", mode)

        # ── Preprocessing (identical for both modes) ──────────────────────────
        logger.info("[ModelSelectionAgent] Splitting features / target …")
        X, y = split_features_target(df)
        self.feature_names_ = X.columns.tolist()

        logger.info("[ModelSelectionAgent] Train/test split …")
        # Regression uses a plain (non-stratified) split because the target
        # is continuous — stratify= only works for classification targets.
        X_train, X_test, y_train, y_test = split_train_test(
            X, y,
            stratify=not self.regression_mode_,
        )

        logger.info("[ModelSelectionAgent] Imputing missing values …")
        X_train, X_test = handle_missing_values(X_train, X_test)

        logger.info("[ModelSelectionAgent] Handling service request types …")
        X_train, X_test = handle_service_request_type(X_train, X_test)

        logger.info("[ModelSelectionAgent] Encoding categoricals …")
        X_train, X_test, encoders = encode_categoricals(X_train, X_test)

        logger.info("[ModelSelectionAgent] Standardizing features …")
        X_train, X_test, scaler = scale_features(X_train, X_test)
        encoders["scaler"] = scaler

        self.encoders_      = encoders
        self.X_train_       = X_train
        self.y_train_       = y_train
        self.X_test_        = X_test
        self.y_test_        = y_test
        self.feature_names_ = X_train.columns.tolist()

        # ── Train + evaluate ──────────────────────────────────────────────────
        if self.regression_mode_:
            detailed_results, comparison_df, best_name, model_map = \
                self._run_regression(X_train, y_train, X_test, y_test)
        else:
            detailed_results, comparison_df, best_name, model_map = \
                self._run_classification(X_train, y_train, X_test, y_test)

        self.best_model_name_   = best_name
        self.best_model_        = model_map[best_name]
        self.detailed_results_  = detailed_results
        self.comparison_df_     = comparison_df

        # ── Persist ───────────────────────────────────────────────────────────
        model_path    = self.models_dir / "best_model.joblib"
        encoders_path = self.models_dir / "encoders.joblib"
        joblib.dump(self.best_model_, model_path)
        joblib.dump(encoders, encoders_path)
        logger.info(
            "[ModelSelectionAgent] Saved best model (%s) → %s",
            best_name, model_path,
        )

        test_set = X_test.copy()
        test_set["target"] = y_test
        test_path = self.models_dir / "latest_test_set.csv"
        test_set.to_csv(test_path, index=False)

        return {
            "mode":             mode,
            "best_model_name":  best_name,
            "detailed_results": detailed_results,
            "split_info": {
                "train_size": len(X_train),
                "test_size":  len(X_test),
                "total_size": len(X_train) + len(X_test),
            },
            "model_path":     str(model_path),
            "encoders_path":  str(encoders_path),
            "test_set_path":  str(test_path),
        }

    # ── Classification branch ─────────────────────────────────────────────────
    def _run_classification(self, X_train, y_train, X_test, y_test):
        logger.info("[ModelSelectionAgent] Training classifiers …")

        model_map = {
            "Logistic Regression": train_logistic_regression(X_train, y_train),
            "Random Forest":       train_random_forest(X_train, y_train),
            "XGBoost":             train_xgboost(X_train, y_train),
        }

        detailed_results = {}
        for name, model in model_map.items():
            detailed_results[name] = evaluate_model(name, model, X_test, y_test)
            if hasattr(model, "feature_importances_"):
                imp = (
                    pd.Series(model.feature_importances_,
                               index=self.feature_names_)
                    .sort_values(ascending=False)
                    .to_dict()
                )
                detailed_results[name]["feature_importance"] = imp

        comparison_df = compare_models(detailed_results, y_test)
        best_name = max(
            detailed_results,
            key=lambda n: detailed_results[n]["ROC_AUC"],
        )
        logger.info("[ModelSelectionAgent] Best classifier: %s (ROC-AUC=%.4f)",
                    best_name, detailed_results[best_name]["ROC_AUC"])
        return detailed_results, comparison_df, best_name, model_map

    # ── Regression branch ─────────────────────────────────────────────────────
    def _run_regression(self, X_train, y_train, X_test, y_test):
        logger.info("[ModelSelectionAgent] Training regressors …")

        model_map = {
            "Decision Tree":       train_decision_tree_regressor(X_train, y_train),
            "Random Forest":       train_random_forest_regressor(X_train, y_train),
            "Gradient Boosting":   train_gradient_boosting_regressor(X_train, y_train),
            "XGBoost":             train_xgboost_regressor(X_train, y_train),
        }

        detailed_results = {}
        for name, model in model_map.items():
            detailed_results[name] = evaluate_regressor(name, model, X_test, y_test)
            if hasattr(model, "feature_importances_"):
                imp = (
                    pd.Series(model.feature_importances_,
                               index=self.feature_names_)
                    .sort_values(ascending=False)
                    .to_dict()
                )
                detailed_results[name]["feature_importance"] = imp

        comparison_df = compare_regressors(detailed_results)
        # Lower RMSE is better
        best_name = min(
            detailed_results,
            key=lambda n: detailed_results[n]["RMSE"],
        )
        logger.info("[ModelSelectionAgent] Best regressor: %s (RMSE=%.2f, R²=%.4f)",
                    best_name,
                    detailed_results[best_name]["RMSE"],
                    detailed_results[best_name]["R2"])
        return detailed_results, comparison_df, best_name, model_map

    # ── validate ──────────────────────────────────────────────────────────────
    def validate(self) -> dict:
        if self.best_model_ is None:
            return {"passed": False, "issues": ["run() has not been called yet."]}

        issues = []
        if not (self.models_dir / "best_model.joblib").exists():
            issues.append("best_model.joblib not found on disk.")

        # Mode-specific quality gate
        if self.regression_mode_:
            r2 = self.detailed_results_.get(self.best_model_name_, {}).get("R2", 0)
            if r2 < 0:
                issues.append(
                    f"Best regressor R²={r2:.4f} is negative — "
                    "model performs worse than predicting the mean."
                )
        else:
            roc = self.detailed_results_.get(self.best_model_name_, {}).get("ROC_AUC", 0)
            MIN_ROC_AUC = 0.60
            if roc < MIN_ROC_AUC:
                issues.append(
                    f"Best classifier ROC-AUC={roc:.4f} is below the "
                    f"minimum threshold of {MIN_ROC_AUC}."
                )

        return {"passed": len(issues) == 0, "issues": issues}

    # ── report ────────────────────────────────────────────────────────────────
    def report(self) -> dict:
        if self.comparison_df_ is None:
            return {"error": "run() has not been called yet."}

        base = {
            "mode":        "regression" if self.regression_mode_ else "classification",
            "best_model":  self.best_model_name_,
            "comparison":  self.comparison_df_.to_dict(orient="records"),
        }

        metrics = self.detailed_results_.get(self.best_model_name_, {})
        if self.regression_mode_:
            base["best_rmse"] = metrics.get("RMSE")
            base["best_mae"]  = metrics.get("MAE")
            base["best_r2"]   = metrics.get("R2")
        else:
            base["best_roc_auc"] = metrics.get("ROC_AUC")
            base["best_f1"]      = metrics.get("F1_Score")

        return base
