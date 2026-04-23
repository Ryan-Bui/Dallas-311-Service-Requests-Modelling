# ============================================
# Agent 4: Model Selection
# Phase 1.5 — Snippet Migration
# ============================================
"""
ModelSelectionAgent
-------------------
Responsibility: Encode features, split data (no leakage), train
Logistic Regression / Random Forest / XGBoost, compare by ROC-AUC,
save the best model + encoders to disk, and return a ranked results dict.

Wraps:
  src.preprocessing → handle_missing_values(), handle_service_request_type(),
                      encode_categoricals(), split_features_target(),
                      split_train_test()
  src.models        → train_logistic_regression(), train_random_forest(),
                      train_xgboost()
  src.evaluation    → evaluate_model(), compare_models()
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
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from src.evaluation import evaluate_model, compare_models

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class ModelSelectionAgent(BaseAgent):
    """Train and rank ML models; persist the best one to disk."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Populated after run()
        self.best_model_     = None
        self.best_model_name_: str | None = None
        self.comparison_df_: pd.DataFrame | None = None
        self.encoders_: dict | None = None
        self.X_train_: pd.DataFrame | None = None
        self.y_train_  = None
        self.X_test_: pd.DataFrame | None = None
        self.y_test_  = None
        self.feature_names_: list[str] | None = None

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> dict:
        """Encode, split, train, evaluate, and save the best model.

        Parameters
        ----------
        df : pd.DataFrame
            Transformed DataFrame from TransformationAgent.

        Returns
        -------
        dict
            Keys: 'best_model_name', 'comparison', 'model_path', 'encoders_path'
        """
        logger.info("[ModelSelectionAgent] Splitting features / target …")
        X, y = split_features_target(df)
        self.feature_names_ = X.columns.tolist()

        logger.info("[ModelSelectionAgent] Train/test split …")
        X_train, X_test, y_train, y_test = split_train_test(X, y)

        logger.info("[ModelSelectionAgent] Imputing missing values …")
        X_train, X_test, imputers = handle_missing_values(X_train, X_test)

        logger.info("[ModelSelectionAgent] Handling service request types …")
        X_train, X_test = handle_service_request_type(X_train, X_test)

        logger.info("[ModelSelectionAgent] Encoding categoricals …")
        X_train, X_test, encoders = encode_categoricals(X_train, X_test)
        
        # Include imputers and original feature list in the encoders payload for manual inference / stage B
        encoders["imputers"] = imputers
        encoders["original_features"] = X.columns.tolist()
        encoders["numeric_cols"] = imputers.get("numeric_cols", [])
        encoders["cat_cols"] = imputers.get("cat_cols", [])

        logger.info("[ModelSelectionAgent] Standardizing features …")
        X_train, X_test, scaler = scale_features(X_train, X_test)
        encoders["scaler"] = scaler

        self.encoders_      = encoders
        self.X_train_       = X_train
        self.y_train_       = y_train
        self.X_test_        = X_test
        self.y_test_        = y_test
        self.feature_names_ = X_train.columns.tolist()

        # Train all models
        logger.info("[ModelSelectionAgent] Training Logistic Regression …")
        log_model = train_logistic_regression(X_train, y_train)

        logger.info("[ModelSelectionAgent] Training Random Forest …")
        rf_model  = train_random_forest(X_train, y_train)

        logger.info("[ModelSelectionAgent] Training XGBoost …")
        xgb_model = train_xgboost(X_train, y_train)

        model_map = {
            "Logistic Regression": log_model,
            "Random Forest":       rf_model,
            "XGBoost":             xgb_model,
        }

        # Evaluate all models with rich metrics
        detailed_results = {}
        for name, model in model_map.items():
            detailed_results[name] = evaluate_model(name, model, X_test, y_test)
            # Add feature importance if available
            if hasattr(model, 'feature_importances_'):
                imp = pd.Series(model.feature_importances_, index=self.feature_names_).sort_values(ascending=False).to_dict()
                detailed_results[name]["feature_importance"] = imp

        # Determine best model
        best_name = max(detailed_results, key=lambda n: detailed_results[n]["ROC_AUC"])
        self.best_model_name_ = best_name
        self.best_model_      = model_map[best_name]

        # Persist
        model_path   = self.models_dir / "best_model.joblib"
        encoders_path = self.models_dir / "encoders.joblib"
        joblib.dump(self.best_model_, model_path)
        joblib.dump(encoders, encoders_path)

        logger.info("[ModelSelectionAgent] Saved best model (%s) to %s", best_name, model_path)

        # Persist test set for historical evaluation
        test_set     = X_test.copy()
        test_set['target'] = y_test
        test_path    = self.models_dir / "latest_test_set.csv"
        test_set.to_csv(test_path, index=False)
        logger.info("[ModelSelectionAgent] Saved test set to %s", test_path)

        return {
            "best_model_name": best_name,
            "detailed_results": detailed_results,
            "split_info": {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "total_size": len(X_train) + len(X_test)
            },
            "model_path":      str(model_path),
            "encoders_path":   str(encoders_path),
            "test_set_path":   str(test_path),
        }

    # ------------------------------------------------------------------
    def validate(self) -> dict:
        if self.best_model_ is None:
            return {"passed": False, "issues": ["run() has not been called yet."]}
        issues = []
        model_path = self.models_dir / "best_model.joblib"
        if not model_path.exists():
            issues.append("best_model.joblib not found on disk.")
        return {"passed": len(issues) == 0, "issues": issues}

    # ------------------------------------------------------------------
    def report(self) -> dict:
        if self.comparison_df_ is None:
            return {"error": "run() has not been called yet or report data cleared."}
        return {
            "best_model":  self.best_model_name_,
            "comparison":  self.comparison_df_.to_dict(orient="records"),
        }

    def clear(self) -> None:
        """Clear large training and test sets and trigger GC."""
        import gc
        self.X_train_ = None
        self.X_test_ = None
        self.y_train_ = None
        self.y_test_ = None
        gc.collect()
        logger.info("[ModelSelectionAgent] Training and test sets cleared (GC triggered).")
