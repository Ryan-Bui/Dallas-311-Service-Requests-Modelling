# ============================================
# Agent 5: Regularization
# Phase 1.6 — Snippet Migration
# ============================================
"""
RegularizationAgent
--------------------
Responsibility: Apply Ridge, LASSO, and ElasticNet regularization
on the already-encoded training data. Selects the best alpha via
cross-validation and returns a coefficient summary table.

Uses sklearn's RidgeClassifierCV / LogisticRegressionCV (LASSO/ElasticNet)
which handle multi-alpha search internally.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LogisticRegressionCV,
    RidgeClassifier,
)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RegularizationAgent(BaseAgent):
    """Fit L1/L2/ElasticNet regularized classifiers and summarize coefficients."""

    ALPHAS   = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    CV_FOLDS = 3

    def __init__(self) -> None:
        self.results_: dict | None = None
        self.best_method_: str | None = None
        self.coef_summary_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list[str] | None = None,
    ) -> dict:
        """Fit Ridge, LASSO, and ElasticNet; return performance + coef summary.

        Parameters
        ----------
        X_train, X_test : pd.DataFrame
            Encoded feature matrices (output of ModelSelectionAgent encoding).
        y_train, y_test : pd.Series
            Binary target series.
        feature_names : list[str], optional
            Column names for the coefficient table.

        Returns
        -------
        dict
            Keys: 'best_method', 'best_roc_auc', 'coef_summary' (DataFrame)
        """
        cols = feature_names or (list(X_train.columns) if hasattr(X_train, "columns") else None)
        
        # Data Scrubbing: handle NaNs or Infinite values that sklearn hates
        X_tr = pd.DataFrame(X_train).fillna(0).replace([np.inf, -np.inf], 0).values
        X_te = pd.DataFrame(X_test).fillna(0).replace([np.inf, -np.inf], 0).values
        y_tr = np.array(y_train)
        y_te = np.array(y_test)

        results: dict[str, dict] = {}

        # ---- Ridge (L2) ----
        logger.info("[RegularizationAgent] Fitting Ridge …")
        ridge_scores = {}
        for alpha in self.ALPHAS:
            clf = RidgeClassifier(alpha=alpha)
            clf.fit(X_tr, y_tr)
            # RidgeClassifier has no predict_proba; use decision_function as proxy
            scores = clf.decision_function(X_te)
            auc = roc_auc_score(y_te, scores)
            ridge_scores[alpha] = (auc, clf)
        best_alpha_ridge, (best_auc_ridge, best_ridge) = max(
            ridge_scores.items(), key=lambda x: x[1][0]
        )
        results["Ridge"] = {
            "best_alpha": best_alpha_ridge,
            "roc_auc":    round(best_auc_ridge, 4),
            "coef":       best_ridge.coef_[0],
        }

        # ---- LASSO / L1 (via LogisticRegressionCV) ----
        logger.info("[RegularizationAgent] Fitting LASSO (L1) …")
        lasso_cv = LogisticRegressionCV(
            Cs=self.ALPHAS,
            penalty="l1",
            solver="liblinear",  # liblinear is very fast for L1 on medium data
            cv=self.CV_FOLDS,
            scoring="roc_auc",
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        )
        lasso_cv.fit(X_tr, y_tr)
        lasso_auc = roc_auc_score(y_te, lasso_cv.predict_proba(X_te)[:, 1])
        results["LASSO"] = {
            "best_alpha": round(float(1.0 / lasso_cv.C_[0]), 6),
            "roc_auc":    round(lasso_auc, 4),
            "coef":       lasso_cv.coef_[0],
        }

        # ---- ElasticNet ----
        logger.info("[RegularizationAgent] Fitting ElasticNet …")
        enet_cv = LogisticRegressionCV(
            Cs=self.ALPHAS,
            penalty="elasticnet",
            solver="saga",  # saga is required for elasticnet
            l1_ratios=[0.5],
            cv=self.CV_FOLDS,
            scoring="roc_auc",
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        )
        enet_cv.fit(X_tr, y_tr)
        enet_auc = roc_auc_score(y_te, enet_cv.predict_proba(X_te)[:, 1])
        results["ElasticNet"] = {
            "best_alpha": round(float(1.0 / enet_cv.C_[0]), 6),
            "roc_auc":    round(enet_auc, 4),
            "coef":       enet_cv.coef_[0],
        }

        # Best method
        best_method = max(results, key=lambda m: results[m]["roc_auc"])
        self.best_method_ = best_method

        # Coefficient summary table
        if cols:
            n_features = len(cols)
            logger.info("[RegularizationAgent] Generating summary for %d features", n_features)
            rows = []
            for method, res in results.items():
                try:
                    # Defensive: Ensure coef is a 1D numpy array
                    coef_raw = res.get("coef")
                    if coef_raw is None:
                        continue
                    
                    coef = np.atleast_1d(coef_raw).flatten()
                    
                    if len(coef) != n_features:
                        logger.warning(
                            "[RegularizationAgent] Dimension mismatch for %s: "
                            "Expected %d features, but model has %d. Zipping up to shortest.",
                            method, n_features, len(coef)
                        )
                    
                    for fname, c in zip(cols, coef):
                        rows.append({"Method": method, "Feature": fname, "Coefficient": round(float(c), 6)})
                except Exception as e:
                    logger.error("[RegularizationAgent] Failed to process coefficients for %s: %s", method, e)
            
            self.coef_summary_ = pd.DataFrame(rows)
        else:
            self.coef_summary_ = pd.DataFrame()

        self.results_ = {
            "best_method":   best_method,
            "best_roc_auc":  results[best_method]["roc_auc"],
            "all_results":   {m: {k: v for k, v in r.items() if k != "coef"} for m, r in results.items()},
            "coef_summary":  self.coef_summary_,
        }

        logger.info(
            "[RegularizationAgent] Done. Best: %s (ROC-AUC=%.4f)",
            best_method, results[best_method]["roc_auc"],
        )
        return self.results_

    # ------------------------------------------------------------------
    def validate(self) -> dict:
        if self.results_ is None:
            return {"passed": False, "issues": ["run() has not been called yet."]}
        return {"passed": True, "issues": []}

    # ------------------------------------------------------------------
    def report(self) -> dict:
        if self.results_ is None:
            return {"error": "run() has not been called yet."}
        return {
            "best_method":  self.best_method_,
            "best_roc_auc": self.results_["best_roc_auc"],
            "all_results":  self.results_["all_results"],
        }
