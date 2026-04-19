# ============================================
# Agent 3: Diagnostics
# Phase 1.4 — Snippet Migration
# ============================================
"""
DiagnosticsAgent
----------------
Responsibility: Run statistical diagnostics on the transformed DataFrame
and return a structured report dict. If diagnostics fail, the Orchestrator
(Phase 3) can re-invoke TransformationAgent with adjusted parameters.

Diagnostics performed:
  1. Normality of `days_to_close`  — D'Agostino K² test
  2. Class imbalance ratio on `target`
  3. Missing value audit
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DiagnosticsAgent(BaseAgent):
    """Run skewness, normality, and imbalance checks on the dataset."""

    # Thresholds
    NORMALITY_P_THRESHOLD = 0.05        # below → non-normal
    IMBALANCE_THRESHOLD   = 0.25        # minority class fraction below → imbalanced

    def __init__(self) -> None:
        self.report_: dict | None = None

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> dict:
        """Run all diagnostic checks.

        Parameters
        ----------
        df : pd.DataFrame
            Transformed DataFrame from TransformationAgent (must contain
            `days_to_close` and `target`).

        Returns
        -------
        dict
            Structured diagnostics report.
        """
        logger.info("[DiagnosticsAgent] Running diagnostics …")
        report: dict = {}

        # 1. Normality of days_to_close
        if "days_to_close" in df.columns:
            sample = df["days_to_close"].dropna()
            # D'Agostino K² works for n > 8
            if len(sample) > 8:
                stat, p = stats.normaltest(sample)
                report["normality_stat"]   = round(float(stat), 4)
                report["normality_p"]      = round(float(p), 6)
                report["normality_pass"]   = bool(p > self.NORMALITY_P_THRESHOLD)
            else:
                report["normality_pass"]   = None
                report["normality_note"]   = "Sample too small for normality test."

            # Log-transform skewness information
            report["skewness"] = round(float(sample.skew()), 4)
            log_skewness = round(float(np.log1p(sample.clip(lower=0)).skew()), 4)
            report["log_skewness"] = log_skewness
            report["log_transform_recommended"] = abs(log_skewness) < abs(report["skewness"])
        else:
            report["normality_pass"] = None
            report["normality_note"] = "'days_to_close' column not found."

        # 2. Class imbalance
        if "target" in df.columns:
            vc = df["target"].value_counts(normalize=True)
            minority_frac = float(vc.min())
            report["class_imbalance_ratio"]   = round(minority_frac, 4)
            report["class_imbalance_detected"] = minority_frac < self.IMBALANCE_THRESHOLD
            report["target_distribution"]      = vc.to_dict()
        else:
            report["class_imbalance_detected"] = None

        # 3. Missing values
        missing = df.isnull().sum()
        report["columns_with_missing"] = missing[missing > 0].to_dict()
        report["total_missing"]        = int(missing.sum())

        report["overall_pass"] = (
            report.get("normality_pass") is not False          # None = skip
            and not report.get("class_imbalance_detected", False)
            and report["total_missing"] == 0
        )

        self.report_ = report
        logger.info("[DiagnosticsAgent] Done. Overall pass: %s", report["overall_pass"])
        return report

    # ------------------------------------------------------------------
    def validate(self) -> dict:
        if self.report_ is None:
            return {"passed": False, "issues": ["run() has not been called yet."]}
        issues = []
        if not self.report_.get("overall_pass"):
            if self.report_.get("normality_pass") is False:
                issues.append("Normality test failed — consider log transform.")
            if self.report_.get("class_imbalance_detected"):
                issues.append("Class imbalance detected — consider class_weight or SMOTE.")
            if self.report_.get("total_missing", 0) > 0:
                issues.append(f"{self.report_['total_missing']} missing values remain.")
        return {"passed": len(issues) == 0, "issues": issues}

    # ------------------------------------------------------------------
    def report(self) -> dict:
        return self.report_ or {"error": "run() has not been called yet."}
