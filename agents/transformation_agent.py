# ============================================
# Agent 2: Transformation
# Phase 1.3 — Snippet Migration
# ============================================
"""
TransformationAgent
-------------------
Responsibility: Take the clean DataFrame from DataPrepAgent and apply
all feature-engineering and stateless preprocessing steps. Returns a
DataFrame with numeric features and a binary target, ready for
diagnostics and modelling.

Wraps:
  src.feature_engineering → parse_dates(), create_response_time(),
                             remove_invalid_rows(), add_time_features(),
                             create_binary_target()
  src.preprocessing       → drop_leakage_columns(), clean_ert()
"""
from __future__ import annotations

import logging

import pandas as pd

from src.feature_engineering import (
    parse_dates,
    create_response_time,
    remove_invalid_rows,
    add_time_features,
    create_binary_target,
)
from src.preprocessing import drop_leakage_columns, clean_ert

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TransformationAgent(BaseAgent):
    """Clean and featurize the 311 dataset."""

    def __init__(self) -> None:
        self.df_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full transformation pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Clean DataFrame from DataPrepAgent.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with `target` and `days_to_close` columns
            present, ready for DiagnosticsAgent or ModelSelectionAgent.
        """
        logger.info("[TransformationAgent] Parsing date columns …")
        df = parse_dates(df)

        logger.info("[TransformationAgent] Creating response-time feature …")
        df = create_response_time(df)

        logger.info("[TransformationAgent] Removing invalid rows …")
        df = remove_invalid_rows(df)

        logger.info("[TransformationAgent] Adding calendar time features …")
        df = add_time_features(df)

        logger.info("[TransformationAgent] Creating binary target …")
        df = create_binary_target(df)

        logger.info("[TransformationAgent] Dropping leakage columns …")
        df = drop_leakage_columns(df)

        logger.info("[TransformationAgent] Cleaning ERT column …")
        df = clean_ert(df)

        self.df_ = df
        logger.info("[TransformationAgent] Done. Shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    def validate(self) -> dict:
        """Check that required output columns exist and target is binary."""
        if self.df_ is None:
            return {"passed": False, "issues": ["run() has not been called yet."]}

        issues: list[str] = []
        df = self.df_

        for col in ("target", "days_to_close", "month", "day_of_week", "hour"):
            if col not in df.columns:
                issues.append(f"Expected column '{col}' is missing.")

        if "target" in df.columns:
            unique_vals = set(df["target"].unique())
            if not unique_vals.issubset({0, 1}):
                issues.append(f"'target' contains non-binary values: {unique_vals}")

        return {"passed": len(issues) == 0, "issues": issues}

    # ------------------------------------------------------------------
    def report(self) -> dict:
        """Return a summary of the transformation output."""
        if self.df_ is None:
            return {"error": "run() has not been called yet or data has been cleared."}

        df = self.df_
        return {
            "shape": df.shape,
            "target_distribution": df["target"].value_counts().to_dict()
            if "target" in df.columns
            else {},
            "days_to_close_summary": df["days_to_close"].describe().to_dict()
            if "days_to_close" in df.columns
            else {},
            "missing_values": df.isnull().sum().sum(),
        }

    def clear(self) -> None:
        """Clear the internal DataFrame and trigger GC."""
        import gc
        self.df_ = None
        gc.collect()
        logger.info("[TransformationAgent] Internal data cleared (GC triggered).")
