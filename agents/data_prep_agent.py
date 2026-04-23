# ============================================
# Agent 1: Data Preparation
# Phase 1.2 — Snippet Migration
# ============================================
"""
DataPrepAgent
-------------
Responsibility: Load the raw CSV, drop unnecessary columns,
group rare departments, and return a stratified sample as a
clean DataFrame ready for transformation.

Wraps:
  src.data_loader   → load_data()
  src.data_cleaning → drop_unnecessary_columns(),
                      group_rare_departments(),
                      sample_data()
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data_loader import load_data
from src.data_cleaning import (
    drop_unnecessary_columns,
    group_rare_departments,
    sample_data,
)
from src import config

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DataPrepAgent(BaseAgent):
    """Load, clean, and sample the raw 311 service-request CSV."""

    def __init__(self, data_path=None) -> None:
        # Accept a str/Path (disk file) or a file-like object (Streamlit UploadedFile)
        if data_path is None:
            self.data_path = config.DATA_PATH
        elif hasattr(data_path, "read"):   # file-like (UploadedFile / BytesIO)
            self.data_path = data_path
        else:
            self.data_path = str(data_path)
        self.df_: pd.DataFrame | None = None          # result after run()

    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Execute the full data-preparation pipeline.

        Returns
        -------
        pd.DataFrame
            Clean, sampled DataFrame ready for TransformationAgent.
        """
        logger.info("[DataPrepAgent] Loading data from: %s", self.data_path)
        df = load_data(self.data_path)

        logger.info("[DataPrepAgent] Dropping unnecessary columns …")
        df = drop_unnecessary_columns(df)

        logger.info("[DataPrepAgent] Grouping rare departments …")
        df = group_rare_departments(df)

        logger.info("[DataPrepAgent] Sampling data (frac=%.2f) …", config.SAMPLE_FRAC)
        df = sample_data(df)

        self.df_ = df
        logger.info("[DataPrepAgent] Done. Shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    def validate(self) -> dict:
        """Basic sanity checks on the prepared DataFrame.

        Returns
        -------
        dict
            Keys: 'passed' (bool), 'issues' (list[str])
        """
        if self.df_ is None:
            return {"passed": False, "issues": ["run() has not been called yet."]}

        issues: list[str] = []
        df = self.df_

        if df.empty:
            issues.append("DataFrame is empty after preparation.")
        if "Department_grouped" not in df.columns:
            issues.append("'Department_grouped' column is missing.")
        if df.duplicated().sum() > 0:
            issues.append(f"{df.duplicated().sum()} duplicate rows detected.")

        return {"passed": len(issues) == 0, "issues": issues}

    # ------------------------------------------------------------------
    def report(self) -> dict:
        """Return a summary report of the prepared dataset."""
        if self.df_ is None:
            return {"error": "run() has not been called yet or data has been cleared."}

        df = self.df_
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "department_counts": df["Department_grouped"].value_counts().to_dict()
            if "Department_grouped" in df.columns
            else {},
            "missing_values": df.isnull().sum().to_dict(),
        }

    def clear(self) -> None:
        """Clear the internal DataFrame to free memory."""
        self.df_ = None
        logger.info("[DataPrepAgent] Internal data cleared.")
