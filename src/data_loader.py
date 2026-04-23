# ============================================
# 1. Data Loading
# ============================================

import io
import os
from pathlib import Path
import pandas as pd
from . import config


def load_data(path=None) -> pd.DataFrame:
    """Load the raw 311 Service Requests CSV with memory efficiency.

    Parameters
    ----------
    path : str | Path | file-like | None
        str / Path  -> read from disk
        file-like   -> read from an in-memory buffer (e.g. Streamlit UploadedFile)
        None        -> fall back to config.DATA_PATH
    """
    if path is None:
        path = config.DATA_PATH

    # If a file-like object is passed (e.g. Streamlit UploadedFile),
    # wrap its bytes in io.BytesIO so pandas always gets a seekable buffer.
    if hasattr(path, "read"):
        path = io.BytesIO(path.read())

    # Memory optimization: only load columns we need for modelling or reasoning
    usecols = [
        "City Council District", "Department", "Service Request Type",
        "ERT (Estimated Response Time)", "Overall Service Request Due Date",
        "Created Date", "Closed Date", "Priority", "Method Received Description"
    ]
    
    # Check if path is a file on disk to verify existence and size
    nrows = None
    if isinstance(path, (str, Path)) and os.path.exists(path):
        file_size = os.path.getsize(path)
        if file_size > 50 * 1024 * 1024:  # > 50MB
            print(f"[Memory] Large file detected ({file_size / 1e6:.1f} MB). Limiting to {config.MAX_ROWS_TO_LOAD} rows.")
            nrows = config.MAX_ROWS_TO_LOAD

    try:
        data_311 = pd.read_csv(
            path,
            usecols=lambda c: c in usecols or c in ["Status", "Update Date", "Outcome"], # Include leakage for cleaner drop later
            dtype=config.CATEGORICAL_DTYPES,
            nrows=nrows,
            engine='c', # Faster engine
            low_memory=True
        )
    except Exception as e:
        print(f"[Memory] Optimized load failed, falling back to basic load: {e}")
        data_311 = pd.read_csv(path, nrows=nrows)

    print("Memory Usage (MB):", round(data_311.memory_usage(deep=True).sum() / 1e6, 2))
    print("Original shape:", data_311.shape)
    return data_311
