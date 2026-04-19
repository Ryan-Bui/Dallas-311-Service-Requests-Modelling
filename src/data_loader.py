# ============================================
# 1. Data Loading
# ============================================

import io
import pandas as pd
from . import config


def load_data(path=None) -> pd.DataFrame:
    """Load the raw 311 Service Requests CSV.

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

    data_311 = pd.read_csv(path)
    print("Original shape:", data_311.shape)
    print("\nColumns:")
    print(data_311.columns)
    print("\nInfo:")
    print(data_311.info())
    print("\nFirst 5 rows:")
    print(data_311.head())
    return data_311
