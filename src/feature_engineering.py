# ============================================
# 3. Feature Engineering
# ============================================

import pandas as pd
from . import config


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Created Date and Closed Date strings into datetime objects."""
    for col in ['Created Date', 'Closed Date']:
        df[col] = pd.to_datetime(
            df[col],
            format=config.DATE_FORMAT,
            errors='coerce',
        )
    print("Missing Created Date after parsing:", df['Created Date'].isna().sum())
    print("Missing Closed Date after parsing:", df['Closed Date'].isna().sum())
    print("Total rows after parsing:", len(df))
    print(df[['Created Date', 'Closed Date']].head())
    return df


def create_response_time(df: pd.DataFrame) -> pd.DataFrame:
    """Create `days_to_close` feature (response time in hours)."""
    df['days_to_close'] = (
        df['Closed Date'] - df['Created Date']
    ).dt.total_seconds() / 3600
    print("\nSummary of response time in hours:")
    print(df['days_to_close'].describe())
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing or negative response times."""
    df = df.dropna(subset=['Created Date', 'Closed Date', 'days_to_close'])
    df = df[df['days_to_close'] >= 0]
    print("\nShape after removing missing/invalid response times:", df.shape)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract month, day_of_week, and hour from Created Date."""
    df['month'] = df['Created Date'].dt.month
    df['day_of_week'] = df['Created Date'].dt.dayofweek
    df['hour'] = df['Created Date'].dt.hour
    return df


def create_binary_target(
    df: pd.DataFrame,
    threshold_hours: float | None = None,
) -> pd.DataFrame:
    """Create binary target: 1 if closed within *threshold_hours*, else 0."""
    if threshold_hours is None:
        threshold_hours = config.TARGET_THRESHOLD_HOURS

    df['target'] = (df['days_to_close'] <= threshold_hours).astype(int)
    print("\nTarget distribution:")
    print(df['target'].value_counts())
    print("\nTarget proportion:")
    print(df['target'].value_counts(normalize=True))
    return df


def create_regression_target(df: pd.DataFrame) -> pd.DataFrame:
    """Expose hours_to_close as the regression target column.

    Adds a ``target`` column equal to ``days_to_close`` (which is stored in
    hours despite its name) so that ModelSelectionAgent can treat both
    classification and regression identically via the same ``target`` column.

    Call this instead of ``create_binary_target`` when
    ``config.REGRESSION_MODE = True``.
    """
    df['target'] = df['days_to_close'].copy()
    print("\nRegression target (hours_to_close) summary:")
    print(df['target'].describe().round(2))
    return df
