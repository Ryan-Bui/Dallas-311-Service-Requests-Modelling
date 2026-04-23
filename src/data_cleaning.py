# ============================================
# 2. Data Cleaning
# ============================================

import pandas as pd
from . import config


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not needed for modelling."""
    df = df.drop(
        columns=[col for col in config.COLUMNS_TO_DROP_INITIAL if col in df.columns],
        errors='ignore',
    )
    print("\nShape after dropping unnecessary columns:", df.shape)
    return df


def group_rare_departments(
    df: pd.DataFrame,
    min_count: int | None = None,
) -> pd.DataFrame:
    """Collapse departments below *min_count* into 'Other'."""
    if min_count is None:
        min_count = config.MIN_DEPARTMENT_COUNT

    if 'Department' in df.columns:
        # Handle Categorical types: ensure 'Other' is a valid category before assignment
        if df['Department'].dtype.name == 'category':
            if 'Other' not in df['Department'].cat.categories:
                df['Department'] = df['Department'].cat.add_categories(['Other'])
        
        dept_counts = df['Department'].value_counts()
        common_depts = dept_counts[dept_counts >= min_count].index
        df['Department_grouped'] = df['Department'].where(
            df['Department'].isin(common_depts),
            'Other',
        )
    else:
        df['Department_grouped'] = 'Other'

    print("\nDepartment grouped counts:")
    print(df['Department_grouped'].value_counts())
    return df


def sample_data(
    df: pd.DataFrame,
    frac: float | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Stratified sample by Department_grouped."""
    if frac is None:
        frac = config.SAMPLE_FRAC
    if random_state is None:
        random_state = config.RANDOM_STATE

    sampled = (
        df.groupby('Department_grouped', group_keys=False)
        .sample(frac=frac, random_state=random_state)
        .reset_index(drop=True)
    )
    print("\nSampled shape:", sampled.shape)
    return sampled
