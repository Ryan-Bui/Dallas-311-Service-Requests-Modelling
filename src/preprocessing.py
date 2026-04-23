# ============================================
# 4. Preprocessing / Encoding
# ============================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from . import config


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that would cause data leakage."""
    df = df.drop(
        columns=[col for col in config.LEAKAGE_COLUMNS if col in df.columns],
        errors='ignore',
    )
    print("\nShape after dropping leakage columns:", df.shape)
    return df


def handle_service_request_type(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    top_n: int | None = None,
):
    """Keep only the top *top_n* service-request types (fitted on train); collapse rest to 'Other'."""
    if top_n is None:
        top_n = config.TOP_SERVICE_REQUEST_TYPES

    if 'Service Request Type' in X_train.columns:
        top_types = X_train['Service Request Type'].value_counts().nlargest(top_n).index
        
        X_train['Service Request Type'] = X_train['Service Request Type'].where(
            X_train['Service Request Type'].isin(top_types),
            'Other',
        )
        
        X_test['Service Request Type'] = X_test['Service Request Type'].where(
            X_test['Service Request Type'].isin(top_types),
            'Other',
        )
    return X_train, X_test


def clean_ert(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric ERT days from the text column."""
    if 'ERT (Estimated Response Time)' in df.columns:
        df['ERT_days'] = (
            df['ERT (Estimated Response Time)']
            .astype(str)
            .str.extract(r'(\d+)')[0]
        )
        df['ERT_days'] = pd.to_numeric(df['ERT_days'], errors='coerce')
        df = df.drop(columns=['ERT (Estimated Response Time)'], errors='ignore')
    return df


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame, initial_encoders: dict | None = None):
    """Ordinal-encode selected columns and one-hot-encode the rest.
    
    If initial_encoders is provided, it uses them to transform (inference mode).
    Otherwise, it fits new encoders on X_train.
    """
    encoders = initial_encoders or {}
    
    # 1. Ordinal encoding
    for col in config.LABEL_ENCODE_COLUMNS:
        if col in X_train.columns:
            # Sanitization for inference: cast to str and fill NaNs to prevent isnan TypeError
            X_train[col] = X_train[col].astype(str).fillna("Unknown")
            X_test[col] = X_test[col].astype(str).fillna("Unknown")
            
            if col in encoders:
                oe = encoders[col]
                X_train[col] = oe.transform(X_train[[col]])
                X_test[col] = oe.transform(X_test[[col]])
            else:
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_train[col] = oe.fit_transform(X_train[[col]])
                X_test[col] = oe.transform(X_test[[col]])
                encoders[col] = oe

    # 2. One-hot encoding
    object_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(object_cols) > 0:
        if "ohe" in encoders:
            ohe = encoders["ohe"]
            feature_names = ohe.get_feature_names_out(object_cols)
            
            train_encoded = ohe.transform(X_train[object_cols])
            train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=X_train.index)
            X_train = pd.concat([X_train.drop(columns=object_cols), train_encoded_df], axis=1)
            
            test_encoded = ohe.transform(X_test[object_cols])
            test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=X_test.index)
            X_test = pd.concat([X_test.drop(columns=object_cols), test_encoded_df], axis=1)
        else:
            ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            train_encoded = ohe.fit_transform(X_train[object_cols])
            feature_names = ohe.get_feature_names_out(object_cols)
            
            train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=X_train.index)
            X_train = pd.concat([X_train.drop(columns=object_cols), train_encoded_df], axis=1)
            
            test_encoded = ohe.transform(X_test[object_cols])
            test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=X_test.index)
            X_test = pd.concat([X_test.drop(columns=object_cols), test_encoded_df], axis=1)
            encoders['ohe'] = ohe

    print(f"\nCategorical encoding complete (Inference: {initial_encoders is not None})")
    return X_train, X_test, encoders


def handle_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fill numeric NaNs with median and categorical NaNs with most frequent."""
    # 1. Numeric Imputation
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_cols:
        imputer = SimpleImputer(strategy='median')
        X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

    # 2. Categorical Imputation
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # 3. Boolean to int
    bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()
    X_train[bool_cols] = X_train[bool_cols].astype(int)
    X_test[bool_cols] = X_test[bool_cols].astype(int)

    print("\nTotal missing values in train after cleaning:", X_train.isnull().sum().sum())
    return X_train, X_test


def split_features_target(df: pd.DataFrame):
    """Return X, y after dropping the target and days_to_close columns."""
    X = df.drop(columns=['target', 'days_to_close'], errors='ignore')
    
    # Drop raw datetime/timedelta columns if they still exist (they break numpy-based models)
    X = X.select_dtypes(exclude=['datetime64', 'timedelta64'])
    
    y = df['target']
    print("\nX shape after safety drop:", X.shape)
    print("y shape:", y.shape)
    return X, y


def split_train_test(X, y, test_size=None, random_state=None):
    """Stratified train-test split."""
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print("\nTrain size:", X_train.shape, y_train.shape)
    print("Test size:", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Standardize numeric features using StandardScaler, fitted on train."""
    # Only scale columns that are numeric and NOT encoded (categorical)
    # Actually, scaling all columns is usually safer for solvers like SAGA
    scaler = StandardScaler()
    
    cols = X_train.columns
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=cols, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=cols, index=X_test.index)
    
    print("\nFeatures standardized using StandardScaler.")
    return X_train, X_test, scaler
