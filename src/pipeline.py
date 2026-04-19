# ============================================
# End-to-end Pipeline Runner
# ============================================
"""
Reproduces the full notebook workflow in a single script.
Usage:
    python -m src.pipeline
    python -m src.pipeline --data path/to/data.csv
"""

import argparse

from .data_loader import load_data
from .data_cleaning import drop_unnecessary_columns, group_rare_departments, sample_data
from .feature_engineering import (
    parse_dates,
    create_response_time,
    remove_invalid_rows,
    add_time_features,
    create_binary_target,
)
from .preprocessing import (
    drop_leakage_columns,
    handle_service_request_type,
    clean_ert,
    encode_categoricals,
    handle_missing_values,
    split_features_target,
    split_train_test,
)
from .models import train_logistic_regression, train_random_forest, train_xgboost
from .evaluation import (
    evaluate_model,
    compare_models,
    plot_feature_importance,
    tune_random_forest,
)


def run(data_path: str | None = None):
    # --- 1. Load ---
    df = load_data(data_path)

    # --- 2. Clean ---
    df = drop_unnecessary_columns(df)
    df = group_rare_departments(df)
    df = sample_data(df)

    # --- 3. Feature Engineering ---
    df = parse_dates(df)
    df = create_response_time(df)
    df = remove_invalid_rows(df)
    df = add_time_features(df)
    df = create_binary_target(df)

    # --- 4. Preprocessing (Stateless) ---
    df = drop_leakage_columns(df)
    df = clean_ert(df)
    
    # Split data before stateful transformations to prevent data leakage
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # --- 5. Preprocessing (Stateful) ---
    X_train, X_test = handle_missing_values(X_train, X_test)
    X_train, X_test = handle_service_request_type(X_train, X_test)
    X_train, X_test, encoders = encode_categoricals(X_train, X_test)

    # --- 6. Train Models ---
    log_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # --- 7. Evaluate ---
    results = {}
    results['Logistic Regression'] = evaluate_model(
        'Logistic Regression', log_model, X_test, y_test,
    )
    results['Random Forest'] = evaluate_model(
        'Random Forest', rf_model, X_test, y_test,
    )
    results['XGBoost'] = evaluate_model(
        'XGBoost', xgb_model, X_test, y_test,
    )
    compare_models(results, y_test)

    # Feature importance (Random Forest)
    plot_feature_importance(rf_model, X_train.columns)

    # Hyper-parameter tuning
    tune_random_forest(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dallas 311 Service Requests ML Pipeline',
    )
    parser.add_argument(
        '--data', type=str, default=None,
        help='Path to the raw CSV (defaults to config.DATA_PATH)',
    )
    args = parser.parse_args()
    run(data_path=args.data)
