# ============================================
# 6. Evaluation & Visualisation
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from . import config


def evaluate_model(name: str, model, X_test, y_test):
    """Print accuracy, ROC-AUC, classification report, and confusion matrix."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return y_pred, y_prob


def compare_models(results: dict, y_test):
    """
    Compare multiple models side-by-side.

    Parameters
    ----------
    results : dict[str, tuple[y_pred, y_prob]]
        Mapping from model name to (y_pred, y_prob) tuple.
    y_test : array-like
        True labels.
    """
    rows = []
    for name, (y_pred, y_prob) in results.items():
        rows.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC_AUC': roc_auc_score(y_test, y_prob),
        })

    comparison = pd.DataFrame(rows)
    print("\n===== Model Comparison =====")
    print(comparison.sort_values(by='ROC_AUC', ascending=False))
    return comparison


def plot_feature_importance(model, feature_names, top_n: int = 10):
    """Bar chart of the top feature importances from a tree-based model."""
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    importance.sort_values().plot(kind='barh')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


def tune_random_forest(X_train, y_train, X_test, y_test):
    """Run RandomizedSearchCV to tune the Random Forest."""
    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=config.RANDOM_STATE),
        param_distributions=config.RF_PARAM_GRID,
        n_iter=config.RF_TUNING_N_ITER,
        cv=config.RF_TUNING_CV,
        scoring='roc_auc',
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)

    print("\n===== Tuned Random Forest =====")
    print("Best Parameters:", rf_search.best_params_)
    print("Best CV ROC-AUC:", rf_search.best_score_)

    best_rf = rf_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)[:, 1]

    print("Tuned RF Accuracy:", accuracy_score(y_test, y_pred))
    print("Tuned RF ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return best_rf, y_pred, y_prob
