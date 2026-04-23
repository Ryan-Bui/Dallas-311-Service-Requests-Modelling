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
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from . import config


def evaluate_model(name: str, model, X_test, y_test):
    """Calculate and return a comprehensive metrics dictionary."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Standard Scores
    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "F1_Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
    }

    # Confusion Matrix
    results["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    # ROC Curve (Downsample to 50 points)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    step = max(1, len(fpr) // 50)
    results["roc_curve"] = {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()}

    # PR Curve (Downsample to 50 points)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    step = max(1, len(prec) // 50)
    results["pr_curve"] = {"precision": prec[::step].tolist(), "recall": rec[::step].tolist()}

    # Threshold Tuning (Precision/Recall vs Threshold)
    thresholds = np.linspace(0, 1, 21)
    t_metrics = []
    for t in thresholds:
        tp = (y_prob >= t).astype(int)
        t_metrics.append({
            "threshold": float(t),
            "precision": float(precision_score(y_test, tp, zero_division=0)),
            "recall": float(recall_score(y_test, tp, zero_division=0))
        })
    results["threshold_tuning"] = t_metrics

    return results


def compare_models(results: dict, y_test):
    """
    Compare multiple models side-by-side using pre-calculated metrics.
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            'Model': name,
            'Accuracy': metrics.get('Accuracy'),
            'ROC_AUC': metrics.get('ROC_AUC'),
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
