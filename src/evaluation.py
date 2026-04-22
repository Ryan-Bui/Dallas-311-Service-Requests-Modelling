# ============================================
# 6. Evaluation & Visualisation
# ============================================

import numpy as np
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
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from . import config


# ── Classification evaluation ─────────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test):
    """Calculate and return a comprehensive metrics dictionary for a classifier."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "ROC_AUC":   roc_auc_score(y_test, y_prob),
        "F1_Score":  f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
    }

    results["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    step = max(1, len(fpr) // 50)
    results["roc_curve"] = {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()}

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    step = max(1, len(prec) // 50)
    results["pr_curve"] = {"precision": prec[::step].tolist(), "recall": rec[::step].tolist()}

    thresholds = np.linspace(0, 1, 21)
    t_metrics = []
    for t in thresholds:
        tp = (y_prob >= t).astype(int)
        t_metrics.append({
            "threshold": float(t),
            "precision": float(precision_score(y_test, tp, zero_division=0)),
            "recall":    float(recall_score(y_test, tp, zero_division=0)),
        })
    results["threshold_tuning"] = t_metrics

    return results


def compare_models(results: dict, y_test):
    """Compare classifiers side-by-side.

    Parameters
    ----------
    results : dict[str, dict]
        Output of evaluate_model() keyed by model name.
    y_test : array-like
        True labels (unused here — metrics already in results dict).
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  round(metrics.get("Accuracy", 0), 4),
            "ROC_AUC":   round(metrics.get("ROC_AUC", 0), 4),
            "F1_Score":  round(metrics.get("F1_Score", 0), 4),
            "Precision": round(metrics.get("Precision", 0), 4),
            "Recall":    round(metrics.get("Recall", 0), 4),
        })
    comparison = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
    print("\n===== Classifier Comparison =====")
    print(comparison.to_string(index=False))
    return comparison


# ── Regression evaluation ─────────────────────────────────────────────────────

def evaluate_regressor(name: str, model, X_test, y_test):
    """Calculate regression metrics for a fitted regressor.

    Returns
    -------
    dict
        Keys: RMSE, MAE, R2, MAPE, plus residual arrays for plotting.
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    # MAPE — guard against zeros in y_test
    nonzero = y_test != 0
    mape = float(np.mean(np.abs(residuals[nonzero] / y_test[nonzero])) * 100)

    # Downsample residual arrays to 500 points for JSON-safe storage
    step = max(1, len(y_pred) // 500)
    results = {
        "RMSE":      round(rmse, 3),
        "MAE":       round(mae, 3),
        "R2":        round(r2, 4),
        "MAPE":      round(mape, 2),
        # Scatter data for predicted-vs-actual plot
        "y_pred_sample":    y_pred[::step].tolist(),
        "y_test_sample":    np.array(y_test)[::step].tolist(),
        "residuals_sample": residuals.values[::step].tolist()
        if hasattr(residuals, "values") else residuals[::step].tolist(),
    }
    return results


def compare_regressors(results: dict) -> pd.DataFrame:
    """Compare regressors side-by-side.

    Parameters
    ----------
    results : dict[str, dict]
        Output of evaluate_regressor() keyed by model name.
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "RMSE":  metrics.get("RMSE"),
            "MAE":   metrics.get("MAE"),
            "R2":    metrics.get("R2"),
            "MAPE":  metrics.get("MAPE"),
        })
    # Lower RMSE is better, so sort ascending
    comparison = pd.DataFrame(rows).sort_values("RMSE", ascending=True)
    print("\n===== Regressor Comparison =====")
    print(comparison.to_string(index=False))
    return comparison


def plot_regression_diagnostics(results: dict, top_n: int = 2):
    """Predicted-vs-actual and residual plots for the top *top_n* regressors."""
    names = list(results.keys())[:top_n]
    fig, axes = plt.subplots(2, len(names), figsize=(7 * len(names), 10))
    if len(names) == 1:
        axes = axes.reshape(2, 1)

    for col, name in enumerate(names):
        r = results[name]
        y_pred = np.array(r["y_pred_sample"])
        y_true = np.array(r["y_test_sample"])
        resid  = np.array(r["residuals_sample"])

        # Predicted vs actual
        ax = axes[0, col]
        ax.scatter(y_true, y_pred, alpha=0.25, s=10, color="#378ADD", edgecolors="none")
        lim = max(y_true.max(), y_pred.max())
        ax.plot([0, lim], [0, lim], color="#E24B4A", linewidth=1.2,
                linestyle="--", label="Perfect prediction")
        ax.set_xlabel("Actual hours to close")
        ax.set_ylabel("Predicted hours to close")
        ax.set_title(f"{name}\nR²={r['R2']:.3f}  RMSE={r['RMSE']:.1f}h",
                     fontweight="500")
        ax.legend(fontsize=9)

        # Residuals vs predicted
        ax = axes[1, col]
        ax.scatter(y_pred, resid, alpha=0.25, s=10,
                   color="#1D9E75", edgecolors="none")
        ax.axhline(0, color="#E24B4A", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Predicted hours to close")
        ax.set_ylabel("Residual (actual − predicted)")
        ax.set_title(f"{name} — residuals\nMAE={r['MAE']:.1f}h  MAPE={r['MAPE']:.1f}%",
                     fontweight="500")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


# ── Shared utilities ──────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, top_n: int = 10):
    """Bar chart of the top feature importances from a tree-based model."""
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    importance.sort_values().plot(kind="barh", color="#378ADD", edgecolor="none")
    plt.title(f"Top {top_n} Feature Importances", fontweight="500")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


def tune_random_forest(X_train, y_train, X_test, y_test):
    """Run RandomizedSearchCV to tune the Random Forest classifier."""
    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=config.RANDOM_STATE),
        param_distributions=config.RF_PARAM_GRID,
        n_iter=config.RF_TUNING_N_ITER,
        cv=config.RF_TUNING_CV,
        scoring="roc_auc",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)

    print("\n===== Tuned Random Forest =====")
    print("Best Parameters:", rf_search.best_params_)
    print("Best CV ROC-AUC:", rf_search.best_score_)

    best_rf = rf_search.best_estimator_
    y_pred  = best_rf.predict(X_test)
    y_prob  = best_rf.predict_proba(X_test)[:, 1]

    print("Tuned RF Accuracy:", accuracy_score(y_test, y_pred))
    print("Tuned RF ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return best_rf, y_pred, y_prob
