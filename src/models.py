# ============================================
# 5. Model Training
# ============================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from . import config


def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(
        max_iter=config.LOG_REG_MAX_ITER,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier with baseline hyper-parameters."""
    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
        class_weight='balanced',
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        subsample=config.XGB_SUBSAMPLE,
        colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
        random_state=config.RANDOM_STATE,
        eval_metric='logloss',
    )
    model.fit(X_train, y_train)
    return model
