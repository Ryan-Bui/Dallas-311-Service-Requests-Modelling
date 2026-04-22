# ============================================
# 5. Model Training
# ============================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

from . import config


# ── Classification ────────────────────────────────────────────────────────────

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


# ── Regression ────────────────────────────────────────────────────────────────

def train_decision_tree_regressor(X_train, y_train):
    """Train a Decision Tree Regressor — single tree, interpretable baseline."""
    model = DecisionTreeRegressor(
        max_depth=config.DT_MAX_DEPTH,
        min_samples_split=config.DT_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.DT_MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_regressor(X_train, y_train):
    """Train a Random Forest Regressor — ensemble of decision trees."""
    model = RandomForestRegressor(
        n_estimators=config.RFR_N_ESTIMATORS,
        max_depth=config.RFR_MAX_DEPTH,
        min_samples_split=config.RFR_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.RFR_MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting_regressor(X_train, y_train):
    """Train a Gradient Boosting Regressor (sklearn).

    Builds trees sequentially, each correcting the residuals of the last.
    More interpretable learning-rate / tree-depth trade-off than XGBoost.
    """
    model = GradientBoostingRegressor(
        n_estimators=config.GBR_N_ESTIMATORS,
        max_depth=config.GBR_MAX_DEPTH,
        learning_rate=config.GBR_LEARNING_RATE,
        subsample=config.GBR_SUBSAMPLE,
        min_samples_leaf=config.GBR_MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost_regressor(X_train, y_train):
    """Train an XGBoost Regressor.

    Identical tree-boosting logic to GBR but with hardware-accelerated
    column sub-sampling and built-in regularisation (reg_alpha / reg_lambda).
    """
    model = XGBRegressor(
        n_estimators=config.XGBR_N_ESTIMATORS,
        max_depth=config.XGBR_MAX_DEPTH,
        learning_rate=config.XGBR_LEARNING_RATE,
        subsample=config.XGBR_SUBSAMPLE,
        colsample_bytree=config.XGBR_COLSAMPLE_BYTREE,
        random_state=config.RANDOM_STATE,
        eval_metric='rmse',
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
