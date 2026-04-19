# ============================================
# Configuration / Constants
# ============================================

# Path to raw data
DATA_PATH = r"C:\Users\Trong Nguyen\Downloads\311_Service_Requests_20260418.csv"

# Columns to drop immediately after loading
COLUMNS_TO_DROP_INITIAL = [
    'Service Request Number',
    'Unique Key',
    'Address',
]

# Minimum count threshold for grouping rare departments
MIN_DEPARTMENT_COUNT = 1000

# Sampling fraction for stratified sampling
SAMPLE_FRAC = 0.1
RANDOM_STATE = 42

# Date format used in the raw CSV
DATE_FORMAT = '%Y %b %d %I:%M:%S %p'

# Binary target threshold (hours) — "fast" means closed within 72 hours
TARGET_THRESHOLD_HOURS = 72

# Columns to drop to avoid data leakage
LEAKAGE_COLUMNS = [
    'Closed Date',      # leakage
    'Update Date',      # leakage
    'Status',           # leakage
    'Outcome',          # borderline leakage
    'Lat_Long Location' # messy/high precision
]

# Number of top service-request types to keep (rest become "Other")
TOP_SERVICE_REQUEST_TYPES = 15

# Categorical columns to label-encode
LABEL_ENCODE_COLUMNS = [
    'Priority',
    'Method Received Description',
    'Department_grouped',
]

# Test size for train/test split
TEST_SIZE = 0.2

# ---- Model hyper-parameters ----

# Logistic Regression
LOG_REG_MAX_ITER = 2000

# Random Forest (baseline)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 10
RF_MIN_SAMPLES_LEAF = 5

# XGBoost
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# Random Forest hyper-parameter tuning grid
RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'class_weight': ['balanced', 'balanced_subsample'],
}

RF_TUNING_N_ITER = 10
RF_TUNING_CV = 3
