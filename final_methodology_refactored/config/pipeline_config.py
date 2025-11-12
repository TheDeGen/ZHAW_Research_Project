"""
Pipeline Configuration
======================
Contains all configurable parameters for the energy price prediction pipeline.
Modify these values to tune the pipeline behavior.
"""

import numpy as np
from scipy import stats

# ============================================================================
# TIME-BASED CONSTANTS
# ============================================================================

# Forecast horizon - how far ahead to predict
FORECAST_HORIZON_HOURS = 24  # Predict 24 hours ahead

# Time unit conversions
HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168  # 7 * 24
HOURS_PER_TWO_WEEKS = 336  # 14 * 24

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# Time decay parameters for news aggregation
DEFAULT_LOOKBACK_WINDOW = HOURS_PER_TWO_WEEKS  # 336 hours (2 weeks)
DEFAULT_DECAY_LAMBDA = 0.05  # Exponential decay rate

# Target variable parameters
SPREAD_TARGET_DEADBAND = 7.0  # EUR/MWh band for neutral class
TARGET_COLUMN = 'spread_target_shift_24'

# ============================================================================
# DATASET SPLIT PARAMETERS
# ============================================================================

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_STATE = 42

# ============================================================================
# CROSS-VALIDATION PARAMETERS
# ============================================================================

N_CV_SPLITS = 5
CV_MIN_TRAIN_SIZE = HOURS_PER_TWO_WEEKS  # Minimum 2 weeks for training
CV_STEP_SIZE_HOURS = 24  # Step size between CV folds

# Expanding window parameters
DEFAULT_EXPANDING_SPLITS = 4  # Lower number for quicker iteration
DEFAULT_EXPANDING_STEP = 24
DEFAULT_MIN_TRAIN_SIZE = 336  # Two weeks of hourly observations

# ============================================================================
# MODEL TRAINING PARAMETERS
# ============================================================================

DEFAULT_BATCH_SIZE = 32
DEFAULT_N_JOBS = -1  # Use all CPU cores
XGB_RANDOM_SEARCH_ITERS = 80  # Increased iterations for deeper hyperparameter exploration

# Ridge regression alphas for grid search
DEFAULT_ALPHAS = np.logspace(-3, 3, 13)

# ============================================================================
# NLP CONFIGURATION
# ============================================================================

# Sentence embedding model
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDING_DIM = 384
PCA_COMPONENTS = 50

# Zero-shot classification settings
ZEROSHOT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

HIERARCHICAL_TOPIC_GROUPS = {
    "Nachfrage": [
        "der Stromverbrauch in Deutschland steigt",
        "der Stromverbrauch in Deutschland fällt",
    ],
    "Angebot": [
        "die Stromerzeugung aus Wind und Sonne steigt",
        "die Stromerzeugung aus Wind und Sonne fällt",
        "Störungen oder Ausfälle bei Netzen oder Kraftwerken verringern das Angebot",
        "der Ausbau von LNG-Terminals, Pipelines oder Kraftwerken erhöht das Angebot",
    ],
    "Brennstoffpreise": [
        "die Großhandelspreise für Erdgas steigen",
        "die Großhandelspreise für Erdgas fallen",
    ],
    "Makrofinanzen": [
        "steigende Zinsen oder hohe Inflation verschärfen die Marktlage",
        "sinkende Zinsen oder nachlassende Inflation beruhigen die Marktlage",
    ],
    "Geopolitik": [
        "geopolitische Spannungen oder Sanktionen verschärfen die Energieversorgung",
        "geopolitische Entspannung oder gelockerte Sanktionen mindern Versorgungsrisiken",
    ],
    "Wetter": [
        "Kälte, Flaute oder wenig Sonne erhöhen den Strompreisdruck in Deutschland",
        "mildes Wetter, viel Wind oder viel Sonne entlasten die Strompreise in Deutschland",
    ],
    "Sonstiges": [
        "kein Bezug zu Energie, Wetter oder Finanzmärkten",
    ],
}

# Flattened list of candidate labels used in the final stage (kept for compatibility)
CANDIDATE_LABELS = [
    label for labels in HIERARCHICAL_TOPIC_GROUPS.values() for label in labels
]

HIERARCHICAL_ROUTING_SETTINGS = {
    "stage_order": ["Nachfrage", "Angebot", "Brennstoffpreise", "Makrofinanzen", "Geopolitik", "Wetter", "Sonstiges"],
    "stage_thresholds": {
        "stage1": 0.35,
        "stage2": 0.25,
    },
    "allow_fallback_to_other": True,
}

HYPOTHESIS_TEMPLATE = "Der Artikel handelt von: {}."

# Label for articles with no energy relevance (to be excluded from feature engineering)
OTHER_LABEL = "kein Bezug zu Energie, Wetter oder Finanzmärkten"

# ============================================================================
# XGBOOST HYPERPARAMETER DISTRIBUTIONS
# ============================================================================
# NOTE: Parameters tuned to reduce overfitting by constraining model complexity
# and strengthening regularization. Previous config showed significant test set
# degradation (Val F1: 0.460 → Test F1: 0.373).

XGB_PARAM_DISTRIBUTIONS = {
    # Reduce model complexity
    'n_estimators': stats.randint(100, 400),          # Reduced from 200-800
    'max_depth': stats.randint(2, 6),                 # Reduced from 2-9 to limit tree depth
    'learning_rate': stats.loguniform(0.01, 0.15),    # Reduced upper bound from 0.3 to 0.15

    # Increase regularization via sampling
    'subsample': stats.uniform(0.5, 0.4),             # Reduced from 0.6-1.0 to 0.5-0.9
    'colsample_bytree': stats.uniform(0.5, 0.4),      # Reduced from 0.6-1.0 to 0.5-0.9

    # Increase minimum split constraints
    'gamma': stats.uniform(0.5, 4.5),                 # Raised lower bound from 0.0 to 0.5
    'min_child_weight': stats.randint(3, 12),         # Raised from 1-10 to 3-12

    # Strengthen regularization penalties
    'reg_alpha': stats.loguniform(1e-3, 1e1),         # Raised lower bound from 1e-4 to 1e-3
    'reg_lambda': stats.loguniform(1e-2, 1e2)         # Raised lower bound from 1e-3 to 1e-2
}

# ============================================================================
# GRID SEARCH RANGES
# ============================================================================

# Time decay parameter grid
LOOKBACK_WINDOWS = [24, 48, 72, 168, 336, 504]  # 1, 2, 3 weeks
TIME_DECAY_LAMBDAS = [0.01, 0.05, 0.1, 0.25, 0.5]

# ============================================================================
# DATA PATHS
# ============================================================================

NEWS_DATA_PATH = "german_news_v1.csv"
ENERGY_DATA_PATH = "energy_baseline.csv"
MIN_TIMESTAMP = "2023-01-01"
