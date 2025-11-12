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
SPREAD_TARGET_DEADBAND = 5.0  # EUR/MWh band for neutral class
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

# Ridge regression alphas for grid search
DEFAULT_ALPHAS = np.logspace(-3, 3, 13)

# ============================================================================
# NLP CONFIGURATION
# ============================================================================

# Sentence embedding model
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDING_DIM = 384
PCA_COMPONENTS = 50

# Zero-shot classification model
ZEROSHOT_MODEL = "Sahajtomar/German_Zeroshot"

# Topic labels for zero-shot classification (German energy news)
CANDIDATE_LABELS = [
    # Nachfrage (Demand)
    "der Stromverbrauch in Deutschland steigt",
    "der Stromverbrauch in Deutschland fällt",

    # Angebot – Erneuerbare & Infrastruktur/Störungen (Supply)
    "die Stromerzeugung aus Wind und Sonne steigt",
    "die Stromerzeugung aus Wind und Sonne fällt",
    "Störungen oder Ausfälle bei Netzen oder Kraftwerken verringern das Angebot",
    "der Ausbau von LNG-Terminals, Pipelines oder Kraftwerken erhöht das Angebot",

    # Brennstoffpreise (fokus Gas) (Fuel prices)
    "die Großhandelspreise für Erdgas steigen",
    "die Großhandelspreise für Erdgas fallen",

    # Zinsen/Inflation
    "steigende Zinsen oder hohe Inflation verschärfen die Marktlage",
    "sinkende Zinsen oder nachlassende Inflation beruhigen die Marktlage",

    # Geopolitik / Versorgung (Geopolitics / Supply)
    "geopolitische Spannungen oder Sanktionen verschärfen die Energieversorgung",
    "geopolitische Entspannung oder gelockerte Sanktionen mindern Versorgungsrisiken",

    # Wetter (DE) (Weather)
    "Kälte, Flaute oder wenig Sonne erhöhen den Strompreisdruck in Deutschland",
    "mildes Wetter, viel Wind oder viel Sonne entlasten die Strompreise in Deutschland",

    # Catch-all
    "kein Bezug zu Energie, Wetter oder Finanzmärkten"
]

HYPOTHESIS_TEMPLATE = "Der Artikel handelt von: {}."

# ============================================================================
# XGBOOST HYPERPARAMETER DISTRIBUTIONS
# ============================================================================

XGB_PARAM_DISTRIBUTIONS = {
    'n_estimators': stats.randint(200, 800),
    'max_depth': stats.randint(2, 9),
    'learning_rate': stats.loguniform(0.01, 0.3),
    'subsample': stats.uniform(0.6, 0.4),
    'colsample_bytree': stats.uniform(0.6, 0.4),
    'gamma': stats.uniform(0.0, 5.0),
    'min_child_weight': stats.randint(1, 10),
    'reg_alpha': stats.loguniform(1e-4, 1e1),
    'reg_lambda': stats.loguniform(1e-3, 1e2)
}

# ============================================================================
# GRID SEARCH RANGES
# ============================================================================

# Time decay parameter grid
LOOKBACK_WINDOWS = [168, 336, 504]  # 1, 2, 3 weeks
TIME_DECAY_LAMBDAS = [0.01, 0.05, 0.1]

# ============================================================================
# DATA PATHS
# ============================================================================

NEWS_DATA_PATH = "german_news_v1.csv"
ENERGY_DATA_PATH = "energy_baseline.csv"
MIN_TIMESTAMP = "2025-01-01"
