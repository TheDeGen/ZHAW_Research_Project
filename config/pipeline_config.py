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
SPREAD_TARGET_DEADBAND = 3.0  # EUR/MWh band for neutral class
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

N_CV_SPLITS = 8
CV_MIN_TRAIN_SIZE = HOURS_PER_TWO_WEEKS  # Minimum 2 weeks for training
CV_STEP_SIZE_HOURS = 72  # Step size between CV folds

# Expanding window parameters
DEFAULT_EXPANDING_SPLITS = 8  # Slightly higher for more stable validation
DEFAULT_EXPANDING_STEP = 72
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

# Zero-shot classification settings
ZEROSHOT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

HIERARCHICAL_TOPIC_GROUPS = {
    "Politik & öffentliche Ordnung": [
        "Regierung, Parlament oder Behörden beschließen strengere Gesetze, Steuern oder Auflagen",
        "Regierung, Parlament oder Behörden lockern Gesetze, Steuern oder Auflagen",
    ],
    "Internationale Beziehungen & Sicherheit": [
        "Krieg, Konflikt, Terrorgefahr oder Sanktionen verschärfen die internationale Sicherheitslage",
        "Waffenruhe, Friedensgespräche oder Aufhebung von Sanktionen entspannen die internationale Sicherheitslage",
    ],
    "Wirtschaft, Unternehmen & Finanzen": [
        "Rezession, Unternehmenskrise, Bankenturbulenzen oder steigende Zinsen belasten die Wirtschaft",
        "Konjunkturerholung, starke Unternehmenszahlen oder sinkende Zinsen stützen die Wirtschaft",
    ],
    "Energie, Klima & Umwelt": [
        "Öl-, Gas- oder Strommärkte stehen unter Druck oder es kommt zu Versorgungsstörungen",
        "Öl-, Gas- oder Strommärkte entspannen sich oder Versorgungslage verbessert sich",
    ],
    "Technologie, Wissenschaft & Gesundheit": [
        "Störungen, Ausfälle oder Sicherheitslücken bei IT/Technologie verursachen Risiken",
        "Durchbrüche in Forschung, Medizin oder Technologie bringen Fortschritt und Entlastung",
    ],
    "Sport, Kultur & Lifestyle": [
        "Sportereignisse, Kultur oder Prominenz sorgen für Kontroversen oder Skandale",
        "Sportereignisse, Kultur oder Prominenz sorgen für Erfolge, Preise oder positive Resonanz",
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
    "stage_order": [
        "Politik & öffentliche Ordnung",
        "Internationale Beziehungen & Sicherheit",
        "Wirtschaft, Unternehmen & Finanzen",
        "Energie, Klima & Umwelt",
        "Technologie, Wissenschaft & Gesundheit",
        "Sport, Kultur & Lifestyle",
        "Sonstiges",
    ],
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

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Standard figure size for all plots (width, height in inches)
DEFAULT_FIGSIZE = (12, 8)
DEFAULT_DPI = 150

# ColorBrewer Dark2 palette (colorblind-safe)
# Reference: https://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=8
VIZ_COLOR_LIST = [
    "#1B9E77",  # Teal
    "#D95F02",  # Orange
    "#7570B3",  # Purple
    "#E7298A",  # Pink
    "#66A61E",  # Green
    "#E6AB02",  # Yellow
    "#A6761D",  # Brown
    "#666666",  # Gray
]

# Semantic color mappings for specific use cases
VIZ_SEMANTIC_COLORS = {
    "long": "#1B9E77",       # Teal for positive/long positions
    "neutral": "#666666",    # Gray for neutral
    "short": "#D95F02",      # Orange for negative/short positions
    "train": "#1B9E77",      # Teal for training metrics
    "validation": "#D95F02", # Orange for validation metrics
    "test": "#7570B3",       # Purple for test metrics
}

# Colormaps for heatmaps (colorblind-safe)
VIZ_CMAP_SEQUENTIAL = "YlGnBu"   # Yellow-Green-Blue for sequential data
VIZ_CMAP_DIVERGING = "RdYlBu"    # Red-Yellow-Blue for diverging data

# Output directory for saved figures
FIGURES_OUTPUT_DIR = "outputs/figures"
