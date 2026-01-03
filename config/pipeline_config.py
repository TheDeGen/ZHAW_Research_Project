"""
Pipeline Configuration
======================
Contains all configurable parameters for the energy price prediction pipeline.
Modify these values to tune the pipeline behavior.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

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
    "Nachfrage (Stromverbrauch)": [
        "Steigender Stromverbrauch durch Wirtschaft, Industrie oder Extremwetter",
        "Sinkender Stromverbrauch durch Konjunkturschwäche oder mildes Wetter",
    ],
    "Angebot (Erzeugung & Infrastruktur)": [
        "Kraftwerksausfälle, Netzengpässe oder geringe erneuerbare Einspeisung",
        "Hohe erneuerbare Einspeisung, neue Kapazitäten oder stabile Netze",
    ],
    "Brennstoffpreise": [
        "Steigende Gas-, Kohle- oder CO₂-Preise",
        "Fallende Gas-, Kohle- oder CO₂-Preise",
    ],
    "Wetter": [
        "Kaltes, windarmes oder bewölktes Wetter",
        "Mildes Wetter, starke Winde oder viel Sonneneinstrahlung",
    ],
    "Wirtschaft & Konjunktur": [
        "Positive Wirtschaftsentwicklung, steigende Industrieproduktion oder Unternehmenswachstum",
        "Rezession, Unternehmenskrise oder sinkende Industrieproduktion",
    ],
    "Finanzmärkte & Geldpolitik": [
        "Zinsentscheidungen, Inflation oder Währungsschwankungen",
        "Börsennachrichten, Unternehmensgewinne oder Investitionen",
    ],
    "Handel & Außenwirtschaft": [
        "Zölle, Handelskonflikte oder Exportbeschränkungen",
        "Handelsabkommen, Marktöffnung oder Lieferkettenentwicklung",
    ],
    "Geopolitik & Konflikte": [
        "Krieg, Sanktionen, Terrorgefahr oder geopolitische Spannungen",
        "Friedensgespräche, Diplomatie oder Aufhebung von Sanktionen",
    ],
    "Technologie & Industrie": [
        "Technologieentwicklung, Halbleiter, Elektromobilität oder Batterieproduktion",
        "Industriepolitik, Produktionsstandorte oder Unternehmensstrategien",
    ],
    "Politik & Regulierung": [
        "Energiepolitik, Klimagesetze oder EU-Regulierung",
        "Innenpolitik, Regierungsbildung oder Wahlen",
    ],
    "Sonstiges": [
        "Sport, Unterhaltung, Kultur oder Lokalnachrichten ohne Wirtschaftsbezug",
    ],
}

# Flattened list of candidate labels used in the final stage (kept for compatibility)
CANDIDATE_LABELS = [
    label for labels in HIERARCHICAL_TOPIC_GROUPS.values() for label in labels
]

HIERARCHICAL_ROUTING_SETTINGS = {
    "stage_order": [
        "Nachfrage (Stromverbrauch)",
        "Angebot (Erzeugung & Infrastruktur)",
        "Brennstoffpreise",
        "Wetter",
        "Wirtschaft & Konjunktur",
        "Finanzmärkte & Geldpolitik",
        "Handel & Außenwirtschaft",
        "Geopolitik & Konflikte",
        "Technologie & Industrie",
        "Politik & Regulierung",
        "Sonstiges",
    ],
    "stage_thresholds": {
        "stage1": 0.25,  # Lowered from 0.35
        "stage2": 0.20,  # Lowered from 0.25
    },
    "allow_fallback_to_other": True,
}

HYPOTHESIS_TEMPLATE = "Der Artikel handelt von: {}."

# Label for articles with no energy relevance (to be excluded from feature engineering)
OTHER_LABEL = "Sport, Unterhaltung, Kultur oder Lokalnachrichten ohne Wirtschaftsbezug"

# Topic valence mapping for positive/negative news classification
# Maps topic labels to their sentiment valence based on semantic content
TOPIC_VALENCE_MAP = {
    # Negative topics (price increasing)
    "Steigender Stromverbrauch durch Wirtschaft, Industrie oder Extremwetter": -1,
    "Kraftwerksausfälle, Netzengpässe oder geringe erneuerbare Einspeisung": -1,
    "Steigende Gas-, Kohle- oder CO₂-Preise": -1,
    "Kaltes, windarmes oder bewölktes Wetter": -1,
    "Rezession, Unternehmenskrise oder sinkende Industrieproduktion": -1,
    "Zinsentscheidungen, Inflation oder Währungsschwankungen": -1,
    "Zölle, Handelskonflikte oder Exportbeschränkungen": -1,
    "Krieg, Sanktionen, Terrorgefahr oder geopolitische Spannungen": -1,
    # Positive topics (price decreasing)
    "Sinkender Stromverbrauch durch Konjunkturschwäche oder mildes Wetter": 1,
    "Hohe erneuerbare Einspeisung, neue Kapazitäten oder stabile Netze": 1,
    "Fallende Gas-, Kohle- oder CO₂-Preise": 1,
    "Mildes Wetter, starke Winde oder viel Sonneneinstrahlung": 1,
    "Positive Wirtschaftsentwicklung, steigende Industrieproduktion oder Unternehmenswachstum": 1,
    "Börsennachrichten, Unternehmensgewinne oder Investitionen": 1,
    "Handelsabkommen, Marktöffnung oder Lieferkettenentwicklung": 1,
    "Friedensgespräche, Diplomatie oder Aufhebung von Sanktionen": 1,
    # Neutral topics
    "Technologieentwicklung, Halbleiter, Elektromobilität oder Batterieproduktion": 0,
    "Industriepolitik, Produktionsstandorte oder Unternehmensstrategien": 0,
    "Energiepolitik, Klimagesetze oder EU-Regulierung": 0,
    "Innenpolitik, Regierungsbildung oder Wahlen": 0,
    "Sport, Unterhaltung, Kultur oder Lokalnachrichten ohne Wirtschaftsbezug": 0,
}

# Topic label shortening: maps full topic labels to shortened category + keyword format
# Each topic gets a unique descriptive keyword to distinguish it
TOPIC_SHORT_LABELS = {
    # Nachfrage
    "Steigender Stromverbrauch durch Wirtschaft, Industrie oder Extremwetter": "Nachfrage (steigend)",
    "Sinkender Stromverbrauch durch Konjunkturschwäche oder mildes Wetter": "Nachfrage (sinkend)",
    # Angebot
    "Kraftwerksausfälle, Netzengpässe oder geringe erneuerbare Einspeisung": "Angebot (Engpass)",
    "Hohe erneuerbare Einspeisung, neue Kapazitäten oder stabile Netze": "Angebot (Ausbau)",
    # Brennstoffpreise
    "Steigende Gas-, Kohle- oder CO₂-Preise": "Brennstoff (steigend)",
    "Fallende Gas-, Kohle- oder CO₂-Preise": "Brennstoff (fallend)",
    # Wetter
    "Kaltes, windarmes oder bewölktes Wetter": "Wetter (kalt)",
    "Mildes Wetter, starke Winde oder viel Sonneneinstrahlung": "Wetter (mild)",
    # Wirtschaft & Konjunktur
    "Positive Wirtschaftsentwicklung, steigende Industrieproduktion oder Unternehmenswachstum": "Wirtschaft (Wachstum)",
    "Rezession, Unternehmenskrise oder sinkende Industrieproduktion": "Wirtschaft (Krise)",
    # Finanzmärkte & Geldpolitik
    "Zinsentscheidungen, Inflation oder Währungsschwankungen": "Finanzen (Inflation)",
    "Börsennachrichten, Unternehmensgewinne oder Investitionen": "Finanzen (Börse)",
    # Handel & Außenwirtschaft
    "Zölle, Handelskonflikte oder Exportbeschränkungen": "Handel (Konflikt)",
    "Handelsabkommen, Marktöffnung oder Lieferkettenentwicklung": "Handel (Öffnung)",
    # Geopolitik & Konflikte
    "Krieg, Sanktionen, Terrorgefahr oder geopolitische Spannungen": "Geopolitik (Konflikt)",
    "Friedensgespräche, Diplomatie oder Aufhebung von Sanktionen": "Geopolitik (Frieden)",
    # Technologie & Industrie
    "Technologieentwicklung, Halbleiter, Elektromobilität oder Batterieproduktion": "Technologie (Innovation)",
    "Industriepolitik, Produktionsstandorte oder Unternehmensstrategien": "Industrie (Politik)",
    # Politik & Regulierung
    "Energiepolitik, Klimagesetze oder EU-Regulierung": "Politik (Energie)",
    "Innenpolitik, Regierungsbildung oder Wahlen": "Politik (Inland)",
    # Sonstiges
    "Sport, Unterhaltung, Kultur oder Lokalnachrichten ohne Wirtschaftsbezug": "Sonstiges",
}

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
MIN_TIMESTAMP = "2025-01-01"

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Standard figure size for all plots (width, height in inches)
DEFAULT_FIGSIZE = (12, 8)
DEFAULT_FIGSIZE_WIDE = (14, 8)  # For plots with long labels
DEFAULT_FIGSIZE_TALL = (12, 10)  # For plots with many rows
DEFAULT_DPI = 200  # Publication quality

# Font sizes for consistent styling
VIZ_TITLE_FONTSIZE = 15
VIZ_LABEL_FONTSIZE = 14
VIZ_TICK_FONTSIZE = 12
VIZ_LEGEND_FONTSIZE = 11
VIZ_ANNOTATION_FONTSIZE = 11

# Label truncation lengths
VIZ_LABEL_MAX_CHARS = 45  # Standard truncation length for y-axis labels
VIZ_LABEL_MAX_CHARS_SHORT = 35  # Shorter truncation for cramped spaces

# Cividis colormap - 10 evenly-spaced colors
_cividis = plt.cm.cividis
VIZ_COLOR_LIST = [_cividis(i / 9) for i in range(10)]  # Returns RGBA tuples

# Semantic color mappings for specific use cases (Cividis-derived)
_cividis_cmap = plt.cm.cividis
VIZ_SEMANTIC_COLORS = {
    "long": _cividis_cmap(0.0),      # Cividis start for positive/long positions
    "neutral": _cividis_cmap(0.5),   # Cividis middle for neutral
    "short": _cividis_cmap(0.85),    # Cividis position for negative/short positions (darker than 1.0)
    "train": _cividis_cmap(0.0),     # Cividis start for training metrics
    "validation": _cividis_cmap(0.7), # Cividis position for validation metrics
    "test": _cividis_cmap(0.9),      # Cividis position for test metrics
}

# Colormaps for heatmaps (Cividis)
VIZ_CMAP_SEQUENTIAL = "cividis"  # Cividis for sequential data
VIZ_CMAP_DIVERGING = "cividis"   # Cividis for diverging data

# Output directory for saved figures
FIGURES_OUTPUT_DIR = "outputs/figures"
