"""
Model Configuration
===================
Contains model-specific hyperparameters and configurations.
"""

# ============================================================================
# XGBOOST CONFIGURATION
# ============================================================================

XGBOOST_BASE_PARAMS = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 3,
    'enable_categorical': False,
}

# Default parameters for XGBoost when not using grid search
XGBOOST_DEFAULT_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.0,
    'min_child_weight': 1,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
}

# ============================================================================
# LIGHTGBM CONFIGURATION
# ============================================================================

LIGHTGBM_BASE_PARAMS = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'is_unbalance': True,  # Handle class imbalance
}

# Grid search parameter ranges for LightGBM
LIGHTGBM_PARAM_GRID = {
    'num_leaves': [31, 50, 100],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100],
    'min_child_samples': [20],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
}

# ============================================================================
# RIDGE CLASSIFIER CONFIGURATION
# ============================================================================

RIDGE_CV_SCORING = 'accuracy'
RIDGE_MAX_ITER = 10000
