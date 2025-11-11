"""
Model Utilities
===============
XGBoost and LightGBM model training utilities.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


class ExpandingWindowSplitter:
    """
    Custom expanding-window splitter with fixed step size for time series CV.

    Args:
        n_splits: Number of splits
        step_size: Step size in hours between splits
        min_train_size: Minimum training size in hours
    """

    def __init__(self, n_splits=5, step_size=24, min_train_size=336):
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        if step_size < 1:
            raise ValueError("step_size must be at least 1")
        if min_train_size < 1:
            raise ValueError("min_train_size must be at least 1")

        self.n_splits = n_splits
        self.step_size = step_size
        self.min_train_size = min_train_size

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None:
            return self.n_splits
        n_samples = len(X)
        if n_samples <= self.min_train_size:
            return 0
        possible_splits = (n_samples - self.min_train_size) // self.step_size
        return max(0, min(self.n_splits, possible_splits))

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        effective_splits = self.get_n_splits(X)
        if effective_splits < self.n_splits:
            raise ValueError(
                f"Requested {self.n_splits} splits, but only {effective_splits} are possible with "
                f"{n_samples} samples. Consider reducing n_splits or the step_size/min_train_size."
            )

        train_end = self.min_train_size
        for split_idx in range(self.n_splits):
            test_start = train_end
            test_end = test_start + self.step_size
            if test_end > n_samples:
                raise ValueError(
                    f"Split {split_idx} exceeds available samples ({n_samples}). "
                    "Try reducing n_splits or step_size."
                )

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            yield train_indices, test_indices

            train_end = test_end


def build_xgb_classifier(random_state: int = 42, device_config: dict | None = None) -> XGBClassifier:
    """
    Build an XGBoost classifier with device-specific optimizations.

    Args:
        random_state: Random seed
        device_config: Device configuration dict from detect_compute_device()

    Returns:
        Configured XGBClassifier instance
    """
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'enable_categorical': False,
        'random_state': random_state
    }

    if device_config is None:
        params.update({'tree_method': 'hist', 'predictor': 'auto', 'n_jobs': -1})
    else:
        params.update({
            'tree_method': device_config.get('tree_method', 'hist'),
            'predictor': device_config.get('predictor', 'auto'),
            'n_jobs': device_config.get('n_jobs', -1)
        })

    return XGBClassifier(**params)


def map_target_to_binary(y: pd.Series) -> np.ndarray:
    """
    Map target variable to binary (0/1) for XGBoost binary classification.

    Args:
        y: Target series with values in {-1, 0, 1}

    Returns:
        Binary array where positive class (1) is mapped to 1, others to 0
    """
    unique_values = set(np.unique(y))
    unexpected_values = unique_values - {-1, 0, 1}
    if unexpected_values:
        raise ValueError(f"Unexpected target values encountered: {unexpected_values}")
    return np.where(y > 0, 1, 0)


def run_xgb_random_search(
    data_dict: dict,
    baseline_features: list,
    target_column: str,
    param_distributions: dict | None = None,
    n_iter: int = 40,
    random_state: int = 42,
    n_splits: int = 4,
    step_size: int = 12,
    min_train_size: int = 336,
    device_config: dict | None = None
):
    """
    GPU-aware XGBoost random search wrapper.

    Args:
        data_dict: Dictionary containing train_df and scaled_news_features
        baseline_features: List of baseline feature names
        target_column: Name of target column
        param_distributions: Parameter distributions for random search
        n_iter: Number of iterations
        random_state: Random seed
        n_splits: Number of CV splits
        step_size: Step size for expanding window
        min_train_size: Minimum training size
        device_config: Device configuration dict

    Returns:
        Tuple of (fitted RandomizedSearchCV object, feature_columns list)
    """
    from scipy import stats

    if param_distributions is None:
        param_distributions = {
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

    train_df = data_dict['train_df']
    scaled_news_features = data_dict['scaled_news_features']
    feature_columns = baseline_features + scaled_news_features
    missing_features = [col for col in feature_columns if col not in train_df.columns]
    if missing_features:
        raise KeyError(f"Missing features in training dataframe: {missing_features}")

    X_train = train_df[feature_columns].fillna(0)
    y_train_raw = train_df[target_column].astype(int)
    y_train = map_target_to_binary(y_train_raw)

    splitter = ExpandingWindowSplitter(
        n_splits=n_splits,
        step_size=step_size,
        min_train_size=min_train_size
    )

    effective_splits = splitter.get_n_splits(X_train)
    if effective_splits < n_splits:
        raise ValueError(
            f"Only {effective_splits} expanding-window splits available. Adjust n_splits, step_size, "
            f"or min_train_size."
        )

    classifier = build_xgb_classifier(random_state=random_state, device_config=device_config)

    if device_config is None:
        search_n_jobs = -1
    elif device_config.get('device') == 'cuda':
        search_n_jobs = 1
    else:
        search_n_jobs = device_config.get('n_jobs', -1)

    if device_config and device_config.get('device') == 'cuda':
        print("Running RandomizedSearchCV with CUDA-accelerated XGBoost (serial CV fits).")
    elif device_config and device_config.get('device') == 'mps':
        print("MPS detected: XGBoost uses CPU hist; parallel CV remains enabled.")
    else:
        print(f"Using CPU-optimised XGBoost with parallel CV fits (n_jobs={search_n_jobs}).")

    search = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='f1_macro',
        cv=splitter,
        n_jobs=search_n_jobs,
        random_state=random_state,
        verbose=1,
        refit=True,
        return_train_score=True
    )

    search.fit(X_train, y_train)
    return search, feature_columns
