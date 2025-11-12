"""
Model Utilities
===============
XGBoost and LightGBM model training utilities.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from config import model_config


def sanitize_feature_names(feature_names):
    """
    Sanitize feature names to be JSON-compatible for LightGBM.

    Replaces German umlauts and special characters that cause
    'Do not support special JSON characters in feature name' errors.

    Args:
        feature_names: List or pandas Index of feature names

    Returns:
        List of sanitized feature names and dictionary mapping sanitized -> original
    """
    import unicodedata
    import re

    sanitized_names = []
    name_mapping = {}

    for name in feature_names:
        # Convert to string if needed
        name_str = str(name)

        # Replace common German characters
        replacements = {
            'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
            'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
            'ß': 'ss'
        }
        sanitized = name_str
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)

        # Normalize unicode characters to ASCII equivalents
        sanitized = unicodedata.normalize('NFKD', sanitized)
        sanitized = sanitized.encode('ascii', 'ignore').decode('ascii')

        # Replace remaining special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '_', sanitized)

        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')

        sanitized_names.append(sanitized)
        name_mapping[sanitized] = name_str

    return sanitized_names, name_mapping


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
        params.update({'tree_method': 'hist', 'n_jobs': -1})
    else:
        tree_method = device_config.get('tree_method', 'hist')
        device_type = device_config.get('device', 'cpu')

        params.update({
            'tree_method': tree_method,
            'n_jobs': device_config.get('n_jobs', -1)
        })

        # Only add predictor parameter for GPU training (not for CPU hist or MPS)
        if tree_method == 'gpu_hist' and device_type == 'cuda':
            params['predictor'] = device_config.get('predictor', 'gpu_predictor')

    return XGBClassifier(**params)


def enrich_with_model_predictions(
    model,
    dataframes: dict[str, pd.DataFrame],
    feature_columns: list[str],
    prediction_prefix: str = "model"
) -> dict[str, pd.DataFrame]:
    """
    Enrich dataframes with model predictions as additional features.

    Args:
        model: Fitted model with predict_proba method
        dataframes: Dict mapping dataset name (e.g., 'train', 'val', 'test') to DataFrame
        feature_columns: List of feature columns to use for prediction
        prediction_prefix: Prefix for new prediction columns

    Returns:
        Dict of enriched DataFrames with same keys as input
    """
    enriched = {}

    for name, df in dataframes.items():
        X = df[feature_columns].fillna(0)
        proba = model.predict_proba(X)

        df_enriched = df.copy()

        # Add probability columns for each class
        for i in range(proba.shape[1]):
            df_enriched[f"{prediction_prefix}_prob_class{i}"] = proba[:, i]

        # Add predicted class
        df_enriched[f"{prediction_prefix}_pred"] = proba.argmax(axis=1)

        enriched[name] = df_enriched

    return enriched


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


def build_lgbm_classifier(
    num_classes: int = 3,
    random_state: int = 42,
    device_config: dict | None = None,
    base_params: dict | None = None,
) -> LGBMClassifier:
    """
    Build a LightGBM classifier configured for CPU/GPU execution.

    Args:
        num_classes: Number of target classes.
        random_state: Random seed for reproducibility.
        device_config: Optional device configuration dict.
        base_params: Optional base parameter overrides.

    Returns:
        Configured LGBMClassifier.
    """
    params = (base_params or model_config.LIGHTGBM_BASE_PARAMS).copy()
    params.update({
        "num_class": num_classes,
        "random_state": random_state,
    })

    if device_config is None:
        params.setdefault("n_jobs", -1)
        params.setdefault("device_type", "cpu")
    else:
        params["n_jobs"] = device_config.get("n_jobs", -1)
        params["device_type"] = "gpu" if device_config.get("lgbm_device") == "gpu" else "cpu"

    return LGBMClassifier(**params)


def run_lgbm_grid_search(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    param_grid: dict | None = None,
    cv: object | None = None,
    scoring: str = "f1_macro",
    device_config: dict | None = None,
    random_state: int = 42,
    verbose: int = 1,
) -> GridSearchCV:
    """
    Run GridSearchCV for a LightGBM classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        param_grid: Parameter grid for GridSearchCV.
        cv: Cross-validation splitter.
        scoring: Scoring metric.
        device_config: Optional device configuration dict.
        random_state: Random seed.
        verbose: Verbosity level for GridSearchCV.

    Returns:
        Fitted GridSearchCV object.
    """
    param_grid = param_grid or model_config.LIGHTGBM_PARAM_GRID
    y_array = np.asarray(y_train)
    estimator = build_lgbm_classifier(
        num_classes=int(np.unique(y_array).size),
        random_state=random_state,
        device_config=device_config,
    )

    n_jobs = device_config.get("n_jobs", -1) if device_config else -1

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )

    grid.fit(X_train, y_train)
    return grid


def prepare_lgbm_datasets(
    best_xgb_model,
    best_dataset: dict,
    best_xgb_feature_columns: list,
    baseline_features: list,
    target_column: str,
    prediction_prefix: str = "xgb"
) -> dict:
    """
    Prepare datasets for LightGBM training by enriching with XGBoost predictions
    and setting up feature columns.

    Args:
        best_xgb_model: Fitted XGBoost model
        best_dataset: Dictionary containing train_df, val_df, test_df, scaled_news_features
        best_xgb_feature_columns: Feature columns used by XGBoost
        baseline_features: List of baseline feature names
        target_column: Name of target column
        prediction_prefix: Prefix for XGBoost prediction features

    Returns:
        Dictionary containing enriched datasets and feature columns
    """
    # Enrich datasets with XGBoost predictions
    enriched_datasets = enrich_with_model_predictions(
        model=best_xgb_model,
        dataframes={
            "train": best_dataset["train_df"],
            "val": best_dataset["val_df"],
            "test": best_dataset["test_df"]
        },
        feature_columns=best_xgb_feature_columns,
        prediction_prefix=prediction_prefix
    )

    # Define feature sets
    xgb_feature_names = [
        f"{prediction_prefix}_prob_class0",
        f"{prediction_prefix}_prob_class1",
        f"{prediction_prefix}_pred"
    ]

    scaled_news_features = best_dataset["scaled_news_features"]
    signal_feature_columns = baseline_features + scaled_news_features + xgb_feature_names
    baseline_feature_columns = baseline_features.copy()

    return {
        "train_df": enriched_datasets["train"],
        "val_df": enriched_datasets["val"],
        "test_df": enriched_datasets["test"],
        "signal_feature_columns": signal_feature_columns,
        "baseline_feature_columns": baseline_feature_columns,
        "xgb_feature_names": xgb_feature_names,
    }


def prepare_lgbm_targets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str):
    """
    Prepare and encode targets for LightGBM training.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        target_column: Name of target column

    Returns:
        Dictionary containing encoded targets and label encoder
    """
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[target_column].astype(int))

    y_train = label_encoder.transform(train_df[target_column].astype(int))
    y_val = label_encoder.transform(val_df[target_column].astype(int))
    y_test = label_encoder.transform(test_df[target_column].astype(int))

    return {
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_encoder": label_encoder,
    }


def evaluate_lgbm_models(
    signal_model,
    baseline_model,
    signal_feature_columns: list,
    baseline_feature_columns: list,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
    signal_column_rename_map: dict,
    baseline_column_rename_map: dict,
):
    """
    Evaluate LightGBM signal and baseline models on validation and test sets.

    Args:
        signal_model: Fitted signal model
        baseline_model: Fitted baseline model
        signal_feature_columns: Signal model feature columns
        baseline_feature_columns: Baseline model feature columns
        val_df: Validation dataframe
        test_df: Test dataframe
        y_val: Validation targets (encoded)
        y_test: Test targets (encoded)
        label_encoder: Label encoder used for targets
        signal_column_rename_map: Mapping for signal feature sanitization
        baseline_column_rename_map: Mapping for baseline feature sanitization

    Returns:
        Dictionary containing predictions, probabilities, and artifacts
    """
    from sklearn.metrics import f1_score, accuracy_score

    # Prepare data with sanitized column names
    val_signal_X = val_df[signal_feature_columns].fillna(0).rename(columns=signal_column_rename_map)
    val_baseline_X = val_df[baseline_feature_columns].fillna(0).rename(columns=baseline_column_rename_map)
    test_signal_X = test_df[signal_feature_columns].fillna(0).rename(columns=signal_column_rename_map)
    test_baseline_X = test_df[baseline_feature_columns].fillna(0).rename(columns=baseline_column_rename_map)

    # Make predictions
    signal_val_pred = signal_model.predict(val_signal_X)
    signal_val_proba = signal_model.predict_proba(val_signal_X)
    signal_test_pred = signal_model.predict(test_signal_X)
    signal_test_proba = signal_model.predict_proba(test_signal_X)

    baseline_val_pred = baseline_model.predict(val_baseline_X)
    baseline_val_proba = baseline_model.predict_proba(val_baseline_X)
    baseline_test_pred = baseline_model.predict(test_baseline_X)
    baseline_test_proba = baseline_model.predict_proba(test_baseline_X)

    # Calculate validation metrics
    val_signal_macro_f1 = f1_score(y_val, signal_val_pred, average="macro", zero_division=0)
    val_signal_accuracy = accuracy_score(y_val, signal_val_pred)
    val_baseline_macro_f1 = f1_score(y_val, baseline_val_pred, average="macro", zero_division=0)
    val_baseline_accuracy = accuracy_score(y_val, baseline_val_pred)

    print("✓ LightGBM validation performance")
    print(f"  Signal model  → Acc={val_signal_accuracy:.3f}, Macro-F1={val_signal_macro_f1:.3f}")
    print(f"  Baseline model → Acc={val_baseline_accuracy:.3f}, Macro-F1={val_baseline_macro_f1:.3f}")

    # Return artifacts
    return {
        "signal": {
            "model": signal_model,
            "feature_columns": signal_feature_columns,
            "val_pred": signal_val_pred,
            "val_proba": signal_val_proba,
            "test_pred": signal_test_pred,
            "test_proba": signal_test_proba,
            "test_X": test_signal_X,
        },
        "baseline": {
            "model": baseline_model,
            "feature_columns": baseline_feature_columns,
            "val_pred": baseline_val_pred,
            "val_proba": baseline_val_proba,
            "test_pred": baseline_test_pred,
            "test_proba": baseline_test_proba,
            "test_X": test_baseline_X,
        },
        "label_encoder": label_encoder,
    }
