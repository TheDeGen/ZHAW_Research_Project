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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

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

    def __init__(self, n_splits=5, step_size=72, min_train_size=336):
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


def build_xgb_classifier(random_state: int = 42, device_config: dict | None = None, num_classes: int = 3) -> XGBClassifier:
    """
    Build an XGBoost classifier with device-specific optimizations.

    Args:
        random_state: Random seed
        device_config: Device configuration dict from detect_compute_device()
        num_classes: Number of classes (2 for binary, 3+ for multiclass)

    Returns:
        Configured XGBClassifier instance

    Note:
        When using GPU (device='cuda:0') with pandas DataFrames, you may see a warning
        about "Falling back to prediction using DMatrix due to mismatched devices."
        This is expected and benign - XGBoost automatically handles CPU->GPU data transfer
        during training and prediction. The performance impact is minimal for typical datasets.
    """
    # Configure objective and metric based on number of classes
    if num_classes == 2:
        objective = 'binary:logistic'
        eval_metric = 'logloss'
        params = {
            'objective': objective,
            'eval_metric': eval_metric,
            'enable_categorical': False,
            'random_state': random_state
        }
    else:
        objective = 'multi:softprob'
        eval_metric = 'mlogloss'
        params = {
            'objective': objective,
            'eval_metric': eval_metric,
            'num_class': num_classes,
            'enable_categorical': False,
            'random_state': random_state
        }

    if device_config is None:
        params.update({'tree_method': 'hist', 'device': 'cpu', 'n_jobs': -1})
    else:
        tree_method = device_config.get('tree_method', 'hist')
        xgb_device = device_config.get('xgb_device', 'cpu')

        # When using CUDA with pandas DataFrames, use 'hist' tree_method
        # to avoid device mismatch warnings. XGBoost will automatically
        # use GPU acceleration when device='cuda:0' is set.
        if xgb_device.startswith('cuda'):
            # Use hist tree_method which handles CPU->GPU transfer internally
            params.update({
                'tree_method': 'hist',
                'device': xgb_device,
                'n_jobs': device_config.get('n_jobs', 1)  # GPU training is serial
            })
        else:
            params.update({
                'tree_method': tree_method,
                'device': xgb_device,
                'n_jobs': device_config.get('n_jobs', -1)
            })

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
    import warnings
    from scipy import stats

    # Suppress expected XGBoost device mismatch warning when using GPU with pandas DataFrames
    # XGBoost automatically handles CPU->GPU transfer, so this warning is benign
    warnings.filterwarnings(
        'ignore',
        message='.*Falling back to prediction using DMatrix due to mismatched devices.*',
        category=UserWarning,
        module='xgboost'
    )

    if param_distributions is None:
        param_distributions = {
            'n_estimators': stats.randint(100, 401),
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
    val_df = data_dict.get('val_df')
    scaled_news_features = data_dict['scaled_news_features']
    feature_columns = baseline_features + scaled_news_features
    missing_features = [col for col in feature_columns if col not in train_df.columns]
    if missing_features:
        raise KeyError(f"Missing features in training dataframe: {missing_features}")

    X_train = train_df[feature_columns].fillna(0)
    y_train_raw = train_df[target_column].astype(int)

    # Encode target for 3-class classification: {-1, 0, 1} → {0, 1, 2}
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    num_classes = len(label_encoder.classes_)

    # Prepare validation set for early stopping
    if val_df is not None:
        X_val = val_df[feature_columns].fillna(0)
        y_val_raw = val_df[target_column].astype(int)
        y_val = label_encoder.transform(y_val_raw)
    else:
        X_val = None
        y_val = None

    print(f"Training XGBoost with {num_classes} classes: {label_encoder.classes_}")

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

    classifier = build_xgb_classifier(random_state=random_state, device_config=device_config, num_classes=num_classes)

    # Add early stopping if validation set is provided
    if X_val is not None:
        classifier.set_params(early_stopping_rounds=20)

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

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_counts[class_counts == 0] = 1
    class_weights = (len(y_train) / (num_classes * class_counts))
    sample_weights = class_weights[y_train]

    # Prepare fit parameters for early stopping
    fit_params = {'sample_weight': sample_weights}
    if X_val is not None and y_val is not None:
        val_sample_weights = class_weights[y_val]
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['sample_weight_eval_set'] = [val_sample_weights]
        fit_params['verbose'] = False

    search.fit(X_train, y_train, **fit_params)
    return search, feature_columns, label_encoder


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
        lgbm_device = device_config.get("lgbm_device")
        if lgbm_device == "gpu":
            # Avoid nested parallelism on GPU backends; one host thread is sufficient
            estimator_jobs = device_config.get("lgbm_n_jobs") or device_config.get("n_jobs") or 1
            params["n_jobs"] = max(1, estimator_jobs)
            params["device_type"] = "gpu"
        else:
            params["n_jobs"] = device_config.get("n_jobs", -1)
            params["device_type"] = "cpu"

    # Mirror LightGBM's internal thread pool with num_threads when explicitly set
    if params.get("n_jobs", -1) > 0:
        params["num_threads"] = params["n_jobs"]

    return LGBMClassifier(**params)


def run_lgbm_grid_search(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    param_grid: dict | None = None,
    cv: object | None = None,
    scoring: str = "f1_macro",
    device_config: dict | None = None,
    random_state: int = 42,
    verbose: int = 0,
) -> GridSearchCV:
    """
    Run GridSearchCV for a LightGBM classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Optional validation features for early stopping.
        y_val: Optional validation labels for early stopping.
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
    num_classes = int(np.unique(y_array).size)

    # Compute inverse-frequency sample weights (same as XGBoost)
    class_counts = np.bincount(y_array, minlength=num_classes)
    class_counts[class_counts == 0] = 1
    class_weights = (len(y_array) / (num_classes * class_counts))
    sample_weights = class_weights[y_array]

    estimator = build_lgbm_classifier(
        num_classes=num_classes,
        random_state=random_state,
        device_config=device_config,
    )

    if device_config:
        lgbm_device = device_config.get("lgbm_device")
        if lgbm_device == "gpu":
            # Serialise CV to prevent libgomp thread exhaustion when LightGBM also parallelises
            n_jobs = max(1, device_config.get("grid_search_n_jobs", 1))
            if n_jobs != 1:
                print(f"⚠ Overriding LightGBM GridSearchCV n_jobs={n_jobs} for GPU execution.")
            else:
                print("☑ Running LightGBM GridSearchCV serially to avoid GPU thread oversubscription.")
        else:
            n_jobs = device_config.get("n_jobs", -1)
    else:
        n_jobs = -1

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

    # Prepare fit parameters with early stopping if validation data provided
    fit_params = {'sample_weight': sample_weights}
    if X_val is not None and y_val is not None:
        from lightgbm import early_stopping, log_evaluation

        val_sample_weights = class_weights[y_val]
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['eval_sample_weight'] = [val_sample_weights]
        fit_params['callbacks'] = [early_stopping(stopping_rounds=20, verbose=False), log_evaluation(0)]

    grid.fit(X_train, y_train, **fit_params)
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
        best_dataset: Dictionary containing train_df, val_df, test_df, and related metadata
        best_xgb_feature_columns: Feature columns used by XGBoost
        baseline_features: List of baseline feature names shared across models
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

    # Determine how many probability columns the XGBoost model produces
    n_classes = getattr(best_xgb_model, "n_classes_", None)
    if n_classes is None and hasattr(best_xgb_model, "classes_"):
        n_classes = len(best_xgb_model.classes_)

    if n_classes is None:
        sample_df = enriched_datasets["train"]
        proba_cols = [
            col for col in sample_df.columns
            if col.startswith(f"{prediction_prefix}_prob_class")
        ]
        n_classes = len(proba_cols)

    if n_classes == 0:
        raise ValueError("Unable to determine number of classes for XGBoost probabilities.")

    prob_feature_names = [
        f"{prediction_prefix}_prob_class{i}"
        for i in range(n_classes)
    ]

    # Define feature sets
    xgb_feature_names = [f"{prediction_prefix}_pred", *prob_feature_names]

    signal_feature_columns = baseline_features + xgb_feature_names
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


def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute class weights for imbalanced datasets.

    Args:
        y_train: Training labels

    Returns:
        Dictionary mapping class labels to weights
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))


def apply_smote_resampling(X_train: pd.DataFrame, y_train: np.ndarray, random_state: int = 42):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    try:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=random_state, k_neighbors=min(5, np.bincount(y_train).min() - 1))
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")

        return X_resampled, y_resampled
    except ImportError:
        print("Warning: imbalanced-learn not installed. Install with: pip install imbalanced-learn")
        print("Returning original data without resampling.")
        return X_train, y_train


def calibrate_classifier(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    method: str = 'sigmoid',
    cv: int | str = 5
):
    """
    Calibrate classifier probabilities using validation set.

    Args:
        model: Fitted classifier
        X_train: Training features (used for CV calibration if cv != 'prefit')
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        method: Calibration method ('sigmoid' for Platt scaling or 'isotonic')
        cv: Number of CV folds or 'prefit' to use the fitted model

    Returns:
        Calibrated classifier
    """
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    print(f"\n{'='*70}")
    print(f"PROBABILITY CALIBRATION ({method.upper()})")
    print(f"{'='*70}")

    # Get uncalibrated probabilities
    if hasattr(model, "predict_proba"):
        y_val_proba_uncal = model.predict_proba(X_val)
    else:
        print("Model does not support predict_proba. Skipping calibration.")
        return model

    # Calibrate the model
    if cv == 'prefit':
        calibrated_model = CalibratedClassifierCV(model, method=method, cv='prefit')
        calibrated_model.fit(X_val, y_val)
    else:
        calibrated_model = CalibratedClassifierCV(model, method=method, cv=cv)
        calibrated_model.fit(X_train, y_train)

    # Get calibrated probabilities
    y_val_proba_cal = calibrated_model.predict_proba(X_val)

    # For binary/multiclass, compute calibration curves for the positive class
    if y_val_proba_uncal.shape[1] == 2:
        # Binary classification
        prob_true_uncal, prob_pred_uncal = calibration_curve(
            y_val, y_val_proba_uncal[:, 1], n_bins=10, strategy='uniform'
        )
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_val, y_val_proba_cal[:, 1], n_bins=10, strategy='uniform'
        )

        print(f"✓ Calibration complete (binary classification)")
    else:
        # Multiclass - show calibration for class with most samples
        print(f"✓ Calibration complete (multiclass: {y_val_proba_uncal.shape[1]} classes)")

    return calibrated_model


def print_class_wise_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_encoder=None, dataset_name: str = ""):
    """
    Print detailed per-class metrics including confusion matrix and recall.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_encoder: Optional LabelEncoder to decode class names
        dataset_name: Name of dataset for display
    """
    from sklearn.metrics import classification_report, confusion_matrix

    if dataset_name:
        print(f"\n{'='*70}")
        print(f"CLASS-WISE METRICS: {dataset_name}")
        print(f"{'='*70}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Decode class names if encoder provided
    if label_encoder is not None:
        class_labels = label_encoder.classes_
    else:
        class_labels = np.unique(y_true)

    print("\nPer-Class Recall:")
    for i, class_label in enumerate(class_labels):
        if i < len(cm):
            recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            print(f"  Class {class_label}: {recall:.3f} ({cm[i, i]}/{cm[i].sum()})")

    print("\nClassification Report:")
    target_names = [str(label) for label in class_labels]
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    # Count predictions per class
    print("\nPredictions per class:")
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    for cls, cnt in zip(unique_pred, counts_pred):
        cls_label = class_labels[cls] if label_encoder and cls < len(class_labels) else cls
        print(f"  Class {cls_label}: {cnt} predictions ({cnt/len(y_pred)*100:.1f}%)")

    print(f"{'='*70}\n")




def train_xgb_candidates(
    top_combinations: list[dict],
    preprocessed_datasets: dict,
    baseline_features: list,
    target_column: str,
    param_distributions: dict,
    n_iter: int,
    random_state: int,
    n_splits: int,
    step_size: int,
    min_train_size: int,
    device_config: dict,
    fallback_params_key: tuple | None = None,
    fallback_dataset: dict | None = None
) -> dict:
    """
    Train and evaluate XGBoost models for multiple parameter combinations.

    Args:
        top_combinations: List of dicts with 'params_key', 'dataset_name', 'lookback_window', 'decay_lambda'
        preprocessed_datasets: Dictionary mapping params_key to dataset dicts
        baseline_features: List of baseline feature names
        target_column: Name of target column
        param_distributions: Parameter distributions for XGBoost random search
        n_iter: Number of random search iterations
        random_state: Random seed
        n_splits: Number of CV splits
        step_size: Step size for expanding window
        min_train_size: Minimum training size
        device_config: Device configuration dict
        fallback_params_key: Optional fallback params key if no candidates provided
        fallback_dataset: Optional fallback dataset if no candidates provided

    Returns:
        Dictionary containing:
            - 'tuning_runs': List of tuning run summaries
            - 'best_models': Dict mapping params_key to fitted model
            - 'feature_columns': Dict mapping params_key to feature column lists
            - 'label_encoders': Dict mapping params_key to label encoders
            - 'best_run': Best run summary dict
            - 'best_params_key': Best params key
            - 'best_model': Best fitted model
            - 'best_feature_columns': Best feature columns
            - 'best_label_encoder': Best label encoder
            - 'best_dataset': Best dataset dict
    """
    from sklearn.metrics import accuracy_score, f1_score

    # Handle fallback if no candidates
    if not top_combinations:
        print("⚠ No candidates provided; falling back to sample dataset.")
        if fallback_params_key is None or fallback_dataset is None:
            raise ValueError("Fallback dataset and params_key required when no candidates provided")
        top_combinations = [{
            "params_key": fallback_params_key,
            "dataset_name": fallback_dataset["dataset_name"],
            "lookback_window": fallback_params_key[0],
            "decay_lambda": fallback_params_key[1],
        }]

    tuning_runs = []
    best_models = {}
    feature_columns_dict = {}
    label_encoders = {}

    for rank, result in enumerate(top_combinations, start=1):
        params_key = result["params_key"]
        dataset = preprocessed_datasets[params_key]

        data_dict = {
            "train_df": dataset["train_df"],
            "val_df": dataset["val_df"],  # Add validation set for early stopping
            "scaled_news_features": dataset["scaled_news_features"],
        }

        # Run XGBoost random search
        search, feature_columns, label_encoder = run_xgb_random_search(
            data_dict=data_dict,
            baseline_features=baseline_features,
            target_column=target_column,
            param_distributions=param_distributions,
            n_iter=n_iter,
            random_state=random_state,
            n_splits=n_splits,
            step_size=step_size,
            min_train_size=min_train_size,
            device_config=device_config,
        )

        best_estimator = search.best_estimator_
        val_df = dataset["val_df"]
        X_val = val_df[feature_columns].fillna(0)
        y_val = label_encoder.transform(val_df[target_column].astype(int))
        val_pred = best_estimator.predict(X_val)

        val_accuracy = accuracy_score(y_val, val_pred)
        val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)

        run_summary = {
            "rank": rank,
            "params_key": params_key,
            "dataset_name": dataset["dataset_name"],
            "lookback_window": result.get("lookback_window", params_key[0]),
            "decay_lambda": result.get("decay_lambda", params_key[1]),
            "best_cv_macro_f1": search.best_score_,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "best_params": search.best_params_,
            "feature_columns": feature_columns,
            "search": search,
            "label_encoder": label_encoder,
        }

        tuning_runs.append(run_summary)
        best_models[params_key] = best_estimator
        feature_columns_dict[params_key] = feature_columns
        label_encoders[params_key] = label_encoder

        print(
            f"#{rank} {dataset['dataset_name']} → "
            f"CV F1={search.best_score_:.3f}, "
            f"Val Acc={val_accuracy:.3f}, "
            f"Val Macro-F1={val_macro_f1:.3f}"
        )

    # Select best run
    best_run = max(tuning_runs, key=lambda r: (r["best_cv_macro_f1"], r["val_macro_f1"]))
    best_params_key = best_run["params_key"]
    best_model = best_models[best_params_key]
    best_feature_columns = feature_columns_dict[best_params_key]
    best_label_encoder = label_encoders[best_params_key]
    best_dataset = preprocessed_datasets[best_params_key]

    print(
        f"\n✓ Selected XGBoost dataset {best_run['dataset_name']} "
        f"(lookback={best_run['lookback_window']}h, lambda={best_run['decay_lambda']})"
    )
    print(f"  Best CV macro-F1: {best_run['best_cv_macro_f1']:.3f}")
    print(f"  Validation macro-F1: {best_run['val_macro_f1']:.3f}")
    print(f"  XGBoost classes: {best_label_encoder.classes_}")

    # Calibrate best model using validation set (Platt scaling)
    print(f"\n✓ Calibrating XGBoost probabilities using validation set...")
    val_df = best_dataset["val_df"]
    X_val = val_df[best_feature_columns].fillna(0)
    y_val = best_label_encoder.transform(val_df[target_column].astype(int))

    # Keep reference to uncalibrated model for learning curves, etc.
    uncalibrated_best_model = best_model

    calibrated_best_model = calibrate_classifier(
        model=best_model,
        X_train=None,  # Not needed for cv='prefit'
        y_train=None,
        X_val=X_val,
        y_val=y_val,
        method='sigmoid',
        cv='prefit'
    )

    # Use calibrated model as default best_model
    best_model = calibrated_best_model
    print(f"  Calibration complete (Platt scaling)")

    return {
        "tuning_runs": tuning_runs,
        "best_models": best_models,
        "feature_columns": feature_columns_dict,
        "label_encoders": label_encoders,
        "best_run": best_run,
        "best_params_key": best_params_key,
        "best_model": best_model,  # Calibrated model (for LightGBM)
        "best_model_uncalibrated": uncalibrated_best_model,  # Uncalibrated for learning curves
        "best_feature_columns": best_feature_columns,
        "best_label_encoder": best_label_encoder,
        "best_dataset": best_dataset,
    }


def evaluate_xgb_test_set(
    model,
    test_df: pd.DataFrame,
    feature_columns: list,
    target_column: str,
    label_encoder,
    model_name: str = "XGBoost"
) -> dict:
    """
    Evaluate XGBoost model on test set and return predictions and metrics.

    Args:
        model: Fitted XGBoost model (calibrated or uncalibrated)
        test_df: Test dataframe
        feature_columns: List of feature column names
        target_column: Name of target column
        label_encoder: Label encoder for target classes
        model_name: Name for display purposes

    Returns:
        Dictionary containing:
            - 'y_test': Encoded test labels
            - 'y_pred': Predicted labels
            - 'y_pred_proba': Prediction probabilities
            - 'X_test': Test feature matrix
            - 'accuracy': Test accuracy
            - 'macro_f1': Test macro F1 score
    """
    from sklearn.metrics import accuracy_score, f1_score

    X_test = test_df[feature_columns].fillna(0)
    y_test_raw = test_df[target_column].astype(int)
    y_test = label_encoder.transform(y_test_raw)

    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n✓ {model_name} Test Evaluation (3-class)")
    print(f"  Argmax predictions → Acc={accuracy:.4f}, Macro-F1={macro_f1:.4f}")

    # Display detailed class-wise metrics
    print_class_wise_metrics(
        y_true=y_test,
        y_pred=y_pred,
        label_encoder=label_encoder,
        dataset_name=f"{model_name} Test Set"
    )

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "X_test": X_test,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def train_lightgbm_pair(
    signal_train_df: pd.DataFrame,
    signal_val_df: pd.DataFrame,
    signal_test_df: pd.DataFrame,
    signal_feature_columns: list,
    baseline_feature_columns: list,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
    param_grid: dict,
    n_cv_splits: int,
    cv_step_size: int,
    cv_min_train_size: int,
    device_config: dict,
    random_state: int = 42
) -> dict:
    """
    Train both LightGBM signal and baseline models.

    This consolidates the training of signal (baseline + XGBoost prediction features) and baseline (price-only)
    models, handling feature sanitization, CV setup, and grid search for both.

    Args:
        signal_train_df: Training dataframe enriched with XGBoost prediction features
        signal_val_df: Validation dataframe enriched with XGBoost prediction features
        signal_test_df: Test dataframe enriched with XGBoost prediction features
        signal_feature_columns: Signal model feature columns (baseline + XGBoost predictions)
        baseline_feature_columns: Baseline model feature columns (shared baseline set)
        y_train: Training labels (encoded)
        y_val: Validation labels (encoded)
        y_test: Test labels (encoded)
        label_encoder: Label encoder for targets
        param_grid: Parameter grid for LightGBM
        n_cv_splits: Number of CV splits
        cv_step_size: CV step size in hours
        cv_min_train_size: Minimum training size for CV
        device_config: Device configuration dict
        random_state: Random seed

    Returns:
        Dictionary containing:
            - 'signal_model': Fitted signal LightGBM model
            - 'baseline_model': Fitted baseline LightGBM model
            - 'signal_grid': Signal GridSearchCV object
            - 'baseline_grid': Baseline GridSearchCV object
            - 'signal_column_rename_map': Signal feature sanitization mapping
            - 'baseline_column_rename_map': Baseline feature sanitization mapping
            - 'evaluation': Evaluation artifacts from evaluate_lgbm_models
    """
    import copy

    print(f"\n{'='*70}")
    print("TRAINING LIGHTGBM SIGNAL & BASELINE MODELS")
    print(f"{'='*70}")

    # Sanitize feature names for both models
    signal_feature_columns_sanitized, _ = sanitize_feature_names(signal_feature_columns)
    baseline_feature_columns_sanitized, _ = sanitize_feature_names(baseline_feature_columns)

    signal_column_rename_map = dict(zip(signal_feature_columns, signal_feature_columns_sanitized))
    baseline_column_rename_map = dict(zip(baseline_feature_columns, baseline_feature_columns_sanitized))

    # --- Train Signal Model ---
    print("Training SIGNAL model (baseline features + XGBoost predictions)...")
    train_signal_X = signal_train_df[signal_feature_columns].fillna(0).rename(columns=signal_column_rename_map)

    # CV splitter for signal model
    base_signal_splitter = ExpandingWindowSplitter(
        n_splits=n_cv_splits,
        step_size=cv_step_size,
        min_train_size=cv_min_train_size,
    )
    available_signal_splits = base_signal_splitter.get_n_splits(train_signal_X)
    if available_signal_splits == 0:
        raise ValueError("Insufficient data for LightGBM signal cross-validation.")

    signal_cv = ExpandingWindowSplitter(
        n_splits=min(n_cv_splits, available_signal_splits),
        step_size=cv_step_size,
        min_train_size=cv_min_train_size,
    )

    # Prepare validation data for early stopping
    val_signal_X = signal_val_df[signal_feature_columns].fillna(0).rename(columns=signal_column_rename_map)

    signal_grid = run_lgbm_grid_search(
        X_train=train_signal_X,
        y_train=y_train,
        X_val=val_signal_X,
        y_val=y_val,
        param_grid=copy.deepcopy(param_grid),
        cv=signal_cv,
        device_config=device_config,
        random_state=random_state,
    )

    signal_model = signal_grid.best_estimator_
    print(f"✓ Signal model trained - Best CV macro-F1: {signal_grid.best_score_:.3f}")
    print(f"  Best params: {signal_grid.best_params_}")

    # --- Train Baseline Model ---
    print("\nTraining BASELINE model (price/temporal features only)...")
    train_baseline_X = signal_train_df[baseline_feature_columns].fillna(0).rename(columns=baseline_column_rename_map)

    # CV splitter for baseline model
    base_baseline_splitter = ExpandingWindowSplitter(
        n_splits=n_cv_splits,
        step_size=cv_step_size,
        min_train_size=cv_min_train_size,
    )
    available_baseline_splits = base_baseline_splitter.get_n_splits(train_baseline_X)
    if available_baseline_splits == 0:
        raise ValueError("Insufficient data for LightGBM baseline cross-validation.")

    baseline_cv = ExpandingWindowSplitter(
        n_splits=min(n_cv_splits, available_baseline_splits),
        step_size=cv_step_size,
        min_train_size=cv_min_train_size,
    )

    # Prepare validation data for early stopping
    val_baseline_X = signal_val_df[baseline_feature_columns].fillna(0).rename(columns=baseline_column_rename_map)

    baseline_grid = run_lgbm_grid_search(
        X_train=train_baseline_X,
        y_train=y_train,
        X_val=val_baseline_X,
        y_val=y_val,
        param_grid=copy.deepcopy(param_grid),
        cv=baseline_cv,
        device_config=device_config,
        random_state=random_state,
    )

    baseline_model = baseline_grid.best_estimator_
    print(f"✓ Baseline model trained - Best CV macro-F1: {baseline_grid.best_score_:.3f}")
    print(f"  Best params: {baseline_grid.best_params_}")

    # --- Evaluate Both Models ---
    print("\nEvaluating both models on validation and test sets...")
    evaluation = evaluate_lgbm_models(
        signal_model=signal_model,
        baseline_model=baseline_model,
        signal_feature_columns=signal_feature_columns,
        baseline_feature_columns=baseline_feature_columns,
        val_df=signal_val_df,
        test_df=signal_test_df,
        y_val=y_val,
        y_test=y_test,
        label_encoder=label_encoder,
        signal_column_rename_map=signal_column_rename_map,
        baseline_column_rename_map=baseline_column_rename_map,
    )

    print(f"{'='*70}\n")

    return {
        "signal_model": signal_model,
        "baseline_model": baseline_model,
        "signal_grid": signal_grid,
        "baseline_grid": baseline_grid,
        "signal_column_rename_map": signal_column_rename_map,
        "baseline_column_rename_map": baseline_column_rename_map,
        "evaluation": evaluation,
    }
