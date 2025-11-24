"""
Save Models Utility
===================
Helper functions to save and load trained models from the pipeline.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def save_model_artifacts(
    output_dir: str = "../outputs/models",
    best_xgb_model=None,
    best_xgb_feature_columns=None,
    best_xgb_label_encoder=None,
    best_dataset=None,
    signal_best_lgbm=None,
    baseline_best_lgbm=None,
    signal_feature_columns=None,
    baseline_feature_columns=None,
    label_encoder=None,
    signal_test_df=None,
    **kwargs
):
    """
    Save trained models and associated artifacts.

    Args:
        output_dir: Directory to save models
        best_xgb_model: Trained XGBoost model
        best_xgb_feature_columns: Feature columns for XGBoost
        best_xgb_label_encoder: Label encoder for XGBoost
        best_dataset: Dataset dictionary with train/val/test splits
        signal_best_lgbm: Trained LightGBM signal model
        baseline_best_lgbm: Trained LightGBM baseline model
        signal_feature_columns: Feature columns for signal model
        baseline_feature_columns: Feature columns for baseline model
        label_encoder: Label encoder for LightGBM
        signal_test_df: Test dataframe
        **kwargs: Additional artifacts to save

    Returns:
        Dictionary with paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    saved_files = {}

    # Save XGBoost artifacts
    if best_xgb_model is not None:
        xgb_path = output_path / "best_xgb_model.pkl"
        with open(xgb_path, 'wb') as f:
            pickle.dump(best_xgb_model, f)
        saved_files['xgb_model'] = str(xgb_path)
        print(f"✓ Saved XGBoost model to {xgb_path}")

    if best_xgb_feature_columns is not None:
        xgb_features_path = output_path / "best_xgb_feature_columns.pkl"
        with open(xgb_features_path, 'wb') as f:
            pickle.dump(best_xgb_feature_columns, f)
        saved_files['xgb_features'] = str(xgb_features_path)
        print(f"✓ Saved XGBoost features to {xgb_features_path}")

    if best_xgb_label_encoder is not None:
        xgb_encoder_path = output_path / "best_xgb_label_encoder.pkl"
        with open(xgb_encoder_path, 'wb') as f:
            pickle.dump(best_xgb_label_encoder, f)
        saved_files['xgb_encoder'] = str(xgb_encoder_path)
        print(f"✓ Saved XGBoost label encoder to {xgb_encoder_path}")

    # Save LightGBM artifacts
    if signal_best_lgbm is not None:
        lgbm_signal_path = output_path / "signal_best_lgbm.pkl"
        with open(lgbm_signal_path, 'wb') as f:
            pickle.dump(signal_best_lgbm, f)
        saved_files['lgbm_signal'] = str(lgbm_signal_path)
        print(f"✓ Saved LightGBM signal model to {lgbm_signal_path}")

    if baseline_best_lgbm is not None:
        lgbm_baseline_path = output_path / "baseline_best_lgbm.pkl"
        with open(lgbm_baseline_path, 'wb') as f:
            pickle.dump(baseline_best_lgbm, f)
        saved_files['lgbm_baseline'] = str(lgbm_baseline_path)
        print(f"✓ Saved LightGBM baseline model to {lgbm_baseline_path}")

    if signal_feature_columns is not None:
        signal_features_path = output_path / "signal_feature_columns.pkl"
        with open(signal_features_path, 'wb') as f:
            pickle.dump(signal_feature_columns, f)
        saved_files['signal_features'] = str(signal_features_path)
        print(f"✓ Saved signal features to {signal_features_path}")

    if baseline_feature_columns is not None:
        baseline_features_path = output_path / "baseline_feature_columns.pkl"
        with open(baseline_features_path, 'wb') as f:
            pickle.dump(baseline_feature_columns, f)
        saved_files['baseline_features'] = str(baseline_features_path)
        print(f"✓ Saved baseline features to {baseline_features_path}")

    if label_encoder is not None:
        encoder_path = output_path / "label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        saved_files['encoder'] = str(encoder_path)
        print(f"✓ Saved label encoder to {encoder_path}")

    # Save test data
    if best_dataset is not None and 'test_df' in best_dataset:
        test_df = best_dataset['test_df']
        test_path = output_path / "test_data_xgb.parquet"
        test_df.to_parquet(test_path)
        saved_files['xgb_test_data'] = str(test_path)
        print(f"✓ Saved XGBoost test data to {test_path}")

    if signal_test_df is not None:
        signal_test_path = output_path / "test_data_lgbm.parquet"
        signal_test_df.to_parquet(signal_test_path)
        saved_files['lgbm_test_data'] = str(signal_test_path)
        print(f"✓ Saved LightGBM test data to {signal_test_path}")

    # Save any additional artifacts
    for key, value in kwargs.items():
        artifact_path = output_path / f"{key}.pkl"
        with open(artifact_path, 'wb') as f:
            pickle.dump(value, f)
        saved_files[key] = str(artifact_path)
        print(f"✓ Saved {key} to {artifact_path}")

    # Save manifest
    manifest_path = output_path / "manifest.pkl"
    with open(manifest_path, 'wb') as f:
        pickle.dump(saved_files, f)
    print(f"\n✓ Saved manifest to {manifest_path}")
    print(f"\nTotal files saved: {len(saved_files)}")

    return saved_files


def load_model_artifacts(
    output_dir: str = "../outputs/models",
    load_xgb: bool = True,
    load_lgbm: bool = True,
    load_test_data: bool = True
):
    """
    Load saved models and artifacts.

    Args:
        output_dir: Directory containing saved models
        load_xgb: Whether to load XGBoost artifacts
        load_lgbm: Whether to load LightGBM artifacts
        load_test_data: Whether to load test data

    Returns:
        Dictionary with loaded artifacts
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        raise FileNotFoundError(f"Model directory not found: {output_dir}")

    # Load manifest
    manifest_path = output_path / "manifest.pkl"
    if manifest_path.exists():
        with open(manifest_path, 'rb') as f:
            manifest = pickle.load(f)
        print(f"✓ Loaded manifest from {manifest_path}")
        print(f"  Available files: {len(manifest)}")
    else:
        print("⚠ No manifest found, will attempt to load files directly")
        manifest = {}

    artifacts = {}

    # Load XGBoost artifacts
    if load_xgb:
        xgb_model_path = output_path / "best_xgb_model.pkl"
        if xgb_model_path.exists():
            with open(xgb_model_path, 'rb') as f:
                artifacts['best_xgb_model'] = pickle.load(f)
            print(f"✓ Loaded XGBoost model from {xgb_model_path}")

        xgb_features_path = output_path / "best_xgb_feature_columns.pkl"
        if xgb_features_path.exists():
            with open(xgb_features_path, 'rb') as f:
                artifacts['best_xgb_feature_columns'] = pickle.load(f)
            print(f"✓ Loaded XGBoost features from {xgb_features_path}")

        xgb_encoder_path = output_path / "best_xgb_label_encoder.pkl"
        if xgb_encoder_path.exists():
            with open(xgb_encoder_path, 'rb') as f:
                artifacts['best_xgb_label_encoder'] = pickle.load(f)
            print(f"✓ Loaded XGBoost label encoder from {xgb_encoder_path}")

    # Load LightGBM artifacts
    if load_lgbm:
        lgbm_signal_path = output_path / "signal_best_lgbm.pkl"
        if lgbm_signal_path.exists():
            with open(lgbm_signal_path, 'rb') as f:
                artifacts['signal_best_lgbm'] = pickle.load(f)
            print(f"✓ Loaded LightGBM signal model from {lgbm_signal_path}")

        lgbm_baseline_path = output_path / "baseline_best_lgbm.pkl"
        if lgbm_baseline_path.exists():
            with open(lgbm_baseline_path, 'rb') as f:
                artifacts['baseline_best_lgbm'] = pickle.load(f)
            print(f"✓ Loaded LightGBM baseline model from {lgbm_baseline_path}")

        signal_features_path = output_path / "signal_feature_columns.pkl"
        if signal_features_path.exists():
            with open(signal_features_path, 'rb') as f:
                artifacts['signal_feature_columns'] = pickle.load(f)
            print(f"✓ Loaded signal features from {signal_features_path}")

        baseline_features_path = output_path / "baseline_feature_columns.pkl"
        if baseline_features_path.exists():
            with open(baseline_features_path, 'rb') as f:
                artifacts['baseline_feature_columns'] = pickle.load(f)
            print(f"✓ Loaded baseline features from {baseline_features_path}")

        encoder_path = output_path / "label_encoder.pkl"
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                artifacts['label_encoder'] = pickle.load(f)
            print(f"✓ Loaded label encoder from {encoder_path}")

    # Load test data
    if load_test_data:
        xgb_test_path = output_path / "test_data_xgb.parquet"
        if xgb_test_path.exists():
            artifacts['test_data_xgb'] = pd.read_parquet(xgb_test_path)
            print(f"✓ Loaded XGBoost test data from {xgb_test_path}")

        lgbm_test_path = output_path / "test_data_lgbm.parquet"
        if lgbm_test_path.exists():
            artifacts['test_data_lgbm'] = pd.read_parquet(lgbm_test_path)
            print(f"✓ Loaded LightGBM test data from {lgbm_test_path}")

    print(f"\n✓ Loaded {len(artifacts)} artifacts")

    return artifacts


def create_test_datasets(artifacts):
    """
    Create X_test, y_test datasets from loaded artifacts.

    Args:
        artifacts: Dictionary from load_model_artifacts()

    Returns:
        Dictionary with prepared datasets
    """
    datasets = {}

    # XGBoost test dataset
    if 'test_data_xgb' in artifacts and 'best_xgb_feature_columns' in artifacts:
        test_df = artifacts['test_data_xgb']
        feature_cols = artifacts['best_xgb_feature_columns']

        X_test_xgb = test_df[feature_cols].fillna(0)

        # Try to get target column
        target_col = 'spread_target_shift_24'
        if target_col in test_df.columns:
            y_test_xgb = test_df[target_col].values

            # Encode if encoder available
            if 'best_xgb_label_encoder' in artifacts:
                y_test_xgb = artifacts['best_xgb_label_encoder'].transform(y_test_xgb.astype(int))
        else:
            y_test_xgb = None

        datasets['X_test_xgb'] = X_test_xgb
        datasets['y_test_xgb'] = y_test_xgb
        print(f"✓ Prepared XGBoost test dataset: {X_test_xgb.shape}")

    # LightGBM test dataset
    if 'test_data_lgbm' in artifacts and 'signal_feature_columns' in artifacts:
        test_df = artifacts['test_data_lgbm']
        feature_cols = artifacts['signal_feature_columns']

        X_test_lgbm = test_df[feature_cols].fillna(0)

        # Try to get target column
        target_col = 'spread_target_shift_24'
        if target_col in test_df.columns:
            y_test_lgbm = test_df[target_col].values

            # Encode if encoder available
            if 'label_encoder' in artifacts:
                y_test_lgbm = artifacts['label_encoder'].transform(y_test_lgbm.astype(int))
        else:
            y_test_lgbm = None

        datasets['X_test_lgbm'] = X_test_lgbm
        datasets['y_test_lgbm'] = y_test_lgbm
        print(f"✓ Prepared LightGBM test dataset: {X_test_lgbm.shape}")

    return datasets


if __name__ == "__main__":
    print("Model save/load utilities")
    print("\nUsage:")
    print("  from scripts.save_models import save_model_artifacts, load_model_artifacts")
    print("\n  # Save models after training")
    print("  save_model_artifacts(")
    print("      best_xgb_model=best_xgb_model,")
    print("      best_xgb_feature_columns=best_xgb_feature_columns,")
    print("      best_xgb_label_encoder=best_xgb_label_encoder,")
    print("      signal_best_lgbm=signal_best_lgbm,")
    print("      # ... other artifacts")
    print("  )")
    print("\n  # Load models for analysis")
    print("  artifacts = load_model_artifacts()")
    print("  datasets = create_test_datasets(artifacts)")
