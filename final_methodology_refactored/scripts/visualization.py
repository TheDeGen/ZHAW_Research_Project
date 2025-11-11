"""
Visualization Utilities
=======================
Plotting functions for model evaluation and feature analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrices(models_dict, y_test, class_labels, label_encoder=None):
    """
    Plot confusion matrices for multiple models.

    Args:
        models_dict: dict[str, tuple[sklearn estimator, pd.DataFrame or np.ndarray]]
            Mapping from model name to (fitted estimator, feature matrix) pair.
        y_test: pd.Series or np.ndarray
            Ground-truth labels (already aligned with all feature matrices).
        class_labels: list
            Class labels to keep fixed across plots.
        label_encoder: Optional[LabelEncoder]
            If provided, predictions produced by the estimators will be inverse-transformed
            into the original label space before computing the confusion matrix.
    """
    y_true = np.asarray(y_test)
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
    axes = axes.ravel()

    for ax, (name, (model, X_test)) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        if label_encoder is not None:
            y_pred = label_encoder.inverse_transform(y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
        ax.set_title(name)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """
    Plot LightGBM/XGBoost feature importance (gain) for a fitted model.

    Args:
        model: Fitted model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model for the plot title
        top_n: Number of top features to display

    Returns:
        DataFrame with feature importances sorted by importance
    """
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not expose feature_importances_.")
        return None

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Feature name count ({len(feature_names)}) does not match model importances "
            f"({len(importances)}). Ensure feature_names reflects the training columns."
        )

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top_df = importance_df.head(top_n)

    plt.figure(figsize=(10, max(6, len(top_df) * 0.4)))
    sns.barplot(
        data=top_df,
        x="importance",
        y="feature",
        palette="Blues_r"
    )
    plt.title(f"{model_name} – Top {len(top_df)} Features (Gain)")
    plt.xlabel("Importance (Gain)")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    return importance_df
