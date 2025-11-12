"""
Visualization Utilities
=======================
Plotting functions for model evaluation and feature analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.inspection import PartialDependenceDisplay
import warnings


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


def plot_cumulative_returns(
    returns_map: dict[str, pd.Series],
    title: str = "Cumulative Strategy Returns",
    xlabel: str | None = None,
    ylabel: str = "Cumulative Return"
) -> None:
    """
    Plot cumulative returns for multiple strategies on a shared axis.

    Args:
        returns_map: Mapping of strategy label to return series.
        title: Plot title.
        xlabel: Optional x-axis label (auto-derived for datetime index).
        ylabel: y-axis label.
    """
    plt.figure(figsize=(12, 6))
    for name, returns in returns_map.items():
        cumulative = returns.cumsum()
        index = cumulative.index
        if isinstance(index, pd.DatetimeIndex):
            if index.tz is not None:
                index = index.tz_convert(None)
            x_values = index.to_pydatetime()
        else:
            x_values = np.arange(len(cumulative))
        plt.plot(x_values, cumulative.values, label=name, linewidth=2)

    plt.title(title)
    plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if isinstance(cumulative.index, pd.DatetimeIndex):
        plt.gcf().autofmt_xdate()
    plt.show()


def plot_roc_curves(models_dict, y_test, label_encoder=None, multiclass_average='macro'):
    """
    Plot ROC curves for multiple models.

    Args:
        models_dict: dict[str, tuple[sklearn estimator, pd.DataFrame or np.ndarray]]
            Mapping from model name to (fitted estimator, feature matrix) pair.
        y_test: pd.Series or np.ndarray
            Ground-truth labels.
        label_encoder: Optional[LabelEncoder]
            If provided, used to handle multiclass encoding.
        multiclass_average: str
            Strategy for multiclass ROC ('macro', 'micro', or None for one-vs-rest)

    Returns:
        Dictionary with AUC scores per model
    """
    y_true = np.asarray(y_test)
    classes = np.unique(y_true)
    n_classes = len(classes)

    plt.figure(figsize=(10, 8))
    auc_scores = {}

    for name, (model, X_test) in models_dict.items():
        y_proba = model.predict_proba(X_test)

        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=classes[1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
            auc_scores[name] = roc_auc
        else:
            # Multiclass - compute macro-average ROC
            y_bin = label_binarize(y_true, classes=classes)

            # Compute ROC curve and ROC area for each class
            fpr_dict = {}
            tpr_dict = {}
            roc_auc_dict = {}

            for i, class_label in enumerate(classes):
                fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

            # Compute macro-average ROC curve
            all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)

            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])

            mean_tpr /= n_classes
            macro_auc = auc(all_fpr, mean_tpr)

            plt.plot(all_fpr, mean_tpr, linewidth=2,
                    label=f'{name} (Macro AUC = {macro_auc:.3f})')
            auc_scores[name] = macro_auc

    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return auc_scores


def plot_drawdown_chart(returns_map, title="Strategy Drawdown Analysis"):
    """
    Plot drawdown charts for multiple strategies.

    Args:
        returns_map: dict[str, pd.Series]
            Mapping of strategy name to return series.
        title: str
            Plot title.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for name, returns in returns_map.items():
        cumulative = returns.cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        drawdown_pct = (drawdown / running_max.replace(0, np.nan)) * 100

        # Handle datetime index
        index = cumulative.index
        if isinstance(index, pd.DatetimeIndex):
            if index.tz is not None:
                index = index.tz_convert(None)
            x_values = index.to_pydatetime()
        else:
            x_values = np.arange(len(cumulative))

        # Cumulative returns
        ax1.plot(x_values, cumulative.values, label=name, linewidth=2)

        # Drawdown
        ax2.fill_between(x_values, 0, drawdown.values, alpha=0.3, label=name)
        ax2.plot(x_values, drawdown.values, linewidth=1.5)

    ax1.set_title(f'{title} - Cumulative Returns', fontsize=14)
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)

    ax2.set_title('Drawdown', fontsize=14)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if isinstance(cumulative.index, pd.DatetimeIndex):
        plt.gcf().autofmt_xdate()
    plt.show()


def plot_predicted_vs_realized(y_test, y_pred, model_name="Model", bins=30):
    """
    Plot predicted vs realized values for classification models.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        bins: Number of bins for histogram
    """
    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot with jitter for categorical data
    unique_true = np.unique(y_test_arr)
    unique_pred = np.unique(y_pred_arr)

    # Add jitter
    jitter_strength = 0.05
    y_test_jitter = y_test_arr + np.random.normal(0, jitter_strength, size=len(y_test_arr))
    y_pred_jitter = y_pred_arr + np.random.normal(0, jitter_strength, size=len(y_pred_arr))

    axes[0].scatter(y_test_jitter, y_pred_jitter, alpha=0.3, s=20)
    axes[0].plot(unique_true, unique_true, 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('True Label', fontsize=12)
    axes[0].set_ylabel('Predicted Label', fontsize=12)
    axes[0].set_title(f'{model_name}: Predicted vs Realized', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Residual plot
    residuals = y_pred_arr - y_test_arr
    axes[1].hist(residuals, bins=bins, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error (Predicted - True)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{model_name}: Prediction Error Distribution', fontsize=14)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_partial_dependence(model, X, feature_names, features_to_plot, model_name="Model"):
    """
    Plot partial dependence plots for selected features.

    Args:
        model: Fitted model
        X: Feature matrix (DataFrame or array)
        feature_names: List of feature names
        features_to_plot: List of feature indices or names to plot
        model_name: Name of the model
    """
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()

    # Convert feature names to indices if needed
    if isinstance(features_to_plot[0], str):
        feature_indices = [list(X_df.columns).index(f) for f in features_to_plot]
    else:
        feature_indices = features_to_plot

    n_features = len(feature_indices)
    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            display = PartialDependenceDisplay.from_estimator(
                model,
                X_df,
                features=feature_indices,
                feature_names=feature_names,
                kind='average',
                ax=axes[:n_features],
                n_jobs=-1
            )

        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'{model_name}: Partial Dependence Plots', fontsize=16, y=1.0)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Warning: Could not generate partial dependence plots: {e}")
        plt.close(fig)


def plot_correlation_heatmap(df, features=None, title="Feature Correlation Matrix",
                             figsize=(12, 10), annot=False):
    """
    Plot correlation heatmap for features.

    Args:
        df: DataFrame with features
        features: List of features to include (None = all numeric columns)
        title: Plot title
        figsize: Figure size
        annot: Whether to annotate cells with correlation values
    """
    if features is None:
        # Select all numeric columns
        df_corr = df.select_dtypes(include=[np.number])
    else:
        df_corr = df[features]

    corr_matrix = df_corr.corr()

    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

    return corr_matrix


def plot_density_plots(df, features, ncols=3, figsize_per_plot=(5, 4), hue=None):
    """
    Plot density distributions for multiple features.

    Args:
        df: DataFrame with features
        features: List of feature names to plot
        ncols: Number of columns in subplot grid
        figsize_per_plot: Size of each subplot (width, height)
        hue: Column name for grouping (e.g., target variable)
    """
    n_features = len(features)
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                            figsize=(figsize_per_plot[0] * ncols,
                                   figsize_per_plot[1] * nrows))
    axes = axes.ravel() if n_features > 1 else [axes]

    for idx, feature in enumerate(features):
        ax = axes[idx]

        if hue is not None and hue in df.columns:
            for group in df[hue].unique():
                subset = df[df[hue] == group][feature].dropna()
                if len(subset) > 0:
                    subset.plot(kind='density', ax=ax, label=f'{hue}={group}', alpha=0.7)
            ax.legend()
        else:
            df[feature].dropna().plot(kind='density', ax=ax, color='steelblue', linewidth=2)

        ax.set_title(f'Density: {feature}', fontsize=12)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.grid(alpha=0.3)

    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_class_distribution(y, title="Target Class Distribution", label_encoder=None):
    """
    Plot distribution of target classes.

    Args:
        y: Target variable (array or Series)
        title: Plot title
        label_encoder: Optional LabelEncoder to decode class labels
    """
    y_arr = np.asarray(y)

    if label_encoder is not None:
        y_decoded = label_encoder.inverse_transform(y_arr)
        unique, counts = np.unique(y_decoded, return_counts=True)
    else:
        unique, counts = np.unique(y_arr, return_counts=True)

    percentages = counts / counts.sum() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    colors = sns.color_palette("Set2", len(unique))
    ax1.bar(range(len(unique)), counts, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(unique)))
    ax1.set_xticklabels(unique)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'{title} - Counts', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        ax1.text(i, count, f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    # Pie chart
    ax2.pie(counts, labels=unique, autopct='%1.1f%%', colors=colors,
           startangle=90, textprops={'fontsize': 11})
    ax2.set_title(f'{title} - Proportions', fontsize=14)

    plt.tight_layout()
    plt.show()

    return dict(zip(unique, zip(counts, percentages)))
