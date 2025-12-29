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
from pathlib import Path

# Import visualization configuration
try:
    from config import pipeline_config as cfg
    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False


# ============================================================================
# HELPER FUNCTIONS FOR CONSISTENT STYLING
# ============================================================================

def _get_colors(n: int = None) -> list:
    """
    Return colorblind-safe color palette (ColorBrewer Dark2).

    Args:
        n: Number of colors needed. If None, returns full palette.

    Returns:
        List of hex color strings.
    """
    if _HAS_CONFIG:
        colors = cfg.VIZ_COLOR_LIST
    else:
        colors = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A",
                  "#66A61E", "#E6AB02", "#A6761D", "#666666"]

    if n is None:
        return colors
    return [colors[i % len(colors)] for i in range(n)]


def _truncate_label(label: str, max_chars: int = None) -> str:
    """
    Truncate label to maximum characters with ellipsis.
    
    Args:
        label: Label string to truncate
        max_chars: Maximum characters (default from config)
    
    Returns:
        Truncated label string
    """
    if max_chars is None:
        max_chars = cfg.VIZ_LABEL_MAX_CHARS if _HAS_CONFIG else 45
    
    if len(label) > max_chars:
        return label[:max_chars-3] + '...'
    return label


def _get_fontsize(element: str) -> int:
    """
    Get consistent font size for element type.
    
    Args:
        element: One of 'title', 'label', 'tick', 'legend', 'annotation'
    
    Returns:
        Font size integer
    """
    if _HAS_CONFIG:
        sizes = {
            'title': cfg.VIZ_TITLE_FONTSIZE,
            'label': cfg.VIZ_LABEL_FONTSIZE,
            'tick': cfg.VIZ_TICK_FONTSIZE,
            'legend': cfg.VIZ_LEGEND_FONTSIZE,
            'annotation': cfg.VIZ_ANNOTATION_FONTSIZE,
        }
    else:
        sizes = {
            'title': 14,
            'label': 12,
            'tick': 10,
            'legend': 9,
            'annotation': 9,
        }
    return sizes.get(element, 10)


def _get_figsize_for_labels(n_labels: int, max_label_len: int, base_width: float = 12) -> tuple:
    """
    Calculate appropriate figure size based on label count and length.
    
    Args:
        n_labels: Number of labels (affects height)
        max_label_len: Maximum label character length (affects width)
        base_width: Base figure width
    
    Returns:
        Tuple of (width, height) in inches
    """
    # Adjust width for long labels (add ~0.1 inch per 10 chars over 30)
    extra_width = max(0, (max_label_len - 30) * 0.08)
    width = min(base_width + extra_width, 18)  # Cap at 18 inches
    
    # Adjust height for many labels (0.35 inch per label, min 6)
    height = max(6, min(n_labels * 0.4, 14))  # Cap between 6 and 14 inches
    
    return (width, height)


def _get_semantic_color(key: str) -> str:
    """
    Get semantic color for specific visualization elements.

    Args:
        key: One of 'long', 'neutral', 'short', 'train', 'validation', 'test'

    Returns:
        Hex color string.
    """
    if _HAS_CONFIG:
        return cfg.VIZ_SEMANTIC_COLORS.get(key, cfg.VIZ_COLOR_LIST[0])

    semantic_colors = {
        "long": "#1B9E77", "neutral": "#666666", "short": "#D95F02",
        "train": "#1B9E77", "validation": "#D95F02", "test": "#7570B3",
    }
    return semantic_colors.get(key, "#1B9E77")


def _get_figsize() -> tuple:
    """Get default figure size from config."""
    if _HAS_CONFIG:
        return cfg.DEFAULT_FIGSIZE
    return (12, 8)


def _get_cmap_sequential() -> str:
    """Get sequential colormap from config."""
    if _HAS_CONFIG:
        return cfg.VIZ_CMAP_SEQUENTIAL
    return "YlGnBu"


def _save_figure(fig, filename: str, output_dir: str = None, dpi: int = None) -> str:
    """
    Save figure to disk.

    Args:
        fig: matplotlib figure object
        filename: Name for the file (without extension)
        output_dir: Output directory (defaults to cfg.FIGURES_OUTPUT_DIR)
        dpi: Resolution (defaults to cfg.DEFAULT_DPI)

    Returns:
        Path to saved file.
    """
    if output_dir is None:
        output_dir = cfg.FIGURES_OUTPUT_DIR if _HAS_CONFIG else "outputs/figures"
    if dpi is None:
        dpi = cfg.DEFAULT_DPI if _HAS_CONFIG else 150

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / f"{filename}.png"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Figure saved: {filepath}")
    return str(filepath)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_confusion_matrices(models_dict, y_test, class_labels=None, label_encoder=None,
                            save_path: str = None, show: bool = True):
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
    y_true_encoded = np.asarray(y_test)

    if label_encoder is not None:
        y_true = label_encoder.inverse_transform(y_true_encoded)
        if class_labels is None:
            class_labels = label_encoder.classes_
        else:
            class_labels = np.asarray(class_labels)
            if (
                np.issubdtype(class_labels.dtype, np.integer)
                and class_labels.min() >= 0
                and class_labels.max() < len(label_encoder.classes_)
            ):
                class_labels = label_encoder.classes_[class_labels]
    else:
        y_true = y_true_encoded
        if class_labels is None:
            class_labels = np.unique(y_true)

    # Ensure labels are numpy array for indexing and ConfusionMatrixDisplay compatibility
    class_labels = np.asarray(class_labels)
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
    axes = axes.ravel()

    for ax, (name, (model, X_test)) in zip(axes, models_dict.items()):
        y_pred_encoded = model.predict(X_test)
        if label_encoder is not None:
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
        else:
            y_pred = y_pred_encoded

        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=ax, values_format='d', cmap=_get_cmap_sequential(), colorbar=False)
        ax.set_title(name, fontweight='bold', fontsize=_get_fontsize('title'))
        ax.set_xlabel('Predicted label', fontweight='bold', fontsize=_get_fontsize('label'))
        ax.set_ylabel('True label', fontweight='bold', fontsize=_get_fontsize('label'))
        ax.tick_params(labelsize=_get_fontsize('tick'))

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_feature_importance(model, feature_names, model_name, top_n=20,
                            save_path: str = None, show: bool = True):
    """
    Plot LightGBM/XGBoost feature importance (gain) for a fitted model.

    Args:
        model: Fitted model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model for the plot title
        top_n: Number of top features to display
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot

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
    
    # Calculate dynamic figure size based on label lengths
    max_label_len = max(len(str(f)) for f in top_df['feature'])
    figsize = _get_figsize_for_labels(len(top_df), max_label_len, base_width=12)
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = _get_colors(len(top_df))
    sns.barplot(
        data=top_df,
        x="importance",
        y="feature",
        hue="feature",
        palette=colors,
        legend=False,
        ax=ax
    )
    ax.set_title(f"{model_name} – Top {len(top_df)} Features (Gain)", 
                 fontweight='bold', fontsize=_get_fontsize('title'))
    ax.set_xlabel("Importance (Gain)", fontweight='bold', fontsize=_get_fontsize('label'))
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=_get_fontsize('tick'))
    ax.grid(alpha=0.3, axis='x')
    
    # Use constrained_layout for better handling of long labels
    fig.set_constrained_layout(True)

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return importance_df


def plot_roc_curves(models_dict, y_test, label_encoder=None, multiclass_average='macro',
                    save_path: str = None, show: bool = True):
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
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot

    Returns:
        Dictionary with AUC scores per model
    """
    y_true = np.asarray(y_test)
    classes = np.unique(y_true)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=_get_figsize())
    auc_scores = {}
    colors = _get_colors(len(models_dict))

    for idx, (name, (model, X_test)) in enumerate(models_dict.items()):
        y_proba = model.predict_proba(X_test)

        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=classes[1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, color=colors[idx],
                    label=f'{name} (AUC = {roc_auc:.3f})')
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

            ax.plot(all_fpr, mean_tpr, linewidth=2, color=colors[idx],
                    label=f'{name} (Macro AUC = {macro_auc:.3f})')
            auc_scores[name] = macro_auc

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=_get_fontsize('label'))
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=_get_fontsize('label'))
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold', 
                 fontsize=_get_fontsize('title'))
    ax.tick_params(labelsize=_get_fontsize('tick'))
    ax.legend(loc="lower right", fontsize=_get_fontsize('legend'))
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return auc_scores


def plot_drawdown_chart(
    returns_map,
    title="Strategy Drawdown Analysis",
    normalizer=None,
    return_mode='percentage',
):
    """
    Plot drawdown charts for multiple strategies.

    Args:
        returns_map: dict[str, pd.Series]
            Mapping of strategy name to return series.
        title: str
            Plot title.
        normalizer: float | None
            Mean absolute spread for percentage conversion.
        return_mode: str
            'absolute' or 'percentage'.

    Raises:
        ValueError: If return_mode='percentage' but normalizer is not provided.
    """
    if return_mode == 'percentage' and normalizer is None:
        raise ValueError("normalizer required for percentage mode")

    scale = (100.0 / normalizer) if return_mode == 'percentage' and normalizer else 1.0
    unit = "%" if return_mode == 'percentage' else "EUR/MWh"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for name, returns in returns_map.items():
        cumulative = returns.cumsum() * scale
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        # Percentage drawdown relative to running max
        drawdown_pct = (drawdown / running_max.replace(0, np.nan)) * 100

        # Handle datetime index
        index = returns.index
        if isinstance(index, pd.DatetimeIndex):
            if index.tz is not None:
                index = index.tz_convert(None)
            x_values = index.to_pydatetime()
        else:
            x_values = np.arange(len(cumulative))

        # Cumulative returns (in % or EUR/MWh)
        ax1.plot(x_values, cumulative.values, label=name, linewidth=2)

        # Drawdown (in % of peak)
        ax2.fill_between(x_values, 0, drawdown_pct.values, alpha=0.3, label=name)
        ax2.plot(x_values, drawdown_pct.values, linewidth=1.5)

    ax1.set_title(f'{title} - Cumulative Returns', fontsize=14)
    ax1.set_ylabel(f'Cumulative Return ({unit})', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)

    ax2.set_title('Drawdown (% of Peak)', fontsize=14)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if isinstance(returns.index, pd.DatetimeIndex):
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


def plot_class_distribution(y, title="Target Class Distribution", label_encoder=None,
                            save_path: str = None, show: bool = True):
    """
    Plot distribution of target classes.

    Args:
        y: Target variable (array or Series)
        title: Plot title
        label_encoder: Optional LabelEncoder to decode class labels
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot
    """
    y_arr = np.asarray(y)

    if label_encoder is not None:
        y_decoded = label_encoder.inverse_transform(y_arr)
        unique, counts = np.unique(y_decoded, return_counts=True)
    else:
        unique, counts = np.unique(y_arr, return_counts=True)

    percentages = counts / counts.sum() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_get_figsize())

    # Bar plot
    colors = _get_colors(len(unique))
    ax1.bar(range(len(unique)), counts, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(unique)))
    ax1.set_xticklabels(unique)
    ax1.set_xlabel('Class', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax1.set_title(f'{title} - Counts', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        ax1.text(i, count, f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    # Pie chart
    ax2.pie(counts, labels=unique, autopct='%1.1f%%', colors=colors,
           startangle=90, textprops={'fontsize': 11})
    ax2.set_title(f'{title} - Proportions', fontweight='bold', fontsize=14)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(zip(unique, zip(counts, percentages)))


def plot_cumulative_returns(
    returns_map,
    title="Cumulative Returns Comparison",
    ylabel=None,
    figsize=None,
    dpi=None,
    show=True,
    normalizer=None,
    return_mode='percentage',
    save_path: str = None,
):
    """
    Plot cumulative returns for multiple strategies.

    Args:
        returns_map: dict[str, pd.Series]
            Dictionary mapping strategy name to return series.
        title: str
            Plot title.
        ylabel: str
            Y-axis label (auto-generated if None based on return_mode).
        figsize: tuple
            Figure size in inches. Defaults to config value.
        dpi: int
            Figure resolution. Defaults to config value.
        show: bool
            Whether to call plt.show() at the end.
        normalizer: float | None
            Mean absolute spread for percentage conversion.
        return_mode: str
            'absolute' or 'percentage'.
        save_path: Optional filename to save (without extension)

    Returns:
        Tuple of (fig, ax)

    Raises:
        ValueError: If return_mode='percentage' but normalizer is not provided.
    """
    if return_mode == 'percentage' and normalizer is None:
        raise ValueError("normalizer required for percentage mode")

    if figsize is None:
        figsize = _get_figsize()
    if dpi is None:
        dpi = cfg.DEFAULT_DPI if _HAS_CONFIG else 200

    scale = (100.0 / normalizer) if return_mode == 'percentage' and normalizer else 1.0

    if ylabel is None:
        ylabel = "Cumulative Return (%)" if return_mode == 'percentage' else "Cumulative Return (EUR/MWh)"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    colors = _get_colors(len(returns_map))
    linestyles = ['-', '--', '-.', ':']

    for idx, (name, returns) in enumerate(returns_map.items()):
        cumulative = returns.cumsum() * scale
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        ax.plot(cumulative, label=name, color=color, linestyle=linestyle, linewidth=2, alpha=0.8)

    ax.set_xlabel("Time Period", fontweight='bold', fontsize=_get_fontsize('label'))
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=_get_fontsize('label'))
    ax.set_title(title, fontweight="bold", fontsize=_get_fontsize('title'))
    ax.tick_params(labelsize=_get_fontsize('tick'))
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.legend(loc='best', framealpha=0.9, fontsize=_get_fontsize('legend'))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_nlp_feature_importance(
    model,
    feature_names,
    nlp_prefix="news_",
    top_n=20,
    figsize=(14, 8),
):
    """
    Compare NLP feature importance against other feature categories.

    Args:
        model: Trained model with `feature_importances_` attribute
        feature_names: List of feature names aligned with model training
        nlp_prefix: Prefix identifying NLP-derived features
        top_n: Number of top features to display in detailed view
        figsize: Figure size in inches

    Returns:
        Tuple of (fig, (ax1, ax2))
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not have 'feature_importances_' attribute")

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(f"Feature count mismatch: {len(feature_names)} names vs {len(importances)} importances")

    # Create importance dataframe
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # Categorize features
    def categorize_feature(name):
        name_lower = str(name).lower()
        if name_lower.startswith(nlp_prefix.lower()):
            return "NLP"
        elif any(kw in name_lower for kw in ["price", "spread", "auction", "spot"]):
            return "Price"
        elif any(kw in name_lower for kw in ["solar", "wind", "hydro", "biomass", "load", "generation"]):
            return "Energy"
        else:
            return "Other"

    importance_df["category"] = importance_df["feature"].apply(categorize_feature)

    # Panel 1: Top N NLP features
    nlp_df = importance_df[importance_df["category"] == "NLP"].head(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("NLP Feature Importance Analysis", fontsize=16, fontweight="bold")

    if len(nlp_df) > 0:
        sns.barplot(data=nlp_df, x="importance", y="feature", ax=ax1, palette="viridis")
        ax1.set_title(f"Top {len(nlp_df)} NLP Features", fontweight="bold")
        ax1.set_xlabel("Importance (Gain)")
        ax1.set_ylabel("")
    else:
        ax1.text(0.5, 0.5, "No NLP features found", ha="center", va="center", fontsize=12)
        ax1.set_title("Top NLP Features", fontweight="bold")

    # Panel 2: Category comparison
    category_importance = importance_df.groupby("category")["importance"].sum().sort_values(ascending=False)

    colors = {"NLP": "#1f77b4", "Price": "#ff7f0e", "Energy": "#2ca02c", "Other": "#d62728"}
    color_list = [colors.get(cat, "#808080") for cat in category_importance.index]

    ax2.bar(category_importance.index, category_importance.values, color=color_list, edgecolor="black", alpha=0.7)
    ax2.set_title("Feature Importance by Category", fontweight="bold")
    ax2.set_xlabel("Feature Category")
    ax2.set_ylabel("Total Importance")
    ax2.grid(True, alpha=0.3, axis="y")

    # Annotate bars with percentages
    total_importance = category_importance.sum()
    for i, (cat, val) in enumerate(category_importance.items()):
        pct = (val / total_importance) * 100
        ax2.text(i, val, f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.show()

    return fig, (ax1, ax2)


def plot_eda_dashboard(master_df: pd.DataFrame, news_df: pd.DataFrame,
                       save_path: str = None, show: bool = True):
    """
    Generate comprehensive EDA dashboard with multiple panels.

    Args:
        master_df: Master dataframe with energy and target data
        news_df: News dataframe with classification data
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, top=0.93, bottom=0.05)
    
    title_fs = _get_fontsize('title') - 2  # Slightly smaller for subplots
    label_fs = _get_fontsize('label') - 1
    tick_fs = _get_fontsize('tick')
    legend_fs = _get_fontsize('legend')
    annot_fs = _get_fontsize('annotation')

    # 1. Target distribution
    ax1 = fig.add_subplot(gs[0, 0])
    target_col = 'spread_target_shift_24'
    if target_col in master_df.columns:
        target_counts = master_df[target_col].value_counts().sort_index()
        colors = [_get_semantic_color('short'), _get_semantic_color('neutral'), _get_semantic_color('long')]
        ax1.bar(target_counts.index, target_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Target Class', fontweight='bold', fontsize=label_fs)
        ax1.set_ylabel('Count', fontweight='bold', fontsize=label_fs)
        ax1.set_title('Target Distribution', fontweight='bold', fontsize=title_fs)
        ax1.tick_params(labelsize=tick_fs)
        ax1.grid(alpha=0.3, axis='y')
        for i, (idx, val) in enumerate(target_counts.items()):
            pct = val / target_counts.sum() * 100
            ax1.text(idx, val, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=annot_fs)
    else:
        ax1.text(0.5, 0.5, f'Column "{target_col}"\nnot found', ha='center', va='center', fontsize=tick_fs)
        ax1.set_title('Target Distribution', fontweight='bold', fontsize=title_fs)

    # 2. Price spreads over time
    ax2 = fig.add_subplot(gs[0, 1:])
    if 'real_spread_abs' in master_df.columns:
        spread_series = master_df['real_spread_abs']
        ax2.plot(spread_series.index, spread_series.values, linewidth=0.5, alpha=0.7)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Time', fontweight='bold', fontsize=label_fs)
        ax2.set_ylabel('Spread (EUR/MWh)', fontweight='bold', fontsize=label_fs)
        ax2.set_title('Price Spread Over Time (Spot - Day Ahead)', fontweight='bold', fontsize=title_fs)
        ax2.tick_params(labelsize=tick_fs)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Column "real_spread_abs"\nnot found', ha='center', va='center', fontsize=tick_fs)
        ax2.set_title('Price Spread Over Time', fontweight='bold', fontsize=title_fs)

    # 3. News classification distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if 'classification' in news_df.columns and len(news_df) > 0:
        class_counts = news_df['classification'].value_counts().head(8)
        ax3.barh(range(len(class_counts)), class_counts.values, color=_get_colors(1)[0], alpha=0.7)
        ax3.set_yticks(range(len(class_counts)))
        ax3.set_yticklabels([_truncate_label(label, max_chars=35) 
                             for label in class_counts.index], fontsize=tick_fs - 1)
        ax3.set_xlabel('Count', fontweight='bold', fontsize=label_fs)
        ax3.set_title('Top 8 News Classifications', fontweight='bold', fontsize=title_fs)
        ax3.tick_params(labelsize=tick_fs)
        ax3.grid(alpha=0.3, axis='x')
    else:
        ax3.text(0.5, 0.5, 'No news classification\ndata available', ha='center', va='center', fontsize=tick_fs)
        ax3.set_title('Top 8 News Classifications', fontweight='bold', fontsize=title_fs)

    # 4. Spread volatility analysis (rolling standard deviation)
    ax4 = fig.add_subplot(gs[1, 1])
    if 'real_spread_abs' in master_df.columns and len(master_df) > 168:  # Need enough data for rolling
        spread_series = master_df['real_spread_abs'].dropna()
        # Calculate rolling std with different windows
        rolling_24h = spread_series.rolling(window=24, min_periods=1).std()
        rolling_168h = spread_series.rolling(window=168, min_periods=1).std()  # 1 week

        ax4.plot(rolling_24h.index, rolling_24h.values, label='24h window', linewidth=1.5, alpha=0.8)
        ax4.plot(rolling_168h.index, rolling_168h.values, label='1-week window', linewidth=1.5, alpha=0.8)
        ax4.set_xlabel('Time', fontweight='bold', fontsize=label_fs)
        ax4.set_ylabel('Volatility (EUR/MWh)', fontweight='bold', fontsize=label_fs)
        ax4.set_title('Spread Volatility Over Time', fontweight='bold', fontsize=title_fs)
        ax4.tick_params(labelsize=tick_fs)
        ax4.legend(loc='best', fontsize=legend_fs)
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for\nvolatility analysis', ha='center', va='center', fontsize=tick_fs)
        ax4.set_title('Spread Volatility Over Time', fontweight='bold', fontsize=title_fs)

    # 5. Load distribution
    ax5 = fig.add_subplot(gs[1, 2])
    if 'Load' in master_df.columns:
        load_series = master_df['Load'].dropna()
        if len(load_series) > 0:
            ax5.hist(load_series, bins=50, color=_get_colors(2)[1], alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Load (MW)', fontweight='bold', fontsize=label_fs)
            ax5.set_ylabel('Frequency', fontweight='bold', fontsize=label_fs)
            ax5.set_title('Load Distribution', fontweight='bold', fontsize=title_fs)
            ax5.tick_params(labelsize=tick_fs)
            ax5.grid(alpha=0.3, axis='y')
            ax5.axvline(load_series.mean(), color=_get_semantic_color('short'), linestyle='--', linewidth=2, label='Mean')
            ax5.legend(fontsize=legend_fs)
        else:
            ax5.text(0.5, 0.5, 'No Load data\navailable', ha='center', va='center', fontsize=tick_fs)
            ax5.set_title('Load Distribution', fontweight='bold', fontsize=title_fs)
    else:
        ax5.text(0.5, 0.5, 'Column "Load"\nnot found', ha='center', va='center', fontsize=tick_fs)
        ax5.set_title('Load Distribution', fontweight='bold', fontsize=title_fs)

    # 6. Hourly price patterns
    ax6 = fig.add_subplot(gs[2, 0])
    if 'hour' in master_df.columns and 'Spot Price' in master_df.columns:
        hourly_price = master_df.groupby('hour')['Spot Price'].mean()
        ax6.plot(hourly_price.index, hourly_price.values, marker='o', linewidth=2, markersize=4)
        ax6.set_xlabel('Hour of Day', fontweight='bold', fontsize=label_fs)
        ax6.set_ylabel('Average Spot Price (EUR/MWh)', fontweight='bold', fontsize=label_fs)
        ax6.set_title('Average Spot Price by Hour', fontweight='bold', fontsize=title_fs)
        ax6.tick_params(labelsize=tick_fs)
        ax6.grid(alpha=0.3)
        ax6.set_xticks(range(0, 24, 3))
    else:
        missing_cols = []
        if 'hour' not in master_df.columns:
            missing_cols.append('hour')
        if 'Spot Price' not in master_df.columns:
            missing_cols.append('Spot Price')
        ax6.text(0.5, 0.5, f'Missing columns:\n{", ".join(missing_cols)}', ha='center', va='center', fontsize=tick_fs)
        ax6.set_title('Average Spot Price by Hour', fontweight='bold', fontsize=title_fs)

    # 7. Day of week patterns
    ax7 = fig.add_subplot(gs[2, 1])
    if 'day_of_week' in master_df.columns and 'real_spread_abs' in master_df.columns:
        dow_spread = master_df.groupby('day_of_week')['real_spread_abs'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax7.bar(range(7), dow_spread.values, color=_get_colors(1)[0], alpha=0.7, edgecolor='black')
        ax7.set_xticks(range(7))
        ax7.set_xticklabels(days, fontsize=tick_fs)
        ax7.set_ylabel('Average Spread (EUR/MWh)', fontweight='bold', fontsize=label_fs)
        ax7.set_title('Average Spread by Day of Week', fontweight='bold', fontsize=title_fs)
        ax7.tick_params(labelsize=tick_fs)
        ax7.grid(alpha=0.3, axis='y')
    else:
        missing_cols = []
        if 'day_of_week' not in master_df.columns:
            missing_cols.append('day_of_week')
        if 'real_spread_abs' not in master_df.columns:
            missing_cols.append('real_spread_abs')
        ax7.text(0.5, 0.5, f'Missing columns:\n{", ".join(missing_cols)}', ha='center', va='center', fontsize=tick_fs)
        ax7.set_title('Average Spread by Day of Week', fontweight='bold', fontsize=title_fs)

    # 8. News volume over time
    ax8 = fig.add_subplot(gs[2, 2])
    if hasattr(news_df.index, 'to_period') and len(news_df) > 0:
        news_daily = news_df.groupby(news_df.index.to_period('D')).size()
        ax8.plot(news_daily.index.to_timestamp(), news_daily.values, linewidth=1.5, color=_get_colors(3)[2])
        ax8.set_xlabel('Date', fontweight='bold', fontsize=label_fs)
        ax8.set_ylabel('Number of Articles', fontweight='bold', fontsize=label_fs)
        ax8.set_title('News Volume Over Time', fontweight='bold', fontsize=title_fs)
        ax8.tick_params(labelsize=tick_fs)
        ax8.grid(alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'No time-indexed\nnews data available', ha='center', va='center', fontsize=tick_fs)
        ax8.set_title('News Volume Over Time', fontweight='bold', fontsize=title_fs)

    fig.suptitle('Exploratory Data Analysis Dashboard', fontsize=_get_fontsize('title') + 2, 
                 fontweight='bold')

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_embedding_quality(news_df: pd.DataFrame, n_samples: int = 500,
                           save_path: str = None, show: bool = True):
    """
    Visualize embedding quality using UMAP dimensionality reduction.

    Args:
        news_df: News dataframe with 'embedding' and 'classification' columns
        n_samples: Number of samples to visualize (for performance)
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot
    """
    if 'embedding' not in news_df.columns:
        print("No embeddings found in news_df. Skipping visualization.")
        return

    # Sample if needed
    if len(news_df) > n_samples:
        news_sample = news_df.sample(n_samples, random_state=42)
    else:
        news_sample = news_df.copy()

    # Extract embeddings
    embeddings = np.vstack(news_sample['embedding'].values)
    labels = news_sample['classification'].values if 'classification' in news_sample.columns else None

    fig, ax = plt.subplots(figsize=_get_figsize())

    # UMAP visualization
    try:
        import umap
        reducer_umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_jobs=-1)
        embedding_2d = reducer_umap.fit_transform(embeddings)

        if labels is not None:
            unique_labels = pd.Series(labels).value_counts().head(10).index
            colors = _get_colors(len(unique_labels))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                          c=colors[i], label=label[:30], alpha=0.5, s=80)
            ax.legend(loc='best', fontsize=8, ncol=2)
        else:
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                      c=_get_colors(1)[0], alpha=0.5, s=80)

        ax.set_xlabel('UMAP Dimension 1', fontweight='bold', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontweight='bold', fontsize=12)
        ax.set_title('News Embedding Visualization (UMAP)', fontweight='bold', fontsize=14)
        ax.grid(alpha=0.3)
    except ImportError:
        ax.text(0.5, 0.5, 'UMAP not installed\npip install umap-learn',
                ha='center', va='center', fontsize=12)
        ax.set_title('UMAP (Not Available)', fontweight='bold')

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_learning_curves(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splitter,
    scoring: str = 'f1_macro',
    model_name: str = 'Model',
    train_sizes: np.ndarray = None,
    save_path: str = None,
    show: bool = True
):
    """
    Plot learning curves to diagnose overfitting/underfitting.

    Args:
        model: Unfitted estimator
        X_train: Training features
        y_train: Training labels
        cv_splitter: Cross-validation splitter
        scoring: Scoring metric
        model_name: Name for the plot title
        train_sizes: Array of training sizes to evaluate
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot
    """
    from sklearn.model_selection import learning_curve

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    print(f"\n{'='*70}")
    print(f"COMPUTING LEARNING CURVES FOR {model_name.upper()}")
    print(f"{'='*70}")
    print("This may take a few minutes...")

    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            random_state=42
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        fig, ax = plt.subplots(figsize=_get_figsize())

        train_color = _get_semantic_color('train')
        val_color = _get_semantic_color('validation')

        # Plot training and validation scores
        ax.plot(train_sizes_abs, train_mean, 'o-', color=train_color, linewidth=2,
               markersize=6, label='Training score')
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color=train_color)

        ax.plot(train_sizes_abs, val_mean, 'o-', color=val_color, linewidth=2,
               markersize=6, label='Validation score')
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color=val_color)

        # Add gap visualization
        gap = train_mean - val_mean
        ax.plot(train_sizes_abs, gap, '--', color='gray', linewidth=1.5,
               alpha=0.7, label='Train-Val Gap')

        ax.set_xlabel('Training Set Size', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'{scoring}', fontweight='bold', fontsize=12)
        ax.set_title(f'{model_name} Learning Curves', fontweight='bold', fontsize=14)
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3)

        # Annotate final scores
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        ax.text(0.02, 0.98,
               f'Final Train: {final_train:.3f}\nFinal Val: {final_val:.3f}\nGap: {gap[-1]:.3f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            _save_figure(fig, save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)

        print(f"✓ Learning curves complete")
        print(f"  Final training score: {final_train:.4f}")
        print(f"  Final validation score: {final_val:.4f}")
        print(f"  Train-val gap: {gap[-1]:.4f}")

        if gap[-1] > 0.1:
            print("  ⚠ Warning: Large gap suggests overfitting")
        elif val_mean[-1] < 0.5:
            print("  ⚠ Warning: Low validation score suggests underfitting")

        print(f"{'='*70}\n")

    except Exception as e:
        print(f"✗ Learning curve generation failed: {e}")
        print(f"{'='*70}\n")


def plot_feature_importance_shap(
    model,
    X_test: pd.DataFrame,
    feature_names: list,
    model_name: str = 'Model',
    max_display: int = 20
):
    """
    Plot SHAP feature importance values.

    Args:
        model: Fitted model
        X_test: Test features
        feature_names: List of feature names
        model_name: Name for the plot title
        max_display: Maximum number of features to display
    """
    try:
        import shap

        print(f"\n{'='*70}")
        print(f"COMPUTING SHAP VALUES FOR {model_name.upper()}")
        print(f"{'='*70}")

        # Create explainer based on model type
        if hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for non-tree models (slower)
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 100))

        shap_values = explainer.shap_values(X_test[:500])  # Limit to 500 samples for speed

        # For multiclass, select the first class
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[0]
        else:
            shap_values_plot = shap_values

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_plot, X_test[:500],
                         feature_names=feature_names,
                         max_display=max_display, show=False)
        plt.title(f'{model_name} - SHAP Feature Importance', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()

        print(f"✓ SHAP analysis complete")
        print(f"{'='*70}\n")

    except ImportError:
        print("SHAP not installed. Install with: pip install shap")
    except Exception as e:
        print(f"SHAP visualization failed: {e}")


def plot_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list,
    model_name: str = 'Model',
    n_repeats: int = 10,
    top_n: int = 20
):
    """
    Plot permutation feature importance.

    Args:
        model: Fitted model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        model_name: Name for the plot title
        n_repeats: Number of times to permute each feature
        top_n: Number of top features to display
    """
    from sklearn.inspection import permutation_importance

    print(f"\n{'='*70}")
    print(f"COMPUTING PERMUTATION IMPORTANCE FOR {model_name.upper()}")
    print(f"{'='*70}")

    try:
        # Compute permutation importance
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

        # Sort by importance
        sorted_idx = result.importances_mean.argsort()[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_idx) * 0.4)))

        # Plot
        box_data = [result.importances[idx] for idx in sorted_idx]
        positions = np.arange(len(sorted_idx))

        bp = ax.boxplot(box_data, positions=positions, vert=False, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))

        ax.set_yticks(positions)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Decrease in Score', fontweight='bold', fontsize=12)
        ax.set_title(f'{model_name} - Permutation Feature Importance (Top {top_n})',
                    fontweight='bold', fontsize=14)
        ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        plt.show()

        print(f"✓ Permutation importance complete")
        print(f"  Top 5 features:")
        for i, idx in enumerate(sorted_idx[:5], 1):
            print(f"    {i}. {feature_names[idx]}: {result.importances_mean[idx]:.4f} "
                 f"± {result.importances_std[idx]:.4f}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"Permutation importance failed: {e}")
        print(f"{'='*70}\n")


def plot_transaction_cost_sensitivity(
    strategy_actions: dict[str, np.ndarray],
    spread_series: pd.Series,
    cost_range: list[float] = None,
    pct_cost_scenarios: dict[str, tuple[float, float]] = None,
    normalizer: float = None,
    return_mode: str = 'percentage',
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Plot transaction cost sensitivity analysis for trading strategies.

    Analyzes how transaction costs impact strategy performance through:
    1. Summary tables for different cost scenarios
    2. Line plots showing total return vs transaction cost
    3. Line plots showing Sharpe ratio vs transaction cost

    Args:
        strategy_actions: Dict mapping strategy name to action arrays
        spread_series: Series of price spreads
        cost_range: List of fixed costs to test (EUR/MWh). Default: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        pct_cost_scenarios: Dict of scenario names to (fixed_cost, pct_cost) tuples.
            Default scenarios: No Costs, Low Cost, Medium Cost, High Cost
        normalizer: Mean absolute spread for percentage conversion.
        return_mode: 'absolute' or 'percentage'.

    Raises:
        ValueError: If return_mode='percentage' but normalizer is not provided.
    """
    from . import evaluation

    if return_mode == 'percentage' and normalizer is None:
        raise ValueError("normalizer required for percentage mode")

    scale = (100.0 / normalizer) if return_mode == 'percentage' and normalizer else 1.0
    unit = "%" if return_mode == 'percentage' else "EUR/MWh"

    # Default cost ranges
    if cost_range is None:
        cost_range = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    if pct_cost_scenarios is None:
        pct_cost_scenarios = {
            "No Costs": (0.0, 0.0),
            "Low Cost (€0.5/MWh)": (0.5, 0.0),
            "Medium Cost (€1.0/MWh)": (1.0, 0.0),
            "High Cost (€2.0/MWh + 0.1%)": (2.0, 0.001),
        }

    print("\n" + "=" * 80)
    print("BACKTESTING WITH TRANSACTION COSTS")
    print("=" * 80)

    # Evaluate scenarios
    for scenario_name, (fixed_cost, pct_cost) in pct_cost_scenarios.items():
        print(f"\n{scenario_name}:")
        print(f"  Fixed: €{fixed_cost}/MWh, Percentage: {pct_cost*100:.2f}%")

        strategy_returns_cost = evaluation.compute_strategy_returns(
            action_map=strategy_actions,
            spread=spread_series,
            transaction_cost=fixed_cost,
            transaction_cost_pct=pct_cost
        )

        returns_summary_cost = evaluation.summarise_strategy_set(
            strategy_returns_cost,
            normalizer=normalizer,
            return_mode=return_mode
        )
        # Get the appropriate column name based on mode
        total_return_col = f"Total Return ({unit})"
        cols_to_show = [total_return_col, 'Sharpe (annualised)', 'Sortino (annualised)']
        print(f"\n{returns_summary_cost[cols_to_show].to_string()}")

    # Visualize impact
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Collect strategy names
    strategy_names = list(strategy_actions.keys())

    # Initialize storage for each strategy
    strategy_returns_by_cost = {name: [] for name in strategy_names}
    strategy_sharpes_by_cost = {name: [] for name in strategy_names}

    # Compute metrics for each cost level
    for cost in cost_range:
        temp_returns = evaluation.compute_strategy_returns(
            action_map=strategy_actions,
            spread=spread_series,
            transaction_cost=cost,
            transaction_cost_pct=0.0
        )
        temp_summary = evaluation.summarise_strategy_set(
            temp_returns,
            normalizer=normalizer,
            return_mode=return_mode
        )

        for name in strategy_names:
            # Scale the raw sum for plotting
            strategy_returns_by_cost[name].append(temp_returns[name].sum() * scale)
            strategy_sharpes_by_cost[name].append(temp_summary.loc[name, "Sharpe (annualised)"])

    # Plot 1: Total return vs cost
    colors = _get_colors(len(strategy_names))
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

    for idx, name in enumerate(strategy_names):
        ax1.plot(cost_range, strategy_returns_by_cost[name],
                marker=markers[idx % len(markers)], linewidth=2, markersize=8,
                color=colors[idx % len(colors)], label=name.replace("LightGBM ", ""))

    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Fixed Transaction Cost (EUR/MWh)', fontweight='bold', fontsize=12)
    ax1.set_ylabel(f'Total Return ({unit})', fontweight='bold', fontsize=12)
    ax1.set_title('Total Return vs Transaction Cost', fontweight='bold', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)

    # Plot 2: Sharpe ratio vs cost
    for idx, name in enumerate(strategy_names):
        ax2.plot(cost_range, strategy_sharpes_by_cost[name],
                marker=markers[idx % len(markers)], linewidth=2, markersize=8,
                color=colors[idx % len(colors)], label=name.replace("LightGBM ", ""))

    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Fixed Transaction Cost (EUR/MWh)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Sharpe Ratio (Annualized)', fontweight='bold', fontsize=12)
    ax2.set_title('Sharpe Ratio vs Transaction Cost', fontweight='bold', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    print("\n" + "=" * 80)


# ============================================================================
# INDIVIDUAL PUBLICATION-READY PLOTS
# ============================================================================

def plot_top_news_classifications(news_df: pd.DataFrame, top_n: int = 8,
                                   save_path: str = None, show: bool = True):
    """
    Plot top N news topic classifications as horizontal bar chart.

    Args:
        news_df: News dataframe with 'classification' column
        top_n: Number of top classifications to show
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot

    Returns:
        DataFrame with classification counts
    """
    if 'classification' not in news_df.columns or len(news_df) == 0:
        print("No classification data available.")
        return None

    class_counts = news_df['classification'].value_counts().head(top_n)
    
    # Calculate dynamic figure size based on label lengths
    max_label_len = max(len(str(label)) for label in class_counts.index)
    figsize = _get_figsize_for_labels(len(class_counts), max_label_len, base_width=12)
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = _get_colors(len(class_counts))

    y_pos = range(len(class_counts))
    ax.barh(y_pos, class_counts.values, color=colors, alpha=0.8, edgecolor='black')

    labels = [_truncate_label(label) for label in class_counts.index]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=_get_fontsize('tick'))
    ax.invert_yaxis()

    ax.set_xlabel('Number of Articles', fontweight='bold', fontsize=_get_fontsize('label'))
    ax.set_title(f'Top {top_n} News Topic Classifications', fontweight='bold', 
                 fontsize=_get_fontsize('title'))
    ax.grid(alpha=0.3, axis='x')

    for i, (count, pct) in enumerate(zip(class_counts.values,
                                          class_counts.values / class_counts.sum() * 100)):
        ax.text(count + 0.5, i, f'{count} ({pct:.1f}%)', va='center', 
                fontsize=_get_fontsize('annotation'))

    fig.set_constrained_layout(True)

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return class_counts.to_frame('count')


def plot_target_distribution(master_df: pd.DataFrame, target_column: str = 'spread_target_shift_24',
                              save_path: str = None, show: bool = True):
    """
    Plot target class distribution (Long/Neutral/Short).

    Args:
        master_df: Master dataframe with target column
        target_column: Name of target column
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot

    Returns:
        DataFrame with class distribution
    """
    if target_column not in master_df.columns:
        print(f"Target column '{target_column}' not found.")
        return None

    fig, ax = plt.subplots(figsize=_get_figsize())

    target_counts = master_df[target_column].value_counts().sort_index()
    colors = [_get_semantic_color('short'), _get_semantic_color('neutral'), _get_semantic_color('long')]
    labels = ['Short (-1)', 'Neutral (0)', 'Long (+1)']

    bars = ax.bar(labels, target_counts.values, color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Target Class', fontweight='bold', fontsize=_get_fontsize('label'))
    ax.set_ylabel('Count', fontweight='bold', fontsize=_get_fontsize('label'))
    ax.set_title('Target Distribution (24h Spread Direction)', fontweight='bold', 
                 fontsize=_get_fontsize('title'))
    ax.tick_params(labelsize=_get_fontsize('tick'))
    ax.grid(alpha=0.3, axis='y')

    total = target_counts.sum()
    for bar, count in zip(bars, target_counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', 
                fontsize=_get_fontsize('annotation'))

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return target_counts.to_frame('count')


def plot_top_news_sources(news_df: pd.DataFrame, top_n: int = 8,
                           save_path: str = None, show: bool = True):
    """
    Plot top N news sources (publishers) as horizontal bar chart.

    Args:
        news_df: News dataframe with 'source' column
        top_n: Number of top sources to show
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot

    Returns:
        DataFrame with source counts
    """
    source_col = None
    if 'source' in news_df.columns:
        source_col = 'source'
    elif 'publisher' in news_df.columns:
        source_col = 'publisher'

    if source_col is None or len(news_df) == 0:
        print("No source/publisher column found.")
        return None

    source_counts = news_df[source_col].value_counts().head(top_n)
    
    # Calculate dynamic figure size
    n_sources = len(source_counts)
    figsize = (12, max(6, n_sources * 0.6))
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = _get_colors(len(source_counts))

    y_pos = range(len(source_counts))
    ax.barh(y_pos, source_counts.values, color=colors, alpha=0.8, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(source_counts.index, fontsize=_get_fontsize('tick'))
    ax.invert_yaxis()

    ax.set_xlabel('Number of Articles', fontweight='bold', fontsize=_get_fontsize('label'))
    ax.set_title(f'Top {top_n} News Sources', fontweight='bold', fontsize=_get_fontsize('title'))
    ax.grid(alpha=0.3, axis='x')

    for i, (count, pct) in enumerate(zip(source_counts.values,
                                          source_counts.values / source_counts.sum() * 100)):
        ax.text(count + 0.5, i, f'{count} ({pct:.1f}%)', va='center', 
                fontsize=_get_fontsize('annotation'))

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return source_counts.to_frame('count')


def plot_news_hourly_coverage(news_df: pd.DataFrame, save_path: str = None, show: bool = True):
    """
    Plot news article count by hour of day.

    Args:
        news_df: News dataframe with datetime index
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot

    Returns:
        Series with hourly counts
    """
    if not isinstance(news_df.index, pd.DatetimeIndex):
        print("News dataframe must have DatetimeIndex.")
        return None

    fig, ax = plt.subplots(figsize=_get_figsize())

    hourly_counts = news_df.groupby(news_df.index.hour).size()

    ax.bar(hourly_counts.index, hourly_counts.values, color=_get_colors(1)[0],
           alpha=0.8, edgecolor='black')

    ax.set_xlabel('Hour of Day', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Articles', fontweight='bold', fontsize=12)
    ax.set_title('News Coverage by Hour', fontweight='bold', fontsize=14)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3, axis='y')

    peak_hour = hourly_counts.idxmax()
    ax.axvline(peak_hour, color=_get_semantic_color('short'), linestyle='--', linewidth=2,
               label=f'Peak: {peak_hour}:00')
    ax.legend(loc='best')

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return hourly_counts


def plot_equity_curve(
    equity_curves: dict[str, pd.Series],
    initial_capital: float = 100_000.0,
    title: str = "Portfolio Equity Curve",
    save_path: str = None,
    show: bool = True,
    show_pct_return: bool = True,
) -> None:
    """
    Plot equity curves for multiple strategies.
    
    Args:
        equity_curves: Dictionary mapping strategy name to equity curve Series
        initial_capital: Initial portfolio value for reference line
        title: Plot title
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot
        show_pct_return: If True, display percentage returns instead of absolute EUR values
    
    Returns:
        Tuple of (fig, ax) if show=False, else None
    """
    fig, ax = plt.subplots(figsize=_get_figsize())
    
    colors = _get_colors(len(equity_curves))
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (name, equity) in enumerate(equity_curves.items()):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        
        # Handle datetime index
        if isinstance(equity.index, pd.DatetimeIndex):
            if equity.index.tz is not None:
                index = equity.index.tz_convert(None)
            else:
                index = equity.index
            x_values = index.to_pydatetime()
        else:
            x_values = equity.index
        
        # Convert to percentage returns if requested
        if show_pct_return:
            equity_values = ((equity.values / initial_capital) - 1) * 100
        else:
            equity_values = equity.values
        
        ax.plot(x_values, equity_values, label=name, color=color, 
                linestyle=linestyle, linewidth=2, alpha=0.8)
    
    # Add reference line (0% for percentage, initial capital for absolute)
    if show_pct_return:
        ax.axhline(0, color='gray', linestyle=':', linewidth=1.5, 
                   alpha=0.5, label='Break-even (0%)')
    else:
        ax.axhline(initial_capital, color='gray', linestyle=':', linewidth=1.5, 
                   alpha=0.5, label=f'Initial Capital (€{initial_capital:,.0f})')
    
    ax.set_xlabel('Time', fontweight='bold', fontsize=_get_fontsize('label'))
    if show_pct_return:
        ax.set_ylabel('Portfolio Return (%)', fontweight='bold', fontsize=_get_fontsize('label'))
    else:
        ax.set_ylabel('Portfolio Value (EUR)', fontweight='bold', fontsize=_get_fontsize('label'))
    ax.set_title(title, fontweight='bold', fontsize=_get_fontsize('title'))
    ax.tick_params(labelsize=_get_fontsize('tick'))
    ax.legend(loc='best', framealpha=0.9, fontsize=_get_fontsize('legend'))
    ax.grid(True, alpha=0.3)
    
    # Format y-axis
    if show_pct_return:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    else:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig, ax


def plot_portfolio_drawdown(
    equity_curves: dict[str, pd.Series],
    initial_capital: float = 100_000.0,
    title: str = "Portfolio Drawdown Analysis",
    save_path: str = None,
    show: bool = True,
    show_pct_return: bool = True,
) -> None:
    """
    Plot drawdown analysis for portfolio equity curves.
    
    Args:
        equity_curves: Dictionary mapping strategy name to equity curve Series
        initial_capital: Initial portfolio value (required for percentage conversion)
        title: Plot title
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot
        show_pct_return: If True, display percentage returns instead of absolute EUR values
    
    Returns:
        Tuple of (fig, axes) if show=False, else None
    """
    figsize = (12, 10)  # Taller for two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    plt.subplots_adjust(hspace=0.15)
    
    colors = _get_colors(len(equity_curves))
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (name, equity) in enumerate(equity_curves.items()):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        
        # Handle datetime index
        if isinstance(equity.index, pd.DatetimeIndex):
            if equity.index.tz is not None:
                index = equity.index.tz_convert(None)
            else:
                index = equity.index
            x_values = index.to_pydatetime()
        else:
            x_values = equity.index
        
        # Calculate drawdown (always use absolute values for drawdown calculation)
        running_max = equity.cummax()
        drawdown = equity - running_max
        drawdown_pct = (drawdown / running_max.replace(0, np.nan)) * 100
        
        # Convert equity curve to percentage returns if requested
        if show_pct_return:
            equity_values = ((equity.values / initial_capital) - 1) * 100
        else:
            equity_values = equity.values
        
        # Plot equity curve
        ax1.plot(x_values, equity_values, label=name, color=color,
                linestyle=linestyle, linewidth=2, alpha=0.8)
        
        # Plot drawdown
        ax2.fill_between(x_values, 0, drawdown_pct.values, alpha=0.3, 
                        label=name, color=color)
        ax2.plot(x_values, drawdown_pct.values, linewidth=1.5, 
                color=color, linestyle=linestyle)
    
    ax1.set_title(f'{title} - Equity Curves', fontsize=_get_fontsize('title'), fontweight='bold')
    if show_pct_return:
        ax1.set_ylabel('Portfolio Return (%)', fontsize=_get_fontsize('label'), fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        ax1.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    else:
        ax1.set_ylabel('Portfolio Value (EUR)', fontsize=_get_fontsize('label'), fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))
    ax1.tick_params(labelsize=_get_fontsize('tick'))
    ax1.legend(loc='best', framealpha=0.9, fontsize=_get_fontsize('legend'))
    ax1.grid(alpha=0.3)
    
    ax2.set_title('Drawdown (% of Peak)', fontsize=_get_fontsize('title'), fontweight='bold')
    ax2.set_xlabel('Time', fontsize=_get_fontsize('label'), fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=_get_fontsize('label'), fontweight='bold')
    ax2.tick_params(labelsize=_get_fontsize('tick'))
    ax2.legend(loc='best', framealpha=0.9, fontsize=_get_fontsize('legend'))
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    
    # Get equity from last iteration for date formatting
    if equity_curves:
        first_equity = next(iter(equity_curves.values()))
        if isinstance(first_equity.index, pd.DatetimeIndex):
            plt.gcf().autofmt_xdate()
    
    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig, (ax1, ax2)


def plot_event_importance_heatmap(
    topic_counts_df: pd.DataFrame,
    spread_series: pd.Series,
    rolling_window_weeks: int = 4,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot event importance heatmap showing rolling correlation between topic counts and spread changes.
    
    Similar to Chakraborty et al. style visualization, this shows how different news event types
    vary in importance for price prediction over time periods.
    
    Args:
        topic_counts_df: DataFrame with datetime index and topic columns (from compute_time_decayed_topic_counts)
        spread_series: Series with datetime index representing price spread (real_spread_abs)
        rolling_window_weeks: Number of weeks for rolling correlation window (default: 4)
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot
    
    Returns:
        DataFrame with correlation values (topics x weeks)
    """
    from scipy.stats import pearsonr
    
    # Align indices and resample to weekly frequency
    topic_weekly = topic_counts_df.resample('W').mean()
    spread_weekly = spread_series.resample('W').mean()
    
    # Compute spread changes (week-over-week difference)
    spread_change = spread_weekly.diff()
    
    # Align time indices
    common_index = topic_weekly.index.intersection(spread_change.index)
    topic_weekly = topic_weekly.loc[common_index]
    spread_change = spread_change.loc[common_index]
    
    # Compute rolling correlations for each topic
    window_size = rolling_window_weeks
    correlation_matrix = []
    topic_names = []
    
    for topic_col in topic_counts_df.columns:
        topic_series = topic_weekly[topic_col]
        
        # Compute rolling correlation
        rolling_corrs = []
        for i in range(window_size, len(common_index)):
            window_topic = topic_series.iloc[i-window_size:i]
            window_spread = spread_change.iloc[i-window_size:i]
            
            # Remove NaN values
            valid_mask = ~(window_topic.isna() | window_spread.isna())
            if valid_mask.sum() < 3:  # Need at least 3 points for correlation
                rolling_corrs.append(np.nan)
            else:
                corr, _ = pearsonr(window_topic[valid_mask], window_spread[valid_mask])
                rolling_corrs.append(corr)
        
        # Pad with NaN for initial weeks
        rolling_corrs = [np.nan] * window_size + rolling_corrs
        correlation_matrix.append(rolling_corrs)
        topic_names.append(topic_col)
    
    # Create DataFrame with correlations
    corr_df = pd.DataFrame(
        correlation_matrix,
        index=topic_names,
        columns=common_index
    )
    
    # Normalize correlations to [0, 1] scale for heatmap visualization
    # Use absolute value and normalize per row
    corr_abs = corr_df.abs()
    corr_normalized = corr_abs.div(corr_abs.max(axis=1), axis=0).fillna(0)
    
    # Calculate figure size based on number of topics
    n_topics = len(topic_names)
    fig_height = max(8, min(n_topics * 0.6, 14))  # 0.6 inch per topic, capped
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    # Create heatmap
    im = ax.imshow(
        corr_normalized.values,
        aspect='auto',
        cmap='Reds',
        interpolation='nearest',
        vmin=0,
        vmax=1
    )
    
    # Truncate labels consistently using helper function
    truncated_labels = [_truncate_label(name, max_chars=45) for name in topic_names]
    
    # Set ticks and labels
    ax.set_yticks(range(len(topic_names)))
    ax.set_yticklabels(truncated_labels, fontsize=_get_fontsize('tick'))
    
    # Format x-axis with weekly labels (show ~15 labels max for readability)
    n_weeks = len(common_index)
    tick_step = max(1, n_weeks // 15)
    ax.set_xticks(range(0, n_weeks, tick_step))
    ax.set_xticklabels(
        [common_index[i].strftime('%Y-%m-%d') for i in range(0, n_weeks, tick_step)],
        rotation=45,
        ha='right',
        fontsize=_get_fontsize('tick') - 1
    )
    
    # Add colorbar with proper sizing
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.8)
    cbar.set_label('Event Importance (Normalized Correlation)', 
                   fontweight='bold', fontsize=_get_fontsize('label'))
    cbar.ax.tick_params(labelsize=_get_fontsize('tick'))
    
    # Add labels
    ax.set_xlabel('Time Period (Weekly)', fontweight='bold', fontsize=_get_fontsize('label'))
    ax.set_ylabel('Event Type', fontweight='bold', fontsize=_get_fontsize('label'))
    ax.set_title(
        f'Variation in Event Importance Over Time\n'
        f'(Rolling {rolling_window_weeks}-week correlation with spread changes)',
        fontweight='bold',
        fontsize=_get_fontsize('title'),
        pad=15
    )
    
    # Use tight_layout with padding
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return corr_df


def plot_news_shocks_vs_price(
    news_df: pd.DataFrame,
    spread_series: pd.Series,
    price_series: pd.Series = None,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot positive and negative news shocks compared with price/spread series.
    
    Creates dual-axis charts showing news shock counts (bars) overlaid with
    price/spread series (line) over time, similar to academic research visualizations.
    
    Args:
        news_df: DataFrame with datetime index and 'classification' column
        spread_series: Series with datetime index representing price spread (real_spread_abs)
        price_series: Optional series for price (Spot Price). If None, uses spread_series
        save_path: Optional filename to save (without extension)
        show: Whether to display the plot
    
    Returns:
        DataFrame with weekly news shock counts
    """
    from config import pipeline_config as cfg
    
    # Ensure news_df has DatetimeIndex
    news_df_copy = news_df.copy()
    if not isinstance(news_df_copy.index, pd.DatetimeIndex):
        if 'publishedAt' in news_df_copy.columns:
            news_df_copy['publishedAt'] = pd.to_datetime(news_df_copy['publishedAt'])
            news_df_copy = news_df_copy.set_index('publishedAt')
        else:
            raise ValueError("news_df must have a DatetimeIndex or 'publishedAt' column")
    
    # Remove timezone if present
    if news_df_copy.index.tz is not None:
        news_df_copy.index = news_df_copy.index.tz_localize(None)
    
    # Get valence mapping
    valence_map = cfg.TOPIC_VALENCE_MAP
    
    # Map classifications to valence
    news_df_copy['valence'] = news_df_copy['classification'].map(valence_map).fillna(0)
    
    # Filter out neutral/other articles for shock visualization
    news_shocks = news_df_copy[news_df_copy['valence'] != 0].copy()
    
    # Resample to weekly frequency
    news_weekly = news_shocks.resample('W').agg({
        'valence': lambda x: (x == 1).sum()  # Count positive
    })
    news_weekly['negative_count'] = news_shocks.resample('W').agg({
        'valence': lambda x: (x == -1).sum()
    })['valence']
    news_weekly['positive_count'] = news_weekly['valence']
    news_weekly = news_weekly.drop(columns=['valence'])
    
    # Resample price/spread series to weekly
    if price_series is not None:
        price_weekly = price_series.resample('W').mean()
        price_label = 'Price'
        price_unit = 'EUR/MWh'
    else:
        price_weekly = spread_series.resample('W').mean()
        price_label = 'Spread'
        price_unit = 'EUR/MWh'
    
    # Align indices
    common_index = news_weekly.index.intersection(price_weekly.index)
    news_weekly = news_weekly.loc[common_index]
    price_weekly = price_weekly.loc[common_index]
    
    # Calculate dynamic bar width based on data range
    if len(common_index) > 1:
        avg_gap = (common_index[-1] - common_index[0]).days / len(common_index)
        bar_width = max(3, min(avg_gap * 0.8, 6))  # 3-6 days width
    else:
        bar_width = 5
    
    # Create figure with two subplots - slightly taller for better spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    plt.subplots_adjust(hspace=0.15)  # Reduce space between subplots
    
    # Plot 1: Positive news shocks
    ax1_twin = ax1.twinx()
    
    # Bars for positive news
    ax1.bar(
        news_weekly.index,
        news_weekly['positive_count'],
        width=bar_width,
        color='#D95F02',  # Orange for positive (matches semantic color scheme)
        alpha=0.7,
        label='Positive News Shocks'
    )
    
    # Line for price/spread
    ax1_twin.plot(
        price_weekly.index,
        price_weekly.values,
        color='black',
        linewidth=2,
        label=f'{price_label} ({price_unit})'
    )
    
    ax1.set_ylabel('Number of Positive News Shocks', fontweight='bold', 
                   fontsize=_get_fontsize('label'), color='#D95F02')
    ax1_twin.set_ylabel(f'{price_label} ({price_unit})', fontweight='bold', 
                        fontsize=_get_fontsize('label'))
    ax1.set_title('Positive News Shocks and Price/Spread', fontweight='bold', 
                  fontsize=_get_fontsize('title'))
    ax1.tick_params(axis='y', labelcolor='#D95F02', labelsize=_get_fontsize('tick'))
    ax1_twin.tick_params(axis='y', labelsize=_get_fontsize('tick'))
    ax1.grid(alpha=0.3, axis='y')
    
    # Combine legends into one box
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
               framealpha=0.9, fontsize=_get_fontsize('legend'))
    
    # Plot 2: Negative news shocks
    ax2_twin = ax2.twinx()
    
    # Bars for negative news
    ax2.bar(
        news_weekly.index,
        news_weekly['negative_count'],
        width=bar_width,
        color='#1B9E77',  # Teal for negative (inverted semantic)
        alpha=0.7,
        label='Negative News Shocks'
    )
    
    # Line for price/spread
    ax2_twin.plot(
        price_weekly.index,
        price_weekly.values,
        color='black',
        linewidth=2,
        label=f'{price_label} ({price_unit})'
    )
    
    ax2.set_xlabel('Time', fontweight='bold', fontsize=_get_fontsize('label'))
    ax2.set_ylabel('Number of Negative News Shocks', fontweight='bold', 
                   fontsize=_get_fontsize('label'), color='#1B9E77')
    ax2_twin.set_ylabel(f'{price_label} ({price_unit})', fontweight='bold', 
                        fontsize=_get_fontsize('label'))
    ax2.set_title('Negative News Shocks and Price/Spread', fontweight='bold', 
                  fontsize=_get_fontsize('title'))
    ax2.tick_params(axis='y', labelcolor='#1B9E77', labelsize=_get_fontsize('tick'))
    ax2_twin.tick_params(axis='y', labelsize=_get_fontsize('tick'))
    ax2.tick_params(axis='x', labelsize=_get_fontsize('tick'))
    ax2.grid(alpha=0.3, axis='y')
    
    # Combine legends into one box
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
               framealpha=0.9, fontsize=_get_fontsize('legend'))
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return news_weekly
