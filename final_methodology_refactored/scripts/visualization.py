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


def plot_confusion_matrices(models_dict, y_test, class_labels=None, label_encoder=None):
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


def plot_cumulative_returns(
    returns_map,
    title="Cumulative Returns Comparison",
    ylabel="Cumulative Return",
    figsize=(14, 8),
    dpi=100,
    show=True,
):
    """
    Plot cumulative returns for multiple strategies.

    Args:
        returns_map: dict[str, pd.Series]
            Dictionary mapping strategy name to return series.
        title: str
            Plot title.
        ylabel: str
            Y-axis label.
        figsize: tuple
            Figure size in inches.
        dpi: int
            Figure resolution.
        show: bool
            Whether to call plt.show() at the end.

    Returns:
        Tuple of (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
    linestyles = ['-', '--', '-.', ':']

    for idx, (name, returns) in enumerate(returns_map.items()):
        cumulative = returns.cumsum()
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        ax.plot(cumulative, label=name, color=color, linestyle=linestyle, linewidth=2, alpha=0.8)

    ax.set_xlabel("Time Period")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

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


def plot_eda_dashboard(master_df: pd.DataFrame, news_df: pd.DataFrame):
    """
    Generate comprehensive EDA dashboard with multiple panels.

    Args:
        master_df: Master dataframe with energy and target data
        news_df: News dataframe with classification data
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Target distribution
    ax1 = fig.add_subplot(gs[0, 0])
    target_col = 'spread_target_shift_24'
    if target_col in master_df.columns:
        target_counts = master_df[target_col].value_counts().sort_index()
        colors = ['red', 'gray', 'green']
        ax1.bar(target_counts.index, target_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Target Class', fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        ax1.set_title('Target Distribution', fontweight='bold', fontsize=12)
        ax1.grid(alpha=0.3, axis='y')
        for i, (idx, val) in enumerate(target_counts.items()):
            pct = val / target_counts.sum() * 100
            ax1.text(idx, val, f'{val}\n({pct:.1f}%)', ha='center', va='bottom')
    else:
        ax1.text(0.5, 0.5, f'Column "{target_col}"\nnot found', ha='center', va='center', fontsize=10)
        ax1.set_title('Target Distribution', fontweight='bold', fontsize=12)

    # 2. Price spreads over time
    ax2 = fig.add_subplot(gs[0, 1:])
    if 'real_spread_abs' in master_df.columns:
        spread_series = master_df['real_spread_abs']
        ax2.plot(spread_series.index, spread_series.values, linewidth=0.5, alpha=0.7)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Time', fontweight='bold')
        ax2.set_ylabel('Spread (EUR/MWh)', fontweight='bold')
        ax2.set_title('Price Spread Over Time (Spot - Day Ahead)', fontweight='bold', fontsize=12)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Column "real_spread_abs"\nnot found', ha='center', va='center', fontsize=10)
        ax2.set_title('Price Spread Over Time', fontweight='bold', fontsize=12)

    # 3. News classification distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if 'classification' in news_df.columns and len(news_df) > 0:
        class_counts = news_df['classification'].value_counts().head(8)
        ax3.barh(range(len(class_counts)), class_counts.values, color='steelblue', alpha=0.7)
        ax3.set_yticks(range(len(class_counts)))
        ax3.set_yticklabels([label[:40] + '...' if len(label) > 40 else label
                             for label in class_counts.index], fontsize=9)
        ax3.set_xlabel('Count', fontweight='bold')
        ax3.set_title('Top 8 News Classifications', fontweight='bold', fontsize=12)
        ax3.grid(alpha=0.3, axis='x')
    else:
        ax3.text(0.5, 0.5, 'No news classification\ndata available', ha='center', va='center', fontsize=10)
        ax3.set_title('Top 8 News Classifications', fontweight='bold', fontsize=12)

    # 4. Spread volatility analysis (rolling standard deviation)
    ax4 = fig.add_subplot(gs[1, 1])
    if 'real_spread_abs' in master_df.columns and len(master_df) > 168:  # Need enough data for rolling
        spread_series = master_df['real_spread_abs'].dropna()
        # Calculate rolling std with different windows
        rolling_24h = spread_series.rolling(window=24, min_periods=1).std()
        rolling_168h = spread_series.rolling(window=168, min_periods=1).std()  # 1 week

        ax4.plot(rolling_24h.index, rolling_24h.values, label='24h window', linewidth=1.5, alpha=0.8)
        ax4.plot(rolling_168h.index, rolling_168h.values, label='1-week window', linewidth=1.5, alpha=0.8)
        ax4.set_xlabel('Time', fontweight='bold')
        ax4.set_ylabel('Volatility (EUR/MWh)', fontweight='bold')
        ax4.set_title('Spread Volatility Over Time', fontweight='bold', fontsize=12)
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for\nvolatility analysis', ha='center', va='center', fontsize=10)
        ax4.set_title('Spread Volatility Over Time', fontweight='bold', fontsize=12)

    # 5. Load distribution
    ax5 = fig.add_subplot(gs[1, 2])
    if 'Load' in master_df.columns:
        load_series = master_df['Load'].dropna()
        if len(load_series) > 0:
            ax5.hist(load_series, bins=50, color='orange', alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Load (MW)', fontweight='bold')
            ax5.set_ylabel('Frequency', fontweight='bold')
            ax5.set_title('Load Distribution', fontweight='bold', fontsize=12)
            ax5.grid(alpha=0.3, axis='y')
            ax5.axvline(load_series.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'No Load data\navailable', ha='center', va='center', fontsize=10)
            ax5.set_title('Load Distribution', fontweight='bold', fontsize=12)
    else:
        ax5.text(0.5, 0.5, 'Column "Load"\nnot found', ha='center', va='center', fontsize=10)
        ax5.set_title('Load Distribution', fontweight='bold', fontsize=12)

    # 6. Hourly price patterns
    ax6 = fig.add_subplot(gs[2, 0])
    if 'hour' in master_df.columns and 'Spot Price' in master_df.columns:
        hourly_price = master_df.groupby('hour')['Spot Price'].mean()
        ax6.plot(hourly_price.index, hourly_price.values, marker='o', linewidth=2, markersize=4)
        ax6.set_xlabel('Hour of Day', fontweight='bold')
        ax6.set_ylabel('Average Spot Price (EUR/MWh)', fontweight='bold')
        ax6.set_title('Average Spot Price by Hour', fontweight='bold', fontsize=12)
        ax6.grid(alpha=0.3)
        ax6.set_xticks(range(0, 24, 3))
    else:
        missing_cols = []
        if 'hour' not in master_df.columns:
            missing_cols.append('hour')
        if 'Spot Price' not in master_df.columns:
            missing_cols.append('Spot Price')
        ax6.text(0.5, 0.5, f'Missing columns:\n{", ".join(missing_cols)}', ha='center', va='center', fontsize=10)
        ax6.set_title('Average Spot Price by Hour', fontweight='bold', fontsize=12)

    # 7. Day of week patterns
    ax7 = fig.add_subplot(gs[2, 1])
    if 'day_of_week' in master_df.columns and 'real_spread_abs' in master_df.columns:
        dow_spread = master_df.groupby('day_of_week')['real_spread_abs'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax7.bar(range(7), dow_spread.values, color='teal', alpha=0.7, edgecolor='black')
        ax7.set_xticks(range(7))
        ax7.set_xticklabels(days)
        ax7.set_ylabel('Average Spread (EUR/MWh)', fontweight='bold')
        ax7.set_title('Average Spread by Day of Week', fontweight='bold', fontsize=12)
        ax7.grid(alpha=0.3, axis='y')
    else:
        missing_cols = []
        if 'day_of_week' not in master_df.columns:
            missing_cols.append('day_of_week')
        if 'real_spread_abs' not in master_df.columns:
            missing_cols.append('real_spread_abs')
        ax7.text(0.5, 0.5, f'Missing columns:\n{", ".join(missing_cols)}', ha='center', va='center', fontsize=10)
        ax7.set_title('Average Spread by Day of Week', fontweight='bold', fontsize=12)

    # 8. News volume over time
    ax8 = fig.add_subplot(gs[2, 2])
    if hasattr(news_df.index, 'to_period') and len(news_df) > 0:
        news_daily = news_df.groupby(news_df.index.to_period('D')).size()
        ax8.plot(news_daily.index.to_timestamp(), news_daily.values, linewidth=1.5, color='purple')
        ax8.set_xlabel('Date', fontweight='bold')
        ax8.set_ylabel('Number of Articles', fontweight='bold')
        ax8.set_title('News Volume Over Time', fontweight='bold', fontsize=12)
        ax8.grid(alpha=0.3)
        fig.autofmt_xdate()
    else:
        ax8.text(0.5, 0.5, 'No time-indexed\nnews data available', ha='center', va='center', fontsize=10)
        ax8.set_title('News Volume Over Time', fontweight='bold', fontsize=12)

    fig.suptitle('Exploratory Data Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.show()


def plot_embedding_quality(news_df: pd.DataFrame, n_samples: int = 500, perplexity: int = 30):
    """
    Visualize embedding quality using UMAP/t-SNE dimensionality reduction.

    Args:
        news_df: News dataframe with 'embedding' and 'classification' columns
        n_samples: Number of samples to visualize (for performance)
        perplexity: t-SNE perplexity parameter
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

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # UMAP visualization
    try:
        import umap
        reducer_umap = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding_2d_umap = reducer_umap.fit_transform(embeddings)

        if labels is not None:
            unique_labels = pd.Series(labels).value_counts().head(10).index
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[0].scatter(embedding_2d_umap[mask, 0], embedding_2d_umap[mask, 1],
                              c=[colors[i]], label=label[:30], alpha=0.3, s=100)
            axes[0].legend(loc='best', fontsize=8, ncol=2)
        else:
            axes[0].scatter(embedding_2d_umap[:, 0], embedding_2d_umap[:, 1],
                          c='steelblue', alpha=0.3, s=100)

        axes[0].set_xlabel('UMAP Dimension 1', fontweight='bold')
        axes[0].set_ylabel('UMAP Dimension 2', fontweight='bold')
        axes[0].set_title('Embedding Visualization (UMAP)', fontweight='bold', fontsize=14)
        axes[0].grid(alpha=0.3)
    except ImportError:
        axes[0].text(0.5, 0.5, 'UMAP not installed\npip install umap-learn',
                    ha='center', va='center', fontsize=12)
        axes[0].set_title('UMAP (Not Available)', fontweight='bold')

    # t-SNE visualization
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(embeddings)-1))
        embedding_2d_tsne = tsne.fit_transform(embeddings)

        if labels is not None:
            unique_labels = pd.Series(labels).value_counts().head(10).index
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[1].scatter(embedding_2d_tsne[mask, 0], embedding_2d_tsne[mask, 1],
                              c=[colors[i]], label=label[:30], alpha=0.3, s=100)
            axes[1].legend(loc='best', fontsize=8, ncol=2)
        else:
            axes[1].scatter(embedding_2d_tsne[:, 0], embedding_2d_tsne[:, 1],
                          c='coral', alpha=0.3, s=100)

        axes[1].set_xlabel('t-SNE Dimension 1', fontweight='bold')
        axes[1].set_ylabel('t-SNE Dimension 2', fontweight='bold')
        axes[1].set_title('Embedding Visualization (t-SNE)', fontweight='bold', fontsize=14)
        axes[1].grid(alpha=0.3)
    except Exception as e:
        axes[1].text(0.5, 0.5, f't-SNE failed:\n{str(e)}', ha='center', va='center', fontsize=10)
        axes[1].set_title('t-SNE (Failed)', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_learning_curves(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splitter,
    scoring: str = 'f1_macro',
    model_name: str = 'Model',
    train_sizes: np.ndarray = None
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

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot training and validation scores
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', linewidth=2,
               markersize=6, label='Training score')
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color='blue')

        ax.plot(train_sizes_abs, val_mean, 'o-', color='red', linewidth=2,
               markersize=6, label='Validation score')
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color='red')

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
        plt.show()

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
    pct_cost_scenarios: dict[str, tuple[float, float]] = None
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
    """
    from . import evaluation

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

        returns_summary_cost = evaluation.summarise_strategy_set(strategy_returns_cost)
        print(f"\n{returns_summary_cost[['Total Return', 'Sharpe (annualised)', 'Sortino (annualised)']].to_string()}")

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
        temp_summary = evaluation.summarise_strategy_set(temp_returns)

        for name in strategy_names:
            strategy_returns_by_cost[name].append(temp_returns[name].sum())
            strategy_sharpes_by_cost[name].append(temp_summary.loc[name, "Sharpe (annualised)"])

    # Plot 1: Total return vs cost
    colors = ['steelblue', 'coral', 'green']
    markers = ['o', 's', '^']

    for idx, name in enumerate(strategy_names):
        ax1.plot(cost_range, strategy_returns_by_cost[name],
                marker=markers[idx % len(markers)], linewidth=2, markersize=8,
                color=colors[idx % len(colors)], label=name.replace("LightGBM ", ""))

    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Fixed Transaction Cost (EUR/MWh)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Total Return (EUR/MWh)', fontweight='bold', fontsize=12)
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
    plt.show()

    print("\n" + "=" * 80)
