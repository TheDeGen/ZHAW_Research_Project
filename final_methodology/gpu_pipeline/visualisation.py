"""Visualization utilities extracted from the `final_v3.ipynb` notebook.

These helpers centralise plotting logic so that notebooks can focus on the
high-level workflow while reusing consistent styling and diagnostics.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def configure_visual_defaults(style: str = "whitegrid", figure_dpi: int = 100) -> None:
    """Set sensible default styling for seaborn/matplotlib plots."""
    sns.set_style(style)
    plt.rcParams["figure.dpi"] = figure_dpi


def plot_feature_importance(
    model,
    feature_names: Sequence[str],
    model_name: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Plot feature importance for tree-based models that expose ``feature_importances_``.

    Parameters
    ----------
    model:
        Fitted estimator exposing a ``feature_importances_`` attribute (e.g., LightGBM, XGBoost).
    feature_names:
        Ordered list of feature names aligned with the model's training matrix.
    model_name:
        Friendly label used in the plot title and printed diagnostics.
    top_n:
        Number of top features to visualise.

    Returns
    -------
    pandas.DataFrame
        DataFrame of all features sorted by importance (descending). The plot only displays
        the top ``top_n`` rows, but the full ranking is returned for further analysis.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not expose feature_importances_.")
        return pd.DataFrame(columns=["feature", "importance"])

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(
            "Feature name count ({}) does not match model importances ({}). "
            "Ensure `feature_names` reflects the training columns.".format(
                len(feature_names), len(importances)
            )
        )

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top_df = importance_df.head(top_n)

    plt.figure(figsize=(10, max(6, len(top_df) * 0.4)))
    sns.barplot(data=top_df, x="importance", y="feature", palette="Blues_r")
    plt.title(f"{model_name} – Top {len(top_df)} Features (Gain)")
    plt.xlabel("Importance (Gain)")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    return importance_df


def plot_confusion_matrices(
    models_dict: Mapping[str, Tuple[object, Sequence]],
    y_test,
    class_labels: Sequence,
    label_encoder: Optional[LabelEncoder] = None,
) -> None:
    """Plot confusion matrices for a collection of fitted estimators.

    Parameters
    ----------
    models_dict:
        Mapping of model name to ``(estimator, X_test)`` pairs.
    y_test:
        Ground-truth labels aligned with the provided feature matrices.
    class_labels:
        Ordered sequence of class labels that should appear in every matrix.
    label_encoder:
        Optional label encoder used to inverse-transform predictions prior to evaluation.
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
        disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
        ax.set_title(name)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    plt.tight_layout()
    plt.show()


def _resolve_first_existing_name(possible_names: Iterable[str], columns: Sequence[str]) -> Optional[str]:
    """Return the first candidate name present in the provided column sequence."""
    for name in possible_names:
        if name in columns:
            return name
    return None


def plot_eda_dashboard(
    master_df: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None,
    *,
    day_ahead_candidates: Optional[Sequence[str]] = None,
    spot_price_candidates: Optional[Sequence[str]] = None,
    target_candidates: Optional[Sequence[str]] = None,
) -> None:
    """Render the exploratory data analysis dashboard previously embedded in the notebook.

    Parameters
    ----------
    master_df:
        Primary dataframe containing price, load, and engineered features.
    news_df:
        Optional dataframe of news articles used to derive signal features.
    day_ahead_candidates / spot_price_candidates / target_candidates:
        Candidate column names used to resolve the appropriate series in ``master_df``.
    """
    configure_visual_defaults()

    day_ahead_candidates = day_ahead_candidates or [
        "day_ahead_price",
        "Day Ahead Auction",
        "day ahead price",
        "Day-ahead Price",
        "day_ahead_auction",
    ]
    spot_price_candidates = spot_price_candidates or [
        "spot_price",
        "Spot Price",
        "spot price",
    ]
    target_candidates = target_candidates or [
        "spread_target_shift_24",
        "spread_target",
        "target",
    ]

    day_ahead_col = _resolve_first_existing_name(day_ahead_candidates, master_df.columns)
    spot_price_col = _resolve_first_existing_name(spot_price_candidates, master_df.columns)

    if day_ahead_col is None or spot_price_col is None:
        raise KeyError(
            "Could not find both required price columns in master_df. "
            f"Found: {day_ahead_col} and {spot_price_col}"
        )

    if "timestamp" in master_df.columns:
        time_axis = pd.to_datetime(master_df["timestamp"])
    elif "Timestamp" in master_df.columns:
        time_axis = pd.to_datetime(master_df["Timestamp"])
    elif isinstance(master_df.index, pd.DatetimeIndex):
        time_axis = master_df.index
    else:
        raise KeyError("No timestamp column or datetime index found in master_df.")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Comprehensive Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.00)

    axes[0, 0].plot(time_axis, master_df[day_ahead_col], label="Day-ahead Price", alpha=0.7, linewidth=0.8)
    axes[0, 0].plot(time_axis, master_df[spot_price_col], label="Spot Price", alpha=0.7, linewidth=0.8)
    axes[0, 0].set_title("Price Time Series", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Price (EUR/MWh)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    spread_series = master_df[day_ahead_col] - master_df[spot_price_col]
    spread_data = spread_series.dropna()
    axes[0, 1].hist(spread_data, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Spread Distribution", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Spread (Day-ahead - Spot, EUR/MWh)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero spread")
    axes[0, 1].legend()

    target_col = _resolve_first_existing_name(target_candidates, master_df.columns)
    if target_col is not None:
        target_counts = master_df[target_col].value_counts().sort_index()
        color_palette = ["#d62728", "#ff7f0e", "#2ca02c"]
        if len(target_counts) > 3:
            color_palette = color_palette + ["#9467bd", "#8c564b"]
        axes[1, 0].bar(
            target_counts.index,
            target_counts.values,
            edgecolor="black",
            alpha=0.7,
            color=color_palette[: len(target_counts)],
        )
        axes[1, 0].set_title("Target Class Distribution", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Target Class")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_xticks(target_counts.index)
        for i, value in enumerate(target_counts.values):
            axes[1, 0].text(
                target_counts.index[i],
                value,
                f"{value:,}\n({value / len(master_df) * 100:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    else:
        axes[1, 0].text(0.5, 0.5, "Target column not found", ha="center", va="center", fontsize=12)
        axes[1, 0].set_title("Target Class Distribution", fontsize=12, fontweight="bold")

    if news_df is not None and len(news_df) > 0:
        if isinstance(news_df.index, pd.DatetimeIndex):
            articles_per_day = news_df.resample("D").size()
        else:
            time_columns = [c for c in news_df.columns if "time" in c.lower() or "date" in c.lower()]
            if time_columns:
                articles_per_day = (
                    news_df.set_index(pd.to_datetime(news_df[time_columns[0]])).resample("D").size()
                )
            else:
                articles_per_day = pd.Series([len(news_df)], index=[pd.Timestamp.now()])

        axes[1, 1].plot(articles_per_day.index, articles_per_day.values, linewidth=1.5)
        axes[1, 1].set_title("News Articles per Day", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Date")
        axes[1, 1].set_ylabel("Article Count")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "News data not available", ha="center", va="center", fontsize=12)
        axes[1, 1].set_title("News Articles per Day", fontsize=12, fontweight="bold")

    energy_features = [col for col in [day_ahead_col, spot_price_col] if col is not None]
    for col in ["solar", "wind_onshore", "wind_offshore", "biomass", "hydro"]:
        if col in master_df.columns:
            energy_features.append(col)

    if len(energy_features) >= 2:
        corr_matrix = master_df[energy_features].corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=axes[2, 0],
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
        )
        axes[2, 0].set_title("Energy Feature Correlation", fontsize=12, fontweight="bold")
    else:
        axes[2, 0].text(
            0.5,
            0.5,
            "Insufficient features for correlation",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[2, 0].set_title("Energy Feature Correlation", fontsize=12, fontweight="bold")

    axes[2, 1].plot(time_axis, spread_series, linewidth=0.8, alpha=0.7)
    axes[2, 1].set_title("Spread Over Time", fontsize=12, fontweight="bold")
    axes[2, 1].set_xlabel("Date")
    axes[2, 1].set_ylabel("Spread (EUR/MWh)")
    axes[2, 1].axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=2)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n{'=' * 60}")
    print("DATA SUMMARY")
    print(f"{'=' * 60}")
    print(f"Date range: {time_axis.min()} to {time_axis.max()}")
    print(f"Total hours: {len(master_df):,}")
    if news_df is not None:
        print(f"Total news articles: {len(news_df):,}")
    if target_col is not None:
        print("\nTarget distribution:")
        for idx, count in master_df[target_col].value_counts().sort_index().items():
            if isinstance(idx, (int, np.integer)) or (isinstance(idx, float) and float(idx).is_integer()):
                idx_str = f"{int(idx):2d}"
            else:
                idx_str = f"{idx}"
            print(f"  Class {idx_str}: {count:6,} ({count / len(master_df) * 100:5.1f}%)")
    else:
        print("\nTarget column not found for distribution statistics.")

    print("\nPrice statistics (EUR/MWh):")
    print(
        f"  Day-ahead: mean={master_df[day_ahead_col].mean():.2f}, "
        f"std={master_df[day_ahead_col].std():.2f}"
    )
    print(
        f"  Spot:      mean={master_df[spot_price_col].mean():.2f}, "
        f"std={master_df[spot_price_col].std():.2f}"
    )
    print(f"  Spread:    mean={spread_data.mean():.2f}, std={spread_data.std():.2f}")
    print(f"{'=' * 60}\n")


# ============================================================================
# PHASE 3: NLP DIAGNOSTIC VISUALIZATIONS
# ============================================================================


def plot_news_feature_diagnostics(
    master_df: pd.DataFrame,
    news_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    price_col: str = "day_ahead_price",
    target_col: str = "spread_target_shift_24",
    timestamp_col: str = "timestamp",
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 100,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Generate comprehensive diagnostics for news-derived features.

    This function creates a 4-panel dashboard to diagnose the quality and
    predictive value of NLP features in the energy trading context.

    Parameters
    ----------
    master_df : pd.DataFrame
        Primary dataframe containing energy prices, features, and targets.
    news_df : pd.DataFrame
        News articles dataframe with timestamp and topic columns.
    feature_cols : Sequence[str]
        List of news-derived feature column names to analyze.
    price_col : str, default="day_ahead_price"
        Column name for the price series used to compute volatility.
    target_col : str, default="spread_target_shift_24"
        Target column for signal classification (Long/Neutral/Short).
    timestamp_col : str, default="timestamp"
        Column name for timestamp in both dataframes.
    figsize : Tuple[int, int], default=(16, 12)
        Figure size in inches.
    dpi : int, default=100
        Figure resolution.
    show : bool, default=True
        Whether to call plt.show() at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    axes : numpy.ndarray
        Array of subplot axes (2x2 grid).

    Examples
    --------
    >>> fig, axes = plot_news_feature_diagnostics(
    ...     master_df=master,
    ...     news_df=news,
    ...     feature_cols=['news_sentiment', 'topic_energy_decay'],
    ... )
    """
    if timestamp_col not in master_df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found in master_df")
    if price_col not in master_df.columns:
        raise KeyError(f"Price column '{price_col}' not found in master_df")

    # Ensure timestamp is datetime
    master_time = pd.to_datetime(master_df[timestamp_col])

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle("NLP Feature Diagnostics Dashboard", fontsize=16, fontweight="bold", y=0.995)

    # Panel 1: News article counts vs price volatility
    if isinstance(news_df.index, pd.DatetimeIndex):
        news_daily = news_df.resample("D").size()
    else:
        news_time_cols = [c for c in news_df.columns if "time" in c.lower() or "date" in c.lower()]
        if news_time_cols:
            news_daily = news_df.set_index(pd.to_datetime(news_df[news_time_cols[0]])).resample("D").size()
        else:
            news_daily = pd.Series([], dtype=int)

    # Compute rolling price volatility
    master_daily = master_df.set_index(master_time).resample("D")[price_col].std()
    master_daily = master_daily.reindex(news_daily.index, fill_value=0)

    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    ax1.scatter(news_daily.index, news_daily.values, alpha=0.5, s=20, label="Article Count", color="steelblue")
    ax1_twin.plot(
        master_daily.index, master_daily.values, alpha=0.7, linewidth=1.5, label="Price Volatility", color="coral"
    )

    # Compute rolling correlation (30-day window)
    if len(news_daily) > 30:
        corr_series = news_daily.rolling(30).corr(master_daily)
        ax1.plot(
            corr_series.index,
            corr_series.values * news_daily.max() * 0.5,
            alpha=0.6,
            linewidth=2,
            color="purple",
            linestyle="--",
            label="30d Correlation (scaled)",
        )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Article Count", color="steelblue")
    ax1_twin.set_ylabel("Price Volatility (EUR/MWh)", color="coral")
    ax1.set_title("News Volume vs Price Volatility", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Heatmap of topic distribution over time
    topic_cols = [c for c in feature_cols if "topic" in c.lower()]
    if topic_cols and len(topic_cols) > 1:
        topic_df = master_df.set_index(master_time)[topic_cols].resample("W").mean()
        # Normalize for visualization
        topic_df_norm = topic_df.div(topic_df.sum(axis=1), axis=0).fillna(0)

        sns.heatmap(
            topic_df_norm.T,
            ax=axes[0, 1],
            cmap="YlOrRd",
            cbar_kws={"label": "Concentration"},
            xticklabels=10,
            yticklabels=True,
        )
        axes[0, 1].set_title("Topic Distribution Over Time (Weekly)", fontweight="bold")
        axes[0, 1].set_xlabel("Week")
        axes[0, 1].set_ylabel("Topic")
    else:
        axes[0, 1].text(0.5, 0.5, "Insufficient topic features for heatmap", ha="center", va="center", fontsize=12)
        axes[0, 1].set_title("Topic Distribution Over Time", fontweight="bold")

    # Panel 3: Sentiment score distribution by target class
    sentiment_cols = [c for c in feature_cols if "sentiment" in c.lower() or "embed" in c.lower()]
    if sentiment_cols and target_col in master_df.columns:
        sentiment_col = sentiment_cols[0]
        target_data = master_df[[sentiment_col, target_col]].dropna()

        # Map numeric targets to labels if needed
        target_map = {-1: "Short", 0: "Neutral", 1: "Long"}
        target_data["Target_Label"] = target_data[target_col].map(
            lambda x: target_map.get(x, str(x)) if x in target_map else str(x)
        )

        sns.violinplot(
            data=target_data, x="Target_Label", y=sentiment_col, ax=axes[1, 0], palette="Set2", inner="quartile"
        )
        axes[1, 0].set_title("Sentiment Distribution by Target Class", fontweight="bold")
        axes[1, 0].set_xlabel("Target Class")
        axes[1, 0].set_ylabel(sentiment_col)
        axes[1, 0].grid(True, alpha=0.3, axis="y")
    else:
        axes[1, 0].text(
            0.5, 0.5, "No sentiment features or target column available", ha="center", va="center", fontsize=12
        )
        axes[1, 0].set_title("Sentiment Distribution by Target Class", fontweight="bold")

    # Panel 4: Correlation matrix between news features and price spread
    if "spread" not in master_df.columns and price_col in master_df.columns:
        spot_candidates = ["spot_price", "Spot Price"]
        spot_col = None
        for candidate in spot_candidates:
            if candidate in master_df.columns:
                spot_col = candidate
                break
        if spot_col:
            spread = master_df[price_col] - master_df[spot_col]
        else:
            spread = master_df[price_col]
    else:
        spread = master_df.get("spread", master_df[price_col])

    valid_features = [c for c in feature_cols if c in master_df.columns]
    if valid_features:
        corr_df = master_df[valid_features].copy()
        corr_df["spread"] = spread
        corr_matrix = corr_df.corr()[["spread"]].drop("spread").sort_values(by="spread", ascending=False)

        sns.heatmap(
            corr_matrix,
            ax=axes[1, 1],
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            cbar_kws={"label": "Correlation"},
            vmin=-0.5,
            vmax=0.5,
        )
        axes[1, 1].set_title("News Features vs Price Spread Correlation", fontweight="bold")
        axes[1, 1].set_xlabel("")
    else:
        axes[1, 1].text(0.5, 0.5, "No valid news features found", ha="center", va="center", fontsize=12)
        axes[1, 1].set_title("News Features Correlation", fontweight="bold")

    plt.tight_layout()

    # Print summary statistics
    print(f"\n{'=' * 70}")
    print("NEWS FEATURE DIAGNOSTICS SUMMARY")
    print(f"{'=' * 70}")
    total_hours = len(master_df)
    if len(news_daily) > 0:
        zero_news_hours = (news_daily == 0).sum()
        print(f"Total news articles: {news_df.shape[0]:,}")
        print(f"Date range: {news_daily.index.min()} to {news_daily.index.max()}")
        print(f"Days with zero news coverage: {zero_news_hours} ({zero_news_hours/len(news_daily)*100:.1f}%)")
        print(f"Average articles per day: {news_daily.mean():.1f} (std: {news_daily.std():.1f})")
    else:
        print("No news data available for statistics")

    if len(valid_features) > 0:
        print(f"\nNews feature coverage: {len(valid_features)}/{len(feature_cols)} features found")
        null_pct = master_df[valid_features].isnull().mean() * 100
        print(f"Average null percentage: {null_pct.mean():.1f}%")
        if null_pct.max() > 50:
            print(f"⚠ Features with >50% nulls: {null_pct[null_pct > 50].index.tolist()}")

    print(f"{'=' * 70}\n")

    if show:
        plt.show()

    return fig, axes


def plot_embedding_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    *,
    perplexity: int = 30,
    n_neighbors: int = 15,
    random_state: int = 42,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualize embedding quality using dimensionality reduction and clustering metrics.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Target labels for coloring points (e.g., [-1, 0, 1] for Short/Neutral/Long).
    method : str, default="tsne"
        Dimensionality reduction method: "tsne" or "umap".
    perplexity : int, default=30
        t-SNE perplexity parameter (ignored for UMAP).
    n_neighbors : int, default=15
        UMAP n_neighbors parameter (ignored for t-SNE).
    random_state : int, default=42
        Random seed for reproducibility.
    figsize : Tuple[int, int], default=(12, 8)
        Figure size in inches.
    dpi : int, default=100
        Figure resolution.
    show : bool, default=True
        Whether to call plt.show() at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The subplot axes.

    Examples
    --------
    >>> fig, ax = plot_embedding_quality(
    ...     embeddings=news_embeddings,
    ...     labels=target_labels,
    ...     method='tsne',
    ... )
    """
    from sklearn.metrics import silhouette_score

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(f"Embeddings ({embeddings.shape[0]}) and labels ({labels.shape[0]}) must have same length")

    # Reduce to 2D
    if method.lower() == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1)
        embedding_2d = reducer.fit_transform(embeddings)
        method_label = "t-SNE"
    elif method.lower() == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
            embedding_2d = reducer.fit_transform(embeddings)
            method_label = "UMAP"
        except ImportError:
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    else:
        raise ValueError(f"Method '{method}' not supported. Choose 'tsne' or 'umap'.")

    # Compute silhouette score
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        silhouette = silhouette_score(embeddings, labels)
    else:
        silhouette = np.nan

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Map labels to colors
    label_map = {-1: "Short", 0: "Neutral", 1: "Long"}
    label_names = [label_map.get(lbl, str(lbl)) for lbl in labels]
    unique_names = [label_map.get(lbl, str(lbl)) for lbl in unique_labels]

    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap="coolwarm",
        s=20,
        alpha=0.6,
        edgecolors="k",
        linewidth=0.3,
    )

    # Add legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=scatter.cmap(scatter.norm(lbl)),
                          markersize=8, label=label_map.get(lbl, str(lbl)))
               for lbl in unique_labels]
    ax.legend(handles=handles, title="Target Class", loc="best")

    # Identify and highlight outliers (points far from centroid)
    if len(embedding_2d) > 10:
        centroid = np.mean(embedding_2d, axis=0)
        distances = np.linalg.norm(embedding_2d - centroid, axis=1)
        outlier_threshold = np.percentile(distances, 95)
        outliers = distances > outlier_threshold
        ax.scatter(
            embedding_2d[outliers, 0],
            embedding_2d[outliers, 1],
            s=100,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="Outliers (95th %ile)",
        )

    ax.set_xlabel(f"{method_label} Dimension 1")
    ax.set_ylabel(f"{method_label} Dimension 2")
    ax.set_title(
        f"Embedding Quality Visualization ({method_label})\nSilhouette Score: {silhouette:.3f}",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    print(f"\n{'=' * 70}")
    print("EMBEDDING QUALITY METRICS")
    print(f"{'=' * 70}")
    print(f"Dimensionality reduction: {method_label}")
    print(f"Original dimensions: {embeddings.shape[1]}")
    print(f"Reduced dimensions: 2")
    print(f"Number of samples: {embeddings.shape[0]}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Silhouette score: {silhouette:.4f}")
    if not np.isnan(silhouette):
        if silhouette > 0.5:
            print("✓ Strong cluster separation (silhouette > 0.5)")
        elif silhouette > 0.25:
            print("○ Moderate cluster separation (0.25 < silhouette < 0.5)")
        else:
            print("✗ Weak cluster separation (silhouette < 0.25)")
    print(f"{'=' * 70}\n")

    if show:
        plt.show()

    return fig, ax


def plot_nlp_feature_importance(
    model,
    feature_names: Sequence[str],
    nlp_prefix: str = "news_",
    *,
    top_n: int = 20,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 100,
    show: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Compare NLP feature importance against other feature categories.

    Parameters
    ----------
    model : object
        Trained model with `feature_importances_` attribute.
    feature_names : Sequence[str]
        List of feature names aligned with model training.
    nlp_prefix : str, default="news_"
        Prefix identifying NLP-derived features.
    top_n : int, default=20
        Number of top features to display in detailed view.
    figsize : Tuple[int, int], default=(14, 8)
        Figure size in inches.
    dpi : int, default=100
        Figure resolution.
    show : bool, default=True
        Whether to call plt.show() at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    axes : Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        Tuple of (left_axis, right_axis).

    Examples
    --------
    >>> fig, (ax1, ax2) = plot_nlp_feature_importance(
    ...     model=lgbm_model,
    ...     feature_names=feature_cols,
    ...     nlp_prefix='news_',
    ... )
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not have 'feature_importances_' attribute")

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(f"Feature count mismatch: {len(feature_names)} names vs {len(importances)} importances")

    # Create importance dataframe
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    # Categorize features
    def categorize_feature(name: str) -> str:
        name_lower = name.lower()
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
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

    # Print summary
    print(f"\n{'=' * 70}")
    print("NLP FEATURE IMPORTANCE SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total features: {len(importance_df)}")
    for cat in ["NLP", "Price", "Energy", "Other"]:
        cat_df = importance_df[importance_df["category"] == cat]
        cat_importance = cat_df["importance"].sum()
        cat_pct = (cat_importance / total_importance) * 100
        print(f"{cat:8s}: {len(cat_df):3d} features, {cat_importance:8.4f} importance ({cat_pct:5.1f}%)")

    nlp_importance = importance_df[importance_df["category"] == "NLP"]["importance"].sum()
    nlp_pct = (nlp_importance / total_importance) * 100
    if len(nlp_df) > 0:
        top5_nlp = importance_df[importance_df["category"] == "NLP"].head(5)
        top5_importance = top5_nlp["importance"].sum()
        top5_pct = (top5_importance / nlp_importance) * 100 if nlp_importance > 0 else 0
        print(f"\nTop 5 NLP features account for {top5_pct:.1f}% of total NLP importance")
        print(f"Top 5 NLP features: {', '.join(top5_nlp['feature'].tolist())}")
    print(f"{'=' * 70}\n")

    if show:
        plt.show()

    return fig, (ax1, ax2)


def plot_topic_impact_on_signals(
    master_df: pd.DataFrame,
    topic_cols: Sequence[str],
    signal_col: str = "spread_target_shift_24",
    threshold: float = 0.5,
    *,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 100,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Analyze which topics are most predictive of trading signals.

    Parameters
    ----------
    master_df : pd.DataFrame
        Dataframe containing topic features and target signals.
    topic_cols : Sequence[str]
        List of topic column names.
    signal_col : str, default="spread_target_shift_24"
        Target signal column.
    threshold : float, default=0.5
        Topic presence threshold for binary classification.
    figsize : Tuple[int, int], default=(14, 8)
        Figure size in inches.
    dpi : int, default=100
        Figure resolution.
    show : bool, default=True
        Whether to call plt.show() at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The subplot axes.

    Examples
    --------
    >>> fig, ax = plot_topic_impact_on_signals(
    ...     master_df=master,
    ...     topic_cols=['topic_energy', 'topic_weather', 'topic_policy'],
    ... )
    """
    from sklearn.metrics import accuracy_score, f1_score

    if signal_col not in master_df.columns:
        raise KeyError(f"Signal column '{signal_col}' not found in master_df")

    valid_topics = [col for col in topic_cols if col in master_df.columns]
    if not valid_topics:
        raise KeyError("No valid topic columns found in master_df")

    results = []
    for topic in valid_topics:
        topic_data = master_df[[topic, signal_col]].dropna()
        if len(topic_data) == 0:
            continue

        # Binary split: topic present vs absent
        topic_present = topic_data[topic] > threshold
        if topic_present.sum() < 10:  # Need minimum samples
            continue

        y_true = topic_data[signal_col].values
        y_true_present = topic_data[topic_present][signal_col].values

        # Compute metrics when topic is present
        if len(y_true_present) > 0 and len(np.unique(y_true_present)) > 1:
            # Use mode prediction when topic is present
            mode_pred = np.full_like(y_true_present, np.bincount(y_true_present.astype(int)).argmax() - 1)
            acc = accuracy_score(y_true_present, mode_pred)
            f1_macro = f1_score(y_true_present, mode_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_true_present, mode_pred, average="weighted", zero_division=0)

            results.append(
                {
                    "topic": topic,
                    "accuracy": acc,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "samples": len(y_true_present),
                }
            )

    if not results:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "Insufficient topic data for analysis", ha="center", va="center", fontsize=12)
        ax.set_title("Topic Impact on Trading Signals", fontweight="bold")
        if show:
            plt.show()
        return fig, ax

    results_df = pd.DataFrame(results).sort_values("f1_weighted", ascending=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create grouped bar chart
    x = np.arange(len(results_df))
    width = 0.35

    ax.barh(x - width / 2, results_df["accuracy"], width, label="Accuracy", alpha=0.8, color="steelblue")
    ax.barh(x + width / 2, results_df["f1_weighted"], width, label="F1 (weighted)", alpha=0.8, color="coral")

    ax.set_yticks(x)
    ax.set_yticklabels(results_df["topic"])
    ax.set_xlabel("Score")
    ax.set_title("Topic Predictive Power for Trading Signals", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate sample counts
    for i, row in enumerate(results_df.itertuples()):
        ax.text(0.02, i, f"n={row.samples}", va="center", fontsize=8, color="black", fontweight="bold")

    plt.tight_layout()

    print(f"\n{'=' * 70}")
    print("TOPIC IMPACT ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Topics analyzed: {len(results_df)}")
    print(f"\nTop 5 most predictive topics (by F1-weighted):")
    top5 = results_df.tail(5)[["topic", "accuracy", "f1_weighted", "samples"]]
    for _, row in top5.iterrows():
        print(f"  {row['topic']:30s}: F1={row['f1_weighted']:.3f}, Acc={row['accuracy']:.3f}, n={row['samples']}")
    print(f"{'=' * 70}\n")

    if show:
        plt.show()

    return fig, ax


# ============================================================================
# PHASE 4: BACKTESTING OUTCOME VISUALIZATIONS
# ============================================================================


def plot_cumulative_pnl(
    trades_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    *,
    timestamp_col: str = "timestamp",
    pnl_col: str = "pnl",
    signal_col: str = "signal",
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 100,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot cumulative P&L over time with benchmark comparison and risk metrics.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trading results with columns: timestamp, pnl, signal.
    benchmark_df : pd.DataFrame, optional
        Benchmark strategy results (e.g., buy-and-hold).
    timestamp_col : str, default="timestamp"
        Timestamp column name.
    pnl_col : str, default="pnl"
        P&L column name.
    signal_col : str, default="signal"
        Signal column name (Long/Short/Neutral).
    figsize : Tuple[int, int], default=(14, 8)
        Figure size in inches.
    dpi : int, default=100
        Figure resolution.
    show : bool, default=True
        Whether to call plt.show() at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The subplot axes.

    Examples
    --------
    >>> fig, ax = plot_cumulative_pnl(
    ...     trades_df=backtest_results,
    ...     benchmark_df=buyhold_results,
    ... )
    """
    required_cols = [timestamp_col, pnl_col]
    for col in required_cols:
        if col not in trades_df.columns:
            raise KeyError(f"Required column '{col}' not found in trades_df")

    trades = trades_df.copy()
    trades[timestamp_col] = pd.to_datetime(trades[timestamp_col])
    trades = trades.sort_values(timestamp_col)

    # Compute cumulative P&L
    trades["cumulative_pnl"] = trades[pnl_col].cumsum()

    # Compute drawdown
    trades["running_max"] = trades["cumulative_pnl"].cummax()
    trades["drawdown"] = trades["cumulative_pnl"] - trades["running_max"]

    # Compute metrics
    total_return = trades["cumulative_pnl"].iloc[-1] if len(trades) > 0 else 0
    max_drawdown = trades["drawdown"].min()

    # Sharpe ratio (assuming daily returns)
    if len(trades) > 1:
        returns = trades[pnl_col].values
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    else:
        sharpe = 0

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot cumulative P&L
    ax.plot(trades[timestamp_col], trades["cumulative_pnl"], linewidth=2, label="Strategy P&L", color="steelblue")

    # Plot benchmark if available
    if benchmark_df is not None and pnl_col in benchmark_df.columns:
        benchmark = benchmark_df.copy()
        benchmark[timestamp_col] = pd.to_datetime(benchmark[timestamp_col])
        benchmark = benchmark.sort_values(timestamp_col)
        benchmark["cumulative_pnl"] = benchmark[pnl_col].cumsum()
        ax.plot(
            benchmark[timestamp_col],
            benchmark["cumulative_pnl"],
            linewidth=2,
            linestyle="--",
            label="Benchmark",
            color="coral",
            alpha=0.7,
        )

    # Shade drawdown periods
    drawdown_mask = trades["drawdown"] < 0
    if drawdown_mask.any():
        ax.fill_between(
            trades[timestamp_col],
            trades["cumulative_pnl"],
            trades["running_max"],
            where=drawdown_mask,
            alpha=0.3,
            color="red",
            label="Drawdown",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (EUR)")
    ax.set_title("Cumulative P&L Over Time", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Annotate key metrics
    metrics_text = f"Total Return: {total_return:,.2f} EUR\nMax Drawdown: {max_drawdown:,.2f} EUR\nSharpe Ratio: {sharpe:.2f}"
    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    print(f"\n{'=' * 70}")
    print("CUMULATIVE P&L METRICS")
    print(f"{'=' * 70}")
    print(f"Total trades: {len(trades)}")
    print(f"Total return: {total_return:,.2f} EUR")
    print(f"Max drawdown: {max_drawdown:,.2f} EUR")
    print(f"Sharpe ratio: {sharpe:.3f}")
    if signal_col in trades.columns:
        print(f"\nSignal distribution:")
        print(trades[signal_col].value_counts().to_string())
    print(f"{'=' * 70}\n")

    if show:
        plt.show()

    return fig, ax


def plot_signal_performance_breakdown(
    trades_df: pd.DataFrame,
    group_by: str = "signal",
    *,
    pnl_col: str = "pnl",
    timestamp_col: str = "timestamp",
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 100,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Break down trading performance by signal type with multiple views.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trading results with columns: signal, pnl, timestamp.
    group_by : str, default="signal"
        Column to group performance analysis by.
    pnl_col : str, default="pnl"
        P&L column name.
    timestamp_col : str, default="timestamp"
        Timestamp column name.
    figsize : Tuple[int, int], default=(16, 10)
        Figure size in inches.
    dpi : int, default=100
        Figure resolution.
    show : bool, default=True
        Whether to call plt.show() at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    axes : numpy.ndarray
        Array of subplot axes (2x2 grid).

    Examples
    --------
    >>> fig, axes = plot_signal_performance_breakdown(
    ...     trades_df=backtest_results,
    ...     group_by='signal',
    ... )
    """
    required_cols = [group_by, pnl_col, timestamp_col]
    for col in required_cols:
        if col not in trades_df.columns:
            raise KeyError(f"Required column '{col}' not found in trades_df")

    trades = trades_df.copy()
    trades[timestamp_col] = pd.to_datetime(trades[timestamp_col])
    trades["hour"] = trades[timestamp_col].dt.hour
    trades["dayofweek"] = trades[timestamp_col].dt.dayofweek
    trades["win"] = trades[pnl_col] > 0

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle("Signal Performance Breakdown", fontsize=16, fontweight="bold")

    # Panel 1: Win rate by signal type
    win_rate = trades.groupby(group_by)["win"].mean() * 100
    avg_pnl = trades.groupby(group_by)[pnl_col].mean()

    ax1 = axes[0, 0]
    x_pos = np.arange(len(win_rate))
    bars = ax1.bar(x_pos, win_rate.values, color="steelblue", alpha=0.7, edgecolor="black")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(win_rate.index)
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Win Rate by Signal Type", fontweight="bold")
    ax1.axhline(50, color="red", linestyle="--", alpha=0.5, label="50% baseline")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Annotate bars
    for i, (bar, val) in enumerate(zip(bars, win_rate.values)):
        ax1.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Panel 2: Average P&L per trade by signal type
    ax2 = axes[0, 1]
    colors = ["green" if x > 0 else "red" for x in avg_pnl.values]
    bars = ax2.bar(x_pos, avg_pnl.values, color=colors, alpha=0.7, edgecolor="black")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(avg_pnl.index)
    ax2.set_ylabel("Average P&L (EUR)")
    ax2.set_title("Average P&L per Trade", fontweight="bold")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, avg_pnl.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom" if val > 0 else "top",
            fontweight="bold"
        )

    # Panel 3: Trade count by hour of day
    ax3 = axes[1, 0]
    hour_counts = trades.groupby("hour").size()
    ax3.bar(hour_counts.index, hour_counts.values, color="coral", alpha=0.7, edgecolor="black")
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Trade Count")
    ax3.set_title("Trade Distribution by Hour", fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: P&L distribution box plots
    ax4 = axes[1, 1]
    signal_groups = [trades[trades[group_by] == sig][pnl_col].values for sig in win_rate.index]
    bp = ax4.boxplot(signal_groups, labels=win_rate.index, patch_artist=True, showfliers=True)

    for patch, color in zip(bp["boxes"], ["lightblue", "lightgreen", "lightyellow"]):
        patch.set_facecolor(color)

    ax4.set_ylabel("P&L (EUR)")
    ax4.set_title("P&L Distribution by Signal", fontweight="bold")
    ax4.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Print summary
    print(f"\n{'=' * 70}")
    print("SIGNAL PERFORMANCE SUMMARY")
    print(f"{'=' * 70}")
    for signal in win_rate.index:
        signal_trades = trades[trades[group_by] == signal]
        print(f"\n{signal}:")
        print(f"  Trade count: {len(signal_trades)}")
        print(f"  Win rate: {win_rate[signal]:.1f}%")
        print(f"  Avg P&L: {avg_pnl[signal]:.2f} EUR")
        print(f"  Total P&L: {signal_trades[pnl_col].sum():.2f} EUR")
        print(f"  Std Dev: {signal_trades[pnl_col].std():.2f} EUR")
    print(f"{'=' * 70}\n")

    if show:
        plt.show()

    return fig, axes


def plot_risk_metrics_dashboard(
    trades_df: pd.DataFrame,
    risk_free_rate: float = 0.02,
    *,
    pnl_col: str = "pnl",
    timestamp_col: str = "timestamp",
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 100,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Comprehensive risk metrics dashboard for trading strategy evaluation.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trading results with columns: timestamp, pnl.
    risk_free_rate : float, default=0.02
        Annual risk-free rate for Sharpe/Sortino calculations.
    pnl_col : str, default="pnl"
        P&L column name.
    timestamp_col : str, default="timestamp"
        Timestamp column name.
    figsize : Tuple[int, int], default=(16, 12)
        Figure size in inches.
    dpi : int, default=100
        Figure resolution.
    show : bool, default=True
        Whether to call plt.show() at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    axes : numpy.ndarray
        Array of subplot axes (2x2 grid).

    Examples
    --------
    >>> fig, axes = plot_risk_metrics_dashboard(
    ...     trades_df=backtest_results,
    ...     risk_free_rate=0.02,
    ... )
    """
    required_cols = [timestamp_col, pnl_col]
    for col in required_cols:
        if col not in trades_df.columns:
            raise KeyError(f"Required column '{col}' not found in trades_df")

    trades = trades_df.copy()
    trades[timestamp_col] = pd.to_datetime(trades[timestamp_col])
    trades = trades.sort_values(timestamp_col)
    trades["cumulative_pnl"] = trades[pnl_col].cumsum()
    trades["running_max"] = trades["cumulative_pnl"].cummax()
    trades["drawdown"] = trades["cumulative_pnl"] - trades["running_max"]

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle("Risk Metrics Dashboard", fontsize=16, fontweight="bold")

    # Panel 1: Rolling Sharpe ratio
    ax1 = axes[0, 0]
    trades["returns"] = trades[pnl_col]
    daily_rf = risk_free_rate / 252

    # 30-day rolling Sharpe
    rolling_30d = trades["returns"].rolling(window=30)
    sharpe_30d = (rolling_30d.mean() - daily_rf) / rolling_30d.std() * np.sqrt(252)

    # 90-day rolling Sharpe
    rolling_90d = trades["returns"].rolling(window=90)
    sharpe_90d = (rolling_90d.mean() - daily_rf) / rolling_90d.std() * np.sqrt(252)

    ax1.plot(trades[timestamp_col], sharpe_30d, label="30-day", linewidth=1.5, alpha=0.7)
    ax1.plot(trades[timestamp_col], sharpe_90d, label="90-day", linewidth=1.5, alpha=0.7)
    ax1.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax1.axhline(1, color="green", linestyle="--", alpha=0.5, label="Sharpe = 1")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Rolling Sharpe Ratio")
    ax1.set_title("Rolling Sharpe Ratio", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Drawdown duration histogram
    ax2 = axes[0, 1]
    in_drawdown = trades["drawdown"] < 0
    drawdown_periods = []
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                drawdown_periods.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        drawdown_periods.append(current_duration)

    if drawdown_periods:
        ax2.hist(drawdown_periods, bins=20, edgecolor="black", alpha=0.7, color="coral")
        ax2.set_xlabel("Duration (days)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Drawdown Duration Distribution", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "No drawdown periods", ha="center", va="center", fontsize=12)
        ax2.set_title("Drawdown Duration Distribution", fontweight="bold")

    # Panel 3: VaR and CVaR
    ax3 = axes[1, 0]
    returns = trades["returns"].dropna()

    if len(returns) > 0:
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        metrics = ["VaR 95%", "VaR 99%", "CVaR 95%", "CVaR 99%"]
        values = [var_95, var_99, cvar_95, cvar_99]
        colors = ["orange", "red", "darkorange", "darkred"]

        bars = ax3.barh(metrics, values, color=colors, alpha=0.7, edgecolor="black")
        ax3.set_xlabel("Value (EUR)")
        ax3.set_title("Value at Risk (VaR) and Conditional VaR", fontweight="bold")
        ax3.axvline(0, color="black", linestyle="-", linewidth=0.8)
        ax3.grid(True, alpha=0.3, axis="x")

        for bar, val in zip(bars, values):
            ax3.text(val, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", ha="right" if val < 0 else "left")
    else:
        ax3.text(0.5, 0.5, "Insufficient data for VaR", ha="center", va="center", fontsize=12)
        ax3.set_title("Value at Risk (VaR)", fontweight="bold")

    # Panel 4: Calmar ratio over time
    ax4 = axes[1, 1]
    trades["rolling_return"] = trades["cumulative_pnl"].diff(252).fillna(0)
    trades["rolling_max_dd"] = trades["drawdown"].rolling(window=252).min()
    trades["calmar"] = trades["rolling_return"] / trades["rolling_max_dd"].abs()
    trades["calmar"] = trades["calmar"].replace([np.inf, -np.inf], np.nan)

    ax4.plot(trades[timestamp_col], trades["calmar"], linewidth=1.5, color="steelblue")
    ax4.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Calmar Ratio")
    ax4.set_title("Calmar Ratio (Return / Max Drawdown)", fontweight="bold")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print summary table
    print(f"\n{'=' * 70}")
    print("RISK METRICS SUMMARY")
    print(f"{'=' * 70}")

    total_return = trades["cumulative_pnl"].iloc[-1] if len(trades) > 0 else 0
    max_dd = trades["drawdown"].min()
    returns_array = trades["returns"].dropna().values

    # Sharpe ratio
    if len(returns_array) > 1 and returns_array.std() > 0:
        sharpe = ((returns_array.mean() - daily_rf) / returns_array.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino ratio (downside deviation)
    downside_returns = returns_array[returns_array < 0]
    if len(downside_returns) > 1 and downside_returns.std() > 0:
        sortino = ((returns_array.mean() - daily_rf) / downside_returns.std()) * np.sqrt(252)
    else:
        sortino = 0

    # Calmar ratio
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    profitable_trades = (returns_array > 0).sum()
    total_trades = len(returns_array)
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

    print(f"Annual return:          {total_return:>10.2f} EUR")
    print(f"Sharpe ratio:           {sharpe:>10.3f}")
    print(f"Sortino ratio:          {sortino:>10.3f}")
    print(f"Max drawdown:           {max_dd:>10.2f} EUR")
    print(f"Calmar ratio:           {calmar:>10.3f}")
    print(f"Profitable trades:      {profitable_trades:>10d} / {total_trades} ({win_rate:.1f}%)")

    if len(returns) > 0:
        print(f"VaR (95%):              {var_95:>10.2f} EUR")
        print(f"VaR (99%):              {var_99:>10.2f} EUR")
        print(f"CVaR (95%):             {cvar_95:>10.2f} EUR")
        print(f"CVaR (99%):             {cvar_99:>10.2f} EUR")

    if drawdown_periods:
        print(f"Avg drawdown duration:  {np.mean(drawdown_periods):>10.1f} days")
        print(f"Max drawdown duration:  {max(drawdown_periods):>10d} days")

    print(f"{'=' * 70}\n")

    if show:
        plt.show()

    return fig, axes


def plot_confusion_matrix_evolution(
    y_true_by_period: Sequence[np.ndarray],
    y_pred_by_period: Sequence[np.ndarray],
    period_labels: Sequence[str],
    *,
    class_labels: Sequence[str] = ("Short", "Neutral", "Long"),
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 100,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Visualize how confusion matrices evolve across time periods.

    Parameters
    ----------
    y_true_by_period : Sequence[np.ndarray]
        List of ground truth labels for each period.
    y_pred_by_period : Sequence[np.ndarray]
        List of predictions for each period.
    period_labels : Sequence[str]
        Labels for each period (e.g., "Q1 2020", "Q2 2020").
    class_labels : Sequence[str], default=("Short", "Neutral", "Long")
        Class labels for confusion matrix display.
    figsize : Tuple[int, int], optional
        Figure size in inches. Auto-computed if None.
    dpi : int, default=100
        Figure resolution.
    show : bool, default=True
        Whether to call plt.show() at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    axes : numpy.ndarray
        Array of subplot axes.

    Examples
    --------
    >>> fig, axes = plot_confusion_matrix_evolution(
    ...     y_true_by_period=[y_true_q1, y_true_q2, y_true_q3],
    ...     y_pred_by_period=[y_pred_q1, y_pred_q2, y_pred_q3],
    ...     period_labels=['Q1', 'Q2', 'Q3'],
    ... )
    """
    n_periods = len(y_true_by_period)
    if n_periods != len(y_pred_by_period) or n_periods != len(period_labels):
        raise ValueError("Lengths of y_true_by_period, y_pred_by_period, and period_labels must match")

    if n_periods == 0:
        raise ValueError("No periods provided")

    # Auto-compute figure size
    if figsize is None:
        cols = min(4, n_periods)
        rows = (n_periods + cols - 1) // cols
        figsize = (4 * cols, 4 * rows)

    cols = min(4, n_periods)
    rows = (n_periods + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes.ravel()

    fig.suptitle("Confusion Matrix Evolution Over Time", fontsize=16, fontweight="bold")

    from sklearn.metrics import confusion_matrix, precision_score

    precision_by_class = {cls: [] for cls in class_labels}

    for idx, (y_true, y_pred, label) in enumerate(zip(y_true_by_period, y_pred_by_period, period_labels)):
        ax = axes[idx]

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Track precision per class
        try:
            prec = precision_score(y_true, y_pred, average=None, labels=list(range(len(class_labels))), zero_division=0)
            for i, cls in enumerate(class_labels):
                if i < len(prec):
                    precision_by_class[cls].append(prec[i])
        except Exception:
            pass

    # Hide unused subplots
    for idx in range(n_periods, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    # Print degradation analysis
    print(f"\n{'=' * 70}")
    print("CONFUSION MATRIX EVOLUTION ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Periods analyzed: {n_periods}")

    for cls in class_labels:
        precisions = precision_by_class[cls]
        if len(precisions) > 1:
            first_prec = precisions[0]
            last_prec = precisions[-1]
            change = last_prec - first_prec
            print(f"\n{cls}:")
            print(f"  Initial precision: {first_prec:.3f}")
            print(f"  Final precision:   {last_prec:.3f}")
            print(f"  Change:            {change:+.3f}")
            if abs(change) > 0.1:
                trend = "degradation" if change < 0 else "improvement"
                print(f"  ⚠ Significant {trend} detected!")

    print(f"{'=' * 70}\n")

    if show:
        plt.show()

    return fig, axes


def plot_feature_stability(
    feature_importances_by_fold: Sequence[np.ndarray],
    feature_names: Sequence[str],
    top_n: int = 20,
    *,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 100,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Analyze feature importance stability across cross-validation folds.

    Parameters
    ----------
    feature_importances_by_fold : Sequence[np.ndarray]
        List of feature importance arrays, one per CV fold.
    feature_names : Sequence[str]
        Feature names aligned with importance arrays.
    top_n : int, default=20
        Number of top features to display.
    figsize : Tuple[int, int], default=(14, 10)
        Figure size in inches.
    dpi : int, default=100
        Figure resolution.
    show : bool, default=True
        Whether to call plt.show() at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The subplot axes.

    Examples
    --------
    >>> fig, ax = plot_feature_stability(
    ...     feature_importances_by_fold=[fold1_imp, fold2_imp, fold3_imp],
    ...     feature_names=feature_cols,
    ... )
    """
    if len(feature_importances_by_fold) == 0:
        raise ValueError("No feature importances provided")

    n_folds = len(feature_importances_by_fold)
    n_features = len(feature_names)

    # Stack importances into a matrix
    importance_matrix = np.array(feature_importances_by_fold)  # shape: (n_folds, n_features)

    if importance_matrix.shape[1] != n_features:
        raise ValueError(f"Feature names ({n_features}) must match importance dimensions ({importance_matrix.shape[1]})")

    # Compute statistics
    mean_importance = importance_matrix.mean(axis=0)
    std_importance = importance_matrix.std(axis=0)
    cv_importance = std_importance / (mean_importance + 1e-10)  # Coefficient of variation

    # Create dataframe
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean": mean_importance,
            "std": std_importance,
            "cv": cv_importance,
        }
    )
    importance_df = importance_df.sort_values("mean", ascending=False).head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    y_pos = np.arange(len(importance_df))
    colors = ["red" if cv > 0.5 else "orange" if cv > 0.3 else "green" for cv in importance_df["cv"]]

    ax.barh(y_pos, importance_df["mean"], xerr=importance_df["std"], color=colors, alpha=0.7, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df["feature"])
    ax.set_xlabel("Feature Importance (mean ± std)")
    ax.set_title(f"Feature Stability Across {n_folds} CV Folds (Top {len(importance_df)})", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", alpha=0.7, label="Stable (CV < 0.3)"),
        Patch(facecolor="orange", alpha=0.7, label="Moderate (CV 0.3-0.5)"),
        Patch(facecolor="red", alpha=0.7, label="Unstable (CV > 0.5)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    # Print summary
    print(f"\n{'=' * 70}")
    print("FEATURE STABILITY ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Cross-validation folds: {n_folds}")
    print(f"Total features: {n_features}")
    print(f"Top features displayed: {len(importance_df)}")

    unstable = importance_df[importance_df["cv"] > 0.5]
    if len(unstable) > 0:
        print(f"\n⚠ Unstable features (CV > 0.5):")
        for _, row in unstable.iterrows():
            print(f"  {row['feature']:30s}: mean={row['mean']:.4f}, std={row['std']:.4f}, CV={row['cv']:.2f}")
    else:
        print("\n✓ All top features are stable (CV ≤ 0.5)")

    # Rank stability
    ranks_by_fold = []
    for fold_importances in feature_importances_by_fold:
        ranks = np.argsort(-fold_importances)
        ranks_by_fold.append(ranks)

    rank_changes = []
    for i, feat_name in enumerate(importance_df["feature"]):
        feat_idx = list(feature_names).index(feat_name)
        ranks = [np.where(fold_ranks == feat_idx)[0][0] for fold_ranks in ranks_by_fold]
        rank_change = max(ranks) - min(ranks)
        rank_changes.append((feat_name, rank_change))

    rank_changes.sort(key=lambda x: x[1], reverse=True)
    print(f"\nFeatures with largest rank changes:")
    for feat, change in rank_changes[:5]:
        print(f"  {feat:30s}: {change} positions")

    print(f"{'=' * 70}\n")

    if show:
        plt.show()

    return fig, ax


