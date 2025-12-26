"""
Evaluation Utilities
====================
Statistical testing, confidence intervals, and model evaluation functions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score


def bootstrap_confidence_interval(
    y_true,
    y_pred_proba,
    metric_func,
    n_bootstrap=1000,
    confidence=0.95,
    random_state=42
):
    """
    Calculate bootstrap confidence interval for a metric.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities or predictions
        metric_func: Function to calculate metric (takes y_true, y_pred as input)
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    np.random.seed(random_state)
    scores = []
    n_samples = len(y_true)

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices] if isinstance(y_true, np.ndarray) else y_true.iloc[indices]
        y_pred_boot = y_pred_proba[indices] if isinstance(y_pred_proba, np.ndarray) else y_pred_proba.iloc[indices]

        try:
            score = metric_func(y_true_boot, y_pred_boot)
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue

    if len(scores) == 0:
        return np.nan, np.nan, np.nan

    # Calculate percentile-based confidence interval
    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    return np.mean(scores), lower, upper


def compare_models_statistically(
    y_test,
    signal_pred,
    baseline_pred,
    signal_proba=None,
    baseline_proba=None
):
    """
    Perform statistical comparison between signal and baseline models.

    Args:
        y_test: True test labels
        signal_pred: Signal model predictions (class labels)
        baseline_pred: Baseline model predictions (class labels)
        signal_proba: Signal model probability predictions (optional)
        baseline_proba: Baseline model probability predictions (optional)

    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*70}")
    print("STATISTICAL MODEL COMPARISON")
    print(f"{'='*70}\n")

    # Convert to numpy arrays and validate shapes
    y_test_array = np.asarray(y_test)
    signal_pred_array = np.asarray(signal_pred)
    baseline_pred_array = np.asarray(baseline_pred)

    if y_test_array.shape != signal_pred_array.shape or y_test_array.shape != baseline_pred_array.shape:
        raise ValueError("Predictions and ground-truth labels must have identical shapes for statistical comparison.")

    observed_labels = np.unique(y_test_array)
    unexpected_signal = np.setdiff1d(np.unique(signal_pred_array), observed_labels)
    unexpected_baseline = np.setdiff1d(np.unique(baseline_pred_array), observed_labels)
    if unexpected_signal.size > 0:
        print(f"⚠ Signal model predicted unseen labels: {unexpected_signal}")
    if unexpected_baseline.size > 0:
        print(f"⚠ Baseline model predicted unseen labels: {unexpected_baseline}")

    signal_accuracy = accuracy_score(y_test_array, signal_pred_array)
    baseline_accuracy = accuracy_score(y_test_array, baseline_pred_array)
    print(f"Signal accuracy:   {signal_accuracy:.4f}")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}\n")

    # McNemar's Test for paired model comparison
    print("1. McNemar's Test (Paired Model Comparison)")
    print("-" * 70)

    # Create contingency table
    table = np.zeros((2, 2))
    table[0, 0] = np.sum((signal_pred_array == y_test_array) & (baseline_pred_array == y_test_array))  # Both correct
    table[0, 1] = np.sum((signal_pred_array == y_test_array) & (baseline_pred_array != y_test_array))  # Only signal correct
    table[1, 0] = np.sum((signal_pred_array != y_test_array) & (baseline_pred_array == y_test_array))  # Only baseline correct
    table[1, 1] = np.sum((signal_pred_array != y_test_array) & (baseline_pred_array != y_test_array))  # Both wrong

    print(f"Both models correct:       {table[0, 0]:.0f}")
    print(f"Only signal correct:       {table[0, 1]:.0f}")
    print(f"Only baseline correct:     {table[1, 0]:.0f}")
    print(f"Both models wrong:         {table[1, 1]:.0f}")
    print(f"\nSignal model correct:      {table[0, 0] + table[0, 1]:.0f}")
    print(f"Baseline model correct:    {table[0, 0] + table[1, 0]:.0f}")

    # Perform McNemar's test (using chi-square approximation)
    b = table[0, 1]  # Signal correct, baseline wrong
    c = table[1, 0]  # Baseline correct, signal wrong

    if b + c > 0:
        mcnemar_stat = (abs(b - c) - 1)**2 / (b + c)
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)

        print(f"\nMcNemar's χ² statistic:    {mcnemar_stat:.4f}")
        print(f"p-value:                   {p_value:.4f}")

        if p_value < 0.001:
            print("\n✓ Signal model is HIGHLY SIGNIFICANTLY better (p < 0.001) ***")
        elif p_value < 0.01:
            print("\n✓ Signal model is SIGNIFICANTLY better (p < 0.01) **")
        elif p_value < 0.05:
            print("\n✓ Signal model is SIGNIFICANTLY better (p < 0.05) *")
        else:
            print("\n✗ No significant difference between models (p >= 0.05)")
    else:
        print("\n⚠ Cannot perform McNemar's test: no discordant pairs")
        p_value = np.nan
        mcnemar_stat = np.nan

    # Bootstrap confidence intervals for accuracy
    print(f"\n\n2. Bootstrap 95% Confidence Intervals for Accuracy")
    print("-" * 70)

    signal_acc_mean, signal_acc_lower, signal_acc_upper = bootstrap_confidence_interval(
        y_test_array, signal_pred_array,
        lambda y, p: accuracy_score(y, p),
        n_bootstrap=1000
    )

    baseline_acc_mean, baseline_acc_lower, baseline_acc_upper = bootstrap_confidence_interval(
        y_test_array, baseline_pred_array,
        lambda y, p: accuracy_score(y, p),
        n_bootstrap=1000
    )

    print(f"Signal model:   {signal_acc_mean:.4f} [{signal_acc_lower:.4f}, {signal_acc_upper:.4f}]")
    print(f"Baseline model: {baseline_acc_mean:.4f} [{baseline_acc_lower:.4f}, {baseline_acc_upper:.4f}]")
    print(f"Difference:     {signal_acc_mean - baseline_acc_mean:+.4f}")

    # Check if confidence intervals overlap
    if signal_acc_lower > baseline_acc_upper:
        print("\n✓ Confidence intervals do NOT overlap - strong evidence of difference")
    elif signal_acc_upper < baseline_acc_lower:
        print("\n✗ Baseline appears better (confidence intervals do not overlap)")
    else:
        print("\n○ Confidence intervals overlap - weaker evidence of difference")

    print(f"\n{'='*70}\n")

    return {
        'mcnemar_statistic': mcnemar_stat,
        'mcnemar_p_value': p_value,
        'signal_acc_mean': signal_acc_mean,
        'signal_acc_ci': (signal_acc_lower, signal_acc_upper),
        'baseline_acc_mean': baseline_acc_mean,
        'baseline_acc_ci': (baseline_acc_lower, baseline_acc_upper)
    }


def _safe_multiclass_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """
    Safely compute multiclass AUC.

    Args:
        y_true: True labels
        proba: Predicted probabilities

    Returns:
        AUC score or NaN if computation fails
    """
    unique_classes = np.unique(y_true)
    if unique_classes.size <= 1:
        return np.nan
    if unique_classes.size == 2:
        # Reduce to binary ROC AUC using the higher label as the positive class
        positive_class = unique_classes.max()
        proba_binary = proba[:, positive_class]
        y_binary = (y_true == positive_class).astype(int)
        return roc_auc_score(y_binary, proba_binary)
    proba_aligned = proba[:, unique_classes]
    return roc_auc_score(
        y_true,
        proba_aligned,
        multi_class='ovo',
        average='macro',
        labels=unique_classes
    )


def actions_to_returns(
    actions: np.ndarray,
    spread: pd.Series,
    transaction_cost: float = 0.0,
    transaction_cost_pct: float = 0.0
) -> pd.Series:
    """
    Convert trading actions to returns based on spread, accounting for transaction costs.

    Args:
        actions: Array of trading actions (-1, 0, 1)
        spread: Series of price spreads
        transaction_cost: Fixed transaction cost per trade (EUR/MWh)
        transaction_cost_pct: Percentage transaction cost (as decimal, e.g., 0.001 for 0.1%)

    Returns:
        Series of returns after transaction costs
    """
    # Base returns (before costs)
    returns = np.where(actions == 1, spread, np.where(actions == -1, -spread, 0.0))

    # Apply transaction costs for non-zero actions
    if transaction_cost > 0 or transaction_cost_pct > 0:
        # Fixed cost
        if transaction_cost > 0:
            cost_fixed = np.where(actions != 0, transaction_cost, 0.0)
            returns = returns - cost_fixed

        # Percentage cost (applied to absolute spread value)
        if transaction_cost_pct > 0:
            cost_pct = np.where(actions != 0, np.abs(spread) * transaction_cost_pct, 0.0)
            returns = returns - cost_pct

    return pd.Series(returns, index=spread.index)


def compute_spread_normalizer(spread: pd.Series) -> float:
    """
    Compute mean absolute spread for percentage normalization.

    Args:
        spread: Series of price spreads (spot - day_ahead)

    Returns:
        Mean absolute spread value for normalization

    Raises:
        ValueError: If mean absolute spread is zero
    """
    mean_abs_spread = np.abs(spread).mean()
    if mean_abs_spread == 0:
        raise ValueError("Mean absolute spread is zero; cannot normalize to percentage.")
    return mean_abs_spread


def summarise_returns(
    returns: pd.Series,
    strategy_label: str,
    periods_per_year: int = 24 * 365,
    normalizer: float | None = None,
    return_mode: str = 'percentage'
) -> dict[str, float | str]:
    """
    Summarise key performance statistics for a strategy's return series.

    Args:
        returns: Strategy return series (in absolute units, e.g., EUR/MWh).
        strategy_label: Name of the strategy.
        periods_per_year: Annualisation factor (default assumes hourly data).
        normalizer: Mean absolute spread for percentage conversion.
            Required if return_mode='percentage'.
        return_mode: 'absolute' (EUR/MWh) or 'percentage' (% of mean spread).

    Returns:
        Dictionary containing strategy statistics with appropriate units.

    Raises:
        ValueError: If return_mode='percentage' but normalizer is not provided.
    """
    if return_mode == 'percentage' and normalizer is None:
        raise ValueError("normalizer must be provided when return_mode='percentage'")

    # Scale factor for percentage conversion
    scale = (100.0 / normalizer) if return_mode == 'percentage' and normalizer else 1.0
    unit_suffix = " (%)" if return_mode == 'percentage' else " (EUR/MWh)"

    cumulative = returns.cumsum()
    drawdown = cumulative - cumulative.cummax()
    mean_return = returns.mean()
    volatility = returns.std(ddof=1)
    sharpe = (mean_return / volatility * np.sqrt(periods_per_year)) if volatility > 0 else np.nan

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std(ddof=1) if len(downside_returns) > 1 else np.nan
    sortino = (mean_return / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else np.nan

    # Win rate
    n_trades = (returns != 0).sum()
    n_wins = (returns > 0).sum()
    win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0.0

    return {
        "Strategy": strategy_label,
        f"Total Return{unit_suffix}": (cumulative.iloc[-1] if not cumulative.empty else 0.0) * scale,
        f"Avg Return{unit_suffix}": mean_return * scale,
        f"Volatility{unit_suffix}": volatility * scale,
        "Sharpe (annualised)": sharpe,
        "Sortino (annualised)": sortino,
        f"Max Drawdown{unit_suffix}": (drawdown.min() if not drawdown.empty else 0.0) * scale,
        "Win Rate (%)": win_rate,
        "Number of Trades": int(n_trades),
    }


def compute_strategy_returns(
    action_map: dict[str, np.ndarray],
    spread: pd.Series,
    transaction_cost: float = 0.0,
    transaction_cost_pct: float = 0.0
) -> dict[str, pd.Series]:
    """
    Convert a mapping of strategy actions to return series with transaction costs.

    Args:
        action_map: Mapping from strategy name to action array.
        spread: Price spread series.
        transaction_cost: Fixed transaction cost per trade (EUR/MWh)
        transaction_cost_pct: Percentage transaction cost (as decimal)

    Returns:
        Mapping from strategy name to return series.
    """
    return {
        name: actions_to_returns(actions, spread, transaction_cost, transaction_cost_pct)
        for name, actions in action_map.items()
    }


def summarise_strategy_set(
    returns_map: dict[str, pd.Series],
    periods_per_year: int = 24 * 365,
    normalizer: float | None = None,
    return_mode: str = 'percentage'
) -> pd.DataFrame:
    """
    Summarise multiple strategies into a DataFrame.

    Args:
        returns_map: Mapping of strategy name to return series.
        periods_per_year: Annualisation factor for Sharpe ratio.
        normalizer: Mean absolute spread for percentage conversion.
        return_mode: 'absolute' or 'percentage'.

    Returns:
        DataFrame indexed by strategy with summary metrics.
    """
    summaries = [
        summarise_returns(series, name, periods_per_year, normalizer, return_mode)
        for name, series in returns_map.items()
    ]
    return pd.DataFrame(summaries).set_index("Strategy")


def get_column_name(possible_names: list, df: pd.DataFrame) -> str | None:
    """
    Return the first matching name in possible_names that exists in df.columns.

    Args:
        possible_names: List of possible column names
        df: DataFrame to search

    Returns:
        First matching column name or None if no match
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def setup_backtest_strategies(
    test_df: pd.DataFrame,
    signal_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    label_encoder=None
) -> tuple[pd.Series, dict[str, np.ndarray], float]:
    """
    Set up backtesting strategies and compute spread series.

    Args:
        test_df: Test dataframe with price columns
        signal_predictions: Signal model predictions (encoded or decoded)
        baseline_predictions: Baseline model predictions (encoded or decoded)
        label_encoder: Optional label encoder to decode predictions

    Returns:
        Tuple of (spread_series, strategy_actions_dict, spread_normalizer)
        where spread_normalizer is the mean absolute spread for % normalization.
    """
    # Resolve column names
    spot_col = get_column_name(
        ["Spot Price", "spot_price", "SpotPrice"],
        test_df
    )
    day_ahead_col = get_column_name(
        ["Day Ahead Auction", "day_ahead_price", "DayAhead"],
        test_df
    )

    if spot_col is None or day_ahead_col is None:
        raise KeyError(
            f"Could not identify spot or day-ahead price columns. "
            f"Available columns: {list(test_df.columns)}"
        )

    # Extract price series and compute spread
    spot_series = test_df[spot_col]
    day_ahead_series = test_df[day_ahead_col]
    spread_series = spot_series - day_ahead_series

    # Compute normalizer for percentage returns
    spread_normalizer = compute_spread_normalizer(spread_series)

    # Decode predictions if encoder provided
    if label_encoder is not None:
        signal_decoded = label_encoder.inverse_transform(signal_predictions)
        baseline_decoded = label_encoder.inverse_transform(baseline_predictions)
    else:
        signal_decoded = signal_predictions
        baseline_decoded = baseline_predictions

    # Create strategy actions
    strategy_actions = {
        "LightGBM Signal (with news)": signal_decoded,
        "LightGBM Baseline (price-only)": baseline_decoded,
        "Naive Buy-DA/Sell-Spot": np.ones_like(signal_decoded, dtype=int),
    }

    return spread_series, strategy_actions, spread_normalizer
