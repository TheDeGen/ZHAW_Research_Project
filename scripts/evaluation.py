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


def run_portfolio_backtest(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    label_encoder=None,
    initial_capital: float = 100_000.0,
    position_pct: float = 0.10,
    fixed_cost_per_mwh: float = 0.06,  # 0.03 EUR/MWh open + 0.03 close
    pct_cost: float = 0.0,             # No percentage cost (kept for API compatibility)
    spread_column: str | None = "real_spread_abs_shift_24",
) -> dict:
    """
    Run portfolio-based backtest simulation.
    
    Simulates trading with a portfolio starting at initial_capital EUR.
    Each trade allocates position_pct of current portfolio value.
    Returns include transaction costs (fixed per MWh traded).
    
    Args:
        test_df: DataFrame with 'Spot Price' and 'Day Ahead Auction' columns
        predictions: Model predictions (encoded or decoded)
        label_encoder: Optional encoder to decode predictions
        initial_capital: Starting portfolio value in EUR
        position_pct: Fraction of portfolio to allocate per trade (e.g., 0.10 = 10%)
        fixed_cost_per_mwh: Transaction cost per MWh in EUR (default 0.06 = 0.03 open + 0.03 close)
        pct_cost: Percentage transaction cost (deprecated, kept for API compatibility)
        spread_column: Column name for the spread to use for P&L calculation.
            Defaults to 'real_spread_abs_shift_24' (future spread at t+24).
            Falls back to computing from current prices if column not found.
    
    Returns:
        dict with keys:
        - 'equity_curve': pd.Series of portfolio values over time
        - 'returns_pct': pd.Series of period returns (%)
        - 'metrics': dict of summary statistics
        - 'trade_log': pd.DataFrame with trade details (optional)
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
    
    # Extract price series
    spot_series = test_df[spot_col].values
    day_ahead_series = test_df[day_ahead_col].values
    
    # Use shifted spread column if available, otherwise compute from current prices
    if spread_column and spread_column in test_df.columns:
        spread_series = test_df[spread_column].values
    else:
        # Fallback to current spread if column not found
        spread_series = spot_series - day_ahead_series
    
    # Decode predictions if encoder provided
    if label_encoder is not None:
        actions = label_encoder.inverse_transform(predictions)
    else:
        actions = predictions
    
    # Handle NaN values in spread (last 24 periods will have NaN for shifted spread)
    valid_mask = ~np.isnan(spread_series)
    n_valid_periods = valid_mask.sum()
    
    # Calculate action distribution for reference (only valid periods)
    if n_valid_periods > 0:
        valid_actions = actions[valid_mask]
        _action_counts = {int(k): int(v) for k, v in zip(*np.unique(valid_actions, return_counts=True))}
    else:
        _action_counts = {}
    
    # Initialize portfolio tracking
    n_periods = len(test_df)
    equity_curve = np.zeros(n_periods)
    returns_pct = np.zeros(n_periods)
    portfolio_value = initial_capital
    
    # Track trade details
    trade_log = []
    
    # Simulate trading period by period (skip periods with NaN spread)
    for i in range(n_periods):
        equity_curve[i] = portfolio_value
        
        # Skip periods with invalid (NaN) spread values
        if not valid_mask[i]:
            # No trade possible - log with NaN spread
            trade_log.append({
                'period': i,
                'action': 0,
                'spot_price': spot_series[i] if i < len(spot_series) else np.nan,
                'day_ahead_price': day_ahead_series[i] if i < len(day_ahead_series) else np.nan,
                'spread': np.nan,
                'position_value': 0.0,
                'mwh_traded': 0.0,
                'gross_pnl': 0.0,
                'fixed_cost': 0.0,
                'pct_cost': 0.0,
                'total_cost': 0.0,
                'net_pnl': 0.0,
                'portfolio_value': portfolio_value,
            })
            # Calculate period return (%)
            if i > 0:
                returns_pct[i] = ((portfolio_value - equity_curve[i]) / equity_curve[i]) * 100
            else:
                returns_pct[i] = 0.0
            continue
        
        action = actions[i]
        spot_price = spot_series[i]
        day_ahead_price = day_ahead_series[i]
        spread = spread_series[i]
        
        if action != 0:  # Trade signal
            # Calculate position size as percentage of CURRENT portfolio value
            # This ensures proper risk management: positions scale down with losses
            position_value = portfolio_value * position_pct
            
            # Calculate MWh traded
            # Use day_ahead_price as reference for position sizing
            # Guard: Use minimum price floor to prevent position explosion from very low prices
            price_floor = 50.0  # EUR/MWh minimum for position sizing
            sizing_price = max(abs(day_ahead_price), price_floor)
            mwh_traded_raw = position_value / sizing_price if portfolio_value > 0 else 0
            
            # Cap MWh to realistic market liquidity (max 200 MWh per trade)
            max_mwh_per_trade = 200.0
            mwh_traded = min(mwh_traded_raw, max_mwh_per_trade)
            
            # Calculate gross P&L
            # Action +1 (Long): profit when spread > 0 (spot > day_ahead)
            # Action -1 (Short): profit when spread < 0 (spot < day_ahead)
            gross_pnl = mwh_traded * spread * action
            
            # Transaction cost: fixed EUR/MWh (default 0.06 = 0.03 open + 0.03 close)
            total_cost = mwh_traded * fixed_cost_per_mwh
            
            # Net P&L
            net_pnl = gross_pnl - total_cost

            # Update portfolio (floor at zero to prevent impossible negative values)
            portfolio_value = max(portfolio_value + net_pnl, 0)
            
            # Log trade details
            trade_log.append({
                'period': i,
                'action': action,
                'spot_price': spot_price,
                'day_ahead_price': day_ahead_price,
                'spread': spread,
                'position_value': position_value,
                'mwh_traded': mwh_traded,
                'gross_pnl': gross_pnl,
                'cost_per_mwh': fixed_cost_per_mwh,
                'fixed_cost': total_cost,
                'pct_cost': 0.0,
                'total_cost': total_cost,
                'net_pnl': net_pnl,
                'portfolio_value': portfolio_value,
            })
        else:
            # No trade - portfolio unchanged
            trade_log.append({
                'period': i,
                'action': 0,
                'spot_price': spot_price,
                'day_ahead_price': day_ahead_price,
                'spread': spread,
                'position_value': 0.0,
                'mwh_traded': 0.0,
                'gross_pnl': 0.0,
                'fixed_cost': 0.0,
                'pct_cost': 0.0,
                'total_cost': 0.0,
                'net_pnl': 0.0,
                'portfolio_value': portfolio_value,
            })
        
        # Calculate period return (%)
        if i > 0:
            returns_pct[i] = ((portfolio_value - equity_curve[i]) / equity_curve[i]) * 100
        else:
            returns_pct[i] = 0.0
    
    # Convert to Series with test_df index
    equity_curve_series = pd.Series(equity_curve, index=test_df.index)
    returns_pct_series = pd.Series(returns_pct, index=test_df.index)
    
    # Calculate summary metrics
    final_value = equity_curve[-1]
    total_return_pct = ((final_value - initial_capital) / initial_capital) * 100
    
    # Annualized return (assuming hourly data, 24*365 periods per year)
    periods_per_year = 24 * 365
    n_years = n_periods / periods_per_year
    if n_years >= 0.01:  # At least ~3.6 days to avoid numerical issues
        annualized_return = ((final_value / initial_capital) ** (1 / n_years) - 1) * 100
    else:
        # Use simple return for very short periods to avoid RuntimeWarning
        annualized_return = total_return_pct
    
    # Risk metrics
    period_returns = returns_pct_series[returns_pct_series != 0]
    if len(period_returns) > 0:
        volatility = period_returns.std() * np.sqrt(periods_per_year)
        sharpe = (annualized_return / volatility) if volatility > 0 else np.nan
        
        # Sortino ratio (downside deviation)
        downside_returns = period_returns[period_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 1 else np.nan
        sortino = (annualized_return / downside_std) if downside_std > 0 and downside_std > 0 else np.nan
    else:
        volatility = np.nan
        sharpe = np.nan
        sortino = np.nan
    
    # Drawdown
    running_max = equity_curve_series.cummax()
    drawdown = equity_curve_series - running_max
    max_drawdown_pct = (drawdown.min() / running_max.max()) * 100 if running_max.max() > 0 else 0.0
    
    # Win rate
    trade_pnls = [t['net_pnl'] for t in trade_log if t['action'] != 0]
    n_trades = len(trade_pnls)
    n_wins = sum(1 for pnl in trade_pnls if pnl > 0)
    win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0.0
    
    # Total costs
    total_costs = sum(t['total_cost'] for t in trade_log)
    
    metrics = {
        'Final Value (EUR)': final_value,
        'Total Return (%)': total_return_pct,
        'Annualized Return (%)': annualized_return,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': max_drawdown_pct,
        'Win Rate (%)': win_rate,
        'Total Trades': n_trades,
        'Total Costs (EUR)': total_costs,
    }
    
    return {
        'equity_curve': equity_curve_series,
        'returns_pct': returns_pct_series,
        'metrics': metrics,
        'trade_log': pd.DataFrame(trade_log),
    }


def analyze_equity_tail(
    backtest_result: dict,
    n_periods: int = 20,
    initial_capital: float = 100_000.0,
) -> None:
    """
    Analyze the tail end of equity curve to investigate large drops.
    
    Prints summary statistics for the last N periods including:
    - Cumulative P&L in the tail period
    - Number of trades and their outcomes
    - Largest individual losses
    - Spread statistics
    
    Args:
        backtest_result: Dictionary returned from run_portfolio_backtest
        n_periods: Number of periods from the end to analyze (default: 20)
        initial_capital: Initial portfolio value for percentage calculations
    """
    trade_log = backtest_result['trade_log']
    equity_curve = backtest_result['equity_curve']
    
    if len(trade_log) < n_periods:
        n_periods = len(trade_log)
    
    # Get tail period data
    tail_log = trade_log.tail(n_periods).copy()
    tail_equity = equity_curve.tail(n_periods)
    
    # Calculate statistics
    tail_start_value = tail_equity.iloc[0]
    tail_end_value = tail_equity.iloc[-1]
    tail_pnl = tail_end_value - tail_start_value
    tail_pnl_pct = (tail_pnl / tail_start_value) * 100
    
    # Filter to actual trades
    tail_trades = tail_log[tail_log['action'] != 0].copy()
    n_tail_trades = len(tail_trades)
    
    print(f"\n{'='*70}")
    print(f"EQUITY TAIL ANALYSIS (Last {n_periods} periods)")
    print(f"{'='*70}\n")
    
    print(f"Period Range: {tail_log.iloc[0]['period']} to {tail_log.iloc[-1]['period']}")
    print(f"Starting Value: €{tail_start_value:,.2f}")
    print(f"Ending Value: €{tail_end_value:,.2f}")
    print(f"Total P&L: €{tail_pnl:,.2f} ({tail_pnl_pct:+.2f}%)")
    print(f"Number of Trades: {n_tail_trades}\n")
    
    if n_tail_trades > 0:
        # Trade statistics
        winning_trades = tail_trades[tail_trades['net_pnl'] > 0]
        losing_trades = tail_trades[tail_trades['net_pnl'] < 0]
        
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {len(winning_trades) / n_tail_trades * 100:.1f}%\n")
        
        # Largest losses
        if len(losing_trades) > 0:
            largest_losses = losing_trades.nsmallest(5, 'net_pnl')
            print("Top 5 Largest Losses:")
            print("-" * 70)
            for idx, row in largest_losses.iterrows():
                print(f"  Period {int(row['period'])}: "
                      f"Action={int(row['action'])}, "
                      f"Spread={row['spread']:.2f} EUR/MWh, "
                      f"P&L=€{row['net_pnl']:.2f}")
            print()
        
        # Spread statistics
        print(f"Spread Statistics (tail period):")
        print(f"  Mean: {tail_log['spread'].mean():.2f} EUR/MWh")
        print(f"  Std: {tail_log['spread'].std():.2f} EUR/MWh")
        print(f"  Min: {tail_log['spread'].min():.2f} EUR/MWh")
        print(f"  Max: {tail_log['spread'].max():.2f} EUR/MWh")
        print()
        
        # Action distribution
        action_counts = tail_trades['action'].value_counts().sort_index()
        print("Action Distribution:")
        for action, count in action_counts.items():
            action_name = "Long" if action == 1 else "Short" if action == -1 else "Neutral"
            print(f"  {action_name} ({action}): {count} trades")
        print()
        
        # Cumulative P&L by action
        if len(tail_trades) > 0:
            print("Cumulative P&L by Action:")
            for action in [-1, 1]:
                action_trades = tail_trades[tail_trades['action'] == action]
                if len(action_trades) > 0:
                    action_pnl = action_trades['net_pnl'].sum()
                    action_name = "Long" if action == 1 else "Short"
                    print(f"  {action_name}: €{action_pnl:,.2f}")
    else:
        print("No trades in tail period.\n")
    
    print(f"{'='*70}\n")
