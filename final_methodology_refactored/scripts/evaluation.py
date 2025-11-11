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


def actions_to_returns(actions: np.ndarray, spread: pd.Series) -> pd.Series:
    """
    Convert trading actions to returns based on spread.

    Args:
        actions: Array of trading actions (-1, 0, 1)
        spread: Series of price spreads

    Returns:
        Series of returns
    """
    returns = np.where(actions == 1, spread, np.where(actions == -1, -spread, 0.0))
    return pd.Series(returns, index=spread.index)


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
