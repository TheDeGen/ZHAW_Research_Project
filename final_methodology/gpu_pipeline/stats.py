"""Statistical testing utilities extracted from the `final_v3.ipynb` notebook."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import accuracy_score


def bootstrap_confidence_interval(
    y_true,
    y_pred,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Estimate a bootstrap confidence interval for an arbitrary metric."""
    rng = np.random.default_rng(random_state)
    scores = []

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n_samples = len(y_true_arr)

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_true = y_true_arr[indices]
        boot_pred = y_pred_arr[indices]
        try:
            score = metric_func(boot_true, boot_pred)
        except Exception:
            continue
        if not np.isnan(score):
            scores.append(score)

    if not scores:
        return np.nan, np.nan, np.nan

    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    return float(np.mean(scores)), float(lower), float(upper)


def compare_models_statistically(
    y_test,
    signal_pred,
    baseline_pred,
    *,
    signal_proba=None,
    baseline_proba=None,
) -> Dict[str, float]:
    """Run McNemar's test and bootstrap accuracy intervals for two classifiers."""
    print(f"\n{'=' * 70}")
    print("STATISTICAL MODEL COMPARISON")
    print(f"{'=' * 70}\n")

    y_test_array = np.asarray(y_test)
    signal_pred_array = np.asarray(signal_pred)
    baseline_pred_array = np.asarray(baseline_pred)

    if (
        y_test_array.shape != signal_pred_array.shape
        or y_test_array.shape != baseline_pred_array.shape
    ):
        raise ValueError(
            "Predictions and ground-truth labels must have identical shapes for statistical comparison."
        )

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

    print("1. McNemar's Test (Paired Model Comparison)")
    print("-" * 70)

    table = np.zeros((2, 2))
    table[0, 0] = np.sum(
        (signal_pred_array == y_test_array) & (baseline_pred_array == y_test_array)
    )
    table[0, 1] = np.sum(
        (signal_pred_array == y_test_array) & (baseline_pred_array != y_test_array)
    )
    table[1, 0] = np.sum(
        (signal_pred_array != y_test_array) & (baseline_pred_array == y_test_array)
    )
    table[1, 1] = np.sum(
        (signal_pred_array != y_test_array) & (baseline_pred_array != y_test_array)
    )

    print(f"Both models correct:       {table[0, 0]:.0f}")
    print(f"Only signal correct:       {table[0, 1]:.0f}")
    print(f"Only baseline correct:     {table[1, 0]:.0f}")
    print(f"Both models wrong:         {table[1, 1]:.0f}")
    print(f"\nSignal model correct:      {table[0, 0] + table[0, 1]:.0f}")
    print(f"Baseline model correct:    {table[0, 0] + table[1, 0]:.0f}")

    b = table[0, 1]
    c = table[1, 0]

    if b + c > 0:
        mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - sp_stats.chi2.cdf(mcnemar_stat, df=1)

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

    print("\n\n2. Bootstrap 95% Confidence Intervals for Accuracy")
    print("-" * 70)

    signal_acc_mean, signal_acc_lower, signal_acc_upper = bootstrap_confidence_interval(
        y_test_array, signal_pred_array, lambda y, p: accuracy_score(y, p), n_bootstrap=1000
    )

    baseline_acc_mean, baseline_acc_lower, baseline_acc_upper = bootstrap_confidence_interval(
        y_test_array,
        baseline_pred_array,
        lambda y, p: accuracy_score(y, p),
        n_bootstrap=1000,
    )

    print(
        f"Signal model:   {signal_acc_mean:.4f} [{signal_acc_lower:.4f}, {signal_acc_upper:.4f}]"
    )
    print(
        f"Baseline model: {baseline_acc_mean:.4f} [{baseline_acc_lower:.4f}, {baseline_acc_upper:.4f}]"
    )
    print(f"Difference:     {signal_acc_mean - baseline_acc_mean:+.4f}")

    if signal_acc_lower > baseline_acc_upper:
        print("\n✓ Confidence intervals do NOT overlap - strong evidence of difference")
    elif signal_acc_upper < baseline_acc_lower:
        print("\n✗ Baseline appears better (confidence intervals do not overlap)")
    else:
        print("\n○ Confidence intervals overlap - weaker evidence of difference")

    if signal_proba is not None or baseline_proba is not None:
        print("\nNote: Probability inputs are currently unused but reserved for future extensions.")

    print(f"\n{'=' * 70}\n")

    return {
        "mcnemar_statistic": float(mcnemar_stat),
        "mcnemar_p_value": float(p_value),
        "signal_acc_mean": float(signal_acc_mean),
        "signal_acc_ci_lower": float(signal_acc_lower),
        "signal_acc_ci_upper": float(signal_acc_upper),
        "baseline_acc_mean": float(baseline_acc_mean),
        "baseline_acc_ci_lower": float(baseline_acc_lower),
        "baseline_acc_ci_upper": float(baseline_acc_upper),
    }


