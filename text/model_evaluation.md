# 3.6 Model Evaluation Framework

This section details the evaluation methodology employed to assess the performance of the trained models on held-out test data. Building upon the training procedures described in Section 3.5, the evaluation framework quantifies predictive accuracy through classification metrics and establishes statistical significance of performance differences between the signal-enriched and baseline models.

---

## 3.6.1 Out-of-Sample Evaluation Protocol

**Purpose:** Establish the rigorous temporal separation between training/validation and test sets, ensuring unbiased performance estimates.

**Key points to cover:**

- Reiterate the chronological split: 70% train / 20% validation / 10% test (first introduced in Section 3.4.6)
- Emphasise that the test set remains completely untouched during all model development, hyperparameter tuning, and feature parameter selection
- Explain the rationale for temporal splitting in time-series contexts—standard random splits would introduce look-ahead bias by allowing the model to learn from future observations (Tashman, 2000)
- Note that all preprocessing parameters (scalers, UMAP transformations) were fitted exclusively on training data to prevent information leakage
- Describe the evaluation procedure: final models (signal and baseline LightGBM classifiers) are applied to test set features to generate predictions, which are then compared against ground-truth labels

---

## 3.6.2 Classification Performance Metrics

**Purpose:** Introduce and justify the metrics used to evaluate three-class (Long/Neutral/Short) classification performance.

**Key points to cover:**

### Primary Metrics

- **Accuracy**: Proportion of correctly classified observations; interpretable but potentially misleading under class imbalance
  - Formula: $\text{Accuracy} = \frac{\text{TP} + \text{TN}}{N}$
  
- **Macro-Averaged F1 Score**: Harmonic mean of precision and recall, averaged equally across all three classes regardless of prevalence
  - Formula: $F1_{\text{macro}} = \frac{1}{K} \sum_{c=1}^{K} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$
  - Rationale: Ensures minority class performance receives equal weight in evaluation, critical for trading where signals across all market conditions matter (Sokolova & Lapalme, 2009)

- **Multiclass AUC-ROC**: Area under the receiver operating characteristic curve, computed using one-versus-one (OvO) averaging
  - Measures the model's ability to discriminate between classes across all probability thresholds
  - OvO averaging computes AUC for each pair of classes and averages the results, providing a robust multiclass extension (Hand & Till, 2001)

### Supplementary Metrics

- **Precision per class**: Proportion of predicted positives that are true positives
- **Recall per class**: Proportion of actual positives correctly identified
- **Confusion matrix**: Full breakdown of predictions versus ground truth to identify systematic misclassification patterns

### Justification for Classification vs. Regression Metrics

- Note that the PDF outline mentions MAE, MSE, RMSE, R² as potential metrics
- Clarify that these are regression metrics; since the pipeline frames the problem as three-class classification (Long/Neutral/Short based on spread direction), classification metrics are more appropriate
- However, if probabilistic outputs are evaluated against continuous spread values, regression metrics could supplement the analysis (optional extension)

---

## 3.6.3 Statistical Significance Testing

**Purpose:** Establish whether observed performance differences between models are statistically meaningful or could arise from sampling variation.

**Key points to cover:**

### McNemar's Test for Paired Model Comparison

- **Rationale**: McNemar's test (1947) is designed for comparing two classifiers on the same dataset by analysing the contingency table of correct/incorrect predictions
- **Contingency table structure**:
  - Both models correct
  - Only signal model correct
  - Only baseline model correct
  - Both models wrong
- **Test statistic**: $\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$ where $b$ = signal correct & baseline wrong, $c$ = baseline correct & signal wrong
- **Interpretation**: A significant p-value (< 0.05) indicates the models perform differently beyond chance; the direction of improvement is determined by comparing $b$ and $c$
- **Advantages over independent-sample tests**: McNemar's test accounts for the paired nature of predictions on the same instances, providing greater statistical power (Dietterich, 1998)

### Bootstrap Confidence Intervals

- **Method**: Non-parametric bootstrap resampling (Efron & Tibshirani, 1993) to estimate uncertainty around point estimates
- **Procedure**:
  1. Resample test set observations with replacement (n = 1,000 iterations)
  2. Compute metric (accuracy, F1, etc.) on each bootstrap sample
  3. Calculate percentile-based confidence intervals (2.5th and 97.5th percentiles for 95% CI)
- **Interpretation**: Non-overlapping confidence intervals between models provide additional evidence of meaningful performance differences
- **Advantages**: Distribution-free method that makes no assumptions about the underlying metric distribution

### Reporting Standards

- Report point estimates alongside 95% confidence intervals
- Indicate statistical significance levels: * (p < 0.05), ** (p < 0.01), *** (p < 0.001)
- Present both signal and baseline model results to enable direct comparison

---

## Key Implementation Notes (for Results section)

The following elements from `evaluation.py` should inform the Results presentation:

1. **`compare_models_statistically()`**: Generates McNemar's test results and bootstrap CIs for the signal vs. baseline comparison
2. **`bootstrap_confidence_interval()`**: Provides uncertainty quantification for any metric
3. **`_safe_multiclass_auc()`**: Handles edge cases in AUC computation (single-class subsets in bootstrap samples)

---

## Transition to Section 3.7

Note that Section 3.6 focuses on **classification accuracy**—how well the models predict spread direction. Section 3.7 (Backtesting Framework) will translate these predictions into **trading actions** and evaluate **economic performance** using metrics such as:

- Total return (EUR/MWh)
- Sharpe ratio (annualised)
- Sortino ratio
- Maximum drawdown
- Win rate

This distinction separates statistical predictive performance (3.6) from practical trading utility (3.7).

---

## References (to be integrated into main bibliography)

Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923. https://doi.org/10.1162/089976698300017197

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

Hand, D. J., & Till, R. J. (2001). A simple generalisation of the area under the ROC curve for multiple class classification problems. *Machine Learning*, 45(2), 171–186. https://doi.org/10.1023/A:1010920819831

McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157. https://doi.org/10.1007/BF02295996

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting*, 16(4), 437–450. https://doi.org/10.1016/S0169-2070(00)00065-0

