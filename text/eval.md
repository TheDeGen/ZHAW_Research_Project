## 3.6 Model Evaluation Framework

The evaluation framework compares the Signal model (incorporating news-derived features) against the Baseline model (utilising price-based features only) through classification metrics, statistical significance testing, and probabilistic assessment. This section details the methodological choices underpinning the evaluation strategy.

### 3.6.1 Classification Metrics and Statistical Testing

Given the three-class prediction task (Long, Neutral, Short), standard classification metrics quantify model performance. Overall accuracy serves as a baseline measure, calculated as the proportion of correctly classified instances. However, accuracy alone provides an incomplete picture when class distributions are imbalanced. To address this limitation, the macro-averaged F1-score is adopted as the primary evaluation metric. The macro F1-score computes the harmonic mean of precision and recall for each class independently, then averages with equal weighting:

$$F1_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

where $K$ denotes the number of classes, and $P_k$ and $R_k$ represent precision and recall for class $k$, respectively. This formulation ensures that minority classes contribute equally to the overall score (Sokolova & Lapalme, 2009). Per-class precision, recall, and F1-scores are additionally reported to enable granular assessment, whilst confusion matrices visualise systematic misclassification patterns.

For probabilistic evaluation, the Area Under the ROC Curve (AUC) assesses discriminative ability independent of the chosen decision threshold. The One-versus-One (OvO) decomposition computes pairwise AUC scores for all $\frac{K(K-1)}{2}$ class pairs, aggregated via macro-averaging. This strategy treats all class pairs with equal importance irrespective of class prevalence.

**Statistical Significance Testing.** Demonstrating that observed performance differences are statistically meaningful requires formal hypothesis testing. McNemar's test is employed for paired model comparison, as it is specifically designed for comparing two classifiers evaluated on identical test instances. The test constructs a 2×2 contingency table based on prediction disagreements, where $b$ represents instances where only the Signal model predicts correctly and $c$ represents instances where only the Baseline model predicts correctly. The test statistic with continuity correction is:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

This statistic follows a chi-square distribution with one degree of freedom under the null hypothesis of equal error rates. By focusing exclusively on discordant pairs, McNemar's test achieves greater statistical power than tests based on aggregate accuracy, whilst the paired design controls for instance-level difficulty.

### 3.6.2 Temporal Validation and Class Imbalance Handling

The evaluation reuses the expanding-window splitter described in Section 3.5 (initial 336-hour window with forward-sliding validation) to preserve chronological order without restating the full training procedure. Because lookback-based features span the boundary between folds, no explicit gap is inserted; the held-out test set remains the primary guardrail against leakage.

**Class Imbalance Handling.** Metrics are computed with the squared inverse-frequency sample weights already defined in Section 3.5 so that evaluation mirrors the training loss. Neutral-class probability thresholds are tuned on validation folds (no retraining) to rebalance precision and recall, with methodological details retained in Section 3.5 to avoid duplication. Feature importance is reported via LightGBM gain rankings, with SHAP values as a complementary local explanation tool.
