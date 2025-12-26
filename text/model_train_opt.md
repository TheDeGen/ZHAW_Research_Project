# 3.5 Model Training and Optimisation

This section details the hierarchical training methodology employed to transform engineered features into actionable trading signals. The framework implements a two-stage stacked ensemble architecture wherein an XGBoost classifier first learns complex interactions between news-derived features and market dynamics, followed by a LightGBM meta-learner that refines these predictions into long/short/hold trading decisions. This approach draws upon established stacking generalisation principles wherein base-level predictions serve as meta-features for subsequent learners, improving overall predictive accuracy (Wolpert, 1992, pp. 241–259).

## 3.5.1 Training Framework Architecture

The training pipeline follows a systematic progression through five computational stages, designed to maintain strict temporal ordering whilst maximising parallel efficiency where dependencies permit. Building upon the preprocessed features described in Section 3.4, the workflow proceeds as follows.

The two-stage stacked architecture addresses a fundamental challenge in news-driven electricity forecasting: the need to extract predictive signal from high-dimensional, noisy textual features whilst avoiding overfitting on limited training samples. As discussed in Section 2.2.3, limitations of NLP in financial applications include the risk of spurious correlations when combining textual and numerical features. The hierarchical approach mitigates this by progressively reducing dimensionality across stages.

The first stage employs XGBoost as the base learner due to its demonstrated effectiveness in electricity price forecasting contexts where ensemble methods consistently outperform single-model alternatives (Lago, De Ridder, & De Schutter, 2018, p. 395). XGBoost's regularised gradient boosting framework provides implicit feature selection that identifies which news-derived signals carry genuine predictive content, though careful regularisation is required to prevent spurious correlations when combining BERT-based news features with price series (Guan et al., 2022, p. 3573).

The second stage employs LightGBM as the meta-learner, operating on a substantially reduced feature space comprising baseline price features and calibrated XGBoost probability outputs. This dimensionality reduction—from 47 features to 7—mitigates overfitting risk whilst enabling the meta-learner to learn optimal weighting of the base model's predictions across different market conditions. The approach aligns with the stacking generalisation framework formalised by Wolpert (1992), wherein base-level predictions serve as meta-features for subsequent learners, improving overall predictive accuracy.

The workflow proceeds through the following stages:

1. **Feature Parameter Selection**: Evaluate 30 time-decay parameter combinations using Ridge regression with cross-validation (CV), selecting the top five configurations for subsequent model training.

2. **XGBoost Hyperparameter Optimisation**: Conduct randomised search across 80 hyperparameter configurations for each selected parameter combination, employing expanding-window CV to preserve temporal dependencies.

3. **Probability Calibration**: Apply Platt scaling to transform XGBoost output scores into well-calibrated probability estimates suitable for meta-feature generation.

4. **Meta-Learner Training**: Train LightGBM classifiers on both signal-enriched features (baseline + XGBoost predictions) and baseline-only features for comparative analysis.

5. **Model Evaluation**: Assess performance on held-out test data using classification metrics with statistical significance testing.

The chronological data split established in Section 3.4.6 is maintained throughout, ensuring that all model development decisions are made without access to future observations.

## 3.5.2 Feature Parameter Selection via Ridge Regression

Prior to committing substantial computational resources to gradient boosting optimisation, the pipeline implements a lightweight screening phase using Ridge regression classifiers. This approach efficiently identifies promising time-decay parameter combinations from the 30-element grid (6 lookback windows × 5 decay rates) described in Section 3.4.4.

**Ridge Classifier Configuration.** Ridge regression incorporates L2 regularisation to constrain coefficient magnitudes, mitigating multicollinearity in high-dimensional feature spaces (Hoerl & Kennard, 1970, pp. 55–67). The regularisation strength α is selected via grid search across 13 logarithmically-spaced values spanning [10⁻³, 10³]. For each of the 30 parameter combinations, Ridge classifiers are fitted using 5-fold CV with accuracy as the scoring metric.

**Parameter Selection Criteria.** Following cross-validated evaluation, the five parameter combinations achieving highest validation accuracy advance to XGBoost training. This two-stage screening approach serves dual purposes: (1) reducing computational burden by eliminating poorly-performing configurations early, and (2) enabling comparative analysis of how different temporal aggregation strategies influence predictive performance. The efficiency of linear models for initial parameter screening has been demonstrated in similar ensemble pipelines (Caruana, Niculescu-Mizil, Crew, & Ksikes, 2004, pp. 83–90).

## 3.5.3 XGBoost Hyperparameter Optimisation

XGBoost implements regularised gradient boosting with built-in L1 and L2 penalties that control model complexity through tree structure and leaf weight constraints (Chen & Guestrin, 2016, pp. 785–794).

**Randomised Search Configuration.** Rather than exhaustive grid search, the pipeline employs randomised search sampling from continuous distributions. Following Bergstra and Bengio (2012, pp. 288–290), random search with 60–100 iterations for a 9-dimensional hyperparameter space achieves comparable results to exhaustive grid search whilst reducing computational burden by orders of magnitude. Specifically, randomised search more efficiently explores high-dimensional parameter spaces by avoiding the combinatorial explosion inherent in grid-based approaches.

The search explores 80 random configurations per dataset, yielding 640 total model fits when combined with 8-fold CV. The hyperparameter distributions are configured to balance model complexity against regularisation strength. Table 1 details the search distributions:

**Table 1: XGBoost Hyperparameter Search Distributions**

| Parameter | Distribution | Range | Purpose |
|-----------|--------------|-------|---------|
| n_estimators | Uniform integer | [100, 400] | Number of boosting rounds |
| max_depth | Uniform integer | [2, 6] | Maximum tree depth |
| learning_rate | Log-uniform | [0.01, 0.15] | Step size shrinkage |
| subsample | Uniform | [0.5, 0.9] | Row sampling ratio |
| colsample_bytree | Uniform | [0.5, 0.9] | Column sampling ratio |
| gamma | Uniform | [0.5, 5.0] | Minimum loss reduction for split |
| min_child_weight | Uniform integer | [3, 12] | Minimum instance weight in child |
| reg_alpha | Log-uniform | [10⁻³, 10¹] | L1 regularisation term |
| reg_lambda | Log-uniform | [10⁻², 10²] | L2 regularisation term |

The distributions are deliberately constrained compared to typical defaults: maximum depth is limited to 6 (rather than the common ceiling of 10–12), learning rate upper bound is reduced to 0.15 (from 0.3), and regularisation lower bounds are elevated. These modifications address the validation-to-test degradation pattern characteristic of overfitting in time-series contexts with limited training samples.

**Expanding Window Cross-Validation.** To preserve the temporal ordering established in Section 3.4.6, the pipeline implements an expanding-window (walk-forward) validation scheme wherein the training set grows monotonically whilst the test window advances forward in time (Tashman, 2000, pp. 437–450). The custom `ExpandingWindowSplitter` uses 8 folds with a 72-hour step size and 336-hour minimum training size, balancing variance estimation against computational cost (Cerqueira, Torgo, & Mozetič, 2020, p. 2005).

**Class Imbalance Handling.** The three-class target distribution exhibits moderate imbalance, with the Neutral class comprising approximately 27% of observations compared to 37% for Short and 36% for Long. The pipeline addresses this through inverse-frequency sample weighting, assigning observation weights inversely proportional to class prevalence, ensuring that minority class observations contribute proportionally more to the loss function gradient (He & Garcia, 2009, pp. 1263–1284).

**Early Stopping Configuration.** To prevent overfitting during boosting iterations, early stopping monitors validation set performance and terminates training when the evaluation metric ceases improving (Prechelt, 1998, pp. 55–69). The configuration employs:

- **Evaluation metric**: Multi-class log-loss (mlogloss)
- **Early stopping rounds**: 20 iterations without improvement
- **Evaluation set**: Held-out validation partition (20% of data)

Early stopping operates independently of CV, providing an additional regularisation mechanism that adapts the effective number of boosting rounds to each hyperparameter configuration. This dynamic stopping criterion reduces computational waste on overparameterised configurations whilst allowing well-regularised models to train longer.

**Objective Function and Evaluation Metric.** For the three-class classification task, the XGBoost objective is configured as `multi:softprob`, which optimises multi-class log-loss whilst outputting probability estimates for each class. The macro-averaged F1 score serves as the primary optimisation target during hyperparameter search, balancing precision and recall across all three classes equally regardless of class prevalence:

$$F1_{\text{macro}} = \frac{1}{K} \sum_{c} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

This choice reflects the trading application's requirement for reliable signals across all market conditions. Macro-averaging ensures that minority class performance receives equal weight in model selection, preventing the optimiser from sacrificing Neutral class accuracy to improve performance on the more prevalent Long and Short classes (Sokolova & Lapalme, 2009, pp. 427–437).

## 3.5.4 Probability Calibration via Platt Scaling

Whilst XGBoost with the `softprob` objective produces normalised probability vectors, these outputs may not be well-calibrated in the sense that a predicted probability of 0.7 corresponds to 70% empirical accuracy (Niculescu-Mizil & Caruana, 2005, pp. 625–632). Poorly calibrated probabilities can degrade downstream model performance when used as meta-features, as the LightGBM meta-learner may learn to compensate for systematic biases rather than exploiting genuine predictive signal.

**Platt Scaling Method.** Platt scaling fits a sigmoid function to transform classifier scores into calibrated probabilities, learning parameters via maximum likelihood on a held-out calibration set (Platt, 1999, pp. 61–74). This approach is well-suited for gradient boosting ensembles, which often exhibit systematic over- or under-confidence (Niculescu-Mizil & Caruana, 2005, pp. 628–630).

**Implementation Details.** The pipeline applies Platt scaling to the best-performing XGBoost model using the `CalibratedClassifierCV` implementation with the following configuration:

- **Method**: Sigmoid (Platt scaling, as opposed to isotonic regression)
- **Cross-validation mode**: `cv='prefit'`, indicating the base model is already trained
- **Calibration set**: Validation partition

Using the prefit mode applies calibration without retraining the base classifier, fitting the sigmoid parameters directly on validation set predictions. This approach avoids the computational cost of refitting XGBoost whilst still correcting for probability miscalibration. The calibrated model replaces the uncalibrated version for all downstream operations, including meta-feature generation for LightGBM training.

## 3.5.5 Meta-Learner Training with LightGBM

LightGBM employs gradient-based sampling and feature bundling optimisations that achieve substantial speedups over traditional gradient boosted decision tree implementations whilst maintaining predictive accuracy (Ke et al., 2017, pp. 3149–3157).

**Stacked Generalisation Architecture.** The training framework implements stacked generalisation (Wolpert, 1992) by using XGBoost predictions as meta-features for LightGBM. In this two-level architecture, the base learner (XGBoost) generates predictions that serve as inputs to the meta-learner (LightGBM), enabling the ensemble to learn complex combinations of base model outputs. Specifically, four features are extracted from calibrated XGBoost output:

1. **xgb_pred**: Predicted class label (-1, 0, or 1)
2. **xgb_prob_class0**: Probability of Neutral class
3. **xgb_prob_class1**: Probability of Long class
4. **xgb_prob_class2**: Probability of Short class

These meta-features capture both the hard prediction and the confidence distribution, enabling the meta-learner to weight XGBoost contributions appropriately. Research on stacking ensembles demonstrates that probability outputs typically provide superior meta-features compared to class labels alone, as they encode uncertainty information lost in hard predictions (Džeroski & Ženko, 2004, pp. 179–206).

**Signal vs. Baseline Model Comparison.** To quantify the contribution of news-derived signals, two LightGBM models are trained in parallel:

**Signal Model**: Combines baseline price features with XGBoost meta-features, representing the full predictive pipeline incorporating news-derived information.

**Baseline Model**: Uses price features only, providing a reference for ablation analysis that isolates the marginal contribution of news signals.

Comparing these models provides direct evidence for whether news-derived signals add predictive value beyond price history. If the signal model significantly outperforms the baseline, this validates the news feature engineering methodology; conversely, if performance is comparable, it suggests limited information content in the news aggregations. This ablation approach follows established practices for evaluating feature contributions in ensemble systems (Breiman, 2001, pp. 5–32).

**LightGBM Grid Search Configuration.** Unlike the XGBoost randomised search, LightGBM employs exhaustive grid search over a more constrained parameter space, reflecting the meta-learner's role in refinement rather than primary feature learning. Exhaustive grid search is feasible for the meta-learner's reduced feature space (7 inputs vs. 47 for XGBoost), where the smaller search volume permits complete enumeration without the sampling variance inherent in random search. Table 2 presents the parameter grid:

**Table 2: LightGBM Grid Search Parameters**

| Parameter | Values | Description |
|-----------|--------|-------------|
| num_leaves | [31, 50, 100] | Maximum leaves per tree |
| max_depth | [5, 10, 15] | Maximum tree depth |
| learning_rate | [0.01, 0.05, 0.1] | Boosting step size |
| n_estimators | [100] | Fixed boosting rounds |
| min_child_samples | [20] | Minimum samples per leaf |
| subsample | [0.8] | Row sampling ratio |
| colsample_bytree | [0.8] | Column sampling ratio |

The grid comprises 27 configurations (3 × 3 × 3), substantially smaller than XGBoost's 80-iteration random search. This reduced search space reflects the meta-learner's simpler optimisation landscape with fewer input features.

**Cross-Validation and Early Stopping.** LightGBM training mirrors the XGBoost configuration with expanding-window CV:

- **Number of splits**: 5 folds, reduced from 8 to reflect the meta-learner's simpler optimisation landscape and the computational efficiency requirements during ablation analysis across multiple model variants
- **Step size**: 72 hours
- **Minimum training size**: 336 hours
- **Early stopping rounds**: 20 iterations without improvement

Class imbalance is addressed identically to XGBoost, using inverse-frequency sample weights computed from training set class prevalence.

---

## References

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305.

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). Ensemble selection from libraries of models. In *Proceedings of the 21st International Conference on Machine Learning* (pp. 83–90). ACM. https://doi.org/10.1145/1015330.1015432

Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. *Machine Learning*, 109, 1997–2028. https://doi.org/10.1007/s10994-020-05910-7

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785

Džeroski, S., & Ženko, B. (2004). Is combining classifiers with stacking better than selecting the best one? *Machine Learning*, 54(3), 179–206. https://doi.org/10.1023/B:MACH.0000015881.36452.6e

Guan, H., Dai, Z., Guan, W., & Ni, A. (2022). Forecasting natural gas prices using highly flexible time-varying parameter models. *Economic Modelling*, 105, 105652. https://doi.org/10.1016/j.econmod.2021.105652

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 3149–3157).

Lago, J., De Ridder, F., & De Schutter, B. (2018). Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms. *Applied Energy*, 221, 386–405. https://doi.org/10.1016/j.apenergy.2018.02.069

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430

Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In A. J. Smola, P. Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), *Advances in Large Margin Classifiers* (pp. 61–74). MIT Press.

Prechelt, L. (1998). Early stopping—But when? In G. B. Orr & K.-R. Müller (Eds.), *Neural Networks: Tricks of the Trade* (pp. 55–69). Springer. https://doi.org/10.1007/3-540-49430-8_3

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting*, 16(4), 437–450. https://doi.org/10.1016/S0169-2070(00)00065-0

Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1


