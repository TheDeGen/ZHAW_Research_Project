
# Proposed Updates for Documentation (Section 3: Methodology)

Based on the recent refactoring of the project's codebase, several key areas of the methodology described in Section 3 of the report need to be updated. The new implementation introduces more sophisticated, robust, and optimized techniques.

Below are the proposed changes, formatted as "Old Text" (inferred from a baseline understanding) versus "New Text" (reflecting the current code).

---

### 1. News Topic Classification

The method for classifying news articles has been significantly enhanced from a simple classification to a hierarchical, two-stage process.

**Old Text:**
> To convert unstructured news text into usable features, we employed a zero-shot classification model. Each article's title was passed to the model, which assigned it to one of several predefined energy-related topics.

**New Text:**
> To convert unstructured news text into usable features, we implemented a **two-stage hierarchical zero-shot classification** pipeline. This approach improves classification accuracy and relevance:
> 
> 1.  **Stage 1 (Routing):** Each article's title is first classified into a broad, high-level category (e.g., "Angebot (Erzeugung & Infrastruktur)", "Nachfrage (Stromverbrauch)"). A confidence threshold of **0.25** is applied; if the score is below this, the article is routed to the "Sonstiges" (Other) category.
> 2.  **Stage 2 (Specific Topic Selection):** The article is then classified again, but only against the specific, fine-grained topics within its assigned high-level category. A confidence threshold of **0.20** is used here. If the score is below this, the article is assigned the default "Sonstiges" label.
> 
> To further refine results, any article still classified as "Sonstiges" after this process undergoes a **re-classification attempt using its description text**, giving it a second chance to be assigned a relevant topic. Articles that are ultimately categorized as "Sonstiges" are excluded from the time-decay feature aggregation steps.

---

### 2. Feature Engineering: Time-Decay Aggregation

The implementation of the time-decay feature calculation has been optimized for performance.

**Old Text:**
> For each hourly timestamp in our dataset, we generated news-based features by looking back over a 336-hour (2-week) window. We applied an exponential time-decay function, `weight = e^(-λ * t)`, to give more importance to recent news. This was done for both topic counts and sentence embeddings.

**New Text:**
> For each hourly timestamp, we generated news-based features by looking back over a configurable window (e.g., 336 hours). We applied an exponential time-decay function, `weight = e^(-λ * t)`, to give more importance to recent news.
> 
> The implementation of this aggregation process is highly optimized:
> - **GPU Acceleration:** The calculations for both topic counts and embeddings are accelerated using **CuPy on NVIDIA GPUs** when available, falling back to a highly vectorized NumPy implementation on CPU.
> - **Parallel Processing:** To find the optimal `lookback_window` and `decay_lambda`, the pipeline precomputes feature sets for all parameter combinations in parallel using `joblib`, significantly reducing processing time.

---

### 3. Model Training: XGBoost Classifier

The training process for the primary XGBoost classifier has been enhanced with more robust techniques for handling class imbalance and preventing overfitting.

**Old Text:**
> We trained an XGBoost classifier to predict the direction of the price spread. To handle the imbalanced nature of the dataset, we used class weights to give more importance to minority classes. The model's hyperparameters were tuned using a random search.

**New Text:**
> We trained a 3-class XGBoost classifier to predict the price spread direction. The training process incorporates several advanced techniques:
> 
> - **Hyperparameter Tuning for Regularization:** The hyperparameter search space (`XGB_PARAM_DISTRIBUTIONS`) was revised to explicitly favor models with stronger regularization to combat overfitting observed in earlier experiments. This includes reducing `max_depth` (from 2-9 to **2-6**), lowering the `learning_rate` (upper bound from 0.3 to **0.15**), and increasing `min_child_weight` (from 1-10 to **3-12**).
> - **Advanced Class Imbalance Handling:** We apply **squared inverse frequency sample weights** during training. This aggressive weighting scheme (`weight = (n_samples / (n_classes * n_samples_in_class))²`) ensures that the model does not ignore severely underrepresented classes.
> - **Time-Series Cross-Validation:** All hyperparameter tuning is performed using a custom **`ExpandingWindowSplitter`** with 8 splits, a minimum training size of 336 hours, and a step size of 72 hours, ensuring the temporal nature of the data is respected.
> - **Early Stopping:** The validation set is used for early stopping during the random search to prevent overfitting and find the optimal number of boosting rounds.

---

### 4. Model Prediction and Calibration

The methodology for generating final predictions from the XGBoost model is now more sophisticated than a simple `argmax` operation.

**Old Text:**
> The final prediction from the model was determined by selecting the class (Long, Short, or Neutral) with the highest predicted probability.

**New Text:**
> The final prediction pipeline includes two critical post-processing steps to improve both the reliability of the model's confidence and the quality of its predictions:
> 
> 1.  **Probability Calibration:** After the best XGBoost model is identified, its probability outputs are calibrated on the validation set using `sklearn.calibration.CalibratedClassifierCV` with **sigmoid scaling (Platt's method)**. This ensures that a predicted probability of, for instance, 80% corresponds more closely to an actual 80% likelihood of being correct.
> 
> 2.  **Neutral Class Thresholding:** Instead of relying on a simple `argmax`, which often fails to predict the neutral class, we implement a **tuned thresholding strategy**. A specific probability threshold for the 'Neutral' class is optimized on the validation set to maximize the macro F1-score. If the calibrated probability for the neutral class exceeds this threshold (e.g., 0.28), the model predicts 'Neutral', otherwise it predicts the higher of the 'Long' or 'Short' classes. This directly addresses the challenge of predicting the neutral state.

---

### 5. Stacking with LightGBM

The documentation should clarify the role of the LightGBM models.

**Old Text:**
> Finally, we trained a LightGBM model to make the final prediction.

**New Text:**
> To test the value of the news-based signals, we employ a stacked modeling approach using two competing LightGBM models:
>
> 1.  **Signal Model:** This model is trained on a feature set that includes baseline market data (e.g., historical prices, load) **plus the calibrated predictions from the XGBoost model**. The XGBoost probabilities serve as high-level features summarizing the news sentiment.
> 2.  **Baseline Model:** This model is trained **only on the baseline market data**, without any input from the news-based XGBoost model.
>
> By comparing the backtest performance of the Signal Model against the Baseline Model, we can rigorously quantify the economic value and predictive lift provided by the entire news processing pipeline. Feature names are sanitized to be JSON-compatible before being passed to LightGBM.
