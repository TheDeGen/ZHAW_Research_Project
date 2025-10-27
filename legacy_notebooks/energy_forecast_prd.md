# **PRD: News-Augmented Hourly Electricity Price Forecasting for Germany**

### **Version: 1.0**
### **Date:** 2025-10-11

---

## 1. Objective

This project aims to develop and evaluate a machine learning model for forecasting the day-ahead hourly electricity price in Germany. The core hypothesis is that incorporating unstructured text data from German news sources can improve the predictive accuracy of a model that relies solely on historical price and power generation data.

## 2. Key Questions & Hypotheses

1.  **Baseline Performance:** Can we establish a reliable baseline forecast using only historical time-series data (prices, total power generation, and time-based features)?
2.  **Value of News Data:** Does the inclusion of German-language news, processed via a BERT-based embedding model, lead to a statistically significant reduction in forecast error (MAE & RMSE) compared to the baseline?
3.  **Feature Importance:** Which features are most influential in predicting price fluctuations? Do the news-derived features rank as important predictors in the final model?

## 3. Data Sources

| Data Type | API / Source | Endpoint / Parameters | Description |
| :--- | :--- | :--- | :--- |
| **Price (Target)** | `api.energy-charts.info` | `price` | Hourly day-ahead spot price for Germany (€/MWh). **Target Variable**. |
| **Power (Input)** | `api.energy-charts.info` | `total_power` | Total hourly power generation for Germany (GW). **Input Feature**. |
| **News (Input)** | `newsapi.org` | `everything` | German news articles. **Sources**: `der-tagesspiegel`, `die-zeit`, `focus`, `handelsblatt`, `spiegel-online`, `wirtschafts-woche`. **Language**: `de`. |

---

## 4. Methodology

This methodology is broken into modular steps, designed for sequential implementation.

### **Module 1: Data Acquisition**

*   [ ] **1.1: Fetch Energy Data:**
    *   Write a function to call the Energy Charts API.
    *   Fetch hourly `day_ahead_price` and `total_power` for Germany (`DE`).
    *   Date Range: A minimum of 3 years is recommended (e.g., 2021-01-01 to yesterday).
    *   Store the result in a pandas DataFrame: `energy_df`.

*   [ ] **1.2: Fetch News Data:**
    *   Write a function to call the NewsAPI.org API.
    *   Iterate through the required date range, fetching all articles from the specified German sources for each day.
    *   Handle API pagination to ensure all articles are retrieved.
    *   Store the result in a pandas DataFrame: `news_df`.

### **Module 2: Data Preprocessing & Alignment**

*   [ ] **2.1: Preprocess Energy Data:**
    *   Convert the index of `energy_df` to a `datetime` object.
    *   Check for and handle any missing values (e.g., using `interpolate()` or `ffill()`).
    *   Ensure data types are correct (floats for price and power).

*   [ ] **2.2: Preprocess News Data:**
    *   For each day in `news_df`, aggregate the `title` and `description` of all articles into a single text document.
    *   Create a new DataFrame `daily_news_df` with columns `['date', 'daily_news_text']`.

*   [ ] **2.3: Merge & Align Datasets:**
    *   Create a `date` column (no time) in the `energy_df`.
    *   Merge `daily_news_df` into `energy_df` on the `date` column.
    *   **Crucial Alignment Step:** Create a feature column for news by shifting the `daily_news_text` forward by 24 hours. This ensures that news from Day `D-1` is used to predict prices for Day `D`.
        ```python
        # Example logic
        df['news_feature_text'] = df.groupby(df.index.date)['daily_news_text'].transform('shift', 24)
        ```
    *   Drop rows with NaN values resulting from the shift (i.e., the first day). This forms the final `master_df`.

### **Module 3: Feature Engineering**

*   [ ] **3.1: Create Baseline Features:**
    *   From the `datetime` index of `master_df`, create the following columns:
        *   `hour`, `day_of_week`, `day_of_year`, `month`, `week_of_year`
    *   Create lag features for both `price` and `total_power`:
        *   `price_lag_24h`, `price_lag_168h`
        *   `power_lag_24h`, `power_lag_168h`
    *   Create rolling window features for both `price` and `total_power`:
        *   `price_rolling_mean_24h`, `price_rolling_std_24h`
        *   `power_rolling_mean_24h`

*   [ ] **3.2: Create Advanced News Features (BERT Embeddings):**
    *   Isolate the unique `news_feature_text` entries.
    *   Load a pre-trained, German-compatible SentenceTransformer model.
        *   **Recommended Model:** `'paraphrase-multilingual-MiniLM-L12-v2'`
    *   Encode the unique news texts into numerical vectors (embeddings).
    *   **(Optional but Recommended):** Use `sklearn.decomposition.PCA` to reduce the dimensionality of the embeddings from 384 to a smaller number (e.g., 30-50). This improves training speed and can reduce noise.
    *   Create a new DataFrame `news_embeddings_df` from these (potentially PCA-reduced) vectors.
    *   Merge `news_embeddings_df` back into `master_df`.

### **Module 4: Weather Feature Engineering (Placeholder)**

*   This module is reserved for future extension.
*   **Potential Data:** Historical hourly weather for a major German city (e.g., Frankfurt) from a source like Open-Meteo.
*   **Potential Features:** `temperature_celsius`, `wind_speed_kmh`, `cloud_cover_percent`, `is_day`.

### **Module 5: Model Training**

*   [ ] **5.1: Define Feature Sets (X) and Target (y):**
    *   `y = master_df['price']`
    *   `X_baseline = master_df[<list of all baseline features>]`
    *   `X_advanced = master_df[<list of all baseline + news embedding features>]`

*   [ ] **5.2: Create Chronological Train-Test Split:**
    *   Split the data based on a timestamp. **Do not use a random split.**
    *   Example: `split_date = '2024-01-01'`
    *   `X_train, X_test = X[X.index < split_date], X[X.index >= split_date]`
    *   Apply this split to both `X_baseline` and `X_advanced` feature sets.

*   [ ] **5.3: Train Baseline Model:**
    *   Initialize a `LinearRegression` model.
    *   Fit the model: `baseline_model.fit(X_train_baseline, y_train)`

*   [ ] **5.4: Train Advanced Model:**
    *   Initialize an `XGBRegressor` with robust starting parameters (e.g., `n_estimators=1000`, `early_stopping_rounds=50`).
    *   Fit the model using the test set for early stopping to prevent overfitting: `advanced_model.fit(X_train_advanced, y_train, eval_set=[(X_test_advanced, y_test)], verbose=False)`

## 5. Evaluation Strategy

*   [ ] **5.1: Generate Predictions:**
    *   Use the trained models to predict on the respective test sets (`X_test_baseline`, `X_test_advanced`).

*   [ ] **5.2: Performance Metrics:**
    *   Calculate the following metrics for both the baseline and advanced models:
        *   **Mean Absolute Error (MAE):** Represents the average forecast error in €.
        *   **Root Mean Squared Error (RMSE):** Punishes larger errors more heavily, important for volatility.

*   [ ] **5.3: Visualization:**
    *   Plot the actual prices vs. the predicted prices from both models for a sample period (e.g., one week) from the test set. This provides a visual confirmation of performance.

*   [ ] **5.4: Interpretation:**
    *   Compare the MAE/RMSE scores in a table.
    *   Plot the feature importance from the `advanced_model` (XGBoost) to identify the top predictors and see if news features contributed significantly.

## 6. Technology Stack

*   **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`
*   **APIs:** `requests`, `newsapi-python`
*   **NLP:** `sentence-transformers`
*   **Visualization:** `matplotlib`, `seaborn`, `plotly`

## 7. Next Steps & Future Work

*   **Integrate Weather Data:** Implement Module 4 by fetching and engineering weather features.
*   **Hyperparameter Tuning:** Use a library like `Optuna` or `GridSearchCV` to find the optimal settings for the XGBoost model.
*   **Alternative NLP Features:** Experiment with topic modeling (e.g., `BERTopic`) on the news text to generate categorical topic features.