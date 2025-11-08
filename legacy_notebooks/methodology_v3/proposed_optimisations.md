# Proposed Optimisations for Baseline Models

## Executive Summary

Analysis of the V3 methodology notebook reveals that **advanced models (with news features) are performing worse than baseline models**, which indicates fundamental issues that need to be addressed. This document outlines critical mistakes and optimization opportunities to improve baseline model performance.

### Current Performance (Test Set)
- **XGBoost Baseline**: MAE 28.51, RMSE 41.42, R² 0.427 ✅ **Best performing**
- **XGBoost Advanced**: MAE 29.61, RMSE 42.38, R² 0.400 ❌ **Worse than baseline**
- **Linear Regression Baseline**: MAE 30.14, RMSE 43.08, R² 0.380
- **Linear Regression Advanced**: MAE 30.81, RMSE 43.60, R² 0.364 ❌ **Worse than baseline**

---

## Critical Issues Identified

### 1. **Inconsistent Feature Scaling** ⚠️ **HIGH PRIORITY**

**Problem:**
- Baseline features are **not scaled** (used raw values)
- Advanced features (news embeddings and topics) are standardized using `StandardScaler`
- This inconsistency can cause:
  - Gradient-based algorithms (XGBoost) to prioritize unscaled features
  - Linear models to have coefficients with vastly different magnitudes
  - Suboptimal convergence during training

**Evidence:**
```python
# Cell 37-41: Baseline features are NOT scaled
X_train_baseline = current_expanded_train[baseline_features].fillna(0)  # No scaling!

# But advanced features ARE scaled
scaler_news.fit(current_news_features)
X_train_news_scaled = scaler_news.transform(...)
```

**Impact:** Baseline models cannot effectively leverage all features equally, potentially losing 5-10% performance.

**Fix:**
```python
# Scale ALL features (baseline + advanced) using the same scaler
scaler_all = StandardScaler()
X_train_all = np.column_stack([
    current_expanded_train[baseline_features].fillna(0),
    current_expanded_train[news_features].fillna(0)
])
scaler_all.fit(X_train_all)
X_train_baseline_scaled = scaler_all.transform(current_expanded_train[baseline_features].fillna(0))
```

---

### 2. **Missing Current Power Feature** ⚠️ **HIGH PRIORITY**

**Problem:**
- Baseline features include `total_power_lag_24` and `total_power_lag_168` (lagged values)
- **Missing**: Current `total_power` feature, which is highly predictive
- The target is next 24h price, so current power availability is a strong signal

**Current Baseline Features:**
```python
baseline_features = [
    'price_lag_24', 'price_lag_168', 
    'total_power_lag_24', 'total_power_lag_168',  # Only lagged!
    'hour', 'month', 'day_of_week', 'day_of_year', 'week_of_year'
]
```

**Fix:**
```python
baseline_features = [
    'price',  # Current price (if available for the prediction timestamp)
    'total_power',  # Current power generation ← ADD THIS
    'price_lag_24', 'price_lag_168', 
    'total_power_lag_24', 'total_power_lag_168',
    'hour', 'month', 'day_of_week', 'day_of_year', 'week_of_year'
]
```

**Expected Impact:** +2-5% improvement in MAE.

---

### 3. **No Hyperparameter Tuning for Baseline Models** ⚠️ **MEDIUM PRIORITY**

**Problem:**
- XGBoost uses **fixed hyperparameters** (n_estimators=100, max_depth=5, learning_rate=0.1)
- No grid search or optimization for baseline models
- These parameters may not be optimal for the baseline feature set

**Current Configuration:**
```python
models['xgb_baseline'] = XGBRegressor(
    n_estimators=100,  # Fixed
    max_depth=5,       # Fixed
    learning_rate=0.1, # Fixed
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
```

**Fix:** Implement hyperparameter tuning specifically for baseline models:
```python
from sklearn.model_selection import RandomizedSearchCV

baseline_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

baseline_xgb = XGBRegressor(random_state=42, n_jobs=-1)
baseline_search = RandomizedSearchCV(
    baseline_xgb, baseline_param_grid, 
    n_iter=50, cv=3, scoring='neg_mean_absolute_error', 
    cv=3, n_jobs=-1, verbose=1
)
baseline_search.fit(X_train_baseline_scaled, y_train)
```

**Expected Impact:** +3-8% improvement.

---

### 4. **No Cyclical Encoding for Temporal Features** ⚠️ **MEDIUM PRIORITY**

**Problem:**
- Temporal features (`hour`, `month`, `day_of_week`) are used as **linear integers**
- This loses the cyclical nature: hour 23 is closer to hour 0 than hour 12
- Model cannot properly capture periodic patterns

**Current Implementation:**
```python
master_df['hour'] = master_df.index.hour  # 0-23 as integers
master_df['month'] = master_df.index.month  # 1-12 as integers
master_df['day_of_week'] = master_df.index.dayofweek  # 0-6 as integers
```

**Fix:** Use sine/cosine transformations:
```python
# Cyclical encoding for hour (24-hour cycle)
master_df['hour_sin'] = np.sin(2 * np.pi * master_df.index.hour / 24)
master_df['hour_cos'] = np.cos(2 * np.pi * master_df.index.hour / 24)

# Cyclical encoding for day of week (7-day cycle)
master_df['day_of_week_sin'] = np.sin(2 * np.pi * master_df.index.dayofweek / 7)
master_df['day_of_week_cos'] = np.cos(2 * np.pi * master_df.index.dayofweek / 7)

# Cyclical encoding for month (12-month cycle)
master_df['month_sin'] = np.sin(2 * np.pi * master_df.index.month / 12)
master_df['month_cos'] = np.cos(2 * np.pi * master_df.index.month / 12)

# Cyclical encoding for day of year (365-day cycle)
master_df['day_of_year_sin'] = np.sin(2 * np.pi * master_df.index.dayofyear / 365)
master_df['day_of_year_cos'] = np.cos(2 * np.pi * master_df.index.dayofyear / 365)
```

**Expected Impact:** +2-4% improvement, especially for linear models.

---

### 5. **Missing Rolling Statistics Features** ⚠️ **MEDIUM PRIORITY**

**Problem:**
- Only uses point-in-time lagged features
- Missing rolling averages and standard deviations, which capture trends and volatility

**Fix:** Add rolling statistics:
```python
# Rolling mean and std of price over last 24h
master_df['price_rolling_mean_24'] = master_df['price'].rolling(window=24, min_periods=1).mean()
master_df['price_rolling_std_24'] = master_df['price'].rolling(window=24, min_periods=1).std()

# Rolling mean and std of price over last 168h (1 week)
master_df['price_rolling_mean_168'] = master_df['price'].rolling(window=168, min_periods=1).mean()
master_df['price_rolling_std_168'] = master_df['price'].rolling(window=168, min_periods=1).std()

# Rolling mean and std of power over last 24h
master_df['power_rolling_mean_24'] = master_df['total_power'].rolling(window=24, min_periods=1).mean()
master_df['power_rolling_std_24'] = master_df['total_power'].rolling(window=24, min_periods=1).std()
```

**Expected Impact:** +1-3% improvement, especially during volatile periods.

---

### 6. **No Feature Interactions** ⚠️ **LOW-MEDIUM PRIORITY**

**Problem:**
- Features are used independently
- Missing interactions like: `hour × price_lag_24` (price patterns vary by time of day)
- Missing: `total_power × price` (power availability affects price differently at different price levels)

**Fix:** Add polynomial or selected interaction features:
```python
# Important interactions
master_df['hour_price_lag24_interaction'] = master_df['hour'] * master_df['price_lag_24']
master_df['power_price_interaction'] = master_df['total_power'] * master_df['price']
master_df['hour_power_interaction'] = master_df['hour'] * master_df['total_power']
```

**Note:** Use feature importance or correlation analysis to select most valuable interactions.

**Expected Impact:** +1-2% improvement.

---

### 7. **Missing Regularization for Linear Regression Baseline** ⚠️ **LOW PRIORITY**

**Problem:**
- Linear Regression baseline uses no regularization
- Could benefit from Ridge or Lasso to prevent overfitting

**Fix:**
```python
# Use Ridge Regression instead of Linear Regression
from sklearn.linear_model import RidgeCV

models['lr_baseline'] = RidgeCV(
    alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
    cv=5
)
models['lr_baseline'].fit(X_train_baseline_scaled, y_train)
```

**Expected Impact:** +0.5-1% improvement, more stable predictions.

---

### 8. **Data Leakage Risk in Expanding Window** ⚠️ **REVIEW NEEDED**

**Potential Issue:**
- Expanding window training uses data up to validation point
- Need to verify that news features at time `t` don't include information from future (after `t`)

**Check:** Ensure time-decayed aggregation only uses articles published **before** the prediction timestamp.

---

## Why Advanced Models Underperform

### Root Causes:

1. **Feature Quality Issues:**
   - News embeddings may contain noise or irrelevant information
   - 13 topic categories may not be optimally defined
   - 33 additional features (13 topics + 20 embeddings) may cause overfitting

2. **Scaling Inconsistency:**
   - Baseline features unscaled, news features scaled → model cannot properly weight features

3. **Curse of Dimensionality:**
   - 45 total features (9 baseline + 33 news) vs 9 baseline
   - With only ~36K training samples, risk of overfitting increases

4. **Missing Feature Selection:**
   - All 33 news features added without checking relevance
   - Should use feature importance or correlation analysis to select most informative features

**Recommendation:** 
- First fix baseline models using optimizations above
- Then re-evaluate advanced models with proper feature selection and regularization

---

## Recommended Implementation Priority

### Phase 1: Critical Fixes (Immediate)
1. ✅ Add feature scaling for baseline features
2. ✅ Add current `total_power` feature
3. ✅ Implement hyperparameter tuning for baseline XGBoost

### Phase 2: Feature Engineering (Next)
4. ✅ Implement cyclical encoding for temporal features
5. ✅ Add rolling statistics features

### Phase 3: Model Improvements (If time permits)
6. ✅ Add Ridge regularization for Linear Regression baseline
7. ✅ Explore feature interactions
8. ✅ Feature selection for advanced models

---

## Expected Overall Impact

If all optimizations are implemented:
- **XGBoost Baseline MAE improvement: 20-30% reduction** (from 28.51 → ~20-23)
- **Linear Regression Baseline MAE improvement: 15-25% reduction** (from 30.14 → ~23-26)
- **Better baseline performance** will make fair comparison with advanced models possible

---

## Code Example: Optimized Baseline Feature Set

```python
# Optimized baseline features
baseline_features_optimized = [
    # Current features
    'price',
    'total_power',
    
    # Lagged features
    'price_lag_24', 'price_lag_168',
    'total_power_lag_24', 'total_power_lag_168',
    
    # Rolling statistics
    'price_rolling_mean_24', 'price_rolling_std_24',
    'price_rolling_mean_168', 'price_rolling_std_168',
    'power_rolling_mean_24', 'power_rolling_std_24',
    
    # Cyclical temporal features
    'hour_sin', 'hour_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'month_sin', 'month_cos',
    'day_of_year_sin', 'day_of_year_cos',
    
    # Interactions (selected based on importance)
    'hour_price_lag24_interaction',
    'power_price_interaction',
]
```

---

## Conclusion

The baseline models have significant room for improvement through proper feature engineering, scaling, and hyperparameter optimization. The fact that advanced models underperform baseline suggests that news features may need better preprocessing or selection, but first we must ensure baseline models are performing at their full potential to make a fair comparison.

**Next Steps:**
1. Implement Phase 1 optimizations
2. Re-run baseline model training and evaluation
3. Compare optimized baseline vs. advanced models
4. Investigate why advanced features aren't helping (feature quality, selection, regularization)

