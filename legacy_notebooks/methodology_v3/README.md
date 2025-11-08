# Methodology V3

## Overview

Methodology V3 represents a unified approach that combines the feature engineering strategies from V1 and V2 into a single comprehensive model for energy price prediction. This methodology leverages both topic-based news features and semantic embedding features to predict future electricity prices.

## Methodology Description

The V3 methodology extracts information from German news articles and combines it with historical energy market data to forecast electricity prices 24 hours ahead. The approach uses two complementary feature sets derived from news articles:

1. **Topic-based features**: News articles are classified into predefined energy-related topics using zero-shot classification. These topic classifications are then aggregated using time-decayed weighted counts that emphasize more recent articles.

2. **Embedding-based features**: News article headlines are converted into semantic embeddings, which are aggregated using the same time-decayed weighting scheme. These embeddings are then reduced in dimensionality using UMAP to create a compact feature representation.

These news-derived features are combined with baseline energy market features (lagged prices, lagged power generation, and temporal features) to train predictive models.

## Workflow

The implementation follows a structured machine learning pipeline:

1. **Data Collection**: Loads German news articles and fetches hourly energy price and power generation data from the Energy Charts API for a one-year period.

2. **Feature Engineering**: 
   - Creates baseline features including lagged prices, lagged power generation, and temporal indicators
   - Classifies news articles into energy-related topics using zero-shot classification
   - Generates semantic embeddings for news article headlines
   - Computes time-decayed aggregations for both topic counts and embeddings

3. **Hyperparameter Optimization**: Performs grid search to identify optimal time-decay parameters (lookback window and decay rate) using Ridge Regression with an expanding window validation approach.

4. **Model Training**: Trains multiple model types (Linear Regression and XGBoost) on the top parameter combinations using an expanding window approach, where models are retrained periodically to adapt to changing patterns.

5. **Model Selection**: Evaluates models on a validation set and selects the best parameter combination based on performance metrics.

6. **Final Evaluation**: Retrains the best model on the full training set and evaluates on a completely held-out test set to obtain unbiased performance estimates.

## Data Split Strategy

The data is split into three distinct sets to ensure proper validation:
- **Training set**: 80% of data used for model development
- **Validation set**: 40% of the training set, used for hyperparameter tuning and model selection
- **Test set**: 20% of data held out until the very end, used only for final performance evaluation

This strict separation ensures that the test set remains completely unseen during all model development, grid search, and selection processes, preventing data leakage and providing reliable performance estimates.

## Key Features

- Time-decayed aggregation ensures recent news has more influence than older news
- Modular design allows easy swapping of feature engineering components
- Expanding window training approach adapts models to temporal changes
- Comprehensive evaluation includes multiple metrics (MAE, RMSE, R²) and visualization

## Limitations

- Currently uses only German news articles without translation capabilities
- Uses auction price data from Energy Charts API rather than spot prices
- Zero-shot classification can be computationally intensive on CPU-only systems (GPU acceleration recommended)
