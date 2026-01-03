# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Energy price prediction pipeline forecasting German electricity prices 24 hours ahead. Predicts 3-class spread targets (Long/Neutral/Short) based on difference between spot and day-ahead auction prices using:
- NLP-based news signals (zero-shot classification + sentence embeddings)
- Time-decay aggregation of news features
- Two-stage model: XGBoost for news features → LightGBM consumes XGBoost predictions as additional features

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# GPU packages (optional, requires conda)
conda install -c rapidsai -c conda-forge -c nvidia cuml-cu12 cupy

# Run pipeline
jupyter notebook notebooks/pipeline_execution.ipynb

# Clear caches (embeddings cached in .cache/, not outputs/)
rm -rf .cache/ notebooks/.cache
```

## Version Constraints

```
scikit-learn<1.6    # Required for cv='prefit' in CalibratedClassifierCV
umap-learn<0.5.9    # Compatibility with scikit-learn < 1.6
```

## Repository Structure

```
├── config/                     # Configuration (NEVER hard-code values)
│   ├── pipeline_config.py      # Time-decay, CV, NLP, grid search params
│   └── model_config.py         # Model hyperparameters
├── scripts/                    # Modular pipeline components
│   ├── device_utils.py         # GPU/CPU detection
│   ├── data_ingestion.py       # Load news and energy data
│   ├── feature_engineering.py  # NLP embeddings, time-decay features
│   ├── model_utils.py          # XGBoost/LightGBM training, calibration
│   ├── evaluation.py           # Metrics, bootstrap CIs, McNemar's test
│   ├── visualization.py        # Plotting functions
│   ├── save_models.py          # Model persistence
│   └── profiling.py            # Performance monitoring
├── notebooks/
│   └── pipeline_execution.ipynb  # Main execution (~25 cells)
├── data/                       # german_news_v1.csv, energy_baseline.csv
├── outputs/                    # Model artifacts
├── .cache/                     # Embeddings, reduced embeddings (auto-created)
└── data_prep/                  # Legacy data fetchers (reference only)
```

## Key Architecture Decisions

### Configuration-Driven Design
All parameters in `config/pipeline_config.py`:
```python
from config import pipeline_config
lookback = pipeline_config.DEFAULT_LOOKBACK_WINDOW  # 336 hours
decay = pipeline_config.DEFAULT_DECAY_LAMBDA        # 0.05
deadband = pipeline_config.SPREAD_TARGET_DEADBAND   # EUR/MWh threshold
```

### Device Detection
Auto-detects CUDA/MPS/CPU and returns optimal settings:
```python
from scripts import device_utils
device_config = device_utils.detect_compute_device(task='embeddings')
# Returns: device, optimal_batch_size, tree_method, xgb_device, n_jobs
```

### Hierarchical Zero-Shot Classification
1. Stage 1: Route to categories (Nachfrage, Angebot, Brennstoffpreise, etc.)
2. Stage 2: Select leaf topic from category's labels
3. Fallback: Re-classify low-confidence "other" articles using descriptions

Configured via `HIERARCHICAL_TOPIC_GROUPS` and `HIERARCHICAL_ROUTING_SETTINGS` in pipeline_config.py.

### Two-Stage Model Architecture
1. **XGBoost**: Trained on baseline + time-decay news features; calibrated using Platt scaling
2. **LightGBM Signal Model**: Consumes baseline features + XGBoost prediction probabilities
3. **LightGBM Baseline Model**: Price/temporal features only (for comparison)

```python
# XGBoost predictions become features for LightGBM
from scripts.model_utils import enrich_with_model_predictions
enriched = enrich_with_model_predictions(xgb_model, dataframes, feature_columns, "xgb")
# Adds: xgb_pred, xgb_prob_class0, xgb_prob_class1, xgb_prob_class2
```

### Time-Decay Features
Formula: `weight = exp(-λ * hours_since)`
- Separate features for topic counts and embedding averages
- Dimensionality reduction: UMAP (GPU-first) or PCA to 20 components
- "Other" articles filtered out before aggregation (no energy relevance)

### Expanding Window CV
```python
from scripts.model_utils import ExpandingWindowSplitter
cv = ExpandingWindowSplitter(n_splits=8, step_size=72, min_train_size=336)
```
Train size grows, test window slides forward. Preserves temporal ordering.

### Class Imbalance Handling
- **Sample weights**: Squared inverse-frequency (`(N / (n_classes * class_count))^2`)
- **Neutral threshold tuning**: Since neutral class is underrepresented, use `tune_neutral_threshold()` to find optimal probability threshold

```python
from scripts.model_utils import tune_neutral_threshold, predict_with_neutral_threshold
result = tune_neutral_threshold(model, X_val, y_val, neutral_class_idx=1)
preds = predict_with_neutral_threshold(model, X_test, threshold=result['best_threshold'])
```

### LightGBM Feature Sanitization
German topic names contain umlauts that break LightGBM's JSON serialization:
```python
from scripts.model_utils import sanitize_feature_names
sanitized, rename_map = sanitize_feature_names(feature_columns)
# "Nachfrage (Stromverbrauch)" → "Nachfrage__Stromverbrauch_"
```

### Caching System
Disk-cached with checksum validation in `.cache/embeddings/`:
- `news_embeddings.parquet` + `manifest.json`
- `td_embeddings_lw{lookback}_dl{lambda}_reduction.parquet`

## Critical Constraints

1. **NEVER shuffle time-series data** - introduces temporal leakage
2. **NEVER hard-code parameters** - use config files
3. **DO NOT modify** `data_prep/` unless explicitly requested (legacy code)
4. **Known limitation**: No temporal gaps between train/val/test splits causes minor leakage through lookback windows (see README for details)

## Available Specialized Agents

### model-validation-checklist
Use before training, after unexpected results, or when reviewing configs. Validates hyperparameters, data splits, and flags temporal leakage issues.

### viz-insights-designer
Use for creating ML diagnostic visualizations: confusion matrices, feature importance, learning curves, time-series plots.

### gpu-parallelism-reviewer
Use after writing GPU code or when experiencing performance issues. Reviews device detection, memory transfers, and parallel workload distribution.

### precise-code-answer
Use for quick, factual answers about specific code behavior without elaboration. Example: "What's the default lookback window?"

## Key Functions Reference

```python
# Full pipeline training
from scripts.model_utils import train_xgb_candidates, train_lightgbm_pair

# XGBoost training with grid search
xgb_results = train_xgb_candidates(
    top_combinations, preprocessed_datasets, baseline_features,
    target_column, param_distributions, ...
)

# LightGBM pair (signal + baseline)
lgbm_results = train_lightgbm_pair(
    signal_train_df, signal_val_df, signal_test_df,
    signal_feature_columns, baseline_feature_columns, ...
)
```

## Adding New Features

```python
# In scripts/feature_engineering.py
def compute_custom_feature(master_df):
    master_df['my_feature'] = ...
    return master_df

# In notebook
from scripts.feature_engineering import compute_custom_feature
feature_df = compute_custom_feature(feature_df)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU not detected | Check `torch.cuda.is_available()`, install correct PyTorch for CUDA version |
| Out of memory | Reduce batch size, use smaller lookback window, sample fewer articles |
| Import errors | Add parent dir to path: `sys.path.insert(0, str(Path('..').resolve()))` |
| Cache issues | Delete `.cache/` directory |
| Low zero-shot confidence | Adjust thresholds in `HIERARCHICAL_ROUTING_SETTINGS` |
| LightGBM JSON error | Use `sanitize_feature_names()` for German topic names |
| No neutral predictions | Use `tune_neutral_threshold()` and `predict_with_neutral_threshold()` |
| CalibratedClassifierCV error | Ensure `scikit-learn<1.6` is installed (cv='prefit' removed in 1.6+) |
