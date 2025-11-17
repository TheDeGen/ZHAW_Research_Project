# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **energy price prediction pipeline** that forecasts German electricity prices 24 hours ahead using:
- **NLP-based news signals**: Zero-shot classification and sentence embeddings from German energy news
- **Time-decay aggregation**: Exponentially weighted news features with configurable lookback windows
- **Energy market features**: Spot prices, day-ahead auction prices, load data, and temporal features
- **Machine learning**: XGBoost and LightGBM classifiers with hyperparameter optimization

The project predicts 3-class spread targets (Long/Neutral/Short) based on the difference between spot prices and day-ahead auction prices.

## Available Specialized Agents

This repository includes custom Claude Code agents for domain-specific tasks. These agents can be invoked when working on specific aspects of the pipeline:

### 1. model-validation-checklist

**When to use**: Before training models, after receiving unexpected results, when reviewing model configurations

**Capabilities**:
- Validates model architecture choices for NLP and finance tasks
- Reviews hyperparameters (learning rate, batch size, regularization, etc.)
- Checks data split strategies for temporal leakage and proper train/val/test ratios
- Verifies evaluation protocols and metrics align with business objectives
- Flags critical issues (look-ahead bias, incorrect loss functions) vs recommendations

**Example use cases**:
- "Review my XGBoost configuration before I start hyperparameter tuning"
- "I'm getting poor test performance after good validation results - what's wrong?"
- "Validate my time-series CV setup for the energy prediction task"

### 2. viz-insights-designer

**When to use**: Creating visualizations for model results, exploring data patterns, generating diagnostic plots

**Capabilities**:
- Designs domain-appropriate visualizations for NLP metrics (confusion matrices, embedding plots, classification reports)
- Creates ML diagnostic charts (learning curves, feature importance, residual plots, ROC curves)
- Builds finance analytics visualizations (time series, drawdown plots, performance attribution)
- Reviews and improves existing visualizations for clarity and insight
- Specifies exact plotting parameters and annotations

**Example use cases**:
- "Create a confusion matrix visualization for my 3-class spread predictions"
- "Plot feature importance from the XGBoost model with proper annotations"
- "Design a time-series plot showing actual vs predicted prices with confidence bands"
- "This learning curve looks weird - can you help diagnose it?"

### 3. gpu-parallelism-reviewer

**When to use**: After writing GPU-accelerated code, when experiencing performance issues, optimizing parallel processing

**Capabilities**:
- Analyzes device detection and GPU availability handling
- Reviews workload splitting and distribution across parallel units
- Identifies unnecessary CPU-GPU memory transfers
- Validates fallback strategies when GPU is unavailable
- Checks for race conditions and synchronization issues
- Suggests performance optimizations (kernel fusion, async transfers, batch sizes)

**Example use cases**:
- "Review my GPU implementation for the embedding computation"
- "My UMAP dimensionality reduction is slower than expected - what's wrong?"
- "Validate the device detection logic in device_utils.py"
- "I'm getting CUDA out-of-memory errors - help optimize memory usage"

### 4. precise-code-answer

**When to use**: Need a quick, factual answer about specific code behavior without elaboration

**Capabilities**:
- Locates exact code snippets relevant to your question
- Provides direct, factual answers (no suggestions or alternatives)
- Quotes minimal necessary code with file references
- Answers questions about specific values, parameters, or implementations

**Example use cases**:
- "What's the default lookback window in the time-decay feature engineering?"
- "Does the data ingestion stage handle missing values in the energy data?"
- "What optimizer does the XGBoost classifier use?"
- "What's the exact threshold for the hierarchical routing stage1?"

**When NOT to use**: Exploratory questions, refactoring suggestions, code reviews, or when you want detailed analysis

### Using Agents Effectively

To invoke an agent, simply ask Claude Code to use it:
```
"Use the model-validation-checklist agent to review my current XGBoost configuration"
"Launch the viz-insights-designer agent to create a feature importance plot"
"Have the gpu-parallelism-reviewer check my embedding computation code"
```

Agents are **context-aware** and can see the full conversation history, so you can reference earlier work without repeating information.

## Repository Structure

### Active Development (Main Pipeline)

**`final_methodology_refactored/`** - Current production-ready implementation
- **`config/`** - All configuration parameters (DO NOT hard-code values; use these configs)
  - `pipeline_config.py` - Data paths, time-decay params, CV settings, NLP config, grid search ranges
  - `model_config.py` - Model-specific configurations
- **`scripts/`** - Modular pipeline components (import these in notebooks)
  - `device_utils.py` - GPU/CPU detection, device management
  - `profiling.py` - Performance profiling and telemetry
  - `data_ingestion.py` - Load and preprocess news and energy data
  - `feature_engineering.py` - NLP embeddings, time-decay, dimensionality reduction
  - `model_utils.py` - XGBoost/LightGBM training utilities
  - `evaluation.py` - Metrics, bootstrap CIs, statistical tests
  - `visualization.py` - Plotting functions
- **`notebooks/`** - Clean execution interface
  - `pipeline_execution.ipynb` - Main notebook (~25 cells, imports from scripts/)
- **`data/`** - Place data files here
  - `german_news_v1.csv` - News articles
  - `energy_baseline.csv` - Energy market data
- **`outputs/`** - Model artifacts and caches
  - `.cache/` - Embedding and feature caches (auto-managed)

### Legacy Code (Reference Only)

- `data_prep/` - News fetchers and data cleaning (completed work)
- `legacy_notebooks/` - Historical notebooks (methodology v1, v2, initial experiments)
- `final_methodology/` - Previous monolithic notebook implementation (~4000 lines)

**IMPORTANT**: Always work in `final_methodology_refactored/`. Do NOT modify legacy code unless explicitly requested.

## Development Workflow

### Running the Pipeline

```bash
# Navigate to refactored directory
cd final_methodology_refactored

# Run the main notebook
jupyter notebook notebooks/pipeline_execution.ipynb
```

The notebook imports modular functions from `scripts/` - execution is ~25 cells total.

### Testing Configuration Changes

```bash
# Edit configuration
vim config/pipeline_config.py  # or model_config.py

# Re-run affected cells in pipeline_execution.ipynb
# Caching ensures only changed components re-execute
```

### Adding New Features

1. **Create function** in appropriate `scripts/` module
2. **Import** in `pipeline_execution.ipynb`
3. **Merge** with `feature_df` before model training

Example:
```python
# In scripts/feature_engineering.py
def compute_custom_feature(master_df):
    master_df['my_feature'] = ...
    return master_df

# In notebook
from scripts.feature_engineering import compute_custom_feature
feature_df = compute_custom_feature(feature_df)
```

## Key Implementation Details

### Configuration Management

**NEVER hard-code parameters** - use centralized config files:

```python
from config import pipeline_config

# Time-decay parameters
lookback_window = pipeline_config.DEFAULT_LOOKBACK_WINDOW  # 336 hours
decay_lambda = pipeline_config.DEFAULT_DECAY_LAMBDA  # 0.05

# Target definition
spread_deadband = pipeline_config.SPREAD_TARGET_DEADBAND  # 7.0 EUR/MWh

# Cross-validation
n_splits = pipeline_config.N_CV_SPLITS  # 8
cv_step = pipeline_config.CV_STEP_SIZE_HOURS  # 72
```

### Device Detection and GPU Usage

The pipeline auto-detects compute devices (CUDA, MPS, CPU):

```python
from scripts import device_utils

# Detect device and get optimal batch sizes
device_config = device_utils.detect_compute_device(task='embeddings', verbose=True)

# Device config provides:
# - device: 'cuda', 'mps', or 'cpu'
# - optimal_batch_size: auto-tuned for GPU memory
# - tree_method: 'hist' or 'gpu_hist' for XGBoost
# - xgb_device: 'cuda' or 'cpu'
# - n_jobs: parallel jobs for CPU operations
```

**GPU Package Installation**:
```bash
# GPU packages (cuML, cupy) require conda installation
conda install -c rapidsai -c conda-forge -c nvidia cuml-cu12 cupy

# CPU-only packages
pip install -r requirements.txt
```

### NLP Pipeline Architecture

**Hierarchical Zero-Shot Classification**:
1. **Stage 1**: Route to high-level categories (Nachfrage, Angebot, Brennstoffpreise, Makrofinanzen, Geopolitik, Wetter, Sonstiges)
2. **Stage 2**: Select leaf topic from category's labels
3. **Fallback**: Low-confidence "other" articles re-classified using descriptions

```python
# Configured in pipeline_config.py
HIERARCHICAL_TOPIC_GROUPS = {
    "Nachfrage": ["der Stromverbrauch in Deutschland steigt", ...],
    "Angebot": ["die Stromerzeugung aus Wind und Sonne steigt", ...],
    # ... more categories
}

HIERARCHICAL_ROUTING_SETTINGS = {
    "stage_order": ["Nachfrage", "Angebot", ...],
    "stage_thresholds": {"stage1": 0.35, "stage2": 0.25},
    "allow_fallback_to_other": True,
}
```

**Embeddings**:
- Model: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
- GPU-accelerated with fp16 on CUDA
- Disk-cached for reuse (`.cache/embeddings/`)

**Time-Decay Aggregation**:
- Formula: `weight = exp(-λ * hours_since)`
- Separate features for topic counts and embedding averages
- Dimensionality reduction: UMAP (GPU-first via cuML) or PCA to 50 components

### Model Training

**XGBoost with Regularization**:
```python
from scripts.model_utils import build_xgb_classifier, run_xgb_random_search

# Build classifier with device optimization
xgb = build_xgb_classifier(
    random_state=42,
    device_config=device_config,
    num_classes=3
)

# Hyperparameter tuning with expanding-window CV
results = run_xgb_random_search(
    X_train, y_train,
    param_distributions=pipeline_config.XGB_PARAM_DISTRIBUTIONS,
    n_iter=pipeline_config.XGB_RANDOM_SEARCH_ITERS,
    device_config=device_config,
    random_state=42
)
```

**Features**:
- Softmax objective for multiclass
- Inverse-frequency sample weights for class imbalance
- Regularized hyperparameter distributions (L1/L2, min_child_weight, gamma)
- Native argmax predictions (no threshold tuning)

### Caching System

The pipeline caches expensive computations to disk:

1. **Embeddings**: `.cache/embeddings/{hash}.pkl`
2. **Reduced embeddings**: `.cache/reduced_embeddings/{hash}.pkl`
3. **Time-decay features**: `.cache/time_decay/{hash}.pkl`

Caches are **automatically invalidated** when input data changes (checksum-based).

**Clear cache manually**:
```bash
rm -rf final_methodology_refactored/outputs/.cache
```

### Cross-Validation Strategy

**Expanding Window Splitter** (preserves temporal ordering):

```python
from scripts.model_utils import ExpandingWindowSplitter

cv = ExpandingWindowSplitter(
    n_splits=8,
    step_size=72,  # hours
    min_train_size=336  # 2 weeks minimum
)

# Train size grows, test window slides forward
# Split 0: train[0:336],    test[336:408]
# Split 1: train[0:408],    test[408:480]
# Split 2: train[0:480],    test[480:552]
# ...
```

**Known Limitation**: No temporal gaps between splits, causing minor data leakage through lookback windows. See README section "Temporal Data Leakage via Lookback Windows" for details.

### Data Splits

**Train/Val/Test** with no shuffling (strict temporal ordering):
```python
# In pipeline_config.py
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
```

Split boundaries preserved chronologically:
- Train: earliest 70% of data
- Val: next 20%
- Test: final 10%

## Common Tasks

### Modify Time-Decay Parameters

```python
# In config/pipeline_config.py
DEFAULT_LOOKBACK_WINDOW = 504  # 3 weeks (was 336)
DEFAULT_DECAY_LAMBDA = 0.01    # Slower decay (was 0.05)
```

### Add New Topic Labels

```python
# In config/pipeline_config.py
HIERARCHICAL_TOPIC_GROUPS["Nachfrage"].append(
    "neue Label-Beschreibung für Nachfrage"
)
```

### Change XGBoost Hyperparameter Search

```python
# In config/pipeline_config.py
XGB_PARAM_DISTRIBUTIONS = {
    'n_estimators': stats.randint(200, 500),  # Increase range
    'max_depth': stats.randint(3, 8),
    # ... other params
}
```

### Debug GPU Issues

```python
# Check GPU availability
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"MPS: {torch.backends.mps.is_available()}")

# Force CPU mode
device_config['device'] = 'cpu'
device_config['xgb_device'] = 'cpu'
```

### Profile Performance

```python
from scripts import profiling

with profiling.StageProfiler("My Stage", device_config):
    # Code to profile
    result = expensive_function()
```

## File Naming Conventions

- **Config files**: `{purpose}_config.py` (e.g., `pipeline_config.py`, `model_config.py`)
- **Script modules**: `{function}.py` (e.g., `feature_engineering.py`, `model_utils.py`)
- **Data files**: `{source}_{version}.csv` (e.g., `german_news_v1.csv`)
- **Notebooks**: `{purpose}_{type}.ipynb` (e.g., `pipeline_execution.ipynb`)

## Code Style Guidelines

1. **Import from scripts/**:
   ```python
   from scripts import device_utils, data_ingestion, feature_engineering
   from config import pipeline_config
   ```

2. **Use config parameters**:
   ```python
   # Good
   horizon = pipeline_config.FORECAST_HORIZON_HOURS

   # Bad
   horizon = 24  # hard-coded
   ```

3. **Device-aware code**:
   ```python
   device_config = device_utils.detect_compute_device()
   batch_size = device_config['optimal_batch_size']
   ```

4. **Preserve temporal ordering** (NEVER shuffle time-series data)

5. **Document performance-critical sections** with profiling

## Data Requirements

### Input Data Format

**`german_news_v1.csv`**:
- `publishedAt` (datetime): Article publication timestamp
- `title` (str): Article title (used for classification and embeddings)
- `description` (str): Article description (fallback for low-confidence classifications)

**`energy_baseline.csv`**:
- `Timestamp` (datetime): Hourly timestamp
- `Spot Price` (float): Realized spot price (EUR/MWh)
- `Day Ahead Auction` (float): Day-ahead auction price (EUR/MWh)
- `Load` (float): Power consumption (MW)

### Minimum Data Requirements

- **Minimum timestamp**: Configurable via `pipeline_config.MIN_TIMESTAMP` (default: "2024-11-01")
- **Minimum training size**: 336 hours (2 weeks) for CV splitter
- **Minimum lookback**: Matches `DEFAULT_LOOKBACK_WINDOW` (336 hours)

## Troubleshooting

### "No GPU detected" warnings
- Check CUDA installation: `torch.cuda.is_available()`
- Install correct PyTorch version for your CUDA version
- MPS (Apple Silicon) automatically detected

### Out of memory errors
- Reduce `optimal_batch_size` in device_config
- Use smaller `DEFAULT_LOOKBACK_WINDOW`
- Sample fewer news articles for testing: `news_sample=10000`

### Import errors from scripts/
```python
# Add parent directory to path (in notebook)
import sys
from pathlib import Path
sys.path.insert(0, str(Path('../scripts').parent.resolve()))
```

### Cache invalidation issues
```bash
# Force cache rebuild
rm -rf outputs/.cache
```

### Zero-shot classification low confidence
- Check `HIERARCHICAL_ROUTING_SETTINGS` thresholds (default: stage1=0.35, stage2=0.25)
- Review `HIERARCHICAL_TOPIC_GROUPS` for label overlap
- Consider re-classification using descriptions (enabled by default)

## Performance Optimization

1. **Use GPU acceleration**: Install cuML/cupy for UMAP, enable CUDA for XGBoost
2. **Enable caching**: First run generates caches, subsequent runs are much faster
3. **Parallel processing**: Set `n_jobs=-1` for CPU-bound operations
4. **Batch size tuning**: Auto-detected based on GPU memory, but can override
5. **Reduce grid search space**: Fewer `LOOKBACK_WINDOWS` and `TIME_DECAY_LAMBDAS`

## Testing Strategy

The pipeline uses **strict temporal validation** (no shuffling):

1. **Train/Val/Test splits**: Fixed 70/20/10 ratio
2. **Expanding window CV**: Growing training set, sliding test window
3. **Bootstrap CIs**: 1000 resamples for metric confidence intervals
4. **McNemar's test**: Statistical comparison between models

**Never shuffle time-series data** - this would introduce temporal leakage.

## Git Workflow

When committing model changes:

```bash
# Check which files changed
git status

# Stage configuration changes
git add final_methodology_refactored/config/*.py

# Stage script changes
git add final_methodology_refactored/scripts/*.py

# DO NOT commit cache directories
# (already in .gitignore if configured properly)

# Commit with descriptive message
git commit -m "Tune XGBoost regularization parameters to reduce overfitting"
```

## Important Notes

1. **DO NOT modify legacy code** (`data_prep/`, `legacy_notebooks/`, `final_methodology/`) unless explicitly requested
2. **DO NOT hard-code parameters** - always use `config/pipeline_config.py`
3. **DO NOT shuffle time-series data** - maintains temporal ordering
4. **DO NOT commit large cache files** - regenerated automatically
5. **DO optimize for GPU when available** - significant speedup for embeddings and UMAP
6. **DO use expanding window CV** - respects temporal dependencies
7. **DO document major changes** in commit messages and README

## Model Artifacts

Trained models can be saved/loaded:

```python
from scripts.save_models import save_model, load_model

# Save model
save_model(xgb_model, "outputs/xgb_best.pkl")

# Load model
loaded_model = load_model("outputs/xgb_best.pkl")
```

## Extending the Pipeline

### Add a new evaluation metric

```python
# In scripts/evaluation.py
def my_custom_metric(y_true, y_pred):
    # Implementation
    return score

# In notebook
from scripts.evaluation import bootstrap_confidence_interval, my_custom_metric
ci = bootstrap_confidence_interval(y_test, y_pred, metric_fn=my_custom_metric)
```

### Add a new dimensionality reduction method

```python
# In scripts/feature_engineering.py
def reduce_with_custom_method(embeddings, n_components=50):
    # Implementation
    return reduced_embeddings

# In notebook
reduced = reduce_with_custom_method(embeddings)
```

### Add a new model type

```python
# In scripts/model_utils.py
def build_custom_classifier(device_config):
    # Implementation
    return classifier

# In config/model_config.py
CUSTOM_PARAM_DISTRIBUTIONS = {...}

# In notebook
from scripts.model_utils import build_custom_classifier
model = build_custom_classifier(device_config)
```
