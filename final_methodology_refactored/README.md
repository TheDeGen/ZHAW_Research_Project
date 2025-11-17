# Energy Price Prediction Pipeline - Refactored

This is a modular, production-ready implementation of the energy price prediction pipeline. The original monolithic notebook (`final_v3.ipynb`) has been refactored into reusable Python modules with a clean, streamlined notebook interface.

## Overview

This pipeline predicts 24-hour ahead energy price movements (Long/Neutral/Short) using:
- **News signals**: German energy news classified and embedded using NLP models
- **Time-decay aggregation**: Exponentially weighted aggregation of news signals
- **Energy market features**: Price, load, and temporal features
- **Machine learning**: XGBoost and LightGBM classifiers with hyperparameter optimization

## Current NLP Implementation

- **Zero-shot topic attribution**: Article titles are processed with `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, returning the top label and confidence score; low-confidence items tagged as "other" are re-run on descriptions to recover additional signal, yet the current median score (~0.26) indicates most articles sit near decision boundaries.
- **Sentence embeddings**: Titles are encoded via `paraphrase-multilingual-MiniLM-L12-v2`, cached as 384-dimensional vectors, and reused across runs to keep notebook execution fast.
- **Time-decayed features**: For every forecast timestamp, topic counts and embedding averages are computed with exponential decay, producing hourly features that capture both discrete classifications and dense semantic context before dimensionality reduction (UMAP or PCA to 50 components).
- **Dataset assembly**: Decayed NLP features are merged with price, load, and temporal baselines, standardised per split, and evaluated through expanding-window RidgeCV to rank decay parameters prior to tree-based model training.
- **Quality diagnostics**: `visualization.plot_embedding_quality` projects the raw article embeddings with UMAP and t-SNE; the present plots show limited clustering by topic, consistent with the low zero-shot confidence and high share of "other" assignments.

## Suggested Enhancements

- **Refine label set and prompts**: Shorten and disambiguate German topic statements, consider hierarchical routing (energy vs. non-energy before fine-grained classes), and drop or merge rarely triggered labels to reduce overlap driving low confidence.
- **Enrich classification inputs**: Concatenate title and description, optionally translate mixed-language snippets, and filter boilerplate updates so the zero-shot model sees fuller, more contextual text.
- **Confidence-aware feature weighting**: Multiply time-decayed topic counts and embedding contributions by the classifier score, track per-window score statistics, and suppress articles with confidence below a chosen threshold to limit noise.
- **Evaluate alternative embeddings**: Benchmark models such as `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` or `intfloat/multilingual-e5-large`, and explore supervised dimensionality reduction (e.g., LDA or supervised UMAP) to improve downstream separability.
- **Supervised calibration path**: Label a small corpus of articles to fine-tune or calibrate the classifier, using the current zero-shot outputs as weak labels; even a lightweight adapter can raise median confidence and sharpen topic-driven signals.
- **Broaden decay search**: Extend lookback/decay grids, test asymmetric windows for topics vs. embeddings, and include alternative decay functions (piecewise linear, sigmoid) to better capture how narratives influence prices over time.

## Project Structure

```
final_methodology_refactored/
├── config/
│   ├── __init__.py
│   ├── pipeline_config.py      # All configurable parameters
│   └── model_config.py          # Model-specific configurations
│
├── scripts/
│   ├── __init__.py
│   ├── device_utils.py          # GPU/CPU detection, resource management
│   ├── profiling.py             # Performance profiling and telemetry
│   ├── data_ingestion.py        # Data loading and preprocessing
│   ├── feature_engineering.py   # NLP, embeddings, time-decay features
│   ├── model_utils.py           # XGBoost/LightGBM model training
│   ├── evaluation.py            # Metrics, bootstrap CIs, statistical tests
│   └── visualization.py         # Plotting functions
│
├── notebooks/
│   └── pipeline_execution.ipynb # Clean, streamlined execution notebook
│
├── data/                        # Place your data files here
│   ├── german_news_v1.csv
│   └── energy_baseline.csv
│
├── outputs/                     # Model outputs and artifacts
│   └── .cache/                  # Embedding and feature caches
│
└── README.md                    # This file
```

## Key Benefits

### 1. **Maintainability**
- Functions organized by purpose in separate modules
- Easy to locate, debug, and test specific functionality
- Clear separation of concerns

### 2. **Reusability**
- All functions can be imported and used in other projects
- No code duplication
- Modular design allows mixing and matching components

### 3. **Configurability**
- All parameters centralized in `config/pipeline_config.py`
- Easy to modify without touching code
- Configuration inheritance and overrides supported

### 4. **Performance**
- GPU-aware device detection and optimization
- Embedding and feature caching to disk
- Parallel processing where applicable
- Resource profiling for bottleneck identification

### 5. **Notebook Simplicity**
- Reduced from 122 cells (~4000 lines) to ~25 cells (~300 lines)
- Focus on results and outputs, not implementation
- Clear, step-by-step execution flow

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all CPU packages from requirements.txt
pip install -r requirements.txt

# Install GPU packages via conda (recommended for Linux/GPU servers)
conda install -c rapidsai -c conda-forge -c nvidia cuml-cu12 cupy
```

**Note**: cuML and cupy are commented out in `requirements.txt` because they require special installation via conda or the RAPIDS pip index. The conda installation method shown above is the most reliable.

### 2. Prepare Data

Place your data files in the `data/` directory:
- `german_news_v1.csv` - News articles with title, description, publishedAt
- `energy_baseline.csv` - Energy market data with price, load, timestamp

### 3. Run the Pipeline

Open `notebooks/pipeline_execution.ipynb` in Jupyter and run cells sequentially.

### 4. Customize Parameters (Optional)

Edit `config/pipeline_config.py` to modify:
- Time-decay parameters (λ, lookback window)
- Target definition (spread deadband)
- Cross-validation settings
- Grid search ranges
- Model hyperparameters
- Hierarchical topic routing (stage definitions, thresholds)

## Module Documentation

### config/pipeline_config.py
Central configuration file containing:
- Time-series parameters (forecast horizon, lookback window)
- Feature engineering parameters (time decay, PCA components)
- Dataset split ratios (train/val/test)
- Cross-validation settings
- NLP configuration (embedding model, topic labels)
- XGBoost hyperparameter distributions
- Data paths

**Key Parameters:**
- `FORECAST_HORIZON_HOURS = 24` - Prediction horizon
- `DEFAULT_LOOKBACK_WINDOW = 336` - News lookback (2 weeks)
- `DEFAULT_DECAY_LAMBDA = 0.05` - Time decay rate
- `SPREAD_TARGET_DEADBAND = 10.0` - Neutral class threshold (EUR/MWh)
- `HIERARCHICAL_TOPIC_GROUPS` - Stage-1 routing categories mapped to leaf topics
- `HIERARCHICAL_ROUTING_SETTINGS` - Stage order, confidence thresholds and fallback behaviour

### scripts/device_utils.py
**Functions:**
- `detect_compute_device()` - Auto-detect GPU/CPU and recommend batch sizes
- `ensure_tensor_device()` - Move tensors to specified device
- `resolve_hf_device()` - Resolve device for HuggingFace models

**Usage:**
```python
from scripts import device_utils
device_config = device_utils.detect_compute_device(task='embeddings', verbose=True)
```

### scripts/profiling.py
**Classes:**
- `StageProfiler` - Context manager for resource monitoring

**Usage:**
```python
from scripts import profiling
with profiling.StageProfiler("Data Loading", device_config):
    # Your code here
    pass
```

### scripts/data_ingestion.py
**Functions:**
- `run_ingestion_stage()` - Load and preprocess news and energy data

**Returns:**
- `news_df` - Preprocessed news dataframe
- `energy_df` - Preprocessed energy dataframe
- `master_df` - Master feature dataframe with target variable

### scripts/feature_engineering.py
**Main Functions:**
- `run_embedding_stage()` - Hierarchical zero-shot topic classification
- `compute_embeddings()` - Generate sentence embeddings (cached)
- `compute_time_decayed_topic_counts()` - Time-weighted topic aggregation
- `compute_time_decayed_embeddings()` - Time-weighted embedding aggregation
- `reduce_embeddings_gpu_first()` - UMAP/PCA dimensionality reduction (cached)
- `grid_search_time_decay_params()` - Parameter optimization using Ridge CV

**Key Features:**
- Disk caching for embeddings and reduced features
- GPU-first computation with CPU fallback
- Efficient binary search for time-window lookups

### scripts/model_utils.py
**Classes:**
- `ExpandingWindowSplitter` - Custom time-series CV splitter

**Functions:**
- `build_xgb_classifier()` - Build XGBoost classifier with device optimization
- `run_xgb_random_search()` - Hyperparameter tuning with RandomizedSearchCV
- `map_target_to_binary()` - Convert 3-class to binary target

### scripts/evaluation.py
**Functions:**
- `bootstrap_confidence_interval()` - Bootstrap 95% CIs for metrics
- `compare_models_statistically()` - McNemar's test for model comparison
- `_safe_multiclass_auc()` - Robust multiclass AUC computation
- `actions_to_returns()` - Convert trading signals to returns
- `get_column_name()` - Helper for flexible column matching

### scripts/visualization.py
**Functions:**
- `plot_confusion_matrices()` - Side-by-side confusion matrices
- `plot_feature_importance()` - Bar plot of top features

## Pipeline Stages

### Stage 1: Data Ingestion
- Load news and energy data
- Create target variable (3-class spread classification)
- Generate temporal features (hour, day, week, month)
- Create lagged features (24h, 168h lags)

### Stage 2: News Processing
**2A. Topic Classification**
- Hierarchical routing with stage-1 categories (Nachfrage, Angebot, Brennstoffpreise, Makrofinanzen, Geopolitik, Wetter, Sonstiges)
- Stage-2 selection of leaf topics ensures compatibility with downstream feature aggregation
- Model: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
- Batch processing with GPU acceleration
- Re-classification of "other" articles using descriptions
- Diagnostics: `classification_stage1`, `classification_stage1_score` columns added for QC

**2B. Sentence Embeddings**
- Generate 384-dim embeddings using multilingual MiniLM
- GPU-accelerated inference (fp16 on CUDA)
- Disk caching for reuse

### Stage 3: Feature Engineering
**3A. Time-Decayed Topic Counts**
- Exponential time decay: `weight = exp(-λ * hours_since)`
- Vectorized computation across timestamps
- Separate weighted count per topic

**3B. Time-Decayed Embeddings**
- Weighted average of embeddings in time window
- Binary search for efficient time-range queries
- Same decay formula as topics

**3C. Dimensionality Reduction**
- Reduce 384-dim embeddings to 50-dim
- GPU-first UMAP (cuML) with CPU fallback
- Disk caching with checksum validation

### Stage 4: Parameter Optimization (Optional)
- Grid search over (lookback_window, decay_lambda) combinations
- Ridge classifier with expanding-window CV
- Parallel evaluation using joblib
- Rank by validation accuracy

### Stage 5: Model Training
- XGBoost RandomizedSearchCV (softmax objective + inverse-frequency sample weights)
- Expanding-window time-series CV
- GPU acceleration when available
- Regularised hyperparameter distributions from config (100–400 estimators, explicit `min_child_weight`, `reg_alpha`, `reg_lambda`)
- Native argmax probabilities downstream (no additional threshold tuning)

### Stage 6: Evaluation
- Confusion matrices
- Classification reports
- Feature importance plots
- Bootstrap confidence intervals
- McNemar's test for model comparison

## Configuration Examples

### Modify Time-Decay Parameters
```python
# In config/pipeline_config.py or notebook
DEFAULT_LOOKBACK_WINDOW = 504  # 3 weeks instead of 2
DEFAULT_DECAY_LAMBDA = 0.01    # Slower decay
```

### Change Data Splits
```python
TRAIN_RATIO = 0.6
VAL_RATIO = 0.3
TEST_RATIO = 0.1
```

### Adjust Cross-Validation
```python
N_CV_SPLITS = 3          # Fewer splits for faster iteration
CV_STEP_SIZE_HOURS = 48  # Larger step size
```

### Modify Grid Search Ranges
```python
LOOKBACK_WINDOWS = [168, 336, 504, 672]  # 1, 2, 3, 4 weeks
TIME_DECAY_LAMBDAS = [0.001, 0.01, 0.05, 0.1, 0.2]
```

## Performance Tips

### 1. Use Caching
- First run generates embeddings and caches to disk
- Subsequent runs load from cache (much faster)
- Cache invalidated automatically on data changes

### 2. GPU Acceleration
- Install CUDA toolkit and cupy for NVIDIA GPUs
- Models automatically detect and use GPU
- Batch sizes auto-adjusted based on GPU memory

### 3. Parallel Processing
- joblib parallelizes grid search and Ridge CV
- Set `n_jobs=-1` to use all cores
- GPU-accelerated models use `n_jobs=1` (GPU handles parallelism)

### 4. Memory Management
- Process data in batches for large datasets
- Use `news_sample` parameter for quick testing
- Clear GPU cache after heavy operations

## Troubleshooting

### Import Errors
```python
# Add parent directory to path
import sys
sys.path.insert(0, str(Path('../scripts').parent.resolve()))
```

### GPU Not Detected
- Check CUDA installation: `torch.cuda.is_available()`
- Install correct PyTorch version for your CUDA version
- Apple Silicon: Use MPS backend (automatic)

### Out of Memory
- Reduce batch size in config
- Use smaller PCA components
- Process fewer news articles for testing

### Slow Embeddings
- Check if GPU is being used
- Ensure caching is working (check `.cache/embeddings/`)
- Reduce batch size if GPU memory is limited

## Known Limitations

### Temporal Data Leakage via Lookback Windows

**Issue**: The current train/validation/test splits do not include temporal gaps between splits, which can cause subtle data leakage through feature engineering.

**Root Cause**: Features use lookback windows (up to 504 hours) for time-decayed news aggregation. Without gaps between splits:
- Last training sample: hour 5089
- First validation sample: hour 5090
- Validation sample's features include news from hours [4586:5090] (504-hour lookback)
- This overlaps with training period [0:5089]
- News from hours [4586:5089] appears in BOTH training targets AND validation features

**Impact**:
- Validation and test metrics may be slightly optimistic
- Model may have access to information from the "future" during training through overlapping lookback windows
- Not a severe issue but affects strict temporal validation

**Acknowledged Workaround**:
We acknowledge this limitation. The temporal ordering is preserved (no shuffling), and the expanding window CV splitter maintains chronological order, so the impact is relatively minor. A proper fix would require adding 24-48 hour gaps between splits at the cost of reduced training data.

**Potential Fix** (not implemented):
```python
# In assemble_time_decay_datasets(), add gap parameter:
gap_size = 24  # 24-hour gap between splits
train_end = int(num_samples * train_ratio)
val_start = train_end + gap_size  # Skip 24 hours
val_end = val_start + int(num_samples * val_ratio)
test_start = val_end + gap_size  # Skip another 24 hours
```

## Extending the Pipeline

### Add New Features
1. Create function in appropriate module (e.g., `feature_engineering.py`)
2. Import in notebook
3. Call function and merge with `feature_df`

### Add New Models
1. Create training function in `model_utils.py`
2. Add configuration to `model_config.py`
3. Call from notebook

### Custom Evaluation Metrics
1. Add metric function to `evaluation.py`
2. Use with `bootstrap_confidence_interval()`

## Comparison with Original Notebook

| Aspect | Original (`final_v3.ipynb`) | Refactored |
|--------|---------------------------|-----------|
| **Lines of code** | ~4000 | ~300 (notebook) + ~2000 (modules) |
| **Notebook cells** | 122 | ~25 |
| **Configuration** | Scattered | Centralized |
| **Reusability** | Low (copy-paste) | High (import) |
| **Testability** | Difficult | Easy |
| **Maintainability** | Low | High |
| **Git diff clarity** | Poor | Excellent |
| **IDE support** | Limited | Full |

## Citation

If you use this pipeline in your research, please cite:
```
@software{energy_price_prediction_pipeline,
  title={Energy Price Prediction Pipeline with News Signals},
  author={ZHAW AREP Team},
  year={2025},
  note={Modular ML pipeline for energy market forecasting}
}
```

## License

[Add your license here]

## Contact

For questions or issues, please contact [your contact information]

---

**Happy modeling! 🚀**
