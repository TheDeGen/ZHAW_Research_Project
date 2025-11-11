"""
Feature Engineering
===================
Functions for NLP embeddings, time-decay aggregation, dimensionality reduction,
and parameter grid search.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from joblib import Parallel, delayed

from . import device_utils
from . import profiling


def run_embedding_stage(
    news_df: pd.DataFrame,
    candidate_labels: list[str],
    hypothesis_template: str,
    device_config: dict,
    batch_size: int | None = None,
    model_name: str = "Sahajtomar/German_Zeroshot",
):
    """
    Run zero-shot classification on news articles.

    Args:
        news_df: News dataframe with 'title' and 'description' columns
        candidate_labels: List of candidate labels for zero-shot classification
        hypothesis_template: Template for zero-shot classification hypothesis
        device_config: Device configuration dict from detect_compute_device()
        batch_size: Batch size for inference (optional, auto-detected if None)
        model_name: HuggingFace model name for zero-shot classification

    Returns:
        dict containing:
            - news_df: News dataframe with 'classification' and 'classification_score' columns
            - classifier: Fitted classifier pipeline
            - batch_size: Effective batch size used
            - hf_device: HuggingFace device used
    """
    news_df = news_df.copy()

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.backends.cudnn.benchmark = True

    primary_device = device_config.get('device', 'cpu')
    effective_batch_size = batch_size or device_config.get("optimal_batch_size", 32)
    hf_device = device_utils.resolve_hf_device(primary_device)
    torch_dtype = torch.float16 if hf_device in (0, "cuda") else torch.float32

    pipeline_kwargs = {
        "model": model_name,
        "batch_size": effective_batch_size,
        "torch_dtype": torch_dtype,
    }

    if hf_device != -1:
        if device_utils.ACCELERATE_AVAILABLE:
            pipeline_kwargs["device"] = hf_device
        else:
            print("⚠️ Hugging Face accelerate is not installed; zero-shot classifier will run on CPU.")
            pipeline_kwargs["torch_dtype"] = torch.float32

    def instantiate_zero_shot(**kwargs):
        return pipeline("zero-shot-classification", **kwargs)

    resolved_hf_device = pipeline_kwargs.get("device", hf_device)
    if not device_utils.ACCELERATE_AVAILABLE and hf_device != -1:
        resolved_hf_device = "cpu"

    try:
        classifier = instantiate_zero_shot(**pipeline_kwargs)
    except ValueError as exc:
        message = str(exc).lower()
        if "cannot be moved to a specific device" in message and pipeline_kwargs.pop("device", None) is not None:
            print("⚠️ accelerate-managed model detected; retrying without explicit device placement.")
            classifier = instantiate_zero_shot(**pipeline_kwargs)
            resolved_hf_device = "accelerate-auto"
        else:
            raise
    except RuntimeError as exc:
        message = str(exc)
        if hf_device == "mps" and "mps" in message.lower():
            print("⚠️ MPS pipeline creation failed; falling back to CPU.")
            pipeline_kwargs.pop("device", None)
            pipeline_kwargs["torch_dtype"] = torch.float32
            classifier = instantiate_zero_shot(**pipeline_kwargs)
            resolved_hf_device = "cpu"
        else:
            raise

    if "device" not in pipeline_kwargs and resolved_hf_device != "accelerate-auto":
        candidate_device = getattr(classifier, "device", None)
        if candidate_device is not None:
            resolved_hf_device = candidate_device

    resolved_hf_device_repr = resolved_hf_device
    if isinstance(resolved_hf_device_repr, torch.device):
        device_type = resolved_hf_device_repr.type
        index = resolved_hf_device_repr.index
        resolved_hf_device_repr = device_type if index is None else f"{device_type}:{index}"

    def classify_batch(texts, labels, hypothesis_template, show_progress=True):
        valid_pairs = [(idx, text) for idx, text in enumerate(texts) if pd.notna(text) and text.strip()]
        if not valid_pairs:
            return {}, {}

        _, filtered_texts = zip(*valid_pairs)
        if show_progress:
            print(f"Processing {len(filtered_texts)} texts with batch_size={effective_batch_size}")

        results = classifier(
            list(filtered_texts),
            labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )
        if isinstance(results, dict):
            results = [results]

        classifications = {}
        scores = {}
        for (idx, _), result in zip(valid_pairs, results):
            classifications[idx] = result['labels'][0]
            scores[idx] = float(result['scores'][0])

        for idx in range(len(texts)):
            classifications.setdefault(idx, None)
            scores.setdefault(idx, 0.0)

        return classifications, scores

    titles = news_df['title'].tolist()
    classifications_dict, scores_dict = classify_batch(
        titles,
        candidate_labels,
        hypothesis_template,
        show_progress=True,
    )

    news_df['classification'] = [classifications_dict[i] for i in range(len(news_df))]
    news_df['classification_score'] = [scores_dict[i] for i in range(len(news_df))]

    # Re-classify "other" using description
    other_label = "kein Bezug zu Energie, Wetter oder Finanzmärkten"
    other_mask = news_df['classification'] == other_label
    num_other = other_mask.sum()

    if num_other > 0:
        other_indices = news_df[other_mask].index
        descriptions = news_df.loc[other_indices, 'description'].tolist()
        other_classifications_dict, other_scores_dict = classify_batch(
            descriptions,
            candidate_labels,
            hypothesis_template,
            show_progress=True,
        )
        for i, idx in enumerate(other_indices):
            news_df.loc[idx, 'classification'] = other_classifications_dict[i]
            news_df.loc[idx, 'classification_score'] = other_scores_dict[i]

    final_other = (news_df['classification'] == other_label).sum()

    print(f"Classification completed: {len(news_df)} articles processed")
    print(f"Articles classified as 'other': {final_other} ({final_other / len(news_df) * 100:.1f}%)")
    print("\nClassification distribution:")
    print(news_df['classification'].value_counts())
    print(f"\nAverage score: {news_df['classification_score'].mean():.3f}")
    print(f"Median score: {news_df['classification_score'].median():.3f}")

    return {
        "news_df": news_df,
        "classifier": classifier,
        "batch_size": effective_batch_size,
        "hf_device": resolved_hf_device_repr,
    }


def compute_embeddings(
    news_df: pd.DataFrame,
    device_config: dict,
    model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
    batch_size: int | None = None,
    show_progress: bool = True,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Attach SentenceTransformer embeddings to the news dataframe with GPU-first caching.

    Args:
        news_df: News dataframe with 'title' column
        device_config: Device configuration dict
        model_name: SentenceTransformer model name
        batch_size: Batch size for embedding generation
        show_progress: Show progress bar
        cache_dir: Directory for caching embeddings

    Returns:
        News dataframe with 'embedding' column added
    """
    import hashlib
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    def _title_checksum(frame: pd.DataFrame) -> str:
        digest = hashlib.sha1()
        for title in frame['title'].fillna('').astype(str):
            digest.update(title.encode('utf-8'))
        return digest.hexdigest()

    cache_root = cache_dir or Path('.cache/embeddings')
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / 'news_embeddings.parquet'
    manifest_path = cache_root / 'manifest.json'

    checksum = _title_checksum(news_df)
    if cache_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('checksum') == checksum:
                embeddings_table = pq.read_table(cache_path)
                vectors = np.array([list_val for list_val in embeddings_table['embedding'].to_pylist()], dtype=np.float32)
                enriched_df = news_df.copy()
                enriched_df['embedding'] = [vec for vec in vectors]
                print(f"Loaded embeddings from cache ({cache_path})")
                return enriched_df
        except Exception as cache_exc:
            print(f"Cache read skipped: {cache_exc}")

    primary_device = device_config.get('device', 'cpu')
    hf_device = primary_device if primary_device in {"cuda", "mps"} else "cpu"
    effective_batch_size = batch_size or device_config.get('optimal_batch_size', 32)

    model = SentenceTransformer(model_name, device=hf_device)
    if hf_device == "cuda":
        try:
            model = model.half()
            print("Model converted to fp16 (half precision)")
        except Exception as exc:
            print(f"fp16 conversion skipped: {exc}")

    texts = [
        (news_df.iloc[idx]['title'] if pd.notna(news_df.iloc[idx]['title']) else '').strip()
        for idx in range(len(news_df))
    ]
    valid_pairs = [(i, text) for i, text in enumerate(texts) if text]
    if not valid_pairs:
        print("Warning: No valid texts found for embedding!")
        return news_df

    _, valid_texts = zip(*valid_pairs)
    print(f"Using batch_size={effective_batch_size} for embedding computation on device={hf_device}")

    try:
        embeddings_array = model.encode(
            list(valid_texts),
            batch_size=effective_batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=False,
        )
    except Exception as primary_exc:
        print(f"Primary embedding pass failed ({primary_exc}); retrying with batch_size=32")
        embeddings_array = model.encode(
            list(valid_texts),
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=False,
        )

    full_embeddings = np.full((len(texts), embeddings_array.shape[1]), np.nan, dtype=np.float32)
    for idx, (row_idx, _) in enumerate(valid_pairs):
        full_embeddings[row_idx] = embeddings_array[idx]

    # Clean up GPU memory
    if hf_device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hf_device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass

    enriched_df = news_df.copy()
    enriched_df['embedding'] = [emb for emb in full_embeddings]

    # Cache embeddings
    try:
        flattened = full_embeddings.reshape(-1)
        list_array = pa.FixedSizeListArray.from_arrays(
            pa.array(flattened, type=pa.float32()), full_embeddings.shape[1]
        )
        pq.write_table(pa.Table.from_arrays([list_array], names=['embedding']), cache_path)
        manifest_path.write_text(
            json.dumps({'checksum': checksum, 'embedding_dim': int(full_embeddings.shape[1])})
        )
    except Exception as write_exc:
        print(f"Embedding cache write skipped: {write_exc}")

    print(f"Embeddings computed: shape {full_embeddings.shape}")
    return enriched_df


def compute_time_decayed_topic_counts(
    news_df: pd.DataFrame,
    master_df: pd.DataFrame,
    lookback_window: int = 336,
    decay_lambda: float = 0.05,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute time-decayed weighted counts for each topic.

    Weight formula: weight = e^(-lambda * hours_since_publication)

    Args:
        news_df: DataFrame with datetime index (publishedAt) and 'classification' column
        master_df: DataFrame with datetime index representing target timestamps
        lookback_window: Number of hours to look back (default: 336h = 2 weeks)
        decay_lambda: Decay rate parameter (default: 0.05)
        verbose: Whether to print progress messages

    Returns:
        DataFrame with datetime index and columns for each topic's weighted count
    """
    td_topic_news_df = news_df.copy()
    if not isinstance(td_topic_news_df.index, pd.DatetimeIndex):
        td_topic_news_df.index = pd.to_datetime(td_topic_news_df.index)
    if td_topic_news_df.index.tz is not None:
        td_topic_news_df.index = td_topic_news_df.index.tz_localize(None)

    topics = td_topic_news_df['classification'].dropna().unique()
    td_topics_df = pd.DataFrame(index=master_df.index)

    if verbose:
        print(f"Computing time-decayed counts for {len(td_topics_df)} timestamps and {len(topics)} topics")
        print(f"Lookback window: {lookback_window}h, decay lambda: {decay_lambda}")
        print("Using vectorized computation for improved performance...")

    timestamps = td_topics_df.index.values
    article_times = td_topic_news_df.index.values
    article_topics = td_topic_news_df['classification'].values

    valid_mask = pd.notna(article_topics)
    article_times_valid = article_times[valid_mask]
    article_topics_valid = article_topics[valid_mask]

    topic_to_idx = {topic: idx for idx, topic in enumerate(topics)}
    weighted_counts_array = np.zeros((len(timestamps), len(topics)))

    if verbose:
        print(f"Processing {len(article_times_valid)} valid articles across {len(timestamps)} timestamps")

    for i, timestamp in enumerate(tqdm(timestamps, desc="Processing timestamps", leave=True, disable=not verbose)):
        cutoff_time = timestamp - np.timedelta64(int(lookback_window), 'h')
        time_mask = (article_times_valid >= cutoff_time) & (article_times_valid <= timestamp)
        if not time_mask.any():
            continue

        valid_article_times = article_times_valid[time_mask]
        valid_article_topics = article_topics_valid[time_mask]
        hours_since = (timestamp - valid_article_times).astype('timedelta64[h]').astype(float)
        weights = np.exp(-decay_lambda * hours_since)

        for topic in topics:
            topic_mask = valid_article_topics == topic
            weighted_counts_array[i, topic_to_idx[topic]] = np.sum(weights[topic_mask])

    for idx, topic in enumerate(topics):
        td_topics_df[topic] = weighted_counts_array[:, idx]

    return td_topics_df


def compute_time_decayed_embeddings(
    news_df: pd.DataFrame,
    master_df: pd.DataFrame,
    lookback_window: int = 336,
    decay_lambda: float = 0.05,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute time-decayed weighted average embeddings (optimized version).

    Weight formula: weight = e^(-lambda * hours_since_publication)

    Args:
        news_df: DataFrame with 'publishedAt' as DatetimeIndex and 'embedding' column
        master_df: DataFrame with datetime index representing target timestamps
        lookback_window: Number of hours to look back (default: 336h = 2 weeks)
        decay_lambda: Decay rate parameter (default: 0.05)
        verbose: Whether to print progress messages

    Returns:
        Array of weighted average embeddings with shape (n_timestamps, embedding_dim)
    """
    if not isinstance(news_df.index, pd.DatetimeIndex):
        raise ValueError("news_df must have a DatetimeIndex")
    if news_df.index.name != 'publishedAt':
        raise ValueError("news_df index must be named 'publishedAt'")

    td_embedding_news_df = news_df.copy()
    if td_embedding_news_df.index.tz is not None:
        td_embedding_news_df.index = td_embedding_news_df.index.tz_localize(None)

    if verbose:
        print(f"Computing time-decayed weighted average embeddings for {len(master_df)} timestamps")
        print(f"Lookback window: {lookback_window}h, decay lambda: {decay_lambda}")
        print("Using optimized vectorized computation...")

    valid_embeddings_mask = td_embedding_news_df['embedding'].notna()
    if not valid_embeddings_mask.any():
        raise ValueError("No valid embeddings found")

    embedding_series = td_embedding_news_df.loc[valid_embeddings_mask, 'embedding']
    index_series = td_embedding_news_df.index[valid_embeddings_mask]

    article_embeddings_list: list[np.ndarray] = []
    for emb in embedding_series:
        if isinstance(emb, pd.Series):
            emb = emb.to_numpy()
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim > 1:
            emb = emb.reshape(-1)
        article_embeddings_list.append(emb)

    if not article_embeddings_list:
        raise ValueError("All embeddings were empty after filtering")

    embedding_lengths = {arr.shape[0] for arr in article_embeddings_list}
    if len(embedding_lengths) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {embedding_lengths}")

    embedding_dim = embedding_lengths.pop()
    article_embeddings_array = np.vstack(article_embeddings_list)
    article_times_valid = index_series.values.astype('datetime64[ns]')

    # Sort by time for efficient searching
    sort_indices = np.argsort(article_times_valid)
    article_times_valid = article_times_valid[sort_indices]
    article_embeddings_array = article_embeddings_array[sort_indices]

    timestamps = master_df.index.values.astype('datetime64[ns]')
    weighted_embeddings_array = np.zeros((len(timestamps), embedding_dim), dtype=np.float32)

    if verbose:
        print(f"Processing {len(article_embeddings_array)} valid embeddings across {len(timestamps)} timestamps")
        print("Using binary search for efficient article lookup...")

    lookback_delta = np.timedelta64(int(lookback_window), 'h')
    cutoff_times = timestamps - lookback_delta

    for i in tqdm(range(len(timestamps)), desc="Processing timestamps", disable=not verbose):
        timestamp = timestamps[i]
        cutoff_time = cutoff_times[i]

        start_idx = np.searchsorted(article_times_valid, cutoff_time, side='left')
        end_idx = np.searchsorted(article_times_valid, timestamp, side='right')

        if start_idx >= end_idx:
            continue

        window_embeddings = article_embeddings_array[start_idx:end_idx]
        window_times = article_times_valid[start_idx:end_idx]

        hours_since = (timestamp - window_times).astype('timedelta64[h]').astype(np.float32)
        weights = np.exp(-decay_lambda * hours_since, dtype=np.float32)

        total_weight = np.sum(weights)
        if total_weight > 1e-10:
            weighted_sum = np.sum(weights[:, np.newaxis] * window_embeddings, axis=0)
            weighted_embeddings_array[i] = weighted_sum / total_weight
        else:
            weighted_embeddings_array[i] = np.zeros(embedding_dim, dtype=np.float32)

    if verbose:
        print(f"\nCompleted time-decayed aggregation")
        print(f"Embedding shape: {weighted_embeddings_array.shape}")

    return weighted_embeddings_array


def reduce_embeddings_gpu_first(
    embeddings: np.ndarray,
    index: pd.Index,
    cache_label: str,
    n_components: int = 20,
    use_umap: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reduce embeddings with GPU-first UMAP and disk caching.

    Args:
        embeddings: Embedding array to reduce
        index: Index for the resulting dataframe
        cache_label: Label for caching
        n_components: Number of components to reduce to
        use_umap: Use UMAP if True, otherwise use PCA
        random_state: Random seed

    Returns:
        DataFrame with reduced embeddings
    """
    import hashlib
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    import umap

    cache_root = Path('.cache/embeddings')
    cache_root.mkdir(parents=True, exist_ok=True)

    checksum = hashlib.sha1(embeddings.tobytes()).hexdigest()
    cache_path = cache_root / f'{cache_label}_reduction.parquet'
    manifest_path = cache_root / f'{cache_label}_reduction.json'

    if cache_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('checksum') == checksum and manifest.get('components') == n_components:
                print(f"Loaded reduced embeddings from cache ({cache_path})")
                return pq.read_table(cache_path).to_pandas()
        except Exception as cache_exc:
            print(f"Cache read failed ({cache_exc}); recomputing")
            cache_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)

    backend = "pca"
    reduced = None

    if use_umap:
        if device_utils.HAS_CUML_UMAP:
            device_info = device_utils.detect_compute_device(task='embeddings', verbose=False)
            if device_info.get('device') == 'cuda':
                try:
                    import cupy as cp  # type: ignore

                    reducer = device_utils.CUML_UMAP(
                        n_components=n_components,
                        n_neighbors=15,
                        min_dist=0.1,
                        random_state=random_state,
                    )
                    reduced_gpu = reducer.fit_transform(cp.asarray(embeddings))
                    reduced = cp.asnumpy(reduced_gpu)
                    backend = "cuml-umap"
                except Exception as gpu_exc:
                    print(f"cuML UMAP unavailable ({gpu_exc}); reverting to CPU UMAP.")

        if reduced is None:
            reducer = umap.UMAP(n_components=n_components, n_jobs=-1, verbose=False)
            reduced = reducer.fit_transform(embeddings)
            backend = "umap-learn"
    else:
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(embeddings)
        backend = "pca"

    reduced_df = pd.DataFrame(
        reduced,
        index=index,
        columns=[f'embedding_dim_{i}' for i in range(reduced.shape[1])]
    )

    try:
        pq.write_table(pa.Table.from_pandas(reduced_df), cache_path)
        manifest_path.write_text(
            json.dumps({'checksum': checksum, 'components': n_components, 'backend': backend})
        )
    except Exception as write_exc:
        print(f"Reduction cache write skipped: {write_exc}")

    print(f"Embedding reduction backend: {backend}")
    return reduced_df


def evaluate_single_parameter_combination(
    params_key,
    data_dict,
    baseline_features,
    target_column,
    alphas=None,
    max_splits=5
):
    """
    Evaluate a single parameter combination using expanding-window RidgeCV.

    Args:
        params_key: Tuple of (lookback_window, decay_lambda)
        data_dict: Dictionary containing train_df, val_df, scaled_news_features
        baseline_features: List of baseline feature names
        target_column: Name of target column
        alphas: Ridge regression alphas to try
        max_splits: Maximum number of CV splits

    Returns:
        Dictionary with evaluation results
    """
    from sklearn.linear_model import RidgeClassifierCV

    lw, dl = params_key
    dataset_name = data_dict['dataset_name']
    train_df = data_dict['train_df']
    val_df = data_dict['val_df']
    scaled_news_features = data_dict['scaled_news_features']

    feature_columns = baseline_features + scaled_news_features
    missing_features = [col for col in feature_columns if col not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing features {missing_features} in dataset {dataset_name}")

    if alphas is None:
        alphas = np.logspace(-3, 3, 13)

    X_train = train_df[feature_columns].fillna(0)
    y_train = train_df[target_column].astype(int)
    X_val = val_df[feature_columns].fillna(0)
    y_val = val_df[target_column].astype(int)

    unique_classes = np.unique(y_train)
    if unique_classes.size < 2:
        return {
            'lookback_window': lw,
            'decay_lambda': dl,
            'dataset_name': dataset_name,
            'best_alpha': None,
            'val_accuracy': np.nan,
            'val_macro_f1': np.nan,
            'model': None,
            'params_key': params_key,
            'skip_reason': 'Training split lacks class diversity'
        }

    if len(X_val) == 0:
        return {
            'lookback_window': lw,
            'decay_lambda': dl,
            'dataset_name': dataset_name,
            'best_alpha': None,
            'val_accuracy': np.nan,
            'val_macro_f1': np.nan,
            'model': None,
            'params_key': params_key,
            'skip_reason': 'Validation split is empty'
        }

    max_possible_splits = max(0, len(X_train) - 1)
    effective_splits = min(max_splits, max_possible_splits)

    if effective_splits < 2:
        return {
            'lookback_window': lw,
            'decay_lambda': dl,
            'dataset_name': dataset_name,
            'best_alpha': None,
            'val_accuracy': np.nan,
            'val_macro_f1': np.nan,
            'model': None,
            'params_key': params_key,
            'skip_reason': 'Insufficient data for expanding-window CV'
        }

    tscv = TimeSeriesSplit(n_splits=effective_splits)
    ridge_cv = RidgeClassifierCV(alphas=alphas, cv=tscv, scoring='accuracy')
    ridge_cv.fit(X_train, y_train)

    val_predictions = ridge_cv.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_macro_f1 = f1_score(y_val, val_predictions, average='macro', zero_division=0)

    return {
        'lookback_window': lw,
        'decay_lambda': dl,
        'dataset_name': dataset_name,
        'best_alpha': ridge_cv.alpha_,
        'val_accuracy': val_accuracy,
        'val_macro_f1': val_macro_f1,
        'model': ridge_cv,
        'params_key': params_key,
        'skip_reason': None
    }


def grid_search_time_decay_params(
    preprocessed_datasets,
    baseline_features,
    target_column,
    alphas=None,
    max_splits=5
):
    """
    Run expanding-window RidgeCV on each precomputed dataset and rank by validation performance.

    Args:
        preprocessed_datasets: Dict mapping (lookback_window, decay_lambda) to data_dict
        baseline_features: List of baseline feature names
        target_column: Name of target column
        alphas: Ridge regression alphas
        max_splits: Maximum CV splits

    Returns:
        List of top results sorted by validation accuracy
    """
    print(f"Grid searching {len(preprocessed_datasets)} parameter combinations...")
    print(f"Using expanding-window RidgeCV confined to training splits (target: {target_column})")
    print(f"Parallelizing evaluation across parameter combinations using joblib...\n")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_single_parameter_combination)(
            params_key,
            data_dict,
            baseline_features,
            target_column,
            alphas,
            max_splits
        )
        for params_key, data_dict in preprocessed_datasets.items()
    )

    valid_results = [res for res in results if res['skip_reason'] is None]
    invalid_results = [res for res in results if res['skip_reason'] is not None]

    if invalid_results:
        print("The following combinations were skipped:")
        for res in invalid_results:
            print(f"  - {res['dataset_name']} (lookback={res['lookback_window']}, "
                  f"lambda={res['decay_lambda']}): {res['skip_reason']}")
        print()

    if not valid_results:
        print("No valid results to rank.")
        return []

    results_sorted = sorted(
        valid_results,
        key=lambda x: (x['val_accuracy'], x['val_macro_f1']),
        reverse=True
    )

    top_results = results_sorted[:3]

    print(f"{'='*80}")
    print("TOP 3 PARAMETER COMBINATIONS:")
    print(f"{'='*80}")
    for idx, result in enumerate(top_results, 1):
        print(
            f"{idx}. dataset={result['dataset_name']} | lookback={result['lookback_window']}h | "
            f"lambda={result['decay_lambda']} | alpha={result['best_alpha']:.4f} | "
            f"Val Accuracy={result['val_accuracy']:.3f} | Val Macro-F1={result['val_macro_f1']:.3f}"
        )

    return top_results
