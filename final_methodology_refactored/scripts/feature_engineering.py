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
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from . import device_utils
from . import profiling


def run_embedding_stage(
    news_df: pd.DataFrame,
    candidate_labels: list[str],
    hypothesis_template: str,
    device_config: dict,
    batch_size: int | None = None,
    model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    hierarchical_topic_groups: dict[str, list[str]] | None = None,
    routing_settings: dict | None = None,
):
    """
    Run zero-shot classification on news articles.

    Args:
        news_df: News dataframe with 'title' and 'description' columns
        candidate_labels: List of candidate labels for zero-shot classification (leaf topics)
        hypothesis_template: Template for zero-shot classification hypothesis
        device_config: Device configuration dict from detect_compute_device()
        batch_size: Batch size for inference (optional, auto-detected if None)
        model_name: HuggingFace model name for zero-shot classification
        hierarchical_topic_groups: Optional mapping of high-level category -> leaf topic labels
        routing_settings: Optional routing configuration (stage order, thresholds, fallback behaviour)

    Returns:
        dict containing:
            - news_df: News dataframe with hierarchical classification columns
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
    dtype = torch.float16 if hf_device in (0, "cuda") else torch.float32

    pipeline_kwargs = {
        "model": model_name,
        "batch_size": effective_batch_size,
        "dtype": dtype,
    }

    if hf_device != -1:
        if device_utils.ACCELERATE_AVAILABLE:
            pipeline_kwargs["device"] = hf_device
        else:
            print("⚠️ Hugging Face accelerate is not installed; zero-shot classifier will run on CPU.")
            pipeline_kwargs["dtype"] = torch.float32

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
            pipeline_kwargs["dtype"] = torch.float32
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

    def zero_shot_predict(texts, labels, hyp_template, show_progress=True):
        valid_pairs = [(idx, text) for idx, text in enumerate(texts) if pd.notna(text) and text.strip()]
        predictions: list[dict | None] = [None] * len(texts)

        if not valid_pairs:
            return predictions

        _, filtered_texts = zip(*valid_pairs)
        if show_progress:
            print(f"Processing {len(filtered_texts)} texts with batch_size={effective_batch_size}")

        def run_pipeline_with_dataset(payload: list[str]) -> list[dict]:
            """Use HuggingFace Dataset for efficient GPU batching."""
            try:
                from datasets import Dataset as HFDataset
                from transformers.pipelines.pt_utils import KeyDataset

                # Create a dataset for efficient GPU processing
                dataset = HFDataset.from_dict({"text": payload})

                # Process using the pipeline with KeyDataset for proper iteration
                # KeyDataset allows the pipeline to properly iterate over dataset rows
                outputs = []
                for out in classifier(
                    KeyDataset(dataset, "text"),
                    candidate_labels=labels,
                    hypothesis_template=hyp_template,
                    multi_label=False,
                    batch_size=effective_batch_size,
                ):
                    outputs.append(out)

                return outputs
            except (ImportError, AttributeError):
                # Fallback to standard approach if datasets/KeyDataset not available
                outputs = classifier(
                    payload,
                    candidate_labels=labels,
                    hypothesis_template=hyp_template,
                    multi_label=False,
                )
                if isinstance(outputs, dict):
                    outputs = [outputs]
                return outputs

        try:
            results = run_pipeline_with_dataset(list(filtered_texts))
        except RuntimeError as gpu_exc:
            if "out of memory" in str(gpu_exc).lower():
                print("GPU OOM during classification; processing in smaller chunks...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                    try:
                        torch.mps.empty_cache()
                    except AttributeError:
                        pass

                chunk_size = max(10, len(filtered_texts) // 4 or 1)
                results = []
                for chunk_start in tqdm(
                    range(0, len(filtered_texts), chunk_size),
                    desc="Processing text chunks",
                    disable=not show_progress,
                ):
                    chunk_end = min(chunk_start + chunk_size, len(filtered_texts))
                    chunk_texts = list(filtered_texts[chunk_start:chunk_end])
                    chunk_results = run_pipeline_with_dataset(chunk_texts)
                    results.extend(chunk_results)
            else:
                raise

        for (idx, _), result in zip(valid_pairs, results):
            predictions[idx] = result

        return predictions

    def hierarchical_classify(texts: list[str], show_progress: bool = True) -> dict[str, list]:
        if not hierarchical_topic_groups:
            flat_predictions = zero_shot_predict(
                texts,
                candidate_labels,
                hypothesis_template,
                show_progress=show_progress,
            )
            final_labels: list[str | None] = []
            final_scores: list[float] = []
            for result in flat_predictions:
                if result is None:
                    final_labels.append(None)
                    final_scores.append(0.0)
                else:
                    final_labels.append(result["labels"][0])
                    final_scores.append(float(result["scores"][0]))
            return {
                "final_labels": final_labels,
                "final_scores": final_scores,
                "stage1_labels": final_labels,
                "stage1_scores": final_scores,
            }

        routing_defaults = routing_settings or {}
        stage_order = routing_defaults.get("stage_order", list(hierarchical_topic_groups.keys()))
        stage_order = [label for label in stage_order if label in hierarchical_topic_groups]
        missing_groups = [label for label in hierarchical_topic_groups.keys() if label not in stage_order]
        stage_order.extend(missing_groups)

        thresholds = routing_defaults.get("stage_thresholds", {})
        stage1_threshold: float = float(thresholds.get("stage1", 0.35))
        stage2_threshold: float = float(thresholds.get("stage2", 0.25))
        allow_fallback = bool(routing_defaults.get("allow_fallback_to_other", True))

        fallback_group = None
        if allow_fallback and "Sonstiges" in hierarchical_topic_groups:
            fallback_group = "Sonstiges"

        stage1_predictions = zero_shot_predict(
            texts,
            stage_order,
            hypothesis_template,
            show_progress=show_progress,
        )

        stage1_labels: list[str] = []
        stage1_scores: list[float] = []
        group_assignments: list[str] = []
        for result in stage1_predictions:
            if result is None:
                stage1_label = fallback_group or (stage_order[0] if stage_order else None)
                stage1_score = 0.0
            else:
                stage1_label = result["labels"][0]
                stage1_score = float(result["scores"][0])

            if stage1_label not in hierarchical_topic_groups:
                stage1_label = fallback_group or stage1_label

            if (
                stage1_label in hierarchical_topic_groups
                and stage1_score < stage1_threshold
                and fallback_group
            ):
                stage1_label = fallback_group

            if stage1_label is None:
                stage1_label = fallback_group or stage_order[0]

            stage1_labels.append(stage1_label)
            stage1_scores.append(stage1_score)
            group_assignments.append(stage1_label)

        fallback_leaf = None
        if fallback_group:
            fallback_candidates = hierarchical_topic_groups.get(fallback_group, [])
            fallback_leaf = fallback_candidates[0] if fallback_candidates else None
        else:
            fallback_leaf = None

        final_labels: list[str | None] = [fallback_leaf] * len(texts)
        final_scores: list[float] = [0.0] * len(texts)

        from collections import defaultdict

        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for idx, group in enumerate(group_assignments):
            if group not in hierarchical_topic_groups:
                group = fallback_group or group
            grouped_indices[group].append(idx)

        for group, indices in grouped_indices.items():
            leaf_labels = hierarchical_topic_groups.get(group, candidate_labels)
            if not leaf_labels:
                continue

            if len(leaf_labels) == 1:
                leaf_label = leaf_labels[0]
                for idx in indices:
                    final_labels[idx] = leaf_label
                    final_scores[idx] = stage1_scores[idx]
                continue

            group_texts = [texts[idx] for idx in indices]
            group_predictions = zero_shot_predict(
                group_texts,
                leaf_labels,
                hypothesis_template,
                show_progress=show_progress,
            )

            for local_idx, prediction in enumerate(group_predictions):
                global_idx = indices[local_idx]
                if prediction is None:
                    final_label = fallback_leaf
                    final_score = 0.0
                else:
                    final_label = prediction["labels"][0]
                    final_score = float(prediction["scores"][0])

                if final_score < stage2_threshold and fallback_leaf is not None:
                    final_label = fallback_leaf
                    final_score = max(final_score, stage1_scores[global_idx])

                final_labels[global_idx] = final_label
                final_scores[global_idx] = final_score

        return {
            "final_labels": final_labels,
            "final_scores": final_scores,
            "stage1_labels": stage1_labels,
            "stage1_scores": stage1_scores,
        }

    titles = news_df['title'].tolist()
    title_results = hierarchical_classify(titles, show_progress=True)

    news_df['classification'] = title_results['final_labels']
    news_df['classification_score'] = title_results['final_scores']
    news_df['classification_stage1'] = title_results['stage1_labels']
    news_df['classification_stage1_score'] = title_results['stage1_scores']

    # Re-classify "other" using description
    other_label = candidate_labels[-1]  # "kein Bezug zu Energie, Wetter oder Finanzmärkten"
    other_mask = news_df['classification'] == other_label
    num_other = other_mask.sum()

    if num_other > 0:
        other_indices = news_df[other_mask].index
        descriptions = news_df.loc[other_indices, 'description'].tolist()
        description_results = hierarchical_classify(descriptions, show_progress=True)
        for i, idx in enumerate(other_indices):
            news_df.loc[idx, 'classification'] = description_results['final_labels'][i]
            news_df.loc[idx, 'classification_score'] = description_results['final_scores'][i]
            news_df.loc[idx, 'classification_stage1'] = description_results['stage1_labels'][i]
            news_df.loc[idx, 'classification_stage1_score'] = description_results['stage1_scores'][i]

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

    # Load model with device-specific optimizations
    model = SentenceTransformer(model_name, device=hf_device)

    # GPU-specific optimizations
    if hf_device == "cuda":
        try:
            # Enable fp16 for faster computation and reduced memory
            model = model.half()
            print("Model converted to fp16 (half precision) for CUDA acceleration")

            # Enable cudnn benchmarking for optimized convolutions
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True

            # Increase batch size for GPU to maximize utilization
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 16:
                effective_batch_size = max(effective_batch_size, 256)
            elif gpu_memory_gb >= 8:
                effective_batch_size = max(effective_batch_size, 128)

            print(f"CUDA optimizations enabled: batch_size={effective_batch_size}")
        except Exception as exc:
            print(f"CUDA optimization skipped: {exc}")
    elif hf_device == "mps":
        # MPS-specific optimizations
        print("Using MPS (Apple Silicon) acceleration for embeddings")
        # MPS works better with moderate batch sizes
        effective_batch_size = min(effective_batch_size, 128)

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
        # Main encoding with optimized batch size
        embeddings_array = model.encode(
            list(valid_texts),
            batch_size=effective_batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=False,
            # Enable conversion to tensor for GPU optimization
            convert_to_tensor=False,  # Keep as False to avoid memory issues
        )
    except RuntimeError as gpu_exc:
        # Handle CUDA/MPS out-of-memory errors
        if "out of memory" in str(gpu_exc).lower():
            print(f"GPU OOM error; reducing batch size and retrying...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except AttributeError:
                    pass

            # Retry with smaller batch size
            reduced_batch_size = max(16, effective_batch_size // 2)
            print(f"Retrying with batch_size={reduced_batch_size}")
            embeddings_array = model.encode(
                list(valid_texts),
                batch_size=reduced_batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                normalize_embeddings=False,
            )
        else:
            raise
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
    verbose: bool = True,
    use_gpu: bool = True,
) -> pd.DataFrame:
    """
    Compute time-decayed weighted counts for each topic with GPU acceleration support.

    Weight formula: weight = e^(-lambda * hours_since_publication)

    Args:
        news_df: DataFrame with datetime index (publishedAt) and 'classification' column
        master_df: DataFrame with datetime index representing target timestamps
        lookback_window: Number of hours to look back (default: 336h = 2 weeks)
        decay_lambda: Decay rate parameter (default: 0.05)
        verbose: Whether to print progress messages
        use_gpu: Use GPU acceleration if available (CuPy)

    Returns:
        DataFrame with datetime index and columns for each topic's weighted count
    """
    from config import pipeline_config as cfg

    td_topic_news_df = news_df.copy()
    if not isinstance(td_topic_news_df.index, pd.DatetimeIndex):
        td_topic_news_df.index = pd.to_datetime(td_topic_news_df.index)
    if td_topic_news_df.index.tz is not None:
        td_topic_news_df.index = td_topic_news_df.index.tz_localize(None)

    # Filter out "other" articles (no energy relevance)
    other_label = cfg.OTHER_LABEL
    other_mask = td_topic_news_df['classification'] == other_label
    articles_filtered = other_mask.sum()
    td_topic_news_df = td_topic_news_df[~other_mask]

    if verbose and articles_filtered > 0:
        print(f"Filtered {articles_filtered} 'other' articles (no energy relevance) from time decay aggregation")

    topics = td_topic_news_df['classification'].dropna().unique()
    td_topics_df = pd.DataFrame(index=master_df.index)

    if verbose:
        print(f"Computing time-decayed counts for {len(td_topics_df)} timestamps and {len(topics)} topics")
        print(f"Lookback window: {lookback_window}h, decay lambda: {decay_lambda}")

    timestamps = td_topics_df.index.values
    article_times = td_topic_news_df.index.values
    article_topics = td_topic_news_df['classification'].values

    valid_mask = pd.notna(article_topics)
    article_times_valid = article_times[valid_mask]
    article_topics_valid = article_topics[valid_mask]

    topic_to_idx = {topic: idx for idx, topic in enumerate(topics)}

    # Try GPU acceleration with CuPy
    use_cupy = False
    if use_gpu and torch.cuda.is_available():
        try:
            import cupy as cp
            use_cupy = True
            if verbose:
                print("Using CuPy GPU acceleration for time-decay computation")
        except ImportError:
            if verbose:
                print("CuPy not available; using vectorized NumPy computation")

    if use_cupy:
        # GPU-accelerated version
        import cupy as cp

        # Sort articles by time for efficient binary search
        sort_idx = np.argsort(article_times_valid)
        article_times_sorted = article_times_valid[sort_idx]
        article_topics_sorted = article_topics_valid[sort_idx]

        weighted_counts_array = np.zeros((len(timestamps), len(topics)), dtype=np.float32)

        # Process in batches to avoid GPU memory issues (adaptive batch sizing)
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 16:
                batch_size = min(10000, len(timestamps))
            elif gpu_memory_gb >= 8:
                batch_size = min(5000, len(timestamps))
            else:
                batch_size = min(2000, len(timestamps))
        else:
            batch_size = min(1000, len(timestamps))

        for batch_start in tqdm(range(0, len(timestamps), batch_size),
                               desc="Processing timestamp batches (GPU)",
                               disable=not verbose):
            batch_end = min(batch_start + batch_size, len(timestamps))
            batch_timestamps = timestamps[batch_start:batch_end]

            for i, timestamp in enumerate(batch_timestamps):
                cutoff_time = timestamp - np.timedelta64(int(lookback_window), 'h')
                start_idx = np.searchsorted(article_times_sorted, cutoff_time, side='left')
                end_idx = np.searchsorted(article_times_sorted, timestamp, side='right')

                if start_idx >= end_idx:
                    continue

                window_times = article_times_sorted[start_idx:end_idx]
                window_topics = article_topics_sorted[start_idx:end_idx]

                # Move to GPU
                window_times_gpu = cp.asarray(window_times.astype('datetime64[ns]').astype(np.int64))
                timestamp_gpu = cp.array([timestamp.astype('datetime64[ns]').astype(np.int64)])

                hours_since_gpu = (timestamp_gpu[0] - window_times_gpu) / (3600 * 1e9)  # Convert to hours
                weights_gpu = cp.exp(-decay_lambda * hours_since_gpu)

                # Compute weighted counts per topic
                weights_cpu = cp.asnumpy(weights_gpu)
                for topic in topics:
                    topic_mask = window_topics == topic
                    weighted_counts_array[batch_start + i, topic_to_idx[topic]] = np.sum(weights_cpu[topic_mask])
    else:
        # CPU-optimized vectorized version with binary search
        sort_idx = np.argsort(article_times_valid)
        article_times_sorted = article_times_valid[sort_idx]
        article_topics_sorted = article_topics_valid[sort_idx]

        weighted_counts_array = np.zeros((len(timestamps), len(topics)), dtype=np.float32)

        if verbose:
            print(f"Processing {len(article_times_valid)} valid articles across {len(timestamps)} timestamps")
            print("Using optimized binary search and vectorization")

        for i, timestamp in enumerate(tqdm(timestamps, desc="Processing timestamps", leave=True, disable=not verbose)):
            cutoff_time = timestamp - np.timedelta64(int(lookback_window), 'h')
            start_idx = np.searchsorted(article_times_sorted, cutoff_time, side='left')
            end_idx = np.searchsorted(article_times_sorted, timestamp, side='right')

            if start_idx >= end_idx:
                continue

            window_times = article_times_sorted[start_idx:end_idx]
            window_topics = article_topics_sorted[start_idx:end_idx]
            hours_since = (timestamp - window_times).astype('timedelta64[h]').astype(np.float32)
            weights = np.exp(-decay_lambda * hours_since, dtype=np.float32)

            for topic in topics:
                topic_mask = window_topics == topic
                weighted_counts_array[i, topic_to_idx[topic]] = np.sum(weights[topic_mask])

    for idx, topic in enumerate(topics):
        td_topics_df[topic] = weighted_counts_array[:, idx]

    return td_topics_df


def compute_time_decayed_embeddings(
    news_df: pd.DataFrame,
    master_df: pd.DataFrame,
    lookback_window: int = 336,
    decay_lambda: float = 0.05,
    verbose: bool = True,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Compute time-decayed weighted average embeddings with GPU acceleration support.

    Weight formula: weight = e^(-lambda * hours_since_publication)

    Args:
        news_df: DataFrame with 'publishedAt' as DatetimeIndex and 'embedding' column
        master_df: DataFrame with datetime index representing target timestamps
        lookback_window: Number of hours to look back (default: 336h = 2 weeks)
        decay_lambda: Decay rate parameter (default: 0.05)
        verbose: Whether to print progress messages
        use_gpu: Use GPU acceleration if available (CuPy)

    Returns:
        Array of weighted average embeddings with shape (n_timestamps, embedding_dim)
    """
    from config import pipeline_config as cfg

    if not isinstance(news_df.index, pd.DatetimeIndex):
        raise ValueError("news_df must have a DatetimeIndex")
    if news_df.index.name != 'publishedAt':
        raise ValueError("news_df index must be named 'publishedAt'")

    td_embedding_news_df = news_df.copy()
    if td_embedding_news_df.index.tz is not None:
        td_embedding_news_df.index = td_embedding_news_df.index.tz_localize(None)

    # Filter out "other" articles (no energy relevance)
    other_label = cfg.OTHER_LABEL
    if 'classification' in td_embedding_news_df.columns:
        other_mask = td_embedding_news_df['classification'] == other_label
        articles_filtered = other_mask.sum()
        td_embedding_news_df = td_embedding_news_df[~other_mask]

        if verbose and articles_filtered > 0:
            print(f"Filtered {articles_filtered} 'other' articles (no energy relevance) from embedding aggregation")

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

    # Try GPU acceleration with CuPy
    use_cupy = False
    if use_gpu and torch.cuda.is_available():
        try:
            import cupy as cp
            use_cupy = True
            if verbose:
                print(f"Processing {len(article_embeddings_array)} valid embeddings across {len(timestamps)} timestamps")
                print("Using CuPy GPU acceleration with binary search for embeddings")
        except ImportError:
            if verbose:
                print(f"Processing {len(article_embeddings_array)} valid embeddings across {len(timestamps)} timestamps")
                print("Using CPU binary search for efficient article lookup...")

    if use_cupy:
        # GPU-accelerated version
        import cupy as cp

        # Transfer embeddings to GPU
        article_embeddings_gpu = cp.asarray(article_embeddings_array)

        lookback_delta = np.timedelta64(int(lookback_window), 'h')
        cutoff_times = timestamps - lookback_delta

        # Process in batches to manage GPU memory (adaptive batch sizing for embeddings)
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 16:
                batch_size = min(5000, len(timestamps))
            elif gpu_memory_gb >= 8:
                batch_size = min(2000, len(timestamps))
            else:
                batch_size = min(1000, len(timestamps))
        else:
            batch_size = min(500, len(timestamps))

        for batch_start in tqdm(range(0, len(timestamps), batch_size),
                               desc="Processing timestamp batches (GPU)",
                               disable=not verbose):
            batch_end = min(batch_start + batch_size, len(timestamps))

            for i in range(batch_start, batch_end):
                timestamp = timestamps[i]
                cutoff_time = cutoff_times[i]

                start_idx = np.searchsorted(article_times_valid, cutoff_time, side='left')
                end_idx = np.searchsorted(article_times_valid, timestamp, side='right')

                if start_idx >= end_idx:
                    continue

                window_embeddings_gpu = article_embeddings_gpu[start_idx:end_idx]
                window_times = article_times_valid[start_idx:end_idx]

                hours_since = (timestamp - window_times).astype('timedelta64[h]').astype(np.float32)
                weights_gpu = cp.exp(-decay_lambda * cp.asarray(hours_since))

                total_weight = cp.sum(weights_gpu)
                if float(total_weight) > 1e-10:
                    weighted_sum_gpu = cp.sum(weights_gpu[:, cp.newaxis] * window_embeddings_gpu, axis=0)
                    weighted_embeddings_array[i] = cp.asnumpy(weighted_sum_gpu / total_weight)
                else:
                    weighted_embeddings_array[i] = np.zeros(embedding_dim, dtype=np.float32)

        # Clean up GPU memory
        del article_embeddings_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # CPU-optimized version
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
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Reduce embeddings with GPU-first UMAP and disk caching.

    Args:
        embeddings: Embedding array to reduce
        index: Index for the resulting dataframe
        cache_label: Label for caching
        n_components: Number of components to reduce to
        use_umap: Use UMAP if True, otherwise use PCA
        random_state: Deprecated - kept for backward compatibility but ignored
                     (UMAP runs without random_state to enable parallel processing)

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

                    # Note: random_state removed to enable parallel processing
                    reducer = device_utils.CUML_UMAP(
                        n_components=n_components,
                        n_neighbors=15,
                        min_dist=0.1,
                    )
                    reduced_gpu = reducer.fit_transform(cp.asarray(embeddings))
                    reduced = cp.asnumpy(reduced_gpu)
                    backend = "cuml-umap"
                except Exception as gpu_exc:
                    print(f"cuML UMAP unavailable ({gpu_exc}); reverting to CPU UMAP.")

        if reduced is None:
            # Note: random_state removed to enable n_jobs parallelism
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


def _process_single_parameter_combination(
    lookback: int,
    decay_lambda: float,
    news_df: pd.DataFrame,
    master_df: pd.DataFrame,
    n_components: int,
    use_umap: bool,
    random_state: int,
) -> tuple[tuple[int, float], dict[str, pd.DataFrame]]:
    """
    Process a single (lookback_window, decay_lambda) parameter combination.

    This helper function is designed to be called in parallel by joblib.

    Args:
        lookback: Lookback window in hours
        decay_lambda: Exponential decay parameter
        news_df: Preprocessed news dataframe with classifications and embeddings
        master_df: Master dataframe with target timestamps
        n_components: Number of components for dimensionality reduction
        use_umap: Whether to use UMAP (otherwise PCA)
        random_state: Random seed (kept for backward compatibility)

    Returns:
        Tuple of (params_key, feature_dict) where params_key is (lookback, decay_lambda)
    """
    params_key = (lookback, decay_lambda)

    # Compute time-decayed topic counts
    td_topics = compute_time_decayed_topic_counts(
        news_df=news_df,
        master_df=master_df,
        lookback_window=lookback,
        decay_lambda=decay_lambda,
        verbose=False,
    )

    # Compute time-decayed embeddings
    weighted_embeddings = compute_time_decayed_embeddings(
        news_df=news_df,
        master_df=master_df,
        lookback_window=lookback,
        decay_lambda=decay_lambda,
        verbose=False,
    )

    # Handle NaNs prior to dimensionality reduction
    nan_mask = np.isnan(weighted_embeddings).any(axis=1)
    embeddings_clean = weighted_embeddings.copy()
    embeddings_clean[nan_mask] = 0.0

    # Reduce embeddings
    cache_label = f"td_embeddings_lw{lookback}_dl{decay_lambda}"
    td_embeddings = reduce_embeddings_gpu_first(
        embeddings=embeddings_clean,
        index=master_df.index,
        cache_label=cache_label,
        n_components=n_components,
        use_umap=use_umap,
        random_state=random_state,
    )

    return params_key, {
        "td_topics": td_topics,
        "td_embeddings": td_embeddings,
    }


def precompute_time_decay_feature_sets(
    news_df: pd.DataFrame,
    master_df: pd.DataFrame,
    lookback_windows: list[int],
    decay_lambdas: list[float],
    n_components: int = 20,
    use_umap: bool = True,
    random_state: int = 42,
    device_config: dict | None = None,
    verbose: bool = True,
    n_jobs: int = -1,
) -> dict[tuple[int, float], dict[str, pd.DataFrame]]:
    """
    Precompute time-decayed topic counts and reduced embeddings for a grid of parameters.

    This function now processes parameter combinations in parallel using joblib.

    Args:
        news_df: Preprocessed news dataframe with zero-shot classifications and embeddings.
        master_df: Master dataframe aligned with target timestamps.
        lookback_windows: List of lookback windows (in hours) to evaluate.
        decay_lambdas: List of exponential decay parameters.
        n_components: Number of components for dimensionality reduction.
        use_umap: Whether to use UMAP (otherwise PCA) for reduction.
        random_state: Random seed for reproducibility.
        device_config: Optional device configuration for logging downstream usage.
        verbose: Print progress information.
        n_jobs: Number of parallel jobs (-1 uses all available cores).
                Note: When using GPU acceleration (CuPy/cuML), consider using n_jobs=1
                to avoid GPU contention, or ensure your GPU supports concurrent execution.

    Returns:
        Dictionary keyed by (lookback_window, decay_lambda) tuples containing:
            {
                'td_topics': DataFrame,
                'td_embeddings': DataFrame
            }
    """
    if not isinstance(master_df.index, pd.DatetimeIndex):
        raise ValueError("master_df must have a DatetimeIndex for time-based aggregation.")

    total_combinations = len(lookback_windows) * len(decay_lambdas)

    if verbose:
        print(f"Precomputing time-decayed features for {total_combinations} parameter combinations...")
        print(f"Parallelizing across parameter combinations using joblib (n_jobs={n_jobs})...\n")

    # Create list of all parameter combinations
    param_combinations = [
        (lookback, decay_lambda)
        for lookback in lookback_windows
        for decay_lambda in decay_lambdas
    ]

    # Process combinations in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_process_single_parameter_combination)(
            lookback=lookback,
            decay_lambda=decay_lambda,
            news_df=news_df,
            master_df=master_df,
            n_components=n_components,
            use_umap=use_umap,
            random_state=random_state,
        )
        for lookback, decay_lambda in param_combinations
    )

    # Convert results list to dictionary
    feature_cache: dict[tuple[int, float], dict[str, pd.DataFrame]] = dict(results)

    if verbose:
        print(f"\n✓ Completed precomputation of {len(feature_cache)} parameter combinations")

    return feature_cache


def assemble_time_decay_datasets(
    master_df: pd.DataFrame,
    feature_cache: dict[tuple[int, float], dict[str, pd.DataFrame]],
    target_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    dataset_prefix: str = "dataset",
) -> dict[tuple[int, float], dict[str, pd.DataFrame | list[str] | str]]:
    """
    Merge precomputed time-decay features with the master dataframe and build dataset splits.

    Args:
        master_df: Baseline feature dataframe containing the target column.
        feature_cache: Output of precompute_time_decay_feature_sets.
        target_column: Name of the target column to retain.
        train_ratio: Training set proportion (0-1).
        val_ratio: Validation set proportion (0-1).
        test_ratio: Test set proportion (0-1).
        dataset_prefix: Prefix for generated dataset names.

    Returns:
        Dictionary keyed by parameter tuples with train/val/test splits and metadata.
    """
    ratio_total = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_total, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0 (received {train_ratio}, {val_ratio}, {test_ratio})"
        )

    preprocessed_datasets: dict[tuple[int, float], dict[str, pd.DataFrame | list[str] | str]] = {}

    for idx, (params_key, feature_dict) in enumerate(feature_cache.items(), start=1):
        lookback, decay_lambda = params_key
        dataset_name = f"{dataset_prefix}_lw{lookback}_dl{decay_lambda}"

        td_topics_df = feature_dict["td_topics"]
        td_embeddings_df = feature_dict["td_embeddings"]

        merged_features = master_df.join([td_topics_df, td_embeddings_df], how="left")
        model_df = merged_features.dropna(subset=[target_column]).copy()
        if model_df.empty:
            raise ValueError(
                f"No samples remain after dropping NaNs for params {params_key}. "
                "Check data coverage and target calculation."
            )

        num_samples = len(model_df)
        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)

        if train_end == 0 or val_end == train_end or val_end >= num_samples:
            raise ValueError(
                f"Invalid split sizes for dataset {dataset_name} "
                f"(train_end={train_end}, val_end={val_end}, total={num_samples})."
            )

        train_df = model_df.iloc[:train_end].copy()
        val_df = model_df.iloc[train_end:val_end].copy()
        test_df = model_df.iloc[val_end:].copy()

        topic_features = td_topics_df.columns.tolist()
        embedding_features = td_embeddings_df.columns.tolist()
        news_features = topic_features + embedding_features

        preprocessed_datasets[params_key] = {
            "dataset_name": dataset_name,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "news_features": news_features,
            "topic_features": topic_features,
            "embedding_features": embedding_features,
        }

    return preprocessed_datasets


def scale_preprocessed_datasets(
    preprocessed_datasets: dict[tuple[int, float], dict[str, pd.DataFrame | list[str] | str]],
    scaler_factory=StandardScaler,
    suffix: str = "_scaled",
) -> dict[tuple[int, float], dict[str, pd.DataFrame | list[str] | str]]:
    """
    Standardise news-derived features for each dataset split and record scaler metadata.

    Args:
        preprocessed_datasets: Datasets produced by assemble_time_decay_datasets.
        scaler_factory: Callable returning a fitted scaler (defaults to sklearn StandardScaler).
        suffix: Suffix appended to scaled news feature names.

    Returns:
        Updated preprocessed_datasets with scaled feature columns and scaler references.
    """
    for params_key, data_dict in preprocessed_datasets.items():
        news_features = data_dict.get("news_features", [])
        if not news_features:
            data_dict["scaled_news_features"] = []
            data_dict["scaler_news"] = None
            continue

        scaler = scaler_factory()
        train_df = data_dict["train_df"].copy()
        val_df = data_dict["val_df"].copy()
        test_df = data_dict["test_df"].copy()

        train_values = train_df[news_features].fillna(0.0).to_numpy(dtype=np.float32, copy=True)
        scaler.fit(train_values)

        scaled_feature_names = [f"{feature}{suffix}" for feature in news_features]

        def _apply_scaler(frame: pd.DataFrame) -> pd.DataFrame:
            values = frame[news_features].fillna(0.0).to_numpy(dtype=np.float32, copy=True)
            transformed = scaler.transform(values)
            transformed_df = pd.DataFrame(
                transformed,
                index=frame.index,
                columns=scaled_feature_names,
            )
            updated = frame.copy()
            # Drop existing scaled columns if function is invoked multiple times
            for col in scaled_feature_names:
                if col in updated.columns:
                    updated = updated.drop(columns=[col])
            updated[scaled_feature_names] = transformed_df
            return updated

        data_dict["train_df"] = _apply_scaler(train_df)
        data_dict["val_df"] = _apply_scaler(val_df)
        data_dict["test_df"] = _apply_scaler(test_df)
        data_dict["scaled_news_features"] = scaled_feature_names
        data_dict["scaler_news"] = scaler

    return preprocessed_datasets


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
    ridge_cv = RidgeClassifierCV(alphas=alphas, cv=tscv, scoring='f1_macro')
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

    top_results = results_sorted[:5]

    print(f"{'='*80}")
    print("TOP 5 PARAMETER COMBINATIONS:")
    print(f"{'='*80}")
    for idx, result in enumerate(top_results, 1):
        print(
            f"{idx}. dataset={result['dataset_name']} | lookback={result['lookback_window']}h | "
            f"lambda={result['decay_lambda']} | alpha={result['best_alpha']:.4f} | "
            f"Val Accuracy={result['val_accuracy']:.3f} | Val Macro-F1={result['val_macro_f1']:.3f}"
        )

    return top_results
