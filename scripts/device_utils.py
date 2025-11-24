"""
Device Utilities
================
GPU/CPU detection and device management utilities.
"""

import torch
import os


def check_accelerate_available() -> bool:
    """Return True when Hugging Face accelerate is importable."""
    try:
        import accelerate  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_cuml_umap() -> tuple[bool, object | None]:
    """Attempt to import cuML's GPU UMAP implementation."""
    try:
        from cuml.manifold import UMAP as CUML_UMAP  # type: ignore
    except ImportError:
        return False, None
    return True, CUML_UMAP


ACCELERATE_AVAILABLE = check_accelerate_available()
HAS_CUML_UMAP, CUML_UMAP = resolve_cuml_umap()


def detect_compute_device(task='general', verbose=True):
    """
    Detect optimal compute device and recommended batch size.

    Args:
        task: Type of task ('general', 'embeddings', 'training')
        verbose: Print device information

    Returns:
        dict with device info and optimal batch size
    """
    cpu_cores = os.cpu_count() or 8
    default_jobs = max(1, cpu_cores - 2)
    device_info = {
        'device': 'cpu',
        'backend': 'cpu',
        'gpu_name': None,
        'gpu_memory_gb': 0,
        'optimal_batch_size': 32,
        'tree_method': 'hist',
        'xgb_device': 'cpu',
        'n_jobs': default_jobs,
        'description': f'CPU-only ({cpu_cores} cores)',
        'lgbm_device': 'cpu',
    }

    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_count = torch.cuda.device_count()

        device_info.update({
            'device': 'cuda',
            'backend': 'cuda',
            'gpu_name': gpu_name,
            'gpu_memory_gb': gpu_memory,
            'description': f'CUDA ({gpu_name})',
            'tree_method': 'hist',
            'xgb_device': 'cuda',
            'n_jobs': max(1, default_jobs // gpu_count),
            'lgbm_device': 'gpu',
        })

        # Adjust batch size based on GPU memory
        if gpu_memory >= 24:
            device_info['optimal_batch_size'] = 512
        elif gpu_memory >= 16:
            device_info['optimal_batch_size'] = 384
        elif gpu_memory >= 8:
            device_info['optimal_batch_size'] = 256
        else:
            device_info['optimal_batch_size'] = 128

        if verbose:
            print(f"✓ CUDA detected: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"  Optimal batch size: {device_info['optimal_batch_size']}")
            print(f"  XGBoost tree method: {device_info['tree_method']} • device: {device_info['xgb_device']}")
            print(f"  Parallel jobs: {device_info['n_jobs']}")

    # Check for MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        device_info.update({
            'device': 'mps',
            'backend': 'mps',
            'gpu_name': 'Apple Silicon GPU',
            'optimal_batch_size': 128,
            'description': 'Apple MPS GPU',
            'tree_method': 'hist',
            'xgb_device': 'cpu',
            'n_jobs': default_jobs,
            'lgbm_device': 'cpu',
        })

        if verbose:
            print("✓ MPS detected: Apple Silicon GPU")
            print(f"  Optimal batch size: {device_info['optimal_batch_size']}")
            print("  XGBoost will fallback to CPU histogram (MPS unsupported)")

    else:
        if verbose:
            print("⚠ No GPU detected, using CPU")
            print(f"  Optimal batch size: {device_info['optimal_batch_size']}")
            print(f"  Parallel jobs: {device_info['n_jobs']}")

    return device_info


def ensure_tensor_device(obj, device):
    """Move torch tensors (or collections thereof) onto the specified device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(ensure_tensor_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: ensure_tensor_device(v, device) for k, v in obj.items()}
    return obj


def resolve_hf_device(primary_device: str):
    """Resolve device for HuggingFace transformers."""
    if primary_device == "cuda" and torch.cuda.is_available():
        return 0
    if primary_device == "mps" and torch.backends.mps.is_available():
        return "mps"
    return -1
