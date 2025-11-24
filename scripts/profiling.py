"""
Profiling Utilities
===================
Resource monitoring and performance profiling utilities.
"""

import time
from contextlib import AbstractContextManager

# Optional dependencies
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    import psutil
except ImportError:
    psutil = None


def _read_gpu_state():
    """Read current GPU utilization and memory state."""
    if not _NVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "mem_used_gb": memory.used / (1024 ** 3),
            "mem_total_gb": memory.total / (1024 ** 3),
        }
    except Exception:
        return None


def _read_cpu_state():
    """Read current CPU and disk I/O state."""
    if psutil is None:
        return None
    cpu_times = psutil.cpu_times_percent(interval=None)
    disk_io = psutil.disk_io_counters() if hasattr(psutil, "disk_io_counters") else None
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "iowait_percent": getattr(cpu_times, "iowait", 0.0),
        "disk_read_mb": (disk_io.read_bytes / (1024 ** 2)) if disk_io else None,
        "disk_write_mb": (disk_io.write_bytes / (1024 ** 2)) if disk_io else None,
    }


class StageProfiler(AbstractContextManager):
    """
    Context manager to capture per-stage resource consumption.

    Usage:
        with StageProfiler("Data Loading", device_config) as profiler:
            # Your code here
            pass
    """

    def __init__(self, stage_name: str, device_config: dict | None = None):
        self.stage_name = stage_name
        self.device_config = device_config or {}
        self._start_time = None
        self._start_cpu = None
        self._start_gpu = None

    def __enter__(self):
        print(f"\n[Stage ⏳] {self.stage_name} — starting")
        self._start_time = time.time()
        self._start_cpu = _read_cpu_state()
        self._start_gpu = _read_gpu_state()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_cpu = _read_cpu_state()
        end_gpu = _read_gpu_state()
        duration = end_time - (self._start_time or end_time)

        metrics: dict[str, float | int | None] = {
            "duration_sec": duration,
        }

        print(f"[Stage ✅] {self.stage_name} — completed in {duration:.2f}s")

        if end_cpu:
            cpu_percent = end_cpu.get("cpu_percent")
            iowait_percent = end_cpu.get("iowait_percent")
            metrics.update({
                "cpu_percent": cpu_percent,
                "cpu_iowait_percent": iowait_percent,
            })
            if cpu_percent is not None:
                print(f"  CPU usage: {cpu_percent:.1f}% • IO wait: {iowait_percent:.2f}%")
            if self._start_cpu and end_cpu:
                read_delta = end_cpu.get("disk_read_mb")
                write_delta = end_cpu.get("disk_write_mb")
                start_read = self._start_cpu.get("disk_read_mb", 0.0) if self._start_cpu else 0.0
                start_write = self._start_cpu.get("disk_write_mb", 0.0) if self._start_cpu else 0.0
                if read_delta is not None and write_delta is not None:
                    read_mb = max(0.0, read_delta - start_read)
                    write_mb = max(0.0, write_delta - start_write)
                    metrics.update({
                        "disk_read_mb": read_mb,
                        "disk_write_mb": write_mb,
                    })
                    print(f"  Disk Δ: +{read_mb:.1f} MB read, +{write_mb:.1f} MB written")

        if end_gpu:
            metrics.update(
                {
                    "gpu_util_percent": end_gpu.get("gpu_util"),
                    "gpu_mem_util_percent": end_gpu.get("mem_util"),
                    "gpu_mem_used_gb": end_gpu.get("mem_used_gb"),
                    "gpu_mem_total_gb": end_gpu.get("mem_total_gb"),
                }
            )
            print(
                "  GPU util: {gpu_util:.0f}% • Mem util: {mem_util:.0f}% "
                "({mem_used_gb:.2f}/{mem_total_gb:.2f} GB)".format(**end_gpu)
            )
        elif self.device_config.get("device") in {"cuda", "mps"}:
            print("  GPU metrics unavailable (install pynvml for CUDA visibility)")

        return False  # propagate exceptions
