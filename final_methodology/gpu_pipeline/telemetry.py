"""
Lightweight telemetry utilities to persist stage-level resource metrics and guard rails.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def ensure_disk_headroom(path: Path, min_free_gb: float = 5.0) -> None:
    """
    Raise an exception if the filesystem containing ``path`` has less than the requested free space.
    """
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient disk space on {path.anchor}: {free_gb:.2f} GB free "
            f"(requires >= {min_free_gb} GB). Consider cleaning artefacts or adjusting cache limits."
        )


@dataclass
class TelemetryLogger:
    """
    Structured telemetry sink that accumulates stage metrics and flushes them to disk.
    """

    log_dir: Path = field(default_factory=lambda: Path("logs/telemetry").resolve())
    run_id: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
    metadata: Dict[str, str] | None = None
    stages: list[dict] = field(default_factory=list)
    created_ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.metadata is None:
            self.metadata = {
                "host": socket.gethostname(),
                "pid": str(os.getpid()),
                "started_at": datetime.utcnow().isoformat() + "Z",
            }

    def log_stage(self, stage_name: str, metrics: dict) -> None:
        entry = {
            "stage": stage_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **metrics,
        }
        self.stages.append(entry)

    def flush(self) -> Path:
        payload = {
            "run_id": self.run_id,
            "metadata": self.metadata,
            "stages": self.stages,
            "duration_sec": time.time() - self.created_ts,
        }
        output_path = self.log_dir / f"{self.run_id}.json"
        with output_path.open("w") as fp:
            json.dump(payload, fp, indent=2)
        return output_path


