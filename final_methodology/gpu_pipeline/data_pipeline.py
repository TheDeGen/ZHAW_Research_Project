"""
GPU-aware streaming data pipeline utilities.

This module provides composable helpers to ingest the German energy and news datasets
without loading them fully into memory. CSV inputs are normalised into parquet-backed,
memory-mapped artefacts that can be streamed in configurable batch sizes. Optional
remote cache synchronisation (e.g. S3/GCS/HTTP) is supported via fsspec-compatible URIs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator, Literal, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq

try:
    import fsspec
except ImportError:  # pragma: no cover - optional dependency
    fsspec = None  # type: ignore[assignment]


DEFAULT_ARTIFACT_DIR = Path(".cache/data_pipeline").resolve()


def _hash_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(slots=True)
class DataPipelineConfig:
    energy_source: Path
    news_source: Path
    artifact_dir: Path = field(default_factory=lambda: DEFAULT_ARTIFACT_DIR)
    cache_max_bytes: int = 64 * 1024**3  # 64 GB guard rail
    batch_hours: int = 24  # stream one day of hourly data by default
    remote_cache_uri: Optional[str] = None
    remote_cache_mode: Literal["read", "write", "sync"] = "sync"

    def __post_init__(self) -> None:
        self.energy_source = self.energy_source.expanduser().resolve()
        self.news_source = self.news_source.expanduser().resolve()
        self.artifact_dir = _ensure_dir(self.artifact_dir)

    @property
    def energy_cache(self) -> Path:
        return self.artifact_dir / "energy.parquet"

    @property
    def news_cache(self) -> Path:
        return self.artifact_dir / "news.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.artifact_dir / "manifest.json"


class StreamingDataPipeline:
    """
    Orchestrates streamed ingestion, memory-mapped caching and remote synchronisation.

    Example:
        config = DataPipelineConfig(
            energy_source=Path(\"final_methodology/energy_data.csv\"),
            news_source=Path(\"final_methodology/german_news_v1.csv\"),
            remote_cache_uri=\"s3://bucket/arep-cache\"
        )
        pipeline = StreamingDataPipeline(config)
        pipeline.prepare_artifacts()
        for batch_df in pipeline.iter_energy(batches=7):
            ...
    """

    def __init__(self, config: DataPipelineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Artifact preparation
    # ------------------------------------------------------------------
    def prepare_artifacts(self, force: bool = False) -> None:
        """
        Convert CSV inputs into parquet datasets (memory-mapped) if they changed.
        """
        manifest = self._load_manifest()

        for source, cache_name in [
            (self.config.energy_source, self.config.energy_cache),
            (self.config.news_source, self.config.news_cache),
        ]:
            checksum = _hash_file(source)
            cache_key = str(source)

            needs_refresh = (
                force
                or cache_key not in manifest
                or manifest[cache_key].get("checksum") != checksum
                or not cache_name.exists()
            )

            if not needs_refresh:
                continue

            cache_name.parent.mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(tempfile.mkdtemp(prefix="pipeline_", dir=str(self.config.artifact_dir)))
            try:
                self._write_parquet_dataset(source, tmp_dir)
                # replace atomically
                if cache_name.exists():
                    if cache_name.is_dir():
                        shutil.rmtree(cache_name)
                    else:
                        cache_name.unlink()
                shutil.move(str(tmp_dir), str(cache_name))
            finally:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            manifest[cache_key] = {
                "checksum": checksum,
                "cache_path": str(cache_name),
                "rows": self._count_rows(cache_name),
            }

        self._save_manifest(manifest)
        self.cleanup_local_cache()
        if self.config.remote_cache_uri and fsspec is not None:
            self._sync_remote_cache(manifest)

    # ------------------------------------------------------------------
    # Iterators
    # ------------------------------------------------------------------
    def iter_energy(self, batches: int | None = None) -> Iterator[pd.DataFrame]:
        """
        Stream the energy dataset as pandas DataFrames.
        """
        dataset = self._ensure_dataset(self.config.energy_cache, self.config.energy_source, "energy")
        yield from self._stream_dataset(dataset, row_batch=self.config.batch_hours, max_batches=batches)

    def iter_news(self, batches: int | None = None) -> Iterator[pd.DataFrame]:
        """
        Stream the news dataset as pandas DataFrames.
        """
        dataset = self._ensure_dataset(self.config.news_cache, self.config.news_source, "news")
        yield from self._stream_dataset(dataset, row_batch=512, max_batches=batches)  # news not hourly indexed

    @contextmanager
    def arrow_scanner(self, which: Literal["energy", "news"]) -> Generator[pa_dataset.Scanner, None, None]:
        """
        Context manager yielding a PyArrow scanner for advanced workflows (e.g. GPU DLPack ingest).
        """
        cache_path = self.config.energy_cache if which == "energy" else self.config.news_cache
        source_path = self.config.energy_source if which == "energy" else self.config.news_source
        dataset = self._ensure_dataset(cache_path, source_path, which)
        scanner = dataset.scanner(batch_size=self.config.batch_hours if which == "energy" else 512)
        try:
            yield scanner
        finally:
            del scanner

    # ------------------------------------------------------------------
    # Local cache housekeeping
    # ------------------------------------------------------------------
    def cleanup_local_cache(self) -> None:
        """
        Remove the oldest cache artefacts when the on-disk footprint exceeds the quota.
        """
        quota = self.config.cache_max_bytes
        current_size = self._dir_size(self.config.artifact_dir)
        if current_size <= quota:
            return

        entries = sorted(self.config.artifact_dir.glob("*"), key=lambda p: p.stat().st_mtime)
        while entries and current_size > quota:
            victim = entries.pop(0)
            size = self._path_size(victim)
            if victim.is_dir():
                shutil.rmtree(victim, ignore_errors=True)
            else:
                victim.unlink(missing_ok=True)
            current_size -= size
            print(f"⛏️  Removed cache artefact {victim.name} to recover {size/1024**3:.2f} GB")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_dataset(self, cache_path: Path, source_path: Path, label: str) -> pa_dataset.Dataset:
        if not cache_path.exists():
            print(f"⚠️  Cache for {label} missing; regenerating.")
            self.prepare_artifacts(force=True)
        return pa_dataset.dataset(cache_path, format="parquet")

    def _write_parquet_dataset(self, csv_path: Path, destination: Path) -> None:
        """
        Stream CSV into a partition-less parquet dataset to enable memory mapping.
        """
        read_options = pa_csv.ReadOptions(
            use_threads=True,
            block_size=32 * 1024**2,
        )
        convert_options = pa_csv.ConvertOptions(
            timestamp_parsers=["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"],
        )
        table_reader = pa_csv.open_csv(
            str(csv_path),
            read_options=read_options,
            convert_options=convert_options,
        )

        table = table_reader.read_all()

        pq.write_to_dataset(
            table,
            root_path=str(destination),
            basename_template="part-{i}.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )

    def _stream_dataset(
        self,
        dataset: pa_dataset.Dataset,
        row_batch: int,
        max_batches: Optional[int] = None,
    ) -> Iterator[pd.DataFrame]:
        scanner = dataset.scanner(batch_size=row_batch)
        for idx, record_batch in enumerate(scanner.to_batches()):
            df = record_batch.to_pandas(types_mapper=pd.ArrowDtype)
            yield df
            if max_batches is not None and idx + 1 >= max_batches:
                break

    def _sync_remote_cache(self, manifest: dict[str, dict[str, str]]) -> None:
        if fsspec is None or not self.config.remote_cache_uri:
            return

        fs, remote_path = fsspec.core.url_to_fs(self.config.remote_cache_uri)
        remote_manifest_path = f"{remote_path.rstrip('/')}/manifest.json"

        if self.config.remote_cache_mode in {"write", "sync"}:
            for entry in (self.config.energy_cache, self.config.news_cache):
                if entry.exists():
                    fs.put(str(entry), f"{remote_path.rstrip('/')}/{entry.name}", recursive=True)

            with fs.open(remote_manifest_path, "w") as remote_manifest:
                json.dump(manifest, remote_manifest, indent=2)

        if self.config.remote_cache_mode in {"read", "sync"}:
            if fs.exists(remote_manifest_path):
                fs.get(remote_manifest_path, str(self.config.manifest_path))

    def _load_manifest(self) -> dict[str, dict[str, str]]:
        if not self.config.manifest_path.exists():
            return {}
        try:
            return json.loads(self.config.manifest_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_manifest(self, manifest: dict[str, dict[str, str]]) -> None:
        with self.config.manifest_path.open("w") as fp:
            json.dump(manifest, fp, indent=2)

    @staticmethod
    def _count_rows(dataset_path: Path) -> int:
        dataset = pa_dataset.dataset(dataset_path, format="parquet")
        return sum(fragment.count_rows() for fragment in dataset.get_fragments())

    @staticmethod
    def _dir_size(path: Path) -> int:
        return sum(StreamingDataPipeline._path_size(child) for child in path.glob("**/*"))

    @staticmethod
    def _path_size(path: Path) -> int:
        try:
            return path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.glob("**/*"))
        except FileNotFoundError:
            return 0


