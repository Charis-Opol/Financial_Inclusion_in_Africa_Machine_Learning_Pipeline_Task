"""Service configuration, read once from environment variables at startup."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceConfig:
    model_dir: Path
    # XGBoost defaults to one thread per core. For single-row requests the
    # thread fan-out costs more than it saves, and inside a CPU-limited
    # container it oversubscribes the quota -- concurrency should come from
    # parallel requests, not from threads inside one prediction.
    inference_threads: int
    log_level: str
    prediction_log_dir: Path
    # ~10k events x ~1 KB = ~10 MB worst case held in memory while the disk stalls.
    prediction_log_queue_size: int = 10_000

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        return cls(
            model_dir=Path(os.environ.get("MODEL_DIR", "models/production")),
            inference_threads=int(os.environ.get("INFERENCE_THREADS", "1")),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            prediction_log_dir=Path(os.environ.get("PREDICTION_LOG_DIR", "logs")),
            prediction_log_queue_size=int(os.environ.get("PREDICTION_LOG_QUEUE_SIZE", "10000")),
        )
