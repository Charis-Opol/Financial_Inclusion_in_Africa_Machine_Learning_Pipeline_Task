"""Saves one run's config, seed, git commit and metrics to week4/results/runs/ (rule 6)."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

WEEK4_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WEEK4_DIR.parent
RESULTS_DIR = WEEK4_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"


def _git_commit() -> str:
    """Current commit, marked `-dirty` if week4/ code differs from it."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--", "week4/src"], cwd=REPO_ROOT, text=True)
    return f"{commit}-dirty" if status.strip() else commit


def save_run(name: str, config: dict, metrics: dict, seed: int | None = None) -> Path:
    """Writes `<name>.json` and returns its path. Rerunning a named run overwrites it."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}.json"
    record = {
        "name": name,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "seed": seed,
        "config": config,
        "metrics": metrics,
    }
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path
