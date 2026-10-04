"""Typed config loader for `config/config.yaml`."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    raw_train: Path
    raw_test: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class DataConfig:
    id_columns: list[str]
    target_column: str


@dataclass(frozen=True)
class CVConfig:
    n_outer_folds: int
    n_inner_folds: int
    stratify_columns: list[str]


@dataclass(frozen=True)
class XAIConfig:
    background_size: int
    explanation_sample_size: int
    oracle_instances: int
    ig_n_steps: int
    bootstrap_reps: int
    model_dir: Path
    artifacts_dir: Path
    figures_dir: Path
    decisions_log: Path


@dataclass(frozen=True)
class Settings:
    seed: int
    paths: PathsConfig
    data: DataConfig
    cv: CVConfig
    xai: XAIConfig


def _resolve(relative: str) -> Path:
    return _PROJECT_ROOT / relative


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate `config.yaml` into a typed `Settings` object.

    Paths in the YAML are relative to the project root and are resolved
    to absolute `Path`s here.
    """
    config_path = config_path or _DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths = PathsConfig(
        raw_train=_resolve(raw["paths"]["raw_train"]),
        raw_test=_resolve(raw["paths"]["raw_test"]),
        interim_dir=_resolve(raw["paths"]["interim_dir"]),
        processed_dir=_resolve(raw["paths"]["processed_dir"]),
        reports_dir=_resolve(raw["paths"]["reports_dir"]),
        figures_dir=_resolve(raw["paths"]["figures_dir"]),
    )
    data = DataConfig(
        id_columns=list(raw["data"]["id_columns"]),
        target_column=raw["data"]["target_column"],
    )
    cv = CVConfig(
        n_outer_folds=raw["cv"]["n_outer_folds"],
        n_inner_folds=raw["cv"]["n_inner_folds"],
        stratify_columns=list(raw["cv"]["stratify_columns"]),
    )
    xai = XAIConfig(
        background_size=_positive_int(raw["xai"], "background_size"),
        explanation_sample_size=_positive_int(raw["xai"], "explanation_sample_size"),
        oracle_instances=_positive_int(raw["xai"], "oracle_instances"),
        ig_n_steps=_positive_int(raw["xai"], "ig_n_steps"),
        bootstrap_reps=_positive_int(raw["xai"], "bootstrap_reps"),
        model_dir=_resolve(raw["xai"]["model_dir"]),
        artifacts_dir=_resolve(raw["xai"]["artifacts_dir"]),
        figures_dir=_resolve(raw["xai"]["figures_dir"]),
        decisions_log=_resolve(raw["xai"]["decisions_log"]),
    )
    return Settings(seed=raw["seed"], paths=paths, data=data, cv=cv, xai=xai)


def _positive_int(section: dict, key: str) -> int:
    value = section[key]
    # bool is an int subclass; `background_size: true` must not pass as 1.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"xai.{key} must be a positive integer, got {value!r}")
    return value
