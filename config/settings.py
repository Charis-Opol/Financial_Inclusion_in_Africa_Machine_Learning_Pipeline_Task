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
class Settings:
    seed: int
    paths: PathsConfig
    data: DataConfig
    cv: CVConfig


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
    return Settings(seed=raw["seed"], paths=paths, data=data, cv=cv)
