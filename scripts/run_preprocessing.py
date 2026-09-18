"""Phase 2 orchestration: raw CSVs -> interim (cleaned) -> processed (encoded) checkpoints.

Encoders are fit on train only (Phase 2.5) and reused to transform test, so
this script is the concrete enforcement point for that leakage-safety rule
-- `fit` is called exactly once per pipeline, on `train`.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from settings import load_settings  # noqa: E402

from fin_inclusion.data.loader import DataLoader  # noqa: E402
from fin_inclusion.preprocessing.cleaning import (  # noqa: E402
    HouseholdSizeOutlierFlagger,
    MissingnessRecoder,
)
from fin_inclusion.preprocessing.encoders import EmbeddingIndexBranch  # noqa: E402
from fin_inclusion.preprocessing.pipeline_factory import (  # noqa: E402
    build_pytorch_pipeline,
    build_xgboost_pipeline,
)
from sklearn.pipeline import Pipeline  # noqa: E402


def _build_cleaning_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("missingness", MissingnessRecoder()),
            ("outlier_flag", HouseholdSizeOutlierFlagger()),
        ]
    )


def main() -> None:
    settings = load_settings()
    settings.paths.interim_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.processed_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        raw_train_path=settings.paths.raw_train, raw_test_path=settings.paths.raw_test
    )
    train = loader.load_train()
    test = loader.load_test()
    print(f"Loaded raw: train={train.shape}, test={test.shape}")

    cleaning = _build_cleaning_pipeline().fit(train)
    train_cleaned = cleaning.transform(train)
    test_cleaned = cleaning.transform(test)
    train_cleaned.to_csv(settings.paths.interim_dir / "train_cleaned.csv", index=False)
    test_cleaned.to_csv(settings.paths.interim_dir / "test_cleaned.csv", index=False)
    print(f"Wrote interim checkpoints to {settings.paths.interim_dir}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        xgb_pipeline = build_xgboost_pipeline().fit(train)
        train_xgb = xgb_pipeline.transform(train)
        test_xgb = xgb_pipeline.transform(test)

        pytorch_pipeline = build_pytorch_pipeline().fit(train)
        train_pt = pytorch_pipeline.transform(train)
        test_pt = pytorch_pipeline.transform(test)

    for w in caught:
        print(f"WARNING: {w.message}")

    train_xgb.to_csv(settings.paths.processed_dir / "train_xgboost.csv", index=False)
    test_xgb.to_csv(settings.paths.processed_dir / "test_xgboost.csv", index=False)
    train_pt.to_csv(settings.paths.processed_dir / "train_pytorch.csv", index=False)
    test_pt.to_csv(settings.paths.processed_dir / "test_pytorch.csv", index=False)

    embedding_branch: EmbeddingIndexBranch = pytorch_pipeline.named_steps["encode"]
    vocab_path = settings.paths.processed_dir / "pytorch_vocab_sizes.json"
    vocab_path.write_text(json.dumps(embedding_branch.vocab_sizes_, indent=2), encoding="utf-8")

    print(f"Wrote processed checkpoints to {settings.paths.processed_dir}")
    print(f"XGBoost branch: train={train_xgb.shape}, test={test_xgb.shape}")
    print(f"PyTorch branch: train={train_pt.shape}, test={test_pt.shape}")
    print(f"PyTorch embedding vocab sizes: {embedding_branch.vocab_sizes_}")


if __name__ == "__main__":
    main()
