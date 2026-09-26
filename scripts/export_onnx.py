"""Exports the production XGBoost model to ONNX and proves it matches native scoring.

    python scripts/export_onnx.py          (serving venv + requirements-onnx.txt)

Writes `models/production/model.onnx` and `onnx_export_report.json` ONLY if
every check below passes; otherwise it exits non-zero and writes nothing,
because an ONNX file that scores differently from the model we validated is
worse than no ONNX file.

Checks:
  1. `onnx.checker` full structural/type check.
  2. Dynamic batch axis really is dynamic: batch sizes 1, 7 and 1000 all run.
  3. Numerical parity with native XGBoost on every row of the raw train and
     test CSVs (~33.6k real respondents) plus synthetic edge cases (every
     category level, numeric range boundaries): max |Δp| <= ATOL.
  4. Decision parity: no row changes Yes/No at the production threshold,
     except rows sitting within ATOL of the threshold (reported, not hidden).

Conversion notes (found by trial, not assumed):
  - onnxmltools only accepts features named f0..fN, so the names are stripped
    from a copy of the booster. That is safe: column order is fixed by the
    serving encoder and verified at model load, and the real names are
    stored in the ONNX metadata_props for consumers.
  - The converter emits a `label` output hard-wired to a 0.5 cut-off, which
    contradicts the production threshold (~0.455). It is removed from the
    graph outputs so no consumer can pick up the wrong decision rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxmltools
import onnxruntime as ort
from onnxmltools.convert.common.data_types import FloatTensorType

from fin_inclusion.serving.predictor import Predictor
from fin_inclusion.serving.raw_records import read_raw_records

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_NAME = "features"
PROBABILITY_OUTPUT = "probabilities"
# Highest opset onnxmltools' XGBoost converter supports (1.16.0).
TARGET_OPSET = 15
# Both runtimes evaluate the same float32 trees; differences come only from
# the summation order of ~141 leaf values (float32 eps ~6e-8). 1e-6 on a
# probability is ~10x that, and far below anything that matters for a decision.
ATOL = 1e-6


def convert(predictor: Predictor) -> onnx.ModelProto:
    anonymous = predictor.booster.copy()
    anonymous.feature_names = None
    anonymous.feature_types = None
    n_features = predictor.encoder.n_features

    model = onnxmltools.convert_xgboost(
        anonymous,
        initial_types=[(INPUT_NAME, FloatTensorType([None, n_features]))],  # None = dynamic batch
        target_opset=TARGET_OPSET,
    )

    kept = [o for o in model.graph.output if o.name == PROBABILITY_OUTPUT]
    if len(kept) != 1:
        raise RuntimeError(f"Expected one {PROBABILITY_OUTPUT!r} output, got {[o.name for o in model.graph.output]}")
    del model.graph.output[:]
    model.graph.output.extend(kept)

    model.doc_string = (
        "P(bank_account) model. Input: float32 [batch, n_features] in feature_columns order. "
        f"Output: {PROBABILITY_OUTPUT}[:, 1] = score for 'Yes'; compare with decision_threshold."
    )
    metadata = predictor.metadata
    onnx.helper.set_model_props(model, {
        "model_version": metadata["model_version"],
        "source_model_sha256": metadata["model_sha256"],
        "decision_threshold": str(metadata["decision_threshold"]),
        "feature_columns": json.dumps(metadata["feature_columns"]),
    })
    onnx.checker.check_model(model, full_check=True)
    return model


def edge_case_records(metadata: dict) -> list[dict]:
    """One-field-at-a-time variations covering every category level and numeric bound."""
    levels = metadata["raw_categorical_levels"]
    base = {col: values[0] for col, values in levels.items()} | {"household_size": 4, "age_of_respondent": 35}
    records = [base | {col: value} for col, values in levels.items() for value in values]
    records += [base | {"household_size": n} for n in (1, 15, 16, 21, 30)]
    records += [base | {"age_of_respondent": a} for a in (16, 100)]
    return records


def build_verification_matrix(predictor: Predictor, csv_paths: list[Path]) -> np.ndarray:
    metadata = predictor.metadata
    records = edge_case_records(metadata)
    for path in csv_paths:
        records += read_raw_records(path, metadata["raw_categorical_levels"], metadata["numeric_columns"])
    return predictor.encoder.encode_batch(records)


def onnx_scores(session: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    return session.run([PROBABILITY_OUTPUT], {INPUT_NAME: X})[0][:, 1]


def verify(session: ort.InferenceSession, predictor: Predictor, X: np.ndarray) -> dict:
    """Raises AssertionError on any parity failure; returns the evidence otherwise."""
    for batch_size in (1, 7, 1000):
        out = onnx_scores(session, X[:batch_size])
        assert out.shape == (batch_size,), f"batch {batch_size}: got output shape {out.shape}"

    native = predictor.booster.inplace_predict(X, validate_features=False)
    exported = onnx_scores(session, X)
    abs_diff = np.abs(native - exported)
    worst = int(abs_diff.argmax())
    assert np.isfinite(exported).all(), "ONNX produced non-finite scores"
    assert abs_diff[worst] <= ATOL, (
        f"max |native - onnx| = {abs_diff[worst]:.3g} at row {worst} "
        f"(native {native[worst]:.8f}, onnx {exported[worst]:.8f}) exceeds {ATOL}"
    )

    threshold = predictor.threshold
    flipped = (native >= threshold) != (exported >= threshold)
    borderline = np.abs(native - threshold) <= ATOL
    assert not (flipped & ~borderline).any(), "ONNX changes the Yes/No decision for non-borderline rows"

    return {
        "rows_compared": int(X.shape[0]),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "tolerance": ATOL,
        "decision_flips_at_threshold": int(flipped.sum()),
        "dynamic_batch_sizes_checked": [1, 7, 1000],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "production")
    parser.add_argument(
        "--data", type=Path, nargs="+",
        default=[PROJECT_ROOT / "data" / "raw" / "Train_v2.csv", PROJECT_ROOT / "data" / "raw" / "Test_v2.csv"],
    )
    args = parser.parse_args()

    predictor = Predictor.load(args.model_dir)
    model = convert(predictor)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    X = build_verification_matrix(predictor, args.data)

    try:
        evidence = verify(session, predictor, X)
    except AssertionError as exc:
        print(f"ONNX VERIFICATION FAILED -- nothing written: {exc}", file=sys.stderr)
        return 1

    onnx_path = args.model_dir / "model.onnx"
    onnx.save_model(model, onnx_path)
    report = {
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "model_version": predictor.model_version,
        "source_model_sha256": predictor.metadata["model_sha256"],
        "target_opset": TARGET_OPSET,
        "versions": {"onnx": onnx.__version__, "onnxmltools": onnxmltools.__version__, "onnxruntime": ort.__version__},
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verification": evidence,
    }
    (args.model_dir / "onnx_export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["verification"], indent=2))
    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
