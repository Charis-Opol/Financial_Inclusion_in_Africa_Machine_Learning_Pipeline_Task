"""The ONNX verifier must pass a faithful export and reject an unfaithful one."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxmltools")
ort = pytest.importorskip("onnxruntime")

from fin_inclusion.serving.predictor import Predictor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "production"
if not (MODEL_DIR / "model.json").is_file():
    pytest.skip("models/production missing -- run scripts/train_production_model.py", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("export_onnx", ROOT / "scripts" / "export_onnx.py")
export_onnx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_onnx)


@pytest.fixture(scope="module")
def exported():
    predictor = Predictor.load(MODEL_DIR)
    model = export_onnx.convert(predictor)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    X = predictor.encoder.encode_batch(export_onnx.edge_case_records(predictor.metadata))
    X = np.repeat(X, 40, axis=0)  # >= 1000 rows so every dynamic batch size can be exercised
    return predictor, model, session, X


def test_faithful_export_passes(exported):
    predictor, model, session, X = exported
    evidence = export_onnx.verify(session, predictor, X)
    assert evidence["max_abs_diff"] <= export_onnx.ATOL
    assert [o.name for o in model.graph.output] == ["probabilities"]  # 0.5-threshold label removed


def test_column_order_bug_is_caught(exported):
    predictor, _, session, X = exported

    class ColumnShufflingSession:
        """Simulates an export whose input columns are wired in the wrong order."""
        def run(self, outputs, feeds):
            return session.run(outputs, {k: v[:, ::-1].copy() for k, v in feeds.items()})

    with pytest.raises(AssertionError, match="max \\|native - onnx\\|"):
        export_onnx.verify(ColumnShufflingSession(), predictor, X)
