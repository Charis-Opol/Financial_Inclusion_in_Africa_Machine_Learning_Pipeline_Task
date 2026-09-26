"""End-to-end tests of the prediction API against the real production artifact.

Run from the serving environment (pydantic v2): `.venv/Scripts/python -m pytest tests/test_serving_api.py`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pydantic", minversion="2")
pytest.importorskip("fastapi")

import xgboost  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fin_inclusion.serving.app import create_app  # noqa: E402
from fin_inclusion.serving.config import ServiceConfig  # noqa: E402
from fin_inclusion.serving.predictor import ModelLoadError, Predictor  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "production"
if not (MODEL_DIR / "model.json").is_file():
    pytest.skip("models/production missing -- run scripts/train_production_model.py", allow_module_level=True)

VALID_RECORD = {
    "country": "Uganda",
    "location_type": "Rural",
    "cellphone_access": "Yes",
    "household_size": 5,
    "age_of_respondent": 34,
    "gender_of_respondent": "Female",
    "relationship_with_head": "Spouse",
    "marital_status": "Married/Living together",
    "education_level": "Primary education",
    "job_type": "Self employed",
}


def _config(model_dir: Path, log_dir: Path) -> ServiceConfig:
    return ServiceConfig(model_dir=model_dir, inference_threads=1, log_level="WARNING", prediction_log_dir=log_dir)


@pytest.fixture(scope="module")
def log_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("prediction-logs")


@pytest.fixture(scope="module")
def client(log_dir):
    with TestClient(create_app(_config(MODEL_DIR, log_dir))) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_version"].startswith("xgb-cw-")


def test_predict_returns_consistent_prediction(client):
    response = client.post("/predict", json=VALID_RECORD, headers={"X-Request-ID": "abc-123"})
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "abc-123" == response.headers["X-Request-ID"]
    assert 0.0 <= body["probability_bank_account"] <= 1.0
    expected = "Yes" if body["probability_bank_account"] >= body["decision_threshold"] else "No"
    assert body["predicted_bank_account"] == expected
    assert body["warnings"] == []


def test_disguised_missing_answer_is_a_valid_input(client):
    response = client.post("/predict", json=VALID_RECORD | {"job_type": "Dont Know/Refuse to answer"})
    assert response.status_code == 200


def test_unsafe_request_id_is_replaced(client):
    response = client.post("/predict", json=VALID_RECORD, headers={"X-Request-ID": "bad id\twith tab"})
    assert response.headers["X-Request-ID"] != "bad id\twith tab"


def test_out_of_training_range_value_is_served_with_warning(client):
    response = client.post("/predict", json=VALID_RECORD | {"household_size": 25})
    assert response.status_code == 200
    assert "household_size=25" in response.json()["warnings"][0]


@pytest.mark.parametrize(
    "override",
    [
        {"country": "uganda"},  # wrong case -> not a known category
        {"country": "Nigeria"},
        {"household_size": 3.5},
        {"household_size": "5"},  # strict: no string coercion
        {"household_size": True},
        {"household_size": 0},
        {"household_size": 31},
        {"age_of_respondent": 15},
        {"age_of_respondent": 101},
        {"year": 2018},  # extra field
    ],
)
def test_invalid_input_is_422(client, override):
    response = client.post("/predict", json=VALID_RECORD | override)
    assert response.status_code == 422, response.text
    assert response.json()["error"] == "invalid_request"


def test_missing_field_is_422(client):
    record = {k: v for k, v in VALID_RECORD.items() if k != "age_of_respondent"}
    assert client.post("/predict", json=record).status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_422_not_500(client, token):
    # Python's json module accepts these non-standard tokens, so they really
    # can arrive; the 422 body must also not crash trying to echo them back.
    body = json.dumps(VALID_RECORD).replace('"household_size": 5', f'"household_size": {token}')
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text


def test_missing_model_gives_503_with_reason(tmp_path):
    with TestClient(create_app(_config(tmp_path, tmp_path / "logs"))) as unready:
        health = unready.get("/health")
        assert health.status_code == 503
        assert "Missing artifact file" in health.json()["detail"]
        predict = unready.post("/predict", json=VALID_RECORD)
        assert predict.status_code == 503
        assert predict.headers["Retry-After"] == "30"


def test_model_failure_gives_500(client, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise xgboost.core.XGBoostError("simulated booster failure")

    predictor = client.app.state.predictor
    monkeypatch.setattr(predictor._booster, "inplace_predict", broken_predict)
    response = client.post("/predict", json=VALID_RECORD)
    assert response.status_code == 500
    assert response.json()["error"] == "inference_failed"
    assert response.json()["request_id"]


def test_non_finite_model_output_gives_500(client, monkeypatch):
    import numpy as np

    predictor = client.app.state.predictor
    monkeypatch.setattr(predictor._booster, "inplace_predict", lambda *a, **k: np.array([np.nan], dtype=np.float32))
    assert client.post("/predict", json=VALID_RECORD).status_code == 500


def _copy_artifact(tmp_path: Path) -> Path:
    target = tmp_path / "artifact"
    shutil.copytree(MODEL_DIR, target)
    return target


def test_load_rejects_checksum_mismatch(tmp_path):
    artifact = _copy_artifact(tmp_path)
    metadata = json.loads((artifact / "metadata.json").read_text())
    metadata["model_sha256"] = "0" * 64
    (artifact / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ModelLoadError, match="checksum"):
        Predictor.load(artifact)


def test_load_rejects_schema_vocabulary_drift(tmp_path):
    artifact = _copy_artifact(tmp_path)
    metadata = json.loads((artifact / "metadata.json").read_text())
    metadata["raw_categorical_levels"]["country"].append("Ghana")
    (artifact / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ModelLoadError, match="disagree"):
        Predictor.load(artifact)


def test_prediction_is_logged_as_json_line(tmp_path):
    log_dir = tmp_path / "logs"
    with TestClient(create_app(_config(MODEL_DIR, log_dir))) as test_client:
        response = test_client.post("/predict", json=VALID_RECORD, headers={"X-Request-ID": "log-test-1"})
        assert response.status_code == 200
        test_client.post("/predict", json=VALID_RECORD | {"country": "Nigeria"})  # 422: not a prediction
    # Leaving the context runs lifespan shutdown, which flushes the writer.
    lines = [json.loads(line) for f in log_dir.glob("predictions-*.jsonl") for line in f.read_text().splitlines()]
    assert len(lines) == 1
    event = lines[0]
    assert event["request_id"] == "log-test-1"
    assert event["features"] == VALID_RECORD
    assert event["probability"] == response.json()["probability_bank_account"]
    assert event["model_version"].startswith("xgb-cw-")
    assert event["latency_ms"] >= event["inference_ms"] > 0
    assert {"timestamp", "prediction", "confidence", "decision_threshold"} <= event.keys()


def test_health_exposes_prediction_log_counters(client):
    stats = client.get("/health").json()["prediction_log"]
    assert stats["dropped"] == 0 and stats["writer_alive"] is True


def test_unwritable_log_dir_gives_503(tmp_path):
    not_a_dir = tmp_path / "occupied"
    not_a_dir.write_text("a file where the log directory should be")
    with TestClient(create_app(_config(MODEL_DIR, not_a_dir))) as unready:
        health = unready.get("/health")
        assert health.status_code == 503
        assert "not writable" in health.json()["detail"]
        assert unready.post("/predict", json=VALID_RECORD).status_code == 503
