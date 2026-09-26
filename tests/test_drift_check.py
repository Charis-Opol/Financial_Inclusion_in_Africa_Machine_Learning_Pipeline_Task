"""Drift-check CLI behaviour, driven by a small synthetic reference and log."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from fin_inclusion.monitoring import drift_check

REFERENCE = {
    "model_version": "m1",
    "segments": {
        "all": {
            "n": 1000,
            "positive_rate": 0.2,
            "features": {
                "age_of_respondent": {"kind": "numeric", "edges": [30.0], "proportions": [0.5, 0.5]},
                "cellphone_access": {"kind": "categorical", "levels": ["No", "Yes"], "proportions": [0.3, 0.7, 0.0]},
                "prediction_score": {"kind": "numeric", "edges": [0.5], "proportions": [0.8, 0.2]},
            },
        }
    },
}


def _event(age, phone, probability, model_version="m1", when=None):
    when = when or datetime.now(timezone.utc)
    return {
        "timestamp": when.isoformat(),
        "request_id": "r",
        "model_version": model_version,
        "features": {"age_of_respondent": age, "cellphone_access": phone, "country": "Uganda"},
        "probability": probability,
        "prediction": "Yes" if probability >= 0.5 else "No",
    }


def _matching_population(n):
    """Events whose bin shares exactly match REFERENCE."""
    events = []
    for i in range(n):
        events.append(_event(age=25 if i % 2 else 40, phone="No" if i % 10 < 3 else "Yes",
                             probability=0.9 if i % 10 < 2 else 0.1))
    return events


def _write_log(tmp_path, events, extra_lines=()):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [json.dumps(e) for e in events] + list(extra_lines)
    (log_dir / f"predictions-{day}-host.jsonl").write_text("\n".join(lines) + "\n")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(REFERENCE))
    return ["--log-dir", str(log_dir), "--reference", str(reference), "--min-samples", "100"]


def test_matching_population_is_stable(tmp_path):
    assert drift_check.main(_write_log(tmp_path, _matching_population(200))) == drift_check.EXIT_STABLE


def test_shifted_population_is_significant(tmp_path, capsys):
    shifted = [_event(age=70, phone="No", probability=0.9) for _ in range(200)]
    assert drift_check.main(_write_log(tmp_path, shifted)) == drift_check.EXIT_SIGNIFICANT
    assert "OVERALL: SIGNIFICANT" in capsys.readouterr().out


def test_too_few_samples_gives_no_verdict(tmp_path):
    assert drift_check.main(_write_log(tmp_path, _matching_population(50))) == drift_check.EXIT_INSUFFICIENT


def test_other_model_versions_are_excluded(tmp_path):
    events = _matching_population(200) + [_event(70, "No", 0.9, model_version="m0") for _ in range(500)]
    assert drift_check.main(_write_log(tmp_path, events)) == drift_check.EXIT_STABLE


def test_events_outside_time_window_are_excluded(tmp_path):
    old = datetime.now(timezone.utc) - timedelta(hours=48)
    events = _matching_population(200) + [_event(70, "No", 0.9, when=old) for _ in range(500)]
    assert drift_check.main(_write_log(tmp_path, events) + ["--hours", "24"]) == drift_check.EXIT_STABLE


def test_corrupt_log_fails_loudly(tmp_path, capsys):
    args = _write_log(tmp_path, _matching_population(200), extra_lines=["{not json"] * 10)
    assert drift_check.main(args) == drift_check.EXIT_ERROR
    assert "malformed" in capsys.readouterr().err


def test_json_report_written(tmp_path):
    out = tmp_path / "report.json"
    drift_check.main(_write_log(tmp_path, _matching_population(200)) + ["--json-out", str(out)])
    report = json.loads(out.read_text())
    assert report["status"] == "stable" and report["n_recent"] == 200


def test_missing_logs_is_an_error(tmp_path):
    (tmp_path / "logs").mkdir()
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(REFERENCE))
    assert drift_check.main(["--log-dir", str(tmp_path / "logs"), "--reference", str(reference)]) == drift_check.EXIT_ERROR


@pytest.mark.parametrize("country", ["Kenya"])
def test_unknown_segment_is_an_error(tmp_path, country):
    args = _write_log(tmp_path, _matching_population(200)) + ["--country", country]
    assert drift_check.main(args) == drift_check.EXIT_ERROR
