import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "config"))
from settings import load_settings  # noqa: E402


def test_default_config_has_typed_xai_settings():
    xai = load_settings().xai

    for name in ("background_size", "explanation_sample_size", "oracle_instances", "ig_n_steps", "bootstrap_reps"):
        assert isinstance(getattr(xai, name), int) and getattr(xai, name) > 0
    # ~100-200 rows: enough to represent 4 countries x 2 classes, small enough for exact Shapley
    assert 100 <= xai.background_size <= 200


def test_xai_paths_resolve_under_the_project_root():
    xai = load_settings().xai

    assert xai.model_dir == ROOT / "models" / "xai"
    assert xai.artifacts_dir == ROOT / "reports" / "xai"
    assert xai.figures_dir == ROOT / "reports" / "figures" / "xai"
    assert xai.decisions_log == ROOT / "reports" / "xai_decisions_log.md"


@pytest.mark.parametrize("bad_value", [0, -5, 1.5, "200", True])
def test_non_positive_or_non_integer_xai_values_fail_loudly(tmp_path, bad_value):
    raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    raw["xai"]["background_size"] = bad_value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="xai.background_size"):
        load_settings(config_path)
