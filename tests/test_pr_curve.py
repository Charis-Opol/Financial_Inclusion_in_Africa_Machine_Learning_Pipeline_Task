import numpy as np

from fin_inclusion.evaluation.pr_curve import plot_pr_curve_overlay


def test_writes_file_and_returns_average_precision(tmp_path):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_score_good = np.clip(y_true + rng.normal(0, 0.3, size=200), 0, 1)
    y_score_bad = rng.uniform(size=200)
    output_path = tmp_path / "pr_curve.png"

    aps = plot_pr_curve_overlay(
        {"Good model": (y_true, y_score_good), "Random model": (y_true, y_score_bad)},
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert set(aps) == {"Good model", "Random model"}
    assert aps["Good model"] > aps["Random model"]
