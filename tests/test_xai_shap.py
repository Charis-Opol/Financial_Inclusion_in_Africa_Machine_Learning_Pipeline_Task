import numpy as np
import pandas as pd
import pytest

from fin_inclusion.interpretability.attribution import LOGIT_SPACE, Attribution
from fin_inclusion.interpretability.bootstrap import bootstrap_ci, bootstrap_mean_abs_importance
from fin_inclusion.interpretability.checks import check_efficiency
from fin_inclusion.interpretability.model_adapters import RawFeatureModel, XGBoostRawModel
from fin_inclusion.interpretability.sampling import LOCAL_EXAMPLE_RULES, select_local_examples
from fin_inclusion.interpretability.shapley import (
    ExactShapleyExplainer,
    coalition_masks,
    exact_shapley_values,
    shapley_from_coalition_values,
)
from fin_inclusion.preprocessing.pipeline_factory import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

RAW_FEATURES = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


class ToyModel(RawFeatureModel):
    """Three raw features (one categorical, two numeric) and any logit function of their codes."""

    name = "toy"
    raw_features = ("colour", "x", "y")
    categorical_levels = {"colour": ("blue", "green", "red")}
    decision_threshold = 0.5

    def __init__(self, fn):
        self.fn = fn

    def predict_logit_codes(self, codes: np.ndarray) -> np.ndarray:
        return self.fn(np.asarray(codes, dtype=np.float64))


def _toy_rows(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"colour": rng.choice(["blue", "green", "red"], size=n), "x": rng.normal(size=n), "y": rng.normal(size=n)}
    )


# --- Codes -----------------------------------------------------------------


def test_codes_round_trip_and_score_identically(xgb_artifact):
    model_dir, survey, _, _ = xgb_artifact
    adapter = XGBoostRawModel.load(model_dir)
    rows = survey[RAW_FEATURES].head(40)

    codes = adapter.to_codes(rows)

    assert codes.shape == (40, len(RAW_FEATURES))
    assert (adapter.from_codes(codes).to_numpy() == rows.to_numpy()).all()
    np.testing.assert_array_equal(adapter.predict_logit_codes(codes), adapter.predict_logit(rows))


def test_swapping_household_size_recomputes_the_outlier_flag(xgb_artifact, mlp_fitted):
    model_dir, survey, _, _ = xgb_artifact
    xgb, mlp = XGBoostRawModel.load(model_dir), mlp_fitted[0]
    codes = xgb.to_codes(survey[RAW_FEATURES].iloc[[1]])
    position = RAW_FEATURES.index("household_size")
    large = codes.copy()
    large[0, position] = 18

    flag = xgb.feature_columns.index("household_size_outlier")
    assert xgb.encode_codes(codes)[0, flag] == 0 and xgb.encode_codes(large)[0, flag] == 1
    flag = mlp.numeric_inputs.index("household_size_outlier")
    assert mlp.encode_codes(codes)[1][0, flag] == 0 and mlp.encode_codes(large)[1][0, flag] == 1


# --- Exact Shapley ---------------------------------------------------------


def test_coalition_masks_enumerate_every_subset_once():
    masks = coalition_masks(3)

    assert masks.shape == (8, 3)
    assert len({tuple(row) for row in masks}) == 8
    assert masks[0].sum() == 0 and masks[-1].all()
    assert masks[0b101].tolist() == [True, False, True]


def test_shapley_matches_a_hand_computed_two_player_game():
    # v(empty)=0, v({0})=1, v({1})=2, v({0,1})=5
    phi = shapley_from_coalition_values(np.array([0.0, 1.0, 2.0, 5.0]), n_players=2)

    # phi_0 = 1/2 (1 - 0) + 1/2 (5 - 2) = 2;  phi_1 = 1/2 (2 - 0) + 1/2 (5 - 1) = 3
    np.testing.assert_allclose(phi, [2.0, 3.0])


def test_additive_model_gets_each_terms_deviation_from_the_background_mean():
    rng = np.random.default_rng(0)
    X, background = rng.normal(size=(6, 3)), rng.normal(size=(25, 3))
    terms = [lambda v: 2.0 * v, lambda v: v**2, lambda v: -np.sin(v)]

    def predict(rows):
        return sum(term(rows[:, j]) for j, term in enumerate(terms)) + 0.7

    phi, base_value = exact_shapley_values(predict, X, background)

    expected = np.column_stack([term(X[:, j]) - term(background[:, j]).mean() for j, term in enumerate(terms)])
    np.testing.assert_allclose(phi, expected, atol=1e-12)
    assert base_value == pytest.approx(predict(background).mean())


def test_interaction_is_split_equally_between_the_interacting_features():
    # f = x0 * x1, one background row (b0, b1); x2 is irrelevant.
    x, b = np.array([[3.0, 5.0, 9.0]]), np.array([[1.0, 2.0, 4.0]])

    phi, base_value = exact_shapley_values(lambda rows: rows[:, 0] * rows[:, 1], x, b)

    # phi_0 = 1/2 [f(x0,b1) - f(b0,b1)] + 1/2 [f(x0,x1) - f(b0,x1)] = 1/2 (6-2) + 1/2 (15-5) = 7
    # phi_1 = 1/2 [f(b0,x1) - f(b0,b1)] + 1/2 [f(x0,x1) - f(x0,b1)] = 1/2 (5-2) + 1/2 (15-6) = 6
    np.testing.assert_allclose(phi, [[7.0, 6.0, 0.0]], atol=1e-12)
    assert base_value == pytest.approx(2.0)


def test_exact_shapley_is_efficient_for_an_arbitrary_nonlinear_model():
    rng = np.random.default_rng(1)
    X, background = rng.normal(size=(10, 4)), rng.normal(size=(30, 4))

    def predict(rows):
        return np.tanh(rows[:, 0] * rows[:, 1]) + np.maximum(rows[:, 2], rows[:, 3]) ** 2

    phi, base_value = exact_shapley_values(predict, X, background)

    np.testing.assert_allclose(phi.sum(axis=1) + base_value, predict(X), atol=1e-12)


def test_exact_shapley_rejects_a_mismatched_background_and_too_many_players():
    with pytest.raises(ValueError, match="background"):
        exact_shapley_values(lambda rows: rows[:, 0], np.zeros((1, 3)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="intractable"):
        exact_shapley_values(lambda rows: rows[:, 0], np.zeros((1, 20)), np.zeros((2, 20)))


def test_exact_shapley_explainer_returns_a_raw_feature_attribution():
    model = ToyModel(lambda codes: 1.5 * codes[:, 0] + codes[:, 1] * codes[:, 2])
    rows, background = _toy_rows(5, seed=0), _toy_rows(20, seed=1)

    attribution = ExactShapleyExplainer(model, background).explain(rows)

    assert isinstance(attribution, Attribution)
    assert attribution.feature_names == ("colour", "x", "y")
    assert (attribution.method, attribution.target_space) == ("exact_shapley", LOGIT_SPACE)
    assert "interventional" in attribution.baseline_description
    assert check_efficiency(attribution, model.predict_logit(rows), atol=1e-9).passed


# --- SHAP-library explainers -------------------------------------------------


def test_kernel_shap_with_all_coalitions_equals_the_oracle():
    pytest.importorskip("shap", reason="shap is required for the SHAP explainers")
    from fin_inclusion.interpretability.shap_explainers import KernelShapExplainer

    model = ToyModel(lambda codes: np.tanh(codes[:, 0] - codes[:, 1]) + codes[:, 1] * codes[:, 2])
    rows, background = _toy_rows(6, seed=0), _toy_rows(20, seed=1)

    kernel_explainer = KernelShapExplainer(model, background)
    kernel = kernel_explainer.explain(rows)
    oracle = ExactShapleyExplainer(model, background).explain(rows)

    assert kernel_explainer.is_exact and kernel.metadata["exact"] is True
    np.testing.assert_allclose(kernel.values, oracle.values, atol=1e-8)
    assert kernel.base_value == pytest.approx(oracle.base_value)


def test_kernel_shap_reports_when_it_is_only_sampling_and_is_reproducible(xgb_artifact):
    pytest.importorskip("shap", reason="shap is required for the SHAP explainers")
    from fin_inclusion.interpretability.shap_explainers import KernelShapExplainer

    model_dir, survey, _, _ = xgb_artifact
    adapter = XGBoostRawModel.load(model_dir)
    rows, background = survey[RAW_FEATURES].iloc[100:104], survey[RAW_FEATURES].iloc[:30]

    explainer = KernelShapExplainer(adapter, background, nsamples=60, seed=3)
    first, second = explainer.explain(rows), explainer.explain(rows)

    assert not explainer.is_exact and first.metadata["exact"] is False
    assert "sampled" in first.baseline_description
    np.testing.assert_array_equal(first.values, second.values)


def test_tree_shap_is_efficient_grouped_and_uses_the_whole_background(xgb_artifact):
    pytest.importorskip("shap", reason="shap is required for the SHAP explainers")
    from fin_inclusion.interpretability.shap_explainers import TreeShapExplainer

    model_dir, survey, _, _ = xgb_artifact
    adapter = XGBoostRawModel.load(model_dir)
    # > 100 rows on purpose: shap subsamples a bare background array to 100
    rows, background = survey[RAW_FEATURES].iloc[150:190], survey[RAW_FEATURES].iloc[:150]

    explainer = TreeShapExplainer(adapter, background)
    grouped, encoded = explainer.explain(rows), explainer.explain_encoded(rows)

    assert grouped.feature_names == tuple(RAW_FEATURES)
    assert encoded.feature_names == adapter.feature_columns
    assert grouped.base_value == pytest.approx(adapter.predict_logit(background).mean(), abs=1e-5)
    assert check_efficiency(grouped, adapter.predict_logit(rows), atol=1e-3).passed
    np.testing.assert_allclose(grouped.values.sum(axis=1), encoded.values.sum(axis=1), atol=1e-9)


def test_grouped_tree_shap_equals_the_oracle_when_trees_cannot_interact(write_xgb_artifact, survey, tmp_path):
    """Depth-1 trees make the model additive over encoded columns, the one
    case where the 40-column game and the 10-feature game must agree."""
    pytest.importorskip("shap", reason="shap is required for the SHAP explainers")
    from fin_inclusion.interpretability.shap_explainers import TreeShapExplainer

    model_dir, _, _ = write_xgb_artifact(tmp_path, n_estimators=30, max_depth=1)
    adapter = XGBoostRawModel.load(model_dir)
    rows, background = survey[RAW_FEATURES].iloc[200:215], survey[RAW_FEATURES].iloc[:60]

    tree = TreeShapExplainer(adapter, background).explain(rows)
    oracle = ExactShapleyExplainer(adapter, background).explain(rows)

    assert np.abs(oracle.values).max() > 0.01  # the comparison is not between two all-zero arrays
    np.testing.assert_allclose(tree.values, oracle.values, atol=1e-4)


def test_gradient_shap_groups_embedding_dimensions_and_is_reproducible(mlp_fitted):
    pytest.importorskip("shap", reason="shap is required for the SHAP explainers")
    from fin_inclusion.interpretability.shap_explainers import GradientShapExplainer

    adapter, _, survey, _, _ = mlp_fitted
    rows, background = survey[RAW_FEATURES].iloc[100:110], survey[RAW_FEATURES].iloc[:40]

    explainer = GradientShapExplainer(adapter, background, nsamples=100, seed=5)
    first, second = explainer.explain(rows), explainer.explain(rows)

    assert first.values.shape == (10, len(RAW_FEATURES))
    assert first.feature_names == tuple(RAW_FEATURES)
    assert first.base_value == pytest.approx(adapter.predict_logit(background).mean(), abs=1e-5)
    np.testing.assert_allclose(first.values, second.values, atol=1e-6)
    assert first.metadata["grouped_from_n_encoded_columns"] == len(CATEGORICAL_COLUMNS) * 3 + 3


# --- Bootstrap ---------------------------------------------------------------


def test_bootstrap_interval_brackets_the_estimate_and_is_reproducible():
    values = np.random.default_rng(0).normal(size=(300, 4))

    first = bootstrap_mean_abs_importance(values, reps=200, seed=42)
    second = bootstrap_mean_abs_importance(values, reps=200, seed=42)

    np.testing.assert_allclose(first.estimate, np.abs(values).mean(axis=0))
    assert (first.lower <= first.estimate).all() and (first.estimate <= first.upper).all()
    assert (first.upper - first.lower > 0).all()
    np.testing.assert_array_equal(first.lower, second.lower)
    assert (first.n_instances, first.reps, first.level) == (300, 200, 0.95)


def test_bootstrap_interval_narrows_with_more_instances():
    rng = np.random.default_rng(1)
    small = bootstrap_mean_abs_importance(rng.normal(size=(50, 1)), reps=300, seed=0)
    large = bootstrap_mean_abs_importance(rng.normal(size=(5000, 1)), reps=300, seed=0)

    assert (large.upper - large.lower)[0] < (small.upper - small.lower)[0]


def test_bootstrap_statistic_receives_row_positions_and_bad_arguments_fail():
    data = np.arange(10.0)

    result = bootstrap_ci(lambda rows: data[rows].mean(), n_instances=10, reps=50, seed=0)

    assert result.estimate == pytest.approx(4.5)
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_ci(lambda rows: 0.0, n_instances=1, reps=10, seed=0)
    with pytest.raises(ValueError, match="level"):
        bootstrap_ci(lambda rows: 0.0, n_instances=5, reps=10, seed=0, level=1.5)


# --- Local examples ----------------------------------------------------------


def test_local_examples_follow_their_stated_rules():
    #            row:   0     1     2     3     4     5     6     7     8     9
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0])
    score = np.array([0.9, 0.95, 0.1, 0.05, 0.2, 0.8, 0.3, 0.7, 0.40, 0.42])
    candidates, focus = np.arange(8), np.array([7, 8, 9])

    examples = select_local_examples(candidates, focus, y, score, threshold=0.5)

    assert examples == {
        "confident_tp": 1,     # y=1, predicted yes, highest score
        "confident_tn": 3,     # y=0, predicted no, lowest score
        "false_negative": 4,   # y=1, predicted no, lowest score
        "false_positive": 5,   # y=0, predicted yes, highest score
        "focus_typical": 9,    # focus scores 0.70, 0.40, 0.42 -> median 0.42
    }
    assert set(examples) == set(LOCAL_EXAMPLE_RULES)


def test_local_examples_break_ties_by_lowest_row_and_fail_loudly_when_a_case_is_absent():
    y = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.9, 0.1, 0.1])

    with pytest.raises(ValueError, match="false_negative"):
        select_local_examples(np.arange(4), np.arange(4), y, score, threshold=0.5)

    y, score = np.array([1, 1, 0, 0, 1, 0]), np.array([0.9, 0.9, 0.1, 0.1, 0.2, 0.8])
    examples = select_local_examples(np.array([5, 4, 3, 2, 1, 0]), np.arange(6), y, score, threshold=0.5)
    assert (examples["confident_tp"], examples["confident_tn"]) == (0, 2)


# --- Plots ---------------------------------------------------------------------


def test_every_plot_writes_a_non_empty_file(tmp_path):
    from fin_inclusion.interpretability.plots import (
        plot_beeswarm,
        plot_dependence,
        plot_importance_bars,
        plot_waterfall,
    )

    rng = np.random.default_rng(0)
    names = ["country", "age", "cellphone"]
    values = rng.normal(size=(80, 3))
    codes = np.column_stack([rng.integers(0, 4, 80), rng.integers(16, 90, 80), rng.integers(0, 2, 80)]).astype(float)
    boot = bootstrap_mean_abs_importance(values, reps=50, seed=0)

    plot_importance_bars({"Pooled": boot, "Uganda only": boot}, names, tmp_path / "bars.png", "title", "caption.")
    plot_beeswarm(values, codes, names, {"age": "dark = larger", "cellphone": "dark = Yes"},
                  tmp_path / "beeswarm.png", "title", "caption.")
    plot_waterfall([f"{n} = v" for n in names], values[0], -1.2, tmp_path / "waterfall.png", "title", "caption.")
    plot_dependence("country", rng.choice(["Kenya", "Uganda"], 80), values[:, 0], True,
                    tmp_path / "dep_cat.png", "title", "caption.")
    plot_dependence("age", codes[:, 1], values[:, 1], False, tmp_path / "dep_num.png", "title", "caption.")

    for name in ("bars", "beeswarm", "waterfall", "dep_cat", "dep_num"):
        assert (tmp_path / f"{name}.png").stat().st_size > 5000
