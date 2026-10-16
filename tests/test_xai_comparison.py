import numpy as np
import pandas as pd
import pytest

from fin_inclusion.interpretability.comparison import agreement_with_ci, cramers_v, per_feature_disagreement
from fin_inclusion.interpretability.faithfulness import (
    coalition_values,
    faithfulness_scores,
    numeric_nudges,
    random_rankings,
    ranking,
    relative_change,
    removal_curves,
)
from fin_inclusion.interpretability.model_adapters import RawFeatureModel
from fin_inclusion.interpretability.shapley import ExactShapleyExplainer, exact_shapley_interactions
from fin_inclusion.preprocessing.pipeline_factory import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

RAW_FEATURES = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


class ToyModel(RawFeatureModel):
    """Three numeric raw features and any logit function of them."""

    name = "toy"
    raw_features = ("x0", "x1", "x2")
    categorical_levels: dict = {}
    decision_threshold = 0.5

    def __init__(self, fn):
        self.fn = fn

    def predict_logit_codes(self, codes):
        return self.fn(np.asarray(codes, dtype=np.float64))


def _frame(values) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(values, dtype=float), columns=["x0", "x1", "x2"])


# --- Agreement ------------------------------------------------------------------------


def test_agreement_with_ci_is_perfect_for_identical_attributions():
    values = np.random.default_rng(0).normal(size=(80, 5))

    result = agreement_with_ci(values, values, reps=100, seed=0)

    for metric in ("global_spearman", "top3_overlap", "instance_spearman_median", "same_top_feature_share",
                   "sign_agreement_share"):
        assert result[metric]["estimate"] == pytest.approx(1.0)
        assert result[metric]["ci"] == pytest.approx([1.0, 1.0])
    assert len(result["instance_spearman_values"]) == 80


def test_agreement_interval_brackets_the_estimate_for_noisy_copies():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(300, 6)) * np.array([3, 2, 1.5, 1, 0.5, 0.2])
    b = a + rng.normal(scale=0.8, size=a.shape)

    result = agreement_with_ci(a, b, reps=200, seed=0)

    for metric in ("global_spearman", "instance_spearman_median", "sign_agreement_share"):
        low, high = result[metric]["ci"]
        assert low <= result[metric]["estimate"] <= high
    assert 0 < result["instance_spearman_median"]["estimate"] < 1


def test_per_feature_disagreement_is_zero_for_equal_and_two_for_opposite():
    a = np.random.default_rng(2).normal(size=(50, 3))

    np.testing.assert_allclose(per_feature_disagreement(a, a), 0.0)
    np.testing.assert_allclose(per_feature_disagreement(a, -a), 2.0)


def test_cramers_v_is_one_for_a_function_and_near_zero_for_independent_columns():
    rng = np.random.default_rng(3)
    a = rng.choice(list("abc"), size=5000)

    assert cramers_v(a, np.char.upper(a)) == pytest.approx(1.0)
    assert cramers_v(a, rng.choice(list("xyz"), size=5000)) < 0.05
    assert cramers_v(a, np.full(5000, "same")) == 0.0


# --- Value function: Shapley vs Ablation at the same reference ---------------------------


def test_without_interactions_shapley_at_a_point_equals_ablation_at_that_point():
    pytest.importorskip("captum", reason="captum is required for the Captum explainers")
    from fin_inclusion.interpretability.captum_explainers import FeatureAblationExplainer

    model = ToyModel(lambda c: 2.0 * c[:, 0] + c[:, 1] ** 2 - np.sin(c[:, 2]))
    rows, point = _frame(np.random.default_rng(4).normal(size=(10, 3))), np.array([0.5, -1.0, 2.0])

    shapley = ExactShapleyExplainer.at_point(model, point, "test point", "test").explain(rows)
    ablation = FeatureAblationExplainer(model, point, "modal").explain(rows)

    np.testing.assert_allclose(shapley.values, ablation.values, atol=1e-12)
    assert shapley.base_value == pytest.approx(model.predict_logit_codes(point[None])[0])
    assert "one baseline point" in shapley.baseline_description


def test_with_pairwise_interactions_ablation_minus_shapley_is_the_pairwise_interaction_sum():
    pytest.importorskip("captum", reason="captum is required for the Captum explainers")
    from fin_inclusion.interpretability.captum_explainers import FeatureAblationExplainer

    model = ToyModel(lambda c: c[:, 0] * c[:, 1] + 0.5 * c[:, 1] * c[:, 2] + c[:, 2])
    rows, point = _frame(np.random.default_rng(5).normal(size=(8, 3))), np.array([0.3, -0.2, 1.0])

    shapley = ExactShapleyExplainer.at_point(model, point, "test point", "test").explain(rows).values
    ablation = FeatureAblationExplainer(model, point, "modal").explain(rows).values
    interactions, _ = exact_shapley_interactions(model.predict_logit_codes, model.to_codes(rows), point[None])
    pair_sum = interactions.sum(axis=2) - np.einsum("nii->ni", interactions)

    np.testing.assert_allclose(ablation - shapley, pair_sum, atol=1e-12)
    assert np.abs(pair_sum).max() > 0.1


# --- Faithfulness ------------------------------------------------------------------------


def test_removal_curves_start_at_the_respondent_and_end_at_the_reference():
    model = ToyModel(lambda c: 3 * c[:, 0] + c[:, 1] - 0.5 * c[:, 2])
    X, reference = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]]), np.zeros((1, 3))
    order = ranking(np.array([[3.0, 2.0, -1.5], [-3.0, 0.0, -2.0]]))

    deletion, insertion = removal_curves(model.predict_logit_codes, X, order, reference)

    f_x, f_ref = model.predict_logit_codes(X), model.predict_logit_codes(reference)[0]
    np.testing.assert_allclose(deletion[:, 0], f_x)
    np.testing.assert_allclose(deletion[:, -1], f_ref)
    np.testing.assert_allclose(insertion[:, 0], f_ref)
    np.testing.assert_allclose(insertion[:, -1], f_x)
    # row 0, k = 1 removes x0 (largest |attribution|): 3*1 is lost
    assert deletion[0, 1] == pytest.approx(f_x[0] - 3.0)


def test_a_correct_ranking_beats_random_orders_on_an_additive_model():
    rng = np.random.default_rng(6)
    weights = np.array([5.0, 1.0, 0.1])
    model = ToyModel(lambda c: c @ weights)
    X, reference = rng.normal(size=(200, 3)), np.zeros((1, 3))
    logit = model.predict_logit_codes(X)
    truth = X * weights  # exact attributions of a linear model against a zero reference

    true_scores = faithfulness_scores(logit, *removal_curves(model.predict_logit_codes, X, ranking(truth), reference))
    random_scores = [faithfulness_scores(logit, *removal_curves(model.predict_logit_codes, X, order, reference))
                     for order in random_rankings(200, 3, 3, seed=0)]

    assert true_scores["deletion_area"].mean() > np.mean([s["deletion_area"].mean() for s in random_scores])
    assert true_scores["insertion_gap"].mean() < np.mean([s["insertion_gap"].mean() for s in random_scores])
    assert set(true_scores) >= {"comprehensiveness@1", "sufficiency@1"}
    assert "comprehensiveness@5" not in true_scores  # only 3 features: k = 5 is undefined, not an error


def test_coalition_values_average_over_reference_rows():
    model = ToyModel(lambda c: c.sum(axis=1))
    X, reference = np.array([[1.0, 1.0, 1.0]]), np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])

    value = coalition_values(model.predict_logit_codes, X, np.array([[True, False, False]]), reference)

    assert value[0] == pytest.approx(1.0 + 2.0)  # x0 kept (1) + mean of the two references for x1, x2 (1 + 1)


def test_random_rankings_are_permutations_and_reproducible():
    orders = random_rankings(20, 5, 4, seed=1)

    assert orders.shape == (4, 20, 5)
    assert (np.sort(orders, axis=2) == np.arange(5)).all()
    np.testing.assert_array_equal(orders, random_rankings(20, 5, 4, seed=1))


# --- Randomisation and stability ------------------------------------------------------------


def test_randomised_mlp_changes_the_copy_not_the_original(mlp_fitted):
    from fin_inclusion.interpretability.faithfulness import randomised_mlp

    mlp, _, survey, _, _ = mlp_fitted
    rows = survey[RAW_FEATURES].head(40)
    before = mlp.predict_logit(rows)

    unchanged, scrambled = randomised_mlp(mlp, 0, seed=0), randomised_mlp(mlp, 1, seed=0)

    np.testing.assert_array_equal(unchanged.predict_logit(rows), before)
    assert not np.allclose(scrambled.predict_logit(rows), before)
    np.testing.assert_array_equal(mlp.predict_logit(rows), before)


def test_numeric_nudges_stay_within_bounds_and_relative_change_is_scaled():
    codes = np.array([[16.0, 3.0], [50.0, 21.0]])

    nudges = numeric_nudges(codes, {"age": 0, "household": 1}, {"age": (16, 100), "household": (1, 21)})

    labels = [label for label, _, _ in nudges]
    assert labels == ["age -1", "age +1", "household -1", "household +1"]
    age_down = dict((label, valid) for label, _, valid in nudges)["age -1"]
    assert age_down.tolist() == [False, True]
    np.testing.assert_allclose(relative_change(np.array([[1.0, -1.0]]), np.array([[1.5, -1.0]])), [0.25])


def test_path_gradient_profile_integrates_to_the_output_change(mlp_fitted):
    pytest.importorskip("captum", reason="captum is required for the Captum explainers")
    from fin_inclusion.interpretability.captum_explainers import baseline_codes, path_gradient_profile

    mlp, _, survey, _, _ = mlp_fitted
    rows = survey[RAW_FEATURES].iloc[10:30]
    baseline = baseline_codes(mlp, mlp.to_codes(survey[RAW_FEATURES]), "modal")

    profile, endpoint = path_gradient_profile(mlp, rows, baseline, n_points=256)

    change = mlp.predict_logit(rows) - mlp.predict_logit_codes(baseline[None])[0]
    np.testing.assert_allclose(profile.mean(axis=1), change, atol=2e-2)
    assert endpoint.shape == (20, len(CATEGORICAL_COLUMNS) * 3 + 3)


# --- Plots ------------------------------------------------------------------------------------


def test_phase6_plots_write_non_empty_files(tmp_path):
    from fin_inclusion.interpretability.plots import (
        plot_curves,
        plot_forest,
        plot_identity_scatter,
        plot_labelled_scatter,
        plot_lines,
        plot_rho_histograms,
    )

    rng = np.random.default_rng(7)
    plot_rho_histograms({"A1": {"pooled": rng.uniform(-1, 1, 100), "uganda": rng.uniform(-1, 1, 80)}},
                        tmp_path / "hist.png", "t", "caption.", ncols=2)
    plot_forest(["a", "b"], {"pooled": (np.array([0.2, 0.5]), np.array([0.1, 0.4]), np.array([0.3, 0.6]))},
                tmp_path / "forest.png", "t", "x", "caption.", reference=0.0)
    plot_curves({"deletion": {"m1": rng.random(11), "random order": rng.random(11)}}, tmp_path / "curves.png",
                "t", "k", "y", "caption.")
    plot_lines(["s0", "s1", "s2"], {"m1": [1.0, 0.5, 0.1]}, tmp_path / "lines.png", "t", "y", "caption.")
    plot_labelled_scatter(rng.random(4), rng.random(4), list("abcd"), tmp_path / "labelled.png", "t", "x", "y",
                          "caption.")
    plot_identity_scatter(rng.normal(size=200), rng.normal(size=200), tmp_path / "identity.png", "t", "x", "y",
                          "caption.")

    for name in ("hist", "forest", "curves", "lines", "labelled", "identity"):
        assert (tmp_path / f"{name}.png").stat().st_size > 5000


def test_shared_zeros_are_excluded_from_the_robust_rank_correlation():
    from fin_inclusion.interpretability.comparison import (
        instance_rank_correlations,
        instance_rank_correlations_excluding_shared_zeros,
    )

    # Features 3-5 are zero in both (the respondent equals the baseline there);
    # on the informative features 0-2 the two orderings are reversed.
    a = np.array([[3.0, 2.0, 1.0, 0.0, 0.0, 0.0]])
    b = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]])

    assert instance_rank_correlations(a, b)[0] > 0  # inflated by the shared zeros
    assert instance_rank_correlations_excluding_shared_zeros(a, b)[0] == pytest.approx(-1.0)
