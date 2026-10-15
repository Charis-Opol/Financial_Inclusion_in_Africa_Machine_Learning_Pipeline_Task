import numpy as np
import pandas as pd
import pytest

from fin_inclusion.interpretability.attribution import LOGIT_SPACE, Attribution
from fin_inclusion.interpretability.comparison import (
    global_agreement,
    instance_rank_correlations,
    summarize_agreement,
)
from fin_inclusion.interpretability.model_adapters import UNKNOWN_CODE, XGBoostRawModel
from fin_inclusion.preprocessing.pipeline_factory import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

RAW_FEATURES = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
captum = pytest.importorskip("captum", reason="captum is required for the Captum explainers")
torch = pytest.importorskip("torch", reason="PyTorch is required for the Captum explainers")

from fin_inclusion.interpretability.captum_explainers import (  # noqa: E402
    CategoricalInputNet,
    FeatureAblationExplainer,
    IntegratedGradientsExplainer,
    baseline_codes,
    describe_baseline,
    hidden_unit_attributions,
    mlp_tensors,
)


def _rows(survey: pd.DataFrame, start: int, stop: int) -> pd.DataFrame:
    return survey[RAW_FEATURES].iloc[start:stop]


# --- The "unknown" code -------------------------------------------------------


def test_unknown_code_is_an_all_zero_block_for_xgboost_and_index_zero_for_the_mlp(xgb_artifact, mlp_fitted):
    model_dir, survey, _, _ = xgb_artifact
    xgb, mlp = XGBoostRawModel.load(model_dir), mlp_fitted[0]
    codes = xgb.to_codes(_rows(survey, 0, 1))
    country = RAW_FEATURES.index("country")
    codes[0, country] = UNKNOWN_CODE

    country_columns = [i for i, c in enumerate(xgb.feature_columns) if c.startswith("country_")]
    assert xgb.encode_codes(codes)[0, country_columns].sum() == 0
    cat, _ = mlp.encode_codes(codes)
    assert cat[0, CATEGORICAL_COLUMNS.index("country")] == 0
    # every other categorical is still encoded normally
    assert (np.delete(cat[0], CATEGORICAL_COLUMNS.index("country")) > 0).all()


# --- Baselines --------------------------------------------------------------------


def test_baseline_codes_take_the_modal_level_or_unknown_and_the_numeric_mean(mlp_fitted):
    mlp, _, survey, _, _ = mlp_fitted
    training = mlp.to_codes(survey[RAW_FEATURES])

    modal, unknown = baseline_codes(mlp, training, "modal"), baseline_codes(mlp, training, "unknown")

    for j, feature in enumerate(RAW_FEATURES):
        if feature in NUMERIC_COLUMNS:
            assert modal[j] == unknown[j] == pytest.approx(survey[feature].mean())
        else:
            assert unknown[j] == UNKNOWN_CODE
            assert mlp.categorical_levels[feature][int(modal[j])] == survey[feature].mode().sort_values().iloc[0]
    assert "most frequent" in describe_baseline(mlp, modal, "modal")
    with pytest.raises(ValueError, match="baseline"):
        baseline_codes(mlp, training, "median")


# --- Integrated Gradients ----------------------------------------------------------


def test_categorical_input_net_reproduces_the_original_network(mlp_fitted):
    mlp, _, survey, _, _ = mlp_fitted
    cat, num = mlp_tensors(mlp, mlp.to_codes(_rows(survey, 0, 30)))
    net = mlp.net.eval()

    with torch.no_grad():
        assert torch.equal(CategoricalInputNet(net)(cat, num), net(cat, num))


@pytest.mark.parametrize("kind", ["modal", "unknown"])
def test_integrated_gradients_is_complete_and_grouped_to_raw_features(mlp_fitted, kind):
    mlp, _, survey, _, _ = mlp_fitted
    baseline = baseline_codes(mlp, mlp.to_codes(survey[RAW_FEATURES]), kind)
    rows = _rows(survey, 100, 130)

    explainer = IntegratedGradientsExplainer(mlp, baseline, kind, n_steps=512)
    encoded, delta = explainer.explain_with_delta(rows)
    grouped = explainer.explain(rows)

    assert encoded.values.shape == (30, len(CATEGORICAL_COLUMNS) * 3 + 3)
    assert grouped.feature_names == tuple(RAW_FEATURES)
    assert grouped.base_value == pytest.approx(mlp.predict_logit_codes(baseline[None])[0])
    assert np.abs(delta).max() < 1e-2
    # completeness: attributions + f(baseline) = f(x), up to the reported delta
    residual = grouped.values.sum(axis=1) + grouped.base_value - mlp.predict_logit(rows)
    np.testing.assert_allclose(residual, delta, atol=1e-5)
    assert grouped.metadata["max_abs_convergence_delta"] == pytest.approx(np.abs(delta).max())


def test_integrated_gradients_gives_nothing_to_a_respondent_equal_to_the_baseline(mlp_fitted):
    mlp, _, survey, _, _ = mlp_fitted
    row = _rows(survey, 5, 6)
    baseline = mlp.to_codes(row)[0]

    attribution = IntegratedGradientsExplainer(mlp, baseline, "modal", n_steps=64).explain(row)

    np.testing.assert_allclose(attribution.values, 0.0, atol=1e-7)


def test_more_integration_steps_shrink_the_convergence_delta(mlp_fitted):
    mlp, _, survey, _, _ = mlp_fitted
    baseline = baseline_codes(mlp, mlp.to_codes(survey[RAW_FEATURES]), "unknown")
    rows = _rows(survey, 0, 60)

    _, coarse = IntegratedGradientsExplainer(mlp, baseline, "unknown", n_steps=4).explain_with_delta(rows)
    _, fine = IntegratedGradientsExplainer(mlp, baseline, "unknown", n_steps=512).explain_with_delta(rows)

    assert np.abs(fine).mean() < np.abs(coarse).mean()


# --- Feature Ablation ------------------------------------------------------------------


@pytest.mark.parametrize("which", ["xgboost", "mlp"])
def test_feature_ablation_is_the_drop_from_resetting_one_feature(which, xgb_artifact, mlp_fitted):
    model_dir, survey, _, _ = xgb_artifact
    model = XGBoostRawModel.load(model_dir) if which == "xgboost" else mlp_fitted[0]
    baseline = baseline_codes(model, model.to_codes(survey[RAW_FEATURES]), "modal")
    rows = _rows(survey, 50, 60)

    attribution = FeatureAblationExplainer(model, baseline, "modal").explain(rows)

    codes = model.to_codes(rows)
    for j in range(len(RAW_FEATURES)):
        ablated = codes.copy()
        ablated[:, j] = baseline[j]
        expected = model.predict_logit_codes(codes) - model.predict_logit_codes(ablated)
        np.testing.assert_allclose(attribution.values[:, j], expected, atol=1e-6)
    assert attribution.feature_names == tuple(RAW_FEATURES)
    assert "not an additive decomposition" in attribution.baseline_description


def test_feature_ablation_supports_the_unknown_baseline_on_xgboost(xgb_artifact):
    model_dir, survey, _, _ = xgb_artifact
    xgb = XGBoostRawModel.load(model_dir)
    baseline = baseline_codes(xgb, xgb.to_codes(survey[RAW_FEATURES]), "unknown")

    attribution = FeatureAblationExplainer(xgb, baseline, "unknown").explain(_rows(survey, 0, 20))

    assert np.isfinite(attribution.values).all()
    assert attribution.base_value == pytest.approx(xgb.predict_logit_codes(baseline[None])[0])


# --- Layer attribution -------------------------------------------------------------------


def test_hidden_unit_attributions_have_one_column_per_unit_and_conduct_the_whole_difference(mlp_fitted):
    mlp, _, survey, _, _ = mlp_fitted
    baseline = baseline_codes(mlp, mlp.to_codes(survey[RAW_FEATURES]), "modal")
    rows = _rows(survey, 0, 25)

    layers = hidden_unit_attributions(mlp, rows, baseline, n_steps=256)

    assert list(layers) == ["hidden_1", "hidden_2"]
    for name, n_units in zip(layers, (8, 4)):
        result = layers[name]
        assert result["conductance"].shape == result["gradient_x_activation"].shape == (25, n_units)
        assert (result["activation"] >= 0).all()  # ReLU outputs
        # all of f(x) - f(baseline) flows through every hidden layer
        flow = result["conductance"].sum(axis=1)
        target = mlp.predict_logit(rows) - mlp.predict_logit_codes(baseline[None])[0]
        np.testing.assert_allclose(flow - target, result["conductance_delta"], atol=1e-5)
        assert np.abs(result["conductance_delta"]).max() < 2e-2


# --- Comparison ---------------------------------------------------------------------------


def _attribution(values, rows=None) -> Attribution:
    return Attribution(
        values=np.asarray(values, dtype=float), base_value=0.0, feature_names=("a", "b", "c", "d"),
        method="m", model_name="toy", target_space=LOGIT_SPACE, baseline_description="test",
        metadata={} if rows is None else {"rows": rows},
    )


def test_identical_attributions_agree_perfectly():
    values = np.random.default_rng(0).normal(size=(50, 4))

    summary = summarize_agreement(_attribution(values), _attribution(values))

    assert summary["global"]["spearman"] == pytest.approx(1.0)
    assert summary["global"]["top3_overlap"] == 1.0
    assert summary["instance_spearman"]["median"] == pytest.approx(1.0)
    assert summary["same_top_feature_share"] == 1.0 and summary["sign_agreement_share"] == 1.0
    assert summary["mean_abs_difference"] == 0.0


def test_reversed_importance_gives_negative_rank_correlation():
    a = np.array([[4.0, 3.0, 2.0, 1.0]] * 5)
    b = np.array([[1.0, 2.0, 3.0, 4.0]] * 5)

    assert global_agreement(a, b)["spearman"] == pytest.approx(-1.0)
    np.testing.assert_allclose(instance_rank_correlations(a, b), -1.0)


def test_instance_correlation_is_undefined_for_a_constant_row():
    a = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]])

    rho = instance_rank_correlations(a, a)

    assert np.isnan(rho[0]) and rho[1] == pytest.approx(1.0)


def test_comparison_refuses_attributions_for_different_rows():
    values = np.ones((2, 4))

    with pytest.raises(ValueError, match="different rows"):
        summarize_agreement(_attribution(values, rows=[1, 2]), _attribution(values, rows=[1, 3]))


# --- Plots ----------------------------------------------------------------------------------


def test_phase4_plots_write_non_empty_files(tmp_path):
    from fin_inclusion.interpretability.plots import plot_convergence, plot_paired_shares, plot_unit_profiles

    names = ["a", "b", "c"]
    plot_convergence([32, 64, 128], {"modal": [0.1, 0.03, 0.01], "unknown": [0.05, 0.02, 0.008]},
                     tmp_path / "conv.png", "t", "x", "y", "caption.", threshold=0.01)
    plot_paired_shares({"p1": (names, {"x": np.array([0.2, 0.3, 0.5]), "y": np.array([0.3, 0.3, 0.4])}),
                        "p2": (names, {"x": np.array([0.1, 0.4, 0.5]), "y": np.array([0.2, 0.2, 0.6])})},
                       tmp_path / "paired.png", "t", "caption.")
    plot_unit_profiles({"hidden_1": (np.random.default_rng(0).random(20), np.zeros(20, dtype=bool))},
                       tmp_path / "units.png", "t", "caption.")

    for name in ("conv", "paired", "units"):
        assert (tmp_path / f"{name}.png").stat().st_size > 5000
    with pytest.raises(ValueError, match="same features"):
        plot_paired_shares({"p1": (names, {"x": np.ones(3)}), "p2": (["c", "b", "a"], {"x": np.ones(3)})},
                           tmp_path / "bad.png", "t", "caption.")
