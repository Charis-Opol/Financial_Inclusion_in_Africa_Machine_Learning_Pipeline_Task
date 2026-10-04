import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fin_inclusion.interpretability.attribution import LOGIT_SPACE, Attribution
from fin_inclusion.interpretability.base import Explainer
from fin_inclusion.interpretability.checks import EfficiencyError, check_efficiency, efficiency_residual
from fin_inclusion.interpretability.grouping import (
    FeatureGrouping,
    group_attribution,
    importance_share,
    mean_abs_importance,
)

CATEGORICALS = ["country", "job_type"]
NUMERICS = ["household_size", "age_of_respondent"]
ONEHOT_COLUMNS = [
    "household_size",
    "age_of_respondent",
    "household_size_outlier",
    "country_Kenya",
    "country_Uganda",
    "job_type_Self employed",
    "job_type_unknown",
]
RAW_FEATURES = ("country", "job_type", "household_size", "age_of_respondent")


def _attribution(values, base_value=0.5, feature_names=("a", "b", "c"), **overrides) -> Attribution:
    fields = dict(
        values=np.asarray(values, dtype=float),
        base_value=base_value,
        feature_names=feature_names,
        method="synthetic",
        model_name="toy",
        target_space=LOGIT_SPACE,
        baseline_description="all-zero input; interventional",
    )
    return Attribution(**(fields | overrides))


# --- Attribution -----------------------------------------------------------


def test_attribution_exposes_shape_and_broadcasts_scalar_base_value():
    attribution = _attribution([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert (attribution.n_instances, attribution.n_features) == (2, 3)
    np.testing.assert_array_equal(attribution.base_values, [0.5, 0.5])


def test_attribution_accepts_per_instance_base_value():
    attribution = _attribution([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], base_value=[0.1, 0.2])

    np.testing.assert_array_equal(attribution.base_values, [0.1, 0.2])


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(values=[1.0, 2.0, 3.0]), "2-D"),
        (dict(feature_names=("a", "b")), "feature names"),
        (dict(feature_names=("a", "a", "b")), "not unique"),
        (dict(values=[[1.0, np.nan, 3.0]]), "NaN/Inf"),
        (dict(base_value=[0.1, 0.2, 0.3]), "base_value"),
        (dict(base_value=np.inf), "base_value"),
        (dict(baseline_description=""), "baseline_description"),
        (dict(target_space=""), "target_space"),
    ],
)
def test_attribution_rejects_malformed_input(overrides, message):
    fields = dict(values=[[1.0, 2.0, 3.0]]) | overrides
    with pytest.raises(ValueError, match=message):
        _attribution(**fields)


def test_select_restricts_rows_and_per_instance_base_values():
    attribution = _attribution([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]], base_value=[0.1, 0.2, 0.3])

    subset = attribution.select(np.array([True, False, True]))

    np.testing.assert_array_equal(subset.values[:, 0], [1.0, 3.0])
    np.testing.assert_array_equal(subset.base_values, [0.1, 0.3])
    assert subset.method == attribution.method


def test_save_load_round_trip_preserves_everything(tmp_path):
    attribution = _attribution(
        [[1.5, -2.0, 0.25], [0.0, 3.0, -1.0]],
        base_value=[0.1, -0.2],
        metadata={"seed": 42, "background_rows": [3, 7]},
    )
    path = tmp_path / "nested" / "attribution.npz"

    attribution.save(path)
    loaded = Attribution.load(path)

    np.testing.assert_array_equal(loaded.values, attribution.values)
    np.testing.assert_array_equal(loaded.base_values, attribution.base_values)
    assert loaded.feature_names == attribution.feature_names
    assert loaded.method == attribution.method
    assert loaded.model_name == attribution.model_name
    assert loaded.target_space == attribution.target_space
    assert loaded.baseline_description == attribution.baseline_description
    assert loaded.metadata == attribution.metadata


def test_save_rejects_metadata_that_is_not_json(tmp_path):
    attribution = _attribution([[1.0, 2.0, 3.0]], metadata={"array": np.arange(3)})

    with pytest.raises(TypeError, match="JSON-serialisable"):
        attribution.save(tmp_path / "attribution.npz")


def test_to_frame_uses_feature_names_as_columns():
    frame = _attribution([[1.0, 2.0, 3.0]]).to_frame(index=pd.Index(["row_7"]))

    assert list(frame.columns) == ["a", "b", "c"]
    assert frame.loc["row_7", "b"] == 2.0


# --- Explainer -------------------------------------------------------------


def test_explainer_cannot_be_instantiated_without_explain():
    with pytest.raises(TypeError):
        Explainer()


def test_explainer_subclass_is_usable_through_the_interface():
    class ConstantExplainer(Explainer):
        def explain(self, X_raw: pd.DataFrame) -> Attribution:
            return _attribution(np.ones((len(X_raw), 3)))

    explainer: Explainer = ConstantExplainer()

    assert explainer.explain(pd.DataFrame({"x": [1, 2]})).n_instances == 2


# --- Grouping --------------------------------------------------------------


def test_onehot_grouping_maps_levels_numerics_and_the_derived_flag():
    grouping = FeatureGrouping.for_onehot(ONEHOT_COLUMNS, CATEGORICALS, NUMERICS)

    assert grouping.raw_features == RAW_FEATURES
    assert grouping.columns_of("country") == ["country_Kenya", "country_Uganda"]
    assert grouping.columns_of("job_type") == ["job_type_Self employed", "job_type_unknown"]
    # the outlier flag is a function of household_size, not an 11th feature
    assert grouping.columns_of("household_size") == ["household_size", "household_size_outlier"]
    assert grouping.columns_of("age_of_respondent") == ["age_of_respondent"]


def test_onehot_grouping_prefers_the_longest_matching_categorical():
    grouping = FeatureGrouping.for_onehot(
        ["job_A", "job_type_A", "x"], categorical_columns=["job", "job_type"], numeric_columns=["x"]
    )

    assert grouping.columns_of("job") == ["job_A"]
    assert grouping.columns_of("job_type") == ["job_type_A"]


def test_onehot_grouping_rejects_an_unmapped_column():
    with pytest.raises(ValueError, match="mystery_column"):
        FeatureGrouping.for_onehot([*ONEHOT_COLUMNS, "mystery_column"], CATEGORICALS, NUMERICS)


def test_grouping_rejects_a_raw_feature_with_no_columns():
    without_country = [c for c in ONEHOT_COLUMNS if not c.startswith("country_")]

    with pytest.raises(ValueError, match="country"):
        FeatureGrouping.for_onehot(without_country, CATEGORICALS, NUMERICS)


def test_embedding_grouping_matches_the_mlp_input_layout():
    grouping = FeatureGrouping.for_embeddings(
        CATEGORICALS, embedding_dim=3, numeric_inputs=[*NUMERICS, "household_size_outlier"]
    )

    assert grouping.raw_features == RAW_FEATURES
    assert grouping.n_encoded == 2 * 3 + 3
    assert grouping.columns_of("job_type") == ["job_type[emb0]", "job_type[emb1]", "job_type[emb2]"]
    assert grouping.columns_of("household_size") == ["household_size", "household_size_outlier"]
    # embeddings first in categorical order, then the numeric inputs in model order
    np.testing.assert_array_equal(grouping.group_index, [0, 0, 0, 1, 1, 1, 2, 3, 2])


def test_aggregate_sums_columns_within_each_raw_feature():
    grouping = FeatureGrouping.for_onehot(ONEHOT_COLUMNS, CATEGORICALS, NUMERICS)
    #         hh   age  flag  KE   UG   self unk
    values = [[1.0, 2.0, 0.5, 3.0, -1.0, 4.0, 0.25]]

    grouped = grouping.aggregate(values)

    # order: country, job_type, household_size, age_of_respondent
    np.testing.assert_allclose(grouped, [[2.0, 4.25, 1.5, 2.0]])


def test_aggregate_rejects_the_wrong_number_of_columns():
    grouping = FeatureGrouping.for_onehot(ONEHOT_COLUMNS, CATEGORICALS, NUMERICS)

    with pytest.raises(ValueError, match="shape"):
        grouping.aggregate(np.zeros((2, 3)))


def test_group_attribution_preserves_row_totals_and_base_value():
    rng = np.random.default_rng(0)
    grouping = FeatureGrouping.for_onehot(ONEHOT_COLUMNS, CATEGORICALS, NUMERICS)
    encoded = _attribution(rng.normal(size=(20, len(ONEHOT_COLUMNS))), feature_names=tuple(ONEHOT_COLUMNS))

    grouped = group_attribution(encoded, grouping)

    assert grouped.feature_names == RAW_FEATURES
    assert grouped.base_value == encoded.base_value
    np.testing.assert_allclose(grouped.values.sum(axis=1), encoded.values.sum(axis=1))
    assert grouped.metadata["grouped_from_n_encoded_columns"] == len(ONEHOT_COLUMNS)


def test_group_attribution_rejects_misaligned_columns():
    grouping = FeatureGrouping.for_onehot(ONEHOT_COLUMNS, CATEGORICALS, NUMERICS)
    shuffled = tuple(reversed(ONEHOT_COLUMNS))
    encoded = _attribution(np.zeros((1, len(shuffled))), feature_names=shuffled)

    with pytest.raises(ValueError, match="do not match"):
        group_attribution(encoded, grouping)


def test_shipped_model_columns_group_into_the_ten_raw_features():
    metadata_path = Path(__file__).resolve().parent.parent / "models" / "production" / "metadata.json"
    if not metadata_path.exists():
        pytest.skip("models/production/metadata.json missing -- run scripts/train_production_model.py")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    grouping = FeatureGrouping.for_onehot(
        metadata["feature_columns"], list(metadata["raw_categorical_levels"]), metadata["numeric_columns"]
    )

    assert len(grouping.raw_features) == 10
    assert grouping.n_encoded == 40
    for column, levels in metadata["raw_categorical_levels"].items():
        assert len(grouping.columns_of(column)) == len(levels)


# --- Importance ------------------------------------------------------------


def test_mean_abs_importance_ignores_sign_and_keeps_feature_order():
    attribution = _attribution([[1.0, -4.0, 0.0], [-3.0, 2.0, 0.0]])

    importance = mean_abs_importance(attribution)

    assert list(importance.index) == ["a", "b", "c"]
    np.testing.assert_allclose(importance.to_numpy(), [2.0, 3.0, 0.0])


def test_importance_share_sums_to_one_and_is_scale_invariant():
    importance = pd.Series([2.0, 3.0, 5.0], index=["a", "b", "c"])

    share = importance_share(importance)

    assert share.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(share.to_numpy(), [0.2, 0.3, 0.5])
    pd.testing.assert_series_equal(importance_share(importance * 1000), share)


@pytest.mark.parametrize("values", [[0.0, 0.0], [1.0, -1.0], [1.0, np.nan]])
def test_importance_share_rejects_degenerate_input(values):
    with pytest.raises(ValueError):
        importance_share(pd.Series(values))


# --- Efficiency ------------------------------------------------------------


def test_efficiency_holds_for_an_exactly_additive_explanation():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(30, 3))
    attribution = _attribution(values, base_value=-1.25)
    model_output = values.sum(axis=1) - 1.25

    report = check_efficiency(attribution, model_output, atol=1e-9)

    assert report.passed
    assert report.n_violations == 0
    assert report.max_abs_error < 1e-12
    report.raise_if_failed()


def test_efficiency_counts_and_reports_violations():
    attribution = _attribution([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], base_value=0.0)
    model_output = np.array([3.0, 3.5])  # second instance is off by 0.5

    report = check_efficiency(attribution, model_output, atol=1e-6)

    assert not report.passed
    assert report.n_violations == 1
    assert report.max_abs_error == pytest.approx(0.5)
    assert report.mean_abs_error == pytest.approx(0.25)
    assert report.to_dict()["passed"] is False
    with pytest.raises(EfficiencyError, match="1/2 instances"):
        report.raise_if_failed()


def test_efficiency_uses_per_instance_base_values():
    attribution = _attribution([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], base_value=[0.0, 10.0])

    np.testing.assert_allclose(efficiency_residual(attribution, np.array([1.0, 11.0])), [0.0, 0.0])


def test_efficiency_is_preserved_by_grouping():
    rng = np.random.default_rng(2)
    grouping = FeatureGrouping.for_onehot(ONEHOT_COLUMNS, CATEGORICALS, NUMERICS)
    values = rng.normal(size=(15, len(ONEHOT_COLUMNS)))
    encoded = _attribution(values, base_value=0.3, feature_names=tuple(ONEHOT_COLUMNS))
    model_output = values.sum(axis=1) + 0.3

    assert check_efficiency(group_attribution(encoded, grouping), model_output, atol=1e-9).passed


def test_efficiency_rejects_misaligned_output_and_missing_tolerance():
    attribution = _attribution([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="shape"):
        check_efficiency(attribution, np.zeros(2), atol=1e-6)
    with pytest.raises(ValueError, match="atol"):
        check_efficiency(attribution, np.zeros(1), atol=0.0)
