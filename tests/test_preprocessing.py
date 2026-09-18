import numpy as np
import pandas as pd
import pytest

from fin_inclusion.preprocessing.base import UnseenCategoryWarning
from fin_inclusion.preprocessing.cleaning import (
    HOUSEHOLD_SIZE_OUTLIER_THRESHOLD,
    HouseholdSizeOutlierFlagger,
    MissingnessRecoder,
)
from fin_inclusion.preprocessing.encoders import EmbeddingIndexBranch, NumericScaler, OneHotBranch
from fin_inclusion.preprocessing.pipeline_factory import (
    build_pytorch_pipeline,
    build_xgboost_pipeline,
    make_country_target_stratify_key,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": ["Kenya", "Kenya", "Uganda", "Rwanda"],
            "location_type": ["Rural", "Urban", "Rural", "Urban"],
            "cellphone_access": ["Yes", "No", "Yes", "Yes"],
            "gender_of_respondent": ["Female", "Male", "Female", "Male"],
            "relationship_with_head": ["Spouse", "Head of Household", "Spouse", "Child"],
            "marital_status": ["Married/Living together", "Dont know", "Widowed", "Single/Never Married"],
            "education_level": ["Secondary education", "Primary education", "Other/Dont know/RTA", "Tertiary education"],
            "job_type": ["Self employed", "Farming and Fishing", "Dont Know/Refuse to answer", "Formally employed Private"],
            "household_size": [3, 5, 21, 2],
            "age_of_respondent": [24, 70, 59, 31],
            "bank_account": ["Yes", "No", "No", "Yes"],
        }
    )


class TestMissingnessRecoder:
    def test_recodes_known_placeholders_to_unknown(self):
        df = _sample_df()

        out = MissingnessRecoder().fit_transform(df)

        assert out.loc[1, "marital_status"] == "unknown"
        assert out.loc[2, "education_level"] == "unknown"
        assert out.loc[2, "job_type"] == "unknown"

    def test_leaves_non_placeholder_values_untouched(self):
        df = _sample_df()

        out = MissingnessRecoder().fit_transform(df)

        assert out.loc[0, "marital_status"] == "Married/Living together"
        assert out.loc[0, "job_type"] == "Self employed"

    def test_does_not_mutate_input(self):
        df = _sample_df()
        original = df.copy()

        MissingnessRecoder().fit_transform(df)

        pd.testing.assert_frame_equal(df, original)


class TestHouseholdSizeOutlierFlagger:
    def test_flags_rows_above_threshold_only(self):
        df = _sample_df()
        assert df.loc[2, "household_size"] > HOUSEHOLD_SIZE_OUTLIER_THRESHOLD

        out = HouseholdSizeOutlierFlagger().fit_transform(df)

        assert out["household_size_outlier"].tolist() == [False, False, True, False]

    def test_never_drops_rows(self):
        df = _sample_df()

        out = HouseholdSizeOutlierFlagger().fit_transform(df)

        assert len(out) == len(df)


class TestOneHotBranch:
    def test_encodes_known_categories(self):
        df = _sample_df()
        branch = OneHotBranch(columns=["country"]).fit(df)

        out = branch.transform(df)

        assert "country_Kenya" in out.columns
        assert out.loc[0, "country_Kenya"] == 1.0
        assert out.loc[2, "country_Kenya"] == 0.0
        assert "country" not in out.columns

    def test_passes_through_other_columns_unchanged(self):
        df = _sample_df()
        branch = OneHotBranch(columns=["country"]).fit(df)

        out = branch.transform(df)

        assert out["household_size"].tolist() == df["household_size"].tolist()

    def test_unseen_category_at_transform_warns_and_encodes_all_zero(self):
        train = _sample_df()
        branch = OneHotBranch(columns=["country"]).fit(train)
        unseen_row = train.iloc[[0]].copy()
        unseen_row["country"] = "Tanzania"

        with pytest.warns(UnseenCategoryWarning):
            out = branch.transform(unseen_row)

        country_cols = [c for c in out.columns if c.startswith("country_")]
        assert out.loc[0, country_cols].sum() == 0.0

    def test_fit_only_on_train_then_transform_test(self):
        train = _sample_df().iloc[:3]
        test = _sample_df().iloc[[3]]
        branch = OneHotBranch(columns=["country"]).fit(train)

        with pytest.warns(UnseenCategoryWarning):
            out = branch.transform(test)

        assert "country_Rwanda" not in out.columns


class TestEmbeddingIndexBranch:
    def test_indexes_start_at_one_reserving_zero_for_unknown(self):
        df = _sample_df()
        branch = EmbeddingIndexBranch(columns=["country"]).fit(df)

        out = branch.transform(df)

        assert out["country"].min() >= 1
        assert branch.vocab_sizes_["country"] == df["country"].nunique() + 1

    def test_unseen_category_maps_to_zero_and_warns(self):
        train = _sample_df().iloc[:3]
        branch = EmbeddingIndexBranch(columns=["country"]).fit(train)
        unseen_row = train.iloc[[0]].copy()
        unseen_row["country"] = "Rwanda"

        with pytest.warns(UnseenCategoryWarning):
            out = branch.transform(unseen_row)

        assert out.loc[0, "country"] == EmbeddingIndexBranch.UNKNOWN_INDEX

    def test_transform_is_deterministic_given_fit(self):
        df = _sample_df()
        branch = EmbeddingIndexBranch(columns=["country"]).fit(df)

        first = branch.transform(df)["country"].tolist()
        second = branch.transform(df)["country"].tolist()

        assert first == second


class TestNumericScaler:
    def test_standardizes_fit_data_to_zero_mean_unit_std(self):
        df = _sample_df()
        scaler = NumericScaler(columns=["age_of_respondent"]).fit(df)

        out = scaler.transform(df)

        assert out["age_of_respondent"].mean() == pytest.approx(0.0, abs=1e-9)
        assert out["age_of_respondent"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)

    def test_uses_train_fitted_stats_not_transform_data_stats(self):
        train = _sample_df().iloc[:2]
        test = _sample_df().iloc[2:]
        scaler = NumericScaler(columns=["age_of_respondent"]).fit(train)

        out = scaler.transform(test)

        assert out["age_of_respondent"].mean() != pytest.approx(0.0, abs=1e-9)

    def test_leaves_other_columns_untouched(self):
        df = _sample_df()
        scaler = NumericScaler(columns=["age_of_respondent"]).fit(df)

        out = scaler.transform(df)

        assert out["household_size"].tolist() == df["household_size"].tolist()


class TestPipelineFactory:
    def test_xgboost_pipeline_produces_one_hot_and_raw_numerics(self):
        df = _sample_df()
        pipeline = build_xgboost_pipeline()

        out = pipeline.fit_transform(df)

        assert "country_Kenya" in out.columns
        assert out["household_size"].tolist() == df["household_size"].tolist()
        assert "household_size_outlier" in out.columns

    def test_pytorch_pipeline_produces_indexed_categoricals_and_scaled_numerics(self):
        df = _sample_df()
        pipeline = build_pytorch_pipeline()

        out = pipeline.fit_transform(df)

        assert np.issubdtype(out["country"].dtype, np.integer)
        assert out["age_of_respondent"].mean() == pytest.approx(0.0, abs=1e-9)

    def test_pipelines_do_not_drop_or_reorder_rows(self):
        df = _sample_df()

        xgb_out = build_xgboost_pipeline().fit_transform(df)
        pt_out = build_pytorch_pipeline().fit_transform(df)

        assert len(xgb_out) == len(df)
        assert len(pt_out) == len(df)


class TestStratifyKey:
    def test_combines_country_and_target(self):
        df = _sample_df()

        key = make_country_target_stratify_key(df, target_col="bank_account")

        assert key.tolist() == ["Kenya_Yes", "Kenya_No", "Uganda_No", "Rwanda_Yes"]

    def test_no_target_falls_back_to_country_only(self):
        df = _sample_df()

        key = make_country_target_stratify_key(df, target_col=None)

        assert key.tolist() == df["country"].tolist()
