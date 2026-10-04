import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fin_inclusion.interpretability.model_adapters import (
    ArtifactError,
    MLPRawModel,
    RawFeatureModel,
    XGBoostRawModel,
)
from fin_inclusion.interpretability.sampling import (
    SampleSets,
    build_sample_sets,
    composition_table,
    stratified_positions,
)
from fin_inclusion.models.xgboost_model import XGBoostModel
from fin_inclusion.preprocessing.cleaning import (
    DISGUISED_MISSING_VALUES,
    HOUSEHOLD_SIZE_OUTLIER_THRESHOLD,
    UNKNOWN_LABEL,
)
from fin_inclusion.preprocessing.pipeline_factory import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_pytorch_pipeline,
    build_xgboost_pipeline,
)
from fin_inclusion.serving.encoder import FeatureEncoder, UnknownCategoryError

RAW_FEATURES = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
LEVELS = {
    "country": ["Kenya", "Rwanda", "Tanzania", "Uganda"],
    "location_type": ["Rural", "Urban"],
    "cellphone_access": ["No", "Yes"],
    "gender_of_respondent": ["Female", "Male"],
    "relationship_with_head": ["Child", "Head of Household", "Spouse"],
    "marital_status": ["Dont know", "Married/Living together", "Single/Never Married"],
    "education_level": ["No formal education", "Other/Dont know/RTA", "Primary education"],
    "job_type": ["Dont Know/Refuse to answer", "Farming and Fishing", "Self employed"],
}


def _synthetic_survey(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Raw-schema rows with every level present, an outlier household, and a
    target that depends on the features so the fitted models are not constant."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.choice(levels, size=n) for c, levels in LEVELS.items()})
    frame["household_size"] = rng.integers(1, 10, size=n)
    frame.loc[0, "household_size"] = 18
    frame["age_of_respondent"] = rng.integers(16, 90, size=n)
    signal = (frame["cellphone_access"] == "Yes").astype(int) + (frame["location_type"] == "Urban").astype(int)
    frame["bank_account"] = np.where(signal + rng.normal(scale=0.6, size=n) > 1.0, "Yes", "No")
    frame["uniqueid"] = [f"uniqueid_{i}" for i in range(n)]
    frame["year"] = 2018
    return frame


def _features(encoded: pd.DataFrame) -> pd.DataFrame:
    return encoded.drop(columns=["uniqueid", "year", "bank_account"])


# --- XGBoost adapter -------------------------------------------------------


@pytest.fixture(scope="module")
def xgb_artifact(tmp_path_factory):
    """A tiny model written in the production artifact's format."""
    train = _synthetic_survey()
    X = _features(build_xgboost_pipeline().fit_transform(train))
    model = XGBoostModel(seed=0, n_estimators=20, max_depth=3).fit(X, train["bank_account"])

    model_dir = tmp_path_factory.mktemp("xgb_artifact")
    model.save(model_dir / "model.json")
    metadata = {
        "model_version": "xgb-test",
        "decision_threshold": 0.4,
        "feature_columns": list(X.columns),
        "numeric_columns": NUMERIC_COLUMNS,
        "raw_categorical_levels": {c: sorted(train[c].unique().tolist()) for c in CATEGORICAL_COLUMNS},
        "disguised_missing_values": DISGUISED_MISSING_VALUES,
        "unknown_label": UNKNOWN_LABEL,
        "household_size_outlier_threshold": HOUSEHOLD_SIZE_OUTLIER_THRESHOLD,
        "model_sha256": hashlib.sha256((model_dir / "model.json").read_bytes()).hexdigest(),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return model_dir, train, X, model


def test_xgboost_adapter_reproduces_the_pipeline_encoding(xgb_artifact):
    model_dir, train, X, _ = xgb_artifact
    adapter = XGBoostRawModel.load(model_dir)

    encoded = adapter.encode(train[RAW_FEATURES])

    np.testing.assert_array_equal(encoded, X.to_numpy(dtype=np.float32))
    assert adapter.raw_features == tuple(RAW_FEATURES)


def test_xgboost_adapter_encoding_matches_the_serving_encoder(xgb_artifact):
    model_dir, train, _, _ = xgb_artifact
    adapter = XGBoostRawModel.load(model_dir)
    records = train[RAW_FEATURES].head(50).to_dict(orient="records")

    serving = FeatureEncoder.from_metadata(adapter.metadata).encode_batch(records)

    np.testing.assert_array_equal(adapter.encode(train[RAW_FEATURES].head(50)), serving)


def test_xgboost_adapter_logit_is_the_logit_of_the_models_score(xgb_artifact):
    model_dir, train, X, model = xgb_artifact
    adapter = XGBoostRawModel.load(model_dir)

    logit = adapter.predict_logit(train[RAW_FEATURES])
    score = model.predict_proba(X)

    assert logit.shape == (len(train),)
    np.testing.assert_allclose(1 / (1 + np.exp(-logit)), score, atol=1e-6)
    np.testing.assert_allclose(adapter.predict_score(train[RAW_FEATURES]), score, atol=1e-6)
    assert logit.std() > 0  # a constant model would make every later check vacuous


def test_xgboost_adapter_rejects_a_tampered_model_file(xgb_artifact, tmp_path):
    model_dir, *_ = xgb_artifact
    (tmp_path / "model.json").write_bytes((model_dir / "model.json").read_bytes() + b" ")
    (tmp_path / "metadata.json").write_bytes((model_dir / "metadata.json").read_bytes())

    with pytest.raises(ArtifactError, match="checksum"):
        XGBoostRawModel.load(tmp_path)


def test_xgboost_adapter_reports_a_missing_artifact(tmp_path):
    with pytest.raises(ArtifactError, match="Missing artifact"):
        XGBoostRawModel.load(tmp_path)


def test_adapters_refuse_an_unseen_category_instead_of_encoding_it_as_unknown(xgb_artifact):
    model_dir, train, _, _ = xgb_artifact
    adapter = XGBoostRawModel.load(model_dir)
    bad = train[RAW_FEATURES].head(3).copy()
    bad.loc[1, "country"] = "uganda"

    with pytest.raises(UnknownCategoryError, match="uganda"):
        adapter.predict_logit(bad)


def test_adapters_report_a_missing_raw_column(xgb_artifact):
    model_dir, train, _, _ = xgb_artifact
    adapter = XGBoostRawModel.load(model_dir)

    with pytest.raises(KeyError, match="job_type"):
        adapter.predict_logit(train[RAW_FEATURES].drop(columns="job_type"))


def test_shipped_xgboost_artifact_loads_and_scores_through_the_adapter():
    model_dir = Path(__file__).resolve().parent.parent / "models" / "production"
    if not (model_dir / "model.json").exists():
        pytest.skip("models/production missing -- run scripts/train_production_model.py")
    adapter = XGBoostRawModel.load(model_dir)
    record = {c: levels[0] for c, levels in adapter.categorical_levels.items()} | {
        "household_size": 4, "age_of_respondent": 30
    }

    logit = adapter.predict_logit(pd.DataFrame([record]))

    assert isinstance(adapter, RawFeatureModel)
    assert len(adapter.raw_features) == 10
    assert np.isfinite(logit).all()


# --- MLP adapter -----------------------------------------------------------


@pytest.fixture(scope="module")
def mlp_fitted():
    pytest.importorskip("torch", reason="PyTorch is required for the MLP adapter")
    from fin_inclusion.models.pytorch_mlp import PyTorchMLP

    train = _synthetic_survey()
    pipeline = build_pytorch_pipeline().fit(train)
    X = _features(pipeline.transform(train))
    numeric_inputs = NUMERIC_COLUMNS + ["household_size_outlier"]
    model = PyTorchMLP(
        categorical_columns=CATEGORICAL_COLUMNS,
        vocab_sizes=pipeline.named_steps["encode"].vocab_sizes_,
        numeric_columns=numeric_inputs,
        hidden_dims=(8, 4),
        embedding_dim=3,
        dropout=0.3,
        epochs=3,
        batch_size=32,
        seed=0,
    ).fit(X, train["bank_account"])
    adapter = MLPRawModel.from_fitted(model, pipeline, decision_threshold=0.6, metadata={"seed": 0})
    return adapter, model, train, X, numeric_inputs


def test_mlp_adapter_reproduces_the_pipeline_encoding(mlp_fitted):
    adapter, _, train, X, numeric_inputs = mlp_fitted

    cat, num = adapter.encode(train[RAW_FEATURES])

    np.testing.assert_array_equal(cat, X[CATEGORICAL_COLUMNS].to_numpy(dtype=np.int64))
    np.testing.assert_array_equal(num, X[numeric_inputs].to_numpy(dtype=np.float32))
    assert adapter.raw_features == tuple(RAW_FEATURES)


def test_mlp_adapter_maps_placeholder_answers_to_the_unknown_level(mlp_fitted):
    adapter, *_ = mlp_fitted
    maps = adapter.preprocessing["category_maps"]

    # the raw placeholder is a level a respondent can give; "unknown" is only its cleaned name
    assert "Dont know" in adapter.categorical_levels["marital_status"]
    assert UNKNOWN_LABEL not in adapter.categorical_levels["marital_status"]
    assert adapter._index["marital_status"]["Dont know"] == maps["marital_status"][UNKNOWN_LABEL]
    # index 0 stays reserved for never-seen categories: no raw level maps to it
    assert all(index != 0 for table in adapter._index.values() for index in table.values())


def test_mlp_adapter_logit_matches_the_model_and_is_deterministic(mlp_fitted):
    adapter, model, train, X, _ = mlp_fitted

    logit = adapter.predict_logit(train[RAW_FEATURES])

    np.testing.assert_allclose(adapter.predict_score(train[RAW_FEATURES]), model.predict_proba(X), atol=1e-6)
    # dropout is 0.3 here; two calls agreeing means the net is scored in eval mode
    np.testing.assert_array_equal(logit, adapter.predict_logit(train[RAW_FEATURES]))
    assert logit.dtype == np.float64 and logit.std() > 0


def test_mlp_adapter_save_load_round_trip_is_exact(mlp_fitted, tmp_path):
    adapter, _, train, _, _ = mlp_fitted

    metadata = adapter.save(tmp_path)
    loaded = MLPRawModel.load(tmp_path)

    np.testing.assert_array_equal(loaded.predict_logit(train[RAW_FEATURES]), adapter.predict_logit(train[RAW_FEATURES]))
    assert loaded.decision_threshold == 0.6
    assert loaded.categorical_levels == adapter.categorical_levels
    assert metadata["weights_sha256"] == hashlib.sha256((tmp_path / "mlp.pt").read_bytes()).hexdigest()
    assert metadata["seed"] == 0


def test_mlp_adapter_refuses_weights_that_do_not_match_the_metadata(mlp_fitted, tmp_path):
    adapter, *_ = mlp_fitted
    adapter.save(tmp_path)
    metadata = json.loads((tmp_path / "mlp_metadata.json").read_text(encoding="utf-8"))
    metadata["weights_sha256"] = "0" * 64
    (tmp_path / "mlp_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match="checksum"):
        MLPRawModel.load(tmp_path)


def test_mlp_adapter_refuses_an_unseen_category(mlp_fitted):
    adapter, _, train, _, _ = mlp_fitted
    bad = train[RAW_FEATURES].head(2).copy()
    bad.loc[0, "job_type"] = "Astronaut"

    with pytest.raises(UnknownCategoryError, match="Astronaut"):
        adapter.predict_logit(bad)


# --- Sample sets -----------------------------------------------------------


def _population(n: int = 2000, seed: int = 3) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    country = pd.Series(rng.choice(["Kenya", "Rwanda", "Tanzania", "Uganda"], size=n, p=[0.3, 0.35, 0.25, 0.1]))
    target = pd.Series(np.where(rng.random(n) < 0.15, "Yes", "No"))
    return country, target


def test_stratified_positions_are_proportional_sorted_and_reproducible():
    country, target = _population()
    strata = country + "_" + target

    chosen = stratified_positions(strata, n=200, seed=42)

    assert chosen.size == 200 and np.unique(chosen).size == 200
    assert (np.diff(chosen) > 0).all()
    np.testing.assert_array_equal(chosen, stratified_positions(strata, n=200, seed=42))
    assert not np.array_equal(chosen, stratified_positions(strata, n=200, seed=7))
    population_share = strata.value_counts(normalize=True)
    sample_share = strata.iloc[chosen].value_counts(normalize=True)
    assert set(sample_share.index) == set(population_share.index)  # no stratum dropped
    assert (sample_share - population_share).abs().max() < 0.01


def test_stratified_positions_honour_exclusions_and_reject_impossible_sizes():
    country, target = _population()
    strata = country + "_" + target
    excluded = np.arange(0, 1000)

    chosen = stratified_positions(strata, n=100, seed=42, exclude=excluded)

    assert np.intersect1d(chosen, excluded).size == 0
    with pytest.raises(ValueError, match="Cannot draw"):
        stratified_positions(strata, n=len(strata), seed=42)


def test_sample_sets_keep_background_and_pooled_disjoint_and_include_every_focus_row():
    country, target = _population()

    sets = build_sample_sets(country, target, background_size=100, explanation_sample_size=300, seed=42,
                             training_data_sha256="abc")

    assert sets.background.size == 100 and sets.pooled.size == 300
    assert np.intersect1d(sets.background, sets.pooled).size == 0
    np.testing.assert_array_equal(sets.focus, np.flatnonzero(country.to_numpy() == "Uganda"))
    assert set(sets.pooled) <= set(sets.explanation) and set(sets.focus) <= set(sets.explanation)
    assert sets.explanation.size == np.union1d(sets.pooled, sets.focus).size


def test_sample_sets_reject_a_focus_country_with_no_rows():
    country, target = _population()

    with pytest.raises(ValueError, match="Burundi"):
        build_sample_sets(country, target, 100, 300, seed=42, training_data_sha256="abc", focus_country="Burundi")


def test_sample_sets_round_trip_and_are_bound_to_their_training_file(tmp_path):
    country, target = _population()
    sets = build_sample_sets(country, target, 100, 300, seed=42, training_data_sha256="abc")
    path = tmp_path / "xai" / "sample_sets.json"

    sets.save(path, composition={"background": composition_table(sets.background, country, target)})
    loaded = SampleSets.load(path, training_data_sha256="abc")

    for name in ("background", "pooled", "focus"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(sets, name))
    assert (loaded.seed, loaded.focus_country) == (42, "Uganda")
    with pytest.raises(ValueError, match="different training file"):
        SampleSets.load(path, training_data_sha256="not-the-same-file")


def test_composition_table_counts_rows_per_country_and_target():
    country = pd.Series(["Kenya", "Kenya", "Uganda", "Uganda", "Uganda"])
    target = pd.Series(["Yes", "No", "No", "No", "Yes"])

    table = composition_table(np.array([0, 2, 3, 4]), country, target)

    assert table == {"Kenya": {"No": 0, "Yes": 1}, "Uganda": {"No": 2, "Yes": 1}}
