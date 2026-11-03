"""Tests of the shared harness: same folds for every arm, fold-only fitting, resumable runs."""
import json

import numpy as np
import pandas as pd
import pytest

from arms import Arm, build_arms
from harness import (
    Prepared,
    RunConfig,
    class_weights,
    country_label_key,
    country_weights,
    fold_paths,
    group_ids,
    loco_folds,
    outer_folds,
    prepare_matrix,
    prepare_mlp,
    prepare_xgb,
    run_arm,
    selection_score,
    within_country_class_weights,
)


def _frame(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "country": np.resize(["Kenya", "Rwanda", "Tanzania", "Uganda"], n),
            "year": 2018,
            "uniqueid": [f"id_{i}" for i in range(n)],
            "bank_account": np.where(rng.random(n) < 0.3, "Yes", "No"),
            "location_type": rng.choice(["Rural", "Urban"], n),
            "cellphone_access": rng.choice(["Yes", "No"], n),
            "household_size": rng.integers(1, 10, n),
            "age_of_respondent": rng.integers(16, 90, n),
            "gender_of_respondent": rng.choice(["Male", "Female"], n),
            "relationship_with_head": rng.choice(["Head of Household", "Spouse"], n),
            "marital_status": rng.choice(["Single/Never Married", "Divorced/Seperated"], n),
            "education_level": rng.choice(["Primary education", "Secondary education"], n),
            "job_type": rng.choice(["Self employed", "No Income"], n),
        }
    )


def test_outer_folds_are_the_same_on_every_call_and_cover_every_row_once():
    frame = _frame()
    first = [val.tolist() for _, _, val in outer_folds(frame, seed=3)]
    second = [val.tolist() for _, _, val in outer_folds(frame, seed=3)]
    assert first == second
    assert sorted(sum(first, [])) == list(range(len(frame)))
    assert first != [val.tolist() for _, _, val in outer_folds(frame, seed=4)]


def test_outer_folds_match_the_audited_sizes_and_never_overlap():
    frame = _frame()
    for _, train_idx, val_idx in outer_folds(frame, seed=0):
        assert not set(train_idx) & set(val_idx)
        assert len(train_idx) + len(val_idx) == len(frame)


def test_loco_holds_out_one_whole_country():
    frame = _frame()
    folds = list(loco_folds(frame))
    assert len(folds) == 4
    for name, train_idx, val_idx in folds:
        held = name.removeprefix("heldout_")
        assert set(frame.iloc[val_idx]["country"]) == {held}
        assert held not in set(frame.iloc[train_idx]["country"])


@pytest.mark.parametrize("prepare", [prepare_xgb, prepare_mlp])
def test_country_is_removed_when_use_country_is_false(prepare):
    frame = _frame()
    prepared = prepare(frame.iloc[:300], frame.iloc[300:], use_country=False)
    for matrix in (prepared.X_train, prepared.X_val):
        assert not [c for c in matrix.columns if c.startswith("country")]


def test_matrix_for_tabpfn_drops_country_and_has_categoricals_first():
    frame = _frame()
    prepared = prepare_matrix(frame.iloc[:300], frame.iloc[300:], use_country=False)
    assert prepared.X_train.shape[1] == 10  # 7 categoricals, 2 numerics, 1 outlier flag.
    assert len(prepared.categorical_columns) == 7
    assert "country" not in prepared.categorical_columns


def test_preprocessing_learns_from_the_training_part_only():
    frame = _frame()
    frame.loc[300:, "age_of_respondent"] = 1000  # An extreme value only in the validation rows.
    frame.loc[300:, "job_type"] = "Level only in validation"
    with pytest.warns():
        prepared = prepare_mlp(frame.iloc[:300], frame.iloc[300:], use_country=True)
    # Scaling learned from training rows has mean 0 there and leaves validation far out.
    assert abs(prepared.X_train["age_of_respondent"].mean()) < 1e-6
    assert prepared.X_val["age_of_respondent"].min() > 10
    assert (prepared.X_val["job_type"] == 0).all()  # The unknown bucket.


def test_prepared_holds_no_validation_labels():
    frame = _frame()
    prepared = prepare_mlp(frame.iloc[:300], frame.iloc[300:].drop(columns=["bank_account"]), use_country=True)
    assert "bank_account" not in prepared.X_val.columns
    assert "bank_account" not in prepared.X_train.columns
    assert len(prepared.y_train) == 300


def test_country_weights_give_each_country_equal_total_weight_with_mean_one():
    country = pd.Series(["A"] * 90 + ["B"] * 10)
    weights = country_weights(country)
    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(weights[:90].sum(), weights[90:].sum())


def test_class_weights_match_the_baseline_strategy():
    from fin_inclusion.imbalance.strategies import ClassWeightStrategy

    y = pd.Series(["Yes"] * 10 + ["No"] * 90)
    assert np.allclose(class_weights(y), ClassWeightStrategy().apply(pd.DataFrame({"x": range(100)}), y)[2])


def test_group_ids_are_contiguous_for_both_definitions():
    country = pd.Series(["B", "A", "B", "C"])
    assert sorted(set(group_ids(country).tolist())) == [0, 1, 2]
    y = pd.Series(["Yes", "No", "No", "No"])
    assert sorted(set(group_ids(country, y).tolist())) == [0, 1, 2, 3]


def test_worst_country_selection_returns_the_lowest_country():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    country = np.array(["A"] * 4 + ["B"] * 4)
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8])  # Perfect in A, inverted in B.
    from sklearn.metrics import average_precision_score

    assert selection_score(scores, y, country, "worst_country") == pytest.approx(
        average_precision_score(y[4:], scores[4:])
    )


class _CountingArm(Arm):
    """No real model: scores are random. Counts fits, and records what labels it was shown."""

    name = "counting"
    selection = None
    fits = 0
    seen_val_columns: list = []

    def prepare(self, train, val, use_country, context):
        type(self).seen_val_columns.append(list(val.columns))
        return Prepared(X_train=train, X_val=val, y_train=train["bank_account"], country_train=train["country"])

    def fit_predict(self, params, prepared, config, seed, tuning):
        type(self).fits += 1
        return np.random.default_rng(seed).random(len(prepared.X_val)), {}


def test_finished_folds_are_skipped_on_rerun(tmp_path):
    frame = _frame(200)
    config = RunConfig(n_outer=2, output_dir=tmp_path / "oof", cache_dir=tmp_path / "cache")
    _CountingArm.fits = 0
    run_arm(_CountingArm(), "nested", 0, frame, config)
    assert _CountingArm.fits == 2
    run_arm(_CountingArm(), "nested", 0, frame, config)
    assert _CountingArm.fits == 2

    # A fold with no metadata counts as unfinished and is run again.
    _, meta = fold_paths(config, "nested", "counting", 0, "fold1")
    meta.unlink()
    run_arm(_CountingArm(), "nested", 0, frame, config)
    assert _CountingArm.fits == 3


def test_outputs_cover_each_row_once_and_val_rows_reach_the_arm_without_labels(tmp_path):
    frame = _frame(200)
    config = RunConfig(n_outer=2, output_dir=tmp_path / "oof", cache_dir=tmp_path / "cache")
    _CountingArm.seen_val_columns = []
    run_arm(_CountingArm(), "nested", 0, frame, config)
    assert all("bank_account" not in columns for columns in _CountingArm.seen_val_columns)

    rows = []
    for fold in ("fold0", "fold1"):
        scores_path, meta_path = fold_paths(config, "nested", "counting", 0, fold)
        output = pd.read_csv(scores_path)
        rows.extend(output["row_id"].tolist())
        assert (output["y"].to_numpy() == (frame.loc[output["row_id"], "bank_account"] == "Yes").to_numpy()).all()
        assert json.loads(meta_path.read_text(encoding="utf-8"))["seed"] == 0
    assert sorted(rows) == list(range(len(frame)))


def test_sweep_arm_takes_parent_params_and_changes_only_its_override(tmp_path):
    config = RunConfig(output_dir=tmp_path / "oof", cache_dir=tmp_path / "cache")
    arms = build_arms()
    sweep = arms["mlp_dro_step_0.1"]
    with pytest.raises(FileNotFoundError):
        sweep.inherited_params(config, "nested", 0, "fold0")
    _, meta = fold_paths(config, "nested", "mlp_dro", 0, "fold0")
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"params": {"lr": 0.002, "step_size": 0.5, "adjustment_c": 1.0}}), encoding="utf-8")
    assert sweep.inherited_params(config, "nested", 0, "fold0") == {"lr": 0.002, "step_size": 0.1, "adjustment_c": 1.0}


def test_every_mlp_arm_in_e3_selects_on_the_same_criterion():
    """D0.4: selection must not differ between ERM, reweighted and Group DRO in E3."""
    arms = build_arms()
    names = ("mlp_erm_wc", "mlp_rw", "mlp_dro", "mlp_erm_within_cw", "mlp_dro_within_cw")
    assert {arms[name].selection for name in names} == {"worst_country"}


def test_within_country_class_weights_balance_classes_inside_each_country():
    """D3.3: inside every country both classes weigh the same, and each country keeps its row count."""
    country = pd.Series(["A"] * 100 + ["B"] * 50)
    y = pd.Series(["Yes"] * 40 + ["No"] * 60 + ["Yes"] * 5 + ["No"] * 45)
    weights = within_country_class_weights(y, country)
    for name, size in (("A", 100), ("B", 50)):
        rows = (country == name).to_numpy()
        yes, no = rows & (y == "Yes").to_numpy(), rows & (y == "No").to_numpy()
        assert np.isclose(weights[yes].sum(), weights[no].sum())
        assert np.isclose(weights[rows].sum(), size)


def test_within_cw_arms_pair_erm_and_group_dro_on_the_same_weights():
    arms = build_arms()
    assert arms["mlp_erm_within_cw"].weighting == arms["mlp_dro_within_cw"].weighting == "within_country"


def test_country_label_key_matches_baseline_stratification():
    frame = _frame(20)
    assert country_label_key(frame).iloc[0] == f"{frame['country'].iloc[0]}_{frame['bank_account'].iloc[0]}"
