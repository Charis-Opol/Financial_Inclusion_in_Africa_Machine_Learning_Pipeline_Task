import pandas as pd
import pytest

from fin_inclusion.data.loader import DataLoader, DuplicateKeyError, SchemaError

VALID_ROW = {
    "country": "Kenya",
    "year": 2018,
    "uniqueid": "uniqueid_1",
    "bank_account": "Yes",
    "location_type": "Rural",
    "cellphone_access": "Yes",
    "household_size": 3,
    "age_of_respondent": 24,
    "gender_of_respondent": "Female",
    "relationship_with_head": "Spouse",
    "marital_status": "Married/Living together",
    "education_level": "Secondary education",
    "job_type": "Self employed",
}


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_train_valid(tmp_path):
    train_path = _write_csv(tmp_path / "train.csv", [VALID_ROW])
    test_path = _write_csv(
        tmp_path / "test.csv", [{k: v for k, v in VALID_ROW.items() if k != "bank_account"}]
    )
    loader = DataLoader(raw_train_path=train_path, raw_test_path=test_path)

    train_df = loader.load_train()
    test_df = loader.load_test()

    assert len(train_df) == 1
    assert len(test_df) == 1
    assert "bank_account" in train_df.columns
    assert "bank_account" not in test_df.columns


def test_load_train_missing_column_raises(tmp_path):
    row = {k: v for k, v in VALID_ROW.items() if k != "job_type"}
    train_path = _write_csv(tmp_path / "train.csv", [row])
    test_path = _write_csv(tmp_path / "test.csv", [row])
    loader = DataLoader(raw_train_path=train_path, raw_test_path=test_path)

    with pytest.raises(SchemaError):
        loader.load_train()


def test_load_train_duplicate_composite_key_raises(tmp_path):
    train_path = _write_csv(tmp_path / "train.csv", [VALID_ROW, VALID_ROW])
    test_path = _write_csv(
        tmp_path / "test.csv", [{k: v for k, v in VALID_ROW.items() if k != "bank_account"}]
    )
    loader = DataLoader(raw_train_path=train_path, raw_test_path=test_path)

    with pytest.raises(DuplicateKeyError):
        loader.load_train()


def test_uniqueid_alone_may_repeat_across_countries(tmp_path):
    row_kenya = dict(VALID_ROW, country="Kenya")
    row_rwanda = dict(VALID_ROW, country="Rwanda")
    train_path = _write_csv(tmp_path / "train.csv", [row_kenya, row_rwanda])
    test_path = _write_csv(
        tmp_path / "test.csv", [{k: v for k, v in VALID_ROW.items() if k != "bank_account"}]
    )
    loader = DataLoader(raw_train_path=train_path, raw_test_path=test_path)

    df = loader.load_train()

    assert df["uniqueid"].duplicated().sum() == 1
    assert df.duplicated(subset=["uniqueid", "country", "year"]).sum() == 0
