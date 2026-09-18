import pandas as pd

from fin_inclusion.data.validators import (
    cardinality_audit,
    check_key_uniqueness,
    check_train_test_category_consistency,
)


def test_check_key_uniqueness_distinguishes_uniqueid_from_composite_key():
    df = pd.DataFrame(
        {
            "uniqueid": ["u1", "u1", "u2"],
            "country": ["Kenya", "Rwanda", "Kenya"],
            "year": [2018, 2016, 2018],
        }
    )

    report = check_key_uniqueness(df)

    assert report.n_duplicates_on_uniqueid_alone == 1
    assert report.n_duplicates_on_composite_key == 0
    assert not report.uniqueid_alone_is_unique
    assert report.composite_key_is_unique


def test_check_train_test_category_consistency_finds_unseen_category():
    train_df = pd.DataFrame({"job_type": ["Self employed", "Farming and Fishing"]})
    test_df = pd.DataFrame({"job_type": ["Self employed", "No Income"]})

    report = check_train_test_category_consistency(train_df, test_df, ["job_type"])

    assert report.unseen_in_test["job_type"] == ["No Income"]
    assert report.has_unseen_categories


def test_check_train_test_category_consistency_no_unseen_categories():
    train_df = pd.DataFrame({"job_type": ["Self employed", "No Income"]})
    test_df = pd.DataFrame({"job_type": ["Self employed"]})

    report = check_train_test_category_consistency(train_df, test_df, ["job_type"])

    assert report.unseen_in_test["job_type"] == []
    assert not report.has_unseen_categories


def test_cardinality_audit_sorted_descending():
    df = pd.DataFrame(
        {
            "binary_col": ["a", "b", "a", "b"],
            "wide_col": ["a", "b", "c", "d"],
        }
    )

    result = cardinality_audit(df, ["binary_col", "wide_col"])

    assert list(result.index) == ["wide_col", "binary_col"]
    assert result["wide_col"] == 4
    assert result["binary_col"] == 2
