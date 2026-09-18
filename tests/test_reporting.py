import numpy as np
import pytest

from fin_inclusion.evaluation.cv_runner import CVFoldResult
from fin_inclusion.evaluation.reporting import (
    aggregate_confusion_matrix,
    compare_paired_pr_auc,
    confusion_matrix_markdown,
    country_table_markdown,
    fold_pr_auc_scores,
    full_results_table_markdown,
    full_results_table_row,
    results_table_markdown,
    results_table_row,
    summarize_country_metrics,
    summarize_metrics,
)


def _fold_results() -> list[CVFoldResult]:
    return [
        CVFoldResult(
            fold=0,
            metrics={"pr_auc": 0.5, "roc_auc": 0.8, "f1": 0.4, "precision": 0.3, "recall": 0.5, "accuracy": 0.7},
            country_metrics={
                "Kenya": {"pr_auc": 0.6, "f1": 0.5},
                "Uganda": {"pr_auc": 0.4, "f1": 0.3},
            },
            confusion_matrix=np.array([[50, 10], [8, 12]]),
        ),
        CVFoldResult(
            fold=1,
            metrics={"pr_auc": 0.7, "roc_auc": 0.9, "f1": 0.6, "precision": 0.5, "recall": 0.7, "accuracy": 0.8},
            country_metrics={
                "Kenya": {"pr_auc": 0.8, "f1": 0.7},
                "Uganda": {"pr_auc": 0.2, "f1": 0.1},
            },
            confusion_matrix=np.array([[55, 5], [6, 14]]),
        ),
    ]


def test_summarize_metrics_computes_mean_and_std():
    summary = summarize_metrics(_fold_results())

    assert summary["pr_auc"][0] == 0.6
    assert summary["pr_auc"][1] > 0.0


def test_summarize_country_metrics_aggregates_per_country():
    df = summarize_country_metrics(_fold_results())

    kenya = df[df["country"] == "Kenya"].iloc[0]
    uganda = df[df["country"] == "Uganda"].iloc[0]
    assert kenya["pr_auc_mean"] == pytest.approx(0.7)
    assert uganda["pr_auc_mean"] == pytest.approx(0.3)


def test_results_table_row_formats_mean_pm_std():
    summary = summarize_metrics(_fold_results())

    row = results_table_row("XGBoost", "class-weight", summary)

    assert row["model"] == "XGBoost"
    assert "±" in row["pr_auc"]


def test_results_table_markdown_has_header_and_one_row_per_entry():
    summary = summarize_metrics(_fold_results())
    row = results_table_row("XGBoost", "class-weight", summary)

    table = results_table_markdown([row])

    assert "| Model | Imbalance strategy | PR-AUC | F1 | ROC-AUC |" in table
    assert "XGBoost" in table


def test_country_table_markdown_has_one_row_per_country():
    df = summarize_country_metrics(_fold_results())

    table = country_table_markdown(df)

    assert "Kenya" in table
    assert "Uganda" in table
    assert table.count("\n") == 3


def test_fold_pr_auc_scores_extracts_in_order():
    scores = fold_pr_auc_scores(_fold_results())

    assert scores == [0.5, 0.7]


def test_compare_paired_pr_auc_no_systematic_difference_is_not_significant():
    scores_a = [0.50, 0.62, 0.55, 0.70, 0.60]
    scores_b = [0.52, 0.60, 0.57, 0.68, 0.61]

    result = compare_paired_pr_auc(scores_a, scores_b)

    assert result.ttest_p_value > 0.3
    assert result.wilcoxon_p_value > 0.3


def test_compare_paired_pr_auc_consistent_difference_is_significant():
    scores_a = [0.60, 0.62, 0.61, 0.63, 0.60]
    scores_b = [0.50, 0.48, 0.51, 0.49, 0.52]

    result = compare_paired_pr_auc(scores_a, scores_b)

    assert result.ttest_p_value < 0.05
    assert result.wilcoxon_p_value < 0.1


def test_full_results_table_row_includes_precision_recall_accuracy():
    summary = summarize_metrics(_fold_results())

    row = full_results_table_row("XGBoost", "class-weight", summary)

    assert "precision" in row
    assert "recall" in row
    assert "accuracy" in row
    assert "±" in row["accuracy"]


def test_full_results_table_markdown_has_all_metric_columns():
    summary = summarize_metrics(_fold_results())
    row = full_results_table_row("XGBoost", "class-weight", summary)

    table = full_results_table_markdown([row])

    assert "Precision" in table
    assert "Recall" in table
    assert "Accuracy" in table


def test_aggregate_confusion_matrix_sums_across_folds():
    total = aggregate_confusion_matrix(_fold_results())

    expected = np.array([[50, 10], [8, 12]]) + np.array([[55, 5], [6, 14]])
    np.testing.assert_array_equal(total, expected)


def test_confusion_matrix_markdown_renders_all_four_cells():
    cm = np.array([[105, 15], [14, 26]])

    table = confusion_matrix_markdown(cm)

    assert "105" in table
    assert "15" in table
    assert "14" in table
    assert "26" in table
    assert "Actual No" in table
    assert "Pred Yes" in table
