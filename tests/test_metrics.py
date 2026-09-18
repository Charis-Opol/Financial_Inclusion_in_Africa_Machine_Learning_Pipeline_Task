import numpy as np
import pytest

from fin_inclusion.evaluation.metrics import compute_classification_metrics, compute_confusion_matrix


def test_perfect_separation_scores_near_one():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.05, 0.2, 0.8, 0.9, 0.95])

    metrics = compute_classification_metrics(y_true, y_score)

    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_inverted_scores_score_near_zero():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.9, 0.95, 0.8, 0.1, 0.05, 0.2])

    metrics = compute_classification_metrics(y_true, y_score)

    assert metrics["roc_auc"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)


def test_threshold_changes_f1_precision_recall_but_not_auc_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.3, 0.6, 0.4, 0.7])

    low = compute_classification_metrics(y_true, y_score, threshold=0.35)
    high = compute_classification_metrics(y_true, y_score, threshold=0.65)

    assert low["pr_auc"] == pytest.approx(high["pr_auc"])
    assert low["roc_auc"] == pytest.approx(high["roc_auc"])
    assert low["f1"] != pytest.approx(high["f1"])


def test_no_predicted_positives_does_not_raise():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = compute_classification_metrics(y_true, y_score, threshold=0.99)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["accuracy"] == 0.5


def test_accuracy_reflects_majority_class_baseline():
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    y_score = np.zeros(10)

    metrics = compute_classification_metrics(y_true, y_score, threshold=0.5)

    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["recall"] == 0.0


class TestConfusionMatrix:
    def test_shape_and_ordering(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.6, 0.4, 0.9])

        cm = compute_confusion_matrix(y_true, y_score, threshold=0.5)

        assert cm.shape == (2, 2)
        tn, fp, fn, tp = cm.ravel()
        assert (tn, fp, fn, tp) == (1, 1, 1, 1)

    def test_perfect_predictions_only_populate_diagonal(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.05, 0.2, 0.8, 0.9, 0.95])

        cm = compute_confusion_matrix(y_true, y_score)

        assert cm[0, 1] == 0
        assert cm[1, 0] == 0
        assert cm.sum() == 6

    def test_total_count_matches_input_size(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=50)
        y_score = rng.uniform(size=50)

        cm = compute_confusion_matrix(y_true, y_score)

        assert cm.sum() == 50
