import math

import numpy as np
import pytest

from fin_inclusion.monitoring import psi


def test_identical_distributions_have_zero_psi():
    assert psi.psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)


def test_psi_matches_hand_computation():
    expected, actual = [0.5, 0.5], [0.8, 0.2]
    by_hand = (0.8 - 0.5) * math.log(0.8 / 0.5) + (0.2 - 0.5) * math.log(0.2 / 0.5)
    assert psi.psi(expected, actual) == pytest.approx(by_hand)


def test_empty_bin_gives_finite_psi():
    assert math.isfinite(psi.psi([0.0, 1.0], [0.5, 0.5]))


@pytest.mark.parametrize("value, status", [(0.0, "stable"), (0.0999, "stable"), (0.10, "moderate"),
                                           (0.25, "moderate"), (0.2501, "significant")])
def test_standard_interpretation_thresholds(value, status):
    assert psi.interpret(value) == status


def test_numeric_bins_are_right_closed_and_open_ended():
    edges = [10.0, 20.0]
    # bins: (-inf, 10], (10, 20], (20, inf)
    proportions = psi.numeric_proportions([5, 10, 11, 20, 21, 1000], edges)
    np.testing.assert_allclose(proportions, [2 / 6, 2 / 6, 2 / 6])
    assert psi.numeric_bin_labels(edges) == ["(-inf, 10]", "(10, 20]", "(20, inf]"]


def test_decile_edges_deduplicate_ties():
    assert psi.decile_edges(np.array([1] * 50 + [2] * 50)) == [1.0, 2.0]


def test_unseen_category_lands_in_its_own_bin():
    proportions = psi.categorical_proportions(["a", "b", "zzz", "a"], ["a", "b"])
    np.testing.assert_allclose(proportions, [0.5, 0.25, 0.25])


def test_empty_sample_is_an_error_not_a_zero():
    with pytest.raises(ValueError):
        psi.numeric_proportions([], [1.0])
