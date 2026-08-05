from __future__ import annotations

from vipragsent.evaluation.metrics import binary_macro_f1, expected_calibration_error
from vipragsent.evaluation.thresholds import tune_binary_threshold
from vipragsent.statistics.bootstrap import hierarchical_bootstrap, holm_bonferroni


def test_binary_macro_f1_uses_fixed_two_class_macro() -> None:
    assert binary_macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) == (2 / 3 + 4 / 5) / 2
    assert binary_macro_f1([0, 0], [0, 0]) == 0.5


def test_threshold_tie_breaks_toward_half_then_smaller() -> None:
    threshold = tune_binary_threshold([0, 1], [0.4, 0.6], start=0.49, stop=0.51, step=0.01)
    assert threshold == 0.5


def test_ece_is_top_label_equal_width() -> None:
    probabilities = [[0.9, 0.1], [0.6, 0.4]]
    assert expected_calibration_error([0, 1], probabilities, bins=2) > 0


def test_hierarchical_bootstrap_is_reproducible_and_paired() -> None:
    true = [0, 1, 0, 1]
    runs = [(true, [0, 1, 0, 1]), (true, [0, 0, 0, 1]), (true, [1, 1, 0, 1])]
    first = hierarchical_bootstrap(runs, binary_macro_f1, resamples=20, seed=7)
    second = hierarchical_bootstrap(runs, binary_macro_f1, resamples=20, seed=7)
    assert first == second
    assert len(first.distribution) == 20


def test_holm_correction_preserves_order() -> None:
    adjusted = holm_bonferroni([0.001, 0.02, 0.5])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
