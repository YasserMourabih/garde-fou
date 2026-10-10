"""Metrics are the argument, so they get checked against hand-computed cases."""

import numpy as np
import pandas as pd
import pytest

from garde_fou.evaluation.metrics import (
    bootstrap_ci,
    by_group,
    point_metrics,
    ranking_metrics,
    threshold_at_fpr,
    threshold_from_benign,
)


def test_point_metrics_match_a_hand_computed_confusion():
    #           attacks: .9 .8 .4   benign: .7 .3 .2 .1
    y = [1, 1, 1, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.4, 0.7, 0.3, 0.2, 0.1]
    m = point_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 4)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["blocked_per_1000_legit"] == pytest.approx(250.0)


@pytest.mark.parametrize("target", [0.001, 0.01, 0.05, 0.1])
def test_threshold_respects_the_false_positive_budget(target):
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(500, int), np.zeros(5000, int)]
    s = np.r_[rng.normal(2, 1, 500), rng.normal(0, 1, 5000)]
    thr = threshold_at_fpr(y, s, target)
    assert point_metrics(y, s, thr)["fpr"] <= target + 1e-9


def test_threshold_is_monotone_in_the_budget():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(200, int), np.zeros(800, int)]
    s = np.r_[rng.normal(1.5, 1, 200), rng.normal(0, 1, 800)]
    thresholds = [threshold_at_fpr(y, s, t) for t in (0.001, 0.01, 0.05, 0.2)]
    assert thresholds == sorted(thresholds, reverse=True)


def test_single_class_slice_degrades_gracefully():
    y = [1, 1, 1]
    s = [0.2, 0.5, 0.9]
    ranking = ranking_metrics(y, s)
    assert np.isnan(ranking["roc_auc"])
    # A threshold above every score blocks nothing rather than crashing.
    assert point_metrics(y, s, threshold_at_fpr(y, s, 0.01))["tp"] == 0


def test_perfect_and_inverted_separations():
    y = [1, 1, 0, 0]
    assert ranking_metrics(y, [0.9, 0.8, 0.2, 0.1])["roc_auc"] == pytest.approx(1.0)
    assert ranking_metrics(y, [0.1, 0.2, 0.8, 0.9])["roc_auc"] == pytest.approx(0.0)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(300, int), np.zeros(900, int)]
    s = np.r_[rng.normal(1.5, 1, 300), rng.normal(0, 1, 900)]
    thr = threshold_at_fpr(y, s, 0.05)
    point = point_metrics(y, s, thr)["recall"]
    lo, hi = bootstrap_ci(
        y, s, lambda yy, ss: point_metrics(yy, ss, thr)["recall"], n_boot=200, seed=0
    )
    assert lo <= point <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_by_group_reports_each_group_at_one_shared_threshold():
    frame = pd.DataFrame(
        {
            "mutator": ["a"] * 40 + ["b"] * 40 + ["tiny"] * 5,
            "label": ([1] * 20 + [0] * 20) * 2 + [1] * 5,
            "score": ([0.9] * 20 + [0.1] * 20) + ([0.2] * 20 + [0.1] * 20) + [0.9] * 5,
        }
    )
    out = by_group(frame, group="mutator", threshold=0.5, min_rows=20)
    assert set(out["mutator"]) == {"a", "b"}  # "tiny" dropped: too few rows
    assert out.set_index("mutator").loc["a", "recall"] == pytest.approx(1.0)
    assert out.set_index("mutator").loc["b", "recall"] == pytest.approx(0.0)


def test_threshold_from_benign_holds_the_budget_on_a_discrete_score():
    """Regression: a rule engine's floor score must not become a block-everything threshold.

    Every benign prompt here gets the same floor score because no rule fires.
    A plain 99th-percentile would return that floor, and `score >= floor` blocks
    100% of traffic while claiming a 1% budget.
    """
    benign = np.full(1500, 0.047)
    thr = threshold_from_benign(benign, 0.01)
    assert (benign >= thr).mean() == 0.0
    # Attacks score above the floor, so they are still caught.
    assert thr <= 0.3


@pytest.mark.parametrize("target", [0.001, 0.01, 0.05, 0.2])
def test_threshold_from_benign_matches_the_requested_rate(target):
    rng = np.random.default_rng(3)
    benign = rng.normal(0, 1, 5000)
    thr = threshold_from_benign(benign, target)
    realised = (benign >= thr).mean()
    assert realised <= target + 1e-9
    # And it is not needlessly strict: relaxing it slightly would overshoot.
    assert realised > target - 0.01


def test_threshold_from_benign_on_an_empty_sample():
    assert threshold_from_benign([], 0.01) == float("inf")
