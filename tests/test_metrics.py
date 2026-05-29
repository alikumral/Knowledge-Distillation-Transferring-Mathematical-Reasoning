"""Tests for the pure-Python metric helpers (numpy-dependent ones are skipped
if numpy is absent)."""

import pytest

from mathdistill import metrics as M


def test_accuracy():
    assert M.accuracy([True, True, False, False]) == 0.5
    assert M.accuracy([]) == 0.0


def test_assign_difficulty_tertiles():
    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    labels = M.assign_difficulty(lengths)
    assert labels[0] == "easy"
    assert labels[-1] == "hard"
    assert set(labels) == {"easy", "medium", "hard"}


def test_accuracy_by_bucket():
    correct = [True, False, True, True]
    buckets = ["easy", "easy", "hard", "hard"]
    res = M.accuracy_by_bucket(correct, buckets)
    assert res["easy"] == 0.5
    assert res["hard"] == 1.0


def test_bootstrap_ci_bounds():
    pytest.importorskip("numpy")
    lo, hi = M.bootstrap_ci([True] * 50 + [False] * 50, n_boot=2000, seed=0)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


def test_aggregate_seeds():
    pytest.importorskip("numpy")
    agg = M.aggregate_seeds([0.40, 0.42, 0.44])
    assert agg["n_seeds"] == 3
    assert abs(agg["mean"] - 0.42) < 1e-9
