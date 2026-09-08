"""
Tests for brain.validation.splits

The property that matters is the absence of leakage: no training index may retain a
label window that reaches into the test block, and the embargo band after the test block
must be removed. These tests assert that directly rather than trusting the shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.validation import splits as S


def assert_no_leakage(split: S.Split, label_end: np.ndarray, embargo: int = 0) -> None:
    """Every kept training row must end its label strictly before the test block."""
    if split.test.size == 0 or split.train.size == 0:
        return
    t_min, t_max = int(split.test.min()), int(split.test.max())
    for i in split.train:
        assert not (label_end[i] >= t_min and i <= t_max), (
            f"train row {i} (label ends {label_end[i]}) leaks into test [{t_min},{t_max}]")
        if embargo:
            assert not (t_max < i <= t_max + embargo), f"train row {i} inside embargo"


class TestPurging:
    def test_overlapping_labels_are_purged(self):
        n, horizon = 100, 5
        label_end = np.minimum(np.arange(n) + horizon, n - 1)
        train = np.arange(0, 60)
        test = np.arange(60, 80)
        kept, purged, _ = S.purge_train_indices(train, test, label_end, embargo=0)
        assert purged > 0
        # Rows 55..59 have labels reaching >= 60 and must be gone.
        assert all(i not in kept for i in range(55, 60))
        assert_no_leakage(S.Split(kept, test), label_end)

    def test_embargo_removes_rows_after_test(self):
        n = 100
        label_end = np.arange(n)  # zero-length labels: only the embargo can bite
        train = np.arange(0, 100)
        test = np.arange(40, 50)
        kept, _, emb = S.purge_train_indices(train, test, label_end, embargo=5)
        assert emb == 5
        assert all(i not in kept for i in range(50, 56) if i < 100 and i > 49 and i <= 54)

    def test_no_purge_when_labels_are_instant_and_no_embargo(self):
        n = 50
        label_end = np.arange(n)
        train = np.arange(0, 20)
        test = np.arange(30, 40)
        kept, purged, emb = S.purge_train_indices(train, test, label_end)
        assert purged == 0 and emb == 0 and kept.size == 20

    def test_empty_inputs_are_safe(self):
        label_end = np.arange(10)
        kept, p, e = S.purge_train_indices(np.arange(5), np.array([], dtype=int), label_end)
        assert kept.size == 5 and p == 0 and e == 0


class TestWalkForward:
    def test_folds_move_forward_and_never_train_on_the_future(self):
        n = 600
        sp = S.walk_forward(n, n_splits=5, label_horizon=4, embargo_pct=0.01, min_train=100)
        assert len(sp) == 5
        label_end = np.minimum(np.arange(n) + 4, n - 1)
        prev_test_start = -1
        for s in sp:
            assert s.train.size > 0 and s.test.size > 0
            # Anchored walk-forward: training must precede the test block.
            assert s.train.max() < s.test.min()
            assert s.test.min() > prev_test_start
            prev_test_start = int(s.test.min())
            assert_no_leakage(s, label_end)

    def test_variable_horizons_accepted(self):
        n = 300
        holds = list(np.arange(n) + np.random.default_rng(5).integers(1, 8, n))
        sp = S.walk_forward(n, n_splits=3, label_horizon=holds, min_train=60)
        assert sp
        label_end = np.minimum(np.maximum(np.asarray(holds), np.arange(n)), n - 1)
        for s in sp:
            assert_no_leakage(s, label_end)

    def test_returns_empty_when_too_little_data(self):
        assert S.walk_forward(20, n_splits=5, min_train=100) == []

    def test_rolling_window_does_not_grow(self):
        sp = S.walk_forward(600, n_splits=4, min_train=100, anchored=False, label_horizon=1)
        sizes = [s.train.size for s in sp]
        assert max(sizes) - min(sizes) < 600  # bounded, unlike an anchored window


class TestPurgedKFold:
    def test_all_observations_tested_exactly_once(self):
        n = 400
        sp = S.purged_kfold(n, n_splits=5, label_horizon=3)
        tested = np.concatenate([s.test for s in sp])
        assert np.array_equal(np.sort(tested), np.arange(n))

    def test_no_leakage_in_any_fold(self):
        n, h = 400, 6
        label_end = np.minimum(np.arange(n) + h, n - 1)
        for s in S.purged_kfold(n, n_splits=5, label_horizon=h, embargo_pct=0.01):
            assert_no_leakage(s, label_end)

    def test_train_and_test_never_intersect(self):
        for s in S.purged_kfold(300, n_splits=4, label_horizon=2):
            assert np.intersect1d(s.train, s.test).size == 0


class TestCPCV:
    def test_path_count_is_the_binomial_coefficient(self):
        from math import comb
        sp = S.combinatorial_purged_cv(600, n_groups=6, n_test_groups=2, label_horizon=2)
        assert len(sp) == comb(6, 2) == 15

    def test_no_leakage_across_all_paths(self):
        n, h = 600, 4
        label_end = np.minimum(np.arange(n) + h, n - 1)
        for s in S.combinatorial_purged_cv(n, 6, 2, label_horizon=h, embargo_pct=0.01):
            assert_no_leakage(s, label_end)
            assert np.intersect1d(s.train, s.test).size == 0

    def test_rejects_bad_group_configuration(self):
        assert S.combinatorial_purged_cv(600, n_groups=4, n_test_groups=4) == []
        assert S.combinatorial_purged_cv(600, n_groups=4, n_test_groups=0) == []

    def test_more_paths_than_walk_forward(self):
        n = 600
        assert len(S.combinatorial_purged_cv(n, 6, 2)) > len(S.walk_forward(n, 5))


class TestHoldout:
    def test_holdout_is_the_most_recent_slice(self):
        sp = S.train_test_holdout(1000, test_frac=0.3, label_horizon=3)
        assert sp is not None
        assert sp.test.min() > sp.train.max()
        assert sp.test.size == pytest.approx(300, abs=1)

    def test_holdout_has_no_leakage(self):
        n, h = 1000, 5
        label_end = np.minimum(np.arange(n) + h, n - 1)
        sp = S.train_test_holdout(n, 0.3, label_horizon=h, embargo_pct=0.01)
        assert sp is not None
        assert_no_leakage(sp, label_end)

    def test_too_small_returns_none(self):
        assert S.train_test_holdout(10) is None
