"""
SignalsBrain — Leakage-Safe Time-Series Splits

Why plain cross-validation is wrong here
----------------------------------------
A trading label is not a point in time, it is an INTERVAL: "buy at 13:15, exit an hour
later" depends on bars up to 14:15. Standard k-fold shuffles rows, so a training row
can contain the future of a test row and the model scores brilliantly on information it
would never have had. Even an ordinary time-series split leaks at the boundary, because
the last training label overlaps the first test bars.

Two corrections, both from López de Prado's "Advances in Financial Machine Learning":

* PURGING — drop any training observation whose label interval overlaps the test set.
* EMBARGO — additionally drop a short band of training observations immediately AFTER
  the test set, because serial correlation makes those nearly as informative as the
  test labels themselves.

On top of that, a single train/test path gives one number and no sense of its variance.
COMBINATORIAL PURGED CV (CPCV) instead forms many train/test paths from the same data,
producing a DISTRIBUTION of out-of-sample results. That distribution is what feeds the
probability-of-backtest-overfitting statistic in `metrics.py`.

Every function here returns index arrays only. It never touches returns, prices or the
signal engine, which keeps it trivially testable and reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Split:
    """One train/test partition, plus a note on what was removed and why."""
    train: np.ndarray
    test: np.ndarray
    purged: int = 0
    embargoed: int = 0
    label: str = ""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Split({self.label} train={self.train.size} test={self.test.size} "
                f"purged={self.purged} embargoed={self.embargoed})")


def _as_horizons(n_samples: int, label_horizon: int | Sequence[int]) -> np.ndarray:
    """
    Normalise the label horizon into a per-observation array of END indices.

    A scalar means every signal is held the same number of bars (our case: the engine
    holds until stop/target/15:20, and we bound it with a max hold). A sequence allows
    a variable hold per trade, which is what the real desk produces.
    """
    if np.isscalar(label_horizon):
        h = int(label_horizon)  # type: ignore[arg-type]
        return np.minimum(np.arange(n_samples) + h, n_samples - 1)
    ends = np.asarray(list(label_horizon), dtype=int)
    if ends.size != n_samples:
        raise ValueError(f"label_horizon length {ends.size} != n_samples {n_samples}")
    return np.minimum(np.maximum(ends, np.arange(n_samples)), n_samples - 1)


def purge_train_indices(train: np.ndarray, test: np.ndarray, label_end: np.ndarray,
                        embargo: int = 0) -> tuple[np.ndarray, int, int]:
    """
    Remove training observations that leak the test set.

    Returns (kept_train, n_purged, n_embargoed).

    Purge rule: drop training index i when its label interval [i, label_end[i]]
    intersects the test window at all. Embargo rule: additionally drop training indices
    in (test_max, test_max + embargo].
    """
    if test.size == 0 or train.size == 0:
        return train, 0, 0

    t_min, t_max = int(test.min()), int(test.max())

    # A train row leaks if its label reaches into the test block, or if the row itself
    # sits inside it (possible for non-contiguous CPCV test groups).
    overlaps = (label_end[train] >= t_min) & (train <= t_max)
    purged_mask = overlaps

    embargo_mask = np.zeros_like(purged_mask)
    if embargo > 0:
        embargo_mask = (train > t_max) & (train <= t_max + embargo)

    drop = purged_mask | embargo_mask
    kept = train[~drop]
    return kept, int(purged_mask.sum()), int(embargo_mask.sum())


def walk_forward(n_samples: int, n_splits: int = 5, label_horizon: int | Sequence[int] = 1,
                 embargo_pct: float = 0.01, min_train: int = 100,
                 anchored: bool = True) -> list[Split]:
    """
    Classic out-of-sample march forward through time — the split that most closely
    mirrors how the strategy would actually have been run.

    `anchored=True` grows the training window from the start (all history available);
    `False` uses a rolling window of roughly the same size as the first fold, which is
    the better choice if you believe the market regime decays.
    """
    if n_samples < min_train + n_splits:
        return []
    label_end = _as_horizons(n_samples, label_horizon)
    embargo = int(round(n_samples * embargo_pct))

    fold = (n_samples - min_train) // n_splits
    if fold <= 0:
        return []

    splits: list[Split] = []
    for k in range(n_splits):
        train_end = min_train + k * fold
        test_start, test_stop = train_end, min(train_end + fold, n_samples)
        if test_stop <= test_start:
            break
        train_start = 0 if anchored else max(0, train_end - min_train - fold)
        train = np.arange(train_start, train_end)
        test = np.arange(test_start, test_stop)
        kept, purged, emb = purge_train_indices(train, test, label_end, embargo)
        splits.append(Split(kept, test, purged, emb, label=f"wf{k + 1}/{n_splits}"))
    return splits


def purged_kfold(n_samples: int, n_splits: int = 5, label_horizon: int | Sequence[int] = 1,
                 embargo_pct: float = 0.01) -> list[Split]:
    """
    K-fold over contiguous time blocks, with purging and embargo applied.

    Unlike walk-forward this trains on data that post-dates the test block, so it is a
    measure of generalisation rather than a simulation of live trading. Useful for
    parameter robustness; never quote it as a live expectation.
    """
    if n_samples < n_splits * 2:
        return []
    label_end = _as_horizons(n_samples, label_horizon)
    embargo = int(round(n_samples * embargo_pct))
    blocks = np.array_split(np.arange(n_samples), n_splits)

    splits: list[Split] = []
    for k, test in enumerate(blocks):
        train = np.setdiff1d(np.arange(n_samples), test, assume_unique=False)
        kept, purged, emb = purge_train_indices(train, test, label_end, embargo)
        splits.append(Split(kept, test, purged, emb, label=f"pk{k + 1}/{n_splits}"))
    return splits


def combinatorial_purged_cv(n_samples: int, n_groups: int = 6, n_test_groups: int = 2,
                            label_horizon: int | Sequence[int] = 1,
                            embargo_pct: float = 0.01) -> list[Split]:
    """
    CPCV: every combination of `n_test_groups` blocks out of `n_groups` becomes a test
    set, giving C(n_groups, n_test_groups) partly-overlapping out-of-sample paths
    instead of one.

    With the defaults that is C(6,2) = 15 paths. The spread across those paths is the
    honest picture of how much a result depends on which slice of history you happened
    to look at — and it is the input to `probability_of_backtest_overfitting`.
    """
    if n_samples < n_groups * 2 or not (1 <= n_test_groups < n_groups):
        return []
    label_end = _as_horizons(n_samples, label_horizon)
    embargo = int(round(n_samples * embargo_pct))
    blocks = np.array_split(np.arange(n_samples), n_groups)

    splits: list[Split] = []
    for combo in combinations(range(n_groups), n_test_groups):
        test = np.concatenate([blocks[b] for b in combo])
        train = np.setdiff1d(np.arange(n_samples), test)
        kept, purged, emb = purge_train_indices(train, test, label_end, embargo)
        splits.append(Split(kept, test, purged, emb,
                            label="cpcv[" + ",".join(str(b) for b in combo) + "]"))
    return splits


def train_test_holdout(n_samples: int, test_frac: float = 0.3,
                       label_horizon: int | Sequence[int] = 1,
                       embargo_pct: float = 0.01) -> Optional[Split]:
    """
    A single, never-reused, most-recent holdout.

    The discipline this enforces matters more than the mathematics: develop and tune on
    the training portion only, then look at the holdout ONCE. Every extra peek turns it
    back into in-sample data and reinstates exactly the selection bias the deflated
    Sharpe ratio exists to punish.
    """
    if n_samples < 20:
        return None
    label_end = _as_horizons(n_samples, label_horizon)
    embargo = int(round(n_samples * embargo_pct))
    cut = int(n_samples * (1.0 - test_frac))
    train, test = np.arange(0, cut), np.arange(cut, n_samples)
    kept, purged, emb = purge_train_indices(train, test, label_end, embargo)
    return Split(kept, test, purged, emb, label=f"holdout{int(test_frac * 100)}%")
