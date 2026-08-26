"""
SignalsBrain — Batch Weight Refit by Ablation

Replaces per-trade weight nudging.

The previous scheme adjusted a factor's weight by +/-0.02-0.03 after every single
trade, and it assigned credit by bulk correlation: when a trade won, EVERY factor
that happened to agree with the signal was boosted, whether or not it contributed
anything. With eight agreeing factors, seven could be irrelevant and all eight
were still rewarded. There was no holdout, no significance test, and no way for a
weight to move back except by accumulating the opposite noise.

This module instead measures each dimension's marginal contribution by
**leave-one-out ablation on a chronological holdout**:

  1. Load completed trades whose P&L is known, ordered by time.
  2. Split by position, never randomly — these are time-ordered outcomes and a
     random split lets the future inform the past.
  3. Fit a baseline logistic map from directional score to win probability on the
     training half, and measure its holdout log-loss.
  4. For each dimension, zero it out, refit, and re-measure holdout log-loss.
  5. `delta = ablated_loss - baseline_loss`. Positive means removing the
     dimension made predictions worse, i.e. it carries real information.
  6. Move the multiplier only when the delta is significant under a paired
     bootstrap over the holdout rows.

A dimension that cannot be shown to help is left at 1.0 rather than drifting.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..memory.pattern_db import PatternDB, WIN_OUTCOMES
from ..state.dimensions import (
    DIRECTION_DIMENSIONS, DIRECTION_CATEGORY_WEIGHTS, DIMENSIONS,
)
from ..reasoning.calibration import PlattCalibrator
from .weight_table import (
    WeightTable, WeightCell, MIN_MULTIPLIER, MAX_MULTIPLIER,
)

# A cell needs this many completed trades before any weight may move.
MIN_CELL_TRADES = 60
# Holdout share, taken from the end of the time-ordered sample.
HOLDOUT_FRACTION = 0.30
# Bootstrap resamples for the paired significance test.
BOOTSTRAP_N = 400
# Two-sided significance level.
ALPHA = 0.05


@dataclass
class TradeSample:
    ts: float
    instrument: str
    direction: str
    regime_band: int      # adx_band, as stored
    gex_band: int         # gex_regime, as stored
    vector: dict[str, float]
    win: int

    @property
    def regime(self) -> str:
        return {-1: "Choppy", 0: "Developing", 1: "Trending", 2: "Trending"}.get(
            self.regime_band, "unknown")

    @property
    def gex(self) -> str:
        return {-1: "Negative", 1: "Positive"}.get(self.gex_band, "unknown")


def _fingerprint_keys() -> list[str]:
    """Must match MarketState.fingerprint ordering exactly."""
    return sorted(k for k, d in DIMENSIONS.items() if d.weight > 0)


def load_samples(db: PatternDB, instrument: Optional[str] = None,
                 since: Optional[float] = None,
                 limit: int = 20000) -> list[TradeSample]:
    """
    Load completed, P&L-known trades in chronological order.

    Rows with `pnl_source='unavailable'` are excluded: they carry no realised
    outcome, and treating a placeholder as performance is exactly the defect this
    pipeline exists to avoid.
    """
    keys = _fingerprint_keys()
    where = ["outcome != ''", "outcome != 'NO_ENTRY'", "state_vector != ''",
             "pnl_pct IS NOT NULL",
             "(pnl_source IS NULL OR pnl_source != 'unavailable')"]
    params: list = []
    if instrument:
        where.append("instrument = ?")
        params.append(instrument)
    if since is not None:
        where.append("timestamp >= ?")
        params.append(float(since))
    sql = (f"SELECT timestamp, instrument, direction, adx_band, gex_regime, "
           f"state_vector, outcome, pnl_pct FROM signals "
           f"WHERE {' AND '.join(where)} ORDER BY timestamp ASC LIMIT ?")
    params.append(limit)

    out: list[TradeSample] = []
    with sqlite3.connect(str(db.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(sql, params):
            try:
                vec = json.loads(row["state_vector"])
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(vec, list) or len(vec) != len(keys):
                continue
            pnl = row["pnl_pct"]
            if pnl is None:
                continue
            out.append(TradeSample(
                ts=float(row["timestamp"]),
                instrument=row["instrument"],
                direction=row["direction"],
                regime_band=int(row["adx_band"]),
                gex_band=int(row["gex_regime"]),
                vector=dict(zip(keys, (float(v) for v in vec))),
                win=1 if float(pnl) > 0 else 0,
            ))
    return out


def _direction_score(vector: dict[str, float],
                     drop: Optional[str] = None,
                     multipliers: Optional[dict[str, float]] = None) -> float:
    """
    Recompute the directional score from a stored vector.

    Mirrors MarketState.compute_composites for the direction channel so ablation
    measures the real scoring function rather than a proxy.
    """
    cat_score: dict[str, float] = {}
    cat_declared: dict[str, float] = {}
    for name, d in DIRECTION_DIMENSIONS.items():
        if d.weight == 0 or name == drop:
            continue
        w = d.weight * (multipliers or {}).get(name, 1.0)
        cat_declared[d.category.value] = cat_declared.get(d.category.value, 0.0) + w
    for name, d in DIRECTION_DIMENSIONS.items():
        if d.weight == 0 or name == drop:
            continue
        v = vector.get(name, 0.0)
        if v == 0.0:
            continue
        w = d.weight * (multipliers or {}).get(name, 1.0)
        cat_score[d.category.value] = cat_score.get(d.category.value, 0.0) + \
            max(-1.0, min(1.0, v)) * w
    total = 0.0
    for cat, cw in DIRECTION_CATEGORY_WEIGHTS.items():
        dec = cat_declared.get(cat.value, 0.0)
        if dec > 0:
            total += (cat_score.get(cat.value, 0.0) / dec) * cw
    return max(-1.0, min(1.0, total / 100.0))


def _oriented(score: float, direction: str) -> float:
    """Score expressed as strength in the traded direction, mapped to 0-100."""
    sign = 1.0 if direction == "BUY" else -1.0
    return max(0.0, min(100.0, score * sign * 100.0))


def _log_losses(cal: PlattCalibrator, scores: Sequence[float],
                outcomes: Sequence[int]) -> list[float]:
    """Per-row log loss, kept unaggregated so the test can be paired."""
    out = []
    for s, y in zip(scores, outcomes):
        p = min(1 - 1e-12, max(1e-12, cal.predict(s)))
        out.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    return out


def _paired_bootstrap_p(base: Sequence[float], alt: Sequence[float],
                        n_resamples: int = BOOTSTRAP_N, seed: int = 17) -> float:
    """
    Two-sided p-value for "mean(alt) - mean(base) differs from zero", using a
    paired bootstrap over holdout rows.

    Paired because both losses are computed on the same rows; an unpaired test
    would ignore that and overstate the uncertainty.
    """
    diffs = [a - b for a, b in zip(alt, base)]
    n = len(diffs)
    if n < 10:
        return 1.0
    observed = sum(diffs) / n
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    # Centre the differences to simulate the null of no effect.
    centred = [d - observed for d in diffs]
    extreme = 0
    for _ in range(n_resamples):
        s = sum(centred[rng.randrange(n)] for _ in range(n)) / n
        if abs(s) >= abs(observed):
            extreme += 1
    return (extreme + 1) / (n_resamples + 1)


def refit(db: PatternDB,
          instrument: Optional[str] = None,
          lookback_days: Optional[float] = 365.0,
          min_cell_trades: int = MIN_CELL_TRADES,
          holdout_fraction: float = HOLDOUT_FRACTION,
          previous_version: int = 0,
          group_by_regime: bool = True) -> WeightTable:
    """
    Fit a new WeightTable by leave-one-out ablation on a chronological holdout.

    Returns a table whose cells are `significant=False` (multiplier 1.0) unless
    the ablation delta passed the bootstrap test. An untrained or
    insufficiently-evidenced system therefore scores exactly as designed.
    """
    since = None if lookback_days is None else time.time() - lookback_days * 86400
    samples = load_samples(db, instrument=instrument, since=since)

    table = WeightTable(version=previous_version + 1, trained_at=time.time(),
                        n_trades=len(samples))

    if len(samples) < min_cell_trades:
        table.notes = (f"{len(samples)} usable trades; need {min_cell_trades} "
                       f"before any weight may move. All multipliers left at 1.0.")
        return table

    # Group into cells. "any|any" is always fitted so a global signal can be
    # learned before per-regime cells have enough data.
    groups: dict[tuple[str, str], list[TradeSample]] = {("any", "any"): list(samples)}
    if group_by_regime:
        for s in samples:
            groups.setdefault((s.regime, s.gex), []).append(s)

    fitted_cells = 0
    for (regime, gex), rows in groups.items():
        if len(rows) < min_cell_trades:
            continue
        rows.sort(key=lambda r: r.ts)
        split = int(len(rows) * (1.0 - holdout_fraction))
        train, hold = rows[:split], rows[split:]
        if len(train) < 20 or len(hold) < 15:
            continue

        # Baseline
        tr_scores = [_oriented(_direction_score(r.vector), r.direction) for r in train]
        tr_y = [r.win for r in train]
        ho_scores = [_oriented(_direction_score(r.vector), r.direction) for r in hold]
        ho_y = [r.win for r in hold]
        base_cal = PlattCalibrator().fit(tr_scores, tr_y)
        base_losses = _log_losses(base_cal, ho_scores, ho_y)
        base_mean = sum(base_losses) / len(base_losses)

        for dim in DIRECTION_DIMENSIONS:
            # Only test dimensions that are actually present in this cell.
            present = sum(1 for r in rows if r.vector.get(dim, 0.0) != 0.0)
            if present < min_cell_trades // 2:
                continue

            tr_a = [_oriented(_direction_score(r.vector, drop=dim), r.direction) for r in train]
            ho_a = [_oriented(_direction_score(r.vector, drop=dim), r.direction) for r in hold]
            alt_cal = PlattCalibrator().fit(tr_a, tr_y)
            alt_losses = _log_losses(alt_cal, ho_a, ho_y)
            alt_mean = sum(alt_losses) / len(alt_losses)

            delta = alt_mean - base_mean          # >0 => removing it hurt => useful
            p = _paired_bootstrap_p(base_losses, alt_losses)
            significant = p < ALPHA and abs(delta) > 1e-4

            cell = WeightCell(
                regime=regime, gex_regime=gex, dimension=dim,
                n_train=len(train), n_holdout=len(hold),
                ablation_delta=delta, p_value=p, significant=significant,
            )
            if not significant:
                cell.multiplier = 1.0
                cell.reason = (f"ablation delta {delta:+.5f} not significant "
                               f"(p={p:.3f}); weight unchanged")
            else:
                # Map the effect size onto a bounded multiplier. The scale is
                # deliberately conservative: a large log-loss effect is a strong
                # claim, and a single refit should not double a weight.
                step = max(-0.5, min(0.5, delta * 20.0))
                cell.multiplier = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, 1.0 + step))
                cell.reason = (
                    f"removing it {'worsened' if delta > 0 else 'improved'} holdout "
                    f"log-loss by {abs(delta):.5f} (p={p:.3f}) over "
                    f"{len(hold)} holdout trades"
                )
                fitted_cells += 1
            table.set_cell(cell)

    table.notes = (f"{len(samples)} usable trades across {len(groups)} cells; "
                   f"{fitted_cells} multipliers moved after ablation "
                   f"(alpha={ALPHA}, holdout={holdout_fraction:.0%}, "
                   f"min_cell_trades={min_cell_trades})")
    return table


def ablation_report(db: PatternDB, instrument: Optional[str] = None,
                    **kwargs) -> dict:
    """Human-readable refit summary, for the API and the CLI."""
    table = refit(db, instrument=instrument, **kwargs)
    return table.summary()
