"""
Tests for the learning loop.

Two defects are pinned here:

  * Learned weights never reached scoring. `get_weight_adjustment` was read only
    by the orchestrator, which nudged the final confidence scalar;
    `DIMENSIONS[*].weight` is frozen and was never updated, so a factor history
    showed to be worthless kept its full influence on `direction_score`.

  * Credit was assigned by bulk correlation. Every factor that agreed with a
    winning signal was boosted by a fixed step, whether or not it contributed,
    with no holdout and no significance test.
"""

from __future__ import annotations

import datetime
import json
import math
import random
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from brain.godmode.batch_refit import (
    MIN_CELL_TRADES, TradeSample, _direction_score, _oriented,
    _paired_bootstrap_p, load_samples, refit,
)
from brain.godmode.weight_table import (
    DEFAULT_MULTIPLIER, MAX_MULTIPLIER, MIN_MULTIPLIER,
    WeightCell, WeightTable, cell_key,
)
from brain.memory.pattern_db import PatternDB
from brain.reasoning.attribution import attribute
from brain.reasoning.confidence_calc import ConfidenceCalculator
from brain.reasoning.engine import ReasoningEngine
from brain.state.dimensions import DIMENSIONS, DIRECTION_DIMENSIONS
from brain.state.market_state import MarketState

IST = ZoneInfo("Asia/Kolkata")


def ts_at(y=2026, m=8, d=24, hh=12, mm=15) -> float:
    return datetime.datetime(y, m, d, hh, mm, tzinfo=IST).timestamp()


def simple_state(**dims) -> MarketState:
    s = MarketState(instrument="NIFTY", market_open=True)
    for name, val in dims.items():
        s.set_dimension(name, val, val)
    s.compute_quality()
    s.compute_composites()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT TABLE
# ══════════════════════════════════════════════════════════════════════════════

def test_untrained_table_is_identity():
    t = WeightTable()
    assert t.multiplier("Trending", "Negative", "ema_stack_score") == DEFAULT_MULTIPLIER
    resolver = t.resolver("Trending", "Negative")
    assert all(resolver(d) == 1.0 for d in DIRECTION_DIMENSIONS)


def test_insignificant_cells_are_ignored():
    """A multiplier that failed its significance test must not be applied."""
    t = WeightTable(version=1)
    t.set_cell(WeightCell(regime="Trending", gex_regime="Negative",
                          dimension="pcr", multiplier=1.8, significant=False))
    assert t.multiplier("Trending", "Negative", "pcr") == 1.0
    t.set_cell(WeightCell(regime="Trending", gex_regime="Negative",
                          dimension="pcr", multiplier=1.8, significant=True))
    assert t.multiplier("Trending", "Negative", "pcr") == pytest.approx(1.8)


def test_multipliers_are_bounded():
    t = WeightTable()
    t.set_cell(WeightCell(regime="any", gex_regime="any", dimension="pcr",
                          multiplier=99.0, significant=True))
    t.set_cell(WeightCell(regime="any", gex_regime="any", dimension="rsi",
                          multiplier=-5.0, significant=True))
    assert t.multiplier("x", "y", "pcr") == MAX_MULTIPLIER
    assert t.multiplier("x", "y", "rsi") == MIN_MULTIPLIER


def test_regime_specific_cell_wins_over_global():
    t = WeightTable()
    t.set_cell(WeightCell(regime="any", gex_regime="any", dimension="pcr",
                          multiplier=1.2, significant=True))
    t.set_cell(WeightCell(regime="Choppy", gex_regime="Positive", dimension="pcr",
                          multiplier=0.4, significant=True))
    assert t.multiplier("Choppy", "Positive", "pcr") == pytest.approx(0.4)
    assert t.multiplier("Trending", "Negative", "pcr") == pytest.approx(1.2)


def test_table_round_trips_through_disk(tmp_path):
    t = WeightTable(version=7, n_trades=500, notes="x")
    t.set_cell(WeightCell(regime="Trending", gex_regime="Negative",
                          dimension="pcr", multiplier=1.35, significant=True,
                          p_value=0.01, ablation_delta=0.0175))
    p = tmp_path / "weights.json"
    t.save(p)
    back = WeightTable.load(p)
    assert back.version == 7
    assert back.multiplier("Trending", "Negative", "pcr") == pytest.approx(1.35)


def test_corrupt_table_falls_back_to_identity(tmp_path):
    p = tmp_path / "weights.json"
    p.write_text("{not json")
    t = WeightTable.load(p)
    assert t.version == 0
    assert t.multiplier("a", "b", "pcr") == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# LEARNED WEIGHTS REACH SCORING
# ══════════════════════════════════════════════════════════════════════════════

def test_learned_weight_changes_the_directional_score():
    """
    The defect: learning could not influence direction_score at all. Weights were
    frozen and only the final confidence scalar was nudged.
    """
    base = simple_state(ema_stack_score=1.0, pcr=-1.0)
    baseline = base.direction_score

    t = WeightTable(version=1)
    t.set_cell(WeightCell(regime="any", gex_regime="any", dimension="pcr",
                          multiplier=MIN_MULTIPLIER, significant=True))
    down = simple_state(ema_stack_score=1.0, pcr=-1.0)
    down.compute_composites(weight_resolver=t.resolver(down.regime, down.gex_regime),
                            weight_version=1)

    # De-emphasising the opposing dimension must make the read more bullish.
    assert down.direction_score > baseline
    assert down.weight_table_version == 1
    assert down.effective_weights["pcr"] < DIMENSIONS["pcr"].weight


def test_attribution_still_reconciles_with_learned_weights():
    """
    Attribution reads the effective weights the scorer recorded, so the invariant
    sum(contributions) + external == total survives non-default weights.
    """
    t = WeightTable(version=3)
    for dim, mult in (("pcr", 1.9), ("ema_stack_score", 0.3), ("rsi", 1.4)):
        t.set_cell(WeightCell(regime="any", gex_regime="any", dimension=dim,
                              multiplier=mult, significant=True))
    s = simple_state(ema_stack_score=0.8, pcr=-0.6, rsi=0.5, supertrend=1.0)
    s.compute_composites(weight_resolver=t.resolver(s.regime, s.gex_regime),
                         weight_version=3)
    for direction in ("BUY", "SELL"):
        att = attribute(s, direction, external_points=3.0)
        assert att.verify(1e-9), (att.attributed_sum(), att.total)


def test_engine_applies_the_attached_weight_table():
    t = WeightTable(version=5)
    t.set_cell(WeightCell(regime="any", gex_regime="any", dimension="pcr",
                          multiplier=MIN_MULTIPLIER, significant=True))

    plain = ReasoningEngine()
    learned = ReasoningEngine(weight_table=t)

    a = simple_state(ema_stack_score=1.0, pcr=-1.0, supertrend=1.0)
    b = simple_state(ema_stack_score=1.0, pcr=-1.0, supertrend=1.0)
    ca = plain.reason(a, confidence_threshold=0, as_of=ts_at())
    cb = learned.reason(b, confidence_threshold=0, as_of=ts_at())

    assert b.weight_table_version == 5
    assert a.weight_table_version == 0
    assert abs(ca.confidence - cb.confidence) > 1e-6


def test_zero_or_negative_multiplier_is_ignored():
    """A malformed resolver must not zero out a dimension."""
    s = simple_state(ema_stack_score=1.0, pcr=1.0)
    baseline = s.direction_score
    s2 = simple_state(ema_stack_score=1.0, pcr=1.0)
    s2.compute_composites(weight_resolver=lambda d: 0.0)
    assert s2.direction_score == pytest.approx(baseline)
    s3 = simple_state(ema_stack_score=1.0, pcr=1.0)
    s3.compute_composites(weight_resolver=lambda d: float("nan"))
    assert s3.direction_score == pytest.approx(baseline)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION MECHANICS
# ══════════════════════════════════════════════════════════════════════════════

def test_direction_score_recomputation_matches_market_state():
    """
    Ablation must measure the real scoring function. This asserts the standalone
    recomputation agrees with MarketState.compute_composites.
    """
    dims = {"ema_stack_score": 0.7, "supertrend": -1.0, "pcr": 0.4,
            "rsi": -0.3, "vwap_position": 1.0, "htf_trend": 1.0}
    s = simple_state(**dims)
    assert _direction_score(dims) == pytest.approx(s.direction_score, abs=1e-9)


def test_dropping_a_dimension_changes_the_recomputed_score():
    dims = {"ema_stack_score": 1.0, "pcr": -1.0, "supertrend": 1.0}
    full = _direction_score(dims)
    without = _direction_score(dims, drop="pcr")
    assert full != pytest.approx(without)


def test_paired_bootstrap_detects_a_real_difference_and_ignores_noise():
    rng = random.Random(1)
    base = [rng.gauss(0.60, 0.05) for _ in range(200)]
    same = [b + rng.gauss(0.0, 0.05) for b in base]
    worse = [b + 0.20 + rng.gauss(0.0, 0.05) for b in base]
    assert _paired_bootstrap_p(base, same) > 0.05
    assert _paired_bootstrap_p(base, worse) < 0.05


def test_oriented_score_is_direction_aware():
    assert _oriented(0.5, "BUY") == pytest.approx(50.0)
    assert _oriented(0.5, "SELL") == pytest.approx(0.0)
    assert _oriented(-0.5, "SELL") == pytest.approx(50.0)


# ══════════════════════════════════════════════════════════════════════════════
# REFIT END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

def _seed_trades(db: PatternDB, n: int, informative_dim: str,
                 noise_dims: list[str], base_ts: float, seed: int = 3):
    """
    Plant a dataset where exactly ONE dimension predicts the outcome and the
    others are pure noise. A correct credit-assignment scheme must single out the
    informative dimension; bulk correlation cannot, because the noise dimensions
    agree with the winning signal just as often.
    """
    rng = random.Random(seed)
    for i in range(n):
        signal = rng.choice([-1.0, 1.0])
        strength = rng.uniform(0.45, 1.0)
        dims = {informative_dim: signal * strength}
        for nd in noise_dims:
            dims[nd] = signal * rng.uniform(0.45, 1.0)  # correlated with the call
        st = simple_state(**dims)
        direction = "BUY" if signal > 0 else "SELL"
        # Win probability depends ONLY on the informative dimension's strength.
        p_win = 0.12 + 0.78 * strength
        win = rng.random() < p_win
        sid = db.record_signal(state=st, direction=direction, confidence=70,
                               entry_spot=24000, atr=100, timestamp=base_ts + i * 60)
        db.record_outcome(sid, "WIN_T2" if win else "STOP_LOSS",
                          24200 if win else 23880, 0,
                          2.0 if win else -1.2, 60,
                          pnl_pct=55.0 if win else -40.0,
                          pnl_source="observed")


def test_refit_declines_to_move_weights_without_enough_data(tmp_path):
    db = PatternDB(tmp_path / "few.db")
    _seed_trades(db, 20, "pcr", ["rsi"], ts_at())
    table = refit(db, instrument="NIFTY", min_cell_trades=MIN_CELL_TRADES)
    assert table.significant_cells() == []
    assert "before any weight may move" in table.notes
    assert table.multiplier("Trending", "Negative", "pcr") == 1.0


def test_refit_isolates_the_informative_dimension(tmp_path):
    """
    The substantive test. `pcr` drives the outcome; `rsi` and `macd_histogram`
    correlate with the signal direction but carry no predictive content.
    """
    db = PatternDB(tmp_path / "signal.db")
    _seed_trades(db, 420, "pcr", ["rsi", "macd_histogram"], ts_at(), seed=11)
    samples = load_samples(db, instrument="NIFTY")
    assert len(samples) == 420

    table = refit(db, instrument="NIFTY", min_cell_trades=80, group_by_regime=False)
    m_pcr = table.multiplier("any", "any", "pcr")
    m_rsi = table.multiplier("any", "any", "rsi")
    m_macd = table.multiplier("any", "any", "macd_histogram")

    # The informative dimension must be favoured over the noise dimensions.
    assert m_pcr > m_rsi
    assert m_pcr > m_macd
    # And the noise dimensions must not be boosted above identity.
    assert m_rsi <= 1.0 + 1e-9
    assert m_macd <= 1.0 + 1e-9


def test_refit_versions_increment_and_are_recorded(tmp_path):
    db = PatternDB(tmp_path / "ver.db")
    _seed_trades(db, 200, "pcr", ["rsi"], ts_at(), seed=5)
    t1 = refit(db, instrument="NIFTY", min_cell_trades=80, group_by_regime=False,
               previous_version=0)
    t2 = refit(db, instrument="NIFTY", min_cell_trades=80, group_by_regime=False,
               previous_version=t1.version)
    assert t1.version == 1 and t2.version == 2
    s = simple_state(pcr=1.0, rsi=0.5)
    s.compute_composites(weight_resolver=t2.resolver(s.regime, s.gex_regime),
                         weight_version=t2.version)
    assert s.weight_table_version == 2


def test_refit_excludes_rows_without_a_known_pnl(tmp_path):
    db = PatternDB(tmp_path / "np.db")
    base = ts_at()
    st = simple_state(pcr=1.0)
    for i in range(30):
        sid = db.record_signal(state=st, direction="BUY", confidence=70,
                               entry_spot=24000, atr=100, timestamp=base + i)
        db.record_outcome(sid, "WIN_T1", 24100, 0, 1.0, 45,
                          pnl_pct=None, pnl_source="unavailable")
    assert load_samples(db, instrument="NIFTY") == []


def test_refit_summary_is_serialisable(tmp_path):
    db = PatternDB(tmp_path / "sum.db")
    _seed_trades(db, 200, "pcr", ["rsi"], ts_at(), seed=9)
    table = refit(db, instrument="NIFTY", min_cell_trades=80, group_by_regime=False)
    payload = json.dumps(table.summary())
    assert "version" in payload and "cells_significant" in payload
