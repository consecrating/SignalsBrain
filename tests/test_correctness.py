"""
Regression tests for the correctness defects found by executing the engine.

Each test names the defect it pins down and asserts the corrected behaviour, so
a future change that reintroduces the defect fails here rather than silently
shipping.
"""

from __future__ import annotations

import datetime
import math
import random
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from brain.memory.option_pricing import (
    black_scholes, implied_pnl_pct, realised_pnl_pct, atm_strike,
)
from brain.memory.outcome_tracker import ActiveTrade, OutcomeTracker
from brain.memory.pattern_db import PatternDB, NEUTRAL_OUTCOMES, outcome_semantic
from brain.memory.statistics import wilson_interval, two_proportion_z, min_samples_for_width
from brain.reasoning.attribution import attribute
from brain.reasoning.blunder_guard import BlunderGuard
from brain.reasoning.calibration import (
    PlattCalibrator, IsotonicCalibrator, evaluate, fit_calibration,
)
from brain.reasoning.confidence_calc import ConfidenceCalculator
from brain.reasoning.engine import ReasoningEngine
from brain.state.dimensions import (
    DIMENSIONS, Channel, DIRECTION_DIMENSIONS, CONVICTION_DIMENSIONS,
    CONTEXT_DIMENSIONS, normalize_pct, normalize_unit, normalize_percentile,
    DIRECTION_CATEGORY_WEIGHTS,
)
from brain.state.market_state import MarketState
from brain.state.state_builder import (
    StateBuilder, is_market_open, session_minutes_for, next_weekly_expiry_dte,
)

IST = ZoneInfo("Asia/Kolkata")


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def market_hours_ts(y=2026, m=8, d=26, hh=12, mm=15) -> float:
    """A Wednesday inside the NSE session."""
    return datetime.datetime(y, m, d, hh, mm, tzinfo=IST).timestamp()


def make_candles(n=140, start=24000.0, drift=0.0006, vol=0.003,
                 seed=42, t_end=None, step=300):
    rng = random.Random(seed)
    o, c, h, l, v, ts = [], [], [], [], [], []
    p = start
    for i in range(n):
        oo = p
        p = oo * (1 + rng.gauss(drift, vol))
        o.append(oo); c.append(p)
        h.append(max(oo, p) * 1.0012); l.append(min(oo, p) * 0.9988)
        v.append(rng.randint(60000, 240000))
    d = {"opens": o, "closes": c, "highs": h, "lows": l, "volumes": v}
    if t_end is not None:
        d["timestamps"] = [t_end - (n - 1 - i) * step for i in range(n)]
    return d


GEX = {
    "pcr": 1.28, "regime": "Negative Gamma", "netGEX": -42.5, "flip": 24050.0,
    "callWall": 24300.0, "putWall": 23800.0, "maxPain": 24100.0,
    "avgIV": 14.8, "putIV": 15.6, "callIV": 14.1, "oiBuildup": "long_buildup",
}


def directional_state(sign: float = 1.0, **clock) -> MarketState:
    """Only directional dimensions set, plus whatever clock values are given."""
    s = MarketState(instrument="NIFTY", market_open=True)
    s.set_dimension("ema_stack_score", sign, sign)
    s.set_dimension("supertrend", sign, sign)
    s.set_dimension("pcr", 1.3 if sign > 0 else 0.6, sign)
    for k, val in clock.items():
        s.set_dimension(k, val[0], val[1])
    s.compute_composites()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL SEPARATION
# ══════════════════════════════════════════════════════════════════════════════

def test_every_dimension_has_exactly_one_channel():
    assert set(DIMENSIONS) == (set(DIRECTION_DIMENSIONS) | set(CONVICTION_DIMENSIONS)
                               | set(CONTEXT_DIMENSIONS))
    assert not (set(DIRECTION_DIMENSIONS) & set(CONVICTION_DIMENSIONS))
    assert not (set(DIRECTION_DIMENSIONS) & set(CONTEXT_DIMENSIONS))
    assert not (set(CONVICTION_DIMENSIONS) & set(CONTEXT_DIMENSIONS))


def test_no_clock_or_calendar_dimension_is_directional():
    """
    day_of_week was normalised to [-1,+1] inside a signed directional sum, which
    made Monday mechanically bearish and Friday mechanically bullish.
    """
    for name in ("session_minutes", "day_of_week", "session_phase", "dte"):
        assert DIMENSIONS[name].channel is Channel.CONTEXT, name


def test_trend_strength_and_volatility_are_not_directional():
    """ADX is strength, not direction. Volatility level is a regime, not a view."""
    for name in ("adx_value", "adx_regime", "atr_pct", "bb_width",
                 "volume_ratio", "gex_regime"):
        assert DIMENSIONS[name].channel is Channel.CONVICTION, name
    for name in ("vix", "vix_change", "atm_iv", "iv_percentile"):
        assert DIMENSIONS[name].channel is Channel.CONTEXT, name


def test_clock_cannot_change_the_directional_read():
    """
    Measured before the fix: identical evidence scored 42.75 base confidence on
    Monday 09:20 and 75.85 on Friday 15:05 — a 33.1-point swing across a
    60-point trade gate.
    """
    from brain.state.dimensions import normalize_range
    scores = []
    for sm, dow, ph in [(5, 1, 0.0), (180, 3, 0.5), (350, 5, 1.0)]:
        s = directional_state(
            1.0,
            session_minutes=(sm, normalize_range(sm, 0, 375)),
            day_of_week=(dow, 0.0),
            session_phase=(ph, normalize_range(ph, 0, 1)),
        )
        scores.append((s.direction_score, s.net_directional_bias, s.agreement_factor))
    assert len({round(d, 9) for d, _, _ in scores}) == 1
    assert len({round(n, 9) for _, n, _ in scores}) == 1
    assert len({round(a, 9) for _, _, a in scores}) == 1


def test_stronger_trend_strengthens_the_directional_signal():
    """
    Before the fix a pure bear setup weakened as ADX rose:
    net_bias -17.8 (ADX 12) -> -3.0 (ADX 50), monotonically backwards.
    """
    seq = []
    for adx in (12, 20, 30, 40, 50):
        s = MarketState(instrument="NIFTY", market_open=True)
        s.set_dimension("ema_stack_score", -1.0, -1.0)
        s.set_dimension("supertrend", -1.0, -1.0)
        s.set_dimension("di_differential", -25, -0.9)
        s.set_dimension("adx_value", adx, normalize_unit(adx, 12, 40))
        s.set_dimension("adx_regime", 1.0 if adx >= 25 else 0.0,
                        1.0 if adx >= 25 else 0.55)
        s.compute_composites()
        seq.append(s.net_directional_bias)
    assert all(seq[i + 1] <= seq[i] + 1e-9 for i in range(len(seq) - 1)), seq
    assert seq[-1] < seq[0]


def test_conviction_scales_and_never_flips_direction():
    for sign in (1.0, -1.0):
        for adx in (12, 25, 45):
            s = MarketState(instrument="NIFTY", market_open=True)
            s.set_dimension("ema_stack_score", sign, sign)
            s.set_dimension("adx_value", adx, normalize_unit(adx, 12, 40))
            s.compute_composites()
            assert 0.0 <= s.conviction_score <= 1.0
            assert math.copysign(1, s.net_directional_bias) == math.copysign(1, sign)


def test_agreement_counts_only_directional_dimensions():
    from brain.state.dimensions import normalize_range
    a = directional_state(1.0)
    b = directional_state(
        1.0,
        session_minutes=(350, normalize_range(350, 0, 375)),
        day_of_week=(5, 0.0),
        session_phase=(1.0, 1.0),
        vix=(22.0, 0.6),
        atm_iv=(26.0, 0.6),
    )
    assert a.agreement_factor == pytest.approx(b.agreement_factor)


# ══════════════════════════════════════════════════════════════════════════════
# ATTRIBUTION / CAUSAL CONFIDENCE
# ══════════════════════════════════════════════════════════════════════════════

def _rich_state() -> MarketState:
    sb = StateBuilder()
    ts = market_hours_ts()
    return sb.build("NIFTY", candles=make_candles(t_end=ts), gex_data=GEX,
                    fii_dii={"fii": 1250.0, "dii": -430.0}, vix=13.4,
                    htf_candles=make_candles(60, 23800, seed=7),
                    daily_candles=make_candles(40, 23500, seed=9),
                    quote={"deliveryPct": 58}, as_of=ts)


def test_attribution_reconciles_exactly():
    """The invariant: sum(contributions) + agreement + external == total."""
    st = _rich_state()
    for direction in ("BUY", "SELL"):
        att = attribute(st, direction, external_points=4.0, external_reason="x")
        assert att.verify(1e-9), (att.attributed_sum(), att.total)


def test_confidence_responds_to_every_contribution():
    """
    Before: confidence was 40.82 with all evidence and 40.82 with evidence=[];
    forcing every impact to +1000 moved it only to 43.82. The audit trail had no
    causal link to the score.
    """
    st = _rich_state()
    cc = ConfidenceCalculator()
    bd = cc.calculate_from_state(st, "BUY")
    assert bd.attribution is not None
    assert bd.attribution.verify(1e-4)

    contribs = [c for c in bd.attribution.contributions if abs(c.points) > 1e-9]
    assert contribs, "expected at least one contributing dimension"

    # Removing any single contributing dimension must move the score.
    for c in contribs[:6]:
        clone = MarketState(instrument=st.instrument, market_open=True)
        clone.dimensions = {k: v for k, v in st.dimensions.items() if k != c.dimension}
        clone.compute_quality()
        clone.compute_composites()
        moved = cc.calculate_from_state(clone, "BUY").final
        assert abs(moved - bd.final) > 1e-6, f"{c.dimension} had no effect on the score"


def test_no_double_counting_stages_remain():
    st = _rich_state()
    bd = ConfidenceCalculator().calculate_from_state(st, "BUY")
    assert bd.regime_modifier == 0.0
    assert bd.gex_modifier == 0.0
    assert bd.mtf_modifier == 0.0
    assert bd.velocity_modifier == 0.0
    assert bd.evidence_quality_modifier == 0.0


def test_confidence_does_not_saturate():
    """
    Before: with everything favourable the score pinned at 99 for any net_bias
    >= 60, so the top 40% of the range was indistinguishable.
    """
    cc = ConfidenceCalculator()
    finals = []
    for mag in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s = MarketState(instrument="NIFTY", market_open=True)
        for name in ("ema_stack_score", "supertrend", "htf_trend", "pcr",
                     "vwap_position", "di_differential", "rsi", "macd_histogram"):
            s.set_dimension(name, mag, mag)
        s.set_dimension("adx_value", 45, 1.0)
        s.set_dimension("adx_regime", 1.0, 1.0)
        s.set_dimension("gex_regime", -1.0, 0.95)
        s.compute_quality()
        s.compute_composites()
        finals.append(round(cc.calculate_from_state(s, "BUY").final, 4))
    assert len(set(finals)) == len(finals), finals
    assert all(finals[i] < finals[i + 1] for i in range(len(finals) - 1)), finals


def test_evidence_impacts_carry_the_correct_sign():
    """
    EMA_STACK used abs(normalized) and SUPERTREND ignored sign entirely, so
    bearish evidence reported positive impact.
    """
    from brain.reasoning.evidence_chain import EvidenceChainBuilder
    s = MarketState(instrument="NIFTY", market_open=True)
    s.set_dimension("ema_stack_score", -0.8, -0.8)
    s.set_dimension("supertrend", -1.0, -1.0)
    s.set_dimension("rsi", 38, -0.5)
    s.compute_composites()
    by = {e.factor: e for e in EvidenceChainBuilder().build(s)}
    assert by["EMA_STACK"].confidence_impact < 0
    assert by["SUPERTREND"].confidence_impact < 0
    if "RSI_BEARISH" in by:
        assert by["RSI_BEARISH"].confidence_impact < 0


def test_fii_finding_has_no_literal_placeholder():
    from brain.reasoning.evidence_chain import EvidenceChainBuilder
    s = MarketState(instrument="NIFTY", market_open=True)
    s.set_dimension("fii_flow", 1500.0, 0.7)
    s.compute_composites()
    ev = [e for e in EvidenceChainBuilder().build(s) if e.factor == "FII_FLOW"]
    assert ev, "expected FII evidence"
    assert "{fii" not in ev[0].finding
    assert "1,500" in ev[0].finding


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZERS / CRASH
# ══════════════════════════════════════════════════════════════════════════════

def test_normalize_pct_never_overflows():
    """
    Reachable crash: velocity divides by elapsed wall-clock time, so two ingests
    ~0.25s apart produced a large negative argument and math.exp raised
    OverflowError, turning POST /brain/ingest into a 500.
    """
    for v in (-1e9, -12000.0, -250.0, 0.0, 250.0, 1e9):
        out = normalize_pct(v, 0, 0.3)
        assert -1.0 <= out <= 1.0
    assert normalize_pct(float("nan"), 0, 0.3) == 0.0
    assert normalize_pct(float("inf"), 0, 0.3) == 0.0
    assert normalize_pct(1.0, 0, 0.0) == 0.0


def test_rapid_successive_scans_do_not_raise():
    sb = StateBuilder()
    ts = market_hours_ts()
    for i, pcr in enumerate((1.9, 0.5, 1.8, 0.4)):
        g = dict(GEX); g["pcr"] = pcr
        sb.build("NIFTY", candles=make_candles(seed=i, t_end=ts + i * 0.05),
                 gex_data=g, vix=13.5, as_of=ts + i * 0.05)


def test_percentile_normaliser_distinguishes_saturated_values():
    """
    normalize_threshold(pcr, 0.7, 1.2) pinned every reading above 1.2 to exactly
    +1.0, so a real move from 1.2 to 1.6 was invisible.
    """
    hist = [0.8 + 0.01 * i for i in range(80)] + [1.3, 1.4, 1.5, 1.6, 1.7]
    a = normalize_percentile(1.25, hist)
    b = normalize_percentile(1.65, hist)
    assert a != b and b > a


def test_conviction_dimensions_are_stored_unsigned():
    st = _rich_state()
    for name, dv in st.dimensions.items():
        if name in CONVICTION_DIMENSIONS:
            assert 0.0 <= dv.normalized <= 1.0, (name, dv.normalized)


# ══════════════════════════════════════════════════════════════════════════════
# TIME / SESSION / BACKTESTABILITY
# ══════════════════════════════════════════════════════════════════════════════

def test_session_minutes_distinguishes_closed_from_first_minute():
    assert session_minutes_for(market_hours_ts(hh=3, mm=46)) is None       # pre-open
    assert session_minutes_for(market_hours_ts(hh=16, mm=0)) is None       # post-close
    assert session_minutes_for(market_hours_ts(2026, 8, 29, 12, 0)) is None  # Saturday
    assert session_minutes_for(market_hours_ts(hh=9, mm=20)) == pytest.approx(5.0)


def test_build_is_deterministic_for_a_given_as_of():
    sb1, sb2 = StateBuilder(), StateBuilder()
    ts = market_hours_ts()
    c = make_candles(t_end=ts)
    a = sb1.build("NIFTY", candles=c, gex_data=GEX, vix=13.4, as_of=ts)
    b = sb2.build("NIFTY", candles=c, gex_data=GEX, vix=13.4, as_of=ts)
    assert a.net_directional_bias == pytest.approx(b.net_directional_bias)
    assert a.dimensions["session_minutes"].raw == b.dimensions["session_minutes"].raw


def test_no_clock_veto_fires_inside_the_session():
    """
    Before: session_minutes was clamped with max(0, ...), so any evaluation
    outside 09:30-15:15 IST looked like minute 0 and tripped OPENING_CHAOS as a
    HARD veto. A 400-session sweep produced 400 NO_TRADE and zero signals, which
    made backtesting and pattern-memory seeding impossible.
    """
    sb = StateBuilder()
    ts = market_hours_ts(2026, 8, 24, 12, 15)  # Monday, ~3 days to expiry
    st = sb.build("NIFTY", candles=make_candles(drift=0.0022, vol=0.0022, t_end=ts),
                  gex_data=GEX, fii_dii={"fii": 2200.0, "dii": 400.0}, vix=12.5,
                  htf_candles=make_candles(60, 23600, drift=0.003, seed=3),
                  as_of=ts)
    assert st.market_open is True
    assert st.dimensions["session_minutes"].raw > 15
    chain = ReasoningEngine().reason(st, confidence_threshold=0, as_of=ts)
    for blocked in ("OPENING_CHAOS", "MARKET_CLOSED", "LATE_SESSION"):
        assert not any(blocked in v for v in chain.vetoes), chain.vetoes


def test_replay_produces_signals_over_a_synthetic_history(tmp_path):
    """
    End-to-end proof that the engine is now evaluable over the past. This is the
    capability that did not exist: measured on the original code, 400 synthetic
    sessions produced 400 NO_TRADE results and zero signals.

    Synthetic data is a random walk and therefore has no edge by construction;
    this asserts only that the harness runs and that decisions are reachable.
    """
    from brain.backtest import Backtester, BacktestConfig, SyntheticFeed

    feed = SyntheticFeed(instrument="NIFTY", days=25, seed=11,
                         lookback=140, warmup=160, future_horizon=48)
    bt = Backtester(BacktestConfig(confidence_threshold=50, step=3),
                    db_path=tmp_path / "bt.db")
    res = bt.run(feed)
    assert res.evaluations > 200
    assert res.signals > 0
    m = res.metrics()
    assert m["fills"] >= 1
    # Every scored fill must carry a determinable P&L, never a placeholder.
    for f in res.scored:
        assert f.pnl_pct is not None
    # Uncertainty must be reported alongside any hit rate.
    if m.get("hit_rate"):
        assert m["hit_rate"]["ci_upper_pct"] >= m["hit_rate"]["ci_lower_pct"]


def test_market_closed_blocks_new_entries():
    sb = StateBuilder()
    ts = market_hours_ts(hh=3, mm=46)
    st = sb.build("NIFTY", candles=make_candles(t_end=ts), gex_data=GEX, as_of=ts)
    assert st.market_open is False
    chain = ReasoningEngine().reason(st, confidence_threshold=0, as_of=ts)
    assert any("MARKET_CLOSED" in v for v in chain.vetoes)
    assert chain.direction == "NO_TRADE"


def test_dte_is_populated_so_expiry_rule_can_fire():
    """dte was never set, so raw("dte", 5) always returned 5 and Rule 4 was dead."""
    sb = StateBuilder()
    ts = market_hours_ts()
    st = sb.build("NIFTY", candles=make_candles(t_end=ts), gex_data=GEX, as_of=ts)
    assert "dte" in st.dimensions
    assert 0.0 <= st.dimensions["dte"].raw <= 7.5

    # Force expiry day with a weak signal and confirm the veto fires.
    expiry_ts = market_hours_ts(2026, 8, 27, 12, 0)  # Thursday
    st2 = sb.build("NIFTY", candles=make_candles(t_end=expiry_ts, seed=5),
                   gex_data=GEX, as_of=expiry_ts)
    assert st2.dimensions["dte"].raw <= 1.2
    vetoes = BlunderGuard().evaluate(st2, "BUY", confidence=65)
    assert any(v.name == "EXPIRY_DAY_WEAK" for v in vetoes)


@pytest.mark.parametrize("kwarg,value,expected", [
    ("session_signals", 3, "MAX_DAILY_SIGNALS"),
    ("session_stops", 2, "CIRCUIT_BREAKER"),
    ("premium_already_moved_pct", 35.0, "PREMIUM_MOVED"),
    ("live_premium_cost", 150000.0, "CAPITAL_EXCEEDED"),
])
def test_previously_unreachable_capital_rules_fire(kwarg, value, expected):
    """
    Five of the fourteen rules were unreachable through the API, and they were
    the ones bounding financial loss: overtrading, consecutive stops, position
    size, expiry theta and chase protection.
    """
    sb = StateBuilder()
    ts = market_hours_ts()
    st = sb.build("NIFTY", candles=make_candles(t_end=ts), gex_data=GEX, as_of=ts)
    vetoes = BlunderGuard().evaluate(st, "BUY", confidence=85, **{kwarg: value})
    assert any(v.name == expected for v in vetoes), [v.name for v in vetoes]


def test_next_weekly_expiry_is_in_the_future():
    for hh in (9, 12, 15):
        ts = market_hours_ts(hh=hh)
        assert 0.0 <= next_weekly_expiry_dte(ts) <= 7.0


# ══════════════════════════════════════════════════════════════════════════════
# OPTION PRICING / OUTCOMES
# ══════════════════════════════════════════════════════════════════════════════

def test_black_scholes_satisfies_put_call_parity():
    spot, strike, iv, dte, r = 24000.0, 24000.0, 15.0, 7.0, 0.065
    c = black_scholes(spot, strike, iv, dte, "CE", r)
    p = black_scholes(spot, strike, iv, dte, "PE", r)
    T = dte / 365.0
    lhs = c.price - p.price
    rhs = spot - strike * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-6)


def test_greeks_have_the_expected_signs():
    c = black_scholes(24000, 24000, 15, 7, "CE")
    p = black_scholes(24000, 24000, 15, 7, "PE")
    assert 0 < c.delta < 1
    assert -1 < p.delta < 0
    assert c.gamma > 0 and p.gamma > 0
    assert c.theta < 0 and p.theta < 0   # long options decay
    assert c.vega > 0 and p.vega > 0


def test_theta_makes_a_flat_round_trip_a_loss():
    """
    The core reason a round trip is not a win: holding a long option while spot
    returns to entry loses the time value. The old `move_atr * 80` proxy could
    not express this, and the old labelling booked it as WIN_T1.
    """
    pnl = implied_pnl_pct(entry_spot=24000, exit_spot=24000, strike=24000,
                          opt_type="CE", iv_pct_entry=15.0, dte_days_entry=3.0,
                          minutes_held=45.0)
    assert pnl is not None and pnl < 0


def test_pnl_is_none_when_it_cannot_be_determined():
    """
    Unknown must stay unknown. The old fallback replaced it with
    `move_atr * 80`, and because tracking began with entry_premium=0 that
    invented figure was what actually reached the database.
    """
    # No IV -> cannot price the option at all.
    assert implied_pnl_pct(24000, 24100, 24000, "CE", 0.0, 3.0, 30.0) is None
    # At expiry an ATM option is worth zero, so a percentage change is undefined.
    assert implied_pnl_pct(24000, 24100, 24000, "CE", 15.0, 0.0, 30.0) is None
    # An ITM option at expiry has intrinsic value, so it is priceable.
    assert implied_pnl_pct(24000, 24100, 23800, "CE", 15.0, 0.02, 5.0) is not None
    # A normal case is priceable.
    assert implied_pnl_pct(24000, 24200, 24000, "CE", 15.0, 3.0, 30.0) is not None

    assert realised_pnl_pct(0.0, 120.0) is None
    assert realised_pnl_pct(None, 120.0) is None
    assert realised_pnl_pct(100.0, 120.0) == pytest.approx(19.0)


def test_round_trip_to_entry_is_scratch_not_a_win():
    db = PatternDB(Path("/tmp/_corr_rt.db"))
    t = ActiveTrade(signal_id=1, instrument="NIFTY", direction="BUY",
                    entry_spot=24000, entry_premium=100, strike=24000,
                    opt_type="CE", atr=100)
    ot = OutcomeTracker(db)
    ot.active_trades[1] = t
    out = None
    for spot in (24050, 24100, 24150, 24120, 24080, 24010, 24001):
        out = ot._check_trade(t, spot) or out
    assert out == "SCRATCH"
    assert "SCRATCH" in NEUTRAL_OUTCOMES
    assert outcome_semantic("SCRATCH") == "neutral"


def test_atm_strike_snaps_to_the_step():
    assert atm_strike(24037, 50) == 24050
    assert atm_strike(24024, 50) == 24000
    assert atm_strike(52180, 100) == 52200


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def test_wilson_interval_for_nine_of_twelve_is_wide():
    """
    9/12 was reported as "Strong historical edge (75% win rate, 12 trades)" and
    earned +8 confidence. The interval spans break-even.
    """
    est = wilson_interval(9, 12, 0.95)
    assert est.point == pytest.approx(0.75)
    assert est.lower < 0.55 < est.upper
    assert est.width > 0.35


def test_small_samples_cannot_buy_a_confidence_boost():
    from brain.memory.matcher import PatternMatcher
    from brain.memory.pattern_db import PatternStats

    pm = PatternMatcher(db=None)
    small = PatternStats(total_trades=12, win_rate=75.0)
    mod, reason = pm._confidence_adjustment(small)
    assert mod == 0.0
    assert "Insufficient history" in reason

    big = PatternStats(total_trades=200, win_rate=75.0)
    mod2, _ = pm._confidence_adjustment(big)
    assert mod2 > 0


def test_clearly_losing_pattern_is_penalised():
    from brain.memory.matcher import PatternMatcher
    from brain.memory.pattern_db import PatternStats
    pm = PatternMatcher(db=None)
    mod, reason = pm._confidence_adjustment(PatternStats(total_trades=80, win_rate=20.0))
    assert mod < 0
    assert "below break-even" in reason


def test_degradation_needs_significance():
    assert two_proportion_z(6, 10, 8, 10)["significant_at_05"] is False
    assert two_proportion_z(20, 200, 120, 200)["significant_at_05"] is True


def test_sample_size_requirement_is_honest():
    assert min_samples_for_width(0.20) >= 90


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def test_calibrator_is_identity_until_it_has_data():
    assert PlattCalibrator().fit([70] * 10, [1] * 10).fitted is False
    assert PlattCalibrator().predict(70) == pytest.approx(0.70)
    assert IsotonicCalibrator().predict(40) == pytest.approx(0.40)


def test_platt_recovers_a_known_relationship():
    rng = random.Random(3)
    scores, outs = [], []
    for _ in range(1500):
        s = rng.uniform(0, 100)
        p_true = 1 / (1 + math.exp(-(0.06 * s - 3.0)))
        scores.append(s)
        outs.append(1 if rng.random() < p_true else 0)
    cal = PlattCalibrator().fit(scores, outs)
    assert cal.fitted
    assert cal.predict(20) < cal.predict(50) < cal.predict(85)
    m = evaluate([cal.predict(s) for s in scores], outs)
    assert m.brier < 0.25          # better than always predicting 0.5
    assert m.ece < 0.06


def test_isotonic_is_monotone():
    rng = random.Random(5)
    scores, outs = [], []
    for _ in range(800):
        s = rng.uniform(0, 100)
        scores.append(s)
        outs.append(1 if rng.random() < min(0.95, max(0.05, s / 120)) else 0)
    cal = IsotonicCalibrator().fit(scores, outs)
    assert cal.fitted
    preds = [cal.predict(x) for x in range(0, 101, 5)]
    assert all(preds[i] <= preds[i + 1] + 1e-9 for i in range(len(preds) - 1))


def test_fit_calibration_uses_a_chronological_holdout():
    rng = random.Random(11)
    scores = [rng.uniform(0, 100) for _ in range(600)]
    outs = [1 if rng.random() < s / 130 else 0 for s in scores]
    model = fit_calibration(scores, outs, holdout_fraction=0.25)
    assert model.n_train == 450 and model.n_holdout == 150
    assert model.holdout_metrics is not None
    assert 0.0 <= model.probability(75.0) <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def _seed(db: PatternDB, st: MarketState, n: int, wins: int, base_ts: float):
    for i in range(n):
        sid = db.record_signal(state=st, direction="BUY", confidence=70,
                               entry_spot=24000, atr=100, timestamp=base_ts + i)
        win = i < wins
        db.record_outcome(sid, "WIN_T1" if win else "STOP_LOSS",
                          24100 if win else 23880, 0,
                          1.0 if win else -1.2, 45,
                          40.0 if win else -55.0, pnl_source="observed")


def test_walk_forward_guard_excludes_the_future(tmp_path):
    db = PatternDB(tmp_path / "wf.db")
    st = _rich_state()
    base = market_hours_ts()
    _seed(db, st, 10, 8, base)
    assert len(db.find_similar(st, "BUY", as_of=base + 5)) == 5
    assert len(db.find_similar(st, "BUY", as_of=base)) == 0
    assert len(db.find_similar(st, "BUY", as_of=None)) == 10


def test_unavailable_pnl_rows_are_excluded_from_statistics(tmp_path):
    db = PatternDB(tmp_path / "np.db")
    st = _rich_state()
    base = market_hours_ts()
    sid = db.record_signal(state=st, direction="BUY", confidence=70,
                           entry_spot=24000, atr=100, timestamp=base)
    db.record_outcome(sid, "WIN_T1", 24100, 0, 1.0, 45,
                      pnl_pct=None, pnl_source="unavailable")
    assert db.find_similar(st, "BUY", require_pnl=True) == []
    assert len(db.find_similar(st, "BUY", require_pnl=False)) == 1


def test_matcher_never_substitutes_another_instrument(tmp_path):
    from brain.memory.matcher import PatternMatcher
    db = PatternDB(tmp_path / "xi.db")
    st = _rich_state()
    other = _rich_state()
    other.instrument = "BANKNIFTY"
    _seed(db, other, 40, 30, market_hours_ts())
    ctx = PatternMatcher(db).get_context(st, "BUY")
    assert ctx.similar_setups == 0
    assert ctx.instrument_scope == "NIFTY"
    assert ctx.confidence_modifier == 0.0


def test_similar_states_still_match_after_a_one_band_drift(tmp_path):
    """
    gex_regime and trend_dir were equality filters, so a one-bucket drift
    returned zero rows: identical matched 12, "gex flipped" matched 0.
    """
    db = PatternDB(tmp_path / "soft.db")
    st = _rich_state()
    _seed(db, st, 20, 14, market_hours_ts())

    flipped = _rich_state()
    flipped.set_dimension("gex_regime", 1.0, 0.35)   # Positive gamma instead
    flipped.compute_composites()
    assert len(db.find_similar(flipped, "BUY", min_match=0.5)) > 0
