"""
SignalsBrain — Dimension Definitions

The dimensions that define the complete market state at any instant.

## Channels (the central correctness invariant)

Every dimension belongs to exactly one CHANNEL, and the channel decides how the
dimension is allowed to influence a decision:

  DIRECTION  — signed [-1,+1]. Carries genuine bull/bear meaning. These are the
               ONLY inputs to the directional score.
  CONVICTION — unsigned [0,1]. Carries "how much do we trust a directional read
               right now" (trend strength, participation, volatility regime).
               These SCALE the directional score. They never add to it.
  CONTEXT    — gates and sizing inputs (clock, expiry, IV level, GEX geometry).
               These never touch the directional score at all.

Why this matters: previously ADX (trend *strength*, no direction), volatility
magnitude, IV level, volume magnitude and even day-of-week were summed into a
signed "directional" bias. That produced two systematic defects:

  1. day_of_week normalised to [-1,+1] made Monday mechanically bearish and
     Friday mechanically bullish, worth ~33 confidence points on identical
     evidence.
  2. adx_value/adx_regime were added as POSITIVE contributions, so a stronger
     bear trend produced a WEAKER bearish signal (net_bias -17.8 at ADX 12 vs
     -3.0 at ADX 50) — monotonically backwards.

Splitting the channels fixes both by construction: a non-directional quantity
has no way to express a direction.

Each dimension has:
  - name: unique identifier
  - channel: DIRECTION / CONVICTION / CONTEXT
  - category: grouping for analysis and reporting
  - weight: importance within its channel (0-10)
  - velocity_relevant: whether rate-of-change matters (not just level)
"""

from dataclasses import dataclass
from enum import Enum
import math


class Channel(str, Enum):
    """How a dimension is permitted to influence the decision."""
    DIRECTION = "direction"    # signed, drives bull/bear
    CONVICTION = "conviction"  # unsigned, scales the directional read
    CONTEXT = "context"        # gates / sizing only, never scores


class DimensionCategory(str, Enum):
    PRICE = "price_structure"
    TREND = "trend"
    MOMENTUM = "momentum"
    OPTIONS = "options_microstructure"
    FLOW = "volume_flow"
    VOLATILITY = "volatility"
    CONTEXT = "time_context"


@dataclass(frozen=True)
class Dimension:
    name: str
    category: DimensionCategory
    channel: Channel
    description: str
    weight: float  # 0-10, importance within its channel
    velocity_relevant: bool = True
    min_val: float = -1.0
    max_val: float = 1.0

    @property
    def is_directional(self) -> bool:
        return self.channel is Channel.DIRECTION


DIMENSIONS: dict[str, Dimension] = {}


def _d(name: str, cat: DimensionCategory, chan: Channel, desc: str,
       weight: float, vel: bool = True) -> Dimension:
    dim = Dimension(name=name, category=cat, channel=chan, description=desc,
                    weight=weight, velocity_relevant=vel)
    DIMENSIONS[name] = dim
    return dim


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTION CHANNEL — signed, genuine bull/bear meaning
# ═══════════════════════════════════════════════════════════════════════════════

# ── Price Structure ───────────────────────────────────────────────────────────
_d("day_change_pct", DimensionCategory.PRICE, Channel.DIRECTION,
   "Intraday price change % vs session open", 6)
_d("day_range_position", DimensionCategory.PRICE, Channel.DIRECTION,
   "Position within today's high-low range (-1=low, +1=high)", 5)
_d("range_20d_position", DimensionCategory.PRICE, Channel.DIRECTION,
   "Position within 20-day high-low range", 4, vel=False)
_d("vwap_deviation", DimensionCategory.PRICE, Channel.DIRECTION,
   "% deviation from session VWAP (institutional benchmark)", 7)
_d("ema_distance", DimensionCategory.PRICE, Channel.DIRECTION,
   "Composite signed distance from EMA9/21/50/200", 6)
_d("orb_status", DimensionCategory.PRICE, Channel.DIRECTION,
   "Opening Range Breakout status (-1=breakdown, 0=inside, +1=breakout)", 5, vel=False)

# ── Trend ─────────────────────────────────────────────────────────────────────
_d("ema_stack_score", DimensionCategory.TREND, Channel.DIRECTION,
   "EMA alignment: +1=perfect bull stack, -1=perfect bear stack", 8)
_d("supertrend", DimensionCategory.TREND, Channel.DIRECTION,
   "SuperTrend direction: +1=bullish, -1=bearish", 7, vel=False)
_d("di_differential", DimensionCategory.TREND, Channel.DIRECTION,
   "+DI minus -DI (signed directional strength)", 6)
_d("htf_trend", DimensionCategory.TREND, Channel.DIRECTION,
   "Higher timeframe (1H) trend: +1=bull, -1=bear", 7, vel=False)

# ── Momentum ──────────────────────────────────────────────────────────────────
_d("rsi", DimensionCategory.MOMENTUM, Channel.DIRECTION,
   "RSI(14) mapped signed: >60 bullish, <40 bearish", 6)
_d("rsi_divergence", DimensionCategory.MOMENTUM, Channel.DIRECTION,
   "Price-RSI divergence: +1=bullish div, -1=bearish div", 7, vel=False)
_d("macd_histogram", DimensionCategory.MOMENTUM, Channel.DIRECTION,
   "MACD histogram value (ATR-normalised)", 6)
_d("macd_direction", DimensionCategory.MOMENTUM, Channel.DIRECTION,
   "MACD histogram sign/expansion", 5, vel=False)
_d("roc_5", DimensionCategory.MOMENTUM, Channel.DIRECTION,
   "5-bar rate of change (%)", 5)
_d("stochastic_zone", DimensionCategory.MOMENTUM, Channel.DIRECTION,
   "Stochastic %K zone, signed", 4)

# ── Options Microstructure ────────────────────────────────────────────────────
_d("pcr", DimensionCategory.OPTIONS, Channel.DIRECTION,
   "Put-Call Ratio from OI. High=put support building=bullish", 9)
_d("pcr_velocity", DimensionCategory.OPTIONS, Channel.DIRECTION,
   "Rate of PCR change. Rising=support building", 10, vel=False)
_d("iv_skew", DimensionCategory.OPTIONS, Channel.DIRECTION,
   "Put IV minus Call IV. Positive=fear/hedging=bearish", 8)
_d("oi_buildup", DimensionCategory.OPTIONS, Channel.DIRECTION,
   "OI buildup: long_buildup(+1)/short_covering(+.5)/short_buildup(-1)/long_unwinding(-.5)", 8)
_d("call_wall_distance", DimensionCategory.OPTIONS, Channel.DIRECTION,
   "Proximity to call wall (resistance magnet). Close=capped upside=bearish", 5, vel=False)
_d("put_wall_distance", DimensionCategory.OPTIONS, Channel.DIRECTION,
   "Proximity to put wall (support magnet). Close=support=bullish", 5, vel=False)
_d("max_pain_distance", DimensionCategory.OPTIONS, Channel.DIRECTION,
   "Signed distance to max pain; pin pull is toward max pain", 4, vel=False)

# ── Volume & Flow ─────────────────────────────────────────────────────────────
_d("vwap_position", DimensionCategory.FLOW, Channel.DIRECTION,
   "Above VWAP (+1) or below (-1)", 7)
_d("fii_flow", DimensionCategory.FLOW, Channel.DIRECTION,
   "FII net flow direction: buying(+1)/selling(-1)", 7, vel=False)
_d("dii_flow", DimensionCategory.FLOW, Channel.DIRECTION,
   "DII net flow direction", 4, vel=False)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVICTION CHANNEL — unsigned [0,1], scales the directional read
# ═══════════════════════════════════════════════════════════════════════════════

_d("adx_value", DimensionCategory.TREND, Channel.CONVICTION,
   "ADX trend STRENGTH (0-100). Direction-free.", 8)
_d("adx_regime", DimensionCategory.TREND, Channel.CONVICTION,
   "Regime: trending/developing/choppy. Direction-free.", 7, vel=False)
_d("trend_acceleration", DimensionCategory.TREND, Channel.CONVICTION,
   "Is ADX rising (trend building) or falling (trend decaying)?", 5, vel=False)
_d("volume_ratio", DimensionCategory.FLOW, Channel.CONVICTION,
   "Current volume / 20-bar average. Participation, not direction.", 6)
_d("volume_trend", DimensionCategory.FLOW, Channel.CONVICTION,
   "Volume accelerating or decelerating", 5, vel=False)
_d("delivery_pct", DimensionCategory.FLOW, Channel.CONVICTION,
   "Delivery % (high=conviction, low=speculation)", 3, vel=False)
_d("atr_pct", DimensionCategory.VOLATILITY, Channel.CONVICTION,
   "ATR as % of price. Is the market alive enough to reach a target?", 7, vel=False)
_d("bb_width", DimensionCategory.VOLATILITY, Channel.CONVICTION,
   "Bollinger Band width % (squeeze = coiled energy)", 5)
_d("gex_regime", DimensionCategory.OPTIONS, Channel.CONVICTION,
   "Gamma regime: Negative=dealers amplify (raises conviction), "
   "Positive=dealers suppress (lowers conviction). Direction-free.", 10, vel=False)


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT CHANNEL — gates and sizing only. Never scores.
# ═══════════════════════════════════════════════════════════════════════════════

_d("ltp", DimensionCategory.PRICE, Channel.CONTEXT, "Last traded price (absolute)", 0, vel=False)
_d("sr_proximity", DimensionCategory.PRICE, Channel.CONTEXT,
   "Unsigned distance to nearest swing S/R in ATR units", 4, vel=False)
_d("atr_percentile", DimensionCategory.VOLATILITY, Channel.CONTEXT,
   "ATR% ranked against this instrument's own recent history (0-1). Timeframe-"
   "agnostic liveness gate; a fixed ATR% threshold is interval-dependent.", 0, vel=False)
_d("gex_net", DimensionCategory.OPTIONS, Channel.CONTEXT,
   "Net GEX magnitude (Rs Cr) — size of dealer exposure", 6)
_d("gex_flip_distance", DimensionCategory.OPTIONS, Channel.CONTEXT,
   "Distance to GEX flip in ATR units — regime-transition geometry", 9)
_d("atm_iv", DimensionCategory.OPTIONS, Channel.CONTEXT, "ATM implied volatility (%)", 6)
_d("iv_percentile", DimensionCategory.OPTIONS, Channel.CONTEXT,
   "Current IV vs its own history (0-100) — cost of optionality", 7, vel=False)
_d("vix", DimensionCategory.VOLATILITY, Channel.CONTEXT, "India VIX level", 6)
_d("vix_change", DimensionCategory.VOLATILITY, Channel.CONTEXT,
   "VIX change today — risk gate, not a direction", 7)
_d("session_minutes", DimensionCategory.CONTEXT, Channel.CONTEXT,
   "Minutes since market open (0-375)", 3, vel=False)
_d("dte", DimensionCategory.CONTEXT, Channel.CONTEXT, "Days to nearest expiry", 6, vel=False)
_d("day_of_week", DimensionCategory.CONTEXT, Channel.CONTEXT,
   "Day (1=Mon..5=Fri). Expiry days differ in character.", 3, vel=False)
_d("session_phase", DimensionCategory.CONTEXT, Channel.CONTEXT,
   "opening/morning/midday/afternoon/closing", 4, vel=False)


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL INDEXES
# ═══════════════════════════════════════════════════════════════════════════════

DIRECTION_DIMENSIONS = {n: d for n, d in DIMENSIONS.items() if d.channel is Channel.DIRECTION}
CONVICTION_DIMENSIONS = {n: d for n, d in DIMENSIONS.items() if d.channel is Channel.CONVICTION}
CONTEXT_DIMENSIONS = {n: d for n, d in DIMENSIONS.items() if d.channel is Channel.CONTEXT}


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZERS — all clamped and overflow-safe
# ═══════════════════════════════════════════════════════════════════════════════

# math.exp overflows above ~709. Clamping the logistic argument well inside that
# bound keeps normalize_pct total: previously a large negative input (reachable
# whenever two scans arrived close together, since velocity divides by elapsed
# wall-clock time) raised OverflowError and turned POST /brain/ingest into a 500.
_LOGISTIC_CLAMP = 60.0


def _safe_logistic(x: float) -> float:
    """Logistic mapped to [-1,+1], saturating instead of overflowing."""
    if x != x:  # NaN
        return 0.0
    if x >= _LOGISTIC_CLAMP:
        return 1.0
    if x <= -_LOGISTIC_CLAMP:
        return -1.0
    return 2.0 / (1.0 + math.exp(-x)) - 1.0


def normalize_pct(value: float, center: float = 0, scale: float = 2.0) -> float:
    """Signed logistic normalisation centred at `center`, scaled by `scale`."""
    if scale == 0 or scale != scale:
        return 0.0
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return _safe_logistic((value - center) / scale)


def normalize_range(value: float, lo: float, hi: float) -> float:
    """Linear map [lo, hi] -> [-1, +1], clamped."""
    if hi <= lo or value != value:
        return 0.0
    return max(-1.0, min(1.0, 2 * (value - lo) / (hi - lo) - 1))


def normalize_unit(value: float, lo: float, hi: float) -> float:
    """Linear map [lo, hi] -> [0, 1], clamped. For CONVICTION dimensions."""
    if hi <= lo or value != value:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def normalize_threshold(value: float, bearish_thresh: float, bullish_thresh: float) -> float:
    """Below bearish = -1, above bullish = +1, linear between."""
    if value != value:
        return 0.0
    if value <= bearish_thresh:
        return -1.0
    if value >= bullish_thresh:
        return 1.0
    mid = (bearish_thresh + bullish_thresh) / 2
    half = (bullish_thresh - bearish_thresh) / 2
    if half == 0:
        return 0.0
    return (value - mid) / half


def normalize_percentile(value: float, history: list[float]) -> float:
    """
    Rank `value` against its own recent distribution -> [-1, +1].

    Replaces hard-coded saturating thresholds. normalize_threshold(pcr, 0.7, 1.2)
    pins every reading above 1.2 to exactly +1.0, so a genuine move from 1.2 to
    1.6 registered as zero change and was invisible to velocity. A percentile
    rank keeps those readings distinguishable and self-adjusts per instrument.
    """
    if not history:
        return 0.0
    clean = [h for h in history if h == h]
    if not clean:
        return 0.0
    below = sum(1 for h in clean if h < value)
    equal = sum(1 for h in clean if h == value)
    pct = (below + 0.5 * equal) / len(clean)
    return max(-1.0, min(1.0, pct * 2 - 1))


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY WEIGHTS
#
# Only DIRECTION categories contribute to the directional score. Volatility and
# time_context are deliberately absent: they hold no directional dimensions.
# ═══════════════════════════════════════════════════════════════════════════════

DIRECTION_CATEGORY_WEIGHTS = {
    DimensionCategory.PRICE: 18,
    DimensionCategory.TREND: 22,
    DimensionCategory.MOMENTUM: 14,
    DimensionCategory.OPTIONS: 30,   # the structural edge
    DimensionCategory.FLOW: 16,
}
assert sum(DIRECTION_CATEGORY_WEIGHTS.values()) == 100

# Retained under the original name for backward compatibility with existing
# imports. It now contains only directional categories.
CATEGORY_WEIGHTS = dict(DIRECTION_CATEGORY_WEIGHTS)


def directional_weight_total() -> float:
    return sum(d.weight for d in DIRECTION_DIMENSIONS.values())


def conviction_weight_total() -> float:
    return sum(d.weight for d in CONVICTION_DIMENSIONS.values())
