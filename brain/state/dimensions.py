"""
SignalsBrain — Dimension Definitions

The 47 dimensions that define the complete market state at any instant.
Each dimension has:
  - name: unique identifier
  - category: grouping for analysis
  - range: expected value range
  - normalizer: function to map raw value to [-1, +1] for cross-comparison
  - weight: how much this dimension matters for signal generation
  - velocity_relevant: whether rate-of-change matters (not just level)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
import math


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
    description: str
    weight: float  # 0-10, how important for signal generation
    velocity_relevant: bool = True  # Track rate-of-change?
    min_val: float = -1.0
    max_val: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# ALL 47 DIMENSIONS
# ═══════════════════════════════════════════════════════════════════════════════

DIMENSIONS: dict[str, Dimension] = {}


def _d(name: str, cat: DimensionCategory, desc: str, weight: float, vel: bool = True):
    dim = Dimension(name=name, category=cat, description=desc, weight=weight, velocity_relevant=vel)
    DIMENSIONS[name] = dim
    return dim


# ── Price Structure (8) ───────────────────────────────────────────────────────
_d("ltp", DimensionCategory.PRICE, "Last traded price (absolute)", 0, vel=False)
_d("day_change_pct", DimensionCategory.PRICE, "Intraday price change %", 6, vel=True)
_d("day_range_position", DimensionCategory.PRICE, "Position within today's high-low range (0=low, 1=high)", 5, vel=True)
_d("range_20d_position", DimensionCategory.PRICE, "Position within 20-day high-low range", 4, vel=False)
_d("vwap_deviation", DimensionCategory.PRICE, "% deviation from VWAP (institutional benchmark)", 7, vel=True)
_d("ema_distance", DimensionCategory.PRICE, "Composite distance from EMA9/21/50/200 (normalized)", 6, vel=True)
_d("orb_status", DimensionCategory.PRICE, "Opening Range Breakout status (-1=breakdown, 0=inside, 1=breakout)", 5, vel=False)
_d("sr_proximity", DimensionCategory.PRICE, "Distance to nearest swing S/R as fraction of ATR", 4, vel=False)

# ── Trend (7) ─────────────────────────────────────────────────────────────────
_d("ema_stack_score", DimensionCategory.TREND, "EMA alignment score: +1=perfect bull stack, -1=perfect bear stack", 8, vel=True)
_d("supertrend", DimensionCategory.TREND, "SuperTrend direction: +1=bullish, -1=bearish", 7, vel=False)
_d("adx_value", DimensionCategory.TREND, "ADX trend strength (0-100)", 7, vel=True)
_d("adx_regime", DimensionCategory.TREND, "Market regime: trending(+1)/developing(0)/choppy(-1)", 8, vel=False)
_d("di_differential", DimensionCategory.TREND, "+DI minus -DI (directional strength)", 6, vel=True)
_d("htf_trend", DimensionCategory.TREND, "Higher timeframe (1H) trend: +1=bull, -1=bear", 7, vel=False)
_d("trend_acceleration", DimensionCategory.TREND, "Is ADX rising (+1) or falling (-1)?", 5, vel=False)

# ── Momentum (6) ──────────────────────────────────────────────────────────────
_d("rsi", DimensionCategory.MOMENTUM, "RSI(14) value normalized: >60=bull, <40=bear", 6, vel=True)
_d("rsi_divergence", DimensionCategory.MOMENTUM, "Price-RSI divergence: +1=bullish div, -1=bearish div, 0=none", 7, vel=False)
_d("macd_histogram", DimensionCategory.MOMENTUM, "MACD histogram value (normalized by ATR)", 6, vel=True)
_d("macd_direction", DimensionCategory.MOMENTUM, "MACD hist expanding(+1) or contracting(-1)", 5, vel=False)
_d("roc_5", DimensionCategory.MOMENTUM, "5-bar rate of change (%)", 5, vel=True)
_d("stochastic_zone", DimensionCategory.MOMENTUM, "Stochastic %K zone: overbought(+1)/neutral(0)/oversold(-1)", 4, vel=True)

# ── Options Microstructure (12) — THE EDGE NO HUMAN CAN TRACK ─────────────────
_d("pcr", DimensionCategory.OPTIONS, "Put-Call Ratio from OI. >1.2=bullish support, <0.7=bearish", 9, vel=True)
_d("pcr_velocity", DimensionCategory.OPTIONS, "Rate of PCR change (per scan). Rising=more puts writing=support building", 10, vel=False)
_d("atm_iv", DimensionCategory.OPTIONS, "ATM implied volatility (%)", 6, vel=True)
_d("iv_percentile", DimensionCategory.OPTIONS, "Current IV vs historical range (0-100)", 7, vel=False)
_d("iv_skew", DimensionCategory.OPTIONS, "Put IV minus Call IV. Positive=fear/hedging, Negative=greed", 8, vel=True)
_d("gex_regime", DimensionCategory.OPTIONS, "Gamma Exposure regime: +1=Positive(stabilize), -1=Negative(amplify)", 10, vel=False)
_d("gex_net", DimensionCategory.OPTIONS, "Net GEX in ₹Cr (magnitude of dealer exposure)", 6, vel=True)
_d("gex_flip_distance", DimensionCategory.OPTIONS, "Distance to GEX flip point in ATR units (signed: +above, -below)", 9, vel=True)
_d("call_wall_distance", DimensionCategory.OPTIONS, "Distance to call wall (resistance magnet) in ATR", 5, vel=False)
_d("put_wall_distance", DimensionCategory.OPTIONS, "Distance to put wall (support magnet) in ATR", 5, vel=False)
_d("max_pain_distance", DimensionCategory.OPTIONS, "Distance to max pain level in ATR (signed)", 4, vel=False)
_d("oi_buildup", DimensionCategory.OPTIONS, "OI buildup interpretation: long_buildup(+1)/short_covering(+0.5)/short_buildup(-1)/long_unwinding(-0.5)", 8, vel=True)

# ── Volume & Flow (6) ─────────────────────────────────────────────────────────
_d("volume_ratio", DimensionCategory.FLOW, "Current volume / 20-bar avg volume", 6, vel=True)
_d("volume_trend", DimensionCategory.FLOW, "Volume accelerating(+1) or decelerating(-1)", 5, vel=False)
_d("vwap_position", DimensionCategory.FLOW, "Above VWAP (+1) or below (-1)", 7, vel=True)
_d("delivery_pct", DimensionCategory.FLOW, "Delivery % (high=conviction, low=speculation)", 3, vel=False)
_d("fii_flow", DimensionCategory.FLOW, "FII net flow direction: buying(+1)/selling(-1)/neutral(0)", 7, vel=False)
_d("dii_flow", DimensionCategory.FLOW, "DII net flow direction", 4, vel=False)

# ── Volatility (4) ────────────────────────────────────────────────────────────
_d("vix", DimensionCategory.VOLATILITY, "India VIX level", 6, vel=True)
_d("vix_change", DimensionCategory.VOLATILITY, "VIX change today (rising=fear, falling=complacency)", 7, vel=True)
_d("bb_width", DimensionCategory.VOLATILITY, "Bollinger Band width % (squeeze < 1.5%)", 5, vel=True)
_d("atr_pct", DimensionCategory.VOLATILITY, "ATR as % of price (market alive/dead threshold)", 6, vel=False)

# ── Time & Context (4) ────────────────────────────────────────────────────────
_d("session_minutes", DimensionCategory.CONTEXT, "Minutes since market open (0-375)", 3, vel=False)
_d("dte", DimensionCategory.CONTEXT, "Days to nearest expiry", 6, vel=False)
_d("day_of_week", DimensionCategory.CONTEXT, "Day (1=Mon...5=Fri). Expiry days have different character.", 3, vel=False)
_d("session_phase", DimensionCategory.CONTEXT, "Phase: opening(0-15min)/morning(15-120)/midday(120-240)/afternoon(240-330)/closing(330-375)", 4, vel=False)


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZERS — map raw values to [-1, +1] for cross-comparison
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_pct(value: float, center: float = 0, scale: float = 2.0) -> float:
    """Sigmoid-like normalization centered at `center`, scaled by `scale`."""
    x = (value - center) / scale
    return max(-1.0, min(1.0, 2 / (1 + math.exp(-x)) - 1))


def normalize_range(value: float, lo: float, hi: float) -> float:
    """Linear map [lo, hi] → [-1, +1]."""
    if hi <= lo:
        return 0.0
    return max(-1.0, min(1.0, 2 * (value - lo) / (hi - lo) - 1))


def normalize_threshold(value: float, bearish_thresh: float, bullish_thresh: float) -> float:
    """Below bearish = -1, above bullish = +1, linear between."""
    if value <= bearish_thresh:
        return -1.0
    if value >= bullish_thresh:
        return 1.0
    mid = (bearish_thresh + bullish_thresh) / 2
    half = (bullish_thresh - bearish_thresh) / 2
    return (value - mid) / half


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION WEIGHTS (for composite scoring)
# Categories contribute to the total signal score weighted by their domain:
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORY_WEIGHTS = {
    DimensionCategory.PRICE: 15,
    DimensionCategory.TREND: 20,
    DimensionCategory.MOMENTUM: 10,
    DimensionCategory.OPTIONS: 25,  # THE key edge — no human can process this in real-time
    DimensionCategory.FLOW: 12,
    DimensionCategory.VOLATILITY: 8,
    DimensionCategory.CONTEXT: 10,
}

# Total = 100
assert sum(CATEGORY_WEIGHTS.values()) == 100
