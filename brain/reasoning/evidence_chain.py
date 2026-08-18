"""
SignalsBrain — Evidence Chain Builder

The brain doesn't just produce a number. It produces a COURT CASE:
  - Prosecution (reasons FOR the trade)
  - Defense (reasons AGAINST)
  - Witnesses (pattern memory, velocity alerts, regime context)
  - Verdict (final decision with confidence)

Every AI model that connects to SignalsBrain receives this full evidence chain.
This is what transforms "SELL at 78%" into a reasoned, auditable decision
that a model can intelligently agree with, challenge, or refine.

The evidence chain structure:
  1. PRIMARY EVIDENCE — the 3-5 strongest factors driving the signal
  2. SUPPORTING EVIDENCE — additional confirming factors
  3. COUNTER-ARGUMENTS — what argues AGAINST the trade
  4. HISTORICAL CONTEXT — pattern memory statistics
  5. RISK ASSESSMENT — what could go wrong and probability of each
  6. TIMING CONTEXT — session phase, DTE, urgency
  7. VERDICT — synthesis of all above into final decision
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..state.market_state import MarketState
from ..state.dimensions import DIMENSIONS, DimensionCategory, CATEGORY_WEIGHTS


class EvidenceWeight(str, Enum):
    CRITICAL = "CRITICAL"  # Can single-handedly drive a decision
    HIGH = "HIGH"          # Strong influence
    MEDIUM = "MEDIUM"      # Moderate influence
    LOW = "LOW"            # Supporting/confirming only
    NEUTRAL = "NEUTRAL"    # Informational, no directional impact


class EvidenceDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    CONFLICTING = "CONFLICTING"


@dataclass
class Evidence:
    """A single piece of evidence in the reasoning chain."""
    factor: str                    # Dimension or composite name
    category: str                  # Which analysis category
    finding: str                   # What was observed (human-readable)
    direction: EvidenceDirection    # Which way does this point?
    weight: EvidenceWeight         # How important is this?
    confidence_impact: float       # +/- points on confidence score
    raw_value: float = 0.0        # The underlying numeric value
    velocity: float = 0.0         # How fast this is changing
    velocity_note: str = ""       # "PCR rising rapidly" etc.

    def to_dict(self) -> dict:
        return {
            "factor": self.factor,
            "category": self.category,
            "finding": self.finding,
            "direction": self.direction.value,
            "weight": self.weight.value,
            "impact": f"{self.confidence_impact:+.0f}",
            "velocity": self.velocity_note if self.velocity_note else None,
        }


@dataclass
class RiskScenario:
    """A specific risk that could torpedo the trade."""
    name: str
    description: str
    probability: float  # 0-1
    impact: str         # "STOP_LOSS", "PARTIAL_LOSS", "DELAYED_ENTRY", "WHIPSAW"
    mitigation: str     # What to do about it


@dataclass
class EvidenceChain:
    """
    The complete reasoning trace for a signal decision.
    This is the core output of the ReasoningEngine — consumed by AI models.
    """
    instrument: str
    timestamp: float = 0.0
    
    # Decision
    direction: str = "NO_TRADE"  # BUY, SELL, NO_TRADE
    confidence: float = 0.0
    confidence_breakdown: dict = field(default_factory=dict)
    
    # Evidence layers
    primary_evidence: list[Evidence] = field(default_factory=list)    # Top 3-5 drivers
    supporting_evidence: list[Evidence] = field(default_factory=list)  # Additional confirmation
    counter_arguments: list[Evidence] = field(default_factory=list)    # Against the trade
    
    # Historical context (from PatternMatcher)
    historical_win_rate: float = 0.0
    historical_setups: int = 0
    historical_modifier: float = 0.0
    historical_narrative: str = ""
    
    # Risk assessment
    risk_scenarios: list[RiskScenario] = field(default_factory=list)
    max_downside: str = ""
    risk_reward_ratio: float = 0.0
    
    # Timing
    urgency: str = "NORMAL"  # IMMEDIATE, HIGH, NORMAL, LOW, WAIT
    timing_note: str = ""
    time_to_expiry: str = ""
    
    # Vetoes (hard blocks)
    vetoes: list[str] = field(default_factory=list)
    veto_overrideable: bool = False  # Can GodMode override?
    
    # Final synthesis
    verdict: str = ""           # One-sentence final call
    reasoning_narrative: str = ""  # 3-5 sentence explanation
    actionable: bool = False
    
    # For AI model consumption
    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "confidence": round(self.confidence, 1),
            "actionable": self.actionable,
            "verdict": self.verdict,
            "reasoning": self.reasoning_narrative,
            "evidence": {
                "primary": [e.to_dict() for e in self.primary_evidence],
                "supporting": [e.to_dict() for e in self.supporting_evidence],
                "counter": [e.to_dict() for e in self.counter_arguments],
            },
            "historical": {
                "setups": self.historical_setups,
                "win_rate": self.historical_win_rate,
                "modifier": self.historical_modifier,
                "narrative": self.historical_narrative,
            },
            "risk": {
                "scenarios": [{"name": r.name, "probability": r.probability, "impact": r.impact, "mitigation": r.mitigation} for r in self.risk_scenarios],
                "max_downside": self.max_downside,
                "risk_reward": self.risk_reward_ratio,
            },
            "timing": {
                "urgency": self.urgency,
                "note": self.timing_note,
                "dte": self.time_to_expiry,
            },
            "vetoes": self.vetoes,
            "confidence_breakdown": self.confidence_breakdown,
        }
    
    def to_prompt(self) -> str:
        """
        Compact representation for injecting into AI model prompts.
        Designed to give the model everything it needs in minimal tokens.
        """
        lines = []
        lines.append(f"═══ SIGNAL BRAIN ANALYSIS: {self.instrument} ═══")
        lines.append(f"Direction: {self.direction} | Confidence: {self.confidence:.0f}% | {'ACTIONABLE' if self.actionable else 'NO TRADE'}")
        lines.append(f"Verdict: {self.verdict}")
        lines.append("")
        
        if self.primary_evidence:
            lines.append("PRIMARY EVIDENCE (strongest factors):")
            for e in self.primary_evidence:
                lines.append(f"  [{e.weight.value}] {e.factor}: {e.finding} ({e.confidence_impact:+.0f} pts)")
        
        if self.counter_arguments:
            lines.append("COUNTER-ARGUMENTS:")
            for e in self.counter_arguments:
                lines.append(f"  [{e.weight.value}] {e.factor}: {e.finding} ({e.confidence_impact:+.0f} pts)")
        
        if self.historical_setups > 0:
            lines.append(f"HISTORY: {self.historical_setups} similar setups, {self.historical_win_rate:.0f}% win rate ({self.historical_modifier:+.0f} conf)")
        
        if self.risk_scenarios:
            lines.append("RISKS:")
            for r in self.risk_scenarios[:3]:
                lines.append(f"  • {r.name} ({r.probability*100:.0f}% prob): {r.description}")
        
        if self.vetoes:
            lines.append(f"VETOES: {' | '.join(self.vetoes)}")
        
        lines.append(f"TIMING: {self.urgency} — {self.timing_note}")
        lines.append(f"R:R = {self.risk_reward_ratio:.1f}")
        lines.append("═══════════════════════════════════════════")
        
        return "\n".join(lines)


class EvidenceChainBuilder:
    """
    Builds the evidence chain from a MarketState.
    Analyzes each dimension category and extracts the reasoning.
    """
    
    def build(self, state: MarketState) -> list[Evidence]:
        """
        Extract all evidence from the current state.
        Returns a flat list; the engine will sort into primary/supporting/counter.
        """
        evidence = []
        
        evidence.extend(self._analyze_trend(state))
        evidence.extend(self._analyze_options(state))
        evidence.extend(self._analyze_momentum(state))
        evidence.extend(self._analyze_flow(state))
        evidence.extend(self._analyze_volatility(state))
        evidence.extend(self._analyze_price(state))
        evidence.extend(self._analyze_velocity(state))
        
        return evidence
    
    def _analyze_trend(self, state: MarketState) -> list[Evidence]:
        """Extract trend evidence."""
        ev = []
        dims = state.dimensions
        
        # EMA stack
        ema = dims.get("ema_stack_score")
        if ema and abs(ema.normalized) > 0.3:
            direction = EvidenceDirection.BULLISH if ema.normalized > 0 else EvidenceDirection.BEARISH
            strength = abs(ema.normalized)
            weight = EvidenceWeight.HIGH if strength > 0.7 else EvidenceWeight.MEDIUM
            ev.append(Evidence(
                factor="EMA_STACK",
                category="trend",
                finding=f"EMAs {'fully aligned bullish (9>21>50)' if ema.normalized > 0.7 else 'fully aligned bearish (9<21<50)' if ema.normalized < -0.7 else 'partially aligned ' + direction.value.lower()}",
                direction=direction,
                weight=weight,
                confidence_impact=strength * 12,
                raw_value=ema.normalized,
            ))
        
        # SuperTrend
        st = dims.get("supertrend")
        if st:
            direction = EvidenceDirection.BULLISH if st.normalized > 0 else EvidenceDirection.BEARISH
            ev.append(Evidence(
                factor="SUPERTREND",
                category="trend",
                finding=f"SuperTrend is {'bullish' if st.normalized > 0 else 'bearish'}",
                direction=direction,
                weight=EvidenceWeight.MEDIUM,
                confidence_impact=5 if abs(st.normalized) > 0 else 0,
                raw_value=st.normalized,
            ))
        
        # ADX regime
        adx = dims.get("adx_value")
        regime = dims.get("adx_regime")
        if adx:
            if adx.raw >= 30:
                ev.append(Evidence(
                    factor="ADX_STRENGTH",
                    category="trend",
                    finding=f"ADX {adx.raw:.0f} — strong trend in progress. Momentum trades have high probability.",
                    direction=EvidenceDirection.NEUTRAL,  # ADX doesn't tell direction, just strength
                    weight=EvidenceWeight.HIGH,
                    confidence_impact=8,
                    raw_value=adx.raw,
                ))
            elif adx.raw < 16:
                ev.append(Evidence(
                    factor="ADX_WEAKNESS",
                    category="trend",
                    finding=f"ADX {adx.raw:.0f} — CHOPPY market. Directional trades have low edge here.",
                    direction=EvidenceDirection.NEUTRAL,
                    weight=EvidenceWeight.HIGH,
                    confidence_impact=-10,
                    raw_value=adx.raw,
                ))
        
        # Higher timeframe
        htf = dims.get("htf_trend")
        if htf and abs(htf.normalized) > 0:
            direction = EvidenceDirection.BULLISH if htf.normalized > 0 else EvidenceDirection.BEARISH
            ev.append(Evidence(
                factor="HIGHER_TF",
                category="trend",
                finding=f"1-Hour timeframe is {'bullish' if htf.normalized > 0 else 'bearish'} — {'confirms' if True else 'contradicts'} lower TF",
                direction=direction,
                weight=EvidenceWeight.MEDIUM,
                confidence_impact=6 * htf.normalized,
                raw_value=htf.normalized,
            ))
        
        return ev
    
    def _analyze_options(self, state: MarketState) -> list[Evidence]:
        """Extract options microstructure evidence — THE key edge."""
        ev = []
        dims = state.dimensions
        
        # GEX Regime
        gex = dims.get("gex_regime")
        flip_dist = dims.get("gex_flip_distance")
        if gex:
            is_neg = gex.normalized < 0
            ev.append(Evidence(
                factor="GEX_REGIME",
                category="options",
                finding=f"{'Negative Gamma — dealers AMPLIFY moves (momentum trades favored)' if is_neg else 'Positive Gamma — dealers SUPPRESS moves (mean-reversion favored)'}",
                direction=EvidenceDirection.NEUTRAL,  # Regime affects strategy, not direction
                weight=EvidenceWeight.CRITICAL,
                confidence_impact=8 if is_neg else -5,
                raw_value=gex.normalized,
            ))
        
        # GEX flip distance
        if flip_dist:
            dist = flip_dist.raw  # In ATR units, signed
            if abs(dist) <= 1.5:
                ev.append(Evidence(
                    factor="GEX_FLIP_PROXIMITY",
                    category="options",
                    finding=f"Spot is only {abs(dist):.1f} ATR from GEX flip — regime transition zone. Explosive moves originate here.",
                    direction=EvidenceDirection.NEUTRAL,
                    weight=EvidenceWeight.CRITICAL,
                    confidence_impact=6,
                    raw_value=dist,
                    velocity=flip_dist.velocity,
                    velocity_note=f"Flip distance {'closing' if flip_dist.velocity < 0 else 'widening'}" if abs(flip_dist.velocity) > 0.05 else "",
                ))
        
        # PCR
        pcr = dims.get("pcr")
        pcr_vel = dims.get("pcr_velocity")
        if pcr:
            if pcr.raw > 1.2:
                direction = EvidenceDirection.BULLISH  # High PCR = heavy put writing = support
                finding = f"PCR {pcr.raw:.2f} — heavy put writing signals institutional support below"
            elif pcr.raw < 0.7:
                direction = EvidenceDirection.BEARISH
                finding = f"PCR {pcr.raw:.2f} — heavy call activity, no put support"
            else:
                direction = EvidenceDirection.NEUTRAL
                finding = f"PCR {pcr.raw:.2f} — neutral range"
            
            weight = EvidenceWeight.HIGH if abs(pcr.normalized) > 0.5 else EvidenceWeight.MEDIUM
            ev.append(Evidence(
                factor="PCR",
                category="options",
                finding=finding,
                direction=direction,
                weight=weight,
                confidence_impact=pcr.normalized * 8,
                raw_value=pcr.raw,
                velocity=pcr.velocity if pcr_vel else 0,
                velocity_note=f"PCR {'rising rapidly (support building)' if pcr.velocity > 0.1 else 'falling rapidly (support withdrawing)' if pcr.velocity < -0.1 else ''}" if hasattr(pcr, 'velocity') and abs(pcr.velocity) > 0.05 else "",
            ))
        
        # IV percentile
        iv_pctl = dims.get("iv_percentile")
        if iv_pctl and iv_pctl.raw > 0:
            if iv_pctl.raw > 80:
                ev.append(Evidence(
                    factor="IV_EXTREME",
                    category="options",
                    finding=f"IV at {iv_pctl.raw:.0f}th percentile — options are EXPENSIVE. Naked longs face IV crush risk.",
                    direction=EvidenceDirection.NEUTRAL,
                    weight=EvidenceWeight.HIGH,
                    confidence_impact=-5,
                    raw_value=iv_pctl.raw,
                ))
            elif iv_pctl.raw < 20:
                ev.append(Evidence(
                    factor="IV_LOW",
                    category="options",
                    finding=f"IV at {iv_pctl.raw:.0f}th percentile — options are CHEAP. Good time for directional longs.",
                    direction=EvidenceDirection.NEUTRAL,
                    weight=EvidenceWeight.MEDIUM,
                    confidence_impact=3,
                    raw_value=iv_pctl.raw,
                ))
        
        return ev
    
    def _analyze_momentum(self, state: MarketState) -> list[Evidence]:
        """Extract momentum evidence."""
        ev = []
        dims = state.dimensions
        
        # RSI
        rsi = dims.get("rsi")
        rsi_div = dims.get("rsi_divergence")
        if rsi:
            if rsi.raw > 70:
                ev.append(Evidence(
                    factor="RSI_OVERBOUGHT", category="momentum",
                    finding=f"RSI {rsi.raw:.0f} — overbought territory. Upside exhaustion risk.",
                    direction=EvidenceDirection.BEARISH, weight=EvidenceWeight.MEDIUM,
                    confidence_impact=-4, raw_value=rsi.raw,
                ))
            elif rsi.raw < 30:
                ev.append(Evidence(
                    factor="RSI_OVERSOLD", category="momentum",
                    finding=f"RSI {rsi.raw:.0f} — oversold. Bounce risk for shorts.",
                    direction=EvidenceDirection.BULLISH, weight=EvidenceWeight.MEDIUM,
                    confidence_impact=-4, raw_value=rsi.raw,
                ))
            elif rsi.raw > 55:
                ev.append(Evidence(
                    factor="RSI_BULLISH", category="momentum",
                    finding=f"RSI {rsi.raw:.0f} — bullish momentum zone",
                    direction=EvidenceDirection.BULLISH, weight=EvidenceWeight.LOW,
                    confidence_impact=3, raw_value=rsi.raw,
                ))
            elif rsi.raw < 45:
                ev.append(Evidence(
                    factor="RSI_BEARISH", category="momentum",
                    finding=f"RSI {rsi.raw:.0f} — bearish momentum zone",
                    direction=EvidenceDirection.BEARISH, weight=EvidenceWeight.LOW,
                    confidence_impact=3, raw_value=rsi.raw,
                ))
        
        # RSI Divergence
        if rsi_div and rsi_div.normalized != 0:
            is_bullish_div = rsi_div.normalized > 0
            ev.append(Evidence(
                factor="RSI_DIVERGENCE", category="momentum",
                finding=f"{'Bullish' if is_bullish_div else 'Bearish'} RSI divergence — momentum {'strengthening under surface' if is_bullish_div else 'weakening despite price'}",
                direction=EvidenceDirection.BULLISH if is_bullish_div else EvidenceDirection.BEARISH,
                weight=EvidenceWeight.HIGH,
                confidence_impact=7 * rsi_div.normalized,
                raw_value=rsi_div.normalized,
            ))
        
        # MACD
        macd = dims.get("macd_histogram")
        if macd and abs(macd.normalized) > 0.2:
            direction = EvidenceDirection.BULLISH if macd.normalized > 0 else EvidenceDirection.BEARISH
            ev.append(Evidence(
                factor="MACD", category="momentum",
                finding=f"MACD histogram {'positive and expanding' if macd.normalized > 0.3 else 'positive' if macd.normalized > 0 else 'negative and expanding' if macd.normalized < -0.3 else 'negative'}",
                direction=direction,
                weight=EvidenceWeight.MEDIUM,
                confidence_impact=macd.normalized * 5,
                raw_value=macd.normalized,
            ))
        
        return ev
    
    def _analyze_flow(self, state: MarketState) -> list[Evidence]:
        """Extract volume and institutional flow evidence."""
        ev = []
        dims = state.dimensions
        
        # Volume
        vol = dims.get("volume_ratio")
        if vol:
            if vol.raw > 2.0:
                ev.append(Evidence(
                    factor="VOLUME_SPIKE", category="flow",
                    finding=f"Volume {vol.raw:.1f}x average — institutional participation confirmed",
                    direction=EvidenceDirection.NEUTRAL,  # Direction from price, not volume alone
                    weight=EvidenceWeight.HIGH,
                    confidence_impact=6, raw_value=vol.raw,
                ))
            elif vol.raw < 0.6:
                ev.append(Evidence(
                    factor="LOW_VOLUME", category="flow",
                    finding=f"Volume only {vol.raw:.1f}x average — low conviction move, don't trust it",
                    direction=EvidenceDirection.NEUTRAL,
                    weight=EvidenceWeight.MEDIUM,
                    confidence_impact=-5, raw_value=vol.raw,
                ))
        
        # VWAP
        vwap = dims.get("vwap_position")
        if vwap and abs(vwap.normalized) > 0:
            direction = EvidenceDirection.BULLISH if vwap.normalized > 0 else EvidenceDirection.BEARISH
            ev.append(Evidence(
                factor="VWAP", category="flow",
                finding=f"Price {'above' if vwap.normalized > 0 else 'below'} VWAP — institutional {'buying' if vwap.normalized > 0 else 'selling'} pressure",
                direction=direction,
                weight=EvidenceWeight.MEDIUM,
                confidence_impact=4 * vwap.normalized, raw_value=vwap.normalized,
            ))
        
        # FII flow
        fii = dims.get("fii_flow")
        if fii and abs(fii.normalized) > 0.2:
            direction = EvidenceDirection.BULLISH if fii.normalized > 0 else EvidenceDirection.BEARISH
            ev.append(Evidence(
                factor="FII_FLOW", category="flow",
                finding=f"FII {'net buyers (₹{fii.raw:.0f} Cr)' if fii.raw > 0 else 'net sellers (₹{abs(fii.raw):.0f} Cr)'} — institutional {'accumulation' if fii.raw > 0 else 'distribution'}",
                direction=direction,
                weight=EvidenceWeight.MEDIUM,
                confidence_impact=fii.normalized * 5, raw_value=fii.raw,
            ))
        
        return ev
    
    def _analyze_volatility(self, state: MarketState) -> list[Evidence]:
        """Extract volatility evidence."""
        ev = []
        dims = state.dimensions
        
        # VIX
        vix = dims.get("vix")
        if vix:
            if vix.raw > 22:
                ev.append(Evidence(
                    factor="VIX_HIGH", category="volatility",
                    finding=f"VIX {vix.raw:.1f} — elevated fear. Options are expensive but moves will be large.",
                    direction=EvidenceDirection.NEUTRAL,
                    weight=EvidenceWeight.MEDIUM,
                    confidence_impact=0, raw_value=vix.raw,
                ))
        
        # BB Squeeze
        bb = dims.get("bb_width")
        if bb and bb.raw < 1.5:
            ev.append(Evidence(
                factor="VOL_SQUEEZE", category="volatility",
                finding=f"Bollinger Band squeeze detected (width {bb.raw:.2f}%) — explosive breakout imminent",
                direction=EvidenceDirection.NEUTRAL,
                weight=EvidenceWeight.HIGH,
                confidence_impact=5, raw_value=bb.raw,
            ))
        
        # Dead market
        atr_pct = dims.get("atr_pct")
        if atr_pct and atr_pct.raw < 0.25:
            ev.append(Evidence(
                factor="DEAD_MARKET", category="volatility",
                finding=f"ATR only {atr_pct.raw:.2f}% of price — market is dead. Theta eats premiums faster than spot moves.",
                direction=EvidenceDirection.NEUTRAL,
                weight=EvidenceWeight.CRITICAL,
                confidence_impact=-15, raw_value=atr_pct.raw,
            ))
        
        return ev
    
    def _analyze_price(self, state: MarketState) -> list[Evidence]:
        """Extract price structure evidence."""
        ev = []
        dims = state.dimensions
        
        # Opening range
        orb = dims.get("orb_status")
        if orb and orb.raw != 0:
            direction = EvidenceDirection.BULLISH if orb.raw > 0 else EvidenceDirection.BEARISH
            ev.append(Evidence(
                factor="ORB", category="price",
                finding=f"Opening Range {'Breakout (bullish)' if orb.raw > 0 else 'Breakdown (bearish)'}",
                direction=direction,
                weight=EvidenceWeight.MEDIUM,
                confidence_impact=4 * orb.raw, raw_value=orb.raw,
            ))
        
        # Day range position
        drp = dims.get("day_range_position")
        if drp:
            if drp.raw > 0.9:
                ev.append(Evidence(
                    factor="DAY_HIGH", category="price",
                    finding="Price at day's high (90%+ of range) — either breakout strength or exhaustion",
                    direction=EvidenceDirection.BULLISH,
                    weight=EvidenceWeight.LOW,
                    confidence_impact=2, raw_value=drp.raw,
                ))
            elif drp.raw < 0.1:
                ev.append(Evidence(
                    factor="DAY_LOW", category="price",
                    finding="Price at day's low (bottom 10%) — either breakdown or bounce zone",
                    direction=EvidenceDirection.BEARISH,
                    weight=EvidenceWeight.LOW,
                    confidence_impact=2, raw_value=drp.raw,
                ))
        
        return ev
    
    def _analyze_velocity(self, state: MarketState) -> list[Evidence]:
        """Extract velocity-based evidence (things CHANGING fast)."""
        ev = []
        dims = state.dimensions
        
        # Find dimensions with high velocity
        for name, dv in dims.items():
            if abs(dv.velocity) < 0.15:
                continue
            dim_def = DIMENSIONS.get(name)
            if not dim_def or dim_def.weight < 6:
                continue
            
            # High velocity on an important dimension = something is happening NOW
            direction = EvidenceDirection.BULLISH if dv.velocity > 0 else EvidenceDirection.BEARISH
            ev.append(Evidence(
                factor=f"VELOCITY_{name.upper()}",
                category="velocity",
                finding=f"{name} is changing rapidly (velocity: {dv.velocity:.3f}/scan). Something is shifting NOW.",
                direction=direction,
                weight=EvidenceWeight.HIGH,
                confidence_impact=abs(dv.velocity) * 10,
                raw_value=dv.velocity,
                velocity=dv.velocity,
                velocity_note=f"{'Accelerating' if dv.acceleration > 0.05 else 'Decelerating' if dv.acceleration < -0.05 else 'Steady rate'}",
            ))
        
        return ev
