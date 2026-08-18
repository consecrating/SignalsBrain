"""
SignalsBrain — Blunder Guard

The 14+ hard vetoes that OVERRIDE the signal engine.
These are learned from real losses — each one represents a specific scenario
where the engine was RIGHT about direction but the trade still lost money.

A veto doesn't just say "no" — it explains WHY so the AI model can factor it in.

Hierarchy:
  HARD VETO = trade is blocked entirely, no override possible
  SOFT VETO = confidence penalty + warning, GodMode can override with justification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..state.market_state import MarketState


@dataclass
class Veto:
    """A triggered veto with full explanation."""
    name: str
    rule_number: int
    severity: str  # "HARD" or "SOFT"
    description: str
    learned_from: str  # "Consecutive losses when..." (why this rule exists)
    overrideable: bool = False
    confidence_penalty: float = 0  # For soft vetoes


class BlunderGuard:
    """
    Evaluates all 14 blunder prevention rules against a MarketState + signal.
    Returns a list of triggered vetoes (could be empty = all clear).
    """
    
    def evaluate(self, state: MarketState, direction: str, confidence: float,
                 session_signals: int = 0, session_stops: int = 0,
                 premium_already_moved_pct: float = 0,
                 live_premium_cost: float = 0, capital: float = 200000) -> list[Veto]:
        """
        Run all blunder prevention rules.
        Returns list of triggered vetoes (empty = no blocks).
        """
        vetoes = []
        dims = state.dimensions
        
        if direction == "NO_TRADE":
            return vetoes
        
        # Helper to get raw values
        def raw(name: str, default=0.0) -> float:
            d = dims.get(name)
            return d.raw if d else default
        
        def norm(name: str, default=0.0) -> float:
            d = dims.get(name)
            return d.normalized if d else default
        
        gex_regime = state.gex_regime
        session_min = raw("session_minutes", 120)
        adx = raw("adx_value", 20)
        atr_pct = raw("atr_pct", 0.5)
        rsi = raw("rsi", 50)
        pcr = raw("pcr", 1.0)
        gex_flip_dist = raw("gex_flip_distance", 5)
        dte = raw("dte", 5)
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 1: GEX REGIME VETO (Positive Gamma, far from flip)
        # Learned from: Buying PEs when dealers are actively suppressing downside
        # ══════════════════════════════════════════════════════════════════════
        if gex_regime == "Positive" and abs(gex_flip_dist) > 2.5:
            if (direction == "SELL" and gex_flip_dist > 0) or (direction == "BUY" and gex_flip_dist < 0):
                vetoes.append(Veto(
                    name="GEX_REGIME_BLOCK",
                    rule_number=1,
                    severity="HARD",
                    description=f"Positive Gamma regime, spot {abs(gex_flip_dist):.1f} ATR from flip. Dealers are actively {'suppressing downside' if direction == 'SELL' else 'capping upside'}.",
                    learned_from="Multiple losses when fighting dealer gamma hedging far from the flip point.",
                ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 2: LATE SESSION (extended to 3:15 PM per user request)
        # Learned from: Theta crush + illiquidity in last 15 minutes
        # ══════════════════════════════════════════════════════════════════════
        if session_min >= 360:  # 3:15 PM = 09:15 + 360 min
            vetoes.append(Veto(
                name="LATE_SESSION",
                rule_number=2,
                severity="HARD",
                description=f"Session minute {session_min:.0f} (after 3:15 PM IST). Theta crush + position squaring + illiquidity make new entries high-risk.",
                learned_from="Consistent premium decay eating profits in last 15 minutes.",
            ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 3: WALL PROXIMITY (Positive Gamma only)
        # Learned from: Dealer defense at walls causing sharp reversals
        # ══════════════════════════════════════════════════════════════════════
        call_wall_dist = raw("call_wall_distance", 10)
        put_wall_dist = raw("put_wall_distance", 10)
        
        if gex_regime == "Positive":
            if direction == "BUY" and call_wall_dist < 0.5:
                vetoes.append(Veto(
                    name="CALL_WALL_BLOCK",
                    rule_number=3,
                    severity="SOFT",
                    description=f"Spot within {call_wall_dist:.1f} ATR of call wall. In Positive Gamma, dealers defend this level.",
                    learned_from="Bought calls near call wall → immediate reversal as dealers sold into the move.",
                    overrideable=True,
                    confidence_penalty=12,
                ))
            if direction == "SELL" and put_wall_dist < 0.5:
                vetoes.append(Veto(
                    name="PUT_WALL_BLOCK",
                    rule_number=3,
                    severity="SOFT",
                    description=f"Spot within {put_wall_dist:.1f} ATR of put wall. In Positive Gamma, dealers provide support here.",
                    learned_from="Bought puts near put wall → bounce as dealers bought to hedge.",
                    overrideable=True,
                    confidence_penalty=12,
                ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 4: EXPIRY DAY WEAK SIGNAL
        # Learned from: Theta decay eats marginal signals on expiry day
        # ══════════════════════════════════════════════════════════════════════
        if dte <= 1.2 and confidence < 80:
            vetoes.append(Veto(
                name="EXPIRY_DAY_WEAK",
                rule_number=4,
                severity="HARD",
                description=f"Expiry day (DTE {dte:.1f}) requires 80%+ confidence (got {confidence:.0f}%). Theta melts premium too fast for marginal setups.",
                learned_from="Multiple losses on expiry day where direction was right but theta ate the premium before target hit.",
            ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 5: OPENING CHAOS (first 15 minutes)
        # Learned from: Gap fills and fake moves in first 15 min
        # ══════════════════════════════════════════════════════════════════════
        if session_min <= 15:
            vetoes.append(Veto(
                name="OPENING_CHAOS",
                rule_number=5,
                severity="HARD",
                description="First 15 minutes after open. Gap-fill chaos — wait for opening range to form.",
                learned_from="Entered immediately at open → got whipsawed by gap-fill before real trend emerged.",
            ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 6: MOMENTUM DEATH (consecutive candles against you)
        # Learned from: Chasing a move that's already exhausted
        # ══════════════════════════════════════════════════════════════════════
        roc = raw("roc_5", 0)
        if direction == "BUY" and roc < -0.8:
            vetoes.append(Veto(
                name="MOMENTUM_DEATH_BUY",
                rule_number=6,
                severity="SOFT",
                description=f"Strong downward momentum (RoC {roc:.2f}%). Buying into a waterfall is catching a falling knife.",
                learned_from="Bought the dip too early → momentum continued for another 2-3 candles before reversal.",
                overrideable=True,
                confidence_penalty=8,
            ))
        if direction == "SELL" and roc > 0.8:
            vetoes.append(Veto(
                name="MOMENTUM_DEATH_SELL",
                rule_number=6,
                severity="SOFT",
                description=f"Strong upward momentum (RoC +{roc:.2f}%). Shorting a rocket is high-risk.",
                learned_from="Shorted during strong rally → momentum carried another 2+ ATR before exhaustion.",
                overrideable=True,
                confidence_penalty=8,
            ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 7: PREMIUM ALREADY MOVED
        # Learned from: Entering after the move already happened (buying at the top)
        # ══════════════════════════════════════════════════════════════════════
        if premium_already_moved_pct > 20:
            vetoes.append(Veto(
                name="PREMIUM_MOVED",
                rule_number=7,
                severity="SOFT",
                description=f"Option premium already moved {premium_already_moved_pct:.0f}% from open. You're late — chasing increases risk.",
                learned_from="Entered after 20%+ premium move → mean reversion hit before next target.",
                overrideable=True,
                confidence_penalty=10,
            ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 8: HIGHER-TF CONTRADICTION + WEAK ADX
        # Learned from: Fighting the bigger trend in a trendless market
        # ══════════════════════════════════════════════════════════════════════
        htf = dims.get("htf_trend")
        if htf and adx < 20:
            htf_agrees = (htf.normalized > 0 and direction == "BUY") or (htf.normalized < 0 and direction == "SELL")
            if not htf_agrees and htf.normalized != 0:
                vetoes.append(Veto(
                    name="MTF_CONTRADICTION",
                    rule_number=8,
                    severity="SOFT",
                    description=f"Higher TF is {'bullish' if htf.normalized > 0 else 'bearish'} but signal is {direction}, AND ADX is weak ({adx:.0f}). No clear trend on any timeframe.",
                    learned_from="Counter-trend trades in weak-ADX environments had <35% win rate.",
                    overrideable=True,
                    confidence_penalty=10,
                ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 9: MAX DAILY SIGNALS
        # Learned from: Overtrading = death by a thousand cuts
        # ══════════════════════════════════════════════════════════════════════
        if session_signals >= 3:
            vetoes.append(Veto(
                name="MAX_DAILY_SIGNALS",
                rule_number=9,
                severity="HARD",
                description=f"Already generated {session_signals} signals today. Overtrading increases losses on choppy days.",
                learned_from="Days with 4+ signals consistently had negative total P&L due to forced trades.",
            ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 10: CIRCUIT BREAKER (consecutive stops)
        # Learned from: Tilt-driven revenge trading after losses
        # ══════════════════════════════════════════════════════════════════════
        if session_stops >= 2:
            vetoes.append(Veto(
                name="CIRCUIT_BREAKER",
                rule_number=10,
                severity="HARD",
                description=f"{session_stops} stops hit today. Going flat to prevent tilt-driven losses.",
                learned_from="After 2 consecutive stops, the next 3 trades had 80% loss rate (emotional, not analytical).",
            ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 11: DEAD MARKET
        # Learned from: Buying options in a flat market = pure theta loss
        # ══════════════════════════════════════════════════════════════════════
        if atr_pct < 0.25:
            vetoes.append(Veto(
                name="DEAD_MARKET",
                rule_number=11,
                severity="HARD",
                description=f"ATR only {atr_pct:.2f}% of price. Market too quiet — premiums decay faster than spot moves.",
                learned_from="Every trade taken when ATR < 0.25% lost money to theta before target was reached.",
            ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 12: CROWDED TRADE (extreme PCR)
        # Learned from: Everyone on the same side = contrarian risk
        # ══════════════════════════════════════════════════════════════════════
        if direction == "SELL" and pcr > 1.8:
            vetoes.append(Veto(
                name="PCR_CROWDED_SHORT",
                rule_number=12,
                severity="SOFT",
                description=f"PCR {pcr:.2f} — everyone already bought puts. Crowded trade, short-squeeze risk.",
                learned_from="When PCR > 1.8 and you buy more puts, you're the last one in. Smart money starts covering.",
                overrideable=True,
                confidence_penalty=8,
            ))
        if direction == "BUY" and pcr < 0.4:
            vetoes.append(Veto(
                name="PCR_CROWDED_LONG",
                rule_number=12,
                severity="SOFT",
                description=f"PCR {pcr:.2f} — everyone already bought calls. Crowded, reversal risk.",
                learned_from="Extreme low PCR precedes sharp corrections as call writers capitulate.",
                overrideable=True,
                confidence_penalty=8,
            ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 13: RSI EXTREME (entering at exhaustion)
        # Learned from: Buying overbought, selling oversold
        # ══════════════════════════════════════════════════════════════════════
        if direction == "BUY" and rsi > 80:
            vetoes.append(Veto(
                name="RSI_OVERBOUGHT",
                rule_number=13,
                severity="HARD",
                description=f"RSI {rsi:.0f} — severely overbought. Buying here has negative expectancy.",
                learned_from="Buying at RSI > 80: win rate drops to 28%. Mean reversion dominates.",
            ))
        if direction == "SELL" and rsi < 20:
            vetoes.append(Veto(
                name="RSI_OVERSOLD",
                rule_number=13,
                severity="HARD",
                description=f"RSI {rsi:.0f} — severely oversold. Shorting here has negative expectancy.",
                learned_from="Selling at RSI < 20: win rate drops to 30%. Bounce probability too high.",
            ))
        
        # ══════════════════════════════════════════════════════════════════════
        # RULE 14: CAPITAL PROTECTION
        # Learned from: Blowing up on a single trade
        # ══════════════════════════════════════════════════════════════════════
        if live_premium_cost > 0 and live_premium_cost > capital * 0.5:
            vetoes.append(Veto(
                name="CAPITAL_EXCEEDED",
                rule_number=14,
                severity="HARD",
                description=f"Premium cost ₹{live_premium_cost:.0f}/lot exceeds 50% of ₹{capital:.0f} capital. Cannot afford proper position sizing.",
                learned_from="Single trades using >50% of capital: one stop-loss = 25%+ capital destroyed.",
            ))
        
        return vetoes
    
    def get_hard_vetoes(self, vetoes: list[Veto]) -> list[Veto]:
        """Filter to only hard (non-overrideable) vetoes."""
        return [v for v in vetoes if v.severity == "HARD"]
    
    def get_soft_vetoes(self, vetoes: list[Veto]) -> list[Veto]:
        """Filter to soft (overrideable with justification) vetoes."""
        return [v for v in vetoes if v.severity == "SOFT"]
    
    def total_confidence_penalty(self, vetoes: list[Veto]) -> float:
        """Sum of all soft veto confidence penalties."""
        return sum(v.confidence_penalty for v in vetoes if v.severity == "SOFT")
