"""
SignalsBrain — Reasoning Engine (Main Orchestrator for Layer 3)

This is the BRAIN. It takes:
  - A MarketState (47 dimensions, from Layer 1)
  - Historical context (from Layer 2, PatternMemory)

And produces:
  - An EvidenceChain (the full reasoning trace)
  - A decision (BUY / SELL / NO_TRADE)
  - A confidence score (with breakdown)
  - Risk scenarios
  - Actionable trade plan parameters

No black boxes. Every decision is traceable, auditable, and explainable.
When connected to an AI model, the model receives the FULL reasoning and can:
  - Agree and execute
  - Challenge specific evidence and request re-evaluation
  - Add context the engine doesn't have (macro events, news, etc.)
  - Override with justification (logged for learning)
"""

from __future__ import annotations

import time
from typing import Optional

from ..state.market_state import MarketState
from ..memory.matcher import PatternMatcher, HistoricalContext
from ..memory.pattern_db import PatternDB
from .evidence_chain import (
    EvidenceChain, EvidenceChainBuilder, Evidence,
    EvidenceWeight, EvidenceDirection, RiskScenario,
)
from .confidence_calc import ConfidenceCalculator, ConfidenceBreakdown
from .blunder_guard import BlunderGuard, Veto


class ReasoningEngine:
    """
    The core decision-making engine.
    
    Usage:
        engine = ReasoningEngine(pattern_db)
        chain = engine.reason(market_state)
        # chain.direction = "SELL"
        # chain.confidence = 78.3
        # chain.to_prompt() = full reasoning for AI model
    """
    
    def __init__(self, pattern_db: Optional[PatternDB] = None):
        self.evidence_builder = EvidenceChainBuilder()
        self.confidence_calc = ConfidenceCalculator()
        self.blunder_guard = BlunderGuard()
        self.pattern_matcher = PatternMatcher(pattern_db) if pattern_db else None
        self.pattern_db = pattern_db
    
    def reason(
        self,
        state: MarketState,
        session_signals: int = 0,
        session_stops: int = 0,
        premium_moved_pct: float = 0,
        live_premium_cost: float = 0,
        capital: float = 200000,
        confidence_threshold: float = 60,
    ) -> EvidenceChain:
        """
        Produce a complete reasoning chain for the current market state.
        
        This is the main entry point. Returns everything: decision, confidence,
        evidence, risks, timing, vetoes — ready for AI model consumption.
        """
        chain = EvidenceChain(
            instrument=state.instrument,
            timestamp=time.time(),
        )
        
        # ── Step 1: Collect all evidence ──────────────────────────────────────
        all_evidence = self.evidence_builder.build(state)
        
        # ── Step 2: Determine direction from net bias ─────────────────────────
        net = state.net_directional_bias
        direction = "BUY" if net > 5 else "SELL" if net < -5 else "NO_TRADE"
        
        # ── Step 3: Get historical context ────────────────────────────────────
        hist_ctx = HistoricalContext()
        if self.pattern_matcher and direction != "NO_TRADE":
            hist_ctx = self.pattern_matcher.get_context(state, direction)
        
        # ── Step 4: Calculate confidence (multi-stage) ────────────────────────
        gex_flip_dist = 999.0
        flip_dim = state.dimensions.get("gex_flip_distance")
        if flip_dim:
            gex_flip_dist = abs(flip_dim.raw)
        
        htf_dim = state.dimensions.get("htf_trend")
        htf_aligned = None
        if htf_dim and htf_dim.normalized != 0 and direction != "NO_TRADE":
            htf_aligned = (htf_dim.normalized > 0 and direction == "BUY") or (htf_dim.normalized < 0 and direction == "SELL")
        
        conf_breakdown = self.confidence_calc.calculate(
            net_bias=net,
            agreement=state.agreement_factor,
            evidence=all_evidence,
            regime=state.regime,
            gex_regime=state.gex_regime,
            gex_flip_distance_atr=gex_flip_dist,
            historical_modifier=hist_ctx.confidence_modifier,
            historical_explanation=hist_ctx.confidence_reason,
            htf_aligned=htf_aligned,
        )
        confidence = conf_breakdown.final
        
        # ── Step 5: Run blunder guard ─────────────────────────────────────────
        vetoes = self.blunder_guard.evaluate(
            state=state,
            direction=direction,
            confidence=confidence,
            session_signals=session_signals,
            session_stops=session_stops,
            premium_already_moved_pct=premium_moved_pct,
            live_premium_cost=live_premium_cost,
            capital=capital,
        )
        
        hard_vetoes = self.blunder_guard.get_hard_vetoes(vetoes)
        soft_vetoes = self.blunder_guard.get_soft_vetoes(vetoes)
        
        # Apply soft veto penalties to confidence
        soft_penalty = self.blunder_guard.total_confidence_penalty(vetoes)
        confidence = max(0, confidence - soft_penalty)
        
        # Hard vetoes kill the signal entirely
        if hard_vetoes:
            direction = "NO_TRADE"
        
        # Confidence gate (threshold check)
        if direction != "NO_TRADE" and confidence < confidence_threshold:
            direction = "NO_TRADE"
        
        # ── Step 6: Sort evidence into categories ─────────────────────────────
        # Determine which evidence supports and which contradicts the final direction
        signal_direction = 1 if net > 0 else -1
        
        supporting = []
        contradicting = []
        
        for e in all_evidence:
            if e.confidence_impact == 0:
                continue
            
            # Does this evidence agree with the signal direction?
            agrees = (
                (e.direction == EvidenceDirection.BULLISH and signal_direction > 0) or
                (e.direction == EvidenceDirection.BEARISH and signal_direction < 0) or
                (e.direction == EvidenceDirection.NEUTRAL and e.confidence_impact > 0)
            )
            
            if agrees:
                supporting.append(e)
            elif e.direction != EvidenceDirection.NEUTRAL:
                contradicting.append(e)
        
        # Sort by absolute impact
        supporting.sort(key=lambda e: abs(e.confidence_impact), reverse=True)
        contradicting.sort(key=lambda e: abs(e.confidence_impact), reverse=True)
        
        chain.primary_evidence = supporting[:5]
        chain.supporting_evidence = supporting[5:]
        chain.counter_arguments = contradicting
        
        # ── Step 7: Build risk scenarios ──────────────────────────────────────
        chain.risk_scenarios = self._build_risk_scenarios(state, direction, confidence, hist_ctx)
        
        # ── Step 8: Timing assessment ─────────────────────────────────────────
        chain.urgency, chain.timing_note = self._assess_timing(state, direction)
        dte_dim = state.dimensions.get("dte")
        chain.time_to_expiry = f"{dte_dim.raw:.1f} days" if dte_dim else "Unknown"
        
        # ── Step 9: Calculate risk/reward ─────────────────────────────────────
        atr_dim = state.dimensions.get("atr_pct")
        if atr_dim and atr_dim.raw > 0:
            # Target: 1 ATR move, Stop: 1.2 ATR → base R:R = 1/1.2 = 0.83
            # Adjusted by T2/T3 probability from history
            if hist_ctx.hit_t2_rate > 40:
                chain.risk_reward_ratio = 2.0
            elif hist_ctx.hit_t1_rate > 60:
                chain.risk_reward_ratio = 1.5
            else:
                chain.risk_reward_ratio = 0.83
            chain.max_downside = "1.2× ATR (stop loss) + slippage"
        
        # ── Step 10: Fill remaining chain fields ──────────────────────────────
        chain.direction = direction
        chain.confidence = confidence
        chain.confidence_breakdown = conf_breakdown.to_dict()
        chain.actionable = (direction != "NO_TRADE")
        
        chain.historical_win_rate = hist_ctx.win_rate
        chain.historical_setups = hist_ctx.similar_setups
        chain.historical_modifier = hist_ctx.confidence_modifier
        chain.historical_narrative = hist_ctx.narrative
        
        chain.vetoes = [f"[{v.severity}] Rule {v.rule_number} {v.name}: {v.description}" for v in vetoes]
        chain.veto_overrideable = len(hard_vetoes) == 0 and len(soft_vetoes) > 0
        
        # Build final verdict and narrative
        chain.verdict = self._build_verdict(chain, state)
        chain.reasoning_narrative = self._build_narrative(chain, state, hist_ctx)
        
        return chain
    
    def _build_risk_scenarios(self, state: MarketState, direction: str,
                              confidence: float, hist: HistoricalContext) -> list[RiskScenario]:
        """Build specific risk scenarios for this trade."""
        risks = []
        
        # Risk 1: Stop loss hit
        sl_prob = (hist.stop_loss_rate / 100) if hist.stop_loss_rate > 0 else 0.3
        risks.append(RiskScenario(
            name="STOP_LOSS",
            description="Price reverses and hits stop loss (1.2 ATR adverse move)",
            probability=min(0.5, sl_prob),
            impact="STOP_LOSS",
            mitigation="Position size so max loss is 2% of capital. Accept it and move on.",
        ))
        
        # Risk 2: Theta decay (DTE-dependent)
        dte_dim = state.dimensions.get("dte")
        if dte_dim and dte_dim.raw <= 2:
            risks.append(RiskScenario(
                name="THETA_DECAY",
                description="Short DTE means aggressive time decay even if direction is correct",
                probability=0.4,
                impact="PARTIAL_LOSS",
                mitigation="Use ITM options for higher delta, or exit before 3 PM. Set time-based stop.",
            ))
        
        # Risk 3: IV Crush
        iv_dim = state.dimensions.get("iv_percentile")
        if iv_dim and iv_dim.raw > 70:
            risks.append(RiskScenario(
                name="IV_CRUSH",
                description="IV is elevated — if it normalizes, premium drops even with favorable price movement",
                probability=0.3,
                impact="PARTIAL_LOSS",
                mitigation="Use debit spread to cap vega risk, or take quick profits at T1.",
            ))
        
        # Risk 4: Whipsaw (choppy market)
        if state.regime == "Choppy":
            risks.append(RiskScenario(
                name="WHIPSAW",
                description="Choppy/range-bound market — likely false breakout followed by reversal",
                probability=0.45,
                impact="STOP_LOSS",
                mitigation="Tighter stops, smaller size, or avoid entirely (best action in chop is no action).",
            ))
        
        # Risk 5: Gap risk (overnight for positional)
        if dte_dim and dte_dim.raw > 1:
            risks.append(RiskScenario(
                name="GAP_RISK",
                description="Overnight gap against position due to global events or news",
                probability=0.1,
                impact="STOP_LOSS",
                mitigation="Intraday only. If holding overnight, use OTM hedge (spread).",
            ))
        
        return risks
    
    def _assess_timing(self, state: MarketState, direction: str) -> tuple[str, str]:
        """Assess urgency and timing context."""
        session = state.dimensions.get("session_minutes")
        if not session:
            return "NORMAL", "No timing data available"
        
        mins = session.raw
        
        if mins <= 15:
            return "WAIT", "Opening 15 minutes — let the noise settle. Patience."
        elif mins <= 30:
            return "HIGH", "Opening range forming. If signal is strong, enter after the range is set."
        elif mins <= 240:
            return "NORMAL", f"Main session ({mins:.0f} min in). Standard execution window."
        elif mins <= 330:
            return "LOW", f"Afternoon session ({mins:.0f} min). Less time for targets — quick trades only."
        else:
            return "WAIT", "Late session. High theta, low liquidity. Better to wait for tomorrow."
    
    def _build_verdict(self, chain: EvidenceChain, state: MarketState) -> str:
        """One-sentence final verdict."""
        if not chain.actionable:
            if chain.vetoes:
                veto_names = [v.split(":")[0].strip("[] ").split(" ")[-1] for v in chain.vetoes[:2]]
                return f"NO TRADE on {state.instrument} — blocked by {', '.join(veto_names)}. Confidence was {chain.confidence:.0f}%."
            return f"NO TRADE on {state.instrument} — insufficient conviction ({chain.confidence:.0f}%). Wait for cleaner setup."
        
        return (
            f"{chain.direction} {state.instrument} with {chain.confidence:.0f}% confidence. "
            f"{'Strong' if chain.confidence >= 75 else 'Moderate'} setup in {state.regime} regime"
            f"{' (Neg Gamma amplifying)' if state.gex_regime == 'Negative' else ''}. "
            f"R:R = {chain.risk_reward_ratio:.1f}."
        )
    
    def _build_narrative(self, chain: EvidenceChain, state: MarketState, hist: HistoricalContext) -> str:
        """3-5 sentence reasoning explanation."""
        if not chain.actionable:
            reasons = []
            if chain.vetoes:
                reasons.append(f"Safety rules triggered ({len(chain.vetoes)} vetoes)")
            if chain.confidence < 60:
                reasons.append(f"confidence only {chain.confidence:.0f}%")
            if chain.counter_arguments:
                reasons.append(f"{len(chain.counter_arguments)} counter-arguments")
            return (
                f"{state.instrument} shows a {state.regime.lower()} market with {'mixed' if state.agreement_factor < 0.6 else 'weak'} signals. "
                f"No trade because: {', '.join(reasons) if reasons else 'insufficient edge'}. "
                f"The smartest action is to wait."
            )
        
        # Actionable signal narrative
        primary = chain.primary_evidence[0] if chain.primary_evidence else None
        primary_text = f"driven primarily by {primary.factor} ({primary.finding})" if primary else "based on multi-factor analysis"
        
        hist_text = ""
        if hist.similar_setups >= 10:
            hist_text = f" Pattern memory shows {hist.win_rate:.0f}% win rate across {hist.similar_setups} similar setups (avg {hist.avg_move_atr:.1f} ATR move in {hist.avg_duration_min:.0f} min)."
        
        counter_text = ""
        if chain.counter_arguments:
            counter_text = f" Counter-arguments: {chain.counter_arguments[0].finding}."
        
        return (
            f"{chain.direction} {state.instrument} at {chain.confidence:.0f}% confidence, "
            f"{primary_text}. "
            f"Market is {state.regime} (ADX-confirmed) with {state.gex_regime} Gamma regime. "
            f"Multi-timeframe {'aligned' if any(e.factor == 'HIGHER_TF' for e in chain.supporting_evidence) else 'mixed'}."
            f"{hist_text}{counter_text}"
        )
