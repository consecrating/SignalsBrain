"""
SignalsBrain — Confidence Calculator

Not a simple formula. A Bayesian-inspired multi-stage calculation:

Stage 1: BASE confidence from net directional bias × agreement
Stage 2: REGIME modifier (ADX trending boosts, choppy penalizes)
Stage 3: GEX modifier (distance-scaled, not blind)
Stage 4: HISTORICAL modifier (pattern memory win rate)
Stage 5: VELOCITY modifier (fast-changing dimensions = something happening)
Stage 6: MULTI-TF modifier (higher TF alignment)
Stage 7: EVIDENCE QUALITY (how many strong evidence pieces do we have?)

Each stage is auditable — the AI model sees exactly what contributed what.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .evidence_chain import Evidence, EvidenceWeight, EvidenceDirection


@dataclass
class ConfidenceBreakdown:
    """Shows exactly where each point of confidence came from."""
    base: float = 0.0
    base_explanation: str = ""
    
    regime_modifier: float = 0.0
    regime_explanation: str = ""
    
    gex_modifier: float = 0.0
    gex_explanation: str = ""
    
    historical_modifier: float = 0.0
    historical_explanation: str = ""
    
    velocity_modifier: float = 0.0
    velocity_explanation: str = ""
    
    mtf_modifier: float = 0.0
    mtf_explanation: str = ""
    
    evidence_quality_modifier: float = 0.0
    evidence_quality_explanation: str = ""
    
    penalty_total: float = 0.0
    bonus_total: float = 0.0
    
    final: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "base": {"value": round(self.base, 1), "reason": self.base_explanation},
            "regime": {"value": round(self.regime_modifier, 1), "reason": self.regime_explanation},
            "gex": {"value": round(self.gex_modifier, 1), "reason": self.gex_explanation},
            "historical": {"value": round(self.historical_modifier, 1), "reason": self.historical_explanation},
            "velocity": {"value": round(self.velocity_modifier, 1), "reason": self.velocity_explanation},
            "mtf": {"value": round(self.mtf_modifier, 1), "reason": self.mtf_explanation},
            "evidence_quality": {"value": round(self.evidence_quality_modifier, 1), "reason": self.evidence_quality_explanation},
            "final": round(self.final, 1),
        }


class ConfidenceCalculator:
    """
    Multi-stage confidence calculation with full audit trail.
    """
    
    def calculate(
        self,
        net_bias: float,
        agreement: float,
        evidence: list[Evidence],
        regime: str = "Unknown",
        gex_regime: str = "Unknown",
        gex_flip_distance_atr: float = 999,
        historical_modifier: float = 0,
        historical_explanation: str = "",
        htf_aligned: Optional[bool] = None,
    ) -> ConfidenceBreakdown:
        """
        Calculate confidence with full breakdown.
        
        Args:
            net_bias: Net directional score [-100, +100]
            agreement: Fraction of dimensions agreeing (0-1)
            evidence: All evidence pieces collected
            regime: Market regime (Trending/Developing/Choppy)
            gex_regime: GEX regime (Positive/Negative)
            gex_flip_distance_atr: Distance to GEX flip in ATR units
            historical_modifier: From pattern memory
            historical_explanation: Why that modifier
            htf_aligned: Higher TF agrees with signal direction?
        """
        bd = ConfidenceBreakdown()
        
        # ── Stage 1: BASE ─────────────────────────────────────────────────────
        # Same formula as the live site: |net| × 0.62 + agreement × 42
        bd.base = min(99, abs(net_bias) * 0.62 + agreement * 42)
        bd.base_explanation = f"|net_bias| {abs(net_bias):.0f} × 0.62 + agreement {agreement:.0%} × 42 = {bd.base:.0f}"
        
        # ── Stage 2: REGIME ───────────────────────────────────────────────────
        if regime == "Trending":
            # In a trend, directional signals are more reliable
            bd.regime_modifier = 8
            bd.regime_explanation = f"Trending market (ADX high): directional signals have strong edge (+8)"
        elif regime == "Developing":
            bd.regime_modifier = 3
            bd.regime_explanation = f"Developing trend: moderate directional edge (+3)"
        elif regime == "Choppy":
            bd.regime_modifier = -10
            bd.regime_explanation = f"Choppy market (ADX low): directional signals unreliable (-10)"
        else:
            bd.regime_modifier = 0
            bd.regime_explanation = "Unknown regime: no adjustment"
        
        # ── Stage 3: GEX (distance-scaled) ────────────────────────────────────
        dist = abs(gex_flip_distance_atr)
        if gex_regime == "Negative":
            # Negative Gamma = dealers amplify. Always good for directional trades.
            bd.gex_modifier = 8
            bd.gex_explanation = "Negative Gamma — dealers amplify moves, favorable for momentum (+8)"
        elif gex_regime == "Positive":
            if dist <= 1.0:
                # Transition zone — regime can flip any moment
                bd.gex_modifier = 4
                bd.gex_explanation = f"Positive Gamma but within 1 ATR of flip ({dist:.1f}). Transition zone — explosive potential (+4)"
            elif dist <= 2.0:
                bd.gex_modifier = -3
                bd.gex_explanation = f"Positive Gamma, {dist:.1f} ATR from flip. Mild dealer suppression (-3)"
            elif dist <= 4.0:
                bd.gex_modifier = -6
                bd.gex_explanation = f"Positive Gamma, {dist:.1f} ATR from flip. Significant dealer suppression (-6)"
            else:
                bd.gex_modifier = -10
                bd.gex_explanation = f"Positive Gamma, {dist:.1f} ATR from flip. Dealers firmly in control (-10)"
        else:
            bd.gex_modifier = 0
            bd.gex_explanation = "No GEX data: no adjustment"
        
        # ── Stage 4: HISTORICAL ───────────────────────────────────────────────
        bd.historical_modifier = historical_modifier
        bd.historical_explanation = historical_explanation or "No historical data"
        
        # ── Stage 5: VELOCITY ─────────────────────────────────────────────────
        # If multiple dimensions are changing fast in the same direction = conviction
        velocity_evidence = [e for e in evidence if e.factor.startswith("VELOCITY_")]
        if len(velocity_evidence) >= 3:
            # Multiple dimensions shifting simultaneously — regime shift in progress
            bd.velocity_modifier = 8
            bd.velocity_explanation = f"{len(velocity_evidence)} dimensions shifting rapidly — regime transition detected (+8)"
        elif len(velocity_evidence) >= 1:
            bd.velocity_modifier = 3
            bd.velocity_explanation = f"{len(velocity_evidence)} dimension(s) with high velocity — something building (+3)"
        else:
            bd.velocity_modifier = 0
            bd.velocity_explanation = "No significant velocity — stable state"
        
        # ── Stage 6: MULTI-TF ─────────────────────────────────────────────────
        if htf_aligned is True:
            bd.mtf_modifier = 7
            bd.mtf_explanation = "Higher timeframe (1H) CONFIRMS signal direction (+7)"
        elif htf_aligned is False:
            bd.mtf_modifier = -8
            bd.mtf_explanation = "Higher timeframe CONTRADICTS signal direction (-8)"
        else:
            bd.mtf_modifier = 0
            bd.mtf_explanation = "No higher-TF data"
        
        # ── Stage 7: EVIDENCE QUALITY ─────────────────────────────────────────
        # How many CRITICAL/HIGH weight evidence pieces support the direction?
        critical_count = sum(1 for e in evidence if e.weight in (EvidenceWeight.CRITICAL, EvidenceWeight.HIGH) and e.confidence_impact > 0)
        counter_count = sum(1 for e in evidence if e.weight in (EvidenceWeight.CRITICAL, EvidenceWeight.HIGH) and e.confidence_impact < 0)
        
        if critical_count >= 4 and counter_count == 0:
            bd.evidence_quality_modifier = 6
            bd.evidence_quality_explanation = f"{critical_count} strong confirmations, 0 strong contradictions — conviction (+6)"
        elif critical_count >= 3:
            bd.evidence_quality_modifier = 3
            bd.evidence_quality_explanation = f"{critical_count} strong factors confirm ({counter_count} counter) (+3)"
        elif counter_count >= 3:
            bd.evidence_quality_modifier = -6
            bd.evidence_quality_explanation = f"{counter_count} strong counter-arguments — conflicting evidence (-6)"
        else:
            bd.evidence_quality_modifier = 0
            bd.evidence_quality_explanation = "Mixed evidence quality: no adjustment"
        
        # ── FINAL CALCULATION ─────────────────────────────────────────────────
        bd.bonus_total = sum(x for x in [
            bd.regime_modifier, bd.gex_modifier, bd.historical_modifier,
            bd.velocity_modifier, bd.mtf_modifier, bd.evidence_quality_modifier,
        ] if x > 0)
        
        bd.penalty_total = sum(x for x in [
            bd.regime_modifier, bd.gex_modifier, bd.historical_modifier,
            bd.velocity_modifier, bd.mtf_modifier, bd.evidence_quality_modifier,
        ] if x < 0)
        
        bd.final = max(0, min(99, bd.base + bd.regime_modifier + bd.gex_modifier +
                              bd.historical_modifier + bd.velocity_modifier +
                              bd.mtf_modifier + bd.evidence_quality_modifier))
        
        return bd
