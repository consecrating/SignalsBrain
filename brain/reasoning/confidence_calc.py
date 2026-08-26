"""
SignalsBrain — Confidence Calculator

Confidence is now computed as an exact sum of attributed contributions:

    confidence = sum(per-dimension points) + agreement_points + external_points

Three defects this replaces:

1. **Non-causal explanation.** The old implementation accepted the evidence list
   but never summed `confidence_impact`; it only counted items. Perturbing any
   impact by any amount left the score unchanged (measured delta: 0.00). The
   displayed reasoning was therefore unrelated to the number.

2. **Double counting.** Stage 2 added up to +8 for "Trending", but `adx_regime`
   (w8) and `adx_value` (w7) were already inside net_bias. Stage 3 added up to
   +8 for the GEX regime, already present via `gex_regime` (w10) and
   `gex_flip_distance` (w9). Stage 6 added up to +7 for higher-timeframe
   agreement, already present via `htf_trend` (w7). Best case summed to 130 and
   was then clamped to 99, discarding 31 points of headroom.

   Those stages are gone. Regime, GEX and MTF each speak exactly once, through
   their own dimension — ADX and GEX now via the conviction multiplier, HTF via
   the direction channel.

3. **Saturation.** With everything favourable the old score pinned at 99 for any
   net_bias >= 60, making the top 40% of the range indistinguishable and the
   number useless for ranking or position sizing. The score is now unbounded
   below 100 by construction and is additionally exposed as a calibrated
   probability (see calibration.py) rather than an arbitrary 0-99 figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .attribution import (
    Attribution, Contribution, attribute, conviction_attribution,
    DIRECTION_POINTS, AGREEMENT_POINTS,
)
from ..state.market_state import MarketState


@dataclass
class ConfidenceBreakdown:
    """
    Where every point came from.

    The legacy stage fields are retained (and reported as 0.0) so existing
    consumers and serialised payloads keep the same shape, with an explicit note
    that the stage no longer exists.
    """
    base: float = 0.0
    base_explanation: str = ""

    # Retained for payload compatibility; always 0.0 now.
    regime_modifier: float = 0.0
    regime_explanation: str = "folded into the conviction multiplier (no longer double counted)"
    gex_modifier: float = 0.0
    gex_explanation: str = "folded into the conviction multiplier (no longer double counted)"
    mtf_modifier: float = 0.0
    mtf_explanation: str = "counted once via the htf_trend direction dimension"
    velocity_modifier: float = 0.0
    velocity_explanation: str = "velocity is reported per dimension, not re-added"
    evidence_quality_modifier: float = 0.0
    evidence_quality_explanation: str = "superseded by exact attribution"

    historical_modifier: float = 0.0
    historical_explanation: str = ""

    # Data-quality discount. This is NOT double counting: it prices the absence
    # of inputs, which no dimension can express (a missing dimension contributes
    # zero, which is indistinguishable from a genuinely neutral reading).
    coverage_modifier: float = 0.0
    coverage_explanation: str = ""

    agreement_points: float = 0.0
    direction_points: float = 0.0
    conviction_multiplier: float = 0.0

    penalty_total: float = 0.0
    bonus_total: float = 0.0
    final: float = 0.0

    attribution: Optional[Attribution] = None
    conviction_detail: list[Contribution] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "base": {"value": round(self.base, 1), "reason": self.base_explanation},
            "regime": {"value": 0.0, "reason": self.regime_explanation},
            "gex": {"value": 0.0, "reason": self.gex_explanation},
            "historical": {"value": round(self.historical_modifier, 1),
                           "reason": self.historical_explanation},
            "velocity": {"value": 0.0, "reason": self.velocity_explanation},
            "mtf": {"value": 0.0, "reason": self.mtf_explanation},
            "evidence_quality": {"value": 0.0, "reason": self.evidence_quality_explanation},
            "coverage": {"value": round(self.coverage_modifier, 1),
                         "reason": self.coverage_explanation},
            "direction_points": round(self.direction_points, 2),
            "agreement_points": round(self.agreement_points, 2),
            "conviction_multiplier": round(self.conviction_multiplier, 3),
            "final": round(self.final, 1),
        }
        if self.attribution is not None:
            d["attribution"] = self.attribution.to_dict()
            d["reconciles"] = self.attribution.verify(1e-4)
        if self.conviction_detail:
            d["conviction_detail"] = [c.to_dict() for c in self.conviction_detail]
        return d


class ConfidenceCalculator:
    """Additive, fully attributable confidence."""

    def calculate_from_state(
        self,
        state: MarketState,
        direction: str,
        historical_modifier: float = 0.0,
        historical_explanation: str = "",
        coverage: Optional[float] = None,
    ) -> ConfidenceBreakdown:
        """
        Primary entry point. Confidence is the attributed sum, nothing else.
        """
        cov = state.quality.coverage if coverage is None else coverage
        cov = max(0.0, min(1.0, cov))
        coverage_penalty = -20.0 * (1.0 - cov)

        att = attribute(
            state,
            direction,
            external_points=historical_modifier + coverage_penalty,
            external_reason=(
                f"{historical_explanation or 'no historical data'}; "
                f"coverage {cov:.0%} ({coverage_penalty:+.1f})"
            ),
        )
        bd = ConfidenceBreakdown()
        bd.attribution = att
        bd.conviction_detail = conviction_attribution(state)
        bd.direction_points = att.direction_points
        bd.agreement_points = att.agreement_points
        bd.conviction_multiplier = att.conviction_multiplier
        bd.historical_modifier = historical_modifier
        bd.historical_explanation = historical_explanation or "no historical data"
        bd.coverage_modifier = coverage_penalty
        bd.coverage_explanation = (
            f"State coverage {cov:.0%}; incomplete inputs reduce confidence "
            f"({coverage_penalty:.1f})"
        )

        bd.base = att.direction_points + att.agreement_points
        bd.base_explanation = (
            f"direction {att.direction_points:+.1f} pts "
            f"(|dir| {abs(state.direction_score):.3f} x conviction "
            f"{state.conviction_score:.3f} x {DIRECTION_POINTS:.0f}) "
            f"+ agreement {att.agreement_points:.1f} pts "
            f"({state.agreement_factor:.0%} x {AGREEMENT_POINTS:.0f})"
        )

        bd.bonus_total = (sum(c.points for c in att.contributions if c.points > 0)
                          + max(0.0, historical_modifier))
        bd.penalty_total = (sum(c.points for c in att.contributions if c.points < 0)
                            + min(0.0, historical_modifier) + coverage_penalty)

        # Clamp only at the hard bounds of the reporting scale. There is no
        # interior ceiling, so distinct states keep distinct scores.
        bd.final = max(0.0, min(100.0, att.total))
        # If the clamp actually binds, fold the difference into the external term
        # so the reported decomposition still reconciles with the reported score.
        if abs(bd.final - att.total) > 1e-9:
            base_sum = sum(c.points for c in att.contributions) + att.agreement_points
            att.external_points = bd.final - base_sum
            att.external_reason = (
                f"{historical_explanation or 'external'} "
                f"(adjusted for the 0-100 reporting clamp)"
            )
            att.total = bd.final
        return bd

    # ──────────────────────────────────────────────────────────────────────────
    # Backwards-compatible shim
    # ──────────────────────────────────────────────────────────────────────────

    def calculate(
        self,
        net_bias: float,
        agreement: float,
        evidence: list = None,
        regime: str = "Unknown",
        gex_regime: str = "Unknown",
        gex_flip_distance_atr: float = 999,
        historical_modifier: float = 0,
        historical_explanation: str = "",
        htf_aligned: Optional[bool] = None,
        coverage: float = 1.0,
    ) -> ConfidenceBreakdown:
        """
        Legacy signature, retained so older callers keep working.

        It no longer applies the regime / GEX / MTF / velocity / evidence-count
        stages, because each of those double counted a dimension already present
        in `net_bias`. The result is the additive core only:

            |net_bias|/100 * 62 + agreement * 38 + historical_modifier

        Prefer `calculate_from_state`, which also returns the exact per-dimension
        attribution.
        """
        cov = max(0.0, min(1.0, coverage))
        coverage_penalty = -20.0 * (1.0 - cov)
        bd = ConfidenceBreakdown()
        bd.direction_points = abs(net_bias) / 100.0 * DIRECTION_POINTS
        bd.agreement_points = agreement * AGREEMENT_POINTS
        bd.historical_modifier = historical_modifier
        bd.historical_explanation = historical_explanation or "no historical data"
        bd.coverage_modifier = coverage_penalty
        bd.coverage_explanation = (
            f"State coverage {cov:.0%}; incomplete inputs reduce confidence "
            f"({coverage_penalty:.1f})"
        )
        bd.base = bd.direction_points + bd.agreement_points
        bd.base_explanation = (
            f"|net_bias| {abs(net_bias):.0f}/100 x {DIRECTION_POINTS:.0f} "
            f"+ agreement {agreement:.0%} x {AGREEMENT_POINTS:.0f}"
        )
        bd.bonus_total = max(0.0, historical_modifier)
        bd.penalty_total = min(0.0, historical_modifier) + coverage_penalty
        bd.final = max(0.0, min(100.0, bd.base + historical_modifier + coverage_penalty))
        return bd
