"""
SignalsBrain — Exact Score Attribution

The point of this module is a single invariant:

    sum(contribution.points for all contributions) + external == final_score

Previously the evidence chain and the confidence score were computed by two
independent code paths. `ConfidenceCalculator` received the evidence list but
only ever *counted* items; it never summed their `confidence_impact` values.
Measured behaviour on a fixed state:

    confidence WITH all evidence : 40.82
    confidence WITH evidence=[]  : 40.82   (delta 0.00)
    every impact forced to +1000 : 43.82   (moved only by a branch on the count)

So the "+8 pts" / "-5 pts" annotations shown to the operator and injected into
the AI model prompt were decorative. For a system whose stated selling point is
"no black boxes, every decision traceable", a fabricated audit trail is worse
than no audit trail, because it invites trust it has not earned.

This module derives the explanation FROM the arithmetic instead of alongside it.
Every reported contribution is the dimension's actual share of the score, and
`verify()` asserts the decomposition adds up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..state.dimensions import (
    DIMENSIONS, Channel, DimensionCategory,
    DIRECTION_DIMENSIONS, CONVICTION_DIMENSIONS, DIRECTION_CATEGORY_WEIGHTS,
)
from ..state.market_state import MarketState


# Scale factors converting the state's internal units into "confidence points".
# Kept close to the original formula (|net| * 0.62 + agreement * 42) so existing
# thresholds remain meaningful, but now every point is attributable.
DIRECTION_POINTS = 62.0   # applied to |direction_score * conviction_score|
AGREEMENT_POINTS = 38.0   # applied to the weighted agreement fraction


@dataclass
class Contribution:
    """One dimension's exact, signed share of the final score."""
    dimension: str
    channel: str
    category: str
    normalized: float
    raw: float
    weight: float
    points: float          # actual signed contribution, in confidence points
    supports_signal: bool  # does it agree with the chosen direction?
    velocity: float = 0.0
    acceleration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "channel": self.channel,
            "category": self.category,
            "value": round(self.normalized, 3),
            "weight": self.weight,
            "points": round(self.points, 2),
            "supports": self.supports_signal,
        }


@dataclass
class Attribution:
    """Full decomposition of a score into its sources."""
    direction_sign: int = 0
    direction_points: float = 0.0
    agreement_points: float = 0.0
    external_points: float = 0.0          # pattern memory, etc.
    external_reason: str = ""
    contributions: list[Contribution] = field(default_factory=list)
    conviction_multiplier: float = 0.0
    total: float = 0.0

    @property
    def supporting(self) -> list[Contribution]:
        return sorted([c for c in self.contributions if c.supports_signal and c.points != 0],
                      key=lambda c: abs(c.points), reverse=True)

    @property
    def opposing(self) -> list[Contribution]:
        return sorted([c for c in self.contributions if not c.supports_signal and c.points != 0],
                      key=lambda c: abs(c.points), reverse=True)

    def attributed_sum(self) -> float:
        return sum(c.points for c in self.contributions) + self.agreement_points + self.external_points

    def verify(self, tolerance: float = 1e-6) -> bool:
        """The invariant. Tested directly in tests/test_attribution.py."""
        return abs(self.attributed_sum() - self.total) <= tolerance

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 2),
            "direction_points": round(self.direction_points, 2),
            "agreement_points": round(self.agreement_points, 2),
            "external_points": round(self.external_points, 2),
            "external_reason": self.external_reason,
            "conviction_multiplier": round(self.conviction_multiplier, 3),
            "attributed_sum": round(self.attributed_sum(), 2),
            "reconciles": self.verify(1e-4),
            "supporting": [c.to_dict() for c in self.supporting],
            "opposing": [c.to_dict() for c in self.opposing],
        }


def attribute(state: MarketState, direction: str,
              external_points: float = 0.0,
              external_reason: str = "") -> Attribution:
    """
    Decompose the score for `state` into exact per-dimension contributions.

    The directional magnitude is |direction_score| * conviction_score * 62. Each
    directional dimension owns the share of that magnitude it actually produced:

        share_d = (normalized_d * weight_d / declared_category_weight)
                  * category_weight / 100

    Summing share_d over all directional dimensions reproduces direction_score
    exactly, so scaling by the same constants yields contributions that sum to
    the directional points. Signs are preserved: a dimension pointing against
    the chosen direction contributes NEGATIVE points.
    """
    att = Attribution()
    att.conviction_multiplier = state.conviction_score
    att.external_points = external_points
    att.external_reason = external_reason

    sign = 0
    if direction == "BUY":
        sign = 1
    elif direction == "SELL":
        sign = -1
    else:
        sign = 1 if state.direction_score >= 0 else -1
    att.direction_sign = sign

    # Effective weights are read from the state, not recomputed. compute_composites
    # records exactly what it used (including any learned multipliers), so the
    # decomposition cannot drift away from the score it is meant to explain.
    eff = state.effective_weights or {}

    def weight_of(d) -> float:
        return eff.get(d.name, d.weight)

    # Declared weight per direction category (missing dims contribute nothing
    # rather than letting a partial category claim full influence).
    declared: dict[str, float] = {}
    for d in DIRECTION_DIMENSIONS.values():
        if d.weight > 0:
            declared[d.category.value] = declared.get(d.category.value, 0.0) + weight_of(d)

    scale = DIRECTION_POINTS * state.conviction_score

    for name, dv in state.dimensions.items():
        d = DIRECTION_DIMENSIONS.get(name)
        if not d or d.weight == 0:
            continue
        cat_declared = declared.get(d.category.value, 0.0)
        cat_weight = DIRECTION_CATEGORY_WEIGHTS.get(d.category, 0)
        if cat_declared <= 0 or cat_weight == 0:
            continue
        signed = max(-1.0, min(1.0, dv.normalized))
        # Share of direction_score produced by this dimension.
        share = (signed * weight_of(d) / cat_declared) * cat_weight / 100.0
        # Points, oriented so that "agrees with the signal" is positive.
        points = share * scale * sign
        att.contributions.append(Contribution(
            dimension=name,
            channel=d.channel.value,
            category=d.category.value,
            normalized=signed,
            raw=dv.raw,
            weight=weight_of(d),
            points=points,
            supports_signal=points >= 0,
            velocity=dv.velocity,
            acceleration=dv.acceleration,
        ))

    att.direction_points = sum(c.points for c in att.contributions)
    att.agreement_points = state.agreement_factor * AGREEMENT_POINTS
    att.total = att.direction_points + att.agreement_points + att.external_points
    return att


def conviction_attribution(state: MarketState) -> list[Contribution]:
    """
    Explain the conviction multiplier separately.

    Conviction is a multiplier, not an addend, so its members do not appear in
    the additive decomposition. Reporting them separately keeps the explanation
    complete without pretending they contribute points.
    """
    out: list[Contribution] = []
    for name, dv in state.dimensions.items():
        d = CONVICTION_DIMENSIONS.get(name)
        if not d or d.weight == 0:
            continue
        v = dv.normalized
        v = (v + 1.0) / 2.0 if v < 0 else min(1.0, v)
        out.append(Contribution(
            dimension=name,
            channel=d.channel.value,
            category=d.category.value,
            normalized=v,
            raw=dv.raw,
            weight=d.weight,
            points=0.0,               # multiplicative, not additive
            supports_signal=v >= 0.5,
            velocity=dv.velocity,
            acceleration=dv.acceleration,
        ))
    out.sort(key=lambda c: c.weight, reverse=True)
    return out
