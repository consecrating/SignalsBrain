"""
SignalsBrain — Sample Statistics

Win rates were previously reported as bare point estimates. 9 wins from 12
trades was surfaced as:

    n=12 win_rate=75.0% modifier=+8.0
    reason: "Strong historical edge (75% win rate, 12 trades)"

The 95% binomial interval for 9/12 runs from roughly 45% to 92% — from "loses
money after costs" to "excellent". The only guard was `total_trades < 10`, so 12
samples bought the maximum confidence bonus.

This module supplies the interval, and callers gate on its LOWER bound. An edge
you cannot distinguish from break-even is not an edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class ProportionEstimate:
    """A proportion with an uncertainty interval."""
    successes: int
    n: int
    point: float          # observed rate, 0-1
    lower: float          # interval lower bound, 0-1
    upper: float          # interval upper bound, 0-1
    method: str
    confidence: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def to_dict(self) -> dict:
        return {
            "successes": self.successes,
            "n": self.n,
            "rate_pct": round(self.point * 100, 1),
            "ci_lower_pct": round(self.lower * 100, 1),
            "ci_upper_pct": round(self.upper * 100, 1),
            "ci_width_pct": round(self.width * 100, 1),
            "method": self.method,
            "confidence": self.confidence,
        }

    def describe(self) -> str:
        return (f"{self.successes}/{self.n} = {self.point*100:.0f}% "
                f"[{self.lower*100:.0f}-{self.upper*100:.0f}% "
                f"{int(self.confidence*100)}% CI]")


# Two-sided z quantiles for common confidence levels.
_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> ProportionEstimate:
    """
    Wilson score interval.

    Preferred over the normal approximation because it stays inside [0,1] and
    behaves sensibly for small n and for rates near 0 or 1 — exactly the regime
    a young pattern database is in.
    """
    successes = max(0, int(successes))
    n = max(0, int(n))
    if n == 0:
        return ProportionEstimate(0, 0, 0.0, 0.0, 1.0, "wilson", confidence)
    z = _Z.get(round(confidence, 2), 1.96)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return ProportionEstimate(
        successes=successes, n=n, point=p,
        lower=max(0.0, centre - margin), upper=min(1.0, centre + margin),
        method="wilson", confidence=confidence,
    )


def mean_with_ci(values: Sequence[float], confidence: float = 0.95) -> dict:
    """Sample mean with a t-ish interval (z approximation for n>=30)."""
    xs = [float(v) for v in values if v == v]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci_lower": None, "ci_upper": None}
    mean = sum(xs) / n
    if n == 1:
        return {"n": 1, "mean": round(mean, 4), "sd": 0.0,
                "ci_lower": None, "ci_upper": None}
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    z = _Z.get(round(confidence, 2), 1.96)
    se = sd / math.sqrt(n)
    return {
        "n": n,
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "ci_lower": round(mean - z * se, 4),
        "ci_upper": round(mean + z * se, 4),
    }


def two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> Optional[dict]:
    """
    Test whether two win rates differ.

    Used for degradation detection: "recent 15% below historical" was previously
    a bare threshold with no significance test, so ordinary sampling noise on a
    small sample was reported to the operator as a pattern losing its edge.
    """
    if n1 < 2 or n2 < 2:
        return None
    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return None
    z = (p1 - p2) / se
    # Two-sided p-value from the normal CDF.
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return {
        "rate_1": round(p1, 4), "n_1": n1,
        "rate_2": round(p2, 4), "n_2": n2,
        "z": round(z, 4),
        "p_value": round(p_value, 5),
        "significant_at_05": p_value < 0.05,
    }


def min_samples_for_width(target_width: float = 0.20, p: float = 0.5,
                          confidence: float = 0.95) -> int:
    """
    How many samples are needed before a win rate is worth acting on?

    At p=0.5 and 95% confidence, a +/-10 point interval needs ~96 trades. This is
    the honest answer to "how much history do I need", and it is why the
    confidence modifier is gated on n>=30 with a lower-bound test rather than
    n>=10 on a point estimate.
    """
    z = _Z.get(round(confidence, 2), 1.96)
    half = max(1e-6, target_width / 2.0)
    return int(math.ceil((z * z * p * (1 - p)) / (half * half)))
