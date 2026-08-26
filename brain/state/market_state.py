"""
SignalsBrain — MarketState

The single most important object in the system. Holds the complete
47-dimension state of any instrument at any instant, plus:
- Velocity (rate of change) for each velocity-relevant dimension
- Timestamp + staleness detection
- Serialization for pattern memory + AI model consumption
- Comparison operators (how different is state A from state B?)

This is what makes it superhuman: a human can maybe track 5-6 of these
dimensions in their head. We track all 47 simultaneously, with velocity,
and cross-correlate them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .dimensions import (
    DIMENSIONS, DimensionCategory, CATEGORY_WEIGHTS, Channel,
    DIRECTION_DIMENSIONS, CONVICTION_DIMENSIONS, DIRECTION_CATEGORY_WEIGHTS,
)


@dataclass
class DimensionValue:
    """A single dimension's current reading."""
    raw: float  # Original value in natural units
    normalized: float  # Mapped to [-1, +1]
    velocity: float = 0.0  # Rate of change per scan (normalized units/scan)
    acceleration: float = 0.0  # 2nd derivative (is velocity increasing?)
    stale: bool = False  # True if data is older than expected


@dataclass
class StateQuality:
    """Completeness and freshness metadata for one state snapshot."""
    coverage: float = 0.0
    freshness_seconds: float = 0.0
    missing_dimensions: list[str] = field(default_factory=list)
    populated_count: int = 0
    declared_count: int = 0
    category_coverage: dict[str, float] = field(default_factory=dict)

    def to_dict(self, freshness_seconds: Optional[float] = None) -> dict:
        freshness = self.freshness_seconds if freshness_seconds is None else freshness_seconds
        return {
            "coverage": round(self.coverage, 4),
            "freshness_seconds": round(max(0.0, freshness), 3),
            "missing_dimensions": list(self.missing_dimensions),
            "populated_count": self.populated_count,
            "declared_count": self.declared_count,
            "category_coverage": {
                name: round(value, 4) for name, value in self.category_coverage.items()
            },
        }


@dataclass
class MarketState:
    """
    Complete 47-dimension market state for one instrument at one instant.
    
    This is the brain's perception of reality. Everything else (reasoning,
    signals, decisions) derives from this object.
    """
    
    instrument: str
    timestamp: float = field(default_factory=time.time)
    
    # All 47 dimensions, keyed by dimension name
    dimensions: dict[str, DimensionValue] = field(default_factory=dict)
    
    # Metadata
    source: str = "unknown"  # Where this data came from
    scan_number: int = 0  # Sequential scan count for this instrument
    market_open: bool = False
    
    # Derived composites (computed after all dimensions are set)
    # The two orthogonal channels, exposed so callers can reason about them
    # separately instead of only seeing the product.
    direction_score: float = 0.0    # [-1, +1]  signed, bull/bear only
    conviction_score: float = 0.0   # [0, 1]    unsigned multiplier
    net_directional_bias: float = 0.0  # direction_score * conviction_score * 100
    agreement_factor: float = 0.0  # 0-1, weighted share of DIRECTION dims agreeing

    # Effective per-dimension weights actually used by compute_composites, after
    # any learned multipliers. Attribution reads these rather than recomputing
    # them, which is what keeps the decomposition reconciling with the score when
    # learned weights are non-default.
    effective_weights: dict[str, float] = field(default_factory=dict)
    weight_table_version: int = 0
    regime: str = "unknown"  # Trending/Developing/Choppy
    gex_regime: str = "unknown"  # Positive/Negative/Unknown
    dominant_category: str = ""  # Which category is driving the signal most
    quality: StateQuality = field(default_factory=StateQuality)
    
    # ──────────────────────────────────────────────────────────────────────────
    # SETTERS
    # ──────────────────────────────────────────────────────────────────────────
    
    def set_dimension(self, name: str, raw: float, normalized: float,
                      velocity: float = 0.0, acceleration: float = 0.0):
        """Set a dimension's value."""
        self.dimensions[name] = DimensionValue(
            raw=raw, normalized=normalized,
            velocity=velocity, acceleration=acceleration,
        )
    
    # ──────────────────────────────────────────────────────────────────────────
    # COMPUTED PROPERTIES
    # ──────────────────────────────────────────────────────────────────────────
    
    def compute_quality(self):
        """Compute declared/populated coverage without treating missing data as neutral evidence."""
        declared = set(DIMENSIONS)
        populated = set(self.dimensions) & declared
        category_coverage: dict[str, float] = {}
        for category in DimensionCategory:
            category_dims = {
                name for name, definition in DIMENSIONS.items()
                if definition.category == category
            }
            category_coverage[category.value] = (
                len(category_dims & populated) / len(category_dims) if category_dims else 1.0
            )
        self.quality = StateQuality(
            coverage=len(populated) / len(declared) if declared else 1.0,
            freshness_seconds=max(0.0, self.age_seconds),
            missing_dimensions=sorted(declared - populated),
            populated_count=len(populated),
            declared_count=len(declared),
            category_coverage=category_coverage,
        )

    def quality_dict(self) -> dict:
        """Return quality with freshness recalculated at read time."""
        return self.quality.to_dict(freshness_seconds=self.age_seconds)

    def compute_composites(self, weight_resolver=None, weight_version: int = 0):
        """
        Compute the directional score, the conviction multiplier, and net bias.

        `weight_resolver` is an optional callable `(dimension_name) -> multiplier`
        supplied by a learned WeightTable. It defaults to identity, so an
        untrained system scores exactly as the hand-specified weights intend.
        Previously nothing could adjust these weights: learned values only nudged
        the final confidence scalar, leaving the state scoring untouched.

            net_directional_bias = direction_score * conviction_score * 100

        Only DIRECTION-channel dimensions may influence direction_score. Only
        CONVICTION-channel dimensions may influence conviction_score. CONTEXT
        dimensions influence neither — they are gates handled downstream.

        This is the invariant that removes the two structural defects:
        the clock can no longer express a direction, and trend STRENGTH (ADX)
        now amplifies whatever direction the signed dimensions actually show
        instead of adding a spurious bullish contribution.
        """
        if not self.dimensions:
            return

        self.weight_table_version = weight_version
        self.effective_weights = {}

        def eff_weight(name: str, base: float) -> float:
            if weight_resolver is None:
                return base
            try:
                m = float(weight_resolver(name))
            except (TypeError, ValueError):
                m = 1.0
            if m != m or m <= 0:
                m = 1.0
            return base * m

        # ── DIRECTION: signed, category-weighted ─────────────────────────────
        category_scores: dict[str, float] = {cat.value: 0.0 for cat in DimensionCategory}
        category_populated_weight: dict[str, float] = {cat.value: 0.0 for cat in DimensionCategory}
        category_declared_weight: dict[str, float] = {cat.value: 0.0 for cat in DimensionCategory}
        # Declared weight uses effective weights too, so a learned de-emphasis
        # does not silently redistribute influence to the rest of the category.
        for definition in DIRECTION_DIMENSIONS.values():
            if definition.weight > 0:
                w = eff_weight(definition.name, definition.weight)
                self.effective_weights[definition.name] = w
                category_declared_weight[definition.category.value] += w

        for dim_name, dim_val in self.dimensions.items():
            dim_def = DIRECTION_DIMENSIONS.get(dim_name)
            if not dim_def or dim_def.weight == 0:
                continue
            cat = dim_def.category.value
            w = self.effective_weights.get(dim_name, dim_def.weight)
            # Direction dimensions are signed; clamp defensively so a bad
            # normaliser cannot inject out-of-range influence.
            signed = max(-1.0, min(1.0, dim_val.normalized))
            category_scores[cat] += signed * w
            category_populated_weight[cat] += w

        direction = 0.0
        for cat_enum, cat_weight in DIRECTION_CATEGORY_WEIGHTS.items():
            declared_weight = category_declared_weight.get(cat_enum.value, 0)
            if declared_weight > 0:
                # Missing dimensions contribute no evidence rather than letting a
                # partly-populated category claim its full configured influence.
                cat_bias = category_scores[cat_enum.value] / declared_weight
                direction += cat_bias * cat_weight
        self.direction_score = max(-1.0, min(1.0, direction / 100.0))

        # ── CONVICTION: unsigned multiplier ──────────────────────────────────
        self.conviction_score = self._compute_conviction()

        # ── NET BIAS ─────────────────────────────────────────────────────────
        net = self.direction_score * self.conviction_score * 100.0
        self.net_directional_bias = max(-100.0, min(100.0, net))

        # ── AGREEMENT: only directional dimensions may vote ──────────────────
        # Previously every populated dimension voted, so three clock dimensions
        # flipping sign mechanically rewrote agreement (0.50 -> 1.00 on identical
        # market evidence) while carrying weight 42 in the confidence formula.
        sign = 1 if self.direction_score >= 0 else -1
        agreeing_w = 0.0
        active_w = 0.0
        for dim_name, dim_val in self.dimensions.items():
            dim_def = DIRECTION_DIMENSIONS.get(dim_name)
            if not dim_def or dim_def.weight == 0 or dim_val.normalized == 0:
                continue
            w = self.effective_weights.get(dim_name, dim_def.weight)
            active_w += w
            if (dim_val.normalized > 0) == (sign > 0):
                agreeing_w += w
        # Weight-based agreement so a weight-10 dimension is not outvoted by
        # three weight-3 ones.
        self.agreement_factor = (agreeing_w / active_w) if active_w > 0 else 0.0

        # ── Dominant directional category ────────────────────────────────────
        max_cat = ""
        max_cat_abs = 0.0
        for cat_enum in DIRECTION_CATEGORY_WEIGHTS:
            cat = cat_enum.value
            declared_weight = category_declared_weight.get(cat, 0)
            if declared_weight > 0 and category_populated_weight.get(cat, 0) > 0:
                cat_abs = abs(category_scores[cat] / declared_weight)
                if cat_abs > max_cat_abs:
                    max_cat_abs = cat_abs
                    max_cat = cat
        self.dominant_category = max_cat

        # ── Regime labels ────────────────────────────────────────────────────
        self._derive_regimes()

    def _compute_conviction(self) -> float:
        """
        Unsigned [0,1] multiplier: how much should we trust a directional read?

        Neutral (no conviction data) is 0.65 rather than 0 or 1, so a state with
        no conviction inputs degrades gracefully instead of zeroing the signal
        or claiming perfect confidence.
        """
        NEUTRAL = 0.65
        total_w = 0.0
        acc = 0.0
        for dim_name, dim_val in self.dimensions.items():
            dim_def = CONVICTION_DIMENSIONS.get(dim_name)
            if not dim_def or dim_def.weight == 0:
                continue
            # Conviction dimensions are stored unsigned in [0,1]. Tolerate
            # legacy signed values by folding them into [0,1].
            v = dim_val.normalized
            v = (v + 1.0) / 2.0 if v < 0 else min(1.0, v)
            acc += v * dim_def.weight
            total_w += dim_def.weight
        if total_w == 0:
            return NEUTRAL
        raw = acc / total_w  # [0,1]
        # Blend toward neutral by how much conviction evidence we actually have,
        # so one populated dimension cannot swing the multiplier to an extreme.
        declared = sum(d.weight for d in CONVICTION_DIMENSIONS.values() if d.weight > 0)
        completeness = min(1.0, total_w / declared) if declared else 0.0
        blended = NEUTRAL + (raw - NEUTRAL) * completeness
        # Floor at 0.25: even a dead market retains a little signal value so
        # ranking still works; the DEAD_MARKET veto handles the hard block.
        return max(0.25, min(1.0, blended))

    def _derive_regimes(self):
        """Derive the human-readable regime labels from their source dimensions."""
        # adx_regime is a CONVICTION dimension: `raw` keeps the signed band
        # (-1 choppy / 0 developing / +1 trending) while `normalized` is the
        # unsigned [0,1] multiplier contribution.
        adx_dim = self.dimensions.get("adx_regime")
        if adx_dim:
            if adx_dim.raw >= 0.5:
                self.regime = "Trending"
            elif adx_dim.raw >= -0.5:
                self.regime = "Developing"
            else:
                self.regime = "Choppy"

        # gex_regime is likewise CONVICTION: `raw` is -1 Negative / +1 Positive,
        # `normalized` is the unsigned multiplier (Negative gamma amplifies, so it
        # scores HIGHER conviction). The label must read `raw`, not `normalized`.
        gex_dim = self.dimensions.get("gex_regime")
        if gex_dim:
            if gex_dim.raw < 0:
                self.gex_regime = "Negative"
            elif gex_dim.raw > 0:
                self.gex_regime = "Positive"
            else:
                self.gex_regime = "Unknown"

    # ──────────────────────────────────────────────────────────────────────────
    # QUERIES
    # ──────────────────────────────────────────────────────────────────────────

    def get_category_bias(self, category: DimensionCategory) -> float:
        """
        Net DIRECTIONAL bias for a category, in [-1, +1].

        Only direction-channel dimensions are considered, so asking for the TREND
        category no longer mixes ADX strength into a signed answer.
        """
        score = 0.0
        weight_sum = 0.0
        for dim_name, dim_val in self.dimensions.items():
            dim_def = DIRECTION_DIMENSIONS.get(dim_name)
            if not dim_def or dim_def.category != category or dim_def.weight == 0:
                continue
            score += dim_val.normalized * dim_def.weight
            weight_sum += dim_def.weight
        return score / weight_sum if weight_sum > 0 else 0.0

    def get_conviction_breakdown(self) -> dict[str, float]:
        """Per-dimension contributions to the conviction multiplier."""
        out: dict[str, float] = {}
        for dim_name, dim_val in self.dimensions.items():
            dim_def = CONVICTION_DIMENSIONS.get(dim_name)
            if not dim_def or dim_def.weight == 0:
                continue
            v = dim_val.normalized
            v = (v + 1.0) / 2.0 if v < 0 else min(1.0, v)
            out[dim_name] = round(v, 3)
        return out

    def get_strongest_signals(self, n: int = 5) -> list[tuple[str, float]]:
        """The N directional dimensions with the strongest weighted signal."""
        scored = []
        for dim_name, dim_val in self.dimensions.items():
            dim_def = DIRECTION_DIMENSIONS.get(dim_name)
            if not dim_def or dim_def.weight == 0:
                continue
            strength = abs(dim_val.normalized) * dim_def.weight
            scored.append((dim_name, dim_val.normalized, strength))
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(name, val) for name, val, _ in scored[:n]]

    def get_contradictions(self) -> list[tuple[str, str]]:
        """
        Directional dimensions that oppose the net read.

        Restricted to the direction channel: a low ADX or a quiet tape is not a
        "contradiction", it is low conviction, and conflating the two produced
        counter-arguments with positive point values.
        """
        sign = 1 if self.direction_score >= 0 else -1
        contradictions = []
        for dim_name, dim_val in self.dimensions.items():
            dim_def = DIRECTION_DIMENSIONS.get(dim_name)
            if not dim_def or dim_def.weight < 5:
                continue  # only flag materially weighted contradictions
            if (dim_val.normalized > 0.3 and sign < 0) or (dim_val.normalized < -0.3 and sign > 0):
                contradictions.append((
                    dim_name,
                    f"{'Bullish' if dim_val.normalized > 0 else 'Bearish'} "
                    f"({dim_val.normalized:.2f}) contradicts net "
                    f"{'Bearish' if sign < 0 else 'Bullish'} direction",
                ))
        return contradictions
    
    def get_velocity_alerts(self, threshold: float = 0.3) -> list[tuple[str, float]]:
        """Find dimensions changing rapidly (potential regime shift incoming)."""
        alerts = []
        for dim_name, dim_val in self.dimensions.items():
            dim_def = DIMENSIONS.get(dim_name)
            if not dim_def or not dim_def.velocity_relevant:
                continue
            if abs(dim_val.velocity) > threshold:
                alerts.append((dim_name, dim_val.velocity))
        alerts.sort(key=lambda x: abs(x[1]), reverse=True)
        return alerts
    
    # ──────────────────────────────────────────────────────────────────────────
    # FINGERPRINT (for pattern memory matching)
    # ──────────────────────────────────────────────────────────────────────────
    
    def fingerprint(self) -> np.ndarray:
        """
        Compress state into a fixed-length numeric vector for pattern matching.
        Only includes dimensions with weight > 0, normalized to [-1, +1].
        """
        # Deterministic order: sorted dimension names with weight > 0
        keys = sorted(k for k, d in DIMENSIONS.items() if d.weight > 0)
        vec = np.zeros(len(keys), dtype=np.float32)
        for i, k in enumerate(keys):
            dv = self.dimensions.get(k)
            if dv:
                vec[i] = dv.normalized
        return vec
    
    def fingerprint_keys(self) -> list[str]:
        """Dimension names in fingerprint order."""
        return sorted(k for k, d in DIMENSIONS.items() if d.weight > 0)
    
    # ──────────────────────────────────────────────────────────────────────────
    # SIMILARITY (how close is this state to another?)
    # ──────────────────────────────────────────────────────────────────────────
    
    def similarity(self, other: "MarketState") -> float:
        """
        Cosine similarity between two states (0 = completely different, 1 = identical).
        Weighted by dimension importance.
        """
        fp1 = self.fingerprint()
        fp2 = other.fingerprint()
        
        # Weight vector
        keys = self.fingerprint_keys()
        weights = np.array([DIMENSIONS[k].weight for k in keys], dtype=np.float32)
        
        # Weighted cosine similarity
        w1 = fp1 * weights
        w2 = fp2 * weights
        
        dot = np.dot(w1, w2)
        norm1 = np.linalg.norm(w1)
        norm2 = np.linalg.norm(w2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    @classmethod
    def from_dict(cls, data: dict, quality: Optional[dict] = None) -> "MarketState":
        """Rehydrate a persisted snapshot without recomputing historical values."""
        state = cls(
            instrument=str(data.get("instrument", "")),
            timestamp=float(data.get("timestamp", time.time())),
            market_open=bool(data.get("market_open", False)),
            scan_number=int(data.get("scan_number", 0)),
            source=str(data.get("source", "snapshot")),
        )
        for name, value in data.get("dimensions", {}).items():
            state.set_dimension(
                name,
                float(value.get("raw", 0.0)),
                float(value.get("normalized", 0.0)),
                float(value.get("velocity", 0.0)),
                float(value.get("acceleration", 0.0)),
            )
        state.net_directional_bias = float(data.get("net_bias", data.get("net_directional_bias", 0.0)))
        state.agreement_factor = float(data.get("agreement", data.get("agreement_factor", 0.0)))
        state.regime = str(data.get("regime", "unknown"))
        state.gex_regime = str(data.get("gex_regime", "unknown"))
        state.dominant_category = str(data.get("dominant_category", ""))
        if quality:
            state.quality = StateQuality(
                coverage=float(quality.get("coverage", 0.0)),
                freshness_seconds=float(quality.get("freshness_seconds", 0.0)),
                missing_dimensions=list(quality.get("missing_dimensions", [])),
                populated_count=int(quality.get("populated_count", 0)),
                declared_count=int(quality.get("declared_count", len(DIMENSIONS))),
                category_coverage=dict(quality.get("category_coverage", {})),
            )
        else:
            state.compute_quality()
        return state

    # ──────────────────────────────────────────────────────────────────────────
    # SERIALIZATION (for AI model consumption + storage)
    # ──────────────────────────────────────────────────────────────────────────
    
    def to_dict(self, include_raw: bool = False, include_quality: bool = False) -> dict:
        """Serialize for JSON / AI model context."""
        dims = {}
        for name, dv in self.dimensions.items():
            entry = {"normalized": round(dv.normalized, 3)}
            if include_raw:
                entry["raw"] = dv.raw
            if dv.velocity != 0:
                entry["velocity"] = round(dv.velocity, 4)
            if dv.acceleration != 0:
                entry["acceleration"] = round(dv.acceleration, 4)
            dims[name] = entry
        
        result = {
            "instrument": self.instrument,
            "timestamp": self.timestamp,
            "market_open": self.market_open,
            "scan_number": self.scan_number,
            "net_bias": round(self.net_directional_bias, 1),
            "direction_score": round(self.direction_score, 4),
            "conviction_score": round(self.conviction_score, 4),
            "conviction_breakdown": self.get_conviction_breakdown(),
            "agreement": round(self.agreement_factor, 3),
            "regime": self.regime,
            "gex_regime": self.gex_regime,
            "dominant_category": self.dominant_category,
            "dimensions": dims,
            "strongest": self.get_strongest_signals(5),
            "contradictions": self.get_contradictions(),
            "velocity_alerts": self.get_velocity_alerts(),
        }
        if include_quality:
            result["quality"] = self.quality_dict()
        return result
    
    def to_compact(self) -> dict:
        """Compact representation for AI prompts (minimize tokens)."""
        direction = "BULLISH" if self.net_directional_bias > 5 else "BEARISH" if self.net_directional_bias < -5 else "NEUTRAL"

        # Split the active dimensions by channel so a reader can never mistake a
        # conviction or context reading for a directional one.
        active_dir: dict[str, float] = {}
        active_conv: dict[str, float] = {}
        for name, dv in self.dimensions.items():
            dim_def = DIMENSIONS.get(name)
            if not dim_def or dim_def.weight < 4:
                continue
            if dim_def.channel is Channel.DIRECTION and abs(dv.normalized) >= 0.2:
                active_dir[name] = round(dv.normalized, 2)
            elif dim_def.channel is Channel.CONVICTION:
                active_conv[name] = round(dv.normalized, 2)

        return {
            "instrument": self.instrument,
            "bias": direction,
            "net_score": round(self.net_directional_bias, 1),
            "direction_score": round(self.direction_score, 3),
            "conviction_score": round(self.conviction_score, 3),
            "regime": self.regime,
            "gex": self.gex_regime,
            "agreement": round(self.agreement_factor * 100, 0),
            "direction_dimensions": active_dir,
            "conviction_dimensions": active_conv,
            "velocity_alerts": [(n, round(v, 3)) for n, v in self.get_velocity_alerts()[:3]],
        }
    
    # ──────────────────────────────────────────────────────────────────────────
    # STALENESS
    # ──────────────────────────────────────────────────────────────────────────
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp
    
    @property
    def is_stale(self) -> bool:
        """State older than 60 seconds during market hours is stale."""
        return self.age_seconds > 60 and self.market_open
