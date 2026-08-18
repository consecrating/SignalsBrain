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

from .dimensions import DIMENSIONS, DimensionCategory, CATEGORY_WEIGHTS


@dataclass
class DimensionValue:
    """A single dimension's current reading."""
    raw: float  # Original value in natural units
    normalized: float  # Mapped to [-1, +1]
    velocity: float = 0.0  # Rate of change per scan (normalized units/scan)
    acceleration: float = 0.0  # 2nd derivative (is velocity increasing?)
    stale: bool = False  # True if data is older than expected


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
    net_directional_bias: float = 0.0  # [-100, +100]
    agreement_factor: float = 0.0  # 0-1 (how many dimensions agree on direction)
    regime: str = "unknown"  # Trending/Developing/Choppy
    gex_regime: str = "unknown"  # Positive/Negative/Unknown
    dominant_category: str = ""  # Which category is driving the signal most
    
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
    
    def compute_composites(self):
        """Calculate net bias, agreement, regime from all dimensions."""
        if not self.dimensions:
            return
        
        # Net directional bias: weighted sum of all normalized dimensions
        # Each dimension contributes: normalized_value × dimension_weight × category_weight
        category_scores: dict[str, float] = {cat.value: 0.0 for cat in DimensionCategory}
        category_counts: dict[str, int] = {cat.value: 0 for cat in DimensionCategory}
        
        for dim_name, dim_val in self.dimensions.items():
            dim_def = DIMENSIONS.get(dim_name)
            if not dim_def or dim_def.weight == 0:
                continue
            
            # Dimension contributes its normalized value × its weight within its category
            cat = dim_def.category.value
            category_scores[cat] += dim_val.normalized * dim_def.weight
            category_counts[cat] += dim_def.weight
        
        # Normalize each category to [-1, +1], then apply category weights
        net = 0.0
        for cat_enum in DimensionCategory:
            cat = cat_enum.value
            total_weight = category_counts.get(cat, 0)
            if total_weight > 0:
                cat_bias = category_scores[cat] / total_weight  # [-1, +1]
                net += cat_bias * CATEGORY_WEIGHTS.get(cat_enum, 0)
        
        # net is now in [-100, +100]
        self.net_directional_bias = max(-100, min(100, net))
        
        # Agreement: how many dimensions agree with the dominant direction
        direction = 1 if net >= 0 else -1
        agreeing = 0
        total_active = 0
        for dim_name, dim_val in self.dimensions.items():
            dim_def = DIMENSIONS.get(dim_name)
            if not dim_def or dim_def.weight == 0:
                continue
            if dim_val.normalized == 0:
                continue
            total_active += 1
            if (dim_val.normalized > 0 and direction > 0) or (dim_val.normalized < 0 and direction < 0):
                agreeing += 1
        
        self.agreement_factor = agreeing / max(1, total_active)
        
        # Find dominant category
        max_cat = ""
        max_cat_abs = 0
        for cat_enum in DimensionCategory:
            cat = cat_enum.value
            total_weight = category_counts.get(cat, 0)
            if total_weight > 0:
                cat_abs = abs(category_scores[cat] / total_weight)
                if cat_abs > max_cat_abs:
                    max_cat_abs = cat_abs
                    max_cat = cat
        self.dominant_category = max_cat
        
        # Regime from ADX dimension
        adx_dim = self.dimensions.get("adx_regime")
        if adx_dim:
            if adx_dim.raw >= 0.5:
                self.regime = "Trending"
            elif adx_dim.raw >= -0.5:
                self.regime = "Developing"
            else:
                self.regime = "Choppy"
        
        # GEX regime
        gex_dim = self.dimensions.get("gex_regime")
        if gex_dim:
            self.gex_regime = "Positive" if gex_dim.normalized > 0 else "Negative" if gex_dim.normalized < 0 else "Unknown"
    
    # ──────────────────────────────────────────────────────────────────────────
    # QUERIES
    # ──────────────────────────────────────────────────────────────────────────
    
    def get_category_bias(self, category: DimensionCategory) -> float:
        """Get the net bias for a specific category."""
        score = 0.0
        weight_sum = 0.0
        for dim_name, dim_val in self.dimensions.items():
            dim_def = DIMENSIONS.get(dim_name)
            if not dim_def or dim_def.category != category or dim_def.weight == 0:
                continue
            score += dim_val.normalized * dim_def.weight
            weight_sum += dim_def.weight
        return score / max(1, weight_sum)
    
    def get_strongest_signals(self, n: int = 5) -> list[tuple[str, float]]:
        """Get the N dimensions with the strongest absolute signal."""
        scored = []
        for dim_name, dim_val in self.dimensions.items():
            dim_def = DIMENSIONS.get(dim_name)
            if not dim_def or dim_def.weight == 0:
                continue
            strength = abs(dim_val.normalized) * dim_def.weight
            scored.append((dim_name, dim_val.normalized, strength))
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(name, val) for name, val, _ in scored[:n]]
    
    def get_contradictions(self) -> list[tuple[str, str]]:
        """Find dimensions that contradict the net bias (potential warning signs)."""
        direction = 1 if self.net_directional_bias >= 0 else -1
        contradictions = []
        for dim_name, dim_val in self.dimensions.items():
            dim_def = DIMENSIONS.get(dim_name)
            if not dim_def or dim_def.weight < 5:
                continue  # Only flag important contradictions
            if (dim_val.normalized > 0.3 and direction < 0) or (dim_val.normalized < -0.3 and direction > 0):
                contradictions.append((dim_name, f"{'Bullish' if dim_val.normalized > 0 else 'Bearish'} ({dim_val.normalized:.2f}) contradicts net {'Bearish' if direction < 0 else 'Bullish'} bias"))
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
    
    # ──────────────────────────────────────────────────────────────────────────
    # SERIALIZATION (for AI model consumption + storage)
    # ──────────────────────────────────────────────────────────────────────────
    
    def to_dict(self, include_raw: bool = False) -> dict:
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
        
        return {
            "instrument": self.instrument,
            "timestamp": self.timestamp,
            "market_open": self.market_open,
            "scan_number": self.scan_number,
            "net_bias": round(self.net_directional_bias, 1),
            "agreement": round(self.agreement_factor, 3),
            "regime": self.regime,
            "gex_regime": self.gex_regime,
            "dominant_category": self.dominant_category,
            "dimensions": dims,
            "strongest": self.get_strongest_signals(5),
            "contradictions": self.get_contradictions(),
            "velocity_alerts": self.get_velocity_alerts(),
        }
    
    def to_compact(self) -> dict:
        """Compact representation for AI prompts (minimize tokens)."""
        direction = "BULLISH" if self.net_directional_bias > 5 else "BEARISH" if self.net_directional_bias < -5 else "NEUTRAL"
        
        # Only include dimensions that are significantly non-zero
        active_dims = {}
        for name, dv in self.dimensions.items():
            if abs(dv.normalized) >= 0.2:
                dim_def = DIMENSIONS.get(name)
                if dim_def and dim_def.weight >= 4:
                    active_dims[name] = round(dv.normalized, 2)
        
        return {
            "instrument": self.instrument,
            "bias": direction,
            "net_score": round(self.net_directional_bias, 1),
            "confidence_input": round(abs(self.net_directional_bias) * 0.62 + self.agreement_factor * 42, 0),
            "regime": self.regime,
            "gex": self.gex_regime,
            "agreement": round(self.agreement_factor * 100, 0),
            "key_dimensions": active_dims,
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
