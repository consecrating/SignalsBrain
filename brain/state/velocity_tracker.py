"""
SignalsBrain — Velocity Tracker

Tracks rate-of-change and acceleration for each dimension across scans.
This is what makes the brain superhuman: "PCR is 1.1" is information.
"PCR went from 0.7 to 1.1 in 5 minutes and is accelerating" is intelligence.

A human trader might notice PCR changing if they're staring at it.
They cannot simultaneously track the velocity of PCR, IV skew, GEX flip distance,
volume ratio, and 40 other dimensions. We do.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .dimensions import DIMENSIONS


@dataclass
class DimensionHistory:
    """Rolling history of a single dimension's normalized values."""
    values: deque = field(default_factory=lambda: deque(maxlen=20))  # Last 20 readings
    timestamps: deque = field(default_factory=lambda: deque(maxlen=20))
    
    def add(self, value: float, ts: float):
        self.values.append(value)
        self.timestamps.append(ts)
    
    @property
    def velocity(self) -> float:
        """Rate of change: (latest - previous) per unit time."""
        if len(self.values) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[-2]
        if dt <= 0:
            return 0.0
        # Normalize to "change per 60 seconds" (one scan period)
        return (self.values[-1] - self.values[-2]) / (dt / 60.0)
    
    @property
    def acceleration(self) -> float:
        """Second derivative: is velocity increasing or decreasing?"""
        if len(self.values) < 3:
            return 0.0
        dt1 = self.timestamps[-1] - self.timestamps[-2]
        dt2 = self.timestamps[-2] - self.timestamps[-3]
        if dt1 <= 0 or dt2 <= 0:
            return 0.0
        v_now = (self.values[-1] - self.values[-2]) / (dt1 / 60.0)
        v_prev = (self.values[-2] - self.values[-3]) / (dt2 / 60.0)
        return v_now - v_prev
    
    @property
    def trend_direction(self) -> int:
        """Is this dimension consistently rising (+1), falling (-1), or mixed (0)?"""
        if len(self.values) < 3:
            return 0
        recent = list(self.values)[-5:]  # Last 5 readings
        rises = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        falls = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
        if rises >= 3 and falls == 0:
            return 1
        if falls >= 3 and rises == 0:
            return -1
        return 0
    
    @property
    def volatility(self) -> float:
        """How much is this dimension jumping around? High = unreliable signal."""
        if len(self.values) < 5:
            return 0.0
        recent = list(self.values)[-10:]
        if len(recent) < 3:
            return 0.0
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return variance ** 0.5
    
    @property
    def min_max_range(self) -> tuple[float, float]:
        """Range of this dimension over the tracked history."""
        if not self.values:
            return (0.0, 0.0)
        return (min(self.values), max(self.values))
    
    @property
    def percentile_position(self) -> float:
        """Where is the current value relative to its own recent range? 0=bottom, 1=top."""
        if len(self.values) < 5:
            return 0.5
        lo, hi = self.min_max_range
        if hi <= lo:
            return 0.5
        return (self.values[-1] - lo) / (hi - lo)


class VelocityTracker:
    """
    Maintains rolling history for all dimensions of one instrument.
    Call `update()` on every scan with the new MarketState to track velocities.
    """
    
    def __init__(self, instrument: str):
        self.instrument = instrument
        self.histories: dict[str, DimensionHistory] = {}
        self.scan_count = 0
    
    def update(self, dim_name: str, normalized_value: float, timestamp: float):
        """Record a new reading for a dimension."""
        if dim_name not in self.histories:
            self.histories[dim_name] = DimensionHistory()
        self.histories[dim_name].add(normalized_value, timestamp)
    
    def get_velocity(self, dim_name: str) -> float:
        """Get current velocity for a dimension."""
        h = self.histories.get(dim_name)
        return h.velocity if h else 0.0
    
    def get_acceleration(self, dim_name: str) -> float:
        """Get current acceleration for a dimension."""
        h = self.histories.get(dim_name)
        return h.acceleration if h else 0.0
    
    def get_all_velocities(self) -> dict[str, float]:
        """Get velocity for all tracked dimensions."""
        return {name: h.velocity for name, h in self.histories.items() if abs(h.velocity) > 0.001}
    
    def get_fastest_movers(self, n: int = 5) -> list[tuple[str, float, float]]:
        """
        Get the N dimensions changing fastest right now.
        Returns: [(dim_name, velocity, acceleration), ...]
        
        This is THE key alert mechanism. When PCR velocity suddenly spikes,
        or GEX flip distance is collapsing — that's the signal humans miss.
        """
        movers = []
        for name, h in self.histories.items():
            dim_def = DIMENSIONS.get(name)
            if not dim_def or not dim_def.velocity_relevant:
                continue
            v = h.velocity
            a = h.acceleration
            # Score: |velocity| × dimension_weight (important fast movers matter more)
            score = abs(v) * dim_def.weight
            if score > 0.01:
                movers.append((name, v, a, score))
        
        movers.sort(key=lambda x: x[3], reverse=True)
        return [(name, v, a) for name, v, a, _ in movers[:n]]
    
    def detect_regime_shift(self) -> Optional[dict]:
        """
        Detect if multiple dimensions are shifting simultaneously in the same direction.
        This indicates a regime change is happening NOW — the most profitable moment.
        
        Returns None if no shift detected, else a dict describing the shift.
        """
        velocities = self.get_all_velocities()
        if not velocities:
            return None
        
        # Count how many important dimensions are moving in the same direction
        bullish_movers = []
        bearish_movers = []
        
        for name, vel in velocities.items():
            dim_def = DIMENSIONS.get(name)
            if not dim_def or dim_def.weight < 5:
                continue
            if vel > 0.1:
                bullish_movers.append((name, vel))
            elif vel < -0.1:
                bearish_movers.append((name, vel))
        
        # Regime shift = 4+ important dimensions moving together
        if len(bullish_movers) >= 4:
            return {
                "type": "BULLISH_SHIFT",
                "strength": len(bullish_movers),
                "movers": bullish_movers,
                "note": f"{len(bullish_movers)} dimensions shifting bullish simultaneously — potential regime change",
            }
        if len(bearish_movers) >= 4:
            return {
                "type": "BEARISH_SHIFT",
                "strength": len(bearish_movers),
                "movers": bearish_movers,
                "note": f"{len(bearish_movers)} dimensions shifting bearish simultaneously — potential regime change",
            }
        
        return None
    
    def detect_divergence(self) -> list[dict]:
        """
        Detect when related dimensions diverge (e.g., price rising but momentum falling).
        Divergences often precede reversals.
        """
        divergences = []
        
        # Price vs Momentum divergence
        price_vel = self.get_velocity("day_change_pct")
        rsi_vel = self.get_velocity("rsi")
        if price_vel > 0.1 and rsi_vel < -0.1:
            divergences.append({
                "type": "BEARISH_DIVERGENCE",
                "description": "Price rising but RSI falling — momentum weakening under the surface",
                "severity": "HIGH",
            })
        elif price_vel < -0.1 and rsi_vel > 0.1:
            divergences.append({
                "type": "BULLISH_DIVERGENCE",
                "description": "Price falling but RSI rising — selling pressure exhausting",
                "severity": "HIGH",
            })
        
        # PCR vs Price divergence (institutional positioning)
        pcr_vel = self.get_velocity("pcr")
        if price_vel > 0.1 and pcr_vel < -0.15:
            divergences.append({
                "type": "PCR_WARNING",
                "description": "Price rising but PCR falling (put support withdrawing) — rally may not sustain",
                "severity": "MEDIUM",
            })
        
        # Volume vs Price (conviction check)
        vol_vel = self.get_velocity("volume_ratio")
        if abs(price_vel) > 0.1 and vol_vel < -0.1:
            divergences.append({
                "type": "LOW_CONVICTION",
                "description": "Price moving but volume declining — move lacks institutional backing",
                "severity": "MEDIUM",
            })
        
        return divergences
