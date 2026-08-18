"""
SignalsBrain — Fingerprint

Compresses a 47-dimension MarketState into a compact, matchable representation.
Two types of fingerprints:

1. FULL VECTOR — 47-float numpy array for cosine similarity matching
   (finds the most similar historical state regardless of what dimensions matched)

2. CATEGORICAL FINGERPRINT — a discrete bucketed representation for exact-match queries
   ("Show me all times when GEX was Negative AND PCR > 1.2 AND ADX > 25")
   This is what makes queries like "last 50 times this setup occurred" fast.

The categorical fingerprint uses 12 key dimensions bucketed into discrete levels.
This gives us a "setup type" classification that's more useful than raw similarity
for historical win-rate calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CategoricalFingerprint:
    """
    Discrete bucketed representation of the 12 most predictive dimensions.
    Each field is a small integer (typically -2 to +2 or an enum-like value).
    
    This enables SQL queries like:
      WHERE gex_regime = -1 AND pcr_band = 2 AND adx_band = 2 AND trend_dir = -1
    """
    # GEX regime: -1=Negative, 0=Unknown, 1=Positive
    gex_regime: int = 0
    
    # GEX flip proximity: -2=far below, -1=near below, 0=at flip, 1=near above, 2=far above
    gex_flip_zone: int = 0
    
    # PCR band: -2=very low(<0.6), -1=low(0.6-0.85), 0=neutral(0.85-1.15), 1=high(1.15-1.5), 2=very high(>1.5)
    pcr_band: int = 0
    
    # ADX band: -1=choppy(<18), 0=developing(18-25), 1=trending(25-35), 2=strong trend(>35)
    adx_band: int = 0
    
    # Trend direction: -2=strong bear, -1=bear, 0=neutral, 1=bull, 2=strong bull
    trend_dir: int = 0
    
    # Momentum (RSI zone): -2=oversold(<25), -1=weak(<40), 0=neutral(40-60), 1=strong(>60), 2=overbought(>75)
    momentum_zone: int = 0
    
    # Volume state: -1=dry(<0.7x), 0=normal(0.7-1.5x), 1=elevated(1.5-2.5x), 2=spike(>2.5x)
    volume_state: int = 0
    
    # IV regime: -1=low(<12%), 0=normal(12-20%), 1=high(20-28%), 2=extreme(>28%)
    iv_regime: int = 0
    
    # VWAP position: -1=below, 0=at, 1=above
    vwap_pos: int = 0
    
    # Session phase: 0=opening, 1=morning, 2=midday, 3=afternoon, 4=closing
    session: int = 0
    
    # DTE band: 0=expiry day, 1=1-2 days, 2=3-5 days, 3=5+ days
    dte_band: int = 0
    
    # FII flow direction: -1=selling, 0=neutral, 1=buying
    fii_dir: int = 0
    
    def to_tuple(self) -> tuple:
        """For hashing and comparison."""
        return (
            self.gex_regime, self.gex_flip_zone, self.pcr_band, self.adx_band,
            self.trend_dir, self.momentum_zone, self.volume_state, self.iv_regime,
            self.vwap_pos, self.session, self.dte_band, self.fii_dir,
        )
    
    def to_dict(self) -> dict:
        return {
            "gex_regime": self.gex_regime,
            "gex_flip_zone": self.gex_flip_zone,
            "pcr_band": self.pcr_band,
            "adx_band": self.adx_band,
            "trend_dir": self.trend_dir,
            "momentum_zone": self.momentum_zone,
            "volume_state": self.volume_state,
            "iv_regime": self.iv_regime,
            "vwap_pos": self.vwap_pos,
            "session": self.session,
            "dte_band": self.dte_band,
            "fii_dir": self.fii_dir,
        }
    
    def match_score(self, other: "CategoricalFingerprint") -> float:
        """
        How many of the 12 categories match exactly? Returns 0-1.
        1.0 = identical setup type. 0.75+ = very similar. 0.5+ = related.
        """
        t1 = self.to_tuple()
        t2 = other.to_tuple()
        matches = sum(1 for a, b in zip(t1, t2) if a == b)
        return matches / len(t1)
    
    def relaxed_match_score(self, other: "CategoricalFingerprint") -> float:
        """
        Relaxed matching: adjacent bands count as half-match.
        E.g., adx_band=1 vs adx_band=2 = 0.5 (close enough).
        """
        t1 = self.to_tuple()
        t2 = other.to_tuple()
        score = 0.0
        for a, b in zip(t1, t2):
            if a == b:
                score += 1.0
            elif abs(a - b) == 1:
                score += 0.5
        return score / len(t1)


def build_categorical_fingerprint(state) -> CategoricalFingerprint:
    """
    Build a CategoricalFingerprint from a MarketState object.
    Buckets each key dimension into discrete levels for exact-match queries.
    """
    dims = state.dimensions
    
    def raw(name: str, default=0.0) -> float:
        d = dims.get(name)
        return d.raw if d else default
    
    def norm(name: str, default=0.0) -> float:
        d = dims.get(name)
        return d.normalized if d else default
    
    # GEX regime
    gex_n = norm("gex_regime")
    gex_regime = -1 if gex_n < -0.3 else (1 if gex_n > 0.3 else 0)
    
    # GEX flip zone (in ATR units)
    flip_dist = raw("gex_flip_distance", 0)
    if flip_dist < -2:
        gex_flip_zone = -2
    elif flip_dist < -0.5:
        gex_flip_zone = -1
    elif flip_dist <= 0.5:
        gex_flip_zone = 0
    elif flip_dist <= 2:
        gex_flip_zone = 1
    else:
        gex_flip_zone = 2
    
    # PCR band
    pcr = raw("pcr", 1.0)
    if pcr < 0.6:
        pcr_band = -2
    elif pcr < 0.85:
        pcr_band = -1
    elif pcr <= 1.15:
        pcr_band = 0
    elif pcr <= 1.5:
        pcr_band = 1
    else:
        pcr_band = 2
    
    # ADX band
    adx = raw("adx_value", 15)
    if adx < 18:
        adx_band = -1
    elif adx < 25:
        adx_band = 0
    elif adx <= 35:
        adx_band = 1
    else:
        adx_band = 2
    
    # Trend direction (from EMA stack + supertrend)
    ema_stack = norm("ema_stack_score", 0)
    st = norm("supertrend", 0)
    trend_raw = (ema_stack + st) / 2
    if trend_raw < -0.6:
        trend_dir = -2
    elif trend_raw < -0.2:
        trend_dir = -1
    elif trend_raw <= 0.2:
        trend_dir = 0
    elif trend_raw <= 0.6:
        trend_dir = 1
    else:
        trend_dir = 2
    
    # Momentum zone (RSI)
    rsi = raw("rsi", 50)
    if rsi < 25:
        momentum_zone = -2
    elif rsi < 40:
        momentum_zone = -1
    elif rsi <= 60:
        momentum_zone = 0
    elif rsi <= 75:
        momentum_zone = 1
    else:
        momentum_zone = 2
    
    # Volume state
    vol_ratio = raw("volume_ratio", 1.0)
    if vol_ratio < 0.7:
        volume_state = -1
    elif vol_ratio <= 1.5:
        volume_state = 0
    elif vol_ratio <= 2.5:
        volume_state = 1
    else:
        volume_state = 2
    
    # IV regime
    iv = raw("atm_iv", 15)
    if iv < 12:
        iv_regime = -1
    elif iv <= 20:
        iv_regime = 0
    elif iv <= 28:
        iv_regime = 1
    else:
        iv_regime = 2
    
    # VWAP position
    vwap_dev = raw("vwap_deviation", 0)
    vwap_pos = -1 if vwap_dev < -0.1 else (1 if vwap_dev > 0.1 else 0)
    
    # Session phase
    session_min = raw("session_minutes", 120)
    if session_min <= 15:
        session = 0
    elif session_min <= 120:
        session = 1
    elif session_min <= 240:
        session = 2
    elif session_min <= 330:
        session = 3
    else:
        session = 4
    
    # DTE band
    dte = raw("dte", 5)
    if dte <= 0.5:
        dte_band = 0
    elif dte <= 2:
        dte_band = 1
    elif dte <= 5:
        dte_band = 2
    else:
        dte_band = 3
    
    # FII flow
    fii = raw("fii_flow", 0)
    fii_dir = -1 if fii < -300 else (1 if fii > 300 else 0)
    
    return CategoricalFingerprint(
        gex_regime=gex_regime,
        gex_flip_zone=gex_flip_zone,
        pcr_band=pcr_band,
        adx_band=adx_band,
        trend_dir=trend_dir,
        momentum_zone=momentum_zone,
        volume_state=volume_state,
        iv_regime=iv_regime,
        vwap_pos=vwap_pos,
        session=session,
        dte_band=dte_band,
        fii_dir=fii_dir,
    )
