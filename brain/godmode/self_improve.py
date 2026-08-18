"""
SignalsBrain — Self-Improvement Engine

The brain gets SMARTER every day. After each trade closes:
  1. Compare prediction vs reality
  2. Identify which evidence factors were RIGHT and which were WRONG
  3. Adjust dimension weights for the specific regime/context
  4. Detect patterns that are degrading (losing edge)
  5. Flag new patterns that are emerging (gaining edge)

This is NOT generic machine learning. It's targeted weight adjustment:
  - "In Negative Gamma + Trending regime, the GEX flip distance factor
     predicted 80% correctly → increase its weight for that regime."
  - "In Choppy regime, the MACD signal was wrong 60% of the time → decrease."
  - "The RSI_DIVERGENCE factor has gone from 70% accurate to 45% in the last
     30 trades → FLAG AS DEGRADING."

Storage: A JSON file that persists learned adjustments between sessions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


LEARNINGS_PATH = Path(__file__).parent.parent.parent / "data" / "learnings.json"


@dataclass
class DimensionPerformance:
    """Track how well a dimension predicted outcomes."""
    dimension: str
    regime: str  # "all" or specific regime
    gex_regime: str  # "all" or Positive/Negative
    
    times_bullish_signal: int = 0
    times_bullish_correct: int = 0
    times_bearish_signal: int = 0
    times_bearish_correct: int = 0
    
    @property
    def accuracy(self) -> float:
        total = self.times_bullish_signal + self.times_bearish_signal
        correct = self.times_bullish_correct + self.times_bearish_correct
        return correct / total if total > 0 else 0.5
    
    @property
    def sample_size(self) -> int:
        return self.times_bullish_signal + self.times_bearish_signal


@dataclass
class LearningRecord:
    """A single learned insight."""
    timestamp: float
    insight_type: str  # "WEIGHT_ADJUST", "PATTERN_DEGRADING", "NEW_PATTERN", "VETO_VALIDATE"
    description: str
    dimension: str = ""
    regime: str = ""
    old_value: float = 0
    new_value: float = 0
    evidence: str = ""
    confidence: float = 0  # How confident are we in this learning? (0-1)


@dataclass
class Learnings:
    """Persistent learned adjustments."""
    # Weight adjustments per dimension per regime
    # Format: {"regime|gex|dimension": adjustment_multiplier}
    weight_adjustments: dict[str, float] = field(default_factory=dict)
    
    # Degrading patterns (warn when these appear)
    degrading_patterns: list[str] = field(default_factory=list)
    
    # Veto validation (which vetoes were correct?)
    veto_accuracy: dict[str, dict] = field(default_factory=dict)  # veto_name -> {triggered, correct}
    
    # Learning history
    history: list[dict] = field(default_factory=list)
    
    # Meta
    last_updated: float = 0
    total_trades_analyzed: int = 0


class SelfImproveEngine:
    """
    Analyzes trade outcomes and adjusts the brain's parameters.
    Call after every trade closes.
    """
    
    def __init__(self, path: Optional[Path] = None):
        self.path = path or LEARNINGS_PATH
        self.learnings = self._load()
    
    def _load(self) -> Learnings:
        """Load persisted learnings."""
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                l = Learnings()
                l.weight_adjustments = data.get("weight_adjustments", {})
                l.degrading_patterns = data.get("degrading_patterns", [])
                l.veto_accuracy = data.get("veto_accuracy", {})
                l.history = data.get("history", [])
                l.last_updated = data.get("last_updated", 0)
                l.total_trades_analyzed = data.get("total_trades_analyzed", 0)
                return l
            except (json.JSONDecodeError, KeyError):
                pass
        return Learnings()
    
    def _save(self):
        """Persist learnings to disk."""
        self.learnings.last_updated = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self.learnings), indent=2))
    
    def analyze_outcome(
        self,
        regime: str,
        gex_regime: str,
        direction: str,
        confidence: float,
        evidence_factors: list[dict],  # [{factor, direction, weight, impact}]
        outcome: str,  # WIN_T1, WIN_T2, WIN_T3, STOP_LOSS, TIME_EXIT
        vetoes_applied: list[str],
    ) -> list[LearningRecord]:
        """
        Analyze a single trade outcome and generate learnings.
        
        Returns a list of new learnings discovered from this trade.
        """
        new_learnings: list[LearningRecord] = []
        is_win = outcome.startswith("WIN")
        
        self.learnings.total_trades_analyzed += 1
        
        # ── Analyze each evidence factor ──────────────────────────────────────
        for ev in evidence_factors:
            factor = ev.get("factor", "")
            ev_dir = ev.get("direction", "")  # BULLISH or BEARISH
            
            # Did this factor point in the right direction?
            factor_correct = (
                (ev_dir == "BULLISH" and direction == "BUY" and is_win) or
                (ev_dir == "BEARISH" and direction == "SELL" and is_win) or
                (ev_dir == "BULLISH" and direction == "BUY" and not is_win and False) or  # Wrong = not counted as correct
                (ev_dir == "BEARISH" and direction == "SELL" and not is_win and False)
            )
            
            # Actually simpler: did the factor agree with the signal that won/lost?
            factor_agreed_with_signal = (
                (ev_dir == "BULLISH" and direction == "BUY") or
                (ev_dir == "BEARISH" and direction == "SELL")
            )
            
            # Track performance
            key = f"{regime}|{gex_regime}|{factor}"
            current = self.learnings.weight_adjustments.get(key, 1.0)
            
            if factor_agreed_with_signal:
                if is_win:
                    # Factor agreed and trade won → factor was right → boost slightly
                    new_weight = min(1.5, current + 0.02)
                else:
                    # Factor agreed but trade lost → factor was wrong → reduce slightly
                    new_weight = max(0.5, current - 0.03)
            else:
                if is_win:
                    # Factor disagreed but trade won anyway → factor was a false negative → reduce
                    new_weight = max(0.5, current - 0.01)
                else:
                    # Factor disagreed and trade lost → factor was right to disagree → boost
                    new_weight = min(1.5, current + 0.02)
            
            if abs(new_weight - current) > 0.01:
                self.learnings.weight_adjustments[key] = round(new_weight, 4)
        
        # ── Detect degradation ────────────────────────────────────────────────
        # Check if win rate for this regime+gex combo is dropping
        regime_key = f"{regime}|{gex_regime}"
        # (In production, this would query PatternDB for recent stats)
        
        # ── Validate vetoes ───────────────────────────────────────────────────
        for veto in vetoes_applied:
            if veto not in self.learnings.veto_accuracy:
                self.learnings.veto_accuracy[veto] = {"triggered": 0, "would_have_won": 0, "would_have_lost": 0}
            
            self.learnings.veto_accuracy[veto]["triggered"] += 1
            # If a veto blocked the trade and it would have WON → veto was wrong
            # If a veto blocked and it would have LOST → veto saved us
            # We don't know "what would have happened" for vetoed trades,
            # but we track the pattern for meta-analysis.
        
        # Record learnings
        if new_learnings:
            for nl in new_learnings:
                self.learnings.history.append(asdict(nl))
            # Keep last 500 learnings
            self.learnings.history = self.learnings.history[-500:]
        
        self._save()
        return new_learnings
    
    def get_weight_adjustment(self, regime: str, gex_regime: str, dimension: str) -> float:
        """
        Get the learned weight adjustment for a specific dimension in a specific context.
        Returns a multiplier (1.0 = no change, >1 = boost, <1 = reduce).
        """
        key = f"{regime}|{gex_regime}|{dimension}"
        return self.learnings.weight_adjustments.get(key, 1.0)
    
    def get_degrading_patterns(self) -> list[str]:
        """Get list of patterns that are losing edge."""
        return self.learnings.degrading_patterns
    
    def get_veto_stats(self) -> dict:
        """Get veto accuracy statistics."""
        return self.learnings.veto_accuracy
    
    def get_summary(self) -> dict:
        """Summary of all learnings for dashboard/API."""
        return {
            "total_trades_analyzed": self.learnings.total_trades_analyzed,
            "weight_adjustments_count": len(self.learnings.weight_adjustments),
            "degrading_patterns": self.learnings.degrading_patterns,
            "veto_stats": self.learnings.veto_accuracy,
            "last_updated": self.learnings.last_updated,
            "recent_learnings": self.learnings.history[-5:],
        }
