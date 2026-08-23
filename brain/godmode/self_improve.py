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

import copy
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback uses process lock
    fcntl = None


LEARNINGS_PATH = Path(__file__).parent.parent.parent / "data" / "learnings.json"
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


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
    event_id: str = ""  # Idempotency key from the durable learning outbox


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
    # Unbounded durable dedup ledger; presentation history remains capped.
    processed_event_ids: list[str] = field(default_factory=list)
    
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
        lock_key = str(self.path.resolve())
        with _PATH_LOCKS_GUARD:
            self._lock = _PATH_LOCKS.setdefault(lock_key, threading.RLock())
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.learnings = self._load()

    @contextmanager
    def _file_lock(self):
        """Serialize reload/merge/write across runtimes and POSIX processes."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    
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
                l.processed_event_ids = data.get("processed_event_ids", [])
                l.last_updated = data.get("last_updated", 0)
                l.total_trades_analyzed = data.get("total_trades_analyzed", 0)
                return l
            except (json.JSONDecodeError, KeyError):
                pass
        return Learnings()
    
    def _save(self):
        """Persist learnings with an atomic same-directory replacement."""
        self.learnings.last_updated = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self.learnings), indent=2, sort_keys=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    
    def analyze_outcome(
        self,
        regime: str,
        gex_regime: str,
        direction: str,
        confidence: float,
        evidence_factors: list[dict],
        outcome: str,
        vetoes_applied: list[str],
        event_id: str = "",
    ) -> list[LearningRecord]:
        """Analyze and persist one outcome as a single synchronized update."""
        with self._lock:
            with self._file_lock():
                # Merge from the latest durable document before applying this
                # event; instance-local snapshots must never overwrite peers.
                self.learnings = self._load()
                before = copy.deepcopy(self.learnings)
                try:
                    return self._analyze_outcome(
                        regime, gex_regime, direction, confidence,
                        evidence_factors, outcome, vetoes_applied, event_id,
                    )
                except Exception:
                    self.learnings = before
                    raise

    def _analyze_outcome(
        self,
        regime: str,
        gex_regime: str,
        direction: str,
        confidence: float,
        evidence_factors: list[dict],  # [{factor, direction, weight, impact}]
        outcome: str,  # WIN_T1, WIN_T2, WIN_T3, STOP_LOSS, TIME_EXIT
        vetoes_applied: list[str],
        event_id: str = "",
    ) -> list[LearningRecord]:
        """
        Analyze a single trade outcome and generate learnings.
        
        Returns a list of new learnings discovered from this trade.
        """
        new_learnings: list[LearningRecord] = []
        if event_id and event_id in self.learnings.processed_event_ids:
            return new_learnings
        is_win = outcome.startswith("WIN_")
        is_loss = outcome == "STOP_LOSS"
        is_neutral = outcome == "TIME_EXIT"
        is_excluded = outcome == "NO_ENTRY"
        if not is_excluded:
            self.learnings.total_trades_analyzed += 1
        
        # ── Analyze each evidence factor ──────────────────────────────────────
        for ev in evidence_factors:
            if is_neutral or is_excluded:
                continue
            factor = ev.get("factor", "")
            ev_dir = ev.get("direction", "")  # BULLISH or BEARISH
            
            # Did this factor agree with the generated signal?
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
            
            if factor and abs(new_weight - current) > 0.01:
                new_weight = round(new_weight, 4)
                self.learnings.weight_adjustments[key] = new_weight
                new_learnings.append(LearningRecord(
                    timestamp=time.time(),
                    insight_type="WEIGHT_ADJUST",
                    description=f"Adjusted {factor} after {outcome} in {regime}/{gex_regime}",
                    dimension=factor,
                    regime=f"{regime}|{gex_regime}",
                    old_value=current,
                    new_value=new_weight,
                    evidence=f"direction={direction}; factor_direction={ev_dir}; outcome={outcome}",
                    confidence=min(1.0, 0.5 + self.learnings.total_trades_analyzed / 100),
                    event_id=event_id,
                ))
        
        # ── Detect degradation ────────────────────────────────────────────────
        # Pattern-level degradation is computed from PatternDB statistics by the runtime.
        
        # ── Validate vetoes ───────────────────────────────────────────────────
        for veto in vetoes_applied:
            if veto not in self.learnings.veto_accuracy:
                self.learnings.veto_accuracy[veto] = {"triggered": 0, "would_have_won": 0, "would_have_lost": 0}
            
            self.learnings.veto_accuracy[veto]["triggered"] += 1
            # If a veto blocked the trade and it would have WON → veto was wrong
            # If a veto blocked and it would have LOST → veto saved us
            # We don't know "what would have happened" for vetoed trades,
            # but we track the pattern for meta-analysis.
        
        # Every closed trade produces an auditable learning record, even when
        # no factor weight crosses an adjustment threshold.
        new_learnings.append(LearningRecord(
            timestamp=time.time(),
            insight_type="OUTCOME_RECORDED",
            description=f"Recorded {outcome} for {direction or 'UNKNOWN'} signal",
            regime=f"{regime}|{gex_regime}",
            evidence=f"confidence={confidence:.1f}; factors={len(evidence_factors)}; vetoes={len(vetoes_applied)}",
            confidence=1.0,
            event_id=event_id,
        ))

        # Record learnings
        for learning in new_learnings:
            self.learnings.history.append(asdict(learning))
        self.learnings.history = self.learnings.history[-500:]
        if event_id:
            self.learnings.processed_event_ids.append(event_id)
        
        self._save()
        return new_learnings
    
    def _refresh_for_read(self) -> None:
        with self._lock:
            with self._file_lock():
                self.learnings = self._load()

    def get_weight_adjustment(self, regime: str, gex_regime: str, dimension: str) -> float:
        """
        Get the learned weight adjustment for a specific dimension in a specific context.
        Returns a multiplier (1.0 = no change, >1 = boost, <1 = reduce).
        """
        self._refresh_for_read()
        key = f"{regime}|{gex_regime}|{dimension}"
        return self.learnings.weight_adjustments.get(key, 1.0)
    
    def get_degrading_patterns(self) -> list[str]:
        """Get list of patterns that are losing edge."""
        self._refresh_for_read()
        return self.learnings.degrading_patterns
    
    def get_veto_stats(self) -> dict:
        """Get veto accuracy statistics."""
        self._refresh_for_read()
        return self.learnings.veto_accuracy
    
    def get_summary(self) -> dict:
        """Summary of all learnings for dashboard/API."""
        self._refresh_for_read()
        return {
            "total_trades_analyzed": self.learnings.total_trades_analyzed,
            "weight_adjustments_count": len(self.learnings.weight_adjustments),
            "degrading_patterns": self.learnings.degrading_patterns,
            "veto_stats": self.learnings.veto_accuracy,
            "last_updated": self.learnings.last_updated,
            "recent_learnings": self.learnings.history[-5:],
        }
