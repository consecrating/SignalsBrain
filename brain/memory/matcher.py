"""
SignalsBrain — Pattern Matcher

The interface layer between the ReasoningEngine and PatternMemory.
Given a current MarketState, it produces a complete historical context:

1. How many times has this exact setup occurred before?
2. What was the win rate?
3. What was the average move (in ATR units)?
4. How long did it typically take?
5. What's the best/worst case?
6. Is this pattern's performance degrading recently?
7. Are there any WARNING signs from history?

This is the data that makes any AI model's response superhuman.
When Claude/GPT receives:
  "Similar setup occurred 47 times. Win rate: 72%. Avg move: 1.8 ATR in 87 min.
   Recent performance (last 20): 65% — slightly degrading."

...it can give SPECIFIC, DATA-BACKED recommendations instead of generic advice.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from ..state.market_state import MarketState
from .pattern_db import PatternDB, PatternStats
from .fingerprint import build_categorical_fingerprint


@dataclass
class HistoricalContext:
    """
    Complete historical context for a signal being evaluated.
    This is what gets injected into every AI model's prompt.
    """
    # Pattern match summary
    similar_setups: int = 0
    exact_matches: int = 0  # Categorical fingerprint match ≥ 0.75
    
    # Statistics (from PatternStats)
    win_rate: float = 0.0
    avg_move_atr: float = 0.0
    avg_duration_min: float = 0.0
    avg_pnl_pct: float = 0.0
    best_pnl_pct: float = 0.0
    worst_pnl_pct: float = 0.0
    hit_t1_rate: float = 0.0
    hit_t2_rate: float = 0.0
    hit_t3_rate: float = 0.0
    stop_loss_rate: float = 0.0
    
    # Confidence adjustment from history
    confidence_modifier: float = 0.0  # +/- points to add to engine confidence
    confidence_reason: str = ""
    
    # Warnings
    is_degrading: bool = False
    degradation_note: str = ""
    warnings: list = None
    
    # Regime-specific performance
    current_regime_win_rate: float = 0.0
    current_regime_note: str = ""
    
    # Narrative (human-readable summary for AI models)
    narrative: str = ""
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_prompt_context(self) -> str:
        """
        Format as a concise string for injection into AI model prompts.
        Minimizes tokens while preserving all critical information.
        """
        if self.similar_setups == 0:
            return "PATTERN MEMORY: No similar historical setups found (new pattern or insufficient data)."
        
        lines = [
            f"PATTERN MEMORY ({self.similar_setups} similar setups found):",
            f"  Win rate: {self.win_rate:.0f}% | Avg move: {self.avg_move_atr:.1f} ATR in {self.avg_duration_min:.0f} min",
            f"  T1 hit: {self.hit_t1_rate:.0f}% | T2: {self.hit_t2_rate:.0f}% | T3: {self.hit_t3_rate:.0f}% | SL: {self.stop_loss_rate:.0f}%",
            f"  P&L range: best +{self.best_pnl_pct:.0f}%, worst {self.worst_pnl_pct:.0f}%, avg {self.avg_pnl_pct:.0f}%",
        ]
        
        if self.confidence_modifier != 0:
            lines.append(f"  Confidence adjustment: {self.confidence_modifier:+.0f} pts ({self.confidence_reason})")
        
        if self.is_degrading:
            lines.append(f"  ⚠️ DEGRADING: {self.degradation_note}")
        
        for w in self.warnings:
            lines.append(f"  ⚠️ {w}")
        
        if self.current_regime_note:
            lines.append(f"  Regime context: {self.current_regime_note}")
        
        return "\n".join(lines)


class PatternMatcher:
    """
    High-level interface to pattern memory.
    Given a MarketState + direction, produces a complete HistoricalContext.
    """
    
    def __init__(self, db: PatternDB):
        self.db = db
    
    def get_context(self, state: MarketState, direction: str) -> HistoricalContext:
        """
        Get full historical context for a signal being evaluated.
        This is called by the ReasoningEngine for every signal.
        """
        ctx = HistoricalContext()
        
        # Get pattern stats
        stats = self.db.get_pattern_stats(state, direction, instrument=state.instrument)
        
        if stats.total_trades == 0:
            # Try without instrument filter (cross-instrument patterns)
            stats = self.db.get_pattern_stats(state, direction)
        
        if stats.total_trades == 0:
            ctx.narrative = "No historical data for this pattern yet. The brain is learning — outcomes will be recorded."
            return ctx
        
        # Fill stats
        ctx.similar_setups = stats.total_signals
        ctx.win_rate = stats.win_rate
        ctx.avg_move_atr = stats.avg_move_atr
        ctx.avg_duration_min = stats.avg_duration_min
        ctx.avg_pnl_pct = stats.avg_pnl_pct
        ctx.best_pnl_pct = stats.best_pnl_pct
        ctx.worst_pnl_pct = stats.worst_pnl_pct
        ctx.hit_t1_rate = stats.hit_t1_rate
        ctx.hit_t2_rate = stats.hit_t2_rate
        ctx.hit_t3_rate = stats.hit_t3_rate
        ctx.stop_loss_rate = stats.stop_loss_rate
        
        # Count exact matches (relaxed score ≥ 0.75)
        fp = build_categorical_fingerprint(state)
        matches = self.db.find_similar(state, direction, min_match=0.75, limit=100, instrument=state.instrument)
        ctx.exact_matches = len(matches)
        
        # Confidence adjustment based on historical win rate
        ctx.confidence_modifier, ctx.confidence_reason = self._confidence_adjustment(stats)
        
        # Degradation check
        ctx.is_degrading = stats.is_degrading
        if stats.is_degrading:
            ctx.degradation_note = (
                f"Recent win rate ({stats.recent_win_rate:.0f}%) is significantly below "
                f"historical ({stats.win_rate:.0f}%). This pattern may be losing edge."
            )
        
        # Warnings
        ctx.warnings = self._generate_warnings(stats, state)
        
        # Regime-specific context
        regime_perf = self.db.get_regime_performance(state.instrument, days=60)
        current_regime = state.regime.lower() if state.regime else "developing"
        regime_key = {"trending": "trending", "developing": "developing", "choppy": "choppy"}.get(current_regime, "developing")
        
        if regime_key in regime_perf and regime_perf[regime_key]["count"] >= 5:
            rp = regime_perf[regime_key]
            ctx.current_regime_win_rate = rp["win_rate"]
            ctx.current_regime_note = (
                f"In {regime_key} markets (last 60d): {rp['count']} trades, "
                f"{rp['win_rate']:.0f}% win rate, avg {rp['avg_pnl']:.1f}% P&L"
            )
        
        # Build narrative
        ctx.narrative = self._build_narrative(ctx, state, direction)
        
        return ctx
    
    def _confidence_adjustment(self, stats: PatternStats) -> tuple[float, str]:
        """
        Calculate confidence modifier from historical data.
        
        High win rate → boost confidence (pattern has edge)
        Low win rate → reduce confidence (pattern is unreliable)
        Too few samples → no adjustment (insufficient evidence)
        """
        if stats.total_trades < 10:
            return 0.0, "Insufficient history (<10 trades) — no adjustment"
        
        # Baseline: 55% win rate = break-even after costs. Below = negative edge.
        if stats.win_rate >= 75:
            return 8.0, f"Strong historical edge ({stats.win_rate:.0f}% win rate, {stats.total_trades} trades)"
        elif stats.win_rate >= 65:
            return 5.0, f"Good historical edge ({stats.win_rate:.0f}% WR)"
        elif stats.win_rate >= 55:
            return 2.0, f"Mild positive edge ({stats.win_rate:.0f}% WR)"
        elif stats.win_rate >= 45:
            return -3.0, f"Near break-even historically ({stats.win_rate:.0f}% WR) — marginal"
        elif stats.win_rate >= 35:
            return -8.0, f"Below break-even historically ({stats.win_rate:.0f}% WR) — caution"
        else:
            return -15.0, f"LOSING pattern historically ({stats.win_rate:.0f}% WR) — strong avoid"
    
    def _generate_warnings(self, stats: PatternStats, state: MarketState) -> list[str]:
        """Generate specific warnings from historical data."""
        warnings = []
        
        if stats.stop_loss_rate > 40:
            warnings.append(f"High stop-loss rate ({stats.stop_loss_rate:.0f}%) — this setup gets stopped out frequently")
        
        if stats.avg_duration_min > 180 and state.dimensions.get("session_minutes", None):
            session_min = state.dimensions["session_minutes"].raw
            remaining = 375 - session_min  # Minutes till close
            if remaining < stats.avg_duration_min:
                warnings.append(f"Avg duration ({stats.avg_duration_min:.0f} min) exceeds remaining session time ({remaining:.0f} min)")
        
        if stats.worst_pnl_pct < -60:
            warnings.append(f"Worst historical loss was {stats.worst_pnl_pct:.0f}% — extreme downside possible")
        
        if stats.is_degrading:
            warnings.append(f"Pattern degrading: recent WR {stats.recent_win_rate:.0f}% vs historical {stats.win_rate:.0f}%")
        
        return warnings
    
    def _build_narrative(self, ctx: HistoricalContext, state: MarketState, direction: str) -> str:
        """Build a human-readable narrative from the historical context."""
        instrument = state.instrument
        
        if ctx.similar_setups < 5:
            return (
                f"Limited historical data for this {direction} setup on {instrument} "
                f"({ctx.similar_setups} similar instances). Confidence based primarily on "
                f"real-time analysis rather than historical precedent."
            )
        
        quality = "strong" if ctx.win_rate >= 65 else "moderate" if ctx.win_rate >= 50 else "weak"
        
        narrative = (
            f"This {direction} setup on {instrument} has a {quality} historical track record: "
            f"{ctx.win_rate:.0f}% win rate across {ctx.similar_setups} similar instances. "
            f"Average move: {ctx.avg_move_atr:.1f}× ATR in {ctx.avg_duration_min:.0f} minutes. "
            f"T1 hit probability: {ctx.hit_t1_rate:.0f}%, stop-loss rate: {ctx.stop_loss_rate:.0f}%."
        )
        
        if ctx.is_degrading:
            narrative += f" WARNING: Recent performance ({ctx.confidence_reason}) suggests this pattern is losing its edge."
        
        return narrative
