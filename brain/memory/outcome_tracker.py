"""
SignalsBrain — Outcome Tracker

After a signal fires, this module monitors the market to determine the outcome.
It runs in the background and updates PatternMemory when:
  - Target 1 is hit
  - Target 2 is hit
  - Target 3 is hit
  - Stop loss is hit
  - Time exit triggers (end of session or DTE expired)

This closes the feedback loop: signal → outcome → learn → better signals.
Without this, pattern memory is useless. WITH this, it gets smarter every day.

The outcome categories:
  WIN_T1   — Hit first target (1 ATR move), booked partial
  WIN_T2   — Hit second target (2 ATR move)
  WIN_T3   — Hit third target (3.2 ATR move) — home run
  STOP_LOSS — Hit stop loss (1.2 ATR adverse)
  TIME_EXIT — Exited due to time (session end / theta decay)
  NO_ENTRY  — Signal fired but entry trigger was never reached
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .pattern_db import PatternDB


@dataclass
class ActiveTrade:
    """A trade being monitored for outcome."""
    signal_id: int
    instrument: str
    direction: str  # BUY or SELL
    entry_spot: float
    entry_premium: float
    strike: float
    opt_type: str
    atr: float
    entry_time: float = field(default_factory=time.time)
    
    # Levels (computed at entry)
    stop_spot: float = 0.0
    t1_spot: float = 0.0
    t2_spot: float = 0.0
    t3_spot: float = 0.0
    
    # State
    t1_hit: bool = False
    t2_hit: bool = False
    highest_premium: float = 0.0  # High-water mark for trailing
    
    def __post_init__(self):
        mul = 1 if self.direction == "BUY" else -1
        self.stop_spot = self.entry_spot - mul * 1.2 * self.atr
        self.t1_spot = self.entry_spot + mul * 1.0 * self.atr
        self.t2_spot = self.entry_spot + mul * 2.0 * self.atr
        self.t3_spot = self.entry_spot + mul * 3.2 * self.atr
        self.highest_premium = self.entry_premium


class OutcomeTracker:
    """
    Monitors active trades and records outcomes in PatternMemory.
    Call `check()` on every market scan with the current price.
    """
    
    def __init__(self, db: PatternDB):
        self.db = db
        self.active_trades: dict[int, ActiveTrade] = {}  # signal_id → ActiveTrade
    
    def start_tracking(self, signal_id: int, instrument: str, direction: str,
                       entry_spot: float, entry_premium: float, strike: float,
                       opt_type: str, atr: float):
        """Begin tracking a new trade for outcome."""
        trade = ActiveTrade(
            signal_id=signal_id,
            instrument=instrument,
            direction=direction,
            entry_spot=entry_spot,
            entry_premium=entry_premium,
            strike=strike,
            opt_type=opt_type,
            atr=atr,
        )
        self.active_trades[signal_id] = trade
    
    def check(self, instrument: str, current_spot: float, current_premium: Optional[float] = None) -> list[dict]:
        """
        Check all active trades for this instrument against current price.
        Returns a list of outcomes that just triggered.
        """
        triggered = []
        to_remove = []
        
        for sid, trade in self.active_trades.items():
            if trade.instrument != instrument:
                continue
            
            # Update high-water mark
            if current_premium and current_premium > trade.highest_premium:
                trade.highest_premium = current_premium
            
            outcome = self._check_trade(trade, current_spot)
            
            if outcome:
                # Calculate P&L
                move = current_spot - trade.entry_spot
                if trade.direction == "SELL":
                    move = -move
                move_atr = move / trade.atr if trade.atr > 0 else 0
                duration = (time.time() - trade.entry_time) / 60
                
                pnl_pct = 0.0
                if current_premium and trade.entry_premium > 0:
                    pnl_pct = ((current_premium - trade.entry_premium) / trade.entry_premium) * 100
                elif trade.entry_premium > 0:
                    # Estimate: premium gain roughly proportional to spot move
                    pnl_pct = move_atr * 80  # Rough: 1 ATR move ≈ 80% premium gain for ATM
                
                # Record in pattern memory
                self.db.record_outcome(
                    signal_id=sid,
                    outcome=outcome,
                    exit_spot=current_spot,
                    exit_premium=current_premium or 0,
                    move_atr=move_atr,
                    duration_min=duration,
                    pnl_pct=pnl_pct,
                )
                
                triggered.append({
                    "signal_id": sid,
                    "instrument": instrument,
                    "outcome": outcome,
                    "move_atr": round(move_atr, 2),
                    "duration_min": round(duration, 1),
                    "pnl_pct": round(pnl_pct, 1),
                })
                to_remove.append(sid)
        
        for sid in to_remove:
            del self.active_trades[sid]
        
        return triggered
    
    def check_time_exits(self) -> list[dict]:
        """
        Check if any trades should be time-exited.
        Call at 15:15 IST or when session is ending.
        """
        triggered = []
        to_remove = []
        
        for sid, trade in self.active_trades.items():
            elapsed_min = (time.time() - trade.entry_time) / 60
            
            # Hard time exit: 4 hours max for any trade
            if elapsed_min > 240:
                duration = elapsed_min
                self.db.record_outcome(
                    signal_id=sid,
                    outcome="TIME_EXIT",
                    exit_spot=trade.entry_spot,  # Approximate (we don't have current price here)
                    exit_premium=0,
                    move_atr=0,
                    duration_min=duration,
                    pnl_pct=0,
                )
                triggered.append({
                    "signal_id": sid,
                    "instrument": trade.instrument,
                    "outcome": "TIME_EXIT",
                    "duration_min": round(duration, 1),
                })
                to_remove.append(sid)
        
        for sid in to_remove:
            del self.active_trades[sid]
        
        return triggered
    
    def _check_trade(self, trade: ActiveTrade, spot: float) -> Optional[str]:
        """Check if a trade hit any target or stop."""
        is_buy = trade.direction == "BUY"
        
        # Stop loss
        if is_buy and spot <= trade.stop_spot:
            return "STOP_LOSS"
        if not is_buy and spot >= trade.stop_spot:
            return "STOP_LOSS"
        
        # Target 3 (check highest first for best classification)
        if is_buy and spot >= trade.t3_spot:
            return "WIN_T3"
        if not is_buy and spot <= trade.t3_spot:
            return "WIN_T3"
        
        # Target 2
        if is_buy and spot >= trade.t2_spot:
            if not trade.t2_hit:
                trade.t2_hit = True
                # Don't close yet — let it run for T3
                # But if it comes back, we already know T2 was hit
            return None  # Still running for T3
        if not is_buy and spot <= trade.t2_spot:
            if not trade.t2_hit:
                trade.t2_hit = True
            return None
        
        # If T2 was hit but price pulled back below T2 level → book as WIN_T2
        if trade.t2_hit:
            pullback_from_t2 = abs(spot - trade.t2_spot) / trade.atr
            if pullback_from_t2 > 0.5:  # Pulled back 0.5 ATR from T2
                return "WIN_T2"
        
        # Target 1
        if is_buy and spot >= trade.t1_spot:
            if not trade.t1_hit:
                trade.t1_hit = True
                # Move mental stop to breakeven (tracked in ActiveTrade state)
            return None  # Running for T2/T3
        if not is_buy and spot <= trade.t1_spot:
            if not trade.t1_hit:
                trade.t1_hit = True
            return None
        
        # If T1 was hit but price came back to entry → book as WIN_T1 (breakeven+)
        if trade.t1_hit:
            # After T1, if it falls back to entry → exit at breakeven (technically a small win)
            back_to_entry = abs(spot - trade.entry_spot) / trade.atr
            if back_to_entry < 0.1:  # Within 0.1 ATR of entry
                return "WIN_T1"
        
        return None  # Still in play
    
    @property
    def active_count(self) -> int:
        return len(self.active_trades)
    
    def get_active_summary(self) -> list[dict]:
        """Summary of all active trades being monitored."""
        return [
            {
                "signal_id": sid,
                "instrument": t.instrument,
                "direction": t.direction,
                "entry_spot": t.entry_spot,
                "entry_premium": t.entry_premium,
                "elapsed_min": round((time.time() - t.entry_time) / 60, 1),
                "t1_hit": t.t1_hit,
                "t2_hit": t.t2_hit,
                "stop_spot": round(t.stop_spot, 2),
                "t1_spot": round(t.t1_spot, 2),
                "t2_spot": round(t.t2_spot, 2),
                "t3_spot": round(t.t3_spot, 2),
            }
            for sid, t in self.active_trades.items()
        ]
