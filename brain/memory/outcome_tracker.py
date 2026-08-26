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
from typing import Callable, Optional

from .pattern_db import PatternDB
from .option_pricing import realised_pnl_pct, implied_pnl_pct


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
    
    # Pricing inputs, so P&L can be modelled when no premium quote is observed.
    iv_at_entry: float = 0.0
    dte_at_entry: float = 0.0

    # State
    t1_hit: bool = False
    t2_hit: bool = False
    highest_premium: float = 0.0  # High-water mark for trailing
    # Maximum favourable / adverse excursion, in ATR units. Needed to judge
    # partial-target behaviour instead of only the final label.
    mfe_atr: float = 0.0
    mae_atr: float = 0.0
    
    def __post_init__(self):
        mul = 1 if self.direction == "BUY" else -1
        self.stop_spot = self.entry_spot - mul * 1.2 * self.atr
        self.t1_spot = self.entry_spot + mul * 1.0 * self.atr
        self.t2_spot = self.entry_spot + mul * 2.0 * self.atr
        self.t3_spot = self.entry_spot + mul * 3.2 * self.atr
        if self.highest_premium <= 0:
            self.highest_premium = self.entry_premium


class OutcomeTracker:
    """
    Monitors active trades and records outcomes in PatternMemory.
    Call `check()` on every market scan with the current price.
    """
    
    def __init__(self, db: PatternDB, record_callback: Optional[Callable[..., object]] = None):
        self.db = db
        self.record_callback = record_callback
        self.active_trades: dict[int, ActiveTrade] = {}  # signal_id → ActiveTrade

    def _record_outcome(self, **kwargs):
        recorder = self.record_callback or self.db.record_outcome
        return recorder(**kwargs)
    
    def start_tracking(self, signal_id: int, instrument: str, direction: str,
                       entry_spot: float, entry_premium: float, strike: float,
                       opt_type: str, atr: float, entry_time: Optional[float] = None,
                       t1_hit: bool = False, t2_hit: bool = False,
                       highest_premium: Optional[float] = None,
                       iv_at_entry: float = 0.0, dte_at_entry: float = 0.0):
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
            entry_time=entry_time if entry_time is not None else time.time(),
            iv_at_entry=iv_at_entry,
            dte_at_entry=dte_at_entry,
            t1_hit=t1_hit,
            t2_hit=t2_hit,
            highest_premium=entry_premium if highest_premium is None else highest_premium,
        )
        self.active_trades[signal_id] = trade
        self.db.upsert_active_trade_progress(
            signal_id,
            t1_hit=trade.t1_hit,
            t2_hit=trade.t2_hit,
            highest_premium=trade.highest_premium,
        )
    
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
            
            durable = self.db.get_active_trade_progress(sid)
            if durable:
                trade.t1_hit = trade.t1_hit or durable["t1_hit"]
                trade.t2_hit = trade.t2_hit or durable["t2_hit"]
                trade.highest_premium = max(trade.highest_premium, durable["highest_premium"])
            before_progress = (trade.t1_hit, trade.t2_hit, trade.highest_premium)
            # Update high-water mark
            if current_premium and current_premium > trade.highest_premium:
                trade.highest_premium = current_premium
            
            outcome = self._check_trade(trade, current_spot)
            after_progress = (trade.t1_hit, trade.t2_hit, trade.highest_premium)
            if not outcome and after_progress != before_progress:
                self.db.upsert_active_trade_progress(
                    sid,
                    t1_hit=trade.t1_hit,
                    t2_hit=trade.t2_hit,
                    highest_premium=trade.highest_premium,
                )
            
            if outcome:
                # Calculate P&L
                move = current_spot - trade.entry_spot
                if trade.direction == "SELL":
                    move = -move
                move_atr = move / trade.atr if trade.atr > 0 else 0
                duration = (time.time() - trade.entry_time) / 60
                
                # P&L priority: observed premiums, then a Black-Scholes mark,
                # then None. The old `move_atr * 80` proxy claimed a 1 ATR move
                # was worth +80% and a 3.2 ATR move +256%, ignoring delta,
                # gamma, theta, vega, strike and IV. Because tracking started
                # with entry_premium=0, that proxy was what actually reached the
                # database and became the "realised" P&L in every statistic.
                pnl_pct = realised_pnl_pct(trade.entry_premium, current_premium)
                pnl_source = "observed"
                if pnl_pct is None:
                    pnl_pct = implied_pnl_pct(
                        entry_spot=trade.entry_spot,
                        exit_spot=current_spot,
                        strike=trade.strike or trade.entry_spot,
                        opt_type=trade.opt_type or ("CE" if trade.direction == "BUY" else "PE"),
                        iv_pct_entry=getattr(trade, "iv_at_entry", 0.0) or 0.0,
                        dte_days_entry=getattr(trade, "dte_at_entry", 0.0) or 0.0,
                        minutes_held=duration,
                    )
                    pnl_source = "modelled"
                if pnl_pct is None:
                    # Unknown stays unknown. The row is recorded for audit but
                    # excluded from statistics rather than being invented.
                    pnl_source = "unavailable"

                # Reclassify on realised P&L: touching a level is not banking a
                # win. A non-positive result cannot be a WIN_*.
                if pnl_pct is not None and outcome.startswith("WIN") and pnl_pct <= 0:
                    outcome = "SCRATCH"

                # Record in pattern memory
                self._record_outcome(
                    signal_id=sid,
                    outcome=outcome,
                    exit_spot=current_spot,
                    exit_premium=current_premium or 0,
                    move_atr=move_atr,
                    duration_min=duration,
                    pnl_pct=pnl_pct,
                    pnl_source=pnl_source,
                )
                
                triggered.append({
                    "signal_id": sid,
                    "instrument": instrument,
                    "outcome": outcome,
                    "move_atr": round(move_atr, 2),
                    "duration_min": round(duration, 1),
                    "pnl_pct": None if pnl_pct is None else round(pnl_pct, 1),
                    "pnl_source": pnl_source,
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
                self._record_outcome(
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
        """
        Classify a trade's outcome from the spot path.

        Note on labelling: reaching a target level is not the same as banking a
        win. Previously, price touching T1 and then round-tripping all the way
        back to entry was booked as WIN_T1 — and because P&L was computed at the
        entry-level premium the row scored a small positive number, which
        `get_pattern_stats` then counted as a win. On a real option a 45-minute
        round trip is a theta loss. Every such path inflated the reported win
        rate.

        A round trip to entry is now SCRATCH, and the final win/loss call is made
        on realised premium P&L net of costs, not on spot touching a level.
        """
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
        
        # T1 was reached and price has round-tripped back to entry. This is a
        # SCRATCH, not a win: the spot gain was given back, and the option has
        # meanwhile paid theta for the whole holding period.
        if trade.t1_hit:
            back_to_entry = abs(spot - trade.entry_spot) / trade.atr
            if back_to_entry < 0.1:
                return "SCRATCH"
        
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
