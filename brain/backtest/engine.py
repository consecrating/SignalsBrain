"""
SignalsBrain — Backtester

The missing foundation. Before this existed there was no way to run the engine
over the past at all: `_build_time_context` read the wall clock, and
`session_minutes` was clamped with `max(0, ...)`, so any replay outside
09:30-15:15 IST clamped to minute 0 and tripped the OPENING_CHAOS hard veto.
A 400-session sweep produced 400 NO_TRADE results and zero signals.

Consequently every performance claim about the engine was an assertion, pattern
memory could never be seeded, and the calibration layer had no data to fit.

What this measures:
  * hit rate with a Wilson interval (not a bare point estimate)
  * expectancy per trade and total return
  * profit factor, max drawdown, Sharpe
  * calibration: Brier score, reliability table, expected calibration error
  * veto attribution, so you can see what actually blocked trades

Deliberate properties:
  * Decisions use only data at or before `as_of`; the forward path is held in a
    separate field and consulted only to resolve an already-open trade.
  * Pattern-memory lookups pass `as_of`, so a trade can never be justified by
    statistics from its own future.
  * P&L is modelled with Black-Scholes on the recorded strike/IV/DTE, never with
    a fixed multiple of the spot move.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..memory.option_pricing import atm_strike, black_scholes, implied_pnl_pct
from ..memory.pattern_db import PatternDB
from ..memory.statistics import wilson_interval, mean_with_ci
from ..reasoning.calibration import CalibrationModel, evaluate, fit_calibration
from ..reasoning.engine import ReasoningEngine
from ..state.state_builder import StateBuilder, is_market_open
from .replay import CandleFeed, Snapshot

DEFAULT_STRIKE_STEP = {
    "NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50,
    "MIDCPNIFTY": 25, "SENSEX": 100,
}


@dataclass
class BacktestConfig:
    confidence_threshold: float = 60.0
    max_signals_per_day: int = 3
    circuit_breaker_stops: int = 2
    stop_atr: float = 1.2
    t1_atr: float = 1.0
    t2_atr: float = 2.0
    t3_atr: float = 3.2
    max_hold_minutes: float = 240.0
    cost_pct: float = 1.0          # round-trip cost on the premium
    risk_free_rate: float = 0.065
    strike_step: Optional[float] = None
    # Bars between evaluations. 1 = every bar.
    step: int = 1
    # Seed pattern memory as the run proceeds, so history accumulates the way it
    # would live. Always combined with the as_of walk-forward guard.
    seed_pattern_memory: bool = True


@dataclass
class Fill:
    """One completed backtest trade."""
    instrument: str
    entry_ts: float
    exit_ts: float
    direction: str
    confidence: float
    probability: float
    entry_spot: float
    exit_spot: float
    strike: float
    opt_type: str
    iv_entry: float
    dte_entry: float
    atr: float
    outcome: str
    pnl_pct: Optional[float]
    minutes_held: float
    move_atr: float
    mfe_atr: float
    mae_atr: float

    @property
    def is_win(self) -> Optional[bool]:
        if self.pnl_pct is None:
            return None
        return self.pnl_pct > 0

    def to_dict(self) -> dict:
        return {
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "direction": self.direction,
            "confidence": round(self.confidence, 1),
            "probability": round(self.probability, 4),
            "outcome": self.outcome,
            "pnl_pct": None if self.pnl_pct is None else round(self.pnl_pct, 2),
            "minutes_held": round(self.minutes_held, 1),
            "move_atr": round(self.move_atr, 3),
            "mfe_atr": round(self.mfe_atr, 3),
            "mae_atr": round(self.mae_atr, 3),
            "strike": self.strike,
            "opt_type": self.opt_type,
        }


@dataclass
class BacktestResult:
    instrument: str = ""
    evaluations: int = 0
    signals: int = 0
    fills: list[Fill] = field(default_factory=list)
    veto_counts: dict[str, int] = field(default_factory=dict)
    no_trade_reasons: dict[str, int] = field(default_factory=dict)
    calibration: Optional[CalibrationModel] = None
    started_at: float = 0.0
    finished_at: float = 0.0

    # ── Derived metrics ──────────────────────────────────────────────────────

    @property
    def scored(self) -> list[Fill]:
        """Fills with a determinable P&L. Unknowns are excluded, not guessed."""
        return [f for f in self.fills if f.pnl_pct is not None]

    def metrics(self) -> dict:
        scored = self.scored
        n = len(scored)
        out: dict = {
            "instrument": self.instrument,
            "evaluations": self.evaluations,
            "signals_emitted": self.signals,
            "fills": len(self.fills),
            "fills_scored": n,
            "fills_unscored": len(self.fills) - n,
            "signal_rate_pct": round(self.signals / self.evaluations * 100, 3) if self.evaluations else 0.0,
            "runtime_seconds": round(self.finished_at - self.started_at, 2),
            "veto_counts": dict(sorted(self.veto_counts.items(), key=lambda kv: -kv[1])),
            "no_trade_reasons": dict(sorted(self.no_trade_reasons.items(), key=lambda kv: -kv[1])),
        }
        if n == 0:
            out["note"] = "no scored fills; nothing can be concluded about edge"
            return out

        wins = [f for f in scored if f.pnl_pct > 0]
        losses = [f for f in scored if f.pnl_pct <= 0]
        pnls = [f.pnl_pct for f in scored]

        hit = wilson_interval(len(wins), n, 0.95)
        out["hit_rate"] = hit.to_dict()
        out["expectancy_pct"] = mean_with_ci(pnls)
        out["total_return_pct"] = round(sum(pnls), 2)

        gross_win = sum(f.pnl_pct for f in wins)
        gross_loss = abs(sum(f.pnl_pct for f in losses))
        out["profit_factor"] = round(gross_win / gross_loss, 3) if gross_loss > 0 else None
        out["avg_win_pct"] = round(gross_win / len(wins), 2) if wins else None
        out["avg_loss_pct"] = round(-gross_loss / len(losses), 2) if losses else None
        out["max_drawdown_pct"] = round(self._max_drawdown(pnls), 2)
        out["sharpe_per_trade"] = self._sharpe(pnls)
        out["avg_minutes_held"] = round(sum(f.minutes_held for f in scored) / n, 1)
        out["outcome_mix"] = self._outcome_mix(scored)
        out["by_confidence_bucket"] = self._by_bucket(scored)

        # Calibration of the emitted probability against the realised outcome.
        probs = [f.probability for f in scored]
        obs = [1 if f.pnl_pct > 0 else 0 for f in scored]
        out["calibration"] = evaluate(probs, obs).to_dict()
        if self.calibration is not None:
            out["calibration_model"] = self.calibration.to_dict()
        return out

    @staticmethod
    def _outcome_mix(fills: list[Fill]) -> dict:
        mix: dict[str, int] = {}
        for f in fills:
            mix[f.outcome] = mix.get(f.outcome, 0) + 1
        return dict(sorted(mix.items(), key=lambda kv: -kv[1]))

    @staticmethod
    def _by_bucket(fills: list[Fill]) -> list[dict]:
        """
        Hit rate by confidence decile.

        This is the test the old scoring could not pass: confidence saturated at
        99 above net_bias 60, so higher confidence carried no extra information.
        A monotone relationship here is what makes the score usable for sizing.
        """
        buckets: dict[int, list[Fill]] = {}
        for f in fills:
            b = min(9, max(0, int(f.confidence // 10)))
            buckets.setdefault(b, []).append(f)
        rows = []
        for b in sorted(buckets):
            grp = buckets[b]
            w = sum(1 for f in grp if f.pnl_pct > 0)
            est = wilson_interval(w, len(grp), 0.95)
            rows.append({
                "confidence": f"{b*10}-{b*10+9}",
                "n": len(grp),
                "hit_rate_pct": round(est.point * 100, 1),
                "ci_lower_pct": round(est.lower * 100, 1),
                "ci_upper_pct": round(est.upper * 100, 1),
                "mean_pnl_pct": round(sum(f.pnl_pct for f in grp) / len(grp), 2),
            })
        return rows

    @staticmethod
    def _max_drawdown(pnls: list[float]) -> float:
        peak = 0.0
        equity = 0.0
        worst = 0.0
        for p in pnls:
            equity += p
            peak = max(peak, equity)
            worst = min(worst, equity - peak)
        return worst

    @staticmethod
    def _sharpe(pnls: list[float]) -> Optional[float]:
        if len(pnls) < 2:
            return None
        sd = statistics.pstdev(pnls)
        if sd == 0:
            return None
        return round((sum(pnls) / len(pnls)) / sd, 4)


class Backtester:
    """
    Replays a feed through the real engine.

    Uses ReasoningEngine and StateBuilder unmodified, so what is measured is the
    production decision path rather than a parallel simulation.
    """

    def __init__(self, config: Optional[BacktestConfig] = None,
                 db_path: Optional[Path] = None):
        self.config = config or BacktestConfig()
        self.db = PatternDB(db_path) if db_path is not None else None
        self.state_builder = StateBuilder()
        self.engine = ReasoningEngine(pattern_db=self.db)

    # ──────────────────────────────────────────────────────────────────────────

    def run(self, feed: CandleFeed,
            progress: Optional[Callable[[int, int], None]] = None) -> BacktestResult:
        cfg = self.config
        res = BacktestResult(instrument=feed.instrument, started_at=time.time())
        step_ms = DEFAULT_STRIKE_STEP.get(feed.instrument.upper())
        strike_step = cfg.strike_step or step_ms or 50

        self.state_builder.reset_history(feed.instrument)

        # Per-session counters so the session-scoped vetoes are exercised.
        day_key: Optional[str] = None
        signals_today = 0
        stops_today = 0

        snapshots = list(feed.iter_snapshots(step=cfg.step))
        total = len(snapshots)

        for idx, snap in enumerate(snapshots):
            if progress and idx % 200 == 0:
                progress(idx, total)

            # Skip anything outside a real session. The engine would veto it
            # anyway (MARKET_CLOSED); skipping keeps the evaluation count honest.
            if not is_market_open(snap.as_of):
                continue

            key = time.strftime("%Y-%m-%d", time.gmtime(snap.as_of))
            if key != day_key:
                day_key = key
                signals_today = 0
                stops_today = 0

            state = self.state_builder.build(
                instrument=feed.instrument,
                candles=snap.candles,
                gex_data=snap.gex_data,
                fii_dii=snap.fii_dii,
                vix=snap.vix,
                htf_candles=snap.htf_candles,
                daily_candles=snap.daily_candles,
                as_of=snap.as_of,
            )
            res.evaluations += 1

            chain = self.engine.reason(
                state,
                session_signals=signals_today,
                session_stops=stops_today,
                confidence_threshold=cfg.confidence_threshold,
                as_of=snap.as_of,
            )

            for v in chain.vetoes:
                name = v.split(":")[0].strip("[] ").split(" ")[-1]
                res.veto_counts[name] = res.veto_counts.get(name, 0) + 1

            if not chain.actionable:
                reason = "veto" if chain.vetoes else "below_threshold"
                res.no_trade_reasons[reason] = res.no_trade_reasons.get(reason, 0) + 1
                continue

            res.signals += 1
            signals_today += 1

            fill = self._resolve(snap, state, chain, strike_step)
            if fill is None:
                continue
            res.fills.append(fill)
            if fill.outcome == "STOP_LOSS":
                stops_today += 1

            if cfg.seed_pattern_memory and self.db is not None:
                self._persist(state, chain, fill)

        res.finished_at = time.time()

        # Fit calibration on the run's own scored fills, chronologically split.
        scored = res.scored
        if scored:
            res.calibration = fit_calibration(
                [f.confidence for f in scored],
                [1 if f.pnl_pct > 0 else 0 for f in scored],
                holdout_fraction=0.25,
                version=1,
                trained_at=time.time(),
            )
        return res

    # ──────────────────────────────────────────────────────────────────────────

    def _resolve(self, snap: Snapshot, state, chain, strike_step: float) -> Optional[Fill]:
        """
        Walk the forward path and resolve the trade.

        The forward bars are the ONLY place the future is consulted, and only
        after the decision has already been made.
        """
        cfg = self.config
        closes = snap.candles["closes"]
        entry_spot = closes[-1]
        atr_dim = state.dimensions.get("atr_pct")
        atr = (atr_dim.raw / 100.0 * entry_spot) if atr_dim and atr_dim.raw else entry_spot * 0.01
        if atr <= 0:
            return None

        iv_dim = state.dimensions.get("atm_iv")
        dte_dim = state.dimensions.get("dte")
        iv = iv_dim.raw if iv_dim else 0.0
        dte = dte_dim.raw if dte_dim else 0.0
        if iv <= 0 or dte <= 0:
            # Without IV and DTE the premium cannot be priced. Recorded as an
            # unscored fill rather than assigned an invented P&L.
            iv = iv or 0.0
            dte = dte or 0.0

        is_buy = chain.direction == "BUY"
        opt_type = "CE" if is_buy else "PE"
        strike = atm_strike(entry_spot, strike_step)
        mul = 1 if is_buy else -1

        stop = entry_spot - mul * cfg.stop_atr * atr
        t1 = entry_spot + mul * cfg.t1_atr * atr
        t2 = entry_spot + mul * cfg.t2_atr * atr
        t3 = entry_spot + mul * cfg.t3_atr * atr

        outcome = "TIME_EXIT"
        exit_spot = entry_spot
        exit_ts = snap.as_of
        t1_hit = t2_hit = False
        mfe = mae = 0.0

        for bar in snap.future_bars:
            minutes = (bar.ts - snap.as_of) / 60.0
            fav = (bar.high - entry_spot) if is_buy else (entry_spot - bar.low)
            adv = (entry_spot - bar.low) if is_buy else (bar.high - entry_spot)
            mfe = max(mfe, fav / atr)
            mae = max(mae, adv / atr)

            # Stop is checked first: within a bar we cannot know the ordering, so
            # assume the adverse excursion happened first. Anything else would
            # flatter the result.
            if (is_buy and bar.low <= stop) or (not is_buy and bar.high >= stop):
                outcome, exit_spot, exit_ts = "STOP_LOSS", stop, bar.ts
                break
            if (is_buy and bar.high >= t3) or (not is_buy and bar.low <= t3):
                outcome, exit_spot, exit_ts = "WIN_T3", t3, bar.ts
                break
            if (is_buy and bar.high >= t2) or (not is_buy and bar.low <= t2):
                t2_hit = True
            if (is_buy and bar.high >= t1) or (not is_buy and bar.low <= t1):
                t1_hit = True
            if t2_hit and ((is_buy and bar.close < t2) or (not is_buy and bar.close > t2)):
                outcome, exit_spot, exit_ts = "WIN_T2", t2, bar.ts
                break
            if t1_hit and abs(bar.close - entry_spot) / atr < 0.1:
                # Gave the move back. SCRATCH, not a win.
                outcome, exit_spot, exit_ts = "SCRATCH", bar.close, bar.ts
                break
            if minutes >= cfg.max_hold_minutes:
                outcome, exit_spot, exit_ts = "TIME_EXIT", bar.close, bar.ts
                break
            exit_spot, exit_ts = bar.close, bar.ts
        else:
            if snap.future_bars:
                outcome = "WIN_T1" if t1_hit else "TIME_EXIT"

        minutes_held = max(0.0, (exit_ts - snap.as_of) / 60.0)

        pnl = None
        if iv > 0 and dte > 0:
            pnl = implied_pnl_pct(
                entry_spot=entry_spot, exit_spot=exit_spot, strike=strike,
                opt_type=opt_type, iv_pct_entry=iv, dte_days_entry=dte,
                minutes_held=minutes_held, rate=cfg.risk_free_rate,
                cost_pct=cfg.cost_pct,
            )
        # A target label with non-positive realised P&L is not a win.
        if pnl is not None and outcome.startswith("WIN") and pnl <= 0:
            outcome = "SCRATCH"

        move_atr = ((exit_spot - entry_spot) * mul) / atr

        prob = chain.confidence / 100.0
        return Fill(
            instrument=feed_instrument(state), entry_ts=snap.as_of, exit_ts=exit_ts,
            direction=chain.direction, confidence=chain.confidence, probability=prob,
            entry_spot=entry_spot, exit_spot=exit_spot, strike=strike, opt_type=opt_type,
            iv_entry=iv, dte_entry=dte, atr=atr, outcome=outcome, pnl_pct=pnl,
            minutes_held=minutes_held, move_atr=move_atr, mfe_atr=mfe, mae_atr=mae,
        )

    def _persist(self, state, chain, fill: Fill):
        """Record the signal and its outcome so later evaluations can recall it."""
        try:
            rec = self.db.record_signal(
                state=state, direction=chain.direction, confidence=chain.confidence,
                entry_spot=fill.entry_spot, entry_premium=0.0, strike=fill.strike,
                opt_type=fill.opt_type, atr=fill.atr, evidence=chain.verdict,
                timestamp=fill.entry_ts,
            )
            sid = rec.id if hasattr(rec, "id") else rec
            self.db.record_outcome(
                signal_id=sid, outcome=fill.outcome, exit_spot=fill.exit_spot,
                exit_premium=0.0, move_atr=fill.move_atr,
                duration_min=fill.minutes_held, pnl_pct=fill.pnl_pct,
                pnl_source="modelled" if fill.pnl_pct is not None else "unavailable",
                mfe_atr=fill.mfe_atr, mae_atr=fill.mae_atr,
            )
        except Exception:
            # Seeding is best-effort; a persistence failure must not silently
            # change the measured result.
            pass


def feed_instrument(state) -> str:
    return getattr(state, "instrument", "")
