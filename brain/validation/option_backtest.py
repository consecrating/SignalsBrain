"""
SignalsBrain — Option-premium backtester

The point of this module
-----------------------
Every backtest this project had run until now measured SPOT direction: "after a BUY
signal, did the index go up?" That is the wrong target variable. The desk does not buy
the index, it buys a call or a put, and an option's P&L is direction MINUS time decay
minus the spread. A strategy can be right about direction 54% of the time and still
lose steadily, because theta is charged every hour whether the view works or not.

So this backtester replays the ACTUAL historical premium of the exact contract the
engine would have bought, pulled from the broker's NFO history via `option_candles`.
Decay is not modelled or assumed — it is present in the data.

Known limitation — expiry selection
-----------------------------------
`run()` prices every signal against ONE expiry (normally the current near-dated one).
For a signal from several weeks ago the contract actually traded would have been that
week's expiry, which decays faster than a far-dated one. So the further back the window
reaches, the more this UNDERSTATES theta and flatters the strategy. Two consequences
worth stating plainly: keep windows short (roughly within the life of one expiry), and
treat any result as an optimistic bound rather than a neutral estimate. Fixing it
properly requires a historical expiry calendar plus per-expiry premium history, which
the broker endpoint does not expose today.

Execution rules mirror the live desks deliberately, including their known frictions:
    * fill at the live premium (production sets slippage to zero — see the fill fix)
    * stop-loss and targets as premium levels derived from ATR, as buildOptionTrade does
    * intraday only: force square-off at 15:20 IST, never carried overnight
    * one position per instrument at a time
A backtest that is kinder than production would simply be lying, so where production
is pessimistic this stays pessimistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Sequence

from .data import Candles, SignalsDataClient, LOT_SIZES, STRIKE_STEPS
from .engine_port import Signal

# Live desk rules (js/paper-trading.js + js/virtual-trading.js).
EOD_SQUAREOFF_MINUTE = 15 * 60 + 20      # 15:20 IST
SESSION_OPEN_MINUTE = 9 * 60 + 15
STOP_ATR_MULT = 1.2                       # buildOptionTrade stop distance
T1_ATR_MULT = 1.0
T2_ATR_MULT = 2.0
NO_ENTRY_AFTER_MINUTE = 15 * 60 + 15


@dataclass
class BacktestTrade:
    """One completed round trip, priced on real historical premiums."""
    symbol: str
    direction: str
    option_type: str
    strike: int
    expiry: str
    entry_time: str
    exit_time: str
    entry_premium: float
    exit_premium: float
    lot_size: int
    bars_held: int
    outcome: str
    pnl: float = 0.0            # rupees, one lot
    return_pct: float = 0.0     # fraction of premium paid — the series metrics use
    max_favourable: float = 0.0
    max_adverse: float = 0.0
    confidence: float = 0.0
    net_score: float = 0.0
    adx: Optional[float] = None
    regime: str = ""
    entry_spot: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _ist_minute(ts: str) -> Optional[int]:
    """Minutes-since-midnight IST from a broker timestamp ('...T13:15:00+05:30')."""
    try:
        t = ts[11:16]
        h, m = t.split(":")
        return int(h) * 60 + int(m)
    except (ValueError, IndexError):
        return None


def _ist_day(ts: str) -> str:
    return ts[:10]


class OptionBacktester:
    """
    Replays signals against real option premiums.

    One position at a time per instrument, which matches how the live executor behaves
    (per-instrument dedup) and avoids the silent leverage of stacking correlated
    entries — the exact concentration the production correlation guard now prevents.
    """

    def __init__(self, client: SignalsDataClient, interval: str = "FIFTEEN_MINUTE",
                 verbose: bool = False):
        self.client = client
        self.interval = interval
        self.verbose = verbose
        self._chain_cache: dict[tuple, Optional[Candles]] = {}
        self.skipped_no_data = 0
        self.skipped_late = 0

    # ── contract selection ─────────────────────────────────────────────────────
    def _atm_strike(self, symbol: str, spot: float) -> int:
        step = STRIKE_STEPS.get(symbol, 50)
        return int(round(spot / step) * step)

    def _premiums(self, symbol: str, expiry: str, strike: int, opt_type: str,
                  days: int) -> Optional[Candles]:
        key = (symbol, expiry, strike, opt_type, self.interval, days)
        if key not in self._chain_cache:
            self._chain_cache[key] = self.client.option_candles(
                symbol, expiry, strike, opt_type, self.interval, days)
        return self._chain_cache[key]

    @staticmethod
    def _index_of(candles: Candles, timestamp: str) -> Optional[int]:
        """
        Align the spot bar to the option bar by timestamp.

        Exact match only. Fuzzy matching here would quietly introduce look-ahead (or
        fill at a price from a different minute), so an unalignable bar is skipped.
        """
        try:
            return candles.timestamps.index(timestamp)
        except ValueError:
            return None

    # ── single trade simulation ────────────────────────────────────────────────
    def simulate_trade(self, sig: Signal, spot_ts: str, expiry: str,
                       history_days: int = 30) -> Optional[BacktestTrade]:
        """
        Buy one lot of the ATM option the engine points at, then walk forward bar by
        bar on the real premium series until a stop, a target, or the 15:20 square-off.
        """
        if not sig.actionable or not sig.atr:
            return None

        minute = _ist_minute(spot_ts)
        if minute is None:
            return None
        if minute >= NO_ENTRY_AFTER_MINUTE:
            self.skipped_late += 1
            return None

        symbol = sig.symbol
        strike = self._atm_strike(symbol, sig.ltp)
        opt_type = sig.option_type
        chain = self._premiums(symbol, expiry, strike, opt_type, history_days)
        if not chain or chain.count < 3:
            self.skipped_no_data += 1
            return None

        i = self._index_of(chain, spot_ts)
        if i is None or i >= chain.count - 1:
            self.skipped_no_data += 1
            return None

        entry = float(chain.closes[i])
        if entry <= 0:
            self.skipped_no_data += 1
            return None

        # Premium-space stop/target. The engine derives these from spot ATR; converting
        # through a ~0.5 ATM delta is the standard approximation and is applied
        # identically to both stop and targets so the reward:risk ratio is preserved.
        delta = 0.5
        stop_prem = max(0.05, entry - STOP_ATR_MULT * sig.atr * delta)
        t1_prem = entry + T1_ATR_MULT * sig.atr * delta
        t2_prem = entry + T2_ATR_MULT * sig.atr * delta

        entry_day = _ist_day(spot_ts)
        lot = LOT_SIZES.get(symbol, 50)
        peak = trough = entry
        outcome, exit_prem, exit_ts, bars = "OPEN", entry, spot_ts, 0

        for j in range(i + 1, chain.count):
            ts_j = chain.timestamps[j]
            px_hi, px_lo, px_close = float(chain.highs[j]), float(chain.lows[j]), float(chain.closes[j])
            bars = j - i
            peak, trough = max(peak, px_hi), min(trough, px_lo)

            # Intraday: never hold past the session, and never past 15:20.
            m_j = _ist_minute(ts_j)
            if _ist_day(ts_j) != entry_day:
                outcome, exit_prem, exit_ts = "EOD_SQUAREOFF", float(chain.closes[j - 1]), chain.timestamps[j - 1]
                bars = j - 1 - i
                break
            if m_j is not None and m_j >= EOD_SQUAREOFF_MINUTE:
                outcome, exit_prem, exit_ts = "EOD_SQUAREOFF", px_close, ts_j
                break

            # Stop is checked before target: within a bar we cannot know which came
            # first, so the pessimistic assumption is taken rather than the flattering
            # one. Getting this backwards is one of the commonest ways a backtest lies.
            if px_lo <= stop_prem:
                outcome, exit_prem, exit_ts = "STOP_LOSS", stop_prem, ts_j
                break
            if px_hi >= t2_prem:
                outcome, exit_prem, exit_ts = "TARGET_T2", t2_prem, ts_j
                break
            if px_hi >= t1_prem:
                outcome, exit_prem, exit_ts = "TARGET_T1", t1_prem, ts_j
                break
        else:
            outcome, exit_prem, exit_ts = "SERIES_END", float(chain.closes[-1]), chain.timestamps[-1]
            bars = chain.count - 1 - i

        if bars <= 0:
            return None

        pnl = (exit_prem - entry) * lot
        return BacktestTrade(
            symbol=symbol, direction=sig.direction, option_type=opt_type, strike=strike,
            expiry=expiry, entry_time=spot_ts, exit_time=exit_ts,
            entry_premium=round(entry, 2), exit_premium=round(exit_prem, 2),
            lot_size=lot, bars_held=bars, outcome=outcome,
            pnl=round(pnl, 2), return_pct=(exit_prem - entry) / entry,
            max_favourable=(peak - entry) / entry, max_adverse=(trough - entry) / entry,
            confidence=sig.confidence, net_score=sig.net_score, adx=sig.adx,
            regime=sig.regime, entry_spot=sig.ltp,
        )

    # ── portfolio-level replay ─────────────────────────────────────────────────
    def run(self, signals: Sequence[Signal], timestamps: Sequence[str], expiry: str,
            history_days: int = 30) -> list[BacktestTrade]:
        """
        Replay a signal stream, enforcing one open position per instrument.

        `signals[k]` is expected to correspond to `timestamps[k]`. Signals arriving
        while a position is still open are ignored rather than stacked, matching the
        live executor's dedup.
        """
        trades: list[BacktestTrade] = []
        busy_until_index = -1
        for k, sig in enumerate(signals):
            if k <= busy_until_index or not sig.actionable:
                continue
            ts = timestamps[k] if k < len(timestamps) else None
            if not ts:
                continue
            tr = self.simulate_trade(sig, ts, expiry, history_days)
            if tr is None:
                continue
            trades.append(tr)
            busy_until_index = k + tr.bars_held
        return trades

    def diagnostics(self) -> dict:
        return {"skipped_no_option_data": self.skipped_no_data,
                "skipped_too_late_in_session": self.skipped_late,
                "contracts_loaded": len(self._chain_cache),
                **self.client.stats()}
