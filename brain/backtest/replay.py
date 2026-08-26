"""
SignalsBrain — Historical Replay Feeds

Supplies bar-by-bar market data with explicit timestamps so the engine can be
evaluated over the past. Every bar carries its own `as_of`, which is what makes
deterministic replay possible now that no builder reads the wall clock.
"""

from __future__ import annotations

import csv
import datetime
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class Bar:
    ts: float          # epoch seconds at the bar CLOSE
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Snapshot:
    """Everything the engine needs to evaluate one instant."""
    as_of: float
    candles: dict
    gex_data: Optional[dict] = None
    fii_dii: Optional[dict] = None
    vix: Optional[float] = None
    htf_candles: Optional[dict] = None
    daily_candles: Optional[dict] = None
    # Forward path used only for outcome resolution, never for the decision.
    future_bars: list[Bar] = field(default_factory=list)


class CandleFeed:
    """
    Rolling-window feed over a list of bars.

    `iter_snapshots` yields a lookback window ending at each bar, plus the future
    bars separately. Keeping the future in its own field makes leakage a visible
    mistake rather than an easy accident.
    """

    def __init__(self, instrument: str, bars: list[Bar], lookback: int = 140,
                 warmup: int = 140, future_horizon: int = 60):
        self.instrument = instrument
        self.bars = sorted(bars, key=lambda b: b.ts)
        self.lookback = lookback
        self.warmup = max(warmup, lookback)
        self.future_horizon = future_horizon

    def _window(self, lo: int, hi: int) -> dict:
        w = self.bars[lo:hi]
        return {
            "opens": [b.open for b in w],
            "highs": [b.high for b in w],
            "lows": [b.low for b in w],
            "closes": [b.close for b in w],
            "volumes": [b.volume for b in w],
            "timestamps": [b.ts for b in w],
        }

    def iter_snapshots(self, step: int = 1) -> Iterator[Snapshot]:
        n = len(self.bars)
        for i in range(self.warmup, n, step):
            lo = max(0, i - self.lookback)
            yield Snapshot(
                as_of=self.bars[i - 1].ts,
                candles=self._window(lo, i),
                future_bars=self.bars[i:i + self.future_horizon],
            )


class SyntheticFeed(CandleFeed):
    """
    Reproducible synthetic sessions for testing the harness itself.

    Generates bars only inside 09:15-15:30 IST on weekdays, so replay exercises
    the same session logic that live operation uses.
    """

    def __init__(self, instrument: str = "NIFTY", days: int = 30,
                 bar_minutes: int = 5, start_price: float = 24000.0,
                 seed: int = 7, daily_drift: float = 0.0,
                 vol: float = 0.0025, start_date: Optional[datetime.date] = None,
                 **kwargs):
        rng = random.Random(seed)
        bars: list[Bar] = []
        day = start_date or datetime.date(2026, 1, 5)  # a Monday
        price = start_price
        produced = 0
        while produced < days:
            if day.isoweekday() > 5:
                day += datetime.timedelta(days=1)
                continue
            # Each session gets its own drift so regimes vary across the run.
            session_drift = rng.gauss(daily_drift, 0.0006)
            t = datetime.datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
            end = datetime.datetime(day.year, day.month, day.day, 15, 30, tzinfo=IST)
            while t < end:
                t += datetime.timedelta(minutes=bar_minutes)
                o = price
                price = o * (1 + rng.gauss(session_drift, vol))
                hi = max(o, price) * (1 + abs(rng.gauss(0, vol / 3)))
                lo = min(o, price) * (1 - abs(rng.gauss(0, vol / 3)))
                bars.append(Bar(ts=t.timestamp(), open=o, high=hi, low=lo,
                                close=price, volume=rng.randint(60000, 240000)))
            produced += 1
            day += datetime.timedelta(days=1)
        super().__init__(instrument, bars, **kwargs)
        self._rng = rng

    def iter_snapshots(self, step: int = 1) -> Iterator[Snapshot]:
        """Attach plausible synthetic options/flow context to each snapshot."""
        for snap in super().iter_snapshots(step=step):
            closes = snap.candles["closes"]
            ltp = closes[-1]
            r = random.Random(int(snap.as_of))
            neg = r.random() < 0.5
            snap.gex_data = {
                "pcr": round(r.uniform(0.55, 1.85), 3),
                "regime": "Negative Gamma" if neg else "Positive Gamma",
                "netGEX": round(r.uniform(-90, 90), 2),
                "flip": round(ltp * (1 + r.uniform(-0.008, 0.008)), 2),
                "callWall": round(ltp * 1.012, 2),
                "putWall": round(ltp * 0.988, 2),
                "maxPain": round(ltp * (1 + r.uniform(-0.005, 0.005)), 2),
                "avgIV": round(r.uniform(10.5, 24.0), 2),
                "putIV": round(r.uniform(11.0, 25.0), 2),
                "callIV": round(r.uniform(10.0, 23.0), 2),
            }
            snap.fii_dii = {"fii": round(r.uniform(-2500, 2500), 1),
                            "dii": round(r.uniform(-1500, 1500), 1)}
            snap.vix = round(r.uniform(10.5, 24.0), 2)
            yield snap


def load_csv_feed(path: Path, instrument: str, tz: str = "Asia/Kolkata",
                  ts_col: str = "timestamp", **kwargs) -> CandleFeed:
    """
    Load bars from CSV.

    Accepts either an epoch-seconds column or an ISO-8601 datetime. Naive
    datetimes are interpreted in `tz` rather than assumed to be UTC, because a
    silent 5h30m shift would move every bar into the wrong session.
    """
    zone = ZoneInfo(tz)
    bars: list[Bar] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row[ts_col]
            try:
                ts = float(raw)
            except ValueError:
                dt = datetime.datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=zone)
                ts = dt.timestamp()
            bars.append(Bar(
                ts=ts,
                open=float(row.get("open", row.get("o", 0)) or 0),
                high=float(row.get("high", row.get("h", 0)) or 0),
                low=float(row.get("low", row.get("l", 0)) or 0),
                close=float(row.get("close", row.get("c", 0)) or 0),
                volume=float(row.get("volume", row.get("v", 0)) or 0),
            ))
    return CandleFeed(instrument, bars, **kwargs)
