"""
SignalsBrain — Live Angel One Feed (via the production proxy)

Replaces SyntheticFeed with real NSE bars for backtesting.

    from brain.backtest import Backtester, BacktestConfig
    from brain.backtest.live_feed import ProxyFeed

    feed = ProxyFeed("NIFTY", interval="FIVE_MINUTE")
    result = Backtester(BacktestConfig()).run(feed)


Two constraints this module refuses to paper over
-------------------------------------------------

**1. Only candles are historical. Everything else is a live snapshot.**

The proxy exposes `option_chain`, `gex`, `fiidii` and `news` as *current* values
only — there is no historical endpoint for any of them. Attaching today's PCR, IV
or FII flow to a bar from nine days ago would be look-ahead bias of the worst
kind: the engine would be scoring the past using information from the future and
the backtest would look far better than the strategy is.

So this feed supplies candles and nothing else. `gex_data`, `fii_dii`, `news` and
`vix` are left as None. The consequence is visible rather than silent: the options
microstructure factor (GEX, PCR, IV, IV skew, max pain, walls) is dark for the
whole run, so any result here measures **only** the price / trend / momentum
factors. That is a real limitation of the available data, not of the engine.

To backtest the options factor, chain snapshots have to be *persisted going
forward* — see `snapshot_hint()`.

**2. The proxy caps history.**

Measured against production:

    interval=ONE_MINUTE      2166 bars   ~6 trading days
    interval=FIVE_MINUTE      438 bars   ~7 trading days
    interval=FIFTEEN_MINUTE   225 bars   ~9 trading days  (2026-08-14 -> 08-26)
    interval=ONE_DAY            5 bars

`days`, `from`, `to` and `count` are accepted but ignored — the window is fixed
server-side. Angel One's own `getCandleData` does take `fromdate`/`todate`, so the
ceiling is in `proxy.php`, not upstream. Until that is passed through, a run here
covers roughly one to two weeks, which is not enough for a conclusion about edge.
`ProxyFeed.coverage_note()` states the shortfall explicitly so it cannot be
mistaken for a verdict.
"""

from __future__ import annotations

import datetime
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator, Optional
from zoneinfo import ZoneInfo

from .replay import Bar, CandleFeed, Snapshot

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_PROXY = "https://ads.sanctify.co.in/signals/api/proxy.php"

# Bars per trading session, by interval (09:15-15:30 = 375 minutes).
BARS_PER_SESSION = {
    "ONE_MINUTE": 375,
    "THREE_MINUTE": 125,
    "FIVE_MINUTE": 75,
    "TEN_MINUTE": 38,
    "FIFTEEN_MINUTE": 25,
    "THIRTY_MINUTE": 13,
    "ONE_HOUR": 7,
    "ONE_DAY": 1,
}


class ProxyFetchError(RuntimeError):
    pass


def fetch_candles(symbol: str, interval: str = "FIFTEEN_MINUTE",
                  proxy_url: str = DEFAULT_PROXY,
                  timeout: float = 45.0) -> dict:
    """GET action=candles and return the payload's `data` object."""
    q = urllib.parse.urlencode({"action": "candles", "symbol": symbol,
                                "interval": interval})
    url = f"{proxy_url}?{q}"
    # The host's WAF returns 406 Not Acceptable to the default python-urllib
    # User-Agent while serving the identical request to curl. Send a conventional
    # UA and an explicit Accept so the request is not filtered.
    req = urllib.request.Request(url, headers={
        "Accept": "application/json, text/plain, */*",
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en-IN,en;q=0.9",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise ProxyFetchError(f"candles fetch failed for {symbol}: {e}") from e

    if not payload.get("status"):
        raise ProxyFetchError(
            f"proxy returned status=false for {symbol}: "
            f"{str(payload)[:200]}")
    data = payload.get("data") or {}
    for k in ("opens", "highs", "lows", "closes", "timestamps"):
        if not data.get(k):
            raise ProxyFetchError(f"candles payload missing '{k}'")
    return data


def _parse_ts(raw) -> float:
    """
    Angel One returns ISO-8601 with an explicit +05:30 offset.

    A naive value is interpreted as IST rather than UTC — assuming UTC would
    silently shift every bar 5h30m and move it into the wrong session, which
    would corrupt the session-anchored VWAP and every clock-gated veto.
    """
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().replace("Z", "+00:00")
    dt = datetime.datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.timestamp()


def bars_from_payload(data: dict) -> list[Bar]:
    o, h, l, c = data["opens"], data["highs"], data["lows"], data["closes"]
    v = data.get("volumes") or [0.0] * len(c)
    ts = data["timestamps"]
    n = min(len(o), len(h), len(l), len(c), len(ts))
    out: list[Bar] = []
    for i in range(n):
        try:
            out.append(Bar(
                ts=_parse_ts(ts[i]),
                open=float(o[i]), high=float(h[i]),
                low=float(l[i]), close=float(c[i]),
                volume=float(v[i]) if i < len(v) else 0.0,
            ))
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda b: b.ts)
    return out


class ProxyFeed(CandleFeed):
    """
    Real NSE bars from the production proxy.

    Candles only, by design. See the module docstring: no historical option
    chain, FII/DII or news exists, so attaching current values to past bars would
    be look-ahead bias. `iter_snapshots` therefore leaves gex_data / fii_dii /
    news / vix as None, and the engine's coverage accounting makes the missing
    options factor explicit rather than scoring it as neutral.
    """

    def __init__(self, symbol: str = "NIFTY", interval: str = "FIFTEEN_MINUTE",
                 proxy_url: str = DEFAULT_PROXY,
                 lookback: int = 140, warmup: Optional[int] = None,
                 future_horizon: int = 48, payload: Optional[dict] = None):
        self.interval = interval
        self.proxy_url = proxy_url
        self.payload = payload if payload is not None else fetch_candles(
            symbol, interval, proxy_url)
        bars = bars_from_payload(self.payload)
        if len(bars) < 60:
            raise ProxyFetchError(
                f"only {len(bars)} bars returned for {symbol}/{interval}; "
                f"too few to evaluate")
        # Warmup must clear the longest indicator window (ADX needs 2*14+1).
        if warmup is None:
            warmup = min(max(lookback, 60), max(60, len(bars) // 3))
        super().__init__(symbol, bars, lookback=lookback, warmup=warmup,
                         future_horizon=future_horizon)

    # ──────────────────────────────────────────────────────────────────────────

    @property
    def first_ts(self) -> float:
        return self.bars[0].ts

    @property
    def last_ts(self) -> float:
        return self.bars[-1].ts

    def sessions(self) -> int:
        return len({datetime.datetime.fromtimestamp(b.ts, IST).date()
                    for b in self.bars})

    def coverage_note(self) -> dict:
        """
        State the sample's limits plainly, so a result cannot be mistaken for a
        verdict on edge.
        """
        per = BARS_PER_SESSION.get(self.interval, 25)
        return {
            "symbol": self.instrument,
            "interval": self.interval,
            "bars": len(self.bars),
            "sessions": self.sessions(),
            "from": datetime.datetime.fromtimestamp(self.first_ts, IST).isoformat(),
            "to": datetime.datetime.fromtimestamp(self.last_ts, IST).isoformat(),
            "approx_bars_per_session": per,
            "factors_available": ["price", "trend", "momentum"],
            "factors_dark": ["options", "volatility", "flow", "context"],
            "why_dark": (
                "the proxy exposes option_chain / gex / fiidii / news as current "
                "values only; applying them to past bars would be look-ahead bias"
            ),
            "history_ceiling": (
                "proxy ignores days/from/to/count, so the window is fixed "
                "server-side; Angel One's getCandleData does accept "
                "fromdate/todate, so the limit is in proxy.php"
            ),
            "sufficient_for_edge_claim": False,
            "note": (
                "A +/-10 point confidence interval on a win rate needs roughly 96 "
                "completed trades. A one-to-two week window will not produce that, "
                "so treat any hit rate here as a smoke test of the pipeline."
            ),
        }

    def iter_snapshots(self, step: int = 1) -> Iterator[Snapshot]:
        # Base class already yields candles + the forward path. Nothing is
        # attached here on purpose: there is no historical options / flow / news
        # data to attach, and inventing it is exactly what this module refuses
        # to do.
        for snap in super().iter_snapshots(step=step):
            yield snap


def snapshot_hint() -> str:
    """How to make the options factor backtestable in future."""
    return (
        "To backtest the options microstructure factor, persist a chain snapshot "
        "on every scan during market hours: timestamp, spot, PCR, ATM IV, put/call "
        "IV, net GEX, flip, call/put wall, max pain, total OI. One row per scan per "
        "instrument is roughly 200 bytes; at a 30s cadence that is about 2 MB per "
        "instrument per year. After a quarter there is enough to replay the options "
        "factor against real chain state instead of leaving it dark."
    )
