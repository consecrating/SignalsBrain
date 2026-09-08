"""
SignalsBrain — Market data client for validation runs

Reads through the SAME proxy the live site uses (`/signals/api/proxy.php`), for one
decisive reason: if validation pulled prices from a different vendor than production
trades on, any difference in the two would silently become "alpha". Testing against the
production feed keeps the experiment honest, warts included.

The endpoint that makes premium-level backtesting possible is `option_candles`, which
returns historical OHLC for one exact listed contract from the broker's NFO history.
Spot candles alone cannot answer whether a strategy is profitable, because the thing
being bought is an option and most of an option's path is decay, not direction.

Everything is cached on disk. A 90-day walk-forward touches the same contract hundreds
of times, and the upstream broker API is both rate-limited and slow.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_BASE = os.environ.get(
    "SIGNALS_PROXY_URL", "https://ads.sanctify.co.in/signals/api/proxy.php")
DEFAULT_CACHE = os.environ.get(
    "SIGNALS_CACHE_DIR", os.path.expanduser("~/.cache/signalsbrain-validation"))

STRIKE_STEPS = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50, "MIDCPNIFTY": 25, "SENSEX": 100}
LOT_SIZES = {"NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120, "SENSEX": 20}


@dataclass
class Candles:
    """OHLCV series, oldest first — the same ordering the live engine assumes."""
    symbol: str
    interval: str
    timestamps: list[str] = field(default_factory=list)
    opens: list[float] = field(default_factory=list)
    highs: list[float] = field(default_factory=list)
    lows: list[float] = field(default_factory=list)
    closes: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.closes)

    @property
    def count(self) -> int:
        return len(self.closes)

    def slice(self, start: int, stop: int) -> "Candles":
        return Candles(
            self.symbol, self.interval,
            self.timestamps[start:stop], self.opens[start:stop], self.highs[start:stop],
            self.lows[start:stop], self.closes[start:stop], self.volumes[start:stop],
        )

    def has_real_volume(self) -> bool:
        """False for every NSE index — the spot feed publishes zeros (verified live)."""
        return sum(self.volumes) > 0


class SignalsDataClient:
    """Thin, cached, dependency-free reader for the production proxy."""

    def __init__(self, base_url: str = DEFAULT_BASE, cache_dir: str = DEFAULT_CACHE,
                 timeout: int = 60, ttl_seconds: Optional[int] = None,
                 verbose: bool = False):
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.timeout = timeout
        # Historical bars for a closed session never change, so the default is to cache
        # them forever. Pass a ttl only when pulling data that includes today.
        self.ttl_seconds = ttl_seconds
        self.verbose = verbose
        self.requests_made = 0
        self.cache_hits = 0
        os.makedirs(self.cache_dir, exist_ok=True)

    # ── plumbing ────────────────────────────────────────────────────────────────
    def _cache_path(self, params: dict) -> str:
        key = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"{params.get('action', 'q')}-{key}.json")

    def _get(self, params: dict) -> Optional[dict]:
        path = self._cache_path(params)
        if os.path.exists(path):
            fresh = self.ttl_seconds is None or (time.time() - os.path.getmtime(path)) < self.ttl_seconds
            if fresh:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        self.cache_hits += 1
                        return json.load(fh)
                except (OSError, json.JSONDecodeError):
                    pass  # fall through and refetch

        url = self.base_url + "?" + urllib.parse.urlencode(params)
        try:
            self.requests_made += 1
            req = urllib.request.Request(url, headers={"User-Agent": "SignalsBrain-Validation/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # network/JSON/HTTP all handled the same way
            if self.verbose:
                print(f"[data] FAILED {params.get('action')} {params}: {exc}")
            return None

        if payload.get("status"):
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
            except OSError:
                pass
        return payload

    # ── endpoints ───────────────────────────────────────────────────────────────
    def spot_candles(self, symbol: str, interval: str = "FIFTEEN_MINUTE",
                     days: int = 90) -> Optional[Candles]:
        """Index/stock OHLC. Note index volumes are always zero (NSE publishes none)."""
        to = time.strftime("%Y-%m-%d")
        frm = time.strftime("%Y-%m-%d", time.localtime(time.time() - days * 86400))
        p = self._get({"action": "candles", "symbol": symbol, "interval": interval,
                       "from": frm, "to": to})
        d = (p or {}).get("data") or {}
        if not d.get("count"):
            return None
        return Candles(symbol, interval, d.get("timestamps", []), d.get("opens", []),
                       d.get("highs", []), d.get("lows", []), d.get("closes", []),
                       d.get("volumes", []))

    def option_candles(self, symbol: str, expiry: str, strike: int, opt_type: str,
                       interval: str = "FIFTEEN_MINUTE", days: int = 30) -> Optional[Candles]:
        """
        Historical PREMIUM series for one exact listed contract. This is what makes a
        real options backtest possible: the decay is in the data rather than modelled.
        """
        to = time.strftime("%Y-%m-%d")
        frm = time.strftime("%Y-%m-%d", time.localtime(time.time() - days * 86400))
        p = self._get({"action": "option_candles", "symbol": symbol, "expiry": expiry,
                       "strike": int(strike), "type": opt_type.upper(),
                       "interval": interval, "from": frm, "to": to})
        d = (p or {}).get("data") or {}
        if not d.get("count"):
            return None
        return Candles(f"{symbol}{strike}{opt_type.upper()}", interval,
                       d.get("timestamps", []), d.get("opens", []), d.get("highs", []),
                       d.get("lows", []), d.get("closes", []), d.get("volumes", []))

    def gex(self, symbol: str) -> Optional[dict]:
        """
        Live dealer-positioning + per-strike OI/volume snapshot.

        Deliberately NOT cached long: it is a live-only endpoint with no history, which
        is precisely why option-chain volume cannot yet be backtested and is therefore
        excluded from the engine's directional score.
        """
        p = self._get({"action": "gex", "symbol": symbol, "cb": str(int(time.time()))})
        return (p or {}).get("data")

    def current_expiry(self, symbol: str) -> Optional[str]:
        """Nearest listed expiry as the broker spells it, e.g. '15SEP2026'."""
        g = self.gex(symbol)
        return (g or {}).get("expiry")

    def stats(self) -> dict:
        return {"requests": self.requests_made, "cache_hits": self.cache_hits,
                "cache_dir": self.cache_dir}
