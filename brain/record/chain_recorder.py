"""
SignalsBrain — Chain Snapshot Recorder

The reason this exists
----------------------

The engine's largest single input is options microstructure — PCR, ATM IV, IV
skew, net GEX, the flip level, call/put walls, max pain. That block carries 30 of
100 model weight, and volatility another 8.

None of it can be backtested, because the proxy exposes `option_chain`, `gex`,
`fiidii` and `news` as **current values only**. There is no historical endpoint.
Measured consequence, on real NIFTY 5-minute bars over 298 evaluations:

    state coverage   median 56% of 47 dimensions
    confidence       median 27.2, p90 37.6, MAX 42.9
    gate             50.0
    cleared          0 / 298

So the engine correctly refuses to trade, and its edge has never been measured on
anything except a synthetic random walk. That cannot be fixed retroactively: the
chain state for a given minute is gone the moment the minute passes.

The only remedy is to start persisting it. Every session this is not running is a
session of data that can never be recovered.

Cost
----

One row per scan per instrument, roughly 200 bytes. At a 30s cadence across a
375-minute session that is ~750 rows/instrument/day, ~2 MB/instrument/year. After
one quarter there is enough to replay the options factor against real chain state
instead of leaving it dark.

Design notes
------------

* **Records what it got, and records what it missed.** A failed lookup is stored
  as NULL with a reason, never as a zero or a carried-forward value. A missing
  input must not later read as a neutral one — that is the defect this whole
  effort exists to remove.
* **Idempotent.** Rows are keyed on (instrument, bucket) where bucket is the
  timestamp floored to the cadence, so a retry or an overlapping poller cannot
  manufacture duplicate history. Polling previously inflated the signals table
  with ~750 duplicate "precedents" a day.
* **Append-only.** No updates, no deletes. This is an observation log.
* **Market-hours aware.** Outside the session the chain is legitimately dark, so
  recording is skipped rather than filling the table with expected nulls.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_PROXY = "https://ads.sanctify.co.in/signals/api/proxy.php"

MARKET_OPEN_MIN = 9 * 60 + 15
MARKET_CLOSE_MIN = 15 * 60 + 30

# The host's WAF returns 406 Not Acceptable to the default python-urllib
# User-Agent while serving the identical request to curl.
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def is_market_open(as_of: float) -> bool:
    t = datetime.datetime.fromtimestamp(as_of, IST)
    if t.isoweekday() > 5:
        return False
    m = t.hour * 60 + t.minute
    return MARKET_OPEN_MIN <= m <= MARKET_CLOSE_MIN


# ══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChainSnapshot:
    """
    One observation of options + flow state.

    Every numeric field is Optional. None means "not observed", and is stored as
    SQL NULL so downstream statistics can exclude it rather than treat a
    placeholder as data.
    """
    instrument: str
    ts: float                      # observation time (epoch seconds)
    bucket: int                    # ts floored to the cadence, for idempotency

    spot: Optional[float] = None
    # options microstructure
    pcr: Optional[float] = None
    atm_iv: Optional[float] = None
    put_iv: Optional[float] = None
    call_iv: Optional[float] = None
    iv_skew: Optional[float] = None
    net_gex: Optional[float] = None
    gex_regime: Optional[str] = None
    gex_flip: Optional[float] = None
    call_wall: Optional[float] = None
    put_wall: Optional[float] = None
    max_pain: Optional[float] = None
    total_oi: Optional[float] = None
    expiry: Optional[str] = None
    dte: Optional[float] = None
    # flow / context
    fii: Optional[float] = None
    dii: Optional[float] = None
    vix: Optional[float] = None
    news_sentiment: Optional[float] = None
    # provenance
    sources_ok: str = ""           # csv of actions that returned data
    sources_failed: str = ""       # csv of action:reason
    coverage: float = 0.0          # fraction of tracked fields observed

    # Fields whose presence defines coverage. Chosen because these are the ones
    # the engine's options and flow factors actually consume.
    TRACKED = (
        "spot", "pcr", "atm_iv", "net_gex", "gex_flip",
        "call_wall", "put_wall", "max_pain", "fii", "dii",
    )

    def compute_coverage(self) -> float:
        got = sum(1 for f in self.TRACKED if getattr(self, f) is not None)
        self.coverage = got / len(self.TRACKED)
        return self.coverage

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# STORE
# ══════════════════════════════════════════════════════════════════════════════

class SnapshotStore:
    """Append-only SQLite log of chain observations."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.db_path), timeout=30.0)
        # WAL so a reader (backtest) never blocks the writer (recorder).
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA busy_timeout=30000")
        return c

    def _init(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS chain_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instrument TEXT NOT NULL,
                    ts REAL NOT NULL,
                    bucket INTEGER NOT NULL,
                    spot REAL, pcr REAL, atm_iv REAL, put_iv REAL, call_iv REAL,
                    iv_skew REAL, net_gex REAL, gex_regime TEXT, gex_flip REAL,
                    call_wall REAL, put_wall REAL, max_pain REAL, total_oi REAL,
                    expiry TEXT, dte REAL,
                    fii REAL, dii REAL, vix REAL, news_sentiment REAL,
                    sources_ok TEXT DEFAULT '',
                    sources_failed TEXT DEFAULT '',
                    coverage REAL DEFAULT 0,
                    UNIQUE(instrument, bucket)
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_cs_inst_ts "
                      "ON chain_snapshots(instrument, ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_cs_cov "
                      "ON chain_snapshots(coverage)")
            c.execute("""CREATE TABLE IF NOT EXISTS meta (
                            key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
            c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                      (str(self.SCHEMA_VERSION),))
            c.commit()

    # ──────────────────────────────────────────────────────────────────────────

    def write(self, snap: ChainSnapshot) -> bool:
        """
        Insert one observation. Returns False if the bucket already exists.

        INSERT OR IGNORE on the (instrument, bucket) unique key makes this safe
        to retry and safe to run from two pollers at once.
        """
        snap.compute_coverage()
        cols = ["instrument", "ts", "bucket", "spot", "pcr", "atm_iv", "put_iv",
                "call_iv", "iv_skew", "net_gex", "gex_regime", "gex_flip",
                "call_wall", "put_wall", "max_pain", "total_oi", "expiry", "dte",
                "fii", "dii", "vix", "news_sentiment",
                "sources_ok", "sources_failed", "coverage"]
        vals = [getattr(snap, c) for c in cols]
        with self._conn() as c:
            cur = c.execute(
                f"INSERT OR IGNORE INTO chain_snapshots ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})", vals)
            c.commit()
            return cur.rowcount == 1

    def count(self, instrument: Optional[str] = None) -> int:
        q = "SELECT COUNT(*) FROM chain_snapshots"
        p: list = []
        if instrument:
            q += " WHERE instrument = ?"
            p.append(instrument)
        with self._conn() as c:
            return int(c.execute(q, p).fetchone()[0])

    def load(self, instrument: str, start: Optional[float] = None,
             end: Optional[float] = None,
             min_coverage: float = 0.0) -> list[ChainSnapshot]:
        q = ["instrument = ?"]
        p: list = [instrument]
        if start is not None:
            q.append("ts >= ?"); p.append(float(start))
        if end is not None:
            q.append("ts <= ?"); p.append(float(end))
        if min_coverage > 0:
            q.append("coverage >= ?"); p.append(float(min_coverage))
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                f"SELECT * FROM chain_snapshots WHERE {' AND '.join(q)} "
                f"ORDER BY ts ASC", p).fetchall()
        out = []
        for r in rows:
            d = {k: r[k] for k in r.keys() if k != "id"}
            out.append(ChainSnapshot(**d))
        return out

    def nearest(self, instrument: str, ts: float,
                max_age_seconds: float = 120.0) -> Optional[ChainSnapshot]:
        """
        Most recent observation at or before `ts`, within `max_age_seconds`.

        At-or-before, never after: returning a later snapshot would leak the
        future into a historical decision, which is the leakage that makes a
        backtest look better than the strategy.
        """
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            r = c.execute(
                "SELECT * FROM chain_snapshots WHERE instrument = ? AND ts <= ? "
                "ORDER BY ts DESC LIMIT 1", (instrument, float(ts))).fetchone()
        if r is None:
            return None
        if float(ts) - float(r["ts"]) > max_age_seconds:
            return None
        d = {k: r[k] for k in r.keys() if k != "id"}
        return ChainSnapshot(**d)

    def summary(self) -> dict:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("""
                SELECT instrument, COUNT(*) n, MIN(ts) first_ts, MAX(ts) last_ts,
                       AVG(coverage) avg_cov,
                       SUM(CASE WHEN pcr IS NOT NULL THEN 1 ELSE 0 END) with_pcr,
                       SUM(CASE WHEN net_gex IS NOT NULL THEN 1 ELSE 0 END) with_gex
                FROM chain_snapshots GROUP BY instrument
            """).fetchall()
        out = {"db": str(self.db_path), "instruments": {}}
        total = 0
        for r in rows:
            total += r["n"]
            sessions = None
            if r["first_ts"] and r["last_ts"]:
                d0 = datetime.datetime.fromtimestamp(r["first_ts"], IST).date()
                d1 = datetime.datetime.fromtimestamp(r["last_ts"], IST).date()
                sessions = (d1 - d0).days + 1
            out["instruments"][r["instrument"]] = {
                "rows": r["n"],
                "from": datetime.datetime.fromtimestamp(r["first_ts"], IST).isoformat() if r["first_ts"] else None,
                "to": datetime.datetime.fromtimestamp(r["last_ts"], IST).isoformat() if r["last_ts"] else None,
                "calendar_days_spanned": sessions,
                "avg_coverage": round(r["avg_cov"] or 0, 4),
                "rows_with_pcr": r["with_pcr"],
                "rows_with_gex": r["with_gex"],
            }
        out["total_rows"] = total
        out["approx_bytes"] = total * 200
        # A +/-10 point interval on a win rate needs ~96 completed trades; with
        # far fewer signals than snapshots, a quarter of recording is the
        # realistic threshold before the options factor is replayable.
        out["ready_to_replay_options"] = total >= 15000
        return out


# ══════════════════════════════════════════════════════════════════════════════
# RECORDER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RecorderConfig:
    proxy_url: str = DEFAULT_PROXY
    instruments: tuple[str, ...] = ("NIFTY", "BANKNIFTY")
    cadence_seconds: int = 30
    timeout: float = 30.0
    skip_when_closed: bool = True


@dataclass
class RecorderStats:
    polls: int = 0
    written: int = 0
    duplicates: int = 0
    skipped_closed: int = 0
    errors: int = 0
    by_source_failed: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ChainRecorder:
    """
    Polls the proxy and appends observations.

    `record_once` is the unit of work; `run` loops it on the configured cadence.
    Both are safe to run concurrently with a backtest reading the same file (WAL)
    and safe to run twice (unique bucket key).
    """

    def __init__(self, store: SnapshotStore,
                 config: Optional[RecorderConfig] = None,
                 fetcher: Optional[Callable[[str, str], Optional[dict]]] = None):
        self.store = store
        self.cfg = config or RecorderConfig()
        # Injectable so tests never touch the network.
        self._fetch = fetcher or self._http_get
        self.stats = RecorderStats()

    # ──────────────────────────────────────────────────────────────────────────

    def _http_get(self, action: str, symbol: str) -> Optional[dict]:
        q = urllib.parse.urlencode({"action": action, "symbol": symbol})
        req = urllib.request.Request(
            f"{self.cfg.proxy_url}?{q}",
            headers={"Accept": "application/json, text/plain, */*",
                     "User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _ok(resp) -> bool:
        return bool(resp) and resp.get("status") not in (False, None) and bool(resp.get("data"))

    @staticmethod
    def _num(v) -> Optional[float]:
        try:
            f = float(v)
            return f if f == f else None
        except (TypeError, ValueError):
            return None

    # ──────────────────────────────────────────────────────────────────────────

    def record_once(self, instrument: str,
                    as_of: Optional[float] = None) -> Optional[ChainSnapshot]:
        """
        Take one observation. Returns the snapshot, or None if skipped.

        A row is written even when the chain is unavailable *during market
        hours*, because "we looked and it was dark" is itself an observation worth
        keeping — it is how an outage is later distinguished from a quiet tape.
        """
        ts = time.time() if as_of is None else float(as_of)
        self.stats.polls += 1

        if self.cfg.skip_when_closed and not is_market_open(ts):
            self.stats.skipped_closed += 1
            return None

        bucket = int(ts // self.cfg.cadence_seconds)
        snap = ChainSnapshot(instrument=instrument, ts=ts, bucket=bucket)
        ok: list[str] = []
        failed: list[str] = []

        # spot
        q = self._fetch("ltp", instrument)
        if self._ok(q):
            d = q["data"]
            snap.spot = self._num(d.get(instrument) if isinstance(d, dict) else None)
            ok.append("ltp")
        else:
            failed.append("ltp:no_data")

        # option chain -> the block that matters
        oc = self._fetch("option_chain", instrument)
        if self._ok(oc):
            d = oc["data"]
            snap.pcr = self._num(d.get("pcr"))
            snap.atm_iv = self._num(d.get("avgIV", d.get("atmIV")))
            snap.put_iv = self._num(d.get("putIV", d.get("avgPutIV")))
            snap.call_iv = self._num(d.get("callIV", d.get("avgCallIV")))
            if snap.put_iv is not None and snap.call_iv is not None:
                snap.iv_skew = snap.put_iv - snap.call_iv
            snap.net_gex = self._num(d.get("netGEX"))
            snap.gex_regime = d.get("regime") or None
            snap.gex_flip = self._num(d.get("flip"))
            snap.call_wall = self._num(d.get("callWall"))
            snap.put_wall = self._num(d.get("putWall"))
            snap.max_pain = self._num(d.get("maxPain"))
            snap.total_oi = self._num(d.get("totalOI", d.get("total_oi")))
            snap.expiry = d.get("expiry") or d.get("expiryDisplay") or None
            snap.dte = self._num(d.get("dte"))
            ok.append("option_chain")
        else:
            reason = "unavailable"
            if isinstance(oc, dict) and oc.get("error"):
                reason = str(oc["error"])[:40].replace(",", ";")
            failed.append(f"option_chain:{reason}")
            self.stats.by_source_failed["option_chain"] = \
                self.stats.by_source_failed.get("option_chain", 0) + 1

        # flow
        fd = self._fetch("fiidii", instrument)
        if self._ok(fd):
            d = fd["data"]
            snap.fii = self._num(d.get("fii"))
            snap.dii = self._num(d.get("dii"))
            ok.append("fiidii")
        else:
            failed.append("fiidii:no_data")

        # news sentiment
        nw = self._fetch("news", instrument)
        if self._ok(nw):
            snap.news_sentiment = self._num((nw["data"] or {}).get("sentiment"))
            ok.append("news")
        else:
            failed.append("news:no_data")

        snap.sources_ok = ",".join(ok)
        snap.sources_failed = ",".join(failed)

        if self.store.write(snap):
            self.stats.written += 1
        else:
            self.stats.duplicates += 1
        return snap

    def record_all(self, as_of: Optional[float] = None) -> list[ChainSnapshot]:
        out = []
        for inst in self.cfg.instruments:
            try:
                s = self.record_once(inst, as_of=as_of)
                if s is not None:
                    out.append(s)
            except Exception:
                self.stats.errors += 1
        return out

    def run(self, max_iterations: Optional[int] = None,
            sleep: Optional[Callable[[float], None]] = None) -> RecorderStats:
        """
        Loop on the configured cadence.

        `max_iterations` bounds the loop for tests and one-shot runs; leave it
        None for a long-lived process (systemd, cron with a wrapper, or a
        supervised worker).
        """
        _sleep = sleep or time.sleep
        i = 0
        while max_iterations is None or i < max_iterations:
            started = time.time()
            self.record_all()
            i += 1
            if max_iterations is not None and i >= max_iterations:
                break
            drift = time.time() - started
            _sleep(max(0.0, self.cfg.cadence_seconds - drift))
        return self.stats
