"""
Tests for the chain snapshot recorder.

The recorder exists because the options factor — 30 of 100 model weight — cannot
be backtested at all: the proxy serves option_chain/gex/fiidii/news as current
values only. These tests pin the properties that make the resulting log
trustworthy: no invented values, no duplicates from retries, and no future
leaking into a past lookup.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from brain.record.chain_recorder import (
    ChainRecorder, ChainSnapshot, RecorderConfig, SnapshotStore, is_market_open,
)

IST = ZoneInfo("Asia/Kolkata")


def ts_at(y=2026, m=8, d=26, hh=12, mm=15, ss=0) -> float:
    return datetime.datetime(y, m, d, hh, mm, ss, tzinfo=IST).timestamp()


# Real payload shapes captured from production.
HEALTHY = {
    "ltp": {"status": True, "data": {"NIFTY": 24207.75}},
    "option_chain": {"status": True, "data": {
        "pcr": 1.28, "avgIV": 14.8, "putIV": 15.6, "callIV": 14.1,
        "netGEX": -42.5, "regime": "Negative Gamma", "flip": 24050.0,
        "callWall": 24300.0, "putWall": 23800.0, "maxPain": 24100.0,
        "totalOI": 9_100_000, "expiry": "01SEP2026", "dte": 5.1}},
    "fiidii": {"status": True, "data": {"fii": 502.63, "dii": 6425.16}},
    "news": {"status": True, "data": {"sentiment": -0.333}},
}
# Exactly what production returns with the chain dark.
CHAIN_DARK = dict(HEALTHY)
CHAIN_DARK["option_chain"] = {
    "error": "Option chain unavailable",
    "triedExpiries": ["01SEP2026", "08SEP2026"]}


def fetcher(table):
    def _f(action, symbol):
        return table.get(action)
    return _f


# ══════════════════════════════════════════════════════════════════════════════
# MARKET HOURS
# ══════════════════════════════════════════════════════════════════════════════

def test_market_hours_gate():
    assert is_market_open(ts_at(hh=12, mm=15)) is True
    assert is_market_open(ts_at(hh=9, mm=15)) is True
    assert is_market_open(ts_at(hh=15, mm=30)) is True
    assert is_market_open(ts_at(hh=9, mm=14)) is False
    assert is_market_open(ts_at(hh=15, mm=31)) is False
    assert is_market_open(ts_at(hh=4, mm=42)) is False
    assert is_market_open(ts_at(2026, 8, 29, 12, 0)) is False  # Saturday


def test_closed_market_records_nothing(tmp_path):
    store = SnapshotStore(tmp_path / "c.db")
    rec = ChainRecorder(store, RecorderConfig(instruments=("NIFTY",)),
                        fetcher=fetcher(HEALTHY))
    assert rec.record_once("NIFTY", as_of=ts_at(hh=4, mm=42)) is None
    assert store.count() == 0
    assert rec.stats.skipped_closed == 1


# ══════════════════════════════════════════════════════════════════════════════
# NO INVENTED VALUES
# ══════════════════════════════════════════════════════════════════════════════

def test_healthy_snapshot_captures_the_options_block(tmp_path):
    store = SnapshotStore(tmp_path / "h.db")
    rec = ChainRecorder(store, RecorderConfig(instruments=("NIFTY",)),
                        fetcher=fetcher(HEALTHY))
    s = rec.record_once("NIFTY", as_of=ts_at())
    assert s is not None
    assert s.spot == pytest.approx(24207.75)
    assert s.pcr == pytest.approx(1.28)
    assert s.net_gex == pytest.approx(-42.5)
    assert s.gex_flip == pytest.approx(24050.0)
    assert s.max_pain == pytest.approx(24100.0)
    # skew derived from the two legs
    assert s.iv_skew == pytest.approx(1.5)
    assert s.coverage == pytest.approx(1.0)
    assert "option_chain" in s.sources_ok
    assert s.sources_failed == ""


def test_dark_chain_stores_nulls_not_zeros(tmp_path):
    """
    A failed lookup must never be persisted as 0 or carried forward. A missing
    input read as a neutral one is the defect this whole effort removes.
    """
    store = SnapshotStore(tmp_path / "d.db")
    rec = ChainRecorder(store, RecorderConfig(instruments=("NIFTY",)),
                        fetcher=fetcher(CHAIN_DARK))
    s = rec.record_once("NIFTY", as_of=ts_at())
    assert s.pcr is None
    assert s.net_gex is None
    assert s.atm_iv is None
    assert s.iv_skew is None
    # observed fields still present
    assert s.spot is not None and s.fii is not None
    assert 0.0 < s.coverage < 1.0
    assert "option_chain:Option chain unavailable" in s.sources_failed

    # and they are SQL NULL on the way back out, not 0.0
    back = store.load("NIFTY")[0]
    assert back.pcr is None and back.net_gex is None


def test_dark_chain_is_still_recorded_during_market_hours(tmp_path):
    """"We looked and it was dark" is an observation worth keeping."""
    store = SnapshotStore(tmp_path / "dr.db")
    rec = ChainRecorder(store, RecorderConfig(instruments=("NIFTY",)),
                        fetcher=fetcher(CHAIN_DARK))
    rec.record_once("NIFTY", as_of=ts_at())
    assert store.count() == 1
    assert rec.stats.by_source_failed["option_chain"] == 1


def test_total_failure_still_yields_a_row_with_zero_coverage(tmp_path):
    store = SnapshotStore(tmp_path / "z.db")
    rec = ChainRecorder(store, RecorderConfig(instruments=("NIFTY",)),
                        fetcher=fetcher({}))
    s = rec.record_once("NIFTY", as_of=ts_at())
    assert s.coverage == 0.0
    assert s.sources_ok == ""
    assert store.count() == 1


# ══════════════════════════════════════════════════════════════════════════════
# IDEMPOTENCY
# ══════════════════════════════════════════════════════════════════════════════

def test_retry_within_the_same_bucket_does_not_duplicate(tmp_path):
    """
    Polling previously inflated the signals table with ~750 duplicate
    "precedents" a day. The (instrument, bucket) unique key prevents that.
    """
    store = SnapshotStore(tmp_path / "i.db")
    rec = ChainRecorder(store, RecorderConfig(instruments=("NIFTY",), cadence_seconds=30),
                        fetcher=fetcher(HEALTHY))
    base = ts_at(hh=12, mm=15, ss=0)
    for offset in (0, 3, 11, 29):          # all inside one 30s bucket
        rec.record_once("NIFTY", as_of=base + offset)
    assert store.count() == 1
    assert rec.stats.written == 1
    assert rec.stats.duplicates == 3

    rec.record_once("NIFTY", as_of=base + 30)   # next bucket
    assert store.count() == 2


def test_two_instruments_do_not_collide(tmp_path):
    store = SnapshotStore(tmp_path / "m.db")
    rec = ChainRecorder(store, RecorderConfig(instruments=("NIFTY", "BANKNIFTY")),
                        fetcher=fetcher(HEALTHY))
    rec.record_all(as_of=ts_at())
    assert store.count() == 2
    assert store.count("NIFTY") == 1


# ══════════════════════════════════════════════════════════════════════════════
# NO LOOK-AHEAD
# ══════════════════════════════════════════════════════════════════════════════

def test_nearest_never_returns_a_future_snapshot(tmp_path):
    """
    Returning a later snapshot would leak the future into a past decision — the
    leakage that makes a backtest look better than the strategy is.
    """
    store = SnapshotStore(tmp_path / "n.db")
    rec = ChainRecorder(store, RecorderConfig(instruments=("NIFTY",), cadence_seconds=30),
                        fetcher=fetcher(HEALTHY))
    base = ts_at(hh=12, mm=0)
    for i in range(6):
        rec.record_once("NIFTY", as_of=base + i * 30)

    got = store.nearest("NIFTY", base + 95)
    assert got is not None and got.ts <= base + 95
    assert got.ts == pytest.approx(base + 90)

    # nothing exists before the first row
    assert store.nearest("NIFTY", base - 60) is None


def test_nearest_rejects_a_stale_snapshot(tmp_path):
    store = SnapshotStore(tmp_path / "s.db")
    rec = ChainRecorder(store, RecorderConfig(instruments=("NIFTY",)),
                        fetcher=fetcher(HEALTHY))
    base = ts_at(hh=12, mm=0)
    rec.record_once("NIFTY", as_of=base)
    assert store.nearest("NIFTY", base + 60, max_age_seconds=120) is not None
    # An hour-old chain is not a description of the current market.
    assert store.nearest("NIFTY", base + 3600, max_age_seconds=120) is None


# ══════════════════════════════════════════════════════════════════════════════
# STORE / REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def test_load_filters_by_window_and_coverage(tmp_path):
    store = SnapshotStore(tmp_path / "f.db")
    base = ts_at(hh=12, mm=0)
    good = ChainRecorder(store, RecorderConfig(instruments=("NIFTY",), cadence_seconds=30),
                         fetcher=fetcher(HEALTHY))
    dark = ChainRecorder(store, RecorderConfig(instruments=("NIFTY",), cadence_seconds=30),
                         fetcher=fetcher(CHAIN_DARK))
    for i in range(4):
        good.record_once("NIFTY", as_of=base + i * 30)
    for i in range(4, 8):
        dark.record_once("NIFTY", as_of=base + i * 30)

    assert len(store.load("NIFTY")) == 8
    assert len(store.load("NIFTY", min_coverage=0.99)) == 4
    assert len(store.load("NIFTY", start=base + 60, end=base + 120)) == 3


def test_summary_reports_readiness_honestly(tmp_path):
    store = SnapshotStore(tmp_path / "r.db")
    rec = ChainRecorder(store, RecorderConfig(instruments=("NIFTY",), cadence_seconds=30),
                        fetcher=fetcher(HEALTHY))
    base = ts_at(hh=10, mm=0)
    for i in range(20):
        rec.record_once("NIFTY", as_of=base + i * 30)
    s = store.summary()
    assert s["total_rows"] == 20
    assert s["instruments"]["NIFTY"]["rows_with_pcr"] == 20
    # 20 rows is nowhere near enough, and it must say so.
    assert s["ready_to_replay_options"] is False


def test_run_loop_is_bounded_and_does_not_sleep_in_tests(tmp_path):
    store = SnapshotStore(tmp_path / "l.db")
    rec = ChainRecorder(store, RecorderConfig(instruments=("NIFTY",)),
                        fetcher=fetcher(HEALTHY))
    slept: list[float] = []
    # Market-hours gate applies, so force it by disabling the skip.
    rec.cfg.skip_when_closed = False
    stats = rec.run(max_iterations=3, sleep=slept.append)
    assert stats.polls == 3
    assert len(slept) == 2          # no sleep after the final iteration


def test_coverage_is_a_fraction_of_tracked_fields():
    s = ChainSnapshot(instrument="NIFTY", ts=0.0, bucket=0)
    assert s.compute_coverage() == 0.0
    s.spot = 1.0
    s.pcr = 1.2
    assert s.compute_coverage() == pytest.approx(2 / len(ChainSnapshot.TRACKED))
