from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain.godmode.multi_model import ModelResponse, MultiModelEngine
from brain.connectors.auth import validate_request, validate_security_configuration
from brain.reasoning.confidence_calc import ConfidenceCalculator
from brain.memory.pattern_db import (
    PatternDB,
    SignalAlreadyClosedError,
    SignalNotFoundError,
)
from brain.runtime import BrainRuntime
from brain.state.dimensions import DIMENSIONS
from brain.state.market_state import MarketState
from brain.state.state_builder import StateBuilder


class FakeChain:
    direction = "BUY"
    confidence = 72.5
    actionable = True
    verdict = "BUY test"
    reasoning_narrative = "Deterministic test chain"
    confidence_breakdown = {"final": 72.5}
    vetoes: list[str] = []
    primary_evidence: list = []
    supporting_evidence: list = []
    counter_arguments: list = []

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "evidence": {"primary": [], "supporting": [], "counter_arguments": []},
            "risk": {"scenarios": []},
            "timing": {"urgency": "NORMAL", "note": "test"},
            "historical": {"similar_setups": 0},
        }

    def to_prompt(self) -> str:
        return "test prompt"


def make_state(instrument: str = "NIFTY") -> MarketState:
    state = MarketState(instrument=instrument, market_open=False)
    state.set_dimension("ltp", 22000, 0)
    state.set_dimension("atr_pct", 1, 0.2)
    state.set_dimension("gex_regime", 1, 1)
    state.set_dimension("ema_stack_score", 1, 1)
    state.set_dimension("supertrend", 1, 1)
    state.set_dimension("adx_value", 30, 0.2)
    state.compute_quality()
    state.compute_composites()
    return state


def test_additive_migration_preserves_legacy_rows_and_is_repeatable(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE signals ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL, "
            "instrument TEXT NOT NULL, direction TEXT NOT NULL, "
            "confidence REAL NOT NULL, net_bias REAL NOT NULL)"
        )
        connection.execute(
            "INSERT INTO signals(timestamp, instrument, direction, confidence, net_bias) "
            "VALUES (1, 'NIFTY', 'BUY', 61, 12)"
        )
        connection.commit()

    db = PatternDB(db_path)
    PatternDB(db_path)

    assert db.count_records() == 1
    assert db.get_signal(1).instrument == "NIFTY"
    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        columns = {row[1] for row in connection.execute("PRAGMA table_info(signals)")}
    assert {
        "schema_metadata", "state_snapshots", "decisions",
        "idempotency_keys", "audit_events",
    } <= tables
    assert {"outcome", "state_vector", "evidence_summary"} <= columns
    assert db.get_schema_metadata() == {"schema_version": "3", "fingerprint_version": "1"}


def test_signal_idempotency_lifecycle_and_atomic_learning(tmp_path: Path):
    runtime = BrainRuntime(
        tmp_path / "patterns.db",
        learnings_path=tmp_path / "learnings.json",
    )
    runtime.state_cache["NIFTY"] = make_state()
    runtime.reasoning_engine.reason = lambda *args, **kwargs: FakeChain()

    first = runtime.create_signal("nifty", idempotency_key="same-request")
    second = runtime.create_signal("NIFTY", idempotency_key="same-request")

    assert first == second
    assert first["signal_id"] is not None
    assert runtime.pattern_db.count_records() == 1
    assert runtime.outcome_tracker.active_count == 1

    # Both retry replay and active monitoring survive process-local runtime state.
    restarted = BrainRuntime(
        tmp_path / "patterns.db",
        learnings_path=tmp_path / "learnings.json",
    )
    assert restarted.outcome_tracker.active_count == 1
    assert restarted.create_signal("NIFTY", idempotency_key="same-request") == first

    restarted.record_outcome(
        signal_id=first["signal_id"],
        outcome="WIN_T1",
        pnl_pct=-5,
    )
    with pytest.raises(SignalAlreadyClosedError):
        restarted.record_outcome(signal_id=first["signal_id"], outcome="STOP_LOSS")
    with pytest.raises(SignalNotFoundError):
        restarted.record_outcome(signal_id=99999, outcome="STOP_LOSS")

    saved = json.loads((tmp_path / "learnings.json").read_text())
    assert saved["total_trades_analyzed"] == 1
    assert saved["history"][-1]["insight_type"] == "OUTCOME_RECORDED"


def test_outcome_stats_use_explicit_semantics_and_chronological_recency(tmp_path: Path):
    db = PatternDB(tmp_path / "patterns.db")
    state = make_state()
    for index in range(45):
        signal_id = db.record_signal(state, "BUY", 70)
        outcome = "WIN_T1" if index < 25 else "STOP_LOSS"
        pnl = -10 if outcome == "WIN_T1" else 10
        db.record_outcome(signal_id, outcome, 0, 0, 0, 0, pnl)
        with db._conn() as connection:
            connection.execute("UPDATE signals SET timestamp = ? WHERE id = ?", (index, signal_id))
            connection.commit()

    stats = db.get_pattern_stats(state, "BUY", instrument="NIFTY")
    assert stats.wins == 25
    assert stats.losses == 20
    assert stats.win_rate == pytest.approx(25 / 45 * 100)
    assert stats.recent_win_rate == 0
    assert stats.is_degrading is True


def test_state_quality_derivatives_and_partial_coverage_penalty():
    candles_one = {
        "opens": [100 + index * 0.1 for index in range(40)],
        "highs": [101 + index * 0.1 for index in range(40)],
        "lows": [99 + index * 0.1 for index in range(40)],
        "closes": [100.5 + index * 0.1 for index in range(40)],
        "volumes": [1000 + index for index in range(40)],
    }
    candles_two = {
        **candles_one,
        "closes": [value + index * 0.2 for index, value in enumerate(candles_one["closes"])],
    }
    builder = StateBuilder()
    with patch("brain.state.state_builder.time.time", side_effect=[100.0, 100.0, 160.0, 160.0]):
        builder.build("NIFTY", candles=candles_one, market_open=False)
        second = builder.build("NIFTY", candles=candles_two, market_open=False)

    quality = second.quality_dict()
    assert quality["declared_count"] == len(DIMENSIONS)
    assert quality["populated_count"] == len(second.dimensions)
    assert 0 < quality["coverage"] < 1
    assert quality["missing_dimensions"]
    assert second.dimensions["day_change_pct"].velocity != 0

    partial = MarketState(instrument="PARTIAL", market_open=False)
    partial.set_dimension("day_change_pct", 1, 1)
    partial.compute_quality()
    partial.compute_composites()
    assert 0 < partial.net_directional_bias < 10
    confidence = ConfidenceCalculator().calculate(
        partial.net_directional_bias,
        partial.agreement_factor,
        [],
        coverage=partial.quality.coverage,
    )
    assert confidence.coverage_modifier < 0


def test_model_parser_and_weighted_confidence_are_unambiguous():
    engine = MultiModelEngine({})
    parsed = engine._parse_model_response(
        "claude",
        "test",
        "DIRECTION: BUY\nCONFIDENCE: 100%\nRISKS: A sell-off is possible.",
        1,
    )
    assert parsed.direction == "BUY"
    assert parsed.confidence == 100

    ambiguous = engine._parse_model_response(
        "claude", "test", "Discussion mentions BUY and SELL but has no direction field.", 1
    )
    assert ambiguous.direction == "UNCERTAIN"

    responses = [
        ModelResponse("claude", "a", "BUY", 100, "buy"),
        ModelResponse("grok", "b", "BUY", 0, "buy"),
    ]
    result = engine._synthesize(responses, "BUY", 50, "Choppy", 1)
    # Choppy weights: Claude 1.15, Grok 0.65, brain 1.5. Weighted mean is 57.6,
    # then unanimous agreement adds five points.
    assert result.weighted_confidence == pytest.approx(62.5757, rel=1e-3)


def test_v2_validation_is_structured_and_v1_routes_remain_present(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SIGNALSBRAIN_ENV", "development")
    import api.main as api_main

    service = BrainRuntime(
        tmp_path / "patterns.db",
        learnings_path=tmp_path / "learnings.json",
    )
    api_main._bind_runtime(service)
    client = TestClient(api_main.app)

    route_paths = {route.path for route in api_main.app.routes}
    assert {
        "/brain/health", "/brain/ingest", "/brain/analyze", "/brain/signal",
        "/brain/ask", "/brain/state/{instrument}", "/brain/history",
        "/brain/outcome", "/brain/dashboard", "/brain/schemas/{model_type}",
    } <= route_paths

    invalid = client.post("/brain/v2/ingest", json={
        "instrument": "NIFTY",
        "candles": {
            "opens": [1, 2], "highs": [2], "lows": [0, 1], "closes": [1, 2]
        },
    })
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "VALIDATION_ERROR"

    candles = {
        "opens": [100 + index for index in range(30)],
        "highs": [101 + index for index in range(30)],
        "lows": [99 + index for index in range(30)],
        "closes": [100.5 + index for index in range(30)],
        "volumes": [1000 + index for index in range(30)],
    }
    valid = client.post("/brain/v2/ingest", json={"instrument": "NIFTY", "candles": candles})
    assert valid.status_code == 200
    assert valid.json()["quality"]["declared_count"] == len(DIMENSIONS)

    bounded = client.post("/brain/v2/signal", json={
        "instrument": "NIFTY", "confidence_threshold": 101
    })
    assert bounded.status_code == 422
    assert bounded.json()["error"]["field"] == "confidence_threshold"



def test_production_auth_rejects_defaults_and_query_credentials(monkeypatch):
    from starlette.requests import Request

    monkeypatch.delenv("SIGNALSBRAIN_ENV", raising=False)
    monkeypatch.delenv("SIGNALSBRAIN_API_KEY", raising=False)
    monkeypatch.delenv("SIGNALSBRAIN_OUTCOME_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SIGNALSBRAIN_API_KEY"):
        validate_security_configuration()

    monkeypatch.setenv("SIGNALSBRAIN_ENV", "development")
    validate_security_configuration()

    monkeypatch.setenv("SIGNALSBRAIN_ENV", "production")
    with pytest.raises(RuntimeError, match="SIGNALSBRAIN_API_KEY"):
        validate_security_configuration()

    monkeypatch.setenv("SIGNALSBRAIN_API_KEY", "production-secret-value")
    with pytest.raises(RuntimeError, match="OUTCOME"):
        validate_security_configuration()
    monkeypatch.setenv("SIGNALSBRAIN_OUTCOME_API_KEY", "production-secret-value")
    with pytest.raises(RuntimeError, match="distinct"):
        validate_security_configuration()
    monkeypatch.setenv("SIGNALSBRAIN_OUTCOME_API_KEY", "outcome-writer-secret")
    validate_security_configuration()

    header_request = Request({
        "type": "http",
        "method": "GET",
        "path": "/brain/dashboard",
        "headers": [(b"x-api-key", b"production-secret-value")],
        "query_string": b"",
    })
    assert validate_request(header_request) is True

    query_request = Request({
        "type": "http",
        "method": "GET",
        "path": "/brain/dashboard",
        "headers": [],
        "query_string": b"api_key=production-secret-value",
    })
    assert validate_request(query_request) is False



def test_failed_learning_write_is_repaired_from_outbox(tmp_path: Path, monkeypatch):
    runtime = BrainRuntime(
        tmp_path / "patterns.db",
        learnings_path=tmp_path / "learnings.json",
    )
    runtime.state_cache["NIFTY"] = make_state()
    runtime.reasoning_engine.reason = lambda *args, **kwargs: FakeChain()
    signal = runtime.create_signal("NIFTY")

    def fail_save():
        raise OSError("simulated learning storage failure")

    original_save = runtime.self_improve._save
    monkeypatch.setattr(runtime.self_improve, "_save", fail_save)
    response = runtime.record_outcome(signal_id=signal["signal_id"], outcome="WIN_T1")
    assert response["ok"] is True

    assert runtime.pattern_db.get_signal(signal["signal_id"]).outcome == "WIN_T1"
    assert runtime.pattern_db.get_pending_learning(signal["signal_id"]) is not None
    with pytest.raises(SignalAlreadyClosedError):
        runtime.record_outcome(signal_id=signal["signal_id"], outcome="WIN_T1")

    monkeypatch.setattr(runtime.self_improve, "_save", original_save)
    runtime.ingest_state(instrument="NIFTY", market_open=False)
    assert runtime.pattern_db.get_pending_learning(signal["signal_id"]) is None
    assert runtime.self_improve.get_summary()["total_trades_analyzed"] == 1

    repaired = BrainRuntime(
        tmp_path / "patterns.db",
        learnings_path=tmp_path / "learnings.json",
    )
    assert repaired.pattern_db.get_pending_learning(signal["signal_id"]) is None
    assert repaired.self_improve.get_summary()["total_trades_analyzed"] == 1


def test_v1_defaults_cli_and_mcp_names_are_stable(tmp_path: Path):
    import tomllib
    import api.main as api_main
    from brain.connectors.schemas import MCP_TOOLS

    assert api_main.SignalRequest.model_fields["confidence_threshold"].default == 60
    assert api_main.AskRequest.model_fields["instrument"].default is None
    assert api_main.HistoryRequest.model_fields["days"].default == 60
    assert api_main.OutcomeRequest.model_fields["exit_spot"].default == 0
    assert api_main.OutcomeRequest.model_fields["exit_premium"].default == 0
    assert api_main.OutcomeRequest.model_fields["pnl_pct"].default == 0
    assert set(BrainRuntime(tmp_path / "contracts.db", learnings_path=tmp_path / "learning.json").health()) == {
        "status", "brain", "pattern_memory_records", "cached_states",
        "active_trades", "time",
    }

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert project["project"]["scripts"]["signalsbrain"] == "api.main:run"
    assert [tool["name"] for tool in MCP_TOOLS] == [
        "signalsbrain_analyze",
        "signalsbrain_signal",
        "signalsbrain_ask",
        "signalsbrain_history",
    ]



def test_automatic_outcome_failure_does_not_poison_future_ingest(tmp_path: Path, monkeypatch):
    runtime = BrainRuntime(
        tmp_path / "patterns.db",
        learnings_path=tmp_path / "learnings.json",
    )
    runtime.state_cache["NIFTY"] = make_state()
    runtime.reasoning_engine.reason = lambda *args, **kwargs: FakeChain()
    signal = runtime.create_signal("NIFTY")
    original_save = runtime.self_improve._save

    monkeypatch.setattr(
        runtime.self_improve,
        "_save",
        lambda: (_ for _ in ()).throw(OSError("temporary learning failure")),
    )
    triggered = runtime.outcome_tracker.check("NIFTY", 23000)
    assert triggered[0]["signal_id"] == signal["signal_id"]
    assert runtime.outcome_tracker.active_count == 0
    assert runtime.pattern_db.get_pending_learning(signal["signal_id"]) is not None

    monkeypatch.setattr(runtime.self_improve, "_save", original_save)
    runtime.ingest_state(instrument="NIFTY", market_open=False)
    assert runtime.pattern_db.get_pending_learning(signal["signal_id"]) is None
    assert runtime.self_improve.get_summary()["total_trades_analyzed"] == 1


def test_streaming_payload_limit_rejects_before_full_buffering():
    import asyncio
    import api.main as api_main

    delivered = bytearray()

    async def consuming_app(scope, receive, send):
        while True:
            message = await receive()
            delivered.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    chunks = iter([
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"5678", "more_body": False},
    ])
    sent = []

    async def receive():
        return next(chunks)

    async def send(message):
        sent.append(message)

    middleware = api_main.BodySizeLimitMiddleware(consuming_app, max_body_bytes=5)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/brain/v2/ingest",
        "headers": [],
    }
    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["status"] == 413
    assert bytes(delivered) == b"1234"
    assert json.loads(sent[1]["body"])["error"]["code"] == "PAYLOAD_TOO_LARGE"



def test_post_commit_outbox_lookup_failure_still_reconciles_tracker(tmp_path: Path, monkeypatch):
    runtime = BrainRuntime(
        tmp_path / "patterns.db",
        learnings_path=tmp_path / "learnings.json",
    )
    runtime.state_cache["NIFTY"] = make_state()
    runtime.reasoning_engine.reason = lambda *args, **kwargs: FakeChain()
    signal = runtime.create_signal("NIFTY")
    original_lookup = runtime.pattern_db.get_pending_learning

    monkeypatch.setattr(
        runtime.pattern_db,
        "get_pending_learning",
        lambda signal_id: (_ for _ in ()).throw(sqlite3.OperationalError("temporary lookup failure")),
    )
    triggered = runtime.outcome_tracker.check("NIFTY", 23000)
    assert triggered[0]["signal_id"] == signal["signal_id"]
    assert runtime.outcome_tracker.active_count == 0

    monkeypatch.setattr(runtime.pattern_db, "get_pending_learning", original_lookup)
    assert runtime.pattern_db.get_pending_learning(signal["signal_id"]) is not None
    runtime.ingest_state(instrument="NIFTY", market_open=False)
    assert runtime.pattern_db.get_pending_learning(signal["signal_id"]) is None


def test_shared_learning_file_merges_independent_runtimes(tmp_path: Path):
    db_path = tmp_path / "patterns.db"
    learning_path = tmp_path / "learnings.json"
    runtime_a = BrainRuntime(db_path, learnings_path=learning_path)
    runtime_b = BrainRuntime(db_path, learnings_path=learning_path)
    for runtime in (runtime_a, runtime_b):
        runtime.state_cache["NIFTY"] = make_state()
        runtime.reasoning_engine.reason = lambda *args, **kwargs: FakeChain()

    signal_a = runtime_a.create_signal("NIFTY")
    signal_b = runtime_b.create_signal("NIFTY")
    runtime_a.record_outcome(signal_id=signal_a["signal_id"], outcome="WIN_T1")
    runtime_b.record_outcome(signal_id=signal_b["signal_id"], outcome="STOP_LOSS")

    durable = json.loads(learning_path.read_text())
    event_ids = {item["event_id"] for item in durable["history"]}
    assert durable["total_trades_analyzed"] == 2
    assert event_ids == {"learning-outbox:1", "learning-outbox:2"}
    assert runtime_a.self_improve.get_summary()["total_trades_analyzed"] == 2



def test_conflicting_outcome_writers_have_one_winner(tmp_path: Path):
    db_path = tmp_path / "patterns.db"
    db = PatternDB(db_path)
    signal_id = db.record_signal(make_state(), "BUY", 70)
    first = PatternDB(db_path)
    second = PatternDB(db_path)
    barrier = threading.Barrier(2)

    def close(database: PatternDB, outcome: str):
        barrier.wait()
        try:
            return database.record_outcome(signal_id, outcome, 0, 0, 0, 0, 0).outcome
        except SignalAlreadyClosedError:
            return "closed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda item: close(*item), [(first, "WIN_T1"), (second, "STOP_LOSS")]))

    assert results.count("closed") == 1
    assert db.get_signal(signal_id).outcome in {"WIN_T1", "STOP_LOSS"}
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM learning_outbox").fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type='outcome_committed'"
        ).fetchone()[0] == 1


def test_concurrent_migrations_are_serialized_and_preserve_rows(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE signals (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL, "
            "instrument TEXT NOT NULL, direction TEXT NOT NULL, confidence REAL NOT NULL, net_bias REAL NOT NULL)"
        )
        connection.execute(
            "INSERT INTO signals(timestamp, instrument, direction, confidence, net_bias) VALUES (1,'NIFTY','BUY',70,1)"
        )
        connection.commit()
    barrier = threading.Barrier(4)

    def migrate(_):
        barrier.wait()
        return PatternDB(db_path).count_records()

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(migrate, range(4))) == [1, 1, 1, 1]
    assert PatternDB(db_path).get_schema_metadata()["schema_version"] == "3"


def test_outcome_tracker_progress_survives_t1_and_t2_restarts(tmp_path: Path):
    db_path = tmp_path / "patterns.db"
    learning_path = tmp_path / "learnings.json"
    runtime = BrainRuntime(db_path, learnings_path=learning_path)
    runtime.state_cache["NIFTY"] = make_state()
    runtime.reasoning_engine.reason = lambda *args, **kwargs: FakeChain()

    t1_signal = runtime.create_signal("NIFTY")
    runtime.outcome_tracker.check("NIFTY", 22230, current_premium=15)
    restarted = BrainRuntime(db_path, learnings_path=learning_path)
    trade = restarted.outcome_tracker.active_trades[t1_signal["signal_id"]]
    assert trade.t1_hit is True
    assert trade.highest_premium == 15
    assert restarted.outcome_tracker.check("NIFTY", 22000)[0]["outcome"] == "WIN_T1"

    restarted.state_cache["NIFTY"] = make_state()
    restarted.reasoning_engine.reason = lambda *args, **kwargs: FakeChain()
    t2_signal = restarted.create_signal("NIFTY")
    restarted.outcome_tracker.check("NIFTY", 22450, current_premium=25)
    restarted_again = BrainRuntime(db_path, learnings_path=learning_path)
    trade = restarted_again.outcome_tracker.active_trades[t2_signal["signal_id"]]
    assert trade.t1_hit is False  # T2 is checked first and is sufficient for classification.
    assert trade.t2_hit is True
    assert trade.highest_premium == 25
    assert restarted_again.outcome_tracker.check("NIFTY", 22300)[0]["outcome"] == "WIN_T2"


def test_snapshot_restore_is_validated_and_decisions_use_absolute_freshness(tmp_path: Path):
    db_path = tmp_path / "patterns.db"
    db = PatternDB(db_path)
    state = make_state()
    state.timestamp = time.time() - 120
    db.record_state_snapshot(state, state.quality_dict())
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO state_snapshots(instrument,timestamp,state_json,quality_json,snapshot_version) "
            "VALUES ('NIFTY', ?, '{bad-json', '{}', 2)",
            (time.time(),),
        )
        connection.commit()

    runtime = BrainRuntime(db_path, decision_max_age_seconds=30)
    assert runtime.state_response("NIFTY")["ok"] is True
    with pytest.raises(Exception, match="fresh state"):
        runtime.analyze("NIFTY")

    future = make_state()
    future.timestamp = time.time() + 60
    db.record_state_snapshot(future, future.quality_dict())
    future_runtime = BrainRuntime(db_path, decision_max_age_seconds=300)
    with pytest.raises(Exception, match="fresh state"):
        future_runtime.create_signal("NIFTY")


def test_idempotent_signal_transaction_includes_audit_and_expires(tmp_path: Path):
    db = PatternDB(tmp_path / "patterns.db", idempotency_ttl_seconds=60)
    runtime = BrainRuntime(pattern_db=db, learnings_path=tmp_path / "learnings.json")
    runtime.state_cache["NIFTY"] = make_state()
    runtime.reasoning_engine.reason = lambda *args, **kwargs: FakeChain()
    first = runtime.create_signal("NIFTY", idempotency_key="retry-key")
    assert runtime.create_signal("NIFTY", idempotency_key="retry-key") == first
    with db._conn() as connection:
        assert connection.execute("SELECT COUNT(*) FROM decisions WHERE operation='signal'").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM audit_events WHERE event_type='signal_created'").fetchone()[0] == 1
        connection.execute("UPDATE idempotency_keys SET expires_at = 0")
        connection.commit()
    second = runtime.create_signal("NIFTY", idempotency_key="retry-key")
    assert second["signal_id"] != first["signal_id"]
    with db._conn() as connection:
        connection.execute(
            "INSERT INTO idempotency_keys(operation,idempotency_key,request_hash,response_json,created_at,expires_at) "
            "VALUES ('other','expired-unrelated','hash','{}',0,0)"
        )
        connection.commit()
    runtime.create_signal("NIFTY", idempotency_key="another-key")
    with db._conn() as connection:
        assert connection.execute("SELECT COUNT(*) FROM decisions WHERE operation='signal'").fetchone()[0] == 3
        assert connection.execute(
            "SELECT COUNT(*) FROM idempotency_keys WHERE idempotency_key='expired-unrelated'"
        ).fetchone()[0] == 0


def test_production_outcome_routes_require_distinct_writer_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SIGNALSBRAIN_ENV", "production")
    monkeypatch.setenv("SIGNALSBRAIN_API_KEY", "general-production-key")
    monkeypatch.setenv("SIGNALSBRAIN_OUTCOME_API_KEY", "outcome-production-key")
    import api.main as api_main

    service = BrainRuntime(tmp_path / "patterns.db", learnings_path=tmp_path / "learnings.json")
    service.state_cache["NIFTY"] = make_state()
    service.reasoning_engine.reason = lambda *args, **kwargs: FakeChain()
    signal = service.create_signal("NIFTY")
    api_main._bind_runtime(service)
    client = TestClient(api_main.app)
    denied = client.post(
        "/brain/outcome",
        headers={"X-API-Key": "general-production-key"},
        json={"signal_id": signal["signal_id"], "outcome": "WIN_T1"},
    )
    assert denied.status_code == 401
    accepted = client.post(
        "/brain/outcome",
        headers={"X-Outcome-API-Key": "outcome-production-key"},
        json={"signal_id": signal["signal_id"], "outcome": "WIN_T1"},
    )
    assert accepted.status_code == 200


def test_v1_models_bound_payloads_but_ignore_extra_fields():
    from pydantic import ValidationError
    from api.main import SignalRequest, StateIngestRequest

    candles = {"opens": [1], "highs": [2], "lows": [0], "closes": [1], "proxy_extra": "kept"}
    parsed = StateIngestRequest.model_validate({"instrument": "NIFTY", "candles": candles, "outer_extra": 1})
    assert parsed.candles["proxy_extra"] == "kept"
    assert "outer_extra" not in parsed.model_dump()
    with pytest.raises(ValidationError):
        StateIngestRequest.model_validate({
            "instrument": "NIFTY",
            "candles": {"opens": [1, 2], "highs": [2], "lows": [0, 1], "closes": [1, 2]},
        })
    with pytest.raises(ValidationError):
        SignalRequest.model_validate({"instrument": "NIFTY", "confidence_threshold": float("inf")})


def test_model_provider_aliases_and_uncertain_votes_abstain():
    engine = MultiModelEngine({})
    responses = [
        ModelResponse("openai", "gpt-4o", "BUY", 100, "buy"),
        ModelResponse("anthropic", "claude-sonnet", "BUY", 0, "buy"),
    ]
    result = engine._synthesize(responses, "BUY", 50, "Trending", 1)
    # Actual config keys resolve to gpt-4o (1.1) and claude (0.95), plus brain (1.5).
    assert result.weighted_confidence == pytest.approx((110 + 0 + 75) / 3.55 + 5, rel=1e-3)

    abstaining = engine._synthesize(
        [
            ModelResponse("openai", "gpt-4o", "UNCERTAIN", 90, "uncertain"),
            ModelResponse("anthropic", "claude", "BUY", 70, "buy"),
        ],
        "BUY", 60, "Trending", 1,
    )
    assert abstaining.direction == "BUY"
    assert abstaining.models_agree == 1
    assert abstaining.models_disagree == 0


def test_outcome_semantics_are_consistent_for_neutral_and_no_entry(tmp_path: Path):
    from brain.godmode.self_improve import SelfImproveEngine

    db = PatternDB(tmp_path / "patterns.db")
    state = make_state()
    for outcome in ("WIN_T1", "STOP_LOSS", "TIME_EXIT", "NO_ENTRY"):
        signal_id = db.record_signal(state, "BUY", 70)
        db.record_outcome(signal_id, outcome, 0, 0, 0, 0, 0)
    stats = db.get_pattern_stats(state, "BUY", instrument="NIFTY")
    assert (stats.wins, stats.losses, stats.total_trades, stats.win_rate) == (1, 1, 3, 50)
    daily = db.get_daily_summary()
    assert daily["trades_taken"] == 3
    assert (daily["wins"], daily["losses"], daily["win_rate"]) == (1, 1, 50.0)
    regime = db.get_regime_performance("NIFTY")
    assert sum(item["count"] for item in regime.values()) == 2

    engine = SelfImproveEngine(tmp_path / "learnings.json")
    evidence = [{"factor": "trend", "direction": "BULLISH"}]
    engine.analyze_outcome("Trending", "Positive", "BUY", 70, evidence, "TIME_EXIT", [], "neutral")
    engine.analyze_outcome("Trending", "Positive", "BUY", 70, evidence, "NO_ENTRY", [], "excluded")
    assert engine.get_weight_adjustment("Trending", "Positive", "trend") == 1.0
    assert engine.get_summary()["total_trades_analyzed"] == 1


def test_recent_pattern_stats_include_low_score_newest_matches(tmp_path: Path):
    db = PatternDB(tmp_path / "patterns.db")
    state = make_state()
    for index in range(230):
        signal_id = db.record_signal(state, "BUY", 70)
        db.record_outcome(signal_id, "WIN_T1", 0, 0, 0, 0, 0)
        with db._conn() as connection:
            connection.execute("UPDATE signals SET timestamp=? WHERE id=?", (index, signal_id))
            connection.commit()
    for index in range(20):
        signal_id = db.record_signal(state, "BUY", 70)
        db.record_outcome(signal_id, "STOP_LOSS", 0, 0, 0, 0, 0)
        with db._conn() as connection:
            connection.execute(
                "UPDATE signals SET timestamp=?, pcr_band=pcr_band+1, momentum_zone=momentum_zone+1 WHERE id=?",
                (1000 + index, signal_id),
            )
            connection.commit()
    assert db.get_pattern_stats(state, "BUY", instrument="NIFTY").recent_win_rate == 0


def test_learning_dedup_ledger_is_not_lost_when_history_is_truncated(tmp_path: Path):
    from brain.godmode.self_improve import SelfImproveEngine

    path = tmp_path / "learnings.json"
    path.write_text(json.dumps({
        "weight_adjustments": {}, "degrading_patterns": [], "veto_accuracy": {},
        "history": [{"event_id": f"event-{index}"} for index in range(2, 502)],
        "processed_event_ids": [f"event-{index}" for index in range(1, 502)],
        "last_updated": 0, "total_trades_analyzed": 501,
    }))
    engine = SelfImproveEngine(path)
    assert engine.analyze_outcome("Trending", "Positive", "BUY", 70, [], "WIN_T1", [], "event-1") == []
    assert engine.get_summary()["total_trades_analyzed"] == 501


def test_mcp_shared_validation_and_generic_errors(tmp_path: Path):
    from brain.connectors.mcp_server import dispatch_mcp_tool
    from brain.connectors.schemas import MCP_TOOLS

    runtime = BrainRuntime(tmp_path / "patterns.db", learnings_path=tmp_path / "learnings.json")
    invalid = dispatch_mcp_tool(runtime, "signalsbrain_signal", {"instrument": "NIFTY", "extra": True})
    assert invalid["error"]["code"] == "INVALID_ARGUMENTS"
    invalid_key = dispatch_mcp_tool(
        runtime,
        "signalsbrain_signal",
        {"instrument": "NIFTY", "idempotency_key": "contains spaces"},
    )
    assert invalid_key["error"]["code"] == "INVALID_ARGUMENTS"
    schema = next(item for item in MCP_TOOLS if item["name"] == "signalsbrain_signal")["inputSchema"]
    assert schema["additionalProperties"] is False
    runtime.state_cache["NIFTY"] = make_state()
    runtime.reasoning_engine.reason = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("secret /tmp/path"))
    failed = dispatch_mcp_tool(runtime, "signalsbrain_signal", {"instrument": "NIFTY"})
    assert failed["error"] == {"code": "RUNTIME_ERROR", "message": "SignalsBrain could not complete the request"}
    assert "secret" not in json.dumps(failed)


def test_cli_is_fail_closed_and_defaults_to_loopback(monkeypatch):
    import api.main as api_main

    monkeypatch.delenv("SIGNALSBRAIN_ENV", raising=False)
    monkeypatch.delenv("SIGNALSBRAIN_API_KEY", raising=False)
    monkeypatch.delenv("SIGNALSBRAIN_OUTCOME_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        api_main.run()

    monkeypatch.setenv("SIGNALSBRAIN_ENV", "development")
    called = {}
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: called.update(kwargs))
    api_main.run()
    assert called["host"] == "127.0.0.1"



def test_active_trade_progress_is_monotonic_across_stale_writers(tmp_path: Path):
    db = PatternDB(tmp_path / "patterns.db")
    signal_id = db.record_signal(make_state(), "BUY", 70, entry_spot=22000, atr=220)
    db.upsert_active_trade_progress(signal_id, t1_hit=True, t2_hit=True, highest_premium=30)
    db.upsert_active_trade_progress(signal_id, t1_hit=True, t2_hit=False, highest_premium=10)
    progress = db.get_active_trade_progress(signal_id)
    assert progress["t1_hit"] is True
    assert progress["t2_hit"] is True
    assert progress["highest_premium"] == 30

    stale_runtime = BrainRuntime(pattern_db=db, learnings_path=tmp_path / "learnings.json")
    stale_trade = stale_runtime.outcome_tracker.active_trades[signal_id]
    stale_trade.t2_hit = False
    stale_trade.highest_premium = 5
    triggered = stale_runtime.outcome_tracker.check("NIFTY", 22300)
    assert triggered[0]["outcome"] == "WIN_T2"



def test_idempotent_no_trade_persists_decision_and_audit_atomically(tmp_path: Path):
    class NoTradeChain(FakeChain):
        direction = "NO_TRADE"
        actionable = False
        verdict = "NO_TRADE test"

    db = PatternDB(tmp_path / "patterns.db")
    runtime = BrainRuntime(pattern_db=db, learnings_path=tmp_path / "learnings.json")
    runtime.state_cache["NIFTY"] = make_state()
    runtime.reasoning_engine.reason = lambda *args, **kwargs: NoTradeChain()
    first = runtime.create_signal("NIFTY", idempotency_key="no-trade-key")
    assert first["signal_id"] is None
    assert runtime.create_signal("NIFTY", idempotency_key="no-trade-key") == first
    with db._conn() as connection:
        assert connection.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM decisions WHERE operation='signal'").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM audit_events WHERE event_type='signal_evaluated'").fetchone()[0] == 1


def test_decision_paths_refresh_newer_persisted_snapshot(tmp_path: Path):
    db_path = tmp_path / "patterns.db"
    runtime = BrainRuntime(
        db_path,
        learnings_path=tmp_path / "learnings.json",
        decision_max_age_seconds=30,
    )
    stale = make_state()
    stale.timestamp = time.time() - 120
    runtime.state_cache["NIFTY"] = stale

    fresh = make_state()
    fresh.timestamp = time.time()
    PatternDB(db_path).record_state_snapshot(fresh, fresh.quality_dict())

    resolved = runtime.get_state_object("NIFTY", require_fresh=True)
    assert resolved.timestamp == fresh.timestamp
    assert runtime.state_cache["NIFTY"].timestamp == fresh.timestamp
