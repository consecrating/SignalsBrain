"""
SignalsBrain — PatternMemory Database

SQLite-backed storage for every signal generated + its real-world outcome.
This is the brain's LONG-TERM MEMORY. It answers:

  "The last 47 times we had Negative Gamma + PCR > 1.2 + ADX trending + bearish,
   the win rate was 72%, average move was 1.8 ATR, and average time-to-T1 was 43 min."

That's not a guess. That's a statistical fact from YOUR OWN trading history.
No AI model alone can give you this — they don't have your specific data.
But when connected to SignalsBrain, they GET this data and use it.

Schema designed for:
  - Fast categorical queries (indexed fingerprint columns)
  - Full vector retrieval for similarity search
  - Outcome tracking with multiple time horizons
  - Regime-specific statistics
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .fingerprint import CategoricalFingerprint, build_categorical_fingerprint


DB_PATH = Path(__file__).parent.parent.parent / "data" / "patterns.db"

WIN_OUTCOMES = frozenset({"WIN_T1", "WIN_T2", "WIN_T3"})
LOSS_OUTCOMES = frozenset({"STOP_LOSS"})
COMPLETED_OUTCOMES = WIN_OUTCOMES | LOSS_OUTCOMES | {"TIME_EXIT", "NO_ENTRY"}


def outcome_semantic(outcome: str) -> str:
    """Return the single lifecycle semantic used by statistics and learning."""
    if outcome in WIN_OUTCOMES:
        return "win"
    if outcome in LOSS_OUTCOMES:
        return "loss"
    if outcome == "TIME_EXIT":
        return "neutral"
    if outcome == "NO_ENTRY":
        return "excluded"
    return "open"


class PatternDBError(RuntimeError):
    """Base class for pattern-memory domain errors."""


class SignalNotFoundError(PatternDBError):
    """Raised when an outcome references a signal that does not exist."""


class SignalAlreadyClosedError(PatternDBError):
    """Raised when an outcome attempts to close an already-closed signal."""


class IdempotencyConflictError(PatternDBError):
    """Raised when a key is reused for a different request payload."""


@dataclass
class SignalRecord:
    """A single signal event stored in pattern memory."""
    id: Optional[int] = None
    timestamp: float = 0.0
    instrument: str = ""
    direction: str = ""  # BUY, SELL, NO_TRADE
    confidence: float = 0.0
    net_bias: float = 0.0
    
    # Categorical fingerprint (12 fields, indexed for fast queries)
    gex_regime: int = 0
    gex_flip_zone: int = 0
    pcr_band: int = 0
    adx_band: int = 0
    trend_dir: int = 0
    momentum_zone: int = 0
    volume_state: int = 0
    iv_regime: int = 0
    vwap_pos: int = 0
    session: int = 0
    dte_band: int = 0
    fii_dir: int = 0
    
    # Full state vector (JSON-serialized numpy array)
    state_vector: str = ""  # JSON array of 47 floats
    
    # Entry details
    entry_spot: float = 0.0
    entry_premium: float = 0.0
    strike: float = 0.0
    opt_type: str = ""  # CE or PE
    atr_at_entry: float = 0.0
    
    # Outcome (filled later when trade closes)
    outcome: str = ""  # WIN_T1, WIN_T2, WIN_T3, STOP_LOSS, TIME_EXIT, NO_ENTRY
    exit_spot: float = 0.0
    exit_premium: float = 0.0
    move_atr: float = 0.0  # How far price moved in ATR units
    duration_minutes: float = 0.0
    pnl_pct: float = 0.0  # % gain/loss on premium
    
    # Metadata
    vetoes_applied: str = ""  # JSON list of veto names that fired
    evidence_summary: str = ""  # Brief reasoning snapshot


@dataclass
class PatternStats:
    """Statistics for a matched pattern group."""
    total_signals: int = 0
    total_trades: int = 0  # Signals that became actual trades (not NO_TRADE)
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_move_atr: float = 0.0
    avg_duration_min: float = 0.0
    avg_pnl_pct: float = 0.0
    best_pnl_pct: float = 0.0
    worst_pnl_pct: float = 0.0
    hit_t1_rate: float = 0.0
    hit_t2_rate: float = 0.0
    hit_t3_rate: float = 0.0
    stop_loss_rate: float = 0.0
    
    # Regime breakdown
    regime_trending_wr: float = 0.0
    regime_choppy_wr: float = 0.0
    
    # Recent performance (last 20 signals — detects degradation)
    recent_win_rate: float = 0.0
    is_degrading: bool = False  # recent_win_rate significantly below historical


class PatternDB:
    """
    SQLite-backed pattern memory with indexed categorical columns for fast queries.
    """
    
    def __init__(self, db_path: Optional[Path] = None, *, idempotency_ttl_seconds: Optional[float] = None):
        self.db_path = db_path or DB_PATH
        configured_ttl = idempotency_ttl_seconds
        if configured_ttl is None:
            configured_ttl = float(os.environ.get("SIGNALSBRAIN_IDEMPOTENCY_TTL_SECONDS", "86400"))
        if not math.isfinite(configured_ttl) or configured_ttl < 60 or configured_ttl > 2_592_000:
            raise ValueError("Idempotency TTL must be finite and between 60 and 2592000 seconds")
        self.idempotency_ttl_seconds = configured_ttl
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.prune_expired_idempotency_keys(limit=1000)
    
    def _init_db(self):
        """Apply additive, repeatable migrations without rebuilding existing rows."""
        with self._conn() as conn:
            # Serialize schema discovery and DDL so concurrent constructors cannot
            # race on ALTER TABLE. SQLite DDL remains transactional here.
            conn.execute("BEGIN EXCLUSIVE")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    instrument TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    net_bias REAL NOT NULL,
                    gex_regime INTEGER NOT NULL DEFAULT 0,
                    gex_flip_zone INTEGER NOT NULL DEFAULT 0,
                    pcr_band INTEGER NOT NULL DEFAULT 0,
                    adx_band INTEGER NOT NULL DEFAULT 0,
                    trend_dir INTEGER NOT NULL DEFAULT 0,
                    momentum_zone INTEGER NOT NULL DEFAULT 0,
                    volume_state INTEGER NOT NULL DEFAULT 0,
                    iv_regime INTEGER NOT NULL DEFAULT 0,
                    vwap_pos INTEGER NOT NULL DEFAULT 0,
                    session INTEGER NOT NULL DEFAULT 0,
                    dte_band INTEGER NOT NULL DEFAULT 0,
                    fii_dir INTEGER NOT NULL DEFAULT 0,
                    state_vector TEXT DEFAULT '',
                    entry_spot REAL DEFAULT 0,
                    entry_premium REAL DEFAULT 0,
                    strike REAL DEFAULT 0,
                    opt_type TEXT DEFAULT '',
                    atr_at_entry REAL DEFAULT 0,
                    outcome TEXT DEFAULT '',
                    exit_spot REAL DEFAULT 0,
                    exit_premium REAL DEFAULT 0,
                    move_atr REAL DEFAULT 0,
                    duration_minutes REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    vetoes_applied TEXT DEFAULT '',
                    evidence_summary TEXT DEFAULT ''
                )
            """)

            # Older v1 databases may have a subset of the current columns. ALTER
            # only what is absent; SQLite preserves every existing row and id.
            signal_columns = {
                "timestamp": "REAL NOT NULL DEFAULT 0",
                "instrument": "TEXT NOT NULL DEFAULT ''",
                "direction": "TEXT NOT NULL DEFAULT ''",
                "confidence": "REAL NOT NULL DEFAULT 0",
                "net_bias": "REAL NOT NULL DEFAULT 0",
                "gex_regime": "INTEGER NOT NULL DEFAULT 0",
                "gex_flip_zone": "INTEGER NOT NULL DEFAULT 0",
                "pcr_band": "INTEGER NOT NULL DEFAULT 0",
                "adx_band": "INTEGER NOT NULL DEFAULT 0",
                "trend_dir": "INTEGER NOT NULL DEFAULT 0",
                "momentum_zone": "INTEGER NOT NULL DEFAULT 0",
                "volume_state": "INTEGER NOT NULL DEFAULT 0",
                "iv_regime": "INTEGER NOT NULL DEFAULT 0",
                "vwap_pos": "INTEGER NOT NULL DEFAULT 0",
                "session": "INTEGER NOT NULL DEFAULT 0",
                "dte_band": "INTEGER NOT NULL DEFAULT 0",
                "fii_dir": "INTEGER NOT NULL DEFAULT 0",
                "state_vector": "TEXT DEFAULT ''",
                "entry_spot": "REAL DEFAULT 0",
                "entry_premium": "REAL DEFAULT 0",
                "strike": "REAL DEFAULT 0",
                "opt_type": "TEXT DEFAULT ''",
                "atr_at_entry": "REAL DEFAULT 0",
                "outcome": "TEXT DEFAULT ''",
                "exit_spot": "REAL DEFAULT 0",
                "exit_premium": "REAL DEFAULT 0",
                "move_atr": "REAL DEFAULT 0",
                "duration_minutes": "REAL DEFAULT 0",
                "pnl_pct": "REAL DEFAULT 0",
                "vetoes_applied": "TEXT DEFAULT ''",
                "evidence_summary": "TEXT DEFAULT ''",
            }
            existing = {row[1] for row in conn.execute("PRAGMA table_info(signals)")}
            for name, ddl in signal_columns.items():
                if name not in existing:
                    conn.execute(f'ALTER TABLE signals ADD COLUMN "{name}" {ddl}')

            statements = (
                """CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL
                )""",
                """CREATE TABLE IF NOT EXISTS state_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instrument TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    state_json TEXT NOT NULL,
                    quality_json TEXT NOT NULL DEFAULT '{}',
                    snapshot_version INTEGER NOT NULL DEFAULT 1
                )""",
                """CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    instrument TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    direction TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0,
                    decision_json TEXT NOT NULL
                )""",
                """CREATE TABLE IF NOT EXISTS idempotency_keys (
                    operation TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    PRIMARY KEY (operation, idempotency_key)
                )""",
                """CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL DEFAULT '',
                    entity_id TEXT NOT NULL DEFAULT '',
                    details_json TEXT NOT NULL DEFAULT '{}'
                )""",
                """CREATE TABLE IF NOT EXISTS learning_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER NOT NULL UNIQUE,
                    outcome TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    completed_at REAL NOT NULL DEFAULT 0
                )""",
                """CREATE TABLE IF NOT EXISTS active_trade_progress (
                    signal_id INTEGER PRIMARY KEY,
                    t1_hit INTEGER NOT NULL DEFAULT 0,
                    t2_hit INTEGER NOT NULL DEFAULT 0,
                    highest_premium REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )""",
                """CREATE TABLE IF NOT EXISTS learning_events (
                    event_id TEXT PRIMARY KEY,
                    processed_at REAL NOT NULL
                )""",
                "CREATE INDEX IF NOT EXISTS idx_instrument ON signals(instrument)",
                "CREATE INDEX IF NOT EXISTS idx_direction ON signals(direction)",
                "CREATE INDEX IF NOT EXISTS idx_outcome ON signals(outcome)",
                "CREATE INDEX IF NOT EXISTS idx_gex_regime ON signals(gex_regime)",
                "CREATE INDEX IF NOT EXISTS idx_pcr_band ON signals(pcr_band)",
                "CREATE INDEX IF NOT EXISTS idx_adx_band ON signals(adx_band)",
                "CREATE INDEX IF NOT EXISTS idx_trend_dir ON signals(trend_dir)",
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON signals(timestamp)",
                """CREATE INDEX IF NOT EXISTS idx_setup_type ON signals(
                    instrument, direction, gex_regime, adx_band, trend_dir, pcr_band
                )""",
                "CREATE INDEX IF NOT EXISTS idx_snapshots_instrument_time ON state_snapshots(instrument, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_decisions_instrument_time ON decisions(instrument, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_audit_event_time ON audit_events(event_type, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_learning_outbox_status ON learning_outbox(status, created_at)",
            )
            for statement in statements:
                conn.execute(statement)

            snapshot_columns = {row[1] for row in conn.execute("PRAGMA table_info(state_snapshots)")}
            if "snapshot_version" not in snapshot_columns:
                conn.execute(
                    "ALTER TABLE state_snapshots ADD COLUMN snapshot_version INTEGER NOT NULL DEFAULT 1"
                )
            idempotency_columns = {row[1] for row in conn.execute("PRAGMA table_info(idempotency_keys)")}
            if "expires_at" not in idempotency_columns:
                conn.execute("ALTER TABLE idempotency_keys ADD COLUMN expires_at REAL")
            conn.execute(
                "UPDATE idempotency_keys SET expires_at = created_at + ? WHERE expires_at IS NULL",
                (self.idempotency_ttl_seconds,),
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_idempotency_expiry ON idempotency_keys(expires_at)"
            )
            now = time.time()
            conn.execute(
                "INSERT OR REPLACE INTO schema_metadata(key, value, updated_at) VALUES (?, ?, ?)",
                ("schema_version", "3", now),
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_metadata(key, value, updated_at) VALUES (?, ?, ?)",
                ("fingerprint_version", "1", now),
            )
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=30)
    
    # ──────────────────────────────────────────────────────────────────────────
    # WRITE OPERATIONS
    # ──────────────────────────────────────────────────────────────────────────
    
    def record_signal(self, state, direction: str, confidence: float,
                      entry_spot: float = 0, entry_premium: float = 0,
                      strike: float = 0, opt_type: str = "",
                      atr: float = 0, vetoes: list = None,
                      evidence: str = "") -> int:
        """Record a signal event and return its stable row id."""
        with self._conn() as conn:
            signal_id = self._insert_signal(
                conn, state, direction, confidence, entry_spot, entry_premium,
                strike, opt_type, atr, vetoes, evidence,
            )
            conn.commit()
            return signal_id

    def get_idempotent_response(
        self,
        operation: str,
        idempotency_key: str,
        request_hash: str,
    ) -> Optional[dict[str, Any]]:
        """Resolve an unexpired retry before ephemeral reasoning is required."""
        now = time.time()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT request_hash, response_json, expires_at FROM idempotency_keys "
                "WHERE operation = ? AND idempotency_key = ?",
                (operation, idempotency_key),
            ).fetchone()
            if row is not None and row[2] is not None and row[2] <= now:
                conn.execute(
                    "DELETE FROM idempotency_keys WHERE operation = ? AND idempotency_key = ?",
                    (operation, idempotency_key),
                )
                conn.commit()
                return None
            if row is None:
                conn.commit()
                return None
            if row[0] != request_hash:
                raise IdempotencyConflictError(
                    "Idempotency key was already used for a different request"
                )
            response = json.loads(row[1])
            conn.commit()
            return response

    def prune_expired_idempotency_keys(self, *, limit: int = 1000, now: Optional[float] = None) -> int:
        """Delete a bounded batch of expired retry keys."""
        if limit < 1 or limit > 10_000:
            raise ValueError("Prune limit must be between 1 and 10000")
        cutoff = time.time() if now is None else now
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM idempotency_keys WHERE rowid IN ("
                "SELECT rowid FROM idempotency_keys WHERE expires_at <= ? ORDER BY expires_at LIMIT ?"
                ")",
                (cutoff, limit),
            )
            conn.commit()
            return cursor.rowcount

    def record_signal_idempotent(
        self,
        *,
        operation: str,
        idempotency_key: str,
        request_hash: str,
        response: dict[str, Any],
        state,
        direction: str,
        confidence: float,
        entry_spot: float = 0,
        entry_premium: float = 0,
        strike: float = 0,
        opt_type: str = "",
        atr: float = 0,
        vetoes: list | None = None,
        evidence: str = "",
    ) -> tuple[int, dict[str, Any], bool]:
        """Atomically insert a signal and cache its response under a retry key."""
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT request_hash, response_json, expires_at FROM idempotency_keys "
                "WHERE operation = ? AND idempotency_key = ?",
                (operation, idempotency_key),
            ).fetchone()
            if existing and existing[2] is not None and existing[2] <= time.time():
                conn.execute(
                    "DELETE FROM idempotency_keys WHERE operation = ? AND idempotency_key = ?",
                    (operation, idempotency_key),
                )
                existing = None
            if existing:
                if existing[0] != request_hash:
                    raise IdempotencyConflictError(
                        "Idempotency key was already used for a different request"
                    )
                cached = json.loads(existing[1])
                return int(cached.get("signal_id") or 0), cached, True

            signal_id = self._insert_signal(
                conn, state, direction, confidence, entry_spot, entry_premium,
                strike, opt_type, atr, vetoes, evidence,
            )
            stable_response = dict(response)
            stable_response["signal_id"] = signal_id
            conn.execute(
                "INSERT INTO idempotency_keys(operation, idempotency_key, request_hash, response_json, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    operation,
                    idempotency_key,
                    request_hash,
                    json.dumps(stable_response, separators=(",", ":"), sort_keys=True),
                    time.time(),
                    time.time() + self.idempotency_ttl_seconds,
                ),
            )
            conn.commit()
            return signal_id, stable_response, False

    def store_idempotent_response(
        self,
        operation: str,
        idempotency_key: str,
        request_hash: str,
        response: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Store/replay an idempotent response for operations without a signal row."""
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT request_hash, response_json, expires_at FROM idempotency_keys "
                "WHERE operation = ? AND idempotency_key = ?",
                (operation, idempotency_key),
            ).fetchone()
            if existing and existing[2] is not None and existing[2] <= time.time():
                conn.execute(
                    "DELETE FROM idempotency_keys WHERE operation = ? AND idempotency_key = ?",
                    (operation, idempotency_key),
                )
                existing = None
            if existing:
                if existing[0] != request_hash:
                    raise IdempotencyConflictError(
                        "Idempotency key was already used for a different request"
                    )
                return json.loads(existing[1]), True
            conn.execute(
                "INSERT INTO idempotency_keys(operation, idempotency_key, request_hash, response_json, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    operation,
                    idempotency_key,
                    request_hash,
                    json.dumps(response, separators=(",", ":"), sort_keys=True),
                    time.time(),
                    time.time() + self.idempotency_ttl_seconds,
                ),
            )
            conn.commit()
            return response, False

    def persist_signal_command(
        self,
        *,
        operation: str,
        idempotency_key: Optional[str],
        request_hash: str,
        response: dict[str, Any],
        state,
        direction: str,
        confidence: float,
        decision: dict[str, Any],
        entry_spot: float = 0,
        entry_premium: float = 0,
        strike: float = 0,
        opt_type: str = "",
        atr: float = 0,
        vetoes: list | None = None,
        evidence: str = "",
    ) -> tuple[Optional[int], dict[str, Any], bool]:
        """Atomically persist signal/no-trade, decision, audit, and retry response."""
        now = time.time()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM idempotency_keys WHERE rowid IN ("
                "SELECT rowid FROM idempotency_keys WHERE expires_at <= ? ORDER BY expires_at LIMIT 1000"
                ")",
                (now,),
            )
            if idempotency_key:
                existing = conn.execute(
                    "SELECT request_hash, response_json, expires_at FROM idempotency_keys "
                    "WHERE operation = ? AND idempotency_key = ?",
                    (operation, idempotency_key),
                ).fetchone()
                if existing and existing[2] is not None and existing[2] <= now:
                    conn.execute(
                        "DELETE FROM idempotency_keys WHERE operation = ? AND idempotency_key = ?",
                        (operation, idempotency_key),
                    )
                    existing = None
                if existing:
                    if existing[0] != request_hash:
                        raise IdempotencyConflictError(
                            "Idempotency key was already used for a different request"
                        )
                    cached = json.loads(existing[1])
                    conn.commit()
                    return cached.get("signal_id"), cached, True

            signal_id: Optional[int] = None
            stable_response = dict(response)
            if direction != "NO_TRADE":
                signal_id = self._insert_signal(
                    conn, state, direction, confidence, entry_spot, entry_premium,
                    strike, opt_type, atr, vetoes, evidence,
                )
                stable_response["signal_id"] = signal_id

            conn.execute(
                "INSERT INTO decisions(timestamp, instrument, operation, direction, confidence, decision_json) "
                "VALUES (?, ?, 'signal', ?, ?, ?)",
                (
                    now, state.instrument.upper(), direction, confidence,
                    json.dumps(decision, separators=(",", ":"), sort_keys=True),
                ),
            )
            event_type = "signal_created" if signal_id else "signal_evaluated"
            conn.execute(
                "INSERT INTO audit_events(timestamp, event_type, entity_type, entity_id, details_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    now,
                    event_type,
                    "signal" if signal_id else "instrument",
                    str(signal_id or state.instrument.upper()),
                    json.dumps(
                        {"direction": direction, "confidence": confidence},
                        separators=(",", ":"), sort_keys=True,
                    ),
                ),
            )
            if idempotency_key:
                conn.execute(
                    "INSERT INTO idempotency_keys(operation, idempotency_key, request_hash, response_json, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        operation, idempotency_key, request_hash,
                        json.dumps(stable_response, separators=(",", ":"), sort_keys=True),
                        now, now + self.idempotency_ttl_seconds,
                    ),
                )
            conn.commit()
            return signal_id, stable_response, False

    @staticmethod
    def _insert_signal(
        conn: sqlite3.Connection,
        state,
        direction: str,
        confidence: float,
        entry_spot: float,
        entry_premium: float,
        strike: float,
        opt_type: str,
        atr: float,
        vetoes: list | None,
        evidence: str,
    ) -> int:
        fp = build_categorical_fingerprint(state)
        vec = state.fingerprint().tolist()
        cursor = conn.execute("""
            INSERT INTO signals (
                timestamp, instrument, direction, confidence, net_bias,
                gex_regime, gex_flip_zone, pcr_band, adx_band, trend_dir,
                momentum_zone, volume_state, iv_regime, vwap_pos, session,
                dte_band, fii_dir, state_vector, entry_spot, entry_premium,
                strike, opt_type, atr_at_entry, vetoes_applied, evidence_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), state.instrument, direction, confidence, state.net_directional_bias,
            fp.gex_regime, fp.gex_flip_zone, fp.pcr_band, fp.adx_band, fp.trend_dir,
            fp.momentum_zone, fp.volume_state, fp.iv_regime, fp.vwap_pos, fp.session,
            fp.dte_band, fp.fii_dir, json.dumps(vec), entry_spot, entry_premium,
            strike, opt_type, atr, json.dumps(vetoes or []), evidence,
        ))
        return int(cursor.lastrowid)

    def record_outcome(self, signal_id: int, outcome: str, exit_spot: float,
                       exit_premium: float, move_atr: float, duration_min: float,
                       pnl_pct: float) -> SignalRecord:
        """Close an open signal exactly once; only the winner emits side effects."""
        if outcome not in COMPLETED_OUTCOMES:
            raise ValueError(f"Unsupported outcome: {outcome}")
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
            if row is None:
                raise SignalNotFoundError(f"Signal {signal_id} does not exist")
            if row["outcome"]:
                raise SignalAlreadyClosedError(
                    f"Signal {signal_id} is already closed with outcome {row['outcome']}"
                )
            cursor = conn.execute("""
                UPDATE signals SET
                    outcome = ?, exit_spot = ?, exit_premium = ?,
                    move_atr = ?, duration_minutes = ?, pnl_pct = ?
                WHERE id = ? AND outcome = ''
            """, (outcome, exit_spot, exit_premium, move_atr, duration_min, pnl_pct, signal_id))
            if cursor.rowcount != 1:
                winner = conn.execute(
                    "SELECT outcome FROM signals WHERE id = ?", (signal_id,)
                ).fetchone()
                winner_outcome = winner["outcome"] if winner else "unknown"
                raise SignalAlreadyClosedError(
                    f"Signal {signal_id} is already closed with outcome {winner_outcome}"
                )
            now = time.time()
            conn.execute(
                "INSERT INTO learning_outbox(signal_id, outcome, status, created_at) "
                "VALUES (?, ?, 'pending', ?)",
                (signal_id, outcome, now),
            )
            conn.execute(
                "DELETE FROM active_trade_progress WHERE signal_id = ?", (signal_id,)
            )
            conn.execute(
                "INSERT INTO audit_events(timestamp, event_type, entity_type, entity_id, details_json) "
                "VALUES (?, 'outcome_committed', 'signal', ?, ?)",
                (now, str(signal_id), json.dumps({"outcome": outcome}, separators=(",", ":"))),
            )
            updated = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
            conn.commit()
            return self._row_to_record(updated)

    def upsert_active_trade_progress(
        self, signal_id: int, *, t1_hit: bool, t2_hit: bool, highest_premium: float
    ) -> None:
        """Persist monitor high-water state so restart cannot change classification."""
        if not math.isfinite(highest_premium):
            raise ValueError("highest_premium must be finite")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO active_trade_progress(signal_id, t1_hit, t2_hit, highest_premium, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(signal_id) DO UPDATE SET "
                "t1_hit=MAX(active_trade_progress.t1_hit, excluded.t1_hit), "
                "t2_hit=MAX(active_trade_progress.t2_hit, excluded.t2_hit), "
                "highest_premium=MAX(active_trade_progress.highest_premium, excluded.highest_premium), "
                "updated_at=excluded.updated_at",
                (signal_id, int(t1_hit), int(t2_hit), highest_premium, time.time()),
            )
            conn.commit()

    def get_active_trade_progress(self, signal_id: int) -> Optional[dict[str, Any]]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT t1_hit, t2_hit, highest_premium, updated_at "
                "FROM active_trade_progress WHERE signal_id = ?",
                (signal_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "t1_hit": bool(row["t1_hit"]),
            "t2_hit": bool(row["t2_hit"]),
            "highest_premium": float(row["highest_premium"]),
            "updated_at": float(row["updated_at"]),
        }

    def get_signal(self, signal_id: int) -> Optional[SignalRecord]:
        """Return one signal without changing lifecycle state."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def get_open_signals(self) -> list[SignalRecord]:
        """Return persisted directional signals whose lifecycle is still open."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM signals WHERE outcome = '' AND direction IN ('BUY', 'SELL') "
                "ORDER BY timestamp, id"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_pending_learning(self, signal_id: int) -> Optional[dict[str, Any]]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM learning_outbox WHERE signal_id = ? AND status != 'completed'",
                (signal_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_pending_learnings(self, limit: int = 100, after_id: int = 0) -> list[dict[str, Any]]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM learning_outbox WHERE status != 'completed' AND id > ? "
                "ORDER BY id LIMIT ?",
                (after_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_learning_completed(self, outbox_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE learning_outbox SET status = 'completed', attempts = attempts + 1, "
                "last_error = '', completed_at = ? WHERE id = ?",
                (time.time(), outbox_id),
            )
            conn.commit()

    def mark_learning_failed(self, outbox_id: int, error: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE learning_outbox SET status = 'pending', attempts = attempts + 1, "
                "last_error = ? WHERE id = ?",
                (error[:500], outbox_id),
            )
            conn.commit()

    def record_state_snapshot(self, state, quality: dict[str, Any]) -> int:
        """Persist an append-only versioned state snapshot for audit and recovery."""
        state_json = json.dumps(state.to_dict(include_raw=True), separators=(",", ":"), sort_keys=True, allow_nan=False)
        quality_json = json.dumps(quality, separators=(",", ":"), sort_keys=True, allow_nan=False)
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO state_snapshots(instrument, timestamp, state_json, quality_json, snapshot_version) "
                "VALUES (?, ?, ?, ?, 2)",
                (state.instrument, state.timestamp, state_json, quality_json),
            )
            conn.commit()
            return int(cursor.lastrowid)

    @classmethod
    def _valid_snapshot_payload(cls, row: sqlite3.Row) -> Optional[dict[str, Any]]:
        try:
            version = int(row["snapshot_version"])
            state = json.loads(row["state_json"])
            quality = json.loads(row["quality_json"])
            timestamp = float(row["timestamp"])
            if version not in (1, 2) or not math.isfinite(timestamp):
                return None
            if not isinstance(state, dict) or not isinstance(quality, dict):
                return None
            if state.get("instrument", "").upper() != str(row["instrument"]).upper():
                return None
            if not isinstance(state.get("dimensions", {}), dict):
                return None

            def finite_tree(value: Any, depth: int = 0) -> bool:
                if depth > 8:
                    return False
                if isinstance(value, float):
                    return math.isfinite(value)
                if isinstance(value, dict):
                    return len(value) <= 200 and all(finite_tree(item, depth + 1) for item in value.values())
                if isinstance(value, list):
                    return len(value) <= 5000 and all(finite_tree(item, depth + 1) for item in value)
                return value is None or isinstance(value, (str, int, bool))

            if not finite_tree(state) or not finite_tree(quality):
                return None
            state_timestamp = float(state.get("timestamp", timestamp))
            if not math.isfinite(state_timestamp) or abs(state_timestamp - timestamp) > 0.001:
                return None
            return {
                "id": row["id"],
                "instrument": row["instrument"],
                "timestamp": timestamp,
                "snapshot_version": version,
                "state": state,
                "quality": quality,
            }
        except (ValueError, TypeError, json.JSONDecodeError, KeyError):
            return None

    def get_latest_state_snapshot(self, instrument: str) -> Optional[dict[str, Any]]:
        """Return the newest valid, recognized snapshot, skipping corrupt rows."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM state_snapshots WHERE instrument = ? "
                "ORDER BY timestamp DESC, id DESC LIMIT 50",
                (instrument.upper(),),
            ).fetchall()
        for row in rows:
            snapshot = self._valid_snapshot_payload(row)
            if snapshot is not None:
                return snapshot
        return None

    def list_snapshot_instruments(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT instrument FROM state_snapshots").fetchall()
        return [str(row[0]) for row in rows]

    def record_decision(self, instrument: str, operation: str, decision: dict[str, Any]) -> int:
        """Append an auditable reasoning result."""
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO decisions(timestamp, instrument, operation, direction, confidence, decision_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    time.time(), instrument.upper(), operation,
                    str(decision.get("direction", "")), float(decision.get("confidence", 0) or 0),
                    json.dumps(decision, separators=(",", ":"), sort_keys=True),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def record_audit_event(
        self,
        event_type: str,
        *,
        entity_type: str = "",
        entity_id: str = "",
        details: Optional[dict[str, Any]] = None,
    ) -> int:
        """Append a lifecycle or security-relevant event."""
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO audit_events(timestamp, event_type, entity_type, entity_id, details_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    time.time(), event_type, entity_type, entity_id,
                    json.dumps(details or {}, separators=(",", ":"), sort_keys=True),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_schema_metadata(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM schema_metadata").fetchall()
        return {str(key): str(value) for key, value in rows}
    
    # ──────────────────────────────────────────────────────────────────────────
    # QUERY OPERATIONS — The Brain's Memory Recall
    # ──────────────────────────────────────────────────────────────────────────
    
    def find_similar(self, state, direction: str, min_match: float = 0.5,
                     limit: Optional[int] = 50, instrument: Optional[str] = None) -> list[SignalRecord]:
        """
        Find historical signals with similar categorical fingerprints.
        
        This is THE key query: "What happened the last N times the market
        looked like this?"
        """
        fp = build_categorical_fingerprint(state)
        
        # Build flexible WHERE clause: match on the most important dimensions first
        # then filter by match_score
        where = ["direction = ?", "outcome != ''"]  # Only completed trades
        params: list = [direction]
        
        if instrument:
            where.append("instrument = ?")
            params.append(instrument)
        
        # Must match on these critical dimensions (hard filter)
        where.append("gex_regime = ?")
        params.append(fp.gex_regime)
        where.append("trend_dir = ?")
        params.append(fp.trend_dir)
        
        # Soft filter: ADX within 1 band
        where.append("ABS(adx_band - ?) <= 1")
        params.append(fp.adx_band)
        
        sql = f"SELECT * FROM signals WHERE {' AND '.join(where)} ORDER BY timestamp DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit * 3)  # Fetch extra, filter by score
        
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        
        # Score each result by full categorical match
        results = []
        for row in rows:
            row_fp = CategoricalFingerprint(
                gex_regime=row["gex_regime"], gex_flip_zone=row["gex_flip_zone"],
                pcr_band=row["pcr_band"], adx_band=row["adx_band"],
                trend_dir=row["trend_dir"], momentum_zone=row["momentum_zone"],
                volume_state=row["volume_state"], iv_regime=row["iv_regime"],
                vwap_pos=row["vwap_pos"], session=row["session"],
                dte_band=row["dte_band"], fii_dir=row["fii_dir"],
            )
            score = fp.relaxed_match_score(row_fp)
            if score >= min_match:
                rec = self._row_to_record(row)
                results.append((score, rec))
        
        # Sort by match score (best match first), take top N
        results.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in (results[:limit] if limit is not None else results)]
    
    def get_pattern_stats(self, state, direction: str,
                          instrument: Optional[str] = None) -> PatternStats:
        """
        Get comprehensive statistics for signals matching this state's pattern.
        
        This is what the AI model receives: not just "buy" but
        "buy — historically this exact setup won 72% of the time with avg +1.8 ATR move."
        """
        matches = self.find_similar(state, direction, min_match=0.5, limit=None, instrument=instrument)
        
        if not matches:
            return PatternStats()
        
        stats = PatternStats()
        stats.total_signals = len(matches)
        
        trades = [m for m in matches if outcome_semantic(m.outcome) != "excluded"]
        resolved = [m for m in trades if outcome_semantic(m.outcome) in ("win", "loss")]
        stats.total_trades = len(trades)

        if not trades:
            return stats

        wins = [trade for trade in resolved if outcome_semantic(trade.outcome) == "win"]
        losses = [trade for trade in resolved if outcome_semantic(trade.outcome) == "loss"]
        stats.wins = len(wins)
        stats.losses = len(losses)
        stats.win_rate = len(wins) / len(resolved) * 100 if resolved else 0
        
        stats.avg_confidence = sum(t.confidence for t in trades) / len(trades)
        stats.avg_move_atr = sum(abs(t.move_atr) for t in trades) / len(trades)
        stats.avg_duration_min = sum(t.duration_minutes for t in trades) / len(trades)
        stats.avg_pnl_pct = sum(t.pnl_pct for t in trades) / len(trades)
        stats.best_pnl_pct = max(t.pnl_pct for t in trades)
        stats.worst_pnl_pct = min(t.pnl_pct for t in trades)
        
        # Target and stop rates use only resolved win/loss outcomes; TIME_EXIT is neutral.
        resolved_count = len(resolved)
        if resolved_count:
            stats.hit_t1_rate = sum(1 for t in resolved if t.outcome in WIN_OUTCOMES) / resolved_count * 100
            stats.hit_t2_rate = sum(1 for t in resolved if t.outcome in ("WIN_T2", "WIN_T3")) / resolved_count * 100
            stats.hit_t3_rate = sum(1 for t in resolved if t.outcome == "WIN_T3") / resolved_count * 100
            stats.stop_loss_rate = sum(1 for t in resolved if t.outcome == "STOP_LOSS") / resolved_count * 100

        # Recency is selected chronologically from every qualifying match, never
        # from a match-score-truncated subset.
        recent = sorted(resolved, key=lambda trade: trade.timestamp, reverse=True)[:20]
        recent_wins = [t for t in recent if outcome_semantic(t.outcome) == "win"]
        stats.recent_win_rate = len(recent_wins) / len(recent) * 100 if recent else 0

        if len(resolved) >= 30 and stats.recent_win_rate < stats.win_rate - 15:
            stats.is_degrading = True
        
        return stats
    
    def get_regime_performance(self, instrument: str, days: int = 30) -> dict:
        """
        Performance breakdown by market regime over the last N days.
        Answers: "Am I better at trending markets or choppy ones?"
        """
        cutoff = time.time() - days * 86400
        
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT adx_band, outcome, pnl_pct, move_atr
                FROM signals
                WHERE instrument = ? AND timestamp > ?
                  AND outcome IN ('WIN_T1', 'WIN_T2', 'WIN_T3', 'STOP_LOSS')
                ORDER BY timestamp DESC
            """, (instrument, cutoff)).fetchall()
        
        regimes = {"choppy": [], "developing": [], "trending": [], "strong_trend": []}
        for row in rows:
            band = row["adx_band"]
            key = {-1: "choppy", 0: "developing", 1: "trending", 2: "strong_trend"}.get(band, "developing")
            regimes[key].append({"outcome": row["outcome"], "pnl": row["pnl_pct"], "move": row["move_atr"]})
        
        result = {}
        for regime, trades in regimes.items():
            if not trades:
                result[regime] = {"count": 0, "win_rate": 0, "avg_pnl": 0}
                continue
            wins = sum(1 for t in trades if t["outcome"] in WIN_OUTCOMES)
            result[regime] = {
                "count": len(trades),
                "win_rate": round(wins / len(trades) * 100, 1),
                "avg_pnl": round(sum(t["pnl"] for t in trades) / len(trades), 2),
                "avg_move_atr": round(sum(abs(t["move"]) for t in trades) / len(trades), 2),
            }
        
        return result
    
    def get_recent_signals(self, instrument: Optional[str] = None,
                           limit: int = 20) -> list[SignalRecord]:
        """Get the most recent signals for review."""
        where = ["1=1"]
        params: list = []
        if instrument:
            where.append("instrument = ?")
            params.append(instrument)
        
        sql = f"SELECT * FROM signals WHERE {' AND '.join(where)} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        
        return [self._row_to_record(row) for row in rows]
    
    def get_daily_summary(self, date: Optional[str] = None) -> dict:
        """Summary of today's (or a specific date's) signals and outcomes."""
        import datetime
        if date:
            day_start = datetime.datetime.strptime(date, "%Y-%m-%d").timestamp()
        else:
            today = datetime.date.today()
            day_start = datetime.datetime(today.year, today.month, today.day).timestamp()
        day_end = day_start + 86400
        
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM signals WHERE timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
            """, (day_start, day_end)).fetchall()
        
        signals = [self._row_to_record(row) for row in rows]
        trades = [s for s in signals if s.direction in ("BUY", "SELL")]
        taken = [s for s in trades if outcome_semantic(s.outcome) != "excluded"]
        completed = [s for s in taken if outcome_semantic(s.outcome) != "open"]
        resolved = [s for s in completed if outcome_semantic(s.outcome) in ("win", "loss")]
        wins = [s for s in resolved if outcome_semantic(s.outcome) == "win"]
        losses = [s for s in resolved if outcome_semantic(s.outcome) == "loss"]
        
        return {
            "date": date or str(datetime.date.today()),
            "total_signals": len(signals),
            "trades_taken": len(taken),
            "completed": len(completed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(resolved) * 100, 1) if resolved else 0,
            "total_pnl_pct": round(sum(s.pnl_pct for s in completed), 2),
            "signals": [asdict(s) for s in signals],
        }
    
    def count_records(self) -> int:
        """Total records in pattern memory."""
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    
    # ──────────────────────────────────────────────────────────────────────────
    # VECTOR SIMILARITY SEARCH (for finding the single most similar historical state)
    # ──────────────────────────────────────────────────────────────────────────
    
    def find_most_similar_vector(self, state, direction: str, top_n: int = 10) -> list[tuple[float, SignalRecord]]:
        """
        Cosine similarity search on the full 47-dimension vector.
        More expensive than categorical but finds subtle similarities.
        """
        target_vec = state.fingerprint()
        
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM signals
                WHERE direction = ? AND state_vector != '' AND outcome != ''
                ORDER BY timestamp DESC LIMIT 500
            """, (direction,)).fetchall()
        
        scored = []
        for row in rows:
            try:
                vec = np.array(json.loads(row["state_vector"]), dtype=np.float32)
                if len(vec) != len(target_vec):
                    continue
                # Cosine similarity
                dot = np.dot(target_vec, vec)
                n1 = np.linalg.norm(target_vec)
                n2 = np.linalg.norm(vec)
                if n1 == 0 or n2 == 0:
                    continue
                sim = float(dot / (n1 * n2))
                scored.append((sim, self._row_to_record(row)))
            except (json.JSONDecodeError, ValueError):
                continue
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_n]
    
    # ──────────────────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _row_to_record(row) -> SignalRecord:
        return SignalRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            instrument=row["instrument"],
            direction=row["direction"],
            confidence=row["confidence"],
            net_bias=row["net_bias"],
            gex_regime=row["gex_regime"],
            gex_flip_zone=row["gex_flip_zone"],
            pcr_band=row["pcr_band"],
            adx_band=row["adx_band"],
            trend_dir=row["trend_dir"],
            momentum_zone=row["momentum_zone"],
            volume_state=row["volume_state"],
            iv_regime=row["iv_regime"],
            vwap_pos=row["vwap_pos"],
            session=row["session"],
            dte_band=row["dte_band"],
            fii_dir=row["fii_dir"],
            state_vector=row["state_vector"],
            entry_spot=row["entry_spot"],
            entry_premium=row["entry_premium"],
            strike=row["strike"],
            opt_type=row["opt_type"],
            atr_at_entry=row["atr_at_entry"],
            outcome=row["outcome"],
            exit_spot=row["exit_spot"],
            exit_premium=row["exit_premium"],
            move_atr=row["move_atr"],
            duration_minutes=row["duration_minutes"],
            pnl_pct=row["pnl_pct"],
            vetoes_applied=row["vetoes_applied"],
            evidence_summary=row["evidence_summary"],
        )
