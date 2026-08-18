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
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .fingerprint import CategoricalFingerprint, build_categorical_fingerprint


DB_PATH = Path(__file__).parent.parent.parent / "data" / "patterns.db"


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
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    instrument TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    net_bias REAL NOT NULL,
                    
                    -- Categorical fingerprint (12 indexed columns)
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
                    
                    -- Full state vector
                    state_vector TEXT DEFAULT '',
                    
                    -- Entry
                    entry_spot REAL DEFAULT 0,
                    entry_premium REAL DEFAULT 0,
                    strike REAL DEFAULT 0,
                    opt_type TEXT DEFAULT '',
                    atr_at_entry REAL DEFAULT 0,
                    
                    -- Outcome (updated later)
                    outcome TEXT DEFAULT '',
                    exit_spot REAL DEFAULT 0,
                    exit_premium REAL DEFAULT 0,
                    move_atr REAL DEFAULT 0,
                    duration_minutes REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    
                    -- Metadata
                    vetoes_applied TEXT DEFAULT '',
                    evidence_summary TEXT DEFAULT ''
                )
            """)
            
            # Indexes for fast categorical queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_instrument ON signals(instrument)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_direction ON signals(direction)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON signals(outcome)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gex_regime ON signals(gex_regime)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pcr_band ON signals(pcr_band)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_adx_band ON signals(adx_band)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trend_dir ON signals(trend_dir)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON signals(timestamp)")
            
            # Composite index for the most common multi-column query
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_setup_type ON signals(
                    instrument, direction, gex_regime, adx_band, trend_dir, pcr_band
                )
            """)
            
            conn.commit()
    
    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))
    
    # ──────────────────────────────────────────────────────────────────────────
    # WRITE OPERATIONS
    # ──────────────────────────────────────────────────────────────────────────
    
    def record_signal(self, state, direction: str, confidence: float,
                      entry_spot: float = 0, entry_premium: float = 0,
                      strike: float = 0, opt_type: str = "",
                      atr: float = 0, vetoes: list = None,
                      evidence: str = "") -> int:
        """
        Record a signal event. Returns the record ID for later outcome update.
        
        Args:
            state: MarketState object (or anything with .dimensions and .fingerprint())
            direction: BUY, SELL, or NO_TRADE
            confidence: 0-100
            entry_spot: Spot price at signal time
            entry_premium: Option premium at entry (if traded)
            strike: Strike price
            opt_type: CE or PE
            atr: ATR at the time (for measuring moves)
            vetoes: List of veto names that were applied
            evidence: Brief reasoning summary
        """
        fp = build_categorical_fingerprint(state)
        vec = state.fingerprint().tolist()
        
        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO signals (
                    timestamp, instrument, direction, confidence, net_bias,
                    gex_regime, gex_flip_zone, pcr_band, adx_band, trend_dir,
                    momentum_zone, volume_state, iv_regime, vwap_pos, session,
                    dte_band, fii_dir,
                    state_vector,
                    entry_spot, entry_premium, strike, opt_type, atr_at_entry,
                    vetoes_applied, evidence_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(), state.instrument, direction, confidence, state.net_directional_bias,
                fp.gex_regime, fp.gex_flip_zone, fp.pcr_band, fp.adx_band, fp.trend_dir,
                fp.momentum_zone, fp.volume_state, fp.iv_regime, fp.vwap_pos, fp.session,
                fp.dte_band, fp.fii_dir,
                json.dumps(vec),
                entry_spot, entry_premium, strike, opt_type, atr,
                json.dumps(vetoes or []), evidence,
            ))
            conn.commit()
            return cursor.lastrowid
    
    def record_outcome(self, signal_id: int, outcome: str, exit_spot: float,
                       exit_premium: float, move_atr: float, duration_min: float,
                       pnl_pct: float):
        """Update a signal record with its real-world outcome."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE signals SET
                    outcome = ?, exit_spot = ?, exit_premium = ?,
                    move_atr = ?, duration_minutes = ?, pnl_pct = ?
                WHERE id = ?
            """, (outcome, exit_spot, exit_premium, move_atr, duration_min, pnl_pct, signal_id))
            conn.commit()
    
    # ──────────────────────────────────────────────────────────────────────────
    # QUERY OPERATIONS — The Brain's Memory Recall
    # ──────────────────────────────────────────────────────────────────────────
    
    def find_similar(self, state, direction: str, min_match: float = 0.5,
                     limit: int = 50, instrument: Optional[str] = None) -> list[SignalRecord]:
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
        
        sql = f"SELECT * FROM signals WHERE {' AND '.join(where)} ORDER BY timestamp DESC LIMIT ?"
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
        return [rec for _, rec in results[:limit]]
    
    def get_pattern_stats(self, state, direction: str,
                          instrument: Optional[str] = None) -> PatternStats:
        """
        Get comprehensive statistics for signals matching this state's pattern.
        
        This is what the AI model receives: not just "buy" but
        "buy — historically this exact setup won 72% of the time with avg +1.8 ATR move."
        """
        matches = self.find_similar(state, direction, min_match=0.5, limit=200, instrument=instrument)
        
        if not matches:
            return PatternStats()
        
        stats = PatternStats()
        stats.total_signals = len(matches)
        
        trades = [m for m in matches if m.outcome and m.outcome != "NO_ENTRY"]
        stats.total_trades = len(trades)
        
        if not trades:
            return stats
        
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        stats.wins = len(wins)
        stats.losses = len(losses)
        stats.win_rate = len(wins) / len(trades) * 100
        
        stats.avg_confidence = sum(t.confidence for t in trades) / len(trades)
        stats.avg_move_atr = sum(abs(t.move_atr) for t in trades) / len(trades)
        stats.avg_duration_min = sum(t.duration_minutes for t in trades) / len(trades)
        stats.avg_pnl_pct = sum(t.pnl_pct for t in trades) / len(trades)
        stats.best_pnl_pct = max(t.pnl_pct for t in trades)
        stats.worst_pnl_pct = min(t.pnl_pct for t in trades)
        
        # Target hit rates
        stats.hit_t1_rate = sum(1 for t in trades if t.outcome in ("WIN_T1", "WIN_T2", "WIN_T3")) / len(trades) * 100
        stats.hit_t2_rate = sum(1 for t in trades if t.outcome in ("WIN_T2", "WIN_T3")) / len(trades) * 100
        stats.hit_t3_rate = sum(1 for t in trades if t.outcome == "WIN_T3") / len(trades) * 100
        stats.stop_loss_rate = sum(1 for t in trades if t.outcome == "STOP_LOSS") / len(trades) * 100
        
        # Recent performance (last 20)
        recent = trades[-20:] if len(trades) >= 20 else trades
        recent_wins = [t for t in recent if t.pnl_pct > 0]
        stats.recent_win_rate = len(recent_wins) / len(recent) * 100 if recent else 0
        
        # Degradation detection: if recent win rate is >15% below historical
        if stats.total_trades >= 30 and stats.recent_win_rate < stats.win_rate - 15:
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
                WHERE instrument = ? AND timestamp > ? AND outcome != '' AND outcome != 'NO_ENTRY'
                ORDER BY timestamp DESC
            """, (instrument, cutoff)).fetchall()
        
        regimes = {"choppy": [], "developing": [], "trending": [], "strong_trend": []}
        for row in rows:
            band = row["adx_band"]
            key = {-1: "choppy", 0: "developing", 1: "trending", 2: "strong_trend"}.get(band, "developing")
            regimes[key].append({"pnl": row["pnl_pct"], "move": row["move_atr"]})
        
        result = {}
        for regime, trades in regimes.items():
            if not trades:
                result[regime] = {"count": 0, "win_rate": 0, "avg_pnl": 0}
                continue
            wins = sum(1 for t in trades if t["pnl"] > 0)
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
        completed = [s for s in trades if s.outcome]
        wins = [s for s in completed if s.pnl_pct > 0]
        
        return {
            "date": date or str(datetime.date.today()),
            "total_signals": len(signals),
            "trades_taken": len(trades),
            "completed": len(completed),
            "wins": len(wins),
            "losses": len(completed) - len(wins),
            "win_rate": round(len(wins) / len(completed) * 100, 1) if completed else 0,
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
