"""Unified application service for SignalsBrain state, reasoning, lifecycle, and learning."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

from .godmode.self_improve import SelfImproveEngine
from .memory.matcher import PatternMatcher
from .memory.outcome_tracker import OutcomeTracker
from .memory.pattern_db import PatternDB, SignalAlreadyClosedError, SignalRecord
from .reasoning.engine import ReasoningEngine
from .state.market_state import MarketState
from .state.state_builder import StateBuilder


VALID_OUTCOMES = ("WIN_T1", "WIN_T2", "WIN_T3", "STOP_LOSS", "TIME_EXIT", "NO_ENTRY")


class BrainRuntimeError(RuntimeError):
    """Base class for runtime domain errors."""


class StateUnavailableError(BrainRuntimeError):
    """Raised when no state exists for an instrument."""


class StateStaleError(BrainRuntimeError):
    """Raised when a market-hours state is too old for a decision."""


class InvalidOutcomeError(BrainRuntimeError):
    """Raised when an unsupported lifecycle outcome is supplied."""


class InvalidIdempotencyKeyError(BrainRuntimeError):
    """Raised when a retry key is outside the bounded runtime contract."""


class BrainRuntime:
    """Own all application-level state and coordinate durable side effects."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        *,
        pattern_db: Optional[PatternDB] = None,
        learnings_path: Optional[Path] = None,
        restore_snapshots: bool = True,
        decision_max_age_seconds: Optional[float] = None,
    ) -> None:
        self.pattern_db = pattern_db or PatternDB(db_path)
        configured_age = decision_max_age_seconds
        if configured_age is None:
            configured_age = float(os.environ.get("SIGNALSBRAIN_DECISION_MAX_AGE_SECONDS", "300"))
        if not math.isfinite(configured_age) or configured_age <= 0 or configured_age > 86_400:
            raise ValueError("Decision freshness must be finite and between 0 and 86400 seconds")
        self.decision_max_age_seconds = configured_age
        self.state_builder = StateBuilder()
        self.reasoning_engine = ReasoningEngine(pattern_db=self.pattern_db)
        self.self_improve = SelfImproveEngine(path=learnings_path)
        self.state_cache: dict[str, MarketState] = {}
        self.outcome_tracker = OutcomeTracker(
            self.pattern_db,
            record_callback=self._record_tracked_outcome,
        )
        self._rehydrate_open_trades()
        self._process_pending_learnings(raise_errors=False)
        if restore_snapshots:
            self._restore_latest_snapshots()

    # ── State ingestion and recovery ────────────────────────────────────────

    def _rehydrate_open_trades(self) -> None:
        """Restore monitorable open signals after process restart."""
        for record in self.pattern_db.get_open_signals():
            if not record.id or record.entry_spot <= 0 or record.atr_at_entry <= 0:
                # Legacy rows without entry/ATR data remain durable but cannot be
                # reconstructed into price-level monitoring safely.
                continue
            progress = self.pattern_db.get_active_trade_progress(record.id)
            self.outcome_tracker.start_tracking(
                signal_id=record.id,
                instrument=record.instrument,
                direction=record.direction,
                entry_spot=record.entry_spot,
                entry_premium=record.entry_premium,
                strike=record.strike,
                opt_type=record.opt_type or ("PE" if record.direction == "SELL" else "CE"),
                atr=record.atr_at_entry,
                entry_time=record.timestamp,
                t1_hit=bool(progress and progress["t1_hit"]),
                t2_hit=bool(progress and progress["t2_hit"]),
                highest_premium=(
                    float(progress["highest_premium"])
                    if progress else record.entry_premium
                ),
            )

    def _restore_latest_snapshots(self) -> None:
        for instrument in self.pattern_db.list_snapshot_instruments():
            self.refresh_snapshot(instrument)

    def refresh_snapshot(self, instrument: str) -> Optional[MarketState]:
        """Refresh one cache entry from the newest valid persisted snapshot."""
        instrument = instrument.upper()
        snapshot = self.pattern_db.get_latest_state_snapshot(instrument)
        if not snapshot:
            return self.state_cache.get(instrument)
        try:
            restored = MarketState.from_dict(snapshot["state"], snapshot.get("quality"))
        except (TypeError, ValueError, OverflowError):
            return self.state_cache.get(instrument)
        if restored.instrument.upper() != instrument or not math.isfinite(restored.timestamp):
            return self.state_cache.get(instrument)
        current = self.state_cache.get(instrument)
        if current is None or restored.timestamp > current.timestamp:
            self.state_cache[instrument] = restored
        return self.state_cache.get(instrument)

    def ingest_state(
        self,
        *,
        instrument: str,
        candles: Optional[dict] = None,
        gex_data: Optional[dict] = None,
        fii_dii: Optional[dict] = None,
        vix: Optional[float] = None,
        htf_candles: Optional[dict] = None,
        market_open: bool = True,
    ) -> tuple[MarketState, list[dict]]:
        instrument = instrument.upper()
        self._process_pending_learnings(raise_errors=False)
        state = self.state_builder.build(
            instrument=instrument,
            candles=candles,
            gex_data=gex_data,
            fii_dii=fii_dii,
            vix=vix,
            htf_candles=htf_candles,
            market_open=market_open,
        )
        self.state_cache[instrument] = state
        self.pattern_db.record_state_snapshot(state, state.quality_dict())

        outcomes: list[dict] = []
        if candles and candles.get("closes"):
            outcomes = self.outcome_tracker.check(instrument, candles["closes"][-1])

        self.pattern_db.record_audit_event(
            "state_ingested",
            entity_type="instrument",
            entity_id=instrument,
            details={
                "snapshot_timestamp": state.timestamp,
                "coverage": state.quality.coverage,
                "outcomes_triggered": len(outcomes),
            },
        )
        return state, outcomes

    def ingest(self, **kwargs: Any) -> dict[str, Any]:
        """Return the stable v1 ingestion shape."""
        state, outcomes = self.ingest_state(**kwargs)
        return {
            "ok": True,
            "instrument": state.instrument,
            "net_bias": round(state.net_directional_bias, 1),
            "regime": state.regime,
            "gex_regime": state.gex_regime,
            "outcomes_triggered": outcomes,
        }

    def get_state_object(self, instrument: str, *, require_fresh: bool = False) -> MarketState:
        instrument = instrument.upper()
        # Decision paths reconcile with durable state on every call so REST,
        # MCP, and other adapters cannot disagree when another process ingests
        # a newer snapshot. refresh_snapshot keeps a newer in-memory state.
        state = self.refresh_snapshot(instrument) if require_fresh else self.state_cache.get(instrument)
        if state is None:
            state = self.refresh_snapshot(instrument)
        if state is None:
            raise StateUnavailableError(f"No state for {instrument}")
        if require_fresh:
            age = state.age_seconds
            if not math.isfinite(age) or age < 0 or age > self.decision_max_age_seconds:
                raise StateStaleError(f"No fresh state for {instrument}")
        return state

    # ── Reasoning and signal lifecycle ──────────────────────────────────────

    def reason_state(
        self,
        state: MarketState,
        *,
        confidence_threshold: float = 60,
        session_signals: int = 0,
        session_stops: int = 0,
    ):
        chain = self.reasoning_engine.reason(
            state,
            confidence_threshold=confidence_threshold,
            session_signals=session_signals,
            session_stops=session_stops,
        )
        self.pattern_db.record_decision(state.instrument, "analyze", chain.to_dict())
        return chain

    def analyze(self, instrument: str) -> dict[str, Any]:
        instrument = instrument.upper()
        try:
            state = self.get_state_object(instrument, require_fresh=True)
        except (StateUnavailableError, StateStaleError) as exc:
            raise StateStaleError(
                f"No fresh state for {instrument}. Ingest data first via POST /brain/ingest"
            ) from exc
        chain = self.reason_state(state)
        return {
            "ok": True,
            "analysis": chain.to_dict(),
            "state_compact": state.to_compact(),
            "prompt": chain.to_prompt(),
        }

    @staticmethod
    def _evidence_metadata(chain, state: MarketState) -> str:
        evidence = []
        for item in chain.primary_evidence + chain.supporting_evidence + chain.counter_arguments:
            direction = getattr(item.direction, "value", str(item.direction))
            evidence.append({
                "factor": item.factor,
                "direction": direction,
                "impact": item.confidence_impact,
            })
        return json.dumps({
            "verdict": chain.verdict,
            "regime": state.regime,
            "gex_regime": state.gex_regime,
            "direction": chain.direction,
            "confidence": chain.confidence,
            "evidence_factors": evidence,
        }, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _signal_response(chain, signal_id: Optional[int]) -> dict[str, Any]:
        chain_data = chain.to_dict()
        return {
            "ok": True,
            "signal_id": signal_id,
            "direction": chain.direction,
            "confidence": round(chain.confidence, 1),
            "actionable": chain.actionable,
            "verdict": chain.verdict,
            "reasoning": chain.reasoning_narrative,
            "evidence": chain_data["evidence"],
            "confidence_breakdown": chain.confidence_breakdown,
            "risk": chain_data["risk"],
            "timing": chain_data["timing"],
            "vetoes": chain.vetoes,
            "historical": chain_data["historical"],
            "prompt_for_ai": chain.to_prompt(),
        }

    def create_signal(
        self,
        instrument: str,
        *,
        confidence_threshold: float = 60,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        instrument = instrument.upper()
        if idempotency_key is not None:
            if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", idempotency_key):
                raise InvalidIdempotencyKeyError(
                    "Idempotency key must be 1-128 URL-safe characters"
                )
        request_hash = hashlib.sha256(json.dumps(
            {"instrument": instrument, "confidence_threshold": confidence_threshold},
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")).hexdigest()
        if idempotency_key:
            cached = self.pattern_db.get_idempotent_response(
                "create_signal", idempotency_key, request_hash
            )
            if cached is not None:
                return cached

        state = self.get_state_object(instrument, require_fresh=True)
        chain = self.reasoning_engine.reason(
            state,
            confidence_threshold=confidence_threshold,
        )
        response = self._signal_response(chain, None)
        ltp_dim = state.dimensions.get("ltp")
        ltp = ltp_dim.raw if ltp_dim else 0
        atr_dim = state.dimensions.get("atr_pct")
        atr = (atr_dim.raw / 100 * ltp) if atr_dim else ltp * 0.01
        evidence = self._evidence_metadata(chain, state)
        signal_id, response, replayed = self.pattern_db.persist_signal_command(
            operation="create_signal",
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            response=response,
            state=state,
            direction=chain.direction,
            confidence=chain.confidence,
            decision=chain.to_dict(),
            entry_spot=ltp,
            opt_type="PE" if chain.direction == "SELL" else "CE",
            atr=atr,
            vetoes=chain.vetoes,
            evidence=evidence,
        )

        if not replayed and signal_id:
            self.outcome_tracker.start_tracking(
                signal_id=signal_id,
                instrument=instrument,
                direction=chain.direction,
                entry_spot=ltp,
                entry_premium=0,
                strike=0,
                opt_type="PE" if chain.direction == "SELL" else "CE",
                atr=atr,
            )
        return response

    # ── Outcome and learning ─────────────────────────────────────────────────

    @staticmethod
    def _learning_context(record: SignalRecord) -> dict[str, Any]:
        try:
            metadata = json.loads(record.evidence_summary) if record.evidence_summary else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        try:
            vetoes = json.loads(record.vetoes_applied) if record.vetoes_applied else []
        except (json.JSONDecodeError, TypeError):
            vetoes = []
        return {
            "regime": metadata.get("regime", ""),
            "gex_regime": metadata.get("gex_regime", ""),
            "direction": metadata.get("direction", record.direction),
            "confidence": float(metadata.get("confidence", record.confidence)),
            "evidence_factors": metadata.get("evidence_factors", []),
            "vetoes": vetoes,
        }

    def _learn_from_outcome(self, record: SignalRecord, *, event_id: str = "") -> list:
        context = self._learning_context(record)
        return self.self_improve.analyze_outcome(
            regime=context["regime"],
            gex_regime=context["gex_regime"],
            direction=context["direction"],
            confidence=context["confidence"],
            evidence_factors=context["evidence_factors"],
            outcome=record.outcome,
            vetoes_applied=context["vetoes"],
            event_id=event_id,
        )

    def _process_learning_outbox(
        self,
        outbox: dict[str, Any],
        *,
        evidence_factors: Optional[list[dict]] = None,
        regime: str = "",
        gex_regime: str = "",
        direction: str = "",
        confidence: float = 0,
        vetoes: Optional[list[str]] = None,
    ) -> list:
        record = self.pattern_db.get_signal(int(outbox["signal_id"]))
        if record is None:
            self.pattern_db.mark_learning_failed(int(outbox["id"]), "Signal row is missing")
            return []
        event_id = f"learning-outbox:{outbox['id']}"
        try:
            if evidence_factors is not None:
                learnings = self.self_improve.analyze_outcome(
                    regime=regime,
                    gex_regime=gex_regime,
                    direction=direction or record.direction,
                    confidence=confidence or record.confidence,
                    evidence_factors=evidence_factors,
                    outcome=record.outcome,
                    vetoes_applied=vetoes or [],
                    event_id=event_id,
                )
            else:
                learnings = self._learn_from_outcome(record, event_id=event_id)
            self.pattern_db.mark_learning_completed(int(outbox["id"]))
            self.pattern_db.record_audit_event(
                "outcome_learning_completed",
                entity_type="signal",
                entity_id=str(record.id),
                details={"outcome": record.outcome, "learnings": len(learnings)},
            )
            return learnings
        except Exception as exc:
            self.pattern_db.mark_learning_failed(int(outbox["id"]), str(exc))
            raise

    def _process_pending_learnings(self, *, raise_errors: bool) -> None:
        after_id = 0
        while True:
            batch = self.pattern_db.list_pending_learnings(limit=100, after_id=after_id)
            if not batch:
                return
            for outbox in batch:
                after_id = int(outbox["id"])
                try:
                    self._process_learning_outbox(outbox)
                except Exception:
                    if raise_errors:
                        raise
            if len(batch) < 100:
                return

    def _record_tracked_outcome(self, **kwargs: Any) -> SignalRecord:
        record = self.pattern_db.record_outcome(**kwargs)
        try:
            outbox = self.pattern_db.get_pending_learning(int(record.id))
            if outbox:
                self._process_learning_outbox(outbox)
        except Exception:
            # Every operation after durable closure is repairable downstream
            # work. Returning the closed record lets OutcomeTracker reconcile
            # its in-memory entry even when storage briefly fails.
            pass
        return record

    def record_outcome(
        self,
        *,
        signal_id: int,
        outcome: str,
        exit_spot: float = 0,
        exit_premium: float = 0,
        pnl_pct: float = 0,
        move_atr: float = 0,
        duration_min: float = 0,
        evidence_factors: Optional[list[dict]] = None,
        regime: str = "",
        gex_regime: str = "",
        direction: str = "",
        confidence: float = 0,
        vetoes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        if outcome not in VALID_OUTCOMES:
            raise InvalidOutcomeError(f"Invalid outcome. Use: {list(VALID_OUTCOMES)}")
        record = self.pattern_db.record_outcome(
            signal_id=signal_id,
            outcome=outcome,
            exit_spot=exit_spot,
            exit_premium=exit_premium,
            move_atr=move_atr,
            duration_min=duration_min,
            pnl_pct=pnl_pct,
        )
        self.outcome_tracker.active_trades.pop(signal_id, None)
        try:
            outbox = self.pattern_db.get_pending_learning(signal_id)
            if outbox:
                self._process_learning_outbox(
                    outbox,
                    evidence_factors=evidence_factors,
                    regime=regime,
                    gex_regime=gex_regime,
                    direction=direction,
                    confidence=confidence,
                    vetoes=vetoes,
                )
        except Exception:
            # SQLite closure is the request success boundary. Learning remains in
            # the durable outbox for startup/ingest repair.
            pass
        return {"ok": True, "message": f"Outcome '{outcome}' recorded for signal {signal_id}"}

    # ── Read models used by REST and MCP ─────────────────────────────────────

    def ask(self, question: str, instrument: Optional[str] = None) -> dict[str, Any]:
        instrument = (instrument or "NIFTY").upper()
        try:
            state = self.get_state_object(instrument, require_fresh=True)
        except (StateUnavailableError, StateStaleError):
            state = None
        context_parts = [f"Question: {question}", f"Instrument: {instrument}"]
        if state:
            context_parts.extend([
                f"\nCurrent State ({instrument}):",
                f"  Net bias: {state.net_directional_bias:.1f}",
                f"  Regime: {state.regime} | GEX: {state.gex_regime}",
                f"  Agreement: {state.agreement_factor:.0%}",
                "\nSignal Brain Assessment:",
                self.reasoning_engine.reason(state).to_prompt(),
            ])
        else:
            context_parts.append(f"\nNo live state available for {instrument}. Market may be closed.")
        daily = self.pattern_db.get_daily_summary()
        context_parts.append(
            f"\nToday's performance: {daily['wins']}W / {daily['losses']}L ({daily['win_rate']:.0f}% WR)"
        )
        return {
            "ok": True,
            "context": "\n".join(context_parts),
            "state_available": state is not None,
            "note": "Use this context to answer the user's question. The brain provides data and reasoning; you provide the natural language response.",
        }

    def state_response(self, instrument: str, *, compact: bool = True, include_quality: bool = False) -> dict[str, Any]:
        state = self.get_state_object(instrument)
        payload = state.to_compact() if compact else state.to_dict(include_raw=True)
        response: dict[str, Any] = {
            "ok": True,
            "state": payload,
            "age_seconds": round(state.age_seconds, 1),
        }
        if include_quality:
            response["quality"] = state.quality_dict()
        return response

    def history(self, instrument: str, direction: str, days: int = 60) -> dict[str, Any]:
        instrument = instrument.upper()
        state = self.state_cache.get(instrument)
        if not state:
            perf = self.pattern_db.get_regime_performance(instrument, days=days)
            recent = self.pattern_db.get_recent_signals(instrument, limit=10)
            return {
                "ok": True,
                "regime_performance": perf,
                "recent_signals": [
                    {"direction": row.direction, "confidence": row.confidence, "outcome": row.outcome, "pnl_pct": row.pnl_pct}
                    for row in recent
                ],
                "note": "No live state — showing historical data only",
            }
        matcher = PatternMatcher(self.pattern_db)
        context = matcher.get_context(state, direction)
        return {
            "ok": True,
            "pattern_match": context.to_dict(),
            "regime_performance": self.pattern_db.get_regime_performance(instrument, days=days),
            "prompt_context": context.to_prompt_context(),
        }

    def dashboard(self) -> dict[str, Any]:
        states_summary = {
            instrument: {
                "net_bias": round(state.net_directional_bias, 1),
                "regime": state.regime,
                "gex": state.gex_regime,
                "age_seconds": round(state.age_seconds, 1),
            }
            for instrument, state in self.state_cache.items()
        }
        return {
            "ok": True,
            "states": states_summary,
            "active_trades": self.outcome_tracker.get_active_summary(),
            "today": self.pattern_db.get_daily_summary(),
            "pattern_memory_total": self.pattern_db.count_records(),
        }

    def health(self) -> dict[str, Any]:
        """Return the exact stable v1 health fields."""
        import time
        return {
            "status": "alive",
            "brain": "SignalsBrain v1.0 — God Mode",
            "pattern_memory_records": self.pattern_db.count_records(),
            "cached_states": len(self.state_cache),
            "active_trades": self.outcome_tracker.active_count,
            "time": time.time(),
        }

    def health_v2(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "alive",
            "runtime": "SignalsBrain v2",
            "schema": self.pattern_db.get_schema_metadata(),
            "pattern_memory_records": self.pattern_db.count_records(),
            "cached_states": len(self.state_cache),
            "active_trades": self.outcome_tracker.active_count,
            "learning": self.self_improve.get_summary(),
        }
