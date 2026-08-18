"""
SignalsBrain — FastAPI Application Entry Point

The server that exposes the brain to the world.
Any AI model (GPT, Claude, Gemini, Grok) connects here via REST API.

Endpoints:
  POST /brain/analyze         Full market analysis
  POST /brain/signal          Generate signal
  POST /brain/ask             Natural language query
  GET  /brain/state/{inst}    Current state vector
  POST /brain/history         Pattern memory lookup
  POST /brain/outcome         Record trade outcome
  GET  /brain/schemas/{type}  Get tool schemas for a model type
  GET  /brain/health          Health check
  GET  /brain/dashboard       Active states summary
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from brain.connectors.auth import require_auth, get_api_key
from brain.connectors.schemas import get_schema_for_model
from brain.memory.pattern_db import PatternDB
from brain.memory.matcher import PatternMatcher
from brain.memory.outcome_tracker import OutcomeTracker
from brain.reasoning.engine import ReasoningEngine
from brain.state.market_state import MarketState
from brain.state.state_builder import StateBuilder


# ─── Lifespan ─────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "data" / "patterns.db"

# Global instances (initialized at startup)
pattern_db: PatternDB = None
reasoning_engine: ReasoningEngine = None
state_builder: StateBuilder = None
outcome_tracker: OutcomeTracker = None

# In-memory state cache (per instrument, latest scan)
state_cache: dict[str, MarketState] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pattern_db, reasoning_engine, state_builder, outcome_tracker
    pattern_db = PatternDB(DB_PATH)
    reasoning_engine = ReasoningEngine(pattern_db=pattern_db)
    state_builder = StateBuilder()
    outcome_tracker = OutcomeTracker(pattern_db)
    yield


def _ensure_initialized():
    """Ensure globals are initialized (for TestClient which may skip lifespan)."""
    global pattern_db, reasoning_engine, state_builder, outcome_tracker
    if pattern_db is None:
        pattern_db = PatternDB(DB_PATH)
    if reasoning_engine is None:
        reasoning_engine = ReasoningEngine(pattern_db=pattern_db)
    if state_builder is None:
        state_builder = StateBuilder()
    if outcome_tracker is None:
        outcome_tracker = OutcomeTracker(pattern_db)


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SignalsBrain — God Mode",
    description="47-dimension market intelligence for Indian F&O. Connect any AI model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def ensure_init_middleware(request: Request, call_next):
    _ensure_initialized()
    return await call_next(request)


# ─── Request Models ───────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    instrument: str

class SignalRequest(BaseModel):
    instrument: str
    confidence_threshold: float = 60

class AskRequest(BaseModel):
    question: str
    instrument: Optional[str] = None

class HistoryRequest(BaseModel):
    instrument: str
    direction: str
    days: int = 60

class OutcomeRequest(BaseModel):
    signal_id: int
    outcome: str
    exit_spot: float = 0
    exit_premium: float = 0
    pnl_pct: float = 0

class StateIngestRequest(BaseModel):
    """For ingesting state from the live site's proxy.php data."""
    instrument: str
    candles: Optional[dict] = None
    gex_data: Optional[dict] = None
    fii_dii: Optional[dict] = None
    vix: Optional[float] = None
    htf_candles: Optional[dict] = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/brain/health")
async def health():
    _ensure_initialized()
    return {
        "status": "alive",
        "brain": "SignalsBrain v1.0 — God Mode",
        "pattern_memory_records": pattern_db.count_records() if pattern_db else 0,
        "cached_states": len(state_cache),
        "active_trades": outcome_tracker.active_count if outcome_tracker else 0,
        "time": time.time(),
    }


@app.get("/brain/schemas/{model_type}")
async def get_schemas(model_type: str):
    """Get tool/function schemas for a specific AI model type."""
    valid = ["openai", "anthropic", "gemini", "mcp", "grok", "openrouter"]
    if model_type not in valid:
        raise HTTPException(400, f"Invalid model_type. Use one of: {valid}")
    return {"model_type": model_type, "tools": get_schema_for_model(model_type)}


@app.post("/brain/ingest")
async def ingest_state(req: StateIngestRequest, _=Depends(require_auth)):
    """
    Ingest raw market data and build/update the brain's state.
    Called by the live site (signal.html) or a data collection service.
    """
    _ensure_initialized()
    state = state_builder.build(
        instrument=req.instrument.upper(),
        candles=req.candles,
        gex_data=req.gex_data,
        fii_dii=req.fii_dii,
        vix=req.vix,
        htf_candles=req.htf_candles,
    )
    state_cache[req.instrument.upper()] = state
    
    # Check outcome tracker for any active trades
    if req.candles and req.candles.get("closes"):
        ltp = req.candles["closes"][-1]
        outcomes = outcome_tracker.check(req.instrument.upper(), ltp)
    else:
        outcomes = []
    
    return {
        "ok": True,
        "instrument": state.instrument,
        "net_bias": round(state.net_directional_bias, 1),
        "regime": state.regime,
        "gex_regime": state.gex_regime,
        "outcomes_triggered": outcomes,
    }


@app.post("/brain/analyze")
async def analyze_market(req: AnalyzeRequest, _=Depends(require_auth)):
    """Full analysis: state + evidence chain + reasoning."""
    instrument = req.instrument.upper()
    state = state_cache.get(instrument)
    
    if not state or state.is_stale:
        raise HTTPException(
            503,
            f"No fresh state for {instrument}. Ingest data first via POST /brain/ingest"
        )
    
    chain = reasoning_engine.reason(state)
    
    return {
        "ok": True,
        "analysis": chain.to_dict(),
        "state_compact": state.to_compact(),
        "prompt": chain.to_prompt(),
    }


@app.post("/brain/signal")
async def generate_signal(req: SignalRequest, _=Depends(require_auth)):
    """Generate actionable signal."""
    instrument = req.instrument.upper()
    state = state_cache.get(instrument)
    
    if not state or state.is_stale:
        raise HTTPException(503, f"No fresh state for {instrument}")
    
    chain = reasoning_engine.reason(state, confidence_threshold=req.confidence_threshold)
    
    # Record in pattern memory
    signal_id = None
    if pattern_db and chain.direction != "NO_TRADE":
        ltp_dim = state.dimensions.get("ltp")
        ltp = ltp_dim.raw if ltp_dim else 0
        atr_dim = state.dimensions.get("atr_pct")
        atr = (atr_dim.raw / 100 * ltp) if atr_dim else ltp * 0.01
        
        signal_id = pattern_db.record_signal(
            state=state,
            direction=chain.direction,
            confidence=chain.confidence,
            entry_spot=ltp,
            atr=atr,
            evidence=chain.verdict,
        )
        
        # Start outcome tracking
        outcome_tracker.start_tracking(
            signal_id=signal_id,
            instrument=instrument,
            direction=chain.direction,
            entry_spot=ltp,
            entry_premium=0,  # Will be updated when live premium is known
            strike=0,
            opt_type="PE" if chain.direction == "SELL" else "CE",
            atr=atr,
        )
    
    return {
        "ok": True,
        "signal_id": signal_id,
        "direction": chain.direction,
        "confidence": round(chain.confidence, 1),
        "actionable": chain.actionable,
        "verdict": chain.verdict,
        "reasoning": chain.reasoning_narrative,
        "evidence": chain.to_dict()["evidence"],
        "confidence_breakdown": chain.confidence_breakdown,
        "risk": chain.to_dict()["risk"],
        "timing": chain.to_dict()["timing"],
        "vetoes": chain.vetoes,
        "historical": chain.to_dict()["historical"],
        "prompt_for_ai": chain.to_prompt(),
    }


@app.post("/brain/ask")
async def ask_brain(req: AskRequest, _=Depends(require_auth)):
    """
    Natural language query interface.
    The brain uses its current state + memory to formulate an answer context.
    The actual answer generation is done by the connected AI model using this context.
    """
    instrument = (req.instrument or "NIFTY").upper()
    state = state_cache.get(instrument)
    
    context_parts = []
    context_parts.append(f"Question: {req.question}")
    context_parts.append(f"Instrument: {instrument}")
    
    if state:
        context_parts.append(f"\nCurrent State ({instrument}):")
        context_parts.append(f"  Net bias: {state.net_directional_bias:.1f}")
        context_parts.append(f"  Regime: {state.regime} | GEX: {state.gex_regime}")
        context_parts.append(f"  Agreement: {state.agreement_factor:.0%}")
        
        # Run reasoning for full context
        chain = reasoning_engine.reason(state)
        context_parts.append(f"\nSignal Brain Assessment:")
        context_parts.append(chain.to_prompt())
    else:
        context_parts.append(f"\nNo live state available for {instrument}. Market may be closed.")
    
    # Historical context
    if pattern_db:
        daily = pattern_db.get_daily_summary()
        context_parts.append(f"\nToday's performance: {daily['wins']}W / {daily['losses']}L ({daily['win_rate']:.0f}% WR)")
    
    return {
        "ok": True,
        "context": "\n".join(context_parts),
        "state_available": state is not None,
        "note": "Use this context to answer the user's question. The brain provides data and reasoning; you provide the natural language response.",
    }


@app.get("/brain/state/{instrument}")
async def get_state(instrument: str, compact: bool = True, _=Depends(require_auth)):
    """Get current state vector."""
    instrument = instrument.upper()
    state = state_cache.get(instrument)
    
    if not state:
        raise HTTPException(404, f"No state for {instrument}")
    
    if compact:
        return {"ok": True, "state": state.to_compact(), "age_seconds": round(state.age_seconds, 1)}
    else:
        return {"ok": True, "state": state.to_dict(include_raw=True), "age_seconds": round(state.age_seconds, 1)}


@app.post("/brain/history")
async def query_history(req: HistoryRequest, _=Depends(require_auth)):
    """Query pattern memory."""
    instrument = req.instrument.upper()
    state = state_cache.get(instrument)
    
    if not state:
        # Return regime performance without live state
        perf = pattern_db.get_regime_performance(instrument, days=req.days)
        recent = pattern_db.get_recent_signals(instrument, limit=10)
        return {
            "ok": True,
            "regime_performance": perf,
            "recent_signals": [{"direction": r.direction, "confidence": r.confidence, "outcome": r.outcome, "pnl_pct": r.pnl_pct} for r in recent],
            "note": "No live state — showing historical data only",
        }
    
    matcher = PatternMatcher(pattern_db)
    ctx = matcher.get_context(state, req.direction)
    perf = pattern_db.get_regime_performance(instrument, days=req.days)
    
    return {
        "ok": True,
        "pattern_match": ctx.to_dict(),
        "regime_performance": perf,
        "prompt_context": ctx.to_prompt_context(),
    }


@app.post("/brain/outcome")
async def record_outcome(req: OutcomeRequest, _=Depends(require_auth)):
    """Record a trade outcome for learning."""
    valid_outcomes = ["WIN_T1", "WIN_T2", "WIN_T3", "STOP_LOSS", "TIME_EXIT", "NO_ENTRY"]
    if req.outcome not in valid_outcomes:
        raise HTTPException(400, f"Invalid outcome. Use: {valid_outcomes}")
    
    pattern_db.record_outcome(
        signal_id=req.signal_id,
        outcome=req.outcome,
        exit_spot=req.exit_spot,
        exit_premium=req.exit_premium,
        move_atr=0,  # Will be calculated from entry if available
        duration_min=0,
        pnl_pct=req.pnl_pct,
    )
    
    return {"ok": True, "message": f"Outcome '{req.outcome}' recorded for signal {req.signal_id}"}


@app.get("/brain/dashboard")
async def dashboard(_=Depends(require_auth)):
    """Dashboard: all active states, trades, and today's summary."""
    states_summary = {}
    for inst, state in state_cache.items():
        states_summary[inst] = {
            "net_bias": round(state.net_directional_bias, 1),
            "regime": state.regime,
            "gex": state.gex_regime,
            "age_seconds": round(state.age_seconds, 1),
        }
    
    daily = pattern_db.get_daily_summary() if pattern_db else {}
    active = outcome_tracker.get_active_summary() if outcome_tracker else []
    
    return {
        "ok": True,
        "states": states_summary,
        "active_trades": active,
        "today": daily,
        "pattern_memory_total": pattern_db.count_records() if pattern_db else 0,
    }


# ─── Run ──────────────────────────────────────────────────────────────────────
def run():
    import uvicorn
    port = int(os.environ.get("PORT", 8400))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False, workers=1)


if __name__ == "__main__":
    run()
