"""FastAPI transport adapter for the SignalsBrain v1 and additive v2 APIs."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Path as PathParam, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from brain.connectors.auth import (
    PRODUCTION_MODES,
    get_security_mode,
    require_auth,
    require_outcome_auth,
    validate_security_configuration,
)
from brain.connectors.schemas import get_schema_for_model
from brain.connectors.validation import (
    AnalyzeRequest,
    AskRequest,
    CandleSeriesV2,
    HistoryRequest,
    InstrumentRequestV2,
    OutcomeRequest,
    OutcomeRequestV2,
    SignalRequest,
    SignalRequestV2,
    StateIngestRequest,
    StateIngestRequestV2,
)
from brain.memory.pattern_db import (
    IdempotencyConflictError,
    PatternDB,
    SignalAlreadyClosedError,
    SignalNotFoundError,
)
from brain.reasoning.engine import ReasoningEngine
from brain.runtime import (
    BrainRuntime,
    InvalidIdempotencyKeyError,
    InvalidOutcomeError,
    StateStaleError,
    StateUnavailableError,
    VALID_OUTCOMES,
)
from brain.state.market_state import MarketState
from brain.state.state_builder import StateBuilder
from brain.memory.outcome_tracker import OutcomeTracker


DB_PATH = Path(__file__).parent.parent / "data" / "patterns.db"
MAX_BODY_BYTES = int(os.environ.get("SIGNALSBRAIN_MAX_BODY_BYTES", 1_048_576))

runtime: Optional[BrainRuntime] = None
pattern_db: Optional[PatternDB] = None
reasoning_engine: Optional[ReasoningEngine] = None
state_builder: Optional[StateBuilder] = None
outcome_tracker: Optional[OutcomeTracker] = None
state_cache: dict[str, MarketState] = {}


def _bind_runtime(instance: BrainRuntime) -> None:
    """Bind legacy module globals to the unified service for import compatibility."""
    global runtime, pattern_db, reasoning_engine, state_builder, outcome_tracker, state_cache
    runtime = instance
    pattern_db = instance.pattern_db
    reasoning_engine = instance.reasoning_engine
    state_builder = instance.state_builder
    outcome_tracker = instance.outcome_tracker
    state_cache = instance.state_cache


def _ensure_initialized() -> BrainRuntime:
    validate_security_configuration()
    if runtime is None:
        _bind_runtime(BrainRuntime(DB_PATH))
    return runtime  # type: ignore[return-value]


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_security_configuration()
    _bind_runtime(BrainRuntime(DB_PATH))
    yield


def _cors_origins() -> list[str]:
    if get_security_mode() in PRODUCTION_MODES:
        return [
            origin.strip()
            for origin in os.environ.get("SIGNALSBRAIN_CORS_ORIGINS", "").split(",")
            if origin.strip() and origin.strip() != "*"
        ]
    return ["*"]


class BodySizeLimitMiddleware:
    """Reject oversized HTTP bodies while consuming ASGI chunks, before buffering."""

    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    @staticmethod
    def _response(path: str, status: int, code: str, message: str) -> JSONResponse:
        if path.startswith("/brain/v2/"):
            return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})
        return JSONResponse(status_code=status, content={"detail": message})

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_bytes:
                    response = self._response(
                        scope.get("path", ""), 413, "PAYLOAD_TOO_LARGE",
                        "Request body exceeds configured limit",
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                response = self._response(
                    scope.get("path", ""), 400, "INVALID_CONTENT_LENGTH",
                    "Invalid Content-Length header",
                )
                await response(scope, receive, send)
                return

        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message.get("type") == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.max_body_bytes:
                    raise _PayloadTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _PayloadTooLarge:
            response = self._response(
                scope.get("path", ""), 413, "PAYLOAD_TOO_LARGE",
                "Request body exceeds configured limit",
            )
            await response(scope, receive, send)


class _PayloadTooLarge(Exception):
    pass


app = FastAPI(
    title="SignalsBrain — God Mode",
    description="Auditable market-state reasoning and signal lifecycle for Indian F&O.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-API-Key", "X-Outcome-API-Key"],
)
app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=MAX_BODY_BYTES)


def _structured_error(status: int, code: str, message: str, field: Optional[str] = None) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if field:
        error["field"] = field
    return JSONResponse(status_code=status, content={"error": error})


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    if not request.url.path.startswith("/brain/v2/"):
        return await http_exception_handler(request, exc)
    code = "AUTHENTICATION_FAILED" if exc.status_code in (401, 403) else f"HTTP_{exc.status_code}"
    return _structured_error(exc.status_code, code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    if not request.url.path.startswith("/brain/v2/"):
        return await request_validation_exception_handler(request, exc)
    first = exc.errors()[0] if exc.errors() else {}
    location = [str(part) for part in first.get("loc", ()) if part not in ("body",)]
    return _structured_error(
        422,
        "VALIDATION_ERROR",
        first.get("msg", "Request validation failed"),
        ".".join(location) or None,
    )


@app.middleware("http")
async def runtime_middleware(request: Request, call_next):
    _ensure_initialized()
    return await call_next(request)


# Request models are shared with MCP where contracts overlap. Imported names
# remain available from this module for v1 integration compatibility.


# ── Stable v1 routes ──────────────────────────────────────────────────────────

@app.get("/brain/health")
async def health():
    return _ensure_initialized().health()


@app.get("/brain/schemas/{model_type}")
async def get_schemas(model_type: str):
    valid = ["openai", "anthropic", "gemini", "mcp", "grok", "openrouter"]
    if model_type not in valid:
        raise HTTPException(400, f"Invalid model_type. Use one of: {valid}")
    return {"model_type": model_type, "tools": get_schema_for_model(model_type)}


@app.post("/brain/ingest")
async def ingest_state(req: StateIngestRequest, _=Depends(require_auth)):
    return _ensure_initialized().ingest(
        instrument=req.instrument,
        candles=req.candles,
        gex_data=req.gex_data,
        fii_dii=req.fii_dii,
        vix=req.vix,
        htf_candles=req.htf_candles,
    )


@app.post("/brain/analyze")
async def analyze_market(req: AnalyzeRequest, _=Depends(require_auth)):
    try:
        return _ensure_initialized().analyze(req.instrument)
    except StateStaleError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/brain/signal")
async def generate_signal(req: SignalRequest, _=Depends(require_auth)):
    try:
        return _ensure_initialized().create_signal(
            req.instrument,
            confidence_threshold=req.confidence_threshold,
        )
    except (StateUnavailableError, StateStaleError) as exc:
        raise HTTPException(503, f"No fresh state for {req.instrument.upper()}") from exc


@app.post("/brain/ask")
async def ask_brain(req: AskRequest, _=Depends(require_auth)):
    return _ensure_initialized().ask(req.question, req.instrument)


@app.get("/brain/state/{instrument}")
async def get_state(instrument: str, compact: bool = True, _=Depends(require_auth)):
    try:
        return _ensure_initialized().state_response(instrument, compact=compact)
    except StateUnavailableError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/brain/history")
async def query_history(req: HistoryRequest, _=Depends(require_auth)):
    return _ensure_initialized().history(req.instrument, req.direction, req.days)


@app.post("/brain/outcome")
async def record_outcome(req: OutcomeRequest, _=Depends(require_outcome_auth)):
    if req.outcome not in VALID_OUTCOMES:
        raise HTTPException(400, f"Invalid outcome. Use: {list(VALID_OUTCOMES)}")
    try:
        return _ensure_initialized().record_outcome(
            signal_id=req.signal_id,
            outcome=req.outcome,
            exit_spot=req.exit_spot,
            exit_premium=req.exit_premium,
            pnl_pct=req.pnl_pct,
        )
    except SignalNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except SignalAlreadyClosedError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/brain/dashboard")
async def dashboard(_=Depends(require_auth)):
    return _ensure_initialized().dashboard()


# ── Additive typed v2 routes ──────────────────────────────────────────────────

@app.get("/brain/v2/health")
async def health_v2():
    return _ensure_initialized().health_v2()


@app.post("/brain/v2/ingest")
async def ingest_state_v2(req: StateIngestRequestV2, _=Depends(require_auth)):
    service = _ensure_initialized()
    result = service.ingest(
        instrument=req.instrument,
        candles=req.candles.model_dump() if req.candles else None,
        gex_data=req.gex_data,
        fii_dii=req.fii_dii,
        vix=req.vix,
        htf_candles=req.htf_candles.model_dump() if req.htf_candles else None,
    )
    result["quality"] = service.get_state_object(req.instrument).quality_dict()
    return result


@app.post("/brain/v2/analyze")
async def analyze_market_v2(req: InstrumentRequestV2, _=Depends(require_auth)):
    try:
        service = _ensure_initialized()
        result = service.analyze(req.instrument)
        result["quality"] = service.get_state_object(req.instrument).quality_dict()
        return result
    except (StateUnavailableError, StateStaleError) as exc:
        return _structured_error(503, "STATE_UNAVAILABLE", str(exc), "instrument")


@app.post("/brain/v2/signal")
async def generate_signal_v2(
    req: SignalRequestV2,
    idempotency_header: Optional[str] = Header(default=None, alias="Idempotency-Key", max_length=128),
    _=Depends(require_auth),
):
    key = idempotency_header or req.idempotency_key
    try:
        return _ensure_initialized().create_signal(
            req.instrument,
            confidence_threshold=req.confidence_threshold,
            idempotency_key=key,
        )
    except (StateUnavailableError, StateStaleError) as exc:
        return _structured_error(503, "STATE_UNAVAILABLE", str(exc), "instrument")
    except IdempotencyConflictError as exc:
        return _structured_error(409, "IDEMPOTENCY_CONFLICT", str(exc), "idempotency_key")
    except InvalidIdempotencyKeyError as exc:
        return _structured_error(422, "VALIDATION_ERROR", str(exc), "idempotency_key")


@app.post("/brain/v2/outcome")
async def record_outcome_v2(req: OutcomeRequestV2, _=Depends(require_outcome_auth)):
    try:
        return _ensure_initialized().record_outcome(**req.model_dump())
    except SignalNotFoundError as exc:
        return _structured_error(404, "SIGNAL_NOT_FOUND", str(exc), "signal_id")
    except SignalAlreadyClosedError as exc:
        return _structured_error(409, "SIGNAL_ALREADY_CLOSED", str(exc), "signal_id")
    except InvalidOutcomeError as exc:
        return _structured_error(400, "INVALID_OUTCOME", str(exc), "outcome")


@app.get("/brain/v2/state/{instrument}")
async def get_state_v2(
    instrument: Annotated[str, PathParam(min_length=1, max_length=32)],
    compact: bool = True,
    _=Depends(require_auth),
):
    try:
        return _ensure_initialized().state_response(
            instrument, compact=compact, include_quality=True
        )
    except StateUnavailableError as exc:
        return _structured_error(404, "STATE_NOT_FOUND", str(exc), "instrument")


# ── CLI entry point (stable) ──────────────────────────────────────────────────

def run():
    import uvicorn
    validate_security_configuration()
    port = int(os.environ.get("PORT", 8400))
    host = os.environ.get("SIGNALSBRAIN_HOST", "127.0.0.1")
    uvicorn.run("api.main:app", host=host, port=port, reload=False, workers=1)


if __name__ == "__main__":
    run()
