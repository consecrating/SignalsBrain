"""Shared bounded request validation for REST and MCP transports."""

from __future__ import annotations

import math
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_INSTRUMENT_LENGTH = 32
MAX_QUESTION_LENGTH = 10_000
MAX_SERIES_LENGTH = 5_000
MAX_OBJECT_FIELDS = 200
MAX_NESTING_DEPTH = 8
MAX_HISTORY_DAYS = 3_650
MAX_IDEMPOTENCY_KEY_LENGTH = 128
VALID_OUTCOMES = ("WIN_T1", "WIN_T2", "WIN_T3", "STOP_LOSS", "TIME_EXIT", "NO_ENTRY")


def validate_finite_tree(value: Any, *, depth: int = 0) -> Any:
    """Reject non-finite or pathologically large nested payloads."""
    if depth > MAX_NESTING_DEPTH:
        raise ValueError("nested payload exceeds maximum depth")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("numeric values must be finite")
    if isinstance(value, dict):
        if len(value) > MAX_OBJECT_FIELDS:
            raise ValueError("object exceeds maximum field count")
        for item in value.values():
            validate_finite_tree(item, depth=depth + 1)
    elif isinstance(value, list):
        if len(value) > MAX_SERIES_LENGTH:
            raise ValueError("array exceeds maximum length")
        for item in value:
            validate_finite_tree(item, depth=depth + 1)
    return value


def validate_candle_mapping(value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Validate legacy candle dictionaries without dropping unknown proxy fields."""
    if value is None:
        return None
    validate_finite_tree(value)
    required = ("opens", "highs", "lows", "closes")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"OHLC series missing required fields: {', '.join(missing)}")
    series_names = list(required)
    if value.get("volumes") is not None:
        series_names.append("volumes")
    lengths: set[int] = set()
    for name in series_names:
        series = value[name]
        if not isinstance(series, list) or not series:
            raise ValueError(f"{name} must be a non-empty array")
        if len(series) > MAX_SERIES_LENGTH:
            raise ValueError(f"{name} exceeds maximum length")
        if any(not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)) for item in series):
            raise ValueError(f"{name} values must be finite numbers")
        lengths.add(len(series))
    if len(lengths) != 1:
        raise ValueError("OHLCV series must have equal lengths")
    return value


class LegacyRequestModel(BaseModel):
    """Bound v1 inputs while preserving permissive unknown-field handling."""

    model_config = ConfigDict(extra="ignore")


class AnalyzeRequest(LegacyRequestModel):
    instrument: str = Field(min_length=1, max_length=MAX_INSTRUMENT_LENGTH)


class SignalRequest(AnalyzeRequest):
    confidence_threshold: float = Field(default=60, ge=0, le=100, allow_inf_nan=False)


class AskRequest(LegacyRequestModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    instrument: Optional[str] = Field(default=None, min_length=1, max_length=MAX_INSTRUMENT_LENGTH)


class HistoryRequest(AnalyzeRequest):
    direction: str = Field(pattern="^(BUY|SELL)$")
    days: int = Field(default=60, ge=1, le=MAX_HISTORY_DAYS)


class OutcomeRequest(LegacyRequestModel):
    signal_id: int = Field(gt=0)
    outcome: str
    exit_spot: float = Field(default=0, allow_inf_nan=False)
    exit_premium: float = Field(default=0, allow_inf_nan=False)
    pnl_pct: float = Field(default=0, allow_inf_nan=False)

    @field_validator("outcome")
    @classmethod
    def validate_outcome(cls, value: str) -> str:
        if value not in VALID_OUTCOMES:
            raise ValueError(f"must be one of {list(VALID_OUTCOMES)}")
        return value


class StateIngestRequest(LegacyRequestModel):
    """Stable v1 shape with bounded, finite legacy dictionaries."""

    instrument: str = Field(min_length=1, max_length=MAX_INSTRUMENT_LENGTH)
    candles: Optional[dict[str, Any]] = None
    gex_data: Optional[dict[str, Any]] = None
    fii_dii: Optional[dict[str, Any]] = None
    vix: Optional[float] = Field(default=None, allow_inf_nan=False)
    htf_candles: Optional[dict[str, Any]] = None

    @field_validator("candles", "htf_candles")
    @classmethod
    def bounded_candles(cls, value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        return validate_candle_mapping(value)

    @field_validator("gex_data", "fii_dii")
    @classmethod
    def bounded_objects(cls, value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        return validate_finite_tree(value) if value is not None else value


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class InstrumentRequestV2(StrictRequestModel):
    instrument: str = Field(min_length=1, max_length=MAX_INSTRUMENT_LENGTH)


class SignalRequestV2(InstrumentRequestV2):
    confidence_threshold: float = Field(default=60, ge=0, le=100, allow_inf_nan=False)
    idempotency_key: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=MAX_IDEMPOTENCY_KEY_LENGTH,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )


class OutcomeRequestV2(StrictRequestModel):
    signal_id: int = Field(gt=0)
    outcome: str
    exit_spot: float = Field(default=0, allow_inf_nan=False)
    exit_premium: float = Field(default=0, allow_inf_nan=False)
    pnl_pct: float = Field(default=0, allow_inf_nan=False)

    _validate_outcome = field_validator("outcome")(OutcomeRequest.validate_outcome.__func__)


class CandleSeriesV2(StrictRequestModel):
    opens: list[float] = Field(min_length=1, max_length=MAX_SERIES_LENGTH)
    highs: list[float] = Field(min_length=1, max_length=MAX_SERIES_LENGTH)
    lows: list[float] = Field(min_length=1, max_length=MAX_SERIES_LENGTH)
    closes: list[float] = Field(min_length=1, max_length=MAX_SERIES_LENGTH)
    volumes: Optional[list[float]] = Field(default=None, min_length=1, max_length=MAX_SERIES_LENGTH)

    @field_validator("opens", "highs", "lows", "closes", "volumes")
    @classmethod
    def finite_series(cls, values: Optional[list[float]]) -> Optional[list[float]]:
        if values is not None and any(not math.isfinite(value) for value in values):
            raise ValueError("series values must be finite")
        return values

    @model_validator(mode="after")
    def equal_lengths(self):
        lengths = {len(self.opens), len(self.highs), len(self.lows), len(self.closes)}
        if self.volumes is not None:
            lengths.add(len(self.volumes))
        if len(lengths) != 1:
            raise ValueError("OHLCV series must have equal lengths")
        return self


class StateIngestRequestV2(InstrumentRequestV2):
    candles: Optional[CandleSeriesV2] = None
    gex_data: Optional[dict[str, Any]] = None
    fii_dii: Optional[dict[str, Any]] = None
    vix: Optional[float] = Field(default=None, allow_inf_nan=False)
    htf_candles: Optional[CandleSeriesV2] = None

    @field_validator("gex_data", "fii_dii")
    @classmethod
    def bounded_finite_objects(cls, value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        return validate_finite_tree(value) if value is not None else value


class MCPAnalyzeRequest(InstrumentRequestV2):
    pass


class MCPSignalRequest(SignalRequestV2):
    pass


class MCPAskRequest(StrictRequestModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    instrument: str = Field(default="NIFTY", min_length=1, max_length=MAX_INSTRUMENT_LENGTH)


class MCPHistoryRequest(InstrumentRequestV2):
    direction: str = Field(pattern="^(BUY|SELL)$")
    days: int = Field(default=60, ge=1, le=MAX_HISTORY_DAYS)


MCP_REQUEST_MODELS = {
    "signalsbrain_analyze": MCPAnalyzeRequest,
    "signalsbrain_signal": MCPSignalRequest,
    "signalsbrain_ask": MCPAskRequest,
    "signalsbrain_history": MCPHistoryRequest,
}
