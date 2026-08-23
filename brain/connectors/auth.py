"""Deployment-mode authentication policies for REST transports."""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request


DEFAULT_DEV_KEY = "dev-key-change-me"
DEVELOPMENT_MODES = frozenset({"development", "dev", "local", "test"})
PRODUCTION_MODES = frozenset({"production", "prod"})


def get_security_mode() -> str:
    """Return the deployment mode; omitted configuration is fail-closed production."""
    return os.environ.get("SIGNALSBRAIN_ENV", "production").strip().lower()


def get_api_key() -> str:
    """Return the general API credential for the active mode."""
    return os.environ.get("SIGNALSBRAIN_API_KEY", DEFAULT_DEV_KEY)


def get_outcome_api_key() -> str:
    """Return the distinct credential authorized to close signal outcomes."""
    return os.environ.get("SIGNALSBRAIN_OUTCOME_API_KEY", DEFAULT_DEV_KEY)


def validate_security_configuration() -> None:
    """Fail closed unless production has distinct non-default credentials."""
    mode = get_security_mode()
    if mode not in DEVELOPMENT_MODES | PRODUCTION_MODES:
        raise RuntimeError(
            "SIGNALSBRAIN_ENV must be one of development, local, test, or production"
        )
    if mode in PRODUCTION_MODES:
        general = os.environ.get("SIGNALSBRAIN_API_KEY", "").strip()
        outcome = os.environ.get("SIGNALSBRAIN_OUTCOME_API_KEY", "").strip()
        if not general or general == DEFAULT_DEV_KEY:
            raise RuntimeError(
                "Production mode requires a non-default SIGNALSBRAIN_API_KEY"
            )
        if not outcome or outcome == DEFAULT_DEV_KEY:
            raise RuntimeError(
                "Production mode requires a non-default SIGNALSBRAIN_OUTCOME_API_KEY"
            )
        if secrets.compare_digest(general.encode(), outcome.encode()):
            raise RuntimeError("Production API and outcome credentials must be distinct")


def _matches(candidate: str, expected: str) -> bool:
    return bool(candidate) and secrets.compare_digest(candidate.encode(), expected.encode())


def validate_request(request: Request, *, outcome_writer: bool = False) -> bool:
    """Validate constant-time header credentials; query keys are development-only."""
    validate_security_configuration()
    mode = get_security_mode()
    expected = get_outcome_api_key() if outcome_writer else get_api_key()

    if mode in DEVELOPMENT_MODES and expected == DEFAULT_DEV_KEY:
        return True

    header = "X-Outcome-API-Key" if outcome_writer else "X-API-Key"
    if _matches(request.headers.get(header, ""), expected):
        return True

    # Preserve the historical Bearer and query transports only for the general
    # development credential. Production outcome mutation uses its dedicated header.
    if not outcome_writer:
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer ") and _matches(authorization[7:], expected):
            return True
        if mode in DEVELOPMENT_MODES and _matches(request.query_params.get("api_key", ""), expected):
            return True

    return False


async def require_auth(request: Request) -> None:
    """FastAPI dependency that enforces general API authentication."""
    if not validate_request(request):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def require_outcome_auth(request: Request) -> None:
    """FastAPI dependency reserved for durable signal outcome mutation."""
    if not validate_request(request, outcome_writer=True):
        raise HTTPException(status_code=401, detail="Invalid or missing outcome API key")
