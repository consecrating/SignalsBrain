"""
SignalsBrain — Authentication

Simple API key authentication for the brain's endpoints.
Supports:
  - Header-based: X-API-Key or Authorization: Bearer <key>
  - Query param: ?api_key=<key> (for quick testing)
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import Request, HTTPException


def get_api_key() -> str:
    """Get the configured API key from environment or config."""
    return os.environ.get("SIGNALSBRAIN_API_KEY", "dev-key-change-me")


def validate_request(request: Request) -> bool:
    """
    Validate an incoming request has a valid API key.
    Checks (in order): X-API-Key header, Authorization Bearer, query param.
    """
    expected = get_api_key()
    if expected == "dev-key-change-me":
        # Dev mode: allow all requests
        return True
    
    # Header: X-API-Key
    key = request.headers.get("X-API-Key", "")
    if key == expected:
        return True
    
    # Header: Authorization: Bearer <key>
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth[7:] == expected:
        return True
    
    # Query param (for testing only)
    key = request.query_params.get("api_key", "")
    if key == expected:
        return True
    
    return False


async def require_auth(request: Request):
    """FastAPI dependency that enforces authentication."""
    if not validate_request(request):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
