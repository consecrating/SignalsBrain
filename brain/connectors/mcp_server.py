"""Bounded MCP transport for the unified SignalsBrain runtime."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from brain.runtime import BrainRuntime, StateStaleError, StateUnavailableError

from .schemas import MCP_TOOLS
from .validation import MCP_REQUEST_MODELS


logger = logging.getLogger(__name__)


def _error(code: str, message: str, *, field: Optional[str] = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    if field:
        payload["field"] = field
    return {"ok": False, "error": payload}


def dispatch_mcp_tool(runtime: BrainRuntime, name: str, arguments: Any) -> dict[str, Any]:
    """Validate and dispatch one MCP call without depending on the optional SDK."""
    model = MCP_REQUEST_MODELS.get(name)
    if model is None:
        return _error("UNKNOWN_TOOL", "Unknown SignalsBrain tool")
    try:
        parsed = model.model_validate(arguments).model_dump()
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ())) or None
        return _error("INVALID_ARGUMENTS", "Invalid tool arguments", field=location)

    try:
        if name in {"signalsbrain_analyze", "signalsbrain_signal"}:
            runtime.refresh_snapshot(parsed["instrument"])
        if name == "signalsbrain_analyze":
            return runtime.analyze(parsed["instrument"])
        if name == "signalsbrain_signal":
            return runtime.create_signal(
                parsed["instrument"],
                confidence_threshold=parsed["confidence_threshold"],
                idempotency_key=parsed["idempotency_key"],
            )
        if name == "signalsbrain_ask":
            return runtime.ask(parsed["question"], parsed["instrument"])
        return runtime.history(parsed["instrument"], parsed["direction"], parsed["days"])
    except (StateUnavailableError, StateStaleError):
        return _error(
            "STATE_UNAVAILABLE",
            "Fresh market state is unavailable; ingest current data before requesting a decision",
        )
    except Exception:
        logger.exception("SignalsBrain MCP tool failed: %s", name)
        return _error("RUNTIME_ERROR", "SignalsBrain could not complete the request")


def create_mcp_server(*, runtime: Optional[BrainRuntime] = None, db_path: Optional[Path] = None):
    """Create the MCP server while preserving all published tool names."""
    if not HAS_MCP:
        raise ImportError("MCP SDK not installed. Run: pip install mcp")

    selected_path = db_path or Path(__file__).parent.parent.parent / "data" / "patterns.db"
    local_runtime = runtime or BrainRuntime(selected_path, restore_snapshots=True)
    server = Server("signalsbrain")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(name=item["name"], description=item["description"], inputSchema=item["inputSchema"])
            for item in MCP_TOOLS
        ]

    def content(payload: dict[str, Any]) -> list:
        return [TextContent(type="text", text=json.dumps(payload, indent=2, sort_keys=True))]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        return content(dispatch_mcp_tool(local_runtime, name, arguments))

    return server


async def main():
    """Run the MCP server via stdio."""
    if not HAS_MCP:
        print("ERROR: MCP SDK not installed. Run: pip install 'signalsbrain[mcp]'", file=sys.stderr)
        sys.exit(1)

    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
