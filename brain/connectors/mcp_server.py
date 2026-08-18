"""
SignalsBrain — MCP Server

Model Context Protocol server for Claude Desktop / Kiro.
Exposes the brain's tools natively via MCP so Claude can use them directly.

Run with: python -m brain.connectors.mcp_server
Or add to Claude Desktop's MCP config:
{
  "mcpServers": {
    "signalsbrain": {
      "command": "python",
      "args": ["-m", "brain.connectors.mcp_server"],
      "cwd": "/path/to/SignalsBrain"
    }
  }
}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# MCP SDK import (optional dependency)
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from .schemas import MCP_TOOLS

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_mcp_server():
    """Create and configure the MCP server with SignalsBrain tools."""
    if not HAS_MCP:
        raise ImportError("MCP SDK not installed. Run: pip install mcp")
    
    server = Server("signalsbrain")
    
    @server.list_tools()
    async def list_tools():
        """List all available brain tools."""
        tools = []
        for t in MCP_TOOLS:
            tools.append(Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            ))
        return tools
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        """Execute a brain tool and return results."""
        # Import here to avoid circular deps at module level
        from brain.memory.pattern_db import PatternDB
        from brain.reasoning.engine import ReasoningEngine
        from brain.state.state_builder import StateBuilder
        from brain.state.market_state import MarketState
        from brain.memory.matcher import PatternMatcher
        
        db_path = Path(__file__).parent.parent.parent / "data" / "patterns.db"
        db = PatternDB(db_path)
        engine = ReasoningEngine(pattern_db=db)
        
        # Note: MCP server runs locally so we use the file-based state cache
        # or fetch from the REST API server if it's running.
        # For now, provide schema-based responses that guide the model.
        
        if name == "signalsbrain_analyze":
            instrument = arguments.get("instrument", "NIFTY").upper()
            return [TextContent(
                type="text",
                text=json.dumps({
                    "note": f"Call the SignalsBrain REST API at POST /brain/analyze with instrument={instrument}",
                    "endpoint": f"http://localhost:8400/brain/analyze",
                    "body": {"instrument": instrument},
                    "alternative": "If the server isn't running, use the brain's Python API directly.",
                }, indent=2)
            )]
        
        elif name == "signalsbrain_signal":
            instrument = arguments.get("instrument", "NIFTY").upper()
            threshold = arguments.get("confidence_threshold", 60)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "note": f"Call POST /brain/signal with instrument={instrument}, threshold={threshold}",
                    "endpoint": "http://localhost:8400/brain/signal",
                    "body": {"instrument": instrument, "confidence_threshold": threshold},
                }, indent=2)
            )]
        
        elif name == "signalsbrain_ask":
            question = arguments.get("question", "")
            instrument = arguments.get("instrument", "NIFTY")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "note": "Natural language query to SignalsBrain",
                    "endpoint": "http://localhost:8400/brain/ask",
                    "body": {"question": question, "instrument": instrument},
                }, indent=2)
            )]
        
        elif name == "signalsbrain_history":
            instrument = arguments.get("instrument", "NIFTY").upper()
            direction = arguments.get("direction", "SELL")
            
            # This one we can answer directly from the DB
            matcher = PatternMatcher(db)
            # Need a state — create a minimal one
            state = MarketState(instrument=instrument)
            ctx = matcher.get_context(state, direction)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "instrument": instrument,
                    "direction": direction,
                    "pattern_match": ctx.to_dict(),
                    "prompt_context": ctx.to_prompt_context(),
                }, indent=2)
            )]
        
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
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
