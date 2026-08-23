"""
SignalsBrain — Universal Model Schemas

Defines the tool/function schemas for every major AI model platform:
  - OpenAI Function Calling (GPT-4, GPT-4o)
  - Anthropic Tool Use (Claude)
  - Google Gemini Function Declarations
  - Grok (xAI) — OpenAI-compatible
  - OpenRouter — OpenAI-compatible

When an AI model connects to SignalsBrain, it receives these schemas
and can call the brain's functions natively in its own format.

The brain exposes 6 core functions:
  1. analyze_market    — Full 47-dimension state + evidence chain
  2. generate_signal   — BUY/SELL/NO_TRADE with confidence and reasoning
  3. ask_brain         — Natural language query about any instrument
  4. get_state         — Current state vector (compact)
  5. query_history     — Pattern memory lookup
  6. record_outcome    — Feed back a trade result for learning
"""

from __future__ import annotations

from .validation import MCP_REQUEST_MODELS


# ═══════════════════════════════════════════════════════════════════════════════
# OPENAI FUNCTION CALLING SCHEMA
# Compatible with: GPT-4, GPT-4o, GPT-4-turbo, OpenRouter, Grok
# ═══════════════════════════════════════════════════════════════════════════════

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_market",
            "description": "Get complete market analysis for an instrument: 47-dimension state, evidence chain, confidence breakdown, risk scenarios, GEX regime, and historical pattern match. This is the brain's full intelligence output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument": {
                        "type": "string",
                        "description": "Trading instrument symbol (e.g., NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, RELIANCE, HDFCBANK)",
                        "enum": ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", "TATAMOTORS", "MARUTI", "WIPRO", "HCLTECH", "ADANIENT", "BAJFINANCE", "TITAN"],
                    },
                },
                "required": ["instrument"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_signal",
            "description": "Generate a trading signal (BUY/SELL/NO_TRADE) with full evidence chain, confidence breakdown, vetoes, risk assessment, and option trade plan. This is the actionable output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument": {
                        "type": "string",
                        "description": "Trading instrument symbol",
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence to emit a signal (default 60)",
                        "default": 60,
                    },
                },
                "required": ["instrument"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_brain",
            "description": "Ask the brain a natural language question about the market, an instrument, or a trading decision. The brain uses its full state, memory, and reasoning to answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question (e.g., 'Should I buy NIFTY 24200 PE now?', 'What's the GEX regime?', 'Why didn't the signal confirm today?')",
                    },
                    "instrument": {
                        "type": "string",
                        "description": "Optional: specific instrument context for the question",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": "Get the current market state vector for an instrument. Returns 47 dimensions with normalized values, velocities, and key alerts. Compact format for quick assessment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument": {
                        "type": "string",
                        "description": "Trading instrument symbol",
                    },
                    "compact": {
                        "type": "boolean",
                        "description": "If true, returns only non-zero significant dimensions (saves tokens)",
                        "default": True,
                    },
                },
                "required": ["instrument"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_history",
            "description": "Query the brain's pattern memory. Find similar historical setups and their outcomes. Answer: 'How many times has this setup occurred? What was the win rate?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument": {
                        "type": "string",
                        "description": "Instrument to query history for",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["BUY", "SELL"],
                        "description": "Signal direction to look up",
                    },
                    "days": {
                        "type": "integer",
                        "description": "How many days of history to search (default 60)",
                        "default": 60,
                    },
                },
                "required": ["instrument", "direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_outcome",
            "description": "Record a trade outcome for pattern learning. The brain uses this to improve future predictions. Feed back: did the trade hit T1? T2? Stop loss?",
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_id": {
                        "type": "integer",
                        "description": "The signal ID returned when the signal was generated",
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["WIN_T1", "WIN_T2", "WIN_T3", "STOP_LOSS", "TIME_EXIT", "NO_ENTRY"],
                        "description": "What happened to the trade",
                    },
                    "exit_spot": {
                        "type": "number",
                        "description": "Spot price at exit",
                    },
                    "exit_premium": {
                        "type": "number",
                        "description": "Option premium at exit",
                    },
                    "pnl_pct": {
                        "type": "number",
                        "description": "P&L percentage on the premium",
                    },
                },
                "required": ["signal_id", "outcome"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# ANTHROPIC TOOL USE SCHEMA (Claude)
# ═══════════════════════════════════════════════════════════════════════════════

ANTHROPIC_TOOLS = [
    {
        "name": "analyze_market",
        "description": "Get complete market analysis for an instrument: 47-dimension state, evidence chain, confidence breakdown, risk scenarios, GEX regime, and historical pattern match. This is the brain's full intelligence output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": {
                    "type": "string",
                    "description": "Trading instrument symbol (e.g., NIFTY, BANKNIFTY)",
                },
            },
            "required": ["instrument"],
        },
    },
    {
        "name": "generate_signal",
        "description": "Generate a trading signal (BUY/SELL/NO_TRADE) with full evidence chain, confidence breakdown, vetoes, risk assessment, and option trade plan.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": {"type": "string", "description": "Trading instrument symbol"},
                "confidence_threshold": {"type": "number", "description": "Minimum confidence (default 60)"},
            },
            "required": ["instrument"],
        },
    },
    {
        "name": "ask_brain",
        "description": "Ask the brain a natural language question about the market, an instrument, or a trading decision.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Natural language question"},
                "instrument": {"type": "string", "description": "Optional instrument context"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "get_state",
        "description": "Get the current market state vector (47 dimensions) for an instrument.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": {"type": "string", "description": "Instrument symbol"},
                "compact": {"type": "boolean", "description": "Compact format (default true)"},
            },
            "required": ["instrument"],
        },
    },
    {
        "name": "query_history",
        "description": "Query pattern memory for similar historical setups and their win rates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": {"type": "string"},
                "direction": {"type": "string", "enum": ["BUY", "SELL"]},
                "days": {"type": "integer", "description": "Days of history (default 60)"},
            },
            "required": ["instrument", "direction"],
        },
    },
    {
        "name": "record_outcome",
        "description": "Record a trade outcome for pattern learning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "signal_id": {"type": "integer"},
                "outcome": {"type": "string", "enum": ["WIN_T1", "WIN_T2", "WIN_T3", "STOP_LOSS", "TIME_EXIT", "NO_ENTRY"]},
                "exit_spot": {"type": "number"},
                "exit_premium": {"type": "number"},
                "pnl_pct": {"type": "number"},
            },
            "required": ["signal_id", "outcome"],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# MCP (Model Context Protocol) TOOL DEFINITIONS — For Claude Desktop/Kiro
# ═══════════════════════════════════════════════════════════════════════════════

_MCP_DESCRIPTIONS = {
    "signalsbrain_analyze": "Full market analysis from SignalsBrain: state, evidence chain, confidence, risk, and historical pattern match.",
    "signalsbrain_signal": "Generate a BUY/SELL/NO_TRADE signal with reasoning and risk assessment.",
    "signalsbrain_ask": "Ask SignalsBrain a bounded natural-language question about Indian F&O markets.",
    "signalsbrain_history": "Query historical pattern memory, win rates, and regime performance.",
}

MCP_TOOLS = [
    {
        "name": name,
        "description": _MCP_DESCRIPTIONS[name],
        "inputSchema": model.model_json_schema(),
    }
    for name, model in MCP_REQUEST_MODELS.items()
]


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI FUNCTION DECLARATIONS
# ═══════════════════════════════════════════════════════════════════════════════

GEMINI_TOOLS = {
    "function_declarations": [
        {
            "name": "analyze_market",
            "description": "Complete market analysis with 47-dimension state, evidence chain, and reasoning.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "instrument": {"type": "STRING", "description": "Instrument symbol"},
                },
                "required": ["instrument"],
            },
        },
        {
            "name": "generate_signal",
            "description": "Generate BUY/SELL/NO_TRADE signal with evidence and trade plan.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "instrument": {"type": "STRING"},
                    "confidence_threshold": {"type": "NUMBER"},
                },
                "required": ["instrument"],
            },
        },
        {
            "name": "ask_brain",
            "description": "Natural language question about the market.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING"},
                    "instrument": {"type": "STRING"},
                },
                "required": ["question"],
            },
        },
    ],
}


def get_schema_for_model(model_type: str) -> dict | list:
    """
    Get the appropriate tool schema for a given model type.
    
    Args:
        model_type: One of 'openai', 'anthropic', 'gemini', 'mcp', 'grok', 'openrouter'
    """
    if model_type in ("openai", "grok", "openrouter"):
        return OPENAI_TOOLS
    elif model_type == "anthropic":
        return ANTHROPIC_TOOLS
    elif model_type == "gemini":
        return GEMINI_TOOLS
    elif model_type == "mcp":
        return MCP_TOOLS
    else:
        return OPENAI_TOOLS  # Default to OpenAI-compatible
