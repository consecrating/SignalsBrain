# SignalsBrain — God Mode

**47-dimension market intelligence for Indian F&O. Connect any AI model.**

A reasoning intelligence layer that processes what no human expert can: 47 market dimensions simultaneously, with velocity tracking, pattern memory across thousands of historical setups, and multi-model AI consensus — all in under 5 milliseconds.

---

## What It Does

| Layer | Purpose | Speed |
|-------|---------|-------|
| **MarketState** | 47-dimension real-time state with velocity + acceleration | 1ms |
| **PatternMemory** | "What happened the last 50 times this exact setup occurred?" | 2ms |
| **ReasoningEngine** | Full evidence chain: WHY a signal fires/doesn't | 1ms |
| **ModelConnector** | Universal API for GPT, Claude, Gemini, Grok, MCP | instant |
| **GodMode** | Multi-model consensus + self-improvement loop | 4ms total |

## Quick Start

```bash
# Clone
git clone https://github.com/consecrating/SignalsBrain.git
cd SignalsBrain

# Install (Python 3.11+)
pip install -e .

# Run the API server
signalsbrain
# or: uvicorn api.main:app --port 8400

# Health check
curl http://localhost:8400/brain/health
```

## Connect an AI Model

### OpenAI / GPT-4
```python
import openai

# Get the tool schemas
tools = requests.get("http://localhost:8400/brain/schemas/openai").json()["tools"]

# Use in your chat completions
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Should I buy NIFTY PE now?"}],
    tools=tools,
)
```

### Claude (Anthropic)
```python
tools = requests.get("http://localhost:8400/brain/schemas/anthropic").json()["tools"]
# Use as tool_definitions in Anthropic API
```

### Claude Desktop / Kiro (MCP)
Add to your MCP config:
```json
{
  "mcpServers": {
    "signalsbrain": {
      "command": "python",
      "args": ["-m", "brain.connectors.mcp_server"],
      "cwd": "/path/to/SignalsBrain"
    }
  }
}
```

### Gemini
```python
tools = requests.get("http://localhost:8400/brain/schemas/gemini").json()["tools"]
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/brain/ingest` | Feed market data into the brain |
| `POST` | `/brain/analyze` | Full 47-dim analysis + evidence chain |
| `POST` | `/brain/signal` | Generate BUY/SELL/NO_TRADE with reasoning |
| `POST` | `/brain/ask` | Natural language query |
| `GET` | `/brain/state/{inst}` | Current state vector |
| `POST` | `/brain/history` | Pattern memory lookup |
| `POST` | `/brain/outcome` | Record trade result (for learning) |
| `GET` | `/brain/schemas/{type}` | Get tool schemas (openai/anthropic/mcp/gemini) |
| `GET` | `/brain/health` | Health check |
| `GET` | `/brain/dashboard` | All states + active trades |

## The 47 Dimensions

**Price Structure (8):** LTP, day change, range position, VWAP deviation, EMA distance, ORB status, S/R proximity, 20d range

**Trend (7):** EMA stack, SuperTrend, ADX value/regime, DI differential, higher-TF, trend acceleration

**Momentum (6):** RSI, RSI divergence, MACD histogram/direction, RoC, Stochastic

**Options Microstructure (12):** PCR, PCR velocity, ATM IV, IV percentile, IV skew, GEX regime, GEX net, GEX flip distance, call/put wall distance, max pain, OI buildup

**Volume & Flow (6):** Volume ratio/trend, VWAP position, delivery %, FII/DII flow

**Volatility (4):** VIX, VIX change, BB width (squeeze), ATR %

**Time & Context (4):** Session minutes, DTE, day of week, session phase

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full 5-layer design.

## License

Proprietary. Not for redistribution.
