# SignalsBrain — God Mode Architecture

## What This Is

A reasoning intelligence layer for Indian F&O markets that no human — not even an expert
with 20 years experience — can replicate. It simultaneously processes 47+ data dimensions,
maintains pattern memory across thousands of historical scenarios, and produces trade
decisions with full evidence chains.

Any AI model (ChatGPT, Claude, Gemini, Grok, Codex) can connect to it and instantly
operate at institutional quant-desk level.

---

## The 5 Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 5: GOD MODE                         │
│  Multi-model orchestrator · Consensus engine · Conflict      │
│  resolution · Self-improvement loop · Meta-reasoning         │
├─────────────────────────────────────────────────────────────┤
│                 LAYER 4: MODEL CONNECTOR                     │
│  OpenAI functions · Anthropic tools · Gemini · Grok          │
│  MCP Server · Universal schema · Auth · Rate limiting        │
├─────────────────────────────────────────────────────────────┤
│               LAYER 3: REASONING ENGINE                      │
│  Evidence chain · Bayesian confidence · Regime detection     │
│  Blunder prevention · Scenario simulation · Decision trees   │
├─────────────────────────────────────────────────────────────┤
│               LAYER 2: PATTERN MEMORY                        │
│  Historical signal outcomes · Setup fingerprints             │
│  Win/loss database · Regime-specific statistics              │
│  "What happened last 50 times?" queries                     │
├─────────────────────────────────────────────────────────────┤
│               LAYER 1: MARKET STATE                          │
│  Real-time 47-dimension state vector                        │
│  GEX · PCR · OI velocity · IV surface · FII/DII            │
│  Multi-TF · ADX regime · Order flow · Breadth · VIX         │
│  Delta exposure · Gamma exposure · Theta decay curve         │
│  Support/Resistance · VWAP · Opening Range · Swing pivots   │
├─────────────────────────────────────────────────────────────┤
│                    DATA SOURCES                              │
│  Angel One SmartAPI · NSE (option chain, FII/DII)           │
│  Computed: GEX, OI velocity, IV term structure              │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1: MarketState (brain/state/)

A single Python object that holds EVERYTHING about the current market at any instant.
Not raw data — pre-computed, cross-referenced, and annotated.

### The 47 Dimensions:

**Price Structure (8):**
1. LTP (last traded price)
2. Day change %
3. Position in day range (0-100%)
4. Position in 20-day range (0-100%)
5. Distance from VWAP (% deviation)
6. Distance from EMA9/21/50/200
7. Opening Range status (inside/breakout/breakdown)
8. Swing S/R proximity

**Trend (7):**
9. EMA stack alignment score (-1 to +1)
10. SuperTrend direction
11. ADX value + regime (Trending/Developing/Choppy)
12. +DI / -DI differential
13. Higher-TF (1H) trend
14. Daily trend (close vs 5-day EMA)
15. Trend acceleration (is ADX rising or falling?)

**Momentum (6):**
16. RSI value
17. RSI divergence (vs price)
18. MACD histogram value
19. MACD histogram direction (expanding/contracting)
20. Rate of change (5-bar)
21. Stochastic %K/%D zone

**Options Microstructure (12):**
22. PCR (Put-Call Ratio from OI)
23. PCR velocity (change rate over last 3 scans)
24. ATM IV (implied volatility)
25. IV percentile (where is current IV vs history)
26. IV skew (put IV - call IV)
27. GEX regime (Positive/Negative Gamma)
28. GEX net value (₹ Cr)
29. GEX flip level
30. Distance to GEX flip (in ATR units)
31. Call wall (max call GEX strike)
32. Put wall (max put GEX strike)
33. Max Pain level

**Volume & Flow (6):**
34. Volume ratio (current vs 20-bar average)
35. Volume trend (accelerating/decelerating)
36. VWAP position (above/below, deviation)
37. Delivery % (if available)
38. FII net flow (₹ Cr)
39. DII net flow (₹ Cr)

**Volatility (4):**
40. India VIX value
41. VIX change % (rising/falling)
42. Bollinger Band width (squeeze detection)
43. ATR as % of price (market alive/dead)

**Time & Context (4):**
44. Time of day (IST minutes from market open)
45. Days to expiry (nearest weekly)
46. Day of week (Monday effect, expiry day)
47. Session phase (opening_15min / morning / midday / afternoon / closing_30min)

### State Transitions:
The MarketState isn't just a snapshot — it tracks VELOCITY and ACCELERATION of each
dimension. "PCR went from 0.8 to 1.2 in 10 minutes" is far more powerful than "PCR is 1.2".

---

## Layer 2: PatternMemory (brain/memory/)

Every time the engine generates a signal (or decides NOT to), the full MarketState
fingerprint is stored along with what happened next (1-hour, 4-hour, EOD outcomes).

### Fingerprint Matching:
When a new signal is being evaluated, PatternMemory answers:
- "In the last 200 times the market had this ADX+GEX+PCR combination, what was the win rate?"
- "When PCR velocity was this high with Negative Gamma, what was the average move?"
- "Has this exact trigger-level + regime setup ever failed? How often?"

### Storage:
- SQLite for portability (can run anywhere without a DB server)
- ~500 bytes per record, 200 records/day = 100KB/day = 36MB/year
- Indexed by: instrument, regime, direction, confidence_band, gex_regime, pcr_band

---

## Layer 3: ReasoningEngine (brain/reasoning/)

Not a black box. Produces a full **evidence chain** — a human-readable AND machine-parseable
explanation of every decision:

```json
{
  "decision": "SELL",
  "confidence": 78,
  "evidence_chain": [
    {"factor": "GEX_REGIME", "finding": "Negative Gamma — dealers will amplify downside", "impact": "+8 conf", "weight": "HIGH"},
    {"factor": "PCR_VELOCITY", "finding": "PCR dropped from 1.1 to 0.7 in 15 min — aggressive call buying or put unwinding", "impact": "+6 conf", "weight": "HIGH"},
    {"factor": "TREND_ALIGNMENT", "finding": "All TFs bearish (1m, 5m, 15m, 1H). ADX 32 = strong trend.", "impact": "+10 conf", "weight": "CRITICAL"},
    {"factor": "PATTERN_MATCH", "finding": "Similar setup occurred 47 times in history. Win rate: 72%. Avg move: 1.8 ATR in 90 min.", "impact": "+5 conf", "weight": "MEDIUM"},
    {"factor": "RISK_FACTOR", "finding": "VIX at 18 (elevated but not extreme). IV 16% (reasonable).", "impact": "0", "weight": "NEUTRAL"},
    {"factor": "TIME_CONTEXT", "finding": "2:15 PM. 75 min to close. Adequate time for 1 ATR move.", "impact": "0", "weight": "OK"}
  ],
  "counter_arguments": [
    {"factor": "GEX_FLIP_DISTANCE", "finding": "Spot is 1.5 ATR above flip. Regime transition possible but not imminent.", "impact": "-3 conf"},
    {"factor": "PRIOR_MOVE", "finding": "Already moved 0.8 ATR today. Some exhaustion possible.", "impact": "-2 conf"}
  ],
  "final_reasoning": "Strong bearish trend confirmed by multi-TF + Negative Gamma amplification + falling PCR velocity. Pattern memory shows 72% win rate for this setup. Counter-arguments (flip distance, prior move) are mild. SELL with 78% confidence.",
  "historical_context": {
    "similar_setups": 47,
    "win_rate": 72,
    "avg_move_atr": 1.8,
    "avg_time_to_target": "87 min",
    "worst_loss": "-1.1 ATR",
    "best_win": "+4.2 ATR"
  }
}
```

---

## Layer 4: ModelConnector (brain/connectors/)

Universal API that speaks every AI model's language:

### Endpoints:
- `POST /brain/analyze` — Full analysis of current market state
- `POST /brain/signal` — Generate a signal with evidence chain
- `POST /brain/ask` — Natural language question ("Should I buy NIFTY 24200 PE now?")
- `GET /brain/state/{instrument}` — Current 47-dimension state vector
- `GET /brain/history/{instrument}` — Pattern match results
- `POST /brain/record` — Record a trade outcome for pattern learning

### Model Schemas:
- **OpenAI Function Calling** — JSON schema for all endpoints
- **Anthropic Tool Use** — Tool definitions for Claude
- **MCP Server** — Full Model Context Protocol server for Claude Desktop/Kiro
- **Gemini** — Function declarations
- **Generic REST** — Any model can call via HTTP

### Authentication:
- API key based (simple, works everywhere)
- Optional: JWT for multi-user

---

## Layer 5: GodMode (brain/godmode/)

The orchestrator that makes this superhuman:

1. **Multi-Model Consensus** — Ask Claude + GPT + Gemini the same question.
   If 3/3 agree → high confidence. If 2/3 → moderate. If split → flag for caution.

2. **Self-Improvement Loop** — After each trade closes, compare prediction vs reality.
   Adjust weights, flag patterns that are degrading, learn new ones.

3. **Meta-Reasoning** — "The engine says SELL at 78% but the last 3 SELL signals all hit
   stop-loss. Should I reduce confidence or is this a different setup?"

4. **Regime Awareness** — Knows when its own model is unreliable (e.g., budget day,
   RBI policy, election results) and says "I don't know" instead of guessing.

5. **Capital Optimization** — Not just "buy this" but "given your capital, max lots,
   and today's existing positions, here's the optimal allocation."

---

## Tech Stack

- **Language:** Python 3.12
- **API Framework:** FastAPI (async, fast, auto-docs)
- **Database:** SQLite (pattern memory, portable)
- **Data Source:** Angel One SmartAPI (via existing proxy.php OR direct Python client)
- **Deployment:** Can run on any VPS, or as a serverless function
- **AI Integration:** httpx for calling external models, MCP SDK for Claude

---

## File Structure

```
SignalsBrain/
├── brain/
│   ├── __init__.py
│   ├── state/              # Layer 1: MarketState
│   │   ├── __init__.py
│   │   ├── market_state.py       # The 47-dimension state object
│   │   ├── state_builder.py      # Builds state from raw API data
│   │   ├── velocity_tracker.py   # Tracks rate-of-change of each dimension
│   │   └── dimensions.py         # Dimension definitions + normalizers
│   ├── memory/             # Layer 2: PatternMemory
│   │   ├── __init__.py
│   │   ├── pattern_db.py         # SQLite storage + query
│   │   ├── fingerprint.py        # Compress state into matchable fingerprint
│   │   ├── matcher.py            # Find similar historical setups
│   │   └── outcome_tracker.py    # Record what actually happened
│   ├── reasoning/          # Layer 3: ReasoningEngine
│   │   ├── __init__.py
│   │   ├── evidence_chain.py     # Build the reasoning trace
│   │   ├── confidence_calc.py    # Bayesian confidence from evidence
│   │   ├── regime_detector.py    # Identify market regime
│   │   ├── blunder_guard.py      # Hard vetoes (the 14 rules, enhanced)
│   │   └── scenario_sim.py       # "What if price moves X?" simulator
│   ├── connectors/         # Layer 4: ModelConnector
│   │   ├── __init__.py
│   │   ├── openai_schema.py      # Function calling definitions
│   │   ├── anthropic_schema.py   # Tool use definitions
│   │   ├── mcp_server.py         # MCP protocol server
│   │   ├── universal_api.py      # Generic REST endpoints
│   │   └── auth.py               # API key validation
│   └── godmode/            # Layer 5: GodMode
│       ├── __init__.py
│       ├── orchestrator.py       # Main God Mode controller
│       ├── multi_model.py        # Ask multiple AIs, consensus
│       ├── self_improve.py       # Learn from outcomes
│       └── meta_reasoning.py     # Reasoning about reasoning
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── routes.py                 # All API routes
│   └── middleware.py             # CORS, logging, rate limiting
├── data/
│   ├── patterns.db               # SQLite pattern memory
│   └── config.yaml               # API keys, instruments, settings
├── tests/
├── docs/
│   └── ARCHITECTURE.md           # This file
├── scripts/
│   ├── seed_history.py           # Backfill pattern memory from historical data
│   └── healthcheck.py            # Verify all systems operational
├── pyproject.toml
├── Dockerfile
└── README.md
```
