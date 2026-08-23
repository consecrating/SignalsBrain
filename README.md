# SignalsBrain

SignalsBrain is an auditable market-state reasoning service for Indian F&O analysis. It ingests market data, derives a declared 47-dimension state model from the inputs that are actually available, produces deterministic evidence chains and safety vetoes, records signal outcomes in SQLite, and exposes the same application runtime through REST, Python, GodMode composition, and MCP.

SignalsBrain does **not** connect to a broker or place orders. External-model consensus is advisory and cannot override deterministic safety vetoes.

## Capabilities

- **State ingestion:** derives price, trend, momentum, options, flow, volatility, and session dimensions from supplied data.
- **State quality:** reports populated versus declared dimensions, category coverage, freshness, and missing dimensions. Missing inputs reduce composite influence and decision confidence.
- **Reasoning:** emits BUY, SELL, or NO_TRADE with evidence, confidence breakdown, risk scenarios, timing, and hard/soft safety vetoes.
- **Pattern memory:** stores signals and explicit lifecycle outcomes in SQLite and calculates pattern/regime statistics from completed records.
- **Durable audit data:** stores schema metadata, state snapshots, decisions, idempotency responses, and audit events through additive migrations.
- **Learning:** records outcome-driven learning history and context-specific factor adjustments using atomic local writes.
- **Model integration:** publishes tool schemas and optional multi-model consensus. External API latency depends on the configured provider and network.

## Install and run

Python 3.11 or newer is required.

```bash
pip install -e .
export SIGNALSBRAIN_ENV=development  # explicit local-only frictionless mode
signalsbrain
# Equivalent local bind: uvicorn api.main:app --host 127.0.0.1 --port 8400
```

The stable CLI entry point starts one worker on loopback port `8400` by default. Override the port with `PORT` or the bind address with `SIGNALSBRAIN_HOST`. Omitting `SIGNALSBRAIN_ENV` is production mode and startup fails closed until production credentials are configured.

```bash
curl http://localhost:8400/brain/health
```

## Security modes

SignalsBrain defaults to fail-closed production mode. Frictionless behavior is available only when local development or test mode is selected explicitly:

```bash
export SIGNALSBRAIN_ENV=development
signalsbrain
```

Production requires two distinct non-default credentials: the general API key and a dedicated outcome-writer key. General routes accept `X-API-Key` or `Authorization: Bearer`. The v1 and v2 outcome routes accept only `X-Outcome-API-Key` in production. Query-string credentials are disabled in production, and CORS origins must be explicitly listed.

```bash
export SIGNALSBRAIN_ENV=production
export SIGNALSBRAIN_API_KEY='replace-with-a-long-random-general-secret'
export SIGNALSBRAIN_OUTCOME_API_KEY='replace-with-a-different-outcome-writer-secret'
export SIGNALSBRAIN_CORS_ORIGINS='https://app.example.com,https://ops.example.com'
export SIGNALSBRAIN_MAX_BODY_BYTES=1048576          # optional; defaults to 1 MiB
export SIGNALSBRAIN_DECISION_MAX_AGE_SECONDS=300    # optional; absolute decision freshness
export SIGNALSBRAIN_IDEMPOTENCY_TTL_SECONDS=86400   # optional; 60 seconds to 30 days
export SIGNALSBRAIN_HOST=0.0.0.0                    # explicit external CLI bind
signalsbrain
```

Example authenticated analysis and outcome calls:

```bash
curl -X POST http://localhost:8400/brain/v2/analyze \
  -H "X-API-Key: $SIGNALSBRAIN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"instrument":"NIFTY"}'

curl -X POST http://localhost:8400/brain/v2/outcome \
  -H "X-Outcome-API-Key: $SIGNALSBRAIN_OUTCOME_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"signal_id":123,"outcome":"WIN_T1"}'
```

## REST API

All original v1 routes, request defaults, and response fields remain available:

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/brain/ingest` | Ingest market data |
| `POST` | `/brain/analyze` | Produce a reasoning chain |
| `POST` | `/brain/signal` | Generate and persist an actionable signal |
| `POST` | `/brain/ask` | Build context for a natural-language answer |
| `GET` | `/brain/state/{instrument}` | Read current state |
| `POST` | `/brain/history` | Query pattern memory |
| `POST` | `/brain/outcome` | Close a signal and invoke learning |
| `GET` | `/brain/schemas/{type}` | Read model tool schemas |
| `GET` | `/brain/health` | Stable v1 health response |
| `GET` | `/brain/dashboard` | State, active-trade, and daily summary |

V1 keeps the same field names, defaults, response shapes, and permissive unknown-field behavior, while defensively rejecting non-finite, over-deep, oversized, or unequal OHLCV inputs. The additive v2 API provides strict typed validation and structured errors:

| Method | Endpoint | Additional behavior |
|---|---|---|
| `GET` | `/brain/v2/health` | Runtime, schema, and learning metadata |
| `POST` | `/brain/v2/ingest` | Finite, bounded, equal-length OHLCV validation; state quality |
| `POST` | `/brain/v2/analyze` | Analysis plus state quality |
| `POST` | `/brain/v2/signal` | Optional `Idempotency-Key` header or `idempotency_key` body field |
| `POST` | `/brain/v2/outcome` | Explicit missing/already-closed lifecycle errors |
| `GET` | `/brain/v2/state/{instrument}` | State plus quality metadata |

A repeated v2 signal request with the same unexpired idempotency key and payload returns the original response without inserting another signal. Reusing the key for a different payload returns `IDEMPOTENCY_CONFLICT`. Keys expire under the bounded TTL policy and are pruned opportunistically; signal/no-trade persistence commits its decision and audit record in the same transaction as the retry response.

## MCP

The four existing MCP tool names are unchanged:

- `signalsbrain_analyze`
- `signalsbrain_signal`
- `signalsbrain_ask`
- `signalsbrain_history`

Install the optional MCP extra and run the local stdio server:

```bash
pip install -e '.[mcp]'
python -m brain.connectors.mcp_server
```

The MCP server invokes the same `BrainRuntime`, validates every tool call with the shared strict models, returns stable generic errors, and refreshes valid persisted snapshots before decision calls. REST and MCP both restore snapshots for read continuity, but analysis, signals, and question reasoning reject stale or future state using the absolute configured decision-freshness threshold.

## State model

The registry declares 47 dimensions across seven categories: price structure, trend, momentum, options microstructure, volume/flow, volatility, and time context. A given snapshot may populate only a subset, depending on supplied data. `/brain/v2/ingest` and `/brain/v2/state/{instrument}` expose the exact coverage and missing-dimension list instead of implying that all dimensions are always present.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the original layer overview.

## Verification

Focused v2 checks cover repeatable migration and row preservation, signal idempotency and lifecycle closure, explicit outcome/recency statistics, state quality and derivative propagation, model parsing/weighting, and v2 request validation.

```bash
python -m compileall -q api brain
python -m pytest -q tests/test_v2_runtime.py
```

## License

Proprietary. Not for redistribution.
