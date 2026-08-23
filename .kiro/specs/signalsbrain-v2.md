# SignalsBrain v2 Runtime Specification

## Objective

Introduce a production-oriented, auditable application runtime that unifies state ingestion, reasoning, signal lifecycle, outcomes, memory, and learning while retaining every existing v1 REST path, request default, response field, CLI entry point, MCP tool name, and compatible SQLite row.

## Requirements

1. **Compatibility**
   - Existing `/brain/*` routes continue to accept and return their documented v1 shapes.
   - Existing `signals` SQLite data is preserved through additive, idempotent migrations.
   - Deterministic safety vetoes remain authoritative; model consensus is advisory only.
2. **Unified runtime**
   - `BrainRuntime` owns ingest, state snapshots, reasoning, signal creation, outcome recording, dashboard/health data, and learning orchestration.
   - REST and GodMode composition delegate to this service where practical.
3. **Durability and auditability**
   - Persist schema metadata, state snapshots, decisions, idempotency records, and audit events.
   - Signal creation supports an idempotency key and outcomes reject missing/already-closed signals.
   - Pattern statistics use explicit outcome semantics and true chronological recency.
4. **State quality**
   - Expose coverage, freshness, missing dimensions, and populated/declared counts.
   - Propagate computed velocity and acceleration into dimensions.
   - Composite confidence is penalized for incomplete category coverage rather than treating partial categories as complete.
5. **Additive v2 API**
   - Add typed `/brain/v2/*` endpoints with structured errors and bounded, finite, equal-length series validation.
   - Keep v1 behavior compatible and route its business logic through `BrainRuntime`.
6. **Model and learning correctness**
   - Fix ambiguous model response parsing, use model weights in confidence synthesis, and pass the actual regime into model consensus.
   - Self-improvement writes real learning records atomically and is invoked from the unified outcome flow.
7. **Security boundary**
   - Production mode rejects default/missing credentials, uses header credentials with constant-time comparison, applies constrained CORS, and bounds payloads.
   - Explicit local development mode retains current frictionless behavior.
8. **Documentation and verification**
   - README describes real capabilities and security modes without unsupported performance claims.
   - Focused tests prove migrations, lifecycle/idempotency, state quality, and v2 validation.

## Design

`BrainRuntime` is the application-service seam. It accepts existing components through dependency injection and emits plain model-compatible objects/dictionaries. `PatternDB` remains the storage adapter and performs additive `CREATE TABLE IF NOT EXISTS`/column migrations. The FastAPI module remains the transport adapter: legacy handlers translate to runtime calls, while v2 handlers validate stricter typed contracts and map domain errors into structured HTTP errors.

New storage is append-oriented where possible:

- `schema_metadata` records schema and fingerprint versions.
- `state_snapshots` records serialized state plus quality metadata.
- `decisions` records auditable analysis results.
- `idempotency_keys` maps operation/key pairs to stable serialized responses.
- `audit_events` records lifecycle/security-relevant events.

## Non-goals

- No broker integration, order placement, or autonomous live trading.
- No model-generated override of hard safety vetoes.
- No replacement of current SQLite storage or destructive schema rebuild.
- No new third-party dependency unless unavoidable.

## Acceptance criteria

- Repository compiles and focused checks pass.
- Existing v1 routes remain present.
- New v2 health/ingest/analyze/signal/outcome/state endpoints are present and validated.
- Repeating a v2 signal request with the same idempotency key returns the original lifecycle result without duplicate persistence.
- Missing or repeated outcomes fail explicitly.
- Existing database migration runs repeatedly without data loss.
- Git diff passes whitespace and secret checks; semantic and security reviews have no unresolved critical finding.
