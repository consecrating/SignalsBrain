# What in this repo is actually running

Written because the README reads as though this is live infrastructure serving the trading
site. It is not, and the gap was large enough to mislead. Nothing here is deleted — this
records what is wired, what is dormant, and what it would take to change that, so the choice
can be made deliberately.

Verified by searching the website repo (`Signals-Live-Website`) for any reference to a brain
host, `/brain/*` endpoint, or port 8400. There are none in any `.php`, `.js` or `.html` file,
and no deployment target exists for the `Dockerfile`.

## In use

| Area | Lines | How it is used |
|---|---:|---|
| `brain/validation/` | 1,684 | **Offline only.** Produces `validation-summary.json`, which is committed into the website and read by `god-dashboard.html`. This is the sole coupling between the two repos, and it is a checked-in file, not a call. |
| `tests/` | 516 | `test_engine_parity.py` pins the Python engine port against the live JS engine; `test_validation_splits.py` covers the CPCV splitting. Run manually. |

That validation harness is the most valuable thing in this repo. It produced the number the
whole system is currently judged by: pooled deflated Sharpe **0.55** against the 0.95 needed,
on 40 option trades — i.e. positive expectancy that is still indistinguishable from luck.

Regenerate after any engine change:

```bash
pip install -e .          # needs numpy, scipy
python -m brain.validation.runner --symbols BANKNIFTY NIFTY FINNIFTY MIDCPNIFTY --days 30 --trials 12
```

A stale `validation-summary.json` is worse than none, which is why the dashboard displays its
age.

Verified working as of 2026-09-10: `python -m pytest tests/ -q` → **47 passed, 16 skipped**
(the skips need network or a node binary). So the live-and-useful part of this repo is in good
order; it is the undeployed remainder below that is the open question.

## Dormant

Reachable from nothing. No HTTP caller, no scheduler, no import from the website.

| Area | Lines | Notes |
|---|---:|---|
| `brain/reasoning/` | 1,483 | `engine.py`, `blunder_guard.py`, `confidence_calc.py`, `evidence_chain.py`. Imported only by each other. |
| `brain/memory/` | 1,331 | `pattern_db.py` expects `data/patterns.db`, **which does not exist**. The live pattern store is `brain-data/god-state.json`, written by the browser and by `api/cron.php`. |
| `brain/state/` | 1,271 | 47-dimension state builder. The live equivalent is `_buildDimensions()` in `js/god-mode.js`, which populates about 10 of the 47. |
| `brain/godmode/` | 1,163 | `orchestrator.py`, `multi_model.py`, `self_improve.py`. `self_improve.py` writes `data/learnings.json`, which does not exist. |
| `brain/connectors/` | 587 | MCP server + model schemas. |
| `api/` | 408 | FastAPI app. `EXPOSE 8400` in the Dockerfile has no deployment behind it. |

Roughly **6,200 lines** that do not execute.

Important: the live God Mode is a **separate implementation** in
`Signals-Live-Website/js/god-mode.js`. It is not a client of this service and never was.
Where the two disagree, the JavaScript is what trades.

## What deploying this would actually require

Not just starting the container. In rough order of effort:

1. **A host.** The website is PHP on shared LiteSpeed hosting (`/home4/sanctqeo/...`) with no
   Python service capability. This needs a separate VPS or container platform, plus TLS and a
   reachable hostname.
2. **Reconciling two divergent brains.** `js/god-mode.js` is the live decision-maker. Pointing
   the site at the Python service would change trading behaviour, so they would first have to
   be compared the way `tools/engine-parity.php` compares the two engines — otherwise the
   deployment silently alters what gets traded.
3. **Migrating the pattern store.** `pattern_db.py` expects SQLite at `data/patterns.db`; the
   real data is JSON in `god-state.json` (425 patterns, of which only a handful carry
   outcomes). A migration and a single source of truth are needed, or the two stores will
   drift and calibration will differ depending on which one answered.
4. **Authentication.** `api/main.py` has no auth story compatible with the site's
   same-origin write-token scheme. An unauthenticated brain endpoint that accepts outcomes
   is a data-integrity hole.
5. **Not being needed yet.** The learning loop only recently started closing, and calibration
   needs 20 completed outcomes before it does anything at all. Multi-model consensus and a
   self-improvement loop cannot be evaluated against 6 labelled trades — they would be tuned
   against noise.

## Recommendation

Leave it dormant for now and keep `brain/validation/` maintained, because points 2 and 5
matter more than the deployment itself: there is no evidence yet that a second brain would
decide better than the one already running, and no way to measure it until many more outcomes
accumulate.

Revisit when the pattern store holds enough labelled outcomes for calibration to be live
(≥20, ideally far more). At that point the sensible first step is not the whole service but
`brain/validation/` plus `self_improve.py` run **offline** against real recorded outcomes — a
measurement, not a live dependency.

If the decision is instead to retire it, delete `brain/godmode/`, `brain/reasoning/`,
`brain/state/`, `brain/connectors/` and `api/`, and keep `brain/validation/` and `tests/`.
That removes about 5,000 lines and costs nothing that currently runs.
