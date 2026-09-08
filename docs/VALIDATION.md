# Validation Harness

`brain/validation` exists so this system can be judged on evidence rather than
assertion. Before it, the only self-reported number was a win rate, and strategy changes
were justified by ad-hoc scripts measuring spot direction.

## The two problems it fixes

**1. The target variable was wrong.** Every earlier backtest asked "after a BUY signal,
did the index go up?" The desk does not buy the index — it buys a call or a put, and an
option's P&L is direction *minus theta minus spread*. A strategy can be right about
direction 54% of the time and still lose steadily.

`option_backtest.py` replays the **actual historical premium** of the exact contract the
engine would have bought, via the broker's NFO history (`option_candles`). Decay is not
modelled; it is present in the data.

**2. The statistics were wrong.** Trying many variants and reporting the best inflates
the winner. During this engine's development roughly a dozen variants were tried
(momentum, mean-reversion, pure inversion, six ADX bands, four conviction buckets) and
the best was kept — textbook selection bias.

`metrics.deflated_sharpe_ratio` corrects an observed Sharpe for the number of trials,
and `splits.combinatorial_purged_cv` replaces one lucky path with a distribution.

## Modules

| Module | Purpose |
|---|---|
| `metrics.py` | Sharpe, Sortino, max drawdown, profit factor, expectancy, PSR, **deflated Sharpe**, **PBO** |
| `splits.py` | Walk-forward, purged K-fold with embargo, **CPCV**, single-use holdout |
| `data.py` | Cached client for the production proxy (spot candles, `option_candles`, GEX) |
| `engine_port.py` | Line-by-line Python port of the deployed `live-engine.js` |
| `option_backtest.py` | Premium-level backtester matching live desk rules |
| `runner.py` | Orchestration + CLI report |

## Usage

```bash
python -m brain.validation.runner --symbols BANKNIFTY NIFTY --days 30 --trials 12
```

`--trials` is not decoration. It must reflect how many variants were tried to arrive at
the current rules, because it sets the bar the deflated Sharpe has to clear.

## Why the engine is ported rather than reimplemented

`engine_port.py` reproduces `live-engine.js` including its Wilder-smoothing
initialisation and its `|| 1` divide-by-zero guards. Validating a *reimplementation*
would prove nothing about production — any difference between the two would surface as
fake alpha. `tests/test_engine_parity.py` runs both implementations over identical
synthetic series and requires indicators to agree to 1e-9 and direction/veto decisions
to agree exactly. Edit one engine without the other and the suite fails.

## Results as of 2026-09-08

Mean-reversion engine (`MR v2`: ADX ≥ 25 gate, conviction window 30–50), 30-day window,
4 indices, real option premiums, `--trials 12`:

| Symbol | Trades | Win | Expectancy | PF | Deflated Sharpe |
|---|---|---|---|---|---|
| MIDCPNIFTY | 13 | 76.9% | +4.22% | 5.56 | 0.74 |
| FINNIFTY | 10 | 70.0% | +3.16% | 3.04 | 0.44 |
| BANKNIFTY | 9 | 66.7% | +1.83% | 1.94 | 0.27 |
| NIFTY | 8 | 50.0% | **−3.23%** | 0.70 | 0.02 |
| **Pooled** | **40** | **67.5%** | **+1.98%** | **1.61** | **0.55** |

**Verdict: NOT DISTINGUISHABLE FROM LUCK.** Expectancy is positive on three of four
indices, but the pooled deflated Sharpe of 0.55 is far below the 0.95 needed, on only 40
trades. This is a promising hypothesis, not a proven edge.

The NIFTY row is the clearest lesson available: a **51.4% spot hit rate produced a
−34.8% option book**. Direction slightly better than a coin flip is not enough to beat
decay.

## Limitations — read before quoting any number

* **Expiry mismatch.** `run()` prices every signal against one expiry (normally the
  current near-dated one). Signals older than that contract's life decay more slowly
  than reality, so results are an **optimistic bound**. Keep windows within roughly one
  expiry's life. Fixing this needs a historical expiry calendar the broker does not
  expose.
* **Option data gaps.** Some signals cannot be priced (FINNIFTY 25, MIDCPNIFTY 47 skips
  in the run above), which shrinks samples further.
* **Chain volume cannot be backtested.** The GEX endpoint keeps no history, which is
  exactly why production measures option volume as a liquidity gate and deliberately
  excludes it from the directional score. The live site now logs it on every signal
  (`option_volume_*` in the brain payload) to build that history.
* **Sample sizes are small.** 40 trades cannot separate skill from luck. Published
  replication work suggests a strategy needs roughly `(1.96 / Sharpe)²` years to prove
  itself.

## Running the tests

```bash
python -m pytest tests/ -q
```

62 tests cover the metrics (including that deflated Sharpe *falls* as trials rise, and
that an 80%-win-rate losing strategy is correctly labelled as losing), leakage-freedom
for every split type, and JS↔Python engine parity.
