"""
SignalsBrain — Validation runner

Wires the pieces into one experiment:

    spot candles -> engine_port signals -> real option premiums -> metrics
                 -> walk-forward / CPCV -> deflated Sharpe -> verdict

The output is deliberately blunt. The failure mode this harness exists to prevent is
the one that already happened on this project: a variant was tuned until it looked
good, the best number was quoted, and the sample turned out to be small and the target
variable wrong. So the report always prints the trade count, the out-of-sample spread,
and the number of variants tried alongside any headline figure.

CLI:
    python -m brain.validation.runner --symbols BANKNIFTY NIFTY --days 60
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import metrics as M
from .data import SignalsDataClient
from .engine_port import Signal, generate_signal
from .option_backtest import BacktestTrade, OptionBacktester
from .splits import combinatorial_purged_cv, walk_forward


@dataclass
class ExperimentResult:
    symbol: str
    trades: list[BacktestTrade] = field(default_factory=list)
    summary: Optional[M.PerformanceSummary] = None
    walk_forward: list[dict] = field(default_factory=list)
    cpcv_sharpes: list[float] = field(default_factory=list)
    pbo: Optional[float] = None
    diagnostics: dict = field(default_factory=dict)
    spot_hit_rate: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_trades": len(self.trades),
            "summary": self.summary.to_dict() if self.summary else None,
            "walk_forward": self.walk_forward,
            "cpcv_paths": len(self.cpcv_sharpes),
            "cpcv_sharpe_mean": float(np.mean(self.cpcv_sharpes)) if self.cpcv_sharpes else None,
            "cpcv_sharpe_std": float(np.std(self.cpcv_sharpes)) if self.cpcv_sharpes else None,
            "cpcv_sharpe_min": float(np.min(self.cpcv_sharpes)) if self.cpcv_sharpes else None,
            "cpcv_sharpe_max": float(np.max(self.cpcv_sharpes)) if self.cpcv_sharpes else None,
            "pbo": self.pbo,
            "spot_hit_rate": self.spot_hit_rate,
            "diagnostics": self.diagnostics,
            "trades": [t.to_dict() for t in self.trades],
        }


def generate_signal_stream(candles, symbol: str, confidence_threshold: float = 60.0,
                           warmup: int = 50) -> tuple[list[Signal], list[str]]:
    """
    Walk the history bar by bar, showing the engine ONLY data up to each bar.

    The slicing is the whole point: it is what makes the result free of look-ahead. A
    vectorised shortcut that computed indicators on the full series first would leak the
    future into every early signal.
    """
    signals: list[Signal] = []
    for i in range(warmup, candles.count):
        sig = generate_signal(
            candles.highs[:i + 1], candles.lows[:i + 1], candles.closes[:i + 1],
            symbol=symbol, confidence_threshold=confidence_threshold,
            bar_index=i, timestamp=candles.timestamps[i],
        )
        signals.append(sig)
    return signals, candles.timestamps[warmup:candles.count]


def spot_hit_rate(signals: Sequence[Signal], closes: Sequence[float], warmup: int,
                  horizon: int = 4) -> Optional[float]:
    """
    The OLD way of measuring, kept only for contrast.

    Reporting it next to the option result makes the central lesson visible: a decent
    spot hit rate and a losing option book are entirely compatible, because this number
    ignores the decay that is charged for holding the position.
    """
    hits = n = 0
    for sig in signals:
        i = sig.bar_index
        if not sig.actionable or i + horizon >= len(closes):
            continue
        fwd = (closes[i + horizon] - closes[i]) / closes[i]
        ret = fwd if sig.direction == "BUY" else -fwd
        n += 1
        hits += 1 if ret > 0 else 0
    return (hits / n) if n else None


def run_symbol(client: SignalsDataClient, symbol: str, days: int = 60,
               confidence_threshold: float = 60.0, trials: int = 1,
               trial_sharpes: Optional[Sequence[float]] = None,
               verbose: bool = True) -> ExperimentResult:
    res = ExperimentResult(symbol=symbol)

    candles = client.spot_candles(symbol, "FIFTEEN_MINUTE", days=days)
    if not candles or candles.count < 80:
        res.diagnostics["error"] = "insufficient spot history"
        return res

    expiry = client.current_expiry(symbol)
    if not expiry:
        res.diagnostics["error"] = "could not resolve expiry"
        return res

    warmup = 50
    signals, timestamps = generate_signal_stream(candles, symbol, confidence_threshold, warmup)
    actionable = [s for s in signals if s.actionable]
    res.spot_hit_rate = spot_hit_rate(signals, candles.closes, warmup)

    bt = OptionBacktester(client, verbose=verbose)
    # option_candles history must reach back at least as far as the spot window.
    res.trades = bt.run(signals, timestamps, expiry, history_days=max(days, 30))
    res.diagnostics = bt.diagnostics()
    res.diagnostics.update({
        "bars": candles.count, "signals_evaluated": len(signals),
        "actionable_signals": len(actionable), "expiry_used": expiry,
        "spot_volume_available": candles.has_real_volume(),
    })

    if not res.trades:
        res.summary = M.summarize([], label=symbol)
        return res

    returns = [t.return_pct for t in res.trades]
    # Per-trade returns: annualise on trade count per year rather than bar count.
    trades_per_year = max(1, int(252 * len(res.trades) / max(1, days)))
    res.diagnostics["trades_per_year_estimate"] = trades_per_year
    res.summary = M.summarize(returns, periods_per_year=trades_per_year,
                              trials=trials, trial_sharpes=trial_sharpes, label=symbol)

    # Walk-forward on the realised trade sequence: is the edge stable over time, or did
    # one lucky stretch carry it?
    holds = [t.bars_held for t in res.trades]
    for sp in walk_forward(len(returns), n_splits=4, label_horizon=holds,
                           embargo_pct=0.02, min_train=max(10, len(returns) // 4)):
        seg = [returns[i] for i in sp.test]
        if len(seg) < 3:
            continue
        res.walk_forward.append({
            "fold": sp.label, "n": len(seg),
            "mean_return_pct": float(np.mean(seg) * 100),
            "win_rate": float(np.mean([1 if x > 0 else 0 for x in seg])),
            "sharpe_per_trade": M.sharpe_ratio(seg, annualised=False),
        })

    for sp in combinatorial_purged_cv(len(returns), n_groups=6, n_test_groups=2,
                                      label_horizon=holds, embargo_pct=0.02):
        seg = [returns[i] for i in sp.test]
        if len(seg) >= 3:
            res.cpcv_sharpes.append(M.sharpe_ratio(seg, annualised=False))

    return res


def print_report(results: Sequence[ExperimentResult]) -> None:
    line = "=" * 78
    print(f"\n{line}\nOPTION-PREMIUM VALIDATION — real historical premiums, theta included\n{line}")
    all_returns: list[float] = []

    for r in results:
        print(f"\n### {r.symbol}")
        d = r.diagnostics
        if d.get("error"):
            print(f"  skipped: {d['error']}")
            continue
        print(f"  bars={d.get('bars')} signals={d.get('signals_evaluated')} "
              f"actionable={d.get('actionable_signals')} expiry={d.get('expiry_used')}")
        print(f"  spot volume available: {d.get('spot_volume_available')} "
              f"(indices publish none — volume is an option-chain liquidity gate)")
        print(f"  option data skips: no-data={d.get('skipped_no_option_data')} "
              f"too-late={d.get('skipped_too_late_in_session')}")
        print(f"  CAVEAT: all trades priced on expiry {d.get('expiry_used')}; signals older "
              f"than that contract's life decay slower than reality, so results are an "
              f"OPTIMISTIC bound")
        if r.spot_hit_rate is not None:
            print(f"  OLD METRIC  spot direction hit rate : {r.spot_hit_rate * 100:.1f}%")

        s = r.summary
        if not s or s.n == 0:
            print("  no completed option trades — cannot judge")
            continue
        all_returns.extend(t.return_pct for t in r.trades)
        print(f"  REAL METRIC option trades              : {s.n}")
        print(f"    win rate            {s.win_rate * 100:.1f}%   "
              f"(avg win {s.avg_win * 100:+.1f}% / avg loss {s.avg_loss * 100:+.1f}%)")
        print(f"    expectancy          {s.expectancy * 100:+.2f}% per trade")
        print(f"    profit factor       {s.profit_factor:.2f}")
        print(f"    total return        {s.total_return * 100:+.1f}%")
        print(f"    max drawdown        {s.max_drawdown * 100:.1f}%")
        print(f"    Sharpe (annual)     {s.sharpe:.2f}   Sortino {s.sortino:.2f}")
        print(f"    t-stat {s.t_stat:+.2f} (p={s.p_value:.3f})   PSR {s.psr:.2f}")
        if s.deflated_sharpe is not None:
            print(f"    deflated Sharpe     {s.deflated_sharpe:.2f}  "
                  f"(after {s.trials} variants tried)")
        outcomes: dict[str, int] = {}
        for t in r.trades:
            outcomes[t.outcome] = outcomes.get(t.outcome, 0) + 1
        print(f"    exits               {outcomes}")
        if r.walk_forward:
            print("    walk-forward (out-of-sample folds):")
            for f in r.walk_forward:
                print(f"      {f['fold']:>10}  n={f['n']:<4} mean {f['mean_return_pct']:+.2f}%  "
                      f"win {f['win_rate'] * 100:.0f}%")
        if r.cpcv_sharpes:
            arr = np.asarray(r.cpcv_sharpes)
            print(f"    CPCV {len(arr)} paths: Sharpe/trade mean {arr.mean():+.3f} "
                  f"sd {arr.std():.3f} range [{arr.min():+.3f}, {arr.max():+.3f}]")
            print(f"      share of paths profitable: {float((arr > 0).mean()) * 100:.0f}%")
        print(f"    VERDICT: {s.verdict}")

    if all_returns:
        print(f"\n{line}\nPOOLED ACROSS SYMBOLS\n{line}")
        # Annualise on TRADE frequency, not bar frequency. Using the 15-minute-bar
        # constant here treated every trade as if it were a 15-minute period and
        # inflated the pooled Sharpe roughly five-fold (39.9 against a per-symbol 8.0).
        pooled_trades_per_year = max(1, sum(
            int(r.diagnostics.get("trades_per_year_estimate", 0)) for r in results) or 252)
        pooled = M.summarize(all_returns, periods_per_year=pooled_trades_per_year,
                             trials=max(1, len(results)), label="pooled")
        print(f"  trades {pooled.n}  win {pooled.win_rate * 100:.1f}%  "
              f"expectancy {pooled.expectancy * 100:+.2f}%/trade  PF {pooled.profit_factor:.2f}")
        print(f"  Sharpe {pooled.sharpe:.2f}  PSR {pooled.psr:.2f}"
              + (f"  deflated {pooled.deflated_sharpe:.2f}" if pooled.deflated_sharpe is not None else ""))
        print(f"  {pooled.verdict}")
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate the signal engine on real option premiums")
    ap.add_argument("--symbols", nargs="+", default=["BANKNIFTY", "NIFTY", "FINNIFTY", "MIDCPNIFTY"])
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--confidence", type=float, default=60.0)
    ap.add_argument("--trials", type=int, default=1,
                    help="how many strategy variants were tried to arrive here "
                         "(drives the deflated Sharpe correction)")
    ap.add_argument("--json", type=str, default=None, help="write full results to this path")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    client = SignalsDataClient(verbose=not a.quiet)
    results = [run_symbol(client, s, days=a.days, confidence_threshold=a.confidence,
                          trials=a.trials, verbose=not a.quiet) for s in a.symbols]
    print_report(results)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in results], fh, indent=1, default=str)
        print(f"full results -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
