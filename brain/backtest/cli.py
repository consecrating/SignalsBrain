"""
SignalsBrain — Backtest CLI

    python -m brain.backtest.cli --synthetic 60
    python -m brain.backtest.cli --csv data/nifty_5m.csv --instrument NIFTY

Reports hit rate with an interval, expectancy, drawdown, calibration and veto
attribution. Read the interval before the point estimate: a 70-point-wide
confidence interval on 4 trades tells you the run proved nothing, which is a
result worth seeing rather than hiding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import Backtester, BacktestConfig
from .replay import SyntheticFeed, load_csv_feed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay SignalsBrain over historical data")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", type=int, metavar="DAYS",
                     help="generate N synthetic sessions (harness check only; "
                          "a random walk has no edge by construction)")
    src.add_argument("--csv", type=Path, help="CSV of OHLCV bars")
    ap.add_argument("--instrument", default="NIFTY")
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--step", type=int, default=1, help="bars between evaluations")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--db", type=Path, default=None,
                    help="pattern database to seed (walk-forward guarded)")
    ap.add_argument("--json", action="store_true", help="emit raw JSON only")
    ap.add_argument("--trades", action="store_true", help="list individual fills")
    args = ap.parse_args(argv)

    if args.synthetic:
        feed = SyntheticFeed(instrument=args.instrument, days=args.synthetic,
                             seed=args.seed, lookback=140, warmup=160,
                             future_horizon=60)
    else:
        feed = load_csv_feed(args.csv, args.instrument, lookback=140,
                             warmup=160, future_horizon=60)

    bt = Backtester(BacktestConfig(confidence_threshold=args.threshold,
                                   step=args.step), db_path=args.db)

    def progress(i, n):
        if not args.json:
            print(f"  ... {i}/{n}", file=sys.stderr)

    result = bt.run(feed, progress=progress)
    metrics = result.metrics()

    if args.json:
        payload = {"metrics": metrics}
        if args.trades:
            payload["fills"] = [f.to_dict() for f in result.fills]
        print(json.dumps(payload, indent=2, default=str))
        return 0

    _report(metrics, result, show_trades=args.trades)
    return 0


def _report(m: dict, result, show_trades: bool = False):
    print()
    print("=" * 74)
    print(f"  BACKTEST — {m['instrument']}")
    print("=" * 74)
    print(f"  evaluations        {m['evaluations']}")
    print(f"  signals emitted    {m['signals_emitted']}  ({m['signal_rate_pct']}% of evaluations)")
    print(f"  fills              {m['fills']}  (scored {m['fills_scored']}, unscored {m['fills_unscored']})")
    print(f"  runtime            {m['runtime_seconds']}s")

    if not m.get("hit_rate"):
        print()
        print(f"  {m.get('note', 'no scored fills')}")
        _vetoes(m)
        return

    hr = m["hit_rate"]
    ex = m["expectancy_pct"]
    print()
    print("  ── performance ──")
    print(f"  hit rate           {hr['rate_pct']}%  "
          f"[{hr['ci_lower_pct']}-{hr['ci_upper_pct']}% 95% CI, n={hr['n']}]")
    if hr["ci_width_pct"] > 25:
        print(f"                     interval is {hr['ci_width_pct']} points wide — "
              f"not enough trades to conclude anything")
    print(f"  expectancy/trade   {ex['mean']}%  "
          f"[{ex['ci_lower']} to {ex['ci_upper']}]")
    print(f"  total return       {m['total_return_pct']}%")
    print(f"  profit factor      {m['profit_factor']}")
    print(f"  avg win / loss     {m['avg_win_pct']}% / {m['avg_loss_pct']}%")
    print(f"  max drawdown       {m['max_drawdown_pct']}%")
    print(f"  sharpe (per trade) {m['sharpe_per_trade']}")
    print(f"  avg hold           {m['avg_minutes_held']} min")
    print(f"  outcome mix        {m['outcome_mix']}")

    if m.get("by_confidence_bucket"):
        print()
        print("  ── hit rate by confidence bucket ──")
        print("    (a monotone relationship is what makes the score usable for sizing)")
        for r in m["by_confidence_bucket"]:
            print(f"    {r['confidence']:>7}  n={r['n']:4d}  "
                  f"hit={r['hit_rate_pct']:5.1f}%  "
                  f"[{r['ci_lower_pct']:.0f}-{r['ci_upper_pct']:.0f}%]  "
                  f"meanP&L={r['mean_pnl_pct']:+7.2f}%")

    c = m.get("calibration") or {}
    if c.get("n"):
        print()
        print("  ── calibration ──")
        print(f"    n={c['n']}  brier={c['brier']}  log_loss={c['log_loss']}  "
              f"ECE={c['expected_calibration_error']}")
        print(f"    base rate={c['base_rate']}  mean prediction={c['mean_prediction']}")
        if c["brier"] > 0.25:
            print("    brier above 0.25: worse than always predicting 0.5 — "
                  "the score has no demonstrated skill")

    _vetoes(m)

    if show_trades:
        print()
        print("  ── fills ──")
        for f in result.fills:
            d = f.to_dict()
            print(f"    {d['direction']:4s} conf={d['confidence']:5.1f} "
                  f"{d['outcome']:10s} pnl={d['pnl_pct']} "
                  f"held={d['minutes_held']}m mfe={d['mfe_atr']} mae={d['mae_atr']}")
    print()


def _vetoes(m: dict):
    if m.get("veto_counts"):
        print()
        print("  ── what blocked trades ──")
        for k, v in list(m["veto_counts"].items())[:12]:
            print(f"    {k:22s} {v}")
    if m.get("no_trade_reasons"):
        print(f"  no-trade reasons     {m['no_trade_reasons']}")


if __name__ == "__main__":
    raise SystemExit(main())
