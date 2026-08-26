"""
SignalsBrain — Recorder CLI

Start recording (run this during market hours, every session):

    python -m brain.record.cli run --db data/chain.db

One observation and exit, useful for cron:

    python -m brain.record.cli once --db data/chain.db

What has been captured so far:

    python -m brain.record.cli status --db data/chain.db

Suggested cron, IST, weekdays only:

    */1 9-15 * * 1-5  cd /path/to/SignalsBrain && \
        python -m brain.record.cli once --db data/chain.db >> logs/rec.log 2>&1

A resident `run` process is preferable to cron at a 30s cadence, since cron's
finest granularity is one minute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chain_recorder import (
    ChainRecorder, RecorderConfig, SnapshotStore, is_market_open,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Record option chain snapshots")
    ap.add_argument("mode", choices=["run", "once", "status"])
    ap.add_argument("--db", type=Path, default=Path("data/chain.db"))
    ap.add_argument("--instruments", default="NIFTY,BANKNIFTY")
    ap.add_argument("--cadence", type=int, default=30, help="seconds between polls")
    ap.add_argument("--iterations", type=int, default=None,
                    help="stop after N polls (default: run until stopped)")
    ap.add_argument("--force", action="store_true",
                    help="record even when the market is closed")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    store = SnapshotStore(args.db)

    if args.mode == "status":
        s = store.summary()
        if args.json:
            print(json.dumps(s, indent=2))
            return 0
        print(f"\n  store            {s['db']}")
        print(f"  total rows       {s['total_rows']:,}  (~{s['approx_bytes']/1e6:.2f} MB)")
        if not s["instruments"]:
            print("\n  Nothing recorded yet. Start with:")
            print(f"    python -m brain.record.cli run --db {args.db}")
            return 0
        for inst, d in s["instruments"].items():
            print(f"\n  {inst}")
            print(f"    rows           {d['rows']:,}")
            print(f"    window         {str(d['from'])[:16]} -> {str(d['to'])[:16]}")
            print(f"    avg coverage   {d['avg_coverage']*100:.1f}%")
            print(f"    with PCR       {d['rows_with_pcr']:,}")
            print(f"    with GEX       {d['rows_with_gex']:,}")
        print(f"\n  options factor replayable: "
              f"{'yes' if s['ready_to_replay_options'] else 'not yet'}")
        if not s["ready_to_replay_options"]:
            need = 15000 - s["total_rows"]
            print(f"    ~{need:,} more rows needed (about a quarter of recording)")
        print()
        return 0

    cfg = RecorderConfig(
        instruments=tuple(x.strip().upper() for x in args.instruments.split(",") if x.strip()),
        cadence_seconds=args.cadence,
        skip_when_closed=not args.force,
    )
    rec = ChainRecorder(store, cfg)

    import time as _t
    if not is_market_open(_t.time()) and not args.force:
        print("Market is closed. The option chain is legitimately dark outside "
              "09:15-15:30 IST on weekdays, so nothing is recorded.\n"
              "Use --force only to test connectivity.", file=sys.stderr)
        return 0

    if args.mode == "once":
        snaps = rec.record_all()
        for s in snaps:
            print(f"  {s.instrument}: coverage {s.coverage*100:.0f}%  "
                  f"ok=[{s.sources_ok}]  failed=[{s.sources_failed}]")
        return 0

    print(f"Recording {','.join(cfg.instruments)} every {cfg.cadence_seconds}s "
          f"-> {args.db}\nCtrl-C to stop.")
    try:
        stats = rec.run(max_iterations=args.iterations)
    except KeyboardInterrupt:
        stats = rec.stats
        print("\nstopped.")
    print(json.dumps(stats.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
