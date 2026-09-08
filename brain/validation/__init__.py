"""
SignalsBrain — Validation

The layer that lets this system be judged on evidence instead of assertion.

Before this existed, the only self-reported number was a win rate computed on whatever
trades happened to be in the log, and strategy changes were justified by ad-hoc scripts
measuring SPOT direction. Two things were consistently wrong with that:

1. The target variable. The desk buys options, so the honest test must replay real
   option premiums — direction minus theta minus spread. `option_backtest` does that
   using the broker's own NFO history.
2. The statistics. Trying many variants and reporting the best one inflates the winner.
   `metrics.deflated_sharpe_ratio` corrects for the number of trials, and
   `splits.combinatorial_purged_cv` replaces a single lucky path with a distribution.

Typical use:

    from brain.validation import SignalsDataClient, run_symbol, print_report

    client = SignalsDataClient()
    res = run_symbol(client, "BANKNIFTY", days=60, trials=12)
    print_report([res])

or from the command line:

    python -m brain.validation.runner --symbols BANKNIFTY NIFTY --days 60 --trials 12
"""

from __future__ import annotations

from .data import Candles, SignalsDataClient
from .engine_port import Signal, generate_signal
from .metrics import (
    PerformanceSummary,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    expectancy,
    max_drawdown,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    summarize,
)
from .option_backtest import BacktestTrade, OptionBacktester
from .splits import (
    Split,
    combinatorial_purged_cv,
    purge_train_indices,
    purged_kfold,
    train_test_holdout,
    walk_forward,
)

def __getattr__(name: str):
    """
    Lazily expose the runner.

    Importing it eagerly here meant `python -m brain.validation.runner` loaded the
    module twice (once via the package, once as __main__) and emitted a RuntimeWarning
    about unpredictable behaviour. Deferring keeps the convenience import working
    without the double load.
    """
    if name in {"ExperimentResult", "run_symbol", "print_report", "generate_signal_stream"}:
        from . import runner as _runner
        return getattr(_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Candles", "SignalsDataClient",
    "Signal", "generate_signal",
    "PerformanceSummary", "summarize", "sharpe_ratio", "sortino_ratio", "max_drawdown",
    "profit_factor", "expectancy", "probabilistic_sharpe_ratio", "deflated_sharpe_ratio",
    "expected_max_sharpe", "probability_of_backtest_overfitting",
    "BacktestTrade", "OptionBacktester",
    "ExperimentResult", "run_symbol", "print_report", "generate_signal_stream",
    "Split", "walk_forward", "purged_kfold", "combinatorial_purged_cv",
    "purge_train_indices", "train_test_holdout",
]
