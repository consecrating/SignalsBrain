"""
SignalsBrain — Validation Metrics

Why this module exists
----------------------
Until now the only number this system reported about itself was a win rate. A win
rate cannot tell you whether a strategy is profitable (you can win 70% of the time
and lose money), cannot tell you whether a result is distinguishable from luck, and
cannot tell you whether it survived because it is real or because dozens of variants
were tried and the best one was kept.

That last problem is the dangerous one and it is the reason the deflated Sharpe ratio
is implemented here. When several strategy variants are tested and only the winner is
reported, the winner's Sharpe is biased upward — even a set of purely random
strategies will produce a few impressive-looking ones. The Deflated Sharpe Ratio
(Bailey & López de Prado, 2014) corrects the observed Sharpe for the number of trials
and for non-normal returns, and answers the only question that matters: what is the
probability that the true Sharpe is above zero?

Conventions
-----------
* `returns` are PER-TRADE or PER-PERIOD simple returns (not log), as a 1-D sequence.
* Sharpe ratios come in two flavours and mixing them up is the classic error:
    - "non-annualised" (a.k.a. per-observation) SR = mean / std. The PSR and DSR
      formulae below require THIS one, together with the observation count.
    - "annualised" SR = non-annualised * sqrt(periods_per_year), for human reporting.
  Both are returned explicitly so a caller can never accidentally use the wrong one.

Everything is implemented on numpy/scipy, which are already dependencies, so the
service gains no new runtime weight. quantstats/pyfolio remain useful for rendering
tearsheets and can be layered on top of these same return series if desired.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy import stats

# Trading-period constants for Indian index F&O.
# 375 minutes per session (09:15-15:30) => 25 fifteen-minute bars per day.
TRADING_DAYS_PER_YEAR = 252
BARS_15M_PER_DAY = 25
BARS_15M_PER_YEAR = TRADING_DAYS_PER_YEAR * BARS_15M_PER_DAY

# Euler-Mascheroni constant, used by the expected-maximum-Sharpe estimate.
_EULER_GAMMA = 0.5772156649015329

# Relative tolerance for treating a dispersion as zero. Constant return series arise
# routinely (a run of identical stop-outs), and floating-point std of a constant series
# is ~1e-18 rather than 0 — dividing by that manufactures an astronomical Sharpe.
_ZERO_TOL = 1e-12


@dataclass
class PerformanceSummary:
    """Everything worth knowing about one return series, in one object."""

    n: int = 0
    total_return: float = 0.0
    mean_return: float = 0.0
    median_return: float = 0.0
    stdev: float = 0.0

    win_rate: float = 0.0
    wins: int = 0
    losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0

    sharpe: float = 0.0                 # annualised, for reporting
    sharpe_per_obs: float = 0.0         # non-annualised, for PSR/DSR
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0

    skew: float = 0.0
    kurtosis: float = 0.0               # non-excess (normal == 3)

    # Statistical significance
    t_stat: float = 0.0
    p_value: float = 1.0
    psr: float = 0.0                    # P(true SR > 0), single trial
    deflated_sharpe: Optional[float] = None   # P(true SR > 0) after N trials
    trials: Optional[int] = None

    verdict: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(returns: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(returns), dtype=float)
    if a.size == 0:
        return a
    return a[np.isfinite(a)]


def sharpe_ratio(returns: Sequence[float], periods_per_year: int = BARS_15M_PER_YEAR,
                 risk_free: float = 0.0, annualised: bool = True) -> float:
    """
    Mean excess return over its standard deviation.

    `risk_free` is expressed in the SAME per-period units as `returns`; the default of
    zero is the honest choice for an intraday strategy that holds no overnight capital
    and therefore earns no carry.
    """
    a = _clean(returns)
    if a.size < 2:
        return 0.0
    excess = a - risk_free
    sd = excess.std(ddof=1)
    # Guard against a NEAR-zero standard deviation, not just an exactly-zero one. A
    # constant series does not produce sd == 0 in floating point (it lands around
    # 1e-18), which sent the ratio to ~4e17 and would have been reported as a
    # spectacular Sharpe. Scale the tolerance to the data so it holds for any units.
    if sd <= _ZERO_TOL * max(1.0, float(np.abs(excess).mean())):
        return 0.0
    sr = excess.mean() / sd
    return float(sr * math.sqrt(periods_per_year)) if annualised else float(sr)


def sortino_ratio(returns: Sequence[float], periods_per_year: int = BARS_15M_PER_YEAR,
                  required_return: float = 0.0) -> float:
    """
    Like Sharpe but punishes only downside deviation. Preferred for option buying,
    whose return distribution is deliberately right-skewed (capped loss, long tail),
    because plain Sharpe penalises the upside tail you are actually paying for.
    """
    a = _clean(returns)
    if a.size < 2:
        return 0.0
    excess = a - required_return
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    dd = math.sqrt(float((downside ** 2).mean()))
    if dd <= _ZERO_TOL * max(1.0, float(np.abs(excess).mean())):
        return 0.0
    return float(excess.mean() / dd * math.sqrt(periods_per_year))


def max_drawdown(returns: Sequence[float]) -> float:
    """Deepest peak-to-trough fall of the compounded equity curve, as a fraction."""
    a = _clean(returns)
    if a.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + a)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(-dd.min())


def profit_factor(returns: Sequence[float]) -> float:
    """Gross win / gross loss. Below 1.0 means the strategy loses money."""
    a = _clean(returns)
    gains = a[a > 0].sum()
    losses = -a[a < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def expectancy(returns: Sequence[float]) -> float:
    """Average return per trade — the number that actually compounds."""
    a = _clean(returns)
    return float(a.mean()) if a.size else 0.0


def probabilistic_sharpe_ratio(sharpe_per_obs: float, n: int, skew: float,
                               kurtosis: float, benchmark: float = 0.0) -> float:
    """
    P(true Sharpe > benchmark), correcting for track-record length and for the
    non-normality of the return distribution (Bailey & López de Prado).

    `sharpe_per_obs` MUST be the non-annualised Sharpe and `kurtosis` the non-excess
    kurtosis (3.0 for a normal distribution).
    """
    if n < 3:
        return 0.0
    denom = 1.0 - skew * sharpe_per_obs + ((kurtosis - 1.0) / 4.0) * sharpe_per_obs ** 2
    if denom <= 0:
        return 0.0
    z = (sharpe_per_obs - benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(trials: int, sharpe_variance: float) -> float:
    """
    Expected value of the LARGEST Sharpe obtainable from `trials` independent
    strategies that all truly have zero edge. This is the bar a selected winner must
    clear to be considered real, and it grows with the number of things you tried.
    """
    if trials < 2 or sharpe_variance <= 0:
        return 0.0
    sd = math.sqrt(sharpe_variance)
    a = stats.norm.ppf(1.0 - 1.0 / trials)
    b = stats.norm.ppf(1.0 - 1.0 / (trials * math.e))
    return float(sd * ((1.0 - _EULER_GAMMA) * a + _EULER_GAMMA * b))


def deflated_sharpe_ratio(returns: Sequence[float], trials: int,
                          trial_sharpes: Optional[Sequence[float]] = None,
                          sharpe_variance: Optional[float] = None) -> float:
    """
    The single most important number in this module.

    Takes the winning strategy's returns and the number of variants that were tried
    to find it, and returns P(true Sharpe > 0) after removing selection bias.

    Supply EITHER the Sharpe ratios of every variant tried (`trial_sharpes`, best) or
    their variance directly. With neither, the variance is estimated from the winner
    alone, which understates the correction — so pass the trial Sharpes when you can.
    """
    a = _clean(returns)
    if a.size < 3:
        return 0.0

    sr_obs = sharpe_ratio(a, annualised=False)

    if trial_sharpes is not None:
        ts = _clean(trial_sharpes)
        var = float(ts.var(ddof=1)) if ts.size > 1 else 0.0
    elif sharpe_variance is not None:
        var = float(sharpe_variance)
    else:
        # Fallback: asymptotic variance of a single Sharpe estimate.
        var = (1.0 + 0.5 * sr_obs ** 2) / a.size

    bar = expected_max_sharpe(max(trials, 2), var)
    return probabilistic_sharpe_ratio(
        sr_obs, a.size, float(stats.skew(a)), float(stats.kurtosis(a, fisher=False)),
        benchmark=bar,
    )


def probability_of_backtest_overfitting(returns_matrix: np.ndarray, n_splits: int = 8) -> float:
    """
    PBO via Combinatorially Symmetric Cross-Validation.

    `returns_matrix` is (observations x variants): the per-period returns of every
    variant that was tried. The data is cut into `n_splits` blocks; for every way of
    using half the blocks in-sample, the in-sample winner is selected and its
    out-of-sample rank measured. PBO is the share of splits where the in-sample
    winner lands in the bottom half out-of-sample.

    Read it plainly: a PBO near 0.5 means the selection procedure has learned nothing
    that generalises — picking the best variant in-sample is a coin flip out-of-sample.
    """
    from itertools import combinations

    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        return float("nan")
    n_splits = max(2, n_splits - (n_splits % 2))  # must be even to halve
    t = m.shape[0]
    if t < n_splits * 2:
        return float("nan")

    blocks = np.array_split(np.arange(t), n_splits)
    half = n_splits // 2
    logits: list[float] = []

    for is_idx in combinations(range(n_splits), half):
        oos_idx = [b for b in range(n_splits) if b not in is_idx]
        is_rows = np.concatenate([blocks[b] for b in is_idx])
        oos_rows = np.concatenate([blocks[b] for b in oos_idx])

        is_perf = np.array([sharpe_ratio(m[is_rows, c], annualised=False) for c in range(m.shape[1])])
        oos_perf = np.array([sharpe_ratio(m[oos_rows, c], annualised=False) for c in range(m.shape[1])])

        best = int(np.nanargmax(is_perf))
        # Relative rank of the chosen variant out-of-sample, in (0, 1].
        order = stats.rankdata(oos_perf)
        rank = order[best] / (m.shape[1] + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))

    if not logits:
        return float("nan")
    # PBO = P(logit <= 0) = share of splits where the winner underperformed the median.
    return float(np.mean(np.asarray(logits) <= 0))


def summarize(returns: Sequence[float], periods_per_year: int = BARS_15M_PER_YEAR,
              trials: int = 1, trial_sharpes: Optional[Sequence[float]] = None,
              label: str = "") -> PerformanceSummary:
    """Full report for one return series, including the honesty-check statistics."""
    a = _clean(returns)
    s = PerformanceSummary(n=int(a.size))
    if a.size == 0:
        s.verdict = "no trades"
        return s

    wins = a[a > 0]
    losses = a[a < 0]

    s.total_return = float(np.prod(1.0 + a) - 1.0)
    s.mean_return = float(a.mean())
    s.median_return = float(np.median(a))
    s.stdev = float(a.std(ddof=1)) if a.size > 1 else 0.0

    s.wins, s.losses = int(wins.size), int(losses.size)
    s.win_rate = float(wins.size / a.size)
    s.avg_win = float(wins.mean()) if wins.size else 0.0
    s.avg_loss = float(losses.mean()) if losses.size else 0.0
    s.profit_factor = profit_factor(a)
    s.expectancy = expectancy(a)
    s.payoff_ratio = float(abs(s.avg_win / s.avg_loss)) if s.avg_loss else 0.0

    s.sharpe = sharpe_ratio(a, periods_per_year)
    s.sharpe_per_obs = sharpe_ratio(a, annualised=False)
    s.sortino = sortino_ratio(a, periods_per_year)
    s.max_drawdown = max_drawdown(a)
    s.calmar = float(s.total_return / s.max_drawdown) if s.max_drawdown > 0 else 0.0

    s.skew = float(stats.skew(a)) if a.size > 2 else 0.0
    s.kurtosis = float(stats.kurtosis(a, fisher=False)) if a.size > 3 else 3.0

    if a.size > 1 and s.stdev > 0:
        t, p = stats.ttest_1samp(a, 0.0)
        s.t_stat, s.p_value = float(t), float(p)

    s.psr = probabilistic_sharpe_ratio(s.sharpe_per_obs, a.size, s.skew, s.kurtosis, 0.0)
    s.trials = int(trials)
    if trials > 1 or trial_sharpes is not None:
        s.deflated_sharpe = deflated_sharpe_ratio(
            a, trials=trials, trial_sharpes=trial_sharpes)

    s.verdict = _verdict(s, label)
    return s


def _verdict(s: PerformanceSummary, label: str = "") -> str:
    """
    A plain-English reading, deliberately conservative. The purpose of this harness is
    to stop wishful interpretation, so the wording never implies an edge that the
    statistics do not support.
    """
    name = f"{label}: " if label else ""
    if s.n < 30:
        return f"{name}INCONCLUSIVE — only {s.n} trades; far too few to judge."
    if s.expectancy <= 0:
        return (f"{name}LOSES MONEY — average return per trade is "
                f"{s.expectancy * 100:.3f}%. No amount of win rate fixes negative expectancy.")
    decisive = s.deflated_sharpe if s.deflated_sharpe is not None else s.psr
    tag = "deflated Sharpe" if s.deflated_sharpe is not None else "PSR"
    if decisive < 0.60:
        return (f"{name}NOT DISTINGUISHABLE FROM LUCK — {tag} {decisive:.2f} "
                f"(want > 0.95). Positive expectancy but well within noise.")
    if decisive < 0.95:
        return (f"{name}PROMISING BUT UNPROVEN — {tag} {decisive:.2f}. "
                f"Keep it on paper and keep collecting trades.")
    return (f"{name}STATISTICALLY SUPPORTED — {tag} {decisive:.2f}, "
            f"expectancy {s.expectancy * 100:.3f}%/trade, PF {s.profit_factor:.2f}.")
