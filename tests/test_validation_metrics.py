"""
Tests for brain.validation.metrics

These pin the properties that make the statistics trustworthy rather than merely
computable — in particular that the deflated Sharpe ratio actually PUNISHES trying many
variants, which is the whole reason it is here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from brain.validation import metrics as M


class TestSharpeAndFriends:
    def test_sharpe_zero_when_no_variance(self):
        assert M.sharpe_ratio([0.01] * 20) == 0.0

    def test_sharpe_sign_follows_mean(self):
        rng = np.random.default_rng(7)
        good = rng.normal(0.004, 0.01, 500)
        bad = rng.normal(-0.004, 0.01, 500)
        assert M.sharpe_ratio(good) > 0
        assert M.sharpe_ratio(bad) < 0

    def test_annualisation_scales_by_sqrt_periods(self):
        rng = np.random.default_rng(3)
        r = rng.normal(0.001, 0.01, 400)
        per_obs = M.sharpe_ratio(r, annualised=False)
        ann = M.sharpe_ratio(r, periods_per_year=252)
        assert ann == pytest.approx(per_obs * math.sqrt(252), rel=1e-9)

    def test_sortino_ignores_upside_volatility(self):
        # Identical downside, wildly different upside: Sortino must not be punished
        # for the right tail, which is exactly what a long-option payoff looks like.
        mild = [0.01, -0.01, 0.01, -0.01, 0.02]
        spiky = [0.30, -0.01, 0.25, -0.01, 0.40]
        assert M.sortino_ratio(spiky) > M.sortino_ratio(mild)

    def test_max_drawdown_known_case(self):
        # +100% then -50% returns to the starting point: drawdown is 50%.
        assert M.max_drawdown([1.0, -0.5]) == pytest.approx(0.5, rel=1e-9)

    def test_max_drawdown_zero_when_monotonic(self):
        assert M.max_drawdown([0.01] * 10) == pytest.approx(0.0, abs=1e-12)

    def test_profit_factor_and_expectancy(self):
        r = [0.10, 0.10, -0.05, -0.05]
        assert M.profit_factor(r) == pytest.approx(2.0, rel=1e-9)
        assert M.expectancy(r) == pytest.approx(0.025, rel=1e-9)

    def test_profit_factor_below_one_means_losing(self):
        assert M.profit_factor([0.05, -0.10, 0.05, -0.10]) < 1.0


class TestOverfittingCorrections:
    def test_expected_max_sharpe_grows_with_trials(self):
        """The core intuition: more variants tried => higher bar to clear."""
        v = 0.04
        assert M.expected_max_sharpe(2, v) < M.expected_max_sharpe(50, v) < M.expected_max_sharpe(5000, v)

    def test_expected_max_sharpe_zero_without_dispersion(self):
        assert M.expected_max_sharpe(100, 0.0) == 0.0

    def test_psr_increases_with_track_record_length(self):
        # Same Sharpe, more observations => more confidence it is real.
        short = M.probabilistic_sharpe_ratio(0.10, 30, 0.0, 3.0)
        long = M.probabilistic_sharpe_ratio(0.10, 3000, 0.0, 3.0)
        assert 0.0 <= short < long <= 1.0

    def test_psr_penalises_negative_skew_and_fat_tails(self):
        base = M.probabilistic_sharpe_ratio(0.10, 500, 0.0, 3.0)
        fat = M.probabilistic_sharpe_ratio(0.10, 500, -1.5, 9.0)
        assert fat < base

    def test_deflated_sharpe_is_lower_than_undeflated_psr(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0.002, 0.01, 400)
        s = M.summarize(r, trials=1)
        dsr = M.deflated_sharpe_ratio(r, trials=200, sharpe_variance=0.05)
        assert dsr < s.psr

    def test_deflated_sharpe_falls_as_trials_rise(self):
        rng = np.random.default_rng(13)
        r = rng.normal(0.003, 0.01, 500)
        few = M.deflated_sharpe_ratio(r, trials=3, sharpe_variance=0.02)
        many = M.deflated_sharpe_ratio(r, trials=2000, sharpe_variance=0.02)
        assert many < few

    def test_noise_strategy_is_not_credited(self):
        """A pure coin flip must not come out 'statistically supported'."""
        rng = np.random.default_rng(17)
        r = rng.normal(0.0, 0.02, 600)
        s = M.summarize(r, trials=50)
        assert s.deflated_sharpe is not None
        assert s.deflated_sharpe < 0.95
        assert "STATISTICALLY SUPPORTED" not in s.verdict

    def test_pbo_near_half_for_pure_noise(self):
        """
        Selecting the best of N random variants should generalise no better than chance,
        so PBO sits near 0.5. Loose bounds: this is a stochastic quantity.
        """
        rng = np.random.default_rng(23)
        mat = rng.normal(0.0, 0.01, (400, 8))
        pbo = M.probability_of_backtest_overfitting(mat, n_splits=8)
        assert not math.isnan(pbo)
        assert 0.2 <= pbo <= 0.8

    def test_pbo_low_when_one_variant_genuinely_dominates(self):
        rng = np.random.default_rng(29)
        mat = rng.normal(0.0, 0.01, (400, 6))
        mat[:, 0] += 0.02  # a real, persistent edge in column 0
        assert M.probability_of_backtest_overfitting(mat, n_splits=8) < 0.3


class TestSummaryVerdicts:
    def test_small_sample_is_inconclusive(self):
        assert "INCONCLUSIVE" in M.summarize([0.05] * 10).verdict

    def test_negative_expectancy_is_called_out(self):
        rng = np.random.default_rng(31)
        r = rng.normal(-0.01, 0.02, 120)
        s = M.summarize(r)
        assert "LOSES MONEY" in s.verdict

    def test_high_win_rate_but_negative_expectancy_still_loses(self):
        """
        The trap this harness was built to expose: 80% winners, still unprofitable.
        A win-rate-only dashboard would have called this a success.
        """
        r = [0.02] * 80 + [-0.20] * 20
        s = M.summarize(r)
        assert s.win_rate == pytest.approx(0.8)
        assert s.expectancy < 0
        assert "LOSES MONEY" in s.verdict

    def test_empty_series(self):
        s = M.summarize([])
        assert s.n == 0 and "no trades" in s.verdict

    def test_summary_fields_populated(self):
        rng = np.random.default_rng(37)
        s = M.summarize(rng.normal(0.002, 0.01, 200), trials=5)
        for f in ("sharpe", "sortino", "max_drawdown", "profit_factor", "expectancy",
                  "psr", "t_stat", "p_value"):
            assert getattr(s, f) is not None
        assert s.to_dict()["n"] == 200
