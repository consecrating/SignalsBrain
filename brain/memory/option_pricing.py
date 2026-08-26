"""
SignalsBrain — Option Pricing

Replaces the P&L proxy `pnl_pct = move_atr * 80`.

That proxy claimed a 1 ATR spot move was worth +80% on the premium and a 3.2 ATR
move +256%, with no reference to delta, gamma, theta, vega, strike selection, IV
level, days to expiry or the spread. Because api/main.py started outcome
tracking with entry_premium=0, the proxy was what actually landed in the
database, and those invented numbers became best_pnl_pct / worst_pnl_pct /
avg_pnl_pct in the statistics the AI model was instructed to trust.

Priority order for P&L now:
  1. Real observed premium at entry and exit (always preferred).
  2. Black-Scholes mark using the recorded strike, IV, rate and time to expiry.
  3. None -> the row is excluded from statistics rather than being fabricated.

Rule 3 is the important one. An unknown value must not be replaced by a
confident guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _n_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _n_cdf(x: float) -> float:
    """Standard normal CDF via erf (exact to double precision)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float   # per calendar day
    vega: float    # per 1 volatility point (1%)
    rho: float

    def to_dict(self) -> dict:
        return {
            "price": round(self.price, 4),
            "delta": round(self.delta, 4),
            "gamma": round(self.gamma, 6),
            "theta_per_day": round(self.theta, 4),
            "vega_per_vol_pt": round(self.vega, 4),
            "rho": round(self.rho, 4),
        }


def black_scholes(spot: float, strike: float, iv_pct: float, dte_days: float,
                  opt_type: str = "CE", rate: float = 0.065,
                  dividend: float = 0.0) -> Optional[Greeks]:
    """
    Black-Scholes-Merton price and greeks for an index/stock option.

    Args:
        spot: underlying price
        strike: strike price
        iv_pct: implied volatility in percent (e.g. 14.8 for 14.8%)
        dte_days: calendar days to expiry (may be fractional)
        opt_type: "CE" (call) or "PE" (put)
        rate: risk-free rate as a decimal. 6.5% is a reasonable Indian default.
        dividend: continuous dividend yield as a decimal

    Returns None when the inputs cannot support a price, rather than guessing.
    """
    if spot <= 0 or strike <= 0 or iv_pct is None or iv_pct <= 0:
        return None
    if dte_days is None or dte_days < 0:
        return None

    is_call = str(opt_type).upper().startswith("C")
    sigma = float(iv_pct) / 100.0
    T = max(float(dte_days), 0.0) / 365.0

    # At expiry the option is worth exactly its intrinsic value.
    if T <= 1e-9 or sigma <= 1e-9:
        intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
        return Greeks(price=intrinsic,
                      delta=(1.0 if intrinsic > 0 else 0.0) * (1 if is_call else -1),
                      gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc_r = math.exp(-rate * T)
    disc_q = math.exp(-dividend * T)

    if is_call:
        price = spot * disc_q * _n_cdf(d1) - strike * disc_r * _n_cdf(d2)
        delta = disc_q * _n_cdf(d1)
        rho = strike * T * disc_r * _n_cdf(d2) / 100.0
        theta = (-spot * disc_q * _n_pdf(d1) * sigma / (2 * sqrtT)
                 - rate * strike * disc_r * _n_cdf(d2)
                 + dividend * spot * disc_q * _n_cdf(d1)) / 365.0
    else:
        price = strike * disc_r * _n_cdf(-d2) - spot * disc_q * _n_cdf(-d1)
        delta = -disc_q * _n_cdf(-d1)
        rho = -strike * T * disc_r * _n_cdf(-d2) / 100.0
        theta = (-spot * disc_q * _n_pdf(d1) * sigma / (2 * sqrtT)
                 + rate * strike * disc_r * _n_cdf(-d2)
                 - dividend * spot * disc_q * _n_cdf(-d1)) / 365.0

    gamma = disc_q * _n_pdf(d1) / (spot * sigma * sqrtT)
    vega = spot * disc_q * _n_pdf(d1) * sqrtT / 100.0

    return Greeks(price=max(0.0, price), delta=delta, gamma=gamma,
                  theta=theta, vega=vega, rho=rho)


def implied_pnl_pct(
    entry_spot: float,
    exit_spot: float,
    strike: float,
    opt_type: str,
    iv_pct_entry: float,
    dte_days_entry: float,
    minutes_held: float,
    iv_pct_exit: Optional[float] = None,
    rate: float = 0.065,
    cost_pct: float = 1.0,
) -> Optional[float]:
    """
    Model the percentage change on the premium between entry and exit.

    Accounts for direction (delta), convexity (gamma), the passage of time
    (theta) and any IV change (vega) by repricing the option at both ends,
    then subtracts a round-trip cost.

    Returns None if the option cannot be priced, so the caller stores NULL and
    excludes the row from statistics.
    """
    if minutes_held is None or minutes_held < 0:
        return None
    entry = black_scholes(entry_spot, strike, iv_pct_entry, dte_days_entry, opt_type, rate)
    if entry is None or entry.price <= 1e-9:
        return None

    dte_exit = max(0.0, float(dte_days_entry) - float(minutes_held) / (60.0 * 24.0))
    iv_exit = iv_pct_entry if iv_pct_exit is None else iv_pct_exit
    exit_ = black_scholes(exit_spot, strike, iv_exit, dte_exit, opt_type, rate)
    if exit_ is None:
        return None

    gross = (exit_.price - entry.price) / entry.price * 100.0
    return gross - float(cost_pct)


def atm_strike(spot: float, step: float) -> float:
    """Nearest tradable strike for an instrument's strike step."""
    if step <= 0:
        return spot
    return round(spot / step) * step


def realised_pnl_pct(entry_premium: Optional[float], exit_premium: Optional[float],
                     cost_pct: float = 1.0) -> Optional[float]:
    """P&L from observed premiums. Preferred over any model estimate."""
    if not entry_premium or entry_premium <= 0:
        return None
    if exit_premium is None or exit_premium < 0:
        return None
    return (exit_premium - entry_premium) / entry_premium * 100.0 - cost_pct
