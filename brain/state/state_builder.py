"""
SignalsBrain — StateBuilder

Constructs a complete MarketState from raw API data.
This is the translation layer: raw JSON -> normalized dimensions.

Three properties this layer must guarantee:

1. **Channel discipline.** DIRECTION dimensions are stored signed in [-1,+1].
   CONVICTION dimensions are stored unsigned in [0,1] (with `raw` retaining the
   natural value or band). CONTEXT dimensions are stored but never scored.

2. **No hidden wall-clock reads.** Every time-derived value comes from the
   explicit `as_of` argument. Previously `_build_time_context` called
   `datetime.now()` internally, which made the engine untestable and
   unbacktestable: outside 09:30-15:15 IST `session_minutes` clamped to 0 and
   tripped the OPENING_CHAOS hard veto, so replaying history could never
   produce a signal and pattern memory could never be seeded.

3. **Session anchoring.** VWAP, day change and day range are computed from the
   current session's bars only. Passing a multi-day candle window previously
   produced a multi-day "intraday" VWAP feeding two weight-7 dimensions.
"""

from __future__ import annotations

import datetime
import math
import time
from typing import Optional
from zoneinfo import ZoneInfo

from .market_state import MarketState
from .dimensions import (
    DIMENSIONS, Channel, DimensionCategory,
    normalize_pct, normalize_range, normalize_threshold, normalize_unit,
    normalize_percentile,
)
from .velocity_tracker import VelocityTracker

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN_MIN = 9 * 60 + 15   # 09:15
MARKET_CLOSE_MIN = 15 * 60 + 30  # 15:30
SESSION_LENGTH_MIN = MARKET_CLOSE_MIN - MARKET_OPEN_MIN  # 375


def _ist(as_of: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(as_of, IST)


def session_minutes_for(as_of: float) -> Optional[float]:
    """
    Minutes since the 09:15 IST open, or None when the market is not open.

    Returning None (rather than clamping to 0) is what lets callers distinguish
    "pre-open" from "first minute of trading". The old `max(0, ...)` clamp
    collapsed those two states together.
    """
    t = _ist(as_of)
    if t.isoweekday() > 5:
        return None
    cur = t.hour * 60 + t.minute + t.second / 60.0
    if cur < MARKET_OPEN_MIN or cur > MARKET_CLOSE_MIN:
        return None
    return cur - MARKET_OPEN_MIN


def is_market_open(as_of: float) -> bool:
    return session_minutes_for(as_of) is not None


def session_start_epoch(as_of: float) -> float:
    """Epoch seconds of 09:15 IST on the calendar day of `as_of`."""
    t = _ist(as_of)
    return t.replace(hour=9, minute=15, second=0, microsecond=0).timestamp()


def next_weekly_expiry_dte(as_of: float, weekday: int = 4) -> float:
    """
    Days to the nearest weekly expiry (default Thursday=4 in isoweekday terms
    Mon=1..Sun=7 -> Thursday=4).

    `dte` was declared with weight 6 but never populated, so `raw("dte", 5)`
    always returned 5. That silently disabled BlunderGuard Rule 4 (expiry-day
    minimum confidence) and the THETA_DECAY risk scenario, while permanently
    enabling GAP_RISK, and froze `dte_band` at 3 for every stored pattern.
    """
    t = _ist(as_of)
    days_ahead = (weekday - t.isoweekday()) % 7
    expiry = (t + datetime.timedelta(days=days_ahead)).replace(
        hour=15, minute=30, second=0, microsecond=0
    )
    if expiry <= t:
        expiry += datetime.timedelta(days=7)
    return max(0.0, (expiry - t).total_seconds() / 86400.0)


class StateBuilder:
    """
    Builds a MarketState from raw data.
    Maintains a VelocityTracker and a rolling raw-value history per instrument.
    """

    def __init__(self, history_len: int = 250):
        self.trackers: dict[str, VelocityTracker] = {}
        self.prev_states: dict[str, MarketState] = {}
        # Rolling raw history per instrument per dimension, for percentile
        # normalisation (replaces hard-coded saturating thresholds).
        self.raw_history: dict[str, dict[str, list[float]]] = {}
        self.history_len = history_len

    def _tracker(self, instrument: str) -> VelocityTracker:
        if instrument not in self.trackers:
            self.trackers[instrument] = VelocityTracker(instrument)
        return self.trackers[instrument]

    def _push_raw(self, instrument: str, name: str, value: float):
        d = self.raw_history.setdefault(instrument, {})
        lst = d.setdefault(name, [])
        lst.append(value)
        if len(lst) > self.history_len:
            del lst[0: len(lst) - self.history_len]

    def _raw_hist(self, instrument: str, name: str) -> list[float]:
        return self.raw_history.get(instrument, {}).get(name, [])

    def reset_history(self, instrument: Optional[str] = None):
        """Clear rolling history (used between backtest runs)."""
        if instrument is None:
            self.raw_history.clear()
            self.trackers.clear()
            self.prev_states.clear()
        else:
            self.raw_history.pop(instrument, None)
            self.trackers.pop(instrument, None)
            self.prev_states.pop(instrument, None)

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD
    # ══════════════════════════════════════════════════════════════════════════

    def build(
        self,
        instrument: str,
        candles: Optional[dict] = None,
        gex_data: Optional[dict] = None,
        fii_dii: Optional[dict] = None,
        quote: Optional[dict] = None,
        vix: Optional[float] = None,
        htf_candles: Optional[dict] = None,
        daily_candles: Optional[dict] = None,
        market_open: Optional[bool] = None,
        as_of: Optional[float] = None,
        expiry_epoch: Optional[float] = None,
        session_start: Optional[float] = None,
    ) -> MarketState:
        """
        Build a complete MarketState from whatever data is available.

        Args:
            as_of: evaluation timestamp (epoch seconds). Defaults to now. Pass an
                historical value to replay the past deterministically.
            expiry_epoch: explicit expiry timestamp; derived from `as_of` if absent.
            session_start: explicit session-open epoch; derived from `as_of` if absent.
            market_open: override the derived market-open state.
            daily_candles: daily bars, used for the 20-day range position.
        """
        ts = time.time() if as_of is None else float(as_of)
        s_start = session_start_epoch(ts) if session_start is None else float(session_start)
        open_now = is_market_open(ts) if market_open is None else bool(market_open)

        state = MarketState(instrument=instrument, timestamp=ts, market_open=open_now)
        tracker = self._tracker(instrument)

        # Restrict candles to the current session for session-scoped measures.
        session_candles = self._slice_session(candles, s_start)

        if candles and candles.get("closes") and len(candles["closes"]) >= 30:
            self._build_from_candles(state, candles, session_candles, tracker, ts)

        if daily_candles and daily_candles.get("closes"):
            self._build_from_daily(state, daily_candles)

        if gex_data:
            self._build_from_gex(state, gex_data, candles, tracker, ts)

        if fii_dii and fii_dii.get("fii") is not None:
            self._build_from_fiidii(state, fii_dii, tracker, ts)

        if quote:
            self._build_from_quote(state, quote)

        if vix is not None:
            self._set(state, "vix", vix, normalize_range(vix, 10, 30))
            prev_vix = self._raw_hist(instrument, "vix")
            if prev_vix:
                chg = vix - prev_vix[-1]
                # Rising VIX is a risk gate, not a direction. CONTEXT channel.
                self._set(state, "vix_change", chg, normalize_pct(chg, 0, 1.5))
            self._push_raw(instrument, "vix", vix)

        if htf_candles and htf_candles.get("closes") and len(htf_candles["closes"]) >= 20:
            closes = htf_candles["closes"]
            e9 = self._ema(closes, 9)
            e21 = self._ema(closes, 21)
            if e9 is not None and e21 is not None:
                htf = 1.0 if e9 > e21 else -1.0 if e9 < e21 else 0.0
                self._set(state, "htf_trend", htf, htf)

        # ── Time context (all from `as_of`) ──────────────────────────────────
        self._build_time_context(state, ts, expiry_epoch)

        # ── Propagate derivatives into the state itself ──────────────────────
        for name, value in state.dimensions.items():
            history = tracker.histories.get(name)
            if history is None or not history.timestamps or history.timestamps[-1] != ts:
                tracker.update(name, value.normalized, ts)
            value.velocity = tracker.get_velocity(name)
            value.acceleration = tracker.get_acceleration(name)

        state.compute_quality()
        state.compute_composites()
        state.scan_number = tracker.scan_count
        tracker.scan_count += 1
        self.prev_states[instrument] = state
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # SETTER — enforces the channel contract
    # ══════════════════════════════════════════════════════════════════════════

    def _set(self, state: MarketState, name: str, raw: float, normalized: float):
        """
        Store a dimension, clamping to the range its channel requires.

        DIRECTION -> [-1,+1] signed.  CONVICTION -> [0,1] unsigned.
        This makes it structurally impossible for a conviction input to express
        a direction, which is what previously let ADX invert trend conviction.
        """
        if raw != raw or normalized != normalized:  # NaN guard
            return
        d = DIMENSIONS.get(name)
        if d is not None and d.channel is Channel.CONVICTION:
            normalized = max(0.0, min(1.0, normalized))
        else:
            normalized = max(-1.0, min(1.0, normalized))
        state.set_dimension(name, raw, normalized)

    @staticmethod
    def _slice_session(candles: Optional[dict], session_start: float) -> Optional[dict]:
        """
        Return only the bars belonging to the current session.

        Uses the `timestamps` array when supplied. Without timestamps we cannot
        know where the session begins, so we return the input unchanged and the
        caller keeps the previous (window-wide) behaviour.
        """
        if not candles:
            return candles
        stamps = candles.get("timestamps") or candles.get("times")
        closes = candles.get("closes") or []
        if not stamps or len(stamps) != len(closes):
            return candles
        idx = [i for i, t in enumerate(stamps) if float(t) >= session_start]
        if not idx:
            return candles
        lo = idx[0]
        out = {}
        for k, v in candles.items():
            out[k] = v[lo:] if isinstance(v, list) and len(v) == len(closes) else v
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # CANDLES
    # ══════════════════════════════════════════════════════════════════════════

    def _build_from_candles(self, state: MarketState, candles: dict,
                            session: dict, tracker: VelocityTracker, ts: float):
        inst = state.instrument
        closes = candles["closes"]
        highs = candles["highs"]
        lows = candles["lows"]
        opens = candles["opens"]
        volumes = candles.get("volumes", [])
        n = len(closes)
        ltp = closes[-1]

        self._set(state, "ltp", ltp, 0.0)

        # ── Session-scoped price structure ───────────────────────────────────
        s_opens = session.get("opens") or opens
        s_highs = session.get("highs") or highs
        s_lows = session.get("lows") or lows
        s_closes = session.get("closes") or closes
        s_vols = session.get("volumes") or volumes

        day_open = s_opens[0] if s_opens else ltp
        day_chg = ((ltp - day_open) / day_open) * 100 if day_open else 0.0
        self._set(state, "day_change_pct", day_chg, normalize_pct(day_chg, 0, 1.5))

        day_high = max(s_highs) if s_highs else ltp
        day_low = min(s_lows) if s_lows else ltp
        rng = day_high - day_low
        drp = (ltp - day_low) / rng if rng > 0 else 0.5
        self._set(state, "day_range_position", drp, drp * 2 - 1)

        # ── Opening range breakout (first 15 minutes of the session) ─────────
        orb = self._orb_status(session, ltp)
        if orb is not None:
            self._set(state, "orb_status", orb, orb)

        # ── EMAs ─────────────────────────────────────────────────────────────
        e9 = self._ema(closes, 9)
        e21 = self._ema(closes, 21)
        e50 = self._sma(closes, 50) if n >= 50 else self._sma(closes, min(n, 30))
        e200 = self._sma(closes, 200) if n >= 200 else None

        stack = 0.0
        if e9 is not None and e21 is not None:
            stack += 0.4 if e9 > e21 else -0.4
        if e50 is not None:
            stack += 0.3 if ltp > e50 else -0.3
            if e21 is not None:
                stack += 0.3 if e21 > e50 else -0.3
        stack = max(-1.0, min(1.0, stack))
        self._set(state, "ema_stack_score", stack, stack)

        # Composite signed distance from the EMA ribbon, in ATR units.
        atr = self._atr(highs, lows, closes)
        if atr and atr > 0:
            dists = [(ltp - e) / atr for e in (e9, e21, e50, e200) if e]
            if dists:
                avg = sum(dists) / len(dists)
                self._set(state, "ema_distance", avg, normalize_pct(avg, 0, 1.5))

        # ── Session VWAP ─────────────────────────────────────────────────────
        if s_vols and len(s_vols) == len(s_closes):
            vwap = self._vwap(s_highs, s_lows, s_closes, s_vols)
            if vwap and vwap > 0:
                dev = ((ltp - vwap) / vwap) * 100
                self._set(state, "vwap_deviation", dev, normalize_pct(dev, 0, 0.5))
                self._set(state, "vwap_position", dev, 1.0 if ltp > vwap else -1.0)

        # ── SuperTrend ───────────────────────────────────────────────────────
        st = self._supertrend(highs, lows, closes)
        st_val = 1.0 if st == "bullish" else -1.0
        self._set(state, "supertrend", st_val, st_val)

        # ── ADX: CONVICTION (unsigned) + signed DI differential ─────────────
        adx, plus_di, minus_di = self._adx(highs, lows, closes)
        if adx is not None:
            # Unsigned strength -> [0,1]. Cannot express a direction.
            self._set(state, "adx_value", adx, normalize_unit(adx, 12, 40))
            band = 1.0 if adx >= 25 else 0.0 if adx >= 18 else -1.0
            # raw keeps the band for regime labelling; normalized is the multiplier.
            self._set(state, "adx_regime", band,
                      1.0 if band > 0 else 0.55 if band == 0 else 0.2)
            # Direction lives here, and only here, for the trend-strength family.
            self._set(state, "di_differential", plus_di - minus_di,
                      normalize_pct(plus_di - minus_di, 0, 20))
            # Is the trend building or decaying? Unsigned conviction.
            prev_adx = self._raw_hist(inst, "adx_value")
            if prev_adx:
                delta = adx - prev_adx[-1]
                self._set(state, "trend_acceleration", delta,
                          normalize_unit(delta, -3.0, 3.0))
            self._push_raw(inst, "adx_value", adx)

        # ── RSI + divergence ─────────────────────────────────────────────────
        rsi, rsi_prev = self._rsi(closes)
        if rsi is not None:
            self._set(state, "rsi", rsi, normalize_threshold(rsi, 40, 60))
            price_up = ltp > closes[-5] if n >= 5 else False
            rsi_up = rsi > (rsi_prev if rsi_prev is not None else rsi)
            if price_up and not rsi_up:
                self._set(state, "rsi_divergence", -1, -1.0)
            elif not price_up and rsi_up:
                self._set(state, "rsi_divergence", 1, 1.0)
            else:
                self._set(state, "rsi_divergence", 0, 0.0)

        # ── Stochastic ───────────────────────────────────────────────────────
        stoch = self._stochastic(highs, lows, closes)
        if stoch is not None:
            self._set(state, "stochastic_zone", stoch, normalize_range(stoch, 20, 80))

        # ── MACD ─────────────────────────────────────────────────────────────
        macd_val, macd_sig, macd_hist = self._macd(closes)
        if macd_hist is not None:
            scale = atr if atr else ltp * 0.01
            self._set(state, "macd_histogram", macd_hist,
                      normalize_pct(macd_hist / scale, 0, 0.5))
            self._set(state, "macd_direction",
                      1.0 if macd_hist > 0 else -1.0,
                      1.0 if macd_hist > 0 else -1.0)

        # ── Volume: CONVICTION (unsigned) ────────────────────────────────────
        if volumes and len(volumes) >= 10:
            avg_vol = sum(volumes[-20:]) / min(20, len(volumes))
            cur = volumes[-1]
            ratio = cur / avg_vol if avg_vol > 0 else 1.0
            self._set(state, "volume_ratio", ratio, normalize_unit(ratio, 0.5, 2.5))
            prev = self._raw_hist(inst, "volume_ratio")
            if prev:
                self._set(state, "volume_trend", ratio - prev[-1],
                          normalize_unit(ratio - prev[-1], -0.8, 0.8))
            self._push_raw(inst, "volume_ratio", ratio)

        # ── Volatility: CONVICTION (unsigned) ────────────────────────────────
        if atr and ltp > 0:
            atr_pct = (atr / ltp) * 100
            self._set(state, "atr_pct", atr_pct, normalize_unit(atr_pct, 0.20, 1.20))
            # Rank ATR% against this instrument's own history. A fixed threshold
            # is interval-dependent: real NIFTY median ATR% is 0.0245 on 1-minute
            # bars vs 0.1118 on 15-minute, so any constant is wrong on some
            # timeframe. -1 signals "not enough history to rank".
            ah = self._raw_hist(inst, "atr_pct")
            if len(ah) >= 30:
                rank = sum(1 for x in ah if x < atr_pct) / len(ah)
                self._set(state, "atr_percentile", rank, rank)
            else:
                self._set(state, "atr_percentile", -1.0, 0.0)
            self._push_raw(inst, "atr_pct", atr_pct)

        bb_u, bb_m, bb_l = self._bollinger(closes)
        if bb_m and bb_m > 0:
            width = ((bb_u - bb_l) / bb_m) * 100
            # A squeeze is coiled energy -> HIGH conviction that a move is coming.
            self._set(state, "bb_width", width, 1.0 - normalize_unit(width, 1.0, 4.0))

        # ── Rate of change ───────────────────────────────────────────────────
        if n >= 6 and closes[-6]:
            roc = ((closes[-1] - closes[-6]) / closes[-6]) * 100
            self._set(state, "roc_5", roc, normalize_pct(roc, 0, 1.0))

        # ── Swing S/R proximity (CONTEXT, unsigned) ──────────────────────────
        if atr and atr > 0:
            prox = self._sr_proximity(highs, lows, ltp, atr)
            if prox is not None:
                self._set(state, "sr_proximity", prox, normalize_unit(prox, 0.0, 3.0))

    def _build_from_daily(self, state: MarketState, daily: dict):
        """20-day range position from daily bars."""
        highs = daily.get("highs") or []
        lows = daily.get("lows") or []
        closes = daily.get("closes") or []
        if not (highs and lows and closes):
            return
        hi = max(highs[-20:])
        lo = min(lows[-20:])
        ltp_dim = state.dimensions.get("ltp")
        ltp = ltp_dim.raw if ltp_dim else closes[-1]
        if hi > lo:
            pos = (ltp - lo) / (hi - lo)
            self._set(state, "range_20d_position", pos, pos * 2 - 1)

    def _build_from_quote(self, state: MarketState, quote: dict):
        """Delivery % is a conviction input when the feed provides it."""
        dp = quote.get("deliveryPct", quote.get("delivery_pct"))
        if dp is not None:
            self._set(state, "delivery_pct", float(dp), normalize_unit(float(dp), 30, 75))

    # ══════════════════════════════════════════════════════════════════════════
    # GEX / OPTIONS
    # ══════════════════════════════════════════════════════════════════════════

    def _build_from_gex(self, state: MarketState, gex: dict, candles: Optional[dict],
                        tracker: VelocityTracker, ts: float):
        inst = state.instrument
        ltp_dim = state.dimensions.get("ltp")
        ltp = ltp_dim.raw if ltp_dim else 0.0
        atr = None
        if candles:
            atr = self._atr(candles["highs"], candles["lows"], candles["closes"])
        if not atr or atr <= 0:
            atr = ltp * 0.01 if ltp else 1.0

        # ── PCR: percentile-ranked, not hard-thresholded ─────────────────────
        pcr = gex.get("pcr")
        if pcr and pcr > 0:
            hist = self._raw_hist(inst, "pcr")
            if len(hist) >= 30:
                # Self-adjusting per instrument. normalize_threshold(pcr,0.7,1.2)
                # pinned everything above 1.2 to exactly +1.0, so a real move
                # from 1.2 to 1.6 was invisible to both scoring and velocity.
                norm = normalize_percentile(pcr, hist)
            else:
                norm = normalize_threshold(pcr, 0.7, 1.2)
            self._set(state, "pcr", pcr, norm)
            self._push_raw(inst, "pcr", pcr)
            pcr_vel = tracker.get_velocity("pcr")
            self._set(state, "pcr_velocity", pcr_vel, normalize_pct(pcr_vel, 0, 0.3))

        # ── GEX regime: CONVICTION (unsigned multiplier) ─────────────────────
        regime = str(gex.get("regime", ""))
        if "Negative" in regime:
            # Dealers amplify moves -> directional trades are more reliable.
            self._set(state, "gex_regime", -1.0, 0.95)
        elif "Positive" in regime:
            # Dealers suppress moves -> directional trades are less reliable.
            self._set(state, "gex_regime", 1.0, 0.35)

        net_gex = gex.get("netGEX")
        if net_gex is not None:
            self._set(state, "gex_net", net_gex, normalize_pct(net_gex, 0, 50))

        flip = gex.get("flip")
        if flip and ltp > 0:
            fd = (ltp - flip) / atr
            self._set(state, "gex_flip_distance", fd, normalize_pct(fd, 0, 3.0))

        # ── Walls: signed direction (proximity to a magnet) ──────────────────
        cw = gex.get("callWall")
        pw = gex.get("putWall")
        if cw and ltp > 0:
            d = (cw - ltp) / atr
            # Near the call wall, upside is capped -> bearish tilt.
            self._set(state, "call_wall_distance", d, -1.0 * (1.0 - normalize_unit(d, 0.0, 4.0)))
        if pw and ltp > 0:
            d = (ltp - pw) / atr
            # Near the put wall, dealers support -> bullish tilt.
            self._set(state, "put_wall_distance", d, 1.0 - normalize_unit(d, 0.0, 4.0))

        mp = gex.get("maxPain")
        if mp and ltp > 0:
            d = (ltp - mp) / atr
            # Pin pull is toward max pain: above it the pull is downward.
            self._set(state, "max_pain_distance", d, -normalize_pct(d, 0, 3.0))

        # ── IV level (CONTEXT) + IV skew (DIRECTION) ─────────────────────────
        avg_iv = gex.get("avgIV")
        if avg_iv:
            self._set(state, "atm_iv", avg_iv, normalize_range(avg_iv, 10, 30))
            iv_hist = self._raw_hist(inst, "atm_iv")
            if len(iv_hist) >= 30:
                pctl = (sum(1 for h in iv_hist if h < avg_iv) / len(iv_hist)) * 100
            else:
                pctl = max(0.0, min(100.0, (avg_iv - 10) / 20 * 100))
            self._set(state, "iv_percentile", pctl, normalize_range(pctl, 20, 80))
            self._push_raw(inst, "atm_iv", avg_iv)

        put_iv = gex.get("putIV", gex.get("avgPutIV"))
        call_iv = gex.get("callIV", gex.get("avgCallIV"))
        if put_iv is not None and call_iv is not None:
            skew = float(put_iv) - float(call_iv)
            # Positive skew = demand for downside protection = bearish.
            self._set(state, "iv_skew", skew, -normalize_pct(skew, 0, 2.0))

        # ── OI buildup (DIRECTION) ───────────────────────────────────────────
        oi = gex.get("oiBuildup", gex.get("oi_buildup"))
        if oi is not None:
            mapping = {
                "long_buildup": 1.0, "short_covering": 0.5,
                "short_buildup": -1.0, "long_unwinding": -0.5,
            }
            val = mapping.get(str(oi).lower()) if not isinstance(oi, (int, float)) else float(oi)
            if val is not None:
                self._set(state, "oi_buildup", val, max(-1.0, min(1.0, val)))
        else:
            derived = self._derive_oi_buildup(state, gex, inst)
            if derived is not None:
                self._set(state, "oi_buildup", derived, derived)

    def _derive_oi_buildup(self, state: MarketState, gex: dict, inst: str) -> Optional[float]:
        """
        Classify OI buildup from the sign of the OI change against the sign of
        the price change, when the feed gives total OI but not a label.
        """
        total_oi = gex.get("totalOI", gex.get("total_oi"))
        if total_oi is None:
            return None
        hist = self._raw_hist(inst, "total_oi")
        self._push_raw(inst, "total_oi", float(total_oi))
        if not hist:
            return None
        d_oi = float(total_oi) - hist[-1]
        chg = state.dimensions.get("day_change_pct")
        if chg is None or d_oi == 0:
            return None
        price_up = chg.raw > 0
        oi_up = d_oi > 0
        if price_up and oi_up:
            return 1.0     # long buildup
        if price_up and not oi_up:
            return 0.5     # short covering
        if not price_up and oi_up:
            return -1.0    # short buildup
        return -0.5        # long unwinding

    # ══════════════════════════════════════════════════════════════════════════
    # FLOW / TIME
    # ══════════════════════════════════════════════════════════════════════════

    def _build_from_fiidii(self, state: MarketState, data: dict,
                           tracker: VelocityTracker, ts: float):
        fii = data.get("fii")
        dii = data.get("dii")
        if fii is not None:
            self._set(state, "fii_flow", fii, normalize_pct(fii, 0, 1500))
        if dii is not None:
            self._set(state, "dii_flow", dii, normalize_pct(dii, 0, 1500))

    def _build_time_context(self, state: MarketState, as_of: float,
                            expiry_epoch: Optional[float]):
        """
        All CONTEXT-channel time values. Derived from `as_of` only.

        These are stored so downstream gates (BlunderGuard, risk scenarios,
        position sizing) can read them. By channel contract they contribute
        nothing to the directional score, so the calendar can no longer make
        Friday mechanically bullish.
        """
        t = _ist(as_of)
        sess = session_minutes_for(as_of)

        if sess is None:
            # Outside the session. Store a sentinel of -1 so gates can tell
            # "market closed" apart from "first minute of trading".
            self._set(state, "session_minutes", -1.0, 0.0)
            self._set(state, "session_phase", -1.0, 0.0)
        else:
            self._set(state, "session_minutes", sess, normalize_range(sess, 0, SESSION_LENGTH_MIN))
            if sess <= 15:
                phase = 0.0
            elif sess <= 120:
                phase = 0.25
            elif sess <= 240:
                phase = 0.5
            elif sess <= 330:
                phase = 0.75
            else:
                phase = 1.0
            self._set(state, "session_phase", phase, normalize_range(phase, 0, 1))

        self._set(state, "day_of_week", float(t.isoweekday()), 0.0)

        if expiry_epoch is not None:
            dte = max(0.0, (float(expiry_epoch) - as_of) / 86400.0)
        else:
            dte = next_weekly_expiry_dte(as_of)
        self._set(state, "dte", dte, normalize_range(dte, 0, 7))

    # ══════════════════════════════════════════════════════════════════════════
    # INDICATORS (self-contained; RSI/ATR/MACD verified against Wilder/reference)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _orb_status(session: dict, ltp: float) -> Optional[float]:
        """+1 above the opening range, -1 below, 0 inside."""
        highs = session.get("highs") or []
        lows = session.get("lows") or []
        if len(highs) < 4 or len(lows) < 4:
            return None
        k = min(3, len(highs) - 1)  # first ~15 minutes on 5-minute bars
        orb_hi = max(highs[:k])
        orb_lo = min(lows[:k])
        if ltp > orb_hi:
            return 1.0
        if ltp < orb_lo:
            return -1.0
        return 0.0

    @staticmethod
    def _sr_proximity(highs: list, lows: list, ltp: float, atr: float) -> Optional[float]:
        """Unsigned distance to the nearest swing pivot, in ATR units."""
        n = len(highs)
        if n < 11:
            return None
        pivots = []
        for i in range(2, n - 2):
            if highs[i] == max(highs[i - 2:i + 3]):
                pivots.append(highs[i])
            if lows[i] == min(lows[i - 2:i + 3]):
                pivots.append(lows[i])
        if not pivots:
            return None
        return min(abs(ltp - p) for p in pivots) / atr

    @staticmethod
    def _stochastic(highs: list, lows: list, closes: list, period: int = 14) -> Optional[float]:
        if len(closes) < period:
            return None
        hh = max(highs[-period:])
        ll = min(lows[-period:])
        if hh <= ll:
            return 50.0
        return (closes[-1] - ll) / (hh - ll) * 100

    @staticmethod
    def _ema(values: list, period: int) -> Optional[float]:
        if not values or len(values) < period:
            return None
        k = 2 / (period + 1)
        ema = sum(values[:period]) / period
        for v in values[period:]:
            ema = (v - ema) * k + ema
        return ema

    @staticmethod
    def _sma(values: list, period: int) -> Optional[float]:
        if not values or len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _rsi(closes: list, period: int = 14) -> tuple[Optional[float], Optional[float]]:
        if len(closes) < period + 1:
            return None, None
        gains = losses = 0.0
        for i in range(1, period + 1):
            d = closes[i] - closes[i - 1]
            if d > 0:
                gains += d
            else:
                losses -= d
        ag, al = gains / period, losses / period
        prev_rsi = None
        rsi = None
        for i in range(period + 1, len(closes)):
            d = closes[i] - closes[i - 1]
            ag = (ag * (period - 1) + (d if d > 0 else 0)) / period
            al = (al * (period - 1) + (-d if d < 0 else 0)) / period
            prev_rsi = rsi
            rsi = 100 - 100 / (1 + ag / al) if al != 0 else 100
        if rsi is None:
            rs = ag / al if al != 0 else 100
            rsi = 100 - 100 / (1 + rs)
        return rsi, prev_rsi

    @staticmethod
    def _ema_series(values: list, period: int) -> list[Optional[float]]:
        out: list[Optional[float]] = [None] * len(values)
        if len(values) < period:
            return out
        k = 2 / (period + 1)
        e = sum(values[:period]) / period
        out[period - 1] = e
        for i in range(period, len(values)):
            e = (values[i] - e) * k + e
            out[i] = e
        return out

    @classmethod
    def _macd(cls, closes: list, fast=12, slow=26, sig=9):
        """Standard MACD. Rewritten for clarity; numerically identical to before."""
        if len(closes) < slow + sig:
            return None, None, None
        ef = cls._ema_series(closes, fast)
        es = cls._ema_series(closes, slow)
        line = [ef[i] - es[i] for i in range(len(closes))
                if ef[i] is not None and es[i] is not None]
        if len(line) < sig:
            return None, None, None
        sl = cls._ema_series(line, sig)
        macd_val = line[-1]
        signal = sl[-1]
        if signal is None:
            return None, None, None
        return macd_val, signal, macd_val - signal

    @staticmethod
    def _atr(highs: list, lows: list, closes: list, period: int = 14) -> Optional[float]:
        n = len(highs)
        if n < period + 1:
            return None
        tr = []
        for i in range(1, n):
            tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                          abs(lows[i] - closes[i - 1])))
        if len(tr) < period:
            return None
        atr = sum(tr[:period]) / period
        for t in tr[period:]:
            atr = (atr * (period - 1) + t) / period
        return atr

    @staticmethod
    def _vwap(highs, lows, closes, volumes) -> Optional[float]:
        tpv = tv = 0.0
        for i in range(len(highs)):
            tp = (highs[i] + lows[i] + closes[i]) / 3
            v = volumes[i] if i < len(volumes) else 1
            tpv += tp * v
            tv += v
        return tpv / tv if tv > 0 else None

    @staticmethod
    def _bollinger(closes: list, period: int = 20, sd: float = 2.0):
        if len(closes) < period:
            return None, None, None
        s = closes[-period:]
        mid = sum(s) / period
        var = sum((x - mid) ** 2 for x in s) / period
        std = var ** 0.5
        return mid + sd * std, mid, mid - sd * std

    @staticmethod
    def _supertrend(highs, lows, closes, period=10, multiplier=3) -> str:
        n = len(closes)
        if n < period + 1:
            return "bearish"
        tr = []
        for i in range(1, n):
            tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                          abs(lows[i] - closes[i - 1])))
        atr_arr = [0.0] * len(tr)
        atr_arr[period - 1] = sum(tr[:period]) / period
        for i in range(period, len(tr)):
            atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i]) / period

        direction = 1
        pu, pl = 0.0, 0.0
        for i in range(period, n):
            av = atr_arr[i - 1]
            hl = (highs[i] + lows[i]) / 2
            bu = hl + multiplier * av
            bl = hl - multiplier * av
            u = bu if (bu < pu or closes[i - 1] > pu) else pu
            lo = bl if (bl > pl or closes[i - 1] < pl) else pl
            if direction == 1 and closes[i] < lo:
                direction = -1
            elif direction == -1 and closes[i] > u:
                direction = 1
            pu, pl = u, lo
        return "bullish" if direction == 1 else "bearish"

    @staticmethod
    def _adx(highs, lows, closes, period=14) -> tuple[Optional[float], float, float]:
        n = len(highs)
        if n < 2 * period + 1:
            return None, 0.0, 0.0
        tr, plus_dm, minus_dm = [], [], []
        for i in range(1, n):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)
            tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                          abs(lows[i] - closes[i - 1])))

        def smooth(arr):
            s = sum(arr[:period])
            out = [s]
            for v in arr[period:]:
                s = s - s / period + v
                out.append(s)
            return out

        s_tr = smooth(tr)
        s_plus = smooth(plus_dm)
        s_minus = smooth(minus_dm)

        dx = []
        for i in range(len(s_tr)):
            pdi = 100 * s_plus[i] / (s_tr[i] or 1)
            mdi = 100 * s_minus[i] / (s_tr[i] or 1)
            dx.append(100 * abs(pdi - mdi) / ((pdi + mdi) or 1))

        if len(dx) < period:
            return None, 0.0, 0.0
        adx = sum(dx[:period]) / period
        for v in dx[period:]:
            adx = (adx * (period - 1) + v) / period

        li = len(s_tr) - 1
        plus_di = 100 * s_plus[li] / (s_tr[li] or 1)
        minus_di = 100 * s_minus[li] / (s_tr[li] or 1)
        return adx, plus_di, minus_di
