"""
SignalsBrain — StateBuilder

Constructs a complete MarketState from raw API data (Angel One proxy responses).
This is the translation layer: raw JSON → 47 normalized dimensions.

Handles:
- Candle data → trend, momentum, price structure dimensions
- GEX response → options microstructure dimensions
- FII/DII response → flow dimensions
- Option chain → IV surface, PCR
- Quote data → price, volume
"""

from __future__ import annotations

import math
import time
from typing import Optional

from .market_state import MarketState
from .dimensions import (
    DimensionCategory, normalize_pct, normalize_range, normalize_threshold
)
from .velocity_tracker import VelocityTracker


class StateBuilder:
    """
    Builds a MarketState from raw data.
    Maintains a VelocityTracker per instrument for velocity calculations.
    """
    
    def __init__(self):
        self.trackers: dict[str, VelocityTracker] = {}
        self.prev_states: dict[str, MarketState] = {}
    
    def _tracker(self, instrument: str) -> VelocityTracker:
        if instrument not in self.trackers:
            self.trackers[instrument] = VelocityTracker(instrument)
        return self.trackers[instrument]
    
    def build(
        self,
        instrument: str,
        candles: Optional[dict] = None,
        gex_data: Optional[dict] = None,
        fii_dii: Optional[dict] = None,
        quote: Optional[dict] = None,
        vix: Optional[float] = None,
        htf_candles: Optional[dict] = None,
        market_open: bool = True,
    ) -> MarketState:
        """
        Build a complete MarketState from available data sources.
        
        Any source can be None (graceful degradation — the brain works
        with whatever data it has, but is more confident with more).
        """
        state = MarketState(instrument=instrument, timestamp=time.time(), market_open=market_open)
        tracker = self._tracker(instrument)
        ts = state.timestamp
        
        # ── From Candles (OHLCV) ──────────────────────────────────────────────
        if candles and candles.get("closes") and len(candles["closes"]) >= 30:
            self._build_from_candles(state, candles, tracker, ts)
        
        # ── From GEX Data ─────────────────────────────────────────────────────
        if gex_data:
            self._build_from_gex(state, gex_data, candles, tracker, ts)
        
        # ── From FII/DII ──────────────────────────────────────────────────────
        if fii_dii and fii_dii.get("fii") is not None:
            self._build_from_fiidii(state, fii_dii, tracker, ts)
        
        # ── VIX ───────────────────────────────────────────────────────────────
        if vix is not None:
            norm_vix = normalize_range(vix, 10, 30)  # 10=calm, 30=panic
            state.set_dimension("vix", vix, norm_vix)
            tracker.update("vix", norm_vix, ts)
        
        # ── Higher Timeframe ──────────────────────────────────────────────────
        if htf_candles and htf_candles.get("closes") and len(htf_candles["closes"]) >= 20:
            closes = htf_candles["closes"]
            e9 = self._ema(closes, 9)
            e21 = self._ema(closes, 21)
            if e9 is not None and e21 is not None:
                htf = 1.0 if e9 > e21 else -1.0 if e9 < e21 else 0.0
                state.set_dimension("htf_trend", htf, htf)
                tracker.update("htf_trend", htf, ts)
        
        # ── Time Context ──────────────────────────────────────────────────────
        self._build_time_context(state)
        
        # ── Compute composites ────────────────────────────────────────────────
        state.compute_composites()
        state.scan_number = tracker.scan_count
        tracker.scan_count += 1
        
        # Store for next comparison
        self.prev_states[instrument] = state
        
        return state
    
    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE: Build from candles
    # ══════════════════════════════════════════════════════════════════════════
    
    def _build_from_candles(self, state: MarketState, candles: dict, tracker: VelocityTracker, ts: float):
        closes = candles["closes"]
        highs = candles["highs"]
        lows = candles["lows"]
        opens = candles["opens"]
        volumes = candles.get("volumes", [])
        n = len(closes)
        
        ltp = closes[-1]
        state.set_dimension("ltp", ltp, 0.0)  # LTP has no directional normalization
        
        # Day change %
        day_open = opens[0] if opens else ltp
        day_chg = ((ltp - day_open) / day_open) * 100 if day_open > 0 else 0
        norm_chg = normalize_pct(day_chg, center=0, scale=1.5)
        state.set_dimension("day_change_pct", day_chg, norm_chg)
        tracker.update("day_change_pct", norm_chg, ts)
        
        # Day range position
        day_high = max(highs) if highs else ltp
        day_low = min(lows) if lows else ltp
        day_range = day_high - day_low
        drp = (ltp - day_low) / day_range if day_range > 0 else 0.5
        norm_drp = drp * 2 - 1  # 0→-1, 0.5→0, 1→+1
        state.set_dimension("day_range_position", drp, norm_drp)
        tracker.update("day_range_position", norm_drp, ts)
        
        # EMAs
        e9 = self._ema(closes, 9)
        e21 = self._ema(closes, 21)
        e50 = self._sma(closes, 50) if n >= 50 else self._sma(closes, min(n, 30))
        
        # EMA stack score
        stack = 0.0
        if e9 is not None and e21 is not None:
            stack += 0.4 if e9 > e21 else -0.4
        if e50 is not None:
            stack += 0.3 if ltp > e50 else -0.3
            if e21 is not None:
                stack += 0.3 if e21 > e50 else -0.3
        stack = max(-1, min(1, stack))
        state.set_dimension("ema_stack_score", stack, stack)
        tracker.update("ema_stack_score", stack, ts)
        
        # VWAP
        if volumes and len(volumes) == n:
            vwap = self._vwap(highs, lows, closes, volumes)
            if vwap and vwap > 0:
                vwap_dev = ((ltp - vwap) / vwap) * 100
                norm_vwap_dev = normalize_pct(vwap_dev, center=0, scale=0.5)
                state.set_dimension("vwap_deviation", vwap_dev, norm_vwap_dev)
                state.set_dimension("vwap_position", vwap_dev, 1.0 if ltp > vwap else -1.0)
                tracker.update("vwap_deviation", norm_vwap_dev, ts)
                tracker.update("vwap_position", 1.0 if ltp > vwap else -1.0, ts)
        
        # SuperTrend
        st = self._supertrend(highs, lows, closes)
        st_val = 1.0 if st == "bullish" else -1.0
        state.set_dimension("supertrend", st_val, st_val)
        tracker.update("supertrend", st_val, ts)
        
        # ADX
        adx, plus_di, minus_di = self._adx(highs, lows, closes)
        if adx is not None:
            regime_norm = 1.0 if adx >= 25 else 0.0 if adx >= 18 else -1.0
            state.set_dimension("adx_value", adx, normalize_range(adx, 0, 50))
            state.set_dimension("adx_regime", regime_norm, regime_norm)
            state.set_dimension("di_differential", plus_di - minus_di, normalize_pct(plus_di - minus_di, 0, 20))
            tracker.update("adx_value", normalize_range(adx, 0, 50), ts)
        
        # RSI
        rsi, rsi_prev = self._rsi(closes)
        if rsi is not None:
            norm_rsi = normalize_threshold(rsi, 40, 60)
            state.set_dimension("rsi", rsi, norm_rsi)
            tracker.update("rsi", norm_rsi, ts)
            
            # RSI divergence
            price_up = ltp > closes[-5] if n >= 5 else False
            rsi_up = rsi > (rsi_prev or rsi)
            if price_up and not rsi_up:
                state.set_dimension("rsi_divergence", -1, -1.0)  # Bearish divergence
            elif not price_up and rsi_up:
                state.set_dimension("rsi_divergence", 1, 1.0)  # Bullish divergence
            else:
                state.set_dimension("rsi_divergence", 0, 0.0)
        
        # MACD
        macd_val, macd_sig, macd_hist = self._macd(closes)
        if macd_hist is not None:
            atr = self._atr(highs, lows, closes) or ltp * 0.01
            norm_hist = normalize_pct(macd_hist / atr, center=0, scale=0.5)
            state.set_dimension("macd_histogram", macd_hist, norm_hist)
            tracker.update("macd_histogram", norm_hist, ts)
            
            # Direction (expanding or contracting)
            state.set_dimension("macd_direction", 1.0 if macd_hist > 0 and macd_val > macd_sig else -1.0, 
                              1.0 if macd_hist > 0 and macd_val > macd_sig else -1.0)
        
        # Volume
        if volumes and len(volumes) >= 10:
            avg_vol = sum(volumes[-20:]) / min(20, len(volumes))
            cur_vol = volumes[-1]
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
            # Normalize: 0.5x = -1, 1x = 0, 2x+ = +1
            norm_vol = normalize_range(min(vol_ratio, 3.0), 0.5, 2.5)
            state.set_dimension("volume_ratio", vol_ratio, norm_vol)
            tracker.update("volume_ratio", norm_vol, ts)
        
        # ATR as % of price
        atr = self._atr(highs, lows, closes)
        if atr and ltp > 0:
            atr_pct = (atr / ltp) * 100
            # Normalize: <0.25% = dead market (-1), >1.5% = very active (+1)
            norm_atr = normalize_range(atr_pct, 0.25, 1.5)
            state.set_dimension("atr_pct", atr_pct, norm_atr)
        
        # Bollinger Band width
        bb_upper, bb_mid, bb_lower = self._bollinger(closes)
        if bb_mid and bb_mid > 0:
            bb_width = ((bb_upper - bb_lower) / bb_mid) * 100
            # <1.5% = squeeze (potential breakout), >4% = expanded
            norm_bb = normalize_range(bb_width, 1.0, 4.0)
            state.set_dimension("bb_width", bb_width, norm_bb)
            tracker.update("bb_width", norm_bb, ts)
        
        # Rate of change (5 bar)
        if n >= 6:
            roc = ((closes[-1] - closes[-6]) / closes[-6]) * 100
            norm_roc = normalize_pct(roc, 0, 1.0)
            state.set_dimension("roc_5", roc, norm_roc)
            tracker.update("roc_5", norm_roc, ts)
    
    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE: Build from GEX data
    # ══════════════════════════════════════════════════════════════════════════
    
    def _build_from_gex(self, state: MarketState, gex: dict, candles: Optional[dict], tracker: VelocityTracker, ts: float):
        ltp_dim = state.dimensions.get("ltp")
        ltp = ltp_dim.raw if ltp_dim else 0
        atr = self._atr(candles["highs"], candles["lows"], candles["closes"]) if candles else ltp * 0.01
        if atr <= 0:
            atr = ltp * 0.01
        
        # PCR
        pcr = gex.get("pcr")
        if pcr and pcr > 0:
            # PCR > 1.2 = bullish (put support building), < 0.7 = bearish
            norm_pcr = normalize_threshold(pcr, 0.7, 1.2)
            state.set_dimension("pcr", pcr, norm_pcr)
            tracker.update("pcr", norm_pcr, ts)
            
            # PCR velocity (from tracker history)
            pcr_vel = tracker.get_velocity("pcr")
            state.set_dimension("pcr_velocity", pcr_vel, normalize_pct(pcr_vel, 0, 0.3))
        
        # GEX regime
        regime = gex.get("regime", "")
        if "Negative" in regime:
            state.set_dimension("gex_regime", -1, -1.0)
        elif "Positive" in regime:
            state.set_dimension("gex_regime", 1, 1.0)
        
        # GEX net
        net_gex = gex.get("netGEX", 0)
        state.set_dimension("gex_net", net_gex, normalize_pct(net_gex, 0, 50))
        tracker.update("gex_net", normalize_pct(net_gex, 0, 50), ts)
        
        # GEX flip distance (in ATR units, signed)
        flip = gex.get("flip")
        if flip and ltp > 0 and atr > 0:
            flip_dist = (ltp - flip) / atr  # Positive = above flip, negative = below
            norm_flip = normalize_pct(flip_dist, 0, 3.0)
            state.set_dimension("gex_flip_distance", flip_dist, norm_flip)
            tracker.update("gex_flip_distance", norm_flip, ts)
        
        # Call wall / Put wall distances
        call_wall = gex.get("callWall")
        put_wall = gex.get("putWall")
        if call_wall and ltp > 0 and atr > 0:
            cw_dist = (call_wall - ltp) / atr
            state.set_dimension("call_wall_distance", cw_dist, normalize_pct(cw_dist, 0, 5))
        if put_wall and ltp > 0 and atr > 0:
            pw_dist = (ltp - put_wall) / atr
            state.set_dimension("put_wall_distance", pw_dist, normalize_pct(pw_dist, 0, 5))
        
        # Max pain distance
        max_pain = gex.get("maxPain")
        if max_pain and ltp > 0 and atr > 0:
            mp_dist = (ltp - max_pain) / atr
            state.set_dimension("max_pain_distance", mp_dist, normalize_pct(mp_dist, 0, 3))
        
        # IV
        avg_iv = gex.get("avgIV")
        if avg_iv:
            state.set_dimension("atm_iv", avg_iv, normalize_range(avg_iv, 10, 30))
            tracker.update("atm_iv", normalize_range(avg_iv, 10, 30), ts)
            
            # IV percentile (simplified: map 10-30 range to 0-100)
            iv_pctl = max(0, min(100, (avg_iv - 10) / 20 * 100))
            state.set_dimension("iv_percentile", iv_pctl, normalize_range(iv_pctl, 20, 80))
        
        # IV skew from GEX data
        # (avgPutIV - avgCallIV would need separate tracking; use what's available)
    
    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE: Build from FII/DII
    # ══════════════════════════════════════════════════════════════════════════
    
    def _build_from_fiidii(self, state: MarketState, data: dict, tracker: VelocityTracker, ts: float):
        fii = data.get("fii", 0)
        dii = data.get("dii", 0)
        
        if fii is not None:
            # FII buying > 500 Cr = bullish, selling < -500 = bearish
            norm_fii = normalize_pct(fii, 0, 1500)
            state.set_dimension("fii_flow", fii, norm_fii)
            tracker.update("fii_flow", norm_fii, ts)
        
        if dii is not None:
            norm_dii = normalize_pct(dii, 0, 1500)
            state.set_dimension("dii_flow", dii, norm_dii)
    
    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE: Time context
    # ══════════════════════════════════════════════════════════════════════════
    
    def _build_time_context(self, state: MarketState):
        import datetime
        try:
            from zoneinfo import ZoneInfo
            ist = datetime.datetime.now(ZoneInfo("Asia/Kolkata"))
        except Exception:
            import os
            os.environ.setdefault("TZ", "Asia/Kolkata")
            ist = datetime.datetime.now()
        
        market_open_min = 9 * 60 + 15  # 09:15
        current_min = ist.hour * 60 + ist.minute
        session_min = max(0, current_min - market_open_min)
        
        state.set_dimension("session_minutes", session_min, normalize_range(session_min, 0, 375))
        state.set_dimension("day_of_week", ist.isoweekday(), normalize_range(ist.isoweekday(), 1, 5))
        
        # Session phase
        if session_min <= 15:
            phase = 0.0  # Opening
        elif session_min <= 120:
            phase = 0.25  # Morning
        elif session_min <= 240:
            phase = 0.5  # Midday
        elif session_min <= 330:
            phase = 0.75  # Afternoon
        else:
            phase = 1.0  # Closing
        state.set_dimension("session_phase", phase, normalize_range(phase, 0, 1))
    
    # ══════════════════════════════════════════════════════════════════════════
    # INDICATOR CALCULATIONS (self-contained, no external deps)
    # ══════════════════════════════════════════════════════════════════════════
    
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
        gains, losses = 0.0, 0.0
        for i in range(1, period + 1):
            d = closes[i] - closes[i-1]
            if d > 0: gains += d
            else: losses -= d
        ag, al = gains / period, losses / period
        prev_rsi = None
        rsi = None
        for i in range(period + 1, len(closes)):
            d = closes[i] - closes[i-1]
            ag = (ag * (period - 1) + (d if d > 0 else 0)) / period
            al = (al * (period - 1) + (-d if d < 0 else 0)) / period
            prev_rsi = rsi
            rsi = 100 - 100 / (1 + ag / al) if al != 0 else 100
        if rsi is None:
            rs = ag / al if al != 0 else 100
            rsi = 100 - 100 / (1 + rs)
        return rsi, prev_rsi
    
    @staticmethod
    def _macd(closes: list, fast=12, slow=26, sig=9) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if len(closes) < slow + sig:
            return None, None, None
        k_f, k_s = 2/(fast+1), 2/(slow+1)
        ef = sum(closes[:fast]) / fast
        es = sum(closes[:slow]) / slow
        for v in closes[fast:]: ef = (v - ef) * k_f + ef
        # Recalculate slow from start
        es = sum(closes[:slow]) / slow
        for v in closes[slow:]: es = (v - es) * k_s + es
        # MACD line history
        ef2 = sum(closes[:fast]) / fast
        es2 = sum(closes[:slow]) / slow
        line = []
        for i, v in enumerate(closes):
            if i < fast: continue
            ef2 = (v - ef2) * k_f + ef2
            if i >= slow:
                es2 = (v - es2) * k_s + es2
                line.append(ef2 - es2)
            elif i == slow - 1:
                es2 = sum(closes[:slow]) / slow
        if len(line) < sig:
            return None, None, None
        k_sig = 2 / (sig + 1)
        sl = sum(line[:sig]) / sig
        for v in line[sig:]: sl = (v - sl) * k_sig + sl
        macd_val = line[-1]
        return macd_val, sl, macd_val - sl
    
    @staticmethod
    def _atr(highs: list, lows: list, closes: list, period: int = 14) -> Optional[float]:
        n = len(highs)
        if n < period + 1:
            return None
        tr = []
        for i in range(1, n):
            tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
        if len(tr) < period:
            return None
        atr = sum(tr[:period]) / period
        for t in tr[period:]:
            atr = (atr * (period - 1) + t) / period
        return atr
    
    @staticmethod
    def _vwap(highs, lows, closes, volumes) -> Optional[float]:
        tpv, tv = 0.0, 0.0
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
            tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
        atr_arr = [0.0] * len(tr)
        atr_arr[period-1] = sum(tr[:period]) / period
        for i in range(period, len(tr)):
            atr_arr[i] = (atr_arr[i-1] * (period-1) + tr[i]) / period
        
        direction = 1
        pu, pl = 0.0, 0.0
        for i in range(period, n):
            av = atr_arr[i-1]
            hl = (highs[i] + lows[i]) / 2
            bu = hl + multiplier * av
            bl = hl - multiplier * av
            u = bu if (bu < pu or closes[i-1] > pu) else pu
            lo = bl if (bl > pl or closes[i-1] < pl) else pl
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
            return None, 0, 0
        tr, plus_dm, minus_dm = [], [], []
        for i in range(1, n):
            up = highs[i] - highs[i-1]
            down = lows[i-1] - lows[i]
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)
            tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
        
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
            return None, 0, 0
        
        adx = sum(dx[:period]) / period
        for v in dx[period:]:
            adx = (adx * (period - 1) + v) / period
        
        li = len(s_tr) - 1
        plus_di = 100 * s_plus[li] / (s_tr[li] or 1)
        minus_di = 100 * s_minus[li] / (s_tr[li] or 1)
        
        return adx, plus_di, minus_di
