"""
SignalsBrain — Python port of the live mean-reversion signal engine

This is a DELIBERATE LINE-BY-LINE PORT of js/live-engine.js as deployed on
ads.sanctify.co.in/signals, not an independent reimplementation. Validating a
reimplementation would prove nothing about production: any difference between the two
would show up as fake alpha (or fake failure). The indicator recurrences below —
Wilder smoothing in RSI and ADX, the SuperTrend band flip — are reproduced with the
same initialisation and the same guards, including the `|| 1` divide-by-zero
fallbacks, so both sides agree bar for bar.

`test_parity_with_js` in the test suite pins this: if someone edits the JS engine and
not this port, parity breaks and the harness stops claiming to measure production.

Ported behaviour (see the JS file for the measurement notes behind each gate):
    components  reversion(25) + extremeFade(20) + rsiFade(10) [+ options/inst/news]
    direction   net > +30 => BUY (fade the dip), net < -30 => SELL (fade the rip)
    gates       ADX >= 25 required; |net| >= 50 refused as re-pricing, not stretch
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

# ── engine constants, mirrored from live-engine.js ──────────────────────────────
MR_DIR_TH = 30.0        # conviction floor
MR_EXTREME = 50.0       # above this the move is re-pricing, not over-extension
ADX_MIN_FADE = 25.0     # sub-25 bands measured negative expectancy
W_REVERSION, W_EXTREME_FADE, W_RSI_FADE = 25.0, 20.0, 10.0
W_OPTIONS, W_VOLUME, W_INSTITUTIONAL, W_NEWS = 20.0, 0.0, 10.0, 5.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ── indicators (ported) ────────────────────────────────────────────────────────
def sma(v: Sequence[float], p: int) -> Optional[float]:
    if not v or len(v) < p:
        return None
    return sum(v[-p:]) / p


def ema_series(v: Sequence[float], p: int) -> list[float]:
    if not v or len(v) < p:
        return []
    k = 2.0 / (p + 1.0)
    e = sum(v[:p]) / p
    out = [e]
    for x in v[p:]:
        e = x * k + e * (1 - k)
        out.append(e)
    return out


def ema(v: Sequence[float], p: int) -> Optional[float]:
    s = ema_series(v, p)
    return s[-1] if s else None


def rsi(c: Sequence[float], p: int = 14) -> Optional[float]:
    """Wilder RSI. Mirrors the JS seeding and the al==0 -> 100 guard exactly."""
    if not c or len(c) < p + 1:
        return None
    g = l = 0.0
    for i in range(1, p + 1):
        d = c[i] - c[i - 1]
        if d > 0:
            g += d
        else:
            l -= d
    ag, al = g / p, l / p
    latest: Optional[float] = None
    for i in range(p + 1, len(c)):
        d = c[i] - c[i - 1]
        ag = (ag * (p - 1) + (d if d > 0 else 0.0)) / p
        al = (al * (p - 1) + (-d if d < 0 else 0.0)) / p
        latest = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    if latest is None:
        rs = 100.0 if al == 0 else ag / al
        latest = 100.0 - 100.0 / (1.0 + rs)
    return latest


def atr(h: Sequence[float], l: Sequence[float], c: Sequence[float], p: int = 14) -> Optional[float]:
    n = len(c)
    if n < p + 1:
        return None
    trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, n)]
    a = sum(trs[:p]) / p
    for i in range(p, len(trs)):
        a = (a * (p - 1) + trs[i]) / p
    return a


def adx(h: Sequence[float], l: Sequence[float], c: Sequence[float], p: int = 14) -> dict:
    """Wilder ADX/DI. Keeps the JS `(x || 1)` denominators so both sides match."""
    n = len(h)
    if n < 2 * p + 1:
        return {"adx": None, "plusDI": None, "minusDI": None}
    tr: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, n):
        up, down = h[i] - h[i - 1], l[i - 1] - l[i]
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))

    def smooth(arr: list[float]) -> list[float]:
        s = sum(arr[:p])
        out = [s]
        for i in range(p, len(arr)):
            s = s - s / p + arr[i]
            out.append(s)
        return out

    st, sp, sm = smooth(tr), smooth(plus_dm), smooth(minus_dm)
    dx: list[float] = []
    for i in range(len(st)):
        denom = st[i] if st[i] else 1.0
        p_di, m_di = 100.0 * sp[i] / denom, 100.0 * sm[i] / denom
        s = (p_di + m_di) or 1.0
        dx.append(100.0 * abs(p_di - m_di) / s)
    if len(dx) < p:
        return {"adx": None, "plusDI": None, "minusDI": None}
    a = sum(dx[:p]) / p
    for i in range(p, len(dx)):
        a = (a * (p - 1) + dx[i]) / p
    li = len(st) - 1
    d = st[li] if st[li] else 1.0
    return {"adx": a, "plusDI": 100.0 * sp[li] / d, "minusDI": 100.0 * sm[li] / d}


def bollinger(c: Sequence[float], p: int = 20, sd: float = 2.0) -> dict:
    if not c or len(c) < p:
        return {"upper": None, "mid": None, "lower": None}
    s = c[-p:]
    mid = sum(s) / p
    var = sum((x - mid) ** 2 for x in s) / p
    std = math.sqrt(var)
    return {"upper": mid + sd * std, "mid": mid, "lower": mid - sd * std}


def super_trend(h: Sequence[float], l: Sequence[float], c: Sequence[float],
                p: int = 10, m: float = 3.0) -> str:
    n = len(c)
    if n < p + 1:
        return "bearish"
    tr = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, n)]
    a_arr = [0.0] * len(tr)
    a = sum(tr[:p]) / p
    a_arr[p - 1] = a
    for i in range(p, len(tr)):
        a = (a * (p - 1) + tr[i]) / p
        a_arr[i] = a
    direction, pu, pl = 1, 0.0, 0.0
    for i in range(p, n):
        av, hl = a_arr[i - 1], (h[i] + l[i]) / 2.0
        bu, bl = hl + m * av, hl - m * av
        u = bu if (bu < pu or c[i - 1] > pu) else pu
        lo = bl if (bl > pl or c[i - 1] < pl) else pl
        if direction == 1 and c[i] < lo:
            direction = -1
        elif direction == -1 and c[i] > u:
            direction = 1
        pu, pl = u, lo
    return "bullish" if direction == 1 else "bearish"


# ── signal ─────────────────────────────────────────────────────────────────────
@dataclass
class Signal:
    """Mirrors the fields of the JS signal object that validation actually consumes."""
    symbol: str
    direction: str = "NO_TRADE"
    net_score: float = 0.0
    confidence: float = 0.0
    ltp: float = 0.0
    atr: Optional[float] = None
    rsi: Optional[float] = None
    adx: Optional[float] = None
    regime: str = "Unknown"
    components: dict = field(default_factory=dict)
    vetoes: list[str] = field(default_factory=list)
    agreement: float = 0.0
    bar_index: int = -1
    timestamp: Optional[str] = None

    @property
    def actionable(self) -> bool:
        return self.direction in ("BUY", "SELL")

    @property
    def option_type(self) -> str:
        """BUY fades a dip => long CE. SELL fades a rip => long PE."""
        return "CE" if self.direction == "BUY" else "PE"


def generate_signal(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
                    symbol: str = "NIFTY", confidence_threshold: float = 60.0,
                    options_bias: float = 0.0, institutional_bias: float = 0.0,
                    news_bias: float = 0.0, mtf_bias: Optional[int] = None,
                    bar_index: int = -1, timestamp: Optional[str] = None) -> Signal:
    """
    Port of generateSignal()'s mean-reversion core.

    Restricted to the price-derived components plus optional externally-supplied
    options/institutional/news biases. The market-hours gate, live-premium checks and
    the option-chain liquidity veto are intentionally excluded: they depend on live
    state that does not exist historically, and faking them would be look-ahead.
    """
    sig = Signal(symbol=symbol, bar_index=bar_index, timestamp=timestamp)
    if not closes or len(closes) < 30:
        sig.vetoes.append("INSUFFICIENT_DATA")
        return sig

    ltp = closes[-1]
    sig.ltp = ltp
    e21 = ema(closes, 21)
    a = atr(highs, lows, closes) or ltp * 0.012
    r = rsi(closes)
    sig.atr, sig.rsi = a, r

    comp: dict[str, dict] = {}

    # Reversion (25): stretch from the 21-EMA in ATRs, inverted.
    stretch = ((ltp - e21) / a) if (e21 is not None and a) else 0.0
    comp["reversion"] = {"bias": clamp(-stretch / 2.2, -1, 1), "weight": W_REVERSION}

    # Extreme fade (20): 20-bar range position + Bollinger tags.
    pb = 0.0
    if len(closes) >= 20:
        h20, l20 = max(highs[-20:]), min(lows[-20:])
        rng = h20 - l20
        if rng > 0:
            pos = (ltp - l20) / rng
            if pos >= 0.85:
                pb -= 0.4
            elif pos <= 0.15:
                pb += 0.4
            else:
                pb += (0.5 - pos) * 0.5
    bb = bollinger(closes, 20, 2)
    if bb["upper"] is not None:
        if ltp >= bb["upper"]:
            pb -= 0.35
        elif ltp <= bb["lower"]:
            pb += 0.35
    comp["extremeFade"] = {"bias": clamp(pb, -1, 1), "weight": W_EXTREME_FADE}

    # RSI fade (10): overbought => fade short, oversold => fade long.
    mb = 0.0
    if r is not None:
        if r >= 70:
            mb -= 0.6
        elif r >= 62:
            mb -= 0.3
        elif r <= 30:
            mb += 0.6
        elif r <= 38:
            mb += 0.3
    comp["rsiFade"] = {"bias": clamp(mb, -1, 1), "weight": W_RSI_FADE}

    comp["options"] = {"bias": clamp(options_bias, -1, 1), "weight": W_OPTIONS}
    # Index spot volume is always zero (NSE publishes none), so weight 0 — matching
    # production, where volume is measured from the option chain as a liquidity gate
    # and deliberately not scored into direction.
    comp["volume"] = {"bias": 0.0, "weight": W_VOLUME, "available": False}
    comp["institutional"] = {"bias": clamp(institutional_bias, -1, 1), "weight": W_INSTITUTIONAL}
    comp["news"] = {"bias": clamp(news_bias, -1, 1), "weight": W_NEWS}

    w_sum = sum(c["weight"] for c in comp.values()) or 1.0
    norm = 100.0 / w_sum
    net = clamp(sum(c["bias"] * c["weight"] * norm for c in comp.values()), -100, 100)
    sig.net_score, sig.components = net, comp

    dom = 1 if net >= 0 else -1
    opinions = [c for c in comp.values() if c["weight"] > 0 and c["bias"] != 0]
    agree = (sum(1 for c in opinions if (1 if c["bias"] > 0 else -1) == dom) / len(opinions)) if opinions else 0.0
    sig.agreement = agree

    conf = min(99.0, abs(net) * 0.62 + agree * 42.0)

    direction = "BUY" if net > MR_DIR_TH else ("SELL" if net < -MR_DIR_TH else "NO_TRADE")

    adx_obj = adx(highs, lows, closes)
    adx_v = adx_obj["adx"]
    sig.adx = adx_v
    sig.regime = ("Unknown" if adx_v is None
                  else "Trending" if adx_v >= 25 else "Developing" if adx_v >= 18 else "Choppy")

    if adx_v is not None:
        if adx_v >= 28:
            conf = min(99.0, conf + 10)
        elif adx_v >= 22:
            conf = min(99.0, conf + 5)
        elif adx_v < 16:
            conf = max(0.0, conf - 12)

    if mtf_bias:
        agrees = (mtf_bias > 0 and direction == "BUY") or (mtf_bias < 0 and direction == "SELL")
        conf = clamp(conf + (4 if agrees else -10), 0, 99)

    # Gate: regime (measured negative expectancy below ADX 25).
    if direction != "NO_TRADE" and adx_v is not None and adx_v < ADX_MIN_FADE:
        sig.vetoes.append(f"WEAK_TREND_NO_FADE: ADX {adx_v:.0f} < {ADX_MIN_FADE:.0f}")
        direction = "NO_TRADE"

    # Gate: extreme stretch is re-pricing, not exhaustion.
    if direction != "NO_TRADE" and abs(net) >= MR_EXTREME:
        sig.vetoes.append(f"EXTREME_STRETCH_NO_FADE: net {net:.0f} beyond +/-{MR_EXTREME:.0f}")
        direction = "NO_TRADE"

    # RSI hard risk filter (retained from production).
    if r is not None:
        if r > 85 and direction == "BUY":
            sig.vetoes.append("RSI_OVERBOUGHT")
            direction = "NO_TRADE"
        if r < 15 and direction == "SELL":
            sig.vetoes.append("RSI_OVERSOLD")
            direction = "NO_TRADE"

    if direction != "NO_TRADE" and conf < confidence_threshold:
        sig.vetoes.append(f"LOW_CONFIDENCE: {conf:.0f}% < {confidence_threshold:.0f}%")
        direction = "NO_TRADE"

    sig.direction, sig.confidence = direction, conf
    return sig
