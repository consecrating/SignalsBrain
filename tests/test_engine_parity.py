"""
Parity tests: brain.validation.engine_port vs the deployed js/live-engine.js

Why this file matters more than it looks
---------------------------------------
The Python engine exists so validation measures PRODUCTION behaviour. The moment the two
drift apart, every number the harness reports becomes a statement about code nobody is
running. These tests execute both implementations over identical synthetic series and
require the indicators to agree to tight tolerance and the direction/veto decisions to
agree exactly.

The JS side is invoked through node against the real repo file, so this fails loudly if
someone edits one engine and not the other. When the sibling repo or node is unavailable
the JS comparisons skip rather than fail — but the pure-Python behavioural tests below
still run, so the suite is never silently empty.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile

import pytest

from brain.validation import engine_port as E

JS_ENGINE = os.environ.get(
    "LIVE_ENGINE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "Signals-Live-Website", "js", "live-engine.js"),
)
HAVE_JS = os.path.isfile(JS_ENGINE) and shutil.which("node") is not None
skip_js = pytest.mark.skipif(not HAVE_JS, reason="node or live-engine.js unavailable")


def synth_series(kind: str, n: int = 120) -> tuple[list[float], list[float], list[float]]:
    """Deterministic series with no RNG, so both languages see identical input."""
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    p = 20000.0
    for i in range(n):
        if kind == "uptrend":
            p += 12.0
        elif kind == "downtrend":
            p -= 12.0
        elif kind == "chop":
            p += 25.0 * math.sin(i / 2.0)
        elif kind == "spike":
            p += 60.0 if i > n - 12 else 3.0 * math.sin(i / 3.0)
        else:
            p += 2.0 * math.cos(i / 5.0)
        closes.append(p)
        highs.append(p + 15.0)
        lows.append(p - 15.0)
    return highs, lows, closes


def run_js(highs, lows, closes, symbol="NIFTY", threshold=60.0) -> dict:
    """Call the deployed engine with the market-hours gate neutralised for replay."""
    with open(JS_ENGINE, "r", encoding="utf-8") as fh:
        code = fh.read()
    code = code.replace(
        "if (_dayCheck === 0 || _dayCheck === 6 || _minCheck < 555 || _minCheck > 930) {",
        "if (false) {")
    with tempfile.TemporaryDirectory() as d:
        mod = os.path.join(d, "engine.mjs")
        with open(mod, "w", encoding="utf-8") as fh:
            fh.write(code)
        driver = os.path.join(d, "run.mjs")
        payload = json.dumps({"highs": highs, "lows": lows, "closes": closes,
                              "symbol": symbol, "threshold": threshold})
        with open(driver, "w", encoding="utf-8") as fh:
            fh.write(
                "import {generateSignal, RSI, ATR, ADX, EMA, bollinger, superTrend} "
                f"from {json.dumps(mod)};\n"
                f"const p = {payload};\n"
                "const snap = {symbol:p.symbol, opens:p.closes, highs:p.highs, lows:p.lows,\n"
                "  closes:p.closes, volumes:p.closes.map(()=>0), timestamps:p.closes.map((_,i)=>String(i)),\n"
                "  ltp:p.closes[p.closes.length-1], pcr:null, iv:0.15, fiiFlow:null, newsSentiment:null,\n"
                "  isIndex:true, lotSize:65, strikeStep:50, optionStrike:null, livePremium:null};\n"
                "const s = generateSignal(snap, {confidenceThreshold:p.threshold});\n"
                "const a = ADX(p.highs,p.lows,p.closes);\n"
                "console.log(JSON.stringify({direction:s.direction, netScore:s.netScore,\n"
                "  vetoes:(s.vetoes||[]).map(v=>String(v).split(':')[0]),\n"
                "  rsi: RSI(p.closes).latest, atr: ATR(p.highs,p.lows,p.closes),\n"
                "  adx: a.adx, ema21: EMA(p.closes,21), bbUpper: bollinger(p.closes,20,2).upper,\n"
                "  st: superTrend(p.highs,p.lows,p.closes)}));\n")
        out = subprocess.run(["node", driver], capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            raise RuntimeError(f"node failed: {out.stderr[:600]}")
        return json.loads(out.stdout.strip().splitlines()[-1])


@skip_js
@pytest.mark.parametrize("kind", ["uptrend", "downtrend", "chop", "spike", "drift"])
class TestIndicatorParity:
    def test_indicators_match_js(self, kind):
        h, l, c = synth_series(kind)
        js = run_js(h, l, c)
        assert E.rsi(c) == pytest.approx(js["rsi"], rel=1e-9, abs=1e-9)
        assert E.atr(h, l, c) == pytest.approx(js["atr"], rel=1e-9, abs=1e-9)
        assert E.ema(c, 21) == pytest.approx(js["ema21"], rel=1e-9, abs=1e-9)
        assert E.bollinger(c, 20, 2)["upper"] == pytest.approx(js["bbUpper"], rel=1e-9, abs=1e-9)
        assert E.super_trend(h, l, c) == js["st"]
        py_adx = E.adx(h, l, c)["adx"]
        if js["adx"] is None:
            assert py_adx is None
        else:
            assert py_adx == pytest.approx(js["adx"], rel=1e-9, abs=1e-9)


@skip_js
@pytest.mark.parametrize("kind", ["uptrend", "downtrend", "chop", "spike", "drift"])
class TestSignalParity:
    def test_direction_and_score_match_js(self, kind):
        h, l, c = synth_series(kind)
        js = run_js(h, l, c)
        py = E.generate_signal(h, l, c, symbol="NIFTY", confidence_threshold=60.0)
        assert py.net_score == pytest.approx(js["netScore"], abs=0.05), (
            f"{kind}: net {py.net_score} vs JS {js['netScore']}")
        assert py.direction == js["direction"], f"{kind}: {py.direction} vs JS {js['direction']}"

    def test_veto_codes_match_js(self, kind):
        h, l, c = synth_series(kind)
        js = run_js(h, l, c)
        py = E.generate_signal(h, l, c, symbol="NIFTY", confidence_threshold=60.0)
        py_codes = {v.split(":")[0] for v in py.vetoes}
        js_codes = set(js["vetoes"])
        # The port omits live-only gates (market hours, live premium, chain liquidity),
        # so JS may legitimately add codes; it must never MISS one the port raises.
        shared = {"WEAK_TREND_NO_FADE", "EXTREME_STRETCH_NO_FADE", "RSI_OVERBOUGHT", "RSI_OVERSOLD"}
        assert (py_codes & shared) == (js_codes & shared), f"{kind}: {py_codes} vs {js_codes}"


class TestPortBehaviour:
    """Pure-Python checks — these run even without node."""

    def test_fade_direction_is_contrarian(self):
        """A rising series must produce SELL (fade the rip) and vice versa."""
        h, l, c = synth_series("uptrend")
        up = E.generate_signal(h, l, c, confidence_threshold=0.0)
        h2, l2, c2 = synth_series("downtrend")
        down = E.generate_signal(h2, l2, c2, confidence_threshold=0.0)
        assert up.net_score < 0
        assert down.net_score > 0

    def test_option_type_follows_direction(self):
        s = E.Signal(symbol="NIFTY", direction="BUY")
        assert s.option_type == "CE"
        s.direction = "SELL"
        assert s.option_type == "PE"

    def test_weak_trend_is_vetoed(self):
        """
        ADX below 25 must block, since sub-25 bands measured negative expectancy.

        The veto is only RECORDED when there was a direction to cancel — a setup that
        already failed the conviction floor is NO_TRADE before the regime gate is
        reached, and inventing a veto for it would misreport why it was refused.
        """
        h, l, c = synth_series("chop", 140)
        sig = E.generate_signal(h, l, c, confidence_threshold=0.0)
        if sig.adx is not None and sig.adx < E.ADX_MIN_FADE:
            assert sig.direction == "NO_TRADE"
            if abs(sig.net_score) > E.MR_DIR_TH:
                assert any("WEAK_TREND_NO_FADE" in v for v in sig.vetoes)

    def test_weak_trend_veto_fires_on_a_qualifying_setup(self):
        """Construct the case directly: strong conviction but a sub-25 ADX."""
        found = False
        for kind in ("chop", "drift", "spike", "uptrend", "downtrend"):
            for n in range(90, 200, 7):
                h, l, c = synth_series(kind, n)
                s = E.generate_signal(h, l, c, confidence_threshold=0.0)
                if (s.adx is not None and s.adx < E.ADX_MIN_FADE
                        and E.MR_DIR_TH < abs(s.net_score) < E.MR_EXTREME):
                    assert s.direction == "NO_TRADE"
                    assert any("WEAK_TREND_NO_FADE" in v for v in s.vetoes)
                    found = True
                    break
            if found:
                break
        if not found:
            pytest.skip("no synthetic series produced conviction with sub-25 ADX")

    def test_extreme_stretch_is_vetoed(self):
        h, l, c = synth_series("spike", 140)
        sig = E.generate_signal(h, l, c, confidence_threshold=0.0)
        if abs(sig.net_score) >= E.MR_EXTREME:
            assert sig.direction == "NO_TRADE"
            assert any("EXTREME_STRETCH_NO_FADE" in v for v in sig.vetoes)

    def test_conviction_floor_respected(self):
        h, l, c = synth_series("drift", 140)
        sig = E.generate_signal(h, l, c, confidence_threshold=0.0)
        if abs(sig.net_score) <= E.MR_DIR_TH:
            assert sig.direction == "NO_TRADE"

    def test_insufficient_data_is_handled(self):
        sig = E.generate_signal([1.0] * 5, [1.0] * 5, [1.0] * 5)
        assert sig.direction == "NO_TRADE"
        assert "INSUFFICIENT_DATA" in sig.vetoes

    def test_index_volume_carries_zero_weight(self):
        """Mirrors production: NSE publishes no index volume, so it cannot be scored."""
        h, l, c = synth_series("uptrend")
        sig = E.generate_signal(h, l, c, confidence_threshold=0.0)
        assert sig.components["volume"]["weight"] == 0.0
