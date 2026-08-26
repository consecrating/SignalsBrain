"""
SignalsBrain — Probability Calibration

`confidence` was never a probability. It was an arbitrary 0-99 figure produced
by summing hand-tuned modifiers, never compared against realised outcomes. "78%
confidence" did not mean 78% of such signals win, and nothing in the codebase
could have told you what it did mean.

That matters for two reasons:
  * Position sizing needs P(win). A number that saturates at 99 for every
    net_bias above 60 cannot rank two setups, let alone size them.
  * Without calibration there is no way to detect that the model has drifted.

This module fits a monotone mapping from raw score to realised win probability
and reports the standard scoring metrics (Brier score, log loss, reliability
table, expected calibration error).

Two fitters:
  PlattCalibrator    — logistic regression on the raw score. Two parameters, so
                       it is usable from ~50 samples. Fitted by Newton-Raphson.
  IsotonicCalibrator — non-parametric monotone fit via pool-adjacent-violators.
                       Needs more data but captures non-linear reliability.

Both are pure Python (no sklearn dependency) and both degrade to the identity
mapping when there is not enough data, so an uncalibrated system reports its
raw score rather than a fabricated probability.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence

MIN_SAMPLES_PLATT = 50
MIN_SAMPLES_ISOTONIC = 200


def _clip(p: float, eps: float = 1e-12) -> float:
    return max(eps, min(1.0 - eps, p))


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_predicted: float
    observed_rate: float

    @property
    def gap(self) -> float:
        return self.observed_rate - self.mean_predicted

    def to_dict(self) -> dict:
        return {
            "range": [round(self.lo, 3), round(self.hi, 3)],
            "n": self.count,
            "predicted": round(self.mean_predicted, 4),
            "observed": round(self.observed_rate, 4),
            "gap": round(self.gap, 4),
        }


@dataclass
class CalibrationMetrics:
    n: int = 0
    brier: float = 0.0
    log_loss: float = 0.0
    ece: float = 0.0            # expected calibration error
    base_rate: float = 0.0
    mean_prediction: float = 0.0
    bins: list[ReliabilityBin] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "brier": round(self.brier, 5),
            "log_loss": round(self.log_loss, 5),
            "expected_calibration_error": round(self.ece, 5),
            "base_rate": round(self.base_rate, 4),
            "mean_prediction": round(self.mean_prediction, 4),
            "reliability": [b.to_dict() for b in self.bins],
        }


def evaluate(predicted: Sequence[float], outcomes: Sequence[int],
             n_bins: int = 10) -> CalibrationMetrics:
    """
    Score a set of probabilistic predictions against binary outcomes.

    Brier score: mean squared error of the probability. Lower is better; 0.25 is
    what you get by always predicting 0.5. A model whose Brier score exceeds the
    base-rate baseline has negative skill and should not be sizing positions.
    """
    m = CalibrationMetrics()
    pairs = [(float(p), int(o)) for p, o in zip(predicted, outcomes)
             if p == p and o in (0, 1)]
    if not pairs:
        return m
    m.n = len(pairs)
    m.base_rate = sum(o for _, o in pairs) / m.n
    m.mean_prediction = sum(p for p, _ in pairs) / m.n
    m.brier = sum((p - o) ** 2 for p, o in pairs) / m.n
    m.log_loss = -sum(o * math.log(_clip(p)) + (1 - o) * math.log(_clip(1 - p))
                      for p, o in pairs) / m.n

    # Reliability table + ECE
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        sel = [(p, o) for p, o in pairs if (lo <= p < hi or (i == n_bins - 1 and p == 1.0))]
        if not sel:
            continue
        mp = sum(p for p, _ in sel) / len(sel)
        obs = sum(o for _, o in sel) / len(sel)
        m.bins.append(ReliabilityBin(lo=lo, hi=hi, count=len(sel),
                                     mean_predicted=mp, observed_rate=obs))
        ece += (len(sel) / m.n) * abs(obs - mp)
    m.ece = ece
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# PLATT (logistic) CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlattCalibrator:
    """
    p = sigmoid(a * x + b), where x is the raw score scaled to [0,1].

    Fitted by Newton-Raphson on the log-likelihood with a small ridge term for
    numerical stability. `fitted` stays False until MIN_SAMPLES_PLATT outcomes
    exist, and `predict` then returns the raw score unchanged rather than
    inventing a probability.
    """
    a: float = 1.0
    b: float = 0.0
    fitted: bool = False
    n_samples: int = 0

    def fit(self, scores: Sequence[float], outcomes: Sequence[int],
            iterations: int = 100, ridge: float = 1e-6) -> "PlattCalibrator":
        pairs = [(max(0.0, min(1.0, float(s) / 100.0)), int(o))
                 for s, o in zip(scores, outcomes) if s == s and o in (0, 1)]
        self.n_samples = len(pairs)
        if len(pairs) < MIN_SAMPLES_PLATT:
            self.fitted = False
            return self
        # Degenerate label set: nothing to learn.
        ys = {o for _, o in pairs}
        if len(ys) < 2:
            self.fitted = False
            return self

        a, b = 1.0, 0.0
        for _ in range(iterations):
            g_a = g_b = h_aa = h_ab = h_bb = 0.0
            for x, y in pairs:
                z = a * x + b
                p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
                d = p - y
                w = p * (1.0 - p)
                g_a += d * x
                g_b += d
                h_aa += w * x * x
                h_ab += w * x
                h_bb += w
            h_aa += ridge
            h_bb += ridge
            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-14:
                break
            da = (g_a * h_bb - g_b * h_ab) / det
            db = (g_b * h_aa - g_a * h_ab) / det
            a -= da
            b -= db
            if abs(da) < 1e-9 and abs(db) < 1e-9:
                break
        self.a, self.b, self.fitted = a, b, True
        return self

    def predict(self, score: float) -> float:
        x = max(0.0, min(1.0, float(score) / 100.0))
        if not self.fitted:
            return x
        z = max(-60.0, min(60.0, self.a * x + self.b))
        return 1.0 / (1.0 + math.exp(-z))

    def to_dict(self) -> dict:
        return {"kind": "platt", "a": self.a, "b": self.b,
                "fitted": self.fitted, "n_samples": self.n_samples}

    @classmethod
    def from_dict(cls, d: dict) -> "PlattCalibrator":
        return cls(a=float(d.get("a", 1.0)), b=float(d.get("b", 0.0)),
                   fitted=bool(d.get("fitted", False)),
                   n_samples=int(d.get("n_samples", 0)))


# ═══════════════════════════════════════════════════════════════════════════════
# ISOTONIC CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IsotonicCalibrator:
    """
    Non-parametric monotone fit via pool-adjacent-violators (PAVA).

    Stored as breakpoints so it serialises cleanly and needs no runtime deps.
    """
    x_points: list[float] = field(default_factory=list)
    y_points: list[float] = field(default_factory=list)
    fitted: bool = False
    n_samples: int = 0

    def fit(self, scores: Sequence[float], outcomes: Sequence[int]) -> "IsotonicCalibrator":
        pairs = sorted(((max(0.0, min(1.0, float(s) / 100.0)), int(o))
                        for s, o in zip(scores, outcomes) if s == s and o in (0, 1)),
                       key=lambda t: t[0])
        self.n_samples = len(pairs)
        if len(pairs) < MIN_SAMPLES_ISOTONIC:
            self.fitted = False
            return self

        xs = [p[0] for p in pairs]
        ys = [float(p[1]) for p in pairs]
        # PAVA: merge adjacent blocks that violate monotonicity.
        vals = list(ys)
        wts = [1.0] * len(ys)
        idx = list(range(len(ys)))
        i = 0
        blocks: list[tuple[float, float, int, int]] = [(vals[k], wts[k], k, k) for k in idx]
        merged = True
        while merged:
            merged = False
            out: list[tuple[float, float, int, int]] = []
            k = 0
            while k < len(blocks):
                cur = blocks[k]
                while k + 1 < len(blocks) and cur[0] > blocks[k + 1][0]:
                    nxt = blocks[k + 1]
                    tw = cur[1] + nxt[1]
                    cur = ((cur[0] * cur[1] + nxt[0] * nxt[1]) / tw, tw, cur[2], nxt[3])
                    k += 1
                    merged = True
                out.append(cur)
                k += 1
            blocks = out
        self.x_points = [xs[b[3]] for b in blocks]
        self.y_points = [b[0] for b in blocks]
        self.fitted = True
        return self

    def predict(self, score: float) -> float:
        x = max(0.0, min(1.0, float(score) / 100.0))
        if not self.fitted or not self.x_points:
            return x
        if x <= self.x_points[0]:
            return self.y_points[0]
        if x >= self.x_points[-1]:
            return self.y_points[-1]
        # Linear interpolation between breakpoints.
        for i in range(1, len(self.x_points)):
            if x <= self.x_points[i]:
                x0, x1 = self.x_points[i - 1], self.x_points[i]
                y0, y1 = self.y_points[i - 1], self.y_points[i]
                if x1 == x0:
                    return y1
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return self.y_points[-1]

    def to_dict(self) -> dict:
        return {"kind": "isotonic", "x": self.x_points, "y": self.y_points,
                "fitted": self.fitted, "n_samples": self.n_samples}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicCalibrator":
        return cls(x_points=list(d.get("x", [])), y_points=list(d.get("y", [])),
                   fitted=bool(d.get("fitted", False)),
                   n_samples=int(d.get("n_samples", 0)))


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPER (persistable, versioned)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CalibrationModel:
    """
    A fitted calibrator plus the metrics that justify trusting it.

    Versioned so every emitted signal can record which calibration produced its
    probability; without that, a later drift investigation cannot attribute
    error to the model that actually made the call.
    """
    version: int = 0
    platt: PlattCalibrator = field(default_factory=PlattCalibrator)
    isotonic: IsotonicCalibrator = field(default_factory=IsotonicCalibrator)
    metrics: Optional[dict] = None
    holdout_metrics: Optional[dict] = None
    trained_at: float = 0.0
    n_train: int = 0
    n_holdout: int = 0

    @property
    def active(self) -> str:
        if self.isotonic.fitted:
            return "isotonic"
        if self.platt.fitted:
            return "platt"
        return "identity"

    def probability(self, score: float) -> float:
        if self.isotonic.fitted:
            return self.isotonic.predict(score)
        if self.platt.fitted:
            return self.platt.predict(score)
        return max(0.0, min(1.0, float(score) / 100.0))

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "active": self.active,
            "platt": self.platt.to_dict(),
            "isotonic": self.isotonic.to_dict(),
            "metrics": self.metrics,
            "holdout_metrics": self.holdout_metrics,
            "trained_at": self.trained_at,
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationModel":
        return cls(
            version=int(d.get("version", 0)),
            platt=PlattCalibrator.from_dict(d.get("platt", {})),
            isotonic=IsotonicCalibrator.from_dict(d.get("isotonic", {})),
            metrics=d.get("metrics"),
            holdout_metrics=d.get("holdout_metrics"),
            trained_at=float(d.get("trained_at", 0.0)),
            n_train=int(d.get("n_train", 0)),
            n_holdout=int(d.get("n_holdout", 0)),
        )


def fit_calibration(scores: Sequence[float], outcomes: Sequence[int],
                    holdout_fraction: float = 0.25,
                    version: int = 1, trained_at: float = 0.0) -> CalibrationModel:
    """
    Fit with a chronological holdout.

    The split is by position, not random, because these are time-ordered trading
    outcomes: a random split would let the future inform the past and produce a
    calibration that looks better than it is.
    """
    pairs = [(float(s), int(o)) for s, o in zip(scores, outcomes)
             if s == s and o in (0, 1)]
    model = CalibrationModel(version=version, trained_at=trained_at)
    if not pairs:
        return model

    split = int(len(pairs) * (1.0 - holdout_fraction))
    split = max(0, min(len(pairs), split))
    train = pairs[:split] if split > 0 else pairs
    hold = pairs[split:] if split < len(pairs) else []

    tr_s = [p for p, _ in train]
    tr_o = [o for _, o in train]
    model.platt = PlattCalibrator().fit(tr_s, tr_o)
    model.isotonic = IsotonicCalibrator().fit(tr_s, tr_o)
    model.n_train = len(train)
    model.n_holdout = len(hold)

    model.metrics = evaluate([model.probability(s) for s in tr_s], tr_o).to_dict()
    if hold:
        h_s = [p for p, _ in hold]
        h_o = [o for _, o in hold]
        model.holdout_metrics = evaluate([model.probability(s) for s in h_s], h_o).to_dict()
    return model


def save_model(model: CalibrationModel, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2))


def load_model(path: Path) -> CalibrationModel:
    if not path.exists():
        return CalibrationModel()
    try:
        return CalibrationModel.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, ValueError, TypeError):
        return CalibrationModel()
