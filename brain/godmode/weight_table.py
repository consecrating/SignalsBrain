"""
SignalsBrain — Learned Weight Table

Closes the loop that was previously open: learning never reached scoring.

`SelfImproveEngine` did write weight adjustments, but the only reader was
`orchestrator.py`, which used them to nudge the FINAL confidence scalar after the
fact. `DIMENSIONS[*].weight` is a frozen dataclass field and was never updated,
so what the engine learned had no effect on how a state was actually scored. A
factor that history showed to be worthless kept its full influence on
`direction_score`; only the summary number moved.

This module supplies a versioned, mutable multiplier table keyed by
`regime | gex_regime | dimension`, which `MarketState.compute_composites` and
`attribution.attribute` both consult. Because both read the same effective
weights, the attribution invariant (`sum(contributions) + external == final`)
continues to hold when weights are non-default.

Every table carries a version, and every emitted signal records the version that
produced it. Without that, a later drift investigation cannot attribute error to
the weights that actually made the call.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

# Multipliers are bounded. A learned weight may meaningfully de-emphasise or
# emphasise a dimension, but it must not be able to invert its meaning or let a
# single dimension dominate the score.
MIN_MULTIPLIER = 0.25
MAX_MULTIPLIER = 2.00

# Applied when no cell matches. Identity, so an untrained system behaves exactly
# as the hand-specified weights intend.
DEFAULT_MULTIPLIER = 1.0


def cell_key(regime: str, gex_regime: str, dimension: str) -> str:
    return f"{regime or 'unknown'}|{gex_regime or 'unknown'}|{dimension}"


@dataclass
class WeightCell:
    """One learned multiplier plus the evidence that justifies it."""
    regime: str
    gex_regime: str
    dimension: str
    multiplier: float = DEFAULT_MULTIPLIER
    n_train: int = 0
    n_holdout: int = 0
    # Change in holdout log-loss when this dimension is removed. Positive means
    # removing it hurt, i.e. the dimension carries genuine information.
    ablation_delta: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "cell": cell_key(self.regime, self.gex_regime, self.dimension),
            "multiplier": round(self.multiplier, 4),
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "ablation_delta": round(self.ablation_delta, 6),
            "p_value": round(self.p_value, 5),
            "significant": self.significant,
            "reason": self.reason,
        }


@dataclass
class WeightTable:
    """A versioned set of learned multipliers."""
    version: int = 0
    cells: dict[str, WeightCell] = field(default_factory=dict)
    trained_at: float = 0.0
    n_trades: int = 0
    notes: str = ""

    # ──────────────────────────────────────────────────────────────────────────

    def multiplier(self, regime: str, gex_regime: str, dimension: str) -> float:
        """
        Look up a multiplier, falling back progressively.

        Exact cell -> regime-agnostic cell -> identity. A dimension learned in
        "Trending|Negative" should not silently govern "Choppy|Positive".
        """
        exact = self.cells.get(cell_key(regime, gex_regime, dimension))
        if exact is not None and exact.significant:
            return exact.multiplier
        anyk = self.cells.get(cell_key("any", "any", dimension))
        if anyk is not None and anyk.significant:
            return anyk.multiplier
        return DEFAULT_MULTIPLIER

    def resolver(self, regime: str, gex_regime: str) -> Callable[[str], float]:
        """A single-argument lookup for the scoring path."""
        def _resolve(dimension: str) -> float:
            return self.multiplier(regime, gex_regime, dimension)
        return _resolve

    def significant_cells(self) -> list[WeightCell]:
        return [c for c in self.cells.values() if c.significant]

    def set_cell(self, cell: WeightCell):
        cell.multiplier = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, cell.multiplier))
        self.cells[cell_key(cell.regime, cell.gex_regime, cell.dimension)] = cell

    # ──────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "trained_at": self.trained_at,
            "n_trades": self.n_trades,
            "notes": self.notes,
            "significant_cells": len(self.significant_cells()),
            "cells": {k: asdict(v) for k, v in self.cells.items()},
        }

    def summary(self) -> dict:
        """Compact view for the API and dashboards."""
        sig = sorted(self.significant_cells(),
                     key=lambda c: abs(c.multiplier - 1.0), reverse=True)
        return {
            "version": self.version,
            "trained_at": self.trained_at,
            "n_trades": self.n_trades,
            "cells_total": len(self.cells),
            "cells_significant": len(sig),
            "notes": self.notes,
            "top_adjustments": [c.to_dict() for c in sig[:15]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeightTable":
        t = cls(
            version=int(d.get("version", 0)),
            trained_at=float(d.get("trained_at", 0.0)),
            n_trades=int(d.get("n_trades", 0)),
            notes=str(d.get("notes", "")),
        )
        for k, raw in (d.get("cells") or {}).items():
            try:
                t.cells[k] = WeightCell(**raw)
            except TypeError:
                continue
        return t

    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "WeightTable":
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, ValueError, TypeError):
            return cls()
