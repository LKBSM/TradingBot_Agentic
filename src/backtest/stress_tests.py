"""Stress testing framework — Sprint 5 batches 5.1-5.4.

Three families of stress tests :

1. **Historical** (5.2) — replays of the algo through known regime
   pathologies : COVID 2020-03 crash, LDI 2022-09 gilts blow-up, SVB
   2023-03, yen 2024-08 carry unwind. The algo is expected to either
   (a) freeze (state machine HOLD throughout), (b) flatten gracefully
   (existing trades hit SL/TP cleanly), or (c) flag risk (RegimeGate
   BLOCK). NOT (d) cascade into multiple losers.

2. **Fuzz** (5.1) — perturbations of OHLCV inputs : NaN injection,
   infinite values, gap insertion (15min, 1h, 1d), spread spikes (2x,
   5x, 10x normal). The algo must not crash and must surface the
   anomaly via :class:`DataQualityValidator`.

3. **Sensitivity** (5.3) — for each tunable hyper-parameter
   (``enter_threshold``, ``exit_threshold``, ``cooldown_bars``, etc.),
   sweep ±20 % and verify that no critical metric (PF, max DD) varies
   by more than 10 %. If it does, the parameter is fragile and must be
   either re-anchored or pinned.

Status (Sprint 5 prep) : **scaffold + dispatch API**. Each individual
stress test is its own ``run_*`` function added in Sprint 5 itself.

Reference
---------
- ``audits/2026-Q2/section_3_7_state_machine.md`` F8 (chaos tests).
- ``audits/2026-Q2/algo_audit_institutional.md`` plan §5 (Sprint 5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Window definitions for historical stress tests (batch 5.2)
# =============================================================================

HISTORICAL_STRESS_WINDOWS = {
    "covid_2020": ("2020-02-15", "2020-04-15"),
    "ldi_gilts_2022": ("2022-09-15", "2022-10-15"),
    "svb_2023": ("2023-03-01", "2023-03-31"),
    "yen_carry_2024": ("2024-07-25", "2024-08-15"),
}


# =============================================================================
# Fuzz perturbations (batch 5.1)
# =============================================================================


@dataclass
class FuzzResult:
    """Outcome of one fuzz perturbation."""

    name: str
    crashed: bool
    surfaced_anomaly: bool
    notes: str = ""


def inject_nans(df: pd.DataFrame, fraction: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """Inject NaN values randomly in OHLCV columns."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    n_inject = int(n * fraction)
    idx = rng.choice(n, size=n_inject, replace=False)
    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out.iloc[idx, out.columns.get_loc(col)] = np.nan
    return out


def inject_gaps(df: pd.DataFrame, n_gaps: int = 5, gap_size: int = 10, seed: int = 42) -> pd.DataFrame:
    """Remove N gap-size chunks of bars."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    keep = np.ones(len(out), dtype=bool)
    for _ in range(n_gaps):
        start = int(rng.integers(0, max(1, len(out) - gap_size)))
        keep[start:start + gap_size] = False
    return out.loc[keep]


def inject_spread_spikes(
    df: pd.DataFrame,
    multiplier: float = 5.0,
    n_spikes: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Multiply (High - Low) at random bars by ``multiplier``."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    idx = rng.choice(n, size=n_spikes, replace=False)
    for i in idx:
        h = float(out.iloc[i]["High"])
        l = float(out.iloc[i]["Low"])
        mid = (h + l) / 2.0
        half_range = (h - l) / 2.0 * multiplier
        out.iat[i, out.columns.get_loc("High")] = mid + half_range
        out.iat[i, out.columns.get_loc("Low")] = mid - half_range
    return out


# =============================================================================
# Sensitivity sweep (batch 5.3)
# =============================================================================


@dataclass
class SensitivityResult:
    """Outcome of a ±20 % sweep on a single parameter."""

    parameter: str
    nominal_value: float
    metric_name: str
    metric_nominal: float
    metric_low: float    # value at -20 %
    metric_high: float   # value at +20 %
    relative_change_pct: float

    @property
    def is_fragile(self) -> bool:
        return abs(self.relative_change_pct) > 10.0


# =============================================================================
# Dispatcher
# =============================================================================


def run_stress_suite(
    historical_runner: Callable[[str, str], dict],
    fuzz_runner: Callable[[pd.DataFrame], dict],
    sensitivity_runner: Callable[[str, float], dict],
) -> dict:
    """Compose the 3 stress families into one suite.

    Each ``*_runner`` is injected so this module stays decoupled from
    :class:`SignalReplay`. The Sprint 5 entrypoint wires them.
    """
    out = {"historical": {}, "fuzz": {}, "sensitivity": {}}
    for name, (start, end) in HISTORICAL_STRESS_WINDOWS.items():
        out["historical"][name] = historical_runner(start, end)
    # Fuzz / sensitivity are populated by their own runners
    return out


__all__ = [
    "HISTORICAL_STRESS_WINDOWS",
    "FuzzResult",
    "SensitivityResult",
    "inject_nans",
    "inject_gaps",
    "inject_spread_spikes",
    "run_stress_suite",
]
