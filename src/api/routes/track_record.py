"""DG-142 — public ``/track-record`` endpoint.

Returns a compact JSON document with the headline track-record figures:
profit factor + 95% bootstrap CI, hit-rate, total trades, equity curve
points, and the backtest window label.

Honesty rules (UE 2024/2811 + AUDIT_ALGO_2026_05_27):
- ``edge_claim`` stays ``False`` until empirical gates (PF > 1.20, DSR > 1.0,
  PBO < 0.5, walk-forward ≥ 2y) are crossed. The endpoint surfaces ``edge_claim``
  to make downstream consumers (the webapp page, the chatbot) reuse the same
  flag rather than rolling their own.
- Numbers are taken from a static snapshot (``reports/calibration/trades_xau_2019_2026.csv``)
  so a flaky live store can't break the page. Updated by a nightly cron that
  also pushes a copy to R2 (filed in Sprint 6).
- When the CSV is absent (e.g. dev container), returns a placeholder with
  ``"OOS validation pending"`` so the consumer renders the disclaimer-only
  state rather than crashing.

The endpoint is intentionally PUBLIC (no auth) so the marketing page can
embed it. PII-free by construction — trade rows are aggregated to summary
statistics only.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter

logger = logging.getLogger(__name__)


# Bootstrap settings — 1000 iterations is the brief minimum and matches
# the gate harness in ``src/research/strategy_gates.py``.
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_ALPHA = 0.05  # 95 % CI

# Equity-curve down-sampling cap. We don't need to send 30k points; the
# page renders fine at ~250 points.
EQUITY_CURVE_MAX_POINTS = 250

# Default CSV path; override via env var for offline tests.
DEFAULT_TRADES_CSV = "reports/calibration/trades_xau_2019_2026.csv"


router = APIRouter(prefix="/api/v1/track-record", tags=["track-record"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profit_factor(pnls: List[float]) -> float:
    """PF = sum(positive pnls) / |sum(negative pnls)|. Returns inf when no
    losses, 0 when no gains, 1.0 on empty."""
    if not pnls:
        return 1.0
    gains = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses <= 0:
        return float("inf") if gains > 0 else 1.0
    return gains / losses


def _bootstrap_pf_ci(
    pnls: List[float],
    *,
    n_bootstraps: int = DEFAULT_BOOTSTRAP_N,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """Return (lower_ci, upper_ci) at confidence 1 - alpha.

    Pure-Python implementation so the endpoint works without numpy in
    minimal containers. For larger samples, switch to
    ``src.research.strategy_gates.profit_factor_bootstrap_ci`` which uses
    numpy random.choice.
    """
    if len(pnls) < 5:
        return (0.0, float("inf"))
    import random as _random

    rng = _random.Random(seed)
    n = len(pnls)
    pfs: List[float] = []
    for _ in range(n_bootstraps):
        sample = [pnls[rng.randrange(n)] for _ in range(n)]
        pf = _profit_factor(sample)
        if math.isfinite(pf):
            pfs.append(pf)
    if not pfs:
        return (0.0, float("inf"))
    pfs.sort()
    lo_idx = int(len(pfs) * (alpha / 2))
    hi_idx = int(len(pfs) * (1 - alpha / 2))
    return (pfs[lo_idx], pfs[min(hi_idx, len(pfs) - 1)])


def _hit_rate(pnls: List[float]) -> float:
    if not pnls:
        return 0.0
    return sum(1 for p in pnls if p > 0) / len(pnls)


def _equity_curve(pnls: List[float], *, max_points: int = EQUITY_CURVE_MAX_POINTS) -> List[float]:
    """Cumulative equity in R-multiples, down-sampled to ``max_points``."""
    if not pnls:
        return [0.0]
    cum = 0.0
    points: List[float] = []
    for p in pnls:
        cum += p
        points.append(round(cum, 4))
    if len(points) <= max_points:
        return points
    # Strided down-sampling — preserves first + last + uniform spacing
    step = len(points) / max_points
    out: List[float] = []
    for i in range(max_points):
        idx = int(i * step)
        out.append(points[idx])
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@dataclass
class _TradesSnapshot:
    pnls: List[float]
    n_trades: int
    backtest_window: str
    source_path: Optional[str]


def _load_trades(csv_path: Path) -> Optional[_TradesSnapshot]:
    """Load PnL columns from the backtest CSV. Returns None if not present."""
    if not csv_path.exists():
        logger.warning("track-record CSV not found at %s", csv_path)
        return None
    pnls: List[float] = []
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use ``r_multiple`` (return in R units) when available; falls
            # back to ``pnl_price``. Both are present in our backtest CSV.
            r = row.get("r_multiple") or row.get("pnl_price")
            if r is None:
                continue
            try:
                pnls.append(float(r))
            except ValueError:
                continue
            ts = row.get("entry_bar")
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
    if not pnls:
        return None
    window = (
        f"{first_ts[:10]} → {last_ts[:10]}"
        if first_ts and last_ts
        else "unknown window"
    )
    return _TradesSnapshot(
        pnls=pnls,
        n_trades=len(pnls),
        backtest_window=window,
        source_path=str(csv_path),
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def _flag_is_on() -> bool:
    """Feature flag — public track-record exposure.

    Default off (post-pivot 2026-05-27 + audit_algo verdict 3/10). The
    raw PF / hit-rate / equity-curve numbers MUST NOT reach the public
    page until the empirical gates (PF > 1.20, DSR > 1.0, PBO < 0.5,
    walk-forward ≥ 2y) have been crossed AND the figures have been
    revalidated by an independent reviewer.

    Operator flip: ``TRACK_RECORD_PUBLIC=1`` on the prod env once the
    gate review is signed off. Anything else (unset, "0", "off",
    "false") keeps the placeholder.
    """
    raw = os.environ.get("TRACK_RECORD_PUBLIC", "").strip().lower()
    return raw in {"1", "true", "on", "yes"}


@router.get("", summary="Public track-record snapshot")
def get_track_record() -> Dict[str, Any]:
    """Public PF + bootstrap CI + equity curve + honesty metadata.

    Cached at the CDN layer when deployed; the underlying file changes
    only on the nightly cron tick.

    GATE: when the feature flag ``TRACK_RECORD_PUBLIC`` is off (default),
    the endpoint returns the placeholder payload only — the raw figures
    are never reachable from the public network even if the CSV is on
    disk. Internal callers needing the live numbers should call the
    helpers directly (``_load_trades`` + ``_profit_factor`` + …).
    """
    if not _flag_is_on():
        return _placeholder_payload()

    csv_path = Path(os.environ.get("TRACK_RECORD_CSV", DEFAULT_TRADES_CSV))
    snap = _load_trades(csv_path)
    if snap is None:
        return _placeholder_payload()

    pnls = snap.pnls
    point_pf = _profit_factor(pnls)
    pf_lo, pf_hi = _bootstrap_pf_ci(pnls)
    hit = _hit_rate(pnls)
    eq = _equity_curve(pnls)
    edge_claim = bool(point_pf > 1.20 and pf_lo > 1.0)

    # JSON cannot encode infinity — convert to None so the page renders
    # "n/a" rather than crashing the parser.
    def _safe(x: float) -> Optional[float]:
        return round(x, 4) if math.isfinite(x) else None

    return {
        "n_trades": snap.n_trades,
        "profit_factor": _safe(point_pf),
        "profit_factor_ci95": [_safe(pf_lo), _safe(pf_hi)],
        "hit_rate": round(hit, 4),
        "equity_curve_r_multiples": eq,
        "backtest_window": snap.backtest_window,
        "edge_claim": edge_claim,
        "bootstrap": {
            "n_iterations": DEFAULT_BOOTSTRAP_N,
            "alpha": DEFAULT_ALPHA,
            "seed": DEFAULT_BOOTSTRAP_SEED,
        },
        "disclaimer": (
            "Performance passée non garantie de performance future. "
            "Backtest paper-trading, hors coûts d'opportunité personnels."
        ),
    }


def _placeholder_payload() -> Dict[str, Any]:
    return {
        "n_trades": 0,
        "profit_factor": None,
        "profit_factor_ci95": [0.0, None],
        "hit_rate": None,
        "equity_curve_r_multiples": [],
        "backtest_window": "OOS validation pending — Sprint 5",
        "edge_claim": False,
        "bootstrap": {
            "n_iterations": DEFAULT_BOOTSTRAP_N,
            "alpha": DEFAULT_ALPHA,
            "seed": DEFAULT_BOOTSTRAP_SEED,
        },
        "disclaimer": (
            "Aucune donnée publiée — la validation OOS indépendante est "
            "en cours."
        ),
    }


__all__ = ["router"]
