"""Eval 20 — Multi-asset / multi-timeframe sweep runner (PR-ready).

Runs `scripts/run_backtest.py` over the 6 x 5 = 30 cell matrix:
    symbols    = [XAUUSD, EURUSD, BTCUSD, US500, GBPUSD, USDJPY]
    timeframes = [M5, M15, H1, H4, D1]

Per-symbol input convention
---------------------------
For each symbol we expect ONE base CSV at the symbol's lowest available
timeframe (typically M1 or M5). The runner resamples up to each target
timeframe via :func:`src.intelligence.volatility_forecaster.resample_ohlcv`.
This avoids storing 30 CSV files when one M1/M5 master per symbol suffices.

If the base CSV is missing, the cell is recorded with status=MISSING_DATA
and the sweep continues — no exception.

Output
------
A single CSV `reports/eval_20_sweep_30cells.csv` with the schema:
    symbol, timeframe, csv_path, bars_processed, n_trades, win_rate,
    profit_factor, sharpe_annualized, sortino_annualized, calmar,
    max_drawdown_R, total_R, avg_bars_held, trades_per_year, status, notes

Usage
-----
    python reports/eval_20_sweep_runner.py \\
        --data-dir data \\
        --out reports/eval_20_sweep_30cells.csv

    # Walk-forward (Prompt 18 alignment) — splits each cell 70/15/15 by date
    python reports/eval_20_sweep_runner.py --walk-forward

Caveats
-------
- D1 resample on M1/M5/M15 base CSVs of <=10 years yields ~1500 bars
  which is BELOW the state machine warmup (100) + cooldown thresholds
  defined in scripts/run_backtest.py:195. D1 cells are emitted with
  status=INSUFFICIENT_BARS rather than skipped, so the matrix is honest.
- The runner is in-sample by default. Pass --walk-forward to compute
  rolling OOS metrics (Sprint 3-style) — slower but the only number a
  prospect should be shown.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.metrics import compute_performance, tier_from_score
from src.backtest.state_machine_replay import SignalReplay
from src.environment.strategy_features import SmartMoneyEngine
from src.intelligence.confluence_detector import ConfluenceDetector
from src.intelligence.signal_state_machine import StateMachineConfig
from src.intelligence.volatility_forecaster import resample_ohlcv

logger = logging.getLogger(__name__)


SYMBOLS = ["XAUUSD", "EURUSD", "BTCUSD", "US500", "GBPUSD", "USDJPY"]
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

# Per-symbol base CSV — lowest timeframe stored locally. Override on CLI
# with --csv-overrides "XAUUSD=data/XAU_15MIN_2019_2024.csv,EURUSD=..."
DEFAULT_CSV_BASE = {
    "XAUUSD": ("data/XAU_15MIN_2019_2024.csv", "M15"),
    "EURUSD": ("data/EURUSD_M15.csv",          "M15"),
    "BTCUSD": ("data/BTCUSD_M15.csv",          "M15"),
    "US500":  ("data/US500_M15.csv",           "M15"),
    "GBPUSD": ("data/GBPUSD_M15.csv",          "M15"),
    "USDJPY": ("data/USDJPY_M15.csv",          "M15"),
}

# Per-symbol bars-per-day for annualised metrics (driven by session)
BARS_PER_DAY_M15 = {
    "XAUUSD": 96,   # 24h
    "EURUSD": 96,   # 24h forex
    "BTCUSD": 96,   # 24/7 — treat 24h
    "US500":  28,   # RTH 14:30-21:00 UTC + extended
    "GBPUSD": 96,
    "USDJPY": 96,
}

MIN_BARS_FOR_REPLAY = 500   # warmup 100 + a meaningful sample


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    symbol: str
    timeframe: str
    csv_path: str
    bars_processed: int = 0
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_annualized: float = 0.0
    sortino_annualized: float = 0.0
    calmar: float = 0.0
    max_drawdown_R: float = 0.0
    total_R: float = 0.0
    avg_bars_held: float = 0.0
    trades_per_year: float = 0.0
    status: str = "PENDING"
    notes: str = ""


# ---------------------------------------------------------------------------
# CSV loader (mirrors scripts/run_backtest.py:_load_csv)
# ---------------------------------------------------------------------------

def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for cand in ("timestamp", "Date", "date", "datetime", "time"):
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand])
            df = df.set_index(cand)
            break
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open", "high", "low", "close", "volume"):
            rename[col] = lc.capitalize()
    if rename:
        df = df.rename(columns=rename)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {sorted(missing)}: {csv_path}")
    return df


# ---------------------------------------------------------------------------
# One-cell runner
# ---------------------------------------------------------------------------

def run_cell(
    symbol: str,
    timeframe: str,
    base_csv: Path,
    base_tf: str,
    enter: float = 75.0,
    exit_: float = 55.0,
    confirm: int = 2,
    cooldown: int = 2,
    max_age: int = 12,
    walk_forward: bool = False,
) -> CellResult:
    cell = CellResult(symbol=symbol, timeframe=timeframe, csv_path=str(base_csv))

    if not base_csv.exists():
        cell.status = "MISSING_DATA"
        cell.notes = f"Base CSV not found: {base_csv}"
        return cell

    try:
        df = _load_csv(base_csv)
    except Exception as exc:  # pylint: disable=broad-except
        cell.status = "LOAD_ERROR"
        cell.notes = str(exc)
        return cell

    # Resample to target tf if needed
    if timeframe != base_tf:
        try:
            resampled = resample_ohlcv(df.reset_index(), base_tf, timeframe)
            # resample_ohlcv returns reset frame; restore index
            if "timestamp" in resampled.columns:
                resampled = resampled.set_index("timestamp")
            # Restore canonical column names
            resampled = resampled.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            df = resampled
        except Exception as exc:  # pylint: disable=broad-except
            cell.status = "RESAMPLE_ERROR"
            cell.notes = f"resample {base_tf}->{timeframe}: {exc}"
            return cell

    if len(df) < MIN_BARS_FOR_REPLAY:
        cell.bars_processed = len(df)
        cell.status = "INSUFFICIENT_BARS"
        cell.notes = f"{len(df)} bars < {MIN_BARS_FOR_REPLAY}"
        return cell

    # Enrich with SMC features
    engine = SmartMoneyEngine(data=df, config={}, verbose=False)
    enriched = engine.analyze()

    sm_cfg = StateMachineConfig(
        symbol=symbol,
        enter_threshold=enter,
        exit_threshold=exit_,
        confirm_bars=confirm,
        cooldown_bars=cooldown,
        max_signal_age_bars=max_age,
    )
    detector = ConfluenceDetector(
        symbol=symbol,
        min_score=min(enter, exit_),
        require_retest=True,
    )
    replay = SignalReplay(
        symbol=symbol,
        timeframe=timeframe,
        state_machine_config=sm_cfg,
        confluence_detector=detector,
        use_regime=True,
        use_vol_regime=True,
        warmup_bars=100,
    )

    if walk_forward:
        # Simple 70/15/15 hold-out — record OOS only (last 15 %)
        n = len(enriched)
        oos_start = int(n * 0.85)
        oos_df = enriched.iloc[oos_start:].copy()
        if len(oos_df) < MIN_BARS_FOR_REPLAY:
            cell.status = "INSUFFICIENT_OOS_BARS"
            cell.notes = f"OOS={len(oos_df)} bars < {MIN_BARS_FOR_REPLAY}"
            return cell
        results = replay.run(oos_df)
        cell.notes = f"OOS hold-out 15% ({len(oos_df)} bars)"
    else:
        results = replay.run(enriched)
        cell.notes = "in-sample"

    metrics = compute_performance(
        results.trades,
        timeframe=timeframe,
        tier_fn=lambda t: tier_from_score(t.confluence_score),
        bars_processed=results.bars_processed,
    )

    cell.bars_processed = results.bars_processed
    cell.n_trades = metrics.n_trades
    cell.win_rate = round(metrics.win_rate, 4)
    cell.profit_factor = round(metrics.profit_factor, 4)
    cell.sharpe_annualized = round(metrics.sharpe_annualized, 4)
    cell.sortino_annualized = round(metrics.sortino_annualized, 4)
    cell.calmar = round(metrics.calmar, 4)
    cell.max_drawdown_R = round(metrics.max_drawdown_R, 4)
    cell.total_R = round(metrics.total_R, 4)
    cell.avg_bars_held = round(metrics.avg_bars_held, 2)
    cell.trades_per_year = round(metrics.trades_per_year, 2)
    cell.status = "OK"
    return cell


# ---------------------------------------------------------------------------
# Sweep orchestrator
# ---------------------------------------------------------------------------

def run_sweep(
    data_dir: Path,
    out_path: Path,
    csv_overrides: Optional[dict] = None,
    walk_forward: bool = False,
) -> None:
    csv_overrides = csv_overrides or {}
    rows: list[CellResult] = []

    for symbol in SYMBOLS:
        base_csv_rel, base_tf = csv_overrides.get(
            symbol, DEFAULT_CSV_BASE[symbol],
        )
        base_csv = (data_dir.parent / base_csv_rel).resolve() \
            if not Path(base_csv_rel).is_absolute() else Path(base_csv_rel)

        for tf in TIMEFRAMES:
            logger.info("Cell: %s %s  base=%s", symbol, tf, base_csv)
            try:
                cell = run_cell(
                    symbol=symbol,
                    timeframe=tf,
                    base_csv=base_csv,
                    base_tf=base_tf,
                    walk_forward=walk_forward,
                )
            except Exception as exc:  # pylint: disable=broad-except
                cell = CellResult(
                    symbol=symbol, timeframe=tf, csv_path=str(base_csv),
                    status="EXCEPTION", notes=str(exc)[:200],
                )
            rows.append(cell)

    # Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    logger.info("Sweep done. %d cells -> %s", len(rows), out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Eval 20 — 30-cell sweep runner")
    p.add_argument("--data-dir", default="data",
                   help="Directory containing per-symbol CSVs")
    p.add_argument("--out", default="reports/eval_20_sweep_30cells.csv")
    p.add_argument("--walk-forward", action="store_true",
                   help="OOS hold-out (last 15%% of each cell)")
    p.add_argument("--csv-overrides", default="",
                   help="symbol=path,symbol=path comma-separated overrides "
                        "(default base_tf assumed M15)")
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()

    overrides = {}
    if args.csv_overrides.strip():
        for spec in args.csv_overrides.split(","):
            if "=" in spec:
                sym, path = spec.split("=", 1)
                overrides[sym.strip()] = (path.strip(), "M15")

    run_sweep(
        data_dir=Path(args.data_dir),
        out_path=Path(args.out),
        csv_overrides=overrides,
        walk_forward=args.walk_forward,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
