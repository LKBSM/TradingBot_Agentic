"""CLI entry point for historical state-machine replay.

Usage
-----
::

    # Default: XAU_15MIN_2019_2025 with default state-machine config
    python scripts/replay_state_machine.py

    # Custom CSV + config overrides
    python scripts/replay_state_machine.py \\
        --csv data/XAU_15MIN_2019_2025.csv \\
        --symbol XAUUSD --timeframe M15 \\
        --enter 75 --exit 55 --confirm 2 --cooldown 2 --max-age 12 \\
        --out replay_report.json --trades-csv replay_trades.csv

    # Limit to last N bars for quick iteration
    python scripts/replay_state_machine.py --last-n 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.backtest.state_machine_replay import SignalReplay
from src.environment.strategy_features import SmartMoneyEngine
from src.intelligence.signal_state_machine import StateMachineConfig


def _load_csv(csv_path: Path) -> pd.DataFrame:
    """Load OHLCV CSV, normalize columns, set DatetimeIndex."""
    df = pd.read_csv(csv_path)
    # Normalize timestamp column
    for candidate in ("timestamp", "Date", "date", "datetime", "time"):
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate])
            df = df.set_index(candidate)
            break
    # Normalize OHLCV column capitalization (SmartMoneyEngine lower-cases internally)
    rename: dict[str, str] = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open", "high", "low", "close", "volume"):
            rename[col] = lc.capitalize()
    if rename:
        df = df.rename(columns=rename)
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {sorted(missing)}: {csv_path}")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("replay")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        return 2
    log.info("Loading %s", csv_path)
    df = _load_csv(csv_path)
    log.info("Loaded %d bars (%s → %s)", len(df), df.index[0], df.index[-1])

    if args.last_n > 0 and len(df) > args.last_n:
        df = df.tail(args.last_n).copy()
        log.info("Truncated to last %d bars", len(df))

    # 1. Enrich with SMC features (vectorised, one-shot)
    log.info("Running SmartMoneyEngine (vectorised)…")
    engine = SmartMoneyEngine(data=df, config={}, verbose=False)
    enriched = engine.analyze()
    log.info("Enrichment complete: %d bars × %d cols", len(enriched), len(enriched.columns))

    # 2. Configure state machine
    cfg = StateMachineConfig(
        symbol=args.symbol,
        enter_threshold=args.enter,
        exit_threshold=args.exit,
        confirm_bars=args.confirm,
        cooldown_bars=args.cooldown,
        max_signal_age_bars=args.max_age,
    )

    # 3. Run replay
    log.info("Running replay (enter=%.0f, exit=%.0f, confirm=%d, cooldown=%d, max_age=%d)…",
             cfg.enter_threshold, cfg.exit_threshold,
             cfg.confirm_bars, cfg.cooldown_bars, cfg.max_signal_age_bars)
    replay = SignalReplay(
        symbol=args.symbol,
        timeframe=args.timeframe,
        state_machine_config=cfg,
        use_regime=not args.no_regime,
        use_vol_regime=not args.no_vol_regime,
        warmup_bars=args.warmup,
        detector_min_score=args.detector_min_score,
    )
    log.info(
        "ConfluenceDetector min_score=%.1f (state machine enter=%.1f exit=%.1f)",
        replay.confluence.min_score, cfg.enter_threshold, cfg.exit_threshold,
    )
    results = replay.run(enriched)

    # 4. Report
    print()
    print(results.pretty())
    print()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(results.to_dict(include_trades=args.emit_trades), indent=2),
            encoding="utf-8",
        )
        log.info("JSON report written to %s", out_path)

    if args.trades_csv and results.trades:
        trades_path = Path(args.trades_csv)
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([t.to_dict() for t in results.trades]).to_csv(
            trades_path, index=False,
        )
        log.info("Trade ledger written to %s", trades_path)

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay state machine on historical OHLCV")
    p.add_argument("--csv", default="data/XAU_15MIN_2019_2025.csv",
                   help="Path to OHLCV CSV")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--timeframe", default="M15")
    p.add_argument("--last-n", type=int, default=0,
                   help="Use only the last N bars (0 = all)")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--enter", type=float, default=75.0)
    p.add_argument("--exit", type=float, default=55.0)
    p.add_argument("--confirm", type=int, default=2)
    p.add_argument("--cooldown", type=int, default=2)
    p.add_argument("--max-age", type=int, default=12)
    p.add_argument("--no-regime", action="store_true",
                   help="Disable regime classifier (test minimal pipeline)")
    p.add_argument("--no-vol-regime", action="store_true",
                   help="Disable vol-regime classifier")
    p.add_argument("--detector-min-score", type=float, default=None,
                   help="Floor applied inside ConfluenceDetector "
                        "(default: min(enter, exit) so state machine is the gate)")
    p.add_argument("--out", default="replay_report.json",
                   help="JSON output path (pass empty to skip)")
    p.add_argument("--emit-trades", action="store_true", default=True,
                   help="Include per-trade records in JSON (default on)")
    p.add_argument("--trades-csv", default="replay_trades.csv",
                   help="Per-trade CSV output path (pass empty to skip)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(run(args))
