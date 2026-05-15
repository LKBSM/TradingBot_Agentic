"""Sprint 2 institutional backtest harness — CLI entry point.

Wraps :mod:`src.backtest.state_machine_replay` with the institutional
metrics from :mod:`src.backtest.metrics` and the report renderer from
:mod:`src.backtest.report`. Produces a text summary on stdout, a JSON
report on disk, and an optional per-trade CSV ledger.

Usage
-----
::

    # Full 6-year Gold M15 baseline
    python scripts/run_backtest.py \\
        --csv data/XAU_15MIN_2019_2026.csv \\
        --symbol XAUUSD --timeframe M15 \\
        --out reports/baseline_2019_2024.json

    # Quick iteration on last 20k bars
    python scripts/run_backtest.py --last-n 20000

    # Custom thresholds (calibrated to a specific regime)
    python scripts/run_backtest.py --enter 65 --exit 45

This differs from ``replay_state_machine.py`` in that it also emits the
institutional metric set (Sortino, Calmar, annualised Sharpe, per-tier
breakdown) — the numbers a prospect will ask for.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.metrics import compute_performance, tier_from_score
from src.backtest.report import render_json, render_text
from src.backtest.state_machine_replay import SignalReplay, TradeRecord
from src.environment.strategy_features import SmartMoneyEngine
from src.intelligence.signal_state_machine import StateMachineConfig


def _load_csv(csv_path: Path) -> pd.DataFrame:
    """Load OHLCV CSV, normalise columns, set DatetimeIndex."""
    df = pd.read_csv(csv_path)
    for candidate in ("timestamp", "Date", "date", "datetime", "time"):
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate])
            df = df.set_index(candidate)
            break
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


def _tier_for_trade(trade: TradeRecord) -> str:
    """Map TradeRecord -> tier label using its confluence score."""
    return tier_from_score(trade.confluence_score)


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("backtest")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        return 2
    log.info("Loading %s", csv_path)
    df = _load_csv(csv_path)
    log.info("Loaded %d bars (%s -> %s)", len(df), df.index[0], df.index[-1])

    if args.last_n > 0 and len(df) > args.last_n:
        df = df.tail(args.last_n).copy()
        log.info("Truncated to last %d bars", len(df))

    # 1. Enrich with SMC features (one-shot vectorised pass)
    log.info("Running SmartMoneyEngine (vectorised)...")
    engine = SmartMoneyEngine(data=df, config={}, verbose=False)
    enriched = engine.analyze()
    log.info(
        "Enrichment complete: %d bars x %d cols",
        len(enriched), len(enriched.columns),
    )

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
    log.info(
        "Running replay (enter=%.0f exit=%.0f confirm=%d cooldown=%d max_age=%d)...",
        cfg.enter_threshold, cfg.exit_threshold,
        cfg.confirm_bars, cfg.cooldown_bars, cfg.max_signal_age_bars,
    )
    # Build detector explicitly so we can pass the retest toggle.
    # IMPORTANT: pass instrument_config so SL/TP multipliers and price_decimals
    # match the symbol. Without this, the XAU defaults (SL=2×ATR, TP=4×ATR)
    # were applied to FX pairs, where ATR units are ~1000× smaller — produced
    # PF ≈ 0 on EURUSD baseline (see reports/eurusd_subset_audit.md).
    from src.intelligence.confluence_detector import ConfluenceDetector
    from src.intelligence.volatility_forecaster import get_instrument_registry
    registry = get_instrument_registry()
    instrument_config = registry.get(args.symbol)
    if instrument_config is None:
        log.warning(
            "No InstrumentConfig for %s; falling back to XAU defaults (SL=2×, TP=4×).",
            args.symbol,
        )
    detector_min = (
        float(args.detector_min_score) if args.detector_min_score is not None
        else min(cfg.exit_threshold, cfg.enter_threshold)
    )
    detector = ConfluenceDetector(
        symbol=args.symbol,
        min_score=detector_min,
        instrument_config=instrument_config,
        require_retest=not args.no_retest,
    )
    log.info(
        "Retest gate: %s",
        "DISABLED (baseline)" if args.no_retest else "ENABLED (require pullback)",
    )
    # Sprint 3 batch 3.X (audit P0-6): wire DynamicSpread/Slippage so backtest
    # is no longer cost-free. Audit `section_3_8_backtest_engine.md` flagged
    # commission=$0 + no slippage as P0 — strat-tear-sheet PF will drop but
    # this is the honest number. --no-costs flag preserves the legacy cost-free
    # behaviour for compat with older reports.
    spread_model = None
    slippage_model = None
    if not args.no_costs:
        from src.environment.execution_model import DynamicSlippageModel, DynamicSpreadModel
        spread_model = DynamicSpreadModel()
        slippage_model = DynamicSlippageModel(base_slippage=0.0001)
        log.info("Costs ENABLED: dynamic spread + slippage models active")
    else:
        log.info("Costs DISABLED (--no-costs flag): legacy cost-free backtest")

    replay = SignalReplay(
        symbol=args.symbol,
        timeframe=args.timeframe,
        state_machine_config=cfg,
        confluence_detector=detector,
        use_regime=not args.no_regime,
        use_vol_regime=not args.no_vol_regime,
        warmup_bars=args.warmup,
        spread_model=spread_model,
        slippage_model=slippage_model,
    )
    log.info(
        "ConfluenceDetector min_score=%.1f  (state machine enter=%.1f exit=%.1f)",
        replay.confluence.min_score, cfg.enter_threshold, cfg.exit_threshold,
    )
    results = replay.run(enriched)

    # 4. Institutional metrics
    log.info("Computing institutional metrics...")
    metrics = compute_performance(
        results.trades,
        timeframe=args.timeframe,
        tier_fn=_tier_for_trade,
        bars_processed=results.bars_processed,
    )

    # 5. Render
    print()
    print(render_text(results, metrics))
    print()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = render_json(results, metrics, include_trades=args.emit_trades)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
    p = argparse.ArgumentParser(
        description="Institutional backtest harness for Smart Sentinel AI",
    )
    p.add_argument("--csv", default="data/XAU_15MIN_2019_2026.csv",
                   help="Path to OHLCV CSV (default: Gold M15 2019-2026, 98.72% coverage)")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--timeframe", default="M15",
                   help="Bar cadence (used to annualise Sharpe/Sortino)")
    p.add_argument("--last-n", type=int, default=0,
                   help="Use only the last N bars (0 = all)")
    p.add_argument("--warmup", type=int, default=100)

    # State-machine config
    p.add_argument("--enter", type=float, default=75.0)
    p.add_argument("--exit", type=float, default=55.0)
    p.add_argument("--confirm", type=int, default=2)
    p.add_argument("--cooldown", type=int, default=2)
    p.add_argument("--max-age", type=int, default=12)

    # Regime toggles
    p.add_argument("--no-regime", action="store_true",
                   help="Disable regime classifier")
    p.add_argument("--no-vol-regime", action="store_true",
                   help="Disable vol-regime classifier")

    # Detector override
    p.add_argument("--detector-min-score", type=float, default=None,
                   help="Floor applied inside ConfluenceDetector "
                        "(default: min(enter, exit) so state machine is the sole gate)")
    p.add_argument("--no-retest", action="store_true",
                   help="Disable BOS pullback/retest gate in ConfluenceDetector "
                        "(pre-retest baseline for comparison).")
    p.add_argument("--no-costs", action="store_true",
                   help="Disable dynamic spread + slippage models (legacy "
                        "cost-free backtest, default since Sprint 3 = costs ON).")

    # Output
    p.add_argument("--out", default="backtest_report.json",
                   help="JSON output path (pass empty to skip)")
    p.add_argument("--emit-trades", action="store_true", default=True,
                   help="Include per-trade records in JSON (default on)")
    p.add_argument("--trades-csv", default="backtest_trades.csv",
                   help="Per-trade CSV output path (pass empty to skip)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(run(args))
