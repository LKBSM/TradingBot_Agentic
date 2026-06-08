"""End-to-end evaluation of the event-driven macro strategy.

Loads the real XAU M15 history (2019-2026) and the ForexFactory HIGH-impact
calendar (2019-2025), runs the strategy, then evaluates it through the
mandatory CPCV + DSR + PBO + DM gates from ``src/research/strategy_gates``.

Outputs a structured JSON report under ``reports/`` and prints a one-line
verdict to stdout. Exit code 0 if all gates pass, 1 otherwise — usable in CI.

Usage
-----
    python scripts/eval_event_driven_macro.py

Optional environment variables:
    OHLCV_PATH       — override the OHLCV CSV path
    CALENDAR_PATH    — override the calendar CSV path
    REPORT_PATH      — override the JSON output destination
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import pandas as pd

from src.intelligence.conformal_wrapper import apply_conformal_filter
from src.intelligence.regime_gate import RegimeGate, RegimeDecision
from src.research.strategy_gates import evaluate_gates
from src.strategies.event_driven_macro import (
    EventStrategyConfig,
    _parse_calendar,
    _parse_ohlcv,
    compute_atr,
    EventDrivenMacroStrategy,
    run_event_strategy_from_csv,
)


LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_event_driven_macro")


def main() -> int:
    ohlcv_path = Path(os.environ.get("OHLCV_PATH", "data/XAU_15MIN_2019_2026.csv"))
    calendar_path = Path(
        os.environ.get("CALENDAR_PATH", "data/economic_calendar_HIGH_IMPACT_2019_2025.csv")
    )
    report_path = Path(
        os.environ.get(
            "REPORT_PATH",
            f"reports/event_driven_macro_eval_{datetime.now().strftime('%Y_%m_%d')}.json",
        )
    )

    if not ohlcv_path.exists():
        logger.error(f"OHLCV file missing: {ohlcv_path}")
        return 2
    if not calendar_path.exists():
        logger.error(f"Calendar file missing: {calendar_path}")
        return 2

    logger.info(f"Loading {ohlcv_path} and {calendar_path}")
    cfg = EventStrategyConfig()
    trades, r_mults = run_event_strategy_from_csv(
        ohlcv_path=ohlcv_path,
        calendar_path=calendar_path,
        cfg=cfg,
    )
    n_trades = len(trades)
    logger.info(f"Strategy emitted {n_trades} trades")

    if n_trades == 0:
        logger.error("Zero trades — cannot evaluate gates")
        return 3

    # Quick summary stats
    win_rate = float((r_mults > 0).mean()) if n_trades else 0.0
    avg_r = float(r_mults.mean()) if n_trades else 0.0
    pnl_cum = float(r_mults.sum())
    logger.info(
        f"Trades={n_trades}, win_rate={win_rate:.2%}, avg_R={avg_r:.3f}, "
        f"sum_R={pnl_cum:.1f}"
    )

    # =========================================================================
    # 3-VARIANT EVALUATION : naive baseline, +conformal (Pilier 2),
    # +regime_gate (Pilier 3), +both
    # =========================================================================

    # Variant A — naive baseline
    gate_result = evaluate_gates(
        r_mults, n_trials=1, min_trades=30, n_bootstraps=1000
    )

    # Variant B — conformal reject-option (Pilier 2)
    # Use the first 50 trades as initial calibration, then filter the rest
    calib_size = min(50, n_trades // 3)
    if calib_size >= 30 and n_trades > calib_size + 30:
        cal = r_mults[:calib_size]
        cand = r_mults[calib_size:]
        filtered_b = apply_conformal_filter(
            cand, cal, alpha=0.10, breakeven=0.0, adaptive=True
        )
        gate_b = evaluate_gates(
            filtered_b, n_trials=1, min_trades=10, n_bootstraps=1000
        )
    else:
        gate_b = None
        filtered_b = np.array([])

    # Variant C — regime gate (Pilier 3) : block trades where, at the entry
    # bar, the regime gate emits BLOCK or REDUCE.
    logger.info("Computing regime gate over OHLCV history for Pilier 3 eval")
    ohlcv_df = _parse_ohlcv(ohlcv_path)
    log_returns = np.log(
        ohlcv_df["close"].to_numpy() / ohlcv_df["close"].shift(1).to_numpy()
    )
    log_returns = np.where(np.isfinite(log_returns), log_returns, 0.0)

    gate = RegimeGate(bipower_window=96)
    # Run BOCPD over the whole series — track block/reduce decisions per bar
    bar_decisions = np.zeros(len(ohlcv_df), dtype=np.int8)
    for i in range(len(ohlcv_df)):
        recent = log_returns[max(0, i - 96 + 1) : i + 1]
        out = gate.update(log_return=float(log_returns[i]), recent_returns=recent)
        if out.decision is RegimeDecision.BLOCK:
            bar_decisions[i] = 2
        elif out.decision is RegimeDecision.REDUCE:
            bar_decisions[i] = 1

    # Map each trade to the decision at its entry bar
    timestamps = ohlcv_df["timestamp"].to_numpy()
    trade_decisions = []
    for t in trades:
        idx_match = np.searchsorted(timestamps, np.datetime64(t.entry_time))
        idx_match = min(idx_match, len(bar_decisions) - 1)
        trade_decisions.append(int(bar_decisions[idx_match]))
    trade_decisions_arr = np.asarray(trade_decisions)
    kept_mask_c = trade_decisions_arr != 2  # drop only BLOCK
    filtered_c = r_mults[kept_mask_c]
    if len(filtered_c) >= 30:
        gate_c = evaluate_gates(filtered_c, n_trials=1, n_bootstraps=1000)
    else:
        gate_c = None

    # Variant D — combined : regime gate + conformal
    if len(filtered_c) >= calib_size + 30 and calib_size >= 30:
        cal_d = filtered_c[:calib_size]
        cand_d = filtered_c[calib_size:]
        filtered_d = apply_conformal_filter(
            cand_d, cal_d, alpha=0.10, breakeven=0.0, adaptive=True
        )
        if len(filtered_d) >= 10:
            gate_d = evaluate_gates(
                filtered_d, n_trials=1, min_trades=10, n_bootstraps=1000
            )
        else:
            gate_d = None
    else:
        gate_d = None

    logger.info("=== 3-PILLAR COMPARATIVE GATES ===")
    for name, gr, n in [
        ("A_naive", gate_result, n_trades),
        ("B_+conformal", gate_b, len(filtered_b)),
        ("C_+regime", gate_c, int(kept_mask_c.sum())),
        ("D_+both", gate_d, len(filtered_d) if gate_d else 0),
    ]:
        if gr is None:
            logger.info(f"  {name:14s}: SKIPPED (insufficient sample)")
            continue
        logger.info(
            f"  {name:14s}: n={n:4d} PF={gr.profit_factor:.3f} "
            f"PF_lo={gr.profit_factor_lo:.3f} DSR={gr.dsr:.2f} "
            f"PBO={gr.pbo:.2f} DM_p={gr.dm_pvalue:.3f} -> "
            f"{'PASS' if gr.all_passed else 'FAIL'}"
        )

    # Compose the report
    by_event: dict = {}
    for t in trades:
        key = t.event_name
        by_event.setdefault(
            key, {"n": 0, "wins": 0, "sum_r": 0.0, "exit_reasons": {}}
        )
        rec = by_event[key]
        rec["n"] += 1
        rec["wins"] += int(t.r_multiple > 0)
        rec["sum_r"] += t.r_multiple
        rec["exit_reasons"][t.exit_reason] = rec["exit_reasons"].get(t.exit_reason, 0) + 1

    report = {
        "generated_at": datetime.now().isoformat(),
        "ohlcv_path": str(ohlcv_path),
        "calendar_path": str(calendar_path),
        "config": {
            "trigger_window_min": cfg.trigger_window_min,
            "trigger_threshold_atr": cfg.trigger_threshold_atr,
            "sl_atr": cfg.sl_atr,
            "tp_atr": cfg.tp_atr,
            "max_hold_bars": cfg.max_hold_bars,
            "currency_filter": cfg.currency_filter,
            "impact_filter": cfg.impact_filter,
        },
        "summary": {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_r_multiple": avg_r,
            "sum_r_multiple": pnl_cum,
            "best_trade_r": float(r_mults.max()),
            "worst_trade_r": float(r_mults.min()),
        },
        "by_event": by_event,
        "gates": gate_result.to_dict(),
        "variants": {
            "A_naive": gate_result.to_dict(),
            "B_conformal": gate_b.to_dict() if gate_b else None,
            "C_regime_gate": gate_c.to_dict() if gate_c else None,
            "D_both": gate_d.to_dict() if gate_d else None,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Report written to {report_path}")

    verdict = "PASS" if gate_result.all_passed else "FAIL"
    logger.info(
        f"VERDICT: {verdict} | "
        f"DSR={gate_result.dsr:.3f} | "
        f"PBO={gate_result.pbo:.3f} | "
        f"PF={gate_result.profit_factor:.3f} | "
        f"PF_lo={gate_result.profit_factor_lo:.3f} | "
        f"DM_p={gate_result.dm_pvalue:.4f}"
    )
    if not gate_result.all_passed:
        logger.warning("Failure reasons:")
        for reason in gate_result.failure_reasons:
            logger.warning(f"  - {reason}")

    return 0 if gate_result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
