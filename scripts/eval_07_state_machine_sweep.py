"""Sprint 2 — Sweep empirique des paramètres SignalStateMachine.

Eval 07 P0: les defaults `enter=75, exit=55, confirm=2, cooldown=2,
max_age=12` ne sont pas dérivés du replay 6-ans. Ce script teste 280+
combinaisons sur XAU M15 2019-2024 et publie un CSV ordonné par PF.

Usage:
    python -m scripts.eval_07_state_machine_sweep
    python -m scripts.eval_07_state_machine_sweep --csv data/XAU_15MIN_2019_2024.csv

Output:
    reports/eval_07_sweep.csv  — toutes cellules (e, x, c, k, m, pf, sharpe, ...)
    reports/eval_07_sweep_top10.md — top 10 PF + comparaison vs default

Coût compute estimé : ~0.5-2 min par cellule × 280 = **2-8 h total**.
Lance en background. Lit `enriched_df` cached pour ne pas re-tourner SMC à
chaque cellule.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add project root to path so the script works from any CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.backtest.state_machine_replay import SignalReplay
from src.environment.strategy_features import SmartMoneyEngine
from src.intelligence.signal_state_machine import StateMachineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_07_sweep")


GRID = list(itertools.product(
    [65, 70, 75, 80, 85],     # enter_threshold
    [40, 45, 50, 55, 60],     # exit_threshold
    [1, 2, 3],                # confirm_bars
    [0, 2, 5],                # cooldown_bars
    [6, 12, 24],              # max_signal_age_bars
))


def load_and_enrich(csv_path: Path) -> pd.DataFrame:
    """Load OHLCV and run SMC enrichment once (~30s, cached for the sweep)."""
    log.info("Loading %s", csv_path)
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
        df["Volume"] = 0.0
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    log.info("Loaded %d bars (%s -> %s)", len(df), df.index.min(), df.index.max())

    log.info("Running SMC enrichment (Numba JIT — first call slow, subsequent fast)")
    engine = SmartMoneyEngine(
        data=df,
        config={
            "RSI_WINDOW": 14, "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
            "BB_WINDOW": 20, "ATR_WINDOW": 14,
            "FRACTAL_WINDOW": 2, "FVG_THRESHOLD": 0.1,
            "OB_REQUIRE_FVG": False,
        },
    )
    enriched = engine.analyze()
    log.info("Enriched %d bars (after NaN drop)", len(enriched))
    return enriched


def run_one(enriched: pd.DataFrame, e: int, x: int, c: int, k: int, m: int) -> Dict[str, Any]:
    """Run a single replay with the given config. Returns row dict."""
    cfg = StateMachineConfig(
        symbol="XAUUSD",
        enter_threshold=float(e),
        exit_threshold=float(x),
        confirm_bars=c,
        cooldown_bars=k,
        max_signal_age_bars=m,
    )
    replay = SignalReplay(
        symbol="XAUUSD", timeframe="M15",
        state_machine_config=cfg,
        use_regime=True,
        use_vol_regime=True,
    )
    res = replay.run(enriched)
    return {
        "enter": e, "exit": x, "confirm": c, "cooldown": k, "max_age": m,
        "n_trades": res.total_trades,
        "wins": res.wins,
        "losses": res.losses,
        "win_rate": round(res.win_rate, 4),
        "profit_factor": round(res.profit_factor, 4),
        "sharpe_per_trade": round(res.sharpe_per_trade, 4),
        "total_pnl_r": round(getattr(res, "total_pnl_r", 0.0), 4),
        "max_drawdown_r": round(getattr(res, "max_drawdown_r", 0.0), 4),
        "avg_lifetime_bars": round(getattr(res, "avg_lifetime_bars", 0.0), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=_PROJECT_ROOT / "data" / "XAU_15MIN_2019_2024.csv")
    ap.add_argument("--out", type=Path, default=_PROJECT_ROOT / "reports" / "eval_07_sweep.csv")
    ap.add_argument("--limit", type=int, default=0, help="Only run first N cells (for dry-run)")
    args = ap.parse_args()

    if not args.csv.exists():
        log.error("CSV not found: %s. Re-download or pass --csv.", args.csv)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    enriched = load_and_enrich(args.csv)

    valid_cells = [(e, x, c, k, m) for (e, x, c, k, m) in GRID if x < e]
    if args.limit > 0:
        valid_cells = valid_cells[: args.limit]
    log.info("Running %d cells (out of %d total grid combinations)", len(valid_cells), len(GRID))

    rows: List[Dict[str, Any]] = []
    for i, (e, x, c, k, m) in enumerate(valid_cells, 1):
        try:
            row = run_one(enriched, e, x, c, k, m)
            rows.append(row)
            log.info(
                "[%d/%d] e=%d x=%d c=%d k=%d m=%d → PF=%.3f WR=%.1f%% n=%d",
                i, len(valid_cells), e, x, c, k, m,
                row["profit_factor"], row["win_rate"] * 100, row["n_trades"],
            )
        except Exception as exc:
            log.warning("Cell e=%d x=%d c=%d k=%d m=%d failed: %s", e, x, c, k, m, exc)

    if not rows:
        log.error("All cells failed — no output written")
        return 3

    # Write CSV
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), args.out)

    # Top-10 PF summary
    rows_sorted = sorted(rows, key=lambda r: r["profit_factor"], reverse=True)
    top_md = args.out.with_name("eval_07_sweep_top10.md")
    with top_md.open("w", encoding="utf-8") as fh:
        fh.write("# Eval 07 — State Machine Sweep, Top 10 PF\n\n")
        fh.write(f"Total cells run : {len(rows)}\n")
        fh.write(f"Default `(75, 55, 2, 2, 12)` :\n\n")
        default = next(
            (r for r in rows if (r["enter"], r["exit"], r["confirm"], r["cooldown"], r["max_age"]) == (75, 55, 2, 2, 12)),
            None,
        )
        if default:
            fh.write(
                f"  PF={default['profit_factor']:.3f} WR={default['win_rate']*100:.1f}% "
                f"n={default['n_trades']} sharpe={default['sharpe_per_trade']:+.3f}\n\n"
            )
        fh.write("## Top 10 by Profit Factor\n\n")
        fh.write("| Rank | enter | exit | confirm | cooldown | max_age | PF | WR | n_trades | Sharpe |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for rank, r in enumerate(rows_sorted[:10], 1):
            fh.write(
                f"| {rank} | {r['enter']} | {r['exit']} | {r['confirm']} | "
                f"{r['cooldown']} | {r['max_age']} | {r['profit_factor']:.3f} | "
                f"{r['win_rate']*100:.1f}% | {r['n_trades']} | "
                f"{r['sharpe_per_trade']:+.3f} |\n"
            )
    log.info("Wrote top-10 summary to %s", top_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
