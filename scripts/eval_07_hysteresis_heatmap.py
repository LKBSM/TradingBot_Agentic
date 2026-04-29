"""Focused 5×5 heatmap PF over (enter_threshold × exit_threshold).

Faster delta vs the full eval_07 sweep (280 cells, 2-8 h). Fixes
confirm_bars=2, cooldown_bars=2, max_signal_age_bars=12 (current defaults)
so the heatmap isolates the hysteresis pair impact.

Usage:
    python scripts/eval_07_hysteresis_heatmap.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.state_machine_replay import SignalReplay  # noqa: E402
from src.environment.strategy_features import SmartMoneyEngine  # noqa: E402
from src.intelligence.signal_state_machine import StateMachineConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_07_heatmap")


def load_enriched(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for cand in ("timestamp", "Date", "date", "datetime", "time"):
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand])
            df = df.set_index(cand)
            break
    rename = {c: c.lower().capitalize() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
    if rename:
        df = df.rename(columns=rename)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    log.info("Loaded %d bars (%s -> %s)", len(df), df.index.min(), df.index.max())
    log.info("SMC enrichment...")
    return SmartMoneyEngine(
        data=df,
        config={
            "RSI_WINDOW": 14, "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
            "BB_WINDOW": 20, "ATR_WINDOW": 14,
            "FRACTAL_WINDOW": 2, "FVG_THRESHOLD": 0.1,
            "OB_REQUIRE_FVG": False,
        },
    ).analyze()


def main():
    csv_path = ROOT / "data" / "XAU_15MIN_2019_2024.csv"
    enriched = load_enriched(csv_path)

    enters = [65, 70, 75, 80, 85]
    exits = [40, 45, 50, 55, 60]

    cells = [(e, x) for e in enters for x in exits if x < e]
    log.info("Running %d cells", len(cells))

    rows = []
    for i, (e, x) in enumerate(cells, 1):
        cfg = StateMachineConfig(
            symbol="XAUUSD",
            enter_threshold=float(e),
            exit_threshold=float(x),
            confirm_bars=2,
            cooldown_bars=2,
            max_signal_age_bars=12,
        )
        replay = SignalReplay(
            symbol="XAUUSD", timeframe="M15",
            state_machine_config=cfg,
            use_regime=True,
            use_vol_regime=True,
        )
        res = replay.run(enriched)
        rows.append({
            "enter": e, "exit": x,
            "n_trades": res.total_trades,
            "win_rate": round(res.win_rate, 4),
            "profit_factor": round(res.profit_factor, 4),
            "sharpe_per_trade": round(res.sharpe_per_trade, 4),
            "total_pnl_r": round(getattr(res, "total_pnl_r", 0.0), 4),
            "max_drawdown_r": round(getattr(res, "max_drawdown_r", 0.0), 4),
        })
        log.info("[%d/%d] e=%d x=%d -> PF=%.3f WR=%.1f%% n=%d",
                 i, len(cells), e, x, rows[-1]["profit_factor"],
                 rows[-1]["win_rate"] * 100, rows[-1]["n_trades"])

    out_dir = ROOT / "reports" / "eval_07"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "hysteresis_heatmap.json", "w") as f:
        json.dump(rows, f, indent=2)

    md = out_dir / "hysteresis_heatmap.md"
    rows_sorted = sorted(rows, key=lambda r: r["profit_factor"], reverse=True)
    with open(md, "w") as f:
        f.write("# Eval 07 — Hysteresis heatmap (enter × exit)\n\n")
        f.write(f"Configuration : confirm_bars=2, cooldown_bars=2, max_signal_age_bars=12 (defaults). XAU M15 2019-2024.\n\n")
        f.write("## Heatmap PF\n\n")
        f.write("| enter \\ exit | " + " | ".join(str(x) for x in exits) + " |\n")
        f.write("|---|" + "---|" * len(exits) + "\n")
        for e in enters:
            line = f"| **{e}** | "
            for x in exits:
                if x < e:
                    r = next((r for r in rows if r["enter"] == e and r["exit"] == x), None)
                    if r:
                        line += f"{r['profit_factor']:.3f} (n={r['n_trades']}) | "
                    else:
                        line += "—  | "
                else:
                    line += "—  | "
            f.write(line + "\n")
        f.write("\n## Top cells by PF\n\n")
        f.write("| Rank | enter | exit | PF | WR | n_trades | Total PnL (R) | MDD (R) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for rank, r in enumerate(rows_sorted[:5], 1):
            f.write(f"| {rank} | {r['enter']} | {r['exit']} | {r['profit_factor']:.3f} | "
                    f"{r['win_rate']*100:.1f}% | {r['n_trades']} | {r['total_pnl_r']:.2f} | "
                    f"{r['max_drawdown_r']:.2f} |\n")

    log.info("Wrote %s and %s", out_dir / "hysteresis_heatmap.json", md)


if __name__ == "__main__":
    main()
