"""DG-034 — Sweep paramétrique 432 cellules du SignalStateMachine.

Grid (6 dimensions, 432 = 4 × 3 × 3 × 2 × 2 × 3) :

- enter_threshold      ∈ {55, 60, 65, 70}            (4)
- exit_threshold       ∈ {25, 35, 45}                (3, exit < enter enforced)
- confirm_bars         ∈ {1, 2, 3}                   (3)
- cooldown_bars        ∈ {0, 4}                      (2)
- max_signal_age_bars  ∈ {8, 16}                     (2)
- silent_bars          ∈ {1, 2, 3}                   (3)

Per cell → run :mod:`scripts.run_backtest` on XAU M15 2019-2026 (default)
and parse the summary JSON + trades CSV. Output :

- ``reports/sweep_432/sweep_results.csv`` (one row per cell)
- ``reports/sweep_432/sweep_summary.md`` (ranked, gate-checked)
- ``reports/sweep_432/cell_<id>/`` per-cell artefacts

Usage
-----
::

    # Full XAU M15 2019-2026 — multi-hour run, designed for Colab
    python scripts/sweep_state_machine_432.py --asset xau_m15

    # Quick smoke test (8 random cells, 20k bars each)
    python scripts/sweep_state_machine_432.py --smoke --asset xau_m15

    # EURUSD instead
    python scripts/sweep_state_machine_432.py --asset eurusd_m15

Empirical defaults will be derived from this sweep after the full run
completes. The deliverable for Sprint Tech 1 is :
- the script + 432-cell coverage demonstrated on a representative subset
- a Colab-ready harness for the full run
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = ROOT / "reports" / "sweep_432"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)


GRID: Dict[str, List[int]] = {
    "enter":     [55, 60, 65, 70],
    "exit":      [25, 35, 45],
    "confirm":   [1, 2, 3],
    "cooldown":  [0, 4],
    "max_age":   [8, 16],
    "silent":    [1, 2, 3],
}

ASSETS = [
    {"name": "xau_m15", "symbol": "XAUUSD", "csv": "data/XAU_15MIN_2019_2026.csv", "tf": "M15"},
    {"name": "eurusd_m15", "symbol": "EURUSD", "csv": "data/EURUSD_15MIN_2019_2025.csv", "tf": "M15"},
]


def cell_id(a: str, e: int, x: int, c: int, cd: int, ma: int, s: int) -> str:
    return f"{a}_E{e}_X{x}_C{c}_CD{cd}_MA{ma}_S{s}"


def enumerate_grid(asset_name: str):
    """Yield all valid grid cells (exit < enter enforced)."""
    for e in GRID["enter"]:
        for x in GRID["exit"]:
            if x >= e:
                continue
            for c in GRID["confirm"]:
                for cd in GRID["cooldown"]:
                    for ma in GRID["max_age"]:
                        for s in GRID["silent"]:
                            yield cell_id(asset_name, e, x, c, cd, ma, s), {
                                "enter": e, "exit": x, "confirm": c,
                                "cooldown": cd, "max_age": ma, "silent": s,
                            }


def run_cell(asset: dict, params: dict, cid: str, last_n: int = 0) -> dict:
    out_dir = SWEEP_DIR / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "summary.json"
    out_trades = out_dir / "trades.csv"

    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_backtest.py"),
        "--csv", asset["csv"],
        "--symbol", asset["symbol"],
        "--timeframe", asset["tf"],
        "--enter", str(params["enter"]),
        "--exit", str(params["exit"]),
        "--confirm", str(params["confirm"]),
        "--cooldown", str(params["cooldown"]),
        "--max-age", str(params["max_age"]),
        "--silent", str(params["silent"]),
        "--out", str(out_json),
        "--trades-csv", str(out_trades),
        "--no-retest",
    ]
    if last_n > 0:
        cmd += ["--last-n", str(last_n)]

    completed = subprocess.run(
        cmd, capture_output=True, text=True, cwd=ROOT,
        encoding="utf-8", errors="replace",
    )
    base = {"cell_id": cid, "asset": asset["name"], **params}
    if completed.returncode != 0:
        base.update({"status": "FAILED", "stderr_tail": (completed.stderr or "")[-300:]})
        return base

    summary = {}
    if out_json.exists():
        try:
            summary = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    s = summary.get("summary", {}) or {}
    ins = summary.get("institutional_metrics", {}) or {}
    risk = ins.get("risk", {}) or {}
    risk_adj = ins.get("risk_adjusted", {}) or {}
    base.update({
        "status": "OK",
        "total_trades": int(s.get("total_trades", 0)),
        "wins": int(s.get("wins", 0)),
        "losses": int(s.get("losses", 0)),
        "win_rate": float(s.get("win_rate", 0.0) or 0.0),
        "profit_factor": float(risk.get("profit_factor") or s.get("profit_factor") or 0.0),
        "expectancy_r": float(s.get("expectancy_r", 0.0) or 0.0),
        "sharpe_per_trade": float(risk_adj.get("sharpe_per_trade") or 0.0),
        "max_drawdown_r": float(risk.get("max_drawdown_r") or s.get("max_drawdown_r") or 0.0),
        "score_max": float(s.get("score_max", 0.0) or 0.0),
        "summary_path": str(out_json.relative_to(ROOT)),
        "trades_path": str(out_trades.relative_to(ROOT)) if out_trades.exists() else "",
    })
    return base


def validate_cell(row: dict, n_trials: int) -> dict:
    if row.get("status") != "OK" or row.get("total_trades", 0) < 1:
        return {"gate_status": "NO_TRADES", "all_gates_passed": False}
    tp = row.get("trades_path")
    if not tp:
        return {"gate_status": "NO_TRADES_FILE", "all_gates_passed": False}
    trades_path = ROOT / tp
    if not trades_path.exists():
        return {"gate_status": "NO_TRADES_FILE", "all_gates_passed": False}
    try:
        from src.backtest.validation import validate_trades_dataframe  # noqa: E402
        import pandas as pd  # noqa: E402
        df = pd.read_csv(trades_path)
        result = validate_trades_dataframe(df, n_trials=n_trials)
        return {
            "gate_status": "EVALUATED",
            "all_gates_passed": bool(result.all_passed),
            "dsr": float(result.dsr),
            "pbo": float(result.pbo),
            "pf_lo": float(result.profit_factor_lo),
            "pf_hi": float(result.profit_factor_hi),
            "dm_pvalue": float(result.dm_pvalue),
        }
    except Exception as exc:
        return {"gate_status": f"GATE_ERROR: {exc}", "all_gates_passed": False}


def write_outputs(cells: list, asset_name: str) -> None:
    csv_path = SWEEP_DIR / f"sweep_results_{asset_name}.csv"
    fieldnames = sorted({k for r in cells for k in r.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in cells:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    cells_ok = [c for c in cells if c.get("status") == "OK"]
    cells_with_trades = [c for c in cells_ok if c.get("total_trades", 0) > 0]
    gate_passers = [c for c in cells_with_trades if c.get("all_gates_passed")]
    cells_with_trades.sort(
        key=lambda r: (-r.get("profit_factor", 0.0), -r.get("total_trades", 0)),
    )

    md_path = SWEEP_DIR / f"sweep_summary_{asset_name}.md"
    lines = [
        f"# Sweep 432 cellules — {asset_name} — {len(cells)} cellules",
        "",
        f"**Grid** : enter ∈ {GRID['enter']}, exit ∈ {GRID['exit']}, "
        f"confirm ∈ {GRID['confirm']}, cooldown ∈ {GRID['cooldown']}, "
        f"max_age ∈ {GRID['max_age']}, silent ∈ {GRID['silent']}.",
        f"**Cells avec trades** : {len(cells_with_trades)} / {len(cells_ok)}",
        f"**Cells qui passent les gates** : {len(gate_passers)}",
        "",
        "## Top 20 cellules par profit factor",
        "",
        "| cell | trades | PF | PF_lo | DSR | PBO | gates |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in cells_with_trades[:20]:
        gates_str = "OK" if c.get("all_gates_passed") else "NO"
        lines.append(
            f"| `{c['cell_id']}` | {c.get('total_trades', 0)} | "
            f"{c.get('profit_factor', 0):.3f} | {c.get('pf_lo', 0):.3f} | "
            f"{c.get('dsr', 0):.3f} | {c.get('pbo', 0.5):.3f} | {gates_str} |"
        )
    if not gate_passers:
        lines += [
            "",
            "## Aucune cellule ne passe TOUTES les gates",
            "",
            "Gates : DSR ≥ 1.5, PBO ≤ 0.35, PF_lo > 1.0, DM_p < 0.05.",
            "Aligne le verdict empirique avec reports/certification/ACTIONS_1_2_3_RESULTS.md.",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="DG-034 432-cell state machine sweep")
    p.add_argument("--asset", default="xau_m15", choices=["xau_m15", "eurusd_m15"])
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 8 random cells, 20k bars each.")
    p.add_argument("--last-n", type=int, default=0)
    p.add_argument("--max-cells", type=int, default=0,
                   help="Limit number of cells (debug/preview).")
    args = p.parse_args()

    asset = next((a for a in ASSETS if a["name"] == args.asset), None)
    if asset is None:
        print(f"Unknown asset: {args.asset}")
        return 1

    all_cells = list(enumerate_grid(args.asset))
    n_total = len(all_cells)
    if args.smoke:
        random.seed(42)
        all_cells = random.sample(all_cells, k=min(8, n_total))
        args.last_n = args.last_n or 20000
    elif args.max_cells > 0:
        all_cells = all_cells[: args.max_cells]

    print(f"=== SWEEP 432 — {args.asset} — {len(all_cells)} cellules "
          f"(grid total = {n_total}) ===\n", flush=True)

    results = []
    for i, (cid, params) in enumerate(all_cells, 1):
        row = run_cell(asset, params, cid, last_n=args.last_n)
        row.update(validate_cell(row, n_trials=n_total))
        results.append(row)
        print(
            f"[{i}/{len(all_cells)}] {cid}: "
            f"trades={row.get('total_trades', 0)}, "
            f"PF={row.get('profit_factor', 0):.3f}, "
            f"gates={row.get('all_gates_passed', False)}",
            flush=True,
        )

    write_outputs(results, args.asset)
    cells_passing = sum(1 for r in results if r.get("all_gates_passed"))
    print(f"\n=== DONE === {len(results)} cells, {cells_passing} passing all gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
