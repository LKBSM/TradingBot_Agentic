"""Sweep paramétrique state machine — Action 2 post Sprint 0 review.

Empirical truth from Sprint 0 baseline : with defaults
``enter_threshold=75``, the additive ConfluenceDetector score plafonne à
``p99 ≈ 70-74`` → 0 trades sur 7 ans.

This script sweeps a small grid of state-machine parameters around the
empirical p90-p99 ceiling and reports each cell's :
- total_trades, profit_factor, sharpe_per_trade
- DSR / PBO / PF lower-CI / DM gates (via src.backtest.validation)
- ranking by (gates_passed, point PF, n_trades)

Outputs
-------
- ``reports/sweep/sweep_results.csv``
- ``reports/sweep/sweep_summary.md`` (ranked table)
- ``reports/sweep/cell_<asset>_<tf>_E<enter>_X<exit>_C<confirm>/`` per-cell
  summary + trades

Usage
-----
::

    python scripts/sweep_state_machine.py [--quick] [--asset XAUUSD]
        --quick : 30k bars per cell (smoke), default = full

Configuration
-------------
Grid : enter ∈ {55, 60, 65, 70}, exit ∈ {35, 40, 45}, confirm ∈ {1, 2}.
= 4 × 3 × 2 = 24 cells per asset.

Reproducibility : seed=42 fixed in run_backtest invocation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = ROOT / "reports" / "sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)


GRID = {
    "enter": [55, 60, 65, 70],
    "exit": [35, 40, 45],
    "confirm": [1, 2],
}

ASSETS = [
    {"name": "xau_m15", "symbol": "XAUUSD", "csv": "data/XAU_15MIN_2019_2026.csv", "tf": "M15"},
    {"name": "eurusd_m15", "symbol": "EURUSD", "csv": "data/EURUSD_15MIN_2019_2025.csv", "tf": "M15"},
]


def cell_id(asset_name: str, enter: int, exitp: int, confirm: int) -> str:
    return f"{asset_name}_E{enter}_X{exitp}_C{confirm}"


def run_cell(asset: dict, enter: int, exitp: int, confirm: int, last_n: int = 0) -> dict:
    cid = cell_id(asset["name"], enter, exitp, confirm)
    out_dir = SWEEP_DIR / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "summary.json"
    out_trades = out_dir / "trades.csv"

    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_backtest.py"),
        "--csv", asset["csv"],
        "--symbol", asset["symbol"],
        "--timeframe", asset["tf"],
        "--enter", str(enter),
        "--exit", str(exitp),
        "--confirm", str(confirm),
        "--cooldown", "2",
        "--max-age", "12",
        "--out", str(out_json),
        "--trades-csv", str(out_trades),
        "--no-retest",
    ]
    if last_n > 0:
        cmd += ["--last-n", str(last_n)]

    print(f"  >>> {cid} ...", flush=True)
    completed = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        return {
            "cell_id": cid,
            "asset": asset["name"],
            "enter": enter, "exit": exitp, "confirm": confirm,
            "status": "FAILED",
            "stderr_tail": (completed.stderr or "")[-500:],
        }

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

    return {
        "cell_id": cid,
        "asset": asset["name"],
        "enter": enter, "exit": exitp, "confirm": confirm,
        "status": "OK",
        "total_trades": int(s.get("total_trades", 0)),
        "wins": int(s.get("wins", 0)),
        "losses": int(s.get("losses", 0)),
        "win_rate": float(s.get("win_rate", 0.0) or 0.0),
        "profit_factor": float(risk.get("profit_factor") or s.get("profit_factor") or 0.0),
        "expectancy_r": float(s.get("expectancy_r", 0.0) or 0.0),
        "sharpe_per_trade": float(risk_adj.get("sharpe_per_trade") or 0.0),
        "sharpe_annualised": (risk_adj.get("sharpe_annualised") or 0.0),
        "max_drawdown_r": float(risk.get("max_drawdown_r") or s.get("max_drawdown_r") or 0.0),
        "score_max": float(s.get("score_max", 0.0) or 0.0),
        "arms_started": int(s.get("arms_started", 0) or 0),
        "summary_path": str(out_json.relative_to(ROOT)),
        "trades_path": str(out_trades.relative_to(ROOT)) if out_trades.exists() else "",
    }


def validate_cell(row: dict) -> dict:
    """Run admission gates on cell trades."""
    if row.get("status") != "OK" or row.get("total_trades", 0) < 1:
        return {"gate_status": "NO_TRADES", "all_gates_passed": False}
    trades_path = ROOT / row["trades_path"]
    if not trades_path.exists():
        return {"gate_status": "NO_TRADES_FILE", "all_gates_passed": False}
    try:
        from src.backtest.validation import validate_trades_dataframe
        import pandas as pd
        df = pd.read_csv(trades_path)
        result = validate_trades_dataframe(df, n_trials=len(GRID["enter"]) * len(GRID["exit"]) * len(GRID["confirm"]))
        return {
            "gate_status": "EVALUATED",
            "all_gates_passed": result.all_passed,
            "dsr": float(result.dsr),
            "pbo": float(result.pbo),
            "pf_lo": float(result.profit_factor_lo),
            "pf_hi": float(result.profit_factor_hi),
            "dm_pvalue": float(result.dm_pvalue),
            "gates_trades": result.trades_pass,
            "gates_dsr": result.dsr_pass,
            "gates_pbo": result.pbo_pass,
            "gates_pf_lo": result.pf_lo_pass,
            "gates_dm": result.dm_pass,
        }
    except Exception as exc:
        return {"gate_status": f"GATE_ERROR: {exc}", "all_gates_passed": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="30k bars per cell (smoke)")
    parser.add_argument("--asset", default=None, help="Only one asset (xau_m15 or eurusd_m15)")
    args = parser.parse_args()

    assets = ASSETS
    if args.asset:
        assets = [a for a in ASSETS if a["name"] == args.asset]
        if not assets:
            print(f"Unknown asset: {args.asset}")
            return 1

    last_n = 30000 if args.quick else 0
    cells = []
    n_total = sum(
        len(GRID["enter"]) * len(GRID["exit"]) * len(GRID["confirm"])
        for _ in assets
    )
    print(f"=== SWEEP — {n_total} cells, mode={'QUICK' if args.quick else 'FULL'} ===\n")

    i = 0
    for asset in assets:
        print(f"[{asset['name']}]")
        for enter in GRID["enter"]:
            for exitp in GRID["exit"]:
                if exitp >= enter:
                    continue
                for confirm in GRID["confirm"]:
                    i += 1
                    row = run_cell(asset, enter, exitp, confirm, last_n=last_n)
                    row.update(validate_cell(row))
                    cells.append(row)
                    print(f"  [{i}/{n_total}] {row['cell_id']}: "
                          f"trades={row.get('total_trades', 0)}, "
                          f"PF={row.get('profit_factor', 0):.3f}, "
                          f"gates={row.get('all_gates_passed', False)}")

    csv_path = SWEEP_DIR / "sweep_results.csv"
    fieldnames = sorted({k for r in cells for k in r.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in cells:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    md_path = SWEEP_DIR / "sweep_summary.md"
    cells_ok = [c for c in cells if c.get("status") == "OK"]
    cells_with_trades = [c for c in cells_ok if c.get("total_trades", 0) > 0]
    gate_passers = [c for c in cells_with_trades if c.get("all_gates_passed")]

    cells_with_trades.sort(
        key=lambda r: (-r.get("profit_factor", 0.0), -r.get("total_trades", 0)),
    )

    lines = [
        f"# Sweep paramétrique state machine — {len(cells)} cells",
        "",
        f"**Mode** : {'QUICK (30k bars)' if args.quick else 'FULL (all bars)'}",
        f"**Grid** : enter ∈ {GRID['enter']}, exit ∈ {GRID['exit']}, confirm ∈ {GRID['confirm']}",
        f"**Cells avec trades** : {len(cells_with_trades)} / {len(cells_ok)}",
        f"**Cells qui passent les gates** : {len(gate_passers)}",
        "",
        "## Top 20 cells par profit factor",
        "",
        "| cell | trades | PF | PF_lo | DSR | PBO | gates |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in cells_with_trades[:20]:
        gates_str = "✅" if c.get("all_gates_passed") else "❌"
        lines.append(
            f"| `{c['cell_id']}` | {c.get('total_trades', 0)} | "
            f"{c.get('profit_factor', 0):.3f} | "
            f"{c.get('pf_lo', 0):.3f} | "
            f"{c.get('dsr', 0):.3f} | "
            f"{c.get('pbo', 0.5):.3f} | "
            f"{gates_str} |"
        )

    if gate_passers:
        lines.append("")
        lines.append("## ✅ Cells qui passent toutes les gates (commercialisable)")
        for c in gate_passers:
            lines.append(f"- `{c['cell_id']}` — PF={c.get('profit_factor', 0):.3f}, "
                         f"DSR={c.get('dsr', 0):.3f}, PF_lo={c.get('pf_lo', 0):.3f}")
    else:
        lines.append("")
        lines.append("## ❌ Aucune cell ne passe toutes les gates")
        lines.append("")
        lines.append("Le sweep a généré des trades mais aucune configuration ne franchit "
                     "simultanément DSR ≥ 1.5, PBO ≤ 0.35, PF_lo > 1.0, DM_p < 0.05.")
        lines.append("Recommandation : pivot ou actions Sprint 4 (logistic L1 sur composantes).")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"\n=== SWEEP DONE === {len(cells_with_trades)} cells with trades, "
          f"{len(gate_passers)} pass gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
