"""Sprint 0 Batch 0.2 — Baseline backtest orchestrator.

Runs the institutional baseline on 2 (asset, timeframe) configurations,
captures JSON + trade CSV outputs, computes bootstrap 95% CI on PF,
writes a consolidated `reports/baseline/baseline_report.{md,json}`.

Configurations
--------------
- XAU M15  (data/XAU_15MIN_2019_2026.csv, 98.72% coverage 2019-2025)
- EURUSD M15 (data/EURUSD_15MIN_2019_2025.csv, 99.41% coverage)

H1 timeframes are deferred to Sprint 1 (resampling without look-ahead must
be formally validated before claiming H1 baselines).

Reproducibility
---------------
- Seeds : numpy / random / pythonhashseed seeded explicitly.
- All outputs hashed (SHA256) into reports/baseline/checksums.txt.
- Config snapshot (hyperparams + libs + commit SHA) in
  reports/baseline/config_snapshot_2026-05-15.json.

Usage
-----
::

    python scripts/run_baseline_sprint0.py [--quick]
        --quick → 20k bars per config (smoke test, ~1 min total)
        full    → all bars per config (~2-5 min per config)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports" / "baseline"
SEED = 42

CONFIGS = [
    {
        "name": "xau_m15",
        "csv": "data/XAU_15MIN_2019_2026.csv",
        "symbol": "XAUUSD",
        "timeframe": "M15",
    },
    {
        "name": "eurusd_m15",
        "csv": "data/EURUSD_15MIN_2019_2025.csv",
        "symbol": "EURUSD",
        "timeframe": "M15",
    },
]


def setup_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_backtest(config: dict, last_n: int = 0) -> dict:
    """Invoke scripts/run_backtest.py for one config, capture artifacts."""
    out_json = REPORTS_DIR / f"{config['name']}_summary.json"
    out_trades = REPORTS_DIR / f"{config['name']}_trades.csv"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_backtest.py"),
        "--csv", config["csv"],
        "--symbol", config["symbol"],
        "--timeframe", config["timeframe"],
        "--out", str(out_json),
        "--trades-csv", str(out_trades),
        "--no-retest",  # baseline = pre-retest engine, fair comparison vs historical reports
    ]
    if last_n > 0:
        cmd.extend(["--last-n", str(last_n)])

    print(f"\n>>> Running: {config['name']}  ({config['symbol']} {config['timeframe']})")
    print(f"    cmd: {' '.join(cmd)}")
    completed = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if completed.returncode != 0:
        print(f"    STDERR (tail):\n{completed.stderr[-2000:]}")
        return {
            "name": config["name"],
            "status": "failed",
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-1000:],
        }

    summary = {}
    if out_json.exists():
        summary = json.loads(out_json.read_text(encoding="utf-8"))

    return {
        "name": config["name"],
        "symbol": config["symbol"],
        "timeframe": config["timeframe"],
        "csv": config["csv"],
        "status": "ok",
        "summary": summary,
        "json_path": str(out_json.relative_to(ROOT)),
        "trades_path": str(out_trades.relative_to(ROOT)),
        "json_sha256": sha256_file(out_json) if out_json.exists() else None,
        "trades_sha256": sha256_file(out_trades) if out_trades.exists() else None,
    }


def bootstrap_pf_ci(trades_path: Path, n_iter: int = 10000, seed: int = SEED) -> dict:
    """Bootstrap 95% CI on profit factor from trade-level returns."""
    import pandas as pd
    if not trades_path.exists():
        return {"status": "no_trades_file"}
    df = pd.read_csv(trades_path)
    if "pnl" not in df.columns or len(df) == 0:
        return {"status": "no_pnl_col_or_empty", "n_trades": int(len(df))}

    pnls = df["pnl"].to_numpy()
    n = len(pnls)
    rng = np.random.default_rng(seed)

    def pf(p):
        gains = p[p > 0].sum()
        losses = -p[p < 0].sum()
        if losses <= 0:
            return float("inf") if gains > 0 else float("nan")
        return float(gains / losses)

    point = pf(pnls)
    boots = np.empty(n_iter)
    for i in range(n_iter):
        sample = pnls[rng.integers(0, n, size=n)]
        boots[i] = pf(sample)

    finite = boots[np.isfinite(boots)]
    pct_low = float(np.percentile(finite, 2.5)) if finite.size else float("nan")
    pct_high = float(np.percentile(finite, 97.5)) if finite.size else float("nan")
    return {
        "status": "ok",
        "n_trades": int(n),
        "point": point,
        "ci_low": pct_low,
        "ci_high": pct_high,
        "n_iter": n_iter,
        "n_finite_boots": int(finite.size),
    }


def snapshot_config() -> dict:
    """Capture configuration, env, code hashes, lib versions."""
    snap = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "seed": SEED,
        "python": sys.version.split()[0],
    }

    try:
        import config as proj
        snap["config_py"] = {
            "HISTORICAL_DATA_FILE": str(getattr(proj, "HISTORICAL_DATA_FILE", "")),
            "DATA_DIR": str(getattr(proj, "DATA_DIR", "")),
        }
    except Exception as exc:
        snap["config_py_error"] = str(exc)

    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        keep = ("pandas", "numpy", "scipy", "scikit-learn", "lightgbm", "hmmlearn",
                "pydantic", "pytest", "matplotlib", "arch")
        snap["libs"] = {
            line.split("==")[0]: line.split("==")[1]
            for line in freeze.splitlines()
            if "==" in line and line.split("==")[0].lower() in keep
        }
    except Exception as exc:
        snap["libs_error"] = str(exc)

    code_files: list[tuple[str, str]] = []
    for sub in ("src/intelligence", "src/backtest", "src/environment"):
        d = ROOT / sub
        if d.exists():
            for f in sorted(d.rglob("*.py")):
                rel = str(f.relative_to(ROOT))
                code_files.append((rel, sha256_file(f)))
    snap["code_hashes"] = dict(code_files[:500])  # cap

    data_files: list[tuple[str, str]] = []
    for f in sorted((ROOT / "data").glob("*.csv")):
        data_files.append((str(f.relative_to(ROOT)), sha256_file(f)))
    snap["data_hashes"] = dict(data_files)

    return snap


def render_markdown(results: list[dict], cis: dict, snap: dict, mode: str) -> str:
    lines = [
        "# Baseline Sprint 0 — Report",
        "",
        f"**Date** : {snap['timestamp_utc']}",
        f"**Mode** : `{mode}`",
        f"**Commit** : `{snap['git_commit']}` (`{snap['git_branch']}`)",
        f"**Seed** : {snap['seed']}",
        "",
        "## Configurations",
        "",
        "| Name | Symbol | TF | CSV | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        status = "✅ ok" if r.get("status") == "ok" else f"❌ {r.get('status')}"
        lines.append(
            f"| {r.get('name')} | {r.get('symbol', '?')} | {r.get('timeframe', '?')} | "
            f"`{r.get('csv', '?')}` | {status} |"
        )

    lines.append("")
    lines.append("## Métriques par configuration")
    lines.append("")
    for r in results:
        lines.append(f"### {r['name']} — {r.get('symbol')} {r.get('timeframe')}")
        if r.get("status") != "ok":
            lines.append(f"❌ Failed (returncode {r.get('returncode')}). Stderr tail:")
            lines.append("```")
            lines.append(r.get("stderr_tail", "")[-800:])
            lines.append("```")
            continue
        summary = r.get("summary", {}) or {}
        metrics = summary.get("metrics") or summary
        # Robust access — schema may vary
        def g(key, default="n/a"):
            v = metrics.get(key) if isinstance(metrics, dict) else None
            return v if v is not None else default
        lines.append(f"- profit_factor : `{g('profit_factor')}`")
        lines.append(f"- sharpe        : `{g('sharpe')}`  (annualized: `{g('sharpe_annualized')}`)")
        lines.append(f"- sortino       : `{g('sortino')}`")
        lines.append(f"- max_drawdown  : `{g('max_drawdown')}`")
        lines.append(f"- win_rate      : `{g('win_rate')}`")
        lines.append(f"- nb_trades     : `{g('trades_count')}` (or `{summary.get('trades_count')}`)")
        ci = cis.get(r["name"], {})
        if ci.get("status") == "ok":
            lines.append(
                f"- **PF 95% CI bootstrap** : point=`{ci['point']:.3f}` "
                f"[`{ci['ci_low']:.3f}`, `{ci['ci_high']:.3f}`] "
                f"(n_trades={ci['n_trades']}, {ci['n_iter']} resamples, finite={ci['n_finite_boots']})"
            )
        else:
            lines.append(f"- PF 95% CI bootstrap : ⚠️ {ci.get('status', 'n/a')}")
        lines.append(f"- JSON: `{r.get('json_path')}` (sha256 `{(r.get('json_sha256') or '?')[:16]}…`)")
        lines.append(f"- Trades: `{r.get('trades_path')}` (sha256 `{(r.get('trades_sha256') or '?')[:16]}…`)")
        lines.append("")

    lines.append("## Lib snapshot")
    for k, v in (snap.get("libs") or {}).items():
        lines.append(f"- `{k}` = `{v}`")
    lines.append("")
    lines.append("## Config")
    cfg = snap.get("config_py", {})
    lines.append(f"- HISTORICAL_DATA_FILE = `{cfg.get('HISTORICAL_DATA_FILE')}`")
    lines.append("")
    lines.append("## Reproductibilité")
    lines.append("")
    lines.append("Pour rejouer cette baseline à l'identique :")
    lines.append("")
    lines.append("```bash")
    lines.append(f"git checkout {snap['git_commit']}")
    lines.append("python scripts/run_baseline_sprint0.py" + ("  --quick" if mode == "quick" else ""))
    lines.append("```")
    lines.append("")
    lines.append("Toute différence dans les SHA256 des fichiers ci-dessus indique une dérive "
                 "(env, data, ou code).")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="20k bars per config (smoke)")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_seeds()

    last_n = 20_000 if args.quick else 0

    results: list[dict] = []
    for cfg in CONFIGS:
        results.append(run_backtest(cfg, last_n=last_n))

    cis: dict[str, dict] = {}
    for r in results:
        if r.get("status") == "ok":
            cis[r["name"]] = bootstrap_pf_ci(ROOT / r["trades_path"])
        else:
            cis[r["name"]] = {"status": "skipped_failed_run"}

    snap = snapshot_config()

    payload = {
        "results": results,
        "bootstrap_ci": cis,
        "snapshot": snap,
        "mode": "quick" if args.quick else "full",
    }
    (REPORTS_DIR / "baseline_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (REPORTS_DIR / "config_snapshot_2026-05-15.json").write_text(
        json.dumps(snap, indent=2, default=str), encoding="utf-8"
    )
    md = render_markdown(results, cis, snap, mode=("quick" if args.quick else "full"))
    (REPORTS_DIR / "baseline_report.md").write_text(md, encoding="utf-8")

    # Checksums
    checksums = []
    for fname in sorted((REPORTS_DIR).glob("*")):
        if fname.is_file():
            checksums.append(f"{sha256_file(fname)}  {fname.relative_to(ROOT)}")
    (REPORTS_DIR / "checksums.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")

    print("\n=== BASELINE COMPLETE ===")
    for r in results:
        status = "✅" if r.get("status") == "ok" else "❌"
        print(f"  {status} {r['name']}")
    print(f"\nReports in {REPORTS_DIR.relative_to(ROOT)}/")
    return 0 if all(r.get("status") == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
