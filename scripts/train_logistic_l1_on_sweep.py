"""Train LogisticL1Scorer on trades produced by the parameter sweep.

Action 3 (minimum viable) — post Sprint 0 review.

Inputs
------
- All ``reports/sweep/cell_*/trades.csv`` (produced by
  ``scripts/sweep_state_machine.py``).
- Each trade has : confluence_score, direction, exit_reason, bars_held,
  entry_bar (timestamp), r_multiple (target = R > 0 = win).

Features built (no refactor of TradeRecord needed)
--------------------------------------------------
- score_z : (confluence_score - 50) / 25  (standardised around the
  empirical center of the additive ceiling).
- is_long : 1 if LONG else 0.
- hour_sin / hour_cos : circular embedding of entry hour (UTC).
- bars_held_log : log1p(bars_held) — proxy for time-in-trade.
- exit_natural : 1 if exit_reason ∈ {take_profit, signal_change, max_age},
                  0 if {stop_loss, opposing_lockout, manual}.

Target
------
y = (r_multiple > 0) (binary win/loss). Breakeven trades (|R| < 0.05) are
dropped before fit.

Output
------
- ``reports/sweep/logistic_l1_report.md`` — coefficients, Brier skill,
  in-sample vs OOS calibration plot data, sparsity.
- ``models/scoring_v3_logistic_l1.pkl`` — trained model.

Usage
-----
::

    python scripts/train_logistic_l1_on_sweep.py [--min-trades 100]
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SWEEP_DIR = ROOT / "reports" / "sweep"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_NAMES = ("score_z", "is_long", "hour_sin", "hour_cos",
                 "bars_held_log", "exit_natural")


def gather_trades() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for trades_csv in sorted(SWEEP_DIR.glob("cell_*/trades.csv")):
        try:
            df = pd.read_csv(trades_csv)
            df["__cell_id"] = trades_csv.parent.name.replace("cell_", "")
            frames.append(df)
        except Exception:
            continue
    # Also pickup direct cell dirs
    for trades_csv in sorted(SWEEP_DIR.glob("*/trades.csv")):
        try:
            df = pd.read_csv(trades_csv)
            df["__cell_id"] = trades_csv.parent.name
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # dedup by signal_id if present (sweep cells overlap)
    if "signal_id" in out.columns:
        out = out.drop_duplicates(subset="signal_id", keep="first")
    return out


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X, y, df_with_features). Drops breakevens and NaNs."""
    out = df.copy()

    # Target : win (R > 0). Drop breakevens.
    if "r_multiple" not in out.columns:
        raise ValueError("trades CSV missing 'r_multiple' column")
    out = out[out["r_multiple"].abs() >= 0.05].copy()
    out["y"] = (out["r_multiple"] > 0).astype(int)

    # Features
    out["score_z"] = (out["confluence_score"] - 50.0) / 25.0
    out["is_long"] = (out["direction"] == "LONG").astype(int)

    # Hour
    out["entry_bar_ts"] = pd.to_datetime(out["entry_bar"], errors="coerce")
    hour = out["entry_bar_ts"].dt.hour.fillna(0).astype(float).to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Bars held
    out["bars_held_log"] = np.log1p(out["bars_held"].astype(float))

    # Exit reason
    natural = {"take_profit", "signal_change", "max_age"}
    out["exit_natural"] = out["exit_reason"].astype(str).isin(natural).astype(int)

    out = out.dropna(subset=["score_z", "hour_sin", "hour_cos", "bars_held_log"])
    X = out[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = out["y"].to_numpy(dtype=int)
    return X, y, out


def time_split(df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
               oos_frac: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = df["entry_bar_ts"].fillna(pd.Timestamp(0)).argsort().to_numpy()
    n = len(order)
    n_train = int(n * (1 - oos_frac))
    idx_train = order[:n_train]
    idx_test = order[n_train:]
    return X[idx_train], X[idx_test], y[idx_train], y[idx_test]


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def brier_skill(p: np.ndarray, y: np.ndarray) -> float:
    """BS skill vs constant base rate."""
    base = float(np.mean(y))
    bs_const = float(np.mean((base - y) ** 2))
    if bs_const < 1e-12:
        return 0.0
    return 1.0 - brier_score(p, y) / bs_const


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--oos-frac", type=float, default=0.3)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    print("=== Aggregating sweep trades ===")
    df = gather_trades()
    if df.empty:
        print("❌ No trades found. Run the sweep first: python scripts/sweep_state_machine.py")
        return 1
    print(f"Loaded {len(df)} unique trades from sweep cells")

    if len(df) < args.min_trades:
        print(f"❌ Only {len(df)} trades — below --min-trades {args.min_trades}. "
              "Need more variety in sweep grid or full-data sweep.")
        return 2

    X, y, df_feat = build_features(df)
    print(f"After feature build : {len(X)} trades  (base rate win={y.mean():.3f})")

    X_tr, X_te, y_tr, y_te = time_split(df_feat, X, y, oos_frac=args.oos_frac)
    print(f"Train : {len(X_tr)}, OOS : {len(X_te)}")

    if len(X_tr) < 30 or len(X_te) < 30:
        print(f"❌ Splits too small (train={len(X_tr)}, oos={len(X_te)})")
        return 3

    from src.intelligence.scoring import LogisticL1Scorer, LGBMScorer

    results = {}
    for name, ctor in [
        ("logistic_l1", lambda: LogisticL1Scorer(C=args.C, component_names=FEATURE_NAMES)),
        ("lgbm", lambda: LGBMScorer(
            num_leaves=15, learning_rate=0.05, n_estimators=200,
            feature_names=FEATURE_NAMES,
        )),
    ]:
        print(f"\n--- {name.upper()} ---")
        scorer = ctor()
        scorer.fit(X_tr, y_tr)
        p_tr = scorer.predict_p_win(X_tr)
        p_te = scorer.predict_p_win(X_te)
        bs_tr = brier_skill(p_tr, y_tr)
        bs_te = brier_skill(p_te, y_te)
        print(f"  Brier skill IS  : {bs_tr:+.4f}")
        print(f"  Brier skill OOS : {bs_te:+.4f}")

        if name == "logistic_l1":
            coefs = scorer.coefficients()
            for k, v in coefs.items():
                marker = "+" if abs(v) > 1e-8 else " "
                print(f"    {marker} {k:18s}  {v:+.4f}")
            extra = {"coefficients": coefs}
        else:
            importance = scorer.feature_importance()
            for k, v in sorted(importance.items(), key=lambda x: -x[1]):
                print(f"    importance {k:18s}  {v}")
            extra = {"feature_importance": importance}

        model_path = MODELS_DIR / f"scoring_v3_{name}.pkl"
        with model_path.open("wb") as f:
            pickle.dump(scorer, f)
        print(f"  Saved : {model_path}")
        results[name] = {
            "bs_tr": bs_tr, "bs_te": bs_te,
            "model_path": str(model_path.relative_to(ROOT)),
            **extra,
        }

    # Pick winner = best OOS Brier skill
    winner = max(results.items(), key=lambda kv: kv[1]["bs_te"])
    print(f"\n🏆 Winner OOS : {winner[0]} (BS skill OOS = {winner[1]['bs_te']:+.4f})")

    # For backward compat with older code, keep variables named after first scorer
    bs_tr = results["logistic_l1"]["bs_tr"]
    bs_te = results["logistic_l1"]["bs_te"]
    coefs = results["logistic_l1"]["coefficients"]
    nz = [k for k, v in coefs.items() if abs(v) > 1e-8]

    # Report
    report = MODELS_DIR.parent / "reports" / "sweep" / "scoring_training_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Action 3 — Scoring training report (L1 vs LightGBM)",
        "",
        f"**Input** : {len(df)} trades aggregated from sweep cells.",
        f"**After feature build** : {len(X)} trades (drop breakevens |R| < 0.05).",
        f"**Train / OOS split** : {len(X_tr)} / {len(X_te)} (time-ordered).",
        f"**Base rate (P(win))** : {y.mean():.3f}",
        f"**Winner OOS** : `{winner[0]}` (BS skill = {winner[1]['bs_te']:+.4f})",
        "",
        "## A/B comparison",
        "",
        "| Model | BS skill IS | BS skill OOS | Verdict |",
        "| --- | --- | --- | --- |",
    ]
    for name, r in results.items():
        verdict = "✅ predictive OOS" if r["bs_te"] > 0.03 else "❌ weak OOS"
        lines.append(f"| `{name}` | {r['bs_tr']:+.4f} | {r['bs_te']:+.4f} | {verdict} |")

    lines.append("")
    lines.append("## Logistic L1 coefficients")
    lines.append("")
    lines.append("| Feature | Coef | Kept |")
    lines.append("| --- | --- | --- |")
    for k, v in coefs.items():
        kept = "✅" if abs(v) > 1e-8 else "❌ dropped"
        lines.append(f"| `{k}` | {v:+.4f} | {kept} |")

    if "feature_importance" in results.get("lgbm", {}):
        lines.append("")
        lines.append("## LightGBM feature importance")
        lines.append("")
        lines.append("| Feature | Importance |")
        lines.append("| --- | --- |")
        for k, v in sorted(results["lgbm"]["feature_importance"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{k}` | {v} |")

    lines.append("")
    lines.append("## Interprétation")
    lines.append("")
    if bs_te > 0.03:
        lines.append("✅ **Modèle prédictif OOS** : Brier skill > +0.03 — gain réel vs base rate.")
        lines.append("Recommandation : passer Sprint 4 batch 4.2 en production (replacer "
                     "l'additif ConfluenceDetector par cet apprentissage).")
    else:
        lines.append("❌ **Modèle non prédictif OOS** : Brier skill ≤ +0.03. "
                     "Soit le sweep n'a pas généré assez de variété, soit les features "
                     "dérivées (score+hour+exit+holding) ne contiennent pas d'edge "
                     "exploitable. Action requise : persister les 8 composantes au "
                     "signal-time (refactor TradeRecord, Sprint 4 batch 4.2 vraie version).")

    lines.append("")
    lines.append(f"**Model file** : `models/scoring_v3_logistic_l1.pkl`")
    lines.append("")
    lines.append("**Reproducibility** : `python scripts/train_logistic_l1_on_sweep.py`")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
