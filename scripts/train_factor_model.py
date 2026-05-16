"""Train institutional FactorModelPredictor on XAU M15 + macro + microstructure.

Pipeline:
1. Load XAU OHLCV 2019-2026 (172k bars).
2. Extract macro factors (PIT-safe FRED + CoT).
3. Extract microstructure proxies.
4. Build target = next-H1 log return.
5. Time-split 70/30 train/OOS.
6. Fit LightGBM regressor.
7. Report:
   - OOS R² (in returns)
   - Directional accuracy
   - Information ratio vs buy-and-hold
   - Feature importance ranking

This replaces the additive ConfluenceDetector scoring (Pearson −0.008)
with a multi-factor regressor on bank-grade features.

Usage:
    python scripts/train_factor_model.py [--quick]
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports" / "factor_model"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_xau() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "XAU_15MIN_2019_2026.csv", parse_dates=["Date"]).set_index("Date")
    df.rename(columns={c: c.capitalize() for c in df.columns
                       if c.lower() in {"open", "high", "low", "close", "volume"}}, inplace=True)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Last 50k bars only (smoke)")
    p.add_argument("--horizon", type=int, default=4, help="Forecast horizon in M15 bars (default 4 = next H1)")
    args = p.parse_args()

    from src.intelligence.macro_factors import MacroFactorExtractor
    from src.intelligence.microstructure import MicrostructureExtractor
    from src.intelligence.factor_model import FactorModelPredictor

    print("=== Loading XAU ===")
    ohlcv = load_xau()
    if args.quick:
        ohlcv = ohlcv.tail(50000)
    print(f"Loaded {len(ohlcv)} bars : {ohlcv.index[0]} → {ohlcv.index[-1]}")

    print("=== Extracting macro factors (PIT-safe) ===")
    macro_ext = MacroFactorExtractor()
    macro_df = macro_ext.extract(ohlcv.index)
    print(f"Macro features : {list(macro_df.columns)}")

    print("=== Extracting microstructure proxies ===")
    micro_ext = MicrostructureExtractor()
    micro_df = micro_ext.extract(ohlcv)
    print(f"Microstructure features : {list(micro_df.columns)}")

    print("=== Concatenating features ===")
    # Drop categorical strings (vix_regime) - keep code
    macro_numeric = macro_df.drop(columns=["vix_regime"], errors="ignore")
    feats = pd.concat([macro_numeric, micro_df], axis=1)
    feats = feats.fillna(0)
    print(f"Total feature matrix : {feats.shape}")

    print("=== Building target (next-H1 log return) ===")
    predictor = FactorModelPredictor(horizon_bars=args.horizon)
    target = predictor.build_target(ohlcv["Close"])
    print(f"Target : log(C_{{t+{args.horizon}}} / C_t)")
    print(f"Target std : {target.std():.6f}  mean : {target.mean():.6f}")

    # Time split 70/30
    n = len(feats)
    n_train = int(n * 0.7)
    X_tr = feats.iloc[:n_train]
    y_tr = target.iloc[:n_train]
    X_te = feats.iloc[n_train:]
    y_te = target.iloc[n_train:]
    print(f"Train : {len(X_tr)}, OOS : {len(X_te)}")

    print("=== Fitting LightGBM ===")
    predictor.fit(X_tr, y_tr)

    print("=== Evaluation OOS ===")
    pred_te = predictor.predict(X_te)
    mask = y_te.notna()
    pred_te_clean = pred_te[mask]
    y_te_clean = y_te[mask]

    # R²
    ss_res = float(((y_te_clean - pred_te_clean) ** 2).sum())
    ss_tot = float(((y_te_clean - y_te_clean.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Directional accuracy
    same_sign = (np.sign(pred_te_clean) == np.sign(y_te_clean))
    dir_acc = float(same_sign.mean())

    # Information ratio vs buy-and-hold
    # Simulate strategy : long if pred>+threshold, short if pred<-threshold
    # threshold = 0.3 * realized vol on rolling 96 bars
    bh_ret = y_te_clean  # passive long
    bh_sharpe = float(bh_ret.mean() / bh_ret.std() * np.sqrt(252 * 24)) if bh_ret.std() > 0 else 0.0

    strat_ret = np.sign(pred_te_clean) * y_te_clean
    strat_sharpe = float(strat_ret.mean() / strat_ret.std() * np.sqrt(252 * 24)) if strat_ret.std() > 0 else 0.0

    # IC (Spearman) — institutional reference
    from scipy.stats import spearmanr
    ic_pearson = float(np.corrcoef(pred_te_clean, y_te_clean)[0, 1])
    ic_spearman = float(spearmanr(pred_te_clean, y_te_clean).statistic)

    print(f"OOS R²              : {r2:+.6f}")
    print(f"OOS directional acc : {dir_acc:.4f}   (>0.52 = exploitable edge)")
    print(f"OOS IC Pearson      : {ic_pearson:+.6f}")
    print(f"OOS IC Spearman     : {ic_spearman:+.6f}")
    print(f"Buy&Hold Sharpe ann : {bh_sharpe:+.3f}")
    print(f"Strategy Sharpe ann : {strat_sharpe:+.3f}")
    print(f"Information Ratio   : {strat_sharpe - bh_sharpe:+.3f}")

    # Feature importance
    importance = predictor.feature_importance()
    print("\nTop 15 features by importance:")
    for k, v in sorted(importance.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k:30s}  {int(v)}")

    # Save model + report
    model_path = MODELS_DIR / "factor_model_v1.pkl"
    with model_path.open("wb") as f:
        pickle.dump(predictor, f)
    print(f"\nSaved : {model_path}")

    # Markdown report
    report = REPORTS_DIR / "training_report.md"
    lines = [
        "# Factor Model — institutional XAU predictor",
        "",
        f"**Date** : {pd.Timestamp.now().isoformat()}",
        f"**Mode** : {'QUICK (50k bars)' if args.quick else 'FULL'}",
        f"**Horizon** : {args.horizon} M15 bars = {args.horizon * 15} min",
        f"**Total bars** : {n}, train {len(X_tr)} / OOS {len(X_te)}",
        f"**Features** : {feats.shape[1]} (macro {macro_numeric.shape[1]} + microstructure {micro_df.shape[1]})",
        "",
        "## OOS Performance",
        "",
        "| Metric | Value | Verdict |",
        "| --- | --- | --- |",
        f"| OOS R² | {r2:+.6f} | {'✅ predictive' if r2 > 0 else '❌ noise'} |",
        f"| Directional accuracy | {dir_acc:.4f} | {'✅ exploitable' if dir_acc > 0.52 else '❌ not enough edge'} |",
        f"| IC Pearson | {ic_pearson:+.6f} | {'✅ signal' if abs(ic_pearson) > 0.02 else '❌ noise'} |",
        f"| IC Spearman | {ic_spearman:+.6f} | {'✅ signal' if abs(ic_spearman) > 0.02 else '❌ noise'} |",
        f"| Buy-and-Hold Sharpe (ann) | {bh_sharpe:+.3f} | baseline |",
        f"| Strategy Sharpe (ann) | {strat_sharpe:+.3f} | strat |",
        f"| **Information Ratio** | **{strat_sharpe - bh_sharpe:+.3f}** | {'✅ alpha' if (strat_sharpe - bh_sharpe) > 0.2 else '❌ no alpha vs BH'} |",
        "",
        "## Top 15 features (LightGBM importance)",
        "",
        "| Feature | Importance |",
        "| --- | --- |",
    ]
    for k, v in sorted(importance.items(), key=lambda x: -x[1])[:15]:
        lines.append(f"| `{k}` | {int(v)} |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if r2 > 0 and dir_acc > 0.52:
        lines.append("✅ **Modèle prédictif OOS**. Signal exploitable. Wire dans le scoring layer.")
    elif dir_acc > 0.51:
        lines.append("🟡 **Edge marginal**. Directional > 51% mais R² faible. Tester en walk-forward.")
    else:
        lines.append("❌ **Pas d'edge OOS** sur ces features. Possibles directions :")
        lines.append("- Ajouter features cross-asset (gold vs silver, gold vs SPX).")
        lines.append("- Régresser sur résidus de modèle macro (separate beta from alpha).")
        lines.append("- Walk-forward refit (les facteurs macro driftent — refit mensuel).")

    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
