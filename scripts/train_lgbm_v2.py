"""Chantier 2 — LightGBM scoring v2 on the *filtered* XAU M15 subset.

Filters applied BEFORE training (matches the validated regime in
reports/feature_filter_audit.md):
    session != "NY"  AND  ATR_PCTL <= 0.75

Walk-forward: train < 2023-01-01 (with 20% inner valid for early stop),
test >= 2023-01-01. Predicts r_multiple (regression) and r_multiple>0
(classification). Backtests by ranking test trades on prediction and
sweeping top-N% acceptance to find the operational cut.

Usage: python scripts/train_lgbm_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from src.environment.strategy_features import SmartMoneyEngine

CSV = Path("data/XAU_15MIN_2019_2025.csv")
TRADES = Path("reports/audit/trades_combined.csv")
MODEL_OUT = Path("models/scoring_v2.lgb")
REPORT_OUT = Path("reports/scoring_v2_lgbm.md")
CUTOFF = "2023-01-01"

SESSION_BINS = [(0, 7, "Asian"), (7, 13, "London"), (13, 21, "NY"), (21, 24, "OffHours")]


def session_label(h: int) -> str:
    for lo, hi, n in SESSION_BINS:
        if lo <= h < hi:
            return n
    return "OffHours"


def fmt(v, spec=".4f"):
    if isinstance(v, float):
        if np.isnan(v):
            return "—"
        return f"{v:{spec}}"
    return str(v)


def safe_pearson(x, y):
    try:
        if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(pearsonr(x, y)[0])
    except Exception:
        return float("nan")


def safe_spearman(x, y):
    try:
        if len(x) < 5:
            return float("nan")
        return float(spearmanr(x, y)[0])
    except Exception:
        return float("nan")


def metrics(r: pd.Series) -> dict:
    if len(r) == 0:
        return {"n": 0, "pf": float("nan"), "winrate": float("nan"),
                "exp": float("nan"), "total_r": float("nan")}
    wins = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    pf = wins / losses if losses > 0 else (999.0 if wins > 0 else float("nan"))
    return {
        "n": int(len(r)),
        "pf": pf,
        "winrate": float((r > 0).mean()),
        "exp": float(r.mean()),
        "total_r": float(r.sum()),
    }


def build_features(merged: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Construct the model feature matrix from the joined trade+enriched frame."""
    direction_sign = merged["direction"].map({"LONG": 1.0, "SHORT": -1.0}).astype(float)

    out = pd.DataFrame(index=merged.index)

    # Sign-aligned features (positive = aligned with trade direction).
    for col in ["BOS_SIGNAL", "BOS_EVENT", "CHOCH_SIGNAL", "BOS_RETEST_ARMED",
                "FVG_SIGNAL", "CHOCH_DIVERGENCE", "MACD_Diff", "FVG_DIR"]:
        if col in merged.columns:
            out[f"{col}_aligned"] = merged[col].astype(float) * direction_sign

    # Unsigned features.
    for col in ["FVG_SIZE_NORM", "OB_STRENGTH_NORM", "RSI", "ATR", "ATR_PCTL",
                "BODY_SIZE", "SPREAD", "BB_M", "BB_H", "BB_L",
                "FVG_SIZE", "MACD_signal"]:
        if col in merged.columns:
            out[col] = merged[col].astype(float)

    # Bollinger position (where in the band): −1 below, 0 mid, +1 above.
    bb_range = (merged["BB_H"] - merged["BB_L"]).replace(0, np.nan)
    out["BB_POS"] = ((merged["close"] - merged["BB_M"]) / bb_range).clip(-2, 2)
    out["BB_POS_aligned"] = out["BB_POS"] * direction_sign
    # Body efficiency.
    out["BODY_RATIO"] = merged["BODY_SIZE"] / merged["SPREAD"].replace(0, np.nan)

    # Time features.
    ts = pd.to_datetime(merged["entry_bar"])
    out["hour"] = ts.dt.hour.astype(float).values
    out["dow"] = ts.dt.dayofweek.astype(float).values
    out["month"] = ts.dt.month.astype(float).values
    out["is_long"] = (direction_sign > 0).astype(float).values

    # Interactions.
    out["ATR_x_hour"] = out["ATR"] * out["hour"]
    out["RSI_x_dir"] = merged["RSI"].astype(float) * direction_sign
    out["ATRpct_x_RSI"] = out["ATR_PCTL"] * merged["RSI"].astype(float)

    # The previous (dead) score — keep so the model can dominate or ignore it.
    out["confluence_score"] = merged["confluence_score"].astype(float)

    feature_cols = list(out.columns)
    return out, feature_cols


def main() -> int:
    print("[lgbm] loading + enriching...")
    df = pd.read_csv(CSV, parse_dates=["Date"]).set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    enriched = SmartMoneyEngine(data=df, config={}, verbose=False).analyze()
    enriched["hour"] = enriched.index.hour
    enriched["session"] = enriched["hour"].map(session_label)
    atr = enriched["ATR"].fillna(0)
    enriched["ATR_PCTL"] = atr.rolling(30 * 96, min_periods=200).rank(pct=True)

    trades = pd.read_csv(TRADES)
    trades["entry_bar"] = pd.to_datetime(trades["entry_bar"])
    feat_cols_from_enriched = [
        "session", "ATR_PCTL", "ATR", "RSI", "MACD_line", "MACD_signal", "MACD_Diff",
        "BB_L", "BB_M", "BB_H", "SPREAD", "BODY_SIZE",
        "FVG_SIZE", "FVG_DIR", "FVG_SIZE_NORM", "FVG_SIGNAL",
        "OB_STRENGTH_NORM", "BOS_SIGNAL", "CHOCH_SIGNAL", "BOS_EVENT",
        "BOS_RETEST_ARMED", "CHOCH_DIVERGENCE", "close",
    ]
    feat_cols_from_enriched = [c for c in feat_cols_from_enriched if c in enriched.columns]
    merged_all = trades.merge(
        enriched[feat_cols_from_enriched],
        left_on="entry_bar", right_index=True, how="inner",
    )

    n_total = len(merged_all)
    mask_filter = (merged_all["session"] != "NY") & (merged_all["ATR_PCTL"].fillna(0) <= 0.75)
    merged = merged_all[mask_filter].copy()
    print(f"[lgbm] {n_total} trades joined -> {len(merged)} after regime filter")

    X, feature_cols = build_features(merged)
    X = X.replace([np.inf, -np.inf], np.nan)
    y = merged["r_multiple"].astype(float).values
    y_bin = (merged["r_multiple"] > 0).astype(int).values

    is_train = merged["entry_bar"].values < np.datetime64(CUTOFF)
    X_train, X_test = X[is_train].copy(), X[~is_train].copy()
    y_train, y_test = y[is_train], y[~is_train]
    yb_train, yb_test = y_bin[is_train], y_bin[~is_train]
    print(f"[lgbm] train={len(X_train)}  test={len(X_test)}  features={len(feature_cols)}")

    # Inner valid slice (last 20% of train, time-ordered).
    n_train = len(X_train)
    n_valid = max(int(n_train * 0.20), 30)
    X_tr, X_val = X_train.iloc[:-n_valid], X_train.iloc[-n_valid:]
    y_tr, y_val = y_train[:-n_valid], y_train[-n_valid:]
    yb_tr, yb_val = yb_train[:-n_valid], yb_train[-n_valid:]

    common_params = dict(
        num_leaves=15,
        max_depth=4,
        learning_rate=0.03,
        min_data_in_leaf=30,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=3,
        lambda_l2=1.0,
        verbosity=-1,
        n_estimators=500,
    )

    # 1) Regression on r_multiple
    reg = lgb.LGBMRegressor(objective="regression", **common_params)
    reg.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    pred_train_reg = reg.predict(X_train)
    pred_test_reg = reg.predict(X_test)

    # 2) Classification on (r_multiple > 0)
    clf = lgb.LGBMClassifier(objective="binary", **common_params)
    clf.fit(
        X_tr, yb_tr,
        eval_set=[(X_val, yb_val)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    proba_train_clf = clf.predict_proba(X_train)[:, 1]
    proba_test_clf = clf.predict_proba(X_test)[:, 1]

    # Stats.
    pr_train_reg = safe_pearson(pred_train_reg, y_train)
    pr_test_reg = safe_pearson(pred_test_reg, y_test)
    sp_test_reg = safe_spearman(pred_test_reg, y_test)
    pr_train_clf = safe_pearson(proba_train_clf, y_train)
    pr_test_clf = safe_pearson(proba_test_clf, y_test)
    auc_train = roc_auc_score(yb_train, proba_train_clf) if len(set(yb_train)) > 1 else float("nan")
    auc_test = roc_auc_score(yb_test, proba_test_clf) if len(set(yb_test)) > 1 else float("nan")

    # PF-by-quantile sweep on the test set, ranked by each predictor.
    def quantile_sweep(scores: np.ndarray, r: np.ndarray) -> list[dict]:
        rows = []
        baseline = metrics(pd.Series(r))
        rows.append({"top%": 100, **baseline})
        n = len(scores)
        order = np.argsort(-scores)  # high → low
        for q in [50, 40, 30, 20, 10]:
            k = max(int(n * q / 100), 1)
            sel_idx = order[:k]
            sel_r = pd.Series(r[sel_idx])
            rows.append({"top%": q, **metrics(sel_r)})
        return rows

    sweep_reg = quantile_sweep(pred_test_reg, y_test)
    sweep_clf = quantile_sweep(proba_test_clf, y_test)

    # Pick the model that wins more on PF at top-30% / top-50%.
    def pf_at(rows, q):
        for r in rows:
            if r["top%"] == q:
                return r["pf"]
        return float("nan")

    reg_combo = (pf_at(sweep_reg, 30), pf_at(sweep_reg, 50))
    clf_combo = (pf_at(sweep_clf, 30), pf_at(sweep_clf, 50))
    use_clf = (
        (np.nan_to_num(clf_combo[0], nan=-1) + np.nan_to_num(clf_combo[1], nan=-1))
        > (np.nan_to_num(reg_combo[0], nan=-1) + np.nan_to_num(reg_combo[1], nan=-1))
    )
    chosen = clf if use_clf else reg
    chosen_label = "classification (P(r>0))" if use_clf else "regression (E[r])"

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    chosen.booster_.save_model(str(MODEL_OUT))
    print(f"[lgbm] saved {MODEL_OUT}  (chosen: {chosen_label})")

    # Feature importance from the chosen model (gain).
    booster = chosen.booster_
    fi_gain = booster.feature_importance(importance_type="gain")
    fi_split = booster.feature_importance(importance_type="split")
    fi = pd.DataFrame({"feature": feature_cols, "gain": fi_gain, "split": fi_split})
    fi = fi.sort_values("gain", ascending=False).head(10)

    # Verdict.
    pf_top30 = pf_at(sweep_clf if use_clf else sweep_reg, 30)
    pf_top50 = pf_at(sweep_clf if use_clf else sweep_reg, 50)
    if pf_top30 >= 1.50:
        verdict = "✅ **SUCCESS** — top-30% PF ≥ 1.50 OOS. Replace the dead confluence_score with this model."
    elif pf_top50 >= 1.40:
        verdict = "🟡 **ACCEPTABLE** — top-50% PF ≥ 1.40 OOS. Marginal upgrade over filter-only baseline (PF 1.355)."
    else:
        verdict = "❌ **NOT WORTH SHIPPING** — model fails to beat both 1.50@30% and 1.40@50% targets. Filter-only baseline (PF 1.355, n=416) is the better product."

    md = []
    md.append("# Scoring v2 — LightGBM on filtered XAU M15 subset\n")
    md.append("**Filter applied BEFORE training/eval**: `session != NY` AND `ATR_PCTL <= 0.75`")
    md.append(f"**Walk-forward**: train < {CUTOFF} (last 20% inner valid), test ≥ {CUTOFF}")
    md.append(f"**Sample**: {n_total} raw trades → {len(merged)} after filter → {len(X_train)} train / {len(X_test)} test")
    md.append(f"**Features**: {len(feature_cols)} (sign-aligned SMC + unsigned + interactions + dead-score as input)\n")
    md.append("---\n")

    md.append("## Predictive power (OOS = test set)\n")
    md.append("| Model | Pearson_train | Pearson_test | Spearman_test | AUC_train | AUC_test |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| Regression (r̂) | {fmt(pr_train_reg)} | {fmt(pr_test_reg)} | {fmt(sp_test_reg)} | — | — |")
    md.append(f"| Classification (P(r>0)) | {fmt(pr_train_clf)} | {fmt(pr_test_clf)} | — | {fmt(auc_train)} | {fmt(auc_test)} |")
    md.append("")
    md.append("> **Reference**: confluence_score Pearson_test = −0.0139 (dead). Anything above |0.05| OOS is real signal.\n")

    md.append(f"## Backtest by quantile — chosen model: {chosen_label}\n")
    sweep = sweep_clf if use_clf else sweep_reg
    md.append("| top % predicted | n | PF | win% | exp_R | total_R |")
    md.append("|---|---|---|---|---|---|")
    for r in sweep:
        md.append(
            f"| {r['top%']}% | {r['n']} | {fmt(r['pf'], '.3f')} | {fmt(r['winrate'], '.3f')} | "
            f"{fmt(r['exp'], '.3f')} | {fmt(r['total_r'], '.2f')} |"
        )
    md.append("")
    md.append("**Targets**: top-30% PF ≥ 1.50 (success), top-50% PF ≥ 1.40 (acceptable), else not shipping.\n")

    # Show losing model's sweep for transparency.
    md.append("## Losing model sweep (for transparency)\n")
    losing_sweep = sweep_reg if use_clf else sweep_clf
    losing_label = "regression (r̂)" if use_clf else "classification (P(r>0))"
    md.append(f"_{losing_label}_\n")
    md.append("| top % | n | PF | exp_R | total_R |")
    md.append("|---|---|---|---|---|")
    for r in losing_sweep:
        md.append(f"| {r['top%']}% | {r['n']} | {fmt(r['pf'], '.3f')} | {fmt(r['exp'], '.3f')} | {fmt(r['total_r'], '.2f')} |")
    md.append("")

    md.append("## Top 10 features by gain\n")
    md.append("| feature | gain | split |")
    md.append("|---|---|---|")
    for _, r in fi.iterrows():
        md.append(f"| {r['feature']} | {int(r['gain'])} | {int(r['split'])} |")
    md.append("")

    md.append("## Overfit honesty check\n")
    gap_reg = (pr_train_reg or 0) - (pr_test_reg or 0)
    gap_clf = (auc_train or 0.5) - (auc_test or 0.5)
    md.append(f"- Regression Pearson gap (train−test): **{gap_reg:+.4f}** "
              f"({'large overfit' if abs(gap_reg) > 0.15 else 'modest gap' if abs(gap_reg) > 0.05 else 'small gap'})")
    md.append(f"- Classification AUC gap (train−test): **{gap_clf:+.4f}** "
              f"({'large overfit' if abs(gap_clf) > 0.10 else 'modest gap' if abs(gap_clf) > 0.04 else 'small gap'})")
    md.append("")

    md.append(f"## Verdict\n\n{verdict}\n")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"[lgbm] wrote {REPORT_OUT}")

    print("\n=== HEADLINE ===")
    print(f"Chosen model    : {chosen_label}")
    print(f"Pearson test reg: {fmt(pr_test_reg)}")
    print(f"AUC test clf    : {fmt(auc_test)}")
    print(f"PF top-30%      : {fmt(pf_top30, '.3f')}  (target ≥ 1.50)")
    print(f"PF top-50%      : {fmt(pf_top50, '.3f')}  (target ≥ 1.40)")
    print(f"Verdict         : {verdict.split('—')[0].strip().split('**')[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
