"""Train the calibrated_conviction pipeline on REAL backtest trades.

Adapter around :mod:`scripts.train_calibrated_conviction` that takes the
component-rich CSV produced by ``scripts/run_backtest.py`` (which writes
``cmp_*`` flat columns + ``components`` JSON) and reshapes it into the
schema expected by the trainer.

Why a separate adapter:
- ``train_calibrated_conviction.py`` expects ``score_bos / score_order_block / ...``
  column names with the 8 ConfluenceDetector composantes.
- ``run_backtest.py`` writes ``cmp_BOS / cmp_OrderBlock / cmp_FVG / cmp_Regime /
  cmp_News / cmp_Volume / cmp_Momentum / cmp_RSI_Divergence / cmp_HTF_Alignment``.
- We map :

    cmp_BOS                -> score_bos
    cmp_OrderBlock         -> score_order_block
    cmp_FVG                -> score_fvg
    cmp_Regime             -> score_regime
    cmp_News               -> score_news
    cmp_RSI_Divergence     -> score_momentum_rsi_div  (RSI divergence subsumes
                                                       directional momentum)
    -- (no retest column persisted) --   -> score_retest  := 0.0 (placeholder)
    -- (no vol_forecast persisted) --    -> score_vol_forecast := 0.0
    outcome := int(r_multiple > 0)
    pnl_r_multiple := r_multiple
    bar_timestamp := entry_bar

Usage
-----
::

    python scripts/train_calibrated_conviction_real.py \\
        --trades-csv reports/calibration/trades_xau_2019_2026.csv \\
        --output-pkl models/calibrated_conviction_v1.pkl \\
        --val-fraction 0.30 --alpha 0.10

The Brier skill is logged. Anything < 0 means the model is worse than
predicting the base rate — which is expected on the current strategy
(see reports/certification/ACTIONS_1_2_3_RESULTS.md) and the empirical
reality of the existing weak edge.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_calibrated_conviction import (  # noqa: E402
    FEATURE_COLUMNS,
    save_pipeline,
    train,
)

logger = logging.getLogger(__name__)


# Map cmp_* (from backtest CSV) → score_* (expected by trainer)
CMP_TO_SCORE = {
    "cmp_BOS": "score_bos",
    "cmp_OrderBlock": "score_order_block",
    "cmp_FVG": "score_fvg",
    "cmp_Regime": "score_regime",
    "cmp_News": "score_news",
    "cmp_RSI_Divergence": "score_momentum_rsi_div",
}


def load_and_reshape(csv_path: Path) -> pd.DataFrame:
    """Load a run_backtest.py trades CSV, reshape to trainer schema."""
    df = pd.read_csv(csv_path)

    missing_cmp = [c for c in CMP_TO_SCORE if c not in df.columns]
    if missing_cmp:
        raise ValueError(
            f"Trades CSV {csv_path} missing component columns: {missing_cmp}. "
            "Re-run scripts/run_backtest.py to capture cmp_* columns."
        )

    out = pd.DataFrame()
    out["signal_id"] = df["signal_id"]
    out["bar_timestamp"] = df["entry_bar"]
    for cmp_name, score_name in CMP_TO_SCORE.items():
        out[score_name] = pd.to_numeric(df[cmp_name], errors="coerce").fillna(0.0)

    # Placeholders for components not persisted in the current TradeRecord
    # (retest is implicit in OB+FVG gating; vol_forecast feeds tier but is
    # not surfaced as a separate component yet).
    out["score_retest"] = 0.0
    out["score_vol_forecast"] = 0.0

    # Sanity check — make sure all expected feature columns are present.
    missing = set(FEATURE_COLUMNS) - set(out.columns)
    if missing:
        raise RuntimeError(f"Reshape missed columns: {missing}")

    # Outcome and r-multiple
    if "r_multiple" not in df.columns:
        raise ValueError("Trades CSV missing 'r_multiple' column")
    out["pnl_r_multiple"] = pd.to_numeric(df["r_multiple"], errors="coerce")
    out["outcome"] = (out["pnl_r_multiple"] > 0).astype(int)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train calibrated_conviction on real trades CSV")
    parser.add_argument("--trades-csv", type=Path, required=True,
                        help="Path to a CSV produced by scripts/run_backtest.py "
                             "with cmp_* columns and r_multiple.")
    parser.add_argument("--output-pkl", type=Path,
                        default=Path("models/calibrated_conviction_v1.pkl"))
    parser.add_argument("--val-fraction", type=float, default=0.30)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--lgbm-leaves", type=int, default=15)
    parser.add_argument("--lgbm-lr", type=float, default=0.05)
    parser.add_argument("--lgbm-estimators", type=int, default=100)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Loading and reshaping %s", args.trades_csv)
    df = load_and_reshape(args.trades_csv)
    logger.info(
        "Reshape produced %d rows with %d wins (%.1f%% base rate)",
        len(df), int(df["outcome"].sum()), df["outcome"].mean() * 100,
    )

    if len(df) < 100:
        logger.warning(
            "Only %d trades — Brier skill estimates will be noisy. "
            "Consider increasing the backtest range or lowering thresholds.",
            len(df),
        )

    lgbm_kwargs = dict(
        num_leaves=args.lgbm_leaves,
        learning_rate=args.lgbm_lr,
        n_estimators=args.lgbm_estimators,
    )
    pipeline = train(
        df,
        val_fraction=args.val_fraction,
        alpha=args.alpha,
        lgbm_kwargs=lgbm_kwargs,
    )
    save_pipeline(pipeline, args.output_pkl)
    logger.info("Done. Pipeline saved to %s", args.output_pkl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
