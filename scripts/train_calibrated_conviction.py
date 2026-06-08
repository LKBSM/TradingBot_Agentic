"""Train the LGBM → Isotonic → ACI calibrated conviction pipeline.

Usage
-----
.. code:: bash

    python scripts/train_calibrated_conviction.py \\
        --replay-csv reports/replay_xau_m15.csv \\
        --output-pkl models/calibrated_conviction_v1.pkl \\
        --alpha 0.10 \\
        --feature-window 8 \\
        --val-fraction 0.30

What this script does
---------------------
1. Loads a trade replay CSV with the following columns (case-sensitive):
     - ``signal_id``                  identifier of the candidate signal
     - ``bar_timestamp``              ISO-8601 of the bar
     - ``score_bos``                  weighted contribution of BOS (0-15)
     - ``score_fvg``                  weighted contribution of FVG (0-15)
     - ``score_order_block``          weighted contribution of OB (0-10)
     - ``score_regime``               weighted contribution of regime (0-25)
     - ``score_news``                 weighted contribution of news (0-20)
     - ``score_volume``               weighted contribution of volume (0-10)
     - ``score_momentum``             weighted contribution of momentum (0-3)
     - ``score_rsi_divergence``       weighted contribution of RSI div (0-2)
     - ``outcome``                    1 for win, 0 for loss (target)
     - ``pnl_r_multiple`` (optional)  R-multiple realized PnL — used by ACI
2. Splits chronologically into train / validation.
3. Fits ``LGBMScorer`` on train.
4. Generates OOF predictions on val.
5. Fits ``IsotonicRecalibrator`` (val predictions → val outcomes).
6. Fits ``AdaptiveConformalScorer`` on val ``pnl_r_multiple``.
7. Pickles a ``CalibratedConvictionPipeline`` ready to load at scanner
   startup.

Synthetic mode
--------------
If ``--synthetic`` is passed, the script generates a synthetic dataset
(useful for smoke-testing the pipeline end-to-end without real data).

Output artefact
---------------
A single pickle containing ``CalibratedConvictionPipeline`` with the
three stages fitted. The scanner loads it via
:func:`load_calibrated_pipeline` (see bottom of this file).
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root in path when invoked as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intelligence.conformal_wrapper import AdaptiveConformalScorer  # noqa: E402
from src.intelligence.scoring.calibrated_conviction import (  # noqa: E402
    CalibratedConvictionPipeline,
)
from src.intelligence.scoring.isotonic_recalibration import IsotonicRecalibrator  # noqa: E402
from src.intelligence.scoring.lgbm_scorer import (  # noqa: E402
    DEFAULT_FEATURE_NAMES,
    LGBMScorer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature schema — the 8 column names expected in the replay CSV
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = (
    "score_bos",
    "score_order_block",
    "score_fvg",
    "score_retest",
    "score_regime",
    "score_vol_forecast",
    "score_news",
    "score_momentum_rsi_div",
)
# Map them onto the 8 default LGBM feature names (must align with
# DEFAULT_FEATURE_NAMES from scoring/lgbm_scorer.py).
FEATURE_NAME_MAP = dict(zip(FEATURE_COLUMNS, DEFAULT_FEATURE_NAMES))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_replay(csv_path: Path) -> pd.DataFrame:
    """Load and validate a replay CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Replay CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        # Tolerate the legacy "score_*" naming or fall back to defaults
        logger.warning(
            "Missing %d expected columns: %s. Filling with zeros (degraded).",
            len(missing), sorted(missing),
        )
        for col in missing:
            df[col] = 0.0
    if "outcome" not in df.columns:
        raise ValueError("Replay CSV must include 'outcome' column (0/1 labels)")
    # Sort chronologically if possible
    if "bar_timestamp" in df.columns:
        df = df.sort_values("bar_timestamp").reset_index(drop=True)
    return df


def synthetic_replay(n_signals: int = 2000, win_rate: float = 0.40,
                     seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic synthetic replay for smoke-testing.

    Features are correlated with outcome — the synthetic dataset has a real
    (if simple) signal, so the LGBM should actually learn something.
    """
    rng = np.random.default_rng(seed)
    n = int(n_signals)

    # Bias features so high-feature signals are more likely to win
    base = rng.uniform(0.0, 1.0, size=(n, 8))
    weights = np.array([0.15, 0.10, 0.15, 0.10, 0.20, 0.15, 0.05, 0.10])
    raw_signal = base @ weights + rng.normal(0, 0.15, size=n)
    target_logit = raw_signal - np.quantile(raw_signal, 1 - win_rate)
    p_win = 1.0 / (1.0 + np.exp(-target_logit * 5))
    outcome = (rng.uniform(0, 1, size=n) < p_win).astype(int)

    # Scale features to typical 0-15 range
    feature_max = np.array([15.0, 10.0, 15.0, 5.0, 25.0, 10.0, 20.0, 5.0])
    scaled = base * feature_max

    pnl = np.where(outcome == 1, rng.normal(1.0, 0.3, size=n), rng.normal(-0.7, 0.2, size=n))
    pnl = np.clip(pnl, -1.5, 3.0)

    df = pd.DataFrame(scaled, columns=list(FEATURE_COLUMNS))
    df["outcome"] = outcome
    df["pnl_r_multiple"] = pnl
    df["bar_timestamp"] = pd.date_range("2019-01-01", periods=n, freq="15min")
    df["signal_id"] = [f"syn_{i:06d}" for i in range(n)]
    return df


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------


def split_chronological(df: pd.DataFrame, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/val split — first (1-val_fraction) is train."""
    if not 0.05 <= val_fraction <= 0.50:
        raise ValueError(f"val_fraction must be in [0.05, 0.50], got {val_fraction}")
    n = len(df)
    cut = int(n * (1.0 - val_fraction))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def train(
    df: pd.DataFrame,
    val_fraction: float = 0.30,
    alpha: float = 0.10,
    lgbm_kwargs: Optional[dict] = None,
) -> CalibratedConvictionPipeline:
    """Train all three stages and return a ready-to-use pipeline."""
    train_df, val_df = split_chronological(df, val_fraction)
    if len(train_df) < 50 or len(val_df) < 30:
        raise ValueError(
            f"Insufficient data: train={len(train_df)}, val={len(val_df)} "
            f"(need ≥50 train and ≥30 val)"
        )

    X_train = train_df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y_train = train_df["outcome"].to_numpy(dtype=int)
    X_val = val_df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y_val = val_df["outcome"].to_numpy(dtype=int)

    # Stage 1: LGBM
    lgbm = LGBMScorer(**(lgbm_kwargs or {}))
    logger.info("Fitting LGBMScorer on %d train rows…", len(train_df))
    lgbm.fit(X_train, y_train)
    p_val = lgbm.predict_p_win(X_val)

    # Stage 2: Isotonic on val predictions
    logger.info("Fitting IsotonicRecalibrator on %d val predictions…", len(val_df))
    isotonic = IsotonicRecalibrator(increasing=True)
    isotonic.fit(p_val, y_val)

    # Stage 3: ACI on val pnl (if available, else on outcomes themselves)
    aci = AdaptiveConformalScorer(alpha_target=alpha)
    if "pnl_r_multiple" in val_df.columns and val_df["pnl_r_multiple"].notna().sum() >= 30:
        outcomes = val_df["pnl_r_multiple"].dropna().to_numpy(dtype=float)
        logger.info("Fitting AdaptiveConformalScorer on %d R-multiples…", len(outcomes))
    else:
        outcomes = isotonic.transform(p_val).astype(float)
        logger.info(
            "No pnl_r_multiple available — fitting ACI on calibrated probabilities "
            "(degraded but functional)."
        )
    aci.fit(outcomes)

    pipeline = CalibratedConvictionPipeline(lgbm=lgbm, isotonic=isotonic, conformal=aci)

    # Quick validation metrics — purely informational
    p_train = lgbm.predict_p_win(X_train)
    train_acc = float(((p_train >= 0.5) == y_train).mean())
    val_acc = float(((p_val >= 0.5) == y_val).mean())
    val_brier = float(np.mean((p_val - y_val) ** 2))
    naive_brier = float(np.mean((y_val.mean() - y_val) ** 2))
    logger.info(
        "Training summary — train_acc=%.3f val_acc=%.3f val_brier=%.4f naive_brier=%.4f "
        "Brier skill=%.3f",
        train_acc, val_acc, val_brier, naive_brier,
        1.0 - val_brier / naive_brier if naive_brier > 0 else 0.0,
    )
    return pipeline


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_pipeline(pipeline: CalibratedConvictionPipeline, output_path: Path) -> None:
    """Pickle the fitted pipeline to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Wrote pipeline → %s (%d bytes)", output_path, output_path.stat().st_size)


def load_calibrated_pipeline(path: Path) -> CalibratedConvictionPipeline:
    """Load a previously pickled pipeline. Called from SentinelScanner.

    Returns
    -------
    CalibratedConvictionPipeline
        Ready for ``score_one()`` calls. If loading fails, returns an
        empty unfitted pipeline (which produces fallback CalibratedConviction).
    """
    try:
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        if not isinstance(pipeline, CalibratedConvictionPipeline):
            raise TypeError(f"Loaded object is not CalibratedConvictionPipeline: {type(pipeline)}")
        logger.info("Loaded calibrated pipeline from %s", path)
        return pipeline
    except Exception as exc:
        logger.warning(
            "Could not load calibrated pipeline from %s (%s) — using unfitted fallback.",
            path, exc,
        )
        return CalibratedConvictionPipeline()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train LGBM→Isotonic→ACI calibrated conviction pipeline.")
    parser.add_argument("--replay-csv", type=Path, default=None,
                        help="Path to a replay CSV (omit with --synthetic).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a synthetic replay (smoke test).")
    parser.add_argument("--synthetic-n", type=int, default=2000,
                        help="Number of rows when --synthetic (default 2000).")
    parser.add_argument("--output-pkl", type=Path, default=Path("models/calibrated_conviction_v1.pkl"),
                        help="Where to write the pickled pipeline.")
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="Nominal miscoverage for conformal interval (default 0.10 ⇒ 90% interval).")
    parser.add_argument("--val-fraction", type=float, default=0.30,
                        help="Chronological val fraction (default 0.30).")
    parser.add_argument("--lgbm-leaves", type=int, default=31)
    parser.add_argument("--lgbm-lr", type=float, default=0.05)
    parser.add_argument("--lgbm-estimators", type=int, default=200)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.synthetic:
        logger.info("Generating synthetic replay (n=%d)…", args.synthetic_n)
        df = synthetic_replay(n_signals=args.synthetic_n)
    elif args.replay_csv is None:
        parser.error("Either --replay-csv or --synthetic is required.")
        return 2
    else:
        df = load_replay(args.replay_csv)
        logger.info("Loaded %d signals from %s", len(df), args.replay_csv)

    pipeline = train(
        df,
        val_fraction=args.val_fraction,
        alpha=args.alpha,
        lgbm_kwargs={
            "num_leaves": args.lgbm_leaves,
            "learning_rate": args.lgbm_lr,
            "n_estimators": args.lgbm_estimators,
        },
    )
    save_pipeline(pipeline, args.output_pkl)

    # Sanity-check: load + score one
    reloaded = load_calibrated_pipeline(args.output_pkl)
    sample_features = df[list(FEATURE_COLUMNS)].iloc[-1].to_numpy(dtype=float)
    cc = reloaded.score_one(sample_features)
    logger.info(
        "Smoke check — conviction_0_100=%d, interval=[%.1f, %.1f], is_fallback=%s",
        cc.conviction_0_100,
        cc.conformal_lower_0_100, cc.conformal_upper_0_100,
        cc.is_fallback,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
