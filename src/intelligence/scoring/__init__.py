"""Scoring layer — calibration & confidence (Sprint 4).

This package will hold the calibrated alternatives to the additive
ConfluenceDetector score :

- ``logistic_l1`` — multi-feature logistic regression with L1 regularisation
  on the 8 weighted_scores. Audit 3.3 finding : Pearson(score, R) = −0.008,
  Brier skill = −0.022 → isotonic regression alone cannot fix this
  (Spearman ≈ 0). The recommendation is logistic L1 over the 8 components
  (Sprint 4 batch 4.2).

- ``isotonic_recalibration`` — monotone post-hoc calibration to align
  bucket-level win rates with score deciles. Used **after** logistic L1
  finds a coordinate where the linear combination has non-zero Spearman.

Pre-Sprint 4 status : **scaffold only**. The actual implementations
require :

1. The 8 per-component continuous scores to be persisted at signal-time
   (currently only the final score is kept). Persistence required in
   ``ConfluenceDetector.analyze()`` as a side-effect.
2. Realised R-multiples per signal (from backtest replay).
3. A walk-forward calibration loop (refit every K trades).

These are Sprint 4 work items.
"""

from src.intelligence.scoring.logistic_l1 import LogisticL1Scorer  # noqa: F401
from src.intelligence.scoring.isotonic_recalibration import IsotonicRecalibrator  # noqa: F401

__all__ = ["LogisticL1Scorer", "IsotonicRecalibrator"]
