"""Multi-factor predictive model — institutional XAU/FX engine.

Pipeline complet :

OHLCV M15 ─┬─► MacroFactorExtractor    (real_rates, DXY, VIX, CoT — 12 features)
           ├─► MicrostructureExtractor (Roll spread, GK vol, RV/session — 13 features)
           └─► SmartMoneyEngine ───────► ICT residuals  (2-3 features, conditioning gate)
                                                                          │
                                                  ───────────► FactorModel.predict(features)
                                                                          │
                                                  ► expected_return ∈ ℝ + confidence

The signal :
    if expected_return > +threshold_atr × ATR  →  LONG
    elif expected_return < −threshold_atr × ATR  →  SHORT
    else HOLD

Where threshold_atr is calibrated via CPCV (typically 0.3-0.5 ATR).

References to bank-level practice :
- AQR (Asness, Frazzini) : factor zoo + sparsity (L1 / boosting).
- Bridgewater : real rates + risk-off regime conditioning.
- Two Sigma : ML on factor residuals (after removing macro beta).
- Renaissance : ensemble of weak signals, none individually predictive.
"""

from src.intelligence.factor_model.predictor import FactorModelPredictor  # noqa: F401

__all__ = ["FactorModelPredictor"]
