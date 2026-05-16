"""Macro factor extraction — institutional-grade signal source.

Why this module exists
----------------------
Audit + user directive 2026-05-16 : "le plus haut niveau bancaire n'utilise
pas l'ICT". Banks (Bridgewater, AQR, Brevan Howard, Two Sigma) build
XAU/FX models on macro factors, not retail price patterns.

Empirical anchors for XAU price drivers (Baur & Lucey 2010, Erb & Harvey
2013, Reboredo 2013) :

- **Real rates 10Y** = DGS10 − BREAKEVEN_10Y. Negative corr ~−0.7 with XAU.
- **DXY (trade-weighted USD)** = DTWEXBGS. Negative corr ~−0.5 with XAU.
- **VIX / risk regime** = VIXCLS. Positive tail corr (safe haven).
- **Yield curve** = T10Y2Y. Recession proxy.
- **CoT Money Manager net** = positioning of speculators (cot_gold.csv).

Point-in-time discipline
------------------------
Every FRED CSV ships `vintage_date` (date data became public). The
factor extractor **only uses rows where ``vintage_date <= bar_timestamp``**
— so a backtest at 2020-03-15 never peeks at data that wasn't available
yet (e.g. delayed BEA releases revising backwards). This is the
non-negotiable institutional discipline that retail backtests skip.

Cf. ``audits/2026-Q2/`` for the position of this module in the new
post-ICT pipeline.
"""

from src.intelligence.macro_factors.fred_loader import (  # noqa: F401
    FREDSeriesLoader,
    load_all_xau_factors,
)
from src.intelligence.macro_factors.extractor import MacroFactorExtractor  # noqa: F401

__all__ = ["FREDSeriesLoader", "load_all_xau_factors", "MacroFactorExtractor"]
