"""Liquidity zones detector — swing high/low aggregation.

Reads recent fractal highs/lows from SmartMoneyEngine output and aggregates
them into liquidity zone bands (compact, mockup-friendly).

Liquidity zone = a cluster of swing extremes where stops are likely parked.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def detect_liquidity_zones(
    enriched_df: pd.DataFrame,
    lookback_bars: int = 200,
    cluster_atr_mult: float = 0.3,
) -> dict[str, Optional[tuple[float, float]]]:
    """Return ``{"upper": (low, high), "lower": (low, high)}`` zones.

    ``upper`` aggregates the swing highs above current price (buy-side liquidity).
    ``lower`` aggregates the swing lows below current price (sell-side liquidity).
    """
    if "UP_FRACTAL" not in enriched_df.columns or "DOWN_FRACTAL" not in enriched_df.columns:
        return {"upper": None, "lower": None}

    # SmartMoneyEngine emits lowercase OHLCV after analyze()
    close_col = "close" if "close" in enriched_df.columns else "Close"
    high_col = "high" if "high" in enriched_df.columns else "High"
    low_col = "low" if "low" in enriched_df.columns else "Low"
    if close_col not in enriched_df.columns:
        return {"upper": None, "lower": None}

    recent = enriched_df.tail(lookback_bars)
    if recent.empty:
        return {"upper": None, "lower": None}
    current_price = float(recent[close_col].iloc[-1])
    atr = float(recent.get("ATR", pd.Series([0.0])).iloc[-1] or 0.0)
    cluster_eps = max(atr * cluster_atr_mult, 1e-6)

    # Swing extremes are values where UP_FRACTAL == High or DOWN_FRACTAL == Low
    swing_highs = recent.loc[recent["UP_FRACTAL"].notna() & (recent["UP_FRACTAL"] > 0), high_col]
    swing_lows = recent.loc[recent["DOWN_FRACTAL"].notna() & (recent["DOWN_FRACTAL"] > 0), low_col]
    # Fallback : use raw fractal values
    if swing_highs.empty:
        swing_highs = recent.loc[recent["UP_FRACTAL"].fillna(0) > 0, "UP_FRACTAL"]
    if swing_lows.empty:
        swing_lows = recent.loc[recent["DOWN_FRACTAL"].fillna(0) > 0, "DOWN_FRACTAL"]

    upper_zone = lower_zone = None
    above = swing_highs[swing_highs > current_price].sort_values()
    below = swing_lows[swing_lows < current_price].sort_values(ascending=False)

    if not above.empty:
        seed = float(above.iloc[0])
        cluster = above[(above >= seed - cluster_eps) & (above <= seed + cluster_eps)]
        if not cluster.empty:
            upper_zone = (float(cluster.min()), float(cluster.max() + cluster_eps * 0.5))

    if not below.empty:
        seed = float(below.iloc[0])
        cluster = below[(below >= seed - cluster_eps) & (below <= seed + cluster_eps)]
        if not cluster.empty:
            lower_zone = (float(cluster.min() - cluster_eps * 0.5), float(cluster.max()))

    return {"upper": upper_zone, "lower": lower_zone}


__all__ = ["detect_liquidity_zones"]
