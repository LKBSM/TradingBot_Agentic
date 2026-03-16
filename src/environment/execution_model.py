# =============================================================================
# EXECUTION MODEL — Dynamic Slippage & Spread (Sprint 10 & 11)
# =============================================================================
# Sprint 10: ATR-proportional slippage replacing fixed SLIPPAGE_PERCENTAGE.
# Sprint 11: Session-dependent spread replacing fixed TRANSACTION_FEE_PERCENTAGE.
#
# Usage:
#     slippage_model = DynamicSlippageModel(base_slippage=0.0001)
#     slippage = slippage_model.get_slippage(current_atr, median_atr)
#
#     spread_model = DynamicSpreadModel()
#     spread = spread_model.get_spread(hour_utc=10, is_news_window=False)
# =============================================================================

import numpy as np


class DynamicSlippageModel:
    """
    ATR-proportional slippage model.

    Slippage = base_slippage * max(1.0, (current_atr / median_atr) ^ atr_scale)

    - When ATR == median: slippage == base (normal conditions)
    - When ATR == 2x median: slippage == 2x base (elevated volatility)
    - When ATR == 5x median: slippage == 5x base (extreme volatility, e.g. NFP)
    - When ATR < median: slippage == base (floor at 1.0x, no bonus for calm)

    Args:
        base_slippage: Baseline slippage as a fraction (e.g., 0.0001 = 1 bp)
        atr_scale_factor: Exponent controlling sensitivity to ATR ratio.
            1.0 = linear scaling (recommended), 0.5 = sqrt (conservative)
    """

    def __init__(self, base_slippage: float = 0.0001, atr_scale_factor: float = 1.0):
        self.base = base_slippage
        self.atr_scale = atr_scale_factor

    def get_slippage(self, current_atr: float, median_atr: float = None) -> float:
        """
        Calculate dynamic slippage for the current bar.

        Args:
            current_atr: ATR value for the current bar
            median_atr: Median ATR across the dataset (computed once in _process_data)

        Returns:
            Slippage as a fraction (e.g., 0.0002 = 2 bps)
        """
        if median_atr is None or median_atr < 1e-10:
            return self.base

        if current_atr < 1e-10:
            return self.base

        atr_ratio = current_atr / median_atr
        # Floor at 1.0: slippage never below base even in calm markets
        return self.base * max(1.0, atr_ratio ** self.atr_scale)


class DynamicSpreadModel:
    """
    Session-dependent spread model for XAU/USD.

    Real-world Gold spreads vary by trading session due to liquidity depth:
    - London/NY overlap: tightest spreads (~3 bps)
    - Asian session: widest regular spreads (~8 bps)
    - Near high-impact news (NFP, FOMC, CPI): 3x wider

    Args:
        news_multiplier: Spread multiplier during news windows (default 3.0)
    """

    # Realistic XAU/USD spreads by session (fraction, not percentage)
    SESSION_SPREADS = {
        0: 0.0008,   # Asian (0:00-8:00 UTC) — thin liquidity
        1: 0.0003,   # London (8:00-13:00 UTC) — tightest
        2: 0.0003,   # London/NY overlap (13:00-17:00) — tightest
        3: 0.0005,   # NY afternoon (17:00-21:00) — moderate
        4: 0.0008,   # After-hours (21:00-0:00) — thin
    }

    def __init__(self, news_multiplier: float = 3.0):
        if news_multiplier < 1.0:
            raise ValueError(f"news_multiplier must be >= 1.0, got {news_multiplier}")
        self.news_multiplier = news_multiplier

    @staticmethod
    def _hour_to_session(hour_utc: int) -> int:
        """Map UTC hour (0-23) to session index."""
        if hour_utc < 8:
            return 0   # Asian
        elif hour_utc < 13:
            return 1   # London
        elif hour_utc < 17:
            return 2   # London/NY overlap
        elif hour_utc < 21:
            return 3   # NY afternoon
        else:
            return 4   # After-hours

    def get_spread(self, hour_utc: int, is_news_window: bool = False) -> float:
        """
        Calculate session-dependent spread for the current bar.

        Args:
            hour_utc: Hour of the bar in UTC (0-23)
            is_news_window: True if within high-impact news window

        Returns:
            Spread as a fraction (e.g., 0.0003 = 3 bps)
        """
        hour_utc = int(hour_utc) % 24
        session = self._hour_to_session(hour_utc)
        spread = self.SESSION_SPREADS[session]
        if is_news_window:
            spread *= self.news_multiplier
        return spread
