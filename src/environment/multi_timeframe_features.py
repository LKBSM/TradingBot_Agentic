# =============================================================================
# MULTI-TIMEFRAME FEATURES - Higher Timeframe Context for RL Agent
# =============================================================================
# This module provides multi-timeframe analysis features that are CRITICAL
# for profitable trading. The RL agent needs to understand:
#   - Is the higher timeframe trending or ranging?
#   - Am I trading with or against the trend?
#   - Where are key support/resistance levels on higher timeframes?
#
# === WHY THIS MATTERS ===
# - 15-min noise vs. 4H trend: The 15-min chart has lots of noise
# - Counter-trend trades fail: Going against the 4H trend is dangerous
# - Key levels from higher TFs: Major support/resistance are more reliable
#
# === FEATURES PROVIDED ===
# 1. HTF_TREND_1H:    1-hour trend direction (-1, 0, 1)
# 2. HTF_TREND_4H:    4-hour trend direction (-1, 0, 1)
# 3. HTF_STRENGTH_1H: 1-hour trend strength (0-1)
# 4. HTF_STRENGTH_4H: 4-hour trend strength (0-1)
# 5. PRICE_VS_HTF_MA: Price relative to higher TF moving averages
# 6. HTF_RSI:         Higher timeframe RSI (smoother momentum)
# 7. SESSION:         Trading session (0=Asian, 1=London, 2=NY)
# 8. DAY_OF_WEEK:     Day of week (0=Mon to 4=Fri)
#
# === USAGE ===
#   mtf = MultiTimeframeFeatures(base_timeframe='15min')
#   mtf.fit(df)  # Initialize with historical data
#
#   # Get features for current bar
#   features = mtf.get_features(current_idx)
#
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from collections import deque


class MultiTimeframeFeatures:
    """
    Calculate multi-timeframe features for the RL agent.

    This class provides higher timeframe context to the 15-minute
    trading decisions, which is essential for profitability.
    """

    def __init__(
        self,
        base_timeframe: str = '15min',
        include_1h: bool = True,
        include_4h: bool = True,
        include_session: bool = True
    ):
        """
        Initialize multi-timeframe feature calculator.

        Args:
            base_timeframe: Base data timeframe (e.g., '15min', '1h')
            include_1h: Include 1-hour timeframe features
            include_4h: Include 4-hour timeframe features
            include_session: Include trading session info
        """
        self.base_timeframe = base_timeframe
        self.include_1h = include_1h
        self.include_4h = include_4h
        self.include_session = include_session

        # Resample ratios (15min base)
        self._resample_ratios = {
            '15min': {
                '1h': 4,    # 4 x 15min = 1 hour
                '4h': 16    # 16 x 15min = 4 hours
            }
        }

        # Storage for resampled data
        self._df: Optional[pd.DataFrame] = None
        self._df_1h: Optional[pd.DataFrame] = None
        self._df_4h: Optional[pd.DataFrame] = None

        # Feature names
        self._feature_names: List[str] = []

    def fit(self, df: pd.DataFrame) -> 'MultiTimeframeFeatures':
        """
        Fit the feature calculator with historical data.

        This creates the higher timeframe OHLCV data and calculates
        indicators for each timeframe.

        Args:
            df: DataFrame with OHLCV data (must have Date/datetime index or column)

        Returns:
            self for chaining
        """
        self._df = df.copy()

        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                self._df['Date'] = pd.to_datetime(df['Date'])
                self._df.set_index('Date', inplace=True)
            elif 'Original_Timestamp' in df.columns:
                self._df['Date'] = pd.to_datetime(df['Original_Timestamp'])
                self._df.set_index('Date', inplace=True)
            else:
                # Create synthetic timestamps if none exist
                self._df.index = pd.date_range(
                    start='2020-01-01',
                    periods=len(df),
                    freq='15min'
                )

        # Resample to higher timeframes
        if self.include_1h:
            self._df_1h = self._resample_ohlcv(self._df, '1h')
            self._calculate_indicators(self._df_1h)

        if self.include_4h:
            self._df_4h = self._resample_ohlcv(self._df, '4h')
            self._calculate_indicators(self._df_4h)

        # Build feature names list
        self._build_feature_names()

        return self

    def _resample_ohlcv(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to higher timeframe."""
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }

        # Only include columns that exist
        agg_dict = {k: v for k, v in ohlc_dict.items() if k in df.columns}

        resampled = df.resample(timeframe).agg(agg_dict)
        resampled.dropna(inplace=True)

        return resampled

    def _calculate_indicators(self, df: pd.DataFrame) -> None:
        """Calculate technical indicators for a timeframe."""
        if len(df) < 50:
            return

        # SMA 20 and 50
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        # Trend direction: SMA_20 vs SMA_50
        df['TREND'] = np.where(
            df['SMA_20'] > df['SMA_50'], 1,
            np.where(df['SMA_20'] < df['SMA_50'], -1, 0)
        )

        # Trend strength: distance between MAs as % of price
        df['TREND_STRENGTH'] = np.abs(df['SMA_20'] - df['SMA_50']) / df['Close']
        df['TREND_STRENGTH'] = df['TREND_STRENGTH'].clip(0, 0.1) * 10  # Normalize to 0-1

        # ATR for volatility context
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # Fill NaN with forward fill then backward fill
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    def _build_feature_names(self) -> None:
        """Build list of feature names."""
        self._feature_names = []

        if self.include_1h:
            self._feature_names.extend([
                'HTF_TREND_1H',
                'HTF_STRENGTH_1H',
                'HTF_RSI_1H',
                'PRICE_VS_SMA20_1H',
                'PRICE_VS_SMA50_1H'
            ])

        if self.include_4h:
            self._feature_names.extend([
                'HTF_TREND_4H',
                'HTF_STRENGTH_4H',
                'HTF_RSI_4H',
                'PRICE_VS_SMA20_4H',
                'PRICE_VS_SMA50_4H'
            ])

        if self.include_session:
            self._feature_names.extend([
                'SESSION',
                'DAY_OF_WEEK',
                'HOUR_SIN',
                'HOUR_COS'
            ])

    def get_features(self, idx: int, current_price: Optional[float] = None) -> Dict[str, float]:
        """
        Get multi-timeframe features for a specific bar index.

        Args:
            idx: Index in the base dataframe
            current_price: Current close price (optional, uses df if not provided)

        Returns:
            Dictionary of feature name -> value
        """
        if self._df is None:
            raise ValueError("Must call fit() before get_features()")

        features = {}

        # Get current timestamp and price
        current_ts = self._df.index[idx]
        if current_price is None:
            current_price = self._df.iloc[idx]['Close']

        # 1H features
        if self.include_1h and self._df_1h is not None:
            htf_features = self._get_htf_features(
                current_ts, current_price, self._df_1h, '1H'
            )
            features.update(htf_features)

        # 4H features
        if self.include_4h and self._df_4h is not None:
            htf_features = self._get_htf_features(
                current_ts, current_price, self._df_4h, '4H'
            )
            features.update(htf_features)

        # Session features
        if self.include_session:
            session_features = self._get_session_features(current_ts)
            features.update(session_features)

        return features

    def _get_htf_features(
        self,
        current_ts: pd.Timestamp,
        current_price: float,
        htf_df: pd.DataFrame,
        suffix: str
    ) -> Dict[str, float]:
        """Get features from a higher timeframe."""
        features = {}

        # Find the most recent HTF bar
        mask = htf_df.index <= current_ts
        if not mask.any():
            # No HTF data available yet
            return {
                f'HTF_TREND_{suffix}': 0.0,
                f'HTF_STRENGTH_{suffix}': 0.0,
                f'HTF_RSI_{suffix}': 0.5,
                f'PRICE_VS_SMA20_{suffix}': 0.0,
                f'PRICE_VS_SMA50_{suffix}': 0.0
            }

        htf_bar = htf_df.loc[mask].iloc[-1]

        # Trend direction
        features[f'HTF_TREND_{suffix}'] = float(htf_bar.get('TREND', 0))

        # Trend strength (normalized 0-1)
        features[f'HTF_STRENGTH_{suffix}'] = float(htf_bar.get('TREND_STRENGTH', 0))

        # RSI (normalized to 0-1)
        features[f'HTF_RSI_{suffix}'] = float(htf_bar.get('RSI', 50)) / 100.0

        # Price vs higher TF MAs (normalized)
        sma20 = htf_bar.get('SMA_20', current_price)
        sma50 = htf_bar.get('SMA_50', current_price)

        if sma20 > 0:
            features[f'PRICE_VS_SMA20_{suffix}'] = np.clip(
                (current_price - sma20) / sma20 * 10, -1, 1
            )
        else:
            features[f'PRICE_VS_SMA20_{suffix}'] = 0.0

        if sma50 > 0:
            features[f'PRICE_VS_SMA50_{suffix}'] = np.clip(
                (current_price - sma50) / sma50 * 10, -1, 1
            )
        else:
            features[f'PRICE_VS_SMA50_{suffix}'] = 0.0

        return features

    def _get_session_features(self, ts: pd.Timestamp) -> Dict[str, float]:
        """Get trading session features."""
        features = {}

        hour = ts.hour

        # Trading sessions (approximate)
        # Asian: 00:00 - 08:00 UTC
        # London: 08:00 - 16:00 UTC
        # New York: 13:00 - 21:00 UTC
        if 0 <= hour < 8:
            session = 0  # Asian
        elif 8 <= hour < 16:
            session = 1  # London
        else:
            session = 2  # New York

        features['SESSION'] = float(session) / 2.0  # Normalize to 0-1

        # Day of week (0=Monday, 4=Friday)
        features['DAY_OF_WEEK'] = float(ts.dayofweek) / 4.0  # Normalize to 0-1

        # Hour as cyclical features (sin/cos encoding)
        hour_rad = 2 * np.pi * hour / 24
        features['HOUR_SIN'] = np.sin(hour_rad)
        features['HOUR_COS'] = np.cos(hour_rad)

        return features

    def get_feature_vector(self, idx: int, current_price: Optional[float] = None) -> np.ndarray:
        """
        Get features as a numpy array (for adding to observation).

        Args:
            idx: Index in base dataframe
            current_price: Current close price

        Returns:
            Numpy array of features in consistent order
        """
        features = self.get_features(idx, current_price)
        return np.array([features.get(name, 0.0) for name in self._feature_names], dtype=np.float32)

    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names in order."""
        return self._feature_names.copy()

    @property
    def num_features(self) -> int:
        """Get total number of features."""
        return len(self._feature_names)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def add_mtf_features_to_df(
    df: pd.DataFrame,
    include_1h: bool = True,
    include_4h: bool = True,
    include_session: bool = True
) -> pd.DataFrame:
    """
    Add multi-timeframe features directly to a DataFrame.

    Args:
        df: DataFrame with OHLCV data
        include_1h: Include 1-hour timeframe
        include_4h: Include 4-hour timeframe
        include_session: Include session info

    Returns:
        DataFrame with MTF features added
    """
    mtf = MultiTimeframeFeatures(
        include_1h=include_1h,
        include_4h=include_4h,
        include_session=include_session
    )
    mtf.fit(df)

    # Add features to each row
    feature_data = []
    for idx in range(len(df)):
        features = mtf.get_features(idx)
        feature_data.append(features)

    features_df = pd.DataFrame(feature_data)
    result = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    return result


def create_mtf_calculator(df: pd.DataFrame) -> MultiTimeframeFeatures:
    """
    Create and fit a multi-timeframe feature calculator.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Fitted MultiTimeframeFeatures instance
    """
    mtf = MultiTimeframeFeatures()
    mtf.fit(df)
    return mtf


# =============================================================================
# TREND ALIGNMENT CHECKER
# =============================================================================

class TrendAlignmentChecker:
    """
    Check if trade direction aligns with higher timeframe trends.

    This is a simple but powerful filter:
    - Only take LONG trades when higher TFs are bullish
    - Only take SHORT trades when higher TFs are bearish
    - Neutral on higher TFs = proceed with caution
    """

    def __init__(self, mtf: MultiTimeframeFeatures):
        """
        Initialize trend alignment checker.

        Args:
            mtf: Fitted MultiTimeframeFeatures instance
        """
        self.mtf = mtf

    def check_alignment(
        self,
        idx: int,
        trade_direction: str,  # "long" or "short"
        require_4h: bool = True,
        require_1h: bool = False
    ) -> Tuple[bool, float, str]:
        """
        Check if a trade aligns with higher timeframe trends.

        Args:
            idx: Current bar index
            trade_direction: "long" or "short"
            require_4h: Require 4H trend alignment
            require_1h: Require 1H trend alignment

        Returns:
            Tuple of (is_aligned, alignment_score, reason)
            - is_aligned: True if trade aligns with HTF
            - alignment_score: 0-1 strength of alignment
            - reason: Explanation string
        """
        features = self.mtf.get_features(idx)

        trend_4h = features.get('HTF_TREND_4H', 0)
        trend_1h = features.get('HTF_TREND_1H', 0)
        strength_4h = features.get('HTF_STRENGTH_4H', 0)
        strength_1h = features.get('HTF_STRENGTH_1H', 0)

        is_long = trade_direction.lower() == "long"
        required_direction = 1 if is_long else -1

        # Check 4H alignment
        if require_4h:
            if trend_4h != required_direction and trend_4h != 0:
                return False, 0.0, f"4H trend is {'bullish' if trend_4h > 0 else 'bearish'}, trade is {trade_direction}"

        # Check 1H alignment
        if require_1h:
            if trend_1h != required_direction and trend_1h != 0:
                return False, 0.0, f"1H trend is {'bullish' if trend_1h > 0 else 'bearish'}, trade is {trade_direction}"

        # Calculate alignment score
        score_4h = (trend_4h == required_direction) * strength_4h
        score_1h = (trend_1h == required_direction) * strength_1h

        # Weight 4H more heavily
        alignment_score = 0.7 * score_4h + 0.3 * score_1h

        if alignment_score > 0.5:
            return True, alignment_score, "Strong alignment with higher timeframes"
        elif alignment_score > 0.2:
            return True, alignment_score, "Moderate alignment with higher timeframes"
        elif trend_4h == 0 and trend_1h == 0:
            return True, 0.3, "Higher timeframes are neutral (ranging)"
        else:
            return True, alignment_score, "Weak alignment - proceed with caution"


# =============================================================================
# KEY LEVEL DETECTOR
# =============================================================================

class KeyLevelDetector:
    """
    Detect key support/resistance levels from higher timeframes.

    Key levels from 4H and Daily timeframes are more significant
    than levels from 15-min charts.
    """

    def __init__(self, df: pd.DataFrame, lookback_4h: int = 50):
        """
        Initialize key level detector.

        Args:
            df: DataFrame with OHLCV data
            lookback_4h: Number of 4H bars to look back
        """
        self.df = df
        self.lookback_4h = lookback_4h

        # Resample to 4H
        if isinstance(df.index, pd.DatetimeIndex):
            self.df_4h = df.resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
        else:
            self.df_4h = df  # Use as-is if no datetime index

        # Calculate swing highs/lows
        self._calculate_swing_points()

    def _calculate_swing_points(self) -> None:
        """Calculate swing highs and lows on 4H."""
        df = self.df_4h

        if len(df) < 5:
            self.swing_highs = []
            self.swing_lows = []
            return

        swing_highs = []
        swing_lows = []

        for i in range(2, len(df) - 2):
            # Swing high: higher than 2 bars on each side
            if (df['High'].iloc[i] > df['High'].iloc[i-1] and
                df['High'].iloc[i] > df['High'].iloc[i-2] and
                df['High'].iloc[i] > df['High'].iloc[i+1] and
                df['High'].iloc[i] > df['High'].iloc[i+2]):
                swing_highs.append({
                    'price': df['High'].iloc[i],
                    'idx': i,
                    'timestamp': df.index[i] if hasattr(df.index, 'tolist') else i
                })

            # Swing low
            if (df['Low'].iloc[i] < df['Low'].iloc[i-1] and
                df['Low'].iloc[i] < df['Low'].iloc[i-2] and
                df['Low'].iloc[i] < df['Low'].iloc[i+1] and
                df['Low'].iloc[i] < df['Low'].iloc[i+2]):
                swing_lows.append({
                    'price': df['Low'].iloc[i],
                    'idx': i,
                    'timestamp': df.index[i] if hasattr(df.index, 'tolist') else i
                })

        self.swing_highs = swing_highs[-self.lookback_4h:] if swing_highs else []
        self.swing_lows = swing_lows[-self.lookback_4h:] if swing_lows else []

    def get_nearest_levels(
        self,
        current_price: float,
        n_levels: int = 3
    ) -> Dict[str, List[float]]:
        """
        Get nearest support and resistance levels.

        Args:
            current_price: Current market price
            n_levels: Number of levels to return on each side

        Returns:
            Dict with 'support' and 'resistance' lists
        """
        # Resistance levels (above current price)
        resistance = sorted([
            sh['price'] for sh in self.swing_highs
            if sh['price'] > current_price
        ])[:n_levels]

        # Support levels (below current price)
        support = sorted([
            sl['price'] for sl in self.swing_lows
            if sl['price'] < current_price
        ], reverse=True)[:n_levels]

        return {
            'support': support,
            'resistance': resistance
        }

    def get_distance_to_levels(self, current_price: float) -> Dict[str, float]:
        """
        Get distance to nearest support/resistance.

        Returns percentages (useful as features for RL agent).
        """
        levels = self.get_nearest_levels(current_price, n_levels=1)

        result = {
            'distance_to_resistance_pct': 0.0,
            'distance_to_support_pct': 0.0,
            'in_key_zone': 0.0  # 1 if near a key level
        }

        if levels['resistance']:
            result['distance_to_resistance_pct'] = (
                (levels['resistance'][0] - current_price) / current_price
            )

        if levels['support']:
            result['distance_to_support_pct'] = (
                (current_price - levels['support'][0]) / current_price
            )

        # Check if in key zone (within 0.3% of any level)
        all_levels = levels['resistance'] + levels['support']
        for level in all_levels:
            if abs(current_price - level) / current_price < 0.003:
                result['in_key_zone'] = 1.0
                break

        return result
