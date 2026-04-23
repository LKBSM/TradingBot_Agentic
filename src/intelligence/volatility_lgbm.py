"""LightGBM meta-learner for volatility forecasting (Rank 2).

Uses a rich feature stack (20+ features) to predict future ATR,
including HAR-RV components, session dummies, calendar proximity,
HMM regime state, and technical indicators.

Expected improvement: 20-35% RMSE over naive ATR (vs 10-15% for HAR-RV alone).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.intelligence.volatility_forecaster import InstrumentConfig, VolatilityForecaster

logger = logging.getLogger(__name__)


# =============================================================================
# LGBM VOLATILITY FORECASTER
# =============================================================================

class LGBMVolForecaster:
    """LightGBM meta-learner for volatility prediction.

    Builds a rich feature set from OHLCV data and predicts future_atr
    (N-bar ahead realized ATR). Can be used standalone or as residual
    corrector on top of HAR-RV (see HybridForecaster in Sprint 5).

    Usage:
        config = InstrumentConfig(symbol="XAUUSD")
        lgbm = LGBMVolForecaster(config)
        stats = lgbm.train(ohlcv_df, calendar_df)
        pred = lgbm.predict(features_dict)
    """

    # Feature names (deterministic order)
    FEATURE_NAMES: List[str] = [
        # HAR-RV
        "rv_daily", "rv_weekly", "rv_monthly",
        # ATR features
        "atr_14", "atr_7", "atr_change_5", "atr_change_20",
        # Returns
        "abs_return_1", "abs_return_5", "rolling_std_20",
        # Session dummies (one-hot)
        "session_asian", "session_london", "session_ny_overlap",
        "session_ny_afternoon", "session_after_hours",
        # Calendar
        "event_proximity_hours",
        # HMM regime
        "regime_state_ord", "regime_multiplier",
        # Technical
        "rsi_14", "bb_pct", "macd_hist_sign",
    ]

    def __init__(
        self,
        config: Optional[InstrumentConfig] = None,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_child_samples: int = 50,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        self._config = config or InstrumentConfig()
        self._model: Any = None
        self._lock = threading.Lock()
        self._is_trained: bool = False
        self._training_bars: int = 0
        self._feature_importance: Dict[str, float] = {}

        # Hyperparameters
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }

        # HAR-RV sub-model for feature extraction
        self._har_forecaster = VolatilityForecaster(self._config)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def config(self) -> InstrumentConfig:
        return self._config

    # ------------------------------------------------------------------ #
    # FEATURE ENGINEERING
    # ------------------------------------------------------------------ #

    def build_features(
        self,
        ohlcv_df: pd.DataFrame,
        calendar_df: Optional[pd.DataFrame] = None,
        har_forecaster: Optional[VolatilityForecaster] = None,
    ) -> pd.DataFrame:
        """Build the full feature matrix from OHLCV data.

        Args:
            ohlcv_df: OHLCV DataFrame.
            calendar_df: Optional economic calendar.
            har_forecaster: Optional calibrated HAR forecaster for regime features.

        Returns:
            DataFrame with all features + 'future_atr' target column.
        """
        forecaster = har_forecaster or self._har_forecaster
        df = forecaster._normalize_columns(ohlcv_df.copy())
        df = forecaster._add_features(df)

        # --- ATR variants ---
        if "atr_7" not in df.columns:
            df["atr_7"] = df["tr"].rolling(7).mean()
        df["atr_change_5"] = df["atr_14"].pct_change(5)
        df["atr_change_20"] = df["atr_14"].pct_change(20)

        # --- Returns features ---
        df["abs_return_1"] = df["returns_pct"].abs()
        df["abs_return_5"] = df["returns_pct"].abs().rolling(5).mean()
        df["rolling_std_20"] = df["returns_pct"].rolling(20).std()

        # --- Session dummies ---
        if "hour" in df.columns:
            sessions = self._config.session_hours
            for name, (start, end) in sessions.items():
                col = f"session_{name}"
                if end > start:
                    df[col] = ((df["hour"] >= start) & (df["hour"] < end)).astype(float)
                else:
                    # Wraps around midnight
                    df[col] = ((df["hour"] >= start) | (df["hour"] < end)).astype(float)
        else:
            for name in self._config.session_hours:
                df[f"session_{name}"] = 0.0

        # --- Calendar proximity ---
        if calendar_df is not None and "timestamp" in df.columns:
            event_times = forecaster._parse_calendar(calendar_df)
            df["event_proximity_hours"] = df["timestamp"].apply(
                lambda ts: self._compute_event_proximity(pd.Timestamp(ts), event_times)
            )
        else:
            df["event_proximity_hours"] = float("inf")

        # Cap event proximity at window size (no signal beyond window)
        max_window = float(self._config.event_window_hours)
        df["event_proximity_hours"] = df["event_proximity_hours"].clip(upper=max_window)

        # --- HMM regime features ---
        if forecaster._hmm_model is not None:
            df["regime_state_ord"], df["regime_multiplier"] = zip(
                *[self._get_regime_features(forecaster, df, i) for i in range(len(df))]
            )
        else:
            df["regime_state_ord"] = 1.0  # normal
            df["regime_multiplier"] = 1.0

        # --- Technical indicators ---
        df["rsi_14"] = self._compute_rsi(df["close"], 14)
        df["bb_pct"] = self._compute_bollinger_pct(df["close"], 20)
        df["macd_hist_sign"] = self._compute_macd_hist_sign(df["close"])

        return df

    @staticmethod
    def _compute_event_proximity(
        ts: pd.Timestamp, event_times: np.ndarray
    ) -> float:
        """Compute hours to nearest event."""
        if len(event_times) == 0:
            return float("inf")

        ts_np = np.datetime64(ts)
        deltas = np.abs(event_times - ts_np)
        min_delta = deltas.min()
        hours = min_delta / np.timedelta64(1, "h")
        return float(hours)

    @staticmethod
    def _get_regime_features(
        forecaster: VolatilityForecaster, df: pd.DataFrame, idx: int
    ) -> Tuple[float, float]:
        """Extract regime ordinal + multiplier for a bar."""
        mult, label = forecaster._get_regime_multiplier(df, idx)
        ordinal = {"low": 0.0, "normal": 1.0, "high": 2.0}.get(label, 1.0)
        return ordinal, mult

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _compute_bollinger_pct(close: pd.Series, period: int = 20) -> pd.Series:
        """Compute Bollinger %B (bandwidth position)."""
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        width = upper - lower
        return ((close - lower) / width.replace(0, np.nan)).fillna(0.5)

    @staticmethod
    def _compute_macd_hist_sign(close: pd.Series) -> pd.Series:
        """Compute MACD histogram sign (-1, 0, +1)."""
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return np.sign(hist)

    def extract_feature_row(self, feature_df: pd.DataFrame, idx: int = -1) -> Dict[str, float]:
        """Extract a single row of features as a dict."""
        row = {}
        for name in self.FEATURE_NAMES:
            if name in feature_df.columns:
                val = float(feature_df[name].iloc[idx])
                row[name] = val if np.isfinite(val) else 0.0
            else:
                row[name] = 0.0
        return row

    # ------------------------------------------------------------------ #
    # TRAINING
    # ------------------------------------------------------------------ #

    def train(
        self,
        ohlcv_df: pd.DataFrame,
        calendar_df: Optional[pd.DataFrame] = None,
        har_forecaster: Optional[VolatilityForecaster] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train LightGBM on OHLCV data.

        Args:
            ohlcv_df: OHLCV training data.
            calendar_df: Optional economic calendar.
            har_forecaster: Pre-calibrated HAR forecaster for feature extraction.
            verbose: Print training progress.

        Returns:
            Training stats dict.
        """
        with self._lock:
            return self._train_impl(ohlcv_df, calendar_df, har_forecaster, verbose)

    def _train_impl(
        self,
        ohlcv_df: pd.DataFrame,
        calendar_df: Optional[pd.DataFrame],
        har_forecaster: Optional[VolatilityForecaster],
        verbose: bool,
    ) -> Dict[str, Any]:
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("lightgbm not installed. Install with: pip install lightgbm")
            return {"trained": False, "error": "lightgbm not installed"}

        # Calibrate internal HAR if no external one provided
        forecaster = har_forecaster or self._har_forecaster
        if not forecaster.is_calibrated:
            logger.info("Calibrating HAR forecaster for feature extraction...")
            forecaster.calibrate(ohlcv_df, calendar_df)

        # Build features
        feature_df = self.build_features(ohlcv_df, calendar_df, forecaster)

        # Prepare training data
        target_col = "future_atr"
        feature_cols = [c for c in self.FEATURE_NAMES if c in feature_df.columns]

        valid_mask = (
            feature_df[feature_cols + [target_col]].notna().all(axis=1) &
            feature_df[feature_cols + [target_col]].apply(
                lambda row: np.all(np.isfinite(row)), axis=1
            )
        )
        valid_df = feature_df[valid_mask].reset_index(drop=True)

        if len(valid_df) < 500:
            logger.warning(
                "Insufficient valid data for LightGBM: %d bars (need 500)", len(valid_df)
            )
            return {"trained": False, "valid_bars": len(valid_df)}

        X = valid_df[feature_cols].values
        y = valid_df[target_col].values

        # Time-series split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)

        params = {
            "objective": "regression",
            "metric": "mae",
            "max_depth": self._params["max_depth"],
            "learning_rate": self._params["learning_rate"],
            "min_child_samples": self._params["min_child_samples"],
            "subsample": self._params["subsample"],
            "colsample_bytree": self._params["colsample_bytree"],
            "verbosity": 1 if verbose else -1,
            "seed": 42,
        }

        callbacks = [lgb.early_stopping(50, verbose=verbose)]
        if verbose:
            callbacks.append(lgb.log_evaluation(50))

        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=self._params["n_estimators"],
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Feature importance
        importance = self._model.feature_importance(importance_type="gain")
        total = importance.sum()
        self._feature_importance = {
            name: float(imp / total) if total > 0 else 0.0
            for name, imp in zip(feature_cols, importance)
        }

        # Validation metrics
        val_pred = self._model.predict(X_val)
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        naive_mae = float(np.mean(np.abs(X_val[:, feature_cols.index("atr_14")] - y_val)))
        improvement = (1 - val_mae / naive_mae) * 100 if naive_mae > 0 else 0.0

        self._is_trained = True
        self._training_bars = len(valid_df)

        logger.info(
            "LightGBM trained: %d bars, val_MAE=%.4f, naive_MAE=%.4f, improvement=%.1f%%",
            len(valid_df), val_mae, naive_mae, improvement,
        )

        return {
            "trained": True,
            "total_bars": len(valid_df),
            "train_bars": len(X_train),
            "val_bars": len(X_val),
            "val_mae": val_mae,
            "naive_mae": naive_mae,
            "improvement_pct": improvement,
            "best_iteration": self._model.best_iteration,
            "feature_importance": self._feature_importance.copy(),
        }

    # ------------------------------------------------------------------ #
    # PREDICTION
    # ------------------------------------------------------------------ #

    def predict(self, features: Dict[str, float]) -> float:
        """Predict future ATR from a feature dictionary.

        Args:
            features: Dict mapping feature names to values.

        Returns:
            Predicted future ATR (clamped to positive).
        """
        with self._lock:
            return self._predict_impl(features)

    def _predict_impl(self, features: Dict[str, float]) -> float:
        if self._model is None:
            raise RuntimeError("LGBMVolForecaster not trained. Call train() first.")

        # Build feature vector in correct order
        feature_vec = np.array([
            features.get(name, 0.0) for name in self.FEATURE_NAMES
            if name in self._get_trained_features()
        ]).reshape(1, -1)

        pred = float(self._model.predict(feature_vec)[0])
        return max(0.01, pred)

    def predict_from_df(self, feature_df: pd.DataFrame, idx: int = -1) -> float:
        """Predict from a feature DataFrame (convenience wrapper)."""
        features = self.extract_feature_row(feature_df, idx)
        return self.predict(features)

    def _get_trained_features(self) -> List[str]:
        """Get feature names used during training."""
        if self._model is not None:
            return self._model.feature_name()
        return self.FEATURE_NAMES

    # ------------------------------------------------------------------ #
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------------ #

    def feature_importance(self) -> Dict[str, float]:
        """Return normalized feature importance (gain-based).

        Returns:
            Dict mapping feature name to importance (0-1, sums to 1).
        """
        return self._feature_importance.copy()

    # ------------------------------------------------------------------ #
    # PERSISTENCE
    # ------------------------------------------------------------------ #

    def save_model(self, path: str) -> None:
        """Save trained model to disk (LightGBM native format)."""
        if self._model is None:
            raise RuntimeError("No model to save. Train first.")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(p))

        # Also save metadata
        import json
        meta_path = p.with_suffix(".meta.json")
        meta = {
            "feature_importance": self._feature_importance,
            "training_bars": self._training_bars,
            "params": self._params,
            "feature_names": self._get_trained_features(),
            "config_symbol": self._config.symbol,
            "config_timeframe": self._config.timeframe,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info("LightGBM model saved to %s", path)

    def load_model(self, path: str) -> bool:
        """Load trained model from disk. Returns True on success."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("lightgbm not installed")
            return False

        p = Path(path)
        if not p.exists():
            logger.warning("Model file not found: %s", path)
            return False

        try:
            self._model = lgb.Booster(model_file=str(p))

            # Load metadata
            import json
            meta_path = p.with_suffix(".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                self._feature_importance = meta.get("feature_importance", {})
                self._training_bars = meta.get("training_bars", 0)

            self._is_trained = True
            logger.info("LightGBM model loaded from %s", path)
            return True

        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

    # ------------------------------------------------------------------ #
    # STATS
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_trained": self._is_trained,
            "training_bars": self._training_bars,
            "symbol": self._config.symbol,
            "timeframe": self._config.timeframe,
            "n_features": len(self.FEATURE_NAMES),
            "params": self._params.copy(),
            "top_features": dict(
                sorted(self._feature_importance.items(), key=lambda x: -x[1])[:5]
            ) if self._feature_importance else {},
        }
