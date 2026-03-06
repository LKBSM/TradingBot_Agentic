import traceback
import sys
import os
from typing import Any, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
from collections import deque
import warnings
from enum import IntEnum
from ta.volatility import average_true_range
from rich.console import Console
from rich.table import Table
from rich.text import Text
import ta
import logging
from src.environment.multi_timeframe_features import add_mtf_features_to_df

logger = logging.getLogger(__name__)


# =============================================================================
# SECURITY FIX: Type-safe Position State Enum
# =============================================================================
class PositionState(IntEnum):
    """
    Type-safe position state enum.

    Using IntEnum for backward compatibility with existing code that uses
    integer comparisons, while providing type safety for new code.
    """
    FLAT = 0
    LONG = 1
    SHORT = -1

    @classmethod
    def from_value(cls, value: int) -> 'PositionState':
        """Convert integer to PositionState with validation."""
        for state in cls:
            if state.value == value:
                return state
        raise ValueError(f"Invalid position state value: {value}. Valid values: {[s.value for s in cls]}")

    def is_valid(self) -> bool:
        """Check if this is a valid position state."""
        return self in PositionState

# ✅ MAINTENANT ON PEUT UTILISER sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.environment.risk_manager import DynamicRiskManager as RiskManager
from src.environment.strategy_features import SmartMoneyEngine

# =============================================================================
# SECURITY FIX: TradeLogger abstraction to decouple from test code
# =============================================================================
# Previously imported from src.tests.monitor_training which is a test module.
# Now we use a lightweight local implementation or optional import.

class TradeLogger:
    """
    Lightweight trade logger for production use.

    This replaces the import from test code (src.tests.monitor_training)
    to properly decouple production code from test infrastructure.
    """

    def __init__(self):
        self.trades = []
        self.episode_logs = []
        self._current_episode = 0

    def log_trade(self, trade_id: int, action: str, entry_price: float,
                  exit_price: float = None, pnl: float = 0.0, **kwargs):
        """Log a trade event."""
        self.trades.append({
            'trade_id': trade_id,
            'episode': self._current_episode,
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            **kwargs
        })

    def new_episode(self):
        """Start a new episode."""
        self._current_episode += 1

    def get_trades(self) -> list:
        """Get all logged trades."""
        return self.trades.copy()

    def get_summary(self) -> dict:
        """Get trade summary statistics."""
        if not self.trades:
            return {'total_trades': 0, 'total_pnl': 0.0}

        total_pnl = sum(t.get('pnl', 0.0) for t in self.trades)
        winning = sum(1 for t in self.trades if t.get('pnl', 0.0) > 0)
        losing = sum(1 for t in self.trades if t.get('pnl', 0.0) < 0)

        return {
            'total_trades': len(self.trades),
            'winning_trades': winning,
            'losing_trades': losing,
            'total_pnl': total_pnl,
            'win_rate': winning / len(self.trades) if self.trades else 0.0
        }


# =============================================================================
# SECURITY FIX: DataFrame Validation Schema
# =============================================================================

class DataFrameValidationError(ValueError):
    """Raised when DataFrame validation fails."""
    pass


def validate_trading_dataframe(df: pd.DataFrame, required_columns: list = None) -> None:
    """
    Validate a trading DataFrame meets minimum requirements.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names (case-insensitive)

    Raises:
        DataFrameValidationError: If validation fails
    """
    if df is None:
        raise DataFrameValidationError("DataFrame cannot be None")

    if not isinstance(df, pd.DataFrame):
        raise DataFrameValidationError(f"Expected pandas DataFrame, got {type(df)}")

    if len(df) == 0:
        raise DataFrameValidationError("DataFrame is empty")

    # Default required columns for OHLCV data
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']

    # Case-insensitive column check
    df_cols_lower = [c.lower() for c in df.columns]
    missing = [col for col in required_columns if col.lower() not in df_cols_lower]

    if missing:
        raise DataFrameValidationError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Check for price columns having valid values
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        col_name = next((c for c in df.columns if c.lower() == col), None)
        if col_name:
            if df[col_name].isna().all():
                raise DataFrameValidationError(f"Column '{col_name}' contains only NaN values")

            non_null = df[col_name].dropna()
            if len(non_null) > 0 and (non_null <= 0).any():
                invalid_count = (non_null <= 0).sum()
                raise DataFrameValidationError(
                    f"Column '{col_name}' contains {invalid_count} non-positive values. "
                    f"Price data must be positive."
                )

    # Check for sufficient data
    if len(df) < 100:
        warnings.warn(
            f"DataFrame has only {len(df)} rows. "
            f"Recommended minimum is 100 for reliable training."
        )

    # Check for duplicate indices
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        warnings.warn(f"DataFrame has {dup_count} duplicate index values")

# IMPORTANT: Import constants from config.py
try:
    from config import (
        LOOKBACK_WINDOW_SIZE, INITIAL_BALANCE, TRANSACTION_FEE_PERCENTAGE,
        TRADE_COMMISSION_PER_TRADE, SLIPPAGE_PERCENTAGE, REWARD_SCALING_FACTOR,
        ALLOW_NEGATIVE_REVENUE_SELL, ALLOW_NEGATIVE_BALANCE, MINIMUM_ALLOWED_BALANCE,
        MIN_TRADE_QUANTITY, ACTION_SPACE_TYPE, FEATURES, OHLCV_COLUMNS,
        DOWNSIDE_PENALTY_MULTIPLIER, OVERNIGHT_HOLDING_PENALTY, HOLD_PENALTY_FACTOR,
        WINNING_TRADE_BONUS, LOSING_TRADE_PENALTY, FAILED_TRADE_ATTEMPT_PENALTY,
        TRADE_COOLDOWN_STEPS, RAPID_TRADE_PENALTY, TRAIN_END_DATE, SMC_CONFIG,
        RISK_PERCENTAGE_PER_TRADE, TAKE_PROFIT_PERCENTAGE, STOP_LOSS_PERCENTAGE,
        TSL_START_PROFIT_MULTIPLIER, TSL_TRAIL_DISTANCE_MULTIPLIER,
        MAX_LEVERAGE, MAX_DURATION_STEPS,
        W_RETURN, W_DRAWDOWN, W_FRICTION, W_LEVERAGE, W_TURNOVER, W_DURATION,
        # NEW: Long/Short action constants
        NUM_ACTIONS, ACTION_NAMES, ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG,
        ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT, POSITION_FLAT, POSITION_LONG, POSITION_SHORT,
        ENABLE_SHORT_SELLING, SHORT_BORROWING_FEE_DAILY, SHORT_OVERNIGHT_SWAP_PCT,
        # Fee constants (moved from hardcoded values)
        TRADE_COMMISSION_PCT_OF_TRADE, TRADE_COMMISSION_MIN_PCT_CAPITAL,
        # Reward scaling hyperparameters
        REWARD_TANH_SCALE, REWARD_OUTPUT_SCALE,
        # Episode length configuration
        FIXED_EPISODE_LENGTH, USE_FIXED_EPISODE_LENGTH
    )
except ImportError as _config_import_error:
    # ==========================================================================
    # SECURITY FIX: FAIL-FAST instead of silent fallback.
    #
    # PROBLEM: The previous fallback silently used DIFFERENT values than config.py:
    #   - LOOKBACK_WINDOW_SIZE: fallback=30 vs config=20
    #   - LOSING_TRADE_PENALTY: fallback=5.0 vs config=0.0
    #   - HOLD_PENALTY_FACTOR: fallback=0.005 vs config=0.01
    #   - FEATURES: fallback=26 features vs config=15 features
    #
    # IMPACT: If config import fails (e.g., wrong PYTHONPATH, missing file),
    # the bot would silently train/trade with completely different parameters,
    # leading to unpredictable behavior and potential financial losses.
    #
    # FIX: Raise immediately with a clear error message explaining how to fix.
    # ==========================================================================
    import logging as _logging
    _logger = _logging.getLogger(__name__)
    _logger.critical(
        "FATAL: Cannot import from config.py. The trading environment CANNOT "
        "operate with fallback parameters because they differ from the validated "
        "configuration. This would cause silent behavior changes.\n"
        "Original error: %s\n"
        "Fix: Ensure config.py is importable. Run: pip install -e . "
        "from the project root, or verify your PYTHONPATH includes the project root.",
        _config_import_error
    )
    raise ImportError(
        f"TradingEnv requires config.py to be importable. Silent fallback is disabled "
        f"to prevent parameter mismatch. Original error: {_config_import_error}"
    ) from _config_import_error

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 30}
    _gym_reset_return_info = True

    # =========================================================================
    # SECURITY FIX: Protected balance property to prevent direct tampering
    # =========================================================================
    @property
    def balance(self) -> float:
        """Get current account balance."""
        return self._balance

    @balance.setter
    def balance(self, value: float) -> None:
        """
        Set account balance with validation.

        SECURITY: Prevents invalid balance states that could corrupt trading logic.
        """
        # Validate type
        if not isinstance(value, (int, float)):
            raise TypeError(f"Balance must be numeric, got {type(value)}")

        value = float(value)

        # Check for NaN/Inf
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"Balance cannot be NaN or Inf: {value}")

        # Check negative balance (unless explicitly allowed)
        if value < 0 and not getattr(self, 'allow_negative_balance', False):
            raise ValueError(f"Balance cannot be negative: {value}. Set allow_negative_balance=True to override.")

        # Check minimum balance threshold
        min_balance = getattr(self, 'minimum_allowed_balance', 0.0)
        if value < min_balance and value > 0:
            import logging
            logging.getLogger(__name__).warning(
                f"Balance {value:.2f} is below minimum threshold {min_balance:.2f}"
            )

        self._balance = value
    # =========================================================================

    # =========================================================================
    # SECURITY FIX: Type-safe position state property
    # =========================================================================
    @property
    def position_type(self) -> int:
        """Get current position type (FLAT=0, LONG=1, SHORT=-1)."""
        return self._position_type

    @position_type.setter
    def position_type(self, value: int) -> None:
        """
        Set position type with validation.

        SECURITY: Prevents invalid position states that could corrupt trading logic.
        """
        # Validate value is a valid position state
        valid_states = {POSITION_FLAT, POSITION_LONG, POSITION_SHORT}
        if value not in valid_states:
            raise ValueError(
                f"Invalid position type: {value}. Must be one of: "
                f"POSITION_FLAT({POSITION_FLAT}), POSITION_LONG({POSITION_LONG}), POSITION_SHORT({POSITION_SHORT})"
            )
        self._position_type = value
    # =========================================================================

    def __init__(self, df: pd.DataFrame, render_mode: str = "none", **kwargs):
        super().__init__()
        self.render_mode = render_mode

        # =====================================================================
        # SECURITY FIX: Validate DataFrame before any processing
        # =====================================================================
        try:
            validate_trading_dataframe(df)
        except DataFrameValidationError as e:
            raise ValueError(f"Invalid trading data: {e}")

        self.lookback_window_size = kwargs.get('lookback_window_size', LOOKBACK_WINDOW_SIZE)
        self.initial_balance = kwargs.get('initial_balance', INITIAL_BALANCE)

        self.transaction_fee_percentage = kwargs.get('transaction_fee_percentage', TRANSACTION_FEE_PERCENTAGE)
        self.trade_commission_per_trade = kwargs.get('trade_commission_per_trade', TRADE_COMMISSION_PER_TRADE)
        self.slippage_percentage = kwargs.get('slippage_percentage', SLIPPAGE_PERCENTAGE)
        self.reward_scaling_factor = kwargs.get('reward_scaling_factor', REWARD_SCALING_FACTOR)
        self.allow_negative_revenue_sell = kwargs.get('allow_negative_revenue_sell', ALLOW_NEGATIVE_REVENUE_SELL)
        self.allow_negative_balance = kwargs.get('allow_negative_balance', ALLOW_NEGATIVE_BALANCE)
        self.minimum_allowed_balance = kwargs.get('minimum_allowed_balance', MINIMUM_ALLOWED_BALANCE)
        self.min_trade_quantity = kwargs.get('min_trade_quantity', MIN_TRADE_QUANTITY)
        self.downside_penalty_multiplier = kwargs.get('downside_penalty_multiplier', DOWNSIDE_PENALTY_MULTIPLIER)
        self.overnight_holding_penalty = kwargs.get('overnight_holding_penalty', OVERNIGHT_HOLDING_PENALTY)
        self.hold_penalty_factor = kwargs.get('hold_penalty_factor', HOLD_PENALTY_FACTOR)
        self.winning_trade_bonus = kwargs.get('winning_trade_bonus', WINNING_TRADE_BONUS)
        self.losing_trade_penalty = kwargs.get('losing_trade_penalty', LOSING_TRADE_PENALTY)
        self.trade_cooldown_steps = kwargs.get('trade_cooldown_steps', TRADE_COOLDOWN_STEPS)
        self.rapid_trade_penalty = kwargs.get('rapid_trade_penalty', RAPID_TRADE_PENALTY)
        self.smc_config = kwargs.get('smc_config', SMC_CONFIG)
        self.features_config = kwargs.get('features', FEATURES)
        self.ohlcv_columns = kwargs.get('ohlcv_columns', OHLCV_COLUMNS)
        self.enable_logging = kwargs.get('enable_logging', False)
        self.trade_logger = TradeLogger() if self.enable_logging else None
        self.trade_id_counter = 0
        self.episode_count = 0
        self.episode_reward = 0.0

        # --- NEW: Reward Weights and Limits (Hyperparameters) ---
        self.max_leverage_limit = kwargs.get('max_leverage_limit', MAX_LEVERAGE)
        # Assurez-vous que cette ligne est présente dans le __init__
        self.max_duration_steps = kwargs.get('max_duration_steps', MAX_DURATION_STEPS)
        self.w_R = kwargs.get('w_R', W_RETURN)
        self.w_DD = kwargs.get('w_DD', W_DRAWDOWN)
        self.w_F = kwargs.get('w_F', W_FRICTION)
        self.w_L = kwargs.get('w_L', W_LEVERAGE)
        self.w_T = kwargs.get('w_T', W_TURNOVER)
        self.w_D = kwargs.get('w_D', W_DURATION)
        self.debug_rewards = False  # Set to True only for debugging
        # --------------------------------------------------------

        self.risk_manager = RiskManager(config=kwargs)
        self._risk_client_id = kwargs.get('risk_client_id', 'default_client')
        self.risk_manager.set_client_profile(
            client_id=self._risk_client_id,
            initial_equity=self.initial_balance,
            max_drawdown_pct=kwargs.get('max_drawdown_pct', 20.0),  # 20% MDD default
            kelly_fraction_limit=kwargs.get('kelly_fraction_limit', 0.1),  # 10% cap by default
            max_trade_risk_pct=kwargs.get('max_trade_risk_pct', 0.01)  # 1% per trade default
        )
        # Fee constants - now from config.py (no more hardcoded defaults)
        self.trade_commission_pct_of_trade = kwargs.get(
            'trade_commission_pct_of_trade',
            TRADE_COMMISSION_PCT_OF_TRADE  # From config.py
        )
        self.trade_commission_min_pct_capital = kwargs.get(
            'trade_commission_min_pct_capital',
            TRADE_COMMISSION_MIN_PCT_CAPITAL  # From config.py
        )

        # Reward scaling hyperparameters - tunable per bot for optimal learning
        self.reward_tanh_scale = kwargs.get('reward_tanh_scale', REWARD_TANH_SCALE)
        self.reward_output_scale = kwargs.get('reward_output_scale', REWARD_OUTPUT_SCALE)

        # Episode length configuration - fixed length for stable PPO training
        self.fixed_episode_length = kwargs.get('fixed_episode_length', FIXED_EPISODE_LENGTH)
        self.use_fixed_episode_length = kwargs.get('use_fixed_episode_length', USE_FIXED_EPISODE_LENGTH)

        self.df_raw = df.copy()

        # MTF flag must be set BEFORE _process_data() which uses it
        from config import ENABLE_MTF_FEATURES, MTF_FEATURES
        self._enable_mtf = kwargs.get('enable_mtf', ENABLE_MTF_FEATURES)

        self.processed_data = self._process_data(self.df_raw)
        self.df = self.processed_data
        self.features = [col for col in self.features_config if col in self.df.columns]

        # Add MTF features if enabled and available in processed data
        if self._enable_mtf:
            for f in MTF_FEATURES:
                if f in self.df.columns and f not in self.features:
                    self.features.append(f)

        # Sprint 6: Feature reducer (set by trainer, None = disabled)
        self._feature_reducer = kwargs.get('feature_reducer', None)

        # Sprint 7: Incremental feature engine (for live mode, None = use batch)
        self._incremental_engine = kwargs.get('incremental_engine', None)

        if 'Close' not in self.df.columns and 'close' in self.df.columns:
            self.df.rename(columns={'close': 'Close'}, inplace=True)
        if 'Close' not in self.df.columns:
            raise ValueError("Processed DataFrame must contain a 'Close' price column.")

        if len(self.df) <= self.lookback_window_size:
            raise ValueError(
                f"Pas assez de données (longueur: {len(self.df)}) pour une fenêtre de {self.lookback_window_size}."
            )

        self.balance = self.initial_balance
        self.stock_quantity = 0.0
        self.net_worth = self.initial_balance
        self.total_fees_paid = 0.0
        self.current_step = self.lookback_window_size
        self.max_steps = len(self.df) - 1
        self.end_idx = self.max_steps  # Will be properly set in reset()
        self.entry_price = np.nan

        # --- NEW: State tracking variables for advanced reward function ---
        self.previous_nav = self.initial_balance
        self.peak_nav = self.initial_balance
        self.previous_drawdown_level = 0.0
        self.current_leverage = 0.0
        self.current_hold_duration = 0
        self.transaction_cost_incurred_step = 0.0
        self.traded_value_step = 0.0
        # ------------------------------------------------------------------

        self.trade_details = {'trade_pnl_abs': 0.0, 'trade_pnl_pct': 0.0, 'trade_type': 'none', 'trade_success': False,
                              'trade_value': 0.0, 'commission': 0.0}
        self.trade_history_summary = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        self.total_fees_paid_episode = 0.0
        self.np_random = np.random.default_rng()
        self.last_action = 0
        self.actual_action_executed = 0
        self.last_trade_step = -np.inf
        self.recent_trade_attempts_steps = deque(maxlen=self.lookback_window_size)

        # Invalid action tracking - helps understand agent behavior
        self.invalid_action_count = 0
        self.invalid_action_types = {
            'already_in_position': 0,
            'no_long_position': 0,
            'no_short_position': 0,
            'short_selling_disabled': 0
        }

        # =====================================================================
        # SCALER CONFIGURATION - DATA LEAKAGE PREVENTION
        # =====================================================================
        # CRITICAL: Scaler must be fit ONLY on training data to prevent data leakage
        # Data leakage = fitting scaler on future data that wouldn't be available
        # in live trading, which leads to overly optimistic backtest results.
        #
        # Options (in order of preference):
        #   1. Pass pre_fitted_scaler (recommended for val/test environments)
        #   2. Pass scaler_fit_end_idx to limit fitting to training portion
        #   3. Set strict_mode=False to allow legacy behavior (NOT RECOMMENDED)

        pre_fitted_scaler = kwargs.get('pre_fitted_scaler', None)
        scaler_fit_end_idx = kwargs.get('scaler_fit_end_idx', None)
        strict_mode = kwargs.get('strict_scaler_mode', True)  # Default: strict

        if pre_fitted_scaler is not None:
            # Use externally fitted scaler (recommended for val/test environments)
            self.scaler = pre_fitted_scaler
            self._scaler_source = "pre_fitted"
        else:
            self.scaler = MinMaxScaler(clip=True)  # clip=True prevents out-of-range values

            if scaler_fit_end_idx is not None:
                # Fit only on training data (up to scaler_fit_end_idx)
                train_data = self.df[self.features].iloc[:scaler_fit_end_idx].dropna()
                if not train_data.empty:
                    self.scaler.fit(train_data.values)
                    self._scaler_source = f"training_data[:{scaler_fit_end_idx}]"
                else:
                    raise ValueError(
                        "No valid training rows for scaler fitting. "
                        "Check that scaler_fit_end_idx points to valid data."
                    )
            else:
                # SECURITY FIX: In strict mode, refuse to fit on all data
                if strict_mode:
                    raise ValueError(
                        "DATA LEAKAGE ERROR: No scaler configuration provided. "
                        "Fitting scaler on all data causes data leakage and inflated backtest results. "
                        "Solutions:\n"
                        "  1. Pass 'pre_fitted_scaler' (fit on training data only)\n"
                        "  2. Pass 'scaler_fit_end_idx' (index where training data ends)\n"
                        "  3. Set 'strict_scaler_mode=False' to allow legacy behavior (NOT RECOMMENDED)\n"
                        "\nExample:\n"
                        "  # Fit scaler on training data only\n"
                        "  train_end_idx = int(len(df) * 0.7)\n"
                        "  env = TradingEnv(df, scaler_fit_end_idx=train_end_idx)"
                    )
                else:
                    # Legacy behavior - fit on all data (NOT RECOMMENDED)
                    valid_rows = self.df[self.features].dropna()
                    if not valid_rows.empty:
                        warnings.warn(
                            "SCALER WARNING: Fitting on ALL data. This causes data leakage! "
                            "Pass 'scaler_fit_end_idx' (training end index) or 'pre_fitted_scaler' "
                            "to fix this issue. Set strict_scaler_mode=True to enforce this.",
                            category=UserWarning
                        )
                        self.scaler.fit(valid_rows.values)
                        self._scaler_source = "all_data_LEAKY"
                    else:
                        raise ValueError(
                            "No valid rows for scaling. Check feature generation pipeline."
                        )
        action_space_type = kwargs.get('action_space_type', ACTION_SPACE_TYPE)
        if action_space_type == "discrete":
            # NEW: 5 actions for long/short trading
            # 0=HOLD, 1=OPEN_LONG, 2=CLOSE_LONG, 3=OPEN_SHORT, 4=CLOSE_SHORT
            self.action_space = spaces.Discrete(NUM_ACTIONS)
        else:
            raise ValueError(f"Type d'espace d'action invalide: {action_space_type}")

        # NEW: Position tracking for long/short
        # POSITION_FLAT=0, POSITION_LONG=1, POSITION_SHORT=-1
        self.position_type = POSITION_FLAT
        self.enable_short_selling = kwargs.get('enable_short_selling', ENABLE_SHORT_SELLING)
        self.short_borrowing_fee_daily = kwargs.get('short_borrowing_fee_daily', SHORT_BORROWING_FEE_DAILY)
        self.short_overnight_swap_pct = kwargs.get('short_overnight_swap_pct', SHORT_OVERNIGHT_SWAP_PCT)

        num_features_per_step = len(self.features)
        raw_obs_size = num_features_per_step * self.lookback_window_size + 3
        # Sprint 6: If a fitted FeatureReducer is attached, observation space shrinks
        if self._feature_reducer is not None and self._feature_reducer.is_fitted:
            expected_obs_size = self._feature_reducer.n_components + 3  # PCA dims + 3 state
        else:
            expected_obs_size = raw_obs_size
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(expected_obs_size,), dtype=np.float32)

        self.reset()

    def _process_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Traite les données brutes OHLCV et génère les features techniques et SMC.
        VERSION PRODUCTION-READY avec gestion intelligente des NaN et validation.
        """

        # =========================================================================
        # ÉTAPE 1: PRÉPARATION ET NORMALISATION DES COLONNES
        # =========================================================================

        df = df_raw.copy()

        # Préserver l'horodatage original
        if df_raw.index.name is not None:
            df = df.reset_index()

        # Identifier et renommer la colonne de date
        date_col_name = str(df.columns[0])
        df.rename(columns={date_col_name: 'Original_Timestamp'}, inplace=True)

        # Normaliser les colonnes OHLCV en minuscules (sauf Original_Timestamp)
        df.columns = [col.lower() if col != 'Original_Timestamp' else col for col in df.columns]

        # Validation des colonnes essentielles
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Colonnes OHLCV manquantes: {missing_cols}")

        # =========================================================================
        # ÉTAPE 2: GÉNÉRATION DES FEATURES (TA + SMC)
        # =========================================================================

        try:
            engine = SmartMoneyEngine(data=df, config=self.smc_config)
            df_processed = engine.analyze()
            logger.info(f"SmartMoneyEngine: {len(df_processed)} lignes generees")
        except Exception as e:
            logger.error(f"Erreur SmartMoneyEngine: {e}")
            raise

        # =========================================================================
        # ÉTAPE 3: NORMALISATION DES COLONNES (Capitalisation)
        # =========================================================================

        # Capitaliser les colonnes OHLCV pour cohérence avec le reste du code
        df_processed.columns = [
            c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume']
            else c for c in df_processed.columns
        ]

        # Assurer la présence de 'Close' (colonne critique)
        if 'close' in df_processed.columns and 'Close' not in df_processed.columns:
            df_processed.rename(columns={'close': 'Close'}, inplace=True)

        # Assurer la présence de 'ATR' (indicateur critique pour RiskManager)
        if 'ATR' not in df_processed.columns:
            if 'atr' in df_processed.columns:
                df_processed.rename(columns={'atr': 'ATR'}, inplace=True)
            else:
                logger.warning("ATR manquant. Calcul de fallback...")
                df_processed['ATR'] = ta.volatility.average_true_range(
                    df_processed['High'],
                    df_processed['Low'],
                    df_processed['Close'],
                    window=self.smc_config.get('ATR_WINDOW', 14)
                )

        # =========================================================================
        # ÉTAPE 4: NETTOYAGE DES INFINITÉS
        # =========================================================================

        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        logger.info(f"Infinites remplacees par NaN")

        # =========================================================================
        # ETAPE 4a: WEEKEND GAP DETECTION (Gold-specific)
        # =========================================================================
        # Try to find timestamps: column 'Original_Timestamp', or DatetimeIndex
        ts = None
        if 'Original_Timestamp' in df_processed.columns:
            ts = pd.to_datetime(df_processed['Original_Timestamp'])
        elif 'original_timestamp' in df_processed.columns:
            ts = pd.to_datetime(df_processed['original_timestamp'])
        elif isinstance(df_processed.index, pd.DatetimeIndex):
            ts = df_processed.index.to_series()
        elif 'Date' in df_processed.columns:
            ts = pd.to_datetime(df_processed['Date'])
        elif 'date' in df_processed.columns:
            ts = pd.to_datetime(df_processed['date'])

        if ts is not None:
            try:
                time_diff = ts.diff()
                # Gap = time difference > 6 hours (normal 15min interval)
                gap_mask = time_diff > pd.Timedelta(hours=6)
                df_processed['WEEKEND_GAP'] = 0.0
                if gap_mask.any():
                    # Calculate gap size as % of Close price
                    gap_returns = df_processed['Close'].pct_change()
                    df_processed.loc[gap_mask, 'WEEKEND_GAP'] = gap_returns[gap_mask].clip(-0.05, 0.05)
                    gap_count = gap_mask.sum()
                    logger.info(f"Weekend gaps detected: {gap_count}")
            except Exception as e:
                df_processed['WEEKEND_GAP'] = 0.0
                logger.warning(f"Weekend gap detection failed (non-fatal): {e}")
        else:
            df_processed['WEEKEND_GAP'] = 0.0
            logger.warning("No timestamp column for gap detection")

        # =========================================================================
        # ETAPE 4b: MULTI-TIMEFRAME FEATURES
        # =========================================================================
        if getattr(self, '_enable_mtf', True):
            try:
                df_processed = add_mtf_features_to_df(
                    df_processed,
                    include_1h=True,
                    include_4h=True,
                    include_session=True
                )
                logger.info("MTF features added: 14 columns")
            except Exception as e:
                logger.warning(f"MTF features failed (non-fatal): {e}")

        # =========================================================================
        # ÉTAPE 5: GESTION INTELLIGENTE DES NaN (CRITIQUE!)
        # =========================================================================

        logger.info("Analyse des NaN avant nettoyage:")
        nan_summary = df_processed.isna().sum()
        nan_summary = nan_summary[nan_summary > 0].sort_values(ascending=False)
        if not nan_summary.empty:
            logger.info(f"\n{nan_summary}")

        # --- 5.1: SUPPRESSION DES LIGNES AVEC NaN CRITIQUES ---
        # Ces colonnes ne peuvent PAS contenir de NaN (risque de crash)
        essential_features = [
            'Open', 'High', 'Low', 'Close',  # Prix
            'RSI',  # Momentum critique
            'MACD_line',  # Tendance critique
            'ATR'  # Volatilité critique
        ]

        rows_before = len(df_processed)
        df_processed.dropna(
            subset=[f for f in essential_features if f in df_processed.columns],
            inplace=True
        )
        rows_dropped = rows_before - len(df_processed)

        if rows_dropped > 0:
            logger.warning(f"{rows_dropped} lignes supprimees (NaN dans colonnes critiques)")

        # Validation post-suppression
        if len(df_processed) < self.lookback_window_size * 2:
            raise ValueError(
                f"❌ Pas assez de données après nettoyage: {len(df_processed)} lignes "
                f"(minimum requis: {self.lookback_window_size * 2})"
            )

        # --- 5.2: FORWARD/BACKWARD FILL POUR INDICATEURS LENTS ---
        # Ces indicateurs peuvent être interpolés de manière sécurisée
        slow_indicators = [
            'RSI', 'MACD_line', 'MACD_signal', 'MACD_Diff',
            'BB_L', 'BB_M', 'BB_H', 'ATR'
        ]

        for col in slow_indicators:
            if col in df_processed.columns:
                # Forward fill (propagation de la dernière valeur connue)
                df_processed[col] = df_processed[col].ffill()
                # Backward fill pour les premières valeurs (si nécessaire)
                df_processed[col] = df_processed[col].bfill()

        logger.info(f"Indicateurs lents interpoles: {slow_indicators}")

        # --- 5.3: REMPLISSAGE PAR 0 POUR SIGNAUX SMC ---
        # Ces colonnes sont des signaux événementiels: NaN = "Pas de signal" = 0
        smc_signal_cols = [
            'FVG_SIGNAL',  # Fair Value Gap signal
            'FVG_SIZE_NORM',  # Taille FVG normalisée
            'BOS_SIGNAL',  # Break of Structure
            'CHOCH_SIGNAL',  # Change of Character
            'OB_STRENGTH_NORM',  # Force Order Block
            'UP_FRACTAL',  # Swing High
            'DOWN_FRACTAL',  # Swing Low
            'BULLISH_OB_HIGH',  # Zone Order Block haussier
            'BULLISH_OB_LOW',
            'BEARISH_OB_HIGH',  # Zone Order Block baissier
            'BEARISH_OB_LOW'
        ]

        for col in smc_signal_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(0.0)

        logger.info(
            f"Signaux SMC initialises a 0: {len([c for c in smc_signal_cols if c in df_processed.columns])} colonnes")

        # --- 5.4: REMPLISSAGE PAR MÉDIANE POUR LE VOLUME ---
        # Le volume peut varier énormément, la médiane est plus robuste que 0
        if 'Volume' in df_processed.columns:
            volume_median = df_processed['Volume'].median()
            nan_count = df_processed['Volume'].isna().sum()
            if nan_count > 0:
                df_processed['Volume'] = df_processed['Volume'].fillna(volume_median)
                logger.info(f"Volume: {nan_count} NaN remplaces par mediane ({volume_median:.0f})")

        # --- 5.5: REMPLISSAGE FINAL (SÉCURITÉ) ---
        # Si des colonnes ont encore des NaN, on remplace par 0 (dernier recours)
        remaining_cols_with_nan = df_processed.columns[df_processed.isna().any()].tolist()

        if remaining_cols_with_nan:
            logger.warning(f"Colonnes avec NaN restants: {remaining_cols_with_nan}")
            df_processed[remaining_cols_with_nan] = df_processed[remaining_cols_with_nan].fillna(0.0)

        # =========================================================================
        # ÉTAPE 6: VALIDATION FINALE DE LA QUALITÉ DES DONNÉES
        # =========================================================================

        # Vérifier qu'il ne reste AUCUN NaN
        final_nan_count = df_processed.isna().sum().sum()
        if final_nan_count > 0:
            raise ValueError(f"❌ {final_nan_count} NaN restants après nettoyage complet!")

        # Vérifier qu'il ne reste AUCUNE infinité
        inf_count = np.isinf(df_processed.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            raise ValueError(f"❌ {inf_count} valeurs infinies détectées!")

        # Vérifier la présence de prix aberrants (0 ou négatifs)
        if (df_processed['Close'] <= 0).any():
            invalid_prices = (df_processed['Close'] <= 0).sum()
            raise ValueError(f"❌ {invalid_prices} prix invalides (≤0) détectés!")

        # Vérifier la détection de mouvements extrêmes (flash crash potentiel)
        price_changes = df_processed['Close'].pct_change().abs()
        extreme_moves = (price_changes > 0.2).sum()  # Mouvements >20%
        if extreme_moves > 0:
            logger.warning(f"{extreme_moves} mouvements de prix >20% detectes (verifier donnees)")

        # Reset index pour un DataFrame propre
        df_processed.reset_index(drop=True, inplace=True)

        # =========================================================================
        # ÉTAPE 7: RAPPORT FINAL
        # =========================================================================

        logger.info(
            f"\n{'=' * 70}\n"
            f"RAPPORT DE TRAITEMENT DES DONNEES\n"
            f"{'=' * 70}\n"
            f"Lignes finales:        {len(df_processed)}\n"
            f"Colonnes totales:      {len(df_processed.columns)}\n"
            f"NaN restants:          {df_processed.isna().sum().sum()}\n"
            f"Valeurs infinies:      {np.isinf(df_processed.select_dtypes(include=[np.number])).sum().sum()}\n"
            f"Periode couverte:      {df_processed.index[0]} -> {df_processed.index[-1]}"
        )

        # Afficher les features disponibles
        feature_categories = {
            'Prix': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'Indicateurs TA': ['RSI', 'MACD_line', 'MACD_signal', 'MACD_Diff', 'BB_L', 'BB_M', 'BB_H', 'ATR'],
            'SMC Signaux': ['BOS_SIGNAL', 'CHOCH_SIGNAL', 'FVG_SIGNAL', 'UP_FRACTAL', 'DOWN_FRACTAL'],
            'SMC Zones': ['BULLISH_OB_HIGH', 'BULLISH_OB_LOW', 'BEARISH_OB_HIGH', 'BEARISH_OB_LOW']
        }

        features_report = "Features disponibles par categorie:\n"
        for category, features in feature_categories.items():
            available = [f for f in features if f in df_processed.columns]
            features_report += f"  {category}: {len(available)}/{len(features)} -> {available}\n"
        features_report += "=" * 70
        logger.info(f"\n{features_report}")

        return df_processed

    def _get_obs(self) -> np.ndarray:
        """
        Returns the flattened observation vector for the current environment step.
        Includes scaled feature window and normalized portfolio state metrics.
        """

        # --- 1. Extract lookback window safely ---
        start_idx = self.current_step - self.lookback_window_size + 1
        obs_df = self.df.iloc[max(0, start_idx):self.current_step + 1].copy()

        # Padding: edge-value replication instead of zero-padding
        # Zero-padding creates phantom signals (e.g., RSI=0 = "extremely oversold")
        # Edge replication means "no change yet" which is semantically correct
        if len(obs_df) < self.lookback_window_size:
            padding_needed = self.lookback_window_size - len(obs_df)
            first_row = obs_df.iloc[[0]]
            padded_df = pd.concat([first_row] * padding_needed, ignore_index=True)
            obs_df = pd.concat([padded_df, obs_df], ignore_index=True)

        # Keep the last lookback_window_size rows only
        obs_df = obs_df.tail(self.lookback_window_size)

        # --- 2. Feature extraction ---
        features_data = obs_df[self.features].values

        # --- 3. Scaling ---
        if self.scaler is not None:
            scaled_features = self.scaler.transform(features_data)
        else:
            scaled_features = features_data
            warnings.warn("⚠️ Scaler is None — features are not normalized.")

        flat_obs = scaled_features.flatten()

        # --- 3b. Sprint 6: Apply PCA dimensionality reduction ---
        if self._feature_reducer is not None and self._feature_reducer.is_fitted:
            try:
                flat_obs = self._feature_reducer.transform(flat_obs)
            except Exception:
                pass  # Fall back to raw features if transform fails

        # --- 4. Portfolio state extraction ---
        # Force scalar conversion to avoid ambiguous truth values
        # --- 4. Portfolio state extraction ---
        step_index = int(self.current_step)
        current_price = float(np.atleast_1d(self.df["Close"].iloc[step_index])[0])

        current_equity = float(self.balance + self.stock_quantity * current_price)

        # --- 5. Leverage calculation ---
        total_position_value = float(self.stock_quantity * current_price)
        if current_equity > 1e-9:
            self.current_leverage = total_position_value / current_equity
        else:
            self.current_leverage = 0.0

        # --- 6. Normalized financial metrics ---
        normalized_balance = float(self.balance / self.initial_balance)
        normalized_net_worth = float(current_equity / self.initial_balance)
        normalized_stock_quantity = float(total_position_value / self.initial_balance)

        # --- 7. Concatenate into a single observation vector ---
        observation = np.append(
            flat_obs, [normalized_balance, normalized_stock_quantity, normalized_net_worth]
        )

        # --- 7b. Clip to observation space bounds for PPO stability ---
        observation = np.clip(observation, -10.0, 10.0)

        # --- 8. Shape and value validation ---
        if self._feature_reducer is not None and self._feature_reducer.is_fitted:
            expected_size = self._feature_reducer.n_components + 3
        else:
            expected_size = len(self.features) * self.lookback_window_size + 3
        if observation.shape[0] != expected_size:
            warnings.warn(
                f"⚠️ Observation shape mismatch: expected {expected_size}, got {observation.shape[0]}."
            )

        # =====================================================================
        # SECURITY FIX: Intelligent NaN/Inf handling
        # =====================================================================
        # Problem: Replacing NaN with 0.0 corrupts indicator semantics
        #   - RSI=0 means "extremely oversold" (wrong signal)
        #   - RSI=NaN means "not enough data" (different meaning)
        #
        # Solution: Context-aware replacement
        #   - Portfolio metrics (balance, etc.): Use actual calculated values
        #   - Technical indicators: Use last known value (forward-fill logic)
        #   - Only fall back to neutral values if no history available

        # Track NaN locations before any modification
        has_nan = np.isnan(observation).any()
        has_inf = np.isinf(observation).any()

        if has_nan or has_inf:
            # Split observation into feature window and portfolio state
            feature_size = len(self.features) * self.lookback_window_size
            feature_part = observation[:feature_size]
            portfolio_part = observation[feature_size:]  # [balance, stock_qty, net_worth]

            # For features: use forward-fill from previous observation if available
            if has_nan or has_inf:
                if hasattr(self, '_last_valid_features') and self._last_valid_features is not None:
                    # Replace NaN/Inf with last known valid values
                    nan_mask = np.isnan(feature_part) | np.isinf(feature_part)
                    if nan_mask.any():
                        feature_part[nan_mask] = self._last_valid_features[nan_mask]

                # Final cleanup: any remaining NaN/Inf get neutral values
                # Use feature-specific neutral values based on typical ranges
                feature_part = np.nan_to_num(
                    feature_part,
                    nan=0.5,     # Neutral value for normalized features
                    posinf=1.0,  # Max for normalized features
                    neginf=0.0   # Min for normalized features
                )

            # For portfolio state: these should never be NaN (calculated values)
            # If they are, it indicates a bug - use safe defaults and log
            if np.isnan(portfolio_part).any() or np.isinf(portfolio_part).any():
                warnings.warn(
                    f"Portfolio state contains NaN/Inf at step {self.current_step}. "
                    f"This may indicate a calculation error. Values: {portfolio_part}"
                )
                portfolio_part = np.nan_to_num(
                    portfolio_part,
                    nan=1.0,    # Neutral: normalized balance = 1.0
                    posinf=10.0,  # Cap extreme values
                    neginf=0.0
                )

            observation = np.concatenate([feature_part, portfolio_part])

            # Log warning periodically (not every step to avoid spam)
            if not hasattr(self, '_nan_warning_count'):
                self._nan_warning_count = 0
            self._nan_warning_count += 1

            if self._nan_warning_count <= 5 or self._nan_warning_count % 100 == 0:
                warnings.warn(
                    f"[Step {self.current_step}] Observation contained NaN/Inf values "
                    f"(occurrence #{self._nan_warning_count}). Applied intelligent fill."
                )

        # Store valid features for forward-fill in next step
        feature_size = len(self.features) * self.lookback_window_size
        current_features = observation[:feature_size]
        if not (np.isnan(current_features).any() or np.isinf(current_features).any()):
            self._last_valid_features = current_features.copy()

        return observation

    def _get_info(self) -> dict:
        current_price = self.df.iloc[self.current_step]['Close']
        self.net_worth = self.balance + self.stock_quantity * current_price
        self.total_trades = self.winning_trades + self.losing_trades
        current_return_percentage = ((self.net_worth - self.initial_balance) / self.initial_balance) * 100 \
            if self.initial_balance > 0 else 0.0

        info = {
            'balance': self.balance,
            'stock_quantity': self.stock_quantity,
            'net_worth': self.net_worth,
            'current_price': current_price,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_fees_paid': self.total_fees_paid_episode,
            'current_step': self.current_step,
            'episode_return_percentage': current_return_percentage,
            'entry_price': self.entry_price if not math.isclose(self.stock_quantity, 0.0, abs_tol=1e-9) else np.nan,
            'current_sl': self.risk_manager.current_stop_loss,
            'current_tp': self.risk_manager.current_take_profit,
            'actual_action_executed': self.actual_action_executed
        }
        info['trade_details'] = self.trade_details
        return info

    def _create_state_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of the current trading state for potential rollback.

        SECURITY FIX: Enables transaction rollback if trade execution fails partway through.

        Returns:
            Dictionary containing all state variables that could be modified during a trade.
        """
        return {
            'balance': self._balance,
            'stock_quantity': self.stock_quantity,
            'position_type': self._position_type,
            'entry_price': self.entry_price,
            'net_worth': self.net_worth,
            'total_fees_paid': self.total_fees_paid,
            'total_fees_paid_episode': self.total_fees_paid_episode,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_trades': self.total_trades,
            'current_hold_duration': self.current_hold_duration,
            'traded_value_step': self.traded_value_step,
            'transaction_cost_incurred_step': self.transaction_cost_incurred_step,
        }

    def _restore_state_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore trading state from a snapshot (rollback).

        SECURITY FIX: Reverts all state changes if trade execution fails.

        Args:
            snapshot: State snapshot created by _create_state_snapshot()
        """
        self._balance = snapshot['balance']
        self.stock_quantity = snapshot['stock_quantity']
        self._position_type = snapshot['position_type']
        self.entry_price = snapshot['entry_price']
        self.net_worth = snapshot['net_worth']
        self.total_fees_paid = snapshot['total_fees_paid']
        self.total_fees_paid_episode = snapshot['total_fees_paid_episode']
        self.winning_trades = snapshot['winning_trades']
        self.losing_trades = snapshot['losing_trades']
        self.total_trades = snapshot['total_trades']
        self.current_hold_duration = snapshot['current_hold_duration']
        self.traded_value_step = snapshot['traded_value_step']
        self.transaction_cost_incurred_step = snapshot['transaction_cost_incurred_step']
        import logging
        logging.getLogger(__name__).warning("Trade state rolled back due to execution failure")

    def _execute_trade(self, trade_type: str, trade_price: float, trade_quantity: float):
        """
        Exécute un trade (BUY ou SELL) avec gestion des coûts de transaction.

        VERSION CORRIGÉE - Capital-Agnostic:
        - Commission relative au capital (scalable de $100 à $100K+)
        - Frais proportionnels au trade value
        - Validation robuste des montants
        - Compatible avec tous les niveaux de capital
        - SECURITY FIX: Transaction rollback on failure

        Args:
            trade_type (str): 'buy' ou 'sell'
            trade_price (float): Prix d'exécution actuel du marché
            trade_quantity (float): Quantité à trader (en lots)

        Returns:
            tuple: (trade_success, effective_trade_value, commission, pnl_abs, pnl_pct)
                - trade_success (bool): True si le trade a été exécuté
                - effective_trade_value (float): Valeur brute du trade
                - commission (float): Commission totale payée
                - pnl_abs (float): P&L absolu en $ (pour SELL uniquement)
                - pnl_pct (float): P&L en % (pour SELL uniquement)
        """
        # SECURITY FIX: Create state snapshot for potential rollback
        state_snapshot = self._create_state_snapshot()

        # =========================================================================
        # INITIALISATION
        # =========================================================================
        trade_success = False
        effective_trade_value = 0.0
        commission = 0.0
        pnl_abs = 0.0
        pnl_pct = 0.0

        # Reset du tracking des coûts pour ce step
        self.traded_value_step = 0.0
        self.transaction_cost_incurred_step = 0.0

        try:
            # =====================================================================
            # BUY LOGIC
            # =====================================================================
            if trade_type == 'buy':
                # --- 1. Calcul du prix effectif avec spread et slippage ---
                # Le prix d'achat inclut le spread (défavorable pour l'acheteur)
                price_with_spread = trade_price * (1 + self.transaction_fee_percentage)
                effective_buy_price = price_with_spread * (1 + self.slippage_percentage)

                # --- 2. Valeur brute du trade (avant commission) ---
                gross_trade_value = effective_buy_price * trade_quantity

                # --- 3. NOUVEAU: Calcul de la commission relative (Capital-Agnostic) ---
                #
                # Commission composée de 2 parties:
                # a) Commission proportionnelle au trade (% de la valeur)
                # b) Commission minimum basée sur le capital (pour éviter trades trop petits)

                # a) Commission du trade (ex: 0.05% de $1,000 = $0.50)
                commission_pct_trade = self.trade_commission_pct_of_trade
                commission_from_trade = gross_trade_value * commission_pct_trade

                # b) Commission minimum relative au capital (ex: 0.01% de $1,000 = $0.10)
                commission_min_pct = self.trade_commission_min_pct_capital
                commission_minimum = self.initial_balance * commission_min_pct

                # La commission finale est le MAXIMUM des deux (protège contre trades trop petits)
                commission = max(commission_from_trade, commission_minimum)

                # --- 4. Coût total de l'achat ---
                total_cost = gross_trade_value + commission

                # --- 5. Validation: Vérifier si le trade est possible ---
                # Condition 1: Balance suffisante
                if self.balance < total_cost:
                    self.transaction_cost_incurred_step = 0.0
                    return False, 0.0, 0.0, 0.0, 0.0

                # Condition 2: Quantité minimale respectée
                if trade_quantity < self.min_trade_quantity:
                    self.transaction_cost_incurred_step = 0.0
                    return False, 0.0, 0.0, 0.0, 0.0

                # --- 6. Exécution du trade ---
                self.balance -= total_cost
                self.stock_quantity += trade_quantity
                self.entry_price = trade_price  # Prix d'entrée (utilisé pour calculer P&L)

                # --- 7. Mise à jour des compteurs ---
                self.total_fees_paid_episode += commission
                self.transaction_cost_incurred_step = commission
                effective_trade_value = trade_price * trade_quantity  # Valeur "propre" sans frais
                self.traded_value_step = effective_trade_value

                trade_success = True

                # --- 8. Logging optionnel ---
                if hasattr(self, 'verbose') and self.verbose:
                    logger.info(f"BUY Executed: {trade_quantity:.4f} @ ${trade_price:,.2f} | "
                               f"Cost: ${total_cost:,.2f} (Commission: ${commission:.2f})")

            # =====================================================================
            # SELL LOGIC
            # =====================================================================
            elif trade_type == 'sell':
                # --- 1. Validation: Vérifier si on a assez de stock ---
                if self.stock_quantity < trade_quantity or self.stock_quantity <= 1e-9:
                    self.transaction_cost_incurred_step = 0.0
                    return False, 0.0, 0.0, 0.0, 0.0

                # --- 2. Calcul du prix effectif avec spread et slippage ---
                # Le prix de vente subit le spread (défavorable pour le vendeur)
                price_with_spread = trade_price * (1 - self.transaction_fee_percentage)
                effective_sell_price = price_with_spread * (1 - self.slippage_percentage)

                # --- 3. Valeur brute du trade ---
                gross_trade_value = effective_sell_price * trade_quantity

                # --- 4. NOUVEAU: Calcul de la commission relative ---
                commission_pct_trade = self.trade_commission_pct_of_trade
                commission_from_trade = gross_trade_value * commission_pct_trade

                commission_min_pct = self.trade_commission_min_pct_capital
                commission_minimum = self.initial_balance * commission_min_pct

                commission = max(commission_from_trade, commission_minimum)

                # --- 5. Revenu net après commission ---
                total_revenue = gross_trade_value - commission

                # --- 6. Calcul du P&L (Profit & Loss) ---
                if not np.isnan(self.entry_price) and not math.isclose(self.entry_price, 0.0):
                    # Coût d'acquisition (prix d'achat * quantité)
                    asset_cost = self.entry_price * trade_quantity

                    # P&L absolu = Revenu - Coût
                    pnl_abs = total_revenue - asset_cost

                    # P&L en pourcentage
                    pnl_pct = (pnl_abs / asset_cost) * 100 if asset_cost > 1e-9 else 0.0
                else:
                    # Cas anormal: pas de prix d'entrée enregistré
                    pnl_abs = 0.0
                    pnl_pct = 0.0

                # --- 7. Exécution du trade ---
                self.balance += total_revenue
                self.stock_quantity -= trade_quantity

                # --- 8. Si position complètement fermée, reset ---
                if math.isclose(self.stock_quantity, 0.0, abs_tol=1e-9):
                    self.entry_price = np.nan
                    self.stock_quantity = 0.0  # Forcer à zéro exact

                    # Reset du risk manager (SL/TP)
                    if hasattr(self, 'risk_manager') and self.risk_manager is not None:
                        self.risk_manager.reset()

                # --- 9. Mise à jour des compteurs ---
                self.total_fees_paid_episode += commission
                self.transaction_cost_incurred_step = commission
                effective_trade_value = trade_price * trade_quantity
                self.traded_value_step = effective_trade_value

                trade_success = True

                # --- 10. Logging optionnel ---
                if hasattr(self, 'verbose') and self.verbose:
                    pnl_symbol = "+" if pnl_abs >= 0 else ""
                    logger.info(f"SELL Executed: {trade_quantity:.4f} @ ${trade_price:,.2f} | "
                               f"Revenue: ${total_revenue:,.2f} | P&L: {pnl_symbol}${pnl_abs:.2f} ({pnl_pct:+.2f}%)")

            else:
                # Type de trade invalide
                raise ValueError(f"Invalid trade_type: {trade_type}. Must be 'buy' or 'sell'.")

        except Exception as e:
            # =====================================================================
            # GESTION DES ERREURS - WITH TRANSACTION ROLLBACK
            # =====================================================================
            logger.error(f"ERROR _EXECUTE_TRADE: An error occurred during {trade_type.upper()} execution: {e}",
                         exc_info=True)

            # SECURITY FIX: Rollback state to pre-trade snapshot
            self._restore_state_snapshot(state_snapshot)

            # Reset des coûts en cas d'échec (already handled by rollback, but explicit)
            self.transaction_cost_incurred_step = 0.0
            self.traded_value_step = 0.0

            # Retourner des valeurs par défaut (trade échoué)
            return False, 0.0, 0.0, 0.0, 0.0

        # =========================================================================
        # RETOUR
        # =========================================================================
        return trade_success, effective_trade_value, commission, pnl_abs, pnl_pct

    def step(self, action: int):
        """
        Execute one step in the environment with LONG/SHORT support.

        NEW ACTION SPACE (5 actions):
            0 = HOLD         : Do nothing
            1 = OPEN_LONG    : Buy to open long position
            2 = CLOSE_LONG   : Sell to close long position
            3 = OPEN_SHORT   : Sell to open short position
            4 = CLOSE_SHORT  : Buy to cover short position

        Position States:
            FLAT (0)  : No position - can OPEN_LONG or OPEN_SHORT
            LONG (1)  : Holding long - can HOLD or CLOSE_LONG
            SHORT (-1): Holding short - can HOLD or CLOSE_SHORT
        """
        # --- 1. Track previous state for reward calculation ---
        previous_net_worth = self.net_worth
        self.previous_drawdown_level = self.peak_nav - previous_net_worth
        self.transaction_cost_incurred_step = 0.0
        self.traded_value_step = 0.0

        self.current_step += 1
        done = False
        truncated = False

        # --- CRITICAL DATA ACCESS (with error handling) ---
        try:
            current_row = self.df.iloc[self.current_step]
            current_market_price = float(current_row['Close'])
            current_atr = float(current_row.get('ATR', 0.0))  # Default if missing
            bos_signal = float(current_row.get('BOS_SIGNAL', 0.0))  # Default if missing
        except (IndexError, KeyError) as e:
            logger.error(f"Data access error at step {self.current_step}: {e}")
            # Return terminal state on data error
            return self._get_obs(), -20.0, True, False, {'error': str(e)}

        # Validate price data
        if np.isnan(current_market_price) or current_market_price <= 0:
            logger.error(f"Invalid price {current_market_price} at step {self.current_step}")
            return self._get_obs(), -20.0, True, False, {'error': 'invalid_price'}

        # --- Update risk manager regime ---
        regime_state = 0 if bos_signal != 0 else 1
        self.risk_manager.market_state['current_regime'] = regime_state

        # --- End of episode: Force close any position ---
        if self.current_step >= self.end_idx:
            truncated = True
            done = True
            # Force close any open position
            if self.position_type == POSITION_LONG and self.stock_quantity > self.min_trade_quantity:
                action = ACTION_CLOSE_LONG
            elif self.position_type == POSITION_SHORT and abs(self.stock_quantity) > self.min_trade_quantity:
                action = ACTION_CLOSE_SHORT
            else:
                action = ACTION_HOLD

        # --- Initialize trade details ---
        self.trade_details = {
            'trade_pnl_abs': 0.0, 'trade_pnl_pct': 0.0, 'trade_type': 'hold',
            'trade_success': False, 'trade_value': 0.0, 'commission': 0.0,
            'position_type': self.position_type
        }
        self.last_action = action
        self.actual_action_executed = action

        # --- ACTION VALIDATION: Convert invalid actions to HOLD ---
        # Can only OPEN_LONG or OPEN_SHORT when FLAT
        # Can only CLOSE_LONG when LONG
        # Can only CLOSE_SHORT when SHORT
        original_action = action
        invalid_reason = None

        if action == ACTION_OPEN_LONG and self.position_type != POSITION_FLAT:
            action = ACTION_HOLD
            invalid_reason = 'already_in_position'
            self.trade_details['trade_type'] = 'invalid_already_in_position'
        elif action == ACTION_OPEN_SHORT and self.position_type != POSITION_FLAT:
            action = ACTION_HOLD
            invalid_reason = 'already_in_position'
            self.trade_details['trade_type'] = 'invalid_already_in_position'
        elif action == ACTION_OPEN_SHORT and not self.enable_short_selling:
            action = ACTION_HOLD
            invalid_reason = 'short_selling_disabled'
            self.trade_details['trade_type'] = 'short_selling_disabled'
        elif action == ACTION_CLOSE_LONG and self.position_type != POSITION_LONG:
            action = ACTION_HOLD
            invalid_reason = 'no_long_position'
            self.trade_details['trade_type'] = 'invalid_no_long_position'
        elif action == ACTION_CLOSE_SHORT and self.position_type != POSITION_SHORT:
            action = ACTION_HOLD
            invalid_reason = 'no_short_position'
            self.trade_details['trade_type'] = 'invalid_no_short_position'

        # Log and count invalid actions
        # FIX "FEARFUL AGENT": Track invalid actions for reward penalty
        self.invalid_action_this_step = (invalid_reason is not None)

        if invalid_reason is not None:
            self.invalid_action_count += 1
            self.invalid_action_types[invalid_reason] += 1
            # Log every 100th invalid action to avoid spam (useful for debugging)
            if self.invalid_action_count % 100 == 1:
                logger.info(f"[INVALID ACTION] Step {self.current_step}: "
                            f"{ACTION_NAMES.get(original_action, original_action)} -> HOLD "
                            f"(reason: {invalid_reason}, total: {self.invalid_action_count})")

        # --- Check for SL/TP/TSL on active positions ---
        if not done and self.position_type != POSITION_FLAT and not np.isnan(self.entry_price):
            self.current_hold_duration += 1
            is_long = (self.position_type == POSITION_LONG)

            self.risk_manager.update_trailing_stop(
                self.entry_price, current_market_price, current_atr, is_long=is_long
            )
            exit_signal = self.risk_manager.check_trade_exit(current_market_price, is_long=is_long)

            if exit_signal == 'TP':
                action = ACTION_CLOSE_LONG if is_long else ACTION_CLOSE_SHORT
                self.actual_action_executed = 10  # TP exit code
            elif exit_signal == 'SL':
                action = ACTION_CLOSE_LONG if is_long else ACTION_CLOSE_SHORT
                self.actual_action_executed = 11  # SL exit code
        elif self.position_type == POSITION_FLAT:
            self.current_hold_duration = 0

        # --- Cooldown enforcement ---
        if action in [ACTION_OPEN_LONG, ACTION_CLOSE_LONG, ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT]:
            if (self.current_step - self.last_trade_step) < self.trade_cooldown_steps:
                self.actual_action_executed = ACTION_HOLD
                self.trade_details['trade_type'] = 'hold_cooldown'
                action = ACTION_HOLD

        # --- Apply borrowing fees for short positions ---
        if self.position_type == POSITION_SHORT and abs(self.stock_quantity) > 0:
            borrowing_fee = abs(self.stock_quantity) * current_market_price * self.short_borrowing_fee_daily
            self.balance -= borrowing_fee
            self.total_fees_paid_episode += borrowing_fee

        # --- Execute trade logic ---
        trade_executed = False
        if not done:
            # ========================================
            # OPEN LONG (Action == 1)
            # ========================================
            if action == ACTION_OPEN_LONG:
                trade_executed = self._execute_open_long(current_market_price, current_atr)

            # ========================================
            # CLOSE LONG (Action == 2)
            # ========================================
            elif action == ACTION_CLOSE_LONG:
                trade_executed = self._execute_close_long(current_market_price)

            # ========================================
            # OPEN SHORT (Action == 3)
            # ========================================
            elif action == ACTION_OPEN_SHORT:
                trade_executed = self._execute_open_short(current_market_price, current_atr)

            # ========================================
            # CLOSE SHORT (Action == 4)
            # ========================================
            elif action == ACTION_CLOSE_SHORT:
                trade_executed = self._execute_close_short(current_market_price)

            # ========================================
            # HOLD (Action == 0)
            # ========================================
            else:
                self.trade_details['trade_type'] = 'hold'
                self.actual_action_executed = ACTION_HOLD

        # --- Portfolio update (handles both long and short) ---
        self._update_portfolio_value(current_market_price)

        # --- Update Drawdown and Leverage Trackers ---
        self.peak_nav = max(self.peak_nav, self.net_worth)

        position_value = abs(self.stock_quantity) * current_market_price
        if self.net_worth > 1e-9:
            self.current_leverage = position_value / self.net_worth
        else:
            self.current_leverage = 0.0

        if self.net_worth <= self.minimum_allowed_balance:
            done = True

        # --- Calculate reward ---
        reward = self._calculate_reward(previous_net_worth)
        self.episode_reward += reward

        observation = self._get_obs()
        info = self._get_info()
        info['position_type'] = self.position_type
        info['position_type_name'] = {POSITION_FLAT: 'FLAT', POSITION_LONG: 'LONG', POSITION_SHORT: 'SHORT'}[self.position_type]

        # Invalid action tracking (helps monitor agent learning)
        info['invalid_action_count'] = self.invalid_action_count
        info['invalid_action_types'] = self.invalid_action_types.copy()

        self.previous_nav = self.net_worth

        return observation, reward, done, truncated, info

    def _execute_open_long(self, current_price: float, current_atr: float) -> bool:
        """Execute OPEN_LONG action (buy to open long position)."""
        sl_distance_abs = self.risk_manager.set_trade_orders(current_price, current_atr, is_long=True)

        # Calculate position size with HARD LEVERAGE ENFORCEMENT
        # The risk manager now caps position size to prevent exceeding max_leverage_limit
        trade_quantity_calc = self.risk_manager.calculate_adaptive_position_size(
            client_id=getattr(self, "_risk_client_id", "default_client"),
            account_equity=self.balance,
            atr_stop_distance=sl_distance_abs,
            win_prob=0.5,
            risk_reward_ratio=1.0,
            current_price=current_price,  # NEW: Pass price for leverage calc
            max_leverage=self.max_leverage_limit,  # NEW: Hard leverage limit
            is_long=True
        )

        try:
            trade_quantity_calc = float(trade_quantity_calc)
        except Exception:
            trade_quantity_calc = 0.0

        if trade_quantity_calc < self.min_trade_quantity:
            trade_quantity_calc = 0.0

        estimated_cost = current_price * trade_quantity_calc * (
            1 + self.transaction_fee_percentage + self.slippage_percentage
        )
        if estimated_cost > self.balance and estimated_cost > 0:
            scale = self.balance / estimated_cost
            trade_quantity_calc *= scale

        if trade_quantity_calc < self.min_trade_quantity:
            self.trade_details['trade_type'] = 'open_long_failed'
            self.actual_action_executed = ACTION_HOLD
            return False

        trade_success, value, commission, _, _ = self._execute_trade(
            'buy', current_price, trade_quantity=trade_quantity_calc
        )

        if trade_success:
            self.position_type = POSITION_LONG
            self.last_trade_step = self.current_step
            self.trade_details.update({
                'trade_success': True, 'trade_type': 'open_long',
                'trade_value': value, 'commission': commission,
                'quantity': trade_quantity_calc
            })
            self.current_hold_duration = 1

            if self.trade_logger:
                self.trade_id_counter += 1
                self.trade_logger.log_trade({
                    'trade_id': self.trade_id_counter,
                    'trade_type': 'open_long',
                    'step': self.current_step,
                    'price': current_price,
                    'quantity': trade_quantity_calc,
                    'balance': self.balance,
                    'net_worth': self.net_worth
                })
            return True
        else:
            self.trade_details['trade_type'] = 'open_long_failed'
            self.actual_action_executed = ACTION_HOLD
            return False

    def _execute_close_long(self, current_price: float) -> bool:
        """Execute CLOSE_LONG action (sell to close long position)."""
        if self.stock_quantity <= self.min_trade_quantity:
            self.actual_action_executed = ACTION_HOLD
            self.trade_details['trade_type'] = 'close_long_no_position'
            return False

        trade_success, value, commission, pnl_abs, pnl_pct = self._execute_trade(
            'sell', current_price, self.stock_quantity
        )

        if trade_success:
            self.position_type = POSITION_FLAT
            self.last_trade_step = self.current_step
            self.trade_history_summary.append({
                'step': self.current_step,
                'pnl_abs': pnl_abs, 'pnl_pct': pnl_pct, 'type': 'close_long'
            })
            self.total_trades += 1
            if pnl_abs > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            self.current_hold_duration = 0

            self.trade_details.update({
                'trade_success': True, 'trade_type': 'close_long',
                'trade_value': value, 'commission': commission,
                'trade_pnl_abs': pnl_abs, 'trade_pnl_pct': pnl_pct
            })

            if self.trade_logger:
                self.trade_logger.log_trade({
                    'trade_id': self.trade_id_counter,
                    'trade_type': 'close_long',
                    'step': self.current_step,
                    'price': current_price,
                    'pnl_abs': pnl_abs,
                    'pnl_pct': pnl_pct,
                    'balance': self.balance,
                    'net_worth': self.net_worth,
                    'duration_bars': self.current_hold_duration
                })
            return True
        else:
            self.trade_details['trade_type'] = 'close_long_failed'
            return False

    def _execute_open_short(self, current_price: float, current_atr: float) -> bool:
        """
        Execute OPEN_SHORT action (sell to open short position).

        Short selling mechanics:
        1. Borrow asset from broker
        2. Sell at current price (receive cash)
        3. Later: buy back to return to broker
        4. Profit if price goes DOWN, loss if price goes UP
        """
        sl_distance_abs = self.risk_manager.set_trade_orders(current_price, current_atr, is_long=False)

        # Calculate position size with HARD LEVERAGE ENFORCEMENT
        # The risk manager now caps position size to prevent exceeding max_leverage_limit
        trade_quantity_calc = self.risk_manager.calculate_adaptive_position_size(
            client_id=getattr(self, "_risk_client_id", "default_client"),
            account_equity=self.balance,
            atr_stop_distance=sl_distance_abs,
            win_prob=0.5,
            risk_reward_ratio=1.0,
            current_price=current_price,  # NEW: Pass price for leverage calc
            max_leverage=self.max_leverage_limit,  # NEW: Hard leverage limit
            is_long=False  # Short position
        )

        try:
            trade_quantity_calc = float(trade_quantity_calc)
        except Exception:
            trade_quantity_calc = 0.0

        if trade_quantity_calc < self.min_trade_quantity:
            self.trade_details['trade_type'] = 'open_short_failed'
            self.actual_action_executed = ACTION_HOLD
            return False

        # For shorts, we receive cash when opening (sell borrowed asset)
        # Apply spread and slippage (unfavorable for seller)
        price_with_spread = current_price * (1 - self.transaction_fee_percentage)
        effective_sell_price = price_with_spread * (1 - self.slippage_percentage)
        gross_value = effective_sell_price * trade_quantity_calc

        # Calculate commission
        commission_pct_trade = self.trade_commission_pct_of_trade
        commission = gross_value * commission_pct_trade

        # Short position: negative stock quantity, cash increases
        self.balance += (gross_value - commission)
        self.stock_quantity = -trade_quantity_calc  # Negative = short position
        self.entry_price = current_price
        self.position_type = POSITION_SHORT
        self.last_trade_step = self.current_step

        self.total_fees_paid_episode += commission
        self.transaction_cost_incurred_step = commission
        self.traded_value_step = current_price * trade_quantity_calc

        self.trade_details.update({
            'trade_success': True, 'trade_type': 'open_short',
            'trade_value': gross_value, 'commission': commission,
            'quantity': trade_quantity_calc
        })
        self.current_hold_duration = 1

        if self.trade_logger:
            self.trade_id_counter += 1
            self.trade_logger.log_trade({
                'trade_id': self.trade_id_counter,
                'trade_type': 'open_short',
                'step': self.current_step,
                'price': current_price,
                'quantity': trade_quantity_calc,
                'balance': self.balance,
                'net_worth': self.net_worth
            })
        return True

    def _execute_close_short(self, current_price: float) -> bool:
        """
        Execute CLOSE_SHORT action (buy to cover short position).

        Closing a short:
        1. Buy back the asset at current price
        2. Return to broker
        3. P&L = entry_price - exit_price (profit if price went down)
        """
        if abs(self.stock_quantity) <= self.min_trade_quantity or self.position_type != POSITION_SHORT:
            self.actual_action_executed = ACTION_HOLD
            self.trade_details['trade_type'] = 'close_short_no_position'
            return False

        quantity_to_cover = abs(self.stock_quantity)

        # Buy to cover: pay current price + spread + slippage
        price_with_spread = current_price * (1 + self.transaction_fee_percentage)
        effective_buy_price = price_with_spread * (1 + self.slippage_percentage)
        gross_cost = effective_buy_price * quantity_to_cover

        # Calculate commission
        commission_pct_trade = self.trade_commission_pct_of_trade
        commission = gross_cost * commission_pct_trade

        total_cost = gross_cost + commission

        # Check if we can afford to close
        if total_cost > self.balance:
            self.trade_details['trade_type'] = 'close_short_insufficient_funds'
            return False

        # Calculate P&L for short: profit if price went DOWN
        # P&L = (entry_price - exit_price) * quantity
        pnl_abs = (self.entry_price - current_price) * quantity_to_cover - commission
        if self.entry_price > 0:
            pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
        else:
            pnl_pct = 0.0

        # Execute: pay to cover, clear position
        self.balance -= total_cost
        self.stock_quantity = 0.0
        self.position_type = POSITION_FLAT
        self.last_trade_step = self.current_step
        self.entry_price = np.nan

        self.total_fees_paid_episode += commission
        self.transaction_cost_incurred_step = commission
        self.traded_value_step = current_price * quantity_to_cover

        # Record trade
        self.trade_history_summary.append({
            'step': self.current_step,
            'pnl_abs': pnl_abs, 'pnl_pct': pnl_pct, 'type': 'close_short'
        })
        self.total_trades += 1
        if pnl_abs > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        self.current_hold_duration = 0

        self.trade_details.update({
            'trade_success': True, 'trade_type': 'close_short',
            'trade_value': gross_cost, 'commission': commission,
            'trade_pnl_abs': pnl_abs, 'trade_pnl_pct': pnl_pct
        })

        if self.trade_logger:
            self.trade_logger.log_trade({
                'trade_id': self.trade_id_counter,
                'trade_type': 'close_short',
                'step': self.current_step,
                'price': current_price,
                'pnl_abs': pnl_abs,
                'pnl_pct': pnl_pct,
                'balance': self.balance,
                'net_worth': self.net_worth,
                'duration_bars': self.current_hold_duration
            })
        return True

    def _update_portfolio_value(self, current_price: float) -> None:
        """
        Update net worth accounting for both long and short positions.

        For LONG positions:
            net_worth = balance + (quantity * price)

        For SHORT positions:
            net_worth = balance - (quantity * price) + (entry_quantity * entry_price)
            Simplified: net_worth = balance + unrealized_pnl
            where unrealized_pnl = (entry_price - current_price) * |quantity|
        """
        if self.position_type == POSITION_LONG:
            # Long: we own the asset, value increases with price
            self.net_worth = self.balance + (self.stock_quantity * current_price)

        elif self.position_type == POSITION_SHORT:
            # Short: we owe the asset, value increases when price decreases
            # unrealized P&L = (entry_price - current_price) * quantity
            quantity = abs(self.stock_quantity)
            if not np.isnan(self.entry_price):
                unrealized_pnl = (self.entry_price - current_price) * quantity
            else:
                unrealized_pnl = 0.0
            # When we opened short, we received: entry_price * quantity
            # To close, we need to pay: current_price * quantity
            # Net worth = what we have - what we owe = balance + unrealized_pnl
            self.net_worth = self.balance + unrealized_pnl

        else:  # FLAT
            self.net_worth = self.balance

    def _calculate_reward(self, previous_net_worth: float) -> float:
        """
        RISK-ADJUSTED REWARD FUNCTION (Sprint 2 Restructuring)
        ======================================================

        Fixes from audit:
        - Holding flat is NEUTRAL (0.0), not penalized
        - Holding profitable positions is REWARDED (let winners run)
        - Losses get CONVEX penalty (bigger losses hurt exponentially more)
        - Trade quality measured by risk-reward ratio, not flat bonus
        - Turnover and friction are penalized to prevent churning

        Expected reward range: [-20, +20]
        Typical rewards: [-5, +5]

        Returns:
            float: Scaled reward for PPO training
        """
        import logging
        _logger = logging.getLogger(__name__)

        # =========================================================================
        # STEP 1: VALIDATION
        # =========================================================================
        if previous_net_worth <= 1e-9 or self.net_worth <= 1e-9:
            return -20.0

        if np.isnan(self.net_worth) or np.isinf(self.net_worth):
            _logger.warning("Invalid net_worth: %s at step %d", self.net_worth, self.current_step)
            return -20.0

        if np.isnan(previous_net_worth) or np.isinf(previous_net_worth):
            _logger.warning("Invalid previous_net_worth: %s at step %d", previous_net_worth, self.current_step)
            return 0.0

        # =========================================================================
        # STEP 2: CORE PROFITABILITY (log-return scaled to 1% = 1.0)
        # =========================================================================
        log_return = np.log(self.net_worth / previous_net_worth)
        profitability_reward = log_return * 100.0

        # =========================================================================
        # STEP 3: RISK-ADJUSTED PENALTIES
        # =========================================================================
        total_penalty = 0.0
        dd_penalty = 0.0
        friction_penalty = 0.0
        leverage_penalty = 0.0
        turnover_penalty = 0.0

        # --- Penalty A: Drawdown Increase (ONLY new drawdown) ---
        current_drawdown = self.peak_nav - self.net_worth
        drawdown_increase = max(0.0, current_drawdown - self.previous_drawdown_level)

        if drawdown_increase > 0:
            dd_penalty = (drawdown_increase / self.initial_balance) * 10.0
            total_penalty += dd_penalty

        # --- Penalty B: Transaction Costs (Friction) ---
        if self.transaction_cost_incurred_step > 0:
            friction_penalty = (self.transaction_cost_incurred_step / self.initial_balance) * 3.0
            total_penalty += friction_penalty

        # --- Penalty C: Leverage Violation (Quadratic) ---
        leverage_excess = max(0.0, self.current_leverage - self.max_leverage_limit)
        if leverage_excess > 0:
            leverage_penalty = (leverage_excess ** 2) * 10.0
            total_penalty += leverage_penalty

        # --- Penalty D: Turnover (anti-churning) ---
        if self.traded_value_step > 0:
            turnover_ratio = self.traded_value_step / max(self.net_worth, 1.0)
            if turnover_ratio > 0.3:
                turnover_penalty = (turnover_ratio - 0.3) * 3.0
                total_penalty += turnover_penalty

        # --- Penalty E: Invalid Actions ---
        if getattr(self, 'invalid_action_this_step', False):
            total_penalty += 0.5

        # NOTE: No hold penalty. Holding flat = reward 0.0 (neutral).
        # NOTE: No duration penalty. Removed to let winners run.

        # =========================================================================
        # STEP 4: POSITION HOLDING BONUS/PENALTY (unrealized P&L feedback)
        # =========================================================================
        hold_reward = 0.0

        if self.position_type != POSITION_FLAT and not np.isnan(self.entry_price):
            # Calculate unrealized P&L
            if self.position_type == POSITION_LONG:
                current_price = self.net_worth - self.balance
                if self.stock_quantity > 0:
                    current_price = current_price / self.stock_quantity
                else:
                    current_price = self.entry_price
                unrealized_pnl = (current_price - self.entry_price) * self.stock_quantity
            else:  # SHORT
                quantity = abs(self.stock_quantity)
                # For short: unrealized_pnl already computed in net_worth
                unrealized_pnl = self.net_worth - self.balance

            unrealized_pnl_pct = unrealized_pnl / self.initial_balance

            if unrealized_pnl > 0:
                # Profitable hold: reward to encourage letting winners run
                hold_reward = min(0.5, unrealized_pnl_pct * 100)
            else:
                # Losing hold: convex penalty (bigger losses hurt more)
                loss_pct = abs(unrealized_pnl_pct)
                hold_reward = max(-5.0, -(loss_pct * 100) ** 1.5)

        # =========================================================================
        # STEP 5: TRADE CLOSE BONUS (Risk-Reward Based)
        # =========================================================================
        trade_bonus = 0.0

        closed_trade_types = ['sell', 'close_long', 'close_short']
        if self.trade_details.get('trade_type') in closed_trade_types and self.trade_details.get('trade_success'):
            trade_pnl_abs = self.trade_details.get('trade_pnl_abs', 0.0)
            trade_pnl_pct = self.trade_details.get('trade_pnl_pct', 0.0)

            if trade_pnl_abs > 0:
                # Winning trade: reward proportional to risk-reward quality
                # Use pnl_pct as proxy for RR (SL is ~1% per config)
                actual_rr = abs(trade_pnl_pct) / max(self.stop_loss_percentage * 100, 0.1)
                trade_bonus = min(3.0, actual_rr)
            else:
                # Losing trade: fixed mild penalty (PnL itself already negative)
                trade_bonus = -0.5

        # =========================================================================
        # STEP 6: COMPOSITE + NORMALIZATION (Tanh Squashing)
        # =========================================================================
        raw_reward = profitability_reward - total_penalty + hold_reward + trade_bonus
        combined_reward = raw_reward

        normalized_reward = np.tanh(combined_reward * self.reward_tanh_scale)
        scaled_reward = normalized_reward * self.reward_output_scale

        # =========================================================================
        # STEP 7: SPECIAL CASES (Terminal Conditions)
        # =========================================================================
        if self.net_worth <= self.minimum_allowed_balance:
            return -20.0

        if current_drawdown > 0:
            dd_ratio = current_drawdown / self.peak_nav
            if dd_ratio > 0.15:
                scaled_reward -= 5.0

        # =========================================================================
        # STEP 8: FINAL SAFETY CLIPPING
        # =========================================================================
        final_reward = np.clip(scaled_reward, -20.0, 20.0)

        # =========================================================================
        # STEP 9: REWARD COMPONENT TRACKING (for TensorBoard logging)
        # =========================================================================
        self._last_reward_components = {
            'profitability': profitability_reward,
            'dd_penalty': dd_penalty,
            'friction_penalty': friction_penalty,
            'leverage_penalty': leverage_penalty,
            'turnover_penalty': turnover_penalty,
            'hold_reward': hold_reward,
            'trade_bonus': trade_bonus,
            'raw_reward': raw_reward,
            'final_reward': final_reward,
        }

        # Optional debug logging (via logger, not print)
        if hasattr(self, 'debug_rewards') and self.debug_rewards:
            _logger.debug(
                "REWARD Step %d | pnl=%.3f dd=%.3f fric=%.3f hold=%.3f trade=%.3f -> %.3f",
                self.current_step, profitability_reward, dd_penalty,
                friction_penalty, hold_reward, trade_bonus, final_reward
            )

        # Final NaN/Inf check
        if np.isnan(final_reward) or np.isinf(final_reward):
            _logger.warning("Invalid final_reward: %s at step %d", final_reward, self.current_step)
            return 0.0

        return final_reward



    def render(self):
        console = Console()
        table = Table(show_header=True, header_style="bold magenta", show_lines=False)

        # Columns
        table.add_column("Step", style="cyan")
        table.add_column("Time", style="white")
        table.add_column("Net Worth", justify="right")
        table.add_column("Δ%", justify="right")
        table.add_column("Balance", justify="right")
        table.add_column("Stock", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("SL", justify="right")
        table.add_column("TP", justify="right")
        table.add_column("TSL", justify="center")
        table.add_column("Fees", justify="right")
        table.add_column("Action", justify="center")

        # Safe time
        try:
            current_time = pd.to_datetime(self.df.index)[self.current_step].strftime('%Y-%m-%d %H:%M')
        except Exception:
            if 'Gmt time' in self.df.columns:
                current_time = pd.to_datetime(
                    self.df.iloc[self.current_step]['Gmt time']
                ).strftime('%Y-%m-%d %H:%M')
            else:
                current_time = f"{self.current_step}"

        # Compute metrics
        net_worth_change = ((self.net_worth - self.initial_balance) / self.initial_balance) * 100
        pnl_color = "green" if net_worth_change >= 0 else "red"
        current_sl = getattr(self.risk_manager, "current_stop_loss", np.nan)
        current_tp = getattr(self.risk_manager, "current_take_profit", np.nan)
        tsl_active = getattr(self.risk_manager, "tsl_activated", False)
        entry_display = (
            f"{self.entry_price:,.2f}" if not math.isclose(self.stock_quantity, 0.0, abs_tol=1e-9) else "—"
        )

        table.add_row(
            str(self.current_step),
            current_time,
            f"${self.net_worth:,.2f}",
            Text(f"{net_worth_change:+.2f}%", style=pnl_color),
            f"${self.balance:,.2f}",
            f"{self.stock_quantity:.4f}",
            entry_display,
            f"${current_sl:,.2f}",
            f"${current_tp:,.2f}",
            "✅" if tsl_active else "—",
            f"${self.total_fees_paid:,.2f}",
            self.trade_details.get("action_taken", "N/A"),
        )

        console.print(table)

    def close(self):
        logger.info("DEBUG ENV: Environment closed.")

    def reset(self, seed: int = None, options: dict = None):
        """
        Resets the trading environment to its initial state before a new episode.
        Returns the first observation and info dict (Gymnasium API standard).
        """
        super().reset(seed=seed)

        # --- 1️⃣ Reset financial state ---
        self.balance = float(self.initial_balance)
        self.net_worth = float(self.initial_balance)
        self.stock_quantity = 0.0
        self.entry_price = np.nan
        self.position_type = POSITION_FLAT  # NEW: Reset position type for long/short

        # --- 2️⃣ Reset Risk Manager and statistics ---
        if hasattr(self, "risk_manager") and self.risk_manager is not None:
            self.risk_manager.reset()

        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees_paid_episode = 0.0
        self.trade_history_summary = deque(maxlen=1000)
        self.history = deque(maxlen=1000)

        # --- 3️⃣ Reset behavioral trackers ---
        self.last_action = 0
        self.actual_action_executed = 0

        # Reset invalid action counters
        self.invalid_action_count = 0
        self.invalid_action_types = {
            'already_in_position': 0,
            'no_long_position': 0,
            'no_short_position': 0,
            'short_selling_disabled': 0
        }

        # --- 4️⃣ Advanced reward trackers ---
        self.previous_nav = self.initial_balance
        self.peak_nav = self.initial_balance
        self.previous_drawdown_level = 0.0
        self.current_leverage = 0.0
        self.current_hold_duration = 0
        self.transaction_cost_incurred_step = 0.0
        self.traded_value_step = 0.0

        # --- 5️⃣ Episode boundaries ---
        min_possible_step = int(self.lookback_window_size - 1)

        if self.use_fixed_episode_length:
            # FIXED EPISODE LENGTH MODE (Recommended for PPO stability)
            # Ensures all episodes have the same length for consistent gradient estimates
            episode_length = self.fixed_episode_length

            # Calculate valid start range: must have enough room for fixed episode + lookback
            max_valid_start = int(len(self.df) - 1 - episode_length)

            if max_valid_start <= min_possible_step:
                raise ValueError(
                    f"❌ Not enough data for fixed episode length ({episode_length}) "
                    f"with lookback ({self.lookback_window_size}). Data length: {len(self.df)}"
                )

            # Random start within valid range
            self.start_idx = int(
                self.np_random.integers(min_possible_step, max_valid_start + 1)
            )

            # Fixed episode length
            self.max_steps = episode_length
            self.end_idx = int(self.start_idx + self.max_steps)
        else:
            # VARIABLE EPISODE LENGTH MODE (Original behavior - not recommended)
            max_valid_start_idx_for_obs = int(len(self.df) - 1 - min_possible_step)

            if max_valid_start_idx_for_obs <= min_possible_step:
                raise ValueError(
                    f"❌ Not enough data to form a valid episode with the given lookback window "
                    f"({self.lookback_window_size}). Data length: {len(self.df)}"
                )

            # Random start with variable length (original behavior)
            self.start_idx = int(
                self.np_random.integers(min_possible_step, max_valid_start_idx_for_obs + 1)
            )
            self.max_steps = int(len(self.df) - 1 - self.start_idx)
            self.end_idx = int(self.start_idx + self.max_steps)

        # Ensure step indices are pure Python ints
        self.current_step = int(self.start_idx)

        # --- 6️⃣ Initial observation and info ---
        observation = self._get_obs()
        info = self._get_info()
        if self.trade_logger and self.episode_count > 0:
            self.trade_logger.log_episode_summary({
                'episode': self.episode_count,
                'total_reward': self.episode_reward,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'final_balance': self.balance
            })
        self.episode_count += 1
        self.episode_reward = 0.0
        # Optional: early print or debug log
        # print(f"🔁 Environment reset at index {self.start_idx} (Step range: {self.start_idx}-{self.end_idx})")

        return observation, info
