# =============================================================================
# CONFIG LOADER
# =============================================================================
# Lazy configuration loading with explicit initialization.
#
# This module solves several problems with direct config imports:
# 1. No side effects at import time (directories not created)
# 2. Allows configuration overrides for testing
# 3. Validates configuration before use
# 4. Provides a clear initialization point
#
# Usage:
#   from src.core.config_loader import get_config, initialize_config
#
#   # At application startup
#   initialize_config()  # Creates directories, validates config
#
#   # Anywhere else
#   config = get_config()
#   print(config.INITIAL_BALANCE)
#
#   # For testing
#   initialize_config(overrides={'INITIAL_BALANCE': 10000.0})
#
# =============================================================================

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from threading import Lock

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION STATE
# =============================================================================

class ConfigurationError(Exception):
    """Raised when configuration is invalid or not initialized."""
    pass


@dataclass
class TradingConfig:
    """
    Centralized configuration container.

    All configuration values are stored here with type hints
    for IDE support and documentation.
    """

    # =========================================================================
    # 1. CORE SYSTEM SETTINGS
    # =========================================================================
    LOGGING_LEVEL: int = logging.INFO
    TRADING_DAYS_YEAR: int = 252
    RISK_FREE_RATE: float = 0.02

    # Paths (set dynamically during initialization)
    PROJECT_ROOT: str = ""
    SRC_DIR: str = ""
    DATA_DIR: str = ""
    MODEL_DIR: str = ""
    LOG_DIR: str = ""
    RESULTS_DIR: str = ""
    REPORTS_DIR: str = ""
    TENSORBOARD_LOG_DIR: str = ""
    CHARTS_DIR: str = ""

    # =========================================================================
    # 2. DATA CONFIGURATION
    # =========================================================================
    HISTORICAL_DATA_FILE: str = ""

    OHLCV_COLUMNS: Dict[str, str] = field(default_factory=lambda: {
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # =========================================================================
    # 3. ENVIRONMENT & FEATURES
    # =========================================================================
    ACTION_SPACE_TYPE: str = 'discrete'
    LOOKBACK_WINDOW_SIZE: int = 20
    FIXED_EPISODE_LENGTH: int = 500
    USE_FIXED_EPISODE_LENGTH: bool = True
    NUM_ACTIONS: int = 5

    # Action constants
    ACTION_HOLD: int = 0
    ACTION_OPEN_LONG: int = 1
    ACTION_CLOSE_LONG: int = 2
    ACTION_OPEN_SHORT: int = 3
    ACTION_CLOSE_SHORT: int = 4

    # Position constants
    POSITION_FLAT: int = 0
    POSITION_LONG: int = 1
    POSITION_SHORT: int = -1

    # =========================================================================
    # 4. RISK MANAGEMENT
    # =========================================================================
    RISK_PERCENTAGE_PER_TRADE: float = 0.01
    TAKE_PROFIT_PERCENTAGE: float = 0.02
    STOP_LOSS_PERCENTAGE: float = 0.01
    TSL_START_PROFIT_MULTIPLIER: float = 1.0
    TSL_TRAIL_DISTANCE_MULTIPLIER: float = 0.5
    GARCH_UPDATE_FREQUENCY: int = 500

    # Short selling
    ENABLE_SHORT_SELLING: bool = True
    SHORT_BORROWING_FEE_DAILY: float = 0.0001
    SHORT_MARGIN_REQUIREMENT: float = 1.0
    SHORT_MAX_POSITION_SIZE_PCT: float = 0.20
    SHORT_MAX_HOLDING_DURATION: int = 40
    SHORT_OVERNIGHT_SWAP_PCT: float = 0.0002

    MAX_DRAWDOWN_LIMIT_PCT: float = 10.0

    # =========================================================================
    # 5. TRADING PARAMETERS
    # =========================================================================
    INITIAL_BALANCE: float = 1000.0
    MINIMUM_ALLOWED_BALANCE: float = 100.0
    ALLOW_NEGATIVE_BALANCE: bool = False
    MIN_TRADE_QUANTITY: float = 0.01

    TRANSACTION_FEE_PERCENTAGE: float = 0.0005
    SLIPPAGE_PERCENTAGE: float = 0.0001
    TRADE_COMMISSION_PER_TRADE: float = 0.0005
    TRADE_COMMISSION_PCT_OF_TRADE: float = 0.0005
    TRADE_COMMISSION_MIN_PCT_CAPITAL: float = 0.0001

    # =========================================================================
    # 6. REWARD FUNCTION
    # =========================================================================
    REWARD_SCALING_FACTOR: float = 100.0
    REWARD_TANH_SCALE: float = 0.3
    REWARD_OUTPUT_SCALE: float = 5.0

    LOSING_TRADE_PENALTY: float = 0.0
    WINNING_TRADE_BONUS: float = 2.0
    W_FRICTION: float = 0.1
    W_RETURN: float = 1.0
    W_DRAWDOWN: float = 0.5
    W_LEVERAGE: float = 1.0
    W_TURNOVER: float = 0.0
    W_DURATION: float = 0.1

    DOWNSIDE_PENALTY_MULTIPLIER: float = 1.0
    HOLD_PENALTY_FACTOR: float = 0.01
    FAILED_TRADE_ATTEMPT_PENALTY: float = 0.0
    TRADE_COOLDOWN_STEPS: int = 2
    RAPID_TRADE_PENALTY: float = 1.0

    RISK_REJECTION_PENALTY: float = 2.0
    RISK_MODIFICATION_PENALTY: float = 0.5
    NEWS_BLOCK_PENALTY: float = 1.0
    MODIFICATION_THRESHOLD: float = 0.5

    MAX_LEVERAGE: float = 1.0
    MAX_DURATION_STEPS: int = 40

    # =========================================================================
    # 7. TRAINING
    # =========================================================================
    MODEL_NAME: str = "PPO_XAU_DayTrader_Production_v2"
    RANDOM_SEED: int = 42
    RENDER_MODE: str = "none"
    TOTAL_TIMESTEPS_PER_BOT: int = 1500000
    EARLY_STOPPING_PATIENCE: int = 5
    EVAL_FREQ: int = 10000
    N_EVAL_EPISODES: int = 5

    N_PARALLEL_BOTS: int = 50
    MAX_WORKERS_GPU: int = 2
    EVALUATION_METRIC: str = 'sharpe_ratio'

    MIN_ACCEPTABLE_SHARPE: float = 1.5
    MIN_ACCEPTABLE_CALMAR: float = 2.0
    MAX_ACCEPTABLE_DD: float = 0.15

    # =========================================================================
    # 8. LOGGING & MONITORING
    # =========================================================================
    ENABLE_TENSORBOARD: bool = True
    ENABLE_WANDB: bool = False
    WANDB_PROJECT: str = "XAU_Trading_Bot_Production"
    WANDB_ENTITY: str = "your_username"

    REPORT_FORMAT: str = 'html'
    INCLUDE_CHARTS: bool = True
    INCLUDE_TRADE_LOG: bool = True
    PROFIT_CURRENCY: str = 'USD'

    # =========================================================================
    # 9. PRODUCTION
    # =========================================================================
    DEPLOYMENT_MIN_CAPITAL: float = 5000.0
    DEPLOYMENT_MAX_LEVERAGE: float = 1.0
    DEPLOYMENT_MONITORING_INTERVAL: int = 300
    AUTO_SHUTDOWN_ON_DD: bool = True
    EMERGENCY_CONTACT_EMAIL: str = "your_email@example.com"

    LIVE_RISK_PER_TRADE: float = 0.005
    LIVE_MAX_DRAWDOWN: float = 0.08
    LIVE_MAX_DAILY_TRADES: int = 10

    # =========================================================================
    # 10. WALK-FORWARD & NEWS FILTER
    # =========================================================================
    USE_WALK_FORWARD: bool = True
    USE_NEWS_FILTER: bool = True

    # Initialized flag
    _initialized: bool = False


# =============================================================================
# GLOBAL STATE (Thread-safe singleton)
# =============================================================================

_config: Optional[TradingConfig] = None
_config_lock = Lock()
_directories_created = False


def get_config() -> TradingConfig:
    """
    Get the current configuration.

    Returns:
        TradingConfig instance

    Raises:
        ConfigurationError: If configuration not initialized
    """
    global _config

    if _config is None or not _config._initialized:
        raise ConfigurationError(
            "Configuration not initialized. "
            "Call initialize_config() at application startup."
        )

    return _config


def is_initialized() -> bool:
    """Check if configuration has been initialized."""
    global _config
    return _config is not None and _config._initialized


def initialize_config(
    overrides: Optional[Dict[str, Any]] = None,
    create_directories: bool = True,
    validate: bool = True,
    strict_mode: bool = False,
    project_root: Optional[str] = None
) -> TradingConfig:
    """
    Initialize configuration with optional overrides.

    This is the single point of configuration initialization.
    Call this once at application startup.

    Args:
        overrides: Dictionary of config values to override
        create_directories: Whether to create required directories
        validate: Whether to validate configuration
        strict_mode: Whether to fail on warnings
        project_root: Override project root path (useful for testing)

    Returns:
        Initialized TradingConfig

    Raises:
        ConfigurationError: If validation fails
    """
    global _config, _directories_created

    with _config_lock:
        # Create new config instance
        _config = TradingConfig()

        # Set paths
        if project_root:
            root = project_root
        else:
            # Auto-detect from config_loader.py location
            # config_loader.py is in src/core/, so go up 2 levels
            root = str(Path(__file__).parent.parent.parent)

        _config.PROJECT_ROOT = root
        _config.SRC_DIR = os.path.join(root, 'src')
        _config.DATA_DIR = os.path.join(root, 'data')
        _config.MODEL_DIR = os.path.join(root, 'trained_models')
        _config.LOG_DIR = os.path.join(root, 'logs')
        _config.RESULTS_DIR = os.path.join(root, 'results')
        _config.REPORTS_DIR = os.path.join(_config.RESULTS_DIR, 'training_reports')
        _config.TENSORBOARD_LOG_DIR = os.path.join(_config.LOG_DIR, 'tensorboard')
        _config.CHARTS_DIR = os.path.join(_config.RESULTS_DIR, 'performance_charts')
        _config.HISTORICAL_DATA_FILE = os.path.join(_config.DATA_DIR, "XAU_15MIN_2019_2025.csv")

        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(_config, key):
                    setattr(_config, key, value)
                else:
                    logger.warning(f"Unknown config key: {key}")

        # Create directories if requested
        if create_directories and not _directories_created:
            _create_directories(_config)
            _directories_created = True

        # Validate if requested
        if validate:
            _validate_config(_config, strict_mode)

        _config._initialized = True
        logger.info("Configuration initialized successfully")

        return _config


def reset_config() -> None:
    """
    Reset configuration to uninitialized state.

    Primarily used for testing.
    """
    global _config, _directories_created

    with _config_lock:
        _config = None
        _directories_created = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _create_directories(config: TradingConfig) -> None:
    """Create required directories."""
    directories = [
        config.DATA_DIR,
        config.MODEL_DIR,
        config.LOG_DIR,
        config.RESULTS_DIR,
        config.REPORTS_DIR,
        config.TENSORBOARD_LOG_DIR,
        config.CHARTS_DIR,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directory ensured: {directory}")


def _validate_config(config: TradingConfig, strict_mode: bool) -> None:
    """
    Validate configuration values.

    Raises:
        ConfigurationError: If validation fails
    """
    errors = []
    warnings = []

    # Security: Check placeholder values
    if config.EMERGENCY_CONTACT_EMAIL == "your_email@example.com":
        if strict_mode:
            errors.append(
                "SECURITY ERROR: EMERGENCY_CONTACT_EMAIL contains placeholder. "
                "Set to a real value in production."
            )
        else:
            warnings.append(
                "SECURITY WARNING: EMERGENCY_CONTACT_EMAIL is placeholder. "
                "Set before production deployment."
            )

    # Risk percentage bounds
    if config.RISK_PERCENTAGE_PER_TRADE <= 0:
        errors.append(
            f"RISK_PERCENTAGE_PER_TRADE must be positive, "
            f"got {config.RISK_PERCENTAGE_PER_TRADE}"
        )
    elif config.RISK_PERCENTAGE_PER_TRADE > 0.1:
        errors.append(
            f"RISK_PERCENTAGE_PER_TRADE ({config.RISK_PERCENTAGE_PER_TRADE:.1%}) "
            f"exceeds maximum 10%"
        )

    # Initial balance
    if config.INITIAL_BALANCE <= 0:
        errors.append(f"INITIAL_BALANCE must be positive, got {config.INITIAL_BALANCE}")

    # Leverage
    if config.MAX_LEVERAGE <= 0:
        errors.append(f"MAX_LEVERAGE must be positive, got {config.MAX_LEVERAGE}")
    elif config.MAX_LEVERAGE > 10:
        warnings.append(f"MAX_LEVERAGE ({config.MAX_LEVERAGE}) is high. Consider reducing.")

    # Drawdown limit
    if config.MAX_DRAWDOWN_LIMIT_PCT <= 0 or config.MAX_DRAWDOWN_LIMIT_PCT > 100:
        errors.append(
            f"MAX_DRAWDOWN_LIMIT_PCT must be between 0-100, "
            f"got {config.MAX_DRAWDOWN_LIMIT_PCT}"
        )

    # Training timesteps
    if config.TOTAL_TIMESTEPS_PER_BOT > 5_000_000:
        errors.append(
            f"TOTAL_TIMESTEPS_PER_BOT ({config.TOTAL_TIMESTEPS_PER_BOT:,}) "
            f"exceeds maximum 5,000,000 (causes overfitting)"
        )

    # Episode length
    if config.FIXED_EPISODE_LENGTH is not None and config.FIXED_EPISODE_LENGTH <= 0:
        errors.append(
            f"FIXED_EPISODE_LENGTH must be positive, "
            f"got {config.FIXED_EPISODE_LENGTH}"
        )

    # Fee percentages
    fee_params = [
        ("TRANSACTION_FEE_PERCENTAGE", config.TRANSACTION_FEE_PERCENTAGE),
        ("SLIPPAGE_PERCENTAGE", config.SLIPPAGE_PERCENTAGE),
    ]
    for name, value in fee_params:
        if value < 0:
            errors.append(f"{name} must be non-negative, got {value}")

    # Log warnings
    for warning in warnings:
        logger.warning(warning)

    # Raise on errors
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigurationError(error_msg)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_action_name(action: int) -> str:
    """Get human-readable name for an action."""
    names = {
        0: 'HOLD',
        1: 'OPEN_LONG',
        2: 'CLOSE_LONG',
        3: 'OPEN_SHORT',
        4: 'CLOSE_SHORT'
    }
    return names.get(action, f'UNKNOWN({action})')


def get_features_list() -> list:
    """Get the list of features used for observation space."""
    return [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'MACD_Diff', 'BB_L', 'BB_H', 'ATR', 'SPREAD',
        'FVG_SIGNAL', 'BOS_SIGNAL', 'CHOCH_SIGNAL', 'OB_STRENGTH_NORM'
    ]


def get_smc_config() -> Dict[str, Any]:
    """Get Smart Money Concepts configuration."""
    return {
        "RSI_WINDOW": 7,
        "MACD_FAST": 8,
        "MACD_SLOW": 17,
        "MACD_SIGNAL": 9,
        "BB_WINDOW": 20,
        "ATR_WINDOW": 7,
        "FRACTAL_WINDOW": 2,
        "FVG_THRESHOLD": 0.0,
    }


def get_hyperparameter_search_space() -> Dict[str, list]:
    """Get hyperparameter search space for training."""
    return {
        'learning_rate': [1e-5, 3e-5, 5e-5, 1e-4],
        'n_steps': [1024, 2048, 4096],
        'batch_size': [64, 128, 256],
        'gamma': [0.99, 0.995, 0.999],
        'ent_coef': [0.02, 0.05, 0.10],
        'clip_range': [0.1, 0.2, 0.3],
        'reward_tanh_scale': [0.2, 0.3, 0.4],
        'reward_output_scale': [3.0, 5.0, 7.0]
    }


def get_model_hyperparameters() -> Dict[str, Any]:
    """Get default model hyperparameters."""
    return {
        "n_steps": 2048,
        "batch_size": 128,
        "gamma": 0.99,
        "learning_rate": 3e-5,
        "ent_coef": 0.05,
        "clip_range": 0.2,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5,
        "n_epochs": 10
    }
