import logging
import os

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# 1. CORE SYSTEM SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

LOGGING_LEVEL = logging.INFO
TRADING_DAYS_YEAR = 252  # Used for Sharpe Ratio annualization
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate (US Treasury)

# Project structure - Use dynamic path detection for portability
# This automatically finds the correct root whether running locally or on Colab
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # Auto-detect project root
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'trained_models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# NEW: Enhanced reporting structure
REPORTS_DIR = os.path.join(RESULTS_DIR, 'training_reports')  # HTML reports
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, 'tensorboard')
CHARTS_DIR = os.path.join(RESULTS_DIR, 'performance_charts')

# Create all directories
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR, REPORTS_DIR,
                  TENSORBOARD_LOG_DIR, CHARTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 2. DATA CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# Data file path - now uses DATA_DIR for portability
# Place your CSV in the 'data' folder of the project
HISTORICAL_DATA_FILE = os.path.join(DATA_DIR, "XAU_15MIN_2019_2024.csv")

# Column mapping (adjust if your CSV uses different names)
OHLCV_COLUMNS = {
    "timestamp": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume"
}

# Data split ratios (professional standard for time series)
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15  # 15% for validation (early stopping)
TEST_RATIO = 0.15  # 15% for final evaluation (out-of-sample)

# Legacy date settings (kept for compatibility, but ratios above take precedence)
TRAIN_END_DATE = "2023-06-30 23:59:00"
VAL_START_DATE = "2023-07-01 00:00:00"
VAL_END_DATE = "2023-12-31 23:59:00"
EVAL_START_DATE = "2024-01-01 00:00:00"
EVAL_END_DATE = "2024-12-31 23:59:00"

# ═════════════════════════════════════════════════════════════════════════════
# 3. ENVIRONMENT & FEATURES
# ═════════════════════════════════════════════════════════════════════════════

ACTION_SPACE_TYPE = 'discrete'  # 5 actions for long/short trading

# ═════════════════════════════════════════════════════════════════════════════
# OPTIMIZED OBSERVATION SPACE (Commercial Performance Update)
# ═════════════════════════════════════════════════════════════════════════════
# Reduced from 633 dimensions to 303 dimensions for:
# - 2x faster training
# - Smaller neural network (less overfitting)
# - Removed sparse/redundant features
#
# Original: 30 bars × 21 features + 3 state = 633 dims
# Optimized: 20 bars × 15 features + 3 state = 303 dims
LOOKBACK_WINDOW_SIZE = 20  # Reduced from 30 - 5 hours of history is sufficient

# Sprint 6: Observation space dimensionality reduction
USE_PCA_REDUCTION = True            # Enable PCA-based dimensionality reduction
PCA_VARIANCE_THRESHOLD = 0.95       # Retain 95% of explained variance
USE_DECORRELATED_FEATURES = True    # Replace OHLC with log_return, hl_range, close_position

# Sprint 7: Async GARCH & Incremental Features
USE_ASYNC_GARCH = True              # Non-blocking GARCH refit in background thread
USE_INCREMENTAL_FEATURES = False    # Incremental TA engine (enable for live trading)

# -----------------------------------------------------------------------------
# EPISODE LENGTH CONFIGURATION (Critical for PPO stability)
# -----------------------------------------------------------------------------
# PPO works best with consistent episode lengths. Random lengths cause:
# - Noisy gradient estimates (mixing different market conditions)
# - Inconsistent learning signal
#
# FIXED_EPISODE_LENGTH: Number of steps per episode
#   - 500 steps = ~5 days of 15-min bars (good for day trading patterns)
#   - Set to None for variable length (not recommended for training)
#
FIXED_EPISODE_LENGTH = 500  # Fixed episode length for stable PPO training
USE_FIXED_EPISODE_LENGTH = True  # Set False to use variable length (not recommended)

# ═════════════════════════════════════════════════════════════════════════════
# ACTION SPACE DEFINITION (Professional Long/Short Trading)
# ═════════════════════════════════════════════════════════════════════════════
# The bot can now trade both LONG and SHORT positions like a professional trader
#
# Action Space:
#   0 = HOLD         : Do nothing, maintain current position
#   1 = OPEN_LONG    : Buy to open a long position (profit when price goes UP)
#   2 = CLOSE_LONG   : Sell to close long position (exit long trade)
#   3 = OPEN_SHORT   : Sell to open a short position (profit when price goes DOWN)
#   4 = CLOSE_SHORT  : Buy to cover short position (exit short trade)
#
# Position States:
#   FLAT  : No position (can OPEN_LONG or OPEN_SHORT)
#   LONG  : Holding long position (can HOLD or CLOSE_LONG)
#   SHORT : Holding short position (can HOLD or CLOSE_SHORT)
#
# Invalid Actions (will be converted to HOLD):
#   - OPEN_LONG when already LONG or SHORT
#   - OPEN_SHORT when already LONG or SHORT
#   - CLOSE_LONG when not LONG
#   - CLOSE_SHORT when not SHORT

NUM_ACTIONS = 5  # Total number of discrete actions

# Action name mapping for logging and debugging
ACTION_NAMES = {
    0: 'HOLD',
    1: 'OPEN_LONG',
    2: 'CLOSE_LONG',
    3: 'OPEN_SHORT',
    4: 'CLOSE_SHORT'
}

# Action constants for cleaner code
ACTION_HOLD = 0
ACTION_OPEN_LONG = 1
ACTION_CLOSE_LONG = 2
ACTION_OPEN_SHORT = 3
ACTION_CLOSE_SHORT = 4

# Position type constants
POSITION_FLAT = 0
POSITION_LONG = 1
POSITION_SHORT = -1

# ═════════════════════════════════════════════════════════════════════════════
# OPTIMIZED FEATURE SET (15 features - removed sparse/redundant)
# ═════════════════════════════════════════════════════════════════════════════
# Removed features (low predictive value for RL):
# - UP_FRACTAL, DOWN_FRACTAL: Sparse (mostly NaN), binary signals
# - BULLISH_OB_HIGH/LOW, BEARISH_OB_HIGH/LOW: Sparse zone markers
# - MACD_line, MACD_signal: Redundant with MACD_Diff
# - BB_M: Redundant (just SMA, less useful than bands)
#
# Impact: 2x faster training, reduced overfitting
FEATURES = [
    # OHLCV Base (5 features)
    'Open', 'High', 'Low', 'Close', 'Volume',

    # Technical Indicators (6 features)
    'RSI',        # Momentum - key oscillator
    'MACD_Diff',  # Trend - most useful MACD component
    'BB_L',       # Volatility - lower band
    'BB_H',       # Volatility - upper band
    'ATR',        # Volatility - essential for risk
    'SPREAD',     # Candle range - intraday volatility

    # Smart Money Concepts (4 features)
    'FVG_SIGNAL',       # Fair Value Gaps - institutional footprint
    'BOS_SIGNAL',       # Break of Structure - trend direction
    'CHOCH_SIGNAL',     # Change of Character - reversals
    'OB_STRENGTH_NORM', # Order Block strength - key SMC signal

    # Gap Detection (1 feature)
    'WEEKEND_GAP',      # Weekend/holiday gap size for Gold
]

# Multi-Timeframe Features (14 features) - Higher timeframe context for Gold
# These are computed by MultiTimeframeFeatures and added dynamically in _process_data()
MTF_FEATURES = [
    'HTF_TREND_1H', 'HTF_STRENGTH_1H', 'HTF_RSI_1H',
    'PRICE_VS_SMA20_1H', 'PRICE_VS_SMA50_1H',
    'HTF_TREND_4H', 'HTF_STRENGTH_4H', 'HTF_RSI_4H',
    'PRICE_VS_SMA20_4H', 'PRICE_VS_SMA50_4H',
    'SESSION', 'DAY_OF_WEEK', 'HOUR_SIN', 'HOUR_COS'
]
ENABLE_MTF_FEATURES = True

# Full feature set (for backward compatibility or experimentation)
FEATURES_FULL = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RSI', 'MACD_Diff', 'MACD_line', 'MACD_signal',
    'BB_L', 'BB_M', 'BB_H', 'ATR', 'SPREAD', 'BODY_SIZE',
    'UP_FRACTAL', 'DOWN_FRACTAL', 'FVG_SIGNAL', 'FVG_SIZE_NORM',
    'BOS_SIGNAL', 'CHOCH_SIGNAL',
    'BULLISH_OB_HIGH', 'BULLISH_OB_LOW',
    'BEARISH_OB_HIGH', 'BEARISH_OB_LOW',
    'OB_STRENGTH_NORM'
]

# Technical indicator parameters
SMC_CONFIG = {
    "RSI_WINDOW": 7,  # Faster RSI for 15min timeframe
    "MACD_FAST": 8,
    "MACD_SLOW": 17,
    "MACD_SIGNAL": 9,
    "BB_WINDOW": 20,
    "ATR_WINDOW": 7,  # Shorter ATR for responsive risk management
    "FRACTAL_WINDOW": 2,  # 5-bar fractals (2 left + center + 2 right)
    "FVG_THRESHOLD": 0.0,  # Any size FVG is valid
}

# ═════════════════════════════════════════════════════════════════════════════
# 4. RISK MANAGEMENT (COMMERCIAL-GRADE CONTROLS)
# ═════════════════════════════════════════════════════════════════════════════

# ⚠️ CRITICAL CHANGE: Reduced from 3% to 1% (professional standard)
RISK_PERCENTAGE_PER_TRADE = 0.01  # Risk 1% of capital per trade

# Take Profit / Stop Loss
TAKE_PROFIT_PERCENTAGE = 0.02  # 2% TP
STOP_LOSS_PERCENTAGE = 0.01  # 1% SL (2:1 Risk:Reward ratio)
RISK_TO_REWARD_RATIO = TAKE_PROFIT_PERCENTAGE / STOP_LOSS_PERCENTAGE

# Trailing Stop Loss
TSL_START_PROFIT_MULTIPLIER = 1.0  # Activate TSL at 1× ATR profit
TSL_TRAIL_DISTANCE_MULTIPLIER = 0.5  # Trail at 0.5× ATR distance

# GARCH Volatility Model
# OPTIMIZED: Increased from 100 to 500 steps between refits
# GARCH refit is expensive (~200-400ms). Between refits, we use fast EWMA approximation.
# This reduces overhead by 5x while maintaining volatility accuracy.
GARCH_UPDATE_FREQUENCY = 500  # Refit GARCH model every N steps (was 100)

# Value at Risk (VaR) Engine
VAR_CONFIDENCE_LEVEL = 0.95           # 95% confidence for primary VaR
VAR_ROLLING_WINDOW = 252              # 252 trading days (~1 year) lookback
VAR_METHOD = 'cornish_fisher'         # Default: 'cornish_fisher' | 'historical' | 'parametric' | 'monte_carlo'
VAR_MAX_PCT = 0.02                    # 2% portfolio VaR limit (triggers kill switch)

# ═════════════════════════════════════════════════════════════════════════════
# SHORT SELLING CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
# Short selling allows profiting from price decreases - essential for day trading
#
# How it works:
#   1. OPEN_SHORT: Borrow asset, sell at current price (entry)
#   2. Price moves DOWN: You profit (buy back cheaper)
#   3. Price moves UP: You lose (buy back more expensive)
#   4. CLOSE_SHORT: Buy back asset to return to lender (exit)
#
# Risk note: Short positions have theoretically unlimited loss potential
#            (price can go to infinity), so strict risk management is critical

ENABLE_SHORT_SELLING = True  # Master switch for short selling

# Short-specific costs (typically higher than long positions)
SHORT_BORROWING_FEE_DAILY = 0.0001  # 0.01% daily borrowing cost (annualized ~3.65%)
SHORT_MARGIN_REQUIREMENT = 1.0     # 100% margin required (no leverage for safety)

# Short position limits (can be more conservative than long)
SHORT_MAX_POSITION_SIZE_PCT = 0.20   # Maximum 20% of portfolio in shorts
SHORT_MAX_HOLDING_DURATION = 40      # Max bars to hold short (same as long)

# Overnight short fees (swap rates - common in forex/CFD)
# Gold typically has negative swap for shorts (you pay to hold overnight)
SHORT_OVERNIGHT_SWAP_PCT = 0.0002    # 0.02% per overnight hold

# ⚠️ CRITICAL CHANGE: Reduced from 15% to 10% (safer for commercial use)
MAX_DRAWDOWN_LIMIT_PCT = 10.0  # Maximum 10% account drawdown

# ═════════════════════════════════════════════════════════════════════════════
# 5. TRADING PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

INITIAL_BALANCE = 1000.0  # Starting capital for backtesting
MINIMUM_ALLOWED_BALANCE = 100.0  # Emergency stop threshold
ALLOW_NEGATIVE_BALANCE = False  # No margin calls
MIN_TRADE_QUANTITY = 0.01  # Minimum position size

# Transaction costs (realistic broker fees)
TRANSACTION_FEE_PERCENTAGE = 0.0005  # 0.05% spread
SLIPPAGE_PERCENTAGE = 0.0001  # 0.01% slippage
TRADE_COMMISSION_PER_TRADE = 0.0005  # $0.50 per $1000 traded

# Commission structure (moved from hardcoded values in environment.py)
TRADE_COMMISSION_PCT_OF_TRADE = 0.0005  # 0.05% of trade value (standard broker)
TRADE_COMMISSION_MIN_PCT_CAPITAL = 0.0001  # 0.01% of initial capital (minimum fee)

ALLOW_NEGATIVE_REVENUE_SELL = False

# ═════════════════════════════════════════════════════════════════════════════
# 6. REWARD FUNCTION PARAMETERS (OPTIMIZED FOR LEARNING)
# ═════════════════════════════════════════════════════════════════════════════

# Core scaling - 1% return = 1.0 reward point
# Keeps rewards in a range [-1, 1] which PPO loves
REWARD_SCALING_FACTOR = 100.0

# -----------------------------------------------------------------------------
# REWARD NORMALIZATION (Now Hyperparameters for tuning)
# -----------------------------------------------------------------------------
# These control how rewards are squashed and scaled before PPO sees them
# Different hyperparameter combinations may work better with different scaling
#
# REWARD_TANH_SCALE: Controls sensitivity of tanh squashing
#   - Lower (0.1-0.2): More compressed, agent less sensitive to reward differences
#   - Higher (0.4-0.5): Wider range, agent more sensitive to reward differences
#
# REWARD_OUTPUT_SCALE: Final multiplier after tanh
#   - Typical PPO rewards are in range [-1, +1] or [-10, +10]
#   - This scales the normalized reward to your desired range
#
REWARD_TANH_SCALE = 0.3      # Default: 0.3 (tunable in hyperparameter search)
REWARD_OUTPUT_SCALE = 5.0    # Default: 5.0 (tunable in hyperparameter search)

# -----------------------------------------------------------------------------
# PENALTIES & BONUSES (The "Lazy Agent" Fix)
# -----------------------------------------------------------------------------
# Sprint 2: Reward restructuring — risk-adjusted, anti-churning design.
# OLD values kept as comments for audit trail.
LOSING_TRADE_PENALTY = 0.5       # Was 0.0 (Sprint 2: mild penalty for losses)
WINNING_TRADE_BONUS = 0.0        # Was 2.0 (Sprint 2: replaced by RR-based bonus in reward fn)

# Friction: penalize transaction costs to discourage churning
W_FRICTION = 0.3                 # Was 0.1 (Sprint 2: respect transaction costs)

# -----------------------------------------------------------------------------
# RISK CONTROL WEIGHTS
# -----------------------------------------------------------------------------
W_RETURN = 1.0       # Primary driver: Make Money
W_DRAWDOWN = 1.0     # Was 0.5 (Sprint 2: restore drawdown awareness)
W_LEVERAGE = 1.0     # Enforce limits
W_TURNOVER = 0.3     # Was 0.0 (Sprint 2: penalize excessive trading)
W_DURATION = 0.0     # Was 0.1 (Sprint 2: removed — let winners run)

# -----------------------------------------------------------------------------
# MISC PENALTIES
# -----------------------------------------------------------------------------
DOWNSIDE_PENALTY_MULTIPLIER = 1.0  # Standard linear punishment for losses.
OVERNIGHT_HOLDING_PENALTY = 0.0    # N/A for 15m timeframe
HOLD_PENALTY_FACTOR = 0.0          # Was 0.01 (Sprint 2: holding is NEUTRAL, not penalized)
FAILED_TRADE_ATTEMPT_PENALTY = 0.0 # No penalty for logic checks
TRADE_COOLDOWN_STEPS = 2           # Allow faster re-entry if signal is good.
RAPID_TRADE_PENALTY = 1.0          # Reduced from 5.0.

# -----------------------------------------------------------------------------
# CREDIT ASSIGNMENT FIX: Risk Sentinel Rejection Penalties
# -----------------------------------------------------------------------------
# When Risk Sentinel rejects an action, we need PPO to learn "that action was bad"
# not "HOLD was neutral". These penalties provide the correct learning signal.
#
# CRITICAL: This fixes the credit assignment bug where PPO associated the reward
# from executing HOLD with the original (rejected) action.
RISK_REJECTION_PENALTY = 2.0       # Penalty when action is fully rejected by risk system
RISK_MODIFICATION_PENALTY = 0.5    # Small penalty when position size is significantly reduced
NEWS_BLOCK_PENALTY = 1.0           # Penalty when blocked by high-impact news event
MODIFICATION_THRESHOLD = 0.5      # Position size reduction threshold to trigger penalty
                                   # (e.g., 0.5 = 50% reduction triggers modification penalty)

# Position limits (Keep safety hard-limits)
MAX_LEVERAGE = 1.0                 # Safe baseline
MAX_DURATION_STEPS = 40            # Increased from 12 (3h) to 40 (10h).
                                   # 12 steps is too short for a trend to develop.
# ═════════════════════════════════════════════════════════════════════════════
# 7. TRAINING CONFIGURATION (⚠️ CRITICAL SECTION)
# ═════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "PPO_XAU_DayTrader_Production_v2"
RANDOM_SEED = 42
RENDER_MODE = "none"  # No visual rendering during training

# ⚠️⚠️⚠️ CRITICAL CHANGE: TIMESTEPS ⚠️⚠️⚠️
# OLD VALUE: 20,000,000 (CAUSES SEVERE OVERFITTING)
# NEW VALUE: 1,500,000 (RESEARCH-BACKED OPTIMAL)
#
# WHY THIS CHANGE IS CRITICAL:
# - Your data: ~20,000 bars
# - At 20M timesteps: Agent sees each pattern 1,000+ times (memorization)
# - At 1.5M timesteps: Agent sees each pattern 75 times (learning)
# - Research shows overfitting starts beyond 2M for trading bots
# - Academic papers use 100K-2M for similar setups
TOTAL_TIMESTEPS_PER_BOT = 1500000
# Early stopping (prevent wasted training)
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement after 5 evaluations
EVAL_FREQ = 10_000  # Evaluate every 10K timesteps
N_EVAL_EPISODES = 5  # Run 5 episodes per evaluation

# Evaluation paths
EVAL_RESULTS_PATH = os.path.join(RESULTS_DIR, 'evaluation_results')

# ═════════════════════════════════════════════════════════════════════════════
# 8. PARALLEL TRAINING CONFIGURATION (NEW!)
# ═════════════════════════════════════════════════════════════════════════════

# How many different bots to train with different hyperparameters
N_PARALLEL_BOTS = 50  # Train 50 bots, pick the best

# GPU capacity control (adjust based on your GPU)
# RTX 3080/3090: 4 workers
# RTX 4090: 6-8 workers
# V100/A100: 8-10 workers
MAX_WORKERS_GPU = 2# Train 4 bots simultaneously

# Selection metric for best model
EVALUATION_METRIC = 'sharpe_ratio'  # Options: 'sharpe_ratio', 'calmar_ratio', 'profit'

# Quality thresholds for commercial deployment
MIN_ACCEPTABLE_SHARPE = 1.5  # Minimum Sharpe to be considered "good"
MIN_ACCEPTABLE_CALMAR = 2.0  # Minimum Calmar ratio
MAX_ACCEPTABLE_DD = 0.15  # Maximum 15% drawdown

# ═════════════════════════════════════════════════════════════════════════════
# 9. HYPERPARAMETER SEARCH SPACE (RESEARCH-BACKED)
# ═════════════════════════════════════════════════════════════════════════════

# Based on FinRL framework and academic research for PPO in finance
# These ranges have been validated in multiple trading papers
# Sprint 8: Corrected search space — removed extreme values that cause training instability
HYPERPARAM_SEARCH_SPACE = {
    'learning_rate': [1e-4, 3e-4, 5e-4],          # Removed 1e-5 (too slow to converge)
    'n_steps': [1024, 2048, 4096],                 # Must be >= batch_size
    'batch_size': [64, 128, 256],                  # Unchanged
    'gamma': [0.99, 0.995, 0.998],                 # Added 0.998, removed 0.999 (too long horizon)
    'ent_coef': [0.005, 0.01, 0.02],              # Removed 0.05/0.10 (excess randomness)
    'clip_range': [0.1, 0.2, 0.3],                # Unchanged
    'n_epochs': [3, 5, 7],                         # Reduced from max=10 (overfitting risk)
    'reward_tanh_scale': [0.2, 0.3, 0.4],         # Unchanged
    'reward_output_scale': [3.0, 5.0, 7.0]        # Unchanged
}

# Sprint 8: Corrected baseline hyperparameters
# Reference: Schulman et al. 2017 (PPO paper), FinRL benchmark suite
MODEL_HYPERPARAMETERS = {
    "n_steps": 1024,       # ~2x episode length (500); shorter rollouts = fresher gradients
    "batch_size": 128,     # Standard PPO minibatch
    "gamma": 0.995,        # Effective horizon ~200 steps (suitable for M15 intraday)
    "learning_rate": 3e-4, # Standard PPO LR (Schulman 2017); was 3e-5 (10x too low)
    "ent_coef": 0.01,      # Moderate exploration; was 0.05 (too random for exploitation)
    "clip_range": 0.2,     # Standard PPO clip
    "gae_lambda": 0.95,    # Standard GAE lambda for advantage estimation
    "max_grad_norm": 0.5,  # Gradient clipping for stability
    "vf_coef": 0.5,        # Value function loss weight
    "n_epochs": 5          # Was 10; fewer epochs = less overfitting to rollout buffer
}

# Sprint 8: Entropy annealing schedule (step_threshold -> ent_coef)
# Starts with higher exploration, gradually shifts to exploitation
ENTROPY_ANNEALING_SCHEDULE = {
    0:       0.05,   # Phase 1 (BASE): High exploration
    100_000: 0.02,   # Phase 2 (ENRICHED): Moderate exploration
    300_000: 0.01,   # Phase 3 (SOFT): Standard
    500_000: 0.005,  # Phase 4 (PRODUCTION): Exploit learned policy
}

# Sprint 8: LR warmup fraction (linear warmup over first N% of training)
LR_WARMUP_FRACTION = 0.05  # 5% of total steps

# ═════════════════════════════════════════════════════════════════════════════
# 10. LOGGING & MONITORING
# ═════════════════════════════════════════════════════════════════════════════

ENABLE_TENSORBOARD = True  # Real-time training visualization
ENABLE_WANDB = False  # Cloud logging (set to True if you have account)
WANDB_PROJECT = "XAU_Trading_Bot_Production"
WANDB_ENTITY = "your_username"  # Replace with your W&B username

# Report generation
REPORT_FORMAT = 'html'  # 'html', 'pdf', or 'markdown'
INCLUDE_CHARTS = True  # Include performance charts in reports
INCLUDE_TRADE_LOG = True  # Include detailed trade log
PROFIT_CURRENCY = 'USD'

# ═════════════════════════════════════════════════════════════════════════════
# 11. PRODUCTION DEPLOYMENT SETTINGS (FOR LIVE TRADING)
# ═════════════════════════════════════════════════════════════════════════════

# Minimum requirements for live deployment
DEPLOYMENT_MIN_CAPITAL = 5000.0  # Don't go live with less than $5K
DEPLOYMENT_MAX_LEVERAGE = 1.0  # NO leverage in production (safety first)
DEPLOYMENT_MONITORING_INTERVAL = 300  # Check every 5 minutes
AUTO_SHUTDOWN_ON_DD = True  # Auto-stop if drawdown exceeds limit
EMERGENCY_CONTACT_EMAIL = "your_email@example.com"  # Alert email

# Live trading risk limits (stricter than backtest)
LIVE_RISK_PER_TRADE = 0.005  # 0.5% (half of backtest risk)
LIVE_MAX_DRAWDOWN = 0.08  # 8% (tighter than backtest)
LIVE_MAX_DAILY_TRADES = 10  # Limit daily activity


# ═════════════════════════════════════════════════════════════════════════════
# 11.1 QUALITY GATES & ENSEMBLE SEEDS (SPRINT 14)
# ═════════════════════════════════════════════════════════════════════════════
# These thresholds must be met before a model is promoted to production.

QUALITY_GATES = {
    'min_sharpe': 1.0,            # Minimum annualised Sharpe ratio
    'max_drawdown': 0.15,         # Maximum drawdown (15 %)
    'min_win_rate': 0.40,         # Minimum win rate (40 %)
    'min_profit_factor': 1.3,     # Minimum profit factor (gross profit / gross loss)
}

# Seeds for multi-seed ensemble training (SophisticatedTrainer.train_ensemble_seeds)
ENSEMBLE_SEEDS = (42, 123, 456)

# ═════════════════════════════════════════════════════════════════════════════
# 11.5 WALK-FORWARD VALIDATION CONFIGURATION (SPRINT 3)
# ═════════════════════════════════════════════════════════════════════════════

# Enable walk-forward validation (CRITICAL for production)
# Walk-forward prevents overfitting to specific market regimes
USE_WALK_FORWARD = True  # Set to False for legacy single-split training

# Walk-Forward Parameters
WALK_FORWARD_CONFIG = {
    # Window sizes (in 15-minute bars)
    'train_window_bars': 6720,      # ~6 months training data
    'validation_window_bars': 2240,  # ~2 months validation
    'test_window_bars': 1120,        # ~1 month out-of-sample test
    'step_size_bars': 1120,          # Slide forward by 1 month each fold
    'purge_gap_bars': 96,            # 1-day gap to prevent look-ahead bias

    # Fold limits
    'min_folds': 3,                  # Minimum for statistical validity
    'max_folds': 6,                  # Maximum to limit compute time (6 is statistically sufficient)

    # Strategy: 'rolling', 'expanding', or 'anchored'
    # - rolling: Fixed-size moving window (recommended for non-stationary)
    # - expanding: Growing training set from start
    # - anchored: Fixed start, expanding end
    'strategy': 'rolling',

    # Early stopping if performance degrades significantly
    'early_stop_degradation_threshold': 0.3,  # Stop if Sharpe drops 30% from best
}


# ═════════════════════════════════════════════════════════════════════════════
# 11.6 NEWS/ECONOMIC CALENDAR FILTER CONFIGURATION (SPRINT 3)
# ═════════════════════════════════════════════════════════════════════════════

# Enable news/economic calendar filtering
# Prevents trading during high-impact news events (FOMC, NFP, CPI, etc.)
USE_NEWS_FILTER = True

# News Filter Configuration
NEWS_FILTER_CONFIG = {
    # High-impact event blocking (FOMC, NFP, CPI, ECB rate decisions)
    'high_impact_block_minutes_before': 30,   # Block new trades 30 min before
    'high_impact_block_minutes_after': 30,    # Block new trades 30 min after

    # Medium-impact event handling (PMI, Retail Sales, etc.)
    'medium_impact_block_minutes_before': 15,
    'medium_impact_block_minutes_after': 15,
    'medium_impact_position_multiplier': 0.5,  # Reduce position size by 50%

    # Events to block completely (no trading allowed)
    'blocked_events': [
        'FOMC',           # Federal Reserve rate decisions
        'NFP',            # Non-Farm Payrolls
        'CPI',            # Consumer Price Index (inflation)
        'ECB',            # European Central Bank decisions
        'BOE',            # Bank of England decisions
        'BOJ',            # Bank of Japan decisions
        'GDP',            # Gross Domestic Product
    ],

    # Events that reduce position size (but don't block)
    'reduced_events': [
        'PMI',            # Purchasing Managers Index
        'Retail Sales',
        'Industrial Production',
        'Unemployment Claims',
        'Trade Balance',
    ],

    # Calendar data file (historical events for backtesting)
    'calendar_file': os.path.join(DATA_DIR, 'economic_calendar_2019_2024.csv'),

    # Currency filter (only care about events affecting XAU/USD)
    'relevant_currencies': ['USD', 'EUR', 'GBP', 'JPY', 'CHF'],
}


# ═════════════════════════════════════════════════════════════════════════════
# 12. SYSTEM VALIDATION & CHECKS
# ═════════════════════════════════════════════════════════════════════════════

def validate_configuration(strict_mode: bool = False):
    """
    Validates configuration for common mistakes and security issues.
    Call this at startup to catch issues early.

    Args:
        strict_mode: If True, raise errors for warnings in production environments

    Raises:
        ValueError: If critical configuration errors are found
    """
    errors = []
    warnings = []

    # =========================================================================
    # SECURITY: Validate placeholder values are replaced
    # =========================================================================
    placeholder_checks = [
        (EMERGENCY_CONTACT_EMAIL, "your_email@example.com", "EMERGENCY_CONTACT_EMAIL"),
    ]

    for value, placeholder, name in placeholder_checks:
        if value == placeholder:
            if strict_mode:
                errors.append(
                    f"SECURITY ERROR: {name} contains placeholder value '{placeholder}'. "
                    f"Set this to a real value in production."
                )
            else:
                warnings.append(
                    f"SECURITY WARNING: {name} contains placeholder value. "
                    f"Set this before production deployment."
                )

    # =========================================================================
    # SECURITY: Validate critical numeric bounds
    # =========================================================================

    # Risk percentage must be positive and reasonable
    if RISK_PERCENTAGE_PER_TRADE <= 0:
        errors.append(f"RISK_PERCENTAGE_PER_TRADE must be positive, got {RISK_PERCENTAGE_PER_TRADE}")
    elif RISK_PERCENTAGE_PER_TRADE > 0.1:
        errors.append(
            f"RISK_PERCENTAGE_PER_TRADE ({RISK_PERCENTAGE_PER_TRADE:.1%}) is dangerously high. "
            f"Maximum allowed: 10%"
        )
    elif RISK_PERCENTAGE_PER_TRADE > 0.02:
        warnings.append(
            f"RISK_PERCENTAGE_PER_TRADE ({RISK_PERCENTAGE_PER_TRADE:.1%}) is aggressive. "
            f"Professional standard: 1%"
        )

    # Initial balance must be positive
    if INITIAL_BALANCE <= 0:
        errors.append(f"INITIAL_BALANCE must be positive, got {INITIAL_BALANCE}")

    # Leverage must be positive and bounded
    if MAX_LEVERAGE <= 0:
        errors.append(f"MAX_LEVERAGE must be positive, got {MAX_LEVERAGE}")
    elif MAX_LEVERAGE > 10:
        warnings.append(f"MAX_LEVERAGE ({MAX_LEVERAGE}) is very high. Consider reducing for safety.")

    # Drawdown limit must be between 0 and 100
    if MAX_DRAWDOWN_LIMIT_PCT <= 0 or MAX_DRAWDOWN_LIMIT_PCT > 100:
        errors.append(f"MAX_DRAWDOWN_LIMIT_PCT must be between 0 and 100, got {MAX_DRAWDOWN_LIMIT_PCT}")
    elif MAX_DRAWDOWN_LIMIT_PCT > 20:
        warnings.append(
            f"MAX_DRAWDOWN_LIMIT_PCT ({MAX_DRAWDOWN_LIMIT_PCT}%) is high. "
            f"Commercial standard: 10-15%"
        )

    # =========================================================================
    # TRAINING: Validate hyperparameters
    # =========================================================================

    # Critical validations
    if TOTAL_TIMESTEPS_PER_BOT > 5_000_000:
        errors.append(
            f"TOTAL_TIMESTEPS_PER_BOT ({TOTAL_TIMESTEPS_PER_BOT:,}) is too high. "
            f"This causes severe overfitting. Maximum: 5,000,000"
        )

    if N_PARALLEL_BOTS < 20:
        warnings.append(
            f"N_PARALLEL_BOTS ({N_PARALLEL_BOTS}) is low. "
            f"Recommended: 50+ for good hyperparameter coverage"
        )

    # Hyperparameter validations
    for key, values in HYPERPARAM_SEARCH_SPACE.items():
        if len(values) < 2:
            warnings.append(f"⚠️ Hyperparameter '{key}' has only {len(values)} values. "
                            f"Need at least 2 for meaningful search")

    # Batch size must be <= n_steps
    max_batch = max(HYPERPARAM_SEARCH_SPACE['batch_size'])
    min_steps = min(HYPERPARAM_SEARCH_SPACE['n_steps'])
    if max_batch > min_steps:
        errors.append(f"❌ batch_size ({max_batch}) cannot exceed n_steps ({min_steps})")

    # =========================================================================
    # ADDITIONAL BOUNDS CHECKING
    # =========================================================================

    # Episode length must be positive
    if FIXED_EPISODE_LENGTH is not None and FIXED_EPISODE_LENGTH <= 0:
        errors.append(f"FIXED_EPISODE_LENGTH must be positive, got {FIXED_EPISODE_LENGTH}")

    # Lookback window must be positive and reasonable
    if LOOKBACK_WINDOW_SIZE <= 0:
        errors.append(f"LOOKBACK_WINDOW_SIZE must be positive, got {LOOKBACK_WINDOW_SIZE}")
    elif LOOKBACK_WINDOW_SIZE > 200:
        warnings.append(
            f"LOOKBACK_WINDOW_SIZE ({LOOKBACK_WINDOW_SIZE}) is very large. "
            f"This increases training time significantly."
        )

    # Fee percentages must be non-negative
    fee_params = [
        ("TRANSACTION_FEE_PERCENTAGE", TRANSACTION_FEE_PERCENTAGE),
        ("SLIPPAGE_PERCENTAGE", SLIPPAGE_PERCENTAGE),
        ("TRADE_COMMISSION_PCT_OF_TRADE", TRADE_COMMISSION_PCT_OF_TRADE),
    ]
    for name, value in fee_params:
        if value < 0:
            errors.append(f"{name} must be non-negative, got {value}")
        elif value > 0.1:  # 10%
            warnings.append(f"{name} ({value:.2%}) seems unusually high")

    if errors:
        for error in errors:
            logger.error("Config error: %s", error)
        raise ValueError("Configuration has critical errors. Fix them before training!")

    if warnings:
        for warning in warnings:
            logger.warning("Config warning: %s", warning)

    if not errors and not warnings:
        logger.info(
            "Configuration validated successfully — %d bots x %s steps, %d GPU workers",
            N_PARALLEL_BOTS, f"{TOTAL_TIMESTEPS_PER_BOT:,}", MAX_WORKERS_GPU,
        )


# ═════════════════════════════════════════════════════════════════════════════
# 13. STARTUP BANNER
# ═════════════════════════════════════════════════════════════════════════════

def log_startup_banner():
    """Log configuration summary on startup."""
    logger.info(
        "XAU TRADING BOT — Version 2.0.0 (PRODUCTION) | Model: %s | "
        "%s steps/bot | %d bots | %d GPU workers | Risk: %s | Max DD: %s%% | "
        "Split: %s/%s/%s | Reports: %s | Models: %s | Logs: %s",
        MODEL_NAME, f"{TOTAL_TIMESTEPS_PER_BOT:,}", N_PARALLEL_BOTS,
        MAX_WORKERS_GPU, f"{RISK_PERCENTAGE_PER_TRADE:.1%}",
        MAX_DRAWDOWN_LIMIT_PCT,
        f"{TRAIN_RATIO:.0%}", f"{VAL_RATIO:.0%}", f"{TEST_RATIO:.0%}",
        REPORTS_DIR, MODEL_DIR, LOG_DIR,
    )


# Keep old name as alias for backward compatibility
print_startup_banner = log_startup_banner




# ═════════════════════════════════════════════════════════════════════════════
# END OF CONFIGURATION
# ═══════════════════════