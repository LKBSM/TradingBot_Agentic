import os
import logging

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
LOOKBACK_WINDOW_SIZE = 30  # 30 bars = 7.5 hours of history (reduced from 60 for faster training)

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

# Feature set (Technical Indicators + Smart Money Concepts)
FEATURES = [
    # OHLCV Base
    'Open', 'High', 'Low', 'Close', 'Volume',

    # Technical Indicators
    'RSI',  # Momentum
    'MACD_Diff', 'MACD_line', 'MACD_signal',  # Trend
    'BB_L', 'BB_M', 'BB_H',  # Volatility bands
    'ATR',  # Volatility
    'SPREAD', 'BODY_SIZE',  # Candle metrics

    # Smart Money Concepts (SMC)
    'UP_FRACTAL', 'DOWN_FRACTAL',  # Swing points
    'FVG_SIGNAL',  # Fair Value Gaps
    'BOS_SIGNAL', 'CHOCH_SIGNAL',  # Structure breaks
    'BULLISH_OB_HIGH', 'BULLISH_OB_LOW',  # Order blocks
    'BEARISH_OB_HIGH', 'BEARISH_OB_LOW',
    'OB_STRENGTH_NORM'  # OB strength
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
GARCH_UPDATE_FREQUENCY = 100  # Refit GARCH model every N steps (expensive operation)

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
# CRITICAL FIX: Removed the massive 5.0 penalty for losing.
# The PnL itself is already negative; adding a fixed penalty causes "fear of trading".
LOSING_TRADE_PENALTY = 0.0       # Was 5.0 (Too harsh, caused fear)
WINNING_TRADE_BONUS = 2.0        # FIX "FEARFUL AGENT": Strong bonus for profitable trades

# Friction: Fees are already subtracted from PnL.
# We don't need a heavy extra penalty, or the bot won't enter trades.
W_FRICTION = 0.1                 # Was 0.8 (Reduced to prevent entry paralysis)

# -----------------------------------------------------------------------------
# RISK CONTROL WEIGHTS
# -----------------------------------------------------------------------------
# We still punish bad behavior, but proportionally.

W_RETURN = 1.0       # Primary driver: Make Money
W_DRAWDOWN = 0.5     # Was 2.0. Reduced so the bot isn't terrified of normal volatility.
W_LEVERAGE = 1.0     # Enforce limits, but don't kill exploration.
W_TURNOVER = 0.0     # Was 0.1. Let it trade as much as needed initially.
W_DURATION = 0.1     # Was 0.3. Slight nudge to not hold forever.

# -----------------------------------------------------------------------------
# MISC PENALTIES
# -----------------------------------------------------------------------------
DOWNSIDE_PENALTY_MULTIPLIER = 1.0  # Was 3.0. Standard linear punishment for losses.
OVERNIGHT_HOLDING_PENALTY = 0.0    # N/A for 15m timeframe
HOLD_PENALTY_FACTOR = 0.01         # FIX "FEARFUL AGENT": Small penalty to encourage trading
FAILED_TRADE_ATTEMPT_PENALTY = 0.0 # No penalty for logic checks
TRADE_COOLDOWN_STEPS = 2           # Was 5. Allow faster re-entry if signal is good.
RAPID_TRADE_PENALTY = 1.0          # Was 5.0. Reduced.

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
HYPERPARAM_SEARCH_SPACE = {
    'learning_rate': [1e-5, 3e-5, 5e-5, 1e-4],  # 4 values
    'n_steps': [1024, 2048, 4096],  # 3 values (must be >= batch_size)
    'batch_size': [64, 128, 256],  # 3 values
    'gamma': [0.99, 0.995, 0.999],  # 3 values (discount factor)
    'ent_coef': [0.02, 0.05, 0.10],  # 3 values (exploration) - INCREASED for more active trading
    'clip_range': [0.1, 0.2, 0.3],  # 3 values (PPO clip)
    # NEW: Reward scaling parameters (interact with learning dynamics)
    'reward_tanh_scale': [0.2, 0.3, 0.4],  # 3 values (sensitivity)
    'reward_output_scale': [3.0, 5.0, 7.0]  # 3 values (final range)
}

# Total possible combinations: 4×3×3×3×3×3×3×3 = 8,748
# We intelligently sample 50 of these (not random, stratified)

# Baseline hyperparameters (FinRL defaults - proven in research)
MODEL_HYPERPARAMETERS = {
    "n_steps": 2048,  # Rollout buffer size
    "batch_size": 128,  # Minibatch size for updates
    "gamma": 0.99,  # Discount factor
    "learning_rate": 3e-5,  # Adam learning rate (conservative)
    "ent_coef": 0.05,  # Entropy coefficient - INCREASED from 0.01 to encourage exploration
    "clip_range": 0.2,  # PPO clipping parameter
    "gae_lambda": 0.95,  # GAE lambda for advantage estimation
    "max_grad_norm": 0.5,  # Gradient clipping (stability)
    "vf_coef": 0.5,  # Value function coefficient
    "n_epochs": 10  # Number of epochs per update
}

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
# 12. SYSTEM VALIDATION & CHECKS
# ═════════════════════════════════════════════════════════════════════════════

def validate_configuration():
    """
    Validates configuration for common mistakes.
    Call this at startup to catch issues early.
    """
    errors = []
    warnings = []

    # Critical validations
    if TOTAL_TIMESTEPS_PER_BOT > 5_000_000:
        errors.append(f"⚠️ CRITICAL: TOTAL_TIMESTEPS_PER_BOT ({TOTAL_TIMESTEPS_PER_BOT:,}) ")

    if N_PARALLEL_BOTS < 20:
        warnings.append(f"⚠️ N_PARALLEL_BOTS ({N_PARALLEL_BOTS}) is low. "
                        f"Recommended: 50+ for good hyperparameter coverage")

    if RISK_PERCENTAGE_PER_TRADE > 0.02:
        warnings.append(f"⚠️ RISK_PERCENTAGE_PER_TRADE ({RISK_PERCENTAGE_PER_TRADE:.1%}) "
                        f"is aggressive. Professional standard: 1%")

    if MAX_DRAWDOWN_LIMIT_PCT > 15:
        warnings.append(f"⚠️ MAX_DRAWDOWN_LIMIT_PCT ({MAX_DRAWDOWN_LIMIT_PCT}%) "
                        f"is high for commercial use. Recommended: 10%")

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

    # Print results
    if errors:
        print("\n" + "=" * 70)
        print("❌ CONFIGURATION ERRORS (MUST FIX):")
        print("=" * 70)
        for error in errors:
            print(error)
        print("=" * 70)
        raise ValueError("Configuration has critical errors. Fix them before training!")

    if warnings:
        print("\n" + "=" * 70)
        print("⚠️  CONFIGURATION WARNINGS (REVIEW RECOMMENDED):")
        print("=" * 70)
        for warning in warnings:
            print(warning)
        print("=" * 70)

    # Success message
    if not errors and not warnings:
        print("\n" + "=" * 70)
        print("✅ CONFIGURATION VALIDATED SUCCESSFULLY")
        print("=" * 70)
        print(f"✅ Training: {N_PARALLEL_BOTS} bots × {TOTAL_TIMESTEPS_PER_BOT:,} timesteps")
        print(f"✅ Total compute: {N_PARALLEL_BOTS * TOTAL_TIMESTEPS_PER_BOT:,} timesteps")
        print(f"✅ GPU workers: {MAX_WORKERS_GPU}")
        print(
            f"✅ Expected time: ~{(N_PARALLEL_BOTS * TOTAL_TIMESTEPS_PER_BOT) / (MAX_WORKERS_GPU * 200_000):.1f} hours")
        print("=" * 70 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# 13. STARTUP BANNER
# ═════════════════════════════════════════════════════════════════════════════

def print_startup_banner():
    """Prints configuration summary on startup"""
    print("\n" + "=" * 70)
    print("    XAU TRADING BOT - PRODUCTION CONFIGURATION")
    print("=" * 70)
    print(f"Version:           2.0.0 (PRODUCTION)")
    print(f"Model:             {MODEL_NAME}")
    print(f"Training timesteps: {TOTAL_TIMESTEPS_PER_BOT:,} per bot")
    print(f"Parallel bots:     {N_PARALLEL_BOTS}")
    print(f"GPU workers:       {MAX_WORKERS_GPU}")
    print(f"Risk per trade:    {RISK_PERCENTAGE_PER_TRADE:.1%}")
    print(f"Max drawdown:      {MAX_DRAWDOWN_LIMIT_PCT}%")
    print(f"Data split:        {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {TEST_RATIO:.0%} test")
    print("=" * 70)
    print(f"Reports:           {REPORTS_DIR}")
    print(f"Models:            {MODEL_DIR}")
    print(f"Logs:              {LOG_DIR}")
    print("=" * 70 + "\n")




# ═════════════════════════════════════════════════════════════════════════════
# END OF CONFIGURATION
# ═══════════════════════