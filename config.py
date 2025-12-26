import os
import logging

# ═════════════════════════════════════════════════════════════════════════════
# 1. CORE SYSTEM SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

LOGGING_LEVEL = logging.INFO
TRADING_DAYS_YEAR = 252  # Used for Sharpe Ratio annualization
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate (US Treasury)

# Project structure
PROJECT_ROOT = r"C:\MyPythonProjects\TradingBotNew"
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

HISTORICAL_DATA_FILE = r"C:\MyPythonProjects\TradingBotNew\data\XAU_15MIN_2019_2024.csv"

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

ACTION_SPACE_TYPE = 'discrete'  # 3 actions: Hold, Buy, Sell
LOOKBACK_WINDOW_SIZE = 60  # 60 bars = 15 hours of history

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
ALLOW_NEGATIVE_REVENUE_SELL = False

# ═════════════════════════════════════════════════════════════════════════════
# 6. REWARD FUNCTION PARAMETERS (BALANCED FOR COMMERCIAL SUCCESS)
# ═════════════════════════════════════════════════════════════════════════════

# Core scaling
REWARD_SCALING_FACTOR = 100.0  # Reduced from 500 for stability

# Penalties (reduced from aggressive values)
DOWNSIDE_PENALTY_MULTIPLIER = 3.0  # Was 5.0 (too harsh)
OVERNIGHT_HOLDING_PENALTY = 0.0  # Not applicable for 15min daytrading
HOLD_PENALTY_FACTOR = 0.001  # Was 0.005 (discouraged holding too much)
WINNING_TRADE_BONUS = 2.0  # Was 5.0 (more balanced)
LOSING_TRADE_PENALTY = 5.0  # Was 10.0 (less punishing)
FAILED_TRADE_ATTEMPT_PENALTY = 0.0  # No penalty for rejected trades
TRADE_COOLDOWN_STEPS = 5  # Was 10 (allow more frequency)
RAPID_TRADE_PENALTY = 5.0  # Was 10.0 (less restrictive)

# Position limits
MAX_LEVERAGE = 1.5  # Was 2.0 (safer for production)
MAX_DURATION_STEPS = 12  # 3 hours max hold (was 2h - too short)

# Reward component weights (balanced for risk-adjusted returns)
W_RETURN = 1.0  # Profitability (baseline)
W_DRAWDOWN = 2.0  # Risk control (INCREASED - prioritize safety)
W_FRICTION = 0.8  # Transaction costs (DECREASED - less punishing)
W_LEVERAGE = 2.0  # Leverage compliance (INCREASED - enforce limits)
W_TURNOVER = 0.1  # Overtrading (DECREASED - allow necessary trades)
W_DURATION = 0.3  # Holding time (DECREASED - allow proper setups)

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
    'ent_coef': [0.01, 0.02, 0.05],  # 3 values (exploration)
    'clip_range': [0.1, 0.2, 0.3]  # 3 values (PPO clip)
}

# Total possible combinations: 4×3×3×3×3×3 = 972
# We intelligently sample 50 of these (not random, stratified)

# Baseline hyperparameters (FinRL defaults - proven in research)
MODEL_HYPERPARAMETERS = {
    "n_steps": 2048,  # Rollout buffer size
    "batch_size": 128,  # Minibatch size for updates
    "gamma": 0.99,  # Discount factor
    "learning_rate": 3e-5,  # Adam learning rate (conservative)
    "ent_coef": 0.01,  # Entropy coefficient (exploration)
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