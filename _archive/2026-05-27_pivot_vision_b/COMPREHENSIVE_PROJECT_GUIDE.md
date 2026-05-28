# COMPREHENSIVE PROJECT GUIDE - TradingBOT_Agentic
# ==============================================================================
# A Complete Technical Reference for Understanding Your Trading Bot
# Version: 2.0 | Last Updated: January 2026 | Sprint 3 Ready
# ==============================================================================

## TABLE OF CONTENTS

```
PART 1: PROJECT OVERVIEW
    1.1  Executive Summary
    1.2  System Architecture Overview
    1.3  Technology Stack
    1.4  Project Structure
    1.5  Key Design Decisions

PART 2: CONFIGURATION SYSTEM (config.py)
    2.1  Project Paths
    2.2  Data Configuration
    2.3  Action Space Definition
    2.4  Feature Engineering Settings
    2.5  Risk Management Parameters
    2.6  Reward Function Weights
    2.7  Training Hyperparameters
    2.8  Parallel Training Configuration

PART 3: TRADING ENVIRONMENT (environment.py)
    3.1  Class Overview
    3.2  Position State Management
    3.3  Data Processing Pipeline
    3.4  Observation Space
    3.5  Action Execution
    3.6  Long Position Lifecycle
    3.7  Short Position Lifecycle
    3.8  Portfolio Valuation
    3.9  Reward Function Deep Dive
    3.10 Episode Management

PART 4: RISK MANAGEMENT SYSTEM (risk_manager.py)
    4.1  Dynamic Risk Manager Overview
    4.2  Client Profile System
    4.3  GARCH Volatility Model
    4.4  Kelly Criterion Position Sizing
    4.5  Stop Loss / Take Profit Logic
    4.6  Trailing Stop Loss
    4.7  Position Size Calculation
    4.8  Market Regime Detection

PART 5: SMART MONEY CONCEPTS ENGINE (strategy_features.py)
    5.1  Technical Indicators
    5.2  Fractal Detection (Swing Points)
    5.3  Fair Value Gaps (FVG)
    5.4  Break of Structure (BOS)
    5.5  Change of Character (CHOCH)
    5.6  Order Blocks

PART 6: AGENTIC SYSTEM
    6.1  Event-Driven Architecture (events.py)
    6.2  Base Agent Framework (base_agent.py)
    6.3  Kill Switch System (kill_switch.py)
    6.4  Multi-Agent Orchestrator (orchestrator.py)
    6.5  Portfolio Risk Manager (portfolio_risk.py)
    6.6  Ensemble Risk Model (ensemble_risk_model.py)

PART 7: TRAINING PIPELINE
    7.1  Agent Trainer (agent_trainer.py)
    7.2  Parallel Training (parallel_training.py)
    7.3  Walk-Forward Validation
    7.4  Hyperparameter Search
    7.5  Model Selection

PART 8: SECURITY AND RELIABILITY FIXES (Sprint 2)
    8.1  Race Condition Fixes
    8.2  Balance Protection
    8.3  Transaction Rollback
    8.4  Kill Switch Hardening
    8.5  Thread Safety Improvements

PART 9: MATHEMATICAL FOUNDATIONS
    9.1  PPO Algorithm
    9.2  GARCH(1,1) Model
    9.3  Kelly Criterion Formula
    9.4  VaR Calculations
    9.5  Sharpe Ratio

PART 10: PRACTICAL EXAMPLES
    10.1 Complete Trading Cycle
    10.2 Risk Management in Action
    10.3 Event Flow Example
    10.4 Training a New Bot
```

---

# ==============================================================================
# PART 1: PROJECT OVERVIEW
# ==============================================================================

## 1.1 Executive Summary

**TradingBOT_Agentic** is a production-grade Reinforcement Learning (RL) trading
system designed for XAU/USD (Gold) trading on 15-minute timeframes. The system
combines cutting-edge machine learning with institutional-grade risk management
and a hierarchical multi-agent architecture for safety and modularity.

### What This System Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRADING DECISION FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. MARKET DATA                                                            │
│      │                                                                      │
│      ▼                                                                      │
│   ┌─────────────────────┐                                                   │
│   │  Smart Money Engine │  ──► Technical Indicators (RSI, MACD, BB, ATR)   │
│   │  (Feature Generator)│  ──► SMC Signals (FVG, BOS, CHOCH, Order Blocks) │
│   └─────────────────────┘                                                   │
│      │                                                                      │
│      ▼                                                                      │
│   ┌─────────────────────┐                                                   │
│   │   Trading Env       │  ──► Normalizes features (MinMaxScaler)          │
│   │   (Gymnasium)       │  ──► Creates observation vector                   │
│   └─────────────────────┘                                                   │
│      │                                                                      │
│      ▼                                                                      │
│   ┌─────────────────────┐                                                   │
│   │    PPO Agent        │  ──► Neural network policy                        │
│   │    (Decision Maker) │  ──► Outputs: HOLD/OPEN_LONG/CLOSE_LONG/         │
│   └─────────────────────┘              OPEN_SHORT/CLOSE_SHORT              │
│      │                                                                      │
│      ▼                                                                      │
│   ┌─────────────────────┐                                                   │
│   │   Risk Manager      │  ──► Position sizing (Kelly Criterion)           │
│   │   (Safety Layer)    │  ──► SL/TP/TSL calculation                       │
│   └─────────────────────┘  ──► Leverage limits                             │
│      │                                                                      │
│      ▼                                                                      │
│   ┌─────────────────────┐                                                   │
│   │   Trade Execution   │  ──► Apply spread, slippage, commission          │
│   │                     │  ──► Update portfolio                             │
│   └─────────────────────┘                                                   │
│      │                                                                      │
│      ▼                                                                      │
│   ┌─────────────────────┐                                                   │
│   │   Reward Calc       │  ──► Profitability reward                        │
│   │                     │  ──► Risk penalties                               │
│   └─────────────────────┘  ──► Trade bonuses                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Capabilities

| Capability              | Description                                          |
|-------------------------|------------------------------------------------------|
| **Long/Short Trading**  | Can profit from both rising and falling markets      |
| **Dynamic Risk**        | GARCH volatility + Kelly Criterion position sizing   |
| **SMC Analysis**        | Fair Value Gaps, Order Blocks, Break of Structure    |
| **Multi-Agent Safety**  | Kill switch, circuit breakers, event-driven alerts   |
| **Parallel Training**   | Train 50+ bots simultaneously with different params  |
| **Walk-Forward Valid**  | Proper out-of-sample testing for realistic results   |

### Performance Targets

| Metric              | Target       | Notes                                |
|---------------------|--------------|--------------------------------------|
| Sharpe Ratio        | > 1.5        | Risk-adjusted return                 |
| Calmar Ratio        | > 2.0        | Return / Max Drawdown                |
| Max Drawdown        | < 15%        | Capital preservation                 |
| Win Rate            | > 52%        | Minimum for profitability            |
| Profit Factor       | > 1.3        | Gross Profit / Gross Loss            |

---

## 1.2 System Architecture Overview

The system follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 1: CONFIGURATION                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  config.py - Central configuration hub                              │   │
│  │  • All hyperparameters in one place                                 │   │
│  │  • Environment variables support                                    │   │
│  │  • Validation on startup                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 2: DATA PROCESSING                          │
│  ┌───────────────────────┐    ┌────────────────────────────────────────┐   │
│  │  Raw OHLCV Data       │───►│  SmartMoneyEngine (strategy_features.py)│   │
│  │  (CSV/API)            │    │  • Technical indicators                 │   │
│  └───────────────────────┘    │  • SMC features                         │   │
│                               └────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 3: ENVIRONMENT                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TradingEnv (environment.py) - Gymnasium-compatible environment     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │ Observation │  │   Action    │  │   Reward    │  │  Episode   │ │   │
│  │  │   Space     │  │  Execution  │  │ Calculation │  │ Management │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DynamicRiskManager (risk_manager.py)                               │   │
│  │  • GARCH volatility forecasting                                     │   │
│  │  • Position sizing                                                  │   │
│  │  • SL/TP/TSL management                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 4: AGENT                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PPO Agent (stable-baselines3)                                      │   │
│  │  • MlpPolicy neural network                                         │   │
│  │  • Actor-Critic architecture                                        │   │
│  │  • Proximal Policy Optimization                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 5: AGENTIC SYSTEM                           │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐ │
│  │   Event Bus   │  │  Orchestrator │  │  Kill Switch  │  │   Alerts    │ │
│  │  (events.py)  │  │(orchestrator) │  │(kill_switch)  │  │   System    │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 6: TRAINING                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Parallel Training System (parallel_training.py)                    │   │
│  │  • Multi-process execution                                          │   │
│  │  • Hyperparameter search                                            │   │
│  │  • Walk-forward validation                                          │   │
│  │  • Best model selection                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.3 Technology Stack

### Core Dependencies

| Package             | Version   | Purpose                                          |
|---------------------|-----------|--------------------------------------------------|
| **Python**          | 3.10+     | Base language                                    |
| **gymnasium**       | 0.29+     | RL environment interface (OpenAI Gym successor)  |
| **stable-baselines3**| 2.0+     | PPO implementation                               |
| **PyTorch**         | 2.0+      | Neural network backend                           |
| **pandas**          | 2.0+      | Data manipulation                                |
| **numpy**           | 1.24+     | Numerical computations                           |
| **scikit-learn**    | 1.3+      | MinMaxScaler, metrics                            |
| **ta**              | 0.10+     | Technical analysis library                       |
| **arch**            | 6.0+      | GARCH volatility modeling                        |
| **scipy**           | 1.11+     | Statistical functions                            |
| **rich**            | 13.0+     | Beautiful console output                         |

### Why These Choices?

**Gymnasium over OpenAI Gym:**
- Gymnasium is the actively maintained successor to OpenAI Gym
- Better type hints and documentation
- More consistent API (reset returns (obs, info) tuple)
- Active community support

**stable-baselines3 for PPO:**
- Production-ready implementations
- Excellent documentation
- Built-in callbacks for monitoring
- Easy integration with custom environments

**PyTorch Backend:**
- Dynamic computation graphs (easier debugging)
- Strong GPU support
- Excellent for research and production

---

## 1.4 Project Structure

```
TradingBOT_Agentic/
│
├── config.py                              # [507 lines] Central configuration
│   └── All hyperparameters, paths, constants
│
├── parallel_training.py                   # [250+ lines] Multi-bot training
│   └── ProcessPoolExecutor, walk-forward validation
│
├── requirements.txt                       # Dependencies
│
├── src/
│   ├── __init__.py
│   │
│   ├── agent_trainer.py                   # [400+ lines] PPO training manager
│   │   └── Training loops, callbacks, evaluation
│   │
│   ├── evaluate_agent.py                  # Model evaluation metrics
│   │
│   ├── weekly_adaptation.py               # Adaptive retraining system
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   │
│   │   ├── environment.py                 # [1,900 lines] Main trading env
│   │   │   ├── TradingEnv class
│   │   │   ├── PositionState enum
│   │   │   ├── Long/Short trade execution
│   │   │   ├── Reward calculation
│   │   │   └── Episode management
│   │   │
│   │   ├── risk_manager.py                # [450 lines] Dynamic risk management
│   │   │   ├── GARCH volatility
│   │   │   ├── Kelly Criterion
│   │   │   ├── SL/TP/TSL logic
│   │   │   └── Position sizing
│   │   │
│   │   ├── strategy_features.py           # [400 lines] Feature engineering
│   │   │   ├── Technical indicators
│   │   │   └── Smart Money Concepts
│   │   │
│   │   └── multi_timeframe_features.py    # Multi-timeframe analysis
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   │
│   │   ├── base_agent.py                  # Abstract agent foundation
│   │   │   └── Lifecycle, metrics, audit logging
│   │   │
│   │   ├── config.py                      # Agent configurations
│   │   │   └── Dataclasses with validation
│   │   │
│   │   ├── events.py                      # [600+ lines] Event-driven system
│   │   │   ├── EventBus class
│   │   │   ├── Event types
│   │   │   ├── Deduplication
│   │   │   └── Persistence
│   │   │
│   │   ├── kill_switch.py                 # [800+ lines] Emergency halt
│   │   │   ├── Circuit breakers
│   │   │   ├── Hard limits
│   │   │   ├── Recovery manager
│   │   │   └── Alert system
│   │   │
│   │   ├── orchestrator.py                # Multi-agent coordination
│   │   │   └── Thread-safe agent management
│   │   │
│   │   ├── portfolio_risk.py              # VaR calculations
│   │   │
│   │   ├── ensemble_risk_model.py         # ML ensemble (XGBoost, LSTM, MLP)
│   │   │
│   │   └── risk_sentinel.py               # Trade validation guardian
│   │
│   └── tests/
│       ├── monitor_training.py            # Trade logging utility
│       └── test_*.py                      # Unit tests
│
├── data/
│   └── XAU_15MIN_2019_2024.csv           # Historical price data
│
├── trained_models/                        # Saved PPO models
│
├── logs/                                  # Training logs
│
└── results/                               # Evaluation results
```

---

## 1.5 Key Design Decisions

### Decision 1: 5-Action Discrete Space (Not Continuous)

**Why Discrete Actions?**

```python
# Our action space
ACTION_HOLD = 0           # Do nothing
ACTION_OPEN_LONG = 1      # Buy to open long position
ACTION_CLOSE_LONG = 2     # Sell to close long position
ACTION_OPEN_SHORT = 3     # Sell to open short position
ACTION_CLOSE_SHORT = 4    # Buy to cover short position
```

**Rationale:**
1. **Simplicity**: Clear, interpretable actions
2. **PPO Performance**: PPO works better with discrete actions
3. **No Position Sizing in Action**: Position size determined by risk manager
4. **Separation of Concerns**: Agent decides WHAT, risk manager decides HOW MUCH

**Alternative Considered (Rejected):**
- Continuous action space with position size
- Problem: Agent would need to learn risk management + timing simultaneously
- Result: Slower convergence, harder to interpret

---

### Decision 2: Fixed Episode Length (500 Steps)

**Why Fixed Length?**

```python
FIXED_EPISODE_LENGTH = 500          # ~5 days of 15-min bars
USE_FIXED_EPISODE_LENGTH = True     # Critical for PPO stability
```

**Rationale:**
1. **PPO Stability**: PPO estimates advantages over trajectories
   - Variable lengths = inconsistent gradient estimates
   - Fixed lengths = stable learning signal

2. **Reproducibility**: Same episode structure across all bots
   - Fair comparison during hyperparameter search

3. **Memory Efficiency**: Predictable memory usage
   - Can batch episodes efficiently

**The Math:**
```
500 steps × 15 min/step = 7,500 minutes = 125 hours ≈ 5 trading days
```

---

### Decision 3: Tanh-Squashed Rewards

**Why Squash Rewards?**

```python
# In _calculate_reward():
normalized_reward = np.tanh(combined_reward * self.reward_tanh_scale)
scaled_reward = normalized_reward * self.reward_output_scale
final_reward = np.clip(scaled_reward, -20.0, 20.0)
```

**Rationale:**
1. **Prevents Exploding Gradients**: Extreme rewards destabilize PPO
2. **Smooth Gradients**: Tanh has smooth derivatives everywhere
3. **Bounded Output**: Guarantees [-20, +20] range

**Visual:**
```
Input (combined_reward):  -100  -10   -1    0    +1   +10  +100
                            │    │    │    │    │     │     │
After tanh(x * 0.3):      -1.0 -1.0 -0.3  0.0 +0.3  +1.0  +1.0
                            │    │    │    │    │     │     │
After * 5.0:              -5.0 -5.0 -1.5  0.0 +1.5  +5.0  +5.0
```

---

### Decision 4: Position Size from Risk Manager (Not Agent)

**Why Separate Position Sizing?**

```python
# Agent decides: WHAT action to take
action = model.predict(observation)  # Returns 0-4

# Risk manager decides: HOW MUCH
position_size = risk_manager.calculate_adaptive_position_size(
    client_id=client_id,
    account_equity=balance,
    atr_stop_distance=sl_distance,
    win_prob=0.5,
    risk_reward_ratio=1.0,
    current_price=price,
    max_leverage=max_leverage
)
```

**Rationale:**
1. **Specialization**: Each component does one thing well
2. **Safety**: Risk limits enforced regardless of agent behavior
3. **Adaptability**: Can change sizing rules without retraining
4. **Compliance**: Position limits are regulatory requirements

---

### Decision 5: Event-Driven Multi-Agent Architecture

**Why Events?**

```python
# Instead of direct calls:
# BAD: orchestrator.notify_risk_sentinel(trade)

# We use events:
# GOOD: event_bus.publish(Event(type=TRADE_PROPOSED, data=trade))
```

**Rationale:**
1. **Loose Coupling**: Agents don't need to know about each other
2. **Extensibility**: Add new agents without changing existing code
3. **Debugging**: All events are logged and can be replayed
4. **Resilience**: Failed agent doesn't crash the system

---

# ==============================================================================
# PART 2: CONFIGURATION SYSTEM (config.py)
# ==============================================================================

## 2.1 Project Paths

All paths are relative to the project root, automatically detected:

```python
# config.py - Lines 1-20

import os

# Auto-detect project root (where config.py lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'trained_models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
```

**Why Auto-Detection?**
- Works regardless of where script is run from
- No hardcoded paths that break on different machines
- Easy to move project to new location

---

## 2.2 Data Configuration

```python
# config.py - Data settings

# Historical data file
HISTORICAL_DATA_FILE = os.path.join(DATA_DIR, "XAU_15MIN_2019_2024.csv")

# Column mapping (your CSV columns → standard names)
OHLCV_COLUMNS = {
    "timestamp": "Date",      # Datetime column
    "open": "Open",           # Opening price
    "high": "High",           # Highest price
    "low": "Low",             # Lowest price
    "close": "Close",         # Closing price
    "volume": "Volume"        # Trading volume
}

# Data splits (CRITICAL for avoiding overfitting)
TRAIN_RATIO = 0.70    # 70% for training
VAL_RATIO = 0.15      # 15% for validation (hyperparameter tuning)
TEST_RATIO = 0.15     # 15% for final testing (NEVER touch during training)

# Important dates
TRAIN_END_DATE = "2023-06-30 23:59:00"  # Last training data point
```

**Data Split Strategy:**

```
2019 ─────────────────────────────────────────────────────────── 2024
│                                                                   │
│◄──────────── TRAIN (70%) ───────────►│◄─ VAL (15%)─►│◄─TEST(15%)─►│
│                                       │              │             │
│  Learn patterns here                  │ Tune params  │ Final eval  │
│  (Backpropagation)                    │ (No grad)    │ (Untouched) │
└───────────────────────────────────────┴──────────────┴─────────────┘
```

**CRITICAL WARNING:**
```
NEVER use test data during training or hyperparameter tuning!
This causes "data leakage" and produces overfitted models that
fail catastrophically in live trading.
```

---

## 2.3 Action Space Definition

```python
# config.py - Action space

# Number of possible actions
NUM_ACTIONS = 5

# Action indices
ACTION_HOLD = 0           # Do nothing - stay in current state
ACTION_OPEN_LONG = 1      # Buy to open long position (profit when price UP)
ACTION_CLOSE_LONG = 2     # Sell to close long position
ACTION_OPEN_SHORT = 3     # Sell to open short position (profit when price DOWN)
ACTION_CLOSE_SHORT = 4    # Buy to cover short position

# Human-readable names for logging
ACTION_NAMES = {
    0: 'HOLD',
    1: 'OPEN_LONG',
    2: 'CLOSE_LONG',
    3: 'OPEN_SHORT',
    4: 'CLOSE_SHORT'
}

# Position states
POSITION_FLAT = 0         # No open position
POSITION_LONG = 1         # Holding long position
POSITION_SHORT = -1       # Holding short position

# Short selling toggle
ENABLE_SHORT_SELLING = True  # Set to False to disable shorts
```

**Action-Position State Machine:**

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
                    ▼                                          │
              ┌──────────┐                                     │
              │   FLAT   │◄─────────────────────┐              │
              │ (pos=0)  │                      │              │
              └──────────┘                      │              │
                │      │                        │              │
   OPEN_LONG(1)│      │OPEN_SHORT(3)            │              │
                │      │                        │              │
                ▼      ▼                        │              │
         ┌──────────┐  ┌──────────┐             │              │
         │   LONG   │  │  SHORT   │             │              │
         │ (pos=1)  │  │ (pos=-1) │             │              │
         └──────────┘  └──────────┘             │              │
                │             │                 │              │
   CLOSE_LONG(2)│             │CLOSE_SHORT(4)   │              │
                │             │                 │              │
                └─────────────┴─────────────────┘              │
                              │                                │
                              └────────────────────────────────┘
                                        HOLD(0)
                                   (stays in current state)
```

**Invalid Action Handling:**
```python
# These actions are converted to HOLD:
OPEN_LONG when position != FLAT     # Already in position
OPEN_SHORT when position != FLAT    # Already in position
CLOSE_LONG when position != LONG    # No long to close
CLOSE_SHORT when position != SHORT  # No short to close
OPEN_SHORT when ENABLE_SHORT_SELLING = False
```

---

## 2.4 Feature Engineering Settings

```python
# config.py - Features

# Features used in observation vector
FEATURES = [
    # ============ OHLCV Base (5 features) ============
    'Open',           # Opening price of bar
    'High',           # Highest price during bar
    'Low',            # Lowest price during bar
    'Close',          # Closing price of bar
    'Volume',         # Trading volume

    # ============ Technical Indicators (10 features) ============
    'RSI',            # Relative Strength Index (momentum)
    'MACD_Diff',      # MACD histogram (MACD - Signal)
    'MACD_line',      # MACD line
    'MACD_signal',    # Signal line
    'BB_L',           # Bollinger Band Lower
    'BB_M',           # Bollinger Band Middle (20-SMA)
    'BB_H',           # Bollinger Band Upper
    'ATR',            # Average True Range (volatility)
    'SPREAD',         # High - Low (bar range)
    'BODY_SIZE',      # |Open - Close| (candle body)

    # ============ Smart Money Concepts (11 features) ============
    'UP_FRACTAL',     # Swing high detected (1 or 0)
    'DOWN_FRACTAL',   # Swing low detected (1 or 0)
    'FVG_SIGNAL',     # Fair Value Gap signal (+1 bullish, -1 bearish)
    'FVG_SIZE_NORM',  # Normalized FVG size
    'BOS_SIGNAL',     # Break of Structure (+1 bullish, -1 bearish)
    'CHOCH_SIGNAL',   # Change of Character (+1 bullish, -1 bearish)
    'BULLISH_OB_HIGH',# Bullish Order Block high price
    'BULLISH_OB_LOW', # Bullish Order Block low price
    'BEARISH_OB_HIGH',# Bearish Order Block high price
    'BEARISH_OB_LOW', # Bearish Order Block low price
    'OB_STRENGTH_NORM'# Order Block strength (normalized)
]

# Total features: 26
# Lookback window: 30 bars
# Observation size: 26 * 30 + 3 (portfolio state) = 783 dimensions
```

**Indicator Configuration:**

```python
# config.py - SMC_CONFIG

SMC_CONFIG = {
    # RSI Settings
    "RSI_WINDOW": 7,        # 7-period RSI (faster for 15-min)

    # MACD Settings
    "MACD_FAST": 8,         # Fast EMA period
    "MACD_SLOW": 17,        # Slow EMA period
    "MACD_SIGNAL": 9,       # Signal line period

    # Bollinger Bands
    "BB_WINDOW": 20,        # 20-period moving average
    "BB_STD": 2,            # 2 standard deviations

    # ATR (Volatility)
    "ATR_WINDOW": 7,        # 7-period ATR

    # Fractals (Swing Points)
    "FRACTAL_WINDOW": 2,    # Look 2 bars each side

    # Fair Value Gap
    "FVG_THRESHOLD": 0.0,   # Minimum gap size (0 = any gap)
}
```

---

## 2.5 Risk Management Parameters

```python
# config.py - Risk settings

# ============ Per-Trade Risk ============
RISK_PERCENTAGE_PER_TRADE = 0.01    # 1% of equity at risk per trade
                                    # If equity = $1000, max loss = $10

# ============ Take Profit / Stop Loss ============
TAKE_PROFIT_PERCENTAGE = 0.02       # 2% profit target
STOP_LOSS_PERCENTAGE = 0.01         # 1% stop loss
                                    # Risk:Reward = 1:2

# ============ Trailing Stop Loss ============
TSL_START_PROFIT_MULTIPLIER = 1.0   # Activate TSL after 1x risk in profit
TSL_TRAIL_DISTANCE_MULTIPLIER = 0.5 # Trail at 0.5x ATR behind price

# ============ Position Limits ============
MAX_LEVERAGE = 1.0                  # No leverage (100% equity max)
MAX_DURATION_STEPS = 40             # 40 bars = 10 hours max hold time

# ============ Account Protection ============
MAX_DRAWDOWN_LIMIT_PCT = 10.0       # 10% max drawdown before halt
MINIMUM_ALLOWED_BALANCE = 100.0     # Episode ends if balance < $100

# ============ Transaction Costs ============
TRANSACTION_FEE_PERCENTAGE = 0.0005  # 0.05% spread
SLIPPAGE_PERCENTAGE = 0.0001         # 0.01% slippage
TRADE_COMMISSION_PCT_OF_TRADE = 0.0005  # 0.05% commission on trade value
TRADE_COMMISSION_MIN_PCT_CAPITAL = 0.0001  # Minimum commission
```

**Risk Calculation Example:**

```
Starting Equity: $1,000
Risk Per Trade: 1% = $10

If we buy at $2000/oz with 1% stop loss:
  Stop Loss Price = $2000 × (1 - 0.01) = $1,980
  Stop Distance = $20 per oz

Max Position Size = $10 (max risk) / $20 (stop distance) = 0.5 oz

Position Value = 0.5 oz × $2000 = $1,000
Leverage = $1,000 / $1,000 equity = 1.0x (within limit)
```

---

## 2.6 Reward Function Weights

```python
# config.py - Reward weights

# ============ Primary Reward Scaling ============
REWARD_SCALING_FACTOR = 100.0    # 1% return = 1.0 base reward
REWARD_TANH_SCALE = 0.3          # Tanh sensitivity (lower = more compressed)
REWARD_OUTPUT_SCALE = 5.0        # Final output multiplier

# ============ Component Weights ============
W_RETURN = 1.0      # Weight for profitability (primary driver)
W_DRAWDOWN = 0.5    # Weight for drawdown penalty
W_FRICTION = 0.1    # Weight for transaction cost penalty
W_LEVERAGE = 1.0    # Weight for leverage violation penalty
W_TURNOVER = 0.0    # Weight for churning penalty (disabled)
W_DURATION = 0.1    # Weight for holding too long

# ============ Trade Outcome Bonuses ============
WINNING_TRADE_BONUS = 0.5        # Bonus for closing profitable trade
LOSING_TRADE_PENALTY = 0.0       # No extra penalty (loss is already negative)

# ============ Behavioral Penalties ============
HOLD_PENALTY_FACTOR = 0.005      # Small penalty for excessive holding
TRADE_COOLDOWN_STEPS = 5         # Must wait 5 bars between trades
RAPID_TRADE_PENALTY = 1.0        # Penalty for trading too fast
```

**Reward Formula (Simplified):**

```
reward = tanh((profit_reward - penalties + bonuses) × 0.3) × 5.0

Where:
  profit_reward = log(net_worth / prev_net_worth) × 100

  penalties = drawdown_penalty + friction_penalty + leverage_penalty
              + duration_penalty + invalid_action_penalty + hold_penalty

  bonuses = winning_trade_bonus (if profitable close)
```

---

## 2.7 Training Hyperparameters

```python
# config.py - PPO training settings

# ============ Training Duration ============
TOTAL_TIMESTEPS_PER_BOT = 1_500_000  # 1.5M steps per bot
                                      # ~3000 episodes of 500 steps

# ============ Early Stopping ============
EARLY_STOPPING_PATIENCE = 5    # Stop if no improvement for 5 evals
EVAL_FREQ = 10_000             # Evaluate every 10K steps
N_EVAL_EPISODES = 5            # Run 5 episodes per evaluation

# ============ PPO Hyperparameters ============
MODEL_HYPERPARAMETERS = {
    # Rollout settings
    "n_steps": 2048,           # Steps before update (2048 default)
    "batch_size": 128,         # Minibatch size for updates

    # Discount and advantage
    "gamma": 0.99,             # Discount factor (future reward importance)
    "gae_lambda": 0.95,        # GAE lambda (bias-variance tradeoff)

    # Learning
    "learning_rate": 3e-5,     # Conservative LR for stability
    "n_epochs": 10,            # SGD epochs per update

    # PPO-specific
    "clip_range": 0.2,         # PPO clip parameter
    "ent_coef": 0.05,          # Entropy bonus (exploration)
    "vf_coef": 0.5,            # Value function coefficient
    "max_grad_norm": 0.5,      # Gradient clipping
}
```

**Hyperparameter Explanations:**

| Parameter      | Value  | Effect if Too Low           | Effect if Too High        |
|----------------|--------|------------------------------|---------------------------|
| learning_rate  | 3e-5   | Slow learning                | Unstable, divergence      |
| n_steps        | 2048   | High variance, unstable      | Slow updates              |
| batch_size     | 128    | Noisy gradients              | Misses fine patterns      |
| gamma          | 0.99   | Short-sighted agent          | Delayed reward issues     |
| ent_coef       | 0.05   | Premature convergence        | Random behavior           |
| clip_range     | 0.2    | Very conservative updates    | Large policy swings       |

---

## 2.8 Parallel Training Configuration

```python
# config.py - Parallel training

# ============ Bot Count ============
N_PARALLEL_BOTS = 50           # Train 50 different configurations
MAX_WORKERS_GPU = 2            # Max concurrent GPU processes
MAX_WORKERS_CPU = 4            # Max concurrent CPU processes

# ============ Hyperparameter Search Space ============
HYPERPARAM_SEARCH_SPACE = {
    'learning_rate': [1e-5, 3e-5, 5e-5, 1e-4],    # 4 options
    'n_steps': [1024, 2048, 4096],                 # 3 options
    'batch_size': [64, 128, 256],                  # 3 options
    'gamma': [0.99, 0.995, 0.999],                 # 3 options
    'ent_coef': [0.02, 0.05, 0.10],                # 3 options
    'clip_range': [0.1, 0.2, 0.3],                 # 3 options
    'reward_tanh_scale': [0.2, 0.3, 0.4],          # 3 options
    'reward_output_scale': [3.0, 5.0, 7.0],        # 3 options
}

# Total combinations: 4 × 3 × 3 × 3 × 3 × 3 × 3 × 3 = 8,748
# We sample 50 random combinations

# ============ Selection Criteria ============
EVALUATION_METRIC = 'sharpe_ratio'     # Primary metric for bot selection
MIN_ACCEPTABLE_SHARPE = 1.5            # Minimum Sharpe to consider
MIN_ACCEPTABLE_CALMAR = 2.0            # Minimum Calmar ratio
MAX_ACCEPTABLE_DD = 0.15               # Maximum 15% drawdown
```

**Bot Selection Process:**

```
1. Train 50 bots with different hyperparameters
2. Evaluate each on validation set
3. Filter by minimum criteria:
   - Sharpe > 1.5
   - Calmar > 2.0
   - Max DD < 15%
4. Rank by Sharpe ratio
5. Select top 3 for final testing
6. Best performer on test set = Production model
```

---

# ==============================================================================
# PART 3: TRADING ENVIRONMENT (environment.py)
# ==============================================================================

## 3.1 Class Overview

The `TradingEnv` class is the heart of the system - a Gymnasium-compatible
environment that simulates trading with realistic market dynamics.

```python
class TradingEnv(gym.Env):
    """
    Gymnasium trading environment for XAU/USD with long/short support.

    Features:
    - 5-action discrete space (HOLD, OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT)
    - Realistic transaction costs (spread, slippage, commission)
    - Dynamic risk management (SL/TP/TSL)
    - Research-grade reward function

    Observation Space:
        Box(shape=(783,), dtype=float32)
        = 26 features × 30 lookback + 3 portfolio metrics

    Action Space:
        Discrete(5)
        0=HOLD, 1=OPEN_LONG, 2=CLOSE_LONG, 3=OPEN_SHORT, 4=CLOSE_SHORT
    """
```

**Key Attributes:**

| Attribute           | Type                 | Description                          |
|---------------------|----------------------|--------------------------------------|
| `balance`           | float                | Cash available (protected property)  |
| `stock_quantity`    | float                | Position size (+long, -short)        |
| `net_worth`         | float                | Total portfolio value                |
| `position_type`     | int                  | FLAT(0), LONG(1), SHORT(-1)         |
| `entry_price`       | float                | Entry price of current position      |
| `risk_manager`      | DynamicRiskManager   | SL/TP/TSL calculator                |
| `scaler`            | MinMaxScaler         | Feature normalization                |
| `current_step`      | int                  | Current bar index                    |
| `episode_reward`    | float                | Cumulative reward this episode       |

---

## 3.2 Position State Management

**Security Fix: Type-Safe Position State**

```python
# environment.py - Lines 23-45

class PositionState(IntEnum):
    """
    Type-safe position state enum.

    Using IntEnum for backward compatibility with existing code that uses
    integer comparisons, while providing type safety for new code.
    """
    FLAT = 0      # No position
    LONG = 1      # Long position (profit when price goes UP)
    SHORT = -1    # Short position (profit when price goes DOWN)

    @classmethod
    def from_value(cls, value: int) -> 'PositionState':
        """Convert integer to PositionState with validation."""
        for state in cls:
            if state.value == value:
                return state
        raise ValueError(f"Invalid position state value: {value}")

    def is_valid(self) -> bool:
        """Check if this is a valid position state."""
        return self in PositionState
```

**Protected Balance Property (Security Fix):**

```python
# environment.py - Lines 185-220

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
        raise ValueError(f"Balance cannot be negative: {value}")

    # Check minimum balance threshold
    min_balance = getattr(self, 'minimum_allowed_balance', 0.0)
    if value < min_balance and value > 0:
        logging.warning(f"Balance {value:.2f} below minimum {min_balance:.2f}")

    self._balance = value
```

**Why Protected Properties?**

```
BEFORE (vulnerable):
  env.balance = "not a number"  # Corrupts state
  env.balance = float('nan')    # Silent corruption
  env.balance = -1000           # Invalid negative balance

AFTER (secure):
  env.balance = "not a number"  # Raises TypeError
  env.balance = float('nan')    # Raises ValueError
  env.balance = -1000           # Raises ValueError (unless allowed)
```

---

## 3.3 Data Processing Pipeline

The `_process_data()` method transforms raw OHLCV data into ML-ready features.

```python
def _process_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    7-Step Data Processing Pipeline:

    1. Column normalization (standardize names)
    2. Feature generation (SmartMoneyEngine)
    3. Column capitalization
    4. Infinity replacement
    5. Intelligent NaN handling
    6. Final validation
    7. Quality report
    """
```

**Step-by-Step Breakdown:**

```
STEP 1: COLUMN NORMALIZATION
────────────────────────────
Input:  {'Gmt time': ..., 'open': ..., 'HIGH': ...}
Output: {'Original_Timestamp': ..., 'open': ..., 'high': ...}

- Preserve original timestamp
- Convert all OHLCV columns to lowercase


STEP 2: FEATURE GENERATION
──────────────────────────
Input:  Raw OHLCV DataFrame
Output: DataFrame with 26+ features

SmartMoneyEngine adds:
  - Technical indicators (RSI, MACD, BB, ATR)
  - SMC features (FVG, BOS, CHOCH, Order Blocks)


STEP 3: COLUMN CAPITALIZATION
─────────────────────────────
Input:  {'open': ..., 'rsi': ..., 'bos_signal': ...}
Output: {'Open': ..., 'RSI': ..., 'BOS_SIGNAL': ...}

- Capitalize OHLCV for consistency
- Preserve indicator names


STEP 4: INFINITY REPLACEMENT
────────────────────────────
df.replace([np.inf, -np.inf], np.nan, inplace=True)

- Infinities from division by zero → NaN
- NaN is easier to handle downstream


STEP 5: NaN HANDLING (CRITICAL!)
────────────────────────────────
Three strategies based on column type:

A. CRITICAL COLUMNS (Drop rows with NaN):
   - Price columns: Open, High, Low, Close
   - Key indicators: RSI, MACD_line, ATR

B. SLOW INDICATORS (Forward/Backward fill):
   - RSI, MACD, BB, ATR
   - These change slowly, interpolation is safe

C. EVENT SIGNALS (Fill with 0):
   - FVG_SIGNAL, BOS_SIGNAL, CHOCH_SIGNAL
   - NaN means "no signal" = 0


STEP 6: FINAL VALIDATION
────────────────────────
Checks:
  - No NaN remaining (FAIL if any)
  - No Inf remaining (FAIL if any)
  - No invalid prices (≤0) (FAIL if any)
  - Enough data for lookback window


STEP 7: QUALITY REPORT
──────────────────────
Prints:
  - Final row count
  - Column count
  - Feature availability by category
```

---

## 3.4 Observation Space

The observation vector contains everything the agent needs to make decisions.

```python
def _get_obs(self) -> np.ndarray:
    """
    Construct observation vector.

    Structure:
        [Feature_1_t-29, Feature_1_t-28, ..., Feature_1_t,
         Feature_2_t-29, Feature_2_t-28, ..., Feature_2_t,
         ...
         Feature_26_t-29, Feature_26_t-28, ..., Feature_26_t,
         normalized_balance,
         normalized_position_value,
         normalized_net_worth]

    Total size: 26 features × 30 timesteps + 3 = 783
    """
```

**Observation Construction:**

```python
# 1. Extract lookback window (last 30 bars)
start_idx = self.current_step - self.lookback_window_size + 1
obs_df = self.df.iloc[start_idx:self.current_step + 1]

# 2. Get feature values
features_data = obs_df[self.features].values  # Shape: (30, 26)

# 3. Scale to [0, 1] range
scaled_features = self.scaler.transform(features_data)

# 4. Flatten to 1D
flat_obs = scaled_features.flatten()  # Shape: (780,)

# 5. Add portfolio state
current_equity = balance + stock_quantity * current_price

normalized_balance = balance / initial_balance
normalized_position_value = (stock_quantity * current_price) / initial_balance
normalized_net_worth = current_equity / initial_balance

# 6. Concatenate
observation = np.append(flat_obs, [
    normalized_balance,
    normalized_position_value,
    normalized_net_worth
])  # Shape: (783,)
```

**Why Normalize Portfolio State?**

```
Without normalization:
  balance = 1523.47        # Raw dollar value
  position_value = 476.53  # Changes wildly with price

With normalization (divide by initial_balance=1000):
  normalized_balance = 1.52        # ~1.5x initial
  normalized_position_value = 0.48 # 48% in position
  normalized_net_worth = 2.00      # Doubled equity

Benefits:
  - Scale-invariant: Same meaning for $1K or $1M accounts
  - Bounded: Typically in [0, 3] range
  - Comparable: Easy to interpret position sizes
```

---

## 3.5 Action Execution

The `step()` method is the core simulation loop.

```python
def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """
    Execute one step in the environment.

    Args:
        action: 0-4 (HOLD, OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT)

    Returns:
        observation: Next state (783-dim vector)
        reward: Scalar reward
        done: Episode finished?
        truncated: Episode truncated (max steps)?
        info: Debug information dict
    """
```

**Step Execution Flow:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                          step(action)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. TRACK PREVIOUS STATE                                            │
│     previous_net_worth = self.net_worth                             │
│     previous_drawdown = peak_nav - net_worth                        │
│                                                                     │
│  2. INCREMENT STEP                                                  │
│     current_step += 1                                               │
│     Check if done (current_step >= max_steps)                       │
│                                                                     │
│  3. GET MARKET DATA                                                 │
│     current_price = df.iloc[current_step]['Close']                  │
│     current_atr = df.iloc[current_step]['ATR']                      │
│                                                                     │
│  4. FORCE CLOSE ON EPISODE END                                      │
│     if done and position != FLAT:                                   │
│         action = CLOSE_LONG or CLOSE_SHORT                          │
│                                                                     │
│  5. VALIDATE ACTION                                                 │
│     Convert invalid actions to HOLD                                 │
│     Track invalid action count                                      │
│                                                                     │
│  6. CHECK SL/TP/TSL                                                 │
│     if in position:                                                 │
│         Update trailing stop                                        │
│         Check for stop loss or take profit hit                      │
│         Override action if exit triggered                           │
│                                                                     │
│  7. APPLY COOLDOWN                                                  │
│     if steps_since_last_trade < cooldown:                           │
│         action = HOLD                                               │
│                                                                     │
│  8. APPLY SHORT FEES                                                │
│     if position == SHORT:                                           │
│         Deduct daily borrowing fee                                  │
│                                                                     │
│  9. EXECUTE ACTION                                                  │
│     switch(action):                                                 │
│         OPEN_LONG  → _execute_open_long()                           │
│         CLOSE_LONG → _execute_close_long()                          │
│         OPEN_SHORT → _execute_open_short()                          │
│         CLOSE_SHORT→ _execute_close_short()                         │
│         HOLD       → do nothing                                     │
│                                                                     │
│  10. UPDATE PORTFOLIO                                               │
│      _update_portfolio_value(current_price)                         │
│      Update peak_nav, leverage, drawdown                            │
│                                                                     │
│  11. CALCULATE REWARD                                               │
│      reward = _calculate_reward(previous_net_worth)                 │
│                                                                     │
│  12. BUILD OBSERVATION                                              │
│      observation = _get_obs()                                       │
│                                                                     │
│  13. BUILD INFO DICT                                                │
│      info = _get_info()                                             │
│                                                                     │
│  14. RETURN                                                         │
│      return (observation, reward, done, truncated, info)            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3.6 Long Position Lifecycle

**Opening a Long Position:**

```python
def _execute_open_long(self, current_price: float, current_atr: float) -> bool:
    """
    Open a long position (buy to profit when price goes UP).

    Steps:
    1. Set SL/TP via risk manager
    2. Calculate position size (triple constraint)
    3. Validate size (min quantity, sufficient balance)
    4. Execute BUY trade
    5. Update position state
    """
```

**Detailed Flow:**

```
OPEN LONG at $2000/oz
─────────────────────

1. SET STOP LOSS / TAKE PROFIT
   ATR = $15
   SL Multiplier = 2.0
   TP Percentage = 2%

   Stop Loss = $2000 - (2.0 × $15) = $1970
   Take Profit = $2000 × 1.02 = $2040
   SL Distance = $30


2. CALCULATE POSITION SIZE (Triple Constraint)

   Constraint A: Risk-Based
     Max Risk = $1000 × 1% = $10
     Size_Risk = $10 / $30 = 0.333 oz

   Constraint B: Kelly Criterion
     Kelly Fraction = 5% (after limits)
     Size_Kelly = ($1000 × 5%) / $30 = 1.67 oz

   Constraint C: Leverage Limit
     Max Position Value = $1000 × 1.0 = $1000
     Size_Leverage = $1000 / $2000 = 0.5 oz

   Final Size = min(0.333, 1.67, 0.5) = 0.333 oz


3. EXECUTE BUY
   Gross Value = 0.333 oz × $2000 = $666
   Spread Cost = $666 × 0.05% = $0.33
   Slippage = $666 × 0.01% = $0.07
   Commission = max($666 × 0.05%, $1000 × 0.01%) = $0.33

   Total Cost = $666 + $0.33 + $0.07 + $0.33 = $666.73


4. UPDATE STATE
   balance: $1000 → $333.27
   stock_quantity: 0 → 0.333
   entry_price: NaN → $2000
   position_type: FLAT → LONG
   net_worth: $1000 → $999.27 (slightly down due to costs)
```

**Closing a Long Position:**

```python
def _execute_close_long(self, current_price: float) -> bool:
    """
    Close a long position (sell to realize P&L).

    Steps:
    1. Validate we have a long position
    2. Execute SELL trade
    3. Calculate P&L
    4. Update win/loss statistics
    5. Reset position state
    """
```

**Example Close:**

```
CLOSE LONG at $2040/oz (TP Hit)
───────────────────────────────

Position: 0.333 oz @ $2000 entry

1. EXECUTE SELL
   Gross Revenue = 0.333 oz × $2040 = $679.32
   Spread Cost = $679.32 × 0.05% = $0.34
   Slippage = $679.32 × 0.01% = $0.07
   Commission = $0.34

   Net Revenue = $679.32 - $0.34 - $0.07 - $0.34 = $678.57


2. CALCULATE P&L
   Cost Basis = 0.333 oz × $2000 = $666
   P&L Absolute = $678.57 - $666 = $12.57
   P&L Percentage = $12.57 / $666 = +1.89%


3. UPDATE STATISTICS
   total_trades: 0 → 1
   winning_trades: 0 → 1 (because P&L > 0)
   trade_history: [{'pnl': $12.57, 'pnl_pct': 1.89%}]


4. RESET POSITION
   balance: $333.27 → $1011.84
   stock_quantity: 0.333 → 0
   entry_price: $2000 → NaN
   position_type: LONG → FLAT
   risk_manager.reset()  # Clear SL/TP
```

---

## 3.7 Short Position Lifecycle

**Opening a Short Position:**

```python
def _execute_open_short(self, current_price: float, current_atr: float) -> bool:
    """
    Open a short position (sell borrowed asset to profit when price goes DOWN).

    Short selling mechanics:
    1. Borrow asset from broker (conceptually)
    2. Sell at current price (receive cash)
    3. Later: buy back to return to broker
    4. Profit if price goes DOWN, loss if price goes UP
    """
```

**Detailed Flow:**

```
OPEN SHORT at $2000/oz
──────────────────────

1. SET STOP LOSS / TAKE PROFIT (Reversed for shorts)
   ATR = $15

   Stop Loss = $2000 + (2.0 × $15) = $2030  (ABOVE entry)
   Take Profit = $2000 × 0.98 = $1960      (BELOW entry)
   SL Distance = $30


2. CALCULATE POSITION SIZE (Same as long)
   Final Size = 0.333 oz


3. EXECUTE SHORT SELL
   We "borrow" 0.333 oz and sell immediately

   Gross Revenue = 0.333 oz × $2000 = $666
   Spread Cost = $666 × 0.05% = $0.33
   Commission = $0.33

   Net Revenue = $666 - $0.33 - $0.33 = $665.34


4. UPDATE STATE
   balance: $1000 → $1665.34  (INCREASED - we received cash)
   stock_quantity: 0 → -0.333  (NEGATIVE = short position)
   entry_price: NaN → $2000
   position_type: FLAT → SHORT


5. NET WORTH CALCULATION FOR SHORTS
   We owe 0.333 oz which we must buy back later

   If price is still $2000:
     Cost to close = 0.333 × $2000 = $666
     Unrealized P&L = $0
     Net Worth = $1665.34 - $666 = $999.34

   If price drops to $1950:
     Cost to close = 0.333 × $1950 = $649.35
     Unrealized P&L = $666 - $649.35 = $16.65 (PROFIT!)
     Net Worth = $1665.34 - $649.35 = $1015.99

   If price rises to $2050:
     Cost to close = 0.333 × $2050 = $682.65
     Unrealized P&L = $666 - $682.65 = -$16.65 (LOSS)
     Net Worth = $1665.34 - $682.65 = $982.69
```

**Closing a Short Position:**

```python
def _execute_close_short(self, current_price: float) -> bool:
    """
    Close a short position (buy to cover).

    Steps:
    1. Calculate cost to buy back
    2. Deduct from balance
    3. Calculate P&L = (entry - exit) × quantity
    4. Update statistics
    5. Reset position
    """
```

**Example Close (Profitable Short):**

```
CLOSE SHORT at $1960/oz (TP Hit)
────────────────────────────────

Position: -0.333 oz @ $2000 entry

1. EXECUTE BUY TO COVER
   Gross Cost = 0.333 oz × $1960 = $652.68
   Spread = $652.68 × 0.05% = $0.33
   Slippage = $652.68 × 0.01% = $0.07
   Commission = $0.33

   Total Cost = $652.68 + $0.33 + $0.07 + $0.33 = $653.41


2. CALCULATE P&L
   Entry Value = 0.333 × $2000 = $666
   Exit Cost = $653.41

   P&L Absolute = $666 - $653.41 = $12.59 (PROFIT - price went DOWN)
   P&L Percentage = $12.59 / $666 = +1.89%


3. UPDATE STATE
   balance: $1665.34 → $1011.93
   stock_quantity: -0.333 → 0
   entry_price: $2000 → NaN
   position_type: SHORT → FLAT
```

---

## 3.8 Portfolio Valuation

```python
def _update_portfolio_value(self, current_price: float) -> None:
    """
    Update net worth accounting for both long and short positions.

    For LONG positions:
        net_worth = balance + (quantity × price)
        (We own the asset, value increases with price)

    For SHORT positions:
        unrealized_pnl = (entry_price - current_price) × |quantity|
        net_worth = balance + unrealized_pnl
        (We owe the asset, value increases when price decreases)

    For FLAT:
        net_worth = balance
    """
```

**Visual Example:**

```
LONG POSITION VALUATION
───────────────────────
Balance: $500
Quantity: 0.5 oz
Current Price: $2000

Net Worth = $500 + (0.5 × $2000) = $1500


SHORT POSITION VALUATION
────────────────────────
Balance: $1500 (received from short sale)
Quantity: -0.5 oz (owe broker)
Entry Price: $2000
Current Price: $1900

Unrealized P&L = ($2000 - $1900) × 0.5 = $50 profit
Net Worth = $1500 + $50 = $1550

OR equivalently:
We have: $1500 cash
We owe: 0.5 oz worth $950 (at current price)
Net Worth = $1500 - $950 + $1000 (original equity)
          = This formula is simplified as: balance + unrealized_pnl
```

---

## 3.9 Reward Function Deep Dive

The reward function is the most critical component for RL training. It defines
what the agent optimizes for.

```python
def _calculate_reward(self, previous_net_worth: float) -> float:
    """
    PRODUCTION-GRADE REWARD FUNCTION FOR PPO TRADING BOT

    8-Step Process:
    1. Validation (prevent NaN/Inf)
    2. Core profitability metric
    3. Risk-adjusted penalties
    4. Composite raw reward
    5. Shaping bonuses
    6. Normalization (tanh squashing)
    7. Special cases (terminal conditions)
    8. Final safety clipping

    Expected range: [-20, +20]
    Typical values: [-5, +5]
    """
```

**Step-by-Step Breakdown:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REWARD CALCULATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: VALIDATION                                                         │
│  ─────────────────                                                          │
│  if previous_net_worth <= 0 or net_worth <= 0:                              │
│      return -20.0  # Critical failure                                       │
│                                                                             │
│  if isnan(net_worth) or isinf(net_worth):                                   │
│      return -20.0  # Data corruption                                        │
│                                                                             │
│                                                                             │
│  STEP 2: CORE PROFITABILITY (Primary Driver)                                │
│  ───────────────────────────────────────────                                │
│  log_return = log(net_worth / previous_net_worth)                           │
│  profitability_reward = log_return × 100                                    │
│                                                                             │
│  Example: 1% gain → log(1.01) × 100 ≈ 1.0                                  │
│  Example: 2% loss → log(0.98) × 100 ≈ -2.0                                 │
│                                                                             │
│  Why log returns?                                                           │
│    - Symmetric: +10% and -10% have equal magnitude                          │
│    - Additive: Multiple period returns sum correctly                        │
│    - Stable: No explosion for large moves                                   │
│                                                                             │
│                                                                             │
│  STEP 3: RISK PENALTIES                                                     │
│  ──────────────────────                                                     │
│                                                                             │
│  A) DRAWDOWN PENALTY (Only for NEW drawdown)                                │
│     current_dd = peak_nav - net_worth                                       │
│     dd_increase = max(0, current_dd - previous_dd)                          │
│     dd_penalty = (dd_increase / initial_balance) × 5.0                      │
│                                                                             │
│     Key insight: Only penalize WORSENING drawdown                           │
│     Don't punish agent for existing market conditions                       │
│                                                                             │
│  B) FRICTION PENALTY (Transaction costs)                                    │
│     friction_penalty = (commission / initial_balance) × 2.0                 │
│                                                                             │
│     Encourages agent to be selective about trades                           │
│                                                                             │
│  C) LEVERAGE PENALTY (Quadratic)                                            │
│     leverage_excess = max(0, current_leverage - max_leverage)               │
│     leverage_penalty = (leverage_excess²) × 10.0                            │
│                                                                             │
│     Quadratic: Small violations = small penalty                             │
│                Large violations = HUGE penalty                              │
│                                                                             │
│  D) DURATION PENALTY (Holding too long)                                     │
│     if hold_duration > max_duration:                                        │
│         excess = hold_duration - max_duration                               │
│         duration_penalty = (excess / max_duration) × 0.5                    │
│                                                                             │
│  E) INVALID ACTION PENALTY (Sprint 2 Fix - "Fearful Agent")                 │
│     if invalid_action_this_step:                                            │
│         invalid_penalty = 0.5                                               │
│                                                                             │
│  F) HOLD PENALTY (Sprint 2 Fix - "Fearful Agent")                           │
│     if position == FLAT and action == HOLD:                                 │
│         hold_penalty = 0.01  # Small pressure to act                        │
│                                                                             │
│  total_penalty = sum of all penalties                                       │
│                                                                             │
│                                                                             │
│  STEP 4: RAW REWARD                                                         │
│  ─────────────────                                                          │
│  raw_reward = profitability_reward - total_penalty                          │
│                                                                             │
│                                                                             │
│  STEP 5: TRADE BONUSES                                                      │
│  ──────────────────────                                                     │
│  if trade_just_closed and trade_success:                                    │
│      if pnl > 0:  # Winning trade                                           │
│          win_bonus = min(2.0, (pnl_pct / 100) × 10)                         │
│          if pnl_pct > 1.5%:                                                 │
│              bonus += 1.0  # Extra for quality wins                         │
│      else:  # Losing trade                                                  │
│          loss_feedback = max(-1.0, (pnl_pct / 100) × 5)                     │
│                                                                             │
│                                                                             │
│  STEP 6: NORMALIZATION (Tanh Squashing)                                     │
│  ──────────────────────────────────────                                     │
│  combined = raw_reward + bonus                                              │
│  normalized = tanh(combined × 0.3)   # Squash to [-1, +1]                   │
│  scaled = normalized × 5.0           # Map to [-5, +5]                      │
│                                                                             │
│  Why tanh?                                                                  │
│    - Smooth gradients everywhere (good for backprop)                        │
│    - Bounded output (prevents exploding values)                             │
│    - Preserves sign (positive stays positive)                               │
│                                                                             │
│                                                                             │
│  STEP 7: SPECIAL CASES                                                      │
│  ──────────────────────                                                     │
│  if net_worth <= minimum_allowed_balance:                                   │
│      return -20.0  # Account blown - severe punishment                      │
│                                                                             │
│  if drawdown_ratio > 15%:                                                   │
│      scaled -= 5.0  # Emergency risk management                             │
│                                                                             │
│                                                                             │
│  STEP 8: FINAL CLIPPING                                                     │
│  ───────────────────────                                                    │
│  final_reward = clip(scaled, -20.0, +20.0)                                  │
│                                                                             │
│  return final_reward                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Reward Component Examples:**

```
EXAMPLE 1: Small Profitable Trade
─────────────────────────────────
Previous NW: $1000
Current NW: $1008 (+0.8%)

log_return = log(1008/1000) = 0.00797
profitability = 0.00797 × 100 = 0.797

No penalties (holding well-managed position)
Bonus: Winning trade (+1.5% pnl) = min(2.0, 1.5/100 × 10) = 0.15

raw_reward = 0.797 - 0 + 0.15 = 0.947
normalized = tanh(0.947 × 0.3) = 0.277
scaled = 0.277 × 5.0 = 1.38

Final reward: +1.38


EXAMPLE 2: Loss with Drawdown
─────────────────────────────
Previous NW: $1000
Current NW: $980 (-2%)
New Drawdown: $20 (from $1000 peak)

log_return = log(980/1000) = -0.0202
profitability = -2.02

dd_penalty = ($20 / $1000) × 5.0 = 0.1

raw_reward = -2.02 - 0.1 = -2.12
normalized = tanh(-2.12 × 0.3) = -0.562
scaled = -0.562 × 5.0 = -2.81

Final reward: -2.81


EXAMPLE 3: Account Blown
────────────────────────
Net worth falls to $50 (below $100 minimum)

Immediate return: -20.0 (maximum punishment)
Episode terminates
```

---

## 3.10 Episode Management

```python
def reset(self, seed: int = None, options: dict = None):
    """
    Reset environment to initial state for new episode.

    Returns:
        observation: Initial observation (783-dim)
        info: Initial info dict

    Fixed Episode Length Mode (Recommended):
        - All episodes have exactly 500 steps
        - Random start within valid data range
        - Consistent gradient estimates for PPO

    Variable Episode Length Mode (Original):
        - Random start and variable length
        - Can cause training instability
    """
```

**Episode Boundary Calculation:**

```
Data: 50,000 bars (2019-2024)
Lookback: 30 bars
Fixed Episode Length: 500 bars

Valid Start Range:
  min_start = lookback - 1 = 29
  max_start = data_length - 1 - episode_length = 50000 - 1 - 500 = 49499

Random selection: start_idx ∈ [29, 49499]

Episode boundaries:
  start_idx = 15000 (randomly selected)
  end_idx = 15000 + 500 = 15500
  current_step begins at 15000

Each reset() picks a new random window → diverse training data
```

**Reset Process:**

```
1. RESET FINANCIAL STATE
   balance = initial_balance ($1000)
   net_worth = initial_balance
   stock_quantity = 0
   entry_price = NaN
   position_type = FLAT

2. RESET RISK MANAGER
   Clear SL/TP/TSL
   Reset drawdown tracking
   Reset client profile

3. RESET STATISTICS
   total_trades = 0
   winning_trades = 0
   losing_trades = 0
   total_fees_paid = 0

4. RESET TRACKERS
   peak_nav = initial_balance
   previous_drawdown = 0
   current_leverage = 0
   hold_duration = 0
   invalid_action_count = 0

5. SELECT EPISODE BOUNDARIES
   Random start within valid range
   Set max_steps (fixed or variable)

6. BUILD INITIAL OBSERVATION
   observation = _get_obs()

7. RETURN
   return (observation, info)
```

---

# ==============================================================================
# PART 4: RISK MANAGEMENT SYSTEM (risk_manager.py)
# ==============================================================================

## 4.1 Dynamic Risk Manager Overview

The `DynamicRiskManager` class provides institutional-grade risk management.

```python
class DynamicRiskManager:
    """
    Production-grade dynamic risk management system.

    Features:
    - GARCH(1,1) volatility forecasting
    - Kelly Criterion position sizing
    - Multi-client profile support
    - Adaptive SL/TP/TSL
    - Market regime awareness
    - Drawdown circuit breakers
    """
```

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DynamicRiskManager                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Client Profile │  │  GARCH Model    │  │  Market State   │             │
│  │  Management     │  │  (Volatility)   │  │  (Regime)       │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           ▼                    ▼                    ▼                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Position Sizing Engine                          │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                        │   │
│  │  │ Risk-Based│  │   Kelly   │  │ Leverage  │                        │   │
│  │  │   Size    │  │   Size    │  │   Limit   │                        │   │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                        │   │
│  │        └──────────────┼──────────────┘                               │   │
│  │                       ▼                                              │   │
│  │               min(risk, kelly, leverage)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   Stop Loss     │  │   Take Profit   │  │  Trailing Stop  │             │
│  │   Calculator    │  │   Calculator    │  │   Manager       │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.2 Client Profile System

Each trading client/account has its own risk profile:

```python
def set_client_profile(
    self,
    client_id: str,
    initial_equity: float,
    max_drawdown_pct: float = 20.0,
    kelly_fraction_limit: float = 0.1,
    max_trade_risk_pct: float = 0.01
) -> None:
    """
    Set risk limits for a specific client.

    Args:
        client_id: Unique identifier for client
        initial_equity: Starting capital
        max_drawdown_pct: Maximum allowed drawdown (e.g., 20%)
        kelly_fraction_limit: Cap on Kelly fraction (e.g., 10%)
        max_trade_risk_pct: Risk per trade (e.g., 1%)
    """

    self.client_profiles[client_id] = {
        'initial_equity': initial_equity,
        'current_equity': initial_equity,
        'max_drawdown_pct': max_drawdown_pct,
        'kelly_fraction_limit': kelly_fraction_limit,
        'max_trade_risk_pct': max_trade_risk_pct,
        'peak_equity': initial_equity,
        'is_halted': False
    }
```

**Profile Usage Example:**

```
Client: "aggressive_trader"
  initial_equity: $10,000
  max_drawdown_pct: 25%
  kelly_fraction_limit: 15%
  max_trade_risk_pct: 2%

Client: "conservative_trader"
  initial_equity: $10,000
  max_drawdown_pct: 10%
  kelly_fraction_limit: 5%
  max_trade_risk_pct: 0.5%

Same model, different risk profiles!
```

---

## 4.3 GARCH Volatility Model

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
time-varying volatility - crucial for financial markets.

```python
def calculate_garch_volatility(
    self,
    returns: np.ndarray,
    force_update: bool = False
) -> float:
    """
    Calculate volatility using GARCH(1,1) model.

    GARCH(1,1) equation:
        σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)

    Where:
        σ²(t) = Variance at time t
        ω = Long-term variance weight
        α = Reaction to recent shock
        β = Persistence of volatility
        ε(t-1) = Previous period's return shock

    Returns:
        Annualized volatility forecast
    """
```

**GARCH Intuition:**

```
GARCH captures "volatility clustering" - periods of high volatility
tend to be followed by high volatility, and vice versa.

Example: After a market crash, volatility stays elevated for weeks.
GARCH models this persistence.

         Volatility
              │
        High  │    ████                    ███
              │   █████                   █████
              │  ██████                  ███████
              │ ████████               █████████
        Low   │███████████████████████████████████
              └─────────────────────────────────────► Time
                    Crash            Another shock

Without GARCH: Assumes constant volatility (wrong!)
With GARCH: Adapts to current market conditions
```

**EWMA Fallback:**

When GARCH fitting fails (insufficient data, convergence issues), we use
Exponentially Weighted Moving Average as a simpler fallback:

```python
# EWMA with λ = 0.94 (RiskMetrics standard)
ewma_variance = returns.ewm(span=30).var().iloc[-1]
ewma_volatility = np.sqrt(ewma_variance) * np.sqrt(252)  # Annualize
```

---

## 4.4 Kelly Criterion Position Sizing

The Kelly Criterion determines the mathematically optimal bet size.

```python
def _calculate_kelly_fraction(
    self,
    win_prob: float,
    risk_reward_ratio: float
) -> float:
    """
    Calculate optimal Kelly fraction.

    Kelly Formula:
        f* = (B × P - Q) / B

    Where:
        f* = Fraction of capital to bet
        B = Odds received (risk:reward ratio)
        P = Probability of winning
        Q = Probability of losing (1 - P)

    Example:
        Win probability: 55%
        Risk:Reward: 2:1 (make $2 for every $1 risked)

        f* = (2 × 0.55 - 0.45) / 2
           = (1.10 - 0.45) / 2
           = 0.65 / 2
           = 0.325 = 32.5% of capital

    In practice, we use "fractional Kelly" (e.g., 25% of full Kelly)
    to reduce volatility and account for estimation errors.
    """
```

**Kelly Calculation Example:**

```
SCENARIO: Trading Gold (XAU/USD)

Historical analysis shows:
  Win rate: 52%
  Average win: $150
  Average loss: $100
  Risk:Reward = 150/100 = 1.5

Full Kelly:
  f* = (1.5 × 0.52 - 0.48) / 1.5
     = (0.78 - 0.48) / 1.5
     = 0.30 / 1.5
     = 0.20 = 20%

Fractional Kelly (25%):
  f_practical = 0.20 × 0.25 = 5%

With Kelly limit of 10%:
  final_kelly = min(5%, 10%) = 5%

If equity = $10,000:
  Kelly-based risk = $10,000 × 5% = $500
```

**Why Fractional Kelly?**

```
Full Kelly optimizes long-term growth rate but has:
- High volatility (50%+ drawdowns common)
- Sensitivity to parameter estimation errors
- Psychological difficulty (large swings)

Fractional Kelly (25-50% of full):
- Reduces volatility significantly
- More robust to estimation errors
- Psychologically manageable
- Still captures most of the edge
```

---

## 4.5 Stop Loss / Take Profit Logic

```python
def set_trade_orders(
    self,
    entry_price: float,
    atr: float,
    is_long: bool = True
) -> float:
    """
    Set stop loss and take profit levels at trade entry.

    For LONG positions:
        SL = entry - (ATR × sl_multiplier)
        TP = entry × (1 + tp_percentage)

    For SHORT positions:
        SL = entry + (ATR × sl_multiplier)
        TP = entry × (1 - tp_percentage)

    Returns:
        SL distance in absolute terms (for position sizing)
    """
```

**SL/TP Calculation Example:**

```
LONG TRADE ENTRY
────────────────
Entry: $2000
ATR: $15
SL Multiplier: 2.0
TP Percentage: 2%

Stop Loss = $2000 - (2.0 × $15) = $1970
Take Profit = $2000 × 1.02 = $2040
SL Distance = $30 (used for position sizing)

Risk:Reward = $40 / $30 = 1.33:1


SHORT TRADE ENTRY
─────────────────
Entry: $2000
ATR: $15
SL Multiplier: 2.0
TP Percentage: 2%

Stop Loss = $2000 + (2.0 × $15) = $2030  (ABOVE entry)
Take Profit = $2000 × 0.98 = $1960       (BELOW entry)
SL Distance = $30

Risk:Reward = $40 / $30 = 1.33:1
```

---

## 4.6 Trailing Stop Loss

```python
def update_trailing_stop(
    self,
    entry_price: float,
    current_price: float,
    atr: float,
    is_long: bool = True
) -> None:
    """
    Update trailing stop loss to lock in profits.

    Activation Condition:
        Profit > (SL distance × TSL_START_MULTIPLIER)

    Trail Distance:
        ATR × TSL_TRAIL_DISTANCE_MULTIPLIER

    TSL moves UP for longs (never down)
    TSL moves DOWN for shorts (never up)
    """
```

**Trailing Stop Example:**

```
LONG POSITION
─────────────
Entry: $2000
Original SL: $1970
TSL Start Multiplier: 1.0 (activate after 1x risk profit)
TSL Trail Distance: 0.5 × ATR = $7.50

Step 1: Price = $2010
  Profit = $10 (0.5%)
  Required for TSL: $30 (1x risk)
  TSL not activated yet

Step 2: Price = $2040
  Profit = $40 (2%)
  Required for TSL: $30 ✓
  TSL ACTIVATED!
  New TSL = $2040 - $7.50 = $2032.50

Step 3: Price = $2060
  TSL trails up
  New TSL = $2060 - $7.50 = $2052.50
  (We've locked in $52.50 profit vs original SL at $1970)

Step 4: Price = $2050
  Price dropped but still above TSL
  TSL stays at $2052.50 (never moves down)

Step 5: Price = $2051
  Still above TSL, position remains open

Step 6: Price = $2048
  BELOW TSL ($2052.50)
  POSITION CLOSED AT MARKET
  Realized profit ≈ $48 (2.4%)
```

---

## 4.7 Position Size Calculation

The **Triple Constraint System** ensures position size never exceeds any limit.

```python
def calculate_adaptive_position_size(
    self,
    client_id: str,
    account_equity: float,
    atr_stop_distance: float,
    win_prob: float = 0.5,
    risk_reward_ratio: float = 1.0,
    current_price: float = None,
    max_leverage: float = 1.0,
    is_long: bool = True
) -> float:
    """
    Calculate position size with TRIPLE CONSTRAINT.

    Constraint 1: Risk-Neutral (Fixed Risk)
        max_risk = equity × risk_percentage
        size = max_risk / sl_distance

    Constraint 2: Kelly Criterion
        kelly = calculate_kelly_fraction(win_prob, rr)
        kelly_limited = min(kelly, kelly_limit) × regime_scale
        size = (equity × kelly_limited) / sl_distance

    Constraint 3: Leverage Limit
        max_position_value = max_leverage × equity
        size = max_position_value / price

    Final Size = min(Constraint1, Constraint2, Constraint3)
    """
```

**Complete Position Sizing Example:**

```
INPUT
─────
Account Equity: $10,000
Current Price: $2000/oz
ATR: $15
SL Distance: 2 × $15 = $30
Win Probability: 55%
Risk:Reward Ratio: 2.0
Max Risk per Trade: 1%
Kelly Limit: 10%
Max Leverage: 1.0
Market Regime: Calm (scale = 1.0)


CONSTRAINT 1: RISK-BASED
────────────────────────
Max Risk = $10,000 × 1% = $100
Size_Risk = $100 / $30 = 3.33 oz


CONSTRAINT 2: KELLY CRITERION
─────────────────────────────
Full Kelly = (2.0 × 0.55 - 0.45) / 2.0 = 0.325 (32.5%)
Capped Kelly = min(32.5%, 10%) = 10%
Regime-Adjusted = 10% × 1.0 = 10%

Kelly Risk = $10,000 × 10% = $1,000
Size_Kelly = $1,000 / $30 = 33.33 oz


CONSTRAINT 3: LEVERAGE LIMIT
────────────────────────────
Max Position Value = 1.0 × $10,000 = $10,000
Size_Leverage = $10,000 / $2,000 = 5.0 oz


FINAL CALCULATION
─────────────────
Final Size = min(3.33, 33.33, 5.0) = 3.33 oz

Position Value = 3.33 oz × $2000 = $6,666
Actual Leverage = $6,666 / $10,000 = 0.67x
Actual Risk = 3.33 oz × $30 = $100 (1% as intended)
```

---

## 4.8 Market Regime Detection

```python
def _get_regime_scaling(self, regime: int) -> float:
    """
    Get position size scaling based on market regime.

    Regime 0 (Calm): Scale = 1.0 (full position size)
    Regime 1 (Volatile/Chaos): Scale = 0.5 (half position size)

    This reduces exposure during uncertain markets.
    """
    return 1.0 if regime == 0 else 0.5


def _get_regime_multiplier(self, regime: int) -> float:
    """
    Get SL distance multiplier based on market regime.

    Regime 0 (Calm): 2.0× ATR (tighter stops)
    Regime 1 (Volatile): 3.0× ATR (wider stops)

    Wider stops in volatile markets prevent premature stop-outs.
    """
    return 2.0 if regime == 0 else 3.0
```

**Regime Adaptation:**

```
CALM MARKET (BOS signal detected = trending)
────────────────────────────────────────────
Position Scale: 100%
SL Multiplier: 2.0× ATR
Rationale: Clear trend, confident positioning


VOLATILE MARKET (No BOS = choppy/unclear)
─────────────────────────────────────────
Position Scale: 50%
SL Multiplier: 3.0× ATR
Rationale: Uncertain conditions, conservative sizing


Example Impact:
  Normal conditions: 5 oz position, $30 SL
  Volatile conditions: 2.5 oz position, $45 SL

  Risk in calm: 5 × $30 = $150
  Risk in volatile: 2.5 × $45 = $112.50 (25% less risk)
```

---

# ==============================================================================
# PART 5: SMART MONEY CONCEPTS ENGINE (strategy_features.py)
# ==============================================================================

## 5.1 Technical Indicators

```python
class SmartMoneyEngine:
    """
    Feature engineering combining traditional TA with Smart Money Concepts.

    Technical Indicators:
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        - ATR (Average True Range)

    Smart Money Concepts:
        - Fractals (Swing Points)
        - Fair Value Gaps (FVG)
        - Break of Structure (BOS)
        - Change of Character (CHOCH)
        - Order Blocks
    """
```

**RSI (Relative Strength Index):**

```
RSI measures momentum on a 0-100 scale.

Formula:
    RS = Average Gain / Average Loss (over N periods)
    RSI = 100 - (100 / (1 + RS))

Interpretation:
    RSI > 70: Overbought (potential sell signal)
    RSI < 30: Oversold (potential buy signal)
    RSI = 50: Neutral momentum

Config: RSI_WINDOW = 7 (fast RSI for 15-min timeframe)
```

**MACD (Moving Average Convergence Divergence):**

```
MACD measures trend and momentum.

Components:
    MACD Line = EMA(8) - EMA(17)
    Signal Line = EMA(9) of MACD Line
    Histogram = MACD Line - Signal Line

Interpretation:
    MACD > Signal: Bullish momentum
    MACD < Signal: Bearish momentum
    Histogram expanding: Trend strengthening
    Histogram contracting: Trend weakening
```

**Bollinger Bands:**

```
Bollinger Bands measure volatility.

Components:
    Middle Band = SMA(20)
    Upper Band = Middle + (2 × StdDev)
    Lower Band = Middle - (2 × StdDev)

Interpretation:
    Price near Upper: Potentially overbought
    Price near Lower: Potentially oversold
    Bands widening: Volatility increasing
    Bands narrowing: Volatility decreasing (squeeze)
```

**ATR (Average True Range):**

```
ATR measures volatility in price movement.

True Range = max(
    High - Low,
    |High - Previous Close|,
    |Low - Previous Close|
)

ATR = Moving Average of True Range (typically 7 or 14 periods)

Uses:
    - Stop loss placement (2× ATR from entry)
    - Position sizing (risk / ATR)
    - Volatility filtering
```

---

## 5.2 Fractal Detection (Swing Points)

```python
def _add_smc_base_features(self):
    """
    Detect swing highs (UP_FRACTAL) and swing lows (DOWN_FRACTAL).

    A fractal is a local extremum:
        UP_FRACTAL: Highest high in a 5-bar window
        DOWN_FRACTAL: Lowest low in a 5-bar window

    Causality: Only confirmed after 2 bars (no lookahead bias)
    """
```

**Fractal Detection Logic:**

```
UP_FRACTAL (Swing High)
───────────────────────
        ▲
       /█\
      / █ \
     /  █  \
    █   █   █
   bar bar bar
   -2  -1   0   (confirmed at bar 0)

Condition:
    High[-1] > High[-2] AND
    High[-1] > High[0]

After 2 bars, we KNOW bar[-1] was a swing high


DOWN_FRACTAL (Swing Low)
────────────────────────
    █   █   █
     \  █  /
      \ █ /
       \█/
        ▼

Condition:
    Low[-1] < Low[-2] AND
    Low[-1] < Low[0]


EXAMPLE:
    Bar  | High  | Low   | UP_FRACTAL | DOWN_FRACTAL
    ─────┼───────┼───────┼────────────┼─────────────
    1    | 2000  | 1990  |     0      |      0
    2    | 2010  | 1995  |     0      |      0
    3    | 2005  | 1992  |     1      |      0  ← Bar 2 was swing high
    4    | 2008  | 1988  |     0      |      0
    5    | 2003  | 1985  |     0      |      0
    6    | 2006  | 1990  |     0      |      1  ← Bar 5 was swing low
```

---

## 5.3 Fair Value Gaps (FVG)

```python
def _detect_fvg(self):
    """
    Detect Fair Value Gaps - price imbalances that often get "filled".

    Bullish FVG:
        Current Low > Previous Close (gap up)
        Creates demand zone below

    Bearish FVG:
        Current High < Previous Close (gap down)
        Creates supply zone above

    FVG_SIGNAL: +1 (bullish), -1 (bearish), 0 (none)
    FVG_SIZE_NORM: Normalized gap size
    """
```

**FVG Visualization:**

```
BULLISH FVG (Gap Up)
────────────────────

        ┌────┐
        │    │  Current bar
        │    │  Low = 2010
        └────┘
           ↑
       GAP │ (FVG zone)
           ↓
   ─ ─ ─ ─ ─ ─  Previous Close = 2005
        ┌────┐
        │    │  Previous bar
        └────┘

FVG_SIGNAL = +1
FVG_SIZE = 2010 - 2005 = 5
FVG_SIZE_NORM = 5 / ATR

Interpretation: Price "jumped" up, leaving unfilled orders.
               Market may return to fill this gap.


BEARISH FVG (Gap Down)
──────────────────────

        ┌────┐
        │    │  Previous bar
        └────┘
   ─ ─ ─ ─ ─ ─  Previous Close = 2000
           ↑
       GAP │ (FVG zone)
           ↓
        ┌────┐
        │    │  Current bar
        │    │  High = 1995
        └────┘

FVG_SIGNAL = -1
FVG_SIZE = 2000 - 1995 = 5

Interpretation: Price "dropped" down, leaving unfilled orders.
               Market may rally back to fill this gap.
```

---

## 5.4 Break of Structure (BOS)

```python
def _calculate_structure_iterative(self):
    """
    Detect Break of Structure - trend continuation signals.

    Bullish BOS:
        Price breaks above previous swing high
        Indicates uptrend continuation

    Bearish BOS:
        Price breaks below previous swing low
        Indicates downtrend continuation

    BOS_SIGNAL: +1 (bullish), -1 (bearish), 0 (none)
    """
```

**BOS Visualization:**

```
BULLISH BOS (Break Above Resistance)
────────────────────────────────────

              ▲ NEW HIGH
             /│
            / │
    ───────●──┼─────── Previous Swing High (resistance)
          /   │
         /    │ BOS!
        ●     │
       /      │
      /       │
     ●        │

When price CLOSES above previous swing high → BOS_SIGNAL = +1
This suggests: "Buyers are in control, trend continues up"


BEARISH BOS (Break Below Support)
─────────────────────────────────

     ●
      \
       \       │
        ●      │
         \     │ BOS!
          \    │
    ───────●──┼─────── Previous Swing Low (support)
            \ │
             \│
              ▼ NEW LOW

When price CLOSES below previous swing low → BOS_SIGNAL = -1
This suggests: "Sellers are in control, trend continues down"
```

---

## 5.5 Change of Character (CHOCH)

```python
def _detect_choch(self):
    """
    Detect Change of Character - potential trend reversal signals.

    Bullish CHOCH:
        After a series of lower lows, price makes a higher high
        Suggests shift from downtrend to uptrend

    Bearish CHOCH:
        After a series of higher highs, price makes a lower low
        Suggests shift from uptrend to downtrend

    CHOCH_SIGNAL: +1 (bullish reversal), -1 (bearish reversal), 0 (none)
    """
```

**CHOCH vs BOS:**

```
BOS = Trend CONTINUATION (break in direction of trend)
CHOCH = Trend REVERSAL (break against direction of trend)


BULLISH CHOCH (Reversal from Down to Up)
────────────────────────────────────────

    ●        (Previous swing high)
     \
      \
       ●     (Lower high - downtrend)
        \
         \
          ●  (Lower low - downtrend continues)
           \
            ●  (Even lower low)
             \
              ●───────●  CHOCH! (Higher high breaks structure)
                     /
                    /

When in downtrend (lower lows, lower highs)
AND price breaks ABOVE previous swing high
→ CHOCH_SIGNAL = +1 (bullish reversal)


BEARISH CHOCH (Reversal from Up to Down)
────────────────────────────────────────

                    \
              ●──────● CHOCH! (Lower low breaks structure)
             /
            /
           ●  (Higher high - uptrend)
          /
         /
        ●  (Higher low - uptrend continues)
       /
      ●
     /
    ●        (Previous swing low)

When in uptrend (higher highs, higher lows)
AND price breaks BELOW previous swing low
→ CHOCH_SIGNAL = -1 (bearish reversal)
```

---

## 5.6 Order Blocks

```python
def _add_smc_order_blocks(self):
    """
    Detect Order Blocks - zones where institutional orders were placed.

    Bullish Order Block:
        Red candle (close < open) followed by
        Green candle (close > open) that makes a higher high
        The red candle's range becomes a demand zone

    Bearish Order Block:
        Green candle (close > open) followed by
        Red candle (close < open) that makes a lower low
        The green candle's range becomes a supply zone

    Features:
        BULLISH_OB_HIGH, BULLISH_OB_LOW: Zone boundaries
        BEARISH_OB_HIGH, BEARISH_OB_LOW: Zone boundaries
        OB_STRENGTH_NORM: Normalized strength measure
    """
```

**Order Block Visualization:**

```
BULLISH ORDER BLOCK
───────────────────

              ┌────┐
              │████│ ← Green candle (closes higher)
              │████│   Makes new high → confirms OB
              │████│
              └────┘
        ┌────┐
        │    │ ← Red candle (closes lower)
        │    │   This becomes the DEMAND ZONE
        │    │
        └────┘
        ▲    ▲
        │    │
        │    └── BULLISH_OB_HIGH
        └─────── BULLISH_OB_LOW

When price returns to this zone, expect buying pressure.


BEARISH ORDER BLOCK
───────────────────

        ┌────┐
        │████│ ← Green candle (closes higher)
        │████│   This becomes the SUPPLY ZONE
        │████│
        └────┘
        ▲    ▲
        │    │
        │    └── BEARISH_OB_HIGH
        └─────── BEARISH_OB_LOW
              ┌────┐
              │    │ ← Red candle (closes lower)
              │    │   Makes new low → confirms OB
              │    │
              └────┘

When price returns to this zone, expect selling pressure.
```

**OB Strength Calculation:**

```python
OB_STRENGTH = (OB_HIGH - OB_LOW) / ATR

Higher strength = larger candle relative to current volatility
                = more significant institutional activity
```

---

# ==============================================================================
# PART 6: AGENTIC SYSTEM
# ==============================================================================

## 6.1 Event-Driven Architecture (events.py)

The event system enables loose coupling between agents.

```python
class EventType(Enum):
    """Types of events in the trading system."""
    # Trade lifecycle
    TRADE_PROPOSED = "trade_proposed"
    TRADE_APPROVED = "trade_approved"
    TRADE_REJECTED = "trade_rejected"
    TRADE_EXECUTED = "trade_executed"
    TRADE_CLOSED = "trade_closed"

    # Risk events
    RISK_ALERT = "risk_alert"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    DRAWDOWN_WARNING = "drawdown_warning"

    # System events
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_ERROR = "agent_error"

    # Market events
    MARKET_REGIME_CHANGE = "market_regime_change"
    VOLATILITY_SPIKE = "volatility_spike"


class DecisionType(Enum):
    """Agent decision types."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    DEFER = "defer"


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
```

**EventBus Implementation:**

```python
class EventBus:
    """
    Central event bus for inter-agent communication.

    Features:
    - Thread-safe pub/sub
    - Event deduplication
    - Event persistence
    - Handler timeouts
    - Event history

    Sprint 2 Fixes:
    - Race condition fix: Keep lock during handler calls
    - Dedicated dedup lock for atomic duplicate checking
    - Buffered persistence (100 events or 5 seconds)
    - O(1) history trimming with deque
    """

    def __init__(self, persist_events: bool = True):
        self._handlers: Dict[EventType, List[Callable]] = {}
        self._lock = RLock()  # Main lock
        self._dedup_lock = Lock()  # Separate lock for deduplication
        self._event_history = deque(maxlen=10000)  # O(1) trimming
        self._processed_event_times: Dict[str, datetime] = {}
        self._persist_buffer: List[Dict] = []
        self._persist_buffer_size = 100
```

**Event Flow Example:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVENT FLOW                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PPO Agent proposes trade                                                │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  event_bus.publish(Event(                                            │   │
│  │      type=TRADE_PROPOSED,                                            │   │
│  │      data={'action': 'OPEN_LONG', 'size': 0.5, 'price': 2000}       │   │
│  │  ))                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     │                                                                       │
│     │ Event Bus distributes to all subscribers                              │
│     │                                                                       │
│     ├───────────────────────────┬───────────────────────────┐              │
│     ▼                           ▼                           ▼              │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│  │ Risk        │         │ Kill Switch │         │ Audit       │          │
│  │ Sentinel    │         │             │         │ Logger      │          │
│  └─────────────┘         └─────────────┘         └─────────────┘          │
│     │                           │                       │                  │
│     │ Validates trade           │ Checks if halted      │ Logs event      │
│     │                           │                       │                  │
│     ▼                           ▼                       │                  │
│  Decision: APPROVE           Status: OK                 │                  │
│     │                           │                       │                  │
│     └───────────────────────────┘                       │                  │
│                 │                                        │                  │
│                 ▼                                        │                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  event_bus.publish(Event(                                            │   │
│  │      type=TRADE_APPROVED,                                            │   │
│  │      data={'action': 'OPEN_LONG', 'size': 0.5, ...}                 │   │
│  │  ))                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     │                                                                       │
│     ▼                                                                       │
│  Trade execution proceeds...                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Thread Safety Fix (Sprint 2):**

```python
# BEFORE (Race Condition):
def publish(self, event: Event):
    with self._lock:
        handlers = self._handlers.get(event.type, [])
    # Lock released here! Another thread could modify handlers!
    for handler in handlers:
        handler(event)  # Handler list might be stale


# AFTER (Fixed):
def publish(self, event: Event):
    with self._lock:
        handlers = list(self._handlers.get(event.type, []))  # Copy
        for handler in handlers:
            handler(event)  # Still under lock protection
```

---

## 6.2 Base Agent Framework (base_agent.py)

```python
class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    Provides:
    - Lifecycle management (start/stop/pause)
    - Health monitoring
    - Metrics collection
    - Audit logging
    - Event integration

    Subclasses must implement:
    - _process_event(event): Handle incoming events
    - _get_status(): Return current status
    """

    def __init__(self, agent_id: str, config: AgentConfig):
        self.agent_id = agent_id
        self.config = config
        self._state = AgentState.INITIALIZED
        self._event_bus: Optional[EventBus] = None
        self._metrics: Dict[str, Any] = {}
        self._last_heartbeat = datetime.now()
        self._error_count = 0
        self._lock = RLock()
```

**Agent Lifecycle:**

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         AGENT LIFECYCLE                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────┐                                                       │
│  │   INITIALIZED   │ ─── Agent created, not running                        │
│  └────────┬────────┘                                                       │
│           │ start()                                                        │
│           ▼                                                                │
│  ┌─────────────────┐                                                       │
│  │    STARTING     │ ─── Connecting to event bus, loading state            │
│  └────────┬────────┘                                                       │
│           │ _on_start() completes                                          │
│           ▼                                                                │
│  ┌─────────────────┐                                                       │
│  │    RUNNING      │ ─── Processing events normally                        │
│  └────────┬────────┘                                                       │
│           │                                                                │
│           ├────── pause() ───────►┌─────────────────┐                      │
│           │                       │     PAUSED      │                      │
│           │◄───── resume() ───────└─────────────────┘                      │
│           │                                                                │
│           │ stop()                                                         │
│           ▼                                                                │
│  ┌─────────────────┐                                                       │
│  │    STOPPING     │ ─── Disconnecting, saving state                       │
│  └────────┬────────┘                                                       │
│           │ _on_stop() completes                                           │
│           ▼                                                                │
│  ┌─────────────────┐                                                       │
│  │    STOPPED      │ ─── Agent fully stopped                               │
│  └─────────────────┘                                                       │
│                                                                            │
│  On any error:                                                             │
│  ┌─────────────────┐                                                       │
│  │     ERROR       │ ─── Agent encountered fatal error                     │
│  └─────────────────┘                                                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6.3 Kill Switch System (kill_switch.py)

The kill switch provides emergency trading halt capabilities.

```python
class KillSwitch:
    """
    Emergency trading halt system with multiple protection layers.

    Layer 1: Circuit Breakers (Automatic)
        - Daily loss limit
        - Weekly loss limit
        - Max drawdown
        - Consecutive losses
        - Loss velocity
        - VaR breach

    Layer 2: Hard Limits (Non-bypassable)
        - Cannot be overridden by code
        - Require manual reset with confirmation

    Layer 3: Manual Controls
        - Emergency halt button
        - Gradual wind-down mode

    Layer 4: Recovery Procedures
        - Cooling-off periods
        - Gradual position rebuild
        - Confirmation requirements

    Sprint 2 Fixes:
        - Peak equity initialization from actual equity
        - Manual halt requires explicit confirmation to clear
        - Recovery tokens use callback instead of return
    """
```

**Circuit Breaker Pattern:**

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         CIRCUIT BREAKER STATES                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────┐                                                       │
│  │     CLOSED      │ ◄─── Normal operation                                 │
│  │  (Normal ops)   │      Monitoring metrics                               │
│  └────────┬────────┘                                                       │
│           │                                                                │
│           │ Threshold breached!                                            │
│           │ (e.g., daily loss > 3%)                                        │
│           │                                                                │
│           ▼                                                                │
│  ┌─────────────────┐                                                       │
│  │      OPEN       │ ◄─── Trading blocked                                  │
│  │  (Blocking)     │      All new trades rejected                          │
│  └────────┬────────┘      Existing positions may close only                │
│           │                                                                │
│           │ Cooldown period elapsed                                        │
│           │ (e.g., 1 hour)                                                 │
│           │                                                                │
│           ▼                                                                │
│  ┌─────────────────┐                                                       │
│  │   HALF_OPEN     │ ◄─── Testing recovery                                 │
│  │  (Testing)      │      Limited trading allowed                          │
│  └────────┬────────┘      Monitoring closely                               │
│           │                                                                │
│           ├─── Success ───► Back to CLOSED                                 │
│           │                                                                │
│           └─── Failure ───► Back to OPEN (longer cooldown)                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**Kill Switch Configuration:**

```python
@dataclass
class KillSwitchConfig:
    """Configuration for kill switch thresholds."""

    # Daily limits
    max_daily_loss_pct: float = 0.03          # 3% daily loss
    max_daily_loss_usd: float = float('inf')  # No USD limit

    # Weekly limits
    max_weekly_loss_pct: float = 0.05         # 5% weekly loss

    # Total limits
    max_drawdown_pct: float = 0.10            # 10% max drawdown

    # Consecutive losses
    max_consecutive_losses: int = 5           # 5 losses in a row

    # Loss velocity
    loss_velocity_window_minutes: int = 30    # Look back 30 min
    loss_velocity_threshold_pct: float = 0.02 # 2% loss in window

    # Recovery
    default_cooldown_seconds: int = 3600      # 1 hour cooldown
    require_manual_reset: bool = True
```

**Sprint 2 Security Fix - Manual Halt Bypass Prevention:**

```python
# BEFORE (Vulnerable):
def confirm_reset(self, token: str) -> bool:
    if self._validate_token(token):
        self._is_halted = False  # Clears ALL halts including manual!
        return True
    return False


# AFTER (Fixed):
def confirm_reset(
    self,
    token: str,
    clear_manual_halt: bool = False,
    reason: str = ""
) -> bool:
    """
    Confirm reset with explicit manual halt clearing.

    SECURITY: Manual halts require:
    1. clear_manual_halt=True explicitly set
    2. Reason string with at least 10 characters
    """
    if self._is_manually_halted:
        if not clear_manual_halt:
            logging.warning("Manual halt requires explicit clear_manual_halt=True")
            return False
        if not reason or len(reason) < 10:
            logging.warning("Manual halt clear requires reason (min 10 chars)")
            return False

    # Proceed with reset...
```

---

## 6.4 Multi-Agent Orchestrator (orchestrator.py)

```python
class AgentOrchestrator:
    """
    Coordinates multiple agents and manages their interactions.

    Responsibilities:
    - Agent lifecycle management
    - Event routing
    - Failure detection and recovery
    - Load balancing
    - Health monitoring

    Sprint 2 Fix:
    - Thread-safe agent failure tracking
    - Circuit breaker for failed agents
    """
```

**Orchestrator Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT ORCHESTRATOR                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Event Bus                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                    │                    │                       │
│           ▼                    ▼                    ▼                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Risk Sentinel  │  │   Kill Switch   │  │  Portfolio Risk │             │
│  │                 │  │                 │  │                 │             │
│  │  Validates      │  │  Emergency      │  │  VaR/CVaR      │             │
│  │  trade signals  │  │  halt system    │  │  calculations   │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│           │                    │                    │                       │
│           └────────────────────┼────────────────────┘                       │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Orchestrator Core                                │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│   │
│  │  │   Agent     │  │  Circuit    │  │   Health    │  │   Failure   ││   │
│  │  │  Registry   │  │  Breaker    │  │   Monitor   │  │   Handler   ││   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘│   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Thread-Safe Failure Tracking (Sprint 2 Fix):**

```python
def _record_agent_failure(self, agent_id: str) -> None:
    """Record an agent failure with thread safety."""
    with self._lock:
        self._agent_failure_counts[agent_id] += 1

        if self._agent_failure_counts[agent_id] >= self._circuit_failure_threshold:
            self._failed_agents.add(agent_id)
            self._circuit_open_time[agent_id] = datetime.now()
            logging.critical(f"Agent {agent_id} circuit breaker OPEN")


def _is_circuit_open(self, agent_id: str) -> bool:
    """Check if agent's circuit breaker is open (thread-safe)."""
    with self._lock:
        if agent_id not in self._failed_agents:
            return False

        # Check if cooldown has elapsed
        open_time = self._circuit_open_time.get(agent_id)
        if open_time and datetime.now() - open_time > self._circuit_cooldown:
            # Reset circuit breaker
            self._failed_agents.discard(agent_id)
            self._agent_failure_counts[agent_id] = 0
            return False

        return True
```

---

## 6.5 Portfolio Risk Manager (portfolio_risk.py)

```python
@dataclass
class VaRResult:
    """
    Value at Risk calculation result.

    Sprint 2 Fix:
    - Added is_valid field
    - Added error_message field
    - Empty data returns invalid result (not zero risk)
    """
    var_amount: float       # VaR in currency
    var_pct: float          # VaR as percentage
    confidence_level: float # e.g., 0.95 for 95%
    time_horizon_days: int  # e.g., 1 for daily VaR
    method: str             # "historical", "parametric", "monte_carlo"
    is_valid: bool = True   # New: indicates if calculation succeeded
    error_message: Optional[str] = None  # New: error details if invalid


class PortfolioRiskManager:
    """
    Portfolio-level risk calculations.

    Methods:
    - calculate_var(): Value at Risk
    - calculate_cvar(): Conditional VaR (Expected Shortfall)
    - calculate_correlation_matrix(): Asset correlations
    - calculate_portfolio_beta(): Market beta
    """
```

**VaR Calculation:**

```
VALUE AT RISK (VaR)
───────────────────

VaR answers: "What is the maximum loss with X% confidence over Y days?"

Example: 95% 1-day VaR = $500
Meaning: We are 95% confident that the loss will not exceed $500 in one day.
         (There's still a 5% chance of losing MORE than $500)


HISTORICAL VAR METHOD:
1. Collect historical returns (e.g., 252 days)
2. Sort returns from worst to best
3. Find the 5th percentile (for 95% VaR)
4. That return × portfolio value = VaR

Example:
    Returns: [-5%, -3%, -2%, -2%, -1%, 0%, 1%, 2%, 3%, 4%]
    5th percentile (worst 5%) = -3%
    Portfolio = $10,000
    VaR = $10,000 × 3% = $300


Sprint 2 Fix - Empty Data Handling:

# BEFORE (Dangerous):
if len(returns) == 0:
    return VaRResult(var_amount=0.0, var_pct=0.0, ...)
    # Problem: Zero risk is WRONG! No data ≠ no risk

# AFTER (Safe):
if len(returns) == 0:
    return VaRResult(
        var_amount=float('inf'),  # Infinite risk (worst case)
        var_pct=1.0,              # 100% potential loss
        is_valid=False,           # Mark as invalid
        error_message="Insufficient data for VaR calculation"
    )
```

---

## 6.6 Ensemble Risk Model (ensemble_risk_model.py)

```python
class EnsembleRiskModel:
    """
    Machine learning ensemble for risk prediction.

    Components:
    1. XGBoost: Gradient boosted trees (non-linear patterns)
    2. LSTM: Recurrent network (temporal patterns)
    3. MLP: Dense network (complex interactions)

    Ensemble combines predictions with weighted averaging.

    Sprint 2 Fix:
    - LSTM sequence length validation with padding/truncation
    """
```

**Ensemble Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENSEMBLE RISK MODEL                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Feature matrix X (batch_size, seq_len, features)                   │
│         │                                                                   │
│         ├───────────────────────┬───────────────────────┐                  │
│         ▼                       ▼                       ▼                  │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│  │    XGBoost      │   │      LSTM       │   │       MLP       │          │
│  │                 │   │                 │   │                 │          │
│  │ • Tree ensemble │   │ • 2 LSTM layers │   │ • 3 Dense layers│          │
│  │ • Handles       │   │ • Temporal      │   │ • Non-linear    │          │
│  │   non-linear    │   │   dependencies  │   │   interactions  │          │
│  │ • Feature       │   │ • Sequences     │   │ • Dropout       │          │
│  │   importance    │   │                 │   │   regularization│          │
│  │                 │   │                 │   │                 │          │
│  │ Weight: 0.4     │   │ Weight: 0.3     │   │ Weight: 0.3     │          │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘          │
│           │                     │                     │                    │
│           │ pred_xgb            │ pred_lstm           │ pred_mlp          │
│           │                     │                     │                    │
│           └─────────────────────┼─────────────────────┘                    │
│                                 │                                          │
│                                 ▼                                          │
│                    ┌─────────────────────────┐                             │
│                    │   Weighted Average       │                             │
│                    │                          │                             │
│                    │ final = 0.4 × xgb +      │                             │
│                    │         0.3 × lstm +     │                             │
│                    │         0.3 × mlp        │                             │
│                    └─────────────────────────┘                             │
│                                 │                                          │
│                                 ▼                                          │
│  OUTPUT: Risk score [0, 1] (0 = low risk, 1 = high risk)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Sprint 2 Fix - LSTM Sequence Length:**

```python
def forward(self, X: np.ndarray) -> np.ndarray:
    """LSTM forward pass with sequence length validation."""

    if self.fitted and X.shape[1] != self.sequence_length:
        logger.warning(
            f"LSTM sequence length mismatch: got {X.shape[1]}, "
            f"expected {self.sequence_length}"
        )

        # Pad if too short
        if X.shape[1] < self.sequence_length:
            padding = np.zeros((
                X.shape[0],
                self.sequence_length - X.shape[1],
                X.shape[2]
            ))
            X = np.concatenate([padding, X], axis=1)

        # Truncate if too long
        else:
            X = X[:, -self.sequence_length:, :]

    # Proceed with forward pass...
```

---

# ==============================================================================
# PART 7: TRAINING PIPELINE
# ==============================================================================

## 7.1 Agent Trainer (agent_trainer.py)

```python
class AgentTrainer:
    """
    PPO training manager with callbacks and evaluation.

    Methods:
    - train_offline(): Full training from scratch
    - continue_training(): Resume from checkpoint
    - fine_tune_online(): Adapt to new market data
    - evaluate(): Assess model performance
    """
```

**Training Process:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. ENVIRONMENT SETUP                                                       │
│     env = TradingEnv(df_train, **hyperparams)                              │
│     eval_env = TradingEnv(df_val, **hyperparams)                           │
│                                                                             │
│  2. MODEL INITIALIZATION                                                    │
│     model = PPO(                                                            │
│         policy="MlpPolicy",                                                 │
│         env=env,                                                            │
│         learning_rate=3e-5,                                                 │
│         n_steps=2048,                                                       │
│         batch_size=128,                                                     │
│         gamma=0.99,                                                         │
│         ...                                                                 │
│     )                                                                       │
│                                                                             │
│  3. CALLBACKS SETUP                                                         │
│     callbacks = [                                                           │
│         EarlyStoppingCallback(patience=5),                                  │
│         EvalCallback(eval_env, eval_freq=10000),                           │
│         CheckpointCallback(save_freq=50000),                               │
│         ProgressCallback()                                                  │
│     ]                                                                       │
│                                                                             │
│  4. TRAINING LOOP                                                           │
│     model.learn(                                                            │
│         total_timesteps=1_500_000,                                         │
│         callback=callbacks,                                                 │
│         progress_bar=True                                                   │
│     )                                                                       │
│                                                                             │
│  5. FINAL EVALUATION                                                        │
│     metrics = evaluate(model, test_env)                                     │
│     # Sharpe, Calmar, Max DD, Win Rate, etc.                               │
│                                                                             │
│  6. SAVE MODEL                                                              │
│     model.save("trained_models/best_bot.zip")                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7.2 Parallel Training (parallel_training.py)

```python
def run_parallel_training():
    """
    Train multiple bots with different hyperparameters in parallel.

    Process:
    1. Load and split data (train/val/test)
    2. Generate N hyperparameter combinations
    3. Train N bots in parallel using ProcessPoolExecutor
    4. Evaluate each on validation set
    5. Select best by primary metric (Sharpe ratio)
    6. Final evaluation on test set

    Sprint 2 Fix:
    - Fixed sequential fallback to use ProcessPoolExecutor
    - Added garbage collection between folds
    """
```

**Parallel Training Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PARALLEL TRAINING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MAIN PROCESS                                                               │
│  ────────────                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. Load data                                                        │   │
│  │  2. Split: train (70%) / val (15%) / test (15%)                     │   │
│  │  3. Generate 50 hyperparameter sets                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           │ Submit to ProcessPoolExecutor                                   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    WORKER PROCESSES                                  │   │
│  │                                                                      │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐           │   │
│  │  │ Bot 1   │  │ Bot 2   │  │ Bot 3   │  ...  │ Bot 50  │           │   │
│  │  │         │  │         │  │         │       │         │           │   │
│  │  │ lr=1e-5 │  │ lr=3e-5 │  │ lr=5e-5 │       │ lr=1e-4 │           │   │
│  │  │ γ=0.99  │  │ γ=0.995 │  │ γ=0.99  │       │ γ=0.999 │           │   │
│  │  │ ent=0.02│  │ ent=0.05│  │ ent=0.10│       │ ent=0.05│           │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘       └────┬────┘           │   │
│  │       │            │            │                 │                 │   │
│  │       ▼            ▼            ▼                 ▼                 │   │
│  │    Train        Train        Train            Train                │   │
│  │    1.5M steps   1.5M steps   1.5M steps       1.5M steps           │   │
│  │       │            │            │                 │                 │   │
│  │       ▼            ▼            ▼                 ▼                 │   │
│  │    Evaluate     Evaluate     Evaluate         Evaluate             │   │
│  │    on Val       on Val       on Val           on Val               │   │
│  │       │            │            │                 │                 │   │
│  │       └────────────┴────────────┴─────────────────┘                 │   │
│  │                          │                                          │   │
│  └──────────────────────────┼──────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RESULTS AGGREGATION                                                 │   │
│  │                                                                      │   │
│  │  Bot 1:  Sharpe=1.2, MaxDD=12%, WinRate=54%                         │   │
│  │  Bot 2:  Sharpe=1.8, MaxDD=8%,  WinRate=57%  ← BEST                 │   │
│  │  Bot 3:  Sharpe=1.5, MaxDD=10%, WinRate=55%                         │   │
│  │  ...                                                                 │   │
│  │  Bot 50: Sharpe=0.9, MaxDD=18%, WinRate=49%                         │   │
│  │                                                                      │   │
│  │  SELECT: Bot 2 (highest Sharpe, meets all criteria)                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FINAL TEST EVALUATION (Bot 2 only)                                  │   │
│  │                                                                      │   │
│  │  Test Set Performance:                                               │   │
│  │    Sharpe: 1.65                                                      │   │
│  │    Max Drawdown: 9.2%                                                │   │
│  │    Win Rate: 56%                                                     │   │
│  │    Total Return: +24.5%                                              │   │
│  │                                                                      │   │
│  │  → Deploy as production model                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7.3 Walk-Forward Validation

```
WALK-FORWARD VALIDATION
═══════════════════════

Traditional train/test split:
    ├──────── TRAIN ────────┼──── TEST ────┤
    2019                    2023           2024

Problem: Single test period may not be representative
         Model could overfit to specific market conditions


Walk-Forward approach:
    Fold 1: Train 2019-2020, Test 2021 Q1
    Fold 2: Train 2019-2021 Q1, Test 2021 Q2
    Fold 3: Train 2019-2021 Q2, Test 2021 Q3
    ...and so on

Timeline:
    2019────2020────2021────2022────2023────2024
    │                                          │
    Fold 1: ├─TRAIN─┤TEST│
    Fold 2: ├──TRAIN──┤TEST│
    Fold 3: ├───TRAIN───┤TEST│
    Fold 4: ├────TRAIN────┤TEST│
    ...

Benefits:
    1. Multiple test periods → more robust performance estimate
    2. Simulates real deployment (train on past, test on future)
    3. Detects regime-specific failures
    4. Produces realistic performance expectations
```

---

## 7.4 Hyperparameter Search

```python
HYPERPARAM_SEARCH_SPACE = {
    # Learning rate: How fast to learn
    'learning_rate': [1e-5, 3e-5, 5e-5, 1e-4],

    # N_steps: Steps before policy update
    'n_steps': [1024, 2048, 4096],

    # Batch size: Samples per gradient update
    'batch_size': [64, 128, 256],

    # Gamma: Discount factor for future rewards
    'gamma': [0.99, 0.995, 0.999],

    # Entropy coefficient: Exploration bonus
    'ent_coef': [0.02, 0.05, 0.10],

    # Clip range: PPO policy constraint
    'clip_range': [0.1, 0.2, 0.3],

    # Reward scaling: Environment hyperparameters
    'reward_tanh_scale': [0.2, 0.3, 0.4],
    'reward_output_scale': [3.0, 5.0, 7.0],
}
```

**Search Strategy: Random Search**

```
Total possible combinations: 4 × 3 × 3 × 3 × 3 × 3 × 3 × 3 = 8,748

Grid search: Test ALL 8,748 → Infeasible (too slow)
Random search: Sample 50 random combinations → Efficient!

Why random search works:
    - Research shows random search finds good hyperparams
      as efficiently as grid search for most problems
    - Each hyperparameter has equal chance of being explored
    - Covers more of the space with fewer trials
```

---

## 7.5 Model Selection

```python
def select_best_model(results: List[Dict]) -> Dict:
    """
    Select best model based on multiple criteria.

    Primary metric: Sharpe Ratio (risk-adjusted return)

    Filters (must pass ALL):
        - Sharpe > 1.5
        - Calmar > 2.0
        - Max Drawdown < 15%
        - Win Rate > 50%

    Ranking: By Sharpe ratio (highest wins)
    """
```

**Selection Process:**

```
INPUT: 50 trained bots with validation metrics

FILTER STAGE:
    Bot 1:  Sharpe=1.2 → REJECT (below 1.5)
    Bot 2:  Sharpe=1.8, DD=8% → PASS
    Bot 3:  Sharpe=1.6, DD=18% → REJECT (DD > 15%)
    Bot 4:  Sharpe=2.1, DD=12% → PASS
    ...
    Bot 50: Sharpe=1.5, DD=14% → PASS

RANK STAGE:
    1. Bot 4:  Sharpe=2.1
    2. Bot 2:  Sharpe=1.8
    3. Bot 50: Sharpe=1.5

FINAL SELECTION:
    Winner: Bot 4 (highest Sharpe that passed all filters)

OUTPUT: Bot 4 model for production deployment
```

---

# ==============================================================================
# PART 8: SECURITY AND RELIABILITY FIXES (Sprint 2)
# ==============================================================================

## 8.1 Race Condition Fixes

**Problem: Event Bus Handler Race**

```python
# BEFORE (Vulnerable to race condition):
def publish(self, event: Event):
    with self._lock:
        handlers = self._handlers.get(event.type, [])
    # Lock released here!
    # Another thread could modify handlers list!
    for handler in handlers:
        handler(event)  # May crash or skip handlers
```

```python
# AFTER (Thread-safe):
def publish(self, event: Event):
    with self._lock:
        handlers = list(self._handlers.get(event.type, []))  # Copy under lock
        for handler in handlers:
            try:
                handler(event)  # Execute while lock held
            except Exception as e:
                logging.error(f"Handler failed: {e}")
```

**Problem: Deduplication Race**

```python
# BEFORE (Race condition between check and insert):
def _is_duplicate(self, event_id: str) -> bool:
    if event_id in self._processed_event_times:  # Check
        return True
    # Another thread could insert same event_id here!
    self._processed_event_times[event_id] = now  # Insert
    return False
```

```python
# AFTER (Atomic check-and-insert):
def _is_duplicate(self, event_id: str) -> bool:
    with self._dedup_lock:  # Dedicated lock for dedup
        if event_id in self._processed_event_times:
            return True
        self._processed_event_times[event_id] = datetime.now()
    return False
```

---

## 8.2 Balance Protection

**Problem: Direct Balance Manipulation**

```python
# BEFORE (No protection):
env.balance = "not a number"  # Silently corrupts state
env.balance = float('nan')    # Causes NaN propagation
env.balance = -1000           # Invalid negative balance
```

```python
# AFTER (Protected property):
@balance.setter
def balance(self, value: float) -> None:
    # Type validation
    if not isinstance(value, (int, float)):
        raise TypeError(f"Balance must be numeric, got {type(value)}")

    value = float(value)

    # NaN/Inf validation
    if np.isnan(value) or np.isinf(value):
        raise ValueError(f"Balance cannot be NaN or Inf: {value}")

    # Negative balance validation
    if value < 0 and not self.allow_negative_balance:
        raise ValueError(f"Balance cannot be negative: {value}")

    self._balance = value
```

---

## 8.3 Transaction Rollback

**Problem: Partial State Corruption on Trade Failure**

```python
# BEFORE (No rollback):
def _execute_trade(self, trade_type, price, quantity):
    self.balance -= cost          # Step 1: Deduct balance
    self.stock_quantity += qty    # Step 2: Add position
    commission = calculate(...)   # Step 3: Calculate commission
    # If Step 3 fails, Steps 1-2 already executed!
    # State is now corrupted
```

```python
# AFTER (With rollback):
def _execute_trade(self, trade_type, price, quantity):
    # Create snapshot BEFORE any changes
    snapshot = self._create_state_snapshot()

    try:
        self.balance -= cost
        self.stock_quantity += qty
        commission = calculate(...)
        return True, value, commission, pnl_abs, pnl_pct

    except Exception as e:
        # Rollback to pre-trade state
        self._restore_state_snapshot(snapshot)
        logging.error(f"Trade failed, state rolled back: {e}")
        return False, 0, 0, 0, 0
```

---

## 8.4 Kill Switch Hardening

**Problem 1: Peak Equity Initialization**

```python
# BEFORE (Wrong initialization):
def __init__(self, config=None):
    self._equity = 100.0       # Hardcoded
    self._peak_equity = 100.0  # Doesn't match actual equity
```

```python
# AFTER (Correct initialization):
def __init__(self, config=None, initial_equity: float = 100.0):
    self._initial_equity = max(initial_equity, 1.0)
    self._equity = self._initial_equity
    self._peak_equity = self._initial_equity  # Matches actual
```

**Problem 2: Manual Halt Bypass**

```python
# BEFORE (Any reset clears manual halt):
def confirm_reset(self, token: str) -> bool:
    if self._validate_token(token):
        self._is_halted = False  # Clears ALL halts!
        return True
```

```python
# AFTER (Explicit confirmation required):
def confirm_reset(
    self,
    token: str,
    clear_manual_halt: bool = False,
    reason: str = ""
) -> bool:
    if self._is_manually_halted:
        if not clear_manual_halt:
            logging.warning("Must explicitly set clear_manual_halt=True")
            return False
        if len(reason) < 10:
            logging.warning("Must provide reason (min 10 chars)")
            return False
    # Proceed with reset...
```

**Problem 3: Token Exposure**

```python
# BEFORE (Token returned directly):
def request_reset(self) -> Tuple[bool, str]:
    token = self._generate_token()
    return True, token  # Token exposed in return value!
```

```python
# AFTER (Token via callback only):
def request_reset(self, notification_callback: Optional[Callable] = None) -> bool:
    token = self._generate_token()
    if notification_callback:
        notification_callback(token)  # Token only to authorized callback
    return True  # Don't return token directly
```

---

## 8.5 Thread Safety Improvements

**Orchestrator Agent Failure Tracking:**

```python
# BEFORE (Race condition):
def _record_agent_failure(self, agent_id: str):
    self._agent_failure_counts[agent_id] += 1  # Not atomic!
    if self._agent_failure_counts[agent_id] >= threshold:
        self._failed_agents.add(agent_id)  # Race with _is_circuit_open


# AFTER (Thread-safe):
def _record_agent_failure(self, agent_id: str):
    with self._lock:  # Single lock for all operations
        self._agent_failure_counts[agent_id] += 1
        if self._agent_failure_counts[agent_id] >= threshold:
            self._failed_agents.add(agent_id)
            self._circuit_open_time[agent_id] = datetime.now()
```

**Config Validation:**

```python
# BEFORE (No validation):
@dataclass
class AgentConfig:
    log_level: str = "INFO"
    max_retries: int = 3


# AFTER (With validation):
@dataclass
class AgentConfig:
    log_level: str = "INFO"
    max_retries: int = 3

    def __post_init__(self):
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}")
        if not 0 <= self.max_retries <= 10:
            raise ValueError(f"max_retries must be 0-10, got {self.max_retries}")
```

---

# ==============================================================================
# PART 9: MATHEMATICAL FOUNDATIONS
# ==============================================================================

## 9.1 PPO Algorithm

**Proximal Policy Optimization (PPO)** is a policy gradient algorithm that
balances sample efficiency with stable learning.

```
PPO OBJECTIVE FUNCTION
══════════════════════

L(θ) = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]

Where:
    θ = Policy parameters (neural network weights)
    r(θ) = π(a|s,θ) / π(a|s,θ_old)  (probability ratio)
    A = Advantage estimate (how much better than expected)
    ε = Clip range (typically 0.2)
    clip() = Constrains ratio to [1-ε, 1+ε]


WHY CLIPPING MATTERS
────────────────────

Without clipping:
    If r(θ) = 10 and A = 1:
        L = 10 × 1 = 10  (Huge update!)
        → Policy changes dramatically
        → Likely to destabilize training

With clipping (ε = 0.2):
    r(θ) = 10 gets clipped to 1.2
    L = 1.2 × 1 = 1.2  (Moderate update)
    → Policy changes gradually
    → Stable training


ADVANTAGE ESTIMATION (GAE)
──────────────────────────

A(t) = δ(t) + γλ·δ(t+1) + (γλ)²·δ(t+2) + ...

Where:
    δ(t) = r(t) + γ·V(s(t+1)) - V(s(t))  (TD error)
    γ = Discount factor (0.99)
    λ = GAE parameter (0.95)

This balances bias (low λ) vs variance (high λ).
```

---

## 9.2 GARCH(1,1) Model

**GARCH** (Generalized Autoregressive Conditional Heteroskedasticity) models
time-varying volatility in financial returns.

```
GARCH(1,1) EQUATION
═══════════════════

σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)

Where:
    σ²(t) = Conditional variance at time t
    ω = Long-term variance weight (constant)
    α = Shock reaction coefficient (ARCH term)
    β = Persistence coefficient (GARCH term)
    ε(t-1) = Previous period's return shock


PARAMETER INTERPRETATION
────────────────────────

ω (omega): Base level of variance
    Higher ω → Higher baseline volatility

α (alpha): Reaction to recent shock
    Higher α → More reactive to yesterday's move
    Typical: 0.05 - 0.15

β (beta): Volatility persistence
    Higher β → Shocks decay slowly
    Typical: 0.80 - 0.95

Constraint: α + β < 1 for stationarity


EXAMPLE CALCULATION
───────────────────

Parameters: ω=0.00001, α=0.10, β=0.85
Yesterday: σ²(t-1)=0.0004, ε(t-1)=0.02 (2% return shock)

σ²(t) = 0.00001 + 0.10×(0.02)² + 0.85×0.0004
      = 0.00001 + 0.10×0.0004 + 0.00034
      = 0.00001 + 0.00004 + 0.00034
      = 0.00039

σ(t) = √0.00039 = 0.0197 = 1.97% daily volatility
Annualized: 1.97% × √252 = 31.3%
```

---

## 9.3 Kelly Criterion Formula

**Kelly Criterion** determines the optimal fraction of capital to bet.

```
KELLY FORMULA
═════════════

f* = (B × P - Q) / B

Where:
    f* = Optimal fraction of capital to bet
    B = Odds received (win amount / loss amount)
    P = Probability of winning
    Q = Probability of losing (1 - P)


DERIVATION INTUITION
────────────────────

Kelly maximizes long-term geometric growth rate:
    G = P × log(1 + f×B) + Q × log(1 - f)

Taking derivative and setting to zero:
    dG/df = P×B/(1+f×B) - Q/(1-f) = 0

Solving: f* = (BP - Q) / B


EXAMPLE
───────

Trading system statistics:
    Win rate: 55%
    Average win: $200
    Average loss: $100
    B = 200/100 = 2.0

    f* = (2.0 × 0.55 - 0.45) / 2.0
       = (1.10 - 0.45) / 2.0
       = 0.325 = 32.5%

Full Kelly says bet 32.5% of capital per trade.
In practice, use fractional Kelly (1/4 to 1/2):
    Practical bet = 32.5% × 0.25 = 8.1%


WHY FRACTIONAL KELLY?
─────────────────────

Full Kelly problems:
    1. Assumes perfect knowledge of P and B (never true)
    2. High volatility (50%+ drawdowns common)
    3. One bad streak can devastate account

Fractional Kelly (e.g., 25%):
    - More robust to parameter estimation errors
    - Significantly lower volatility
    - Captures ~75% of growth with ~50% of variance
```

---

## 9.4 VaR Calculations

**Value at Risk (VaR)** quantifies the maximum expected loss.

```
VAR DEFINITION
══════════════

VaR(α, t) answers:
"What is the maximum loss over time t with α% confidence?"

Example: 95% 1-day VaR = $500 means:
    "We are 95% confident the loss will not exceed $500 tomorrow."
    (5% chance of losing MORE than $500)


HISTORICAL VAR METHOD
─────────────────────

1. Collect N historical returns (e.g., 252 days)
2. Sort returns from worst to best
3. Find the (100-α)th percentile

Example:
    Returns: [-5%, -4%, -3%, -2%, -2%, -1%, -1%, 0%, 1%, 2%, ...]
    For 95% VaR: Find 5th percentile
    5th percentile of 100 returns = 5th worst = -3%

    Portfolio = $10,000
    VaR = $10,000 × 3% = $300


PARAMETRIC VAR (Normal Distribution)
────────────────────────────────────

VaR = μ + z_α × σ

Where:
    μ = Mean return (often assumed 0 for short periods)
    z_α = Standard normal quantile (z_0.95 = -1.645)
    σ = Standard deviation of returns

Example:
    μ = 0
    σ = 2% daily
    z_0.95 = -1.645

    VaR = 0 + (-1.645) × 2% = -3.29%
    For $10,000: VaR = $329


CONDITIONAL VAR (CVaR / Expected Shortfall)
───────────────────────────────────────────

CVaR = Expected loss GIVEN that loss exceeds VaR

If 95% VaR = $300, then 95% CVaR might be $450,
meaning "when we do lose more than $300, we lose $450 on average."

CVaR is considered more robust than VaR because:
    - It considers tail severity, not just frequency
    - It's coherent (satisfies subadditivity)
```

---

## 9.5 Sharpe Ratio

**Sharpe Ratio** measures risk-adjusted return.

```
SHARPE RATIO FORMULA
════════════════════

Sharpe = (R_p - R_f) / σ_p

Where:
    R_p = Portfolio return (annualized)
    R_f = Risk-free rate (e.g., Treasury yield)
    σ_p = Portfolio standard deviation (annualized)


INTERPRETATION
──────────────

Sharpe < 1.0: Subpar (return doesn't justify risk)
Sharpe 1.0-2.0: Good (acceptable risk-adjusted return)
Sharpe 2.0-3.0: Excellent (strong edge)
Sharpe > 3.0: Outstanding (rare, verify not overfitting)


EXAMPLE CALCULATION
───────────────────

Strategy performance:
    Annual return: 20%
    Annual volatility: 15%
    Risk-free rate: 5%

    Sharpe = (20% - 5%) / 15%
           = 15% / 15%
           = 1.0

Interpretation: For every unit of risk, you earn one unit of excess return.


ANNUALIZATION
─────────────

From daily to annual:
    Annual Return = Daily Return × 252
    Annual Volatility = Daily Volatility × √252

From monthly to annual:
    Annual Return = Monthly Return × 12
    Annual Volatility = Monthly Volatility × √12
```

---

# ==============================================================================
# PART 10: PRACTICAL EXAMPLES
# ==============================================================================

## 10.1 Complete Trading Cycle

```
COMPLETE TRADING CYCLE EXAMPLE
══════════════════════════════

INITIAL STATE
─────────────
Balance: $1,000
Position: FLAT
Net Worth: $1,000

STEP 1: MARKET ANALYSIS (step=100)
──────────────────────────────────
Current Price: $2,000/oz
ATR: $15
RSI: 32 (oversold)
BOS_SIGNAL: +1 (bullish break)
FVG_SIGNAL: +1 (bullish gap)

Observation vector created:
    - 30 bars of 26 features (scaled 0-1)
    - Portfolio: [1.0, 0.0, 1.0] (normalized)

STEP 2: AGENT DECISION
──────────────────────
PPO policy processes observation:
    Action probabilities: [0.1, 0.65, 0.05, 0.15, 0.05]
    Selected action: 1 (OPEN_LONG) with 65% confidence

STEP 3: RISK MANAGER CALCULATIONS
─────────────────────────────────
Stop Loss: $2,000 - (2 × $15) = $1,970
Take Profit: $2,000 × 1.02 = $2,040
SL Distance: $30

Position Size (Triple Constraint):
    Risk-based: $10 risk / $30 distance = 0.333 oz
    Kelly: $50 / $30 = 1.67 oz (capped by limit)
    Leverage: $1,000 / $2,000 = 0.5 oz

    Final: min(0.333, 1.67, 0.5) = 0.333 oz

STEP 4: TRADE EXECUTION
───────────────────────
Buy 0.333 oz @ $2,000
    Gross: $666
    Spread: $0.33
    Slippage: $0.07
    Commission: $0.33
    Total Cost: $666.73

STATE AFTER OPEN:
    Balance: $333.27
    Position: LONG 0.333 oz @ $2,000
    Net Worth: $999.27 (down $0.73 from costs)

STEP 5: HOLDING PERIOD (steps 101-115)
──────────────────────────────────────
Price moves: $2,000 → $2,015 → $2,025 → $2,035

At $2,035:
    Unrealized P&L: 0.333 × ($2,035 - $2,000) = $11.65
    Profit > 1× risk ($10) → TSL ACTIVATES
    TSL set at: $2,035 - $7.50 = $2,027.50

Price continues: $2,035 → $2,042

STEP 6: TAKE PROFIT HIT (step=116)
──────────────────────────────────
Price: $2,042 > TP ($2,040)
Risk manager triggers: exit_signal = 'TP'
Action overridden to: CLOSE_LONG

STEP 7: CLOSE EXECUTION
───────────────────────
Sell 0.333 oz @ $2,042
    Gross: $679.99
    Spread: $0.34
    Slippage: $0.07
    Commission: $0.34
    Net Revenue: $679.24

P&L Calculation:
    Cost basis: $666
    Net revenue: $679.24
    P&L: $13.24 (+1.99%)

FINAL STATE:
    Balance: $1,012.51
    Position: FLAT
    Net Worth: $1,012.51 (+1.25%)
    Total Trades: 1
    Winning Trades: 1

REWARD CALCULATION:
    log_return = log(1012.51/999.27) = 0.0132
    profitability = 0.0132 × 100 = 1.32
    bonus (winning trade) = min(2.0, 1.99/100 × 10) = 0.199
    combined = 1.32 + 0.199 = 1.52
    normalized = tanh(1.52 × 0.3) = 0.42
    scaled = 0.42 × 5.0 = 2.1

    Final Reward: +2.1
```

---

## 10.2 Risk Management in Action

```
RISK MANAGEMENT SCENARIO
════════════════════════

SCENARIO: Volatile Market with Consecutive Losses

STARTING STATE:
    Equity: $10,000
    Peak Equity: $10,000
    Daily Loss: $0
    Consecutive Losses: 0

TRADE 1: Loss
─────────────
Open Short @ $2,000, Close @ $2,020
P&L: -$60 (-0.6%)

State:
    Equity: $9,940
    Daily Loss: $60 (0.6%)
    Consecutive: 1
    Kill Switch: GREEN ✓

TRADE 2: Loss
─────────────
Open Long @ $2,015, Close @ $1,995
P&L: -$80 (-0.8%)

State:
    Equity: $9,860
    Daily Loss: $140 (1.4%)
    Consecutive: 2
    Kill Switch: GREEN ✓

TRADE 3: Loss
─────────────
Open Long @ $2,000, Close @ $1,970
P&L: -$120 (-1.2%)

State:
    Equity: $9,740
    Daily Loss: $260 (2.6%)
    Consecutive: 3
    Kill Switch: YELLOW ⚠ (approaching 3% daily limit)

TRADE 4: Loss
─────────────
Open Short @ $1,980, Close @ $2,000
P&L: -$80 (-0.8%)

State:
    Equity: $9,660
    Daily Loss: $340 (3.4%)
    Consecutive: 4

    ╔════════════════════════════════════════════════╗
    ║  CIRCUIT BREAKER TRIGGERED!                    ║
    ║  Reason: Daily loss exceeded 3% limit          ║
    ║  Action: Trading HALTED                        ║
    ║  Cooldown: 1 hour                              ║
    ╚════════════════════════════════════════════════╝

DURING HALT:
    - All OPEN_LONG/OPEN_SHORT actions → HOLD
    - Can still close existing positions
    - Cannot open new positions

AFTER 1 HOUR:
    Circuit breaker enters HALF_OPEN state
    Limited trading allowed (50% position size)
    Monitoring for further losses

IF NEXT TRADE WINS:
    Circuit breaker → CLOSED (normal operation)

IF NEXT TRADE LOSES:
    Circuit breaker → OPEN (extended 2-hour cooldown)
```

---

## 10.3 Event Flow Example

```
EVENT FLOW: Trade Proposal to Execution
═══════════════════════════════════════

1. PPO AGENT PROPOSES TRADE
───────────────────────────
Agent wants to open long position

event_bus.publish(Event(
    type=TRADE_PROPOSED,
    source="ppo_agent",
    data={
        'action': 'OPEN_LONG',
        'confidence': 0.72,
        'size_requested': 0.5,
        'entry_price': 2000,
        'sl_price': 1970,
        'tp_price': 2040
    }
))


2. RISK SENTINEL RECEIVES EVENT
────────────────────────────────
Validates against 15+ rules:

    ✓ Position size within limits
    ✓ Not exceeding max drawdown
    ✓ Leverage within limits
    ✓ Not in cooldown period
    ✓ Not too many open positions
    ✓ Volatility filter passed
    ...

Decision: APPROVE

event_bus.publish(Event(
    type=TRADE_APPROVED,
    source="risk_sentinel",
    data={
        'original_request': {...},
        'approved_size': 0.5,
        'risk_score': 0.3,
        'validation_results': {...}
    }
))


3. KILL SWITCH CHECKS
─────────────────────
Verifies trading is allowed:

    ✓ Not in HALT state
    ✓ Daily loss limit not reached
    ✓ No circuit breakers tripped

Status: TRADING ALLOWED


4. AUDIT LOGGER RECORDS
───────────────────────
Logs event to persistent storage:

{
    "timestamp": "2024-01-15T14:30:00Z",
    "event_type": "TRADE_APPROVED",
    "agent": "risk_sentinel",
    "trade_id": "TRD-2024-001234",
    "details": {...}
}


5. EXECUTION ENGINE PROCESSES
─────────────────────────────
Receives TRADE_APPROVED event
Executes trade in environment

event_bus.publish(Event(
    type=TRADE_EXECUTED,
    source="execution_engine",
    data={
        'trade_id': 'TRD-2024-001234',
        'executed_price': 2000.50,
        'executed_size': 0.5,
        'slippage': 0.025%,
        'commission': $0.50
    }
))


6. PORTFOLIO MANAGER UPDATES
────────────────────────────
Updates portfolio state:
    - New position added
    - Balance updated
    - Risk metrics recalculated


7. MONITORING AGENT ALERTS
──────────────────────────
If configured, sends notification:
    - Discord webhook
    - Email alert
    - Dashboard update
```

---

## 10.4 Training a New Bot

```
TRAINING A NEW BOT: Step-by-Step Guide
══════════════════════════════════════

STEP 1: PREPARE DATA
────────────────────
Place your CSV file in data/ directory:
    data/XAU_15MIN_2019_2024.csv

Required columns:
    - Date (or timestamp)
    - Open, High, Low, Close
    - Volume

STEP 2: CONFIGURE HYPERPARAMETERS
─────────────────────────────────
Edit config.py:

    # Training duration
    TOTAL_TIMESTEPS_PER_BOT = 1_500_000

    # PPO settings
    MODEL_HYPERPARAMETERS = {
        "learning_rate": 3e-5,
        "n_steps": 2048,
        "batch_size": 128,
        "gamma": 0.99,
        "ent_coef": 0.05,
    }

    # Risk settings
    RISK_PERCENTAGE_PER_TRADE = 0.01
    MAX_LEVERAGE = 1.0

STEP 3: RUN SINGLE BOT TRAINING
───────────────────────────────
python -c "
from src.agent_trainer import AgentTrainer
import pandas as pd

# Load data
df = pd.read_csv('data/XAU_15MIN_2019_2024.csv')

# Split
train_end = int(len(df) * 0.7)
df_train = df[:train_end]
df_val = df[train_end:]

# Train
trainer = AgentTrainer(df_train, df_val)
model = trainer.train_offline(timesteps=1_500_000)

# Save
model.save('trained_models/my_bot.zip')
"

STEP 4: RUN PARALLEL TRAINING (50 BOTS)
───────────────────────────────────────
python parallel_training.py

Output:
    Training 50 bots with different hyperparameters...
    [============================] 100%

    Results:
    Bot 23: Sharpe=1.85, MaxDD=9.2%, WinRate=57%  ← BEST
    Bot 7:  Sharpe=1.72, MaxDD=10.1%, WinRate=55%
    Bot 41: Sharpe=1.68, MaxDD=11.3%, WinRate=54%
    ...

    Best model saved to: trained_models/best_bot_20240115.zip

STEP 5: EVALUATE ON TEST SET
────────────────────────────
python -c "
from stable_baselines3 import PPO
from src.environment.environment import TradingEnv
import pandas as pd

# Load test data (last 15%)
df = pd.read_csv('data/XAU_15MIN_2019_2024.csv')
test_start = int(len(df) * 0.85)
df_test = df[test_start:]

# Load model
model = PPO.load('trained_models/best_bot.zip')

# Evaluate
env = TradingEnv(df_test)
obs, _ = env.reset()
total_reward = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done or truncated:
        break

print(f'Total Reward: {total_reward}')
print(f'Final Balance: {info[\"balance\"]}')
print(f'Return: {info[\"episode_return_percentage\"]:.2f}%')
print(f'Total Trades: {info[\"total_trades\"]}')
"

STEP 6: DEPLOY (Optional)
─────────────────────────
For paper trading or live trading,
integrate with broker API (future sprint).
```

---

# ==============================================================================
# APPENDIX A: QUICK REFERENCE TABLES
# ==============================================================================

## Action Space Quick Reference

| Action | Code | Valid When | Effect |
|--------|------|------------|--------|
| HOLD | 0 | Always | Do nothing |
| OPEN_LONG | 1 | FLAT | Buy to profit on price UP |
| CLOSE_LONG | 2 | LONG | Sell to realize P&L |
| OPEN_SHORT | 3 | FLAT | Sell to profit on price DOWN |
| CLOSE_SHORT | 4 | SHORT | Buy to cover |

## Reward Components Quick Reference

| Component | Weight | Range | Purpose |
|-----------|--------|-------|---------|
| Profitability | 1.0 | [-∞, +∞] | Core return signal |
| Drawdown | 5.0 | [0, +∞] | Capital preservation |
| Friction | 2.0 | [0, +∞] | Discourage overtrading |
| Leverage | 10.0 | [0, +∞] | Enforce risk limits |
| Duration | 0.5 | [0, +∞] | Encourage timely exits |
| Invalid Action | 0.5 | {0, 0.5} | Discourage bad actions |
| Hold Penalty | 0.01 | {0, 0.01} | Encourage activity |
| Win Bonus | — | [0, 3] | Reward good trades |
| Loss Feedback | — | [-1, 0] | Learn from mistakes |

## Key Hyperparameters Quick Reference

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| learning_rate | 3e-5 | [1e-5, 1e-4] | Learning speed |
| n_steps | 2048 | [512, 4096] | Update frequency |
| batch_size | 128 | [32, 512] | Gradient stability |
| gamma | 0.99 | [0.9, 0.999] | Future reward importance |
| ent_coef | 0.05 | [0.01, 0.2] | Exploration bonus |
| clip_range | 0.2 | [0.1, 0.4] | Policy update constraint |

---

# ==============================================================================
# APPENDIX B: TROUBLESHOOTING GUIDE
# ==============================================================================

## Common Issues and Solutions

**Issue: Agent always chooses HOLD**
```
Cause: "Fearful agent" - learned that trading is risky
Solution:
    1. Increase HOLD penalty (hold_penalty = 0.01)
    2. Add invalid action penalty
    3. Reduce losing_trade_penalty
    4. Increase winning_trade_bonus
```

**Issue: High invalid action count**
```
Cause: Agent hasn't learned state machine
Solution:
    1. Add invalid action penalty
    2. Ensure observation includes position_type
    3. Train longer
    4. Check action masking implementation
```

**Issue: Training loss exploding**
```
Cause: Unstable gradients
Solution:
    1. Reduce learning_rate
    2. Increase batch_size
    3. Reduce reward scale
    4. Check for NaN in observations
```

**Issue: Poor generalization (train good, test bad)**
```
Cause: Overfitting
Solution:
    1. Use walk-forward validation
    2. Increase entropy coefficient
    3. Reduce model capacity
    4. Add dropout to policy network
```

**Issue: Memory errors during training**
```
Cause: Large batches or many parallel workers
Solution:
    1. Reduce n_steps
    2. Reduce batch_size
    3. Reduce MAX_WORKERS
    4. Add gc.collect() between folds
```

---

# ==============================================================================
# END OF DOCUMENTATION
# ==============================================================================

Document Statistics:
    Total Lines: ~4,500
    Sections: 10 major parts
    Code Examples: 50+
    Diagrams: 20+

Last Updated: January 2026
Version: 2.0 (Sprint 3 Ready)
Author: Claude Code Assistant

For questions or issues:
    GitHub: https://github.com/anthropics/claude-code/issues
