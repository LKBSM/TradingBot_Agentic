# Trading Bot Training Architecture — Complete Technical Guide

## Table of Contents

1. [Overview](#1-overview)
2. [End-to-End Training Flow](#2-end-to-end-training-flow)
3. [File Map](#3-file-map)
4. [Phase 1: Curriculum Learning](#4-phase-1-curriculum-learning)
5. [Phase 2: Ensemble Training](#5-phase-2-ensemble-training)
6. [Phase 3: Meta-Learning](#6-phase-3-meta-learning)
7. [Phase 4: Validation & Packaging](#7-phase-4-validation--packaging)
8. [Environment & Observation Space](#8-environment--observation-space)
9. [Reward System](#9-reward-system)
10. [Risk Management Layer](#10-risk-management-layer)
11. [Configuration Reference](#11-configuration-reference)
12. [Data Flow Diagram](#12-data-flow-diagram)

---

## 1. Overview

The training system is built around **Proximal Policy Optimization (PPO)**, a reinforcement learning algorithm where an agent learns to trade Gold (XAU/USD) on 15-minute bars by interacting with a simulated market environment.

The system is **not** a simple "train one model" pipeline. It is a **4-phase orchestrated pipeline** designed to produce a robust, regime-adaptive trading agent:

| Phase | Name | Compute Budget | Purpose |
|-------|------|---------------|---------|
| 1 | Curriculum Learning | 40% | Solve domain shift between training and production |
| 2 | Ensemble Training | 35% | Create diverse specialists for robustness |
| 3 | Meta-Learning | 25% | Enable fast adaptation to market regime changes |
| 4 | Integration | Synchronous | Validate, quality-gate, and package for deployment |

The master orchestrator is **`SophisticatedTrainer`** (`src/training/sophisticated_trainer.py`), which coordinates all four phases and produces a production-ready artifact.

---

## 2. End-to-End Training Flow

```
 STEP 1: Load Configuration
   config.py  -->  hyperparameters, risk limits, feature definitions
        |
 STEP 2: Load & Split Data
   Gold CSV (OHLCV) --> 70% train / 15% validation / 15% test
        |
 STEP 3: Phase 1 — Curriculum Learning (600K steps)
   CurriculumTrainer --> UnifiedAgenticEnv(BASE -> ENRICHED -> SOFT -> PRODUCTION)
        |                  + AdvancedRewardShaper (10 reward objectives)
        |                  + EntropyAnnealingCallback (exploration -> exploitation)
        v
   curriculum_model (PPO)
        |
 STEP 4: Phase 2 — Ensemble Training (525K steps)
   EnsembleTrainer --> 5-7 diverse specialists (different hyperparams/seeds)
        |               + Adaptive weight calibration
        v
   EnsembleModel (voting/weighted)
        |
 STEP 5: Phase 3 — Meta-Learning (375K steps)
   MetaLearner --> Regime detection + MAML-style inner/outer loop
        |           + OnlineAdapter for live regime switching
        v
   meta_model + OnlineAdapter
        |
 STEP 6: Phase 4 — Integration & Validation
   Walk-Forward Validation --> rolling train/test folds with purge gaps
   Quality Gates            --> Sharpe >= 1.0, DD <= 15%, WinRate >= 40%
   Artifact Packaging       --> model.zip + PCA + config + SHA-256 manifest
        |
        v
   production_artifact/  (ready for deployment)
```

---

## 3. File Map

Every file involved in the training pipeline and why it exists:

### Core Training Pipeline (`src/training/`)

| File | Lines | Role |
|------|-------|------|
| `sophisticated_trainer.py` | ~1100 | **Master orchestrator.** Coordinates all 4 phases, implements walk-forward validation, quality gates, artifact packaging, and multi-seed ensemble training. This is the single entry point for the entire pipeline. |
| `curriculum_trainer.py` | ~550 | **Phase 1 driver.** Implements progressive difficulty training across 4 stages (BASE -> ENRICHED -> SOFT -> PRODUCTION). Each stage introduces more agent constraints so the model gradually learns to work with the full production environment. |
| `ensemble_trainer.py` | ~670 | **Phase 2 driver.** Trains 5-7 diverse specialist models with different hyperparameters, then combines them into an ensemble with adaptive voting weights. Correlation penalties ensure model diversity. |
| `meta_learner.py` | ~500 | **Phase 3 driver.** Uses MAML-inspired meta-learning to find model parameters that can adapt quickly to market regime changes. Includes a `RegimeDetector` (6 regimes) and `OnlineAdapter` for live deployment. |
| `unified_agentic_env.py` | ~400 | **Environment wrapper.** Wraps the base `TradingEnv` to provide a constant 323-dimensional observation space across all training modes. This solves the critical domain shift problem between training and production. |
| `advanced_reward_shaper.py` | ~400 | **Reward calculator.** Computes multi-objective rewards from 10 components (profit, Sharpe, Sortino, Calmar, win rate, profit factor, risk-reward, exploration, timing, regime adaptation) with phase-adaptive weights. |
| `checkpoint_manager.py` | ~300 | **Checkpoint I/O.** Saves model checkpoints to local disk and Google Drive with SHA-256 integrity verification. Enables resume-from-checkpoint when Colab disconnects. |

### Environment Layer (`src/environment/`)

| File | Role |
|------|------|
| `environment.py` | **Base trading environment.** Gymnasium environment with 5-action discrete space (HOLD, OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT). Manages positions, PnL tracking, stop-loss/take-profit, transaction costs. Provides the raw 303-dimensional observation space. |
| `strategy_features.py` | **Feature engine.** Computes Smart Money Concept (SMC) features: RSI, MACD, Bollinger Bands, ATR, Fair Value Gaps, Break of Structure, Change of Character, Order Block strength. Uses Numba JIT for 50-100x speedup. Produces the 15 features per bar that form the observation space. |
| `risk_manager.py` | **Position sizing and risk.** Dynamic risk management with async GARCH(1,1) volatility modeling, Kelly criterion position sizing, regime-scaled stop-losses, and Value-at-Risk calculations. Called by the environment on every trade. |
| `feature_reducer.py` | **Dimensionality reduction.** IncrementalPCA that reduces the 303-dim observation space to ~60-80 dimensions while retaining 95% of variance. The fitted PCA transformer is saved alongside the model for production use. |

### Configuration

| File | Role |
|------|------|
| `config.py` | **Central config.** Contains ALL hyperparameters, risk limits, feature definitions, walk-forward settings, quality gates, ensemble seeds, and environment parameters. Every training module reads from this file. See [Section 11](#11-configuration-reference) for details. |

### Entry Point

| File | Role |
|------|------|
| `notebooks/Colab_Full_Training_Script.py` | **Colab driver.** Thin script that mounts Google Drive, clones the repo, downloads data, configures `SophisticatedTrainer`, runs the full pipeline, and saves results. This is what you run in Google Colab. |

---

## 4. Phase 1: Curriculum Learning

**File:** `src/training/curriculum_trainer.py`

### Why Curriculum Learning?

The production trading system uses multiple AI agents (news filter, risk sentinel, market regime detector, orchestrator) that modify the bot's actions. If you train the bot without these agents and then deploy it with them, the bot sees a completely different world — this is called **domain shift**, and it causes 30-60% of the bot's actions to be rejected in production.

Curriculum learning solves this by **gradually introducing the agents** across 4 stages:

### The 4 Stages

| Stage | Steps | Environment Mode | What the Bot Sees |
|-------|-------|-----------------|-------------------|
| **BASE** | 0 — 300K | Agent signals = 0 | Pure market data only. The bot learns basic trading: when to buy, when to sell, how to manage risk. |
| **ENRICHED** | 300K — 700K | Agent signals visible | The bot now sees agent recommendations (news impact, risk score, regime type) as extra observations, but agents don't constrain actions yet. |
| **SOFT** | 700K — 1.1M | Soft penalties | Agents start penalizing bad actions (e.g., trading during high-impact news gets a reward penalty) but don't block them. |
| **PRODUCTION** | 1.1M — 1.5M | Hard constraints | Full production mode. Agents can reject actions outright (e.g., risk sentinel blocks oversized positions). |

### Entropy Annealing

Alongside the curriculum, the **entropy coefficient** (which controls how much the bot explores vs. exploits) is annealed:

```
Step 0:        ent_coef = 0.050  (high exploration — try many strategies)
Step 100,000:  ent_coef = 0.020  (moderate)
Step 300,000:  ent_coef = 0.010  (standard)
Step 500,000:  ent_coef = 0.005  (exploit learned policy)
```

This is implemented by `EntropyAnnealingCallback` in `sophisticated_trainer.py`.

### Reward Weight Progression

The `AdvancedRewardShaper` adjusts reward weights per phase:

| Component | BASE | ENRICHED | SOFT | PRODUCTION |
|-----------|------|----------|------|------------|
| Profit | 1.0 | 1.0 | 1.0 | 1.0 |
| Sharpe | 0.2 | 0.4 | 0.6 | 0.8 |
| Exploration | 0.2 | 0.1 | 0.05 | 0.02 |
| Drawdown penalty | 0.0 | 0.3 | 0.6 | 1.0 |
| Risk-Reward | 0.1 | 0.3 | 0.5 | 0.5 |

Early training focuses on raw profit and exploration; late training focuses on risk-adjusted returns.

---

## 5. Phase 2: Ensemble Training

**File:** `src/training/ensemble_trainer.py`

### Why Ensemble Training?

A single model is brittle — it may overfit to specific market conditions. An **ensemble** of diverse models is more robust because different models capture different patterns.

### How It Works

The ensemble trainer creates 5-7 specialist models, each with intentionally different hyperparameters:

| Specialist | Learning Rate | Entropy | Gamma | Character |
|-----------|--------------|---------|-------|-----------|
| Conservative Explorer | 1e-5 | 0.10 | 0.990 | Cautious, wide exploration |
| Aggressive Learner | 5e-5 | 0.02 | 0.995 | Fast learning, narrow focus |
| Balanced Trader | 3e-5 | 0.05 | 0.990 | Middle ground |
| Long-Horizon | 2e-5 | 0.08 | 0.999 | Looks far ahead |
| Short-Horizon | 4e-5 | 0.03 | 0.980 | Reacts quickly |

### Ensemble Voting

When making predictions, the ensemble combines individual model votes:

- **VOTING:** Each model votes for an action, majority wins
- **WEIGHTED:** Votes weighted by each model's recent Sharpe ratio (exponential softmax)
- **SPECIALIST:** Route to the model that performs best in the current regime (trending/ranging/volatile)
- **MIXTURE:** Soft routing where observation features determine model weights

A **correlation penalty** ensures diversity: if two models agree on >70% of actions, the more correlated one gets downweighted.

---

## 6. Phase 3: Meta-Learning

**File:** `src/training/meta_learner.py`

### Why Meta-Learning?

Markets change regimes (trending -> ranging -> volatile -> calm). A model trained on trending data performs poorly when the market starts ranging. Meta-learning finds model parameters that can **adapt quickly** to new regimes with just a few gradient steps.

### Regime Detection

The `RegimeDetector` classifies market conditions into 6 types:

| Regime | Trigger |
|--------|---------|
| TRENDING_UP | trend_strength > +0.15 |
| TRENDING_DOWN | trend_strength < -0.15 |
| RANGING | |trend| < 0.075, low volatility |
| VOLATILE | volatility > 0.3 |
| CALM | volatility < 0.1, |trend| < 0.15 |
| BREAKOUT | volatile + high momentum |

Detection uses: linear regression slope (trend), return standard deviation (volatility), rate of change (momentum), and variance ratio (Hurst exponent approximation).

### MAML-Style Training

The meta-learning process:

1. **Segment** training data by detected regime
2. **Create tasks**: Each task is a (support_set, query_set) pair from one regime
3. **Inner loop (adaptation)**: Take 5 gradient steps on the support set
4. **Outer loop (meta-update)**: Evaluate on query set, update initial parameters

This teaches the model: "start from parameters that, with just 5 gradient steps, work well for ANY regime."

### Online Adapter (for live trading)

The `OnlineAdapter` monitors regime changes during live trading:
- When regime confidence > 0.6, perform 5 gradient steps to adapt
- Also adapts every 100 steps regardless
- Stores regime transition history for analysis

---

## 7. Phase 4: Validation & Packaging

**File:** `src/training/sophisticated_trainer.py` (methods: `run_walk_forward`, `check_quality_gates`, `package_production_artifact`)

### Walk-Forward Validation

Standard train/test splits can leak future information. Walk-forward validation uses **rolling windows** with **purge gaps**:

```
|-------- Train (6720 bars) --------|--Purge (96)--|-- Test (1120) --|
                                    |-------- Train (6720 bars) --------|--Purge--|-- Test --|
                                                                        |--- Train ---|...
```

- **Train window:** 6720 bars (~6 months)
- **Test window:** 1120 bars (~1 month)
- **Purge gap:** 96 bars (1 day) to prevent look-ahead bias
- **Step size:** 1120 bars (slide forward 1 month per fold)
- **Max folds:** 12

Each fold trains a fresh lightweight model and evaluates on the test set. The aggregate statistics (mean Sharpe, std, worst case) tell you how the strategy performs across different market periods.

### Quality Gates

Before a model can be promoted to production, it must pass these thresholds:

| Gate | Threshold | Why |
|------|-----------|-----|
| Sharpe Ratio | >= 1.0 | Risk-adjusted return must justify trading costs |
| Max Drawdown | <= 15% | Capital preservation is paramount |
| Win Rate | >= 40% | Need minimum hit rate for psychological sustainability |
| Profit Factor | >= 1.3 | Gross profits must exceed gross losses by 30% |

If any gate fails, the artifact is still packaged but marked `quality_gate_passed: false`.

### Production Artifact

The final output is a self-contained directory:

```
production_artifact/
  model.zip                  -- PPO model weights (stable-baselines3 format)
  pca_transformer.pkl        -- Fitted PCA reducer (sklearn IncrementalPCA)
  config.json                -- Hyperparameters used for training
  training_metadata.json     -- Final metrics, quality gate results, timestamps
  walk_forward_results.json  -- Per-fold and aggregate walk-forward statistics
  manifest.json              -- SHA-256 hash of every file (integrity check)
```

The SHA-256 manifest lets you verify that no file was corrupted or tampered with after packaging.

---

## 8. Environment & Observation Space

**Files:** `src/environment/environment.py`, `src/training/unified_agentic_env.py`

### Action Space (5 discrete actions)

```
0 = HOLD          -- Do nothing
1 = OPEN_LONG     -- Buy (profit when price goes UP)
2 = CLOSE_LONG    -- Exit long position
3 = OPEN_SHORT    -- Sell short (profit when price goes DOWN)
4 = CLOSE_SHORT   -- Exit short position
```

Invalid actions (e.g., OPEN_LONG when already long) are converted to HOLD.

### Observation Space (323 dimensions)

The bot sees a **flattened window** of the last 20 bars, each with 15 features, plus 3 state variables and 20 agent signals:

**Per-bar features (15):**

| # | Feature | Source | Purpose |
|---|---------|--------|---------|
| 1-5 | Open, High, Low, Close, Volume | Raw OHLCV | Price action |
| 6 | RSI (7-bar) | ta library | Momentum oscillator |
| 7 | MACD Diff | ta library | Trend direction |
| 8-9 | Bollinger Low/High | ta library | Volatility bands |
| 10 | ATR (7-bar) | ta library | Volatility measure |
| 11 | Spread (High-Low) | Computed | Intraday volatility |
| 12 | FVG Signal | SMC engine | Fair Value Gap (institutional footprint) |
| 13 | BOS Signal | SMC engine | Break of Structure |
| 14 | CHOCH Signal | SMC engine | Change of Character (reversals) |
| 15 | OB Strength Norm | SMC engine | Order Block strength |

**Total:** 20 bars x 15 features = 300 dimensions

**State variables (3):** Current position (-1/0/1), normalized PnL, current drawdown

**Agent signals (20):** News sentiment, risk score, regime confidence, orchestrator suggestions, etc.

**Grand total:** 300 + 3 + 20 = **323 dimensions**

### Why UnifiedAgenticEnv?

The original system had separate environments for training (`TradingEnv`, 303 dims) and production (`OrchestratedTradingEnv`, different dynamics). This meant the bot trained in one world and deployed in another — causing massive performance degradation.

`UnifiedAgenticEnv` fixes this by keeping the observation space **constant at 323 dims** across all training modes. In BASE mode, the 20 agent signal dims are zero-filled. In PRODUCTION mode, they contain real agent outputs. The bot always sees the same shape.

---

## 9. Reward System

**File:** `src/training/advanced_reward_shaper.py`

The reward function has **10 components**, each weighted differently depending on the curriculum phase:

| Component | What It Measures | Weight Range |
|-----------|-----------------|--------------|
| **Profit** | Raw PnL as % of capital | Always 1.0 |
| **Sharpe** | Rolling risk-adjusted return (50-bar window) | 0.2 -> 0.8 |
| **Sortino** | Downside risk-adjusted return | 0 -> 0.4 |
| **Calmar** | Return / max drawdown | 0 -> 0.5 |
| **Win Rate** | Fraction of winning trades | 0 -> 0.3 |
| **Profit Factor** | Gross profit / gross loss | 0 -> 0.3 |
| **Risk-Reward** | Average win / average loss | 0.1 -> 0.5 |
| **Exploration** | Entropy bonus for diverse actions | 0.2 -> 0.02 |
| **Timing** | Quality of entry/exit relative to regime | 0 -> 0.3 |
| **Regime Adaptation** | Alignment with detected market regime | 0 -> 0.2 |

Additional penalties:
- **Drawdown penalty:** Progressive penalty as drawdown approaches the limit (not just at end of episode)
- **Friction penalty:** Transaction costs reduce reward to discourage churning
- **Turnover penalty:** Excessive trading is penalized
- **Risk rejection penalty:** When an agent rejects an action, the bot receives a penalty (fixes credit assignment bug)

All rewards are passed through `tanh` squashing (scale=0.3) and output scaling (5.0x) to keep them in a PPO-friendly range.

---

## 10. Risk Management Layer

**File:** `src/environment/risk_manager.py`

### Position Sizing

- **Kelly criterion** with regime scaling:
  - Calm market: 1.0x Kelly
  - Volatile market: 0.5x Kelly
- **Max risk per trade:** 1% of capital
- **Max leverage:** 1.0x (no leverage)

### Stop-Loss / Take-Profit

- **Stop-loss:** 1% (ATR-based adjustment)
- **Take-profit:** 2% (2:1 risk-reward ratio)
- **Trailing stop:** Activates at 1x ATR profit, trails at 0.5x ATR

### Volatility Modeling

- **GARCH(1,1):** Refitted every 500 steps (~expensive, 200-400ms)
- **EWMA fallback:** Between refits, fast approximation (<0.01ms)
- **Async execution:** GARCH runs in a background thread, never blocks trading

### Value at Risk

- **Method:** Cornish-Fisher (adjusts for skew and kurtosis)
- **Confidence:** 95%
- **Lookback:** 252 trading days
- **Limit:** 2% portfolio VaR (triggers kill switch if exceeded)

---

## 11. Configuration Reference

**File:** `config.py`

### Key Training Parameters

```python
TOTAL_TIMESTEPS_PER_BOT = 1_500_000   # Total training steps
EARLY_STOPPING_PATIENCE = 5            # Stop after 5 evals without improvement
EVAL_FREQ = 10_000                     # Evaluate every 10K steps
N_PARALLEL_BOTS = 50                   # Bots in hyperparameter search
```

### PPO Hyperparameters

```python
MODEL_HYPERPARAMETERS = {
    "n_steps": 1024,        # Rollout length (~2x episode length)
    "batch_size": 128,      # Minibatch size
    "gamma": 0.995,         # Discount factor (horizon ~200 steps)
    "learning_rate": 3e-4,  # Standard PPO (Schulman 2017)
    "ent_coef": 0.01,       # Entropy coefficient (annealed during training)
    "clip_range": 0.2,      # PPO clip range
    "gae_lambda": 0.95,     # GAE advantage estimation
    "max_grad_norm": 0.5,   # Gradient clipping
    "vf_coef": 0.5,         # Value function loss weight
    "n_epochs": 5,          # Optimization epochs per rollout
}
```

### Walk-Forward Validation

```python
WALK_FORWARD_CONFIG = {
    'train_window_bars': 6720,       # ~6 months
    'validation_window_bars': 2240,  # ~2 months
    'test_window_bars': 1120,        # ~1 month
    'step_size_bars': 1120,          # Slide 1 month per fold
    'purge_gap_bars': 96,            # 1-day gap (prevent look-ahead)
    'min_folds': 3,
    'max_folds': 12,
    'strategy': 'rolling',
}
```

### Quality Gates

```python
QUALITY_GATES = {
    'min_sharpe': 1.0,
    'max_drawdown': 0.15,
    'min_win_rate': 0.40,
    'min_profit_factor': 1.3,
}
```

### Ensemble Seeds

```python
ENSEMBLE_SEEDS = (42, 123, 456)
```

---

## 12. Data Flow Diagram

```
                        config.py
                           |
                    (hyperparameters)
                           |
    XAU_15MIN_2019_2024.csv
            |
     [Data Loading]
            |
     70% train / 15% val / 15% test
            |
            v
  +--------------------+
  | strategy_features  |  <-- Numba-JIT SMC engine
  | (15 features/bar)  |      RSI, MACD, BB, ATR, FVG, BOS, CHOCH, OB
  +--------------------+
            |
            v
  +--------------------+
  |   TradingEnv       |  <-- Base Gymnasium environment
  | (303 dims + risk)  |      5 actions, position management, PnL tracking
  +--------------------+
            |
            v
  +--------------------+
  | UnifiedAgenticEnv  |  <-- Constant 323-dim wrapper
  | (+20 agent signals)|      Modes: BASE -> ENRICHED -> SOFT -> PRODUCTION
  +--------------------+
            |
            v
  +------------------------------------+
  |      SophisticatedTrainer          |
  |  (Master Orchestrator)             |
  |                                    |
  |  Phase 1: CurriculumTrainer       |  <-- 4-stage progressive training
  |    + AdvancedRewardShaper          |      + 10-objective reward shaping
  |    + EntropyAnnealingCallback      |      + entropy decay
  |    + CheckpointManager             |      + Colab-safe checkpoints
  |                                    |
  |  Phase 2: EnsembleTrainer          |  <-- 5-7 diverse specialists
  |    + Adaptive weight calibration   |      + correlation penalty
  |                                    |
  |  Phase 3: MetaLearner             |  <-- MAML-style regime adaptation
  |    + RegimeDetector (6 regimes)    |      + OnlineAdapter for live use
  |                                    |
  |  Phase 4: Validation              |
  |    + Walk-Forward (rolling folds)  |
  |    + Quality Gates (4 thresholds)  |
  |    + Artifact Packaging (SHA-256)  |
  +------------------------------------+
            |
            v
  production_artifact/
    model.zip
    pca_transformer.pkl
    config.json
    training_metadata.json
    walk_forward_results.json
    manifest.json
```

---

## Summary

Your trading bot's training system is a **4-pclhase pipeline** that:

1. **Teaches basic trading** through curriculum learning (gradually introducing production constraints)
2. **Creates diversity** through ensemble training (multiple specialists with different strategies)
3. **Enables adaptation** through meta-learning (fast adjustment to market regime changes)
4. **Validates rigorously** through walk-forward testing and quality gates before deployment

The key innovation is the **UnifiedAgenticEnv** which maintains a constant observation space across all training phases, solving the domain shift problem that plagues most RL trading systems. Combined with the multi-objective reward shaper, entropy annealing, and checkpoint resilience, this creates a production-grade training system.
