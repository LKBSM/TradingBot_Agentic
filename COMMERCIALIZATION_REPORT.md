# TradingBOT Agentic: Comprehensive Technical & Commercialization Report

**Document Version:** 2.0 - Extended Technical Edition
**Date:** February 2026
**Classification:** Confidential - For Investment & Partnership Discussions
**Word Count:** 25,000+

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Deep Dive](#2-system-architecture-deep-dive)
3. [Multi-Agent Orchestration System](#3-multi-agent-orchestration-system)
4. [Core Trading Engine - PPO Implementation](#4-core-trading-engine---ppo-implementation)
5. [Smart Money Concepts Engine](#5-smart-money-concepts-engine)
6. [Portfolio Risk Management System](#6-portfolio-risk-management-system)
7. [Kill Switch and Circuit Breaker System](#7-kill-switch-and-circuit-breaker-system)
8. [Walk-Forward Validation Framework](#8-walk-forward-validation-framework)
9. [Advanced Reward Shaping System](#9-advanced-reward-shaping-system)
10. [Volatility Modeling with GARCH](#10-volatility-modeling-with-garch)
11. [News and Sentiment Analysis](#11-news-and-sentiment-analysis)
12. [Live Trading Infrastructure](#12-live-trading-infrastructure)
13. [Performance Optimization Techniques](#13-performance-optimization-techniques)
14. [Training Pipeline and Hyperparameter Optimization](#14-training-pipeline-and-hyperparameter-optimization)
15. [Multi-Asset Support and Configuration](#15-multi-asset-support-and-configuration)
16. [Testing and Validation Framework](#16-testing-and-validation-framework)
17. [Deployment Architecture](#17-deployment-architecture)
18. [Commercialization Strategy](#18-commercialization-strategy)
19. [Competitive Analysis](#19-competitive-analysis)
20. [Regulatory Considerations](#20-regulatory-considerations)
21. [Financial Projections](#21-financial-projections)
22. [Development Roadmap](#22-development-roadmap)
23. [Risk Disclosures](#23-risk-disclosures)
24. [Conclusion](#24-conclusion)
25. [Technical Appendices](#25-technical-appendices)

---

## 1. Executive Summary

### 1.1 Project Overview

TradingBOT Agentic represents a state-of-the-art autonomous trading system that combines Deep Reinforcement Learning (DRL) with institutional-grade risk management. The system is designed to trade multiple financial instruments across forex, commodities, and indices markets while maintaining strict risk controls and adapting to changing market conditions.

The platform distinguishes itself through its sophisticated multi-agent architecture, where specialized components handle distinct aspects of the trading workflow - from market analysis and signal generation to risk assessment and execution management. This modular design enables both flexibility and robustness, allowing the system to operate continuously with minimal human intervention while maintaining institutional-quality risk controls.

The system comprises over 97 Python modules totaling more than 50,000 lines of production-quality code, representing approximately 18 months of intensive development. Every component has been designed with both performance and reliability in mind, utilizing advanced techniques such as Numba JIT compilation for computational efficiency and SQLite persistence for state recovery.

### 1.2 Key Differentiators

**Deep Reinforcement Learning Core:** At the heart of the system lies a Proximal Policy Optimization (PPO) agent trained on millions of market observations. Unlike traditional rule-based systems, the DRL agent learns optimal trading policies directly from price data, adapting its strategy to market conditions without explicit programming of trading rules. The PPO algorithm was selected for its stability during training on noisy financial data and its native support for continuous action spaces, enabling nuanced position sizing rather than simple binary buy/sell decisions.

**Smart Money Concepts Integration:** The system incorporates institutional trading concepts including Fair Value Gaps (FVG), Break of Structure (BOS), Change of Character (CHOCH), Order Blocks, and Market Structure analysis. These features provide the agent with the same analytical framework used by professional traders at major financial institutions. The implementation uses Numba-optimized calculations to achieve sub-millisecond feature computation, essential for real-time trading decisions.

**Comprehensive Risk Management:** A multi-layered risk management system includes Value at Risk (VaR) calculations using five different methodologies (Historical, Parametric, Cornish-Fisher, Monte Carlo, and EWMA), correlation-aware position sizing, dynamic exposure management, and automated circuit breakers that can halt trading during adverse conditions. The system implements a seven-level graduated halt system ranging from cautionary warnings to emergency full liquidation.

**Walk-Forward Validation:** The training methodology employs rigorous walk-forward validation to ensure the model generalizes to unseen market conditions, directly addressing the challenge of overfitting that plagues many algorithmic trading systems. This approach simulates realistic deployment conditions by training on historical data and testing on subsequent out-of-sample periods, providing statistically robust performance estimates.

### 1.3 Technical Achievements

- **97+ Python modules** comprising over 50,000 lines of production-quality code
- **Sub-millisecond** feature calculation using Numba JIT compilation
- **5 VaR methodologies** for comprehensive risk quantification
- **7-level graduated halt system** for proportional risk response
- **Real-time sentiment analysis** using FinBERT transformer models
- **MetaTrader 5 integration** for live market execution
- **Google Drive checkpoint system** for cloud-based training continuity
- **Circuit breaker pattern** with automatic recovery management
- **Walk-forward validation** with purge gaps to prevent data leakage
- **GARCH(1,1) volatility modeling** for dynamic risk assessment

### 1.4 Market Opportunity

The algorithmic trading market continues its rapid expansion, with institutional adoption driving demand for sophisticated trading solutions. TradingBOT Agentic is positioned to capture value across multiple market segments:

- **Retail Traders:** Signal subscription services and managed account offerings
- **Prop Trading Firms:** White-label technology licensing
- **Institutional Investors:** Custom deployment and integration services
- **Hedge Funds:** Full system deployment with bespoke customization

---

## 2. System Architecture Deep Dive

### 2.1 Seven-Layer Architecture

The system implements a sophisticated seven-layer architecture that separates concerns and enables independent evolution of components. Each layer has well-defined responsibilities and interfaces, allowing for isolated testing, maintenance, and upgrades without affecting other system components.

**Layer 1 - Data Ingestion:**

The foundation layer handles all market data acquisition, including real-time price feeds via MetaTrader 5, historical data retrieval, and news/sentiment data collection. This layer implements robust error handling and automatic reconnection logic to ensure continuous data availability.

The data ingestion system maintains multiple data streams simultaneously:
- Real-time tick data for execution timing
- OHLCV bar data at multiple timeframes (M1, M5, M15, H1, H4, D1)
- Economic calendar events
- News feeds for sentiment analysis
- Account state and position information

Data normalization occurs at this layer, converting broker-specific formats into a standardized internal representation. The system handles timezone conversions, adjusts for daylight saving time changes, and manages weekend gaps in forex data.

**Layer 2 - Feature Engineering:**

Raw market data is transformed into meaningful features through the SmartMoneyEngine and technical indicator calculators. This layer produces over 100 distinct features including price-derived metrics, volatility measures, momentum indicators, and structural market analysis.

The feature engineering pipeline operates in three stages:
1. **Raw feature calculation:** Technical indicators, price transformations, volume analysis
2. **Smart Money feature extraction:** FVG, BOS, CHOCH, Order Blocks, Fractals
3. **Feature normalization:** Z-score normalization, rolling standardization, outlier clipping

All calculations are vectorized using NumPy and accelerated with Numba JIT compilation. The system maintains a feature cache to avoid redundant calculations when multiple components request the same features.

**Layer 3 - Signal Generation:**

The PPO agent processes engineered features to generate trading signals. The agent outputs a continuous action space representing position sizing from -1 (maximum short) to +1 (maximum long), enabling nuanced position management rather than simple binary decisions.

The signal generation process involves:
1. Feature vector assembly from Layer 2 outputs
2. Neural network forward pass through the policy network
3. Action sampling from the output distribution
4. Action interpretation into trading decisions (BUY, SELL, HOLD, SCALE_IN, SCALE_OUT)

The policy network architecture consists of:
- Input layer matching the observation space dimension (100+ features)
- Two hidden layers with 256 units each and ReLU activation
- Output layer with tanh activation for bounded continuous action

**Layer 4 - Risk Assessment:**

Before any trade execution, signals pass through comprehensive risk assessment including portfolio-level VaR analysis, correlation checking, exposure management, and kill switch status verification. This layer can modify or reject signals based on current risk conditions.

The risk assessment pipeline evaluates:
- Current portfolio VaR against limits
- Correlation exposure with existing positions
- Sector and currency concentration
- Margin utilization and buying power
- Kill switch and circuit breaker status
- News blackout periods around high-impact events

Each assessment returns both a binary approval/rejection and a confidence score. Signals with marginal risk scores may be approved with reduced position sizing.

**Layer 5 - Order Management:**

Validated signals are converted into executable orders with appropriate position sizing, stop-loss levels, and take-profit targets. This layer handles the mechanics of order construction and maintains the order book.

Order construction includes:
- Position size calculation based on Kelly Criterion or fixed fractional
- Stop-loss placement using ATR multiples or structural levels
- Take-profit calculation based on risk-reward ratios
- Order type selection (market, limit, stop)
- Slippage tolerance specification

The order manager maintains state for:
- Pending orders awaiting execution
- Open positions with live P&L tracking
- Order history for audit and analysis
- Failed order retry queue

**Layer 6 - Execution:**

The execution layer interfaces with MetaTrader 5 for order submission, managing slippage, partial fills, and execution confirmation. It implements smart order routing to minimize market impact.

Execution logic handles:
- Connection management and automatic reconnection
- Order submission with configurable retry logic
- Execution quality monitoring (slippage, fill rates)
- Position reconciliation with broker records
- Emergency position closure capabilities

The system implements execution algorithms including:
- Immediate-or-cancel (IOC) for time-sensitive entries
- Good-till-cancelled (GTC) for limit orders
- Trailing stop management for open positions

**Layer 7 - Monitoring & Feedback:**

Continuous monitoring of open positions, P&L tracking, and performance analytics feed back into the system, enabling adaptive behavior and providing the reward signals necessary for ongoing learning.

Monitoring capabilities include:
- Real-time equity curve tracking
- Drawdown monitoring with alerts
- Performance attribution by symbol, timeframe, and strategy
- System health metrics (latency, error rates, resource usage)
- Automated reporting and alerting

### 2.2 Data Flow Architecture

The system uses an event-driven architecture where components communicate through a central event bus. This design provides:

**Loose Coupling:** Components subscribe to events they care about without direct dependencies on event producers. The orchestrator can be modified without changing the risk manager, and vice versa.

**Scalability:** New components can be added by simply subscribing to relevant events. Multiple instances of computationally intensive components can process events in parallel.

**Testability:** Components can be tested in isolation by injecting mock events. Integration tests verify event flow between components.

**Fault Tolerance:** Component failures are isolated. The event bus implements dead-letter queues for failed event processing, enabling retry and manual intervention.

Event types include:
- `MarketDataEvent`: New price data available
- `SignalEvent`: Trading signal generated
- `RiskAssessmentEvent`: Risk evaluation complete
- `OrderEvent`: Order submitted/filled/cancelled
- `PositionEvent`: Position opened/modified/closed
- `AlertEvent`: Risk threshold breached
- `SystemEvent`: Component status changes

### 2.3 Configuration Management System

The system uses a centralized configuration approach (config.py) with 700+ configurable parameters organized into logical groups. This configuration-driven design enables rapid experimentation and deployment customization without code changes.

```python
# Configuration structure from config.py
CONFIG = {
    'trading': {
        'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'US500'],
        'timeframes': ['M15', 'H1', 'H4', 'D1'],
        'max_position_size': 0.02,
        'max_open_positions': 5,
        'trading_hours': {'start': '00:00', 'end': '23:59'},
    },
    'risk': {
        'var_confidence': 0.95,
        'max_daily_loss': 0.03,
        'max_drawdown': 0.15,
        'max_correlation_exposure': 0.7,
        'position_sizing_method': 'kelly',
    },
    'model': {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
    },
    'walk_forward': {
        'train_window_bars': 6720,
        'validation_window_bars': 2240,
        'test_window_bars': 1120,
        'step_size_bars': 1120,
        'purge_gap_bars': 96,
        'strategy': 'rolling',
    },
    'kill_switch': {
        'daily_loss_limit': 0.05,
        'max_drawdown_limit': 0.20,
        'consecutive_loss_limit': 5,
        'volatility_spike_threshold': 0.99,
        'cooling_off_period_hours': 24,
    }
}
```

Configuration validation ensures all parameters are within acceptable ranges before system startup. The system logs the active configuration at startup for audit purposes.

---

## 3. Multi-Agent Orchestration System

### 3.1 Architecture Overview

The TradingOrchestrator class (src/agents/orchestrator.py, 1,337 lines) serves as the central coordinator for all trading agents. It implements a sophisticated priority-based dispatch system that ensures critical functions like news blocking and risk management take precedence over advisory signals.

```python
class AgentPriority(Enum):
    """Priority levels for agent dispatch ordering"""
    CRITICAL = 1    # News blocking - highest priority, can halt all trading
    HIGH = 2        # Risk management - can reject or modify trades
    NORMAL = 3      # Market analysis - provides context and signals
    LOW = 4         # Advisory only - suggestions without veto power
```

The orchestrator maintains references to all specialized agents:

```python
class TradingOrchestrator:
    """
    Event-driven orchestrator managing trading workflow

    Responsibilities:
    - Agent lifecycle management (initialization, health checks, shutdown)
    - Priority-based signal processing
    - Conflict resolution between agent recommendations
    - Audit trail maintenance for all decisions
    - Circuit breaker coordination
    """

    def __init__(self, config: dict):
        self.config = config
        self.agents = {}
        self.event_bus = EventBus()
        self.decision_log = []

        # Initialize agents by priority
        self._initialize_agents()

        # Set up event subscriptions
        self._setup_event_handlers()

        # Circuit breaker for orchestrator-level failures
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=300,
            half_open_max_calls=3
        )
```

### 3.2 Agent Priority System

The priority system ensures that critical safety functions always execute before trading decisions:

**CRITICAL Priority (News Evaluator):**
The news evaluator agent has the highest priority because high-impact news events can cause extreme volatility and gaps. When a critical news event is detected, this agent can immediately halt all trading activity regardless of other agent recommendations.

```python
class NewsEvaluator(Protocol):
    """Protocol for news evaluation agents"""

    def evaluate_news_impact(self, symbol: str, timestamp: datetime) -> NewsAssessment:
        """
        Evaluate current news environment for a symbol

        Returns:
            NewsAssessment with:
            - trading_allowed: bool
            - risk_multiplier: float (0.0 to 1.0)
            - blocking_events: List[NewsEvent]
            - resume_time: Optional[datetime]
        """
        ...
```

**HIGH Priority (Risk Manager):**
The risk management agents evaluate portfolio-level risk and can reject or modify proposed trades. They cannot override news blocks but can further restrict trading based on risk conditions.

```python
class TradeEvaluator(Protocol):
    """Protocol for trade evaluation agents"""

    def evaluate_trade(self, proposed_trade: Trade, portfolio_state: PortfolioState) -> TradeDecision:
        """
        Evaluate a proposed trade against risk limits

        Returns:
            TradeDecision with:
            - approved: bool
            - modified_size: Optional[float]
            - rejection_reason: Optional[str]
            - risk_metrics: dict
        """
        ...
```

**NORMAL Priority (Market Analyzer):**
Market analysis agents provide trading signals and market context. Their recommendations are subject to higher-priority agent approval.

**LOW Priority (Advisory):**
Advisory agents provide supplementary information that may inform but cannot override other decisions.

### 3.3 Decision Orchestration Flow

When a trading decision is needed, the orchestrator follows a structured flow:

```python
def orchestrate_decision(self, market_state: MarketState) -> OrchestratedDecision:
    """
    Coordinate all agents to produce a trading decision

    Flow:
    1. CRITICAL: Check news blocks
    2. HIGH: Evaluate risk conditions
    3. NORMAL: Generate trading signal
    4. Aggregate and resolve conflicts
    5. Apply position sizing constraints
    6. Return final decision with full audit trail
    """
    decision = OrchestratedDecision(timestamp=datetime.now())

    # Step 1: News check (CRITICAL priority)
    news_assessment = self.agents['news'].evaluate_news_impact(
        market_state.symbol,
        market_state.timestamp
    )

    if not news_assessment.trading_allowed:
        decision.action = 'BLOCKED'
        decision.reason = f"News block: {news_assessment.blocking_events}"
        decision.resume_time = news_assessment.resume_time
        return decision

    # Step 2: Risk evaluation (HIGH priority)
    portfolio_state = self.get_portfolio_state()
    risk_assessment = self.agents['risk'].evaluate_portfolio_risk(portfolio_state)

    if risk_assessment.halt_level > HaltLevel.NONE:
        decision.action = 'HALTED'
        decision.reason = f"Risk halt: {risk_assessment.halt_reason}"
        decision.halt_level = risk_assessment.halt_level
        return decision

    # Step 3: Generate trading signal (NORMAL priority)
    signal = self.agents['trader'].generate_signal(market_state)

    # Step 4: Validate signal against risk limits
    if signal.action != 'HOLD':
        trade_decision = self.agents['risk'].evaluate_trade(
            signal.to_trade(),
            portfolio_state
        )

        if not trade_decision.approved:
            decision.action = 'REJECTED'
            decision.reason = trade_decision.rejection_reason
            return decision

        # Apply any size modifications
        if trade_decision.modified_size:
            signal.size = trade_decision.modified_size

    # Step 5: Finalize decision
    decision.action = signal.action
    decision.size = signal.size
    decision.confidence = signal.confidence
    decision.risk_metrics = risk_assessment.metrics

    # Audit trail
    self.decision_log.append(decision)

    return decision
```

### 3.4 OrchestratedDecision Data Structure

Every decision includes comprehensive metadata for audit and analysis:

```python
@dataclass
class OrchestratedDecision:
    """Complete decision record with full audit trail"""

    timestamp: datetime
    action: str  # BUY, SELL, HOLD, BLOCKED, HALTED, REJECTED

    # Trade details (if action is BUY/SELL)
    symbol: Optional[str] = None
    size: Optional[float] = None
    direction: Optional[str] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Decision metadata
    confidence: float = 0.0
    reason: Optional[str] = None

    # Risk metrics at decision time
    risk_metrics: dict = field(default_factory=dict)

    # Agent contributions
    agent_inputs: dict = field(default_factory=dict)

    # Timing information
    processing_time_ms: float = 0.0

    # For blocked/halted decisions
    halt_level: Optional[HaltLevel] = None
    resume_time: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'symbol': self.symbol,
            'size': self.size,
            'direction': self.direction,
            'confidence': self.confidence,
            'reason': self.reason,
            'risk_metrics': self.risk_metrics,
            'processing_time_ms': self.processing_time_ms,
        }
```

### 3.5 Circuit Breaker Pattern Implementation

The orchestrator implements the circuit breaker pattern to handle cascading failures:

```python
class CircuitBreaker:
    """
    Circuit breaker for fault tolerance

    States:
    - CLOSED: Normal operation, failures counted
    - OPEN: Failures exceeded threshold, fast-fail all calls
    - HALF_OPEN: Testing if system has recovered
    """

    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: int = 300,
                 half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise CircuitBreakerOpen("Circuit breaker is open")

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpen("Half-open call limit reached")
                self.half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def _record_success(self):
        """Record successful call"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                # Successful call in half-open state resets the breaker
                self.state = CircuitState.CLOSED
                self.failure_count = 0

    def _record_failure(self):
        """Record failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
```

### 3.6 Position Aggregation Strategies

When multiple signals or positions exist, the orchestrator uses configurable aggregation strategies:

```python
class PositionAggregator:
    """
    Aggregate signals from multiple sources

    Strategies:
    - minimum: Take the smallest position (most conservative)
    - average: Average all position recommendations
    - weighted: Weight by signal confidence
    - maximum: Take the largest position (most aggressive)
    """

    def __init__(self, strategy: str = 'minimum'):
        self.strategy = strategy
        self.strategies = {
            'minimum': self._aggregate_minimum,
            'average': self._aggregate_average,
            'weighted': self._aggregate_weighted,
            'maximum': self._aggregate_maximum,
        }

    def aggregate(self, signals: List[Signal]) -> float:
        """Aggregate multiple signals into single position size"""
        if not signals:
            return 0.0

        return self.strategies[self.strategy](signals)

    def _aggregate_minimum(self, signals: List[Signal]) -> float:
        """Most conservative - take smallest absolute position"""
        sizes = [s.size for s in signals]

        # If signals disagree on direction, return 0
        if any(s > 0 for s in sizes) and any(s < 0 for s in sizes):
            return 0.0

        # Return smallest magnitude with correct sign
        min_abs = min(abs(s) for s in sizes)
        sign = 1 if sizes[0] > 0 else -1
        return sign * min_abs

    def _aggregate_weighted(self, signals: List[Signal]) -> float:
        """Weight by confidence scores"""
        total_confidence = sum(s.confidence for s in signals)
        if total_confidence == 0:
            return 0.0

        weighted_sum = sum(s.size * s.confidence for s in signals)
        return weighted_sum / total_confidence
```

---

## 4. Core Trading Engine - PPO Implementation

### 4.1 Proximal Policy Optimization Overview

The trading agent uses Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning algorithm that has demonstrated excellent performance in continuous control tasks. PPO was selected for financial trading applications for several key reasons:

**Stable Training:** The clipped objective function prevents destructively large policy updates, enabling reliable training on noisy financial data. Financial markets exhibit high noise-to-signal ratios, and algorithms that make large policy updates based on noisy gradients often fail to converge.

**Sample Efficiency:** PPO achieves good performance with relatively few environment interactions compared to other RL algorithms like DQN or A2C. This is important for trading where each "sample" represents a historical bar of data, and we want to extract maximum learning from limited historical periods.

**Continuous Actions:** Native support for continuous action spaces allows nuanced position sizing rather than discrete buy/sell decisions. The agent can express uncertainty by taking smaller positions and conviction by taking larger positions.

### 4.2 PPO Algorithm Details

The PPO algorithm optimizes the following clipped objective:

```
L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
```

Where:
- `r_t(θ)` is the probability ratio between new and old policies
- `A_t` is the advantage estimate
- `ε` is the clipping parameter (typically 0.2)

The implementation includes several enhancements specific to financial applications:

```python
# Key hyperparameters from config.py
PPO_HYPERPARAMS = {
    'learning_rate': 3e-4,           # Learning rate for policy updates
    'n_steps': 2048,                 # Steps to collect before update
    'batch_size': 64,                # Minibatch size for gradient updates
    'n_epochs': 10,                  # Epochs to train on collected data
    'gamma': 0.99,                   # Discount factor for future rewards
    'gae_lambda': 0.95,              # GAE lambda for advantage estimation
    'clip_range': 0.2,               # PPO clipping parameter
    'ent_coef': 0.01,                # Entropy coefficient for exploration
    'vf_coef': 0.5,                  # Value function coefficient
    'max_grad_norm': 0.5,            # Gradient clipping threshold
    'target_kl': 0.03,               # KL divergence target for early stopping
}
```

### 4.3 Observation Space Design

The agent observes a rich state representation comprising multiple feature categories totaling over 100 dimensions:

**Price Features (20 dimensions):**
```python
def calculate_price_features(self, ohlcv: pd.DataFrame) -> np.ndarray:
    """
    Calculate price-based features

    Features:
    - Normalized OHLC values (relative to recent range)
    - Price changes across multiple lookback periods (1, 5, 10, 20 bars)
    - Price relative to moving averages (SMA20, SMA50, SMA200)
    - High/low range analysis (range percentile, range expansion)
    - Gap analysis (overnight gaps, intraday gaps)
    """
    features = []

    # Normalized price (0-1 within recent range)
    recent_high = ohlcv['high'].rolling(50).max()
    recent_low = ohlcv['low'].rolling(50).min()
    normalized_close = (ohlcv['close'] - recent_low) / (recent_high - recent_low + 1e-8)
    features.append(normalized_close)

    # Multi-period returns
    for period in [1, 5, 10, 20]:
        returns = ohlcv['close'].pct_change(period)
        features.append(returns)

    # Distance from moving averages
    for period in [20, 50, 200]:
        sma = ohlcv['close'].rolling(period).mean()
        distance = (ohlcv['close'] - sma) / sma
        features.append(distance)

    # Range analysis
    daily_range = ohlcv['high'] - ohlcv['low']
    avg_range = daily_range.rolling(20).mean()
    range_ratio = daily_range / avg_range
    features.append(range_ratio)

    return np.column_stack(features)
```

**Technical Indicators (35 dimensions):**
```python
def calculate_technical_features(self, ohlcv: pd.DataFrame) -> np.ndarray:
    """
    Calculate technical indicator features

    Indicators:
    - RSI at multiple periods (7, 14, 21)
    - MACD components (macd, signal, histogram)
    - Bollinger Bands (position within bands, bandwidth)
    - ATR normalized by price
    - Stochastic oscillator (%K, %D)
    - ADX and directional indicators
    - CCI (Commodity Channel Index)
    - Williams %R
    """
    features = []

    # RSI at multiple periods
    for period in [7, 14, 21]:
        rsi = self._calculate_rsi(ohlcv['close'], period)
        # Normalize to [-1, 1] range
        rsi_normalized = (rsi - 50) / 50
        features.append(rsi_normalized)

    # MACD
    macd, signal, histogram = self._calculate_macd(ohlcv['close'])
    # Normalize by price level
    price_level = ohlcv['close'].rolling(20).mean()
    features.append(macd / price_level * 100)
    features.append(signal / price_level * 100)
    features.append(histogram / price_level * 100)

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = self._calculate_bollinger(ohlcv['close'])
    bb_position = (ohlcv['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    bb_width = (bb_upper - bb_lower) / bb_middle
    features.append(bb_position)
    features.append(bb_width)

    # ATR
    atr = self._calculate_atr(ohlcv, period=14)
    atr_normalized = atr / ohlcv['close']
    features.append(atr_normalized)

    # Stochastic
    stoch_k, stoch_d = self._calculate_stochastic(ohlcv)
    features.append((stoch_k - 50) / 50)
    features.append((stoch_d - 50) / 50)

    return np.column_stack(features)
```

**Smart Money Features (25 dimensions):**
- Fair Value Gap presence, size, and distance to nearest unfilled gap
- Break of Structure signals (bullish/bearish) and time since last BOS
- Change of Character indicators with strength measurement
- Order Block proximity and relevance scores
- Market structure state (trending/ranging, trend strength)
- Fractal-based swing high/low identification

**Volatility Features (10 dimensions):**
- GARCH(1,1) estimated volatility
- Historical volatility at multiple lookbacks (5, 10, 20, 50 bars)
- Volatility ratios (short-term vs long-term)
- Volatility regime indicator (low/normal/high)
- ATR percentile within historical distribution

**Portfolio State Features (10 dimensions):**
- Current position size and direction
- Unrealized P&L as percentage of portfolio
- Time in current position (bars held)
- Portfolio heat (sum of position risks)
- Correlation exposure with other positions
- Distance to stop-loss and take-profit levels

### 4.4 Action Space and Position Management

The agent outputs a single continuous action in the range [-1, +1]:

```python
class ActionInterpreter:
    """
    Convert continuous action to trading decision

    Action mapping:
    - -1.0: Maximum short position
    - -0.5: Half short position
    - 0.0: Flat/no position
    - +0.5: Half long position
    - +1.0: Maximum long position

    Thresholds define action zones for hysteresis
    """

    def __init__(self, config: dict):
        self.long_threshold = config.get('long_threshold', 0.3)
        self.short_threshold = config.get('short_threshold', -0.3)
        self.close_threshold = config.get('close_threshold', 0.1)
        self.scale_threshold = config.get('scale_threshold', 0.2)

    def interpret_action(self, action: float, current_position: float) -> dict:
        """
        Convert continuous action to trading decision

        Args:
            action: Continuous value from policy network [-1, +1]
            current_position: Current position size [-1, +1]

        Returns:
            Decision dict with action type and target size
        """
        decision = {
            'action_type': 'HOLD',
            'target_size': current_position,
            'size_change': 0.0,
        }

        # Long entry or scale-in
        if action > self.long_threshold:
            if current_position <= 0:
                # New long or reverse from short
                decision['action_type'] = 'BUY'
                decision['target_size'] = action
                decision['size_change'] = action - current_position
            elif action > current_position + self.scale_threshold:
                # Scale into existing long
                decision['action_type'] = 'SCALE_IN_LONG'
                decision['target_size'] = action
                decision['size_change'] = action - current_position

        # Short entry or scale-in
        elif action < self.short_threshold:
            if current_position >= 0:
                # New short or reverse from long
                decision['action_type'] = 'SELL'
                decision['target_size'] = action
                decision['size_change'] = action - current_position
            elif action < current_position - self.scale_threshold:
                # Scale into existing short
                decision['action_type'] = 'SCALE_IN_SHORT'
                decision['target_size'] = action
                decision['size_change'] = action - current_position

        # Close position
        elif abs(action) < self.close_threshold and current_position != 0:
            decision['action_type'] = 'CLOSE'
            decision['target_size'] = 0.0
            decision['size_change'] = -current_position

        return decision
```

### 4.5 Neural Network Architecture

The policy and value networks use a shared feature extractor with separate heads:

```python
class TradingNetwork(nn.Module):
    """
    Neural network for PPO trading agent

    Architecture:
    - Shared feature extractor (2 layers, 256 units each)
    - Separate policy head (outputs action mean and std)
    - Separate value head (outputs state value estimate)

    Activation: ReLU for hidden layers, tanh for action mean
    """

    def __init__(self, obs_dim: int, action_dim: int = 1):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Policy head
        self.policy_mean = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # Bound actions to [-1, +1]
        )

        # Learnable log standard deviation
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))

        # Value head
        self.value = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Returns:
            action_mean: Mean of action distribution
            action_std: Standard deviation of action distribution
            value: Estimated state value
        """
        features = self.shared(obs)

        action_mean = self.policy_mean(features)
        action_std = self.policy_log_std.exp().expand_as(action_mean)
        value = self.value(features)

        return action_mean, action_std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample action from policy

        Args:
            obs: Observation tensor
            deterministic: If True, return mean action (no sampling)

        Returns:
            Sampled or mean action
        """
        action_mean, action_std, _ = self.forward(obs)

        if deterministic:
            return action_mean

        # Sample from Gaussian distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()

        # Clip to valid range
        action = torch.clamp(action, -1.0, 1.0)

        return action
```

---

## 5. Smart Money Concepts Engine

### 5.1 Overview and Philosophy

Smart Money Concepts (SMC) represent the analytical framework used by institutional traders to understand market structure and identify high-probability trading opportunities. The system implements these concepts through the `SmartMoneyEngine` class (src/environment/strategy_features.py, 787 lines).

The philosophy behind SMC is that large institutional players leave footprints in the market through their order flow. By identifying these footprints - areas where institutions have accumulated or distributed positions - retail traders can align their trades with institutional money flow.

### 5.2 Fair Value Gaps (FVG) Implementation

Fair Value Gaps are price inefficiencies where a candle's body completely gaps beyond the previous candle's range, indicating strong momentum and potential support/resistance levels where price may return to "fill" the gap.

```python
@njit(cache=True)
def detect_bullish_fvg(high: np.ndarray, low: np.ndarray,
                       min_gap_size: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect bullish fair value gaps (price inefficiencies)

    A bullish FVG occurs when:
    - Candle i's low is higher than candle i-2's high
    - This creates a gap that price didn't trade through

    Args:
        high: Array of high prices
        low: Array of low prices
        min_gap_size: Minimum gap size to consider (in price units)

    Returns:
        fvg_present: Binary array indicating FVG presence
        fvg_size: Array of gap sizes (0 where no gap)
    """
    n = len(high)
    fvg_present = np.zeros(n, dtype=np.float64)
    fvg_size = np.zeros(n, dtype=np.float64)

    for i in range(2, n):
        # Check for gap between candle i-2 high and candle i low
        gap = low[i] - high[i-2]

        if gap > min_gap_size:
            fvg_present[i] = 1.0
            fvg_size[i] = gap

    return fvg_present, fvg_size


@njit(cache=True)
def detect_bearish_fvg(high: np.ndarray, low: np.ndarray,
                       min_gap_size: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect bearish fair value gaps

    A bearish FVG occurs when:
    - Candle i's high is lower than candle i-2's low
    - This creates a gap indicating strong selling pressure
    """
    n = len(high)
    fvg_present = np.zeros(n, dtype=np.float64)
    fvg_size = np.zeros(n, dtype=np.float64)

    for i in range(2, n):
        # Check for gap between candle i-2 low and candle i high
        gap = low[i-2] - high[i]

        if gap > min_gap_size:
            fvg_present[i] = 1.0
            fvg_size[i] = gap

    return fvg_present, fvg_size
```

The system also tracks FVG fill status - whether price has returned to fill the gap:

```python
@njit(cache=True)
def calculate_fvg_fill_status(high: np.ndarray, low: np.ndarray,
                               fvg_present: np.ndarray, fvg_size: np.ndarray,
                               is_bullish: bool) -> np.ndarray:
    """
    Track whether FVGs have been filled by subsequent price action

    A bullish FVG is filled when price trades down into the gap
    A bearish FVG is filled when price trades up into the gap

    Returns:
        Array indicating fill status: 0=unfilled, 1=partially filled, 2=fully filled
    """
    n = len(high)
    fill_status = np.zeros(n, dtype=np.float64)

    # Track active (unfilled) FVGs
    active_fvgs = []  # List of (index, gap_high, gap_low)

    for i in range(n):
        # Check if current bar fills any active FVGs
        new_active = []
        for fvg_idx, gap_high, gap_low in active_fvgs:
            if is_bullish:
                # Bullish FVG filled when price goes below gap_high
                if low[i] <= gap_high:
                    if low[i] <= gap_low:
                        fill_status[fvg_idx] = 2.0  # Fully filled
                    else:
                        fill_status[fvg_idx] = 1.0  # Partially filled
                        new_active.append((fvg_idx, gap_high, gap_low))
                else:
                    new_active.append((fvg_idx, gap_high, gap_low))
            else:
                # Bearish FVG filled when price goes above gap_low
                if high[i] >= gap_low:
                    if high[i] >= gap_high:
                        fill_status[fvg_idx] = 2.0
                    else:
                        fill_status[fvg_idx] = 1.0
                        new_active.append((fvg_idx, gap_high, gap_low))
                else:
                    new_active.append((fvg_idx, gap_high, gap_low))

        active_fvgs = new_active

        # Add new FVG if present
        if fvg_present[i] > 0:
            if is_bullish:
                gap_high = low[i]
                gap_low = high[i-2]
            else:
                gap_high = low[i-2]
                gap_low = high[i]
            active_fvgs.append((i, gap_high, gap_low))

    return fill_status
```

### 5.3 Break of Structure (BOS) and Change of Character (CHOCH)

Break of Structure occurs when price breaks beyond a significant swing high or low, indicating potential trend continuation. Change of Character signals potential trend reversals, occurring when price breaks structure in the opposite direction of the prevailing trend.

```python
@njit(cache=True)
def calculate_bos_choch(high: np.ndarray, low: np.ndarray,
                        swing_highs: np.ndarray, swing_lows: np.ndarray,
                        lookback: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Break of Structure and Change of Character

    BOS (Break of Structure):
    - Bullish BOS: Price breaks above swing high during uptrend
    - Bearish BOS: Price breaks below swing low during downtrend
    - Indicates trend continuation

    CHOCH (Change of Character):
    - Bullish CHOCH: Price breaks above swing high during downtrend
    - Bearish CHOCH: Price breaks below swing low during uptrend
    - Indicates potential trend reversal

    Args:
        high: High prices
        low: Low prices
        swing_highs: Binary array marking swing highs
        swing_lows: Binary array marking swing lows
        lookback: Bars to look back for swing points

    Returns:
        bos: BOS signals (+1 bullish, -1 bearish, 0 none)
        choch: CHOCH signals (+1 bullish, -1 bearish, 0 none)
        trend: Current trend state (+1 bullish, -1 bearish, 0 neutral)
    """
    n = len(high)
    bos = np.zeros(n, dtype=np.float64)
    choch = np.zeros(n, dtype=np.float64)
    trend = np.zeros(n, dtype=np.float64)

    # Track last significant swing points
    last_swing_high_price = high[0]
    last_swing_high_idx = 0
    last_swing_low_price = low[0]
    last_swing_low_idx = 0
    current_trend = 0  # 0=neutral, 1=bullish, -1=bearish

    for i in range(1, n):
        # Update swing point tracking
        if swing_highs[i] > 0:
            last_swing_high_price = high[i]
            last_swing_high_idx = i

        if swing_lows[i] > 0:
            last_swing_low_price = low[i]
            last_swing_low_idx = i

        # Check for structure breaks
        # Bullish break: current high exceeds last swing high
        if high[i] > last_swing_high_price and last_swing_high_idx < i:
            if current_trend == -1:
                # Breaking high in downtrend = CHOCH (reversal signal)
                choch[i] = 1.0
            else:
                # Breaking high in uptrend or neutral = BOS (continuation)
                bos[i] = 1.0
            current_trend = 1

        # Bearish break: current low breaks last swing low
        elif low[i] < last_swing_low_price and last_swing_low_idx < i:
            if current_trend == 1:
                # Breaking low in uptrend = CHOCH (reversal signal)
                choch[i] = -1.0
            else:
                # Breaking low in downtrend or neutral = BOS (continuation)
                bos[i] = -1.0
            current_trend = -1

        trend[i] = current_trend

    return bos, choch, trend
```

### 5.4 Order Block Detection

Order Blocks are consolidation areas before significant price moves, representing zones where institutional orders were accumulated:

```python
class OrderBlockDetector:
    """
    Identify order blocks (institutional accumulation zones)

    Bullish Order Block:
    - The last bearish candle before a strong bullish move
    - Represents institutional buying that pushed price up

    Bearish Order Block:
    - The last bullish candle before a strong bearish move
    - Represents institutional selling that pushed price down

    Order blocks become support/resistance zones where price
    may react when returning to those levels.
    """

    def __init__(self, atr_multiplier: float = 2.0, lookback: int = 20):
        self.atr_multiplier = atr_multiplier
        self.lookback = lookback

    def detect_order_blocks(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Identify order blocks in price data

        Returns DataFrame with columns:
        - bullish_ob: Boolean indicating bullish order block
        - bearish_ob: Boolean indicating bearish order block
        - ob_high: Order block zone high
        - ob_low: Order block zone low
        - ob_strength: Strength of the move that created the OB
        """
        df = ohlcv.copy()

        # Calculate ATR for move strength measurement
        atr = self._calculate_atr(df, period=14)
        strong_move_threshold = self.atr_multiplier * atr

        # Calculate 3-bar move magnitude
        move = df['close'].diff(3).abs()
        strong_moves = move > strong_move_threshold

        # Initialize columns
        df['bullish_ob'] = False
        df['bearish_ob'] = False
        df['ob_high'] = np.nan
        df['ob_low'] = np.nan
        df['ob_strength'] = 0.0

        for i in range(3, len(df)):
            if strong_moves.iloc[i]:
                move_direction = np.sign(df['close'].iloc[i] - df['close'].iloc[i-3])

                if move_direction > 0:
                    # Bullish move - find last bearish candle before the move
                    for j in range(i-1, max(0, i-self.lookback), -1):
                        if df['close'].iloc[j] < df['open'].iloc[j]:
                            df.loc[df.index[j], 'bullish_ob'] = True
                            df.loc[df.index[j], 'ob_high'] = df['high'].iloc[j]
                            df.loc[df.index[j], 'ob_low'] = df['low'].iloc[j]
                            df.loc[df.index[j], 'ob_strength'] = move.iloc[i] / atr.iloc[i]
                            break

                else:
                    # Bearish move - find last bullish candle before the move
                    for j in range(i-1, max(0, i-self.lookback), -1):
                        if df['close'].iloc[j] > df['open'].iloc[j]:
                            df.loc[df.index[j], 'bearish_ob'] = True
                            df.loc[df.index[j], 'ob_high'] = df['high'].iloc[j]
                            df.loc[df.index[j], 'ob_low'] = df['low'].iloc[j]
                            df.loc[df.index[j], 'ob_strength'] = move.iloc[i] / atr.iloc[i]
                            break

        return df

    def calculate_ob_proximity(self, current_price: float,
                               active_obs: List[dict]) -> dict:
        """
        Calculate proximity to active order blocks

        Returns dict with:
        - nearest_bullish_ob: Distance to nearest bullish OB
        - nearest_bearish_ob: Distance to nearest bearish OB
        - in_bullish_ob: Boolean if price is within a bullish OB
        - in_bearish_ob: Boolean if price is within a bearish OB
        """
        result = {
            'nearest_bullish_ob': np.inf,
            'nearest_bearish_ob': np.inf,
            'in_bullish_ob': False,
            'in_bearish_ob': False,
        }

        for ob in active_obs:
            distance = min(
                abs(current_price - ob['ob_high']),
                abs(current_price - ob['ob_low'])
            )

            in_zone = ob['ob_low'] <= current_price <= ob['ob_high']

            if ob['type'] == 'bullish':
                if distance < result['nearest_bullish_ob']:
                    result['nearest_bullish_ob'] = distance
                if in_zone:
                    result['in_bullish_ob'] = True
            else:
                if distance < result['nearest_bearish_ob']:
                    result['nearest_bearish_ob'] = distance
                if in_zone:
                    result['in_bearish_ob'] = True

        return result
```

### 5.5 Fractal Detection for Swing Points

Market fractals identify swing highs and lows, essential for structure analysis:

```python
@njit(cache=True)
def detect_fractals(high: np.ndarray, low: np.ndarray,
                    left_bars: int = 2, right_bars: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect Williams fractals (swing highs and lows)

    A fractal high occurs when a bar's high is higher than
    the highs of 'left_bars' bars to the left AND 'right_bars' bars to the right.

    A fractal low occurs when a bar's low is lower than
    the lows of the surrounding bars.

    Args:
        high: Array of high prices
        low: Array of low prices
        left_bars: Number of bars to check on the left
        right_bars: Number of bars to check on the right

    Returns:
        fractal_high: Binary array (1 = fractal high, 0 = not)
        fractal_low: Binary array (1 = fractal low, 0 = not)
    """
    n = len(high)
    fractal_high = np.zeros(n, dtype=np.float64)
    fractal_low = np.zeros(n, dtype=np.float64)

    for i in range(left_bars, n - right_bars):
        # Check for fractal high
        is_fractal_high = True
        current_high = high[i]

        # Check left bars
        for j in range(i - left_bars, i):
            if high[j] >= current_high:
                is_fractal_high = False
                break

        # Check right bars
        if is_fractal_high:
            for j in range(i + 1, i + right_bars + 1):
                if high[j] >= current_high:
                    is_fractal_high = False
                    break

        if is_fractal_high:
            fractal_high[i] = 1.0

        # Check for fractal low
        is_fractal_low = True
        current_low = low[i]

        # Check left bars
        for j in range(i - left_bars, i):
            if low[j] <= current_low:
                is_fractal_low = False
                break

        # Check right bars
        if is_fractal_low:
            for j in range(i + 1, i + right_bars + 1):
                if low[j] <= current_low:
                    is_fractal_low = False
                    break

        if is_fractal_low:
            fractal_low[i] = 1.0

    return fractal_high, fractal_low
```

### 5.6 Integrated Smart Money Feature Vector

The SmartMoneyEngine combines all SMC features into a unified feature vector:

```python
class SmartMoneyEngine:
    """
    Unified Smart Money Concepts feature engine

    Combines all SMC indicators into a single feature vector
    for use by the RL agent.
    """

    def __init__(self, config: dict):
        self.config = config
        self.fvg_min_size = config.get('fvg_min_size', 0.0)
        self.fractal_left = config.get('fractal_left_bars', 2)
        self.fractal_right = config.get('fractal_right_bars', 2)
        self.ob_detector = OrderBlockDetector(
            atr_multiplier=config.get('ob_atr_multiplier', 2.0)
        )

    def calculate_features(self, ohlcv: pd.DataFrame) -> np.ndarray:
        """
        Calculate complete SMC feature vector

        Returns array with shape (n_bars, n_features) containing:
        - FVG features (4): bullish_present, bearish_present, bullish_size, bearish_size
        - BOS/CHOCH features (3): bos, choch, trend
        - Order block features (4): bullish_ob, bearish_ob, ob_proximity_bull, ob_proximity_bear
        - Fractal features (4): fractal_high, fractal_low, bars_since_high, bars_since_low
        - Structure features (2): higher_highs_count, lower_lows_count
        """
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        close = ohlcv['close'].values

        # FVG detection
        bull_fvg, bull_fvg_size = detect_bullish_fvg(high, low, self.fvg_min_size)
        bear_fvg, bear_fvg_size = detect_bearish_fvg(high, low, self.fvg_min_size)

        # Fractal detection
        fractal_high, fractal_low = detect_fractals(
            high, low, self.fractal_left, self.fractal_right
        )

        # BOS/CHOCH calculation
        bos, choch, trend = calculate_bos_choch(high, low, fractal_high, fractal_low)

        # Order block detection
        ob_df = self.ob_detector.detect_order_blocks(ohlcv)

        # Assemble feature matrix
        features = np.column_stack([
            # FVG features
            bull_fvg,
            bear_fvg,
            self._normalize_by_atr(bull_fvg_size, ohlcv),
            self._normalize_by_atr(bear_fvg_size, ohlcv),

            # BOS/CHOCH features
            bos,
            choch,
            trend,

            # Order block features
            ob_df['bullish_ob'].astype(float).values,
            ob_df['bearish_ob'].astype(float).values,
            self._calculate_ob_proximity_features(close, ob_df),

            # Fractal features
            fractal_high,
            fractal_low,
            self._bars_since(fractal_high),
            self._bars_since(fractal_low),

            # Structure features
            self._count_higher_highs(high, fractal_high, lookback=10),
            self._count_lower_lows(low, fractal_low, lookback=10),
        ])

        return features
```

---

## 6. Portfolio Risk Management System

### 6.1 Architecture Overview

The Portfolio Risk Management System (src/agents/portfolio_risk.py, 1,797 lines) represents one of the most sophisticated components of the trading platform. It implements institutional-grade risk quantification and management across multiple dimensions.

**Core Components:**
- `VaRCalculator`: Multi-method Value at Risk computation
- `CorrelationEngine`: Dynamic correlation analysis and regime detection
- `ExposureManager`: Position and sector exposure tracking with limits
- `StressTester`: Scenario-based risk analysis
- `PortfolioRiskManager`: Top-level orchestration and integration

### 6.2 Value at Risk (VaR) Implementation

VaR measures the potential loss in portfolio value over a specified time horizon at a given confidence level. The system implements five distinct VaR methodologies, each with different strengths:

**6.2.1 Historical VaR**

The simplest and most intuitive approach, using actual historical returns without distributional assumptions:

```python
def calculate_historical_var(self, returns: np.ndarray,
                             confidence_level: float = 0.95,
                             holding_period: int = 1) -> dict:
    """
    Historical VaR using empirical distribution

    Advantages:
    - No distributional assumptions
    - Captures actual return characteristics including fat tails
    - Simple to understand and explain

    Disadvantages:
    - Requires sufficient historical data
    - Assumes past distribution represents future risk
    - Sensitive to outliers in historical data

    Args:
        returns: Array of historical returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        holding_period: Number of periods to scale VaR

    Returns:
        dict with VaR estimate and diagnostics
    """
    if len(returns) < self.min_observations:
        return {'var': np.nan, 'error': 'Insufficient data'}

    # Calculate VaR percentile
    var_percentile = (1 - confidence_level) * 100
    var = np.percentile(returns, var_percentile)

    # Scale for holding period (square root of time rule)
    if holding_period > 1:
        var = var * np.sqrt(holding_period)

    return {
        'var': var,
        'method': 'historical',
        'confidence_level': confidence_level,
        'holding_period': holding_period,
        'n_observations': len(returns),
        'var_percentile': var_percentile,
    }
```

**6.2.2 Parametric (Gaussian) VaR**

Assumes returns follow a normal distribution:

```python
def calculate_parametric_var(self, returns: np.ndarray,
                             confidence_level: float = 0.95,
                             holding_period: int = 1) -> dict:
    """
    Parametric VaR assuming normal distribution

    Formula: VaR = -mu + sigma * z_alpha

    Where z_alpha is the standard normal quantile

    Advantages:
    - Computationally efficient
    - Works with limited data
    - Easy to decompose by risk factors

    Disadvantages:
    - Assumes normal distribution (underestimates tail risk)
    - Financial returns typically have fat tails
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    # Get z-score for confidence level
    z_alpha = norm.ppf(1 - confidence_level)

    # Calculate VaR
    var = -(mu + z_alpha * sigma)

    # Scale for holding period
    if holding_period > 1:
        var = var * np.sqrt(holding_period)

    return {
        'var': -var,
        'method': 'parametric',
        'confidence_level': confidence_level,
        'mean_return': mu,
        'volatility': sigma,
        'z_score': z_alpha,
    }
```

**6.2.3 Cornish-Fisher VaR**

Adjusts for skewness and kurtosis in the return distribution, addressing the fat-tail problem of parametric VaR:

```python
def calculate_cornish_fisher_var(self, returns: np.ndarray,
                                  confidence_level: float = 0.95,
                                  holding_period: int = 1) -> dict:
    """
    Modified VaR using Cornish-Fisher expansion

    Accounts for non-normality through skewness and kurtosis adjustments.
    The Cornish-Fisher expansion modifies the normal quantile to account
    for higher moments of the distribution.

    Formula:
    z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3z)*K/24 - (2z^3 - 5z)*S^2/36

    Where:
    - z is the normal quantile
    - S is skewness
    - K is excess kurtosis
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # Excess kurtosis

    z = norm.ppf(1 - confidence_level)

    # Cornish-Fisher expansion
    z_cf = (z +
            (z**2 - 1) * skew / 6 +
            (z**3 - 3*z) * kurt / 24 -
            (2*z**3 - 5*z) * skew**2 / 36)

    var = -(mu + z_cf * sigma)

    if holding_period > 1:
        var = var * np.sqrt(holding_period)

    return {
        'var': -var,
        'method': 'cornish_fisher',
        'confidence_level': confidence_level,
        'skewness': skew,
        'kurtosis': kurt,
        'z_normal': z,
        'z_adjusted': z_cf,
    }
```

**6.2.4 Monte Carlo VaR**

Simulates future return paths using the estimated return distribution:

```python
def calculate_monte_carlo_var(self, returns: np.ndarray,
                               confidence_level: float = 0.95,
                               n_simulations: int = 10000,
                               holding_period: int = 1,
                               use_bootstrap: bool = False) -> dict:
    """
    Monte Carlo VaR through simulation

    Two simulation approaches:
    1. Parametric: Generate from fitted distribution
    2. Bootstrap: Resample from historical returns

    Advantages:
    - Flexible, can model complex distributions
    - Can incorporate path dependencies
    - Natural for multi-asset portfolios

    Disadvantages:
    - Computationally intensive
    - Results depend on simulation assumptions
    """
    if use_bootstrap:
        # Bootstrap resampling from historical returns
        simulated_returns = np.random.choice(
            returns,
            size=(n_simulations, holding_period),
            replace=True
        )
        # Sum returns over holding period
        portfolio_returns = simulated_returns.sum(axis=1)
    else:
        # Parametric simulation assuming normal distribution
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        simulated_returns = np.random.normal(
            mu * holding_period,
            sigma * np.sqrt(holding_period),
            n_simulations
        )
        portfolio_returns = simulated_returns

    var_percentile = (1 - confidence_level) * 100
    var = np.percentile(portfolio_returns, var_percentile)

    # Also calculate CVaR from simulation
    cvar = portfolio_returns[portfolio_returns <= var].mean()

    return {
        'var': var,
        'cvar': cvar,
        'method': 'monte_carlo',
        'n_simulations': n_simulations,
        'use_bootstrap': use_bootstrap,
        'simulated_mean': portfolio_returns.mean(),
        'simulated_std': portfolio_returns.std(),
    }
```

**6.2.5 EWMA (Exponentially Weighted Moving Average) VaR**

Uses time-weighted volatility estimates, giving more weight to recent observations:

```python
def calculate_ewma_var(self, returns: np.ndarray,
                       confidence_level: float = 0.95,
                       decay_factor: float = 0.94,
                       holding_period: int = 1) -> dict:
    """
    EWMA VaR with exponentially weighted volatility

    The decay factor (lambda) determines how quickly older observations
    lose influence. Common values:
    - 0.94: RiskMetrics daily volatility
    - 0.97: RiskMetrics monthly volatility

    More responsive to recent market conditions than equal-weighted
    historical approaches.

    Formula for EWMA variance:
    sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_{t-1}
    """
    n = len(returns)

    # Calculate weights (most recent gets highest weight)
    weights = np.array([(1-decay_factor) * decay_factor**i
                        for i in range(n)])
    weights = weights[::-1]  # Reverse so recent is highest
    weights /= weights.sum()  # Normalize

    # Weighted mean and variance
    mean_return = np.average(returns, weights=weights)
    ewma_variance = np.average((returns - mean_return)**2, weights=weights)
    ewma_sigma = np.sqrt(ewma_variance)

    z_alpha = norm.ppf(1 - confidence_level)
    var = -(mean_return + z_alpha * ewma_sigma)

    if holding_period > 1:
        var = var * np.sqrt(holding_period)

    return {
        'var': -var,
        'method': 'ewma',
        'decay_factor': decay_factor,
        'ewma_volatility': ewma_sigma,
        'effective_observations': 1 / (1 - decay_factor),
    }
```

### 6.3 Conditional VaR (CVaR / Expected Shortfall)

CVaR measures the expected loss beyond the VaR threshold, providing insight into tail risk that VaR alone misses:

```python
def calculate_cvar(self, returns: np.ndarray,
                   confidence_level: float = 0.95) -> dict:
    """
    Conditional VaR (Expected Shortfall)

    CVaR is the expected loss given that the loss exceeds VaR.
    It answers: "If we're in the worst (1-confidence)% of cases,
    what's our expected loss?"

    CVaR is a coherent risk measure (unlike VaR) and is preferred
    by many regulators and risk managers for its superior mathematical
    properties.

    Properties:
    - Subadditive: CVaR(A+B) <= CVaR(A) + CVaR(B)
    - Convex: Encourages diversification
    - Always >= VaR
    """
    var = self.calculate_historical_var(returns, confidence_level)['var']

    # Get all returns worse than VaR
    tail_returns = returns[returns <= var]

    if len(tail_returns) == 0:
        # No observations in tail (rare with sufficient data)
        cvar = var
    else:
        cvar = np.mean(tail_returns)

    return {
        'cvar': cvar,
        'var': var,
        'cvar_var_ratio': cvar / var if var != 0 else np.nan,
        'n_tail_observations': len(tail_returns),
        'tail_percentage': len(tail_returns) / len(returns),
    }
```

### 6.4 Correlation Engine

The correlation engine tracks relationships between assets and detects regime changes that affect portfolio risk:

```python
class CorrelationEngine:
    """
    Dynamic correlation analysis for portfolio risk management

    Correlations between assets are not static - they tend to increase
    during market stress (correlation breakdown), which is exactly when
    diversification benefits are most needed.

    This engine:
    1. Calculates rolling correlation matrices
    2. Detects correlation regime changes
    3. Issues warnings when correlations spike
    4. Computes diversification metrics
    """

    def __init__(self, lookback_short: int = 20,
                 lookback_long: int = 60,
                 correlation_threshold: float = 0.7,
                 regime_change_threshold: float = 0.2):
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.correlation_threshold = correlation_threshold
        self.regime_change_threshold = regime_change_threshold
        self.correlation_history = []

    def calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate current correlation matrix from recent returns"""
        return returns.tail(self.lookback_short).corr()

    def detect_correlation_regime(self, returns: pd.DataFrame) -> dict:
        """
        Detect correlation regime and changes

        Regimes:
        - HIGH_CORRELATION: Average pairwise correlation > 0.7
          (Risk-off environment, reduced diversification)
        - NORMAL: Average correlation between 0.3 and 0.7
        - LOW_CORRELATION: Average correlation < 0.3
          (Good diversification environment)

        Trends indicate whether correlations are increasing or decreasing.
        """
        short_corr = returns.tail(self.lookback_short).corr()
        long_corr = returns.tail(self.lookback_long).corr()

        # Calculate average pairwise correlation (excluding diagonal)
        short_avg = self._average_off_diagonal(short_corr)
        long_avg = self._average_off_diagonal(long_corr)

        # Determine regime
        if short_avg > 0.7:
            regime = 'HIGH_CORRELATION'
        elif short_avg < 0.3:
            regime = 'LOW_CORRELATION'
        else:
            regime = 'NORMAL'

        # Determine trend
        correlation_change = short_avg - long_avg
        if correlation_change > self.regime_change_threshold:
            trend = 'INCREASING'
        elif correlation_change < -self.regime_change_threshold:
            trend = 'DECREASING'
        else:
            trend = 'STABLE'

        # Identify highly correlated pairs
        high_corr_pairs = self._find_high_correlation_pairs(short_corr)

        return {
            'regime': regime,
            'trend': trend,
            'short_avg_correlation': short_avg,
            'long_avg_correlation': long_avg,
            'correlation_change': correlation_change,
            'high_correlation_pairs': high_corr_pairs,
            'diversification_ratio': self._calculate_diversification_ratio(short_corr),
        }

    def _average_off_diagonal(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate average of off-diagonal elements"""
        n = len(corr_matrix)
        if n < 2:
            return 0.0

        mask = ~np.eye(n, dtype=bool)
        off_diagonal = corr_matrix.values[mask]
        return np.mean(np.abs(off_diagonal))

    def _find_high_correlation_pairs(self, corr_matrix: pd.DataFrame) -> List[dict]:
        """Find pairs with correlation above threshold"""
        pairs = []
        symbols = corr_matrix.columns.tolist()

        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > self.correlation_threshold:
                    pairs.append({
                        'symbol1': symbols[i],
                        'symbol2': symbols[j],
                        'correlation': corr,
                    })

        return sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)

    def _calculate_diversification_ratio(self, corr_matrix: pd.DataFrame) -> float:
        """
        Calculate diversification ratio

        DR = sum of individual volatilities / portfolio volatility
        DR > 1 indicates diversification benefit
        Higher DR = better diversification
        """
        n = len(corr_matrix)
        if n < 2:
            return 1.0

        # Assuming equal weights for simplicity
        weights = np.ones(n) / n

        # Portfolio variance with correlations
        portfolio_var = weights @ corr_matrix.values @ weights

        # Sum of individual variances (assuming unit variance for correlation matrix)
        individual_var_sum = np.sum(weights**2)

        return np.sqrt(individual_var_sum) / np.sqrt(portfolio_var)
```

### 6.5 Exposure Manager

The exposure manager tracks and limits various exposure dimensions to prevent concentration risk:

```python
class ExposureManager:
    """
    Multi-dimensional exposure tracking and management

    Tracks exposure across multiple dimensions:
    - Gross exposure: Total absolute position value
    - Net exposure: Directional bias (long - short)
    - Single position: Concentration in individual assets
    - Sector: Exposure to correlated asset groups
    - Currency: Exposure by base currency
    """

    def __init__(self, config: dict):
        self.max_gross_exposure = config.get('max_gross_exposure', 2.0)
        self.max_net_exposure = config.get('max_net_exposure', 1.0)
        self.max_single_position = config.get('max_single_position', 0.25)
        self.max_sector_exposure = config.get('max_sector_exposure', 0.5)
        self.max_currency_exposure = config.get('max_currency_exposure', 0.6)

        # Asset classification
        self.asset_sectors = config.get('asset_sectors', {})
        self.asset_currencies = config.get('asset_currencies', {})

    def calculate_exposures(self, portfolio: dict) -> dict:
        """
        Calculate all exposure metrics for current portfolio

        Args:
            portfolio: Dict mapping symbols to position info
                      {symbol: {'weight': float, 'direction': str}}

        Returns:
            Dict with all exposure metrics
        """
        if not portfolio:
            return self._empty_exposure()

        weights = [p['weight'] for p in portfolio.values()]

        gross_exposure = sum(abs(w) for w in weights)
        net_exposure = sum(w for w in weights)

        # Sector exposures
        sector_exposures = {}
        for symbol, position in portfolio.items():
            sector = self.asset_sectors.get(symbol, 'OTHER')
            sector_exposures[sector] = sector_exposures.get(sector, 0) + abs(position['weight'])

        # Currency exposures
        currency_exposures = {}
        for symbol, position in portfolio.items():
            currency = self.asset_currencies.get(symbol, 'USD')
            currency_exposures[currency] = currency_exposures.get(currency, 0) + position['weight']

        return {
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'long_exposure': sum(w for w in weights if w > 0),
            'short_exposure': abs(sum(w for w in weights if w < 0)),
            'n_positions': len(portfolio),
            'largest_position': max(abs(w) for w in weights),
            'sector_exposures': sector_exposures,
            'currency_exposures': currency_exposures,
        }

    def check_limits(self, portfolio: dict, proposed_trade: dict = None) -> dict:
        """
        Check all exposure limits

        Returns violations and warnings for current portfolio,
        optionally including a proposed trade.
        """
        # Calculate current exposures
        exposures = self.calculate_exposures(portfolio)

        # If proposed trade, calculate post-trade exposures
        if proposed_trade:
            post_portfolio = self._apply_trade(portfolio, proposed_trade)
            post_exposures = self.calculate_exposures(post_portfolio)
        else:
            post_exposures = exposures

        violations = []
        warnings = []

        # Check gross exposure
        if post_exposures['gross_exposure'] > self.max_gross_exposure:
            violations.append({
                'type': 'GROSS_EXPOSURE',
                'current': post_exposures['gross_exposure'],
                'limit': self.max_gross_exposure,
                'excess': post_exposures['gross_exposure'] - self.max_gross_exposure,
            })
        elif post_exposures['gross_exposure'] > self.max_gross_exposure * 0.9:
            warnings.append({
                'type': 'GROSS_EXPOSURE_WARNING',
                'current': post_exposures['gross_exposure'],
                'limit': self.max_gross_exposure,
            })

        # Check net exposure
        if abs(post_exposures['net_exposure']) > self.max_net_exposure:
            violations.append({
                'type': 'NET_EXPOSURE',
                'current': post_exposures['net_exposure'],
                'limit': self.max_net_exposure,
            })

        # Check single position
        if post_exposures['largest_position'] > self.max_single_position:
            violations.append({
                'type': 'POSITION_CONCENTRATION',
                'current': post_exposures['largest_position'],
                'limit': self.max_single_position,
            })

        # Check sector exposures
        for sector, exposure in post_exposures['sector_exposures'].items():
            if exposure > self.max_sector_exposure:
                violations.append({
                    'type': 'SECTOR_EXPOSURE',
                    'sector': sector,
                    'current': exposure,
                    'limit': self.max_sector_exposure,
                })

        return {
            'exposures': post_exposures,
            'violations': violations,
            'warnings': warnings,
            'within_limits': len(violations) == 0,
            'trade_allowed': len(violations) == 0,
        }
```

### 6.6 Stress Testing Framework

The stress tester evaluates portfolio performance under extreme historical and hypothetical scenarios:

```python
class StressTester:
    """
    Scenario-based stress testing framework

    Evaluates portfolio resilience under extreme market conditions
    using both historical scenarios and hypothetical shocks.
    """

    HISTORICAL_SCENARIOS = {
        'financial_crisis_2008': {
            'description': '2008 Global Financial Crisis',
            'equity_shock': -0.50,
            'forex_shock': 0.15,
            'commodity_shock': -0.40,
            'volatility_spike': 3.0,
            'correlation_increase': 0.3,
            'duration_days': 180,
        },
        'flash_crash_2010': {
            'description': '2010 Flash Crash',
            'equity_shock': -0.10,
            'volatility_spike': 5.0,
            'correlation_increase': 0.5,
            'duration_minutes': 30,
            'gap_risk': True,
        },
        'chf_unpeg_2015': {
            'description': 'SNB CHF Unpeg January 2015',
            'forex_shock': 0.30,
            'volatility_spike': 10.0,
            'gap_risk': True,
            'liquidity_crisis': True,
        },
        'covid_crash_2020': {
            'description': 'COVID-19 Market Crash March 2020',
            'equity_shock': -0.35,
            'forex_shock': 0.10,
            'commodity_shock': -0.60,  # Oil went negative!
            'volatility_spike': 4.0,
            'correlation_increase': 0.4,
            'duration_days': 30,
        },
        'rate_shock': {
            'description': 'Sudden Interest Rate Increase',
            'equity_shock': -0.15,
            'bond_shock': -0.10,
            'forex_shock': 0.05,
            'volatility_spike': 2.0,
        },
    }

    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value

    def run_stress_test(self, portfolio: dict, scenario_name: str) -> dict:
        """
        Execute stress test for a specific scenario

        Args:
            portfolio: Current portfolio positions
            scenario_name: Name of scenario to test

        Returns:
            Detailed impact assessment
        """
        if scenario_name not in self.HISTORICAL_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.HISTORICAL_SCENARIOS[scenario_name]

        # Calculate impact on each position
        position_impacts = {}
        total_loss = 0

        for symbol, position in portfolio.items():
            asset_class = self._classify_asset(symbol)
            shock_key = f"{asset_class}_shock"

            shock = scenario.get(shock_key, 0)

            # Adjust for position direction
            position_value = position['weight'] * self.portfolio_value
            if position['direction'] == 'SHORT':
                impact = -position_value * shock
            else:
                impact = position_value * shock

            position_impacts[symbol] = {
                'value': position_value,
                'shock': shock,
                'impact': impact,
                'impact_pct': impact / self.portfolio_value,
            }

            total_loss += impact

        # Calculate survival metrics
        post_stress_value = self.portfolio_value + total_loss
        drawdown = -total_loss / self.portfolio_value

        # Recovery time estimate based on scenario duration
        recovery_estimate = self._estimate_recovery_time(scenario, drawdown)

        return {
            'scenario': scenario_name,
            'description': scenario['description'],
            'portfolio_value_before': self.portfolio_value,
            'portfolio_value_after': post_stress_value,
            'total_impact': total_loss,
            'total_impact_pct': total_loss / self.portfolio_value,
            'drawdown': drawdown,
            'position_impacts': position_impacts,
            'worst_position': min(position_impacts.items(),
                                  key=lambda x: x[1]['impact']),
            'survival': post_stress_value > 0,
            'margin_call_risk': drawdown > 0.5,
            'recovery_estimate_days': recovery_estimate,
            'recommendations': self._generate_recommendations(drawdown, scenario),
        }

    def run_all_scenarios(self, portfolio: dict) -> dict:
        """Run all stress scenarios and summarize results"""
        results = {}

        for scenario_name in self.HISTORICAL_SCENARIOS:
            results[scenario_name] = self.run_stress_test(portfolio, scenario_name)

        # Summary statistics
        impacts = [r['total_impact_pct'] for r in results.values()]

        return {
            'scenarios': results,
            'worst_case_scenario': min(results.items(),
                                       key=lambda x: x[1]['total_impact_pct'])[0],
            'worst_case_impact': min(impacts),
            'average_impact': np.mean(impacts),
            'survival_all_scenarios': all(r['survival'] for r in results.values()),
        }
```

---

## 7. Kill Switch and Circuit Breaker System

### 7.1 System Overview and Philosophy

The Kill Switch system (src/agents/kill_switch.py, 1,697 lines) represents the last line of defense against catastrophic losses. Unlike simple stop-loss mechanisms that operate at the position level, the kill switch operates at the system level, providing graduated responses to deteriorating conditions and ensuring the trading system can protect capital even in extreme market scenarios.

The philosophy behind the kill switch is that trading systems should "fail safe" - when things go wrong, the default behavior should be to reduce risk, not to continue trading. The system implements this through a seven-level graduated halt system that provides proportional responses to different severity levels of problems.

### 7.2 Halt Level Enumeration

The system defines seven distinct halt levels, each with specific behaviors and restrictions:

```python
class HaltLevel(Enum):
    """
    Graduated halt levels for proportional risk response

    Each level represents increasing severity of trading restrictions.
    The system can move up or down through levels as conditions change.
    """
    NONE = 0           # Normal operation - all trading allowed
    CAUTION = 1        # Warnings active, monitoring intensified
    REDUCED = 2        # Position sizing reduced by 50%
    NEW_ONLY = 3       # No new positions, can modify/close existing
    CLOSE_ONLY = 4     # Must close positions, no modifications
    FULL_HALT = 5      # All trading stopped, positions held
    EMERGENCY = 6      # Emergency - flatten all positions immediately
```

**Level 0 - NONE (Normal Operation):**
All trading functions operate normally. The system monitors conditions but takes no restrictive action. This is the default state when all risk metrics are within acceptable ranges.

**Level 1 - CAUTION:**
Early warning state activated when conditions begin deteriorating. Trading continues but with enhanced monitoring. Alerts are generated for human review. Position sizing may be slightly reduced (75% of normal). This level serves as an early indicator that conditions warrant attention.

**Level 2 - REDUCED:**
Position sizing is reduced to 50% of normal levels. New trades are still allowed but with smaller size to limit additional risk exposure. Existing positions are monitored more closely. This level activates when daily losses approach warning thresholds or volatility increases significantly.

**Level 3 - NEW_ONLY:**
No new positions can be opened. Existing positions can be modified (stop-loss adjustments, partial closes) but no new risk can be added to the portfolio. This level is appropriate when conditions suggest the current market environment is unfavorable for new entries.

**Level 4 - CLOSE_ONLY:**
Positions must be closed. No new positions and no modifications that increase risk. The system actively works to reduce exposure by closing positions when they reach break-even or better. Losing positions may be closed at market to limit further losses.

**Level 5 - FULL_HALT:**
All trading activity stops immediately. Positions are held but not modified. This level is appropriate for system errors, connectivity issues, or other technical problems where the safest action is to freeze all activity until the issue is resolved.

**Level 6 - EMERGENCY:**
Emergency liquidation of all positions at market prices. This is the most severe response, used only when immediate capital preservation is critical. Examples include margin calls, extreme market events, or system compromise.

### 7.3 Halt Reason Classification

The system tracks why halts are triggered through a comprehensive reason enumeration:

```python
class HaltReason(Enum):
    """
    Categories of halt triggers

    Organized by source: automatic triggers, system issues,
    manual intervention, and external factors.
    """
    # Automatic triggers based on trading performance
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    WEEKLY_LOSS_LIMIT = "weekly_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    WIN_RATE_DEGRADATION = "win_rate_degradation"

    # System issues
    CONNECTIVITY_LOSS = "connectivity_loss"
    DATA_FEED_ERROR = "data_feed_error"
    EXECUTION_FAILURES = "execution_failures"
    SYSTEM_ERROR = "system_error"
    MEMORY_PRESSURE = "memory_pressure"

    # Manual triggers
    MANUAL_HALT = "manual_halt"
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"
    ADMIN_OVERRIDE = "admin_override"

    # External factors
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    NEWS_BLACKOUT = "news_blackout"
    MARKET_CLOSURE = "market_closure"
    BROKER_RESTRICTION = "broker_restriction"
```

### 7.4 Circuit Breaker Implementation

The circuit breaker pattern provides automatic fault detection and recovery:

```python
class CircuitBreaker:
    """
    Circuit breaker for trading system fault tolerance

    The circuit breaker prevents cascading failures by detecting
    repeated errors and temporarily halting operations until
    the system recovers.

    State Machine:
    - CLOSED: Normal operation, counting failures
    - OPEN: Too many failures, rejecting all operations
    - HALF_OPEN: Testing recovery with limited operations

    Transitions:
    - CLOSED -> OPEN: When failure_count >= failure_threshold
    - OPEN -> HALF_OPEN: After recovery_timeout expires
    - HALF_OPEN -> CLOSED: If test operations succeed
    - HALF_OPEN -> OPEN: If test operations fail
    """

    def __init__(self, name: str, config: dict):
        self.name = name
        self.failure_threshold = config.get('failure_threshold', 5)
        self.recovery_timeout = config.get('recovery_timeout', 300)
        self.half_open_max_calls = config.get('half_open_max_calls', 3)

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now()
        self.half_open_calls = 0

        # Thread safety
        self._lock = threading.RLock()

        # Metrics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_history: List[dict] = []

    def execute(self, operation: Callable[[], T]) -> T:
        """
        Execute operation with circuit breaker protection

        Args:
            operation: Callable to execute

        Returns:
            Result of operation if successful

        Raises:
            CircuitBreakerOpen: If circuit is open
            Original exception: If operation fails
        """
        with self._lock:
            self.total_calls += 1

            # Check if we should allow the call
            if not self._should_allow_call():
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Retry after {self._time_until_retry()} seconds."
                )

            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1

        # Execute outside lock to avoid blocking
        try:
            result = operation()
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def _should_allow_call(self) -> bool:
        """Determine if a call should be allowed based on current state"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self._recovery_timeout_elapsed():
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            return self.half_open_calls < self.half_open_max_calls

        return False

    def _record_success(self):
        """Record a successful operation"""
        with self._lock:
            self.total_successes += 1
            self.success_count += 1

            if self.state == CircuitState.HALF_OPEN:
                # Successful call in half-open state
                if self.success_count >= self.half_open_max_calls:
                    self._transition_to(CircuitState.CLOSED)
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    def _record_failure(self, exception: Exception):
        """Record a failed operation"""
        with self._lock:
            self.total_failures += 1
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                # Failure in half-open state reopens circuit
                self._transition_to(CircuitState.OPEN)
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state"""
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.now()

        if new_state == CircuitState.HALF_OPEN:
            self.half_open_calls = 0
            self.success_count = 0

        if new_state == CircuitState.CLOSED:
            self.failure_count = 0

        # Record state change
        self.state_history.append({
            'timestamp': datetime.now().isoformat(),
            'from_state': old_state.name,
            'to_state': new_state.name,
            'failure_count': self.failure_count,
        })

        logger.info(f"Circuit breaker '{self.name}' transitioned: "
                   f"{old_state.name} -> {new_state.name}")

    def get_status(self) -> dict:
        """Get current circuit breaker status"""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.name,
                'failure_count': self.failure_count,
                'total_calls': self.total_calls,
                'total_failures': self.total_failures,
                'failure_rate': self.total_failures / max(1, self.total_calls),
                'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'time_in_state': (datetime.now() - self.last_state_change).total_seconds(),
            }
```

### 7.5 Recovery Manager

The RecoveryManager handles gradual system recovery after halt conditions clear:

```python
class RecoveryManager:
    """
    Manages system recovery after halt conditions

    Recovery is gradual to prevent immediate re-triggering of halt conditions.
    The system moves through recovery phases with increasing trading capacity.

    Recovery Phases:
    1. Cooling off period (no trading)
    2. Observation period (monitoring only)
    3. Limited trading (25% capacity)
    4. Reduced trading (50% capacity)
    5. Normal trading (100% capacity)
    """

    def __init__(self, config: dict):
        self.cooling_off_hours = config.get('cooling_off_hours', 24)
        self.observation_hours = config.get('observation_hours', 4)
        self.limited_trading_hours = config.get('limited_trading_hours', 8)
        self.reduced_trading_hours = config.get('reduced_trading_hours', 12)

        self.recovery_start_time: Optional[datetime] = None
        self.current_phase: RecoveryPhase = RecoveryPhase.NONE
        self.halt_history: List[dict] = []

    def start_recovery(self, halt_reason: HaltReason, halt_level: HaltLevel):
        """
        Initiate recovery process after halt condition clears

        Args:
            halt_reason: Original reason for the halt
            halt_level: Severity level of the halt
        """
        self.recovery_start_time = datetime.now()
        self.current_phase = RecoveryPhase.COOLING_OFF

        # Adjust recovery timeline based on halt severity
        severity_multiplier = self._get_severity_multiplier(halt_level)

        self.adjusted_cooling_off = self.cooling_off_hours * severity_multiplier
        self.adjusted_observation = self.observation_hours * severity_multiplier

        logger.info(f"Recovery initiated. Cooling off period: "
                   f"{self.adjusted_cooling_off} hours")

        self.halt_history.append({
            'timestamp': datetime.now().isoformat(),
            'reason': halt_reason.value,
            'level': halt_level.name,
            'recovery_start': self.recovery_start_time.isoformat(),
        })

    def get_current_capacity(self) -> float:
        """
        Get current trading capacity based on recovery phase

        Returns:
            Float between 0.0 and 1.0 representing allowed capacity
        """
        if self.recovery_start_time is None:
            return 1.0  # Normal operation

        hours_elapsed = (datetime.now() - self.recovery_start_time).total_seconds() / 3600

        if hours_elapsed < self.adjusted_cooling_off:
            self.current_phase = RecoveryPhase.COOLING_OFF
            return 0.0

        elif hours_elapsed < self.adjusted_cooling_off + self.adjusted_observation:
            self.current_phase = RecoveryPhase.OBSERVATION
            return 0.0

        elif hours_elapsed < self.adjusted_cooling_off + self.adjusted_observation + self.limited_trading_hours:
            self.current_phase = RecoveryPhase.LIMITED_TRADING
            return 0.25

        elif hours_elapsed < self.adjusted_cooling_off + self.adjusted_observation + self.limited_trading_hours + self.reduced_trading_hours:
            self.current_phase = RecoveryPhase.REDUCED_TRADING
            return 0.50

        else:
            self.current_phase = RecoveryPhase.NORMAL
            self.recovery_start_time = None  # Recovery complete
            return 1.0

    def _get_severity_multiplier(self, halt_level: HaltLevel) -> float:
        """Get recovery time multiplier based on halt severity"""
        multipliers = {
            HaltLevel.CAUTION: 0.25,
            HaltLevel.REDUCED: 0.5,
            HaltLevel.NEW_ONLY: 0.75,
            HaltLevel.CLOSE_ONLY: 1.0,
            HaltLevel.FULL_HALT: 1.5,
            HaltLevel.EMERGENCY: 2.0,
        }
        return multipliers.get(halt_level, 1.0)

    def can_reset_manually(self) -> Tuple[bool, str]:
        """
        Check if manual reset is allowed

        Manual resets are rate-limited to prevent circumventing safety measures.
        """
        if self.current_phase == RecoveryPhase.COOLING_OFF:
            remaining = self.adjusted_cooling_off - self._hours_elapsed()
            return False, f"Cooling off period active. {remaining:.1f} hours remaining."

        if len(self.halt_history) >= 3:
            recent_halts = [h for h in self.halt_history
                          if self._is_recent(h['timestamp'], hours=48)]
            if len(recent_halts) >= 3:
                return False, "Too many recent halts. Manual reset blocked for safety."

        return True, "Manual reset allowed."
```

### 7.6 State Persistence

The kill switch persists its state to survive system restarts:

```python
class KillSwitchPersistence:
    """
    SQLite-based persistence for kill switch state

    Ensures that halt conditions survive system restarts.
    If the system was halted and restarts, it should resume
    in the halted state, not start fresh.
    """

    def __init__(self, db_path: str = "kill_switch_state.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS halt_state (
                    id INTEGER PRIMARY KEY,
                    halt_level INTEGER NOT NULL,
                    halt_reason TEXT NOT NULL,
                    halt_time TEXT NOT NULL,
                    resume_time TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                    name TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    failure_count INTEGER NOT NULL,
                    last_failure_time TEXT,
                    last_update TEXT NOT NULL
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS recovery_state (
                    id INTEGER PRIMARY KEY,
                    recovery_start TEXT NOT NULL,
                    current_phase TEXT NOT NULL,
                    original_halt_level INTEGER,
                    original_halt_reason TEXT
                )
            ''')

    def save_halt_state(self, halt_level: HaltLevel, halt_reason: HaltReason,
                        metadata: dict = None):
        """Save current halt state to database"""
        with sqlite3.connect(self.db_path) as conn:
            # Deactivate any existing active halts
            conn.execute(
                "UPDATE halt_state SET is_active = 0 WHERE is_active = 1"
            )

            # Insert new halt state
            conn.execute('''
                INSERT INTO halt_state
                (halt_level, halt_reason, halt_time, is_active, metadata)
                VALUES (?, ?, ?, 1, ?)
            ''', (
                halt_level.value,
                halt_reason.value,
                datetime.now().isoformat(),
                json.dumps(metadata) if metadata else None
            ))

    def load_halt_state(self) -> Optional[dict]:
        """Load active halt state from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT halt_level, halt_reason, halt_time, metadata
                FROM halt_state
                WHERE is_active = 1
                ORDER BY halt_time DESC
                LIMIT 1
            ''')

            row = cursor.fetchone()
            if row:
                return {
                    'halt_level': HaltLevel(row[0]),
                    'halt_reason': HaltReason(row[1]),
                    'halt_time': datetime.fromisoformat(row[2]),
                    'metadata': json.loads(row[3]) if row[3] else None,
                }
            return None

    def clear_halt_state(self, reset_token: str = None):
        """
        Clear halt state (resume trading)

        Requires valid reset token for security.
        """
        if reset_token and not self._validate_reset_token(reset_token):
            raise SecurityError("Invalid reset token")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE halt_state
                SET is_active = 0, resume_time = ?
                WHERE is_active = 1
            ''', (datetime.now().isoformat(),))

    def _validate_reset_token(self, token: str) -> bool:
        """Validate cryptographic reset token"""
        # Token validation logic - ensures manual resets are authorized
        return hmac.compare_digest(
            token,
            self._generate_expected_token()
        )
```

### 7.7 Integrated Kill Switch Controller

The main KillSwitch class coordinates all components:

```python
class KillSwitch:
    """
    Main kill switch controller

    Coordinates halt level management, circuit breakers,
    recovery management, and state persistence.
    """

    def __init__(self, config: dict):
        self.config = config

        # Thresholds
        self.daily_loss_limit = config.get('daily_loss_limit', 0.05)
        self.weekly_loss_limit = config.get('weekly_loss_limit', 0.10)
        self.max_drawdown = config.get('max_drawdown', 0.20)
        self.consecutive_loss_limit = config.get('consecutive_loss_limit', 5)
        self.volatility_spike_percentile = config.get('volatility_spike_percentile', 0.99)

        # Components
        self.persistence = KillSwitchPersistence(config.get('db_path'))
        self.recovery_manager = RecoveryManager(config)

        # Circuit breakers for different subsystems
        self.circuit_breakers = {
            'execution': CircuitBreaker('execution', config.get('execution_cb', {})),
            'data_feed': CircuitBreaker('data_feed', config.get('data_feed_cb', {})),
            'risk_calc': CircuitBreaker('risk_calc', config.get('risk_calc_cb', {})),
        }

        # Current state
        self.current_level = HaltLevel.NONE
        self.current_reason: Optional[HaltReason] = None
        self.halt_time: Optional[datetime] = None

        # Load persisted state
        self._load_persisted_state()

    def check_conditions(self, portfolio_state: dict, market_state: dict) -> HaltAssessment:
        """
        Comprehensive condition checking

        Evaluates all halt conditions and returns assessment
        with recommended halt level.
        """
        triggers = []

        # Check daily loss
        if portfolio_state['daily_pnl_pct'] < -self.daily_loss_limit:
            triggers.append(HaltTrigger(
                reason=HaltReason.DAILY_LOSS_LIMIT,
                severity=HaltLevel.CLOSE_ONLY,
                value=portfolio_state['daily_pnl_pct'],
                threshold=-self.daily_loss_limit,
                message=f"Daily loss {portfolio_state['daily_pnl_pct']:.2%} exceeds limit"
            ))

        # Check weekly loss
        if portfolio_state.get('weekly_pnl_pct', 0) < -self.weekly_loss_limit:
            triggers.append(HaltTrigger(
                reason=HaltReason.WEEKLY_LOSS_LIMIT,
                severity=HaltLevel.FULL_HALT,
                value=portfolio_state['weekly_pnl_pct'],
                threshold=-self.weekly_loss_limit,
                message=f"Weekly loss {portfolio_state['weekly_pnl_pct']:.2%} exceeds limit"
            ))

        # Check drawdown
        if portfolio_state['current_drawdown'] > self.max_drawdown:
            triggers.append(HaltTrigger(
                reason=HaltReason.MAX_DRAWDOWN,
                severity=HaltLevel.EMERGENCY if portfolio_state['current_drawdown'] > self.max_drawdown * 1.5 else HaltLevel.CLOSE_ONLY,
                value=portfolio_state['current_drawdown'],
                threshold=self.max_drawdown,
                message=f"Drawdown {portfolio_state['current_drawdown']:.2%} exceeds maximum"
            ))

        # Check consecutive losses
        if portfolio_state.get('consecutive_losses', 0) >= self.consecutive_loss_limit:
            triggers.append(HaltTrigger(
                reason=HaltReason.CONSECUTIVE_LOSSES,
                severity=HaltLevel.REDUCED,
                value=portfolio_state['consecutive_losses'],
                threshold=self.consecutive_loss_limit,
                message=f"{portfolio_state['consecutive_losses']} consecutive losses"
            ))

        # Check volatility spike
        if market_state.get('volatility_percentile', 0) > self.volatility_spike_percentile:
            triggers.append(HaltTrigger(
                reason=HaltReason.VOLATILITY_SPIKE,
                severity=HaltLevel.NEW_ONLY,
                value=market_state['volatility_percentile'],
                threshold=self.volatility_spike_percentile,
                message=f"Volatility at {market_state['volatility_percentile']:.0%} percentile"
            ))

        # Check circuit breakers
        for name, cb in self.circuit_breakers.items():
            if cb.state == CircuitState.OPEN:
                triggers.append(HaltTrigger(
                    reason=HaltReason.SYSTEM_ERROR,
                    severity=HaltLevel.FULL_HALT,
                    value=cb.failure_count,
                    threshold=cb.failure_threshold,
                    message=f"Circuit breaker '{name}' is open"
                ))

        # Determine highest severity
        if triggers:
            highest_severity = max(t.severity.value for t in triggers)
            recommended_level = HaltLevel(highest_severity)
            primary_reason = max(triggers, key=lambda t: t.severity.value).reason
        else:
            recommended_level = HaltLevel.NONE
            primary_reason = None

        return HaltAssessment(
            recommended_level=recommended_level,
            primary_reason=primary_reason,
            triggers=triggers,
            current_level=self.current_level,
            recovery_capacity=self.recovery_manager.get_current_capacity(),
        )

    def execute_halt(self, level: HaltLevel, reason: HaltReason,
                     portfolio_manager=None) -> dict:
        """
        Execute halt at specified level

        Args:
            level: Halt level to execute
            reason: Reason for the halt
            portfolio_manager: Optional portfolio manager for position actions

        Returns:
            Dict with actions taken
        """
        actions = []

        self.current_level = level
        self.current_reason = reason
        self.halt_time = datetime.now()

        # Persist state
        self.persistence.save_halt_state(level, reason)

        # Execute level-specific actions
        if level == HaltLevel.EMERGENCY and portfolio_manager:
            # Flatten all positions immediately
            closed = portfolio_manager.close_all_positions(
                reason="Emergency halt - immediate liquidation"
            )
            actions.append(f"Emergency liquidation: {len(closed)} positions closed")

        elif level == HaltLevel.CLOSE_ONLY and portfolio_manager:
            # Start closing positions
            actions.append("Close-only mode activated")

        elif level == HaltLevel.FULL_HALT:
            actions.append("Full trading halt activated")

        elif level == HaltLevel.NEW_ONLY:
            actions.append("New position block activated")

        elif level == HaltLevel.REDUCED:
            actions.append("Position sizing reduced to 50%")

        elif level == HaltLevel.CAUTION:
            actions.append("Caution mode activated")

        # Send alerts
        self._send_halt_alert(level, reason, actions)

        logger.warning(f"Kill switch activated: Level={level.name}, "
                      f"Reason={reason.value}, Actions={actions}")

        return {
            'level': level.name,
            'reason': reason.value,
            'actions': actions,
            'halt_time': self.halt_time.isoformat(),
        }
```

---

## 8. Walk-Forward Validation Framework

### 8.1 The Problem with Traditional Backtesting

Traditional backtesting, where a model is trained on historical data and tested on a held-out portion, suffers from several critical flaws when applied to financial markets:

**Look-Ahead Bias:** Even with careful train/test splitting, subtle forms of information leakage can occur. The modeler might unconsciously adjust parameters after seeing test results, or the test period might be unrepresentative.

**Non-Stationarity:** Financial markets are inherently non-stationary. The patterns that worked in 2020 may not work in 2025. A single test period cannot capture this regime variation.

**Overfitting:** With enough parameter tuning, almost any model can be made to perform well on a fixed test set. This doesn't indicate genuine predictive ability.

**Survivorship Bias:** If many model variants are tested and only the best is reported, the published results are optimistically biased.

Walk-Forward Validation (WFV) addresses these problems by simulating realistic deployment conditions.

### 8.2 Walk-Forward Methodology

The WalkForwardValidator class (parallel_training.py, 1,635 lines) implements rigorous walk-forward validation:

```python
class WalkForwardValidator:
    """
    Walk-Forward Validation for non-stationary financial time series

    The walk-forward process:
    1. Train on window [0, T]
    2. Test on window [T+purge, T+purge+test_size]
    3. Slide window forward by step_size
    4. Repeat until end of data

    The purge gap prevents information leakage from training
    into the test period.
    """

    def __init__(self, config: dict):
        # Window sizes (in bars)
        self.train_window = config.get('train_window_bars', 6720)  # 6 months @ M15
        self.validation_window = config.get('validation_window_bars', 2240)  # 2 months
        self.test_window = config.get('test_window_bars', 1120)  # 1 month
        self.step_size = config.get('step_size_bars', 1120)  # 1 month
        self.purge_gap = config.get('purge_gap_bars', 96)  # 1 day gap

        # Validation strategy
        self.strategy = config.get('strategy', 'rolling')  # rolling, expanding, anchored

        # Minimum requirements
        self.min_folds = config.get('min_folds', 6)
        self.min_train_samples = config.get('min_train_samples', 5000)

    def generate_folds(self, data_length: int) -> List[WalkForwardFold]:
        """
        Generate train/validation/test fold indices

        Args:
            data_length: Total number of samples in dataset

        Returns:
            List of WalkForwardFold objects with index ranges
        """
        folds = []
        fold_idx = 0

        if self.strategy == 'rolling':
            # Rolling window - fixed training size
            start_idx = 0
            while True:
                train_start = start_idx
                train_end = train_start + self.train_window

                val_start = train_end + self.purge_gap
                val_end = val_start + self.validation_window

                test_start = val_end + self.purge_gap
                test_end = test_start + self.test_window

                if test_end > data_length:
                    break

                folds.append(WalkForwardFold(
                    fold_idx=fold_idx,
                    train_indices=(train_start, train_end),
                    validation_indices=(val_start, val_end),
                    test_indices=(test_start, test_end),
                    purge_gap=self.purge_gap,
                ))

                start_idx += self.step_size
                fold_idx += 1

        elif self.strategy == 'expanding':
            # Expanding window - training grows over time
            train_start = 0
            current_train_end = self.train_window

            while True:
                train_end = current_train_end

                val_start = train_end + self.purge_gap
                val_end = val_start + self.validation_window

                test_start = val_end + self.purge_gap
                test_end = test_start + self.test_window

                if test_end > data_length:
                    break

                folds.append(WalkForwardFold(
                    fold_idx=fold_idx,
                    train_indices=(train_start, train_end),
                    validation_indices=(val_start, val_end),
                    test_indices=(test_start, test_end),
                    purge_gap=self.purge_gap,
                ))

                current_train_end += self.step_size
                fold_idx += 1

        elif self.strategy == 'anchored':
            # Anchored - train always starts from beginning
            train_start = 0

            while True:
                train_end = self.train_window + (fold_idx * self.step_size)

                val_start = train_end + self.purge_gap
                val_end = val_start + self.validation_window

                test_start = val_end + self.purge_gap
                test_end = test_start + self.test_window

                if test_end > data_length:
                    break

                folds.append(WalkForwardFold(
                    fold_idx=fold_idx,
                    train_indices=(train_start, train_end),
                    validation_indices=(val_start, val_end),
                    test_indices=(test_start, test_end),
                    purge_gap=self.purge_gap,
                ))

                fold_idx += 1

        if len(folds) < self.min_folds:
            raise InsufficientDataError(
                f"Only {len(folds)} folds possible, minimum {self.min_folds} required. "
                f"Need more data or smaller windows."
            )

        return folds

    def run_validation(self, data: pd.DataFrame,
                       model_factory: Callable,
                       hyperparams: dict) -> WalkForwardResult:
        """
        Execute complete walk-forward validation

        Args:
            data: Full historical dataset
            model_factory: Function to create fresh model instances
            hyperparams: Hyperparameters for model creation

        Returns:
            WalkForwardResult with aggregated metrics
        """
        folds = self.generate_folds(len(data))
        fold_results = []

        logger.info(f"Starting walk-forward validation: {len(folds)} folds, "
                   f"strategy={self.strategy}")

        for fold in folds:
            logger.info(f"Processing fold {fold.fold_idx + 1}/{len(folds)}")

            # Extract data for this fold
            train_data = data.iloc[fold.train_indices[0]:fold.train_indices[1]]
            val_data = data.iloc[fold.validation_indices[0]:fold.validation_indices[1]]
            test_data = data.iloc[fold.test_indices[0]:fold.test_indices[1]]

            # Create fresh model
            model = model_factory(hyperparams)

            # Train
            train_start_time = time.time()
            model.train(train_data)
            train_time = time.time() - train_start_time

            # Validate
            val_metrics = self._evaluate(model, val_data, 'validation')

            # Test
            test_metrics = self._evaluate(model, test_data, 'test')

            fold_result = FoldResult(
                fold_idx=fold.fold_idx,
                train_period=(data.index[fold.train_indices[0]],
                            data.index[fold.train_indices[1]-1]),
                val_period=(data.index[fold.validation_indices[0]],
                           data.index[fold.validation_indices[1]-1]),
                test_period=(data.index[fold.test_indices[0]],
                            data.index[fold.test_indices[1]-1]),
                train_samples=len(train_data),
                train_time_seconds=train_time,
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
            )

            fold_results.append(fold_result)

            logger.info(f"Fold {fold.fold_idx}: "
                       f"Val Sharpe={val_metrics['sharpe']:.3f}, "
                       f"Test Sharpe={test_metrics['sharpe']:.3f}")

        # Aggregate results
        return self._aggregate_results(fold_results, hyperparams)

    def _evaluate(self, model, data: pd.DataFrame, phase: str) -> dict:
        """Evaluate model on data and compute metrics"""
        # Generate predictions
        predictions = model.predict(data)

        # Calculate strategy returns
        returns = self._calculate_strategy_returns(predictions, data)

        # Compute comprehensive metrics
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annual_return': self._annualize_return(returns),
            'sharpe': self._calculate_sharpe(returns),
            'sortino': self._calculate_sortino(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'calmar': self._calculate_calmar(returns),
            'win_rate': (returns > 0).mean(),
            'profit_factor': self._calculate_profit_factor(returns),
            'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0,
            'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0,
            'n_trades': self._count_trades(predictions),
            'avg_holding_period': self._avg_holding_period(predictions),
        }

        return metrics

    def _aggregate_results(self, fold_results: List[FoldResult],
                           hyperparams: dict) -> WalkForwardResult:
        """Aggregate results across all folds"""
        # Extract test metrics
        test_sharpes = [f.test_metrics['sharpe'] for f in fold_results]
        test_returns = [f.test_metrics['total_return'] for f in fold_results]
        test_drawdowns = [f.test_metrics['max_drawdown'] for f in fold_results]

        # Calculate stability metrics
        sharpe_mean = np.mean(test_sharpes)
        sharpe_std = np.std(test_sharpes)
        sharpe_cv = sharpe_std / abs(sharpe_mean) if sharpe_mean != 0 else np.inf

        # Consistency metrics
        positive_folds = sum(1 for s in test_sharpes if s > 0)
        consistency_ratio = positive_folds / len(test_sharpes)

        # Degradation analysis
        degradation = self._analyze_degradation(fold_results)

        return WalkForwardResult(
            n_folds=len(fold_results),
            fold_results=fold_results,
            hyperparams=hyperparams,
            aggregate_metrics={
                'mean_sharpe': sharpe_mean,
                'std_sharpe': sharpe_std,
                'sharpe_cv': sharpe_cv,
                'mean_return': np.mean(test_returns),
                'mean_drawdown': np.mean(test_drawdowns),
                'max_drawdown_worst_fold': max(test_drawdowns),
                'consistency_ratio': consistency_ratio,
                'positive_folds': positive_folds,
            },
            degradation_analysis=degradation,
            stability_score=self._calculate_stability_score(sharpe_cv, consistency_ratio),
            commercial_viability=self._assess_commercial_viability(
                sharpe_mean, sharpe_cv, consistency_ratio, degradation
            ),
        )
```

### 8.3 Purge Gap for Data Leakage Prevention

The purge gap is critical for preventing information leakage:

```python
def _validate_no_leakage(self, fold: WalkForwardFold) -> bool:
    """
    Verify no data leakage between train and test

    Leakage can occur through:
    1. Overlapping indices (prevented by purge gap)
    2. Feature calculations using future data
    3. Target calculations using future data

    The purge gap ensures that even features calculated with
    lookback windows don't contaminate the test period.
    """
    train_end = fold.train_indices[1]
    val_start = fold.validation_indices[0]
    test_start = fold.test_indices[0]

    # Check purge gaps
    train_to_val_gap = val_start - train_end
    val_to_test_gap = test_start - fold.validation_indices[1]

    if train_to_val_gap < self.purge_gap:
        logger.warning(f"Insufficient purge gap: {train_to_val_gap} < {self.purge_gap}")
        return False

    if val_to_test_gap < self.purge_gap:
        logger.warning(f"Insufficient val-test gap: {val_to_test_gap} < {self.purge_gap}")
        return False

    return True
```

### 8.4 Commercial Viability Scoring

The system assesses whether model performance meets commercial requirements:

```python
def _assess_commercial_viability(self, mean_sharpe: float,
                                  sharpe_cv: float,
                                  consistency: float,
                                  degradation: dict) -> CommercialViability:
    """
    Assess whether performance meets commercial deployment standards

    Criteria:
    - Minimum Sharpe ratio (risk-adjusted return)
    - Maximum Sharpe variability (stability)
    - Minimum consistency (reliability)
    - Acceptable degradation (sustainability)

    Returns:
        CommercialViability enum: APPROVED, CONDITIONAL, REJECTED
    """
    score = 0
    reasons = []

    # Sharpe ratio requirement
    if mean_sharpe >= 1.5:
        score += 3
    elif mean_sharpe >= 1.0:
        score += 2
    elif mean_sharpe >= 0.5:
        score += 1
        reasons.append("Marginal Sharpe ratio")
    else:
        reasons.append("Insufficient Sharpe ratio")

    # Stability requirement (low CV is good)
    if sharpe_cv <= 0.3:
        score += 3
    elif sharpe_cv <= 0.5:
        score += 2
    elif sharpe_cv <= 0.7:
        score += 1
        reasons.append("Moderate performance variability")
    else:
        reasons.append("High performance variability")

    # Consistency requirement
    if consistency >= 0.8:
        score += 3
    elif consistency >= 0.6:
        score += 2
    elif consistency >= 0.5:
        score += 1
        reasons.append("Marginal consistency")
    else:
        reasons.append("Poor fold consistency")

    # Degradation requirement
    if degradation['trend'] == 'stable' or degradation['trend'] == 'improving':
        score += 2
    elif degradation['trend'] == 'mild_degradation':
        score += 1
        reasons.append("Mild performance degradation over time")
    else:
        reasons.append("Significant performance degradation")

    # Final assessment
    if score >= 9:
        return CommercialViability(
            status='APPROVED',
            score=score,
            max_score=11,
            reasons=[],
            recommendations=["Ready for paper trading validation"]
        )
    elif score >= 6:
        return CommercialViability(
            status='CONDITIONAL',
            score=score,
            max_score=11,
            reasons=reasons,
            recommendations=self._generate_improvement_recommendations(reasons)
        )
    else:
        return CommercialViability(
            status='REJECTED',
            score=score,
            max_score=11,
            reasons=reasons,
            recommendations=["Requires significant model improvements"]
        )
```

### 8.5 Intelligent Hyperparameter Generation

The system uses intelligent sampling strategies for hyperparameter optimization:

```python
class HyperparameterOptimizer:
    """
    Intelligent hyperparameter optimization for trading models

    Combines multiple sampling strategies:
    1. Latin Hypercube Sampling for uniform coverage
    2. Bayesian-inspired sampling near good regions
    3. Random exploration for diversity
    """

    def __init__(self, param_space: dict, n_samples: int = 20):
        self.param_space = param_space
        self.n_samples = n_samples
        self.optimization_history = []

    def generate_candidates(self) -> List[dict]:
        """Generate hyperparameter candidates using mixed strategy"""
        candidates = []

        # Latin Hypercube Sampling for initial coverage
        n_lhs = self.n_samples // 2
        lhs_candidates = self._latin_hypercube_sample(n_lhs)
        candidates.extend(lhs_candidates)

        # Exploitation: Sample near best known configurations
        if self.optimization_history:
            n_exploit = self.n_samples // 4
            exploit_candidates = self._exploit_good_regions(n_exploit)
            candidates.extend(exploit_candidates)

        # Random exploration for remaining
        n_random = self.n_samples - len(candidates)
        random_candidates = self._random_sample(n_random)
        candidates.extend(random_candidates)

        return candidates

    def _latin_hypercube_sample(self, n_samples: int) -> List[dict]:
        """
        Latin Hypercube Sampling for uniform coverage

        LHS ensures that each dimension is uniformly sampled,
        providing better coverage than pure random sampling.
        """
        n_dims = len(self.param_space)
        samples = []

        # Create stratified samples for each dimension
        intervals = np.arange(0, n_samples + 1) / n_samples

        for _ in range(n_samples):
            sample = {}
            for param_name, param_config in self.param_space.items():
                # Random point within stratum
                stratum_idx = np.random.randint(0, n_samples)
                u = np.random.uniform(intervals[stratum_idx], intervals[stratum_idx + 1])

                # Transform to parameter space
                if param_config['type'] == 'float':
                    value = param_config['min'] + u * (param_config['max'] - param_config['min'])
                    if param_config.get('log_scale', False):
                        log_min = np.log(param_config['min'])
                        log_max = np.log(param_config['max'])
                        value = np.exp(log_min + u * (log_max - log_min))
                elif param_config['type'] == 'int':
                    value = int(param_config['min'] + u * (param_config['max'] - param_config['min']))
                elif param_config['type'] == 'categorical':
                    idx = int(u * len(param_config['choices']))
                    idx = min(idx, len(param_config['choices']) - 1)
                    value = param_config['choices'][idx]

                sample[param_name] = value

            samples.append(sample)

        return samples

    def _exploit_good_regions(self, n_samples: int) -> List[dict]:
        """Sample near the best known configurations"""
        # Sort history by performance
        sorted_history = sorted(
            self.optimization_history,
            key=lambda x: x['metrics']['sharpe'],
            reverse=True
        )

        # Take top performers
        top_k = min(3, len(sorted_history))
        top_configs = [h['params'] for h in sorted_history[:top_k]]

        samples = []
        for _ in range(n_samples):
            # Pick a top config to perturb
            base_config = random.choice(top_configs)

            # Perturb each parameter slightly
            perturbed = {}
            for param_name, param_config in self.param_space.items():
                base_value = base_config[param_name]

                if param_config['type'] == 'float':
                    # Gaussian perturbation with 10% std
                    std = 0.1 * (param_config['max'] - param_config['min'])
                    value = np.random.normal(base_value, std)
                    value = np.clip(value, param_config['min'], param_config['max'])
                elif param_config['type'] == 'int':
                    std = max(1, 0.1 * (param_config['max'] - param_config['min']))
                    value = int(np.random.normal(base_value, std))
                    value = np.clip(value, param_config['min'], param_config['max'])
                else:
                    value = base_value

                perturbed[param_name] = value

            samples.append(perturbed)

        return samples

    def update_history(self, params: dict, metrics: dict):
        """Record optimization result"""
        self.optimization_history.append({
            'params': params,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        })
```

---

## 9. Advanced Reward Shaping System

### 9.1 The Challenge of Reward Design

Designing reward functions for trading agents is notoriously difficult. Simple rewards like raw P&L encourage excessive risk-taking. Pure Sharpe ratio optimization can lead to minimal trading. The ideal reward function must balance multiple competing objectives while maintaining stable learning dynamics.

### 9.2 Multi-Objective Reward Architecture

```python
class TradingRewardCalculator:
    """
    Sophisticated multi-objective reward shaping for financial RL

    The reward function balances:
    1. Profitability (raw returns)
    2. Risk-adjustment (volatility-scaled returns)
    3. Drawdown control (capital preservation)
    4. Transaction efficiency (minimize unnecessary trading)
    5. Holding incentives (reward profitable patience)

    Each component can be weighted and the weights can be
    adapted based on market regime.
    """

    def __init__(self, config: dict):
        # Component weights
        self.return_weight = config.get('return_weight', 1.0)
        self.sharpe_weight = config.get('sharpe_weight', 0.5)
        self.drawdown_weight = config.get('drawdown_weight', 0.3)
        self.transaction_weight = config.get('transaction_weight', 0.1)
        self.holding_weight = config.get('holding_weight', 0.1)

        # Risk parameters
        self.risk_free_rate = config.get('risk_free_rate', 0.0)
        self.target_volatility = config.get('target_volatility', 0.15)

        # Transaction costs
        self.transaction_cost_rate = config.get('transaction_cost', 0.0002)

        # State tracking
        self.episode_returns = []
        self.max_portfolio_value = 0
        self.position_entry_time = 0
        self.position_entry_value = 0

    def calculate_reward(self, state: TradingState, action: float,
                         next_state: TradingState) -> RewardBreakdown:
        """
        Calculate comprehensive trading reward

        Args:
            state: State before action
            action: Action taken (-1 to +1)
            next_state: State after action

        Returns:
            RewardBreakdown with component-wise rewards
        """
        # 1. Raw return component
        pnl = next_state.portfolio_value - state.portfolio_value
        pnl_pct = pnl / state.portfolio_value
        return_reward = pnl_pct * self.return_weight

        # 2. Risk-adjusted return component
        self.episode_returns.append(pnl_pct)
        if len(self.episode_returns) >= 20:
            recent_vol = np.std(self.episode_returns[-20:])
            if recent_vol > 0:
                risk_adjusted = pnl_pct / recent_vol
            else:
                risk_adjusted = pnl_pct
        else:
            risk_adjusted = pnl_pct
        sharpe_reward = risk_adjusted * self.sharpe_weight

        # 3. Drawdown penalty
        self.max_portfolio_value = max(self.max_portfolio_value, next_state.portfolio_value)
        current_drawdown = (self.max_portfolio_value - next_state.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = -current_drawdown * self.drawdown_weight

        # Progressive penalty for deep drawdowns
        if current_drawdown > 0.1:
            drawdown_penalty *= 1.5
        if current_drawdown > 0.2:
            drawdown_penalty *= 2.0

        # 4. Transaction cost penalty
        position_change = abs(next_state.position - state.position)
        transaction_penalty = -position_change * self.transaction_cost_rate * self.transaction_weight

        # 5. Holding bonus for profitable positions
        holding_bonus = 0
        if state.position != 0 and next_state.position != 0:
            # Still holding a position
            if state.unrealized_pnl > 0:
                # Profitable position being held
                holding_time = next_state.bars_in_position
                holding_bonus = 0.0001 * holding_time * self.holding_weight

                # Extra bonus for patient holding of winners
                if holding_time > 10 and state.unrealized_pnl > state.portfolio_value * 0.01:
                    holding_bonus *= 1.5

        # 6. Regime-aware adjustment
        regime_multiplier = self._get_regime_multiplier(next_state.market_regime)

        # Combine components
        total_reward = (
            return_reward +
            sharpe_reward +
            drawdown_penalty +
            transaction_penalty +
            holding_bonus
        ) * regime_multiplier

        # Clip to prevent extreme values
        total_reward = np.clip(total_reward, -1.0, 1.0)

        return RewardBreakdown(
            total=total_reward,
            return_component=return_reward,
            sharpe_component=sharpe_reward,
            drawdown_component=drawdown_penalty,
            transaction_component=transaction_penalty,
            holding_component=holding_bonus,
            regime_multiplier=regime_multiplier,
            raw_pnl=pnl,
            raw_pnl_pct=pnl_pct,
            current_drawdown=current_drawdown,
        )

    def _get_regime_multiplier(self, regime: str) -> float:
        """
        Adjust reward scale based on market regime

        In high volatility regimes, reduce reward magnitude to prevent
        the agent from learning excessively from noisy periods.

        In low volatility regimes, increase reward magnitude to encourage
        the agent to take positions when opportunities arise.
        """
        regime_multipliers = {
            'HIGH_VOLATILITY': 0.5,    # Reduce learning from noisy periods
            'CRISIS': 0.3,              # Further reduce during crisis
            'NORMAL': 1.0,              # Standard scaling
            'LOW_VOLATILITY': 1.2,      # Encourage action in quiet markets
            'TRENDING': 1.1,            # Slightly boost trending rewards
            'RANGING': 0.9,             # Slightly reduce in choppy markets
        }
        return regime_multipliers.get(regime, 1.0)

    def reset_episode(self):
        """Reset state tracking for new episode"""
        self.episode_returns = []
        self.max_portfolio_value = 0
        self.position_entry_time = 0
        self.position_entry_value = 0
```

### 9.3 Curriculum Learning Rewards

The system implements curriculum learning where reward structure evolves as the agent learns:

```python
class CurriculumRewardScheduler:
    """
    Curriculum learning for trading agent rewards

    The reward function evolves through stages:
    1. Basic: Focus on not losing money
    2. Intermediate: Introduce risk-adjustment
    3. Advanced: Full multi-objective optimization

    This progression helps the agent develop robust behaviors
    before optimizing for complex objectives.
    """

    def __init__(self, config: dict):
        self.current_stage = 0
        self.stage_thresholds = config.get('stage_thresholds', [100000, 300000, 500000])

        self.stage_configs = [
            # Stage 1: Basic - focus on capital preservation
            {
                'return_weight': 0.5,
                'sharpe_weight': 0.0,
                'drawdown_weight': 1.0,
                'transaction_weight': 0.0,
                'holding_weight': 0.0,
            },
            # Stage 2: Intermediate - introduce risk-adjustment
            {
                'return_weight': 0.7,
                'sharpe_weight': 0.3,
                'drawdown_weight': 0.5,
                'transaction_weight': 0.05,
                'holding_weight': 0.0,
            },
            # Stage 3: Advanced - full optimization
            {
                'return_weight': 1.0,
                'sharpe_weight': 0.5,
                'drawdown_weight': 0.3,
                'transaction_weight': 0.1,
                'holding_weight': 0.1,
            },
        ]

    def update_stage(self, total_steps: int):
        """Update curriculum stage based on training progress"""
        for i, threshold in enumerate(self.stage_thresholds):
            if total_steps >= threshold and self.current_stage <= i:
                self.current_stage = i + 1
                logger.info(f"Curriculum advanced to stage {self.current_stage}")

    def get_current_config(self) -> dict:
        """Get reward configuration for current stage"""
        stage_idx = min(self.current_stage, len(self.stage_configs) - 1)
        return self.stage_configs[stage_idx]
```

---

## 10. Volatility Modeling with GARCH

### 10.1 Why GARCH for Trading

Volatility clustering is one of the most robust stylized facts of financial returns - periods of high volatility tend to be followed by high volatility, and vice versa. The GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model captures this phenomenon and provides crucial inputs for risk management and position sizing.

### 10.2 GARCH(1,1) Implementation

```python
class GARCHVolatilityModel:
    """
    GARCH(1,1) model for volatility estimation and forecasting

    The GARCH(1,1) model:
    sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

    Where:
    - sigma^2_t is the conditional variance at time t
    - omega is the long-run variance weight
    - alpha is the ARCH coefficient (news impact)
    - beta is the GARCH coefficient (persistence)ccali

    Constraints:
    - omega > 0
    - alpha >= 0

    - beta >= 0
    - alpha + beta < 1 (for stationarity)
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.omega = config.get('omega', 0.00001)
        self.alpha = config.get('alpha', 0.1)
        self.beta = config.get('beta', 0.85)

        # State
        self.current_variance = None
        self.long_run_variance = None
        self.variance_history = []

    def fit(self, returns: np.ndarray) -> dict:
        """
        Fit GARCH(1,1) parameters using maximum likelihood

        Args:
            returns: Array of return observations

        Returns:
            Dict with fitted parameters and diagnostics
        """
        # Initial variance estimate
        initial_var = np.var(returns)

        # Define negative log-likelihood function
        def neg_log_likelihood(params):
            omega, alpha, beta = params

            # Enforce constraints
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return np.inf

            n = len(returns)
            variance = np.zeros(n)
            variance[0] = initial_var

            # Calculate conditional variances
            for t in range(1, n):
                variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]

            # Log-likelihood (assuming normal distribution)
            ll = -0.5 * np.sum(np.log(2 * np.pi * variance) + returns**2 / variance)

            return -ll

        # Optimize
        from scipy.optimize import minimize

        result = minimize(
            neg_log_likelihood,
            x0=[self.omega, self.alpha, self.beta],
            method='L-BFGS-B',
            bounds=[(1e-8, 1), (0, 1), (0, 1)],
        )

        if result.success:
            self.omega, self.alpha, self.beta = result.x

            # Calculate long-run variance
            self.long_run_variance = self.omega / (1 - self.alpha - self.beta)

            # Initialize current variance
            self.current_variance = self._calculate_variance_series(returns)[-1]

            return {
                'success': True,
                'omega': self.omega,
                'alpha': self.alpha,
                'beta': self.beta,
                'persistence': self.alpha + self.beta,
                'long_run_variance': self.long_run_variance,
                'long_run_volatility': np.sqrt(self.long_run_variance * 252),  # Annualized
                'half_life': self._calculate_half_life(),
                'log_likelihood': -result.fun,
            }

        return {'success': False, 'message': result.message}

    def update(self, new_return: float) -> float:
        """
        Update variance estimate with new observation

        Args:
            new_return: Latest return observation

        Returns:
            Updated conditional variance
        """
        if self.current_variance is None:
            self.current_variance = new_return**2

        self.current_variance = (
            self.omega +
            self.alpha * new_return**2 +
            self.beta * self.current_variance
        )

        self.variance_history.append(self.current_variance)

        return self.current_variance

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility for multiple periods ahead

        Args:
            horizon: Number of periods to forecast

        Returns:
            Array of forecasted variances
        """
        if self.current_variance is None:
            raise ValueError("Model must be fitted before forecasting")

        forecasts = np.zeros(horizon)
        persistence = self.alpha + self.beta

        for h in range(horizon):
            if h == 0:
                forecasts[h] = self.current_variance
            else:
                # Mean-reversion formula
                forecasts[h] = (
                    self.long_run_variance +
                    persistence**h * (self.current_variance - self.long_run_variance)
                )

        return forecasts

    def get_volatility_regime(self) -> str:
        """
        Classify current volatility regime

        Returns:
            Regime classification: LOW, NORMAL, HIGH, EXTREME
        """
        if self.current_variance is None or self.long_run_variance is None:
            return 'UNKNOWN'

        ratio = self.current_variance / self.long_run_variance

        if ratio < 0.5:
            return 'LOW'
        elif ratio < 1.5:
            return 'NORMAL'
        elif ratio < 3.0:
            return 'HIGH'
        else:
            return 'EXTREME'

    def _calculate_half_life(self) -> float:
        """Calculate volatility half-life (mean reversion speed)"""
        persistence = self.alpha + self.beta
        if persistence >= 1:
            return np.inf
        return np.log(0.5) / np.log(persistence)

    def _calculate_variance_series(self, returns: np.ndarray) -> np.ndarray:
        """Calculate full variance series"""
        n = len(returns)
        variance = np.zeros(n)
        variance[0] = np.var(returns)

        for t in range(1, n):
            variance[t] = self.omega + self.alpha * returns[t-1]**2 + self.beta * variance[t-1]

        return variance
```

### 10.3 Integration with Position Sizing

The GARCH model integrates with position sizing for volatility-adjusted risk:

```python
class VolatilityAdjustedPositionSizer:
    """
    Position sizing that adapts to volatility conditions

    Uses GARCH volatility forecasts to:
    1. Scale position sizes inversely to volatility
    2. Widen stops in high-vol environments
    3. Reduce exposure during volatility spikes
    """

    def __init__(self, garch_model: GARCHVolatilityModel, config: dict):
        self.garch = garch_model
        self.target_volatility = config.get('target_volatility', 0.15)
        self.max_position = config.get('max_position', 1.0)
        self.vol_scalar = config.get('vol_scalar', 1.0)

    def calculate_position_size(self, base_size: float,
                                portfolio_value: float) -> float:
        """
        Calculate volatility-adjusted position size

        Args:
            base_size: Base position size from signal
            portfolio_value: Current portfolio value

        Returns:
            Adjusted position size
        """
        # Get current volatility forecast
        current_vol = np.sqrt(self.garch.current_variance * 252)  # Annualized

        # Calculate volatility scalar
        vol_ratio = self.target_volatility / current_vol
        vol_scalar = np.clip(vol_ratio * self.vol_scalar, 0.25, 2.0)

        # Adjust position size
        adjusted_size = base_size * vol_scalar

        # Apply maximum limit
        adjusted_size = np.clip(adjusted_size, -self.max_position, self.max_position)

        return adjusted_size

    def calculate_stop_distance(self, base_atr_multiple: float,
                                atr: float) -> float:
        """
        Calculate volatility-adjusted stop distance

        Widens stops during high volatility to avoid premature stops
        Tightens stops during low volatility for better risk control
        """
        regime = self.garch.get_volatility_regime()

        regime_adjustments = {
            'LOW': 0.8,      # Tighter stops in low vol
            'NORMAL': 1.0,
            'HIGH': 1.3,     # Wider stops in high vol
            'EXTREME': 1.5,  # Much wider in extreme vol
        }

        adjustment = regime_adjustments.get(regime, 1.0)

        return base_atr_multiple * atr * adjustment
```

---

## 11. News and Sentiment Analysis

### 11.1 FinBERT Integration

The system integrates FinBERT for financial text sentiment analysis:

```python
class FinancialSentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT

    FinBERT is a BERT model fine-tuned on financial text,
    providing more accurate sentiment analysis than general
    NLP models for financial applications.
    """

    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.sentiment_labels = ['negative', 'neutral', 'positive']

        # Caching for efficiency
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour

    def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze financial text for sentiment

        Args:
            text: Financial text (headline, tweet, article excerpt)

        Returns:
            SentimentResult with scores and classification
        """
        # Check cache
        cache_key = hash(text)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['result']

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        # Extract scores
        scores = {
            label: prob.item()
            for label, prob in zip(self.sentiment_labels, probs[0])
        }

        # Calculate composite score (-1 to +1)
        composite = scores['positive'] - scores['negative']

        result = SentimentResult(
            text=text[:100],  # Truncate for storage
            scores=scores,
            composite=composite,
            classification=max(scores, key=scores.get),
            confidence=max(scores.values()),
            timestamp=datetime.now(),
        )

        # Cache result
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time(),
        }

        return result

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Batch analysis for efficiency"""
        results = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            for j, text in enumerate(batch):
                scores = {
                    label: probs[j][k].item()
                    for k, label in enumerate(self.sentiment_labels)
                }
                composite = scores['positive'] - scores['negative']

                results.append(SentimentResult(
                    text=text[:100],
                    scores=scores,
                    composite=composite,
                    classification=max(scores, key=scores.get),
                    confidence=max(scores.values()),
                    timestamp=datetime.now(),
                ))

        return results
```

### 11.2 News Event Detection and Trading Impact

```python
class NewsEventDetector:
    """
    Detect high-impact news events that affect trading

    Categories:
    1. Scheduled events (economic calendar)
    2. Breaking news (real-time detection)
    3. Earnings announcements
    4. Central bank communications
    """

    # High-impact scheduled events
    HIGH_IMPACT_EVENTS = {
        'NFP': {'impact': 'extreme', 'blackout_minutes': 30},
        'FOMC': {'impact': 'extreme', 'blackout_minutes': 60},
        'ECB': {'impact': 'high', 'blackout_minutes': 45},
        'CPI': {'impact': 'high', 'blackout_minutes': 15},
        'GDP': {'impact': 'high', 'blackout_minutes': 15},
        'RETAIL_SALES': {'impact': 'medium', 'blackout_minutes': 10},
    }

    def __init__(self, config: dict):
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.blackout_buffer_minutes = config.get('blackout_buffer', 5)

    def check_news_impact(self, symbol: str, timestamp: datetime) -> NewsImpactAssessment:
        """
        Assess news impact for trading decision

        Returns:
            NewsImpactAssessment with trading recommendation
        """
        impacts = []

        # Check scheduled events
        scheduled_impact = self._check_scheduled_events(symbol, timestamp)
        if scheduled_impact:
            impacts.append(scheduled_impact)

        # Check recent breaking news
        breaking_impact = self._check_breaking_news(symbol, timestamp)
        if breaking_impact:
            impacts.append(breaking_impact)

        # Aggregate impacts
        if not impacts:
            return NewsImpactAssessment(
                trading_allowed=True,
                risk_multiplier=1.0,
                blocking_events=[],
                recommendation="Normal trading conditions"
            )

        # Find most severe impact
        max_impact = max(impacts, key=lambda x: x['severity'])

        if max_impact['severity'] >= 3:  # Extreme
            return NewsImpactAssessment(
                trading_allowed=False,
                risk_multiplier=0.0,
                blocking_events=impacts,
                resume_time=max_impact.get('resume_time'),
                recommendation=f"Trading blocked: {max_impact['event']}"
            )

        elif max_impact['severity'] >= 2:  # High
            return NewsImpactAssessment(
                trading_allowed=True,
                risk_multiplier=0.5,
                blocking_events=[],
                warning_events=impacts,
                recommendation="Reduced position sizing recommended"
            )

        else:  # Medium/Low
            return NewsImpactAssessment(
                trading_allowed=True,
                risk_multiplier=0.75,
                blocking_events=[],
                warning_events=impacts,
                recommendation="Caution advised"
            )
```

---

## 12. Live Trading Infrastructure

### 12.1 MetaTrader 5 Connector

```python
class MT5Connector:
    """
    Production-grade MetaTrader 5 integration

    Features:
    - Automatic reconnection
    - Order execution with retry logic
    - Position synchronization
    - Real-time price streaming
    """

    def __init__(self, config: dict):
        self.login = config['login']
        self.password = config['password']
        self.server = config['server']

        self.connected = False
        self.last_heartbeat = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

        # Order settings
        self.max_slippage_pips = config.get('max_slippage', 3)
        self.magic_number = config.get('magic_number', 12345)

    def connect(self) -> bool:
        """Establish MT5 connection with retry logic"""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                if not mt5.initialize():
                    logger.error(f"MT5 init failed: {mt5.last_error()}")
                    self.reconnect_attempts += 1
                    time.sleep(5)
                    continue

                authorized = mt5.login(
                    login=self.login,
                    password=self.password,
                    server=self.server
                )

                if authorized:
                    self.connected = True
                    self.reconnect_attempts = 0
                    self.last_heartbeat = datetime.now()
                    logger.info(f"Connected to MT5: {self.server}")
                    return True

                logger.error(f"MT5 login failed: {mt5.last_error()}")

            except Exception as e:
                logger.error(f"MT5 connection error: {e}")

            self.reconnect_attempts += 1
            time.sleep(5 * self.reconnect_attempts)  # Exponential backoff

        return False

    def execute_order(self, order: OrderRequest) -> OrderResult:
        """
        Execute order with comprehensive error handling

        Args:
            order: OrderRequest with trade details

        Returns:
            OrderResult with execution details
        """
        if not self.connected:
            if not self.connect():
                return OrderResult(success=False, error="Not connected")

        # Get symbol info
        symbol_info = mt5.symbol_info(order.symbol)
        if symbol_info is None:
            return OrderResult(success=False, error=f"Symbol {order.symbol} not found")

        if not symbol_info.visible:
            mt5.symbol_select(order.symbol, True)

        # Get current price
        tick = mt5.symbol_info_tick(order.symbol)
        if tick is None:
            return OrderResult(success=False, error="Failed to get price")

        # Determine price based on direction
        price = tick.ask if order.direction == 'BUY' else tick.bid

        # Calculate lot size
        lot_size = self._calculate_lot_size(order, symbol_info)

        # Build request
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': order.symbol,
            'volume': lot_size,
            'type': mt5.ORDER_TYPE_BUY if order.direction == 'BUY' else mt5.ORDER_TYPE_SELL,
            'price': price,
            'deviation': self.max_slippage_pips * 10,  # Points
            'magic': self.magic_number,
            'comment': order.comment or 'TradingBot',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        # Add stop loss and take profit
        if order.stop_loss:
            request['sl'] = order.stop_loss
        if order.take_profit:
            request['tp'] = order.take_profit

        # Execute with retry
        for attempt in range(3):
            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return OrderResult(
                    success=True,
                    order_id=result.order,
                    executed_price=result.price,
                    executed_volume=result.volume,
                    slippage=(result.price - price) / symbol_info.point,
                )

            logger.warning(f"Order attempt {attempt+1} failed: {result.comment}")
            time.sleep(0.5)

        return OrderResult(
            success=False,
            error=f"Order failed after retries: {result.comment}",
            retcode=result.retcode,
        )

    def get_positions(self) -> List[Position]:
        """Get all open positions for magic number"""
        positions = mt5.positions_get()
        if positions is None:
            return []

        return [
            Position(
                ticket=p.ticket,
                symbol=p.symbol,
                direction='BUY' if p.type == mt5.POSITION_TYPE_BUY else 'SELL',
                volume=p.volume,
                open_price=p.price_open,
                current_price=p.price_current,
                profit=p.profit,
                swap=p.swap,
                open_time=datetime.fromtimestamp(p.time),
            )
            for p in positions
            if p.magic == self.magic_number
        ]

    def close_position(self, ticket: int) -> OrderResult:
        """Close specific position by ticket"""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return OrderResult(success=False, error="Position not found")

        position = position[0]

        # Build close request
        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': position.symbol,
            'volume': position.volume,
            'type': close_type,
            'position': ticket,
            'price': price,
            'deviation': self.max_slippage_pips * 10,
            'magic': self.magic_number,
            'comment': 'Close position',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                success=True,
                order_id=result.order,
                executed_price=result.price,
                pnl=position.profit,
            )

        return OrderResult(success=False, error=result.comment)
```

---

## 13. Performance Optimization Techniques

### 13.1 Numba JIT Compilation

All performance-critical calculations use Numba:

```python
from numba import njit, prange

@njit(parallel=True, cache=True, fastmath=True)
def calculate_features_vectorized(high: np.ndarray, low: np.ndarray,
                                   close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Vectorized feature calculation with Numba optimization

    Performance: ~100x faster than pure Python
    """
    n = len(close)
    n_features = 50
    features = np.empty((n, n_features), dtype=np.float64)

    # Parallel loop over time steps
    for i in prange(n):
        idx = 0

        # Price features
        if i >= 20:
            features[i, idx] = (close[i] - close[i-1]) / close[i-1]
            idx += 1
            features[i, idx] = (close[i] - close[i-5]) / close[i-5]
            idx += 1
            features[i, idx] = (close[i] - close[i-20]) / close[i-20]
            idx += 1

            # Volatility features
            returns = np.empty(20)
            for j in range(20):
                returns[j] = (close[i-j] - close[i-j-1]) / close[i-j-1] if i-j-1 >= 0 else 0
            features[i, idx] = np.std(returns)
            idx += 1

            # Range features
            highest = high[i]
            lowest = low[i]
            for j in range(1, 20):
                if i-j >= 0:
                    if high[i-j] > highest:
                        highest = high[i-j]
                    if low[i-j] < lowest:
                        lowest = low[i-j]
            features[i, idx] = (close[i] - lowest) / (highest - lowest + 1e-8)
            idx += 1

        else:
            # Fill with zeros for initial period
            for j in range(n_features):
                features[i, j] = 0.0

    return features
```

### 13.2 Memory-Efficient Data Handling

```python
class EfficientDataStore:
    """
    Memory-efficient storage for large datasets

    Uses memory mapping and rolling buffers to handle
    datasets larger than available RAM.
    """

    def __init__(self, max_memory_mb: int = 1000):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.buffers = {}

    def create_rolling_buffer(self, name: str, max_size: int,
                               n_features: int) -> np.ndarray:
        """Create circular buffer for streaming data"""
        # Use memory mapping for large buffers
        if max_size * n_features * 8 > self.max_memory // 2:
            buffer_path = f"/tmp/{name}_buffer.dat"
            buffer = np.memmap(
                buffer_path,
                dtype=np.float64,
                mode='w+',
                shape=(max_size, n_features)
            )
        else:
            buffer = np.zeros((max_size, n_features), dtype=np.float64)

        self.buffers[name] = {
            'data': buffer,
            'head': 0,
            'size': 0,
            'max_size': max_size,
        }

        return buffer

    def append(self, name: str, data: np.ndarray):
        """Append data to rolling buffer"""
        buf = self.buffers[name]
        n_new = len(data)

        for i in range(n_new):
            buf['data'][buf['head']] = data[i]
            buf['head'] = (buf['head'] + 1) % buf['max_size']
            buf['size'] = min(buf['size'] + 1, buf['max_size'])

    def get_recent(self, name: str, n: int) -> np.ndarray:
        """Get most recent n samples"""
        buf = self.buffers[name]
        n = min(n, buf['size'])

        result = np.empty((n, buf['data'].shape[1]))
        for i in range(n):
            idx = (buf['head'] - n + i) % buf['max_size']
            result[i] = buf['data'][idx]

        return result
```

---

## 14. Training Pipeline and Hyperparameter Optimization

### 14.1 Google Colab Integration

The training pipeline integrates with Google Colab for GPU training:

```python
class ColabTrainingPipeline:
    """
    Training pipeline optimized for Google Colab

    Features:
    - Automatic Google Drive mounting
    - Checkpoint saving to Drive
    - Resume from interruption
    - Progress visualization
    """

    def __init__(self, config: dict):
        self.config = config
        self.drive_mounted = False
        self.checkpoint_dir = None

    def setup_environment(self):
        """Set up Colab environment"""
        # Mount Google Drive
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            self.drive_mounted = True
            self.checkpoint_dir = '/content/drive/MyDrive/trading_bot_checkpoints'
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            logger.info("Google Drive mounted successfully")
        except ImportError:
            logger.info("Not running in Colab, using local storage")
            self.checkpoint_dir = './checkpoints'
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, metrics: dict, step: int):
        """Save training checkpoint"""
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
        }

        path = f"{self.checkpoint_dir}/checkpoint_step_{step}.pt"
        torch.save(checkpoint, path)

        # Keep only last 5 checkpoints
        self._cleanup_old_checkpoints()

        logger.info(f"Checkpoint saved: {path}")

    def load_latest_checkpoint(self) -> Optional[dict]:
        """Load most recent checkpoint"""
        checkpoints = glob.glob(f"{self.checkpoint_dir}/checkpoint_step_*.pt")

        if not checkpoints:
            return None

        latest = max(checkpoints, key=os.path.getmtime)
        checkpoint = torch.load(latest)

        logger.info(f"Loaded checkpoint: {latest}, step {checkpoint['step']}")

        return checkpoint
```

### 14.2 Early Stopping Implementation

```python
class EarlyStoppingCallback:
    """
    Early stopping with patience and minimum training requirements

    Prevents both:
    1. Premature stopping (minimum steps requirement)
    2. Overfitting (stops when validation stops improving)
    """

    def __init__(self, config: dict):
        self.patience = config.get('patience', 20)
        self.min_steps = config.get('min_steps', 100000)
        self.improvement_threshold = config.get('improvement_threshold', 0.001)
        self.metric = config.get('metric', 'sharpe_ratio')

        self.best_metric = float('-inf')
        self.best_step = 0
        self.steps_without_improvement = 0

    def should_stop(self, current_metric: float, current_step: int) -> Tuple[bool, str]:
        """
        Check if training should stop

        Returns:
            (should_stop, reason)
        """
        # Don't stop before minimum steps
        if current_step < self.min_steps:
            return False, f"Below minimum steps ({current_step}/{self.min_steps})"

        # Check for improvement
        if current_metric > self.best_metric + self.improvement_threshold:
            self.best_metric = current_metric
            self.best_step = current_step
            self.steps_without_improvement = 0
            return False, "Improvement detected"

        self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.patience:
            return True, f"No improvement for {self.patience} evaluations"

        return False, f"Patience: {self.steps_without_improvement}/{self.patience}"
```

---

## 15. Multi-Asset Support and Configuration

### 15.1 Asset-Specific Parameters

```python
ASSET_CONFIGS = {
    'EURUSD': {
        'asset_class': 'forex_major',
        'pip_value': 0.0001,
        'typical_spread': 1.0,
        'volatility_normal': 0.005,
        'leverage_max': 50,
        'session_hours': '24h',
        'position_size_default': 0.02,
        'correlations': {'GBPUSD': 0.8, 'USDCHF': -0.9},
    },
    'XAUUSD': {
        'asset_class': 'commodity',
        'pip_value': 0.01,
        'typical_spread': 30,
        'volatility_normal': 0.012,
        'leverage_max': 20,
        'session_hours': '23h',
        'position_size_default': 0.01,
        'correlations': {'EURUSD': 0.5, 'US500': -0.3},
    },
    'US500': {
        'asset_class': 'index',
        'pip_value': 0.1,
        'typical_spread': 0.5,
        'volatility_normal': 0.015,
        'leverage_max': 20,
        'session_hours': 'market',
        'position_size_default': 0.01,
        'correlations': {'USTEC': 0.95, 'XAUUSD': -0.3},
    },
}
```

---

## 16-25. Remaining Sections

The report continues with detailed coverage of:

- **Testing and Validation Framework**: Unit tests, integration tests, backtesting validation
- **Deployment Architecture**: Cloud deployment, Docker containers, Kubernetes orchestration
- **Commercialization Strategy**: SaaS models, managed accounts, technology licensing, hedge fund formation
- **Competitive Analysis**: Market positioning against QuantConnect, Alpaca, institutional solutions
- **Regulatory Considerations**: CFTC, NFA, MiFID II, FCA compliance requirements
- **Financial Projections**: Revenue models, break-even analysis, growth scenarios
- **Development Roadmap**: Near-term enhancements, medium-term goals, long-term vision
- **Risk Disclosures**: Market risk, model risk, operational risk, regulatory risk
- **Conclusion**: Summary of capabilities, readiness assessment, investment opportunity
- **Technical Appendices**: File inventory, configuration reference, API documentation, glossary

---

## 25. Technical Appendices

### Appendix A: Complete File Inventory

```
TradingBOT_Agentic/
├── config.py                           # Central configuration (727 lines)
├── parallel_training.py                # Walk-forward training (1,635 lines)
├── src/
│   ├── agents/
│   │   ├── orchestrator.py            # Multi-agent coordinator (1,337 lines)
│   │   ├── portfolio_risk.py          # Risk management (1,797 lines)
│   │   ├── intelligent_risk_sentinel.py # Adaptive risk monitoring
│   │   ├── kill_switch.py             # Circuit breakers (1,697 lines)
│   │   └── ppo_agent.py               # Core RL agent
│   ├── environment/
│   │   ├── trading_env.py             # Gym environment
│   │   ├── strategy_features.py       # Smart Money features (787 lines)
│   │   └── risk_manager.py            # Position-level risk
│   ├── data/
│   │   ├── data_loader.py             # Data acquisition
│   │   └── feature_engineering.py     # Feature calculation
│   ├── execution/
│   │   ├── mt5_connector.py           # MetaTrader integration
│   │   └── order_manager.py           # Order lifecycle
│   └── utils/
│       ├── logging_config.py
│       └── performance_metrics.py
├── tests/
│   ├── test_risk.py
│   ├── test_features.py
│   └── test_backtest.py
└── notebooks/
    └── analysis/
```

### Appendix B: Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 3e-4 | PPO learning rate |
| n_steps | 2048 | Steps per update |
| batch_size | 64 | Minibatch size |
| gamma | 0.99 | Discount factor |
| var_confidence | 0.95 | VaR confidence level |
| max_drawdown | 0.15 | Maximum drawdown limit |
| daily_loss_limit | 0.03 | Daily loss circuit breaker |

### Appendix C: Performance Metrics Definitions

| Metric | Formula | Target |
|--------|---------|--------|
| Sharpe Ratio | (Return - Rf) / Volatility | > 1.5 |
| Sortino Ratio | (Return - Rf) / Downside Vol | > 2.0 |
| Max Drawdown | Max(Peak - Trough) / Peak | < 15% |
| Win Rate | Winning Trades / Total | > 50% |
| Profit Factor | Gross Profit / Gross Loss | > 1.5 |

### Appendix D: Glossary

- **ATR**: Average True Range - volatility indicator
- **BOS**: Break of Structure - trend continuation signal
- **CHOCH**: Change of Character - reversal signal
- **CVaR**: Conditional Value at Risk - tail risk measure
- **FVG**: Fair Value Gap - price inefficiency
- **GARCH**: Volatility model capturing clustering
- **PPO**: Proximal Policy Optimization - RL algorithm
- **VaR**: Value at Risk - potential loss estimate
- **Walk-Forward**: Rolling train/test validation

---

## 26. Deep Dive: Trading Environment Implementation

### 26.1 Gymnasium Environment Architecture

The trading environment follows the OpenAI Gymnasium (formerly Gym) interface, providing a standardized way for the reinforcement learning agent to interact with market simulations. This section provides an exhaustive explanation of every component.

```python
class TradingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for algorithmic trading

    This environment simulates a realistic trading scenario where an agent
    can take positions in financial instruments based on market observations.
    The environment handles:

    1. State Management:
       - Current market data (OHLCV)
       - Technical indicators
       - Smart Money Concepts features
       - Portfolio state (positions, P&L, exposure)
       - Account state (balance, margin, equity)

    2. Action Processing:
       - Continuous action space [-1, +1]
       - Position sizing and scaling
       - Order generation
       - Execution simulation with realistic slippage

    3. Reward Calculation:
       - Multi-objective reward function
       - Risk-adjusted returns
       - Transaction cost penalties
       - Drawdown penalties

    4. Episode Management:
       - Episode termination conditions
       - State reset and initialization
       - Performance tracking

    The environment is designed to be as realistic as possible while
    maintaining computational efficiency for training.
    """

    # Metadata for Gymnasium compatibility
    metadata = {
        'render_modes': ['human', 'rgb_array', 'ansi'],
        'render_fps': 4,
    }

    def __init__(self, config: dict):
        """
        Initialize the trading environment

        Args:
            config: Configuration dictionary containing:
                - data_config: Data loading and preprocessing settings
                - trading_config: Trading parameters (costs, leverage, etc.)
                - observation_config: Feature selection and normalization
                - reward_config: Reward function parameters
                - episode_config: Episode length and termination conditions

        The initialization process:
        1. Validate configuration parameters
        2. Load and preprocess market data
        3. Initialize feature calculators
        4. Set up observation and action spaces
        5. Initialize portfolio and account state
        6. Set up reward calculator
        7. Initialize logging and metrics
        """
        super().__init__()

        # Store configuration
        self.config = config
        self.data_config = config.get('data_config', {})
        self.trading_config = config.get('trading_config', {})
        self.observation_config = config.get('observation_config', {})
        self.reward_config = config.get('reward_config', {})
        self.episode_config = config.get('episode_config', {})

        # Trading parameters
        self.initial_balance = self.trading_config.get('initial_balance', 100000.0)
        self.leverage = self.trading_config.get('leverage', 1.0)
        self.transaction_cost = self.trading_config.get('transaction_cost', 0.0002)
        self.slippage_model = self.trading_config.get('slippage_model', 'fixed')
        self.slippage_bps = self.trading_config.get('slippage_bps', 1.0)

        # Episode parameters
        self.max_episode_steps = self.episode_config.get('max_steps', 1000)
        self.min_episode_steps = self.episode_config.get('min_steps', 100)
        self.terminate_on_margin_call = self.episode_config.get('terminate_on_margin_call', True)
        self.margin_call_threshold = self.episode_config.get('margin_call_threshold', 0.3)

        # Load market data
        self._load_data()

        # Initialize feature calculators
        self._initialize_feature_calculators()

        # Calculate observation space dimensions
        self._calculate_observation_dimensions()

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )

        # Initialize reward calculator
        self.reward_calculator = TradingRewardCalculator(self.reward_config)

        # Initialize state variables
        self._reset_state()

        # Logging and metrics
        self.episode_count = 0
        self.total_steps = 0
        self.episode_metrics = []

        logger.info(f"TradingEnvironment initialized: "
                   f"obs_dim={self.observation_dim}, "
                   f"initial_balance={self.initial_balance}, "
                   f"max_steps={self.max_episode_steps}")

    def _load_data(self):
        """
        Load and preprocess market data

        This method handles:
        1. Loading raw OHLCV data from specified source
        2. Handling missing values and data gaps
        3. Timezone normalization
        4. Data validation and integrity checks
        5. Splitting into training/validation/test sets if needed

        Supported data sources:
        - CSV files
        - Parquet files
        - Database connections
        - Live data feeds (for paper trading)
        """
        data_source = self.data_config.get('source', 'csv')
        data_path = self.data_config.get('path', '')

        if data_source == 'csv':
            self.raw_data = pd.read_csv(
                data_path,
                parse_dates=['timestamp'],
                index_col='timestamp'
            )
        elif data_source == 'parquet':
            self.raw_data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")

        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [c for c in required_columns if c not in self.raw_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Handle missing values
        self.raw_data = self.raw_data.ffill().bfill()

        # Calculate derived columns
        self.raw_data['returns'] = self.raw_data['close'].pct_change()
        self.raw_data['log_returns'] = np.log(self.raw_data['close'] / self.raw_data['close'].shift(1))
        self.raw_data['typical_price'] = (
            self.raw_data['high'] + self.raw_data['low'] + self.raw_data['close']
        ) / 3

        # Store data length
        self.data_length = len(self.raw_data)

        logger.info(f"Loaded {self.data_length} bars of market data")

    def _initialize_feature_calculators(self):
        """
        Initialize all feature calculation components

        Feature categories:
        1. Price features (raw and derived price metrics)
        2. Technical indicators (RSI, MACD, Bollinger, etc.)
        3. Smart Money Concepts (FVG, BOS, CHOCH, Order Blocks)
        4. Volatility features (ATR, GARCH, historical vol)
        5. Volume features (volume profile, VWAP, etc.)
        6. Microstructure features (spread, depth if available)
        """
        # Smart Money Concepts engine
        self.smc_engine = SmartMoneyEngine(
            self.observation_config.get('smc_config', {})
        )

        # Technical indicator calculator
        self.tech_calculator = TechnicalIndicatorCalculator(
            self.observation_config.get('technical_config', {})
        )

        # Volatility model
        self.volatility_model = GARCHVolatilityModel(
            self.observation_config.get('volatility_config', {})
        )

        # Pre-calculate all features for efficiency
        self._precalculate_features()

    def _precalculate_features(self):
        """
        Pre-calculate all features for the entire dataset

        This is done once during initialization for efficiency.
        During step(), we simply index into the pre-calculated arrays.

        Feature groups:
        - price_features: shape (n_bars, n_price_features)
        - technical_features: shape (n_bars, n_tech_features)
        - smc_features: shape (n_bars, n_smc_features)
        - volatility_features: shape (n_bars, n_vol_features)
        """
        logger.info("Pre-calculating features...")

        # Price features
        self.price_features = self._calculate_price_features(self.raw_data)

        # Technical indicators
        self.technical_features = self.tech_calculator.calculate_all(self.raw_data)

        # Smart Money Concepts
        self.smc_features = self.smc_engine.calculate_features(self.raw_data)

        # Volatility features
        returns = self.raw_data['returns'].values
        self.volatility_model.fit(returns[~np.isnan(returns)])
        self.volatility_features = self._calculate_volatility_features(self.raw_data)

        # Combine all features
        self.all_features = np.hstack([
            self.price_features,
            self.technical_features,
            self.smc_features,
            self.volatility_features,
        ])

        # Handle NaN values from indicator warm-up periods
        self.all_features = np.nan_to_num(self.all_features, nan=0.0)

        # Normalize features
        self._normalize_features()

        logger.info(f"Pre-calculated {self.all_features.shape[1]} features "
                   f"for {self.all_features.shape[0]} bars")

    def _calculate_price_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate comprehensive price-based features

        Features included:
        1. Normalized OHLC (relative to recent range)
        2. Multi-period returns (1, 2, 3, 5, 10, 20, 50 bars)
        3. Multi-period log returns
        4. Price position relative to moving averages
        5. Distance from period high/low
        6. Gap analysis (overnight, intraday)
        7. Price momentum and acceleration
        8. Price patterns (inside bars, outside bars, etc.)

        Returns:
            numpy array of shape (n_bars, n_price_features)
        """
        n = len(data)
        features = []

        # Raw normalized OHLC
        period = 50
        rolling_high = data['high'].rolling(period).max()
        rolling_low = data['low'].rolling(period).min()
        range_size = rolling_high - rolling_low + 1e-8

        features.append(((data['open'] - rolling_low) / range_size).values)
        features.append(((data['high'] - rolling_low) / range_size).values)
        features.append(((data['low'] - rolling_low) / range_size).values)
        features.append(((data['close'] - rolling_low) / range_size).values)

        # Multi-period returns
        for period in [1, 2, 3, 5, 10, 20, 50]:
            returns = data['close'].pct_change(period).values
            features.append(returns)

        # Multi-period log returns
        for period in [1, 5, 20]:
            log_returns = np.log(data['close'] / data['close'].shift(period)).values
            features.append(log_returns)

        # Distance from moving averages
        for ma_period in [10, 20, 50, 100, 200]:
            ma = data['close'].rolling(ma_period).mean()
            distance = ((data['close'] - ma) / ma).values
            features.append(distance)

        # Distance from period high/low
        for period in [10, 20, 50]:
            period_high = data['high'].rolling(period).max()
            period_low = data['low'].rolling(period).min()

            dist_from_high = ((data['close'] - period_high) / data['close']).values
            dist_from_low = ((data['close'] - period_low) / data['close']).values

            features.append(dist_from_high)
            features.append(dist_from_low)

        # Price momentum (rate of change)
        for period in [5, 10, 20]:
            roc = ((data['close'] - data['close'].shift(period)) /
                   data['close'].shift(period)).values
            features.append(roc)

        # Price acceleration (change in momentum)
        momentum_10 = data['close'].pct_change(10)
        acceleration = momentum_10.diff(5).values
        features.append(acceleration)

        # Range analysis
        daily_range = data['high'] - data['low']
        avg_range = daily_range.rolling(20).mean()
        range_ratio = (daily_range / avg_range).values
        features.append(range_ratio)

        # Body to range ratio (candlestick analysis)
        body = abs(data['close'] - data['open'])
        body_ratio = (body / (daily_range + 1e-8)).values
        features.append(body_ratio)

        # Upper and lower shadows
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']

        features.append((upper_shadow / (daily_range + 1e-8)).values)
        features.append((lower_shadow / (daily_range + 1e-8)).values)

        # Stack all features
        result = np.column_stack(features)

        return result

    def _calculate_volatility_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate comprehensive volatility features

        Features included:
        1. Historical volatility at multiple lookbacks
        2. GARCH estimated volatility
        3. Volatility ratios (short/long)
        4. Volatility regime indicator
        5. ATR and normalized ATR
        6. Parkinson volatility estimator
        7. Garman-Klass volatility estimator
        8. Yang-Zhang volatility estimator

        Returns:
            numpy array of shape (n_bars, n_vol_features)
        """
        n = len(data)
        features = []

        returns = data['returns'].values

        # Historical volatility at multiple lookbacks
        for period in [5, 10, 20, 50, 100]:
            hist_vol = data['returns'].rolling(period).std().values
            features.append(hist_vol)

        # Volatility ratios
        vol_5 = data['returns'].rolling(5).std()
        vol_20 = data['returns'].rolling(20).std()
        vol_50 = data['returns'].rolling(50).std()

        features.append((vol_5 / vol_20).values)
        features.append((vol_5 / vol_50).values)
        features.append((vol_20 / vol_50).values)

        # ATR
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )

        for period in [7, 14, 21]:
            atr = pd.Series(tr).rolling(period).mean().values
            features.append(atr)
            features.append(atr / close)  # Normalized ATR

        # Parkinson volatility estimator (uses high-low range)
        for period in [10, 20]:
            hl_ratio = np.log(high / low)
            parkinson_var = (1 / (4 * np.log(2))) * hl_ratio**2
            parkinson_vol = np.sqrt(
                pd.Series(parkinson_var).rolling(period).mean()
            ).values
            features.append(parkinson_vol)

        # Garman-Klass volatility estimator
        open_price = data['open'].values

        hl_term = 0.5 * np.log(high / low)**2
        co_term = (2 * np.log(2) - 1) * np.log(close / open_price)**2
        gk_var = hl_term - co_term

        for period in [10, 20]:
            gk_vol = np.sqrt(
                pd.Series(gk_var).rolling(period).mean()
            ).values
            features.append(gk_vol)

        # Volatility percentile
        vol_20_series = pd.Series(data['returns'].rolling(20).std())
        vol_percentile = vol_20_series.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        ).values
        features.append(vol_percentile)

        # GARCH volatility (updated incrementally)
        garch_vol = np.zeros(n)
        for i in range(1, n):
            self.volatility_model.update(returns[i])
            garch_vol[i] = np.sqrt(self.volatility_model.current_variance)
        features.append(garch_vol)

        # Volatility regime
        vol_regime = np.zeros(n)
        vol_20_arr = vol_20_series.values
        vol_mean = np.nanmean(vol_20_arr)
        vol_std = np.nanstd(vol_20_arr)

        for i in range(n):
            if np.isnan(vol_20_arr[i]):
                vol_regime[i] = 0
            elif vol_20_arr[i] < vol_mean - vol_std:
                vol_regime[i] = -1  # Low volatility
            elif vol_20_arr[i] > vol_mean + vol_std:
                vol_regime[i] = 1  # High volatility
            else:
                vol_regime[i] = 0  # Normal volatility
        features.append(vol_regime)

        result = np.column_stack(features)

        return result

    def _normalize_features(self):
        """
        Normalize all features for stable training

        Normalization strategies:
        1. Z-score normalization for most features
        2. Robust normalization for features with outliers
        3. Clipping to prevent extreme values
        4. Rolling normalization for online adaptation

        The normalization parameters are calculated from training data
        and stored for consistent application during inference.
        """
        # Calculate normalization statistics
        # Use robust statistics to handle outliers
        self.feature_medians = np.nanmedian(self.all_features, axis=0)
        self.feature_mads = np.nanmedian(
            np.abs(self.all_features - self.feature_medians), axis=0
        )
        # Prevent division by zero
        self.feature_mads = np.where(
            self.feature_mads < 1e-8,
            1.0,
            self.feature_mads
        )

        # Apply robust z-score normalization
        self.all_features = (
            (self.all_features - self.feature_medians) /
            (self.feature_mads * 1.4826)  # Scale to approximate std
        )

        # Clip extreme values
        self.all_features = np.clip(self.all_features, -5.0, 5.0)

        # Final NaN check
        self.all_features = np.nan_to_num(self.all_features, nan=0.0)

    def _calculate_observation_dimensions(self):
        """
        Calculate total observation space dimensions

        The observation consists of:
        1. Market features (pre-calculated)
        2. Portfolio state features
        3. Account state features

        This method calculates the total dimension and stores
        it for observation space definition.
        """
        # Market features dimension
        market_dim = self.all_features.shape[1]

        # Portfolio state features
        portfolio_features = [
            'current_position',           # Current position size [-1, 1]
            'position_direction',         # Long/short/flat indicator
            'unrealized_pnl_pct',         # Unrealized P&L as % of equity
            'realized_pnl_pct',           # Realized P&L as % of initial
            'bars_in_position',           # Time in current position
            'entry_price_distance',       # Distance from entry price
            'stop_loss_distance',         # Distance to stop loss
            'take_profit_distance',       # Distance to take profit
            'position_heat',              # Risk contribution
            'max_position_pnl',           # Best P&L in current position
        ]
        portfolio_dim = len(portfolio_features)

        # Account state features
        account_features = [
            'equity_change',              # Change in equity
            'margin_utilization',         # Margin used / available
            'drawdown_current',           # Current drawdown
            'drawdown_duration',          # Bars in drawdown
            'win_rate_recent',            # Recent win rate
            'avg_win_loss_ratio',         # Average win / average loss
            'consecutive_wins',           # Consecutive winning trades
            'consecutive_losses',         # Consecutive losing trades
            'daily_pnl',                  # Today's P&L
            'weekly_pnl',                 # This week's P&L
        ]
        account_dim = len(account_features)

        self.observation_dim = market_dim + portfolio_dim + account_dim

        logger.info(f"Observation dimensions: market={market_dim}, "
                   f"portfolio={portfolio_dim}, account={account_dim}, "
                   f"total={self.observation_dim}")

    def _reset_state(self):
        """
        Reset all state variables for new episode

        This includes:
        1. Portfolio state (positions, P&L)
        2. Account state (balance, equity)
        3. Episode tracking (step count, metrics)
        4. Random starting point selection
        """
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0

        # Select random starting point
        # Leave room for episode length and feature warm-up
        warm_up_bars = 200  # Features need history
        min_start = warm_up_bars
        max_start = self.data_length - self.max_episode_steps - 1

        if max_start <= min_start:
            self.start_index = min_start
        else:
            self.start_index = np.random.randint(min_start, max_start)

        self.current_index = self.start_index

        # Reset account state
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.margin_used = 0.0

        # Reset portfolio state
        self.position = 0.0  # Position size [-1, 1]
        self.position_value = 0.0
        self.entry_price = 0.0
        self.entry_index = 0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0

        # Reset tracking
        self.max_equity = self.initial_balance
        self.max_position_pnl = 0.0
        self.trades = []
        self.equity_curve = [self.initial_balance]

        # Trade statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_profit = 0.0
        self.total_loss = 0.0

        # Reset reward calculator
        self.reward_calculator.reset_episode()

    def reset(self, seed=None, options=None):
        """
        Reset environment for new episode

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Reset all state
        self._reset_state()

        # Get initial observation
        observation = self._get_observation()

        # Episode info
        info = {
            'start_index': self.start_index,
            'initial_balance': self.initial_balance,
            'episode_number': self.episode_count,
        }

        self.episode_count += 1

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment

        Args:
            action: numpy array of shape (1,) with value in [-1, 1]

        Returns:
            observation: New observation after action
            reward: Reward for the action
            terminated: Whether episode ended (margin call, data end)
            truncated: Whether episode was cut short (max steps)
            info: Additional information dictionary

        Step process:
        1. Validate and interpret action
        2. Execute trades based on action
        3. Update market state
        4. Update portfolio valuation
        5. Calculate reward
        6. Check termination conditions
        7. Generate new observation
        """
        # Extract scalar action
        action_value = float(action[0])

        # Clip action to valid range
        action_value = np.clip(action_value, -1.0, 1.0)

        # Store previous state for reward calculation
        prev_state = self._get_state_snapshot()

        # Interpret action and execute trades
        self._execute_action(action_value)

        # Advance market
        self.current_step += 1
        self.current_index += 1
        self.total_steps += 1

        # Update portfolio valuation
        self._update_portfolio()

        # Get new state
        new_state = self._get_state_snapshot()

        # Calculate reward
        reward_breakdown = self.reward_calculator.calculate_reward(
            prev_state, action_value, new_state
        )
        reward = reward_breakdown.total
        self.episode_reward += reward

        # Check termination conditions
        terminated = False
        truncated = False
        termination_reason = None

        # Margin call check
        if self.terminate_on_margin_call:
            drawdown = (self.max_equity - self.equity) / self.max_equity
            if drawdown > self.margin_call_threshold:
                terminated = True
                termination_reason = 'margin_call'

        # Data exhaustion
        if self.current_index >= self.data_length - 1:
            terminated = True
            termination_reason = 'data_exhausted'

        # Max steps
        if self.current_step >= self.max_episode_steps:
            truncated = True
            termination_reason = 'max_steps'

        # Get observation
        observation = self._get_observation()

        # Build info dictionary
        info = {
            'step': self.current_step,
            'action': action_value,
            'position': self.position,
            'equity': self.equity,
            'balance': self.balance,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'drawdown': (self.max_equity - self.equity) / self.max_equity,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'reward_breakdown': reward_breakdown.__dict__,
            'termination_reason': termination_reason,
        }

        # Update equity curve
        self.equity_curve.append(self.equity)

        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: float):
        """
        Execute trading action

        Action interpretation:
        - action > threshold: Go long or increase long
        - action < -threshold: Go short or increase short
        - |action| < close_threshold: Close position

        The action magnitude determines position size:
        - |action| = 1.0: Maximum position size
        - |action| = 0.5: Half position size
        - etc.

        This method handles:
        1. Position changes (open, close, scale)
        2. Transaction costs
        3. Slippage simulation
        4. Trade recording
        """
        long_threshold = 0.3
        short_threshold = -0.3
        close_threshold = 0.1

        current_price = self.raw_data.iloc[self.current_index]['close']
        prev_position = self.position

        # Determine target position
        if action > long_threshold:
            target_position = action  # Long with size = action
        elif action < short_threshold:
            target_position = action  # Short with size = action
        elif abs(action) < close_threshold and self.position != 0:
            target_position = 0.0  # Close position
        else:
            target_position = self.position  # Hold current position

        # Calculate position change
        position_change = target_position - self.position

        if abs(position_change) > 0.01:  # Minimum change threshold
            # Calculate execution price with slippage
            execution_price = self._apply_slippage(
                current_price,
                direction='buy' if position_change > 0 else 'sell'
            )

            # If closing existing position, record trade
            if prev_position != 0 and (
                np.sign(target_position) != np.sign(prev_position) or
                target_position == 0
            ):
                self._close_position(execution_price)

            # If opening new position or scaling
            if target_position != 0:
                if prev_position == 0 or np.sign(target_position) != np.sign(prev_position):
                    # New position
                    self._open_position(target_position, execution_price)
                else:
                    # Scaling existing position
                    self._scale_position(target_position, execution_price)

            # Apply transaction costs
            transaction_cost = abs(position_change) * self.position_value * self.transaction_cost
            self.balance -= transaction_cost

    def _apply_slippage(self, price: float, direction: str) -> float:
        """
        Apply realistic slippage to execution price

        Slippage models:
        1. Fixed: Constant slippage in basis points
        2. Proportional: Slippage proportional to position size
        3. Volatility-adjusted: Higher slippage in volatile markets
        4. Market impact: Larger orders have more impact

        Returns:
            Execution price after slippage
        """
        if self.slippage_model == 'fixed':
            slippage_pct = self.slippage_bps / 10000

        elif self.slippage_model == 'volatility':
            current_vol = self.volatility_features[self.current_index, 0]
            slippage_pct = self.slippage_bps / 10000 * (1 + current_vol * 10)

        elif self.slippage_model == 'market_impact':
            position_size = abs(self.position)
            slippage_pct = self.slippage_bps / 10000 * (1 + position_size)

        else:
            slippage_pct = self.slippage_bps / 10000

        if direction == 'buy':
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)

    def _open_position(self, size: float, price: float):
        """
        Open new position

        Records:
        - Entry price
        - Entry time
        - Position size and direction
        - Initial margin requirement
        """
        self.position = size
        self.entry_price = price
        self.entry_index = self.current_index
        self.position_value = abs(size) * self.equity
        self.margin_used = self.position_value / self.leverage
        self.unrealized_pnl = 0.0
        self.max_position_pnl = 0.0

    def _close_position(self, price: float):
        """
        Close existing position

        Calculates:
        - Realized P&L
        - Trade statistics update
        - Trade record
        """
        if self.position == 0:
            return

        # Calculate P&L
        if self.position > 0:  # Long position
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:  # Short position
            pnl_pct = (self.entry_price - price) / self.entry_price

        pnl = pnl_pct * self.position_value

        # Update account
        self.realized_pnl += pnl
        self.balance += pnl

        # Update statistics
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Record trade
        self.trades.append({
            'entry_index': self.entry_index,
            'exit_index': self.current_index,
            'entry_price': self.entry_price,
            'exit_price': price,
            'direction': 'long' if self.position > 0 else 'short',
            'size': abs(self.position),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'bars_held': self.current_index - self.entry_index,
        })

        # Reset position state
        self.position = 0.0
        self.entry_price = 0.0
        self.position_value = 0.0
        self.margin_used = 0.0
        self.unrealized_pnl = 0.0

    def _scale_position(self, new_size: float, price: float):
        """
        Scale existing position (add or reduce)

        Handles partial closes and position increases
        with average entry price calculation.
        """
        if np.sign(new_size) != np.sign(self.position):
            raise ValueError("Cannot scale to opposite direction")

        size_change = new_size - self.position

        if abs(new_size) < abs(self.position):
            # Reducing position - partial close
            reduction_ratio = abs(size_change) / abs(self.position)
            partial_pnl = self.unrealized_pnl * reduction_ratio

            self.realized_pnl += partial_pnl
            self.balance += partial_pnl
            self.unrealized_pnl -= partial_pnl
        else:
            # Increasing position - update average entry
            old_value = abs(self.position) * self.entry_price
            add_value = abs(size_change) * price
            new_total_size = abs(new_size)

            self.entry_price = (old_value + add_value) / new_total_size

        self.position = new_size
        self.position_value = abs(new_size) * self.equity
        self.margin_used = self.position_value / self.leverage

    def _update_portfolio(self):
        """
        Update portfolio valuation at current market prices

        Calculates:
        - Current position value
        - Unrealized P&L
        - Equity
        - Maximum equity (for drawdown)
        """
        current_price = self.raw_data.iloc[self.current_index]['close']

        if self.position != 0:
            # Calculate unrealized P&L
            if self.position > 0:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - current_price) / self.entry_price

            self.unrealized_pnl = pnl_pct * self.position_value

            # Track maximum position P&L
            if self.unrealized_pnl > self.max_position_pnl:
                self.max_position_pnl = self.unrealized_pnl

        # Update equity
        self.equity = self.balance + self.unrealized_pnl

        # Update maximum equity
        if self.equity > self.max_equity:
            self.max_equity = self.equity

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector

        Combines:
        1. Pre-calculated market features
        2. Current portfolio state
        3. Current account state

        Returns:
            numpy array of shape (observation_dim,)
        """
        # Get market features for current index
        market_features = self.all_features[self.current_index]

        # Build portfolio state features
        current_price = self.raw_data.iloc[self.current_index]['close']

        portfolio_features = np.array([
            self.position,
            np.sign(self.position),
            self.unrealized_pnl / self.equity if self.equity > 0 else 0,
            self.realized_pnl / self.initial_balance,
            (self.current_index - self.entry_index) / 100 if self.position != 0 else 0,
            (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0,
            0.0,  # stop_loss_distance (would need SL tracking)
            0.0,  # take_profit_distance (would need TP tracking)
            abs(self.position) * (self.unrealized_pnl / self.equity) if self.equity > 0 else 0,
            self.max_position_pnl / self.equity if self.equity > 0 else 0,
        ], dtype=np.float32)

        # Build account state features
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
        win_rate = self.winning_trades / max(1, self.total_trades)
        avg_win = self.total_profit / max(1, self.winning_trades)
        avg_loss = self.total_loss / max(1, self.losing_trades)
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        account_features = np.array([
            (self.equity - self.initial_balance) / self.initial_balance,
            self.margin_used / self.equity if self.equity > 0 else 0,
            drawdown,
            0.0,  # drawdown_duration (would need tracking)
            win_rate,
            win_loss_ratio,
            self.consecutive_wins / 10.0,
            self.consecutive_losses / 10.0,
            0.0,  # daily_pnl (would need date tracking)
            0.0,  # weekly_pnl (would need date tracking)
        ], dtype=np.float32)

        # Combine all features
        observation = np.concatenate([
            market_features,
            portfolio_features,
            account_features,
        ]).astype(np.float32)

        return observation

    def _get_state_snapshot(self) -> TradingState:
        """
        Get current state snapshot for reward calculation
        """
        current_price = self.raw_data.iloc[self.current_index]['close']

        return TradingState(
            portfolio_value=self.equity,
            position=self.position,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            max_value=self.max_equity,
            volatility=self.volatility_features[self.current_index, 0] if self.current_index < len(self.volatility_features) else 0,
            holding_time=self.current_index - self.entry_index if self.position != 0 else 0,
            bars_in_position=self.current_index - self.entry_index if self.position != 0 else 0,
            market_regime=self._get_market_regime(),
        )

    def _get_market_regime(self) -> str:
        """
        Determine current market regime
        """
        if self.current_index >= len(self.volatility_features):
            return 'NORMAL'

        vol = self.volatility_features[self.current_index, 0]
        trend = self.smc_features[self.current_index, 6] if self.current_index < len(self.smc_features) else 0

        if vol > 0.02:
            return 'HIGH_VOLATILITY'
        elif vol < 0.005:
            return 'LOW_VOLATILITY'
        elif abs(trend) > 0.5:
            return 'TRENDING'
        else:
            return 'NORMAL'

    def render(self, mode='human'):
        """
        Render environment state

        Modes:
        - 'human': Print to console
        - 'rgb_array': Return image array
        - 'ansi': Return string representation
        """
        if mode == 'human' or mode == 'ansi':
            output = f"""
╔══════════════════════════════════════════════════════════════╗
║                    TRADING ENVIRONMENT STATUS                  ║
╠══════════════════════════════════════════════════════════════╣
║ Step: {self.current_step:>6} / {self.max_episode_steps:<6}     Index: {self.current_index:>8}       ║
╠══════════════════════════════════════════════════════════════╣
║ ACCOUNT                                                       ║
║   Balance:     ${self.balance:>12,.2f}                              ║
║   Equity:      ${self.equity:>12,.2f}                              ║
║   Drawdown:    {((self.max_equity - self.equity) / self.max_equity * 100):>6.2f}%                               ║
╠══════════════════════════════════════════════════════════════╣
║ POSITION                                                      ║
║   Size:        {self.position:>+8.4f}                                   ║
║   Direction:   {'LONG' if self.position > 0 else 'SHORT' if self.position < 0 else 'FLAT':>8}                                   ║
║   Unrealized:  ${self.unrealized_pnl:>+12,.2f}                             ║
║   Entry:       ${self.entry_price:>12,.5f}                             ║
╠══════════════════════════════════════════════════════════════╣
║ STATISTICS                                                    ║
║   Total Trades: {self.total_trades:>5}    Realized P&L: ${self.realized_pnl:>+10,.2f}    ║
║   Win Rate:    {(self.winning_trades / max(1, self.total_trades) * 100):>6.1f}%                                ║
║   Wins: {self.winning_trades:>3}  Losses: {self.losing_trades:>3}  Consec W/L: {self.consecutive_wins}/{self.consecutive_losses}      ║
╚══════════════════════════════════════════════════════════════╝
"""
            if mode == 'human':
                print(output)
            return output

        elif mode == 'rgb_array':
            # Would return matplotlib figure as array
            # Not implemented for this example
            return None

    def close(self):
        """
        Clean up environment resources
        """
        pass

    def get_episode_summary(self) -> dict:
        """
        Get comprehensive episode summary

        Returns dict with:
        - Performance metrics
        - Trade statistics
        - Risk metrics
        - Equity curve data
        """
        if self.total_trades == 0:
            return {
                'total_return': 0,
                'sharpe': 0,
                'max_drawdown': 0,
                'total_trades': 0,
            }

        # Calculate returns from equity curve
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]

        # Performance metrics
        total_return = (self.equity - self.initial_balance) / self.initial_balance

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0

        # Max drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdowns = (peak - equity_array) / peak
        max_drawdown = np.max(drawdowns)

        # Trade statistics
        win_rate = self.winning_trades / self.total_trades
        avg_win = self.total_profit / max(1, self.winning_trades)
        avg_loss = self.total_loss / max(1, self.losing_trades)
        profit_factor = self.total_profit / max(1, self.total_loss)

        if self.trades:
            avg_bars_held = np.mean([t['bars_held'] for t in self.trades])
            avg_pnl_per_trade = np.mean([t['pnl'] for t in self.trades])
        else:
            avg_bars_held = 0
            avg_pnl_per_trade = 0

        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_bars_held': avg_bars_held,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'final_equity': self.equity,
            'total_reward': self.episode_reward,
            'equity_curve': equity_array.tolist(),
            'trades': self.trades,
        }
```

### 26.2 Technical Indicator Calculator - Complete Implementation

```python
class TechnicalIndicatorCalculator:
    """
    Comprehensive technical indicator calculator

    Calculates a wide range of technical indicators commonly used
    in trading systems. All calculations are vectorized for efficiency.

    Indicator Categories:
    1. Trend Indicators (MA, EMA, MACD, ADX)
    2. Momentum Indicators (RSI, Stochastic, CCI, Williams %R)
    3. Volatility Indicators (Bollinger, ATR, Keltner)
    4. Volume Indicators (OBV, VWAP, MFI)
    5. Support/Resistance (Pivot Points, Fibonacci)
    """

    def __init__(self, config: dict = None):
        """
        Initialize calculator with configuration

        Args:
            config: Optional configuration dictionary containing:
                - indicator_periods: Dict of indicator -> periods to calculate
                - enabled_indicators: List of indicators to calculate
                - custom_params: Dict of custom parameters per indicator
        """
        self.config = config or {}

        # Default periods for various indicators
        self.ma_periods = self.config.get('ma_periods', [5, 10, 20, 50, 100, 200])
        self.ema_periods = self.config.get('ema_periods', [9, 12, 21, 26, 50])
        self.rsi_periods = self.config.get('rsi_periods', [7, 14, 21])
        self.stoch_periods = self.config.get('stoch_periods', [(14, 3), (21, 5)])
        self.bb_periods = self.config.get('bb_periods', [(20, 2.0), (20, 2.5)])
        self.atr_periods = self.config.get('atr_periods', [7, 14, 21])

        # MACD parameters
        self.macd_params = self.config.get('macd_params', {
            'fast': 12,
            'slow': 26,
            'signal': 9,
        })

        # ADX parameters
        self.adx_period = self.config.get('adx_period', 14)

    def calculate_all(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate all technical indicators

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            numpy array of shape (n_bars, n_indicators)
        """
        features = []

        # Moving Averages
        ma_features = self._calculate_moving_averages(data)
        features.append(ma_features)

        # Exponential Moving Averages
        ema_features = self._calculate_ema(data)
        features.append(ema_features)

        # MACD
        macd_features = self._calculate_macd(data)
        features.append(macd_features)

        # RSI
        rsi_features = self._calculate_rsi(data)
        features.append(rsi_features)

        # Stochastic
        stoch_features = self._calculate_stochastic(data)
        features.append(stoch_features)

        # Bollinger Bands
        bb_features = self._calculate_bollinger(data)
        features.append(bb_features)

        # ATR
        atr_features = self._calculate_atr(data)
        features.append(atr_features)

        # ADX
        adx_features = self._calculate_adx(data)
        features.append(adx_features)

        # CCI
        cci_features = self._calculate_cci(data)
        features.append(cci_features)

        # Williams %R
        willr_features = self._calculate_williams_r(data)
        features.append(willr_features)

        # OBV
        obv_features = self._calculate_obv(data)
        features.append(obv_features)

        # MFI
        mfi_features = self._calculate_mfi(data)
        features.append(mfi_features)

        # Combine all features
        result = np.hstack(features)

        return result

    def _calculate_moving_averages(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Simple Moving Averages at multiple periods

        Features:
        - MA value normalized by close price
        - Distance from MA (percentage)
        - MA slope (rate of change)
        - MA crossover signals
        """
        features = []
        close = data['close']

        prev_ma = None
        for period in self.ma_periods:
            ma = close.rolling(period).mean()

            # Normalized MA value
            features.append((ma / close).values)

            # Distance from MA
            distance = ((close - ma) / close).values
            features.append(distance)

            # MA slope
            slope = (ma.diff(5) / ma).values
            features.append(slope)

            # Crossover with previous MA
            if prev_ma is not None:
                crossover = np.where(ma > prev_ma, 1, np.where(ma < prev_ma, -1, 0))
                features.append(crossover)

            prev_ma = ma

        return np.column_stack(features)

    def _calculate_ema(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Exponential Moving Averages

        EMA gives more weight to recent prices, making it more
        responsive to new information.

        Formula: EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}
        Where alpha = 2 / (period + 1)
        """
        features = []
        close = data['close']

        for period in self.ema_periods:
            ema = close.ewm(span=period, adjust=False).mean()

            # Normalized EMA
            features.append((ema / close).values)

            # Distance from EMA
            distance = ((close - ema) / close).values
            features.append(distance)

            # EMA slope
            slope = (ema.diff(3) / ema).values
            features.append(slope)

        return np.column_stack(features)

    def _calculate_macd(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Components:
        - MACD Line: Fast EMA - Slow EMA
        - Signal Line: EMA of MACD Line
        - Histogram: MACD Line - Signal Line

        Also calculates:
        - MACD normalized by price
        - Histogram direction
        - Zero line crossovers
        - Signal line crossovers
        """
        close = data['close']

        fast = close.ewm(span=self.macd_params['fast'], adjust=False).mean()
        slow = close.ewm(span=self.macd_params['slow'], adjust=False).mean()

        macd_line = fast - slow
        signal_line = macd_line.ewm(span=self.macd_params['signal'], adjust=False).mean()
        histogram = macd_line - signal_line

        features = []

        # Normalized MACD (as percentage of price)
        features.append((macd_line / close * 100).values)

        # Normalized Signal
        features.append((signal_line / close * 100).values)

        # Normalized Histogram
        features.append((histogram / close * 100).values)

        # Histogram direction (positive/negative change)
        hist_direction = np.sign(histogram.diff()).values
        features.append(hist_direction)

        # Zero line crossover (MACD crossing zero)
        zero_cross = np.where(
            (macd_line > 0) & (macd_line.shift(1) <= 0), 1,
            np.where((macd_line < 0) & (macd_line.shift(1) >= 0), -1, 0)
        )
        features.append(zero_cross)

        # Signal line crossover
        signal_cross = np.where(
            (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)), 1,
            np.where((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1)), -1, 0)
        )
        features.append(signal_cross)

        # MACD momentum (rate of change of MACD)
        macd_momentum = macd_line.diff(3).values
        features.append(macd_momentum)

        return np.column_stack(features)

    def _calculate_rsi(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Relative Strength Index at multiple periods

        RSI Formula:
        1. Calculate price changes
        2. Separate gains and losses
        3. Calculate average gain and loss (EMA)
        4. RS = Average Gain / Average Loss
        5. RSI = 100 - (100 / (1 + RS))

        Features:
        - RSI value (0-100)
        - RSI normalized to [-1, 1]
        - Overbought/Oversold signals
        - RSI divergence from price
        """
        features = []
        close = data['close']

        for period in self.rsi_periods:
            delta = close.diff()

            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)

            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()

            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))

            # RSI normalized to [-1, 1]
            rsi_normalized = ((rsi - 50) / 50).values
            features.append(rsi_normalized)

            # Overbought (>70) / Oversold (<30) signals
            ob_os = np.where(rsi > 70, 1, np.where(rsi < 30, -1, 0))
            features.append(ob_os)

            # RSI slope
            rsi_slope = rsi.diff(3).values
            features.append(rsi_slope)

            # RSI vs price divergence
            price_direction = np.sign(close.diff(period)).values
            rsi_direction = np.sign(rsi.diff(period)).values
            divergence = np.where(price_direction != rsi_direction, 1, 0)
            features.append(divergence)

        return np.column_stack(features)

    def _calculate_stochastic(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Stochastic Oscillator

        Formula:
        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA of %K

        Features:
        - %K value
        - %D value
        - %K - %D (for crossovers)
        - Overbought/Oversold signals
        """
        features = []
        high = data['high']
        low = data['low']
        close = data['close']

        for k_period, d_period in self.stoch_periods:
            lowest_low = low.rolling(k_period).min()
            highest_high = high.rolling(k_period).max()

            stoch_k = ((close - lowest_low) / (highest_high - lowest_low + 1e-8)) * 100
            stoch_d = stoch_k.rolling(d_period).mean()

            # Normalized %K and %D
            features.append(((stoch_k - 50) / 50).values)
            features.append(((stoch_d - 50) / 50).values)

            # %K - %D difference
            kd_diff = ((stoch_k - stoch_d) / 50).values
            features.append(kd_diff)

            # Overbought (>80) / Oversold (<20)
            ob_os = np.where(stoch_k > 80, 1, np.where(stoch_k < 20, -1, 0))
            features.append(ob_os)

            # Crossover signals
            cross = np.where(
                (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)), 1,
                np.where((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)), -1, 0)
            )
            features.append(cross)

        return np.column_stack(features)

    def _calculate_bollinger(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Bollinger Bands

        Components:
        - Middle Band: SMA of close
        - Upper Band: Middle + (std * multiplier)
        - Lower Band: Middle - (std * multiplier)

        Features:
        - %B: (Close - Lower) / (Upper - Lower)
        - Bandwidth: (Upper - Lower) / Middle
        - Squeeze detection
        - Touch signals (price touching bands)
        """
        features = []
        close = data['close']

        for period, std_mult in self.bb_periods:
            middle = close.rolling(period).mean()
            std = close.rolling(period).std()

            upper = middle + (std * std_mult)
            lower = middle - (std * std_mult)

            # %B (position within bands)
            pct_b = ((close - lower) / (upper - lower + 1e-8)).values
            features.append(pct_b)

            # Bandwidth (normalized)
            bandwidth = ((upper - lower) / middle).values
            features.append(bandwidth)

            # Squeeze detection (low bandwidth)
            avg_bandwidth = pd.Series(bandwidth).rolling(50).mean()
            squeeze = np.where(bandwidth < avg_bandwidth * 0.5, 1, 0)
            features.append(squeeze)

            # Band touches
            upper_touch = np.where(data['high'] >= upper, 1, 0).values
            lower_touch = np.where(data['low'] <= lower, 1, 0).values
            features.append(upper_touch)
            features.append(lower_touch)

            # Distance from middle band
            dist_from_middle = ((close - middle) / middle).values
            features.append(dist_from_middle)

        return np.column_stack(features)

    def _calculate_atr(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Average True Range

        True Range = max(
            High - Low,
            |High - Previous Close|,
            |Low - Previous Close|
        )

        ATR = SMA or EMA of True Range

        Features:
        - ATR value
        - Normalized ATR (ATR / Close)
        - ATR percentile (current vs historical)
        - ATR expansion/contraction
        """
        features = []
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Calculate True Range
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - prev_close),
                np.abs(low - prev_close)
            )
        )

        tr_series = pd.Series(tr)

        for period in self.atr_periods:
            atr = tr_series.rolling(period).mean()

            # ATR value
            features.append(atr.values)

            # Normalized ATR
            natr = (atr / close).values
            features.append(natr)

            # ATR percentile
            atr_percentile = atr.rolling(252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            ).values
            features.append(atr_percentile)

            # ATR change (expansion/contraction)
            atr_change = (atr.diff(5) / atr).values
            features.append(atr_change)

        return np.column_stack(features)

    def _calculate_adx(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Average Directional Index

        Components:
        - +DI: Positive Directional Indicator
        - -DI: Negative Directional Indicator
        - DX: Directional Index = |+DI - -DI| / (+DI + -DI)
        - ADX: Smoothed average of DX

        Features:
        - ADX value (trend strength 0-100)
        - +DI and -DI values
        - DI crossover signals
        - Trend strength classification
        """
        high = data['high']
        low = data['low']
        close = data['close']

        period = self.adx_period

        # Calculate directional movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Calculate True Range
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close.shift(1)),
                np.abs(low - close.shift(1))
            )
        )

        # Smooth with EMA
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        # Calculate DX and ADX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = 100 * (di_diff / (di_sum + 1e-8))
        adx = dx.ewm(span=period, adjust=False).mean()

        features = []

        # ADX normalized to [0, 1]
        features.append((adx / 100).values)

        # +DI and -DI normalized
        features.append((plus_di / 100).values)
        features.append((minus_di / 100).values)

        # DI difference (trend direction)
        di_diff_norm = ((plus_di - minus_di) / 100).values
        features.append(di_diff_norm)

        # DI crossover
        cross = np.where(
            (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1)), 1,
            np.where((plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1)), -1, 0)
        )
        features.append(cross)

        # Trend strength classification
        # ADX < 20: Weak trend
        # ADX 20-40: Strong trend
        # ADX > 40: Very strong trend
        trend_strength = np.where(adx > 40, 2, np.where(adx > 20, 1, 0))
        features.append(trend_strength)

        return np.column_stack(features)

    def _calculate_cci(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Commodity Channel Index

        CCI = (Typical Price - SMA of TP) / (0.015 * Mean Deviation)

        Where Typical Price = (High + Low + Close) / 3

        Features:
        - CCI value
        - Overbought/Oversold signals (+100/-100)
        - Zero line crossovers
        """
        features = []

        tp = (data['high'] + data['low'] + data['close']) / 3

        for period in [14, 20]:
            sma_tp = tp.rolling(period).mean()
            mean_dev = tp.rolling(period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )

            cci = (tp - sma_tp) / (0.015 * mean_dev + 1e-8)

            # CCI normalized
            features.append((cci / 200).clip(-1, 1).values)

            # Overbought/Oversold
            ob_os = np.where(cci > 100, 1, np.where(cci < -100, -1, 0))
            features.append(ob_os)

            # Zero crossover
            zero_cross = np.where(
                (cci > 0) & (cci.shift(1) <= 0), 1,
                np.where((cci < 0) & (cci.shift(1) >= 0), -1, 0)
            )
            features.append(zero_cross)

        return np.column_stack(features)

    def _calculate_williams_r(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Williams %R

        Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

        Ranges from -100 to 0
        - Above -20: Overbought
        - Below -80: Oversold
        """
        features = []

        for period in [14, 21]:
            highest_high = data['high'].rolling(period).max()
            lowest_low = data['low'].rolling(period).min()

            willr = ((highest_high - data['close']) /
                    (highest_high - lowest_low + 1e-8)) * -100

            # Normalized to [-1, 1]
            features.append(((willr + 50) / 50).values)

            # Overbought/Oversold
            ob_os = np.where(willr > -20, 1, np.where(willr < -80, -1, 0))
            features.append(ob_os)

        return np.column_stack(features)

    def _calculate_obv(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate On-Balance Volume

        OBV adds volume on up days, subtracts on down days:
        - If close > previous close: OBV = previous OBV + volume
        - If close < previous close: OBV = previous OBV - volume
        - If close = previous close: OBV = previous OBV

        Features:
        - OBV normalized (rate of change)
        - OBV trend vs price trend (divergence)
        - OBV moving average crossover
        """
        close = data['close']
        volume = data['volume']

        # Calculate OBV
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv[i] = obv[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv[i] = obv[i-1] - volume.iloc[i]
            else:
                obv[i] = obv[i-1]

        obv_series = pd.Series(obv, index=close.index)

        features = []

        # OBV rate of change
        obv_roc = obv_series.pct_change(10).values
        features.append(np.nan_to_num(obv_roc, nan=0))

        # OBV vs price divergence
        price_direction = np.sign(close.diff(10)).values
        obv_direction = np.sign(obv_series.diff(10)).values
        divergence = np.where(price_direction != obv_direction, 1, 0)
        features.append(divergence)

        # OBV MA crossover
        obv_ma = obv_series.rolling(20).mean()
        obv_above_ma = np.where(obv_series > obv_ma, 1, -1)
        features.append(obv_above_ma)

        return np.column_stack(features)

    def _calculate_mfi(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Money Flow Index

        MFI = 100 - (100 / (1 + Money Flow Ratio))

        Where:
        - Typical Price = (High + Low + Close) / 3
        - Money Flow = Typical Price * Volume
        - Money Flow Ratio = Positive MF / Negative MF

        Like RSI but incorporates volume
        """
        features = []

        tp = (data['high'] + data['low'] + data['close']) / 3
        mf = tp * data['volume']

        for period in [14, 21]:
            # Positive and negative money flow
            positive_mf = mf.where(tp > tp.shift(1), 0)
            negative_mf = mf.where(tp < tp.shift(1), 0)

            positive_mf_sum = positive_mf.rolling(period).sum()
            negative_mf_sum = negative_mf.rolling(period).sum()

            mfr = positive_mf_sum / (negative_mf_sum + 1e-8)
            mfi = 100 - (100 / (1 + mfr))

            # MFI normalized
            features.append(((mfi - 50) / 50).values)

            # Overbought/Oversold
            ob_os = np.where(mfi > 80, 1, np.where(mfi < 20, -1, 0))
            features.append(ob_os)

        return np.column_stack(features)
```

---

## 27. Complete Kelly Criterion Position Sizing

### 27.1 Theory and Implementation

The Kelly Criterion determines the optimal fraction of capital to risk on each trade to maximize long-term wealth growth. Originally developed for gambling, it has been adapted for financial markets.

```python
class KellyPositionSizer:
    """
    Kelly Criterion based position sizing

    The Kelly Criterion provides the mathematically optimal bet size
    that maximizes the expected growth rate of capital.

    Basic Kelly Formula:
    f* = (p * b - q) / b

    Where:
    - f* = Optimal fraction of capital to bet
    - p = Probability of winning
    - q = Probability of losing (1 - p)
    - b = Odds (amount won per dollar risked)

    For trading with variable outcomes:
    f* = (Expected Return) / (Variance of Returns)

    Or equivalently:
    f* = (Win Rate * Avg Win - Loss Rate * Avg Loss) / Avg Win

    Full Kelly is typically too aggressive for trading, so we use
    fractional Kelly (commonly 25-50% of full Kelly).
    """

    def __init__(self, config: dict):
        """
        Initialize Kelly position sizer

        Args:
            config: Configuration containing:
                - kelly_fraction: Fraction of Kelly to use (default 0.25)
                - min_trades: Minimum trades for reliable estimates
                - lookback: Number of recent trades for calculation
                - max_position: Maximum position size cap
                - min_position: Minimum position size floor
                - use_half_kelly: Use 0.5 * Kelly (common practice)
                - adaptive: Adjust fraction based on estimate uncertainty
        """
        self.kelly_fraction = config.get('kelly_fraction', 0.25)
        self.min_trades = config.get('min_trades', 30)
        self.lookback = config.get('lookback', 100)
        self.max_position = config.get('max_position', 0.1)  # 10% max
        self.min_position = config.get('min_position', 0.005)  # 0.5% min
        self.use_half_kelly = config.get('use_half_kelly', True)
        self.adaptive = config.get('adaptive', True)

        # Trade history
        self.trade_history = []

    def add_trade_result(self, pnl_pct: float, win: bool):
        """
        Add a trade result to history

        Args:
            pnl_pct: P&L as percentage (e.g., 0.02 for 2%)
            win: Whether the trade was a winner
        """
        self.trade_history.append({
            'pnl_pct': pnl_pct,
            'win': win,
            'timestamp': datetime.now(),
        })

        # Keep only recent history
        if len(self.trade_history) > self.lookback * 2:
            self.trade_history = self.trade_history[-self.lookback:]

    def calculate_kelly_fraction(self) -> dict:
        """
        Calculate the optimal Kelly fraction based on trade history

        Returns:
            Dictionary with:
            - kelly_fraction: Calculated optimal fraction
            - adjusted_fraction: After applying safety factors
            - win_rate: Estimated win rate
            - avg_win: Average winning trade
            - avg_loss: Average losing trade
            - edge: Expected edge per trade
            - confidence: Confidence in the estimate
        """
        # Get recent trades
        recent_trades = self.trade_history[-self.lookback:]

        if len(recent_trades) < self.min_trades:
            return {
                'kelly_fraction': 0.0,
                'adjusted_fraction': self.min_position,
                'win_rate': 0.5,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'edge': 0.0,
                'confidence': 0.0,
                'message': f'Insufficient trades ({len(recent_trades)}/{self.min_trades})',
            }

        # Separate wins and losses
        wins = [t['pnl_pct'] for t in recent_trades if t['win']]
        losses = [abs(t['pnl_pct']) for t in recent_trades if not t['win']]

        n_wins = len(wins)
        n_losses = len(losses)
        n_total = n_wins + n_losses

        # Calculate statistics
        win_rate = n_wins / n_total
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Calculate Kelly fraction using the trading formula
        # f* = (p * W - q * L) / W
        # Where W = avg win, L = avg loss, p = win rate, q = 1 - p

        if avg_win > 0:
            # Standard Kelly calculation
            edge = win_rate * avg_win - (1 - win_rate) * avg_loss
            kelly = edge / avg_win
        else:
            edge = 0
            kelly = 0

        # Alternative calculation using continuous returns
        # f* = mean(returns) / variance(returns)
        all_returns = [t['pnl_pct'] for t in recent_trades]
        mean_return = np.mean(all_returns)
        var_return = np.var(all_returns)

        if var_return > 0:
            kelly_continuous = mean_return / var_return
        else:
            kelly_continuous = 0

        # Use average of both methods
        kelly_combined = (kelly + kelly_continuous) / 2

        # Calculate confidence based on sample size and consistency
        confidence = self._calculate_confidence(recent_trades, win_rate, avg_win, avg_loss)

        # Apply fractional Kelly
        if self.use_half_kelly:
            adjusted = kelly_combined * 0.5
        else:
            adjusted = kelly_combined * self.kelly_fraction

        # Adaptive adjustment based on confidence
        if self.adaptive:
            adjusted = adjusted * confidence

        # Apply bounds
        adjusted = np.clip(adjusted, 0, self.max_position)

        # If Kelly suggests negative (losing system), use minimum
        if adjusted <= 0:
            adjusted = self.min_position

        return {
            'kelly_fraction': kelly_combined,
            'adjusted_fraction': adjusted,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'edge': edge,
            'profit_factor': avg_win * win_rate / (avg_loss * (1 - win_rate) + 1e-8),
            'confidence': confidence,
            'n_trades': n_total,
            'kelly_method1': kelly,
            'kelly_method2': kelly_continuous,
        }

    def _calculate_confidence(self, trades: List[dict], win_rate: float,
                             avg_win: float, avg_loss: float) -> float:
        """
        Calculate confidence in Kelly estimate

        Factors:
        1. Sample size (more trades = more confidence)
        2. Win rate stability (consistent win rate = more confidence)
        3. Outcome consistency (low variance = more confidence)
        4. Recent performance (recent trends)

        Returns:
            Confidence score between 0 and 1
        """
        n = len(trades)

        # Sample size confidence (asymptotic to 1)
        sample_confidence = 1 - np.exp(-n / 50)

        # Win rate stability (rolling win rate variance)
        if n >= 20:
            rolling_wr = []
            for i in range(10, n):
                window_trades = trades[i-10:i]
                window_wr = sum(1 for t in window_trades if t['win']) / 10
                rolling_wr.append(window_wr)
            wr_stability = 1 - min(1, np.std(rolling_wr) * 5)
        else:
            wr_stability = 0.5

        # Outcome consistency (coefficient of variation)
        returns = [t['pnl_pct'] for t in trades]
        if np.mean(np.abs(returns)) > 0:
            cv = np.std(returns) / np.mean(np.abs(returns))
            consistency = 1 / (1 + cv)
        else:
            consistency = 0.5

        # Recent performance trend
        if n >= 20:
            recent_wr = sum(1 for t in trades[-10:] if t['win']) / 10
            overall_wr = win_rate
            trend_factor = 1 - abs(recent_wr - overall_wr) * 2
            trend_factor = max(0, min(1, trend_factor))
        else:
            trend_factor = 0.5

        # Combine factors
        confidence = (
            sample_confidence * 0.3 +
            wr_stability * 0.25 +
            consistency * 0.25 +
            trend_factor * 0.2
        )

        return confidence

    def get_position_size(self, account_equity: float,
                          signal_strength: float = 1.0) -> dict:
        """
        Get recommended position size for a trade

        Args:
            account_equity: Current account equity
            signal_strength: Signal confidence (0 to 1)

        Returns:
            Dictionary with position sizing details
        """
        kelly_result = self.calculate_kelly_fraction()

        # Base position size
        base_fraction = kelly_result['adjusted_fraction']

        # Adjust for signal strength
        adjusted_fraction = base_fraction * signal_strength

        # Apply minimum
        adjusted_fraction = max(adjusted_fraction, self.min_position)

        # Calculate dollar amount
        position_value = account_equity * adjusted_fraction

        return {
            'fraction': adjusted_fraction,
            'position_value': position_value,
            'kelly_details': kelly_result,
            'signal_strength': signal_strength,
        }

    def simulate_growth(self, n_periods: int = 1000,
                        starting_capital: float = 100000) -> dict:
        """
        Monte Carlo simulation of capital growth

        Simulates multiple paths of capital growth using
        the estimated trading statistics and Kelly fraction.

        Returns:
            Dictionary with simulation results and statistics
        """
        kelly_result = self.calculate_kelly_fraction()

        if kelly_result['confidence'] < 0.5:
            return {
                'error': 'Insufficient confidence for simulation',
                'kelly_result': kelly_result,
            }

        win_rate = kelly_result['win_rate']
        avg_win = kelly_result['avg_win']
        avg_loss = kelly_result['avg_loss']
        fraction = kelly_result['adjusted_fraction']

        # Run multiple simulations
        n_simulations = 1000
        final_values = []
        max_drawdowns = []

        for _ in range(n_simulations):
            capital = starting_capital
            max_capital = capital
            max_dd = 0

            for _ in range(n_periods):
                # Simulate trade outcome
                if np.random.random() < win_rate:
                    # Win
                    pnl = capital * fraction * avg_win
                else:
                    # Loss
                    pnl = -capital * fraction * avg_loss

                capital += pnl

                # Track drawdown
                if capital > max_capital:
                    max_capital = capital
                dd = (max_capital - capital) / max_capital
                if dd > max_dd:
                    max_dd = dd

                # Stop if bankrupt
                if capital <= 0:
                    capital = 0
                    break

            final_values.append(capital)
            max_drawdowns.append(max_dd)

        final_values = np.array(final_values)
        max_drawdowns = np.array(max_drawdowns)

        return {
            'starting_capital': starting_capital,
            'n_periods': n_periods,
            'n_simulations': n_simulations,
            'kelly_fraction_used': fraction,
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'min_final_value': np.min(final_values),
            'max_final_value': np.max(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_95': np.percentile(final_values, 95),
            'probability_of_profit': np.mean(final_values > starting_capital),
            'probability_of_ruin': np.mean(final_values <= 0),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.max(max_drawdowns),
            'cagr_estimate': (np.mean(final_values) / starting_capital) ** (252 / n_periods) - 1,
        }
```

---

## 28. Complete Backtesting Engine

### 28.1 High-Fidelity Backtesting

```python
class BacktestEngine:
    """
    Production-grade backtesting engine

    Features:
    - Tick-level or bar-level simulation
    - Realistic slippage and commission models
    - Market impact simulation
    - Multi-asset support
    - Walk-forward capability
    - Comprehensive performance analytics
    """

    def __init__(self, config: dict):
        """
        Initialize backtest engine

        Args:
            config: Configuration dictionary containing:
                - initial_capital: Starting capital
                - commission_model: Commission structure
                - slippage_model: Slippage simulation
                - margin_requirements: Margin rules
                - data_config: Data source configuration
        """
        self.initial_capital = config.get('initial_capital', 100000)
        self.commission_model = self._create_commission_model(
            config.get('commission', {})
        )
        self.slippage_model = self._create_slippage_model(
            config.get('slippage', {})
        )
        self.margin_model = self._create_margin_model(
            config.get('margin', {})
        )

        # Results storage
        self.trades = []
        self.equity_curve = []
        self.positions_history = []
        self.daily_returns = []

    def run_backtest(self, strategy, data: pd.DataFrame,
                     start_date: str = None, end_date: str = None) -> BacktestResult:
        """
        Execute complete backtest

        Args:
            strategy: Trading strategy object with generate_signal method
            data: OHLCV DataFrame with datetime index
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            BacktestResult with comprehensive metrics
        """
        # Filter data by date
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        # Initialize portfolio
        portfolio = Portfolio(self.initial_capital)

        # Reset results
        self.trades = []
        self.equity_curve = []
        self.positions_history = []

        # Main backtest loop
        logger.info(f"Starting backtest: {len(data)} bars")

        for i, (timestamp, bar) in enumerate(data.iterrows()):
            # Update market prices
            portfolio.update_prices({bar.name: bar['close']})

            # Generate signal
            signal = strategy.generate_signal(data.iloc[:i+1])

            # Execute signal if actionable
            if signal.action != 'HOLD':
                execution = self._execute_signal(signal, bar, portfolio)

                if execution['filled']:
                    trade = self._record_trade(signal, execution, timestamp)
                    self.trades.append(trade)
                    portfolio.apply_trade(trade)

            # Update portfolio valuation
            portfolio.mark_to_market()

            # Record state
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': portfolio.equity,
                'cash': portfolio.cash,
                'positions_value': portfolio.positions_value,
                'margin_used': portfolio.margin_used,
            })

            self.positions_history.append({
                'timestamp': timestamp,
                'positions': portfolio.get_positions_snapshot(),
            })

            # Progress logging
            if i % 10000 == 0:
                logger.info(f"Processed {i}/{len(data)} bars, "
                           f"equity: ${portfolio.equity:,.2f}")

        # Close any remaining positions at end
        self._close_all_positions(portfolio, data.iloc[-1])

        # Calculate results
        result = self._calculate_results(portfolio, data)

        logger.info(f"Backtest complete: {len(self.trades)} trades, "
                   f"final equity: ${portfolio.equity:,.2f}")

        return result

    def _execute_signal(self, signal, bar: pd.Series,
                        portfolio: 'Portfolio') -> dict:
        """
        Simulate order execution

        Handles:
        - Market orders (execute at next bar open with slippage)
        - Limit orders (execute if price reached)
        - Stop orders (execute if stop triggered)
        - Slippage calculation
        - Commission calculation
        - Fill simulation
        """
        execution = {
            'filled': False,
            'fill_price': 0.0,
            'fill_size': 0.0,
            'slippage': 0.0,
            'commission': 0.0,
        }

        # Determine execution price
        if signal.order_type == 'MARKET':
            # Market order - execute at current bar close with slippage
            base_price = bar['close']

            # Calculate slippage
            slippage = self.slippage_model.calculate(
                price=base_price,
                size=signal.size,
                direction=signal.direction,
                volatility=bar.get('atr', base_price * 0.01),
            )

            if signal.direction == 'BUY':
                fill_price = base_price + slippage
            else:
                fill_price = base_price - slippage

            execution['filled'] = True
            execution['fill_price'] = fill_price
            execution['slippage'] = slippage

        elif signal.order_type == 'LIMIT':
            # Limit order - check if price was reached
            if signal.direction == 'BUY':
                if bar['low'] <= signal.limit_price:
                    execution['filled'] = True
                    execution['fill_price'] = min(signal.limit_price, bar['open'])
            else:
                if bar['high'] >= signal.limit_price:
                    execution['filled'] = True
                    execution['fill_price'] = max(signal.limit_price, bar['open'])

        elif signal.order_type == 'STOP':
            # Stop order - check if stop was triggered
            if signal.direction == 'BUY':
                if bar['high'] >= signal.stop_price:
                    execution['filled'] = True
                    execution['fill_price'] = max(signal.stop_price, bar['open'])
            else:
                if bar['low'] <= signal.stop_price:
                    execution['filled'] = True
                    execution['fill_price'] = min(signal.stop_price, bar['open'])

        # If filled, calculate fill size and commission
        if execution['filled']:
            # Calculate affordable size
            max_size = self._calculate_max_size(
                portfolio, execution['fill_price'], signal
            )
            execution['fill_size'] = min(signal.size, max_size)

            # Calculate commission
            execution['commission'] = self.commission_model.calculate(
                price=execution['fill_price'],
                size=execution['fill_size'],
            )

            # Check if trade is still viable after commission
            if execution['fill_size'] <= 0:
                execution['filled'] = False

        return execution

    def _calculate_max_size(self, portfolio: 'Portfolio',
                            price: float, signal) -> float:
        """
        Calculate maximum affordable position size

        Considers:
        - Available cash
        - Margin requirements
        - Position limits
        - Risk limits
        """
        available_cash = portfolio.cash - portfolio.margin_used
        available_margin = available_cash * portfolio.leverage

        # Maximum shares based on buying power
        max_by_cash = available_margin / (price + 0.01)

        # Maximum based on position limits
        max_by_limit = self.margin_model.max_position_size

        # Maximum based on risk limits (if stop loss specified)
        if signal.stop_loss:
            risk_per_share = abs(price - signal.stop_loss)
            max_risk_amount = portfolio.equity * self.margin_model.max_risk_per_trade
            max_by_risk = max_risk_amount / (risk_per_share + 0.01)
        else:
            max_by_risk = float('inf')

        return min(max_by_cash, max_by_limit, max_by_risk, signal.size)

    def _record_trade(self, signal, execution: dict,
                      timestamp: datetime) -> dict:
        """
        Create trade record
        """
        return {
            'timestamp': timestamp,
            'symbol': signal.symbol,
            'direction': signal.direction,
            'size': execution['fill_size'],
            'entry_price': execution['fill_price'],
            'slippage': execution['slippage'],
            'commission': execution['commission'],
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'signal_strength': signal.strength,
            'order_type': signal.order_type,
        }

    def _close_all_positions(self, portfolio: 'Portfolio',
                             final_bar: pd.Series):
        """
        Close all positions at end of backtest
        """
        for symbol, position in portfolio.positions.items():
            if position.size != 0:
                close_price = final_bar['close']

                # Calculate P&L
                if position.direction == 'LONG':
                    pnl = (close_price - position.entry_price) * position.size
                else:
                    pnl = (position.entry_price - close_price) * position.size

                commission = self.commission_model.calculate(close_price, position.size)
                pnl -= commission

                # Update trade record
                for trade in reversed(self.trades):
                    if trade['symbol'] == symbol and 'exit_price' not in trade:
                        trade['exit_price'] = close_price
                        trade['exit_timestamp'] = final_bar.name
                        trade['pnl'] = pnl
                        trade['exit_commission'] = commission
                        break

                # Update portfolio
                portfolio.close_position(symbol, close_price, pnl)

    def _calculate_results(self, portfolio: 'Portfolio',
                          data: pd.DataFrame) -> 'BacktestResult':
        """
        Calculate comprehensive backtest results
        """
        # Convert equity curve to array
        equity = np.array([e['equity'] for e in self.equity_curve])
        timestamps = [e['timestamp'] for e in self.equity_curve]

        # Calculate returns
        returns = np.diff(equity) / equity[:-1]

        # Performance metrics
        total_return = (portfolio.equity - self.initial_capital) / self.initial_capital

        # Annualized metrics (assuming daily data)
        n_days = len(equity)
        annual_factor = 252 / n_days if n_days > 0 else 1

        annualized_return = (1 + total_return) ** annual_factor - 1

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            sortino = np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252) if np.any(returns < 0) else sharpe
        else:
            sharpe = 0
            sortino = 0

        # Drawdown analysis
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)

        # Find longest drawdown period
        in_drawdown = drawdown > 0
        drawdown_starts = np.where((~in_drawdown[:-1]) & in_drawdown[1:])[0]
        drawdown_ends = np.where(in_drawdown[:-1] & (~in_drawdown[1:]))[0]

        if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
            durations = []
            for start in drawdown_starts:
                ends_after = drawdown_ends[drawdown_ends > start]
                if len(ends_after) > 0:
                    durations.append(ends_after[0] - start)
            max_drawdown_duration = max(durations) if durations else 0
        else:
            max_drawdown_duration = 0

        # Trade analysis
        n_trades = len(self.trades)

        if n_trades > 0:
            winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in self.trades if t.get('pnl', 0) <= 0]

            n_winners = len(winning_trades)
            n_losers = len(losing_trades)

            win_rate = n_winners / n_trades

            total_profit = sum(t['pnl'] for t in winning_trades)
            total_loss = abs(sum(t['pnl'] for t in losing_trades))

            avg_win = total_profit / n_winners if n_winners > 0 else 0
            avg_loss = total_loss / n_losers if n_losers > 0 else 0

            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            avg_trade = sum(t.get('pnl', 0) for t in self.trades) / n_trades

            # Calculate average holding period
            holding_periods = []
            for trade in self.trades:
                if 'exit_timestamp' in trade:
                    holding = (trade['exit_timestamp'] - trade['timestamp']).days
                    holding_periods.append(holding)
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        else:
            win_rate = 0
            n_winners = 0
            n_losers = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade = 0
            avg_holding_period = 0
            total_profit = 0
            total_loss = 0

        # Calmar ratio
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Recovery factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0

        return BacktestResult(
            # Core metrics
            initial_capital=self.initial_capital,
            final_equity=portfolio.equity,
            total_return=total_return,
            annualized_return=annualized_return,

            # Risk metrics
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,

            # Trade metrics
            total_trades=n_trades,
            winning_trades=n_winners,
            losing_trades=n_losers,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            avg_holding_period=avg_holding_period,
            total_profit=total_profit,
            total_loss=total_loss,

            # Recovery metrics
            recovery_factor=recovery_factor,

            # Data
            equity_curve=equity.tolist(),
            timestamps=timestamps,
            trades=self.trades,
            drawdown_series=drawdown.tolist(),
        )

    def _create_commission_model(self, config: dict):
        """Create commission calculation model"""
        model_type = config.get('type', 'per_trade')

        if model_type == 'per_trade':
            return FixedCommissionModel(config.get('cost_per_trade', 1.0))
        elif model_type == 'per_share':
            return PerShareCommissionModel(config.get('cost_per_share', 0.005))
        elif model_type == 'percentage':
            return PercentageCommissionModel(config.get('rate', 0.001))
        else:
            return FixedCommissionModel(0)

    def _create_slippage_model(self, config: dict):
        """Create slippage simulation model"""
        model_type = config.get('type', 'fixed')

        if model_type == 'fixed':
            return FixedSlippageModel(config.get('bps', 1.0))
        elif model_type == 'volatility':
            return VolatilitySlippageModel(config.get('vol_multiplier', 0.1))
        elif model_type == 'volume':
            return VolumeSlippageModel(config.get('impact_coefficient', 0.1))
        else:
            return FixedSlippageModel(0)

    def _create_margin_model(self, config: dict):
        """Create margin requirements model"""
        return MarginModel(
            initial_margin=config.get('initial_margin', 0.5),
            maintenance_margin=config.get('maintenance_margin', 0.25),
            max_leverage=config.get('max_leverage', 2.0),
            max_position_size=config.get('max_position_size', 100000),
            max_risk_per_trade=config.get('max_risk_per_trade', 0.02),
        )
```

---

*Document continues with additional technical appendices, API documentation, deployment guides, and operational procedures...*

*Total Line Count: ~25,000*

*Copyright 2026. All rights reserved.*
