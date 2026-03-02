# =============================================================================
# UNIFIED AGENTIC ENVIRONMENT
# =============================================================================
# Maintains a CONSTANT observation space across all training modes, solving the
# critical domain shift problem between training and production.
#
# KEY INSIGHT: The original system trained on TradingEnv (303 dims) but deployed
# with OrchestratedTradingEnv which has different dynamics. This causes 30-60%
# action rejection at deployment due to domain shift.
#
# SOLUTION: This unified environment always has the same observation space
# (303 base + 20 agent signals = 323 dims) but progressively introduces agent
# constraints based on the training mode.
#
# Training Modes:
# 1. BASE: Agent signals are zero-filled (pure market learning)
# 2. ENRICHED: Agent signals included as observation (no constraints)
# 3. SOFT: Agent signals + soft penalties for rejected actions
# 4. PRODUCTION: Full agent integration with hard constraints
# =============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import logging
from collections import deque
import warnings

# Import base environment
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.environment.environment import TradingEnv
import config


class TrainingMode(Enum):
    """Training mode determines how agent signals affect the environment."""
    BASE = auto()       # Pure market learning - agent signals zeroed
    ENRICHED = auto()   # Signals as observation only - no constraints
    SOFT = auto()       # Soft penalties for would-be rejected actions
    PRODUCTION = auto() # Full agent integration with hard constraints


@dataclass
class AgentSignals:
    """Container for all agent-generated signals."""
    # News Agent signals
    news_sentiment: float = 0.0          # [-1, 1] bearish to bullish
    news_impact_score: float = 0.0       # [0, 1] impact magnitude
    news_event_proximity: float = 0.0    # [0, 1] time to next high-impact event
    news_blocking_active: bool = False   # True if news is blocking trades

    # Risk Sentinel signals
    risk_score: float = 0.0              # [0, 1] current risk level
    position_size_multiplier: float = 1.0 # [0, 1] allowed position size
    drawdown_proximity: float = 0.0      # [0, 1] how close to max drawdown
    volatility_regime: float = 0.5       # [0, 1] low to high volatility
    kelly_fraction: float = 0.1          # Optimal Kelly fraction

    # Market Regime Agent signals
    regime_trend: float = 0.0            # [-1, 1] downtrend to uptrend
    regime_volatility: float = 0.5       # [0, 1] regime volatility
    regime_momentum: float = 0.0         # [-1, 1] momentum direction
    regime_confidence: float = 0.5       # [0, 1] regime detection confidence

    # Orchestrator meta-signals
    action_allowed: np.ndarray = field(default_factory=lambda: np.ones(5))  # [5] action mask
    suggested_action: int = 0            # Orchestrator's suggested action
    urgency_score: float = 0.0           # [0, 1] trade urgency
    consensus_score: float = 0.5         # [0, 1] agent agreement level

    # Historical context (for temporal patterns)
    recent_rejection_rate: float = 0.0   # [0, 1] recent action rejections
    recent_news_blocks: float = 0.0      # [0, 1] recent news blockings

    def to_array(self) -> np.ndarray:
        """Convert all signals to a flat numpy array (20 dimensions)."""
        return np.array([
            # News signals (4)
            self.news_sentiment,
            self.news_impact_score,
            self.news_event_proximity,
            float(self.news_blocking_active),
            # Risk signals (5)
            self.risk_score,
            self.position_size_multiplier,
            self.drawdown_proximity,
            self.volatility_regime,
            self.kelly_fraction,
            # Regime signals (4)
            self.regime_trend,
            self.regime_volatility,
            self.regime_momentum,
            self.regime_confidence,
            # Orchestrator signals (5)
            self.suggested_action / 4.0,  # Normalize to [0, 1]
            self.urgency_score,
            self.consensus_score,
            # Historical (2)
            self.recent_rejection_rate,
            self.recent_news_blocks,
            # Action mask (compressed to 2 dims: long_allowed, short_allowed)
            float(self.action_allowed[1] and self.action_allowed[2]),  # Long actions
            float(self.action_allowed[3] and self.action_allowed[4]),  # Short actions
        ], dtype=np.float32)


class MockNewsAgent:
    """Simulated News Agent for training with historical data."""

    def __init__(self, economic_calendar: Optional[pd.DataFrame] = None):
        self.calendar = economic_calendar
        self._rng = np.random.default_rng(42)

    def get_signals(self, timestamp: pd.Timestamp, current_price: float) -> Dict[str, float]:
        """Generate news signals (simulated or from calendar)."""
        if self.calendar is not None and timestamp in self.calendar.index:
            event = self.calendar.loc[timestamp]
            return {
                'sentiment': event.get('sentiment', 0.0),
                'impact_score': event.get('impact', 0.5),
                'event_proximity': event.get('proximity', 1.0),
                'blocking': event.get('impact', 0) > 0.8
            }

        # Simulated random news (rare high-impact events)
        if self._rng.random() < 0.02:  # 2% chance of news event
            impact = self._rng.uniform(0.3, 1.0)
            sentiment = self._rng.uniform(-1, 1)
            return {
                'sentiment': sentiment,
                'impact_score': impact,
                'event_proximity': self._rng.uniform(0, 0.5),
                'blocking': impact > 0.8
            }

        return {
            'sentiment': 0.0,
            'impact_score': 0.0,
            'event_proximity': 1.0,
            'blocking': False
        }


class MockRiskSentinel:
    """Simulated Risk Sentinel for training."""

    def __init__(self, max_drawdown_pct: float = 0.10, max_volatility: float = 0.05):
        self.max_drawdown_pct = max_drawdown_pct
        self.max_volatility = max_volatility
        self._equity_peak = 0.0
        self._returns_buffer = deque(maxlen=50)

    def get_signals(self, equity: float, current_return: float) -> Dict[str, float]:
        """Calculate risk signals based on equity curve."""
        self._equity_peak = max(self._equity_peak, equity)
        self._returns_buffer.append(current_return)

        # Drawdown calculation
        drawdown = (self._equity_peak - equity) / self._equity_peak if self._equity_peak > 0 else 0
        drawdown_proximity = min(drawdown / self.max_drawdown_pct, 1.0)

        # Volatility calculation
        if len(self._returns_buffer) >= 10:
            volatility = np.std(list(self._returns_buffer))
            volatility_regime = min(volatility / self.max_volatility, 1.0)
        else:
            volatility_regime = 0.5

        # Risk score composite
        risk_score = 0.5 * drawdown_proximity + 0.5 * volatility_regime

        # Position size based on Kelly criterion approximation
        if len(self._returns_buffer) >= 20:
            returns = np.array(list(self._returns_buffer))
            win_rate = np.mean(returns > 0)
            avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.01
            avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0.01
            kelly = (win_rate - (1 - win_rate) / (avg_win / avg_loss)) if avg_loss > 0 else 0.1
            kelly = np.clip(kelly, 0.01, 0.25)  # Cap at 25%
        else:
            kelly = 0.1

        # Position multiplier (reduce in high risk)
        position_multiplier = max(0.1, 1.0 - risk_score)

        return {
            'risk_score': risk_score,
            'position_multiplier': position_multiplier,
            'drawdown_proximity': drawdown_proximity,
            'volatility_regime': volatility_regime,
            'kelly_fraction': kelly
        }

    def reset(self):
        """Reset for new episode."""
        self._equity_peak = 0.0
        self._returns_buffer.clear()


class MockMarketRegimeAgent:
    """Simulated Market Regime Detection Agent."""

    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self._price_buffer = deque(maxlen=lookback)

    def get_signals(self, current_price: float, volume: float = 0) -> Dict[str, float]:
        """Detect market regime from price history."""
        self._price_buffer.append(current_price)

        if len(self._price_buffer) < 20:
            return {
                'trend': 0.0,
                'volatility': 0.5,
                'momentum': 0.0,
                'confidence': 0.3
            }

        prices = np.array(list(self._price_buffer))

        # Trend detection (linear regression slope)
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        trend = np.tanh(slope * 100)  # Normalize to [-1, 1]

        # Volatility (normalized std)
        volatility = np.std(np.diff(prices) / prices[:-1])
        volatility_norm = np.clip(volatility / 0.02, 0, 1)  # 2% as reference

        # Momentum (rate of change)
        if len(prices) >= 10:
            momentum = (prices[-1] - prices[-10]) / prices[-10]
            momentum_norm = np.tanh(momentum * 20)
        else:
            momentum_norm = 0.0

        # Confidence (based on trend consistency)
        returns = np.diff(prices) / prices[:-1]
        if np.std(returns) > 0:
            trend_consistency = abs(np.mean(returns)) / np.std(returns)
            confidence = np.clip(trend_consistency, 0, 1)
        else:
            confidence = 0.5

        return {
            'trend': trend,
            'volatility': volatility_norm,
            'momentum': momentum_norm,
            'confidence': confidence
        }

    def reset(self):
        """Reset for new episode."""
        self._price_buffer.clear()


class MockOrchestrator:
    """Simulated Orchestrator for combining agent signals."""

    def __init__(self):
        self._rejection_history = deque(maxlen=20)
        self._news_block_history = deque(maxlen=20)

    def get_meta_signals(
        self,
        news_signals: Dict,
        risk_signals: Dict,
        regime_signals: Dict,
        current_action: int,
        position_type: int
    ) -> Dict[str, Any]:
        """Combine all agent signals into orchestrator meta-signals."""

        # Determine action mask based on agent signals
        action_allowed = np.ones(5, dtype=np.float32)

        # News blocking
        if news_signals.get('blocking', False):
            action_allowed[1:] = 0.0  # Block all except HOLD
            self._news_block_history.append(1)
        else:
            self._news_block_history.append(0)

        # Risk-based restrictions
        risk_score = risk_signals.get('risk_score', 0)
        if risk_score > 0.8:  # High risk - only allow closing
            if position_type == 1:  # LONG
                action_allowed[1] = 0.0  # Block OPEN_LONG
                action_allowed[3] = 0.0  # Block OPEN_SHORT
                action_allowed[4] = 0.0  # Block CLOSE_SHORT
            elif position_type == -1:  # SHORT
                action_allowed[1] = 0.0
                action_allowed[2] = 0.0
                action_allowed[3] = 0.0
            else:  # FLAT
                action_allowed[1] = 0.0
                action_allowed[3] = 0.0

        # Suggested action based on regime
        trend = regime_signals.get('trend', 0)
        confidence = regime_signals.get('confidence', 0.5)

        if position_type == 0:  # FLAT
            if trend > 0.3 and confidence > 0.5:
                suggested = 1  # OPEN_LONG
            elif trend < -0.3 and confidence > 0.5:
                suggested = 3  # OPEN_SHORT
            else:
                suggested = 0  # HOLD
        elif position_type == 1:  # LONG
            if trend < -0.2 or risk_score > 0.7:
                suggested = 2  # CLOSE_LONG
            else:
                suggested = 0
        else:  # SHORT
            if trend > 0.2 or risk_score > 0.7:
                suggested = 4  # CLOSE_SHORT
            else:
                suggested = 0

        # Urgency based on momentum and volatility
        momentum = abs(regime_signals.get('momentum', 0))
        volatility = regime_signals.get('volatility', 0.5)
        urgency = np.clip(momentum * (1 + volatility), 0, 1)

        # Consensus (agreement between agents)
        signals_agree = (
            (trend > 0 and risk_score < 0.5) or
            (trend < 0 and risk_score < 0.5) or
            (abs(trend) < 0.2)  # Neutral agreement
        )
        consensus = 0.8 if signals_agree else 0.3

        # Track rejections
        would_reject = (
            action_allowed[current_action] < 0.5 if current_action > 0 else False
        )
        self._rejection_history.append(1 if would_reject else 0)

        return {
            'action_allowed': action_allowed,
            'suggested_action': suggested,
            'urgency': urgency,
            'consensus': consensus,
            'recent_rejections': np.mean(list(self._rejection_history)) if self._rejection_history else 0,
            'recent_news_blocks': np.mean(list(self._news_block_history)) if self._news_block_history else 0
        }

    def reset(self):
        """Reset for new episode."""
        self._rejection_history.clear()
        self._news_block_history.clear()


class UnifiedAgenticEnv(gym.Env):
    """
    Unified environment that maintains constant observation space across all
    training modes while progressively introducing agent constraints.

    Observation Space: 623 dimensions (with MTF enabled)
    - 603 from base TradingEnv (20 lookback × 30 features + 3 state)
    - 20 from agent signals

    This solves the domain shift problem by ensuring the model always sees
    the same observation structure, regardless of training mode.
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    # Agent signal dimensions
    AGENT_SIGNAL_DIM = 20

    def __init__(
        self,
        df: pd.DataFrame,
        mode: TrainingMode = TrainingMode.BASE,
        economic_calendar: Optional[pd.DataFrame] = None,
        render_mode: str = "none",
        soft_penalty_scale: float = 1.0,
        enable_logging: bool = False,
        **kwargs
    ):
        """
        Initialize the Unified Agentic Environment.

        Args:
            df: OHLCV DataFrame
            mode: Training mode (BASE, ENRICHED, SOFT, PRODUCTION)
            economic_calendar: Optional economic calendar for news simulation
            render_mode: Rendering mode
            soft_penalty_scale: Scale for soft penalties in SOFT mode
            enable_logging: Enable detailed logging
            **kwargs: Additional arguments passed to base TradingEnv
        """
        super().__init__()

        self.mode = mode
        self.soft_penalty_scale = soft_penalty_scale
        self.enable_logging = enable_logging
        self.render_mode = render_mode
        self._logger = logging.getLogger(__name__)

        # Create base environment
        # Default scaler to fit on all data (safe because training pipeline
        # already splits train/val/test before creating environments)
        if 'scaler_fit_end_idx' not in kwargs and 'pre_fitted_scaler' not in kwargs:
            kwargs['scaler_fit_end_idx'] = len(df)

        self._base_env = TradingEnv(
            df=df,
            render_mode=render_mode,
            enable_logging=enable_logging,
            **kwargs
        )

        # Get base observation space dimensions
        base_obs_shape = self._base_env.observation_space.shape[0]
        self._base_obs_dim = base_obs_shape

        # Total observation space: base + agent signals
        total_dim = base_obs_shape + self.AGENT_SIGNAL_DIM

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(total_dim,),
            dtype=np.float32
        )

        # Action space same as base
        self.action_space = self._base_env.action_space

        # Initialize mock agents for training
        self._news_agent = MockNewsAgent(economic_calendar)
        self._risk_sentinel = MockRiskSentinel(
            max_drawdown_pct=kwargs.get('max_drawdown_pct', 0.10),
            max_volatility=kwargs.get('max_volatility', 0.05)
        )
        self._regime_agent = MockMarketRegimeAgent(lookback=50)
        self._orchestrator = MockOrchestrator()

        # Current agent signals
        self._current_signals = AgentSignals()

        # Statistics tracking
        self._episode_rejections = 0
        self._episode_news_blocks = 0
        self._episode_soft_penalties = 0.0
        self._total_steps = 0

        self._logger.info(
            f"UnifiedAgenticEnv initialized: mode={mode.name}, "
            f"obs_dim={total_dim} (base={base_obs_shape} + agent={self.AGENT_SIGNAL_DIM})"
        )

    def _get_agent_signals(self, action: int) -> AgentSignals:
        """Compute all agent signals for current state."""
        # Get current state from base env
        current_price = self._base_env.df.iloc[self._base_env.current_step]['Close']
        volume = self._base_env.df.iloc[self._base_env.current_step].get('Volume', 0)
        equity = self._base_env.balance + getattr(self._base_env, 'unrealized_pnl', 0)
        position_type = self._base_env.position_type

        # Get timestamp if available
        if hasattr(self._base_env.df, 'index') and isinstance(self._base_env.df.index, pd.DatetimeIndex):
            timestamp = self._base_env.df.index[self._base_env.current_step]
        else:
            timestamp = pd.Timestamp.now()

        # Get last return
        if hasattr(self._base_env, '_last_portfolio_value'):
            last_value = self._base_env._last_portfolio_value
            current_return = (equity - last_value) / last_value if last_value > 0 else 0
        else:
            current_return = 0

        # News Agent signals
        news = self._news_agent.get_signals(timestamp, current_price)

        # Risk Sentinel signals
        risk = self._risk_sentinel.get_signals(equity, current_return)

        # Market Regime signals
        regime = self._regime_agent.get_signals(current_price, volume)

        # Orchestrator meta-signals
        meta = self._orchestrator.get_meta_signals(
            news, risk, regime, action, position_type
        )

        # Build AgentSignals object
        signals = AgentSignals(
            # News
            news_sentiment=news['sentiment'],
            news_impact_score=news['impact_score'],
            news_event_proximity=news['event_proximity'],
            news_blocking_active=news['blocking'],
            # Risk
            risk_score=risk['risk_score'],
            position_size_multiplier=risk['position_multiplier'],
            drawdown_proximity=risk['drawdown_proximity'],
            volatility_regime=risk['volatility_regime'],
            kelly_fraction=risk['kelly_fraction'],
            # Regime
            regime_trend=regime['trend'],
            regime_volatility=regime['volatility'],
            regime_momentum=regime['momentum'],
            regime_confidence=regime['confidence'],
            # Orchestrator
            action_allowed=meta['action_allowed'],
            suggested_action=meta['suggested_action'],
            urgency_score=meta['urgency'],
            consensus_score=meta['consensus'],
            recent_rejection_rate=meta['recent_rejections'],
            recent_news_blocks=meta['recent_news_blocks']
        )

        return signals

    def _build_observation(self, base_obs: np.ndarray, signals: AgentSignals) -> np.ndarray:
        """Combine base observation with agent signals."""
        if self.mode == TrainingMode.BASE:
            # Zero out agent signals in BASE mode
            agent_obs = np.zeros(self.AGENT_SIGNAL_DIM, dtype=np.float32)
        else:
            # Include real signals in other modes
            agent_obs = signals.to_array()

        return np.concatenate([base_obs, agent_obs]).astype(np.float32)

    def _apply_mode_constraints(
        self,
        action: int,
        signals: AgentSignals
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Apply training-mode-specific constraints to the action.

        Returns:
            Tuple of (modified_action, penalty, info_dict)
        """
        info = {
            'original_action': action,
            'action_modified': False,
            'rejection_reason': None,
            'soft_penalty': 0.0
        }

        penalty = 0.0
        modified_action = action

        if self.mode == TrainingMode.BASE:
            # No constraints in BASE mode
            return action, 0.0, info

        elif self.mode == TrainingMode.ENRICHED:
            # Signals as observation only, no constraints
            return action, 0.0, info

        elif self.mode == TrainingMode.SOFT:
            # Soft penalties for would-be rejected actions
            if signals.news_blocking_active and action != 0:
                penalty = config.NEWS_BLOCK_PENALTY * self.soft_penalty_scale
                info['soft_penalty'] = penalty
                info['rejection_reason'] = 'news_block'
                self._episode_soft_penalties += penalty

            elif signals.action_allowed[action] < 0.5:
                penalty = config.RISK_REJECTION_PENALTY * self.soft_penalty_scale
                info['soft_penalty'] = penalty
                info['rejection_reason'] = 'risk_rejection'
                self._episode_soft_penalties += penalty

            elif signals.position_size_multiplier < config.MODIFICATION_THRESHOLD:
                penalty = config.RISK_MODIFICATION_PENALTY * self.soft_penalty_scale
                info['soft_penalty'] = penalty
                info['rejection_reason'] = 'position_modified'
                self._episode_soft_penalties += penalty

            # Action still executes (soft penalty)
            return action, penalty, info

        elif self.mode == TrainingMode.PRODUCTION:
            # Full hard constraints
            if signals.news_blocking_active and action != 0:
                modified_action = 0  # Force HOLD
                info['action_modified'] = True
                info['rejection_reason'] = 'news_block'
                self._episode_news_blocks += 1
                penalty = config.NEWS_BLOCK_PENALTY

            elif signals.action_allowed[action] < 0.5:
                modified_action = 0  # Force HOLD
                info['action_modified'] = True
                info['rejection_reason'] = 'risk_rejection'
                self._episode_rejections += 1
                penalty = config.RISK_REJECTION_PENALTY

            return modified_action, penalty, info

        return action, 0.0, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        The step behavior depends on the training mode:
        - BASE: Direct execution, zero agent signals
        - ENRICHED: Direct execution with real agent signals
        - SOFT: Execution with soft penalties for bad actions
        - PRODUCTION: Hard constraints (action may be modified)
        """
        self._total_steps += 1

        # Get agent signals BEFORE action execution
        signals = self._get_agent_signals(action)
        self._current_signals = signals

        # Apply mode-specific constraints
        modified_action, mode_penalty, mode_info = self._apply_mode_constraints(
            action, signals
        )

        # Execute action in base environment
        base_obs, base_reward, done, truncated, base_info = self._base_env.step(modified_action)

        # Apply mode penalty to reward
        total_reward = base_reward - mode_penalty

        # Build unified observation
        obs = self._build_observation(base_obs, signals)

        # Merge info dicts
        info = {**base_info, **mode_info}
        info['agent_signals'] = {
            'news_sentiment': signals.news_sentiment,
            'risk_score': signals.risk_score,
            'regime_trend': signals.regime_trend,
            'suggested_action': signals.suggested_action,
        }
        info['mode'] = self.mode.name
        info['mode_penalty'] = mode_penalty

        return obs, total_reward, done, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        # Reset base environment
        base_obs, base_info = self._base_env.reset(seed=seed, options=options)

        # Reset mock agents
        self._risk_sentinel.reset()
        self._regime_agent.reset()
        self._orchestrator.reset()

        # Reset statistics
        episode_stats = {
            'rejections': self._episode_rejections,
            'news_blocks': self._episode_news_blocks,
            'soft_penalties': self._episode_soft_penalties
        }
        self._episode_rejections = 0
        self._episode_news_blocks = 0
        self._episode_soft_penalties = 0.0

        # Get initial signals (with HOLD action)
        signals = self._get_agent_signals(0)
        self._current_signals = signals

        # Build unified observation
        obs = self._build_observation(base_obs, signals)

        # Merge info
        info = {**base_info, 'previous_episode_stats': episode_stats, 'mode': self.mode.name}

        return obs, info

    def set_mode(self, mode: TrainingMode):
        """
        Change training mode dynamically.

        This allows curriculum learning to progressively increase difficulty
        without creating a new environment.
        """
        old_mode = self.mode
        self.mode = mode
        self._logger.info(f"Training mode changed: {old_mode.name} -> {mode.name}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'total_steps': self._total_steps,
            'current_mode': self.mode.name,
            'episode_rejections': self._episode_rejections,
            'episode_news_blocks': self._episode_news_blocks,
            'episode_soft_penalties': self._episode_soft_penalties,
            'observation_dim': self.observation_space.shape[0],
            'base_dim': self._base_obs_dim,
            'agent_signal_dim': self.AGENT_SIGNAL_DIM
        }

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self._base_env.unwrapped

    def render(self):
        """Render the environment."""
        return self._base_env.render()

    def close(self):
        """Close the environment."""
        self._base_env.close()
