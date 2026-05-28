# =============================================================================
# ADVANCED REWARD SHAPER - Multi-Objective Optimization
# =============================================================================
# Implements sophisticated reward shaping with multiple objectives:
# 1. Risk-Adjusted Returns (Sharpe, Sortino, Calmar)
# 2. Drawdown Control (continuous penalty, not just end-of-episode)
# 3. Trade Quality (win rate, profit factor, risk-reward)
# 4. Behavior Shaping (exploration, regime adaptation, timing)
#
# The key innovation is DYNAMIC WEIGHTING that adjusts based on:
# - Training phase (early: exploration, late: exploitation)
# - Market regime (trending: follow, ranging: mean-revert)
# - Agent performance (struggling: simpler rewards, excelling: harder targets)
#
# This creates an adaptive curriculum within the reward function itself.
# =============================================================================

import numpy as np
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging


class RewardObjective(Enum):
    """Individual reward objectives."""
    PROFIT = auto()           # Raw profit/loss
    SHARPE = auto()           # Sharpe ratio optimization
    SORTINO = auto()          # Downside risk optimization
    CALMAR = auto()           # Drawdown-adjusted returns
    WIN_RATE = auto()         # Increase winning trades
    PROFIT_FACTOR = auto()    # Gross profit / gross loss
    RISK_REWARD = auto()      # Average win / average loss
    EXPLORATION = auto()      # Action diversity bonus
    TIMING = auto()           # Entry/exit timing quality
    REGIME_ADAPTATION = auto() # Adapting to market regime


@dataclass
class RewardWeights:
    """Weights for each objective component."""
    profit: float = 1.0
    sharpe: float = 0.5
    sortino: float = 0.3
    calmar: float = 0.2
    win_rate: float = 0.2
    profit_factor: float = 0.1
    risk_reward: float = 0.1
    exploration: float = 0.1
    timing: float = 0.1
    regime_adaptation: float = 0.2
    drawdown_penalty: float = 0.5
    volatility_penalty: float = 0.1

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def for_phase(cls, phase: int, total_phases: int = 4) -> 'RewardWeights':
        """
        Get phase-appropriate weights for curriculum learning.

        Sprint 2 restructuring:
        Phase 1 (BASE): Profit + exploration, minimal penalties
        Phase 2 (ENRICHED): Add Sharpe + risk-reward focus
        Phase 3 (SOFT): Add drawdown control, reduce exploration
        Phase 4 (PRODUCTION): Full risk-adjusted, low exploration
        """
        progress = phase / total_phases

        return cls(
            profit=1.0,                             # Always the primary signal
            sharpe=0.2 + 0.6 * progress,            # Grows: risk-adjusted becomes dominant
            sortino=0.1 + 0.5 * progress,           # Grows: downside awareness
            calmar=0.1 + 0.4 * progress,            # Grows: drawdown-adjusted returns
            win_rate=0.2 - 0.15 * progress,         # Shrinks: less hand-holding late
            profit_factor=0.1 + 0.3 * progress,     # Grows: gross profit/loss ratio
            risk_reward=0.2 + 0.3 * progress,       # Grows: Sprint 2 RR-based bonus
            exploration=0.2 - 0.18 * progress,      # High early, near-zero late
            timing=0.1 + 0.2 * progress,            # Entry/exit quality
            regime_adaptation=0.1 + 0.3 * progress, # Adapt to vol regimes
            drawdown_penalty=0.5 + 0.5 * progress,  # Sprint 2: stricter (was 0.3+0.4)
            volatility_penalty=0.1 + 0.2 * progress
        )


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_price: float
    exit_price: float
    entry_step: int
    exit_step: int
    pnl: float
    pnl_percent: float
    is_long: bool
    holding_duration: int
    max_favorable_excursion: float  # Best unrealized profit
    max_adverse_excursion: float    # Worst unrealized loss


class AdvancedRewardShaper:
    """
    Sophisticated reward shaper with multi-objective optimization.

    Key Features:
    1. Rolling window metrics (Sharpe, Sortino, Calmar)
    2. Trade quality metrics (win rate, profit factor)
    3. Continuous drawdown penalty (not just end-of-episode)
    4. Exploration bonus with decay
    5. Regime-aware reward scaling
    6. Intrinsic curiosity reward
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
        weights: Optional[RewardWeights] = None,
        window_size: int = 50,
        risk_free_rate: float = 0.02,
        trading_days: int = 252,
        target_sharpe: float = 2.0,
        target_win_rate: float = 0.55,
        max_drawdown_limit: float = 0.10,
        enable_curiosity: bool = True
    ):
        """
        Initialize the Advanced Reward Shaper.

        Args:
            initial_balance: Starting capital
            weights: RewardWeights object (None for defaults)
            window_size: Rolling window for metric calculation
            risk_free_rate: Annual risk-free rate
            trading_days: Trading days per year
            target_sharpe: Target Sharpe ratio
            target_win_rate: Target win rate
            max_drawdown_limit: Maximum allowed drawdown
            enable_curiosity: Enable intrinsic curiosity reward
        """
        self.initial_balance = initial_balance
        self.weights = weights or RewardWeights()
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.target_sharpe = target_sharpe
        self.target_win_rate = target_win_rate
        self.max_drawdown_limit = max_drawdown_limit
        self.enable_curiosity = enable_curiosity

        self._logger = logging.getLogger(__name__)

        # Rolling buffers
        self._returns_buffer: deque = deque(maxlen=window_size)
        self._equity_buffer: deque = deque(maxlen=window_size)
        self._action_buffer: deque = deque(maxlen=window_size)

        # Trade tracking
        self._completed_trades: List[TradeRecord] = []
        self._current_trade: Optional[Dict] = None

        # Drawdown tracking
        self._equity_peak = initial_balance
        self._current_equity = initial_balance

        # State tracking
        self._step_count = 0
        self._last_observation = None

        # Curiosity module state
        self._observation_counts: Dict[int, int] = {}
        self._curiosity_scale = 0.1

    def reset(self):
        """Reset for new episode."""
        self._returns_buffer.clear()
        self._equity_buffer.clear()
        self._action_buffer.clear()
        self._completed_trades.clear()
        self._current_trade = None
        self._equity_peak = self.initial_balance
        self._current_equity = self.initial_balance
        self._step_count = 0
        self._last_observation = None
        # Don't reset curiosity counts (persistent across episodes)

    def set_weights(self, weights: RewardWeights):
        """Update reward weights (for curriculum learning)."""
        self.weights = weights

    def _compute_rolling_sharpe(self) -> float:
        """Compute rolling Sharpe ratio."""
        if len(self._returns_buffer) < 10:
            return 0.0

        returns = np.array(list(self._returns_buffer))
        if np.std(returns) == 0:
            return 0.0

        annualized_return = np.mean(returns) * self.trading_days
        annualized_vol = np.std(returns) * np.sqrt(self.trading_days)
        sharpe = (annualized_return - self.risk_free_rate) / annualized_vol

        return np.clip(sharpe, -5, 5)  # Clip extreme values

    def _compute_rolling_sortino(self) -> float:
        """Compute rolling Sortino ratio (downside risk only)."""
        if len(self._returns_buffer) < 10:
            return 0.0

        returns = np.array(list(self._returns_buffer))
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return self._compute_rolling_sharpe()  # Fallback

        annualized_return = np.mean(returns) * self.trading_days
        downside_vol = np.std(downside_returns) * np.sqrt(self.trading_days)
        sortino = (annualized_return - self.risk_free_rate) / downside_vol

        return np.clip(sortino, -5, 5)

    def _compute_calmar(self) -> float:
        """Compute Calmar ratio (return / max drawdown)."""
        if len(self._returns_buffer) < 10:
            return 0.0

        returns = np.array(list(self._returns_buffer))
        annualized_return = np.mean(returns) * self.trading_days

        # Calculate max drawdown from equity curve
        equity = np.array(list(self._equity_buffer))
        if len(equity) < 2:
            return 0.0

        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown)

        if max_dd == 0:
            return annualized_return * 10  # Arbitrary high value

        calmar = annualized_return / max_dd
        return np.clip(calmar, -10, 10)

    def _compute_trade_metrics(self) -> Dict[str, float]:
        """Compute metrics from completed trades."""
        if len(self._completed_trades) < 3:
            return {
                'win_rate': 0.5,
                'profit_factor': 1.0,
                'risk_reward': 1.0,
                'avg_holding': 0.0
            }

        wins = [t for t in self._completed_trades if t.pnl > 0]
        losses = [t for t in self._completed_trades if t.pnl < 0]

        win_rate = len(wins) / len(self._completed_trades)

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 2.0

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 1
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 2.0

        avg_holding = np.mean([t.holding_duration for t in self._completed_trades])

        return {
            'win_rate': win_rate,
            'profit_factor': np.clip(profit_factor, 0, 5),
            'risk_reward': np.clip(risk_reward, 0, 5),
            'avg_holding': avg_holding
        }

    def _compute_drawdown_penalty(self) -> float:
        """Compute continuous drawdown penalty."""
        if self._equity_peak == 0:
            return 0.0

        drawdown = (self._equity_peak - self._current_equity) / self._equity_peak

        if drawdown <= 0:
            return 0.0

        # Progressive penalty: gentle for small DD, severe for large
        # Uses a modified sigmoid that accelerates as DD approaches limit
        dd_ratio = drawdown / self.max_drawdown_limit
        penalty = dd_ratio ** 2 * (1 + dd_ratio)  # Quadratic with cubic kicker

        # Hard cap at limit
        if drawdown >= self.max_drawdown_limit:
            penalty += 2.0  # Additional severe penalty

        return penalty

    def _compute_exploration_bonus(self, action: int) -> float:
        """Compute exploration bonus for action diversity."""
        if len(self._action_buffer) < 10:
            return 0.0

        # Calculate action entropy
        actions = np.array(list(self._action_buffer))
        _, counts = np.unique(actions, return_counts=True)
        probs = counts / len(actions)
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normalize to [0, 1] (max entropy for 5 actions is log(5) ≈ 1.61)
        max_entropy = np.log(5)
        normalized_entropy = entropy / max_entropy

        # Decay over time (encourage early exploration)
        decay = np.exp(-self._step_count / 10000)

        return normalized_entropy * decay

    def _compute_curiosity_reward(self, observation: np.ndarray) -> float:
        """Compute intrinsic curiosity reward for novel states."""
        if not self.enable_curiosity or observation is None:
            return 0.0

        # Discretize observation for counting
        # Use coarse binning (5 bins per dimension, but only sample first 10 dims)
        sampled_obs = observation[:10]
        bins = np.digitize(sampled_obs, np.linspace(-3, 3, 5))
        state_hash = hash(tuple(bins))

        # Count visits
        visit_count = self._observation_counts.get(state_hash, 0)
        self._observation_counts[state_hash] = visit_count + 1

        # Curiosity reward: 1/sqrt(n+1)
        curiosity = self._curiosity_scale / np.sqrt(visit_count + 1)

        return curiosity

    def _compute_timing_quality(
        self,
        action: int,
        regime_trend: float,
        regime_momentum: float
    ) -> float:
        """
        Compute reward for good entry/exit timing.

        Good timing:
        - Open long when trend is positive and momentum increasing
        - Open short when trend is negative and momentum decreasing
        - Close positions at reversal signals
        """
        timing_score = 0.0

        if action == 1:  # OPEN_LONG
            timing_score = regime_trend * (1 + regime_momentum)
        elif action == 3:  # OPEN_SHORT
            timing_score = -regime_trend * (1 - regime_momentum)
        elif action == 2:  # CLOSE_LONG
            # Good to close when trend reversing
            timing_score = -regime_trend if regime_momentum < 0 else 0
        elif action == 4:  # CLOSE_SHORT
            timing_score = regime_trend if regime_momentum > 0 else 0

        return np.clip(timing_score, -1, 1)

    def _compute_regime_adaptation(
        self,
        action: int,
        regime_trend: float,
        regime_volatility: float
    ) -> float:
        """
        Reward for adapting behavior to market regime.

        In trending markets: reward following the trend
        In ranging markets: reward quick scalping
        In high volatility: reward position reduction
        """
        adaptation_score = 0.0

        is_trending = abs(regime_trend) > 0.3
        is_ranging = abs(regime_trend) < 0.15
        is_volatile = regime_volatility > 0.7

        if is_trending:
            # Trend following: reward positions aligned with trend
            if action == 1 and regime_trend > 0:  # Long in uptrend
                adaptation_score = 0.5
            elif action == 3 and regime_trend < 0:  # Short in downtrend
                adaptation_score = 0.5
            elif action == 0:  # HOLD in trend is ok
                adaptation_score = 0.1

        elif is_ranging:
            # Mean reversion: reward quick trades
            recent_actions = list(self._action_buffer)[-5:] if len(self._action_buffer) >= 5 else []
            if action in [2, 4] and len(recent_actions) > 0:  # Closing positions
                adaptation_score = 0.3

        if is_volatile:
            # Risk reduction in high volatility
            if action in [0, 2, 4]:  # HOLD or close
                adaptation_score += 0.2
            elif action in [1, 3]:  # Opening positions in high vol
                adaptation_score -= 0.3

        return np.clip(adaptation_score, -1, 1)

    def compute_reward(
        self,
        raw_pnl: float,
        action: int,
        current_equity: float,
        observation: np.ndarray,
        agent_signals: Optional[Dict[str, float]] = None,
        trade_completed: bool = False,
        trade_record: Optional[TradeRecord] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the total shaped reward.

        Args:
            raw_pnl: Raw profit/loss from the step
            action: Action taken
            current_equity: Current portfolio equity
            observation: Current observation
            agent_signals: Signals from mock agents (trend, volatility, etc.)
            trade_completed: Whether a trade was closed this step
            trade_record: Record of completed trade (if any)

        Returns:
            Tuple of (total_reward, breakdown_dict)
        """
        self._step_count += 1

        # Update tracking
        prev_equity = self._current_equity
        self._current_equity = current_equity
        self._equity_peak = max(self._equity_peak, current_equity)

        step_return = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        self._returns_buffer.append(step_return)
        self._equity_buffer.append(current_equity)
        self._action_buffer.append(action)

        if trade_completed and trade_record:
            self._completed_trades.append(trade_record)

        # Extract agent signals
        regime_trend = agent_signals.get('regime_trend', 0.0) if agent_signals else 0.0
        regime_volatility = agent_signals.get('regime_volatility', 0.5) if agent_signals else 0.5
        regime_momentum = agent_signals.get('regime_momentum', 0.0) if agent_signals else 0.0

        # Compute all reward components
        components = {}

        # 1. Raw profit (scaled)
        profit_reward = raw_pnl / self.initial_balance * 100  # Per-step % return
        components['profit'] = profit_reward

        # 2. Sharpe-based reward
        sharpe = self._compute_rolling_sharpe()
        sharpe_reward = (sharpe - self.target_sharpe / 2) / self.target_sharpe  # Normalized
        components['sharpe'] = sharpe_reward

        # 3. Sortino-based reward
        sortino = self._compute_rolling_sortino()
        sortino_reward = (sortino - self.target_sharpe / 2) / self.target_sharpe
        components['sortino'] = sortino_reward

        # 4. Calmar-based reward
        calmar = self._compute_calmar()
        calmar_reward = np.tanh(calmar / 3)  # Normalize to [-1, 1]
        components['calmar'] = calmar_reward

        # 5. Trade metrics
        trade_metrics = self._compute_trade_metrics()
        win_rate_reward = (trade_metrics['win_rate'] - self.target_win_rate) * 2
        components['win_rate'] = win_rate_reward
        components['profit_factor'] = np.tanh(trade_metrics['profit_factor'] - 1)
        components['risk_reward'] = np.tanh(trade_metrics['risk_reward'] - 1)

        # 6. Drawdown penalty (negative)
        dd_penalty = self._compute_drawdown_penalty()
        components['drawdown_penalty'] = -dd_penalty

        # 7. Volatility penalty (equity volatility, not market)
        if len(self._returns_buffer) >= 10:
            equity_vol = np.std(list(self._returns_buffer))
            vol_penalty = equity_vol * 10  # Penalize high equity volatility
            components['volatility_penalty'] = -vol_penalty
        else:
            components['volatility_penalty'] = 0.0

        # 8. Exploration bonus
        exploration = self._compute_exploration_bonus(action)
        components['exploration'] = exploration

        # 9. Timing quality
        timing = self._compute_timing_quality(action, regime_trend, regime_momentum)
        components['timing'] = timing

        # 10. Regime adaptation
        regime_adapt = self._compute_regime_adaptation(action, regime_trend, regime_volatility)
        components['regime_adaptation'] = regime_adapt

        # 11. Curiosity (intrinsic motivation)
        curiosity = self._compute_curiosity_reward(observation)
        components['curiosity'] = curiosity

        # Compute weighted total
        w = self.weights
        total_reward = (
            w.profit * components['profit'] +
            w.sharpe * components['sharpe'] +
            w.sortino * components['sortino'] +
            w.calmar * components['calmar'] +
            w.win_rate * components['win_rate'] +
            w.profit_factor * components['profit_factor'] +
            w.risk_reward * components['risk_reward'] +
            w.exploration * components['exploration'] +
            w.timing * components['timing'] +
            w.regime_adaptation * components['regime_adaptation'] +
            w.drawdown_penalty * components['drawdown_penalty'] +
            w.volatility_penalty * components['volatility_penalty'] +
            curiosity  # Always included
        )

        # Final normalization (tanh + scale)
        final_reward = np.tanh(total_reward * 0.3) * 5.0

        # Store for debugging
        self._last_observation = observation

        breakdown = {
            'raw_components': components,
            'weights': w.to_dict(),
            'total_pre_norm': total_reward,
            'final_reward': final_reward,
            'rolling_sharpe': sharpe,
            'rolling_sortino': sortino,
            'rolling_calmar': calmar,
            'current_drawdown': (self._equity_peak - self._current_equity) / self._equity_peak,
            'trade_metrics': trade_metrics,
            'step': self._step_count
        }

        return final_reward, breakdown

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the episode."""
        return {
            'total_trades': len(self._completed_trades),
            'final_sharpe': self._compute_rolling_sharpe(),
            'final_sortino': self._compute_rolling_sortino(),
            'final_calmar': self._compute_calmar(),
            'max_drawdown': (self._equity_peak - min(list(self._equity_buffer))) / self._equity_peak if self._equity_buffer else 0,
            'trade_metrics': self._compute_trade_metrics(),
            'total_steps': self._step_count,
            'unique_states_visited': len(self._observation_counts)
        }
