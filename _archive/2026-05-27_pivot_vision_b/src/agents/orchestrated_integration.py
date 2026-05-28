# =============================================================================
# ORCHESTRATED TRADING ENVIRONMENT - Full Multi-Agent Coordination
# =============================================================================
# This module provides the COMPLETE integration layer that combines ALL agents:
#   1. NewsAnalysisAgent (CRITICAL priority - blocks during high-impact events)
#   2. IntelligentRiskSentinel (HIGH priority - ML-powered risk management)
#   3. MarketRegimeAgent (HIGH priority - market context)
#   4. TradingOrchestrator (Coordinator - hierarchical decision making)
#
# This is the most advanced trading environment in the system.
#
# === KEY FEATURES ===
#
# 1. NEWS-AWARE TRADING
#    - Automatically blocks trading during high-impact news (FOMC, NFP, CPI)
#    - Reduces position sizes during medium-impact events
#    - Sentiment analysis affects position sizing
#
# 2. HIERARCHICAL DECISION MAKING
#    - CRITICAL agents (News) can override all others
#    - HIGH agents (Risk) can reject/modify trades
#    - Position size is the minimum of all recommendations
#
# 3. ENHANCED OBSERVATION SPACE
#    - Original features + Regime features (12) + News features (8)
#    - RL agent can learn to adapt to news/regime context
#
# === USAGE ===
#
#   from src.agents.orchestrated_integration import create_orchestrated_env
#
#   env = create_orchestrated_env(df, enable_news_blocking=True)
#   obs, info = env.reset()
#
#   # Observation now includes news/regime features
#   # info includes detailed decision breakdown
#
# =============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

# Import orchestrator and agents
from src.agents.orchestrator import (
    TradingOrchestrator,
    OrchestratorConfig,
    AgentPriority,
    OrchestratedDecision,
    create_trading_orchestrator
)
from src.agents.news_analysis_agent import (
    NewsAnalysisAgent,
    NewsAgentConfig,
    NewsAssessment,
    NewsDecision,
    create_news_analysis_agent
)
from src.agents.intelligent_risk_sentinel import (
    IntelligentRiskSentinel,
    create_intelligent_risk_sentinel
)
from src.agents.market_regime_agent import (
    MarketRegimeAgent,
    create_market_regime_agent,
    RegimeType,
    RegimeAnalysis
)
from src.agents.events import TradeProposal, DecisionType, EventBus
from src.agents.config import RiskSentinelConfig, ConfigPreset, get_risk_sentinel_config

# Import the original environment
from src.environment.environment import TradingEnv

# Import action constants and credit assignment penalties
try:
    from config import (
        ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG,
        ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT,
        POSITION_FLAT, POSITION_LONG, POSITION_SHORT,
        ACTION_NAMES, NUM_ACTIONS,
        # Credit assignment fix penalties
        RISK_REJECTION_PENALTY, RISK_MODIFICATION_PENALTY,
        NEWS_BLOCK_PENALTY, MODIFICATION_THRESHOLD
    )
except ImportError as _e:
    raise ImportError(
        f"OrchestratedTradingEnv requires config.py to be importable. "
        f"Silent fallback is disabled to prevent parameter mismatch. "
        f"Fix: pip install -e . from the project root. Original error: {_e}"
    ) from _e


# =============================================================================
# ORCHESTRATED TRADING ENVIRONMENT
# =============================================================================


class OrchestratedTradingEnv(gym.Wrapper):
    """
    Unified Trading Environment with Full Agent Orchestration.

    This environment combines all agents under a central orchestrator:

    1. NEWS ANALYSIS AGENT (CRITICAL)
       - Monitors economic calendar
       - Blocks trading during high-impact events
       - Provides sentiment scores

    2. INTELLIGENT RISK SENTINEL (HIGH)
       - ML-powered risk prediction
       - Adaptive position sizing
       - Learns from trade outcomes

    3. MARKET REGIME AGENT (HIGH)
       - Classifies market regime
       - Provides position multiplier
       - Context-aware recommendations

    4. TRADING ORCHESTRATOR
       - Coordinates all agents
       - Priority-based decision making
       - Aggregates position recommendations

    === OBSERVATION SPACE ===

    Original:  [market_features (303)]
    Enhanced:  [market_features (303) + regime_info (12) + news_info (8)]

    news_info includes:
    - is_blocked (1): Binary trading block status
    - impact_level (3): One-hot [HIGH, MEDIUM, LOW]
    - sentiment_score (1): -1 to +1
    - hours_to_event (1): Normalized 0-1
    - position_multiplier (1): From news assessment
    - confidence (1): News analysis confidence
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        env: Optional[TradingEnv] = None,
        risk_preset: str = "moderate",
        news_config: Optional[NewsAgentConfig] = None,
        orchestrator_config: Optional[OrchestratorConfig] = None,
        enable_news_blocking: bool = True,
        enable_regime_in_obs: bool = True,
        enable_news_in_obs: bool = True,
        enable_logging: bool = True,
        **env_kwargs
    ):
        """
        Initialize the Orchestrated Trading Environment.

        Args:
            df: DataFrame with OHLCV data (required if env is None)
            env: Existing TradingEnv to wrap (optional)
            risk_preset: Risk preset ("conservative", "moderate", "aggressive", "backtesting")
            news_config: Custom NewsAgentConfig
            orchestrator_config: Custom OrchestratorConfig
            enable_news_blocking: Whether to block during high-impact news
            enable_regime_in_obs: Add regime features to observation
            enable_news_in_obs: Add news features to observation
            enable_logging: Whether to log decisions
            **env_kwargs: Additional kwargs passed to TradingEnv
        """
        # === CREATE OR WRAP ENVIRONMENT ===
        if env is not None:
            self._base_env = env
        elif df is not None:
            self._base_env = TradingEnv(df, **env_kwargs)
        else:
            raise ValueError("Either df or env must be provided")

        # Initialize gym.Wrapper
        super().__init__(self._base_env)

        # === CONFIGURATION ===
        self._enable_news_blocking = enable_news_blocking
        self._enable_regime_in_obs = enable_regime_in_obs
        self._enable_news_in_obs = enable_news_in_obs
        self._enable_logging = enable_logging

        # === CREATE EVENT BUS ===
        self._event_bus = EventBus(enable_logging=enable_logging)

        # === CREATE ORCHESTRATOR ===
        self._orchestrator_config = orchestrator_config or OrchestratorConfig()
        self._orchestrator = TradingOrchestrator(
            config=self._orchestrator_config,
            event_bus=self._event_bus
        )

        # === CREATE AND REGISTER AGENTS ===

        # 1. News Analysis Agent (CRITICAL priority)
        self._news_config = news_config or NewsAgentConfig()
        if enable_news_blocking:
            self._news_agent = NewsAnalysisAgent(
                config=self._news_config,
                event_bus=self._event_bus
            )
            self._orchestrator.register_agent(
                self._news_agent,
                priority=AgentPriority.CRITICAL,
                is_critical=True
            )
        else:
            self._news_agent = None

        # 2. Intelligent Risk Sentinel (HIGH priority)
        preset_map = {
            "conservative": "conservative",
            "moderate": "moderate",
            "aggressive": "aggressive",
            "backtesting": "backtesting"
        }
        self._risk_sentinel = create_intelligent_risk_sentinel(
            preset=preset_map.get(risk_preset, "moderate")
        )
        self._orchestrator.register_agent(
            self._risk_sentinel,
            priority=AgentPriority.HIGH,
            is_critical=False
        )

        # 3. Market Regime Agent (HIGH priority)
        self._regime_agent = create_market_regime_agent()
        self._orchestrator.register_agent(
            self._regime_agent,
            priority=AgentPriority.HIGH,
            is_critical=False
        )

        # Start all agents via orchestrator
        self._orchestrator.start_all()

        # === OBSERVATION SPACE ENHANCEMENT ===
        original_obs_dim = self._base_env.observation_space.shape[0]
        extra_dims = 0

        if enable_regime_in_obs:
            extra_dims += 12  # 8 regime one-hot + 4 metrics

        if enable_news_in_obs and enable_news_blocking:
            extra_dims += 8  # News features

        if extra_dims > 0:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(original_obs_dim + extra_dims,),
                dtype=np.float32
            )

        # === TRACKING ===
        self._logger = logging.getLogger("orchestrated_trading_env")

        # Statistics
        self._total_proposals = 0
        self._total_approvals = 0
        self._total_rejections = 0
        self._total_modifications = 0
        self._total_blocks = 0

        # Current state
        self._current_regime: Optional[RegimeAnalysis] = None
        self._last_news_assessment: Optional[NewsAssessment] = None
        self._last_orchestrated_decision: Optional[OrchestratedDecision] = None

        # Trade tracking for learning
        self._pending_trade_entry: Optional[Dict] = None

        self._logger.info(
            f"OrchestratedTradingEnv initialized with {risk_preset} preset"
        )
        self._logger.info(f"  - News blocking: {enable_news_blocking}")
        self._logger.info(f"  - Regime in obs: {enable_regime_in_obs}")
        self._logger.info(f"  - News in obs: {enable_news_in_obs}")

    # =========================================================================
    # GYM INTERFACE
    # =========================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and all agents.

        Returns:
            Tuple of (observation, info)
        """
        # Reset base environment
        obs, info = self._base_env.reset(seed=seed, options=options)

        # Reset regime agent
        self._regime_agent.reset()

        # Initialize regime
        self._update_regime()

        # Get initial news assessment
        if self._news_agent:
            self._update_news_assessment()

        # Reset statistics
        self._total_proposals = 0
        self._total_approvals = 0
        self._total_rejections = 0
        self._total_modifications = 0
        self._total_blocks = 0
        self._pending_trade_entry = None

        # Enhance observation
        obs = self._enhance_observation(obs)

        # Add agent info
        info['orchestrated'] = True
        info['agents_registered'] = len(self._orchestrator.get_all_agents())

        if self._current_regime:
            info['regime'] = self._current_regime.regime.value
            info['regime_confidence'] = self._current_regime.confidence

        if self._last_news_assessment:
            info['news_blocked'] = not self._last_news_assessment.is_trading_allowed()
            info['news_sentiment'] = self._last_news_assessment.sentiment_score

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with orchestrated agent processing.

        The action from the RL agent is processed through:
        1. News Analysis (can block)
        2. Risk Sentinel (can reject/modify)
        3. Regime Agent (provides context)
        4. Orchestrator aggregates all decisions

        CREDIT ASSIGNMENT FIX:
        When an action is rejected/modified by the risk system, we apply a penalty
        to the reward so PPO learns "that action was bad in this state" rather than
        incorrectly learning "HOLD was neutral". This fixes the bug where PPO
        associated rewards from executing HOLD with the original (rejected) action.

        Args:
            action: Action from RL agent (0-4)

        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        raw_action = action

        # === UPDATE AGENTS ===
        self._update_regime()
        if self._news_agent:
            self._update_news_assessment()

        # === ORCHESTRATED DECISION ===
        risk_actions = [ACTION_OPEN_LONG, ACTION_OPEN_SHORT]

        # Credit assignment tracking
        credit_assignment_penalty = 0.0
        rejection_type = None  # 'news_block', 'risk_rejection', 'modification', None
        original_position_size = 0.0
        final_position_size = 0.0

        if action in risk_actions:
            # Create trade proposal
            proposal = self._create_proposal(action)
            self._total_proposals += 1
            original_position_size = proposal.quantity

            # Get orchestrated decision from all agents
            decision = self._orchestrator.coordinate_decision(proposal)
            self._last_orchestrated_decision = decision
            final_position_size = decision.final_position_size

            if decision.is_approved():
                if decision.final_decision == DecisionType.APPROVE:
                    self._total_approvals += 1
                else:
                    self._total_modifications += 1

                    # CREDIT ASSIGNMENT FIX: Check if modification is significant
                    if original_position_size > 0:
                        reduction_ratio = 1.0 - (final_position_size / original_position_size)
                        if reduction_ratio >= MODIFICATION_THRESHOLD:
                            # Significant reduction - apply modification penalty
                            credit_assignment_penalty = RISK_MODIFICATION_PENALTY * reduction_ratio
                            rejection_type = 'modification'

                            if self._enable_logging:
                                self._logger.debug(
                                    f"Credit assignment: modification penalty {credit_assignment_penalty:.3f} "
                                    f"(reduction: {reduction_ratio:.1%})"
                                )

                approved_action = action

                # Store for learning
                self._pending_trade_entry = {
                    'entry_price': proposal.entry_price,
                    'equity': proposal.current_equity,
                    'action': action,
                    'position_size': decision.final_position_size
                }

                if self._enable_logging and decision.final_decision == DecisionType.MODIFY:
                    self._logger.info(
                        f"Trade MODIFIED: size {proposal.quantity:.4f} -> "
                        f"{decision.final_position_size:.4f}"
                    )

            else:  # REJECT
                self._total_rejections += 1
                approved_action = ACTION_HOLD

                # CREDIT ASSIGNMENT FIX: Determine rejection type and apply penalty
                if decision.blocking_agent and 'News' in decision.blocking_agent:
                    self._total_blocks += 1
                    credit_assignment_penalty = NEWS_BLOCK_PENALTY
                    rejection_type = 'news_block'
                else:
                    credit_assignment_penalty = RISK_REJECTION_PENALTY
                    rejection_type = 'risk_rejection'

                if self._enable_logging:
                    reason = decision.reasoning[0] if decision.reasoning else "Unknown"
                    self._logger.info(
                        f"Trade REJECTED: {ACTION_NAMES.get(action, 'UNKNOWN')} | "
                        f"Blocked by: {decision.blocking_agent or 'Unknown'} | {reason} | "
                        f"Credit penalty: {credit_assignment_penalty:.2f}"
                    )
        else:
            approved_action = action
            self._last_orchestrated_decision = None

        # === EXECUTE ACTION ===
        obs, reward, done, truncated, info = self._base_env.step(approved_action)

        # === CREDIT ASSIGNMENT FIX: Apply rejection/modification penalty ===
        # This ensures PPO learns "the original action was bad" not "HOLD was neutral"
        original_reward = reward
        if credit_assignment_penalty > 0:
            reward = reward - credit_assignment_penalty

            if self._enable_logging:
                self._logger.debug(
                    f"Credit assignment applied: reward {original_reward:.4f} -> {reward:.4f} "
                    f"(penalty: -{credit_assignment_penalty:.4f}, type: {rejection_type})"
                )

        # === LEARN FROM TRADE OUTCOME ===
        if info.get('trade_details', {}).get('trade_success'):
            pnl = info['trade_details'].get('trade_pnl_abs', 0.0)
            pnl_pct = info['trade_details'].get('trade_pnl_pct', 0.0)

            # Record outcome for Risk Sentinel learning
            self._risk_sentinel.record_trade_outcome(
                pnl=pnl,
                pnl_pct=pnl_pct,
                max_adverse_excursion=0.0
            )

            self._pending_trade_entry = None

        # Record step
        self._risk_sentinel.record_step()

        # === ENHANCE OBSERVATION ===
        obs = self._enhance_observation(obs)

        # === SYNC AGENT STATE ===
        self._sync_agent_state()

        # === AUGMENT INFO ===
        info['raw_action'] = raw_action
        info['approved_action'] = approved_action
        info['action_approved'] = (raw_action == approved_action)

        # CREDIT ASSIGNMENT FIX: Add info for custom training callbacks
        # These fields allow advanced users to implement buffer correction if needed
        info['credit_assignment'] = {
            'penalty_applied': credit_assignment_penalty,
            'rejection_type': rejection_type,
            'original_reward': original_reward,
            'adjusted_reward': reward,
            'original_position_size': original_position_size,
            'final_position_size': final_position_size,
            'action_was_rejected': rejection_type in ['news_block', 'risk_rejection'],
            'action_was_modified': rejection_type == 'modification'
        }

        # Regime info
        if self._current_regime:
            info['regime'] = self._current_regime.regime.value
            info['regime_confidence'] = self._current_regime.confidence
            info['trend_direction'] = self._current_regime.trend_direction.value
            info['position_multiplier'] = self._current_regime.position_size_multiplier

        # News info
        if self._last_news_assessment:
            info['news_decision'] = self._last_news_assessment.decision.value
            info['news_blocked'] = not self._last_news_assessment.is_trading_allowed()
            info['news_sentiment'] = self._last_news_assessment.sentiment_score
            info['news_impact'] = self._last_news_assessment.current_impact_level.value
            info['hours_to_event'] = self._last_news_assessment.hours_to_next_high_impact

        # Orchestrated decision info
        if self._last_orchestrated_decision:
            info['orchestrated_decision'] = self._last_orchestrated_decision.final_decision.name
            info['orchestrated_confidence'] = self._last_orchestrated_decision.confidence
            info['blocking_agent'] = self._last_orchestrated_decision.blocking_agent
            info['agent_decisions'] = self._last_orchestrated_decision.agent_decisions

        return obs, reward, done, truncated, info

    def close(self) -> None:
        """Close the environment and stop all agents."""
        self._orchestrator.stop_all()
        self._base_env.close()

    # =========================================================================
    # OBSERVATION ENHANCEMENT
    # =========================================================================

    def _enhance_observation(self, obs: np.ndarray) -> np.ndarray:
        """Enhance observation with regime and news features."""
        features_to_add = []

        # Add regime features
        if self._enable_regime_in_obs:
            regime_features = self._get_regime_features()
            features_to_add.append(regime_features)

        # Add news features
        if self._enable_news_in_obs and self._news_agent:
            news_features = self._get_news_features()
            features_to_add.append(news_features)

        if features_to_add:
            return np.concatenate([obs] + features_to_add)

        return obs

    def _get_regime_features(self) -> np.ndarray:
        """Get regime features for observation."""
        if self._current_regime is None:
            return np.zeros(12, dtype=np.float32)

        # One-hot encode regime type (8 types)
        regime_one_hot = np.zeros(8, dtype=np.float32)
        regime_idx = {
            RegimeType.STRONG_UPTREND: 0,
            RegimeType.WEAK_UPTREND: 1,
            RegimeType.STRONG_DOWNTREND: 2,
            RegimeType.WEAK_DOWNTREND: 3,
            RegimeType.RANGING: 4,
            RegimeType.HIGH_VOLATILITY: 5,
            RegimeType.LOW_VOLATILITY: 6,
            RegimeType.TRANSITION: 7
        }.get(self._current_regime.regime, 7)
        regime_one_hot[regime_idx] = 1.0

        # Additional metrics
        regime_metrics = np.array([
            self._current_regime.trend_strength,
            self._current_regime.volatility_percentile / 100.0,
            self._current_regime.confidence,
            self._current_regime.position_size_multiplier
        ], dtype=np.float32)

        return np.concatenate([regime_one_hot, regime_metrics])

    def _get_news_features(self) -> np.ndarray:
        """Get news features for observation."""
        if self._last_news_assessment is None:
            return np.zeros(8, dtype=np.float32)

        assessment = self._last_news_assessment

        # Feature 1: Is blocked (binary)
        is_blocked = 1.0 if assessment.decision == NewsDecision.BLOCK else 0.0

        # Features 2-4: Impact level one-hot (HIGH, MEDIUM, LOW)
        from src.agents.news.economic_calendar import NewsImpact
        impact_one_hot = np.zeros(3, dtype=np.float32)
        impact_idx = {
            NewsImpact.HIGH: 0,
            NewsImpact.MEDIUM: 1,
            NewsImpact.LOW: 2
        }.get(assessment.current_impact_level, 2)
        impact_one_hot[impact_idx] = 1.0

        # Feature 5: Sentiment score (-1 to +1)
        sentiment = assessment.sentiment_score

        # Feature 6: Hours to next event (normalized 0-1, capped at 24h)
        hours_normalized = min(1.0, assessment.hours_to_next_high_impact / 24.0)

        # Feature 7: Position multiplier
        position_mult = assessment.position_multiplier

        # Feature 8: Confidence
        confidence = assessment.sentiment_confidence

        return np.array([
            is_blocked,
            impact_one_hot[0],  # HIGH
            impact_one_hot[1],  # MEDIUM
            impact_one_hot[2],  # LOW
            sentiment,
            hours_normalized,
            position_mult,
            confidence
        ], dtype=np.float32)

    # =========================================================================
    # AGENT UPDATES
    # =========================================================================

    def _update_regime(self) -> None:
        """Update market regime from current price data."""
        env = self._base_env

        # Get price history
        start_idx = max(0, env.current_step - 100)
        end_idx = env.current_step + 1

        if end_idx <= start_idx:
            return

        prices = env.df['Close'].iloc[start_idx:end_idx].values
        highs = env.df['High'].iloc[start_idx:end_idx].values if 'High' in env.df.columns else None
        lows = env.df['Low'].iloc[start_idx:end_idx].values if 'Low' in env.df.columns else None
        volumes = env.df['Volume'].iloc[start_idx:end_idx].values if 'Volume' in env.df.columns else None

        self._current_regime = self._regime_agent.analyze(
            prices=prices,
            highs=highs,
            lows=lows,
            volumes=volumes
        )

    def _update_news_assessment(self) -> None:
        """Update news assessment for current state."""
        if not self._news_agent:
            return

        # Create a minimal proposal just for news assessment
        env = self._base_env
        current_row = env.df.iloc[env.current_step]

        proposal = TradeProposal(
            action="CHECK",
            asset="XAU/USD",
            quantity=0.0,
            entry_price=float(current_row['Close']),
            current_balance=env.balance,
            current_position=env.stock_quantity,
            current_equity=env.balance,
            market_data={'Close': float(current_row['Close'])}
        )

        self._last_news_assessment = self._news_agent.evaluate_news_impact(proposal)

    def _sync_agent_state(self) -> None:
        """Synchronize agent state with environment state."""
        env = self._base_env

        current_price = float(env.df.iloc[env.current_step]['Close'])
        position_type = getattr(env, 'position_type', POSITION_FLAT)
        entry_price = getattr(env, 'entry_price', 0.0)

        # Calculate equity
        if position_type == POSITION_LONG:
            equity = env.balance + env.stock_quantity * current_price
        elif position_type == POSITION_SHORT:
            short_pnl = abs(env.stock_quantity) * (entry_price - current_price)
            equity = env.balance + short_pnl
        else:
            equity = env.balance

        # Update Risk Sentinel
        self._risk_sentinel.update_portfolio_state(
            equity=equity,
            position=env.stock_quantity,
            entry_price=entry_price,
            current_step=env.current_step
        )

    def _create_proposal(self, action: int) -> TradeProposal:
        """Create a TradeProposal from current environment state."""
        env = self._base_env

        action_str = ACTION_NAMES.get(action, "HOLD")

        # Get current market data
        current_row = env.df.iloc[env.current_step]
        market_data = {
            'Close': float(current_row['Close']),
            'ATR': float(current_row.get('ATR', 0.0)),
            'RSI': float(current_row.get('RSI', 50.0)),
            'Volume': float(current_row.get('Volume', 0.0)),
        }

        # Calculate position size
        atr = market_data['ATR'] if market_data['ATR'] > 0 else market_data['Close'] * 0.01
        sl_distance = atr * 2.0

        quantity = env.risk_manager.calculate_adaptive_position_size(
            client_id=getattr(env, "_risk_client_id", "default_client"),
            account_equity=env.balance,
            atr_stop_distance=sl_distance,
            win_prob=0.5,
            risk_reward_ratio=1.0,
            is_long=(action == ACTION_OPEN_LONG)
        )

        # Apply regime multiplier
        if self._current_regime:
            quantity *= self._current_regime.position_size_multiplier

        # Apply news multiplier
        if self._last_news_assessment:
            quantity *= self._last_news_assessment.position_multiplier

        # Calculate equity
        position_type = getattr(env, 'position_type', POSITION_FLAT)
        if position_type == POSITION_LONG:
            current_equity = env.balance + env.stock_quantity * market_data['Close']
        elif position_type == POSITION_SHORT:
            entry_price = getattr(env, 'entry_price', market_data['Close'])
            short_pnl = abs(env.stock_quantity) * (entry_price - market_data['Close'])
            current_equity = env.balance + short_pnl
        else:
            current_equity = env.balance

        return TradeProposal(
            action=action_str,
            asset="XAU/USD",
            quantity=float(quantity),
            entry_price=market_data['Close'],
            current_balance=env.balance,
            current_position=env.stock_quantity,
            current_equity=current_equity,
            unrealized_pnl=0.0,
            market_data=market_data,
            metadata={
                'step': env.current_step,
                'regime': self._current_regime.regime.value if self._current_regime else 'unknown',
                'news_impact': self._last_news_assessment.current_impact_level.value if self._last_news_assessment else 'none'
            }
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from orchestrator and all agents."""
        orch_stats = self._orchestrator.get_statistics()
        risk_stats = self._risk_sentinel.get_statistics()
        regime_stats = self._regime_agent.get_statistics()

        news_stats = {}
        if self._news_agent:
            news_stats = self._news_agent.get_statistics()

        return {
            'episode_stats': {
                'total_proposals': self._total_proposals,
                'approvals': self._total_approvals,
                'rejections': self._total_rejections,
                'modifications': self._total_modifications,
                'news_blocks': self._total_blocks,
                'approval_rate': self._total_approvals / max(1, self._total_proposals)
            },
            'orchestrator': orch_stats,
            'risk_sentinel': risk_stats,
            'regime_agent': regime_stats,
            'news_agent': news_stats,
            'current_state': {
                'regime': self._current_regime.regime.value if self._current_regime else 'unknown',
                'news_blocked': not self._last_news_assessment.is_trading_allowed() if self._last_news_assessment else False,
                'trading_enabled': self._orchestrator.is_trading_enabled()
            }
        }

    def print_dashboard(self) -> None:
        """Print comprehensive dashboard from all agents."""
        self._logger.info(self._orchestrator.get_dashboard())

        if self._news_agent:
            self._logger.info(self._news_agent.get_news_dashboard())

        self._logger.info(self._risk_sentinel.get_risk_dashboard())

    def is_trading_blocked(self) -> Tuple[bool, Optional[str]]:
        """Check if trading is currently blocked by any agent."""
        if self._last_news_assessment and not self._last_news_assessment.is_trading_allowed():
            return True, f"News: {self._last_news_assessment.reasoning[0] if self._last_news_assessment.reasoning else 'High-impact event'}"

        if not self._orchestrator.is_trading_enabled():
            return True, "Orchestrator disabled trading"

        return False, None

    @property
    def orchestrator(self) -> TradingOrchestrator:
        """Access the Trading Orchestrator directly."""
        return self._orchestrator

    @property
    def news_agent(self) -> Optional[NewsAnalysisAgent]:
        """Access the News Analysis Agent directly."""
        return self._news_agent

    @property
    def risk_sentinel(self) -> IntelligentRiskSentinel:
        """Access the Intelligent Risk Sentinel directly."""
        return self._risk_sentinel

    @property
    def regime_agent(self) -> MarketRegimeAgent:
        """Access the Market Regime Agent directly."""
        return self._regime_agent


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_orchestrated_env(
    df: pd.DataFrame,
    risk_preset: str = "backtesting",
    enable_news_blocking: bool = True,
    enable_regime_in_obs: bool = True,
    enable_news_in_obs: bool = True,
    newsapi_key: Optional[str] = None,
    **env_kwargs
) -> OrchestratedTradingEnv:
    """
    Create an OrchestratedTradingEnv with all agents.

    This is the recommended way to create a production trading environment
    with full news-aware, regime-aware trading.

    Args:
        df: DataFrame with OHLCV data
        risk_preset: One of "conservative", "moderate", "aggressive", "backtesting"
        enable_news_blocking: Block trading during high-impact events
        enable_regime_in_obs: Add regime info to observation
        enable_news_in_obs: Add news info to observation
        newsapi_key: Optional NewsAPI key for headline fetching
        **env_kwargs: Additional arguments passed to TradingEnv

    Returns:
        Configured OrchestratedTradingEnv

    Example:
        env = create_orchestrated_env(df, risk_preset="moderate")
        obs, info = env.reset()

        logger.info(f"Regime: {info['regime']}")
        logger.info(f"News blocked: {info.get('news_blocked', False)}")
        logger.info(f"Agents: {info['agents_registered']}")

        obs, reward, done, truncated, info = env.step(action)
    """
    news_config = None
    if newsapi_key:
        news_config = NewsAgentConfig(newsapi_key=newsapi_key)

    return OrchestratedTradingEnv(
        df=df,
        risk_preset=risk_preset,
        news_config=news_config,
        enable_news_blocking=enable_news_blocking,
        enable_regime_in_obs=enable_regime_in_obs,
        enable_news_in_obs=enable_news_in_obs,
        **env_kwargs
    )


def upgrade_to_orchestrated(
    env: TradingEnv,
    risk_preset: str = "backtesting",
    enable_news_blocking: bool = True
) -> OrchestratedTradingEnv:
    """
    Upgrade an existing TradingEnv with full orchestration.

    Use this when you already have a configured TradingEnv and want
    to add news-aware, orchestrated multi-agent trading.

    Args:
        env: Existing TradingEnv instance
        risk_preset: Risk configuration preset
        enable_news_blocking: Whether to enable news blocking

    Returns:
        OrchestratedTradingEnv wrapping the original env
    """
    return OrchestratedTradingEnv(
        env=env,
        risk_preset=risk_preset,
        enable_news_blocking=enable_news_blocking
    )
