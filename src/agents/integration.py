# =============================================================================
# AGENTIC INTEGRATION - Connect Agents to Trading Environment
# =============================================================================
# This module provides the integration layer between the Agentic AI system
# and the existing TradingEnv. It includes:
#
#   1. AgenticTradingEnv: A wrapper around TradingEnv that routes actions
#      through agents before execution
#
#   2. AgentOrchestrator: Manages multiple agents and coordinates decisions
#
#   3. Helper functions for easy setup
#
# === ARCHITECTURE ===
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │                         AgenticTradingEnv                               │
#   │  ┌─────────────┐                                    ┌───────────────┐  │
#   │  │  RL Agent   │──── action ────>┌────────────────┐ │  TradingEnv   │  │
#   │  │   (PPO)     │                 │  Risk Sentinel │─>│  (Execute)    │  │
#   │  └─────────────┘<── obs/reward ──│                │ │               │  │
#   │                                  └────────────────┘ └───────────────┘  │
#   └─────────────────────────────────────────────────────────────────────────┘
#
# === USAGE ===
#
#   from src.agents.integration import create_agentic_env
#
#   # Create environment with Risk Sentinel
#   env = create_agentic_env(df, risk_preset="moderate")
#
#   # Use like normal Gym environment
#   obs, info = env.reset()
#   action = agent.predict(obs)
#   obs, reward, done, truncated, info = env.step(action)
#
# =============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

# Import our agent system
from src.agents.base_agent import AgentState
from src.agents.events import (
    TradeProposal,
    RiskAssessment,
    DecisionType,
    EventBus
)
from src.agents.risk_sentinel import RiskSentinelAgent, create_risk_sentinel
from src.agents.config import RiskSentinelConfig, ConfigPreset, get_risk_sentinel_config

# Import the original environment
from src.environment.environment import TradingEnv

# Import action constants for long/short trading
try:
    from src.config import (
        ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG,
        ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT,
        POSITION_FLAT, POSITION_LONG, POSITION_SHORT,
        ACTION_NAMES, NUM_ACTIONS
    )
except ImportError:
    # Fallback definitions
    ACTION_HOLD = 0
    ACTION_OPEN_LONG = 1
    ACTION_CLOSE_LONG = 2
    ACTION_OPEN_SHORT = 3
    ACTION_CLOSE_SHORT = 4
    POSITION_FLAT = 0
    POSITION_LONG = 1
    POSITION_SHORT = -1
    ACTION_NAMES = {0: 'HOLD', 1: 'OPEN_LONG', 2: 'CLOSE_LONG', 3: 'OPEN_SHORT', 4: 'CLOSE_SHORT'}
    NUM_ACTIONS = 5


# =============================================================================
# AGENTIC TRADING ENVIRONMENT
# =============================================================================


class AgenticTradingEnv(gym.Wrapper):
    """
    Gym Wrapper that adds Agentic AI capabilities to TradingEnv.

    This wrapper intercepts all actions from the RL agent and routes them
    through the Risk Sentinel for approval before execution. If an action
    is rejected, it's replaced with HOLD (action=0).

    === KEY FEATURES ===

    1. Transparent Integration:
       - Works exactly like the original TradingEnv
       - Same observation space, action space, rewards
       - Existing training code works without changes

    2. Risk Gate:
       - Every trade proposal is evaluated by Risk Sentinel
       - Rejected trades become HOLD actions
       - Info dict contains rejection reasons

    3. State Synchronization:
       - Agent state is synced with environment state
       - Portfolio, positions, market data all shared

    4. Rich Logging:
       - All decisions are logged for audit
       - Statistics available via get_agent_stats()

    === USAGE ===

    ```python
    # Option 1: Simple creation
    env = AgenticTradingEnv(df, risk_preset="moderate")

    # Option 2: Custom config
    config = RiskSentinelConfig(max_drawdown_pct=0.08)
    env = AgenticTradingEnv(df, risk_config=config)

    # Option 3: Wrap existing env
    base_env = TradingEnv(df)
    env = AgenticTradingEnv.wrap(base_env)

    # Use normally
    obs, info = env.reset()
    action = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)

    # Check if action was approved
    if info.get('risk_approved'):
        print("Trade executed!")
    else:
        print(f"Trade rejected: {info.get('risk_reason')}")
    ```
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        env: Optional[TradingEnv] = None,
        risk_preset: str = "moderate",
        risk_config: Optional[RiskSentinelConfig] = None,
        enable_logging: bool = True,
        **env_kwargs
    ):
        """
        Initialize the Agentic Trading Environment.

        Args:
            df: DataFrame with OHLCV data (required if env is None)
            env: Existing TradingEnv to wrap (optional)
            risk_preset: Risk preset ("conservative", "moderate", "aggressive", "backtesting")
            risk_config: Custom RiskSentinelConfig (overrides preset)
            enable_logging: Whether to log decisions
            **env_kwargs: Additional kwargs passed to TradingEnv

        Note: Either df or env must be provided, not both.
        """
        # === CREATE OR WRAP ENVIRONMENT ===
        if env is not None:
            # Wrap existing environment
            self._base_env = env
        elif df is not None:
            # Create new environment
            self._base_env = TradingEnv(df, **env_kwargs)
        else:
            raise ValueError("Either df or env must be provided")

        # Initialize gym.Wrapper
        super().__init__(self._base_env)

        # === CREATE RISK SENTINEL AGENT ===
        if risk_config is not None:
            self._risk_config = risk_config
        else:
            preset_map = {
                "conservative": ConfigPreset.CONSERVATIVE,
                "moderate": ConfigPreset.MODERATE,
                "aggressive": ConfigPreset.AGGRESSIVE,
                "backtesting": ConfigPreset.BACKTESTING
            }
            self._risk_config = get_risk_sentinel_config(
                preset_map.get(risk_preset, ConfigPreset.MODERATE)
            )

        self._risk_sentinel = RiskSentinelAgent(
            config=self._risk_config,
            name="RiskSentinel-Env"
        )

        # Start the agent
        self._risk_sentinel.start()

        # === TRACKING ===
        self._enable_logging = enable_logging
        self._logger = logging.getLogger("agentic_env")

        # Decision statistics
        self._total_proposals = 0
        self._total_approvals = 0
        self._total_rejections = 0
        self._rejection_reasons: List[str] = []

        # Last decision (for info dict)
        self._last_assessment: Optional[RiskAssessment] = None
        self._last_raw_action: int = 0
        self._last_approved_action: int = 0

        self._logger.info(
            f"AgenticTradingEnv initialized with {risk_preset} risk preset"
        )

    @classmethod
    def wrap(
        cls,
        env: TradingEnv,
        risk_preset: str = "moderate",
        **kwargs
    ) -> 'AgenticTradingEnv':
        """
        Factory method to wrap an existing TradingEnv.

        Args:
            env: Existing TradingEnv instance
            risk_preset: Risk preset to use
            **kwargs: Additional configuration

        Returns:
            AgenticTradingEnv wrapping the original env
        """
        return cls(env=env, risk_preset=risk_preset, **kwargs)

    # =========================================================================
    # GYM INTERFACE
    # =========================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and agents.

        Returns:
            Tuple of (observation, info)
        """
        # Reset base environment
        obs, info = self._base_env.reset(seed=seed, options=options)

        # Reset agent state
        self._sync_agent_state()

        # Reset statistics for new episode
        self._total_proposals = 0
        self._total_approvals = 0
        self._total_rejections = 0
        self._rejection_reasons = []

        # Add agent info
        info['risk_sentinel_active'] = self._risk_config.enabled
        info['risk_preset'] = self._risk_config.__class__.__name__

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with Risk Sentinel gate.

        The action from the RL agent is first evaluated by the Risk Sentinel.
        If approved, the original action is executed.
        If rejected, HOLD (action=0) is executed instead.

        NEW ACTION SPACE (5 actions):
            0 = HOLD         : Do nothing
            1 = OPEN_LONG    : Buy to open long position
            2 = CLOSE_LONG   : Sell to close long position
            3 = OPEN_SHORT   : Sell to open short position
            4 = CLOSE_SHORT  : Buy to cover short position

        Args:
            action: Action from RL agent (0-4 for the 5 actions)

        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        self._last_raw_action = action

        # === RISK GATE ===
        # Only evaluate OPEN actions (CLOSE and HOLD are safe)
        # OPEN_LONG (1) and OPEN_SHORT (3) need risk evaluation
        risk_actions = [ACTION_OPEN_LONG, ACTION_OPEN_SHORT]
        if action in risk_actions and self._risk_config.enabled:
            # Create trade proposal
            proposal = self._create_proposal(action)

            # Evaluate with Risk Sentinel
            assessment = self._risk_sentinel.evaluate_trade(proposal)
            self._last_assessment = assessment
            self._total_proposals += 1

            if assessment.is_approved():
                # Trade approved - execute original action
                self._total_approvals += 1
                approved_action = action
            else:
                # Trade rejected - execute HOLD instead
                self._total_rejections += 1
                approved_action = 0  # HOLD

                # Track rejection reason
                if assessment.violations:
                    reason = assessment.violations[0].rule_description
                else:
                    reason = "Unknown"
                self._rejection_reasons.append(reason)

                if self._enable_logging:
                    self._logger.info(
                        f"Step {self._base_env.current_step}: "
                        f"Action {action} REJECTED -> HOLD | "
                        f"Reason: {reason}"
                    )
        else:
            # HOLD action or Risk Sentinel disabled
            approved_action = action
            self._last_assessment = None

        self._last_approved_action = approved_action

        # === EXECUTE ACTION ===
        obs, reward, done, truncated, info = self._base_env.step(approved_action)

        # === SYNC STATE AFTER STEP ===
        self._sync_agent_state()

        # Record step in Risk Sentinel
        self._risk_sentinel.record_step()

        # Record trade result if a trade was executed
        if info.get('trade_details', {}).get('trade_success'):
            pnl = info['trade_details'].get('trade_pnl_abs', 0.0)
            self._risk_sentinel.record_trade_result(pnl)

        # === AUGMENT INFO DICT ===
        info['risk_raw_action'] = self._last_raw_action
        info['risk_approved_action'] = approved_action
        info['risk_approved'] = (self._last_raw_action == approved_action)

        if self._last_assessment:
            info['risk_decision'] = self._last_assessment.decision.name
            info['risk_score'] = self._last_assessment.risk_score
            info['risk_level'] = self._last_assessment.risk_level.name
            if self._last_assessment.violations:
                info['risk_reason'] = self._last_assessment.violations[0].rule_description
            else:
                info['risk_reason'] = None
        else:
            info['risk_decision'] = 'N/A'
            info['risk_score'] = 0.0
            info['risk_level'] = 'LOW'
            info['risk_reason'] = None

        return obs, reward, done, truncated, info

    def close(self) -> None:
        """Close the environment and stop agents."""
        self._risk_sentinel.stop()
        self._base_env.close()

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _create_proposal(self, action: int) -> TradeProposal:
        """
        Create a TradeProposal from current environment state.

        NEW ACTION SPACE (5 actions):
            0 = HOLD         : Do nothing
            1 = OPEN_LONG    : Buy to open long position
            2 = CLOSE_LONG   : Sell to close long position
            3 = OPEN_SHORT   : Sell to open short position
            4 = CLOSE_SHORT  : Buy to cover short position

        Args:
            action: The proposed action (0-4)

        Returns:
            TradeProposal object for Risk Sentinel evaluation
        """
        env = self._base_env

        # Map action to string using the ACTION_NAMES from config
        # ACTION_NAMES = {0: 'HOLD', 1: 'OPEN_LONG', 2: 'CLOSE_LONG', 3: 'OPEN_SHORT', 4: 'CLOSE_SHORT'}
        action_str = ACTION_NAMES.get(action, "HOLD")

        # Get current market data
        current_row = env.df.iloc[env.current_step]
        market_data = {
            'Close': float(current_row['Close']),
            'ATR': float(current_row.get('ATR', 0.0)),
            'RSI': float(current_row.get('RSI', 50.0)),
            'Volume': float(current_row.get('Volume', 0.0)),
            'BOS_SIGNAL': float(current_row.get('BOS_SIGNAL', 0.0)),
        }

        # Calculate proposed quantity based on action type
        # OPEN_LONG (1) and OPEN_SHORT (3) calculate new position size
        # CLOSE_LONG (2) and CLOSE_SHORT (4) use existing position
        if action == ACTION_OPEN_LONG:
            # Calculate position size for opening a LONG position
            atr = market_data['ATR']
            if atr > 0:
                sl_distance = atr * env.risk_manager._get_regime_multiplier(
                    env.risk_manager.market_state.get('current_regime', 0)
                )
            else:
                sl_distance = market_data['Close'] * 0.01

            quantity = env.risk_manager.calculate_adaptive_position_size(
                client_id=getattr(env, "_risk_client_id", "default_client"),
                account_equity=env.balance,
                atr_stop_distance=sl_distance,
                win_prob=0.5,
                risk_reward_ratio=1.0,
                is_long=True  # Long position
            )
        elif action == ACTION_OPEN_SHORT:
            # Calculate position size for opening a SHORT position
            atr = market_data['ATR']
            if atr > 0:
                sl_distance = atr * env.risk_manager._get_regime_multiplier(
                    env.risk_manager.market_state.get('current_regime', 0)
                )
            else:
                sl_distance = market_data['Close'] * 0.01

            quantity = env.risk_manager.calculate_adaptive_position_size(
                client_id=getattr(env, "_risk_client_id", "default_client"),
                account_equity=env.balance,
                atr_stop_distance=sl_distance,
                win_prob=0.5,
                risk_reward_ratio=1.0,
                is_long=False  # Short position
            )
        elif action == ACTION_CLOSE_LONG:
            # Closing a long - use current position size
            quantity = env.stock_quantity
        elif action == ACTION_CLOSE_SHORT:
            # Closing a short - use current position size (absolute value)
            quantity = abs(env.stock_quantity)
        else:
            # HOLD
            quantity = 0.0

        # Determine if this is a long or short position for equity calculation
        position_type = getattr(env, 'position_type', POSITION_FLAT)
        current_position = env.stock_quantity

        # Calculate current equity accounting for position type
        if position_type == POSITION_LONG:
            # Long position: equity = balance + (quantity * current_price)
            current_equity = env.balance + current_position * market_data['Close']
            unrealized_pnl = current_position * (market_data['Close'] - getattr(env, 'entry_price', market_data['Close']))
        elif position_type == POSITION_SHORT:
            # Short position: equity = balance + (entry_value - current_value)
            entry_price = getattr(env, 'entry_price', market_data['Close'])
            short_pnl = abs(current_position) * (entry_price - market_data['Close'])
            current_equity = env.balance + short_pnl
            unrealized_pnl = short_pnl
        else:
            # Flat (no position)
            current_equity = env.balance
            unrealized_pnl = 0.0

        # Create proposal with all relevant information
        proposal = TradeProposal(
            action=action_str,
            asset="XAU/USD",
            quantity=float(quantity),
            entry_price=market_data['Close'],
            current_balance=env.balance,
            current_position=current_position,
            current_equity=current_equity,
            unrealized_pnl=unrealized_pnl,
            market_data=market_data,
            metadata={
                'step': env.current_step,
                'episode': getattr(env, 'episode_count', 0),
                'regime': env.risk_manager.market_state.get('current_regime', 0),
                'position_type': position_type,  # Track if LONG, SHORT, or FLAT
                'is_long': action == ACTION_OPEN_LONG,  # True for long, False for short
                'is_short': action == ACTION_OPEN_SHORT
            }
        )

        return proposal

    def _sync_agent_state(self) -> None:
        """
        Synchronize Risk Sentinel state with environment state.

        Called after reset and each step to keep agents informed of
        portfolio changes, position updates, etc.

        Properly handles both LONG and SHORT positions for equity calculation.
        """
        env = self._base_env

        # Get current price
        current_price = float(env.df.iloc[env.current_step]['Close'])

        # Get position type (FLAT=0, LONG=1, SHORT=-1)
        position_type = getattr(env, 'position_type', POSITION_FLAT)
        entry_price = getattr(env, 'entry_price', 0.0)

        # Calculate equity based on position type
        # This is crucial for correct risk calculations
        if position_type == POSITION_LONG:
            # Long position: equity = balance + (quantity * current_price)
            # Profit when price goes UP
            equity = env.balance + env.stock_quantity * current_price
        elif position_type == POSITION_SHORT:
            # Short position: equity = balance + unrealized_pnl
            # Profit when price goes DOWN
            # unrealized_pnl = quantity * (entry_price - current_price)
            short_quantity = abs(env.stock_quantity)
            unrealized_pnl = short_quantity * (entry_price - current_price)
            equity = env.balance + unrealized_pnl
        else:
            # Flat (no position) - equity equals balance
            equity = env.balance

        # Update Risk Sentinel with current portfolio state
        self._risk_sentinel.update_portfolio_state(
            equity=equity,
            position=env.stock_quantity,
            entry_price=entry_price,
            current_step=env.current_step
        )

        # Update market regime based on BOS signal
        bos_signal = float(env.df.iloc[env.current_step].get('BOS_SIGNAL', 0.0))
        regime = 0 if bos_signal != 0 else 1
        self._risk_sentinel.set_market_regime(regime)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the Risk Sentinel agent.

        Returns:
            Dictionary with agent statistics
        """
        sentinel_stats = self._risk_sentinel.get_statistics()

        return {
            'episode_proposals': self._total_proposals,
            'episode_approvals': self._total_approvals,
            'episode_rejections': self._total_rejections,
            'episode_approval_rate': (
                self._total_approvals / self._total_proposals
                if self._total_proposals > 0 else 1.0
            ),
            'top_rejection_reasons': self._get_top_rejection_reasons(),
            **sentinel_stats
        }

    def _get_top_rejection_reasons(self, n: int = 5) -> Dict[str, int]:
        """Get the top N rejection reasons."""
        from collections import Counter
        counts = Counter(self._rejection_reasons)
        return dict(counts.most_common(n))

    def get_risk_dashboard(self) -> str:
        """
        Get a formatted risk dashboard from the Risk Sentinel.

        Returns:
            Formatted string with current risk status
        """
        return self._risk_sentinel.get_risk_dashboard()

    def print_risk_summary(self) -> None:
        """Print a summary of risk decisions this episode."""
        stats = self.get_agent_stats()
        print("\n" + "=" * 60)
        print("RISK SENTINEL EPISODE SUMMARY")
        print("=" * 60)
        print(f"Total Proposals:    {stats['episode_proposals']}")
        print(f"Approved:           {stats['episode_approvals']}")
        print(f"Rejected:           {stats['episode_rejections']}")
        print(f"Approval Rate:      {stats['episode_approval_rate']:.1%}")
        print("-" * 60)
        print("Top Rejection Reasons:")
        for reason, count in stats['top_rejection_reasons'].items():
            print(f"  - {reason}: {count}")
        print("=" * 60 + "\n")

    @property
    def risk_sentinel(self) -> RiskSentinelAgent:
        """Access the Risk Sentinel agent directly."""
        return self._risk_sentinel

    @property
    def risk_config(self) -> RiskSentinelConfig:
        """Access the risk configuration."""
        return self._risk_config


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_agentic_env(
    df: pd.DataFrame,
    risk_preset: str = "backtesting",
    **env_kwargs
) -> AgenticTradingEnv:
    """
    Create an AgenticTradingEnv with the specified risk preset.

    This is the recommended way to create an agentic environment.

    Args:
        df: DataFrame with OHLCV data
        risk_preset: One of "conservative", "moderate", "aggressive", "backtesting"
        **env_kwargs: Additional arguments passed to TradingEnv

    Returns:
        Configured AgenticTradingEnv

    Example:
        env = create_agentic_env(df, risk_preset="moderate")
        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step(action)
    """
    return AgenticTradingEnv(
        df=df,
        risk_preset=risk_preset,
        **env_kwargs
    )


def wrap_existing_env(
    env: TradingEnv,
    risk_preset: str = "backtesting"
) -> AgenticTradingEnv:
    """
    Wrap an existing TradingEnv with Agentic capabilities.

    Use this when you already have a configured TradingEnv and want
    to add the Risk Sentinel gate.

    Args:
        env: Existing TradingEnv instance
        risk_preset: Risk configuration preset

    Returns:
        AgenticTradingEnv wrapping the original env
    """
    return AgenticTradingEnv.wrap(env, risk_preset=risk_preset)


# =============================================================================
# ORCHESTRATOR (For Future Multi-Agent Support)
# =============================================================================


class AgentOrchestrator:
    """
    Orchestrates multiple agents for complex decision-making.

    This is a foundation for Phase 2+ where we'll add more agents
    (Execution Agent, Research Agent, etc.) that need coordination.

    Currently supports:
        - Risk Sentinel Agent (Phase 1)

    Future support:
        - Execution Agent (Phase 2)
        - News Sentiment Agent (Phase 3)
        - Research Agent (Phase 4)
        - Portfolio Manager (Phase 5)
    """

    def __init__(self):
        """Initialize the orchestrator."""
        self._agents: Dict[str, 'BaseAgent'] = {}
        self._event_bus = EventBus()
        self._logger = logging.getLogger("orchestrator")

    def register_agent(self, agent: 'BaseAgent') -> None:
        """
        Register an agent with the orchestrator.

        Args:
            agent: Agent instance to register
        """
        self._agents[agent.full_id] = agent
        self._logger.info(f"Registered agent: {agent.full_id}")

    def start_all(self) -> None:
        """Start all registered agents."""
        for agent in self._agents.values():
            agent.start()
        self._logger.info(f"Started {len(self._agents)} agents")

    def stop_all(self) -> None:
        """Stop all registered agents."""
        for agent in self._agents.values():
            agent.stop()
        self._logger.info(f"Stopped {len(self._agents)} agents")

    def get_agent(self, agent_id: str) -> Optional['BaseAgent']:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all agents."""
        return {
            agent_id: agent.health_check()
            for agent_id, agent in self._agents.items()
        }
