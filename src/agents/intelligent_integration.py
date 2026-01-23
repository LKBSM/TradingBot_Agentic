# =============================================================================
# INTELLIGENT AGENTIC INTEGRATION - ML-Powered Trading Environment
# =============================================================================
# This module provides the INTELLIGENT integration layer that combines:
#   1. IntelligentRiskSentinel (ML-powered risk prediction)
#   2. MarketRegimeAgent (real-time regime detection)
#   3. Adaptive Position Sizing (Kelly + learning)
#
# This is the PRODUCTION-GRADE replacement for the basic AgenticTradingEnv.
#
# === KEY DIFFERENCES FROM BASIC INTEGRATION ===
#
# Basic AgenticTradingEnv:
#   - Rule-based risk checks (if drawdown > X: reject)
#   - Fixed position sizing
#   - No market context awareness
#
# IntelligentAgenticEnv:
#   - ML predicts risk BEFORE it happens
#   - Learns from every trade outcome
#   - Adapts position sizing based on performance
#   - Adjusts strategy based on market regime
#   - Provides regime information in observation
#
# === USAGE ===
#
#   from src.agents.intelligent_integration import create_intelligent_env
#
#   # Create environment with intelligent agents
#   env = create_intelligent_env(df, risk_preset="moderate")
#
#   # Use like normal - intelligence is automatic
#   obs, info = env.reset()
#   action = model.predict(obs)
#   obs, reward, done, truncated, info = env.step(action)
#
#   # Info now includes:
#   #   - info['regime']: Current market regime
#   #   - info['risk_prediction']: ML risk assessment
#   #   - info['position_multiplier']: Suggested size adjustment
#
# =============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

# Import our intelligent agents
from src.agents.intelligent_risk_sentinel import (
    IntelligentRiskSentinel,
    create_intelligent_risk_sentinel,
    MarketRegime
)
from src.agents.market_regime_agent import (
    MarketRegimeAgent,
    create_market_regime_agent,
    RegimeType,
    RegimeAnalysis
)
from src.agents.events import TradeProposal, DecisionType
from src.agents.config import RiskSentinelConfig, ConfigPreset, get_risk_sentinel_config

# Import the original environment
from src.environment.environment import TradingEnv

# Import action constants
try:
    from src.config import (
        ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG,
        ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT,
        POSITION_FLAT, POSITION_LONG, POSITION_SHORT,
        ACTION_NAMES, NUM_ACTIONS
    )
except ImportError:
    ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG = 0, 1, 2
    ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT = 3, 4
    POSITION_FLAT, POSITION_LONG, POSITION_SHORT = 0, 1, -1
    ACTION_NAMES = {0: 'HOLD', 1: 'OPEN_LONG', 2: 'CLOSE_LONG', 3: 'OPEN_SHORT', 4: 'CLOSE_SHORT'}
    NUM_ACTIONS = 5


# =============================================================================
# INTELLIGENT AGENTIC TRADING ENVIRONMENT
# =============================================================================

class IntelligentAgenticEnv(gym.Wrapper):
    """
    Intelligent Trading Environment with ML-Powered Agents.

    This wrapper adds three layers of intelligence to the basic TradingEnv:

    1. MARKET REGIME DETECTION
       - Classifies market into regimes (trending, ranging, volatile, etc.)
       - Adjusts strategy recommendations based on regime
       - Provides regime info to the RL agent via observation

    2. INTELLIGENT RISK SENTINEL
       - ML model predicts risk BEFORE trades
       - Learns from every trade outcome
       - Adapts to changing market conditions
       - Provides confidence scores for decisions

    3. ADAPTIVE POSITION SIZING
       - Kelly Criterion base
       - Adjusts based on recent win rate
       - Considers regime and prediction confidence
       - Reduces size after losses, increases on winning streaks

    === OBSERVATION SPACE ENHANCEMENT ===

    Original observation: [market_features, portfolio_state]
    Enhanced observation: [market_features, portfolio_state, regime_info]

    regime_info includes:
    - regime_type (one-hot encoded)
    - trend_strength
    - volatility_percentile
    - regime_confidence
    - position_multiplier
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        env: Optional[TradingEnv] = None,
        risk_preset: str = "moderate",
        risk_config: Optional[RiskSentinelConfig] = None,
        enable_regime_in_obs: bool = True,
        enable_logging: bool = True,
        **env_kwargs
    ):
        """
        Initialize the Intelligent Agentic Environment.

        Args:
            df: DataFrame with OHLCV data (required if env is None)
            env: Existing TradingEnv to wrap (optional)
            risk_preset: Risk preset ("conservative", "moderate", "aggressive", "backtesting")
            risk_config: Custom RiskSentinelConfig (overrides preset)
            enable_regime_in_obs: Add regime features to observation space
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

        # === CREATE INTELLIGENT AGENTS ===

        # 1. Risk Config
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

        # 2. Intelligent Risk Sentinel (ML-powered)
        self._risk_sentinel = create_intelligent_risk_sentinel(preset=risk_preset)
        self._risk_sentinel.start()

        # 3. Market Regime Agent
        self._regime_agent = create_market_regime_agent()
        self._regime_agent.start()

        # === OBSERVATION SPACE ENHANCEMENT ===
        self._enable_regime_in_obs = enable_regime_in_obs

        if enable_regime_in_obs:
            # Add 8 regime features to observation space
            # [regime_one_hot(8), trend_strength, volatility_pct, confidence, position_mult]
            original_obs_dim = self._base_env.observation_space.shape[0]
            new_obs_dim = original_obs_dim + 12  # 8 regime types + 4 metrics
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(new_obs_dim,),
                dtype=np.float32
            )

        # === TRACKING ===
        self._enable_logging = enable_logging
        self._logger = logging.getLogger("intelligent_agentic_env")

        # Statistics
        self._total_proposals = 0
        self._total_approvals = 0
        self._total_rejections = 0
        self._total_modifications = 0

        # Current state
        self._current_regime: Optional[RegimeAnalysis] = None
        self._last_risk_prediction: Optional[Dict] = None
        self._last_position_multiplier: float = 1.0

        # Trade tracking for learning
        self._pending_trade_entry: Optional[Dict] = None

        self._logger.info(
            f"IntelligentAgenticEnv initialized with {risk_preset} preset"
        )
        self._logger.info(
            f"  - Regime in obs: {enable_regime_in_obs}"
        )

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

        # Initialize regime with available price data
        self._update_regime()

        # Reset statistics for new episode
        self._total_proposals = 0
        self._total_approvals = 0
        self._total_rejections = 0
        self._total_modifications = 0
        self._pending_trade_entry = None

        # Enhance observation with regime info
        if self._enable_regime_in_obs:
            obs = self._enhance_observation(obs)

        # Add agent info
        info['intelligent_agents_active'] = True
        info['regime'] = self._current_regime.regime.value if self._current_regime else 'unknown'
        info['regime_confidence'] = self._current_regime.confidence if self._current_regime else 0.0

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with intelligent agent processing.

        The action from the RL agent is processed through:
        1. Market Regime Analysis
        2. Intelligent Risk Sentinel evaluation
        3. Adaptive position sizing

        Args:
            action: Action from RL agent (0-4)

        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        raw_action = action

        # === UPDATE REGIME ===
        self._update_regime()

        # === INTELLIGENT RISK GATE ===
        risk_actions = [ACTION_OPEN_LONG, ACTION_OPEN_SHORT]

        if action in risk_actions and self._risk_config.enabled:
            # Create trade proposal
            proposal = self._create_proposal(action)

            # Evaluate with Intelligent Risk Sentinel
            assessment = self._risk_sentinel.evaluate_trade(proposal)
            self._total_proposals += 1

            if assessment.decision == DecisionType.APPROVE:
                self._total_approvals += 1
                approved_action = action

                # Store for learning
                self._pending_trade_entry = {
                    'entry_price': proposal.entry_price,
                    'equity': proposal.current_equity,
                    'action': action
                }

            elif assessment.decision == DecisionType.MODIFY:
                self._total_modifications += 1
                approved_action = action  # Still execute, but note modification

                # Log modification
                if self._enable_logging and assessment.modified_params:
                    suggested = assessment.modified_params.get('suggested_quantity', 0)
                    self._logger.info(
                        f"Position size modified: {proposal.quantity:.4f} -> {suggested:.4f}"
                    )

            else:  # REJECT
                self._total_rejections += 1
                approved_action = ACTION_HOLD

                if self._enable_logging:
                    reason = assessment.reasoning[0] if assessment.reasoning else "Unknown"
                    self._logger.info(
                        f"Trade REJECTED: {ACTION_NAMES.get(action, 'UNKNOWN')} | {reason}"
                    )

            self._last_risk_prediction = {
                'decision': assessment.decision.name,
                'risk_score': assessment.risk_score,
                'risk_level': assessment.risk_level.name
            }
        else:
            approved_action = action
            self._last_risk_prediction = None

        # === EXECUTE ACTION ===
        obs, reward, done, truncated, info = self._base_env.step(approved_action)

        # === LEARN FROM TRADE OUTCOME ===
        if info.get('trade_details', {}).get('trade_success'):
            pnl = info['trade_details'].get('trade_pnl_abs', 0.0)
            pnl_pct = info['trade_details'].get('trade_pnl_pct', 0.0)

            # Record outcome for learning
            self._risk_sentinel.record_trade_outcome(
                pnl=pnl,
                pnl_pct=pnl_pct,
                max_adverse_excursion=0.0  # Could track this
            )

            self._pending_trade_entry = None

        # Record step in agents
        self._risk_sentinel.record_step()

        # === ENHANCE OBSERVATION ===
        if self._enable_regime_in_obs:
            obs = self._enhance_observation(obs)

        # === SYNC AGENT STATE ===
        self._sync_agent_state()

        # === AUGMENT INFO ===
        info['raw_action'] = raw_action
        info['approved_action'] = approved_action
        info['action_approved'] = (raw_action == approved_action)

        # Regime info
        if self._current_regime:
            info['regime'] = self._current_regime.regime.value
            info['regime_confidence'] = self._current_regime.confidence
            info['trend_direction'] = self._current_regime.trend_direction.value
            info['recommended_strategy'] = self._current_regime.recommended_strategy
            info['position_multiplier'] = self._current_regime.position_size_multiplier

        # Risk prediction info
        if self._last_risk_prediction:
            info['risk_prediction'] = self._last_risk_prediction

        return obs, reward, done, truncated, info

    def close(self) -> None:
        """Close the environment and stop agents."""
        self._risk_sentinel.stop()
        self._regime_agent.stop()
        self._base_env.close()

    # =========================================================================
    # REGIME HANDLING
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

        # Analyze regime
        self._current_regime = self._regime_agent.analyze(
            prices=prices,
            highs=highs,
            lows=lows,
            volumes=volumes
        )

        self._last_position_multiplier = self._current_regime.position_size_multiplier

    def _enhance_observation(self, obs: np.ndarray) -> np.ndarray:
        """Enhance observation with regime information."""
        if self._current_regime is None:
            # Add zeros if no regime info
            regime_features = np.zeros(12, dtype=np.float32)
        else:
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

            # Additional regime metrics
            regime_metrics = np.array([
                self._current_regime.trend_strength,
                self._current_regime.volatility_percentile / 100.0,
                self._current_regime.confidence,
                self._current_regime.position_size_multiplier
            ], dtype=np.float32)

            regime_features = np.concatenate([regime_one_hot, regime_metrics])

        return np.concatenate([obs, regime_features])

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

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
            'BOS_SIGNAL': float(current_row.get('BOS_SIGNAL', 0.0)),
        }

        # Calculate position size based on regime
        atr = market_data['ATR'] if market_data['ATR'] > 0 else market_data['Close'] * 0.01
        sl_distance = atr * 2.0

        # Base position sizing from risk manager
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
                'position_multiplier': self._last_position_multiplier
            }
        )

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

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_regime(self) -> Optional[RegimeAnalysis]:
        """Get current market regime analysis."""
        return self._current_regime

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all agents."""
        risk_stats = self._risk_sentinel.get_statistics()
        regime_stats = self._regime_agent.get_statistics()

        return {
            'episode_proposals': self._total_proposals,
            'episode_approvals': self._total_approvals,
            'episode_rejections': self._total_rejections,
            'episode_modifications': self._total_modifications,
            'approval_rate': self._total_approvals / max(1, self._total_proposals),
            'risk_sentinel': risk_stats,
            'regime_agent': regime_stats,
            'current_regime': self._current_regime.regime.value if self._current_regime else 'unknown'
        }

    def print_dashboard(self) -> None:
        """Print comprehensive dashboard from all agents."""
        print(self._risk_sentinel.get_risk_dashboard())

        stats = self.get_agent_stats()
        regime = stats['regime_agent']

        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    MARKET REGIME AGENT STATUS                        ║
╠══════════════════════════════════════════════════════════════════════╣
║ Current Regime:    {stats['current_regime']:15}                          ║
║ Regime Duration:   {regime['regime_duration']:>6} bars                          ║
║ Total Changes:     {regime['total_regime_changes']:>6}                               ║
║ Data Points:       {regime['data_points']:>6}                               ║
╚══════════════════════════════════════════════════════════════════════╝
""")

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

def create_intelligent_env(
    df: pd.DataFrame,
    risk_preset: str = "backtesting",
    enable_regime_in_obs: bool = True,
    **env_kwargs
) -> IntelligentAgenticEnv:
    """
    Create an IntelligentAgenticEnv with ML-powered agents.

    This is the recommended way to create a production trading environment.

    Args:
        df: DataFrame with OHLCV data
        risk_preset: One of "conservative", "moderate", "aggressive", "backtesting"
        enable_regime_in_obs: Add regime info to observation (recommended)
        **env_kwargs: Additional arguments passed to TradingEnv

    Returns:
        Configured IntelligentAgenticEnv

    Example:
        env = create_intelligent_env(df, risk_preset="moderate")
        obs, info = env.reset()

        print(f"Regime: {info['regime']}")
        print(f"Recommended strategy: {info['recommended_strategy']}")

        obs, reward, done, truncated, info = env.step(action)
    """
    return IntelligentAgenticEnv(
        df=df,
        risk_preset=risk_preset,
        enable_regime_in_obs=enable_regime_in_obs,
        **env_kwargs
    )


def upgrade_to_intelligent(
    env: TradingEnv,
    risk_preset: str = "backtesting"
) -> IntelligentAgenticEnv:
    """
    Upgrade an existing TradingEnv with intelligent agents.

    Use this when you already have a configured TradingEnv and want
    to add ML-powered risk management and regime detection.

    Args:
        env: Existing TradingEnv instance
        risk_preset: Risk configuration preset

    Returns:
        IntelligentAgenticEnv wrapping the original env
    """
    return IntelligentAgenticEnv(env=env, risk_preset=risk_preset)
