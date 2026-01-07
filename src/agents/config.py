# =============================================================================
# AGENT CONFIGURATION - Settings for All Trading Agents
# =============================================================================
# This module defines configuration classes for all agents using Pydantic
# for validation. Each agent type has its own config class with:
#   - Default values (production-ready)
#   - Validation rules (prevent misconfiguration)
#   - Documentation (explain each parameter)
#
# Benefits of Pydantic:
#   - Runtime type validation
#   - Auto-generated documentation
#   - Easy serialization (JSON/YAML)
#   - Environment variable support
#
# =============================================================================

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

# =============================================================================
# BASE CONFIGURATION
# =============================================================================


@dataclass
class AgentConfig:
    """
    Base configuration for all agents.

    These settings apply to EVERY agent regardless of type.
    Subclasses add agent-specific settings.

    Attributes:
        enabled: Whether the agent is active (False = bypass)
        log_level: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
        max_retries: How many times to retry on failure
        timeout_ms: Maximum time for a single operation
        enable_metrics: Whether to collect performance metrics
        enable_audit_log: Whether to log all decisions for compliance
    """
    # --- Basic Settings ---
    enabled: bool = True                 # Is this agent active?
    log_level: str = "INFO"             # Logging verbosity
    max_retries: int = 3                # Retry count on failure
    timeout_ms: int = 1000              # Max operation time (1 second)

    # --- Monitoring ---
    enable_metrics: bool = True         # Collect performance metrics?
    enable_audit_log: bool = True       # Log all decisions?
    metrics_interval_sec: int = 60      # How often to emit metrics

    # --- Integration ---
    event_bus_enabled: bool = True      # Use event bus for communication?

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'enabled': self.enabled,
            'log_level': self.log_level,
            'max_retries': self.max_retries,
            'timeout_ms': self.timeout_ms,
            'enable_metrics': self.enable_metrics,
            'enable_audit_log': self.enable_audit_log,
            'metrics_interval_sec': self.metrics_interval_sec,
            'event_bus_enabled': self.event_bus_enabled
        }


# =============================================================================
# RISK SENTINEL CONFIGURATION
# =============================================================================


@dataclass
class RiskSentinelConfig(AgentConfig):
    """
    Configuration for the Risk Sentinel Agent.

    The Risk Sentinel is responsible for:
        1. Validating trade proposals against risk rules
        2. Monitoring portfolio risk levels
        3. Enforcing position limits and drawdown caps
        4. Detecting unusual market conditions

    === POSITION SIZING RULES ===
    These parameters control how much capital can be risked per trade.

    Attributes:
        max_position_size_pct: Maximum position as % of portfolio (default: 20%)
            Example: With $10,000 portfolio, max position = $2,000

        max_risk_per_trade_pct: Maximum risk per trade as % of portfolio (default: 1%)
            Example: With $10,000 portfolio, max risk = $100 loss
            This is the PROFESSIONAL STANDARD - never risk more than 1-2%

        min_position_size: Minimum trade size in units (default: 0.01)
            Prevents dust trades that aren't worth the fees

    === DRAWDOWN PROTECTION ===
    These parameters protect against catastrophic losses.

    Attributes:
        max_drawdown_pct: Maximum allowed drawdown before halting (default: 10%)
            If portfolio drops 10% from peak, ALL trading stops
            This is a HARD LIMIT - non-negotiable

        drawdown_warning_pct: Warning threshold (default: 7%)
            At 7% drawdown, risk limits are tightened

        daily_loss_limit_pct: Maximum loss allowed in a single day (default: 3%)
            Prevents bad days from becoming catastrophic

    === EXPOSURE LIMITS ===
    These parameters control overall portfolio exposure.

    Attributes:
        max_leverage: Maximum leverage ratio (default: 1.0 = no leverage)
            For safety, we don't use leverage in production

        max_open_positions: Maximum concurrent positions (default: 1)
            For single-asset bot, this is 1. Increase for multi-asset.

        max_correlation_exposure: Maximum exposure to correlated assets (default: 50%)
            Prevents overexposure to correlated risks

    === TRADE VALIDATION ===
    These parameters validate individual trades.

    Attributes:
        min_risk_reward_ratio: Minimum R:R to accept a trade (default: 1.5)
            If potential reward isn't 1.5x risk, reject

        max_spread_pct: Maximum acceptable spread (default: 0.1%)
            Reject trades during high-spread periods

        required_atr_multiplier: Stop loss must be at least N * ATR (default: 1.0)
            Ensures stops are not too tight

    === MARKET CONDITION FILTERS ===
    These parameters detect unusual market conditions.

    Attributes:
        volatility_spike_threshold: ATR multiple for "unusual" (default: 2.5)
            If current ATR > 2.5 * average ATR, flag as spike

        min_volume_percentile: Minimum volume to trade (default: 20)
            Don't trade if volume is in bottom 20%

        regime_check_enabled: Whether to check market regime (default: True)
            Uses HMM-detected regime for risk adjustment

    === BEHAVIORAL RULES ===
    These parameters prevent bad trading behavior.

    Attributes:
        cooldown_after_loss_steps: Steps to wait after a loss (default: 2)
            Prevents revenge trading

        max_trades_per_day: Maximum trades in 24 hours (default: 20)
            Prevents overtrading

        require_confirmation: Require human confirmation for large trades (default: False)
            Set True for production with real money
    """

    # === POSITION SIZING ===
    max_position_size_pct: float = 0.20          # 20% of portfolio max
    max_risk_per_trade_pct: float = 0.01         # 1% risk per trade
    min_position_size: float = 0.01              # Minimum trade size

    # === DRAWDOWN PROTECTION ===
    max_drawdown_pct: float = 0.10               # 10% max drawdown (HARD STOP)
    drawdown_warning_pct: float = 0.07           # 7% drawdown warning
    daily_loss_limit_pct: float = 0.03           # 3% max daily loss

    # === EXPOSURE LIMITS ===
    max_leverage: float = 1.0                    # No leverage
    max_open_positions: int = 1                  # Single position
    max_correlation_exposure: float = 0.50       # 50% correlated exposure

    # === TRADE VALIDATION ===
    min_risk_reward_ratio: float = 1.5           # Minimum 1.5:1 R:R
    max_spread_pct: float = 0.001                # 0.1% max spread
    required_atr_multiplier: float = 1.0         # SL >= 1 * ATR
    max_position_duration_steps: int = 40        # Max 40 bars holding

    # === MARKET CONDITION FILTERS ===
    volatility_spike_threshold: float = 2.5      # ATR spike detection
    min_volume_percentile: float = 20.0          # Min volume filter
    regime_check_enabled: bool = True            # Use regime detection
    high_volatility_reduction: float = 0.5       # Reduce size 50% in chaos

    # === BEHAVIORAL RULES ===
    cooldown_after_loss_steps: int = 2           # Wait 2 bars after loss
    max_trades_per_day: int = 20                 # Max 20 trades/day
    require_confirmation: bool = False           # Human confirmation
    allow_hold_action: bool = True               # Allow HOLD without penalty

    # === RULE ENGINE SETTINGS ===
    strict_mode: bool = True                     # Reject on ANY violation
    soft_violation_threshold: int = 2            # Max soft violations before reject
    enable_rule_explanations: bool = True        # Include reasoning in output

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        base = super().to_dict()
        base.update({
            # Position sizing
            'max_position_size_pct': self.max_position_size_pct,
            'max_risk_per_trade_pct': self.max_risk_per_trade_pct,
            'min_position_size': self.min_position_size,
            # Drawdown protection
            'max_drawdown_pct': self.max_drawdown_pct,
            'drawdown_warning_pct': self.drawdown_warning_pct,
            'daily_loss_limit_pct': self.daily_loss_limit_pct,
            # Exposure limits
            'max_leverage': self.max_leverage,
            'max_open_positions': self.max_open_positions,
            'max_correlation_exposure': self.max_correlation_exposure,
            # Trade validation
            'min_risk_reward_ratio': self.min_risk_reward_ratio,
            'max_spread_pct': self.max_spread_pct,
            'required_atr_multiplier': self.required_atr_multiplier,
            'max_position_duration_steps': self.max_position_duration_steps,
            # Market condition filters
            'volatility_spike_threshold': self.volatility_spike_threshold,
            'min_volume_percentile': self.min_volume_percentile,
            'regime_check_enabled': self.regime_check_enabled,
            'high_volatility_reduction': self.high_volatility_reduction,
            # Behavioral rules
            'cooldown_after_loss_steps': self.cooldown_after_loss_steps,
            'max_trades_per_day': self.max_trades_per_day,
            'require_confirmation': self.require_confirmation,
            'allow_hold_action': self.allow_hold_action,
            # Rule engine
            'strict_mode': self.strict_mode,
            'soft_violation_threshold': self.soft_violation_threshold,
            'enable_rule_explanations': self.enable_rule_explanations,
        })
        return base

    @classmethod
    def conservative(cls) -> 'RiskSentinelConfig':
        """
        Create a conservative configuration for cautious trading.

        Use this for:
            - Live trading with real money
            - Risk-averse clients
            - High-volatility market periods
        """
        return cls(
            max_position_size_pct=0.10,       # 10% max position
            max_risk_per_trade_pct=0.005,    # 0.5% risk per trade
            max_drawdown_pct=0.05,           # 5% max drawdown
            max_leverage=1.0,                # No leverage
            min_risk_reward_ratio=2.0,       # Require 2:1 R:R
            strict_mode=True,
            require_confirmation=True        # Human in loop
        )

    @classmethod
    def aggressive(cls) -> 'RiskSentinelConfig':
        """
        Create an aggressive configuration for growth-focused trading.

        Use this for:
            - Paper trading
            - Backtesting
            - Risk-tolerant clients with small accounts

        WARNING: Not recommended for large accounts or risk-averse clients.
        """
        return cls(
            max_position_size_pct=0.30,       # 30% max position
            max_risk_per_trade_pct=0.02,     # 2% risk per trade
            max_drawdown_pct=0.15,           # 15% max drawdown
            max_leverage=1.0,                # Still no leverage (safety)
            min_risk_reward_ratio=1.2,       # Accept 1.2:1 R:R
            strict_mode=False,               # Allow some violations
            soft_violation_threshold=3       # Allow up to 3 soft violations
        )

    @classmethod
    def backtesting(cls) -> 'RiskSentinelConfig':
        """
        Create a configuration for backtesting.

        Less strict to allow the RL agent to explore, but still
        enforces core safety limits.
        """
        return cls(
            max_position_size_pct=0.25,
            max_risk_per_trade_pct=0.01,
            max_drawdown_pct=0.10,
            strict_mode=False,
            soft_violation_threshold=5,
            require_confirmation=False,
            enable_rule_explanations=False   # Faster without explanations
        )


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================


class ConfigPreset(Enum):
    """
    Predefined configuration presets for common scenarios.

    Use these to quickly configure agents for different use cases.
    """
    CONSERVATIVE = "conservative"    # Safe, for live trading
    MODERATE = "moderate"            # Balanced, default
    AGGRESSIVE = "aggressive"        # Growth-focused, more risk
    BACKTESTING = "backtesting"      # For historical simulation


def get_risk_sentinel_config(preset: ConfigPreset = ConfigPreset.MODERATE) -> RiskSentinelConfig:
    """
    Get a RiskSentinelConfig based on preset.

    Args:
        preset: Which preset to use

    Returns:
        Configured RiskSentinelConfig instance
    """
    if preset == ConfigPreset.CONSERVATIVE:
        return RiskSentinelConfig.conservative()
    elif preset == ConfigPreset.AGGRESSIVE:
        return RiskSentinelConfig.aggressive()
    elif preset == ConfigPreset.BACKTESTING:
        return RiskSentinelConfig.backtesting()
    else:  # MODERATE (default)
        return RiskSentinelConfig()


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================


def validate_risk_config(config: RiskSentinelConfig) -> List[str]:
    """
    Validate a RiskSentinelConfig for common mistakes.

    Returns:
        List of warning/error messages (empty if valid)
    """
    issues = []

    # Position sizing checks
    if config.max_position_size_pct > 0.50:
        issues.append(
            f"WARNING: max_position_size_pct ({config.max_position_size_pct:.0%}) "
            "is very high. Consider reducing to 20-30%."
        )

    if config.max_risk_per_trade_pct > 0.02:
        issues.append(
            f"WARNING: max_risk_per_trade_pct ({config.max_risk_per_trade_pct:.1%}) "
            "exceeds professional standard of 1-2%."
        )

    # Drawdown checks
    if config.max_drawdown_pct > 0.20:
        issues.append(
            f"WARNING: max_drawdown_pct ({config.max_drawdown_pct:.0%}) "
            "is very high. Consider 10-15% for commercial use."
        )

    if config.drawdown_warning_pct >= config.max_drawdown_pct:
        issues.append(
            "ERROR: drawdown_warning_pct must be less than max_drawdown_pct"
        )

    # Leverage checks
    if config.max_leverage > 2.0:
        issues.append(
            f"WARNING: max_leverage ({config.max_leverage}x) is high. "
            "Leverage amplifies both gains AND losses."
        )

    # Risk:Reward checks
    if config.min_risk_reward_ratio < 1.0:
        issues.append(
            f"ERROR: min_risk_reward_ratio ({config.min_risk_reward_ratio}) "
            "should be >= 1.0. You're risking more than you can gain."
        )

    return issues
