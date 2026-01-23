# =============================================================================
# PORTFOLIO RISK MANAGEMENT - Institutional-Grade Risk Analytics
# =============================================================================
# This module provides sophisticated portfolio risk management capabilities
# used by major financial institutions:
#
#   1. VALUE AT RISK (VaR) - Multiple methodologies
#      - Historical VaR (non-parametric)
#      - Parametric VaR (variance-covariance)
#      - Monte Carlo VaR (simulation-based)
#
#   2. CONDITIONAL VAR (CVaR/Expected Shortfall)
#      - Measures tail risk beyond VaR
#      - Required by Basel III/IV regulations
#
#   3. CORRELATION ANALYSIS
#      - Dynamic correlation matrices
#      - Correlation breakdown detection
#      - Regime-dependent correlations
#
#   4. EXPOSURE MANAGEMENT
#      - Net exposure by currency/asset class
#      - Concentration risk metrics
#      - Gross/Net exposure ratios
#
# Architecture:
#   ┌─────────────────────────────────────────────────────────────────┐
#   │                    PortfolioRiskManager                         │
#   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
#   │  │VaRCalculator│ │CorrelEngine │ │ExposureMgr  │ │StressTester│ │
#   │  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
#   └─────────────────────────────────────────────────────────────────┘
#
# =============================================================================

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import norm, t as student_t
from typing import Dict, Any, Optional, List, Tuple, Union, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import deque
import logging
import warnings
from abc import ABC, abstractmethod

# Suppress scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# ENUMS AND TYPE DEFINITIONS
# =============================================================================

class VaRMethod(Enum):
    """Value at Risk calculation methodology."""
    HISTORICAL = "historical"           # Non-parametric, uses actual returns
    PARAMETRIC = "parametric"           # Assumes normal distribution
    CORNISH_FISHER = "cornish_fisher"   # Adjusts for skewness/kurtosis
    MONTE_CARLO = "monte_carlo"         # Simulation-based
    EWMA = "ewma"                       # Exponentially weighted


class RiskMetricType(Enum):
    """Types of risk metrics."""
    VAR = "var"
    CVAR = "cvar"
    VOLATILITY = "volatility"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    MAX_DRAWDOWN = "max_drawdown"
    BETA = "beta"
    CORRELATION = "correlation"


class ExposureType(Enum):
    """Types of exposure measurements."""
    GROSS = "gross"     # Sum of absolute values
    NET = "net"         # Sum with signs
    LONG = "long"       # Long positions only
    SHORT = "short"     # Short positions only


class AlertSeverity(Enum):
    """Severity levels for risk alerts."""
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Position:
    """
    Represents a trading position for risk calculations.

    Attributes:
        symbol: Asset identifier (e.g., "EURUSD", "XAUUSD")
        quantity: Position size (positive=long, negative=short)
        entry_price: Average entry price
        current_price: Current market price
        currency: Base currency of the position
        asset_class: Classification (forex, commodity, equity, etc.)
    """
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    currency: str = "USD"
    asset_class: str = "forex"

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.quantity * (self.current_price - self.entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0


@dataclass
class VaRResult:
    """
    Result of a Value at Risk calculation.

    Attributes:
        var_amount: VaR in currency units
        var_pct: VaR as percentage of portfolio
        confidence: Confidence level (e.g., 0.95)
        horizon_days: Time horizon in days
        method: Calculation method used
        timestamp: When calculated
    """
    var_amount: float
    var_pct: float
    confidence: float
    horizon_days: int
    method: VaRMethod
    timestamp: datetime = field(default_factory=datetime.now)

    # Additional statistics
    cvar_amount: Optional[float] = None
    cvar_pct: Optional[float] = None
    volatility: Optional[float] = None
    sample_size: int = 0
    # SECURITY FIX: Flag to indicate if result is valid
    is_valid: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'var_amount': round(self.var_amount, 2) if self.var_amount != float('inf') else 'inf',
            'var_pct': round(self.var_pct * 100, 4),
            'cvar_amount': round(self.cvar_amount, 2) if self.cvar_amount else None,
            'cvar_pct': round(self.cvar_pct * 100, 4) if self.cvar_pct else None,
            'confidence': self.confidence,
            'horizon_days': self.horizon_days,
            'method': self.method.value,
            'volatility': round(self.volatility * 100, 4) if self.volatility else None,
            'sample_size': self.sample_size,
            'timestamp': self.timestamp.isoformat(),
            # SECURITY FIX: Include validity info
            'is_valid': self.is_valid,
            'error_message': self.error_message
        }


@dataclass
class CorrelationAlert:
    """Alert when correlation structure changes significantly."""
    asset_pair: Tuple[str, str]
    old_correlation: float
    new_correlation: float
    change_magnitude: float
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExposureReport:
    """
    Comprehensive exposure report for the portfolio.
    """
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float

    # By currency
    exposure_by_currency: Dict[str, float] = field(default_factory=dict)

    # By asset class
    exposure_by_asset_class: Dict[str, float] = field(default_factory=dict)

    # Concentration metrics
    largest_position_pct: float = 0.0
    hhi_concentration: float = 0.0  # Herfindahl-Hirschman Index

    # Limits
    gross_limit: float = 0.0
    net_limit: float = 0.0
    utilization_pct: float = 0.0

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'gross_exposure': round(self.gross_exposure, 2),
            'net_exposure': round(self.net_exposure, 2),
            'long_exposure': round(self.long_exposure, 2),
            'short_exposure': round(self.short_exposure, 2),
            'exposure_by_currency': {
                k: round(v, 2) for k, v in self.exposure_by_currency.items()
            },
            'exposure_by_asset_class': {
                k: round(v, 2) for k, v in self.exposure_by_asset_class.items()
            },
            'largest_position_pct': round(self.largest_position_pct * 100, 2),
            'hhi_concentration': round(self.hhi_concentration, 4),
            'utilization_pct': round(self.utilization_pct * 100, 2),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RiskLimits:
    """
    Configurable risk limits for the portfolio.

    All limits are expressed as decimals (0.05 = 5%).
    """
    # VaR limits
    max_var_pct: float = 0.02               # 2% daily VaR limit
    max_cvar_pct: float = 0.03              # 3% daily CVaR limit

    # Exposure limits
    max_gross_exposure: float = 2.0          # 200% gross exposure
    max_net_exposure: float = 1.0            # 100% net exposure
    max_single_position_pct: float = 0.20    # 20% in single position
    max_currency_exposure: float = 0.50      # 50% in single currency
    max_asset_class_exposure: float = 0.60   # 60% in single asset class

    # Correlation limits
    max_correlated_exposure: float = 0.30    # 30% in highly correlated assets
    correlation_threshold: float = 0.70      # Consider >70% as "highly correlated"

    # Drawdown limits
    max_daily_loss: float = 0.03            # 3% daily loss limit
    max_weekly_loss: float = 0.05           # 5% weekly loss limit
    max_drawdown: float = 0.10              # 10% max drawdown

    # Concentration limits (HHI)
    max_hhi: float = 0.25                   # Max concentration index

    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_var_pct': f"{self.max_var_pct:.1%}",
            'max_cvar_pct': f"{self.max_cvar_pct:.1%}",
            'max_gross_exposure': f"{self.max_gross_exposure:.0%}",
            'max_net_exposure': f"{self.max_net_exposure:.0%}",
            'max_single_position_pct': f"{self.max_single_position_pct:.0%}",
            'max_currency_exposure': f"{self.max_currency_exposure:.0%}",
            'max_drawdown': f"{self.max_drawdown:.0%}",
        }


# =============================================================================
# VAR CALCULATOR - Multiple Methodologies
# =============================================================================

class VaRCalculator:
    """
    Institutional-grade Value at Risk calculator.

    Supports multiple methodologies:
    - Historical: Uses actual return distribution
    - Parametric: Assumes normal distribution
    - Cornish-Fisher: Adjusts for fat tails
    - Monte Carlo: Simulation-based
    - EWMA: Exponentially weighted for recent data

    Example:
        calculator = VaRCalculator()
        returns = np.array([...])  # Daily returns

        # Calculate 95% 1-day VaR using different methods
        var_hist = calculator.calculate_historical(returns, confidence=0.95)
        var_para = calculator.calculate_parametric(returns, confidence=0.95)
        var_mc = calculator.calculate_monte_carlo(returns, confidence=0.95)
    """

    def __init__(
        self,
        default_confidence: float = 0.95,
        default_horizon: int = 1,
        min_observations: int = 30,
        ewma_lambda: float = 0.94
    ):
        """
        Initialize VaR calculator.

        Args:
            default_confidence: Default confidence level (0.95 = 95%)
            default_horizon: Default time horizon in days
            min_observations: Minimum data points required
            ewma_lambda: Decay factor for EWMA (RiskMetrics uses 0.94)
        """
        self.default_confidence = default_confidence
        self.default_horizon = default_horizon
        self.min_observations = min_observations
        self.ewma_lambda = ewma_lambda
        self._logger = logging.getLogger("portfolio_risk.var")

    def calculate(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        method: VaRMethod = VaRMethod.HISTORICAL,
        confidence: Optional[float] = None,
        horizon: Optional[int] = None
    ) -> VaRResult:
        """
        Calculate VaR using specified method.

        Args:
            returns: Array of historical returns (as decimals)
            portfolio_value: Current portfolio value
            method: Calculation methodology
            confidence: Confidence level (default: 0.95)
            horizon: Time horizon in days (default: 1)

        Returns:
            VaRResult with VaR and CVaR metrics
        """
        confidence = confidence or self.default_confidence
        horizon = horizon or self.default_horizon

        # Validate inputs
        if len(returns) < self.min_observations:
            self._logger.warning(
                f"Insufficient data: {len(returns)} < {self.min_observations}. "
                "VaR estimate may be unreliable."
            )

        # Remove NaN values
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            # SECURITY FIX: Return invalid result instead of zero VaR
            # Zero VaR is dangerous as it implies no risk, which is incorrect
            self._logger.error("VaR calculation failed: No valid return data after NaN removal")
            return VaRResult(
                var_amount=float('inf'),  # Conservative: assume infinite risk
                var_pct=1.0,  # 100% risk as conservative estimate
                confidence=confidence,
                horizon_days=horizon,
                method=method,
                sample_size=0,
                is_valid=False,
                error_message="Insufficient data: no valid returns after NaN removal"
            )

        # Calculate VaR based on method
        method_map = {
            VaRMethod.HISTORICAL: self._historical_var,
            VaRMethod.PARAMETRIC: self._parametric_var,
            VaRMethod.CORNISH_FISHER: self._cornish_fisher_var,
            VaRMethod.MONTE_CARLO: self._monte_carlo_var,
            VaRMethod.EWMA: self._ewma_var
        }

        calculator = method_map.get(method, self._historical_var)
        var_pct, cvar_pct, volatility = calculator(returns, confidence, horizon)

        # Convert to amounts
        var_amount = portfolio_value * abs(var_pct)
        cvar_amount = portfolio_value * abs(cvar_pct) if cvar_pct else None

        return VaRResult(
            var_amount=var_amount,
            var_pct=abs(var_pct),
            confidence=confidence,
            horizon_days=horizon,
            method=method,
            cvar_amount=cvar_amount,
            cvar_pct=abs(cvar_pct) if cvar_pct else None,
            volatility=volatility,
            sample_size=len(returns)
        )

    def _historical_var(
        self,
        returns: np.ndarray,
        confidence: float,
        horizon: int
    ) -> Tuple[float, float, float]:
        """
        Historical (non-parametric) VaR.

        Simply takes the percentile of actual returns.
        No distribution assumptions.
        """
        # Scale returns for horizon (square root of time rule)
        scaled_returns = returns * np.sqrt(horizon)

        # VaR is the percentile of losses
        alpha = 1 - confidence
        var = np.percentile(scaled_returns, alpha * 100)

        # CVaR is the mean of returns beyond VaR
        cvar = np.mean(scaled_returns[scaled_returns <= var])

        # Volatility
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        return var, cvar, volatility

    def _parametric_var(
        self,
        returns: np.ndarray,
        confidence: float,
        horizon: int
    ) -> Tuple[float, float, float]:
        """
        Parametric (variance-covariance) VaR.

        Assumes returns are normally distributed.
        VaR = -μ + σ * z_α
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        # Scale for horizon
        mu_h = mu * horizon
        sigma_h = sigma * np.sqrt(horizon)

        # Z-score for confidence level
        z = norm.ppf(1 - confidence)

        # VaR = mean - z * std (negative z for losses)
        var = mu_h + z * sigma_h

        # CVaR for normal distribution
        # E[X | X < VaR] = μ - σ * φ(z) / (1-Φ(z))
        cvar = mu_h - sigma_h * norm.pdf(z) / (1 - confidence)

        volatility = sigma * np.sqrt(252)

        return var, cvar, volatility

    def _cornish_fisher_var(
        self,
        returns: np.ndarray,
        confidence: float,
        horizon: int
    ) -> Tuple[float, float, float]:
        """
        Cornish-Fisher VaR with skewness/kurtosis adjustment.

        Adjusts the normal quantile for non-normal distributions
        using the first four moments.
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        # Calculate skewness and excess kurtosis
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis

        # Standard normal quantile
        z = norm.ppf(1 - confidence)

        # Cornish-Fisher expansion
        z_cf = (
            z +
            (z**2 - 1) * skew / 6 +
            (z**3 - 3*z) * kurt / 24 -
            (2*z**3 - 5*z) * skew**2 / 36
        )

        # Scale for horizon
        mu_h = mu * horizon
        sigma_h = sigma * np.sqrt(horizon)

        var = mu_h + z_cf * sigma_h

        # Approximate CVaR
        cvar = var * 1.2  # Simple approximation for CF

        volatility = sigma * np.sqrt(252)

        return var, cvar, volatility

    def _monte_carlo_var(
        self,
        returns: np.ndarray,
        confidence: float,
        horizon: int,
        n_simulations: int = 10000
    ) -> Tuple[float, float, float]:
        """
        Monte Carlo VaR using simulation.

        Simulates future returns based on historical distribution
        characteristics with optional fat-tail modeling.
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        # Fit Student-t distribution for fat tails
        # Estimate degrees of freedom
        try:
            df, loc, scale = student_t.fit(returns)
            df = max(3, min(df, 30))  # Constrain df
        except Exception:
            df = 5  # Default to moderately fat tails

        # Simulate returns using Student-t
        simulated = student_t.rvs(
            df=df,
            loc=mu * horizon,
            scale=sigma * np.sqrt(horizon),
            size=n_simulations
        )

        # Calculate VaR and CVaR from simulations
        alpha = 1 - confidence
        var = np.percentile(simulated, alpha * 100)
        cvar = np.mean(simulated[simulated <= var])

        volatility = sigma * np.sqrt(252)

        return var, cvar, volatility

    def _ewma_var(
        self,
        returns: np.ndarray,
        confidence: float,
        horizon: int
    ) -> Tuple[float, float, float]:
        """
        EWMA (Exponentially Weighted Moving Average) VaR.

        Gives more weight to recent observations.
        Uses RiskMetrics methodology (lambda=0.94 for daily).
        """
        n = len(returns)

        # Calculate EWMA variance
        weights = np.array([
            (1 - self.ewma_lambda) * (self.ewma_lambda ** i)
            for i in range(n)
        ])[::-1]  # Reverse so recent data gets more weight

        weights = weights / weights.sum()  # Normalize

        # EWMA mean and variance
        ewma_mean = np.sum(weights * returns)
        ewma_var = np.sum(weights * (returns - ewma_mean) ** 2)
        ewma_std = np.sqrt(ewma_var)

        # Scale for horizon
        mu_h = ewma_mean * horizon
        sigma_h = ewma_std * np.sqrt(horizon)

        # VaR using normal quantile
        z = norm.ppf(1 - confidence)
        var = mu_h + z * sigma_h

        # CVaR
        cvar = mu_h - sigma_h * norm.pdf(z) / (1 - confidence)

        volatility = ewma_std * np.sqrt(252)

        return var, cvar, volatility

    def calculate_component_var(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        portfolio_value: float,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate component VaR (contribution of each asset to total VaR).

        Args:
            returns_matrix: Matrix of returns (n_observations x n_assets)
            weights: Portfolio weights for each asset
            portfolio_value: Total portfolio value
            confidence: Confidence level

        Returns:
            Dict mapping asset index to its VaR contribution
        """
        # Portfolio returns
        portfolio_returns = returns_matrix @ weights

        # Covariance matrix
        cov_matrix = np.cov(returns_matrix.T)

        # Portfolio variance
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_var)

        # Marginal VaR for each asset
        marginal_var = (cov_matrix @ weights) / portfolio_std

        # Component VaR
        z = norm.ppf(1 - confidence)
        component_var = {}

        for i, (w, mv) in enumerate(zip(weights, marginal_var)):
            component_var[f"asset_{i}"] = abs(w * mv * z * portfolio_value)

        return component_var

    def calculate_incremental_var(
        self,
        current_returns: np.ndarray,
        new_position_returns: np.ndarray,
        current_var: float,
        portfolio_value: float,
        new_position_value: float,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate incremental VaR for adding a new position.

        Args:
            current_returns: Current portfolio returns
            new_position_returns: Returns of new position
            current_var: Current portfolio VaR
            portfolio_value: Current portfolio value
            new_position_value: Value of new position
            confidence: Confidence level

        Returns:
            Incremental VaR (how much VaR increases)
        """
        # Combine returns
        new_weight = new_position_value / (portfolio_value + new_position_value)
        old_weight = 1 - new_weight

        combined_returns = (
            old_weight * current_returns +
            new_weight * new_position_returns
        )

        # Calculate new VaR
        new_result = self.calculate(
            combined_returns,
            portfolio_value + new_position_value,
            VaRMethod.HISTORICAL,
            confidence
        )

        return new_result.var_amount - current_var


# =============================================================================
# CORRELATION ENGINE
# =============================================================================

class CorrelationEngine:
    """
    Dynamic correlation analysis for portfolio risk management.

    Features:
    - Rolling correlation matrices
    - Correlation breakdown detection
    - DCC-GARCH style dynamic correlations
    - Regime-dependent correlation analysis

    Example:
        engine = CorrelationEngine()

        # Update with new returns
        engine.update({'EURUSD': 0.01, 'GBPUSD': 0.008, 'XAUUSD': -0.005})

        # Get current correlation matrix
        corr_matrix = engine.get_correlation_matrix()

        # Check for correlation breakdown
        alerts = engine.detect_correlation_breakdown()
    """

    def __init__(
        self,
        lookback_short: int = 20,
        lookback_long: int = 60,
        breakdown_threshold: float = 0.3,
        min_history: int = 30
    ):
        """
        Initialize correlation engine.

        Args:
            lookback_short: Short-term window for correlation
            lookback_long: Long-term window for baseline
            breakdown_threshold: Threshold for breakdown detection
            min_history: Minimum data points before calculation
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.breakdown_threshold = breakdown_threshold
        self.min_history = min_history

        # Data storage
        self._returns_history: Dict[str, deque] = {}
        self._correlation_history: deque = deque(maxlen=100)

        # Current state
        self._current_correlation: Optional[np.ndarray] = None
        self._baseline_correlation: Optional[np.ndarray] = None
        self._assets: List[str] = []

        self._logger = logging.getLogger("portfolio_risk.correlation")

    def update(self, returns: Dict[str, float]) -> None:
        """
        Update correlation engine with new returns.

        Args:
            returns: Dict mapping asset symbol to return value
        """
        # Initialize storage for new assets
        for asset in returns:
            if asset not in self._returns_history:
                self._returns_history[asset] = deque(maxlen=self.lookback_long * 2)
                if asset not in self._assets:
                    self._assets.append(asset)

        # Store returns
        for asset, ret in returns.items():
            self._returns_history[asset].append(ret)

        # Recalculate correlations if enough data
        if self._has_sufficient_data():
            self._recalculate_correlations()

    def _has_sufficient_data(self) -> bool:
        """Check if we have enough data for correlation calculation."""
        if not self._returns_history:
            return False
        min_len = min(len(h) for h in self._returns_history.values())
        return min_len >= self.min_history

    def _recalculate_correlations(self) -> None:
        """Recalculate correlation matrices."""
        # Build returns matrix
        n_assets = len(self._assets)
        if n_assets < 2:
            return

        # Get aligned returns
        min_len = min(len(self._returns_history[a]) for a in self._assets)

        # Short-term correlation
        short_len = min(self.lookback_short, min_len)
        returns_short = np.array([
            list(self._returns_history[a])[-short_len:]
            for a in self._assets
        ])

        if returns_short.shape[1] >= 2:
            self._current_correlation = np.corrcoef(returns_short)

        # Long-term correlation (baseline)
        long_len = min(self.lookback_long, min_len)
        returns_long = np.array([
            list(self._returns_history[a])[-long_len:]
            for a in self._assets
        ])

        if returns_long.shape[1] >= self.min_history:
            self._baseline_correlation = np.corrcoef(returns_long)

        # Store in history
        if self._current_correlation is not None:
            self._correlation_history.append({
                'timestamp': datetime.now(),
                'correlation': self._current_correlation.copy()
            })

    def get_correlation_matrix(
        self,
        window: str = "short"
    ) -> Optional[np.ndarray]:
        """
        Get correlation matrix.

        Args:
            window: "short" for recent, "long" for baseline

        Returns:
            Correlation matrix as numpy array
        """
        if window == "short":
            return self._current_correlation
        return self._baseline_correlation

    def get_correlation(self, asset1: str, asset2: str) -> Optional[float]:
        """
        Get correlation between two assets.

        Args:
            asset1: First asset symbol
            asset2: Second asset symbol

        Returns:
            Correlation coefficient or None if not available
        """
        if self._current_correlation is None:
            return None

        try:
            i = self._assets.index(asset1)
            j = self._assets.index(asset2)
            return float(self._current_correlation[i, j])
        except (ValueError, IndexError):
            return None

    def get_correlation_dataframe_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Get correlation matrix as nested dictionary.

        Returns:
            Dict of dicts suitable for DataFrame conversion
        """
        if self._current_correlation is None:
            return {}

        result = {}
        for i, asset_i in enumerate(self._assets):
            result[asset_i] = {}
            for j, asset_j in enumerate(self._assets):
                result[asset_i][asset_j] = float(self._current_correlation[i, j])

        return result

    def detect_correlation_breakdown(self) -> List[CorrelationAlert]:
        """
        Detect significant changes in correlation structure.

        Returns:
            List of CorrelationAlert objects for significant changes
        """
        alerts = []

        if self._current_correlation is None or self._baseline_correlation is None:
            return alerts

        n_assets = len(self._assets)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                old_corr = self._baseline_correlation[i, j]
                new_corr = self._current_correlation[i, j]
                change = abs(new_corr - old_corr)

                if change >= self.breakdown_threshold:
                    severity = (
                        AlertSeverity.CRITICAL if change >= 0.5
                        else AlertSeverity.WARNING if change >= 0.3
                        else AlertSeverity.INFO
                    )

                    alerts.append(CorrelationAlert(
                        asset_pair=(self._assets[i], self._assets[j]),
                        old_correlation=old_corr,
                        new_correlation=new_corr,
                        change_magnitude=change,
                        severity=severity,
                        message=(
                            f"Correlation between {self._assets[i]} and {self._assets[j]} "
                            f"changed from {old_corr:.2f} to {new_corr:.2f} "
                            f"(change: {change:+.2f})"
                        )
                    ))

        return alerts

    def get_highly_correlated_pairs(
        self,
        threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        Get pairs of assets with high correlation.

        Args:
            threshold: Minimum correlation to include

        Returns:
            List of (asset1, asset2, correlation) tuples
        """
        pairs = []

        if self._current_correlation is None:
            return pairs

        n_assets = len(self._assets)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr = abs(self._current_correlation[i, j])
                if corr >= threshold:
                    pairs.append((self._assets[i], self._assets[j], corr))

        # Sort by correlation descending
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def calculate_correlation_adjusted_weights(
        self,
        positions: List[Position],
        max_correlation: float = 0.7
    ) -> Dict[str, float]:
        """
        Calculate correlation-adjusted position weights.

        Reduces weights for highly correlated positions.

        Args:
            positions: List of positions
            max_correlation: Maximum allowed correlation

        Returns:
            Dict mapping symbol to adjusted weight multiplier
        """
        weights = {p.symbol: 1.0 for p in positions}

        if self._current_correlation is None:
            return weights

        # Find highly correlated pairs
        highly_correlated = self.get_highly_correlated_pairs(max_correlation)

        # Reduce weights for correlated positions
        for asset1, asset2, corr in highly_correlated:
            if asset1 in weights and asset2 in weights:
                # Penalize both positions proportionally to correlation
                penalty = 1.0 - (corr - max_correlation) / (1.0 - max_correlation)
                penalty = max(0.3, penalty)  # Floor at 30%

                weights[asset1] = min(weights[asset1], penalty)
                weights[asset2] = min(weights[asset2], penalty)

        return weights


# =============================================================================
# EXPOSURE MANAGER
# =============================================================================

class ExposureManager:
    """
    Portfolio exposure tracking and management.

    Tracks:
    - Gross/Net exposure
    - Exposure by currency
    - Exposure by asset class
    - Concentration metrics

    Example:
        manager = ExposureManager(equity=100000)
        manager.update_positions([
            Position("EURUSD", 50000, 1.10, 1.11, "EUR", "forex"),
            Position("XAUUSD", -20000, 1950, 1960, "USD", "commodity")
        ])

        report = manager.get_exposure_report()
        print(f"Net exposure: {report.net_exposure}")
    """

    def __init__(
        self,
        equity: float,
        limits: Optional[RiskLimits] = None
    ):
        """
        Initialize exposure manager.

        Args:
            equity: Account equity (must be > 0)
            limits: Risk limits configuration

        Raises:
            ValueError: If equity is zero or negative
        """
        if equity <= 0:
            raise ValueError(
                f"Equity must be positive, got {equity}. "
                "Cannot initialize risk manager with zero/negative equity."
            )
        self._equity = equity
        self.limits = limits or RiskLimits()
        self._positions: List[Position] = []
        self._logger = logging.getLogger("portfolio_risk.exposure")

    @property
    def equity(self) -> float:
        """Get current equity (protected against zero)."""
        return max(self._equity, 1e-10)  # Prevent division by zero

    @equity.setter
    def equity(self, value: float) -> None:
        """Set equity with validation."""
        if value <= 0:
            self._logger.error(f"Attempted to set zero/negative equity: {value}")
            raise ValueError(
                f"Equity must be positive, got {value}. "
                "Zero/negative equity indicates critical system failure."
            )
        self._equity = value

    def update_equity(self, equity: float) -> None:
        """Update account equity (must be > 0)."""
        self.equity = equity  # Uses setter validation

    def update_positions(self, positions: List[Position]) -> None:
        """Update position list."""
        self._positions = positions

    def add_position(self, position: Position) -> None:
        """Add a new position."""
        self._positions.append(position)

    def remove_position(self, symbol: str) -> bool:
        """Remove a position by symbol."""
        for i, pos in enumerate(self._positions):
            if pos.symbol == symbol:
                self._positions.pop(i)
                return True
        return False

    def get_exposure_report(self) -> ExposureReport:
        """
        Generate comprehensive exposure report.

        Returns:
            ExposureReport with all exposure metrics
        """
        if not self._positions or self.equity == 0:
            return ExposureReport(
                gross_exposure=0,
                net_exposure=0,
                long_exposure=0,
                short_exposure=0
            )

        # Calculate exposures
        long_exposure = sum(
            pos.market_value for pos in self._positions
            if pos.is_long
        )
        short_exposure = abs(sum(
            pos.market_value for pos in self._positions
            if pos.is_short
        ))

        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        # By currency
        exposure_by_currency: Dict[str, float] = {}
        for pos in self._positions:
            curr = pos.currency
            exposure_by_currency[curr] = (
                exposure_by_currency.get(curr, 0) + pos.market_value
            )

        # By asset class
        exposure_by_asset_class: Dict[str, float] = {}
        for pos in self._positions:
            ac = pos.asset_class
            exposure_by_asset_class[ac] = (
                exposure_by_asset_class.get(ac, 0) + abs(pos.market_value)
            )

        # Concentration metrics
        position_values = [abs(p.market_value) for p in self._positions]
        total_value = sum(position_values) if position_values else 1

        largest_position_pct = max(position_values) / self.equity if position_values else 0

        # Herfindahl-Hirschman Index
        weights = [v / total_value for v in position_values] if total_value > 0 else []
        hhi = sum(w**2 for w in weights)

        # Utilization
        utilization = gross_exposure / (self.equity * self.limits.max_gross_exposure)

        return ExposureReport(
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            exposure_by_currency=exposure_by_currency,
            exposure_by_asset_class=exposure_by_asset_class,
            largest_position_pct=largest_position_pct,
            hhi_concentration=hhi,
            gross_limit=self.equity * self.limits.max_gross_exposure,
            net_limit=self.equity * self.limits.max_net_exposure,
            utilization_pct=min(1.0, utilization)
        )

    def check_new_position(
        self,
        new_position: Position,
        correlation_engine: Optional[CorrelationEngine] = None
    ) -> Tuple[bool, List[str], float]:
        """
        Check if a new position would violate exposure limits.

        Args:
            new_position: Position to evaluate
            correlation_engine: Optional correlation engine for correlation checks

        Returns:
            Tuple of (is_allowed, list of violations, recommended size multiplier)
        """
        violations = []
        multiplier = 1.0

        # Current exposure
        report = self.get_exposure_report()

        # Simulate new exposure
        new_value = abs(new_position.market_value)
        new_gross = report.gross_exposure + new_value

        if new_position.is_long:
            new_net = report.net_exposure + new_value
        else:
            new_net = report.net_exposure - new_value

        # Check gross exposure
        if new_gross > self.equity * self.limits.max_gross_exposure:
            violations.append(
                f"Gross exposure would exceed limit: "
                f"{new_gross/self.equity:.1%} > {self.limits.max_gross_exposure:.0%}"
            )
            # Calculate allowed size
            allowed_gross = self.equity * self.limits.max_gross_exposure - report.gross_exposure
            if new_value > 0:
                multiplier = min(multiplier, max(0, allowed_gross / new_value))

        # Check net exposure
        if abs(new_net) > self.equity * self.limits.max_net_exposure:
            violations.append(
                f"Net exposure would exceed limit: "
                f"{abs(new_net)/self.equity:.1%} > {self.limits.max_net_exposure:.0%}"
            )

        # Check single position concentration
        if new_value > self.equity * self.limits.max_single_position_pct:
            violations.append(
                f"Position size exceeds limit: "
                f"{new_value/self.equity:.1%} > {self.limits.max_single_position_pct:.0%}"
            )
            allowed_single = self.equity * self.limits.max_single_position_pct
            multiplier = min(multiplier, allowed_single / new_value)

        # Check currency concentration
        currency_exposure = report.exposure_by_currency.get(new_position.currency, 0)
        new_currency_exposure = currency_exposure + new_value

        if new_currency_exposure > self.equity * self.limits.max_currency_exposure:
            violations.append(
                f"Currency exposure would exceed limit: "
                f"{new_currency_exposure/self.equity:.1%} > {self.limits.max_currency_exposure:.0%}"
            )
            allowed_currency = self.equity * self.limits.max_currency_exposure - currency_exposure
            if new_value > 0:
                multiplier = min(multiplier, max(0, allowed_currency / new_value))

        # Check correlation-based exposure
        if correlation_engine is not None:
            corr_weights = correlation_engine.calculate_correlation_adjusted_weights(
                self._positions + [new_position],
                self.limits.correlation_threshold
            )
            corr_multiplier = corr_weights.get(new_position.symbol, 1.0)

            if corr_multiplier < 1.0:
                violations.append(
                    f"Position correlated with existing positions: "
                    f"recommended reduction to {corr_multiplier:.0%}"
                )
                multiplier = min(multiplier, corr_multiplier)

        is_allowed = len(violations) == 0 or multiplier > 0.1

        return is_allowed, violations, max(0.1, multiplier)

    def get_net_currency_exposure(self) -> Dict[str, float]:
        """Get net exposure by currency as percentage of equity."""
        report = self.get_exposure_report()
        return {
            curr: val / self.equity
            for curr, val in report.exposure_by_currency.items()
        }


# =============================================================================
# STRESS TESTER
# =============================================================================

class StressTester:
    """
    Portfolio stress testing using historical and hypothetical scenarios.

    Scenarios include:
    - Historical: 2008 crisis, 2015 CHF flash crash, 2020 COVID
    - Hypothetical: Rate shock, equity crash, volatility spike

    Example:
        tester = StressTester()
        results = tester.run_all_scenarios(portfolio_value, positions)
    """

    # Predefined stress scenarios (asset class -> shock in percentage)
    HISTORICAL_SCENARIOS = {
        "2008_financial_crisis": {
            "forex": -0.15,      # Major moves in EUR, GBP
            "commodity": -0.25,  # Gold initially fell
            "equity": -0.50,    # S&P dropped 50%
            "description": "2008 Global Financial Crisis"
        },
        "2015_chf_flash_crash": {
            "forex": -0.20,      # CHF pairs moved 20%+ instantly
            "commodity": 0.05,   # Gold rallied
            "equity": -0.03,
            "description": "2015 SNB CHF De-peg"
        },
        "2020_covid_crash": {
            "forex": -0.08,
            "commodity": -0.30,  # Oil went negative
            "equity": -0.35,    # Fastest crash in history
            "description": "2020 COVID-19 Market Crash"
        },
        "2022_rate_shock": {
            "forex": -0.12,      # USD strength
            "commodity": -0.20,
            "equity": -0.25,
            "description": "2022 Fed Rate Hike Cycle"
        }
    }

    HYPOTHETICAL_SCENARIOS = {
        "extreme_vol_spike": {
            "forex": -0.10,
            "commodity": -0.15,
            "equity": -0.20,
            "description": "VIX spikes to 80+"
        },
        "flash_crash": {
            "forex": -0.05,
            "commodity": -0.10,
            "equity": -0.10,
            "description": "10-minute flash crash"
        },
        "liquidity_crisis": {
            "forex": -0.08,
            "commodity": -0.12,
            "equity": -0.15,
            "description": "Major liquidity withdrawal"
        }
    }

    def __init__(self):
        self._logger = logging.getLogger("portfolio_risk.stress")

    def run_scenario(
        self,
        scenario_name: str,
        positions: List[Position],
        portfolio_value: float,
        custom_shocks: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run a single stress scenario.

        Args:
            scenario_name: Name of scenario to run
            positions: Current positions
            portfolio_value: Portfolio value
            custom_shocks: Override default shocks

        Returns:
            Dict with scenario results
        """
        # Get scenario shocks
        if scenario_name in self.HISTORICAL_SCENARIOS:
            scenario = self.HISTORICAL_SCENARIOS[scenario_name]
        elif scenario_name in self.HYPOTHETICAL_SCENARIOS:
            scenario = self.HYPOTHETICAL_SCENARIOS[scenario_name]
        else:
            scenario = {
                "forex": -0.10,
                "commodity": -0.10,
                "equity": -0.10,
                "description": "Custom scenario"
            }

        if custom_shocks:
            scenario = {**scenario, **custom_shocks}

        # Calculate stressed P&L
        total_loss = 0.0
        position_losses = {}

        for pos in positions:
            shock = scenario.get(pos.asset_class, -0.10)
            loss = pos.market_value * shock
            total_loss += loss
            position_losses[pos.symbol] = loss

        # Calculate stressed portfolio value
        stressed_value = portfolio_value + total_loss
        loss_pct = total_loss / portfolio_value if portfolio_value > 0 else 0

        return {
            "scenario_name": scenario_name,
            "description": scenario.get("description", ""),
            "portfolio_value_before": portfolio_value,
            "portfolio_value_after": stressed_value,
            "total_loss": total_loss,
            "loss_pct": loss_pct,
            "position_losses": position_losses,
            "survives": stressed_value > 0
        }

    def run_all_scenarios(
        self,
        positions: List[Position],
        portfolio_value: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all predefined stress scenarios.

        Returns:
            Dict mapping scenario name to results
        """
        results = {}

        # Historical scenarios
        for name in self.HISTORICAL_SCENARIOS:
            results[name] = self.run_scenario(name, positions, portfolio_value)

        # Hypothetical scenarios
        for name in self.HYPOTHETICAL_SCENARIOS:
            results[name] = self.run_scenario(name, positions, portfolio_value)

        return results

    def get_worst_case(
        self,
        positions: List[Position],
        portfolio_value: float
    ) -> Dict[str, Any]:
        """
        Find the worst-case scenario for current portfolio.

        Returns:
            Results from worst-case scenario
        """
        all_results = self.run_all_scenarios(positions, portfolio_value)

        worst = min(
            all_results.values(),
            key=lambda x: x["portfolio_value_after"]
        )

        return worst


# =============================================================================
# PORTFOLIO RISK MANAGER - Main Interface
# =============================================================================

class PortfolioRiskManager:
    """
    Main interface for portfolio risk management.

    Combines all risk components:
    - VaR calculation
    - Correlation analysis
    - Exposure management
    - Stress testing

    Example:
        manager = PortfolioRiskManager(
            equity=100000,
            limits=RiskLimits(max_var_pct=0.02)
        )

        # Update with positions
        manager.update_positions([...])

        # Get comprehensive risk report
        risk_report = manager.get_risk_report()

        # Check if new trade is allowed
        allowed, violations, multiplier = manager.check_trade(new_position)
    """

    def __init__(
        self,
        equity: float,
        limits: Optional[RiskLimits] = None,
        var_confidence: float = 0.95,
        var_horizon: int = 1
    ):
        """
        Initialize portfolio risk manager.

        Args:
            equity: Initial account equity (must be > 0)
            limits: Risk limits configuration
            var_confidence: VaR confidence level
            var_horizon: VaR time horizon in days

        Raises:
            ValueError: If equity is zero or negative
        """
        if equity <= 0:
            raise ValueError(
                f"Equity must be positive, got {equity}. "
                "Cannot initialize risk manager with zero/negative equity."
            )
        self._equity = equity
        self.limits = limits or RiskLimits()

        # Initialize components
        self.var_calculator = VaRCalculator(
            default_confidence=var_confidence,
            default_horizon=var_horizon
        )
        self.correlation_engine = CorrelationEngine()
        self.exposure_manager = ExposureManager(equity, self.limits)
        self.stress_tester = StressTester()

        # State
        self._positions: List[Position] = []
        self._returns_history: deque = deque(maxlen=500)
        self._peak_equity = equity
        self._current_drawdown = 0.0
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0

        # Latest calculations
        self._latest_var: Optional[VaRResult] = None
        self._latest_exposure: Optional[ExposureReport] = None

        self._logger = logging.getLogger("portfolio_risk.manager")

    @property
    def equity(self) -> float:
        """Get current equity (protected against zero)."""
        return max(self._equity, 1e-10)  # Prevent division by zero

    @equity.setter
    def equity(self, value: float) -> None:
        """Set equity with validation."""
        if value <= 0:
            self._logger.error(f"Attempted to set zero/negative equity: {value}")
            raise ValueError(
                f"Equity must be positive, got {value}. "
                "Zero/negative equity indicates critical system failure."
            )
        self._equity = value

    def update_equity(self, equity: float) -> None:
        """
        Update account equity and track drawdown.

        Args:
            equity: New equity value (must be > 0)

        Raises:
            ValueError: If equity is zero or negative
        """
        # Track PnL (use getter to avoid issues if _equity not set)
        old_equity = self._equity if hasattr(self, '_equity') else equity
        pnl = equity - old_equity
        self.equity = equity  # Uses setter with validation
        self.exposure_manager.update_equity(equity)

        # Update peak and drawdown
        if equity > self._peak_equity:
            self._peak_equity = equity

        self._current_drawdown = (
            (self._peak_equity - equity) / self._peak_equity
            if self._peak_equity > 0 else 0
        )

        # Track daily/weekly PnL
        self._daily_pnl += pnl

    def update_positions(self, positions: List[Position]) -> None:
        """
        Update current positions.

        Args:
            positions: List of current positions
        """
        self._positions = positions
        self.exposure_manager.update_positions(positions)

    def record_return(self, portfolio_return: float) -> None:
        """
        Record a portfolio return for VaR calculation.

        Args:
            portfolio_return: Period return as decimal (0.01 = 1%)
        """
        self._returns_history.append(portfolio_return)

        # Update correlation engine with per-asset returns if available
        # (This would need asset-level returns in production)

    def calculate_var(
        self,
        method: VaRMethod = VaRMethod.HISTORICAL
    ) -> VaRResult:
        """
        Calculate current portfolio VaR.

        Args:
            method: VaR calculation method

        Returns:
            VaRResult with VaR and CVaR
        """
        returns = np.array(list(self._returns_history))

        self._latest_var = self.var_calculator.calculate(
            returns=returns,
            portfolio_value=self.equity,
            method=method
        )

        return self._latest_var

    def check_trade(
        self,
        proposed_position: Position
    ) -> Tuple[bool, List[str], float]:
        """
        Check if a proposed trade passes all risk checks.

        Args:
            proposed_position: Position to evaluate

        Returns:
            Tuple of (is_allowed, list of violations, size multiplier)
        """
        violations = []
        multiplier = 1.0

        # 1. Check exposure limits
        exp_allowed, exp_violations, exp_mult = self.exposure_manager.check_new_position(
            proposed_position,
            self.correlation_engine
        )
        violations.extend(exp_violations)
        multiplier = min(multiplier, exp_mult)

        # 2. Check VaR limit
        if self._latest_var and self._latest_var.var_pct > self.limits.max_var_pct:
            violations.append(
                f"Portfolio VaR ({self._latest_var.var_pct:.2%}) exceeds limit "
                f"({self.limits.max_var_pct:.1%}). No new positions allowed."
            )
            multiplier = 0.0

        # 3. Check drawdown limits
        if self._current_drawdown >= self.limits.max_drawdown:
            violations.append(
                f"Max drawdown reached: {self._current_drawdown:.1%}. "
                f"Trading halted."
            )
            multiplier = 0.0
        elif self._current_drawdown >= self.limits.max_drawdown * 0.7:
            # Warning zone - reduce position size
            dd_mult = 1.0 - (self._current_drawdown / self.limits.max_drawdown)
            multiplier = min(multiplier, dd_mult)
            violations.append(
                f"Drawdown warning: {self._current_drawdown:.1%}. "
                f"Position reduced to {dd_mult:.0%}"
            )

        # 4. Check daily loss limit
        daily_loss_pct = abs(min(0, self._daily_pnl)) / self.equity
        if daily_loss_pct >= self.limits.max_daily_loss:
            violations.append(
                f"Daily loss limit reached: {daily_loss_pct:.1%}. "
                f"No new positions today."
            )
            multiplier = 0.0

        is_allowed = multiplier > 0.1

        return is_allowed, violations, max(0.0, multiplier)

    def get_risk_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.

        Returns:
            Dict with all risk metrics
        """
        # Calculate VaR if we have data
        var_result = None
        if len(self._returns_history) >= 30:
            var_result = self.calculate_var()

        # Get exposure report
        exposure_report = self.exposure_manager.get_exposure_report()

        # Get correlation alerts
        corr_alerts = self.correlation_engine.detect_correlation_breakdown()

        # Run stress tests
        stress_results = self.stress_tester.run_all_scenarios(
            self._positions, self.equity
        )
        worst_case = self.stress_tester.get_worst_case(self._positions, self.equity)

        return {
            "timestamp": datetime.now().isoformat(),
            "equity": self.equity,
            "peak_equity": self._peak_equity,
            "current_drawdown": self._current_drawdown,
            "daily_pnl": self._daily_pnl,

            "var": var_result.to_dict() if var_result else None,
            "exposure": exposure_report.to_dict(),

            "correlation_alerts": [
                {
                    "pair": alert.asset_pair,
                    "old": alert.old_correlation,
                    "new": alert.new_correlation,
                    "severity": alert.severity.name
                }
                for alert in corr_alerts
            ],

            "stress_test_worst_case": {
                "scenario": worst_case["scenario_name"],
                "loss_pct": worst_case["loss_pct"],
                "survives": worst_case["survives"]
            } if worst_case else None,

            "limits": self.limits.to_dict(),
            "positions_count": len(self._positions)
        }

    def reset_daily_pnl(self) -> None:
        """Reset daily PnL tracker (call at start of day)."""
        self._daily_pnl = 0.0

    def get_dashboard(self) -> str:
        """
        Generate text-based risk dashboard.

        Returns:
            Formatted string for display
        """
        report = self.get_risk_report()
        var = report.get("var", {})
        exp = report.get("exposure", {})

        var_str = f"{var.get('var_pct', 0):.2f}%" if var else "N/A"
        cvar_str = f"{var.get('cvar_pct', 0):.2f}%" if var else "N/A"

        return f"""
================================================================================
                     PORTFOLIO RISK DASHBOARD
================================================================================

  EQUITY STATUS
  ─────────────────────────────────────────────────────────────────────────────
  Current Equity:      ${report['equity']:>15,.2f}
  Peak Equity:         ${report['peak_equity']:>15,.2f}
  Current Drawdown:    {report['current_drawdown']*100:>14.2f}%  (Limit: {self.limits.max_drawdown*100:.0f}%)
  Daily P&L:           ${report['daily_pnl']:>15,.2f}

  VALUE AT RISK (95%, 1-Day)
  ─────────────────────────────────────────────────────────────────────────────
  VaR:                 {var_str:>15}  (Limit: {self.limits.max_var_pct*100:.1f}%)
  CVaR (Expected Shortfall): {cvar_str:>8}
  Sample Size:         {var.get('sample_size', 'N/A'):>15}

  EXPOSURE
  ─────────────────────────────────────────────────────────────────────────────
  Gross Exposure:      ${exp.get('gross_exposure', 0):>15,.2f}
  Net Exposure:        ${exp.get('net_exposure', 0):>15,.2f}
  Long / Short:        ${exp.get('long_exposure', 0):>10,.0f} / ${exp.get('short_exposure', 0):>10,.0f}
  Utilization:         {exp.get('utilization_pct', 0):>14.1f}%
  Concentration (HHI): {exp.get('hhi_concentration', 0):>15.4f}

  POSITIONS: {report['positions_count']}
  CORRELATION ALERTS: {len(report.get('correlation_alerts', []))}

  WORST-CASE STRESS TEST
  ─────────────────────────────────────────────────────────────────────────────
  Scenario:            {report.get('stress_test_worst_case', {}).get('scenario', 'N/A')}
  Potential Loss:      {(report.get('stress_test_worst_case', {}).get('loss_pct', 0) or 0)*100:.1f}%
  Survives:            {'YES' if report.get('stress_test_worst_case', {}).get('survives', True) else 'NO'}

================================================================================
"""


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_portfolio_risk_manager(
    equity: float,
    preset: str = "moderate"
) -> PortfolioRiskManager:
    """
    Create a PortfolioRiskManager with preset configuration.

    Args:
        equity: Initial equity
        preset: "conservative", "moderate", "aggressive"

    Returns:
        Configured PortfolioRiskManager
    """
    presets = {
        "conservative": RiskLimits(
            max_var_pct=0.01,
            max_cvar_pct=0.015,
            max_gross_exposure=1.0,
            max_drawdown=0.05,
            max_daily_loss=0.02
        ),
        "moderate": RiskLimits(
            max_var_pct=0.02,
            max_cvar_pct=0.03,
            max_gross_exposure=1.5,
            max_drawdown=0.10,
            max_daily_loss=0.03
        ),
        "aggressive": RiskLimits(
            max_var_pct=0.03,
            max_cvar_pct=0.05,
            max_gross_exposure=2.0,
            max_drawdown=0.15,
            max_daily_loss=0.05
        )
    }

    limits = presets.get(preset, presets["moderate"])

    return PortfolioRiskManager(
        equity=equity,
        limits=limits
    )
