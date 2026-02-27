# =============================================================================
# VaR ENGINE - Production Value-at-Risk Service
# =============================================================================
# Wraps VectorizedRiskCalculator with a rolling returns buffer for real-time
# VaR computation. Designed to feed the kill switch and risk manager.
#
# Usage:
#   engine = VaREngine(confidence=0.95, window=252, method='cornish_fisher')
#   engine.update(portfolio_return=0.001)
#   result = engine.compute()
#   # result.var_95, result.cvar_95, etc.
# =============================================================================

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from src.performance.vectorized_risk import VectorizedRiskCalculator

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Result of a VaR computation cycle."""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    method: str = ''
    window_size: int = 0
    computation_time_ms: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'var_95': round(self.var_95, 6),
            'var_99': round(self.var_99, 6),
            'cvar_95': round(self.cvar_95, 6),
            'cvar_99': round(self.cvar_99, 6),
            'method': self.method,
            'window_size': self.window_size,
            'computation_time_ms': round(self.computation_time_ms, 3),
        }


# Minimum observations required for meaningful VaR
_MIN_OBSERVATIONS = 30


class VaREngine:
    """
    Production VaR service wrapping VectorizedRiskCalculator.

    Maintains a rolling buffer of portfolio returns and computes VaR
    on demand using the configured method. Designed for integration
    with KillSwitch.update(var_pct=...) and DynamicRiskManager.

    Thread-safe: all public methods are safe to call from any thread.
    """

    # Valid VaR methods
    METHODS = ('historical', 'parametric', 'monte_carlo', 'cornish_fisher')

    def __init__(
        self,
        confidence: float = 0.95,
        window: int = 252,
        method: str = 'cornish_fisher',
    ):
        if method not in self.METHODS:
            raise ValueError(f"Unknown VaR method '{method}'. Valid: {self.METHODS}")
        if not 0.5 < confidence < 1.0:
            raise ValueError(f"Confidence must be in (0.5, 1.0), got {confidence}")
        if window < _MIN_OBSERVATIONS:
            raise ValueError(f"Window must be >= {_MIN_OBSERVATIONS}, got {window}")

        self._confidence = confidence
        self._window = window
        self._method = method
        self._buffer: deque = deque(maxlen=window)
        self._calc = VectorizedRiskCalculator()
        self._last_result: Optional[VaRResult] = None

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def method(self) -> str:
        return self._method

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True when enough observations have been collected."""
        return len(self._buffer) >= _MIN_OBSERVATIONS

    def update(self, portfolio_return: float) -> None:
        """
        Append a new portfolio return to the rolling buffer.

        Args:
            portfolio_return: Single-period return (e.g., 0.001 = +0.1%)
        """
        self._buffer.append(float(portfolio_return))

    def update_batch(self, returns: np.ndarray) -> None:
        """
        Append multiple returns at once (e.g., backfill on startup).

        Args:
            returns: Array of returns, oldest first.
        """
        for r in returns:
            self._buffer.append(float(r))

    def compute(self, method: Optional[str] = None) -> VaRResult:
        """
        Compute VaR using the configured (or overridden) method.

        Args:
            method: Override the default method for this call.

        Returns:
            VaRResult with var_95, var_99, cvar_95, cvar_99.
        """
        method = method or self._method
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'")

        result = VaRResult(
            method=method,
            window_size=len(self._buffer),
            timestamp=time.time(),
        )

        if not self.is_ready:
            logger.debug(
                "VaR engine not ready: %d/%d observations",
                len(self._buffer), _MIN_OBSERVATIONS
            )
            self._last_result = result
            return result

        returns = np.array(self._buffer)
        t0 = time.perf_counter()

        # Compute VaR at 95% and 99%
        result.var_95 = self._compute_var(returns, 0.95, method)
        result.var_99 = self._compute_var(returns, 0.99, method)

        # CVaR (always historical — it's the expected shortfall beyond VaR)
        result.cvar_95 = self._calc.cvar(returns, 0.95)
        result.cvar_99 = self._calc.cvar(returns, 0.99)

        result.computation_time_ms = (time.perf_counter() - t0) * 1000
        self._last_result = result

        return result

    def compute_all_methods(self) -> Dict[str, VaRResult]:
        """
        Compute VaR using all 4 methods for comparison/validation.

        Returns:
            Dict mapping method name to VaRResult.
        """
        results = {}
        for m in self.METHODS:
            results[m] = self.compute(method=m)
        return results

    @property
    def last_result(self) -> Optional[VaRResult]:
        """Last computed VaR result (cached)."""
        return self._last_result

    def _compute_var(
        self, returns: np.ndarray, confidence: float, method: str
    ) -> float:
        """Dispatch to the correct VaR method."""
        if method == 'historical':
            return self._calc.var_historical(returns, confidence)
        elif method == 'parametric':
            return self._calc.var_parametric(returns, confidence)
        elif method == 'monte_carlo':
            return self._calc.var_monte_carlo(returns, confidence)
        elif method == 'cornish_fisher':
            return self._calc.var_cornish_fisher(returns, confidence)
        else:
            raise ValueError(f"Unknown method: {method}")

    def reset(self) -> None:
        """Clear the returns buffer and cached result."""
        self._buffer.clear()
        self._last_result = None
