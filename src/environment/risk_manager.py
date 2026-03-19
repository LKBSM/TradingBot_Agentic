import logging
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Any, Tuple, Optional
import warnings

# Try to import arch for GARCH modeling
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("arch library not installed. Using fallback volatility estimation. "
                  "Install with: pip install arch")

_logger = logging.getLogger(__name__)


# =============================================================================
# Sprint 7: Async GARCH Manager — non-blocking volatility estimation
# =============================================================================

class AsyncGARCHManager:
    """
    Non-blocking GARCH(1,1) volatility estimator with double-buffering.

    - EWMA fast-path always returns in <0.01ms (the "current" buffer).
    - Full GARCH refit runs in a background thread (the "pending" buffer).
    - When the refit finishes, the EWMA state is re-seeded from the new GARCH
      estimate, giving a smooth transition with zero main-thread blocking.
    """

    def __init__(self, refit_interval: int = 2000, ewma_lambda: float = 0.94):
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="garch-refit"
        )
        self._lock = threading.Lock()
        self._refit_future: Optional[Future] = None

        # EWMA state (fast path)
        self._ewma_variance = 0.0001
        self._ewma_lambda = ewma_lambda

        # Refit scheduling
        self._refit_interval = refit_interval
        self._steps_since_refit = 0

        # Last GARCH sigma (for monitoring / metrics)
        self._last_garch_sigma: Optional[float] = None
        self._refit_count = 0

    def get_volatility(self, returns: np.ndarray) -> float:
        """
        Get current volatility estimate. Always non-blocking (<0.01ms).

        Triggers a background GARCH refit when due.
        """
        if len(returns) < 2:
            return 0.01

        # 1. Always update EWMA (fast path)
        latest_return = float(returns[-1])
        with self._lock:
            self._ewma_variance = (
                self._ewma_lambda * self._ewma_variance
                + (1 - self._ewma_lambda) * (latest_return ** 2)
            )
            sigma = float(np.sqrt(max(self._ewma_variance, 1e-12)))

        # 2. Check for completed background refit
        if self._refit_future is not None and self._refit_future.done():
            try:
                new_sigma = self._refit_future.result()
                if new_sigma is not None and new_sigma > 0:
                    with self._lock:
                        self._ewma_variance = new_sigma ** 2
                        sigma = new_sigma
                    self._last_garch_sigma = new_sigma
                    self._refit_count += 1
            except Exception:
                _logger.debug("GARCH background refit failed, continuing with EWMA")
            self._refit_future = None

        # 3. Schedule new refit if due
        self._steps_since_refit += 1
        if (self._steps_since_refit >= self._refit_interval
                and len(returns) >= 100
                and self._refit_future is None):
            self._steps_since_refit = 0
            # Copy returns to avoid race conditions
            returns_copy = returns.copy()
            self._refit_future = self._executor.submit(
                self._do_refit, returns_copy
            )

        return sigma

    @staticmethod
    def _do_refit(returns: np.ndarray) -> Optional[float]:
        """Run full GARCH(1,1) refit. Executes in background thread."""
        if not ARCH_AVAILABLE:
            return None
        try:
            scaled = returns * 100
            model = arch_model(
                scaled, vol='Garch', p=1, q=1, mean='Zero', rescale=False
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.fit(disp='off', show_warning=False)
            forecast = fitted.forecast(horizon=1, reindex=False)
            variance = forecast.variance.values[-1, 0]
            return float(np.sqrt(variance) / 100)
        except Exception:
            return None

    @property
    def ewma_sigma(self) -> float:
        with self._lock:
            return float(np.sqrt(max(self._ewma_variance, 1e-12)))

    @property
    def is_refitting(self) -> bool:
        return self._refit_future is not None and not self._refit_future.done()

    @property
    def refit_count(self) -> int:
        return self._refit_count

    def shutdown(self) -> None:
        """Cleanly shut down the background executor."""
        self._executor.shutdown(wait=False)

    def reset(self) -> None:
        """Reset step counter but keep EWMA state (volatility is persistent)."""
        self._steps_since_refit = 0


class DynamicRiskManager:

    # --- Initialization and Core State Storage ---

    def __init__(self, config: Dict[str, Any]):
        # Hard limits and risk appetite per client
        self.client_profiles = {}
        # Market state for dynamic risk scaling
        self.market_state = {'current_regime': 0, 'garch_sigma': 0.0, 'current_var': 0.0}
        # 0 = Calm/Low Volatility (Default), 1 = Chaos/High Volatility

        # Configuration parameters for risk management
        self.risk_percentage_per_trade = config.get('RISK_PERCENTAGE_PER_TRADE', 0.005)
        self.stop_loss_percentage = config.get('STOP_LOSS_PERCENTAGE', 0.01)
        self.take_profit_percentage = config.get('TAKE_PROFIT_PERCENTAGE', 0.02)
        self.min_trade_quantity = config.get('MIN_TRADE_QUANTITY', 0.01)
        self.tsl_start_profit_multiplier = config.get('TSL_START_PROFIT_MULTIPLIER', 1.0)
        self.tsl_trail_distance_multiplier = config.get('TSL_TRAIL_DISTANCE_MULTIPLIER', 0.5)

        # Variables d'état
        self.current_stop_loss = np.nan
        self.current_take_profit = np.nan
        self.tsl_activated = False
        self.is_long_position = True

        # GARCH model state (optimized with EWMA approximation between refits)
        self._garch_model = None
        self._garch_fitted = None
        self._last_garch_update = 0
        # PERFORMANCE FIX: Increased from 500 to 2000 steps
        # GARCH refitting takes 200-400ms, so reducing frequency saves ~40ms/step on average
        # EWMA approximation is used between refits (accurate to within 5% of GARCH)
        self.garch_update_frequency = config.get('GARCH_UPDATE_FREQUENCY', 2000)

        # EWMA state for fast volatility updates between GARCH refits
        self._ewma_variance = 0.0001  # Initial variance estimate
        self._ewma_lambda = 0.94  # RiskMetrics standard decay factor

        # Sprint 7: Async GARCH manager (non-blocking background refit)
        self._async_garch = AsyncGARCHManager(
            refit_interval=self.garch_update_frequency,
            ewma_lambda=self._ewma_lambda,
        )
        self._use_async_garch = config.get('USE_ASYNC_GARCH', False)

    # --- Helper Functions for Regime Adaptation ---

    def _get_regime_scaling(self, regime_state: int) -> float:
        """
        Adjusts the position sizing aggressiveness based on the market regime.
        CHANGED: 0.25 -> 0.5 (was too conservative, bot couldn't take meaningful positions)
        """
        if regime_state == 0:  # Low Volatility (Calm)
            return 1.0
        elif regime_state == 1:  # High Volatility (Chaos)
            return 0.5  # Was 0.25 - increased to allow bot to trade during learning
        return 1.0

    def _get_regime_multiplier(self, regime_state: int) -> float:
        """
        Adjusts the ATR multiplier for stop-loss distance based on the market regime.
        """
        if regime_state == 0:  # Low Volatility (Calm)
            return 2.0
        elif regime_state == 1:  # High Volatility (Chaos)
            return 3.0
        return 2.0

    def _calculate_kelly_fraction(self, win_prob: float, risk_reward_ratio: float) -> float:
        """
        Calculates the theoretical optimal Kelly fraction (f*).

        SECURITY FIX: Now logs warnings for edge cases instead of silent failure.
        """
        P = win_prob
        B = risk_reward_ratio
        Q = 1 - P

        # Kelly criterion formula: f* = (B*P - Q) / B
        if B <= 1e-9:
            # SECURITY FIX: Log warning for edge case
            import logging
            logging.getLogger(__name__).warning(
                f"Kelly edge case: risk_reward_ratio too small (B={B:.6f}), returning 0"
            )
            return 0.0

        if B * P - Q <= 0:
            # SECURITY FIX: Log warning for negative expectation
            import logging
            logging.getLogger(__name__).debug(
                f"Kelly negative expectation: P={P:.3f}, B={B:.3f}, E[R]={B*P-Q:.4f}, returning 0"
            )
            return 0.0

        return (B * P - Q) / B

    # --- Client Profile and Hard Limits ---

    def set_client_profile(self, client_id: str, initial_equity: float, max_drawdown_pct: float,
                           kelly_fraction_limit: float, max_trade_risk_pct: float):
        """
        Initializes hard risk guarantees and risk appetite per client.
        """
        if client_id not in self.client_profiles:
            self.client_profiles[client_id] = {
                'max_drawdown_pct': max_drawdown_pct,
                'kelly_fraction_limit': kelly_fraction_limit,
                'max_trade_risk_pct': max_trade_risk_pct,
                'equity_peak': initial_equity,
                'current_equity': initial_equity,
                'is_trading_halted': False
            }
        else:
            self.client_profiles[client_id].update({
                'max_drawdown_pct': max_drawdown_pct,
                'kelly_fraction_limit': kelly_fraction_limit,
                'max_trade_risk_pct': max_trade_risk_pct
            })

    def check_client_drawdown_limit(self, client_id: str, current_equity: float) -> bool:
        """
        Monitors MDD against the client's hard tolerance.
        """
        profile = self.client_profiles.get(client_id)
        if not profile or profile['is_trading_halted']:
            return True

        profile['equity_peak'] = max(profile['equity_peak'], current_equity)
        profile['current_equity'] = current_equity

        if profile['equity_peak'] > 0:
            drawdown_pct = 1.0 - (current_equity / profile['equity_peak'])
        else:
            drawdown_pct = 0.0

        if drawdown_pct * 100 >= profile['max_drawdown_pct']:
            profile['is_trading_halted'] = True
            import logging
            logging.getLogger(__name__).critical(
                "Client MDD limit breached - trading halted",
                extra={'client_id': client_id, 'drawdown_pct': round(drawdown_pct * 100, 2)}
            )
            return True

        return False

    # --- Strategic Market State Updates (Input Feeds) ---

    def calculate_garch_volatility(self, returns: np.ndarray, force_update: bool = False) -> float:
        """
        Calculates forward-looking volatility using GARCH(1,1) with EWMA approximation.

        OPTIMIZED VERSION:
        - Full GARCH refit every 500 steps (expensive, ~200-400ms)
        - Fast EWMA update every other step (~0.01ms)
        - Net result: 10-20x faster volatility estimation

        GARCH(1,1) models volatility as:
            σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)

        EWMA approximation (between refits):
            σ²(t) = λ·σ²(t-1) + (1-λ)·r²(t-1)
            where λ = 0.94 (RiskMetrics standard)

        Args:
            returns: Array of historical returns
            force_update: If True, refit the model even if not due

        Returns:
            Forecasted 1-step ahead volatility (sigma)
        """
        # Minimum data requirement
        if len(returns) < 20:
            self.market_state['garch_sigma'] = 0.01
            return 0.01

        # Short history: use simple rolling volatility
        if len(returns) < 100:
            self.market_state['garch_sigma'] = float(np.std(returns[-20:]))
            return self.market_state['garch_sigma']

        # Sprint 7: Non-blocking async path
        if self._use_async_garch and not force_update:
            sigma = self._async_garch.get_volatility(returns)
            self.market_state['garch_sigma'] = sigma
            return sigma

        # Check if we should do full GARCH refit
        self._last_garch_update += 1
        should_refit = force_update or (self._last_garch_update >= self.garch_update_frequency)

        # ═══════════════════════════════════════════════════════════════════════
        # FAST PATH: EWMA approximation between GARCH refits (~0.01ms)
        # ═══════════════════════════════════════════════════════════════════════
        if not should_refit and self._ewma_variance > 0:
            # Update EWMA variance with latest return
            latest_return = returns[-1]
            self._ewma_variance = (
                self._ewma_lambda * self._ewma_variance +
                (1 - self._ewma_lambda) * (latest_return ** 2)
            )
            self.market_state['garch_sigma'] = float(np.sqrt(self._ewma_variance))
            return self.market_state['garch_sigma']

        # ═══════════════════════════════════════════════════════════════════════
        # SLOW PATH: Full GARCH refit (~200-400ms, every 500 steps)
        # ═══════════════════════════════════════════════════════════════════════
        if ARCH_AVAILABLE:
            try:
                # Scale returns to percentage for numerical stability
                scaled_returns = returns * 100

                # GARCH(1,1) with normal distribution
                model = arch_model(
                    scaled_returns,
                    vol='Garch',
                    p=1,
                    q=1,
                    mean='Zero',
                    rescale=False
                )

                # Fit with suppressed output
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._garch_fitted = model.fit(disp='off', show_warning=False)

                self._garch_model = model
                self._last_garch_update = 0

                # Forecast 1-step ahead variance
                forecast = self._garch_fitted.forecast(horizon=1, reindex=False)
                variance_forecast = forecast.variance.values[-1, 0]

                # Convert back from percentage scale
                sigma = np.sqrt(variance_forecast) / 100
                self.market_state['garch_sigma'] = float(sigma)

                # Initialize EWMA with GARCH estimate for smooth transitions
                self._ewma_variance = sigma ** 2

            except Exception:
                # Fallback: use EWMA
                self._calculate_ewma_volatility(returns)
        else:
            # No GARCH available: use pure EWMA
            self._calculate_ewma_volatility(returns)

        return self.market_state['garch_sigma']

    def _calculate_ewma_volatility(self, returns: np.ndarray) -> None:
        """
        Calculate volatility using Exponentially Weighted Moving Average.

        This is the RiskMetrics approach with λ=0.94.
        Used as fallback when GARCH is unavailable or fails.
        """
        # Initialize EWMA if needed
        # SECURITY FIX: Add floor to prevent zero variance (which would cause divide-by-zero downstream)
        if self._ewma_variance <= 0:
            calculated_var = float(np.var(returns[-20:]))
            self._ewma_variance = max(1e-8, calculated_var)  # Floor at 1e-8

        # Calculate full EWMA for initialization/reset
        weights = np.array([
            (1 - self._ewma_lambda) * (self._ewma_lambda ** i)
            for i in range(min(75, len(returns)))
        ])
        weights = weights[::-1]
        weights = weights / weights.sum()

        recent_returns = returns[-len(weights):]
        self._ewma_variance = float(np.sum(weights * (recent_returns ** 2)))
        self.market_state['garch_sigma'] = float(np.sqrt(self._ewma_variance))
        self._last_garch_update = 0

    def get_volatility_forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Get multi-step volatility forecast.

        Args:
            returns: Historical returns array
            horizon: Number of steps to forecast

        Returns:
            Array of forecasted volatilities for each step
        """
        if not ARCH_AVAILABLE or self._garch_fitted is None:
            # Fallback: constant volatility forecast
            current_vol = self.market_state.get('garch_sigma', 0.01)
            return np.full(horizon, current_vol)

        try:
            forecast = self._garch_fitted.forecast(horizon=horizon, reindex=False)
            variances = forecast.variance.values[-1, :]
            return np.sqrt(variances) / 100  # Convert from percentage scale
        except Exception:
            current_vol = self.market_state.get('garch_sigma', 0.01)
            return np.full(horizon, current_vol)

    def get_regime_state(self, historical_returns):
        """
        Simulates update from a trained HMM predicting the market state. (Placeholder implementation)
        """
        recent_vol = np.std(historical_returns[-20:])
        if recent_vol > 0.01:
            self.market_state['current_regime'] = 1
        else:
            self.market_state['current_regime'] = 0

    # --- Dynamic Calculation Logic ---

    def calculate_atr_stop_loss(self, entry_price: float, atr: float, is_long: bool) -> Tuple[float, float]:
        """
        Calculates the stop loss price and absolute distance based on ATR and market regime.
        """
        regime = self.market_state.get('current_regime', 0)
        atr_multiplier = self._get_regime_multiplier(regime)

        sl_distance = atr_multiplier * atr

        if is_long:
            stop_loss_price = entry_price - sl_distance
        else:
            stop_loss_price = entry_price + sl_distance

        return stop_loss_price, sl_distance

    def set_trade_orders(self, entry_price: float, atr: float, is_long: bool) -> float:
        """
        Sets the dynamic Stop Loss and Take Profit levels at trade entry.
        Returns the absolute stop loss distance (in price units).
        """
        self.is_long_position = is_long

        # 1. Calculate SL distance based on regime-adjusted ATR
        stop_loss_price, sl_distance = self.calculate_atr_stop_loss(entry_price, atr, is_long)
        self.current_stop_loss = stop_loss_price
        # v4: Store ATR multiplier for live R:R calculation
        regime = self.market_state.get('current_regime', 0)
        self.atr_multiplier = self._get_regime_multiplier(regime)

        # 2. Calculate ATR-based TP (v4: proportional to ATR, symmetric with SL)
        # TP_ATR_MULTIPLIER / SL_ATR_MULTIPLIER = R:R ratio (4.0 / 2.0 = 2:1)
        from config import TP_ATR_MULTIPLIER
        tp_distance = TP_ATR_MULTIPLIER * atr
        if self.is_long_position:
            self.current_take_profit = entry_price + tp_distance
        else:
            self.current_take_profit = entry_price - tp_distance

        self.tsl_activated = False
        return sl_distance

    def update_trailing_stop(self, entry_price: float, current_price: float,
                             current_atr: float, is_long: bool,
                             high: float = None, low: float = None):
        """
        Updates the Trailing Stop Loss if the position is in profit.

        Sprint 9: Uses bar High (for longs) or Low (for shorts) as the best
        price for TSL advancement, giving more accurate trailing behavior.
        """
        self.is_long_position = is_long

        if np.isnan(self.current_stop_loss) or np.isnan(entry_price) or current_atr <= 1e-9:
            return

        # Sprint 9: Use High for longs, Low for shorts as best intra-bar price
        if self.is_long_position:
            best_price = high if high is not None else current_price
        else:
            best_price = low if low is not None else current_price

        # Calculate absolute profit using best intra-bar price
        profit_abs = best_price - entry_price if self.is_long_position else entry_price - best_price

        # TSL activation threshold is based on a multiplier of current ATR
        tsl_activation_threshold = self.tsl_start_profit_multiplier * current_atr

        if profit_abs > tsl_activation_threshold:
            self.tsl_activated = True

            # TSL trail distance is a multiple of ATR
            tsl_trail_distance = self.tsl_trail_distance_multiplier * current_atr

            # Determine the new stop loss price
            if self.is_long_position:
                new_sl = best_price - tsl_trail_distance
                # Only raise the SL (never lower it)
                self.current_stop_loss = max(self.current_stop_loss, new_sl)
            else:
                new_sl = best_price + tsl_trail_distance
                # Only lower the SL (never raise it)
                self.current_stop_loss = min(self.current_stop_loss, new_sl)

    def check_trade_exit(self, current_price: float, is_long: bool,
                         high: float = None, low: float = None) -> Tuple[str, float]:
        """
        Checks if exit conditions (TP, SL) are met using intra-bar High/Low.

        Sprint 9: Uses High and Low to detect SL/TP touches within the bar,
        not just at Close. Returns (signal, fill_price) where fill_price is
        the SL/TP level (not the Close), simulating realistic broker fills.

        Returns:
            Tuple of (exit_signal, fill_price):
            - ('SL', sl_price) if stop loss was hit
            - ('TP', tp_price) if take profit was hit
            - ('none', current_price) if no exit
        """
        self.is_long_position = is_long
        bar_high = high if high is not None else current_price
        bar_low = low if low is not None else current_price

        if self.is_long_position:
            # Check Stop Loss: Low touches or crosses below SL
            if not np.isnan(self.current_stop_loss) and bar_low <= self.current_stop_loss:
                return 'SL', self.current_stop_loss
            # Check Take Profit: High touches or crosses above TP
            if not np.isnan(self.current_take_profit) and bar_high >= self.current_take_profit:
                return 'TP', self.current_take_profit
        else:  # Short position
            # Check Stop Loss: High touches or crosses above SL
            if not np.isnan(self.current_stop_loss) and bar_high >= self.current_stop_loss:
                return 'SL', self.current_stop_loss
            # Check Take Profit: Low touches or crosses below TP
            if not np.isnan(self.current_take_profit) and bar_low <= self.current_take_profit:
                return 'TP', self.current_take_profit

        return 'none', current_price

    def reset(self) -> None:
        """
        Resets the internal state for a new episode.

        Note: GARCH model and EWMA state are preserved between episodes
        (expensive to refit, and volatility is persistent)
        """
        self.current_stop_loss = np.nan
        self.current_take_profit = np.nan
        self.tsl_activated = False
        self.is_long_position = True
        self.atr_multiplier = 2.0  # Default regime multiplier
        # Reset GARCH update counter but keep the fitted model and EWMA state
        self._last_garch_update = 0
        # Don't reset _ewma_variance - volatility is persistent across episodes
        # Sprint 7: Reset async GARCH step counter
        self._async_garch.reset()

    def calculate_adaptive_position_size(self, client_id: str, account_equity: float, atr_stop_distance: float,
                                         win_prob: float,
                                         risk_reward_ratio: float,
                                         current_price: float = None,
                                         max_leverage: float = 1.0,
                                         is_long: bool = True,
                                         training_mode: bool = True) -> float:
        """
        Calculates the optimal trade size using a TRIPLE constraint system:
        1. Fixed Risk Limit (Risk Neutral)
        2. Dynamic Risk Appetite (Kelly Criterion, scaled by regime)
        3. HARD LEVERAGE LIMIT (NEW - prevents overleveraged positions)

        Args:
            client_id: Client identifier for profile lookup
            account_equity: Current account equity
            atr_stop_distance: Stop loss distance in price units
            win_prob: Estimated win probability (0-1)
            risk_reward_ratio: Expected R:R ratio
            current_price: Current asset price (needed for leverage calc)
            max_leverage: Maximum allowed leverage (default 1.0 = no leverage)
            is_long: True for long positions, False for shorts
            training_mode: If True, apply Kelly floor for exploration.
                          If False (eval/live), Kelly=0 → position_size=0 (no edge = no trade).

        Returns:
            Position size capped by all three constraints
        """
        profile = self.client_profiles.get(client_id)
        regime = self.market_state['current_regime']

        if not profile or account_equity <= 0:
            return 0.0

        # SECURITY FIX: Use fallback ATR instead of returning 0 on zero ATR
        if atr_stop_distance <= 1e-9:
            if current_price is not None and current_price > 0:
                # Fallback: Use 1% of price as minimum ATR distance
                atr_stop_distance = current_price * 0.01
                import logging
                logging.getLogger(__name__).warning(
                    f"ATR too small, using fallback: {atr_stop_distance:.4f}"
                )
            else:
                # No price available, cannot calculate safe position size
                return 0.0

        # --- 1. Risk Neutral (RN) Sizing: Fixed dollar risk limit ---
        max_risk_dollar = account_equity * profile['max_trade_risk_pct']
        # Size = Risk in $ / Risk per Unit
        size_rn = max_risk_dollar / atr_stop_distance

        # --- 2. Kelly Criterion (FK) Sizing: Dynamic risk appetite ---
        full_kelly_fraction = self._calculate_kelly_fraction(win_prob, risk_reward_ratio)

        # Apply regime scaling to the Kelly limit (less aggressive in Chaos regime)
        kelly_fraction_limit = profile['kelly_fraction_limit'] * self._get_regime_scaling(regime)

        # Sprint 5: Conditional Kelly floor based on training_mode
        # Training: floor at 0.02 so the RL agent can explore even with no edge
        # Live/Eval: respect Kelly=0 → no mathematical edge = no trade
        if training_mode:
            effective_kelly_fraction = max(0.02, min(full_kelly_fraction, kelly_fraction_limit))
        else:
            if full_kelly_fraction <= 0:
                return 0.0  # No edge → no trade
            effective_kelly_fraction = min(full_kelly_fraction, kelly_fraction_limit)
        capital_alloc_kelly = account_equity * effective_kelly_fraction

        # Size = Capital Allocated / Risk per Unit (if ATR distance is used as a proxy for price)
        size_fk = capital_alloc_kelly / atr_stop_distance

        # --- 3. HARD LEVERAGE LIMIT (NEW) ---
        # This ensures position value never exceeds max_leverage × equity
        # Formula: max_position_value = max_leverage × equity
        #          max_quantity = max_position_value / price
        if current_price is not None and current_price > 0:
            max_position_value = max_leverage * account_equity
            size_leverage_limit = max_position_value / current_price
        else:
            # If no price provided, skip leverage check (backward compatible)
            size_leverage_limit = float('inf')

        # The final position size is the MOST CONSERVATIVE result from ALL methods
        final_size = min(size_rn, size_fk, size_leverage_limit)

        # Sprint 7: Removed dead correlation_multiplier code.
        # Was always 1.0 (no-op) in single-asset (XAU/USD) backtesting.
        # TODO: If multi-asset support is added, implement cross-asset correlation
        # adjustment here using RiskIntegrationAgent output.

        # Ensure minimum trade quantity
        if final_size < self.min_trade_quantity:
            return 0.0

        return final_size