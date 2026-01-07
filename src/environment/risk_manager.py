import numpy as np
from typing import Dict, Any, Tuple


class DynamicRiskManager:

    # --- Initialization and Core State Storage ---

    def __init__(self, config: Dict[str, Any]):
        # Hard limits and risk appetite per client
        self.client_profiles = {}
        # Market state for dynamic risk scaling
        self.market_state = {'current_regime': 0, 'garch_sigma': 0.0, 'aggregate_cvar': 0.0}
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
        """
        P = win_prob
        B = risk_reward_ratio
        Q = 1 - P

        # Kelly criterion formula: f* = (B*P - Q) / B
        if B * P - Q <= 0 or B <= 1e-9:  # Prevent negative or zero stakes
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
            print(f"CRITICAL: Client {client_id} MDD limit breached ({drawdown_pct * 100:.2f}%). Trading halted.")
            return True

        return False

    # --- Strategic Market State Updates (Input Feeds) ---

    def calculate_garch_volatility(self, historical_data):
        """
        Simulates update from an external GARCH model. (Placeholder implementation)
        """
        if np.mean(historical_data[-50:]) > 0.005:
            self.market_state['garch_sigma'] = 0.02
        else:
            self.market_state['garch_sigma'] = 0.005

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

        # 2. Calculate fixed TP based on configured percentage
        if self.is_long_position:
            self.current_take_profit = entry_price * (1 + self.take_profit_percentage)
        else:
            self.current_take_profit = entry_price * (1 - self.take_profit_percentage)

        self.tsl_activated = False
        return sl_distance

    def update_trailing_stop(self, entry_price: float, current_price: float, current_atr: float, is_long: bool):
        """
        Updates the Trailing Stop Loss if the position is in profit.
        """
        self.is_long_position = is_long

        if np.isnan(self.current_stop_loss) or np.isnan(entry_price) or current_atr <= 1e-9:
            return

        # Calculate absolute profit
        profit_abs = current_price - entry_price if self.is_long_position else entry_price - current_price

        # TSL activation threshold is based on a multiplier of current ATR
        tsl_activation_threshold = self.tsl_start_profit_multiplier * current_atr

        if profit_abs > tsl_activation_threshold:
            self.tsl_activated = True

            # TSL trail distance is a multiple of ATR
            tsl_trail_distance = self.tsl_trail_distance_multiplier * current_atr

            # Determine the new stop loss price
            if self.is_long_position:
                new_sl = current_price - tsl_trail_distance
                # Only raise the SL (never lower it)
                self.current_stop_loss = max(self.current_stop_loss, new_sl)
            else:
                new_sl = current_price + tsl_trail_distance
                # Only lower the SL (never raise it)
                self.current_stop_loss = min(self.current_stop_loss, new_sl)

    def check_trade_exit(self, current_price: float, is_long: bool) -> str:
        """
        Checks if exit conditions (TP, SL) are met.
        """
        self.is_long_position = is_long

        if self.is_long_position:
            # Check Stop Loss (price hits or crosses below SL)
            if not np.isnan(self.current_stop_loss) and current_price <= self.current_stop_loss:
                return 'SL'
            # Check Take Profit (price hits or crosses above TP)
            if not np.isnan(self.current_take_profit) and current_price >= self.current_take_profit:
                return 'TP'
        else:  # Short position
            # Check Stop Loss (price hits or crosses above SL)
            if not np.isnan(self.current_stop_loss) and current_price >= self.current_stop_loss:
                return 'SL'
            # Check Take Profit (price hits or crosses below TP)
            if not np.isnan(self.current_take_profit) and current_price <= self.current_take_profit:
                return 'TP'

        return 'none'

    def reset(self) -> None:
        """
        Resets the internal state for a new episode.
        """
        self.current_stop_loss = np.nan
        self.current_take_profit = np.nan
        self.tsl_activated = False
        self.is_long_position = True

    def calculate_adaptive_position_size(self, client_id: str, account_equity: float, atr_stop_distance: float,
                                         win_prob: float,
                                         risk_reward_ratio: float,
                                         current_price: float = None,
                                         max_leverage: float = 1.0,
                                         is_long: bool = True) -> float:
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

        Returns:
            Position size capped by all three constraints
        """
        profile = self.client_profiles.get(client_id)
        regime = self.market_state['current_regime']

        if not profile or atr_stop_distance <= 1e-9 or account_equity <= 0:
            return 0.0

        # --- 1. Risk Neutral (RN) Sizing: Fixed dollar risk limit ---
        max_risk_dollar = account_equity * profile['max_trade_risk_pct']
        # Size = Risk in $ / Risk per Unit
        size_rn = max_risk_dollar / atr_stop_distance

        # --- 2. Kelly Criterion (FK) Sizing: Dynamic risk appetite ---
        full_kelly_fraction = self._calculate_kelly_fraction(win_prob, risk_reward_ratio)

        # Apply regime scaling to the Kelly limit (less aggressive in Chaos regime)
        kelly_fraction_limit = profile['kelly_fraction_limit'] * self._get_regime_scaling(regime)

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

        # Ensure minimum trade quantity
        if final_size < self.min_trade_quantity:
            return 0.0

        return final_size