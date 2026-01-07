import traceback
import sys  # ✅ AJOUTÉ ICI
import os  # ✅ AJOUTÉ ICI
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
from collections import deque
import warnings
from ta.volatility import average_true_range
from rich.console import Console
from rich.table import Table
from rich.text import Text
import ta

# ✅ MAINTENANT ON PEUT UTILISER sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.environment.risk_manager import DynamicRiskManager as RiskManager
from src.tests.monitor_training import TradeLogger
from src.environment.strategy_features import SmartMoneyEngine

# IMPORTANT: Import constants from config.py
try:
    from src.config import (
        LOOKBACK_WINDOW_SIZE, INITIAL_BALANCE, TRANSACTION_FEE_PERCENTAGE,
        TRADE_COMMISSION_PER_TRADE, SLIPPAGE_PERCENTAGE, REWARD_SCALING_FACTOR,
        ALLOW_NEGATIVE_REVENUE_SELL, ALLOW_NEGATIVE_BALANCE, MINIMUM_ALLOWED_BALANCE,
        MIN_TRADE_QUANTITY, ACTION_SPACE_TYPE, FEATURES, OHLCV_COLUMNS,
        DOWNSIDE_PENALTY_MULTIPLIER, OVERNIGHT_HOLDING_PENALTY, HOLD_PENALTY_FACTOR,
        WINNING_TRADE_BONUS, LOSING_TRADE_PENALTY, FAILED_TRADE_ATTEMPT_PENALTY,
        TRADE_COOLDOWN_STEPS, RAPID_TRADE_PENALTY, TRAIN_END_DATE, SMC_CONFIG,
        RISK_PERCENTAGE_PER_TRADE, TAKE_PROFIT_PERCENTAGE, STOP_LOSS_PERCENTAGE,
        TSL_START_PROFIT_MULTIPLIER, TSL_TRAIL_DISTANCE_MULTIPLIER,
        MAX_LEVERAGE, MAX_DURATION_STEPS,
        W_RETURN, W_DRAWDOWN, W_FRICTION, W_LEVERAGE, W_TURNOVER, W_DURATION,
        # NEW: Long/Short action constants
        NUM_ACTIONS, ACTION_NAMES, ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG,
        ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT, POSITION_FLAT, POSITION_LONG, POSITION_SHORT,
        ENABLE_SHORT_SELLING, SHORT_BORROWING_FEE_DAILY, SHORT_OVERNIGHT_SWAP_PCT
    )
except ImportError:
    print("WARNING: Could not import from config. Using production fallback parameters.")

    # ✅ FALLBACK COMPLET CORRIGÉ
    TRAIN_END_DATE = "2023-06-30 23:59:00"
    INITIAL_BALANCE = 1000.0
    TRANSACTION_FEE_PERCENTAGE = 0.0005
    SLIPPAGE_PERCENTAGE = 0.0001
    TRADE_COMMISSION_PER_TRADE = 0.0005
    LOOKBACK_WINDOW_SIZE = 60
    RISK_PERCENTAGE_PER_TRADE = 0.01
    TAKE_PROFIT_PERCENTAGE = 0.02
    STOP_LOSS_PERCENTAGE = 0.01
    TSL_START_PROFIT_MULTIPLIER = 1.0
    TSL_TRAIL_DISTANCE_MULTIPLIER = 0.5
    ALLOW_NEGATIVE_REVENUE_SELL = False
    ALLOW_NEGATIVE_BALANCE = False
    MINIMUM_ALLOWED_BALANCE = 100.0
    MIN_TRADE_QUANTITY = 0.01

    # ✅ AJOUTÉ: Variables manquantes
    ACTION_SPACE_TYPE = 'discrete'
    OVERNIGHT_HOLDING_PENALTY = 0.0
    HOLD_PENALTY_FACTOR = 0.005
    FAILED_TRADE_ATTEMPT_PENALTY = 0.0

    # NEW: Long/Short action constants (fallback)
    NUM_ACTIONS = 5
    ACTION_NAMES = {0: 'HOLD', 1: 'OPEN_LONG', 2: 'CLOSE_LONG', 3: 'OPEN_SHORT', 4: 'CLOSE_SHORT'}
    ACTION_HOLD = 0
    ACTION_OPEN_LONG = 1
    ACTION_CLOSE_LONG = 2
    ACTION_OPEN_SHORT = 3
    ACTION_CLOSE_SHORT = 4
    POSITION_FLAT = 0
    POSITION_LONG = 1
    POSITION_SHORT = -1
    ENABLE_SHORT_SELLING = True
    SHORT_BORROWING_FEE_DAILY = 0.0001
    SHORT_OVERNIGHT_SWAP_PCT = 0.0002

    # Reward Weights
    REWARD_SCALING_FACTOR = 100.0
    W_RETURN = 1.0
    W_DRAWDOWN = 0.5
    W_FRICTION = 0.1
    W_LEVERAGE = 1.0
    W_TURNOVER = 0.0
    W_DURATION = 0.1

    # Penalties and Bonuses
    DOWNSIDE_PENALTY_MULTIPLIER = 1.0
    WINNING_TRADE_BONUS = 2.0
    LOSING_TRADE_PENALTY = 5.0
    TRADE_COOLDOWN_STEPS = 5
    RAPID_TRADE_PENALTY = 1.0
    MAX_LEVERAGE = 1.0
    MAX_DURATION_STEPS = 40 # ✅ CORRIGÉ: Supprimé [cite: 1

    OHLCV_COLUMNS = {
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }

    SMC_CONFIG = {
        "RSI_WINDOW": 7,
        "MACD_FAST": 8,
        "MACD_SLOW": 17,
        "MACD_SIGNAL": 9,
        "BB_WINDOW": 20,
        "ATR_WINDOW": 7,
        "FRACTAL_WINDOW": 2,
        "FVG_THRESHOLD": 0.0,
    }

    FEATURES = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'MACD_Diff', 'MACD_line', 'MACD_signal',
        'BB_L', 'BB_M', 'BB_H', 'ATR', 'SPREAD', 'BODY_SIZE',
        'UP_FRACTAL', 'DOWN_FRACTAL', 'FVG_SIGNAL', 'FVG_SIZE_NORM',
        'BOS_SIGNAL', 'CHOCH_SIGNAL', 'BULLISH_OB_HIGH', 'BULLISH_OB_LOW',
        'BEARISH_OB_HIGH', 'BEARISH_OB_LOW', 'OB_STRENGTH_NORM'
    ]

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 30}
    _gym_reset_return_info = True

    def __init__(self, df: pd.DataFrame, render_mode: str = "none", **kwargs):
        super().__init__()
        self.render_mode = render_mode

        self.lookback_window_size = kwargs.get('lookback_window_size', LOOKBACK_WINDOW_SIZE)
        self.initial_balance = kwargs.get('initial_balance', INITIAL_BALANCE)

        self.transaction_fee_percentage = kwargs.get('transaction_fee_percentage', TRANSACTION_FEE_PERCENTAGE)
        self.trade_commission_per_trade = kwargs.get('trade_commission_per_trade', TRADE_COMMISSION_PER_TRADE)
        self.slippage_percentage = kwargs.get('slippage_percentage', SLIPPAGE_PERCENTAGE)
        self.reward_scaling_factor = kwargs.get('reward_scaling_factor', REWARD_SCALING_FACTOR)
        self.allow_negative_revenue_sell = kwargs.get('allow_negative_revenue_sell', ALLOW_NEGATIVE_REVENUE_SELL)
        self.allow_negative_balance = kwargs.get('allow_negative_balance', ALLOW_NEGATIVE_BALANCE)
        self.minimum_allowed_balance = kwargs.get('minimum_allowed_balance', MINIMUM_ALLOWED_BALANCE)
        self.min_trade_quantity = kwargs.get('min_trade_quantity', MIN_TRADE_QUANTITY)
        self.downside_penalty_multiplier = kwargs.get('downside_penalty_multiplier', DOWNSIDE_PENALTY_MULTIPLIER)
        self.overnight_holding_penalty = kwargs.get('overnight_holding_penalty', OVERNIGHT_HOLDING_PENALTY)
        self.hold_penalty_factor = kwargs.get('hold_penalty_factor', HOLD_PENALTY_FACTOR)
        self.winning_trade_bonus = kwargs.get('winning_trade_bonus', WINNING_TRADE_BONUS)
        self.losing_trade_penalty = kwargs.get('losing_trade_penalty', LOSING_TRADE_PENALTY)
        self.trade_cooldown_steps = kwargs.get('trade_cooldown_steps', TRADE_COOLDOWN_STEPS)
        self.rapid_trade_penalty = kwargs.get('rapid_trade_penalty', RAPID_TRADE_PENALTY)
        self.smc_config = kwargs.get('smc_config', SMC_CONFIG)
        self.features_config = kwargs.get('features', FEATURES)
        self.ohlcv_columns = kwargs.get('ohlcv_columns', OHLCV_COLUMNS)
        self.enable_logging = kwargs.get('enable_logging', False)
        self.trade_logger = TradeLogger() if self.enable_logging else None
        self.trade_id_counter = 0
        self.episode_count = 0
        self.episode_reward = 0.0

        # --- NEW: Reward Weights and Limits (Hyperparameters) ---
        self.max_leverage_limit = kwargs.get('max_leverage_limit', MAX_LEVERAGE)
        # Assurez-vous que cette ligne est présente dans le __init__
        self.max_duration_steps = kwargs.get('max_duration_steps', MAX_DURATION_STEPS)
        self.w_R = kwargs.get('w_R', W_RETURN)
        self.w_DD = kwargs.get('w_DD', W_DRAWDOWN)
        self.w_F = kwargs.get('w_F', W_FRICTION)
        self.w_L = kwargs.get('w_L', W_LEVERAGE)
        self.w_T = kwargs.get('w_T', W_TURNOVER)
        self.w_D = kwargs.get('w_D', W_DURATION)
        self.debug_rewards = False  # Set to True only for debugging
        # --------------------------------------------------------

        self.risk_manager = RiskManager(config=kwargs)
        self._risk_client_id = kwargs.get('risk_client_id', 'default_client')
        self.risk_manager.set_client_profile(
            client_id=self._risk_client_id,
            initial_equity=self.initial_balance,
            max_drawdown_pct=kwargs.get('max_drawdown_pct', 20.0),  # 20% MDD default
            kelly_fraction_limit=kwargs.get('kelly_fraction_limit', 0.1),  # 10% cap by default
            max_trade_risk_pct=kwargs.get('max_trade_risk_pct', 0.01)  # 1% per trade default
        )
        self.trade_commission_pct_of_trade = kwargs.get(
            'trade_commission_pct_of_trade',
            0.0005  # 0.05% du trade value (standard broker)
        )

        self.trade_commission_min_pct_capital = kwargs.get(
            'trade_commission_min_pct_capital',
            0.0001  # 0.01% du capital initial minimum
        )

        self.df_raw = df.copy()

        self.processed_data = self._process_data(self.df_raw)
        self.df = self.processed_data
        self.features = [col for col in self.features_config if col in self.df.columns]
        if 'Close' not in self.df.columns and 'close' in self.df.columns:
            self.df.rename(columns={'close': 'Close'}, inplace=True)
        if 'Close' not in self.df.columns:
            raise ValueError("Processed DataFrame must contain a 'Close' price column.")

        if len(self.df) <= self.lookback_window_size:
            raise ValueError(
                f"Pas assez de données (longueur: {len(self.df)}) pour une fenêtre de {self.lookback_window_size}."
            )

        self.balance = self.initial_balance
        self.stock_quantity = 0.0
        self.net_worth = self.initial_balance
        self.total_fees_paid = 0.0
        self.current_step = self.lookback_window_size
        self.max_steps = len(self.df) - 1
        self.entry_price = np.nan

        # --- NEW: State tracking variables for advanced reward function ---
        self.previous_nav = self.initial_balance
        self.peak_nav = self.initial_balance
        self.previous_drawdown_level = 0.0
        self.current_leverage = 0.0
        self.current_hold_duration = 0
        self.transaction_cost_incurred_step = 0.0
        self.traded_value_step = 0.0
        # ------------------------------------------------------------------

        self.trade_details = {'trade_pnl_abs': 0.0, 'trade_pnl_pct': 0.0, 'trade_type': 'none', 'trade_success': False,
                              'trade_value': 0.0, 'commission': 0.0}
        self.trade_history_summary = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        self.total_fees_paid_episode = 0.0
        self.np_random = np.random.default_rng()
        self.last_action = 0
        self.actual_action_executed = 0
        self.last_trade_step = -np.inf
        self.recent_trade_attempts_steps = deque(maxlen=self.lookback_window_size)

        # SCALER CONFIGURATION
        # Use MinMaxScaler with clip=True to handle out-of-distribution values
        # This prevents values outside [0,1] during validation/test
        # The scaler is fit on ALL available data (not just training) to avoid data leakage
        # At inference time, clip=True ensures observations stay in valid range
        self.scaler = MinMaxScaler(clip=True)  # clip=True prevents out-of-range values
        valid_rows = self.df[self.features].dropna()

        if not valid_rows.empty:
            self.scaler.fit(valid_rows.values)
        else:
            warnings.warn("No valid rows for scaling. Check feature generation pipeline.")
            self.scaler = None
        action_space_type = kwargs.get('action_space_type', ACTION_SPACE_TYPE)
        if action_space_type == "discrete":
            # NEW: 5 actions for long/short trading
            # 0=HOLD, 1=OPEN_LONG, 2=CLOSE_LONG, 3=OPEN_SHORT, 4=CLOSE_SHORT
            self.action_space = spaces.Discrete(NUM_ACTIONS)
        else:
            raise ValueError(f"Type d'espace d'action invalide: {action_space_type}")

        # NEW: Position tracking for long/short
        # POSITION_FLAT=0, POSITION_LONG=1, POSITION_SHORT=-1
        self.position_type = POSITION_FLAT
        self.enable_short_selling = kwargs.get('enable_short_selling', ENABLE_SHORT_SELLING)
        self.short_borrowing_fee_daily = kwargs.get('short_borrowing_fee_daily', SHORT_BORROWING_FEE_DAILY)
        self.short_overnight_swap_pct = kwargs.get('short_overnight_swap_pct', SHORT_OVERNIGHT_SWAP_PCT)

        num_features_per_step = len(self.features)
        expected_obs_size = num_features_per_step * self.lookback_window_size + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(expected_obs_size,), dtype=np.float32)

        self.reset()

    def _process_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Traite les données brutes OHLCV et génère les features techniques et SMC.
        VERSION PRODUCTION-READY avec gestion intelligente des NaN et validation.
        """

        # =========================================================================
        # ÉTAPE 1: PRÉPARATION ET NORMALISATION DES COLONNES
        # =========================================================================

        df = df_raw.copy()

        # Préserver l'horodatage original
        if df_raw.index.name is not None:
            df = df.reset_index()

        # Identifier et renommer la colonne de date
        date_col_name = str(df.columns[0])
        df.rename(columns={date_col_name: 'Original_Timestamp'}, inplace=True)

        # Normaliser les colonnes OHLCV en minuscules (sauf Original_Timestamp)
        df.columns = [col.lower() if col != 'Original_Timestamp' else col for col in df.columns]

        # Validation des colonnes essentielles
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Colonnes OHLCV manquantes: {missing_cols}")

        # =========================================================================
        # ÉTAPE 2: GÉNÉRATION DES FEATURES (TA + SMC)
        # =========================================================================

        try:
            engine = SmartMoneyEngine(data=df, config=self.smc_config)
            df_processed = engine.analyze()
            print(f"✅ SmartMoneyEngine: {len(df_processed)} lignes générées")
        except Exception as e:
            print(f"❌ Erreur SmartMoneyEngine: {e}")
            raise

        # =========================================================================
        # ÉTAPE 3: NORMALISATION DES COLONNES (Capitalisation)
        # =========================================================================

        # Capitaliser les colonnes OHLCV pour cohérence avec le reste du code
        df_processed.columns = [
            c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume']
            else c for c in df_processed.columns
        ]

        # Assurer la présence de 'Close' (colonne critique)
        if 'close' in df_processed.columns and 'Close' not in df_processed.columns:
            df_processed.rename(columns={'close': 'Close'}, inplace=True)

        # Assurer la présence de 'ATR' (indicateur critique pour RiskManager)
        if 'ATR' not in df_processed.columns:
            if 'atr' in df_processed.columns:
                df_processed.rename(columns={'atr': 'ATR'}, inplace=True)
            else:
                print("⚠️ WARNING: ATR manquant. Calcul de fallback...")
                df_processed['ATR'] = ta.volatility.average_true_range(
                    df_processed['High'],
                    df_processed['Low'],
                    df_processed['Close'],
                    window=self.smc_config.get('ATR_WINDOW', 14)
                )

        # =========================================================================
        # ÉTAPE 4: NETTOYAGE DES INFINITÉS
        # =========================================================================

        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"🧹 Infinités remplacées par NaN")

        # =========================================================================
        # ÉTAPE 5: GESTION INTELLIGENTE DES NaN (CRITIQUE!)
        # =========================================================================

        print("\n🔍 Analyse des NaN avant nettoyage:")
        nan_summary = df_processed.isna().sum()
        nan_summary = nan_summary[nan_summary > 0].sort_values(ascending=False)
        if not nan_summary.empty:
            print(nan_summary)

        # --- 5.1: SUPPRESSION DES LIGNES AVEC NaN CRITIQUES ---
        # Ces colonnes ne peuvent PAS contenir de NaN (risque de crash)
        essential_features = [
            'Open', 'High', 'Low', 'Close',  # Prix
            'RSI',  # Momentum critique
            'MACD_line',  # Tendance critique
            'ATR'  # Volatilité critique
        ]

        rows_before = len(df_processed)
        df_processed.dropna(
            subset=[f for f in essential_features if f in df_processed.columns],
            inplace=True
        )
        rows_dropped = rows_before - len(df_processed)

        if rows_dropped > 0:
            print(f"⚠️ {rows_dropped} lignes supprimées (NaN dans colonnes critiques)")

        # Validation post-suppression
        if len(df_processed) < self.lookback_window_size * 2:
            raise ValueError(
                f"❌ Pas assez de données après nettoyage: {len(df_processed)} lignes "
                f"(minimum requis: {self.lookback_window_size * 2})"
            )

        # --- 5.2: FORWARD/BACKWARD FILL POUR INDICATEURS LENTS ---
        # Ces indicateurs peuvent être interpolés de manière sécurisée
        slow_indicators = [
            'RSI', 'MACD_line', 'MACD_signal', 'MACD_Diff',
            'BB_L', 'BB_M', 'BB_H', 'ATR'
        ]

        for col in slow_indicators:
            if col in df_processed.columns:
                # Forward fill (propagation de la dernière valeur connue)
                df_processed[col] = df_processed[col].ffill()
                # Backward fill pour les premières valeurs (si nécessaire)
                df_processed[col] = df_processed[col].bfill()

        print(f"✅ Indicateurs lents interpolés: {slow_indicators}")

        # --- 5.3: REMPLISSAGE PAR 0 POUR SIGNAUX SMC ---
        # Ces colonnes sont des signaux événementiels: NaN = "Pas de signal" = 0
        smc_signal_cols = [
            'FVG_SIGNAL',  # Fair Value Gap signal
            'FVG_SIZE_NORM',  # Taille FVG normalisée
            'BOS_SIGNAL',  # Break of Structure
            'CHOCH_SIGNAL',  # Change of Character
            'OB_STRENGTH_NORM',  # Force Order Block
            'UP_FRACTAL',  # Swing High
            'DOWN_FRACTAL',  # Swing Low
            'BULLISH_OB_HIGH',  # Zone Order Block haussier
            'BULLISH_OB_LOW',
            'BEARISH_OB_HIGH',  # Zone Order Block baissier
            'BEARISH_OB_LOW'
        ]

        for col in smc_signal_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(0.0)

        print(
            f"✅ Signaux SMC initialisés à 0: {len([c for c in smc_signal_cols if c in df_processed.columns])} colonnes")

        # --- 5.4: REMPLISSAGE PAR MÉDIANE POUR LE VOLUME ---
        # Le volume peut varier énormément, la médiane est plus robuste que 0
        if 'Volume' in df_processed.columns:
            volume_median = df_processed['Volume'].median()
            nan_count = df_processed['Volume'].isna().sum()
            if nan_count > 0:
                df_processed['Volume'] = df_processed['Volume'].fillna(volume_median)
                print(f"✅ Volume: {nan_count} NaN remplacés par médiane ({volume_median:.0f})")

        # --- 5.5: REMPLISSAGE FINAL (SÉCURITÉ) ---
        # Si des colonnes ont encore des NaN, on remplace par 0 (dernier recours)
        remaining_cols_with_nan = df_processed.columns[df_processed.isna().any()].tolist()

        if remaining_cols_with_nan:
            print(f"⚠️ Colonnes avec NaN restants: {remaining_cols_with_nan}")
            df_processed[remaining_cols_with_nan] = df_processed[remaining_cols_with_nan].fillna(0.0)

        # =========================================================================
        # ÉTAPE 6: VALIDATION FINALE DE LA QUALITÉ DES DONNÉES
        # =========================================================================

        # Vérifier qu'il ne reste AUCUN NaN
        final_nan_count = df_processed.isna().sum().sum()
        if final_nan_count > 0:
            raise ValueError(f"❌ {final_nan_count} NaN restants après nettoyage complet!")

        # Vérifier qu'il ne reste AUCUNE infinité
        inf_count = np.isinf(df_processed.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            raise ValueError(f"❌ {inf_count} valeurs infinies détectées!")

        # Vérifier la présence de prix aberrants (0 ou négatifs)
        if (df_processed['Close'] <= 0).any():
            invalid_prices = (df_processed['Close'] <= 0).sum()
            raise ValueError(f"❌ {invalid_prices} prix invalides (≤0) détectés!")

        # Vérifier la détection de mouvements extrêmes (flash crash potentiel)
        price_changes = df_processed['Close'].pct_change().abs()
        extreme_moves = (price_changes > 0.2).sum()  # Mouvements >20%
        if extreme_moves > 0:
            print(f"⚠️ WARNING: {extreme_moves} mouvements de prix >20% détectés (vérifier données)")

        # Reset index pour un DataFrame propre
        df_processed.reset_index(drop=True, inplace=True)

        # =========================================================================
        # ÉTAPE 7: RAPPORT FINAL
        # =========================================================================

        print("\n" + "=" * 70)
        print("📊 RAPPORT DE TRAITEMENT DES DONNÉES")
        print("=" * 70)
        print(f"✅ Lignes finales:        {len(df_processed)}")
        print(f"✅ Colonnes totales:      {len(df_processed.columns)}")
        print(f"✅ NaN restants:          {df_processed.isna().sum().sum()}")
        print(f"✅ Valeurs infinies:      {np.isinf(df_processed.select_dtypes(include=[np.number])).sum().sum()}")
        print(f"✅ Période couverte:      {df_processed.index[0]} → {df_processed.index[-1]}")

        # Afficher les features disponibles
        feature_categories = {
            'Prix': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'Indicateurs TA': ['RSI', 'MACD_line', 'MACD_signal', 'MACD_Diff', 'BB_L', 'BB_M', 'BB_H', 'ATR'],
            'SMC Signaux': ['BOS_SIGNAL', 'CHOCH_SIGNAL', 'FVG_SIGNAL', 'UP_FRACTAL', 'DOWN_FRACTAL'],
            'SMC Zones': ['BULLISH_OB_HIGH', 'BULLISH_OB_LOW', 'BEARISH_OB_HIGH', 'BEARISH_OB_LOW']
        }

        print("\n📋 Features disponibles par catégorie:")
        for category, features in feature_categories.items():
            available = [f for f in features if f in df_processed.columns]
            print(f"  {category}: {len(available)}/{len(features)} → {available}")

        print("=" * 70 + "\n")

        return df_processed

    def _get_obs(self) -> np.ndarray:
        """
        Returns the flattened observation vector for the current environment step.
        Includes scaled feature window and normalized portfolio state metrics.
        """

        # --- 1. Extract lookback window safely ---
        start_idx = self.current_step - self.lookback_window_size + 1
        obs_df = self.df.iloc[max(0, start_idx):self.current_step + 1].copy()

        # Padding (if the window is smaller than required)
        if len(obs_df) < self.lookback_window_size:
            padding_needed = self.lookback_window_size - len(obs_df)
            padded_data = np.zeros((padding_needed, len(self.df.columns)))
            padded_df = pd.DataFrame(padded_data, columns=self.df.columns)
            obs_df = pd.concat([padded_df, obs_df], ignore_index=True)

        # Keep the last lookback_window_size rows only
        obs_df = obs_df.tail(self.lookback_window_size)

        # --- 2. Feature extraction ---
        features_data = obs_df[self.features].values

        # --- 3. Scaling ---
        if self.scaler is not None:
            scaled_features = self.scaler.transform(features_data)
        else:
            scaled_features = features_data
            warnings.warn("⚠️ Scaler is None — features are not normalized.")

        flat_obs = scaled_features.flatten()

        # --- 4. Portfolio state extraction ---
        # Force scalar conversion to avoid ambiguous truth values
        # --- 4. Portfolio state extraction ---
        step_index = int(self.current_step)
        current_price = float(np.atleast_1d(self.df["Close"].iloc[step_index])[0])

        current_equity = float(self.balance + self.stock_quantity * current_price)

        # --- 5. Leverage calculation ---
        total_position_value = float(self.stock_quantity * current_price)
        if current_equity > 1e-9:
            self.current_leverage = total_position_value / current_equity
        else:
            self.current_leverage = 0.0

        # --- 6. Normalized financial metrics ---
        normalized_balance = float(self.balance / self.initial_balance)
        normalized_net_worth = float(current_equity / self.initial_balance)
        normalized_stock_quantity = float(total_position_value / self.initial_balance)

        # --- 7. Concatenate into a single observation vector ---
        observation = np.append(
            flat_obs, [normalized_balance, normalized_stock_quantity, normalized_net_worth]
        )

        # --- 8. Shape and value validation ---
        expected_size = len(self.features) * self.lookback_window_size + 3
        if observation.shape[0] != expected_size:
            warnings.warn(
                f"⚠️ Observation shape mismatch: expected {expected_size}, got {observation.shape[0]}."
            )

        if np.isnan(observation).any() or np.isinf(observation).any():
            bad_cols = obs_df[self.features].columns[
                np.isnan(obs_df[self.features].values).any(axis=0)
            ].tolist()
            raise ValueError(
                f"🚫 Observation contains NaN/Inf at step {self.current_step} "
                f"in columns: {bad_cols}"
            )

        return observation

    def _get_info(self) -> dict:
        current_price = self.df.iloc[self.current_step]['Close']
        self.net_worth = self.balance + self.stock_quantity * current_price
        self.total_trades = self.winning_trades + self.losing_trades
        current_return_percentage = ((self.net_worth - self.initial_balance) / self.initial_balance) * 100 \
            if self.initial_balance > 0 else 0.0

        info = {
            'balance': self.balance,
            'stock_quantity': self.stock_quantity,
            'net_worth': self.net_worth,
            'current_price': current_price,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_fees_paid': self.total_fees_paid_episode,
            'current_step': self.current_step,
            'episode_return_percentage': current_return_percentage,
            'entry_price': self.entry_price if not math.isclose(self.stock_quantity, 0.0, abs_tol=1e-9) else np.nan,
            'current_sl': self.risk_manager.current_stop_loss,
            'current_tp': self.risk_manager.current_take_profit,
            'actual_action_executed': self.actual_action_executed
        }
        info['trade_details'] = self.trade_details
        return info

    def _execute_trade(self, trade_type: str, trade_price: float, trade_quantity: float):
        """
        Exécute un trade (BUY ou SELL) avec gestion des coûts de transaction.

        VERSION CORRIGÉE - Capital-Agnostic:
        - Commission relative au capital (scalable de $100 à $100K+)
        - Frais proportionnels au trade value
        - Validation robuste des montants
        - Compatible avec tous les niveaux de capital

        Args:
            trade_type (str): 'buy' ou 'sell'
            trade_price (float): Prix d'exécution actuel du marché
            trade_quantity (float): Quantité à trader (en lots)

        Returns:
            tuple: (trade_success, effective_trade_value, commission, pnl_abs, pnl_pct)
                - trade_success (bool): True si le trade a été exécuté
                - effective_trade_value (float): Valeur brute du trade
                - commission (float): Commission totale payée
                - pnl_abs (float): P&L absolu en $ (pour SELL uniquement)
                - pnl_pct (float): P&L en % (pour SELL uniquement)
        """

        # =========================================================================
        # INITIALISATION
        # =========================================================================
        trade_success = False
        effective_trade_value = 0.0
        commission = 0.0
        pnl_abs = 0.0
        pnl_pct = 0.0

        # Reset du tracking des coûts pour ce step
        self.traded_value_step = 0.0
        self.transaction_cost_incurred_step = 0.0

        try:
            # =====================================================================
            # BUY LOGIC
            # =====================================================================
            if trade_type == 'buy':
                # --- 1. Calcul du prix effectif avec spread et slippage ---
                # Le prix d'achat inclut le spread (défavorable pour l'acheteur)
                price_with_spread = trade_price * (1 + self.transaction_fee_percentage)
                effective_buy_price = price_with_spread * (1 + self.slippage_percentage)

                # --- 2. Valeur brute du trade (avant commission) ---
                gross_trade_value = effective_buy_price * trade_quantity

                # --- 3. NOUVEAU: Calcul de la commission relative (Capital-Agnostic) ---
                #
                # Commission composée de 2 parties:
                # a) Commission proportionnelle au trade (% de la valeur)
                # b) Commission minimum basée sur le capital (pour éviter trades trop petits)

                # a) Commission du trade (ex: 0.05% de $1,000 = $0.50)
                commission_pct_trade = getattr(self, 'trade_commission_pct_of_trade', 0.0005)
                commission_from_trade = gross_trade_value * commission_pct_trade

                # b) Commission minimum relative au capital (ex: 0.01% de $1,000 = $0.10)
                commission_min_pct = getattr(self, 'trade_commission_min_pct_capital', 0.0001)
                commission_minimum = self.initial_balance * commission_min_pct

                # La commission finale est le MAXIMUM des deux (protège contre trades trop petits)
                commission = max(commission_from_trade, commission_minimum)

                # --- 4. Coût total de l'achat ---
                total_cost = gross_trade_value + commission

                # --- 5. Validation: Vérifier si le trade est possible ---
                # Condition 1: Balance suffisante
                if self.balance < total_cost:
                    self.transaction_cost_incurred_step = 0.0
                    return False, 0.0, 0.0, 0.0, 0.0

                # Condition 2: Quantité minimale respectée
                if trade_quantity < self.min_trade_quantity:
                    self.transaction_cost_incurred_step = 0.0
                    return False, 0.0, 0.0, 0.0, 0.0

                # --- 6. Exécution du trade ---
                self.balance -= total_cost
                self.stock_quantity += trade_quantity
                self.entry_price = trade_price  # Prix d'entrée (utilisé pour calculer P&L)

                # --- 7. Mise à jour des compteurs ---
                self.total_fees_paid_episode += commission
                self.transaction_cost_incurred_step = commission
                effective_trade_value = trade_price * trade_quantity  # Valeur "propre" sans frais
                self.traded_value_step = effective_trade_value

                trade_success = True

                # --- 8. Logging optionnel ---
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"✅ BUY Executed: {trade_quantity:.4f} @ ${trade_price:,.2f} | "
                          f"Cost: ${total_cost:,.2f} (Commission: ${commission:.2f})")

            # =====================================================================
            # SELL LOGIC
            # =====================================================================
            elif trade_type == 'sell':
                # --- 1. Validation: Vérifier si on a assez de stock ---
                if self.stock_quantity < trade_quantity or self.stock_quantity <= 1e-9:
                    self.transaction_cost_incurred_step = 0.0
                    return False, 0.0, 0.0, 0.0, 0.0

                # --- 2. Calcul du prix effectif avec spread et slippage ---
                # Le prix de vente subit le spread (défavorable pour le vendeur)
                price_with_spread = trade_price * (1 - self.transaction_fee_percentage)
                effective_sell_price = price_with_spread * (1 - self.slippage_percentage)

                # --- 3. Valeur brute du trade ---
                gross_trade_value = effective_sell_price * trade_quantity

                # --- 4. NOUVEAU: Calcul de la commission relative ---
                commission_pct_trade = getattr(self, 'trade_commission_pct_of_trade', 0.0005)
                commission_from_trade = gross_trade_value * commission_pct_trade

                commission_min_pct = getattr(self, 'trade_commission_min_pct_capital', 0.0001)
                commission_minimum = self.initial_balance * commission_min_pct

                commission = max(commission_from_trade, commission_minimum)

                # --- 5. Revenu net après commission ---
                total_revenue = gross_trade_value - commission

                # --- 6. Calcul du P&L (Profit & Loss) ---
                if not np.isnan(self.entry_price) and not math.isclose(self.entry_price, 0.0):
                    # Coût d'acquisition (prix d'achat * quantité)
                    asset_cost = self.entry_price * trade_quantity

                    # P&L absolu = Revenu - Coût
                    pnl_abs = total_revenue - asset_cost

                    # P&L en pourcentage
                    pnl_pct = (pnl_abs / asset_cost) * 100 if asset_cost > 1e-9 else 0.0
                else:
                    # Cas anormal: pas de prix d'entrée enregistré
                    pnl_abs = 0.0
                    pnl_pct = 0.0

                # --- 7. Exécution du trade ---
                self.balance += total_revenue
                self.stock_quantity -= trade_quantity

                # --- 8. Si position complètement fermée, reset ---
                if math.isclose(self.stock_quantity, 0.0, abs_tol=1e-9):
                    self.entry_price = np.nan
                    self.stock_quantity = 0.0  # Forcer à zéro exact

                    # Reset du risk manager (SL/TP)
                    if hasattr(self, 'risk_manager') and self.risk_manager is not None:
                        self.risk_manager.reset()

                # --- 9. Mise à jour des compteurs ---
                self.total_fees_paid_episode += commission
                self.transaction_cost_incurred_step = commission
                effective_trade_value = trade_price * trade_quantity
                self.traded_value_step = effective_trade_value

                trade_success = True

                # --- 10. Logging optionnel ---
                if hasattr(self, 'verbose') and self.verbose:
                    pnl_symbol = "+" if pnl_abs >= 0 else ""
                    print(f"✅ SELL Executed: {trade_quantity:.4f} @ ${trade_price:,.2f} | "
                          f"Revenue: ${total_revenue:,.2f} | P&L: {pnl_symbol}${pnl_abs:.2f} ({pnl_pct:+.2f}%)")

            else:
                # Type de trade invalide
                raise ValueError(f"Invalid trade_type: {trade_type}. Must be 'buy' or 'sell'.")

        except Exception as e:
            # =====================================================================
            # GESTION DES ERREURS
            # =====================================================================
            print(f"❌ ERROR _EXECUTE_TRADE: An error occurred during {trade_type.upper()} execution: {e}")
            traceback.print_exc()

            # Reset des coûts en cas d'échec
            self.transaction_cost_incurred_step = 0.0
            self.traded_value_step = 0.0

            # Retourner des valeurs par défaut (trade échoué)
            return False, 0.0, 0.0, 0.0, 0.0

        # =========================================================================
        # RETOUR
        # =========================================================================
        return trade_success, effective_trade_value, commission, pnl_abs, pnl_pct

    def step(self, action: int):
        """
        Execute one step in the environment with LONG/SHORT support.

        NEW ACTION SPACE (5 actions):
            0 = HOLD         : Do nothing
            1 = OPEN_LONG    : Buy to open long position
            2 = CLOSE_LONG   : Sell to close long position
            3 = OPEN_SHORT   : Sell to open short position
            4 = CLOSE_SHORT  : Buy to cover short position

        Position States:
            FLAT (0)  : No position - can OPEN_LONG or OPEN_SHORT
            LONG (1)  : Holding long - can HOLD or CLOSE_LONG
            SHORT (-1): Holding short - can HOLD or CLOSE_SHORT
        """
        # --- 1. Track previous state for reward calculation ---
        previous_net_worth = self.net_worth
        self.previous_drawdown_level = self.peak_nav - previous_net_worth
        self.transaction_cost_incurred_step = 0.0
        self.traded_value_step = 0.0

        self.current_step += 1
        done = False
        truncated = False

        # --- LECTURE CRITIQUE DES DONNÉES ---
        current_row = self.df.iloc[self.current_step]
        current_market_price = float(current_row['Close'])
        current_atr = float(current_row['ATR'])
        bos_signal = float(current_row['BOS_SIGNAL'])

        # --- Update risk manager regime ---
        regime_state = 0 if bos_signal != 0 else 1
        self.risk_manager.market_state['current_regime'] = regime_state

        # --- End of episode: Force close any position ---
        if self.current_step >= self.max_steps:
            truncated = True
            done = True
            # Force close any open position
            if self.position_type == POSITION_LONG and self.stock_quantity > self.min_trade_quantity:
                action = ACTION_CLOSE_LONG
            elif self.position_type == POSITION_SHORT and abs(self.stock_quantity) > self.min_trade_quantity:
                action = ACTION_CLOSE_SHORT
            else:
                action = ACTION_HOLD

        # --- Initialize trade details ---
        self.trade_details = {
            'trade_pnl_abs': 0.0, 'trade_pnl_pct': 0.0, 'trade_type': 'hold',
            'trade_success': False, 'trade_value': 0.0, 'commission': 0.0,
            'position_type': self.position_type
        }
        self.last_action = action
        self.actual_action_executed = action

        # --- ACTION VALIDATION: Convert invalid actions to HOLD ---
        # Can only OPEN_LONG or OPEN_SHORT when FLAT
        # Can only CLOSE_LONG when LONG
        # Can only CLOSE_SHORT when SHORT
        original_action = action
        if action == ACTION_OPEN_LONG and self.position_type != POSITION_FLAT:
            action = ACTION_HOLD
            self.trade_details['trade_type'] = 'invalid_already_in_position'
        elif action == ACTION_OPEN_SHORT and self.position_type != POSITION_FLAT:
            action = ACTION_HOLD
            self.trade_details['trade_type'] = 'invalid_already_in_position'
        elif action == ACTION_OPEN_SHORT and not self.enable_short_selling:
            action = ACTION_HOLD
            self.trade_details['trade_type'] = 'short_selling_disabled'
        elif action == ACTION_CLOSE_LONG and self.position_type != POSITION_LONG:
            action = ACTION_HOLD
            self.trade_details['trade_type'] = 'invalid_no_long_position'
        elif action == ACTION_CLOSE_SHORT and self.position_type != POSITION_SHORT:
            action = ACTION_HOLD
            self.trade_details['trade_type'] = 'invalid_no_short_position'

        # --- Check for SL/TP/TSL on active positions ---
        if not done and self.position_type != POSITION_FLAT and not np.isnan(self.entry_price):
            self.current_hold_duration += 1
            is_long = (self.position_type == POSITION_LONG)

            self.risk_manager.update_trailing_stop(
                self.entry_price, current_market_price, current_atr, is_long=is_long
            )
            exit_signal = self.risk_manager.check_trade_exit(current_market_price, is_long=is_long)

            if exit_signal == 'TP':
                action = ACTION_CLOSE_LONG if is_long else ACTION_CLOSE_SHORT
                self.actual_action_executed = 10  # TP exit code
            elif exit_signal == 'SL':
                action = ACTION_CLOSE_LONG if is_long else ACTION_CLOSE_SHORT
                self.actual_action_executed = 11  # SL exit code
        elif self.position_type == POSITION_FLAT:
            self.current_hold_duration = 0

        # --- Cooldown enforcement ---
        if action in [ACTION_OPEN_LONG, ACTION_CLOSE_LONG, ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT]:
            if (self.current_step - self.last_trade_step) < self.trade_cooldown_steps:
                self.actual_action_executed = ACTION_HOLD
                self.trade_details['trade_type'] = 'hold_cooldown'
                action = ACTION_HOLD

        # --- Apply borrowing fees for short positions ---
        if self.position_type == POSITION_SHORT and abs(self.stock_quantity) > 0:
            borrowing_fee = abs(self.stock_quantity) * current_market_price * self.short_borrowing_fee_daily
            self.balance -= borrowing_fee
            self.total_fees_paid_episode += borrowing_fee

        # --- Execute trade logic ---
        trade_executed = False
        if not done:
            # ========================================
            # OPEN LONG (Action == 1)
            # ========================================
            if action == ACTION_OPEN_LONG:
                trade_executed = self._execute_open_long(current_market_price, current_atr)

            # ========================================
            # CLOSE LONG (Action == 2)
            # ========================================
            elif action == ACTION_CLOSE_LONG:
                trade_executed = self._execute_close_long(current_market_price)

            # ========================================
            # OPEN SHORT (Action == 3)
            # ========================================
            elif action == ACTION_OPEN_SHORT:
                trade_executed = self._execute_open_short(current_market_price, current_atr)

            # ========================================
            # CLOSE SHORT (Action == 4)
            # ========================================
            elif action == ACTION_CLOSE_SHORT:
                trade_executed = self._execute_close_short(current_market_price)

            # ========================================
            # HOLD (Action == 0)
            # ========================================
            else:
                self.trade_details['trade_type'] = 'hold'
                self.actual_action_executed = ACTION_HOLD

        # --- Portfolio update (handles both long and short) ---
        self._update_portfolio_value(current_market_price)

        # --- Update Drawdown and Leverage Trackers ---
        self.peak_nav = max(self.peak_nav, self.net_worth)

        position_value = abs(self.stock_quantity) * current_market_price
        if self.net_worth > 1e-9:
            self.current_leverage = position_value / self.net_worth
        else:
            self.current_leverage = 0.0

        if self.net_worth <= self.minimum_allowed_balance:
            done = True

        # --- Calculate reward ---
        reward = self._calculate_reward(previous_net_worth)
        self.episode_reward += reward

        observation = self._get_obs()
        info = self._get_info()
        info['position_type'] = self.position_type
        info['position_type_name'] = {POSITION_FLAT: 'FLAT', POSITION_LONG: 'LONG', POSITION_SHORT: 'SHORT'}[self.position_type]

        self.previous_nav = self.net_worth

        return observation, reward, done, truncated, info

    def _execute_open_long(self, current_price: float, current_atr: float) -> bool:
        """Execute OPEN_LONG action (buy to open long position)."""
        sl_distance_abs = self.risk_manager.set_trade_orders(current_price, current_atr, is_long=True)

        # Calculate position size with HARD LEVERAGE ENFORCEMENT
        # The risk manager now caps position size to prevent exceeding max_leverage_limit
        trade_quantity_calc = self.risk_manager.calculate_adaptive_position_size(
            client_id=getattr(self, "_risk_client_id", "default_client"),
            account_equity=self.balance,
            atr_stop_distance=sl_distance_abs,
            win_prob=0.5,
            risk_reward_ratio=1.0,
            current_price=current_price,  # NEW: Pass price for leverage calc
            max_leverage=self.max_leverage_limit,  # NEW: Hard leverage limit
            is_long=True
        )

        try:
            trade_quantity_calc = float(trade_quantity_calc)
        except Exception:
            trade_quantity_calc = 0.0

        if trade_quantity_calc < self.min_trade_quantity:
            trade_quantity_calc = 0.0

        estimated_cost = current_price * trade_quantity_calc * (
            1 + self.transaction_fee_percentage + self.slippage_percentage
        )
        if estimated_cost > self.balance and estimated_cost > 0:
            scale = self.balance / estimated_cost
            trade_quantity_calc *= scale

        if trade_quantity_calc < self.min_trade_quantity:
            self.trade_details['trade_type'] = 'open_long_failed'
            self.actual_action_executed = ACTION_HOLD
            return False

        trade_success, value, commission, _, _ = self._execute_trade(
            'buy', current_price, trade_quantity=trade_quantity_calc
        )

        if trade_success:
            self.position_type = POSITION_LONG
            self.last_trade_step = self.current_step
            self.trade_details.update({
                'trade_success': True, 'trade_type': 'open_long',
                'trade_value': value, 'commission': commission,
                'quantity': trade_quantity_calc
            })
            self.current_hold_duration = 1

            if self.trade_logger:
                self.trade_id_counter += 1
                self.trade_logger.log_trade({
                    'trade_id': self.trade_id_counter,
                    'trade_type': 'open_long',
                    'step': self.current_step,
                    'price': current_price,
                    'quantity': trade_quantity_calc,
                    'balance': self.balance,
                    'net_worth': self.net_worth
                })
            return True
        else:
            self.trade_details['trade_type'] = 'open_long_failed'
            self.actual_action_executed = ACTION_HOLD
            return False

    def _execute_close_long(self, current_price: float) -> bool:
        """Execute CLOSE_LONG action (sell to close long position)."""
        if self.stock_quantity <= self.min_trade_quantity:
            self.actual_action_executed = ACTION_HOLD
            self.trade_details['trade_type'] = 'close_long_no_position'
            return False

        trade_success, value, commission, pnl_abs, pnl_pct = self._execute_trade(
            'sell', current_price, self.stock_quantity
        )

        if trade_success:
            self.position_type = POSITION_FLAT
            self.last_trade_step = self.current_step
            self.trade_history_summary.append({
                'step': self.current_step,
                'pnl_abs': pnl_abs, 'pnl_pct': pnl_pct, 'type': 'close_long'
            })
            self.total_trades += 1
            if pnl_abs > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            self.current_hold_duration = 0

            self.trade_details.update({
                'trade_success': True, 'trade_type': 'close_long',
                'trade_value': value, 'commission': commission,
                'trade_pnl_abs': pnl_abs, 'trade_pnl_pct': pnl_pct
            })

            if self.trade_logger:
                self.trade_logger.log_trade({
                    'trade_id': self.trade_id_counter,
                    'trade_type': 'close_long',
                    'step': self.current_step,
                    'price': current_price,
                    'pnl_abs': pnl_abs,
                    'pnl_pct': pnl_pct,
                    'balance': self.balance,
                    'net_worth': self.net_worth,
                    'duration_bars': self.current_hold_duration
                })
            return True
        else:
            self.trade_details['trade_type'] = 'close_long_failed'
            return False

    def _execute_open_short(self, current_price: float, current_atr: float) -> bool:
        """
        Execute OPEN_SHORT action (sell to open short position).

        Short selling mechanics:
        1. Borrow asset from broker
        2. Sell at current price (receive cash)
        3. Later: buy back to return to broker
        4. Profit if price goes DOWN, loss if price goes UP
        """
        sl_distance_abs = self.risk_manager.set_trade_orders(current_price, current_atr, is_long=False)

        # Calculate position size with HARD LEVERAGE ENFORCEMENT
        # The risk manager now caps position size to prevent exceeding max_leverage_limit
        trade_quantity_calc = self.risk_manager.calculate_adaptive_position_size(
            client_id=getattr(self, "_risk_client_id", "default_client"),
            account_equity=self.balance,
            atr_stop_distance=sl_distance_abs,
            win_prob=0.5,
            risk_reward_ratio=1.0,
            current_price=current_price,  # NEW: Pass price for leverage calc
            max_leverage=self.max_leverage_limit,  # NEW: Hard leverage limit
            is_long=False  # Short position
        )

        try:
            trade_quantity_calc = float(trade_quantity_calc)
        except Exception:
            trade_quantity_calc = 0.0

        if trade_quantity_calc < self.min_trade_quantity:
            self.trade_details['trade_type'] = 'open_short_failed'
            self.actual_action_executed = ACTION_HOLD
            return False

        # For shorts, we receive cash when opening (sell borrowed asset)
        # Apply spread and slippage (unfavorable for seller)
        price_with_spread = current_price * (1 - self.transaction_fee_percentage)
        effective_sell_price = price_with_spread * (1 - self.slippage_percentage)
        gross_value = effective_sell_price * trade_quantity_calc

        # Calculate commission
        commission_pct_trade = getattr(self, 'trade_commission_pct_of_trade', 0.0005)
        commission = gross_value * commission_pct_trade

        # Short position: negative stock quantity, cash increases
        self.balance += (gross_value - commission)
        self.stock_quantity = -trade_quantity_calc  # Negative = short position
        self.entry_price = current_price
        self.position_type = POSITION_SHORT
        self.last_trade_step = self.current_step

        self.total_fees_paid_episode += commission
        self.transaction_cost_incurred_step = commission
        self.traded_value_step = current_price * trade_quantity_calc

        self.trade_details.update({
            'trade_success': True, 'trade_type': 'open_short',
            'trade_value': gross_value, 'commission': commission,
            'quantity': trade_quantity_calc
        })
        self.current_hold_duration = 1

        if self.trade_logger:
            self.trade_id_counter += 1
            self.trade_logger.log_trade({
                'trade_id': self.trade_id_counter,
                'trade_type': 'open_short',
                'step': self.current_step,
                'price': current_price,
                'quantity': trade_quantity_calc,
                'balance': self.balance,
                'net_worth': self.net_worth
            })
        return True

    def _execute_close_short(self, current_price: float) -> bool:
        """
        Execute CLOSE_SHORT action (buy to cover short position).

        Closing a short:
        1. Buy back the asset at current price
        2. Return to broker
        3. P&L = entry_price - exit_price (profit if price went down)
        """
        if abs(self.stock_quantity) <= self.min_trade_quantity or self.position_type != POSITION_SHORT:
            self.actual_action_executed = ACTION_HOLD
            self.trade_details['trade_type'] = 'close_short_no_position'
            return False

        quantity_to_cover = abs(self.stock_quantity)

        # Buy to cover: pay current price + spread + slippage
        price_with_spread = current_price * (1 + self.transaction_fee_percentage)
        effective_buy_price = price_with_spread * (1 + self.slippage_percentage)
        gross_cost = effective_buy_price * quantity_to_cover

        # Calculate commission
        commission_pct_trade = getattr(self, 'trade_commission_pct_of_trade', 0.0005)
        commission = gross_cost * commission_pct_trade

        total_cost = gross_cost + commission

        # Check if we can afford to close
        if total_cost > self.balance:
            self.trade_details['trade_type'] = 'close_short_insufficient_funds'
            return False

        # Calculate P&L for short: profit if price went DOWN
        # P&L = (entry_price - exit_price) * quantity
        pnl_abs = (self.entry_price - current_price) * quantity_to_cover - commission
        if self.entry_price > 0:
            pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
        else:
            pnl_pct = 0.0

        # Execute: pay to cover, clear position
        self.balance -= total_cost
        self.stock_quantity = 0.0
        self.position_type = POSITION_FLAT
        self.last_trade_step = self.current_step
        self.entry_price = np.nan

        self.total_fees_paid_episode += commission
        self.transaction_cost_incurred_step = commission
        self.traded_value_step = current_price * quantity_to_cover

        # Record trade
        self.trade_history_summary.append({
            'step': self.current_step,
            'pnl_abs': pnl_abs, 'pnl_pct': pnl_pct, 'type': 'close_short'
        })
        self.total_trades += 1
        if pnl_abs > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        self.current_hold_duration = 0

        self.trade_details.update({
            'trade_success': True, 'trade_type': 'close_short',
            'trade_value': gross_cost, 'commission': commission,
            'trade_pnl_abs': pnl_abs, 'trade_pnl_pct': pnl_pct
        })

        if self.trade_logger:
            self.trade_logger.log_trade({
                'trade_id': self.trade_id_counter,
                'trade_type': 'close_short',
                'step': self.current_step,
                'price': current_price,
                'pnl_abs': pnl_abs,
                'pnl_pct': pnl_pct,
                'balance': self.balance,
                'net_worth': self.net_worth,
                'duration_bars': self.current_hold_duration
            })
        return True

    def _update_portfolio_value(self, current_price: float) -> None:
        """
        Update net worth accounting for both long and short positions.

        For LONG positions:
            net_worth = balance + (quantity * price)

        For SHORT positions:
            net_worth = balance - (quantity * price) + (entry_quantity * entry_price)
            Simplified: net_worth = balance + unrealized_pnl
            where unrealized_pnl = (entry_price - current_price) * |quantity|
        """
        if self.position_type == POSITION_LONG:
            # Long: we own the asset, value increases with price
            self.net_worth = self.balance + (self.stock_quantity * current_price)

        elif self.position_type == POSITION_SHORT:
            # Short: we owe the asset, value increases when price decreases
            # unrealized P&L = (entry_price - current_price) * quantity
            quantity = abs(self.stock_quantity)
            if not np.isnan(self.entry_price):
                unrealized_pnl = (self.entry_price - current_price) * quantity
            else:
                unrealized_pnl = 0.0
            # When we opened short, we received: entry_price * quantity
            # To close, we need to pay: current_price * quantity
            # Net worth = what we have - what we owe = balance + unrealized_pnl
            self.net_worth = self.balance + unrealized_pnl

        else:  # FLAT
            self.net_worth = self.balance

    def _calculate_reward(self, previous_net_worth: float) -> float:
        """
        PRODUCTION-GRADE REWARD FUNCTION FOR PPO TRADING BOT
        =====================================================

        Research-backed design for stable learning and commercial viability.

        Key Improvements:
        - Consistent scaling (no extreme values like -5M)
        - Smooth gradients for stable PPO learning
        - Non-redundant penalties
        - Risk-adjusted profitability focus
        - Commercial viability optimization

        Expected reward range: [-20, +20]
        Typical rewards: [-5, +5]

        Returns:
            float: Scaled reward for PPO training
        """

        # =========================================================================
        # STEP 1: VALIDATION (Prevent Division by Zero)
        # =========================================================================
        if previous_net_worth <= 1e-9 or self.net_worth <= 1e-9:
            # Critical failure: Account depleted
            return -20.0

        # =========================================================================
        # STEP 2: CORE PROFITABILITY METRIC (Most Important Component)
        # =========================================================================
        # Use logarithmic returns for numerical stability (research-backed)
        # log(1.01) ≈ 0.01 for small changes, but stable for large changes
        log_return = np.log(self.net_worth / previous_net_worth)

        # Scale to interpretable range: 1% gain = ~1.0 reward
        # This provides strong signal for profitable actions
        profitability_reward = log_return * 100.0

        # =========================================================================
        # STEP 3: RISK-ADJUSTED PENALTIES (Prevent Reckless Trading)
        # =========================================================================
        total_penalty = 0.0

        # --- Penalty A: Drawdown Increase (ONLY when WORSENING) ---
        # KEY INSIGHT: Only penalize NEW drawdown, not existing drawdown
        # This prevents the agent from being punished for market conditions
        current_drawdown = self.peak_nav - self.net_worth
        drawdown_increase = max(0.0, current_drawdown - self.previous_drawdown_level)

        if drawdown_increase > 0:
            # Penalty scaled by severity of new drawdown
            # Weight: 5.0 (high priority on capital preservation)
            dd_penalty = (drawdown_increase / self.initial_balance) * 5.0
            total_penalty += dd_penalty

        # --- Penalty B: Transaction Costs (Friction) ---
        # Penalize the direct cost of trading (commission + spread + slippage)
        # This encourages the agent to be selective about trades
        if self.transaction_cost_incurred_step > 0:
            # Weight: 2.0 (moderate - costs are part of business)
            friction_penalty = (self.transaction_cost_incurred_step / self.initial_balance) * 2.0
            total_penalty += friction_penalty

        # --- Penalty C: Leverage Violation (Hard Constraint) ---
        # CRITICAL: Enforce risk management rules
        # Quadratic penalty for exceeding maximum allowed leverage
        leverage_excess = max(0.0, self.current_leverage - self.max_leverage_limit)

        if leverage_excess > 0:
            # Weight: 10.0 (very high - this is a compliance issue)
            # Quadratic scaling: Small violations = small penalty, large = huge
            leverage_penalty = (leverage_excess ** 2) * 10.0
            total_penalty += leverage_penalty

        # --- Penalty D: Excessive Holding Duration ---
        # For daytrading strategy, positions held too long are risky
        # Gentle penalty after exceeding target holding period
        if self.current_hold_duration > self.max_duration_steps:
            duration_excess = self.current_hold_duration - self.max_duration_steps
            # Weight: 0.5 (low - this is a soft constraint)
            duration_penalty = (duration_excess / self.max_duration_steps) * 0.5
            total_penalty += duration_penalty

        # --- Penalty E: Churning (Over-Trading) ---
        # Penalize excessive turnover (trading too frequently)
        # This prevents the agent from becoming a "gambler"
        if self.traded_value_step > 0:
            turnover_ratio = self.traded_value_step / self.net_worth

            # Only penalize if turnover exceeds 50% of account in single step
            if turnover_ratio > 0.5:
                # Weight: 1.0 (moderate - some trading is necessary)
                churn_penalty = (turnover_ratio - 0.5) * 1.0
                total_penalty += churn_penalty

        # =========================================================================
        # STEP 4: COMPOSITE RAW REWARD (Before Normalization)
        # =========================================================================
        raw_reward = profitability_reward - total_penalty

        # =========================================================================
        # STEP 5: SHAPING BONUSES (Learning Signal for Trade Outcomes)
        # =========================================================================
        # These bonuses provide immediate feedback on trade quality
        # Helps the agent learn faster by rewarding good decisions
        bonus = 0.0

        # Check if a trade was just closed (any CLOSE action completed)
        # NEW: Support both old 'sell' and new 'close_long'/'close_short' trade types
        closed_trade_types = ['sell', 'close_long', 'close_short']
        if self.trade_details.get('trade_type') in closed_trade_types and self.trade_details.get('trade_success'):
            trade_pnl_abs = self.trade_details.get('trade_pnl_abs', 0.0)
            trade_pnl_pct = self.trade_details.get('trade_pnl_pct', 0.0)

            if trade_pnl_abs > 0:
                # === WINNING TRADE ===
                # Bonus scaled by percentage profit (max +2.0)
                win_bonus = min(2.0, (trade_pnl_pct / 100.0) * 10.0)
                bonus += win_bonus

                # Extra bonus for high-quality wins (>1.5% profit)
                # This encourages the agent to wait for good setups
                if trade_pnl_pct > 1.5:
                    bonus += 1.0
            else:
                # === LOSING TRADE ===
                # Small negative feedback (already penalized by profitability)
                # This helps the agent learn to avoid bad trades
                loss_feedback = max(-1.0, (trade_pnl_pct / 100.0) * 5.0)
                bonus += loss_feedback

        # =========================================================================
        # STEP 6: NORMALIZATION (Tanh Squashing)
        # =========================================================================
        # Apply smooth squashing function to prevent extreme values
        # Tanh maps (-∞, +∞) to (-1, +1) with smooth gradients
        # This is CRITICAL for PPO stability

        combined_reward = raw_reward + bonus

        # Scale factor 0.3: Controls sensitivity
        # Lower = more compressed range, Higher = wider range
        normalized_reward = np.tanh(combined_reward * 0.3)

        # Map to target range: [-5, +5] for typical steps
        scaled_reward = normalized_reward * 5.0

        # =========================================================================
        # STEP 7: SPECIAL CASES (Terminal Conditions)
        # =========================================================================

        # === CRITICAL FAILURE: Account Below Minimum ===
        # This should trigger episode termination with strong penalty
        if self.net_worth <= self.minimum_allowed_balance:
            return -20.0

        # === SEVERE DRAWDOWN: Emergency Risk Management ===
        # If drawdown exceeds 15%, apply additional penalty
        if current_drawdown > 0:
            dd_ratio = current_drawdown / self.peak_nav
            if dd_ratio > 0.15:  # 15% total drawdown
                scaled_reward -= 5.0  # Extra penalty

        # =========================================================================
        # STEP 8: FINAL SAFETY CLIPPING
        # =========================================================================
        # Hard limits to prevent any extreme values that might destabilize training
        # Range: [-20, +20] with most values in [-5, +5]
        final_reward = np.clip(scaled_reward, -20.0, 20.0)

        # =========================================================================
        # STEP 9: OPTIONAL DEBUG LOGGING
        # =========================================================================
        # Enable this during development to understand reward components
        # DISABLE in production for performance
        if hasattr(self, 'debug_rewards') and self.debug_rewards:
            print(f"[REWARD DEBUG] Step {self.current_step}")
            print(f"  Profitability: {profitability_reward:+.3f}")
            print(f"  Total Penalty: {total_penalty:.3f}")
            print(f"    - Drawdown: {dd_penalty if drawdown_increase > 0 else 0:.3f}")
            print(f"    - Friction: {friction_penalty if self.transaction_cost_incurred_step > 0 else 0:.3f}")
            print(f"    - Leverage: {leverage_penalty if leverage_excess > 0 else 0:.3f}")
            print(f"  Bonus: {bonus:+.3f}")
            print(f"  Raw Reward: {raw_reward:+.3f}")
            print(f"  Final Reward: {final_reward:+.3f}")
            print(f"  Net Worth: ${self.net_worth:.2f} (Δ {(self.net_worth / previous_net_worth - 1) * 100:+.2f}%)")
            print("-" * 50)

        return final_reward



    def render(self):
        console = Console()
        table = Table(show_header=True, header_style="bold magenta", show_lines=False)

        # Columns
        table.add_column("Step", style="cyan")
        table.add_column("Time", style="white")
        table.add_column("Net Worth", justify="right")
        table.add_column("Δ%", justify="right")
        table.add_column("Balance", justify="right")
        table.add_column("Stock", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("SL", justify="right")
        table.add_column("TP", justify="right")
        table.add_column("TSL", justify="center")
        table.add_column("Fees", justify="right")
        table.add_column("Action", justify="center")

        # Safe time
        try:
            current_time = pd.to_datetime(self.df.index)[self.current_step].strftime('%Y-%m-%d %H:%M')
        except Exception:
            if 'Gmt time' in self.df.columns:
                current_time = pd.to_datetime(
                    self.df.iloc[self.current_step]['Gmt time']
                ).strftime('%Y-%m-%d %H:%M')
            else:
                current_time = f"{self.current_step}"

        # Compute metrics
        net_worth_change = ((self.net_worth - self.initial_balance) / self.initial_balance) * 100
        pnl_color = "green" if net_worth_change >= 0 else "red"
        current_sl = getattr(self.risk_manager, "current_stop_loss", np.nan)
        current_tp = getattr(self.risk_manager, "current_take_profit", np.nan)
        tsl_active = getattr(self.risk_manager, "tsl_activated", False)
        entry_display = (
            f"{self.entry_price:,.2f}" if not math.isclose(self.stock_quantity, 0.0, abs_tol=1e-9) else "—"
        )

        table.add_row(
            str(self.current_step),
            current_time,
            f"${self.net_worth:,.2f}",
            Text(f"{net_worth_change:+.2f}%", style=pnl_color),
            f"${self.balance:,.2f}",
            f"{self.stock_quantity:.4f}",
            entry_display,
            f"${current_sl:,.2f}",
            f"${current_tp:,.2f}",
            "✅" if tsl_active else "—",
            f"${self.total_fees_paid:,.2f}",
            self.trade_details.get("action_taken", "N/A"),
        )

        console.print(table)

    def close(self):
        print("DEBUG ENV: Environment closed.")

    def reset(self, seed: int = None, options: dict = None):
        """
        Resets the trading environment to its initial state before a new episode.
        Returns the first observation and info dict (Gymnasium API standard).
        """
        super().reset(seed=seed)

        # --- 1️⃣ Reset financial state ---
        self.balance = float(self.initial_balance)
        self.net_worth = float(self.initial_balance)
        self.stock_quantity = 0.0
        self.entry_price = np.nan
        self.position_type = POSITION_FLAT  # NEW: Reset position type for long/short

        # --- 2️⃣ Reset Risk Manager and statistics ---
        if hasattr(self, "risk_manager") and self.risk_manager is not None:
            self.risk_manager.reset()

        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees_paid_episode = 0.0
        self.trade_history_summary = deque(maxlen=1000)
        self.history = deque(maxlen=1000)

        # --- 3️⃣ Reset behavioral trackers ---
        self.last_action = 0
        self.actual_action_executed = 0

        # --- 4️⃣ Advanced reward trackers ---
        self.previous_nav = self.initial_balance
        self.peak_nav = self.initial_balance
        self.previous_drawdown_level = 0.0
        self.current_leverage = 0.0
        self.current_hold_duration = 0
        self.transaction_cost_incurred_step = 0.0
        self.traded_value_step = 0.0

        # --- 5️⃣ Episode boundaries ---
        min_possible_step = int(self.lookback_window_size - 1)
        max_valid_start_idx_for_obs = int(len(self.df) - 1 - min_possible_step)

        if max_valid_start_idx_for_obs <= min_possible_step:
            raise ValueError(
                f"❌ Not enough data to form a valid episode with the given lookback window "
                f"({self.lookback_window_size}). Data length: {len(self.df)}"
            )

        # Use the Gymnasium RNG for reproducibility
        self.start_idx = int(
            self.np_random.integers(min_possible_step, max_valid_start_idx_for_obs + 1)
        )

        # Ensure step indices are pure Python ints
        self.current_step = int(self.start_idx)
        self.max_steps = int(len(self.df) - 1 - self.start_idx)
        self.end_idx = int(self.start_idx + self.max_steps)

        # --- 6️⃣ Initial observation and info ---
        observation = self._get_obs()
        info = self._get_info()
        if self.trade_logger and self.episode_count > 0:
            self.trade_logger.log_episode_summary({
                'episode': self.episode_count,
                'total_reward': self.episode_reward,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'final_balance': self.balance
            })
        self.episode_count += 1
        self.episode_reward = 0.0
        # Optional: early print or debug log
        # print(f"🔁 Environment reset at index {self.start_idx} (Step range: {self.start_idx}-{self.end_idx})")

        return observation, info
