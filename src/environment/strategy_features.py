import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange

from src.agent_trainer import AgentTrainer

# Configure le logging pour un usage commercial
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- I. Configuration Management (Pydantic) ---
class SMCConfig(BaseModel):
    """Modèle de configuration validé pour le Smart Money Engine."""
    # Indicateurs Techniques
    RSI_WINDOW: int = Field(
        default=10,
        ge=5, le=14,
        description="Fenêtre RSI réduite pour la sensibilité en daytrading."
    )
    MACD_FAST: int = Field(
        default=8,
        ge=5,
        description="EMA rapide MACD."
    )
    MACD_SLOW: int = Field(
        default=17,
        ge=10,
        description="EMA lente MACD."
    )
    MACD_SIGNAL: int = Field(
        default=9,
        ge=5,
        description="EMA signal MACD."
    )
    BB_WINDOW: int = Field(
        default=20,
        ge=10,
        description="Fenêtre Bollinger Bands."
    )
    ATR_WINDOW: int = Field(
        default=7,
        ge=5, le=14,
        description="Fenêtre ATR réduite pour le risque en temps réel."
    )
    FRACTAL_WINDOW: int = Field(
        default=2,
        ge=2,
        description="Périodes de chaque côté pour identifier un fractal (swing point). 2 = 5 bougies."
    )
    FVG_THRESHOLD: float = Field(
        default=0.0,
        ge=0.0,
        description="Seuil minimal de FVG (en valeur absolue ou normalisée)."
    )


# --- II. Core Analysis Engine (Class Architecture) ---
class SmartMoneyEngine:
    """
    Moteur d'analyse haute performance pour les indicateurs TA classiques et les SMC.
    Utilise la vectorisation pour les calculs rapides et l'itération pour la structure SMC.
    """

    def __init__(self, data: pd.DataFrame, config: Dict[str, Any]):
        self.df = data.copy()
        try:
            self.config = SMCConfig(**config)
        except Exception as e:
            logger.error(f"Erreur de validation de la configuration SMC: {e}")
            # En cas d'échec, utilise les valeurs par défaut
            self.config = SMCConfig()
            logger.warning("Utilisation des valeurs par défaut pour la configuration SMC.")

        # S'assure que les colonnes OHLCV sont en minuscules (convention)
        self.df.columns = [col.lower() for col in self.df.columns]
        self.ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.df.columns for col in self.ohlcv_cols):
            raise ValueError(f"Le DataFrame doit contenir les colonnes OHLCV: {self.ohlcv_cols}")

    def _add_ta_indicators(self) -> None:
        """Calcule et ajoute les indicateurs techniques classiques."""
        cfg = self.config

        # --- Relative Strength Index (RSI) ---
        rsi_indicator = RSIIndicator(close=self.df['close'], window=cfg.RSI_WINDOW)
        self.df['RSI'] = rsi_indicator.rsi()

        # --- Moving Average Convergence Divergence (MACD) ---
        macd = MACD(
            close=self.df['close'],
            window_slow=cfg.MACD_SLOW,
            window_fast=cfg.MACD_FAST,
            window_sign=cfg.MACD_SIGNAL
        )
        self.df['MACD_line'] = macd.macd()
        self.df['MACD_signal'] = macd.macd_signal()
        self.df['MACD_Diff'] = macd.macd_diff()

        # --- Bollinger Bands (BB) ---
        bollinger = BollingerBands(close=self.df['close'], window=cfg.BB_WINDOW)
        self.df['BB_L'] = bollinger.bollinger_lband()
        self.df['BB_M'] = bollinger.bollinger_mavg()
        self.df['BB_H'] = bollinger.bollinger_hband()

        # --- Average True Range (ATR) ---
        atr = AverageTrueRange(high=self.df['high'], low=self.df['low'], close=self.df['close'], window=cfg.ATR_WINDOW)
        self.df['ATR'] = atr.average_true_range()

        # --- Simple Candle Features ---
        self.df['SPREAD'] = self.df['high'] - self.df['low']
        self.df['BODY_SIZE'] = abs(self.df['open'] - self.df['close'])

    def _add_smc_base_features(self) -> None:
    """
    Détection CAUSALE des Fractals (Swing Points) et du Fair Value Gap (FVG).
    VERSION CORRIGÉE: Aucun data leakage - Utilise uniquement les données passées.
    """
    
    N = self.config.FRACTAL_WINDOW  # e.g., 2
    
    # Initialize columns
    self.df['UP_FRACTAL'] = np.nan
    self.df['DOWN_FRACTAL'] = np.nan
    
    # =========================================================================
    # 1. FRACTAL DETECTION (CORRECTED - NO DATA LEAKAGE)
    # =========================================================================
    
    # We need 2*N bars of history to confirm a fractal
    # The fractal is at position (i-N), confirmed at position i
    for i in range(2 * N, len(self.df)):
        # Window for checking: [i-2N to i] (inclusive)
        center_idx = i - N  # The potential fractal bar
        
        high_window = self.df['high'].iloc[i - 2*N : i + 1]
        low_window = self.df['low'].iloc[i - 2*N : i + 1]
        
        # Verify window size
        if len(high_window) == 2*N + 1 and len(low_window) == 2*N + 1:
            # UP FRACTAL: center bar has highest high
            if high_window.iloc[N] == high_window.max():
                self.df.iloc[center_idx, self.df.columns.get_loc('UP_FRACTAL')] = self.df.iloc[center_idx]['high']
            
            # DOWN FRACTAL: center bar has lowest low
            if low_window.iloc[N] == low_window.min():
                self.df.iloc[center_idx, self.df.columns.get_loc('DOWN_FRACTAL')] = self.df.iloc[center_idx]['low']
    
    # ✅ CRITICAL: Force last N bars to NaN (can't be confirmed yet)
    self.df.iloc[-N:, self.df.columns.get_loc('UP_FRACTAL')] = np.nan
    self.df.iloc[-N:, self.df.columns.get_loc('DOWN_FRACTAL')] = np.nan
    
    # =========================================================================
    # 2. FAIR VALUE GAP (FVG) - DÉJÀ CAUSAL
    # =========================================================================
    
    # Bullish FVG: Le low de la bar actuelle > high de la bar d'il y a 2 périodes
    bullish_fvg_size = np.where(
        self.df['low'] > self.df['high'].shift(2),
        self.df['low'] - self.df['high'].shift(2),
        0.0
    )
    
    # Bearish FVG: Le high de la bar actuelle < low de la bar d'il y a 2 périodes
    bearish_fvg_size = np.where(
        self.df['high'] < self.df['low'].shift(2),
        self.df['low'].shift(2) - self.df['high'],
        0.0
    )
    
    # Taille totale du FVG (bullish ou bearish)
    self.df['FVG_SIZE'] = bullish_fvg_size + bearish_fvg_size
    
    # Direction du FVG: +1 (bullish), -1 (bearish), 0 (aucun)
    self.df['FVG_DIR'] = np.where(
        bullish_fvg_size > 0, 1,
        np.where(bearish_fvg_size > 0, -1, 0)
    )
    
    # Normalisation par l'ATR
    self.df['FVG_SIZE_NORM'] = np.where(
        self.df['ATR'] > 0,
        self.df['FVG_SIZE'] / self.df['ATR'],
        0.0
    )
    
    # Signal FVG final
    self.df['FVG_SIGNAL'] = np.where(
        np.abs(self.df['FVG_SIZE_NORM']) > self.config.FVG_THRESHOLD,
        self.df['FVG_DIR'],
        0
    )
    
    # =========================================================================
    # 3. VALIDATION ET RAPPORT
    # =========================================================================
    
    # Compter les fractals détectés
    n_up_fractals = self.df['UP_FRACTAL'].notna().sum()
    n_down_fractals = self.df['DOWN_FRACTAL'].notna().sum()
    n_fvg_signals = (self.df['FVG_SIGNAL'] != 0).sum()
    
    # Rapport de détection
    print(f"\n{'=' * 60}")
    print(f"📊 SMC BASE FEATURES - Rapport de Détection (CAUSAL)")
    print(f"{'=' * 60}")
    print(f"✅ UP_FRACTAL détectés:    {n_up_fractals:>6} ({n_up_fractals / len(self.df) * 100:.2f}%)")
    print(f"✅ DOWN_FRACTAL détectés:  {n_down_fractals:>6} ({n_down_fractals / len(self.df) * 100:.2f}%)")
    print(f"✅ FVG_SIGNAL détectés:    {n_fvg_signals:>6} ({n_fvg_signals / len(self.df) * 100:.2f}%)")
    
    # Avertissement si trop peu de fractals
    if n_up_fractals < 10 or n_down_fractals < 10:
        print(f"\n⚠️  WARNING: Très peu de fractals détectés!")
        print(f"    → Considérez réduire FRACTAL_WINDOW (actuellement: {N})")
        print(f"    → Ou vérifiez que vos données ont assez de volatilité")
    
    # Vérification du data leakage
    last_n_up = self.df['UP_FRACTAL'].iloc[-N:].notna().sum()
    last_n_down = self.df['DOWN_FRACTAL'].iloc[-N:].notna().sum()
    
    if last_n_up > 0 or last_n_down > 0:
        print(f"\n🚨 ERREUR CRITIQUE: Data leakage détecté!")
        print(f"   {last_n_up} UP_FRACTAL dans les {N} dernières bars")
        print(f"   {last_n_down} DOWN_FRACTAL dans les {N} dernières bars")
        print(f"   → Les fractals doivent être NaN dans les {N} dernières bars")
        raise ValueError("Data leakage dans la détection de fractals!")
    else:
        print(f"✅ Validation: Pas de fractals dans les {N} dernières bars (causal OK)")
    
    print(f"{'=' * 60}\n")

    def _calculate_structure_iterative(self) -> None:
        """
        Calcul itératif (non vectorisé) de BOS et CHOCH. Essentiel pour la précision SMC.
        """

        bos_signal = [0] * len(self.df)
        choch_signal = [0] * len(self.df)

        # Initialize current confirmed structure points
        current_high_structure = np.nan
        current_low_structure = np.nan

        # Determine initial structure from the first few confirmed fractals
        initial_swings = self.df.loc[self.df['UP_FRACTAL'].notna() | self.df['DOWN_FRACTAL'].notna()].head(2)

        if len(initial_swings) >= 2:
            current_high_structure = initial_swings['high'].max().item()
            current_low_structure = initial_swings['low'].min().item()
        elif len(self.df) > 0:
            current_high_structure = self.df.iloc[0]['high'].item()
            current_low_structure = self.df.iloc[0]['low'].item()

        # State machine for structure tracking
        for i in range(1, len(self.df)):
            current_close = self.df.iloc[i]['close'].item()
            current_high = self.df.iloc[i]['high'].item()
            current_low = self.df.iloc[i]['low'].item()

            # --- CHOCH (Change of Character) ---
            if bos_signal[i - 1] == -1 and not np.isnan(
                    current_high_structure) and current_close > current_high_structure:
                choch_signal[i] = 1
                current_low_structure = current_low
                current_high_structure = current_high
                bos_signal[i] = 1

            elif bos_signal[i - 1] == 1 and not np.isnan(
                    current_low_structure) and current_close < current_low_structure:
                choch_signal[i] = -1
                current_high_structure = current_high
                current_low_structure = current_low
                bos_signal[i] = -1

            # --- BOS (Break of Structure) ---
            elif choch_signal[i] == 0:
                # Bullish BOS
                if bos_signal[i - 1] >= 0 and not np.isnan(
                        current_high_structure) and current_close > current_high_structure:
                    bos_signal[i] = 1
                    current_high_structure = current_high

                # Bearish BOS
                elif bos_signal[i - 1] <= 0 and not np.isnan(
                        current_low_structure) and current_close < current_low_structure:
                    bos_signal[i] = -1
                    current_low_structure = current_low

                else:
                    # Maintain previous structure bias
                    bos_signal[i] = bos_signal[i - 1]

        self.df['BOS_SIGNAL'] = bos_signal
        self.df['CHOCH_SIGNAL'] = choch_signal

    def _add_smc_order_blocks(self) -> None:
        """
        Détecte les Order Blocks (OB) et définit leur zone (High/Low).
        Optimisé pour inclure la validation du FVG pour une meilleure qualité de signal.
        """

        # --- Étape 1: Conditions de base ---
        bullish_ob_condition = (
                (self.df['close'].shift(1) < self.df['open'].shift(1)) &
                (self.df['close'] > self.df['open']) &
                (self.df['high'] > self.df['high'].shift(1))
        )

        bearish_ob_condition = (
                (self.df['close'].shift(1) > self.df['open'].shift(1)) &
                (self.df['close'] < self.df['open']) &
                (self.df['low'] < self.df['low'].shift(1))
        )

        # --- Étape 2: Validation par FVG ---
        fvg_confirmation = (self.df['FVG_SIGNAL'] != 0).shift(1).fillna(False)

        bullish_ob_final = bullish_ob_condition & fvg_confirmation
        bearish_ob_final = bearish_ob_condition & fvg_confirmation

        # --- Étape 3: Définition des zones de l'Order Block ---
        self.df['BULLISH_OB_HIGH'] = np.where(bullish_ob_final, self.df['high'].shift(1), np.nan)
        self.df['BULLISH_OB_LOW'] = np.where(bullish_ob_final, self.df['low'].shift(1), np.nan)
        self.df['BEARISH_OB_HIGH'] = np.where(bearish_ob_final, self.df['high'].shift(1), np.nan)
        self.df['BEARISH_OB_LOW'] = np.where(bearish_ob_final, self.df['low'].shift(1), np.nan)

        # --- Étape 4: Calcul de la force OB normalisée ---
        ob_size = np.where(
            self.df['BULLISH_OB_HIGH'].notna(),
            self.df['BULLISH_OB_HIGH'] - self.df['BULLISH_OB_LOW'],
            np.where(
                self.df['BEARISH_OB_HIGH'].notna(),
                self.df['BEARISH_OB_HIGH'] - self.df['BEARISH_OB_LOW'],
                0.0
            )
        )

        self.df['OB_STRENGTH_NORM'] = np.where(self.df['ATR'] > 0, ob_size / self.df['ATR'], 0.0)

    def analyze(self) -> pd.DataFrame:
        """Exécute l'analyse complète et retourne le DataFrame enrichi."""

        # 1. Analyse Technique (TA)
        logger.info("Démarrage de l'analyse des indicateurs TA.")
        self._add_ta_indicators()

        # 2. Analyse SMC (Vectorisation & OB/FVG)
        logger.info("Démarrage de l'analyse des fonctionnalités SMC (vectorisation).")
        self._add_smc_base_features()
        self._add_smc_order_blocks()

        # 3. Analyse Structurelle (BOS/CHOCH)
        logger.info("Démarrage de l'analyse structurelle (BOS/CHOCH) - Processus itératif.")
        self._calculate_structure_iterative()

        # --- CORRECTION CRITIQUE DU NETTOYAGE ---

        # CRITIQUE: Cibler uniquement les colonnes essentielles pour le RL (RSI, MACD, ATR)
        # et supprimer toutes les lignes ayant UN SEUL NaN dans toutes les colonnes.
        # Ceci est la méthode la plus sûre pour garantir que l'IA reçoit un Tensor propre.
        self.df.dropna(subset=['RSI', 'MACD_line', 'ATR'], inplace=True)

        # ------------------------------------------

        logger.info(f"Analyse terminée. DataFrame final: {self.df.shape}")

        if len(self.df) < self.config.BB_WINDOW * 2:
            logger.warning(f"Le nettoyage a laissé très peu de données ({len(self.df)} lignes).")

        return self.df


if __name__ == '__main__':

    # --- CORRECTION FINALE : IMPORTATION LOCALE DE LA CLASSE D'ENTRAÎNEMENT ---
    # Cette étape est CRITIQUE. Elle retarde l'importation de la classe AgentTrainer
    # jusqu'au moment où toutes ses dépendances (comme TradingEnv) sont chargées.
    # ---------------------------------------------

    # Correction pour l'importation de config et l'ajout au sys.path
    import sys
    import os

    # Ajoute le dossier parent au chemin pour trouver 'src' et donc 'config'
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import config

    # 0. Data Preparation (Mock Data)
    # NOTE: Utilisez vos vraies données (XAU_15M_2019_2024_FILTRE.csv) ici.
    data_points = 10000
    prices = 100 + np.cumsum(np.random.randn(data_points) * 0.1)
    df_data = pd.DataFrame({
        'Date': pd.to_datetime(pd.date_range(start='2024-01-01', periods=data_points, freq='15min')),
        'Open': prices + np.random.uniform(-0.1, 0.1, data_points),
        'High': prices + np.random.uniform(0.1, 0.2, data_points),
        'Low': prices - np.random.uniform(0.1, 0.2, data_points),
        'Close': prices,
        'Volume': np.random.randint(100, 1000, data_points)
    }).set_index('Date')

    # Split data: Training (80%) and Testing (20%)
    split_index = int(len(df_data) * 0.8)
    df_train = df_data.iloc[:split_index].copy()
    df_test = df_data.iloc[split_index:].copy()

    if df_train.empty or len(df_train) < config.LOOKBACK_WINDOW_SIZE * 2:
        raise ValueError("Le DataFrame d'entraînement est trop petit pour l'initialisation.")

    # --- NOUVEAU : ENTRAÎNEMENT HORS LIGNE EN BOUCLE (Utilisation de TRAINING_TIMESTEPS) ---
    n_sessions = 10  # Entraînement en 3 sessions pour plus de stabilité

    # L'initialisation de l'AgentTrainer est possible grâce à l'importation locale
    trainer = AgentTrainer(df_historical=df_train)

    # Définition des pas de temps par session
    total_timesteps_per_session = config.TRAINING_TIMESTEPS // n_sessions

    # 1. Entraînement initial (Création du premier modèle)
    trained_agent = trainer.train_offline(total_timesteps=total_timesteps_per_session)

    # 2. Sessions suivantes d'entraînement en continu
    for i in range(1, n_sessions):
        # Sauvegarde de la session précédente pour le chargement
        model_name = f"model_offline_session_{i}"
        trainer.agent.save(os.path.join(config.MODEL_DIR, f"{model_name}.zip"))

        # Continuer l'entraînement à partir du modèle précédemment sauvegardé
        trained_agent = trainer.continue_training(
            model_name=model_name,
            total_timesteps=total_timesteps_per_session
        )

    # Évaluation de l'agent final
