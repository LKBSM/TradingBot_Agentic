# =============================================================================
# TRADING BOT - FULL TRAINING SCRIPT FOR GOOGLE COLAB
# =============================================================================
#
# HOW TO USE ON GOOGLE COLAB:
# 1. Open Google Colab: https://colab.research.google.com/
# 2. Create a new notebook
# 3. Copy-paste this ENTIRE script into a single cell
# 4. Run the cell (Ctrl+Enter or click Play)
# 5. When prompted, upload your XAU_15MIN_2019_2024.csv file
#
# IMPORTANT: Enable GPU before running!
# Runtime → Change runtime type → GPU (T4)
#
# Training time: ~2-4 hours on GPU
# =============================================================================

# =============================================================================
# STEP 1: INSTALL PACKAGES
# =============================================================================
print("=" * 70)
print("STEP 1: Installing packages...")
print("=" * 70)

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

packages = [
    "stable-baselines3[extra]",
    "gymnasium",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "ta",
    "torch",
    "tensorboard"
]

for pkg in packages:
    try:
        install(pkg)
    except:
        print(f"Warning: Could not install {pkg}")

print("✅ Packages installed!\n")

# =============================================================================
# STEP 2: IMPORTS
# =============================================================================
print("=" * 70)
print("STEP 2: Importing libraries...")
print("=" * 70)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
import ta

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Check GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No GPU detected - training will be slower")

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("✅ Libraries imported!\n")

# =============================================================================
# STEP 3: CONFIGURATION
# =============================================================================
print("=" * 70)
print("STEP 3: Loading configuration...")
print("=" * 70)

@dataclass
class TradingConfig:
    """Optimized trading configuration."""

    # Capital
    initial_balance: float = 10000

    # Realistic costs for Gold
    spread: float = 0.00025          # 0.025% (2.5 pips)
    commission: float = 0.00010      # 0.01% per side
    slippage_base: float = 0.00010   # 0.01%
    slippage_volatility_mult: float = 0.5
    slippage_news_mult: float = 2.0

    # Risk Management
    max_position: float = 1.0
    stop_loss_pct: float = 0.02      # 2% stop-loss
    take_profit_pct: float = 0.04    # 4% take-profit
    max_daily_drawdown: float = 0.05 # 5% max daily DD
    max_total_drawdown: float = 0.15 # 15% max total DD

    # Reward Weights
    reward_pnl_weight: float = 1.0
    reward_sharpe_weight: float = 0.5
    reward_drawdown_penalty: float = 2.0
    reward_overtrade_penalty: float = 0.3
    reward_consistency_bonus: float = 0.2

    # Anti-Overfitting
    price_noise: float = 0.0001
    dropout_prob: float = 0.02

    # Training
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    early_stopping_patience: int = 5

config = TradingConfig()

print(f"Configuration loaded:")
print(f"  Initial balance: ${config.initial_balance:,}")
print(f"  Stop-loss: {config.stop_loss_pct:.1%}")
print(f"  Take-profit: {config.take_profit_pct:.1%}")
print(f"  Max Drawdown: {config.max_total_drawdown:.1%}")
print(f"  Total timesteps: {config.total_timesteps:,}")
print(f"  Estimated cost per trade: ~{(config.spread + 2*config.commission + config.slippage_base)*100:.2f}%")
print("✅ Configuration loaded!\n")

# =============================================================================
# STEP 4: ECONOMIC CALENDAR DATA (REAL HISTORICAL VALUES)
# =============================================================================
print("=" * 70)
print("STEP 4: Creating Economic Calendar with REAL data...")
print("=" * 70)

# NFP (Non-Farm Payrolls) - in thousands of jobs
NFP_DATA = {
    2019: {1: 304, 2: 56, 3: 189, 4: 263, 5: 75, 6: 224, 7: 164, 8: 130, 9: 136, 10: 128, 11: 266, 12: 147},
    2020: {1: 225, 2: 273, 3: -701, 4: -20687, 5: 2509, 6: 4800, 7: 1763, 8: 1371, 9: 661, 10: 638, 11: 245, 12: -140},
    2021: {1: 233, 2: 468, 3: 916, 4: 278, 5: 614, 6: 962, 7: 1091, 8: 366, 9: 312, 10: 546, 11: 249, 12: 510},
    2022: {1: 504, 2: 714, 3: 431, 4: 428, 5: 390, 6: 372, 7: 528, 8: 315, 9: 263, 10: 261, 11: 263, 12: 223},
    2023: {1: 517, 2: 311, 3: 236, 4: 294, 5: 339, 6: 306, 7: 187, 8: 236, 9: 297, 10: 150, 11: 199, 12: 216},
    2024: {1: 353, 2: 275, 3: 315, 4: 165, 5: 272, 6: 206, 7: 114, 8: 142, 9: 254, 10: 227, 11: 256, 12: 212},
}

# CPI Year-over-Year (%)
CPI_DATA = {
    2019: {1: 1.6, 2: 1.5, 3: 1.9, 4: 2.0, 5: 1.8, 6: 1.6, 7: 1.8, 8: 1.7, 9: 1.7, 10: 1.8, 11: 2.1, 12: 2.3},
    2020: {1: 2.5, 2: 2.3, 3: 1.5, 4: 0.3, 5: 0.1, 6: 0.6, 7: 1.0, 8: 1.3, 9: 1.4, 10: 1.2, 11: 1.2, 12: 1.4},
    2021: {1: 1.4, 2: 1.7, 3: 2.6, 4: 4.2, 5: 5.0, 6: 5.4, 7: 5.4, 8: 5.3, 9: 5.4, 10: 6.2, 11: 6.8, 12: 7.0},
    2022: {1: 7.5, 2: 7.9, 3: 8.5, 4: 8.3, 5: 8.6, 6: 9.1, 7: 8.5, 8: 8.3, 9: 8.2, 10: 7.7, 11: 7.1, 12: 6.5},
    2023: {1: 6.4, 2: 6.0, 3: 5.0, 4: 4.9, 5: 4.0, 6: 3.0, 7: 3.2, 8: 3.7, 9: 3.7, 10: 3.2, 11: 3.1, 12: 3.4},
    2024: {1: 3.1, 2: 3.2, 3: 3.5, 4: 3.4, 5: 3.3, 6: 3.0, 7: 2.9, 8: 2.5, 9: 2.4, 10: 2.6, 11: 2.7, 12: 2.9},
}

# Federal Funds Rate (%)
FED_RATE_DATA = {
    2019: {1: 2.50, 2: 2.50, 3: 2.50, 4: 2.50, 5: 2.50, 6: 2.50, 7: 2.25, 8: 2.25, 9: 2.00, 10: 1.75, 11: 1.75, 12: 1.75},
    2020: {1: 1.75, 2: 1.75, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.25, 7: 0.25, 8: 0.25, 9: 0.25, 10: 0.25, 11: 0.25, 12: 0.25},
    2021: {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.25, 7: 0.25, 8: 0.25, 9: 0.25, 10: 0.25, 11: 0.25, 12: 0.25},
    2022: {1: 0.25, 2: 0.25, 3: 0.50, 4: 0.50, 5: 1.00, 6: 1.75, 7: 2.50, 8: 2.50, 9: 3.25, 10: 3.25, 11: 4.00, 12: 4.50},
    2023: {1: 4.50, 2: 4.75, 3: 5.00, 4: 5.00, 5: 5.25, 6: 5.25, 7: 5.50, 8: 5.50, 9: 5.50, 10: 5.50, 11: 5.50, 12: 5.50},
    2024: {1: 5.50, 2: 5.50, 3: 5.50, 4: 5.50, 5: 5.50, 6: 5.50, 7: 5.50, 8: 5.50, 9: 5.00, 10: 5.00, 11: 4.75, 12: 4.50},
}

# FOMC Meeting Dates
FOMC_DATES = {
    2019: ['01-30', '03-20', '05-01', '06-19', '07-31', '09-18', '10-30', '12-11'],
    2020: ['01-29', '03-03', '03-15', '04-29', '06-10', '07-29', '09-16', '11-05', '12-16'],
    2021: ['01-27', '03-17', '04-28', '06-16', '07-28', '09-22', '11-03', '12-15'],
    2022: ['01-26', '03-16', '05-04', '06-15', '07-27', '09-21', '11-02', '12-14'],
    2023: ['02-01', '03-22', '05-03', '06-14', '07-26', '09-20', '11-01', '12-13'],
    2024: ['01-31', '03-20', '05-01', '06-12', '07-31', '09-18', '11-07', '12-18'],
}

def get_first_friday(year: int, month: int) -> datetime:
    """Get first Friday of the month (NFP release day)."""
    first_day = datetime(year, month, 1)
    days_until_friday = (4 - first_day.weekday()) % 7
    return first_day + timedelta(days=days_until_friday)

def create_economic_calendar() -> pd.DataFrame:
    """Create economic calendar with real historical data."""
    events = []

    for year in range(2019, 2025):
        for month in range(1, 13):
            # NFP - First Friday
            nfp_date = get_first_friday(year, month)
            nfp_value = NFP_DATA.get(year, {}).get(month, 200)
            nfp_prev = NFP_DATA.get(year, {}).get(month - 1) if month > 1 else NFP_DATA.get(year - 1, {}).get(12, 200)

            if nfp_prev and nfp_prev != 0:
                nfp_surprise = (nfp_value - nfp_prev) / abs(nfp_prev)
            else:
                nfp_surprise = 0

            events.append({
                'datetime': nfp_date.replace(hour=12, minute=30),
                'event': 'NFP',
                'impact': 'HIGH',
                'actual': nfp_value,
                'previous': nfp_prev,
                'surprise': nfp_surprise
            })

            # CPI - Around 12th of month
            cpi_date = datetime(year, month, 12)
            if cpi_date.weekday() >= 5:  # Weekend
                cpi_date = cpi_date - timedelta(days=cpi_date.weekday() - 4)

            cpi_value = CPI_DATA.get(year, {}).get(month, 2.0)
            cpi_prev = CPI_DATA.get(year, {}).get(month - 1) if month > 1 else CPI_DATA.get(year - 1, {}).get(12, 2.0)

            events.append({
                'datetime': cpi_date.replace(hour=12, minute=30),
                'event': 'CPI',
                'impact': 'HIGH',
                'actual': cpi_value,
                'previous': cpi_prev,
                'surprise': cpi_value - (cpi_prev or 0)
            })

        # FOMC Meetings
        for date_str in FOMC_DATES.get(year, []):
            try:
                month = int(date_str.split('-')[0])
                day = int(date_str.split('-')[1])
                fomc_date = datetime(year, month, day, 18, 0)

                rate = FED_RATE_DATA.get(year, {}).get(month, 2.0)
                rate_prev = FED_RATE_DATA.get(year, {}).get(month - 1) if month > 1 else FED_RATE_DATA.get(year - 1, {}).get(12, 2.0)

                events.append({
                    'datetime': fomc_date,
                    'event': 'FOMC',
                    'impact': 'HIGH',
                    'actual': rate,
                    'previous': rate_prev,
                    'surprise': rate - (rate_prev or rate)
                })
            except:
                continue

    df = pd.DataFrame(events)
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

# Create calendar
calendar_df = create_economic_calendar()
print(f"✅ Economic Calendar created: {len(calendar_df)} events")
print(f"   NFP: {len(calendar_df[calendar_df['event'] == 'NFP'])} events")
print(f"   CPI: {len(calendar_df[calendar_df['event'] == 'CPI'])} events")
print(f"   FOMC: {len(calendar_df[calendar_df['event'] == 'FOMC'])} events")
print()

# =============================================================================
# STEP 5: UPLOAD AND LOAD GOLD DATA
# =============================================================================
print("=" * 70)
print("STEP 5: Upload your Gold data file...")
print("=" * 70)

# Try to import Google Colab's file upload
try:
    from google.colab import files
    print("\n📤 Please upload your XAU_15MIN_2019_2024.csv file:")
    uploaded = files.upload()

    # Get the filename
    gold_filename = list(uploaded.keys())[0]

    # Move to data folder
    import shutil
    shutil.move(gold_filename, f'data/{gold_filename}')
    gold_filepath = f'data/{gold_filename}'

except ImportError:
    # Not on Colab, try local file
    print("Not on Colab - looking for local file...")
    gold_filepath = 'data/XAU_15MIN_2019_2024.csv'
    if not os.path.exists(gold_filepath):
        raise FileNotFoundError(f"Please place your Gold data file at: {gold_filepath}")

# Load Gold data
print(f"\n📂 Loading Gold data from: {gold_filepath}")
gold_df = pd.read_csv(gold_filepath, parse_dates=['Date'])
gold_df = gold_df.set_index('Date')
gold_df = gold_df.sort_index()

print(f"✅ Gold data loaded: {len(gold_df):,} bars")
print(f"   Period: {gold_df.index.min()} → {gold_df.index.max()}")
print(f"   Columns: {list(gold_df.columns)}")
print()

# =============================================================================
# STEP 6: MERGE GOLD DATA WITH ECONOMIC CALENDAR
# =============================================================================
print("=" * 70)
print("STEP 6: Merging Gold data with Economic Calendar...")
print("=" * 70)

def add_news_features(df: pd.DataFrame, calendar: pd.DataFrame,
                      window_before: int = 60, window_after: int = 120) -> pd.DataFrame:
    """Add news features to price data."""
    df = df.copy()

    # Initialize news columns
    df['news_event'] = 0
    df['news_impact'] = 0.0
    df['news_surprise'] = 0.0
    df['news_type'] = ''
    df['minutes_to_news'] = 0

    for _, event in calendar.iterrows():
        event_time = event['datetime']

        # Find bars in window
        window_start = event_time - timedelta(minutes=window_before)
        window_end = event_time + timedelta(minutes=window_after)

        mask = (df.index >= window_start) & (df.index <= window_end)

        if mask.any():
            df.loc[mask, 'news_event'] = 1
            df.loc[mask, 'news_impact'] = 1.0 if event['impact'] == 'HIGH' else 0.5
            df.loc[mask, 'news_surprise'] = event['surprise']
            df.loc[mask, 'news_type'] = event['event']

            # Calculate minutes to event
            for idx in df.loc[mask].index:
                minutes_diff = (event_time - idx).total_seconds() / 60
                current_val = df.loc[idx, 'minutes_to_news']
                if current_val == 0 or abs(minutes_diff) < abs(current_val):
                    df.loc[idx, 'minutes_to_news'] = minutes_diff

    return df

# Merge
gold_df = add_news_features(gold_df, calendar_df)

news_bars = gold_df['news_event'].sum()
print(f"✅ Data merged!")
print(f"   Bars with news events: {news_bars:,} ({news_bars/len(gold_df)*100:.1f}%)")
print()

# =============================================================================
# STEP 7: CREATE TRAIN/VAL/TEST SPLITS
# =============================================================================
print("=" * 70)
print("STEP 7: Creating chronological data splits...")
print("=" * 70)

n = len(gold_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = gold_df.iloc[:train_end].copy()
df_val = gold_df.iloc[train_end:val_end].copy()
df_test = gold_df.iloc[val_end:].copy()

print(f"✅ Data splits created:")
print(f"   Train:      {len(df_train):>8,} bars ({df_train.index.min().date()} → {df_train.index.max().date()})")
print(f"   Validation: {len(df_val):>8,} bars ({df_val.index.min().date()} → {df_val.index.max().date()})")
print(f"   Test:       {len(df_test):>8,} bars ({df_test.index.min().date()} → {df_test.index.max().date()})")
print()

# =============================================================================
# STEP 8: TRADING ENVIRONMENT
# =============================================================================
print("=" * 70)
print("STEP 8: Creating optimized trading environment...")
print("=" * 70)

class OptimizedTradingEnv(gym.Env):
    """
    Optimized Trading Environment with:
    - Multi-objective reward function
    - Realistic transaction costs
    - Built-in risk management (stop-loss, take-profit)
    - Anti-overfitting (data augmentation)
    - Stationary features
    """

    def __init__(self, df: pd.DataFrame, config: TradingConfig, training: bool = True):
        super().__init__()

        self.config = config
        self.training = training
        self.df_original = df.copy()

        # Prepare features
        self._prepare_features()

        # Spaces
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features,), dtype=np.float32
        )

        # Tracking for multi-objective reward
        self.returns_history = []
        self.trade_count = 0
        self.win_count = 0

        self.reset()

    def _prepare_features(self):
        """Prepare ROBUST and STATIONARY features."""
        df = self.df_original.copy()

        # Returns (stationary)
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # RSI normalized (z-score around 50)
        df['rsi_raw'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['rsi'] = (df['rsi_raw'] - 50) / 25

        # MACD normalized by ATR
        macd = ta.trend.MACD(df['Close'])
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['macd_norm'] = macd.macd() / (df['atr'] + 1e-8)
        df['macd_signal_norm'] = macd.macd_signal() / (df['atr'] + 1e-8)
        df['macd_hist_norm'] = macd.macd_diff() / (df['atr'] + 1e-8)

        # Bollinger position (0-1)
        bb = ta.volatility.BollingerBands(df['Close'])
        bb_high = bb.bollinger_hband()
        bb_low = bb.bollinger_lband()
        df['bb_position'] = (df['Close'] - bb_low) / (bb_high - bb_low + 1e-8)
        df['bb_position'] = df['bb_position'].clip(0, 1)

        # Price vs MAs (ratio, not absolute difference)
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['price_sma20_ratio'] = (df['Close'] / df['sma_20']) - 1
        df['price_sma50_ratio'] = (df['Close'] / df['sma_50']) - 1

        # Realized volatility (annualized)
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252 * 24 * 4)
        df['volatility_zscore'] = (df['volatility'] - df['volatility'].rolling(100).mean()) / (df['volatility'].rolling(100).std() + 1e-8)

        # Volume ratio
        df['volume_ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
        df['volume_ratio'] = np.clip(df['volume_ratio'], 0, 5)

        # Multi-timeframe momentum
        df['momentum_5'] = df['Close'].pct_change(5)
        df['momentum_20'] = df['Close'].pct_change(20)

        # Market regime (trend vs range)
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / (df['atr'] + 1e-8)

        # News features
        if 'news_event' not in df.columns:
            df['news_event'] = 0
            df['news_impact'] = 0
            df['news_surprise'] = 0

        # Select features
        self.feature_columns = [
            'returns', 'log_returns',
            'rsi', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
            'bb_position', 'price_sma20_ratio', 'price_sma50_ratio',
            'volatility_zscore', 'volume_ratio',
            'momentum_5', 'momentum_20', 'trend_strength',
            'news_event', 'news_impact', 'news_surprise'
        ]

        # Clean NaN
        df = df.dropna()

        # Normalize
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(df[self.feature_columns])
        self.prices = df['Close'].values
        self.highs = df['High'].values
        self.lows = df['Low'].values
        self.volatility = df['volatility'].fillna(0.2).values
        self.news_events = df['news_event'].values
        self.df_clean = df

        # +3 for position, pnl, drawdown
        self.n_features = len(self.feature_columns) + 3

    def _apply_data_augmentation(self, features):
        """Apply data augmentation for anti-overfitting."""
        if not self.training:
            return features

        augmented = features.copy()

        # Gaussian noise
        noise = np.random.normal(0, self.config.price_noise, features.shape)
        augmented += noise

        # Feature dropout
        if np.random.random() < self.config.dropout_prob:
            drop_idx = np.random.randint(0, len(features))
            augmented[drop_idx] = 0

        return augmented

    def _calculate_transaction_cost(self, price, is_news=False):
        """Calculate REALISTIC transaction costs."""
        cost = self.config.spread + self.config.commission

        # Slippage (variable)
        slippage = self.config.slippage_base

        # Higher slippage during high volatility
        current_vol = self.volatility[self.current_step] if self.current_step < len(self.volatility) else 0.2
        slippage *= (1 + self.config.slippage_volatility_mult * min(current_vol, 1.0))

        # Higher slippage during news
        if is_news:
            slippage *= self.config.slippage_news_mult

        cost += slippage

        return cost * price

    def _calculate_reward(self, pnl, action):
        """Calculate MULTI-OBJECTIVE reward."""
        reward = 0

        # 1. PnL normalized (base)
        pnl_norm = pnl / self.config.initial_balance
        reward += pnl_norm * self.config.reward_pnl_weight

        # 2. Rolling Sharpe (if enough data)
        if len(self.returns_history) >= 20:
            returns = np.array(self.returns_history[-20:])
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
                reward += sharpe * self.config.reward_sharpe_weight * 0.1

        # 3. Drawdown penalty
        current_dd = self._calculate_drawdown()
        if current_dd > 0.05:
            reward -= current_dd * self.config.reward_drawdown_penalty

        # 4. Overtrading penalty
        if action != 0:
            bars_since_last_trade = getattr(self, 'bars_since_trade', 100)
            if bars_since_last_trade < 4:
                reward -= self.config.reward_overtrade_penalty

        # 5. Consistency bonus
        if pnl > 0:
            self.consecutive_wins = getattr(self, 'consecutive_wins', 0) + 1
            if self.consecutive_wins >= 3:
                reward += self.config.reward_consistency_bonus
        else:
            self.consecutive_wins = 0

        return reward

    def _calculate_drawdown(self):
        """Calculate current drawdown."""
        current_value = self.balance + self._get_unrealized_pnl()
        if current_value >= self.peak_value:
            self.peak_value = current_value
            return 0
        return (self.peak_value - current_value) / self.peak_value

    def _get_unrealized_pnl(self):
        """Calculate unrealized PnL."""
        if self.position == 0:
            return 0
        current_price = self.prices[self.current_step]
        if self.position > 0:
            return (current_price - self.entry_price) * self.position_size
        else:
            return (self.entry_price - current_price) * abs(self.position_size)

    def _check_stop_loss_take_profit(self):
        """Check and execute stops."""
        if self.position == 0:
            return False, None

        current_price = self.prices[self.current_step]

        if self.position > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        if pnl_pct <= -self.config.stop_loss_pct:
            return True, 'stop_loss'

        if pnl_pct >= self.config.take_profit_pct:
            return True, 'take_profit'

        return False, None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.config.initial_balance
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.total_pnl = 0
        self.peak_value = self.config.initial_balance
        self.returns_history = []
        self.trade_count = 0
        self.win_count = 0
        self.bars_since_trade = 100
        self.consecutive_wins = 0

        return self._get_observation(), {}

    def _get_observation(self):
        features = self.features[self.current_step].copy()
        features = self._apply_data_augmentation(features)

        position_feat = self.position
        pnl_feat = self.total_pnl / self.config.initial_balance
        dd_feat = self._calculate_drawdown()

        return np.concatenate([features, [position_feat, pnl_feat, dd_feat]]).astype(np.float32)

    def step(self, action):
        current_price = self.prices[self.current_step]
        is_news = self.news_events[self.current_step] > 0 if self.current_step < len(self.news_events) else False
        reward = 0
        pnl = 0

        # Check stop-loss / take-profit
        triggered, trigger_type = self._check_stop_loss_take_profit()
        if triggered:
            pnl = self._close_position(current_price, is_news)
            if trigger_type == 'stop_loss':
                reward -= 0.1

        # Check max drawdown
        current_dd = self._calculate_drawdown()
        if current_dd >= self.config.max_total_drawdown:
            if self.position != 0:
                pnl += self._close_position(current_price, is_news)
            reward -= 1.0
            self.current_step = len(self.prices) - 1
        else:
            # Execute action
            if action == 1:  # BUY
                if self.position <= 0:
                    if self.position < 0:
                        pnl = self._close_position(current_price, is_news)
                    self._open_position(1, current_price, is_news)

            elif action == 2:  # SELL
                if self.position >= 0:
                    if self.position > 0:
                        pnl = self._close_position(current_price, is_news)
                    self._open_position(-1, current_price, is_news)

            reward = self._calculate_reward(pnl, action)

            self.current_step += 1
            self.bars_since_trade += 1

        self.returns_history.append(pnl / self.config.initial_balance if self.config.initial_balance > 0 else 0)

        done = self.current_step >= len(self.prices) - 1

        if done and self.position != 0:
            final_pnl = self._close_position(self.prices[self.current_step], False)
            reward += self._calculate_reward(final_pnl, 0)

        info = {
            'total_pnl': self.total_pnl,
            'position': self.position,
            'drawdown': self._calculate_drawdown(),
            'trade_count': self.trade_count,
            'win_rate': self.win_count / max(1, self.trade_count),
            'balance': self.balance
        }

        return self._get_observation(), reward, done, False, info

    def _open_position(self, direction, price, is_news):
        cost = self._calculate_transaction_cost(price, is_news)
        self.position = direction
        self.position_size = self.config.max_position
        self.entry_price = price
        self.balance -= cost
        self.bars_since_trade = 0

    def _close_position(self, price, is_news):
        if self.position == 0:
            return 0

        cost = self._calculate_transaction_cost(price, is_news)

        if self.position > 0:
            pnl = (price - self.entry_price) * self.position_size - cost
        else:
            pnl = (self.entry_price - price) * abs(self.position_size) - cost

        self.total_pnl += pnl
        self.balance += pnl
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1

        self.position = 0
        self.position_size = 0
        self.entry_price = 0

        return pnl

# Create environments
train_env = OptimizedTradingEnv(df_train, config, training=True)
val_env = OptimizedTradingEnv(df_val, config, training=False)
test_env = OptimizedTradingEnv(df_test, config, training=False)

print(f"✅ Environments created!")
print(f"   Observation shape: {train_env.observation_space.shape}")
print(f"   Action space: {train_env.action_space}")
print(f"   Features: {train_env.n_features}")
print()

# =============================================================================
# STEP 9: TRAINING CALLBACK WITH EARLY STOPPING
# =============================================================================
print("=" * 70)
print("STEP 9: Setting up training callback...")
print("=" * 70)

class OptimizedCallback(BaseCallback):
    """Callback with early stopping based on validation performance."""

    def __init__(self, eval_env, config, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.config = config

        self.best_sharpe = -np.inf
        self.no_improvement_count = 0
        self.results = []

    def _on_step(self):
        if self.n_calls % self.config.eval_freq == 0:
            metrics = self._evaluate()

            self.results.append({
                'timestep': self.n_calls,
                **metrics
            })

            print(f"\n📊 Step {self.n_calls:,}:")
            print(f"   Sharpe: {metrics['sharpe']:.2f} | MaxDD: {metrics['max_dd']:.1%} | "
                  f"WinRate: {metrics['win_rate']:.1%} | Trades: {metrics['trade_count']}")

            # Early stopping check
            if metrics['sharpe'] > self.best_sharpe + 0.01:
                self.best_sharpe = metrics['sharpe']
                self.no_improvement_count = 0
                self.model.save('models/best_model')
                print(f"   ⭐ New best model! Sharpe: {self.best_sharpe:.2f}")
            else:
                self.no_improvement_count += 1
                print(f"   No improvement ({self.no_improvement_count}/{self.config.early_stopping_patience})")

            if self.no_improvement_count >= self.config.early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered!")
                return False

        return True

    def _evaluate(self):
        obs, _ = self.eval_env.reset()
        done = False
        portfolio_values = [self.config.initial_balance]

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = self.eval_env.step(action)
            portfolio_values.append(self.config.initial_balance + info['total_pnl'])

        pv = np.array(portfolio_values)
        returns = np.diff(pv) / (pv[:-1] + 1e-8)

        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 4)
        peak = np.maximum.accumulate(pv)
        max_dd = np.max((peak - pv) / (peak + 1e-8))

        return {
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_return': (pv[-1] / pv[0]) - 1,
            'win_rate': info.get('win_rate', 0),
            'trade_count': info.get('trade_count', 0)
        }

callback = OptimizedCallback(val_env, config)
print("✅ Callback created!")
print()

# =============================================================================
# STEP 10: CREATE AND TRAIN MODEL
# =============================================================================
print("=" * 70)
print("STEP 10: Training the model...")
print("=" * 70)

# Optimized hyperparameters
HYPERPARAMS = {
    'learning_rate': 3e-5,
    'n_steps': 2048,
    'batch_size': 128,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.05,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device.upper()}")
print(f"Total timesteps: {config.total_timesteps:,}")
print(f"Eval frequency: {config.eval_freq:,}")
print(f"Early stopping patience: {config.early_stopping_patience}")
print()

# Create model
model = PPO(
    'MlpPolicy',
    train_env,
    verbose=0,
    tensorboard_log='logs/',
    device=device,
    **HYPERPARAMS
)

print("🚀 Starting training...")
print("=" * 70)

# Train
model.learn(
    total_timesteps=config.total_timesteps,
    callback=callback,
    progress_bar=True
)

# Save final model
model.save('models/final_model')

print()
print("=" * 70)
print("✅ Training complete!")
print(f"   Best validation Sharpe: {callback.best_sharpe:.2f}")
print("=" * 70)
print()

# =============================================================================
# STEP 11: FINAL EVALUATION ON TEST DATA
# =============================================================================
print("=" * 70)
print("STEP 11: Final evaluation on TEST data (never seen before)...")
print("=" * 70)

# Load best model
best_model = PPO.load('models/best_model')

# Evaluate on test
obs, _ = test_env.reset()
done = False
portfolio_values = [config.initial_balance]
actions_taken = []
positions = [0]

while not done:
    action, _ = best_model.predict(obs, deterministic=True)
    obs, reward, done, _, info = test_env.step(action)

    portfolio_values.append(config.initial_balance + info['total_pnl'])
    actions_taken.append(action)
    positions.append(info['position'])

# Calculate metrics
pv = np.array(portfolio_values)
returns = np.diff(pv) / (pv[:-1] + 1e-8)

# Sharpe Ratio
sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 4)

# Sortino Ratio
downside = returns[returns < 0]
sortino = np.mean(returns) / (np.std(downside) + 1e-8) * np.sqrt(252 * 24 * 4) if len(downside) > 0 else 0

# Max Drawdown
peak = np.maximum.accumulate(pv)
drawdown = (peak - pv) / (peak + 1e-8)
max_dd = np.max(drawdown)

# Calmar Ratio
annual_return = (pv[-1] / pv[0]) ** (252 * 24 * 4 / len(pv)) - 1 if len(pv) > 1 else 0
calmar = annual_return / max_dd if max_dd > 0 else 0

# Win Rate
win_rate = info['win_rate']

# Cumulative Return
cum_return = (pv[-1] / pv[0]) - 1

# Print results
print()
print("=" * 70)
print("                    FINAL RESULTS - TEST DATA")
print("=" * 70)
print()
print(f"  Sharpe Ratio:      {sharpe:>10.2f}  {'✅' if sharpe >= 1.0 else '⚠️'} (target: > 1.0)")
print(f"  Sortino Ratio:     {sortino:>10.2f}  {'✅' if sortino >= 1.5 else '⚠️'} (target: > 1.5)")
print(f"  Calmar Ratio:      {calmar:>10.2f}  {'✅' if calmar >= 1.0 else '⚠️'} (target: > 1.0)")
print(f"  Max Drawdown:      {max_dd:>10.1%}  {'✅' if max_dd <= 0.15 else '❌'} (target: < 15%)")
print(f"  Win Rate:          {win_rate:>10.1%}  {'✅' if win_rate >= 0.50 else '⚠️'} (target: > 50%)")
print(f"  Cumulative Return: {cum_return:>10.1%}")
print()
print(f"  Initial Capital:   ${pv[0]:>10,.0f}")
print(f"  Final Capital:     ${pv[-1]:>10,.0f}")
print(f"  Profit/Loss:       ${pv[-1]-pv[0]:>+10,.0f}")
print(f"  Number of Trades:  {info['trade_count']:>10}")
print()
print("=" * 70)

# Verdict
if sharpe >= 1.0 and max_dd <= 0.15 and win_rate >= 0.45:
    print()
    print("🎉 " + "=" * 66 + " 🎉")
    print("   BOT IS READY FOR PAPER TRADING!")
    print("   Next step: 4 weeks of paper trading on MT5 demo account")
    print("🎉 " + "=" * 66 + " 🎉")
else:
    print()
    print("⚠️ Bot needs more work. Suggestions:")
    if sharpe < 1.0:
        print("   - Sharpe too low: increase training or adjust rewards")
    if max_dd > 0.15:
        print("   - Drawdown too high: strengthen risk management")
    if win_rate < 0.45:
        print("   - Win rate low: review strategy or features")

print()

# =============================================================================
# STEP 12: VISUALIZATIONS
# =============================================================================
print("=" * 70)
print("STEP 12: Creating visualizations...")
print("=" * 70)

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# 1. Portfolio Value
axes[0].plot(pv, 'b-', linewidth=0.8, label='Portfolio')
axes[0].axhline(y=config.initial_balance, color='gray', linestyle='--', label='Initial')
axes[0].fill_between(range(len(pv)), pv, config.initial_balance,
                     where=pv >= config.initial_balance, alpha=0.3, color='green')
axes[0].fill_between(range(len(pv)), pv, config.initial_balance,
                     where=pv < config.initial_balance, alpha=0.3, color='red')
axes[0].set_title(f'Portfolio Value (Sharpe: {sharpe:.2f}, Return: {cum_return:.1%})', fontsize=14)
axes[0].set_ylabel('Value ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Drawdown
axes[1].fill_between(range(len(drawdown)), -drawdown*100, 0, alpha=0.7, color='red')
axes[1].axhline(y=-config.max_total_drawdown*100, color='darkred', linestyle='--',
                label=f'Max DD Limit ({config.max_total_drawdown:.0%})')
axes[1].set_title(f'Drawdown (Max: {max_dd:.1%})', fontsize=14)
axes[1].set_ylabel('Drawdown (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Actions Distribution
action_counts = pd.Series(actions_taken).value_counts().sort_index()
labels = ['HOLD', 'BUY', 'SELL']
colors = ['gray', 'green', 'red']
bars = axes[2].bar([labels[i] for i in action_counts.index],
                   action_counts.values,
                   color=[colors[i] for i in action_counts.index])
axes[2].set_title('Actions Distribution', fontsize=14)
axes[2].set_ylabel('Count')
for bar, count in zip(bars, action_counts.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{count}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('results/test_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Visualization saved to results/test_results.png")
print()

# =============================================================================
# STEP 13: SAVE AND DOWNLOAD
# =============================================================================
print("=" * 70)
print("STEP 13: Saving results and preparing download...")
print("=" * 70)

# Save training history
if callback.results:
    history_df = pd.DataFrame(callback.results)
    history_df.to_csv('results/training_history.csv', index=False)
    print("✅ Training history saved to results/training_history.csv")

# Save final metrics
metrics_dict = {
    'sharpe_ratio': sharpe,
    'sortino_ratio': sortino,
    'calmar_ratio': calmar,
    'max_drawdown': max_dd,
    'win_rate': win_rate,
    'cumulative_return': cum_return,
    'initial_capital': pv[0],
    'final_capital': pv[-1],
    'profit_loss': pv[-1] - pv[0],
    'total_trades': info['trade_count'],
    'training_timesteps': config.total_timesteps,
    'best_val_sharpe': callback.best_sharpe
}

metrics_df = pd.DataFrame([metrics_dict])
metrics_df.to_csv('results/final_metrics.csv', index=False)
print("✅ Final metrics saved to results/final_metrics.csv")

# Try to create zip and download (Colab only)
try:
    import shutil
    shutil.make_archive('trading_bot_results', 'zip', '.', 'models')
    shutil.make_archive('trading_bot_full', 'zip', '.')

    from google.colab import files
    print("\n📥 Downloading files...")
    files.download('trading_bot_results.zip')
    files.download('results/test_results.png')
    files.download('results/final_metrics.csv')
    print("✅ Files downloaded!")
except:
    print("\n📁 Files saved locally in 'models/' and 'results/' folders")

print()
print("=" * 70)
print("                    TRAINING COMPLETE!")
print("=" * 70)
print(f"""
Summary:
- Best Model: models/best_model.zip
- Final Model: models/final_model.zip
- Results: results/test_results.png
- Metrics: results/final_metrics.csv

Next Steps:
1. If Sharpe > 1.0 and MaxDD < 15%: Start paper trading
2. Paper trade for 4+ weeks on MT5 demo
3. If paper trading is profitable: Go live with 2% capital
""")
print("=" * 70)
