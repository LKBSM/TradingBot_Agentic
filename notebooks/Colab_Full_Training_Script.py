# =============================================================================
# TRADING BOT - COMPLETE COLAB TRAINING SCRIPT
# =============================================================================
#
# HOW TO USE:
# 1. Open Google Colab: https://colab.research.google.com/
# 2. Create a new notebook
# 3. Copy this ENTIRE script into a cell
# 4. Set your GitHub token and repo info in the CONFIG section below
# 5. Enable GPU: Runtime → Change runtime type → GPU (T4)
# 6. Run the cell (Ctrl+Enter)
#
# Training time: ~2-4 hours on GPU T4
# =============================================================================

# =============================================================================
# CONFIG - MODIFY THESE VALUES
# =============================================================================

# Your GitHub Personal Access Token (with repo access)
# Create one at: https://github.com/settings/tokens
GITHUB_TOKEN = "your_github_token_here"  # <-- REPLACE THIS

# Your repository info
GITHUB_USERNAME = "LKBSM"
GITHUB_REPO = "TradingBot_Agentic"

# Release info (your Gold data is stored as a release asset)
RELEASE_TAG = "v1.0"  # The release tag
DATA_FILENAME = "XAU_15MIN_2019_2024.csv"  # Asset filename

# Local path where data will be saved
GOLD_DATA_PATH = "data/XAU_15MIN_2019_2024.csv"

# Training configuration
TOTAL_TIMESTEPS = 500_000  # Reduce to 100_000 for quick test
EVAL_FREQ = 10_000
EARLY_STOPPING_PATIENCE = 5

# =============================================================================
# 🚀 STEP 1: SETUP AND INSTALL
# =============================================================================
print("=" * 70)
print("STEP 1: Installing packages...")
print("=" * 70)

import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

packages = [
    "stable-baselines3[extra]",
    "gymnasium",
    "pandas",
    "numpy",
    "matplotlib",
    "scikit-learn",
    "ta",
    "torch",
    "tensorboard"
]

for pkg in packages:
    try:
        install(pkg)
        print(f"  ✓ {pkg}")
    except Exception as e:
        print(f"  ✗ {pkg}: {e}")

print("\n✅ Packages installed!")

# =============================================================================
# STEP 2: CLONE REPOSITORY AND DOWNLOAD DATA FROM RELEASE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Cloning repository and downloading data...")
print("=" * 70)

import requests

# Clone with token authentication (main branch)
repo_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
clone_dir = f"/content/{GITHUB_REPO}"

if os.path.exists(clone_dir):
    print(f"  Repository already exists at {clone_dir}")
    os.chdir(clone_dir)
    subprocess.run(["git", "pull"], capture_output=True)
else:
    subprocess.run(["git", "clone", repo_url, clone_dir], capture_output=True)
    os.chdir(clone_dir)

print(f"  Repository cloned to {clone_dir}")

# Create data directory
os.makedirs("data", exist_ok=True)

# Download Gold data from GitHub Release
print(f"\n  Downloading Gold data from release {RELEASE_TAG}...")

# Get release assets via GitHub API
api_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/releases/tags/{RELEASE_TAG}"
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

response = requests.get(api_url, headers=headers)
if response.status_code == 200:
    release_data = response.json()
    assets = release_data.get("assets", [])

    # Find the data file
    data_asset = None
    for asset in assets:
        if asset["name"] == DATA_FILENAME:
            data_asset = asset
            break

    if data_asset:
        # Download the asset
        download_url = data_asset["url"]
        download_headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/octet-stream"
        }

        print(f"  Downloading {DATA_FILENAME} ({data_asset['size'] / 1024 / 1024:.1f} MB)...")
        data_response = requests.get(download_url, headers=download_headers, stream=True)

        if data_response.status_code == 200:
            with open(GOLD_DATA_PATH, "wb") as f:
                for chunk in data_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  Gold data downloaded to {GOLD_DATA_PATH}")
        else:
            print(f"  ERROR downloading: {data_response.status_code}")
    else:
        print(f"  ERROR: Asset {DATA_FILENAME} not found in release")
else:
    print(f"  ERROR getting release: {response.status_code}")

# Verify Gold data exists
if os.path.exists(GOLD_DATA_PATH):
    file_size = os.path.getsize(GOLD_DATA_PATH) / 1024 / 1024
    print(f"  Gold data ready: {GOLD_DATA_PATH} ({file_size:.1f} MB)")
else:
    print(f"  FAILED: Gold data NOT found at: {GOLD_DATA_PATH}")

# =============================================================================
# 🚀 STEP 3: IMPORTS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Importing libraries...")
print("=" * 70)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Optional
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

print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

print("✅ Libraries imported!")

# =============================================================================
# 🚀 STEP 4: TRADING CONFIGURATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Loading configuration...")
print("=" * 70)

@dataclass
class TradingConfig:
    """Optimized trading configuration."""
    initial_balance: float = 10000

    # Realistic costs for Gold
    spread: float = 0.00025
    commission: float = 0.00010
    slippage_base: float = 0.00010
    slippage_volatility_mult: float = 0.5
    slippage_news_mult: float = 2.0

    # Risk Management
    max_position: float = 1.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_total_drawdown: float = 0.15

    # Reward Weights
    reward_pnl_weight: float = 1.0
    reward_sharpe_weight: float = 0.5
    reward_drawdown_penalty: float = 2.0
    reward_overtrade_penalty: float = 0.3
    reward_consistency_bonus: float = 0.2

    # Anti-Overfitting
    price_noise: float = 0.0001
    dropout_prob: float = 0.02

config = TradingConfig()
print(f"  Initial balance: ${config.initial_balance:,}")
print(f"  Stop-loss: {config.stop_loss_pct:.1%}")
print(f"  Take-profit: {config.take_profit_pct:.1%}")
print(f"  Max Drawdown: {config.max_total_drawdown:.1%}")
print("✅ Configuration loaded!")

# =============================================================================
# 🚀 STEP 5: ECONOMIC CALENDAR (REAL DATA)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Creating Economic Calendar...")
print("=" * 70)

NFP_DATA = {
    2019: {1: 304, 2: 56, 3: 189, 4: 263, 5: 75, 6: 224, 7: 164, 8: 130, 9: 136, 10: 128, 11: 266, 12: 147},
    2020: {1: 225, 2: 273, 3: -701, 4: -20687, 5: 2509, 6: 4800, 7: 1763, 8: 1371, 9: 661, 10: 638, 11: 245, 12: -140},
    2021: {1: 233, 2: 468, 3: 916, 4: 278, 5: 614, 6: 962, 7: 1091, 8: 366, 9: 312, 10: 546, 11: 249, 12: 510},
    2022: {1: 504, 2: 714, 3: 431, 4: 428, 5: 390, 6: 372, 7: 528, 8: 315, 9: 263, 10: 261, 11: 263, 12: 223},
    2023: {1: 517, 2: 311, 3: 236, 4: 294, 5: 339, 6: 306, 7: 187, 8: 236, 9: 297, 10: 150, 11: 199, 12: 216},
    2024: {1: 353, 2: 275, 3: 315, 4: 165, 5: 272, 6: 206, 7: 114, 8: 142, 9: 254, 10: 227, 11: 256, 12: 212},
}

CPI_DATA = {
    2019: {1: 1.6, 2: 1.5, 3: 1.9, 4: 2.0, 5: 1.8, 6: 1.6, 7: 1.8, 8: 1.7, 9: 1.7, 10: 1.8, 11: 2.1, 12: 2.3},
    2020: {1: 2.5, 2: 2.3, 3: 1.5, 4: 0.3, 5: 0.1, 6: 0.6, 7: 1.0, 8: 1.3, 9: 1.4, 10: 1.2, 11: 1.2, 12: 1.4},
    2021: {1: 1.4, 2: 1.7, 3: 2.6, 4: 4.2, 5: 5.0, 6: 5.4, 7: 5.4, 8: 5.3, 9: 5.4, 10: 6.2, 11: 6.8, 12: 7.0},
    2022: {1: 7.5, 2: 7.9, 3: 8.5, 4: 8.3, 5: 8.6, 6: 9.1, 7: 8.5, 8: 8.3, 9: 8.2, 10: 7.7, 11: 7.1, 12: 6.5},
    2023: {1: 6.4, 2: 6.0, 3: 5.0, 4: 4.9, 5: 4.0, 6: 3.0, 7: 3.2, 8: 3.7, 9: 3.7, 10: 3.2, 11: 3.1, 12: 3.4},
    2024: {1: 3.1, 2: 3.2, 3: 3.5, 4: 3.4, 5: 3.3, 6: 3.0, 7: 2.9, 8: 2.5, 9: 2.4, 10: 2.6, 11: 2.7, 12: 2.9},
}

FOMC_DATES = {
    2019: ['01-30', '03-20', '05-01', '06-19', '07-31', '09-18', '10-30', '12-11'],
    2020: ['01-29', '03-03', '03-15', '04-29', '06-10', '07-29', '09-16', '11-05', '12-16'],
    2021: ['01-27', '03-17', '04-28', '06-16', '07-28', '09-22', '11-03', '12-15'],
    2022: ['01-26', '03-16', '05-04', '06-15', '07-27', '09-21', '11-02', '12-14'],
    2023: ['02-01', '03-22', '05-03', '06-14', '07-26', '09-20', '11-01', '12-13'],
    2024: ['01-31', '03-20', '05-01', '06-12', '07-31', '09-18', '11-07', '12-18'],
}

def get_first_friday(year, month):
    first_day = datetime(year, month, 1)
    days_until_friday = (4 - first_day.weekday()) % 7
    return first_day + timedelta(days=days_until_friday)

def create_economic_calendar():
    events = []
    for year in range(2019, 2025):
        for month in range(1, 13):
            # NFP
            nfp_date = get_first_friday(year, month)
            nfp_val = NFP_DATA.get(year, {}).get(month, 200)
            nfp_prev = NFP_DATA.get(year, {}).get(month-1) if month > 1 else NFP_DATA.get(year-1, {}).get(12, 200)
            surprise = (nfp_val - nfp_prev) / abs(nfp_prev) if nfp_prev and nfp_prev != 0 else 0
            events.append({
                'datetime': nfp_date.replace(hour=12, minute=30),
                'event': 'NFP', 'impact': 'HIGH',
                'actual': nfp_val, 'previous': nfp_prev, 'surprise': surprise
            })

            # CPI
            cpi_date = datetime(year, month, 12)
            cpi_val = CPI_DATA.get(year, {}).get(month, 2.0)
            cpi_prev = CPI_DATA.get(year, {}).get(month-1) if month > 1 else CPI_DATA.get(year-1, {}).get(12, 2.0)
            events.append({
                'datetime': cpi_date.replace(hour=12, minute=30),
                'event': 'CPI', 'impact': 'HIGH',
                'actual': cpi_val, 'previous': cpi_prev, 'surprise': cpi_val - (cpi_prev or 0)
            })

        # FOMC
        for date_str in FOMC_DATES.get(year, []):
            try:
                m, d = int(date_str.split('-')[0]), int(date_str.split('-')[1])
                events.append({
                    'datetime': datetime(year, m, d, 18, 0),
                    'event': 'FOMC', 'impact': 'HIGH',
                    'actual': 0, 'previous': 0, 'surprise': 0
                })
            except: pass
    return pd.DataFrame(events).sort_values('datetime')

calendar_df = create_economic_calendar()
print(f"  ✅ Calendar created: {len(calendar_df)} events")

# =============================================================================
# 🚀 STEP 6: LOAD GOLD DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Loading Gold data...")
print("=" * 70)

gold_df = pd.read_csv(GOLD_DATA_PATH, parse_dates=['Date'])
gold_df = gold_df.set_index('Date').sort_index()

print(f"  ✅ Loaded: {len(gold_df):,} bars")
print(f"  ✅ Period: {gold_df.index.min()} → {gold_df.index.max()}")

# Merge with calendar
def add_news_features(df, calendar, window=60):
    df = df.copy()
    df['news_event'] = 0
    df['news_impact'] = 0.0
    df['news_surprise'] = 0.0

    for _, ev in calendar.iterrows():
        mask = (df.index >= ev['datetime'] - timedelta(minutes=window)) & \
               (df.index <= ev['datetime'] + timedelta(minutes=window*2))
        df.loc[mask, 'news_event'] = 1
        df.loc[mask, 'news_impact'] = 1.0
        df.loc[mask, 'news_surprise'] = ev['surprise']
    return df

gold_df = add_news_features(gold_df, calendar_df)
print(f"  ✅ News features added: {gold_df['news_event'].sum():,} bars with news")

# =============================================================================
# 🚀 STEP 7: CREATE DATA SPLITS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: Creating data splits...")
print("=" * 70)

n = len(gold_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = gold_df.iloc[:train_end].copy()
df_val = gold_df.iloc[train_end:val_end].copy()
df_test = gold_df.iloc[val_end:].copy()

print(f"  Train: {len(df_train):,} bars ({df_train.index.min().date()} → {df_train.index.max().date()})")
print(f"  Val:   {len(df_val):,} bars ({df_val.index.min().date()} → {df_val.index.max().date()})")
print(f"  Test:  {len(df_test):,} bars ({df_test.index.min().date()} → {df_test.index.max().date()})")

# =============================================================================
# 🚀 STEP 8: TRADING ENVIRONMENT
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: Creating trading environment...")
print("=" * 70)

class OptimizedTradingEnv(gym.Env):
    def __init__(self, df, config, training=True):
        super().__init__()
        self.config = config
        self.training = training
        self.df_original = df.copy()
        self._prepare_features()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)
        self.returns_history = []
        self.reset()

    def _prepare_features(self):
        df = self.df_original.copy()

        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        df['rsi_raw'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['rsi'] = (df['rsi_raw'] - 50) / 25

        macd = ta.trend.MACD(df['Close'])
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['macd_norm'] = macd.macd() / (df['atr'] + 1e-8)
        df['macd_signal_norm'] = macd.macd_signal() / (df['atr'] + 1e-8)
        df['macd_hist_norm'] = macd.macd_diff() / (df['atr'] + 1e-8)

        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_position'] = ((df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)).clip(0, 1)

        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['price_sma20_ratio'] = (df['Close'] / df['sma_20']) - 1
        df['price_sma50_ratio'] = (df['Close'] / df['sma_50']) - 1

        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252 * 24 * 4)
        df['volatility_zscore'] = (df['volatility'] - df['volatility'].rolling(100).mean()) / (df['volatility'].rolling(100).std() + 1e-8)

        df['volume_ratio'] = np.clip(df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8), 0, 5)
        df['momentum_5'] = df['Close'].pct_change(5)
        df['momentum_20'] = df['Close'].pct_change(20)
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / (df['atr'] + 1e-8)

        if 'news_event' not in df.columns:
            df['news_event'] = 0
            df['news_impact'] = 0
            df['news_surprise'] = 0

        self.feature_columns = [
            'returns', 'log_returns', 'rsi', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
            'bb_position', 'price_sma20_ratio', 'price_sma50_ratio',
            'volatility_zscore', 'volume_ratio', 'momentum_5', 'momentum_20', 'trend_strength',
            'news_event', 'news_impact', 'news_surprise'
        ]

        df = df.dropna()
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(df[self.feature_columns])
        self.prices = df['Close'].values
        self.volatility = df['volatility'].fillna(0.2).values
        self.news_events = df['news_event'].values
        self.n_features = len(self.feature_columns) + 3

    def _apply_augmentation(self, features):
        if not self.training: return features
        augmented = features.copy()
        augmented += np.random.normal(0, self.config.price_noise, features.shape)
        if np.random.random() < self.config.dropout_prob:
            augmented[np.random.randint(0, len(features))] = 0
        return augmented

    def _calc_cost(self, price, is_news=False):
        cost = self.config.spread + self.config.commission
        slippage = self.config.slippage_base
        vol = self.volatility[self.current_step] if self.current_step < len(self.volatility) else 0.2
        slippage *= (1 + self.config.slippage_volatility_mult * min(vol, 1.0))
        if is_news: slippage *= self.config.slippage_news_mult
        return (cost + slippage) * price

    def _calc_reward(self, pnl, action):
        reward = pnl / self.config.initial_balance * self.config.reward_pnl_weight

        if len(self.returns_history) >= 20:
            rets = np.array(self.returns_history[-20:])
            if np.std(rets) > 0:
                reward += (np.mean(rets) / np.std(rets)) * self.config.reward_sharpe_weight * 0.1

        dd = self._calc_drawdown()
        if dd > 0.05: reward -= dd * self.config.reward_drawdown_penalty

        if action != 0 and getattr(self, 'bars_since_trade', 100) < 4:
            reward -= self.config.reward_overtrade_penalty

        if pnl > 0:
            self.consecutive_wins = getattr(self, 'consecutive_wins', 0) + 1
            if self.consecutive_wins >= 3: reward += self.config.reward_consistency_bonus
        else:
            self.consecutive_wins = 0

        return reward

    def _calc_drawdown(self):
        current = self.balance + self._unrealized_pnl()
        if current >= self.peak_value:
            self.peak_value = current
            return 0
        return (self.peak_value - current) / self.peak_value

    def _unrealized_pnl(self):
        if self.position == 0: return 0
        price = self.prices[self.current_step]
        if self.position > 0:
            return (price - self.entry_price) * self.position_size
        return (self.entry_price - price) * abs(self.position_size)

    def _check_stops(self):
        if self.position == 0: return False, None
        price = self.prices[self.current_step]
        pnl_pct = ((price - self.entry_price) / self.entry_price) if self.position > 0 else ((self.entry_price - price) / self.entry_price)
        if pnl_pct <= -self.config.stop_loss_pct: return True, 'sl'
        if pnl_pct >= self.config.take_profit_pct: return True, 'tp'
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
        return self._get_obs(), {}

    def _get_obs(self):
        features = self._apply_augmentation(self.features[self.current_step].copy())
        return np.concatenate([features, [self.position, self.total_pnl / self.config.initial_balance, self._calc_drawdown()]]).astype(np.float32)

    def step(self, action):
        price = self.prices[self.current_step]
        is_news = self.news_events[self.current_step] > 0 if self.current_step < len(self.news_events) else False
        pnl = 0
        reward = 0

        triggered, _ = self._check_stops()
        if triggered: pnl = self._close(price, is_news)

        if self._calc_drawdown() >= self.config.max_total_drawdown:
            if self.position != 0: pnl += self._close(price, is_news)
            reward -= 1.0
            self.current_step = len(self.prices) - 1
        else:
            if action == 1 and self.position <= 0:
                if self.position < 0: pnl = self._close(price, is_news)
                self._open(1, price, is_news)
            elif action == 2 and self.position >= 0:
                if self.position > 0: pnl = self._close(price, is_news)
                self._open(-1, price, is_news)

            reward = self._calc_reward(pnl, action)
            self.current_step += 1
            self.bars_since_trade += 1

        self.returns_history.append(pnl / self.config.initial_balance)
        done = self.current_step >= len(self.prices) - 1

        if done and self.position != 0:
            pnl = self._close(self.prices[self.current_step], False)
            reward += self._calc_reward(pnl, 0)

        info = {'total_pnl': self.total_pnl, 'position': self.position, 'drawdown': self._calc_drawdown(),
                'trade_count': self.trade_count, 'win_rate': self.win_count / max(1, self.trade_count)}
        return self._get_obs(), reward, done, False, info

    def _open(self, direction, price, is_news):
        self.balance -= self._calc_cost(price, is_news)
        self.position = direction
        self.position_size = self.config.max_position
        self.entry_price = price
        self.bars_since_trade = 0

    def _close(self, price, is_news):
        if self.position == 0: return 0
        cost = self._calc_cost(price, is_news)
        pnl = ((price - self.entry_price) if self.position > 0 else (self.entry_price - price)) * self.position_size - cost
        self.total_pnl += pnl
        self.balance += pnl
        self.trade_count += 1
        if pnl > 0: self.win_count += 1
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        return pnl

train_env = OptimizedTradingEnv(df_train, config, training=True)
val_env = OptimizedTradingEnv(df_val, config, training=False)
test_env = OptimizedTradingEnv(df_test, config, training=False)

print(f"  ✅ Environments created!")
print(f"     Observation: {train_env.observation_space.shape}")
print(f"     Actions: {train_env.action_space}")

# =============================================================================
# 🚀 STEP 9: TRAINING CALLBACK
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: Setting up training...")
print("=" * 70)

class TrainingCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, patience):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.best_sharpe = -np.inf
        self.no_improve = 0
        self.results = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            metrics = self._evaluate()
            self.results.append({'step': self.n_calls, **metrics})

            print(f"\n📊 Step {self.n_calls:,}: Sharpe={metrics['sharpe']:.2f}, MaxDD={metrics['max_dd']:.1%}, WinRate={metrics['win_rate']:.1%}")

            if metrics['sharpe'] > self.best_sharpe + 0.01:
                self.best_sharpe = metrics['sharpe']
                self.no_improve = 0
                self.model.save('models/best_model')
                print(f"   ⭐ New best! Sharpe: {self.best_sharpe:.2f}")
            else:
                self.no_improve += 1
                print(f"   No improvement ({self.no_improve}/{self.patience})")

            if self.no_improve >= self.patience:
                print(f"\n⚠️ Early stopping!")
                return False
        return True

    def _evaluate(self):
        obs, _ = self.eval_env.reset()
        done = False
        pv = [config.initial_balance]
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, info = self.eval_env.step(action)
            pv.append(config.initial_balance + info['total_pnl'])

        pv = np.array(pv)
        rets = np.diff(pv) / (pv[:-1] + 1e-8)
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252 * 24 * 4)
        peak = np.maximum.accumulate(pv)
        max_dd = np.max((peak - pv) / (peak + 1e-8))
        return {'sharpe': sharpe, 'max_dd': max_dd, 'total_return': (pv[-1]/pv[0])-1, 'win_rate': info['win_rate'], 'trades': info['trade_count']}

callback = TrainingCallback(val_env, EVAL_FREQ, EARLY_STOPPING_PATIENCE)

HYPERPARAMS = {
    'learning_rate': 3e-5, 'n_steps': 2048, 'batch_size': 128, 'n_epochs': 10,
    'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.05,
    'vf_coef': 0.5, 'max_grad_norm': 0.5,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device.upper()}")
print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")

# =============================================================================
# 🚀 STEP 10: TRAIN!
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10: TRAINING...")
print("=" * 70)

model = PPO('MlpPolicy', train_env, verbose=0, tensorboard_log='logs/', device=device, **HYPERPARAMS)

print("\n🚀 Training started!\n")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)
model.save('models/final_model')

print(f"\n✅ Training complete!")
print(f"   Best Sharpe: {callback.best_sharpe:.2f}")

# =============================================================================
# 🚀 STEP 11: FINAL EVALUATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 11: Final evaluation on TEST data...")
print("=" * 70)

best_model = PPO.load('models/best_model')
obs, _ = test_env.reset()
done = False
pv = [config.initial_balance]
actions = []

while not done:
    action, _ = best_model.predict(obs, deterministic=True)
    obs, _, done, _, info = test_env.step(action)
    pv.append(config.initial_balance + info['total_pnl'])
    actions.append(action)

pv = np.array(pv)
rets = np.diff(pv) / (pv[:-1] + 1e-8)
sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252 * 24 * 4)
down_rets = rets[rets < 0]
sortino = np.mean(rets) / (np.std(down_rets) + 1e-8) * np.sqrt(252 * 24 * 4) if len(down_rets) > 0 else 0
peak = np.maximum.accumulate(pv)
dd = (peak - pv) / (peak + 1e-8)
max_dd = np.max(dd)
win_rate = info['win_rate']
cum_ret = (pv[-1] / pv[0]) - 1

print(f"""
{'='*60}
                 FINAL RESULTS - TEST DATA
{'='*60}

  Sharpe Ratio:      {sharpe:>10.2f}  {'✅' if sharpe >= 1.0 else '⚠️'}
  Sortino Ratio:     {sortino:>10.2f}  {'✅' if sortino >= 1.5 else '⚠️'}
  Max Drawdown:      {max_dd:>10.1%}  {'✅' if max_dd <= 0.15 else '❌'}
  Win Rate:          {win_rate:>10.1%}  {'✅' if win_rate >= 0.50 else '⚠️'}
  Cumulative Return: {cum_ret:>10.1%}

  Initial Capital:   ${pv[0]:>10,.0f}
  Final Capital:     ${pv[-1]:>10,.0f}
  Profit/Loss:       ${pv[-1]-pv[0]:>+10,.0f}
  Total Trades:      {info['trade_count']:>10}

{'='*60}
""")

if sharpe >= 1.0 and max_dd <= 0.15:
    print("🎉 BOT READY FOR PAPER TRADING! 🎉")
else:
    print("⚠️ Bot needs more work")

# =============================================================================
# 🚀 STEP 12: SAVE AND DOWNLOAD
# =============================================================================
print("\n" + "=" * 70)
print("STEP 12: Saving results...")
print("=" * 70)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
axes[0].plot(pv, 'b-', lw=0.8)
axes[0].axhline(config.initial_balance, color='gray', ls='--')
axes[0].fill_between(range(len(pv)), pv, config.initial_balance, where=pv>=config.initial_balance, alpha=0.3, color='green')
axes[0].fill_between(range(len(pv)), pv, config.initial_balance, where=pv<config.initial_balance, alpha=0.3, color='red')
axes[0].set_title(f'Portfolio (Sharpe: {sharpe:.2f}, Return: {cum_ret:.1%})')
axes[0].grid(True, alpha=0.3)

axes[1].fill_between(range(len(dd)), -dd*100, 0, alpha=0.7, color='red')
axes[1].axhline(-config.max_total_drawdown*100, color='darkred', ls='--')
axes[1].set_title(f'Drawdown (Max: {max_dd:.1%})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/performance.png', dpi=150)
plt.show()

# Save metrics
pd.DataFrame([{'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd, 'win_rate': win_rate,
               'cum_return': cum_ret, 'trades': info['trade_count']}]).to_csv('results/metrics.csv', index=False)

print("✅ Results saved!")

# Download
try:
    from google.colab import files
    import shutil
    shutil.make_archive('trading_bot', 'zip', 'models')
    files.download('trading_bot.zip')
    files.download('results/performance.png')
    files.download('results/metrics.csv')
    print("📥 Files downloaded!")
except:
    print("📁 Files saved locally")

print("\n" + "=" * 70)
print("                    ✅ COMPLETE!")
print("=" * 70)
