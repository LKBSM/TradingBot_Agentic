# =============================================================================
# TRADING BOT - FULL CURRICULUM TRAINING SCRIPT FOR GOOGLE COLAB
# =============================================================================
#
# HOW TO USE ON GOOGLE COLAB:
# 1. Open Google Colab: https://colab.research.google.com/
# 2. Create a new notebook
# 3. In a cell, run:
#      !wget -q https://raw.githubusercontent.com/LKBSM/TradingBot_Agentic/main/scripts/colab_training_full.py
#      %run colab_training_full.py
# 4. Or copy-paste this ENTIRE script into a single cell and run
#
# IMPORTANT: Enable GPU before running!
# Runtime -> Change runtime type -> GPU (T4)
#
# Training: 4-phase curriculum (BASE -> ENRICHED -> SOFT -> PRODUCTION)
# Total timesteps: 1.5M (configurable)
# Estimated time: ~3-5 hours on T4 GPU
# =============================================================================

import os
import sys
import subprocess
import time

# =============================================================================
# STEP 1: INSTALL PACKAGES
# =============================================================================
print("=" * 70)
print("STEP 1: Installing packages...")
print("=" * 70)

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
    except Exception:
        print(f"Warning: Could not install {pkg}")

print("Packages installed!\n")

# =============================================================================
# STEP 2: CLONE REPOSITORY
# =============================================================================
print("=" * 70)
print("STEP 2: Cloning TradingBot_Agentic repository...")
print("=" * 70)

REPO_URL = "https://github.com/LKBSM/TradingBot_Agentic.git"
REPO_DIR = "TradingBot_Agentic"

# Detect if we're already inside the repo (re-run scenario)
if os.path.exists('config.py') and os.path.exists('src/environment/environment.py'):
    print(f"Already inside repository at {os.getcwd()}, pulling latest...")
    subprocess.run(["git", "pull", "--ff-only"], check=False)
elif os.path.exists(REPO_DIR):
    print(f"Repository already exists at {REPO_DIR}, pulling latest...")
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=False)
    os.chdir(REPO_DIR)
else:
    print(f"Cloning from {REPO_URL}...")
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL])
    os.chdir(REPO_DIR)

# Add repo to Python path so imports work
sys.path.insert(0, os.getcwd())

print(f"Working directory: {os.getcwd()}")
print("Repository ready!\n")

# =============================================================================
# STEP 3: IMPORTS
# =============================================================================
print("=" * 70)
print("STEP 3: Importing libraries...")
print("=" * 70)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
from stable_baselines3 import PPO

# Import project modules
import config
from src.training.unified_agentic_env import UnifiedAgenticEnv, TrainingMode
from src.training.curriculum_trainer import CurriculumTrainer, CurriculumConfig

# Check GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected - training will be slower")

# Create directories
for d in [config.DATA_DIR, config.MODEL_DIR, config.LOG_DIR, config.RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("Libraries imported!\n")

# =============================================================================
# STEP 4: DOWNLOAD DATA FROM GITHUB RELEASES
# =============================================================================
print("=" * 70)
print("STEP 4: Downloading training data from GitHub Releases...")
print("=" * 70)

GITHUB_REPO_URL = "https://github.com/LKBSM/TradingBot_Agentic"
GOLD_DATA_URL = f"{GITHUB_REPO_URL}/releases/latest/download/XAU_15MIN_2019_2025.csv"
ECON_CALENDAR_URL = f"{GITHUB_REPO_URL}/releases/latest/download/economic_calendar_2019_2025.csv"

gold_filepath = os.path.join(config.DATA_DIR, 'XAU_15MIN_2019_2025.csv')
econ_calendar_filepath = os.path.join(config.DATA_DIR, 'economic_calendar_2019_2025.csv')

def download_file(url, filepath, description):
    """Download a file using wget or requests."""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"{description} already exists: {filepath} ({size_mb:.1f} MB)")
        return True

    print(f"Downloading {description}...")
    try:
        subprocess.check_call(['wget', '-q', '-O', filepath, url])
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"Downloaded: {filepath} ({size_mb:.1f} MB)")
        return True
    except Exception:
        try:
            import requests as req
            resp = req.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"Downloaded: {filepath} ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            print(f"FAILED to download {description}: {e}")
            return False

download_file(GOLD_DATA_URL, gold_filepath, "Gold M15 OHLCV data")
download_file(ECON_CALENDAR_URL, econ_calendar_filepath, "Economic calendar")

if not os.path.exists(gold_filepath):
    raise FileNotFoundError(
        f"Gold data not found at {gold_filepath}. "
        f"Download manually from: {GITHUB_REPO_URL}/releases"
    )

print("Data ready!\n")

# =============================================================================
# STEP 5: LOAD AND PREPARE DATA
# =============================================================================
print("=" * 70)
print("STEP 5: Loading and preparing data...")
print("=" * 70)

# Load Gold data
gold_df = pd.read_csv(gold_filepath)
gold_df['Date'] = pd.to_datetime(gold_df['Date'])
gold_df.set_index('Date', inplace=True)
gold_df.sort_index(inplace=True)

print(f"Gold data loaded: {len(gold_df):,} bars")
print(f"Period: {gold_df.index.min()} -> {gold_df.index.max()}")
print(f"Price range: ${gold_df['Close'].min():.2f} - ${gold_df['Close'].max():.2f}")
print(f"Columns: {list(gold_df.columns)}")

# Load economic calendar (optional but recommended)
calendar_df = None
if os.path.exists(econ_calendar_filepath):
    try:
        calendar_df = pd.read_csv(econ_calendar_filepath)
        if 'datetime' in calendar_df.columns:
            calendar_df['datetime'] = pd.to_datetime(calendar_df['datetime'])
        elif 'date' in calendar_df.columns:
            calendar_df['datetime'] = pd.to_datetime(calendar_df['date'])
        print(f"Economic calendar loaded: {len(calendar_df)} events")
    except Exception as e:
        print(f"Warning: Could not load economic calendar: {e}")
        calendar_df = None
else:
    print("No economic calendar found (training will use simulated news events)")

print()

# =============================================================================
# STEP 6: MERGE NEWS FEATURES INTO PRICE DATA
# =============================================================================
print("=" * 70)
print("STEP 6: Merging news features into price data...")
print("=" * 70)

def add_news_features(df: pd.DataFrame, calendar: pd.DataFrame,
                      window_before: int = 60, window_after: int = 120) -> pd.DataFrame:
    """Add news event features to price data.

    For each economic event, marks bars within a time window with:
    - news_event: 1 if within window, 0 otherwise
    - news_impact: 1.0 for HIGH impact, 0.5 for medium
    - news_surprise: surprise factor
    - minutes_to_news: signed minutes until event
    """
    df = df.copy()
    df['news_event'] = 0
    df['news_impact'] = 0.0
    df['news_surprise'] = 0.0
    df['minutes_to_news'] = 0

    if calendar is None or len(calendar) == 0:
        print("No calendar data - news features set to zero")
        return df

    merged_count = 0
    for _, event in calendar.iterrows():
        event_time = event['datetime']
        window_start = event_time - timedelta(minutes=window_before)
        window_end = event_time + timedelta(minutes=window_after)

        mask = (df.index >= window_start) & (df.index <= window_end)
        if mask.any():
            merged_count += 1
            df.loc[mask, 'news_event'] = 1
            impact_val = 1.0
            if 'impact' in event:
                impact_val = 1.0 if event['impact'] == 'HIGH' else 0.5
            df.loc[mask, 'news_impact'] = impact_val

            surprise = event.get('surprise', 0.0)
            if pd.notna(surprise):
                df.loc[mask, 'news_surprise'] = float(surprise)

            for idx in df.loc[mask].index:
                minutes_diff = (event_time - idx).total_seconds() / 60
                current_val = df.loc[idx, 'minutes_to_news']
                if current_val == 0 or abs(minutes_diff) < abs(current_val):
                    df.loc[idx, 'minutes_to_news'] = minutes_diff

    return df

gold_df = add_news_features(gold_df, calendar_df)

news_bars = gold_df['news_event'].sum()
print(f"Data merged!")
print(f"  Bars with news events: {news_bars:,} ({news_bars/len(gold_df)*100:.1f}%)")
print()

# =============================================================================
# STEP 7: CREATE TRAIN/VAL/TEST SPLITS
# =============================================================================
print("=" * 70)
print("STEP 7: Creating chronological data splits...")
print("=" * 70)

n = len(gold_df)
train_end = int(n * config.TRAIN_RATIO)
val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

df_train = gold_df.iloc[:train_end].copy()
df_val = gold_df.iloc[train_end:val_end].copy()
df_test = gold_df.iloc[val_end:].copy()

print(f"Data splits created:")
print(f"  Train:      {len(df_train):>8,} bars ({df_train.index.min().date()} -> {df_train.index.max().date()})")
print(f"  Validation: {len(df_val):>8,} bars ({df_val.index.min().date()} -> {df_val.index.max().date()})")
print(f"  Test:       {len(df_test):>8,} bars ({df_test.index.min().date()} -> {df_test.index.max().date()})")
print()

# =============================================================================
# STEP 8: 4-PHASE CURRICULUM TRAINING
# =============================================================================
print("=" * 70)
print("STEP 8: Starting 4-phase Curriculum Training...")
print("=" * 70)
print()
print("Training Phases:")
print("  Phase 1 - BASE:       Pure market learning (agent signals zeroed)")
print("  Phase 2 - ENRICHED:   Agent signals as observation (no constraints)")
print("  Phase 3 - SOFT:       Soft penalties for rejected actions")
print("  Phase 4 - PRODUCTION: Full agent integration with hard constraints")
print()

# Configure curriculum
TOTAL_TIMESTEPS = 1_500_000  # 1.5M total steps across all phases

curriculum_config = CurriculumConfig(
    total_timesteps=TOTAL_TIMESTEPS,
    model_save_dir=os.path.join(config.MODEL_DIR, 'curriculum'),
    tensorboard_log_dir=os.path.join(config.LOG_DIR, 'curriculum'),
    eval_episodes=10,
    patience=3,
)

# PPO hyperparameters optimized for Gold M15
base_hyperparams = {
    'n_steps': 1024,
    'batch_size': 128,
    'gamma': 0.995,
    'learning_rate': 3e-4,
    'ent_coef': 0.01,
    'clip_range': 0.2,
    'gae_lambda': 0.95,
    'max_grad_norm': 0.5,
    'vf_coef': 0.5,
    'n_epochs': 5,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device.upper()}")
print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Observation space: 30 features x 20 lookback + 3 state + 20 agent signals = 623 dims")
print(f"Action space: 5 actions (HOLD, OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT)")
print()

# Create trainer
trainer = CurriculumTrainer(
    df_train=df_train,
    df_val=df_val,
    config=curriculum_config,
    base_hyperparams=base_hyperparams,
    economic_calendar=calendar_df,
    verbose=1,
)

# Train
print("Starting training...")
print("=" * 70)
start_time = time.time()

model, summary = trainer.train(seed=42)

elapsed = time.time() - start_time
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)

print()
print("=" * 70)
print(f"CURRICULUM TRAINING COMPLETE! ({hours}h {minutes}m)")
print("=" * 70)
print()

# Print phase summary
if 'phases_completed' in summary:
    for phase in summary['phases_completed']:
        print(f"  Phase {phase['phase']+1} ({phase['mode']}): "
              f"Best Sharpe = {phase.get('best_sharpe', 0):.2f}")

print()

# =============================================================================
# STEP 9: FINAL EVALUATION ON TEST DATA
# =============================================================================
print("=" * 70)
print("STEP 9: Final evaluation on TEST data (never seen during training)...")
print("=" * 70)

# Create test environment in PRODUCTION mode (full agent integration)
test_env = UnifiedAgenticEnv(
    df=df_test,
    mode=TrainingMode.PRODUCTION,
    economic_calendar=calendar_df,
    enable_logging=False,
)

# Load best model
best_model_path = os.path.join(config.MODEL_DIR, 'curriculum', 'best', 'best_model.zip')
if os.path.exists(best_model_path):
    best_model = PPO.load(best_model_path)
    print(f"Loaded best model from: {best_model_path}")
else:
    best_model = model
    print("Using final model for evaluation")

# Run evaluation
obs, info = test_env.reset()
done = False
portfolio_values = [info.get('net_worth', config.INITIAL_BALANCE)]
actions_taken = []

while not done:
    action, _ = best_model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(int(action))
    portfolio_values.append(info.get('net_worth', portfolio_values[-1]))
    actions_taken.append(int(action))
    done = done or truncated

# Calculate metrics
pv = np.array(portfolio_values)
returns = np.diff(pv) / (pv[:-1] + 1e-8)

# Sharpe Ratio (annualized for M15)
periods_per_year = 252 * 24 * 4  # Trading days * hours * bars/hour
sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(periods_per_year)

# Sortino Ratio
downside = returns[returns < 0]
sortino = np.mean(returns) / (np.std(downside) + 1e-8) * np.sqrt(periods_per_year) if len(downside) > 0 else 0

# Max Drawdown
peak = np.maximum.accumulate(pv)
drawdown = (peak - pv) / (peak + 1e-8)
max_dd = np.max(drawdown)

# Calmar Ratio
cum_return = (pv[-1] / pv[0]) - 1
annual_return = (pv[-1] / pv[0]) ** (periods_per_year / len(pv)) - 1 if len(pv) > 1 else 0
calmar = annual_return / max_dd if max_dd > 0 else 0

# Action distribution
unique_actions, action_counts = np.unique(actions_taken, return_counts=True)
action_dist = dict(zip(unique_actions.tolist(), action_counts.tolist()))

# Trade count (count OPEN_LONG + OPEN_SHORT)
trade_count = action_dist.get(1, 0) + action_dist.get(3, 0)

# Win rate from returns
win_rate = np.mean(returns > 0) if len(returns) > 0 else 0

# Print results
print()
print("=" * 70)
print("                    FINAL RESULTS - TEST DATA")
print("=" * 70)
print()
print(f"  Sharpe Ratio:      {sharpe:>10.2f}  {'PASS' if sharpe >= 1.0 else 'NEEDS WORK'} (target: > 1.0)")
print(f"  Sortino Ratio:     {sortino:>10.2f}  {'PASS' if sortino >= 1.5 else 'NEEDS WORK'} (target: > 1.5)")
print(f"  Calmar Ratio:      {calmar:>10.2f}  {'PASS' if calmar >= 1.0 else 'NEEDS WORK'} (target: > 1.0)")
print(f"  Max Drawdown:      {max_dd:>10.1%}  {'PASS' if max_dd <= 0.15 else 'FAIL'} (target: < 15%)")
print(f"  Win Rate:          {win_rate:>10.1%}  {'PASS' if win_rate >= 0.50 else 'NEEDS WORK'} (target: > 50%)")
print(f"  Cumulative Return: {cum_return:>10.1%}")
print()
print(f"  Initial Capital:   ${pv[0]:>10,.0f}")
print(f"  Final Capital:     ${pv[-1]:>10,.0f}")
print(f"  Profit/Loss:       ${pv[-1]-pv[0]:>+10,.0f}")
print(f"  Trades Opened:     {trade_count:>10}")
print()

# Action distribution
print("  Action Distribution:")
for action_id, count in sorted(action_dist.items()):
    name = config.ACTION_NAMES.get(action_id, f'ACTION_{action_id}')
    pct = count / len(actions_taken) * 100
    print(f"    {name:>15}: {count:>6} ({pct:.1f}%)")

print()
print("=" * 70)

# Verdict
if sharpe >= 1.0 and max_dd <= 0.15 and win_rate >= 0.45:
    print()
    print("=" * 70)
    print("   BOT IS READY FOR PAPER TRADING!")
    print("   Next step: 4 weeks of paper trading on MT5 demo account")
    print("=" * 70)
else:
    print()
    print("Bot needs more work. Suggestions:")
    if sharpe < 1.0:
        print("  - Sharpe too low: increase training steps or tune rewards")
    if max_dd > 0.15:
        print("  - Drawdown too high: strengthen risk management")
    if win_rate < 0.45:
        print("  - Win rate low: review features or strategy")

print()

# =============================================================================
# STEP 10: VISUALIZATIONS
# =============================================================================
print("=" * 70)
print("STEP 10: Creating visualizations...")
print("=" * 70)

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# 1. Portfolio Value
axes[0].plot(pv, 'b-', linewidth=0.8, label='Portfolio')
axes[0].axhline(y=pv[0], color='gray', linestyle='--', label='Initial')
axes[0].fill_between(range(len(pv)), pv, pv[0],
                     where=pv >= pv[0], alpha=0.3, color='green')
axes[0].fill_between(range(len(pv)), pv, pv[0],
                     where=pv < pv[0], alpha=0.3, color='red')
axes[0].set_title(f'Portfolio Value (Sharpe: {sharpe:.2f}, Return: {cum_return:.1%})', fontsize=14)
axes[0].set_ylabel('Value ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Drawdown
axes[1].fill_between(range(len(drawdown)), -drawdown*100, 0, alpha=0.7, color='red')
axes[1].axhline(y=-15, color='darkred', linestyle='--', label='Max DD Limit (15%)')
axes[1].set_title(f'Drawdown (Max: {max_dd:.1%})', fontsize=14)
axes[1].set_ylabel('Drawdown (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Action Distribution
action_labels = [config.ACTION_NAMES.get(i, f'A{i}') for i in range(config.NUM_ACTIONS)]
action_values = [action_dist.get(i, 0) for i in range(config.NUM_ACTIONS)]
colors = ['gray', 'green', 'lightgreen', 'red', 'lightsalmon']
bars = axes[2].bar(action_labels, action_values, color=colors)
axes[2].set_title('Actions Distribution (5-Action Space)', fontsize=14)
axes[2].set_ylabel('Count')
for bar, count in zip(bars, action_values):
    if count > 0:
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{count}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

results_dir = config.RESULTS_DIR
plt.savefig(os.path.join(results_dir, 'test_results.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"Visualization saved to {results_dir}/test_results.png")
print()

# =============================================================================
# STEP 11: SAVE AND DOWNLOAD
# =============================================================================
print("=" * 70)
print("STEP 11: Saving results and preparing download...")
print("=" * 70)

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
    'total_trades': trade_count,
    'total_timesteps': TOTAL_TIMESTEPS,
    'training_time_minutes': elapsed / 60,
    'device': device,
}

metrics_df = pd.DataFrame([metrics_dict])
metrics_df.to_csv(os.path.join(results_dir, 'final_metrics.csv'), index=False)
print(f"Final metrics saved to {results_dir}/final_metrics.csv")

# Save model paths
model_dir = os.path.join(config.MODEL_DIR, 'curriculum')
print(f"Models saved in: {model_dir}/")
for f in os.listdir(model_dir) if os.path.exists(model_dir) else []:
    fpath = os.path.join(model_dir, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath) / 1024 / 1024
        print(f"  {f}: {size:.1f} MB")

# Download files in Colab
try:
    import shutil
    shutil.make_archive('trading_bot_results', 'zip', '.', model_dir)

    from google.colab import files
    print("\nDownloading files...")
    files.download('trading_bot_results.zip')
    files.download(os.path.join(results_dir, 'test_results.png'))
    files.download(os.path.join(results_dir, 'final_metrics.csv'))
    print("Files downloaded!")
except ImportError:
    print("\nNot running on Colab - files saved locally")
except Exception as e:
    print(f"\nCould not auto-download: {e}")
    print(f"Files are in: {model_dir}/ and {results_dir}/")

print()
print("=" * 70)
print("                    TRAINING COMPLETE!")
print("=" * 70)
print(f"""
Summary:
- Training: 4-phase curriculum ({TOTAL_TIMESTEPS:,} steps, {hours}h {minutes}m)
- Best Model: {model_dir}/best/best_model.zip
- Final Model: {model_dir}/final_curriculum_model.zip
- Results: {results_dir}/test_results.png
- Metrics: {results_dir}/final_metrics.csv

Key Results:
- Sharpe Ratio: {sharpe:.2f}
- Max Drawdown: {max_dd:.1%}
- Cumulative Return: {cum_return:.1%}

Next Steps:
1. If Sharpe > 1.0 and MaxDD < 15%: Start paper trading
2. Paper trade for 4+ weeks on MT5 demo
3. If paper trading is profitable: Go live with 2% capital
""")
print("=" * 70)
