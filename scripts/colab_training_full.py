

import os
import sys
import subprocess
import time
import shutil

# =============================================================================
# STEP 0: MOUNT GOOGLE DRIVE (crash-safe storage)
# =============================================================================
print("=" * 70)
print("STEP 0: Mounting Google Drive for crash-safe storage...")
print("=" * 70)

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# Create persistent directory on Google Drive
DRIVE_BASE = '/content/drive/MyDrive/TradingBot_Training'
DRIVE_MODELS = os.path.join(DRIVE_BASE, 'models')
DRIVE_CHECKPOINTS = os.path.join(DRIVE_BASE, 'checkpoints')
DRIVE_RESULTS = os.path.join(DRIVE_BASE, 'results')
DRIVE_LOGS = os.path.join(DRIVE_BASE, 'logs')

for d in [DRIVE_BASE, DRIVE_MODELS, DRIVE_CHECKPOINTS, DRIVE_RESULTS, DRIVE_LOGS]:
    os.makedirs(d, exist_ok=True)

print(f"Google Drive mounted!")
print(f"All training outputs saved to: {DRIVE_BASE}")
print()

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
# STEP 2: CLONE REPOSITORY (Sprint 15: Pinned to verified commit)
# =============================================================================
# SECURITY: Clone a specific verified commit to prevent supply-chain attacks.
# After each release, update VERIFIED_COMMIT to the new commit hash.
# The script will verify critical file checksums after checkout.
# =============================================================================
print("=" * 70)
print("STEP 2: Cloning TradingBot_Agentic repository (pinned commit)...")
print("=" * 70)

import hashlib

REPO_URL = "https://github.com/LKBSM/TradingBot_Agentic.git"
REPO_DIR = "TradingBot_Agentic"

# Sprint 15: Pin to a verified commit hash
# UPDATE THIS after each release — run: git rev-parse HEAD
VERIFIED_COMMIT = "08d8c6d"  # v4: DSR reward overhaul + 18 institutional-grade fixes

# Sprint 15: SHA-256 checksums of critical files (update with each release)
# Generate with: python -c "import hashlib; print(hashlib.sha256(open('file','rb').read()).hexdigest())"
# Set to None to skip checksum verification (first-time setup)
CRITICAL_FILE_CHECKSUMS = None  # Set to dict after first verified deployment
# Example when configured:
# CRITICAL_FILE_CHECKSUMS = {
#     'config.py': 'abc123...',
#     'src/environment/environment.py': 'def456...',
#     'src/environment/risk_manager.py': 'ghi789...',
# }

def _verify_checksums(base_dir: str, checksums: dict) -> None:
    """Verify SHA-256 checksums of critical files after checkout."""
    if checksums is None:
        print("  Checksum verification skipped (CRITICAL_FILE_CHECKSUMS=None)")
        print("  To enable: run the checksum generator and update this script")
        return

    print(f"  Verifying {len(checksums)} critical file checksums...")
    for filepath, expected_hash in checksums.items():
        full_path = os.path.join(base_dir, filepath)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"SECURITY: Critical file missing: {filepath}")
        actual = hashlib.sha256(open(full_path, 'rb').read()).hexdigest()
        if actual != expected_hash:
            raise RuntimeError(
                f"SECURITY: File {filepath} has been tampered with!\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual}\n"
                f"  This may indicate a supply-chain compromise."
            )
        print(f"  OK: {filepath}")
    print("  All checksums verified!")

# Detect if we're already inside the repo (re-run scenario)
if os.path.exists('config.py') and os.path.exists('src/environment/environment.py'):
    print(f"Already inside repository at {os.getcwd()}")
    # Verify we're on the correct commit
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    if not current_commit.startswith(VERIFIED_COMMIT[:7]):
        print(f"  WARNING: Current commit {current_commit} != verified {VERIFIED_COMMIT}")
        print(f"  Checking out verified commit...")
        subprocess.check_call(["git", "fetch", "origin"])
        subprocess.check_call(["git", "checkout", VERIFIED_COMMIT])
    else:
        print(f"  Commit verified: {current_commit}")
    _verify_checksums(os.getcwd(), CRITICAL_FILE_CHECKSUMS)
elif os.path.exists(REPO_DIR):
    print(f"Repository already exists at {REPO_DIR}")
    subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "origin"])
    subprocess.check_call(["git", "-C", REPO_DIR, "checkout", VERIFIED_COMMIT])
    os.chdir(REPO_DIR)
    _verify_checksums(os.getcwd(), CRITICAL_FILE_CHECKSUMS)
else:
    print(f"Cloning from {REPO_URL}...")
    subprocess.check_call(["git", "clone", REPO_URL])
    os.chdir(REPO_DIR)
    print(f"Checking out verified commit: {VERIFIED_COMMIT}")
    subprocess.check_call(["git", "checkout", VERIFIED_COMMIT])
    _verify_checksums(os.getcwd(), CRITICAL_FILE_CHECKSUMS)

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
import logging
logging.getLogger('jupyter_client').setLevel(logging.CRITICAL)
logging.getLogger('src.environment.environment').setLevel(logging.ERROR)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Import project modules
import config
from src.training.unified_agentic_env import UnifiedAgenticEnv, TrainingMode
from src.training.curriculum_trainer import (
    CurriculumTrainer, CurriculumConfig, CurriculumCallback
)

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
        # Normalize column names to lowercase
        calendar_df.columns = [c.lower() for c in calendar_df.columns]
        # Find the datetime column (could be 'date', 'datetime', etc.)
        date_col = None
        for col in ['datetime', 'date', 'time']:
            if col in calendar_df.columns:
                date_col = col
                break
        if date_col:
            calendar_df['datetime'] = pd.to_datetime(calendar_df[date_col])
        else:
            raise ValueError(f"No date column found in: {list(calendar_df.columns)}")
        print(f"Economic calendar loaded: {len(calendar_df)} events")
        print(f"  Columns: {list(calendar_df.columns)}")
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
    """Add news event features to price data."""
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
            if 'impact' in event.index:
                impact_str = str(event['impact']).upper()
                impact_val = 1.0 if impact_str == 'HIGH' else 0.5
            df.loc[mask, 'news_impact'] = impact_val

            surprise = event.get('surprise', 0.0) if 'surprise' in event.index else 0.0
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
# STEP 8: 4-PHASE CURRICULUM TRAINING (CRASH-SAFE with Google Drive)
# =============================================================================
print("=" * 70)
print("STEP 8: Starting 4-phase Curriculum Training (CRASH-SAFE)...")
print("=" * 70)
print()
print("Training Phases:")
print("  Phase 1 - BASE:       Pure market learning (agent signals zeroed)")
print("  Phase 2 - ENRICHED:   Agent signals as observation (no constraints)")
print("  Phase 3 - SOFT:       Soft penalties for rejected actions")
print("  Phase 4 - PRODUCTION: Full agent integration with hard constraints")
print()
print(f"All checkpoints saved to Google Drive: {DRIVE_BASE}")
print()

# Configure curriculum
# 2M steps: ~12x coverage of 170K-bar dataset. 3M caused overfitting with broken rewards.
# With fixed short accounting, 2M is sufficient + early stopping catches sweet spot.
TOTAL_TIMESTEPS = 2_000_000  # 2M total steps across all phases

# Use Google Drive for model storage (crash-safe)
curriculum_config = CurriculumConfig(
    total_timesteps=TOTAL_TIMESTEPS,
    model_save_dir=DRIVE_MODELS,
    tensorboard_log_dir=DRIVE_LOGS,
    eval_episodes=10,
    patience=3,
)

# PPO hyperparameters optimized for Gold M15 (623-dim obs space)
base_hyperparams = {
    'n_steps': 4096,          # v4: More episodes per rollout (4096/200 = ~20 episodes, was ~4)
    'batch_size': 256,        # Larger batches for stable gradients
    'gamma': 0.995,
    'learning_rate': 2e-4,    # Gentler LR for larger network
    'ent_coef': 0.01,
    'clip_range': 0.2,
    'gae_lambda': 0.95,
    'max_grad_norm': 0.5,
    'vf_coef': 0.5,
    'n_epochs': 5,
    'policy_kwargs': {
        'net_arch': dict(pi=[256, 128], vf=[256, 128]),  # v4: Separate policy/value heads (was shared [512, 256])
        'activation_fn': torch.nn.Tanh,  # Matches bounded obs space [-10, 10]
    },
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device.upper()}")
print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Network architecture: pi=[256,128] vf=[256,128] separate heads, Tanh")
print(f"Observation space: 23 features x 20 lookback + 8 state + 20 agent signals = 488 dims")
print(f"Action space: 5 actions (HOLD, OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT)")
print()


# =========================================================================
# TRAINING PROGRESS CALLBACK (detailed live logs)
# =========================================================================
class TrainingProgressCallback(BaseCallback):
    """Logs detailed training progress: steps, %, ETA, speed, phase, reward."""

    def __init__(
        self,
        total_timesteps: int,
        log_freq: int = 10_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_freq = log_freq
        self._start_time = None
        self._last_log_step = 0
        self._last_log_time = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._last_log_time = self._start_time
        print()
        print(f"{'Step':>10} | {'Progress':>8} | {'Speed':>12} | {'Elapsed':>10} | {'ETA':>10} | {'Avg Reward':>10} | Phase")
        print("-" * 90)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log_step >= self.log_freq:
            now = time.time()
            elapsed = now - self._start_time
            interval = now - self._last_log_time

            # Speed (steps per second)
            steps_in_interval = self.num_timesteps - self._last_log_step
            speed = steps_in_interval / interval if interval > 0 else 0

            # Progress
            pct = self.num_timesteps / self.total_timesteps * 100
            remaining_steps = self.total_timesteps - self.num_timesteps

            # ETA
            if speed > 0:
                eta_seconds = remaining_steps / speed
                eta_h = int(eta_seconds // 3600)
                eta_m = int((eta_seconds % 3600) // 60)
                eta_s = int(eta_seconds % 60)
                eta_str = f"{eta_h}h{eta_m:02d}m{eta_s:02d}s"
            else:
                eta_str = "calculating"

            # Elapsed
            el_h = int(elapsed // 3600)
            el_m = int((elapsed % 3600) // 60)
            el_s = int(elapsed % 60)
            elapsed_str = f"{el_h}h{el_m:02d}m{el_s:02d}s"

            # Average reward from recent episodes
            avg_reward = 0.0
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                recent = list(self.model.ep_info_buffer)[-10:]
                avg_reward = np.mean([ep['r'] for ep in recent])

            # Current phase (from curriculum callback if available)
            phase_str = "---"
            if hasattr(self, '_curriculum_cb') and self._curriculum_cb is not None:
                cb = self._curriculum_cb
                phase_idx = cb.current_phase_idx + 1
                phase_name = cb.current_phase.mode.name
                phase_pct = cb.phase_timesteps / cb.current_phase.timesteps * 100 if cb.current_phase.timesteps > 0 else 0
                phase_str = f"P{phase_idx}/{len(cb.config.phases)} {phase_name} ({phase_pct:.0f}%)"

            print(
                f"{self.num_timesteps:>10,} | "
                f"{pct:>6.1f}%  | "
                f"{speed:>8,.0f} st/s | "
                f"{elapsed_str:>10} | "
                f"{eta_str:>10} | "
                f"{avg_reward:>+10.2f} | "
                f"{phase_str}"
            )

            self._last_log_step = self.num_timesteps
            self._last_log_time = now

        return True


# =========================================================================
# GOOGLE DRIVE CHECKPOINT CALLBACK (saves every N steps)
# =========================================================================
class DriveCheckpointCallback(BaseCallback):
    """Saves model checkpoints to Google Drive every N steps.

    If Colab crashes, training can resume from the last checkpoint.
    """

    def __init__(
        self,
        save_freq: int = 50_000,
        drive_path: str = DRIVE_CHECKPOINTS,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.drive_path = drive_path
        self._last_save_step = 0
        os.makedirs(drive_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_step >= self.save_freq:
            self._last_save_step = self.num_timesteps

            # Save checkpoint to Google Drive
            checkpoint_path = os.path.join(
                self.drive_path,
                f"checkpoint_{self.num_timesteps:07d}.zip"
            )
            self.model.save(checkpoint_path)

            # Also save as "latest" for easy resume
            latest_path = os.path.join(self.drive_path, "latest_checkpoint.zip")
            self.model.save(latest_path)

            # Save progress info (include version for crash recovery validation)
            progress_path = os.path.join(self.drive_path, "progress.txt")
            with open(progress_path, 'w') as f:
                f.write(f"timesteps={self.num_timesteps}\n")
                f.write(f"checkpoint={checkpoint_path}\n")
                f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"version={TRAINING_VERSION}\n")

            if self.verbose > 0:
                print(f"  >> SAVED TO DRIVE: {self.num_timesteps:,} steps -> Google Drive")

        return True


# =========================================================================
# CHECK FOR EXISTING CHECKPOINT (resume after crash)
# =========================================================================
# IMPORTANT: v3 marks models trained with fixed short position accounting + scaler pipeline.
# v2 had catastrophic short net_worth double-counting (Sharpe -32.83).
# Old v1/v2 checkpoints MUST NOT be resumed — they learned broken behavior.
TRAINING_VERSION = "v4_dsr_reward"
resume_from = None
latest_checkpoint = os.path.join(DRIVE_CHECKPOINTS, "latest_checkpoint.zip")
progress_file = os.path.join(DRIVE_CHECKPOINTS, "progress.txt")

if os.path.exists(latest_checkpoint):
    saved_steps = 0
    saved_version = None
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            for line in f:
                if line.startswith('timesteps='):
                    saved_steps = int(line.strip().split('=')[1])
                if line.startswith('version='):
                    saved_version = line.strip().split('=')[1]

    if saved_version != TRAINING_VERSION:
        print(f"OLD CHECKPOINT DETECTED (version={saved_version}). Clearing and starting fresh.")
        print(f"  Reason: Reward function was fixed — old model is incompatible.")
        # Clear old checkpoints
        for f in os.listdir(DRIVE_CHECKPOINTS):
            os.remove(os.path.join(DRIVE_CHECKPOINTS, f))
        # Clear old models
        for f in os.listdir(DRIVE_MODELS):
            fpath = os.path.join(DRIVE_MODELS, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
    elif saved_steps > 0 and saved_steps < TOTAL_TIMESTEPS:
        print(f"CRASH RECOVERY: Found checkpoint at {saved_steps:,} / {TOTAL_TIMESTEPS:,} steps")
        print(f"  Resuming training from: {latest_checkpoint}")
        resume_from = latest_checkpoint
    elif saved_steps >= TOTAL_TIMESTEPS:
        print(f"TRAINING ALREADY COMPLETE ({saved_steps:,} steps). Skipping to evaluation.")
        resume_from = "COMPLETE"
    else:
        print("Found checkpoint but no progress info. Starting fresh.")
else:
    print("No previous checkpoint found. Starting fresh training.")

print()


# =========================================================================
# CREATE TRAINER AND RUN
# =========================================================================
if resume_from != "COMPLETE":
    # Create trainer
    trainer = CurriculumTrainer(
        df_train=df_train,
        df_val=df_val,
        config=curriculum_config,
        base_hyperparams=base_hyperparams,
        economic_calendar=calendar_df,
        verbose=1,
    )

    # Train (with resume support)
    print("Starting training...")
    print("=" * 70)
    start_time = time.time()

    # Override trainer.train() to add our Drive checkpoint callback
    # and resume support
    import logging
    _logger = logging.getLogger(__name__)

    # Create training environment
    trainer.env = trainer._create_env(df_train, TrainingMode.BASE)

    # Extract the fitted scaler from training env for reuse by val/test
    # CRITICAL: This prevents data leakage — scaler must be fit on train data ONLY.
    # Previously, each env fitted its own scaler on its own data, causing distribution
    # mismatch between train/val/test feature spaces.
    train_scaler = trainer.env._base_env.scaler
    print(f"Scaler fitted on training data (source: {trainer.env._base_env._scaler_source})")

    # Create reward shaper
    from src.training.advanced_reward_shaper import AdvancedRewardShaper
    trainer.reward_shaper = AdvancedRewardShaper(
        initial_balance=trainer.env._base_env.initial_balance,
        weights=curriculum_config.phases[0].reward_weights
    )

    # Create or load model
    if resume_from and resume_from != "COMPLETE":
        print(f"Loading model from checkpoint: {resume_from}")
        trainer.model = PPO.load(resume_from, env=trainer.env, device=device)
    else:
        trainer.model = PPO(
            'MlpPolicy',
            trainer.env,
            verbose=0,
            seed=42,
            device=device,
            tensorboard_log=DRIVE_LOGS,
            **base_hyperparams
        )

    # v4: EWC regularization — prevents catastrophic forgetting during phase transitions
    from src.training.ewc_regularization import EWCCallback
    ewc_callback = EWCCallback(ewc_lambda=1000.0, fisher_samples=2048, verbose=1)

    # Create callbacks
    curriculum_callback = CurriculumCallback(
        curriculum_config=curriculum_config,
        env=trainer.env,
        reward_shaper=trainer.reward_shaper,
        base_hyperparams=base_hyperparams,
        ewc_callback=ewc_callback,
        verbose=1
    )

    drive_checkpoint_cb = DriveCheckpointCallback(
        save_freq=50_000,  # Save to Drive every 50K steps
        drive_path=DRIVE_CHECKPOINTS,
        verbose=1,
    )

    progress_cb = TrainingProgressCallback(
        total_timesteps=TOTAL_TIMESTEPS,
        log_freq=10_000,  # Log every 10K steps
        verbose=1,
    )
    # Link progress callback to curriculum callback for phase info
    progress_cb._curriculum_cb = curriculum_callback

    # Validation environment — uses training scaler to avoid data leakage
    # Also uses ENRICHED mode (not PRODUCTION) to avoid random mock agent constraints
    from stable_baselines3.common.callbacks import EvalCallback
    env_val = trainer._create_env(df_val, TrainingMode.ENRICHED,
                                  pre_fitted_scaler=train_scaler)

    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=os.path.join(DRIVE_MODELS, 'best'),
        log_path=os.path.join(DRIVE_LOGS, 'eval'),
        eval_freq=25_000,
        n_eval_episodes=curriculum_config.eval_episodes,
        deterministic=True,
        render=False,
        verbose=0  # Quiet - progress_cb handles logging
    )

    # Train with all callbacks
    remaining_steps = TOTAL_TIMESTEPS
    if resume_from and resume_from != "COMPLETE":
        # Read how many steps were already done
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                for line in f:
                    if line.startswith('timesteps='):
                        done_steps = int(line.strip().split('=')[1])
                        remaining_steps = TOTAL_TIMESTEPS - done_steps
                        print(f"Remaining steps: {remaining_steps:,}")

    trainer.model.learn(
        total_timesteps=remaining_steps,
        callback=[curriculum_callback, progress_cb, drive_checkpoint_cb, eval_callback],
        reset_num_timesteps=resume_from is None,
    )

    # Save final model to Google Drive
    final_path = os.path.join(DRIVE_MODELS, 'final_curriculum_model.zip')
    trainer.model.save(final_path)
    print(f"Final model saved to Google Drive: {final_path}")

    # Save completion marker (include version for crash recovery validation)
    with open(os.path.join(DRIVE_CHECKPOINTS, "progress.txt"), 'w') as f:
        f.write(f"timesteps={TOTAL_TIMESTEPS}\n")
        f.write(f"status=COMPLETE\n")
        f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"version={TRAINING_VERSION}\n")

    model = trainer.model
    summary = curriculum_callback.get_training_summary()

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

else:
    elapsed = 0
    hours = 0
    minutes = 0
    print("Training was already completed. Loading final model...")


# =============================================================================
# STEP 9: FINAL EVALUATION ON TEST DATA
# =============================================================================
print("=" * 70)
print("STEP 9: Final evaluation on TEST data (never seen during training)...")
print("=" * 70)

# Create test environment in ENRICHED mode (signals visible, no hard constraints)
# FIX 1: Use ENRICHED instead of PRODUCTION — mock agents randomly block trades
#         in PRODUCTION mode, degrading evaluation metrics unfairly.
# FIX 2: Pass training scaler to prevent data leakage (test features must be
#         scaled with the same min/max as training features).
# FIX 3: training_mode=False so Kelly criterion properly gates trades (no floor).
test_env = UnifiedAgenticEnv(
    df=df_test,
    mode=TrainingMode.ENRICHED,
    economic_calendar=calendar_df,
    enable_logging=False,
    pre_fitted_scaler=train_scaler,
    training_mode=False,
)

# Load best model from Google Drive
best_model_path = os.path.join(DRIVE_MODELS, 'best', 'best_model.zip')
final_model_path = os.path.join(DRIVE_MODELS, 'final_curriculum_model.zip')

if os.path.exists(best_model_path):
    best_model = PPO.load(best_model_path)
    print(f"Loaded best model from: {best_model_path}")
elif os.path.exists(final_model_path):
    best_model = PPO.load(final_model_path)
    print(f"Loaded final model from: {final_model_path}")
elif resume_from != "COMPLETE":
    best_model = model
    print("Using final model from training")
else:
    raise FileNotFoundError(f"No model found in {DRIVE_MODELS}")

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
# STEP 10: VISUALIZATIONS (saved to Google Drive)
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

# Save to Google Drive
chart_path = os.path.join(DRIVE_RESULTS, 'test_results.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Visualization saved to Google Drive: {chart_path}")
print()

# =============================================================================
# STEP 11: SAVE ALL RESULTS TO GOOGLE DRIVE
# =============================================================================
print("=" * 70)
print("STEP 11: Saving all results to Google Drive...")
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
    'training_time_minutes': elapsed / 60 if elapsed else 0,
    'device': device,
}

metrics_df = pd.DataFrame([metrics_dict])
metrics_path = os.path.join(DRIVE_RESULTS, 'final_metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"Metrics saved: {metrics_path}")

# Save portfolio values for later analysis
portfolio_df = pd.DataFrame({
    'step': range(len(pv)),
    'portfolio_value': pv,
})
portfolio_path = os.path.join(DRIVE_RESULTS, 'portfolio_values.csv')
portfolio_df.to_csv(portfolio_path, index=False)
print(f"Portfolio values saved: {portfolio_path}")

# List all saved files
print()
print("=" * 70)
print(f"ALL FILES SAVED TO GOOGLE DRIVE: {DRIVE_BASE}")
print("=" * 70)
print()

for root, dirs, files in os.walk(DRIVE_BASE):
    level = root.replace(DRIVE_BASE, '').count(os.sep)
    indent = '  ' * level
    folder = os.path.basename(root)
    print(f"{indent}{folder}/")
    for file in sorted(files):
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath) / 1024 / 1024
        print(f"{indent}  {file} ({size:.1f} MB)")

print()
print("=" * 70)
print("                    TRAINING COMPLETE!")
print("=" * 70)
print(f"""
Summary:
- Training: 4-phase curriculum ({TOTAL_TIMESTEPS:,} steps)
- All models saved to: {DRIVE_MODELS}
- All checkpoints saved to: {DRIVE_CHECKPOINTS}
- Results saved to: {DRIVE_RESULTS}
- TensorBoard logs: {DRIVE_LOGS}

Google Drive Location: MyDrive/TradingBot_Training/

Key Results:
- Sharpe Ratio: {sharpe:.2f}
- Max Drawdown: {max_dd:.1%}
- Cumulative Return: {cum_return:.1%}

CRASH RECOVERY: If Colab crashed during training, just re-run this cell.
Training will automatically resume from the last checkpoint on Google Drive.

Next Steps:
1. If Sharpe > 1.0 and MaxDD < 15%: Start paper trading
2. Paper trade for 4+ weeks on MT5 demo
3. If paper trading is profitable: Go live with 2% capital
""")
print("=" * 70)
