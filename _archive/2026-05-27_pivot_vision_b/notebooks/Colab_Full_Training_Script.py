# =============================================================================
# TRADING BOT - COLAB TRAINING SCRIPT (v3 — Sprint 14 Thin Driver)
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
# This script delegates all heavy lifting to SophisticatedTrainer, which runs:
#   Phase 1  Curriculum learning (BASE → PRODUCTION)
#   Phase 2  Ensemble training (diverse specialists)
#   Phase 3  Meta-learning (regime adaptation)
#   Phase 4  Final validation + quality gates + artifact packaging
#
# =============================================================================

# =============================================================================
# CONFIG — MODIFY THESE VALUES
# =============================================================================

GITHUB_USERNAME = "LKBSM"
GITHUB_REPO = "TradingBot_Agentic"
GITHUB_TOKEN = ""  # <-- PASTE YOUR GITHUB TOKEN HERE (required for private repos)
RELEASE_TAG = "v1.0"
DATA_FILENAME = "XAU_15MIN_2019_2025.csv"
GOLD_DATA_PATH = "data/XAU_15MIN_2019_2025.csv"

# Training overrides (set to None to use defaults from config.py)
TOTAL_TIMESTEPS = 2_000_000       # None → config.TOTAL_TIMESTEPS_PER_BOT
TRAINING_STRATEGY = "FULL_PIPELINE"  # CURRICULUM_ONLY | FULL_PIPELINE | etc.
ENSEMBLE_SEEDS = (42, 123, 456)   # Seeds for multi-seed ensemble

from datetime import datetime
TRAINING_NAME = f"GoldBot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# =============================================================================
# STEP 1: MOUNT GOOGLE DRIVE + INSTALL PACKAGES
# =============================================================================
print("=" * 70)
print("STEP 1: Mounting Google Drive and installing packages...")
print("=" * 70)

import subprocess
import sys
import os
import time

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_PATH = f"/content/drive/MyDrive/TradingBot_Results/{TRAINING_NAME}"
    os.makedirs(SAVE_PATH, exist_ok=True)
    for sub in ("models", "results", "logs", "checkpoints"):
        os.makedirs(f"{SAVE_PATH}/{sub}", exist_ok=True)
    print(f"  Google Drive mounted!")
    print(f"  Results will be saved to: {SAVE_PATH}")
    DRIVE_MOUNTED = True
except Exception as e:
    print(f"  Could not mount Google Drive: {e}")
    SAVE_PATH = "/content/results"
    for sub in ("models", "results", "logs", "checkpoints"):
        os.makedirs(f"{SAVE_PATH}/{sub}", exist_ok=True)
    DRIVE_MOUNTED = False

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

packages = [
    "stable-baselines3[extra]", "gymnasium", "pandas", "numpy",
    "matplotlib", "scikit-learn", "ta", "torch", "tensorboard",
    "rich", "pydantic",
]
for pkg in packages:
    try:
        install(pkg)
        print(f"  + {pkg}")
    except Exception as e:
        print(f"  x {pkg}: {e}")

print("\nPackages installed!")

# =============================================================================
# STEP 2: CLONE REPOSITORY AND DOWNLOAD DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Cloning repository and downloading data...")
print("=" * 70)

import requests

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

# Ensure repo root is on sys.path so config / src imports work
if clone_dir not in sys.path:
    sys.path.insert(0, clone_dir)

os.makedirs("data", exist_ok=True)

print(f"\n  Downloading Gold data from release {RELEASE_TAG}...")

api_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/releases/tags/{RELEASE_TAG}"
headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

response = requests.get(api_url, headers=headers)
if response.status_code == 200:
    release_data = response.json()
    assets = release_data.get("assets", [])
    data_asset = next((a for a in assets if a["name"] == DATA_FILENAME), None)
    if data_asset:
        download_url = data_asset["url"]
        download_headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/octet-stream"}
        print(f"  Downloading {DATA_FILENAME} ({data_asset['size'] / 1024 / 1024:.1f} MB)...")
        data_response = requests.get(download_url, headers=download_headers, stream=True)
        if data_response.status_code == 200:
            with open(GOLD_DATA_PATH, "wb") as f:
                for chunk in data_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  Gold data downloaded!")
        else:
            print(f"  ERROR downloading: {data_response.status_code}")
    else:
        print(f"  ERROR: Asset {DATA_FILENAME} not found in release")
else:
    print(f"  ERROR getting release: {response.status_code}")

if os.path.exists(GOLD_DATA_PATH):
    file_size = os.path.getsize(GOLD_DATA_PATH) / 1024 / 1024
    print(f"  Gold data ready: {file_size:.1f} MB")
else:
    raise FileNotFoundError(f"Gold data NOT found at: {GOLD_DATA_PATH}")

# =============================================================================
# STEP 3: IMPORTS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Importing libraries...")
print("=" * 70)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

import config as cfg
from src.training.sophisticated_trainer import (
    SophisticatedTrainer,
    SophisticatedTrainerConfig,
    TrainingStrategy,
)

print("Libraries imported!")

# =============================================================================
# STEP 4: LOAD & SPLIT DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Loading Gold data and creating splits...")
print("=" * 70)

gold_df = pd.read_csv(GOLD_DATA_PATH, parse_dates=['Date'])
gold_df = gold_df.set_index('Date').sort_index()
print(f"  Loaded: {len(gold_df):,} bars")

n = len(gold_df)
train_end = int(n * cfg.TRAIN_RATIO)
val_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))

df_train = gold_df.iloc[:train_end].copy()
df_val = gold_df.iloc[train_end:val_end].copy()
df_test = gold_df.iloc[val_end:].copy()

print(f"  Train: {len(df_train):,} bars ({cfg.TRAIN_RATIO:.0%})")
print(f"  Val:   {len(df_val):,} bars ({cfg.VAL_RATIO:.0%})")
print(f"  Test:  {len(df_test):,} bars ({cfg.TEST_RATIO:.0%})")

# =============================================================================
# STEP 5: CONFIGURE & RUN SOPHISTICATED TRAINER
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Configuring SophisticatedTrainer...")
print("=" * 70)

strategy = TrainingStrategy[TRAINING_STRATEGY]
timesteps = TOTAL_TIMESTEPS or cfg.TOTAL_TIMESTEPS_PER_BOT

trainer_config = SophisticatedTrainerConfig(
    strategy=strategy,
    total_timesteps=timesteps,
    base_save_dir=f"{SAVE_PATH}/models",
    tensorboard_log_dir=f"{SAVE_PATH}/logs",
    checkpoint_local_dir=f"/content/checkpoints",
    checkpoint_drive_dir=f"{SAVE_PATH}/checkpoints" if DRIVE_MOUNTED else None,
    use_feature_reducer=cfg.USE_PCA_REDUCTION,
    base_hyperparams=cfg.MODEL_HYPERPARAMETERS,
)

print(f"  Strategy: {strategy.name}")
print(f"  Timesteps: {timesteps:,}")
print(f"  Feature Reducer: {trainer_config.use_feature_reducer}")
print(f"  Saving to: {SAVE_PATH}")

trainer = SophisticatedTrainer(
    df_train=df_train,
    df_val=df_val,
    df_test=df_test,
    config=trainer_config,
)

# =============================================================================
# STEP 6: TRAIN (with crash-safe partial save)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: TRAINING (this will take a while)...")
print("=" * 70)
sys.stdout.flush()

import json, traceback

results = None
wf_results = None
passed = False
failures = []

try:
    results = trainer.train(seed=42)
    print("  Training complete!")

    # --- STEP 7: Walk-forward validation ---
    print("\n" + "=" * 70)
    print("STEP 7: Walk-forward validation...")
    print("=" * 70)

    wf_results = trainer.run_walk_forward(gold_df, wf_config=cfg.WALK_FORWARD_CONFIG)
    agg = wf_results["aggregate"]
    print(f"  Folds: {agg['n_folds']}")
    print(f"  Mean Sharpe: {agg['mean_sharpe']:.2f} (+/- {agg['std_sharpe']:.2f})")
    print(f"  Mean Win Rate: {agg['mean_win_rate']:.1%}")
    print(f"  Mean Max DD: {agg['mean_max_drawdown']:.1%}")

    # --- STEP 8: Quality gates ---
    print("\n" + "=" * 70)
    print("STEP 8: Quality gates check...")
    print("=" * 70)

    gate_metrics = {
        "sharpe_ratio": results.final_sharpe,
        "max_drawdown": results.final_max_drawdown,
        "win_rate": results.final_win_rate,
        "profit_factor": 0.0,
    }

    passed, failures = trainer.check_quality_gates(gate_metrics, gates=cfg.QUALITY_GATES)
    if passed:
        print("  ALL QUALITY GATES PASSED")
    else:
        print(f"  FAILED gates: {', '.join(failures)}")

    # --- STEP 9: Package production artifact ---
    print("\n" + "=" * 70)
    print("STEP 9: Packaging production artifact...")
    print("=" * 70)

    artifact_dir = f"{SAVE_PATH}/production_artifact"
    trainer.package_production_artifact(output_dir=artifact_dir, results=results)
    print(f"  Artifact saved to: {artifact_dir}")

except KeyboardInterrupt:
    print("\n  INTERRUPTED by user. Saving partial results...")
except Exception as exc:
    print(f"\n  ERROR during training: {exc}")
    traceback.print_exc()
    print("  Saving partial results...")
finally:
    # Always save whatever we have — critical for Colab disconnects
    partial = {
        "training_name": TRAINING_NAME,
        "completed": results is not None,
        "walk_forward_done": wf_results is not None,
        "quality_gate_passed": passed,
        "quality_gate_failures": failures,
        "error": None,
    }
    if results is not None:
        partial.update({
            "strategy": str(results.strategy),
            "final_sharpe": results.final_sharpe,
            "final_max_drawdown": results.final_max_drawdown,
            "final_win_rate": results.final_win_rate,
            "training_hours": results.training_duration_seconds / 3600,
        })
    with open(f"{SAVE_PATH}/results/partial_results.json", "w") as f:
        json.dump(partial, f, indent=2, default=str)
    print(f"  Partial results saved to: {SAVE_PATH}/results/partial_results.json")

    # Copy checkpoints to Drive if training was interrupted mid-way
    if DRIVE_MOUNTED and os.path.exists("/content/checkpoints"):
        import shutil
        drive_ckpt = f"{SAVE_PATH}/checkpoints"
        for fname in os.listdir("/content/checkpoints"):
            src_file = f"/content/checkpoints/{fname}"
            dst_file = f"{drive_ckpt}/{fname}"
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
        print(f"  Checkpoints synced to Drive: {drive_ckpt}")

# =============================================================================
# STEP 10: FINAL RESULTS
# =============================================================================
if results is not None:
    agg = wf_results["aggregate"] if wf_results else {}
    print(f"\n{'='*60}")
    print(f"                 FINAL RESULTS — TEST DATA")
    print(f"{'='*60}")
    print(f"  Strategy:          {results.strategy}")
    print(f"  Sharpe Ratio:      {results.final_sharpe:>10.2f}")
    print(f"  Max Drawdown:      {results.final_max_drawdown:>10.1%}")
    print(f"  Win Rate:          {results.final_win_rate:>10.1%}")
    print(f"  Cumulative Return: {results.final_cumulative_return:>10.1%}")
    print(f"  Quality Gate:      {'PASSED' if passed else 'FAILED'}")
    if agg:
        print(f"  WF Mean Sharpe:    {agg['mean_sharpe']:>10.2f}")
        print(f"  WF Worst Sharpe:   {agg['worst_sharpe']:>10.2f}")
    print(f"  Ensemble:          {results.has_ensemble}")
    print(f"  Meta-Adapter:      {results.has_meta_adapter}")
    print(f"{'='*60}")

    # Save summary CSV
    row = {
        'strategy': results.strategy,
        'sharpe': results.final_sharpe,
        'max_dd': results.final_max_drawdown,
        'win_rate': results.final_win_rate,
        'cum_return': results.final_cumulative_return,
        'quality_gate': 'PASSED' if passed else 'FAILED',
        'training_hours': results.training_duration_seconds / 3600,
    }
    if agg:
        row['wf_mean_sharpe'] = agg['mean_sharpe']
    pd.DataFrame([row]).to_csv(f'{SAVE_PATH}/results/metrics.csv', index=False)

print(f"""
{'='*70}
                    COMPLETE!
{'='*70}

Results saved to: {SAVE_PATH}

   My Drive > TradingBot_Results > {TRAINING_NAME}

{'='*70}
""")
