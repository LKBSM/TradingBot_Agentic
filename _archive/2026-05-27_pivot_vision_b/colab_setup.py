#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TRADING BOT - GOOGLE COLAB SETUP SCRIPT
================================================================================
This script sets up everything needed to train the trading bot on Google Colab.

Features:
- Auto-mount Google Drive
- Clone/update repository
- Download dataset (141K+ bars of XAU 15-min data)
- Optimize configuration for Colab
- Verify all dependencies
- Prepare for training

Usage on Colab:
    !wget -q https://raw.githubusercontent.com/LKBSM/TradingBotNew/main/colab_setup.py
    %run colab_setup.py

Or copy the entire script into a Colab cell and run it.
================================================================================
"""

import os
import sys
import shutil
import subprocess

print("=" * 80)
print("   TRADING BOT - COLAB SETUP v2.0")
print("   Production Training Environment")
print("=" * 80)
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: MOUNT GOOGLE DRIVE
# ═══════════════════════════════════════════════════════════════════════════════

print("☁️  STEP 1/8: Mounting Google Drive")
print("-" * 80)

try:
    from google.colab import drive

    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive', force_remount=False)

    # Create backup folder structure
    drive_root = '/content/drive/MyDrive/TradingBot_Production'
    folders = [
        drive_root,
        os.path.join(drive_root, 'models'),
        os.path.join(drive_root, 'results'),
        os.path.join(drive_root, 'checkpoints'),
        os.path.join(drive_root, 'data'),
        os.path.join(drive_root, 'logs'),
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print(f"✅ Google Drive mounted")
    print(f"✅ Backup folder: {drive_root}")

except ImportError:
    print("⚠️  Not running on Colab - skipping Drive mount")
    drive_root = None
except Exception as e:
    print(f"⚠️  Drive mount failed: {e}")
    drive_root = None

print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: CLONE OR UPDATE REPOSITORY
# ═══════════════════════════════════════════════════════════════════════════════

print("📦 STEP 2/8: Setting up repository")
print("-" * 80)

repo_url = "https://github.com/LKBSM/TradingBotNew.git"
repo_path = "/content/TradingBotNew"

if os.path.exists(repo_path):
    print(f"📂 Repository exists, pulling latest changes...")
    os.chdir(repo_path)
    result = subprocess.run(['git', 'pull'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Repository updated")
    else:
        print(f"⚠️  Git pull issue: {result.stderr}")
        print("   Removing and re-cloning...")
        os.chdir('/content')
        shutil.rmtree(repo_path, ignore_errors=True)
        subprocess.run(['git', 'clone', repo_url], check=True)
        print(f"✅ Repository cloned fresh")
else:
    print(f"📥 Cloning repository...")
    os.chdir('/content')
    subprocess.run(['git', 'clone', repo_url], check=True)
    print(f"✅ Repository cloned: {repo_path}")

os.chdir(repo_path)
print(f"✅ Working directory: {os.getcwd()}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════

print("📚 STEP 3/8: Installing dependencies")
print("-" * 80)

# Install requirements
requirements_path = os.path.join(repo_path, 'requirements.txt')
if os.path.exists(requirements_path):
    print("📥 Installing from requirements.txt...")
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '-r', requirements_path],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("✅ Dependencies installed")
    else:
        print(f"⚠️  Some packages may have issues: {result.stderr[:200]}")
else:
    print("⚠️  requirements.txt not found, installing core packages...")
    packages = [
        'stable-baselines3[extra]',
        'gymnasium',
        'pandas',
        'numpy',
        'torch',
        'scikit-learn',
        'ta',
        'rich',
        'arch',
    ]
    for pkg in packages:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg])
    print("✅ Core packages installed")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: DOWNLOAD DATASET
# ═══════════════════════════════════════════════════════════════════════════════

print("📊 STEP 4/8: Downloading dataset (170K+ bars)")
print("-" * 80)

import urllib.request

dataset_url = "https://github.com/LKBSM/TradingBot_Agentic/releases/latest/download/XAU_15MIN_2019_2025.csv"
dataset_path = os.path.join(repo_path, 'data', 'XAU_15MIN_2019_2025.csv')

os.makedirs(os.path.join(repo_path, 'data'), exist_ok=True)

if os.path.exists(dataset_path):
    file_size = os.path.getsize(dataset_path) / 1e6
    print(f"✅ Dataset already exists: {file_size:.2f} MB")
else:
    print(f"📥 Downloading from: {dataset_url}")
    try:
        urllib.request.urlretrieve(dataset_url, dataset_path)
        file_size = os.path.getsize(dataset_path) / 1e6
        print(f"✅ Dataset downloaded: {file_size:.2f} MB")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("   Please manually upload XAU_15MIN_2019_2025.csv to data/ folder")

# Backup to Drive
if drive_root and os.path.exists(dataset_path):
    try:
        drive_dataset = os.path.join(drive_root, 'data', 'XAU_15MIN_2019_2025.csv')
        if not os.path.exists(drive_dataset):
            shutil.copy(dataset_path, drive_dataset)
            print(f"✅ Backup to Drive: {drive_dataset}")
        else:
            print(f"✅ Drive backup exists")
    except Exception as e:
        print(f"⚠️  Drive backup failed: {e}")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: VERIFY PROJECT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

print("🔍 STEP 5/8: Verifying project structure")
print("-" * 80)

required_files = [
    'config.py',
    'parallel_training.py',
    'src/agent_trainer.py',
    'src/environment/environment.py',
    'src/environment/risk_manager.py',
    'src/agents/base_agent.py',
    'data/XAU_15MIN_2019_2025.csv',
]

all_good = True
for file in required_files:
    full_path = os.path.join(repo_path, file)
    if os.path.exists(full_path):
        print(f"✅ {file}")
    else:
        print(f"❌ MISSING: {file}")
        all_good = False

if not all_good:
    print("\n⚠️  Some files are missing!")
    print("   The training may not work correctly.")
else:
    print("\n✅ All required files present")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: OPTIMIZE CONFIGURATION FOR COLAB
# ═══════════════════════════════════════════════════════════════════════════════

print("⚙️  STEP 6/8: Optimizing configuration for Colab")
print("-" * 80)

config_path = os.path.join(repo_path, 'config.py')

# Read current config
with open(config_path, 'r') as f:
    config_content = f.read()

# Optimizations for Colab (free tier has time limits)
optimizations = {
    # Reduce bots for faster completion on free Colab
    ('N_PARALLEL_BOTS = 50', 'N_PARALLEL_BOTS = 25  # Optimized for Colab'),
    # Use 3 workers (good for T4 GPU)
    ('MAX_WORKERS_GPU = 2', 'MAX_WORKERS_GPU = 3  # Optimized for Colab T4'),
}

changes_made = []
for old, new in optimizations:
    if old in config_content:
        config_content = config_content.replace(old, new)
        changes_made.append(new.split('=')[0].strip())

if changes_made:
    with open(config_path, 'w') as f:
        f.write(config_content)
    print("✅ Configuration optimized:")
    for change in changes_made:
        print(f"   • {change}")
else:
    print("✅ Configuration already optimized")

print("\n📊 Current settings:")
# Re-import to show current values
sys.path.insert(0, repo_path)
try:
    import importlib
    import config
    importlib.reload(config)
    print(f"   • Bots to train: {config.N_PARALLEL_BOTS}")
    print(f"   • Timesteps per bot: {config.TOTAL_TIMESTEPS_PER_BOT:,}")
    print(f"   • Workers: {config.MAX_WORKERS_GPU}")
    print(f"   • Walk-forward: {config.USE_WALK_FORWARD}")
except Exception as e:
    print(f"   ⚠️  Could not read config: {e}")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: CREATE REQUIRED DIRECTORIES
# ═══════════════════════════════════════════════════════════════════════════════

print("📁 STEP 7/8: Creating required directories")
print("-" * 80)

directories = [
    'trained_models',
    'results',
    'results/training_reports',
    'results/performance_charts',
    'results/checkpoints',
    'logs',
    'logs/tensorboard',
    'logs/events',
]

for dir_name in directories:
    dir_path = os.path.join(repo_path, dir_name)
    os.makedirs(dir_path, exist_ok=True)

print("✅ All directories created")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: VERIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("🔬 STEP 8/8: Running verification tests")
print("-" * 80)

test_results = []

# Test 1: Import config
try:
    import config
    test_results.append(("Config import", True, f"Dataset: {os.path.basename(config.HISTORICAL_DATA_FILE)}"))
except Exception as e:
    test_results.append(("Config import", False, str(e)))

# Test 2: Dataset readable
try:
    import pandas as pd
    df = pd.read_csv(dataset_path, nrows=10)
    n_rows = sum(1 for _ in open(dataset_path)) - 1  # Count rows
    test_results.append(("Dataset", True, f"{n_rows:,} rows, columns: {list(df.columns)[:5]}"))
except Exception as e:
    test_results.append(("Dataset", False, str(e)))

# Test 3: GPU check
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        test_results.append(("GPU", True, f"{gpu_name} ({gpu_mem:.1f} GB)"))
    else:
        test_results.append(("GPU", False, "No GPU detected - training will be slow"))
except Exception as e:
    test_results.append(("GPU", False, str(e)))

# Test 4: Import environment
try:
    from src.environment.environment import TradingEnv
    test_results.append(("TradingEnv", True, "Import successful"))
except Exception as e:
    test_results.append(("TradingEnv", False, str(e)[:50]))

# Test 5: Import trainer
try:
    from src.agent_trainer import AgentTrainer
    test_results.append(("AgentTrainer", True, "Import successful"))
except Exception as e:
    test_results.append(("AgentTrainer", False, str(e)[:50]))

# Print results
for name, success, msg in test_results:
    status = "✅" if success else "❌"
    print(f"{status} {name}: {msg}")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP COMPLETE - SHOW NEXT STEPS
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("   SETUP COMPLETE!")
print("=" * 80)
print()

all_tests_passed = all(result[1] for result in test_results)

if all_tests_passed:
    print("✅ All tests passed! Ready for training.")
else:
    print("⚠️  Some tests failed. Review the errors above.")

print()
print("=" * 80)
print("   NEXT STEPS")
print("=" * 80)
print()
print("1️⃣  [OPTIONAL] Enable anti-disconnect (copy to new cell):")
print()
print('    from IPython.display import Javascript, display')
print('    display(Javascript("""')
print('    function KeepAlive(){')
print('        console.log("Keeping alive...");')
print('        document.querySelector("colab-connect-button")?.click();')
print('    }')
print('    setInterval(KeepAlive, 60000);')
print('    """))')
print("    print('✅ Anti-disconnect activated')")
print()
print("2️⃣  START TRAINING (copy to new cell):")
print()
print("    %cd /content/TradingBotNew")
print("    !python parallel_training.py")
print()
print("=" * 80)
print()
print("📊 Estimated training time:")
print(f"   • Bots: {config.N_PARALLEL_BOTS if 'config' in dir() else 25}")
print(f"   • Per bot: ~20-30 min")
print(f"   • Total: ~8-15 hours (with T4 GPU)")
print()
print("💾 Models will be saved to:")
print(f"   • Local: {repo_path}/trained_models/")
if drive_root:
    print(f"   • Drive: {drive_root}/models/")
print()
print("=" * 80)
