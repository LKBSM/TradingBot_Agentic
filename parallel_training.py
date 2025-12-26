# ═════════════════════════════════════════════════════════════════════════════
# 🏢 PARALLEL TRAINING - PRODUCTION VERSION WITH AUTO-DRIVE & CHECKPOINTS
# ═════════════════════════════════════════════════════════════════════════════
#
# Features:
# - ✅ Auto-mount Google Drive (no manual intervention)
# - ✅ Checkpoint resume capability (survive Colab disconnections)
# - ✅ Automatic model backups to Drive
# - ✅ Intelligent hyperparameter search
# - ✅ Multi-worker GPU training
# - ✅ Comprehensive reporting
#
# ═════════════════════════════════════════════════════════════════════════════

import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import os
import sys
import pandas as pd
import numpy as np
import torch
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# Rich console
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
from rich.panel import Panel
from rich import box

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.agent_trainer import AgentTrainer, evaluate_agent

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
# 🔧 GOOGLE DRIVE AUTO-MOUNT
# ═════════════════════════════════════════════════════════════════════════════

def setup_google_drive():
    """
    Automatically mounts Google Drive if running on Colab.
    Creates backup folder structure.

    Returns:
        str: Backup root path if successful, None otherwise
    """
    # Detect if running on Colab
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False

    if not is_colab:
        console.print("[cyan]ℹ️  Not running on Colab - Drive mount skipped[/cyan]")
        return None

    # Check if already mounted
    if os.path.exists('/content/drive/MyDrive'):
        console.print("[green]✅ Google Drive already mounted[/green]")
    else:
        # Mount Drive
        try:
            console.print("[cyan]☁️  Mounting Google Drive...[/cyan]")
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)

            if os.path.exists('/content/drive/MyDrive'):
                console.print("[green]✅ Google Drive mounted successfully![/green]")
            else:
                console.print("[yellow]⚠️  Drive mounted but not accessible[/yellow]")
                return None
        except Exception as e:
            console.print(f"[yellow]⚠️  Drive mount failed: {e}[/yellow]")
            console.print("[yellow]   → Local save only[/yellow]")
            return None

    # Create folder structure
    try:
        backup_root = '/content/drive/MyDrive/TradingBot_Results'
        folders = [
            backup_root,
            os.path.join(backup_root, 'checkpoints'),
            os.path.join(backup_root, 'models'),
            os.path.join(backup_root, 'results'),
            os.path.join(backup_root, 'logs')
        ]

        for folder in folders:
            os.makedirs(folder, exist_ok=True)

        console.print(f"[green]✅ Drive folders ready: {backup_root}[/green]")
        return backup_root

    except Exception as e:
        console.print(f"[yellow]⚠️  Folder creation failed: {e}[/yellow]")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# 💾 CHECKPOINT SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

def save_checkpoint(completed_bots: int, results: List[Dict], backup_root: Optional[str] = None):
    """
    Saves training checkpoint to local and Drive.

    Args:
        completed_bots: Number of bots completed
        results: List of bot results
        backup_root: Drive backup path (None if unavailable)
    """
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'completed_bots': completed_bots,
        'total_bots': config.N_PARALLEL_BOTS,
        'progress_pct': (completed_bots / config.N_PARALLEL_BOTS) * 100,
        'results': results
    }

    # 1. Local save (always)
    checkpoint_dir = os.path.join(config.RESULTS_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.json')
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    console.print(
        f"[green]✅ Checkpoint: {completed_bots}/{config.N_PARALLEL_BOTS} "
        f"({checkpoint['progress_pct']:.0f}%)[/green]"
    )

    # 2. Drive backup (if available)
    if backup_root:
        try:
            drive_checkpoint = os.path.join(backup_root, 'checkpoints', 'checkpoint_latest.json')
            shutil.copy(checkpoint_path, drive_checkpoint)

            # Also timestamped backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            drive_backup = os.path.join(backup_root, 'checkpoints', f'checkpoint_{timestamp}.json')
            shutil.copy(checkpoint_path, drive_backup)

            console.print(f"[green]💾 Drive checkpoint saved[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Drive backup failed: {e}[/yellow]")


def load_checkpoint(backup_root: Optional[str] = None) -> Optional[Dict]:
    """
    Loads checkpoint from Drive (priority) or local.

    Args:
        backup_root: Drive backup path

    Returns:
        Checkpoint dict or None
    """
    # Try Drive first
    if backup_root:
        drive_checkpoint = os.path.join(backup_root, 'checkpoints', 'checkpoint_latest.json')
        if os.path.exists(drive_checkpoint):
            try:
                with open(drive_checkpoint, 'r') as f:
                    checkpoint = json.load(f)
                console.print(
                    f"[green]✅ Checkpoint loaded from Drive: "
                    f"{checkpoint['completed_bots']}/{checkpoint['total_bots']}[/green]"
                )
                return checkpoint
            except Exception as e:
                console.print(f"[yellow]⚠️  Drive checkpoint read error: {e}[/yellow]")

    # Fallback to local
    local_checkpoint = os.path.join(config.RESULTS_DIR, 'checkpoints', 'checkpoint_latest.json')
    if os.path.exists(local_checkpoint):
        try:
            with open(local_checkpoint, 'r') as f:
                checkpoint = json.load(f)
            console.print(
                f"[green]✅ Local checkpoint: "
                f"{checkpoint['completed_bots']}/{checkpoint['total_bots']}[/green]"
            )
            return checkpoint
        except Exception as e:
            console.print(f"[yellow]⚠️  Local checkpoint read error: {e}[/yellow]")

    return None


# ═════════════════════════════════════════════════════════════════════════════
# 🧠 INTELLIGENT HYPERPARAMETER GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_intelligent_hyperparam_sets(n_bots: int = 50) -> List[Dict]:
    """
    Generates hyperparameter sets using stratified sampling.

    Strategy:
    1. Baseline (FinRL defaults)
    2. LR × Gamma grid (most important)
    3. Architecture variations
    4. Exploration parameters
    5. Random diversity

    Args:
        n_bots: Number of configurations to generate

    Returns:
        List of hyperparameter dicts
    """
    console.print("\n[bold cyan]🧠 Generating Intelligent Hyperparameter Sets[/bold cyan]")
    console.print(f"   Target: {n_bots} unique configurations\n")

    lr_space = config.HYPERPARAM_SEARCH_SPACE['learning_rate']
    n_steps_space = config.HYPERPARAM_SEARCH_SPACE['n_steps']
    batch_space = config.HYPERPARAM_SEARCH_SPACE['batch_size']
    gamma_space = config.HYPERPARAM_SEARCH_SPACE['gamma']
    ent_space = config.HYPERPARAM_SEARCH_SPACE['ent_coef']
    clip_space = config.HYPERPARAM_SEARCH_SPACE['clip_range']

    hyperparam_sets = []

    # STRATEGY 1: Baseline
    baseline = {
        'learning_rate': 3e-5,
        'n_steps': 2048,
        'batch_size': 128,
        'gamma': 0.99,
        'ent_coef': 0.01,
        'clip_range': 0.2
    }
    hyperparam_sets.append(baseline)
    console.print("   ✅ Added baseline (FinRL research)")

    # STRATEGY 2: LR × Gamma Grid
    priority_count = 0
    for lr in lr_space:
        for gamma in gamma_space:
            if len(hyperparam_sets) >= 16:
                break
            if lr == 3e-5 and gamma == 0.99:  # Skip baseline duplicate
                continue
            hyperparam_sets.append({
                'learning_rate': lr,
                'n_steps': 2048,
                'batch_size': 128,
                'gamma': gamma,
                'ent_coef': 0.01,
                'clip_range': 0.2
            })
            priority_count += 1
        if len(hyperparam_sets) >= 16:
            break
    console.print(f"   ✅ Added {priority_count} LR×Gamma variations")

    # STRATEGY 3: Architecture
    arch_count = 0
    for n_steps in n_steps_space:
        for batch_size in batch_space:
            if len(hyperparam_sets) >= 28:
                break
            if batch_size > n_steps:  # Constraint
                continue
            if any(hp['n_steps'] == n_steps and hp['batch_size'] == batch_size
                   for hp in hyperparam_sets):
                continue
            hyperparam_sets.append({
                'learning_rate': 3e-5,
                'n_steps': n_steps,
                'batch_size': batch_size,
                'gamma': 0.99,
                'ent_coef': 0.01,
                'clip_range': 0.2
            })
            arch_count += 1
        if len(hyperparam_sets) >= 28:
            break
    console.print(f"   ✅ Added {arch_count} architecture variations")

    # STRATEGY 4: Exploration
    explore_count = 0
    for ent in ent_space:
        for clip in clip_space:
            if len(hyperparam_sets) >= 37:
                break
            if ent == 0.01 and clip == 0.2:  # Skip baseline
                continue
            hyperparam_sets.append({
                'learning_rate': 3e-5,
                'n_steps': 2048,
                'batch_size': 128,
                'gamma': 0.99,
                'ent_coef': ent,
                'clip_range': clip
            })
            explore_count += 1
        if len(hyperparam_sets) >= 37:
            break
    console.print(f"   ✅ Added {explore_count} exploration variations")

    # STRATEGY 5: Random
    random_count = 0
    np.random.seed(config.RANDOM_SEED)
    max_attempts = 1000
    attempts = 0

    while len(hyperparam_sets) < n_bots and attempts < max_attempts:
        attempts += 1
        candidate = {
            'learning_rate': np.random.choice(lr_space),
            'n_steps': np.random.choice(n_steps_space),
            'batch_size': np.random.choice(batch_space),
            'gamma': np.random.choice(gamma_space),
            'ent_coef': np.random.choice(ent_space),
            'clip_range': np.random.choice(clip_space)
        }
        if candidate['batch_size'] > candidate['n_steps']:
            continue
        if any(all(hp[k] == candidate[k] for k in candidate.keys())
               for hp in hyperparam_sets):
            continue
        hyperparam_sets.append(candidate)
        random_count += 1

    console.print(f"   ✅ Added {random_count} random explorations")

    final_sets = hyperparam_sets[:n_bots]
    console.print(f"\n[bold green]✅ Generated {len(final_sets)} unique configurations[/bold green]\n")

    return final_sets


# ═════════════════════════════════════════════════════════════════════════════
# 🤖 SINGLE BOT TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_single_bot(
        bot_id: int,
        hyperparams: Dict,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        backup_root: Optional[str] = None,
        verbose: bool = False
) -> Dict:
    """
    Trains a single bot with given hyperparameters.

    Args:
        bot_id: Bot number (1-indexed)
        hyperparams: Hyperparameter dict
        df_train: Training data
        df_val: Validation data
        df_test: Test data
        backup_root: Drive backup path
        verbose: Print detailed logs

    Returns:
        Results dict with metrics
    """
    try:
        start_time = datetime.now()

        print(f"\n{'=' * 70}")
        print(f"🤖 BOT {bot_id}/{config.N_PARALLEL_BOTS} - DÉMARRAGE")
        print(f"{'=' * 70}")
        print(f"⏰ {start_time.strftime('%H:%M:%S')}")
        print(f"🎯 LR: {hyperparams['learning_rate']:.2e} | Gamma: {hyperparams['gamma']}")
        print(f"{'=' * 70}")

        # Update config
        config.MODEL_HYPERPARAMETERS.update(hyperparams)

        # Train
        trainer = AgentTrainer(df_historical=df_train)
        agent = trainer.train_offline(
            total_timesteps=config.TOTAL_TIMESTEPS_PER_BOT,
            use_early_stopping=False,
            seed=config.RANDOM_SEED + bot_id
        )

        training_duration = (datetime.now() - start_time).total_seconds()
        print(f"✅ Training: {training_duration / 60:.1f} min")

        # Save model
        model_filename = f"bot_{bot_id:03d}_lr{hyperparams['learning_rate']:.0e}_g{hyperparams['gamma']}.zip"
        model_path = os.path.join(config.MODEL_DIR, model_filename)
        agent.save(model_path)

        # Backup to Drive
        if backup_root:
            try:
                drive_model_path = os.path.join(backup_root, 'models', model_filename)
                shutil.copy(model_path, drive_model_path)
                print(f"💾 Drive backup: OK")
            except Exception as e:
                print(f"⚠️  Drive backup failed: {e}")

        # Evaluate
        train_metrics = evaluate_agent(agent, df_train)
        val_metrics = evaluate_agent(agent, df_val)
        test_metrics = evaluate_agent(agent, df_test)

        initial_capital = config.INITIAL_BALANCE
        test_profit = initial_capital * test_metrics[0]

        print(f"📊 Test Sharpe: {test_metrics[1]:.2f} | "
              f"Return: {test_metrics[0] * 100:.2f}% | "
              f"Profit: ${test_profit:.2f}")

        # Overfitting detection
        sharpe_gap = train_metrics[1] - val_metrics[1]
        if sharpe_gap > 1.5:
            overfit_status = "SEVERE_OVERFIT"
        elif sharpe_gap > 0.8:
            overfit_status = "MILD_OVERFIT"
        elif sharpe_gap < -0.5:
            overfit_status = "UNDERFIT"
        else:
            overfit_status = "GOOD_FIT"

        # Commercial viability
        is_profitable = test_profit > 0
        meets_sharpe = test_metrics[1] >= config.MIN_ACCEPTABLE_SHARPE
        meets_calmar = test_metrics[3] >= config.MIN_ACCEPTABLE_CALMAR
        meets_dd = test_metrics[4] < config.MAX_ACCEPTABLE_DD
        commercial_score = sum([is_profitable, meets_sharpe, meets_calmar, meets_dd])

        if commercial_score >= 3:
            commercial_status = "APPROVED"
        elif commercial_score == 2:
            commercial_status = "CONDITIONAL"
        else:
            commercial_status = "REJECTED"

        print(f"🎯 Status: {commercial_status} ({commercial_score}/4)")
        print(f"{'=' * 70}\n")

        result = {
            'bot_id': bot_id,
            'model_path': model_path,
            'training_duration_sec': training_duration,
            'learning_rate': hyperparams['learning_rate'],
            'n_steps': hyperparams['n_steps'],
            'batch_size': hyperparams['batch_size'],
            'gamma': hyperparams['gamma'],
            'ent_coef': hyperparams['ent_coef'],
            'clip_range': hyperparams['clip_range'],
            'train_return': train_metrics[0],
            'train_sharpe': train_metrics[1],
            'train_profit_usd': initial_capital * train_metrics[0],
            'val_return': val_metrics[0],
            'val_sharpe': val_metrics[1],
            'val_profit_usd': initial_capital * val_metrics[0],
            'test_return': test_metrics[0],
            'test_sharpe': test_metrics[1],
            'test_sortino': test_metrics[2],
            'test_calmar': test_metrics[3],
            'test_max_dd': test_metrics[4],
            'test_profit_usd': test_profit,
            'overfit_status': overfit_status,
            'sharpe_gap_train_val': sharpe_gap,
            'is_profitable': is_profitable,
            'meets_sharpe_target': meets_sharpe,
            'meets_calmar_target': meets_calmar,
            'meets_dd_target': meets_dd,
            'commercial_status': commercial_status,
            'commercial_score': commercial_score,
            'overall_score': (test_metrics[1] + test_metrics[3]) / 2
        }

        return result

    except Exception as e:
        print(f"❌ Bot {bot_id} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'bot_id': bot_id,
            'error': str(e),
            'commercial_status': 'ERROR',
            'test_sharpe': -np.inf,
            'test_profit_usd': -np.inf,
            'is_profitable': False
        }


# ═════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN TRAINING ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_parallel_training():
    """Main training pipeline with checkpoint resume."""

    console.print(Panel.fit(
        "[bold white]🏢 XAU TRADING BOT - PRODUCTION TRAINING[/bold white]\n"
        f"Training {config.N_PARALLEL_BOTS} bots × {config.TOTAL_TIMESTEPS_PER_BOT:,} timesteps\n"
        f"Target: Sharpe >{config.MIN_ACCEPTABLE_SHARPE} | Calmar >{config.MIN_ACCEPTABLE_CALMAR}",
        box=box.DOUBLE,
        style="cyan"
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 0: Setup Google Drive
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]☁️  Step 0: Google Drive Setup[/cyan]")
    backup_root = setup_google_drive()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 0.5: Check for checkpoint
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]🔄 Checking for existing checkpoint...[/cyan]")
    checkpoint = load_checkpoint(backup_root)

    if checkpoint:
        completed_bots = checkpoint['completed_bots']
        previous_results = checkpoint['results']

        console.print(Panel.fit(
            f"[bold yellow]🔄 RESUME DETECTED![/bold yellow]\n\n"
            f"Progress: {completed_bots}/{config.N_PARALLEL_BOTS} bots ({checkpoint['progress_pct']:.0f}%)\n"
            f"Timestamp: {checkpoint['timestamp']}\n"
            f"Remaining: {config.N_PARALLEL_BOTS - completed_bots} bots",
            box=box.HEAVY,
            style="yellow"
        ))

        # Auto-resume in Colab
        console.print("[green]✅ Auto-resume enabled[/green]")
        resume = True
    else:
        console.print("[cyan]ℹ️  Aucun checkpoint trouvé - Nouveau training[/cyan]")
        completed_bots = 0
        previous_results = []
        resume = False

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Load Data
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]📂 Step 1: Loading Data[/cyan]")

    try:
        df_full = pd.read_csv(config.HISTORICAL_DATA_FILE)
        console.print(f"[green]✅ Loaded {len(df_full):,} bars from {config.HISTORICAL_DATA_FILE}[/green]")
    except FileNotFoundError:
        console.print(f"[yellow]⚠️  Using mock data[/yellow]")
        n_points = 20000
        prices = 1800 + np.cumsum(np.random.randn(n_points) * 2)
        df_full = pd.DataFrame({
            'Date': pd.date_range(start='2019-01-01', periods=n_points, freq='15min'),
            'Open': prices + np.random.uniform(-1, 1, n_points),
            'High': prices + np.random.uniform(1, 3, n_points),
            'Low': prices - np.random.uniform(1, 3, n_points),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n_points)
        }).set_index('Date')

    # Split data
    n = len(df_full)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    df_train = df_full.iloc[:train_end].copy()
    df_val = df_full.iloc[train_end:val_end].copy()
    df_test = df_full.iloc[val_end:].copy()

    console.print(f"[cyan]   Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}[/cyan]")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Generate Hyperparameters
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]📊 Step 2: Hyperparameter Generation[/cyan]")
    hyperparam_sets = generate_intelligent_hyperparam_sets(config.N_PARALLEL_BOTS)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: GPU Check
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]💻 Step 3: GPU Check[/cyan]")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[green]✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)[/green]")
    else:
        console.print("[yellow]⚠️  No GPU - using CPU[/yellow]")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Parallel Training
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]🚀 Step 4: Parallel Training[/cyan]")

    # Determine which bots to train
    if resume:
        bots_to_train = list(range(completed_bots, config.N_PARALLEL_BOTS))
        console.print(f"[yellow]🆕 Nouveau: Training bots {completed_bots + 1} à {config.N_PARALLEL_BOTS}[/yellow]")
    else:
        bots_to_train = list(range(config.N_PARALLEL_BOTS))
        console.print(f"[cyan]🆕 Nouveau: Training bots 1 à {config.N_PARALLEL_BOTS}[/cyan]")

    results = previous_results.copy()
    start_time = datetime.now()

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
    ) as progress:

        task = progress.add_task(
            f"[cyan]Training {len(bots_to_train)} bots...",
            total=len(bots_to_train)
        )

        with ProcessPoolExecutor(max_workers=config.MAX_WORKERS_GPU) as executor:
            futures = {
                executor.submit(
                    train_single_bot,
                    i + 1,
                    hyperparam_sets[i],
                    df_train,
                    df_val,
                    df_test,
                    backup_root
                ): i for i in bots_to_train
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                # Save checkpoint after each bot
                current_completed = len(results)
                save_checkpoint(current_completed, results, backup_root)

                progress.update(task, advance=1)

    total_duration = (datetime.now() - start_time).total_seconds()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Process Results
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]📈 Step 5: Processing Results[/cyan]")

    df_results = pd.DataFrame(results)
    df_valid = df_results[~df_results.get('error', pd.Series([None] * len(df_results))).notna()].copy()

    if len(df_valid) == 0:
        console.print("[red]❌ All bots failed![/red]")
        return

    df_valid = df_valid.sort_values('overall_score', ascending=False)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Generate Report
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]📄 Step 6: Generating Reports[/cyan]")
    generate_comprehensive_report(df_valid, total_duration, backup_root)

    console.print("\n[bold green]🎉 TRAINING COMPLETE![/bold green]\n")


# ═════════════════════════════════════════════════════════════════════════════
# 📊 COMPREHENSIVE REPORTING
# ═════════════════════════════════════════════════════════════════════════════

def generate_comprehensive_report(df_results: pd.DataFrame, duration: float, backup_root: Optional[str] = None):
    """Generates comprehensive reports with Drive backup."""

    n_bots = len(df_results)
    n_profitable = df_results['is_profitable'].sum()
    n_approved = (df_results['commercial_status'] == 'APPROVED').sum()

    console.print("\n" + "=" * 80)
    console.print("[bold cyan]📊 TRAINING RESULTS SUMMARY[/bold cyan]")
    console.print("=" * 80 + "\n")

    console.print(Panel.fit(
        f"[bold white]EXECUTIVE SUMMARY[/bold white]\n\n"
        f"Total Bots: {n_bots}\n"
        f"Profitable: {n_profitable} ({n_profitable / n_bots * 100:.1f}%)\n"
        f"Approved: {n_approved} ({n_approved / n_bots * 100:.1f}%)\n\n"
        f"Avg Sharpe: {df_results['test_sharpe'].mean():.2f}\n"
        f"Avg Profit: ${df_results['test_profit_usd'].mean():,.2f}\n"
        f"Best Profit: ${df_results['test_profit_usd'].max():,.2f}\n\n"
        f"Duration: {duration / 3600:.2f}h",
        box=box.HEAVY,
        style="green"
    ))

    # Top 10 table
    console.print("\n[bold]🏆 TOP 10 BOTS[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Rank", style="cyan", justify="center")
    table.add_column("Bot", style="yellow")
    table.add_column("Sharpe", justify="right", style="green")
    table.add_column("Profit", justify="right", style="magenta")
    table.add_column("DD", justify="right", style="red")
    table.add_column("Status", justify="center")

    for rank, (idx, row) in enumerate(df_results.head(10).iterrows(), 1):
        profit_color = "green" if row['test_profit_usd'] > 0 else "red"
        status = "✅" if row['commercial_status'] == 'APPROVED' else "⚠️" if row[
                                                                                'commercial_status'] == 'CONDITIONAL' else "❌"

        table.add_row(
            str(rank),
            f"Bot_{row['bot_id']:03d}",
            f"{row['test_sharpe']:.2f}",
            f"[{profit_color}]${row['test_profit_usd']:,.2f}[/{profit_color}]",
            f"{row['test_max_dd']:.2%}",
            status
        )

    console.print(table)

    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(config.RESULTS_DIR, f"results_{timestamp}.csv")
    df_results.to_csv(csv_path, index=False)
    console.print(f"\n[green]✅ CSV: {csv_path}[/green]")

    # Backup CSV to Drive
    if backup_root:
        try:
            drive_csv = os.path.join(backup_root, 'results', f'results_{timestamp}.csv')
            shutil.copy(csv_path, drive_csv)
            console.print(f"[green]💾 Drive CSV: {drive_csv}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Drive CSV backup failed: {e}[/yellow]")

    # Best model
    best_bot = df_results.iloc[0]
    console.print(f"\n[bold green]🎯 BEST MODEL[/bold green]")
    console.print(f"   Bot: {best_bot['bot_id']}")
    console.print(f"   Sharpe: {best_bot['test_sharpe']:.2f}")
    console.print(f"   Profit: ${best_bot['test_profit_usd']:,.2f}")
    console.print(f"   Status: {best_bot['commercial_status']}")

    # Copy best model
    prod_path = os.path.join(config.MODEL_DIR, "MODEL_PRODUCTION_BEST.zip")
    shutil.copy(best_bot['model_path'], prod_path)
    console.print(f"\n[bold cyan]✅ Best model: {prod_path}[/bold cyan]\n")

    # Backup best model to Drive
    if backup_root:
        try:
            drive_best = os.path.join(backup_root, 'models', 'MODEL_PRODUCTION_BEST.zip')
            shutil.copy(prod_path, drive_best)
            console.print(f"[green]💾 Drive best model: {drive_best}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Drive best model backup failed: {e}[/yellow]")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_parallel_training()