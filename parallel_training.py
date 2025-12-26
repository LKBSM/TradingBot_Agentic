# ═════════════════════════════════════════════════════════════════════════════
# 🏢 PARALLEL TRAINING - VERSION PRODUCTION AVEC CHECKPOINTS & GOOGLE DRIVE
# ═════════════════════════════════════════════════════════════════════════════

import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Déjà défini (ignore)

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
from itertools import product
import warnings

warnings.filterwarnings('ignore')

# Rich console for beautiful output
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
from rich.panel import Panel
from rich import box

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
import config
from src.agent_trainer import AgentTrainer, evaluate_agent

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
# 🔄 CHECKPOINT & BACKUP SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

def setup_google_drive_backup():
    """Monte Google Drive et configure les backups automatiques."""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)

        # Créer dossiers backup
        backup_root = '/content/drive/MyDrive/TradingBot_Results'
        os.makedirs(f'{backup_root}/models', exist_ok=True)
        os.makedirs(f'{backup_root}/checkpoints', exist_ok=True)
        os.makedirs(f'{backup_root}/results', exist_ok=True)

        console.print("[green]✅ Google Drive monté et configuré[/green]")
        return backup_root
    except Exception as e:
        console.print(f"[yellow]⚠️ Google Drive non disponible: {e}[/yellow]")
        console.print("[yellow]   Sauvegarde locale uniquement[/yellow]")
        return None


def save_checkpoint(completed_bots: int, results: List[Dict], backup_root: Optional[str] = None):
    """
    Sauvegarde checkpoint local + backup Google Drive.

    Args:
        completed_bots: Nombre de bots terminés
        results: Liste des résultats
        backup_root: Chemin racine du backup Drive (None si Drive indisponible)
    """
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'completed_bots': completed_bots,
        'total_bots': config.N_PARALLEL_BOTS,
        'progress_pct': (completed_bots / config.N_PARALLEL_BOTS) * 100,
        'results': results
    }

    # 1️⃣ Sauvegarde locale (toujours)
    checkpoint_dir = os.path.join(config.RESULTS_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.json')
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    console.print(
        f"[green]✅ Checkpoint local: {completed_bots}/{config.N_PARALLEL_BOTS} bots ({checkpoint['progress_pct']:.0f}%)[/green]")

    # 2️⃣ Backup Google Drive (si disponible)
    if backup_root:
        try:
            drive_checkpoint_path = f'{backup_root}/checkpoints/checkpoint_latest.json'
            shutil.copy(checkpoint_path, drive_checkpoint_path)

            # Aussi sauvegarder avec timestamp (historique)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            drive_checkpoint_backup = f'{backup_root}/checkpoints/checkpoint_{timestamp}.json'
            shutil.copy(checkpoint_path, drive_checkpoint_backup)

            console.print(f"[green]💾 Checkpoint Drive: {drive_checkpoint_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Backup Drive échoué: {e}[/yellow]")


def load_checkpoint(backup_root: Optional[str] = None) -> Optional[Dict]:
    """
    Charge le dernier checkpoint (Drive prioritaire, sinon local).

    Args:
        backup_root: Chemin racine du backup Drive

    Returns:
        Dict du checkpoint ou None si inexistant
    """
    checkpoint_loaded = None

    # 1️⃣ Essayer Google Drive en priorité
    if backup_root:
        drive_checkpoint_path = f'{backup_root}/checkpoints/checkpoint_latest.json'
        if os.path.exists(drive_checkpoint_path):
            try:
                with open(drive_checkpoint_path, 'r') as f:
                    checkpoint_loaded = json.load(f)
                console.print(
                    f"[green]✅ Checkpoint chargé depuis Drive: {checkpoint_loaded['completed_bots']}/{checkpoint_loaded['total_bots']} bots[/green]")
                return checkpoint_loaded
            except Exception as e:
                console.print(f"[yellow]⚠️ Erreur lecture Drive: {e}[/yellow]")

    # 2️⃣ Fallback: Checkpoint local
    local_checkpoint_path = os.path.join(config.RESULTS_DIR, 'checkpoints', 'checkpoint_latest.json')
    if os.path.exists(local_checkpoint_path):
        try:
            with open(local_checkpoint_path, 'r') as f:
                checkpoint_loaded = json.load(f)
            console.print(
                f"[green]✅ Checkpoint local trouvé: {checkpoint_loaded['completed_bots']}/{checkpoint_loaded['total_bots']} bots[/green]")
            return checkpoint_loaded
        except Exception as e:
            console.print(f"[yellow]⚠️ Erreur lecture local: {e}[/yellow]")

    return None


# ═════════════════════════════════════════════════════════════════════════════
# 1. INTELLIGENT HYPERPARAMETER GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_intelligent_hyperparam_sets(n_bots: int = 50) -> List[Dict]:
    """Generates hyperparameter sets using INTELLIGENT STRATIFIED SAMPLING."""
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
        'learning_rate': 3e-5, 'n_steps': 2048, 'batch_size': 128,
        'gamma': 0.99, 'ent_coef': 0.01, 'clip_range': 0.2
    }
    hyperparam_sets.append(baseline)
    console.print("   ✅ Added baseline (FinRL research)")

    # STRATEGY 2: LR × Gamma Grid
    priority_count = 0
    for lr in lr_space:
        for gamma in gamma_space:
            if len(hyperparam_sets) >= 16:
                break
            if lr == 3e-5 and gamma == 0.99:
                continue
            hyperparam_sets.append({
                'learning_rate': lr, 'n_steps': 2048, 'batch_size': 128,
                'gamma': gamma, 'ent_coef': 0.01, 'clip_range': 0.2
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
            if batch_size > n_steps:
                continue
            if any(hp['n_steps'] == n_steps and hp['batch_size'] == batch_size for hp in hyperparam_sets):
                continue
            hyperparam_sets.append({
                'learning_rate': 3e-5, 'n_steps': n_steps, 'batch_size': batch_size,
                'gamma': 0.99, 'ent_coef': 0.01, 'clip_range': 0.2
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
            if ent == 0.01 and clip == 0.2:
                continue
            hyperparam_sets.append({
                'learning_rate': 3e-5, 'n_steps': 2048, 'batch_size': 128,
                'gamma': 0.99, 'ent_coef': ent, 'clip_range': clip
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
        if any(all(hp[k] == candidate[k] for k in candidate.keys()) for hp in hyperparam_sets):
            continue
        hyperparam_sets.append(candidate)
        random_count += 1

    console.print(f"   ✅ Added {random_count} random explorations")

    final_sets = hyperparam_sets[:n_bots]
    console.print(f"\n[bold green]✅ Generated {len(final_sets)} unique configurations[/bold green]\n")

    return final_sets


# ═════════════════════════════════════════════════════════════════════════════
# 2. SINGLE BOT TRAINING FUNCTION (AVEC BACKUP)
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
    """Trains a single bot with automatic Drive backup."""
    try:
        start_time = datetime.now()

        print(f"\n{'=' * 70}")
        print(f"🤖 BOT {bot_id}/{config.N_PARALLEL_BOTS} - DÉMARRAGE")
        print(f"{'=' * 70}")
        print(f"⏰ {start_time.strftime('%H:%M:%S')}")
        print(f"🎯 LR: {hyperparams['learning_rate']:.2e} | Gamma: {hyperparams['gamma']}")
        print(f"{'=' * 70}")

        # Training
        config.MODEL_HYPERPARAMETERS.update(hyperparams)
        trainer = AgentTrainer(df_historical=df_train)
        agent = trainer.train_offline(
            total_timesteps=config.TOTAL_TIMESTEPS_PER_BOT,
            use_early_stopping=False,
            seed=config.RANDOM_SEED + bot_id
        )

        training_duration = (datetime.now() - start_time).total_seconds()
        print(f"✅ Training: {training_duration / 60:.1f} min")

        # Save model locally
        model_filename = f"bot_{bot_id:03d}_lr{hyperparams['learning_rate']:.0e}_g{hyperparams['gamma']}.zip"
        model_path = os.path.join(config.MODEL_DIR, model_filename)
        agent.save(model_path)

        # ⭐ BACKUP GOOGLE DRIVE
        if backup_root:
            try:
                drive_model_path = f'{backup_root}/models/{model_filename}'
                shutil.copy(model_path, drive_model_path)
                print(f"💾 Drive backup: OK")
            except Exception as e:
                print(f"⚠️ Drive backup failed: {e}")

        # Evaluation
        train_metrics = evaluate_agent(agent, df_train)
        val_metrics = evaluate_agent(agent, df_val)
        test_metrics = evaluate_agent(agent, df_test)

        initial_capital = config.INITIAL_BALANCE
        test_profit = initial_capital * test_metrics[0]

        print(
            f"📊 Test Sharpe: {test_metrics[1]:.2f} | Return: {test_metrics[0] * 100:.2f}% | Profit: ${test_profit:.2f}")

        # Overfitting detection
        sharpe_gap = train_metrics[1] - val_metrics[1]
        overfit_status = (
            "SEVERE_OVERFIT" if sharpe_gap > 1.5 else
            "MILD_OVERFIT" if sharpe_gap > 0.8 else
            "UNDERFIT" if sharpe_gap < -0.5 else
            "GOOD_FIT"
        )

        # Commercial viability
        is_profitable = test_profit > 0
        meets_sharpe = test_metrics[1] >= config.MIN_ACCEPTABLE_SHARPE
        meets_calmar = test_metrics[3] >= config.MIN_ACCEPTABLE_CALMAR
        meets_dd = test_metrics[4] < config.MAX_ACCEPTABLE_DD
        commercial_score = sum([is_profitable, meets_sharpe, meets_calmar, meets_dd])
        commercial_status = "APPROVED" if commercial_score >= 3 else "CONDITIONAL" if commercial_score == 2 else "REJECTED"

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
# 3. MAIN TRAINING ORCHESTRATOR (AVEC CHECKPOINTS)
# ═════════════════════════════════════════════════════════════════════════════

def run_parallel_training():
    """Main training pipeline with checkpoint resume capability."""

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
    console.print("\n[cyan]☁️ Step 0: Google Drive Setup[/cyan]")
    backup_root = setup_google_drive_backup()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 0.5: Check for existing checkpoint
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]🔄 Checking for existing checkpoint...[/cyan]")
    checkpoint = load_checkpoint(backup_root)

    if checkpoint:
        completed_bots = checkpoint['completed_bots']
        previous_results = checkpoint['results']

        console.print(Panel.fit(
            f"[bold yellow]🔄 REPRISE DÉTECTÉE![/bold yellow]\n\n"
            f"Progression: {completed_bots}/{config.N_PARALLEL_BOTS} bots ({checkpoint['progress_pct']:.0f}%)\n"
            f"Date checkpoint: {checkpoint['timestamp']}\n"
            f"Bots restants: {config.N_PARALLEL_BOTS - completed_bots}",
            box=box.HEAVY,
            style="yellow"
        ))

        # Ask user confirmation
        console.print("\n[bold yellow]Voulez-vous reprendre depuis ce checkpoint? (o/n)[/bold yellow]")
        # Auto-accept in Colab (headless)
        resume = True
        console.print("[green]✅ Reprise automatique activée[/green]")
    else:
        console.print("[cyan]ℹ️ Aucun checkpoint trouvé - Nouveau training[/cyan]")
        completed_bots = 0
        previous_results = []
        resume = False

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Load Data
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]📂 Step 1: Loading Data[/cyan]")

    try:
        df_full = pd.read_csv(config.HISTORICAL_DATA_FILE)
        console.print(f"[green]✅ Loaded {len(df_full):,} bars[/green]")
    except FileNotFoundError:
        console.print(f"[yellow]⚠️ Using mock data[/yellow]")
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
        console.print("[yellow]⚠️ No GPU - using CPU[/yellow]")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Parallel Training (AVEC CHECKPOINTS)
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]🚀 Step 4: Parallel Training[/cyan]")

    # Determine which bots to train
    if resume:
        bots_to_train = list(range(completed_bots, config.N_PARALLEL_BOTS))
        console.print(f"[yellow]🔄 Reprise: Training bots {completed_bots + 1} à {config.N_PARALLEL_BOTS}[/yellow]")
    else:
        bots_to_train = list(range(config.N_PARALLEL_BOTS))
        console.print(f"[cyan]🆕 Nouveau: Training bots 1 à {config.N_PARALLEL_BOTS}[/cyan]")

    results = previous_results.copy()  # Start with previous results if resuming
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
                    backup_root  # ⭐ Pass backup_root
                ): i for i in bots_to_train
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                # ⭐ SAVE CHECKPOINT APRÈS CHAQUE BOT
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
# 4. COMPREHENSIVE REPORTING (AVEC BACKUP)
# ═════════════════════════════════════════════════════════════════════════════

def generate_comprehensive_report(df_results: pd.DataFrame, duration: float, backup_root: Optional[str] = None):
    """Generates reports with Drive backup."""

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

    # ⭐ BACKUP CSV TO DRIVE
    if backup_root:
        try:
            drive_csv_path = f'{backup_root}/results/results_{timestamp}.csv'
            shutil.copy(csv_path, drive_csv_path)
            console.print(f"[green]💾 Drive CSV: {drive_csv_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Drive CSV backup failed: {e}[/yellow]")

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

    # ⭐ BACKUP BEST MODEL TO DRIVE
    if backup_root:
        try:
            drive_best_path = f'{backup_root}/models/MODEL_PRODUCTION_BEST.zip'
            shutil.copy(prod_path, drive_best_path)
            console.print(f"[green]💾 Drive best model: {drive_best_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Drive best model backup failed: {e}[/yellow]")

    # Discord notification (si configuré)
    discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
    if discord_webhook:
        console.print("\n[cyan]📤 Sending to Discord...[/cyan]")
        try:
            from discord_uploader import send_discord_notification

            results_summary = {
                'total_bots': n_bots,
                'profitable_bots': int(n_profitable),
                'approved_bots': int(n_approved),
                'best_bot_id': int(best_bot['bot_id']),
                'best_sharpe': float(best_bot['test_sharpe']),
                'best_return': float(best_bot['test_return'] * 100),
                'best_profit': float(best_bot['test_profit_usd']),
                'avg_sharpe': float(df_results['test_sharpe'].mean()),
                'win_rate': float(n_profitable / n_bots * 100) if n_bots > 0 else 0.0,
                'duration_hours': duration / 3600
            }

            send_discord_notification(discord_webhook, results_summary, csv_path, prod_path)
            console.print("[bold green]✅ Discord notification sent![/bold green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Discord failed: {e}[/yellow]")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_parallel_training()