# ═════════════════════════════════════════════════════════════════════════════
# 🏢 PARALLEL TRAINING - PRODUCTION VERSION WITH WALK-FORWARD VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
#
# Features:
# - ✅ Walk-Forward Validation (critical for non-stationary markets)
# - ✅ Auto-mount Google Drive (no manual intervention)
# - ✅ Checkpoint resume capability (survive Colab disconnections)
# - ✅ Automatic model backups to Drive
# - ✅ Intelligent hyperparameter search
# - ✅ Multi-worker GPU training
# - ✅ Comprehensive reporting
# - ✅ Commercial-grade robustness testing
#
# Walk-Forward Validation Strategy:
#   - Rolling windows: Train on N months, validate on next M months
#   - Anchored expanding: Train from start to T, validate T to T+M
#   - Purged cross-validation: Gap between train/test to prevent leakage
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
# 📊 WALK-FORWARD VALIDATION CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# Walk-Forward Parameters (commercial-grade defaults)
WALK_FORWARD_CONFIG = {
    # Rolling window approach
    'train_window_bars': 6720,      # 6 months of 15-min data (~6720 bars)
    'validation_window_bars': 2240,  # 2 months validation (~2240 bars)
    'test_window_bars': 1120,        # 1 month test (~1120 bars)
    'step_size_bars': 1120,          # Slide by 1 month each fold
    'purge_gap_bars': 96,            # 1 day gap to prevent leakage (24h * 4)

    # Minimum requirements
    'min_folds': 3,                  # Minimum folds for statistical validity
    'max_folds': 12,                 # Maximum folds to prevent excessive compute

    # Validation strategy: 'rolling', 'expanding', or 'anchored'
    'strategy': 'rolling',

    # Early stopping across folds
    'early_stop_degradation_threshold': 0.3,  # Stop if Sharpe drops 30% from best
}


# ═════════════════════════════════════════════════════════════════════════════
# 🔄 WALK-FORWARD VALIDATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class WalkForwardValidator:
    """
    Commercial-grade Walk-Forward Validation for non-stationary financial data.

    Why Walk-Forward is Critical:
    - Markets are non-stationary: patterns that work today may fail tomorrow
    - Standard train/test split gives overly optimistic results
    - Walk-forward simulates real trading: train on past, test on future

    Strategies:
    1. Rolling Window: Fixed-size moving window
       [====TRAIN====][VAL][TEST]
                     [====TRAIN====][VAL][TEST]
                                   [====TRAIN====][VAL][TEST]

    2. Expanding Window: Growing training set (anchored at start)
       [==TRAIN==][VAL][TEST]
       [=====TRAIN=====][VAL][TEST]
       [========TRAIN========][VAL][TEST]

    3. Anchored: Fixed start, expanding end
       [TRAIN][VAL][TEST]
       [TRAIN===][VAL][TEST]
       [TRAIN======][VAL][TEST]
    """

    def __init__(self, df: pd.DataFrame, config: Dict = None):
        """
        Initialize Walk-Forward Validator.

        Args:
            df: Full historical DataFrame
            config: Walk-forward configuration dict
        """
        self.df = df
        self.config = config or WALK_FORWARD_CONFIG
        self.folds = []
        self.fold_results = []

    def generate_folds(self) -> List[Dict]:
        """
        Generate train/validation/test splits for walk-forward validation.

        Returns:
            List of fold dictionaries with indices
        """
        n = len(self.df)
        train_size = self.config['train_window_bars']
        val_size = self.config['validation_window_bars']
        test_size = self.config['test_window_bars']
        step_size = self.config['step_size_bars']
        purge_gap = self.config['purge_gap_bars']
        strategy = self.config['strategy']

        folds = []
        fold_id = 0

        if strategy == 'rolling':
            # Rolling window: fixed-size train window slides forward
            start = 0
            while True:
                train_start = start
                train_end = train_start + train_size
                val_start = train_end + purge_gap
                val_end = val_start + val_size
                test_start = val_end + purge_gap
                test_end = test_start + test_size

                if test_end > n:
                    break

                folds.append({
                    'fold_id': fold_id,
                    'train_start': train_start,
                    'train_end': train_end,
                    'val_start': val_start,
                    'val_end': val_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_bars': train_end - train_start,
                    'val_bars': val_end - val_start,
                    'test_bars': test_end - test_start,
                })

                fold_id += 1
                start += step_size

                if fold_id >= self.config['max_folds']:
                    break

        elif strategy == 'expanding':
            # Expanding window: train from beginning, grows each fold
            train_start = 0
            current_end = train_size

            while True:
                train_end = current_end
                val_start = train_end + purge_gap
                val_end = val_start + val_size
                test_start = val_end + purge_gap
                test_end = test_start + test_size

                if test_end > n:
                    break

                folds.append({
                    'fold_id': fold_id,
                    'train_start': train_start,
                    'train_end': train_end,
                    'val_start': val_start,
                    'val_end': val_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_bars': train_end - train_start,
                    'val_bars': val_end - val_start,
                    'test_bars': test_end - test_start,
                })

                fold_id += 1
                current_end += step_size

                if fold_id >= self.config['max_folds']:
                    break

        elif strategy == 'anchored':
            # Anchored: fixed start, variable end
            train_start = 0
            for fold_id in range(self.config['max_folds']):
                train_end = train_size + (fold_id * step_size)
                val_start = train_end + purge_gap
                val_end = val_start + val_size
                test_start = val_end + purge_gap
                test_end = test_start + test_size

                if test_end > n:
                    break

                folds.append({
                    'fold_id': fold_id,
                    'train_start': train_start,
                    'train_end': train_end,
                    'val_start': val_start,
                    'val_end': val_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_bars': train_end - train_start,
                    'val_bars': val_end - val_start,
                    'test_bars': test_end - test_start,
                })

        if len(folds) < self.config['min_folds']:
            console.print(f"[yellow]⚠️  Only {len(folds)} folds possible (min: {self.config['min_folds']})[/yellow]")
            console.print(f"[yellow]    Consider reducing window sizes or using more data[/yellow]")

        self.folds = folds
        return folds

    def get_fold_data(self, fold: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract train/val/test DataFrames for a specific fold.

        Args:
            fold: Fold dictionary with indices

        Returns:
            Tuple of (df_train, df_val, df_test)
        """
        df_train = self.df.iloc[fold['train_start']:fold['train_end']].copy()
        df_val = self.df.iloc[fold['val_start']:fold['val_end']].copy()
        df_test = self.df.iloc[fold['test_start']:fold['test_end']].copy()

        return df_train, df_val, df_test

    def print_fold_summary(self):
        """Print a summary of all folds."""
        if not self.folds:
            console.print("[red]No folds generated. Call generate_folds() first.[/red]")
            return

        table = Table(
            title="Walk-Forward Validation Folds",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        table.add_column("Fold", style="cyan", justify="center")
        table.add_column("Train Period", justify="center")
        table.add_column("Val Period", justify="center")
        table.add_column("Test Period", justify="center")
        table.add_column("Train Bars", justify="right")

        for fold in self.folds:
            # Convert bar indices to approximate dates if index is datetime
            if hasattr(self.df.index, 'strftime'):
                train_period = f"{self.df.index[fold['train_start']].strftime('%Y-%m-%d')} → {self.df.index[fold['train_end']-1].strftime('%Y-%m-%d')}"
                val_period = f"{self.df.index[fold['val_start']].strftime('%m-%d')} → {self.df.index[fold['val_end']-1].strftime('%m-%d')}"
                test_period = f"{self.df.index[fold['test_start']].strftime('%m-%d')} → {self.df.index[fold['test_end']-1].strftime('%m-%d')}"
            else:
                train_period = f"{fold['train_start']} → {fold['train_end']}"
                val_period = f"{fold['val_start']} → {fold['val_end']}"
                test_period = f"{fold['test_start']} → {fold['test_end']}"

            table.add_row(
                str(fold['fold_id']),
                train_period,
                val_period,
                test_period,
                f"{fold['train_bars']:,}"
            )

        console.print(table)
        console.print(f"\n[cyan]Strategy: {self.config['strategy'].upper()}[/cyan]")
        console.print(f"[cyan]Purge Gap: {self.config['purge_gap_bars']} bars (prevents leakage)[/cyan]\n")


def run_walk_forward_training(
    hyperparams: Dict,
    df_full: pd.DataFrame,
    wf_config: Dict = None,
    backup_root: Optional[str] = None,
    bot_id: int = 1
) -> Dict:
    """
    Run walk-forward training for a single hyperparameter configuration.

    This trains the model on multiple time periods and aggregates results
    to get a robust estimate of out-of-sample performance.

    MEMORY NOTE: When running in parallel via ProcessPoolExecutor, df_full is
    pickled and copied to each worker process. For large datasets, consider:
    - Reducing the number of parallel workers
    - Using memory-mapped files or shared memory
    - Loading data from disk in each worker

    Args:
        hyperparams: Hyperparameter configuration
        df_full: Full historical DataFrame
        wf_config: Walk-forward configuration
        backup_root: Drive backup path
        bot_id: Bot identifier

    Returns:
        Aggregated results across all folds
    """
    import gc

    wf_config = wf_config or WALK_FORWARD_CONFIG
    validator = WalkForwardValidator(df_full, wf_config)
    folds = validator.generate_folds()

    if len(folds) == 0:
        return {
            'bot_id': bot_id,
            'error': 'No valid folds could be generated',
            'commercial_status': 'ERROR'
        }

    console.print(f"\n[bold cyan]🔄 Walk-Forward Training: Bot {bot_id}[/bold cyan]")
    console.print(f"   Folds: {len(folds)} | Strategy: {wf_config['strategy']}")

    fold_results = []
    best_sharpe = -np.inf

    for fold in folds:
        console.print(f"\n[cyan]━━━ Fold {fold['fold_id'] + 1}/{len(folds)} ━━━[/cyan]")

        df_train, df_val, df_test = validator.get_fold_data(fold)

        # Train on this fold
        result = train_single_bot(
            bot_id=bot_id,
            hyperparams=hyperparams,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            backup_root=backup_root,
            verbose=False
        )

        result['fold_id'] = fold['fold_id']
        result['train_start'] = fold['train_start']
        result['train_end'] = fold['train_end']
        result['test_start'] = fold['test_start']
        result['test_end'] = fold['test_end']

        fold_results.append(result)

        # MEMORY FIX: Explicitly free fold data to reduce memory pressure
        del df_train, df_val, df_test
        gc.collect()

        # Track best and check for degradation
        if result.get('test_sharpe', -np.inf) > best_sharpe:
            best_sharpe = result['test_sharpe']

        # Early stopping if performance degraded significantly
        current_sharpe = result.get('test_sharpe', 0)
        if best_sharpe > 0 and current_sharpe < best_sharpe * (1 - wf_config['early_stop_degradation_threshold']):
            console.print(f"[yellow]⚠️  Early stopping: Sharpe degraded from {best_sharpe:.2f} to {current_sharpe:.2f}[/yellow]")
            break

    # Aggregate results across folds
    aggregated = aggregate_walk_forward_results(fold_results, bot_id, hyperparams)

    return aggregated


def aggregate_walk_forward_results(
    fold_results: List[Dict],
    bot_id: int,
    hyperparams: Dict
) -> Dict:
    """
    Aggregate results from multiple walk-forward folds.

    Uses robust statistics (median, IQR) instead of mean to handle outliers.

    Args:
        fold_results: List of results from each fold
        bot_id: Bot identifier
        hyperparams: Hyperparameter configuration

    Returns:
        Aggregated result dictionary
    """
    valid_results = [r for r in fold_results if 'error' not in r]

    if len(valid_results) == 0:
        return {
            'bot_id': bot_id,
            'error': 'All folds failed',
            'commercial_status': 'ERROR',
            **hyperparams
        }

    # Extract metrics from valid folds
    test_sharpes = [r.get('test_sharpe', 0) for r in valid_results]
    test_returns = [r.get('test_return', 0) for r in valid_results]
    test_profits = [r.get('test_profit_usd', 0) for r in valid_results]
    test_max_dds = [r.get('test_max_dd', 1) for r in valid_results]
    test_calmars = [r.get('test_calmar', 0) for r in valid_results]

    # Robust statistics (median + IQR for variance estimation)
    def robust_stats(values):
        arr = np.array(values)
        return {
            'median': float(np.median(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25))
        }

    sharpe_stats = robust_stats(test_sharpes)
    return_stats = robust_stats(test_returns)
    profit_stats = robust_stats(test_profits)
    dd_stats = robust_stats(test_max_dds)

    # Stability score: How consistent is performance across folds?
    # Lower coefficient of variation = more stable
    sharpe_cv = sharpe_stats['std'] / abs(sharpe_stats['mean']) if sharpe_stats['mean'] != 0 else np.inf
    stability_score = max(0, 1 - sharpe_cv)  # 0 = unstable, 1 = perfectly stable

    # Commercial viability based on walk-forward results
    # Use median (robust) instead of mean
    is_profitable = profit_stats['median'] > 0
    meets_sharpe = sharpe_stats['median'] >= config.MIN_ACCEPTABLE_SHARPE
    meets_dd = dd_stats['median'] < config.MAX_ACCEPTABLE_DD
    is_stable = stability_score > 0.5  # At least 50% stability

    commercial_score = sum([is_profitable, meets_sharpe, meets_dd, is_stable])

    if commercial_score >= 4:
        commercial_status = "APPROVED"
    elif commercial_score >= 3:
        commercial_status = "CONDITIONAL"
    else:
        commercial_status = "REJECTED"

    # Print summary
    console.print(f"\n[bold]📊 Walk-Forward Summary: Bot {bot_id}[/bold]")
    console.print(f"   Folds completed: {len(valid_results)}/{len(fold_results)}")
    console.print(f"   Median Sharpe: {sharpe_stats['median']:.2f} (±{sharpe_stats['std']:.2f})")
    console.print(f"   Median Profit: ${profit_stats['median']:,.2f}")
    console.print(f"   Stability Score: {stability_score:.2%}")
    console.print(f"   Status: {commercial_status}")

    return {
        'bot_id': bot_id,
        'model_path': valid_results[-1].get('model_path', ''),  # Use last fold's model

        # Hyperparameters
        **hyperparams,

        # Walk-forward aggregated metrics (use median for robustness)
        'wf_folds_completed': len(valid_results),
        'wf_folds_total': len(fold_results),

        'test_sharpe': sharpe_stats['median'],
        'test_sharpe_mean': sharpe_stats['mean'],
        'test_sharpe_std': sharpe_stats['std'],
        'test_sharpe_min': sharpe_stats['min'],
        'test_sharpe_max': sharpe_stats['max'],

        'test_return': return_stats['median'],
        'test_return_std': return_stats['std'],

        'test_profit_usd': profit_stats['median'],
        'test_profit_std': profit_stats['std'],
        'test_profit_total': sum(test_profits),  # Total across all folds

        'test_max_dd': dd_stats['median'],
        'test_max_dd_worst': dd_stats['max'],

        'test_calmar': float(np.median(test_calmars)),

        # Stability metrics
        'stability_score': stability_score,
        'sharpe_cv': sharpe_cv,

        # Commercial assessment
        'is_profitable': is_profitable,
        'meets_sharpe_target': meets_sharpe,
        'meets_dd_target': meets_dd,
        'is_stable': is_stable,
        'commercial_status': commercial_status,
        'commercial_score': commercial_score,

        # For ranking
        'overall_score': sharpe_stats['median'] * stability_score,  # Penalize unstable

        # Individual fold results for analysis
        'fold_results': fold_results,
    }


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

        # Evaluate - CRITICAL: Pass pre_fitted_scaler to prevent data leakage
        # The scaler was fitted on training data only during AgentTrainer init
        train_scaler = trainer.env_train.scaler
        train_metrics = evaluate_agent(agent, df_train, pre_fitted_scaler=train_scaler)
        val_metrics = evaluate_agent(agent, df_val, pre_fitted_scaler=train_scaler)
        test_metrics = evaluate_agent(agent, df_test, pre_fitted_scaler=train_scaler)

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

    # Check if data file exists BEFORE attempting to load
    if not os.path.exists(config.HISTORICAL_DATA_FILE):
        console.print(f"\n[bold red]{'='*70}[/bold red]")
        console.print(f"[bold red]❌ CRITICAL ERROR: DATA FILE NOT FOUND![/bold red]")
        console.print(f"[bold red]{'='*70}[/bold red]")
        console.print(f"\n[red]Expected file: {config.HISTORICAL_DATA_FILE}[/red]")
        console.print(f"[red]Data folder:   {config.DATA_DIR}[/red]")
        console.print(f"\n[yellow]To fix this:[/yellow]")
        console.print(f"[yellow]1. Upload your XAU 15-minute historical data CSV[/yellow]")
        console.print(f"[yellow]2. Rename it to: XAU_15MIN_2019_2024.csv[/yellow]")
        console.print(f"[yellow]3. Place it in: {config.DATA_DIR}[/yellow]")
        console.print(f"\n[yellow]Required CSV columns: Date, Open, High, Low, Close, Volume[/yellow]")
        console.print(f"[bold red]{'='*70}[/bold red]\n")
        raise FileNotFoundError(f"Training data not found: {config.HISTORICAL_DATA_FILE}")

    try:
        df_full = pd.read_csv(config.HISTORICAL_DATA_FILE)
        console.print(f"[green]✅ Loaded {len(df_full):,} bars from {config.HISTORICAL_DATA_FILE}[/green]")

        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df_full.columns and col.lower() not in df_full.columns]
        if missing_cols:
            console.print(f"[red]❌ Missing columns: {missing_cols}[/red]")
            raise ValueError(f"Data file missing required columns: {missing_cols}")

    except Exception as e:
        console.print(f"[red]❌ Failed to load data: {e}[/red]")
        raise

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
# 🚀 WALK-FORWARD PARALLEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def run_walk_forward_parallel_training(use_walk_forward: bool = True):
    """
    Main training pipeline with Walk-Forward Validation.

    This is the RECOMMENDED entry point for commercial training.
    Walk-forward validation provides realistic out-of-sample estimates.

    Args:
        use_walk_forward: If True, use walk-forward validation (recommended).
                         If False, use standard train/val/test split.
    """
    console.print(Panel.fit(
        "[bold white]🏢 XAU TRADING BOT - WALK-FORWARD PRODUCTION TRAINING[/bold white]\n\n"
        f"Training {config.N_PARALLEL_BOTS} bots × {config.TOTAL_TIMESTEPS_PER_BOT:,} timesteps\n"
        f"Walk-Forward: {'ENABLED ✅' if use_walk_forward else 'DISABLED ❌'}\n"
        f"Strategy: {WALK_FORWARD_CONFIG['strategy'].upper()}\n"
        f"Target: Sharpe >{config.MIN_ACCEPTABLE_SHARPE} | Stability >50%",
        box=box.DOUBLE,
        style="cyan"
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 0: Setup
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]☁️  Step 0: Setup[/cyan]")
    backup_root = setup_google_drive()
    checkpoint = load_checkpoint(backup_root)

    if checkpoint:
        completed_bots = checkpoint['completed_bots']
        previous_results = checkpoint['results']
        console.print(f"[green]✅ Resuming from checkpoint: {completed_bots}/{config.N_PARALLEL_BOTS}[/green]")
    else:
        completed_bots = 0
        previous_results = []

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Load Data
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]📂 Step 1: Loading Data[/cyan]")

    # Check if data file exists BEFORE attempting to load
    if not os.path.exists(config.HISTORICAL_DATA_FILE):
        console.print(f"\n[bold red]{'='*70}[/bold red]")
        console.print(f"[bold red]❌ CRITICAL ERROR: DATA FILE NOT FOUND![/bold red]")
        console.print(f"[bold red]{'='*70}[/bold red]")
        console.print(f"\n[red]Expected file: {config.HISTORICAL_DATA_FILE}[/red]")
        console.print(f"[red]Data folder:   {config.DATA_DIR}[/red]")
        console.print(f"\n[yellow]To fix this:[/yellow]")
        console.print(f"[yellow]1. Upload your XAU 15-minute historical data CSV[/yellow]")
        console.print(f"[yellow]2. Rename it to: XAU_15MIN_2019_2024.csv[/yellow]")
        console.print(f"[yellow]3. Place it in: {config.DATA_DIR}[/yellow]")
        console.print(f"\n[yellow]Required CSV columns: Date, Open, High, Low, Close, Volume[/yellow]")
        console.print(f"[bold red]{'='*70}[/bold red]\n")
        raise FileNotFoundError(f"Training data not found: {config.HISTORICAL_DATA_FILE}")

    try:
        df_full = pd.read_csv(config.HISTORICAL_DATA_FILE)
        # Ensure datetime index
        if 'Date' in df_full.columns:
            df_full['Date'] = pd.to_datetime(df_full['Date'])
            df_full.set_index('Date', inplace=True)
        console.print(f"[green]✅ Loaded {len(df_full):,} bars[/green]")

        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_cols = [c.lower() for c in df_full.columns] + [c.lower() for c in (df_full.index.names if hasattr(df_full.index, 'names') else [])]
        missing_cols = [col for col in required_cols if col.lower() not in df_cols]
        if missing_cols:
            console.print(f"[red]❌ Missing columns: {missing_cols}[/red]")
            raise ValueError(f"Data file missing required columns: {missing_cols}")

    except FileNotFoundError:
        raise
    except Exception as e:
        console.print(f"[red]❌ Failed to load data: {e}[/red]")
        raise

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Walk-Forward Setup
    # ═══════════════════════════════════════════════════════════════════════
    if use_walk_forward:
        console.print("\n[cyan]📊 Step 2: Walk-Forward Configuration[/cyan]")
        validator = WalkForwardValidator(df_full, WALK_FORWARD_CONFIG)
        folds = validator.generate_folds()
        validator.print_fold_summary()

        if len(folds) < WALK_FORWARD_CONFIG['min_folds']:
            console.print(f"[red]❌ Insufficient data for walk-forward validation[/red]")
            console.print(f"[red]   Need at least {WALK_FORWARD_CONFIG['min_folds']} folds[/red]")
            return
    else:
        # Standard split
        n = len(df_full)
        train_end = int(n * config.TRAIN_RATIO)
        val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
        df_train = df_full.iloc[:train_end].copy()
        df_val = df_full.iloc[train_end:val_end].copy()
        df_test = df_full.iloc[val_end:].copy()
        console.print(f"[cyan]   Standard split: Train {len(df_train):,} | Val {len(df_val):,} | Test {len(df_test):,}[/cyan]")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Generate Hyperparameters
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]📊 Step 3: Hyperparameter Generation[/cyan]")
    hyperparam_sets = generate_intelligent_hyperparam_sets(config.N_PARALLEL_BOTS)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: GPU Check
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]💻 Step 4: GPU Check[/cyan]")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[green]✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)[/green]")
    else:
        console.print("[yellow]⚠️  No GPU - using CPU[/yellow]")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Parallel Training
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]🚀 Step 5: Parallel Training[/cyan]")

    bots_to_train = list(range(completed_bots, config.N_PARALLEL_BOTS))
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

        # PERFORMANCE FIX: Use parallel execution for walk-forward training
        # Previously this was sequential, causing N x training_time overhead
        if use_walk_forward:
            # Parallel walk-forward training
            # Note: Each bot still processes folds sequentially for early stopping logic
            # but multiple bots train in parallel
            max_workers = min(config.MAX_WORKERS_GPU, len(bots_to_train))
            console.print(f"[cyan]   Using {max_workers} parallel workers for walk-forward training[/cyan]")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        run_walk_forward_training,
                        hyperparams=hyperparam_sets[i],
                        df_full=df_full,
                        wf_config=WALK_FORWARD_CONFIG,
                        backup_root=backup_root,
                        bot_id=i + 1
                    ): i for i in bots_to_train
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        save_checkpoint(len(results), results, backup_root)
                        progress.update(task, advance=1)
                    except Exception as e:
                        bot_idx = futures[future]
                        console.print(f"[red]Bot {bot_idx + 1} failed: {e}[/red]")
                        results.append({
                            'bot_id': bot_idx + 1,
                            'error': str(e),
                            'commercial_status': 'ERROR'
                        })
                        progress.update(task, advance=1)
        else:
            # Legacy parallel training (already parallel)
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
                    try:
                        result = future.result()
                        results.append(result)
                        save_checkpoint(len(results), results, backup_root)
                        progress.update(task, advance=1)
                    except Exception as e:
                        bot_idx = futures[future]
                        console.print(f"[red]Bot {bot_idx + 1} failed: {e}[/red]")
                        results.append({
                            'bot_id': bot_idx + 1,
                            'error': str(e),
                            'commercial_status': 'ERROR'
                        })
                        progress.update(task, advance=1)

    total_duration = (datetime.now() - start_time).total_seconds()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Process Results
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]📈 Step 6: Processing Results[/cyan]")

    df_results = pd.DataFrame(results)

    # Remove error entries
    df_valid = df_results[~df_results.get('error', pd.Series([None] * len(df_results))).notna()].copy()

    if len(df_valid) == 0:
        console.print("[red]❌ All bots failed![/red]")
        return

    # Sort by overall score (includes stability penalty for walk-forward)
    df_valid = df_valid.sort_values('overall_score', ascending=False)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: Generate Report
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]📄 Step 7: Generating Reports[/cyan]")

    if use_walk_forward:
        generate_walk_forward_report(df_valid, total_duration, backup_root)
    else:
        generate_comprehensive_report(df_valid, total_duration, backup_root)

    console.print("\n[bold green]🎉 TRAINING COMPLETE![/bold green]\n")


def generate_walk_forward_report(df_results: pd.DataFrame, duration: float, backup_root: Optional[str] = None):
    """Generate comprehensive walk-forward validation report."""

    n_bots = len(df_results)
    n_approved = (df_results['commercial_status'] == 'APPROVED').sum()
    n_stable = (df_results.get('is_stable', False)).sum()

    console.print("\n" + "=" * 80)
    console.print("[bold cyan]📊 WALK-FORWARD VALIDATION RESULTS[/bold cyan]")
    console.print("=" * 80 + "\n")

    # Executive summary with stability metrics
    avg_stability = df_results.get('stability_score', pd.Series([0])).mean()

    console.print(Panel.fit(
        f"[bold white]EXECUTIVE SUMMARY (Walk-Forward)[/bold white]\n\n"
        f"Total Bots Tested: {n_bots}\n"
        f"Approved (Production-Ready): {n_approved} ({n_approved/n_bots*100:.1f}%)\n"
        f"Stable (CV < 50%): {n_stable} ({n_stable/n_bots*100:.1f}%)\n\n"
        f"[bold]Aggregated Metrics (Median across folds):[/bold]\n"
        f"  Median Sharpe: {df_results['test_sharpe'].median():.2f}\n"
        f"  Avg Stability Score: {avg_stability:.2%}\n"
        f"  Best Total Profit: ${df_results['test_profit_total'].max():,.2f}\n\n"
        f"Duration: {duration/3600:.2f} hours",
        box=box.HEAVY,
        style="green"
    ))

    # Top 10 table with stability
    console.print("\n[bold]🏆 TOP 10 BOTS (Ranked by Sharpe × Stability)[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Rank", style="cyan", justify="center")
    table.add_column("Bot", style="yellow")
    table.add_column("Sharpe (Med)", justify="right")
    table.add_column("Sharpe (Std)", justify="right", style="dim")
    table.add_column("Stability", justify="right")
    table.add_column("Profit", justify="right")
    table.add_column("Folds", justify="center")
    table.add_column("Status", justify="center")

    for rank, (idx, row) in enumerate(df_results.head(10).iterrows(), 1):
        stability_color = "green" if row.get('stability_score', 0) > 0.7 else "yellow" if row.get('stability_score', 0) > 0.5 else "red"
        profit_color = "green" if row.get('test_profit_total', 0) > 0 else "red"
        status = "✅" if row['commercial_status'] == 'APPROVED' else "⚠️" if row['commercial_status'] == 'CONDITIONAL' else "❌"

        table.add_row(
            str(rank),
            f"Bot_{row['bot_id']:03d}",
            f"{row['test_sharpe']:.2f}",
            f"±{row.get('test_sharpe_std', 0):.2f}",
            f"[{stability_color}]{row.get('stability_score', 0):.0%}[/{stability_color}]",
            f"[{profit_color}]${row.get('test_profit_total', 0):,.0f}[/{profit_color}]",
            f"{row.get('wf_folds_completed', '?')}/{row.get('wf_folds_total', '?')}",
            status
        )

    console.print(table)

    # Save CSV with walk-forward metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(config.RESULTS_DIR, f"wf_results_{timestamp}.csv")

    # Remove nested fold_results for CSV export
    export_df = df_results.drop(columns=['fold_results'], errors='ignore')
    export_df.to_csv(csv_path, index=False)
    console.print(f"\n[green]✅ CSV: {csv_path}[/green]")

    # Backup to Drive
    if backup_root:
        try:
            drive_csv = os.path.join(backup_root, 'results', f'wf_results_{timestamp}.csv')
            shutil.copy(csv_path, drive_csv)
            console.print(f"[green]💾 Drive backup: {drive_csv}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Drive backup failed: {e}[/yellow]")

    # Best model recommendation
    best_bot = df_results.iloc[0]
    console.print(f"\n[bold green]🎯 RECOMMENDED PRODUCTION MODEL[/bold green]")
    console.print(f"   Bot: {best_bot['bot_id']}")
    console.print(f"   Median Sharpe: {best_bot['test_sharpe']:.2f} (across {best_bot.get('wf_folds_completed', '?')} folds)")
    console.print(f"   Stability: {best_bot.get('stability_score', 0):.0%}")
    console.print(f"   Total Profit: ${best_bot.get('test_profit_total', 0):,.2f}")
    console.print(f"   Status: {best_bot['commercial_status']}")

    # Copy best model
    if best_bot.get('model_path'):
        prod_path = os.path.join(config.MODEL_DIR, "MODEL_PRODUCTION_BEST_WF.zip")
        try:
            shutil.copy(best_bot['model_path'], prod_path)
            console.print(f"\n[bold cyan]✅ Best model saved: {prod_path}[/bold cyan]")

            if backup_root:
                drive_best = os.path.join(backup_root, 'models', 'MODEL_PRODUCTION_BEST_WF.zip')
                shutil.copy(prod_path, drive_best)
        except Exception as e:
            console.print(f"[yellow]⚠️  Model copy failed: {e}[/yellow]")

    # Warning if best model is not approved
    if best_bot['commercial_status'] != 'APPROVED':
        console.print(f"\n[bold yellow]⚠️  WARNING: Best model is {best_bot['commercial_status']}[/bold yellow]")
        console.print(f"[yellow]   Consider adjusting hyperparameters or collecting more data[/yellow]")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='XAU Trading Bot Training')
    parser.add_argument('--walk-forward', '-wf', action='store_true', default=True,
                       help='Use walk-forward validation (recommended, default: True)')
    parser.add_argument('--no-walk-forward', action='store_true',
                       help='Disable walk-forward validation (use standard split)')
    parser.add_argument('--legacy', action='store_true',
                       help='Use legacy training without walk-forward')

    args = parser.parse_args()

    if args.no_walk_forward or args.legacy:
        console.print("[yellow]Using legacy training (no walk-forward)[/yellow]")
        run_parallel_training()
    else:
        run_walk_forward_parallel_training(use_walk_forward=True)