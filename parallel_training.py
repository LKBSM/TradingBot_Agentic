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
from rich.layout import Layout
from rich.live import Live

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
import config
from src.agent_trainer import AgentTrainer, evaluate_agent

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
# 1. INTELLIGENT HYPERPARAMETER GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_intelligent_hyperparam_sets(n_bots: int = 50) -> List[Dict]:
    """
    Generates hyperparameter sets using INTELLIGENT STRATIFIED SAMPLING.

    This is FAR better than random grid search because:
    1. Ensures good coverage of the search space
    2. Prioritizes important dimensions (LR, gamma)
    3. Includes research-backed baseline
    4. No redundant combinations

    Strategy:
    - 1 baseline (FinRL defaults)
    - 15 exploring LR × gamma (most important)
    - 12 exploring architecture (n_steps, batch_size)
    - 9 exploring exploration (entropy, clip)
    - 13 random for diversity

    Returns:
        List of n_bots hyperparameter dictionaries
    """
    console.print("\n[bold cyan]🧠 Generating Intelligent Hyperparameter Sets[/bold cyan]")
    console.print(f"   Target: {n_bots} unique configurations\n")

    # Extract search space
    lr_space = config.HYPERPARAM_SEARCH_SPACE['learning_rate']
    n_steps_space = config.HYPERPARAM_SEARCH_SPACE['n_steps']
    batch_space = config.HYPERPARAM_SEARCH_SPACE['batch_size']
    gamma_space = config.HYPERPARAM_SEARCH_SPACE['gamma']
    ent_space = config.HYPERPARAM_SEARCH_SPACE['ent_coef']
    clip_space = config.HYPERPARAM_SEARCH_SPACE['clip_range']

    hyperparam_sets = []

    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY 1: Baseline (FinRL Research Defaults)
    # ═══════════════════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY 2: LR × Gamma Grid (MOST IMPORTANT)
    # Research shows these 2 parameters have biggest impact
    # ═══════════════════════════════════════════════════════════════════════
    priority_count = 0
    for lr in lr_space:
        for gamma in gamma_space:
            if len(hyperparam_sets) >= n_bots:
                break
            # Skip baseline (already added)
            if lr == 3e-5 and gamma == 0.99:
                continue
            hyperparam_sets.append({
                'learning_rate': lr,
                'n_steps': 2048,  # Keep stable
                'batch_size': 128,  # Keep stable
                'gamma': gamma,
                'ent_coef': 0.01,  # Keep stable
                'clip_range': 0.2  # Keep stable
            })
            priority_count += 1
            if len(hyperparam_sets) >= 16:  # Take first 15 + baseline
                break
        if len(hyperparam_sets) >= 16:
            break

    console.print(f"   ✅ Added {priority_count} LR×Gamma variations")

    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY 3: Architecture Variations (n_steps, batch_size)
    # ═══════════════════════════════════════════════════════════════════════
    arch_count = 0
    for n_steps in n_steps_space:
        for batch_size in batch_space:
            if len(hyperparam_sets) >= 28:  # Stop at 28 total
                break
            # Constraint: batch_size must be <= n_steps
            if batch_size > n_steps:
                continue
            # Skip if already exists
            if any(hp['n_steps'] == n_steps and hp['batch_size'] == batch_size
                   for hp in hyperparam_sets):
                continue
            hyperparam_sets.append({
                'learning_rate': 3e-5,  # Stable
                'n_steps': n_steps,
                'batch_size': batch_size,
                'gamma': 0.99,  # Stable
                'ent_coef': 0.01,
                'clip_range': 0.2
            })
            arch_count += 1
        if len(hyperparam_sets) >= 28:
            break

    console.print(f"   ✅ Added {arch_count} architecture variations")

    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY 4: Exploration Parameters (entropy, clip)
    # ═══════════════════════════════════════════════════════════════════════
    explore_count = 0
    for ent in ent_space:
        for clip in clip_space:
            if len(hyperparam_sets) >= 37:  # Stop at 37 total
                break
            # Skip if baseline values
            if ent == 0.01 and clip == 0.2:
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

    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY 5: Random Sampling (Diversity)
    # ═══════════════════════════════════════════════════════════════════════
    random_count = 0
    np.random.seed(config.RANDOM_SEED)
    max_attempts = 1000
    attempts = 0

    while len(hyperparam_sets) < n_bots and attempts < max_attempts:
        attempts += 1

        # Random sample
        candidate = {
            'learning_rate': np.random.choice(lr_space),
            'n_steps': np.random.choice(n_steps_space),
            'batch_size': np.random.choice(batch_space),
            'gamma': np.random.choice(gamma_space),
            'ent_coef': np.random.choice(ent_space),
            'clip_range': np.random.choice(clip_space)
        }

        # Check constraint
        if candidate['batch_size'] > candidate['n_steps']:
            continue

        # Check if already exists
        if any(all(hp[k] == candidate[k] for k in candidate.keys())
               for hp in hyperparam_sets):
            continue

        hyperparam_sets.append(candidate)
        random_count += 1

    console.print(f"   ✅ Added {random_count} random explorations")

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL VALIDATION
    # ═══════════════════════════════════════════════════════════════════════
    final_sets = hyperparam_sets[:n_bots]  # Take exactly n_bots

    console.print(f"\n[bold green]✅ Generated {len(final_sets)} unique hyperparameter configurations[/bold green]")

    # Show distribution
    lr_dist = {}
    for hp in final_sets:
        lr = hp['learning_rate']
        lr_dist[lr] = lr_dist.get(lr, 0) + 1

    console.print("\n[cyan]📊 Distribution by Learning Rate:[/cyan]")
    for lr, count in sorted(lr_dist.items()):
        console.print(f"   {lr:.0e}: {count} bots")

    console.print()
    return final_sets


# ═════════════════════════════════════════════════════════════════════════════
# 2. SINGLE BOT TRAINING FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def train_single_bot(
        bot_id: int,
        hyperparams: Dict,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        verbose: bool = False
) -> Dict:
    """
    Trains a single bot with given hyperparameters.

    Returns comprehensive metrics including:
    - Training performance
    - Validation performance (for early stopping)
    - Test performance (out-of-sample, unseen)
    - Profit in USD
    - Commercial viability assessment
    """
    try:
        start_time = datetime.now()

        # Update config with bot's hyperparameters
        config.MODEL_HYPERPARAMETERS.update(hyperparams)

        # Create trainer
        trainer = AgentTrainer(df_historical=df_train)

        # Train with early stopping
        agent = trainer.train_offline(
            total_timesteps=config.TOTAL_TIMESTEPS_PER_BOT,
            use_early_stopping=False,  # Disabled for full training on Railway,
            seed=config.RANDOM_SEED + bot_id
        )

        training_duration = (datetime.now() - start_time).total_seconds()

        # Save model
        model_filename = f"bot_{bot_id:03d}_lr{hyperparams['learning_rate']:.0e}_g{hyperparams['gamma']}.zip"
        model_path = os.path.join(config.MODEL_DIR, model_filename)
        agent.save(model_path)

        # Evaluate on all three sets
        train_metrics = evaluate_agent(agent, df_train)
        val_metrics = evaluate_agent(agent, df_val)
        test_metrics = evaluate_agent(agent, df_test)

        # Calculate profits
        initial_capital = config.INITIAL_BALANCE

        train_profit = initial_capital * train_metrics[0]
        val_profit = initial_capital * val_metrics[0]
        test_profit = initial_capital * test_metrics[0]

        # Detect overfitting
        train_sharpe = train_metrics[1]
        val_sharpe = val_metrics[1]
        test_sharpe = test_metrics[1]

        sharpe_gap_train_val = train_sharpe - val_sharpe
        sharpe_gap_val_test = val_sharpe - test_sharpe

        if sharpe_gap_train_val > 1.5:
            overfit_status = "SEVERE_OVERFIT"
        elif sharpe_gap_train_val > 0.8:
            overfit_status = "MILD_OVERFIT"
        elif sharpe_gap_train_val < -0.5:
            overfit_status = "UNDERFIT"
        else:
            overfit_status = "GOOD_FIT"

        # Commercial viability check
        is_profitable = test_profit > 0
        meets_sharpe = test_sharpe >= config.MIN_ACCEPTABLE_SHARPE
        meets_calmar = test_metrics[3] >= config.MIN_ACCEPTABLE_CALMAR
        meets_dd = test_metrics[4] < config.MAX_ACCEPTABLE_DD

        commercial_score = sum([is_profitable, meets_sharpe, meets_calmar, meets_dd])

        if commercial_score >= 3:
            commercial_status = "APPROVED"
        elif commercial_score == 2:
            commercial_status = "CONDITIONAL"
        else:
            commercial_status = "REJECTED"

        # Compile results
        result = {
            'bot_id': bot_id,
            'model_path': model_path,
            'training_duration_sec': training_duration,

            # Hyperparameters
            'learning_rate': hyperparams['learning_rate'],
            'n_steps': hyperparams['n_steps'],
            'batch_size': hyperparams['batch_size'],
            'gamma': hyperparams['gamma'],
            'ent_coef': hyperparams['ent_coef'],
            'clip_range': hyperparams['clip_range'],

            # Train metrics
            'train_return': train_metrics[0],
            'train_sharpe': train_metrics[1],
            'train_profit_usd': train_profit,

            # Validation metrics
            'val_return': val_metrics[0],
            'val_sharpe': val_metrics[1],
            'val_profit_usd': val_profit,

            # Test metrics (MOST IMPORTANT - out-of-sample)
            'test_return': test_metrics[0],
            'test_sharpe': test_metrics[1],
            'test_sortino': test_metrics[2],
            'test_calmar': test_metrics[3],
            'test_max_dd': test_metrics[4],
            'test_profit_usd': test_profit,

            # Quality metrics
            'overfit_status': overfit_status,
            'sharpe_gap_train_val': sharpe_gap_train_val,
            'sharpe_gap_val_test': sharpe_gap_val_test,

            # Commercial assessment
            'is_profitable': is_profitable,
            'meets_sharpe_target': meets_sharpe,
            'meets_calmar_target': meets_calmar,
            'meets_dd_target': meets_dd,
            'commercial_status': commercial_status,
            'commercial_score': commercial_score,
            'overall_score': (test_sharpe + test_metrics[3]) / 2
        }

        if verbose:
            console.print(f"[green]✅ Bot {bot_id}: Complete![/green]")
            console.print(f"   Test Sharpe: {test_sharpe:.2f}")
            console.print(f"   Test Profit: ${test_profit:.2f}")
            console.print(f"   Status: {commercial_status}")
            console.print(f"   Duration: {training_duration:.0f}s\n")

        return result

    except Exception as e:
        console.print(f"[red]❌ Bot {bot_id}: FAILED - {str(e)}[/red]")
        return {
            'bot_id': bot_id,
            'error': str(e),
            'commercial_status': 'ERROR',
            'test_sharpe': -np.inf,
            'test_profit_usd': -np.inf,
            'is_profitable': False
        }


# ═════════════════════════════════════════════════════════════════════════════
# 3. MAIN TRAINING ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_parallel_training():
    """
    Main training pipeline:
    1. Load and split data
    2. Generate hyperparameters
    3. Train bots in parallel
    4. Generate reports
    5. Select best model
    """

    console.print(Panel.fit(
        "[bold white]🏢 XAU TRADING BOT - PRODUCTION TRAINING SYSTEM[/bold white]\n"
        f"Training {config.N_PARALLEL_BOTS} bots with {config.TOTAL_TIMESTEPS_PER_BOT:,} timesteps each\n"
        f"Target: Sharpe >{config.MIN_ACCEPTABLE_SHARPE} | Calmar >{config.MIN_ACCEPTABLE_CALMAR}",
        box=box.DOUBLE,
        style="cyan"
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Load Data
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]📂 Step 1: Loading Data[/cyan]")

    try:
        df_full = pd.read_csv(config.HISTORICAL_DATA_FILE)
        console.print(f"[green]✅ Loaded {len(df_full):,} bars from {config.HISTORICAL_DATA_FILE}[/green]")
    except FileNotFoundError:
        console.print(f"[yellow]⚠️  File not found: {config.HISTORICAL_DATA_FILE}[/yellow]")
        console.print("[yellow]   Using mock data for demonstration[/yellow]")

        # Mock data
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

    console.print(f"[cyan]   Training:   {len(df_train):,} bars ({config.TRAIN_RATIO:.0%})[/cyan]")
    console.print(f"[cyan]   Validation: {len(df_val):,} bars ({config.VAL_RATIO:.0%})[/cyan]")
    console.print(f"[cyan]   Test:       {len(df_test):,} bars ({config.TEST_RATIO:.0%})[/cyan]")

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
        console.print(f"[green]✅ GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)[/green]")
    else:
        console.print("[yellow]⚠️  No GPU detected - training will use CPU (slower)[/yellow]")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Parallel Training
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]🚀 Step 4: Parallel Training[/cyan]")
    console.print(f"   Max workers: {config.MAX_WORKERS_GPU}")
    console.print(
        f"   Estimated time: ~{(config.N_PARALLEL_BOTS * config.TOTAL_TIMESTEPS_PER_BOT) / (config.MAX_WORKERS_GPU * 200_000):.1f} hours\n")

    results = []
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
            f"[cyan]Training {config.N_PARALLEL_BOTS} bots...",
            total=config.N_PARALLEL_BOTS
        )

        with ProcessPoolExecutor(max_workers=config.MAX_WORKERS_GPU) as executor:
            futures = {
                executor.submit(
                    train_single_bot,
                    i + 1,
                    hyperparam_sets[i],
                    df_train,
                    df_val,
                    df_test
                ): i for i in range(config.N_PARALLEL_BOTS)
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                progress.update(task, advance=1)

    total_duration = (datetime.now() - start_time).total_seconds()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Process Results
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]📈 Step 5: Processing Results[/cyan]")

    df_results = pd.DataFrame(results)

    # Filter out failed bots (those with errors)
    df_valid = df_results[~df_results.get('error', pd.Series([None] * len(df_results))).notna()].copy()

    if len(df_valid) == 0:
        console.print("[red]❌ All bots failed! Check errors above.[/red]")
        console.print("\n[yellow]Common issues:[/yellow]")
        console.print("  1. Data file not found or incorrect format")
        console.print("  2. Column names mismatch (check OHLCV columns)")
        console.print("  3. Insufficient data after processing")
        return

    # Sort by overall score
    df_valid = df_valid.sort_values('overall_score', ascending=False)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Generate Report
    # ═══════════════════════════════════════════════════════════════════════
    console.print("\n[cyan]📄 Step 6: Generating Reports[/cyan]")
    generate_comprehensive_report(df_results, total_duration)

    console.print("\n[bold green]🎉 TRAINING COMPLETE![/bold green]\n")


# ═════════════════════════════════════════════════════════════════════════════
# 4. COMPREHENSIVE REPORTING
# ═════════════════════════════════════════════════════════════════════════════

def generate_comprehensive_report(df_results: pd.DataFrame, duration: float):
    """Generates console output and files"""

    n_bots = len(df_results)
    n_profitable = df_results['is_profitable'].sum()
    n_approved = (df_results['commercial_status'] == 'APPROVED').sum()

    # Console summary
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]📊 TRAINING RESULTS SUMMARY[/bold cyan]")
    console.print("=" * 80 + "\n")

    console.print(Panel.fit(
        f"[bold white]EXECUTIVE SUMMARY[/bold white]\n\n"
        f"Total Bots Trained: {n_bots}\n"
        f"Profitable Bots: {n_profitable} ({n_profitable / n_bots * 100:.1f}%)\n"
        f"Approved for Production: {n_approved} ({n_approved / n_bots * 100:.1f}%)\n\n"
        f"Average Test Sharpe: {df_results['test_sharpe'].mean():.2f}\n"
        f"Average Test Profit: ${df_results['test_profit_usd'].mean():,.2f}\n"
        f"Best Profit: ${df_results['test_profit_usd'].max():,.2f}\n\n"
        f"Total Training Time: {duration / 3600:.2f} hours",
        box=box.HEAVY,
        style="green"
    ))

    # Top 10 table
    console.print("\n[bold]🏆 TOP 10 PERFORMING BOTS[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Rank", style="cyan", justify="center")
    table.add_column("Bot ID", style="yellow")
    table.add_column("LR", justify="right")
    table.add_column("Test Sharpe", justify="right", style="green")
    table.add_column("Test Profit", justify="right", style="magenta")
    table.add_column("Max DD", justify="right", style="red")
    table.add_column("Status", justify="center")

    for rank, (idx, row) in enumerate(df_results.head(10).iterrows(), 1):
        profit_color = "green" if row['test_profit_usd'] > 0 else "red"
        status_icon = "✅" if row['commercial_status'] == 'APPROVED' else (
            "⚠️" if row['commercial_status'] == 'CONDITIONAL' else "❌")

        table.add_row(
            str(rank),
            f"Bot_{row['bot_id']:03d}",
            f"{row['learning_rate']:.1e}",
            f"{row['test_sharpe']:.2f}",
            f"[{profit_color}]${row['test_profit_usd']:,.2f}[/{profit_color}]",
            f"{row['test_max_dd']:.2%}",
            status_icon
        )

    console.print(table)

    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(config.RESULTS_DIR, f"parallel_training_results_{timestamp}.csv")
    df_results.to_csv(csv_path, index=False)
    console.print(f"\n[green]✅ Results saved to: {csv_path}[/green]")

    # Best model
    best_bot = df_results.iloc[0]
    console.print(f"\n[bold green]🎯 BEST MODEL FOR PRODUCTION:[/bold green]")
    console.print(f"   Bot ID: {best_bot['bot_id']}")
    console.print(f"   Test Sharpe: {best_bot['test_sharpe']:.2f}")
    console.print(f"   Test Profit: ${best_bot['test_profit_usd']:,.2f}")
    console.print(f"   Max Drawdown: {best_bot['test_max_dd']:.2%}")
    console.print(f"   Commercial Status: {best_bot['commercial_status']}")

    # Copy to production
    prod_path = os.path.join(config.MODEL_DIR, "MODEL_PRODUCTION_BEST.zip")
    shutil.copy(best_bot['model_path'], prod_path)
    console.print(f"\n[bold cyan]✅ Best model copied to: {prod_path}[/bold cyan]\n")

    # ═══════════════════════════════════════════════════════════════════════
    # DISCORD NOTIFICATION
    # ═══════════════════════════════════════════════════════════════════════
    discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
    if discord_webhook:
        console.print("\n[cyan]📤 Sending results to Discord...[/cyan]")
        try:
            from discord_uploader import send_discord_notification
            
            # Prepare results summary
            results_summary = {
                'total_bots': n_bots,
                'profitable_bots': int(n_profitable),
                'approved_bots': int(n_approved),
                'best_bot_id': int(best_bot['bot_id']),
                'best_sharpe': float(best_bot['test_sharpe']),
                'best_return': float(best_bot['test_return'] * 100),
                'best_drawdown': float(best_bot['test_max_dd'] * 100),
                'best_profit': float(best_bot['test_profit_usd']),
                'avg_sharpe': float(df_results['test_sharpe'].mean()),
                'avg_profit': float(df_results['test_profit_usd'].mean()),
                'win_rate': float(n_profitable / n_bots * 100) if n_bots > 0 else 0.0,
                'duration_hours': duration / 3600
            }
            
            # Send to Discord
            send_discord_notification(
                webhook_url=discord_webhook,
                results_summary=results_summary,
                csv_path=csv_path,
                model_path=prod_path
            )
            
            console.print("[bold green]✅ Results sent to Discord![/bold green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Discord upload failed: {e}[/yellow]")
            console.print(f"[yellow]   Results are still saved locally in: {config.RESULTS_DIR}[/yellow]")
    else:
        console.print("\n[yellow]⚠️ DISCORD_WEBHOOK_URL not set - skipping Discord notification[/yellow]")
        console.print(f"[yellow]   Results saved locally in: {config.RESULTS_DIR}[/yellow]")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_parallel_training()


