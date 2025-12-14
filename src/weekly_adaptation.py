import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import warnings
import shutil
from typing import Tuple, Dict, Optional
import json

# Rich console for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
import config
from src.agent_trainer import AgentTrainer, evaluate_agent
from src.environment.environment import TradingEnv
from stable_baselines3 import PPO

# Setup
console = Console()
warnings.filterwarnings('ignore')

# Configure logging
log_file = os.path.join(config.LOG_DIR, f"weekly_adaptation_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

class AdaptationConfig:
    """Configuration for weekly adaptation process"""

    # Time window for recent data (how many weeks to use)
    LOOKBACK_WEEKS = 4  # Use last 4 weeks of data

    # Fine-tuning parameters
    FINE_TUNE_TIMESTEPS = 100_000  # Conservative - prevents overfitting

    # Validation thresholds
    MIN_ACCEPTABLE_SHARPE = 1.3  # Lower than core model (we're being defensive)
    MAX_TRAIN_VAL_GAP = 0.5  # Max acceptable difference (overfitting detector)
    MIN_IMPROVEMENT_PCT = -10.0  # Allow 10% degradation (market changes)

    # Model paths
    CORE_MODEL_PATH = os.path.join(config.MODEL_DIR, "MODEL_PRODUCTION_BEST.zip")
    ADAPTIVE_MODEL_DIR = os.path.join(config.MODEL_DIR, "adaptive_models")
    BACKUP_DIR = os.path.join(config.MODEL_DIR, "backups")

    # Alert settings
    ENABLE_EMAIL_ALERTS = False  # Set to True in production
    ALERT_EMAIL = "your_email@example.com"

    # Safety settings
    ENABLE_PAPER_TRADING_TEST = True  # Test on paper before live deployment
    PAPER_TEST_DURATION_HOURS = 24  # Test for 24 hours before going live


# ═════════════════════════════════════════════════════════════════════════════
# 2. DATA MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

def load_recent_data(weeks: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads recent market data and splits into train/val/test sets.

    Args:
        weeks: Number of weeks of recent data to load

    Returns:
        tuple: (df_train, df_val, df_test)

    Why 4 weeks?
    - Enough data to detect regime changes (~2,700 bars at 15min)
    - Recent enough to capture current market conditions
    - Not too old (2-month old data may be obsolete)
    """
    console.print("\n[cyan]📂 Step 1: Loading Recent Market Data[/cyan]")

    try:
        # Load full historical data
        df_full = pd.read_csv(config.HISTORICAL_DATA_FILE)
        console.print(f"   ✅ Loaded {len(df_full):,} bars from database")

        # Calculate date range for recent data
        # Assuming data is sorted by date (most recent last)
        n_bars_needed = weeks * 7 * 96  # 96 bars per day (15min intervals)

        if len(df_full) < n_bars_needed:
            console.print(f"   [yellow]⚠️  Only {len(df_full)} bars available (need {n_bars_needed})[/yellow]")
            n_bars_needed = len(df_full)

        # Extract recent data
        df_recent = df_full.iloc[-n_bars_needed:].copy()
        df_recent.reset_index(drop=True, inplace=True)

        console.print(f"   ✅ Extracted {len(df_recent):,} recent bars ({weeks} weeks)")

        # Split into train/val/test (70/15/15)
        n = len(df_recent)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        df_train = df_recent.iloc[:train_end].copy()
        df_val = df_recent.iloc[train_end:val_end].copy()
        df_test = df_recent.iloc[val_end:].copy()

        console.print(f"   📊 Split: Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")

        # Validation checks
        if len(df_train) < config.LOOKBACK_WINDOW_SIZE * 2:
            raise ValueError(f"Insufficient training data: {len(df_train)} bars")

        if len(df_val) < config.LOOKBACK_WINDOW_SIZE:
            raise ValueError(f"Insufficient validation data: {len(df_val)} bars")

        logger.info(f"Successfully loaded {weeks} weeks of recent data")
        return df_train, df_val, df_test

    except FileNotFoundError:
        console.print(f"[red]❌ Data file not found: {config.HISTORICAL_DATA_FILE}[/red]")
        logger.error(f"Data file not found: {config.HISTORICAL_DATA_FILE}")
        raise
    except Exception as e:
        console.print(f"[red]❌ Error loading data: {str(e)}[/red]")
        logger.error(f"Data loading error: {str(e)}")
        raise


def get_current_production_model_path() -> str:
    """
    Returns the path to the currently deployed production model.

    Priority order:
    1. Latest adaptive model (from last week)
    2. Core model (if no adaptive model exists)

    Returns:
        str: Path to model file
    """
    adaptive_dir = AdaptationConfig.ADAPTIVE_MODEL_DIR

    if os.path.exists(adaptive_dir):
        # Find most recent adaptive model
        adaptive_models = sorted(
            [f for f in os.listdir(adaptive_dir) if f.endswith('.zip')],
            reverse=True  # Most recent first
        )

        if adaptive_models:
            latest_model = os.path.join(adaptive_dir, adaptive_models[0])
            console.print(f"   📦 Current production model: {adaptive_models[0]}")
            logger.info(f"Using adaptive model: {adaptive_models[0]}")
            return latest_model

    # Fallback to core model
    if os.path.exists(AdaptationConfig.CORE_MODEL_PATH):
        console.print(f"   📦 Using core model (no adaptive model found)")
        logger.info("Using core model")
        return AdaptationConfig.CORE_MODEL_PATH

    raise FileNotFoundError("No production model found! Run parallel_training.py first.")


# ═════════════════════════════════════════════════════════════════════════════
# 3. FINE-TUNING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def fine_tune_model(
        base_model_path: str,
        df_train: pd.DataFrame,
        timesteps: int = 100_000
) -> PPO:
    """
    Fine-tunes the base model on recent data.

    Args:
        base_model_path: Path to the model to fine-tune
        df_train: Recent training data
        timesteps: Number of training timesteps (keep low to avoid overfitting)

    Returns:
        PPO: Fine-tuned agent

    CRITICAL: Uses SMALL number of timesteps to prevent catastrophic forgetting
    """
    console.print("\n[cyan]🎯 Step 2: Fine-Tuning Model[/cyan]")
    console.print(f"   Base model: {os.path.basename(base_model_path)}")
    console.print(f"   Training bars: {len(df_train):,}")
    console.print(f"   Timesteps: {timesteps:,}")

    try:
        # Create trainer with recent data
        trainer = AgentTrainer(df_historical=df_train)

        # Load base model
        console.print(f"   📥 Loading base model...")
        trainer.agent = PPO.load(base_model_path, env=trainer.env_train)
        logger.info(f"Loaded base model: {base_model_path}")

        # Fine-tune
        console.print(f"   🔄 Fine-tuning for {timesteps:,} timesteps...")

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Fine-tuning...",
                total=timesteps
            )

            # Perform fine-tuning
            trainer.agent.learn(
                total_timesteps=timesteps,
                reset_num_timesteps=False,  # Continue from base model
                progress_bar=False  # We have our own progress bar
            )

            progress.update(task, completed=timesteps)

        console.print(f"   ✅ Fine-tuning complete!")
        logger.info(f"Fine-tuning completed: {timesteps} timesteps")

        return trainer.agent

    except Exception as e:
        console.print(f"[red]❌ Fine-tuning failed: {str(e)}[/red]")
        logger.error(f"Fine-tuning error: {str(e)}")
        raise


# ═════════════════════════════════════════════════════════════════════════════
# 4. VALIDATION & QUALITY CONTROL
# ═════════════════════════════════════════════════════════════════════════════

def validate_model(
        agent: PPO,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        base_model_path: str
) -> Tuple[bool, Dict]:
    """
    Validates the fine-tuned model against safety criteria.

    Validation Checks:
    1. Performance threshold: Sharpe > MIN_ACCEPTABLE_SHARPE
    2. Overfitting detection: train_sharpe - val_sharpe < MAX_TRAIN_VAL_GAP
    3. Regression check: Not significantly worse than base model
    4. Test set validation: Out-of-sample performance

    Args:
        agent: Fine-tuned agent to validate
        df_train: Training data
        df_val: Validation data
        df_test: Test data
        base_model_path: Path to base model (for comparison)

    Returns:
        tuple: (is_valid: bool, metrics: dict)
    """
    console.print("\n[cyan]🔍 Step 3: Validating Fine-Tuned Model[/cyan]")

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluate fine-tuned model on all sets
    # ─────────────────────────────────────────────────────────────────────────
    console.print("   📊 Evaluating fine-tuned model...")

    train_metrics = evaluate_agent(agent, df_train)
    val_metrics = evaluate_agent(agent, df_val)
    test_metrics = evaluate_agent(agent, df_test)

    # Extract metrics
    ft_train_sharpe = train_metrics[1]
    ft_val_sharpe = val_metrics[1]
    ft_test_sharpe = test_metrics[1]
    ft_test_return = test_metrics[0]
    ft_test_calmar = test_metrics[3]
    ft_test_dd = test_metrics[4]

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluate base model for comparison
    # ─────────────────────────────────────────────────────────────────────────
    console.print("   📊 Evaluating base model (for comparison)...")
    base_agent = PPO.load(base_model_path)

    base_val_metrics = evaluate_agent(base_agent, df_val)
    base_test_metrics = evaluate_agent(base_agent, df_test)

    base_val_sharpe = base_val_metrics[1]
    base_test_sharpe = base_test_metrics[1]

    # ─────────────────────────────────────────────────────────────────────────
    # Calculate validation criteria
    # ─────────────────────────────────────────────────────────────────────────
    train_val_gap = ft_train_sharpe - ft_val_sharpe
    improvement_vs_base = ((ft_val_sharpe - base_val_sharpe) / abs(base_val_sharpe)) * 100

    # ─────────────────────────────────────────────────────────────────────────
    # Display results
    # ─────────────────────────────────────────────────────────────────────────
    table = Table(title="📈 Validation Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Fine-Tuned", justify="right", style="green")
    table.add_column("Base Model", justify="right", style="yellow")
    table.add_column("Change", justify="right")

    table.add_row(
        "Train Sharpe",
        f"{ft_train_sharpe:.2f}",
        "N/A",
        ""
    )
    table.add_row(
        "Val Sharpe",
        f"{ft_val_sharpe:.2f}",
        f"{base_val_sharpe:.2f}",
        f"{improvement_vs_base:+.1f}%"
    )
    table.add_row(
        "Test Sharpe",
        f"{ft_test_sharpe:.2f}",
        f"{base_test_sharpe:.2f}",
        f"{((ft_test_sharpe - base_test_sharpe) / abs(base_test_sharpe) * 100):+.1f}%"
    )
    table.add_row(
        "Train/Val Gap",
        f"{train_val_gap:.2f}",
        "",
        "✅" if train_val_gap < AdaptationConfig.MAX_TRAIN_VAL_GAP else "❌"
    )
    table.add_row(
        "Test Return",
        f"{ft_test_return:.2%}",
        f"{base_test_metrics[0]:.2%}",
        ""
    )
    table.add_row(
        "Max Drawdown",
        f"{ft_test_dd:.2%}",
        f"{base_test_metrics[4]:.2%}",
        ""
    )

    console.print(table)

    # ─────────────────────────────────────────────────────────────────────────
    # Validation checks
    # ─────────────────────────────────────────────────────────────────────────
    checks = {
        'sharpe_threshold': ft_val_sharpe >= AdaptationConfig.MIN_ACCEPTABLE_SHARPE,
        'overfitting_check': train_val_gap < AdaptationConfig.MAX_TRAIN_VAL_GAP,
        'regression_check': improvement_vs_base >= AdaptationConfig.MIN_IMPROVEMENT_PCT,
        'test_performance': ft_test_sharpe > 0  # At least positive
    }

    # Display check results
    console.print("\n   🔍 Validation Checks:")
    console.print(
        f"      {'✅' if checks['sharpe_threshold'] else '❌'} Sharpe Threshold: {ft_val_sharpe:.2f} >= {AdaptationConfig.MIN_ACCEPTABLE_SHARPE}")
    console.print(
        f"      {'✅' if checks['overfitting_check'] else '❌'} Overfitting Check: Gap {train_val_gap:.2f} < {AdaptationConfig.MAX_TRAIN_VAL_GAP}")
    console.print(
        f"      {'✅' if checks['regression_check'] else '❌'} Regression Check: {improvement_vs_base:+.1f}% >= {AdaptationConfig.MIN_IMPROVEMENT_PCT}%")
    console.print(
        f"      {'✅' if checks['test_performance'] else '❌'} Test Performance: Sharpe {ft_test_sharpe:.2f} > 0")

    is_valid = all(checks.values())

    # Compile metrics
    metrics = {
        'train_sharpe': ft_train_sharpe,
        'val_sharpe': ft_val_sharpe,
        'test_sharpe': ft_test_sharpe,
        'test_return': ft_test_return,
        'test_calmar': ft_test_calmar,
        'test_max_dd': ft_test_dd,
        'train_val_gap': train_val_gap,
        'improvement_vs_base': improvement_vs_base,
        'base_val_sharpe': base_val_sharpe,
        'base_test_sharpe': base_test_sharpe,
        'validation_passed': is_valid,
        'checks': checks
    }

    # Log results
    if is_valid:
        console.print("\n   [bold green]✅ VALIDATION PASSED - Model ready for deployment[/bold green]")
        logger.info("Model validation PASSED")
    else:
        console.print("\n   [bold red]❌ VALIDATION FAILED - Keeping previous model[/bold red]")
        logger.warning("Model validation FAILED")

        # Identify which checks failed
        failed_checks = [k for k, v in checks.items() if not v]
        logger.warning(f"Failed checks: {failed_checks}")

    return is_valid, metrics


# ═════════════════════════════════════════════════════════════════════════════
# 5. DEPLOYMENT & BACKUP
# ═════════════════════════════════════════════════════════════════════════════

def deploy_model(agent: PPO, metrics: Dict) -> str:
    """
    Deploys validated model to production.

    Steps:
    1. Backup current production model
    2. Save new adaptive model with timestamp
    3. Update production symlink
    4. Log deployment

    Args:
        agent: Validated fine-tuned agent
        metrics: Validation metrics

    Returns:
        str: Path to deployed model
    """
    console.print("\n[cyan]🚀 Step 4: Deploying Model to Production[/cyan]")

    # Create directories if they don't exist
    os.makedirs(AdaptationConfig.ADAPTIVE_MODEL_DIR, exist_ok=True)
    os.makedirs(AdaptationConfig.BACKUP_DIR, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Backup current production model
    # ─────────────────────────────────────────────────────────────────────────
    try:
        current_model = get_current_production_model_path()
        timestamp_backup = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(
            AdaptationConfig.BACKUP_DIR,
            f"backup_{timestamp_backup}_{os.path.basename(current_model)}"
        )
        shutil.copy(current_model, backup_path)
        console.print(f"   💾 Backed up current model to: {os.path.basename(backup_path)}")
        logger.info(f"Backed up current model: {backup_path}")
    except Exception as e:
        console.print(f"   [yellow]⚠️  Could not backup current model: {str(e)}[/yellow]")
        logger.warning(f"Backup failed: {str(e)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Save new adaptive model with timestamp and metrics
    # ─────────────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    week_number = datetime.now().isocalendar()[1]
    year = datetime.now().year

    model_filename = f"adaptive_week_{week_number}_{year}_{timestamp}.zip"
    model_path = os.path.join(AdaptationConfig.ADAPTIVE_MODEL_DIR, model_filename)

    agent.save(model_path)
    console.print(f"   💾 Saved adaptive model: {model_filename}")
    logger.info(f"Deployed new model: {model_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Save metadata (metrics) alongside model
    # ─────────────────────────────────────────────────────────────────────────
    metadata = {
        'deployment_time': datetime.now().isoformat(),
        'week_number': week_number,
        'year': year,
        'model_path': model_path,
        'metrics': {
            'val_sharpe': float(metrics['val_sharpe']),
            'test_sharpe': float(metrics['test_sharpe']),
            'test_return': float(metrics['test_return']),
            'test_max_dd': float(metrics['test_max_dd']),
            'improvement_vs_base': float(metrics['improvement_vs_base'])
        },
        'validation_checks': metrics['checks']
    }

    metadata_path = model_path.replace('.zip', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    console.print(f"   📄 Saved metadata: {os.path.basename(metadata_path)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Update production symlink (or copy to fixed name)
    # ─────────────────────────────────────────────────────────────────────────
    production_link = os.path.join(config.MODEL_DIR, "MODEL_PRODUCTION_CURRENT.zip")

    try:
        # Remove old link if exists
        if os.path.exists(production_link):
            os.remove(production_link)

        # Create new link (or copy on Windows)
        if os.name == 'nt':  # Windows
            shutil.copy(model_path, production_link)
        else:  # Unix/Linux/Mac
            os.symlink(model_path, production_link)

        console.print(f"   🔗 Updated production link: MODEL_PRODUCTION_CURRENT.zip")
        logger.info(f"Updated production link to: {model_path}")
    except Exception as e:
        console.print(f"   [yellow]⚠️  Could not update production link: {str(e)}[/yellow]")
        logger.warning(f"Production link update failed: {str(e)}")

    console.print("\n   [bold green]✅ DEPLOYMENT COMPLETE[/bold green]")

    return model_path


def cleanup_old_models(keep_last_n: int = 10):
    """
    Removes old adaptive models to save disk space.
    Keeps the last N models for rollback capability.

    Args:
        keep_last_n: Number of recent models to keep
    """
    console.print(f"\n[cyan]🧹 Cleaning up old models (keeping last {keep_last_n})[/cyan]")

    adaptive_dir = AdaptationConfig.ADAPTIVE_MODEL_DIR

    if not os.path.exists(adaptive_dir):
        return

    # Get all adaptive models sorted by creation time
    models = []
    for f in os.listdir(adaptive_dir):
        if f.endswith('.zip'):
            full_path = os.path.join(adaptive_dir, f)
            models.append((full_path, os.path.getctime(full_path)))

    # Sort by creation time (newest first)
    models.sort(key=lambda x: x[1], reverse=True)

    # Delete old models
    deleted_count = 0
    for model_path, _ in models[keep_last_n:]:
        try:
            os.remove(model_path)

            # Also remove metadata if exists
            metadata_path = model_path.replace('.zip', '_metadata.json')
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

            deleted_count += 1
            logger.info(f"Deleted old model: {os.path.basename(model_path)}")
        except Exception as e:
            logger.warning(f"Could not delete {model_path}: {str(e)}")

    if deleted_count > 0:
        console.print(f"   🗑️  Removed {deleted_count} old model(s)")
    else:
        console.print(f"   ✅ No cleanup needed")


# ═════════════════════════════════════════════════════════════════════════════
# 6. MAIN WORKFLOW
# ═════════════════════════════════════════════════════════════════════════════

def run_weekly_adaptation():
    """
    Main workflow for weekly adaptation.

    Steps:
    1. Load recent data (4 weeks)
    2. Fine-tune base model
    3. Validate performance
    4. Deploy if validated, rollback otherwise
    5. Cleanup old models
    """

    # ═════════════════════════════════════════════════════════════════════════
    # BANNER
    # ═════════════════════════════════════════════════════════════════════════
    console.print(Panel.fit(
        "[bold white]🔄 WEEKLY ADAPTATION PROCESS[/bold white]\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Week: {datetime.now().isocalendar()[1]}\n"
        f"Purpose: Fine-tune model on recent {AdaptationConfig.LOOKBACK_WEEKS} weeks",
        box=box.DOUBLE,
        style="cyan"
    ))

    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("WEEKLY ADAPTATION STARTED")
    logger.info("=" * 70)

    try:
        # ═════════════════════════════════════════════════════════════════════
        # STEP 1: Load Recent Data
        # ═════════════════════════════════════════════════════════════════════
        df_train, df_val, df_test = load_recent_data(weeks=AdaptationConfig.LOOKBACK_WEEKS)

        # ═════════════════════════════════════════════════════════════════════
        # STEP 2: Get Current Production Model
        # ═════════════════════════════════════════════════════════════════════
        console.print("\n[cyan]📦 Locating Current Production Model[/cyan]")
        base_model_path = get_current_production_model_path()

        # ═════════════════════════════════════════════════════════════════════
        # STEP 3: Fine-Tune Model
        # ═════════════════════════════════════════════════════════════════════
        fine_tuned_agent = fine_tune_model(
            base_model_path=base_model_path,
            df_train=df_train,
            timesteps=AdaptationConfig.FINE_TUNE_TIMESTEPS
        )

        # ═════════════════════════════════════════════════════════════════════
        # STEP 4: Validate Fine-Tuned Model
        # ═════════════════════════════════════════════════════════════════════
        is_valid, metrics = validate_model(
            agent=fine_tuned_agent,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            base_model_path=base_model_path
        )

        # ═════════════════════════════════════════════════════════════════════
        # STEP 5: Deploy or Rollback
        # ═════════════════════════════════════════════════════════════════════
        if is_valid:
            deployed_path = deploy_model(fine_tuned_agent, metrics)

            # ═════════════════════════════════════════════════════════════════
            # STEP 6: Cleanup
            # ═════════════════════════════════════════════════════════════════
            cleanup_old_models(keep_last_n=10)

            # ═════════════════════════════════════════════════════════════════
            # SUCCESS SUMMARY
            # ═════════════════════════════════════════════════════════════════
            duration = (datetime.now() - start_time).total_seconds() / 60

            console.print("\n" + "=" * 70)
            console.print(Panel.fit(
                "[bold green]✅ WEEKLY ADAPTATION SUCCESSFUL[/bold green]\n\n"
                f"New Model Deployed: {os.path.basename(deployed_path)}\n"
                f"Validation Sharpe: {metrics['val_sharpe']:.2f}\n"
                f"Test Sharpe: {metrics['test_sharpe']:.2f}\n"
                f"Improvement: {metrics['improvement_vs_base']:+.1f}%\n"
                f"Duration: {duration:.1f} minutes",
                box=box.HEAVY,
                style="green"
            ))

            logger.info(f"Weekly adaptation completed successfully in {duration:.1f} minutes")
            logger.info(f"Deployed model: {deployed_path}")

        else:
            # ═════════════════════════════════════════════════════════════════
            # ROLLBACK (Validation Failed)
            # ═════════════════════════════════════════════════════════════════
            console.print("\n" + "=" * 70)
            console.print(Panel.fit(
                "[bold yellow]⚠️  VALIDATION FAILED - ROLLBACK[/bold yellow]\n\n"
                "Fine-tuned model did not meet quality criteria.\n"
                "Previous production model remains active.\n\n"
                "Failed Checks:\n" +
                "\n".join([f"  ❌ {k}" for k, v in metrics['checks'].items() if not v]),
                box=box.HEAVY,
                style="yellow"
            ))

            logger.warning("Weekly adaptation failed validation - rollback to previous model")

    except Exception as e:
        # ═════════════════════════════════════════════════════════════════════
        # ERROR HANDLING
        # ═════════════════════════════════════════════════════════════════════
        console.print("\n" + "=" * 70)
        console.print(Panel.fit(
            f"[bold red]❌ CRITICAL ERROR[/bold red]\n\n"
            f"Error: {str(e)}\n\n"
            "Previous production model remains active.\n"
            "Check logs for details.",
            box=box.HEAVY,
            style="red"
        ))

        logger.error(f"Critical error during weekly adaptation: {str(e)}", exc_info=True)
        raise

    finally:
        logger.info("=" * 70)
        logger.info("WEEKLY ADAPTATION COMPLETED")
        logger.info("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
# 7. COMMAND LINE INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """
    Run weekly adaptation.

    Usage:
        python weekly_adaptation.py

    Automation:
        - Linux/Mac (cron): 0 23 * * 0 /path/to/python /path/to/weekly_adaptation.py
        - Windows (Task Scheduler): Run every Sunday at 11:00 PM
    """

    # Validate configuration first
    console.print("\n[cyan]🔧 Validating Configuration[/cyan]")
    config.validate_configuration()

    # Run adaptation
    run_weekly_adaptation()






