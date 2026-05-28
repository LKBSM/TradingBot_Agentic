# =============================================================================
# SOPHISTICATED TRAINER - Master Orchestrator
# =============================================================================
# The ultimate training system that combines all advanced components:
#
# 1. UNIFIED AGENTIC ENVIRONMENT: Consistent observation space solving domain shift
# 2. ADVANCED REWARD SHAPING: Multi-objective optimization (Sharpe, Sortino, etc.)
# 3. CURRICULUM LEARNING: 4-phase progressive difficulty
# 4. ENSEMBLE TRAINING: Model diversity for robustness
# 5. META-LEARNING: Fast regime adaptation
#
# TRAINING PIPELINE:
# ┌────────────────────────────────────────────────────────────────────────────┐
# │  Phase 1: FOUNDATION TRAINING (Curriculum Learning)                        │
# │  └─ BASE → ENRICHED → SOFT → PRODUCTION                                   │
# │     - Solves domain shift                                                  │
# │     - Progressive constraint introduction                                  │
# ├────────────────────────────────────────────────────────────────────────────┤
# │  Phase 2: SPECIALIZATION (Ensemble Training)                               │
# │  └─ Train diverse specialists with different objectives                   │
# │     - Hyperparameter diversity                                             │
# │     - Objective diversity (Sharpe vs Sortino vs Calmar)                   │
# │     - Temporal diversity (different market periods)                        │
# ├────────────────────────────────────────────────────────────────────────────┤
# │  Phase 3: ADAPTATION (Meta-Learning)                                       │
# │  └─ Find parameters that adapt quickly to regime changes                  │
# │     - MAML-inspired training                                               │
# │     - Regime-aware task construction                                       │
# │     - Online adaptation capability                                         │
# ├────────────────────────────────────────────────────────────────────────────┤
# │  Phase 4: INTEGRATION (Final Assembly)                                     │
# │  └─ Combine best components into production-ready system                  │
# │     - Ensemble selection                                                   │
# │     - Meta-adapter integration                                             │
# │     - Final validation and benchmarking                                    │
# └────────────────────────────────────────────────────────────────────────────┘
#
# This creates a trading agent that:
# - Handles domain shift between training and production
# - Optimizes multiple risk-adjusted metrics simultaneously
# - Leverages model diversity for robustness
# - Adapts quickly to changing market regimes
# =============================================================================

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import pandas as pd
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .unified_agentic_env import UnifiedAgenticEnv, TrainingMode
from .advanced_reward_shaper import AdvancedRewardShaper, RewardWeights
from .curriculum_trainer import CurriculumTrainer, CurriculumConfig, CurriculumPhase
from .ensemble_trainer import EnsembleTrainer, EnsembleConfig, EnsembleModel, EnsembleStrategy
from .meta_learner import MetaLearner, MetaLearnerConfig, OnlineAdapter, RegimeType
from .checkpoint_manager import CheckpointManager, CheckpointInfo


# =============================================================================
# Sprint 8: Entropy Annealing Callback
# =============================================================================

class EntropyAnnealingCallback(BaseCallback):
    """
    Reduce entropy coefficient across training according to a step-based schedule.

    This encourages exploration early in training and exploitation later,
    matching the curriculum phases (BASE=explore, PRODUCTION=exploit).

    Args:
        schedule: Dict mapping step thresholds to ent_coef values.
                  E.g. {0: 0.05, 100000: 0.02, 300000: 0.01, 500000: 0.005}
        verbose: Verbosity level
    """

    def __init__(self, schedule: Dict[int, float], verbose: int = 0):
        super().__init__(verbose)
        self.schedule = sorted(schedule.items())  # [(step, ent_coef), ...]
        self._last_applied: Optional[float] = None

    def _on_step(self) -> bool:
        new_ent = self.schedule[0][1]  # default: first entry
        for step_threshold, ent_coef in reversed(self.schedule):
            if self.num_timesteps >= step_threshold:
                new_ent = ent_coef
                break

        if new_ent != self._last_applied:
            self.model.ent_coef = new_ent
            self._last_applied = new_ent
            if self.verbose > 0:
                self.logger.record("train/ent_coef_annealed", new_ent)
        return True


# =============================================================================
# Sprint 8: Learning Rate Warmup Schedule
# =============================================================================

def lr_warmup_schedule(warmup_fraction: float = 0.05):
    """
    Create a learning rate schedule with linear warmup then constant.

    Args:
        warmup_fraction: Fraction of training for warmup (0.05 = first 5%)

    Returns:
        Callable for SB3's learning_rate parameter.
        SB3 passes progress_remaining (1.0 → 0.0), multiplied by base LR.
    """
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining  # 0.0 → 1.0
        if progress < warmup_fraction:
            # Linear warmup: 0 → 1 over warmup period
            return max(progress / warmup_fraction, 1e-3)  # Floor at 0.1% to avoid zero
        return 1.0  # Full LR after warmup
    return schedule


class TrainingStrategy(Enum):
    """Training strategy options."""
    CURRICULUM_ONLY = auto()      # Just curriculum learning
    ENSEMBLE_ONLY = auto()        # Just ensemble training
    META_ONLY = auto()            # Just meta-learning
    CURRICULUM_ENSEMBLE = auto()  # Curriculum + Ensemble
    CURRICULUM_META = auto()      # Curriculum + Meta
    FULL_PIPELINE = auto()        # All components (maximum sophistication)


@dataclass
class SophisticatedTrainerConfig:
    """Configuration for the sophisticated training pipeline."""
    # Strategy selection
    strategy: TrainingStrategy = TrainingStrategy.FULL_PIPELINE

    # Total compute budget
    total_timesteps: int = 1_500_000

    # Allocation by phase (as fractions, must sum to 1.0)
    curriculum_fraction: float = 0.40  # 40% to curriculum
    ensemble_fraction: float = 0.35    # 35% to ensemble
    meta_fraction: float = 0.25        # 25% to meta-learning

    # Component configs (None uses defaults)
    curriculum_config: Optional[CurriculumConfig] = None
    ensemble_config: Optional[EnsembleConfig] = None
    meta_config: Optional[MetaLearnerConfig] = None

    # Sprint 8: Corrected base hyperparameters (see config.py for rationale)
    base_hyperparams: Dict[str, Any] = field(default_factory=lambda: {
        'n_steps': 1024,       # ~2x episode length (500)
        'batch_size': 128,
        'gamma': 0.995,        # Effective horizon ~200 steps for M15
        'learning_rate': 3e-4, # Standard PPO (Schulman 2017)
        'ent_coef': 0.01,      # Moderate; annealed via EntropyAnnealingCallback
        'clip_range': 0.2,
        'gae_lambda': 0.95,
        'max_grad_norm': 0.5,
        'vf_coef': 0.5,
        'n_epochs': 5          # Reduced from 10 to prevent rollout overfitting
    })

    # Directories
    base_save_dir: str = "trained_models/sophisticated"
    tensorboard_log_dir: str = "logs/sophisticated"

    # Validation
    val_episodes: int = 20
    min_sharpe_threshold: float = 1.0
    min_win_rate_threshold: float = 0.50
    max_drawdown_threshold: float = 0.15

    # Advanced options
    use_reward_shaping: bool = True
    use_feature_reducer: bool = True
    save_checkpoints: bool = True
    checkpoint_freq: int = 100_000
    verbose: int = 1

    # Sprint 4: Checkpoint manager settings
    checkpoint_local_dir: str = "checkpoints/local"
    checkpoint_drive_dir: Optional[str] = None  # Set to Drive path on Colab
    checkpoint_keep: int = 5
    resume_from_checkpoint: bool = True

    def __post_init__(self):
        """Validate and initialize sub-configs."""
        # Validate fractions
        total_fraction = self.curriculum_fraction + self.ensemble_fraction + self.meta_fraction
        if abs(total_fraction - 1.0) > 0.01:
            logging.warning(f"Training fractions sum to {total_fraction}, normalizing to 1.0")
            self.curriculum_fraction /= total_fraction
            self.ensemble_fraction /= total_fraction
            self.meta_fraction /= total_fraction

        # Calculate timesteps per phase
        self.curriculum_timesteps = int(self.total_timesteps * self.curriculum_fraction)
        self.ensemble_timesteps = int(self.total_timesteps * self.ensemble_fraction)
        self.meta_timesteps = int(self.total_timesteps * self.meta_fraction)

        # Initialize sub-configs with calculated timesteps
        if self.curriculum_config is None:
            self.curriculum_config = CurriculumConfig(
                total_timesteps=self.curriculum_timesteps,
                model_save_dir=os.path.join(self.base_save_dir, 'curriculum'),
                tensorboard_log_dir=os.path.join(self.tensorboard_log_dir, 'curriculum')
            )

        if self.ensemble_config is None:
            timesteps_per_model = self.ensemble_timesteps // 5  # 5 models default
            self.ensemble_config = EnsembleConfig(
                n_models=5,
                timesteps_per_model=timesteps_per_model,
                save_dir=os.path.join(self.base_save_dir, 'ensemble')
            )

        if self.meta_config is None:
            self.meta_config = MetaLearnerConfig(
                save_dir=os.path.join(self.base_save_dir, 'meta')
            )


@dataclass
class TrainingResults:
    """Results from sophisticated training pipeline."""
    strategy: str
    total_timesteps: int
    training_duration_seconds: float

    # Phase results
    curriculum_results: Optional[Dict[str, Any]] = None
    ensemble_results: Optional[Dict[str, Any]] = None
    meta_results: Optional[Dict[str, Any]] = None

    # Final validation
    final_sharpe: float = 0.0
    final_win_rate: float = 0.0
    final_max_drawdown: float = 0.0
    final_cumulative_return: float = 0.0

    # Model paths
    best_model_path: str = ""
    ensemble_path: str = ""
    meta_model_path: str = ""

    # Component availability
    has_ensemble: bool = False
    has_meta_adapter: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy,
            'total_timesteps': self.total_timesteps,
            'training_duration_seconds': self.training_duration_seconds,
            'curriculum_results': self.curriculum_results,
            'ensemble_results': self.ensemble_results,
            'meta_results': self.meta_results,
            'final_metrics': {
                'sharpe_ratio': self.final_sharpe,
                'win_rate': self.final_win_rate,
                'max_drawdown': self.final_max_drawdown,
                'cumulative_return': self.final_cumulative_return
            },
            'model_paths': {
                'best_model': self.best_model_path,
                'ensemble': self.ensemble_path,
                'meta_model': self.meta_model_path
            },
            'capabilities': {
                'has_ensemble': self.has_ensemble,
                'has_meta_adapter': self.has_meta_adapter
            }
        }


class SophisticatedTrainer:
    """
    Master trainer orchestrating the full sophisticated training pipeline.

    This class combines curriculum learning, ensemble training, and meta-learning
    into a unified training system designed for maximum market performance.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: Optional[pd.DataFrame] = None,
        config: Optional[SophisticatedTrainerConfig] = None,
        economic_calendar: Optional[pd.DataFrame] = None
    ):
        """
        Initialize the Sophisticated Trainer.

        Args:
            df_train: Training data (OHLCV DataFrame)
            df_val: Validation data
            df_test: Optional test data for final evaluation
            config: Training configuration
            economic_calendar: Optional economic calendar for news simulation
        """
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test if df_test is not None else df_val
        self.config = config or SophisticatedTrainerConfig()
        self.economic_calendar = economic_calendar

        self._logger = logging.getLogger(__name__)

        # Create directories
        os.makedirs(self.config.base_save_dir, exist_ok=True)
        os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)

        # Components (initialized during training)
        self.curriculum_model: Optional[PPO] = None
        self.ensemble: Optional[EnsembleModel] = None
        self.meta_model: Optional[PPO] = None
        self.online_adapter: Optional[OnlineAdapter] = None
        self.reward_shaper: Optional[AdvancedRewardShaper] = None

        # Sprint 4: Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            local_dir=self.config.checkpoint_local_dir,
            drive_dir=self.config.checkpoint_drive_dir,
            keep=self.config.checkpoint_keep,
        )
        self._resumed_step: int = 0

        # Sprint 6/14: Feature reducer (PCA dimensionality reduction)
        self._feature_reducer = None
        if self.config.use_feature_reducer:
            try:
                from src.environment.feature_reducer import FeatureReducer
                self._feature_reducer = FeatureReducer(
                    method='pca', variance_threshold=0.95
                )
                self._logger.info("FeatureReducer initialized (PCA, 95%% variance)")
            except ImportError:
                self._logger.warning(
                    "FeatureReducer not available — training with raw observation space"
                )

        # Sprint 14: Walk-forward and quality gate results
        self._walk_forward_results: Optional[Dict[str, Any]] = None
        self._quality_gate_passed: Optional[bool] = None
        self._quality_gate_failures: List[str] = []

        # Results tracking
        self.results: Optional[TrainingResults] = None

        self._logger.info(f"SophisticatedTrainer initialized")
        self._logger.info(f"  Strategy: {self.config.strategy.name}")
        self._logger.info(f"  Total timesteps: {self.config.total_timesteps:,}")
        self._logger.info(f"  Training data: {len(df_train)} bars")
        self._logger.info(f"  Validation data: {len(df_val)} bars")

    def train(self, seed: Optional[int] = None) -> TrainingResults:
        """
        Execute the full sophisticated training pipeline.

        Args:
            seed: Random seed for reproducibility

        Returns:
            TrainingResults object containing all outcomes
        """
        start_time = datetime.now()
        self._logger.info(f"\n{'='*70}")
        self._logger.info(f"SOPHISTICATED TRAINING PIPELINE - START")
        self._logger.info(f"{'='*70}\n")

        results = TrainingResults(
            strategy=self.config.strategy.name,
            total_timesteps=self.config.total_timesteps,
            training_duration_seconds=0
        )

        # Sprint 4: Attempt resume from checkpoint
        if self.config.resume_from_checkpoint:
            resumed = self.checkpoint_manager.load_latest()
            if resumed is not None:
                _, ckpt_info = resumed
                self._resumed_step = ckpt_info.step
                self._logger.info(
                    "Resumed from checkpoint step %d (phase %d, sharpe %.2f)",
                    ckpt_info.step, ckpt_info.curriculum_phase, ckpt_info.sharpe,
                )

        try:
            # Phase 1: Curriculum Learning
            if self._should_run_curriculum():
                self._logger.info("\n" + "="*60)
                self._logger.info("PHASE 1: CURRICULUM LEARNING")
                self._logger.info("="*60 + "\n")

                curriculum_result = self._run_curriculum_phase(seed)
                results.curriculum_results = curriculum_result

                # Sprint 4: Checkpoint after curriculum phase
                if self.config.save_checkpoints and self.curriculum_model is not None:
                    self.checkpoint_manager.save(
                        self.curriculum_model,
                        step=self.config.curriculum_timesteps,
                        metrics={
                            "sharpe": curriculum_result.get("best_sharpe", 0.0),
                            "best_reward": curriculum_result.get("best_reward", 0.0),
                            "curriculum_phase": 1,
                        },
                    )

            # Phase 2: Ensemble Training
            if self._should_run_ensemble():
                self._logger.info("\n" + "="*60)
                self._logger.info("PHASE 2: ENSEMBLE TRAINING")
                self._logger.info("="*60 + "\n")

                ensemble_result = self._run_ensemble_phase(seed)
                results.ensemble_results = ensemble_result
                results.has_ensemble = True
                results.ensemble_path = self.config.ensemble_config.save_dir

            # Phase 3: Meta-Learning
            if self._should_run_meta():
                self._logger.info("\n" + "="*60)
                self._logger.info("PHASE 3: META-LEARNING")
                self._logger.info("="*60 + "\n")

                meta_result = self._run_meta_phase(seed)
                results.meta_results = meta_result
                results.has_meta_adapter = True
                results.meta_model_path = self.config.meta_config.save_dir

            # Phase 4: Final Integration and Validation
            self._logger.info("\n" + "="*60)
            self._logger.info("PHASE 4: FINAL INTEGRATION & VALIDATION")
            self._logger.info("="*60 + "\n")

            final_metrics = self._run_final_validation()
            results.final_sharpe = final_metrics['sharpe_ratio']
            results.final_win_rate = final_metrics['win_rate']
            results.final_max_drawdown = final_metrics['max_drawdown']
            results.final_cumulative_return = final_metrics['cumulative_return']
            results.best_model_path = final_metrics['best_model_path']

        except Exception as e:
            self._logger.error(f"Training failed with error: {e}")
            raise

        # Calculate duration
        end_time = datetime.now()
        results.training_duration_seconds = (end_time - start_time).total_seconds()

        # Save results
        self._save_results(results)

        # Print summary
        self._print_final_summary(results)

        self.results = results
        return results

    def _should_run_curriculum(self) -> bool:
        """Check if curriculum phase should run."""
        return self.config.strategy in [
            TrainingStrategy.CURRICULUM_ONLY,
            TrainingStrategy.CURRICULUM_ENSEMBLE,
            TrainingStrategy.CURRICULUM_META,
            TrainingStrategy.FULL_PIPELINE
        ]

    def _should_run_ensemble(self) -> bool:
        """Check if ensemble phase should run."""
        return self.config.strategy in [
            TrainingStrategy.ENSEMBLE_ONLY,
            TrainingStrategy.CURRICULUM_ENSEMBLE,
            TrainingStrategy.FULL_PIPELINE
        ]

    def _should_run_meta(self) -> bool:
        """Check if meta-learning phase should run."""
        return self.config.strategy in [
            TrainingStrategy.META_ONLY,
            TrainingStrategy.CURRICULUM_META,
            TrainingStrategy.FULL_PIPELINE
        ]

    def _run_curriculum_phase(self, seed: Optional[int]) -> Dict[str, Any]:
        """Execute curriculum learning phase."""
        trainer = CurriculumTrainer(
            df_train=self.df_train,
            df_val=self.df_val,
            config=self.config.curriculum_config,
            base_hyperparams=self.config.base_hyperparams,
            economic_calendar=self.economic_calendar,
            verbose=self.config.verbose
        )

        self.curriculum_model, summary = trainer.train(seed=seed)

        return summary

    def _run_ensemble_phase(self, seed: Optional[int]) -> Dict[str, Any]:
        """Execute ensemble training phase."""
        # If we have a curriculum model, use it as a starting point
        def env_factory(df):
            return UnifiedAgenticEnv(
                df=df,
                mode=TrainingMode.PRODUCTION,
                economic_calendar=self.economic_calendar
            )

        trainer = EnsembleTrainer(
            df_train=self.df_train,
            df_val=self.df_val,
            config=self.config.ensemble_config,
            env_factory=env_factory,
            verbose=self.config.verbose
        )

        self.ensemble, summary = trainer.train()

        return summary

    def _run_meta_phase(self, seed: Optional[int]) -> Dict[str, Any]:
        """Execute meta-learning phase."""
        def env_factory(df):
            return UnifiedAgenticEnv(
                df=df,
                mode=TrainingMode.PRODUCTION,
                economic_calendar=self.economic_calendar
            )

        learner = MetaLearner(
            df_train=self.df_train,
            config=self.config.meta_config,
            env_factory=env_factory,
            base_hyperparams=self.config.base_hyperparams,
            verbose=self.config.verbose
        )

        # Start from curriculum model if available
        base_model = self.curriculum_model

        # Determine number of meta iterations based on allocated timesteps
        # Rough approximation: 1 meta-iteration ~ 500 timesteps
        n_iterations = max(50, self.config.meta_timesteps // 500)

        self.meta_model, summary = learner.meta_train(
            n_meta_iterations=n_iterations,
            base_model=base_model
        )

        # Create online adapter for production
        self.online_adapter = learner.create_online_adapter(self.meta_model)

        return summary

    def _run_final_validation(self) -> Dict[str, Any]:
        """Run final validation and select best model."""
        self._logger.info("Running final validation on test set...")

        # Collect all available models
        models_to_evaluate = {}

        if self.curriculum_model is not None:
            models_to_evaluate['curriculum'] = self.curriculum_model

        if self.meta_model is not None:
            models_to_evaluate['meta'] = self.meta_model

        if self.ensemble is not None:
            # Evaluate individual ensemble members
            for model_id, model in self.ensemble.models.items():
                models_to_evaluate[f'ensemble_{model_id}'] = model

        # Evaluate all models
        evaluation_results = {}
        for name, model in models_to_evaluate.items():
            metrics = self._evaluate_model(model, name)
            evaluation_results[name] = metrics
            self._logger.info(
                f"  {name}: Sharpe={metrics['sharpe_ratio']:.2f}, "
                f"WinRate={metrics['win_rate']:.1%}, "
                f"MaxDD={metrics['max_drawdown']:.1%}"
            )

        # Select best model based on Sharpe ratio
        best_name = max(evaluation_results, key=lambda k: evaluation_results[k]['sharpe_ratio'])
        best_metrics = evaluation_results[best_name]

        self._logger.info(f"\nBest model: {best_name}")

        # Save best model
        best_model_path = os.path.join(self.config.base_save_dir, 'best_model.zip')
        models_to_evaluate[best_name].save(best_model_path)

        # Sprint 6: Save PCA transformer alongside model if available
        if hasattr(self, '_feature_reducer') and self._feature_reducer is not None:
            pca_path = os.path.join(self.config.base_save_dir, 'pca_transformer.pkl')
            self._feature_reducer.save(pca_path)
            self._logger.info("PCA transformer saved to %s", pca_path)

        # Also evaluate ensemble (if available) as a combined system
        if self.ensemble is not None:
            ensemble_metrics = self._evaluate_ensemble()
            self._logger.info(
                f"  ensemble_combined: Sharpe={ensemble_metrics['sharpe_ratio']:.2f}, "
                f"WinRate={ensemble_metrics['win_rate']:.1%}"
            )

            # If ensemble is better, use it
            if ensemble_metrics['sharpe_ratio'] > best_metrics['sharpe_ratio']:
                best_metrics = ensemble_metrics
                best_model_path = self.config.ensemble_config.save_dir
                self._logger.info("Ensemble outperformed individual models!")

        best_metrics['best_model_path'] = best_model_path

        return best_metrics

    def _evaluate_model(self, model: PPO, name: str) -> Dict[str, Any]:
        """Evaluate a single model on test data."""
        env = UnifiedAgenticEnv(
            self.df_test,
            mode=TrainingMode.PRODUCTION,
            economic_calendar=self.economic_calendar
        )

        obs, _ = env.reset()
        done = False
        portfolio_values = [1000.0]
        actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            portfolio_values.append(info.get('net_worth', portfolio_values[-1]))
            actions.append(int(action))
            done = done or truncated

        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        sharpe = 0
        if len(returns) > 10 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)

        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_dd = np.max(drawdown)

        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'cumulative_return': cumulative_return,
            'win_rate': np.mean(returns > 0) if len(returns) > 0 else 0,
            'total_steps': len(actions),
            'model_name': name
        }

    def _evaluate_ensemble(self) -> Dict[str, Any]:
        """Evaluate ensemble as a combined system."""
        if self.ensemble is None:
            return {'sharpe_ratio': -np.inf}

        env = UnifiedAgenticEnv(
            self.df_test,
            mode=TrainingMode.PRODUCTION,
            economic_calendar=self.economic_calendar
        )

        obs, _ = env.reset()
        done = False
        portfolio_values = [1000.0]
        actions = []

        while not done:
            action, _ = self.ensemble.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            portfolio_values.append(info.get('net_worth', portfolio_values[-1]))
            actions.append(action)
            done = done or truncated

        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        sharpe = 0
        if len(returns) > 10 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)

        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_dd = np.max(drawdown)

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'cumulative_return': (portfolio_values[-1] / portfolio_values[0]) - 1,
            'win_rate': np.mean(returns > 0) if len(returns) > 0 else 0,
            'model_name': 'ensemble_combined'
        }

    def _save_results(self, results: TrainingResults) -> None:
        """Save training results to disk."""
        results_path = os.path.join(self.config.base_save_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        self._logger.info(f"Results saved to: {results_path}")

    def _print_final_summary(self, results: TrainingResults) -> None:
        """Print final training summary."""
        hours = results.training_duration_seconds / 3600

        self._logger.info(f"\n{'='*70}")
        self._logger.info("SOPHISTICATED TRAINING COMPLETE")
        self._logger.info(f"{'='*70}")
        self._logger.info(f"\nStrategy: {results.strategy}")
        self._logger.info(f"Total timesteps: {results.total_timesteps:,}")
        self._logger.info(f"Training duration: {hours:.2f} hours")
        self._logger.info(f"\nFinal Metrics:")
        self._logger.info(f"  Sharpe Ratio:      {results.final_sharpe:.2f}")
        self._logger.info(f"  Win Rate:          {results.final_win_rate:.1%}")
        self._logger.info(f"  Max Drawdown:      {results.final_max_drawdown:.1%}")
        self._logger.info(f"  Cumulative Return: {results.final_cumulative_return:.1%}")
        self._logger.info(f"\nCapabilities:")
        self._logger.info(f"  Ensemble Available:     {results.has_ensemble}")
        self._logger.info(f"  Meta-Adapter Available: {results.has_meta_adapter}")
        self._logger.info(f"\nModel Paths:")
        self._logger.info(f"  Best Model: {results.best_model_path}")
        if results.ensemble_path:
            self._logger.info(f"  Ensemble:   {results.ensemble_path}")
        if results.meta_model_path:
            self._logger.info(f"  Meta Model: {results.meta_model_path}")
        self._logger.info(f"{'='*70}\n")

    def get_production_system(self) -> Dict[str, Any]:
        """
        Get the production-ready trading system components.

        Returns a dictionary with all components needed for live trading.
        """
        system = {
            'primary_model': None,
            'ensemble': None,
            'online_adapter': None,
            'config': self.config
        }

        # Select primary model (prefer meta-learned if available)
        if self.meta_model is not None:
            system['primary_model'] = self.meta_model
        elif self.curriculum_model is not None:
            system['primary_model'] = self.curriculum_model

        # Include ensemble if available
        if self.ensemble is not None:
            system['ensemble'] = self.ensemble

        # Include online adapter for regime adaptation
        if self.online_adapter is not None:
            system['online_adapter'] = self.online_adapter

        return system

    # =================================================================
    # Sprint 14: Walk-Forward Validation
    # =================================================================

    def run_walk_forward(
        self,
        df_full: pd.DataFrame,
        wf_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run walk-forward validation on *df_full*.

        Splits data into rolling train/test folds with purge gaps, trains
        a fresh model per fold, and collects per-fold metrics.

        Args:
            df_full: Complete DataFrame covering all periods.
            wf_config: Walk-forward parameters (defaults to config.py WALK_FORWARD_CONFIG).
            seed: Random seed.

        Returns:
            Dict with ``folds`` (list of per-fold dicts) and ``aggregate`` summary.
        """
        try:
            from config import WALK_FORWARD_CONFIG
            cfg = wf_config or WALK_FORWARD_CONFIG
        except ImportError:
            cfg = wf_config or {}

        train_w = cfg.get("train_window_bars", 6720)
        test_w = cfg.get("test_window_bars", 1120)
        purge = cfg.get("purge_gap_bars", 96)
        step = cfg.get("step_size_bars", 1120)
        max_folds = cfg.get("max_folds", 12)
        min_folds = cfg.get("min_folds", 3)

        total_bars = len(df_full)
        folds: List[Dict[str, Any]] = []
        start = 0

        while start + train_w + purge + test_w <= total_bars and len(folds) < max_folds:
            train_end = start + train_w
            test_start = train_end + purge
            test_end = test_start + test_w

            df_train_fold = df_full.iloc[start:train_end].copy()
            df_test_fold = df_full.iloc[test_start:test_end].copy()

            fold_metrics = self._evaluate_model_on_data(
                df_train_fold, df_test_fold, seed
            )
            fold_metrics["fold"] = len(folds) + 1
            fold_metrics["train_range"] = (start, train_end)
            fold_metrics["test_range"] = (test_start, test_end)
            folds.append(fold_metrics)

            self._logger.info(
                "WF Fold %d: Sharpe=%.2f  WinRate=%.1f%%  MaxDD=%.1f%%",
                fold_metrics["fold"],
                fold_metrics.get("sharpe_ratio", 0),
                fold_metrics.get("win_rate", 0) * 100,
                fold_metrics.get("max_drawdown", 0) * 100,
            )

            start += step

        if len(folds) < min_folds:
            self._logger.warning(
                "Only %d folds generated (minimum %d). Data may be too short.",
                len(folds), min_folds,
            )

        # Aggregate
        sharpes = [f["sharpe_ratio"] for f in folds if "sharpe_ratio" in f]
        win_rates = [f["win_rate"] for f in folds if "win_rate" in f]
        drawdowns = [f["max_drawdown"] for f in folds if "max_drawdown" in f]

        aggregate = {
            "n_folds": len(folds),
            "mean_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
            "std_sharpe": float(np.std(sharpes)) if sharpes else 0.0,
            "mean_win_rate": float(np.mean(win_rates)) if win_rates else 0.0,
            "mean_max_drawdown": float(np.mean(drawdowns)) if drawdowns else 0.0,
            "worst_sharpe": float(np.min(sharpes)) if sharpes else 0.0,
        }

        result = {"folds": folds, "aggregate": aggregate}
        self._walk_forward_results = result
        return result

    def _evaluate_model_on_data(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train a lightweight model on *df_train* and evaluate on *df_test*."""
        env_train = UnifiedAgenticEnv(df_train, mode=TrainingMode.PRODUCTION)
        env_test = UnifiedAgenticEnv(df_test, mode=TrainingMode.PRODUCTION)

        model = PPO(
            "MlpPolicy",
            env_train,
            seed=seed,
            verbose=0,
            **self.config.base_hyperparams,
        )
        # Use 20% of total timesteps per fold for speed
        fold_steps = min(50_000, self.config.total_timesteps // 20)
        model.learn(total_timesteps=fold_steps)

        return self._evaluate_model(model, "wf_fold")

    # =================================================================
    # Sprint 14: Quality Gates
    # =================================================================

    def check_quality_gates(
        self, metrics: Dict[str, Any], gates: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, List[str]]:
        """Check whether *metrics* pass production quality gates.

        Args:
            metrics: Dict with keys ``sharpe_ratio``, ``max_drawdown``,
                     ``win_rate``, ``profit_factor``.
            gates: Override thresholds (defaults to config.py QUALITY_GATES).

        Returns:
            (passed, failures) — *passed* is True if all gates pass,
            *failures* lists names of failed gates.
        """
        try:
            from config import QUALITY_GATES
            g = gates or QUALITY_GATES
        except ImportError:
            g = gates or {}

        checks = [
            (
                f"Sharpe >= {g.get('min_sharpe', 1.0)}",
                metrics.get("sharpe_ratio", 0) >= g.get("min_sharpe", 1.0),
            ),
            (
                f"Max DD <= {g.get('max_drawdown', 0.15):.0%}",
                metrics.get("max_drawdown", 1.0) <= g.get("max_drawdown", 0.15),
            ),
            (
                f"Win Rate >= {g.get('min_win_rate', 0.40):.0%}",
                metrics.get("win_rate", 0) >= g.get("min_win_rate", 0.40),
            ),
            (
                f"Profit Factor >= {g.get('min_profit_factor', 1.3)}",
                metrics.get("profit_factor", 0) >= g.get("min_profit_factor", 1.3),
            ),
        ]

        failures = [name for name, passed in checks if not passed]
        all_passed = len(failures) == 0

        self._quality_gate_passed = all_passed
        self._quality_gate_failures = failures

        if all_passed:
            self._logger.info("QUALITY GATES: ALL PASSED")
        else:
            self._logger.warning("QUALITY GATES FAILED: %s", ", ".join(failures))

        return all_passed, failures

    # =================================================================
    # Sprint 14: Production Artifact Packaging
    # =================================================================

    def package_production_artifact(
        self,
        output_dir: str = "production_model",
        results: Optional[TrainingResults] = None,
    ) -> str:
        """Package the trained model into a self-contained production artifact.

        Creates *output_dir* containing model weights, PCA transformer,
        config, metadata, walk-forward results, and a SHA-256 manifest.

        Returns:
            Path to the output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        results = results or self.results

        # 1. Save model weights
        model = self.curriculum_model or self.meta_model
        if model is not None:
            model.save(os.path.join(output_dir, "model"))

        # 2. Save PCA transformer
        if self._feature_reducer is not None:
            pca_path = os.path.join(output_dir, "pca_transformer.pkl")
            self._feature_reducer.save(pca_path)

        # 3. Config
        config_data = {
            "hyperparameters": self.config.base_hyperparams,
            "strategy": self.config.strategy.name,
            "total_timesteps": self.config.total_timesteps,
            "use_feature_reducer": self.config.use_feature_reducer,
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2)

        # 4. Training metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "quality_gate_passed": self._quality_gate_passed,
            "quality_gate_failures": self._quality_gate_failures,
        }
        if results is not None:
            metadata.update({
                "training_duration_seconds": results.training_duration_seconds,
                "final_sharpe": results.final_sharpe,
                "final_win_rate": results.final_win_rate,
                "final_max_drawdown": results.final_max_drawdown,
                "final_cumulative_return": results.final_cumulative_return,
                "has_ensemble": results.has_ensemble,
                "has_meta_adapter": results.has_meta_adapter,
            })
        with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # 5. Walk-forward results
        if self._walk_forward_results is not None:
            wf_path = os.path.join(output_dir, "walk_forward_results.json")
            # Convert numpy types for JSON
            wf_data = json.loads(json.dumps(
                self._walk_forward_results, default=str
            ))
            with open(wf_path, "w") as f:
                json.dump(wf_data, f, indent=2)

        # 6. SHA-256 manifest
        manifest: Dict[str, str] = {}
        for fname in os.listdir(output_dir):
            fpath = os.path.join(output_dir, fname)
            if os.path.isfile(fpath) and fname != "manifest.json":
                h = hashlib.sha256()
                with open(fpath, "rb") as fh:
                    for chunk in iter(lambda: fh.read(8192), b""):
                        h.update(chunk)
                manifest[fname] = f"sha256:{h.hexdigest()}"

        with open(os.path.join(output_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        self._logger.info("Production artifact packaged at: %s", output_dir)
        self._logger.info("  Files: %s", ", ".join(sorted(manifest.keys())))
        return output_dir

    # =================================================================
    # Sprint 14: Multi-Seed Ensemble Training
    # =================================================================

    @classmethod
    def train_ensemble_seeds(
        cls,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: Optional[pd.DataFrame] = None,
        seeds: Tuple[int, ...] = (42, 123, 456),
        config: Optional[SophisticatedTrainerConfig] = None,
        **kwargs,
    ) -> Dict[int, TrainingResults]:
        """Train multiple models with different random seeds.

        Args:
            df_train, df_val, df_test: Data splits.
            seeds: Random seeds to use.
            config: Shared training config.
            **kwargs: Extra kwargs for ``SophisticatedTrainer.__init__``.

        Returns:
            Dict mapping seed → TrainingResults.
        """
        logger = logging.getLogger(__name__)
        results: Dict[int, TrainingResults] = {}

        for i, seed in enumerate(seeds):
            logger.info(
                "=== Ensemble seed %d/%d (seed=%d) ===", i + 1, len(seeds), seed
            )
            trainer = cls(
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                config=config,
                **kwargs,
            )
            results[seed] = trainer.train(seed=seed)

        # Summary
        sharpes = [r.final_sharpe for r in results.values()]
        logger.info(
            "Ensemble complete: %d seeds, Sharpe range [%.2f, %.2f], mean=%.2f",
            len(seeds), min(sharpes), max(sharpes), np.mean(sharpes),
        )
        return results


def quick_train(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    strategy: TrainingStrategy = TrainingStrategy.CURRICULUM_ONLY,
    total_timesteps: int = 500_000,
    verbose: int = 1
) -> Tuple[PPO, TrainingResults]:
    """
    Quick training with simplified configuration.

    Good for initial experimentation and testing.

    Args:
        df_train: Training data
        df_val: Validation data
        strategy: Training strategy
        total_timesteps: Total timesteps
        verbose: Verbosity level

    Returns:
        Tuple of (best_model, results)
    """
    config = SophisticatedTrainerConfig(
        strategy=strategy,
        total_timesteps=total_timesteps,
        verbose=verbose
    )

    trainer = SophisticatedTrainer(
        df_train=df_train,
        df_val=df_val,
        config=config
    )

    results = trainer.train()

    # Load and return best model
    best_model = PPO.load(results.best_model_path)

    return best_model, results
