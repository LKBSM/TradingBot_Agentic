# =============================================================================
# CURRICULUM TRAINER - Progressive Difficulty Training
# =============================================================================
# Implements a 4-phase curriculum learning approach that progressively
# introduces agent constraints while maintaining a constant observation space.
#
# PHASES:
# 1. BASE (Steps 0-300K): Pure market learning
#    - Agent signals zeroed, no constraints
#    - Focus: Learn basic market patterns and trading mechanics
#    - Reward: Primarily profit-based with exploration bonus
#
# 2. ENRICHED (Steps 300K-700K): Signal awareness
#    - Agent signals included as observation features
#    - No action constraints yet
#    - Focus: Learn to interpret agent signals
#    - Reward: Add risk-adjusted metrics (Sharpe)
#
# 3. SOFT (Steps 700K-1.1M): Soft constraints
#    - Agent signals + soft penalties for rejected actions
#    - Learn consequences of ignoring agents
#    - Focus: Balance profit vs. agent guidance
#    - Reward: Multi-objective with drawdown emphasis
#
# 4. PRODUCTION (Steps 1.1M-1.5M): Full integration
#    - Hard constraints (actions may be overridden)
#    - Simulate production environment exactly
#    - Focus: Adapt to rejection dynamics
#    - Reward: Full institutional-grade metrics
#
# The key insight is that each phase builds on the previous, allowing the
# model to gradually adapt rather than face domain shift at deployment.
# =============================================================================

import os
import logging
from enum import IntEnum
from typing import Optional, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .unified_agentic_env import UnifiedAgenticEnv, TrainingMode
from .advanced_reward_shaper import AdvancedRewardShaper, RewardWeights


class CurriculumPhase(IntEnum):
    """Training curriculum phases."""
    BASE = 1
    ENRICHED = 2
    SOFT = 3
    PRODUCTION = 4


@dataclass
class PhaseConfig:
    """Configuration for a curriculum phase."""
    mode: TrainingMode
    timesteps: int
    reward_weights: RewardWeights
    eval_freq: int
    min_sharpe_to_advance: float = 0.5
    min_win_rate_to_advance: float = 0.45
    soft_penalty_scale: float = 1.0
    learning_rate_multiplier: float = 1.0
    entropy_coef_multiplier: float = 1.0
    description: str = ""


@dataclass
class CurriculumConfig:
    """Full curriculum configuration."""
    phases: List[PhaseConfig] = field(default_factory=list)
    total_timesteps: int = 1_500_000
    model_save_dir: str = "trained_models/curriculum"
    tensorboard_log_dir: str = "logs/curriculum"
    eval_episodes: int = 10
    patience: int = 3  # Phases to wait before forced advancement

    def __post_init__(self):
        if not self.phases:
            self.phases = self._default_phases()

    def _default_phases(self) -> List[PhaseConfig]:
        """Create default 4-phase curriculum."""
        total = self.total_timesteps

        return [
            # Sprint 8: Entropy multipliers produce correct annealed values
            # with base ent_coef=0.01: 5.0→0.05, 2.0→0.02, 1.0→0.01, 0.5→0.005
            # v4: Rebalanced phase budgets (35/25/25/15) + reduced Phase 1 entropy
            PhaseConfig(
                mode=TrainingMode.BASE,
                timesteps=int(total * 0.35),  # 35% (was 20%) — hardest learning phase
                reward_weights=RewardWeights.for_phase(1, 4),
                eval_freq=10_000,
                min_sharpe_to_advance=0.3,
                min_win_rate_to_advance=0.40,
                soft_penalty_scale=0.0,
                learning_rate_multiplier=1.0,
                entropy_coef_multiplier=2.0,   # 0.01 * 2.0 = 0.02 (was 5.0→0.05 which dominated loss)
                description="BASE: Pure market learning"
            ),
            PhaseConfig(
                mode=TrainingMode.ENRICHED,
                timesteps=int(total * 0.25),  # 25% (was 27%)
                reward_weights=RewardWeights.for_phase(2, 4),
                eval_freq=15_000,
                min_sharpe_to_advance=0.5,
                min_win_rate_to_advance=0.45,
                soft_penalty_scale=0.0,
                learning_rate_multiplier=0.8,
                entropy_coef_multiplier=1.5,   # 0.01 * 1.5 = 0.015
                description="ENRICHED: Signal awareness"
            ),
            PhaseConfig(
                mode=TrainingMode.SOFT,
                timesteps=int(total * 0.25),  # 25% (was 27%)
                reward_weights=RewardWeights.for_phase(3, 4),
                eval_freq=20_000,
                min_sharpe_to_advance=0.7,
                min_win_rate_to_advance=0.48,
                soft_penalty_scale=1.0,
                learning_rate_multiplier=0.6,
                entropy_coef_multiplier=1.0,   # 0.01 * 1.0 = 0.01
                description="SOFT: Soft constraints"
            ),
            PhaseConfig(
                mode=TrainingMode.PRODUCTION,
                timesteps=int(total * 0.15),  # 15% (was 26%) — only fine-tuning
                reward_weights=RewardWeights.for_phase(4, 4),
                eval_freq=25_000,
                min_sharpe_to_advance=1.0,  # Final target
                min_win_rate_to_advance=0.50,
                soft_penalty_scale=1.0,
                learning_rate_multiplier=0.4,
                entropy_coef_multiplier=0.5,   # 0.01 * 0.5 = 0.005 (exploit)
                description="PRODUCTION: Full integration"
            )
        ]


class CurriculumCallback(BaseCallback):
    """
    Callback that manages curriculum phase transitions and logging.
    """

    def __init__(
        self,
        curriculum_config: CurriculumConfig,
        env: UnifiedAgenticEnv,
        reward_shaper: AdvancedRewardShaper,
        base_hyperparams: Dict[str, Any],
        ewc_callback=None,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.config = curriculum_config
        self.env = env
        self.reward_shaper = reward_shaper
        self.base_hyperparams = base_hyperparams

        # Sprint 14: EWC callback reference for phase-transition snapshots
        self._ewc_callback = ewc_callback

        self.current_phase_idx = 0
        self.phase_start_timestep = 0
        self.phase_timesteps = 0
        self.phases_completed = []

        self._best_sharpe_in_phase = -np.inf
        self._patience_counter = 0
        self._phase_history: List[Dict] = []

        self._logger = logging.getLogger(__name__)

    @property
    def current_phase(self) -> PhaseConfig:
        return self.config.phases[self.current_phase_idx]

    def _on_training_start(self) -> None:
        """Initialize first phase."""
        self._apply_phase_config(self.current_phase)
        self._logger.info(f"Starting curriculum training: {self.current_phase.description}")

    def _on_step(self) -> bool:
        """Check for phase transitions with quality gates."""
        self.phase_timesteps = self.num_timesteps - self.phase_start_timestep

        # Check if current phase timestep budget is reached
        if self.phase_timesteps >= self.current_phase.timesteps:
            if self.current_phase_idx < len(self.config.phases) - 1:
                # Quality gate: only advance if criteria met or patience exceeded
                if self._should_advance_phase():
                    self._transition_to_next_phase()
                elif self.phase_timesteps >= self.current_phase.timesteps * 1.5:
                    # Hard cap: advance after 1.5x budget even without quality gate
                    self._logger.warning(
                        f"Phase {self.current_phase_idx + 1} hard cap reached "
                        f"({self.phase_timesteps:,} / {self.current_phase.timesteps:,} budget). "
                        f"Advancing despite not meeting quality criteria."
                    )
                    self._transition_to_next_phase()
            else:
                self._logger.info("Final phase complete. Training finished.")
                return True  # Allow training to complete naturally

        return True

    def _on_rollout_end(self) -> None:
        """Evaluate phase progress at end of each rollout."""
        # Get recent episode stats from model's buffer
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            recent_rewards = [ep['r'] for ep in list(self.model.ep_info_buffer)[-10:]]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0

            # Log phase progress
            if self.verbose > 0 and self.phase_timesteps % 50000 < self.current_phase.eval_freq:
                phase_progress = self.phase_timesteps / self.current_phase.timesteps * 100
                self._logger.info(
                    f"Phase {self.current_phase_idx + 1}/{len(self.config.phases)} "
                    f"({self.current_phase.mode.name}): {phase_progress:.1f}% | "
                    f"Avg Reward: {avg_reward:.2f}"
                )

    def _should_advance_phase(self) -> bool:
        """Determine if ready to advance to next phase."""
        # Get evaluation metrics
        summary = self.reward_shaper.get_episode_summary()

        sharpe = summary.get('final_sharpe', 0)
        win_rate = summary.get('trade_metrics', {}).get('win_rate', 0)

        # Track best performance
        if sharpe > self._best_sharpe_in_phase:
            self._best_sharpe_in_phase = sharpe
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        # Check advancement criteria
        meets_sharpe = sharpe >= self.current_phase.min_sharpe_to_advance
        meets_win_rate = win_rate >= self.current_phase.min_win_rate_to_advance
        patience_exceeded = self._patience_counter >= self.config.patience

        if meets_sharpe and meets_win_rate:
            self._logger.info(
                f"Phase {self.current_phase_idx + 1} criteria met: "
                f"Sharpe={sharpe:.2f}, WinRate={win_rate:.2%}"
            )
            return True
        elif patience_exceeded:
            self._logger.warning(
                f"Phase {self.current_phase_idx + 1} patience exceeded. "
                f"Best Sharpe: {self._best_sharpe_in_phase:.2f}. Advancing anyway."
            )
            return True

        return False

    def _transition_to_next_phase(self) -> None:
        """Transition to the next curriculum phase."""
        # Record phase completion
        self.phases_completed.append({
            'phase': self.current_phase_idx,
            'mode': self.current_phase.mode.name,
            'timesteps': self.phase_timesteps,
            'best_sharpe': self._best_sharpe_in_phase,
            'final_metrics': self.reward_shaper.get_episode_summary()
        })

        # Save checkpoint
        checkpoint_path = os.path.join(
            self.config.model_save_dir,
            f"phase_{self.current_phase_idx + 1}_checkpoint.zip"
        )
        self.model.save(checkpoint_path)
        self._logger.info(f"Phase {self.current_phase_idx + 1} checkpoint saved: {checkpoint_path}")

        # Sprint 14: Take EWC snapshot before advancing (protects learned weights)
        if self._ewc_callback is not None:
            self._ewc_callback.snapshot(self.model)
            self._logger.info(
                f"EWC snapshot taken at phase transition "
                f"(snapshot #{self._ewc_callback.n_snapshots})"
            )

        # Advance to next phase
        self.current_phase_idx += 1
        self.phase_start_timestep = self.num_timesteps
        self._best_sharpe_in_phase = -np.inf
        self._patience_counter = 0

        # Apply new phase config
        self._apply_phase_config(self.current_phase)

        self._logger.info(
            f"\n{'='*60}\n"
            f"ADVANCING TO PHASE {self.current_phase_idx + 1}: "
            f"{self.current_phase.description}\n"
            f"{'='*60}\n"
        )

    def _apply_phase_config(self, phase: PhaseConfig) -> None:
        """Apply phase-specific configuration to environment and model."""
        # Update environment mode
        self.env.set_mode(phase.mode)
        self.env.soft_penalty_scale = phase.soft_penalty_scale

        # Update reward shaper weights (for metrics tracking)
        self.reward_shaper.set_weights(phase.reward_weights)

        # Wire curriculum reward weights to the base environment's _calculate_reward()
        # This ensures penalty scaling follows the curriculum phase progression:
        #   Phase 1 (BASE): Lower penalties (focus on exploration)
        #   Phase 4 (PRODUCTION): Full penalties (focus on risk-adjusted returns)
        base_env = self.env._base_env
        rw = phase.reward_weights
        base_env.w_DD = rw.drawdown_penalty   # Drawdown penalty scaling
        base_env.w_F = rw.profit_factor       # Friction penalty (reuse profit_factor weight)
        base_env.w_T = rw.volatility_penalty  # Turnover penalty (reuse volatility_penalty weight)
        # w_L (leverage) stays at 1.0 — always enforce hard leverage limits

        # Update model hyperparameters (if supported)
        if self.model is not None:
            # Adjust learning rate
            new_lr = self.base_hyperparams.get('learning_rate', 3e-5) * phase.learning_rate_multiplier
            self.model.learning_rate = new_lr

            # Adjust entropy coefficient
            # Sprint 13: Entropy is controlled here by phase multiplier.
            # EntropyAnnealingCallback must NOT be used alongside CurriculumCallback.
            base_ent = self.base_hyperparams.get('ent_coef', 0.05)
            new_ent = base_ent * phase.entropy_coef_multiplier
            self.model.ent_coef = new_ent

            self._logger.info(
                f"Phase {phase.description}: ent_coef set to {new_ent:.4f} "
                f"(base={base_ent:.4f} x multiplier={phase.entropy_coef_multiplier})"
            )
            self._logger.info(
                f"Phase config applied: mode={phase.mode.name}, "
                f"lr={new_lr:.2e}, ent_coef={new_ent:.3f}, "
                f"w_DD={rw.drawdown_penalty:.2f}, w_F={rw.profit_factor:.2f}"
            )

    def get_training_summary(self) -> Dict[str, Any]:
        """Get complete training summary across all phases."""
        return {
            'phases_completed': self.phases_completed,
            'current_phase': self.current_phase_idx + 1,
            'total_timesteps': self.num_timesteps,
            'final_metrics': self.reward_shaper.get_episode_summary()
        }


class CurriculumTrainer:
    """
    Main curriculum trainer that orchestrates multi-phase training.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        config: Optional[CurriculumConfig] = None,
        base_hyperparams: Optional[Dict[str, Any]] = None,
        economic_calendar: Optional[pd.DataFrame] = None,
        verbose: int = 1
    ):
        """
        Initialize the Curriculum Trainer.

        Args:
            df_train: Training data
            df_val: Validation data
            config: Curriculum configuration
            base_hyperparams: Base PPO hyperparameters
            economic_calendar: Optional economic calendar for news simulation
            verbose: Verbosity level
        """
        self.df_train = df_train
        self.df_val = df_val
        self.config = config or CurriculumConfig()
        self.economic_calendar = economic_calendar
        self.verbose = verbose

        # Sprint 8: Updated default hyperparams to match corrected config
        self.base_hyperparams = base_hyperparams or {
            'n_steps': 1024,
            'batch_size': 128,
            'gamma': 0.995,
            'learning_rate': 3e-4,
            'ent_coef': 0.01,
            'clip_range': 0.2,
            'gae_lambda': 0.95,
            'max_grad_norm': 0.5,
            'vf_coef': 0.5,
            'n_epochs': 5
        }

        self._logger = logging.getLogger(__name__)
        self.model: Optional[PPO] = None
        self.env: Optional[UnifiedAgenticEnv] = None
        self.reward_shaper: Optional[AdvancedRewardShaper] = None

        # Create directories
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)

    def _create_env(self, df: pd.DataFrame, mode: TrainingMode,
                    pre_fitted_scaler=None) -> UnifiedAgenticEnv:
        """Create a unified agentic environment.

        Args:
            df: OHLCV DataFrame
            mode: Training mode
            pre_fitted_scaler: Optional pre-fitted scaler (for val/test to avoid
                              data leakage — scaler should be fit on training data only)
        """
        kwargs = {}
        if pre_fitted_scaler is not None:
            kwargs['pre_fitted_scaler'] = pre_fitted_scaler

        return UnifiedAgenticEnv(
            df=df,
            mode=mode,
            economic_calendar=self.economic_calendar,
            enable_logging=self.verbose > 1,
            **kwargs
        )

    def train(
        self,
        seed: Optional[int] = None,
        continue_from: Optional[str] = None
    ) -> Tuple[PPO, Dict[str, Any]]:
        """
        Execute full curriculum training.

        Args:
            seed: Random seed for reproducibility
            continue_from: Path to checkpoint to continue from

        Returns:
            Tuple of (trained_model, training_summary)
        """
        self._logger.info(f"Starting Curriculum Training: {self.config.total_timesteps:,} total timesteps")
        self._logger.info(f"Phases: {len(self.config.phases)}")
        for i, phase in enumerate(self.config.phases):
            self._logger.info(f"  Phase {i+1}: {phase.description} ({phase.timesteps:,} steps)")

        # Create training environment (starts in BASE mode)
        self.env = self._create_env(self.df_train, TrainingMode.BASE)

        # Create reward shaper
        self.reward_shaper = AdvancedRewardShaper(
            initial_balance=self.env._base_env.initial_balance,
            weights=self.config.phases[0].reward_weights
        )

        # Create or load model
        if continue_from and os.path.exists(continue_from):
            self._logger.info(f"Loading model from: {continue_from}")
            self.model = PPO.load(continue_from, env=self.env)
        else:
            self.model = PPO(
                'MlpPolicy',
                self.env,
                verbose=0,
                seed=seed,
                tensorboard_log=self.config.tensorboard_log_dir,
                **self.base_hyperparams
            )

        # Create curriculum callback
        curriculum_callback = CurriculumCallback(
            curriculum_config=self.config,
            env=self.env,
            reward_shaper=self.reward_shaper,
            base_hyperparams=self.base_hyperparams,
            verbose=self.verbose
        )

        # Create validation environment for evaluation
        env_val = self._create_env(self.df_val, TrainingMode.PRODUCTION)

        # Eval callback
        eval_callback = EvalCallback(
            env_val,
            best_model_save_path=os.path.join(self.config.model_save_dir, 'best'),
            log_path=os.path.join(self.config.tensorboard_log_dir, 'eval'),
            eval_freq=25_000,
            n_eval_episodes=self.config.eval_episodes,
            deterministic=True,
            render=False,
            verbose=self.verbose
        )

        # Train with callbacks
        self._logger.info("Beginning training...")
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=[curriculum_callback, eval_callback],
            reset_num_timesteps=continue_from is None
        )

        # Save final model
        final_path = os.path.join(self.config.model_save_dir, 'final_curriculum_model.zip')
        self.model.save(final_path)
        self._logger.info(f"Final model saved: {final_path}")

        # Get training summary
        summary = curriculum_callback.get_training_summary()
        summary['model_path'] = final_path

        # Final evaluation
        final_metrics = self._evaluate_final(env_val)
        summary['final_evaluation'] = final_metrics

        self._logger.info("\n" + "="*60)
        self._logger.info("CURRICULUM TRAINING COMPLETE")
        self._logger.info("="*60)
        self._logger.info(f"Total timesteps: {summary['total_timesteps']:,}")
        self._logger.info(f"Final Sharpe: {final_metrics.get('sharpe_ratio', 0):.2f}")
        self._logger.info(f"Final Win Rate: {final_metrics.get('win_rate', 0):.2%}")
        self._logger.info(f"Max Drawdown: {final_metrics.get('max_drawdown', 0):.2%}")
        self._logger.info("="*60 + "\n")

        return self.model, summary

    def _evaluate_final(self, env: UnifiedAgenticEnv) -> Dict[str, Any]:
        """Run final evaluation on validation set."""
        if self.model is None:
            return {}

        obs, info = env.reset()
        done = False
        total_reward = 0
        portfolio_values = [info.get('net_worth', 1000)]
        actions_taken = []

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            portfolio_values.append(info.get('net_worth', portfolio_values[-1]))
            actions_taken.append(int(action))
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

        # Action distribution
        unique, counts = np.unique(actions_taken, return_counts=True)
        action_dist = dict(zip(unique.tolist(), (counts / len(actions_taken)).tolist()))

        return {
            'total_reward': total_reward,
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': np.mean(returns > 0) if len(returns) > 0 else 0,
            'total_steps': len(actions_taken),
            'action_distribution': action_dist
        }


def create_default_curriculum(
    total_timesteps: int = 1_500_000,
    aggressive: bool = False
) -> CurriculumConfig:
    """
    Factory function to create standard curriculum configurations.

    Args:
        total_timesteps: Total training timesteps
        aggressive: If True, use more aggressive advancement criteria

    Returns:
        CurriculumConfig instance
    """
    config = CurriculumConfig(total_timesteps=total_timesteps)

    if aggressive:
        # More relaxed criteria for faster advancement
        for phase in config.phases:
            phase.min_sharpe_to_advance *= 0.8
            phase.min_win_rate_to_advance *= 0.95

    return config
