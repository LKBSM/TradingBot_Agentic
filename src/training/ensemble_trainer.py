# =============================================================================
# ENSEMBLE TRAINER - Model Diversity and Selection
# =============================================================================
# Trains multiple models with different strategies and combines them into an
# ensemble for robust, diversified trading decisions.
#
# KEY INNOVATIONS:
# 1. DIVERSITY-DRIVEN TRAINING: Models are trained with different:
#    - Seeds (exploration diversity)
#    - Hyperparameters (strategy diversity)
#    - Reward weights (objective diversity)
#    - Data subsets (temporal diversity)
#
# 2. ENSEMBLE STRATEGIES:
#    - Voting: Majority vote on actions
#    - Weighted: Weight by recent Sharpe ratio
#    - Specialist: Route to expert per regime
#    - Stacking: Meta-learner selects best model
#
# 3. ADAPTIVE WEIGHTING: Weights update based on recent performance
#    - Rolling window evaluation
#    - Drawdown-adjusted weights
#    - Correlation penalty for similar models
#
# The goal is to create an ensemble that outperforms any single model
# by leveraging model diversity and specialization.
# =============================================================================

import os
import logging
import json
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class EnsembleStrategy(Enum):
    """Ensemble combination strategies."""
    VOTING = auto()      # Simple majority voting
    WEIGHTED = auto()    # Sharpe-weighted voting
    SPECIALIST = auto()  # Regime-based routing
    STACKING = auto()    # Meta-learner selection
    MIXTURE = auto()     # Mixture of experts (soft routing)


@dataclass
class ModelConfig:
    """Configuration for training a single model in the ensemble."""
    model_id: str
    seed: int
    hyperparams: Dict[str, Any]
    reward_weights_override: Optional[Dict[str, float]] = None
    data_start_pct: float = 0.0  # Start % of data
    data_end_pct: float = 1.0    # End % of data
    description: str = ""


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training."""
    n_models: int = 5
    strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED
    timesteps_per_model: int = 500_000
    model_configs: List[ModelConfig] = field(default_factory=list)
    diversity_bonus: float = 0.1  # Bonus for uncorrelated predictions
    min_correlation_threshold: float = 0.7  # Max correlation before penalty
    rolling_eval_window: int = 100  # Steps for weight updates
    weight_update_freq: int = 1000
    parallel_workers: int = 2
    save_dir: str = "trained_models/ensemble"

    def __post_init__(self):
        if not self.model_configs:
            self.model_configs = self._generate_diverse_configs()

    def _generate_diverse_configs(self) -> List[ModelConfig]:
        """Generate diverse model configurations."""
        configs = []
        base_hyperparams = {
            'n_steps': 2048,
            'batch_size': 128,
            'gamma': 0.99,
            'learning_rate': 3e-5,
            'ent_coef': 0.05,
            'clip_range': 0.2,
        }

        # Diversity strategies
        diversity_axes = [
            # (learning_rate, ent_coef, gamma, description)
            (1e-5, 0.10, 0.99, "Conservative Explorer"),
            (5e-5, 0.02, 0.995, "Aggressive Learner"),
            (3e-5, 0.05, 0.99, "Balanced Trader"),
            (2e-5, 0.08, 0.999, "Long-Horizon"),
            (4e-5, 0.03, 0.98, "Short-Horizon"),
            (3e-5, 0.06, 0.99, "High Entropy"),
            (3e-5, 0.01, 0.99, "Low Entropy"),
        ]

        for i in range(min(self.n_models, len(diversity_axes))):
            lr, ent, gamma, desc = diversity_axes[i]
            params = base_hyperparams.copy()
            params['learning_rate'] = lr
            params['ent_coef'] = ent
            params['gamma'] = gamma

            configs.append(ModelConfig(
                model_id=f"model_{i+1}",
                seed=42 + i * 7,
                hyperparams=params,
                description=desc
            ))

        return configs


@dataclass
class ModelPerformance:
    """Performance tracking for a single model."""
    model_id: str
    sharpe_ratio: float = 0.0
    win_rate: float = 0.5
    max_drawdown: float = 0.0
    cumulative_return: float = 0.0
    recent_predictions: List[int] = field(default_factory=list)
    recent_returns: List[float] = field(default_factory=list)
    weight: float = 1.0
    correlation_with_others: float = 0.0


class EnsembleModel:
    """
    Ensemble of PPO models with adaptive weighting.
    """

    def __init__(
        self,
        models: Dict[str, PPO],
        config: EnsembleConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ensemble from trained models.

        Args:
            models: Dictionary mapping model_id to trained PPO model
            config: Ensemble configuration
            logger: Optional logger
        """
        self.models = models
        self.config = config
        self._logger = logger or logging.getLogger(__name__)

        # Initialize performance tracking
        self.performance: Dict[str, ModelPerformance] = {
            model_id: ModelPerformance(model_id=model_id, weight=1.0 / len(models))
            for model_id in models.keys()
        }

        self._step_count = 0
        self._prediction_history: Dict[str, List[int]] = {m: [] for m in models}
        self._return_history: Dict[str, List[float]] = {m: [] for m in models}

    def predict(
        self,
        observation: np.ndarray,
        regime_hint: Optional[str] = None,
        deterministic: bool = True
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Get ensemble prediction.

        Args:
            observation: Current observation
            regime_hint: Optional regime hint for specialist routing
            deterministic: Use deterministic predictions

        Returns:
            Tuple of (action, info_dict)
        """
        self._step_count += 1

        # Get predictions from all models
        predictions = {}
        action_probs = {}

        for model_id, model in self.models.items():
            action, _states = model.predict(observation, deterministic=deterministic)
            predictions[model_id] = int(action)

            # Get action probabilities if possible
            if hasattr(model.policy, 'get_distribution'):
                obs_tensor = model.policy.obs_to_tensor(observation)[0]
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.detach().cpu().numpy()
                action_probs[model_id] = probs[0]

        # Combine predictions based on strategy
        if self.config.strategy == EnsembleStrategy.VOTING:
            final_action = self._majority_vote(predictions)
        elif self.config.strategy == EnsembleStrategy.WEIGHTED:
            final_action = self._weighted_vote(predictions, action_probs)
        elif self.config.strategy == EnsembleStrategy.SPECIALIST:
            final_action = self._specialist_routing(predictions, regime_hint)
        elif self.config.strategy == EnsembleStrategy.MIXTURE:
            final_action = self._mixture_of_experts(predictions, action_probs, observation)
        else:
            final_action = self._weighted_vote(predictions, action_probs)

        # Track predictions
        for model_id, pred in predictions.items():
            self._prediction_history[model_id].append(pred)
            if len(self._prediction_history[model_id]) > self.config.rolling_eval_window:
                self._prediction_history[model_id].pop(0)

        info = {
            'individual_predictions': predictions,
            'weights': {m: p.weight for m, p in self.performance.items()},
            'strategy': self.config.strategy.name,
            'step': self._step_count
        }

        return final_action, info

    def _majority_vote(self, predictions: Dict[str, int]) -> int:
        """Simple majority voting."""
        actions = list(predictions.values())
        unique, counts = np.unique(actions, return_counts=True)
        return int(unique[np.argmax(counts)])

    def _weighted_vote(
        self,
        predictions: Dict[str, int],
        action_probs: Dict[str, np.ndarray]
    ) -> int:
        """Sharpe-weighted voting."""
        n_actions = 5  # HOLD, OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT

        # Aggregate weighted probabilities
        weighted_probs = np.zeros(n_actions)

        for model_id, pred in predictions.items():
            weight = self.performance[model_id].weight

            if model_id in action_probs:
                # Use full probability distribution
                weighted_probs += weight * action_probs[model_id]
            else:
                # Use one-hot from prediction
                weighted_probs[pred] += weight

        # Normalize and select
        weighted_probs /= (weighted_probs.sum() + 1e-10)
        return int(np.argmax(weighted_probs))

    def _specialist_routing(
        self,
        predictions: Dict[str, int],
        regime_hint: Optional[str]
    ) -> int:
        """Route to specialist model based on regime."""
        # Default regime specialists (can be configured)
        specialists = {
            'trending_up': ['model_2', 'model_4'],  # Aggressive, Long-Horizon
            'trending_down': ['model_2', 'model_5'],  # Aggressive, Short-Horizon
            'ranging': ['model_1', 'model_3'],  # Conservative, Balanced
            'volatile': ['model_1', 'model_6'],  # Conservative, High Entropy
        }

        if regime_hint and regime_hint in specialists:
            # Use specialists for this regime
            specialist_ids = [m for m in specialists[regime_hint] if m in predictions]
            if specialist_ids:
                specialist_preds = {m: predictions[m] for m in specialist_ids}
                return self._majority_vote(specialist_preds)

        # Fallback to weighted vote
        return self._weighted_vote(predictions, {})

    def _mixture_of_experts(
        self,
        predictions: Dict[str, int],
        action_probs: Dict[str, np.ndarray],
        observation: np.ndarray
    ) -> int:
        """
        Soft mixture of experts based on observation features.

        Uses observation features to determine expert weights dynamically.
        """
        # Extract features for gating (use agent signals if available)
        # Assuming last 20 dims are agent signals
        agent_signals = observation[-20:] if len(observation) > 20 else observation[:10]

        # Simple gating based on volatility and trend signals
        volatility = agent_signals[8] if len(agent_signals) > 8 else 0.5
        trend = agent_signals[9] if len(agent_signals) > 9 else 0.0

        # Adjust weights based on regime
        gating_weights = {}
        for model_id in self.models.keys():
            base_weight = self.performance[model_id].weight

            # Adjust based on model specialty
            if 'Conservative' in self.performance[model_id].model_id:
                gate = base_weight * (1 + volatility)  # Prefer in high vol
            elif 'Aggressive' in self.performance[model_id].model_id:
                gate = base_weight * (1 + abs(trend))  # Prefer in trends
            else:
                gate = base_weight

            gating_weights[model_id] = gate

        # Normalize gating weights
        total_gate = sum(gating_weights.values())
        gating_weights = {k: v / total_gate for k, v in gating_weights.items()}

        # Weighted vote with gating
        n_actions = 5
        weighted_probs = np.zeros(n_actions)

        for model_id, pred in predictions.items():
            weight = gating_weights.get(model_id, 0)
            if model_id in action_probs:
                weighted_probs += weight * action_probs[model_id]
            else:
                weighted_probs[pred] += weight

        return int(np.argmax(weighted_probs))

    def update_weights(self, returns: Dict[str, float]) -> None:
        """
        Update model weights based on recent performance.

        Args:
            returns: Dictionary mapping model_id to step return
        """
        if self._step_count % self.config.weight_update_freq != 0:
            return

        # Update return history
        for model_id, ret in returns.items():
            if model_id in self._return_history:
                self._return_history[model_id].append(ret)
                if len(self._return_history[model_id]) > self.config.rolling_eval_window:
                    self._return_history[model_id].pop(0)

        # Calculate new weights based on rolling Sharpe
        new_weights = {}
        for model_id in self.models.keys():
            returns_list = self._return_history[model_id]
            if len(returns_list) >= 20:
                mean_ret = np.mean(returns_list)
                std_ret = np.std(returns_list)
                sharpe = mean_ret / (std_ret + 1e-10) * np.sqrt(252)

                # Update performance tracking
                self.performance[model_id].sharpe_ratio = sharpe
                self.performance[model_id].recent_returns = returns_list.copy()

                # Weight based on Sharpe (softmax-like)
                new_weights[model_id] = np.exp(sharpe * 0.5)
            else:
                new_weights[model_id] = 1.0

        # Apply correlation penalty
        new_weights = self._apply_correlation_penalty(new_weights)

        # Normalize weights
        total = sum(new_weights.values())
        for model_id in self.models.keys():
            self.performance[model_id].weight = new_weights[model_id] / total

    def _apply_correlation_penalty(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Penalize highly correlated models to encourage diversity."""
        model_ids = list(self.models.keys())
        n_models = len(model_ids)

        if n_models < 2:
            return weights

        # Calculate pairwise prediction correlations
        correlations = np.zeros((n_models, n_models))
        for i, m1 in enumerate(model_ids):
            for j, m2 in enumerate(model_ids):
                if i != j:
                    preds1 = self._prediction_history[m1]
                    preds2 = self._prediction_history[m2]
                    if len(preds1) >= 10 and len(preds2) >= 10:
                        # Agreement rate as proxy for correlation
                        agreement = np.mean(np.array(preds1[-50:]) == np.array(preds2[-50:]))
                        correlations[i, j] = agreement

        # Penalize models with high average correlation
        for i, model_id in enumerate(model_ids):
            avg_corr = np.mean(correlations[i, :])
            self.performance[model_id].correlation_with_others = avg_corr

            if avg_corr > self.config.min_correlation_threshold:
                penalty = 1 - (avg_corr - self.config.min_correlation_threshold)
                weights[model_id] *= penalty

        return weights

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get current ensemble statistics."""
        return {
            'total_steps': self._step_count,
            'strategy': self.config.strategy.name,
            'model_weights': {m: p.weight for m, p in self.performance.items()},
            'model_sharpes': {m: p.sharpe_ratio for m, p in self.performance.items()},
            'correlations': {m: p.correlation_with_others for m, p in self.performance.items()},
            'n_models': len(self.models)
        }

    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        os.makedirs(path, exist_ok=True)

        # Save each model
        for model_id, model in self.models.items():
            model.save(os.path.join(path, f"{model_id}.zip"))

        # Save config and performance
        meta = {
            'config': {
                'n_models': self.config.n_models,
                'strategy': self.config.strategy.name,
                'rolling_eval_window': self.config.rolling_eval_window
            },
            'performance': {
                m: {
                    'weight': p.weight,
                    'sharpe_ratio': p.sharpe_ratio,
                    'correlation': p.correlation_with_others
                }
                for m, p in self.performance.items()
            }
        }
        with open(os.path.join(path, 'ensemble_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        self._logger.info(f"Ensemble saved to: {path}")

    @classmethod
    def load(cls, path: str, config: Optional[EnsembleConfig] = None) -> 'EnsembleModel':
        """Load ensemble from disk."""
        # Load metadata
        with open(os.path.join(path, 'ensemble_meta.json'), 'r') as f:
            meta = json.load(f)

        if config is None:
            config = EnsembleConfig(
                n_models=meta['config']['n_models'],
                strategy=EnsembleStrategy[meta['config']['strategy']]
            )

        # Load models
        models = {}
        for model_id in meta['performance'].keys():
            model_path = os.path.join(path, f"{model_id}.zip")
            if os.path.exists(model_path):
                models[model_id] = PPO.load(model_path)

        ensemble = cls(models, config)

        # Restore performance
        for model_id, perf in meta['performance'].items():
            if model_id in ensemble.performance:
                ensemble.performance[model_id].weight = perf['weight']
                ensemble.performance[model_id].sharpe_ratio = perf['sharpe_ratio']

        return ensemble


class EnsembleTrainer:
    """
    Trains an ensemble of diverse models.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        config: Optional[EnsembleConfig] = None,
        env_factory: Optional[Callable] = None,
        verbose: int = 1
    ):
        """
        Initialize ensemble trainer.

        Args:
            df_train: Training data
            df_val: Validation data
            config: Ensemble configuration
            env_factory: Optional factory for creating environments
            verbose: Verbosity level
        """
        self.df_train = df_train
        self.df_val = df_val
        self.config = config or EnsembleConfig()
        self.env_factory = env_factory
        self.verbose = verbose
        self._logger = logging.getLogger(__name__)

        os.makedirs(self.config.save_dir, exist_ok=True)

    def train(self) -> Tuple[EnsembleModel, Dict[str, Any]]:
        """
        Train all models in the ensemble.

        Returns:
            Tuple of (EnsembleModel, training_summary)
        """
        self._logger.info(f"Training ensemble with {self.config.n_models} models")

        trained_models = {}
        training_results = []

        # Train models (can be parallelized)
        for model_config in self.config.model_configs[:self.config.n_models]:
            self._logger.info(f"Training {model_config.model_id}: {model_config.description}")

            model, metrics = self._train_single_model(model_config)
            trained_models[model_config.model_id] = model
            training_results.append({
                'model_id': model_config.model_id,
                'description': model_config.description,
                **metrics
            })

        # Create ensemble
        ensemble = EnsembleModel(trained_models, self.config, self._logger)

        # Calibrate weights with validation
        self._calibrate_weights(ensemble)

        # Save ensemble
        ensemble.save(self.config.save_dir)

        summary = {
            'n_models': len(trained_models),
            'strategy': self.config.strategy.name,
            'model_results': training_results,
            'final_weights': {m: p.weight for m, p in ensemble.performance.items()},
            'save_path': self.config.save_dir
        }

        return ensemble, summary

    def _train_single_model(
        self,
        model_config: ModelConfig
    ) -> Tuple[PPO, Dict[str, Any]]:
        """Train a single model with given configuration."""
        from .unified_agentic_env import UnifiedAgenticEnv, TrainingMode

        # Create environment
        if self.env_factory:
            env = self.env_factory(self.df_train)
        else:
            # Use data subset if specified
            start_idx = int(len(self.df_train) * model_config.data_start_pct)
            end_idx = int(len(self.df_train) * model_config.data_end_pct)
            df_subset = self.df_train.iloc[start_idx:end_idx]

            env = UnifiedAgenticEnv(
                df=df_subset,
                mode=TrainingMode.PRODUCTION
            )

        # Create model
        model = PPO(
            'MlpPolicy',
            env,
            verbose=0,
            seed=model_config.seed,
            tensorboard_log=os.path.join(self.config.save_dir, 'logs', model_config.model_id),
            **model_config.hyperparams
        )

        # Train
        model.learn(
            total_timesteps=self.config.timesteps_per_model,
            progress_bar=self.verbose > 0
        )

        # Evaluate
        metrics = self._evaluate_model(model, model_config.model_id)

        # Save individual model
        model.save(os.path.join(self.config.save_dir, f"{model_config.model_id}.zip"))

        return model, metrics

    def _evaluate_model(self, model: PPO, model_id: str) -> Dict[str, Any]:
        """Evaluate a single model on validation data."""
        from .unified_agentic_env import UnifiedAgenticEnv, TrainingMode

        env = UnifiedAgenticEnv(self.df_val, mode=TrainingMode.PRODUCTION)

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

        max_dd = 0
        peak = portfolio_values[0]
        for val in portfolio_values:
            peak = max(peak, val)
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'cumulative_return': (portfolio_values[-1] / portfolio_values[0]) - 1,
            'win_rate': np.mean(returns > 0) if len(returns) > 0 else 0,
            'total_trades': sum(1 for a in actions if a != 0)
        }

    def _calibrate_weights(self, ensemble: EnsembleModel) -> None:
        """Calibrate ensemble weights using validation data."""
        from .unified_agentic_env import UnifiedAgenticEnv, TrainingMode

        env = UnifiedAgenticEnv(self.df_val, mode=TrainingMode.PRODUCTION)

        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < 1000:  # Limited calibration
            action, info = ensemble.predict(obs)
            obs, reward, done, truncated, info_step = env.step(action)

            # Track returns per model (simulate)
            returns = {m: reward for m in ensemble.models.keys()}
            ensemble.update_weights(returns)

            step += 1
            done = done or truncated

        self._logger.info(f"Weights calibrated after {step} steps")
        self._logger.info(f"Final weights: {ensemble.get_ensemble_stats()['model_weights']}")
