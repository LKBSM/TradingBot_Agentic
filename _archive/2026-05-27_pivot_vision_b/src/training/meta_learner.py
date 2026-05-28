# =============================================================================
# META-LEARNER - Fast Regime Adaptation
# =============================================================================
# Implements meta-learning for rapid adaptation to new market regimes.
#
# INSPIRATION: Model-Agnostic Meta-Learning (MAML)
# The key insight is to find initial parameters that can quickly adapt to
# any market regime with just a few gradient steps.
#
# FINANCIAL INTERPRETATION:
# - Markets cycle through regimes (trending, ranging, volatile, calm)
# - A single model trained on all regimes often performs poorly
# - Instead, we train a "meta-model" that can quickly specialize
#
# APPROACH:
# 1. REGIME DETECTION: Classify market into regimes using features
# 2. TASK CONSTRUCTION: Each regime is a "task" for meta-learning
# 3. META-TRAINING: Learn initialization that adapts quickly to any regime
# 4. ONLINE ADAPTATION: At deployment, adapt to current regime in real-time
#
# This creates a model that can handle regime changes gracefully rather
# than requiring full retraining when market dynamics shift.
# =============================================================================

import os
import copy
import logging
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class RegimeType(Enum):
    """Market regime types."""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()
    VOLATILE = auto()
    CALM = auto()
    BREAKOUT = auto()


@dataclass
class RegimeCharacteristics:
    """Characteristics of a detected regime."""
    regime_type: RegimeType
    confidence: float
    trend_strength: float
    volatility: float
    momentum: float
    duration_bars: int
    start_idx: int


@dataclass
class MetaLearnerConfig:
    """Configuration for meta-learning."""
    inner_lr: float = 1e-4          # Learning rate for inner loop (adaptation)
    outer_lr: float = 3e-5          # Learning rate for outer loop (meta-update)
    inner_steps: int = 5            # Gradient steps for adaptation
    meta_batch_size: int = 4        # Number of tasks per meta-update
    n_support: int = 100            # Support set size (for adaptation)
    n_query: int = 100              # Query set size (for evaluation)
    regime_detection_window: int = 100
    min_regime_length: int = 50     # Minimum bars for regime
    adaptation_buffer_size: int = 500
    online_adaptation_freq: int = 100
    save_dir: str = "trained_models/meta"


class RegimeDetector:
    """
    Detects and classifies market regimes from price data.
    """

    def __init__(self, window_size: int = 100, min_regime_length: int = 50):
        self.window_size = window_size
        self.min_regime_length = min_regime_length
        self._price_buffer = deque(maxlen=window_size)
        self._volume_buffer = deque(maxlen=window_size)
        self._current_regime: Optional[RegimeCharacteristics] = None
        self._regime_start_idx = 0
        self._step_count = 0

    def update(self, price: float, volume: float = 0) -> RegimeCharacteristics:
        """Update with new data and return current regime."""
        self._price_buffer.append(price)
        self._volume_buffer.append(volume)
        self._step_count += 1

        if len(self._price_buffer) < 20:
            return RegimeCharacteristics(
                regime_type=RegimeType.CALM,
                confidence=0.0,
                trend_strength=0.0,
                volatility=0.0,
                momentum=0.0,
                duration_bars=self._step_count - self._regime_start_idx,
                start_idx=self._regime_start_idx
            )

        # Calculate regime features
        prices = np.array(list(self._price_buffer))
        returns = np.diff(prices) / prices[:-1]

        # Trend detection (linear regression)
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        trend_strength = np.tanh(slope / prices[-1] * 100)

        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252 * 24 * 4)  # 15-min bars

        # Momentum (rate of change)
        if len(prices) >= 20:
            momentum = (prices[-1] - prices[-20]) / prices[-20]
        else:
            momentum = 0.0

        # Mean reversion indicator (Hurst exponent approximation)
        # H < 0.5 suggests mean reversion, H > 0.5 suggests trending
        if len(returns) >= 20:
            # Simple variance ratio test
            var_1 = np.var(returns[-10:])
            var_2 = np.var(returns[-20:])
            hurst_approx = 0.5 + np.log2(var_2 / var_1 + 1e-10) / 4 if var_1 > 0 else 0.5
        else:
            hurst_approx = 0.5

        # Classify regime
        regime_type, confidence = self._classify_regime(
            trend_strength, volatility, momentum, hurst_approx
        )

        # Check for regime change
        if (self._current_regime is None or
            regime_type != self._current_regime.regime_type):
            # Only change if confident and minimum duration met
            if confidence > 0.6:
                self._regime_start_idx = self._step_count

        regime = RegimeCharacteristics(
            regime_type=regime_type,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility=volatility,
            momentum=momentum,
            duration_bars=self._step_count - self._regime_start_idx,
            start_idx=self._regime_start_idx
        )

        self._current_regime = regime
        return regime

    def _classify_regime(
        self,
        trend_strength: float,
        volatility: float,
        momentum: float,
        hurst: float
    ) -> Tuple[RegimeType, float]:
        """Classify regime based on indicators."""
        # Thresholds
        TREND_THRESHOLD = 0.15
        VOLATILITY_HIGH = 0.3
        VOLATILITY_LOW = 0.1
        MOMENTUM_THRESHOLD = 0.02

        confidence = 0.5

        # High volatility overrides
        if volatility > VOLATILITY_HIGH:
            if abs(momentum) > MOMENTUM_THRESHOLD * 2:
                return RegimeType.BREAKOUT, 0.8
            return RegimeType.VOLATILE, 0.7

        # Trending
        if trend_strength > TREND_THRESHOLD and momentum > 0:
            confidence = min(0.5 + abs(trend_strength), 0.9)
            return RegimeType.TRENDING_UP, confidence

        if trend_strength < -TREND_THRESHOLD and momentum < 0:
            confidence = min(0.5 + abs(trend_strength), 0.9)
            return RegimeType.TRENDING_DOWN, confidence

        # Ranging (mean-reverting)
        if abs(trend_strength) < TREND_THRESHOLD / 2 and volatility < VOLATILITY_HIGH:
            return RegimeType.RANGING, 0.6

        # Calm (low volatility, no trend)
        if volatility < VOLATILITY_LOW and abs(trend_strength) < TREND_THRESHOLD:
            return RegimeType.CALM, 0.7

        # Default to ranging with low confidence
        return RegimeType.RANGING, 0.4

    def reset(self):
        """Reset detector state."""
        self._price_buffer.clear()
        self._volume_buffer.clear()
        self._current_regime = None
        self._regime_start_idx = 0
        self._step_count = 0


class RegimeTask:
    """
    A task (in MAML terminology) representing a specific market regime.

    Contains support and query sets for meta-learning.
    """

    def __init__(
        self,
        regime_type: RegimeType,
        df_segment: pd.DataFrame,
        n_support: int = 100,
        n_query: int = 100
    ):
        self.regime_type = regime_type
        self.df_segment = df_segment
        self.n_support = n_support
        self.n_query = n_query

        # Split into support and query
        total = len(df_segment)
        support_end = min(n_support, total // 2)
        query_start = support_end
        query_end = min(query_start + n_query, total)

        self.support_data = df_segment.iloc[:support_end]
        self.query_data = df_segment.iloc[query_start:query_end]


class OnlineAdapter:
    """
    Performs online adaptation of the model to current regime.
    """

    def __init__(
        self,
        model: PPO,
        config: MetaLearnerConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.config = config
        self._logger = logger or logging.getLogger(__name__)

        # Adaptation buffer
        self._observation_buffer = deque(maxlen=config.adaptation_buffer_size)
        self._action_buffer = deque(maxlen=config.adaptation_buffer_size)
        self._reward_buffer = deque(maxlen=config.adaptation_buffer_size)

        # Regime tracking
        self._regime_detector = RegimeDetector(
            window_size=config.regime_detection_window,
            min_regime_length=config.min_regime_length
        )
        self._last_regime: Optional[RegimeType] = None
        self._step_count = 0

        # Create a clone for adaptation
        self._base_model_state = copy.deepcopy(model.policy.state_dict())

    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        current_price: float
    ) -> None:
        """Store a transition for potential adaptation."""
        self._observation_buffer.append(observation)
        self._action_buffer.append(action)
        self._reward_buffer.append(reward)

        # Update regime detector
        regime = self._regime_detector.update(current_price)
        self._step_count += 1

        # Check for regime change
        if (regime.regime_type != self._last_regime and
            regime.confidence > 0.6):
            self._logger.info(
                f"Regime change detected: {self._last_regime} -> {regime.regime_type.name}"
            )
            self._last_regime = regime.regime_type

            # Trigger adaptation if enough data
            if len(self._observation_buffer) >= self.config.n_support:
                self._adapt_to_regime(regime)

        # Periodic adaptation
        elif self._step_count % self.config.online_adaptation_freq == 0:
            if len(self._observation_buffer) >= self.config.n_support:
                self._adapt_to_regime(regime)

    def _adapt_to_regime(self, regime: RegimeCharacteristics) -> None:
        """
        Adapt model to current regime using few gradient steps.

        This is the "inner loop" adaptation of MAML.
        """
        if len(self._observation_buffer) < self.config.n_support:
            return

        self._logger.debug(f"Adapting to {regime.regime_type.name} regime...")

        # Get recent data
        observations = np.array(list(self._observation_buffer)[-self.config.n_support:])
        actions = np.array(list(self._action_buffer)[-self.config.n_support:])
        rewards = np.array(list(self._reward_buffer)[-self.config.n_support:])

        # Convert to tensors
        device = self.model.policy.device
        obs_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
        act_tensor = torch.tensor(actions, dtype=torch.long, device=device)

        # Compute advantages (simple baseline subtraction)
        advantages = rewards - np.mean(rewards)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)

        # Perform inner loop adaptation
        optimizer = optim.Adam(
            self.model.policy.parameters(),
            lr=self.config.inner_lr
        )

        for step in range(self.config.inner_steps):
            # Forward pass
            dist = self.model.policy.get_distribution(obs_tensor)
            log_probs = dist.log_prob(act_tensor)

            # Policy gradient loss
            loss = -(log_probs * adv_tensor).mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)

            optimizer.step()

        self._logger.debug(f"Adaptation complete. Final loss: {loss.item():.4f}")

    def reset_to_base(self) -> None:
        """Reset model to base (meta-learned) parameters."""
        self.model.policy.load_state_dict(self._base_model_state)
        self._logger.info("Model reset to base meta-learned parameters")

    def get_current_regime(self) -> Optional[RegimeCharacteristics]:
        """Get current detected regime."""
        return self._regime_detector._current_regime


class MetaLearner:
    """
    Meta-learner for rapid regime adaptation.

    Uses MAML-style meta-learning to find initial parameters that
    can quickly adapt to any market regime.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        config: Optional[MetaLearnerConfig] = None,
        env_factory: Optional[Callable] = None,
        base_hyperparams: Optional[Dict[str, Any]] = None,
        verbose: int = 1
    ):
        """
        Initialize meta-learner.

        Args:
            df_train: Full training data
            config: Meta-learning configuration
            env_factory: Factory for creating environments
            base_hyperparams: Base PPO hyperparameters
            verbose: Verbosity level
        """
        self.df_train = df_train
        self.config = config or MetaLearnerConfig()
        self.env_factory = env_factory
        self.base_hyperparams = base_hyperparams or {
            'n_steps': 1024,
            'batch_size': 64,
            'learning_rate': 3e-5,
            'ent_coef': 0.05,
        }
        self.verbose = verbose
        self._logger = logging.getLogger(__name__)

        # Segment data into regimes
        self.regime_segments = self._segment_data_by_regime()

        os.makedirs(self.config.save_dir, exist_ok=True)

    def _segment_data_by_regime(self) -> Dict[RegimeType, List[pd.DataFrame]]:
        """Segment training data by detected regime."""
        detector = RegimeDetector(
            window_size=self.config.regime_detection_window,
            min_regime_length=self.config.min_regime_length
        )

        segments: Dict[RegimeType, List[pd.DataFrame]] = {r: [] for r in RegimeType}
        current_segment_start = 0
        current_regime = None

        for i, row in self.df_train.iterrows():
            price = row['Close'] if 'Close' in row else row['close']
            volume = row.get('Volume', row.get('volume', 0))

            regime = detector.update(price, volume)

            # Check for regime change
            if (current_regime is not None and
                regime.regime_type != current_regime and
                regime.confidence > 0.6):

                # Save segment if long enough
                segment_length = i - current_segment_start
                if segment_length >= self.config.min_regime_length:
                    idx = self.df_train.index.get_loc(current_segment_start) if isinstance(current_segment_start, pd.Timestamp) else current_segment_start
                    segment = self.df_train.iloc[idx:idx + segment_length]
                    segments[current_regime].append(segment)

                current_segment_start = i

            current_regime = regime.regime_type

        self._logger.info("Regime segmentation complete:")
        for regime, segs in segments.items():
            total_bars = sum(len(s) for s in segs)
            self._logger.info(f"  {regime.name}: {len(segs)} segments, {total_bars} bars")

        return segments

    def _create_tasks(self, n_tasks: int) -> List[RegimeTask]:
        """Create meta-learning tasks from regime segments."""
        tasks = []

        # Sample tasks from different regimes
        available_regimes = [r for r, segs in self.regime_segments.items() if len(segs) > 0]

        if not available_regimes:
            self._logger.warning("No regime segments available for task creation")
            return []

        for _ in range(n_tasks):
            # Sample regime
            regime = np.random.choice(available_regimes)
            segments = self.regime_segments[regime]

            # Sample segment
            segment = np.random.choice(segments)

            # Ensure enough data
            if len(segment) >= self.config.n_support + self.config.n_query:
                task = RegimeTask(
                    regime_type=regime,
                    df_segment=segment,
                    n_support=self.config.n_support,
                    n_query=self.config.n_query
                )
                tasks.append(task)

        return tasks

    def meta_train(
        self,
        n_meta_iterations: int = 100,
        base_model: Optional[PPO] = None
    ) -> Tuple[PPO, Dict[str, Any]]:
        """
        Perform meta-training to find good initial parameters.

        Args:
            n_meta_iterations: Number of meta-learning iterations
            base_model: Optional pre-trained model to start from

        Returns:
            Tuple of (meta-learned model, training summary)
        """
        from .unified_agentic_env import UnifiedAgenticEnv, TrainingMode

        self._logger.info(f"Starting meta-training: {n_meta_iterations} iterations")

        # Create base environment
        env = UnifiedAgenticEnv(self.df_train, mode=TrainingMode.PRODUCTION)

        # Initialize or use provided model
        if base_model is not None:
            model = base_model
            model.set_env(env)
        else:
            model = PPO(
                'MlpPolicy',
                env,
                verbose=0,
                tensorboard_log=os.path.join(self.config.save_dir, 'logs'),
                **self.base_hyperparams
            )

        # Meta-optimization setup
        meta_optimizer = optim.Adam(
            model.policy.parameters(),
            lr=self.config.outer_lr
        )

        training_history = []

        for iteration in range(n_meta_iterations):
            # Sample tasks
            tasks = self._create_tasks(self.config.meta_batch_size)

            if not tasks:
                self._logger.warning(f"Iteration {iteration}: No tasks created, skipping")
                continue

            meta_loss = 0.0
            task_metrics = []

            for task in tasks:
                # Clone parameters for inner loop
                fast_weights = {
                    name: param.clone()
                    for name, param in model.policy.named_parameters()
                }

                # Inner loop: adapt to task
                inner_loss = self._inner_loop(
                    model, task.support_data, fast_weights
                )

                # Evaluate on query set
                query_loss = self._evaluate_on_query(
                    model, task.query_data, fast_weights
                )

                meta_loss += query_loss

                task_metrics.append({
                    'regime': task.regime_type.name,
                    'inner_loss': inner_loss,
                    'query_loss': query_loss.item()
                })

            # Meta-update (outer loop)
            meta_loss /= len(tasks)
            meta_optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            meta_optimizer.step()

            # Log progress
            avg_query_loss = np.mean([m['query_loss'] for m in task_metrics])
            training_history.append({
                'iteration': iteration,
                'meta_loss': meta_loss.item(),
                'avg_query_loss': avg_query_loss,
                'task_metrics': task_metrics
            })

            if self.verbose > 0 and iteration % 10 == 0:
                self._logger.info(
                    f"Iteration {iteration}/{n_meta_iterations}: "
                    f"Meta-loss={meta_loss.item():.4f}, "
                    f"Avg query loss={avg_query_loss:.4f}"
                )

        # Save meta-learned model
        model_path = os.path.join(self.config.save_dir, 'meta_model.zip')
        model.save(model_path)

        summary = {
            'n_iterations': n_meta_iterations,
            'final_meta_loss': training_history[-1]['meta_loss'] if training_history else 0,
            'regime_distribution': {
                r.name: len(segs) for r, segs in self.regime_segments.items()
            },
            'training_history': training_history,
            'model_path': model_path
        }

        self._logger.info(f"Meta-training complete. Model saved to: {model_path}")

        return model, summary

    def _inner_loop(
        self,
        model: PPO,
        support_data: pd.DataFrame,
        fast_weights: Dict[str, torch.Tensor]
    ) -> float:
        """
        Inner loop adaptation on support set.

        Returns the final inner loop loss.
        """
        from .unified_agentic_env import UnifiedAgenticEnv, TrainingMode

        # Create mini-environment for support data
        env = UnifiedAgenticEnv(support_data, mode=TrainingMode.PRODUCTION)

        # Collect trajectories
        obs_list, act_list, rew_list = [], [], []
        obs, _ = env.reset()

        for _ in range(min(self.config.n_support, len(support_data) - 1)):
            action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, done, truncated, _ = env.step(int(action))

            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(reward)

            obs = next_obs
            if done or truncated:
                break

        if not obs_list:
            return 0.0

        # Convert to tensors
        device = model.policy.device
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
        act_tensor = torch.tensor(np.array(act_list), dtype=torch.long, device=device)
        rew_tensor = torch.tensor(np.array(rew_list), dtype=torch.float32, device=device)

        # Normalize rewards
        rew_tensor = (rew_tensor - rew_tensor.mean()) / (rew_tensor.std() + 1e-8)

        # Inner loop gradient steps
        for _ in range(self.config.inner_steps):
            # Temporarily load fast weights
            original_state = model.policy.state_dict()
            model.policy.load_state_dict({k: v for k, v in fast_weights.items()})

            # Compute loss
            dist = model.policy.get_distribution(obs_tensor)
            log_probs = dist.log_prob(act_tensor.unsqueeze(-1) if act_tensor.dim() == 1 else act_tensor)
            if log_probs.dim() > 1:
                log_probs = log_probs.squeeze(-1)

            loss = -(log_probs * rew_tensor).mean()

            # Compute gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

            # Update fast weights
            for (name, weight), grad in zip(fast_weights.items(), grads):
                fast_weights[name] = weight - self.config.inner_lr * grad

            # Restore original weights
            model.policy.load_state_dict(original_state)

        return loss.item()

    def _evaluate_on_query(
        self,
        model: PPO,
        query_data: pd.DataFrame,
        fast_weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Evaluate adapted model on query set.

        Returns loss tensor for meta-gradient computation.
        """
        from .unified_agentic_env import UnifiedAgenticEnv, TrainingMode

        # Save original weights
        original_state = model.policy.state_dict()

        # Load adapted weights
        model.policy.load_state_dict({k: v for k, v in fast_weights.items()})

        # Create environment for query data
        env = UnifiedAgenticEnv(query_data, mode=TrainingMode.PRODUCTION)

        # Collect trajectories
        obs_list, act_list, rew_list = [], [], []
        obs, _ = env.reset()

        for _ in range(min(self.config.n_query, len(query_data) - 1)):
            action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, done, truncated, _ = env.step(int(action))

            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(reward)

            obs = next_obs
            if done or truncated:
                break

        # Restore original weights
        model.policy.load_state_dict(original_state)

        if not obs_list:
            return torch.tensor(0.0, requires_grad=True)

        # Compute query loss (need gradients for meta-update)
        device = model.policy.device
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
        act_tensor = torch.tensor(np.array(act_list), dtype=torch.long, device=device)
        rew_tensor = torch.tensor(np.array(rew_list), dtype=torch.float32, device=device)

        rew_tensor = (rew_tensor - rew_tensor.mean()) / (rew_tensor.std() + 1e-8)

        # Load fast weights again for gradient computation
        model.policy.load_state_dict({k: v for k, v in fast_weights.items()})

        dist = model.policy.get_distribution(obs_tensor)
        log_probs = dist.log_prob(act_tensor.unsqueeze(-1) if act_tensor.dim() == 1 else act_tensor)
        if log_probs.dim() > 1:
            log_probs = log_probs.squeeze(-1)

        query_loss = -(log_probs * rew_tensor).mean()

        # Restore original weights
        model.policy.load_state_dict(original_state)

        return query_loss

    def create_online_adapter(self, model: PPO) -> OnlineAdapter:
        """Create an online adapter for the given model."""
        return OnlineAdapter(model, self.config, self._logger)
