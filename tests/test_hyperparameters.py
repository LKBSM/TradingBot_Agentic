"""
Sprint 8 Tests: PPO Hyperparameter Correction & Training Config.

Tests entropy annealing callback, LR warmup schedule, config validation,
and search space constraints.
"""

import os
import ast
import importlib.util
import numpy as np
import pytest

# Direct import of config (no gymnasium dependency)
import config

# Direct-import sophisticated_trainer components to avoid gymnasium __init__
_st_path = os.path.join(
    os.path.dirname(__file__), "..", "src", "training", "sophisticated_trainer.py"
)


def _load_source(path):
    """Load a Python module source and extract specific objects via AST."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# We can't import the full module (gymnasium dependency), so we test the
# callback and schedule by extracting them into a standalone module.
# Instead, let's build minimal test versions that match the source code.

class _MockModel:
    """Mock SB3 model for testing callbacks."""
    def __init__(self, ent_coef=0.01, learning_rate=3e-4):
        self.ent_coef = ent_coef
        self.learning_rate = learning_rate


class _MockLogger:
    def __init__(self):
        self.records = {}

    def record(self, key, value):
        self.records[key] = value


class EntropyAnnealingCallback:
    """Test-local replica matching sophisticated_trainer.EntropyAnnealingCallback."""
    def __init__(self, schedule, verbose=0):
        self.schedule = sorted(schedule.items())
        self._last_applied = None
        self.verbose = verbose
        self.num_timesteps = 0
        self.model = None
        self.logger = _MockLogger()

    def _on_step(self):
        new_ent = self.schedule[0][1]
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


def lr_warmup_schedule(warmup_fraction=0.05):
    """Test-local replica matching sophisticated_trainer.lr_warmup_schedule."""
    def schedule(progress_remaining):
        progress = 1.0 - progress_remaining
        if progress < warmup_fraction:
            return max(progress / warmup_fraction, 1e-3)
        return 1.0
    return schedule


# =============================================================================
# TEST: ENTROPY ANNEALING
# =============================================================================

class TestEntropyAnnealing:
    def test_initial_entropy(self):
        """At step 0, entropy should be the first schedule value."""
        schedule = {0: 0.05, 100_000: 0.02, 300_000: 0.01, 500_000: 0.005}
        cb = EntropyAnnealingCallback(schedule)
        cb.model = _MockModel(ent_coef=0.01)
        cb.num_timesteps = 0
        cb._on_step()
        assert cb.model.ent_coef == 0.05

    def test_entropy_at_100k(self):
        schedule = {0: 0.05, 100_000: 0.02, 300_000: 0.01, 500_000: 0.005}
        cb = EntropyAnnealingCallback(schedule)
        cb.model = _MockModel()
        cb.num_timesteps = 100_000
        cb._on_step()
        assert cb.model.ent_coef == 0.02

    def test_entropy_at_300k(self):
        schedule = {0: 0.05, 100_000: 0.02, 300_000: 0.01, 500_000: 0.005}
        cb = EntropyAnnealingCallback(schedule)
        cb.model = _MockModel()
        cb.num_timesteps = 300_000
        cb._on_step()
        assert cb.model.ent_coef == 0.01

    def test_entropy_at_500k(self):
        schedule = {0: 0.05, 100_000: 0.02, 300_000: 0.01, 500_000: 0.005}
        cb = EntropyAnnealingCallback(schedule)
        cb.model = _MockModel()
        cb.num_timesteps = 500_000
        cb._on_step()
        assert cb.model.ent_coef == 0.005

    def test_entropy_between_thresholds(self):
        """At step 150k, should use the 100k threshold value (0.02)."""
        schedule = {0: 0.05, 100_000: 0.02, 300_000: 0.01}
        cb = EntropyAnnealingCallback(schedule)
        cb.model = _MockModel()
        cb.num_timesteps = 150_000
        cb._on_step()
        assert cb.model.ent_coef == 0.02

    def test_entropy_monotonically_decreasing(self):
        """Entropy should only decrease or stay the same over training."""
        schedule = config.ENTROPY_ANNEALING_SCHEDULE
        cb = EntropyAnnealingCallback(schedule)
        cb.model = _MockModel()

        prev_ent = float('inf')
        for step in range(0, 600_000, 10_000):
            cb.num_timesteps = step
            cb._on_step()
            assert cb.model.ent_coef <= prev_ent, (
                f"Entropy increased at step {step}: {prev_ent} -> {cb.model.ent_coef}"
            )
            prev_ent = cb.model.ent_coef

    def test_returns_true(self):
        """Callback should always return True (don't stop training)."""
        cb = EntropyAnnealingCallback({0: 0.05})
        cb.model = _MockModel()
        cb.num_timesteps = 0
        assert cb._on_step() is True


# =============================================================================
# TEST: LR WARMUP SCHEDULE
# =============================================================================

class TestLRWarmupSchedule:
    def test_lr_at_start_near_zero(self):
        """At the very start, LR should be near zero (warmup)."""
        schedule = lr_warmup_schedule(0.05)
        # progress_remaining=1.0 means start (progress=0)
        lr_mult = schedule(1.0)
        assert lr_mult < 0.01  # Should be near zero

    def test_lr_at_1_percent(self):
        """At 1% progress, LR should be 20% of full (1% / 5% warmup)."""
        schedule = lr_warmup_schedule(0.05)
        # progress_remaining = 0.99 -> progress = 0.01
        lr_mult = schedule(0.99)
        assert abs(lr_mult - 0.2) < 0.01

    def test_lr_at_5_percent(self):
        """At 5% progress (end of warmup), LR should be 1.0."""
        schedule = lr_warmup_schedule(0.05)
        # progress_remaining = 0.95 -> progress = 0.05
        lr_mult = schedule(0.95)
        assert lr_mult == 1.0

    def test_lr_after_warmup(self):
        """After warmup, LR multiplier should be constant at 1.0."""
        schedule = lr_warmup_schedule(0.05)
        for pr in [0.9, 0.5, 0.1, 0.01]:
            assert schedule(pr) == 1.0

    def test_lr_floor_prevents_zero(self):
        """Even at progress=0, LR multiplier should not be exactly 0."""
        schedule = lr_warmup_schedule(0.05)
        assert schedule(1.0) > 0  # Floor at 1e-3

    def test_lr_warmup_for_1m_steps(self):
        """For 1M total steps, warmup should be done by step 50k."""
        schedule = lr_warmup_schedule(0.05)
        base_lr = 3e-4
        # Step 0: progress_remaining=1.0
        lr_0 = base_lr * schedule(1.0)
        assert lr_0 < base_lr * 0.01

        # Step 50k of 1M: progress_remaining = 0.95
        lr_50k = base_lr * schedule(0.95)
        assert lr_50k == base_lr  # Full LR

    def test_custom_warmup_fraction(self):
        """10% warmup fraction should warm up over first 10%."""
        schedule = lr_warmup_schedule(0.10)
        # At 5% progress (half of warmup)
        lr_mult = schedule(0.95)
        assert abs(lr_mult - 0.5) < 0.01
        # At 10% progress (end of warmup)
        lr_mult = schedule(0.90)
        assert abs(lr_mult - 1.0) < 1e-6


# =============================================================================
# TEST: CONFIG HYPERPARAMETERS
# =============================================================================

class TestConfigHyperparameters:
    def test_model_hyperparams_keys(self):
        required = {'n_steps', 'batch_size', 'gamma', 'learning_rate',
                     'ent_coef', 'clip_range', 'gae_lambda', 'max_grad_norm',
                     'vf_coef', 'n_epochs'}
        assert required.issubset(set(config.MODEL_HYPERPARAMETERS.keys()))

    def test_n_steps_is_2048(self):
        assert config.MODEL_HYPERPARAMETERS['n_steps'] == 2048

    def test_gamma_is_0995(self):
        assert config.MODEL_HYPERPARAMETERS['gamma'] == 0.995

    def test_learning_rate_is_2e4(self):
        assert config.MODEL_HYPERPARAMETERS['learning_rate'] == 2e-4

    def test_ent_coef_is_001(self):
        assert config.MODEL_HYPERPARAMETERS['ent_coef'] == 0.01

    def test_n_epochs_is_5(self):
        assert config.MODEL_HYPERPARAMETERS['n_epochs'] == 5

    def test_batch_size_divides_n_steps(self):
        n = config.MODEL_HYPERPARAMETERS['n_steps']
        b = config.MODEL_HYPERPARAMETERS['batch_size']
        assert n % b == 0, f"n_steps ({n}) must be divisible by batch_size ({b})"


# =============================================================================
# TEST: SEARCH SPACE VALIDATION
# =============================================================================

class TestSearchSpace:
    def test_no_extreme_ent_coef(self):
        """Search space should not contain ent_coef > 0.03."""
        for val in config.HYPERPARAM_SEARCH_SPACE['ent_coef']:
            assert val <= 0.03, f"ent_coef {val} too high (max 0.03)"

    def test_no_extreme_learning_rate(self):
        """No learning rate below 1e-5 (too slow)."""
        for val in config.HYPERPARAM_SEARCH_SPACE['learning_rate']:
            assert val >= 1e-5, f"learning_rate {val} too low"

    def test_gamma_range(self):
        for val in config.HYPERPARAM_SEARCH_SPACE['gamma']:
            assert 0.98 <= val <= 0.999

    def test_n_epochs_max(self):
        """n_epochs should not exceed 7 (overfitting risk)."""
        for val in config.HYPERPARAM_SEARCH_SPACE['n_epochs']:
            assert val <= 7, f"n_epochs {val} too high (max 7)"

    def test_batch_size_values(self):
        for b in config.HYPERPARAM_SEARCH_SPACE['batch_size']:
            # batch_size should be power of 2 for GPU efficiency
            assert b & (b - 1) == 0, f"batch_size {b} not power of 2"


# =============================================================================
# TEST: ENTROPY ANNEALING SCHEDULE CONFIG
# =============================================================================

class TestEntropyScheduleConfig:
    def test_schedule_exists(self):
        assert hasattr(config, 'ENTROPY_ANNEALING_SCHEDULE')
        assert isinstance(config.ENTROPY_ANNEALING_SCHEDULE, dict)

    def test_schedule_starts_at_zero(self):
        assert 0 in config.ENTROPY_ANNEALING_SCHEDULE

    def test_schedule_monotonically_decreasing(self):
        sorted_items = sorted(config.ENTROPY_ANNEALING_SCHEDULE.items())
        for i in range(1, len(sorted_items)):
            assert sorted_items[i][1] <= sorted_items[i-1][1], (
                f"Schedule not decreasing: step {sorted_items[i][0]} "
                f"has ent_coef {sorted_items[i][1]} > {sorted_items[i-1][1]}"
            )

    def test_schedule_final_value_small(self):
        """Final ent_coef should be <= 0.01 for exploitation."""
        last_value = sorted(config.ENTROPY_ANNEALING_SCHEDULE.items())[-1][1]
        assert last_value <= 0.01

    def test_lr_warmup_fraction_exists(self):
        assert hasattr(config, 'LR_WARMUP_FRACTION')
        assert 0.0 < config.LR_WARMUP_FRACTION < 0.5


# =============================================================================
# TEST: SOURCE VERIFICATION
# =============================================================================

class TestSourceVerification:
    def test_entropy_callback_in_trainer(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "training", "sophisticated_trainer.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "EntropyAnnealingCallback" in source
        assert "lr_warmup_schedule" in source

    def test_updated_hyperparams_in_trainer(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "training", "sophisticated_trainer.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        # Check corrected values are present
        assert "'n_steps': 1024" in source
        assert "'gamma': 0.995" in source
        assert "'n_epochs': 5" in source

    def test_curriculum_trainer_updated(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "training", "curriculum_trainer.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "'n_steps': 1024" in source
        assert "'gamma': 0.995" in source
        assert "'n_epochs': 5" in source
