# =============================================================================
# Sprint 13 Validation: Align Entropy Schedule with Curriculum Phases (ML-1)
# =============================================================================
# Verifies that:
# 1. ENTROPY_ANNEALING_SCHEDULE has clarifying comment
# 2. CurriculumCallback controls entropy via phase multiplier
# 3. Colab script does NOT use EntropyAnnealingCallback
# 4. PhaseConfig has entropy_coef_multiplier field
#
# Run with: python -m pytest tests/test_sprint13_entropy_alignment.py -v
# =============================================================================

import sys
import os
import importlib
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _clean_mock_modules():
    """Remove MagicMock entries from sys.modules injected by test_training_pipeline.py.

    test_training_pipeline.py stubs src.training.*, stable_baselines3.*, etc.
    We must remove ALL MagicMock modules so real imports can resolve.
    """
    for mod_name in list(sys.modules):
        if isinstance(sys.modules[mod_name], MagicMock):
            del sys.modules[mod_name]


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Config schedule has warning comment
# ─────────────────────────────────────────────────────────────────────────────
def test_config_has_warning_comment():
    """ENTROPY_ANNEALING_SCHEDULE should have Sprint 13 warning comment."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.py')
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    assert 'Sprint 13' in content, "Sprint 13 comment missing from config.py"
    assert 'Do NOT use both CurriculumCallback and EntropyAnnealingCallback' in content


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: PhaseConfig has entropy_coef_multiplier
# ─────────────────────────────────────────────────────────────────────────────
def test_phase_config_has_entropy_multiplier():
    """PhaseConfig should have entropy_coef_multiplier field."""
    _clean_mock_modules()
    from src.training.curriculum_trainer import PhaseConfig
    from src.training.unified_agentic_env import TrainingMode
    from src.training.advanced_reward_shaper import RewardWeights

    # Force reimport if we got a MagicMock
    if isinstance(PhaseConfig, MagicMock):
        importlib.reload(sys.modules['src.training.curriculum_trainer'])
        from src.training.curriculum_trainer import PhaseConfig

    phase = PhaseConfig(
        mode=TrainingMode.BASE,
        timesteps=100_000,
        reward_weights=RewardWeights(),
        eval_freq=10_000,
        entropy_coef_multiplier=0.5,
    )
    assert phase.entropy_coef_multiplier == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Default entropy multiplier is 1.0
# ─────────────────────────────────────────────────────────────────────────────
def test_default_entropy_multiplier_is_one():
    """Default entropy_coef_multiplier should be 1.0 (no change)."""
    _clean_mock_modules()
    from src.training.curriculum_trainer import PhaseConfig
    from src.training.unified_agentic_env import TrainingMode
    from src.training.advanced_reward_shaper import RewardWeights

    if isinstance(PhaseConfig, MagicMock):
        importlib.reload(sys.modules['src.training.curriculum_trainer'])
        from src.training.curriculum_trainer import PhaseConfig

    phase = PhaseConfig(
        mode=TrainingMode.BASE,
        timesteps=100_000,
        reward_weights=RewardWeights(),
        eval_freq=10_000,
    )
    assert phase.entropy_coef_multiplier == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Colab script does not import EntropyAnnealingCallback
# ─────────────────────────────────────────────────────────────────────────────
def test_colab_no_entropy_annealing_callback():
    """colab_training_full.py should not use EntropyAnnealingCallback."""
    colab_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'colab_training_full.py')
    if not os.path.exists(colab_path):
        return  # Skip if file doesn't exist

    with open(colab_path, 'r', encoding='utf-8') as f:
        content = f.read()

    assert 'EntropyAnnealingCallback' not in content, \
        "colab_training_full.py should NOT use EntropyAnnealingCallback (Sprint 13)"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: CurriculumCallback has entropy management
# ─────────────────────────────────────────────────────────────────────────────
def test_curriculum_callback_manages_entropy():
    """CurriculumCallback source should reference entropy_coef."""
    trainer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'training', 'curriculum_trainer.py')
    with open(trainer_path, 'r', encoding='utf-8') as f:
        content = f.read()

    assert 'entropy_coef' in content, "CurriculumCallback should manage entropy_coef"
