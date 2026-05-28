"""Sprint 14 — Training Pipeline Tests.

Covers:
- Quality gate logic
- Walk-forward validation
- Production artifact packaging
- FeatureReducer integration flag
- Multi-seed ensemble classmethod
- Config values (QUALITY_GATES, ENSEMBLE_SEEDS)

Note: The training modules depend on gymnasium/stable-baselines3 which may
not be installed in the CI environment.  We mock the heavy transitive imports
so the pure-logic methods can be tested without GPU libraries.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import sophisticated_trainer directly (bypass __init__.py import chain).
#
# The normal import path (src.training.sophisticated_trainer) triggers
# src/training/__init__.py which imports unified_agentic_env → environment
# → gymnasium/ta/stable-baselines3 etc.  Many of those are not installed
# in the CI/local test environment.
#
# We stub the *direct* dependencies of sophisticated_trainer.py itself,
# then load the module file directly with importlib.
# ---------------------------------------------------------------------------

import importlib.util

def _stub(name: str) -> MagicMock:
    """Insert a MagicMock module into sys.modules if not already present."""
    if name not in sys.modules:
        m = MagicMock()
        m.__path__ = []
        m.__name__ = name
        sys.modules[name] = m
    return sys.modules[name]


# Stub the *direct* imports of sophisticated_trainer.py
_STUB_PACKAGES = [
    # SB3
    "stable_baselines3", "stable_baselines3.common",
    "stable_baselines3.common.callbacks",
    # The relative imports inside sophisticated_trainer.py
    "src.training.unified_agentic_env",
    "src.training.advanced_reward_shaper",
    "src.training.curriculum_trainer",
    "src.training.ensemble_trainer",
    "src.training.meta_learner",
    "src.training.checkpoint_manager",
    # Also stub the package itself so Python treats it as already loaded
    "src.training",
]

for _pkg in _STUB_PACKAGES:
    _stub(_pkg)

# BaseCallback must be a real class (subclassed by EntropyAnnealingCallback)
_BCB = type("BaseCallback", (), {
    "__init__": lambda self, verbose=0: setattr(self, "verbose", verbose),
})
sys.modules["stable_baselines3.common.callbacks"].BaseCallback = _BCB  # type: ignore[attr-defined]

# Make the relative imports resolve to MagicMock objects with sensible names
sys.modules["src.training.unified_agentic_env"].UnifiedAgenticEnv = MagicMock  # type: ignore[attr-defined]
sys.modules["src.training.unified_agentic_env"].TrainingMode = MagicMock()  # type: ignore[attr-defined]
sys.modules["src.training.curriculum_trainer"].CurriculumConfig = MagicMock  # type: ignore[attr-defined]
sys.modules["src.training.curriculum_trainer"].CurriculumPhase = MagicMock()  # type: ignore[attr-defined]
sys.modules["src.training.curriculum_trainer"].CurriculumTrainer = MagicMock  # type: ignore[attr-defined]
sys.modules["src.training.ensemble_trainer"].EnsembleConfig = MagicMock  # type: ignore[attr-defined]
sys.modules["src.training.ensemble_trainer"].EnsembleTrainer = MagicMock  # type: ignore[attr-defined]
sys.modules["src.training.ensemble_trainer"].EnsembleModel = MagicMock  # type: ignore[attr-defined]
sys.modules["src.training.ensemble_trainer"].EnsembleStrategy = MagicMock()  # type: ignore[attr-defined]
sys.modules["src.training.meta_learner"].MetaLearnerConfig = MagicMock  # type: ignore[attr-defined]
sys.modules["src.training.meta_learner"].MetaLearner = MagicMock  # type: ignore[attr-defined]
sys.modules["src.training.meta_learner"].OnlineAdapter = MagicMock  # type: ignore[attr-defined]
sys.modules["src.training.meta_learner"].RegimeType = MagicMock()  # type: ignore[attr-defined]
sys.modules["src.training.checkpoint_manager"].CheckpointManager = MagicMock  # type: ignore[attr-defined]
sys.modules["src.training.checkpoint_manager"].CheckpointInfo = MagicMock  # type: ignore[attr-defined]

# PPO needs to be a class-like object
sys.modules["stable_baselines3"].PPO = MagicMock  # type: ignore[attr-defined]

# Load the module file directly
_spec = importlib.util.spec_from_file_location(
    "src.training.sophisticated_trainer",
    os.path.join(os.path.dirname(__file__), "..", "src", "training", "sophisticated_trainer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["src.training.sophisticated_trainer"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

SophisticatedTrainer = _mod.SophisticatedTrainer
SophisticatedTrainerConfig = _mod.SophisticatedTrainerConfig
TrainingResults = _mod.TrainingResults
TrainingStrategy = _mod.TrainingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with *n* bars."""
    rng = np.random.RandomState(42)
    close = 2000 + rng.randn(n).cumsum()
    return pd.DataFrame({
        "Open": close + rng.randn(n) * 0.5,
        "High": close + abs(rng.randn(n)),
        "Low": close - abs(rng.randn(n)),
        "Close": close,
        "Volume": rng.randint(100, 10000, size=n).astype(float),
    }, index=pd.date_range("2023-01-01", periods=n, freq="15min"))


def _bare_trainer():
    """Build a minimal SophisticatedTrainer without running __init__."""
    import logging

    with patch.object(SophisticatedTrainer, "__init__", lambda self, *a, **kw: None):
        trainer = SophisticatedTrainer.__new__(SophisticatedTrainer)

    trainer._logger = logging.getLogger("test")
    trainer._quality_gate_passed = None
    trainer._quality_gate_failures = []
    trainer._walk_forward_results = None
    trainer._feature_reducer = None
    trainer.curriculum_model = None
    trainer.meta_model = None
    trainer.results = None

    cfg = MagicMock()
    cfg.base_hyperparams = {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 64}
    cfg.strategy.name = "FULL_PIPELINE"
    cfg.total_timesteps = 10_000
    cfg.use_feature_reducer = False
    trainer.config = cfg

    return trainer


# ============================================================================
# Quality Gate Tests
# ============================================================================

class TestQualityGates:
    """Test SophisticatedTrainer.check_quality_gates()."""

    GATES = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.15,
        "min_win_rate": 0.40,
        "min_profit_factor": 1.3,
    }

    def test_all_gates_pass(self):
        trainer = _bare_trainer()
        metrics = {
            "sharpe_ratio": 1.5, "max_drawdown": 0.10,
            "win_rate": 0.55, "profit_factor": 1.8,
        }
        passed, failures = trainer.check_quality_gates(metrics, self.GATES)
        assert passed is True
        assert failures == []
        assert trainer._quality_gate_passed is True

    def test_sharpe_fails(self):
        trainer = _bare_trainer()
        metrics = {
            "sharpe_ratio": 0.5, "max_drawdown": 0.10,
            "win_rate": 0.55, "profit_factor": 1.8,
        }
        passed, failures = trainer.check_quality_gates(metrics, self.GATES)
        assert passed is False
        assert len(failures) == 1
        assert "Sharpe" in failures[0]

    def test_multiple_failures(self):
        trainer = _bare_trainer()
        metrics = {
            "sharpe_ratio": 0.2, "max_drawdown": 0.30,
            "win_rate": 0.20, "profit_factor": 0.8,
        }
        passed, failures = trainer.check_quality_gates(metrics, self.GATES)
        assert passed is False
        assert len(failures) == 4

    def test_edge_equal_to_threshold(self):
        """Exact boundary values should pass."""
        trainer = _bare_trainer()
        metrics = {
            "sharpe_ratio": 1.0, "max_drawdown": 0.15,
            "win_rate": 0.40, "profit_factor": 1.3,
        }
        passed, _ = trainer.check_quality_gates(metrics, self.GATES)
        assert passed is True

    def test_missing_metric_defaults_to_fail(self):
        """Missing keys default to 0 (or 1.0 for dd) -> should fail."""
        trainer = _bare_trainer()
        passed, failures = trainer.check_quality_gates({}, self.GATES)
        assert passed is False
        assert len(failures) >= 3


# ============================================================================
# Artifact Packaging Tests
# ============================================================================

class TestArtifactPackaging:
    """Test SophisticatedTrainer.package_production_artifact()."""

    def _trainer_with_results(self):
        trainer = _bare_trainer()
        trainer.results = TrainingResults(
            strategy="FULL_PIPELINE",
            total_timesteps=1_000_000,
            training_duration_seconds=3600.0,
            final_sharpe=1.5,
            final_win_rate=0.55,
            final_max_drawdown=0.10,
            final_cumulative_return=0.25,
        )
        return trainer

    def test_creates_manifest(self, tmp_path):
        trainer = self._trainer_with_results()
        out = str(tmp_path / "artifact")
        trainer.package_production_artifact(output_dir=out)

        manifest_path = os.path.join(out, "manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "config.json" in manifest
        assert "training_metadata.json" in manifest
        for val in manifest.values():
            assert val.startswith("sha256:")

    def test_metadata_contains_quality_gate(self, tmp_path):
        trainer = self._trainer_with_results()
        trainer._quality_gate_passed = False
        trainer._quality_gate_failures = ["Sharpe >= 1.0"]

        out = str(tmp_path / "artifact2")
        trainer.package_production_artifact(output_dir=out)

        with open(os.path.join(out, "training_metadata.json")) as f:
            meta = json.load(f)
        assert meta["quality_gate_passed"] is False
        assert "Sharpe >= 1.0" in meta["quality_gate_failures"]

    def test_walk_forward_included_when_present(self, tmp_path):
        trainer = self._trainer_with_results()
        trainer._walk_forward_results = {
            "folds": [{"fold": 1, "sharpe_ratio": 1.2}],
            "aggregate": {"n_folds": 1, "mean_sharpe": 1.2},
        }

        out = str(tmp_path / "artifact3")
        trainer.package_production_artifact(output_dir=out)
        assert os.path.exists(os.path.join(out, "walk_forward_results.json"))

    def test_config_json_has_hyperparams(self, tmp_path):
        trainer = self._trainer_with_results()
        out = str(tmp_path / "artifact4")
        trainer.package_production_artifact(output_dir=out)

        with open(os.path.join(out, "config.json")) as f:
            config_data = json.load(f)
        assert "hyperparameters" in config_data
        assert "strategy" in config_data


# ============================================================================
# Config Values Tests
# ============================================================================

class TestConfigValues:
    """Verify QUALITY_GATES and ENSEMBLE_SEEDS exist in config.py."""

    def test_quality_gates_exists(self):
        import config
        assert hasattr(config, "QUALITY_GATES")
        gates = config.QUALITY_GATES
        assert "min_sharpe" in gates
        assert "max_drawdown" in gates
        assert "min_win_rate" in gates
        assert "min_profit_factor" in gates

    def test_quality_gates_values_sensible(self):
        import config
        g = config.QUALITY_GATES
        assert 0 < g["min_sharpe"] <= 3.0
        assert 0 < g["max_drawdown"] <= 1.0
        assert 0 < g["min_win_rate"] <= 1.0
        assert g["min_profit_factor"] > 1.0

    def test_ensemble_seeds_exists(self):
        import config
        assert hasattr(config, "ENSEMBLE_SEEDS")
        seeds = config.ENSEMBLE_SEEDS
        assert len(seeds) >= 2
        assert all(isinstance(s, int) for s in seeds)
        assert len(set(seeds)) == len(seeds)

    def test_walk_forward_config_exists(self):
        import config
        assert hasattr(config, "WALK_FORWARD_CONFIG")
        wf = config.WALK_FORWARD_CONFIG
        assert "train_window_bars" in wf
        assert "purge_gap_bars" in wf
        assert wf["purge_gap_bars"] > 0


# ============================================================================
# FeatureReducer Integration Flag Tests
# ============================================================================

class TestFeatureReducerFlag:
    """Test that use_feature_reducer flag is present on config dataclass."""

    def test_config_default_true(self):
        cfg = SophisticatedTrainerConfig()
        assert cfg.use_feature_reducer is True

    def test_config_can_disable(self):
        cfg = SophisticatedTrainerConfig(use_feature_reducer=False)
        assert cfg.use_feature_reducer is False


# ============================================================================
# Walk-Forward Validation Unit Tests
# ============================================================================

class TestWalkForward:
    """Test walk-forward fold generation logic."""

    @patch.object(SophisticatedTrainer, "_evaluate_model_on_data")
    def test_generates_correct_fold_count(self, mock_eval):
        mock_eval.return_value = {
            "sharpe_ratio": 1.0, "win_rate": 0.5, "max_drawdown": 0.1,
        }
        trainer = _bare_trainer()

        df = _make_ohlcv(1000)
        wf_cfg = {
            "train_window_bars": 300, "test_window_bars": 100,
            "purge_gap_bars": 10, "step_size_bars": 100,
            "max_folds": 50, "min_folds": 1,
        }
        result = trainer.run_walk_forward(df, wf_config=wf_cfg)
        # start + 300 + 10 + 100 <= 1000 → start <= 590 → starts 0,100,...,500 → 6 folds
        assert result["aggregate"]["n_folds"] == 6
        assert len(result["folds"]) == 6

    @patch.object(SophisticatedTrainer, "_evaluate_model_on_data")
    def test_max_folds_limit(self, mock_eval):
        mock_eval.return_value = {
            "sharpe_ratio": 1.0, "win_rate": 0.5, "max_drawdown": 0.1,
        }
        trainer = _bare_trainer()
        df = _make_ohlcv(5000)
        wf_cfg = {
            "train_window_bars": 300, "test_window_bars": 100,
            "purge_gap_bars": 10, "step_size_bars": 100,
            "max_folds": 3, "min_folds": 1,
        }
        result = trainer.run_walk_forward(df, wf_config=wf_cfg)
        assert result["aggregate"]["n_folds"] == 3

    @patch.object(SophisticatedTrainer, "_evaluate_model_on_data")
    def test_stores_results_on_instance(self, mock_eval):
        mock_eval.return_value = {
            "sharpe_ratio": 1.0, "win_rate": 0.5, "max_drawdown": 0.1,
        }
        trainer = _bare_trainer()
        df = _make_ohlcv(500)
        wf_cfg = {
            "train_window_bars": 200, "test_window_bars": 50,
            "purge_gap_bars": 10, "step_size_bars": 50,
            "max_folds": 10, "min_folds": 1,
        }
        trainer.run_walk_forward(df, wf_config=wf_cfg)
        assert trainer._walk_forward_results is not None
        assert "aggregate" in trainer._walk_forward_results

    @patch.object(SophisticatedTrainer, "_evaluate_model_on_data")
    def test_aggregate_stats(self, mock_eval):
        """Aggregate dict has correct mean/std."""
        mock_eval.side_effect = [
            {"sharpe_ratio": 1.0, "win_rate": 0.6, "max_drawdown": 0.05},
            {"sharpe_ratio": 2.0, "win_rate": 0.4, "max_drawdown": 0.15},
        ]
        trainer = _bare_trainer()
        df = _make_ohlcv(1000)
        wf_cfg = {
            "train_window_bars": 300, "test_window_bars": 100,
            "purge_gap_bars": 10, "step_size_bars": 500,
            "max_folds": 10, "min_folds": 1,
        }
        result = trainer.run_walk_forward(df, wf_config=wf_cfg)
        agg = result["aggregate"]
        assert agg["n_folds"] == 2
        assert abs(agg["mean_sharpe"] - 1.5) < 1e-6
        assert agg["worst_sharpe"] == 1.0


# ============================================================================
# Multi-Seed Ensemble Tests
# ============================================================================

class TestMultiSeedEnsemble:
    """Test train_ensemble_seeds classmethod signature and flow."""

    def test_classmethod_exists(self):
        assert hasattr(SophisticatedTrainer, "train_ensemble_seeds")
        assert isinstance(
            SophisticatedTrainer.__dict__["train_ensemble_seeds"],
            classmethod,
        )

    def test_seeds_tuple_in_config(self):
        import config
        seeds = config.ENSEMBLE_SEEDS
        assert isinstance(seeds, tuple)
        assert 42 in seeds

    def test_training_results_to_dict(self):
        """TrainingResults.to_dict() returns all expected keys."""
        r = TrainingResults(
            strategy="TEST", total_timesteps=100,
            training_duration_seconds=10.0,
            final_sharpe=1.0, final_win_rate=0.5,
            final_max_drawdown=0.1, final_cumulative_return=0.05,
        )
        d = r.to_dict()
        assert d["strategy"] == "TEST"
        assert d["final_metrics"]["sharpe_ratio"] == 1.0
        assert "model_paths" in d
        assert "capabilities" in d
