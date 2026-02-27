"""
Sprint 4 Tests: CheckpointManager — SHA-256 verification, dual-write,
resume-from-checkpoint, rotation, and alert on failure.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Direct import to avoid gymnasium dependency from training/__init__.py
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "checkpoint_manager",
    os.path.join(os.path.dirname(__file__), "..", "src", "training", "checkpoint_manager.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["checkpoint_manager"] = _mod
_spec.loader.exec_module(_mod)
CheckpointManager = _mod.CheckpointManager
CheckpointInfo = _mod.CheckpointInfo


# =============================================================================
# HELPERS
# =============================================================================

class FakeModel:
    """Minimal model stub that mimics SB3 model.save() / model.load()."""

    def __init__(self, data: bytes = b"FAKE_MODEL_DATA" * 20):
        self.data = data

    def save(self, path: str):
        # SB3 PPO.save() writes a .zip — we just write raw bytes
        if not path.endswith(".zip"):
            path = path  # keep as-is, CheckpointManager handles both
        with open(path, "wb") as f:
            f.write(self.data)

    @classmethod
    def load(cls, path: str):
        return cls()


def make_manager(tmp_path, drive=True, keep=5, alert_cb=None):
    """Create a CheckpointManager with temp directories."""
    local_dir = str(tmp_path / "local")
    drive_dir = str(tmp_path / "drive") if drive else None
    return CheckpointManager(
        local_dir=local_dir,
        drive_dir=drive_dir,
        keep=keep,
        alert_callback=alert_cb,
    )


# =============================================================================
# TEST: SAVE & MANIFEST
# =============================================================================

class TestCheckpointSave:
    def test_save_creates_manifest(self, tmp_path):
        mgr = make_manager(tmp_path)
        model = FakeModel()
        info = mgr.save(model, step=1000, metrics={"sharpe": 1.5, "best_reward": 2.3})

        assert info.step == 1000
        assert info.sharpe == 1.5
        assert info.best_reward == 2.3
        assert info.verified is True

        # Manifest file exists
        manifest_path = Path(info.local_path) / "checkpoint_step_1000.manifest.json"
        assert manifest_path.exists()

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["step"] == 1000
        assert "files" in data
        assert len(data["files"]) >= 1

    def test_save_files_have_sha256(self, tmp_path):
        mgr = make_manager(tmp_path)
        info = mgr.save(FakeModel(), step=2000)

        for filename, hash_str in info.files.items():
            assert hash_str.startswith("sha256:")
            assert len(hash_str) > len("sha256:") + 10  # hex digest present

    def test_save_extra_files(self, tmp_path):
        mgr = make_manager(tmp_path)

        # Create an extra artifact
        extra_file = tmp_path / "scaler.pkl"
        extra_file.write_bytes(b"SCALER_DATA_BYTES" * 20)

        info = mgr.save(FakeModel(), step=3000, extra_files={"scaler.pkl": str(extra_file)})
        assert "scaler.pkl" in info.files

    def test_save_with_drive(self, tmp_path):
        mgr = make_manager(tmp_path, drive=True)
        info = mgr.save(FakeModel(), step=4000)

        # Drive copy should exist
        drive_path = tmp_path / "drive" / "checkpoint_step_4000"
        assert drive_path.exists()
        assert info.drive_path == str(drive_path)

    def test_save_without_drive(self, tmp_path):
        mgr = make_manager(tmp_path, drive=False)
        info = mgr.save(FakeModel(), step=5000)
        assert info.drive_path == ""


# =============================================================================
# TEST: VERIFY
# =============================================================================

class TestCheckpointVerify:
    def test_verify_valid_checkpoint(self, tmp_path):
        mgr = make_manager(tmp_path)
        info = mgr.save(FakeModel(), step=1000)
        assert mgr.verify(info.local_path) is True

    def test_verify_corrupted_file(self, tmp_path):
        mgr = make_manager(tmp_path)
        info = mgr.save(FakeModel(), step=2000)

        # Corrupt the model file by truncating 1 byte
        model_files = list(Path(info.local_path).glob("model*"))
        assert len(model_files) >= 1
        model_file = model_files[0]
        data = model_file.read_bytes()
        model_file.write_bytes(data[:-1])  # Truncate

        assert mgr.verify(info.local_path) is False

    def test_verify_missing_file(self, tmp_path):
        mgr = make_manager(tmp_path)
        info = mgr.save(FakeModel(), step=3000)

        # Delete the model file
        model_files = list(Path(info.local_path).glob("model*"))
        for f in model_files:
            f.unlink()

        assert mgr.verify(info.local_path) is False

    def test_verify_no_manifest(self, tmp_path):
        # Create a directory without a manifest
        fake_dir = tmp_path / "local" / "checkpoint_step_999"
        fake_dir.mkdir(parents=True)
        (fake_dir / "model.zip").write_bytes(b"data" * 50)

        mgr = make_manager(tmp_path)
        assert mgr.verify(str(fake_dir)) is False

    def test_verify_empty_file(self, tmp_path):
        mgr = make_manager(tmp_path)
        info = mgr.save(FakeModel(), step=4000)

        # Replace model with tiny file (below _MIN_FILE_SIZE)
        model_files = list(Path(info.local_path).glob("model*"))
        for f in model_files:
            f.write_bytes(b"x")  # 1 byte — below 100 byte minimum

        assert mgr.verify(info.local_path) is False


# =============================================================================
# TEST: RESUME
# =============================================================================

class TestCheckpointResume:
    def test_load_latest_returns_newest(self, tmp_path):
        mgr = make_manager(tmp_path)
        mgr.save(FakeModel(), step=100)
        mgr.save(FakeModel(), step=200)
        mgr.save(FakeModel(), step=300)

        result = mgr.load_latest()
        assert result is not None
        _, info = result
        assert info.step == 300

    def test_load_latest_skips_corrupted(self, tmp_path):
        mgr = make_manager(tmp_path, keep=10)
        mgr.save(FakeModel(), step=100)
        info_200 = mgr.save(FakeModel(), step=200)
        mgr.save(FakeModel(), step=300)

        # Corrupt step 300
        ckpt_300_dir = Path(mgr.local_dir) / "checkpoint_step_300"
        for f in ckpt_300_dir.glob("model*"):
            f.write_bytes(b"x")

        result = mgr.load_latest()
        assert result is not None
        _, info = result
        assert info.step == 200  # Fell back to step 200

    def test_load_latest_restores_from_drive(self, tmp_path):
        mgr = make_manager(tmp_path, keep=10)
        mgr.save(FakeModel(), step=100)

        # Delete local, keep Drive
        local_dir = Path(mgr.local_dir) / "checkpoint_step_100"
        shutil.rmtree(local_dir)
        assert not local_dir.exists()

        result = mgr.load_latest()
        assert result is not None
        _, info = result
        assert info.step == 100
        # Local should be restored from Drive
        assert local_dir.exists()

    def test_load_latest_returns_none_when_empty(self, tmp_path):
        mgr = make_manager(tmp_path)
        assert mgr.load_latest() is None

    def test_load_with_model_cls(self, tmp_path):
        mgr = make_manager(tmp_path)
        mgr.save(FakeModel(), step=500)

        result = mgr.load_latest(model_cls=FakeModel)
        assert result is not None
        model, info = result
        assert isinstance(model, FakeModel)
        assert info.step == 500

    def test_resume_continues_from_correct_step(self, tmp_path):
        mgr = make_manager(tmp_path)
        mgr.save(FakeModel(), step=150000, metrics={"curriculum_phase": 2, "learning_rate": 1e-5})

        result = mgr.load_latest()
        _, info = result
        assert info.step == 150000
        assert info.curriculum_phase == 2
        assert info.learning_rate == 1e-5


# =============================================================================
# TEST: ROTATION
# =============================================================================

class TestCheckpointRotation:
    def test_rotation_keeps_n_most_recent(self, tmp_path):
        mgr = make_manager(tmp_path, keep=3)

        for step in [100, 200, 300, 400, 500]:
            mgr.save(FakeModel(), step=step)

        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 3
        steps = [c.step for c in checkpoints]
        assert steps == [500, 400, 300]

    def test_rotation_deletes_drive_copies(self, tmp_path):
        mgr = make_manager(tmp_path, keep=2)

        mgr.save(FakeModel(), step=100)
        mgr.save(FakeModel(), step=200)
        mgr.save(FakeModel(), step=300)

        # Step 100 should be gone from both local and Drive
        assert not (tmp_path / "local" / "checkpoint_step_100").exists()
        assert not (tmp_path / "drive" / "checkpoint_step_100").exists()

    def test_cleanup_explicit_call(self, tmp_path):
        mgr = make_manager(tmp_path, keep=10)  # High keep so auto-rotate doesn't kick in

        for step in [100, 200, 300, 400, 500]:
            mgr.save(FakeModel(), step=step)

        deleted = mgr.cleanup(keep=2)
        assert deleted > 0
        assert len(mgr.list_checkpoints()) == 2


# =============================================================================
# TEST: LIST
# =============================================================================

class TestCheckpointList:
    def test_list_sorted_newest_first(self, tmp_path):
        mgr = make_manager(tmp_path, keep=10)

        mgr.save(FakeModel(), step=300)
        mgr.save(FakeModel(), step=100)
        mgr.save(FakeModel(), step=200)

        checkpoints = mgr.list_checkpoints()
        steps = [c.step for c in checkpoints]
        assert steps == [300, 200, 100]

    def test_list_deduplicates_local_and_drive(self, tmp_path):
        mgr = make_manager(tmp_path, keep=10)
        mgr.save(FakeModel(), step=100)

        # Same checkpoint exists in both local and drive
        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 1


# =============================================================================
# TEST: ALERT CALLBACK
# =============================================================================

class TestAlertCallback:
    def test_alert_on_drive_failure(self, tmp_path):
        alerts = []
        mgr = CheckpointManager(
            local_dir=str(tmp_path / "local"),
            drive_dir=str(tmp_path / "drive"),
            keep=5,
            alert_callback=lambda msg: alerts.append(msg),
        )

        # Save normally first
        mgr.save(FakeModel(), step=100)

        # Make Drive read-only to trigger failure
        drive_dir = tmp_path / "drive"
        # Simulate by patching shutil.copytree to fail
        with patch("checkpoint_manager.shutil.copytree", side_effect=OSError("Drive full")):
            mgr.save(FakeModel(), step=200)

        assert len(alerts) == 1
        assert "Drive copy failed" in alerts[0]

    def test_alert_on_verification_failure(self, tmp_path):
        alerts = []
        mgr = CheckpointManager(
            local_dir=str(tmp_path / "local"),
            drive_dir=None,
            keep=5,
            alert_callback=lambda msg: alerts.append(msg),
        )
        info = mgr.save(FakeModel(), step=100)

        # Corrupt the file
        for f in Path(info.local_path).glob("model*"):
            f.write_bytes(b"x")

        with pytest.raises(ValueError, match="verification FAILED"):
            mgr.load(info.local_path)

        assert len(alerts) == 1
        assert "verification FAILED" in alerts[0]


# =============================================================================
# TEST: CHECKPOINT INFO
# =============================================================================

class TestCheckpointInfo:
    def test_to_dict_roundtrip(self):
        info = CheckpointInfo(
            step=1000,
            timestamp="2026-02-13T10:00:00+00:00",
            best_reward=2.5,
            sharpe=1.8,
            curriculum_phase=2,
            files={"model.zip": "sha256:abc123"},
        )
        d = info.to_dict()
        restored = CheckpointInfo.from_dict(d)
        assert restored.step == 1000
        assert restored.sharpe == 1.8
        assert restored.files == {"model.zip": "sha256:abc123"}


# =============================================================================
# TEST: SOURCE CODE VERIFICATION
# =============================================================================

class TestSourceVerification:
    def test_no_print_in_checkpoint_manager(self):
        import ast
        src_path = os.path.join(os.path.dirname(__file__), "..", "src", "training", "checkpoint_manager.py")
        with open(src_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    pytest.fail(f"Found print() call at line {node.lineno} in checkpoint_manager.py")

    def test_checkpoint_manager_in_trainer_imports(self):
        src_path = os.path.join(os.path.dirname(__file__), "..", "src", "training", "sophisticated_trainer.py")
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "from .checkpoint_manager import CheckpointManager" in source

    def test_colab_uses_checkpoint_manager(self):
        """Colab script delegates to SophisticatedTrainer which uses CheckpointManager internally."""
        src_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "Colab_Full_Training_Script.py")
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        # Sprint 14: Colab is now a thin driver — it imports SophisticatedTrainer
        # which internally creates and uses a CheckpointManager.
        assert "SophisticatedTrainer" in source
        assert "checkpoint" in source.lower()
