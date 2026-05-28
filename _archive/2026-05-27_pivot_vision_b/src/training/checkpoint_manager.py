# =============================================================================
# CHECKPOINT MANAGER - Training Resilience (Sprint 4)
# =============================================================================
# Robust checkpoint system designed for Colab's unreliable runtime:
#
#   - SHA-256 verification: Every saved file gets a hash in the manifest
#   - Dual-write: Save locally first (fast), then copy to Drive (persistent)
#   - Resume-from-checkpoint: Scan, verify, resume from latest valid checkpoint
#   - Rotation: Keep N most recent checkpoints (Drive free tier = 15GB)
#   - Alert on failure: Optional Telegram alert if verification fails
#
# =============================================================================

import os
import json
import shutil
import hashlib
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CheckpointInfo:
    """Metadata for a single checkpoint."""
    step: int
    timestamp: str
    best_reward: float = 0.0
    sharpe: float = 0.0
    curriculum_phase: int = 0
    learning_rate: float = 3e-5
    files: Dict[str, str] = field(default_factory=dict)  # filename -> "sha256:hexdigest"
    local_path: str = ""
    drive_path: str = ""
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointInfo":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """
    Production checkpoint manager with integrity verification and dual-write.

    Usage:
        mgr = CheckpointManager(local_dir="/content/checkpoints",
                                 drive_dir="/content/drive/MyDrive/checkpoints")
        # Save
        info = mgr.save(model, step=100000, metrics={'sharpe': 1.5})
        # Resume
        model, metadata = mgr.load_latest()
        # Cleanup
        mgr.cleanup(keep=5)
    """

    MANIFEST_SUFFIX = ".manifest.json"
    _MIN_FILE_SIZE = 100  # Minimum bytes for a valid model file

    def __init__(
        self,
        local_dir: str,
        drive_dir: Optional[str] = None,
        keep: int = 5,
        alert_callback: Optional[Any] = None,
    ):
        """
        Args:
            local_dir: Fast local storage (e.g. /content/checkpoints)
            drive_dir: Persistent storage (e.g. Google Drive path). None to disable.
            keep: Number of checkpoints to retain during rotation.
            alert_callback: Optional callable(message: str) for failure alerts.
        """
        self.local_dir = Path(local_dir)
        self.drive_dir = Path(drive_dir) if drive_dir else None
        self.keep = max(1, keep)
        self._alert = alert_callback

        # Create directories
        self.local_dir.mkdir(parents=True, exist_ok=True)
        if self.drive_dir:
            try:
                self.drive_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning("Drive directory unavailable: %s", e)
                self.drive_dir = None

        logger.info(
            "CheckpointManager initialized: local=%s, drive=%s, keep=%d",
            self.local_dir, self.drive_dir, self.keep,
        )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    def save(
        self,
        model: Any,
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        extra_files: Optional[Dict[str, str]] = None,
    ) -> CheckpointInfo:
        """
        Save a checkpoint with SHA-256 manifest.

        Args:
            model: SB3 model with a .save() method.
            step: Current training step.
            metrics: Optional dict of metrics (sharpe, reward, etc.).
            extra_files: Optional dict of {dest_name: source_path} for extra artifacts.

        Returns:
            CheckpointInfo with paths and hashes.
        """
        metrics = metrics or {}
        checkpoint_name = f"checkpoint_step_{step}"
        local_ckpt_dir = self.local_dir / checkpoint_name
        local_ckpt_dir.mkdir(parents=True, exist_ok=True)

        file_hashes: Dict[str, str] = {}

        # 1. Save model
        model_path = str(local_ckpt_dir / "model.zip")
        model.save(model_path)
        # SB3 may or may not add .zip extension
        actual_model_path = model_path if os.path.exists(model_path) else model_path + ".zip"
        if not os.path.exists(actual_model_path):
            # Try without .zip
            actual_model_path = model_path.replace(".zip", "")
        if os.path.exists(actual_model_path):
            file_hashes[os.path.basename(actual_model_path)] = self._sha256(actual_model_path)

        # 2. Save extra files (scaler, PCA transformer, etc.)
        if extra_files:
            for dest_name, source_path in extra_files.items():
                if os.path.exists(source_path):
                    dest_path = str(local_ckpt_dir / dest_name)
                    shutil.copy2(source_path, dest_path)
                    file_hashes[dest_name] = self._sha256(dest_path)

        # 3. Build manifest
        info = CheckpointInfo(
            step=step,
            timestamp=datetime.now(timezone.utc).isoformat(),
            best_reward=metrics.get("best_reward", 0.0),
            sharpe=metrics.get("sharpe", 0.0),
            curriculum_phase=metrics.get("curriculum_phase", 0),
            learning_rate=metrics.get("learning_rate", 3e-5),
            files=file_hashes,
            local_path=str(local_ckpt_dir),
            verified=True,
        )

        logger.info("Checkpoint saved: step=%d, files=%d, path=%s", step, len(file_hashes), local_ckpt_dir)

        # 4. Dual-write to Drive
        if self.drive_dir:
            drive_ckpt_dir = self.drive_dir / checkpoint_name
            try:
                if drive_ckpt_dir.exists():
                    shutil.rmtree(drive_ckpt_dir)
                info.drive_path = str(drive_ckpt_dir)
                # Write manifest first (with drive_path), then copy entire dir
                manifest_path = local_ckpt_dir / (checkpoint_name + self.MANIFEST_SUFFIX)
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(info.to_dict(), f, indent=2)
                shutil.copytree(str(local_ckpt_dir), str(drive_ckpt_dir))
                logger.info("Checkpoint copied to Drive: %s", drive_ckpt_dir)
            except OSError as e:
                info.drive_path = ""
                logger.warning("Drive copy failed (will retry next save): %s", e)
                if self._alert:
                    self._alert(f"Checkpoint Drive copy failed at step {step}: {e}")

        # 5. Write manifest (or re-write if drive copy updated drive_path)
        manifest_path = local_ckpt_dir / (checkpoint_name + self.MANIFEST_SUFFIX)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(info.to_dict(), f, indent=2)

        # 6. Rotate old checkpoints
        self.cleanup(keep=self.keep)

        return info

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    def load(self, checkpoint_path: str, model_cls: Any = None) -> Tuple[Any, CheckpointInfo]:
        """
        Load and verify a specific checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.
            model_cls: SB3 model class (e.g. PPO). If None, returns path only.

        Returns:
            Tuple of (loaded_model_or_path, CheckpointInfo).

        Raises:
            ValueError: If checkpoint fails verification.
        """
        ckpt_dir = Path(checkpoint_path)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Find manifest
        manifests = list(ckpt_dir.glob("*" + self.MANIFEST_SUFFIX))
        if not manifests:
            raise ValueError(f"No manifest found in {checkpoint_path}")

        with open(manifests[0], "r", encoding="utf-8") as f:
            info = CheckpointInfo.from_dict(json.load(f))
        info.local_path = str(ckpt_dir)

        # Verify integrity
        if not self.verify(checkpoint_path):
            msg = f"Checkpoint verification FAILED: {checkpoint_path}"
            logger.error(msg)
            if self._alert:
                self._alert(msg)
            raise ValueError(msg)

        info.verified = True

        # Load model if class provided
        if model_cls is not None:
            model_file = ckpt_dir / "model.zip"
            if not model_file.exists():
                model_file = ckpt_dir / "model"
            model = model_cls.load(str(model_file))
            return model, info

        return str(ckpt_dir), info

    def load_latest(self, model_cls: Any = None) -> Optional[Tuple[Any, CheckpointInfo]]:
        """
        Find and load the latest valid checkpoint.

        Scans local first, then Drive. Tries each checkpoint from newest
        to oldest until one passes verification.

        Returns:
            Tuple of (model_or_path, CheckpointInfo) or None if no valid checkpoint.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            logger.info("No checkpoints found to resume from")
            return None

        for info in checkpoints:
            path = info.local_path
            # If local doesn't exist, try Drive
            if not os.path.exists(path) and info.drive_path and os.path.exists(info.drive_path):
                logger.info("Restoring from Drive: %s", info.drive_path)
                try:
                    shutil.copytree(info.drive_path, path)
                except OSError as e:
                    logger.warning("Failed to restore from Drive: %s", e)
                    continue

            if not os.path.exists(path):
                continue

            try:
                return self.load(path, model_cls=model_cls)
            except (ValueError, FileNotFoundError) as e:
                logger.warning("Checkpoint step %d failed: %s, trying older...", info.step, e)
                continue

        logger.warning("No valid checkpoints found")
        return None

    # -------------------------------------------------------------------------
    # VERIFY
    # -------------------------------------------------------------------------

    def verify(self, checkpoint_path: str) -> bool:
        """
        Verify checkpoint integrity using SHA-256 hashes from the manifest.

        Returns True if all files match their recorded hashes.
        """
        ckpt_dir = Path(checkpoint_path)
        manifests = list(ckpt_dir.glob("*" + self.MANIFEST_SUFFIX))
        if not manifests:
            logger.error("No manifest in %s", checkpoint_path)
            return False

        with open(manifests[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        file_hashes = data.get("files", {})
        if not file_hashes:
            logger.warning("Empty file manifest in %s", checkpoint_path)
            return False

        for filename, expected_hash in file_hashes.items():
            file_path = ckpt_dir / filename
            if not file_path.exists():
                logger.error("Missing file: %s", file_path)
                return False

            # Check minimum file size
            if file_path.stat().st_size < self._MIN_FILE_SIZE:
                logger.error("File too small (likely corrupted): %s (%d bytes)", file_path, file_path.stat().st_size)
                return False

            actual_hash = self._sha256(str(file_path))
            if actual_hash != expected_hash:
                logger.error(
                    "Hash mismatch for %s: expected %s, got %s",
                    filename, expected_hash[:16] + "...", actual_hash[:16] + "...",
                )
                return False

        logger.debug("Checkpoint verified: %s", checkpoint_path)
        return True

    # -------------------------------------------------------------------------
    # LIST
    # -------------------------------------------------------------------------

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """
        List all checkpoints sorted by step (newest first).

        Scans both local and Drive directories.
        """
        seen_steps = set()
        checkpoints: List[CheckpointInfo] = []

        for base_dir in [self.local_dir, self.drive_dir]:
            if base_dir is None or not base_dir.exists():
                continue
            is_drive = (base_dir == self.drive_dir)
            for manifest_path in base_dir.rglob("*" + self.MANIFEST_SUFFIX):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        info = CheckpointInfo.from_dict(json.load(f))
                    if info.step in seen_steps:
                        continue
                    seen_steps.add(info.step)
                    # Set paths based on where we found the manifest
                    if is_drive:
                        info.drive_path = str(manifest_path.parent)
                        if not info.local_path:
                            # Infer local path from checkpoint name
                            info.local_path = str(self.local_dir / manifest_path.parent.name)
                    else:
                        info.local_path = str(manifest_path.parent)
                    checkpoints.append(info)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning("Corrupt manifest at %s: %s", manifest_path, e)

        checkpoints.sort(key=lambda c: c.step, reverse=True)
        return checkpoints

    # -------------------------------------------------------------------------
    # CLEANUP / ROTATION
    # -------------------------------------------------------------------------

    def cleanup(self, keep: Optional[int] = None) -> int:
        """
        Delete old checkpoints, keeping the N most recent.

        Args:
            keep: Number of checkpoints to keep. Defaults to self.keep.

        Returns:
            Number of checkpoints deleted.
        """
        keep = keep if keep is not None else self.keep
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep:
            return 0

        to_delete = checkpoints[keep:]
        deleted = 0

        for info in to_delete:
            for path_str in [info.local_path, info.drive_path]:
                if path_str and os.path.exists(path_str):
                    try:
                        shutil.rmtree(path_str)
                        logger.info("Deleted old checkpoint: %s", path_str)
                        deleted += 1
                    except OSError as e:
                        logger.warning("Failed to delete %s: %s", path_str, e)

        return deleted

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------

    @staticmethod
    def _sha256(filepath: str) -> str:
        """Compute SHA-256 hash of a file. Returns 'sha256:<hex>' string."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"
