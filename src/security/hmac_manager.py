# =============================================================================
# HMAC KEY MANAGER - Persistent HMAC Key Storage for Audit Integrity
# =============================================================================
# Ensures audit log integrity can be verified across system restarts.
#
# Features:
#   - Persistent key storage (Vault, encrypted file, HSM simulation)
#   - Key versioning and rotation
#   - Backward compatibility for verification
#   - Secure key derivation
#   - Audit trail of key operations
#
# Usage:
#   manager = HMACKeyManager(config)
#   key = manager.get_current_key()
#   signature = manager.sign(data)
#   is_valid = manager.verify(data, signature)
#
# =============================================================================

import os
import json
import hmac
import hashlib
import secrets
import base64
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import threading


# =============================================================================
# CONFIGURATION
# =============================================================================

class KeyStorageBackend(Enum):
    """Available key storage backends."""
    VAULT = "vault"
    ENCRYPTED_FILE = "encrypted_file"
    ENVIRONMENT = "environment"
    HSM_SIMULATED = "hsm_simulated"  # For development/testing


@dataclass
class HMACKeyConfig:
    """Configuration for HMACKeyManager."""
    # Storage backend
    storage_backend: KeyStorageBackend = KeyStorageBackend.ENCRYPTED_FILE

    # Key settings
    key_size_bytes: int = 32  # 256-bit keys
    key_rotation_days: int = 90
    max_key_versions: int = 10  # Keep last N versions for verification

    # File storage
    key_file_path: str = ".hmac_keys.enc"
    master_key_env: str = "HMAC_MASTER_KEY"

    # Vault storage
    vault_path: str = "trading-bot/hmac-keys"

    # Audit settings
    log_all_operations: bool = True

    @classmethod
    def from_environment(cls) -> 'HMACKeyConfig':
        """Create config from environment variables."""
        backend_str = os.getenv('HMAC_STORAGE_BACKEND', 'encrypted_file')
        backend = KeyStorageBackend(backend_str)

        return cls(
            storage_backend=backend,
            key_file_path=os.getenv('HMAC_KEY_FILE', '.hmac_keys.enc'),
        )


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HMACKeyVersion:
    """A versioned HMAC key."""
    version: int
    key: bytes
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool = True
    rotation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage (key is base64 encoded)."""
        return {
            'version': self.version,
            'key': base64.b64encode(self.key).decode('ascii'),
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active,
            'rotation_reason': self.rotation_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HMACKeyVersion':
        """Deserialize from storage."""
        return cls(
            version=data['version'],
            key=base64.b64decode(data['key']),
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            is_active=data.get('is_active', True),
            rotation_reason=data.get('rotation_reason'),
        )


@dataclass
class SignedData:
    """Data with HMAC signature and metadata."""
    data: bytes
    signature: str
    key_version: int
    algorithm: str = "HMAC-SHA256"
    signed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signature': self.signature,
            'key_version': self.key_version,
            'algorithm': self.algorithm,
            'signed_at': self.signed_at.isoformat(),
        }


# =============================================================================
# HMAC KEY MANAGER
# =============================================================================

class HMACKeyManager:
    """
    Production-grade HMAC key management with persistence.

    Features:
    - Persistent key storage across restarts
    - Key versioning for rotation without breaking verification
    - Multiple storage backends (Vault, encrypted file, HSM simulation)
    - Secure key generation
    - Thread-safe operations

    Example:
        config = HMACKeyConfig.from_environment()
        manager = HMACKeyManager(config)

        # Sign data
        signature = manager.sign(audit_data)

        # Verify data
        is_valid = manager.verify(audit_data, signature)

        # Rotate key (e.g., after 90 days)
        manager.rotate_key(reason="Scheduled rotation")
    """

    def __init__(self, config: HMACKeyConfig, secret_manager=None):
        """
        Initialize HMAC Key Manager.

        Args:
            config: HMACKeyConfig configuration
            secret_manager: Optional SecretManager for Vault backend
        """
        self.config = config
        self.secret_manager = secret_manager
        self._logger = logging.getLogger("security.hmac")
        self._lock = threading.RLock()

        # Key storage
        self._keys: Dict[int, HMACKeyVersion] = {}
        self._current_version: int = 0

        # Encryption for file storage
        self._fernet: Optional[Fernet] = None
        self._init_encryption()

        # Load existing keys
        self._load_keys()

        # Ensure we have at least one key
        if not self._keys:
            self._generate_initial_key()

    def _init_encryption(self) -> None:
        """Initialize encryption for file-based storage."""
        master_key = os.getenv(self.config.master_key_env)

        if master_key:
            # Derive Fernet key from master key
            salt = b'hmac_key_manager_salt_v1'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
            self._fernet = Fernet(key)
            self._logger.info("File encryption initialized")
        else:
            # Generate and store a new master key
            self._logger.warning(
                f"No master key found in {self.config.master_key_env}. "
                "Generating new master key - SAVE THIS TO YOUR ENVIRONMENT!"
            )
            new_master_key = secrets.token_urlsafe(32)
            print(f"\n{'='*70}")
            print("CRITICAL: Save this master key to your environment:")
            print(f"export {self.config.master_key_env}='{new_master_key}'")
            print(f"{'='*70}\n")

            # Use it for this session
            salt = b'hmac_key_manager_salt_v1'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(new_master_key.encode()))
            self._fernet = Fernet(key)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_current_key(self) -> bytes:
        """
        Get the current active HMAC key.

        Returns:
            Current HMAC key bytes
        """
        with self._lock:
            if self._current_version not in self._keys:
                raise RuntimeError("No active HMAC key available")
            return self._keys[self._current_version].key

    def get_key_version(self, version: int) -> Optional[bytes]:
        """
        Get a specific key version (for verification of old signatures).

        Args:
            version: Key version number

        Returns:
            Key bytes if found, None otherwise
        """
        with self._lock:
            if version in self._keys:
                return self._keys[version].key
            return None

    def sign(self, data: bytes) -> SignedData:
        """
        Sign data with current HMAC key.

        Args:
            data: Data to sign (bytes)

        Returns:
            SignedData with signature and metadata
        """
        with self._lock:
            key = self.get_current_key()
            signature = hmac.new(key, data, hashlib.sha256).hexdigest()

            self._log_operation("sign", self._current_version)

            return SignedData(
                data=data,
                signature=signature,
                key_version=self._current_version,
            )

    def sign_dict(self, data: Dict[str, Any]) -> str:
        """
        Sign a dictionary (serialized as sorted JSON).

        Args:
            data: Dictionary to sign

        Returns:
            Signature string
        """
        json_data = json.dumps(data, sort_keys=True, default=str)
        signed = self.sign(json_data.encode())
        return signed.signature

    def verify(
        self,
        data: bytes,
        signature: str,
        key_version: Optional[int] = None
    ) -> bool:
        """
        Verify a signature.

        Args:
            data: Original data
            signature: Signature to verify
            key_version: Key version used for signing (tries all if None)

        Returns:
            True if signature is valid
        """
        with self._lock:
            if key_version is not None:
                # Verify with specific version
                key = self.get_key_version(key_version)
                if key is None:
                    self._logger.warning(f"Key version {key_version} not found")
                    return False

                expected = hmac.new(key, data, hashlib.sha256).hexdigest()
                is_valid = hmac.compare_digest(expected, signature)

                self._log_operation("verify", key_version, success=is_valid)
                return is_valid

            else:
                # Try all versions (for backward compatibility)
                for version, key_obj in sorted(
                    self._keys.items(),
                    reverse=True  # Try newest first
                ):
                    expected = hmac.new(key_obj.key, data, hashlib.sha256).hexdigest()
                    if hmac.compare_digest(expected, signature):
                        self._log_operation("verify", version, success=True)
                        return True

                self._log_operation("verify", None, success=False)
                return False

    def verify_dict(
        self,
        data: Dict[str, Any],
        signature: str,
        key_version: Optional[int] = None
    ) -> bool:
        """
        Verify a dictionary signature.

        Args:
            data: Original dictionary
            signature: Signature to verify
            key_version: Key version (optional)

        Returns:
            True if signature is valid
        """
        json_data = json.dumps(data, sort_keys=True, default=str)
        return self.verify(json_data.encode(), signature, key_version)

    def rotate_key(self, reason: str = "Manual rotation") -> int:
        """
        Rotate to a new HMAC key.

        Args:
            reason: Reason for rotation (for audit)

        Returns:
            New key version number
        """
        with self._lock:
            # Mark current key as inactive
            if self._current_version in self._keys:
                self._keys[self._current_version].is_active = False
                self._keys[self._current_version].rotation_reason = reason

            # Generate new key
            new_version = self._current_version + 1
            new_key = secrets.token_bytes(self.config.key_size_bytes)

            self._keys[new_version] = HMACKeyVersion(
                version=new_version,
                key=new_key,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=self.config.key_rotation_days),
                is_active=True,
            )

            self._current_version = new_version

            # Prune old versions
            self._prune_old_keys()

            # Persist
            self._save_keys()

            self._logger.info(f"Key rotated to version {new_version}: {reason}")
            self._log_operation("rotate", new_version)

            return new_version

    def check_rotation_needed(self) -> Tuple[bool, Optional[str]]:
        """
        Check if key rotation is needed.

        Returns:
            Tuple of (needs_rotation, reason)
        """
        with self._lock:
            if self._current_version not in self._keys:
                return True, "No active key"

            current_key = self._keys[self._current_version]

            # Check expiration
            if current_key.expires_at and datetime.utcnow() > current_key.expires_at:
                return True, f"Key expired on {current_key.expires_at}"

            # Check age
            age = datetime.utcnow() - current_key.created_at
            if age > timedelta(days=self.config.key_rotation_days):
                return True, f"Key is {age.days} days old"

            return False, None

    def get_key_info(self) -> Dict[str, Any]:
        """
        Get information about current key state.

        Returns:
            Dict with key metadata (not the actual keys)
        """
        with self._lock:
            current = self._keys.get(self._current_version)

            return {
                'current_version': self._current_version,
                'total_versions': len(self._keys),
                'current_key_created': current.created_at.isoformat() if current else None,
                'current_key_expires': current.expires_at.isoformat() if current and current.expires_at else None,
                'needs_rotation': self.check_rotation_needed()[0],
                'versions_available': sorted(self._keys.keys()),
            }

    # =========================================================================
    # STORAGE OPERATIONS
    # =========================================================================

    def _generate_initial_key(self) -> None:
        """Generate the initial HMAC key."""
        self._logger.info("Generating initial HMAC key")

        initial_key = secrets.token_bytes(self.config.key_size_bytes)

        self._keys[1] = HMACKeyVersion(
            version=1,
            key=initial_key,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=self.config.key_rotation_days),
            is_active=True,
        )

        self._current_version = 1
        self._save_keys()

        self._log_operation("generate_initial", 1)

    def _load_keys(self) -> None:
        """Load keys from storage backend."""
        try:
            if self.config.storage_backend == KeyStorageBackend.ENCRYPTED_FILE:
                self._load_from_file()
            elif self.config.storage_backend == KeyStorageBackend.VAULT:
                self._load_from_vault()
            elif self.config.storage_backend == KeyStorageBackend.ENVIRONMENT:
                self._load_from_environment()
            elif self.config.storage_backend == KeyStorageBackend.HSM_SIMULATED:
                self._load_from_file()  # Same as file for simulation

        except FileNotFoundError:
            self._logger.info("No existing keys found, will generate new ones")
        except Exception as e:
            self._logger.error(f"Failed to load keys: {e}")

    def _save_keys(self) -> None:
        """Save keys to storage backend."""
        try:
            if self.config.storage_backend == KeyStorageBackend.ENCRYPTED_FILE:
                self._save_to_file()
            elif self.config.storage_backend == KeyStorageBackend.VAULT:
                self._save_to_vault()
            elif self.config.storage_backend == KeyStorageBackend.HSM_SIMULATED:
                self._save_to_file()

            self._logger.debug("Keys saved successfully")

        except Exception as e:
            self._logger.error(f"Failed to save keys: {e}")
            raise

    def _load_from_file(self) -> None:
        """Load keys from encrypted file."""
        if not self._fernet:
            raise RuntimeError("Encryption not initialized")

        file_path = Path(self.config.key_file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Key file not found: {file_path}")

        encrypted_data = file_path.read_bytes()
        decrypted_data = self._fernet.decrypt(encrypted_data)
        data = json.loads(decrypted_data.decode())

        self._current_version = data['current_version']
        self._keys = {
            int(k): HMACKeyVersion.from_dict(v)
            for k, v in data['keys'].items()
        }

        self._logger.info(
            f"Loaded {len(self._keys)} key versions, current: {self._current_version}"
        )

    def _save_to_file(self) -> None:
        """Save keys to encrypted file."""
        if not self._fernet:
            raise RuntimeError("Encryption not initialized")

        data = {
            'current_version': self._current_version,
            'keys': {k: v.to_dict() for k, v in self._keys.items()},
            'saved_at': datetime.utcnow().isoformat(),
        }

        json_data = json.dumps(data, indent=2)
        encrypted_data = self._fernet.encrypt(json_data.encode())

        file_path = Path(self.config.key_file_path)
        file_path.write_bytes(encrypted_data)

        # Secure file permissions (Unix-like systems)
        try:
            os.chmod(file_path, 0o600)
        except (OSError, AttributeError):
            pass  # Windows or permission error

    def _load_from_vault(self) -> None:
        """Load keys from Vault."""
        if not self.secret_manager:
            raise RuntimeError("SecretManager required for Vault backend")

        data = self.secret_manager.get_secret(self.config.vault_path)

        self._current_version = data['current_version']
        self._keys = {
            int(k): HMACKeyVersion.from_dict(v)
            for k, v in data['keys'].items()
        }

    def _save_to_vault(self) -> None:
        """Save keys to Vault."""
        if not self.secret_manager:
            raise RuntimeError("SecretManager required for Vault backend")

        data = {
            'current_version': self._current_version,
            'keys': {str(k): v.to_dict() for k, v in self._keys.items()},
        }

        self.secret_manager.store_secret(self.config.vault_path, data)

    def _load_from_environment(self) -> None:
        """Load key from environment variable."""
        key_b64 = os.getenv('HMAC_SIGNING_KEY')
        if not key_b64:
            raise FileNotFoundError("HMAC_SIGNING_KEY not set")

        key = base64.b64decode(key_b64)

        self._keys[1] = HMACKeyVersion(
            version=1,
            key=key,
            created_at=datetime.utcnow(),
            expires_at=None,  # No expiration for env-based key
            is_active=True,
        )
        self._current_version = 1

    def _prune_old_keys(self) -> None:
        """Remove old key versions beyond max_key_versions."""
        if len(self._keys) <= self.config.max_key_versions:
            return

        # Sort versions and remove oldest
        versions = sorted(self._keys.keys())
        to_remove = versions[:-self.config.max_key_versions]

        for version in to_remove:
            del self._keys[version]
            self._logger.info(f"Pruned old key version {version}")

    def _log_operation(
        self,
        operation: str,
        version: Optional[int],
        success: bool = True
    ) -> None:
        """Log key operation for audit."""
        if not self.config.log_all_operations:
            return

        self._logger.debug(
            f"HMAC operation: {operation}, version={version}, success={success}"
        )
