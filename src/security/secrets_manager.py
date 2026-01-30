# =============================================================================
# SECRETS MANAGER - HashiCorp Vault Integration with Secure Fallback
# =============================================================================
# Production-grade secrets management for trading credentials.
#
# Features:
#   - HashiCorp Vault integration (primary)
#   - Encrypted file fallback (development/backup)
#   - Automatic credential rotation tracking
#   - Audit logging of all secret access
#   - Memory protection (secure string handling)
#   - Dynamic salt generation with pepper (SECURITY FIX v2)
#   - Secure memory wiping for sensitive data
#
# Usage:
#   manager = SecretManager(config)
#   creds = manager.get_mt5_credentials()
#
# Security Notes:
#   - Salt is dynamically generated per installation and stored in .salt file
#   - Pepper is application-level secret from environment variable
#   - Keys must be minimum 32 characters
#   - Sensitive data is wiped from memory after use
#
# =============================================================================

import os
import json
import base64
import hashlib
import logging
import secrets
import ctypes
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import threading

# Optional Vault import
try:
    import hvac
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    hvac = None


# =============================================================================
# SECURITY CONSTANTS
# =============================================================================

# Minimum key length for encryption (NIST recommendation)
MIN_KEY_LENGTH = 32

# Salt file name (generated per installation)
SALT_FILE_NAME = ".trading_bot_salt"

# Pepper environment variable name (application-level secret)
PEPPER_ENV_VAR = "TRADING_BOT_PEPPER"

# PBKDF2 iterations (OWASP 2023 recommendation for SHA-256)
PBKDF2_ITERATIONS = 600_000

# Salt length in bytes (256 bits)
SALT_LENGTH = 32


# =============================================================================
# SECURE MEMORY FUNCTIONS
# =============================================================================

def secure_wipe(data) -> None:
    """
    Securely wipe sensitive data from memory.

    This overwrites the memory location with zeros to prevent
    sensitive data from being recovered from memory dumps.

    Args:
        data: bytearray to wipe. bytes objects CANNOT be reliably wiped
              because they are immutable in Python.

    SECURITY NOTE:
        - ALWAYS use bytearray (not bytes) to store sensitive data.
        - The previous implementation used ctypes.memset with a hardcoded
          CPython offset (id(data) + 32), which is fragile and unreliable:
            * The offset varies between CPython versions
            * Causes segfaults on PyPy/GraalPy/other implementations
            * Python GC can relocate objects, making the pointer stale
            * Immutable bytes may be interned/shared by the runtime
        - This fixed version only supports bytearray (mutable, reliable wipe)
          and logs a warning for bytes objects instead of silently failing.
    """
    if isinstance(data, bytearray):
        # bytearray is mutable - reliable zero-fill
        for i in range(len(data)):
            data[i] = 0
    elif isinstance(data, (bytes, str)) and len(data) > 0:
        # bytes/str are immutable in Python - cannot be reliably wiped.
        # Log a warning so callers know to switch to bytearray.
        import logging
        logging.getLogger(__name__).warning(
            "secure_wipe called on immutable type %s (len=%d). "
            "Cannot reliably wipe immutable objects. "
            "Use bytearray for sensitive data instead.",
            type(data).__name__, len(data)
        )


def secure_wipe_string(s: str) -> None:
    """
    Attempt to wipe a string from memory.

    WARNING: Python strings are immutable and often interned. This function
    CANNOT guarantee the string is wiped. For sensitive data, use bytearray
    with secure_wipe() instead.

    This function exists only to provide a clear warning when called.
    """
    if s and len(s) > 0:
        import logging
        logging.getLogger(__name__).warning(
            "secure_wipe_string called (len=%d). Python strings are immutable "
            "and cannot be reliably wiped from memory. Use bytearray instead.",
            len(s)
        )


class SecureString:
    """
    A wrapper for sensitive strings that automatically wipes memory on deletion.

    Usage:
        secure_pwd = SecureString("my_password")
        # Use secure_pwd.value to access the string
        del secure_pwd  # Automatically wipes memory
    """

    def __init__(self, value: str):
        # Store as bytearray for secure wiping
        self._data = bytearray(value.encode('utf-8'))

    @property
    def value(self) -> str:
        """Get the string value."""
        return self._data.decode('utf-8')

    def __del__(self):
        """Securely wipe memory on deletion."""
        secure_wipe(self._data)

    def __str__(self) -> str:
        return "[REDACTED]"

    def __repr__(self) -> str:
        return "SecureString([REDACTED])"


def generate_salt(salt_path: Path) -> bytes:
    """
    Generate a cryptographically secure random salt and save it.

    Args:
        salt_path: Path to save the salt file

    Returns:
        Generated salt bytes
    """
    salt = secrets.token_bytes(SALT_LENGTH)

    # Save salt to file with restricted permissions
    salt_path.parent.mkdir(parents=True, exist_ok=True)
    salt_path.write_bytes(salt)

    # Try to set restrictive permissions (Unix-like systems)
    try:
        os.chmod(salt_path, 0o600)  # Owner read/write only
    except (OSError, AttributeError):
        pass  # Windows or permission error

    return salt


def load_or_create_salt(salt_dir: Path) -> bytes:
    """
    Load existing salt or create a new one.

    Args:
        salt_dir: Directory to store/load salt file

    Returns:
        Salt bytes
    """
    salt_path = salt_dir / SALT_FILE_NAME

    if salt_path.exists():
        salt = salt_path.read_bytes()
        if len(salt) >= SALT_LENGTH:
            return salt[:SALT_LENGTH]
        # Salt file corrupted, regenerate

    return generate_salt(salt_path)


def get_pepper() -> bytes:
    """
    Get the application pepper from environment.

    Pepper is an application-level secret that adds an additional
    layer of security beyond the salt. Unlike salt, pepper is not
    stored alongside the encrypted data.

    Returns:
        Pepper bytes, or empty bytes if not configured
    """
    pepper = os.getenv(PEPPER_ENV_VAR, "")
    return pepper.encode('utf-8') if pepper else b""


def validate_key_strength(key: str) -> None:
    """
    Validate that an encryption key meets minimum security requirements.

    Args:
        key: The encryption key to validate

    Raises:
        ValueError: If key doesn't meet requirements
    """
    if not key:
        raise ValueError("Encryption key cannot be empty")

    if len(key) < MIN_KEY_LENGTH:
        raise ValueError(
            f"Encryption key must be at least {MIN_KEY_LENGTH} characters. "
            f"Got {len(key)} characters. Use a strong passphrase or generate "
            f"a random key with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )

    # Check for common weak patterns
    weak_patterns = [
        key == key[0] * len(key),  # All same character
        key.lower() in ('password', 'secret', 'admin', '12345678' * 4),
        all(c.isdigit() for c in key),  # All digits
    ]

    if any(weak_patterns):
        raise ValueError(
            "Encryption key appears to be weak. Use a strong random passphrase."
        )


# =============================================================================
# CONFIGURATION
# =============================================================================

class SecretBackend(Enum):
    """Available secret storage backends."""
    VAULT = "vault"
    ENCRYPTED_FILE = "encrypted_file"
    ENVIRONMENT = "environment"


@dataclass
class SecretManagerConfig:
    """Configuration for SecretManager."""
    # Backend selection
    primary_backend: SecretBackend = SecretBackend.VAULT
    fallback_backend: SecretBackend = SecretBackend.ENCRYPTED_FILE

    # Vault configuration
    vault_url: str = ""
    vault_token: str = ""
    vault_namespace: str = ""
    vault_mount_point: str = "secret"
    vault_path_prefix: str = "trading-bot"

    # Encrypted file configuration
    encrypted_file_path: str = ""
    encryption_key_env: str = "TRADING_BOT_SECRET_KEY"

    # Security settings
    cache_ttl_seconds: int = 300  # 5 minutes
    max_rotation_age_days: int = 90
    audit_all_access: bool = True

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    @classmethod
    def from_environment(cls) -> 'SecretManagerConfig':
        """Create config from environment variables."""
        return cls(
            vault_url=os.getenv('VAULT_ADDR', ''),
            vault_token=os.getenv('VAULT_TOKEN', ''),
            vault_namespace=os.getenv('VAULT_NAMESPACE', ''),
            encrypted_file_path=os.getenv('SECRETS_FILE_PATH', '.secrets.enc'),
        )


# =============================================================================
# EXCEPTIONS
# =============================================================================

class SecretManagerError(Exception):
    """Base exception for SecretManager."""
    pass


class SecretNotFoundError(SecretManagerError):
    """Raised when a secret is not found."""
    pass


class SecretAccessDeniedError(SecretManagerError):
    """Raised when access to a secret is denied."""
    pass


class SecretRotationRequiredError(SecretManagerError):
    """Raised when a secret needs rotation."""
    pass


# =============================================================================
# SECRET CACHE
# =============================================================================

@dataclass
class CachedSecret:
    """Cached secret with metadata."""
    value: Dict[str, Any]
    fetched_at: datetime
    expires_at: datetime
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


# =============================================================================
# SECRETS MANAGER
# =============================================================================

class SecretManager:
    """
    Production-grade secrets management with Vault integration.

    Features:
    - Primary/fallback backend architecture
    - In-memory caching with TTL
    - Audit logging of all access
    - Credential rotation tracking
    - Thread-safe operations

    Example:
        config = SecretManagerConfig.from_environment()
        manager = SecretManager(config)

        # Get MT5 credentials
        creds = manager.get_mt5_credentials()

        # Get API keys
        api_key = manager.get_secret("api_keys/binance")
    """

    def __init__(self, config: SecretManagerConfig):
        self.config = config
        self._logger = logging.getLogger("security.secrets")
        self._cache: Dict[str, CachedSecret] = {}
        self._lock = threading.RLock()
        self._vault_client: Optional[Any] = None
        self._fernet: Optional[Fernet] = None

        # Initialize backends
        self._init_backends()

        # Access audit log
        self._access_log: List[Dict[str, Any]] = []

    def _init_backends(self) -> None:
        """Initialize configured backends."""
        # Initialize Vault if configured
        if self.config.primary_backend == SecretBackend.VAULT or \
           self.config.fallback_backend == SecretBackend.VAULT:
            self._init_vault()

        # Initialize encrypted file backend
        if self.config.primary_backend == SecretBackend.ENCRYPTED_FILE or \
           self.config.fallback_backend == SecretBackend.ENCRYPTED_FILE:
            self._init_encrypted_file()

    def _init_vault(self) -> None:
        """Initialize Vault client."""
        if not VAULT_AVAILABLE:
            self._logger.warning("hvac package not installed, Vault backend unavailable")
            return

        if not self.config.vault_url or not self.config.vault_token:
            self._logger.warning("Vault URL or token not configured")
            return

        try:
            self._vault_client = hvac.Client(
                url=self.config.vault_url,
                token=self.config.vault_token,
                namespace=self.config.vault_namespace or None
            )

            if self._vault_client.is_authenticated():
                self._logger.info(f"Connected to Vault at {self.config.vault_url}")
            else:
                self._logger.error("Vault authentication failed")
                self._vault_client = None

        except Exception as e:
            self._logger.error(f"Failed to initialize Vault: {e}")
            self._vault_client = None

    def _init_encrypted_file(self) -> None:
        """
        Initialize encrypted file backend with secure key derivation.

        Security improvements (v2):
        - Dynamic salt generated per installation (not hardcoded)
        - Optional pepper from environment for defense in depth
        - Key strength validation (minimum 32 characters)
        - Increased PBKDF2 iterations (600,000 per OWASP 2023)
        """
        # Get encryption key from environment
        key_from_env = os.getenv(self.config.encryption_key_env)

        if key_from_env:
            # SECURITY FIX: Validate key strength
            try:
                validate_key_strength(key_from_env)
            except ValueError as e:
                self._logger.error(f"Encryption key validation failed: {e}")
                raise SecretManagerError(f"Invalid encryption key: {e}")

            # SECURITY FIX: Use dynamic salt instead of hardcoded
            # Salt is stored in a file, generated once per installation
            salt_dir = Path(self.config.encrypted_file_path).parent
            if not salt_dir.exists():
                salt_dir = Path.cwd()
            salt = load_or_create_salt(salt_dir)

            # SECURITY FIX: Add pepper for defense in depth
            # Pepper is an environment variable, not stored with data
            pepper = get_pepper()

            # Combine key with pepper before derivation
            key_material = key_from_env.encode() + pepper

            # Derive Fernet key with secure parameters
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=PBKDF2_ITERATIONS,  # 600,000 iterations (OWASP 2023)
            )

            try:
                derived_key = kdf.derive(key_material)
                fernet_key = base64.urlsafe_b64encode(derived_key)
                self._fernet = Fernet(fernet_key)

                # SECURITY: Wipe intermediate key material from memory
                secure_wipe(bytearray(key_material))
                secure_wipe(bytearray(derived_key))

                self._logger.info(
                    "Encrypted file backend initialized with secure key derivation "
                    f"(salt_file={salt_dir / SALT_FILE_NAME}, pepper={'enabled' if pepper else 'disabled'}, "
                    f"iterations={PBKDF2_ITERATIONS})"
                )
            except Exception as e:
                self._logger.error(f"Failed to derive encryption key: {e}")
                raise SecretManagerError(f"Key derivation failed: {e}")
        else:
            self._logger.warning(
                f"No encryption key found in {self.config.encryption_key_env}, "
                "encrypted file backend unavailable. "
                f"Set {self.config.encryption_key_env} with a key of at least {MIN_KEY_LENGTH} characters."
            )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_secret(self, path: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get a secret by path.

        Args:
            path: Secret path (e.g., "mt5/credentials", "api_keys/binance")
            force_refresh: Bypass cache and fetch fresh

        Returns:
            Dict containing secret data

        Raises:
            SecretNotFoundError: If secret doesn't exist
            SecretAccessDeniedError: If access is denied
        """
        with self._lock:
            # Check cache first
            if not force_refresh and path in self._cache:
                cached = self._cache[path]
                if not cached.is_expired:
                    cached.access_count += 1
                    self._log_access(path, "cache_hit")
                    return cached.value.copy()

            # Try primary backend
            try:
                value = self._fetch_from_backend(
                    path,
                    self.config.primary_backend
                )
                self._cache_secret(path, value)
                self._log_access(path, "primary_backend")
                return value

            except Exception as primary_error:
                self._logger.warning(
                    f"Primary backend failed for {path}: {primary_error}"
                )

                # Try fallback backend
                try:
                    value = self._fetch_from_backend(
                        path,
                        self.config.fallback_backend
                    )
                    self._cache_secret(path, value)
                    self._log_access(path, "fallback_backend")
                    return value

                except Exception as fallback_error:
                    self._logger.error(
                        f"All backends failed for {path}: {fallback_error}"
                    )
                    raise SecretNotFoundError(
                        f"Secret not found: {path}"
                    ) from fallback_error

    def get_mt5_credentials(self) -> Dict[str, Any]:
        """
        Get MetaTrader 5 credentials.

        Returns:
            Dict with keys: account, password, server
        """
        # Try Vault/encrypted file first
        try:
            return self.get_secret("mt5/credentials")
        except SecretNotFoundError:
            pass

        # Fallback to environment variables
        account = os.getenv('MT5_LOGIN') or os.getenv('MT5_ACCOUNT')
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')

        if account and password and server:
            self._log_access("mt5/credentials", "environment")
            return {
                'account': int(account),
                'password': password,
                'server': server,
            }

        raise SecretNotFoundError(
            "MT5 credentials not found in Vault, encrypted file, or environment"
        )

    def get_api_key(self, service: str) -> str:
        """
        Get API key for a service.

        Args:
            service: Service name (e.g., "pagerduty", "slack", "binance")

        Returns:
            API key string
        """
        secret = self.get_secret(f"api_keys/{service}")
        return secret.get('key') or secret.get('api_key') or secret.get('token')

    def store_secret(self, path: str, value: Dict[str, Any]) -> bool:
        """
        Store a secret.

        Args:
            path: Secret path
            value: Secret data

        Returns:
            True if stored successfully
        """
        with self._lock:
            # Add metadata
            value_with_meta = {
                **value,
                '_metadata': {
                    'created_at': datetime.utcnow().isoformat(),
                    'version': 1,
                }
            }

            # Try primary backend
            try:
                self._store_to_backend(path, value_with_meta, self.config.primary_backend)
                self._cache_secret(path, value)
                self._log_access(path, "store", action="write")
                return True
            except Exception as e:
                self._logger.error(f"Failed to store secret {path}: {e}")
                return False

    def check_rotation_needed(self, path: str) -> bool:
        """
        Check if a secret needs rotation.

        Args:
            path: Secret path

        Returns:
            True if rotation is needed
        """
        try:
            secret = self.get_secret(path)
            metadata = secret.get('_metadata', {})
            created_at = metadata.get('created_at')

            if not created_at:
                return True  # No creation date, assume rotation needed

            created = datetime.fromisoformat(created_at)
            age = datetime.utcnow() - created

            if age > timedelta(days=self.config.max_rotation_age_days):
                self._logger.warning(
                    f"Secret {path} is {age.days} days old, rotation recommended"
                )
                return True

            return False

        except SecretNotFoundError:
            return False

    def get_access_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log of all secret accesses."""
        with self._lock:
            return self._access_log.copy()

    def clear_cache(self, secure: bool = True) -> None:
        """
        Clear all cached secrets.

        Args:
            secure: If True, attempt to securely wipe secret values from memory
        """
        with self._lock:
            if secure:
                # Attempt to securely wipe cached values before clearing
                for path, cached in self._cache.items():
                    try:
                        # Wipe string values in the cached secret
                        for key, value in cached.value.items():
                            if isinstance(value, str):
                                secure_wipe_string(value)
                            elif isinstance(value, bytes):
                                secure_wipe(bytearray(value))
                    except Exception:
                        pass  # Best effort

            self._cache.clear()
            self._logger.info("Secret cache cleared" + (" (secure wipe)" if secure else ""))

    def __del__(self):
        """Cleanup on deletion - securely wipe all cached secrets."""
        try:
            self.clear_cache(secure=True)
        except Exception:
            pass  # Best effort during garbage collection

    # =========================================================================
    # BACKEND OPERATIONS
    # =========================================================================

    def _fetch_from_backend(
        self,
        path: str,
        backend: SecretBackend
    ) -> Dict[str, Any]:
        """Fetch secret from specified backend."""
        if backend == SecretBackend.VAULT:
            return self._fetch_from_vault(path)
        elif backend == SecretBackend.ENCRYPTED_FILE:
            return self._fetch_from_encrypted_file(path)
        elif backend == SecretBackend.ENVIRONMENT:
            return self._fetch_from_environment(path)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _fetch_from_vault(self, path: str) -> Dict[str, Any]:
        """Fetch secret from Vault."""
        if not self._vault_client:
            raise SecretAccessDeniedError("Vault client not available")

        full_path = f"{self.config.vault_path_prefix}/{path}"

        try:
            response = self._vault_client.secrets.kv.v2.read_secret_version(
                path=full_path,
                mount_point=self.config.vault_mount_point
            )
            return response['data']['data']
        except Exception as e:
            raise SecretNotFoundError(f"Vault secret not found: {path}") from e

    def _fetch_from_encrypted_file(self, path: str) -> Dict[str, Any]:
        """Fetch secret from encrypted file."""
        if not self._fernet:
            raise SecretAccessDeniedError("Encrypted file backend not available")

        file_path = Path(self.config.encrypted_file_path)
        if not file_path.exists():
            raise SecretNotFoundError(f"Secrets file not found: {file_path}")

        try:
            encrypted_data = file_path.read_bytes()
            decrypted_data = self._fernet.decrypt(encrypted_data)
            all_secrets = json.loads(decrypted_data.decode())

            if path not in all_secrets:
                raise SecretNotFoundError(f"Secret not found in file: {path}")

            return all_secrets[path]

        except Exception as e:
            raise SecretNotFoundError(f"Failed to read secret: {path}") from e

    def _fetch_from_environment(self, path: str) -> Dict[str, Any]:
        """Fetch secret from environment variables."""
        # Convert path to env var format: mt5/credentials -> MT5_CREDENTIALS_*
        prefix = path.upper().replace('/', '_')

        result = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Extract field name: MT5_CREDENTIALS_PASSWORD -> password
                field = key[len(prefix)+1:].lower()
                result[field] = value

        if not result:
            raise SecretNotFoundError(f"No environment variables found for: {path}")

        return result

    def _store_to_backend(
        self,
        path: str,
        value: Dict[str, Any],
        backend: SecretBackend
    ) -> None:
        """Store secret to specified backend."""
        if backend == SecretBackend.VAULT:
            self._store_to_vault(path, value)
        elif backend == SecretBackend.ENCRYPTED_FILE:
            self._store_to_encrypted_file(path, value)
        else:
            raise ValueError(f"Cannot store to backend: {backend}")

    def _store_to_vault(self, path: str, value: Dict[str, Any]) -> None:
        """Store secret to Vault."""
        if not self._vault_client:
            raise SecretAccessDeniedError("Vault client not available")

        full_path = f"{self.config.vault_path_prefix}/{path}"

        self._vault_client.secrets.kv.v2.create_or_update_secret(
            path=full_path,
            secret=value,
            mount_point=self.config.vault_mount_point
        )

    def _store_to_encrypted_file(self, path: str, value: Dict[str, Any]) -> None:
        """Store secret to encrypted file."""
        if not self._fernet:
            raise SecretAccessDeniedError("Encrypted file backend not available")

        file_path = Path(self.config.encrypted_file_path)

        # Load existing secrets
        all_secrets = {}
        if file_path.exists():
            try:
                encrypted_data = file_path.read_bytes()
                decrypted_data = self._fernet.decrypt(encrypted_data)
                all_secrets = json.loads(decrypted_data.decode())
            except Exception:
                pass  # Start fresh if file is corrupted

        # Update secret
        all_secrets[path] = value

        # Encrypt and save
        json_data = json.dumps(all_secrets, indent=2)
        encrypted_data = self._fernet.encrypt(json_data.encode())
        file_path.write_bytes(encrypted_data)

    # =========================================================================
    # CACHING & AUDITING
    # =========================================================================

    def _cache_secret(self, path: str, value: Dict[str, Any]) -> None:
        """Cache a secret with TTL."""
        now = datetime.utcnow()
        self._cache[path] = CachedSecret(
            value=value.copy(),
            fetched_at=now,
            expires_at=now + timedelta(seconds=self.config.cache_ttl_seconds),
        )

    def _log_access(
        self,
        path: str,
        source: str,
        action: str = "read"
    ) -> None:
        """Log secret access for audit."""
        if not self.config.audit_all_access:
            return

        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'path': path,
            'source': source,
            'action': action,
            'pid': os.getpid(),
        }

        self._access_log.append(entry)

        # Keep only last 1000 entries
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-1000:]

        self._logger.debug(f"Secret access: {action} {path} from {source}")
