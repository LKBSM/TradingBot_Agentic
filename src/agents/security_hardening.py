# =============================================================================
# SECURITY HARDENING MODULE - Critical Security Infrastructure
# =============================================================================
"""
Centralized Security Module for Trading Bot

This module provides critical security infrastructure including:
- Input validation and sanitization
- Rate limiting for API protection
- Secure token generation and verification
- Audit trail with cryptographic integrity
- Thread-safe operations

All security-critical operations should use this module.

Version: 1.0.0
Author: TradingBot Security Team
License: Proprietary - Commercial Use
"""

import hashlib
import hmac
import secrets
import re
import os
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from collections import OrderedDict, deque
from functools import wraps
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum admin key length
MIN_ADMIN_KEY_LENGTH = 32

# Default rate limits
DEFAULT_RATE_LIMIT_REQUESTS = 100
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60

# Token configuration
TOKEN_BYTES = 32  # 256 bits
TOKEN_TTL_SECONDS = 300  # 5 minutes

# Path validation
ALLOWED_LOG_ROOTS = [
    "./logs",
    "./data",
    "/var/log/tradingbot",
    "/tmp/tradingbot"
]


# =============================================================================
# EXCEPTIONS
# =============================================================================

class SecurityError(Exception):
    """Base exception for security-related errors."""
    pass


class ValidationError(SecurityError):
    """Input validation failed."""
    pass


class RateLimitError(SecurityError):
    """Rate limit exceeded."""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed."""
    pass


class IntegrityError(SecurityError):
    """Data integrity check failed."""
    pass


# =============================================================================
# INPUT VALIDATION
# =============================================================================

class InputValidator:
    """
    Comprehensive input validation for trading operations.

    All external inputs should be validated through this class
    before being used in trading logic.
    """

    # Valid trading symbols pattern
    SYMBOL_PATTERN = re.compile(r'^[A-Z]{3,6}(/[A-Z]{3,6})?$')

    # Valid actions
    VALID_ACTIONS = frozenset([
        'BUY', 'SELL', 'HOLD',
        'OPEN_LONG', 'CLOSE_LONG',
        'OPEN_SHORT', 'CLOSE_SHORT'
    ])

    # Valid session ID pattern (alphanumeric + underscore + hyphen)
    SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')

    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """
        Validate trading symbol format.

        Args:
            symbol: Trading symbol (e.g., "EURUSD", "XAU/USD")

        Returns:
            Validated symbol (uppercase)

        Raises:
            ValidationError: If symbol is invalid
        """
        if not isinstance(symbol, str):
            raise ValidationError(f"Symbol must be string, got {type(symbol)}")

        symbol = symbol.strip().upper()

        if not symbol:
            raise ValidationError("Symbol cannot be empty")

        if len(symbol) > 20:
            raise ValidationError(f"Symbol too long: {len(symbol)} chars (max 20)")

        if not InputValidator.SYMBOL_PATTERN.match(symbol):
            raise ValidationError(f"Invalid symbol format: {symbol}")

        return symbol

    @staticmethod
    def validate_action(action: str) -> str:
        """
        Validate trading action.

        Args:
            action: Trading action string

        Returns:
            Validated action (uppercase)

        Raises:
            ValidationError: If action is invalid
        """
        if not isinstance(action, str):
            raise ValidationError(f"Action must be string, got {type(action)}")

        action = action.strip().upper()

        if action not in InputValidator.VALID_ACTIONS:
            raise ValidationError(
                f"Invalid action: {action}. "
                f"Valid actions: {', '.join(sorted(InputValidator.VALID_ACTIONS))}"
            )

        return action

    @staticmethod
    def validate_quantity(quantity: float, min_qty: float = 0.0001,
                         max_qty: float = 1_000_000) -> float:
        """
        Validate trade quantity.

        Args:
            quantity: Trade quantity
            min_qty: Minimum allowed quantity
            max_qty: Maximum allowed quantity

        Returns:
            Validated quantity

        Raises:
            ValidationError: If quantity is invalid
        """
        if not isinstance(quantity, (int, float)):
            raise ValidationError(f"Quantity must be numeric, got {type(quantity)}")

        if np.isnan(quantity) or np.isinf(quantity):
            raise ValidationError(f"Quantity cannot be NaN or Inf: {quantity}")

        if quantity <= 0:
            raise ValidationError(f"Quantity must be positive: {quantity}")

        if quantity < min_qty:
            raise ValidationError(f"Quantity {quantity} below minimum {min_qty}")

        if quantity > max_qty:
            raise ValidationError(f"Quantity {quantity} exceeds maximum {max_qty}")

        return float(quantity)

    @staticmethod
    def validate_price(price: float, min_price: float = 0.00001,
                      max_price: float = 1_000_000) -> float:
        """
        Validate price value.

        Args:
            price: Price value
            min_price: Minimum allowed price
            max_price: Maximum allowed price

        Returns:
            Validated price

        Raises:
            ValidationError: If price is invalid
        """
        if not isinstance(price, (int, float)):
            raise ValidationError(f"Price must be numeric, got {type(price)}")

        if np.isnan(price) or np.isinf(price):
            raise ValidationError(f"Price cannot be NaN or Inf: {price}")

        if price <= 0:
            raise ValidationError(f"Price must be positive: {price}")

        if price < min_price:
            raise ValidationError(f"Price {price} below minimum {min_price}")

        if price > max_price:
            raise ValidationError(f"Price {price} exceeds maximum {max_price}")

        return float(price)

    @staticmethod
    def validate_equity(equity: float, allow_zero: bool = False) -> float:
        """
        Validate equity/balance value.

        Args:
            equity: Equity value
            allow_zero: Whether zero equity is allowed

        Returns:
            Validated equity

        Raises:
            ValidationError: If equity is invalid
        """
        if not isinstance(equity, (int, float)):
            raise ValidationError(f"Equity must be numeric, got {type(equity)}")

        if np.isnan(equity) or np.isinf(equity):
            raise ValidationError(f"Equity cannot be NaN or Inf: {equity}")

        if equity < 0:
            raise ValidationError(f"Equity cannot be negative: {equity}")

        if not allow_zero and equity == 0:
            raise ValidationError("Equity cannot be zero (total loss condition)")

        return float(equity)

    @staticmethod
    def validate_percentage(value: float, name: str = "value",
                           min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Validate percentage value.

        Args:
            value: Percentage value (0-1 scale)
            name: Name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated percentage

        Raises:
            ValidationError: If value is invalid
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be numeric, got {type(value)}")

        if np.isnan(value) or np.isinf(value):
            raise ValidationError(f"{name} cannot be NaN or Inf")

        if value < min_val or value > max_val:
            raise ValidationError(
                f"{name} must be between {min_val} and {max_val}, got {value}"
            )

        return float(value)

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        """
        Validate session ID format (prevents path traversal).

        Args:
            session_id: Session identifier

        Returns:
            Validated session ID

        Raises:
            ValidationError: If session ID is invalid
        """
        if not isinstance(session_id, str):
            raise ValidationError(f"Session ID must be string, got {type(session_id)}")

        session_id = session_id.strip()

        if not session_id:
            raise ValidationError("Session ID cannot be empty")

        if not InputValidator.SESSION_ID_PATTERN.match(session_id):
            raise ValidationError(
                f"Invalid session ID format: {session_id}. "
                "Use only alphanumeric, underscore, hyphen (max 64 chars)"
            )

        return session_id

    @staticmethod
    def validate_path(path: str, allowed_roots: List[str] = None) -> Path:
        """
        Validate file path (prevents directory traversal).

        Args:
            path: File path to validate
            allowed_roots: List of allowed root directories

        Returns:
            Validated Path object

        Raises:
            ValidationError: If path is invalid or outside allowed roots
        """
        if allowed_roots is None:
            allowed_roots = ALLOWED_LOG_ROOTS

        if not isinstance(path, str):
            raise ValidationError(f"Path must be string, got {type(path)}")

        # Resolve to absolute path
        try:
            resolved = Path(path).resolve()
        except Exception as e:
            raise ValidationError(f"Invalid path: {path} ({e})")

        # Check against allowed roots
        path_str = str(resolved)
        allowed = False

        for root in allowed_roots:
            try:
                root_resolved = str(Path(root).resolve())
                if path_str.startswith(root_resolved):
                    allowed = True
                    break
            except Exception:
                continue

        if not allowed:
            raise ValidationError(
                f"Path {path} is outside allowed directories: {allowed_roots}"
            )

        return resolved

    @staticmethod
    def validate_numpy_array(arr: np.ndarray, name: str = "array",
                            allow_nan: bool = False,
                            allow_inf: bool = False,
                            min_len: int = 1) -> np.ndarray:
        """
        Validate numpy array for numerical stability.

        Args:
            arr: NumPy array to validate
            name: Name for error messages
            allow_nan: Whether NaN values are allowed
            allow_inf: Whether Inf values are allowed
            min_len: Minimum required length

        Returns:
            Validated array

        Raises:
            ValidationError: If array is invalid
        """
        if not isinstance(arr, np.ndarray):
            raise ValidationError(f"{name} must be numpy array, got {type(arr)}")

        if len(arr) < min_len:
            raise ValidationError(
                f"{name} length {len(arr)} below minimum {min_len}"
            )

        if not allow_nan and np.any(np.isnan(arr)):
            nan_count = np.isnan(arr).sum()
            raise ValidationError(f"{name} contains {nan_count} NaN values")

        if not allow_inf and np.any(np.isinf(arr)):
            inf_count = np.isinf(arr).sum()
            raise ValidationError(f"{name} contains {inf_count} Inf values")

        return arr


# =============================================================================
# RATE LIMITING
# =============================================================================

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests: int = DEFAULT_RATE_LIMIT_REQUESTS
    window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS
    burst_multiplier: float = 1.5  # Allow bursts up to 1.5x normal rate


class RateLimiter:
    """
    Token bucket rate limiter for API protection.

    Prevents abuse by limiting request frequency per client/operation.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """
        Check if request is allowed under rate limit.

        Args:
            key: Identifier for rate limiting (e.g., client_id, operation)

        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        window_start = now - self.config.window_seconds

        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = deque(maxlen=int(
                    self.config.max_requests * self.config.burst_multiplier
                ))

            bucket = self._buckets[key]

            # Remove old timestamps
            while bucket and bucket[0] < window_start:
                bucket.popleft()

            # Check limit
            if len(bucket) >= self.config.max_requests:
                return False

            # Record this request
            bucket.append(now)
            return True

    def acquire(self, key: str) -> None:
        """
        Acquire rate limit slot, raising exception if exceeded.

        Args:
            key: Identifier for rate limiting

        Raises:
            RateLimitError: If rate limit exceeded
        """
        if not self.check(key):
            raise RateLimitError(
                f"Rate limit exceeded for {key}: "
                f"{self.config.max_requests} requests per {self.config.window_seconds}s"
            )

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        window_start = now - self.config.window_seconds

        with self._lock:
            bucket = self._buckets.get(key, deque())

            # Count requests in window
            count = sum(1 for ts in bucket if ts >= window_start)

            return max(0, self.config.max_requests - count)

    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate limit for key or all keys."""
        with self._lock:
            if key:
                self._buckets.pop(key, None)
            else:
                self._buckets.clear()


def rate_limited(limiter: RateLimiter, key_func: Callable[..., str] = None):
    """
    Decorator for rate-limited functions.

    Args:
        limiter: RateLimiter instance
        key_func: Function to extract rate limit key from args
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = func.__name__

            limiter.acquire(key)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# SECURE TOKENS
# =============================================================================

class SecureTokenManager:
    """
    Secure token generation and verification.

    Used for reset confirmations, API authentication, etc.
    """

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize token manager.

        Args:
            secret_key: HMAC secret key (generated if not provided)
        """
        if secret_key:
            if len(secret_key) < MIN_ADMIN_KEY_LENGTH:
                raise SecurityError(
                    f"Secret key must be at least {MIN_ADMIN_KEY_LENGTH} characters"
                )
            self._secret = secret_key.encode()
        else:
            self._secret = secrets.token_bytes(64)

        self._tokens: Dict[str, Tuple[str, datetime]] = {}
        self._failed_attempts: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()

        # Configuration
        self.token_ttl = timedelta(seconds=TOKEN_TTL_SECONDS)
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=5)

    def generate_token(self, purpose: str) -> str:
        """
        Generate a secure token for a specific purpose.

        Args:
            purpose: Token purpose identifier

        Returns:
            Secure token string
        """
        token = secrets.token_hex(TOKEN_BYTES)

        with self._lock:
            self._tokens[purpose] = (token, datetime.now())

        return token

    def verify_token(self, purpose: str, token: str) -> bool:
        """
        Verify a token using constant-time comparison.

        Args:
            purpose: Token purpose identifier
            token: Token to verify

        Returns:
            True if valid, False otherwise
        """
        # Check lockout
        if self._is_locked_out(purpose):
            logger.warning(f"Token verification blocked: {purpose} is locked out")
            return False

        with self._lock:
            stored = self._tokens.get(purpose)

            if not stored:
                self._record_failed_attempt(purpose)
                return False

            stored_token, created_at = stored

            # Check expiry
            if datetime.now() - created_at > self.token_ttl:
                del self._tokens[purpose]
                self._record_failed_attempt(purpose)
                return False

            # Constant-time comparison
            if not hmac.compare_digest(token, stored_token):
                self._record_failed_attempt(purpose)
                return False

            # Success - clear token and failed attempts
            del self._tokens[purpose]
            self._failed_attempts.pop(purpose, None)

            return True

    def invalidate_token(self, purpose: str) -> None:
        """Invalidate a token before its expiry."""
        with self._lock:
            self._tokens.pop(purpose, None)

    def _is_locked_out(self, purpose: str) -> bool:
        """Check if purpose is locked out due to failed attempts."""
        with self._lock:
            attempts = self._failed_attempts.get(purpose, [])

            if len(attempts) < self.max_failed_attempts:
                return False

            # Check if lockout has expired
            cutoff = datetime.now() - self.lockout_duration
            recent_attempts = [a for a in attempts if a > cutoff]

            return len(recent_attempts) >= self.max_failed_attempts

    def _record_failed_attempt(self, purpose: str) -> None:
        """Record a failed verification attempt."""
        now = datetime.now()

        if purpose not in self._failed_attempts:
            self._failed_attempts[purpose] = []

        self._failed_attempts[purpose].append(now)

        # Cleanup old attempts
        cutoff = now - self.lockout_duration
        self._failed_attempts[purpose] = [
            a for a in self._failed_attempts[purpose] if a > cutoff
        ]

        logger.warning(
            f"Failed token verification for {purpose}: "
            f"{len(self._failed_attempts[purpose])} recent attempts"
        )


# =============================================================================
# CRYPTOGRAPHIC INTEGRITY
# =============================================================================

class IntegrityChecker:
    """
    Cryptographic integrity verification for audit records.

    Uses HMAC-SHA256 for tamper detection.
    """

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize integrity checker.

        Args:
            secret_key: HMAC secret key
        """
        if secret_key:
            self._secret = secret_key.encode()
        else:
            # Generate or load from environment
            env_key = os.environ.get("AUDIT_INTEGRITY_KEY")
            if env_key and len(env_key) >= MIN_ADMIN_KEY_LENGTH:
                self._secret = env_key.encode()
            else:
                logger.warning(
                    "AUDIT_INTEGRITY_KEY not set or too short. "
                    "Generating random key (won't persist across restarts)"
                )
                self._secret = secrets.token_bytes(64)

    def compute_checksum(self, data: Dict[str, Any]) -> str:
        """
        Compute HMAC-SHA256 checksum for data.

        Args:
            data: Dictionary to compute checksum for

        Returns:
            Hex-encoded checksum
        """
        import json

        # Canonical JSON representation
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))

        # HMAC-SHA256
        checksum = hmac.new(
            self._secret,
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        return checksum

    def verify_checksum(self, data: Dict[str, Any], checksum: str) -> bool:
        """
        Verify checksum for data.

        Args:
            data: Dictionary to verify
            checksum: Expected checksum

        Returns:
            True if valid, False otherwise
        """
        computed = self.compute_checksum(data)
        return hmac.compare_digest(computed, checksum)

    def sign_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign a record by adding checksum field.

        Args:
            record: Record to sign

        Returns:
            Record with checksum field added
        """
        # Create copy without existing checksum
        data = {k: v for k, v in record.items() if k != 'checksum'}

        # Add checksum
        data['checksum'] = self.compute_checksum(data)

        return data

    def verify_record(self, record: Dict[str, Any]) -> bool:
        """
        Verify a signed record.

        Args:
            record: Record to verify (must contain checksum field)

        Returns:
            True if valid, False otherwise

        Raises:
            IntegrityError: If checksum missing or invalid
        """
        if 'checksum' not in record:
            raise IntegrityError("Record missing checksum field")

        checksum = record['checksum']
        data = {k: v for k, v in record.items() if k != 'checksum'}

        if not self.verify_checksum(data, checksum):
            raise IntegrityError("Record checksum verification failed")

        return True


# =============================================================================
# THREAD-SAFE OPERATIONS
# =============================================================================

class ThreadSafeCounter:
    """Thread-safe counter for metrics."""

    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            return self._value

    def decrement(self, amount: int = 1) -> int:
        with self._lock:
            self._value -= amount
            return self._value

    def get(self) -> int:
        with self._lock:
            return self._value

    def set(self, value: int) -> None:
        with self._lock:
            self._value = value


class ThreadSafeDict:
    """Thread-safe dictionary wrapper."""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> Any:
        with self._lock:
            return self._data.pop(key, None)

    def update(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self._data.update(data)

    def items(self) -> List[Tuple[str, Any]]:
        with self._lock:
            return list(self._data.items())

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())

    def values(self) -> List[Any]:
        with self._lock:
            return list(self._data.values())

    def copy(self) -> Dict[str, Any]:
        with self._lock:
            return self._data.copy()


class ThreadSafeCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self._data: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                self._misses += 1
                return None

            value, timestamp = self._data[key]

            # Check TTL
            if datetime.now() - timestamp > self._ttl:
                del self._data[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            # Remove old entry if exists
            if key in self._data:
                del self._data[key]

            # Add new entry
            self._data[key] = (value, datetime.now())

            # Evict oldest if over capacity
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                'size': len(self._data),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate
            }


# =============================================================================
# EVENT DEDUPLICATION
# =============================================================================

class EventDeduplicator:
    """
    Prevents duplicate event processing.

    Uses event IDs with TTL to detect replays.
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 100000):
        self._seen: OrderedDict = OrderedDict()
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_size = max_size
        self._lock = threading.Lock()
        self._duplicates_blocked = 0

    def is_duplicate(self, event_id: str) -> bool:
        """
        Check if event is a duplicate.

        Args:
            event_id: Unique event identifier

        Returns:
            True if duplicate, False if new
        """
        now = datetime.now()

        with self._lock:
            # Cleanup expired entries
            self._cleanup(now)

            if event_id in self._seen:
                self._duplicates_blocked += 1
                logger.warning(f"Duplicate event blocked: {event_id}")
                return True

            # Record this event
            self._seen[event_id] = now
            return False

    def mark_seen(self, event_id: str) -> None:
        """Mark an event as seen without checking."""
        now = datetime.now()

        with self._lock:
            self._seen[event_id] = now
            self._cleanup(now)

    def _cleanup(self, now: datetime) -> None:
        """Remove expired entries."""
        cutoff = now - self._ttl

        # Remove expired
        expired = [
            eid for eid, ts in self._seen.items()
            if ts < cutoff
        ]

        for eid in expired:
            del self._seen[eid]

        # Remove oldest if over capacity
        while len(self._seen) > self._max_size:
            self._seen.popitem(last=False)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'tracked_events': len(self._seen),
                'duplicates_blocked': self._duplicates_blocked,
                'ttl_seconds': self._ttl.total_seconds()
            }


# =============================================================================
# ADMIN KEY VALIDATION
# =============================================================================

def validate_admin_key(key: Optional[str], env_var: str = "KILL_SWITCH_ADMIN_KEY",
                       min_length: int = MIN_ADMIN_KEY_LENGTH) -> str:
    """
    Validate admin key for security operations.

    Args:
        key: Admin key to validate (or None to use env var)
        env_var: Environment variable name for key
        min_length: Minimum required key length

    Returns:
        Validated admin key

    Raises:
        AuthenticationError: If key is invalid
    """
    if key:
        admin_key = key
    else:
        admin_key = os.environ.get(env_var, "")

    if not admin_key:
        raise AuthenticationError(
            f"Admin key not provided and {env_var} environment variable not set"
        )

    if len(admin_key) < min_length:
        raise AuthenticationError(
            f"Admin key too short: {len(admin_key)} chars (minimum {min_length})"
        )

    # Check entropy (basic check for weak keys)
    unique_chars = len(set(admin_key))
    if unique_chars < 10:
        raise AuthenticationError(
            f"Admin key has low entropy: only {unique_chars} unique characters"
        )

    return admin_key


# =============================================================================
# SANITIZATION
# =============================================================================

def sanitize_log_data(data: Dict[str, Any],
                      sensitive_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Sanitize data for logging (mask sensitive fields).

    Args:
        data: Data to sanitize
        sensitive_keys: Keys to mask (defaults to common sensitive keys)

    Returns:
        Sanitized copy of data
    """
    if sensitive_keys is None:
        sensitive_keys = [
            'password', 'token', 'secret', 'key', 'api_key',
            'access_token', 'refresh_token', 'admin_key',
            'private_key', 'credential'
        ]

    sensitive_keys_lower = [k.lower() for k in sensitive_keys]

    def _sanitize(obj: Any, depth: int = 0) -> Any:
        if depth > 10:  # Prevent infinite recursion
            return "[MAX_DEPTH]"

        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                k_lower = k.lower()
                if any(sk in k_lower for sk in sensitive_keys_lower):
                    result[k] = "[REDACTED]"
                else:
                    result[k] = _sanitize(v, depth + 1)
            return result

        elif isinstance(obj, list):
            return [_sanitize(item, depth + 1) for item in obj]

        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        else:
            return str(obj)

    return _sanitize(data)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    'SecurityError',
    'ValidationError',
    'RateLimitError',
    'AuthenticationError',
    'IntegrityError',

    # Validation
    'InputValidator',

    # Rate Limiting
    'RateLimitConfig',
    'RateLimiter',
    'rate_limited',

    # Tokens
    'SecureTokenManager',

    # Integrity
    'IntegrityChecker',

    # Thread Safety
    'ThreadSafeCounter',
    'ThreadSafeDict',
    'ThreadSafeCache',

    # Deduplication
    'EventDeduplicator',

    # Utilities
    'validate_admin_key',
    'sanitize_log_data',
]
