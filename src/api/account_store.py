"""User account store — registration, login, sessions, password reset, roles.

Security-first account layer for MIA Markets. Mirrors the house pattern of
``KeyStore`` / ``UserTierManager`` (raw ``sqlite3`` + WAL + ``threading.RLock``
+ ``SCHEMA_VERSION`` migrations) so there is ONE storage idiom in the codebase
and accounts live next to the existing tier/key data (single source of truth).

ALL cryptography is delegated to proven libraries — there is no home-made
hashing or token crypto here:

* **argon2-cffi** (``PasswordHasher``) hashes passwords with Argon2id, the
  PHC password-hashing competition winner. Hashes carry their own parameters,
  so ``check_needs_rehash`` transparently upgrades them over time.
* Session and reset tokens are opaque ``secrets.token_urlsafe(32)`` values;
  only their SHA-256 hash is stored, so a database leak never exposes a usable
  token (same discipline as ``KeyStore`` API keys).

Roles: ``'user'`` (default) and ``'owner'``. The owner account is seeded from
the environment at first boot (see :meth:`seed_owner`) — its password is hashed
at creation time and never stored or hard-coded in clear text.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerifyMismatchError

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION + POLICY
# =============================================================================

# Deliberately liberal but safe. The goal is to reject obvious garbage at the
# store boundary (defence in depth) without re-implementing full RFC 5322.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_USERNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.\-]{1,30})[A-Za-z0-9]$")

MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 256  # argon2 has no hard cap, but bound memory/DoS

# Roles. ``tester`` is the closed-beta role — a normal, non-owner account seeded
# by scripts/seed_testers.py and distinct from ``owner`` (which keeps unlimited
# operator access). It carries the same access as ``user`` today; the label lets
# us tell beta testers apart from organic sign-ups and target them later.
VALID_ROLES = ("user", "tester", "owner")
VALID_CONSENT_DOCS = ("terms", "privacy")

# Default session lifetime (30 days) and reset-token lifetime (1 hour).
DEFAULT_SESSION_TTL_S = 30 * 86400.0
DEFAULT_RESET_TTL_S = 3600.0


class AccountError(ValueError):
    """Base class for account-store validation/conflict errors.

    Distinct from generic ``ValueError`` so routes can translate it to a
    deterministic 4xx without leaking internals. ``code`` is a stable,
    machine-readable string for the API layer.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# =============================================================================
# ACCOUNT STORE
# =============================================================================


class AccountStore:
    """Thread-safe account store with SQLite WAL persistence."""

    SCHEMA_VERSION = 3

    def __init__(self, db_path: str = "./data/accounts.db"):
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        # One shared PasswordHasher with library defaults (Argon2id). Reusable
        # and thread-safe for hash()/verify().
        self._hasher = PasswordHasher()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("AccountStore initialised at %s", self._db_path)

    # --------------------------------------------------------------------- #
    # SQLite helpers
    # --------------------------------------------------------------------- #
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path), timeout=30.0, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_database(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_version "
                    "(version INTEGER PRIMARY KEY)"
                )
                cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
                row = cur.fetchone()
                current = row["version"] if row else 0
                if current < self.SCHEMA_VERSION:
                    self._migrate(conn, current)
            finally:
                conn.close()

    def _migrate(self, conn: sqlite3.Connection, from_v: int) -> None:
        if from_v < 1:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    username       TEXT    NOT NULL,
                    username_lower TEXT    NOT NULL UNIQUE,
                    email          TEXT    NOT NULL,
                    email_lower    TEXT    NOT NULL UNIQUE,
                    password_hash  TEXT    NOT NULL,
                    role           TEXT    NOT NULL DEFAULT 'user',
                    age_confirmed  INTEGER NOT NULL DEFAULT 0,
                    is_active      INTEGER NOT NULL DEFAULT 1,
                    created_at     TEXT    NOT NULL
                );
                CREATE TABLE IF NOT EXISTS account_consents (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id  INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                    doc         TEXT    NOT NULL,
                    version     TEXT    NOT NULL,
                    accepted_at TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_consents_account
                    ON account_consents(account_id);
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT    PRIMARY KEY,
                    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                    created_at REAL    NOT NULL,
                    expires_at REAL    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_account
                    ON sessions(account_id);
                CREATE TABLE IF NOT EXISTS password_resets (
                    token_hash TEXT    PRIMARY KEY,
                    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                    created_at REAL    NOT NULL,
                    expires_at REAL    NOT NULL,
                    used_at    REAL
                );
            """)
        if from_v < 2:
            # Payments mission ② — subscription state keyed to the account.
            # ONE row per account (account_id PK) holding the Stripe linkage and
            # the RESOLVED subscription state. NO card data EVER lives here — only
            # opaque Stripe IDs and the status Stripe reports. ``processed_webhooks``
            # makes webhook handling idempotent (each Stripe event id applied once).
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    account_id             INTEGER PRIMARY KEY
                        REFERENCES accounts(id) ON DELETE CASCADE,
                    stripe_customer_id     TEXT,
                    stripe_subscription_id TEXT,
                    status                 TEXT,
                    price_id               TEXT,
                    current_period_end     REAL,
                    cancel_at_period_end   INTEGER NOT NULL DEFAULT 0,
                    trial_end              REAL,
                    updated_at             TEXT    NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_subscriptions_customer
                    ON subscriptions(stripe_customer_id)
                    WHERE stripe_customer_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS processed_webhooks (
                    event_id     TEXT PRIMARY KEY,
                    event_type   TEXT,
                    processed_at REAL NOT NULL
                );
            """)
        if from_v < 3:
            # Subscription-gate mission ③ — per-account daily chat counter that
            # backs the freemium quota (free tier = N messages/day). ONE row per
            # (account, UTC day); enforcement lives in ``entitlements``. No PII,
            # just a count — old days can be pruned at will.
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chat_usage (
                    account_id INTEGER NOT NULL
                        REFERENCES accounts(id) ON DELETE CASCADE,
                    day        TEXT    NOT NULL,
                    count      INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (account_id, day)
                );
            """)
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (self.SCHEMA_VERSION,),
        )

    # --------------------------------------------------------------------- #
    # Crypto helpers (delegated to proven libraries)
    # --------------------------------------------------------------------- #
    @staticmethod
    def _hash_token(raw_token: str) -> str:
        """SHA-256 of an opaque high-entropy token — same as KeyStore keys.

        SHA-256 (not Argon2) is correct here: the token is 256 bits of CSPRNG
        output, so it is not brute-forceable and needs no slow KDF — only a
        fast one-way map so the raw value never sits in the DB.
        """
        return hashlib.sha256(raw_token.encode()).hexdigest()

    @staticmethod
    def _new_token() -> str:
        return secrets.token_urlsafe(32)

    def _hash_password(self, password: str) -> str:
        return self._hasher.hash(password)

    # --------------------------------------------------------------------- #
    # Validation
    # --------------------------------------------------------------------- #
    @staticmethod
    def _normalise_email(email: str) -> str:
        return email.strip().lower()

    def _validate_new_credentials(
        self, username: str, email: str, password: str
    ) -> None:
        if not username or not _USERNAME_RE.match(username):
            raise AccountError(
                "invalid_username",
                "Le nom d'utilisateur doit faire 3 à 32 caractères "
                "alphanumériques (., _, - autorisés à l'intérieur).",
            )
        if not email or not _EMAIL_RE.match(email.strip()):
            raise AccountError("invalid_email", "Adresse e-mail invalide.")
        if not password or len(password) < MIN_PASSWORD_LENGTH:
            raise AccountError(
                "weak_password",
                f"Le mot de passe doit faire au moins {MIN_PASSWORD_LENGTH} "
                "caractères.",
            )
        if len(password) > MAX_PASSWORD_LENGTH:
            raise AccountError(
                "password_too_long",
                f"Le mot de passe ne peut pas dépasser {MAX_PASSWORD_LENGTH} "
                "caractères.",
            )

    # --------------------------------------------------------------------- #
    # Account creation
    # --------------------------------------------------------------------- #
    def create_account(
        self,
        username: str,
        email: str,
        password: str,
        *,
        age_confirmed: bool,
        consents: Sequence[Tuple[str, str]],
        role: str = "user",
    ) -> Dict[str, Any]:
        """Create an account.

        ``consents`` is a sequence of ``(doc, version)`` pairs — both
        ``'terms'`` and ``'privacy'`` are REQUIRED (Loi 25 / RGPD consent:
        version + timestamp recorded per account). ``age_confirmed`` MUST be
        True (18+ self-declaration). Raises :class:`AccountError` on any
        violation or uniqueness conflict; nothing is written in that case.
        """
        username = username.strip()
        email = email.strip()
        self._validate_new_credentials(username, email, password)

        if role not in VALID_ROLES:
            raise AccountError("invalid_role", f"Rôle inconnu : {role!r}.")

        if not age_confirmed:
            raise AccountError(
                "age_not_confirmed",
                "Vous devez déclarer avoir 18 ans ou plus pour créer un compte.",
            )

        docs = {d for d, _ in consents}
        missing = [d for d in VALID_CONSENT_DOCS if d not in docs]
        if missing:
            raise AccountError(
                "consent_required",
                "Vous devez accepter les Conditions d'utilisation et la "
                "Politique de confidentialité pour créer un compte.",
            )
        for doc, version in consents:
            if doc not in VALID_CONSENT_DOCS:
                raise AccountError("invalid_consent_doc", f"Document inconnu : {doc!r}.")
            if not version or not str(version).strip():
                raise AccountError(
                    "invalid_consent_version",
                    "La version du document de consentement est manquante.",
                )

        password_hash = self._hash_password(password)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        email_lower = self._normalise_email(email)
        username_lower = username.lower()

        with self._lock:
            conn = self._get_connection()
            try:
                # Pre-check for clearer error codes than a raw IntegrityError.
                cur = conn.execute(
                    "SELECT 1 FROM accounts WHERE username_lower = ?",
                    (username_lower,),
                )
                if cur.fetchone() is not None:
                    raise AccountError(
                        "username_taken", "Ce nom d'utilisateur est déjà pris."
                    )
                cur = conn.execute(
                    "SELECT 1 FROM accounts WHERE email_lower = ?", (email_lower,)
                )
                if cur.fetchone() is not None:
                    raise AccountError(
                        "email_taken", "Cette adresse e-mail est déjà utilisée."
                    )

                conn.execute("BEGIN")
                cur = conn.execute(
                    "INSERT INTO accounts "
                    "(username, username_lower, email, email_lower, "
                    " password_hash, role, age_confirmed, is_active, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)",
                    (
                        username,
                        username_lower,
                        email,
                        email_lower,
                        password_hash,
                        role,
                        1 if age_confirmed else 0,
                        now_iso,
                    ),
                )
                account_id = cur.lastrowid
                for doc, version in consents:
                    conn.execute(
                        "INSERT INTO account_consents "
                        "(account_id, doc, version, accepted_at) "
                        "VALUES (?, ?, ?, ?)",
                        (account_id, doc, str(version), now_iso),
                    )
                conn.execute("COMMIT")
            except AccountError:
                conn.rollback()
                raise
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                # Lost a race on the UNIQUE constraint.
                raise AccountError(
                    "account_conflict",
                    "Ce compte existe déjà (nom d'utilisateur ou e-mail).",
                ) from exc
            finally:
                conn.close()

        logger.info("account created id=%s role=%s", account_id, role)
        return self._public_account(account_id)

    # --------------------------------------------------------------------- #
    # Lookup
    # --------------------------------------------------------------------- #
    def _row_to_public(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "role": row["role"],
            "age_confirmed": bool(row["age_confirmed"]),
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
        }

    def _public_account(self, account_id: int) -> Dict[str, Any]:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT * FROM accounts WHERE id = ?", (account_id,)
                )
                row = cur.fetchone()
            finally:
                conn.close()
        if row is None:  # pragma: no cover - internal invariant
            raise AccountError("not_found", "Compte introuvable.")
        return self._row_to_public(row)

    def get_account(self, account_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT * FROM accounts WHERE id = ?", (account_id,)
                )
                row = cur.fetchone()
            finally:
                conn.close()
        return self._row_to_public(row) if row else None

    def get_consents(self, account_id: int) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT doc, version, accepted_at FROM account_consents "
                    "WHERE account_id = ? ORDER BY id",
                    (account_id,),
                )
                return [
                    {
                        "doc": r["doc"],
                        "version": r["version"],
                        "accepted_at": r["accepted_at"],
                    }
                    for r in cur.fetchall()
                ]
            finally:
                conn.close()

    def get_account_by_identifier(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Return the public account for a username OR email, or None.

        A password-less lookup (unlike :meth:`verify_credentials`) used by admin
        tooling — e.g. the tester seed script checking idempotently whether an
        account already exists. Never exposes the password hash.
        """
        if not identifier:
            return None
        ident = identifier.strip().lower()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT * FROM accounts "
                    "WHERE username_lower = ? OR email_lower = ?",
                    (ident, ident),
                )
                row = cur.fetchone()
            finally:
                conn.close()
        return self._row_to_public(row) if row else None

    # --------------------------------------------------------------------- #
    # Credential verification (login by username OR email)
    # --------------------------------------------------------------------- #
    def verify_credentials(
        self, identifier: str, password: str
    ) -> Optional[Dict[str, Any]]:
        """Verify a login by username OR email + password.

        Returns the public account dict on success, ``None`` on any failure
        (unknown identifier, wrong password, inactive account). The argon2
        verification runs even when the identifier is unknown is approximated
        by a constant-ish path — we still do a dummy hash compare to reduce
        username-enumeration timing signals.
        """
        if not identifier or not password:
            return None
        ident = identifier.strip().lower()

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT * FROM accounts "
                    "WHERE username_lower = ? OR email_lower = ?",
                    (ident, ident),
                )
                row = cur.fetchone()
            finally:
                conn.close()

        if row is None or not row["is_active"]:
            # Mitigate user-enumeration timing: spend ~one verify either way.
            try:
                self._hasher.verify(
                    "$argon2id$v=19$m=65536,t=3,p=4$"
                    "c29tZXNhbHRzb21lc2FsdA$"
                    "0000000000000000000000000000000000000000000",
                    password,
                )
            except Exception:
                pass
            return None

        try:
            self._hasher.verify(row["password_hash"], password)
        except (VerifyMismatchError, InvalidHash):
            return None
        except Exception:  # pragma: no cover - defensive
            logger.exception("argon2 verify error for account id=%s", row["id"])
            return None

        # Transparent parameter upgrade if the stored hash is now outdated.
        try:
            if self._hasher.check_needs_rehash(row["password_hash"]):
                self._set_password_hash(row["id"], self._hash_password(password))
        except Exception:  # pragma: no cover - non-fatal
            logger.warning("rehash check failed for account id=%s", row["id"])

        return self._row_to_public(row)

    # --------------------------------------------------------------------- #
    # Sessions (opaque revocable tokens, hashed at rest)
    # --------------------------------------------------------------------- #
    def create_session(
        self, account_id: int, *, ttl_seconds: float = DEFAULT_SESSION_TTL_S
    ) -> str:
        """Create a session, returning the RAW token (shown once, set in cookie)."""
        raw = self._new_token()
        token_hash = self._hash_token(raw)
        now = time.time()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT INTO sessions "
                    "(token_hash, account_id, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?)",
                    (token_hash, account_id, now, now + ttl_seconds),
                )
            finally:
                conn.close()
        return raw

    def resolve_session(self, raw_token: str) -> Optional[Dict[str, Any]]:
        """Return the public account for a valid, unexpired session token."""
        if not raw_token:
            return None
        token_hash = self._hash_token(raw_token)
        now = time.time()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT a.* FROM sessions s "
                    "JOIN accounts a ON a.id = s.account_id "
                    "WHERE s.token_hash = ? AND s.expires_at > ? "
                    "AND a.is_active = 1",
                    (token_hash, now),
                )
                row = cur.fetchone()
            finally:
                conn.close()
        return self._row_to_public(row) if row else None

    def delete_session(self, raw_token: str) -> bool:
        if not raw_token:
            return False
        token_hash = self._hash_token(raw_token)
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "DELETE FROM sessions WHERE token_hash = ?", (token_hash,)
                )
                return cur.rowcount > 0
            finally:
                conn.close()

    def purge_expired_sessions(self) -> int:
        now = time.time()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "DELETE FROM sessions WHERE expires_at <= ?", (now,)
                )
                return cur.rowcount
            finally:
                conn.close()

    # --------------------------------------------------------------------- #
    # Password reset (single-use opaque tokens, hashed at rest)
    # --------------------------------------------------------------------- #
    def create_reset_token(
        self, identifier: str, *, ttl_seconds: float = DEFAULT_RESET_TTL_S
    ) -> Optional[str]:
        """Create a single-use reset token for username/email, or None.

        Returns ``None`` when the identifier matches no active account. The
        ROUTE must NOT leak that difference to the client (always answer the
        same) — but the token is only emailed when an account actually exists.
        """
        if not identifier:
            return None
        ident = identifier.strip().lower()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT id FROM accounts "
                    "WHERE (username_lower = ? OR email_lower = ?) AND is_active = 1",
                    (ident, ident),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                account_id = row["id"]
                raw = self._new_token()
                now = time.time()
                conn.execute(
                    "INSERT INTO password_resets "
                    "(token_hash, account_id, created_at, expires_at, used_at) "
                    "VALUES (?, ?, ?, ?, NULL)",
                    (self._hash_token(raw), account_id, now, now + ttl_seconds),
                )
                return raw
            finally:
                conn.close()

    def consume_reset_token(self, raw_token: str, new_password: str) -> bool:
        """Atomically validate + burn a reset token and set the new password."""
        if not raw_token:
            return False
        if not new_password or len(new_password) < MIN_PASSWORD_LENGTH:
            raise AccountError(
                "weak_password",
                f"Le mot de passe doit faire au moins {MIN_PASSWORD_LENGTH} "
                "caractères.",
            )
        if len(new_password) > MAX_PASSWORD_LENGTH:
            raise AccountError(
                "password_too_long",
                f"Le mot de passe ne peut pas dépasser {MAX_PASSWORD_LENGTH} "
                "caractères.",
            )
        token_hash = self._hash_token(raw_token)
        now = time.time()
        new_hash = self._hash_password(new_password)
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN")
                cur = conn.execute(
                    "SELECT account_id, expires_at, used_at "
                    "FROM password_resets WHERE token_hash = ?",
                    (token_hash,),
                )
                row = cur.fetchone()
                if (
                    row is None
                    or row["used_at"] is not None
                    or row["expires_at"] <= now
                ):
                    conn.execute("ROLLBACK")
                    return False
                account_id = row["account_id"]
                conn.execute(
                    "UPDATE accounts SET password_hash = ? WHERE id = ?",
                    (new_hash, account_id),
                )
                conn.execute(
                    "UPDATE password_resets SET used_at = ? WHERE token_hash = ?",
                    (now, token_hash),
                )
                # Revoke all existing sessions on password change.
                conn.execute(
                    "DELETE FROM sessions WHERE account_id = ?", (account_id,)
                )
                conn.execute("COMMIT")
                return True
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def update_email(self, account_id: int, new_email: str) -> Dict[str, Any]:
        """Update an account's email. Raises AccountError on invalid/conflict."""
        new_email = new_email.strip()
        if not new_email or not _EMAIL_RE.match(new_email):
            raise AccountError("invalid_email", "Adresse e-mail invalide.")
        email_lower = self._normalise_email(new_email)
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT id FROM accounts WHERE email_lower = ? AND id != ?",
                    (email_lower, account_id),
                )
                if cur.fetchone() is not None:
                    raise AccountError(
                        "email_taken", "Cette adresse e-mail est déjà utilisée."
                    )
                conn.execute(
                    "UPDATE accounts SET email = ?, email_lower = ? WHERE id = ?",
                    (new_email, email_lower, account_id),
                )
            finally:
                conn.close()
        return self._public_account(account_id)

    def _set_password_hash(self, account_id: int, password_hash: str) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "UPDATE accounts SET password_hash = ? WHERE id = ?",
                    (password_hash, account_id),
                )
            finally:
                conn.close()

    def set_password(self, account_id: int, new_password: str) -> None:
        """Admin-set an account's password (hashed) and revoke its sessions.

        Used by the tester seed script's ``--reset`` path to rotate a tester's
        password out-of-band (e.g. when the one-time printout was lost). Same
        Argon2id hashing as everywhere else — the plaintext is never persisted.
        Revoking existing sessions forces re-login with the new credential.
        """
        if not new_password or len(new_password) < MIN_PASSWORD_LENGTH:
            raise AccountError(
                "weak_password",
                f"Le mot de passe doit faire au moins {MIN_PASSWORD_LENGTH} "
                "caractères.",
            )
        if len(new_password) > MAX_PASSWORD_LENGTH:
            raise AccountError(
                "password_too_long",
                f"Le mot de passe ne peut pas dépasser {MAX_PASSWORD_LENGTH} "
                "caractères.",
            )
        new_hash = self._hash_password(new_password)
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN")
                conn.execute(
                    "UPDATE accounts SET password_hash = ? WHERE id = ?",
                    (new_hash, account_id),
                )
                conn.execute(
                    "DELETE FROM sessions WHERE account_id = ?", (account_id,)
                )
                conn.execute("COMMIT")
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # --------------------------------------------------------------------- #
    # Subscriptions / Stripe linkage (payments mission ②)
    #
    # NO card data is ever stored here. We persist only the opaque Stripe
    # customer/subscription IDs and the subscription STATE Stripe reports, so
    # the paywall (subscription_gate.account_has_access) can answer offline.
    # --------------------------------------------------------------------- #
    @staticmethod
    def _row_to_subscription(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        return {
            "account_id": row["account_id"],
            "stripe_customer_id": row["stripe_customer_id"],
            "stripe_subscription_id": row["stripe_subscription_id"],
            "status": row["status"],
            "price_id": row["price_id"],
            "current_period_end": row["current_period_end"],
            "cancel_at_period_end": bool(row["cancel_at_period_end"]),
            "trial_end": row["trial_end"],
            "updated_at": row["updated_at"],
        }

    def get_subscription(self, account_id: int) -> Optional[Dict[str, Any]]:
        """Return the subscription row for an account, or None if never linked."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT * FROM subscriptions WHERE account_id = ?",
                    (account_id,),
                )
                return self._row_to_subscription(cur.fetchone())
            finally:
                conn.close()

    def link_stripe_customer(self, account_id: int, stripe_customer_id: str) -> None:
        """Bind a Stripe customer id to an account (created at checkout time).

        Idempotent: upserts the ``subscriptions`` row so a customer is linked
        before any subscription exists. Never overwrites an existing customer id
        with a different one (the first binding wins — a guard against a stray
        second customer for the same account).
        """
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT INTO subscriptions "
                    "(account_id, stripe_customer_id, cancel_at_period_end, updated_at) "
                    "VALUES (?, ?, 0, ?) "
                    "ON CONFLICT(account_id) DO UPDATE SET "
                    "  stripe_customer_id = COALESCE(subscriptions.stripe_customer_id, excluded.stripe_customer_id), "
                    "  updated_at = excluded.updated_at",
                    (account_id, stripe_customer_id, now_iso),
                )
            finally:
                conn.close()

    def get_account_by_stripe_customer(
        self, stripe_customer_id: str
    ) -> Optional[Dict[str, Any]]:
        """Resolve the account behind a Stripe customer id (webhook join key)."""
        if not stripe_customer_id:
            return None
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT a.* FROM subscriptions s "
                    "JOIN accounts a ON a.id = s.account_id "
                    "WHERE s.stripe_customer_id = ?",
                    (stripe_customer_id,),
                )
                row = cur.fetchone()
            finally:
                conn.close()
        return self._row_to_public(row) if row else None

    def upsert_subscription(
        self,
        account_id: int,
        *,
        stripe_customer_id: Optional[str] = None,
        stripe_subscription_id: Optional[str] = None,
        status: Optional[str] = None,
        price_id: Optional[str] = None,
        current_period_end: Optional[float] = None,
        cancel_at_period_end: Optional[bool] = None,
        trial_end: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create/update the subscription row from a (verified) Stripe event.

        Only non-None fields are written, so a partial event (e.g. an invoice
        carrying just a status change) never wipes the subscription id or period
        already on record. Returns the resulting subscription dict.
        """
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        fields = {
            "stripe_customer_id": stripe_customer_id,
            "stripe_subscription_id": stripe_subscription_id,
            "status": status,
            "price_id": price_id,
            "current_period_end": current_period_end,
            "cancel_at_period_end": (
                None if cancel_at_period_end is None else int(cancel_at_period_end)
            ),
            "trial_end": trial_end,
        }
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN")
                cur = conn.execute(
                    "SELECT account_id FROM subscriptions WHERE account_id = ?",
                    (account_id,),
                )
                exists = cur.fetchone() is not None
                if not exists:
                    conn.execute(
                        "INSERT INTO subscriptions (account_id, cancel_at_period_end, updated_at) "
                        "VALUES (?, 0, ?)",
                        (account_id, now_iso),
                    )
                set_cols = [(k, v) for k, v in fields.items() if v is not None]
                if set_cols:
                    assignments = ", ".join(f"{k} = ?" for k, _ in set_cols)
                    params = [v for _, v in set_cols]
                    params.append(now_iso)
                    params.append(account_id)
                    conn.execute(
                        f"UPDATE subscriptions SET {assignments}, updated_at = ? "
                        "WHERE account_id = ?",
                        params,
                    )
                else:
                    conn.execute(
                        "UPDATE subscriptions SET updated_at = ? WHERE account_id = ?",
                        (now_iso, account_id),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        sub = self.get_subscription(account_id)
        assert sub is not None  # just upserted
        return sub

    def mark_webhook_processed(self, event_id: str, event_type: str = "") -> bool:
        """Record a Stripe event id; return True if NEW, False if already seen.

        The PRIMARY KEY makes this atomic — a duplicate delivery of the same
        event id is a no-op insert, so callers gate their side effects on the
        returned bool to stay idempotent under Stripe's at-least-once delivery.
        """
        if not event_id:
            # No id to dedup on — treat as new but never persist an empty key.
            return True
        now = time.time()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO processed_webhooks "
                    "(event_id, event_type, processed_at) VALUES (?, ?, ?)",
                    (event_id, event_type, now),
                )
                return cur.rowcount > 0
            finally:
                conn.close()

    # --------------------------------------------------------------------- #
    # Freemium chat quota (per-account, per-UTC-day counter)
    #
    # Backs the free-tier "N messages/day" limit (subscription-gate mission ③).
    # The policy/limit lives in ``entitlements``; the store only counts.
    # --------------------------------------------------------------------- #
    def get_chat_usage(self, account_id: int, day: str) -> int:
        """Return today's message count for an account (0 if none yet)."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT count FROM chat_usage WHERE account_id = ? AND day = ?",
                    (account_id, day),
                )
                row = cur.fetchone()
            finally:
                conn.close()
        return int(row["count"]) if row else 0

    def increment_chat_usage(self, account_id: int, day: str) -> int:
        """Atomically add one to the day's counter and return the new total.

        The ``ON CONFLICT`` upsert + RLock make this safe under concurrency, so
        two parallel chat turns can never both slip past the same quota boundary.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT INTO chat_usage (account_id, day, count) VALUES (?, ?, 1) "
                    "ON CONFLICT(account_id, day) DO UPDATE SET count = count + 1",
                    (account_id, day),
                )
                cur = conn.execute(
                    "SELECT count FROM chat_usage WHERE account_id = ? AND day = ?",
                    (account_id, day),
                )
                row = cur.fetchone()
            finally:
                conn.close()
        return int(row["count"]) if row else 1

    # --------------------------------------------------------------------- #
    # Owner seeding (idempotent, from environment at first boot)
    # --------------------------------------------------------------------- #
    def seed_owner(
        self, username: str, email: str, password: str
    ) -> Dict[str, Any]:
        """Idempotently ensure an owner account exists, seeded from env.

        First boot: creates the account with role ``owner`` (password hashed at
        creation). Subsequent boots: if an account with this username/email
        already exists it is promoted to ``owner`` (and reactivated) WITHOUT
        touching the password — so rotating ``OWNER_PASSWORD`` in the env does
        not silently reset a password the operator may have already changed.

        Raises :class:`AccountError` on invalid env values (caller logs +
        continues so a bad env var never aborts startup).
        """
        username = username.strip()
        email = email.strip()
        self._validate_new_credentials(username, email, password)
        email_lower = self._normalise_email(email)
        username_lower = username.lower()

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT id FROM accounts "
                    "WHERE username_lower = ? OR email_lower = ?",
                    (username_lower, email_lower),
                )
                row = cur.fetchone()
                if row is not None:
                    conn.execute(
                        "UPDATE accounts SET role = 'owner', is_active = 1 "
                        "WHERE id = ?",
                        (row["id"],),
                    )
                    logger.info(
                        "owner account already present (id=%s) — ensured role=owner",
                        row["id"],
                    )
                    return self._public_account(row["id"])

                now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
                cur = conn.execute(
                    "INSERT INTO accounts "
                    "(username, username_lower, email, email_lower, "
                    " password_hash, role, age_confirmed, is_active, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 'owner', 1, 1, ?)",
                    (
                        username,
                        username_lower,
                        email,
                        email_lower,
                        self._hash_password(password),
                        now_iso,
                    ),
                )
                owner_id = cur.lastrowid
                # Owner is the operator — record implicit consent to the current
                # legal docs so the consent table is never empty for them.
                logger.info("owner account seeded id=%s", owner_id)
            finally:
                conn.close()
        return self._public_account(owner_id)
