#!/usr/bin/env python3
"""Seed the closed-beta tester accounts into the REAL account store.

Creates N tester accounts (default 10) directly in the same SQLite auth database
the API uses (``ACCOUNTS_DB_PATH``, default ``./data/accounts.db``). Each account:

* gets a **strong random password** (128 bits from :mod:`secrets`), which is
  **hashed with Argon2id** by :class:`AccountStore` before it touches the disk —
  the plaintext is NEVER stored anywhere;
* is created with the ``tester`` role (distinct from ``owner``);
* records the required Loi 25 / RGPD consents (terms + privacy) at the current
  legal version, with ``age_confirmed=True`` (these are operator-provisioned
  accounts, not public sign-ups).

The 10 identifier / password pairs are printed to the console **once** so the
operator can distribute them. They cannot be recovered afterwards (only the hash
is kept). Re-run to rotate a lost password with ``--reset``.

Idempotency
-----------
Usernames/emails are deterministic (``beta01`` … ``betaNN``), so re-running does
NOT create duplicates: an existing tester is reported ``EXISTS`` (password left
unchanged) unless ``--reset`` is passed, in which case a fresh password is set
and shown.

Usage
-----
    python scripts/seed_testers.py                 # create/ensure 10 testers
    python scripts/seed_testers.py --count 15      # a different fleet size
    python scripts/seed_testers.py --reset         # rotate passwords for all
    ACCOUNTS_DB_PATH=/app/data/accounts.db python scripts/seed_testers.py

Run it INSIDE the backend container so it writes to the persistent volume, e.g.:
    docker compose exec backend python scripts/seed_testers.py
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the repo root importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.api.account_store import AccountError, AccountStore  # noqa: E402

try:
    # Single source of truth for the consent version (matches the API).
    from src.api.routes.legal import LAST_UPDATED as LEGAL_VERSION  # noqa: E402
except Exception:  # pragma: no cover - defensive; keep the seed self-contained
    LEGAL_VERSION = "v0"

DEFAULT_COUNT = int(os.environ.get("TESTER_COUNT", "10"))
DEFAULT_DOMAIN = os.environ.get("TESTER_EMAIL_DOMAIN", "beta.mia.markets")
DEFAULT_PREFIX = os.environ.get("TESTER_USERNAME_PREFIX", "beta")
DEFAULT_DB = os.environ.get("ACCOUNTS_DB_PATH", "./data/accounts.db")

# Password length in random bytes → ~1.33× chars in url-safe base64. 16 bytes =
# 128 bits of entropy, comfortably above the store's 10-char minimum.
_PASSWORD_BYTES = 16


def _generate_password() -> str:
    """A strong, URL-safe random password (128 bits)."""
    return secrets.token_urlsafe(_PASSWORD_BYTES)


def seed_testers(
    store: AccountStore,
    *,
    count: int = DEFAULT_COUNT,
    prefix: str = DEFAULT_PREFIX,
    domain: str = DEFAULT_DOMAIN,
    reset: bool = False,
) -> List[Dict[str, Any]]:
    """Create/ensure ``count`` tester accounts. Returns per-account records.

    Each record: ``{username, email, role, status, password}`` where ``status``
    is ``created`` | ``reset`` | ``exists`` and ``password`` is present ONLY for
    ``created``/``reset`` (an unchanged existing account never re-exposes one).
    Idempotent: safe to run repeatedly.
    """
    consents = [("terms", LEGAL_VERSION), ("privacy", LEGAL_VERSION)]
    records: List[Dict[str, Any]] = []

    for i in range(1, count + 1):
        username = f"{prefix}{i:02d}"
        email = f"{username}@{domain}"
        existing = store.get_account_by_identifier(email)

        if existing is not None:
            if reset:
                password = _generate_password()
                store.set_password(existing["id"], password)
                records.append(
                    {
                        "username": existing["username"],
                        "email": existing["email"],
                        "role": existing["role"],
                        "status": "reset",
                        "password": password,
                    }
                )
            else:
                records.append(
                    {
                        "username": existing["username"],
                        "email": existing["email"],
                        "role": existing["role"],
                        "status": "exists",
                        "password": None,
                    }
                )
            continue

        password = _generate_password()
        try:
            account = store.create_account(
                username,
                email,
                password,
                age_confirmed=True,
                consents=consents,
                role="tester",
            )
        except AccountError as exc:
            # A username/email collision on a non-matching field (rare) — report
            # it rather than aborting the whole batch.
            records.append(
                {
                    "username": username,
                    "email": email,
                    "role": "tester",
                    "status": f"error:{exc.code}",
                    "password": None,
                }
            )
            continue

        records.append(
            {
                "username": account["username"],
                "email": account["email"],
                "role": account["role"],
                "status": "created",
                "password": password,
            }
        )

    return records


def _print_report(records: List[Dict[str, Any]], db_path: str) -> None:
    created = sum(1 for r in records if r["status"] == "created")
    reset = sum(1 for r in records if r["status"] == "reset")
    exists = sum(1 for r in records if r["status"] == "exists")
    errors = [r for r in records if r["status"].startswith("error")]

    line = "=" * 72
    print(line)
    print("  SEED TESTEURS — BETA PRIVÉE")
    print(f"  Base d'auth : {db_path}")
    print(f"  Créés={created}  Réinitialisés={reset}  Déjà présents={exists}"
          f"  Erreurs={len(errors)}")
    print(line)
    print("  ⚠️  MOTS DE PASSE AFFICHÉS UNE SEULE FOIS — copie-les maintenant.")
    print("      Seul le HASH Argon2id est stocké ; ils sont irrécupérables.")
    print(line)
    print(f"  {'IDENTIFIANT (email)':<34} {'MOT DE PASSE':<26} {'ÉTAT'}")
    print("  " + "-" * 68)
    for r in records:
        pwd = r["password"] if r["password"] else "— (inchangé)"
        print(f"  {r['email']:<34} {pwd:<26} {r['status']}")
    print(line)
    if exists and not (created or reset):
        print("  Tous les comptes existaient déjà (idempotent). "
              "Utilise --reset pour régénérer les mots de passe.")
        print(line)
    if errors:
        print("  ⚠️  Des comptes n'ont pas pu être créés — voir la colonne ÉTAT.")
        print(line)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Seed closed-beta tester accounts.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help=f"Number of tester accounts (default {DEFAULT_COUNT}).")
    parser.add_argument("--db", default=DEFAULT_DB,
                        help=f"Accounts SQLite path (default {DEFAULT_DB}).")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help=f"Username/email prefix (default {DEFAULT_PREFIX}).")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN,
                        help=f"Email domain (default {DEFAULT_DOMAIN}).")
    parser.add_argument("--reset", action="store_true",
                        help="Rotate the password of any existing tester.")
    args = parser.parse_args(argv)

    if args.count < 1:
        print("--count must be >= 1", file=sys.stderr)
        return 2

    store = AccountStore(db_path=args.db)
    records = seed_testers(
        store,
        count=args.count,
        prefix=args.prefix,
        domain=args.domain,
        reset=args.reset,
    )
    _print_report(records, args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
