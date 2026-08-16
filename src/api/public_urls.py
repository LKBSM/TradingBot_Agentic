"""Public base URLs — the single source for every outward-facing redirect (PAY-2).

No redirect URL is hard-coded anywhere. Two bases, both env-driven:

  APP_PUBLIC_URL  the PUBLIC url of the FRONTEND (what the user's browser sees)
                  → Stripe Checkout success/cancel, Stripe Customer Portal return,
                    Google OAuth "back to the app", and the email links
                    (verification, password reset).
  API_PUBLIC_URL  the PUBLIC url of the BACKEND
                  → the Google OAuth ``redirect_uri`` (Google calls back to the API).

In development both fall back to localhost, so nothing changes for local runs and
tests. In PRODUCTION (``ENVIRONMENT=production``/``prod``) they MUST be set to a
real https origin: a missing value or one pointing at localhost makes the app
REFUSE TO BOOT (:func:`assert_public_urls_configured`). A failed deploy is far
better than a paying customer bounced to ``localhost`` after Stripe or Google.

Back-compat: older env names are still honoured as fallbacks (``APP_URL`` /
``FRONTEND_BASE_URL`` for the app base, ``API_BASE_URL`` for the api base), and
the per-redirect overrides (``STRIPE_SUCCESS_URL`` …, ``GOOGLE_REDIRECT_URI``)
keep working — they simply default THROUGH these bases instead of localhost.
"""

from __future__ import annotations

import os

_DEV_APP_URL = "http://localhost:3000"
_DEV_API_URL = "http://localhost:8000"


class PublicURLMisconfigured(RuntimeError):
    """Raised at startup when a production deploy lacks a real public URL."""


def _is_production() -> bool:
    return os.environ.get("ENVIRONMENT", "").strip().lower() in {"production", "prod"}


def _clean(url: object) -> str | None:
    if not isinstance(url, str):
        return None
    stripped = url.strip().rstrip("/")
    return stripped or None


def _app_raw() -> str | None:
    return (
        _clean(os.environ.get("APP_PUBLIC_URL"))
        or _clean(os.environ.get("APP_URL"))
        or _clean(os.environ.get("FRONTEND_BASE_URL"))
    )


def _api_raw() -> str | None:
    return (
        _clean(os.environ.get("API_PUBLIC_URL"))
        or _clean(os.environ.get("API_BASE_URL"))
    )


def app_public_url() -> str:
    """Frontend public base URL (browser-facing), no trailing slash."""
    return _app_raw() or _DEV_APP_URL


def api_public_url() -> str:
    """Backend public base URL (OAuth callback target), no trailing slash."""
    return _api_raw() or _DEV_API_URL


def _looks_local(url: str) -> bool:
    low = url.lower()
    return any(host in low for host in ("localhost", "127.0.0.1", "0.0.0.0"))


def assert_public_urls_configured() -> None:
    """Fail-fast in production if a public URL is missing or points at localhost.

    A no-op outside production (dev/CI/tests keep the localhost fallbacks). Called
    at app startup so a misconfigured deploy dies loudly instead of silently
    sending customers to localhost after a Stripe/Google redirect.
    """
    if not _is_production():
        return
    problems: list[str] = []
    app_raw = _app_raw()
    api_raw = _api_raw()
    if not app_raw:
        problems.append(
            "APP_PUBLIC_URL is required in production "
            "(the frontend's public https origin, e.g. https://mia.markets)."
        )
    elif _looks_local(app_raw):
        problems.append(
            f"APP_PUBLIC_URL must not point at localhost in production (got {app_raw!r})."
        )
    if not api_raw:
        problems.append(
            "API_PUBLIC_URL is required in production "
            "(the backend's public https origin, e.g. https://api.mia.markets)."
        )
    elif _looks_local(api_raw):
        problems.append(
            f"API_PUBLIC_URL must not point at localhost in production (got {api_raw!r})."
        )
    if problems:
        raise PublicURLMisconfigured(
            "Refusing to start — outward-facing redirect URLs are misconfigured:\n  - "
            + "\n  - ".join(problems)
        )
