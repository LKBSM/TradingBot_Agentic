"""The Stripe preflight script must EXIST in the deployed image (PAY-4).

Why this file exists
-------------------
On 2026-09-27 the founder ran the documented go-live command in the Render shell
and got::

    python: can't open file '/app/scripts/stripe_preflight.py':
    [Errno 2] No such file or directory

The Dockerfile was fine (``COPY . .``, and ``.dockerignore`` excludes no ``.py``
under ``scripts/``). The script simply had never been merged to ``main``: it
lived only on an unmerged branch, so the image built from ``main`` could not
contain it. A documented operational command that does not exist in production is
worse than no command at all — it reads as "checked" when nothing was checked.

These tests make that impossible to repeat: the file is committed, it imports,
and the amounts it insists on are the ones actually being charged.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "stripe_preflight.py"


def _load():
    spec = importlib.util.spec_from_file_location("stripe_preflight_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_is_committed_at_the_documented_path():
    """The path the runbook tells the founder to type, verbatim."""
    assert SCRIPT.is_file(), (
        "scripts/stripe_preflight.py is missing. The go-live runbook says to run "
        "`python scripts/stripe_preflight.py` in the Render shell; if the file is "
        "not on main, that command cannot work in the deployed image."
    )


def test_nothing_in_dockerignore_excludes_the_scripts_directory():
    """`COPY . .` only copies what the build context contains.

    ``test_*.py`` and ``*.md`` in .dockerignore are ROOT-LEVEL globs (Docker's
    ``*`` does not cross ``/``), so they cannot reach ``scripts/``. This test
    fails if someone adds a pattern that would.
    """
    patterns = [
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    offenders = [
        p for p in patterns
        if p.rstrip("/") in {"scripts", "*"}
        or p.startswith("scripts/")
        or p in {"**/scripts", "**/*.py", "*.py"}
    ]
    assert offenders == [], f".dockerignore would exclude the script: {offenders}"


def test_the_script_imports_without_a_stripe_key():
    """It must be loadable in the Render shell before anything is configured.

    An import-time crash would be indistinguishable, to the reader, from a real
    Stripe misconfiguration.
    """
    module = _load()
    assert module.CORRECT_WEBHOOK_PATH == "/api/billing/webhook"
    assert module.RETIRED_WEBHOOK_PATH == "/api/v1/billing/webhook"


def test_it_expects_the_amounts_that_are_actually_charged():
    """3999 and 35988 cents — the live Stripe prices, to the cent."""
    module = _load()
    assert module.GO_LIVE_CENTS == {"MONTHLY": 3999, "ANNUAL": 35988}


def test_the_expected_amounts_agree_with_the_single_source():
    """The second witness must not contradict config/pricing.json.

    The script's constant exists so a WRONG config is caught; the two disagreeing
    means one of them was edited alone, which is exactly the drift to catch here
    rather than in the Render shell.
    """
    module = _load()
    cfg = json.loads((REPO_ROOT / "config" / "pricing.json").read_text(encoding="utf-8"))
    assert round(cfg["plans"]["monthly"]["amount"] * 100) == module.GO_LIVE_CENTS["MONTHLY"]
    assert round(cfg["plans"]["annual"]["amountPerYear"] * 100) == module.GO_LIVE_CENTS["ANNUAL"]


def test_the_catalog_check_passes_on_the_shipped_config():
    module = _load()
    module._results.clear()
    module.check_catalog()
    failures = [msg for level, msg in module._results if level == module.FAIL]
    assert failures == [], failures


def test_the_catalog_check_fails_on_a_wrong_amount(monkeypatch):
    """Proves the check can actually fail — a green light that cannot go red is
    not a check.
    """
    module = _load()
    monkeypatch.setitem(module.GO_LIVE_CENTS, "MONTHLY", 3900)
    module._results.clear()
    module.check_catalog()
    failures = [msg for level, msg in module._results if level == module.FAIL]
    assert any("MONTHLY" in msg for msg in failures), module._results


def test_it_checks_every_event_the_paywall_needs():
    """The script reads the live event list rather than a copy of it."""
    module = _load()
    from src.billing.stripe_client import ACCOUNT_SUBSCRIPTION_EVENTS

    assert module.ACCOUNT_SUBSCRIPTION_EVENTS is ACCOUNT_SUBSCRIPTION_EVENTS
    assert len(module.ACCOUNT_SUBSCRIPTION_EVENTS) == 9


def test_it_never_prints_a_key(monkeypatch, capsys):
    """A preflight run happens in a shell whose output gets pasted into chats."""
    fake_key = "sk_live_THIS_MUST_NEVER_BE_PRINTED"
    monkeypatch.setenv("STRIPE_SECRET_KEY", fake_key)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_THIS_MUST_NEVER_BE_PRINTED")
    module = _load()
    module._results.clear()
    module.check_env(allow_test=False)
    out = capsys.readouterr().out
    assert fake_key not in out
    assert "whsec_THIS_MUST_NEVER_BE_PRINTED" not in out
    # It still says WHICH mode the key is in — that is the useful, safe part.
    assert "LIVE" in out


@pytest.mark.parametrize("cadence,interval", [("monthly", "month"), ("annual", "year")])
def test_cadences_map_to_stripe_intervals(cadence, interval):
    module = _load()
    assert module.cadence_to_interval(cadence) == interval
