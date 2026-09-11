"""SC-4 — a fresh /translate throttle per test, for the files that need one.

``/api/scanner/translate`` is capped by a MODULE-LEVEL throttle (40 calls per
5 minutes). A counter shared across tests is a flake waiting to happen: the 40th
translate call of a session would 429 a test that has nothing to do with
throttling.

Deliberately NOT an autouse fixture in the root ``conftest.py``. That version
imported a FastAPI router before every one of the ~3 000 tests in this suite —
a global side effect paid by every unrelated test, for the benefit of three
files. Import this module's fixture only where translate calls actually happen::

    from tests.conftest_translate_throttle import fresh_translate_throttle  # noqa: F401

Tests that exercise the cap itself install their own throttle, which takes
precedence over this default.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def fresh_translate_throttle(monkeypatch):
    from src.api.auth_throttle import AuthThrottle
    from src.api.routes import scanner_translate

    monkeypatch.setattr(scanner_translate, "_TRANSLATE_THROTTLE", AuthThrottle())
