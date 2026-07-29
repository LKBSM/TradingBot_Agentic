"""TF-1 GUARD — no timeframe enumeration may live outside the single registry.

This is the test that stops the regression from coming back: it scans the backend
source for the tell-tale shapes of a hand-copied timeframe map (minutes map,
provider-string map, or a hardcoded 3-unit tuple) and fails if any reappears
outside ``config/timeframes.json`` + ``src/intelligence/timeframe_registry.py``.
Adding a seventh unit must be one edit to the JSON — enforced here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

# Files allowed to name timeframes literally: the registry itself.
_ALLOWED = {
    _SRC / "intelligence" / "timeframe_registry.py",
}

# Signatures of a hand-copied enumeration.
_FORBIDDEN = [
    (re.compile(r'"M15"\s*:\s*15'), "inline timeframe→minutes map"),
    (re.compile(r'"M5"\s*:\s*"5min"'), "inline timeframe→provider-string map"),
    (re.compile(r'"D1"\s*:\s*1440'), "inline timeframe→minutes map"),
    (re.compile(r'\(\s*"M15"\s*,\s*"H1"\s*,\s*"H4"\s*\)'), "hardcoded 3-unit tuple"),
    (re.compile(r'\{\s*"m15"\s*,\s*"h1"\s*,\s*"h4"'), "hardcoded MTF key set"),
]


def _py_files():
    for p in _SRC.rglob("*.py"):
        if "__pycache__" in p.parts or p in _ALLOWED:
            continue
        yield p


def test_no_hardcoded_timeframe_enumeration_in_backend():
    offenders = []
    for path in _py_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern, what in _FORBIDDEN:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(_ROOT)} → {what}")
    assert offenders == [], (
        "Timeframe enumeration found outside the registry (TF-1). Derive it from "
        "src/intelligence/timeframe_registry.py instead:\n" + "\n".join(offenders)
    )


def test_registry_is_the_only_minutes_source():
    """The registry file itself is the one place a minutes map is allowed."""
    reg = (_SRC / "intelligence" / "timeframe_registry.py").read_text(encoding="utf-8")
    # Sanity: the registry does NOT itself hardcode minutes (it reads the JSON).
    assert '"M15": 15' not in reg
