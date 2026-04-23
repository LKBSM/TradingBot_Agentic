"""State-machine persistence — save on shutdown, reload on startup.

Without persistence, every process restart wipes the state machine: any
active position, cooldown timer, and confirmation progress is lost. A
client watching the dashboard would see a phantom ``HOLD`` even though
the market just confirmed a LONG seconds before the restart.

This module provides a thin, atomic JSON persistence layer. The state
machine itself already knows how to :meth:`to_dict` / :meth:`from_dict`;
we just add disk I/O, atomic writes (to avoid a half-written file
leaving the scanner without state), and a staleness guard.

Staleness guard
---------------
If the saved state is older than ``max_staleness_bars`` bars (measured
by comparing ``last_bar_ts`` against the current bar on load), we
deliberately drop the saved state and start fresh. Arming on stale
market state is worse than losing it — a cooldown from 3 hours ago is
no longer relevant, and an active signal whose invalidation we missed
is dangerous.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from src.intelligence.signal_state_machine import SignalStateMachine

logger = logging.getLogger(__name__)


def save_state_machine(
    machine: SignalStateMachine,
    path: Path,
) -> bool:
    """Serialise ``machine`` to ``path`` atomically.

    Writes to a temp file in the same directory, then renames — so a
    crash mid-write can never corrupt the existing snapshot.

    Returns ``True`` on success, ``False`` on any failure (logged).
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = machine.to_dict()
        # Atomic write: tempfile in same dir, then rename
        fd, tmp_path = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp_path, path)  # atomic on POSIX + Windows (Python 3.3+)
            logger.info("State machine persisted to %s", path)
            return True
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as exc:
        logger.error("Failed to save state machine to %s: %s", path, exc)
        return False


def load_state_machine(
    path: Path,
    signal_rehydrator: Optional[Callable[[dict], Any]] = None,
    current_bar_ts: Optional[str] = None,
    max_staleness_bars: int = 4,
    bar_minutes: int = 15,
) -> Optional[SignalStateMachine]:
    """Load and rehydrate a state machine from ``path``.

    Returns ``None`` if the file is missing, corrupt, wrong schema, or
    stale (older than ``max_staleness_bars`` relative to ``current_bar_ts``).
    The caller is expected to construct a fresh machine in that case.

    ``signal_rehydrator`` lets the caller rebuild a real ``ConfluenceSignal``
    from the stored dict — without it, the active-signal field is left as a
    dict placeholder.
    """
    path = Path(path)
    if not path.exists():
        logger.info("No persisted state at %s — starting fresh", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("State snapshot at %s unreadable: %s — starting fresh", path, exc)
        return None

    # Staleness guard — skip if saved bar is too far behind current
    saved_ts = payload.get("last_bar_ts")
    if saved_ts and current_bar_ts and max_staleness_bars > 0:
        try:
            a = pd.to_datetime(saved_ts)
            b = pd.to_datetime(current_bar_ts)
            gap_min = abs((b - a).total_seconds() / 60.0)
            gap_bars = gap_min / bar_minutes
            if gap_bars > max_staleness_bars:
                logger.warning(
                    "State snapshot at %s is %d bars stale (saved=%s, current=%s) — discarding",
                    path, int(gap_bars), saved_ts, current_bar_ts,
                )
                return None
        except (TypeError, ValueError):
            # Un-parseable timestamp — trust the caller, load anyway
            pass

    try:
        machine = SignalStateMachine.from_dict(
            payload, signal_rehydrator=signal_rehydrator
        )
        logger.info(
            "State machine restored from %s (phase=%s, last_bar_ts=%s)",
            path, payload.get("phase"), saved_ts,
        )
        return machine
    except Exception as exc:
        logger.error(
            "State snapshot at %s failed to rehydrate: %s — starting fresh",
            path, exc,
        )
        return None
