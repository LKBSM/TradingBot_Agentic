"""Raw provider-response snapshots — reproducibility for MarketReading audits.

Audit DETECTION_QUALITY_REVIEW_2026_06_12 §T3: the feed revises forming bars
between fetches, so a published reading could not be replayed from the final
candles stored in ``candles_cache`` (one reading's close_price existed in no
stored candle). Persisting the raw response per generation makes every reading
replayable bit-for-bit.

Format: one JSON line per generation, appended to a daily-rotated file
``<dir>/<instrument>_<timeframe>_<YYYYMMDD>.jsonl``::

    {"fetched_at": "...", "instrument": "XAUUSD", "timeframe": "M15",
     "candles": [{"ts": "...", "open": ..., "high": ..., "low": ...,
                  "close": ..., "volume": ...}, ...]}

Config (env):
  - ``PROVIDER_SNAPSHOT_ENABLED`` — truthy/falsy, default ON (personal-testing
    phase; ~20 KB per generation, daily files are trivial to prune).
  - ``PROVIDER_SNAPSHOT_DIR``     — default ``./data/provider_snapshots``.

Failure policy: best-effort. A snapshot write must never break reading
generation — errors are logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_ENABLED_ENV_VAR = "PROVIDER_SNAPSHOT_ENABLED"
_DIR_ENV_VAR = "PROVIDER_SNAPSHOT_DIR"
_DEFAULT_DIR = "./data/provider_snapshots"


def _enabled() -> bool:
    raw = os.environ.get(_ENABLED_ENV_VAR)
    if raw is None or raw == "":
        return True
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _candle_to_dict(candle: Any) -> dict[str, Any]:
    ts = getattr(candle, "ts", None)
    return {
        "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        "open": float(candle.open),
        "high": float(candle.high),
        "low": float(candle.low),
        "close": float(candle.close),
        "volume": float(getattr(candle, "volume", 0.0) or 0.0),
    }


def snapshot_provider_response(
    instrument: str,
    timeframe: str,
    candles: Sequence[Any],
    fetched_at: datetime,
) -> None:
    """Append the raw candle list to today's snapshot file (best-effort)."""
    if not _enabled() or not candles:
        return
    try:
        ts = (
            fetched_at.astimezone(timezone.utc)
            if fetched_at.tzinfo
            else fetched_at.replace(tzinfo=timezone.utc)
        )
        snapshot_dir = Path(os.environ.get(_DIR_ENV_VAR) or _DEFAULT_DIR)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / f"{instrument}_{timeframe}_{ts.strftime('%Y%m%d')}.jsonl"
        line = json.dumps(
            {
                "fetched_at": ts.isoformat(),
                "instrument": instrument,
                "timeframe": timeframe,
                "candles": [_candle_to_dict(c) for c in candles],
            },
            separators=(",", ":"),
        )
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:  # never break reading generation for observability
        logger.warning(
            "provider snapshot write failed for %s/%s: %s", instrument, timeframe, exc
        )


__all__ = ["snapshot_provider_response"]
