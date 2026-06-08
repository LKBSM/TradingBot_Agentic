"""Snapshot store — Sprint 6 batch 6.3.

Per-signal snapshot persisting feature values, scoring components, model
versions, and data hashes at the exact moment a signal was published.

Audit reference
---------------
- ``audits/2026-Q2/section_3_7_state_machine.md`` P0-16, F15.
- ``audits/2026-Q2/algo_audit_institutional.md`` plan §6 (Sprint 6).

Why this matters
----------------
Commercial readiness criterion #3 (brief §6) :

> *Reproductibilité : tout signal des 12 derniers mois peut être rejoué
>   à l'identique.*

Without a snapshot store, the team can produce **today's** signal but
cannot replay **last March's** signal because the model parameters, the
RSI definition, the calendar version, or the BOS retest tolerance may
have drifted in the meantime. The snapshot store freezes :

- the 8 component scores at signal-time,
- the OHLCV bar timestamp + sha256 of the input series tail,
- the model versions (HAR-RV state, scoring weights, prompt registry id),
- the state machine config snapshot,
- the regime label assigned at time t,
- the conformal interval published (if any).

Storage
-------
JSON-lines (one snapshot per line) under
``data/snapshots/{symbol}_{YYYYMMDD}.jsonl``. Append-only. Daily file
rotation. Compression at archive time.

Status (Sprint 6 prep) : **scaffold + write/read API**. Wiring into
:class:`ConfluenceDetector` and :class:`SignalReplay` is Sprint 6 batch
6.3 itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SignalSnapshot:
    """Frozen view of everything needed to replay one signal."""

    signal_id: str
    symbol: str
    timeframe: str
    bar_timestamp_iso: str

    # Scoring
    components: dict[str, float]
    final_score: float
    tier: str

    # Engine versions
    code_commit_sha: str
    confluence_weights_id: str  # hash of DEFAULT_WEIGHTS
    state_machine_config_id: str
    har_rv_state_id: Optional[str] = None
    lgbm_model_id: Optional[str] = None
    prompt_registry_id: Optional[str] = None  # narrative engine (out of algo scope but logged)

    # Data lineage
    input_csv_sha256: Optional[str] = None
    input_bar_count: Optional[int] = None
    calendar_file_sha256: Optional[str] = None

    # Regime
    regime_label: Optional[str] = None
    regime_confidence: Optional[float] = None

    # Conformal interval (if any)
    conformal_lower: Optional[float] = None
    conformal_upper: Optional[float] = None
    conformal_alpha: Optional[float] = None

    # Bookkeeping
    snapshot_created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class SnapshotStore:
    """JSONL append store for :class:`SignalSnapshot`."""

    def __init__(self, root_dir: str | Path = "data/snapshots"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, symbol: str, dt_iso: str) -> Path:
        day = dt_iso[:10].replace("-", "")
        return self.root_dir / f"{symbol}_{day}.jsonl"

    def write(self, snap: SignalSnapshot) -> None:
        path = self._path_for(snap.symbol, snap.bar_timestamp_iso)
        line = snap.to_json_line()
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_day(self, symbol: str, yyyymmdd: str) -> list[dict[str, Any]]:
        path = self.root_dir / f"{symbol}_{yyyymmdd}.jsonl"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def find_by_signal_id(self, signal_id: str) -> Optional[dict[str, Any]]:
        for path in self.root_dir.glob("*.jsonl"):
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if obj.get("signal_id") == signal_id:
                        return obj
        return None


def hash_file_sha256(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_weights_dict(weights: dict[str, float]) -> str:
    """Stable hash of a weights dict for traceability."""
    items = sorted(weights.items())
    payload = json.dumps(items, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


__all__ = ["SignalSnapshot", "SnapshotStore", "hash_file_sha256", "hash_weights_dict"]
