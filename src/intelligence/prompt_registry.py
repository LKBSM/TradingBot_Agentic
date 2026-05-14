"""Frozen prompt template registry — Sprint LLM-2B.10.

The narrative pipeline emits thousands of LLM calls per day under
constraints that change as compliance/regulatory tooling evolves
(UE 2024/2811, source-citation rules, format constraints). When a
prompt changes, the *behaviour* of the LLM downstream changes, and any
post-hoc evaluation of insight quality has to know which prompt
generated which output.

This module is the source of truth for "what prompt was in flight when
this insight was rendered":

- prompts register themselves with ``register(template_id, version,
  body)`` at import time,
- the registry computes a SHA-256 fingerprint of the body on
  registration; rendering ``body`` later doesn't change the fingerprint
  (registries are *immutable* after registration),
- ``get(template_id)`` returns the latest registered version's record;
  ``get(template_id, version=...)`` pins a specific one,
- every LLM call logs ``{template_id, version, sha256}`` alongside its
  request id and audit-ledger seq.

Why fingerprint at registration, not at lookup
----------------------------------------------
Registration is the only moment when the prompt body is *guaranteed*
to be the source-of-truth string the developer typed. Computing the
sha256 once at registration, and refusing to recompute on lookup, means
a buggy mutator (e.g. someone reaches into ``record.body = ...``)
cannot silently invalidate the audit trail — fingerprints stay stable
as a property of the registration event.

Immutability is enforced by:
- ``PromptRecord`` is a frozen dataclass,
- ``register()`` refuses to overwrite an existing (template_id, version)
  unless ``replace=True`` is explicit, and
- the version list is sorted at insert; ``get()`` always reads the
  highest-version record for that id, so callers can't accidentally
  use a deprecated version unless they ask for it by version.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _fingerprint(body: str) -> str:
    """16-hex-char SHA-256 prefix of the prompt body.

    16 chars (64 bits) collision-resistance is more than enough for the
    audit use-case (we expect ~hundreds of prompts over the product
    lifetime, not adversarial inputs).
    """
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PromptRecord:
    """An immutable registered prompt entry."""

    template_id: str
    version: int
    sha256: str
    body: str
    description: str = ""

    def render(self, **kwargs: str) -> str:
        """Substitute ``{name}`` placeholders without changing the sha256.

        Uses ``str.format_map`` so missing keys do not silently produce
        empty strings — they raise ``KeyError`` and the bug is loud.
        Templates without any ``{}`` placeholders return ``body``
        unchanged (the body is its own format-spec for plain strings).
        """
        return self.body.format_map(kwargs)


class _MissingVersionError(KeyError):
    pass


class PromptRegistry:
    """Process-wide, append-only registry.

    Thread-safe to support unit tests that register/lookup concurrently.
    Lookup is O(1) for "latest version of template_id"; iteration over
    all registered records is supported via ``records()``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (template_id, version) → PromptRecord
        self._by_id_version: Dict[Tuple[str, int], PromptRecord] = {}
        # template_id → sorted list of versions
        self._versions: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def register(
        self,
        template_id: str,
        version: int,
        body: str,
        *,
        description: str = "",
        replace: bool = False,
    ) -> PromptRecord:
        if not template_id or not template_id.strip():
            raise ValueError("template_id is required")
        if version < 1:
            raise ValueError("version must be >= 1")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("body must be a non-empty string")

        record = PromptRecord(
            template_id=template_id,
            version=version,
            sha256=_fingerprint(body),
            body=body,
            description=description,
        )
        key = (template_id, version)
        with self._lock:
            if key in self._by_id_version and not replace:
                existing = self._by_id_version[key]
                if existing.sha256 != record.sha256:
                    raise ValueError(
                        f"prompt {template_id} v{version} already registered "
                        f"with a different body (sha {existing.sha256}); "
                        f"bump the version or pass replace=True"
                    )
                # Idempotent re-registration of the *same* body — return
                # the existing record so import-order doesn't break.
                return existing
            self._by_id_version[key] = record
            vlist = self._versions.setdefault(template_id, [])
            if version not in vlist:
                vlist.append(version)
                vlist.sort()
        return record

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(
        self, template_id: str, *, version: Optional[int] = None
    ) -> PromptRecord:
        with self._lock:
            if template_id not in self._versions:
                raise KeyError(f"no prompt registered with id {template_id!r}")
            versions = self._versions[template_id]
            if version is None:
                version = versions[-1]
            elif version not in versions:
                raise _MissingVersionError(
                    f"prompt {template_id} v{version} not registered "
                    f"(known: {versions})"
                )
            return self._by_id_version[(template_id, version)]

    def latest_version(self, template_id: str) -> int:
        return self.get(template_id).version

    def versions(self, template_id: str) -> List[int]:
        with self._lock:
            if template_id not in self._versions:
                return []
            return list(self._versions[template_id])

    def template_ids(self) -> List[str]:
        with self._lock:
            return sorted(self._versions.keys())

    def records(self) -> List[PromptRecord]:
        """All registered records, sorted by (template_id, version)."""
        with self._lock:
            return [
                self._by_id_version[key]
                for key in sorted(self._by_id_version.keys())
            ]

    def to_audit_dict(self, template_id: str, version: Optional[int] = None) -> dict:
        """Compact {id, version, sha256} for logging on each LLM call.

        Never includes the body — audit logs stay small and the sha is
        the canonical reference back to the registry record.
        """
        rec = self.get(template_id, version=version)
        return {
            "template_id": rec.template_id,
            "version": rec.version,
            "sha256": rec.sha256,
        }

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in self._versions
        if isinstance(item, tuple) and len(item) == 2:
            return item in self._by_id_version
        return False


# Module-level default registry. Tests can construct fresh ones; the
# production narrative engines reach for ``DEFAULT_REGISTRY`` by
# default.
DEFAULT_REGISTRY = PromptRegistry()


def register_default_prompts(registry: Optional[PromptRegistry] = None) -> PromptRegistry:
    """Seed the registry with the current RAG system prompts.

    Idempotent — calling twice with the same bodies is a no-op. Lives
    here (rather than at the bottom of rag/prompts.py) so the registry
    has no inbound dependency on the prompts module, which lets the
    registry be unit-tested in isolation.
    """
    reg = registry or DEFAULT_REGISTRY
    # Local import avoids a circular dependency when rag.prompts is
    # itself importing pieces from this module in the future.
    from src.intelligence.rag.prompts import (
        SYSTEM_PROMPT_DE,
        SYSTEM_PROMPT_EN,
        SYSTEM_PROMPT_ES,
        SYSTEM_PROMPT_FR,
    )

    for lang, body in (
        ("fr", SYSTEM_PROMPT_FR),
        ("en", SYSTEM_PROMPT_EN),
        ("de", SYSTEM_PROMPT_DE),
        ("es", SYSTEM_PROMPT_ES),
    ):
        reg.register(
            template_id=f"rag.system.{lang}",
            version=1,
            body=body,
            description=f"RAG narrative system prompt ({lang}) — UE 2024/2811",
        )
    return reg


__all__ = [
    "DEFAULT_REGISTRY",
    "PromptRecord",
    "PromptRegistry",
    "register_default_prompts",
]
