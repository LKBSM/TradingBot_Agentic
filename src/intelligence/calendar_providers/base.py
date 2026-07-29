"""Calendar provider interface + neutral event (NW-1 / NW-1b).

MANDATORY ABSTRACTION. No provider-specific field ever crosses this boundary:
the service, store, API schema and front see only ``ProviderEvent`` and its
neutral fields, modelled on the OFFICIAL sources (BLS, BEA, Census, Federal
Reserve, Eurostat, ECB). Swapping or adding an adapter must touch nothing
outside ``src/intelligence/calendar_providers/``.

Every event carries its own ``source`` and ``license_label`` so two publications
from two organisms coexist with unambiguous provenance. Each provider also
declares an :class:`ProviderAttribution` (organism + reuse-policy URL) so the
page can render the licence-required attribution block generically.

An adapter that cannot supply a field leaves it ``None``; a ``None`` is rendered
as ABSENT downstream — never fabricated, never defaulted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass(frozen=True)
class ProviderEvent:
    """A scheduled economic event as returned by a calendar adapter — neutral,
    source-agnostic. Modelled on official statistical releases.

    NW-1b: no ``impact`` (no organism ranks its releases) and no ``forecast``
    (no organism publishes an analyst consensus). Values are the indicator's own,
    in the indicator's own unit — never a price, never converted.
    """

    # Source identity + stable recurring identifier.
    source: str                        # e.g. "bls", "ecb", "forexfactory"
    provider_ref: str                  # native unique id of THIS release within the source
    series_code: Optional[str]         # stable recurring code (BLS series id…) — None for FF
    # What the event is.
    event: str
    currency: str
    # When (UTC) + the publisher's own timezone (e.g. "America/New_York").
    scheduled_at: datetime
    source_timezone: Optional[str] = None
    # Whether the publication TIME is confirmed from an official source. False →
    # the record is marked and EXCLUDED from amplitude measures; never approximated.
    time_confirmed: bool = True
    # Who publishes it + its cadence.
    organism: Optional[str] = None     # e.g. "Bureau of Labor Statistics" — None for FF
    periodicity: Optional[str] = None  # "monthly" | "quarterly" | "eight_per_year"
    # Values (in the indicator's own unit — never a price, never converted).
    value_unit: Optional[str] = None
    actual: Optional[float] = None
    actual_initial: Optional[float] = None  # value first published for this release
    previous: Optional[float] = None
    revised: bool = False
    revised_at: Optional[datetime] = None
    # Provenance / rights, per record.
    license_label: Optional[str] = None


@dataclass(frozen=True)
class ProviderAttribution:
    """Licence attribution for one source: named, with its reuse-policy URL.
    Rendered in the page's attribution block (a condition of every source's
    reuse policy, not an option)."""

    source: str
    organism: str
    license_label: str
    policy_url: str


@dataclass(frozen=True)
class ProviderFetch:
    """Result of a provider fetch: the events plus the source-declared coverage
    window (so the service can report honest coverage without guessing)."""

    events: List[ProviderEvent] = field(default_factory=list)
    coverage_start: Optional[datetime] = None
    coverage_end: Optional[datetime] = None


class CalendarProvider(ABC):
    """A source of scheduled economic events. The ONLY place a concrete data
    format is allowed to exist."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Stable machine name stamped onto every event (``ProviderEvent.source``)."""

    @abstractmethod
    def fetch(self) -> ProviderFetch:
        """Return the currently-known events + declared coverage window. Network
        or parse errors should be swallowed into an empty fetch (graceful): a
        source that fails to refresh must never erase already-stored data."""

    def attributions(self) -> List[ProviderAttribution]:
        """Licence attributions for every organism this provider can emit. The
        service renders one block entry per source that actually produced a
        served event. Default: none (e.g. the dev-only prototype)."""
        return []


__all__ = [
    "ProviderEvent",
    "ProviderAttribution",
    "ProviderFetch",
    "CalendarProvider",
]
