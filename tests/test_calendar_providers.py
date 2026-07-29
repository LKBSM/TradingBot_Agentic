"""Calendar adapters (NW-1 / NW-1b).

Covers the dev-only ForexFactory prototype (neutral mapping, no impact/consensus
emitted, attribution stamped) AND the per-organism official adapters (catalog
join, DST-aware scheduling, attribution, adapter substitution)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.intelligence.calendar_providers.base import ProviderEvent
from src.intelligence.calendar_providers.forexfactory_provider import (
    ForexFactoryCalendarProvider,
)
from src.intelligence.calendar_providers.official_provider import (
    OfficialCalendarProvider,
)
from src.intelligence.calendar_providers.official_sources.base_official import (
    ReleaseInstance,
    load_catalog,
)
from src.intelligence.calendar_providers.official_sources.organisms import (
    ALL_OFFICIAL_PROVIDERS,
    BLSProvider,
    ECBProvider,
)

# --------------------------------------------------------------------------- #
# ForexFactory prototype (dev-only)
# --------------------------------------------------------------------------- #
_RAW = [
    {"title": "CPI y/y", "country": "USD", "impact": "high",
     "date": "2026-07-28T08:30:00-04:00", "actual": "", "forecast": "3.1%", "previous": "3.3%"},
    {"title": "Consumer Confidence", "country": "USD", "impact": "low",
     "date": "2026-07-28T10:00:00-04:00", "forecast": "99.1", "previous": "98.0"},
    {"title": "Bank Holiday", "country": "EUR", "impact": "holiday",
     "date": "2026-07-28T00:00:00-04:00"},
    {"title": "Malformed", "country": "", "impact": "high", "date": ""},
]


def _ff():
    return ForexFactoryCalendarProvider(fetch_fn=lambda: list(_RAW))


def test_ff_emits_no_impact_and_no_forecast_fields():
    # The neutral event carries neither an impact ranking nor a consensus.
    assert not hasattr(ProviderEvent, "impact")
    assert not hasattr(ProviderEvent, "forecast")
    ev = _ff().fetch().events[0]
    assert not hasattr(ev, "impact")
    assert not hasattr(ev, "forecast")


def test_ff_drops_holiday_and_malformed():
    titles = {e.event for e in _ff().fetch().events}
    assert titles == {"CPI y/y", "Consumer Confidence"}


def test_ff_only_fields_are_none_never_fabricated():
    ev = next(e for e in _ff().fetch().events if e.event == "CPI y/y")
    assert ev.series_code is None
    assert ev.organism is None
    assert ev.value_unit is None
    assert ev.periodicity is None


def test_ff_actual_previous_parsed_forecast_ignored():
    ev = next(e for e in _ff().fetch().events if e.event == "CPI y/y")
    assert ev.previous == 3.3
    assert ev.actual is None  # empty string → None, not 0


def test_ff_attribution_present():
    atts = _ff().attributions()
    assert len(atts) == 1
    assert atts[0].source == "forexfactory"
    assert atts[0].policy_url


# --------------------------------------------------------------------------- #
# Official per-organism adapters
# --------------------------------------------------------------------------- #
def test_catalog_loads_all_sources():
    cat = load_catalog()
    assert len(cat) >= 13
    assert {c.source for c in cat.values()} == {
        "bls", "bea", "census", "federal_reserve", "eurostat", "ecb",
    }


def test_bls_adapter_joins_catalog_and_schedules_dst_aware():
    cat = load_catalog()
    provider = BLSProvider(
        catalog=cat,
        date_source=lambda c: [
            ReleaseInstance(event_key="us_cpi", release_date="2026-08-12",
                            actual=322.1, previous=321.0)
        ],
    )
    events = provider.fetch().events
    assert len(events) == 1
    e = events[0]
    assert e.source == "bls"
    assert e.organism == "Bureau of Labor Statistics"
    assert e.periodicity == "monthly"
    assert e.value_unit and "indice" in e.value_unit
    # 08:30 America/New_York in August (EDT = UTC-4) → 12:30 UTC.
    assert e.scheduled_at == datetime(2026, 8, 12, 12, 30, tzinfo=timezone.utc)
    assert e.time_confirmed is True


def test_official_adapter_attribution_has_policy_url():
    att = BLSProvider().attributions()
    assert att and att[0].source == "bls"
    assert att[0].policy_url.startswith("https://")
    assert att[0].organism == "Bureau of Labor Statistics"


def test_ecb_values_shown_as_published_not_converted():
    cat = load_catalog()
    provider = ECBProvider(
        catalog=cat,
        date_source=lambda c: [
            ReleaseInstance(event_key="ea_ecb_rate", release_date="2026-09-10", actual=2.4)
        ],
    )
    e = provider.fetch().events[0]
    assert e.actual == 2.4  # exact, never rounded/converted
    assert e.currency == "EUR"


def test_aggregator_composes_all_sources_and_is_named_official():
    agg = OfficialCalendarProvider()
    assert agg.source_name == "official"
    # empty schedule ships → honestly empty, not fabricated
    assert agg.fetch().events == []
    # attribution is available for every organism it can emit
    assert {a.source for a in agg.attributions()} == {
        "bls", "bea", "census", "federal_reserve", "eurostat", "ecb",
    }


def test_adapter_substitution_changes_nothing_in_the_interface():
    # Every official adapter honours the same interface: source_name + fetch +
    # attributions. Substituting one for another needs no downstream change.
    for cls in ALL_OFFICIAL_PROVIDERS:
        p = cls()
        assert isinstance(p.source_name, str) and p.source_name
        assert p.fetch().events == []  # empty schedule
        assert isinstance(p.attributions(), list)
