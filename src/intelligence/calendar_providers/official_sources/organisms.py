"""One adapter per issuing organism (NW-1b).

Each class is a thin :class:`OfficialSourceProvider` that fixes its source key +
reuse-policy URL. The heavy lifting (catalog join, timezone-aware scheduling,
attribution, graceful failure) is shared in the base. Each docstring records the
official machine-readable path a live ``date_source`` should use — the seam is a
single injected callable, so wiring a source's live feed touches only this file.

No source-specific format ever crosses ``ProviderEvent``: substituting or adding
an organism changes nothing in the service, store, schema or front.
"""

from __future__ import annotations

from src.intelligence.calendar_providers.official_sources.base_official import (
    OfficialSourceProvider,
)


class BLSProvider(OfficialSourceProvider):
    """Bureau of Labor Statistics — CPI, PPI, Employment Situation.

    Live dates: the ``bls.ics`` iCalendar feed
    (https://www.bls.gov/schedule/news_release/bls.ics) — the HTML schedule
    blocks bots. Values: the v2 time-series API (POST, free key). The API returns
    only the latest (revised) value; first-print vintages are BLS download files.
    """

    SOURCE = "bls"
    POLICY_URL = "https://www.bls.gov/opub/copyright-information.htm"


class BEAProvider(OfficialSourceProvider):
    """Bureau of Economic Analysis — GDP, PCE price index.

    Live dates: the news schedule (JSON via API + ``.ics``,
    https://www.bea.gov/news/schedule). Values: ``apps.bea.gov/api/data``
    (free UserID). Vintages are not in the API (see audit).
    """

    SOURCE = "bea"
    POLICY_URL = "https://www.bea.gov/help/faq/147"


class CensusProvider(OfficialSourceProvider):
    """U.S. Census Bureau — retail sales, housing starts, durable goods.

    Live dates: the release calendar (page/PDF) + the ``indicator.xml`` RSS for
    detecting actual releases. Values: the EITS time-series JSON API
    (https://api.census.gov/data/timeseries/eits). Advance vs final are separate
    series (revision surfaces by diffing them).
    """

    SOURCE = "census"
    POLICY_URL = "https://www.census.gov/data/developers/about/terms-of-service.html"


class FederalReserveProvider(OfficialSourceProvider):
    """Federal Reserve Board — FOMC rate decision.

    Live dates: the FOMC calendar (HTML, ``fomccalendars.htm``) + the
    ``press_monetary.xml`` RSS for the statement. The decision is published at
    14:00 ET (stable since 2013). Target range via H.15.
    """

    SOURCE = "federal_reserve"
    POLICY_URL = "https://www.federalreserve.gov/disclaimer.htm"


class EurostatProvider(OfficialSourceProvider):
    """Eurostat — HICP flash, euro-area GDP flash, unemployment.

    Live dates: the euro-indicators release calendar ``.ics`` feed. Values: the
    free JSON-stat 2.0 / SDMX API (no key). First releases at 11:00 CET
    (impartiality protocol). CC BY 4.0 — attribution mandatory.
    """

    SOURCE = "eurostat"
    POLICY_URL = "https://ec.europa.eu/eurostat/help/copyright-notice"


class ECBProvider(OfficialSourceProvider):
    """European Central Bank — monetary-policy (rate) decision.

    Live dates: the Governing Council calendar (HTML, ``mgcgc``). Values: the
    free SDMX 2.1 Data Portal API (no key), with ``includeHistory=true`` for
    vintages. Decision published 14:15 CET (stable since 2022-07-21). Reuse
    requires the statistics NOT be modified — values shown exactly as published.
    """

    SOURCE = "ecb"
    POLICY_URL = "https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html"


ALL_OFFICIAL_PROVIDERS = (
    BLSProvider,
    BEAProvider,
    CensusProvider,
    FederalReserveProvider,
    EurostatProvider,
    ECBProvider,
)


__all__ = [
    "BLSProvider",
    "BEAProvider",
    "CensusProvider",
    "FederalReserveProvider",
    "EurostatProvider",
    "ECBProvider",
    "ALL_OFFICIAL_PROVIDERS",
]
