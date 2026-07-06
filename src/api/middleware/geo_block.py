"""Geo-blocking middleware — denies access from sanctioned / unlicensed jurisdictions.

Required by P29 compliance audit. MIA Markets does not hold a securities
or commodities licence in:

  * United States (SEC Investment Advisers Act §202(a)(11))
  * United Kingdom (FCA — restricted financial promotion regime)

It is also barred from doing business with persons in OFAC SDN-comprehensive
sanction territories: Cuba, Iran, North Korea, Russia, Syria, Belarus.

Le Québec (CA-QC) n'est PAS bloqué : c'est la juridiction de rattachement de
l'entreprise (stratégie légale Loi 25 + LPC québécoises). Le blocage CA-QC
présent jusqu'au 2026-07-05 venait d'un boilerplate généré et contredisait la
réalité — décision fondateur de le retirer (cf. CGU §4 alignées même date).

The middleware resolves the client's country via (in order of priority):

  1. CDN-provided header (Cloudflare ``CF-IPCountry``, AWS CloudFront
     ``CloudFront-Viewer-Country``, Fastly ``Fastly-GeoIP-Country``).
     This is the cheap, production-friendly path.
  2. MaxMind GeoLite2 database, if ``geoip2`` is installed and a DB path
     is configured via ``GEOIP_DB_PATH``.
  3. A test-only injectable resolver (``country_resolver``).

When no resolver yields a country, the request is **allowed** by default —
geo-blocking is a hard barrier on the *known* deny-list, not a closed-by-default
filter. This avoids accidentally bricking the API for misconfigured deployments
while still satisfying the compliance requirement that *known* US/UK/OFAC
clients receive HTTP 451.

Allowlisted paths (always served regardless of origin):
    /health, /api/v1/health, /api/docs, /openapi.json,
    /api/v1/terms, /api/v1/privacy

Configuration
-------------
Environment variables:
    GEO_BLOCK_DISABLED=1           — bypass entirely (default: enabled)
    GEOIP_DB_PATH=/path/GeoLite2-Country.mmdb  — MaxMind DB
    GEO_BLOCK_EXTRA_COUNTRIES=...  — comma-separated ISO-3166 codes to
                                     append to the deny-list

Response shape (HTTP 451 — Unavailable For Legal Reasons):
    {
        "error": "geo_blocked",
        "detail": "Service unavailable in your jurisdiction (US).",
        "country": "US",
        "reference": "https://example.com/api/v1/terms"
    }
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Iterable, Optional, Set

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ─── Deny-lists ────────────────────────────────────────────────────────────

#: Country-level block. ISO-3166-1 alpha-2 codes, uppercase.
BLOCKED_COUNTRIES: Set[str] = {
    # No securities/commodities license
    "US",   # SEC Investment Advisers Act §202(a)(11)
    "GB",   # FCA financial promotion regime — restricted
    # OFAC comprehensive sanctions
    "CU",   # Cuba
    "IR",   # Iran
    "KP",   # North Korea
    "RU",   # Russia
    "SY",   # Syria
    "BY",   # Belarus
}

#: Sub-country regions blocked. Resolved via CDN headers when available.
#: Vide depuis 2026-07-05 : le Québec (CA-QC) a été retiré — juridiction de
#: rattachement de l'entreprise (Loi 25 + LPC), le bloquer était une erreur de
#: boilerplate. La mécanique région reste en place pour un besoin futur.
BLOCKED_REGIONS: Set[str] = set()

#: Paths always served regardless of origin. These must remain accessible
#: so a blocked client can still reach the legal terms explaining the block.
ALLOWED_PATHS: Set[str] = {
    "/health",
    "/api/v1/health",
    "/api/docs",
    "/openapi.json",
    "/api/v1/terms",
    "/api/v1/privacy",
}


# ─── Resolver helpers ──────────────────────────────────────────────────────

#: CDN headers that carry a 2-letter country code.
_CDN_COUNTRY_HEADERS = (
    "cf-ipcountry",                    # Cloudflare
    "cloudfront-viewer-country",       # AWS CloudFront
    "fastly-geoip-country",            # Fastly
    "x-country-code",                  # generic / test
)

#: CDN headers that carry a region code (e.g. "QC" for Quebec).
_CDN_REGION_HEADERS = (
    "cloudfront-viewer-country-region",  # AWS CloudFront
    "cf-region-code",                    # Cloudflare Enterprise
    "x-region-code",                     # generic / test
)


def _country_from_headers(request: Request) -> Optional[str]:
    for h in _CDN_COUNTRY_HEADERS:
        value = request.headers.get(h)
        if value:
            return value.strip().upper()[:2]
    return None


def _region_from_headers(request: Request) -> Optional[str]:
    """Return a region code in ISO-3166-2 form ``CC-RR`` when possible.

    CDN providers emit just the region segment (``QC``) in
    ``CloudFront-Viewer-Country-Region`` while the country sits in a
    sibling header. We compose ``CC-RR`` from the pair so the deny-list
    can use the standard form.
    """
    for h in _CDN_REGION_HEADERS:
        raw = request.headers.get(h)
        if not raw:
            continue
        value = raw.strip().upper()
        if "-" in value:
            return value
        country = _country_from_headers(request)
        if country:
            return f"{country}-{value}"
        return value
    return None


def _country_from_geoip(ip: Optional[str], db_path: Optional[str]) -> Optional[str]:
    """Best-effort MaxMind lookup. Returns None when geoip2 is unavailable."""
    if not ip or not db_path:
        return None
    try:
        import geoip2.database  # type: ignore
    except ImportError:
        return None
    try:
        with geoip2.database.Reader(db_path) as reader:
            return reader.country(ip).country.iso_code
    except Exception as exc:
        logger.debug("MaxMind lookup failed for %s: %s", ip, exc)
        return None


# ─── Middleware ────────────────────────────────────────────────────────────


class GeoBlockMiddleware(BaseHTTPMiddleware):
    """Block requests from sanctioned or unlicensed jurisdictions.

    Parameters
    ----------
    app:
        ASGI application.
    blocked_countries:
        Override the default :data:`BLOCKED_COUNTRIES` set.
    blocked_regions:
        Override the default :data:`BLOCKED_REGIONS` set.
    allowed_paths:
        Additional paths that bypass the check (added to
        :data:`ALLOWED_PATHS`).
    country_resolver:
        Optional callable ``(request) -> Optional[str]`` used in tests to
        inject a country code without going through CDN headers.
    region_resolver:
        Optional callable ``(request) -> Optional["CC-RR"]`` for region
        injection in tests.
    geoip_db_path:
        Path to a MaxMind GeoLite2-Country.mmdb. Falls back to
        ``GEOIP_DB_PATH`` env var.
    disabled:
        Skip the check entirely. Falls back to ``GEO_BLOCK_DISABLED=1``.
    """

    def __init__(
        self,
        app,
        *,
        blocked_countries: Optional[Iterable[str]] = None,
        blocked_regions: Optional[Iterable[str]] = None,
        allowed_paths: Optional[Iterable[str]] = None,
        country_resolver: Optional[Callable[[Request], Optional[str]]] = None,
        region_resolver: Optional[Callable[[Request], Optional[str]]] = None,
        geoip_db_path: Optional[str] = None,
        disabled: Optional[bool] = None,
    ) -> None:
        super().__init__(app)
        extra = os.environ.get("GEO_BLOCK_EXTRA_COUNTRIES", "")
        extra_set = {c.strip().upper() for c in extra.split(",") if c.strip()}

        self._blocked_countries: Set[str] = set(blocked_countries) if blocked_countries is not None else set(BLOCKED_COUNTRIES)
        self._blocked_countries |= extra_set
        self._blocked_regions: Set[str] = set(blocked_regions) if blocked_regions is not None else set(BLOCKED_REGIONS)

        self._allowed_paths: Set[str] = set(ALLOWED_PATHS)
        if allowed_paths is not None:
            self._allowed_paths.update(allowed_paths)

        self._country_resolver = country_resolver
        self._region_resolver = region_resolver
        self._geoip_db_path = geoip_db_path or os.environ.get("GEOIP_DB_PATH")

        if disabled is None:
            disabled = os.environ.get("GEO_BLOCK_DISABLED", "0") == "1"
        self._disabled = bool(disabled)

        if self._disabled:
            logger.warning("Geo-block middleware is DISABLED (GEO_BLOCK_DISABLED=1)")
        else:
            logger.info(
                "Geo-block enabled — countries=%s, regions=%s",
                sorted(self._blocked_countries),
                sorted(self._blocked_regions),
            )

    async def dispatch(self, request: Request, call_next):
        if self._disabled:
            return await call_next(request)

        if request.url.path in self._allowed_paths:
            return await call_next(request)

        country = self._resolve_country(request)
        region = self._resolve_region(request)

        # Region check first: "CA-QC" is more specific than "CA".
        if region and region in self._blocked_regions:
            return self._blocked_response(region, request)
        if country and country in self._blocked_countries:
            return self._blocked_response(country, request)

        return await call_next(request)

    # -- resolution -------------------------------------------------------

    def _resolve_country(self, request: Request) -> Optional[str]:
        if self._country_resolver is not None:
            value = self._country_resolver(request)
            if value:
                return value.upper()[:2]
        header_value = _country_from_headers(request)
        if header_value:
            return header_value
        if self._geoip_db_path:
            ip = request.client.host if request.client else None
            return _country_from_geoip(ip, self._geoip_db_path)
        return None

    def _resolve_region(self, request: Request) -> Optional[str]:
        if self._region_resolver is not None:
            value = self._region_resolver(request)
            if value:
                return value.upper()
        return _region_from_headers(request)

    # -- response ---------------------------------------------------------

    @staticmethod
    def _blocked_response(code: str, request: Request) -> JSONResponse:
        terms_url = str(request.url.replace(path="/api/v1/terms", query=""))
        return JSONResponse(
            status_code=451,
            content={
                "error": "geo_blocked",
                "detail": f"Service unavailable in your jurisdiction ({code}).",
                "country": code,
                "reference": terms_url,
            },
        )
