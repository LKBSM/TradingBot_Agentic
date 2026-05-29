"""Tests for src/api/middleware/geo_block.py — P29 compliance gate."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware.geo_block import (
    ALLOWED_PATHS,
    BLOCKED_COUNTRIES,
    BLOCKED_REGIONS,
    GeoBlockMiddleware,
)


def _make_app(**mw_kwargs) -> FastAPI:
    """Build a minimal FastAPI app with geo-block + a couple of routes."""
    app = FastAPI()
    app.add_middleware(GeoBlockMiddleware, **mw_kwargs)

    @app.get("/api/v1/health")
    def health():
        return {"ok": True}

    @app.get("/api/v1/terms")
    def terms():
        return {"terms": "..."}

    @app.get("/api/v1/private")
    def private():
        return {"private": True}

    return app


# ─── Deny-list shape ───────────────────────────────────────────────────────


class TestDenyLists:
    def test_us_in_blocked_countries(self):
        assert "US" in BLOCKED_COUNTRIES

    def test_uk_in_blocked_countries(self):
        assert "GB" in BLOCKED_COUNTRIES

    def test_ofac_sanctioned_countries_present(self):
        for code in ("CU", "IR", "KP", "RU", "SY", "BY"):
            assert code in BLOCKED_COUNTRIES, f"Missing OFAC code {code}"

    def test_quebec_in_blocked_regions(self):
        assert "CA-QC" in BLOCKED_REGIONS

    def test_health_and_terms_in_allowed_paths(self):
        for path in ("/health", "/api/v1/health", "/api/v1/terms", "/api/v1/privacy"):
            assert path in ALLOWED_PATHS


# ─── Allow path: requests without country header ──────────────────────────


class TestUnknownCountry:
    def test_no_country_header_is_allowed(self):
        app = _make_app(disabled=False)
        client = TestClient(app)
        r = client.get("/api/v1/private")
        assert r.status_code == 200
        assert r.json() == {"private": True}


# ─── Block path: blocked country ──────────────────────────────────────────


class TestBlockedCountry:
    def test_us_via_cf_header_returns_451(self):
        app = _make_app(disabled=False)
        client = TestClient(app)
        r = client.get("/api/v1/private", headers={"CF-IPCountry": "US"})
        assert r.status_code == 451
        body = r.json()
        assert body["error"] == "geo_blocked"
        assert body["country"] == "US"
        assert "/api/v1/terms" in body["reference"]

    def test_uk_via_cloudfront_header_blocked(self):
        app = _make_app(disabled=False)
        client = TestClient(app)
        r = client.get(
            "/api/v1/private",
            headers={"CloudFront-Viewer-Country": "GB"},
        )
        assert r.status_code == 451
        assert r.json()["country"] == "GB"

    def test_ofac_country_blocked(self):
        app = _make_app(disabled=False)
        client = TestClient(app)
        r = client.get("/api/v1/private", headers={"X-Country-Code": "IR"})
        assert r.status_code == 451
        assert r.json()["country"] == "IR"

    def test_lowercase_header_normalised(self):
        app = _make_app(disabled=False)
        client = TestClient(app)
        r = client.get("/api/v1/private", headers={"X-Country-Code": "us"})
        assert r.status_code == 451
        assert r.json()["country"] == "US"

    def test_allowed_country_passes(self):
        app = _make_app(disabled=False)
        client = TestClient(app)
        r = client.get("/api/v1/private", headers={"CF-IPCountry": "FR"})
        assert r.status_code == 200


# ─── Block path: Quebec ───────────────────────────────────────────────────


class TestQuebecBlocking:
    def test_canada_quebec_blocked_via_cloudfront_pair(self):
        # Realistic CloudFront shape: country in one header, region in the other.
        app = _make_app(disabled=False)
        client = TestClient(app)
        r = client.get(
            "/api/v1/private",
            headers={
                "CloudFront-Viewer-Country": "CA",
                "CloudFront-Viewer-Country-Region": "QC",
            },
        )
        assert r.status_code == 451
        assert r.json()["country"] == "CA-QC"

    def test_canada_quebec_blocked_via_full_iso_form(self):
        # Tests with explicit "CA-QC" in the region header should also work.
        app = _make_app(disabled=False)
        client = TestClient(app)
        r = client.get(
            "/api/v1/private",
            headers={
                "CloudFront-Viewer-Country": "CA",
                "CloudFront-Viewer-Country-Region": "CA-QC",
            },
        )
        assert r.status_code == 451

    def test_canada_other_provinces_pass(self):
        app = _make_app(disabled=False)
        client = TestClient(app)
        r = client.get(
            "/api/v1/private",
            headers={
                "CloudFront-Viewer-Country": "CA",
                "CloudFront-Viewer-Country-Region": "ON",
            },
        )
        assert r.status_code == 200


# ─── Allowlisted paths bypass the check ───────────────────────────────────


class TestAllowlist:
    def test_terms_reachable_from_blocked_country(self):
        app = _make_app(disabled=False)
        client = TestClient(app)
        r = client.get("/api/v1/terms", headers={"CF-IPCountry": "US"})
        assert r.status_code == 200

    def test_health_reachable_from_blocked_country(self):
        app = _make_app(disabled=False)
        client = TestClient(app)
        r = client.get("/api/v1/health", headers={"CF-IPCountry": "RU"})
        assert r.status_code == 200


# ─── Disabled flag bypasses everything ────────────────────────────────────


class TestDisabledFlag:
    def test_disabled_via_kwarg(self):
        app = _make_app(disabled=True)
        client = TestClient(app)
        r = client.get("/api/v1/private", headers={"CF-IPCountry": "US"})
        assert r.status_code == 200

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("GEO_BLOCK_DISABLED", "1")
        app = _make_app()
        client = TestClient(app)
        r = client.get("/api/v1/private", headers={"CF-IPCountry": "US"})
        assert r.status_code == 200


# ─── Custom resolver injection ────────────────────────────────────────────


class TestCustomResolver:
    def test_resolver_can_force_block(self):
        app = _make_app(country_resolver=lambda req: "US", disabled=False)
        client = TestClient(app)
        r = client.get("/api/v1/private")
        assert r.status_code == 451

    def test_resolver_can_force_pass(self):
        app = _make_app(country_resolver=lambda req: "FR", disabled=False)
        client = TestClient(app)
        r = client.get("/api/v1/private", headers={"CF-IPCountry": "US"})
        # Resolver wins over the header
        assert r.status_code == 200

    def test_extra_countries_via_env(self, monkeypatch):
        monkeypatch.setenv("GEO_BLOCK_EXTRA_COUNTRIES", "DE,AT")
        app = _make_app(disabled=False)
        client = TestClient(app)
        r = client.get("/api/v1/private", headers={"CF-IPCountry": "DE"})
        assert r.status_code == 451
        r = client.get("/api/v1/private", headers={"CF-IPCountry": "FR"})
        assert r.status_code == 200
