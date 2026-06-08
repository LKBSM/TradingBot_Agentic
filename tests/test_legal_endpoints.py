"""Tests for src/api/routes/legal.py — public Terms & Privacy endpoints."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware.geo_block import GeoBlockMiddleware
from src.api.routes import legal


def _make_app(geo_blocked: bool = False) -> FastAPI:
    app = FastAPI()
    if geo_blocked:
        app.add_middleware(GeoBlockMiddleware, disabled=False)
    app.include_router(legal.router)
    return app


class TestTermsEndpoint:
    def test_default_language_is_english(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/terms")
        assert r.status_code == 200
        assert "Terms of Service" in r.text
        assert r.headers["content-language"] == "en"

    def test_french_via_query(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/terms?lang=fr")
        assert r.status_code == 200
        assert "Conditions Générales" in r.text
        assert r.headers["content-language"] == "fr"

    def test_german_via_query(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/terms?lang=de")
        assert "Nutzungsbedingungen" in r.text

    def test_spanish_via_query(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/terms?lang=es")
        assert "Términos de Servicio" in r.text

    def test_unknown_language_falls_back(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/terms?lang=ja")
        assert r.status_code == 200
        assert "Terms of Service" in r.text

    def test_query_param_beats_accept_language(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/terms?lang=fr", headers={"Accept-Language": "de"})
        assert "Conditions Générales" in r.text

    def test_accept_language_used_when_no_query(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/terms", headers={"Accept-Language": "fr-FR,fr;q=0.9"})
        assert "Conditions Générales" in r.text

    def test_content_type_is_markdown(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/terms")
        assert "text/markdown" in r.headers["content-type"]

    def test_terms_mention_us_qc_uk_block(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/terms?lang=en")
        body = r.text
        assert "United States" in body
        assert "Quebec" in body
        assert "United Kingdom" in body

    def test_reachable_even_from_blocked_country(self):
        client = TestClient(_make_app(geo_blocked=True))
        r = client.get("/api/v1/terms", headers={"CF-IPCountry": "US"})
        assert r.status_code == 200


class TestPrivacyEndpoint:
    def test_default_language_is_english(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/privacy")
        assert r.status_code == 200
        assert "Privacy Policy" in r.text

    def test_french_translation(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/privacy?lang=fr")
        assert "Politique de confidentialité" in r.text

    def test_mentions_rgpd_articles(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/privacy?lang=fr")
        assert "RGPD" in r.text

    def test_lists_subprocessors(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/privacy?lang=en")
        body = r.text
        assert "Anthropic" in body
        assert "Stripe" in body
        assert "Telegram" in body

    def test_reachable_from_blocked_country(self):
        client = TestClient(_make_app(geo_blocked=True))
        r = client.get("/api/v1/privacy", headers={"CF-IPCountry": "US"})
        assert r.status_code == 200


class TestLegalVersion:
    def test_version_endpoint_returns_dates(self):
        client = TestClient(_make_app())
        r = client.get("/api/v1/legal/version")
        assert r.status_code == 200
        data = r.json()
        assert "terms_version" in data
        assert "privacy_version" in data
        assert "supported_languages" in data
        assert set(data["supported_languages"]) == {"fr", "en", "de", "es"}
