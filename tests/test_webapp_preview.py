"""Tests for the UX-2B.1 webapp insight preview endpoint."""

from __future__ import annotations

import os

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes.qa import build_default_rag_pipeline


@pytest.fixture(autouse=True)
def _force_testing_mode():
    with patch("src.api.auth.TESTING_MODE", True), patch(
        "src.api.routes.enrich.TESTING_MODE", True
    ):
        yield


@pytest.fixture(scope="module")
def populated_pipeline():
    return build_default_rag_pipeline()


@pytest.fixture
def client(populated_pipeline):
    return TestClient(create_app(rag_pipeline=populated_pipeline))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


_BASE_QUERY = {
    "instrument": "XAUUSD",
    "timeframe": "M15",
    "direction": "BULLISH_SETUP",
    "entry": 2350.0,
    "stop": 2340.0,
    "target_1": 2370.0,
    "language": "en",
}


def test_preview_returns_html_doctype(client):
    resp = client.get("/api/v1/insights/preview", params=_BASE_QUERY)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    text = resp.text
    assert text.startswith("<!DOCTYPE html>")
    assert '<html lang="en">' in text


def test_preview_contains_semantic_html5_tags(client):
    resp = client.get("/api/v1/insights/preview", params=_BASE_QUERY)
    text = resp.text
    for tag in ("<article", "<header", "<section", "<footer"):
        assert tag in text


def test_preview_renders_levels_grid(client):
    resp = client.get("/api/v1/insights/preview", params=_BASE_QUERY)
    text = resp.text
    assert "Entry" in text
    assert "Stop" in text
    assert "Target 1" in text
    assert "2350" in text


def test_preview_includes_disclaimer(client):
    resp = client.get("/api/v1/insights/preview", params=_BASE_QUERY)
    assert "investment advice" in resp.text.lower()


@pytest.mark.parametrize(
    "lang,marker",
    [
        ("fr", "Aperçu"),
        ("en", "Insight preview"),
        ("de", "Insight-Vorschau"),
        ("es", "Vista previa"),
    ],
)
def test_preview_localises_title(client, lang, marker):
    params = dict(_BASE_QUERY, language=lang)
    resp = client.get("/api/v1/insights/preview", params=params)
    assert marker in resp.text
    assert f'<html lang="{lang}">' in resp.text


def test_preview_neutral_setup_drops_levels(client):
    params = {
        "instrument": "XAUUSD",
        "timeframe": "H1",
        "direction": "NEUTRAL",
        "language": "en",
    }
    resp = client.get("/api/v1/insights/preview", params=params)
    assert resp.status_code == 200
    # Neutral payload should show the "no levels" copy.
    assert "No levels published" in resp.text


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


def test_preview_escapes_html_in_instrument(client):
    """The instrument is reflected into the title — escape XSS attempts."""
    params = dict(_BASE_QUERY, instrument="XAU<sc")  # 6 chars, fits min_length
    resp = client.get("/api/v1/insights/preview", params=params)
    assert resp.status_code == 200
    # Raw < / > should not pass through to HTML — must be escaped.
    assert "<sc" not in resp.text
    assert "&lt;sc" in resp.text


def test_preview_no_inline_script_tag(client):
    """Sanity: there is no <script> tag in our output (no JS dependency)."""
    resp = client.get("/api/v1/insights/preview", params=_BASE_QUERY)
    assert "<script" not in resp.text.lower()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_preview_invalid_timeframe_rejected(client):
    params = dict(_BASE_QUERY, timeframe="M2")
    resp = client.get("/api/v1/insights/preview", params=params)
    assert resp.status_code == 422


def test_preview_invalid_language_rejected(client):
    params = dict(_BASE_QUERY, language="zz")
    resp = client.get("/api/v1/insights/preview", params=params)
    assert resp.status_code == 422


def test_preview_503_when_pipeline_missing():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/v1/insights/preview", params=_BASE_QUERY)
    assert resp.status_code == 503
