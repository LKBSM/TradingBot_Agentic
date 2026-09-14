"""Tests for src/api/routes/legal.py — public Terms & Privacy endpoints.

LEG-1 rewrote both documents and made ``docs/legal/*.{fr,en,es}.md`` their only
source. The assertions here deliberately encode the DECISIONS taken in that
mission, so a silent drift back to the superseded text fails the suite:

  * no blanket liability exclusion (unenforceable in Quebec);
  * no "final sale" / "non-refundable" wording, in any language;
  * Quebec law, not the EU/AMF/CNIL framing that used to be here;
  * contact@mia.markets, not a personal mailbox;
  * the market-data redistribution bar (our data licence requires it).
"""

from __future__ import annotations

import re

import pytest
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


@pytest.fixture()
def client() -> TestClient:
    return TestClient(_make_app())


DOC_ROUTES = ("/api/v1/legal/conditions", "/api/v1/legal/privacy")
ALIAS_ROUTES = ("/api/v1/terms", "/api/v1/privacy")


# =============================================================================
# Language resolution
# =============================================================================

class TestLanguageResolution:
    @pytest.mark.parametrize("route", DOC_ROUTES + ALIAS_ROUTES)
    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_each_published_language_is_served(self, client, route, lang):
        r = client.get(f"{route}?lang={lang}")
        assert r.status_code == 200
        assert r.headers["content-language"] == lang
        assert len(r.text) > 500

    def test_unpublished_language_falls_back_to_english(self, client):
        # de/it/pt/nl/pl/ar are shipped UI locales but we publish no legal text
        # in them — English is served, and the document itself says so.
        for lang in ("de", "it", "pt", "nl", "pl", "ar", "ja"):
            r = client.get(f"/api/v1/legal/conditions?lang={lang}")
            assert r.status_code == 200
            assert r.headers["content-language"] == legal.LEGAL_FALLBACK_LANG

    def test_query_param_beats_accept_language(self, client):
        r = client.get(
            "/api/v1/legal/conditions?lang=fr", headers={"Accept-Language": "en"}
        )
        assert r.headers["content-language"] == "fr"

    def test_accept_language_used_when_no_query(self, client):
        r = client.get(
            "/api/v1/legal/conditions", headers={"Accept-Language": "fr-CA,fr;q=0.9"}
        )
        assert r.headers["content-language"] == "fr"

    def test_content_type_is_markdown(self, client):
        r = client.get("/api/v1/legal/conditions?lang=fr")
        assert "text/markdown" in r.headers["content-type"]

    def test_version_header_on_every_document(self, client):
        for route in DOC_ROUTES + ALIAS_ROUTES:
            r = client.get(f"{route}?lang=fr")
            assert r.headers["X-Document-Version"] == legal.LAST_UPDATED


# =============================================================================
# One source of truth
# =============================================================================

class TestSingleSource:
    def test_documents_are_served_verbatim_from_disk(self, client):
        for doc, route in (("terms", "/api/v1/legal/conditions"),
                           ("privacy", "/api/v1/legal/privacy")):
            for lang in legal.LEGAL_LANGS:
                on_disk = legal.document_path(doc, lang).read_text(encoding="utf-8")
                assert client.get(f"{route}?lang={lang}").text == on_disk

    def test_alias_routes_serve_the_same_bytes(self, client):
        for canonical, alias in (
            ("/api/v1/legal/conditions", "/api/v1/terms"),
            ("/api/v1/legal/privacy", "/api/v1/privacy"),
        ):
            for lang in legal.LEGAL_LANGS:
                assert (
                    client.get(f"{canonical}?lang={lang}").text
                    == client.get(f"{alias}?lang={lang}").text
                )

    def test_every_published_document_exists(self):
        for doc in legal.DOCUMENT_STEMS:
            for lang in legal.LEGAL_LANGS:
                assert legal.document_path(doc, lang).is_file()

    def test_markdown_header_version_matches_the_code(self):
        # The consent stamp and the rendered version can never drift: every
        # document header carries LAST_UPDATED.
        for doc in legal.DOCUMENT_STEMS:
            for lang in legal.LEGAL_LANGS:
                head = legal.document_path(doc, lang).read_text(encoding="utf-8")[:400]
                assert legal.LAST_UPDATED in head

    def test_missing_document_is_503_not_an_empty_page(self, client, monkeypatch):
        monkeypatch.setattr(
            legal, "_LEGAL_DIR", legal._LEGAL_DIR / "does-not-exist"
        )
        assert client.get("/api/v1/legal/conditions?lang=fr").status_code == 503


# =============================================================================
# What the text must (and must not) say — LEG-1 decisions
# =============================================================================

def _terms(client, lang: str) -> str:
    return client.get(f"/api/v1/legal/conditions?lang={lang}").text


def _privacy(client, lang: str) -> str:
    return client.get(f"/api/v1/legal/privacy?lang={lang}").text


class TestTermsContent:
    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_states_it_is_not_advice_and_gives_no_signal(self, client, lang):
        body = _terms(client, lang).lower()
        assert any(w in body for w in ("conseil", "advice", "asesoramiento"))
        assert any(w in body for w in ("signal", "señal"))

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_states_the_risk_of_loss_and_leverage(self, client, lang):
        body = _terms(client, lang).lower()
        assert any(w in body for w in ("perte", "loss", "pérdida"))
        assert any(w in body for w in ("levier", "leverage", "apalancamiento"))

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_states_the_age_requirement(self, client, lang):
        assert "18" in _terms(client, lang)

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_states_the_territory_canada_and_united_states(self, client, lang):
        # LEG-1 (2026-09-14) — décision fondateur : Canada ET États-Unis.
        body = _terms(client, lang)
        assert any(w in body for w in ("Canada", "Canadá"))
        assert any(w in body for w in ("États-Unis", "United\nStates", "United States", "Estados Unidos"))

    def test_the_texts_and_the_geo_block_agree_on_the_territory(self):
        # Le texte ne doit jamais annoncer un territoire que le code refuse.
        from src.api.middleware.geo_block import BLOCKED_COUNTRIES

        assert "US" not in BLOCKED_COUNTRIES
        assert "CA" not in BLOCKED_COUNTRIES

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_bars_redistribution_of_market_data(self, client, lang):
        # Mandatory: this clause is what our data licence requires us to pass on.
        body = _terms(client, lang).lower()
        assert any(w in body for w in ("redistribuer", "redistribute", "redistribuir"))

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_states_price_currency_and_stripe(self, client, lang):
        body = _terms(client, lang)
        assert "39" in body and "348" in body
        assert "USD" in body
        assert "Stripe" in body

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_states_the_14_day_annual_guarantee_and_quebec_rights(self, client, lang):
        body = _terms(client, lang)
        assert "14" in body
        assert "protection du consommateur" in body  # named in all three

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_quebec_is_the_governing_law(self, client, lang):
        body = _terms(client, lang)
        assert "Quebec" in body or "Québec" in body

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_contact_is_the_company_mailbox(self, client, lang):
        assert "contact@mia.markets" in _terms(client, lang)
        assert "loukmanebessam@gmail.com" not in _terms(client, lang)

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_no_mandatory_arbitration_or_class_action_waiver(self, client, lang):
        body = _terms(client, lang).lower()
        for banned in ("arbitrage obligatoire", "mandatory arbitration",
                       "class action", "action collective", "acción colectiva"):
            assert banned not in body


class TestPrivacyContent:
    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_names_the_actual_subprocessors(self, client, lang):
        body = _privacy(client, lang)
        assert "Stripe" in body
        assert "Anthropic" in body
        assert "TwelveData" in body

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_does_not_name_processors_we_do_not_use(self, client, lang):
        # Clerk was explicitly rejected (AUDIT-pay-1) and Telegram is not part of
        # the subscription product. Naming either would be a false declaration.
        body = _privacy(client, lang)
        assert "Clerk" not in body
        assert "Telegram" not in body

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_discloses_that_agent_questions_are_sent_to_anthropic(self, client, lang):
        body = _privacy(client, lang).lower()
        assert any(w in body for w in ("questions", "preguntas"))

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_names_the_hosting_country_and_the_quebec_regulator(self, client, lang):
        body = _privacy(client, lang)
        assert any(w in body for w in ("États-Unis", "United States", "Estados Unidos"))
        assert "Commission d'accès à l'information" in body

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_mentions_the_incident_register(self, client, lang):
        body = _privacy(client, lang).lower()
        assert any(w in body for w in ("registre", "register", "registro"))

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_does_not_claim_a_completed_privacy_assessment(self, client, lang):
        # No PIA has been carried out yet. The text may say one is under way; it
        # must never say one "was carried out".
        body = _privacy(client, lang).lower()
        for false_claim in ("a été réalisée", "has been carried out",
                            "has been completed", "ha sido realizada",
                            "se ha realizado"):
            assert false_claim not in body

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_states_no_card_data_is_kept(self, client, lang):
        body = _privacy(client, lang).lower()
        assert any(w in body for w in ("carte", "card", "tarjeta"))


class TestForbiddenWording:
    """No document, in any language, may harden the refund promise."""

    FORBIDDEN = (
        "vente finale", "non remboursable", "non-remboursable",
        "final sale", "no refund", "non-refundable", "nonrefundable",
        "venta final", "sin reembolso", "no reembolsable",
    )

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_no_hardened_refund_wording(self, client, lang):
        for doc_route in DOC_ROUTES:
            body = client.get(f"{doc_route}?lang={lang}").text.lower()
            for banned in self.FORBIDDEN:
                assert banned not in body, f"{doc_route} [{lang}] contient « {banned} »"

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_no_blanket_liability_exclusion(self, client, lang):
        # The superseded §6 excluded liability outright. Unenforceable in Quebec
        # and forbidden by the mission — it must not come back.
        body = _terms(client, lang).lower()
        for banned in (
            "ne saurait être tenue responsable",
            "décline toute responsabilité",
            "shall not be liable",
            "no se hace responsable",
            "declina toda responsabilidad",
        ):
            assert banned not in body

    @pytest.mark.parametrize("lang", legal.LEGAL_LANGS)
    def test_no_predictive_future_about_markets(self, client, lang):
        # Product speech may say what the tool does NOT do, never what a market
        # is going to do.
        body = _terms(client, lang).lower() + _privacy(client, lang).lower()
        predictive = (
            r"\bva (?:monter|baisser|augmenter|chuter)\b",
            r"\bvont (?:monter|baisser)\b",
            r"\bprice will (?:rise|fall|drop|increase)\b",
            r"\bwill (?:rise|fall|outperform)\b",
            r"\b(?:subirá|bajará|caerá)\b",
        )
        for pattern in predictive:
            assert not re.search(pattern, body), f"[{lang}] futur prédictif : {pattern}"

    def test_no_unverifiable_loss_statistic(self, client):
        # "74% to 89% of retail accounts lose money" was an ESMA figure, unsourced
        # here and inapplicable to our market. Removed, must stay removed.
        for lang in legal.LEGAL_LANGS:
            body = _terms(client, lang)
            assert "74" not in body
            assert "89" not in body


# =============================================================================
# Reachability & metadata
# =============================================================================

class TestReachability:
    @pytest.mark.parametrize("route", DOC_ROUTES + ALIAS_ROUTES)
    def test_reachable_even_from_a_geo_blocked_country(self, route):
        # A legal document must open for anyone — including someone we cannot
        # sell to. Serving 451 on the terms page would hide the reason.
        client = TestClient(_make_app(geo_blocked=True))
        r = client.get(route, headers={"CF-IPCountry": "US"})
        assert r.status_code == 200


class TestLegalVersion:
    def test_version_endpoint(self, client):
        data = client.get("/api/v1/legal/version").json()
        assert data["terms_version"] == legal.LAST_UPDATED
        assert data["privacy_version"] == legal.LAST_UPDATED
        assert data["conditions_version"] == legal.LAST_UPDATED
        assert set(data["supported_languages"]) == {"fr", "en", "es"}
        assert data["fallback_language"] == "en"

    def test_conditions_meta(self, client):
        data = client.get("/api/v1/legal/conditions/meta").json()
        assert data["version"] == legal.LAST_UPDATED
        assert data["last_updated"] == legal.LAST_UPDATED
