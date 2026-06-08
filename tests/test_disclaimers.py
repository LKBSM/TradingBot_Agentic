"""Tests for src/api/disclaimers.py — multi-language risk disclosures."""

from __future__ import annotations

import pytest

from src.api.disclaimers import (
    DEFAULT_LANG,
    SUPPORTED_LANGS,
    detect_language_from_request,
    get_disclaimer,
    get_footer,
)


class TestSupportedLanguages:
    def test_four_languages_supported(self):
        assert set(SUPPORTED_LANGS) == {"fr", "en", "de", "es"}

    def test_default_language_is_supported(self):
        assert DEFAULT_LANG in SUPPORTED_LANGS


class TestGetDisclaimer:
    @pytest.mark.parametrize("lang", ["fr", "en", "de", "es"])
    def test_each_language_returns_non_empty_string(self, lang):
        text = get_disclaimer(lang)
        assert isinstance(text, str)
        assert len(text) > 50

    def test_french_mentions_conseil(self):
        assert "conseil" in get_disclaimer("fr").lower()

    def test_english_mentions_advice(self):
        assert "advice" in get_disclaimer("en").lower()

    def test_german_mentions_anlageberatung(self):
        assert "anlageberatung" in get_disclaimer("de").lower()

    def test_spanish_mentions_asesoramiento(self):
        assert "asesoramiento" in get_disclaimer("es").lower()

    def test_all_disclaimers_mention_risk_percentage(self):
        for lang in SUPPORTED_LANGS:
            assert "74" in get_disclaimer(lang)

    def test_all_disclaimers_link_to_terms(self):
        for lang in SUPPORTED_LANGS:
            assert "/api/v1/terms" in get_disclaimer(lang)

    def test_unknown_language_falls_back(self):
        text = get_disclaimer("xx")
        assert text == get_disclaimer(DEFAULT_LANG)

    def test_none_falls_back(self):
        assert get_disclaimer(None) == get_disclaimer(DEFAULT_LANG)


class TestGetFooter:
    def test_footer_short_enough_for_telegram(self):
        # Telegram-friendly: well below the 4096 limit, with room for body.
        for lang in SUPPORTED_LANGS:
            assert len(get_footer(lang)) < 200

    def test_footer_mentions_smart_sentinel(self):
        for lang in SUPPORTED_LANGS:
            assert "Smart Sentinel" in get_footer(lang)


class TestDetectLanguageFromRequest:
    def test_no_headers_returns_default(self):
        assert detect_language_from_request(None) == DEFAULT_LANG
        assert detect_language_from_request({}) == DEFAULT_LANG

    def test_french_accept_language(self):
        assert detect_language_from_request({"accept-language": "fr-FR,fr;q=0.9"}) == "fr"

    def test_german_accept_language(self):
        assert detect_language_from_request({"Accept-Language": "de-DE,de;q=0.9,en;q=0.5"}) == "de"

    def test_unknown_language_falls_back_default(self):
        assert detect_language_from_request({"accept-language": "ja-JP"}) == DEFAULT_LANG

    def test_first_supported_language_wins(self):
        # English (supported) before German (also supported) — first wins
        assert detect_language_from_request({"accept-language": "en;q=1.0,de;q=0.5"}) == "en"
