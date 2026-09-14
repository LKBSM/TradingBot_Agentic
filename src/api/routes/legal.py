"""Legal endpoints: Terms of Use & Privacy Policy.

Public, unauthenticated, geo-block-bypassed. Required by:
  * Stripe / payment processors (T&C URL must be reachable);
  * Quebec's Law 25 (information to the person concerned);
  * the consent recorded before checkout (version + timestamp per account).

LEG-1 — SINGLE SOURCE OF TRUTH. Before this mission the legal text lived in
THREE places that contradicted each other (this module's inline dicts, the
canonical markdown, and the i18n bundles). It now lives in exactly one:

    docs/legal/conditions-utilisation.{fr,en,es}.md
    docs/legal/politique-confidentialite.{fr,en,es}.md

Those files are served VERBATIM. Nothing here rewrites, summarises or
paraphrases them — the endpoints only pick the language and stamp the version.
The French file is canonical; the other two are faithful translations and say so
in their own header.

Languages: ``?lang=fr|en|es`` (the three commercialised locales), else the
``Accept-Language`` header, else English. A locale we do not publish a legal
document for (de, it, pt, nl, pl, ar) falls back to English, which the documents
themselves disclose.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from src.api.disclaimers import detect_language_from_request

router = APIRouter(tags=["legal"])

# ─── Version ──────────────────────────────────────────────────────────────
#
# 2026-09-13 (LEG-1): complete rewrite of both documents ahead of taking the
# first payment. Territory narrowed to Canada, prices and billing stated,
# cancellation/refund stated, market-data redistribution barred, the blanket
# liability exclusion removed (unenforceable in Quebec), the EU/RGPD framing
# replaced by Quebec Law 25, contact moved to contact@mia.markets.
#
# Bumping this string is what makes every account re-consent: it is the stamp
# written into ``account_consents`` and rendered as ``X-Document-Version``. It
# MUST stay in lockstep with the date in each markdown header.
LAST_UPDATED = "2026-09-13"

#: Back-compat alias — the canonical document version (same stamp).
CONDITIONS_VERSION = LAST_UPDATED

#: The locales we publish legal documents for. NOT the same list as the app's
#: nine UI locales: a legal text is only published in a language we can vouch
#: for. Everything else is served in English.
LEGAL_LANGS = ("fr", "en", "es")

#: Fallback for any locale outside :data:`LEGAL_LANGS`.
LEGAL_FALLBACK_LANG = "en"

#: Repo root — this file is ``src/api/routes/legal.py``.
_LEGAL_DIR = Path(__file__).resolve().parents[3] / "docs" / "legal"

#: Document key → file stem under ``docs/legal/``.
DOCUMENT_STEMS: Dict[str, str] = {
    "terms": "conditions-utilisation",
    "privacy": "politique-confidentialite",
}


def document_path(doc: str, lang: str) -> Path:
    """Absolute path of the ``doc`` document in ``lang``."""
    return _LEGAL_DIR / f"{DOCUMENT_STEMS[doc]}.{lang}.md"


def _resolve_lang(query_lang: Optional[str], request: Optional[Request]) -> str:
    """Pick a published legal language: query param, then header, then English."""
    if query_lang:
        code = query_lang.strip().lower()[:2]
        if code in LEGAL_LANGS:
            return code
    if request is not None:
        detected = detect_language_from_request(dict(request.headers))
        if detected in LEGAL_LANGS:
            return detected
    return LEGAL_FALLBACK_LANG


def read_document(doc: str, lang: str) -> str:
    """Read a legal document VERBATIM. 503 when the file is missing.

    A missing legal document is an availability failure, never a silent empty
    page: the consent screen links here, so serving nothing would let someone
    accept a text they could not read.
    """
    try:
        return document_path(doc, lang).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Legal document unavailable",
        ) from exc


def _document_response(doc: str, lang: str) -> PlainTextResponse:
    return PlainTextResponse(
        read_document(doc, lang),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Language": lang,
            "X-Document-Version": LAST_UPDATED,
            "Cache-Control": "public, max-age=3600",
        },
    )


# ─── Documents ────────────────────────────────────────────────────────────


@router.get("/api/v1/legal/conditions", response_class=PlainTextResponse)
async def conditions_document(
    request: Request,
    lang: Optional[str] = Query(None, description="Language: fr, en, es"),
):
    """Serve the Terms of Use document verbatim (webapp ``/conditions``)."""
    return _document_response("terms", _resolve_lang(lang, request))


@router.get("/api/v1/legal/privacy", response_class=PlainTextResponse)
async def privacy_document(
    request: Request,
    lang: Optional[str] = Query(None, description="Language: fr, en, es"),
):
    """Serve the Privacy Policy document verbatim (webapp ``/confidentialite``)."""
    return _document_response("privacy", _resolve_lang(lang, request))


# ─── Back-compatible aliases ──────────────────────────────────────────────
# Kept because they are public URLs that may already be registered with Stripe
# and elsewhere. They now serve the SAME files as the endpoints above — there is
# no second version of the text to drift out of sync.


@router.get("/api/v1/terms", response_class=PlainTextResponse)
async def terms_of_service(
    request: Request,
    lang: Optional[str] = Query(None, description="Language: fr, en, es"),
):
    """Public Terms of Use. No authentication required."""
    return _document_response("terms", _resolve_lang(lang, request))


@router.get("/api/v1/privacy", response_class=PlainTextResponse)
async def privacy_policy(
    request: Request,
    lang: Optional[str] = Query(None, description="Language: fr, en, es"),
):
    """Public Privacy Policy. No authentication required."""
    return _document_response("privacy", _resolve_lang(lang, request))


# ─── Machine-readable metadata ────────────────────────────────────────────


@router.get("/api/v1/legal/version")
async def legal_version():
    """Version stamp for clients to detect a legal-document update."""
    return {
        "terms_version": LAST_UPDATED,
        "privacy_version": LAST_UPDATED,
        "conditions_version": CONDITIONS_VERSION,
        "supported_languages": list(LEGAL_LANGS),
        "fallback_language": LEGAL_FALLBACK_LANG,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/legal/conditions/meta")
async def conditions_meta():
    """Version + date for the canonical Terms document (machine-readable)."""
    return {
        "version": CONDITIONS_VERSION,
        "last_updated": LAST_UPDATED,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
