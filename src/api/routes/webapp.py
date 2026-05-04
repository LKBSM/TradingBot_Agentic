"""Server-side rendered insight preview — Sprint UX-2B.1 (slice).

GET /api/v1/insights/preview
----------------------------
Body via query: instrument, timeframe, direction, levels, language. The
endpoint runs the same enrichment pipeline as ``/api/v1/enrich`` but
returns a self-contained, accessibility-first HTML document instead of
JSON. Useful for embedding in webapps that don't have a JS rendering
layer yet, and for email templates.

Design constraints
------------------
- ZERO JavaScript. Server-side rendering only.
- Inline ``<style>`` (no external assets) so the document survives
  email clients that strip ``<link>`` tags.
- Semantic HTML5 (``article``, ``section``, ``aside``, ``footer``).
- WCAG AA contrast on the default palette.
- Uses ``html.escape`` everywhere user-controllable text reaches the
  template, so a malicious ``broker_context`` cannot inject markup.
- Fully internationalised (FR / EN / DE / ES) — labels pulled from a
  per-language LABELS table.

The full client-side webapp (UX-2B.1 main scope, 18h) builds on this
contract: same data, richer interactions on top.
"""

from __future__ import annotations

import html
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from src.api.auth import require_api_key
from src.api.disclaimers import get_disclaimer
from src.api.models import EnrichRequest

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/insights", tags=["webapp"])


# ---------------------------------------------------------------------------
# Localisation
# ---------------------------------------------------------------------------


LABELS: dict[str, dict[str, str]] = {
    "fr": {
        "title": "Aperçu de l'analyse",
        "instrument": "Instrument",
        "timeframe": "Horizon",
        "direction": "Setup",
        "conviction": "Conviction",
        "levels": "Niveaux",
        "entry": "Entrée",
        "stop": "Stop",
        "target_1": "Cible 1",
        "target_2": "Cible 2",
        "narrative": "Analyse",
        "sources": "Sources citées",
        "compliance": "Conformité",
        "no_levels": "Aucun niveau publié pour ce setup neutre.",
        "no_sources": "Aucune source citée.",
    },
    "en": {
        "title": "Insight preview",
        "instrument": "Instrument",
        "timeframe": "Timeframe",
        "direction": "Setup",
        "conviction": "Conviction",
        "levels": "Levels",
        "entry": "Entry",
        "stop": "Stop",
        "target_1": "Target 1",
        "target_2": "Target 2",
        "narrative": "Analysis",
        "sources": "Cited sources",
        "compliance": "Compliance",
        "no_levels": "No levels published for this neutral setup.",
        "no_sources": "No sources cited.",
    },
    "de": {
        "title": "Insight-Vorschau",
        "instrument": "Instrument",
        "timeframe": "Zeitrahmen",
        "direction": "Setup",
        "conviction": "Überzeugung",
        "levels": "Niveaus",
        "entry": "Einstieg",
        "stop": "Stop",
        "target_1": "Ziel 1",
        "target_2": "Ziel 2",
        "narrative": "Analyse",
        "sources": "Zitierte Quellen",
        "compliance": "Compliance",
        "no_levels": "Keine Niveaus für dieses neutrale Setup veröffentlicht.",
        "no_sources": "Keine Quellen zitiert.",
    },
    "es": {
        "title": "Vista previa del análisis",
        "instrument": "Instrumento",
        "timeframe": "Marco temporal",
        "direction": "Setup",
        "conviction": "Convicción",
        "levels": "Niveles",
        "entry": "Entrada",
        "stop": "Stop",
        "target_1": "Objetivo 1",
        "target_2": "Objetivo 2",
        "narrative": "Análisis",
        "sources": "Fuentes citadas",
        "compliance": "Cumplimiento",
        "no_levels": "No se publican niveles para este setup neutro.",
        "no_sources": "Sin fuentes citadas.",
    },
}


# ---------------------------------------------------------------------------
# CSS (inline)
# ---------------------------------------------------------------------------


# WCAG AA contrast: bg #ffffff vs text #1a1f29 (ratio 14.7:1).
# Card background #f8fafc with #1a1f29 text (ratio 13.8:1).
_INLINE_CSS = """
:root {
  --bg: #ffffff;
  --bg-card: #f8fafc;
  --bg-source: #fef3c7;
  --text: #1a1f29;
  --text-muted: #475569;
  --accent-bull: #047857;
  --accent-bear: #b91c1c;
  --accent-neutral: #475569;
  --border: #e2e8f0;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 16px;
  line-height: 1.6;
  margin: 0; padding: 1.5rem;
  background: var(--bg); color: var(--text);
}
.insight-preview {
  max-width: 720px; margin: 0 auto;
}
header h1 {
  font-size: 1.5rem; margin: 0 0 0.5rem 0;
}
header p {
  color: var(--text-muted); margin: 0 0 1.5rem 0;
}
.kv-grid {
  display: grid; grid-template-columns: max-content 1fr;
  gap: 0.5rem 1.25rem; padding: 1rem;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 8px;
}
.kv-grid dt { color: var(--text-muted); font-weight: 500; }
.kv-grid dd { margin: 0; font-weight: 600; }
.direction-bull { color: var(--accent-bull); }
.direction-bear { color: var(--accent-bear); }
.direction-neutral { color: var(--accent-neutral); }
section { margin-top: 2rem; }
section h2 {
  font-size: 1.1rem; margin: 0 0 0.75rem 0;
  border-bottom: 2px solid var(--border); padding-bottom: 0.25rem;
}
.narrative {
  white-space: pre-wrap;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 8px; padding: 1rem;
}
.sources {
  list-style: none; padding: 0; margin: 0;
}
.sources li {
  background: var(--bg-source); padding: 0.6rem 0.9rem;
  border-radius: 6px; margin-bottom: 0.5rem;
  font-size: 0.95rem;
}
.sources li .src-label { font-weight: 600; }
.sources li .src-type {
  color: var(--text-muted); margin-left: 0.5rem;
  text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.04em;
}
footer.disclaimer {
  margin-top: 2.5rem; padding: 1rem;
  background: #fff7ed; border-left: 4px solid #f59e0b;
  font-size: 0.85rem; color: #78350f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a; --bg-card: #1e293b; --bg-source: #422006;
    --text: #f1f5f9; --text-muted: #94a3b8; --border: #334155;
  }
}
"""


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _direction_class(direction: str) -> str:
    if direction == "BULLISH_SETUP":
        return "direction-bull"
    if direction == "BEARISH_SETUP":
        return "direction-bear"
    return "direction-neutral"


def _render_levels(payload, lbl: dict[str, str]) -> str:
    levels = payload.levels
    if all(
        v is None
        for v in (levels.entry, levels.stop, levels.target_1, levels.target_2)
    ):
        return f'<p class="muted">{html.escape(lbl["no_levels"])}</p>'
    rows = []
    for key in ("entry", "stop", "target_1", "target_2"):
        v = getattr(levels, key)
        if v is not None:
            rows.append(
                f'<dt>{html.escape(lbl[key])}</dt>'
                f'<dd>{html.escape(f"{v:.5f}".rstrip("0").rstrip("."))}</dd>'
            )
    return f'<dl class="kv-grid">{"".join(rows)}</dl>'


def _render_sources(payload, lbl: dict[str, str]) -> str:
    if not payload.sources_cited:
        return f'<p class="muted">{html.escape(lbl["no_sources"])}</p>'
    items = []
    for src in payload.sources_cited:
        items.append(
            "<li>"
            f'<span class="src-label">{html.escape(src.label)}</span>'
            f'<span class="src-type">{html.escape(src.type.value)}</span>'
            "</li>"
        )
    return f'<ul class="sources">{"".join(items)}</ul>'


def _render_html(payload, language: str) -> str:
    lbl = LABELS.get(language, LABELS["en"])
    direction_cls = _direction_class(payload.direction.value)
    return f"""<!DOCTYPE html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(lbl["title"])} — {html.escape(payload.instrument)}</title>
<style>{_INLINE_CSS}</style>
</head>
<body>
<article class="insight-preview" lang="{language}">
<header>
  <h1>{html.escape(lbl["title"])}</h1>
  <p>{html.escape(payload.narrative_short)}</p>
</header>

<dl class="kv-grid">
  <dt>{html.escape(lbl["instrument"])}</dt>
  <dd>{html.escape(payload.instrument)}</dd>
  <dt>{html.escape(lbl["timeframe"])}</dt>
  <dd>{html.escape(payload.timeframe.value)}</dd>
  <dt>{html.escape(lbl["direction"])}</dt>
  <dd class="{direction_cls}">{html.escape(payload.direction.value.replace("_", " ").title())}</dd>
  <dt>{html.escape(lbl["conviction"])}</dt>
  <dd>{int(payload.conviction_0_100)} / 100</dd>
</dl>

<section aria-labelledby="levels-title">
  <h2 id="levels-title">{html.escape(lbl["levels"])}</h2>
  {_render_levels(payload, lbl)}
</section>

<section aria-labelledby="narrative-title">
  <h2 id="narrative-title">{html.escape(lbl["narrative"])}</h2>
  <div class="narrative">{html.escape(payload.narrative_long)}</div>
</section>

<section aria-labelledby="sources-title">
  <h2 id="sources-title">{html.escape(lbl["sources"])}</h2>
  {_render_sources(payload, lbl)}
</section>

<footer class="disclaimer" role="contentinfo">
  {html.escape(get_disclaimer(language))}
</footer>
</article>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/preview",
    response_class=HTMLResponse,
    responses={
        503: {"description": "RAG pipeline not configured"},
        422: {"description": "Invalid query parameters"},
    },
)
async def preview(
    request: Request,
    instrument: str = Query(..., min_length=2, max_length=12),
    timeframe: str = Query(..., pattern=r"^(M1|M5|M15|M30|H1|H4|D1|W1)$"),
    direction: str = Query(..., pattern=r"^(BULLISH_SETUP|BEARISH_SETUP|NEUTRAL)$"),
    entry: float | None = Query(default=None, gt=0),
    stop: float | None = Query(default=None, gt=0),
    target_1: float | None = Query(default=None, gt=0),
    target_2: float | None = Query(default=None, gt=0),
    language: str = Query(default="en", pattern=r"^(fr|en|de|es)$"),
    subscriber: dict = Depends(require_api_key),
) -> HTMLResponse:
    # Reuse the /enrich pipeline (single source of truth) by calling the
    # underlying handler. Avoid HTTP self-referral overhead by importing
    # the function directly.
    from src.api.routes.enrich import enrich as enrich_handler

    body = EnrichRequest(
        instrument=instrument,
        timeframe=timeframe,
        direction=direction,
        entry=entry,
        stop=stop,
        target_1=target_1,
        target_2=target_2,
        language=language,
    )
    payload = await enrich_handler(body, request, subscriber)
    return HTMLResponse(content=_render_html(payload, language))
