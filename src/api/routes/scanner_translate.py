"""SC-2 — Scanner conversationnel : endpoint de traduction phrase → palette.

POST /api/scanner/translate
  body: { "text": str (1..500), "locale": "fr"|"en" (défaut "fr") }
  → 200 ScannerTranslateResponse :
      { outcome, refusal?, conditions[], condition_sources[], assumptions[],
        untranslatable[] }
  → 422 sur corps invalide (texte vide / trop long)
  → 429 au-delà du plafond SC-4 (40 traductions / 5 min, par compte ET par IP)
  → 503 si le traducteur n'est pas câblé (SCANNER_TRANSLATOR_ENABLED=false ou clé absente)

SC-4 — ``condition_sources`` donne, pour chaque condition et ALIGNÉ PAR INDEX,
le fragment de la phrase de l'utilisateur qui l'a produite, vérifié verbatim
côté serveur (``None`` si la citation était absente du texte : on ne fabrique
jamais une origine). C'est ce qui permet à la saisie en direct de respecter un
retrait manuel sans jamais refuser une condition que l'utilisateur vient de
réécrire.

M.I.A ne reçoit AUCUNE donnée de marché ici : elle traduit une phrase. La sortie
est re-validée contre la palette fermée AVANT de quitter le serveur, puis chaque
condition passe par le modèle ``ScanCondition`` du endpoint de scan — un type
hors palette ou une valeur hors domaine ne peut donc jamais atteindre
l'évaluateur (même verrou que l'``id`` des zones dans le chatbot).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from src.api.auth_throttle import AuthThrottle, client_ip
from src.api.subscription_gate import enforce_access
from src.api.routes.conditions_scan import ScanCondition
from src.api.session_auth import optional_account

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scanner", tags=["scanner-conversational"])

MAX_TEXT_LENGTH = 500

def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer — falling back to %d", name, raw, default)
        return default


# ── SC-4 — a cap on this endpoint specifically ───────────────────────────────
#
# Until SC-4 the only way to spend a translation was to click a button, so human
# patience was the ceiling. SC-4 fires a translation on every natural pause in
# typing, and every one of them is a paid LLM call — so the ceiling has to be in
# the code. It cannot rely on the per-IP limiter in ``create_app``: the deployed
# entrypoint is ``src.api.asgi:app`` -> ``create_app()`` with no arguments, where
# ``rate_limiter is None`` and that middleware is a no-op (the same finding that
# produced ``AuthThrottle``, reused here rather than reinvented).
#
# Keyed on the ACCOUNT when there is one — so a shared office IP does not punish
# a whole team — and ALWAYS also on the IP, so an unauthenticated flood is capped
# too. 40 per 5 minutes: a real live-typing session spends 4-7, so a human never
# meets this; a script meets it immediately. Per-process and in-memory, like
# every other throttle here — a brake, not a global quota (a shared store is the
# next layer). ``TRANSLATE_THROTTLE_MAX=0`` disables it.
_TRANSLATE_THROTTLE = AuthThrottle(
    max_attempts=_int_env("TRANSLATE_THROTTLE_MAX", 40),
    window_s=_int_env("TRANSLATE_THROTTLE_WINDOW_S", 300),
)


def _enforce_translate_throttle(request: Request, account: Optional[Dict[str, Any]]) -> None:
    """429 when either the account or the IP has spent its window. Records the
    hit as it goes: unlike a login, there is no "success clears it" notion —
    every translation costs money whether or not it understood anything."""
    keys = [f"translate:ip:{client_ip(request)}"]
    account_id = (account or {}).get("id")
    if account_id is not None:
        keys.append(f"translate:acct:{account_id}")
    for key in keys:
        allowed, retry_after = _TRANSLATE_THROTTLE.hit(key)
        if not allowed:
            logger.warning("translate: throttled key=%s retry_after=%ss", key, retry_after)
            raise HTTPException(
                status_code=429,
                detail="Trop de traductions en peu de temps. Réessayez dans un instant.",
                headers={"Retry-After": str(retry_after)},
            )


class TranslateRequest(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    locale: Literal["fr", "en"] = "fr"


class RefusalOut(BaseModel):
    kind: Literal["ranking", "prediction", "recommendation"]


class AssumptionOut(BaseModel):
    condition_type: str
    control: str
    value: Optional[str] = None
    source_phrase: Optional[str] = None


class UntranslatableOut(BaseModel):
    fragment: Optional[str] = None
    category: str


class ScannerTranslateResponse(BaseModel):
    # Stable machine outcome — the webapp localises everything user-facing.
    outcome: Literal["translated", "partial", "none", "refused", "empty", "error"]
    refusal: Optional[RefusalOut] = None
    # Wire-shaped conditions, ready to POST verbatim to /api/conditions-scan.
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    # SC-4 — for each condition, the fragment of the user's own sentence it came
    # from, ALIGNED BY INDEX with ``conditions``. Rides alongside rather than
    # inside, because ``ScanCondition`` forbids extra fields and ``conditions``
    # must stay POST-able verbatim. ``None`` = origin unknown (the model did not
    # cite, or cited something absent from the text and the server rejected it);
    # the webapp then falls back to identity-only removal for that condition.
    condition_sources: List[Optional[str]] = Field(default_factory=list)
    assumptions: List[AssumptionOut] = Field(default_factory=list)
    untranslatable: List[UntranslatableOut] = Field(default_factory=list)


def _revalidate_conditions(
    conditions: List[Dict[str, Any]], sources: List[Optional[str]]
) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
    """Belt-and-suspenders: every condition must survive the SAME model the scan
    endpoint enforces. Anything that fails is dropped here — it can never reach
    the evaluator. In practice ``sanitize_translation`` already guarantees this;
    this is the boundary that makes the guarantee independent of the translator.
    """
    safe: List[Dict[str, Any]] = []
    safe_sources: List[Optional[str]] = []
    # zip_longest-free: a shorter ``sources`` simply yields None for the tail.
    padded = list(sources) + [None] * max(0, len(conditions) - len(sources))
    for cond, source in zip(conditions, padded):
        try:
            ScanCondition(**cond)
        except (ValidationError, TypeError):
            logger.warning("translate: dropping condition rejected by ScanCondition: %r", cond)
            continue
        safe.append(cond)
        safe_sources.append(source)
    return safe, safe_sources


@router.post("/translate", response_model=ScannerTranslateResponse)
async def scanner_translate(
    request: Request,
    body: TranslateRequest,
    account: Optional[Dict[str, Any]] = Depends(optional_account),
) -> ScannerTranslateResponse:
    # Same paid-only gate as the scanner it feeds (no-op while the gate is OFF).
    enforce_access(request, account)
    # SC-4 — every translation is a paid LLM call, and SC-4 fires them on typing
    # pauses rather than on a click. Cap BEFORE reaching the translator.
    _enforce_translate_throttle(request, account)

    translator = getattr(request.app.state.app_state, "scanner_translator", None)
    if translator is None:
        raise HTTPException(status_code=503, detail="Scanner translator not configured")

    try:
        result = translator.translate(body.text, body.locale)
    except Exception:
        logger.exception("scanner_translator.translate failed")
        raise HTTPException(status_code=500, detail="Internal translation error")

    conditions, condition_sources = _revalidate_conditions(
        list(result.get("conditions", [])), list(result.get("condition_sources", []))
    )
    # If the belt-and-suspenders pass emptied the list, keep the outcome honest.
    outcome = result.get("outcome", "none")
    if not conditions and outcome in ("translated", "partial"):
        outcome = "none" if not result.get("untranslatable") else "partial"

    return ScannerTranslateResponse(
        outcome=outcome,
        refusal=result.get("refusal"),
        conditions=conditions,
        condition_sources=condition_sources,
        assumptions=result.get("assumptions", []),
        untranslatable=result.get("untranslatable", []),
    )
