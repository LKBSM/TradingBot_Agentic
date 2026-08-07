"""SC-2 — Scanner conversationnel : endpoint de traduction phrase → palette.

POST /api/scanner/translate
  body: { "text": str (1..500), "locale": "fr"|"en" (défaut "fr") }
  → 200 ScannerTranslateResponse :
      { outcome, refusal?, conditions[], assumptions[], untranslatable[] }
  → 422 sur corps invalide (texte vide / trop long)
  → 503 si le traducteur n'est pas câblé (SCANNER_TRANSLATOR_ENABLED=false ou clé absente)

M.I.A ne reçoit AUCUNE donnée de marché ici : elle traduit une phrase. La sortie
est re-validée contre la palette fermée AVANT de quitter le serveur, puis chaque
condition passe par le modèle ``ScanCondition`` du endpoint de scan — un type
hors palette ou une valeur hors domaine ne peut donc jamais atteindre
l'évaluateur (même verrou que l'``id`` des zones dans le chatbot).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from src.api.subscription_gate import enforce_access
from src.api.routes.conditions_scan import ScanCondition
from src.api.session_auth import optional_account

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scanner", tags=["scanner-conversational"])

MAX_TEXT_LENGTH = 500


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
    assumptions: List[AssumptionOut] = Field(default_factory=list)
    untranslatable: List[UntranslatableOut] = Field(default_factory=list)


def _revalidate_conditions(conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Belt-and-suspenders: every condition must survive the SAME model the scan
    endpoint enforces. Anything that fails is dropped here — it can never reach
    the evaluator. In practice ``sanitize_translation`` already guarantees this;
    this is the boundary that makes the guarantee independent of the translator.
    """
    safe: List[Dict[str, Any]] = []
    for cond in conditions:
        try:
            ScanCondition(**cond)
        except (ValidationError, TypeError):
            logger.warning("translate: dropping condition rejected by ScanCondition: %r", cond)
            continue
        safe.append(cond)
    return safe


@router.post("/translate", response_model=ScannerTranslateResponse)
async def scanner_translate(
    request: Request,
    body: TranslateRequest,
    account: Optional[Dict[str, Any]] = Depends(optional_account),
) -> ScannerTranslateResponse:
    # Same paid-only gate as the scanner it feeds (no-op while the gate is OFF).
    enforce_access(request, account)

    translator = getattr(request.app.state.app_state, "scanner_translator", None)
    if translator is None:
        raise HTTPException(status_code=503, detail="Scanner translator not configured")

    try:
        result = translator.translate(body.text, body.locale)
    except Exception:
        logger.exception("scanner_translator.translate failed")
        raise HTTPException(status_code=500, detail="Internal translation error")

    conditions = _revalidate_conditions(list(result.get("conditions", [])))
    # If the belt-and-suspenders pass emptied the list, keep the outcome honest.
    outcome = result.get("outcome", "none")
    if not conditions and outcome in ("translated", "partial"):
        outcome = "none" if not result.get("untranslatable") else "partial"

    return ScannerTranslateResponse(
        outcome=outcome,
        refusal=result.get("refusal"),
        conditions=conditions,
        assumptions=result.get("assumptions", []),
        untranslatable=result.get("untranslatable", []),
    )
