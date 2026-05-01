"""
InsightSignal v2 — unified Pydantic v2 contract for B2C + B2B surfaces.

Sprint UX-1.1 (Inès, 5h). See `reports/roadmap_2026_2027/PLAN_12_MOIS.md`
Partie II.2 Agent 6.

This is the *lingua franca* of the product: every B2C surface (Telegram,
webapp, email) and every B2B surface (REST insights, webhook payloads,
audit-trail rows) is a derivation of this model. Adding a new surface
means writing a renderer; modifying a field means updating exactly one
type.

Design constraints
------------------
- Pydantic v2 (the project's std for new models)
- Strict UE 2024/2811 finfluencer compliance:
  * `direction` exposes BULLISH_SETUP / BEARISH_SETUP / NEUTRAL
    (NEVER raw "BUY" / "SELL" in user-facing strings, even in JSON)
  * Disclaimer text + jurisdiction block flags are first-class fields
  * Sources cited are required for any RAG-backed narrative (Phase 2B)
- Backward-compat shim: `from_v1_signal(signal_response)` lifts the
  existing `SignalResponse` model from `src/api/models.py` to v2 with
  defensible defaults for the new fields.
- All timestamps are tz-aware UTC.

Backward compatibility
----------------------
Existing API routes can keep returning their current `SignalResponse`. The
v2 model is consumed by NEW B2C/B2B routes and the InsightSignal-v2 audit
trail (DATA-2A.7 / DATA-2B.4). Migration is opt-in per route.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SetupDirection(str, Enum):
    """User-facing direction labels.

    Per UE 2024/2811 finfluencer regulation (eval_29 finding 1), we never
    expose raw "BUY" / "SELL" / "achetez" / "vendez" verbs. Algorithmic
    setup classification is *contextual*, not a recommendation to act.
    """

    BULLISH_SETUP = "BULLISH_SETUP"
    BEARISH_SETUP = "BEARISH_SETUP"
    NEUTRAL = "NEUTRAL"


class Timeframe(str, Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"


class ConvictionLabel(str, Enum):
    """Bucketed conviction (no exact score reveal — protect IP)."""

    WEAK = "weak"  # 0-39
    MODERATE = "moderate"  # 40-59
    STRONG = "strong"  # 60-79
    INSTITUTIONAL = "institutional"  # 80-100


def conviction_to_label(score: int | float) -> ConvictionLabel:
    s = max(0, min(100, int(score)))
    if s < 40:
        return ConvictionLabel.WEAK
    if s < 60:
        return ConvictionLabel.MODERATE
    if s < 80:
        return ConvictionLabel.STRONG
    return ConvictionLabel.INSTITUTIONAL


class SourceType(str, Enum):
    """Provenance categories for narrative citations (Phase 2B RAG)."""

    PAPER = "paper"  # academic paper
    REPORT = "report"  # institutional research (LBMA, WGC, BIS)
    DATA = "data"  # primary data source (CFTC COT, FOMC minutes)
    EDUCATION = "education"  # generalist (Investopedia, BabyPips)
    INTERNAL = "internal"  # our own backtest / forward-test artefact


class NarrativeLanguage(str, Enum):
    FR = "fr"
    EN = "en"
    DE = "de"
    ES = "es"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class SignalLevels(BaseModel):
    """Price levels carried by the signal. All optional for HOLD setups."""

    entry: Optional[float] = Field(default=None, description="Entry price")
    stop: Optional[float] = Field(default=None, description="Stop level")
    target_1: Optional[float] = Field(default=None, description="First target")
    target_2: Optional[float] = Field(default=None, description="Extended target")
    invalidation: Optional[float] = Field(
        default=None,
        description="Structural invalidation level (separate from stop)",
    )

    @model_validator(mode="after")
    def _stop_consistent_with_direction(self) -> "SignalLevels":
        # No structural assertion enforced here — the direction is on the
        # parent InsightSignal. Validation lives at the parent level.
        return self


class Source(BaseModel):
    """One narrative citation. Required for any RAG-backed narrative_long."""

    type: SourceType
    ref: str = Field(description="URL or canonical identifier")
    label: str = Field(description="Human-readable short label")
    quoted_excerpt: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Up to 500-char excerpt used in narrative (for audit)",
    )


class ComplianceMeta(BaseModel):
    """Compliance / regulatory metadata attached to every signal."""

    disclaimer_lang: NarrativeLanguage = NarrativeLanguage.FR
    jurisdiction_blocked: list[str] = Field(
        default_factory=list,
        description=(
            "List of jurisdiction codes (ISO-3166-1 alpha-2 + 'US-CA' style "
            "subdivisions) for which delivery is blocked. Set by geo-block "
            "middleware (sprint W1+W2+W3)."
        ),
    )
    edge_claim: bool = Field(
        default=False,
        description=(
            "True ONLY when the algorithmic edge has been validated "
            "(post-CP-A1 Phase 2A). Always False on the 2B narrative-first "
            "branch. UI rendering must NEVER claim 'edge prouvé' when False."
        ),
    )
    is_paper_demo: bool = Field(
        default=True,
        description=(
            "True when the signal is part of a transparent paper-trade "
            "track-record demonstration (Phase 2B feature) rather than a "
            "monetised live signal. UI must label as 'demonstration'."
        ),
    )


class VolatilityContext(BaseModel):
    """Optional volatility regime context for narrative enrichment."""

    regime: Optional[str] = Field(default=None, description="low / normal / high")
    forecast_atr_pct: Optional[float] = None
    naive_atr_pct: Optional[float] = None


# ---------------------------------------------------------------------------
# Top-level v2 model
# ---------------------------------------------------------------------------


SCHEMA_VERSION = "2.0.0"


class InsightSignalV2(BaseModel):
    """Unified B2C + B2B insight signal contract.

    All product surfaces derive from this single source of truth:
      * Telegram B2C (compact, FR-first)
      * Webapp B2C (full narrative + sources panel)
      * Email digest (subset)
      * B2B REST GET /api/v2/insights (full payload)
      * B2B webhook POST (full payload + delivery metadata wrapper)
      * Audit trail row (full payload + hash chain)
    """

    schema_version: str = Field(default=SCHEMA_VERSION, frozen=True)

    # Identity
    id: str = Field(description="Globally unique signal identifier")
    instrument: str = Field(description="e.g. XAUUSD, EURUSD, USOIL")
    timeframe: Timeframe

    # Algorithmic call
    direction: SetupDirection
    conviction_0_100: int = Field(ge=0, le=100)
    levels: SignalLevels = Field(default_factory=SignalLevels)
    volatility: Optional[VolatilityContext] = None

    # Narrative + provenance
    narrative_short: str = Field(
        max_length=400,
        description="Telegram-ready summary (≤400 chars)",
    )
    narrative_long: str = Field(
        default="",
        description="Webapp/B2B full narrative — RAG-sourced in Phase 2B",
    )
    narrative_language: NarrativeLanguage = NarrativeLanguage.FR
    sources_cited: list[Source] = Field(
        default_factory=list,
        description="Required when narrative_long is non-empty in Phase 2B",
    )

    # Compliance
    compliance: ComplianceMeta = Field(default_factory=ComplianceMeta)

    # Lifecycle
    created_at_utc: datetime
    valid_until_utc: Optional[datetime] = None

    # Free-form for renderer hints (NOT for primary data — keep payload typed)
    extras: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "schema_version": "2.0.0",
                    "id": "0193c7a4-2f1b-7c3d-9e8a-5b2c1d4e7f89",
                    "instrument": "XAUUSD",
                    "timeframe": "M15",
                    "direction": "BULLISH_SETUP",
                    "conviction_0_100": 72,
                    "levels": {
                        "entry": 2350.00,
                        "stop": 2340.00,
                        "target_1": 2370.00,
                        "target_2": 2390.00,
                        "invalidation": 2335.00,
                    },
                    "narrative_short": (
                        "Setup haussier XAU M15. Cassure de structure + retest FVG. "
                        "Régime normal vol. Analyse algorithmique éducative."
                    ),
                    "compliance": {
                        "disclaimer_lang": "fr",
                        "edge_claim": False,
                        "is_paper_demo": True,
                    },
                    "created_at_utc": "2026-05-01T12:00:00+00:00",
                    "valid_until_utc": "2026-05-01T16:00:00+00:00",
                }
            ]
        }
    }

    # ----- Validators -----

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must be non-empty")
        return v

    @field_validator("created_at_utc", "valid_until_utc")
    @classmethod
    def _ensure_tz_utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _hold_has_no_levels(self) -> "InsightSignalV2":
        if self.direction == SetupDirection.NEUTRAL:
            # NEUTRAL setups should not advertise entry/stop/target — the
            # narrative explains why we wait. Levels stay None.
            if any(
                v is not None
                for v in (
                    self.levels.entry,
                    self.levels.stop,
                    self.levels.target_1,
                )
            ):
                raise ValueError(
                    "NEUTRAL direction must not carry entry/stop/target levels"
                )
        return self

    @model_validator(mode="after")
    def _bullish_bearish_levels_consistent(self) -> "InsightSignalV2":
        L = self.levels
        if (
            self.direction == SetupDirection.BULLISH_SETUP
            and L.entry is not None
            and L.stop is not None
        ):
            if L.stop >= L.entry:
                raise ValueError("BULLISH_SETUP requires stop < entry")
            if L.target_1 is not None and L.target_1 <= L.entry:
                raise ValueError("BULLISH_SETUP requires target_1 > entry")
        if (
            self.direction == SetupDirection.BEARISH_SETUP
            and L.entry is not None
            and L.stop is not None
        ):
            if L.stop <= L.entry:
                raise ValueError("BEARISH_SETUP requires stop > entry")
            if L.target_1 is not None and L.target_1 >= L.entry:
                raise ValueError("BEARISH_SETUP requires target_1 < entry")
        return self

    @model_validator(mode="after")
    def _narrative_long_requires_sources_in_phase2b(self) -> "InsightSignalV2":
        # Soft guard: narrative_long is allowed without sources in Phase 1
        # (template mode). When `is_paper_demo=False` and `narrative_long` is
        # populated, sources must be cited (Phase 2B RAG hard requirement).
        # Phase 1 leaves this informational only — actual enforcement is in
        # the renderer / compliance check.
        return self

    # ----- Convenience -----

    @property
    def conviction_label(self) -> ConvictionLabel:
        return conviction_to_label(self.conviction_0_100)

    @property
    def rr_ratio(self) -> Optional[float]:
        L = self.levels
        if L.entry is None or L.stop is None or L.target_1 is None:
            return None
        risk = abs(L.entry - L.stop)
        reward = abs(L.target_1 - L.entry)
        if risk == 0:
            return None
        return round(reward / risk, 2)


# ---------------------------------------------------------------------------
# Backward-compat shim from v1 SignalResponse
# ---------------------------------------------------------------------------


def from_v1_signal(
    signal: Any,
    narrative_short: str = "",
    narrative_long: str = "",
    language: NarrativeLanguage = NarrativeLanguage.FR,
    timeframe: Timeframe = Timeframe.M15,
    edge_claim: bool = False,
    is_paper_demo: bool = True,
) -> InsightSignalV2:
    """Lift a legacy ``SignalResponse``-like object into v2.

    The v1 model carries ``action`` (OPEN_LONG / OPEN_SHORT / HOLD), prices,
    and rr_ratio. We map:
      OPEN_LONG / LONG / CLOSE_LONG ⇒ BULLISH_SETUP
      OPEN_SHORT / SHORT / CLOSE_SHORT ⇒ BEARISH_SETUP
      HOLD ⇒ NEUTRAL

    Parameters
    ----------
    signal : Any
        Object with attributes ``signal_id``, ``action``, ``symbol``,
        ``entry_price``, ``stop_loss``, ``take_profit``, ``created_at``.
    narrative_short, narrative_long, language : str
        Optional narrative payloads (default empty for header-only conversions).
    edge_claim, is_paper_demo : bool
        Compliance flags; sensible Phase 1 defaults.
    """
    action = str(getattr(signal, "action", "HOLD")).upper().split(".")[-1]
    if "LONG" in action:
        direction = SetupDirection.BULLISH_SETUP
    elif "SHORT" in action:
        direction = SetupDirection.BEARISH_SETUP
    else:
        direction = SetupDirection.NEUTRAL

    if direction == SetupDirection.NEUTRAL:
        levels = SignalLevels()
    else:
        levels = SignalLevels(
            entry=getattr(signal, "entry_price", None),
            stop=getattr(signal, "stop_loss", None),
            target_1=getattr(signal, "take_profit", None),
        )

    created_at = getattr(signal, "created_at", None) or datetime.now(timezone.utc)

    return InsightSignalV2(
        id=str(getattr(signal, "signal_id", "")) or "unknown",
        instrument=str(getattr(signal, "symbol", "XAUUSD")),
        timeframe=timeframe,
        direction=direction,
        conviction_0_100=int(getattr(signal, "conviction", 50) or 50),
        levels=levels,
        narrative_short=narrative_short or _default_narrative(direction, levels, language),
        narrative_long=narrative_long,
        narrative_language=language,
        compliance=ComplianceMeta(
            disclaimer_lang=language,
            edge_claim=edge_claim,
            is_paper_demo=is_paper_demo,
        ),
        created_at_utc=created_at,
    )


def _default_narrative(
    direction: SetupDirection,
    levels: SignalLevels,
    language: NarrativeLanguage,
) -> str:
    """Minimal compliance-safe placeholder narrative when v1 source has none."""
    if language == NarrativeLanguage.FR:
        if direction == SetupDirection.BULLISH_SETUP:
            return (
                f"Setup haussier algorithmique. Niveaux : entrée {levels.entry}, "
                f"stop {levels.stop}, cible {levels.target_1}. Analyse éducative."
            )
        if direction == SetupDirection.BEARISH_SETUP:
            return (
                f"Setup baissier algorithmique. Niveaux : entrée {levels.entry}, "
                f"stop {levels.stop}, cible {levels.target_1}. Analyse éducative."
            )
        return "Contexte neutre. Le système attend une cassure de structure."
    # Default EN
    if direction == SetupDirection.BULLISH_SETUP:
        return (
            f"Bullish algorithmic setup. Levels: entry {levels.entry}, "
            f"stop {levels.stop}, target {levels.target_1}. Educational analysis."
        )
    if direction == SetupDirection.BEARISH_SETUP:
        return (
            f"Bearish algorithmic setup. Levels: entry {levels.entry}, "
            f"stop {levels.stop}, target {levels.target_1}. Educational analysis."
        )
    return "Neutral context. The system awaits a structural break."


# ---------------------------------------------------------------------------
# Surface renderers (each surface = one tiny adapter)
# ---------------------------------------------------------------------------


def to_telegram_b2c(signal: InsightSignalV2) -> str:
    """Compact Telegram message (≤800 chars). HTML parse_mode."""
    direction_label = {
        SetupDirection.BULLISH_SETUP: "🟢 SETUP HAUSSIER",
        SetupDirection.BEARISH_SETUP: "🔴 SETUP BAISSIER",
        SetupDirection.NEUTRAL: "⚪ NEUTRE",
    }[signal.direction]
    label = signal.conviction_label.value.upper()

    parts = [
        f"<b>Smart Sentinel — Analyse algorithmique</b>",
        f"<b>Setup :</b> {direction_label}",
        f"<b>Actif :</b> {signal.instrument} · {signal.timeframe.value}",
        f"<b>Conviction :</b> {label}",
    ]
    L = signal.levels
    if L.entry is not None:
        parts.append(f"<b>Entrée :</b> {L.entry}")
    if L.stop is not None:
        parts.append(f"<b>Stop :</b> {L.stop}")
    if L.target_1 is not None:
        parts.append(f"<b>Cible :</b> {L.target_1}")
    parts.append("")
    parts.append(signal.narrative_short)
    parts.append("")
    parts.append("<i>Analyse éducative algorithmique. Pas un conseil en investissement.</i>")
    msg = "\n".join(parts)
    return msg[:800]


def to_b2b_dict(signal: InsightSignalV2) -> dict:
    """Full B2B JSON payload — alias of `model_dump(mode='json')` with
    enum values stringified."""
    return signal.model_dump(mode="json")


def to_audit_row(signal: InsightSignalV2) -> dict:
    """Compact dict suitable for the append-only audit trail (DATA-2A.7).

    Keeps only deterministic, hash-stable fields. Excludes free-text
    narratives (those are hashed separately as ``narrative_md5`` upstream).
    """
    return {
        "signal_id": signal.id,
        "schema_version": signal.schema_version,
        "instrument": signal.instrument,
        "timeframe": signal.timeframe.value,
        "direction": signal.direction.value,
        "conviction_0_100": signal.conviction_0_100,
        "entry": signal.levels.entry,
        "stop": signal.levels.stop,
        "target_1": signal.levels.target_1,
        "edge_claim": signal.compliance.edge_claim,
        "is_paper_demo": signal.compliance.is_paper_demo,
        "created_at_utc": signal.created_at_utc.isoformat(),
    }
