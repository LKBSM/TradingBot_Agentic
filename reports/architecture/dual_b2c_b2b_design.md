# Dual B2C + B2B Architecture Design — Smart Sentinel AI

**Date** : 2026-04-30
**Auteur** : Loukmane Bessam (solo founder)
**Statut** : `DRAFT — review pending` (à valider avant Sprint 1)
**Pivot référence** : `memory/decision_matrix_2026_04_30.md` (pivot B2B après décision matrix 0/4 strats > 1.0 PF lo)

---

## 1. Vision en une phrase

> Un seul moteur d'analyse XAU/USD M15 produit un objet `InsightSignal` unifié, et deux portes de sortie le formatent différemment selon l'audience : **B2C grand public** (Telegram + webapp + email) et **B2B brokers** (API REST authentifiée + webhooks).

Aucune duplication de logique d'analyse. Un seul modèle canonique. Deux formatters.

---

## 2. Diagramme composants (ASCII)

```
                  ┌─────────────────────────────────────────────────┐
                  │           DATA LAYER (existant, inchangé)       │
                  │  DataProvider (CSV/MT5)  ◀── ohlcv stream       │
                  └──────────────┬──────────────────────────────────┘
                                 │
                  ┌──────────────▼──────────────────────────────────┐
                  │     ANALYSIS ENGINE (existant, inchangé)        │
                  │  SmartMoneyEngine → ConfluenceDetector          │
                  │  → VolForecaster → LLMNarrativeEngine           │
                  │  → SemanticCache  → SignalStore (SQLite)        │
                  │  → SignalStateMachine (HOLD/BUY/SELL trust)     │
                  └──────────────┬──────────────────────────────────┘
                                 │ ConfluenceSignal + SignalNarrative
                                 │ + StateMachineSnapshot + ComplianceCtx
                  ┌──────────────▼──────────────────────────────────┐
                  │       ⭐ NEW : InsightSignalBuilder             │
                  │   Assemble le contrat canonique InsightSignal   │
                  │   (Pydantic v2). Enrichit avec:                 │
                  │     • disclaimer multilingue                    │
                  │     • jurisdiction blocking                     │
                  │     • expires_at (TTL fonction du tier signal)  │
                  │     • narrative tiers (court/full)              │
                  └──────────────┬──────────────────────────────────┘
                                 │ InsightSignal (canonique)
              ┌──────────────────┴───────────────────┐
              │                                      │
   ┌──────────▼──────────────┐         ┌─────────────▼──────────────┐
   │   B2C_Formatter         │         │   B2B_API_Server           │
   │   ─────────────         │         │   ────────────             │
   │   • Telegram (≤800 ch)  │         │   • REST: /api/v1/insights │
   │   • Webapp HTML+JSON    │         │   • OpenAPI 3.0 docs       │
   │   • Email digest hebdo  │         │   • Webhooks push (HMAC)   │
   │   • Vocab: SETUP, pas   │         │   • Bearer API key tiered  │
   │     BUY/SELL            │         │   • Toutes les sections    │
   │   • Score arrondi       │         │     exposées (audit)       │
   │     (Strong/Moderate)   │         │   • White-label optionnel  │
   │   • Disclaimer renforcé │         │   • SLA 99.5% / p95<200ms  │
   └─────────────────────────┘         └────────────────────────────┘
              │                                      │
              ▼                                      ▼
       Retail trader                          Broker integration
       (Telegram bot, webapp,                 (IC Markets, Exness,
        email)                                 Pepperstone, etc.)
```

**Composants existants réutilisés sans modification** :
- `src/intelligence/main.py` (build_system, scanner orchestration)
- `src/intelligence/sentinel_scanner.py` (publish loop)
- `src/intelligence/confluence_detector.py` (scoring)
- `src/intelligence/llm_narrative_engine.py` (Claude cascade Haiku/Sonnet/Opus)
- `src/intelligence/volatility_forecaster.py` + `volatility_lgbm.py`
- `src/api/auth.py` (KeyStore SHA-256, 60s cache)
- `src/api/disclaimers.py` (get_disclaimer / get_footer FR/EN/DE/ES)

**Nouveaux composants** :
- `src/models/insight_signal.py` — contrat canonique Pydantic
- `src/distribution/insight_builder.py` — InsightSignalBuilder (assemble depuis ConfluenceSignal + SignalNarrative + state machine)
- `src/distribution/b2c_formatter.py` — Telegram + webapp + email
- `src/distribution/b2b_api.py` — endpoints REST `/api/v1/insights/*`
- `src/distribution/b2b_auth.py` — Bearer tiered (Pilot/Standard/Enterprise) + rate-limit per tier
- `src/distribution/webhook_publisher.py` — push HMAC signé vers brokers

---

## 3. Schéma `InsightSignal` complet (Pydantic v2)

> **Décision tooling** : Pydantic v2 (déjà dans `requirements.txt: pydantic>=2.0.0`).
> Validators avec `model_validator(mode="after")`, sérialisation JSON via `model_dump_json()`.
> Schéma versioné via `version_schema` (semver) pour faire évoluer sans casser les clients B2B.

```python
# src/models/insight_signal.py

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Literal, Tuple
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class StructureBias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ScoreLabel(str, Enum):
    WEAK = "weak"               # 0-39
    MODERATE = "moderate"       # 40-54
    STRONG = "strong"           # 55-69
    VERY_STRONG = "very_strong" # 70-100


class VolRegime(str, Enum):
    LOW = "low_vol"
    NORMAL = "normal"
    HIGH = "high_vol"
    EXTREME = "extreme"


class HMMRegime(str, Enum):
    TREND_BULLISH = "trend_bullish"
    TREND_BEARISH = "trend_bearish"
    RANGE = "range"
    VOLATILE = "volatile"


class Session(str, Enum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    OFF = "off"


class NarrativeLanguage(str, Enum):
    FR = "fr"
    EN = "en"
    DE = "de"
    ES = "es"


class NarrativeModel(str, Enum):
    HAIKU = "claude-haiku"
    SONNET = "claude-sonnet"
    OPUS = "claude-opus"
    TEMPLATE = "template"  # fallback déterministe


# ─── Sub-schemas ──────────────────────────────────────────────────────────

class KeyLevels(BaseModel):
    """Niveaux structurels factuels — JAMAIS présentés comme ordres."""
    support_zone: Tuple[float, float]
    resistance_zone: Tuple[float, float]
    structural_invalidation: float
    first_target: float
    second_target: Optional[float] = None


class VolatilityForecast(BaseModel):
    next_hour_usd: float = Field(description="Forecast volatility next hour in USD")
    vs_atr14_pct: float = Field(description="Pct vs ATR14 (e.g. +18.0 = +18%)")
    regime_label: VolRegime


class MLProbability(BaseModel):
    """ML proba — null tant que pas de modèle entraîné (sprint futur)."""
    probability: float = Field(ge=0.0, le=1.0)
    model_version: str
    confidence_interval: Tuple[float, float]


class Scenario(BaseModel):
    scenario_label: Literal["main", "alternative_1", "alternative_2"]
    condition: str = Field(max_length=200)
    expected_outcome: str = Field(max_length=200)


class ComponentBreakdown(BaseModel):
    """Contribution individuelle au score (B2B audit, B2C agrégé)."""
    name: str
    weighted_score: float
    weight: float
    reasoning: str = Field(max_length=300)


class ComplianceContext(BaseModel):
    disclaimer_text: str
    jurisdiction_blocked: List[str] = Field(
        default_factory=lambda: ["US", "QC", "UK", "OFAC"]
    )
    regulatory_notice: str = Field(
        default="UE 2024/2811 — algorithmic analysis, not investment advice"
    )


# ─── Contrat canonique ────────────────────────────────────────────────────

class InsightSignal(BaseModel):
    """
    Contrat unifié source-of-truth pour B2C ET B2B.

    Le moteur produit UN SEUL InsightSignal par (asset, timeframe, bar_ts).
    Les formatters B2C / B2B le consomment et masquent / agrègent
    selon l'audience. Tous les champs optionnels nullables sont `None`
    quand non calculés (jamais omis dans la sérialisation pour stabilité
    du contrat downstream).
    """

    # ─── Identité ──
    insight_id: UUID
    generated_at: datetime
    asset: str = Field(pattern=r"^[A-Z0-9]{3,8}$")  # XAUUSD, EURUSD, ...
    timeframe: Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
    expires_at: datetime
    version_schema: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")

    # ─── Analyse structurelle (factuels, pas d'ordre) ──
    structure_bias: StructureBias
    confluence_score: int = Field(ge=0, le=100)
    confluence_score_label: ScoreLabel
    key_levels: KeyLevels
    volatility_forecast: VolatilityForecast

    # ─── Probabiliste (ML futur) ──
    ml: Optional[MLProbability] = None

    # ─── Narrative ──
    narrative_short: str = Field(max_length=200, description="Telegram/SMS")
    narrative_full: str = Field(max_length=1500, description="Webapp/B2B")
    narrative_language: NarrativeLanguage
    narrative_model: NarrativeModel
    scenarios: List[Scenario] = Field(min_length=1, max_length=3)

    # ─── Composants (B2B transparency, B2C agrégé) ──
    components_active: List[str] = Field(default_factory=list)
    components_score_breakdown: List[ComponentBreakdown] = Field(default_factory=list)
    regime_hmm: HMMRegime
    news_blackout_active: bool = False
    session: Session

    # ─── Compliance ──
    compliance: ComplianceContext

    # ─── Validators ──
    @model_validator(mode="after")
    def _validate_score_label(self) -> "InsightSignal":
        """Vérifie cohérence score numérique ↔ label."""
        s = self.confluence_score
        expected = (
            ScoreLabel.VERY_STRONG if s >= 70
            else ScoreLabel.STRONG if s >= 55
            else ScoreLabel.MODERATE if s >= 40
            else ScoreLabel.WEAK
        )
        if self.confluence_score_label != expected:
            raise ValueError(
                f"score_label {self.confluence_score_label} incohérent "
                f"avec score {s} (attendu {expected})"
            )
        return self

    @model_validator(mode="after")
    def _validate_expiry(self) -> "InsightSignal":
        if self.expires_at <= self.generated_at:
            raise ValueError("expires_at must be > generated_at")
        return self

    @model_validator(mode="after")
    def _validate_invalidation(self) -> "InsightSignal":
        """Le niveau d'invalidation doit être hors de la zone target."""
        kl = self.key_levels
        if self.structure_bias == StructureBias.BULLISH:
            if kl.structural_invalidation >= kl.first_target:
                raise ValueError("Bullish: invalidation >= first_target")
        elif self.structure_bias == StructureBias.BEARISH:
            if kl.structural_invalidation <= kl.first_target:
                raise ValueError("Bearish: invalidation <= first_target")
        return self
```

### Décision : `expires_at` — TTL fixe à la génération

| Score label    | TTL M15 | TTL H1  | Raison                                              |
| -------------- | ------- | ------- | --------------------------------------------------- |
| `very_strong`  | 4 h     | 12 h    | Confluences fortes restent actives plus longtemps   |
| `strong`       | 2 h     | 6 h     | Setup standard                                      |
| `moderate`     | 1 h     | 3 h     | Setup faible, fenêtre courte                        |
| `weak`         | 30 min  | 1 h     | Quasi-bruit, expire rapidement                      |

**Invalidation dynamique** : la `SignalStateMachine` existante (cf. `memory/signal_state_machine.md`) marque `setup invalidé` plus tôt si le prix casse `structural_invalidation`. `expires_at` est l'**upper bound**, le state machine peut clôturer avant.

---

## 4. Flux B2C (consumer-facing)

### 4.1 Telegram

**Cible** : retail traders, abonnés FREE/ANALYST/STRATEGIST.
**Format** : Markdown HTML Telegram, ≤ 800 chars (vs 4096 max API). Header emoji bias, score arrondi en label, narrative_short, scenarios bullets, disclaimer footer.

**Vocabulaire imposé** (UE 2024/2811) :
- `Setup haussier` / `Setup baissier` — JAMAIS `BUY`/`SELL` (déjà appliqué par `DIRECTION_LABEL` dans le notifier actuel).
- `Score: Strong (paliers 40/55/70)` — JAMAIS `77/100` brut côté B2C.
- `Objectif atteint` / `Setup invalidé` / `Scénario expiré` à la clôture — JAMAIS USD ni R-multiple.
- Pas de position size affichée. Pas de R-multiple sur exit.

### 4.2 Webapp HTML

**Stack** : page statique HTML + CSS hérité de `mockups/tradingview_dashboard_mockup.html` (tokens `--bg/--panel/--accent/--gold`), JS minimal pour polling `/api/v1/state`.

**Layout** : card avec score visuel (ring 0-100), zones structurelles affichées sur mini-chart canvas, narrative_full, scenarios bullets, disclaimer permanent en footer fixe, bouton "Voir la méthodologie".

### 4.3 Email digest hebdomadaire

**Format** : HTML responsive, top 5 setups de la semaine (best confluence_score), preview narrative_short + lien webapp.
**Cadence** : lundi 08:00 UTC.
**Audience** : ANALYST + STRATEGIST.

---

## 5. Flux B2B (broker-facing)

### 5.1 Endpoints REST

| Méthode | Path                              | Auth     | Description                          |
| ------- | --------------------------------- | -------- | ------------------------------------ |
| GET     | `/api/v1/insights/latest`         | Bearer   | Dernier `InsightSignal` complet      |
| GET     | `/api/v1/insights/{insight_id}`   | Bearer   | Insight historique par UUID          |
| GET     | `/api/v1/insights/historical`     | Bearer   | Liste paginée pour audit             |
| GET     | `/api/v1/health`                  | public   | Status moteur, dernier insight ts    |
| POST    | `/api/v1/webhooks/subscribe`      | Bearer   | Enregistre URL push broker           |
| DELETE  | `/api/v1/webhooks/{webhook_id}`   | Bearer   | Désabonne webhook                    |
| GET     | `/api/v1/docs`                    | public   | OpenAPI 3.0 auto-générée             |

**Rate-limit per-tier** (cf. §6) : 429 si dépassé. **403** si clé invalide ou plan insuffisant.

### 5.2 Webhooks push (HMAC signé)

À chaque nouvel `InsightSignal` publié, le `WebhookPublisher` boucle sur les souscriptions actives et envoie :

```http
POST <broker_callback_url>
Content-Type: application/json
X-Sentinel-Signature: sha256=<hmac>
X-Sentinel-Timestamp: <unix_ts>
X-Sentinel-Insight-Id: <uuid>

{ ...InsightSignal payload... }
```

`X-Sentinel-Signature` = `HMAC_SHA256(secret, timestamp + "." + body)`. Secret partagé à la souscription. Replay-attack guard : timestamp > 5 min → reject côté broker.

### 5.3 White-label optionnel

Le broker tier `STANDARD` ou `ENTERPRISE` peut activer `white_label=true` à la souscription. Effet : le `narrative_full` est régénéré sans la mention "Smart Sentinel AI" et avec la marque broker passée en paramètre. **Implémenté côté `LLMNarrativeEngine`** via system prompt dynamique (pas de duplication de moteur).

---

## 6. Pricing & packaging dual

### 6.1 Bilan tarifaire actuel vs cible

> ⚠️ **Delta avec le code en place**. `src/api/tier_manager.py:35` définit aujourd'hui :
> `FREE $0 / ANALYST $49 / STRATEGIST $99 / INSTITUTIONAL $149`.
> Le brief vise **B2C 3 tiers ($0 / $14 / $39)** et **B2B 3 tiers séparés ($500 / $2500 / $10000)**.
> Cela représente une baisse de prix B2C significative et une migration de l'INSTITUTIONAL B2C → B2B PILOT.

### 6.2 Grille cible

| Pilier | Tier              | Prix       | Calls/jour | Assets        | Webhooks | Narrative       |
| ------ | ----------------- | ---------- | ---------- | ------------- | -------- | --------------- |
| B2C    | `FREE`            | $0         | 3 insights | 1 (XAU)       | ❌       | short uniquement |
| B2C    | `ANALYST`         | $14/mo     | illim.     | tous          | ❌       | full + email    |
| B2C    | `STRATEGIST`      | $39/mo     | illim.     | tous          | ❌       | + scenarios + composants + 90j historique |
| B2B    | `BROKER_PILOT`    | $500/mo    | 100        | 1             | 1        | full            |
| B2B    | `BROKER_STANDARD` | $2 500/mo  | 5 000      | 5             | 5        | full + white-label |
| B2B    | `BROKER_ENTERPRISE` | $10 000/mo | illim.   | tous          | illim.   | full + custom integration |

### 6.3 Migration tarifaire (compatibilité descendante)

Contrainte non-nego : "l'API JSON actuelle reste accessible pendant la transition". Plan :

1. **Sprint 1** : ajouter colonne `account_type ENUM('B2C','B2B')` à la table `users`. Default `B2C`. Migration en place.
2. **Sprint 3** : ajouter nouveau tier enum `B2BTier(PILOT/STANDARD/ENTERPRISE)`. L'enum existant `UserTier` reste intact pour B2C.
3. **Sprint 4** : grandfathering — les abonnés INSTITUTIONAL $149 actuels conservent leur tier et leur prix jusqu'à résiliation volontaire ; les nouveaux sont orientés vers B2B PILOT.
4. **Plus tard (post-MVP)** : sunset des nouveaux abonnements B2C INSTITUTIONAL.

---

## 7. Refactor du pipeline existant

```
DataProvider → SmartMoneyEngine → ConfluenceDetector → VolForecaster
            → LLMNarrativeEngine → SemanticCache → SignalStore
            → SignalStateMachine
            → ⭐ InsightSignalBuilder
                  ├─→ B2C_Formatter ─→ Telegram + Webapp + Email
                  └─→ B2B_API_Server ─→ REST + Webhooks
```

**Point d'intégration unique** : `SentinelScanner._publish_signal()` (`src/intelligence/sentinel_scanner.py:661`) — actuellement appelle `_send_notification_safe()` puis `signal_store.add_signal()`. Insertion :

```python
# AVANT (actuel)
def _publish_signal(self, signal, narrative_data):
    self._send_notification_safe(signal, narrative_data)
    self._signal_store.add_signal(signal, narrative_data)

# APRÈS (Sprint 1)
def _publish_signal(self, signal, narrative_data):
    insight = self._insight_builder.build(
        signal=signal,
        narrative=narrative_data,
        state_snapshot=self._state_machine.snapshot(),
    )
    # B2C path (legacy notifier conservé pendant migration)
    self._b2c_formatter.dispatch(insight)
    # B2B path
    self._b2b_publisher.publish(insight)  # webhooks + signal_store
    # Compat descendante: ancien signal_store toujours alimenté
    self._signal_store.add_signal(signal, narrative_data)
```

**Aucune modification du moteur d'analyse**. ConfluenceDetector / SmartMoneyEngine / VolForecaster / LLM cascade restent identiques. C'est uniquement l'**output final** qui change.

---

## 8. Compliance — viser 8/10 (vs 7/10 actuel)

Hérité de `memory/sprint_w1_compliance_2026_04_29.md` (W1+W2+W3 livré) :
- ✅ Geo-block US/QC/UK/OFAC déjà actif.
- ✅ Disclaimers FR/EN/DE/ES déjà actifs (`src/api/disclaimers.py`).
- ✅ `BULLISH SETUP / BEARISH SETUP` déjà actif (`telegram_notifier.py:28`).
- ✅ TelegramLangStore déjà en place pour résolution langue par chat.

**Renforcements pour atteindre 8/10** :
1. **`InsightSignal.compliance.disclaimer_text`** est obligatoire dans CHAQUE payload (B2C ET B2B). Pas d'opt-out côté broker.
2. **Disclaimer permanent visible** dans la webapp B2C (footer fixe, pas dismissible).
3. **B2B Terms of Service spécifique** : le broker s'engage contractuellement à propager le disclaimer dans son UI utilisateur final (clause incluse dans la souscription Pilot/Standard/Enterprise).
4. **Audit trail** : chaque appel B2B logue `key_id, endpoint, insight_id, ts` dans `api_usage` table — preuve de diligence en cas de litige.

---

## 9. Contraintes non-négociables (rappel)

- ❌ Aucune modification du moteur d'analyse (BOS/FVG/HMM/scoring/vol restent identiques).
- ✅ Compatibilité descendante : routes `/api/v1/narratives`, `/api/v1/state`, `/api/v1/signals` actuelles **restent fonctionnelles** pendant la transition.
- ✅ Compliance : 8/10 cible (vs 7/10 actuel).
- ✅ Multilingue FR/EN obligatoire dès Sprint 2 (DE/ES déjà en place).

---

## 10. Décisions techniques résumées

| # | Décision                          | Choix                                | Rationale |
| - | --------------------------------- | ------------------------------------ | --------- |
| 1 | Pydantic version                  | **v2** (`>=2.0.0` confirmé)          | Déjà dans `requirements.txt`, validators `mode="after"` plus propres |
| 2 | `expires_at` calcul               | **TTL fixe** par tier de score à la génération | Cohérent avec le state machine qui invalide dynamiquement plus tôt si `structural_invalidation` cassé |
| 3 | B2B tiers                         | Enum séparé `B2BTier`                | Évite de casser `UserTier` existant ; sépare les rate-limits B2C/B2B |
| 4 | Webhook security                  | HMAC SHA256 + timestamp ≤ 5min       | Standard industrie (Stripe, GitHub) |
| 5 | InsightSignal storage             | Stocké dans nouvelle table `insights` | `signals.db` actuel reste pour compat descendante |
| 6 | White-label                       | Param système prompt LLM             | Pas de duplication moteur |
| 7 | Schéma versioning                 | `version_schema` semver dans payload | Permet migrations B2B sans casser clients existants |
| 8 | Multi-langue narrative            | Déjà géré par `disclaimers.py` + `TelegramLangStore` ; étendu au LLM via system prompt FR/EN | Réutilise infra W1 |

---

## 11. Diagramme séquence (génération + dispatch)

```
SentinelScanner ─bar tick─▶ ConfluenceDetector
                            │
                            ▼ ConfluenceSignal (score, levels, components)
                            ▼
                            VolForecaster ▶ vol_forecast_atr, regime
                            ▼
                            LLMNarrativeEngine ▶ SignalNarrative (FR/EN, short/full)
                            ▼
                            SignalStateMachine ▶ snapshot (HOLD/BUY/SELL ctx)
                            │
                            ▼
                    ⭐ InsightSignalBuilder.build(signal, narrative, snapshot)
                            │
                            ▼ InsightSignal (canonique, validé Pydantic v2)
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
         B2C_Formatter             B2B_API_Server
              │                           │
              ├─▶ TelegramNotifier        ├─▶ /api/v1/insights/latest cache
              ├─▶ WebappBroadcast         ├─▶ insights_table (SQLite)
              └─▶ EmailDigestQueue        └─▶ WebhookPublisher
                                                  │ HMAC + retry exp.backoff
                                                  ▼
                                          Broker callback URL
```

---

## 12. Critères de validation Go/No-Go avant Sprint 1

- ✅ Brief produit complet (livré).
- ✅ Pydantic v2 confirmé (vérifié `requirements.txt`).
- ✅ Pipeline existant cartographié (sentinel_scanner.py:661 = point d'insertion unique).
- ⏳ Validation user du delta tarifaire B2C (réduction ANALYST $49 → $14, STRATEGIST $99 → $39).
- ⏳ Validation user du grandfathering INSTITUTIONAL $149 actuels.
- ⏳ Validation user des 4 mockups (Telegram, webapp, B2B JSON, webhook).

---

## 13. Annexes

- `reports/architecture/sprints_plan.md` — plan de tickets détaillé Sprint 1-4.
- `mockups/telegram_b2c.txt` — exemple message Telegram B2C rendu.
- `mockups/webapp_b2c.html` — webapp B2C standalone.
- `mockups/b2b_insight.json` — exemple full payload B2B.
- `mockups/b2b_webhook_payload.json` — exemple webhook push broker.

---

**Fin du document d'architecture. À valider avant Sprint 1.**
