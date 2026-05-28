# M.I.A. Markets — Architecture cible

**Document de référence architecturale** · v1.0 · 2026-05-27
**Auteur** : instance Claude Code (mission architecture)
**Statut** : Parties 1-2-3 livrées, en attente de validation utilisateur avant Parties 4-8

---

## Préambule — Vision non-négociable (rappel)

M.I.A. Markets est un **indicateur de marché conversationnel** Or + FX, B2C retail large, **bilingue FR+EN J1**, 9 pays Phase 1 (CA + FR + BE + CH + LU + UK + AU + NZ + IE). Architecture en 3 couches sur la même donnée :

| Couche | Contenu | Cible temps |
|---|---|---|
| **C1 — FOCUS** | Verdict 1 phrase + PF historique + alerte event imminent | ≤ 10 s |
| **C2 — CO-PILOT** | 6 cartes hiérarchisées + chatbot sidebar permanent | 30-60 s |
| **C3 — EXPERT** | Waterfall 8 composantes + conformal viz + sources RAG | à la demande |

**Trois invariants** :
1. Le **chatbot Sentinel** est le moat — il définit le jargon, décompose la conviction, refuse pédagogiquement les ordres.
2. La **compliance UE 2024/2811** est par construction (pas de promesse de gain, `edge_claim=False` codifié, posture éducative).
3. Le **mobile-first** est la cible primaire (60-70 % du retail vit là).

Toute proposition qui ne sert pas C1/C2/C3, le chatbot, l'acquisition B2C ou la compliance non-négociable est suspecte.

---

# Partie 1 — État des lieux du code actuel

## 1.1 Arbre commenté du repo

```
TradingBOT_Agentic/
├── src/
│   ├── intelligence/          ✅ Cœur algo : pipeline signal complet (~32 modules, 10k LOC)
│   │   ├── insight_v2/        ✅ Contrat InsightSignalV2 — source de vérité
│   │   ├── scoring/           🟡 Pipeline calibré LGBM→Iso→ACI (model EXISTE, non-wired)
│   │   ├── rag/               🟡 15 fichiers — RAG pour chatbot, état WIP
│   │   ├── rules_engine/      🟡 Pattern detection, peu utilisé
│   │   ├── macro_factors/     🟡 FRED, COT, calendar — en cours
│   │   ├── microstructure/    🟡 Order flow, stub
│   │   ├── factor_model/      🔬 Recherche A1
│   │   ├── conformal/         🟡 ACI Gibbs-Candès
│   │   └── smart_money/       ✅ Alias vers environment.strategy_features
│   ├── api/                   ✅ FastAPI + 18 routes
│   │   └── routes/            ✅ signals, narratives, qa, billing, admin, …
│   ├── agents/                ⚠️ Hybride : 60 % legacy RL, 40 % modern (~25k LOC)
│   ├── environment/           ⚠️ God module 2 423 LOC (environment.py) + strategy_features ✅
│   ├── delivery/              ✅ Telegram, Discord, webhook B2B (queue WIP)
│   ├── training/              ❌ 100 % legacy RL (~7,5k LOC, mort)
│   ├── research/              🔬 6 modules CPCV, A1, gates
│   ├── live_trading/          ⚠️ 9 modules MT5 — usage limité
│   ├── backtest/              🔬 8 modules harness
│   ├── performance/           ✅ Monitoring SLO
│   ├── security/              ✅ Hardening
│   ├── risk/, core/, utils/   ✅ Petits modules support
│   └── interfaces/            🟡 Contrats partiels
├── webapp/                    ✅ Next.js 15 — V2.0-2.4 livré 2026-05-27 (rebrand M.I.A. fait)
│   ├── app/[locale]/          ✅ i18n FR/EN, layout, page hero+cards+chat
│   ├── app/api/chat/route.ts  ✅ SSE Anthropic Claude (Haiku/Sonnet)
│   ├── components/{chat,insight,landing}/  ✅ Architecture progressive uniforme
│   ├── lib/                   🟡 Mocks encore en place, zod absent
│   ├── messages/              ✅ i18n FR/EN
│   ├── tests/                 ✅ Vitest 40 verts + Playwright scaffold
│   └── public/                ✅ PWA manifest, icons SVG
├── infrastructure/            ✅ Dockerfile + docker-compose + prometheus + grafana
├── scripts/                   🔬 75 scripts research/audit (~40 % legacy RL, ~60 % actifs)
├── data/                      ✅ XAU/EUR 7 ans, macro FRED/COT, calendar
├── models/                    🟡 7 fichiers .pkl/.lgb (1 prod calibrated_conviction, 1 LGBM non-wired)
├── docs/
│   ├── governance/            ✅ ADRs (DG-001+), 5 locks politiques, dev_focus_plan
│   ├── value/                 ✅ client_information_explained, client_relevance_review
│   ├── frontend/              ✅ TODO_NEXT_SPRINTS, component_inventory
│   ├── branding/              ✅ naming_research
│   ├── algo/, deployment/, product/  🟡 Stubs
│   └── architecture/          ⚠️ N'existe pas avant ce document
├── mockups/                   ✅ v3/best_concept_demo.html (1489 L, à jour 2026-05-26)
├── tests/                     ✅ 169 fichiers, 1366+ tests, 0 régressions
├── parallel_training.py       ❌ 1 634 LOC RL legacy à la racine
├── COMPLETE_PROJECT_DOCUMENTATION.md  🟡 Source historique mais date 2026-05-21
└── 15+ .md racine             ⚠️ Pollution documentaire racine (à archiver)
```

Légende : ✅ production · 🟡 WIP · 🔬 recherche · ⚠️ dette · ❌ legacy à supprimer

## 1.2 Matrice de santé par sous-système

| Sous-système | LOC env. | Statut | Verdict | Justification |
|---|---|---|---|---|
| **src/intelligence/** (cœur) | ~10 000 | ✅ PROD | **KEEP** | Pipeline complet, modulaire, testé. C'est l'actif technique principal. |
| **src/intelligence/scoring/** | ~500 | 🟡 WIP | **KEEP+WIRE** | LGBM v3 entraîné mais non branché — chemin critique audit 2026-05-27. |
| **src/intelligence/rag/** (15 fichiers) | ~1 500 | 🟡 WIP | **KEEP minimal + DEFER complet** | Pipeline complet pré-mature. Garder 12 papers curés + mini-fiches (DG-058a). |
| **src/intelligence/insight_v2/** | ~600 | ✅ PROD | **KEEP** | Contrat source. À promouvoir source unique. |
| **src/api/** + 18 routes | ~3 600 | ✅ PROD | **KEEP** | Bien designée, auth + tier OK, OpenAPI exposable. |
| **src/api/models.py** | ~800 | ⚠️ DETTE | **REFACTOR (split)** | 30+ modèles en un fichier — splitter par domaine. |
| **src/agents/** modernes (risk_sentinel, kill_switch, news_analysis, portfolio_risk, ensemble_risk, multi_timeframe, sentiment_analyzer, orchestrator, audit_logger, monitoring) | ~12 000 | ✅ PROD | **KEEP+CONSOLIDER** | Utilisés en prod. Mais 5+ "RiskManager" éparpillés (cf. §1.3). |
| **src/agents/** legacy RL (integration.py, intelligent_integration.py, orchestrated_integration.py, risk_integration.py, market_regime_agent.py, regime_predictor.py, sprint2_intelligence.py) | ~3 000 | ❌ MORT | **DROP (archive)** | Wrappers d'une RL Phase 2A morte (verdict A1 2026-05-01). |
| **src/environment/strategy_features.py** (SmartMoneyEngine) | 1 213 | ✅ PROD | **KEEP+EXTRACT** | Activement utilisé par sentinel_scanner. À sortir de environment/. |
| **src/environment/environment.py** | 2 423 | ⚠️ GOD MODULE | **REFACTOR (split)** | 50 % RL legacy + 50 % features actives. Extraire features, archiver RL. |
| **src/environment/risk_manager.py** | ~260 | ⚠️ DOUBLON | **CONSOLIDATE** | Un des 5+ "RiskManager" — voir §1.3. |
| **src/delivery/telegram_notifier.py** | ~400 | ✅ PROD | **KEEP** | Canal FREE essentiel, multilingue, retry+dedup. |
| **src/delivery/discord_notifier.py** | ~250 | ✅ PROD | **KEEP** | Symétrique Telegram. |
| **src/delivery/webhook_queue.py + webhook_signer.py + webhook_drain_worker.py** | ~400 | 🟡 WIP | **DEFER (POST-B2B)** | Infra B2B signée mais pas wired en main.py. Conserver, brancher quand DG-071. |
| **src/training/** (9 modules) | ~7 500 | ❌ MORT | **DROP (archive .old/)** | Stack RL training — verdict A1 = mort. |
| **src/research/** | ~1 500 | 🔬 RECHERCHE | **KEEP (sandbox)** | CPCV harness + A1 features utiles pour validation hors-ligne. |
| **src/live_trading/** | ~2 000 | ⚠️ INCERTAIN | **DEFER décision** | MT5 — utile si on garde paper-demo MT5, sinon orpheline. À décider Sprint 2. |
| **src/backtest/** | ~1 500 | 🔬 RECHERCHE | **KEEP** | Harness backtest pour evals régulières. |
| **src/performance/** | ~1 200 | ✅ PROD | **KEEP** | Monitoring SLO actif. |
| **src/security/** | ~800 | ✅ PROD | **KEEP** | Hardening (geo-block, sanitize, HMAC). |
| **src/risk/, core/, utils/, interfaces/** | ~1 500 | 🟡 MIXTE | **AUDIT** | Petits modules — vérifier usages réels avant drop. |
| **webapp/** (Next.js 15) | ~6 000 TS | ✅ PROD (V2.4) | **KEEP** | Rebrand fait, chat live SSE, PWA, mobile-first, Vitest 40/40. |
| **webapp/lib/mocks.ts + types/** | — | 🟡 DETTE FRONT | **REPLACE** | Remplacer mocks par client OpenAPI typé (zod). |
| **infrastructure/** | — | ✅ PROD | **KEEP** | Entry point Dockerfile correct (`src.intelligence.main`), pas legacy. |
| **scripts/** (75) | — | 🔬 60 % actif | **PURGE 40 %** | Archiver scripts RL training, conserver eval/audit/data download. |
| **data/** | — | ✅ PROD | **KEEP+nettoyer** | Garder XAU_2019_2026, EUR_2019_2025, calendar, FRED, COT. Archiver feed 63 % (DG-004). |
| **models/** (7) | — | 🟡 MIXTE | **CURATE** | Garder calibrated_conviction_v1.pkl + scoring_v3_lgbm.pkl. Archiver le reste. |
| **tests/** (169) | — | ✅ PROD | **KEEP+TRIER** | Bon niveau. Tester quels touchent legacy RL → mise à jour. |
| **docs/governance/** | — | ✅ PROD | **KEEP** | Source de vérité plan d'exécution. |
| **parallel_training.py** (racine) | 1 634 | ❌ MORT | **DROP** | Premier candidat. |
| **15+ .md à la racine** (BUSINESS_PLAN, COMMERCIALIZATION_REPORT, INSTITUTIONAL_AUDIT, SPRINT_PLAN, AGENTS_SYSTEM_ANALYSIS, etc.) | — | ⚠️ POLLUTION | **ARCHIVE** | Déplacer dans `docs/archive/` ou supprimer si dans git. |

**Synthèse santé** :

| Catégorie | Décompte |
|---|---|
| ✅ KEEP tel quel | 14 sous-systèmes |
| 🟡 KEEP + WIRE / SPLIT / CONSOLIDATE | 8 |
| 🔬 KEEP en sandbox recherche | 3 |
| ⚠️ REFACTOR (god modules, doublons) | 4 |
| ❌ DROP (archive) | 5 (RL stack + parallel_training + pollution racine) |
| 🟡 DEFER décision | 2 (live_trading, webhook B2B) |

**Verdict global** : la codebase est **~60 % production-ready, ~25 % WIP utile, ~15 % legacy mort**. Le noyau (intelligence + api + delivery + webapp) est solide. La dette principale est concentrée dans `agents/*_integration.py`, `environment/environment.py`, `training/`, et 5+ classes "RiskManager" éparpillées.

## 1.3 Incohérences architecturales déjà visibles

### Doublons et confusion d'ownership

| # | Doublon | Sévérité | Action |
|---|---|---|---|
| 1 | **5+ classes "RiskManager"** : `agents/risk_sentinel.py`, `agents/intelligent_risk_sentinel.py`, `agents/portfolio_risk.py`, `agents/ensemble_risk_model.py`, `environment/risk_manager.py`, `risk/` modules | 🔴 HIGH | DG-039 expose `risk_score 0-100` + `kill_level` unique dans InsightSignalV2 mode EXPERT. Tous les autres deviennent internes ou archive. |
| 2 | `regime_filter.py` (180 L) **vs** `regime_gate.py` (120 L) | 🔴 HIGH | Audit : si redondants → DROP le moins utilisé. |
| 3 | `volatility_forecaster.py` (1 561 L, multi-modes HAR/LGBM/Hybrid) **vs** `volatility_lgbm.py` (180 L, alternative) | 🟡 MED | Décider standard unique. VOL_MODE=har en prod (eval_04 2026-04-29). DROP `volatility_lgbm.py`. |
| 4 | `intelligence/insight_v2/` (modèle InsightSignalV2) **vs** `api/insight_signal_v2.py` + `api/models.py` | 🔴 HIGH | Single source of truth = `intelligence/insight_v2/`. L'API importe depuis là. |
| 5 | `intelligence/semantic_cache.py` (LLM cache) **vs** `intelligence/rag/cache.py` (RAG embeddings cache) | 🟡 MED | Si distincts, clarifier nommage : `llm_response_cache` vs `rag_embedding_cache`. |
| 6 | `orchestrator.py` (sync) **vs** `async_orchestrator.py` | 🟢 LOW | Acceptable si sync utilisé en CLI/scripts et async en runtime API. À documenter. |
| 7 | 3 "integration" RL : `integration.py`, `intelligent_integration.py`, `orchestrated_integration.py` | 🔴 HIGH | Tous morts. DROP tous. |
| 8 | `intelligence/bocpd.py` (80 L, orphelin) **vs** logique BOCPD utilisée dans `regime_gate.py` | 🟡 MED | Vérifier appelants : si bocpd.py jamais importé, DROP. |
| 9 | `delivery/webhook_*.py` (queue, signer, drain) | 🟢 LOW | OK séparation. Mais pas wired en main → DEFER. |
| 10 | `intelligence/main.py` couple directement Telegram + Discord notifiers (DI manquante) | 🟡 MED | Introduire interface `IDeliveryChannel` + factory. |

### Couplages forts qui empêchent l'évolution

| Couplage | Impact |
|---|---|
| `main.py` instancie directement les notifiers concrets | Difficile de tester sans monkey-patcher. Bloquera l'ajout d'un canal (Discord, Email, Webhook B2B) sans toucher au noyau. |
| `api/routes/qa.py` importe directement `intelligence/rag/` | Route qui dépend de l'implémentation au lieu d'un service. Devra passer par `chat_service.ask()` propre. |
| `environment/strategy_features.py` (SmartMoneyEngine) vit dans `environment/` historiquement RL | Confusion. À promouvoir comme module de premier rang (`intelligence/smart_money/`). |
| `api/insight_signal_v2.py` réimplémente partiellement le modèle de `intelligence/insight_v2/` | Drift garanti à chaque évolution du contrat. |
| Webapp `lib/api/` parle à l'API via fetch hand-coded (pas de client généré OpenAPI) | Chaque modification du contrat backend casse silencieusement le front. |
| `intelligence/main.py` connaît les chemins de fichiers (CSV, DB) directement | Pas de DataProvider injecté = duplication des chemins entre scripts et prod. |
| `signal_state_machine.py` lit/écrit son propre fichier JSON | OK pour V1 mais bloque multi-worker (DG-024 DEFER). |

### Points de friction observables

1. **Scoring rule-based en prod ≠ scoring LGBM disponible** : `scoring_v3_lgbm.pkl` (81 KB) entraîné, audité, **mais non wired**. C'est le chemin critique #1 (DG-025).
2. **Architecture progressive vs vieilles surfaces** : webapp en architecture progressive uniforme, mais Telegram render encore basé sur l'ancien template "compact ≤ 800 chars" non hiérarchisé.
3. **`backtest_window` figé dans `historical_stats`** : "XAU M15 2019-2025 walk-forward" écrit en dur — devrait venir d'une config injectée.
4. **Frontend mocks vs API réelle** : `webapp/lib/mocks.ts` encore présent. La V2.4 livre l'UI sans vrai branchement signaux temps réel.
5. **i18n FR+EN** : webapp prêt côté tooling (`messages/`), mais Telegram renderer encore monolingue dans certains modules.

## 1.4 Verdict honnête final

> Le noyau (`src/intelligence/` + `src/api/` + `src/delivery/` + `webapp/`) est de **qualité production**.
> Autour, il y a **15 % de code mort** qu'on a peur de jeter et **25 % d'incohérences** qui ralentissent toute évolution.
> **Sans nettoyage architectural** : chaque nouveau canal de distribution (TradingView, mobile, B2B, …) sera bricolé en y ajoutant des couplages directs.
> **Avec un refactor ciblé de ~80-120h** (Partie 5), le système devient prêt à recevoir N canaux sans toucher au noyau.

Les 4 chantiers les plus urgents :
1. **Wire LGBM** : remplacer le scoring cosmétique (Pearson −0.023) par le pipeline calibré disponible.
2. **DROP du legacy RL** : archiver `parallel_training.py`, `src/training/`, `agents/*_integration.py` (zéro perte fonctionnelle, énorme gain de lisibilité).
3. **Unifier le contrat** : `intelligence/insight_v2/` devient source unique, `api/models.py` importe.
4. **Introduire les ports/adapters** : `IDeliveryChannel`, `IDataProvider`, `INarrativeEngine`, `ISignalStore` — afin que l'ajout d'un canal ne touche pas le noyau.

---

# Partie 2 — Architecture cible

## 2.1 Diagramme ASCII (vue système)

```
                                  ┌─────────────────────────────────────────────────────┐
                                  │                   CLIENT-FACING SURFACES                  │
                                  │                                                       │
   📱 Mobile WebApp (V1)   🌐 Desktop WebApp (V1)   💬 Telegram Bot (V1)                  │
   📊 TradingView Pine (V2)   🤖 Discord Bot (V2)   ✉️ Email Digest (V2)                  │
   🧩 Chrome Extension (V3)   🔌 API REST B2B (V3)  🪝 Webhooks B2B (V3)                  │
   📱 Mobile Native (V4)      🖥️ Desktop App (V4)    ⌚ Wearables (NEVER/V4)               │
   ───────────────────────────────────────────────────────────────────────────────────────
                                  ▲                ▲                ▲
                                  │                │                │
                          ┌───────┴───────┐ ┌─────┴──────┐  ┌──────┴─────┐
                          │  HTTPS / REST │ │ Bot APIs   │  │ Webhooks   │
                          │  + SSE chat   │ │ (long-poll)│  │ (HMAC)     │
                          └───────┬───────┘ └─────┬──────┘  └──────┬─────┘
                                  │               │                │
   ╔══════════════════════════════╪═══════════════╪════════════════╪═══════════════════╗
   ║                              ▼               ▼                ▼                   ║
   ║                  ┌────────────────────────────────────────────────────┐           ║
   ║                  │              EDGE / GATEWAY LAYER                  │           ║
   ║                  │  Cloudflare → Fly.io (Paris) → FastAPI (Gunicorn)  │           ║
   ║                  │  • CORS · GeoBlock · Rate-limit per tier           │           ║
   ║                  │  • Auth (API key + JWT) · Compliance disclaimer    │           ║
   ║                  └────────────────────────┬───────────────────────────┘           ║
   ║                                           │                                       ║
   ║                  ┌────────────────────────▼───────────────────────────┐           ║
   ║                  │              SERVICE LAYER (verbs)                 │           ║
   ║                  │  InsightService · ChatService · DeliveryService    │           ║
   ║                  │  BillingService · AccountService · AnalyticsService│           ║
   ║                  │  TrackRecordService · ComplianceService            │           ║
   ║                  └────┬───────────────┬──────────────────┬────────────┘           ║
   ║                       │               │                  │                        ║
   ║                       ▼               ▼                  ▼                        ║
   ║          ┌──────────────────┐  ┌─────────────┐  ┌────────────────────┐            ║
   ║          │  CORE PIPELINE   │  │ NARRATIVE   │  │  DELIVERY ENGINE   │            ║
   ║          │  (read-only DAG) │  │   ENGINE    │  │ (port/adapter)     │            ║
   ║          │                  │  │             │  │                    │            ║
   ║          │ DataProvider     │  │ LLMEngine   │  │ IDeliveryChannel   │            ║
   ║          │   ↓              │  │  Cascade:   │  │  ├─ TelegramAdapter│            ║
   ║          │ SmartMoneyEngine │  │  Haiku/     │  │  ├─ DiscordAdapter │            ║
   ║          │   ↓              │  │  Sonnet/    │  │  ├─ EmailAdapter   │            ║
   ║          │ VolatilityFore-  │  │  Opus       │  │  ├─ WebhookAdapter │            ║
   ║          │   caster         │  │             │  │  ├─ TVAdapter      │            ║
   ║          │   ↓              │  │ Template-   │  │  └─ MobilePushAdpt │            ║
   ║          │ RegimeGate +     │  │  Engine     │  │                    │            ║
   ║          │   BOCPD + HMM    │  │  (fallback) │  │ Idempotency store  │            ║
   ║          │   ↓              │  │             │  │ Retry+backoff      │            ║
   ║          │ ConfluenceDetect.│  │ Forbidden-  │  │ HMAC signing       │            ║
   ║          │   ↓              │  │  token guard│  │                    │            ║
   ║          │ Calibrated-      │  │             │  │                    │            ║
   ║          │  Conviction      │  │ RAG retrieve│  │                    │            ║
   ║          │  (LGBM→Iso→ACI)  │  │ → cite      │  │                    │            ║
   ║          │   ↓              │  │             │  │                    │            ║
   ║          │ NewsAnalysis     │  └─────┬───────┘  └────────┬───────────┘            ║
   ║          │   ↓              │        │                   │                        ║
   ║          │ SignalState-     │        │                   │                        ║
   ║          │   Machine        │        ▼                   ▼                        ║
   ║          │   ↓              │   ┌─────────────────────────────────┐                ║
   ║          │ InsightAssembler │──▶│   InsightSignalV2 (contrat)     │                ║
   ║          └──────┬───────────┘   └──┬──────────────────────────────┘                ║
   ║                 │                  │                                                ║
   ║                 ▼                  ▼                                                ║
   ║          ┌─────────────┐    ┌───────────────────────────────────────────┐          ║
   ║          │ SignalStore │    │             RENDERERS (per-canal)         │          ║
   ║          │ (SQLite→PG) │    │  to_focus_card · to_copilot_card          │          ║
   ║          │             │    │  to_expert_full · to_telegram_b2c         │          ║
   ║          │ Audit table │    │  to_discord_embed · to_email_digest       │          ║
   ║          │ Idempotency │    │  to_tradingview_alert · to_b2b_json       │          ║
   ║          └─────────────┘    │  to_mobile_push · to_extension_overlay    │          ║
   ║                             └───────────────────────────────────────────┘          ║
   ║                                                                                    ║
   ║   ┌────────────────────────────────────────────────────────────────────────┐       ║
   ║   │             TRANSVERSE SERVICES (cross-cutting concerns)                │       ║
   ║   │                                                                        │       ║
   ║   │  🔐 Auth (Clerk/Supabase)   💳 Billing (Stripe)   📊 Analytics (Plausible)│     ║
   ║   │  📡 Observability (Sentry + OTel)   ⚖️ Compliance Service              │       ║
   ║   │  🔑 Secrets (Fly.io)        🧠 LLM Router (Anthropic SDK)              │       ║
   ║   │  💾 Cache (Redis V2+)       🗃️ Vector Store (Chroma/Qdrant V2+)        │       ║
   ║   └────────────────────────────────────────────────────────────────────────┘       ║
   ╚════════════════════════════════════════════════════════════════════════════════════╝

   Légende :
     ──▶  flux de données   ◀──  appel inverse   ░░  données fraîches    ║║  périmètre app
```

## 2.2 Récit d'architecture

### 2.2.1 Le noyau (Core Pipeline)

C'est un **DAG read-only** qui produit des `InsightSignalV2` à partir d'une barre OHLCV close. Il **ne sait rien des canaux de livraison** ni des clients.

```
DataProvider          (CSV / MT5 / live)
  ▼
SmartMoneyEngine      (BOS, CHOCH, FVG, OB, retest) — Numba-optimized
  ▼
VolatilityForecaster  (HAR-RV + diurnal + calendar + HMM mult)
  ▼
RegimeGate            (HMM 3-état + BOCPD + jump_ratio Barndorff-Nielsen)
  ▼
ConfluenceDetector    (8 composantes → score brut, mode A)
  ▼
CalibratedConviction  (LGBM → Isotonic → ACI) — mode B, à wirer
  ▼
NewsAnalysisAgent     (calendar blackout + sentiment 24h)
  ▼
SignalStateMachine    (hysteresis, cooldown, lifetime, opposing-lockout)
  ▼
InsightAssembler      (compose InsightSignalV2 final)
```

**Principes** : déterminisme (signal_id SHA-1), reproductibilité bit-à-bit, zéro effet de bord, zéro I/O direct (passe par DataProvider injecté). C'est le seul code qui doit être ré-exécutable historiquement sans dépendre du temps présent.

### 2.2.2 La couche de service (verbs)

Au-dessus du noyau vivent des **services** qui orchestrent les cas d'usage métier. Chaque route API et chaque worker passe par un service — jamais directement par le noyau.

| Service | Verbes exposés | Dépend de |
|---|---|---|
| `InsightService` | `get_current(symbol)` · `get_history(symbol, range)` · `replay(symbol, ts)` | Core pipeline, SignalStore |
| `ChatService` | `ask(question, context)` · `stream(...)` · `suggest_questions(signal)` | LLMEngine, RAG, current InsightSignalV2 |
| `DeliveryService` | `publish(insight, channels[])` · `subscribe(user, channel)` · `unsubscribe(...)` | DeliveryEngine (ports), SignalStore |
| `BillingService` | `start_trial(user, tier)` · `upgrade(...)` · `cancel(...)` · `usage(user)` | Stripe API, AccountService |
| `AccountService` | `register(...)` · `login(...)` · `update_prefs(...)` · `gdpr_export(user)` | Auth provider, UserStore |
| `TrackRecordService` | `get_public_pf(window)` · `get_user_paper_pnl(user)` · `monthly_report(...)` | SignalStore, OutcomeStore |
| `ComplianceService` | `disclaimer(lang)` · `is_geoblocked(ip)` · `audit_record(...)` | GeoIP, compliance config |
| `AnalyticsService` | `track(event, props)` · `funnel(user)` · `cohort(window)` | Plausible self-hosted |

**Règle** : pas de logique business dans les routes API. Une route = parse + auth + call service + render. Une route fait < 30 lignes.

### 2.2.3 La couche d'API (endpoints)

Couche fine FastAPI. Routes groupées par domaine.

| Route | Endpoints | Service |
|---|---|---|
| `/api/v1/insights/` | `GET current/{symbol}`, `GET history/{symbol}`, `GET {signal_id}`, `GET {id}/breakdown` | InsightService |
| `/api/v1/chat/` | `POST ask` (SSE stream), `GET suggestions/{signal_id}` | ChatService |
| `/api/v1/track-record/` | `GET public`, `GET user/me`, `GET monthly/{yyyy-mm}` | TrackRecordService |
| `/api/v1/subscriptions/` | `POST checkout`, `POST cancel`, `GET status`, `GET usage` | BillingService |
| `/api/v1/account/` | `POST register`, `POST login`, `GET me`, `PATCH prefs`, `GET gdpr-export` | AccountService |
| `/api/v1/webhooks/` (B2B) | `POST subscribe`, `DELETE unsubscribe`, `GET dlq` | DeliveryService |
| `/api/v1/legal/` | `GET terms`, `GET privacy`, `GET disclaimer/{lang}` | ComplianceService |
| `/api/v1/health/`, `/metrics`, `/healthz/deep` | (operational) | infra |

**Versionnage** : v1 figé une fois publié. v2 = nouvelle URL en parallèle, header `Accept: application/vnd.mia.v2+json` accepté.

### 2.2.4 La couche de delivery (ports/adapters)

Un seul **moteur de livraison** lit `InsightSignalV2`, choisit les adapters cibles selon les souscriptions, et publie. Aucun renderer ne sait rien du contenu fonctionnel — il fait juste la projection.

```python
class IDeliveryChannel(Protocol):
    name: str
    def render(self, insight: InsightSignalV2, locale: str) -> Payload: ...
    async def publish(self, payload: Payload, target: ChannelTarget) -> DeliveryResult: ...
```

Adapters : `TelegramAdapter`, `DiscordAdapter`, `EmailAdapter`, `WebhookAdapter`, `TradingViewAdapter`, `MobilePushAdapter`, `ExtensionOverlayAdapter`.

Chaque adapter sait :
- comment **render** l'insight dans son format (markdown Telegram ≤ 800 chars, embed Discord, Pine alert JSON, etc.),
- comment **publier** au transport (HTTP, bot API, push notification gateway),
- comment **gérer ses erreurs** (retry, backoff, DLQ).

### 2.2.5 Les interfaces clientes (surfaces)

- **Webapp** (Next.js 15) : C1+C2+C3 sur la même donnée. Hero card permanent + sections collapsibles + chatbot sidebar/FAB.
- **Bot Telegram** : C1 compact + lien webapp pour C2/C3 + accès au chat via bot conversationnel.
- **Bot Discord** : C1 embed + serveur communautaire pour rétention.
- **Email digest** : C1 quotidien + lien webapp.
- **TradingView Pine** (V2) : alert minimal "lecture haussière X · invalidation Y" + lien webapp.
- **Extension navigateur** (V3) : overlay C1 sur sites trading.
- **API B2B** (V3+) : payload complet machine-readable + webhooks.
- **Application mobile native** (V4 conditionnel) : C1+C2+C3 fluides, push notifications natives.

### 2.2.6 Les services transverses

Chacun est **un service indépendant** branché par interface :
- **Auth** : Clerk (recommandé V1) ou Supabase Auth. JWT bearer. Pas d'auth maison.
- **Billing** : Stripe (locked, DG-043). Customer Portal pour self-service.
- **Observability** : Sentry (free tier, KEEP V1). OTel + Tempo DEFER team > 1.
- **Analytics** : Plausible self-hosted (DG-160, P0-strict V1) + event tracking core 6 events (DG-161).
- **LLM Router** : Anthropic SDK avec routing par tier (Haiku VISUAL / Sonnet NARRATOR / Opus INSTITUTIONAL).
- **Cache** : in-memory V1, Redis V2 (DG-020 DEFER MAU > 200).
- **Vector store** : Chroma local V1 (12 papers, DG-058a), Qdrant V2+ (DG-058b).
- **Secrets** : Fly.io secrets natifs V1 ; Doppler/Vault DEFER MRR > $5k.

## 2.3 Contrats de données

**Principe** : un seul contrat canonique riche (`InsightSignalV2`), N projections par canal.

```
                       ┌────────────────────────────────────────┐
                       │   InsightSignalV2 (canonical, full)    │
                       │   Pydantic v2, ~12 sous-modèles         │
                       │   src/intelligence/insight_v2/         │
                       └───────────────┬────────────────────────┘
                                       │
       ┌───────────────────┬───────────┼─────────────────┬──────────────────┐
       ▼                   ▼           ▼                 ▼                  ▼
┌─────────────┐    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐
│ FocusCard   │    │ CopilotCard  │  │ ExpertFull   │  │ TelegramB2C  │  │ B2B JSON      │
│ ≤ 200 chars │    │ ~ 6 sections │  │ + breakdown  │  │ ≤ 800 chars  │  │ Full payload  │
│ (V1 web/    │    │ (V1 web)     │  │ + waterfall  │  │ HTML/MD      │  │ + scenarios   │
│  email/SMS) │    │              │  │ + sources    │  │              │  │ + telemetry   │
└─────────────┘    └──────────────┘  └──────────────┘  └──────────────┘  └───────────────┘
       │                   │                  │                │                  │
       ▼                   ▼                  ▼                ▼                  ▼
┌──────────────┐  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐  ┌────────────┐
│ Mobile push  │  │ Discord embed  │  │ TV Pine alert  │  │ Extension over- │  │ Webhook    │
│ payload      │  │ JSON           │  │ JSON           │  │ lay JSON        │  │ B2B push   │
└──────────────┘  └────────────────┘  └────────────────┘  └─────────────────┘  └────────────┘
```

Chaque projection a un **schéma versionné** (`v1`, `v2`) et un **renderer pur** (input = `InsightSignalV2`, output = `Payload`).

**Versioning** :
- `InsightSignalV2` versionné en MAJOR.MINOR (ex. 2.1.0).
- Chaque projection (TelegramB2C v2.1, B2BJSON v3.0, …) versionnée indépendamment.
- Changement breaking → nouvelle MAJOR sur la projection concernée, pas sur le contrat canonique.
- Header de réponse API : `X-Schema-Version: insight_v2_2.1.0`.

Documents détaillés par contrat : `docs/architecture/CONTRACTS/*.md` (livré en Partie 4+).

## 2.4 Principes architecturaux non-négociables

1. **Découplage pipeline / livraison** : un seul moteur, N surfaces. Aucun adapter ne reformule l'algorithme.
2. **Idempotence des publications** : un même `signal_id` ne produit qu'une notification par canal (clé d'idempotence DB).
3. **Pas de logique business dans les renderers** : un renderer projette, point. S'il faut "décider", c'est qu'on a un service à faire ailleurs.
4. **Pas d'appel LLM hors du `LLMNarrativeEngine`** : aucun service ne parle à Anthropic directement.
5. **Pas d'I/O direct depuis les routes** : toujours via service layer. Une route ≤ 30 lignes.
6. **Reproductibilité bit-à-bit** : `signal_id = SHA-1(symbol|bar_ts|direction|score:.4f)[:12]`. Rejouer une barre 6 mois plus tard produit le même ID.
7. **Compliance par construction** : `ComplianceService` injecte disclaimers + filtre les forbidden tokens. Pas de wrapping post-hoc.
8. **Mobile-first** : tout endpoint client doit servir une réponse usable sur 375px en < 200 ms (cache hit).
9. **Bilingue FR+EN J1** : tout contenu utilisateur passe par i18n. Pas de chaîne hardcodée.
10. **Un seul contrat source de vérité** : `intelligence/insight_v2/`. Tout le reste importe.
11. **Auditabilité** : chaque insight publié écrit dans `SignalStore` (immutable). Replay = re-read, jamais recalcul depuis zéro.
12. **`edge_claim=False` codifié** : tant que les 4 critères (PF rolling > 1.20, DSR > 1.0, PBO < 0.5, walk-forward ≥ 2 ans hors-sample) ne sont pas franchis, aucune surface ne peut afficher "edge prouvé".

---

# Partie 3 — Inventaire des canaux de distribution

## 3.1 Matrice complète (21 canaux évalués)

Chaque canal noté sur 9 dimensions. **Effort** et **Coût** ordonnés en classes (TF/F/M/E/TE). **Score différenciation** et **Score ROI** sur 1-5. **Verdict** ∈ {V1, V2, V3, V4, NEVER}.

| # | Canal | Effort | Coût/an | Tier cible | Forme | Latence | Risque compliance | Diff. | ROI | Dépendances | Verdict | Justification |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Webapp SaaS (Next.js)** | F (déjà livré V2.4) | $0-$240 (Vercel free+Cloudflare) | FREE→INST | Pull+SSE chat | < 1s temps réel | Moyen (compliance UE inline) | 5 | 5 | API REST | **V1** | Surface principale. Tout le reste pointe ici. Déjà fait à 80 %. |
| 2 | **Bot Telegram** | F (déjà livré) | $0 | FREE→PRO | Push + chat conv. | < 5s | Moyen (geo-block actif) | 4 | 5 | Bot API, queue | **V1** | Canal FREE essentiel, conversion vers paid. Maintenance faible. |
| 3a | **TradingView Pine indicator showcase (pur, no API)** | F (20-30h) | $0 (compte TV gratuit OK) | FREE→PRO (acquisition) | Pull + lien webapp | N/A (statique sur chart) | Faible (Pine local, pas de wording externe) | 5 | 5 | Compte TV + Pine script signé + profil soigné | **V1** | Hub où vit la cible. Acquisition gratuite via communauté TradingView (publication, profil, tags `SMC`, trending). Pattern prouvé (LuxAlgo, BigBeluga, etc.). Zéro infra côté MIA. |
| 4 | **TradingView webhook receiver (alerts → cross-canal)** | M (30-40h) | $0 + TV Pro side $15/mo user | PRO+ (rétention) | Push alert | < 10s | Moyen (alert wording compliance) | 4 | 4 | Webhook endpoint, normalizer, dedup | **V2** | Connecte les alerts TV des users payants au flux MIA (Telegram, mobile push). Distinct du showcase : nécessite infra côté MIA. |
| 3b | **Email digest quotidien (marketing/rétention)** | F | $60-$300 (Resend/Postmark digest tier) | STARTER+ | Push + lien webapp | Asynchrone (matin) | Faible | 3 | 4 | Email service + template + audience ≥ 100 users | **V2** | Canal de **rétention**, pas d'acquisition. Construire en V1 = canal sans audience = effort wasted. Activer quand MAU > 100. (Note : email **transactionnel** Welcome/Stripe/Trial-end reste infra V1, voir §3.5.) |
| 5 | **Bot Discord** | F | $0 | FREE→PRO | Push + chat conv. | < 5s | Moyen (idem TG) | 3 | 3 | Bot API | **V2** | Communauté retention. Moins prio que Telegram en FR-first. |
| 6 | **API REST publique B2B** | M | $0 (mutualisé) | INSTITUTIONAL | Pull | < 200ms p95 | Faible | 4 | 3 (B2B-dependent) | OpenAPI, tier auth | **V3** | Pré-mature avant traction B2C (DG-071 DEFER MRR B2C > $5k 3 mois). |
| 7 | **Webhooks B2B (push)** | F (infra déjà signée HMAC) | $0 | INSTITUTIONAL | Push | < 30s | Faible | 4 | 3 | DeliveryAdapter + queue Redis | **V3** | Infra prête (cf. §1.2). Activer quand DG-071 déclenché. |
| 8 | **Mobile cross-platform (React Native ou Flutter)** | E | $0 dev + $99 Apple + $25 Google | STARTER+ | Push + UI complète | < 1s temps réel | Moyen (idem webapp) | 4 | 4 | API REST, push notif gateway (FCM/APNs) | **V2 (M6-M9)** | 60-70 % du retail mobile, mais investissement lourd. Reporter post-PMF webapp validée. |
| 9 | **PWA installable (déjà partie webapp)** | TF (déjà livré V2.3) | $0 | FREE→INST | Push + UI | < 1s | Idem webapp | 3 | 4 | manifest + SW | **V1** | Quick win — PWA donne 70-80 % de l'UX mobile native sans dev natif. Couvre le besoin mobile en V1. |
| 10 | **SMS alertes premium** | F | $0,03-0,06/SMS | PRO+ | Push | < 30s | Élevé (opt-in CNIL strict) | 2 | 2 | Twilio/Vonage | **V3** | Coût par message tue la marge FREE. Réserver event-critique payant. |
| 11 | **Extension Chrome/Firefox** | E | $5 dev Chrome + $0 Firefox + maintenance | PRO+ | Pull + overlay | < 2s | Moyen (CSP sites tiers) | 5 | 3 | API REST, manifest v3 | **V3** | Différenciation forte ("MIA overlay sur TradingView/Bloomberg") mais maintenance compatibilité élevée. Reporter. |
| 12 | **Widget embeddable (script JS)** | M | $0 | INSTITUTIONAL B2B | Pull | < 1s | Moyen (iframe + disclaimer obligatoire) | 4 | 3 | API REST, CSP | **V3** | Couplé au B2B-API. Brokers/éducateurs intègrent un encart. |
| 13 | **MetaTrader 4/5 EA/indicateur** | E | $0 | PRO+ | Push alert | < 5s | Élevé (MT5 communauté = vente signaux régulée) | 3 | 2 | MQL5 dev, webhook receiver | **V3 (conditionnel)** | Reach énorme mais compliance MT5 = piège. Vérifier avec avocat avant. |
| 14 | **NinjaTrader / cTrader / Sierra Chart** | E | $0 | PRO+ | Push alert | < 5s | Idem MT5 | 2 | 1 | SDK propriétaire par plateforme | **V4** | Audience trop petite par plateforme, effort multiplié par N plateformes. |
| 15 | **Plugin WordPress** | M | $0 | INSTITUTIONAL B2B | Pull | < 2s | Moyen | 3 | 2 | API REST | **V3** | Niche B2B (brokers/éducateurs WP). Faible volume. |
| 16 | **Application desktop (Electron type Bloomberg-light)** | E | $0 | INST | UI complète | < 1s | Moyen | 3 | 1 | API REST, Electron build, code-signing certificat | **V4** | Web est suffisant. Desktop = re-package, faible valeur ajoutée. |
| 17 | **Apple Watch / wearables** | M | $99 Apple dev | PRO+ | Push alert | < 5s | Faible | 2 | 1 | Mobile native iOS | **NEVER** | Gimmick. Trader ne décide pas via une montre. Mention marketing OK, dev NON. |
| 18 | **Alexa / Google Home (briefing vocal)** | E | $0 | STARTER+ | Pull vocal | < 3s | Moyen (TTS du disclaimer obligatoire) | 4 | 1 | Skill/Action dev, TTS Anthropic | **V4** | Gadget intriguant mais audience résiduelle (< 1 % traders). Brand-play, pas revenue-play. |
| 19 | **ChatGPT plugin / Claude tool integration** | M | $0 | STARTER+ | Conversational pull | < 3s | Élevé (LLM tiers peut altérer le disclaimer) | 5 | 3 | OpenAPI public, OAuth | **V3** | Reach énorme via ChatGPT/Claude, mais perte de contrôle sur compliance wording. Étudier sérieusement post-PMF. |
| 20 | **Push web (Web Push API)** | TF | $0 | STARTER+ | Push | < 5s | Faible | 2 | 3 | SW déjà en place | **V2** | Faible effort, complète le PWA. Notification d'alerte event imminent. |
| 21 | **Audio briefing podcast quotidien (TTS automatisé)** | F | $0 + $20/mo ElevenLabs ou Anthropic TTS | STARTER+ | Pull | Asynchrone (matin) | Faible | 4 | 2 | TTS service, RSS hosting | **V4** | Différenciant brand, mais nichaire. À tester en V4 si bande passante. |

### Synthèse des verdicts (post-correction utilisateur 2026-05-27)

| Vague | Count | Canaux |
|---|---|---|
| **V1 (MVP 0-3 mois)** | 4 | Webapp SaaS · PWA installable · Bot Telegram · TradingView Pine showcase |
| **V2 (Acquisition 3-9 mois)** | 5 | Email digest (rétention) · TradingView webhook receiver · Bot Discord · Mobile cross-platform · Push web |
| **V3 (Expansion 9-18 mois)** | 8 | API REST B2B · Webhooks B2B · SMS premium · Extension navigateur · Widget embeddable · MT4/5 (cond.) · Plugin WordPress · ChatGPT/Claude tool |
| **V4 (Innovation 18+ mois)** | 4 | NinjaTrader/cTrader · Desktop Electron · Alexa/Google Home · Audio podcast TTS |
| **NEVER** | 1 | Apple Watch / wearables |

## 3.2 Justifications des arbitrages structurants

### Pourquoi TradingView Pine showcase en V1 (re-justifié post-correction)

**Le contre-point utilisateur tient et je m'incline.** L'objection initiale ("Pine ne parle pas à une API externe") visait la **version hybride** (Pine + webhook receiver). Mais le canal qui compte pour l'acquisition est différent : c'est le **Pine showcase pur, sans aucune API externe**.

Argumentaire V1 reconsidéré :
- **TradingView est le hub où vit la cible** (B2C retail FX/Gold). Toute la communauté SMC FR + EN y publie. Ignorer ce canal au démarrage = laisser 3-9 mois d'acquisition gratuite sur la table.
- **Le pattern showcase Pine pur est prouvé** : LuxAlgo, BigBeluga, KIVANC, ChartArt, etc. ont tous percé via un script Pine gratuit publié + profil TradingView soigné + lien webapp en description. Aucune dépendance API externe.
- **Effort réel = 20-30 h**, pas 30-50 h : il s'agit d'un script Pine qui détecte les BOS/CHOCH/FVG/OB **localement sur le chart de l'utilisateur** (calcul Pine natif), affiche un score simplifié (5 facteurs au lieu de 8, sans calibration LGBM), et renvoie l'utilisateur vers `mia.markets` pour la version conversationnelle complète. Zéro infra côté MIA.
- **Compliance facile** : pas de wording prescriptif (zones visuelles + niveau), disclaimer dans la description du script.
- **Acquisition gratuite mesurable** : TradingView fournit views, favorites, ratings — funnel observable dans Plausible via UTM `?source=tradingview`.

**La version V2** garde son sens distinct : **webhook receiver** côté MIA qui reçoit les alerts Pine envoyées par les users payants (TV Pro). C'est le pont qui amène les alerts TV dans le flux MIA cross-canal (Telegram, mobile push, etc.). C'est de la **rétention/intégration**, pas de l'acquisition.

### Pourquoi Email digest en V2, pas V1 (re-justifié post-correction)

**Tu as raison, je m'incline.** Un email digest quotidien adressé à 0 abonné = effort sans retour. Avant ~100 MAU, l'email marketing coûte plus en setup qu'il ne rapporte en rétention.

Argumentaire V2 reconsidéré :
- **Le digest est un canal de rétention**, pas d'acquisition. Sa courbe de valeur démarre à ~50-100 abonnés actifs (en dessous, le coût opérationnel — template maintenance, soft-bounces, désinscriptions, opt-in CNIL — dépasse le bénéfice).
- **Coût opportunité V1** : 12-20 h investies en V1 dans un digest = 12-20 h pas investies sur Telegram, TradingView, ou le chatbot. ROI immédiat plus fort ailleurs.
- **Activation V2 conditionnée à MAU > 100** : à ce stade, on a une audience qui justifie le templating + l'A/B test sujet ligne, et la rétention M1 commence à compter.

**Note** : l'**email transactionnel** (Welcome, Stripe receipts, trial-end notice, GDPR-export) reste un service d'infrastructure obligatoire en V1 (DG-043 Stripe live + DG-038 DSAR). Ce n'est pas un canal de distribution au sens de cette matrice — c'est plumbing du funnel signup. Cf. §3.5 ci-dessous.

### Pourquoi PWA en V1 et Mobile natif en V2 — limitations explicites (re-détaillé post-correction)

La PWA (manifest + Service Worker déjà livrés V2.3) **couvre les besoins V1**, mais **avec des limitations réelles** qu'il faut documenter pour savoir exactement quand le natif sera justifié (et pas juste "parce que c'est plus joli").

#### 🍎 Limitations iOS (Safari Mobile) — sacrifices PWA V1

| # | Limitation | Impact | Sévérité V1 |
|---|---|---|---|
| 1 | **Web Push iOS 16.4+ obligatoire** (mars 2023) | Tous les iPhones < iOS 16.4 (env. 5 % parc actif fin 2026) ne reçoivent aucune notification. iPhone 7/8 obsolètes ne supportent pas iOS 16. | 🟡 Acceptable (95 % du parc). |
| 2 | **Push impossible sans "Add to Home Screen" préalable + opt-in explicite** | iOS exige que la PWA soit installée (geste manuel utilisateur), PUIS que l'utilisateur autorise les notifications. Funnel à 2 étapes vs 1 étape en natif. Taux d'opt-in attendu : 5-15 % (vs 40-60 % natif). | 🔴 **Critique pour event alerts** (FOMC ≤ 4h, DG-122). Si rétention dépend des push, V2 mobile natif s'impose plus tôt. |
| 3 | **Pas d'icône badge unread count** | Impossible d'afficher "3 nouveaux insights" sur l'icône MIA. Utilisateur doit ouvrir l'app pour voir. | 🟡 Cosmétique mais réduit ré-engagement. |
| 4 | **Service Worker tué après 7 jours d'inactivité** | Si user ouvre MIA puis revient 10 jours après → SW dormant → première interaction = cold start (latence +500ms à +2s). | 🟡 Acceptable, observable via Plausible. |
| 5 | **Pas dans l'App Store** | Zéro discoverabilité via recherche Apple ("MIA Markets" dans App Store = vide). Pas de notes/reviews sociales. | 🔴 **Crédibilité brand limitée pour prospects B2B** (qui googlent "MIA Markets app" sans résultat). Accélère V2 si signal B2B observé. |
| 6 | **Pas d'Apple Pay one-tap** | Stripe Checkout fonctionne en web mais l'expérience iOS native (Apple Pay 1-tap) reste plus fluide pour conversion paid. | 🟡 Acceptable, Stripe Checkout convertit bien. |
| 7 | **Splash screen statique** (screenshot) | Pas d'animation native au launch. UX moins "premium". | 🟢 Mineur. |
| 8 | **Pas de Face ID/Touch ID natif** sauf via Passkeys (récent, partiellement supporté) | Login par mot de passe + magic link OK V1 ; biométrie demandera natif. | 🟡 Acceptable V1, sera demandé en V2 par power-users. |
| 9 | **Background sync limité** : pas de pre-fetch périodique | Insights ne sont pas rafraîchis en arrière-plan. Utilisateur voit toujours le dernier au moment d'ouverture. | 🟢 Acceptable (notre fréquence M15 = 4 insights/h max). |
| 10 | **Pas d'accès contacts/photos/files/audio background** | Limites fortes pour features avancées (partager insight via WhatsApp, exporter PDF dans Files, briefing vocal en BG). | 🟢 Hors-scope V1. |

#### 🤖 Limitations Android (Chrome Mobile) — sacrifices PWA V1

| # | Limitation | Impact | Sévérité V1 |
|---|---|---|---|
| 1 | **Pas dans le Play Store nativement** | Sauf via TWA (Trusted Web Activity, wrapper Bubblewrap) qui demande build Android et compte Google Play Developer ($25 one-time). | 🟡 Possible V1.5 si crédibilité brand le justifie ($25 + 4-6 h dev TWA). |
| 2 | **Push retardés par "Doze Mode"** (batterie faible) | Une alerte FOMC ≤ 4h peut être livrée avec 15-30 min de retard si l'Android est en deep doze. | 🟡 Acceptable, pas critique sur fenêtre 4h. |
| 3 | **Add-to-home-screen prompt UX moins propre** que iOS | Chrome propose via banner discret, beaucoup d'utilisateurs ignorent. Install rate attendu : 5-10 % (vs ~15 % iOS). | 🟡 Acceptable, mesurable. |
| 4 | **Pas d'icône badge unread count** sur la plupart des launchers Android | Idem iOS. | 🟡 Cosmétique. |
| 5 | **Biométrie WebAuthn fonctionne mais UX moins fluide** que natif | OK pour V1. | 🟢 Mineur. |
| 6 | **Discovery zéro hors TWA** | Identique iOS hors Play Store. | 🔴 Idem crédibilité brand. |

#### 🎯 Quand passer en Mobile natif (V2) sera vraiment justifié

Le mobile natif (React Native, choix argumenté en Partie 6) s'impose si **au moins 2 des 5 conditions** suivantes sont franchies :

1. **> 30 % du trafic webapp vient du mobile** ET **> 40 % du chiffre vient d'utilisateurs ayant installé la PWA** (mesurable via Plausible event `pwa_installed` + revenu Stripe corrélé).
2. **Taux d'opt-in push iOS < 20 %** mesuré sur cohortes installées PWA, ET rétention M1 corrélée fortement à la réception de push (Plausible : users push-actifs vs push-inactifs).
3. **Demandes utilisateurs explicites Face ID/Touch ID** ≥ 10 cas documentés (issues GitHub, mails support, chat MIA).
4. **Prospects B2B ou éducateurs influents** (≥ 3 cas) qui mentionnent l'absence d'App Store comme blocage à l'adoption (perception de sérieux).
5. **MRR > $5k 3 mois consécutifs** : la trésorerie permet d'investir 150-250 h dans une app native sans compromettre les autres axes.

Si **0-1 condition** atteinte : rester PWA en V2 et investir ailleurs (TradingView, B2B). Si **2+** : mobile natif devient P0 V2.

### Pourquoi Telegram en V1 mais Discord en V2 ?

Telegram = wedge FR-first identifié (eval_25). Marché retail FR sur-représenté Telegram (canaux signaux). Discord = US/EN-dominant, plus communauté. **En V1 on suit l'audience, en V2 on l'élargit.** Cohérent aussi avec la stratégie bilingue FR+EN J1 : la webapp est bilingue J1, Telegram FR-dominant en V1, Discord prend le relais EN/community en V2.

### Pourquoi API B2B en V3 ?

C'est la décision politique DG-071 : pivot B2B **conditionnel à MRR B2C > $5k 3 mois consécutifs**. Avant, construire une API B2B = feature pour un produit inexistant. L'infra webhook B2B (signing HMAC, queue) existe déjà — on l'active à la demande. **Pas d'effort wasted, mais pas de promesse marketing avant traction.**

### Pourquoi Apple Watch en NEVER ?

Un trader ne décide pas via une montre. La promesse "alerte poignet" est un gadget marketing, pas un cas d'usage. Coût de maintenance (Watch OS évolue chaque année) ≫ valeur. **À mentionner en mockup brand-play, jamais dev.**

### Pourquoi ChatGPT plugin / Claude tool en V3 ?

Reach énorme (250M+ utilisateurs ChatGPT, 10M+ Claude), mais **perte de contrôle sur le wording de compliance** (le LLM tiers peut reformuler le disclaimer et nous mettre en infraction). Étude sérieuse à mener post-PMF B2C, peut-être avec un mode "MIA Tool restreint" qui ne renvoie que des cards finalisées non-reformulables par le LLM tiers.

### Pourquoi MT4/5 en V3 conditionnel ?

Audience massive (50M+ traders MT4/5 dans le monde) mais l'écosystème MT5 est saturé de vendeurs de signaux régulés. Notre wording "analyse pas signal" risque d'être brouillé par la convention de l'écosystème. **Avant tout dev MT5, vérifier avec un avocat fintech FR la posture défendable.**

### Pourquoi Push web en V2 ?

Effort très faible (Web Push API + service worker déjà en place V2.3). Complète le PWA. Permet une alerte "FOMC dans 1h" sans dépendre du natif. **Le seul reason de ne pas le mettre en V1 est qu'il faut un Plausible event = engagement chat avant d'arroser en push.**

## 3.3 Cinq vagues de déploiement

```
VAGUE 1 — MVP (mois 0-3)  ◆◆◆◆ 4 canaux                                       │
├─ 🌐 Webapp SaaS (PWA installable inclus)                                     │ Objectif :
├─ 💬 Bot Telegram (FR-first wedge)                                            │ 1er
└─ 📊 TradingView Pine showcase (acquisition gratuite via hub)                  │ paiement

VAGUE 2 — Acquisition (mois 3-9)  ▲▲▲▲▲ 5 canaux                                │
├─ ✉️ Email digest quotidien (rétention, déclenche à MAU > 100)                 │ Objectif :
├─ 🪝 TradingView webhook receiver (alerts TV → cross-canal MIA)                │ MRR > $5k
├─ 🤖 Bot Discord (EN/community)                                                │ MAU > 200
├─ 📱 Mobile cross-platform (React Native, choix Partie 6)                      │
└─ 🔔 Push web (alerte event imminent)                                          │

VAGUE 3 — Expansion (mois 9-18)  ■■■■■■■■ 8 canaux                              │
├─ 🔌 API REST B2B publique           ┐                                         │ Objectif :
├─ 🪝 Webhooks B2B push                ├─ pivot B2B sur DG-071 trigger          │ B2B
├─ 🧩 Extension Chrome/Firefox        │                                         │ traction
├─ 📦 Widget embeddable               │                                         │
├─ 📲 SMS premium                      │                                         │
├─ 🪟 Plugin WordPress                 │                                         │
├─ 🤖 ChatGPT/Claude tool integration │                                         │
└─ 🏦 MetaTrader 4/5 (conditionnel)    ┘                                         │

VAGUE 4 — Innovation (mois 18+)  ◇◇◇◇ 4 canaux                                  │
├─ 🖥️  Application desktop (Electron)                                            │ Objectif :
├─ 📈 NinjaTrader/cTrader/Sierra                                                │ Élargir
├─ 🔊 Alexa / Google Home (briefing vocal)                                       │ brand
└─ 🎙️  Audio podcast quotidien (TTS)                                            │

NEVER  ❌                                                                       │
└─ ⌚ Apple Watch / wearables — Gadget marketing, jamais dev                    │
```

## 3.4 Implication architecturale

L'architecture cible (Partie 2) **doit** être prête à recevoir tous ces canaux **sans toucher au noyau** :

- Un seul `InsightSignalV2` produit le contenu.
- Un nouvel adapter = nouveau fichier `delivery/adapters/<canal>_adapter.py` + nouveau renderer `renderers/to_<canal>.py`.
- Aucun changement dans `core/pipeline` ni dans `services/`.

**C'est le seul critère qui valide l'architecture** : si je veux ajouter Discord demain, combien de fichiers je touche en dehors du dossier `delivery/` ? Cible : **0** (à part la config de subscriptions).

## 3.5 Note : email transactionnel ≠ email digest

Pour clarifier la séparation post-correction :

| Type d'email | Vague | Stack | Justification |
|---|---|---|---|
| **Transactionnel** (welcome, Stripe receipts, trial-end notice, password reset, GDPR export delivery) | **V1 (infrastructure obligatoire)** | Resend ou Postmark transactional (free tier suffit pour V1) | Bloquant DG-043 Stripe live + DG-038 DSAR + cycle signup. Pas un canal de distribution — c'est plumbing du funnel. |
| **Digest quotidien** (lecture du matin, top insights de la semaine, newsletter) | **V2 (rétention)** | Postmark broadcasts ou ConvertKit, tier paid | Canal de rétention conditionné à MAU > 100. Activer après l'audience. |
| **Alertes event-critique** (FOMC ≤ 4h, NFP imminent) | **V2 ou jamais** | Resend + queue | Préférer push web (PWA) + Telegram pour latence. Email = trop lent (15-60 min en pratique). |

Le service `EmailService` est instancié en V1 avec uniquement le profil transactionnel. Le digest est un nouveau `EmailDigestService` ajouté en V2.

---

# Synthèse partielle (parties 1-2-3) pour validation

## Verdict de santé du repo

> **60 % production-ready, 25 % WIP utile, 15 % legacy mort.** Le noyau (intelligence + api + delivery + webapp) est solide. Quatre chantiers urgents : wirer le scoring LGBM (audit 2026-05-27), archiver le legacy RL (`parallel_training.py` + `src/training/` + `agents/*_integration.py`), unifier le contrat `InsightSignalV2` sur `intelligence/insight_v2/`, et introduire les ports/adapters (`IDeliveryChannel`, `IDataProvider`, `INarrativeEngine`, `ISignalStore`) pour pouvoir multiplier les canaux sans toucher au cœur.

## Architecture cible (résumé)

> Un **noyau read-only** (DataProvider → SmartMoney → Vol → RegimeGate → Confluence → CalibratedConviction → News → StateMachine → InsightAssembler) produit un `InsightSignalV2` canonique. Au-dessus, **8 services** (Insight, Chat, Delivery, Billing, Account, TrackRecord, Compliance, Analytics) orchestrent les cas d'usage. La couche API FastAPI est fine (routes < 30 lignes). La couche de **delivery est un port** (`IDeliveryChannel`) avec N adapters. Le frontend webapp est une des N surfaces, pas une exception architecturale. Aucun renderer ne sait ce qu'est un BOS ou un FVG — il projette.

## Classification V1/V2/V3/V4/NEVER des 21 canaux (post-corrections 2026-05-27)

| Vague | Canaux retenus |
|---|---|
| **V1 (0-3 mois)** | Webapp SaaS (PWA inclus) · Bot Telegram · TradingView Pine showcase |
| **V2 (3-9 mois)** | Email digest · TradingView webhook receiver · Bot Discord · Mobile cross-platform · Push web |
| **V3 (9-18 mois)** | API REST B2B · Webhooks B2B · Extension navigateur · Widget embeddable · SMS premium · Plugin WordPress · ChatGPT/Claude tool · MT4/5 (cond.) |
| **V4 (18+ mois)** | Desktop Electron · NinjaTrader/cTrader · Alexa/Google · Audio podcast |
| **NEVER** | Apple Watch / wearables |

**Note infra V1** : email transactionnel (Welcome/Stripe/Trial-end) reste service obligatoire V1, distinct du digest qui passe en V2.

---

✅ **Parties 1-2-3 validées par l'utilisateur** avec les 3 corrections (TradingView V1, Email V2, limitations PWA explicites) intégrées ci-dessus.

---

# Partie 4 — Focus sur les 3 canaux V1

Architecture détaillée par canal : flux end-to-end, endpoints API consommés, format payload, cache, erreurs, métriques, effort.

## 4.1 Canal V1.A — Webapp SaaS (Next.js + PWA)

### Flux end-to-end

```
Utilisateur (mobile/desktop)
   │
   │ HTTPS via mia.markets
   ▼
Cloudflare (CDN + DDoS)
   │
   ▼
Vercel (Next.js 15 — SSG landing + ISR pages + SSR app)
   │
   │ /[locale]/page.tsx     → SSG (landing)
   │ /[locale]/insight/...  → ISR 60s revalidate
   │ /[locale]/chat         → SSR + Edge runtime
   │ /api/chat              → Edge runtime, SSE proxy
   │
   ▼
Backend API (Fly.io Paris — FastAPI)
   │
   │ /api/v1/insights/current/XAUUSD
   │ /api/v1/insights/{id}
   │ /api/v1/insights/{id}/breakdown
   │ /api/v1/chat/ask          (SSE)
   │ /api/v1/track-record/public
   │ /api/v1/subscriptions/checkout
   │
   ▼
InsightService → SignalStore (SQLite) → SmartMoneyEngine → ...
ChatService → LLMNarrativeEngine → Anthropic Claude API
```

### Endpoints API consommés

| Endpoint | Fréquence appel | Cache stratégie | Tier requis |
|---|---|---|---|
| `GET /api/v1/insights/current/{symbol}` | 1×/30s polling OU SSE long-poll | ISR Vercel 30s + browser SWR | FREE+ |
| `GET /api/v1/insights/{id}/breakdown` | 1×/clic section EXPERT | ISR 5 min (insight immuable post-publication) | PRO+ |
| `POST /api/v1/chat/ask` | 1×/question utilisateur (SSE stream) | aucun (live LLM) | FREE+ (cap quotas par tier) |
| `GET /api/v1/chat/suggestions/{signal_id}` | 1×/changement insight | Edge cache 5 min | FREE+ |
| `GET /api/v1/track-record/public` | 1×/load landing + 1×/h refresh | ISR Vercel 1 h | public (FREE) |
| `POST /api/v1/subscriptions/checkout` | 1×/click upgrade | aucun | FREE → STARTER+ |
| `GET /api/v1/account/me` | 1×/login + 1×/h | Browser cache 60s | authentifié |

### Payload échangés (référence courte — détail en `CONTRACTS/`)

- **`InsightSignalV2`** (canonique riche, ~12 sous-modèles) pour `/insights/{id}`.
- **`FocusCard`** (verdict + PF + event banner, ~200 chars utiles) pour hero.
- **`CopilotCard`** (6 sections collapsibles) pour body principal.
- **`ExpertFull`** (waterfall 8 composantes + conformal viz + sources) pour mode dépliable.
- **`ChatSSEFrame`** (chunks SSE Claude streaming) pour le chatbot.

### Fréquence de mise à jour

- **Insights** : nouvelle barre M15 → nouveau signal publié si `SignalStateMachine` transition (~3-6 signaux/jour XAU).
- **Track record public** : recalculé J+1 (job nightly).
- **Chat** : temps réel, aucune mise à jour planifiée.

### Stratégie de cache

| Niveau | Outil | TTL | Invalidation |
|---|---|---|---|
| CDN edge | Cloudflare | 60 s landing + 0 s `/api/*` | par tag `signal:{id}` |
| Vercel ISR | Next.js | 30 s `/insight/current`, 5 min `/insight/{id}`, 1 h `/track-record` | on-demand (publication) |
| Browser SWR | `swr` lib | 30 s + revalidateOnFocus | auto |
| Backend service | in-memory Python dict V1, Redis V2 | 60 s `current_insight`, 5 min `breakdown` | TTL + invalidation publication |

**Principe** : cache lourd côté edge (Cloudflare + ISR), backend ne sert que sur miss + writes. Coût bande passante minimisé.

### Gestion des erreurs et UX dégradée

| Erreur backend | Comportement UX |
|---|---|
| API down (5xx) | Affichage dernière `InsightSignalV2` en localStorage (max 4 h) + bannière "données en cache · service en cours de récupération". |
| Chat LLM down (CircuitBreaker open) | Fallback `TemplateNarrativeEngine` côté backend (déjà câblé). Chatbot répond template-based, banner "mode dégradé · réponses simplifiées". |
| Rate limit dépassé (429) | Modal "limite quota atteinte · upgrade" avec lien pricing. |
| Geo-block (451) | Page locale dédiée "M.I.A. Markets n'est pas disponible dans votre région" + lien contact si erreur géolocalisation. |
| Auth expired (401) | Refresh JWT silencieux, sinon redirection `/login` avec deep-link return. |
| Offline (PWA SW) | Service worker affiche dernier insight + chat scripted (5 réponses pré-écrites) + bannière offline. |

### Métriques clés à instrumenter (Plausible events DG-161)

| Event | Properties | Pourquoi mesurer |
|---|---|---|
| `signal_view` | `signal_id`, `tier`, `symbol`, `device` | Engagement de base |
| `section_expanded` | `section_name` (regime / structure / breakdown), `signal_id` | Profondeur d'usage |
| `chatbot_question` | `category` (why_score / explain_term / refusal), `signal_id` | Moat chatbot fonctionne ? |
| `upgrade_clicked` | `from_tier`, `to_tier`, `cta_location` | Funnel |
| `signup` | `source` (utm), `language` | Acquisition source |
| `paid_conversion` | `tier`, `trial_path` (dual_14_14), `amount` | Conversion |
| `pwa_installed` | `platform` (ios/android/desktop) | Adoption PWA |
| `push_opt_in` | `platform`, `tier` | Pré-requis V2 mobile natif |

### Effort dév estimé

| Item | Heures | Status actuel |
|---|---|---|
| Landing + hero card (DG-120) | 8-12 | ✅ V2.4 livré, polish à faire |
| Architecture progressive uniforme (DG-101) | 16-24 | ✅ V2.4 livré |
| Mobile-first responsive (DG-103) | 16 | ✅ V2.4 livré |
| Chatbot wiring 8 composantes (DG-110) | 20-30 | 🟡 V2.1 livré scaffolding, manque context-injection |
| 3 questions suggérées (DG-114-REDUCED) | 6-8 | 🟡 à implémenter |
| Tests adversariaux refus pédagogique (DG-112) | 6-10 | 🟡 à implémenter |
| Page pricing avec decoy + dual trial (DG-132) | 10-16 | 🟡 à implémenter |
| Track-record public mensuel (DG-142) | 14-20 | 🟡 à implémenter |
| Plausible self-hosted + 6 events (DG-160+161) | 16-20 | 🟡 à implémenter |
| Replacement mocks → client API typé (zod) | 20-30 | 🔴 à faire avant V1 lancement |
| Service Worker push notifications (Web Push) | 8-12 | 🔴 à faire pour event-alert PWA |
| **TOTAL Webapp V1 complet** | **140-198 h** | ~30 % déjà livré |

## 4.2 Canal V1.B — Bot Telegram

### Flux end-to-end

```
SignalStateMachine publie un nouvel InsightSignalV2
   │
   ▼
DeliveryService.publish(insight, channels=["telegram"])
   │
   ▼
TelegramAdapter.render(insight, locale)
   │  → to_telegram_b2c() : markdown ≤ 800 chars
   │  → langue détectée depuis TelegramLangStore (sprint W3 déjà livré)
   ▼
Idempotency check (signal_id + chat_id déjà envoyé ?)
   │
   ▼
NotificationQueue → CircuitBreaker (threshold=5, timeout=120s)
   │
   ▼
Telegram Bot API (sendMessage)
   │
   ▼
Utilisateur reçoit dans son client Telegram
   + bouton inline "💬 Demander à Sentinel" → deep-link webapp /chat?signal_id=...
   + bouton inline "📊 Voir tout" → deep-link webapp /insight/{id}
```

### Endpoints/API consommés

- **Telegram Bot API** : `sendMessage`, `editMessageReplyMarkup`, `setMyCommands`.
- Pas d'appel REST MIA sortant (le flux est push-only).
- Inbound bot (conversational) consomme `/api/v1/chat/ask` côté MIA quand l'utilisateur tape dans le bot.

### Format du payload (rendu Telegram)

```
🟢 Lecture haussière XAUUSD · M15

Conviction : STRONG (72/100)
Structure : BOS 2391.5 · FVG 2378-2381 · retest armé
Invalidation : 2378.0
Régime : trend bullish · gate TRADE
Volatilité : normale · forecast +10% vs naïf
⚠ FOMC Minutes dans 18h

« Cassure haussière confirmée par retest FVG.
Lecture algorithmique éducative. Ne constitue ni un
signal ni un conseil en investissement. »

📊 Voir détails    💬 Demander à Sentinel
```

(Schéma détaillé dans `CONTRACTS/telegram_render.md`)

### Fréquence de mise à jour

- Push à chaque transition `SignalStateMachine` (~3-6/jour XAU M15).
- Pas d'edits intermédiaires (un signal = un message ; cycle de vie suit `valid_until_utc`).
- Cooldown 60 min minimum entre 2 push même symbol/direction.

### Stratégie de cache

- Pas de cache outbound (push, pas pull).
- Inbound chatbot Telegram → utilise le même `/api/v1/chat/ask` que la webapp → bénéficie du `SemanticCache` LLM (60-90s TTL).

### Gestion des erreurs

| Erreur | Comportement |
|---|---|
| 429 Telegram rate limit | Retry exponentiel jusqu'à 5 fois, ensuite DLQ + alerte Sentry. |
| 403 user a bloqué le bot | Marquer `chat_id` inactif, ne plus retenter. |
| 5xx Telegram down | CircuitBreaker open après 5 erreurs consécutives, fallback désactivation 120s, retry après. |
| Message > 4096 chars (limite Telegram) | Truncate avec `…` final + lien "voir tout" webapp. |
| LLM down sur question conversationnelle | Réponse fallback template "Je rencontre un souci, peux-tu reformuler ?" |

### Métriques

| Event | Properties |
|---|---|
| `telegram_push_sent` | `signal_id`, `chat_id` (hash), `lang` |
| `telegram_push_delivered` | idem + delivery_latency_ms |
| `telegram_push_failed` | idem + error_code |
| `telegram_inline_click` | `button` (chat/voir-tout), `signal_id` |
| `telegram_chat_question` | (déjà géré par chatbot_question commun) |

### Effort dév estimé

| Item | Heures | Status actuel |
|---|---|---|
| Renderer `to_telegram_b2c()` mise à jour C1+lien | 8 | 🟡 existe, polish C1 |
| Buttons inline + deep-link webapp | 6-8 | 🔴 à ajouter |
| Idempotency store partagé webhook B2B | 4-6 | 🟡 existe, à wirer |
| TelegramLangStore (FR/EN J1) | 4 | ✅ livré sprint W3 |
| Inbound conversational bot (forward vers `/api/v1/chat/ask`) | 12-16 | 🔴 à implémenter |
| Tests E2E push + idempotency | 6-8 | 🔴 à compléter |
| **TOTAL Telegram V1 complet** | **40-50 h** | ~40 % déjà livré |

## 4.3 Canal V1.C — TradingView Pine showcase

### Flux end-to-end

```
Utilisateur TradingView (mia.markets pas requis pour voir l'indicateur)
   │
   │ Charge le script "M.I.A. Markets — SMC Sentinel" depuis TradingView (publié public/gratuit)
   ▼
Pine Script v5 exécuté côté TradingView, calculs locaux uniquement
   │
   │ Détecte localement :
   │   • BOS / CHOCH (Williams fractal 2-bar)
   │   • FVG (3-bar gap) + size_atr
   │   • OB (engulfing + last opposite candle)
   │   • Retest state (idle/awaiting/armed/consumed)
   │   • Score simplifié 5 facteurs (BOS, FVG, OB, retest, ATR-vol)
   │
   ▼
Affichage chart natif TradingView :
   • Lignes BOS niveau (label "BOS 2391.5")
   • Boîtes FVG (couleur direction)
   • Boîtes OB
   • Annotation "Lecture haussière · Score 65/100"
   • Annotation "Invalidation 2378"
   • Disclaimer ligne bottom : "Lecture éducative · version complète + chatbot sur mia.markets"
   │
   ▼
CTA utilisateur sur TradingView :
   • Lien dans description du script → mia.markets/tv?utm_source=tradingview
   • Profil TradingView "MIA Markets" avec bio + lien
   │
   ▼
Acquisition gratuite vers webapp (mesurable Plausible utm)
```

### Endpoints/API consommés

**Aucun.** C'est tout l'argument du showcase pur. Le script tourne uniquement en sandbox Pine, ne fait pas de `request.security()` externe, ne dépend d'aucune infra côté MIA.

### Format du "payload" (rendu chart)

Pas de payload réseau. Le rendu est constitué de :
- `line.new()` pour les niveaux BOS/invalidation
- `box.new()` pour les zones FVG / OB
- `label.new()` pour les annotations score + direction
- Un panel d'info en bas-droite avec : score 5 facteurs, direction, prochain event high-impact (depuis `request.economic_calendar(...)` Pine natif si disponible — sinon panel events manuel via input).

### Fréquence de mise à jour

- À chaque tick (Pine natif). Pas d'effort côté MIA.

### Stratégie de cache

- N/A (zero infra MIA).
- L'utilisateur paie le compute TradingView (côté son navigateur + serveurs TV).

### Gestion des erreurs

| Type | Comportement |
|---|---|
| Données manquantes sur instrument (e.g. crypto exotique non-XAU/EUR) | Annotation "Score indicatif · indicateur calibré XAU/EUR" |
| Pine script timeout | Géré par TradingView (timeout natif), pas d'action requise |
| Calendrier économique inaccessible (Pine `request.economic_calendar`) | Panel event affiche "Calendrier non disponible · voir mia.markets" |

### Métriques

**Côté TradingView** (dans TV creator dashboard, hors Plausible) :
- Views du script
- Favorites
- Likes / ratings
- Comments

**Côté MIA** (Plausible) :
- `visits?utm_source=tradingview` → entrant TV
- `signup?utm_source=tradingview` → conversion
- Funnel TV → signup → trial → paid (segmenté par source)

### Effort dév estimé

| Item | Heures |
|---|---|
| Pine v5 — détection BOS/CHOCH locale (Williams fractal) | 4-6 |
| Pine v5 — détection FVG + size_atr | 2-3 |
| Pine v5 — détection OB (engulfing) | 2-3 |
| Pine v5 — retest state machine (4 états) | 3-4 |
| Pine v5 — score 5 facteurs + annotations chart | 3-4 |
| Pine v5 — panel info bottom-right + disclaimer | 2-3 |
| Profil TradingView MIA Markets (bio + screenshots + lien) | 2-3 |
| Description du script (FR + EN, compliance review) | 2 |
| Compliance check (avocat ou Iubenda) wording Pine + description | 2 |
| Page d'atterrissage webapp `/tv` dédiée (UTM funnel) | 4-6 |
| **TOTAL TradingView V1 complet** | **26-36 h** |

## 4.4 Verdict croisé V1 — TradingView reconfirmé en V1

| Canal V1 | Effort total | Acquisition / Rétention | Coût récurrent |
|---|---|---|---|
| Webapp + PWA | 140-198 h | Hub central — conversion + rétention | Vercel + Cloudflare $0-$240/an |
| Bot Telegram | 40-50 h | Wedge FR-first, push rétention | $0 |
| **TradingView Pine showcase** | **26-36 h** | **Acquisition gratuite via hub cible** | **$0** |

Le TradingView Pine showcase est **le canal V1 avec le meilleur ratio acquisition/effort** : 26-36h pour ouvrir un canal d'acquisition gratuit dans le hub où vit la cible. À comparer avec n'importe quel autre canal V2/V3 qui demande effort + investissement marketing pour générer du trafic.

---

# Partie 5 — Refactor recommandé du code existant

Objectif : passer de l'état actuel à l'architecture cible **sans casser le système qui tourne** (1366+ tests verts).

## 5.1 Renommages de modules / dossiers

| Avant | Après | Pourquoi |
|---|---|---|
| `src/environment/strategy_features.py` (SmartMoneyEngine) | `src/intelligence/smart_money/engine.py` | Promouvoir au premier rang. Sortir de `environment/` (historiquement RL). |
| `src/api/insight_signal_v2.py` | (supprimer) ; importer depuis `src/intelligence/insight_v2/` | Source unique du contrat. |
| `src/intelligence/main.py` | `src/intelligence/scanner_runtime.py` | Plus clair : c'est le runtime du scanner, pas "main" de l'app. |
| `src/intelligence/scoring/` | `src/intelligence/calibration/` | Le dossier décrit le **pipeline de calibration**, pas juste le scoring. |
| `src/api/app.py` | `src/api/bootstrap.py` | Décrit ce qu'il fait (bootstrap FastAPI). |
| `src/delivery/` | `src/delivery/adapters/` (sous-dossier) + `src/delivery/__init__.py` exposant `IDeliveryChannel` | Clarté pattern ports/adapters. |
| `src/intelligence/rag/` | `src/services/rag/` | Le RAG est un service, pas une part du pipeline algo. |
| Renderers `to_*` éparpillés | `src/delivery/renderers/{focus,copilot,expert,telegram,discord,b2b,tradingview}.py` | Concentrer la couche projection. |

## 5.2 Modules à découper

| Module actuel | LOC | Découper en |
|---|---|---|
| `src/environment/environment.py` | 2 423 | `src/intelligence/smart_money/features.py` (KEEP, utile) + `_legacy/rl_env.py` (ARCHIVE) + `_legacy/reward_shaper.py` (ARCHIVE) |
| `src/intelligence/volatility_forecaster.py` | 1 561 | `volatility/har.py` + `volatility/hmm_mult.py` + `volatility/calendar_mult.py` + `volatility/diurnal_mult.py` + `volatility/blender.py` + `volatility/tcp_intervals.py` |
| `src/agents/kill_switch.py` | 1 861 | `risk/kill/streak_loss.py` + `risk/kill/drawdown.py` + `risk/kill/vol_spike.py` + `risk/kill/heartbeat.py` + `risk/kill/orchestrator.py` |
| `src/agents/orchestrator.py` | 1 339 | `services/orchestration/coordinator.py` + `services/orchestration/audit.py` |
| `src/api/models.py` | ~800 | `api/schemas/{auth,billing,insight,chat,account,track_record,webhooks}.py` |
| `src/intelligence/sentinel_scanner.py` | 1 274 | `services/scanner/loop.py` + `services/scanner/instruments.py` + `services/scanner/state.py` |

## 5.3 Modules à fusionner

| Modules à fusionner | Résultat |
|---|---|
| 5+ "RiskManager" éparpillés (`agents/risk_sentinel.py`, `agents/intelligent_risk_sentinel.py`, `agents/portfolio_risk.py`, `agents/ensemble_risk_model.py`, `environment/risk_manager.py`) | **Un seul `services/risk/risk_service.py`** qui expose `risk_score(insight) → 0-100` + `kill_decision(...) → TRADE/REDUCE/BLOCK`. Tous les autres deviennent **internes** (stratégies internes votées dans le service) ou **archivés**. (DG-039 MODIFY) |
| `regime_filter.py` + `regime_gate.py` | **Un seul `intelligence/regime/gate.py`** qui combine HMM + BOCPD + jump_ratio. |
| `intelligence/semantic_cache.py` + `intelligence/rag/cache.py` | À auditer : si responsabilités distinctes, renommer (`llm_response_cache.py` vs `rag_embedding_cache.py`) ; si chevauchement, fusionner. |
| `api/models.py` (InsightSignalV2 partial) + `api/insight_signal_v2.py` + `intelligence/insight_v2/` | **Un seul `intelligence/insight_v2/`** ; api importe. |

## 5.4 Modules morts à supprimer (avec archive backup)

**Procédure** : créer un dossier `_archive/2026-05-XX_pivot_b/` avec git mv pour préserver l'historique. Pas de `rm -rf`.

| Module | LOC | Action |
|---|---|---|
| `parallel_training.py` (racine) | 1 634 | `git mv` → `_archive/2026-05-XX_pivot_b/parallel_training.py` |
| `src/training/` (9 fichiers) | ~7 500 | `git mv` → `_archive/2026-05-XX_pivot_b/training/` |
| `src/agents/integration.py` | ~400 | idem |
| `src/agents/intelligent_integration.py` | ~500 | idem |
| `src/agents/orchestrated_integration.py` | ~600 | idem |
| `src/agents/risk_integration.py` | ~500 | idem |
| `src/agents/regime_predictor.py` | ~300 | idem (orphelin) |
| `src/agents/sprint2_intelligence.py` | ~50 | idem (stub) |
| `src/agents/market_regime_agent.py` | — | à vérifier (peut-être encore appelé par RL legacy uniquement) |
| `src/intelligence/bocpd.py` | 80 | à vérifier appelants ; si orphelin → archive |
| `src/intelligence/regime_gate.py` OU `regime_filter.py` | 120 / 180 | après merge, l'un des deux archive |
| `src/intelligence/volatility_lgbm.py` | 180 | si VOL_MODE=har figé → archive |
| `tests/test_long_short_trading.py` (déjà cassé) | — | `git rm` (DG-002) |
| `parallel_training.py` config legacy : `Procfile`, `railway.toml` | — | `git rm` (DG-012) |
| 15+ .md de pollution racine (`BUSINESS_PLAN_*`, `COMMERCIALIZATION_REPORT*`, `INSTITUTIONAL_AUDIT*`, `SPRINT_*`, `AGENTS_SYSTEM_ANALYSIS`, etc.) | — | `git mv` → `docs/archive/` |

**Total LOC archivé** : ~12 000 lignes Python + 15+ markdown racine.

## 5.5 Interfaces à introduire (ports/adapters)

Créer le package `src/interfaces/` contenant les ports :

```python
# src/interfaces/delivery.py
class IDeliveryChannel(Protocol):
    name: str
    supports_languages: list[str]
    async def render(self, insight: InsightSignalV2, locale: str) -> Payload: ...
    async def publish(self, payload: Payload, target: ChannelTarget) -> DeliveryResult: ...

# src/interfaces/data.py
class IDataProvider(Protocol):
    async def get_ohlcv(self, symbol: str, tf: str, lookback: int) -> DataFrame: ...
    async def get_latest_bar(self, symbol: str, tf: str) -> Bar: ...

# src/interfaces/narrative.py
class INarrativeEngine(Protocol):
    async def generate_short(self, insight: InsightSignalV2, lang: str) -> str: ...
    async def generate_long(self, insight: InsightSignalV2, lang: str) -> str: ...

# src/interfaces/signal_store.py
class ISignalStore(Protocol):
    async def save(self, insight: InsightSignalV2) -> None: ...
    async def get(self, signal_id: str) -> Optional[InsightSignalV2]: ...
    async def query(self, filters: QueryFilter) -> list[InsightSignalV2]: ...

# src/interfaces/llm.py
class ILLMRouter(Protocol):
    async def complete(self, prompt: Prompt, tier: SignalTier, lang: str) -> str: ...
    async def stream(self, prompt: Prompt, tier: SignalTier, lang: str) -> AsyncIterator[str]: ...
```

Ces interfaces deviennent les **points de découplage**. Les adapters concrets vivent dans `src/delivery/adapters/`, `src/intelligence/data_providers/`, etc. Les services dépendent des interfaces, jamais des concrétions.

## 5.6 Ordre d'exécution du refactor (sans casser le système)

**Règle** : chaque étape commit + tests verts avant la suivante. Pas de big-bang.

| Ordre | Étape | Heures | Risque casse | Mitigation |
|---|---|---|---|---|
| 1 | **Archive legacy RL** (`git mv` parallel_training, training/, integration*.py) | 2-3 | 🟢 nul | Code mort = pas de test à mettre à jour. |
| 2 | **Archive .md racine** vers `docs/archive/` | 1 | 🟢 nul | Documentation uniquement. |
| 3 | **Merger regime_filter + regime_gate** | 4-6 | 🟡 moyen | Renommer prudemment, lancer tests régression. |
| 4 | **Unifier `InsightSignalV2`** : `intelligence/insight_v2/` source unique, `api/models.py` importe | 6-10 | 🔴 fort | Faire en 1 PR, exécuter pleine suite tests. |
| 5 | **Promouvoir SmartMoneyEngine** (`environment/strategy_features.py` → `intelligence/smart_money/engine.py`) | 3-4 | 🟡 moyen | Renommage simple, search-replace imports. |
| 6 | **Introduire les ports** (`src/interfaces/`) | 4-6 | 🟢 nul | Création pure, pas de casse. |
| 7 | **Refactor delivery** : créer adapters concrets, `IDeliveryChannel` injecté dans main | 12-16 | 🟡 moyen | Adapter par adapter, tests dédiés. |
| 8 | **Splitter `environment/environment.py`** : extraire features actives, archiver RL | 6-10 | 🟡 moyen | Tests features couvrent déjà |
| 9 | **Splitter `volatility_forecaster.py`** en sous-modules | 8-12 | 🟡 moyen | Tests vol forecaster doivent rester verts |
| 10 | **Consolider RiskManagers** vers `services/risk/risk_service.py` (DG-039) | 12-18 | 🔴 fort | Identifier appelants, refactor incrémental |
| 11 | **Splitter `api/models.py`** en `api/schemas/{...}.py` | 4-6 | 🟡 moyen | Re-export depuis `api/schemas/__init__.py` pour compat |
| 12 | **Wire LGBM `scoring_v3_lgbm.pkl`** dans `CalibratedConvictionPipeline` (DG-025) | 20-30 | 🔴 fort | Chemin critique — A/B tests, validation Brier OOS |

**Total estimé refactor critique** : **82-122 h** (≈ 10-15 semaines à 8-9h/sem en parallèle d'autres travaux).

## 5.7 Effort estimé global du refactor

| Lot | Heures | Priorité |
|---|---|---|
| **Lot 1 — Cleanup** (étapes 1-2) | 3-4 | 🔴 P0 immédiat (5 min de valeur déclarée + 5 min de tâche) |
| **Lot 2 — Unification contrats + merges** (étapes 3-5) | 13-20 | 🔴 P0 — bloque l'ajout de nouveaux canaux |
| **Lot 3 — Ports/adapters** (étapes 6-7) | 16-22 | 🔴 P0 — pré-requis V1 TradingView/Telegram propre |
| **Lot 4 — Splits god modules** (étapes 8-9, 11) | 18-28 | 🟡 P1 — réduit dette technique |
| **Lot 5 — Risk consolidation** (étape 10) | 12-18 | 🟡 P1 — DG-039 MODIFY |
| **Lot 6 — Wire LGBM** (étape 12) | 20-30 | 🔴 P0 chemin critique algo (audit 2026-05-27 #1) |
| **TOTAL** | **82-122 h** | dont ~60 h P0 |

---

# Partie 6 — Stack technique cible

Pour chaque couche : techno + justification courte + ce qu'elle remplace + coût mensuel.

| Couche | Techno V1 | Justification | Remplace | Coût/mo V1 |
|---|---|---|---|---|
| **Backend pipeline algo** | Python 3.11 + Pydantic v2 + NumPy + pandas + Numba | Déjà en place, écosystème quant solide, Numba pour SMC hot-path | — | $0 |
| **Backend API** | FastAPI + Uvicorn + Gunicorn (1 worker V1) | Async natif, OpenAPI auto, Pydantic v2 first-class, communauté quant | — | $0 |
| **Base de données primaire** | SQLite + WAL mode | Suffit MAU < 500, zero ops, embarquée Fly volume | — | $0 |
| **Migration DB** | Postgres 16 (Fly.io managed) | Quand MRR > $5k OU paid subs > 100 (DG-026 DEFER) | SQLite | $20-50/mo V2+ |
| **Cache** | In-memory dict + functools.lru_cache | Suffit V1 mono-worker | — | $0 |
| **Migration cache** | Redis (Upstash serverless) | DG-020 DEFER MAU > 200 | in-memory | $10-25/mo V2+ |
| **File d'attente** | Asyncio queues V1 (in-process) ; RQ V2 | Pas de Celery (over-engineered solo). RQ = simple, Python-natif, sur Redis | — | inclus dans Redis V2 |
| **Front-end web** | Next.js 15 + React 18 + Tailwind + shadcn/ui | Locked DG-023, déjà livré V2.4 | — | $0 (Vercel free) |
| **Mobile (V2)** | **React Native + Expo (managed workflow)** | (1) Partage stack JS avec webapp Next.js, (2) bibliothèque OpenAPI client commune via `openapi-typescript-codegen`, (3) communauté + recrutement freelance abondant si besoin, (4) Expo simplifie release iOS+Android pour un solo dev sans Xcode local complexe, (5) Hot reload sympa, (6) Fastlane intégré. **Flutter écarté** : écosystème Dart isolé, pas de réutilisation avec webapp. **Natif Swift/Kotlin écarté** : 2× le code à maintenir. | PWA pour le subset push iOS / App Store crédibilité | $99/an Apple + $25 one-time Google + $0 Expo free |
| **TradingView V1** | Pine Script v5 + publication TradingView | Natif TV, zero infra | — | $0 |
| **TradingView V2 (webhook)** | FastAPI endpoint `/api/v1/tv-webhook` + HMAC verify | Normalise alerts TV vers `InsightSignalV2` | — | $0 (compute marginal) |
| **Hébergement backend** | Fly.io région Paris (CDG) | DG-022, latence FR<50ms, secrets natifs, scale-to-zero possible | — | $5-15/mo V1 |
| **Hébergement front** | Vercel free tier | Next.js first-class, ISR + Edge runtime | — | $0 V1 |
| **CDN + DNS** | Cloudflare free tier | DDoS, cache edge, DNS rapide | — | $0 |
| **Email transactionnel** | Resend (free tier 100/jour) | Simple API, deliverability bonne | — | $0 V1, $20/mo V2 (digest) |
| **SMS (V3)** | Twilio ou Vonage pay-as-you-go | Standard | — | $0,03-0,06/SMS |
| **Observability — Erreurs** | Sentry free tier (5k events/mo) | DG-033-MODIFIED KEEP V1 | — | $0 V1 |
| **Observability — Traces** | OpenTelemetry + Tempo/Grafana | DG-033 DEFER team > 1 | — | $0 V3+ |
| **Observability — Metrics** | Prometheus + Grafana (Fly.io managed) | Déjà câblé infrastructure/ | — | inclus Fly.io |
| **Analytics produit** | Plausible self-hosted (DG-160 P0-strict V1) | Privacy-first, CNIL-friendly, pas de cookie banner | Google Analytics | $5-10/mo (1 Fly machine partagée) |
| **Auth** | **Clerk** | (1) UI prête (sign-in, social, MFA), (2) JWT bearer standard, (3) free tier généreux (10k MAU), (4) webhooks pour sync DB locale, (5) Stripe customer link natif. **Supabase Auth écarté** : couple à toute la stack Supabase qu'on ne veut pas. **NextAuth (Auth.js)** : self-hosted gratuit mais demande maintenance UI flow + recovery + MFA = trop de plomberie pour un solo. | maison | $0 V1 (< 10k MAU), $25/mo après |
| **Billing** | Stripe live (DG-043) + Customer Portal + Stripe Tax UE (DG-044) | Locked, standard SaaS, gestion VAT EU auto | — | 2,9% + $0,30/tx + 0,5% Tax addon |
| **LLM** | Anthropic Claude (Haiku/Sonnet/Opus) | Locked, qualité narratif, cache prompt natif | — | variable, monitoring DG-052 obligatoire ($10-50/mo cible V1) |
| **Vector store (RAG V1)** | Chroma local (file-based) avec 12 papers curés | DG-058a, suffit V1 | — | $0 |
| **Vector store (V2+)** | Qdrant Cloud (free tier 1GB) | DG-058b DEFER MAU > 500 | Chroma | $0 free tier puis $25/mo |
| **Secrets** | Fly.io secrets natifs | DG-029-MODIFIED, audit basique suffit V1 | maison .env | $0 |
| **Secrets (V3+)** | Doppler ou Vault | DEFER MRR > $5k OU team > 1 | — | $20/mo V3+ |
| **Backup DB** | LiteFS sync + S3 (Cloudflare R2 $0,015/GB) | Snapshots quotidiens SQLite | — | $1-2/mo |
| **i18n** | next-intl + JSON messages/ + lookup côté backend | FR+EN J1 (locks 2026-05-26) | — | $0 |
| **CI/CD** | GitHub Actions (free tier public repos) | Pipelines tests + Docker build + deploy Fly+Vercel | — | $0 V1 |
| **Lint/Format** | ruff + mypy + prettier + biome | Standard moderne | — | $0 |
| **Status page** | Statuspal ou cstate self-hosted | V2 trust badge | — | $0 V1, $20/mo V2+ |

### Coût mensuel total V1 estimé

| Poste | Coût V1 |
|---|---|
| Hébergement Fly.io | $10/mo |
| Vercel + Cloudflare | $0 |
| Plausible self-hosted | $7/mo (Fly machine) |
| Sentry | $0 (free tier) |
| Resend | $0 (free tier) |
| Backup R2 | $2/mo |
| Anthropic LLM | ~$30/mo (cible) |
| Stripe | variable (% revenue) |
| Domaine `mia.markets` | $1-3/mo |
| **TOTAL V1 fixe** | **~$50/mo** |

### Coût mensuel projeté V2 (MAU 200-500)

| Poste additionnel | Coût |
|---|---|
| Postgres managed | +$25/mo |
| Redis Upstash | +$15/mo |
| Resend Pro (digest) | +$20/mo |
| Sentry Team | +$26/mo |
| Apple Developer | +$8/mo |
| Anthropic LLM scaled | +$50-150/mo |
| **TOTAL V2 fixe** | **~$200-300/mo** |

### Coût mensuel projeté V3 (MAU > 1k, B2B)

| Poste additionnel | Coût |
|---|---|
| Qdrant Cloud | +$25/mo |
| Doppler secrets | +$20/mo |
| Trading Economics API | +$79/mo (DG-027) |
| Twilio SMS | variable |
| **TOTAL V3 fixe** | **~$400-600/mo** |

---

# Partie 7 — Plan de migration vers l'architecture cible

Séquence ordonnée en 6 phases. Chaque phase a un livrable, une durée, un gate de sortie.

## Phase A — Cleanup (sans rien ajouter)

**Objectif** : enlever le bruit avant de construire.

| Livrable | Durée | Détails |
|---|---|---|
| Archive legacy RL (`parallel_training.py`, `src/training/`, agents `*_integration.py`, etc.) vers `_archive/2026-05-XX_pivot_b/` | 2-3 h | git mv préserve historique |
| Archive 15+ .md de pollution racine vers `docs/archive/` | 1 h | |
| `git rm` `tests/test_long_short_trading.py` (DG-002), `Procfile`, `railway.toml` (DG-012) | 0,5 h | |
| Réorganisation `src/intelligence/` : promotion `SmartMoneyEngine` depuis `environment/strategy_features.py` | 3-4 h | |
| Merge `regime_filter.py` + `regime_gate.py` → `intelligence/regime/gate.py` | 4-6 h | |
| Audit `bocpd.py`, `volatility_lgbm.py`, `semantic_cache.py vs rag/cache.py` ; archive si orphelins | 2-3 h | |

**Durée totale Phase A** : 12-17 h (≈ 1,5-2 sem solo).
**Gate de sortie** : `git log` montre les archives, suite de tests verte (1366+ tests, hors test_long_short_trading), CI build OK.

## Phase B — Découplage pipeline / delivery (introduction d'interfaces)

**Objectif** : pouvoir ajouter N canaux sans toucher au noyau.

| Livrable | Durée | Détails |
|---|---|---|
| Création `src/interfaces/{delivery,data,narrative,signal_store,llm}.py` | 4-6 h | Protocols Pydantic-like |
| Refactor `delivery/` : sous-dossier `adapters/` + `renderers/` + factory | 8-12 h | Pas de breaking change runtime |
| Injection des adapters dans `scanner_runtime.py` (ex-`main.py`) via factory | 4-6 h | Replace direct imports |
| Unification `InsightSignalV2` : suppression `api/insight_signal_v2.py`, importation depuis `intelligence/insight_v2/` | 6-10 h | Source unique, refactor critique |
| Splitter `api/models.py` en `api/schemas/{...}.py` | 4-6 h | Re-export depuis `__init__` pour compat |
| Tests d'intégration pour chaque adapter (Telegram + futur TradingView webhook) | 8-12 h | |

**Durée totale Phase B** : 34-52 h (≈ 4-6 sem solo).
**Gate de sortie** : ajout d'un adapter dummy (`StdoutAdapter` qui print) ne touche AUCUN fichier hors `delivery/`.

## Phase C — Stabilisation de l'API REST et des contrats

**Objectif** : contrats v1 figés, client TypeScript généré, webapp branchée en propre.

| Livrable | Durée | Détails |
|---|---|---|
| OpenAPI 3.1 export complet + tags compliance | 4-6 h | Déjà partiellement fait |
| Génération client TypeScript dans `webapp/lib/api/generated/` via `openapi-typescript-codegen` | 4-6 h | Remplace les fetch hand-coded |
| Remplacement mocks `webapp/lib/mocks.ts` par appels API réels | 12-20 h | Toutes les pages |
| Zod schemas générés depuis OpenAPI pour validation runtime webapp | 4-6 h | |
| Versioning header `X-Schema-Version` côté API + check webapp | 3-4 h | |
| Tests E2E Playwright étendus | 8-12 h | |

**Durée totale Phase C** : 35-54 h (≈ 4-6 sem solo).
**Gate de sortie** : webapp en prod consomme l'API réelle, mocks `webapp/lib/mocks.ts` supprimé, tests E2E couvrent landing + insight + chat + checkout.

## Phase D — Livraison V1 (3 canaux retenus)

**Objectif** : 1er paiement Stripe live encaissé.

| Livrable | Durée | Détails |
|---|---|---|
| Wire LGBM `scoring_v3_lgbm.pkl` dans `CalibratedConvictionPipeline` (DG-025) | 20-30 h | Chemin critique #1 |
| Chatbot wiring 8 composantes + refus pédagogique + 3 questions (DG-110/112/114) | 32-48 h | Moat #1 |
| Landing hero card track-record (DG-120) + page pricing decoy (DG-132) + track-record public (DG-142) | 32-48 h | |
| Plausible self-hosted + 6 events core (DG-160/161) | 16-20 h | |
| Telegram inline buttons + idempotency + inbound conversational | 22-30 h | |
| TradingView Pine showcase script + profil + landing `/tv` | 26-36 h | |
| Service Worker Web Push (PWA push iOS 16.4+) | 8-12 h | |
| Compliance Stripe live activation (DG-043 + Tax UE DG-044 + geo-block DG-045) | 6-10 h | |
| Bootstrap legal — Iubenda templates + RC Pro | hors-dev (admin) | |

**Durée totale Phase D** : 162-234 h (≈ 20-29 sem solo) — recouvre Vague 1 Sprint S3-S6 du plan dev_focus.
**Gate de sortie** : 1er paiement Stripe live encaissé. Track-record Telegram public ≥ 30 trades clôturés visibles. Audit compliance UE 2024/2811 + bootstrap légal passé.

## Phase E — Livraison V2

**Objectif** : MRR > $5k, MAU > 200.

| Livrable | Durée | Détails |
|---|---|---|
| Email digest quotidien (déclenche MAU > 100) | 12-20 h | |
| TradingView webhook receiver | 30-40 h | |
| Bot Discord | 16-24 h | |
| Push web (PWA) production | 8-12 h | (déjà service worker, ajouter logique alerte) |
| Mobile cross-platform React Native + Expo (si conditions §3.2 atteintes) | 200-300 h | Gros chantier conditionnel |
| Migration Redis (DG-020), multi-worker Gunicorn (DG-024) si MAU > 200 | 12-20 h | |
| Sentry Team + onboarding 4-step + email cycle (DG-121/131) | 28-46 h | |
| Conformal viz + waterfall pédagogique EXPERT (DG-170-173) | 60-80 h | |

**Durée totale Phase E** : 366-542 h (≈ 45-67 sem) — distribué sur 6 mois si Vague 2 démarre M3.
**Gate de sortie** : Conv landing→trial > 2 %, activation trial→paid > 15 %, churn M1 < 25 %, MAU > 200, MRR > $5k.

## Phase F — Livraison V3

**Objectif** : pivot B2B + expansion géographique.

| Livrable | Durée | Détails |
|---|---|---|
| API REST B2B public + OpenAPI public + portal docs | 40-60 h | |
| Webhook B2B push (queue + DLQ + signing) — déjà partial | 20-30 h | |
| Extension navigateur Chrome/Firefox | 80-120 h | |
| Widget embeddable + plugin WordPress | 40-60 h | |
| SMS premium (Twilio) | 12-20 h | |
| ChatGPT/Claude tool integration | 30-50 h | |
| Migration Postgres (DG-026) si MRR > $5k | 16-24 h | |
| RAG complet BM25 + dense + RRF (DG-058b) si conditions | 60-80 h | |

**Durée totale Phase F** : 298-444 h (≈ 37-55 sem) — distribué sur 9 mois.
**Gate de sortie** : 1 partenariat B2B signé (broker ou éducateur), MRR > $15k, présence multi-pays Phase 1 confirmée.

## 7.7 Résumé du plan de migration

| Phase | Durée | Cumul | Vague |
|---|---|---|---|
| A — Cleanup | 12-17 h | 17 h | pre-V1 |
| B — Découplage | 34-52 h | 69 h | pre-V1 |
| C — Stabilisation contrats | 35-54 h | 123 h | pre-V1 |
| D — V1 delivery | 162-234 h | 357 h | V1 (S1-S6) |
| E — V2 acquisition | 366-542 h | 899 h | V2 |
| F — V3 expansion | 298-444 h | 1343 h | V3 |

**Pre-V1 (cleanup + ports/adapters + contrats)** : 81-123 h ≈ **10-15 sem à 8-9h/sem**. C'est l'investissement architectural avant de livrer V1.

**V1 total cumulé** : ~360 h ≈ **45 sem (un peu moins d'un an)** — cohérent avec le pivot dev-focus exclusif 10-12 sem signé 2026-05-27 SI on accepte de paralléliser pré-V1 et V1.

---

# Partie 8 — Synthèse pour décision

## Les 3 canaux V1 retenus + justification commerciale

| # | Canal | Pourquoi V1 |
|---|---|---|
| 1 | **Webapp SaaS (PWA inclus)** | Hub central où s'organise toute la promesse produit (3 couches FOCUS/CO-PILOT/EXPERT + chatbot). Sans, rien à vendre. PWA couvre mobile 70-80 % sans dev natif. |
| 2 | **Bot Telegram** | Wedge FR-first identifié (eval_25). Canal FREE essentiel pour conversion. Audience retail FR sur-représentée Telegram. Effort modéré (40-50 h). |
| 3 | **TradingView Pine showcase** | **Acquisition gratuite** dans le hub où vit la cible. Pattern prouvé (LuxAlgo et autres). Zéro infra côté MIA (Pine pur, pas de webhook). Effort 26-36 h. Repousser à V2 = perdre 3-9 mois d'acquisition gratuite. |

## Les 3 décisions tech à prendre maintenant qui engagent le futur

| # | Décision | Pourquoi maintenant |
|---|---|---|
| 1 | **Source unique du contrat `InsightSignalV2`** = `src/intelligence/insight_v2/`. L'API et la webapp importent. | Plus on retarde, plus le drift entre `api/models.py` et `intelligence/insight_v2/` devient coûteux à réconcilier. À faire Phase B. |
| 2 | **Auth = Clerk** (et pas Supabase Auth ni NextAuth) | Changer d'auth provider plus tard coûte 80-150 h et casse les utilisateurs existants. Décider maintenant = construire avec le bon JWT bearer + webhook signup. |
| 3 | **Mobile V2 = React Native + Expo** (et pas Flutter ni natif) | Le choix du framework mobile détermine 200-300 h d'effort. Décider maintenant permet d'aligner le client API généré (TS) + monorepo possible si désiré, et de ne pas se peindre dans un coin Dart. |

## Le refactor immédiat le plus urgent

**Wire LGBM (DG-025) + Archive legacy RL (Phase A)**.

- Wire LGBM = chemin critique #1 de l'audit 2026-05-27 (scoring rule-based Pearson −0.023 → modèle calibré disponible non branché). Tant que la **Gate de promotion premium** n'est pas franchie (Brier skill OOS > +2 % ET DSR > 1.0 ET PBO < 0.5 — cf. `pivot_positioning_2026_05_27`), **aucun chiffre de performance n'est publiable** ; le positionnement est "outil de compréhension augmentée", pas "edge prédictif chiffré".
- Archive legacy RL = 12 000 LOC de bruit mental + risque de redémarrage accidentel d'un pipeline mort. À faire **avant** tout autre refactor (Phase A, 12-17 h).

## Le coût mensuel total estimé de la stack cible

| Vague | Coût mensuel fixe | Coût variable |
|---|---|---|
| **V1** (MAU < 200) | **~$50/mo** | + Stripe 2,9 % + $0,30/tx + LLM ~$30/mo cible |
| **V2** (MAU 200-500, MRR > $5k) | **~$200-300/mo** | + LLM scaled $50-150/mo + Apple Dev $8/mo |
| **V3** (MAU > 1k, B2B) | **~$400-600/mo** | + Trading Economics $79/mo + SMS variable + Qdrant $25/mo |

V1 est **délibérément maigre** ($50/mo fixe). Couvre Fly.io backend Paris + Plausible self-hosted + Cloudflare DNS + Resend transactionnel + Anthropic LLM + R2 backups + domaine.

## Le ratio effort/ROI par vague

| Vague | Effort cumulé | ROI attendu | Ratio |
|---|---|---|---|
| **Pre-V1 (A+B+C)** | 81-123 h | Architecture saine prête à recevoir N canaux | **Multiplicateur** : sans, chaque canal V2/V3 coûte 2× plus cher. |
| **V1 (Phase D)** | 162-234 h | 1er paiement Stripe + audience départ + canal TV acquisition gratuite | **Élevé** : transition 0 → revenu. Tout dépend de ça. |
| **V2 (Phase E)** | 366-542 h | MRR $5-15k, MAU 200-500 | **Moyen** : effort lourd mais rétention + acquisition compound. |
| **V3 (Phase F)** | 298-444 h | Diversification B2B + mobile + expansion | **Conditionnel** : ROI dépend de DG-071 (MRR B2C > $5k 3 mois). Si conditions remplies, gain massif. Sinon, gaspillage. |

## ⭐ Si tu ne devais retenir qu'une chose

> **L'architecture sert N surfaces sur un seul moteur. Le `InsightSignalV2` canonique est la pièce centrale, le `DeliveryEngine` est la voie de sortie unique, les renderers sont jetables. Chaque canal de distribution devient un fichier `adapters/<canal>.py` + `renderers/to_<canal>.py` — jamais un changement dans le noyau algo, jamais une duplication du contrat. C'est la seule architecture qui te permet de partir à 3 canaux V1 et de finir à 12 canaux en V3 sans réécrire ton système.**

> Tout le reste (Postgres vs SQLite, Clerk vs NextAuth, React Native vs Flutter) est subordonné à cette propriété cardinale : **un noyau, N surfaces**.

