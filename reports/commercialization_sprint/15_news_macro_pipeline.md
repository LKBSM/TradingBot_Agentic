# Plan de Commercialisation — Catégorie 15 : News & Macro Pipeline

> **Smart Sentinel AI** · Branch `institutional-overhaul` · Date 2026-05-21
> Auteur : agent IA (claude-opus-4-7) · loukmanebessam@gmail.com
> Périmètre : economic calendar (ForexFactory + alternatives), `NewsAnalysisAgent`,
> blackout windows, news sentiment, macro events, central bank speakers,
> surprise calculation.
>
> Doc source (lus avant rédaction) :
> - `reports/eval_29_compliance.md` (FF zone grise, AMF/MiFID II finfluencers 2026)
> - `src/agents/news/economic_calendar.py:130-674` (fetcher CSV + scraper FF + fallback builtin)
> - `src/agents/news_analysis_agent.py:1-786` (orchestration blackout + sentiment + obs space)
> - `src/agents/news/fetchers.py:1-120` (NewsAPI free tier + RSS BCE/Fed)
> - `src/agents/news/sentiment.py:1-100` (rule-based keyword sentiment)
> - `src/agents/news/aggregator.py:1-80` (multi-source aggregator Sprint 3)
> - `src/agents/news/websocket_feed.py:1-80` (skeleton WS jamais wiré en prod)
> - `src/agents/news/sources/{rss_adapter,fed_watch_adapter,cot_adapter,twitter_adapter}.py`
> - `scripts/fetch_forexfactory_live.py:1-184` (cron ff_calendar_thisweek/nextweek)
> - `scripts/crosscheck_mt5_calendar.py:1-80` (validation MT5 vs FF)
> - `src/backtest/news_replay.py:1-60` (replay blackout-only sur CSV)
> - `src/strategies/event_driven_macro.py:1-80` (Pilier 1, post-release momentum)
> - `src/intelligence/sentinel_scanner.py:78-440` (wiring `news_agent.evaluate_news_impact()`)
> - `data/macro/{cot_gold.csv, fred_*.csv}` (DXY, VIX, real yields, breakevens, COT)
> - `data/economic_calendar_2019_2025.csv` (1 380 lignes — **Actual/Forecast vides**)
> - `data/economic_calendar_HIGH_IMPACT_2019_2025.csv` (876 lignes, idem)

---

## 0. TL;DR — 90 secondes

| Axe | Note actuelle /10 | Cible 30 j | Cible 90 j |
|---|---|---|---|
| Légalité source calendrier | **2** (FF scrapé/JSON sans licence commerciale) | 6 | 9 |
| Fraîcheur & couverture | 5 (cron toutes 30 min, 1 fournisseur, 0 fallback) | 8 | 9 |
| Surprise scoring | **0** (Actual/Forecast 100 % vides en CSV historique) | 7 | 9 |
| Live news / sentiment | **2** (NewsAPI free 100 req/j + WS jamais wiré + sentiment keyword) | 5 | 8 |
| Central bank speakers | 1 (RSS Fed/ECB skeleton, pas de wiring) | 4 | 7 |
| Tests & dataloss SLA | 3 (zéro test pipeline live, replay couvre seulement BLOCK) | 7 | 9 |
| **Note globale News & Macro** | **2.4 / 10** | **6.2 / 10** | **8.5 / 10** |

**3 verdicts factuels :**
1. **News=0 et Vol=0 en replay plafonnent le score Confluence à 70/100** (cf. `reports/audit_backtest_2026_04_24.md`) — le P0 critique pour commercialiser n'est pas une nouvelle source mais **enrichir le CSV historique** avec Actual/Forecast/Previous depuis Trading Economics ou FRED API pour ressusciter le surprise et la calibration backtest.
2. **`scripts/fetch_forexfactory_live.py` utilise `nfs.faireconomy.media`** — feed JSON officieux qu'utilisent les EAs MT4/MT5 depuis 10 ans, mais **CGU ForexFactory interdit usage commercial** (`reports/eval_29_compliance.md` §5.3). Tant qu'on facture, **risque cease-and-desist réel** (FF cible Cloudflare les scrapers). Migration vers licence commerciale = P0 absolu pour go-live payant.
3. **`event_driven_macro.py:14-17` reconnaît explicitement** : « *the historical ForexFactory CSV we have does NOT contain Actual / Forecast columns* ». L'institutional plan Pilier 1 (80 h) **ne peut pas exploiter la surprise** sans nouvelle source — c'est le **goulot d'étranglement quant**.

---

## 1. État actuel — Audit du pipeline

### 1.1 Chaîne actuelle (data flow)

```
                         OFFLINE / NIGHTLY
┌─────────────────────────────────────────────────────────────────────┐
│  scripts/fetch_forexfactory_live.py                                 │
│  ── GET https://nfs.faireconomy.media/ff_calendar_thisweek.json     │
│  ── GET https://nfs.faireconomy.media/ff_calendar_nextweek.json     │
│  ── normalise ET→UTC, upsert (Date, Currency, Event)                │
│  ── écrit data/economic_calendar_live.csv                           │
│     (col: Date, Currency, Event, Impact, Actual, Forecast, Previous)│
└──────────────────────────────┬──────────────────────────────────────┘
                               │ cron 30 min (Win Task Scheduler)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EconomicCalendarFetcher(csv_path=$CALENDAR_PATH)                   │
│  src/agents/news/economic_calendar.py:130-457                       │
│  ── source priority: CSV > HTML scraper FF > builtin schedule       │
│  ── cache mtime-aware, refresh auto sans restart                    │
│  ── _classify_impact() heuristique mots-clés                        │
│  ── _detect_currency() heuristique                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  NewsAnalysisAgent.evaluate_news_impact()                           │
│  src/agents/news_analysis_agent.py:332-466                          │
│  ── blackout HIGH: 30 min before + 30 min after                     │
│  ── reduce MEDIUM: 15 min before/after, position ×0.5               │
│  ── sentiment: SentimentAnalyzer (keyword rule-based)               │
│  ── sentiment_impact_on_sizing = 0.1 max                            │
│  ── retourne NewsAssessment{decision, position_multiplier, …}       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SentinelScanner._scan_once()                                       │
│  src/intelligence/sentinel_scanner.py:390-440                       │
│  ── construit TradeProposal stub (BUY si BOS_SIGNAL>0, asset=XAU)   │
│  ── appelle news_agent.evaluate_news_impact(proposal)               │
│  ── passe NewsAssessment à ConfluenceDetector (gate blackout)       │
└─────────────────────────────────────────────────────────────────────┘

                         BACKTEST PATH
┌─────────────────────────────────────────────────────────────────────┐
│  BacktestNewsProvider                                               │
│  src/backtest/news_replay.py:1-60                                   │
│  ── charge CSV HIGH_IMPACT_2019_2025                                │
│  ── retourne NewsAssessment(BLOCK) si bar ∈ blackout                │
│  ── retourne None sinon (news=0, plafonne score à 70/100)           │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Composants disponibles mais **non wirés**

| Composant | Fichier | État | Observation |
|---|---|---|---|
| `WebSocketNewsFeed` | `src/agents/news/websocket_feed.py:77-…` | skeleton complet, reconnect exp backoff | Aucun URL prod, jamais appelé depuis `NewsAnalysisAgent`. Sprint 3 abandonné. |
| `NewsAggregator` | `src/agents/news/aggregator.py:76-…` | multi-source + dedup + EventBus | Pas wiré dans `sentinel_scanner.py` ; double pipeline avec `NewsHeadlineFetcher`. |
| `RSSAdapter` | `src/agents/news/sources/rss_adapter.py:50-…` | feedparser+aiohttp, feeds Reuters/Bloomberg/FF | Dépendances optionnelles non installées (`HAS_FEEDPARSER=False` en prod). |
| `FedWatchAdapter` | `src/agents/news/sources/fed_watch_adapter.py:1-40` | skeleton CME FedWatch | Pas d'API key Finnhub, pas de fallback fonctionnel. |
| `TwitterAdapter` | `src/agents/news/sources/twitter_adapter.py` | skeleton | X API v2 payant ($100/mo basic), pas utilisable solo. |
| `COTAdapter` | `src/agents/news/sources/cot_adapter.py` | CFTC COT report | `data/macro/cot_gold.csv` existe — partiel. |
| Macros FRED | `data/macro/fred_*.csv` | DXY, VIX, real yields, breakevens, 10Y-2Y | Téléchargés mais **jamais consommés** par `NewsAnalysisAgent`. |
| `crosscheck_mt5_calendar.py` | `scripts/crosscheck_mt5_calendar.py:1-130` | validation MT5 vs FF | Pas branché en startup gate (ligne 19 mentionne intention mais TODO). |

### 1.3 Risques identifiés (factuels, citables)

| # | Risque | Fichier:ligne | Sévérité |
|---|---|---|---|
| R1 | **FF JSON usage commercial non licencié** | `scripts/fetch_forexfactory_live.py:47-50` + `eval_29_compliance.md` §5.3 | 🔴 Bloque go-live payant |
| R2 | **Actual/Forecast/Previous vides** dans CSV historique 7 ans | `data/economic_calendar_2019_2025.csv` (`grep Actual non-empty: 0`) | 🔴 Bloque surprise scoring + Pilier 1 |
| R3 | **HTML scraper FF fragile** — placeholder timestamp `now+1h` | `economic_calendar.py:519-526` | 🟠 Si JSON tombe, fallback inutile (heures fausses) |
| R4 | **Builtin schedule = placeholders** `now + i days + 14h30` | `economic_calendar.py:594-606` | 🟠 Données factices servies si CSV+scraper KO |
| R5 | **TradeProposal stub mono-asset** (BUY hardcodé) | `sentinel_scanner.py:393-398` | 🟡 Sentiment biaisé long, pas testé short |
| R6 | **Sentiment keyword anglais uniquement** | `sentiment.py:51-92` | 🟡 Manque FR, DE, ES (hub FR-first prévu) |
| R7 | **NewsAPI free 100 req/j** — partage avec 17+ users = quota | `fetchers.py:79-91` | 🟡 Premium $449/mo ou switch fournisseur |
| R8 | **MT5 calendar crosscheck non gated** | `crosscheck_mt5_calendar.py:19` (TODO) | 🟢 Améliore SLA mais pas bloquant |
| R9 | **Pas de pipeline live news (push)** | `websocket_feed.py` skeleton, pas wiré | 🟠 Latence 60s+ vs Reuters Eikon <1s |
| R10 | **No central bank speakers tracker** | absent | 🟠 Powell/Lagarde discours sont 30 % des jumps gold |
| R11 | **No retry / dead letter queue** sur `fetch_forexfactory_live.py` | script:166-180 exit 1 silent | 🟡 Si feed down 4h pendant NFP = trade aveugle |
| R12 | **Aucun horodatage UTC strict côté agent** — mix `datetime.now()` local | `news_analysis_agent.py:381,388,602` | 🟠 Erreurs blackout off-by-Xh selon serveur |

---

## 2. Vision cible — Pipeline news/macro institutional-grade

```
                    LICENSED PROVIDERS (P0)
┌─────────────────────────────────────────────────────────────┐
│  Trading Economics REST API (calendar + indicators)         │
│  ── $79-499/mo · TOS commercial · Actual/Forecast/Previous  │
│  ── 80+ countries · history depuis 1980                     │
│                                                             │
│  FRED API (free, gov US public domain) — DXY, VIX, yields   │
│  ── enrichissement macro features                           │
│                                                             │
│  ECB SDW API (free, EUR public domain) — rates, M3, HICP    │
│  BLS API (free, US gov) — fallback NFP/CPI/PCE              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  CalendarIngestService (NEW)                                │
│  src/agents/news/ingest_service.py                          │
│  ── pull TE primary, fallback FRED/BLS/ECB                  │
│  ── store SQLite data/calendar.db + CSV legacy compat       │
│  ── compute surprise = (actual - forecast) / σ_historical   │
│  ── normalize event_id stable (cross-provider keying)       │
│  ── DLQ + circuit breaker (5 retries exp backoff)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LIVE NEWS STREAMING (P1)                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  NewsAPI.ai (Event Registry)  — $200/mo commercial   │  │
│  │  Benzinga News API           — $300/mo (calls equity)│  │
│  │  Reuters/Bloomberg via Refinitiv (institutional, $$$)│  │
│  │  Federal Reserve RSS, ECB RSS, BoE RSS (free)        │  │
│  │  X API v2 Basic $100/mo (cb_speakers monitoring)     │  │
│  └────────────────────┬──────────────────────────────────┘  │
│                       │ WebSocket / SSE / poll 60s          │
│                       ▼                                     │
│  NewsStreamService (NEW, async)                             │
│  ── dedupe par URL hash + Levenshtein 90% headline          │
│  ── pub EventBus → SemanticCache → LLM (sentiment)          │
│  ── store data/news.db (24h hot, 90j warm, S3 cold)         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  NewsAnalysisAgent v2 (refactored)                          │
│  ── blackout = surprise-aware (block si |z|>2 attendu)      │
│  ── sentiment LLM (Claude Haiku $0.25/M tok)                │
│  ── per-currency aggregation + time-decay                   │
│  ── feeds ConfluenceDetector + LLMNarrativeEngine           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  ConfluenceDetector / event_driven_macro.py                 │
│  ── consume news_surprise feature (numeric, z-score)        │
│  ── enable Pilier 1 event-driven strategy with surprise gate│
└─────────────────────────────────────────────────────────────┘
```

**Coût providers cible v1 (FR-first solo) :**
- Trading Economics « Smart » plan : **$79/mo** (calendar + 1 country + REST)
- NewsAPI.ai Basic : **$200/mo** (news commercial, sentiment build-in)
- FRED / BLS / ECB / BoE : **$0** (public domain)
- X API v2 Basic (CB speakers) : **$100/mo** (optionnel S2)
- **TOTAL P0+P1 : $279/mo** (~250 €/mo) — marge intacte ~95 % sur tier ANALYST $29
- Upgrade Strategist+ : Trading Economics Pro $499/mo, Benzinga $300/mo, Refinitiv Eikon $2k/mo

---

## 3. Gap analysis — État vs cible

| Capacité | Actuel | Cible | Gap |
|---|---|---|---|
| Source calendrier licencée | FF non-com | TE $79/mo | Achat licence + intégration REST |
| Couverture Actual/Forecast/Previous | 0 % historique | 100 % depuis 2010 | Backfill TE history dump |
| Surprise scoring | absent | `(actual-forecast)/σ` z-score persisté | Calcul + storage + feature wiring |
| Blackout dynamique (tier 1 only) | binaire HIGH | tiered (T1/T2 + surprise gate) | Refonte `_check_event_blocking()` |
| Live news streaming | NewsAPI free + skeletons | NewsAPI.ai WebSocket + RSS BC | Refactor `NewsAggregator` + wire WS |
| Sentiment | keyword EN | LLM multi-lang Claude Haiku | Remplacement `SentimentAnalyzer` |
| Central bank speakers | aucun | X feed + RSS + LLM extract | Nouveau `CBSpeakerTracker` |
| Replay enrichi | 0 (News=0) | full Actual/Forecast/surprise | Backfill + provider mode replay |
| Calendar consistency check | manuel | startup gate + alerte Slack | Brancher `crosscheck_mt5_calendar` |
| Audit license | dette | DPA/CGU TE + Anthropic signés | Légal §6 ci-dessous |

---

## 4. Plan d'exécution

> **Convention** : chaque tâche liste *Fichiers*, *Heures*, *Acceptance*, *Dépendances*.
> Heures = solo founder mid-senior, sandbox dev + 1 cycle revue.

### P0 — Bloquant commercialisation (Semaine 1-3, ~70 h)

#### P0.1 — Migration ForexFactory → Trading Economics (licencié) — **18 h**

- **Fichiers** :
  - **NEW** `src/agents/news/providers/trading_economics.py` (client REST + auth)
  - **NEW** `src/agents/news/providers/base_calendar_provider.py` (interface)
  - **MOD** `src/agents/news/economic_calendar.py:194-231` (accept `provider=` injection au lieu de csv_path only)
  - **MOD** `scripts/fetch_forexfactory_live.py` → renommer `fetch_calendar_live.py`, ajouter flag `--provider {te,fred,ff_archive}`, garder FF en mode **dev-only** (env `ALLOW_FF=1`)
  - **NEW** `tests/test_trading_economics_provider.py`
  - **MOD** `config.py` : ajouter `TE_API_KEY`, `CALENDAR_PROVIDER` env vars
  - **MOD** `requirements.txt` : `httpx>=0.27`
- **Acceptance** :
  - `pytest tests/test_trading_economics_provider.py -v` → 100 % green
  - `python scripts/fetch_calendar_live.py --provider te --days 14` produit CSV avec colonnes Actual/Forecast/Previous **non vides** sur ≥ 80 % des rows
  - Smoke : avec `TE_API_KEY` invalide → fallback gracieux FRED/ECB sans crash scanner
  - Audit : `grep -r "forexfactory" src/ --include='*.py' | grep -v test` → 0 référence active hors archive
- **Dépendances** : compte Trading Economics ($79/mo plan « Smart »), test sandbox 7 j gratuit

#### P0.2 — Backfill Actual/Forecast/Previous CSV historique 7 ans — **12 h**

- **Fichiers** :
  - **NEW** `scripts/backfill_calendar_history.py` (TE history API + FRED bulk)
  - **OUT** `data/economic_calendar_2019_2025_v2.csv` (enrichi)
  - **OUT** `data/economic_calendar_surprise_2019_2025.csv` (avec z-score, σ historique)
  - **NEW** `tests/test_calendar_backfill.py` (vérifie ≥ 90 % NFP/CPI/FOMC enrichis)
- **Acceptance** :
  - `awk -F, 'NR>1 && $5!="" {a++} END {print a}'` ≥ 1 200 lignes (vs 0 aujourd'hui)
  - `data/economic_calendar_surprise_2019_2025.csv` contient colonne `Surprise_Z` calculée sur σ(actual-forecast) glissant 24 mois
  - Replay backtest avec CSV v2 produit `news_score > 0` sur ≥ 30 % des bars NFP/CPI (vs 0 % aujourd'hui)
- **Dépendances** : P0.1 (provider TE), `src/backtest/news_replay.py:1-60` étendu

#### P0.3 — Surprise calculation → feature `news_surprise` — **8 h**

- **Fichiers** :
  - **NEW** `src/agents/news/surprise_scorer.py` (`compute_surprise(actual, forecast, sigma_window=24M)`)
  - **MOD** `src/agents/news/economic_calendar.py:63-86` (ajouter `surprise_z: float`, `surprise_signed: bool` à `EconomicEvent`)
  - **MOD** `src/agents/news_analysis_agent.py:332-466` (intégrer `news_surprise` dans `NewsAssessment`)
  - **MOD** `src/intelligence/confluence_detector.py` (nouveau composant `news_surprise_score` à blender, weight 0 Phase 1 → empirique Phase 2)
  - **NEW** `tests/test_surprise_scorer.py` (cas NFP +250k vs forecast 180k → z>0, BUY USD)
  - **MOD** `src/intelligence/insight_v2/contract.py` : ajouter `news_surprise_z`, `news_event_name` au schema
- **Acceptance** :
  - Test unit : `compute_surprise(actual=250e3, forecast=180e3, sigma=50e3)` → +1.4 ± 0.05
  - Smoke replay 2024 sur NFP : `news_surprise` populé sur 12/12 release
  - Insight V2 payload contient `news_surprise_z` quand event-window
- **Dépendances** : P0.2 (données enrichies)

#### P0.4 — Calendar replay enrichment (backfill News=0) — **10 h**

- **Fichiers** :
  - **MOD** `src/backtest/news_replay.py:1-60` (consume CSV v2 avec Actual/Forecast)
  - **NEW** `src/backtest/news_replay.py:_compute_event_surprise()` 
  - **NEW** `tests/test_news_replay_with_surprise.py`
  - **MOD** `scripts/run_backtest_xau.py` (si existe — sinon job ad-hoc) pour rejouer 2019-2025 avec nouveau pipeline
  - **NEW** `reports/backtest/news_enriched_xau_2019_2025.md` (compare métriques avant/après)
- **Acceptance** :
  - Replay XAU M15 2019-2025 avec CSV v2 : taux `news_score > 0` passe de 0 % → ≥ 15 % des bars
  - Plafond score Confluence remonte de 70 → ≥ 85
  - PF backtest 2019-2025 mis à jour, **document écrit même si PF stagne** (les findings importent)
- **Dépendances** : P0.2, P0.3

#### P0.5 — Blackout windows validation per instrument — **8 h**

- **Fichiers** :
  - **MOD** `src/agents/news_analysis_agent.py:111-130` : ajouter `blackout_windows: Dict[str, Tuple[int,int]]` per event-type, par instrument
  - **NEW** `config/blackout_matrix.yaml` :
    ```yaml
    XAUUSD:
      NFP:   {before: 15, after: 60}
      FOMC:  {before: 30, after: 120}
      CPI_US: {before: 15, after: 45}
      ECB:   {before: 10, after: 30}    # XAU moins sensible ECB
    EURUSD:
      ECB:   {before: 30, after: 90}
      FOMC:  {before: 15, after: 60}
      ...
    ```
  - **MOD** `src/backtest/news_replay.py:36-43` (DEFAULT_AFFECTING_CURRENCIES → YAML)
  - **NEW** `tests/test_blackout_matrix.py` (chaque preset = ≥ 5 cas)
- **Acceptance** :
  - Pour chaque instrument des 6 presets, blackout matrix définie pour ≥ 4 event-types
  - Test : `is_within_blackout("XAUUSD", "NFP", now=T-10min)` → True ; `is_within_blackout("XAUUSD", "NFP", now=T+65min)` → False
  - Audit replay : `XAUUSD` NFP 30 min avant/après 60 min = 5 760 trades exclus / 7 ans (à mesurer empiriquement)
- **Dépendances** : P0.3 (surprise gates si surprise<seuil → no-block même HIGH)

#### P0.6 — Compliance license audit + DPA — **6 h** (90 % légal externe)

- **Fichiers** :
  - **NEW** `compliance/dpa_trading_economics.pdf` (à demander à TE)
  - **NEW** `compliance/license_audit_calendar.md` (mémo : TE = primary, FRED/BLS/ECB = public domain, FF = **archived only**)
  - **MOD** `reports/eval_29_compliance.md` §5.3 (mettre à jour statut FF)
  - **NEW** `.env.example` : `CALENDAR_PROVIDER=te`, `ALLOW_FF=0` (default prod)
- **Acceptance** :
  - DPA TE signé scanné dans `compliance/`
  - `grep -r "forexfactory.com" src/ scripts/` retourne 0 matches en prod (seulement archive flag)
  - Avocat fintech (cf. eval_29 §6.4) valide via mémo écrit court
- **Dépendances** : P0.1, contact TE support

#### P0.7 — Tests pipeline + circuit breaker — **8 h**

- **Fichiers** :
  - **NEW** `tests/test_calendar_pipeline_e2e.py` (mock TE API → CalendarFetcher → NewsAgent → blackout)
  - **NEW** `tests/test_calendar_freshness_sla.py` (fail si dernier fetch > 2 h)
  - **MOD** `src/agents/news/economic_calendar.py:287-347` : wrap dans `@circuit_breaker` (cf. `src/intelligence/circuit_breaker.py`)
  - **NEW** `src/api/routes/health.py` : endpoint `/health/calendar` retourne `{last_fetch, age_minutes, sources_up}`
- **Acceptance** :
  - 8 nouveaux tests, tous verts
  - Simul outage TE → fallback FRED en < 5 s sans bug scanner
  - Endpoint `/health/calendar` retourne 503 si stale > 2 h (alertable Grafana)
- **Dépendances** : P0.1

---

### P1 — Live news streaming + sentiment LLM (Semaine 4-6, ~60 h)

#### P1.1 — Wire NewsAPI.ai (Event Registry) commercial — **16 h**

- **Fichiers** :
  - **NEW** `src/agents/news/providers/newsapi_ai.py` (client REST + WebSocket)
  - **MOD** `src/agents/news/aggregator.py:76-…` (register NewsAPI.ai as PRIMARY)
  - **MOD** `src/agents/news/fetchers.py:79-…` (déprécier NewsAPI free, conserver pour dev)
  - **NEW** `tests/test_newsapi_ai_provider.py`
  - **MOD** `requirements.txt` : `eventregistry>=8.12`, `websockets>=12`
  - **MOD** `config.py` : `NEWSAPI_AI_KEY`, `NEWS_PROVIDERS=newsapi_ai,reuters_rss,fed_rss`
- **Acceptance** :
  - 50 articles/h ingérés en moyenne (mesure 1 sem)
  - Latence p50 < 30 s, p95 < 90 s entre publication source et `NewsAggregator.publish()`
  - Dedup cross-provider : ≤ 5 % duplicates dans data/news.db
- **Dépendances** : compte NewsAPI.ai Basic ($200/mo), DPA NewsAPI.ai

#### P1.2 — Wire WebSocketNewsFeed en prod — **10 h**

- **Fichiers** :
  - **MOD** `src/agents/news/websocket_feed.py:77-…` (compléter `connect()`, `_message_loop()`, dedup window 60s)
  - **MOD** `src/agents/news_analysis_agent.py:239-291` (créer + démarrer WS task au `initialize()`)
  - **NEW** `tests/test_websocket_news_feed.py` (mock WS server)
  - **MOD** `src/api/routes/health.py` : ajouter `ws_news_connected: bool` au healthcheck
- **Acceptance** :
  - WS feed reconnecte automatiquement après cut réseau (exp backoff 1-60 s)
  - Heartbeat échec → reconnect en < 15 s
  - `/health/news_stream` retourne `connected=true, last_message_age_s<60`
- **Dépendances** : P1.1 (provider WS support)

#### P1.3 — Remplacer SentimentAnalyzer keyword par Claude Haiku — **14 h**

- **Fichiers** :
  - **NEW** `src/agents/news/sentiment_llm.py` (wrapper Claude Haiku, cache, multi-lang)
  - **MOD** `src/agents/news/sentiment.py:36-…` : conserver en fallback offline (env `SENTIMENT_MODE=keyword|llm`)
  - **MOD** `src/agents/news_analysis_agent.py:559-588` : `_calculate_aggregated_sentiment()` accepte mode
  - **NEW** `tests/test_sentiment_llm.py` (10 cas FR + EN + DE)
  - **MOD** `src/intelligence/semantic_cache.py` : cache key inclut hash headline (TTL 6h)
- **Acceptance** :
  - 100 headlines analysées via Haiku coûtent < $0.02 (cible cost-aware)
  - Multi-lang : 10 headlines FR + DE + ES → sentiment cohérent (≥ 80 % accord vs gold-standard humain)
  - Latence p50 < 800 ms, p95 < 2 s
  - Fallback automatique → keyword si Anthropic circuit open
- **Dépendances** : `ANTHROPIC_API_KEY` (déjà présent), tier-routing (cf. eval_05)

#### P1.4 — Reuters/Bloomberg RSS (free legal) — **6 h**

- **Fichiers** :
  - **MOD** `src/agents/news/sources/rss_adapter.py:50-…` (activer feeds Reuters Markets, BBG Economics)
  - **MOD** `requirements.txt` : `feedparser>=6.0` (currently optional → required)
  - **NEW** `tests/test_rss_adapter.py`
- **Acceptance** :
  - 10 feeds RSS pollés toutes 5 min, 0 erreur 24 h
  - Disclaimer RSS « rights belong to publisher » dans `compliance/license_audit_news.md`

#### P1.5 — `news_score` feature wiring dans Confluence — **8 h**

- **Fichiers** :
  - **MOD** `src/intelligence/confluence_detector.py` (ajout composant `news_sentiment_score` weight 0 → empirique)
  - **MOD** `src/intelligence/insight_v2/builder.py` (read sentiment depuis NewsAssessment)
  - **NEW** `tests/test_confluence_with_news.py`
- **Acceptance** :
  - Replay XAU 2024 : composant news non-null sur ≥ 60 % des bars
  - Backtest : Pearson(news_sentiment_score, futur_return_4h) calculé et reporté

#### P1.6 — Tests SLA streaming + circuit breaker news — **6 h**

- **Fichiers** :
  - **NEW** `tests/test_news_stream_resilience.py`
  - **MOD** `src/intelligence/circuit_breaker.py` (wrap NewsAggregator)
- **Acceptance** :
  - Simul outage NewsAPI.ai → fallback RSS sans crash
  - SLO documenté : 99 % uptime, RPO 5 min, RTO 60 s

---

### P1.bis — Central bank speakers tracker (Semaine 6-7, ~24 h)

#### P1b.1 — `CBSpeakerTracker` (Fed/ECB/BoE/BoJ) — **14 h**

- **Fichiers** :
  - **NEW** `src/agents/news/cb_speaker_tracker.py`
  - Sources : Fed RSS press_monetary, ECB press releases, BoE news, BoJ news + X API v2 ($100/mo Basic) pour live tweets @federalreserve, @ecb
  - Calibration impact : `cb_speaker_score = base_impact × dovishness_z` (LLM-extracted)
  - **NEW** `data/cb_speakers/speakers.yaml` (Powell/Lagarde/Bailey/Ueda + hawkishness baseline)
  - **NEW** `tests/test_cb_speaker_tracker.py`
- **Acceptance** :
  - Détection Powell discours en < 2 min après publication
  - 5 derniers discours Powell 2025 → dovishness extraite ±0.3 vs gold-standard
  - Wired dans `NewsAssessment.cb_speaker_active: bool`

#### P1b.2 — Blackout cb_speaker dans Confluence — **6 h**

- **Fichiers** :
  - **MOD** `src/intelligence/confluence_detector.py` (cb_speaker_blackout 30 min around)
  - **MOD** `src/backtest/news_replay.py` (backfill speakers 2019-2025 via FRASER / archives)
- **Acceptance** :
  - Replay XAU 2024 : cb_speaker_blackout fire sur ≥ 8 discours Powell

#### P1b.3 — Tests + métriques — **4 h**

- **NEW** `tests/test_cb_blackout.py`

---

### P2 — Event-driven backtest mode + COT/macro factors (Semaine 8-12, ~80 h)

#### P2.1 — Activer Pilier 1 event-driven avec surprise — **24 h**

- **Fichiers** :
  - **MOD** `src/strategies/event_driven_macro.py:14-17` (lire surprise depuis CSV v2)
  - **MOD** `EventStrategyConfig` : ajouter `min_surprise_abs_z: float = 1.0` (trade only if |z|≥1)
  - **NEW** `scripts/run_event_driven_backtest.py`
  - **NEW** `reports/backtest/event_driven_xau_with_surprise.md`
- **Acceptance** :
  - Backtest 2019-2025 sur 329 trades initiaux → reflow avec surprise gate
  - Si DSR > 0.5 et PBO < 0.5 → graduate to production
  - Sinon : report écrit avec findings + iter

#### P2.2 — COT integration (positionnement Commercials/Funds) — **20 h**

- **Fichiers** :
  - **MOD** `src/agents/news/sources/cot_adapter.py` (existing, à finaliser)
  - **NEW** `src/intelligence/features/cot_features.py` (`commercials_net_pct`, `funds_extreme_z`)
  - **MOD** `src/intelligence/confluence_detector.py` (composant `cot_positioning_score`)
  - **NEW** `tests/test_cot_features.py`
- **Acceptance** :
  - `commercials_net_pct(XAU)` calculé sur dernière publish CFTC (vendredi 15h30 ET)
  - Backtest XAU 2010-2025 : feature non-null sur 100 % des bars hors-FRI

#### P2.3 — FRED macro features → Confluence — **16 h**

- **Fichiers** :
  - **NEW** `src/intelligence/features/macro_fred.py` (DXY momentum, real_yields_5d_chg, vix_regime)
  - **MOD** `src/intelligence/confluence_detector.py` (composant `macro_regime_score`)
  - **MOD** `data/macro/` : refresh weekly via `scripts/fetch_fred_macro.py` (existing ?)
  - **NEW** `tests/test_macro_features.py`
- **Acceptance** :
  - 5 features macro extractibles sur tout l'historique 2010-2025
  - Backtest IC bootstrap : au moins 1 macro feature avec p<0.05 vs futur 4h-return

#### P2.4 — Validation finale + DSR/CPCV — **20 h**

- **Fichiers** :
  - **NEW** `scripts/cpcv_with_news_features.py` (CPCV 28 paths sur full feature stack)
  - **NEW** `reports/backtest/cpcv_news_macro_full.md`
- **Acceptance** : DSR, PBO, PF_lo, DM_p reportés. Décision GO/NO-GO publique.

---

## 5. Tests & validation

### 5.1 Couverture cible

| Domaine | Tests existants | Tests à ajouter | Cible coverage |
|---|---|---|---|
| `economic_calendar.py` | 0 | 12 (CSV parse, mtime cache, FF fallback, error handling) | 85 % |
| `news_analysis_agent.py` | partiel via `test_news_pipeline.py` | 8 (blackout matrix, surprise wiring, multi-currency) | 80 % |
| `surprise_scorer.py` | n/a (new) | 10 | 95 % |
| `trading_economics provider` | n/a (new) | 8 | 90 % |
| `news_replay.py` | 1-2 | 6 (CSV v2 enriched, surprise pass-through) | 90 % |
| `cb_speaker_tracker.py` | n/a (new) | 8 | 80 % |
| Pipeline E2E | 0 | 5 scénarios (NFP, FOMC, ECB, BoE, BoJ) | n/a |

### 5.2 Scénarios E2E critiques (golden tests)

1. **NFP 2024-01-05 13:30 UTC** — Actual 216k, Forecast 170k → surprise z≈+1.3 → BUY USD bias, XAU blackout 30 min before + 60 min after
2. **FOMC 2024-03-20 18:00 UTC** — Hawkish hold → cb_speaker_score positif après Powell Q&A
3. **ECB 2024-06-06 12:15 UTC** — First cut 25 bps → news_surprise<0 EUR, blackout EURUSD étendu
4. **CPI 2024-05-15 12:30 UTC** — Actual 3.4 % vs Forecast 3.4 % → surprise z≈0, no extra blackout
5. **Powell speech 2024-08-23 14:00 UTC (Jackson Hole)** — cb_speaker blackout XAU 30 min around

### 5.3 Calendar consistency check (startup gate)

- **Fichier** : `scripts/crosscheck_mt5_calendar.py:19` (déjà existe, à wire)
- **MOD** `src/intelligence/main.py` : appeler `crosscheck` au boot, fail-fast si > 5 % mismatch HIGH-impact ; warn si 1-5 %
- **Acceptance** : startup time < 30 s même avec crosscheck, alerte Slack/Telegram si fail

### 5.4 Surprise accuracy validation

- Backtest historique : recalculer surprise pour 50 NFP releases 2019-2025, comparer avec Bloomberg consensus (manuel) → MAE < 0.2 z-units
- Reporter dans `reports/backtest/surprise_validation.md`

---

## 6. Sécurité

### 6.1 Gestion des API keys

| Provider | Variable env | Stockage prod | Rotation |
|---|---|---|---|
| Trading Economics | `TE_API_KEY` | Railway secret + `.env.local` gitignored | 90 j |
| NewsAPI.ai | `NEWSAPI_AI_KEY` | idem | 90 j |
| X API v2 | `X_BEARER_TOKEN` | idem | 180 j |
| Anthropic | `ANTHROPIC_API_KEY` | idem | 90 j |
| FRED | `FRED_API_KEY` (free public, optionnel) | env | jamais |

- **Audit** : `git secrets --scan` pre-commit hook (déjà installé ?), `gitleaks` CI
- **Logs** : Aucune key dans logs (mask en `repr()`), reviewer toutes les `logger.info(*api_key*)`
- **Backup** : keys dupliquées dans 1Password / Bitwarden vault personnel

### 6.2 Licence compliance audit

- **NEW** `compliance/license_audit_calendar.md` (mémo : sources, CGU, usage commercial autorisé)
- **NEW** `compliance/license_audit_news.md` (idem news)
- Audit trimestriel par solo founder, validation avocat annuelle (eval_29 §6.4)
- Disclaimer attribution dans `/api/v1/about` : « Calendar data: Trading Economics; News: NewsAPI.ai; Macro: FRED. »

### 6.3 Rate limits & quotas

- Trading Economics Smart : **1 req/s, 10 000 req/jour** — calendar refresh toutes 15 min (96/jour), backfill ≤ 1 fois/sem
- NewsAPI.ai Basic : **5 req/s, 50 000/mo** — streaming WS ne consomme pas le quota REST
- FRED : **120 req/min** — refresh hebdomadaire OK
- **Implémentation** : `aiolimiter` ou `asyncio.Semaphore` par provider dans `providers/base_calendar_provider.py`
- **Alerte** : si 80 % quota atteint → log WARN + Slack

### 6.4 Input validation côté agent

- Calendar event names : sanitize via `src/api/security.py:sanitize_string()` (eval_10-15 #14)
- News headlines : strip HTML, limit 500 chars avant LLM
- WebSocket payloads : JSON schema validate (NewsArticle pydantic v2 strict)

---

## 7. Métriques opérationnelles

### 7.1 KPIs en continu (Prometheus + Grafana)

| Métrique | Cible | Alerte |
|---|---|---|
| `calendar_last_fetch_age_seconds` | < 1800 (30 min) | > 7200 (2 h) → 🔴 Slack |
| `calendar_events_known_24h` | ≥ 20 (US+EU) | < 5 → 🟠 |
| `news_stream_messages_per_minute` | ≥ 0.5 (24/7) | 0 pendant 15 min → 🔴 |
| `news_stream_ws_connected` | true | false > 5 min → 🟠 |
| `surprise_compute_latency_ms_p95` | < 50 | > 200 → 🟡 |
| `blackout_fired_per_day` | 3-15 (XAU) | > 30 → check matrix |
| `sentiment_llm_cost_usd_per_day` | < $2 | > $10 → revoir cache |
| `provider_circuit_open_count` | 0 | ≥ 1 → 🔴 |

### 7.2 KPIs business

- **Surprise correlation with realized vol** (rolling 30 j) — cible Pearson > 0.30 NFP/CPI
- **Backtest impact** : avant/après enrichment, PF, Sharpe, max DD reportés mensuellement
- **Customer-facing** : « Smart Sentinel a anticipé X événements macro ce mois » dans email mensuel
- **Coverage geographic** : 6 instruments × 4+ event-types blackout définis (matrice §P0.5)

### 7.3 Reporting hebdo

- Auto-generated `reports/news_macro_weekly_<date>.md` :
  - Top 5 events traités (surprise, blackout duration, impact réalisé)
  - Provider uptime
  - Coût providers cumulé semaine
  - Anomalies détectées (e.g., FF/TE divergence > 5 %)

---

## 8. Risques & mitigations

| Risque | Probabilité | Impact | Mitigation | Owner |
|---|---|---|---|---|
| TE outage > 4 h pendant NFP | Faible | 🔴 trade aveugle | Fallback automatique FRED + cache 24 h + alerte | P0.7 |
| NewsAPI.ai schema breaking change | Moyenne (annuel) | 🟠 sentiment KO 24 h | Schema versionned, contract test CI, fallback RSS | P1.1 |
| FF C&D (legal action après go-live payant) | Moyenne 🔴 | Bloque produit | Migration P0.1 + archive FF read-only `ALLOW_FF=0` prod | P0.1, P0.6 |
| Surprise computation bias (σ historique biaisé) | Moyenne | 🟠 fausses entrées | Rolling 24 mois min, refit trim. + validation §5.4 | P0.3 |
| CB speakers : faux positif (event « not Powell ») | Moyenne | 🟡 over-blackout | LLM filter `speakers.yaml` whitelist + confidence > 0.7 | P1b.1 |
| Anthropic LLM circuit breaker triggered | Faible | 🟡 sentiment dégradé | Fallback keyword 100 % autonomous | P1.3 |
| Schema drift FRED API | Très faible | 🟢 | API v1 stable depuis 2014, no-op | n/a |
| X API price hike (cf. 2023 Musk) | Moyenne | 🟡 cb_speakers v2 délai | Defer X à S2, RSS Fed/ECB couvrent 80 % | P1b.1 |
| Latence WS > 90 s (provider tier) | Faible (NewsAPI.ai SLA p95 < 60 s) | 🟡 edge dégradé | Métrique 7.1 + bascule provider | P1.6 |
| Coût LLM sentiment > $10/j | Moyenne (croissance) | 🟡 marge | Cache SemanticCache, downscale Haiku→template si > seuil | P1.3 |
| Schrems III invalide DPF (Anthropic US) | Faible 2026 | 🟠 | Bascule Mistral / Anthropic-EU si publié | eval_29 §10 |
| FF JSON `nfs.faireconomy.media` arrêt | Moyenne | 🔴 dev sans calendrier dev-mode | TE primary, archive FF en read-only sur snapshot | P0.1 |

---

## 9. Dépendances

### 9.1 Internes (modules)

- `src/intelligence/circuit_breaker.py` — wrap providers TE, NewsAPI.ai (déjà utilisé pour LLM/Telegram)
- `src/intelligence/semantic_cache.py` — cache sentiment LLM (TTL 6 h, key = hash headline)
- `src/intelligence/confluence_detector.py` — réception features news_surprise, cb_speaker_score, cot_positioning, macro_regime
- `src/intelligence/insight_v2/contract.py` — extension schema Pydantic V2
- `src/agents/events.py` — `NewsEvent`, `MacroEvent`, `CBSpeakerEvent` types EventBus
- `src/backtest/news_replay.py` — replay enrichi pour CPCV
- `src/api/routes/health.py` — endpoints `/health/calendar`, `/health/news_stream`
- `src/strategies/event_driven_macro.py` — Pilier 1, consume surprise_z

### 9.2 Externes (providers)

- **Trading Economics** (account + Smart plan + DPA)
- **NewsAPI.ai** (account + Basic plan + DPA)
- **FRED API** (free key)
- **ECB SDW API** (free, public domain)
- **BLS API** (free key)
- **X API v2 Basic** ($100/mo, S2)
- **Anthropic API** (déjà en place)

### 9.3 Compliance & légal

- DPA Trading Economics signé (cf. P0.6)
- DPA NewsAPI.ai signé
- Mémo avocat fintech sur licences data (cf. eval_29 §6.4, J22-J30)
- Bandeau attribution `/about` (déjà prévu)
- CGU §X « Sources de données » : citer providers + disclaimer

### 9.4 Infrastructure

- SQLite `data/calendar.db` + `data/news.db` (déjà en place pour signal_store)
- Stockage S3 cold news (90 j+) optionnel S3
- Prometheus exporter pour métriques §7.1 (eval_16 — `metrics` endpoint à compléter)
- Grafana dashboard (template à créer)
- Slack webhook pour alertes critiques

### 9.5 Autres catégories du sprint commercialization

- **Cat. 8 Data Providers** : licence Dukascopy → Polygon.io affecte budget global, cf. eval_29 §5.1
- **Cat. 18 Backtest** : enrichment news débloque IC bootstrap propre (eval_18 — 2/10 → cible 6/10)
- **Cat. 29 Compliance** : geo-block, disclaimer (sprint W1+W2+W3 livré 2026-04-29)
- **Cat. 19 Risk** : blackout = kill-switch déclenché automatiquement
- **Cat. 1 Architecture** : provider abstraction = pattern à généraliser pour vol/OHLCV

---

## 10. Estimation totale & timeline

### 10.1 Récapitulatif heures

| Phase | Tâches | Heures |
|---|---|---|
| **P0 — Bloquant commercialisation** | P0.1 → P0.7 | **70 h** |
| **P1 — Live news + sentiment LLM** | P1.1 → P1.6 | **60 h** |
| **P1.bis — CB speakers** | P1b.1 → P1b.3 | **24 h** |
| **P2 — Event-driven backtest + COT + macro** | P2.1 → P2.4 | **80 h** |
| **Total** | | **234 h** |

### 10.2 Timeline solo founder (8-9 h/sem, hypothèse eval_28)

- **Semaine 1-3** (~70 h cumulées, full focus) — P0 complet, **go-live conditionnel possible**
- **Semaine 4-6** (~60 h, parallèle marketing) — P1 streaming + sentiment LLM
- **Semaine 7** (~24 h, focused sprint) — P1.bis CB speakers
- **Semaine 8-12** (~80 h, full dev) — P2 event-driven backtest + macro features

**Total : 12 semaines** avec P0 livré à S3 (compatible go-live FR-first beta payante).

### 10.3 Coût providers récurrent

| Phase | Mensuel | Annuel |
|---|---|---|
| P0 (TE Smart) | **$79** | $948 |
| P0+P1 (TE + NewsAPI.ai) | **$279** | $3 348 |
| P0+P1+P1.bis (+ X Basic) | **$379** | $4 548 |
| P2 (idem, FRED/COT free) | **$379** | $4 548 |
| **Marge tier ANALYST $29** avec coût $379/mo | breakeven à 14 abonnés |

### 10.4 Coût one-shot

- Trading Economics setup (intégration + tests) : intégré P0 (18 h dev)
- DPA légal (TE + NewsAPI.ai + X) : ~600 € avocat
- Backfill historique 2019-2025 : intégré P0.2 (12 h dev)
- **Total one-shot : ~600 € externe + 234 h dev solo** (valorisé à ~150 €/h = 35 k€ cost-of-capital)

### 10.5 Décision matrix go/no-go par phase

| Gate | Critère | Si KO |
|---|---|---|
| Après P0 | TE wired, surprise computed, blackout matrix défini, 8 tests verts, DPA signé | Reporter go-live payant, continuer testing-mode |
| Après P1 | NewsAPI.ai streaming, latence p95 < 90 s, sentiment LLM cost < $2/j | Bascule sentiment keyword en prod, NewsAPI.ai en optionnel |
| Après P1.bis | CB speaker fire ≥ 5/mois | Defer P1.bis, focus P2 |
| Après P2 | DSR > 0.5 event-driven OR ≥ 1 macro feature p < 0.05 | Reporter au verdict A1 (cf. roadmap 2026-2027) |

---

## Synthèse — 5 lignes pour caller

- **Livrable** : `C:\MyPythonProjects\TradingBOT_Agentic\reports\commercialization_sprint\15_news_macro_pipeline.md`
- **Top 3 P0 bloquants commercialisation** : (1) **P0.1 Migration ForexFactory → Trading Economics** (licence commerciale, 18 h) ; (2) **P0.2 Backfill Actual/Forecast/Previous CSV 7 ans** (débloque surprise, 12 h) ; (3) **P0.5 Blackout matrix YAML per instrument** (6 presets, 8 h)
- **Heures P0** : 70 h (3 semaines solo) — **P0+P1+P1.bis+P2 = 234 h** sur 12 semaines
- **Coût providers récurrent cible v1** : **$279/mo** (TE Smart $79 + NewsAPI.ai Basic $200), upgrade $379/mo avec X Basic — breakeven à 14 abonnés ANALYST $29
- **Verdict global** : note actuelle **2.4/10** → cible **6.2/10 à J+30** (P0 done) → **8.5/10 à J+90** (P0+P1+P1.bis) ; **risque #1 = CGU FF**, **risque #2 = Actual/Forecast vides plafonnent Confluence à 70/100 en replay**
