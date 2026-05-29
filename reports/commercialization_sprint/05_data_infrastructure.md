# Plan de Commercialisation — Catégorie 5 : Data Infrastructure & Feeds

> **Objectif** : passer d'un assemblage CSV ad-hoc (Dukascopy gratuit, ForexFactory scraping, FRED static dumps) à un pipeline data production-grade, licencié, fiable et live, capable de soutenir la promesse commerciale « real-time market intelligence ».
>
> **Branch** : `institutional-overhaul` · **Date** : 2026-05-21 · **Périmètre** : `src/intelligence/data_providers.py`, `src/intelligence/data_quality.py`, `data/`, `scripts/download_*.py`, `scripts/fetch_forexfactory_live.py`, pipeline news (`src/agents/news/economic_calendar.py`).
>
> **Verdict actuel (eval_08, 3.5 / 10)** : ❌ NO-GO commercial. Quatre blocages : feed XAU à 63 % coverage, 5/6 presets sans CSV, licence Dukascopy zone grise, aucun pipeline live.

---

## 1. État actuel (Audit)

### 1.1 Inventaire data physique

| Fichier `data/` | Lignes | Plage | Coverage* | Statut |
|---|---:|---|---:|---|
| `XAU_15MIN_2019_2024.csv` | 141 525 | 2019-01 → 2024-12 | 97.6 % | ✅ feed historique propre |
| `XAU_15MIN_2019_2025.csv` | 106 644 | 2019-01 → 2025-?? | **63 %** | ❌ feed corrompu (source distincte) |
| `XAU_15MIN_2019_2026.csv` | 172 875 | 2019-01 → 2026-?? | à vérifier | ⚠️ produit par `scripts/merge_xau_2019_2026.py` |
| `XAU_15MIN_2025_2026_dukascopy.csv` | ? | 2025-01 → 2026-?? | à vérifier | ⚠️ extension Dukascopy (licence flag) |
| `EURUSD_15MIN_2019_2025.csv` | 174 507 | 2019-01 → 2025-?? | 99.6 %† | ✅ feed FX EUR/USD |
| `economic_calendar_2019_2025.csv` | ? | 2019 → 2025 | ForexFactory archive | ⚠️ usage commercial déguisé |
| `economic_calendar_HIGH_IMPACT_2019_2025.csv` | ? | filtre HIGH | dérivé | ⚠️ idem |
| `research/cot_gold.csv` | CFTC weekly | 2019 → 2025 | public domain | ✅ |
| `research/fred_DGS10.csv` + 4 autres FRED | static dumps | 2019 → 2025 | FRED ToS | ✅ |
| `research/a1_matrix_2019_2026.parquet` | ML matrix | derived | local artefact | ✅ |
| `macro/cot_gold.csv` + 5 FRED | doublons `research/` | — | — | ⚠️ duplication |

\* d'après `reports/eval_08_data_providers.md` §0 et `scripts/audit_xau_coverage.py:60-89` (cible Mon-Fri 07:00-21:00 UTC).
† d'après `memory/eval_05_09_refresh_2026_04_29.md` (coverage EUR ajoutée).

**Symboles configurés (presets) sans CSV** : BTCUSD, US500, GBPUSD, USDJPY (4/6 presets sont du vapor-ware).

### 1.2 Inventaire code

| Fichier | LOC | Rôle | Note audit |
|---|---:|---|---|
| `src/intelligence/data_providers.py` | 375 | `DataProvider` ABC + `CSVDataProvider` + `MT5DataProvider` | 10 bugs identifiés eval_08 §2 |
| `src/intelligence/data_quality.py` | 191 | `validate_ohlcv()` + `ValidationReport` + `DataQualityError` | 9 checks structurels, pas de coverage gate fail-fast |
| `scripts/download_dukascopy_xau.py` | 234 | Téléchargement tick → OHLCV LZMA | gratuit, licence personnelle seulement |
| `scripts/fetch_forexfactory_live.py` | 185 | Calendrier FF JSON `nfs.faireconomy.media` | usage commercial déguisé |
| `scripts/audit_data_quality.py` | 106 | Diagnostic ad-hoc | non-CI |
| `scripts/audit_xau_coverage.py` | 158 | Coverage by year | non-CI |
| `scripts/download_economic_calendar.py` | ? | (déprécié, scraper HTML cassé) | à supprimer |
| `scripts/download_real_economic_data.py` | ? | FRED + manual | one-shot, pas planifié |
| `scripts/download_xau_data.py` | ? | helper legacy | à dépréquer |
| `scripts/crosscheck_mt5_calendar.py` | ? | Reconcile MT5 vs FF | one-shot diagnostic |
| `scripts/export_mt5_history.py` | ? | Export OHLCV broker | local Windows uniquement |

### 1.3 Bugs critiques (eval_08 §2, copie résumée)

| # | Localisation | Impact |
|---|---|---|
| 1 | `data_providers.py:85-89` — cache RAM full CSV, jamais purgé | mémoire balloon en multi-asset |
| 2 | `data_providers.py:78-90` — pas de tail-incremental load | re-parse 20 MB/min en live-sim |
| 3 | `data_providers.py:92-133` — **aucune validation feed source au boot** | scanner consomme XAU à 63 % silently |
| 4 | `data_providers.py:295-316` — `MT5DataProvider` pas thread-safe | race `_validated_symbols` / `_volume_source` |
| 5 | `data_providers.py` — aucun fallback fournisseur | broker down = scanner down |
| 6 | `data_providers.py:92-133` — pas de schema versioning | shift TZ Dukascopy silent |
| 7 | `data_providers.py:175-176` — `MAX_RECONNECT=3 × DELAY=2s` = 6 s bloquant | manque 1 bar M15 |
| 8 | `data_providers.py:318-369` — aucun log freshness `last_bar` vs `now` | stale feed aveugle |
| 9 | `data_quality.py:182-189` — `strict=True` lève → crash scanner | pas de DLQ |
| 10 | `tests/` — aucun test d'intégration feed réel | regression silencieuse au format CSV |

### 1.4 Licensing — état des lieux

| Source actuelle | Usage | Licence officielle | Risque commercial |
|---|---|---|---|
| **Dukascopy tick history** (`scripts/download_dukascopy_xau.py:46-48`) | Backtest XAU 2019-2026 + extension live | Personal-use only (clause TPS) | ⚠️ Zone grise revente dérivée / signaux. Cf. `memory/eval_29_compliance_findings.md`. |
| **ForexFactory JSON** (`scripts/fetch_forexfactory_live.py:47-50`) | Calendrier macro | TOS interdit usage commercial sans accord | ❌ Usage commercial déguisé documenté |
| **MT5 broker history** (`data_providers.py:147+`) | Live + backtest | Liée au compte trader, varies by broker | ⚠️ Dépendant du broker, pas de SLA |
| **FRED St. Louis** (`data/research/fred_*.csv`) | Macro yields, VIX | FRED ToS — usage non-commercial OK, redistribution restreinte | ✅ OK pour calculs internes |
| **CFTC COT** (`data/research/cot_gold.csv`) | Positionnement | Public domain US Federal | ✅ libre |

**Conclusion compliance** : **deux sources sur quatre en infraction directe ou zone grise** pour une activité monétisée. Bloque Stripe + AMF/MiFID (cf. eval_29). Voir aussi `data/rag/sources_manifest.yaml` qui mentionne « fair use commentary » pour les sources éducatives mais reste muet sur Dukascopy/FF.

### 1.5 Pipeline live — état

- **Aucun WebSocket**. `grep -ri "websocket\|asyncio" src/intelligence/` → 0 hits.
- **Polling 60 s** : `SentinelScanner._poll_interval` (default 60.0 s, `sentinel_scanner.py:92`).
- Conséquence : SLA marketing « signal en 30 s » impossible. Bar M15 close → scan = jusqu'à 60 s + 1-2 s pipeline.
- **MT5DataProvider est synchrone** (`get_ohlcv` bloque le thread scanner, `data_providers.py:318+`).
- Aucune file de messages (Redis, Kafka, SQS). Le scanner pull-only.

### 1.6 Quality monitoring — état

- `validate_ohlcv()` instancié dans le scanner ? **NON visible**. Aucun appel `validate_ohlcv` ni `DataQualityError` dans `sentinel_scanner.py`. La validation est définie mais jamais branchée. Régression latente.
- Aucune métrique exportée vers `/metrics` ou Prometheus (cf. eval_16 §1 : `/metrics` payload vide en prod).
- Aucun alerting sur gap > 30 min en session active (la métrique existe `scripts/audit_xau_coverage.py:93-97` mais reste batch).
- Aucun journal applicatif `data freshness` (bug #8 eval_08).

---

## 2. Vision cible

### 2.1 Properties attendues post-Sprint

1. **Couverture ≥ 98 %** sur XAU + EURUSD M15 2019-aujourd'hui (cible MVP — `audit_xau_coverage.py:103`).
2. **Licences propres et documentées** : Tiingo (FX) + Polygon.io (FX/Indices) ou Databento (institutionnel) contrats signés.
3. **Pipeline live WebSocket** : tick → bar close → publish < **5 s** P95 (vs 60 s aujourd'hui).
4. **Multi-source redundancy** : `CompositeDataProvider` enchaîne MT5 → Tiingo → CSV → DLQ.
5. **Schema validation et versioning** : chaque CSV/Parquet a un `.metadata.json` (source, license, fetched_at, schema_version, coverage_pct).
6. **Quality gate au boot** (P0) : `assert_coverage(symbol, tf) ≥ 0.95` ou refus de démarrage.
7. **Monitoring temps réel** : `data_freshness_seconds`, `data_gap_count_1h`, `provider_failover_total` exposés Prometheus.
8. **Data lake historical** : Parquet partitionné `s3://sentinel-data/{symbol}/{year}/{month}.parquet` (P2).
9. **Point-in-time queries** : reconstituer la donnée telle qu'elle était à T (révisions FRED, COT, prix corrigés) — exigence backtest reproductible (cf. eval_23 train/serve skew).
10. **API keys management** : provider secrets en `.env` + rotation 90 j + `python-decouple` ou Doppler/SOPS pour la prod.

### 2.2 Architecture cible (texte)

```
                      ┌─────────────────────────────────────┐
                      │      ProviderRegistry (toml conf)   │
                      │  primary: Polygon (FX)              │
                      │  fallback1: Tiingo                  │
                      │  fallback2: MT5 (Windows-only)      │
                      │  fallback3: CSV (cold-start)        │
                      └────────────────┬────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                              ▼
   ┌──────────────────────┐                       ┌──────────────────────┐
   │  Live WebSocket pool │                       │  REST historical     │
   │  asyncio.Queue       │                       │  (backfill / gaps)   │
   │  tick → bar bucket   │                       │  Parquet writer      │
   └──────────┬───────────┘                       └──────────┬───────────┘
              │ on_bar_close                                  │
              ▼                                              │
   ┌──────────────────────┐                                  │
   │  validate_ohlcv()    │── DLQ ──► alerts/quarantine      │
   │  + freshness guard   │                                  │
   └──────────┬───────────┘                                  │
              ▼                                              │
   ┌──────────────────────┐                                  │
   │  In-memory ring buf  │  ← read-through fallback to Parquet
   │  (last N bars/symbol)│                                  │
   └──────────┬───────────┘                                  │
              ▼                                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │           SentinelScanner / Insight pipeline           │
   └─────────────────────────────────────────────────────────┘
```

Pour news/macro : pipeline parallèle Trading Economics ou Econoday (commercial) + FRED API + CFTC FTP, avec contrat de licence signé.

---

## 3. Gap analysis (état → cible)

| Dimension | Aujourd'hui | Cible | Gap |
|---|---|---|---|
| Coverage XAU | 63 % (live feed actif) | ≥ 98 % | ré-télécharger ou souscrire feed propre |
| Multi-asset | 1/6 (XAU réel, EUR partiel) | 4/6 (XAU+EUR+US500+BTC) | 3 nouveaux feeds licenciés |
| Licence data | Dukascopy zone grise + FF infraction | Contrats Tiingo + Polygon + Econoday | $0 → $109-329/mo |
| Latence | 60 s polling | < 5 s WebSocket P95 | refonte scanner async |
| Fallback | aucun | chaîne 3 providers | nouveau `CompositeDataProvider` |
| Quality gate boot | non | abort si cov < 95 % | 10 LOC dans `main.py` |
| Monitoring | absent | Prometheus + alertes | gauges + scrape config |
| Schema versioning | absent | `.metadata.json` par CSV | gabarit + migration |
| Storage scaling | 100 % CSV local | Parquet partitionné + S3 P2 | DuckDB + boto3 |
| Tests d'intégration | mocks uniquement | tests live (sandbox keys) | CI matrix par provider |

---

## 4. Plan d'exécution

> Les tâches sont chiffrées en heures-dev solo (Loukmane) et en coût provider mensuel marginal. Acceptance criteria = condition de done. Dépendances explicites (catégorie 4 compliance, catégorie 7 backtest, catégorie 8 ML).

### 4.1 P0 — Coverage gate fail-fast et nettoyage du feed corrompu

**T0.1 — Boot-time coverage gate** (P0, blocker)
- Fichier : `src/intelligence/main.py` (insertion dans `_calibrate_system` / build_system), nouvelle helper dans `src/intelligence/data_quality.py`.
- Implémentation : `assert_coverage(provider, symbol, timeframe, min_pct=0.95)`, abort `SystemExit` si fail.
- Effort : **3 h** (code + 4 tests unitaires).
- Acceptance : démarrage refuse avec `XAU_15MIN_2019_2025.csv` (63 %), accepte avec `XAU_15MIN_2019_2024.csv`. Log structuré + exit code 78 (config error).
- Dépendances : aucune.

**T0.2 — Purge fichier corrompu + standardisation nommage** (P0)
- Action : retirer `data/XAU_15MIN_2019_2025.csv` du flux par défaut, renommer en `.deprecated.csv`, créer `data/README.md` documentant chaque CSV (source, license, coverage, fetched_at).
- Symboliser un canonical `XAU_15MIN.csv` pointant vers `XAU_15MIN_2019_2026.csv` (validé ≥ 95 %).
- Effort : **2 h** (audit + rename + README).
- Acceptance : `scripts/audit_xau_coverage.py` retourne coverage ≥ 95 % sur le canonical. Test E2E `tests/test_data_canonical_csv.py`.
- Dépendances : T0.1.

**T0.3 — Validation systématique branchée dans SentinelScanner** (P0)
- Bug eval_08 #9 : `validate_ohlcv()` n'est jamais appelé. Câbler dans `sentinel_scanner._scan_once` avec `strict=False` + DLQ.
- Fichier : `src/intelligence/sentinel_scanner.py:_scan_once` + nouveau module `src/intelligence/data_dlq.py` (in-memory deque + log).
- Effort : **4 h**.
- Acceptance : un bar corrompu (H<L injecté) est skip avec log WARNING, le scanner continue. 3 tests.
- Dépendances : T0.1.

**T0.4 — Freshness guard sur chaque pull** (P0)
- Bug eval_08 #8. Logger `data freshness: last_bar=<ts>, lag=<s>` après chaque `get_ohlcv`. Alerter si lag > 2 × timeframe.
- Fichier : `src/intelligence/data_providers.py` (hook dans wrapper) ou décorateur dans `sentinel_scanner.py`.
- Effort : **2 h**.
- Acceptance : log JSON systématique + métrique Prometheus `data_freshness_seconds{symbol,tf}`.

**Total P0 lot coverage** : **11 h** · coût provider : **$0**.

---

### 4.2 P0 — Licensing audit et remediation

**T1.1 — Audit légal écrit des feeds actuels** (P0, blocker légal)
- Action : envoyer email à `commercial@dukascopy.com` (clarification clause TPS pour service B2C dérivé) et `faireconomy.media` (FF JSON usage). Conserver réponse en `compliance/data_license_correspondence.md`.
- Effort : **3 h** (rédaction + suivi).
- Acceptance : réponse écrite archivée OU mention « no reply 30 days » → assumer infraction, migrer.
- Dépendances : se coordonner avec catégorie 4 (Compliance) — cf. eval_29.

**T1.2 — Souscription Tiingo (FX historique + EOD)** (P0)
- Tier : **IEX + Tiingo FX Starter** $30/mo (couvre EUR/USD, USD/JPY, GBP/USD historique 1996+, minute-level).
- Action : créer compte business, signer DPA, stocker API key dans Doppler/vault.
- Nouveau module : `src/intelligence/data/tiingo_provider.py` impl `DataProvider` avec `requests` + retry exponential. Endpoint `https://api.tiingo.com/tiingo/fx/<ticker>/prices`.
- Effort : **8 h** (souscription + provider + 10 tests dont 3 live sandbox).
- Acceptance : `TiingoProvider().get_ohlcv("EURUSD", "M15", 200)` retourne DataFrame valide. Test backfill 7 jours.
- Coût : **$30/mo**.
- Dépendances : T1.1.

**T1.3 — Souscription Polygon.io Currencies Starter** (P0, requis pour Stripe launch)
- Tier : **Currencies Starter** $79/mo (FX REST + WebSocket realtime, history depuis 2003).
- Couvre : EUR/USD, GBP/USD, USD/JPY, XAU/USD synthétique (via deux jambes USD/USD pas idéal — fallback Tiingo pour XAU).
- Nouveau module : `src/intelligence/data/polygon_provider.py` impl `DataProvider` avec :
  - REST via `requests`
  - WebSocket via `websockets` + `asyncio`
- Effort : **20 h** (souscription + provider sync + async WebSocket + 15 tests).
- Acceptance : `PolygonProvider().stream(symbols=["EURUSD"], on_bar=callback)` reçoit bar close M15 en < 5 s P95. 24 h de soak test.
- Coût : **$79/mo**.
- Dépendances : T1.1.

**T1.4 — Fournisseur dédié XAU/USD spot** (P0)
- Polygon ne couvre pas XAU spot proprement (futures `GC` oui, spot via FX cross suboptimal).
- Options :
  - **MetalsAPI Pro** : $99/mo, spot XAU/USD historique + live REST.
  - **Tradermade FX & Metals** : $50-200/mo selon volume.
  - **Twelve Data Pro** : $79/mo, inclut XAU/USD spot tick-level.
- Recommandation : **Twelve Data Pro** $79/mo (un seul provider couvre FX + Metals + Crypto + Indices, simplifie ops).
- Nouveau module : `src/intelligence/data/twelvedata_provider.py`.
- Effort : **12 h** (souscription + provider + tests).
- Acceptance : XAU M15 historique 2019-2026 ≥ 98 % coverage + WebSocket spot tick.
- Coût : **$79/mo** (peut remplacer Polygon si on choisit single-vendor TwelveData).
- Dépendances : T1.1.

**T1.5 — Calendrier macro légal (remplace ForexFactory)** (P0)
- Options :
  - **Trading Economics** : $99/mo Subscription Plan API (commercial OK).
  - **Econoday** : enterprise pricing $300-1000/mo (inadapté solo).
  - **Investing.com API** : $79/mo via RapidAPI Reseller.
  - **MarketAux** : $19/mo économique.
  - **DIY FRED** : Calendrier release dates seulement (US uniquement, gratuit FRED ToS OK).
- Recommandation : **Trading Economics $99/mo** (couverture mondiale + actual/forecast/previous historique).
- Fichier : `src/agents/news/te_calendar_provider.py` remplace `EconomicCalendarFetcher` scraping HTML.
- Effort : **10 h** (souscription + provider + migration + 8 tests).
- Acceptance : `te_provider.fetch(window="next_7_days")` retourne CSV format `Date,Currency,Event,Impact,Actual,Forecast,Previous` compatible avec `economic_calendar_2019_2025.csv` schema.
- Coût : **$99/mo**.
- Dépendances : catégorie 4 (compliance review).

**T1.6 — Décommissionnement ForexFactory + Dukascopy de la prod** (P0)
- Action : remplacer `scripts/fetch_forexfactory_live.py` par cron Trading Economics ; conserver `scripts/download_dukascopy_xau.py` **uniquement pour usage R&D personnel** + warning banner.
- Documenter dans `data/README.md` quel CSV provient de quelle source et sa licence.
- Effort : **3 h**.
- Acceptance : `grep -r forexfactory.com src/` → 0 hits hors scripts/legacy/ marqué `# DEPRECATED — PERSONAL USE ONLY`.

**Total P0 lot licensing** : **56 h** · coût provider : **$30 (Tiingo) + $79 (TwelveData) + $99 (TE) = $208/mo**. (Sans Polygon doublon : si TwelveData couvre tout, Polygon optionnel P1.)

---

### 4.3 P0 — Live data pipeline (WebSocket vs polling)

**T2.1 — Refactor `DataProvider` interface async-first** (P0)
- Étendre l'ABC `DataProvider` (`data_providers.py:22-48`) avec :
  - `async def stream(symbols, timeframe, on_bar)` (signal callback)
  - `async def fetch_history(symbol, tf, start, end)` (backfill)
  - Conserver `get_ohlcv()` synchrone pour backward compat (wrapper sur asyncio.run en context test).
- Effort : **6 h** (design + impl + tests refacto existant).
- Acceptance : `CSVDataProvider` toujours fonctionnel via shim sync. 0 régression sur 1366 tests existants.

**T2.2 — `CompositeDataProvider` avec failover chain** (P0)
- Nouveau fichier : `src/intelligence/data/composite_provider.py`.
- Pattern : liste ordonnée de providers, premier qui répond OK gagne. Sur exception → log + fallback. Sur fallback persistant (3× consécutifs) → circuit breaker (réutiliser `src/intelligence/circuit_breaker.py`).
- Métriques : `provider_failover_total{from,to}`, `provider_latency_seconds{provider}` histogram.
- Effort : **10 h** (impl + 12 tests dont chaos test simulate provider down).
- Acceptance : kill primary → fallback responds en < 2 s ; 0 perte de bar sur soak 1 h.
- Dépendances : T2.1, T1.2, T1.3 ou T1.4.

**T2.3 — Tick aggregator → bar close** (P0)
- Nouveau fichier : `src/intelligence/data/tick_aggregator.py`.
- Bucketize ticks WebSocket en bars (M1, M5, M15, H1) avec lookforward de close (`floor(ts, freq)`).
- État interne thread-safe (`asyncio.Lock` ou `threading.RLock` selon mode).
- Effort : **8 h** + 10 tests (gap, late tick, DST boundary).
- Acceptance : 24 h de stream WebSocket simulé → bars identiques à `pandas.resample` d'un dump ticks de référence.

**T2.4 — Bar queue (in-memory dedup + replay-safe)** (P0)
- Nouveau fichier : `src/intelligence/data/bar_queue.py`.
- `asyncio.Queue` + dedup par `(symbol, timeframe, bar_ts)`. Persistance SQLite légère pour replay au crash (table `bar_queue_wal`).
- Effort : **6 h** + 6 tests.
- Acceptance : kill -9 du process → restart consomme la queue WAL sans perte. Dedup à 100 %.
- Dépendances : T2.3.

**T2.5 — Scanner async refactor** (P0)
- Refactor `SentinelScanner._run_loop` (`sentinel_scanner.py:273+`) : passer de `time.sleep(60)` polling à consommation de `BarQueue.get()` (push-driven).
- Conserver mode polling pour `CSVDataProvider` (mode replay/backtest).
- Effort : **14 h** (refacto + 8 tests régression + 2 tests intégration WebSocket).
- Acceptance : latence bar_close → signal publish P95 < 5 s sur 100 bars consécutifs. Mode replay (CSV) inchangé.
- Dépendances : T2.4.

**Total P0 pipeline live** : **44 h** · coût : déjà couvert par souscription provider WebSocket (T1.3 ou T1.4).

---

### 4.4 P0 — Data quality monitoring continu

**T3.1 — Métriques Prometheus pour data layer** (P0)
- Nouveau fichier : `src/intelligence/data/metrics.py`.
- Gauges/counters : `data_freshness_seconds`, `data_gap_count_total`, `data_validation_errors_total`, `data_validation_warnings_total`, `provider_failover_total`, `provider_latency_seconds`, `bar_queue_depth`, `data_coverage_pct` (mis à jour par job batch).
- Effort : **5 h** (impl + intégration dans 4 modules existants).
- Acceptance : `curl :8000/metrics` retourne les 8 séries. Test smoke.
- Dépendances : catégorie 9 (Observability) pour stack Grafana.

**T3.2 — Alertes (PagerDuty/Telegram ops)** (P0)
- Règles :
  - `data_freshness_seconds > 600` → alerte critique
  - `data_validation_errors_total[5m] > 5` → alerte
  - `data_coverage_pct < 0.95` → alerte hebdo
  - `provider_failover_total[1h] > 10` → alerte
- Fichier : `infrastructure/alertmanager/data_alerts.yml`.
- Effort : **3 h**.
- Acceptance : alerte de test reçue sur canal Telegram ops + e-mail.
- Dépendances : T3.1.

**T3.3 — Batch coverage audit hebdomadaire** (P0)
- Cron `scripts/audit_xau_coverage.py` → écrit rapport `reports/data_coverage/{YYYY-WW}.md` + push métrique `data_coverage_pct` à Prometheus pushgateway.
- Effort : **2 h** (cron + push).
- Acceptance : rapport généré chaque lundi 06:00 UTC.

**T3.4 — Schema validation + versioning** (P0)
- Nouveau fichier : `src/intelligence/data/schema.py` + `data/_schemas/v1_ohlcv.json` (JSON Schema).
- Chaque CSV writer génère un `.metadata.json` à côté : `{schema_version, source, license, fetched_at, coverage_pct, sha256}`.
- Loader compare schema avant ingestion. Mismatch → abort.
- Effort : **6 h** (schema + writer + loader + 8 tests + migration des CSV existants).
- Acceptance : `data/XAU_15MIN.csv.metadata.json` présent et conforme.

**Total P0 lot QA monitoring** : **16 h** · coût : **$0** (Prometheus + Grafana free tier).

---

### 4.5 P1 — Multi-source redundancy renforcée

**T4.1 — Secondary fallback FRED + EODHD** (P1)
- Ajouter `EODHistoricalDataProvider` ($19.99/mo) en deuxième fallback pour cas où Tiingo + Polygon down.
- Effort : **8 h**.
- Acceptance : test chaos 3 providers down → 4e prend le relais.
- Coût : **+$20/mo**.

**T4.2 — Cross-validation deux providers** (P1)
- Job batch quotidien compare last 7 days entre Tiingo et TwelveData (ou Polygon). Écart > 0.05 % sur close → alerte.
- Effort : **5 h**.
- Acceptance : test deliberate corruption → alerte déclenchée.

**T4.3 — MT5 broker comme fallback live (Windows VM)** (P1)
- Conserver `MT5DataProvider` mais le placer en dernier dans le composite (broker-dependent).
- Fixer bug #4 (thread-safety) et bug #7 (reconnect async).
- Effort : **6 h**.
- Acceptance : 3 tests concurrence + 1 test reconnect non-bloquant.

**Total P1 redundancy** : **19 h** · coût : **+$20/mo**.

---

### 4.6 P2 — Historical data lake & point-in-time queries

**T5.1 — Parquet partitionné par symbole/année** (P2)
- Migration des CSV historiques vers `data/parquet/{symbol}/{year}/{month}.parquet` (pyarrow / DuckDB).
- Compression Snappy, partitionnement par `year, month`.
- Effort : **8 h** (migration script + reader + tests).
- Acceptance : taille disque ÷ 4 ; lecture 6 ans < 200 ms via DuckDB.

**T5.2 — Object storage S3 (Backblaze B2)** (P2)
- Sync Parquet vers Backblaze B2 ($0.005/GB/mo). À 5 GB de données → $0.025/mo, négligeable.
- Effort : **4 h**.
- Coût : **+$0.10/mo** (5-20 GB).

**T5.3 — Point-in-time queries** (P2)
- Pour reconstituer données telles que connues à T (révisions FRED, COT, prix corrigés).
- Implémentation : chaque ligne CSV/Parquet a `fetched_at` ; query API `as_of=2024-06-15T12:00:00Z`.
- Effort : **12 h** (schema migration + DuckDB views + 10 tests).
- Acceptance : backtest reproduit chiffre à chiffre un run d'il y a 6 mois.
- Dépendances : critique pour eval_23 train/serve skew + eval_18 backtest credibility.

**T5.4 — Data versioning DVC ou LakeFS** (P2)
- DVC ($0, git-like) ou LakeFS auto-hosted.
- Effort : **6 h**.
- Acceptance : `dvc pull` rétablit n'importe quel snapshot historique.

**Total P2 data lake** : **30 h** · coût : **+$1-5/mo** (B2 + DVC remote).

---

### 4.7 Récapitulatif tâches

| Lot | # tâches | Heures | $/mo additionnel |
|---|---:|---:|---:|
| P0 Coverage gate & cleanup | 4 | 11 | $0 |
| P0 Licensing migration | 6 | 56 | $208 |
| P0 Pipeline live | 5 | 44 | $0 (inclus) |
| P0 QA monitoring | 4 | 16 | $0 |
| **Sous-total P0** | **19** | **127** | **$208/mo** |
| P1 Redundancy | 3 | 19 | $20 |
| P2 Data lake | 4 | 30 | $5 |
| **Total** | **26** | **176** | **$233/mo** |

---

## 5. Tests & validation

### 5.1 Tests unitaires (sans réseau)

- `tests/test_data_providers_csv.py` : extends existing, ajoute coverage gate.
- `tests/test_data_providers_composite.py` : 12 tests failover, circuit.
- `tests/test_tick_aggregator.py` : 10 tests bucketize, DST, gap.
- `tests/test_bar_queue.py` : 6 tests dedup, replay WAL.
- `tests/test_data_schema.py` : 8 tests metadata.json + JSON Schema.
- `tests/test_data_quality_strict.py` : 5 tests `validate_ohlcv` + DLQ.
- `tests/test_coverage_gate.py` : 4 tests fail-fast au boot.

### 5.2 Tests intégration (sandbox keys)

- `tests/integration/test_tiingo_live.py` (skip si `TIINGO_API_KEY` absent) : 5 tests REST.
- `tests/integration/test_twelvedata_live.py` : 7 tests REST + WS.
- `tests/integration/test_polygon_ws.py` : 4 tests WebSocket smoke.
- `tests/integration/test_trading_economics.py` : 3 tests calendrier.

### 5.3 Tests de charge

- `tests/load/test_websocket_soak_1h.py` : 1 h de stream → ≥ 240 bars M15 reçus, latence P95 < 5 s.
- `tests/load/test_composite_failover_chaos.py` : kill primary à T+5 min, vérifier 0 perte de bar.

### 5.4 Tests de schéma

- `tests/test_csv_metadata_compatibility.py` : valide chaque CSV de `data/` contre `data/_schemas/v1_ohlcv.json`.

### 5.5 CI matrix

GitHub Actions (cf. catégorie 9) :
- Ubuntu + Python 3.11
- Marker `integration` skippé sans secrets
- Job nightly avec secrets → exécute tests live

---

## 6. Sécurité

### 6.1 Secrets management
- **Aujourd'hui** : aucun secret data provider (Dukascopy gratuit). Sera critique post-T1.2.
- **Cible** : variables `.env` non commitées + production via Doppler/AWS Secrets Manager.
- Rotation 90 j obligatoire (mise à jour cron `scripts/rotate_data_keys.py`).

### 6.2 Rate limits & abuse prevention
- Tiingo : 1000 req/h Starter → wrapper rate-limiter (token bucket) dans le provider.
- Polygon : 5 calls/min sur tier Starter REST + WebSocket unlimited.
- TwelveData : 800 req/day Pro → optimiser via aggregator (1 stream WS plutôt que N pulls REST).
- Trading Economics : 500 req/day → cache 15 min + batched fetch hebdo.

### 6.3 Egress encryption
- Tous les providers : HTTPS / WSS obligatoire. TLS 1.2+ enforced via `ssl_context` Python.

### 6.4 Audit log
- Chaque pull / stream loggue `provider`, `endpoint`, `key_id` (hash), `bytes_received`, `latency_ms`.
- Stocké dans `logs/data_audit.log` rotation hebdo. Conserver 1 an pour audit licence.

### 6.5 Protection contre key leak
- Pre-commit hook `detect-secrets` (cf. catégorie 4) + `git-secrets`.
- API keys jamais en `logger.info()` (vérifier `data_providers.py` post-impl).

---

## 7. Métriques

### 7.1 SLI (Service Level Indicators) cibles

| Métrique | Mesure | Cible MVP | Cible Y1 |
|---|---|---:|---:|
| `data_coverage_pct{symbol="XAUUSD"}` | bars actuels / attendus sur session active | ≥ 98 % | ≥ 99.5 % |
| `data_freshness_seconds{symbol,tf="M15"}` | `now - last_bar_ts` | < 120 s P95 | < 30 s P95 |
| `bar_to_signal_latency_seconds` | bar_close → signal publish | < 5 s P95 | < 2 s P95 |
| `data_validation_errors_total` | counter erreurs structurelles | < 1/jour | 0/jour |
| `data_gap_count_total[1h]` | gaps en session active | < 2/h | < 1/h |
| `provider_failover_total[1h]` | nombre fallbacks | < 5/h | < 1/h |
| `data_cost_usd_total` | dépense providers mensuelle | < $300 | < $500 |

### 7.2 KPI business
- Coût data par signal publié : (somme abonnements / nb signaux) → cible < $0.10 / signal à 100 paid users.
- Coverage par instrument : tableau de bord exposé interne (Grafana panel).

---

## 8. Risques & mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|:---:|:---:|---|
| Provider downtime (Tiingo / Polygon) | Moyenne | Critique | `CompositeDataProvider` 3-tier (T2.2) + alerte freshness |
| License breach Dukascopy (réclamation) | Faible-Moy | Critique légal | T1.6 décom + T1.1 audit écrit + flag personal-use only |
| FF JSON URL change ou TOS update | Élevée | Moyenne | T1.5 migrer Trading Economics avant fin Sprint |
| Schema change provider (rename column) | Moyenne | Moyenne | T3.4 schema validator + tests intégration nightly |
| API key leak | Faible | Critique | T6 secrets management + pre-commit + rotation 90j |
| Cost overrun (rate limit hit triggering paid burst) | Moyenne | Moyenne | Rate limiter dans provider + alerte `provider_request_total > quota * 0.8` |
| Tick aggregator off-by-one (bar close timing) | Moyenne | Élevée | T2.3 batterie de 10 tests DST + late tick + reference dump |
| Backbone WebSocket TLS handshake fail (broker side) | Faible | Moyenne | Reconnect exponential + circuit breaker T2.2 |
| Données corrigées rétroactivement (FRED revisions) | Élevée (mensuel) | Moyenne | T5.3 point-in-time queries + audit log |
| Coverage temporaire < 95 % suite outage | Moyenne | Moyenne | Backfill REST automatique sur reconnexion (T2.1 `fetch_history`) |

---

## 9. Dépendances inter-catégories

| Cat | Dépendance | Direction |
|---|---|---|
| **4 Compliance** | Audit légal Dukascopy/FF avant migration ; mention licence dans `/terms` | bidirectionnel |
| **6 Backtest** | Couverture ≥ 98 % et point-in-time queries pour reproductibilité chiffres | data → backtest |
| **7 ML** | Schema versioning prévient skew train/serve (eval_23) | data → ML |
| **8 API/Infra** | WebSocket scanner async dépend de refonte event-loop ; `/metrics` exposé via FastAPI | data → API |
| **9 Observability** | Prometheus + Grafana fournis par cat 9 ; data publie gauges | data → obs |
| **10 GTM/Pricing** | Coût data dans unit economics (cf. eval_24) | data → GTM |

---

## 10. Estimation totale & timeline

### 10.1 Effort

- **P0 (blocker commercialisation)** : 127 h ≈ 16 jours-dev plein temps (3-4 sem solo à 30-40 h/sem).
- **P1** : 19 h ≈ 1 semaine additionnelle.
- **P2** : 30 h ≈ 1 semaine additionnelle.
- **Total complet** : **176 h** ≈ **5-6 semaines solo**.

### 10.2 Calendrier proposé

| Sem | Bloc | Livrables |
|---:|---|---|
| S1 | P0 Coverage gate + cleanup (11 h) + Licensing emails T1.1 (3 h) + setup Tiingo T1.2 (8 h) | Démarrage refuse feed corrompu, première souscription |
| S2 | Polygon ou TwelveData T1.3/T1.4 (20-32 h) | Provider FX live opérationnel |
| S3 | Trading Economics T1.5 (10 h) + Décom FF/Duka T1.6 (3 h) + Pipeline live T2.1-T2.3 (24 h) | Calendrier macro légal + scanner async démarré |
| S4 | T2.4-T2.5 BarQueue + Scanner refactor (20 h) + QA monitoring T3 (16 h) | Pipeline production-grade, monitoring live |
| S5 | P1 redundancy (19 h) + buffer / debugging | Failover chaîné testé en chaos |
| S6 | P2 data lake (30 h) | Parquet + S3 + point-in-time queries |

### 10.3 Coût provider 12 mois

| Phase | Mois | Stack | $/mo | Cumul |
|---|---|---|---:|---:|
| M1 Audit + setup | M1 | Tiingo $30 | $30 | $30 |
| M2 Pipeline live | M2 | + TwelveData $79 + TE $99 | $208 | $238 |
| M3-12 Run | M3-12 | + EODHD $20 (P1) | $228 | $238 + 10×$228 = **$2 518/an** |

À comparer aux revenus cibles eval_28 GTM ($5-7k MRR M12) → coût data = **~3-5 % du CA**, parfaitement viable.

### 10.4 Définition de Done globale (catégorie 5)

- [ ] Boot abort si coverage < 95 % sur tout symbole actif (T0.1)
- [ ] 0 référence ForexFactory ou Dukascopy hors `scripts/legacy/` (T1.6)
- [ ] Au moins 2 providers commerciaux contractualisés + DPA signés (T1.2 + T1.3/T1.4)
- [ ] Calendrier macro alimenté via Trading Economics (T1.5)
- [ ] `CompositeDataProvider` en prod avec failover testé (T2.2)
- [ ] Latence bar_close → signal < 5 s P95 mesurée (T2.5)
- [ ] 8 séries Prometheus exposées + 4 règles d'alerte actives (T3.1-T3.2)
- [ ] Schema validation + `.metadata.json` sur tous les CSV (T3.4)
- [ ] Couverture tests ≥ 80 % sur `src/intelligence/data/` (cat 9)
- [ ] Audit licence écrit archivé `compliance/data_license_correspondence.md` (T1.1)

---

## Annexe A — Matrice comparative des providers

| Provider | Couverture instruments | History | Latence | Licence commerciale | Coût/mo (entry) | API style | Verdict pour Sentinel |
|---|---|---|---:|---|---:|---|---|
| **Dukascopy** (actuel) | FX + Metals + CFDs tick | 1990+ | T+1 | ❌ Personal only (TPS) | $0 | HTTP LZMA (`.bi5`) | À retirer prod, garder R&D |
| **MT5 broker history** | Broker-dependent | ~10 ans | < 1 s | ⚠️ Liée au compte trader | $0 (avec compte) | MT5 Python SDK | Fallback uniquement, Windows-only |
| **Polygon.io Currencies Starter** | FX + (Metals via cross) | 2003+ | < 100 ms (WS) | ✅ Commercial OK | **$79** | REST + WebSocket | Recommandé FX |
| **Polygon.io Stocks Starter** | Equities US | 2003+ | < 100 ms | ✅ | $79 | REST + WS | Optionnel pour US500 |
| **Tiingo IEX + FX Starter** | Stocks US + FX | 1996+ FX | T+0 EOD, intraday IEX | ✅ Commercial | **$30** | REST | Bon rapport prix, FX EUR/JPY/GBP |
| **Twelve Data Pro** | FX + Metals + Crypto + Indices + Stocks | 1990s+ | < 500 ms | ✅ Commercial | **$79** | REST + WS | **Single-vendor préféré** (couvre tout) |
| **Twelve Data Ultra** | idem + higher quota | idem | < 200 ms | ✅ | $229 | REST + WS | Si > 100 paid users |
| **Databento Standard** | Tick-level institutional CME/ICE | 2010+ | < 1 ms | ✅ Enterprise | **$250-2000** | gRPC + REST | Overkill solo, garder pour Y2 institutionnel |
| **Alpaca Markets Algo Trader+** | Stocks US + Crypto | 2015+ | < 100 ms | ✅ | $99 | REST + WS | Pas de FX/Metals, hors scope MVP |
| **IEX Cloud Launch** | Stocks US | 5 ans | T+0 | ✅ | $19 | REST | Stocks only, complément niche |
| **MetalsAPI Pro** | Spot metals (XAU/XAG/Pt/Pd) | 2000+ | T+0 | ✅ | $99 | REST | Spécialisé XAU si TD insuffisant |
| **Tradermade FX & Metals** | FX + Metals | 2000+ | < 500 ms | ✅ | $50-200 | REST + WS | Alternative TD |
| **Trading Economics API** | Calendrier macro + indicators | 2010+ | T+0 | ✅ Commercial | **$99** | REST | **Choix recommandé** vs FF |
| **Econoday** | Calendrier macro premium | 2000+ | T+0 | ✅ | $300-1000 | REST/Excel | Enterprise, overkill MVP |
| **MarketAux** | News + calendrier basique | 2018+ | T+0 | ✅ | $19 | REST | Économique mais light |
| **EODHD** | Stocks/FX/Indices EOD | 1990s+ | EOD | ✅ | **$19.99** | REST | Excellent backup 2e fallback |
| **Refinitiv (LSEG)** | Institutional all asset | 1980+ | Real-time | ✅ Enterprise | $1500+ | Eikon/Workspace | Hors budget solo |
| **ICE Data Services** | Institutional refs | Real-time | $1000+ | ✅ | $1000+ | API | Hors budget |
| **Bloomberg B-PIPE** | Institutional | Real-time | $2000+ | ✅ | $2000+ | BLPAPI | Hors budget |

### Recommandation finale provider mix

- **Sprint MVP (S1-S2)** : Tiingo $30 (FX) + Trading Economics $99 (macro) = **$129/mo**.
- **Pre-launch Stripe (S3-S4)** : + TwelveData Pro $79 (XAU + WS) = **$208/mo**.
- **Scale (M6+)** : + EODHD $20 (backup) = **$228/mo**.
- **Year 2 institutionnel** : remplacer TD par Polygon Stocks+FX $199 + Databento $500 si tier INSTITUTIONAL > 10 clients = **$900/mo**, marges toujours > 80 %.

---

## Annexe B — Mapping bugs eval_08 ↔ tâches du plan

| Bug eval_08 | Tâche | Statut post-Sprint |
|---|---|---|
| #1 cache RAM full | T2.1 refactor + lazy tail load | Résolu |
| #2 pas de tail incremental | T2.1 | Résolu |
| #3 pas de validation feed source | T0.1 coverage gate | Résolu (P0) |
| #4 MT5 pas thread-safe | T4.3 | Résolu (P1) |
| #5 pas de fallback fournisseur | T2.2 CompositeDataProvider | Résolu (P0) |
| #6 pas de schema versioning | T3.4 metadata.json | Résolu (P0) |
| #7 reconnect 6 s bloquant | T4.3 async reconnect | Résolu (P1) |
| #8 pas de log freshness | T0.4 | Résolu (P0) |
| #9 strict=True crash scanner | T0.3 DLQ | Résolu (P0) |
| #10 0 tests intégration feed réel | §5.2 sandbox keys | Résolu (P0) |

---

## Annexe C — Fichiers à créer / modifier

### Nouveaux modules
- `src/intelligence/data/__init__.py`
- `src/intelligence/data/composite_provider.py`
- `src/intelligence/data/tiingo_provider.py`
- `src/intelligence/data/polygon_provider.py` (ou `twelvedata_provider.py`)
- `src/intelligence/data/tick_aggregator.py`
- `src/intelligence/data/bar_queue.py`
- `src/intelligence/data/schema.py`
- `src/intelligence/data/metrics.py`
- `src/intelligence/data/dlq.py`
- `src/agents/news/te_calendar_provider.py`
- `data/_schemas/v1_ohlcv.json`
- `data/README.md`
- `compliance/data_license_correspondence.md`
- `infrastructure/alertmanager/data_alerts.yml`
- `scripts/rotate_data_keys.py`
- `scripts/migrate_csv_to_parquet.py` (P2)

### Modifications
- `src/intelligence/data_providers.py` — étendre ABC, ajouter `async` (T2.1)
- `src/intelligence/data_quality.py` — ajouter `assert_coverage()` (T0.1)
- `src/intelligence/sentinel_scanner.py` — wire validation + bar queue consumer (T0.3, T2.5)
- `src/intelligence/main.py` — boot-time coverage gate (T0.1)
- `requirements.txt` — `websockets`, `httpx`, `pyarrow`, `duckdb`, `prometheus_client`
- `infrastructure/Dockerfile` — env vars secrets management
- `tests/test_data_quality.py` — coverage gate cases
- `tests/test_sentinel_scanner.py` — DLQ injection cases

### Suppressions / dépréciations
- `scripts/download_economic_calendar.py` (HTML scraper cassé) → archive `scripts/legacy/`
- `data/XAU_15MIN_2019_2025.csv` → rename `.deprecated.csv` puis suppression Sprint+1
- `data/macro/` duplique `data/research/` → unifier sous `data/research/macro/`
- Toute mention `forexfactory.com` ou `dukascopy.com` hors `scripts/legacy/`

---

**Fin du plan catégorie 5.**

---

Chemin : `C:\MyPythonProjects\TradingBOT_Agentic\reports\commercialization_sprint\05_data_infrastructure.md`
Top 3 P0 : (1) Coverage gate fail-fast + purge feed 63 % (T0.1+T0.2, 5 h, $0) — (2) Souscription Tiingo + TwelveData + Trading Economics (T1.2+T1.4+T1.5, 30 h, $208/mo) — (3) Pipeline live WebSocket + CompositeDataProvider + scanner async (T2.1-T2.5, 44 h, inclus).
Heures P0 : 127 h ≈ 4 sem solo · Total complet (P0+P1+P2) : 176 h ≈ 5-6 sem.
Coût providers/mo estimé : $208/mo MVP launch · $228/mo run cruising · $900/mo Year 2 institutionnel.
Coût Y1 cumulé : ~$2 518/an (3-5 % des revenus cibles eval_28).
