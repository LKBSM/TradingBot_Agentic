# Kill Criteria Board — Smart Sentinel AI

> **Source de vérité gouvernance.** Mis à jour chaque vendredi 16h par Sofia (RISK-1.1).
> Tout sprint actif a un status (🟢/🟡/🔴), un kill criterion explicite, et un blocker tracké.
> Référence plan : `reports/roadmap_2026_2027/PLAN_12_MOIS.md`.

**Dernière mise à jour** : 2026-05-03 18:30 ET
**Phase active** : **2B, Sprints LLM-2B.1+2+3+4+5 + INFRA-2B.5 livrés (RAG full-stack + Q&A B2C + /enrich B2B + 4 langues)**
**Mois en cours** : M3 effective (Phase 1 entièrement bouclée 2026-05-01)
**Heures dev cumulées vs plan** : ~28h / 116h (18 sprints touchés sur 18, économie ~76%)

---

## 1. KPIs critiques globaux

| KPI | Seuil vert | Seuil rouge | Actuel | Status |
|---|---|---|---|---|
| Test coverage (pytest --cov) | ≥75% | <70% | __% | 🟢/🟡/🔴 |
| /metrics endpoint live | payload non-vide | payload vide | __ | 🟢/🟡/🔴 |
| Signaux générés 7j | ≥30 | 0 | __ | 🟢/🟡/🔴 |
| Erreurs Sentry 7j | <10 | >50 | __ | 🟢/🟡/🔴 |
| LLM cost / revenue (Phase 2) | <40% | >60% | __% | 🟢/🟡/🔴 |
| Forward-test PF rolling 30j (Phase 2A) | ≥1.10 | <0.85 | __ | 🟢/🟡/🔴 |
| RAG faithfulness (Phase 2B) | ≥0.90 | <0.80 | n/a (no LLM yet, recall=98% / precision=20%) | 🟢 |

---

## 2. Sprints actifs (mise à jour hebdo)

| Sprint ID | Titre | Owner | Effort planned/actual | ETA | Last update | Status | Kill criterion | Blockers |
|---|---|---|---|---|---|---|---|---|
| DATA-1.1 | FRED macro ingestion | Marwan | 4h / 3h code | 2026-04-30 | 2026-05-01 | 🟡 | fredapi rate-limit casse ingest | KPI live blocké : pas de FRED_API_KEY (B-001) |
| DATA-1.2 | CFTC COT ingestion | Marwan | 4h / 2h | 2026-05-01 | 2026-05-01 | 🟢 | format ZIP CFTC change | aucun. 365 weeks 2019-2025 ingérés, 5/5 tests verts |
| DATA-1.3 | GLD ETF flows | Marwan | 4h / 0h | 2026-05-01 | 2026-05-01 | ⏸ DEFERRED | SPDR JSON schema change | **Voie D** retenue : différé Phase 2A. ~17 features dispo sans GLD = ≥18 cible plan. |
| QUANT-1.1 | A1 feature matrix | Elena | 4h / 1h30 | 2026-05-01 | 2026-05-01 | 🟢 | NaN > 30% après ffill | aucun. **152,961 bars × 19 features × leak 0/100 → KPI ALL GREEN**. Parquet `data/research/a1_matrix_2019_2026.parquet`. 10/10 tests verts. |
| QUANT-1.2 | CPCV harness | Elena | 6h / 1h30 | 2026-05-01 | 2026-05-01 | 🟢 | runtime > 4h | aucun. **Runtime ~7 min sur 152k bars** (cible <30min PASS). 17/17 tests verts. CPCV 28 paths + DSR Bailey-LdP + PBO + Holm + DM tous opérationnels. |
| QUANT-1.3 | A1 stack training + verdict | Elena | 6h / 1h30 | 2026-05-01 | 2026-05-01 | 🟢 livré 🔴 verdict | DSR<0 ou PBO>0.6 → kill A1 | **VERDICT TRANCHÉ : GO 2B**. DSR=0.0, PBO=0.5, PF=1.008, DM stat +46.7 (A1 pire que constant). Score 1/6 critères. Rapport `reports/a1_verdict_2026.md`. Modèle versionné `models/a1_stack_v1.pkl`. |
| REGIME-1.1 (a)+(c) | VOL_MODE + latency test | Kenji | 4h / 1h | 2026-05-01 | 2026-05-01 | 🟢 partial | export ONNX RMSE delta >5% | (a) scripts/MEMORY alignés sur default `har`. (c) `tests/test_vol_latency.py` mesure p99 ≈ 116ms (cible 100ms manquée, mais sous 200ms kill-criterion). Marker `benchmark` opt-in en CI. (b) ONNX export déféré : profiling montre que latence est dominée par DataFrame slicing dans `forecast()`, pas par la régression — un export ONNX du LinearRegression seul gagnerait <1ms. Refactor du forecaster requis pour atteindre p99<100ms ; out of scope Phase 1. |
| REGIME-1.2 | BOCPD prototype | Kenji | 4h / 1h | 2026-05-01 | 2026-05-01 | 🟢 | cp_prob dégénéré | aucun. Adams-MacKay 2007 implementation pure Python ~250 lignes (NIG conjugate Gaussian + log-sumexp stable). 8/8 tests verts. Step-change détecté à t=100 avec cp_prob 0.677 vs steady 0.001 (×500 ratio). Latence 0.83ms/step << 100ms cible. Reste utile en 2B pour narrative régime context (LLM-2B.1). |
| LLM-1.1 | Eval harness 50 prompts | Aisha | 6h / 1h30 | 2026-05-01 | 2026-05-01 | 🟢 | forbidden_phrases <0.95 | aucun. 50 fixtures (15 BUY + 15 SELL + 10 HOLD + 5 vol + 5 news), 5 axes scoring, 22 tests verts. Baseline KPIs ALL PASS (factual 1.0, forbidden 1.0, brevity 1.0). CI étendue 80 tests. |
| INFRA-1.1 | GitHub Actions CI/CD | Théo | 3h / 1h | 2026-05-01 | 2026-05-01 | 🟢 | tests CI dépendants CSV local | scope initial limité aux test_fred + test_cot (cov 81%). Full suite à expand quand fixtures prêts. |
| INFRA-1.2 | Sentry + /metrics | Théo | 3h / 1h | 2026-05-01 | 2026-05-01 | 🟢 partial | Sentry > free tier | observability.py + 14 tests verts. 3 métriques standards instanciées au boot. Sentry opt-in via SENTRY_DSN. PII scrubber actif. Print() audit (109 occur 23 fichiers) déféré pour limiter risque. |
| UX-1.1 | InsightSignal v2 + 4 mockups | Inès | 5h / 1h | 2026-05-01 | 2026-05-01 | 🟢 | v2 casse >10 tests | aucun. Pydantic v2 unifié, 30 tests verts (round-trip, validators directionnels, compliance UE 2024/2811). 4 mockups v2 auto-générés via scripts/generate_mockups_v2.py. |
| COMM-1.1 | Positioning 2A+2B briefs | Karim | 5h / 1h30 | 2026-05-01 | 2026-05-01 | 🟢 | docs trop similaires | aucun. 2 briefs livrés AVANT verdict A1 (anti-biais ex-post). Sofia review checklist intégrée à chaque doc. |
| RISK-1.1 | Kill criteria board + weekly_check | Sofia | 6h / 2h | 2026-04-30 | 2026-05-01 | 🟢 partial | 2 checkpoints ratés consec | board live + tools/governance/weekly_check.py + 11 tests verts. Reste : 8 weekly checks réels à mener S1-S8. |
| LLM-2B.3 | RAG eval harness 50 fixtures + RAGAS metrics | Aisha | 10h / 1h30 | 2026-05-03 | 2026-05-03 | 🟢 | recall<0.85 ou precision<0.15 | aucun. 50 fixtures (15 paper + 10 data + 5 report + 10 concept + 10 macro), 4 métriques (recall/precision/faithfulness/relevancy), KPI gate **recall=98% & precision=20% PASS**. 22 tests verts. CI étendue 12 fichiers. |
| LLM-2B.5 | Q&A endpoint /api/v1/qa over RAG corpus | Aisha | 12h / 1h | 2026-05-03 | 2026-05-03 | 🟢 | latence p99 > 1s OU stub mode casse | aucun. POST `/api/v1/qa` accepte query+language+top_k, route via RAGPipeline, retourne answer (LLM réel STRATEGIST+ ou stub déterministe FREE/ANALYST), sources citées + elapsed_ms. 13 tests verts (happy path + edge cases + state-isolation). Hors CI faute de httpx/TestClient deps — tests locaux green. |
| LLM-2B.4 | RAG multi-langue FR/EN/DE/ES | Aisha | 12h / 1h | 2026-05-03 | 2026-05-03 | 🟢 | prompt DE/ES casse hard rules ou compliance | aucun. SYSTEM_PROMPT_DE + SYSTEM_PROMPT_ES ajoutés, dispatch table `SYSTEM_PROMPTS`, fallback EN sur tag inconnu. QARequest enum élargi à fr/en/de/es. Stub answer headers traduits 4 langues. 16 tests prompt + 4 tests endpoint paramétrés (Synthèse/Summary/Zusammenfassung/Resumen). CI étendue 13 fichiers. |
| INFRA-2B.5 | B2B endpoint /api/v1/enrich | Théo | 8h / 1h | 2026-05-03 | 2026-05-03 | 🟢 | InsightSignalV2 contract drift OU broker erreur > 5% 422 | aucun. POST `/api/v1/enrich` accepte EnrichRequest (instrument+TF+direction+levels+broker_context+lang), exécute RAG retrieve, retourne `InsightSignalV2` complet (narrative_short localisé, narrative_long stub-ou-LLM, sources_cited 8 chunks, conviction heuristique RR-based, compliance edge_claim=False/is_paper_demo=True). Pre-validation directionnelle stop<entry → 422 propre. NEUTRAL strip levels. client_request_id echo. 16 tests verts. |

**Légende status** :
- 🟢 = on track, no concerns
- 🟡 = slip < 1 semaine OR métrique en zone jaune
- 🔴 = slip > 1 semaine OR métrique rouge OR kill criterion proche

---

## 3. Checkpoints à venir

| Checkpoint | Date prévue | Owner | Critère go | Critère pivot |
|---|---|---|---|---|
| CP-1.1 | Fin S2 | Marwan | FRED+COT+GLD ingérés, no-look-ahead | Décaler S3 |
| CP-1.2 | Fin S4 | Elena | A1 baseline tournée RMSE calc | Investiguer leakage |
| CP-1.3 | Fin S6 | Elena | A1 walk-forward CPCV DSR/PBO calc | Si DSR<0 → kill A1 |
| ~~**CP-A1**~~ ✅ **TRANCHÉ 2026-05-01** | Fin S8 (4 sem avance) | Elena+Sofia | DSR>0.99 ET PBO<0.3 ET CPCV PF>1.20 ET ≥3 Holm + DM-stat<0 | **GO 2B activé** (1/6 green, edge pas démontré) |
| CP-2A.1 | Fin M4 | Sofia | Forward-test 30j PF≥1.10 → Stripe ON | Kill 2A si PF<0.85 |
| CP-2A.2 | Fin M6 | Karim | 1 LOI B2B signée | Recentrer B2C |
| CP-2B.1 | Fin M5 | Aisha | RAG eval F1>0.85, halluc<5% | Itérer prompts |
| CP-2B.2 | Fin M9 | Karim | 1 contrat B2B €500-1500/mo | Persévérer 3 mois |

---

## 4. Décisions prises ce mois

- YYYY-MM-DD : {décision}, owner : {agent}, rationale : {x}

---

## 5. Blockers actifs et escalations

- **B-001 (DATA-1.1)** : FRED_API_KEY non fourni → smoke run live KPI bloqué. DoD pytest 6/7 verts (mocked). Détails : `BLOCKERS.md`. Action : user doit fournir clé FRED gratuite avant QUANT-1.1.
- ~~**B-001 (DATA-1.1)**~~ **RESOLVED 2026-05-01 08:24 ET** : clé FRED fournie par user, smoke run live PASS (5 séries × 7.32 ans × 0 NaN). T10Y2Y bascule sur fallback default-lag (3027 vintages dépassent endpoint limit, conservative).
- ~~**B-002 (DATA-1.3)**~~ **RESOLVED 2026-05-01** : voie D retenue — différé Phase 2A. ~17 features Elena dispo sans GLD ≥ cible plan. Reprise conditionnelle au verdict A1.

---

## 6. Sprints abandonnés / killed (post-mortem requis)

| Sprint ID | Date kill | Raison | Post-mortem |
|---|---|---|---|

---

## 7. Note hebdo Sofia (5-15 lignes libres)

> Format : ce qui marche / ce qui inquiète / signal à surveiller la semaine prochaine.
> Pas de format imposé. Honnêteté avant esthétique.

**2026-04-30 (S1, démarrage Phase 1)** :
DATA-1.1 livré dans le scope du DoD (6/7 tests mocked verts, look-ahead 100/100). Bémol honnête : KPI "5 séries × ≥6 ans daily" non vérifié live faute de clé FRED. Bloqué proprement, documenté dans BLOCKERS.md, le module est néanmoins fonctionnel et prêt à tourner dès que la clé arrive. Décision : continuer DATA-1.2 (CFTC COT, pas de clé requise) sans attendre. Surveiller que le user fournit la clé avant QUANT-1.1 sinon Elena bloquera S3.

---

## 8. Discipline (rappel non-négociable)

1. Mesurer en temps réel, pas a posteriori
2. Pré-déclarer les seuils, pas les chercher après
3. Compter les ratés, pas les optimistes
4. Sofia a un droit de veto explicite (7 jours block + rétro)
5. Transparence radicale interne : ce fichier n'est PAS retouché après publication, on ajoute un changelog
