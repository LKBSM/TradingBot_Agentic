# Smart Sentinel AI — Plan d'amélioration 12 mois (équipe 8 agents)

**Document de pilotage produit composite — solo founder Montréal, 8-9h/sem, budget <500€/an**

Daté du 2026-04-30. Source de vérité pour les sprints, kill criteria, et bascule de phase.
Versionné dans `reports/roadmap_2026_2027/`. Référencé dans `MEMORY.md` (auto-memory persistante).

> **Méthodologie** : 8 agents spécialisés (Marwan Data, Elena Quant, Kenji Regime/Vol, Aisha LLM, Théo Infra/Compliance, Inès UX, Karim Commercial, Sofia Risk/Governance) collaborent sur un produit SaaS composite à 7 couches (data, algo, stat, LLM, infra, UX, GTM). Plan structuré en 3 phases avec embranchement conditionnel sur le verdict A1 (test décisif fin M2).

---

## Table des matières

- **PARTIE I** — Vision et structure
- **PARTIE II** — Phase 1 (Mois 1-2) DÉTAILLÉE
- **PARTIE III** — Phase 2A (Mois 3-12, edge confirmé)
- **PARTIE IV** — Phase 2B (Mois 3-12, edge non démontré)
- **PARTIE V** — Synthèse comparative 2A vs 2B
- **PARTIE VI** — Plan de mesure et reporting

---

# PARTIE I — Vision et structure

## I.1 Synthèse exécutive globale (12 mois)

Smart Sentinel AI doit être traité **non comme un algorithme**, mais comme une **plateforme SaaS composite à 7 couches** dont l'edge algorithmique (couche 2) n'est qu'un des 7 actifs commercialisables. L'audit CIO 3.46/10 a montré que la couche 2 est aujourd'hui non-significative (Pearson −0.023, 0/7 composants Holm) ; **le verdict A1 fin mois 2 décide si la couche 2 mérite un investissement de 320h supplémentaires (Phase 2A) ou si on reroute vers la valeur des 6 autres couches (Phase 2B)**.

**Calendrier maître :**
- **Phase 1 (mois 1-2, 64h dev)** : test décisif A1 + hardening minimal des autres couches. Identique pour les deux scénarios. Coût ~150€.
- **Phase 2A (mois 3-12, 320h dev)** : si A1 valide (PBO < 0.3, DSR > 1.0, CPCV PF > 1.20). Cap algorithmique + scaling B2C + amorçage B2B-API broker. Coût ~600€/an.
- **Phase 2B (mois 3-12, 320h dev)** : si A1 invalide. Pivot produit "intelligence narrative + RAG sourcé" pour B2C éducatif et B2B "qualité de signal augmentée". Coût ~350€/an.

**Total annuel : 384h dev (8h/sem × 48 sem actives), <500€ infra fixe (600€ en 2A justifié), LLM API variable selon traction (~50-300€/mois en régime).**

**Probabilité a priori que A1 réussisse :** d'après l'audit Quant Senior (`falsification_2026_04_30`), bootstrap CI [0.70, 0.88] et corrélation β-capture 0.96 → estimation honnête **P(A1 succès) ≈ 25-35%**. Le plan traite 2A et 2B à parité — 2B est le scénario probable (~70%).

## I.2 Présentation des 8 agents

| # | Agent | Domaine couche | Heures Phase 1 | Heures Phase 2A | Heures Phase 2B |
|---|---|---|---|---|---|
| 1 | **Marwan** — Data Engineer | Couche 1 (data) | 12h | 50h | 35h |
| 2 | **Elena** — Quant Researcher | Couche 2 (algo) | 16h | 70h | 25h |
| 3 | **Kenji** — Regime/Vol | Couche 2-3 (stat) | 8h | 45h | 25h |
| 4 | **Aisha** — LLM/Narrative | Couche 4 (LLM) | 6h | 35h | 80h |
| 5 | **Théo** — Infra/Compliance | Couche 5 (infra) | 6h | 45h | 50h |
| 6 | **Inès** — Product/UX | Couche 6 (UX) | 5h | 30h | 50h |
| 7 | **Karim** — Commercial/Growth | Couche 7 (GTM) | 5h | 30h | 40h |
| 8 | **Sofia** — Risk/Governance | Transverse | 6h | 15h | 15h |
| | **TOTAL** | | **64h** | **320h** | **320h** |

**Lecture :** en Phase 2A l'effort est concentré sur Elena+Kenji+Marwan (algorithmique, 51% du temps) ; en Phase 2B il bascule vers Aisha+Inès+Théo (narrative+UX+compliance, 56%). Sofia reste constante : la gouvernance ne dépend pas du scénario.

## I.3 Points de décision clés

| Checkpoint | Date | Décision | Critère go | Critère pivot |
|---|---|---|---|---|
| **CP-1.1** | Fin S2 | Macro features dispo | FRED+COT+GLD ingérés, no-look-ahead vérifié | Sinon décaler S3 |
| **CP-1.2** | Fin S4 | A1 baseline tournée | RMSE LightGBM vs HAR baseline calculé | Sinon investiguer leakage |
| **CP-1.3** | Fin S6 | A1 walk-forward CPCV | DSR estimé, PBO calculé | Si DSR<0 → kill A1 immédiat |
| **CP-A1** ⚠️ | **Fin S8** | **VERDICT A1** | PBO<0.3 ET DSR>1.0 ET CPCV PF>1.20 ET ≥3 Holm | Sinon → branche 2B |
| CP-2A.1 | Fin M4 | Forward-test paper 30j | PF live ≥ 1.10 | Kill 2A si PF<0.85 |
| CP-2A.2 | Fin M6 | B2B-API démo broker #1 | 1 LOI signée | Recentrer B2C |
| CP-2B.1 | Fin M5 | RAG eval avec 100 prompts | F1 sourcing > 0.85, hallucination < 5% | Itérer prompts |
| CP-2B.2 | Fin M9 | Premier client B2B "qualité signal" | 1 contrat €500-1500/mo | Persévérer 3 mois |

## I.4 Vue d'ensemble du chemin critique

**Phase 1 (chemin critique) :**
```
DATA-1.1 (FRED) → DATA-1.2 (COT) → QUANT-1.1 (features) →
QUANT-1.2 (CPCV harness) → QUANT-1.3 (LightGBM stack) → CP-A1
```
8 semaines, ~30h sur le chemin critique. Le reste (Aisha, Inès, Karim) **parallélisable**.

**Phase 2A (chemin critique) :**
```
QUANT-2A.1 → REGIME-2A.1 → INFRA-2A.1 → INFRA-2A.2 (forward-test gate) →
COMM-2A.3 (outbound) → COMM-2A.4 (premier client B2B)
```

**Phase 2B (chemin critique) :**
```
LLM-2B.1 (RAG architecture) → LLM-2B.2 (sources curées) →
LLM-2B.3 (eval harness étendu) → UX-2B.1 (webapp narrative) →
COMM-2B.1 (positioning éducatif) → COMM-2B.4 (premier client B2B data-quality)
```

---

# PARTIE II — Phase 1 (Mois 1-2) DÉTAILLÉE

## II.1 Synthèse exécutive Phase 1

**Objectif :** trancher en 8 semaines la question "le système a-t-il un edge prédictif" via le test A1 (stacked LightGBM avec features macro), tout en maintenant l'exploitation actuelle (TESTING_MODE FREE) et en préparant les fondations communes aux deux scénarios.

**Heures totales :** 64h (8h/sem × 8 sem).
**Coût :** ~150€ (Polygon free tier suffisant ; FRED, CFTC, Yahoo gratuits ; LLM eval ~50€).

**KPI globaux :**
- ✅ FRED+COT+GLD ingérés avec timestamps audités (no look-ahead)
- ✅ CPCV walk-forward harness opérationnel et reproductible
- ✅ Verdict A1 chiffré (DSR, PBO, Holm, RMSE Diebold-Mariano)
- ✅ Eval harness LLM (50 prompts) tourné en CI
- ✅ Kill criteria board live
- ✅ Pipeline existant non régressé (1366+ tests verts)

**Critère de bascule fin S8 :**
- Si **PBO<0.3 ET DSR>1.0 ET CPCV PF>1.20 ET ≥3 composants Holm-significatifs** → Phase 2A
- Sinon → Phase 2B (par défaut, sans appel à l'optimisme)

## II.2 Section par agent

### Agent 1 — DATA ENGINEER (Marwan) — 12h

**Vision Phase 1 :** Mes 12h livrent les fondations data dont Elena a besoin pour A1. Sans macro propre (timestamps audités, vintage FRED, COT décalage publication), le test A1 est invalide quel que soit le résultat. Je ne touche pas aux feeds existants (XAU CSV, FF news) — ils restent en l'état.

#### Sprint DATA-1.1 — FRED macro ingestion — 4h
- **Owner** : Marwan / **Phase** : 1 / **Effort** : 4h / **Dépendances** : aucune
- **Spec** : créer `src/agents/data/fred_provider.py`. Récupérer via `fredapi` (clé gratuite) les séries : DGS10 (10y yield), DFII10 (TIPS 10y → breakeven = DGS10 - DFII10), DTWEXBGS (DXY broad), VIXCLS, T10Y2Y. Stocker en CSV daily à `data/macro/fred_{series}.csv` avec colonnes `date_utc, value, vintage_date` (vintage = ALFRED, pas just FRED, pour éviter look-ahead sur révisions). Resampler M15 par forward-fill.
- **DoD** : `pytest tests/test_fred_provider.py` 5 tests verts, vintage timestamps présents, `assert macro_at(t) timestamp <= t - publication_lag` passe sur 100 random dates.
- **KPI succès** : 5 séries × ≥6 ans de daily, 0 NaN après ffill, look-ahead test 100/100 OK.
- **Critère kill** : si `fredapi` rate-limit casse l'ingest ou si vintages absents → fallback yfinance + accepter MNAR documenté.
- **Référence** : ALFRED docs https://alfred.stlouisfed.org/docs/alfred_API.pdf, lib `fredapi` (PyPI).
- **Risques** : confondre release date et observation date (cause classique de look-ahead). Mitigation : tests de vintage explicites.

#### Sprint DATA-1.2 — CFTC COT ingestion — 4h
- **Owner** : Marwan / **Phase** : 1 / **Effort** : 4h / **Dépendances** : aucune
- **Spec** : créer `src/agents/data/cot_provider.py`. Télécharger weekly Disaggregated Futures Only Reports CFTC pour Gold (code 088691). Source : https://www.cftc.gov/dea/newcot/deafutu.txt + `dea_fut_disagg_xls_2026.zip`. Champs : `Managed_Money_Long`, `Managed_Money_Short`, `Producer_Net`, `Open_Interest`. Calculer ratios : `mm_net_pct = (mm_long - mm_short) / open_interest` et z-score sur fenêtre 52 semaines. **PIÈGE** : COT publié vendredi 15h30 ET pour positions au close mardi. Implémenter `cot_at(t)` qui respecte cette logique de lag.
- **DoD** : `tests/test_cot_provider.py` avec 3 tests dont fixture qui vérifie qu'à 14h30 ET vendredi on retourne la COT semaine -1, et à 16h00 ET vendredi celle de la semaine courante.
- **KPI succès** : 6+ ans de COT weekly, lag publication respecté sur 100/100 timestamps test.
- **Critère kill** : si CFTC change le format ZIP — fallback parser Legacy text format.
- **Référence** : CFTC COT docs.
- **Risques** : look-ahead très facile à introduire. Mitigation : test stress.

#### Sprint DATA-1.3 — GLD ETF flows + yfinance baseline — 4h
- **Owner** : Marwan / **Phase** : 1 / **Effort** : 4h / **Dépendances** : aucune
- **Spec** : créer `src/agents/data/gld_provider.py`. Récupérer via `yfinance` les daily OHLCV de GLD, IAU, SLV. Calculer flows via `(close - open) * volume` en proxy. Calculer SPDR holdings en tonnes via scraping JSON `https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.json`. Stocker `data/macro/gld_holdings_tons.csv`.
- **DoD** : 3 tests pytest, dont 1 qui vérifie monotonie holdings (drop > 5% en 1 jour = anomalie loggée).
- **KPI succès** : 6 ans daily, 0 gap >2 jours ouvrés.
- **Critère kill** : si SPDR change leur JSON — accepter perte du signal holdings, garder seulement OHLCV proxy.
- **Référence** : `yfinance` PyPI, SPDR archives.

**Synthèse Marwan Phase 1 :** À fin S2 j'aurai livré 3 nouvelles sources macro avec timestamps auditables et tests de look-ahead. Cela débloque Elena pour QUANT-1.1.

### Agent 2 — QUANT RESEARCHER (Elena) — 16h

**Vision Phase 1 :** Mes 16h sont presque entièrement dédiées au test A1. Je rejette explicitement de toucher à la stack ConfluenceDetector existante (Pearson −0.023 avéré). A1 = est-ce qu'avec des features macro propres, un LightGBM stacké peut prédire le signe ou la magnitude des returns XAU M15 forward h=4 et h=16, de façon CPCV-stable, avec PBO<0.3 et DSR>1 ?

#### Sprint QUANT-1.1 — A1 feature matrix construction — 4h
- **Owner** : Elena / **Phase** : 1 / **Effort** : 4h / **Dépendances** : DATA-1.1, DATA-1.2, DATA-1.3
- **Spec** : créer `src/research/a1_features.py`. DataFrame XAU M15 avec : (a) features price-based : returns r_1, r_4, r_16, ATR_14_pct, RSI_14, MACD_signal_diff ; (b) features intra : `bar_minute_of_day`, `dow`, `is_lunch_hour` ; (c) features macro horodatées : DGS10, breakeven, DXY, VIX, COT mm_net_pct_z52, GLD holdings z52 ; (d) calendar proximity : `min_to_next_red_news`, `min_since_last_red_news`. Cible : `r_forward_4` et `r_forward_16`. Sauvegarder en parquet.
- **DoD** : matrice ≥150k lignes, 0 leak (test : valeur macro à t doit toujours être <= macro publié avant t).
- **KPI succès** : ≥150k bars, ≥18 features, 100/100 leak tests OK.
- **Critère kill** : si NaN > 30% après ffill macro → revoir.
- **Référence** : López de Prado, *Advances in Financial Machine Learning*, ch. 2-3.

#### Sprint QUANT-1.2 — CPCV walk-forward harness — 6h
- **Owner** : Elena / **Phase** : 1 / **Effort** : 6h / **Dépendances** : QUANT-1.1
- **Spec** : créer `src/research/cpcv_harness.py`. Combinatorial Purged Cross-Validation (López de Prado ch. 7) avec N=8 folds, k=2 test folds, embargo = 4h × M15 = 16 barres. Pour chaque combinaison (28 paths), entraîner et évaluer. Fournir `run_cpcv(model_factory, X, y) -> CPCVResult` avec `paths_returns`, `dsr`, `pbo`, `holm_pvalues_per_feature`. DSR : Bailey & López de Prado 2014. PBO : rank logit method.
- **DoD** : 4 tests unitaires (purge correct, embargo respecté, DSR formule sur cas connu, PBO sur jeu synthétique noise = ~0.5).
- **KPI succès** : harness reproductible (seed fixé), exécution bout-en-bout < 30 min sur dataset 150k bars / 18 features / LightGBM 100 trees.
- **Critère kill** : si runtime > 4h → revue d'archi, réduire à N=6.
- **Référence** : López de Prado *AFML* ch. 7 ; Bailey, Borwein, López de Prado, Zhu (2014).

#### Sprint QUANT-1.3 — A1 stacked LightGBM training + verdict — 6h
- **Owner** : Elena / **Phase** : 1 / **Effort** : 6h / **Dépendances** : QUANT-1.1, QUANT-1.2
- **Spec** : créer `src/research/a1_train.py`. Stack 2 niveaux : niveau 1 = LightGBM (régression) sur 3 sous-ensembles (price-only, macro-only, calendar+intra), niveau 2 = LightGBM méta-régresseur. Hyperparams : `n_estimators=200, max_depth=5, learning_rate=0.05, min_data_in_leaf=200`. Évaluer via CPCV-1.2 contre baseline = `r_forward_h.mean()` ET HAR-RV historique. Diebold-Mariano test. Holm-Bonferroni sur SHAP top-7. Cible : `y = sign(r_forward_4)` + régression magnitude. Produire `reports/a1_verdict_2026.md`.
- **DoD** : `reports/a1_verdict_2026.md` rempli avec chiffres réels, `models/a1_stack_v1.pkl` versionné.
- **KPI succès** : verdict tranché. Idéal : DSR>1.0, PBO<0.3, ≥3 Holm-significatives, PF backtest CPCV moyen > 1.20.
- **Critère kill** : DSR<0 ou PBO>0.6 → post-mortem et bascule Phase 2B.
- **Référence** : Wolpert (1992) Stacked Generalization ; LightGBM docs ; SHAP TreeExplainer.

**Synthèse Elena Phase 1 :** À fin S8 le verdict A1 est rendu, chiffré, public en interne. Mes 16h ne sont PAS supposées générer un edge — elles sont supposées générer un **verdict honnête sur l'existence d'un edge**.

### Agent 3 — REGIME & VOLATILITY SPECIALIST (Kenji) — 8h

**Vision Phase 1 :** Mes 8h ont 2 buts : (1) feature de régime exploitable dans A1 (cp_prob), (2) corriger la bavure VOL_MODE=hybrid (eval_04, latence 1.6-5s/forecast inacceptable, switch HAR par défaut).

#### Sprint REGIME-1.1 — HAR-RV optim + bavure VOL_MODE — 4h
- **Owner** : Kenji / **Phase** : 1 / **Effort** : 4h / **Dépendances** : aucune
- **Spec** : (a) corriger `config.py` pour `VOL_MODE="har"` par défaut ; (b) refactorer `volatility_forecaster.py` pour exporter HAR-RV en ONNX via skl2onnx — but : forecast latence < 50ms ; (c) `tests/test_vol_latency.py` asserte p99 < 100ms sur 1000 inférences.
- **DoD** : test latence vert, default VOL_MODE=har dans config + Procfile, CHANGELOG.md mis à jour.
- **KPI succès** : p99 forecast latence < 100ms, RMSE conservé (±2%).
- **Critère kill** : si export ONNX dégrade RMSE > 5% → fallback python pur, accepter latence 200ms.
- **Référence** : Corsi (2009) HAR ; skl2onnx docs.

#### Sprint REGIME-1.2 — BOCPD prototype sur returns XAU — 4h
- **Owner** : Kenji / **Phase** : 1 / **Effort** : 4h / **Dépendances** : QUANT-1.1
- **Spec** : Bayesian Online Changepoint Detection (Adams & MacKay 2007) sur returns XAU M15. Lib `bocd` (PyPI) ou ~50 lignes maison. Hazard = 1/240. Sauvegarder `cp_prob` comme feature pour A1 v2.
- **DoD** : feature `cp_prob` dans matrice A1 v2, test pytest synthétique step-change.
- **KPI succès** : feature ajoutée, test vert, calcul live < 100ms par bar.
- **Critère kill** : si cp_prob dégénérée (toujours <0.05 ou >0.5) → hazard mal calibré, abandonner.
- **Référence** : Adams & MacKay (2007), arXiv:0710.3742.

**Synthèse Kenji Phase 1 :** Feature régime pour A1 + bavure prod résolue. HAR-PD-REQ et JumpModel arrivent en Phase 2 si A1 réussit.

### Agent 4 — LLM & NARRATIVE ARCHITECT (Aisha) — 6h

**Vision Phase 1 :** Mes 6h ne touchent pas à l'archi LLM (déjà refactorée eval_05_implementation). Je construis l'**eval harness** qui mesurera la qualité narrative — utile dans les deux scénarios.

#### Sprint LLM-1.1 — Eval harness 50 prompts ground-truthed — 6h
- **Owner** : Aisha / **Phase** : 1 / **Effort** : 6h / **Dépendances** : aucune
- **Spec** : créer `tests/eval_llm/` avec 50 fixtures input (`InsightSignal` JSON) : 15 BUY haute conviction, 15 SELL, 10 HOLD, 5 vol haute, 5 events news. Pour chaque, écrire 1 ground-truth narrative attendue. Implémenter `eval_harness.py` qui : feed LLM via `llm_narrative_engine`, score sur 5 axes : factual_consistency (LLM-as-judge sur claims), reading_level (Flesch-Kincaid), forbidden_phrases (pas de "achetez", "vendez", "100% sûr"), source_attribution, brevity (<400 chars). Stocker `eval_results/{date}.json`. Ajouter à GitHub Actions.
- **DoD** : 50 fixtures + script + CI, eval baseline tournée et stockée.
- **KPI succès** : score baseline ≥ 0.75 sur factual_consistency, ≥ 0.95 sur forbidden_phrases (compliance), ≥ 0.80 brevity.
- **Critère kill** : si forbidden_phrases < 0.95 → bug compliance critique, escalation.
- **Référence** : Anthropic eval docs ; Liu et al. (2023) *G-Eval*.

**Synthèse Aisha Phase 1 :** L'eval harness sera réutilisé en Phase 2A et explosé en Phase 2B. Pas de RAG en Phase 1 — c'est conscient.

### Agent 5 — INFRASTRUCTURE & COMPLIANCE (Théo) — 6h

**Vision Phase 1 :** CI/CD GitHub Actions (eval_22 a relevé 0 GHA) et observabilité minimale (eval_16 score 3.2/10). Pas de compliance W4 — attente Phase 2.

#### Sprint INFRA-1.1 — GitHub Actions CI/CD — 3h
- **Owner** : Théo / **Phase** : 1 / **Effort** : 3h / **Dépendances** : aucune
- **Spec** : `.github/workflows/ci.yml` avec 3 jobs : (1) lint (ruff + black --check), (2) test (pytest avec `--cov=src --cov-fail-under=70`), (3) docker-build. Trigger : pull_request + push main. Cache pip + cache pytest. Job durée cible < 8 min.
- **DoD** : workflow vert sur main, badge ajouté à README.
- **KPI succès** : CI tourne en < 8 min, fail = block merge.
- **Critère kill** : si pytest CI échoue à cause de dépendances CSV → fixture `tests/fixtures/xau_mini.csv` (1000 lignes).
- **Référence** : GitHub Actions docs ; pytest-cov.

#### Sprint INFRA-1.2 — Observabilité minimale — 3h
- **Owner** : Théo / **Phase** : 1 / **Effort** : 3h / **Dépendances** : aucune
- **Spec** : (a) audit `print()` calls (109 dans 23 fichiers), remplacer par `logger.info` avec `LOG_FORMAT=json` ; (b) intégrer Sentry SDK (free tier) avec `SENTRY_DSN` env var ; (c) `prometheus_client` registry instancié au boot avec 3 métriques minimum : `signals_generated_total`, `llm_latency_seconds`, `circuit_breaker_open_total`. Endpoint /metrics fonctionnel.
- **DoD** : 0 print() dans `src/`, Sentry capture erreur test, /metrics retourne payload non-vide.
- **KPI succès** : Sentry reçoit 1 error test, /metrics ≥ 3 métriques, logs JSON parsables par jq.
- **Critère kill** : Sentry > free tier → désactiver si > 5k events/mois.
- **Référence** : Sentry Python SDK ; prometheus_client.

**Synthèse Théo Phase 1 :** CI/CD opérationnelle = filet de sécurité. Observabilité minimale = visibilité prod.

### Agent 6 — PRODUCT & UX DESIGN (Inès) — 5h

**Vision Phase 1 :** Format `InsightSignal` Pydantic v2 unifié (déjà draft `dual_b2c_b2b_architecture.md`). Indépendant du verdict A1 — sert dans les deux scénarios.

#### Sprint UX-1.1 — InsightSignal Pydantic v2 unifié + 4 mockups — 5h
- **Owner** : Inès / **Phase** : 1 / **Effort** : 5h / **Dépendances** : aucune
- **Spec** : créer `src/api/models/insight_signal_v2.py` Pydantic BaseModel avec : `id`, `instrument`, `timeframe`, `direction` (BUY/SELL/HOLD), `conviction_0_100`, `levels: {entry, stop, target_1, target_2}`, `narrative_short`, `narrative_long`, `sources_cited: List[Source]`, `compliance: {disclaimer_lang, jurisdiction_blocked}`, `created_at_utc`, `valid_until_utc`. Refactorer existing v1 en v2 avec backward-compat shim. Mettre à jour 4 mockups : `mockups/telegram_b2c.txt`, `mockups/webapp_b2c.html`, `mockups/b2b_insight.json`, `mockups/b2b_webhook_payload.json`.
- **DoD** : modèle v2 + 4 mockups cohérents + test round-trip vert.
- **KPI succès** : 1 modèle source de vérité, 4 surfaces dérivées, test round-trip OK.
- **Critère kill** : si v2 casse > 10 tests existants → adapter v1→v2 progressif.
- **Référence** : Pydantic v2 docs ; eval_29 compliance.

**Synthèse Inès Phase 1 :** InsightSignal v2 = "lingua franca" du produit. Webapp interactive arrive en 2B (35h) ou en 2A (15h plus léger).

### Agent 7 — COMMERCIAL & GROWTH (Karim) — 5h

**Vision Phase 1 :** UN livrable conscient : 2 briefs de positionnement, écrits AVANT le verdict A1. Pour ne pas que le résultat A1 biaise l'analyse marché. Aucun GTM en Phase 1.

#### Sprint COMM-1.1 — Brief positionnement A et B (préparé en amont) — 5h
- **Owner** : Karim / **Phase** : 1 / **Effort** : 5h / **Dépendances** : aucune
- **Spec** : produire 2 docs markdown dans `reports/positioning/` :
  - `positioning_2A_edge_confirmed.md` (2 pages) : audience cible (XAU traders FR-first $20-79/mo + brokers B2B-API), claims autorisés ("backtested edge CPCV-validated"), proof points (DSR, PBO publiables), pricing recommandé ($29 ANALYST, $79 STRATEGIST, $1990 INSTITUTIONAL d'après eval_27), GTM channel mix
  - `positioning_2B_narrative_first.md` (2 pages) : audience cible (apprenants trading FR + brokers B2B "data quality"), claims autorisés ("intelligence contextuelle, pas de signaux"), proof points (qualité narrative LLM, sources citées RAG), pricing recommandé ($19 LITE, $39 PRO, $99 PRO+, $499 B2B/mo), GTM channel mix
  - dans chaque doc : analyse 5 concurrents (TradingView Premium, Trade Ideas, FinChat, etc.)
- **DoD** : 2 docs livrés AVANT fin S6 (avant verdict A1).
- **KPI succès** : 2 docs revus par Sofia, conformes eval_29 (pas de claims interdits MiFID II 2026/2811).
- **Critère kill** : si docs trop similaires → escalation Sofia.
- **Référence** : eval_25, eval_27, eval_28, eval_29.

**Synthèse Karim Phase 1 :** Stratégique, pas tactique. Aucun outbound ni paid ni content en Phase 1. Patience disciplinée.

### Agent 8 — RISK & GOVERNANCE (Sofia) — 6h

**Vision Phase 1 :** Instancier le **kill criteria board** + 1h de surveillance/sem. Sans gouvernance, dérive garantie.

#### Sprint RISK-1.1 — Kill criteria board + checkpoint discipline — 6h
- **Owner** : Sofia / **Phase** : 1 / **Effort** : 6h (3h instanciation + 3h surveillance hebdo S2-S8) / **Dépendances** : INFRA-1.2, QUANT-1.3
- **Spec** : créer `reports/governance/kill_criteria_board.md` versionné, mis à jour vendredi 16h. Pour chaque sprint actif : owner, ETA, last update, status (🟢🟡🔴), kill criterion explicite, blockers. 4 KPIs critiques globaux. `tools/governance/weekly_check.py` pull /metrics + git stats + pytest pass count. Section "post-mortem A1" template à compléter fin S8.
- **DoD** : board en place, script tourne, post-mortem template prêt, 8 checks hebdo planifiés.
- **KPI succès** : 8/8 checkpoints hebdo réalisés (binaire).
- **Critère kill** : si 2 checkpoints ratés consec → réduire scope du board.
- **Référence** : López de Prado *Strategy Risk* ch. 15 *AFML* ; eval_19 risk findings.

**Synthèse Sofia Phase 1 :** Sans livrable spectaculaire mais critique. À fin S8, post-mortem A1 quel que soit le résultat — c'est l'artefact qui justifie la bascule 2A ou 2B.

## II.3 Diagramme de Gantt Phase 1 (semaines 1-8)

```
Sem  | S1     S2     S3     S4     S5     S6     S7     S8
-----+----------------------------------------------------------
Marwan| D1.1   D1.2   D1.3   .      .      .      .      .
Elena| .      .      Q1.1   Q1.2   Q1.2   Q1.3   Q1.3   Q1.3-VRD
Kenji| .      .      .      R1.1   R1.2   .      .      .
Aisha| .      L1.1   L1.1   L1.1   .      .      .      .
Théo | I1.1   I1.2   .      .      .      .      .      .
Inès | .      .      U1.1   U1.1   .      .      .      .
Karim| .      .      .      .      C1.1   C1.1   .      .
Sofia| RK1.1  surv.  surv.  surv.  surv.  surv.  surv.  PM-A1

Chemin critique : D1.1 → D1.2 → Q1.1 → Q1.2 → Q1.3 → VRD (~30h)
```

**Parallélisable** : Aisha (LLM-1.1) + Théo (INFRA) + Inès (UX-1.1) + Karim (COMM-1.1) sont 100% indépendants du chemin critique.

**Bottleneck identifié** : si Marwan glisse, Elena bloquée S3-S4 et Phase 1 dérape. **Mitigation** : Marwan finit S1-S2, 1 semaine de buffer pour Elena.

## II.4 Matrice des dépendances Phase 1

| Dépend ↓ | DATA | QUANT | REGIME | LLM | INFRA | UX | COMM | RISK |
|---|---|---|---|---|---|---|---|---|
| **DATA** | — | | | | | | | |
| **QUANT** | DATA-1.1, 1.2, 1.3 | — | | | | | | |
| **REGIME** | | QUANT-1.1 | — | | | | | |
| **LLM** | | | | — | | | | |
| **INFRA** | | | | | — | | | |
| **UX** | | | | | | — | | |
| **COMM** | | | | | | | — | |
| **RISK** | | QUANT-1.3 | | | INFRA-1.2 | | | — |

**Bottleneck d'agent** : Marwan (3 sprints en série) et Elena (3 sprints en série, dépendant de Marwan).

## II.5 Budget effort par agent Phase 1

| Agent | Heures | % du total | Justification |
|---|---|---|---|
| Elena (Quant) | 16h | 25% | Test décisif, charge max acceptée |
| Marwan (Data) | 12h | 19% | Inputs Elena |
| Kenji (Regime) | 8h | 12% | Feature régime + bavure prod |
| Aisha (LLM) | 6h | 9% | Eval harness uniquement |
| Théo (Infra) | 6h | 9% | CI/CD + obs minimale |
| Sofia (Risk) | 6h | 9% | Gouvernance |
| Inès (UX) | 5h | 8% | Format unifié |
| Karim (Commercial) | 5h | 8% | 2 briefs anticipés |
| **Total** | **64h** | **100%** | |

## II.6 Verdict A1 et critères de bascule (CP-A1 fin S8)

**Format du verdict** (`reports/a1_verdict_2026.md`) :

```
DSR (Deflated Sharpe Ratio) : ___ (cible > 1.0)
PBO (Probability of Backtest Overfitting) : ___ (cible < 0.3)
CPCV PF moyen : ___ (cible > 1.20)
CPCV PF p25 : ___ (cible > 1.05)
Holm-significant features (α=0.05) : ___ (cible ≥ 3)
DM test vs HAR baseline p-value : ___ (cible < 0.05)
DM test vs constant baseline p-value : ___ (cible < 0.01)
SHAP top-3 features : ___, ___, ___ (cible : ≥1 macro non-redondante avec price)
```

**Décision :**
- **GO Phase 2A** si : DSR>1.0 ET PBO<0.3 ET CPCV PF>1.20 ET ≥3 Holm ET ≥1 macro feature top-3 SHAP
- **GO Phase 2B** sinon. **Pas d'entre-deux** — discipline exécutée par Sofia.

**Synthèse globale Phase 1 :** 64h pour acheter une réponse honnête à une question coûteuse. Si verdict positif, on amortit en Phase 2A. Si négatif, les 64h ont quand même livré : eval harness LLM, CI/CD, observabilité, format InsightSignal v2, briefs commerciaux, kill criteria board. Aucune heure perdue.

---

# PARTIE III — Phase 2A (Mois 3-12, scénario edge confirmé)

## III.1 Synthèse exécutive Phase 2A

**Trigger d'entrée :** verdict A1 ✅ (DSR>1.0, PBO<0.3, CPCV PF>1.20, ≥3 Holm, ≥1 macro top-3 SHAP).

**Objectif :** capitaliser sur un edge algorithmique honnêtement validé pour construire un produit B2C premium + amorcer un canal B2B-API broker. **Discipline non négociable :** pas de monétisation avant 30 jours de forward-test paper avec PF live > 1.10 (CP-2A.1, mois 4).

**Heures totales :** 320h sur 40 semaines actives.

**Coût annualisé :**
- Polygon Starter ou Alpaca live data : ~30€/mois × 10 mois = 300€
- Hébergement Railway/Render : ~20€/mois × 12 = 240€
- LLM Anthropic API : variable (~100-300€/mois en régime)
- Domaine + Cloudflare + Sentry + Plausible : 30€/an
- **Total fixe : ~600€/an** (au-dessus du budget < 500€ — justification : data feed live non-négociable en 2A car claim "edge live-validated" requiert data live)

**Pricing recommandé Phase 2A :**

| Tier | Prix | Cible | Justification |
|---|---|---|---|
| FREE | 0€ | acquisition | 1 signal/jour, 24h delay, narrative court |
| **ANALYST** | **29€/mo** | retail FR core | signaux temps réel XAU+EURUSD, narrative complet |
| **STRATEGIST** | **79€/mo** | power users | + multi-TF + 6 instruments + alertes Telegram personnalisées + API key |
| INSTITUTIONAL | 199€/mo (tier décoratif) | rares acheteurs | + accès historique + support |
| **B2B-API** | **1500-3000€/mo** | brokers/IBs | 1k-10k req/mois, SLA 99,5%, white-label optionnel |

**MRR cibles Phase 2A :**

| Mois | MRR B2C | MRR B2B | MRR total | P(hit) |
|---|---|---|---|---|
| **M6** | 1500-2500€ (30-50 ANALYST + 5-10 STRATEGIST) | 0€ | **2000€** | **55%** |
| **M9** | 4000-6000€ (80-120 ANALYST + 15-20 STRATEGIST) | 1500€ (1 LOI signée) | **6500€** | **40%** |
| **M12** | 8000-11000€ (150-200 utilisateurs payants) | 3000€ (2 contrats B2B) | **11000-14000€** | **30%** |

**P(hit MRR M12 ≥ 10k€) ≈ 30%** — les 70% restants se répartissent entre "MRR plus modeste mais business viable" (~50%) et "edge live se dégrade, repli partiel sur Phase 2B" (~20%).

**KPI globaux Phase 2A :**
- ✅ Forward-test paper 30-60j PF live ≥ 1.10 (CP-2A.1, kill criterion)
- ✅ A1 v2 production-deployed avec drift monitoring PSI < 0.2
- ✅ B2C : 150+ users payants M12
- ✅ B2B-API : 2 contrats signés M12, 1 broker actif
- ✅ Webapp MVP live + Telegram opérationnel
- ✅ Multi-asset (XAU+EURUSD min) en production
- ✅ Compliance W4 closed (relecture juridique CGU)

**Critères kill par mois :**
- M3 : si forward-test paper PF < 0.85 sur 30 jours → repli partiel 2B
- M5 : si M5 = 0 user payant → revoir pricing/positionnement
- M9 : si MRR M9 < 1500€ → réduire ambition B2B, doubler effort B2C
- M11 : si zéro LOI B2B après 10 prospects outbound → drop canal B2B fin 2A

## III.2 Section par agent Phase 2A

### Agent 1 — Marwan (Data) — 50h

**Vision Phase 2A :** 4 choses : (1) basculer XAU sur feed live qualité prod, (2) étendre EURUSD au niveau de XAU pour valider transferabilité de l'edge, (3) ajouter sentiment LLM-extrait, (4) audit trail immutable pour B2B.

#### Sprint DATA-2A.1 — Feed live XAU production-grade — 8h
- **Spec** : remplacer lecture CSV par feed live via Polygon Starter ($29/mo, FX/CFD inclus) ou Alpaca CDA. `src/agents/data/live_feed_xau.py` avec WebSocket + buffer ring 10k bars + reconnect auto + heartbeat. Persister SQLite `data/live/xau_m15.db`. Reconciliation quotidienne avec Dukascopy CSV.
- **DoD** : 7 jours uptime sans gap, reconcil delta < 0.05% des bars.
- **KPI** : disponibilité ≥99,5%, latence ingestion < 2s post-bar-close.
- **Kill** : si Polygon spread XAU > 0.5$ moyen → switch Alpaca.

#### Sprint DATA-2A.2 — Macro freshness monitor + alertes — 6h
- **Spec** : cron horaire vérifie FRED ≤26h, COT ≤8j, GLD ≤26h. Si stale → alerte Sentry + flag `macro_stale=true`.
- **DoD** : cron horaire, alerte test forcée OK.
- **KPI** : détection stale en < 1h.

#### Sprint DATA-2A.3 — Extension EURUSD multi-asset — 8h
- **Spec** : appliquer DATA-2A.1 à EURUSD M15. Macro EUR : ECB rate (ECBDFR), Bund 10y, ESI sentiment. Adapter `cot_provider.py` pour code EURUSD (099741).
- **DoD** : EURUSD live + macro EUR ingérés, tests vintage OK.
- **KPI** : ≥6 mois historique + live.
- **Kill** : si Elena ne valide pas transfer A1→EURUSD → freeze.

#### Sprint DATA-2A.4 — Sentiment LLM-extrait — 8h
- **Spec** : pipeline pull headlines FF + Reuters RSS gold + Bloomberg gold tag toutes les 15min. Classification "relevant" via Claude Haiku (~5€/mo). Si relevant : extraction `{sentiment: -1..+1, asset_impact, time_horizon}`. Persister `data/sentiment/xau_sentiment.parquet`. Aggregate horaire EWMA half-life=4h.
- **DoD** : 30 jours d'historique, eval manuel 50 headlines accuracy ≥ 75%.
- **KPI** : feature `sentiment_ewma_4h` ajoutée matrice A1.
- **Kill** : si SHAP montre importance < 1% → freeze pour 2A, garder pour 2B RAG.

#### Sprint DATA-2A.5 — Data quality dashboard — 6h
- **Spec** : panel Grafana : feed uptime XAU+EUR, gaps détectés 24h, macro freshness, sentiment volume, COT lag.
- **DoD** : dashboard live, screenshot dans `reports/ops/`.
- **KPI** : 7 panels minimum, refresh 1min.

#### Sprint DATA-2A.6 — Backup feed redondance — 6h
- **Spec** : feed secondaire passif (yfinance 15min delayed pour fallback). En cas de Polygon down >5min, basculer auto sur secondary avec flag `feed=secondary`.
- **DoD** : test fail-over réussit en < 30s.
- **KPI** : 0 trou >5min dans 30 jours.

#### Sprint DATA-2A.7 — Audit trail B2B compliance — 8h
- **Spec** : pour chaque signal émis, persister `data/audit/signals_audit.parquet` (append-only, hash-chaîné SHA256) : `signal_id, timestamp_emit, instrument, direction, conviction, levels, model_version, feature_snapshot_md5, narrative_md5, sources_used`. Endpoint B2B `/audit/{signal_id}`.
- **DoD** : 30 jours d'audit accumulés, hash chain validé.
- **KPI** : 100% des signaux audités.
- **Référence** : MiFID II Art. 16(7) record-keeping.

**Synthèse Marwan 2A :** Stack data prod-grade : 2 instruments live, macro live monitorée, sentiment intégré, redondance, audit trail signable. Soutient claims B2B et tier B2B-API.

### Agent 2 — Elena (Quant) — 70h

**Vision Phase 2A :** Transformer un edge "validé en backtest CPCV" en edge "validé en live + scalable". Trois pivots : (1) durcir A1 v1 → v2 production avec drift monitoring, (2) prouver transfer EURUSD pour multi-asset, (3) calibrer la sortie probabiliste. Je m'interdis d'ajouter des features sans Holm.

#### Sprint QUANT-2A.1 — A1 v2 production hardening — 12h
- **Spec** : industrialiser stack A1 v1. Export ONNX, inférence < 50ms p99, refresh quotidien (re-fit fenêtre roulante 36 mois, PSI features check). Versioning `models/a1_v{semver}/` avec metadata.json. Rollback automatique si PSI > 0.25 ou perf rolling 7j < 0.8 × baseline.
- **DoD** : modèle déployé, latence vérifiée, rollback testé.
- **KPI** : 0 incident drift non détecté en 90 jours.
- **Kill** : si refresh dégrade perf (CPCV PF < 1.10) → fenêtre fixe.

#### Sprint QUANT-2A.2 — Walk-forward live monitoring — 8h
- **Spec** : compute rolling DSR + PBO sur fenêtre live. Stocker chaque signal avec prédiction et outcome (forward 4h/16h). Hebdo : `reports/live_perf/week_{N}.md` avec rolling Sharpe, PF, DSR, hit rate par bucket.
- **DoD** : rapport hebdo automatique, alerte si rolling PF 30j < 1.0.
- **KPI** : rolling DSR 90j > 0.5.
- **Kill** : forward 30j PF < 0.85 → CP-2A.1 kill, repli 2B.

#### Sprint QUANT-2A.3 — A1 v2 features étendues — 10h
- **Spec** : ajouter à A1 v2 : (a) sentiment EWMA, (b) HAR-PD-REQ vol forecast résidu, (c) régime jump-model state probs, (d) COT delta z52. Pour chaque, refit CPCV+Holm. Conserver UNIQUEMENT celles qui passent Holm α=0.05 sur ≥2 horizons.
- **DoD** : `reports/a1_v2_feature_eval.md` avec décisions documentées.
- **KPI** : ≥1 nouvelle feature passe Holm.
- **Kill** : 0 feature passe → garder A1 v1.

#### Sprint QUANT-2A.4 — Position sizing algo — 10h
- **Spec** : ajouter à chaque signal `position_size_recommended_pct_equity` calculé par Kelly fractional (0.25 × Kelly) plafonné par vol-target ATR (max 1% equity at ATR distance). Sortie B2B-API uniquement (B2C reste pédagogique). Backtest 6 ans : compare equal-weight vs sized.
- **DoD** : champ ajouté à InsightSignal v2, backtest comparison report.
- **KPI** : sized > equal-weight Sharpe sur backtest.
- **Référence** : Thorp (2006) Kelly Criterion ; Vince *Mathematics of Money Management*.

#### Sprint QUANT-2A.5 — Transfer A1 → EURUSD validation — 12h
- **Spec** : appliquer pipeline A1 stack à EURUSD avec features adaptées (Bund 10y au lieu de DGS, COT EURUSD). Run CPCV harness sur EURUSD. Évaluer : (a) modèle XAU appliqué tel quel (transfer naïf), (b) modèle EURUSD réentraîné. Si transfer naïf passe DSR>0.5, edge généralisable.
- **DoD** : `reports/eurusd_transfer_eval.md`.
- **KPI** : DSR EURUSD > 0.5.
- **Kill** : DSR EURUSD < 0 → drop EURUSD pour 2A, focus XAU only.

#### Sprint QUANT-2A.6 — Score → calibrated probability — 8h
- **Spec** : transformer sortie LightGBM en probabilité calibrée via isotonic regression (bins fold-out CPCV). Afficher "78% prob de mouvement >0.5R sur 4h". Reliability diagram : ECE < 0.05.
- **DoD** : calibrator persisté, reliability diagram.
- **KPI** : ECE < 0.05 sur fold-out.
- **Référence** : Niculescu-Mizil & Caruana (2005) ; sklearn CalibratedClassifierCV.

#### Sprint QUANT-2A.7 — Forward-test analysis report mensuel — 10h (étalé)
- **Spec** : chaque dernier dimanche du mois, `reports/monthly_perf/{YYYY}-{MM}.md` avec signaux émis, hit rate, R-multiples, comparaison live vs backtest CPCV, drift PSI, narrative qualitative.
- **DoD** : 10 rapports mensuels livrés.
- **KPI** : 10/10 rapports.

**Synthèse Elena 2A :** 70h installent une discipline rare en retail : edge live-monitoré, drift-detecté, calibré, multi-asset validé.

### Agent 3 — Kenji (Regime/Vol) — 45h

**Vision Phase 2A :** Couche régime/vol scientifique : HAR-PD-REQ pour vol forecasting SOTA, Statistical Jump Model pour régimes discrets, ensemble régime-conditionnel. Aucun TSFM, aucun RNN.

#### Sprint REGIME-2A.1 — HAR-PD-REQ implementation — 12h
- **Spec** : Patton-Sheppard 2015 Realized Quarticity-adjusted HAR avec Positive/negative jumps decomposition. RV à 5min sur XAU. Comparer à HAR baseline. Export ONNX, latence p99 < 100ms.
- **DoD** : RMSE OOS HAR-PD-REQ ≤ 0.95 × HAR (5%+ amélioration).
- **KPI** : RMSE -5% min, latence respectée.
- **Référence** : Patton & Sheppard (2015) *Good Volatility, Bad Volatility*.

#### Sprint REGIME-2A.2 — Statistical Jump Model — 12h
- **Spec** : Nystrup-Lindström-Madsen 2020. 3 régimes (low-vol trending, low-vol ranging, high-vol stress). Output `regime_state` + `regime_confidence`. Online forward filtering (no Viterbi backward pour éviter look-ahead).
- **DoD** : modèle convergent sur 6 ans XAU, états interprétables.
- **KPI** : in-sample log-likelihood > HMM baseline.
- **Référence** : Nystrup et al. (2020) Expert Systems with Applications.

#### Sprint REGIME-2A.3 — Régime-conditional A1 ensemble — 10h
- **Spec** : entraîner 3 sous-modèles A1 (un par régime) ; weighted average par régime confidence. CPCV pour valider amélioration vs A1 unique.
- **DoD** : `reports/regime_ensemble_eval.md`.
- **KPI** : DSR ensemble > DSR A1 single d'au moins 0.2.
- **Kill** : pas d'amélioration → garder A1 single, exposer régime comme contexte narrative.

#### Sprint REGIME-2A.4 — Realized Volatility intraday estimator — 6h
- **Spec** : compute RV à 5min sur XAU rolling 1h, 4h, 24h. Stocker comme features pour A1 et narrative.
- **DoD** : 3 features RV ajoutées, tests latence OK.

#### Sprint REGIME-2A.5 — Vol forecasting eval extended — 5h
- **Spec** : étendre eval vol au modèle EURUSD post DATA-2A.3. Compare HAR vs HAR-PD-REQ vs naïf. Publier en transparence sur webapp.
- **DoD** : `reports/vol_forecast_eval.md` mis à jour.

**Synthèse Kenji 2A :** SOTA légitime sans sacrifier simplicité. Gain attendu : +5-10% RMSE vol, +0.2 DSR sur ensemble régime.

### Agent 4 — Aisha (LLM) — 35h

**Vision Phase 2A :** Améliorer qualité narrative sans investir massivement en RAG. En 2A, narrative est secondaire (l'edge est le héros). Focus : tone par tier, multi-langue minimal, cost monitoring, régression CI.

#### Sprint LLM-2A.1 — Narrative tone tuning par tier — 6h
- **Spec** : 3 prompts différenciés FREE/ANALYST/STRATEGIST. FREE = 80 mots, ANALYST = 200 mots, STRATEGIST = 350 mots avec scénarios alternatifs. Eval harness vérifie tier-token-budget.
- **DoD** : `prompts/narrative_v3_{tier}.md`, eval vert.
- **KPI** : token budget ±10% sur 50 prompts test.

#### Sprint LLM-2A.2 — Eval harness extension à 100 prompts — 6h
- **Spec** : doubler fixtures à 100. Ajouter axe "compliance_check" (LLM-as-judge applique checklist eval_29 reformulation MiFID 2024/2811).
- **DoD** : 100 fixtures, CI mise à jour.
- **KPI** : score moyen ≥ 0.78.

#### Sprint LLM-2A.3 — Cache sémantique v2 — 6h
- **Spec** : refacto SemanticCache pour vrai sémantique : embedding query (Voyage AI ~0.02$/1M tok) → cosine > 0.92 + same-bucket score → hit. Bump `SCORE_BUCKET_PTS` 5→10 (eval_05_09_refresh ×4.3 hit rate).
- **DoD** : cache hit rate live > 25% sur 7 jours.
- **KPI** : économie LLM ≥ 30%.
- **Kill** : si embedding cost > économie LLM → désactiver.

#### Sprint LLM-2A.4 — Multi-langue EN→FR — 8h
- **Spec** : prompts FR primaire + EN secondaire. Détection langue via TelegramLangStore. Eval 100 prompts × 2 langues.
- **DoD** : FR+EN couverts, eval ≥ 0.75 par langue.
- **KPI** : 0 divergence factuelle EN/FR > 5%.

#### Sprint LLM-2A.5 — Régression CI gate — 4h
- **Spec** : ajouter eval_llm au CI GitHub Actions. Si score baseline drop > 5% → block merge. Cost CI ≤ 5€/run.
- **DoD** : workflow vert, alerte test forcée OK.

#### Sprint LLM-2A.6 — Cost monitoring + budget caps — 5h
- **Spec** : daily LLM cost tracker via Anthropic API usage endpoint. Hard cap mensuel par tier. Si dépassement → fallback template + alerte Sentry.
- **DoD** : dashboard cost Grafana, hard cap testé.
- **KPI** : cost actual ≤ 1.2× budget prévu.

**Synthèse Aisha 2A :** Base narrative solide sans RAG (réservé 2B). Cache sémantique + multi-langue + compliance check = STRATEGIST défendable.

### Agent 5 — Théo (Infra) — 45h

**Vision Phase 2A :** 4 livrables structurants : ONNX serving, forward-test paper trading harness (gate de monétisation), Stripe + tier enforcement, compliance W4 closed.

#### Sprint INFRA-2A.1 — ONNX model serving — 8h
- **Spec** : `src/intelligence/model_serving.py` charge ONNX au boot, expose `predict(features) -> CalibratedScore` thread-safe. Background reload sur model file change.
- **DoD** : test pytest latence vert, hot-reload testé.
- **KPI** : p99 < 50ms.

#### Sprint INFRA-2A.2 — Forward-test paper harness — 12h ⚠️ GATE
- **Spec** : harness simule compte paper $10k, applique signaux temps réel, track equity, R-multiples, drawdown, Sharpe rolling 30j. Stocker `data/forward_test/equity.parquet`. Endpoint `/forward-test/snapshot`. **GATE NON-NÉGOCIABLE** : `Stripe live = false` tant que Sofia ne valide pas `forward_test_30d_PF >= 1.10` (RISK-2A.1).
- **DoD** : harness tourne 24/7 dès M3, gate Stripe codé en dur.
- **KPI** : 0 monétisation avant forward-test gate validé.
- **Kill** : forward 60j PF < 0.85 → CP-2A.1 kill, repli 2B.

#### Sprint INFRA-2A.3 — Stripe + tier enforcement — 10h
- **Spec** : Stripe Checkout (FREE→ANALYST€29, ANALYST→STRATEGIST€79, INSTITUTIONAL€199). Webhook `customer.subscription.updated` → MAJ `tier_manager.py`. Trial 14j sans carte (eval_27 +$1168 MRR vs freemium-only). Middleware tier enforcement.
- **DoD** : 3 tiers payables, trial OK, upgrade/downgrade testés.
- **KPI** : 0 utilisateur sur tier supérieur sans paiement valide.
- **Kill** : si tax compliance UE complexe → utiliser Lemon Squeezy (5% fee mais MoR).

#### Sprint INFRA-2A.4 — Compliance W4 legal review — 6h
- **Spec** : engager avocat fintech FR pour relire CGU + politique confidentialité + claims marketing 2A. Budget 800-1500€. Livrable : doc validé + checklist red-flags marketing.
- **DoD** : CGU validée par avocat, signée par solo founder.
- **KPI** : 0 mention legal-risk dans claims marketing.
- **Kill** : si avocat dit "vous êtes en activité réglementée AMF" → STOP, ré-évaluer modèle.

#### Sprint INFRA-2A.5 — Backup + DR — 5h
- **Spec** : SQLite backup quotidien vers Backblaze B2 (5€/mo) avec rotation 30j. Test restore mensuel. Documenter RTO/RPO (RTO < 4h, RPO < 24h).
- **DoD** : 1 restore réussi en test.

#### Sprint INFRA-2A.6 — Rate limit per-tier — 4h
- **Spec** : middleware Redis (ou in-memory pour < 1000 users) rate limit FREE 60req/h, ANALYST 600/h, STRATEGIST 6000/h, B2B custom. Headers RFC 6585.
- **DoD** : test pytest rate limit vert.

**Synthèse Théo 2A :** Industrialise edge en produit vendable. INFRA-2A.2 (forward-test gate) = digue éthique. Compliance W4 ferme dernier risque légal.

### Agent 6 — Inès (UX) — 30h

**Vision Phase 2A :** Webapp MVP narrative-light + onboarding + Telegram polish. En 2A héros = signal/edge → UX sobre orientée "preuve de performance" : equity curve transparente, R-multiples, hit rate.

#### Sprint UX-2A.1 — Webapp dashboard MVP — 12h
- **Spec** : webapp Next.js (ou Streamlit) déployée sur Vercel. Pages : (1) Landing FR avec equity curve forward-test live, (2) Dashboard logged-in : derniers signaux, equity, perf 30j/90j/1y, (3) Tier upgrade. Minimaliste — Plotly only.
- **DoD** : webapp live à `app.smartsentinel.ai`, 4 pages fonctionnelles.
- **KPI** : Lighthouse perf ≥ 85.
- **Kill** : Next.js > 15h → rabattement Streamlit (-50% temps).

#### Sprint UX-2A.2 — Onboarding flow — 6h
- **Spec** : signup → email magic link (no password) → tier selection → Telegram bot link → first signal received. Mesurer drop-off chaque étape.
- **DoD** : 5 utilisateurs test font flow bout-en-bout sans assistance.
- **KPI** : drop-off step-to-step < 30%.

#### Sprint UX-2A.3 — Telegram UX upgrades — 6h
- **Spec** : commandes nouvelles : `/perf` (perf 30j), `/history` (10 derniers signaux), `/stop` (pause notif), `/settings`, `/upgrade`. Inline buttons. Markdown clean.
- **DoD** : 5 commandes opérationnelles, tests pytest.
- **KPI** : NPS interne (5 testers) ≥ 7/10.

#### Sprint UX-2A.4 — B2B API documentation — 6h
- **Spec** : OpenAPI 3.1 spec complète, hosted Swagger UI, Postman collection, code examples (Python, JS, curl) pour 5 endpoints clés.
- **DoD** : doc en ligne, 1 dev externe peut intégrer en < 30min sans aide.

**Synthèse Inès 2A :** UX cohérente minimaliste pour soutenir conversion FREE→ANALYST→STRATEGIST. Pas de chat narratif (2B), pas de mobile native.

### Agent 7 — Karim (Commercial) — 30h

**Vision Phase 2A :** SEO foundation FR (slow build composé), outbound brokers (high-leverage), community minimal. Pas de paid ads (eval_28 : pas de runway).

#### Sprint COMM-2A.1 — SEO foundation FR — 6h
- **Spec** : 5 pages cornerstone SEO FR : "analyse XAU/USD intraday", "trading or M15 SMC", "signaux trading or vs forex", "robot trading or open source", "comprendre les COT or". Cible KD<20.
- **DoD** : 5 pages publiées, 2000+ mots chacune, schemas.org Article.
- **KPI** : 1 page top 10 Google FR à M9.

#### Sprint COMM-2A.2 — Content cadence M3-M12 — 8h (étalé)
- **Spec** : 1 article/2 semaines = 12 articles M3-M12. Format : "marché XAU semaine N analysé" + édu "qu'est-ce que [concept]". Reuse narratives LLM + edits.
- **DoD** : 12 articles publiés.
- **KPI** : trafic organique blog 500+ visites/mo à M12.

#### Sprint COMM-2A.3 — Outbound brokers wave 1 — 6h
- **Spec** : 5 prospects ciblés : IC Markets, Pepperstone, Exness, FP Markets, Tickmill. LinkedIn + email IB liaison contact. Pitch : "white-label XAU intraday signals API, 3000€/mo, edge live-validated, audit-trail signable". Track Airtable.
- **DoD** : 5 contacts, ≥ 2 réponses, ≥ 1 demo bookée.
- **KPI** : 1 LOI signée à M9.
- **Kill** : si 0 réponse sur 5 → revue pitch + Sofia.

#### Sprint COMM-2A.4 — B2B API pilot LOI — 6h
- **Spec** : pour le 1er broker intéressé, négocier pilot 60j gratuit → conversion 1500€/mo. Contrat simple (DocuSign / HelloSign free), juridiction Québec.
- **DoD** : 1 contrat signé.
- **KPI** : 1 contrat M9 minimum.
- **Kill** : si broker exige certif AMF/SEC → refuser, refocus B2B-data-quality.

#### Sprint COMM-2A.5 — Community Discord public — 4h
- **Spec** : Discord server FR + EN, 5 channels. Telegram channel public broadcast. 1 message/sem minimum.
- **DoD** : 100 membres Discord à M12.
- **KPI** : 50 actifs/sem M12.

**Synthèse Karim 2A :** SEO+content+outbound+community en mode minimum-viable. MRR cible M12 = 11k€ avec 1-2 contrats B2B atteignable mais P(hit)=30%.

### Agent 8 — Sofia (Risk) — 15h

**Vision Phase 2A :** Light mais critique. Gardienne de la discipline forward-test (CP-2A.1, gate non-négociable), drift monitoring, reviews trimestrielles, compliance des claims.

#### Sprint RISK-2A.1 — Forward-test gate ⚠️ NON-NÉGOCIABLE — 4h
- **Spec** : check hebdo automatisé. Tant que `forward_test_age_days < 30` OU `forward_test_PF_30d < 1.10` → flag `monetization_locked=true` (lu par Stripe webhook). Reporting public sur webapp dès J1 (transparence). Si gate franchie → Sofia signe `reports/governance/cp_2a1_decision.md` autorisant Stripe ouverture. Si gate échoue M4 → kill 2A et bascule 2B partiel.
- **DoD** : gate codé, 1 check hebdo automatique, décision M4 archivée.
- **KPI** : Stripe ouvre seulement après gate franchie.
- **Kill** : forward-test PF 60j < 0.85 → kill 2A.

#### Sprint RISK-2A.2 — Live drift monitor PSI — 5h
- **Spec** : Population Stability Index sur 18 features A1, calculé hebdo. Si PSI > 0.25 sur ≥ 1 feature → alerte ; si > 0.5 → block trading flag, rollback model.
- **DoD** : dashboard PSI live.

#### Sprint RISK-2A.3 — Quarterly review framework — 3h (1h × 3 trimestres)
- **Spec** : revue trimestrielle Q2/Q3/Q4 : KPIs vs target, kill criteria status, post-mortem sprints abandonnés. `reports/quarterly/Q{N}.md`.
- **DoD** : 3 reviews écrites.

#### Sprint RISK-2A.4 — Incident response playbook — 3h
- **Spec** : runbook : (1) Polygon down → fallback yfinance + comm Discord, (2) modèle drift → rollback + comm transparente, (3) Telegram bot down → status page, (4) data leak suspecté → freeze Stripe + audit. Communication templates pré-écrits.
- **DoD** : runbook 1 page, 4 scénarios.

**Synthèse Sofia 2A :** 15h pour empêcher solo founder de se mentir et réagir vite. RISK-2A.1 forward-test gate est ma raison d'être en 2A.

## III.3 Diagramme de Gantt Phase 2A (mois 3-12)

```
M3 (S9-12)     | Marwan: D2A.1, D2A.2  | Elena: Q2A.1 | Théo: I2A.1, I2A.2 (forward-test ON) | Sofia: R2A.1
M4 (S13-16)    | Marwan: D2A.3        | Elena: Q2A.2 | Kenji: REG2A.1               | CP-2A.1 ⚠️
M5 (S17-20)    | Marwan: D2A.4, D2A.7 | Elena: Q2A.6 | Kenji: REG2A.2               | Théo: I2A.3 Stripe ON
M6 (S21-24)    | Inès: U2A.1, U2A.2   | Aisha: L2A.1, L2A.3 | Karim: C2A.1     | Sofia: R2A.2
M7 (S25-28)    | Marwan: D2A.5, D2A.6 | Elena: Q2A.3 | Kenji: REG2A.3               | Théo: I2A.4 (legal)
M8 (S29-32)    | Inès: U2A.3, U2A.4   | Aisha: L2A.2, L2A.4 | Karim: C2A.2 (rolling)| Sofia: R2A.3 Q
M9 (S33-36)    | Elena: Q2A.5         | Kenji: REG2A.4 | Karim: C2A.3 (outbound)       | CP-2A.2 ⚠️
M10 (S37-40)   | Elena: Q2A.4         | Aisha: L2A.5, L2A.6 | Karim: C2A.4 (LOI)        |
M11 (S41-44)   | Marwan: ops          | Elena: Q2A.7 (rolling) | Karim: C2A.5         | Sofia: R2A.4
M12 (S45-52)   | Théo: I2A.5, I2A.6   | Inès: polish | Karim: C2A.2 fin                | Sofia: review annuelle

Chemin critique: D2A.1 → I2A.1 → I2A.2 (forward-test) → CP-2A.1 (gate M4) → I2A.3 → C2A.3 → C2A.4 (LOI M9)
```

## III.4 Matrice des dépendances Phase 2A

| Sprint | Dépend de |
|---|---|
| QUANT-2A.1 | INFRA-2A.1 |
| QUANT-2A.2 | QUANT-2A.1, INFRA-2A.2 |
| QUANT-2A.3 | DATA-2A.4, REGIME-2A.1 |
| QUANT-2A.5 | DATA-2A.3 |
| REGIME-2A.3 | REGIME-2A.2, QUANT-2A.3 |
| INFRA-2A.2 | INFRA-2A.1 |
| INFRA-2A.3 | INFRA-2A.2 (gate) |
| UX-2A.1 | INFRA-2A.2 |
| UX-2A.2 | UX-2A.1 |
| UX-2A.4 | DATA-2A.7, INFRA-2A.3 |
| COMM-2A.1 | UX-2A.1 |
| COMM-2A.3 | INFRA-2A.2 (forward-test 30j passé) |
| COMM-2A.4 | COMM-2A.3 |
| RISK-2A.1 | INFRA-2A.2 |
| RISK-2A.2 | QUANT-2A.1 |
| LLM-2A.4 | LLM-2A.1, INFRA-2A.4 |

**Bottleneck** : INFRA-2A.2 (forward-test) gate 6 sprints downstream. Si Théo glisse M3 → tout glisse. Mitigation : INFRA-2A.2 priorisé en M3.

## III.5 Budget effort par agent Phase 2A

| Agent | Heures | % | Justification |
|---|---|---|---|
| Elena | 70h | 22% | Edge à industrialiser, multi-asset à valider |
| Marwan | 50h | 16% | Live data + extension EUR + audit B2B |
| Théo | 45h | 14% | ONNX serving + Stripe + compliance + DR |
| Kenji | 45h | 14% | HAR-PD-REQ + Jump Model + ensemble |
| Aisha | 35h | 11% | Tone tuning + cache + multi-langue |
| Inès | 30h | 9% | Webapp + Telegram + B2B doc |
| Karim | 30h | 9% | SEO + content + outbound + community |
| Sofia | 15h | 5% | Gouvernance + drift + reviews |
| **Total** | **320h** | **100%** | |

---

# PARTIE IV — Phase 2B (Mois 3-12, scénario edge non démontré)

## IV.1 Synthèse exécutive Phase 2B

**Trigger d'entrée :** verdict A1 ❌ (PBO≥0.5 OU DSR<1.0 OU CPCV PF<1.20 OU <3 Holm). **Probabilité a priori 65-75%** — scénario le plus probable.

**Repositionnement produit :** Smart Sentinel AI passe de "signal trading edge-validated" à **"intelligence contextuelle pour traders auto-dirigés"** — un produit qui :
1. **N'affirme pas posséder un edge** (compliance + honnêteté)
2. Délivre **narrative LLM riche, sourcée, RAG-backed** sur l'état du marché XAU
3. Aide à **mieux comprendre** chaque set-up sans dire "achetez/vendez"
4. Pour le B2B, devient un **service de "qualité de signal augmentée"** : on enrichit les signaux d'un broker/EA avec contexte macro+sentiment+régime
5. **Transparence radicale** : forward-test paper publié, pas de claim de perf

**Cette Phase 2B n'est PAS un plan B dégradé.** C'est un produit narratif différent qui adresse un marché potentiellement plus large avec un ARPU plus modeste mais un plafond plus haut grâce à RAG et au B2B "data quality".

**Heures totales :** 320h sur 40 semaines.

**Coût annualisé :**
- Hosting Railway/Render : 240€/an
- LLM API (intensif RAG) : 200-500€/mois variable (le coeur du coût)
- Embeddings (Voyage AI) : ~30€/mois
- Domaine + Sentry + Plausible : 30€/an
- **Coût fixe : ~350€/an** + LLM variable scalant avec MRR
- Pas de Polygon live ($300/an économisés vs 2A) — yfinance 15min delayed suffit
- Avocat compliance W4 : 800€ ponctuel

**Pricing recommandé Phase 2B (positionnement éducatif/contextuel) :**

| Tier | Prix | Cible | Justification |
|---|---|---|---|
| FREE | 0€ | acquisition large | Newsletter weekly + 1 narrative/jour delayed |
| **LITE** | **19€/mo** | apprenants débutants | Webapp daily narrative + glossaire |
| **PRO** | **39€/mo** | traders auto-dirigés | + Q&A chat + signaux paper transparents + multi-asset |
| **PRO+** | **99€/mo** | power users | + Telegram alerts + accès historique + multi-langue |
| **B2B Data Quality** | **499-1500€/mo** | brokers, EA dev, copy-trading | API "enrichis nos signaux avec contexte" |

**MRR cibles Phase 2B :**

| Mois | MRR B2C | MRR B2B | MRR total | P(hit) |
|---|---|---|---|---|
| **M6** | 1500-2500€ (50-80 LITE+PRO) | 0€ | **2000€** | **50%** |
| **M9** | 3500-5000€ (130 utilisateurs payants) | 800€ (1-2 deals B2B small) | **4500€** | **45%** |
| **M12** | 6500-9000€ (200-280 utilisateurs) | 2000€ (2-3 deals B2B) | **8500-11000€** | **40%** |

**P(hit MRR M12 ≥ 8k€) ≈ 40%**, supérieur à 2A car la barre est plus basse.

**KPI globaux Phase 2B :**
- ✅ RAG architecture opérationnelle, F1 sourcing > 0.85
- ✅ Webapp narrative-first live à M5
- ✅ Forward-test paper PUBLIÉ en transparence (pas un gate, une feature marketing)
- ✅ B2C : 200+ users payants M12
- ✅ B2B-API "data quality" : 2 contrats signés M12
- ✅ Multi-langue FR+EN+DE (+ES si bandwidth)
- ✅ 12+ articles SEO + 12+ vidéos YouTube
- ✅ Compliance W4 closed avec finfluencer 2026 ready

**Critères kill par mois :**
- M5 : si webapp launch retardé > 30j → priorité Inès+Théo, geler tout sauf RAG
- M6 : si 0 conversion FREE→LITE après 200 signups → revoir pricing/onboarding
- M9 : si MRR < 2500€ → réduire ambitions B2B, doubler B2C/SEO
- M11 : si LLM cost > 60% revenue → optimisation cache + Haiku-first urgent

## IV.2 Section par agent Phase 2B

### Agent 1 — Marwan (Data) — 35h

**Vision Phase 2B :** Pivote : moins de live data (pas critique sans claim edge), plus de sources narrative-rich. En 2B, donnée nourrit narrative, pas modèle.

#### Sprint DATA-2B.1 — Corpus news étendu — 8h
- **Spec** : aggregator RSS : ForexFactory + Reuters + Bloomberg gold + WSJ markets + Kitco + FT commodities. Stockage append-only `data/news/{date}/{source}.jsonl`. Dedup by title hash. ~50 articles/jour.
- **DoD** : 30 jours d'historique, dedup OK.
- **KPI** : 5+ sources, 95% uptime ingest.

#### Sprint DATA-2B.2 — Sentiment LLM-extrait + sourced — 8h
- **Spec** : pour chaque article, Claude Haiku extrait `{sentiment, asset_impact, time_horizon, key_facts: [{claim, source_paragraph}], confidence}`. Différent de DATA-2A.4 : on conserve `key_facts` avec paragraphe source pour RAG (Aisha).
- **DoD** : 30 jours sentiment + facts indexés.
- **KPI** : 80% des articles ≥ 1 fact actionnable.
- **Kill** : si Haiku hallucine sources (test 10 facts) → switch Sonnet 4.6.

#### Sprint DATA-2B.3 — Multi-instrument data quality 3 instruments — 8h
- **Spec** : étendre data quality à XAU + EURUSD + USOIL (preset eval_20). OHLCV daily yfinance, macro features minimales, news tagged. PAS de live tick.
- **DoD** : 3 instruments avec qualité documentée.
- **KPI** : 6 ans daily, 0 gap > 5j.

#### Sprint DATA-2B.4 — Audit trail B2B "data quality" service — 6h
- **Spec** : pour chaque enrichissement B2B, persister `client_id, input_signal_hash, output_context, sources_cited, timestamp, model_version`. Endpoint client `/audit/{request_id}`.
- **DoD** : 1 mois audit accumulé.
- **KPI** : 100% requêtes auditées.

#### Sprint DATA-2B.5 — Data freshness monitoring + SLA — 5h
- **Spec** : SLA news < 30min lag, sentiment < 45min, macro < 26h. Dashboard public webapp (transparence). Alertes Sentry.
- **DoD** : dashboard live, SLA documenté CGU.
- **KPI** : SLA respecté 95%.

**Synthèse Marwan 2B :** Corpus narratif riche et auditable. Moins de live data (économie €300/an), plus de sentiment+sources structurés.

### Agent 2 — Elena (Quant) — 25h

**Vision Phase 2B :** Réduit volontairement. Sans edge, mon expertise sert ailleurs : (1) calibrer scoring **non pour trader** mais pour gauger conviction narrative, (2) forward-test paper transparent, (3) service B2B "audit qualité de signal", (4) explainer features pour Aisha.

#### Sprint QUANT-2B.1 — Calibration check pour conviction narrative — 6h
- **Spec** : reprendre scoring ConfluenceDetector (même non-prédictif) et calibrer sur "setup quality narrative" : score 80+ = "setup avec plusieurs facteurs alignés". Ce n'est PAS prédictif mais utile pédagogiquement.
- **DoD** : `reports/scoring_for_narrative.md`, mapping bucket→labelnarratif.
- **KPI** : narrative test sur 50 signaux ne dit jamais "à acheter" même score 95.

#### Sprint QUANT-2B.2 — Forward-test paper "transparency mode" — 8h
- **Spec** : harness paper trading qui suit "signaux fictifs" (pas vendus). Publier en temps réel sur webapp avec disclaimer "Démonstration paper-trading. Smart Sentinel ne prétend PAS posséder un edge. Cette courbe est éducative." Publié quel que soit le résultat.
- **DoD** : webapp publie equity curve live à M5, disclaimer clair.
- **KPI** : utilisateurs (n=10 user test M9) confirment "j'ai compris que ce n'est pas un signal d'achat".
- **Kill** : si AMF perçoit comme "résultats hypothétiques" interdits → adapter disclaimer.

#### Sprint QUANT-2B.3 — Backtest harness pour B2B "signal quality audit" service — 8h
- **Spec** : industrialiser harness CPCV+DSR+PBO en service. Broker/EA dev fournit historique signaux + ground truth → audit complet : DSR, PBO, Holm, β-capture detection, leak audit. Format : rapport PDF 5-10 pages. Pricing : 499-2000€ par audit one-shot, ou 999€/mo abonnement.
- **DoD** : 1 audit pilote livré (peut être synthétique pour démo).
- **KPI** : 1 audit vendu à un prospect M9-M12.

#### Sprint QUANT-2B.4 — Feature importance explainer — 3h
- **Spec** : SHAP TreeExplainer sur LightGBM existant, génère pour chaque signal "top 3 features qui poussent le score haut/bas". Format JSON consommable par Aisha.
- **DoD** : champ `feature_explanations` dans InsightSignal v2.
- **KPI** : utilisé dans 100% des narratives.

**Synthèse Elena 2B :** 25h modestes, c'est cohérent : sans edge prouvé, mon expertise sert ailleurs (B2B audit, calibration narrative, transparence). Cette réduction libère 45h pour Aisha (RAG) et Inès (webapp).

### Agent 3 — Kenji (Regime/Vol) — 25h

**Vision Phase 2B :** Régime + vol servent **narrative et éducation**, pas modèle prédictif. Régime classifier 3-state pour contexte, vol regime visualization, diurnal/calendar context, corrélations cross-asset.

#### Sprint REGIME-2B.1 — Régime classifier 3-state — 8h
- **Spec** : HMM 3 états (low-vol trending, low-vol ranging, high-vol stress) sur returns XAU. Output `regime_state` + `regime_confidence` consommable narrative. Pas de jump model SOTA — HMM suffit.
- **DoD** : régimes interprétables, transitions documentées.
- **KPI** : utilisé par Aisha dans 100% narratives 2B.

#### Sprint REGIME-2B.2 — Vol regime visualization — 6h
- **Spec** : composant Plotly pour webapp : timeline avec backround couleur par régime + ATR overlay. Interactif (zoom, hover).
- **DoD** : composant intégré dans dashboard.
- **KPI** : NPS user-test ≥ 7.

#### Sprint REGIME-2B.3 — Diurnal/calendar context — 6h
- **Spec** : pour chaque heure, calculer histo vol + skew + hit-rate par direction (sur 6 ans XAU). Idem day-of-week, distance to FOMC. Stocker comme "stylized facts".
- **DoD** : table stylized facts publiée.
- **KPI** : 5+ stylized facts utilisés en moyenne par narrative.

#### Sprint REGIME-2B.4 — Cross-asset correlation context — 5h
- **Spec** : rolling correlation 30j XAU vs DXY, SPX, US10Y, BTC. Heatmap. Narrative s'appuie ("XAU anticorrélé DXY -0.78 cette semaine").
- **DoD** : heatmap dans webapp.
- **KPI** : utilisé dans ≥ 40% narratives.

**Synthèse Kenji 2B :** Contexte narrative riche scientifiquement défendable. Aide à la décision, pas signal.

### Agent 4 — Aisha (LLM) — 80h ⭐ AGENT CENTRAL

**Vision Phase 2B :** **Coeur du produit en 2B**. Le narrative LLM avec RAG sourcé devient le héros. 8 sprints majeurs : RAG architecture, source curation, eval extended, multi-langue 4 langues, Q&A chat, prompt versioning, qualité dashboard, cost optimization. 25% du budget Phase 2B.

#### Sprint LLM-2B.1 — RAG architecture — 14h
- **Spec** : architecture RAG : (1) embedding store ChromaDB local (gratuit, 0 ops), (2) chunking 500 tokens overlap 100, (3) retriever hybride BM25 + dense (Voyage AI `voyage-3-large` 0.18$/1M tok), (4) re-ranker Cohere ou cross-encoder MiniLM. Pipeline : query → embed → retrieve top-20 → rerank top-5 → LLM avec contexte. Prompt anti-hallucination : "ne réponds qu'avec sources fournies, cite explicitement". Cache embeddings.
- **DoD** : RAG end-to-end, latence p99 < 4s, cost < 0.01€/query.
- **KPI** : F1 sourcing > 0.85, hallucination < 5%.
- **Kill** : si RAG hallucine > 15% → revoir prompt + retriever.
- **Référence** : Lewis et al. (2020) RAG paper ; ChromaDB docs ; Voyage AI.

#### Sprint LLM-2B.2 — Source curation 50 sources tagged — 10h
- **Spec** : curation manuelle 50 sources tagged : (a) papers académiques (gold market, SMC, vol forecasting) — 15 sources, (b) reports institutionnels (LBMA, WGC, BIS) — 15 sources, (c) data sources auditables (CFTC reports, FOMC minutes) — 10 sources, (d) educational (Investopedia, BabyPips) — 10 sources. Chaque source ingérée avec metadata `{source_type, authority_score, date, language, license}`.
- **DoD** : 50 sources indexées.
- **KPI** : retrieval quality manual eval n=50 ≥ 80% top-5 pertinent.
- **Kill** : si licences interdisent indexation → pivot summaries-only.

#### Sprint LLM-2B.3 — Eval harness étendu 200 prompts + RAG faithfulness — 10h
- **Spec** : étendre eval à 200 fixtures couvrant régimes, multi-asset, narrative Q&A, stylized facts. Ajouter **RAGAS metrics** : faithfulness, answer_relevancy, context_precision, context_recall. Cible : faithfulness > 0.90.
- **DoD** : 200 fixtures + RAGAS pipeline en CI.
- **KPI** : faithfulness ≥ 0.90, hallucination ≤ 5%.
- **Référence** : Es et al. (2023) *RAGAS*, arXiv:2309.15217.

#### Sprint LLM-2B.4 — Multi-langue FR/EN/DE/ES — 12h
- **Spec** : prompts FR primaire (Montréal+France+Belgique), EN secondaire, DE+ES tertiaire. Stylized facts traduits via LLM avec QC manuel n=20/langue. Disclaimers compliance traduits par avocat.
- **DoD** : 4 langues couvertes, eval ≥ 0.78 par langue.
- **KPI** : pas de divergence factuelle inter-langue > 5%.
- **Kill** : si DE+ES drainent qualité → drop, garder FR+EN.

#### Sprint LLM-2B.5 — Q&A chat endpoint — 12h
- **Spec** : endpoint `/chat` qui prend query + context (signal en cours, historique chat 5 derniers) → RAG response avec sources citées. Streaming tokens. Rate limit 10 msg/h FREE, 100/h PRO+. Anti-prompt-injection.
- **DoD** : chat fonctionnel, intégration UX-2B.3.
- **KPI** : NPS user (n=20) ≥ 8/10.
- **Référence** : OWASP LLM Top 10.

#### Sprint LLM-2B.6 — Prompt versioning + A/B testing — 6h
- **Spec** : git-versioné `prompts/` + meta `prompts/manifest.yaml` avec semver. `prompt_loader.py` avec sticky bucket per user_id pour A/B. Eval auto compare versions.
- **DoD** : 1 A/B test mené avec décision documentée.

#### Sprint LLM-2B.7 — Narrative quality dashboard — 6h
- **Spec** : dashboard Grafana avec faithfulness rolling 7j, cost rolling, hallucination flags, top-5 worst narratives (manual review queue).
- **DoD** : dashboard live, weekly review process.

#### Sprint LLM-2B.8 — Cost optimization aggressive — 10h
- **Spec** : (a) Haiku-first pour Q&A simple ($0.25/$1.25 vs Sonnet $3/$15), upgrade Sonnet sur trigger qualité, (b) prompt caching Anthropic (1h TTL, 90% discount sur cache reads) — système prompt 2840 tok caché, (c) batch API pour eval offline (50% off), (d) rate limit token-budget par user/jour.
- **DoD** : cost reduction documentée vs baseline naïve.
- **KPI** : cost per active user/mo < 1.5€ (FREE), < 4€ (PRO).
- **Kill** : si > 60% revenue spent on LLM → freeze new users tier FREE.

**Synthèse Aisha 2B :** 80h livrent coeur du produit 2B : RAG sourcé fiable, eval rigoureuse, multi-langue, Q&A chat, optim coût. Sans ces 80h, 2B est un ConfluenceDetector renommé — avec, c'est un produit narrative-first défendable face à TradingView Premium ou Trade Ideas.

### Agent 5 — Théo (Infra) — 50h

**Vision Phase 2B :** 5 piliers : webapp infra (canal principal en 2B), forward-test paper transparent (feature marketing), Stripe 4 tiers, compliance W4 + finfluencer 2026, B2B-API "data quality", backup.

#### Sprint INFRA-2B.1 — Webapp infrastructure — 12h
- **Spec** : FastAPI backend + Next.js (ou Astro statique pour public) frontend. Vercel pour frontend, Railway pour backend. Auth via magic link. CDN Cloudflare. SSR pour SEO.
- **DoD** : webapp en prod, Lighthouse SEO ≥ 90.
- **KPI** : deploy < 5min, uptime 99.9%.
- **Kill** : si Next.js > 18h → fallback Astro + HTMX.

#### Sprint INFRA-2B.2 — Forward-test paper "transparent track record" — 10h
- **Spec** : harness paper trading 24/7, equity curve publiée live sur webapp avec disclaimers radicaux. Différence avec 2A : ici c'est une **feature marketing transparence**, pas un gate. Affiché à tous (FREE inclus). On embrasse les drawdowns au lieu de les cacher.
- **DoD** : equity curve live sur landing.
- **KPI** : utilisateurs (n=10 test) confirment "j'ai compris que c'est démonstratif".

#### Sprint INFRA-2B.3 — Stripe 4 tiers — 8h
- **Spec** : 4 tiers (FREE/LITE19/PRO39/PRO+99) + B2B custom invoicing. Trial 14j sans carte sur LITE+PRO. Tier enforcement.
- **DoD** : 4 tiers payables, trial OK.

#### Sprint INFRA-2B.4 — Compliance W4 + finfluencer 2026 ready — 8h
- **Spec** : avocat fintech FR (1500€) relit CGU + privacy + claims marketing 2B + nouvelle régul finfluencer mars 2026. Deliverable : checklist "claims autorisés/interdits" pour Karim. Adapter narrative LLM pour respecter.
- **DoD** : CGU + privacy validés, checklist 1 page.
- **Kill** : si avocat exige enregistrement AMF → refuser, ré-évaluer (potentiellement positionner comme édition site éditorial commenté MiFID-art-21).
- **Référence** : eval_29 ; AMF position 2024-09 ; règlement 2024/2811.

#### Sprint INFRA-2B.5 — B2B-API "data quality enrichment" — 8h
- **Spec** : endpoint `/enrich` qui prend signal client `{instrument, direction, price, time}` → retourne `{regime_context, vol_context, macro_context, sentiment, narrative, sources_cited}`. Pricing 499€/mo basic 1k req, 1500€/mo pro 10k req. Auth API key + rate limit + audit.
- **DoD** : endpoint testé, OpenAPI documenté, 1 client pilote en M9.
- **KPI** : 1 contrat signé M12.
- **Kill** : si après 10 prospects 0 intérêt → pivot use case.

#### Sprint INFRA-2B.6 — Backup + DR — 4h
- **Spec** : SQLite + ChromaDB backup quotidien Backblaze, restore mensuel testé.
- **DoD** : 1 restore test réussi.

**Synthèse Théo 2B :** Stack webapp-first compliant + B2B "data quality" + transparence radicale. Compliance W4 closed = pas de risque AMF/MiFID II.

### Agent 6 — Inès (UX) — 50h ⭐

**Vision Phase 2B :** Presque double de 2A — c'est conscient. En 2B, l'UX **est** le produit. La webapp narrative-rich, le chat Q&A, le mobile-responsive, multi-langue — tout doit être bien fait sinon aucune différenciation.

#### Sprint UX-2B.1 — Webapp dashboard MVP narrative-rich — 18h ⭐
- **Spec** : Next.js webapp avec : (1) Landing FR avec proposition valeur "comprenez le marché XAU sans qu'on vous dise quoi faire", (2) Dashboard logged-in : narrative du jour (Aisha), forward-test transparent, régimes, corrélations, calendar economic, (3) Pages éducatives glossary, (4) Tier comparison + upgrade. Storybook. TailwindCSS + shadcn/ui.
- **DoD** : 6 pages live, design cohérent, Lighthouse perf ≥ 80.
- **KPI** : NPS landing test (n=20) ≥ 7/10.
- **Kill** : si > 22h → couper pages éducatives (-4h).

#### Sprint UX-2B.2 — Onboarding éducatif avec free hooks — 8h
- **Spec** : signup → tutorial 4 étapes (XAU c'est quoi, comment lire narrative, régime, transparence) → premier narrative. Mesure conversion FREE→LITE J7, J14, J30.
- **DoD** : flow live, tracking analytics OK.
- **KPI** : conversion FREE→LITE M3 = 2.5%.

#### Sprint UX-2B.3 — Q&A chat interface — 8h
- **Spec** : composant React chat avec streaming tokens, sources citées en cards, history. Mobile responsive. Disclaimer permanent footer.
- **DoD** : chat fonctionnel webapp.
- **KPI** : 50% PRO users utilisent chat ≥ 1×/sem M9.

#### Sprint UX-2B.4 — Mobile responsive — 8h
- **Spec** : webapp 100% responsive 375px+ (iPhone SE jusqu'à desktop). Touch-friendly. PWA optionnel.
- **DoD** : Lighthouse mobile perf ≥ 80, PWA installable.
- **KPI** : 40% sessions mobile à M9.

#### Sprint UX-2B.5 — Glossary + tooltip system — 4h
- **Spec** : 50 termes (BOS, CHOCH, ATR, RSI, breakeven yield, COT, FOMC) avec tooltips inline. Glossary page dédiée.
- **DoD** : 50 termes, tooltips fonctionnels.
- **KPI** : 30% sessions affichent ≥ 1 tooltip.

#### Sprint UX-2B.6 — Multi-language UI — 4h
- **Spec** : i18n via `next-intl` (FR/EN/DE/ES). Locale sticky par cookie + détection browser.
- **DoD** : 4 langues UI, switch fonctionnel.
- **KPI** : 30% trafic non-FR à M12.

**Synthèse Inès 2B :** UX qui justifie de payer pour produit narrative-first. Sans webapp polish, le pivot 2B s'effondre.

### Agent 7 — Karim (Commercial) — 40h

**Vision Phase 2B :** +33% vs 2A car en 2B il faut éduquer le marché pour catégorie nouvelle ("intelligence contextuelle, pas signaux"). SEO éducatif lourd, YouTube weekly market wrap (signal de différenciation : aucun concurrent ne fait ça en FR), B2B "data quality" outbound.

#### Sprint COMM-2B.1 — SEO content éducatif FR 10 cornerstone — 12h
- **Spec** : 10 articles cornerstone (3000+ mots) : "qu'est-ce que SMC ?", "comprendre le COT or", "régimes de marché XAU", "vol forecasting expliqué", "SL/TP calcul intelligent", etc. Cible KD<25.
- **DoD** : 10 articles publiés, schemas.org.
- **KPI** : 3 articles top-10 Google FR M12.

#### Sprint COMM-2B.2 — YouTube weekly market wrap — 12h (étalé)
- **Spec** : 1 vidéo/sem 5-10min FR : recap XAU semaine + narrative LLM-assisté. Format simple (screenshare, audio). 24 vidéos/an. **Canal unique en FR sur XAU intraday** — différenciation forte.
- **DoD** : 24 vidéos publiées.
- **KPI** : 500 abonnés YouTube M12.

#### Sprint COMM-2B.3 — Outbound B2B "data quality" 5 prospects — 6h
- **Spec** : 5 prospects ciblés copy-trading platforms (eToro, ZuluTrade, Darwinex), EA dev shops (Forex Tester, MQL5 vendors), small brokers. Pitch : "enrichissez vos signaux avec contexte LLM sourcé, 499€/mo, audit-trail, multi-langue".
- **DoD** : 5 contacts, ≥ 2 réponses, ≥ 1 démo.
- **KPI** : 1 contrat M12.

#### Sprint COMM-2B.4 — Community Discord + content remix — 6h
- **Spec** : Discord public 5 channels + Telegram + LinkedIn weekly post. Content remix : 1 article SEO → 1 vidéo YT → 5 LinkedIn → 10 Twitter/X.
- **DoD** : 200 Discord members M12.
- **KPI** : 50 actifs/sem M12.

#### Sprint COMM-2B.5 — Pricing experiments — 4h
- **Spec** : 2 A/B tests pricing : (1) trial 7j vs 14j sur LITE, (2) bundle PRO+annual -20% vs monthly. Statsig free tier.
- **DoD** : 2 tests menés, décisions documentées.
- **KPI** : ≥ 1 win statistique p<0.05.

**Synthèse Karim 2B :** Faire émerger une catégorie produit "intelligence contextuelle XAU FR". SEO + YouTube différencient fortement face à TradingView/TradingIdeas.

### Agent 8 — Sofia (Risk) — 15h

**Vision Phase 2B :** Constante 2A vs 2B. Mon rôle ne dépend pas du scénario.

#### Sprint RISK-2B.1 — Forward-test transparency commitment — 4h
- **Spec** : check hebdo que equity curve est publiée live, sans massage des chiffres. Doc publique commitment + audit log SHA256 hebdo.
- **DoD** : commitment doc publié, 40 audits hebdo.

#### Sprint RISK-2B.2 — Compliance monitor claims — 5h
- **Spec** : check mensuel automatisé (LLM-as-judge avec checklist Théo) sur tous contenus marketing, narratives, articles SEO, vidéos. Aucun "achetez", "vendez", "100% sûr", "edge prouvé". Si violation → fix avant publication.
- **DoD** : 10 audits mensuels.
- **KPI** : 0 violation publiée.

#### Sprint RISK-2B.3 — Quarterly review — 3h
- **Spec** : 3 reviews Q2/Q3/Q4 avec MRR vs target, churn, feature adoption.
- **DoD** : 3 reviews écrites.

#### Sprint RISK-2B.4 — Incident response — 3h
- **Spec** : runbook (a) RAG hallucination détectée → freeze + correction + comm, (b) Stripe down, (c) data leak, (d) AMF inquiry.
- **DoD** : runbook 1 page, dry-run.

**Synthèse Sofia 2B :** Discipline transparence radicale = ce qui différencie 2B.

## IV.3 Diagramme de Gantt Phase 2B (mois 3-12)

```
M3 (S9-12)   | Théo: I2B.1 webapp ⭐ | Aisha: L2B.1 RAG ⭐ | Marwan: D2B.1, D2B.2 | Sofia: R2B.1
M4 (S13-16)  | Aisha: L2B.2, L2B.3 | Inès: U2B.1 webapp ⭐ | Théo: I2B.2 forward-test ON
M5 (S17-20)  | Inès: U2B.1 (suite) | Théo: I2B.3 Stripe   | Aisha: L2B.4 multi-lang
M6 (S21-24)  | Aisha: L2B.5 chat ⭐ | Inès: U2B.3 chat    | Karim: C2B.1 SEO       | Théo: I2B.4 legal
M7 (S25-28)  | Inès: U2B.2 onbo, U2B.4 mobile | Aisha: L2B.6, L2B.7 | Marwan: D2B.3 multi-asset
M8 (S29-32)  | Karim: C2B.2 YT (rolling) | Théo: I2B.5 B2B API   | Aisha: L2B.8 cost  | Sofia: R2B.2
M9 (S33-36)  | Karim: C2B.3 outbound | Inès: U2B.5, U2B.6 | Marwan: D2B.4, D2B.5 | CP-2B.2 ⚠️
M10-M12      | Karim: C2B.2 (suite), C2B.4 community, C2B.5 pricing tests | Sofia: R2B.3 quarterly | I2B.6 backup

Chemin critique: I2B.1 → L2B.1 → L2B.2 → L2B.3 → U2B.1 → I2B.3 → C2B.1 → C2B.4
```

## IV.4 Matrice des dépendances Phase 2B

| Sprint | Dépend de |
|---|---|
| LLM-2B.1 | DATA-2B.1, DATA-2B.2 |
| LLM-2B.2 | LLM-2B.1 |
| LLM-2B.3 | LLM-1.1, LLM-2B.1 |
| LLM-2B.5 | LLM-2B.1, UX-2B.3 |
| UX-2B.1 | INFRA-2B.1 |
| UX-2B.2 | UX-2B.1 |
| UX-2B.3 | LLM-2B.5 |
| UX-2B.4 | UX-2B.1 |
| INFRA-2B.2 | QUANT-2B.2, INFRA-2B.1 |
| INFRA-2B.3 | INFRA-2B.1 |
| INFRA-2B.5 | DATA-2B.4, LLM-2B.1 |
| COMM-2B.1 | UX-2B.1 |
| COMM-2B.2 | LLM-2B.5 |
| COMM-2B.3 | INFRA-2B.5 |
| RISK-2B.1 | INFRA-2B.2 |

**Bottleneck** : Aisha + Théo + Inès en parallèle M3-M5. Si un dérape, tout glisse. Mitigation : Aisha et Théo en parallèle (RAG + webapp infra séparables), Inès attend webapp infra.

## IV.5 Budget effort par agent Phase 2B

| Agent | Heures | % | Justification |
|---|---|---|---|
| Aisha | 80h | 25% | Coeur du produit narrative |
| Inès | 50h | 16% | Webapp = canal principal |
| Théo | 50h | 16% | Webapp infra + B2B + compliance |
| Karim | 40h | 12% | Education marché nouvelle catégorie |
| Marwan | 35h | 11% | Corpus narrative + multi-asset |
| Elena | 25h | 8% | Réduit volontairement |
| Kenji | 25h | 8% | Réduit volontairement (régime narrative) |
| Sofia | 15h | 5% | Constante gouvernance |
| **Total** | **320h** | **100%** | |

---

# PARTIE V — Synthèse comparative 2A vs 2B et recommandation conditionnelle

## V.1 Tableau récapitulatif comparatif

| Dimension | Phase 2A — Edge confirmé | Phase 2B — Narrative-first |
|---|---|---|
| **Probabilité d'occurrence** | 25-35% | 65-75% |
| **Heures dev** | 320h | 320h |
| **Coût annuel infra** | ~600€ | ~350€ |
| **Coût LLM annuel** | 1200-3600€ | 2400-6000€ |
| **Coût legal one-shot** | 800-1500€ | 800-1500€ |
| **Tier le moins cher payant** | 29€/mo (ANALYST) | 19€/mo (LITE) |
| **Tier le plus cher B2C** | 199€/mo (INSTITUTIONAL) | 99€/mo (PRO+) |
| **Tier B2B principal** | 1500-3000€/mo white-label signal | 499-1500€/mo enrichment |
| **MRR cible M6** | 2000€ (P=55%) | 2000€ (P=50%) |
| **MRR cible M9** | 6500€ (P=40%) | 4500€ (P=45%) |
| **MRR cible M12** | 11000-14000€ (P=30%) | 8500-11000€ (P=40%) |
| **MRR espérance probabilisée** | E[MRR M12] = **3750€** | E[MRR M12] = **3900€** |
| **Users B2C cible M12** | 150-200 | 200-280 |
| **Contrats B2B cible M12** | 2 (high-value) | 2-3 (mid-value) |
| **Agent dominant** | Elena (22%) | Aisha (25%) |
| **Risque kill majeur** | Forward-test PF<0.85 M4 | LLM cost > 60% revenue |
| **Risque réglementaire** | Élevé : claim "edge" attire AMF | Modéré : narrative + transparence = positionnement éditorial |
| **Différenciation vs concurrence** | "edge live-validated CPCV+DSR" | "narrative FR sourcé + transparence radicale" |
| **Time-to-revenue** | M4 (post forward-test) | M5 (post webapp+RAG) |
| **Optionalité future** | Si edge valide → fund/prop firm angle | Si narrative scale → educator/coach angle |
| **Ceiling 36 mois** | 50-100k€ MRR (réussite B2B brokers) | 30-60k€ MRR (saturation marché éducation FR XAU) |
| **Plancher pessimiste 12 mois** | 1500-3000€ MRR | 2000-4000€ MRR |

## V.2 Lecture du tableau — 4 enseignements stratégiques

**Enseignement 1 : Espérance probabilisée presque identique.** E[MRR M12 | 2A] = 3750€ vs E[MRR M12 | 2B] = 3900€. **Du point de vue du solo founder rationnel, les deux scénarios ont une valeur attendue comparable**. La décision Phase 2A vs 2B n'est PAS une dégradation — c'est un changement de produit cible.

**Enseignement 2 : 2B a un meilleur plancher, 2A a un meilleur plafond.** Si risk-averse → 2B rassurant. Si risk-seeking → 2A optionalité plus haute via canal B2B-API broker.

**Enseignement 3 : Le risque réglementaire est asymétrique.** En 2A, claim "edge prouvé" + signaux entry/SL/TP ≈ recommandation personnalisée AMF/MiFID II. En 2B, positionnement éditorial-narrative-transparence = analogue à un newsletter financier. Phase 2B est **structurellement plus sûre** sous régul finfluencer 2026.

**Enseignement 4 : Le scénario hybride existe partiellement.** Si A1 produit verdict mitigé (DSR=0.7, PBO=0.4), voie 2B+ qui réutilise QUANT-2A.6 (calibration probabilité) et REGIME-2A.2 (Jump Model). Sofia tranche ce cas en CP-A1.

## V.3 Recommandation conditionnelle (à n'appliquer qu'après CP-A1)

### V.3.1 Avant le verdict A1 (mois 1-2) — Phase 1 stricte

**Recommandation : exécuter Phase 1 telle quelle, sans préférence pour 2A.** Anti-mesures déjà encodées :
- DSR + PBO + Holm + DM = critères pré-spécifiés, calculés mécaniquement (QUANT-1.3)
- Sofia rédige post-mortem A1 quel que soit le résultat (RISK-1.1)
- Karim a écrit briefs 2A ET 2B en S5-S6 **avant** le verdict (COMM-1.1)
- Critère de bascule binaire (pas d'entre-deux verbal)

**Action concrète :** ne pas relire Partie III plus que Partie IV pendant les 8 semaines de Phase 1.

### V.3.2 Après le verdict A1 — embranchement

**Cas 1 — Verdict 2A clair (DSR>1.0, PBO<0.3, ≥3 Holm, CPCV PF>1.20) :**
- Bascule Phase 2A immédiate
- Garder discipline forward-test gate INFRA-2A.2 + RISK-2A.1 ⚠️ **non négociable même en cas d'enthousiasme**

**Cas 2 — Verdict 2B clair (DSR<0 OU PBO>0.5 OU 0 Holm) :**
- Bascule Phase 2B immédiate, sans rationalisation
- Écrire publiquement (blog interne + Discord) le pivot — la transparence renforce le positionnement 2B
- Garder Elena à 25h Phase 2B sans culpabilité

**Cas 3 — Verdict mitigé (DSR=0.5, PBO=0.35, 2 Holm) :**
- **NE PAS forcer 2A**. Sofia tranche : par défaut → 2B+
- Voie 2B+ = Phase 2B avec emprunt sélectif :
  - QUANT-2A.6 (calibration probabilité) intégré en 2B
  - REGIME-2A.2 (Jump Model) intégré en 2B
- **NE PAS** lancer Stripe en mode 2A avec edge mitigé

### V.3.3 Décision à 6 mois (CP-2A.1 ou CP-2B.1)

À M6, revoir l'engagement :
- En 2A : si MRR M6 < 1000€ ET forward-test PF rolling 60j < 1.05 → **partial pivot vers 2B** (~30h sur M7-M8)
- En 2B : si MRR M6 < 800€ ET conversion FREE→LITE < 1% → **revoir pricing** + ICP

### V.3.4 Recommandation méta — éviter le piège du choix binaire

Le verdict A1 produit un signal (edge / pas edge), pas une identité produit. **Smart Sentinel AI restera la même plateforme à 7 couches dans les deux cas.** Ce qui change :
- En 2A : la couche 2 (algo) est le héros narrative commercial
- En 2B : les couches 4 (LLM) + 6 (UX) + 1 (data) sont les héros

**Aucun des sprints Phase 1 n'est "perdu" en 2B**, et **aucun des sprints Phase 2B n'est "consolation"**.

---

# PARTIE VI — Plan de mesure et reporting

## VI.1 Cadence de revue

| Cadence | Quand | Durée | Owner | Livrable |
|---|---|---|---|---|
| **Daily standup intime** | Lundi 9h00 | 15min | Solo founder | Mental note |
| **Weekly check** | Vendredi 16h-17h | 1h | Sofia | `kill_criteria_board.md` |
| **Bi-weekly retro** | 2e+4e vendredi du mois | 30min | Solo founder | `retros/{date}.md` |
| **Monthly checkpoint** | Dernier dimanche du mois | 2h | Solo founder + Sofia | `reports/monthly_checkpoint/{YYYY}-{MM}.md` |
| **Quarterly review** | Fin Q2/Q3/Q4 | 4h | Solo founder + Sofia | `reports/quarterly/Q{N}.md` |
| **Annual review** | Fin M12 | 1 jour | Solo founder | `reports/annual/2027.md` |

**Charge totale gouvernance :** 84h/an sur 384h dev = ~22%. Élevé mais nécessaire en mode solo.

## VI.2 Métriques de pilotage par agent

(Voir gabarit `kill_criteria_board.md` pour le détail des seuils vert/jaune/rouge par agent.)

Chaque agent a **3 métriques propres** suivies hebdomadairement par Sofia. Si une métrique passe seuil rouge, escalation au monthly checkpoint.

## VI.3 Format du checkpoint mensuel

Voir `reports/governance/monthly_checkpoint_template.md` (gabarit complet à dupliquer chaque mois dans `reports/monthly_checkpoint/{YYYY}-{MM}.md`).

Sections : phase active, dashboard agents, sprints livrés, KPI commerciaux, forward-test perf, kill criteria status, décisions, blockers, next month focus, réflexion qualitative, décision principale.

## VI.4 Format de la post-mortem A1

Voir `reports/governance/a1_postmortem_template.md` (gabarit complet à remplir fin S8).

Sections : verdict mécanique 8 critères, décision automatique, apprentissages techniques (features, hypothèses, modèle), apprentissages méthodologiques, implications produit, plan immédiat (selon GO 2A/2B/2B+), engagement écrit anti-rationalisation.

## VI.5 Outillage minimal de mesure

| Catégorie | Outil | Coût |
|---|---|---|
| Logs structurés | Python `logging` JSON + Sentry free | 0€ |
| Métriques produit | `prometheus_client` + Grafana free | 0€ |
| Web analytics | Plausible (cloud 9€/mo) | 9€/mo |
| Stripe analytics | Stripe Sigma | 0€ |
| LLM cost tracking | Anthropic API usage endpoint custom | 0€ |
| Outbound CRM | Airtable free tier | 0€ |
| User feedback | Posthog free | 0€ |
| Error tracking | Sentry free tier | 0€ |
| Uptime monitoring | UptimeRobot free | 0€ |

**Total outillage mesure : ~110€/an.**

## VI.6 Discipline de mesure — 5 règles non négociables

1. **Mesurer en temps réel, pas a posteriori.** Forward-test PF est calculé live, pas reconstitué.
2. **Pré-déclarer les seuils, pas les chercher après.** Les vert/jaune/rouge sont écrits **maintenant**.
3. **Compter les ratés, pas les optimistes.** "MRR 1500€ alors que cible était 2000€" se note 75% atteint.
4. **Sofia a un droit de veto explicite.** Sur kill et pivot, Sofia peut bloquer 7 jours avec demande de rétro.
5. **La transparence radicale s'applique aussi en interne.** Les checkpoints ne sont PAS retouchés après publication. Erreur → changelog en bas.

---

# Conclusion du document

**Récap visuel de la roadmap 12 mois :**

```
M1  ─────── PHASE 1 (mois 1-2, 64h) ─────── M2 ─── CP-A1 ───┐
                                                            │
                  ┌──────────────────────────────┬──────────┘
                  ▼                              ▼
            P(35%) GO 2A                    P(65%) GO 2B
            (320h, 600€/an)                 (320h, 350€/an)
            MRR M12 11k€ E[3750€]           MRR M12 8.5k€ E[3900€]
                  │                              │
                  ▼                              ▼
            Forward-test gate              RAG + webapp + Q&A
            Stripe ouverture M4            Stripe ouverture M5
            B2B-API broker                 B2B-API enrichment
            Premium pricing                Accessible pricing
                  │                              │
                  └──────────────┬───────────────┘
                                 ▼
                       M12 — Annual review
                       Décision 2027
```

**Ce que ce plan livre, indépendamment du verdict A1 :**

| Bénéfice | Phase 1 | Phase 2A | Phase 2B |
|---|---|---|---|
| Edge live-validé scientifiquement | partiel (verdict) | ✓ pleinement | ✗ (par design) |
| Plateforme SaaS opérationnelle | ✓ baseline | ✓ premium | ✓ accessible |
| Compliance MiFID II 2026 | ✓ W4 prep | ✓ closed | ✓ closed + finfluencer-ready |
| Multi-langue | ✗ | partiel FR+EN | ✓ FR+EN+DE+ES |
| RAG sourcé | ✗ | ✗ (pas prio) | ✓ pleinement |
| B2B-API canal | ✗ | ✓ broker premium | ✓ enrichment mid-range |
| Forward-test transparent | ✗ | ✓ gate (interne) | ✓ feature marketing |
| Discipline scientifique | ✓ DSR, PBO, Holm | ✓ + drift PSI | ✓ + RAGAS faithfulness |
| Audit trail B2B | ✗ | ✓ | ✓ |
| MRR M12 cible | n/a | 11k€ (P=30%) | 8.5k€ (P=40%) |

**Aucune des 384h annuelles n'est perdue dans aucun scénario.** C'est la garantie de design du plan.

**Recommandation finale :** exécuter Phase 1 à la lettre, accepter mécaniquement le verdict A1, exécuter la phase 2 correspondante avec engagement signé. La discipline est l'asset principal de ce solo founder — pas l'optimisme.

---

**Premier sprint à exécuter : DATA-1.1 (FRED macro ingestion, 4h) — voir Partie II.2 Agent 1.**

**Documents associés :**
- `reports/governance/kill_criteria_board.md` — board hebdo
- `reports/governance/monthly_checkpoint_template.md` — gabarit mensuel
- `reports/governance/a1_postmortem_template.md` — gabarit verdict A1
- `reports/positioning/positioning_2{A,B}_*.md` — à écrire par Karim S5-S6
- Memory pointer : `MEMORY.md` → `roadmap_2026_2027.md`

**Versionning** : ce document est versionné en git. Pour modifications majeures, créer `PLAN_12_MOIS_v{N}.md` plutôt que retoucher l'historique.
