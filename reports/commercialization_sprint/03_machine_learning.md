# Plan de Commercialisation — Catégorie 3 : Machine Learning Prédictif

> **Date** : 2026-05-21
> **Auteur** : Audit ML / agent Catégorie 3
> **Statut** : DRAFT exécutable — à arbitrer en revue de sprint dès cette semaine
> **Contexte décisif** : verdict A1 (`reports/a1_verdict_2026.md`) DSR=0.0000, PBO=0.5000, CPCV PF=1.008, DM stat=+46.7 ⇒ stack LightGBM 2 niveaux + 19 features + CPCV 28 paths **n'a démontré aucun edge** sur XAU M15 OHLCV+macro+calendrier. Tout effort ML doit dorénavant soit (a) prouver qu'un nouveau jeu de données change la conclusion, soit (b) servir à *calibrer la conviction* et *expliquer* — pas à prédire.

---

## 1. État actuel (Audit)

### 1.1 Briques ML livrées

| Brique | Statut | Fichier | Verdict empirique |
|---|---|---|---|
| **Feature matrix A1 vintage-safe** | Livrée | `src/research/a1_features.py:23-38` | 19 features ≥ cible 18, anti-leak validé |
| **CPCV harness (DSR/PBO/DM/Holm)** | Livrée | `src/research/cpcv_harness.py:1-100` | 28 paths fonctionnels |
| **Strategy gates (CPCV+DSR+PBO+DM+PF-lo)** | Livrée | `src/research/strategy_gates.py:73-79` | DSR≥1.5, PBO≤0.35, PF-lo>1.00, DM<0.05 |
| **A1 stacked LightGBM** | Livré + **ÉCHEC** | `src/research/a1_train.py:14-39`, modèle `models/a1_stack_v1.pkl` | DSR=0, PBO=0.5, PF=1.008 — **aucun edge** |
| **FactorModelPredictor (LGBM régression macro+price)** | Livré | `src/intelligence/factor_model/predictor.py:25-99`, modèle `models/factor_model_v1.pkl` | Présenté comme passant "5/5 gates" dans `scoring/lgbm_scoring_engine.py:1-25` — **à re-vérifier** (incohérent avec verdict A1) |
| **LGBM scoring (8 features confluence)** | Livré | `src/intelligence/scoring/lgbm_scorer.py:38-70`, modèle `models/scoring_v3_lgbm.pkl` | OK pour calibration P(win), pas pour prédiction directionnelle |
| **Logistic L1 baseline** | Livré | `src/intelligence/scoring/logistic_l1.py`, `scripts/train_logistic_l1_on_sweep.py:1-100`, modèle `models/scoring_v3_logistic_l1.pkl` | Baseline simple, audit 3.3 |
| **Isotonic recalibration (Platt-like, monotone)** | Livré | `src/intelligence/scoring/isotonic_recalibration.py:17-41` | Brique calibration prête |
| **CalibratedConviction pipeline (LGBM → Isotonic → Conformal)** | Livré | `src/intelligence/scoring/calibrated_conviction.py:1-80` | Honnête (`edge_claim=False`) |
| **Score → bucket calibration (narrative honnête)** | Livré | `src/intelligence/score_calibration.py:1-60` | UE 2024/2811 compliant, pas de claim probabiliste |
| **Conformal wrapper (Split + ACI)** | Livré | `src/intelligence/conformal_wrapper.py:90-307` | Rejette tout sur edge faible (correct) |
| **Mondrian conformal (stratifié régime)** | Livré | `src/intelligence/conformal/mondrian.py:42-60` | Adresse audit P0-20 PICP 43.6% au lieu 80% |
| **RegimeGate (BOCPD + bipower jumps)** | Livré | `src/intelligence/regime_gate.py:167-294` | +0.16 DSR sur Pillar 1, insuffisant seul |
| **HMM régime predictor** | Livré | `src/agents/regime_predictor.py` (référencé plan §1.1) | Lag 5-10 bars |
| **MacroFactorExtractor (real_10Y, DXY, VIX, COT)** | Livré | `src/intelligence/macro_factors/extractor.py:1-40` | PIT-safe |
| **Microstructure proxies (Kyle's λ, VPIN-BVC)** | Livré (dossier) | `src/intelligence/microstructure/proxies.py` | Features régime, non-directionnel |
| **Cross-asset correlation (lead-lag)** | Livré | `src/intelligence/cross_asset_correlation.py` | XAU vs DXY/UST/SPX/VIX |
| **Volatility LGBM (référence ML)** | Livré | `src/intelligence/volatility_lgbm.py` | VOL_MODE=har par défaut (latence) |
| **RAG pipeline (BM25+dense+RRF)** | Livré | `src/intelligence/rag/pipeline.py:1-20` | Brique narrative-first (Phase 2B) |
| **Narrative quality tracker (faithfulness, hallucination)** | Livré | `src/intelligence/narrative_quality.py:1-20` | Anti-hallucination dashboard |

### 1.2 Faiblesses critiques détectées (audit)

1. **Aucun model registry** : 5 `.pkl` dans `models/` sans manifest, sans hash, sans datalog, sans champ "passes-gates". Pas de versioning sémantique. Re-déploiement risqué.
2. **Incohérence interne `factor_model` vs A1** : `lgbm_scoring_engine.py:1-25` affirme "walk-forward LightGBM factor model (5/5 institutional gates passed)" alors que A1 verdict est sans appel ⇒ soit gates assouplies (P1 grave), soit factor_model n'a pas été soumis aux mêmes gates (P0 à audit immédiat).
3. **Pas de train/serve consistency test** : aucun test ne vérifie que la séquence (features build + scaler + model) en prod produit bit-by-bit les mêmes outputs que la séquence training.
4. **Pas de monitoring dérive** : Brier/log-loss/ECE en live non trackés. `narrative_quality.py` track faithfulness LLM, pas dérive ML.
5. **Calibration ECE absente** : isotonic est appliquée mais on ne mesure pas Expected Calibration Error sur 10 buckets après calibration. Sans ECE, on ne sait pas si la calibration tient.
6. **Pas de cohorte slicing** : hit-rate non décomposé par régime (high vol / low vol), par horaire (Asia/EU/US), par event-window (NFP/CPI/FOMC ±30min). Aveugle aux pockets d'edge.
7. **Conformal `apply_conformal_filter` lit le futur en backtest** (`conformal_wrapper.py:354-380`) : ACI `observe(r)` même quand rejet ⇒ le rejet courant ne biaise pas, mais le t+1 voit le réalisé du t courant. À auditer rigoureusement, car en walk-forward live le réalisé n'est connu qu'après `signal_lifetime_bars`.
8. **Pas de feature store** : `a1_features.py` reconstruit le matrix from scratch à chaque run. Pas de versioning vintage des features.
9. **Bocpd_step appelé bar-by-bar** dans `regime_gate.py:212-282` : O(N × max_run_length) = O(N²) potentiel ; OK offline mais en prod scanner 60s ça reste à benchmarker.
10. **HMM batch-fit une fois** (référence plan §1.2 point 7) : non-stationnarité non gérée.

### 1.3 Cartographie des modèles persistés

```
models/
├── a1_stack_v1.pkl          1.4 MB  — Stack 2-niveaux LightGBM, VERDICT ÉCHEC (DSR=0)
├── factor_model_v1.pkl      1.2 MB  — FactorModelPredictor LGBM régression macro+price
├── scoring_v2.lgb             25 KB — Modèle scoring v2 (legacy, plus utilisé ?)
├── scoring_v3_lgbm.pkl        83 KB — LGBMScorer 8 features confluence
└── scoring_v3_logistic_l1.pkl 903 B — LogisticL1 baseline 14 features
```

Aucun n'a de fichier compagnon `model_card.md` ni de `manifest.json` (hash, features, gates passées, date d'entraînement, hyperparams).

### 1.4 État de la donnée

- `data/research/a1_matrix_2019_2026.parquet` — matrix A1 ✅
- `data/macro/` : `cot_gold.csv`, `fred_BREAKEVEN_10Y.csv`, `fred_DFII10.csv`, `fred_DGS10.csv`, `fred_DTWEXBGS.csv`, `fred_T10Y2Y.csv`, `fred_VIXCLS.csv` ✅
- `data/macro/cot_cache/` — cache CFTC ✅
- **MANQUANT** : **Bloomberg/Reuters consensus** ⇒ surprise = actual − consensus impossible à calculer. C'est la donnée bloquante #1 pour Pilier 1 event-driven (cf. `reports/3_pillars_implementation_2026_05_13.md:96-99`).
- **MANQUANT** : DXY M15, SPX M15, VIX M15, UST10Y intra-day. FRED en daily seulement.

---

## 2. Vision cible — ML qui sert la commercialisation

Étant donné le verdict A1, deux familles de ML sont commercialisables :

### Vision A — ML PRÉDICTIF VALIDÉ (le pari haut-risque haut-réward)

Re-tirer A1 avec un **jeu de données qualitativement supérieur** (consensus Bloomberg + DXY/UST/SPX intra + COT vintage), un **target régime-conditionnel** (event-window vs hors-event) et des **gates inchangées**. P(succès post-Bloomberg ingestion) ≈ 30-45 % (cf. `reports/institutional_quant_transformation_plan.md` Pilier 1).

Si succès : revendication `edge_claim=true` autorisée sur asset/TF testé ⇒ unlock tier STRATEGIST/INSTITUTIONAL à prix premium, brouille la copy `bullish/bearish setup` vers `bullish/bearish bias avec proba calibrée`.

### Vision B — ML EXPLICATIF / DE CONFIANCE (le pari bas-risque, narrative-first)

Reconnaître que A1 a échoué et **arrêter de chercher l'edge directionnel pur**. Reposer le ML sur trois usages **non-prédictifs mais commercialisables** :

1. **Calibration honnête de conviction** : transformer le score confluence 0-100 (Pearson −0.023 vs PnL, `reports/eval_02_confluence.md`) en une **probabilité empirique P(R>0 | features)** via LGBM → isotonic → conformal. Promesse client : *"75% de conviction = historiquement, sur ce setup, 1 fois sur N quelque chose s'est passé"*. PAS *"75% chance de gagner"*. UE 2024/2811 compliant.
2. **Reject-option conformal** : ne pas générer de signal quand l'intervalle conforme passe sous breakeven. Améliore la **selectivité perçue** et la **rétention**.
3. **Régime gate explicatif** : BOCPD + bipower jumps disent *quand se taire*. C'est un trust-builder, pas un prédicteur — l'utilisateur voit `Regime: BLOCK — bipower jump 0.42` et apprend à respecter la machine.

Vision B est **commercialisable immédiatement** à condition de **ne JAMAIS revendiquer un edge**. Tarification cohérente : tier FREE → STRATEGIST orienté éducation+sélectivité+confiance, pas alpha.

### Recommandation : Vision B en chemin par défaut, Vision A en option conditionnée à l'ingestion Bloomberg

Le plan ci-dessous est construit pour que **Vision B soit live en 6-8 semaines**, et **Vision A soit décidable au sprint S+6** une fois la nouvelle donnée acquise (ou abandonnée si trop chère).

---

## 3. Gap analysis (ciblée commercialisation)

| Capacité requise commercialement | État actuel | Gap |
|---|---|---|
| Score 0-100 avec interprétation honnête | OK (`score_calibration.py:59-67`) | Aucun — déjà conforme UE 2024/2811 |
| Calibration P(R>0) avec ECE < 0.05 | Pipeline existe (`calibrated_conviction.py:1-80`) mais aucun rapport ECE | **Manque eval + dashboard ECE** |
| Conformal interval [lo, hi] publié sur chaque signal | Briques OK, **intégration dans `InsightSignalV2` partielle** | **Manque renderer + tests train/serve** |
| Reject-option sélective sur live | `apply_conformal_filter` offline OK | **Manque hook live dans `SignalStateMachine` + replay PF avant/après** |
| Régime gate live dans scanner | `regime_gate.py:167-294` OK offline | **Manque intégration `sentinel_scanner.py` + benchmark perf** |
| Model registry + manifest | Inexistant | **P0** |
| Train/serve consistency tests | Inexistant | **P0** |
| Dérive monitoring (PSI, KS, ECE rolling) | Inexistant | **P1** |
| Cohort slicing (régime × heure × event-window) | Inexistant | **P1** |
| Feature store (versioning vintage) | Inexistant (rebuild from scratch) | **P2** |
| Online learning (ACI déjà partiellement) | ACI live OK ; modèles batch | **P2** |
| Refit weekly/monthly automatisé | Inexistant | **P2** |
| Bloomberg consensus ingestion | Inexistant | **P0 conditionnel Vision A** |
| Edge claim `true` sur au moins 1 asset/TF | Aucun (sauf revendication non vérifiée `factor_model`) | **Vision A uniquement** |

---

## 4. Plan d'exécution

### P0 — Décision stratégique : pivot Vision B (narrative-first ML) — Semaine 1 (4h)

**Tâche P0.1 — Réunion go/no-go A1 retry vs Vision B**
- **Livrable** : décision documentée dans `reports/commercialization_sprint/03_ml_decision_2026_05.md`
- **Critères go A1 retry** :
  - Bloomberg/Reuters consensus accessible (budget < $500/mo OU partenariat broker) **dans 30 jours**
  - DXY/UST/SPX M15 disponible (Dukascopy DXY existe, à confirmer)
  - 80h dev disponibles sans bloquer roadmap commerciale
- **Critères go Vision B** : tout le reste
- **Dépendance** : revue par solo-founder + Sofia
- **Heures** : 4h
- **Probabilité par défaut** : 70% Vision B (déjà tranché dans memory `a1_verdict_2026_05_01.md` GO Phase 2B)

**Tâche P0.2 — Audit immédiat de l'incohérence `factor_model` vs A1**
- **Fichiers** : `src/intelligence/scoring/lgbm_scoring_engine.py:1-25` (claim 5/5 gates) vs `src/intelligence/factor_model/predictor.py:25-99` + `reports/a1_verdict_2026.md`
- **Objectif** : déterminer si `factor_model_v1.pkl` a été soumis aux mêmes gates strictes que `a1_stack_v1.pkl`. Si non ⇒ rétracter la claim `edge_claim=true` dans `lgbm_scoring_engine.py` et `confluence_detector` doit re-pointer.
- **Livrable** : `reports/audits/factor_model_vs_a1_consistency.md`
- **Critère réussite** : un et un seul des deux modèles peut prétendre passer gates, ou les deux sont marqués `edge_claim=false`.
- **Heures** : 4h
- **Risque** : embarras réputationnel interne si claim falsifiée s'est propagée en prod ; corriger urgemment.

---

### P0 — Si pivot Vision B : ML pour calibration de conviction + features explicatives (Sprints 1-3, 80h)

#### Sprint 1 (Semaine 2, 24h) — Calibration honnête bout-en-bout

**P0-B.1.1 — ECE evaluation harness**
- **Fichier** : `src/research/calibration_metrics.py` (NOUVEAU) + `scripts/eval_calibration.py` (NOUVEAU)
- **Tâche** : calcul ECE (Expected Calibration Error, M=10 buckets), MCE, Brier score, log-loss, reliability diagram (matplotlib PNG) sur sortie de `CalibratedConviction`.
- **Dépendance** : `src/intelligence/scoring/calibrated_conviction.py:1-80` déjà livré.
- **Critère réussite** : ECE < 0.05 sur OOS 2024-2025 ; sinon retravailler isotonic OU ajouter Platt sigmoidal.
- **Heures** : 8h

**P0-B.1.2 — Cohorte slicing**
- **Fichier** : `src/research/cohort_eval.py` (NOUVEAU)
- **Tâche** : reproduire ECE + hit-rate + Brier par cohorte :
  - Régime vol (low/normal/high — via `regime_classifier.py`)
  - Heure de la journée (Asia/EU/US sessions)
  - Event-window (±30min red news vs hors event, via `min_to_next_red_news`)
  - Direction (long-bias vs short-bias)
- **Livrable** : `reports/calibration_cohorts_2026_05.md` avec tableau 4×4 + reliability diagrams par cohorte.
- **Critère** : identifier ≥1 cohorte avec hit-rate − baseline > 5pp (= pocket d'edge faible mais réel).
- **Heures** : 8h

**P0-B.1.3 — Intégration `InsightSignalV2` du `CalibratedConviction`**
- **Fichiers** : `src/api/insight_signal_v2.py` (ajouter `UncertaintyContext` si absent), `src/intelligence/insight_v2/builder.py` (renderer), `src/intelligence/sentinel_scanner.py` (wire-in)
- **Tâche** : exposer `conviction_p_win`, `conviction_0_100`, `conformal_lower_0_100`, `conformal_upper_0_100`, `edge_claim=false`, `calibration_method="lgbm_isotonic_conformal"` dans chaque signal produit.
- **Tests** : `tests/test_insight_signal_v2_enrichment.py` (déjà modifié dans gitStatus) à étendre.
- **Heures** : 8h

#### Sprint 2 (Semaines 3-4, 28h) — Reject-option live + régime gate live

**P0-B.2.1 — Reject-option dans `SignalStateMachine`**
- **Fichiers** : `src/intelligence/signal_state_machine.py` (ajouter gate `conformal_pre_emit_check`), `src/intelligence/conformal_wrapper.py` (méthode `should_reject_live` côté serveur, sans observer le futur)
- **Tâche** : avant chaque emit (HOLD→BUY/SELL transition), interroger `AdaptiveConformalScorer.should_reject(breakeven=0.0)`. Si reject, rester HOLD et logger raison.
- **Audit critique** : éviter le bug `observe` qui voit le futur en live (cf. §1.2 point 7) ; `observe` ne doit être appelé qu'après `signal_lifetime_bars` quand le réalisé est connu, via callback dans `signal_tracker.py:src/api/signal_tracker.py`.
- **Tests** :
  - `tests/test_conformal_wrapper.py` : ajouter test "live mode no future leakage"
  - `tests/test_state_machine_replay.py` : replay XAU 2019-2025 pré/post conformal, attendu : trades −30 à −50 %, PF +0.10 à +0.30 absolu.
- **Critère réussite** : PF replay 2024-2025 OOS améliore d'au moins +0.15 absolu, nombre de trades restant ≥ 50 % du baseline.
- **Heures** : 12h

**P0-B.2.2 — `RegimeGate` intégré dans `SentinelScanner`**
- **Fichiers** : `src/intelligence/sentinel_scanner.py` (importer `RegimeGate`, appeler à chaque bar), `src/intelligence/insight_v2/builder.py` (exposer `regime_decision`, `regime_reason` dans `InsightSignalV2`).
- **Tâche** : `RegimeGate.update(log_return, recent_returns)` à chaque tick ; si `BLOCK`, suppression d'émission ; si `REDUCE`, demi-position dans `risk_manager`.
- **Benchmark perf** : `RegimeGate.update` < 5ms p99 sur 7 ans XAU M15 (172k bars). Si > 5ms, refacto vers `run_regime_gate` vectorisé.
- **Tests** : `tests/test_regime_gate.py` existant + nouveau `tests/test_sentinel_scanner_regime_integration.py`.
- **Critère** : 0 régression scanner ; latence end-to-end < 100ms p99 par bar.
- **Heures** : 10h

**P0-B.2.3 — Mondrian conformal stratifié régime (si Sprint 2 OOS gains < 0.15 PF)**
- **Fichiers** : `src/intelligence/conformal/mondrian.py:42-60` (déjà livré), intégration dans `CalibratedConviction`.
- **Tâche** : remplacer `SplitConformal` / `ACI` global par `MondrianConformal` taxonomy = `regime_label ∈ {LOW, NORMAL, HIGH, CRISIS}` (cf. `macro_factors/extractor.py:11`).
- **Adresse audit P0-20** PICP 43.6% vs cible 80% — la cause probable est non-stationnarité, Mondrian la corrige.
- **Tests** : `tests/test_conformal_mondrian.py` (NOUVEAU) — vérifier PICP par stratum.
- **Critère** : PICP conditionnel ≥ 70% par stratum.
- **Heures** : 6h

#### Sprint 3 (Semaines 5-6, 28h) — Honnêteté commerciale + monitoring

**P0-B.3.1 — `edge_claim` audit + UI/copy compliance UE 2024/2811**
- **Fichiers** : `src/intelligence/score_calibration.py:1-60` (guard phrases), `src/api/insight_signal_v2.py` (champs `edge_claim`, `disclaimers`, `calibration_method`), `src/delivery/telegram_notifier.py`, `mockups/telegram_b2c.txt`
- **Tâche** : vérifier qu'**aucune sortie utilisateur** ne contient "probabilité de gain", "high-probability", "buy signal" tant que `edge_claim=false`. Forcer `bullish/bearish setup` + guard phrase. Conviction présentée comme *score de confluence calibré historiquement*, pas comme *proba de gain*.
- **Tests** : `tests/test_telegram_notifier.py` (à créer) + `tests/test_compliance_copy.py` (NOUVEAU) regex sur outputs interdits.
- **Heures** : 10h

**P0-B.3.2 — Dérive monitoring (PSI, ECE rolling, hit-rate rolling)**
- **Fichier** : `src/intelligence/drift_monitor.py` (NOUVEAU), endpoint `GET /api/v1/metrics/ml-drift` dans `src/api/routes/health.py`
- **Tâche** :
  - PSI (Population Stability Index) sur les 8 features confluence : training distribution vs trailing 30-day distribution. Alerte si PSI > 0.25.
  - Rolling ECE 30 jours.
  - Rolling hit-rate 30 jours + CUSUM control chart.
- **Alerte** : log structuré + Telegram admin si dérive détectée.
- **Tests** : `tests/test_drift_monitor.py` (NOUVEAU).
- **Heures** : 10h

**P0-B.3.3 — Model registry + manifest**
- **Fichiers** : `models/registry.json` (NOUVEAU), `src/intelligence/model_registry.py` (NOUVEAU), `scripts/register_model.py` (NOUVEAU)
- **Schéma manifest** :
  ```json
  {
    "name": "scoring_v3_lgbm",
    "version": "3.1.0",
    "trained_at": "2026-05-16T01:16:00Z",
    "training_data": {"source": "reports/sweep/cell_*", "rows": 12345, "hash": "sha256:..."},
    "features": ["smc_structure", "..."],
    "hyperparams": {...},
    "gates_passed": {"DSR": 1.8, "PBO": 0.28, "PF_lo": 1.07, "DM_p": 0.012, "all_passed": true},
    "ece_oos": 0.04,
    "model_card": "models/scoring_v3_lgbm.model_card.md"
  }
  ```
- **Tâche** : chargement dynamique en prod via `model_registry.load("scoring_v3_lgbm@latest_passing")`. Refuser de charger un modèle sans `all_passed=true`.
- **Tests** : `tests/test_model_registry.py` (NOUVEAU).
- **Heures** : 8h

---

### P0 — Si re-run Vision A : roadmap 250h (event-driven 80h / conformal 30-50h / HAR-J+BOCPD 50-70h)

Reproduit `reports/institutional_quant_transformation_plan.md` §6 — **À NE LANCER QUE SI** P0.1 a confirmé l'accès données Bloomberg ET les 80h disponibles.

**Sprint A1 (Semaines 7-10, 80h) — Pilier 1 Event-Driven Macro**
- Reprendre `src/strategies/event_driven_macro.py` (livré, FAIL gates, cf. `reports/3_pillars_implementation_2026_05_13.md:50-54`)
- Ajouter feature **surprise = actual − consensus_bloomberg / std_historical** (P0 critique)
- Re-feature : COT delta last-week, news LLM sentiment classifier sur headline
- Test : re-passer `scripts/eval_event_driven_macro.py`, attendu PF_lo > 1.05
- Critère continuation : si PF_lo > 1.05 après surprise feature ⇒ continuer Sprint A2 ; sinon abandon
- **Heures** : 80h

**Sprint A2 (Semaines 11-12, 30-50h) — Conformal Wrapper sur event-driven validé**
- Réutiliser `src/intelligence/conformal_wrapper.py:90-307`
- Calibration set : 2019-2023, test 2024-2025
- Critère : ΔPF (after − before) > +0.15
- **Heures** : 40h

**Sprint A3 (Semaines 13-15, 50-70h) — HAR-J + BOCPD régime gate sur event-driven validé**
- `RegimeGate` déjà livré (`regime_gate.py:167-294`)
- Manque : HAR-RV-J avec décomposition jumps explicite (BV/RV ratio) — adapter `volatility_forecaster.py`
- Critère : Δ drawdown −20% OU Δ Sharpe +0.3
- **Heures** : 60h

**Sprint A4 (Semaine 16, 30h) — Risk layer quick wins**
- Fractional Kelly + CVaR + drawdown cap dans `src/intelligence/sizing.py` (NOUVEAU)
- Cross-asset lead-lag (déjà livré `cross_asset_correlation.py`) — étendre features
- Kyle's λ + VPIN-BVC (déjà livré `microstructure/proxies.py`) — wire dans confluence
- **Heures** : 30h

**Total Vision A** : ≈ 210h dev + 40h ops/data ingestion Bloomberg = **250h total** ; gate global mi-parcours fin Sprint A2 ⇒ si 0/2 piliers passent gates, **pivot B2B-API** (cf. `reports/decision_matrix_2026_04_30.md`).

---

### P1 — Industrialisation : Model registry, train/serve skew, CPCV pipeline (Sprints 4-5, 60h)

**P1.1 — Model registry production-grade**
- Étend P0-B.3.3. Ajoute :
  - Lazy-load avec cache mémoire
  - API endpoint `GET /api/v1/models` (liste les modèles enregistrés avec gates)
  - CLI `python -m src.intelligence.model_registry list/show/promote`
- **Fichier** : `src/intelligence/model_registry.py`
- **Heures** : 8h
- **Dépendance** : P0-B.3.3

**P1.2 — Train/serve consistency test suite**
- **Fichier** : `tests/test_train_serve_consistency.py` (NOUVEAU)
- **Tâches** :
  - Pour chaque modèle enregistré, snapshot 100 lignes de features brutes en train, scorer batch, sauver `models/<name>.train_snapshot.parquet`
  - En CI, recharger snapshot, scorer en mode serve (via `model_registry.load`), assert `np.allclose(preds_train, preds_serve, atol=1e-6)`
  - Détecte : modifications scaler implicites, version sklearn/lightgbm changée, feature reorder.
- **Heures** : 12h

**P1.3 — CPCV pipeline industrialisée**
- **Fichier** : `scripts/cpcv_run.py` (NOUVEAU) wrappant `src/research/cpcv_harness.py`
- **Tâche** : CLI standardisée pour tout nouveau modèle :
  ```
  python scripts/cpcv_run.py --model factor_model_v1 \
                              --data data/research/a1_matrix_2019_2026.parquet \
                              --target r_forward_4 \
                              --folds 8 --test-folds 2 --embargo 16 \
                              --output reports/cpcv/factor_model_v1.json
  ```
- **Hook CI** : tout PR qui modifie `models/` ou `src/intelligence/scoring/` DOIT joindre un rapport CPCV récent (< 7j) ET passer `evaluate_gates(...).all_passed`.
- **Heures** : 14h

**P1.4 — Walk-forward refit hebdomadaire automatisé**
- **Fichier** : `scripts/refit_weekly.py` (NOUVEAU), cron entry dans `infrastructure/docker-compose.yml`
- **Tâche** : chaque dimanche 02:00 UTC :
  - Recharger `a1_matrix` ou équivalent jusqu'à `now()`
  - Re-fit `scoring_v3_lgbm`, `calibrated_conviction` isotonic, ACI buffer
  - Re-run `evaluate_gates` sur la dernière fenêtre 12 mois
  - Si gates passées ⇒ enregistrer `scoring_v3_lgbm@v3.X+1` dans registry, promouvoir `latest_passing`
  - Sinon ⇒ alerte, garde le modèle précédent
- **Heures** : 16h

**P1.5 — Feature versioning / lineage**
- **Fichier** : `src/research/feature_lineage.py` (NOUVEAU)
- **Tâche** : pour chaque feature, tracer (source_dataset_hash, build_function_name, build_function_version). Stocker dans `data/research/feature_lineage.json`. Anti-leak vérification : `assert feature.vintage_date <= bar_ts` à la construction.
- **Heures** : 10h

---

### P2 — Online learning, feature store (Sprints 6-8, 60h)

**P2.1 — Feature store léger (Parquet partitionné)**
- **Fichier** : `src/research/feature_store.py` (NOUVEAU)
- **Tâche** : remplacer rebuild from scratch (`a1_features.py:23-38`) par lecture partitionnée par mois ; mise à jour incrémentale en CRON quotidien.
- **Heures** : 20h

**P2.2 — Online learning étendu (ACI déjà partiel, ajouter EnbPI)**
- **Fichier** : `src/intelligence/conformal/enbpi.py` (NOUVEAU)
- **Référence** : Xu & Xie (2021) "Conformal prediction interval for dynamic time-series".
- **Tâche** : bootstrap online des intervalles conformes, plus robuste qu'ACI sur séries fortement non-stationnaires.
- **Heures** : 16h

**P2.3 — Online LightGBM (incrémental)**
- **Fichier** : `src/intelligence/scoring/online_lgbm.py` (NOUVEAU)
- **Tâche** : `lgbm.train(init_model=previous_model, keep_training_booster=True, num_iterations=10)` chaque semaine au lieu de refit complet. Réduit cold-start.
- **Critère** : refit hebdo < 5 min wall-clock.
- **Heures** : 12h

**P2.4 — Bandit / contextual sélection modèle**
- **Fichier** : `src/intelligence/model_selector.py` (NOUVEAU)
- **Tâche** : si plusieurs modèles passent gates (e.g. `scoring_v3_lgbm`, `factor_model_v1`), Thompson sampling bandit choisit en live celui qui maximise Brier rolling 30j sur la cohorte courante.
- **Heures** : 12h

---

## 5. Tests & validation (CPCV, DSR, PBO, DM, gates institutionnels)

### 5.1 Gates obligatoires (intangibles)

Définis dans `src/research/strategy_gates.py:73-79`, **à ne pas affaiblir sans signature solo-founder + Sofia** :

- **DSR ≥ 1.5** (Bailey & López de Prado 2014)
- **PBO ≤ 0.35** (Bailey-Borwein-LdP-Zhu 2014)
- **PF lower bootstrap CI 95% > 1.00**
- **Diebold-Mariano p-value < 0.05** vs baseline constant
- **n_trades ≥ 30** (min sample size pour estimation fiable)

### 5.2 Suites de tests à compléter

| Test | Statut | Action |
|---|---|---|
| `tests/test_strategy_gates.py` | 12/12 ✅ | Maintenir |
| `tests/test_cpcv_harness.py` | OK | Maintenir |
| `tests/test_conformal_wrapper.py` | 14/14 ✅ | Ajouter "no future leakage in live mode" (P0-B.2.1) |
| `tests/test_regime_gate.py` | 11/11 ✅ | Ajouter test perf (< 5ms p99) (P0-B.2.2) |
| `tests/test_calibrated_conviction.py` | OK | Ajouter assert ECE < 0.05 (P0-B.1.1) |
| `tests/test_score_calibration.py` | OK | Maintenir |
| `tests/test_lgbm_vol.py` | OK | Maintenir |
| `tests/test_train_serve_consistency.py` | À CRÉER (P1.2) | **P1** |
| `tests/test_drift_monitor.py` | À CRÉER (P0-B.3.2) | **P0** |
| `tests/test_model_registry.py` | À CRÉER (P0-B.3.3) | **P0** |
| `tests/test_conformal_mondrian.py` | À CRÉER (P0-B.2.3) | **P1** |
| `tests/test_compliance_copy.py` | À CRÉER (P0-B.3.1) | **P0** |
| `tests/test_sentinel_scanner_regime_integration.py` | À CRÉER (P0-B.2.2) | **P0** |

### 5.3 Gates de commercialisation (au-delà des gates ML)

Avant **toute mise en avant marketing** d'un modèle :
1. CPCV 28 paths, gates 5/5 passées sur OOS 2024-2025 minimum.
2. ECE < 0.05 sur OOS.
3. Hit-rate cohorte best ≥ baseline + 5pp.
4. Manifest enregistré dans `models/registry.json` avec `gates_passed.all_passed=true`.
5. Train/serve consistency test vert (P1.2).
6. Drift monitoring déployé en prod ≥ 7 jours sans alerte critique (P0-B.3.2).

---

## 6. Sécurité (data leakage, look-ahead, train/serve consistency)

### 6.1 Anti-look-ahead (déjà partiellement enforced)

- **PIT vintage** : `a1_features.py:23-38` enforce `vintage_date <= bar_timestamp` ✅
- **Calendar features** : utilisent `scheduled_time` pas `actual_time` ✅
- **CPCV embargo** : 16 bars par défaut dans `cpcv_harness.py` ✅
- **À RENFORCER** :
  - Ajouter test paramétré sur 100 timestamps aléatoires de la matrix, vérifier qu'aucune feature ne fuit (déjà fait pour macro, à étendre aux features cross-asset si Vision A).
  - **Conformal `observe` ne doit JAMAIS être appelé en live avec le réalisé du même bar** ⇒ wire-in via callback `on_signal_outcome` dans `signal_tracker.py` après `signal_lifetime_bars` expiré.
  - Audit complet `apply_conformal_filter` (`conformal_wrapper.py:354-380`) : OK offline, mais en prod c'est `should_reject_live` qui doit être utilisé (à créer P0-B.2.1).

### 6.2 Train/serve skew prévention

- Snapshots Parquet train (P1.2) + assertion `allclose` en CI.
- **Pinning des dépendances** : `requirements.txt` doit figer `lightgbm==X.X.X`, `scikit-learn==X.X.X`, `numpy==X.X.X` ; tout bump = re-run CPCV + ECE.
- **Hash du modèle** : `manifest.json` contient SHA256 du `.pkl` ; refus de charger si mismatch.

### 6.3 Modèle non-deterministe ⇒ interdit prod

- `LGBMScorer.__init__` doit forcer `deterministic=True, force_row_wise=True` (déjà fait `factor_model/predictor.py:59-60`).
- Random seed fixé (déjà 42).
- Tests : `tests/test_determinism.py` (NOUVEAU 4h) bit-for-bit reproducible.

### 6.4 Risque modèle "passe gates par chance"

- **PBO ≤ 0.35** : strict.
- **Holm-Bonferroni** sur Sharpe inter-paths avec α=0.05 ⇒ si < ⌈k×28⌉ paths Holm-significants où k=0.5, rejeter.
- **Out-of-time forward test** : après gates passées en CPCV, run en paper trading 30 jours avant promotion prod (cf. `src/intelligence/forward_test_paper.py`).

---

## 7. Métriques (Brier, log-loss, calibration ECE, hit rate par cohorte)

### 7.1 Métriques modèle (live tracking obligatoire)

| Métrique | Cible | Source | Fréquence |
|---|---|---|---|
| Brier score | < 0.245 (vs base 0.25 = baseline P=0.5 maxi) | `calibration_metrics.py` (P0-B.1.1) | Daily rolling 30j |
| Log-loss | < 0.69 (= ln 2 baseline) | id. | Daily |
| ECE (M=10) | < 0.05 | id. | Daily |
| MCE | < 0.10 | id. | Daily |
| Reliability slope | ∈ [0.85, 1.15] | id. | Weekly |
| Hit-rate (overall) | > 50.5% | `signal_tracker.py` | Daily rolling 30j |
| Hit-rate × régime | best ≥ baseline + 5pp | `cohort_eval.py` (P0-B.1.2) | Weekly |
| AUC ROC | > 0.55 | `calibration_metrics.py` | Weekly |
| Conformal PICP | within ±5% de 1-α (cible 90%) | `conformal_wrapper.py` | Daily |
| PSI features | < 0.25 par feature | `drift_monitor.py` (P0-B.3.2) | Daily |

### 7.2 Métriques business (cross-réf catégorie 7 marketing)

| Métrique | Cible | Source |
|---|---|---|
| % signals émis (post conformal reject) | 40-70% du baseline | `signal_store.py` count par jour |
| Taux retention M+1 utilisateurs FREE | > 35% (vs baseline 20%) | À piper dans analytics |
| NPS post-30j d'usage | > 30 | Survey externe |

### 7.3 Dashboard

- Endpoint `GET /api/v1/metrics/ml-dashboard` (P1.1) ⇒ JSON consolidant les 11 lignes du tableau 7.1.
- Telegram bot admin reçoit alerte si :
  - ECE > 0.07 sur 3 jours consécutifs
  - PSI > 0.25 sur 1 jour
  - Hit-rate rolling 30j < baseline (50%)

---

## 8. Risques & mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| **A1 retry échoue à nouveau** | 55-70% | Vision A inutile, 250h perdues | Décider Vision B par défaut, A en option conditionnée Bloomberg |
| **Bloomberg consensus inaccessible / trop cher** | 40% | Vision A impossible | Backup : Trading Economics API ($299/mo) ou FF predictions (free, qualité moindre) |
| **Conformal `observe` fuit le futur en live (bug subtil)** | 25% | Backtest inflaté, déception client | Wire via callback post-`signal_lifetime_bars` + tests `test_no_future_leakage` |
| **Reject-option trop sélectif ⇒ 0 signaux** | 30% | Produit invisible, churn | Knob `breakeven` ajustable par tier, fallback à `RegimeGate.REDUCE` au lieu de BLOCK |
| **Train/serve skew silencieux** | 35% | Modèle "passé gates" produit mauvais signaux | Tests `test_train_serve_consistency.py` (P1.2) en CI bloquant |
| **Dérive régime (post-FOMC pivot, war shocks)** | 50% | ECE explose, calibration falsifiée | Drift monitor (P0-B.3.2) + ACI/Mondrian + refit weekly |
| **`factor_model_v1.pkl` claim 5/5 gates non-vérifiable** | 60% (memory ambiguë) | Crédibilité interne entamée | Audit P0.2 immédiat |
| **Surcoût LLM > 5% MRR (RAG narrative Phase 2B)** | 25% | Marge dégradée | Cf. catégorie LLM ; cap quotas |
| **Compliance UE 2024/2811 : claim probabiliste détecté** | 20% | Risque amende / blocage Stripe | `test_compliance_copy.py` (P0-B.3.1) regex sur outputs |
| **Survivorship bias dans calibration set** | 15% | ECE biaisée optimiste | Inclure trades skipped (state-machine) ET émis dans calib set |
| **PBO ≤ 0.35 pas atteint sur Vision B** | 30% | Pas de claim edge même calibré | Acceptable : Vision B ne revendique pas edge, juste calibration honnête |
| **Cohort slicing révèle 0 pocket d'edge** | 35% | Pas de pricing tier "STRATEGIST signals premium" | Plan B : pricing par features (event-window alerts) plutôt que par signal |

---

## 9. Dépendances

### 9.1 Données

- ✅ `data/research/a1_matrix_2019_2026.parquet` — matrix livré
- ✅ `data/macro/` — FRED + COT
- ✅ XAU M15 2019-2026 (`data/XAU_15MIN_2019_2026.csv`)
- ❌ **Bloomberg/Reuters consensus** — bloqueur Vision A (P0.1)
- ❌ **DXY/UST/SPX M15** intra-day — souhaitable pour Pilier 1 et cross-asset features (cf. cat. 1 data)
- ⚠️ **News feed live** : ForexFactory CSV OK historique, live RSS à fiabiliser (cf. cat. 4 news)

### 9.2 Briques sœurs (autres catégories)

- **Catégorie 1 Data Pipeline** : feature store (P2.1), Bloomberg ingestion, DXY/UST/SPX intra. **Sans data layer fiable, ML est suspendu.**
- **Catégorie 2 Backtest** : `cpcv_harness.py` + `strategy_gates.py` déjà livrés. Cat. 2 doit assurer la qualité OHLCV (cf. `data_quality_audit_2026_04_23.md` — XAU 2019-2025 à 63% coverage).
- **Catégorie 4 News** : surprise score (Vision A) dépend du news pipeline. `economic_calendar.py` déjà modifié dans gitStatus.
- **Catégorie 5 LLM/Narrative** : `score_calibration.py` fournit narrative phrases ; narrative quality tracker (`narrative_quality.py:1-20`) doit consommer hit-rate par cohorte.
- **Catégorie 6 Risk** : `sizing.py` (à créer dans Sprint A4 ou cat. 6) consomme `conviction_p_win` + `conformal_lower_0_100`.
- **Catégorie 8 Compliance** : interdit `edge_claim=true` tant que UE 2024/2811 validation pas signée Sofia.
- **Catégorie 9 Observability** : `drift_monitor.py` (P0-B.3.2) doit être surfacé dans `/metrics` (eval_16 a noté `/metrics` vide).

### 9.3 Bibliothèques

- ✅ `lightgbm`, `scikit-learn` (isotonic, sklearn), `scipy.stats`, `numpy`, `pandas` — déjà dans `requirements.txt`
- ⚠️ Pin versions à figer (catégorie infra / déploiement)

### 9.4 Décisions humaines bloquantes

- Go/No-go Vision A vs B (P0.1) — 1 semaine
- Audit cohérence factor_model vs A1 (P0.2) — 1 semaine
- Budget Bloomberg (si Vision A) — décision finance

---

## 10. Estimation totale & timeline

### 10.1 Effort par voie

| Voie | Sprints | Heures | Délai |
|---|---|---|---|
| **P0 Vision B (default)** | S1-S3 | 80h | 6-8 semaines (3-4 h/jour solo) |
| **P0 Vision A (conditional)** | S4-S8 | 210h dev + 40h data | 12-14 semaines additionnelles |
| **P1 Industrialisation** | S4-S5 (parallèle Vision A ou suite Vision B) | 60h | 4-5 semaines |
| **P2 Online + Feature store** | S6-S8 | 60h | 5-6 semaines |

### 10.2 Trois scénarios timeline

**Scénario Bleu (Vision B seule)** : 80h P0 + 60h P1 = **140h, ~10 semaines**, commercialisable Sprint S+10 en tier FREE+ANALYST orientés calibration honnête, pas de claim edge.

**Scénario Vert (Vision B + Vision A après data ingestion)** : 80h P0 Vision B (live commercial dès S+8) + 250h Vision A en parallèle ⇒ **330h sur ~6 mois**, tier STRATEGIST/INSTITUTIONAL unlocked si gates passent fin S+18.

**Scénario Rouge (Vision A solo, pas de Vision B)** : 250h Vision A + 60h industrialisation = **310h sur ~14 semaines**, MAIS **aucun produit commercialisable avant gates passés**. Risque commercial élevé : 55-70% chance d'arriver à S+14 sans pouvoir vendre.

### 10.3 Recommandation engagée

**Scénario Bleu d'abord, basculer en Vert si P0.1 valide Vision A.**

- Semaine 1 : P0.1 + P0.2 (8h)
- Semaines 2-7 : P0-B.1 + P0-B.2 + P0-B.3 (80h) ⇒ produit commercialisable
- Semaines 4-10 (parallèle) : P1.1 → P1.5 (60h)
- Semaines 6-14 (si Vision A go) : Sprint A1 → A4 (210h)
- Semaines 14-20 : P2 (60h)

**Total engagé immédiat** : 140h (Bleu) = **8-10 semaines** pour avoir un produit ML commercialisable avec gates honnêtes, calibration ECE prouvée, et reject-option live.

---

## Synthèse exécutive (5 lignes)

- **Chemin du livrable** : `C:\MyPythonProjects\TradingBOT_Agentic\reports\commercialization_sprint\03_machine_learning.md`
- **Top 3 P0** :
  1. P0.1 décision Vision A retry vs Vision B narrative-first (4h) — par défaut Vision B (verdict A1 sans ambiguïté)
  2. P0.2 audit incohérence `factor_model` (`lgbm_scoring_engine.py:1-25` claim 5/5 gates) vs A1 verdict (4h) — bloquer claim edge tant que non audité
  3. P0-B.1 calibration honnête bout-en-bout (ECE eval + cohorte slicing + intégration `InsightSignalV2`, 24h) — produit commercialisable en 8 semaines sans revendiquer edge prédictif
- **Heures totales engagées** : Vision B + P1 = **140h** (8-10 semaines) ; Vision A optionnelle +210h conditionnée à ingestion Bloomberg ; P2 +60h sur S+14-20. **Plafond max scénario complet : 330h sur 6 mois**.
