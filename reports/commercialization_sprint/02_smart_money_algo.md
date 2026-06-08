# Plan de Commercialisation — Catégorie 2 : Système Algorithmique d'Analyse Marché (Smart Money / Confluence)

> **Statut** : plan exhaustif Sprint 0 → Go-Live. AUCUNE modification code dans ce livrable.
> **Auteur** : Claude (Lead Quant / Smart Money), 2026-05-21.
> **Périmètre strict** : production de signal (SmartMoneyEngine + ConfluenceDetector + retest state machine + scoring + tiers).
> **Hors périmètre** : Volatility forecasting (cat. 3), News (cat. 4), LLM narratives (cat. 5), API/Delivery (cat. 6), MLOps/Backtest (cat. 7), Data Layer (cat. 1).
> **Travail parallèle référencé** :
>   * MTF rewiring Phase 1.1 (lookback 200 → 800 + warm-up guard) **DONE** — cf. `src/intelligence/sentinel_scanner.py:86-90,181-186,427`.
>   * MTF Phase 1.6 (intégration tests) en cours.
>   * MTF Phase 2 (empirical replay XAU 6 ans) à venir.
>   * MTF Phase 3 (activer `htf_alignment` weight + re-sweep tiers) à venir.
>   * Ce plan **ne dédouble pas** ce travail : il le suppose acquis et construit le scoring ML par-dessus en consommant `HTF_TREND_1H/4H` comme features.

---

## Table des matières

1. État actuel (audit, citations file:line)
2. Vision cible (institutional-grade)
3. Gap analysis (Current / Target / Severity)
4. Plan d'exécution
   * P0 — Bloquants commercialisation
   * P1 — Performance & robustesse
   * P2 — Extensions / différenciation
5. Tests & validation (replay 6-ans, walk-forward, CPCV, IC bootstrap, DSR/PBO)
6. Sécurité (input validation OHLCV, edge cases divisions zéro, lookback warm-up, parité Numba)
7. Métriques de succès
8. Risques & mitigations
9. Dépendances autres catégories
10. Estimation totale & timeline (Gantt + résumé)
11. Annexes — sources et chemins

---

## 1. État actuel (Audit)

### 1.1 Cartographie code

| Bloc | Fichier:lignes | Statut |
|---|---|---|
| Fractals causaux (Bill Williams 5 bars, shift N) | `src/environment/strategy_features.py:606-644` | ✅ Conforme, raise on leak `:692-696` |
| FVG 3-bougies + threshold ATR | `src/environment/strategy_features.py:650-679` | ⚠️ Seuil `FVG_THRESHOLD=0.1` ≈ spread broker |
| Numba BOS/CHOCH | `src/environment/strategy_features.py:32-134` | ✅ post-Sprint 1 fix |
| Python fallback BOS/CHOCH | `src/environment/strategy_features.py:159-234` | Parité validée |
| Retest state machine Numba | `src/environment/strategy_features.py:259-334` | ⚠️ `RETEST_TOL_ATR=0.5` ≈ spread |
| Retest state machine Python | `src/environment/strategy_features.py:337-407` | Parité conditionnelle (cf. P1-F2) |
| **Order Blocks (engulfing, PAS ICT)** | `src/environment/strategy_features.py:756-815` | ❌ Non-conforme ICT (P0-F1) |
| RSI Divergence (bug indexage) | `src/environment/strategy_features.py:817-879` | ❌ Compare `lows[i]` au lieu de `down_fractals[i]` (P1-F3) |
| `SMCConfig` Pydantic v1 | `src/environment/strategy_features.py:440-518` | OK (seul point de paramétrage) |
| `SmartMoneyEngine.analyze()` pipeline | `src/environment/strategy_features.py:881-950` | Recompute full à chaque appel (P2-F1) |
| Facade institutionnelle | `src/intelligence/smart_money/__init__.py:63-75` | Re-export legacy, extraction physique = Sprint 6 |
| `ConfluenceDetector` 8 composants (somme pondérée) | `src/intelligence/confluence_detector.py:117-132, 195-385` | ⚠️ Pearson(score, R) = −0.023 — non prédictif |
| Default weights | `src/intelligence/confluence_detector.py:117-132` | ❌ Aucune CV/grid search (P0-F2) |
| `_classify_tier` cutpoints 55/40/25 | `src/intelligence/confluence_detector.py:715-726` | Recalibrés 2026-04-29 sur p90/p50/p10 |
| Renormalisation données absentes | `src/intelligence/confluence_detector.py:272-291` | ✅ |
| Gates upstream (news+BOS+retest) | `src/intelligence/confluence_detector.py:228-258` | OK (mais sélectivité ≠ score) |
| `htf_alignment` component (weight=0, observability only) | `src/intelligence/confluence_detector.py:126-132, 626-690` | ✅ Phase 1 MTF — wired |
| Scanner orchestrateur | `src/intelligence/sentinel_scanner.py:310-536` | Polling 60s, regime filter wire OK |
| Logistic L1 scorer (scaffold) | `src/intelligence/scoring/logistic_l1.py:74-100` | Requires component scores persistés (absents) |

### 1.2 Métriques empiriques (sources : `reports/eval_02_confluence.md`, `reports/eval_03/eval_03_smart_money.md`, `reports/audit_2026_04_30_v2_timeout64.md`, `audits/2026-Q2/section_3_2_smart_money.md`, `audits/2026-Q2/section_3_3_confluence.md`)

| Métrique | Valeur | Référence |
|---|---:|---|
| Pearson(confluence_score, R) | **−0.023** (eval_02) / **−0.0075** (timeout=64) | `reports/eval_02_confluence.md:176`, `reports/audit_2026_04_30_v2_timeout64.md:24` |
| Spearman(score, R) | **−0.016** | `reports/eval_02_confluence.md:178` |
| Brier model vs baseline | 0.2551 vs 0.2456 (modèle **pire** que constant) | `reports/eval_02_confluence.md:166-167` |
| Plafond empirique du score | **55.5/100** (max observé sur 106 518 bars) | `reports/eval_02_confluence.md:122-126` |
| % bars ≥ 80 (PREMIUM pré-recal) | **0 %** sur 7 ans | `reports/eval_02_confluence.md:131-137` |
| Profit factor armed v1 (timeout=24) | **0.94** | `reports/eval_03/eval_03_smart_money.md:78`, `reports/audit_2026_04_30_v2_timeout64.md:13` |
| PF v2 (timeout=64, quick-win) | **1.037** | `reports/audit_2026_04_30_v2_timeout64.md:13` |
| Win rate armed (LONG / SHORT) | 33.5 % / **30.3 %** | `reports/eval_03/eval_03_smart_money.md:76` |
| OB count / 6 ans | 34 722 (≈ 1 OB tous les 4 bars) | `reports/eval_03/eval_03_smart_money.md:65-99` |
| OB anchored ≤ 20 bars d'un BOS | **59.5 %** bull / 59.3 % bear → 40 % non-ICT | `audits/2026-Q2/section_3_2_smart_money.md:213-219` |
| FVG count / 6 ans | 22 548 (1 bar sur 6) | `reports/eval_03/eval_03_smart_money.md:46-47` |
| Retest success rate (tol 0.5 ATR) | **89.8 %** (faux confirmes massifs) | `reports/eval_03/eval_03_smart_money.md:15, 68` |
| Filter empirique R1+R2+R3 (Q4_high + NY + Tuesday + long-only) | PF_test **1.472** sur 315 trades | `reports/feature_filter_audit.md:9-25` |
| Régime filter wire | PF 1.13 → **1.355 OOS** | `reports/feature_filter_audit.md`, `src/intelligence/sentinel_scanner.py:133-138` |
| FVG_SIGNAL_aligned (seul feature avec edge stable) | Pearson_test **+0.071** | `reports/feature_edge_audit.md:21` |
| Performance pipeline (Python fallback, 172k bars) | **12.1 s** | `audits/2026-Q2/section_3_2_smart_money.md:450-459` |
| Performance Numba (extrapolé) | ~0.5 s | `audits/2026-Q2/section_3_2_smart_money.md:476` |

### 1.3 Notes globales (eval / audit)

| Source | Note | Verdict |
|---|---:|---|
| `reports/eval_02_confluence.md` (ConfluenceDetector) | **2/10** | « Remplacer la fonction de score, garder le framework » |
| `reports/eval_03/eval_03_smart_money.md` (SMC engine) | **4.5/10** | « Difficile à défendre face à smc-python / LuxAlgo OSS sans durcir » |
| `audits/2026-Q2/section_3_2_smart_money.md` (Sprint 0 audit code-path) | **6.0/10** | « Code path solide, sémantique ICT partielle (OB), perf marginale » |
| `audits/2026-Q2/section_3_3_confluence.md` (Sprint 0 audit confluence) | **3.0/10** | « Empiriquement non prédictif, poids sans justification empirique » |

### 1.4 Synthèse — pourquoi ça ne se vend pas (encore)

1. **Le scoring est sans pouvoir prédictif** (Pearson −0.023, Brier pire que baseline) → un client SMC-savvy va le détecter via reliability diagram en < 1h.
2. **Les Order Blocks ne sont pas des Order Blocks** (engulfing brut sans anchor BOS) → 40 % des OB ne sont **pas** près d'un BOS, signal ≈ bruit. Un concurrent comme LuxAlgo / smc-python (Joshyattridge) implémente l'OB correctement.
3. **Le retest tol 0.5 ATR ≈ spread broker XAU** → 89.8 % de retests « confirmés » mécaniquement, faux confirmes.
4. **Tiers PREMIUM (≥55 post-recal, ≥80 pre-recal) quasi-vides** sur 7 ans de XAU M15. Le marketing « PREMIUM = haute conviction » est non substanciable.
5. **0 / 4 stratégies passent PF lower-bound CI 95 % > 1.0** (`reports/decision_matrix_2026_04_30.md`) → pas de claim de track-record bootstrap-validé possible.
6. **Quick-win timeout 24 → 64 a déjà fait passer le PF 0.79 → 1.04** mais le bug exit "opposite" (35.6 % des sorties) reste à neutraliser (`reports/audit_2026_04_30_v2_timeout64.md:88-99`).

---

## 2. Vision cible — Système algorithmique commercialisable

Un système algo « institutional-grade » qu'on peut vendre $99-$299/mois (B2C) ou $999-$4 999/mois (B2B-API) doit satisfaire **simultanément** :

### 2.1 Conformité conceptuelle (ce qu'achète le client SMC)

* **OB ICT-strict** : anchor obligatoire à un BOS event antérieur (< N bars), filtre body/ATR ≥ 1.0 sur la bougie d'impulse, tracking de mitigation (l'OB persiste tant que non touché).
* **FVG threshold ≥ 0.4 ATR** : élimine les micro-gaps de la taille du spread. Tracking `FVG_FILLED_PCT` pour rendu visuel.
* **Retest strict** : `low ∈ [level − 0.25 ATR, level + 0.25 ATR]` ET retour au-dessus dans la même bar (touch). Tolérance < spread broker.
* **BOS/CHOCH** : déjà conforme post-Sprint 1 (`bos_event` 1/−1 only on break bar, suppression répétée via `last_bos_up_level`). **Garder tel quel**, ajouter test parité Numba (gap actuel).
* **RSI Divergence indexée correctement** : comparer `down_fractals[i]` (swing low) et `rsi[i-N]` (RSI au swing), pas `lows[i]` et `rsi[i]`.
* **Liquidity Sweep detector** (extension Sprint 4) : equal highs/lows ± 0.1 ATR, balayage strict.
* **Premium/Discount zones** (extension Sprint 5) : Fibonacci institutionnel 50 % swing range.

### 2.2 Pouvoir prédictif statistiquement audité

* **Scoring supervisé** (Logistic L1 ou LightGBM calibré isotonic) entraîné walk-forward, target = `R_multiple > 0`.
* **Brier skill score ≥ +5 %** vs base rate (vs −2.2 % aujourd'hui).
* **Spearman(p_win, R) ≥ +0.10** out-of-sample.
* **Win-rate PREMIUM − Win-rate STANDARD ≥ +8pp** prouvé par bootstrap CI 95 %.
* **CPCV (Combinatorial Purged Cross-Validation)** sur 5 folds avec embargo ≥ 50 bars, DSR ≥ 1.0, PBO ≤ 0.30, DM-test p < 0.05 contre baseline naïve.

### 2.3 Robustesse opérationnelle

* **Latence par bar < 50 ms p99** en live scan (vs 70 µs / bar batch actuel, mais recompute total = inacceptable en stream — voir P1).
* **Cache incrémental** : ne re-calcule que la fenêtre touchée par la nouvelle bar.
* **Parité Numba ↔ Python** testée sur BOS/CHOCH (actuellement seul retest est testé — gap P2-F8).
* **Assertion `NUMBA_AVAILABLE`** au démarrage prod, sinon alerte (sans ça, latence × 25 — pas vu en local Windows).
* **Validation OHLCV stricte** : coverage ≥ 95 %, pas de NaN, monotone temporelle, no negative spreads.
* **Lookback warm-up guard** déjà appliqué Phase 1.1 MTF (`src/intelligence/sentinel_scanner.py:181-186`), à étendre au niveau confluence (FRACTAL_WINDOW × 10 minimum).

### 2.4 Différenciation commerciale (rendu, explainability)

* **Component breakdown** persisté par signal : chaque `weighted_score` exposé via `/signals/{id}/breakdown`.
* **Chart renderer** PNG annoté (OB shaded, FVG coloré, BOS/CHOCH étiquetés, retest state) — livrable Telegram + API. Différencie de smc-python.
* **SHAP values** sur le modèle ML pour expliquer pourquoi un signal est PREMIUM.
* **Multi-timeframe** : H1 + H4 alignement comme feature de scoring (déjà observability-only — Phase 3 lift weight).
* **Mitigation tracking** : `BULLISH_OB_HIGH/LOW` persiste tant que non touché. Compte des mitigations.

### 2.5 Composantes du « contrat client »

Un client paye pour :
1. **Un signal directionnel** (BUY/SELL/HOLD) — state machine `signal_state_machine.py` ✅ ok.
2. **Un niveau de conviction** (PREMIUM / STANDARD / WEAK) — actuellement non substancié.
3. **Une explication** (composants score + narrative LLM) — UX existe, scoring n'a pas de valeur prédictive donc l'explication n'a aucun ancrage.
4. **Un track-record auditable** (PF, Sharpe, DSR, IC bootstrap) — pas livrable aujourd'hui (CI < 1.0).
5. **Une UI/livraison** (Telegram, webapp, API JSON) — hors périmètre cat. 2 mais consomme le signal.

---

## 3. Gap Analysis

| # | Dimension | Current | Target | Severity | Effort |
|---|---|---|---|:---:|:---:|
| G1 | OB détection | Engulfing brut (`strategy_features.py:766-789`), 34 722 OB / 6 ans dont 40 % sans BOS anchor | OB ICT-strict : anchor BOS ≤ 20 bars + body/ATR ≥ 1.0 + mitigation tracking | **P0** | L |
| G2 | FVG threshold | 0.1 ATR (≈ spread broker, 22 548 FVG / 6 ans) | 0.4 ATR + `FVG_FILLED_PCT` (≈ 4 700 FVG / 6 ans) | **P0** | S |
| G3 | Retest tolerance | 0.5 ATR (faux confirmes 89.8 %) | 0.25 ATR + touch strict (low traverse level) | **P0** | S |
| G4 | Scoring fn | Somme pondérée additive (`confluence_detector.py:269`), Pearson −0.023 | Logistic L1 ou LightGBM walk-forward, calibration isotonic, Brier skill ≥ +5 % | **P0** | XL |
| G5 | Component scores persistés | Non (seul `total_score` stocké) | `component_scores[]` colonne `score_distribution.csv` + `trades_combined.csv` + signal_id table | **P0** | S |
| G6 | RSI Divergence indexage | `lows[i]` au lieu de `down_fractals[i]` (`strategy_features.py:849-857`) | `down_fractals[i]` + `rsi[i-N]` (N=FRACTAL_WINDOW) | **P1** | XS |
| G7 | `armed_window` default mismatch | 5 (`strategy_features.py:420`) vs 30 (`SMCConfig.RETEST_ARMED_WINDOW:510`) | Alignés ou `kwargs` required | **P1** | XS |
| G8 | DEFAULT_WEIGHTS justification | Aucune CV / grid search (`confluence_detector.py:117-132`) | Sweep CPCV-validé OU obsolescence si G4 ML | **P0** | (couvert par G4) |
| G9 | Tier cutpoints empiriques | 55/40/25 alignés p90/p50/p10 distribution post-RegimeFilter | Cutpoints validés bootstrap CI sur win-rate par tier | **P1** | S |
| G10 | Numba parity test BOS/CHOCH | Absent (uniquement retest) | Test parité Numba ↔ Python systématique | **P1** | XS |
| G11 | Numba assertion en prod | Aucune | Assertion `NUMBA_AVAILABLE` au boot + alerte Sentry | **P1** | XS |
| G12 | Pipeline incrémental | Recompute full à chaque appel (`strategy_features.py:881-950`) | Sliding window cache + incremental update | **P1** | M |
| G13 | Chart renderer | Absent (`reports/eval_03/eval_03_smart_money.md:128-140`) | Module `src/intelligence/chart_renderer.py` PNG annoté | **P2** | M |
| G14 | Liquidity sweep detector | Absent | Equal highs/lows ± 0.1 ATR + balayage | **P2** | M |
| G15 | Premium/discount zones | Absent | Fibonacci institutionnel 50 % swing | **P2** | S |
| G16 | Mitigation tracking | Absent | OB persiste tant que non touché + `mitigation_count` | **P2** | M |
| G17 | Module physique `smart_money/` | Facade re-export only (`src/intelligence/smart_money/__init__.py:63-75`) | Extraction physique post-fixes ICT | **P1** | M |
| G18 | Bug exit "opposite" | 35.6 % des sorties (timeout=64) | Exit policy déterministe (TP/SL/timeout strict) | **P0** | M |
| G19 | DSR / PBO / CPCV gates | Naïf event-driven FAIL (`reports/3_pillars_implementation_2026_05_13.md`) | Gates institutionnels validés sur scoring v2 | **P0** | (couvert par G4 + dépendance cat. 7) |
| G20 | MTF feature wiring | Observability only (htf_alignment weight=0, `confluence_detector.py:126-132`) | Phase 3 lift weight après empirical validation Phase 2 | **P1** | (autre conv) |
| G21 | Input validation OHLCV | Validation amont (`sentinel_scanner.py:334-340`) OK | Étendre à : monotone timestamps, no negative spread, gap detection | **P2** | S |
| G22 | Lookback warm-up confluence | Pas de check explicite côté confluence | Refuser < N × max(periods) bars (avec N≥10) | **P1** | XS |
| G23 | Cross-instrument generalisation | Stats EUR comparables à XAU (audit 3.2 §4.3) | Walk-forward XAU + EUR + USOIL (cat. 1 dep) | **P1** | M |
| G24 | Différenciation commerciale | Note 3/10 eval_26 | Track-record audit + chart renderer + SHAP explainability | **P2** | (couvert par G4+G13) |

**Légende effort** : XS<2h, S=2-8h, M=1-3j, L=3-7j, XL>7j.

---

## 4. Plan d'exécution

### Convention

Chaque tâche : **Titre** / Fichiers / Heures / Acceptance Criteria / Dépendances. Effort `h` = développeur senior solo, hors review.

---

### P0 — Bloquants commercialisation (≤ 5 sem)

#### Task P0-1 : Order Block ICT-strict (anchor BOS + impulse filter + mitigation)

* **Fichiers concernés** :
  * `src/environment/strategy_features.py:756-815` (réécrire `_add_smc_order_blocks`)
  * `src/environment/strategy_features.py:440-518` (ajouter params `OB_BODY_ATR_MIN`, `OB_LOOKBACK_BARS`, `OB_MITIGATION_TRACKING` à `SMCConfig`)
  * `tests/test_sprint3_order_blocks.py` (étendre tests : anchor BOS, body filter, mitigation, comptage par BOS)
* **Heures** : 16-24 h (3 j)
* **Acceptance Criteria** :
  * (a) Pour chaque `bos_event=±1` à bar `i`, l'OB est la **dernière** bougie opposée (close < open pour BOS_UP, inversement) dans `[i - OB_LOOKBACK_BARS, i-1]` avec `|body| / ATR ≥ OB_BODY_ATR_MIN`.
  * (b) Stats : OB count / 6 ans XAU passe de **34 722 → < 6 000** (90 % anchored, body/ATR ≥ 1.0).
  * (c) `tests/test_sprint3_order_blocks_ict.py` : un OB sans BOS dans la fenêtre = 0 ; un OB avec BOS et body/ATR=0.5 = 0 ; un OB anchored avec body/ATR=1.5 = strength > 0.
  * (d) Colonne `BULLISH_OB_ANCHORED_BOS_TS` (timestamp du BOS qui anchored cet OB) persisté.
  * (e) Mitigation tracking : si `MITIGATION_TRACKING=True`, l'OB persiste tant que `low ≤ OB_HIGH AND high ≥ OB_LOW` ne se déclenche pas. Colonne `OB_MITIGATION_COUNT`.
* **Dépendances** : aucune (BOS event déjà conforme).

#### Task P0-2 : FVG threshold durci + tracking remplissage

* **Fichiers concernés** :
  * `src/environment/strategy_features.py:478-482` (`FVG_THRESHOLD: 0.1 → 0.4`, garder configurable)
  * `src/environment/strategy_features.py:650-679` (ajout colonne `FVG_FILLED_PCT`, `FVG_AGE_BARS`)
  * `tests/test_sprint5_fvg_threshold.py` (étendre : sweep 0.1/0.4/0.5, FVG_FILLED_PCT)
* **Heures** : 6-10 h (1 j)
* **Acceptance Criteria** :
  * (a) `FVG_THRESHOLD=0.4` par défaut, override possible.
  * (b) Stats : FVG count / 6 ans XAU passe de **22 548 → ~4 700** (cf. sweep `audits/2026-Q2/section_3_2_smart_money.md:144-150` : 0.4 ATR = 842 / 10k bars).
  * (c) `FVG_FILLED_PCT` ∈ [0, 1] mesure le % de la zone remplie par les bougies subséquentes. 1.0 = mitigé.
  * (d) `FVG_AGE_BARS` compte les bars depuis création (NaN si pas de FVG).
  * (e) Test : un FVG bullish 50 cents sur XAU (ATR=3 $) avec threshold 0.4 → `FVG_SIGNAL=0`. À 1.5 $ → `FVG_SIGNAL=1`.
* **Dépendances** : aucune.

#### Task P0-3 : Retest strict (tol 0.25 ATR + touch obligatoire)

* **Fichiers concernés** :
  * `src/environment/strategy_features.py:493-498` (`RETEST_TOL_ATR: 0.5 → 0.25`)
  * `src/environment/strategy_features.py:259-407` (modifier `_calculate_bos_retest_numba` + Python parity : exiger touch strict, pas juste proximity)
  * `tests/test_bos_retest.py` (étendre)
* **Heures** : 8-12 h (1.5 j)
* **Acceptance Criteria** :
  * (a) Transition `AWAITING → ARMED` exige : `low ≤ level + 0.25 × ATR` ET `high ≥ level − 0.25 × ATR` (low croise vraiment le level, pas juste approche).
  * (b) Stats : retest success rate passe de **89.8 % → < 60 %**.
  * (c) Win-rate post-armed (LONG) attendue **31.9 % → > 38 %** (cf. eval_03 §8 estimation).
  * (d) Test Numba/Python parity sur retest strict (cf. `tests/test_bos_retest.py::test_numba_and_python_agree`).
  * (e) Bug `armed_window=5 vs 30` (P1-F2) résolu en passant : alignement defaults OU `kwargs` obligatoires.
* **Dépendances** : aucune.

#### Task P0-4 : Persistence component_scores + reset DEFAULT_WEIGHTS-justification trail

* **Fichiers concernés** :
  * `src/intelligence/confluence_detector.py:84-110` (`to_dict`: déjà OK, ajouter colonne `component_scores_raw[]`)
  * `src/api/signal_store.py` (nouveau champ JSON `component_scores`)
  * `scripts/audit_backtest.py` (logger `component_scores` dans `score_distribution.csv`)
  * `reports/eval_02/score_distribution.csv` (re-générer 7 ans avec breakdown)
  * Doc : `docs/algo/scoring_weights_history.md` (registre changements weights)
* **Heures** : 8-12 h (1.5 j)
* **Acceptance Criteria** :
  * (a) Tous les `ConfluenceSignal` persistent les 8-9 component scores raw + weighted.
  * (b) `score_distribution.csv` regénéré avec colonnes `comp_bos, comp_fvg, comp_ob, ..., comp_htf` pour calculer corrélation 8×8 empirique (eval_02 §10 quick-win #1).
  * (c) Tableau matrice corrélation empirique généré dans `reports/eval_02/component_correlation.json`.
  * (d) Doc weights : pour chaque value de `DEFAULT_WEIGHTS`, citation de la source (eval_X, audit_X, ou « heuristique pre-data »).
* **Dépendances** : P0-1, P0-2, P0-3 (sinon données pourries).

#### Task P0-5 : Refonte scoring — Logistic L1 calibré (puis LightGBM)

* **Fichiers concernés** :
  * `src/intelligence/scoring/logistic_l1.py:74-100` (compléter implémentation, déjà scaffold)
  * `src/intelligence/scoring/calibration_loop.py` (NOUVEAU — walk-forward refit toutes les K trades)
  * `src/intelligence/scoring/isotonic.py` (NOUVEAU — calibration finale Probabilité→Probabilité)
  * `src/intelligence/confluence_detector.py:195-385` (`analyze`: nouvelle branche `if self._ml_scorer is not None: return self._ml_scorer.predict_proba(...)`)
  * `models/logistic_l1_xau_m15.pkl` (artefact pickle versionné)
  * `scripts/train_logistic_l1_on_sweep.py` (existe déjà — étendre walk-forward CPCV)
  * `tests/test_logistic_l1_scorer.py` (NOUVEAU)
* **Heures** : 40-56 h (5-7 j)
* **Acceptance Criteria** :
  * (a) Dataset walk-forward : XAU 2019-2023 train, 2024-2025 OOS test. Features = 8 component scores + HTF features + macro context (hour, dow, ATR_PCTL, BB_POS).
  * (b) Logistic L1 fit avec `class_weight='balanced'`, `solver='saga'`, grid search `C ∈ {0.01, 0.1, 1.0, 10}` sur 5-fold purged CV avec embargo 50 bars (CPCV).
  * (c) Calibration isotonic sur le 5e fold (held-out).
  * (d) Metrics OOS :
    * Brier skill score ≥ **+5 %** vs base rate (vs −2.2 % actuel).
    * Pearson(p_win, R_realized) ≥ **+0.10**.
    * Spearman ≥ **+0.12**.
    * Reliability diagram monotone (`is_winrate_monotone_up=true`).
    * AUC ≥ 0.55.
  * (e) Comparaison head-to-head vs Confluence additif : DM-test p < 0.05 sur Brier delta.
  * (f) Output JSON `reports/scoring/logistic_l1_eval.json` avec metrics + 95 % bootstrap CI.
  * (g) Si Logistic L1 < target, escalade LightGBM (effort +24 h) avec `max_depth=4, n_estimators=200, min_data_in_leaf=50, isotonic post-cal`.
  * (h) Shadow mode 2 semaines avant bascule prod (logs `score_v1` et `score_v2` côte-à-côte).
* **Dépendances** : P0-4 (component_scores persistés), cat. 7 (CPCV harness `src/research/strategy_gates.py`), cat. 1 (data layer XAU + EUR coverage ≥ 95 %).

#### Task P0-6 : Exit policy bug fix (suppression "opposite" surcomptage)

* **Fichiers concernés** :
  * `src/backtest/state_machine_replay.py` (audit la logique d'exit)
  * `src/intelligence/signal_state_machine.py` (revoir transition `ACTIVE → HOLD` sur "opposite")
  * `scripts/quant_audit_2026_04_30.py:CFG["max_lifetime_bars"] = 64` (déjà fixé, garder)
* **Heures** : 12-16 h (2 j)
* **Acceptance Criteria** :
  * (a) Exit reasons audités : `TP, SL, TIMEOUT, OPPOSITE_SIGNAL, SCORE_DECAYED` — distribution acceptable sur replay v2.
  * (b) Exit "opposite" passe de **35.6 % → < 15 %** (clamp : un signal opposé ne ferme pas la position avant 75 % de la durée moyenne d'un trade).
  * (c) Distribution finale attendue : TP ~15-20 %, SL ~25-30 %, TIMEOUT ~30-40 %, OPPOSITE ~10-15 %.
  * (d) Test régression replay : PF 1.04 → cible ≥ 1.20 grâce à la correction exit policy seule.
* **Dépendances** : P0-1, P0-2, P0-3 (sinon dataset trades pourri).

#### Task P0-7 : Replay validation 7 ans XAU avec gates institutionnels

* **Fichiers concernés** :
  * `src/research/strategy_gates.py` (déjà existant — cat. 7)
  * `scripts/replay_xau_v2_post_fixes.py` (NOUVEAU)
  * `reports/commercialization_sprint/02_replay_validation.md` (livrable)
* **Heures** : 16-24 h (3 j) + 8 h analyse
* **Acceptance Criteria** :
  * (a) Replay 7 ans XAU M15 post P0-1 à P0-6 fixes (OB ICT + FVG 0.4 + retest 0.25 + Logistic L1 score + exit fix).
  * (b) Metrics passées par gates institutionnels :
    * **PF ≥ 1.30** OOS (vs 1.04 v2 quick-win).
    * **PF lower-bound CI 95 % bootstrap ≥ 1.10**.
    * **Sharpe ann. ≥ 0.80**.
    * **DSR ≥ 1.0** (Deflated Sharpe Ratio).
    * **PBO ≤ 0.30** (Probability of Backtest Overfitting).
    * **DM-test p < 0.05** vs baseline « buy-and-hold filtré régime ».
    * **Max DD ≤ 35 %**.
    * **≥ 5 années sur 7 avec E[R] > 0**.
  * (c) Cross-instrument : run identique sur EUR/USD M15 — PF ≥ 1.15 (transferable edge proof).
  * (d) Si ≥ 1 gate FAIL, retour P0-5 avec features additionnelles ou pivot narrative-first (`memory:a1_verdict_2026_05_01.md` decision tree).
* **Dépendances** : P0-1 à P0-6, cat. 1 (data), cat. 7 (backtest harness).

#### Task P0-8 : Tier cutpoints validation bootstrap

* **Fichiers concernés** :
  * `src/intelligence/confluence_detector.py:715-726` (`_classify_tier`)
  * `scripts/calibrate_tier_cutpoints.py` (NOUVEAU)
  * `reports/scoring/tier_calibration.md` (livrable)
* **Heures** : 6-8 h (1 j)
* **Acceptance Criteria** :
  * (a) Sur dataset post-Logistic-L1, sweep cutpoints `p_premium ∈ {0.55, 0.60, 0.65, 0.70}` × `p_standard ∈ {0.40, 0.45, 0.50}`.
  * (b) Pour chaque (premium, standard), calculer : n_signals_premium / an, E[R] tier, win-rate tier, bootstrap CI 95 % sur win-rate.
  * (c) Choisir cutpoints qui maximisent **WR(PREMIUM) − WR(STANDARD) ≥ 8 pp** avec n_premium ≥ 30 / an (volume marketable).
  * (d) Doc final : table 4 colonnes (tier, cutpoint, signals/an, E[R], WR, CI).
* **Dépendances** : P0-5, P0-7.

**Sub-total P0** : **112-160 h** (~14-20 j senior solo) = **3-4 semaines** avec marge.

---

### P1 — Performance & robustesse (2-3 sem)

#### Task P1-1 : RSI Divergence indexage fix

* **Fichiers** : `src/environment/strategy_features.py:849-857, 865-873`
* **Heures** : 2-4 h
* **Acceptance** :
  * Remplacer `lows[i]` par `down_fractals[i]` (= prix au swing low) et `rsi[i]` par `rsi[i-N]` (RSI au swing). Idem up_fractals + bearish.
  * Test `tests/test_sprint7_rsi_divergence.py` étendu : un fractal swing à i-2 avec low=1900, RSI=30 ; bar de confirmation à i avec low=1905, RSI=35 → divergence détectée correctement sur 1900/30, pas 1905/35.
  * Stat empirique : décalage moyen `DOWN_FRACTAL − low(i) = -1.18 $` sur XAU (audit 3.2 §3.6) → après fix, divergences re-comptées sur les vraies valeurs swing.
* **Dépendances** : aucune.

#### Task P1-2 : armed_window default alignement

* **Fichiers** : `src/environment/strategy_features.py:420, 510-518`
* **Heures** : 1-2 h
* **Acceptance** : `calculate_bos_retest_fast` default `armed_window=5 → 30` OU obliger kwargs. Test régression dans `tests/test_bos_retest.py` qui force la cohérence (fail si mismatch).
* **Dépendances** : aucune.

#### Task P1-3 : Numba parity test BOS/CHOCH

* **Fichiers** : `tests/test_bos_choch_numba_parity.py` (NOUVEAU)
* **Heures** : 3-4 h
* **Acceptance** : Sur XAU 5k bars + synthetic edge cases, `_calculate_bos_choch_numba` et `_calculate_bos_choch_python` produisent les 4 arrays bit-identiques. Si Numba indispo, fallback testé. Test ajouté au CI.
* **Dépendances** : aucune.

#### Task P1-4 : Assertion Numba prod + alerte

* **Fichiers** : `src/intelligence/main.py:1-50` (entry point), `src/intelligence/sentinel_scanner.py:start()`
* **Heures** : 2-3 h
* **Acceptance** :
  * Au démarrage prod (`NARRATIVE_MODE=llm OR ENV=production`), si `NUMBA_AVAILABLE=False`, raise `RuntimeError("Numba required in production — fallback latency ×25 unacceptable")`.
  * En staging/test, log warning + métrique Prometheus `numba_fallback_active=1`.
  * Test `tests/test_main_numba_check.py`.
* **Dépendances** : aucune.

#### Task P1-5 : Pipeline incrémental (sliding window cache)

* **Fichiers** :
  * `src/environment/strategy_features.py:881-950` (refactor `analyze()` en `analyze_incremental(new_bars: pd.DataFrame, prev_state: SMCState)`)
  * `src/environment/strategy_features.py:537-562` (`SmartMoneyEngine` : maintenir `_state: SMCState` interne)
  * `tests/test_smc_incremental.py` (NOUVEAU)
* **Heures** : 24-32 h (3-4 j)
* **Acceptance** :
  * (a) Sur 100 appels successifs avec 1 nouvelle bar à chaque : latence p99 < **50 ms / bar** (vs ~12.1 s / 172 k bars = 70 µs en batch, mais en stream = recompute total).
  * (b) Résultats bit-identiques à `analyze()` batch sur les mêmes 100 bars.
  * (c) Memory footprint stable (< 100 MB pour XAU 6 ans sliding window 1000 bars).
* **Dépendances** : aucune (mais utile pour P2-F1 reco audit).

#### Task P1-6 : Cross-instrument validation (EUR/USD, USOIL)

* **Fichiers** :
  * `scripts/replay_eurusd_post_fixes.py` (NOUVEAU)
  * `scripts/replay_usoil_post_fixes.py` (NOUVEAU)
  * `reports/commercialization_sprint/02_cross_instrument.md` (livrable)
* **Heures** : 16-24 h (2-3 j, dépend data layer cat. 1)
* **Acceptance** :
  * EUR/USD M15 PF ≥ 1.15 OOS post-fixes.
  * USOIL M15 PF ≥ 1.10 OOS post-fixes (si dataset cat. 1 dispo).
  * Stable across regimes (bull/bear/range) — pas de one-trick-pony.
* **Dépendances** : P0-7, cat. 1 (data layer EUR + USOIL).

#### Task P1-7 : Lookback warm-up guard confluence

* **Fichiers** : `src/intelligence/confluence_detector.py:195-220` (early return si `len(smc_df) < FRACTAL_WINDOW × 10`)
* **Heures** : 2-3 h
* **Acceptance** :
  * Refuser tout `analyze()` avec moins de `max(RSI_WINDOW × 5, FRACTAL_WINDOW × 10) = 50` bars warm-up.
  * Test : 30 bars d'OHLCV → `analyze()` return None + log warning.
* **Dépendances** : aucune.

#### Task P1-8 : Module physique `src/intelligence/smart_money/`

* **Fichiers** :
  * `src/intelligence/smart_money/fractals.py` (extract `:606-644`)
  * `src/intelligence/smart_money/fvg.py` (extract `:650-679`)
  * `src/intelligence/smart_money/bos_choch.py` (extract `:32-234, :712-754`)
  * `src/intelligence/smart_money/order_blocks.py` (P0-1 ré-écrit ici)
  * `src/intelligence/smart_money/retest.py` (extract `:259-436`)
  * `src/intelligence/smart_money/divergence.py` (P1-1 fixé ici)
  * `src/intelligence/smart_money/engine.py` (`SmartMoneyEngine` réorganisé)
  * `src/environment/strategy_features.py` : ne garder que RSI/MACD/BB/ATR (legacy compat)
* **Heures** : 24-32 h (3-4 j)
* **Acceptance** :
  * Tous les imports `from src.intelligence.smart_money import SmartMoneyEngine` fonctionnent (facade existante préservée).
  * Tous les 1366+ tests passent sans modification.
  * Audit dependency : `strategy_features.py` n'est plus dans le sys path de prod (que pour RL legacy).
* **Dépendances** : P0-1, P1-1 (sinon on déplace du code buggé).

**Sub-total P1** : **74-104 h** (~10-13 j) = **2-3 semaines**.

---

### P2 — Extensions / différenciation (3-5 sem, mais peut s'étaler post Go-Live)

#### Task P2-1 : Chart renderer (PNG annoté)

* **Fichiers** : `src/intelligence/chart_renderer.py` (NOUVEAU), `src/delivery/telegram_notifier.py` (attacher PNG), `src/api/routes/signals.py` (endpoint `/signals/{id}/chart`)
* **Heures** : 24-32 h (3-4 j)
* **Acceptance** :
  * Génère PNG 1200×800 avec : 200 bars OHLC, OB shaded (boxes), FVG colorés, BOS/CHOCH labels, retest state visuel, SL/TP lines.
  * Latence < 500 ms / chart.
  * Envoyé en attachment Telegram par `send_signal(signal, narrative_data, chart_png_bytes)`.
* **Dépendances** : P0-1 (OB ICT pour rendu propre), librairie `mplfinance` ou `lightweight-charts`.

#### Task P2-2 : Liquidity sweep detector

* **Fichiers** : `src/intelligence/smart_money/liquidity_sweeps.py` (NOUVEAU), `src/environment/strategy_features.py` ajouter colonne `LIQ_SWEEP_SIGNAL`
* **Heures** : 16-20 h (2-3 j)
* **Acceptance** :
  * Détecter `equal_highs` : 2 swings high dans `[i-50, i]` avec écart ≤ 0.1 ATR.
  * Détecter `sweep` : prix dépasse equal_high de < 0.2 ATR puis ferme en-dessous dans la bar suivante.
  * Stat sur XAU 6 ans : N sweeps détectés, % suivi d'un retournement ≥ 1 ATR dans les 5 bars (validation ICT).
  * Feature `LIQ_SWEEP_SIGNAL` ∈ {-1, 0, +1} ajouté au confluence engine (weight initial 0, observability only — comme htf_alignment).
* **Dépendances** : aucune.

#### Task P2-3 : Premium/Discount zones (Fibonacci institutionnel)

* **Fichiers** : `src/intelligence/smart_money/premium_discount.py` (NOUVEAU)
* **Heures** : 8-12 h (1-1.5 j)
* **Acceptance** :
  * Pour chaque swing range complet (low → high), calculer `eq_zone = mid(low, high)`, `premium = [mid, high]`, `discount = [low, mid]`.
  * Colonne `PRICE_ZONE` ∈ {premium, discount, equilibrium}.
  * Feature consommée par Logistic L1 (P0-5).
* **Dépendances** : aucune.

#### Task P2-4 : SHAP values dans narrative

* **Fichiers** : `src/intelligence/scoring/explain.py` (NOUVEAU), `src/intelligence/llm_narrative_engine.py` (consommer SHAP top-3 features)
* **Heures** : 12-16 h (1.5-2 j)
* **Acceptance** :
  * Pour chaque signal Logistic L1 / LightGBM, retourner top-3 SHAP contributions (positives + négatives).
  * LLM narrative prompt enrichi : « ce signal est PREMIUM parce que [SHAP+1: HTF aligned H4 +0.18, SHAP+2: FVG 0.6 ATR +0.12, SHAP+3: NY session avoided +0.09] ».
  * Test : `signal.scoring_explanation.top_drivers` non vide.
* **Dépendances** : P0-5.

#### Task P2-5 : OB mitigation tracking & viz

* **Fichiers** : `src/intelligence/smart_money/order_blocks.py` (extend), `src/intelligence/chart_renderer.py` (extend)
* **Heures** : 8-12 h
* **Acceptance** :
  * OB persiste tant que `low > OB_LOW AND high < OB_HIGH` n'est pas vrai. Sinon `OB_MITIGATED=True` + `OB_MITIGATION_BAR_IDX=i`.
  * Compteur `OB_MITIGATION_COUNT` (n fois prix touché sans casser).
  * Render : OB virage gris (mitigé) ou plein (actif).
* **Dépendances** : P0-1, P2-1.

#### Task P2-6 : MTF Phase 3 lift (post Phase 2 empirical)

* **Note** : géré par l'autre conversation, ici **on attend** Phase 2 results pour décider si on lift `htf_alignment` weight 0 → ~13 ou si on retire.
* **Fichiers** : `src/intelligence/confluence_detector.py:126-132` (lifter weight) + re-sweep tiers (P0-8)
* **Heures** : 4-6 h (modification ponctuelle + re-validation tiers)
* **Acceptance** : décision documentée dans `reports/commercialization_sprint/02_mtf_phase3_decision.md` selon Phase 2 empirical.
* **Dépendances** : Phase 2 MTF (autre conv) + P0-5 (re-fit Logistic L1 avec htf features actives).

#### Task P2-7 : Component breakdown API

* **Fichiers** : `src/api/routes/signals.py` (endpoint `/signals/{id}/breakdown`)
* **Heures** : 4-6 h
* **Acceptance** : `GET /signals/{id}/breakdown` renvoie JSON avec les 8-9 component_scores + SHAP values (si P2-4 livré) + tier classification + raw inputs.
* **Dépendances** : P0-4.

**Sub-total P2** : **76-104 h** (~10-13 j) = **2-3 semaines** étalable.

---

### Phase intégration & shadow

Après P0 + P1 livrés, **2 semaines de shadow mode** :
* Scoring v1 (Confluence additif) et v2 (Logistic L1) tournent côte-à-côte.
* Logger les deux scores par signal.
* Comparaison Brier, Pearson, win-rate par tier sur dataset live.
* Bascule prod v2 si v2 > v1 stat significatif (DM-test p<0.05) sur 2 semaines.

Effort intégration & shadow : **24-32 h** (3-4 j).

---

## 5. Tests & validation

### 5.1 Tests unitaires (chaque P0 + P1 task a ses tests)

| Task | Test file | Couverture |
|---|---|---|
| P0-1 | `tests/test_sprint3_order_blocks_ict.py` (NOUVEAU) | OB anchor BOS, body filter, mitigation |
| P0-2 | `tests/test_sprint5_fvg_threshold.py` (étendu) | Threshold 0.4, `FVG_FILLED_PCT`, age |
| P0-3 | `tests/test_bos_retest.py` (étendu) | Touch strict, tol 0.25, armed_window default |
| P0-4 | `tests/test_component_scores_persistence.py` (NOUVEAU) | Composant_scores serialisés/désérialisés |
| P0-5 | `tests/test_logistic_l1_scorer.py` (NOUVEAU) | Fit/predict, calibration, walk-forward |
| P0-6 | `tests/test_exit_policy.py` (NOUVEAU) | Exit "opposite" < 15 % |
| P1-1 | `tests/test_sprint7_rsi_divergence.py` (fix indexage) | Comparer swing low, pas confirmation low |
| P1-2 | `tests/test_bos_retest_default_alignment.py` (NOUVEAU) | armed_window 30 dans fast path |
| P1-3 | `tests/test_bos_choch_numba_parity.py` (NOUVEAU) | Parité bit-à-bit |
| P1-4 | `tests/test_main_numba_check.py` (NOUVEAU) | Assertion en prod |
| P1-5 | `tests/test_smc_incremental.py` (NOUVEAU) | Latence p99 < 50ms, bit-identique |
| P1-7 | `tests/test_confluence_warmup_guard.py` (NOUVEAU) | < 50 bars = None |
| P1-8 | tous existants doivent passer | Refactor seul |

**Test coverage target** : `src/intelligence/smart_money/**.py` ≥ **85 % lines** (vs ~70 % actuel sur `strategy_features.py`). Mesure : `pytest --cov=src/intelligence/smart_money`.

### 5.2 Replay walk-forward (P0-7)

* **Split** : 2019-2022 train (4 ans), 2023-2025 OOS test (3 ans).
* **CPCV** : 5 folds purgés avec embargo 50 bars (≈ 12.5h XAU M15) — librairie `mlfinlab` ou implémentation maison `src/research/strategy_gates.py`.
* **Gates institutionnels** (cf. `reports/3_pillars_implementation_2026_05_13.md`) :
  * **DSR ≥ 1.0** (Deflated Sharpe, Bailey-Lopez de Prado 2014).
  * **PBO ≤ 0.30** (Probability Backtest Overfitting, BLdP).
  * **PF lower-bound CI 95 % ≥ 1.10** (bootstrap 10 000 resamples avec block size = 50 bars).
  * **DM-test p < 0.05** (Diebold-Mariano vs naïve baseline).
* **Bootstrap blocks** : block bootstrap (block_size=50, 10 000 iterations) sur returns par trade pour générer IC PF, Sharpe.
* **Validation cross-instrument** : EUR/USD M15 (PF ≥ 1.15) + USOIL (si data dispo).
* **Validation par régime** : PF Bull, Bear, Range tous > 1.0 individuellement (élimine biais saisonnier).
* **Validation par année** : ≥ 5 années sur 7 avec E[R] > 0.
* **Validation par side** : LONG PF ≥ 1.20, SHORT PF ≥ 1.10 (vs SHORT 0.81 actuel — peut justifier long-only conditionnel).

### 5.3 Validation calibration scoring (P0-5)

* **Reliability diagram** : 10 buckets de probabilité, win-rate par bucket. `is_winrate_monotone_up=true` (test eval_02 `reports/eval_02/monotonicity.json`).
* **Brier score** : modèle < baseline (vs 0.2551 > 0.2456 actuel).
* **Brier skill score** : ≥ +5 %.
* **Hosmer-Lemeshow goodness-of-fit** : p > 0.05 (modèle bien calibré).
* **Expected calibration error (ECE)** ≤ 0.05.

### 5.4 Tests régression / non-régression

* Tous les 1366+ tests existants doivent passer (sauf `test_long_short_trading.py` broken import déjà documenté).
* Smoke E2E `tests/test_smoke_e2e.py` : 9 tests passent post-refactor.
* `tests/test_production_wiring.py` : circuit breakers, rate limit, JSON logging intacts.

### 5.5 Edge cases (cf. §6)

* Données OHLCV avec NaN au milieu → refus + log.
* Bar volume = 0 → renormalisation correcte.
* ATR = 0 (marché flat) → FVG/OB strength = 0 sans NaN.
* Lookback < warm-up minimum → return None.
* Timestamps non monotones → raise.

---

## 6. Sécurité (input validation + edge cases)

### 6.1 Validation OHLCV (étendre `sentinel_scanner.py:334-340`)

Le scanner utilise déjà `validate_ohlcv` (`src/intelligence/data_quality.py`). **Étendre** :

| Check | Action si fail | Référence |
|---|---|---|
| Coverage ≥ 95 % sur la fenêtre | log warning, continue | `data_quality.py` |
| No NaN dans OHLCV | log error + return None | `data_quality.py` |
| Timestamps monotones strictement croissants | raise | NOUVEAU check à ajouter |
| `high ≥ low` ET `high ≥ open` ET `high ≥ close` | raise (corrupt bar) | NOUVEAU |
| `spread = high − low ≥ 0` | raise si négatif (data leak) | NOUVEAU |
| Gap timestamps : pas de saut > 4 × période (sauf weekends) | log warning | NOUVEAU |
| Volume ≥ 0 | raise si négatif | NOUVEAU |
| Coverage feed < seuil bloquant | refuser scan (kill switch) | déjà géré `data_quality.py` |

**Effort** : 4-6 h dans cat. 1 dependency, ou en P1-7 si quick-add.

### 6.2 Division par zéro et NaN propagation

| Risque | Localisation | Mitigation |
|---|---|---|
| `volume_ma == 0 OR None` | `confluence_detector.py:282-283` | déjà géré, mais `volume_ma == 0.01` reste comptabilisé → ajouter `if volume_ma < 1e-6` (cf. eval_02 §3.1 `:113`) |
| `atr == 0` (marché flat extrême) | `strategy_features.py:669-679` (FVG_SIZE_NORM) | déjà `np.where(ATR > 0, ..., 0.0)`, ✅ |
| `atr nan` dans retest | `strategy_features.py:293, 370` | déjà géré `if not np.isnan(atr[i]) and atr[i] > 0 else 0.0`, ✅ |
| `volume_ma` négatif (corruption feed) | `confluence_detector.py:282` | déjà `volume_ma <= 0`, ✅ |
| `sum(weights) != 100` après modification | `confluence_detector.py:187-189` | déjà raise, ✅ |
| `present_weight = 0` (toutes données absentes) | `confluence_detector.py:289-290` | déjà `if absent_weight > 0 AND present_weight > 0` ✅ |
| Division dans LightGBM (P0-5) | `models/logistic_l1_xau_m15.pkl` | sklearn gère NaN avec `imputer` à ajouter dans pipeline |
| `body_size = 0` (doji) dans OB filter (P0-1) | `strategy_features.py:756-815` | filtrer `body/ATR ≥ OB_BODY_ATR_MIN > 0` ✅ |

### 6.3 Lookback warm-up

| Composant | Warm-up min | Source |
|---|---:|---|
| RSI(14) | 14 bars | TA-lib |
| MACD(12,26,9) | 26+9 = 35 bars | TA-lib |
| BB(20) | 20 bars | TA-lib |
| ATR(14) | 14 bars | TA-lib |
| Fractals(N=2) | 2 bars left + 2 right + 1 = 5 bars min, mais shift N causal | `strategy_features.py:606-644` |
| BOS warm-up seed | `min(50, n)` bars | `strategy_features.py:77-86, 181-190` |
| Retest state machine | dépend du BOS event, donc ≥ 50 bars effectif | `strategy_features.py:259-407` |
| MTF H4 features (SMA50 + RSI14) | 200 H4 bars ≈ 800 M15 bars | `sentinel_scanner.py:181-186, 427` (déjà appliqué) |

**Cible globale** : exiger `len(df) ≥ 800` (le maxi MTF). Implémenté en Phase 1.1 MTF. Étendre en P1-7 au `ConfluenceDetector` (early return).

### 6.4 Parité Numba ↔ Python (sécurité prod)

* Numba indispo en prod → latence ×25 → SLA scanner 30s impossible.
* P1-3 + P1-4 traitent ce risque (test parité + assertion boot).

### 6.5 Race conditions multi-thread

* Scanner mono-thread (`sentinel_scanner.py:229` `daemon=True` mais single instance) → pas de race actuellement.
* Si futur multi-worker (cat. 6) : `SmartMoneyEngine` instance par worker (pas de shared state mutable).
* `ConfluenceDetector` : `self.weights` mutable mais init au boot, jamais modifié runtime → ok thread-safe en lecture.

### 6.6 Input sanitization (signaux émis)

* `signal_id` : déjà hash SHA1 déterministe (`confluence_detector.py:353-364`), pas de UUID4 (audit institutional fix).
* `bar_timestamp` : déjà sanitisé via `_bts = bar_timestamp.isoformat()` ou `str(bar_timestamp)`.
* Pas d'injection possible côté production de signal (sortie pure).

---

## 7. Métriques de succès

### 7.1 KPIs algo (mesurés sur replay walk-forward post P0)

| KPI | Baseline (now) | Target P0 | Source / méthode |
|---|---:|---:|---|
| Profit Factor 7 ans XAU M15 | 1.037 (v2 timeout=64) | **≥ 1.30** OOS | `reports/audit_2026_04_30_v2_timeout64.md:13` → cible cat. 7 backtest |
| PF lower-bound CI 95 % | inconnu (estimé ~0.92) | **≥ 1.10** | Bootstrap block (10k resamples, block=50) |
| Sharpe ratio (mensuel ann.) | +0.274 (v2) | **≥ 0.80** | `reports/audit_2026_04_30_v2_timeout64.md:16` |
| Sortino | +0.454 (v2) | ≥ 1.0 | idem |
| Max Drawdown | -48 % (v2) | **≤ -35 %** | idem |
| DSR (Deflated Sharpe) | 0.65 (3-pillars naïve) | **≥ 1.0** | `reports/3_pillars_implementation_2026_05_13.md`, `src/research/strategy_gates.py` |
| PBO (Backtest Overfitting) | 0.50 (3-pillars) | **≤ 0.30** | idem |
| DM-test p-value vs baseline | 0.52 (3-pillars) | **< 0.05** | idem |
| Brier score scoring | 0.2551 (additif) | **≤ 0.215** | `reports/eval_02_confluence.md:166` |
| Brier skill score | −2.2 % | **≥ +5 %** | idem |
| Pearson(score, R) OOS | −0.023 | **≥ +0.10** | idem |
| Spearman OOS | −0.016 | **≥ +0.12** | idem |
| Win-rate PREMIUM tier | 0 % (0 signaux) | **≥ 52 %**, ≥ 30 signaux/an | P0-8 calibration |
| Win-rate STANDARD tier | ~44 % | **≥ 48 %** | idem |
| WR(PREMIUM) − WR(STANDARD) | NaN (PREMIUM vide) | **≥ +8 pp** | idem |
| Exit "opposite" % | 35.6 % (v2) | **< 15 %** | P0-6 |
| Win-rate armed LONG | 33.5 % | **≥ 38 %** | `reports/eval_03/eval_03_smart_money.md:76` cible P0-3 |
| Win-rate armed SHORT | 30.3 % | **≥ 35 %** | idem |
| ≥ 5 années / 7 avec E[R] > 0 | 4 (v2 : 2020, 2024, 2025, 2026) | **≥ 5** | `reports/audit_2026_04_30_v2_timeout64.md:32-44` |
| Cross-instrument PF (EUR/USD) | inconnu | **≥ 1.15** | P1-6 |

### 7.2 KPIs détecteurs SMC (post-P0-1/2/3)

| KPI | Baseline | Target |
|---|---:|---:|
| OB count / 6 ans XAU | 34 722 | **< 6 000** (réduction 80 %) |
| OB anchored BOS ≤ 20 bars | 59 % | **≥ 95 %** (ICT-strict) |
| FVG count / 6 ans XAU | 22 548 | **< 5 000** (réduction 78 %) |
| Retest success rate | 89.8 % | **< 60 %** (touch strict) |
| % bars ≥ tier PREMIUM | 0 % (pre-recal) / ~1 % (post-recal) | **0.5-3 %** (P0-8) |
| % bars ≥ tier STANDARD | ~5 % | **5-15 %** |

### 7.3 KPIs perf (latence)

| KPI | Baseline | Target |
|---|---:|---:|
| Pipeline complet 172k bars (Python fallback) | 12.1 s | **< 1 s avec Numba** |
| Pipeline incrémental 1 bar | recompute (~12s) | **< 50 ms p99** (P1-5) |
| Confluence detector `analyze()` | < 1 ms | **< 1 ms p99** (already ok) |
| Logistic L1 predict | NA | **< 5 ms p99** (P0-5) |
| Scan complet 1 symbol | ~12 s (sans Numba) | **< 2 s p99** (avec Numba + incremental) |

### 7.4 KPIs business (commerciaux directs)

| KPI | Baseline | Target Go-Live |
|---|---:|---:|
| Signaux PREMIUM / an / instrument | 0 | 30-100 |
| Signaux STANDARD / an / instrument | ~50 | 200-500 |
| % faux positifs claimed PREMIUM (WR < 50 %) | 100 % | **0 %** |
| Marketing claim « PREMIUM = haute conviction » substanciable | **NO** | **YES** (WR_PREMIUM − WR_STANDARD ≥ +8pp avec bootstrap CI) |
| Tear sheet exportable (PDF + JSON) | partial (MD only) | **YES** (Sprint 7 pandoc) |
| Track-record auditable bootstrap | inconnu | publié sur landing page |
| Differenciation vs smc-python | aucune (notre OB est pire) | **OB ICT-strict + chart renderer + scoring ML** |

---

## 8. Risques & mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|:---:|:---:|---|
| **R1** : Post-fixes P0-1/2/3, le nombre de signaux/an chute si bas qu'il n'y a plus de stat power pour entrainer Logistic L1 | Moyenne | **Élevé** | (a) Relâcher temporairement seuils tier pour collecter data, (b) ajouter EUR/USD + USOIL (P1-6) pour augmenter trades, (c) si N < 500 sur 7 ans, fallback rule-based with manual weights cv |
| **R2** : Logistic L1 / LightGBM échoue les gates DSR / PBO / DM-test (similaire au verdict A1) | Moyenne-Élevée | **Élevé** | (a) Cf. `reports/a1_verdict_2026.md` — pivot narrative-first si pas d'edge, (b) Multi-asset pooling (XAU + EUR + USOIL training set), (c) Feature engineering profond (cross-asset correlation, time-of-day fixed effects) |
| **R3** : Refonte OB casse 28 fichiers consommateurs | Moyenne | Moyen | Facade `src/intelligence/smart_money/` préservée (audit déjà acté). Tests régression intégrale avant merge. |
| **R4** : Numba absent en prod Railway | Faible | **Élevé** | P1-4 assertion boot + Sentry alert. Pip install numba dans Dockerfile vérifié. |
| **R5** : Bug "opposite exit" subtil après fix P0-6 | Moyenne | Moyen | A/B replay v2 vs v3 sur 7 ans, comparer distribution exit reasons |
| **R6** : MTF Phase 3 lift weight casse calibration tiers | Faible-Moyenne | Moyen | P2-6 inclut re-sweep tiers (P0-8 ré-exécuté), shadow mode 2 sem |
| **R7** : Dépendance data layer cat. 1 (XAU + EUR coverage ≥ 95 %) | Élevée (déjà identifié `eval_08`) | **Élevé** | Lock data layer en P0-0 avec cat. 1 owner. Sans XAU ≥ 95 % et EUR ≥ 95 %, P0-5 et P0-7 non testables. |
| **R8** : Shadow mode prouve v2 < v1 (overfitting) | Faible-Moyenne | Élevé | Walk-forward CPCV strict + OOS test pré-shadow. Si shadow fail, revenir v1 + investigate. |
| **R9** : Chart renderer ralentit la livraison Telegram | Faible | Faible | Génération async + cache (signal_id → PNG bytes), 24h TTL. Skip si latence > 500ms. |
| **R10** : SHAP values pas explicables au grand public | Moyenne | Faible | UX layer : afficher top-3 « raisons » en langage naturel (LLM narrative consomme SHAP, pas l'utilisateur). |
| **R11** : Bug "BOS sur 100 % des bars" reapparait avec data feed corrompu | Faible | **Élevé** | Test `test_data_quality_bos_regression.py` actif. Data quality gate `sentinel_scanner.py:334-340` strict. Coverage ≥ 95 % en hard fail. |
| **R12** : RSI Divergence fix change distribution → re-calibre tiers à 0 | Faible | Moyen | P1-1 fix puis re-run P0-8 calibration. |
| **R13** : Performance Logistic L1 dégrade sur out-of-sample non-stationnaire (ex: 2025 régime change) | Moyenne | Moyen | `calibration_loop.py` (P0-5) refit toutes les K=200 trades + drift detection (Bocpd `src/intelligence/bocpd.py`) → alerte si distribution shift |
| **R14** : Compliance / claims marketing (memory `eval_29_compliance`) | Élevée | Élevé | Reformuler tier names en internal use ; le marketing doit s'aligner (cf. cat. 9 légal). Pas un risque algo direct mais conditionne le claim. |

---

## 9. Dépendances autres catégories

| Catégorie | Dépendance | Sévérité | Détails |
|---|---|:---:|---|
| **Cat. 1 — Data Layer** | XAU coverage ≥ 95 % sur 2019-2026 | **Bloquant P0-5, P0-7** | `XAU_15MIN_2019_2026.csv` à valider (audit batch 0.0). Sans data propre, training set pourri. Cf. `audits/2026-Q2/sprint_0_decisions.md` Décision A. |
| Cat. 1 | EUR/USD M15 coverage ≥ 95 % | Bloquant P1-6 | `EURUSD_15MIN_2019_2025.csv` à 99.41 % ✅. |
| Cat. 1 | USOIL M15 coverage si dispo | Souhaitable P1-6 | Optionnel — pivot si bloquant. |
| Cat. 1 | News feed (ForexFactory + crosscheck MT5) | Bloquant si news component activé | Cf. `memory:news_pipeline_2026_04_24`. |
| **Cat. 3 — Volatility** | `VolatilityForecast` (HAR-RV / LGBM / Hybrid) | Non bloquant (déjà wired) | `src/intelligence/volatility_forecaster.py`. `VOL_MODE=har` par défaut (eval_04). Le scoring ML pourra ingérer `vol_forecast_atr` comme feature. |
| **Cat. 4 — News** | `NewsAssessment.position_multiplier` et `sentiment_score` | Non bloquant (déjà wired) | `src/agents/news_analysis_agent.py`. Cf. `confluence_detector.py:537-557`. |
| **Cat. 5 — LLM** | Narrative consume `component_scores[]` + SHAP | Bloquant P2-4 | `src/intelligence/llm_narrative_engine.py` — prompt enrichi avec top-3 SHAP. |
| **Cat. 6 — API/Delivery** | `/signals/{id}/breakdown` endpoint | Bloquant P2-7 | `src/api/routes/signals.py` (existe partiel). |
| Cat. 6 | Chart attachment Telegram | Bloquant P2-1 | `src/delivery/telegram_notifier.py` — méthode `send_signal` doit accepter PNG bytes. |
| **Cat. 7 — Backtest / MLOps** | CPCV harness, bootstrap CI, DSR/PBO | **Bloquant P0-5, P0-7** | `src/research/strategy_gates.py` (existe `reports/3_pillars_implementation_2026_05_13.md`). |
| Cat. 7 | Walk-forward refit schedule | Bloquant P0-5 | `src/intelligence/scoring/calibration_loop.py` (à créer). |
| Cat. 7 | Model registry (versioned `models/*.pkl`) | Bloquant P0-5 deployment | À mettre en place côté MLOps. |
| **Cat. 8 — Compliance** | Reformulation tier names | Non bloquant algo, bloquant marketing | Cf. memory `sprint_w1_compliance_2026_04_29`. Reformuler "PREMIUM signal" → "PREMIUM analysis tier" dans copy. |
| **Cat. 9 — Risk Engine** | `risk_manager.py` (frozen pendant Sprint 0-4) | Non bloquant | Cf. Décision B `audits/2026-Q2/sprint_0_decisions.md:28-40`. |
| **Travail parallèle MTF** | Phase 1.6 (intégration tests) | Bloquant pour P1-1 (lookout warmup vérifié) | Référence `src/intelligence/sentinel_scanner.py:181-186`. |
| Travail parallèle MTF | Phase 2 (empirical replay) | Bloquant P2-6 | Décide si htf_alignment lift. |
| Travail parallèle MTF | Phase 3 (lift + re-sweep tiers) | Bloquant final calibration | P0-8 dépend si htf_alignment final weight ≠ 0. |

**Interface contrats** :
* Cat. 2 expose `ConfluenceSignal.to_dict()` (`confluence_detector.py:84-110`) — stabilité contractuelle Pydantic v2.
* Cat. 2 consomme `VolatilityForecast` (cat. 3), `NewsAssessment` (cat. 4), `RegimeAnalysis` (cat. 9), `MTFFeatures` (parallèle).
* Cat. 2 produit `InsightSignalV2` via `InsightAssembler` (`src/intelligence/insight_assembler.py`) — contrat v2.1.0.

---

## 10. Estimation totale & timeline

### 10.1 Budget effort (heures, dev senior solo)

| Bloc | Heures | Jours (8h/j) |
|---|---:|---:|
| P0-1 OB ICT-strict | 16-24 | 2-3 |
| P0-2 FVG threshold | 6-10 | 1 |
| P0-3 Retest strict | 8-12 | 1.5 |
| P0-4 Component scores persistence | 8-12 | 1.5 |
| P0-5 Logistic L1 + calibration loop (+ option LightGBM +24h) | 40-56 | 5-7 |
| P0-6 Exit policy fix | 12-16 | 2 |
| P0-7 Replay 7 ans + gates institutionnels | 16-24 | 3 |
| P0-8 Tier cutpoints validation | 6-8 | 1 |
| **Sub-total P0** | **112-162** | **14-20 j** |
| P1-1 RSI Divergence fix | 2-4 | 0.5 |
| P1-2 armed_window default | 1-2 | 0.25 |
| P1-3 Numba parity BOS/CHOCH | 3-4 | 0.5 |
| P1-4 Assertion Numba prod | 2-3 | 0.5 |
| P1-5 Pipeline incrémental | 24-32 | 3-4 |
| P1-6 Cross-instrument | 16-24 | 2-3 |
| P1-7 Lookback warmup guard | 2-3 | 0.5 |
| P1-8 Module physique smart_money/ | 24-32 | 3-4 |
| **Sub-total P1** | **74-104** | **10-13 j** |
| P2-1 Chart renderer | 24-32 | 3-4 |
| P2-2 Liquidity sweep | 16-20 | 2-3 |
| P2-3 Premium/discount | 8-12 | 1-1.5 |
| P2-4 SHAP values | 12-16 | 1.5-2 |
| P2-5 OB mitigation viz | 8-12 | 1-1.5 |
| P2-6 MTF Phase 3 lift | 4-6 | 0.5-1 |
| P2-7 Breakdown API | 4-6 | 0.5-1 |
| **Sub-total P2** | **76-104** | **10-13 j** |
| Intégration & shadow mode | 24-32 | 3-4 |
| Buffer review / tests / docs | 24-32 | 3-4 |
| **GRAND TOTAL** | **310-434 h** | **40-54 j** |

À 30 h/sem (dev solo realistic, hors emails / réunions / autre cat.) : **10-14 semaines** soit **~2.5-3.5 mois**.

À 40 h/sem (focus dédié) : **8-11 semaines** soit **~2-3 mois**.

### 10.2 Timeline Gantt (calendrier indicatif, 30 h/sem)

```
Sem 1-2   : P0-0 (data lock, cat. 1) + P0-4 (component_scores) + P1-1 + P1-2 + P1-3 + P1-4
Sem 3     : P0-2 (FVG) + P0-3 (Retest strict)
Sem 4-5   : P0-1 (OB ICT-strict) + P1-7 (warmup) + P1-8 partial (smart_money/ extract)
Sem 6     : P0-6 (exit policy) + P1-5 (incremental) start
Sem 7-9   : P0-5 (Logistic L1 + calibration + isotonic + walk-forward)
Sem 10    : P0-7 (replay 7 ans gates) + P0-8 (tier calibration)
Sem 11    : P1-6 (cross-instrument) + buffer
Sem 12-13 : Shadow mode (2 sem)
Sem 14    : Go-Live decision + intégration + commercial readiness
[Post Go-Live, étalable]
Sem 15-17 : P2-1 (chart renderer) + P2-4 (SHAP) + P2-7 (breakdown API)
Sem 18-20 : P2-2 (liquidity sweep) + P2-3 (premium/discount) + P2-5 (mitigation viz)
Sem 21    : P2-6 (MTF Phase 3 lift après Phase 2 empirical)
```

### 10.3 Jalons décisifs

| Jalon | Sem | Critère |
|---|:---:|---|
| **J1 — Data lock** | 1 | XAU + EUR coverage ≥ 95 %, news feed CSV pipeline OK (cat. 1) |
| **J2 — SMC ICT-clean** | 5 | OB count < 6k, FVG < 5k, retest WR > 38 % |
| **J3 — Scoring v2 calibré** | 9 | Brier skill ≥ +5 %, Pearson OOS ≥ +0.10 |
| **J4 — Gates institutionnels passés** | 10 | DSR ≥ 1.0, PBO ≤ 0.30, DM-test p < 0.05, PF_lo CI ≥ 1.10 |
| **J5 — Tier substanciable** | 10 | WR_PREMIUM − WR_STANDARD ≥ +8 pp |
| **J6 — Cross-instrument validé** | 11 | EUR PF ≥ 1.15 OOS |
| **J7 — Shadow mode passed** | 13 | v2 > v1 stat significatif (DM-test) sur 2 sem |
| **J8 — Go-Live** | 14 | Tous gates passés + chart renderer + SHAP + breakdown API |

### 10.4 Critères d'arrêt (kill switches plan)

* **Kill A** : Après P0-5, si Logistic L1 ne passe pas Brier skill ≥ 0 → fallback LightGBM. Si LightGBM échoue aussi → pivot **narrative-first** (cf. `reports/a1_verdict_2026.md` decision tree).
* **Kill B** : Après P0-7, si gates institutionnels échouent même avec scoring v2 → **PAS de commercialisation cat. 2 standalone**. Options :
   * (a) Pivot B2B-API broker (cf. `reports/decision_matrix_2026_04_30.md` : « pivot B2B-API brokers ($310k ARR cible, 80h dev MVP) »).
   * (b) Pivot narrative-first (cat. 5 dominante, cat. 2 reste support).
   * (c) Réduction de scope à XAU only + long-only (preuve β-capture acceptée).
* **Kill C** : Si EUR/USD PF < 1.05 après P1-6 → claim « multi-asset » abandonné, focus XAU-only.

---

## 11. Annexes — Sources et chemins

### 11.1 Eval reports cités (chemins absolus)

* `C:\MyPythonProjects\TradingBOT_Agentic\reports\eval_02_confluence.md`
* `C:\MyPythonProjects\TradingBOT_Agentic\reports\eval_03\eval_03_smart_money.md`
* `C:\MyPythonProjects\TradingBOT_Agentic\reports\audit_2026_04_30_v2_timeout64.md`
* `C:\MyPythonProjects\TradingBOT_Agentic\reports\feature_edge_audit.md`
* `C:\MyPythonProjects\TradingBOT_Agentic\reports\feature_filter_audit.md`
* `C:\MyPythonProjects\TradingBOT_Agentic\reports\decision_matrix_2026_04_30.md`
* `C:\MyPythonProjects\TradingBOT_Agentic\reports\3_pillars_implementation_2026_05_13.md`
* `C:\MyPythonProjects\TradingBOT_Agentic\reports\a1_verdict_2026.md`

### 11.2 Audits Sprint 0

* `C:\MyPythonProjects\TradingBOT_Agentic\audits\2026-Q2\section_3_2_smart_money.md` (note 6.0/10)
* `C:\MyPythonProjects\TradingBOT_Agentic\audits\2026-Q2\section_3_3_confluence.md` (note 3.0/10)
* `C:\MyPythonProjects\TradingBOT_Agentic\audits\2026-Q2\section_3_2_smart_money_stats.json`
* `C:\MyPythonProjects\TradingBOT_Agentic\audits\2026-Q2\sprint_0_decisions.md`

### 11.3 Code source clé

* `C:\MyPythonProjects\TradingBOT_Agentic\src\environment\strategy_features.py` (1213 LOC, `SmartMoneyEngine`)
* `C:\MyPythonProjects\TradingBOT_Agentic\src\intelligence\confluence_detector.py` (727 LOC)
* `C:\MyPythonProjects\TradingBOT_Agentic\src\intelligence\sentinel_scanner.py` (orchestrateur)
* `C:\MyPythonProjects\TradingBOT_Agentic\src\intelligence\smart_money\__init__.py` (facade)
* `C:\MyPythonProjects\TradingBOT_Agentic\src\intelligence\scoring\logistic_l1.py` (scaffold P0-5)
* `C:\MyPythonProjects\TradingBOT_Agentic\src\intelligence\signal_state_machine.py`
* `C:\MyPythonProjects\TradingBOT_Agentic\src\intelligence\regime_filter.py` (cat. 9 wire)
* `C:\MyPythonProjects\TradingBOT_Agentic\src\intelligence\insight_assembler.py` (output contract v2.1.0)
* `C:\MyPythonProjects\TradingBOT_Agentic\src\research\strategy_gates.py` (cat. 7 dépendance — gates institutionnels)

### 11.4 Tests existants à protéger

* `tests/test_confluence_detector.py` (579 LOC)
* `tests/test_bos_no_repeated_fire.py`
* `tests/test_bos_retest.py`
* `tests/test_data_quality_bos_regression.py`
* `tests/test_sprint2_choch_reset.py`
* `tests/test_sprint3_order_blocks.py`
* `tests/test_sprint5_fvg_threshold.py`
* `tests/test_sprint7_rsi_divergence.py`
* `tests/test_state_machine_replay.py`
* `tests/test_smoke_e2e.py`

### 11.5 Mémoire projet (contexte stratégique)

* `C:\Users\bessa\.claude\projects\C--MyPythonProjects-TradingBOT-Agentic\memory\MEMORY.md`
* `C:\Users\bessa\.claude\projects\C--MyPythonProjects-TradingBOT-Agentic\memory\three_pillars_implementation_2026_05_13.md`
* `C:\Users\bessa\.claude\projects\C--MyPythonProjects-TradingBOT-Agentic\memory\institutional_quant_plan_2026_05_13.md`
* `C:\Users\bessa\.claude\projects\C--MyPythonProjects-TradingBOT-Agentic\memory\a1_verdict_2026_05_01.md`
* `C:\Users\bessa\.claude\projects\C--MyPythonProjects-TradingBOT-Agentic\memory\decision_matrix_2026_04_30.md`
* `C:\Users\bessa\.claude\projects\C--MyPythonProjects-TradingBOT-Agentic\memory\feedback_multi_view_ux.md` (référence MTF Phase 1)

---

## Synthèse exécutive — 5 lignes

* **Fichier livré** : `C:\MyPythonProjects\TradingBOT_Agentic\reports\commercialization_sprint\02_smart_money_algo.md`
* **Top 3 P0** : (1) **P0-5 Refonte scoring Logistic L1 + calibration walk-forward CPCV** (~40-56 h, target Brier skill ≥ +5 %, Pearson OOS ≥ +0.10) ; (2) **P0-1 Order Blocks ICT-strict** avec anchor BOS + body/ATR filter + mitigation (~16-24 h, réduction OB count 34k → < 6k) ; (3) **P0-7 Replay 7 ans avec gates institutionnels** (DSR ≥ 1.0, PBO ≤ 0.30, PF_lo CI ≥ 1.10, ~16-24 h) — kill-switch si fail.
* **Effort total P0** : 112-162 h soit ~3-4 sem | **P0+P1** : 186-266 h soit ~6-8 sem | **P0+P1+P2 + shadow** : 310-434 h soit ~10-14 sem (2.5-3.5 mois).
* **Go-Live cible** : Semaine 14 (P0+P1+shadow mode passés) ; P2 étalé post Go-Live.
* **Dépendances bloquantes** : Cat. 1 (data XAU+EUR ≥ 95 % coverage) et Cat. 7 (CPCV harness `strategy_gates.py`). MTF Phase 2 doit avoir conclu avant P2-6.
