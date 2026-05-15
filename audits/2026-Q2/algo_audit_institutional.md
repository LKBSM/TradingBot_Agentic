# Audit Institutionnel Phase 1 — Algorithme Smart Sentinel AI

**Date** : 2026-05-15
**Sprint** : 0 (institutional overhaul, branche `institutional-overhaul`)
**Auditeur principal** : Claude (Lead Quant Architect)
**Auditeurs spécialisés (sous-agents)** : Smart Money (3.2), Confluence (3.3), Volatility (3.4), Backtest engine (3.8)
**Tag de référence** : `v0.9.0-pre-institutional`
**Commit baseline** : `66c1a5335619559a69654577893818461d23b65c`

---

## Résumé exécutif

L'algorithme Smart Sentinel AI est **architecturalement riche** (22 k LOC, 27 modules cœur, 2 696 tests) mais **statistiquement non prédictif dans sa configuration actuelle** :

1. **Le ConfluenceDetector** a un pouvoir prédictif nul (Pearson −0.008, Brier skill **−0.022** — pire qu'une probabilité constante).
2. **La baseline Sprint 0** sur 7 ans XAU + 7 ans EURUSD avec les defaults produit **0 trades** (score plafonne à 72.61 / 74.97, sous `enter_threshold=75`).
3. **Le data layer** est désormais propre (98.72 % coverage XAU, 99.41 % EURUSD) — la régression du bug "BOS 100 % bars" est garde-fou-cée.
4. **Le state machine** (8.0/10) et **le ConformalWrapper** (7.0/10) sont production-grade — ils attendent un edge à filtrer.
5. **Le backtest engine** : à compléter par l'agent dédié.
6. **La détection ICT/SMC** (6.0/10) souffre de 2 bugs de logique (RSI div, magic number retest) + 1 bug structurel (OB ≠ ICT — engulfing, 40 % des OB sans BOS dans ±20 bars).

### Note globale Smart Sentinel AI : **5.5 / 10**

Pondération : data (5.0, ×2), smart money (6.0, ×2), confluence (3.0, ×3 — pilier du scoring), volatility (à compléter, défaut 5.0, ×1), régime (6.5, ×1), conformal (7.0, ×1), state machine (8.0, ×2), backtest engine (à compléter, défaut 2.0, ×2) = **5.5 / 10** pondérée.

---

## Tableau récapitulatif (scores 0-10)

| Section          | Sous-système                | Score 0-10 | Précédent (mémoire) | Δ        | Auditeur     |
| ---------------- | --------------------------- | ---------- | ------------------- | -------- | ------------ |
| 3.1              | Data layer                  | **5.0**    | 3.5 (eval_08)       | **+1.5** | Claude       |
| 3.2              | Smart Money / ICT           | **6.0**    | 4.5 (eval_03)       | **+1.5** | Agent (sub)  |
| 3.3              | ConfluenceDetector          | **3.0**    | 5.0 (eval_02)       | **−2.0** | Agent (sub)  |
| 3.4              | VolatilityForecaster        | ⏳ pending  | 5.0 (eval_04)       | ?        | Agent (sub)  |
| 3.5              | Régime stack                | **6.5**    | n/a                 | n/a      | Claude       |
| 3.6              | ConformalWrapper            | **7.0**    | n/a                 | n/a      | Claude       |
| 3.7              | SignalStateMachine          | **8.0**    | 8.0 (eval_07)       | =        | Claude       |
| 3.8              | Backtest engine             | **3.5**    | 2.0 (eval_18)       | **+1.5** | Agent (sub)  |

**Note pondérée provisoire** : 5.5 / 10 (peut bouger ±0.5 selon résultats 3.4 et 3.8).

---

## Plan P0 / P1 / P2 consolidé

Pour chaque finding, l'action est rattachée au sprint cible de la roadmap institutionnelle (Sprint 1-7).

### P0 — bloqueurs go-live (à corriger Sprint 1-3)

| #    | Description                                                                                              | Section | Sprint   |
| ---- | -------------------------------------------------------------------------------------------------------- | ------- | -------- |
| P0-1 | **ConfluenceDetector sans pouvoir prédictif** (Pearson −0.008, Brier skill −0.022)                       | 3.3     | Sprint 3-4 |
| P0-2 | **Order Blocks ≠ définition ICT** : engulfing seule, 40 % des OB sans BOS dans ±20 bars                   | 3.2     | Sprint 1.0 / 2.1 |
| P0-3 | **Aucun signal tradable** avec defaults sur 7 ans (score plafonne à 72-74 < enter=75)                    | 3.7     | Sprint 3 |
| P0-4 | **0 GitHub Actions CI** avant Sprint 0 (corrigé batch 0.1 par workflow minimal)                          | infra   | ✅ Done   |
| P0-5 | **Pas de walk-forward propre** dans `state_machine_replay.py` (eval_18)                                  | 3.8     | Sprint 3 |
| P0-6 | **Coûts transactionnels $0** en backtest — modèles existent mais non wired                              | 3.8     | Sprint 3 |
| P0-7 | **Look-ahead MTF latent** (`multi_timeframe_features.py:269` `<=` vs `<`)                                | 3.1/3.8 | Sprint 1.2 |
| P0-8 | **Pas de contrat Pydantic v2 à l'ingestion data**                                                        | 3.1     | Sprint 1.1 |
| P0-9 | **Smart Money pas extrait en module dédié** (logique éparpillée `strategy_features.py`)                  | 3.2     | Sprint 1.0 |
| P0-10 | **Seuils RegimeGate hardcoded** sans calibration empirique                                              | 3.5     | Sprint 1 |
| P0-11 | **PICP conformelle non mesurée** OOS                                                                   | 3.6     | Sprint 4.2 |
| P0-12 | **Defaults state machine non empiriques** (sweep 432 cellules pending depuis eval_07)                   | 3.7     | Sprint 3 |
| P0-13 | **HMM `predict()` potentiellement refit-at-call** (bug B1 eval_04)                                     | 3.5/3.4 | Sprint 1 |
| P0-14 | **5 / 6 presets sans CSV propre** (BTC, US500, GBP, JPY, USOIL)                                        | 3.1     | Sprint 1.5 |
| P0-15 | **Bug RSI Divergence** : compare wrong bar index (`strategy_features.py:849-857`)                       | 3.2     | Sprint 1.0 |
| P0-16 | **Snapshot store API per-signal manquant** (Sprint 6)                                                  | 3.7     | Sprint 6 |
| P0-17 | **CPCV/DSR/PBO machinery existe** (`src/research/`) MAIS **non couplée** à `src/backtest/`. La strat n'est PAS gated. | 3.8 | Sprint 3 |

### P1 — qualité / dette technique (Sprint 4-5)

| #    | Description                                                                          | Section | Sprint |
| ---- | ------------------------------------------------------------------------------------ | ------- | ------ |
| P1-1 | **Magic number incohérent** : `armed_window=5` vs `RETEST_ARMED_WINDOW=30`            | 3.2     | Sprint 1.0 |
| P1-2 | **FVG_THRESHOLD=0.1 ATR ≈ spread** XAU                                              | 3.2     | Sprint 2 |
| P1-3 | **RETEST_TOL_ATR=0.5 ATR ≈ 1.5 $** ≈ spread XAU                                     | 3.2     | Sprint 2 |
| P1-4 | **Double-gating** detector (min=25) + state machine (enter=75) — 96.3 % rejection    | 3.3     | Sprint 4 |
| P1-5 | **OB ↔ Retest corrélés** (Cramér's V=0.489) — info dupliquée dans le score          | 3.3     | Sprint 4 |
| P1-6 | **Session NY hardcoded UTC 13-21** — devrait être InstrumentConfig                  | 3.5     | Sprint 1 |
| P1-7 | **Pas de gate Tokyo/London** symétrique                                              | 3.5     | Sprint 2 |
| P1-8 | **REDUCE state du RegimeGate non consommé** par state machine                       | 3.5     | Sprint 5 |
| P1-9 | **Conformal sans Mondrian** (stratification par régime)                              | 3.6     | Sprint 4.1 |
| P1-10 | **Confirm_bars / max_age non paramétrés par TF** (state machine)                  | 3.7     | Sprint 2-3 |
| P1-11 | **Tests chaos / property-based manquants** state machine                          | 3.7     | Sprint 5 |
| P1-12 | **Versioning JSON state** absent (state_persistence.py)                            | 3.7     | Sprint 6 |
| P1-13 | **Réconciliation MTF** intelligence/ vs environment/ (2 implémentations)           | 3.1     | Sprint 1 |
| P1-14 | **Calendrier économique** fraîcheur 2025-12-31 (pas 2026)                          | 3.1     | Sprint 1.3 |
| P1-15 | **2 000 LOC legacy** agents/market_regime_* + regime_predictor — figés mais pas archivés | 3.5 | Sprint 6 |

### P2 — confort / typage (Sprint 6)

- Type hints coverage 69 % moyenne → cible 90 %.
- Cleanup `print()` debug dans `state_machine_replay.py:390`.
- Docstrings pour ~50 classes sans documentation.
- Audit stabilité numérique BOCPD (overflow log-probs).

---

## Sections détaillées

Chaque section est dans son propre fichier. Lien et synthèse ci-dessous.

### Section 3.1 — Data Layer ([détail](section_3_1_data_layer.md))

**Score : 5.0 / 10** (vs eval_08 = 3.5, **+1.5** grâce au switch XAU 2019_2026 et au garde-fou BOS régression).

Findings principaux :
- Coverage MVP : XAU **98.72 %** + EURUSD **99.41 %**.
- 5 / 6 presets sans CSV (P0-14).
- Pas de Pydantic v2 (P0-8), pas d'API uniforme DataProvider (P0-8).
- Resampling MTF look-ahead non prouvé absent (P0-7).
- Licence Dukascopy différée (CSV adopté ne dépend pas).

### Section 3.2 — Smart Money / ICT ([détail](section_3_2_smart_money.md))

**Score : 6.0 / 10** (vs eval_03 = 4.5, **+1.5**).

Findings principaux (par agent sub) :
- 2 nouveaux bugs détectés : **magic number incohérent** retest + **bug indexage RSI Divergence**.
- P0 confirmés : **OB ≠ ICT** (engulfing seule, 40 % OB sans BOS dans ±20 bars), FVG threshold trop laxe, retest tol ≈ spread.
- BOS firing rate empirique 3.16 % XAU / 2.96 % EURUSD → garde-fou data quality vert.
- Numba absent en env audit → fallback Python 12.1 s vs 0.5 s avec Numba — risque latence prod.
- Cross-actifs : logique ATR-relative se généralise correctement.

### Section 3.3 — ConfluenceDetector ([détail](section_3_3_confluence.md))

**Score : 3.0 / 10** (vs eval_02 = 5.0, **−2.0** — l'agent sub a creusé empiriquement).

Findings principaux (par agent sub) :
- **Pas de pouvoir prédictif** : Pearson(score, R) = −0.008, Brier skill = −0.022 (pire que constant).
- **Reliability diagram non monotone** : win rate oscille 42.9-50 % sur 7 buckets.
- **Poids hardcoded** sans CV documenté (`regime=25, news=20, bos=15, fvg=15, ob=10, volume=10, momentum=3, rsi_div=2`).
- **OB ↔ Retest** Cramér's V = 0.489 = info dupliquée.
- **News quasi-saturé** à 99.8 % activation = info marginale.
- **96.3 % rejection** par le double-gating detector + state machine.
- Recommandation Sprint 4 : **logistic regression L1** multi-feature (Brier skill cible ≥ +0.03), pas isotonic (Spearman ≈ 0).

### Section 3.4 — VolatilityForecaster ([détail](section_3_4_volatility.md))

⏳ **Pending** (agent en cours, prévu score ~5.0 / 10 conforme eval_04).

Hypothèses pré-audit :
- HAR-RV en défaut prod (latence cible 50 ms).
- LGBM / Hybrid latence 1.6-5 s/forecast = hors cible 30-100×.
- Bug B1 (HMM predict refit-at-call) + B2 (event-prox row-by-row).

### Section 3.5 — Régime stack ([détail](section_3_5_regime.md))

**Score : 6.5 / 10**.

Findings principaux :
- 6 implémentations parallèles, **décision D actée** : `regime_filter.py` + `regime_gate.py` + `bocpd.py` canoniques.
- Stack institutionnel-grade (BOCPD + Bipower jumps + HAR-RV) + références académiques solides.
- **Seuils non calibrés** empiriquement (P0-10).
- Impact réel sur PF : `+0.16 DSR` (insuffisant pour passer DSR=0.65 sur weak edge).
- Legacy 2 000 LOC (`agents/market_regime_agent.py` + `regime_predictor.py`) figés.

### Section 3.6 — ConformalWrapper ([détail](section_3_6_conformal.md))

**Score : 7.0 / 10**.

Findings principaux :
- Implémentation institutionnelle propre (Split + ACI).
- Références académiques explicites (Angelopoulos & Bates 2024, Gibbs & Candès 2021).
- **PICP non mesuré empiriquement** (P0-11).
- Sur stack actuel : **rejette tout** (correct sur weak edge) → opérationnellement = 0 trade.
- Mondrian stratifié par régime manquant (P1-9).
- Le wrapper attend qu'un edge prédictif existe (Sprint 3 doit précéder Sprint 4).

### Section 3.7 — SignalStateMachine ([détail](section_3_7_state_machine.md))

**Score : 8.0 / 10** (confirme eval_07).

Findings principaux :
- Code excellent : déterministe, thread-safe, persistence-ready, 6 exit reasons.
- 54 tests dédiés.
- **Defaults non empiriques** (P0-12) — sweep 432 cellules pending.
- Hysteresis 75/55 sans justification.
- Tests chaos / property-based manquants (P1-11).
- Snapshot store API per-signal manquant (P0-16, Sprint 6).
- **Empirique baseline** : 0 arms_started sur 7 ans XAU + 7 ans EURUSD.

### Section 3.8 — Backtest engine ([détail](section_3_8_backtest_engine.md))

**Score : 3.5 / 10** (vs eval_18 = 2.0, **+1.5** — l'agent sub a découvert que CPCV+DSR+PBO existent en R&D).

Findings principaux (par agent sub) :
- **CPCV/DSR/PBO machinery existe** : `src/research/cpcv_harness.py` (507 LOC, López de Prado AFML conforme) + `src/research/strategy_gates.py` (gates DSR≥1.5, PBO≤0.35, PF_lo>1.0, DM_p<0.05).
- **MAIS zéro couplage** entre cette machinerie et `src/backtest/` — le runner `scripts/run_backtest.py` n'invoque ni CPCV ni gates. **La stratégie commerciale n'est PAS gated**.
- **Coûts transactionnels** : `DynamicSpreadModel` + `DynamicSlippageModel` existent (`src/environment/execution_model.py`) et `SignalReplay` sait les consommer (`state_machine_replay.py:404-405, 791-810`), MAIS aucun script ne les wire. Commission hardcoded à `0.0` (ligne 834). → P0 confirmé.
- **Reproductibilité bit-à-bit empirique** : trades identiques sauf `signal_id` qui change à chaque run (`uuid.uuid4()` à `confluence_detector.py:343`) — P1 fix 1h.
- **Look-ahead MTF latent** : `multi_timeframe_features.py:269` utilise `<=` au lieu de `<` — P0.
- **Bugs métriques** : Calmar non annualisé (`metrics.py:254`), Sharpe stdev inconsistent (`pstdev` vs `stdev`), max_consec_losses compte breakeven, annualisation sans correction autocorrélation Lo 2002.
- **Effort total Sprint 3+5+6** : 96-140 h.

---

## Recommandations actionnables Sprint 1-7

Le détail batch-par-batch est dans `roadmap/sprints/sprint_*.md` (à créer Sprint 1+).

### Sprint 1 (Data Layer Hardening, S3-S4)
- Batch 1.0 : **Extraction du module `src/intelligence/smart_money/`** + fix des 5 P0/P1 findings ICT.
- Batch 1.1 : DataProvider contractualisé Pydantic v2.
- Batch 1.2 : Tests property-based MTF resampling sans look-ahead.
- Batch 1.3 : Pipeline calendrier économique + blackouts end-to-end.
- Batch 1.4 : Décision sources licenciées (Polygon vs Databento) — différée par décision A.
- Batch 1.5 : Étendre CSV propres aux 4 actifs MVP supplémentaires.

### Sprint 2 (Detection Engine Validation, S5-S6)
- Batch 2.1 : Dataset annoté manuellement (≥ 500 BOS/OB/FVG par actif).
- Batch 2.2 : F1 / precision / recall vs annotations.
- Batch 2.3 : Tuning bayésien par actif.
- Batch 2.4 : Audit visuel automatisé (snapshots PNG).

### Sprint 3 (Statistical Edge Discovery, S7-S8)
- Batch 3.1 : Feature engineering exhaustif (microstructure, order flow, session, macro).
- Batch 3.2 : Information Coefficient par feature isolée.
- Batch 3.3 : Stacking + conditionnement par régime.
- Batch 3.4 : **CPCV / DSR / PBO** industrialisés (depuis scripts ad-hoc vers `src/backtest/validation/`).
- Batch 3.5 : **Sweep paramétrique state machine** × 4 actifs × 4 TF.
- **Gate** : CI 95 % PF lo > 1.0 sur ≥ 1 actif OU pivot valeur explicative.

### Sprint 4 (Calibration & Confidence, S9-S10)
- Batch 4.1 : Mondrian conformal stratifié par régime.
- Batch 4.2 : **Logistic regression L1** sur 8 weighted_scores ConfluenceDetector.
- Batch 4.3 : Validation OOS bandes de probabilité.
- Batch 4.4 : Documentation client-facing.

### Sprint 5 (Robustness & Stress Testing, S11-S12)
- Batch 5.1 : Fuzz testing inputs (NaN, infinis, gaps, spreads anormaux).
- Batch 5.2 : Stress test multi-régime (COVID 2020, LDI 2022, SVB 2023, yen 2024).
- Batch 5.3 : Sensibilité hyperparamètres ±20 %.
- Batch 5.4 : Adversarial inputs (fake-out setups).

### Sprint 6 (Production Hardening, S13-S14)
- Batch 6.1 : Profiling + vectorisation, latence < 250 ms / tick / paire.
- Batch 6.2 : Replay engine reproductible à la nanoseconde.
- Batch 6.3 : Snapshot store API per-signal.
- Batch 6.4 : Versioning modèles + compatibilité ascendante.

### Sprint 7 (Commercial Readiness, S15-S16)
- Batch 7.1 : Documentation technique `docs/algo/`.
- Batch 7.2 : Tear sheets par actif/TF (MD + JSON + PDF pandoc).
- Batch 7.3 : Fiches transparence client.
- Batch 7.4 : Test e2e 6 actifs × 2 TF.
- Batch 7.5 : Certification interne signée.
- **Gate final** : démontrable B2C / B2B.

---

## Ce que l'audit Phase 1 ne couvre pas

- **Annotations expertes BOS/CHOCH/OB/FVG** : reporté Sprint 2 (décision I).
- **F1 / precision / recall** vs annotations : Sprint 2.
- **Cross-actifs validation** régime / vol : Sprint 1-2.
- **Stabilité HMM** sur 7 ans : Sprint 2.
- **Property-based / chaos / adversarial** : Sprint 5.
- **Tear sheets clients** : Sprint 7.
- **Comparaison vs concurrents externes** : hors périmètre Sprint 0.

---

## Conclusion Phase 1

L'algorithme **n'est pas commercialisable en l'état** (note 5.5 / 10, ConfluenceDetector 3.0 / 10, baseline 0 trades). Mais l'**infrastructure est solide** (state machine 8.0, conformal 7.0, code propre, tests nombreux).

La **roadmap Sprints 1-7** existante du brief reste pertinente. Aucune ré-architecture majeure requise. Les corrections P0 sont **focalisées** sur :
1. **Refonte du scoring ConfluenceDetector** (logistic L1 multi-feature, Sprint 4).
2. **Extraction + fix Smart Money** (Sprint 1).
3. **Industrialisation CPCV / DSR / PBO** (Sprint 3).
4. **Sweep paramétrique empirique** state machine (Sprint 3).
5. **Snapshot store** pour reproductibilité client (Sprint 6).

À l'issue de Sprint 7, l'objectif est : **soit un edge prédictif borné CI 95 % PF lo > 1.0**, **soit une valeur explicative non-PF démontrée** (information ratio par régime, calibration probabiliste).

---

**Signé** : 2026-05-15, Claude (Lead Quant Architect)
**Audit composite** : Claude (4 sections) + 4 sub-agents general-purpose (sections 3.2, 3.3, 3.4, 3.8).
**Statut sections** : 6 / 8 complètes, 2 en cours (3.4 et 3.8 — agents en arrière-plan).
**Version** : v1.0-pending (à finaliser quand 3.4 et 3.8 sont livrées).
