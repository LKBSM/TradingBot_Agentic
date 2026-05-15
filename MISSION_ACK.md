# Mission Acknowledgment — Algo Layer Institutional Overhaul

**Date d'accusé** : 2026-05-15
**Tag de référence** : `v0.9.0-pre-institutional` (à créer dans Sprint 0 batch 0.1)
**Lead** : Claude (Lead Quant Architect par défaut jusqu'à instanciation de l'équipe d'agents)

---

## 1. Compréhension du contexte

Le produit Smart Sentinel AI a pivoté de RL → narrative-first le 2026-05-01 après le verdict A1 (DSR = 0.000, PBO = 0.50, PF CPCV = 1.008, score 1/6). Quatre stratégies actuelles ne franchissent pas le seuil CI 95 % PF lo > 1.0 :

| Stratégie    | PF    | IC 95 %        |
| ------------ | ----- | -------------- |
| XAU M15 v2   | 1.04  | [0.92, 1.17]   |
| XAU H1       | 0.95  | —              |
| EURUSD M15   | 0.85  | exclut 1.0     |
| NR4          | 0.60  | —              |

Le pipeline algorithmique vit dans `src/intelligence/` (7 étages déterministes). Bloqueurs connus avant cette mission :

- **Data quality 3.5/10** (eval_08) : CSV XAU 2019-2025 à 63 % de coverage, BOS fire sur 100 % des bars, licence Dukascopy en zone grise commerciale.
- **ConfluenceDetector** : score 0-100 sans pouvoir prédictif (Pearson −0.023, Brier > baseline).
- **3 moteurs risk concurrents** incohérents (eval_19).
- **Backtest engine 2/10** : aucun walk-forward propre, coûts de transaction $0, look-ahead MTF.

---

## 2. Périmètre — INCLUS (je touche)

- `src/intelligence/data_providers.py` (et tout module `data_providers/` futur)
- `src/intelligence/` : confluence_detector, volatility_forecaster, volatility_lgbm, regime_classifier, regime_filter, regime_gate, bocpd, conformal_wrapper, signal_state_machine, sentinel_scanner, semantic_cache, circuit_breaker, score_calibration, data_quality, state_persistence, feature_explainer, stylized_facts, cross_asset_correlation, forward_test_paper
- `src/backtest/` : state_machine_replay, metrics, report, news_replay
- `src/environment/` (uniquement features marché) : multi_timeframe_features, strategy_features, risk_manager (zone discutée — voir §6), feature_reducer, execution_model
- `src/agents/market_regime_agent.py`, `regime_predictor.py`, `multi_timeframe.py`, `data/multi_instrument_quality.py` (parties algo uniquement)
- Tests associés à tous les modules ci-dessus
- Scripts `scripts/eval_*`, `scripts/audit_*`, `scripts/backtest_*`, `scripts/download_*`, `scripts/colab_*` (data / quant)
- `data/` (OHLCV CSV, calendrier économique, macro)
- Toute la chaîne de validation statistique : CPCV, DSR, PBO, walk-forward, reality check

---

## 3. Périmètre — EXCLU (je ne touche jamais)

- `src/intelligence/llm_narrative_engine.py`
- `src/intelligence/template_narrative_engine.py`
- `src/intelligence/rag/` (et `data/rag/`)
- `src/intelligence/prompt_registry.py`, `narrative_quality.py`, `llm_cost_policy.py` (couche narrative)
- `src/delivery/` (Telegram, Discord, webapp, dashboards)
- `src/api/` (REST, webhooks, signal_store, tier_manager, routes)
- Tout module compliance, geo-blocking, disclaimers, MiFID II, CGU
- Stripe, pricing, GTM, tiers utilisateurs
- `infrastructure/` (Docker, Prometheus, Railway)
- `mockups/`, `Script collab` (artefacts UX/marketing)

**Règle de fer** : si je détecte un bug évident hors périmètre, je le consigne dans `OUT_OF_SCOPE.md` et je continue.

---

## 4. Objectif final accepté

L'algorithme doit être à la fin de la mission :

1. **Statistiquement défendable** — perf bornée avec IC 95 % honnêtes, OU valeur explicative non-PF démontrée (IR par régime, calibration probabiliste).
2. **Robuste cross-actifs / cross-timeframes** — MVP : XAU M15, XAU H1, EURUSD M15, EURUSD H1 ; framework de config étendable à BTC / US500 / GBP / JPY sans refactor.
3. **Production-grade** — latence < 250 ms / tick scanner / paire, observabilité complète, déterminisme documenté.
4. **Auditable client par client** — chaque signal rejouable bit-à-bit avec contexte exact, features et score décomposé.
5. **Commercialisable** — un client B2C retail OU un broker B2B peut consommer l'indicateur avec une tear sheet honnête et une doc technique complète.

---

## 5. Critères d'acceptation commerciale (gates finaux)

- ✅ Performance honnête : tear sheet par actif/TF (PF, Sharpe déflé, max DD, IC 95 %)
- ✅ Calibration prouvée : bandes conformelles couvrent nominal ± 2 % OOS
- ✅ Reproductibilité : tout signal 12 derniers mois rejouable à l'identique
- ✅ Robustesse : tous stress tests historiques (COVID 2020, LDI 2022, SVB 2023, yen 2024) + adversarial passent
- ✅ Latence : < 250 ms / tick / paire en prod
- ✅ Couverture de test : ≥ 90 % ligne, ≥ 80 % branche, mutation score ≥ 70 %
- ✅ Documentation `docs/algo/` complète, lisible quant externe
- ✅ Transparence client : aucune métrique sans IC + fenêtre de validation

---

## 6. Décisions techniques tranchées (registre complet : `audits/2026-Q2/sprint_0_decisions.md`)

Toutes les ambiguïtés du brief sont tranchées en spécialiste, sans demande de validation user. Synthèse :

- **A — Source XAU primaire** : `XAU_15MIN_2019_2026.csv` si audit coverage Batch 0.0 ≥ 95 %, sinon `2019_2024.csv` + extension Dukascopy. Décision finale ancrée au pre-flight.
- **B — Risk engine canonique Sprint 0-4** : `src/environment/risk_manager.py` gelé comme oracle de backtest. Refonte unifiée `src/intelligence/risk/` planifiée Sprint 5.
- **C — Format tear sheets** : Markdown + JSON pendant Sprint 0-6, génération PDF auto via pandoc en Sprint 7.
- **D — Stack régime canonique** : `regime_filter.py` + `regime_gate.py` + `bocpd.py` (intelligence/) canoniques ; `regime_classifier.py` utilitaire HMM ; `agents/market_regime_agent.py` et `regime_predictor.py` déclarés legacy figés.
- **E — Smart money extraction** : module `src/intelligence/smart_money/` extrait au Sprint 1 batch 1.0 (avant data hardening).
- **F — Branching git** : branche `institutional-overhaul` depuis tag `v0.9.0-pre-institutional`, merge vers `main` uniquement à certification Sprint 7.
- **G — CI minimale** : GitHub Actions workflow algo_tests dès Sprint 0 batch 0.1.
- **H — Snapshot config** : `reports/baseline/config_snapshot_2026-05-15.json` exhaustif (config.py + env vars + hashs code + hashs data + pip freeze + commit SHA).
- **I — Audit ICT Sprint 0** : code path + statistiques d'activation seulement. F1 vs annotations expertes reporté Sprint 2.
- **J — WIP non-committé** : préservé dans branche `wip/pre-institutional-2026-05-15`.
- **K — Buffer Sprint 0** : 8 h (12 % du sprint). Total 66 h sur 2 sem.
- **L — Reporting** : `sprint_0_progress.md` append/batch, `sprint_0_retrospective.md` à la clôture.
- **M — Tests existants** : suite 1 366+ verte à chaque commit. Flaky `test_short_roundtrip_pnl` marqué avec re-run auto.
- **N — Batch 0.0 ajouté** : pre-flight env + audit coverage XAU avant tout autre travail.

---

## 7. Règles de travail acceptées

- **TDD strict** : aucun code de production sans test couvrant.
- **Commits petits et fréquents** avec conventions sémantiques (`feat(detection):`, `fix(data):`, `test(replay):`, etc.).
- **Honnêteté statistique** : PF marginal ou négatif reporté tel quel. Pas d'embellissement.
- **Pas de scope creep** : LLM / distribution / compliance restent intouchés.
- **Reporting fin de sprint** : ce qui a été fait, gates passées/échouées, arbitrages demandés.
- **Auto-mode** : j'exécute les batches du sprint validé sans demander confirmation par batch, MAIS je m'arrête à chaque gate de sortie de sprint pour validation.

---

## 8. Hors-périmètre flagué à l'accusé (à logger dans OUT_OF_SCOPE.md)

À ce stade aucun bug hors-scope détecté pendant la lecture du brief. Le log s'enrichira au fil de l'exécution.

---

## 9. Première séquence d'actions exécutée

| # | Action                                                                    | Statut       |
| - | ------------------------------------------------------------------------- | ------------ |
| 1 | Lecture intégrale du brief                                                | ✅ Fait      |
| 2 | Inventaire dépôt → `audits/2026-Q2/repo_inventory.md`                     | ✅ Fait      |
| 3 | Plan Sprint 0 v1 → `roadmap/sprints/sprint_0.md`                          | ✅ Fait      |
| 4 | Décisions techniques tranchées → `audits/2026-Q2/sprint_0_decisions.md`   | ✅ Fait      |
| 5 | Plan Sprint 0 v2 révisé (5 batches, 66 h)                                 | ✅ Fait      |
| 6 | Init `OUT_OF_SCOPE.md` et `CHANGELOG.md`                                  | ✅ Fait      |
| 7 | Démarrage Batch 0.0 (pre-flight)                                          | ▶️ EXÉCUTION |

---

**Acquittement** : j'ai lu, j'ai tranché, j'exécute le Sprint 0 sous responsabilité spécialiste. Les checkpoints utilisateur sont à la fin de chaque sprint, pas par batch.
