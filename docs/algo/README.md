# Smart Sentinel AI — Algorithm Documentation

**Version** : 1.0-institutional (Sprint 7 commercial readiness)
**Audience** : external quant reviewers, sophisticated clients, B2B integrators
**Date d'édition** : 2026-05-15

---

## Pourquoi cette documentation

Conformément aux critères de commercialisation (brief §6) :
> *Documentation : `docs/algo/` complet, lisible par un quant externe, sans dette implicite.*

Ce dossier expose la totalité de la couche algorithmique de manière à ce qu'un quant externe puisse :
1. Comprendre le pipeline en moins de 30 minutes.
2. Reproduire la baseline bit-à-bit.
3. Vérifier les claims de performance (avec leurs IC).
4. Identifier les limites (ce que l'algo ne sait PAS faire).

---

## Table des matières

| Fichier                          | Sujet                                                  |
| -------------------------------- | ------------------------------------------------------ |
| [architecture.md](architecture.md) | Pipeline 7 étages, dépendances, flux de données       |
| [data_layer.md](data_layer.md)   | Sources OHLCV, calendrier économique, MTF, validation |
| [smart_money.md](smart_money.md) | BOS, CHOCH, Order Blocks, FVG, retest                  |
| [confluence_detector.md](confluence_detector.md) | Scoring 0-100, 8 composantes, calibration |
| [volatility_forecaster.md](volatility_forecaster.md) | HAR-RV, LightGBM, hybride, calibration |
| [regime_stack.md](regime_stack.md) | HMM, BOCPD, RegimeFilter, RegimeGate                  |
| [conformal.md](conformal.md)     | Split conformal, ACI, Mondrian stratifié               |
| [state_machine.md](state_machine.md) | HOLD/BUY/SELL, hystérésis, cooldown, lockout       |
| [backtest_engine.md](backtest_engine.md) | Replay, CPCV, DSR, PBO, walk-forward             |
| [snapshot_store.md](snapshot_store.md) | Reproductibilité per-signal Sprint 6              |
| [glossary.md](glossary.md)       | Termes ICT, métriques quant, références                |

---

## Repère rapide

### Pipeline (7 étages, ordre d'exécution)

```
1. DataProvider       → OHLCV M15 + multi-TF resample
2. SmartMoneyEngine   → BOS / CHOCH / OB / FVG / retest
3. ConfluenceDetector → score 0-100 (8 composantes)
4. VolatilityForecaster → HAR-RV (défaut), LGBM, hybride
5. RegimeFilter + RegimeGate → blackout / REDUCE / BLOCK
6. ConformalWrapper   → bandes de confiance OOS
7. SignalStateMachine → HOLD / BUY / SELL avec hystérésis
   → Snapshot store (audit per-signal)
   → Notification queue
```

### Décisions techniques actées

Toutes documentées dans `audits/2026-Q2/sprint_0_decisions.md` :

- **A** — Source XAU primaire = `XAU_15MIN_2019_2026.csv` (98.72 % coverage).
- **D** — Régime canonique = `regime_filter` + `regime_gate` + `bocpd`.
- **E** — Smart money module facade Sprint 1, extraction physique Sprint 6.

---

## Limites connues (transparence)

| Limite                                                       | Statut                                                     |
| ------------------------------------------------------------ | ---------------------------------------------------------- |
| Score ConfluenceDetector empiriquement non prédictif (Pearson −0.008) | À corriger Sprint 4 (logistic L1 + isotonic)             |
| Smart Money OB = engulfing (≠ ICT)                          | À corriger Sprint 2 (P0-2)                                 |
| 0 signal tradable avec defaults sur 7 ans                    | Sweep paramétrique Sprint 3                                |
| Pas d'annotations expertes BOS/CHOCH/OB/FVG                 | Sprint 2 batch 2.1                                         |
| Conformal PICP 43.6 % vs cible 80 %                         | Mondrian par régime Sprint 4                               |
| HAR perd contre naive sur QLIKE 2024                        | Sprint 4                                                   |
| Latence HAR p99 91 ms (cible 50 ms)                         | Vectorisation Sprint 6                                     |

---

## Reproductibilité (commande unique)

```bash
git checkout v0.9.0-pre-institutional   # tag Sprint 0
python scripts/run_baseline_sprint0.py  # ~5 min, XAU + EURUSD M15
sha256sum reports/baseline/checksums.txt
```

Toute différence indique une dérive (env, data, ou code).

---

## Audit ancestral

L'algorithme a été audité Phase 1 en Sprint 0 (2026-05-15) — voir `audits/2026-Q2/algo_audit_institutional.md`. Note pondérée : 5.61 / 10. 21 P0 + 15 P1 + 7 P2 identifiés.

Cette documentation s'enrichira à chaque sprint qui clôture des P0/P1.

---

**Maintainer** : Lead Quant Architect (`agents/lead_quant_architect/CHARTER.md`).
**Issue tracker** : GitHub Issues sous le tag `algo`.
