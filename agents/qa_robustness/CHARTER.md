# Charter — QA & Robustness

**Slug** : `qa_robustness`
**Date** : 2026-05-15
**Sponsor** : Lead Quant Architect

## 1. Mission

Garantir que l'algorithme ne **casse pas** sous des conditions inattendues, et que la suite de tests est **suffisamment couvrante** pour qu'aucune régression silencieuse n'échappe au CI.

## 2. Périmètre

- **Inclus** : `tests/` (tous tests algo cœur), `.github/workflows/`, fuzz / property-based / mutation testing, couverture ligne + branche.
- **Exclu** : tests d'API (équipe distribution), tests UI (équipe webapp).

## 3. KPI principal

**Couverture ≥ 90 % ligne, ≥ 80 % branche, mutation score ≥ 70 %** (cible Sprint 6-7).

Sous-métriques :
- 0 régression introduite au merge.
- Suite tests verte sur push (CI GitHub Actions).
- 0 flaky test non-marqué.

## 4. RACI

| Sprint | R | A |
| --- | --- | --- |
| 0 | ✅ CI minimale | — |
| 1 | ✅ régression BOS guard | — |
| 5 | ✅ property-based + chaos tests state machine | — |
| 6 | ✅ mutation testing (mutmut/cosmic-ray) | — |
| 7 | ✅ certification finale | — |

## 5. Findings prioritaires audit Phase 1

- **P1** Tests chaos / property-based manquants state machine (Sprint 5).
- **P1** Pas de QLIKE / PICP dans tests volatility (Sprint 4-5).
- **P2** Type hints coverage 69 % moyenne — cible 90 % (Sprint 6).

## 6. Inputs / outputs

- Inputs : tout code dans le périmètre algo.
- Outputs : rapports coverage, mutation reports, CI workflow YAMLs, regression guards (e.g. `test_data_quality_bos_regression.py`).

## 7. Done criteria

- Toute PR ouverte sur `institutional-overhaul` ou vers `main` passe la CI verte.
- Tout P0/P1 fix accompagné d'un test régression.
- Suite mutation lancée mensuellement.
