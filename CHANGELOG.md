# Changelog

Tous les changements notables de la couche algorithmique Smart Sentinel AI sont consignés ici.

Format basé sur [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versionnage [SemVer](https://semver.org/lang/fr/).

Périmètre : `src/intelligence/`, `src/backtest/`, `src/environment/` (features), `src/agents/` (algo-relevant), `scripts/`, `data/`, `tests/` associés.

Hors périmètre (consigné dans `OUT_OF_SCOPE.md`) : LLM, API, delivery, compliance, infra.

---

## [Unreleased] — Sprint 0 en cours

### Préparation (avant code)
- Ajout `MISSION_ACK.md` (accusé de réception du brief + décisions tranchées).
- Ajout `audits/2026-Q2/repo_inventory.md` (cartographie initiale du dépôt algo).
- Ajout `audits/2026-Q2/sprint_0_decisions.md` (registre des 14 décisions techniques A→N tranchées par le spécialiste).
- Ajout `roadmap/sprints/sprint_0.md` (plan révisé, 5 batches, 66 h).
- Ajout `OUT_OF_SCOPE.md` (log des findings hors périmètre).
- Ajout de ce `CHANGELOG.md`.

### À venir (exécution Sprint 0)
- Batch 0.0 : pre-flight env + audit coverage XAU 2019-2026.
- Batch 0.1 : tag `v0.9.0-pre-institutional` + branche `institutional-overhaul` + CI minimale.
- Batch 0.2 : baseline backtest reproductible (XAU M15/H1, EURUSD M15/H1) + IC 95 % bootstrap.
- Batch 0.3 : audit institutionnel Phase 1 (8 sections).
- Batch 0.4 : fix data P0 + test régression BOS.

---

## [0.9.0-pre-institutional] — À créer Sprint 0 batch 0.1

Tag de freeze de l'état pré-institutionnel. À utiliser comme référence immuable pour comparer toute évolution algo ultérieure.

---

**Init** : 2026-05-15
