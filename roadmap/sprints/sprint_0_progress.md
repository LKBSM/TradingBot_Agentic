# Sprint 0 — Progress log

Append-only log, un bloc par batch. Statut ✅ ok / ⚠️ partiel / ❌ bloqué.

---

## Batch 0.0 — Pre-flight env + audit data layer — ✅ ok

**Date** : 2026-05-15
**Charge effective** : ~1 h (vs 4 h estimé — efficient car l'env était sain et l'audit s'est bien passé)
**Livrables** :
- `audits/2026-Q2/preflight_env.md` — Python 3.12.6, pytest 9.0.2, pandas 3.0.0, scikit-learn 1.8.0, lightgbm 4.6.0. `arch` non installé (warning bénin).
- `audits/2026-Q2/preflight_imports.md` — 17/17 imports algo OK, 2 696 tests collectés (vs 1 366 en mémoire — la suite a grandi).
- `audits/2026-Q2/xau_coverage_audit.md` — audit empirique 5 CSV.
- `audits/2026-Q2/data_layer_pre_flight.md` — **Décision A actée**.
- `scripts/audit_xau_coverage.py` — audit reproductible.
- `.gitignore` mis à jour pour `backups/`.

**Décision A actée** :
- Source XAU primaire = `data/XAU_15MIN_2019_2026.csv` (98.72 % coverage 2019-2025, fraîcheur 15 jours).
- Source EURUSD primaire = `data/EURUSD_15MIN_2019_2025.csv` (99.41 %).
- Licence Dukascopy **différée** : le CSV adopté couvre déjà 2025-2026 sans dépendre du Dukascopy.

**Bonus inattendu** : la décision A simplifie le Sprint 1 batch 1.4 (la question de la licence n'est plus bloquante).

---
