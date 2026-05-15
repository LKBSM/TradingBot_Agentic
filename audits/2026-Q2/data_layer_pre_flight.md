# Pre-flight — Décision data layer XAU + EURUSD

**Date** : 2026-05-15
**Batch** : Sprint 0 — 0.0
**Source primaire** : `audits/2026-Q2/xau_coverage_audit.md`

---

## Tableau de synthèse — Coverage session active (Mon-Fri 07:00-21:00 UTC)

| CSV                                  | Lignes  | Plage                       | Coverage agrégée | Fraîcheur |
| ------------------------------------ | ------- | --------------------------- | ---------------- | --------- |
| `XAU_15MIN_2019_2024.csv`            | 141 524 | 2019-01-02 → 2024-12-30     | **98.73 %**      | 500 jours |
| `XAU_15MIN_2019_2025.csv`            | 106 643 | 2019-01-02 → 2025-12-31     | **63.71 %** ❌   | 134 jours |
| `XAU_15MIN_2019_2026.csv`            | 172 874 | 2019-01-02 → 2026-04-29     | **90.35 %** (2026 partiel à 31 %) | 15 jours |
| `XAU_15MIN_2025_2026_dukascopy.csv`  | 31 442  | 2024-12-30 → 2026-04-29     | 43.55 % (2024 résiduel)  | 15 jours |
| `EURUSD_15MIN_2019_2025.csv`         | 174 506 | 2019-01-01 → 2025-12-31     | **99.41 %** ✅   | 134 jours |

## Coverage 2019-2025 (intervalle MVP)

- `XAU_15MIN_2019_2026.csv` sur 2019-2025 : **98.72 %**
- `XAU_15MIN_2019_2024.csv` sur 2019-2024 : 98.73 %
- `XAU_15MIN_2019_2025.csv` sur 2019-2025 : 63.71 % (broken)

➡️ Le CSV `2019_2026` est identique au `2019_2024` sur 2019-2023 (mêmes lignes), **étend proprement** 2024-2026 avec quasi-mêmes critères qualité.

---

## Décision A — Tranchée

**Source XAU primaire** : `data/XAU_15MIN_2019_2026.csv`.

- Coverage 2019-2025 = **98.72 %** ≥ 95 % cible.
- Fraîcheur 15 jours (vs 500 pour 2019_2024).
- Couvre déjà 2026 partiel (31 % de l'année à mi-mai 2026, cohérent avec date du jour).
- **Pas besoin** de concaténer avec Dukascopy → la question de la licence Dukascopy est **différée** au sprint commercial (Sprint 7) et n'est plus bloquante pour Sprint 0-6.

**Source EURUSD primaire** : `data/EURUSD_15MIN_2019_2025.csv`.

- Coverage 99.41 % sur 2019-2025.
- Fraîcheur 134 jours — acceptable pour le MVP. À rafraîchir Sprint 1 batch 1.5 (étendre vers 2026).

---

## Sources legacy / à figer

| Fichier                              | Statut                                                            |
| ------------------------------------ | ----------------------------------------------------------------- |
| `XAU_15MIN_2019_2025.csv` (63 %)     | 🔒 **Interdit** comme source primaire. Garde-fou via test régression batch 0.4. Marqué `legacy_low_coverage` dans la doc. |
| `XAU_15MIN_2019_2024.csv` (98.73 %)  | ✅ Acceptable en backup, mais figé. Plus utilisé par défaut.       |
| `XAU_15MIN_2025_2026_dukascopy.csv`  | 🟡 Disponible pour cross-check Sprint 1, **pas de prod commerciale** tant que licence non clarifiée. |

---

## Findings de gaps

- `XAU_2019_2026.csv` : **1 892 gaps > 30 min** en session active sur 7 ans = ~270/an ≈ ~5/semaine. Cohérent avec fermetures weekend (Sun afternoon, US holidays). À auditer plus finement au batch 0.3 section 3.1.
- `XAU_2019_2025.csv` (broken) : 6 093 gaps = ×3 → cohérent avec sa coverage 63 %.

---

## Conséquences immédiates

1. ✅ Décision A actée — source `2019_2026`.
2. ⏭️ Batch 0.2 (baseline) utilisera ce CSV.
3. ⏭️ Batch 0.4 va patcher `config.py` pour pointer vers `XAU_15MIN_2019_2026.csv`.
4. ⏭️ Batch 0.4 va créer un test régression "BOS 100 % bars" qui tombera rouge si on revient sur le CSV cassé.
5. 📝 La licence Dukascopy est différée (logged comme finding mais non bloquant Sprint 0-6).

---

**Signé** : 2026-05-15, Claude (Lead Quant Architect)
