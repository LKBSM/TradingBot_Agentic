# Décision — M1 retiré du périmètre produit (5 unités : M5 → D1)

**Date** : 2026-08-16
**Mission** : DATA-1 (audit consommation fournisseur de données)
**Statut** : **TRANCHÉE**
**Portée** : périmètre des unités de temps exposées, scannées et rafraîchies.

---

## Décision

Le périmètre produit est arrêté à **5 unités de temps : M5, M15, H1, H4, D1**.
**M1 est retiré du périmètre** — ni exposé, ni scanné, ni rafraîchi en continu.

## Raisons

1. **Qualité SMC insuffisante en 1 minute.** La structure Smart Money en M1 est trop bruitée
   pour produire des zones (OB / FVG / liquidité) exploitables : le rapport signal/bruit y est
   trop bas pour une lecture descriptive fiable. Le produit ne doit afficher que ce qu'il peut
   défendre.
2. **Plancher de consommation incompressible.** M1 clôture *chaque minute* : à N marchés, il
   impose un plancher de **N crédits/minute non étalable** (impossible à lisser par jitter,
   contrairement à M5 qui ne clôture que toutes les 5 min). À 80 marchés, M1 seul = 1440
   appels/marché/jour et un plancher de 80 crédits/min — le facteur dominant de la facture
   fournisseur. Cf. `docs/audits/AUDIT-data-1.md` §3–§5.

## État constaté au moment de la décision (déjà conforme)

M1 était déjà hors des surfaces produit dans `origin/main` :

| Surface | M1 | Preuve |
|---|---|---|
| Sélecteur d'unités (/app) | absent | `webapp/lib/market-reading/perimeter.ts` — `DISPLAY_TIMEFRAMES` filtre M1 sauf `NEXT_PUBLIC_LB1_ENABLE_M1` (défaut off) |
| Scanner | non scanné | `conditions_scan.py` — `SCAN_COMBOS = enabled_combos()` ; `enabled_timeframes()` retire M1 tant que `LB1_ENABLE_M1` off |
| Scheduler (warm) | non warm | `lookback_config.live_warm_combos()` exclut M1 |
| Validation API | accepté (deep-link) | `SUPPORTED_TIMEFRAMES` inclut M1 — aucune surface UI n'y mène |

## Vérifié : le retrait ne casse rien

- **Conditions du scanner** : M1 n'est déjà pas scanné → aucun changement.
- **Alignement multi-unités** : `alignment_timeframes()` / `upper_timeframes()` ne regardent que
  vers le haut ; aucune unité ne référence une unité en dessous d'elle. M1 (barreau le plus bas)
  n'est source de rien.
- **Lecture narrée / régime / MTF bias** : par-combo ; `build_cache_mtf_provider` lit les unités
  supérieures uniquement.
- `lower_timeframe(M5) == "M1"` existe dans le registre mais **n'a aucun appelant** dans `src/`.
- Les listes M1 résiduelles (`security.VALID_TIMEFRAMES`, `insight_signal_v2`, `backtest/metrics`)
  sont des sur-ensembles inoffensifs.

## Conséquence opérationnelle

`LB1_ENABLE_M1` / `NEXT_PUBLIC_LB1_ENABLE_M1` restent **off** (posture par défaut, désormais
posture *arrêtée*). Si un besoin M1 réapparaît, il devra être justifié contre ces deux raisons et
ne jamais être activé en polling live intégral (historique + paresseux avec âge affiché seulement).

**Réf. chiffrage** : `docs/audits/AUDIT-data-1.md`.
