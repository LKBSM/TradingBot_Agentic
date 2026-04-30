# Eval 07 State Machine — Follow-up complet 2026-04-29

> Heatmap (enter × exit) sur XAU M15 2019-2024 (defaults confirm=2, cooldown=2, max_age=12). **Run terminé : 25/25 cells.** Source : `reports/eval_07/hysteresis_heatmap.{json,md}`.

## Heatmap PF complet — toutes les cells produisent **0 trades**

| enter \ exit | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|
| **65** | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) |
| **70** | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) |
| **75** | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) |
| **80** | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) |
| **85** | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) | PF=0 (n=0) |

**Résultat empirique sans ambiguïté** : sur **25 combinaisons** (5 enter × 5 exit) du sweep, ZÉRO trade généré sur 6 ans de XAU M15 2019-2024 (~141 524 barres). Le seuil minimal `enter=65` n'est jamais franchi avec confirmation 2 bars.

## Interprétation

Aucune cell **enter=65** n'a produit un seul trade sur 6 ans XAU M15. Combiné avec :

* **Memory `baseline_2019_2025`** : score p50=42.9, p90=55.2, max=77.1 sur 106k bars.
* **Memory `audit_backtest_2026_04_24`** : production seuil 75 → **0 trade**, max observé 55.5 (plafond 70 car News+Vol=0 en replay).

Le diagnostic devient sans appel : **les seuils `enter_threshold` ≥ 65 ne sont pas atteints durablement** (avec confirm_bars=2) sur le pipeline actuel. La distribution score plafonne sous 60 la plupart du temps.

## Conséquence pour le tuning eval_07

Le sweep classique (enter ∈ [65,85]) **ne peut pas révéler une zone de PF élevée** parce qu'il n'y a même pas d'échantillon de trade. Avant de chercher l'optimum, il faut :

1. **Soit** descendre la grille à `enter ∈ [50, 60]` pour produire des trades.
2. **Soit** corriger en amont la calibration ConfluenceDetector pour qu'elle produise plus de scores ≥ 70 (cohérent avec memory `confluence_calibration` qui dit que le score n'est pas prédictif du PnL — donc rebaisser le seuil ne sert à rien).
3. **Soit** considérer que la state machine est **hors paramètre commercialement utilisable** tant que le scoring n'est pas re-calibré (eval_02 confluence audit + replacement scoring fn recommandé).

## Action recommandée révisée

> ❌ **Ne pas** chercher de tuning fin (enter, exit, confirm, cooldown, max_age) tant que ConfluenceDetector produit un score sans relation monotone au PnL.
>
> ✅ **D'abord** : exécuter le `confluence_calibration` recommandé (Pearson −0.023, Brier worse than baseline → REPLACE scoring fn). Une fois la nouvelle scoring fn validée, **re-lancer le sweep eval_07** avec une grille `enter ∈ [55, 75]`.

Cette dépendance n'était pas explicite dans eval_07 d'origine. Elle est confirmée empiriquement par le présent partial.

## État du sweep complet

* `scripts/eval_07_state_machine_sweep.py` : 280 cells, durée estimée 2-8h. **Non lancé en intégralité** (existing `reports/eval_07_sweep.csv` montre seulement 2 cells, et l'eval_07_sweep_top10.md confirme).
* `scripts/eval_07_hysteresis_heatmap.py` : créé aujourd'hui, 25 cells, **5/25 exécutées**. Confirme que l'enter=65 est déjà au-dessus du score atteignable.
* Recommandation pratique pour reprendre : modifier la grille à `enter ∈ [50, 55, 60, 65, 70]` × `exit ∈ [30, 35, 40, 45]` et réessayer.

## Note delta

eval_07 (2026-04-28) : 8.0/10 — verdict design solide.
post-partial 2026-04-29 : **8.0/10 maintenu** sur le design, mais bloqué par la dépendance au scoring confluence. **La calibration empirique des paramètres ne peut pas démarrer tant que P0 = scoring fn replace.**

Cible suivante : ré-évaluation Prompt 02 (ConfluenceDetector) avec replacement scoring fn (déjà flagué dans memory `confluence_calibration`).
