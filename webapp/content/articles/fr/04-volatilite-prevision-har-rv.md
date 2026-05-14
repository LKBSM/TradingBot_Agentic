---
title: "Prévision de volatilité sur l'or — pourquoi HAR-RV bat ATR"
slug: volatilite-prevision-har-rv
locale: fr
published_at: 2026-05-13
tags: [volatility, har-rv, education]
keyword_target: "prevision volatilite or"
---

# Prévision de volatilité sur l'or

Prédire **dans quel range va évoluer le prix** dans les prochaines heures n'est pas la même chose que prédire la direction. La volatilité est statistiquement plus **persistante** que les rendements (clustering), donc plus prévisible.

## Pourquoi ATR_14 est insuffisant

Le **Average True Range sur 14 périodes** est la mesure de volatilité la plus utilisée par les retail traders. Problème : ATR_14 a une **autocorrélation de 0.95**, ce qui le rend très lent à réagir. Une prévision basée sur ATR_14 est essentiellement "demain ≈ aujourd'hui", ce qui est correct 90% du temps mais inutile aux moments qui comptent (transitions de régime).

## HAR-RV : trois horizons combinés

Le modèle **Heterogeneous Autoregressive de Realized Volatility** (Corsi 2009) combine trois mémoires :

- volatilité **journalière** (dernière séance),
- volatilité **hebdomadaire** (moyenne 5 dernières séances),
- volatilité **mensuelle** (moyenne 22 dernières séances).

L'intuition : différents acteurs ont différents horizons de décision. Un day-trader réagit à la vol d'hier ; un fund weekly réagit à la vol hebdo ; un asset allocator mensuel réagit à la vol mensuelle. HAR-RV agrège leurs influences linéairement.

## Performance observée

Sur 6 ans XAU/USD M15 (rolling walk-forward), notre implémentation HAR-RV produit :

| Modèle | RMSE relatif | Latence forecast |
|---|---|---|
| Naïve (vol_t+1 = vol_t) | 1.00 (baseline) | < 1ms |
| ATR_14 | 1.05 (légèrement pire) | < 1ms |
| HAR-RV (notre prod) | 0.85 (−15% RMSE) | ~3ms |
| LightGBM 50 features | 0.69 (−31% RMSE) | ~1.6s |
| HAR + LGBM résidu (hybrid) | 0.70 (−30% RMSE) | ~5s |

LightGBM bat HAR-RV statistiquement (Diebold-Mariano p < 1e-12), **mais** sa latence d'inférence le rend impraticable pour notre cible 50ms/forecast. **Nous opérons donc en HAR-RV** par défaut (variable `VOL_MODE=har`).

> Source : eval interne avril 2026, voir `reports/eval_04_volatility_findings.md`.

## Utilité narrative

Une prévision de vol sert à :

1. **Calibrer les niveaux de stop** : un stop à 0.5 ATR en régime calme n'est pas le même qu'un stop à 0.5 ATR en stress.
2. **Sizing implicite** : un risque fixe en R suppose une vol stable ; en réalité un même R en stress correspond à un mouvement plus grand.
3. **Conviction de l'analyse** : haute vol = plus d'incertitude → narrative plus prudente.

## Limites

- HAR-RV ne capte **pas les sauts** discontinus (gaps de week-end, news shocks). Combinaison nécessaire avec un detector dédié.
- Pas de propriétés théoriques sur la distribution prédictive : on a une moyenne, pas un intervalle de confiance fiable.
- Diurnal effect ignoré dans la version vanilla : la vol XAU à 14h UTC n'est pas comparable à 3h UTC.

## Pour aller plus loin

- Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility", JFE 2009.
- Andersen, Bollerslev, Diebold, Labys, "The Distribution of Realized Exchange Rate Volatility", JASA 2001.

*Analyse algorithmique éducative. Pas un conseil en investissement.*
