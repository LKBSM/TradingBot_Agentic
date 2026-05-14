---
title: "Les régimes de marché sur l'or — pourquoi un trading qui marche en range échoue en stress"
slug: regimes-marche-xau
locale: fr
published_at: 2026-05-13
tags: [regime, xau, volatility, education]
keyword_target: "regimes de marche or"
schema_type: Article
---

# Les régimes de marché sur l'or

L'or ne se comporte pas de la même façon selon les périodes. Une stratégie de mean-reversion qui fonctionne magnifiquement en marché calme s'effondre en pleine panique post-FOMC. Comprendre les **régimes de marché**, c'est admettre que **les statistiques du prix changent dans le temps**.

## Trois régimes utiles

Notre algorithme (`src/intelligence/regime_classifier.py`) classifie chaque période XAU/USD M15 en trois états via un **HMM 3-états** sur les rendements :

1. **`low_vol_trending`** — faible volatilité, drift directionnel persistant. Le marché monte ou descend en pente douce. Mean-reversion meurt ; trend-following respire.

2. **`low_vol_ranging`** — faible volatilité, drift nul. Le prix oscille dans un canal. Mean-reversion brille ; trend-following se fait essorer.

3. **`high_vol_stress`** — forte volatilité, drift incertain. Typique après FOMC, NFP, ou crise géopolitique. **Toutes** les stratégies systématiques perdent de la valeur : les coûts (spread, slippage) explosent.

## Pourquoi un HMM et pas un seuil ATR ?

Un seuil ATR (par exemple "vol > 1.5× moyenne 30j → stress") capte la **variance**, mais rate le **drift** : un range serré sans tendance n'est pas le même régime qu'une tendance lente et serrée. Un HMM apprend conjointement les deux dimensions.

Les transitions HMM sont **persistantes** : une fois en `high_vol_stress`, le marché y reste typiquement plusieurs heures à plusieurs jours — c'est exactement le comportement qu'on veut capter.

## Distribution observée

Sur 6 ans d'historique XAU/USD M15 (Dukascopy 2019-2024, ~210 000 bars), notre classifier produit la distribution suivante :

| Régime | Part du temps |
|---|---|
| `low_vol_ranging` | ~52% |
| `low_vol_trending` | ~31% |
| `high_vol_stress` | ~17% |

> Source : backtest interne 6 ans, voir [transparency page](/fr/transparency).

## Utilité narrative

Le régime sert principalement à **calibrer l'attente** dans une analyse. Une cassure de structure en régime `low_vol_trending` continue typiquement ; en `low_vol_ranging` elle est plus susceptible d'échouer (range respecté) ; en `high_vol_stress` elle peut être un faux signal de panique.

## Limites

- **Pas un signal d'entrée** : connaître le régime ne dit pas quand entrer, juste comment lire le contexte.
- **Latence** : le HMM identifie un régime quand il est déjà installé — il n'anticipe pas le changement.
- **Spillover sessions** : la transition Asie → Londres peut produire un "faux stress" qui se normalise rapidement.

## Pour aller plus loin

- Hamilton, "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle", Econometrica 1989 — paper fondateur sur les régimes HMM.
- Bemporad et al. "Statistical Jump Models" 2018 — alternative SOTA aux HMM (que nous n'avons pas adoptée en Phase 2B par souci de simplicité).

*Analyse algorithmique éducative. Pas un conseil en investissement.*
