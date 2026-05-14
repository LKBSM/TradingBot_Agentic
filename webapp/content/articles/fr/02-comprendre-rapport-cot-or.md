---
title: "Comprendre le rapport COT sur l'or — guide 2026"
slug: comprendre-rapport-cot-or
locale: fr
published_at: 2026-05-13
author: Smart Sentinel AI
tags: [cot, xau, macro, education]
keyword_target: "rapport cot or"
schema_type: Article
---

# Comprendre le rapport COT sur l'or

Chaque vendredi à 15h30 ET, la CFTC publie son **Commitment of Traders (COT)** : la photographie des positions ouvertes sur les contrats à terme de l'or à la date de mardi précédent. Pour qui suit XAU/USD, c'est l'une des **rares fenêtres officielles** sur le positionnement réel des grands acteurs.

## Trois catégories d'acteurs

Le COT segmente les open interests en trois groupes :

1. **Commercials** — hedgers professionnels (mineurs d'or, banques de bullion, raffineurs). Ils utilisent les futures pour couvrir leur exposition physique. Historiquement **contrarians par construction** : ils vendent les rallys et achètent les creux.

2. **Large Speculators (Managed Money + Other Reportable)** — hedge funds et CTAs. Tendance à suivre le momentum. Quand leur exposition longue atteint des extrêmes, c'est souvent un **signal d'épuisement**.

3. **Small Speculators (Non-reportable)** — investisseurs particuliers. Historiquement la catégorie qui se trompe le plus aux tournants.

## La lecture utile : les extrêmes

Une lecture intelligente du COT ne se fait pas en absolu mais en **percentile sur N semaines glissantes**. Notre algo (`src/agents/data/cot_provider.py`) calcule :

- `commercials_net_short_pct_52w` : où se situe le net-short commerciaux par rapport aux 52 dernières semaines ?
- `managed_money_long_pct_52w` : idem pour les longs spéculatifs.

Quand les commerciaux atteignent **leur plus gros net-short en 52 semaines**, cela signale historiquement une **sur-extension haussière du marché** (les hedgers couvrent agressivement parce que le prix est jugé élevé).

> Sources : CFTC Disaggregated Futures-Only reports, rapports hebdomadaires 1986-2025 indexés dans notre RAG `[source:cftc-disagg]`.

## Limites

1. **Lag de 3 jours** : le rapport publié vendredi reflète mardi. Pour le swing-trading H4+, c'est acceptable ; pour l'intraday M15, c'est inutilisable.
2. **Comportement non-linéaire** : un commerce net-short extrême peut le rester pendant des semaines avant retournement.
3. **Pas un signal direct** : le COT est un **contexte**, pas une stratégie autonome. Une lecture extrême combinée à une cassure de structure et à un calendrier macro favorable a plus de poids qu'isolée.

## Conclusion

Le COT est l'un des rares jeux de données **vraiment institutionnels et publics** sur l'or. Apprenez à lire les extrêmes percentiles plutôt que les niveaux absolus, et utilisez-le comme contexte multi-couches — jamais comme déclencheur seul.

*Analyse algorithmique éducative. Pas un conseil en investissement.*
