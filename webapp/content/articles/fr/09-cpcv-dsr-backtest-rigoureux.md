---
title: "Backtester sérieusement : CPCV, DSR, PBO — la trinité anti-overfitting"
slug: cpcv-dsr-backtest-rigoureux
locale: fr
published_at: 2026-05-13
tags: [backtest, methodology, education]
---

# Backtester sérieusement

99% des backtests retail sont **inutiles**. Pourquoi ? Parce qu'ils répondent à la mauvaise question. La question n'est pas "ma stratégie a-t-elle marché sur les données passées ?" — c'est "**dans quelle mesure le résultat passé est-il dû au hasard ou à l'overfitting ?**"

Trois outils académiques permettent de répondre.

## CPCV — Combinatorial Purged Cross-Validation

La cross-validation classique a deux trous fatals pour les séries temporelles :

1. **Fuite future → passé** : sans purge, les features calculés à t peuvent contenir de l'information de t+5 (par exemple, un label "trade gagnant" connu uniquement après expiration).
2. **K-fold aléatoire** : mélanger des dates passées et futures fait fuir le futur dans le train.

**CPCV** (López de Prado 2018) résout les deux :

- découpe le dataset en K blocs **temporellement contigus**,
- entraîne sur K-2 blocs, **purge** (drop) les n bars autour des frontières pour casser la fuite,
- ajoute une **embargo zone** post-test pour éviter le data snooping inverse,
- combine les paths CV de plusieurs façons (28 paths pour K=6 typique).

Notre `src/research/cpcv_harness.py` implémente cette logique.

## DSR — Deflated Sharpe Ratio

Le Sharpe classique suppose **une seule** stratégie testée. Si vous en testez 100 et gardez la meilleure, votre Sharpe affiché est **inflationné** par la sélection.

**DSR** (Bailey & López de Prado 2014) **dégonfle** le Sharpe pour tenir compte du nombre d'essais. Formule simplifiée :

```
DSR = (Sharpe_observed − E[max_Sharpe_under_null])
       /  sqrt(var_estimator)
```

Si vous avez testé 100 configurations et obtenu un Sharpe de 1.2 sur la meilleure, le DSR peut tomber à 0.3 — c'est-à-dire que **votre "meilleure stratégie" n'est statistiquement pas distinguable du hasard**.

**Notre verdict A1** sur XAU M15 LightGBM : `DSR = 0.0000`. Verdict : pas d'edge prédictif.

## PBO — Probability of Backtest Overfitting

**PBO** (Bailey, Borwein, López de Prado, Zhu 2017) répond directement : "**quelle est la probabilité que mon meilleur backtest perde de l'argent en out-of-sample ?**"

Méthode :
1. Découper N stratégies × K paths CPCV
2. Trouver la meilleure stratégie sur chaque sous-ensemble train,
3. Mesurer son rang dans le test associé,
4. PBO = fréquence à laquelle la meilleure in-sample finit dans la moitié basse out-of-sample.

**PBO > 0.5** = votre processus de sélection produit en moyenne des perdants. **Notre A1 a un PBO de 0.50** — pure aléatoire.

## Le cas Smart Sentinel

Notre verdict (`reports/a1_verdict_2026.md`) :

- DSR 0.00 (sur 100+ configurations testées)
- PBO 0.50
- CPCV PF (out-of-sample) : 1.008 — quasiment break-even avant coûts, perte après spread

**Conclusion qu'on en a tirée** : pas d'edge. Pivot Phase 2B narrative-first.

Plutôt que de truquer un backtest pour "prouver" un edge inexistant, on a **publié la défaite** et changé de produit. C'est cette transparence qui justifie nos analyses éditoriales.

## Pour aller plus loin

- López de Prado, *Advances in Financial Machine Learning*, Wiley 2018 (chap 7-8)
- Bailey, Borwein, López de Prado, Zhu, "The Probability of Backtest Overfitting", JCF 2017
- Bailey & López de Prado, "The Deflated Sharpe Ratio", JPM 2014

*Analyse algorithmique éducative. Pas un conseil en investissement.*
