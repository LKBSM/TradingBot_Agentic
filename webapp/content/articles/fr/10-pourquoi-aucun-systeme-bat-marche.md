---
title: "Pourquoi quasi aucun système ne bat le marché — et que faire quand même"
slug: pourquoi-aucun-systeme-bat-marche
locale: fr
published_at: 2026-05-13
tags: [efficient-market, education, philosophy]
---

# Pourquoi quasi aucun système ne bat le marché

Soyons brutalement honnêtes : **plus de 95% des stratégies systématiques retail perdent de l'argent après coûts**, même celles qui montrent un beau backtest. Pourquoi ? Et que peut-on faire d'utile à la place ?

## Trois raisons fondamentales

### 1. L'efficience adaptative

Eugene Fama a posé l'hypothèse d'**efficience des marchés** en 1970. L'idée : le prix incorpore toute l'information publique disponible quasi instantanément. Les corrections au fil du temps (Andrew Lo 2004, "Adaptive Markets Hypothesis") ont nuancé : les marchés s'**adaptent**, et les anomalies sont temporaires. Un edge découvert aujourd'hui par 10 traders disparaît dans 10 ans quand 10 000 le connaissent.

### 2. Le bias de sélection

Vous voyez les success stories — Renaissance, Two Sigma, Citadel. Vous ne voyez pas les **dizaines de milliers de fonds quants disparus** depuis 1990. Et même les survivants ont des Sharpe nets de fees autour de 1.5-2.0, pas les Sharpe 5+ qu'un retail peut "obtenir" sur backtest.

### 3. Les coûts éclipsent souvent l'edge

Sur XAU M15 retail :
- spread typique : 0.3 pip = ~30 cents par lot
- slippage typique : 0.1 pip = ~10 cents
- swap nocturne : −2.50$ par lot sur position longue

Un edge brut de +0.1R par trade peut se transformer en **−0.05R après coûts**. Le backtest a omis ce détail.

## Que reste-t-il alors ?

Trois choses utiles **qui ne nécessitent pas d'edge prédictif** :

### A. Comprendre le contexte

Savoir lire le régime (`low_vol_trending` vs `high_vol_stress`), interpréter un COT extrême, anticiper une volatilité FOMC — c'est de la **culture financière**. Ça ne fait pas gagner d'argent direct, mais ça évite d'en perdre par ignorance.

### B. Gérer le risque

Un trader qui sait sizer ses positions, calculer un RR rationnel, et résister à l'overtrading aura **moins de drawdowns** que celui qui ignore ces mécaniques, même avec une stratégie médiocre.

### C. Transparence radicale

Notre approche Phase 2B repose sur **publier la vérité** : pas d'edge, pas de promesses, paper-trading curve en direct avec disclaimer. C'est l'antithèse des chaînes Telegram qui vendent "des signaux 90% gagnants".

## Le pari Smart Sentinel

Plutôt que de prétendre à un edge que nous n'avons pas démontré, nous avons fait un pari opposé :

- **Honnêteté radicale** comme produit principal,
- **Éducation contextuelle** (régimes, macro, sessions) comme valeur ajoutée,
- **Audit trail B2B** pour les brokers qui veulent justifier leurs signaux,
- **Transparence forensique** : DSR=0, PBO=0.5 publiés tels quels.

C'est un **pari commercial** qu'une frange du marché valorise la vérité plus que les promesses. Le temps le dira.

## Pour aller plus loin

- Lo, "The Adaptive Markets Hypothesis", JPM 2004
- Harvey, Liu, Zhu, "...and the Cross-Section of Expected Returns", RFS 2016
- Notre transparency page : [/fr/transparency](/fr/transparency)

## Conclusion

La phrase la plus dangereuse en finance est : **"cette fois c'est différent"**. La deuxième plus dangereuse est : **"j'ai trouvé un edge"**. La sagesse consiste à accepter qu'on n'en a probablement pas, et à construire son rapport au marché sur autre chose que des promesses de gains.

*Analyse algorithmique éducative. Pas un conseil en investissement.*
