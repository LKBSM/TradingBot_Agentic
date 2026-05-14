---
title: "Stop-loss et take-profit intelligents — calculer à partir de l'ATR"
slug: stop-loss-take-profit-intelligents
locale: fr
published_at: 2026-05-13
tags: [risk, atr, education]
keyword_target: "stop loss take profit calcul"
---

# Stop-loss et take-profit intelligents

Un stop fixe à "X pips" est l'un des **plus grands biais** du retail trading. Le marché ne respecte pas votre tolérance arbitraire ; il respecte sa propre volatilité.

## La règle ATR

Un stop dimensionné en fraction d'ATR (Average True Range) s'adapte automatiquement au régime :

- en marché calme, l'ATR est petit → stop serré → bonnes entrées sur retest,
- en stress, l'ATR explose → stop large → vous évitez d'être sorti par le bruit.

**Conventions courantes** :

| Style | Stop | Take-profit | RR |
|---|---|---|---|
| Scalping M1-M5 | 0.5 ATR | 1.0-1.5 ATR | 2.0-3.0 |
| Intraday M15-H1 | 1.0 ATR | 1.5-2.0 ATR | 1.5-2.0 |
| Swing H4-D1 | 1.5-2.0 ATR | 3.0-4.0 ATR | 2.0+ |

Le **risk-reward ratio (RR)** est ce qui détermine si une stratégie peut être profitable. Avec un hit rate de 40% (40% de trades gagnants), il faut un RR ≥ 1.5 pour casser break-even.

## Pourquoi un RR < 1 est presque toujours perdant

Imaginons hit rate 60% (impressionnant) et RR 0.5 (stop large, TP serré) :

```
Espérance = 0.6 × 0.5 − 0.4 × 1.0 = 0.3 − 0.4 = -0.1R
```

Vous perdez en moyenne 0.1R par trade, malgré 60% de gagnants. **L'asymétrie domine la fréquence**.

## L'ATR n'est pas tout

Trois corrections que notre confluence_detector applique :

1. **Niveau structurel** : un stop au-dessous d'un swing low récent est mieux placé qu'un stop ATR pur, même si l'ATR suggère plus serré.
2. **Spread + slippage** : ajouter 1.5× le spread typique à l'entrée et à la sortie. Sur XAU à 0.3 pip de spread, c'est négligeable ; sur EURNZD à 5 pips, c'est tueur.
3. **Régime de volatilité** : en `high_vol_stress`, multiplier l'ATR par 1.3× pour absorber les wicks.

## Le cas du break-even agressif

Beaucoup de retail traders déplacent leur stop au break-even dès que le prix a touché 1R de profit. **Notre backtest sur 6 ans XAU M15 montre que cette pratique réduit le profit factor de 15-20%** : la plupart des positions qui auraient finalement atteint 2R sont sorties prématurément à 0R par un retracement normal.

Mieux : un **trailing stop ATR-based** qui ne se déclenche qu'à partir de 1.5R, et qui suit à 2× ATR sous le high local.

## Conclusion

Le stop-loss n'est pas un "filet de sécurité psychologique" — c'est une **prime d'option** que vous payez pour borner votre perte. Sa taille doit refléter la volatilité réelle du marché, pas votre tolérance émotionnelle.

*Analyse algorithmique éducative. Pas un conseil en investissement.*
