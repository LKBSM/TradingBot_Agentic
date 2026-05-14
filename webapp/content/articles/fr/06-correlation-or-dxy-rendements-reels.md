---
title: "Corrélation or / DXY / rendements réels — la trinité macro à connaître"
slug: correlation-or-dxy-rendements-reels
locale: fr
published_at: 2026-05-13
tags: [macro, correlation, real-yields, education]
---

# Or, DXY et rendements réels — la trinité macro

L'or est un actif macro avant d'être un actif technique. Trois variables expliquent **80% de sa variance long terme** :

1. **DXY** (dollar index) — l'or est libellé en USD ; un dollar fort = or moins cher en autres devises = demande qui baisse.
2. **Rendements réels américains** (Treasury 10y - TIPS) — l'or ne paie pas de coupon ; quand le real yield monte, le coût d'opportunité de détenir de l'or augmente.
3. **Anticipations d'inflation** — l'or est historiquement vu comme couverture inflation, même si la corrélation est moins directe qu'on ne le pense.

## Les corrélations roulantes XAU

Notre `cross_asset_correlation.py` calcule en continu sur 30 jours :

| Paire | Corrélation typique 2019-2025 | Range |
|---|---|---|
| XAU vs DXY | −0.65 à −0.85 | [−0.95, −0.30] |
| XAU vs US10Y real | −0.55 à −0.80 | [−0.90, −0.10] |
| XAU vs SPX | +0.10 à +0.40 (faible) | [−0.30, +0.70] |
| XAU vs BTC | +0.20 à +0.50 (variable) | [−0.20, +0.85] |

**Ce qui frappe** : la corrélation XAU/SPX est instable. Pendant le COVID Mars 2020, les deux ont chuté simultanément (corrélation +0.85 sur 30j). En "risk-off classique", l'or monte et SPX baisse (corrélation −0.50). Le régime de marché change le signe.

## Pourquoi cela importe pour l'analyse

Une narrative "haussière XAU" n'a pas le même poids selon le contexte macro :

- **DXY baisse + real yields baissent** : double vent favorable, narrative forte.
- **DXY baisse + real yields montent** : tension, le marché doit choisir.
- **DXY monte + real yields montent** : double vent contraire, narrative haussière improbable.

Notre algorithme intègre ces corrélations dans le scoring contextuel (composant `news_macro_alignment`).

## Limites

1. **Corrélation roulante 30j** : trop lente pour capturer un événement comme FOMC qui rebascule le régime en 4h.
2. **Causalité ambiguë** : DXY n'est pas une cause indépendante — il est lui-même fonction des differentials de taux et de risk-off.
3. **Effets non-linéaires** : à partir d'un seuil (par exemple real yield > 2.5%), la sensibilité de l'or change discontinuement.

## Sources

- Federal Reserve Economic Data (FRED) — `DGS10`, `DFII10`, `T10YIE`
- World Gold Council — Gold Demand Trends quarterly
- Indexés dans notre RAG `[source:fred-dgs10]`, `[source:wgc-gdt]`

## Conclusion

Trader (ou analyser) l'or sans suivre DXY + real yields, c'est comme regarder un match sans connaître le score. Ces variables ne remplacent pas l'analyse technique mais l'**ancrent dans la réalité macro**.

*Analyse algorithmique éducative. Pas un conseil en investissement.*
