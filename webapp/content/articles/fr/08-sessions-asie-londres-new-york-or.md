---
title: "Sessions Asie / Londres / New York — diurnal de l'or M15"
slug: sessions-asie-londres-new-york-or
locale: fr
published_at: 2026-05-13
tags: [diurnal, xau, sessions, education]
---

# Sessions Asie / Londres / New York sur l'or

Le marché XAU/USD est ouvert 23h/24, du dimanche 23h ET au vendredi 17h ET. Mais **la liquidité n'est pas répartie uniformément** — trois sessions dominent.

## Les trois sessions

| Session | Horaire UTC | Volume relatif XAU |
|---|---|---|
| Asie (Tokyo, Singapour) | 23h-08h | 18% |
| Londres | 08h-12h | 32% |
| Overlap LDN/NY | 12h-17h | 38% |
| New York seule | 17h-22h | 12% |

> Source : agrégat Dukascopy tick volume 2019-2024.

## Stylized facts diurnal

Notre `stylized_facts.hourly_facts()` produit la table suivante sur XAU M15 (6 ans) :

| Heure UTC | Vol relative | Hit rate up |
|---|---|---|
| 02h | 0.45× | 49% |
| 08h (Londres open) | 1.20× | 51% |
| 12h (NY pre-open) | 1.35× | 50% |
| 14h (NY open) | 1.60× | 48% |
| 14h30 (data releases) | 2.10× | 47% |
| 17h (LDN close) | 1.10× | 50% |
| 22h | 0.50× | 50% |

**Constat 1** : la volatilité forme une **double bosse** — un pic à 8h UTC (Londres open) et un second plus marqué à 14h-14h30 UTC (NY open + data).

**Constat 2** : le **hit rate directionnel est plat** à ~50% sur toutes les heures. La volatilité change ; la direction reste imprévisible.

## Implications narratives

1. **Sessions = qualité du setup** : un BOS détecté à 03h UTC en pleine Asie est statistiquement moins informatif qu'un BOS à 14h UTC pendant l'overlap.
2. **Stop sizing horaire** : un stop ATR fixe est trop large à 02h (sur-protection) et trop serré à 14h30 (risque de wick-out).
3. **Liquidity sweeps** : ils se concentrent autour des **opens** (Londres et NY). 60% des sweeps observés se produisent dans la fenêtre +/- 30min du open.

## Day-of-week effect

| Jour | Vol relative |
|---|---|
| Lundi (calme) | 0.85× |
| Mardi | 1.05× |
| **Mercredi (FOMC moyen)** | **1.40×** |
| Jeudi (NFP semaine) | 1.20× |
| Vendredi (close) | 0.95× |

Le **mercredi tire** statistiquement la vol — explicable par les FOMC qui tombent toujours mardi/mercredi.

## Limites

- L'agrégat hourly ne capte **pas** les events particuliers (NFP first-Friday, FOMC). Pour ces dates, il faut conditionner.
- Le tick volume Dukascopy n'est pas le volume CME or — l'aggregate est cohérent mais pas exhaustif.
- 6 ans n'est pas une éternité ; un nouveau régime macro (par exemple récession profonde) peut changer la distribution.

## Conclusion

Le diurnal de l'or n'est pas une feature exotique — c'est un **fait empirique** que toute stratégie systématique doit intégrer, ne serait-ce que pour ne pas trader en environnement à coût de transaction défavorable.

*Analyse algorithmique éducative. Pas un conseil en investissement.*
