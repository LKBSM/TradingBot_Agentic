# Éval multi-marchés — le moteur de détection généralise-t-il ? (2026-06-17)

**Branche** : `diagnostic/multi-market-eval` · **Périmètre** : moteur SMC (`SmartMoneyEngine` — OB / FVG / BOS / CHOCH / Retest) appliqué hors XAU/EUR
**Discipline** : **LECTURE SEULE**. Aucune modification de détection, de seuil ou du produit. Aucun marché ajouté au produit. Scripts jetables non commités ; seul ce rapport est versionné.

> ⚠️ **SANITY-CHECK, PAS validation de justesse.** Ce document répond à une seule question : « le moteur tourne-t-il et produit-il des comptes plausibles (ni 0, ni absurde) sur d'autres marchés ? ». Il ne dit **rien** de la justesse des zones détectées — ça reste l'annotation founder. Un compte « plausible » ici ≠ « zone correcte ».

---

## 0. Résumé exécutif

1. **Le moteur est entièrement scale-invariant.** Aucun seuil absolu (points/pips) nulle part : fractals = extrema locaux, FVG/Retest = multiples d'ATR, BOS/CHOCH = comparaison au niveau structurel fractal, OB = forme de bougie + force normalisée ATR. Vérifié empiriquement : le taux de FVG est comparable (~12–15/100 barres) de l'EURUSD (prix ~1,1) au JPY (~150), à l'or (~2 000) et au BTC (~100 000). **Le moteur transfère entre échelles sans recalibrage** (check A). 
2. **Une faille de transfert nette : FVG ≠ gap de session.** La détection FVG est le pattern ICT 3 bougies `low[i] > high[i-2]` (resp. baissier) — une imbalance **intra-session**. Elle **ne distingue pas** le gap d'ouverture/session. Sur les marchés à sessions (indices, actions), les gaps d'ouverture sont taggés comme des FVG : le taux de FVG explose ×2 à ×4 (GER40 M15 : **54,7/100** vs ~13/100 en FX) et **16 à 70 % des frontières de session produisent un FVG** (check B).
3. **Verdict par classe** : FX **PLAUSIBLE**, Métaux **PLAUSIBLE** (réf. XAU), Crypto **PLAUSIBLE**, Indices **SUSPECT** (conflation gap/FVG + quasi rien sur le plan), Actions **SUSPECT** (même conflation), Matières **NON ÉVALUABLE** (hors plan).
4. **Le moteur ne plante sur aucun marché récupéré** (12/16 marchés, 23 combos marché×TF OK, 0 erreur moteur). Les échecs sont **uniquement** de la couverture data (plan Twelve Data), pas du code.

---

## 1. Check A — Seuils RELATIFS vs ABSOLUS

Source : `src/environment/strategy_features.py`. **Tous les seuils sont relatifs ou structurels — aucun seuil absolu en unités de prix.**

| Détecteur | Logique | Type de seuil | Constante (défaut) | Réf. |
|---|---|---|---|---|
| Fractal (swing) | `high == rolling_max(2N+1)` / `low == rolling_min` | **Relatif** (extrema locaux, fenêtre 5 barres) | `FRACTAL_WINDOW=2` | `:698-730`, `:486-489` |
| FVG | `low[i] > high[i-2]` (haussier) / `high[i] < low[i-2]` (baissier), taille **normalisée ATR** | **Relatif (ATR)** | `FVG_THRESHOLD=0.1` (gap ≥ 0,1·ATR) | `:742-771`, `:491-494` |
| BOS / CHOCH | `close > current_high_structure` / `close < current_low_structure` (niveau = fractal) | **Structurel** (vs niveau fractal, pas de tolérance ni de point fixe) | — | `:113-143` |
| OB | Engulfing 2 bougies (`prev bearish & cur bullish & high>prev high`), force = `range/ATR` | **Relatif** (forme + force ATR) | `OB_FVG_BONUS=0.2`, `OB_REQUIRE_FVG=False` | `:848-907`, `:496-503` |
| Retest | `low <= level + k·ATR`, invalidation `close < level − k·ATR` | **Relatif (ATR)** | `RETEST_TOL_ATR=0.5`, `INVALID=1.0`, fenêtres 20/30 barres | `:313-324`, `:506-530` |

**Conclusion A** : le moteur a structurellement **toutes ses chances de transférer entre échelles de prix**. Confirmé empiriquement (§3) : taux de détection homogènes sur des prix variant de 1,1 à 100 000. La seule normalisation est l'ATR, qui absorbe l'échelle.

---

## 2. Check B — FVG vs GAP DE SESSION (point critique)

**Définition FVG dans le code** (`:742-751`) — pattern 3 bougies, imbalance intra-séquence :
```
Bullish FVG  ⟺  low[i]  > high[i-2]
Bearish FVG  ⟺  high[i] < low[i-2]
```
Le détecteur compare la barre `i` à la barre `i-2` (saute `i-1`). **Aucune notion de session, d'ouverture ni de `close→open`.** Rien dans le code ne teste le temps écoulé entre barres ni ne marque une frontière de session. Donc **un gap d'ouverture est, mécaniquement, un FVG** dès que `low[i] > high[i-2]` (ou symétrique) — ce qui est presque toujours vrai au franchissement d'un gap d'ouverture.

**Mesure empirique** (frontière de session = barre dont l'écart temporel au précédent > 2,5× l'écart médian) :

| Marché | TF | FVG /100 | Frontières détectées | FVG **sur** frontière | Lecture |
|---|---|---|---|---|---|
| FX / Métaux / Crypto (tous) | M15+H1 | 7–19 | **0** | **0** | marché continu : pas de gap, pas de conflation |
| GER40 (DAX) | M15 | **54,7** | 60 | **42** (70 % des frontières) | gaps de session taggés FVG |
| GER40 (DAX) | H1 | **36,2** | 70 | **45** (64 %) | idem |
| AAPL | H1 | 24,0 | 67 | 19 (28 %) | idem |
| TSLA | M15 | 17,9 | 18 | 8 (44 %) | idem |
| TSLA | H1 | 26,7 | 67 | 25 (37 %) | idem |

**Conclusion B (constat, pas correctif)** : sur les marchés à **gaps de session** (indices, actions), le moteur **tague les gaps d'ouverture comme des FVG**. Double effet :
- une **fraction directe** des FVG sont en réalité des gaps de session (28–70 % des frontières en génèrent un) ;
- le **taux global** de FVG est gonflé ×2 à ×4 vs les marchés continus, même hors frontières (les gaps fragmentent la structure de prix et multiplient les imbalances résiduelles).

Sur les marchés **continus 24 h/24** (FX, métaux, crypto), il n'y a pas de gap de session intra-semaine → **0 frontière détectée, 0 conflation** → le compteur FVG reste dans la plage de référence XAU/EUR. La faille est donc **spécifique aux classes à sessions**, pas un défaut général.

---

## 3. Check C — Sanité par marché (comptes sur 475 barres)

Comptes bruts (et /100 barres pour FVG). « breaks » = `BOS_EVENT≠0` (toutes cassures), « choch » = `CHOCH_SIGNAL≠0` (sous-ensemble : reversals), « OB » = barres avec `BULLISH/BEARISH_OB_HIGH` non nul. Tous les marchés listés **tournent sans erreur moteur**.

### FX majeurs — **PLAUSIBLE** ✅
| Marché | TF | FVG (/100) | breaks | choch | OB |
|---|---|---|---|---|---|
| GBPUSD | M15 | 72 (15,2) | 17 | 7 | 95 |
| GBPUSD | H1 | 60 (12,6) | 9 | 4 | 109 |
| USDJPY | M15 | 61 (12,8) | 3 | 2 | 129 |
| USDJPY | H1 | 35 (7,4) | 2 | 0 | 101 |
| AUDUSD | M15 | 62 (13,1) | 3 | 2 | 88 |
| AUDUSD | H1 | 56 (11,8) | 7 | 3 | 96 |
| USDCAD | M15 | 70 (14,7) | 20 | 6 | 98 |
| USDCAD | H1 | 69 (14,5) | 12 | 6 | 99 |
| **EURUSD (réf)** | M15 | 56 (11,8) | 13 | 6 | 117 |
| **EURUSD (réf)** | H1 | 62 (13,1) | 12 | 6 | 107 |

Comptes alignés sur la référence EURUSD/XAU. 0 frontière de session. USDJPY (prix ~150) ne se distingue pas des autres → **scale-invariance confirmée**.

### Métaux — **PLAUSIBLE** ✅ (référence)
| Marché | TF | FVG (/100) | breaks | choch | OB |
|---|---|---|---|---|---|
| XAUUSD (réf) | M15 | 64 (13,5) | 15 | 7 | 110 |
| XAUUSD (réf) | H1 | 69 (14,5) | 9 | 2 | 80 |
| XAGUSD | — | *non évaluable (plan)* | | | |

XAU = baseline déjà validée. XAGUSD (argent) partage la microstructure continue de l'or → **plausibilité attendue**, mais **non testée** (hors plan Twelve Data free).

### Crypto — **PLAUSIBLE** ✅
| Marché | TF | FVG (/100) | breaks | choch | OB |
|---|---|---|---|---|---|
| BTCUSD | M15 | 74 (15,6) | 20 | 9 | 130 |
| BTCUSD | H1 | 63 (13,3) | 21 | 6 | 120 |
| ETHUSD | M15 | 89 (18,7) | 21 | 11 | 128 |
| ETHUSD | H1 | 76 (16,0) | 11 | 1 | 116 |

24 h/24, 0 frontière. Taux FVG/breaks légèrement supérieurs (volatilité plus haute) mais **dans la plage**. Prix BTC ~100 000 sans effet sur les comptes → scale-invariance confirmée à l'autre bout de l'échelle.

### Indices — **SUSPECT** ⚠️
| Marché | TF | FVG (/100) | breaks | choch | OB | Statut |
|---|---|---|---|---|---|---|
| GER40 (DAX) | M15 | 260 (**54,7**) | 11 | 5 | 32 | gap/FVG conflation |
| GER40 (DAX) | H1 | 172 (**36,2**) | 16 | 7 | 96 | gap/FVG conflation |
| US30 (DJI) | — | — | | | | symbole invalide |
| NAS100 (IXIC) | — | — | | | | symbole invalide |
| SPX500 (SPX) | — | — | | | | hors plan |

Le moteur **tourne** sur GER40 mais la sortie FVG est **distordue** (×3–4, §2) par la conflation gap/FVG. Compteur OB anormalement bas en M15 (32) — autre signature de la fragmentation par gaps. **3 indices sur 4 ne sont même pas accessibles** sur le plan free → classe globalement non couverte + détecteur biaisé là où elle l'est.

### Actions — **SUSPECT** ⚠️
| Marché | TF | FVG (/100) | breaks | choch | OB | Statut |
|---|---|---|---|---|---|---|
| AAPL | M15 | — | | | | 429 (rate-limit transitoire) |
| AAPL | H1 | 114 (24,0) | 25 | 12 | 114 | gap/FVG conflation |
| TSLA | M15 | 85 (17,9) | 13 | 5 | 119 | gap/FVG conflation |
| TSLA | H1 | 127 (26,7) | 28 | 11 | 121 | gap/FVG conflation |

Tourne sans erreur, mais FVG gonflé ×2 et frontières de session taggées (§2). Les gaps overnight/week-end des actions sont structurels → la faille B s'applique systématiquement.

### Matières — **NON ÉVALUABLE**
USOIL (WTI/USD) : hors plan Twelve Data free. Microstructure proche d'un future quasi-continu → comportement probablement de type métal, mais **non testé**.

---

## 4. Couverture data Twelve Data (plan free) — à signaler

Sur 16 marchés × 2 TF (32 combos), **23 OK, 9 échecs** — tous **data, aucun moteur** :

| Marché | Cause | Couvert ? |
|---|---|---|
| XAGUSD, SPX500, USOIL | « available starting with Grow/Venture plan » | ❌ plan |
| US30 (DJI), NAS100 (IXIC) | « symbol invalid » (ticker non reconnu / hors plan) | ❌ symbole/plan |
| AAPL M15 | HTTP 429 (crédits/minute épuisés — throttle 8 s insuffisant ponctuellement) | ⚠️ transitoire (AAPL H1 a réussi) |

> **Rappel produit** : le provider de production (`twelve_data_provider.py:31-34`, `_map_symbol` `:171-178`) ne mappe que **XAUUSD et EURUSD** et **lève `ValueError` sur tout autre symbole**. Cette éval a contourné ce mapping dans un script jetable (frappe API directe en format natif `XXX/YYY`) — **sans toucher au produit**. Étendre le produit à d'autres marchés exigerait (hors-scope ici) : élargir `_SYMBOL_MAP`, un plan Twelve Data payant pour indices/argent/pétrole, et **traiter la faille FVG/gap (§2) avant tout marché à sessions**.

---

## 5. Verdict synthétique par classe d'actif

| Classe | Verdict | Justification |
|---|---|---|
| **FX majeurs** | 🟢 **PLAUSIBLE** | Comptes alignés sur EURUSD/XAU, 0 gap de session, scale-invariant (testé JPY ~150). |
| **Métaux** | 🟢 **PLAUSIBLE** | XAU = référence validée ; XAG même microstructure (non testé, hors plan). |
| **Crypto** | 🟢 **PLAUSIBLE** | 24/7, 0 gap, comptes dans la plage (BTC ~100 000 sans effet d'échelle). |
| **Indices** | 🟠 **SUSPECT** | Tourne, mais FVG ×3–4 par conflation gap de session (§2) ; 3/4 hors plan. |
| **Actions** | 🟠 **SUSPECT** | Tourne, mais FVG ×2 + frontières de session taggées (§2). |
| **Matières** | ⚪ **NON ÉVALUABLE** | USOIL hors plan free. |

**Aucun marché « CASSÉ »** au sens « plante / sort 0 / sort de l'absurde ». La distinction utile est : **continu (plausible) vs à sessions (suspect par conflation FVG/gap)**. La cause racine du « suspect » est **unique et identifiée** : la définition FVG ignore les gaps de session (check B).

---

## 6. Limites de cet exercice

- **Sanity-check, pas justesse** : des comptes « plausibles » ne prouvent pas que les zones sont correctes (réservé à l'annotation founder).
- **Échantillon court** : 475 barres/marché (quota free tier ménagé : ~23 requêtes utiles, throttle 8 s, ≪ 800/j).
- **Fenêtre récente unique** : pas de robustesse multi-régimes/multi-périodes.
- **Frontière de session = heuristique** (écart temporel > 2,5× médian) : approxime, ne définit pas, une session.
- **Couverture partielle** : argent, S&P, US30, NAS100, pétrole non testés (data, pas moteur).

*Fin du rapport — lecture seule, aucune modification de code.*
