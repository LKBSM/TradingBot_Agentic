# MIA Markets — Descriptive Quality Assessment

**Date** : 2026-05-27
**Auditeur** : Claude (descriptive-quality remit)
**Périmètre** : M.I.A. Markets en tant qu'**indicateur descriptif** (pas système de trading)
**Univers** : XAUUSD M15 + EURUSD M15
**Fenêtres** : TRAIN 2019-01-01 → 2023-12-31 · OOS 2024-01-01 → fin de fichier (XAU ~2026-04, EUR ~2025-12)
**Échantillons OOS** : XAU 54 996 bars · EUR 49 898 bars
**Données brutes reproductibles** : [`descriptive_quality_data.json`](descriptive_quality_data.json)
**Code de l'audit** : [`scripts/audit/descriptive_quality/`](../../scripts/audit/descriptive_quality/)

---

## Partie 1 — Méthodologie

### Question centrale
Quand l'algo annonce un événement (BOS, FVG, OB, regime, vol, jump, blackout, intervalle conformel), cet événement est-il **réellement présent**, **stable** sur la fenêtre annoncée, et — pour les claims probabilistes — **bien calibré** ?

Ce n'est **pas** la question de l'audit du 2026-05-27 (`AUDIT_ALGO_2026_05_27.md`) qui mesurait la rentabilité (PF 0.786, return −62 %). Cet audit-ci ignore le PnL.

### Trois questions par bloc

- **Q1 Justesse factuelle** — l'événement existe-t-il ? (sanity définitionnel + cross-method F1 contre une référence indépendante)
- **Q2 Stabilité temporelle** — l'info reste-t-elle valide sur la fenêtre annoncée (16 bars = 4h M15 pour SMC ; 30 bars pour ARMED retest) ?
- **Q3 Calibration** — les probas/intervalles ont-ils la couverture promise ?

### Échelle de verdict
| Statut | F1 | ECE | |PICP − nominal| |
|---|---|---|---|
| 🟢 | ≥ 0.85 | ≤ 0.05 | ≤ 2 pp |
| 🟡 | 0.65 – 0.85 | 0.05 – 0.10 | 2 – 5 pp |
| 🔴 | < 0.65 | > 0.10 | > 5 pp |

### Pondération par importance commerciale
| Bloc | Poids | Justification |
|---|---|---|
| BOS · FVG · OB · Retest | 3 | Cœur de la promesse SMC client |
| HMM · HAR-RV vol | 2 | Affirmations probabilistes exposées |
| BOCPD · Jump · Calendar | 1 | Contexte secondaire |
| Metadata | 0.5 | Sanity contrat |

### Stack technique de l'audit
- Engine prod : `SmartMoneyEngine` (config par défaut, FRACTAL_WINDOW=2, FVG_THRESHOLD=0.1, RETEST_TOL_ATR=0.5, etc.)
- Référence indépendante SMC : détecteur 3-barres swing (Tom DeMark) + FVG textbook sans seuil + OB à impulsion (≥0.8 × ATR)
- Stats : bootstrap CI 1 000 iter percentile, F1 multi-fenêtres (W=2/5/10), Welch t-test variance, Diebold-Mariano, ECE binné

### Hors-périmètre déclarés
Cf. [`OUT_OF_SCOPE.md`](OUT_OF_SCOPE.md). Sentiment news (RSS non archivés), intervalle conformel sur conviction (outcome = R-multiple → audit trading), sources RAG (Phase 2B), liquidity_zone_upper/lower (null en prod).

---

## Partie 2 — Évaluation par bloc

### 2.1 — BOS / break_level / event_age — poids 3

**Données** : XAU 1 790 events (1 024 bull / 766 bear) · EUR 1 693 events.
**Sortie** : [`data/bos_eval.json`](data/bos_eval.json)

| Métrique | XAU | EUR | Verdict |
|---|---|---|---|
| Q1.a Sanity close-vs-level | 1.000 | 1.000 | 🟢 |
| Q1.b Level reality (exact match ≤ 0.05×ATR, 500 bars) | 0.998 | 0.998 | 🟢 |
| Q1.c Cross-method F1 (W=5 bars) | 0.249 [0.240, 0.271] | 0.264 [0.255, 0.286] | 🔴 |
| Q1.c F1 (W=10 bars) | 0.315 | 0.342 | 🔴 |
| Q2 hold-rate mean @16 bars | 0.607 [0.591, 0.625] | 0.593 [0.575, 0.611] | — |
| Q2 médiane time-to-recross | 2 bars | 2 bars | — |
| Q2 no-opposite-BOS rate @16 bars | 0.792 | 0.741 | — |

**Lecture**
- La justesse définitionnelle est parfaite : 99.8 % des `bos_break_level` annoncés correspondent à un extrême OHLC réellement présent dans les 500 dernières barres. **L'événement BOS existe bel et bien**.
- Le F1 cross-method est faible (~0.25) parce que la définition prod (fractal Williams 2-barres) est plus stricte que la référence 3-barres : prod produit 1 790 events vs ref 3 985 (×2.2). C'est un **désaccord de définition entre deux implémentations SMC valides**, pas une fausse détection. À W=10 le F1 monte à 0.31-0.34.
- Stabilité : la médiane de recross du niveau cassé est de **2 barres**, et 60 % des clôtures suivantes respectent la direction sur 16 barres. Le BOS est donc un événement **tactique très court** — sa fenêtre `valid_until=4h` exposée au client est **trop longue** par rapport à la dynamique réelle.

**Verdict bloc** : 🟡 — la détection est descriptivement honnête ; la fenêtre de validité affichée est en revanche optimiste.

---

### 2.2 — CHOCH (Change of Character) — poids 3

**Données** : XAU 666 events · EUR 678 events.
**Sortie** : [`data/choch_eval.json`](data/choch_eval.json)

| Métrique | XAU | EUR | Verdict |
|---|---|---|---|
| Q1 Definitional sanity (CHOCH ≡ BOS_EVENT) | 1.000 | 1.000 | 🟢 |
| Q1 Reversal sanity (tendance précédente opposée) | 1.000 | 1.000 | 🟢 |
| Q1 Cross-method F1 (W=5) | 0.181  P=0.631 R=0.105 | 0.202  P=0.646 R=0.120 | 🔴 |
| Q2 hold-rate mean @16 bars | 0.604 [0.578, 0.630] | 0.614 [0.588, 0.643] | — |
| Q2 no-immediate-reversal rate | 0.815 | 0.785 | — |

**Lecture**
- Même structure que BOS : la définition prod est correcte ; le désaccord vient de la sensibilité au choix du swing-detector.
- **Précision = 0.63** : quand prod annonce un CHOCH, la référence détecte aussi un événement de même signe environ 63 % du temps. **Recall = 0.10** : la référence détecte 6× plus de CHOCHs (plus permissive).
- Stabilité honnête : 78-82 % des CHOCHs n'ont pas de retournement opposé dans les 16 barres suivantes — le claim "trend reversal" tient à court terme.

**Verdict bloc** : 🟡 — CHOCH est définitionnellement correct ; le F1 bas reflète la conservativité de prod, pas une erreur.

---

### 2.3 — FVG (Fair Value Gap) — poids 3

**Données** : XAU 8 732 events (taille médiane 0.36 × ATR) · EUR 8 835 events.
**Sortie** : [`data/fvg_eval.json`](data/fvg_eval.json)

| Métrique | XAU | EUR | Verdict |
|---|---|---|---|
| Q1 Inequality sanity | 0.9999 | 0.9998 | 🟢 |
| Q1 Threshold sanity (\|SIZE_NORM\| > 0.1) | 1.000 | 1.000 | 🟢 |
| Q1 Cross-method F1 (vs textbook sans seuil) | 0.875  P=1.000 R=0.777 | 0.884  P=1.000 R=0.792 | 🟢 |
| Q2 Mitigation rate @16 bars | 0.814 [0.806, 0.822] | 0.825 [0.816, 0.833] | — |
| Q2 Médiane time-to-mitigation | 1 bar | 1 bar | — |

**Lecture**
- Précision parfaite (P=1.00) : tout FVG annoncé par prod est un vrai gap 3-barres par construction. Le recall 0.78 reflète le filtrage par seuil `FVG_THRESHOLD=0.1×ATR` (les FVG plus petits ne sont pas exposés).
- **Stabilité paradoxale** : 81-83 % des FVG sont mitigés (price re-rentre dans le gap) dans les 16 barres, avec **médiane 1 bar**. La lecture "magnet zone" est cohérente, mais le claim "valid_until = 4h" est mécaniquement faux : à 0.36 × ATR médian, ces gaps sont des micro-niveaux comblés quasi-immédiatement.

**Verdict bloc** : 🟢 sur la détection. 🟡 sur l'horizon affiché.

---

### 2.4 — Order Block (OB) — poids 3

**Données** : XAU 13 810 events (force médiane 0.79 × ATR) · EUR 12 465 events.
**Sortie** : [`data/ob_eval.json`](data/ob_eval.json)

| Métrique | XAU | EUR | Verdict |
|---|---|---|---|
| Q1 Pattern sanity (engulfing + break) | 1.000 | 1.000 | 🟢 |
| Q1 Zone sanity (high > low) | 1.000 | 1.000 | 🟢 |
| Q1 Strength sanity ((h-l)/ATR ± FVG bonus) | 1.000 | 1.000 | 🟢 |
| Q1 Cross-method F1 (vs OB à impulsion ≥0.8×ATR) | 0.457  P=0.296 R=1.000 | 0.503  P=0.336 R=1.000 | 🔴 |
| Q2 Reaction-on-retest @16 bars | 0.772 [0.765, 0.779] | 0.771 [0.765, 0.779] | — |
| Q2 Never-touched rate | 0.071 | 0.077 | — |

**Lecture**
- Definition correcte. P=0.30 avec R=1.00 : **prod détecte tous les OB à impulsion stricts**, mais en plus ~3× plus d'OBs sans filtre d'impulsion. La référence (≥0.8 × ATR de corps) ne trouve que 4 091 events vs 13 810 pour prod. Conséquence : **un OB exposé au client peut être un simple engulfing de bruit**, pas un vrai déséquilibre institutionnel.
- Réaction au retest : 77 % des OBs voient le prix sortir de la zone en direction du OB dans les 16 barres. C'est **un bon support/résistance descriptif**, mais voir caveat ci-dessus.

**Verdict bloc** : 🟡 — pattern OK, sélectivité faible (claim "institutional supply/demand" affaibli par l'absence de filtre d'impulsion).

---

### 2.5 — Retest State Machine — poids 3

**Données** : XAU 1 790 BOS suivis · EUR 1 693 BOS suivis.
**Sortie** : [`data/retest_eval.json`](data/retest_eval.json)

| Métrique | XAU | EUR | Verdict |
|---|---|---|---|
| Q1 Definitional sanity (touch ≤ 0.5×ATR sur transition AWAITING→ARMED) | 0.960 | 0.958 | 🟡 |
| Q1 Conversion rate (BOS → ARMED) | 0.950 | 0.950 | 🟢 |
| Q1 Médiane time-to-arm | 1 bar | 1 bar | — |
| Q2 ARMED outcome @30 bars : continuation | 0.598 [0.576, 0.622] | 0.575 [0.551, 0.598] | — |
| Q2 ARMED outcome : invalidation | 0.351 | 0.370 | — |
| Q2 Density ARMED bars / total | 30.8 % | 30.6 % | — |

**Lecture**
- 95 % des BOS atteignent l'état ARMED — la machine d'état fonctionne. La sanity 96 % (vs 100 % attendu) provient de cas-limites numériques (ATR NaN sur barre de transition).
- **30 % des barres OOS ont une signature ARMED active**. C'est très dense — pas un signal rare. Cohérent avec time-to-arm médian = 1 bar (le retest se fait immédiatement, pas une vraie consolidation).
- Sur 30 barres, le déséquilibre 60/35 (continuation vs invalidation) est légèrement biaisé positif mais 35 % d'invalidations = signal bruité.

**Verdict bloc** : 🟢 sur le fonctionnement, 🟡 sur l'utilité pratique (la state machine fonctionne mais déclenche trop souvent pour servir de filtre fort).

---

### 2.6 — Calendar / blackout — poids 1

**Données** : 249 events HIGH USD en fenêtre OOS commune.
**Sortie** : [`data/calendar_eval.json`](data/calendar_eval.json)

| Métrique | XAU | EUR | Verdict |
|---|---|---|---|
| Q1 Structural sanity (0 dups, 0 NaN) | passed | passed | 🟢 |
| Q1 Coverage OOS bars (any-ccy ±30 min) | 1.51 % | 1.68 % | — |
| Q1 Vol elevation ratio med\|r\|_blocked / med\|r\|_free | 1.115 [1.034, 1.238] | **1.707** [1.558, 1.864] | 🟢 EUR / 🟡 XAU |

**Note doc/code** : la documentation client mentionne "blackout 30 min avant / 60 min après" ; le code emploie 30/30 (`news_analysis_agent.py:112-113`). **Incohérence doc à corriger**.

**Lecture**
- Calendrier structurellement clean. Coverage faible (~1.5 % des bars) cohérent avec ~250 events × 60 min sur 1.4M minutes OOS.
- **Le blackout sélectionne effectivement des bars plus volatiles** : ratio 1.11 (XAU, marginalement significatif) et **1.71 (EUR)**. Sur EUR/USD le filtre est utile ; sur XAU l'effet est faible car l'or est multi-driver (USD + risk-off + Asia retail).

**Verdict bloc** : 🟢 EUR · 🟡 XAU.

---

### 2.7 — Metadata sanity — poids 0.5

**Sortie** : [`data/metadata_eval.json`](data/metadata_eval.json)

Tous les checks passent pour XAU et EUR :
- ATR strictement positif (54 996 / 49 898 bars)
- BOS_EVENT ↔ BOS_BREAK_LEVEL consistency (0 mismatch)
- FVG_DIR ∈ {-1, 0, 1} · CHOCH_SIGNAL ∈ {-1, 0, 1}
- OB zone high > low (0 inversion)
- Échelle de prix cohérente avec `price_decimals`
- Espacement bar = 15 min strict intra-session

**Verdict bloc** : 🟢.

---

### 2.8 — HMM regime (low_vol_trending / low_vol_ranging / high_vol_stress) — poids 2

**Données** : entraîné sur 117k+ returns train, prédit sur 54 995 / 49 897 returns OOS.
**Sortie** : [`data/hmm_eval.json`](data/hmm_eval.json)

| Métrique | XAU | EUR | Verdict |
|---|---|---|---|
| Q1 Stress label sanity (var ratio stress/non-stress) | **6.48** | **8.51** | 🟢 |
| Q1 Trending label sanity (drift ratio t/r) | 1.64 | **1.00** | 🟡 / 🔴 |
| Q2 Flicker rate (state change par bar) | 0.218 | **0.994** | 🟡 / 🔴 |
| Q2 Médiane dwell time, low_vol_trending | 1 bar | 1 bar | — |
| Q2 Médiane dwell time, low_vol_ranging | 4 bars | 1 bar | — |
| Q2 Médiane dwell time, high_vol_stress | 1 bar | 1 bar | — |
| Q3 ECE posterior calibration | **0.539** | **0.182** | 🔴 |

**Lecture**
- Le label `high_vol_stress` est descriptif : la variance forward est **6.5× à 8.5×** plus élevée sur ces barres que sur les autres. Bien.
- Sur EUR, **le label `trending` n'a aucun pouvoir discriminant de drift** vs `ranging` (ratio 1.00). Sur XAU c'est marginal (1.64).
- **Flicker EUR = 99.4 %** : le HMM change d'état presque à chaque barre, c'est-à-dire qu'il n'y a aucune persistance. Le claim "le marché EUR est en régime X" n'a aucun sens stable. Sur XAU c'est ~22 % mais les régimes `trending` et `stress` ne persistent qu'**1 barre médiane**.
- **Calibration du posterior : catastrophique sur XAU**. À confiance ~0.99, le label est **correct dans seulement 41.7 % des cas** (ECE 0.54). Le posterior **trompe le client** : un score 99 % ressemble à une certitude, c'est en réalité un coin-flip.

**Verdict bloc** : 🔴 — exposer `hmm_label` + `hmm_posterior` au client tels quels est **descriptivement non-supportable**. Seul `high_vol_stress` (variance discriminée) tient.

---

### 2.9 — BOCPD changepoint probability — poids 1

**Sortie** : [`data/bocpd_eval.json`](data/bocpd_eval.json)

| Métrique | XAU | EUR | Verdict |
|---|---|---|---|
| Q1 Distribution cp_prob (q95 / q99 / max) | 0.0007 / 0.0014 / 0.0036 | 0.0007 / 0.0014 / 0.0036 | 🔴 |
| Q1 Bars avec cp_prob ≥ 0.5 | **0 / 54 995** | **0 / 49 897** | 🔴 |
| Variance shifts détectés indépendamment (Welch t-test) | 5 395 | 7 776 | — |
| Q3 ECE | 0.268 | 0.366 | 🔴 |

**Lecture**
- Le `cp_prob` est plaqué à la valeur prior (~1/240 = 0.00417). **Jamais une seule barre OOS n'atteint 0.5**. Pourtant 5 395 (XAU) / 7 776 (EUR) barres ont un changement de variance significatif à p < 0.01.
- Le détecteur **est essentiellement constant**, indépendamment des données. Hyperparamètres (priors NIG, hazard) ou code à investiguer — mais en l'état, **`bocpd_changepoint_prob` n'informe rien au client**.

**Verdict bloc** : 🔴 — retirer ou recalibrer avant exposition client.

---

### 2.10 — Jump ratio (bipower variation) — poids 1

**Sortie** : [`data/jump_eval.json`](data/jump_eval.json)

| Métrique | XAU | EUR | Verdict |
|---|---|---|---|
| Q1 Unit-interval | True | True | 🟢 |
| Q1 Distribution (médiane / q95 / q99) | 0.061 / 0.308 / 0.434 | 0.054 / 0.367 / 0.634 | — |
| Q1 Extreme-return alignment (top-5 % jump_ratio contient un \|r\| top-1 %) | **0.839** | **0.998** | 🟢 |
| Q2 Autocorr lag-1 | 0.969 | 0.974 | — |
| Q2 Autocorr lag-10 | 0.860 | 0.869 | — |

**Lecture**
- Le `jump_ratio` est descriptivement honnête : quand il est élevé, il y a effectivement un return extrême dans la fenêtre 96-bars (84 % XAU, 100 % EUR).
- Très autocorrélé (≈0.97 à lag-1) — par construction (statistique roulante 96 bars). Ce n'est **pas un détecteur d'événement temps-réel** mais une **signature de période**.

**Verdict bloc** : 🟢 — sain. Le seul caveat est l'horizon : exposer `jump_ratio` comme "il y a un jump maintenant" est trompeur ; "cette fenêtre a contenu un jump" est correct.

---

### 2.11 — HAR-RV forecast + conformal CI — poids 2

**Sortie** : [`data/volatility_eval.json`](data/volatility_eval.json)
**Échantillon** : 800 barres OOS par instrument (52 ms/bar sur XAU, 45 ms/bar sur EUR — cohérent avec eval_04 sur la latence)

| Métrique | XAU | EUR | Verdict |
|---|---|---|---|
| Q1 RMSE HAR (blended) vs naive ATR_14 | 4.589 vs 4.328 (**-6 % pire**) | 0.0022 vs 0.0004 (**-527 % pire**) | 🔴 |
| Q1 R² HAR | 0.626 | **-22.84** | 🔴 |
| Q1 R² naive | 0.668 | 0.393 | — |
| Q1 Diebold-Mariano stat | +0.576  p(HAR better)=0.72 | +29.31  p(HAR better)=1.00 | 🔴 |
| Q3 PICP empirique (nominal 0.95) | **0.513** [0.477, 0.546] | **0.386** [0.351, 0.420] | 🔴 |
| Q3 MPIW absolu | 2.56 | 0.0004 | — |
| Q3 MPIW relatif (vs ATR moyen) | 0.41 | 0.60 | — |

**Lecture**
- **Sur XAU, HAR est 6 % moins bon que naive ATR_14 en OOS**. Diebold-Mariano non significatif (p=0.72) : indistinguable statistiquement, ou très légèrement pire.
- **Sur EUR, HAR est catastrophiquement pire que naive** (RMSE × 5, R² −22.8, DM p=1.00). La calibration walk-forward du blend_weight (=0.30) reflète déjà cette défaillance ("improvement = -891.5 %" dans les logs).
- **L'intervalle conformel sous-couvre massivement le nominal 95 % : 51 % (XAU) et 39 % (EUR)**. C'est-à-dire que sur 100 bars, l'intervalle exposé au client **rate** la vraie volatilité 49-61 fois alors que la promesse est de la rater 5 fois.
- L'écart |PICP − nominal| = 0.44 (XAU) / 0.56 (EUR) est **bien au-delà du seuil 🔴 (5 pp)**.

**Verdict bloc** : 🔴 — la prévision HAR-RV **n'améliore pas** la baseline naive, et l'intervalle conformel **ne respecte pas sa promesse de couverture**. Le mode prod (`VOL_MODE=har`) ne devrait pas être affiché comme "supérieur à ATR" et les bornes `confidence_interval` ne devraient pas porter le label "95 %".

---

## Partie 3 — Synthèse catégorique

### Score pondéré

| Bloc | Poids | Verdict | Score (1.0 = 🟢, 0.5 = 🟡, 0.0 = 🔴) |
|---|---|---|---|
| BOS | 3 | 🟡 | 0.5 |
| CHOCH | 3 | 🟡 | 0.5 |
| FVG | 3 | 🟢 (détection) / 🟡 (horizon) | 0.75 |
| OB | 3 | 🟡 | 0.5 |
| Retest | 3 | 🟢/🟡 | 0.75 |
| Calendar | 1 | 🟢 EUR / 🟡 XAU (moyen) | 0.75 |
| Metadata | 0.5 | 🟢 | 1.0 |
| HMM | 2 | 🔴 | 0.0 |
| BOCPD | 1 | 🔴 | 0.0 |
| Jump | 1 | 🟢 | 1.0 |
| HAR-RV + conformal | 2 | 🔴 | 0.0 |
| **Total** | **22.5** | | **9.25 / 22.5 = 41 %** |

### Score qualitatif global : **🟡 (mitigé)**
La couche **SMC structurelle** (BOS / CHOCH / FVG / OB / Retest) est descriptivement honnête : les événements existent, les niveaux sont réels, les patterns sont satisfaits par construction. Les divergences cross-method viennent de choix de définition légitimes, pas d'erreur. **Mais les horizons de validité affichés au client (`valid_until = 4h`) sont systématiquement plus longs que la stabilité observée (médiane 1-2 bars).**

La couche **statistique** (HMM, BOCPD, HAR-RV + conformal) est **non-supportable en l'état** :
- HMM expose des posteriors trompeurs (XAU ECE 0.54 — 99 % de confiance pour 42 % d'accuracy)
- BOCPD ne sort jamais de son prior (0 changepoint flag sur 105k barres OOS)
- HAR-RV est pire que naive ATR sur les deux instruments, et son intervalle 95 % couvre seulement 39-51 %

### Sentiment et conformal-on-conviction
Hors-périmètre déclarés (cf. [`OUT_OF_SCOPE.md`](OUT_OF_SCOPE.md)). **Recommandation** : retirer du visible client tant que non-supportés (cf. Partie 4).

---

## Partie 4 — Implications commerciales

### Ce qui peut être affiché honnêtement aujourd'hui

| Champ | OK à exposer ? | Conditions |
|---|---|---|
| BOS_EVENT + BOS_BREAK_LEVEL | ✅ | Documenter "événement tactique court terme, validité ~2-4 bars" — pas 4h |
| CHOCH_SIGNAL | ✅ | Idem |
| FVG zone + size_atr | ✅ | Préciser "magnet zone — comblement médian 1 bar" |
| OB zone + strength | ⚠️ | Préciser "engulfing pattern, pas filtré par impulsion" |
| Retest state (ARMED / VALIDATED) | ✅ | OK mais signaler densité ~30 % des bars (pas un signal rare) |
| Calendar blackout flag | ✅ | Corriger doc 30/60 → **30/30** |
| Metadata (timeframe, decimals, asset) | ✅ | RAS |
| Jump ratio | ✅ | Préciser "signature 24h, pas alerte temps-réel" |

### Ce qu'il faut **retirer ou minimiser** dans la copie client immédiatement

| Champ | Action | Raison |
|---|---|---|
| `hmm_posterior` | **RETIRER** de la webapp B2C | ECE 0.54 — trompe le client |
| `hmm_label` (autre que `high_vol_stress`) | **MINIMISER** | trending/ranging non discriminé sur EUR |
| `bocpd_changepoint_prob` | **RETIRER** | jamais > 0.5 sur 105k bars, valeur sans information |
| `volatility_readout.confidence_interval` avec label "95 %" | **RETIRER ou renommer** | couverture réelle 39-51 % |
| `volatility_readout.forecast_vs_naive_pct` | **NE PAS afficher comme avantage** | HAR ≤ naive en OOS |
| `valid_until = 4h` sur BOS/CHOCH/FVG/OB | **RÉDUIRE à 30-60 min** | médiane invalidation = 1-2 bars |
| Conformal-on-conviction (cf. OUT_OF_SCOPE.md §2) | **RETIRER** déjà recommandé | outcome = R-multiple → audit trading, pas descriptif |

### Cohérence avec le pivot du 2026-05-27
Cette synthèse est **cohérente avec la décision pivot** (cf. mémoire `pivot_positioning_2026_05_27`) : tout claim non validé doit être retiré du visible client. L'audit ajoute des éléments concrets à retirer (HMM posterior, BOCPD, conformal 95 % sur vol) que la décision originale n'avait pas identifiés explicitement.

### Cohérence avec l'audit algo du 2026-05-27
Pas de chevauchement direct (cet audit ignore PnL/PF). Mais les findings convergent : **l'algo voit honnêtement la structure SMC**, mais **la couche probabiliste / forecast n'a pas de pouvoir prédictif fiable**. La même conclusion sous deux angles.

### Plan d'action immédiat (sans dev infra)
1. **Webapp/Telegram** : retirer 4 champs (hmm_posterior, bocpd_cp, vol_CI_95%, conformal_conviction) ou les passer en mode "expert read-only sans label de confiance"
2. **Documentation** : aligner blackout 30/30, valid_until BOS/CHOCH 30-60 min, "OB = engulfing pattern" sans "institutional"
3. **Backend** : conserver tous les calculs côté serveur (audit interne / re-évaluation future) — seule la couche d'exposition change

### Plan de re-évaluation (post-fix algo)
À ré-évaluer si Sprint 1 livre :
- Recalibration BOCPD (priors + hazard)
- Re-tuning HAR-RV blend ou bascule sur LightGBM (eval_04 montre déjà LGBM bat naïf -31 % RMSE)
- Recalibration TCP residuals avec α = 0.05 effectif (vs comportement actuel sous-couvrant)

Date de re-audit conditionnelle : **2026-08-27** (3 mois) si Sprint 1 est livré.

---

## Partie 5 — Verdict final

> **MIA Markets, en tant qu'indicateur descriptif, voit honnêtement la structure SMC (BOS/CHOCH/FVG/OB/Retest) sur XAU et EUR M15. Mais la couche probabiliste qu'il expose (`hmm_posterior`, `bocpd_changepoint_prob`, `volatility_confidence_interval` à 95 %) sur-promet systématiquement — ECE 0.18-0.54, couverture conformelle 39-51 % vs 95 % annoncés, et le forecast HAR-RV n'améliore pas la baseline naïve.**
>
> **Recommandation** : conserver la couche SMC visible avec horizons raccourcis. **Retirer immédiatement** les 4 champs probabilistes trompeurs (HMM posterior, BOCPD cp_prob, intervalle conformel 95 % sur vol, conformal sur conviction). Re-évaluer dans 3 mois après corrections.

**Score global pondéré** : 41 % (9.25 / 22.5).
**Score qualitatif** : 🟡 (mitigé — descriptivement honnête sur SMC, non-supportable sur la couche stat).

---

*Audit conforme à la mission présentée dans [`MISSION_ACK_DESCRIPTIVE.md`](MISSION_ACK_DESCRIPTIVE.md). Données reproductibles : [`descriptive_quality_data.json`](descriptive_quality_data.json). Hors-périmètre : [`OUT_OF_SCOPE.md`](OUT_OF_SCOPE.md).*
