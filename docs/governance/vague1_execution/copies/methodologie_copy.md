# Copy — Page Méthodologie

Page publique accessible sans auth depuis le footer. Objectif : substancier la promesse "honest confidence" en exposant la méthodologie scientifique.

---

## Header

**Eyebrow** : `MÉTHODOLOGIE`

**Titre H1** : Comment nous lisons le marché.

**Sub-titre** : M.I.A. Markets combine 5 méthodes scientifiques publiques pour décrire l'état du marché Or et FX, en mesurant explicitement son propre niveau d'incertitude.

---

## Section 1 — Le pipeline en 12 étapes

> **Pour chaque barre 15 minutes XAU/USD qui clôture, voici la chaîne de calcul exacte :**

1. **DataProvider** — récupère les 200 dernières barres OHLCV.
2. **Smart Money Engine** — calcule BOS (cassures de structure), FVG (zones de déséquilibre), OB (zones d'absorption), retests, fractales Williams.
3. **Volatility Forecaster** — modèle HAR-RV (Corsi 2009) avec multiplicateurs diurnaux + événementiels + régime HMM.
4. **Regime Gate** — Bayesian Online Changepoint Detection (Adams & MacKay 2007) + Bipower Variation pour les sauts (Barndorff-Nielsen 2004).
5. **News Analysis** — consultation du calendrier économique (Trading Economics API), agrégation sentiment 24h.
6. **Confluence Detector** — combine les 8 composantes en un score 0-100.
7. **Calibrated Conviction** — LightGBM → Isotonic Recalibration → Adaptive Conformal Inference (Gibbs & Candès 2021).
8. **Signal State Machine** — filtre hysteresis + cooldown + lifetime.
9. **LLM Narrative Engine** — génère le narratif en langue choisie (FR / EN / DE / ES) avec validation post-génération anti-vocabulaire interdit.
10. **Insight Assembler** — compose le payload final `InsightSignalV2`.
11. **Signal Store** — persiste en base pour audit + historique.
12. **Renderers + livraison** — webapp / Telegram / API B2B selon canal client.

---

## Section 2 — Les 8 composantes du score

| # | Composante | Poids max | Mesure |
|---|---|---|---|
| 1 | Smart Money | 25 % | Cassures de structure, FVG armé, OB présent |
| 2 | Volatilité | 15 % | Forecast HAR-RV vs ATR naïf |
| 3 | Multi-Timeframe | 12 % | Alignement H1 / H4 / D1 |
| 4 | Liquidité | 10 % | Zones d'absorption institutionnelles |
| 5 | Sessions | 8 % | Asian / London / NY Overlap / NY Afternoon |
| 6 | Régime | 15 % | HMM 3-état + posterior bayésien |
| 7 | Technical | 10 % | RSI, MACD, Bollinger |
| 8 | News | 5 % | Sentiment news + proximité events |

> **La conviction calibrée** combine ces 8 contributions via un LightGBM puis recalibre la probabilité avec une isotonic regression. Le résultat est ensuite enveloppé d'un intervalle de confiance par Adaptive Conformal Inference (couverture garantie distribution-free).

---

## Section 3 — Validation empirique

### Walk-forward 7 ans

- **Période** : 2019 - 2025
- **Données** : XAU/USD M15 (Dukascopy, 97,6 % coverage validé)
- **Méthode** : Combinatorial Purged Cross-Validation (López de Prado 2018)
- **Aucun look-ahead** : modèle entraîné uniquement sur le passé strict de chaque fenêtre

### Métriques agrégées — Statut 2026-05-27

> ⚠️ **Validation statistique OOS en cours**. Les métriques détaillées (PF, win rate, IC) sont **retirées de la publication** tant que le moteur de scoring n'a pas été validé out-of-sample. Voir `docs/governance/AUDIT_ALGO_2026_05_27.md`.

**Données disponibles** : 7 années de données XAU/USD M15 (Dukascopy, 98.4 % coverage). Pipeline complet 5 briques implémenté et testé (46 tests verts).

**Ce que nous AFFICHERONS** dès validation OOS (Brier skill > +2 % AND DSR > 1.0 AND PBO < 0.5) :
- Profit factor avec IC 95 % bootstrap (1000 itérations)
- Win rate observé
- Drawdown maximum
- Walk-forward CPCV k=5 folds purged (López de Prado 2018)
- PBO + DSR (Bailey-López de Prado 2014)

### Intervalle conformel — méthode

L'algorithme estime sa propre marge d'erreur via **Adaptive Conformal Inference** (Gibbs & Candès, NeurIPS 2021), garantie distribution-free.

Sur les outcomes historiques mémorisés, on calcule les résidus de non-conformité et leur quantile (1−α). L'intervalle est mis à jour en ligne pour maintenir la couverture nominale même sous régime drift.

**Validation OOS de la couverture** : en cours (Sprint 1). Chiffres publiables après validation.

---

## Section 4 — Limites assumées

### Ce que l'algorithme NE FAIT PAS

- ❌ Prédire le prix exact
- ❌ Garantir un gain
- ❌ Donner des recommandations d'achat ou de vente
- ❌ Couvrir tous les actifs (XAU et EUR uniquement en phase d'accès anticipé)
- ❌ Performer également dans tous les régimes (volatilité extrême = dégradation prévue)

### Ce que l'algorithme RECONNAÎT ne pas savoir

- L'algorithme expose son **intervalle conformel** explicitement à chaque lecture
- Les performances passées (walk-forward) sont **paper-trading uniquement**, pas un track-record live
- `edge_claim = False` : nous n'affirmons pas avoir prouvé un edge statistiquement significatif sur 12 mois live forward
- Nous publierons cet edge claim **uniquement** quand nos critères seront satisfaits : PF > 1.20 sur 12 mois live, DSR > 1.0, PBO < 0.5

---

## Section 5 — Sources académiques

Toutes nos méthodes sont publiques, citées, et vérifiables :

- **López de Prado (2018)** — *Advances in Financial Machine Learning*, Wiley
- **Corsi (2009)** — *A Simple Approximate Long-Memory Model of Realized Volatility*, J. Fin. Econometrics
- **Gibbs & Candès (2021)** — *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS
- **Barndorff-Nielsen & Shephard (2004)** — *Power and Bipower Variation*, J. Fin. Econometrics
- **Adams & MacKay (2007)** — *Bayesian Online Changepoint Detection*, arXiv:0710.3742
- **Angelopoulos & Bates (2024)** — *Conformal Prediction: A Gentle Introduction*

---

## Section 6 — Posture éthique

> **« Nous ne vous disons pas quoi faire — nous vous donnons les meilleurs outils pour décider. »**
>
> M.I.A. Markets n'est pas un prestataire de services d'investissement. Le chatbot **refuse** par construction de donner un ordre d'achat ou de vente. Cette posture est conforme au Règlement délégué (UE) 2024/2811 sur les finfluenceurs entrant en vigueur en mars 2026.
>
> L'utilisateur reste seul décideur, en pleine autonomie, à ses propres risques.

---

## Footer page

> Toute question méthodologique : `methodology@mia.markets`
> Audit indépendant : nos données et notebooks de backtest sont disponibles sur demande, sous accord de confidentialité, pour clients Strategist+ et Institutional.
