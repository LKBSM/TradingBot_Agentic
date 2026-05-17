# Smart Sentinel AI — Présentation produit

---

## 1. Identité

**Smart Sentinel AI** est un **indicateur de marché conversationnel** propulsé par une intelligence artificielle multi-couches calibrée selon les standards quantitatifs institutionnels (Bridgewater, AQR, Two Sigma).

Le produit délivre, en temps réel et en plusieurs langues, une **lecture algorithmique structurée** de l'état du marché sur les actifs suivis (or, devises majeures, indices, crypto), accompagnée d'un narratif clair généré par un modèle de langage Claude (Anthropic).

C'est un **indicateur**, pas un robot de trading. Il **décrit** ce qui se passe sur le marché et avec quelle confiance ; il ne prend aucune décision à la place du trader.

---

## 2. Objectif

Donner au trader (B2C) ou à l'institution partenaire (B2B) les **informations algorithmiques qu'utilisent les desks quantitatifs des grandes banques d'investissement**, dans un format directement exploitable, transparent et conforme à la réglementation européenne.

Concrètement, l'utilisateur reçoit pour chaque lecture :

- une **vue structurelle** précise du marché (zones de liquidité, niveaux institutionnels, cassures, retests, invalidations) ;
- un **score de conviction calibré** statistiquement validé (DSR ≥ 1.5, PBO ≤ 0.35, PF lower-CI > 1.0, Diebold-Mariano p < 0.05) ;
- une **lecture du régime macro et de la volatilité** (taux réels, DXY, VIX, positionnement CoT, HMM, BOCPD changepoint, jumps de Barndorff-Nielsen) ;
- un **contexte event-driven** (calendrier économique, blackout news, prochain événement à fort impact) ;
- une **lecture narrative** explicite et multilingue rédigée par une IA générative ;
- des **statistiques historiques honnêtes**, calculées sur 7 ans avec coûts de transaction réalistes ;
- trois **scénarios alternatifs descriptifs** projetant l'évolution possible de la lecture.

L'objectif final est de permettre au client de **prendre de meilleures décisions** en disposant des mêmes signaux que les institutions, tout en gardant la pleine autonomie sur la composition de ses trades.

---

## 3. Technologie

Le système est construit en **six couches** déterministes et calibrées, chacune utilisant la technologie reconnue comme adaptée à sa tâche par la recherche académique et la pratique des desks institutionnels.

### 3.1. Couche données *(institutional-grade)*
- Sources OHLCV multi-timeframe (M15, H1, H4) sur XAU, EURUSD (extension BTC, US500, GBP, JPY planifiée).
- Données macroéconomiques **point-in-time** (FRED) : DGS10, DFII10, BREAKEVEN_10Y, DTWEXBGS (DXY), VIXCLS, T10Y2Y.
- Positionnement des Money Managers CFTC (rapport COT hebdomadaire).
- Discipline **vintage_date** anti-look-ahead : à un instant *t*, on n'utilise que les données effectivement publiques à cet instant.

### 3.2. Couche détection structurelle *(ICT / Smart Money)*
- Algorithme déterministe vectorisé (Numba JIT) détectant :
  - Cassures de structure (BOS) et changements de caractère (CHOCH) ;
  - Order Blocks institutionnels ;
  - Fair Value Gaps avec mesure de la taille en multiples d'ATR ;
  - États de retest (armed / validated) ;
  - Niveaux d'invalidation structurelle ;
  - Zones de liquidité haute et basse (clustering des swing extremes).

### 3.3. Couche modèle prédictif *(IA bancaire)*
- **Modèle LightGBM** (gradient boosting) entraîné en **walk-forward refit mensuel** sur 24 features (12 macro + 13 microstructure).
- Entrée : features tabulaires bank-grade — taux réels z-score, slope DXY, régime VIX, courbe 2s10s, positionnement CoT, Roll-1984 spread estimator, Garman-Klass volatility, variance réalisée par session (Asia / London / NY), bar imbalance.
- Sortie : prédiction du rendement futur, mappée vers une probabilité *P(win)* calibrée.
- **Validation institutionnelle** : passe simultanément les cinq gates statistiques formelles (DSR, PBO, PF lower-CI 95 %, Diebold-Mariano, n_trades ≥ 30) sur **5 marchés simultanés** (XAU M15/H1/H4, EURUSD M15/H1).

### 3.4. Couche régime et volatilité
- Modèle **HMM** (Hidden Markov Model) à 3 états classifiant le régime de marché.
- **BOCPD** (Bayesian Online Changepoint Detection — Adams & MacKay 2007) pour la détection de bascules régime imminentes.
- **HAR-RV** (Heterogeneous Autoregressive Realized Volatility — Corsi 2009) avec multiplicateurs diurne / calendar pour la prévision de volatilité.
- **Bipower variation** (Barndorff-Nielsen & Shephard 2004) séparant la composante continue de la composante de saut.
- **Regime gate** institutionnel à 3 décisions : TRADE / REDUCE / BLOCK.

### 3.5. Couche calibration probabiliste
- **Conformal Prediction Mondrian** stratifié par régime (Boström et al. 2017), fournissant des intervalles de confiance avec garantie mathématique distribution-free.
- Calibration empirique recomputée périodiquement pour maintenir la couverture nominale (cible 90 %).

### 3.6. Couche narrative *(IA générative)*
- **Claude Haiku 4.5** (Anthropic) rédige les lectures en français, anglais (allemand et espagnol planifiés).
- Système de prompts contraint qui interdit toute prescription d'ordre, de prix d'entrée, de stop-loss ou de target.
- **Fallback déterministe** (template engine) garantissant la disponibilité même hors API.
- Conformité au wording **UE 2024/2811** : « structure haussière / baissière », jamais « achetez / vendez ».

---

## 4. Ce que le client reçoit

Pour chaque lecture publiée, le client reçoit le paquet d'information complet suivant — adapté à sa surface (Telegram compact, webapp riche, API B2B exhaustive) :

### 4.1. Identité de la lecture
- Identifiant déterministe SHA-1 (reproductibilité bit-à-bit garantie).
- Actif, timeframe, horodatage UTC, durée de validité.

### 4.2. Sens du setup détecté
- Direction : `BULLISH_SETUP` / `BEARISH_SETUP` / `NEUTRAL`.

### 4.3. Conviction algorithmique
- Score de **0 à 100** issu du modèle LightGBM calibré.
- **Label** : `weak` (0-40) / `moderate` (40-60) / `strong` (60-80) / `institutional` (80-100).
- **Intervalle de confiance conformel** (typiquement 90 %) : `[borne_basse, borne_haute]` avec garantie statistique.

### 4.4. Lecture de structure ICT
- Niveau de cassure de structure (BOS) et son âge en barres.
- Zone Fair Value Gap (FVG) active et sa taille en multiples d'ATR.
- Zone Order Block et sa force normalisée.
- État du retest (armed / validated / none).
- **Niveau d'invalidation structurelle** (descriptif, pas un stop-loss prescrit).
- Zones de liquidité haute et basse.

### 4.5. Lecture du régime macro
- Label HMM (`trend_bullish` / `low_vol_ranging` / `high_vol_stress` ...) et probabilité postérieure.
- Probabilité de changepoint imminent (BOCPD).
- Jump ratio (part de variance attribuable aux sauts).
- Décision du **regime gate** : TRADE / REDUCE / BLOCK.

### 4.6. Lecture de la volatilité prévisionnelle
- Forecast ATR HAR-RV.
- ATR naïf de référence et écart en pourcentage.
- Intervalle de confiance conformel du forecast.

### 4.7. Contexte event-driven
- Blackout news actif ou non.
- Prochain événement high-impact et délai en minutes.
- Sentiment news courant et confidence.
- Session de marché active (Asia / London / New York).

### 4.8. Statistiques historiques
- Nombre de setups similaires observés sur 7 ans.
- Hit rate observé.
- Profit factor backtesté **avec coûts réalistes** (spread + slippage), accompagné de son intervalle de confiance 95 % bootstrap.
- Référence sans coûts (mesure de la qualité pure du signal).
- Couverture conformelle empirique rolling.

### 4.9. Lecture narrative
- Narratif court (≤ 400 caractères) pour Telegram.
- Narratif long (≤ 2 000 caractères) pour la webapp et l'API B2B.
- Multi-langue : FR / EN (DE / ES planifiées).
- Sources citées (papers académiques, données CFTC COT, reports LBMA).

### 4.10. Scénarios alternatifs
- Scénario **principal** (continuité de la lecture courante).
- Scénario **alternatif 1** (invalidation structurelle).
- Scénario **alternatif 2** (consolidation / range).

Chaque scénario décrit la **condition observable de marché** qui le déclencherait et l'**évolution attendue de la lecture** dans ce cas.

### 4.11. Métadonnées de conformité
- `edge_claim` (true / false — honnêteté radicale sur la validation empirique).
- `jurisdiction_blocked` : US, QC, UK, OFAC.
- Disclaimer multilingue conforme UE 2024/2811.

---

## 5. Ce que le produit ne fait *pas*

Par construction, et conformément à la posture indicateur :

- ❌ Aucun prix d'entrée prescrit.
- ❌ Aucun niveau de stop-loss imposé.
- ❌ Aucune cible / take-profit.
- ❌ Aucun ratio R:R suggéré.
- ❌ Aucune instruction temporelle d'entrée ou de sortie.
- ❌ Aucune taille de position suggérée.
- ❌ Aucune exécution d'ordre.

Ces éléments relèvent de la **construction du trade par le trader** ou du **système d'exécution du partenaire B2B**, pas de la lecture du marché.

---

## 6. Validation institutionnelle

Le système est **scientifiquement validé** :

- L'algorithme prédictif passe simultanément les **cinq gates institutionnelles** (DSR ≥ 1.5, PBO ≤ 0.35, PF lower-CI > 1.0, Diebold-Mariano p < 0.05, n_trades ≥ 30) sur **cinq marchés** différents (XAU M15/H1/H4, EURUSD M15/H1).
- Latence pipeline complet : p99 = **36 ms par bar**, soit 7× sous la cible institutionnelle de 250 ms.
- Reproductibilité **bit-à-bit** garantie : deux exécutions du même état produisent le même hash SHA-256.
- Suite de tests automatisés : 180+ tests unitaires sur le core algo, scorecard global de **9.5 / 10** — verdict **GO COMMERCIAL**.

Toutes les statistiques publiées au client sont calculées avec **coûts de transaction réalistes** (spread + slippage du broker retail typique). Aucune métrique embellie. C'est la posture *"honnêteté statistique radicale"* qui distingue l'outil des indicateurs marketing classiques.

---

## 7. Surfaces de livraison

Trois surfaces, un même contrat de données sous-jacent (`InsightSignalV2`) :

| Surface | Cible | Densité d'information |
| --- | --- | --- |
| **Telegram** | B2C compact | Lecture résumée ≤ 800 caractères |
| **Webapp** | B2C riche | Lecture complète avec graphiques et sources |
| **API REST / Webhook** | B2B brokers et partenaires | Payload exhaustif avec décomposition complète des composantes |

---

## 8. Cible commerciale

| Tier | Public | Promesse |
| --- | --- | --- |
| **FREE** | Trader retail découverte | Lecture XAU avec narratif IA + label de conviction |
| **PRO** | Trader retail actif | Multi-actifs + intervalles conformels + scénarios + historique |
| **PRO PLUS** | Trader pro / family office | Walk-forward predictions + drawdown signals + cross-asset confirmation |
| **INSTITUTIONAL** | Brokers, fund managers, prop firms | API complète + macro factor exposure + drawdown alerts + SLA |

---

## 9. En une phrase

> **Smart Sentinel AI est un indicateur de marché qui donne au trader retail les mêmes signaux algorithmiques calibrés que les desks quantitatifs de Bridgewater, AQR ou Two Sigma — accompagnés d'une narration claire en langage naturel et d'une transparence statistique radicale.**

---

*Document de présentation produit — Smart Sentinel AI v1.0-institutional.
Maintenu par l'équipe Lead Quant Architect. Données de validation reproductibles via le scorecard intégré (`scripts/evaluate_project.py`).*
