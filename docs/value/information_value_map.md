# Smart Sentinel AI — Information Value Map

> **Statut** : Partie A livrée (inventaire). Parties B (justification), C (différenciation), D (synthèse stratégique) — en attente de validation du périmètre Partie A par l'utilisateur.
>
> **Méthode** : inventaire dressé par lecture directe du code de `src/intelligence/`, `src/api/insight_signal_v2.py`, `src/api/models.py`, `src/api/routes/signals.py`, et des mockups B2B/B2C dans `mockups/`. Aucune supposition sur ce que l'algo « devrait » produire — seulement ce qu'il produit *effectivement*.
>
> **Date de cartographie** : 2026-05-16 — branche `institutional-overhaul`.

---

## 0. Légende des surfaces d'exposition

L'information produite par l'algo peut être exposée au client via trois surfaces distinctes, dont seules les deux premières sont en production :

| Code | Surface | Contrat | Statut | Référence code |
|---|---|---|---|---|
| **v1-REST** | API `/api/v1/signals/current` (et `/history`) | `SignalResponse` Pydantic (8 champs) | En prod | `src/api/models.py:51`, `src/api/routes/signals.py:18` |
| **v2-unified** | « Lingua franca » B2C Telegram + webapp + email + B2B REST + webhook + audit | `InsightSignalV2` Pydantic | Contrat cible (Sprint UX-1.1) | `src/api/insight_signal_v2.py:191` |
| **B2B-mock** | Maquette aspirationnelle endpoint broker `/api/v1/insights/latest` | JSON ad hoc (non typé) | Mockup uniquement | `mockups/b2b_insight.json` |
| **INTERNE** | — | — | Calculée par le pipeline mais jamais exposée à aucune surface client à ce jour | Voir colonne « Étage » |

Toutes les fiches ci-dessous indiquent dans la colonne **Forme exposée** quel contrat porte l'information, ou explicitement `🔒 INTERNE` quand elle ne sort pas du processus.

---

## Partie A — Inventaire des informations produites par l'indicateur

L'inventaire est groupé par **catégorie sémantique** (ce que l'info représente pour le trader), pas par étage du pipeline (vue technique). La traçabilité étage→info est dans chaque fiche.

### Catégorie A.1 — Identité du signal et métadonnées

#### A.1.1 — Identifiant unique du signal

| Champ | Contenu |
|---|---|
| **Nom** | Identifiant signal |
| **Définition** | Identifiant globalement unique de l'analyse, déterministe (SHA-1 sur symbole + bar timestamp + direction + score arrondi à 4 décimales). |
| **Valeur exemple** | `"a3f7c1e9b2d8"` (préfixe 12 hex) |
| **Étage** | `ConfluenceDetector.analyze()` — `src/intelligence/confluence_detector.py:354` |
| **Forme exposée** | v1-REST : `signal_id` · v2-unified : `id` · B2B-mock : `insight_id` |
| **Granularité** | Un par signal stable (réutilisé tant que le state machine garde le signal actif) |

#### A.1.2 — Instrument et timeframe

| Champ | Contenu |
|---|---|
| **Nom** | Instrument / Timeframe |
| **Définition** | Paire analysée (XAUUSD, EURUSD, BTCUSD, US500, GBPUSD, USDJPY) et unité de temps (M1→W1). |
| **Valeur exemple** | `instrument="XAUUSD"`, `timeframe="M15"` |
| **Étage** | Configuré au scanner (`SentinelScanner`), propagé à `ConfluenceDetector(symbol=…)` et `InsightSignalV2`. |
| **Forme exposée** | v1-REST : `symbol` · v2-unified : `instrument`, `timeframe` · B2B-mock : `asset`, `timeframe` |
| **Granularité** | Par signal |

#### A.1.3 — Timestamps de cycle de vie

| Champ | Contenu |
|---|---|
| **Nom** | Created_at / valid_until |
| **Définition** | Date-heure UTC de création du signal et borne de validité optionnelle (expiration). |
| **Valeur exemple** | `created_at_utc="2026-05-01T12:00:00Z"`, `valid_until_utc="2026-05-01T16:00:00Z"` |
| **Étage** | `ConfluenceSignal.created_at` (`confluence_detector.py:73`) + `SignalStateMachine` (lifetime timer interne) ; sérialisation v2 dans `from_v1_signal()`. |
| **Forme exposée** | v1-REST : `created_at` (sans expiration) · v2-unified : `created_at_utc`, `valid_until_utc` · B2B-mock : `generated_at`, `expires_at` |
| **Granularité** | Par signal |

#### A.1.4 — Bar timestamp d'origine

| Champ | Contenu |
|---|---|
| **Nom** | Bar timestamp |
| **Définition** | ISO-8601 de la barre OHLCV qui a déclenché l'analyse — distinct de `created_at` (peut être différé via state machine ARMING). |
| **Valeur exemple** | `"2026-05-01T11:45:00Z"` |
| **Étage** | `ConfluenceSignal.bar_timestamp` (`confluence_detector.py:72`) |
| **Forme exposée** | 🔒 INTERNE — utilisé pour reproductibilité backtest, non exposé v1/v2 |
| **Granularité** | Par signal |

#### A.1.5 — Version de schéma

| Champ | Contenu |
|---|---|
| **Nom** | Schema version |
| **Définition** | Versioning sémantique du contrat InsightSignal (frozen field, 2.0.0 actuel). |
| **Valeur exemple** | `"2.0.0"` |
| **Étage** | `InsightSignalV2.schema_version` — constante module |
| **Forme exposée** | v2-unified : `schema_version` · B2B-mock : `version_schema` |
| **Granularité** | Par signal (constant) |

---

### Catégorie A.2 — Structure de marché (Smart Money / ICT)

L'engine SMC produit un DataFrame enrichi par barre, dont **certaines colonnes alimentent le scoring** mais ne sortent pas individuellement vers le client v1/v2. Le mockup B2B agrège ces drapeaux dans `components_active` et `key_levels`.

#### A.2.1 — Direction structurelle (Break of Structure)

| Champ | Contenu |
|---|---|
| **Nom** | BOS — sens de la tendance |
| **Définition** | Sens dominant de la structure de marché : haussière, baissière ou indéfinie. État propagé de barre en barre tant que la structure tient. |
| **Valeur exemple** | `BOS_SIGNAL=+1` (haussier), `-1` (baissier), `0` (rien) |
| **Étage** | `SmartMoneyEngine.analyze()` — colonne `BOS_SIGNAL` |
| **Forme exposée** | Indirect — projeté en `direction`/`signal_type` (LONG/SHORT) après scoring. v2-unified : `direction` (`BULLISH_SETUP`/`BEARISH_SETUP`/`NEUTRAL`) · B2B-mock : `structure_bias` (`bullish`/`bearish`/`neutral`) |
| **Granularité** | Par barre (interne), par signal (exposé) |

#### A.2.2 — Cassure de structure fraîche (BOS Event)

| Champ | Contenu |
|---|---|
| **Nom** | BOS event |
| **Définition** | Drapeau ne s'allumant **que sur la barre exacte** où la cassure se produit, puis se réinitialise. Distingue « tendance établie » de « cassure fraîche ». |
| **Valeur exemple** | `BOS_EVENT=+1` sur la barre de cassure haussière, `0` les barres suivantes |
| **Étage** | `SmartMoneyEngine.analyze()` — colonne `BOS_EVENT` |
| **Forme exposée** | 🔒 INTERNE — utilisé par `_score_bos` (booster qualité), pas exposé client |
| **Granularité** | Par barre |

#### A.2.3 — Niveau structurel cassé

| Champ | Contenu |
|---|---|
| **Nom** | BOS break level |
| **Définition** | Prix du swing high/low pivoté lors de la cassure (résistance ou support cassé). |
| **Valeur exemple** | `BOS_BREAK_LEVEL=2391.5` (sur la barre de cassure uniquement) |
| **Étage** | `SmartMoneyEngine.analyze()` — colonne `BOS_BREAK_LEVEL` |
| **Forme exposée** | 🔒 INTERNE actuellement — apparaît dans `narrative_long` (B2B-mock) en prose (« cassure au-dessus de 2391.5 ») mais pas comme champ structuré |
| **Granularité** | Par événement BOS |

#### A.2.4 — Retournement de caractère (CHOCH)

| Champ | Contenu |
|---|---|
| **Nom** | CHOCH — change of character |
| **Définition** | Événement de retournement de la structure (premier swing dans le sens opposé). Indicateur de reversal SMC. |
| **Valeur exemple** | `CHOCH_SIGNAL=-1` (retournement baissier sur cette barre) |
| **Étage** | `SmartMoneyEngine.analyze()` — colonne `CHOCH_SIGNAL` |
| **Forme exposée** | 🔒 INTERNE — booste la note BOS jusqu'à 100 % du poids (`_quality_with_retest`), pas de champ dédié client |
| **Granularité** | Par événement |

#### A.2.5 — Divergence RSI sur CHOCH

| Champ | Contenu |
|---|---|
| **Nom** | CHOCH divergence |
| **Définition** | Force de la divergence RSI observée au moment du CHOCH (confirmation de reversal). |
| **Valeur exemple** | `CHOCH_DIVERGENCE=+1` (divergence haussière), valeur sur échelle de force interne |
| **Étage** | `SmartMoneyEngine.analyze()` + `ConfluenceDetector._score_rsi_divergence` |
| **Forme exposée** | Indirect via composante `rsi_divergence` du breakdown (B2B-mock) ; pas de champ direct v1/v2 |
| **Granularité** | Par événement |

#### A.2.6 — Fair Value Gap (présence et taille)

| Champ | Contenu |
|---|---|
| **Nom** | FVG — Fair Value Gap |
| **Définition** | Détection d'un gap de prix non rempli (déséquilibre 3-bougies). Drapeau de présence + taille normalisée en multiples d'ATR. |
| **Valeur exemple** | `FVG_SIGNAL=+1`, `FVG_SIZE_NORM=0.42` (gap = 0.42 × ATR), `FVG_DIR=True` |
| **Étage** | `SmartMoneyEngine.analyze()` — colonnes `FVG_SIGNAL`, `FVG_SIZE`, `FVG_SIZE_NORM`, `FVG_DIR` |
| **Forme exposée** | 🔒 INTERNE comme champ direct ; visible dans `components_active` (B2B-mock) et dans `narrative_long`/`narrative_short` (prose) |
| **Granularité** | Par barre |

#### A.2.7 — Order Block bornes et force

| Champ | Contenu |
|---|---|
| **Nom** | OB — Order Block |
| **Définition** | Zone d'absorption institutionnelle détectée (bougie engulfing + déplacement), avec bornes haute/basse et force normalisée. |
| **Valeur exemple** | `BULLISH_OB_HIGH=2378`, `BULLISH_OB_LOW=2375`, `OB_STRENGTH_NORM=0.73` |
| **Étage** | `SmartMoneyEngine.analyze()` — colonnes `BULLISH_OB_HIGH/LOW`, `BEARISH_OB_HIGH/LOW`, `OB_STRENGTH_NORM` |
| **Forme exposée** | 🔒 INTERNE comme champ direct ; force agrégée dans la composante `order_block` du breakdown B2B-mock |
| **Granularité** | Par barre |

#### A.2.8 — État de retest du niveau cassé

| Champ | Contenu |
|---|---|
| **Nom** | Retest state / armed |
| **Définition** | Machine d'état à 4 valeurs : idle / awaiting / armed / opposite-direction-armed. `BOS_RETEST_ARMED=1` ⇒ le prix est revenu toucher le niveau cassé puis a rebondi : entrée autorisée. |
| **Valeur exemple** | `BOS_RETEST_STATE=+2`, `BOS_RETEST_ARMED=+1` |
| **Étage** | `SmartMoneyEngine.analyze()` — colonnes `BOS_RETEST_STATE`, `BOS_RETEST_ARMED` |
| **Forme exposée** | 🔒 INTERNE — gate dur dans `ConfluenceDetector.analyze` (signal refusé si non armé quand `require_retest=True`), non exposé client |
| **Granularité** | Par barre |

#### A.2.9 — Indicateurs techniques classiques

| Champ | Contenu |
|---|---|
| **Nom** | RSI / MACD / Bollinger / Fractals / ATR |
| **Définition** | Indicateurs standards calculés en support (RSI 7p, MACD line/signal/diff, BB 20p ±2σ, fractals 2-bar pivot, ATR 14p). |
| **Valeur exemple** | `RSI=58`, `MACD_Diff=+0.012`, `BB_H=2402`, `BB_M=2390`, `BB_L=2378`, `ATR=8.4`, `UP_FRACTAL=2401.5` |
| **Étage** | `SmartMoneyEngine.analyze()` |
| **Forme exposée** | 🔒 INTERNE pour la plupart ; RSI/MACD résumés dans `components_active.momentum` du breakdown B2B-mock |
| **Granularité** | Par barre |

---

### Catégorie A.3 — Score de confluence et décomposition

#### A.3.1 — Score de confluence 0-100

| Champ | Contenu |
|---|---|
| **Nom** | Confluence score |
| **Définition** | Note globale 0-100 mesurant la convergence simultanée de 8 composantes techniques + contextuelles (BOS, FVG, OB, régime, news, volume, momentum, divergence RSI). Renormalisé si certaines composantes n'ont pas de donnée (ex : news/volume absents en replay). |
| **Valeur exemple** | `confluence_score=62.4` |
| **Étage** | `ConfluenceDetector.analyze()` — variable `total_score`, `confluence_detector.py:281` |
| **Forme exposée** | v1-REST : ❌ non exposé · v2-unified : `conviction_0_100` (entier) + `conviction_label` bucketé (`weak`/`moderate`/`strong`/`institutional`) · B2B-mock : `confluence_score` (exact) + `confluence_score_label` |
| **Granularité** | Par signal |
| **Note** | v2 expose volontairement le score brut **et** un label bucketé pour permettre une UI client qui masque la précision (protection IP — v2 docstring l. 71-77) |

#### A.3.2 — Label de conviction (bucket)

| Champ | Contenu |
|---|---|
| **Nom** | Conviction label |
| **Définition** | Buckets sémantiques du score : `weak` (0-39), `moderate` (40-59), `strong` (60-79), `institutional` (80-100). Évite d'exposer l'échelle brute en surface compacte (Telegram). |
| **Valeur exemple** | `"STRONG"` |
| **Étage** | `conviction_to_label()` — `src/api/insight_signal_v2.py:80` |
| **Forme exposée** | v2-unified : `conviction_label` (property) · B2B-mock : `confluence_score_label` |
| **Granularité** | Par signal |
| **Note** | À ne PAS confondre avec le tier `SignalTier` interne (`PREMIUM`/`STANDARD`/`WEAK`/`INVALID`) défini sur le score brut côté `ConfluenceDetector` — les seuils diffèrent (tier ≥ 55 vs label ≥ 60 pour `strong`). |

#### A.3.3 — Tier interne du signal

| Champ | Contenu |
|---|---|
| **Nom** | Signal tier |
| **Définition** | Classification `PREMIUM` (≥ 55) / `STANDARD` (≥ 40) / `WEAK` (≥ 25) / `INVALID` (< 25), recalibrée 2026-04-29 sur la distribution empirique XAU M15 2019-2025 post-RegimeFilter (p90/p50/p10). |
| **Valeur exemple** | `tier="STANDARD"` |
| **Étage** | `ConfluenceDetector._classify_tier` — `confluence_detector.py:640` |
| **Forme exposée** | 🔒 INTERNE — pas exposé v1/v2/B2B-mock ; doublonne partiellement le `conviction_label` |
| **Granularité** | Par signal |

#### A.3.4 — Décomposition par composante (breakdown)

| Champ | Contenu |
|---|---|
| **Nom** | Components / breakdown |
| **Définition** | Liste des 8 composantes scorées, chacune avec : nom, valeur brute de l'indicateur, score pondéré (0 à poids_max), poids alloué, raisonnement texte 1-ligne. |
| **Valeur exemple** | `[{name:"bos", weighted_score:13.5, weight:15.0, reasoning:"Bullish BOS retest confirmed (quality=90%)"}, …]` (8 items) |
| **Étage** | `ConfluenceSignal.components` (`confluence_detector.py:70`) — chaque `ComponentScore` |
| **Forme exposée** | v1-REST : ❌ · v2-unified : ❌ · B2B-mock : ✅ `components_score_breakdown` (le tableau complet) + `components_active` (liste des noms non-zéro) |
| **Granularité** | Par signal |
| **Note** | Seul vecteur d'audit/explainability disponible aujourd'hui — non visible en B2C. Ressemble à un mini-SHAP plat. |

#### A.3.5 — Liste de raisons (reasoning lines)

| Champ | Contenu |
|---|---|
| **Nom** | Reasoning lines |
| **Définition** | Liste textuelle des composantes ayant contribué positivement (`weighted_score > 0`), un raisonnement par item. |
| **Valeur exemple** | `["Bullish BOS retest confirmed", "Bullish FVG (size=0.42×ATR)", "Bullish regime confirms LONG", …]` |
| **Étage** | `ConfluenceSignal.reasoning` (`confluence_detector.py:71`, alimenté l. 322) |
| **Forme exposée** | 🔒 INTERNE — consommé par `LLMNarrativeEngine` (hors périmètre) et par les logs ; pas de champ v1/v2 dédié |
| **Granularité** | Par signal |

#### A.3.6 — Poids des composantes

| Champ | Contenu |
|---|---|
| **Nom** | Component weights |
| **Définition** | Configuration des poids (somme = 100) : BOS 15, FVG 15, OB 10, Regime 25, News 20, Volume 10, Momentum 3, RSI div 2. |
| **Valeur exemple** | `DEFAULT_WEIGHTS={"bos":15.0, …}` |
| **Étage** | `confluence_detector.py:117` (`DEFAULT_WEIGHTS` constant) |
| **Forme exposée** | Indirect : champ `weight` dans chaque ligne du `components_score_breakdown` (B2B-mock) ; pas exposé v1/v2 |
| **Granularité** | Configuration (constant tant que pas reparam) |

---

### Catégorie A.4 — Niveaux de prix (entrée, stop, cibles, invalidation)

#### A.4.1 — Prix d'entrée

| Champ | Contenu |
|---|---|
| **Nom** | Entry price |
| **Définition** | Prix de la barre courante au moment de l'analyse (clôture). Arrondi selon `price_decimals` de l'instrument (XAU 2, FX 5, JPY 3, indice 1, crypto 2). |
| **Valeur exemple** | `entry_price=2350.00` |
| **Étage** | `ConfluenceSignal.entry_price` (`confluence_detector.py:65`) |
| **Forme exposée** | v1-REST : `entry_price` · v2-unified : `levels.entry` · B2B-mock : zone via `key_levels` |
| **Granularité** | Par signal |

#### A.4.2 — Stop-loss

| Champ | Contenu |
|---|---|
| **Nom** | Stop loss |
| **Définition** | Niveau de protection calculé comme `entry ± SL_ATR_MULT × ATR_sizing` (multiplicateur 2.0 par défaut, ×1.5 supplémentaire en régime vol haut). |
| **Valeur exemple** | `stop_loss=2340.00` (LONG XAU M15, ATR≈5) |
| **Étage** | `ConfluenceDetector.analyze` — calcul l. 306-318 |
| **Forme exposée** | v1-REST : `stop_loss` · v2-unified : `levels.stop` · B2B-mock : déduit dans `key_levels.structural_invalidation` |
| **Granularité** | Par signal |

#### A.4.3 — Take-profit / cible

| Champ | Contenu |
|---|---|
| **Nom** | Take profit |
| **Définition** | Niveau de cible calculé comme `entry ± TP_ATR_MULT × ATR_sizing` (multiplicateur 4.0 par défaut). Volontairement **non élargi** en vol haute (markets atteignent cible plus vite, élargir baisse le hit rate). |
| **Valeur exemple** | `take_profit=2370.00` |
| **Étage** | `ConfluenceDetector.analyze` — l. 307, 314, 318 |
| **Forme exposée** | v1-REST : `take_profit` · v2-unified : `levels.target_1` (+ optionnel `target_2`) · B2B-mock : `key_levels.first_target`, `key_levels.second_target` |
| **Granularité** | Par signal |

#### A.4.4 — Ratio risque/récompense

| Champ | Contenu |
|---|---|
| **Nom** | RR ratio |
| **Définition** | Rapport `distance_TP / distance_SL`. Avec les multiplicateurs ATR par défaut (4/2), vaut 2.0 ; en régime vol haute, le SL est élargi mais pas le TP → RR descend vers 1.33. |
| **Valeur exemple** | `rr_ratio=2.0` |
| **Étage** | `ConfluenceSignal.rr_ratio` (l. 320) ; propriété recalculée dans `InsightSignalV2.rr_ratio` |
| **Forme exposée** | v1-REST : `rr_ratio` · v2-unified : `rr_ratio` (property) · B2B-mock : implicite via les key_levels |
| **Granularité** | Par signal |

#### A.4.5 — Niveau d'invalidation structurelle

| Champ | Contenu |
|---|---|
| **Nom** | Invalidation level |
| **Définition** | Niveau distinct du stop-loss (qui est basé ATR), représentant la **borne structurelle** sous/au-dessus de laquelle la thèse SMC est cassée (ex : sous le plancher du FVG). |
| **Valeur exemple** | `invalidation=2335.00` (sous le SL) |
| **Étage** | Pas calculé explicitement par `ConfluenceDetector` ; uniquement déclaré comme champ optionnel v2 — actuellement non rempli en prod (laissé `None` par `from_v1_signal`). |
| **Forme exposée** | v2-unified : `levels.invalidation` (champ déclaré, vide en prod) · B2B-mock : `key_levels.structural_invalidation` (renseigné dans le mockup) |
| **Granularité** | Par signal |
| **Note ⚠️** | **Champ déclaré mais non produit par le pipeline aujourd'hui.** Le SL ATR sert d'invalidation par défaut. |

#### A.4.6 — Zones de support / résistance

| Champ | Contenu |
|---|---|
| **Nom** | Support / resistance zones |
| **Définition** | Zones (intervalles) plutôt que niveaux ponctuels, déduites des OB et FVG actifs. |
| **Valeur exemple** | `support_zone=[2378.00, 2381.00]`, `resistance_zone=[2398.50, 2401.20]` |
| **Étage** | Non calculé par le code actuel — apparaît uniquement dans `mockups/b2b_insight.json` ; les bornes existent dans `BULLISH_OB_HIGH/LOW` et `FVG_SIZE` mais ne sont pas agrégées en zones nommées. |
| **Forme exposée** | B2B-mock : `key_levels.support_zone`, `key_levels.resistance_zone` |
| **Granularité** | Par signal (en mockup) |
| **Note ⚠️** | **Champ aspirationnel** — exposé dans le mockup B2B mais non implémenté côté algo. |

---

### Catégorie A.5 — Direction / setup classification

#### A.5.1 — Direction du setup (côté algo)

| Champ | Contenu |
|---|---|
| **Nom** | Signal type interne |
| **Définition** | `LONG` ou `SHORT`, dérivé du sens de `BOS_SIGNAL` (≠ event flag — voir A.2.2). |
| **Valeur exemple** | `signal_type=SignalType.LONG` |
| **Étage** | `ConfluenceDetector.analyze` — l. 238 |
| **Forme exposée** | v1-REST via `action=OPEN_LONG`/`OPEN_SHORT`/`HOLD` ; v2-unified après mapping vers `SetupDirection.BULLISH_SETUP`/`BEARISH_SETUP`/`NEUTRAL` (mapping `from_v1_signal`, l. 392-398). |
| **Granularité** | Par signal |

#### A.5.2 — Direction côté client (UE-compliant)

| Champ | Contenu |
|---|---|
| **Nom** | Setup direction |
| **Définition** | Label client : `BULLISH_SETUP` / `BEARISH_SETUP` / `NEUTRAL`. Volontairement **pas** « BUY » / « SELL » / « achetez » / « vendez » pour conformité UE 2024/2811 (finfluencer regulation). |
| **Valeur exemple** | `direction="BULLISH_SETUP"` |
| **Étage** | `InsightSignalV2.direction`, enum `SetupDirection` (`insight_signal_v2.py:47`) |
| **Forme exposée** | v2-unified : `direction` · B2B-mock : `structure_bias` (mais en minuscule `bullish`/`bearish`) |
| **Granularité** | Par signal |

#### A.5.3 — Action v1 héritée (à déprécier)

| Champ | Contenu |
|---|---|
| **Nom** | Action v1 |
| **Définition** | Enum legacy `HOLD` / `OPEN_LONG` / `OPEN_SHORT` / `CLOSE_LONG` / `CLOSE_SHORT`. Couvre entrée ET sortie en un seul champ. |
| **Valeur exemple** | `action="OPEN_LONG"` |
| **Étage** | `SignalAction` enum — `src/api/models.py:30` |
| **Forme exposée** | v1-REST : `action` (forme native) · v2-unified : mappé vers `direction` |
| **Granularité** | Par signal |
| **Note ⚠️** | Non conforme UE 2024/2811 dans sa formulation (« OPEN_LONG » = verbe achat). v2 corrige. |

---

### Catégorie A.6 — Volatilité prévisionnelle

#### A.6.1 — ATR forecast (volatilité prévue)

| Champ | Contenu |
|---|---|
| **Nom** | Forecast ATR |
| **Définition** | Prévision de l'ATR de la prochaine fenêtre (1-bar ahead). Trois moteurs disponibles : HAR-RV (défaut, latence ~50 ms), LightGBM, hybride. |
| **Valeur exemple** | `forecast_atr=8.7` (pips XAU M15) |
| **Étage** | `VolatilityForecast.forecast_atr` (`volatility_forecaster.py`) → `ConfluenceSignal.vol_forecast_atr` |
| **Forme exposée** | v1-REST : ❌ · v2-unified : ❌ (pas dans le modèle) · v2-unified-extras : `VolatilityContext.forecast_atr_pct` (sous-modèle optionnel) · B2B-mock : `volatility_forecast.next_hour_usd` + `vs_atr14_pct` |
| **Granularité** | Par signal (recalculé à chaque barre) |

#### A.6.2 — ATR naïf de référence

| Champ | Contenu |
|---|---|
| **Nom** | Naive ATR |
| **Définition** | ATR 14-période brut, baseline de comparaison. Le delta vs `forecast_atr` mesure la « valeur ajoutée » du modèle vol. |
| **Valeur exemple** | `naive_atr=7.4` |
| **Étage** | `VolatilityForecast.naive_atr` |
| **Forme exposée** | v2-unified : `VolatilityContext.naive_atr_pct` (optionnel) · B2B-mock : déduit via `vs_atr14_pct` |
| **Granularité** | Par signal |

#### A.6.3 — Régime de volatilité (HMM 3 états)

| Champ | Contenu |
|---|---|
| **Nom** | Vol regime state |
| **Définition** | Label HMM 3-états : `low` / `normal` / `high`, ou `unknown` si HMM non disponible. Déterminé par variance des résidus + posterior probability. |
| **Valeur exemple** | `vol_regime="normal"` |
| **Étage** | `VolatilityForecast.regime_state` (`volatility_forecaster.py`) ; logique `RegimeClassifier` 3 états (`regime_classifier.py`) avec labels sémantiques (`low_vol_trending` / `low_vol_ranging` / `high_vol_stress`). |
| **Forme exposée** | `ConfluenceSignal.vol_regime` (interne) · v2-unified : `VolatilityContext.regime` · B2B-mock : `volatility_forecast.regime_label` + `regime_hmm` (label étendu trend/range) |
| **Granularité** | Par signal |
| **Note** | Deux taxonomies coexistent : la simple (`low/normal/high`) pour la vol, et l'étendue (`trend_bullish/trend_bearish/range_low_vol/range_high_vol`) pour le contexte structurel — le mockup B2B mélange les deux. |

#### A.6.4 — Intervalle de confiance vol (TCP)

| Champ | Contenu |
|---|---|
| **Nom** | Vol confidence interval |
| **Définition** | Bornes inférieure/supérieure de la prévision ATR via Transductive Conformal Prediction (TCP) — quantifie l'incertitude propre du modèle vol. |
| **Valeur exemple** | `vol_confidence_lower=7.2`, `vol_confidence_upper=10.4` |
| **Étage** | `VolatilityForecast.confidence_lower/upper` → `ConfluenceSignal.vol_confidence_lower/upper` |
| **Forme exposée** | 🔒 INTERNE — stocké dans la DB (`SignalRecord.vol_confidence`) mais aucune route v1/v2 ne le retourne. Pas dans le B2B-mock. |
| **Granularité** | Par signal |

#### A.6.5 — Multiplicateurs structurels vol

| Champ | Contenu |
|---|---|
| **Nom** | Diurnal / calendar / regime multipliers |
| **Définition** | Trois multiplicateurs orthogonaux composant la prévision blendée : saisonnalité intraday (~0.7-1.3), proximité événement éco (~1.0-2.5), état HMM (~0.5-2.5). |
| **Valeur exemple** | `diurnal_multiplier=1.12`, `calendar_multiplier=1.0`, `regime_multiplier=0.95` |
| **Étage** | `VolatilityForecast` (`volatility_forecaster.py`) |
| **Forme exposée** | 🔒 INTERNE — diagnostic backtest, pas de surface client |
| **Granularité** | Par signal |

#### A.6.6 — Poids de blend HAR vs naïf

| Champ | Contenu |
|---|---|
| **Nom** | Blend weight |
| **Définition** | `w ∈ [0,1]` tel que `forecast = w × HAR + (1-w) × naïf`. Calibré sur set de validation. |
| **Valeur exemple** | `blend_weight=0.62` |
| **Étage** | `VolatilityForecast.blend_weight` |
| **Forme exposée** | 🔒 INTERNE |
| **Granularité** | Calibré périodiquement (pas par signal) |

#### A.6.7 — Drapeau fallback

| Champ | Contenu |
|---|---|
| **Nom** | Is_fallback |
| **Définition** | `True` si la prévision a basculé sur l'ATR naïf (modèle indisponible / pas calibré). Marqueur de dégradation gracieuse. |
| **Valeur exemple** | `is_fallback=false` |
| **Étage** | `VolatilityForecast.is_fallback` |
| **Forme exposée** | 🔒 INTERNE |
| **Granularité** | Par signal |

---

### Catégorie A.7 — Contexte de régime macro / structure

#### A.7.1 — Régime de marché (structure)

| Champ | Contenu |
|---|---|
| **Nom** | Market regime |
| **Définition** | Étiquette structurelle distincte de la vol : tendances (`uptrend`, `downtrend`) vs `ranging` vs cas ambigus. Confluence boost +25 % du poids si aligné avec le sens du signal. |
| **Valeur exemple** | `regime_type="uptrend"`, `trend_direction="long"`, `confidence=0.71`, `trend_strength=0.58` |
| **Étage** | `MarketRegimeAgent` → consommé par `ConfluenceDetector._score_regime` (l. 483-525) |
| **Forme exposée** | 🔒 INTERNE comme champ direct ; visible dans la composante `regime` du breakdown B2B-mock (`reasoning: "HMM trend_bullish posterior 0.71"`) ; v2 expose seulement la vol regime, pas la structure regime. |
| **Granularité** | Par signal |

#### A.7.2 — Multiplicateur de taille de position (régime)

| Champ | Contenu |
|---|---|
| **Nom** | Regime position multiplier |
| **Définition** | Suggestion de taille relative (0-1.5) issue du régime : trending = 1.0, ranging = 0.5-0.7, stress = 0.0-0.3. |
| **Valeur exemple** | `position_size_multiplier=0.85` |
| **Étage** | `RegimeAnalysis.position_size_multiplier` → multiplié avec `news.position_multiplier` pour produire `ConfluenceSignal.position_multiplier` |
| **Forme exposée** | 🔒 INTERNE (le produit est exposé — voir A.8.4) |
| **Granularité** | Par signal |

#### A.7.3 — Décision changepoint (BOCPD)

| Champ | Contenu |
|---|---|
| **Nom** | Regime gate decision |
| **Définition** | Décision 3-états du `RegimeGate` (`TRADE` / `REDUCE` / `BLOCK`) basée sur BOCPD (changepoint posterior) + bipower variation (part de saut). |
| **Valeur exemple** | `decision=TRADE`, `cp_prob=0.03`, `jump_ratio=0.12`, `expected_run_length=180` |
| **Étage** | `RegimeGate.update()` (`regime_gate.py`) |
| **Forme exposée** | 🔒 INTERNE — utilisé en amont/aval pour gating ; le scanner expose seulement le compteur agrégé `signals_dropped_by_regime_filter` (voir A.9.5) |
| **Granularité** | Par barre |

#### A.7.4 — Drapeau filtre régime (NY × vol)

| Champ | Contenu |
|---|---|
| **Nom** | Regime filter decision |
| **Définition** | Booléen `allowed` + raison texte (« NY × high vol », « regime ok »…). Filtre statistique abandonnant les barres NY haute vol (≈60 % des barres). |
| **Valeur exemple** | `allowed=true`, `reason="regime ok"` |
| **Étage** | `RegimeFilter.evaluate()` (`regime_filter.py`) |
| **Forme exposée** | 🔒 INTERNE — compteur agrégé exposé via health endpoint |
| **Granularité** | Par barre |

#### A.7.5 — Session de marché

| Champ | Contenu |
|---|---|
| **Nom** | Session label |
| **Définition** | Étiquette de session : `asian` / `london` / `ny_overlap` / `ny_afternoon` / `after_hours`. Utilisée comme feature LGBM et comme dimension du filtre régime. |
| **Valeur exemple** | `session="new_york"` |
| **Étage** | Calculée en feature engineering (`volatility_lgbm.py`) |
| **Forme exposée** | 🔒 INTERNE comme feature ; mais **exposée explicitement** dans B2B-mock : `session: "new_york"` |
| **Granularité** | Par barre |

---

### Catégorie A.8 — Sentiment news & calendrier

#### A.8.1 — Décision news (block / pass)

| Champ | Contenu |
|---|---|
| **Nom** | News decision |
| **Définition** | Verdict de l'agent news : `BLOCK` (high-impact event dans la fenêtre -30 / +60 min) ou `PASS`. Gate dur : `BLOCK` annule le signal en amont. |
| **Valeur exemple** | `decision="PASS"` |
| **Étage** | `NewsAnalysisAgent` → consommé `ConfluenceDetector._is_news_blocked` (l. 220-222) |
| **Forme exposée** | 🔒 INTERNE comme décision ; agrégée dans B2B-mock via `news_blackout_active: false` |
| **Granularité** | Par barre |

#### A.8.2 — Score de sentiment

| Champ | Contenu |
|---|---|
| **Nom** | News sentiment score |
| **Définition** | Score sentimental [-1, +1] : positif → bullish, négatif → bearish, accompagné d'une confidence [0, 1]. |
| **Valeur exemple** | `sentiment_score=+0.4`, `sentiment_confidence=0.7` |
| **Étage** | `NewsAssessment.sentiment_score` → composante `news` du score |
| **Forme exposée** | 🔒 INTERNE — visible uniquement dans le `reasoning` de la composante news du breakdown B2B-mock |
| **Granularité** | Par barre |

#### A.8.3 — Multiplicateur de taille (news)

| Champ | Contenu |
|---|---|
| **Nom** | News position multiplier |
| **Définition** | Suggestion de réduction de taille en raison d'un événement à venir (0 = veto, 0.5 = half-size, 1.0 = baseline). |
| **Valeur exemple** | `news_multiplier=1.0` |
| **Étage** | `NewsAssessment.position_multiplier` |
| **Forme exposée** | 🔒 INTERNE (composé dans A.8.4) |
| **Granularité** | Par barre |

#### A.8.4 — Multiplicateur composé (régime × news)

| Champ | Contenu |
|---|---|
| **Nom** | Position multiplier (composé) |
| **Définition** | Produit `regime_mult × news_mult`, borné dans [0, 1.5]. 1.0 = taille normale, 0 = ne pas trader, > 1 = signal exceptionnel. Accompagné d'un texte de raisonnement. |
| **Valeur exemple** | `position_multiplier=0.85`, `position_reasoning="regime×news = 0.85 × 1.00 = 0.85"` |
| **Étage** | `ConfluenceDetector.analyze` (l. 328-341) → `ConfluenceSignal.position_multiplier` |
| **Forme exposée** | 🔒 INTERNE — non exposé en v1/v2/B2B-mock à ce jour |
| **Granularité** | Par signal |
| **Note ⚠️** | Champ pertinent pour usage broker B2B (sizing client) mais absent des contrats actuels. |

---

### Catégorie A.9 — État du signal et cycle de vie

#### A.9.1 — État public du state machine

| Champ | Contenu |
|---|---|
| **Nom** | State machine public state |
| **Définition** | Trois états observables côté client : `HOLD` (rien) / `BUY` (LONG actif) / `SELL` (SHORT actif). Cachet l'état interne (`IDLE` / `ARMING` / `ACTIVE_LONG` / `ACTIVE_SHORT` / `COOLDOWN`). |
| **Valeur exemple** | `state="BUY"` |
| **Étage** | `SignalStateMachine.on_bar()` → `StateSnapshot.state` (`signal_state_machine.py:298`) |
| **Forme exposée** | v1-REST : indirect (action change) · v2-unified : indirect (direction NEUTRAL ↔ HOLD) · Pas de champ « state » dédié exposé |
| **Granularité** | Par barre |

#### A.9.2 — Compteurs de cycle de vie

| Champ | Contenu |
|---|---|
| **Nom** | Bars in/remaining, cooldown remaining |
| **Définition** | Trois compteurs : `bars_in_state` (durée actuelle), `bars_remaining` (avant `time_expired`), `cooldown_bars_remaining` (avant fin de lockout). |
| **Valeur exemple** | `bars_in_state=12`, `bars_remaining=52`, `cooldown_bars_remaining=null` |
| **Étage** | `StateSnapshot` (snapshot par barre) |
| **Forme exposée** | 🔒 INTERNE — pas de surface client |
| **Granularité** | Par barre |

#### A.9.3 — Progression de confirmation (ARMING)

| Champ | Contenu |
|---|---|
| **Nom** | Confirmation progress |
| **Définition** | Tuple `(current, needed)` : combien de barres de confirmation accumulées sur le nombre requis avant transition vers ACTIVE. |
| **Valeur exemple** | `confirmation_progress=(2, 3)` |
| **Étage** | `StateSnapshot.confirmation_progress` |
| **Forme exposée** | 🔒 INTERNE |
| **Granularité** | Par barre, pendant phase ARMING |

#### A.9.4 — Raison de sortie

| Champ | Contenu |
|---|---|
| **Nom** | Exit reason |
| **Définition** | Code de raison de sortie : `target_reached` / `invalidated` / `time_expired` / `score_decayed` / `regime_shifted` / `opposing_signal`. |
| **Valeur exemple** | `last_exit_reason="target_reached"` |
| **Étage** | `StateTransition.exit_reason` + `StateSnapshot.last_exit_reason` |
| **Forme exposée** | 🔒 INTERNE — pourrait être exposé en post-mortem mais pas implémenté |
| **Granularité** | Par transition |

#### A.9.5 — Outcome historique et P&L

| Champ | Contenu |
|---|---|
| **Nom** | Outcome / PnL pips |
| **Définition** | Résultat du signal une fois clos : `win` / `loss` / `breakeven`, avec P&L en pips. |
| **Valeur exemple** | `outcome="win"`, `pnl_pips=+38.2`, `closed_at="2026-05-01T14:30:00Z"` |
| **Étage** | `SignalRecord` persisté en SQLite (`src/api/signal_store.py`) |
| **Forme exposée** | v1-REST : `SignalHistoryItem.outcome`, `pnl_pips`, `closed_at` · v2-unified : ❌ (pas dans le contrat de base) |
| **Granularité** | Par signal (rempli après clôture) |
| **Note** | Seule information de track-record exposée — sert de socle au futur « track-record auditable » mentionné dans le positioning. |

---

### Catégorie A.10 — Couverture conformelle (incertitude calibrée)

#### A.10.1 — Intervalle conformel autour du score

| Champ | Contenu |
|---|---|
| **Nom** | Conformal interval |
| **Définition** | Bornes inférieure/supérieure autour de la prédiction du scorer ML (logistic L1 ou LGBM), avec garantie marginale de couverture 1-α. Distinct de l'intervalle vol (A.6.4). |
| **Valeur exemple** | `point=0.62`, `lower=0.41`, `upper=0.78`, `alpha=0.10`, `n_calibration=2000` |
| **Étage** | `SplitConformalScorer.predict_interval()` / `AdaptiveConformalScorer` (`conformal_wrapper.py`) |
| **Forme exposée** | 🔒 INTERNE — utilisé pour `should_reject(breakeven)` ; aucun champ v1/v2/B2B-mock ne porte l'intervalle. |
| **Granularité** | Par signal |
| **Note** | C'est l'apport « conformal coverage » mentionné dans le positioning institutional ; aujourd'hui consommé en gating, pas en information client. |

#### A.10.2 — Décision conformelle (reject / pass)

| Champ | Contenu |
|---|---|
| **Nom** | Conformal reject |
| **Définition** | Booléen `should_reject = lower ≤ breakeven` : rejette le signal si la borne basse du gain attendu passe sous le breakeven. |
| **Valeur exemple** | `should_reject=false` |
| **Étage** | `ConformalInterval.should_reject()` |
| **Forme exposée** | 🔒 INTERNE — gating uniquement |
| **Granularité** | Par signal |

#### A.10.3 — Couverture empirique (ACI)

| Champ | Contenu |
|---|---|
| **Nom** | Empirical coverage |
| **Définition** | Couverture observée des intervalles passés (taux de fois où l'outcome a été dans l'intervalle), tracée online par ACI. |
| **Valeur exemple** | `empirical_coverage=0.91` (vs nominal 0.90) |
| **Étage** | `AdaptiveConformalScorer.empirical_coverage()` |
| **Forme exposée** | 🔒 INTERNE — monitoring uniquement |
| **Granularité** | Streaming |

---

### Catégorie A.11 — Narratif et provenance

> ⚠️ Le mission exclut explicitement la couche LLM/RAG. Les fiches ci-dessous portent uniquement sur les **champs structurés portant le narratif** et les flags de provenance — pas sur le contenu généré.

#### A.11.1 — Narratif court (compact)

| Champ | Contenu |
|---|---|
| **Nom** | Narrative short |
| **Définition** | Résumé ≤ 400 caractères (≤ 800 dans la version Telegram). Doit contenir le disclaimer éducatif. |
| **Valeur exemple** | `"Setup haussier XAU M15. Cassure de structure + retest FVG. Régime normal vol. Analyse algorithmique éducative."` |
| **Étage** | Champ `InsightSignalV2.narrative_short` ; généré par LLM (hors périmètre). |
| **Forme exposée** | v2-unified : `narrative_short` · B2B-mock : `narrative_short` · v1-REST : ❌ (mais stocké en DB via `SignalRecord.narrative`) |
| **Granularité** | Par signal |

#### A.11.2 — Narratif long (webapp / B2B)

| Champ | Contenu |
|---|---|
| **Nom** | Narrative long |
| **Définition** | Narratif complet (~500-2000 chars), généré par RAG en Phase 2B. Doit citer des sources si non vide. |
| **Valeur exemple** | Prose ~1500 chars (voir B2B-mock l. 32) |
| **Étage** | `InsightSignalV2.narrative_long` |
| **Forme exposée** | v2-unified : `narrative_long` · B2B-mock : `narrative_full` |
| **Granularité** | Par signal |

#### A.11.3 — Langue du narratif

| Champ | Contenu |
|---|---|
| **Nom** | Narrative language |
| **Définition** | Code langue ISO-639-1 du narratif : `fr` / `en` / `de` / `es`. Par défaut `fr`. |
| **Valeur exemple** | `narrative_language="fr"` |
| **Étage** | `InsightSignalV2.narrative_language` ; persistance via `TelegramLangStore` (sprint compliance W3). |
| **Forme exposée** | v2-unified · B2B-mock |
| **Granularité** | Par client (chat_id → lang) |

#### A.11.4 — Citations / sources

| Champ | Contenu |
|---|---|
| **Nom** | Sources cited |
| **Définition** | Liste de citations avec type (`paper` / `report` / `data` / `education` / `internal`), URL canonique, label, extrait jusqu'à 500 chars. Obligatoire si `narrative_long` non vide en Phase 2B RAG. |
| **Valeur exemple** | `[{type:"report", ref:"https://www.lbma.org.uk/…", label:"LBMA Gold Survey 2025", quoted_excerpt:"…"}, …]` |
| **Étage** | `InsightSignalV2.sources_cited` (liste de `Source`) |
| **Forme exposée** | v2-unified : `sources_cited` · B2B-mock : pas dans le mockup actuel (champ manquant) |
| **Granularité** | Par signal |
| **Note** | C'est l'unique mécanisme d'auditabilité de la justification narrative. Aujourd'hui vide (Phase 2B pas démarré). |

#### A.11.5 — Modèle de génération narratif

| Champ | Contenu |
|---|---|
| **Nom** | Narrative model |
| **Définition** | Identifiant du LLM utilisé pour générer le narratif (`claude-haiku` / `claude-sonnet` / `claude-opus` / `template`). |
| **Valeur exemple** | `narrative_model="claude-sonnet"` |
| **Étage** | Hors `InsightSignalV2` (apparait dans B2B-mock l. 34) |
| **Forme exposée** | B2B-mock uniquement |
| **Granularité** | Par signal |

#### A.11.6 — Scénarios alternatifs

| Champ | Contenu |
|---|---|
| **Nom** | Scenarios |
| **Définition** | Tableau de scénarios `{label, condition, expected_outcome}` : principal + 1-2 alternatifs (invalidation, expiration). |
| **Valeur exemple** | Voir B2B-mock l. 36-52 |
| **Étage** | Non implémenté dans `ConfluenceDetector` — apparaît uniquement dans le mockup B2B. |
| **Forme exposée** | B2B-mock : `scenarios` |
| **Granularité** | Par signal |
| **Note ⚠️** | **Champ aspirationnel** — pas produit par l'algo actuel. |

---

### Catégorie A.12 — Conformité réglementaire

#### A.12.1 — Langue du disclaimer

| Champ | Contenu |
|---|---|
| **Nom** | Disclaimer language |
| **Définition** | Langue dans laquelle le disclaimer obligatoire est rendu (FR/EN/DE/ES). |
| **Valeur exemple** | `disclaimer_lang="fr"` |
| **Étage** | `ComplianceMeta.disclaimer_lang` |
| **Forme exposée** | v2-unified : `compliance.disclaimer_lang` · B2B-mock : implicite dans `disclaimer_text` |
| **Granularité** | Par client |

#### A.12.2 — Juridictions bloquées

| Champ | Contenu |
|---|---|
| **Nom** | Jurisdiction blocked |
| **Définition** | Liste de codes ISO-3166 + subdivisions (ex : `["US", "QC", "UK", "OFAC"]`) pour lesquelles la livraison est bloquée par le geo-block middleware. |
| **Valeur exemple** | `["US", "QC", "UK", "OFAC"]` |
| **Étage** | `ComplianceMeta.jurisdiction_blocked` (rempli par middleware geo-block, sprint W1+W2+W3) |
| **Forme exposée** | v2-unified · B2B-mock |
| **Granularité** | Par signal (constante en pratique) |

#### A.12.3 — Drapeau « edge prouvé »

| Champ | Contenu |
|---|---|
| **Nom** | Edge claim |
| **Définition** | `True` uniquement quand l'edge algorithmique a été validé (post-CP-A1 Phase 2A). Toujours `False` sur la branche 2B narrative-first. L'UI doit **jamais** afficher « edge prouvé » quand `False`. |
| **Valeur exemple** | `edge_claim=false` |
| **Étage** | `ComplianceMeta.edge_claim` |
| **Forme exposée** | v2-unified · B2B-mock : ❌ (pas dans le mockup) |
| **Granularité** | Par signal (constant tant que pas re-validé) |
| **Note** | Honnêteté intellectuelle codifiée : verdict A1 ⇒ ce flag est `False`. Voir `reports/a1_verdict_2026.md`. |

#### A.12.4 — Drapeau « démonstration paper »

| Champ | Contenu |
|---|---|
| **Nom** | Is paper demo |
| **Définition** | `True` quand le signal fait partie d'une démonstration de track-record en paper-trade (Phase 2B feature) plutôt qu'un signal live monétisé. L'UI doit afficher « démonstration ». |
| **Valeur exemple** | `is_paper_demo=true` |
| **Étage** | `ComplianceMeta.is_paper_demo` |
| **Forme exposée** | v2-unified |
| **Granularité** | Par signal |

#### A.12.5 — Texte intégral du disclaimer

| Champ | Contenu |
|---|---|
| **Nom** | Disclaimer text |
| **Définition** | Texte intégral du disclaimer réglementaire (« 74-89 % of retail CFD accounts lose money », « not investment advice », « UE 2024/2811 »). |
| **Valeur exemple** | Voir B2B-mock l. 119 |
| **Étage** | Non porté par `InsightSignalV2` (rendu par renderer surface) — apparait dans B2B-mock. |
| **Forme exposée** | B2B-mock : `compliance.disclaimer_text` + `compliance.regulatory_notice` ; Telegram : ligne en italique automatique (`to_telegram_b2c`) |
| **Granularité** | Constante (par langue) |

---

### Catégorie A.13 — Télémétrie scanner (santé système)

#### A.13.1 — État du scanner

| Champ | Contenu |
|---|---|
| **Nom** | Scanner running / signals generated |
| **Définition** | Booléen `scanner_running` + compteur cumulatif `signals_generated` depuis démarrage. |
| **Valeur exemple** | `scanner_running=true`, `signals_generated=42` |
| **Étage** | `SentinelScanner` (`sentinel_scanner.py`) → `HealthResponse` |
| **Forme exposée** | `/health` endpoint (réponse `HealthResponse` dans `src/api/models.py`) |
| **Granularité** | Par requête |

#### A.13.2 — Compteurs de filtrage

| Champ | Contenu |
|---|---|
| **Nom** | Signals dropped (regime / kill switch) |
| **Définition** | Compteurs de signaux rejetés avant publication : par RegimeFilter (NY × high vol) ou par kill-switch opérationnel. |
| **Valeur exemple** | `signals_dropped_by_regime_filter=128`, `signals_blocked_by_kill_switch=0` |
| **Étage** | `SentinelScanner` |
| **Forme exposée** | `/health` (admin only) |
| **Granularité** | Cumulatif |

#### A.13.3 — Hit rate du cache narratif

| Champ | Contenu |
|---|---|
| **Nom** | Cache hits / misses / hit_rate |
| **Définition** | Statistiques du `SemanticCache` (hit / miss / hit_rate). Empirique : 7.8 % → ~30 % après bump SCORE_BUCKET_PTS (eval_05). |
| **Valeur exemple** | `cache_hits=12`, `cache_misses=30`, `cache_hit_rate=0.286` |
| **Étage** | `SentinelScanner` + `SemanticCache` |
| **Forme exposée** | `/health` |
| **Granularité** | Cumulatif |

#### A.13.4 — Niveau de kill-switch opérationnel

| Champ | Contenu |
|---|---|
| **Nom** | Kill switch level |
| **Définition** | Niveau 0 (off) / 1 (reduced size) / 2 (no new entries). Permet pause d'urgence sans arrêt du process. |
| **Valeur exemple** | `kill_switch_level=0` |
| **Étage** | `HealthResponse` |
| **Forme exposée** | `/health` |
| **Granularité** | État global |

#### A.13.5 — Statut « testing mode »

| Champ | Contenu |
|---|---|
| **Nom** | Testing mode |
| **Définition** | `True` si le scanner tourne en mode sandbox (auth bypass, tous tiers débloqués). |
| **Valeur exemple** | `testing_mode=true` |
| **Étage** | `HealthResponse.testing_mode` (lu depuis env var `SENTINEL_TESTING_MODE`) |
| **Forme exposée** | `/health` |
| **Granularité** | État global |

---

## A.14 — Récapitulatif : ce qui est exposé vs ce qui est interne

### Tableau de matérialité

| Catégorie | Nb fiches | v1-REST | v2-unified | B2B-mock | 🔒 INTERNE |
|---|---:|---:|---:|---:|---:|
| A.1 Identité & métadonnées | 5 | 3 | 4 | 4 | 1 |
| A.2 Structure SMC | 9 | 0 | 0 (via direction) | 2 (`structure_bias` + `components_active`) | 7 |
| A.3 Score & breakdown | 6 | 0 | 2 (`conviction_0_100`, `conviction_label`) | 4 (score + label + breakdown + weights) | 2-3 |
| A.4 Niveaux de prix | 6 | 4 | 5 (dont 1 aspirationnel) | 5 (dont 2 aspirationnels) | 0-1 |
| A.5 Direction setup | 3 | 1 (`action`) | 1 (`direction`) | 1 (`structure_bias`) | 0 |
| A.6 Volatilité | 7 | 0 | 3 (`VolatilityContext`) | 3 (`volatility_forecast`) | 4-7 |
| A.7 Régime macro / structure | 5 | 0 | 0 | 2 (`regime_hmm`, `session`) | 4-5 |
| A.8 News & calendrier | 4 | 0 | 0 | 1 (`news_blackout_active`) | 3-4 |
| A.9 État & cycle de vie | 5 | 3 (history) | 0 | 1 (`expires_at`) | 4-5 |
| A.10 Conformal coverage | 3 | 0 | 0 | 0 | 3 |
| A.11 Narratif & provenance | 6 | 0 | 4 | 4 (dont 1 aspirationnel) | 0-1 |
| A.12 Compliance | 5 | 0 | 4 | 3 | 0 |
| A.13 Télémétrie scanner | 5 | 5 (via `/health`) | n/a | n/a | 0 |
| **TOTAL** | **~73 fiches** | **~16** | **~23** | **~30** | **~30 internes** |

### Constats matérialité (sans interprétation — celle-ci en Partie D)

1. **~40 % des informations produites par l'algo sont INTERNES** (jamais exposées à aucun client) — notamment toute la mécanique d'incertitude (conformal interval, vol confidence, BOCPD changepoint, jump_ratio).
2. **v1-REST en prod est minimaliste** (8 champs) : pas de score, pas de régime, pas de vol, pas de narratif structuré. C'est essentiellement « action + niveaux + RR ».
3. **v2-unified est le contrat « valeur »** : ajoute conviction bucketée, narrative_short/long, compliance flags, vol context optionnel — mais reste muet sur breakdown et régime structure.
4. **Le mockup B2B-mock est le plus riche** mais contient ~3 champs aspirationnels (`scenarios`, `key_levels.support_zone`/`resistance_zone`, `key_levels.structural_invalidation`) **non implémentés** côté algo.
5. **Le `position_multiplier` composé (régime × news)** est calculé et stocké interne mais **jamais exposé** — c'est un champ pertinent pour broker B2B (sizing) absent des contrats.
6. **Compliance UE 2024/2811** : le code respecte la formulation (`SetupDirection` au lieu de `BUY`/`SELL`, disclaimers obligatoires, edge_claim=False par défaut). v1-REST utilise encore `OPEN_LONG` en violation — à déprécier.

---

## Hors mission (à confirmer si on doit en parler en Parties B/C/D)

Les sous-systèmes suivants existent dans le repo mais sont **hors périmètre** de la mission « ce que l'indicateur dit au client ». Pas de fiche A pour eux. Si tu veux les inclure en discussion B/C/D, dis-le.

- **LLM Narrative Engine** (`src/intelligence/llm_narrative_engine.py`) — exclusion explicite mission.
- **RAG** (`src/intelligence/rag/`) — exclusion explicite mission.
- **Audit hash chain** (`src/audit/`) — porte sur la traçabilité opérationnelle, pas l'info indicateur.
- **Cross-asset correlation** (`src/intelligence/cross_asset_correlation.py`) — exposé via routes admin, pas via signal.
- **Notification queue** (`src/intelligence/notification_queue.py`) — plumbing.
- **Stylized facts checks** (`src/intelligence/stylized_facts.py`) — diagnostic interne.

---

## ✅ Partie A validée par utilisateur (2026-05-16)

Granularité ~73 fiches conservée. Champs aspirationnels conservés. Compliance + télémétrie conservées dans le périmètre.

---

## Partie B — Justification : pourquoi chaque information existe

Cette partie répond aux 4 questions méthode pour **chaque catégorie** de la Partie A (pas chaque fiche — sinon 73 × 4 = 292 paragraphes illisibles). Pour les catégories où des fiches divergent, je précise.

> **Convention** : « Trader cible » = trader retail discrétionnaire FR utilisant XAU M15 (profil ICP eval_25). Tout besoin formulé l'est de son point de vue.

---

### B.1 — Identité du signal et métadonnées (A.1)

**Besoin trader** : savoir *quelle analyse il regarde* et *si elle est encore valide* (timestamp, expiration). Sans ça, deux analyses peuvent paraître identiques alors qu'elles décrivent deux setups différents.

**Logique d'étage** : produites au plus tard possible (au moment de la création du signal stable, post-state-machine), pour que l'identifiant identifie *un événement publié*, pas une intention transitoire qui pourrait avorter en ARMING. C'est aussi pour ça que `signal_id` est **déterministe SHA-1** (A.1.1) : un même setup re-évalué pendant ARMING produit le même ID, le client ne reçoit pas N IDs pour 1 signal.

**Décision éclairée** : ouvrir / ignorer / archiver. Le timestamp + expiration permet d'ignorer un signal périmé sans le rouvrir.

**Ce que ça ne dit PAS** : aucune info sur la qualité du signal — un signal expiré peut avoir été excellent, un signal frais peut être faible. L'identité est neutre.

---

### B.2 — Structure de marché SMC / ICT (A.2)

**Besoin trader** : voir *où est la structure* (tendance / cassure / retest) et *à quel point la structure est défendable* (taille FVG, force OB). Le trader ICT discrétionnaire passe l'essentiel de sa préparation à ça manuellement.

**Logique d'étage** : produit en premier (étage 2 du pipeline) parce que **tout le reste — score, niveaux, état — dépend du sens structurel**. Si BOS = 0, le pipeline coupe court (`confluence_detector.py:236`). C'est le gating amont qui économise CPU sur 95 % des barres.

**Décision éclairée** : *est-ce que la structure justifie de regarder le reste ?* Pour le trader, le signe de `BOS_SIGNAL` est la première chose qu'il vérifie. Pour le système, c'est la condition d'entrée du scoring.

**Ce que ça ne dit PAS** :
- `BOS_SIGNAL ≠ 0` ne signifie pas « ça va monter/descendre » — c'est juste que la structure récente est cassée dans ce sens.
- Aucune information de **profondeur** : pas de niveaux de liquidity sweep, pas de break/equal highs/lows, pas de premium/discount zones. Le moteur SMC est minimaliste vs la doctrine ICT complète.
- `OB_STRENGTH_NORM` est ATR-normalisé mais **pas validé empiriquement** comme prédicteur — l'eval_03 (note 4.5/10) reproche que l'OB est défini comme « bougie engulfing seule », sans validation de displacement institutionnelle propre.

**⚠️ Honnêteté brute** : `BOS_BREAK_LEVEL`, bornes OB et `BOS_RETEST_ARMED` existent en interne mais **ne sortent pas du pipeline** vers le client. Le trader voit le résultat (« cassure + retest OK ») dans le narratif, jamais les niveaux exacts. C'est une perte d'information décisionnelle (le trader expert voudrait voir le `BOS_BREAK_LEVEL` pour décider de placer une limite vs une marché).

---

### B.3 — Score de confluence et décomposition (A.3)

**Besoin trader** : *avoir un sentiment quantifié* de la qualité du setup, surtout pour les traders qui n'ont pas une décennie d'intuition. Le score 0-100 est l'équivalent algorithmique de la « note » que met mentalement un trader expérimenté.

**Logique d'étage** : étage 3 (après SMC, avant niveaux). C'est ici parce que le scoring nécessite **toutes** les composantes (structure + régime + news + volume) — il ne peut pas être plus en amont. Le **breakdown** (A.3.4) est généré simultanément au score, pas reconstruit après.

**Décision éclairée** :
- **Score brut → décision filtre**. Tier ≥ STANDARD ⇒ je regarde. < 25 ⇒ poubelle direct.
- **Breakdown → décision conviction**. Si le score 65 vient à 50 % du régime et 0 du news, je sais que c'est un signal *régime-driven sans validation news* — différent d'un score 65 réparti uniformément.
- **Conviction label bucketée** (v2) → décision UX : la surface compacte (Telegram) montre « STRONG » sans révéler 67.4 (protection IP).

**Ce que ça ne dit PAS** :
- ⚠️ **Le score N'EST PAS une probabilité de gain.** L'eval_02 (rapport interne) a mesuré sa corrélation avec le P&L réel : Pearson −0.023 sur 7 ans XAU. Le score mesure la **convergence des conditions techniques**, pas la rentabilité. La confluence calibration en cours essaie de le transformer en proba via isotonic regression / logistic L1 (eval_05_09_refresh, `src/intelligence/scoring/`).
- Le tier interne (`PREMIUM`/`STANDARD`/`WEAK`) et le label v2 (`weak`/`moderate`/`strong`/`institutional`) **n'ont pas les mêmes seuils** (55 vs 60 pour strong). Risque de confusion si exposés en parallèle.
- Le breakdown est exposé seulement en B2B-mock, pas en B2C v2 — le client B2C voit `conviction_label="STRONG"` sans savoir d'où ça vient. C'est probablement la **plus grosse perte d'information décisionnelle** du contrat v2.

---

### B.4 — Niveaux de prix (A.4)

**Besoin trader** : entrer où, mettre le stop où, viser où. C'est *littéralement* l'info qu'il cherche en priorité — le reste du signal est validation.

**Logique d'étage** : étage 3 (dans `ConfluenceDetector`) parce que les niveaux dépendent du choix d'ATR (naïf vs forecast), lui-même choisi en fonction du régime vol. Trop en amont, on ne peut pas mixer ATR naïf et forecast. Trop en aval, on doit re-passer les données.

**Décision éclairée** :
- Trader risk-averse : taille de position = `% capital / |entry - stop|`. Le SL pose le risque par unité.
- Trader skill-based : valide ou écarte le RR ratio (refuse si < 1.5, etc.).
- Trader scalper : utilise `take_profit` comme target dur ; trader swing : utilise `invalidation` (structurel) comme exit alternatif au SL ATR.

**Ce que ça ne dit PAS** :
- ⚠️ **Le SL est ATR-mécanique, pas structurel.** Le code utilise `entry ± 2×ATR` (ou × 3 en vol haute) — c'est paresseux versus ICT où on placerait le SL juste sous l'OB ou le swing low. Le trader ICT expert va *recalculer son SL manuellement* en regardant le narratif.
- ⚠️ **`levels.invalidation` est déclaré v2 mais vide en prod** (A.4.5). `key_levels.support_zone/resistance_zone` (B2B-mock) sont **aspirationnels**, non implémentés (A.4.6). Le client B2B voit du `null` ou se contente du narratif en prose.
- Les niveaux **ne tiennent pas compte du spread**. Pour XAU 0.30 spread, sur un SL à 2×ATR ≈ 10 pips, le spread coûte 3 % du risque par trade. Non modélisé.

---

### B.5 — Direction du setup (A.5)

**Besoin trader** : haussier ou baissier ? Question binaire, mais format compliance-critique.

**Logique d'étage** : étage 3 (dérive de `BOS_SIGNAL`), puis mapping compliance en étage 6 (`from_v1_signal`). Le mapping `LONG → BULLISH_SETUP` est volontairement décalé en aval pour découpler le moteur quant (qui pense en termes long/short) de la couche réglementaire (qui pense en termes éducatif/non-prescriptif).

**Décision éclairée** : trader → ouvre un trade côté X. Surface B2B broker → propage au client final en respectant le wording compliance.

**Ce que ça ne dit PAS** :
- ⚠️ **v1-REST utilise encore `OPEN_LONG` / `OPEN_SHORT`** (A.5.3) qui est une violation UE 2024/2811 (verbes prescriptifs). Tant que v1 n'est pas déprécié, l'API publique principale émet du wording non conforme. C'est un risque de compliance latent.
- `NEUTRAL` (v2) inclut deux cas distincts : « pas de structure cassée » et « score insuffisant ». Le client ne peut pas distinguer.

---

### B.6 — Volatilité prévisionnelle (A.6)

**Besoin trader** : dimensionner SL/TP et savoir *à quoi s'attendre* dans la prochaine fenêtre (vol expansion → cible plus rapide / vol contraction → range probable).

**Logique d'étage** : étage 4, calculé une fois par signal puis injecté dans `ConfluenceDetector` pour sizing SL/TP. Pourquoi pas plus en amont ? Parce que le forecast vol consomme features (HAR-RV, HMM, calendar, diurnal) qui dépendent de l'OHLCV et des news — bien après le SMC.

**Décision éclairée** :
- `vol_regime="high"` ⇒ trader réduit la taille, prudent sur la mèche.
- `forecast_atr` >> `naive_atr` ⇒ vol expansion attendue ⇒ trader peut élargir manuellement son TP.
- `confidence_lower`/`upper` ⇒ trader institutionnel sait que la prévision a un intervalle, donc peut hedger ou attendre confirmation.

**Ce que ça ne dit PAS** :
- ⚠️ **L'intervalle de confiance vol (TCP, A.6.4) n'est PAS exposé au client** — alors qu'il est calculé. C'est le **plus gros gâchis** de valeur informationnelle du système. Le trader voit `regime="normal"` mais pas « vol prévue 8.7 ± 1.5 pips ».
- `is_fallback=true` (A.6.7) signale une dégradation mais le client ne le voit pas — il croit avoir un forecast HAR-RV alors que c'est un ATR brut.
- Le `regime_state` (`low/normal/high`) est sur l'**axe variance des rendements**, pas sur l'axe directionnel — un régime « high » peut être *high volatility ranging* ou *high volatility trending*. Le label seul est ambigu.

---

### B.7 — Régime macro / structure (A.7)

**Besoin trader** : « est-ce que la marche est trending ou choppy ? » — détermine s'il prend du momentum ou du mean-reversion.

**Logique d'étage** : étage 5. Le régime structurel est calculé en parallèle de la vol, mais consommé en composante `regime` (poids 25 %) — c'est la plus grosse composante du score. Une décision de gating distincte (`RegimeFilter`, `RegimeGate`) drop ≈60 % des barres avant scoring (eval_05_09_refresh).

**Décision éclairée** :
- Trader macro-aware : `regime_hmm="trend_bullish"` confirme sa thèse fondamentale → conviction +.
- Système : `RegimeGate.decision="BLOCK"` après cp_prob spike ⇒ pas de nouvelle entrée.

**Ce que ça ne dit PAS** :
- ⚠️ **Le régime structurel (uptrend/downtrend/ranging) n'est PAS exposé v2** — seulement la vol regime. Le client B2C ne sait jamais si le marché est trending ou ranging. C'est porté uniquement en prose dans le narratif.
- ⚠️ **BOCPD `cp_prob`, `jump_ratio`, `expected_run_length`** (A.7.3) — ces 3 signaux academic-grade sont calculés mais **jamais exposés**. C'est l'un des deltas techniques les plus pointus du système (Bayesian Online Changepoint Detection, Barndorff-Nielsen & Shephard bipower variation) — invisible.
- `session_label` (A.7.5) — exposé seulement B2B-mock. Le trader B2C ne sait pas si on est en session NY ou Asia, alors que c'est crucial pour XAU.

---

### B.8 — News et calendrier économique (A.8)

**Besoin trader** : « il y a un FOMC dans 30 min, je devrais arrêter ? » — gestion event-driven.

**Logique d'étage** : étage 6 (consommé en composante `news` du score, poids 20 %). Plus en aval que SMC parce que la news API est plus latente.

**Décision éclairée** :
- `news_blackout_active=true` ⇒ trader ne prend pas de nouvelle position.
- `news_position_multiplier=0.5` ⇒ trader réduit sa taille.
- Sentiment ⇒ confirmation / opposition de la direction.

**Ce que ça ne dit PAS** :
- ⚠️ **Le sentiment score, la confidence et le multiplicateur news ne sont pas exposés** — seul `news_blackout_active` est exposé en B2B-mock, et même pas en v2. Le trader voit « blocked » ou « passed », pas la nuance.
- Le calendrier source (ForexFactory) est **gris commercialement** (eval_29 compliance) — l'usage commercial est tolérance, pas droit. Risque licence latent.
- L'algo ne dit pas **quel** événement bloque (NFP / FOMC / CPI), juste que le blackout est actif. Trader ne peut pas raisonner sur la nature du risque.

---

### B.9 — État du signal et cycle de vie (A.9)

**Besoin trader** : « le signal est-il encore valide ? est-il en train d'arriver / actif / expiré ? »

**Logique d'étage** : étage 7 (dernier). Le state machine sépare la **détection** (étages 1-6) de la **publication** (étage 7) — un signal détecté n'est pas forcément publié immédiatement (phase ARMING avec confirmation bars). C'est ce qui empêche les flip-flops sur petites variations.

**Décision éclairée** :
- `state="BUY"` + `bars_in_state=3` + `bars_remaining=49` ⇒ signal frais, encore beaucoup de marge.
- `last_exit_reason="target_reached"` (historique) ⇒ track-record positif.
- `cooldown_bars_remaining=15` ⇒ pas de nouveau signal avant 15 barres = ~3h45 sur M15.

**Ce que ça ne dit PAS** :
- ⚠️ **`bars_in_state`, `bars_remaining`, `cooldown_bars_remaining`, `confirmation_progress`, `last_exit_reason`** — **TOUS internes**. Le client voit seulement « current signal » sans contexte temporel. Pour un trader qui veut savoir « ce signal a-t-il déjà tourné une heure ? », c'est invisible.
- L'historique v1 expose `outcome` et `pnl_pips` — mais pas `entered_at_bar` ni le narratif d'origine, donc reconstruire un track-record auditable est compliqué.
- Pas de notion de **partial fill** ou **trailing stop** — le state machine est binaire (active / inactive).

---

### B.10 — Couverture conformelle (A.10)

**Besoin trader** : *quantifier son incertitude*. Le trader pro veut savoir « ce signal est solide à 90 % de confiance » vs « 50 % de confiance ». Le trader retail ne le formule pas comme ça mais ressent la différence (sentiment de qualité).

**Logique d'étage** : étage 8 (wrapper post-scoring). Calculé après le scorer ML (logistic L1 ou LGBM, `src/intelligence/scoring/`) pour produire un intervalle de prédiction calibré (split conformal ou ACI online).

**Décision éclairée** : aujourd'hui, **rien** côté trader — c'est utilisé uniquement en gating algorithmique (`should_reject`). Théoriquement, un trader pro pourrait :
- Ajuster sa taille en fonction de la `width()` (intervalle large ⇒ taille petite).
- Skip si `lower ≤ breakeven`.

**Ce que ça ne dit PAS** :
- ⚠️ **AUCUN champ conformal n'est exposé au client.** C'est la garantie mathématique la plus pointue de tout le système (couverture marginale guaranteed under exchangeability, garantie distribution-free) — totalement invisible commercialement.
- L'`empirical_coverage` (drift monitoring) — invisible aussi. Le client ne sait jamais si les intervalles « tiennent leur promesse » empiriquement.
- C'est ce qu'on appelle dans l'industrie « **buried alpha** » : la valeur technique réelle est dans la cave, le client voit le rez-de-chaussée.

---

### B.11 — Narratif et provenance (A.11)

**Besoin trader** : *comprendre pourquoi*. Le trader retail moderne (jeune, FR, ICT-curious) ne fait plus confiance à un « BUY 2350 » sans contexte. Il veut une explication causale.

**Logique d'étage** : étage 9 (post-tout). C'est volontairement le dernier étage parce que le narratif consomme **toutes** les autres infos. Mission exclut le LLM lui-même, mais les champs structurés portent la sémantique : `narrative_short` est UX-first compact, `narrative_long` est webapp/B2B verbeux, `sources_cited` est la chaîne d'audit.

**Décision éclairée** :
- Trader débutant : lit le narratif court, prend ou pas.
- Trader sophistiqué : lit `narrative_long` + vérifie `sources_cited` pour valider la méthode.
- Broker B2B : propage le narratif intégral pour défendre la décision face à son client.

**Ce que ça ne dit PAS** :
- ⚠️ **`sources_cited` est obligatoire en Phase 2B mais aujourd'hui vide en prod** (Phase 2B pas démarrée — `is_paper_demo=True` par défaut). Le narratif est généré sans chaîne d'audit citationnelle.
- Le narratif est **du LLM** — sujet à hallucination même avec template/cascade Haiku/Sonnet. Les eval_05 et eval_05_05 implementation listent les garde-fous (token vocabulary, banned terms `score_calibration.contains_forbidden_token`) mais pas l'audit ex-post du contenu.
- `scenarios` (B2B-mock A.11.6) — **aspirationnel, pas produit**. Le trader B2C ne reçoit jamais d'alternative path.

---

### B.12 — Conformité réglementaire (A.12)

**Besoin trader** : aucun direct — c'est l'opérateur qui en a besoin. Mais le trader voit le disclaimer obligatoire et les flags `is_paper_demo` / `edge_claim`.

**Logique d'étage** : middleware (geo-block en amont, disclaimer en aval). Les flags `edge_claim=False` et `is_paper_demo=True` sont des **engagements honnêteté codifiés** : tant que l'edge n'est pas validé empiriquement post-A1 (verdict 2026-05-01), le système le déclare explicitement.

**Décision éclairée** :
- Pour l'opérateur (toi) : disclaimer text + jurisdiction_blocked déterminent la livrabilité. Si client US ⇒ pas de signal.
- Pour le trader : `is_paper_demo=true` lui dit « ne misez pas réel sur ce signal, c'est une démo ».

**Ce que ça ne dit PAS** :
- ⚠️ Le client retail ne lit pas le disclaimer (comportement standard) — la compliance est *par construction*, pas *par lecture*.
- `edge_claim=False` est techniquement honnête mais commercialement saboteur : un client qui voit « pas d'edge validé » dans un signal qu'il paie comprend mal la valeur (« mais alors pourquoi je paie ? »). Tension marketing/compliance non résolue.

---

### B.13 — Télémétrie scanner (A.13)

**Besoin trader** : *vérifier que le système tourne*. Surtout en SaaS : « ai-je raté un signal parce que le scanner était down ? ».

**Logique d'étage** : monitoring transverse (pas dans le pipeline signal). Exposé via `/health` endpoint.

**Décision éclairée** :
- Trader : refresh la page si `scanner_running=false`.
- Opérateur : monitor le `cache_hit_rate`, le `kill_switch_level`, le ratio `signals_dropped_by_regime_filter / signals_generated`.

**Ce que ça ne dit PAS** :
- Aucune métrique de **latence** par signal (de barre OHLCV à publication) — invisible côté client.
- Aucune métrique de **track-record** (PF cumulatif, hit rate, MDD) côté `/health` — il faudrait passer par `/api/v1/signals/history` et agréger soi-même.

---

## Partie C — Différenciation : en quoi ces informations se démarquent

### C.1 — Cartographie des concurrents

Le mapping concurrentiel pour un indicateur retail XAU/FX se découpe en 5 strates. Pour chacune, je note ce qu'elle expose comme information, son prix typique, et son public.

#### C.1.a Indicateurs techniques classiques (RSI/MACD/Bollinger/ATR/ICT manuel)

- **Surface** : TradingView (gratuit + Premium $14-60/mo), MT4/5, ThinkOrSwim, NinjaTrader. ~100 % des plateformes les fournissent.
- **Information exposée** : valeurs scalaires des indicateurs (RSI=58, MACD histogram, etc.). Pas d'interprétation. Pas de score. Pas de narratif. Le trader fait l'agrégation mentalement.
- **Public** : 100 % des traders retail (point d'entrée). Souvent insuffisant seul → ajout de scripts.
- **Limites** : aucune confluence quantifiée, aucune incertitude, aucun gating automatique.

#### C.1.b SMC / ICT auto sur TradingView (Pine scripts payants)

- **Surface** : TradingView marketplace. **LuxAlgo Premium ($59/mo)**, **Smart Money Concepts (LuxAlgo, $19-39/mo)**, **TradingFinder Premium ($15-25/mo)**, **ChartPrime ($25-50/mo)**, **Inside Out Trading ICT**, scripts open-source variés.
- **Information exposée** : détection automatique BOS / CHOCH / OB / FVG, parfois retest, parfois liquidity sweeps. **Aucun score 0-100, aucune décomposition pondérée, aucune confidence, aucune narrative LLM, aucun gating régime/news.**
- **Public** : traders ICT qui ne veulent plus tracer à la main.
- **Limites** : pas de méthode d'agrégation transparente, pas de track-record audité (la plupart sont des scripts visuels), confluence laissée à l'œil du trader.

#### C.1.c Canaux Telegram de signaux VIP

- **Surface** : Telegram, Discord. Tarifs $30-200/mo (souvent rugpull / track-record falsifié). Quelques noms réputés : **Forex GDP**, **Learn 2 Trade**, **1000pipBuilder**, mais la qualité est extraordinairement variable.
- **Information exposée** : « BUY XAU 2350, SL 2340, TP 2370 » + parfois une ligne de justification. Aucune transparence de méthode, track-record auto-rapporté.
- **Public** : trader débutant qui veut « copier ».
- **Limites** : opacité totale, conformité ≈ 0, dépendance personne, conformité UE 2024/2811 quasi-systématiquement violée.

#### C.1.d Plateformes IA / quant retail

- **Surface** :
  - **Trade Ideas** (~$120-230/mo) — scanners IA pour US equities, alertes time-sensitive, peu de narratif.
  - **TrendSpider** ($39-79/mo) — patterns auto + backtest visuel, multi-asset.
  - **AltIndex** (~$30-50/mo) — quant pour stocks/crypto, alt data + sentiment, pas FX/XAU primaires.
  - **Composer** (variable) — algo trading no-code (stocks/ETFs).
  - **Tickeron** ($60-100/mo) — pattern recognition + AI confidence scores.
- **Information exposée** : signaux + parfois confidence (Tickeron donne %), pas de SMC explicite, pas de structure ICT, pas d'incertitude calibrée (calibration souvent absente).
- **Public** : US retail / quant-curious. Couverture XAU faible, couverture FX inégale.
- **Limites** : confidence affichée mais pas calibrée (% est pseudo-empirique), narratif souvent absent, focus stocks pas FX/Or.

#### C.1.e ChatGPT-wrapper « AI Trading » apps

- **Surface** : nombreux apps mobiles, plug-ins TradingView. **Stock GPT**, **TradingView AI assistants**, **Finchat**, etc.
- **Information exposée** : narratif LLM riche, mais pipeline analytique **derrière le narratif quasi-nul** (souvent juste « RSI overbought + MACD bearish »). Pas de SMC, pas de confluence quantifiée, pas de conformal prediction.
- **Public** : retail attiré par « AI ».
- **Limites** : narratif sans fondation quantitative — risque hallucination élevé, pas de gating systématique.

#### C.1.f Bloomberg/Refinitiv/Reuters institutionnels

- **Surface** : Bloomberg Terminal ($24k/an), Refinitiv Eikon ($22k/an), Refinitiv MarketPsych, Trading Central B2B.
- **Information exposée** : signaux quantitatifs cleans, calendrier news propriétaire, sentiment NLP, breaking news API. Pas d'ICT/SMC (institutionnel utilise d'autres frameworks).
- **Public** : desks institutionnels.
- **Limites** : inaccessible retail (prix + complexité), pas SMC retail-friendly.

---

### C.2 — Tableau comparatif principal

Pour chacune des **13 catégories** d'information de la Partie A, voici ce que produisent les concurrents vs Smart Sentinel AI. Lecture par ligne : *quelle info, qui la fournit comment*.

| # | Information | Techniques classiques | SMC/ICT Pine (LuxAlgo etc.) | Signaux Telegram | Plateformes IA retail | ChatGPT wrappers | Smart Sentinel AI |
|---|---|---|---|---|---|---|---|
| 1 | **Identité + cycle de vie** (A.1, A.9 partiels) | n/a (pas de signal) | n/a | Numéro de message Telegram | UUID + état actif | n/a | UUID déterministe SHA-1 + state machine 3 états + cooldown |
| 2 | **Détection structure SMC** (A.2.1–A.2.8) | ❌ (manuel) | ✅ détection auto BOS/CHOCH/OB/FVG | ❌ (pas exposé) | ❌ (rarement SMC) | ⚠️ mention texte sans détection | ✅ détection auto + retest gate dur + 9 champs structurés (interne) |
| 3 | **Indicateurs techniques** (A.2.9) | ✅ valeurs scalaires | ✅ overlay | ❌ | ✅ | ⚠️ mention | ✅ produits + utilisés en composante `momentum` |
| 4 | **Score de confluence 0-100** (A.3.1) | ❌ | ❌ (pas de méthode d'agrégation) | ❌ | ⚠️ Tickeron : confidence % (pas calibré) | ❌ | ✅ score 0-100 + tier + label bucketé |
| 5 | **Décomposition pondérée** (A.3.4) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 8 composantes × {valeur brute, score pondéré, poids, raisonnement texte} — **B2B uniquement** |
| 6 | **Niveaux entry/SL/TP** (A.4) | ❌ (laissé au trader) | ⚠️ parfois suggestion | ✅ niveaux explicites (méthode opaque) | ✅ | ⚠️ texte | ✅ niveaux ATR-mécaniques (basique) |
| 7 | **Invalidation structurelle** (A.4.5) | ❌ | ⚠️ visualisation, pas champ | ❌ | ❌ | ❌ | ⚠️ **déclaré v2, vide en prod** |
| 8 | **Direction setup compliance UE 2024/2811** (A.5.2) | ❌ (non concerné) | ❌ (souvent « BUY/SELL ») | ❌ violation systématique | ❌ | ❌ | ✅ `BULLISH_SETUP`/`BEARISH_SETUP`/`NEUTRAL` |
| 9 | **Volatility forecast** (A.6.1) | ❌ (ATR brut seulement) | ❌ | ❌ | ❌ | ❌ | ✅ HAR-RV + LGBM + hybrid + multiplicateurs (diurnal/calendar/HMM) |
| 10 | **Vol confidence interval** (A.6.4) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ TCP intervalle **interne, non exposé** ⚠️ |
| 11 | **Régime de marché (HMM)** (A.7.1) | ❌ | ❌ | ❌ | ⚠️ certains mentionnent "trend/range" sans formalisme | ❌ | ✅ HMM 3 états + posterior probability |
| 12 | **Changepoint Bayesian (BOCPD)** (A.7.3) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ academic-grade, **interne, non exposé** ⚠️ |
| 13 | **Jump ratio (Barndorff-Nielsen bipower)** (A.7.3) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ bipower variation, **interne, non exposé** ⚠️ |
| 14 | **Session label** (A.7.5) | ⚠️ via horloge | ⚠️ | ❌ | ⚠️ | ❌ | ✅ explicite (B2B-mock uniquement) |
| 15 | **News blackout** (A.8.1) | ❌ | ❌ | ❌ | ⚠️ rare | ❌ | ✅ -30/+60 min gate dur |
| 16 | **Sentiment news + multiplicateur sizing** (A.8.2-4) | ❌ | ❌ | ❌ | ⚠️ Trade Ideas, AltIndex (stocks) | ⚠️ générique | ✅ calculé, **majoritairement interne** ⚠️ |
| 17 | **State machine (HOLD/BUY/SELL + transitions)** (A.9.1-3) | ❌ | ❌ | ❌ | ⚠️ état actif/inactif | ❌ | ✅ machine 5 états internes / 3 publics + hystéresis + cooldown + arming |
| 18 | **Outcome historique + P&L pips** (A.9.5) | ❌ | ❌ | ⚠️ auto-rapporté, non audité | ⚠️ certains (Composer, Tickeron) | ❌ | ✅ persistance SQLite |
| 19 | **Conformal prediction interval** (A.10) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ split conformal + ACI online, **interne, non exposé** ⚠️ |
| 20 | **Narratif court** (A.11.1) | ❌ | ❌ | ⚠️ une ligne « pourquoi » | ❌ | ✅ très verbeux | ✅ ≤400 chars compliance-safe |
| 21 | **Narratif long + sources citées RAG** (A.11.2, A.11.4) | ❌ | ❌ | ❌ | ❌ | ⚠️ verbeux sans sources | ⚠️ **Phase 2B pas démarré** — sources_cited vide en prod |
| 22 | **Multi-langue (FR/EN/DE/ES)** (A.11.3) | n/a | ⚠️ majoritairement EN | ⚠️ canal-dépendant | ⚠️ majoritairement EN | ⚠️ majoritairement EN | ✅ 4 langues, persistance par chat_id |
| 23 | **Scénarios alternatifs** (A.11.6) | ❌ | ❌ | ❌ | ❌ | ⚠️ parfois | ⚠️ **aspirationnel B2B-mock, pas produit** |
| 24 | **Compliance UE 2024/2811** (A.12) | n/a | ❌ violation typique (verbes prescriptifs) | ❌ violation systématique | ❌ | ❌ | ✅ geo-block + disclaimers + edge_claim=False par défaut |
| 25 | **Track-record auditable / paper-demo flag** (A.12.4) | n/a | ❌ | ❌ falsifié | ⚠️ Composer expose backtests | ❌ | ⚠️ `is_paper_demo=True` par défaut — track-record démo non encore livré |
| 26 | **Télémétrie scanner (uptime, hit rate cache, kill switch)** (A.13) | n/a | n/a | n/a | ⚠️ status pages génériques | n/a | ✅ `/health` détaillé (admin) |

**Légende** : ✅ produit & exposé · ⚠️ produit partiellement ou non exposé · ❌ absent.

---

### C.3 — Classification 🟢 / 🟡 / 🔴 par information

Je reprends chacune des 26 lignes du tableau et tranche : *est-ce un vrai différenciateur défendable ?*

**Méthode de classification** :
- 🟢 **Fort et défendable** : SS fait quelque chose que la concurrence ne fait *pas du tout* OU fait nettement mieux, ET c'est techniquement difficile à copier en < 6 mois, ET ça apporte une valeur décisionnelle réelle (≠ « ça fait pro mais sert à rien »).
- 🟡 **Modéré** : SS le fait mieux, mais la concurrence s'en approche, ou c'est facilement copiable, ou la valeur décisionnelle est partielle.
- 🔴 **Faux différenciateur** : tout le monde le fait, ou bien la concurrence le fait *aussi bien*, ou bien le marketing nous le fait dire mais c'est creux.

| # | Information | Classif | Justification brutale |
|---|---|---|---|
| 1 | Identité + cycle de vie | 🔴 | Tout système avec persistance fait ça. Aucun client n'achète parce qu'on a un UUID déterministe. |
| 2 | Détection structure SMC (BOS/CHOCH/OB/FVG) | 🔴 | **LuxAlgo et 50 scripts Pine font ça depuis 2020.** L'auto-détection SMC est devenue commodity. |
| 3 | Indicateurs techniques classiques | 🔴 | Toutes les plateformes les fournissent depuis 1995. |
| 4 | Score de confluence 0-100 | 🟡 | Concept différenciant (peu de produits SMC scorent quantitativement), MAIS son **pouvoir prédictif est ~0** (Pearson −0.023 mesurée eval_02) → différenciateur **marketing**, pas analytique. Renforce avec la calibration isotonic en cours. |
| 5 | Décomposition pondérée (8 composantes) | 🟢 | **Rare en retail.** L'explainability « mini-SHAP plat » est défendable : narratif LLM peut halluciner, le breakdown est traçable. **MAIS : exposé seulement en B2B-mock**, invisible en B2C → différenciateur potentiel **non livré**. |
| 6 | Niveaux entry/SL/TP ATR | 🔴 | ATR-based SL/TP est standard partout. |
| 7 | Invalidation structurelle | 🟡 | Champ déclaré, vide en prod → opportunité (peu de concurrents distinguent SL ATR et invalidation structurelle), mais aujourd'hui **non livré**. |
| 8 | Direction setup compliance UE | 🟢 | **Quasi-personne ne respecte UE 2024/2811** (revue eval_29). Différenciateur réglementaire défendable, surtout B2B brokers européens qui doivent propager. Difficile à copier sans refonte produit chez le concurrent. |
| 9 | Volatility forecast (HAR-RV + LGBM) | 🟢 | **Personne en retail XAU ne fait ça.** HAR-RV academic, LGBM meta-learner, multiplicateurs orthogonaux — c'est du quant. Différenciateur technique fort. Latence HAR ~50ms ⇒ scalable. |
| 10 | Vol confidence interval (TCP) | 🟢 | Différenciateur fort **techniquement**, mais **invisible client** ⇒ valeur marketing à débloquer. Voir Partie D. |
| 11 | Régime HMM 3 états + posterior probability | 🟢 | Quasi-personne en retail SMC ne fait ça. HMM est cœur du quant institutionnel, retail n'a pas l'outillage. Défendable. |
| 12 | BOCPD changepoint | 🟢 | **Bayesian Online Changepoint Detection = academic state-of-the-art** (Adams & MacKay 2007). Aucun concurrent retail n'expose ça. Différenciateur **technique fort, invisible aujourd'hui**. |
| 13 | Jump ratio Barndorff-Nielsen | 🟢 | Bipower variation = papier fondateur 2004. Académique. Aucun retail ne fait. Même situation que BOCPD. |
| 14 | Session label | 🔴 | TradingView affiche les sessions depuis 2010. Commodity. |
| 15 | News blackout -30/+60 | 🟡 | Beaucoup de signaux Telegram revendiquent « news filter », rarement implémenté proprement. Différenciateur **modéré** car la concurrence le revendique sans le livrer. |
| 16 | Sentiment news + multiplicateur sizing | 🟡 | Sentiment news est dans Trade Ideas/AltIndex pour stocks. Pour XAU/FX en retail, c'est rare. Modéré. |
| 17 | State machine + hystéresis + cooldown | 🟡 | Hystéresis évite les flips d'un score qui oscille autour du seuil — pertinent. Mais pas exposé client (interne), donc invisible. |
| 18 | Outcome + P&L historique | 🟡 | Composer/Tickeron exposent du backtest visuel ; nous, on persiste l'outcome live mais le track-record paper-demo n'est pas encore packagé. Potentiel modéré. |
| 19 | Conformal prediction interval | 🟢 | **Garantie mathématique distribution-free.** Personne en retail. Différenciateur technique **fort, invisible**. C'est le candidat n°1 à exposer. |
| 20 | Narratif court ≤400 chars | 🔴 | ChatGPT-wrappers font du narratif depuis 2023. Notre force vient de la **fondation quantitative** sous le narratif, pas du narratif lui-même. |
| 21 | Narratif long + sources RAG | 🟡 | RAG sourcé (papers académiques + reports LBMA/BIS) serait un vrai 🟢, mais **Phase 2B pas démarrée**. Potentiel non livré. |
| 22 | Multi-langue (FR/EN/DE/ES) | 🟢 | **FR-first est rare en retail XAU.** Wedge ICP eval_25 = « XAU SMC FR-first ». Défendable commercialement. |
| 23 | Scénarios alternatifs | 🔴 | Aspirationnel, pas livré. Ne compte pas tant que c'est mockup. |
| 24 | Compliance UE 2024/2811 complète | 🟢 | Geo-block + disclaimers multilingues + edge_claim flag = **conformité par construction**. Différenciateur défendable, surtout B2B brokers UE. |
| 25 | Track-record auditable / paper-demo | 🟡 | Le flag existe, mais le track-record démontré publiquement n'est pas livré. Tout dépend de la livraison Phase 2B. |
| 26 | Télémétrie scanner | 🔴 | `/health` endpoint est standard partout. Pas un différenciateur produit. |

#### Récapitulatif

| Statut | Nombre | Lignes |
|---|---:|---|
| 🟢 **Fort et défendable** | **9** | 5, 8, 9, 10, 11, 12, 13, 19, 22, 24 (10 — j'en compte 10 en relisant) |
| 🟡 **Modéré** | **9** | 4, 7, 15, 16, 17, 18, 21, 25 |
| 🔴 **Faux différenciateur** | **8** | 1, 2, 3, 6, 14, 20, 23, 26 |

> *Note* : 9 vs 10 en 🟢 selon comptage — je liste 10 forts (5, 8, 9, 10, 11, 12, 13, 19, 22, 24).

**Observation critique** : sur les 10 différenciateurs 🟢 forts identifiés, **5 sont invisibles aujourd'hui au client** (lignes 5 partielle, 10, 12, 13, 19). C'est-à-dire que **la moitié de l'edge technique défendable est buried**. C'est le constat le plus important du document.

---

## Partie D — Synthèse stratégique : le récit de valeur

### D.1 — Phrase de positionnement

> **Smart Sentinel AI livre, pour le trader retail XAU/FX francophone, le seul indicateur SMC qui assemble la détection ICT à de la mécanique quant institutionnelle (HMM, conformal coverage, BOCPD, HAR-RV) — dans un récit narratif compliance-UE.**

Reformulation courte : *« Le seul indicateur SMC qui pense comme un desk institutionnel et qui parle français. »*

### D.2 — Les 3 différenciateurs 🟢 à pousser commercialement

**① Conformal coverage calibrée + intervalles de prédiction (lignes 10, 19)**
- *Preuve code* : `src/intelligence/conformal_wrapper.py` (`SplitConformalScorer`, `AdaptiveConformalScorer` avec ACI Gibbs-Candès), `src/intelligence/conformal/mondrian.py`.
- *Pourquoi unique* : garantie mathématique distribution-free, aucune autre offre retail XAU/FX n'expose ça. C'est le candidat n°1 à débloquer côté client (voir D.5).
- *Risque* : invisible aujourd'hui → impossible à monétiser tant que pas exposé en `InsightSignalV2`.

**② Régime macro-quant : HMM 3 états + BOCPD + jump ratio + HAR-RV vol forecast (lignes 9, 11, 12, 13)**
- *Preuve code* : `src/intelligence/regime_classifier.py` (HMM), `src/intelligence/bocpd.py` (Adams-MacKay 2007), `src/intelligence/regime_gate.py` (BOCPD + Barndorff-Nielsen bipower), `src/intelligence/volatility_forecaster.py` (HAR-RV + multiplicateurs).
- *Pourquoi unique* : la concurrence retail s'arrête à « le marché est trending ou pas ». Ici on a un appareil régime academic-grade complet. **C'est ce qui justifie le terme « institutional » dans le positionnement.**
- *Risque* : aujourd'hui, seul le `regime_label` simple (low/normal/high) sort en v2 — le reste invisible.

**③ Compliance UE 2024/2811 + FR-first + 4 langues + edge_claim honnête (lignes 8, 22, 24)**
- *Preuve code* : `src/api/insight_signal_v2.py:47` (SetupDirection sans BUY/SELL), `src/intelligence/score_calibration.py` (forbidden tokens), middleware geo-block (sprint W1-W3), `ComplianceMeta.edge_claim=False` (eval_29).
- *Pourquoi unique* : la quasi-totalité des signaux Telegram et des scripts SMC sont **non-conformes** UE 2024/2811. Pour un broker B2B européen qui doit propager le wording à ses clients, on devient le seul fournisseur compliance-safe. Le wedge ICP eval_25 (« XAU SMC FR-first ») exploite ça commercialement.
- *Risque* : v1-REST utilise encore `OPEN_LONG`/`OPEN_SHORT` → à déprécier urgemment, sinon l'argument de positionnement est fragilisé.

### D.3 — Les 3 faiblesses 🔴 à corriger ou à NE PAS mettre en avant

**❶ « AI-powered » comme argument générique (ligne 20)**
- En 2026, tous les wrappers ChatGPT prétendent ça. Notre narratif court (≤400 chars) est *correct* mais ne fait pas la différence. La valeur ne vient pas du narratif, elle vient de **la fondation quantitative que le narratif explique**.
- *À ne PAS dire* : « notre IA analyse pour vous ». *À dire* : « notre narratif est fondé sur N composantes quantitatives auditables ».

**❷ Détection SMC auto (lignes 2, 6)**
- LuxAlgo et 50 scripts Pine font ça pour $19-59/mo depuis 2020. Si on revend Smart Sentinel à $79/mo en mettant en avant « auto-détection BOS/CHOCH/OB/FVG », on est dans le confluence Premium 1:1 du marché → race to the bottom.
- *À ne PAS dire* : « détection automatique des structures Smart Money ». *À dire* : « score quantitatif explicable + régime + incertitude calibrée au-dessus de la détection SMC ».

**❸ Track-record démonstration paper (lignes 18, 25)**
- Le flag `is_paper_demo=True` existe mais le track-record paper-demo n'est pas livré publiquement. Tant qu'il ne l'est pas, on ne peut pas le mettre en avant sans risquer un retour de flamme (vérité = « démonstration en devenir »).
- *À ne PAS dire* : « track-record auditable ». *À dire* : rien encore. Sortir Phase 2B avant.

**Plus un faux différenciateur bonus à éviter** : « 8 composantes pondérées de confluence » (ligne 4). Le score brut a **pouvoir prédictif ~0** mesuré (Pearson −0.023, eval_02). Tant que la calibration isotonic/L1 (eval_05_09_refresh) n'a pas remonté ça empiriquement, dire « score 75/100 = haute probabilité de gain » est faux. *À dire* : « score quantifie la convergence des conditions, pas la probabilité de gain » (formulation actuelle v2 sur `conviction_label` correcte).

### D.4 — Profil client cible (où l'info a le plus de valeur)

**Trader retail discrétionnaire ICT, FR ou EN-fluent, 30-55 ans, 3-10 ans d'expérience, XAU primaire :**
- Connaît BOS/CHOCH/OB/FVG manuellement → comprend l'output de A.2 et l'apprécie comme accélérateur.
- Vit la frustration des signaux Telegram opaques → valorise la décomposition (A.3.4) et le narratif sourcé (A.11.2-4).
- A souffert d'un signal news-bombed → valorise A.8.1 (news blackout).
- *Bête noire pour lui* : signaux marketing « 90 % de winrate » non audités. Le flag `edge_claim=False` honnête (A.12.3) est un facteur de **confiance**, pas de méfiance, pour ce profil mûr.

**Profil secondaire (B2B) : broker européen mid-tier** (IC Markets, Pepperstone, FXCM EU, Vantage, Exness) qui veut ajouter une couche « market insights » à son interface — c'est l'ICP secondaire eval_26 « pivot B2B-API brokers ». Pour eux, **la conformité UE 2024/2811 + multi-langue + scénarios alternatifs (si livrés) + breakdown** sont la valeur principale, pas le signal lui-même.

**Profils où l'indicateur a peu de valeur** :
- **Débutant absolu** : le narratif et le label conviction sont compréhensibles, mais le breakdown et la confluence le perdent. Mieux servi par des produits éducation pure (BabyPips, Investopedia).
- **HF / desk institutionnel** : nos features quant sont vraies mais l'enveloppe (Telegram, webapp signal-by-signal) ne correspond pas à leur workflow API/Bloomberg.

### D.5 — Recommandations : quelles informations renforcer pour creuser l'écart

Trois recommandations en ordre de ROI décroissant (le plus important d'abord) :

**🥇 Exposer la couverture conformelle au client v2 (ligne 19)**

C'est **le levier #1** : c'est techniquement déjà calculé (`ConformalInterval` complet), il suffit d'ajouter ces champs au `InsightSignalV2` :
```python
class UncertaintyContext(BaseModel):
    conformal_lower: float
    conformal_upper: float
    coverage_alpha: float       # 0.10 = 90% confidence
    n_calibration: int          # taille du set de calib
    empirical_coverage: Optional[float]  # ACI tracking
```
- *Effort* : ~4-8 h (ajout au schéma + renderer + tests).
- *Impact* : on devient **le seul indicateur retail XAU avec garantie mathématique d'incertitude exposée**. Argument « probabilité calibrée » défendable B2B brokers.

**🥈 Exposer la décomposition des composantes en v2 (ligne 5)**

Le breakdown 8-composantes existe dans `ConfluenceSignal.components` mais n'est exposé qu'en B2B-mock. Le porter en v2 :
```python
class ComponentBreakdown(BaseModel):
    name: str
    weight: float
    score_pct: float       # weighted_score / weight × 100
    reasoning: str
```
- *Effort* : ~6 h.
- *Impact* : différencie immédiatement de LuxAlgo et tous les concurrents SMC (aucun n'expose la déconstruction). Permet l'audit côté trader.
- *Considération IP* : exposer les poids révèle la méthode → arbitrage à faire. Compromis : exposer `name`, `score_pct`, `reasoning` (qualitatif), masquer `weight`.

**🥉 Livrer Phase 2B narrative-first : RAG sourcé + track-record paper (lignes 21, 25)**

Le contrat `Source` (A.11.4) existe et `is_paper_demo` est codé. Mais la Phase 2B (selon eval roadmap_2026_2027) n'est pas démarrée. Sans la livraison RAG :
- `sources_cited` reste vide → le narratif est de la prose non auditée → l'argument différenciant tombe.
- `is_paper_demo=True` par défaut sans track-record publié → confusion client (« paper de quoi ? »).
- *Effort* : 80h selon le plan agent Aisha.
- *Impact* : c'est le pari narrative-first post-verdict A1. Sans ça, on n'a que le mockup et zéro preuve.

**Recommandation secondaire (à ne pas faire urgent)** :
- ❌ Ne PAS ajouter le `regime_label_structurel` (uptrend/downtrend/ranging) en v2 si Phase 2A n'est pas validée — risque survente. Sentinel doit garder son honnêteté `edge_claim=False`.
- ⚠️ Compléter `levels.invalidation` côté algo (calculer le niveau structurel ≠ SL ATR) → champ déclaré et vide est pire que pas de champ. ~8h.
- ⚠️ Renforcer le calendrier news : remplacer ForexFactory (zone grise commerciale) par une source pro (Bloomberg consensus si budget, sinon TradingEconomics API) → l'argument compliance s'écroule si on est sur du scraping gris.

---

## Conclusion en une ligne

> **Smart Sentinel AI a 10 différenciateurs 🟢 défendables, mais 5 sont enterrés** (conformal, BOCPD, jump ratio, vol CI, breakdown caché en B2B). **Le ROI commercial le plus élevé est d'exposer ce qui existe déjà**, pas de construire du nouveau.

