# Indicateur visuel descriptif — Analyse de faisabilité

> **Mission 100 % lecture seule.** Aucun code produit modifié. Ce document est le seul livrable.
> **Date** : 2026-06-08 · **Branche** : `audit/indicateur-visuel-faisabilite` (issue de `institutional-overhaul`)
> **Auteur** : Claude (remit faisabilité) · **Founder** : Loukmane Bessam
> **Positionnement de référence** : niveau 1.5 strict, `edge_claim=false`
> (`docs/governance/decisions/2026-05-27_pivot_positioning_audit.md`).
> **Contrainte cardinale** : l'indicateur est **descriptif, pas prédictif**. Il dessine les
> structures déjà détectées sur le prix passé/présent et **s'arrête à la dernière bougie clôturée**.

---

## 0. Verdict global

> **✅ FAISABLE dans le périmètre descriptif, sans toucher au moteur ni au mapper.**

Trois faits décisifs ressortent de la lecture du code :

1. **La géométrie est déjà là.** Depuis le correctif niveaux du 2026-06-08 (F1/F2/F3,
   `FIX_NIVEAUX_AVANT_APRES_2026_06_08.md`), le modèle `MarketReading` porte les **vrais**
   niveaux de prix et bornes de zones (BOS/CHOCH `level`, OB/FVG `level_high`/`level_low`),
   chacun horodaté. Aucun nouveau calcul moteur n'est nécessaire pour tracer les overlays.
2. **Le contrat servi à `/app` est déjà descriptif par construction.** Le pipeline par défaut
   de l'assembler renvoie `confluence_signal=None` (`market_reading_assembler.py:84-132`) :
   `MarketReading` **décrit un état, pas un setup** — aucun score 0-100, aucune cible de prix,
   aucun `valid_until`, aucune flèche de biais. Les champs prédictifs problématiques
   (`hmm_posterior`, `bocpd_changepoint_prob`, `forecast`, conformal CI 95 %) vivent dans un
   **autre** modèle (`InsightSignalV2`) qui **n'est pas servi** à `/app`. La surface est donc
   conforme au niveau 1.5 *par défaut*.
3. **Le seul vrai manque est le flux de bougies OHLC vers le front.** Les bougies sont bien
   persistées en SQLite (`candles_cache`) mais aucun endpoint ne les expose. Il faut **2 ajouts
   purement additifs** (une méthode de lecture + une route REST), tous deux en lecture seule,
   qui ne touchent ni le moteur ni le mapper.

**Bonus de dé-risquage** : un **prototype `ReadingChart.tsx` (286 lignes) existe déjà** en WIP
non commité (voir §H). Il prouve empiriquement que Lightweight Charts rend candles + niveaux +
zones avec nos données. La Phase 1 n'est donc pas théorique.

**Réserve unique** : le risque n'est pas technique, il est **éditorial** — l'audit qualité
(`descriptive_quality_assessment.md`) impose de ne PAS sur-promettre les horizons (médiane
d'invalidation 1-2 bars, pas « 4h ») et de ne pas qualifier les OB d'« institutionnels ».

---

## Plan phasé recommandé (en tête)

| Phase | Contenu | Effort | Avant bêta ? |
|---|---|---|---|
| **Phase 1 — 80 % de la valeur** | Chandeliers + lignes de niveau horizontales (BOS/CHOCH/retest) + badge « dernière bougie clôturée HH:MM » | **~3,5–5 j** | ✅ **Oui** |
| **Phase 2 — zones** | Boîtes Order Block / Fair Value Gap (overlay DOM ou primitive) + bande de fenêtre d'événement | **~3–5 j** | ⚠️ Partiel (boîtes oui ; bande événement reportable V1.1) |
| **Hors-portée V1** | Flux tick temps réel, toute projection/cône/cible, multi-actifs au-delà de XAU/EUR, TF hors M15/H1/H4 | — | ❌ Non |

Détail action par action en **§F**.

---

## A. Inventaire des données disponibles

Modèle : `src/intelligence/market_reading_schema.py` (`MarketReading`, `schema_version 2.x`).
Mapper : `src/intelligence/market_reading_mappers.py`. Moteur : `SmartMoneyEngine`.

### A.1 Géométrie par structure

| Structure | Niveau / bornes prix | Horodatage | État | Champs (fichier) |
|---|---|---|---|---|
| **BOS** | ✅ `level` (float, vrai niveau cassé) | ✅ `broken_at` (datetime) | **PRÉSENT** | `BOSRecent` — `market_reading_schema.py` |
| **CHOCH** | ✅ `level` (partage `BOS_BREAK_LEVEL`) | ✅ `broken_at` | **PRÉSENT** | `CHOCHRecent` |
| **Order Block** | ✅ `level_high` + `level_low` (zone réelle) | ✅ `created_at` | **PRÉSENT** | `OrderBlock` (+ `status`, `tested`) |
| **Fair Value Gap** | ✅ `level_high` + `level_low` (bornes 3-bougies) | ✅ `created_at` | **PRÉSENT** | `FairValueGap` (+ `status`, `tested`) |
| **Retest** | ✅ `level` | ✅ `started_at` | **PRÉSENT** | `RetestInProgress` (+ `type`) |
| **Phase / régime** | label texte (`trend`, `volatility_observed`, `market_phase`, `mtf_confluence`) | implicite (bougie courante) | **PRÉSENT (label)** | `MarketReadingRegime` — pas de posterior HMM exposé ✅ |
| **Événement / news** | timestamps d'événement (`scheduled_at`, `published_at`), `time_to_event_min`, `impact`, `currency` | ✅ par événement | **PARTIEL** | `NewsUpcoming` / `NewsJustPublished` — voir §A.3 |

> **Conséquence directe** : pour BOS/CHOCH/Retest (lignes) et OB/FVG (boîtes), **100 % de la
> géométrie nécessaire est déjà dans le contrat servi**. Aucun sérialiseur additif requis pour
> les overlays de structure. C'est le résultat du correctif F1/F2/F3 (avant : `BOS.level` =
> prix de clôture proxy ; après : 0/60 proxy, match moteur 5/5).

### A.2 Historique OHLC en SQLite (fenêtre de N bougies)

- **Table** `candles_cache` — `src/storage/candles_cache_store.py` :
  `(instrument, timeframe, ts, open, high, low, close, volume)`, PK composite
  `(instrument, timeframe, ts)`, index `(instrument, timeframe, ts DESC)`. DB par défaut
  `./data/candles.db` (override `CANDLES_DB_PATH`).
- **Persistance confirmée** : l'assembler upsert les bougies à chaque génération
  (`market_reading_assembler.py`, `DEFAULT_LOOKBACK = 200`).
- **Tirer N bougies par combo** : la requête est triviale (clé + index déjà en place) **mais la
  méthode n'existe pas encore** : seul `get_latest_candle()` est implémenté. → manque additif
  (§B).
- **Timestamps** : ISO-8601, **UTC** (les fournisseurs convertissent `utc=True`). Le store
  readings normalise explicitement en `…Z` ; `candles_cache` stocke via `.isoformat()` (UTC sans
  suffixe `Z` systématique — cosmétique, à normaliser côté sérialiseur).

### A.3 Données d'événement (timing du blackout)

- **Source** : CSV calendrier (`CALENDAR_PATH`) → `EconomicEvent`
  (`src/agents/news/economic_calendar.py`) : `scheduled_time` (datetime), `impact`, `currency`,
  `actual/forecast/previous`.
- **Blackout** : fenêtre **30 min avant / 30 min après** (HIGH) calculée à la volée dans
  `is_within_window()` (`news_analysis_agent.py:112-113`). ⚠️ **La doc client dit 30/60** —
  incohérence déjà relevée (`descriptive_quality_assessment.md` §2.6) ; le **code fait 30/30**.
- **Exposé dans `MarketReading`** : la **liste** d'événements avec leurs horodatages
  (`scheduled_at` / `published_at`) + minutes au prochain événement. **PARTIEL** : il n'y a
  **pas** de borne `blackout_start`/`blackout_end` ni de flag « blackout actif » dans
  `MarketReading`. Le flag `news_blackout_active` existe seulement dans `InsightSignalV2`
  (non servi à `/app`). La bande horaire d'événement est donc soit à **reconstruire côté front**
  (`scheduled_at ± 30 min`), soit à exposer via un sérialiseur additif.

### A.4 Horodatage de clôture de bougie (« dernière bougie clôturée »)

- `MarketReadingHeader.candle_close_ts` (datetime) + `close_price` portent **exactement**
  l'information nécessaire au badge « dernière bougie clôturée HH:MM ».
- `expected_last_candle_close(timeframe, now)` (`market_reading_assembler.py:60-78`) calcule la
  frontière de la dernière bougie pleinement close (UTC) — base honnête et déjà testée.

---

## B. Écart données existantes → données nécessaires

| # | Manque | Nature du correctif | Touche moteur/mapper ? | Effort |
|---|---|---|---|---|
| B1 | Pas de lecture « N dernières bougies » | Ajouter `CandlesCacheStore.get_last_n_candles(instrument, timeframe, n)` — `SELECT … ORDER BY ts DESC LIMIT n` puis renverser | ❌ Non (lecture pure) | **Petit ~0,5 j** |
| B2 | Pas d'endpoint OHLC | Ajouter route `GET /api/candles?instrument&timeframe&limit` (réutilise le store) | ❌ Non (additif, sur le modèle de `market_reading.py`) | **Petit ~0,5 j** |
| B3 | Front : aucun fetch de bougies, pas de type `Candle` côté contrat | Hook `useCandles()` + type TS ; le type `MarketReading` front n'a pas de tableau OHLC | ❌ Non | **Petit ~0,5–1 j** |
| B4 | `lightweight-charts` absent de `package.json` commité | `npm i lightweight-charts` + fichier `NOTICE` (Apache 2.0) | ❌ Non | **Petit ~0,25 j** |
| B5 | Bande de fenêtre d'événement non bornée dans `MarketReading` | (a) reconstruire front `scheduled_at ± 30 min`, **ou** (b) sérialiseur additif exposant `blackout_start/end` déjà calculés | ❌ Non (option b = additif lecture seule) | **Moyen ~1–1,5 j** |

> **Confirmation discipline** : **aucun** de ces manques ne nécessite de modifier le moteur de
> détection (`strategy_features.py`) ni le mapper (`market_reading_mappers.py`). Les overlays de
> structure consomment des champs **déjà présents** ; les seuls ajouts sont des **lectures
> additives** (méthode store + route + hook front). Cohérent avec la règle « 1 noyau, N surfaces ».

---

## C. Conformité positionnement (niveau 1.5 strict)

**✅ Une vue purement descriptive est rendable avec les données actuelles, sans dérive.**

- `MarketReading` ne contient **par défaut** aucun champ prédictif : pas de score, pas de cible,
  pas de `valid_until`, pas de posterior. Le pipeline renvoie `confluence_signal=None`.
- Le graphe se borne naturellement à `candle_close_ts` (dernière bougie close) → **aucune
  projection forward possible** depuis ce contrat.
- Les overlays (lignes de niveau + boîtes) sont des **objets posés sur le passé**, pas des
  vecteurs directionnels.

### Risques de dérive de positionnement à surveiller (signalés, pas corrigés)

| Risque | Détail | Garde-fou recommandé |
|---|---|---|
| **R-pos-1 — champs `InsightSignalV2`** | Ce modèle parallèle porte `forecast_atr_pips`, `confidence_interval` « 95 % », `target_1/2`, `valid_until_utc`, `hmm_posterior`, `bocpd_changepoint_prob`. Audit : non-supportables (couverture conformelle 39-51 %, ECE HMM 0.54). | **Ne jamais brancher `InsightSignalV2` dans le chart.** Le chart ne lit que `MarketReadingStructure`. |
| **R-pos-2 — horizons de validité** | L'audit montre médiane recross BOS = 2 bars, mitigation FVG = 1 bar. Afficher/laisser entendre « zone valide 4h » serait faux. | Si une légende temporelle est ajoutée : « événement tactique court terme », **pas** de durée fixe. |
| **R-pos-3 — sémantique OB** | OB = simple engulfing non filtré par impulsion (P=0.30 vs référence). | Étiqueter « Order Block (engulfing) », **jamais** « bloc institutionnel ». |
| **R-pos-4 — marqueur de biais** | Tentation d'ajouter une flèche directionnelle au dernier point. | **Interdit** : un marqueur ne décrit qu'un événement passé (ex. « BOS ↑ ici »), jamais une intention future. |

---

## D. Faisabilité de rendu (TradingView Lightweight Charts)

Lib open-source **Apache 2.0** — on fournit nos propres données (aucun flux propriétaire requis).

### D.1 Natif vs plugin

| Élément | Support | Mécanisme |
|---|---|---|
| Chandeliers | ✅ **Natif** | `CandlestickSeries` + `series.setData()` |
| Lignes de niveau horizontales (BOS/CHOCH/Retest) | ✅ **Natif** | `series.createPriceLine({price, title, lineStyle})` |
| Marqueurs ponctuels (ex. point d'événement) | ✅ **Natif** | `series.setMarkers()` |
| **Boîtes / zones OB & FVG** | ⚠️ **NON natif** | Deux voies : (a) **overlay DOM** — `<div>` positionnés via `series.priceToCoordinate()`, resynchronisés sur pan/zoom/resize ; (b) **primitive v5** (`ISeriesPrimitive`) dessinée sur canvas |
| Bande verticale d'événement | ⚠️ Non natif | Même logique d'overlay/primitive sur l'axe temps |

> Le prototype existant (§H) implémente la **voie (a)** : overlay DOM resynchronisé via
> `subscribeVisibleLogicalRangeChange` + `ResizeObserver`. Fonctionne, mais demande un soin
> particulier sur la perf et le re-rendu (voir §G).

### D.2 Attribution Apache 2.0

- Conserver l'attribution TradingView : le prototype active `attributionLogo: true` dans
  `layout`. Ajouter un fichier `NOTICE` (déjà esquissé en WIP) listant la licence Apache 2.0 de
  `lightweight-charts`. **Exigence légale légère mais obligatoire.**

### D.3 Composants `/app` à modifier (insertion colonne centrale)

| Fichier | Rôle | Action |
|---|---|---|
| `webapp/components/market-reading/MarketReadingCard.tsx` | Carte centrale | Insérer `<ReadingChart>` (entre header et `MarketPhasePanel`) |
| `webapp/components/app/ReadingColumn.tsx` | Colonne centrale (états vide/loading/erreur) | Câbler états « graphique indisponible » |
| `webapp/components/app/AppWorkspace.tsx` | Grille 3 colonnes + `useMarketReading(pollMs:60_000)` | Ajouter le fetch des bougies |
| `webapp/lib/market-reading/hooks.ts` | Hook de données | Ajouter `useCandles()` (ou étendre) |
| `webapp/types/market-reading.ts` | Types contrat | Ajouter type `Candle` |
| `webapp/package.json` + `webapp/NOTICE` | Dépendance + attribution | `lightweight-charts` + NOTICE |

### D.4 Responsive / mobile

- Layout desktop : grille `md:grid-cols-[240px_1fr_360px]` ; mobile (<768px) :
  `MobileWorkspace.tsx` à onglets (Marchés / Lecture / Chat).
- Le prototype gère déjà la hauteur responsive (`h-[280px] sm:h-[340px]`) + `aria-label`. Le
  chart doit vivre dans l'onglet « Lecture » en mobile. i18n `fr/en/de/es` (`next-intl`) pour les
  libellés d'overlay.

---

## E. Cadence de rafraîchissement (sans flux tick temps réel)

Le temps réel est **exclu** (positionnement + free tier Twelve Data). Le modèle existant est
déjà « pull à la clôture », ce qui colle parfaitement :

1. **Backend** : `MarketReadingScheduler` (`src/intelligence/scheduler.py`) tick **60 s**,
   régénère chaque combo *actif* dès qu'une **nouvelle bougie a clôturé** depuis la lecture
   stockée ; auto-stop après **24 h** sans accès. Mode hybride : 1er accès paresseux (endpoint)
   puis régénération continue.
2. **Frontend** : `useMarketReading` poll **60 s** (`pollMs: 60_000`) sur
   `GET /api/market-reading`. Le futur `useCandles()` peut **partager la même cadence** (ou ne
   re-fetch que si `candle_close_ts` a changé — économie de bande passante, cohérent avec le free
   tier).
3. **Honnêteté temporelle** : afficher `header.candle_close_ts` formaté « Dernière bougie close :
   HH:MM (UTC/locale) ». C'est le garde-fou contre la contradiction « en direct / bougie
   ancienne » : on **n'affiche jamais** une bougie en formation, et le badge dit explicitement
   l'âge de la donnée.

> **Recommandation** : déclencher le re-fetch des bougies **uniquement** sur changement de
> `candle_close_ts`, pas à chaque tick — limite la charge Twelve Data (8 req/min free tier).

---

## F. Plan d'implémentation phasé

### Phase 1 — Chandeliers + lignes de niveau + marqueurs (le 80 % de la valeur) — **tenable avant bêta**

| Action | Tag | Effort |
|---|---|---|
| F1-a `CandlesCacheStore.get_last_n_candles()` | Petit | 0,5 j |
| F1-b Route `GET /api/candles` (additive) | Petit | 0,5 j |
| F1-c Hook `useCandles()` + type `Candle` front | Petit | 0,5–1 j |
| F1-d Installer `lightweight-charts` + `NOTICE` Apache | Petit | 0,25 j |
| F1-e Finaliser `ReadingChart` : candles + price lines BOS/CHOCH/Retest (prototype existant à câbler sur données réelles, retirer la dépendance `mockReadings`) | Moyen | 1–1,5 j |
| F1-f Insertion dans `MarketReadingCard` + états vide/erreur + responsive mobile + badge `candle_close_ts` | Petit-Moyen | 1 j |
| **Sous-total Phase 1** | | **~3,5–5 j** |

### Phase 2 — Zones OB/FVG + bande événement — **partiellement reportable V1.1**

| Action | Tag | Effort |
|---|---|---|
| F2-a Boîtes OB/FVG (overlay DOM du prototype à durcir, ou primitive v5) | Moyen | 1,5–2,5 j (DOM) / 3–4 j (primitive) |
| F2-b Bande de fenêtre d'événement (reconstruction front `± 30 min` **ou** sérialiseur additif `blackout_start/end`) | Moyen | 1–1,5 j |
| **Sous-total Phase 2** | | **~3–5 j** |

**Recommandation de découpe bêta** : livrer **Phase 1 complète** + **F2-a (boîtes)** pour la
bêta (le prototype les fait déjà), et **reporter F2-b (bande événement) en V1.1** — c'est le
seul morceau qui demande une décision (reconstruction front vs ajout sérialiseur) et dont l'audit
montre une couverture faible (~1,5 % des bars).

### Hors-portée (à refuser explicitement)

- Flux **tick temps réel** (WebSocket) — exclu positionnement + free tier. **Hors-portée.**
- Tout **cône de prévision, ligne forward, cible de prix, flèche de biais projetée, score
  0-100**. **Hors-portée (interdit niveau 1.5).**
- Multi-actifs au-delà de **XAUUSD/EURUSD** et TF hors **M15/H1/H4** (périmètre V1 de l'endpoint).
  **Hors-portée V1.**
- Re-câblage de `InsightSignalV2` (champs prédictifs). **Hors-portée + risque positionnement.**

---

## G. Risques & pièges

| # | Risque | Gravité | Mitigation |
|---|---|---|---|
| G1 | **Dérive prédictive** (cf. §C) — réintroduire score/cible/forecast via `InsightSignalV2` ou une flèche de biais | 🔴 Élevée (positionnement) | Le chart ne lit QUE `MarketReadingStructure`. Revue de code dédiée sur tout overlay directionnel. |
| G2 | **Horizon sur-promis** — laisser entendre persistance « 4h » | 🟠 Moyenne | Légendes sans durée fixe ; aligner sur audit (médiane 1-2 bars). |
| G3 | **Boîtes par overlay DOM** — désync au pan/zoom/resize, coût de re-rendu | 🟠 Moyenne | `subscribeVisibleLogicalRangeChange` + `ResizeObserver` (déjà dans le prototype) ; envisager primitive v5 si perf insuffisante. |
| G4 | **Encombrement visuel** — beaucoup d'OB/FVG (audit : 13 810 OB XAU sur l'historique) | 🟠 Moyenne | Le contrat ne renvoie que les zones *actives* à la bougie courante ; plafonner l'affichage (ex. n dernières) + `faded` sur `status != active`. |
| G5 | **Timezone** — `candles_cache` en UTC sans `Z` systématique vs front local | 🟡 Faible | Normaliser en `…Z` dans le sérialiseur `/api/candles` ; afficher TZ explicite. |
| G6 | **Contradiction « en direct / bougie ancienne »** | 🟡 Faible | Badge `candle_close_ts` obligatoire ; jamais de bougie en formation. |
| G7 | **Dépendance Lightweight Charts** (API v4→v5 : primitives, `addSeries(CandlestickSeries)`) | 🟡 Faible | Lib Apache 2.0 mature, self-hosted ; figer la version ; isoler dans `ReadingChart`. |
| G8 | **Charge free tier Twelve Data** (8 req/min) si re-fetch bougies à chaque tick | 🟡 Faible | Re-fetch conditionné au changement de `candle_close_ts` (§E). |
| G9 | **Incohérence doc blackout 30/60 vs code 30/30** | 🟡 Faible (mais à corriger) | Si bande événement implémentée, utiliser **30/30** (vérité code) et corriger la doc. |

---

## H. Note d'état — prototype WIP existant (hors base canonique)

Lors du setup, le working tree portait un **prototype non commité** (travail founder de la
branche `feat/app-shell-chart-pin-error-states`), qui a voyagé jusqu'à la branche d'audit. Il
**n'existe sur aucune branche commitée** (`institutional-overhaul` : 0) :

- `webapp/components/app/ReadingChart.tsx` (286 lignes) — chandeliers + price lines
  BOS/CHOCH/Retest + boîtes OB/FVG (overlay DOM), dark/light, responsive, `aria-label`,
  `attributionLogo: true`. **Dépend de** `lightweight-charts` (dans `package.json` *stashé*) et
  d'un `Candle` issu de `lib/mockReadings.ts` (**mock, stashé**) → ne compile pas en l'état actuel
  du tree (dépendances séparées).
- `webapp/lib/market-reading/pins.ts`, `webapp/NOTICE` (attribution Apache), tests associés —
  non commités.

**Implication faisabilité** : la Phase 1 (et F2-a) est **déjà prototypée et démontrée**. L'essentiel
du travail restant est de **remplacer la source mock par les bougies réelles** (F1-a→c) et de
**commiter proprement** ce qui est aujourd'hui du WIP épars. Cela conforte le verdict ✅.

> ⚠️ **Discipline respectée** : ce rapport n'a **pas** modifié, commité ni supprimé ce WIP. Pour
> le restaurer après l'audit : revenir sur `feat/app-shell-chart-pin-error-states` puis
> `git stash pop` (stash `WIP chart-pin avant audit indicateur visuel 2026-06-08`, = `stash@{0}`).
> Le `package.json`, `package-lock.json`, `hooks.ts` et `mockReadings.ts` y sont préservés.

---

## Annexe — fichiers lus (lecture seule)

**Backend** : `market_reading_schema.py`, `market_reading_mappers.py`, `market_reading_assembler.py`,
`scheduler.py`, `src/api/routes/market_reading.py`, `src/storage/candles_cache_store.py`,
`src/storage/market_readings_store.py`, `src/api/signal_store.py`, `src/api/insight_signal_v2.py`,
`src/agents/news/economic_calendar.py`, `src/agents/news_analysis_agent.py`,
`src/environment/strategy_features.py`.
**Frontend** : `webapp/app/[locale]/app/page.tsx`, `AppWorkspace.tsx`, `ReadingColumn.tsx`,
`MarketReadingCard.tsx`, `MobileWorkspace.tsx`, `webapp/lib/market-reading/hooks.ts`,
`webapp/types/market-reading.ts`, `webapp/components/app/ReadingChart.tsx` (prototype WIP).
**Docs** : `FIX_NIVEAUX_AVANT_APRES_2026_06_08.md`, `descriptive_quality_assessment.md`,
`FRONTEND_OBSERVATIONS_2026_06_05.md`, `2026-05-27_pivot_positioning_audit.md`.
*(Non trouvé au chemin annoncé : `docs/audits/RD_AUDIT_SYNTHESE_PRIORISEE.md` — absent du dépôt.)*

---

*Fin du rapport. Aucun code produit modifié. Seul livrable : ce fichier.*
