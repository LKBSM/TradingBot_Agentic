# AUDIT NW-7 — Page de publication : livrer la maquette en entier

Branche : `feat/nw-7-page-publication-complete` (depuis `origin/main` @ `b713909`, PR #119 NW-6).
Maquette de référence : `docs/design/reference-publication.html` — **déjà présente sur `origin/main`
et byte-identique** à `mia-page-news-v5.html` (29 689 o). Rien à pousser, DOM lu en entier.
Statut : **DIAGNOSTIC — aucun code écrit. En attente du GO.**

> ⚠️ Correction de cadrage (identique à NW-6). La prémisse de la mission — « NW-5 et NW-6 ont
> livré la page sans la courbe ni les quatre questions » — décrit un **état antérieur**. Sur le
> vrai `origin/main` à jour, la page est bien plus complète que la mission ne le suppose :
> **la courbe est bâtie, 3 des 4 questions sont implémentées et câblées pour `us_cpi`, les
> révisions sont modélisées, les 6 organismes ont leurs liens sources, 17 fiches pédagogiques
> réelles existent, un seul bloc d'avertissement.** Ce qui reste est **étroit et précis** :
> (1) faire *arriver* la donnée jusqu'à l'écran (clé + source de données des mesures),
> (2) la mesure #3 (cycle de vie des zones), (3) `us_jolts` non mesurable/sans liens,
> (4) l'unité de la courbe IPC (indice vs % annuel), (5) révisions *sur la courbe*.

---

## A. LE CHEMIN DE LA DONNÉE, DE BOUT EN BOUT

Il y a **deux flux distincts** vers la page, à ne pas confondre : la **courbe** (valeurs
publiées par l'organisme) et les **quatre questions** (mesures du moteur sur le prix). Ils
ont des sources et des points de blocage différents.

### A.1 — Flux COURBE (les 12 derniers chiffres publiés)

```
catalogue (config/calendar_catalog.json : source=bls, series_code=CUUR0000SA0 / JTS…JOL)
  → build_value_fetcher()               [base_value.py:98]  gaté CALENDAR_VALUES_LIVE=1 (prod: ON)
      └─ BLS wiré SEULEMENT si BLS_API_KEY présent [base_value.py:120]  ← PROD: sync:false, NON POSÉE
  → CalendarService.get_event(event_id) [calendar_service.py:258]
      └─ value_series = series_for(source, series_code)  → BLSValueFetcher.fetch_series(limit=12)
         [bls_values.py:58]  ← IMPLÉMENTÉ depuis NW-6 (12 points mensuels, chronologiques)
  → GET /api/calendar/event/{event_id}  [routes/calendar.py:147]  → CalendarResponse.events[0].value_series
  → useCalendarEvent(eventId) → CurveCard  [CalendarEventDetail.tsx]  rendu SSI value_series.length>0
```

**Où ça s'arrête aujourd'hui — `us_cpi` ET `us_jolts` (les deux `bls`) :**
au maillon **`BLS_API_KEY`**. La clé est déclarée `sync: false` dans `render.yaml:99` → elle
doit être posée **manuellement au dashboard Render**. Absente : `build_value_fetcher` ne wire
pas BLS → `series_for` renvoie `[]` → `value_series` vide → `CurveCard` non rendue. Le code
est complet en amont et en aval ; **seule la clé manque.** (ECB/Eurostat, eux, n'ont pas de
clé et affichent déjà leur courbe.)

**Piège d'unité (à trancher, cf. §B) :** `series_code=CUUR0000SA0` renvoie l'**indice**
(niveau ~3xx, base 1982-84=100), pas le **% de variation annuelle** que montre la maquette
(3,1 % ; plage 2,4–3,4 %). Livrer la courbe brute afficherait des niveaux d'indice, pas des %.

### A.2 — Flux QUATRE QUESTIONS (mesures moteur sur le prix)

```
catalogue (event_key)
  → GET /api/publications/{event_key}/measures  [routes/calendar.py:111]
      └─ market = _MEASURABLE_MARKETS.get(event_key)   {"us_cpi": "XAUUSD"}  ← us_jolts ABSENT → {} vide
  → load_default_measures(event_key, market)  [publication_measures.py:563]
      ├─ lit les DATES de parution :  data/economic_calendar_2019_2025.csv   (config.NEWS_FILTER_CONFIG)
      └─ lit les BOUGIES XAU M15   :  data/XAU_15MIN_2019_2026.csv           (config.HISTORICAL_DATA_FILE)
         ← DEUX CSV GITIGNORÉS. ABSENTS du worktree (data/*.csv vide). Absents → return None.
  → compute_publication_measures()  [publication_measures.py:490]  (PUR ; #1 calm, #2 structure, #4 retour)
  → usePublicationMeasures(eventKey) → QuestionsSection  rendu SSI hasAnyMeasure(measures)
```

**Où ça s'arrête aujourd'hui :**
- **`us_cpi`** : le câblage est **complet** (`_MEASURABLE_MARKETS`, endpoint, 3 mesures, rendu).
  Le blocage est **la source de données** : `load_default_measures` lit **deux CSV gitignorés**
  (`XAU_15MIN_2019_2026.csv`, `economic_calendar_2019_2025.csv`). S'ils ne sont pas présents
  sur le disque `/app/data` en prod, `load_default_measures` renvoie `None` → aucune question.
  **C'est le point le plus probable de l'échec « faute de données » vécu en prod** : le rendu
  existe mais la source (gros CSV non versionné) n'atteint pas le conteneur. À CONFIRMER côté
  prod ; le correctif propre est de **lire depuis `candles.db`** (source chaude prod, Twelve
  Data) au lieu d'un CSV bundlé, ou de provisionner le CSV sur le disque.
- **`us_jolts`** : bloqué **plus haut** — `event_key` **absent de `_MEASURABLE_MARKETS`** →
  l'endpoint renvoie des mesures vides par conception → section jamais rendue. Aucune mesure
  n'est calculée pour JOLTS aujourd'hui.
- **Mesure #3** (cycle de vie des zones) : **différée et absente du schéma** pour *tous* les
  indicateurs (`publication_measures_schema.py` n'a aucun champ zone-lifecycle).

### A.3 — Ce qui est DÉJÀ bâti (à conserver, ne pas réécrire)

| Section maquette | État sur `origin/main` | Emplacement |
|---|---|---|
| A. En-tête (3 heures, passé/futur) | **BÂTIE** | `CalendarEventDetail.tsx` (header inline) |
| B. Courbe 12 valeurs (point à venir vide/pointillé) | **BÂTIE** | `CurveCard` |
| C. 4 questions | **BÂTIE — 3 cartes** (#1/#2/#4 ; #3 différée) | `QuestionsSection`, `publication_measures.py` |
| Bloc « comment lire » | **BÂTI, unique** | `.warn` |
| D. M.I.A | **BÂTIE** (mêmes mesures, pas de recalcul) | `MiaBlock` |
| E. Aller à la source | **BÂTIE — 13 publications, 6 organismes** | `sourceLinks.ts` (`us_jolts` manquant) |
| F. Ce que mesure l'indicateur | **BÂTIE — 17 fiches réelles fr+en** | `PEDAGOGY_FICHES` + `messages/*.json` |
| G. Un seul avertissement | **FAIT** (NW-6 a fusionné les 3) | `.cal-nono` |
| Révisions (initiale vs révisée) | **MODÉLISÉES + PERSISTÉES** | `calendar_cache_store.py` (`actual_initial`, `revised`, `revised_at`) |

---

## B. LA COURBE

- **Clé BLS en prod ?** NON. `BLS_API_KEY` = `render.yaml:99 sync:false` → **variable à créer :
  `BLS_API_KEY`** dans le dashboard Render du service backend. `CALENDAR_VALUES_LIVE=1` est déjà
  posé (`render.yaml:78`). C'est **la seule variable** à poser pour débloquer les courbes BLS
  (IPC, JOLTS, IPP, NFP). (BEA/Census ont leurs propres clés `sync:false`, hors périmètre courbe IPC/JOLTS.)
- **Combien de valeurs historiques une fois la clé posée ?** `fetch_series(limit=12)` demande
  une fenêtre de 2 années civiles et **garde les 12 dernières observations mensuelles**
  (`bls_values.py:58-85`). Donc **12 points** pour IPC, JOLTS, NFP, IPP (indicateurs mensuels).
  Moins de 12 si la série est plus courte : le rendu affiche « ce qui existe » (déjà géré).
- **L'API expose-t-elle la série à la page ?** OUI — `value_series` est peuplé par
  `get_event` → exposé par `/api/calendar/event/{id}` → consommé par `CurveCard`. Rien à ajouter côté transport.
- **Révisions distinguées de la valeur initiale ?** OUI **pour la valeur courante** de la
  parution (`actual_initial` verrouillé, `actual` mis à jour, `revised`/`revised_at`).
  **MAIS PAS sur les 12 points de la courbe** : `SeriesPoint(period, value)` ne porte que la
  dernière valeur publiée (déjà révisée) — pas la valeur d'origine par point. La maquette, elle,
  montre 2 points corrigés avec l'ancienne valeur au survol (§184-185 du DOM). **Écart à combler**
  si on veut les révisions *sur la courbe* : il faut une seconde série « telle que publiée à
  l'origine » (BLS ne l'expose pas simplement — nécessiterait un historique de snapshots).

**Décision requise — unité de la courbe IPC.** La maquette affiche un **% annuel** ; le
`series_code` catalogue renvoie l'**indice**. Trois options (à trancher au GO) :
1. **Garder l'indice** (fidèle à « telle que publiée, sans conversion »), et adapter le libellé
   d'axe → s'écarte visuellement de la maquette (niveaux ~3xx au lieu de 2–3 %).
2. **Série % annuel** : BLS publie aussi la variation sur 12 mois ; on change le `series_code`
   pour la série de % officielle → colle à la maquette, reste « tel que publié ».
3. **Calculer le % 12 mois** depuis l'indice (13 points) → colle à la maquette mais c'est une
   *conversion* (tension avec la règle « sans conversion ni réarrondi »).
**Recommandation : option 2** (variation annuelle publiée par BLS = ce que la maquette montre,
et reste une valeur *publiée*, non recalculée).

---

## C. LES QUATRE MESURES DU MOTEUR — CHIFFRAGE

Rappel infrastructure (vérifiée) : moteur de détection **rejouable en lecture seule et pur**
(`build_enriched_frame` → `collect_zones`/`collect_structure_events`, sans look-ahead,
`READING_LOGIC_VERSION=5`) ; référence horaire diurne déjà calculée
(`_build_hour_reference`, médiane INTERNE, 60 jours sans parution) ; compteur de touches
horodatées **`touch_ats`** mergé sur main (`feat/zone-touch-counter`, `market_reading_mappers.py`).
Stockage profond LB-1 : XAU M15 par défaut **1 mois** dans `candles.db` (insuffisant pour 12
parutions mensuelles → nécessite backfill si on passe du CSV à `candles.db`).

| # | Mesure | Calculable aujourd'hui | Ce qui manque | Où vit le calcul | Coût données |
|---|---|---|---|---|---|
| 1 | Le calme avant | **OUI** (implémentée, testée) | rien (code) ; **la source XAU M15** doit arriver en prod | `publication_measures.py:_compute_calm_before` | 0 requête si CSV/candles présents |
| 2 | Structure à l'instant | **OUI** (rejeu 1200 bougies) | idem #1 | `_compute_structure_state` (+ rejeu moteur) | 0 requête (CPU : replay ~5-10 s 1er appel, caché 1 h) |
| 3 | Cycle de vie des zones | **NON** (différée) | schéma `ZoneLifecycleMeasure` + `_compute_zone_lifecycle` + rendu carte | nouveau, dans `publication_measures.py` | 0 requête (réutilise mêmes bougies) |
| 4 | Retour au calme | **OUI** (implémentée, testée) | idem #1 | `_compute_return_to_calm` | 0 requête |

**Chiffrage données (le cœur de la question) :**
- **Profondeur d'historique** pour 12 parutions d'un indicateur mensuel + ~60 jours de
  référence sans publication : **≈ 13 mois de XAU M15** au minimum (les 12 parutions couvrent
  ~12 mois, les jours de référence sont intercalés). Le CSV bundlé (`XAU_15MIN_2019_2026.csv`,
  2019→2026) **couvre déjà largement** ; `candles.db` par défaut (1 mois) **NON**.
- **Requêtes fournisseur** :
  - via **CSV bundlé** (état actuel du code) : **0 requête** — mais dépend d'un fichier
    gitignoré présent sur le disque prod (risque = l'échec observé).
  - via **`candles.db`** (correctif robuste prod) : **backfill unique** de XAU M15 sur ~13-24
    mois. À ~5 000 bougies/req Twelve Data et ~96 bougies M15/jour : 24 mois ≈ 69 000 bougies
    ≈ **~14 requêtes one-shot**, puis 0 (le warm quotidien entretient). Uniquement M15.
- **Le stockage profond existant couvre-t-il une partie ?** Partiellement : M15 est stocké,
  mais à **1 mois** par défaut → il faut **approfondir M15 à ≥ 13 mois** (édition
  `config/lookback_depths.json`) OU garder la lecture CSV.
- **Coût pour UN indicateur** : les 12 dates de parution + les bougies XAU M15 déjà réunies →
  **0 requête supplémentaire** (mêmes bougies pour #1/#2/#3/#4).
- **Coût par indicateur SUPPLÉMENTAIRE sur le MÊME marché (XAU)** — ex. IPP, JOLTS, NFP sur
  l'or : **0 requête** (mêmes bougies XAU ; il suffit d'ajouter l'`event_key` à
  `_MEASURABLE_MARKETS` et de fournir ses dates de parution). **C'est le vrai levier : un seul
  historique XAU sert tous les indicateurs mesurés sur l'or.**
- **Coût par MARCHÉ supplémentaire** (ex. mesurer sur EUR/USD) : **1 backfill M15** de ce marché.
- **Le compteur de touches horodatées couvre-t-il la mesure #3 ?** **Partiellement.** `touch_ats`
  donne l'horodatage des touches d'une zone → **la moitié « délai avant mitigation »** de #3.
  Il ne donne PAS l'autre moitié : **compter les zones *créées* dans l'heure suivant la parution**
  (il faut rejouer la détection sur [T, T+1 h] et filtrer par `created_at`). Réutilisation réelle
  mais partielle.

---

## D. DÉCOUPAGE PROPOSÉ

Principe (repris de la mission) : « deux questions solides sur un indicateur > quatre
approximations sur tous ». Ordre proposé, du plus haut ROI au plus coûteux :

**Lot 1 — FAIRE ARRIVER LA DONNÉE (débloque la maquette telle quelle, ~0 nouveau calcul).**
  a. Poser `BLS_API_KEY` au dashboard (toi) → courbe IPC/JOLTS.
  b. Trancher l'unité courbe IPC (§B, reco : série % annuel publiée).
  c. Fiabiliser la source des mesures : lire XAU M15 depuis `candles.db` (prod-chaud) avec
     repli CSV, + approfondir M15 à ≥ 13 mois → les **3 questions existantes** de `us_cpi`
     s'affichent réellement en prod, prouvé par un smoke test « mesures non nulles ».
  → Résultat : **courbe + 3 questions réelles pour IPC**, la maquette est déjà à ~85 %.

**Lot 2 — MESURE #3 (cycle de vie des zones) pour `us_cpi`.**
  Schéma `ZoneLifecycleMeasure` + `_compute_zone_lifecycle` (rejeu [T, T+1 h], zones créées,
  délai jusqu'à mitigation via `touch_ats`, répartition en tranches + extrêmes datés) + 4e carte
  + i18n fr/en + test non-régression détection. → **les 4 questions complètes pour IPC.**

**Lot 3 — ÉTENDRE À `us_jolts` (2e indicateur solide, coût données nul).**
  `us_jolts` → `_MEASURABLE_MARKETS` (même marché XAU, 0 requête) + ses dates de parution +
  entrée `sourceLinks.ts` (`us_jolts`, domaine bls.gov) + vérifier fiche pédagogique (déjà réelle).
  → **IPC et JOLTS tous deux complets.**

**Recommandation :** livrer **Lot 1 + Lot 2 sur `us_cpi`** en priorité (la maquette entière,
sur l'IPC), puis **Lot 3** pour JOLTS. Décision au GO : périmètre indicateurs (IPC seul, ou
IPC + JOLTS) et l'option d'unité de courbe.

---

## Reste à écrire / écarts connus (à la sortie du STOP)
- Révisions **sur la courbe** (2e série « telle que publiée à l'origine ») : non trivial, BLS
  ne l'expose pas → à discuter (option : n'afficher les révisions que sur la valeur courante,
  pas sur les 12 points, et l'assumer comme écart maquette).
- Confirmation prod : présence (ou non) de `XAU_15MIN_2019_2026.csv` sur `/app/data`.
- `us_jolts` : sans liens sources ni mesures aujourd'hui (Lot 3).

---

# PARTIE 2 — LIVRÉ (après GO)

Décisions du fondateur : **périmètre = IPC seul (Lots 1+2)** ; **courbe = série % annuel
publiée par BLS**.

## Clé BLS — vérifiée en direct
La clé (`2f4ad4b8…`) est **valide et reconnue** (message BLS « the user with registration key … »)
— PAS un rejet type Census. Elle était au **plafond journalier** (v2 = 500 req/jour) au moment
du test ; se réinitialise chaque jour. D'où le cache économe (ci-dessous). L'API et les deux
series_code sont valides (test v1 sans clé : IPC `CUUR0000SA0`, JOLTS `JTS…JOL`). Note réelle :
**octobre 2025 « unavailable due to the 2025 lapse in appropriations »** → trou de série à afficher
honnêtement (le code écarte les points sans valeur/sans % — jamais d'estimation).

## Lot 1.2 — Courbe IPC en % annuel
- Catalogue = source de vérité : `us_cpi` porte `series_kind: "yoy_percent"` +
  `value_unit: "% de variation annuelle"` ; `CatalogEvent.series_kind` + résolveur caché
  `series_kind_for(series_code)` (`base_official.py`).
- `kind` threadé `series_for → fetch_series` (base + 5 fetchers ; non-BLS l'ignorent).
- `BLSValueFetcher.fetch_series(kind="yoy_percent")` demande `calculations:true` et lit
  `pct_changes["12"]` **tel que publié par BLS** (jamais recalculé) ; mois sans calcul → pas de
  point. `calendar_service.get_event` passe le `kind`. En-tête + stats portent l'unité %.
- Tests : `test_bls_fetch_series_yoy_percent_uses_published_change`, `…_level_ignores_calculations`.

## Lot 1.3 — Source des mesures fiabilisée (cause racine)
- Cause confirmée : `load_default_measures` lisait **deux CSV gitignorés** (bougies XAU **et**
  calendrier) → absents du conteneur → `None` → aucune question. C'est l'échec « faute de données ».
- Loader ré-architecturé à couches : bougies XAU M15 **depuis `candles.db`** (`CandlesCacheStore`)
  → repli CSV ; dates de parution **depuis `calendar_cache.db`** (`get_events_between`, match par
  series_code) → repli CSV. Injectable pour tests. `purge_old_events` jamais appelé → le store
  accumule (les parutions passées persistent).
- Profondeur `XAUUSD.M15 → 24mo` (`lookback_depths.json`) pour couvrir 12–24 parutions + jours
  de référence. Backfill unique (~11 req Twelve Data) ; **fenêtre de détection inchangée**.
- Tests : `test_load_default_measures_computes_from_injected_stores`, `…_none_when_stores_empty`.

## Lot 1 — Cache économe BLS
- `CalendarService` : cache par (source, series_code, kind), TTL **6h** (`CALENDAR_SERIES_CACHE_TTL_S`,
  `.env.example`). Ne met en cache qu'une série **non vide** (un échec réseau/quota retente à la
  prochaine vue). ≤ quelques appels/jour → sous le plafond de 500/j.

## Lot 2 — Mesure #3 (cycle de vie des zones)
- `collect_zone_lifecycles` **fonction additive** dans `market_reading_mappers.py` : recense TOUTES
  les zones nées dans une fenêtre (y compris celles que l'affichage écarte une fois consommées —
  ce dont un recensement de cycle de vie a besoin), avec created_at/mitigated_at/status. **Zéro
  modification des règles de détection** (réutilise les mêmes prédicats de lifecycle).
- `_compute_zone_lifecycle` (`publication_measures.py`) : rejeu lecture seule [T, T+1 h] pour les
  zones nées, observation ~26 h pour la mitigation ; tranches de durée de vie (<1 h / 1–2 h /
  2 h–1 j), extrêmes datés, `never_mitigated`. `ZoneLifecycleMeasure` au schéma + `PublicationMeasures`.
- Frontend : `ZoneLifecycleCard` (Q3, entre structure et retour au calme), classe `.pub-nevermit`
  distincte. i18n **9 locales** (fr+en natifs + de/es/it/nl/pl/pt/ar traduits ; parité stricte OK).
  Mitigation expliquée en ligne de source ; aucune « bougie/médiane/moyenne » ; dénominateur porté.
- Tests : `test_zone_lifecycle_measure.py` (recensement garde ce que l'affichage écarte ;
  **non-régression** : sortie de `collect_zones` inchangée ; forme du calcul sur série synthétique).

## Vérification
- **pytest** ciblé calendrier/mesures/mappers/structure : **vert**. Deux échecs **pré-existants
  sur `origin/main`** (prouvé par `git stash`), hors périmètre : `test_tr1_structural_trend`
  (import `_eval_mtf_aligned` retiré par TR-1) et `test_enricher_flags_revision_across_cycles`.
- **vitest** : **848/848**. **tsc** : **vert** (soigné 2 casts pré-existants rouges sur main dans
  `nw6.test.tsx`). **build** : vert.
- **Playwright** : `nw7-publication` **20/20** (5 états × 1280×800 & 390×844 × 2 projets) — FULL
  (courbe % + 4 cartes dont zone), PASSÉE, À VENIR, SANS VALEURS, SANS FICHE ; `nw5+nw6` **32/32**
  (aucune régression). Captures dans `webapp/test-results/nw7-*.png`.
- Vérif **live** de la courbe % : différée au reset du quota BLS (clé valide, plafond atteint).

## Écarts assumés vs maquette (signalés)
- **Badge de la courbe** reste générique (« Valeur de l'indicateur ») ; l'unité « % de variation
  annuelle » s'affiche dans l'en-tête + la ligne stats (pas dans le badge). Cosmétique.
- **Révisions** distinguées sur la **valeur courante** (`actual_initial`/`revised`), **pas sur les
  12 points** de la courbe (BLS n'expose pas de série « telle que publiée à l'origine » par point).
- **Q3** : livré compte de zones créées + répartition durée-de-vie + extrêmes + jamais-mitigées.
  Les deux sous-lignes secondaires de la maquette (« zones préexistantes traversées », « poches de
  liquidité prises dans l'heure ») sont **différées** (la 2e demande le suivi du cycle de vie des
  poches, un autre sous-système).
- **`us_jolts`** hors périmètre (décision fondateur : IPC seul) — reste sans mesures ni liens.

## Action fondateur restante
- Le quota BLS se réinitialise quotidiennement : la courbe % s'affichera en live sans action.
  (Option rigueur : régénérer la clé, exposée en clair dans le chat, et la remettre au dashboard.)
