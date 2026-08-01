# AUDIT PERF-1 — Chargement de /app et fluidité de navigation

**Branche** : `fix/perf-1-chargement` (worktree dédié `wt-perf-1`, depuis `origin/main` d90cbd3)
**Date** : 2026-08-01
**Statut** : DIAGNOSTIC LIVRÉ — en attente du GO avant toute correction. Aucun code applicatif modifié.

---

## 0. TL;DR

Le symptôme « ~20 s de squelettes puis *données indisponibles* » **n'est ni la détection ni la
base**. Les deux sont rapides :

- lecture base (`get_latest_reading`, indexée) : **~6 ms** ;
- détection SMC complète sur 2233 bougies M15 (BOS/CHOCH/OB/FVG/liquidité), **sans numba** : **~200 ms** ;
- payload sérialisé : **~19 Ko**.

Le temps est **entièrement dans l'appel réseau au fournisseur** (`TwelveDataProvider.fetch_candles`),
et il est déclenché **à chaque ouverture** parce que **100 % des lectures en cache sont invalidées** :

1. `READING_LOGIC_VERSION = 5` dans le code, mais **toutes** les lectures stockées sont `v=4`/`None` → mismatch de version.
2. Même à version égale : `expected_close` (calculé pour « maintenant », 2026‑08‑01) ne peut jamais égaler
   le `candle_close_ts` stocké (dernières bougies réelles au 2026‑07‑31 / 2026‑07‑03) → mismatch de fraîcheur.

Chaque miss appelle le réseau. `candles.db` (qui contient pourtant 2233 bougies M15 exploitables)
**n'est jamais consulté sur un miss** — il n'est utilisé qu'en écriture (write‑through). Le fetch
TwelveData a un **timeout HTTP de 20,0 s**, plus un rate‑limiter (≤ 7,5 s) et 4 retries à backoff
1/2/4/8 s (pire cas ~100 s). Le front, lui, abandonne avant (gate 8 s + reading 8 s + 1 retry 8 s)
et affiche un message générique « Données indisponibles » qui **ne distingue pas** délai dépassé /
serveur injoignable / aucune donnée.

> **Note de méthode / honnêteté.** Les composants locaux (base, détection, route `/api/candles`,
> tailles de payload) sont **mesurés** (millisecondes réelles ci‑dessous). Le palier de ~20 s du
> fournisseur est **dérivé du code** (`REQUEST_TIMEOUT_S = 20.0` + rate‑limiter + retries) et
> **concorde exactement** avec le symptôme ; il n'a pas pu être chronométré contre le fournisseur
> live ici, faute de clé/terminal fournisseur dans cet environnement. Aucun chiffre n'est inventé :
> ce qui est mesuré est marqué *mesuré*, ce qui est dérivé du code est marqué *dérivé*.

---

## A. Décomposition des ~20 secondes — XAUUSD M15, cache vide/invalidé

| # | Étape | Temps | Source | Fichier |
|---|-------|-------|--------|---------|
| 1 | Front — gate `GET /api/access/me` (bloquant, avant le fetch reading) | ~0,2 s nominal, **timeout 8 s** | dérivé | `webapp/components/access/SubscriptionGate.tsx:50`, `webapp/lib/access/api-client.ts:48` |
| 2 | Front → `GET /api/market-reading?instrument=XAUUSD&timeframe=M15` (déclenche `get_or_generate`) | — | — | `webapp/lib/market-reading/api-client.ts:65` |
| 3 | Backend — lookup cache `get_latest_reading` (indexé) | **~6 ms** | **mesuré** | `src/storage/market_readings_store.py:169` |
| 4 | Backend — **décision cache : MISS (100 %)** — `v4/None ≠ v5` **ET** `expected_close(2026‑08‑01) ≠ candle_close_ts stocké` | ~0 | mesuré (contenu DB) | `src/intelligence/market_reading_assembler.py:310-321` |
| 5 | Backend — **`provider.fetch_candles` (TwelveData, réseau)** — timeout HTTP **20,0 s** + rate‑limiter ≤ 7,5 s + retries backoff 1/2/4/8 s (pire cas ~100 s) | **jusqu'à ~20 s** (dominant) | dérivé | `src/intelligence/market_reading_assembler.py:577` → `src/intelligence/data_providers/twelve_data_provider.py:139,240,242,250,269` |
| 6 | Backend — détection SMC (`build_enriched_frame` + `_default_smc_pipeline`) sur 2233 bougies, **sans numba** | **~200 ms** | **mesuré** | `src/intelligence/market_reading_assembler.py:145-235` |
| 7 | Backend — mapping structure/régime + sérialisation (payload ~19 Ko) | **< 50 ms** | estimé (mesuré : payload 19 441 car.) | `src/intelligence/market_reading_assembler.py:619-682` |
| 8 | Front — abandon reading à 8 s **+ 1 retry à 8 s** → jette `MarketReadingError(status 0)` | **~16 s perçues** | dérivé | `webapp/lib/market-reading/api-client.ts:19,70-84,97-120` |
| 9 | Front — rendu bougies | **masqué** : l'état d'erreur du reading remplace toute la colonne (chart compris) | dérivé | `webapp/components/app/DesktopReading.tsx:199` |
| — | *(À titre de comparaison)* `GET /api/candles` = lecture pure `candles.db`, aucun appel fournisseur | **~6 ms** | **mesuré** (même requête indexée) | `src/api/routes/candles.py:16-17,112` |

**Terme dominant, unique** : étape 5 — le fetch fournisseur (jusqu'à 20 s) exécuté **à chaque
ouverture** parce que le cache est invalidé à 100 % (étape 4) et que `candles.db` n'est pas lu sur un miss.

Mesures brutes reproductibles (script ad hoc, DB de `TradingBOT_Agentic/data/`) :

```
[DB]     get_latest_reading XAUUSD/M15  ~6 ms     (payload 19 441 car.)
[DETECT] XAUUSD/M15 n= 500  full_pipeline ~128–143 ms
[DETECT] XAUUSD/M15 n=1000  full_pipeline ~125–174 ms
[DETECT] XAUUSD/M15 n=2233  full_pipeline ~200–212 ms
env: python 3.12.6, pandas 3.0.0, numba ABSENT (ImportError)
```

État réel des bases (au diagnostic) :

```
market_readings.db (6,4 Mo) — versions des lectures les plus récentes :
  XAUUSD M15  v=4   |  XAUUSD H1  v=4  |  XAUUSD H4  v=4
  EURUSD M15  v=4   |  EURUSD H1  v=None | EURUSD H4 v=None
  → code attend v=5  ⇒  0 lecture servable depuis le cache.
candles.db (1,7 Mo) — XAUUSD M15 : 2233 bougies (2026‑06‑10 → 2026‑07‑03).
  → données présentes et exploitables, mais jamais lues sur un miss de reading.
```

---

## B. Cause du message « Données indisponibles »

Ce n'est **pas** un vrai « aucune donnée pour ce combo ». C'est un **délai dépassé côté front** :

- Le fetch reading est avorté par `AbortController` à 8 s, **rejoué une fois**, avorté de nouveau à 8 s
  (`api-client.ts:70-120`), puis transformé en `MarketReadingError(status 0)`.
- Le composant mappe **toute** erreur non‑503/400 vers la clé i18n générique `placeholders.errorGeneric`
  sous le titre `placeholders.dataUnavailableTitle` = « Données indisponibles »
  (`webapp/components/app/ReadingPlaceholders.tsx:37-73`, `webapp/messages/fr.json:1308-1311`).
- Pendant ce temps le backend est **toujours bloqué** dans l'appel TwelveData de 20 s : personne n'écoute
  la réponse quand elle arrive.
- Le message **ne distingue pas** : (a) délai dépassé, (b) serveur injoignable, (c) aucune donnée pour ce
  combo. Seul un 503 (assembler non câblé) et un 400 (combo hors périmètre) ont un texte propre.
- Aggravant : quand seule la lecture échoue, le graphique — dont les bougies sont pourtant disponibles en
  ~6 ms — est **masqué** aussi (`DesktopReading.tsx:199` renvoie l'état d'erreur pour toute la colonne).

Journal backend attendu au moment du message (dérivé du code) : soit un blocage silencieux dans
`_fetch_dataframe` (timeout/rate‑limit), soit `build_fresh failed … — serving stored reading if any`
(`market_reading_assembler.py:320`) suivi, faute de lecture v5 stockée servable, d'un `HTTPException 500`
(`src/api/routes/market_reading.py:74-80`).

---

## C. Les trois plus gros postes de temps + gain estimé de leur correction

### 1. Fetch fournisseur sur **chaque** ouverture — ~20 s → cible **< 0,4 s** (gain ~98 %)
**Cause** : cache reading invalidé à 100 % (bump `READING_LOGIC_VERSION` 4→5 + `expected_close` toujours
en avance sur les bougies stockées) **et** `candles.db` jamais consulté sur un miss (write‑through
uniquement, `assembler.py:577` puis `:592`).
**Piste (à valider au GO)** : lecture *read‑through* de `candles.db` sur un miss / quand le fournisseur
est lent ou indisponible, avec le **badge « Données en retard » déjà existant** (honnêteté préservée,
règle produit MC‑1/DG‑1). La détection tourne alors en local (~200 ms) sans aucun appel réseau.
**Gain estimé** : 20 s → < 0,4 s cache chaud ; supprime le cas « indisponible » quand des bougies
existent localement.

### 2. Cascade front + empilement des retries (8 s + 8 s) + erreur trop vague — ~16 s perçues
**Cause** : gate d'accès bloquant en amont du reading ; retry qui **double** le timeout ; message unique
pour trois pannes distinctes ; graphique masqué par l'erreur de reading.
**Piste (à valider au GO)** : ne pas bloquer le reading derrière le gate (le paralléliser ; en
`SENTINEL_TESTING_MODE` le gate est un no‑op) ; distinguer *délai dépassé* / *injoignable* /
*aucune donnée* ; ne pas masquer le graphique quand seules les métadonnées de lecture manquent ;
revoir la politique de retry (un seul essai, timeout explicite).
**Gain estimé** : temps perçu ramené au temps réel backend ; bougies affichées < 1 s même si la lecture
tarde ; message qui dit la vérité sur l'état.

### 3. Détection sans numba + recalcul forcé par les bumps de version — ~200 ms (mineur)
**Cause** : `numba` absent de l'environnement → passe BOS/CHOCH en Python pur ; chaque bump de
`READING_LOGIC_VERSION` force un recalcul complet.
**Piste (à valider au GO)** : installer `numba` dans le runtime (accélère la passe BOS/CHOCH) ;
éventuellement persister/réutiliser la détection entre bumps. **Mineur** une fois le poste #1 réglé
(200 ms est déjà sous la cible).
**Gain estimé** : ~200 ms → ~30 ms.

---

## D. Réponses au questionnaire de diagnostic (§2 de la mission)

- **A. Décomposition** → tableau §A ci‑dessus.
- **B. « Indisponible » après 20 s** → §B : timeout front (8 s ×2), pas une erreur « pas de données » ;
  déclenché par délai, pas par absence réelle ; backend bloqué dans le fetch fournisseur (20 s).
- **C. Volume demandé** → Le front **borne déjà** ses bougies : chart `limit=400`, prix `limit=300`
  (`webapp/lib/market-reading/hooks.ts:267,350`) ; serveur **plafonne** à `MAX_LIMIT=1000`
  (`src/api/routes/candles.py:54,84`). **La suspicion « le front demande tout l'historique » est
  INFIRMÉE pour les bougies.** Le reading n'a pas de paramètre de fenêtre mais renvoie un objet compact
  (~19 Ko) — pas un problème de volume. Le backfill LB‑1 n'a **pas** gonflé le transfert ; il a gonflé la
  *fenêtre d'analyse* (M15 ~2880 bougies) — mais la détection reste à ~200 ms.
- **D. Recalculé à chaque requête** → Oui : `_build_fresh` rejoue détection **+ fetch réseau** à chaque
  miss, et les bumps de version rendent chaque ouverture un miss. Zones/événements recalculés une fois par
  ouverture (pas de N+1).
- **E. Requêtes en cascade** → 1 gate bloquant (`/api/access/me`) **puis** en parallèle
  `/api/market-reading` + `/api/candles?limit=400` + `/api/candles?limit=300` (prix M15). Le prix relit
  des bougies M15 en plus du chart (léger doublon volontaire, cadence 45 s). MTF (`useMtfTrends`) tire les
  unités supérieures en parallèle via `Promise.all` seulement si la carte Régime est montée.
- **F. Base** → **Bien indexée** : `candles_cache` PK `(instrument,timeframe,ts)` + index couvrant
  `(instrument,timeframe,ts DESC)` ; `market_readings` UNIQUE + index couvrant. Aucun full‑scan sur le
  chemin chaud. WAL + `synchronous=NORMAL`, connexion par appel (OK pour SQLite). Tailles : candles.db
  1,7 Mo, market_readings.db 6,4 Mo. **La base n'est pas un goulot.**
- **G. Navigation** → Pas de cache client (useState/useEffect, ni SWR ni React‑Query). Changer d'unité,
  de marché, ou revenir de Scanner/Zones **refetch** tout (démontage de composant). À traiter au GO
  (rétention mémoire des combos déjà obtenus).

---

## E. Prochaines étapes (APRÈS GO EXPLICITE)

Une correction à la fois, mesurée après chacune ; interdits §1 de la mission respectés (aucune perte de
précision, aucun cache muet, aucun repli factice, aucun masquage d'erreur, détection inchangée prouvée
par non‑régression, aucun chargement de fond trompeur). Ordre proposé, à arbitrer :

1. Read‑through `candles.db` sur miss/fournisseur lent + badge « en retard » (poste #1).
2. Honnêteté du chargement : timeouts partout, messages distincts (délai/injoignable/aucune donnée),
   ne pas masquer le graphique quand seule la lecture tarde (poste #2).
3. Paralléliser gate/reading, revoir retry (poste #2).
4. Test de garde « un appel ne peut pas demander plus de N bougies » (N documenté), test de
   non‑régression détection identique avant/après sur les 6 unités, Playwright 1280×800 / 390×844
   (chargement complet, lent simulé, serveur injoignable, combo sans données).
5. `numba` runtime + éventuelle persistance de détection (poste #3, mineur).

---

## F. Corrections appliquées

### Correction 1 — Read‑through de `candles.db` sur le chemin interactif (poste #1)

**Fichiers** : `src/intelligence/market_reading_assembler.py` (helpers `_fetch_candles_for_build`
+ `_fetch_from_provider_bounded`, param `bound_provider` sur `get_or_generate`/`_build_fresh`,
budget `provider_fetch_timeout_s`), `src/intelligence/scheduler.py` (passe `bound_provider=False`),
`src/api/bootstrap.py` (`env_float` + wiring `SENTINEL_PROVIDER_FETCH_TIMEOUT_S`).

**Principe** : sur un miss, le chemin **interactif** tente le fournisseur dans un budget wall‑clock
(défaut **5 s**, env `SENTINEL_PROVIDER_FETCH_TIMEOUT_S`) puis, en cas de délai/échec/réponse vide,
lit **à travers** `candles.db` (bougies réelles). Le badge « Données en retard » existant
(`market_status` vs `candle_close_ts` dérivé des vraies bougies) signale déjà tout retard — aucune
donnée périmée servie comme fraîche, aucune donnée factice. Si le fournisseur **et** le cache sont
vides → l'erreur est **remontée** (jamais de lecture blanche). Le chemin **scheduler** reste
**patient** (`bound_provider=False`, fetch non borné) pour continuer à faire avancer `candles.db`.
Un fetch qui dépasse le budget continue en tâche de fond dans son worker et réchauffe le cache TTL
60 s du fournisseur → la lecture *suivante* obtient les bougies fraîches gratuitement.

**Interdits respectés** : détection **inchangée** (mêmes bougies → même pipeline ; prouvé par
`test_detection_input_identical_provider_vs_readthrough` sur les 6 unités) ; pas de cache muet
(badge) ; pas de repli factice (bougies réelles du cache) ; pas de masquage (erreur remontée quand
rien n'existe) ; pas de chargement de fond trompeur (on attend le budget puis on sert du réel).

**Gain mesuré** (feed simulé indisponible, `candles.db` réel, pipeline de détection réel) :

| Combo | Avant (fournisseur lent/absent) | Après (read‑through) |
|-------|----------------------------------|----------------------|
| XAUUSD M15 | ~20 s puis « indisponible » | **~0,9 s** (à froid, import pandas inclus ; ~0,2–0,4 s à chaud) |
| XAUUSD H1  | ~20 s puis « indisponible » | **~0,39 s** |
| XAUUSD H4  | ~20 s puis « indisponible » | **~0,33 s** |

→ **20 s → < 1 s**, et le cas « indisponible » disparaît dès que des bougies existent localement.
La lecture porte un `candle_close_ts` honnête (dernière bougie réelle en cache) → badge « en retard ».

**Tests** : `tests/test_market_reading_readthrough_perf1.py` (14 tests : provider gagne / échoue /
vide / lent‑borné / vide‑des‑deux‑côtés / chemin scheduler patient / e2e feed down / non‑régression
6 TF) + mocks scheduler mis à jour. Surface reading+scheduler+bootstrap+candles : **154 passés, 0 régression**.

**Restant sur ce poste (à décider)** : le budget de 5 s reste payé quand le fournisseur est réellement
lent‑mais‑vivant ; le front (parallélisation gate/reading, rétention mémoire de navigation) reste à faire.

### Correction 2 — Honnêteté du chargement (taxonomie d'erreurs + repli progressif)

**Fichiers** : `webapp/lib/market-reading/api-client.ts` (champ `reason` sur `MarketReadingError`,
nouvelle `MarketReadingNoDataError` 404, retry limité au transitoire réseau), `webapp/components/app/
ReadingPlaceholders.tsx` (copie distincte par mode d'échec + `SlowLoadHint`), `DesktopReading.tsx` /
`ReadingColumn.tsx` (repli progressif câblé), `src/api/routes/market_reading.py` + `market_reading_
assembler.py` (`MarketReadingDataUnavailable` → **404**), 9 locales `messages/*.json` (4 clés natives).

**Principe** : le message unique « Données indisponibles » est remplacé par une distinction honnête :
- **délai dépassé** (le service met trop de temps) ≠ **serveur injoignable** (la connexion échoue) —
  distingués côté front via `reason: 'timeout' | 'network'` (le front sait lequel s'est produit) ;
- **aucune donnée pour ce combo** — nouveau **404** backend (`MarketReadingDataUnavailable`, combo valide
  mais ni fournisseur, ni cache, ni lecture stockée) distinct du **500** (vrai bug interne) ;
- **combo non supporté** (400) et **service non câblé** (503) inchangés.
- **Repli progressif** (`SlowLoadHint`) : au‑delà de 6 s de chargement, un message « le chargement prend
  plus de temps que prévu — récupération des données en cours » distingue « ça charge » de « c'est
  cassé » ; au‑delà du budget de 8 s, le message de délai dépassé prend le relais.
- **Retry** : ne rejoue plus qu'un transitoire **réseau** — jamais un **timeout** (qui a déjà consommé
  tout le budget), ce qui supprime le doublement du délai (~16 s → ~8 s) constaté au diagnostic.

**Interdits respectés** : aucun masquage d'erreur (chaque mode a son message, l'erreur brute est chaînée
en `__cause__` côté backend et jamais fuitée à l'écran) ; pas de chargement de fond trompeur (le repli
progressif dit explicitement que ça charge encore) ; détection intacte (aucune touche moteur).

**Tests** :
- Front unit (`api-client.test.ts` +4, `reading-load-honesty.test.tsx` +7) : 404→NoData, timeout tagué
  `timeout` **non rejoué**, réseau tagué `network` **rejoué une fois**, et copie distincte à l'écran pour
  timeout / injoignable / aucune donnée / combo non supporté / 503 / générique + repli progressif.
- Backend (`test_market_reading_endpoint.py` +1) : `MarketReadingDataUnavailable` → **404** distinct du 500,
  message interne non fuité.
- **Playwright 1280×800 + 390×844** (`perf1-load-honesty.spec.ts`, **8/8**) : chargement complet (canvas
  rendu, aucune copie d'erreur), chargement lent (hint puis délai dépassé, jamais figé), serveur injoignable
  (« ne répond pas »), combo sans données (« aucune donnée … cette combinaison »).
- Global : **tsc 0**, **vitest 243**, **build vert**, **backend 94** (reading+scheduler+endpoint+bootstrap+
  read‑through), 0 régression.

**Note d'architecture (honnête)** : sur mobile l'onglet « Lecture » (et donc `SlowLoadHint`) se monte au clic
utilisateur, après le démarrage du fetch ; un chargement > 8 s y surface donc directement le message de délai
dépassé plutôt que le hint. Les deux sont des signaux « pas figé » valides ; le hint reste utile sur desktop
et lors d'un rafraîchissement quand l'onglet est déjà ouvert.

### Correction 3 — Rétention mémoire à la navigation (SWR côté client)

**Fichiers** : `webapp/lib/market-reading/hooks.ts` (`readingCache` + `candlesCache` au niveau module,
seed instantané + revalidation obligatoire ; `__resetReadingRetention` pour les tests), + tests
(`hooks.test.ts`, `useCandles.test.ts`).

**Principe** : `useMarketReading` / `useCandles` blanchissaient les données à chaque changement de combo
et se démontaient à la navigation (retour de Scanner/Zones) → refetch complet à chaque fois. Ajout d'un
cache mémoire (par `source:instrument:timeframe`) : au **revisit** (changement d'unité/instrument, retour
sur /app), la dernière valeur connue de CE combo s'affiche **instantanément** pendant qu'une revalidation
d'arrière-plan la rafraîchit (style SWR). Le voyant **`isRefreshing`** est le signal d'honnêteté, et
**chaque hit est revalidé** → rien de périmé n'est servi *sans le dire*. Première visite (cache froid) :
squelette honnête inchangé. Cache borné (≤ 6 × 6 combos), mémoire de process uniquement, jamais persisté.

**Cibles atteintes** : changement d'unité **< 1 s** (affichage immédiat du cache + refresh), navigation
Scanner/Zones → retour /app **sans re-blank** (données déjà obtenues ré-affichées instantanément).

**Interdits respectés** : pas de cache muet (revalidation systématique + voyant `isRefreshing` + badge de
fraîcheur du reading) ; pas de donnée factice (uniquement la dernière valeur réelle de CE combo) ;
détection intacte (front only).

**Tests** : `hooks.test.ts` (+2 : revisit → cache instantané `isRefreshing` non `isLoading` + revalidation ;
première visite froide → blank+squelette), `useCandles.test.ts` (reset cache). **178 tests lib+access
verts**, tsc 0, build vert, **Playwright 8/8** inchangés.

**Écartée : parallélisation gate/lecture.** Tentée (rendu optimiste de /app pendant `/api/access/me`)
puis **retirée** : elle cassait l'affichage des états d'erreur en e2e (interaction de remontage avec le
fetch de lecture) et touche une frontière sécurité/vie privée pour un gain marginal (en pratique
`/api/access/me` répond vite ou échoue vite ; ce n'était pas le goulot — cf. §A). Le gate reste inchangé.

**Différé : `numba` runtime.** Changement de dépendances backend (accélère BOS/CHOCH ~200 ms → ~30 ms) —
mineur une fois le poste #1 réglé ; à décider hors de ce lot front.
