# AUDIT PERF-2 — Fluidité de bout en bout

Branche : `perf/perf-2-fluidite` (worktree dédié `wt-perf-2`, depuis `origin/main` @ `c695da8`).
Statut : **CORRECTIONS APPLIQUÉES — mesurées avant/après. En attente de la validation live pour merge.**
Date : 2026-08-06.

> Diagnostic (phase 3) en §1‑3. Corrections (phase 4+) en §7‑9, chacune avec sa mesure
> avant/après.

---

## 0. Résumé exécutif — les trois faits qui gouvernent la suite

1. **Le défaut de rafraîchissement est réel et localisé** : le flux des **bougies**
   (le graphique) n'a **aucun chemin de reprise autonome**. Il n'est re-sollicité que
   lorsque `candle_close_ts` change ou que le combo change. Une panne transitoire
   (timeout, réseau, réponse vide sur combo froid) laisse le graphique vide jusqu'à la
   **prochaine clôture de bougie** — 15 min en M15, **jusqu'à 24 h en D1** — ou jusqu'à
   un **rechargement manuel de la page**. La lecture textuelle, elle, se répare seule
   (poll 60 s). D'où « il faut rafraîchir pour que le graphique apparaisse ».

2. **Le test « AppWorkspace / skeleton pendant le fetch » est VERT sur `origin/main`**
   (6/6, mesuré ci-dessous). Il a été réparé par DETTE-1 (commit `0615c93`, ajout de
   `__resetReadingRetention()` dans `beforeEach`). L'alarme que la mission décrit comme
   rouge est en réalité verte aujourd'hui. **Je ne la touche pas**, elle reste le critère
   de validation. **Mais elle ne couvre pas le vrai défaut** (elle teste le skeleton de
   la *lecture*, pas la reprise du *graphique*) : il faut un **nouveau** test pour le
   défaut de rafraîchissement.

3. **Le poste de temps dominant est l'appel au fournisseur externe (Twelve Data)
   effectué PENDANT la requête client**, sur un défaut de cache (combo froid, ou bump de
   `READING_LOGIC_VERSION`). C'est borné à 5 s puis repli sur `candles.db`. La détection
   SMC rejouée sur ~2880 barres (M15) est le second poste. La sérialisation/SQL est
   marginale.

---

## 1. LE DÉFAUT DE RAFRAÎCHISSEMENT (priorité 1)

### 1.1 Cause racine

Fichier : `webapp/lib/market-reading/hooks.ts`, hook `useCandles` (lignes 318‑393).

Le flux des bougies se re-déclenche **uniquement** sur ce tableau de dépendances
(ligne 390) :

```ts
}, [instrument, timeframe, source, candleCloseTs]);
```

- **Aucun poll** propre (contrairement à `useMarketReading`, qui poll `POLL_MS = 60_000`).
- **Aucune reprise auto sur échec** : `api-client.ts` ne retente que `reason === 'network'`,
  jamais un `timeout` (ligne 100‑104). Un timeout de 8 s (`DEFAULT_TIMEOUT_MS`) ne se
  retente donc pas.
- `candleCloseTs` vient de `reading?.header.candle_close_ts` (DesktopReading.tsx:102).
  Il ne change qu'à la **clôture d'une bougie de l'unité active**.

**Conséquence.** Si la requête `/api/candles` échoue/expire/renvoie vide au chargement :
- `setCandles(null)` → le slot affiche `<ChartUnavailable />` (DesktopReading.tsx:234‑235) ;
- `useMarketReading` poll toutes les 60 s, mais tant que la **même** bougie est en cours,
  `candle_close_ts` est identique → `useCandles` **ne se re-déclenche pas** ;
- le graphique reste vide jusqu'à la prochaine clôture (M15 : ≤15 min ; H1 : ≤1 h ;
  H4 : ≤4 h ; **D1 : ≤24 h** ; W1 : jusqu'à une semaine) — ou jusqu'au rechargement manuel,
  qui remonte le composant et relance un fetch neuf. **C'est exactement le symptôme décrit.**

**Aggravant — même le bouton « Réessayer » ne répare pas le graphique.** `ChartUnavailable`
accepte pourtant une prop `onRetry` (ReadingPlaceholders.tsx:118‑121), **mais elle n'est pas
câblée** : DesktopReading rend `<ChartUnavailable />` sans `onRetry` (ligne 235), et
`useCandles` n'expose **aucun** `refresh`. Le bouton « Réessayer » de l'écran d'erreur, lui,
appartient à la *lecture* (`useMarketReading.refresh` → bump `refreshNonce`) et ne re-tire pas
les bougies tant que `candle_close_ts` ne bouge pas.

### 1.2 Ce que dit chaque symptôme

| Observation | Explication |
|---|---|
| « Parfois il faut rafraîchir la page » | Panne transitoire du fetch bougies + absence de reprise autonome. Le reload remonte le composant → nouveau fetch. |
| Plus fréquent/persistant sur unités hautes | `candle_close_ts` change rarement (D1 : 1×/jour) → fenêtre de blocage énorme. |
| La lecture (texte) finit par s'afficher, pas le graphique | La lecture poll 60 s et se répare ; les bougies non. Asymétrie de conception. |
| Combo « froid » = graphique vide qui ne se remplit pas | Le read‑through peut renvoyer vide au 1er appel ; rien ne re-tire les bougies avant la clôture. |

### 1.3 Ce qui est déjà correct (à ne pas casser)

Le hook `useMarketReading` est robuste : `AbortController`, garde de séquence monotone
(`seq !== requestSeq.current` sur `.then/.catch/.finally`), retry réseau, timeout 8 s,
distinction de 6 modes d'erreur (`reading-load-honesty.test.tsx`), rétention SWR PERF‑1.
**Le défaut n'est pas là** — il est dans l'absence d'équivalent côté bougies.

### 1.4 État du test « alarme » (mesuré)

`npx vitest run components/app/__tests__/AppWorkspace.test.tsx` →
**Test Files 1 passed / Tests 6 passed** (11,5 s). La ligne 103‑110
(« shows a skeleton while the initial fetch is in flight ») **passe**. Le fix DETTE‑1
(`__resetReadingRetention` dans `beforeEach`, lignes 63‑71) est présent sur `origin/main`.

> **Arbitrage à valider.** Le test « alarme » est déjà vert et ne capte PAS le vrai
> défaut. Je propose (a) de le garder intact comme critère, et (b) d'**ajouter** un test
> qui reproduit le défaut de rafraîchissement (fetch bougies qui échoue → le graphique
> doit se rétablir sans rechargement, via reprise auto/poll/bouton). Je ne modifie pas
> l'existant.

---

## 2. DÉCOMPOSITION DU TEMPS — méthode et état

> **Honnêteté sur les chiffres.** La décomposition wall‑clock de bout en bout exige la
> pile complète en marche (uvicorn + Twelve Data + SQLite peuplé). Ce que j'ai **mesuré**
> directement à ce stade : le temps du test frontend, les constantes de code, les
> tailles de fenêtres. Ce qui est **estimé** vient de la lecture du code + des
> caractéristiques connues de Twelve Data. Les colonnes sont étiquetées.
> **Étape 0 après GO : capturer les ms réelles** (instrumentation serveur + Playwright
> réseau) avant toute optimisation — conformément à « interdiction d'optimiser avant
> d'avoir des chiffres ». Je n'optimise rien sur des estimations.

### 2.1 Constantes vérifiées (mesurées dans le code)

| Élément | Valeur | Source |
|---|---|---|
| Timeout fetch client (lecture + bougies) | 8 000 ms | `api-client.ts:19` |
| Poll lecture | 60 000 ms | `AppWorkspace.tsx` (`POLL_MS`) |
| Poll bougies | **aucun** | `hooks.ts:390` (dépendances) |
| Fenêtre bougies demandée par le front | 400 | `hooks.ts:306` (`CHART_CANDLE_LIMIT`) |
| Cap serveur `/api/candles` | `MAX_LIMIT = 1000` (`ge=1, le=1000`) | `routes/candles.py:54,84` |
| Défaut `/api/candles` | `DEFAULT_LIMIT = 300` | `routes/candles.py:53` |
| Timeout fournisseur borné (interactif) | 5,0 s (`SENTINEL_PROVIDER_FETCH_TIMEOUT_S`) | `market_reading_assembler.py:282` |
| `READING_LOGIC_VERSION` | 6 | `market_reading_assembler.py:55` |
| Fenêtre d'analyse détection | M15 ≈2880 barres, H1 ≈720, H4 ≈180, D1 500 | `lookback_config.py` |
| Route `/api/market-reading` | `def` (sync, threadpool) — anti‑gel event loop (REC‑1) | `routes/market_reading.py:42` |

### 2.2 MESURES RÉELLES (avant correction) — 2026-08-06

Pile réelle : `uvicorn src.api.asgi:app` (code de branche = `origin/main`), `.env` de
`wt-run-main` (clé Twelve Data), copie des DBs (`candles.db`, `market_readings.db` du
31/07). Scheduler OFF, auth bypass. `curl -w time_total/time_starttransfer`.

**Contexte critique du run : quota Twelve Data ÉPUISÉ (HTTP 429 sur chaque fetch) — ce
qui est la condition réelle de la prod** (cf. mémoire : prod à 1302/800 appels/jour).
Crédit Anthropic aussi épuisé (Haiku 400) — la description retombe sur le gabarit.

| Endpoint / combo | Appel #1 | Appel #2 | Appel #3 | Taille |
|---|---|---|---|---|
| `/api/market-reading` XAUUSD **M15** | **7,04 s** | **7,02 s** | **7,09 s** | 19,5 Ko |
| `/api/market-reading` XAUUSD **D1** | **5,63 s** | **5,75 s** | **5,79 s** | 8,4 Ko |
| `/api/candles` XAUUSD M15 (limit 400) | **0,020 s** | **0,028 s** | — | 38 Ko |
| `/api/candles` XAUUSD D1 (limit 400) | **0,024 s** | **0,017 s** | — | 39 Ko |

**Faits mesurés :**
- Le TTFB ≈ le total → tout le temps est backend, avant le premier octet.
- **La lecture coûte 5‑7 s à CHAQUE appel, y compris les « chauds ».** Le cache ne sert
  JAMAIS de hit dans ces conditions. Journaux : `TwelveData HTTP 429 ... provider fetch
  exceeded 5.0s ... reading through candle cache` à chaque requête.
- Les **bougies** sont déjà à ~20 ms (lecture SQLite pure) : le chemin rapide existe.
- M15 (7 s) > D1 (5,6 s) : l'écart ≈ coût de la détection (2880 vs 500 barres) par‑dessus
  le plafond fournisseur de 5 s.

### 2.3 CAUSE RACINE du « jamais chaud » (confirmée en base)

`_payload_matches` (assembler:589) compare `payload["header"]["candle_close_ts"]` à
`expected_close`. Or `_persist_reading` (assembler:386) stocke la **colonne**
`candle_close_ts = expected_close`, tandis que le **header du payload** porte la bougie
réellement utilisée par la construction. Quand le fournisseur est bridé (429) et que la
construction retombe sur le **read‑through du cache de bougies (données périmées)**, le
header porte la bougie périmée — qui ne peut JAMAIS égaler `expected_close` (qui avance
avec l'horloge tant que le marché est ouvert). Vérifié en base :

| Combo | COLONNE `candle_close_ts` | PAYLOAD `header.candle_close_ts` | Match |
|---|---|---|---|
| XAUUSD M15 | `2026-08-06T22:30:00Z` | `2026-07-31T21:00:00Z` | **False** |
| XAUUSD D1 | `2026-08-06T00:00:00Z` | `2026-07-31T00:00:00Z` | **False** |

⇒ `_payload_matches` renvoie toujours `False` ⇒ **rebuild à chaque requête** ⇒ 5 s de
plafond fournisseur + détection à chaque appel. Le cache est **structurellement défait**
dès que le fournisseur n'atteint pas `expected_close` — c'est‑à‑dire la condition bridée
de la prod. **Le client attend le fournisseur à chaque appel** — ce que la mission
interdit.

> En prod SAINE (fournisseur à jour), `header.candle_close_ts == expected_close` → hit
> rapide. Mais au moindre retard/quota du fournisseur, la latence retombe à 5‑7 s par
> appel — et la mémoire atteste que la prod DÉPASSE le quota. Les « 5‑8 s » observés sont
> donc le régime nominal, pas un cas rare.

---

## 3. LES TROIS PLUS GROS POSTES + GAIN ESTIMÉ

| # | Poste | Où | Gain visé | Piste (à confirmer post‑GO) |
|---|---|---|---|---|
| 1 | **Appel fournisseur pendant la requête client** (défaut de cache) | `market_reading_assembler._fetch_from_provider_bounded` | **~3‑4 s → ~0 s** sur le chemin chaud | Garantir que **toute** lecture servie au client provient du précalcul planifié (scheduler) ; le client ne déclenche jamais un fetch fournisseur synchrone. Élargir le périmètre « always‑warm » aux 6 combos affichables ; sur défaut de cache, servir `candles.db` + **badge d'âge** plutôt qu'attendre le réseau. |
| 2 | **Détection SMC rejouée sur cache vide** (~2880 barres M15) | `_default_smc_pipeline` / `build_enriched_frame` | **~0,8‑1,2 s → ~0 s** sur le chemin chaud | Précalculer la lecture à la **clôture de bougie** et la persister (déjà partiellement fait par le scheduler) ; garantir zéro rejeu de détection dans la requête client. **Aucune** modification des règles de détection (test de non‑régression obligatoire). |
| 3 | **Défaut de rafraîchissement du graphique** (perçu comme « page qui ne s'affiche pas ») | `useCandles` (front) | Page vide **permanente → reprise < 1 s** | Reprise autonome du flux bougies : reprise auto bornée sur échec réseau, poll léger aligné sur la lecture, et bouton « Réessayer » **câblé** sur `useCandles.refresh`. Distinguer timeout / serveur injoignable / aucune donnée. |

> Le #3 n'est pas un poste de *latence* mais de *correction* — et la mission le classe en
> premier (« une page qui ne s'affiche pas est pire qu'une page lente »). Il sera traité
> en priorité, avec un test qui reproduit le défaut.

---

## 4. AUTRES ACTIONS CHRONOMÉTRÉES (survol code — à confirmer en live)

| Action | Appels réseau | Devrait être | Risque |
|---|---|---|---|
| Bascule d'une couche (OB/FVG/liquidité) | 0 | instantané | OK (état d'affichage) |
| Filtre « mitigées » | 0 | instantané | OK |
| Clic sur une zone (highlight/focus) | 0 | instantané | OK |
| Changement d'unité | lecture + bougies (parallèle) | <500 ms si en cache | dépend du cache combo |
| Changement de marché | lecture + bougies | <800 ms si en cache | 1er accès = cache froid → lent |
| Ouverture Scanner (run) | POST `/api/conditions-scan` | <1,5 s | à mesurer (6 combos) |
| Ouverture Zones | lecture + bougies (cache partagé /app) | instantané si visité | OK (cache partagé) |
| Calendrier / publication | GET `/api/calendar[...]` | <1 s | à mesurer |
| Message M.I.A | POST `/api/chatbot/message` | latence modèle | hors périmètre latence front |
| Navigation A→B→A | rétention mémoire (PERF‑1) | pas de rechargement | OK (Map module) |

---

## 5. GARDE‑FOUS DÉJÀ EN PLACE (à conserver/renforcer)

- Cap serveur bougies : `le=1000` (`routes/candles.py`). ⇒ **le test de garde « pas plus
  de N bougies » existe implicitement ; N = 1000, à documenter et couvrir explicitement.**
- Route lecture en `def` (threadpool) : ne gèle plus l'event loop (REC‑1).
- Rétention SWR côté client (PERF‑1), indexes SQLite composites + DESC, WAL résilient.

---

## 6. GARDE‑FOUS DÉJÀ EN PLACE (conservés)

- Route lecture en `def` (threadpool) : ne gèle plus l'event loop (REC‑1).
- Rétention SWR côté client (PERF‑1), indexes SQLite composites + DESC, WAL résilient.

---

## 7. CORRECTION A — Défaut de rafraîchissement du graphique (priorité mission)

**Cause** (§1) : `useCandles` n'avait aucune reprise autonome ; une panne transitoire du
flux bougies laissait le graphique vide jusqu'à la prochaine clôture (≤24 h en D1) ou un
rechargement manuel.

**Correctifs** (front) :
- `api-client.ts` : `CandlesError` porte désormais un `reason` (`timeout`/`network`/
  `nodata`/`server`/`parse`) ; `fetchCandles` **retente une fois sur un transitoire réseau**
  (même politique que `fetchMarketReading`).
- `hooks.ts` `useCandles` : expose `refresh()` + **reprise auto bornée** (3 essais, backoff
  linéaire 1,5 s) sur transitoire ; un 404 déterministe n'est PAS retenté auto (laissé au
  bouton). Timer nettoyé au démontage/décrochage combo.
- `ReadingPlaceholders.ChartUnavailable` : message **distinct par cause** (réutilise les clés
  d'erreur de lecture déjà traduites dans les 9 locales — 0 nouvelle clé i18n) + bouton
  « Réessayer ».
- `DesktopReading.tsx` + `ReadingColumn.tsx` (mobile) : **câblent** `onRetry` sur
  `useCandles.refresh` et passent le `reason` au placeholder.

**Test de reproduction** (`useCandles.test.ts`, 3 nouveaux) : (a) un transitoire se répare
seul sans clôture ni reload ; (b) la reprise auto est **bornée** (1 initial + 3, puis stop) ;
(c) un 404 n'est pas retenté auto mais `refresh()` le récupère. Le test « AppWorkspace /
skeleton » reste **vert et non modifié** (6/6).

## 8. CORRECTION B — « Précalculer et servir » : le client n'attend plus le fournisseur

**Cause** (§2.3) : sous fournisseur bridé, `_payload_matches` échouait toujours → rebuild
+ 5 s de plafond fournisseur **à chaque requête**.

**Correctif** (`market_reading_assembler.get_or_generate`) : sur le chemin **interactif**
(`bound_provider=True`), quand une lecture **de la version de logique courante** est déjà
stockée mais ne matche pas `expected_close`, on la **sert telle quelle instantanément**
(lecture SQLite) + `mark_combination_active` — au lieu de reconstruire en attendant le
fournisseur. `market_status` badge l'âge honnêtement (« en retard »). Le **seul** chemin qui
touche le fournisseur est désormais le **scheduler de fond** (`bound_provider=False`).
Invariants préservés : une lecture d'une **version de logique périmée** est toujours
reconstruite (MT‑D1/D4) ; cold‑start (rien en stock) construit comme avant (borné +
read‑through). Réversible via `SENTINEL_INTERACTIVE_SERVE_STORED=0`.

**Non‑régression détection** : le correctif ne touche NI la détection NI `_build_fresh` ;
la lecture servie est le **payload stocké verbatim**. Prouvé par
`test_current_version_stale_reading_is_served_without_rebuild` (sortie servie = payload
stocké) + les 6 `test_detection_input_identical_provider_vs_readthrough[M1..D1]` inchangés.

**Tests ajoutés** (`test_market_reading_assembler.py`) : serve‑stored sans rebuild ;
kill‑switch restaure le rebuild ; le scheduler reconstruit toujours ; **garde de budget**
`test_serve_stored_does_not_wait_on_a_slow_provider` (fournisseur qui dort 3 s → réponse
< 0,5 s, `call_count == 0`). **Garde N bougies** : `test_limit_cannot_exceed_documented_cap`
(N = `MAX_LIMIT` = 1000).

## 9. MESURES APRÈS + BUDGETS

Même pile / `.env` / DBs qu'au §2.2 (fournisseur toujours bridé 429), code de branche
après correctifs, `SENTINEL_INTERACTIVE_SERVE_STORED` par défaut (on) :

| Endpoint / combo | AVANT (§2.2) | APRÈS | Budget mission | Verdict |
|---|---|---|---|---|
| `/api/market-reading` XAUUSD **M15** | 7,04 s | **0,029‑0,045 s** | lecture chaude < 1 s | ✅ |
| `/api/market-reading` XAUUSD **D1** | 5,63 s | **0,023‑0,052 s** | lecture chaude < 1 s | ✅ |
| `/api/candles` XAUUSD M15/D1 (400) | 0,02 s | 0,02 s | bougies chaudes < 500 ms | ✅ |
| Lignes journal `429 / provider fetch` (interactif) | à chaque appel | **0** | — | ✅ fournisseur hors chemin |

**Gain principal** : lecture chaude **~200×** (7 s → ~0,03 s). Payload identique (19 523 B) —
détection servie verbatim.

Budgets §4 de la mission — état :
- bougies chaud < 500 ms ✅ ; lecture complète chaude < 1 s ✅.
- bascule de couche : aucun appel réseau ✅ (état d'affichage, inchangé).
- navigation A→B→A : rétention mémoire, pas de rechargement ✅ (PERF‑1, inchangé).
- première ouverture, cache vide < 2,5 s : **non mesurable ici** (fournisseur bridé) — après
  le tout premier build persistant, tous les appels suivants sont en serve‑stored (~0,03 s).
  À revalider en prod avec quota fournisseur sain.
- changement d'unité/marché : servi en serve‑stored si la lecture existe (~0,03 s) ; sinon
  cold‑start (rare). Le front tire lecture + bougies en parallèle.
- recherche scanner < 1,5 s : hors périmètre de ce correctif (POST `/api/conditions-scan`),
  à mesurer séparément.

## 10. CE QUI RESTE

- **Cold‑start absolu** (aucune lecture stockée + fournisseur bridé) : reste borné mais peut
  dépasser 2,5 s. Atténué en prod par le scheduler qui pré‑peuple les 6 combos affichables.
- **Playwright** : scénarios réseau (chargement, réseau lent, serveur injoignable, combo
  sans données, rafraîchissement nécessaire) — cf. §11.
- **Fraîcheur de fond** : dépend du scheduler (tick 60 s) ; sans scheduler, l'âge reste
  badgé honnêtement mais n'avance pas.

---

*Corrections A + B appliquées et mesurées. Détection inchangée (prouvé). Suites : vitest
854/854, pytest lecture/api 595 passants (échecs pré‑existants isolés : `test_tr1_structural_trend`
collect, 2 smoke d'environnement), tsc 0, build vert. En attente de validation live avant
merge sur main.*
