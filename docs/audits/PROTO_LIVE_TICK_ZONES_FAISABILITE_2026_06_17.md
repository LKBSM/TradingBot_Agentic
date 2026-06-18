# Prototype — interaction de zones EN DIRECT au tick (WebSocket) — FAISABILITÉ

> **Date** : 2026-06-17 · **Branche** : `feat/proto-live-tick-zones`
> **Étape** : 1/2 — **faisabilité uniquement, AUCUNE implémentation**. STOP demandé pour GO.
> **Cadre** : DEV / free tier / non commercial. Descriptif uniquement (zéro prédiction).

---

## 0. Verdict — ✅ FAISABLE sur le free tier

**Le WebSocket gratuit (trial) de Twelve Data streame bien EUR/USD ET XAU/USD** sur le compte
de ce poste. Sonde réelle exécutée (jetable, supprimée) sur `wss://ws.twelvedata.com/v1/quotes/price` :

```
SUBSCRIBE-STATUS: status=ok
  success: [ {XAU/USD, COMMODITY, PRECIOUS_METAL}, {EUR/USD, PHYSICAL CURRENCY, PHYSICAL_CURRENCY} ]
  fails: null
PRICE: EUR/USD 1.15202 …   PRICE: XAU/USD 4317.78929 …   (18 ticks en ~20 s, les 2 symboles)
SUMMARY: 18 price ticks, symbols with ticks: ['EUR/USD', 'XAU/USD']
```

→ Les **deux** symboles cibles sont souscrits sans échec et émettent des ticks en continu.
Le prototype « interaction de zones en direct » est donc **réalisable** sur le compte actuel.

---

## 1. Limites du free tier / trial (à respecter et documenter)

Source : support.twelvedata.com (trial / credits) + sonde réelle.

| Contrainte trial WS | Valeur | Conséquence design |
|---|---|---|
| Crédits WebSocket | **8** | 2 symboles = **2 crédits** → marge OK |
| Connexions simultanées | **1 seule** | ⚠️ **UNE seule connexion WS pour tout le backend** (multiplexée vers N clients). Interdiction d'ouvrir une WS par navigateur. |
| Symboles par souscription | ≤ 8 | 2 symboles → OK |
| Marchés | FX + crypto + métaux en temps réel/différé | XAU/USD + EUR/USD confirmés streamables |
| Plan requis « officiel » | WS = Pro (indiv.) / Venture (business) ; le trial donne un accès **limité de test** | ✅ suffisant pour le prototype DEV. ❌ **Le commercial exigera le plan Business** (cf. §5). |

**Implication architecture forte** : la limite **1 connexion** impose un **bridge WS côté backend**
(une connexion partagée, clé serveur) et **interdit** une WS ouverte depuis le navigateur — ce qui
au passage préserve la discipline existante « le front ne détient aucune clé »
(cf. `LIVE_DATA_WIRING_2026_06_08.md`).

---

## 2. État actuel — ce qui se met à jour, et quand

Aujourd'hui, **tout** se rafraîchit **à la clôture de bougie**, jamais au tick :

- **Backend** : `MarketReadingAssembler` → moteur SMC → schéma `MarketReading`. Bougies servies
  par `GET /api/candles` (lecture pure du cache SQLite, **0 appel TD**). Le scheduler régénère un
  combo quand une **nouvelle bougie clôture** (REST `time_series`, rate-limité 8/min · 800/jour).
- **Schéma zones** (`src/intelligence/market_reading_schema.py`) — déjà porteur des champs
  d'interaction, calculés **à la clôture** :
  - `OrderBlock` : `status` (`active|mitigated|invalidated`), `tested: bool`, `mitigated_at`.
  - `FairValueGap` : `status` (`active|partially_filled|filled`), `tested: bool`,
    **`fill_level`** (mèche la plus profonde dans la bande, read-only), `mitigated_at`.
- **Frontend** : `useMarketReading` / `useCandles` (`webapp/lib/market-reading/hooks.ts`)
  **re-fetch uniquement quand `candle_close_ts` change** — explicitement **pas de polling tick**.
- **Rendu zones** (`webapp/components/app/ReadingChart.tsx` + `webapp/lib/chart/zoneLayout.ts`) :
  - `buildZoneModels(structure)` → boîtes OB/FVG localisées (x = `created_at` → `mitigated_at`/barre courante).
  - **`openFvgBand(fvg)`** : si `status==='partially_filled'` et `fill_level` dans la bande,
    la boîte **rétrécit déjà** vers la portion ouverte (bullish : `high=fill_level` ; bearish : `low=fill_level`).
  - `tested` (`status!=='active'`) → style estompé « ghost » vs `active` net.

> **Conclusion clé** : toute la mécanique « FVG qui rétrécit » et « OB testé » **existe déjà**, mais
> n'est alimentée qu'à la clôture. Le prototype = **alimenter cette même géométrie en intra-bougie**
> à partir du prix live, sans rien recalculer côté détection.

---

## 3. Points d'intégration repérés (où brancher)

1. **Arrivée du prix live** *(nouveau, backend)* : bridge WS Twelve Data, **1 connexion partagée**,
   clé serveur. Voisin naturel de `src/intelligence/data_providers/twelve_data_provider.py` (REST),
   p.ex. `twelve_data_ws.py`. Maintient « dernier tick par instrument » en mémoire.
2. **Exposition au front** *(nouveau, backend)* : un flux **SSE** `GET /api/live-price?instrument=…`
   (prix + timestamp **descriptifs uniquement**), à côté de `src/api/routes/candles.py`. SSE car
   unidirectionnel serveur→client, multiplexable depuis l'unique connexion WS amont.
3. **Calcul de l'interaction live** : la logique de bande existe **déjà** côté front
   (`openFvgBand` dans `zoneLayout.ts`). Le tick live alimente une **variante provisoire** de cette
   même logique (prix dans la bande → rétrécit / OB « en cours de test »), **sans toucher** au
   `structure` issu de la clôture.
4. **Rendu** : `ReadingChart.tsx` ajoute un **état visuel distinct** « provisoire / live » (≠ `active`
   confirmé, ≠ `tested` confirmé). La ligne de prix courante existe déjà ; le tick l'anime.
5. **Garde-fous existants à préserver** : `useCandles`/`useMarketReading` keyés sur `candle_close_ts`
   restent le chemin par défaut ; le live est **additif** derrière flag.

---

## 4. Plan d'intégration proposé (POUR GO — non implémenté)

**Principe** : le live au tick **n'alimente QUE l'interaction de zones** (comblement FVG, touche OB).
Détection inchangée (clôture/REST/cache). BOS/CHOCH **aucun état live**.

1. **Flag / opt-in** (défaut = comportement actuel intact) :
   - Backend : `LIVE_TICK_ENABLED` (défaut `false`) — gate du bridge WS + route SSE.
   - Front : `NEXT_PUBLIC_LIVE_TICK` (défaut `off`) — gate de la souscription SSE + overlay.
   - **Mode par défaut = refresh à la clôture, strictement inchangé et fonctionnel.**
2. **Bridge WS backend** (`twelve_data_ws.py`) : **1** connexion `/quotes/price`, souscription
   `XAU/USD,EUR/USD`, reconnexion/backoff, dernier tick par instrument en mémoire. Clé **jamais**
   exposée au front.
3. **Route SSE** `GET /api/live-price` : pousse `{instrument, price, ts}` (descriptif). Une seule
   connexion amont → N abonnés SSE (respecte la limite 1-connexion trial).
4. **Interaction live, zones uniquement** :
   - **FVG** : `fill_level` **provisoire** recalculé au tick → la portion ouverte rétrécit en direct
     (réutilise la logique `openFvgBand`, marquée provisoire).
   - **OB** : badge « en cours de test » quand le prix entre dans `[level_low, level_high]`.
   - **BOS / CHOCH** : **rien** au tick. Jamais de « cassure en cours ». Validés à la clôture seule (loi SMC).
5. **Séparation d'honnêteté (critique)** :
   - État live = **provisoire / intra-bougie**, visuellement ET logiquement distinct du **confirmé**
     (clôture). Jamais afficher comme confirmé : invalidation d'OB par close-through, cassure
     BOS/CHOCH, forme finale de bougie.
   - **Réversibilité** : si le prix se retire avant la clôture, l'overlay live **revient proprement**
     à l'état clôture — aucun « confirmé » qui se dé-passe. Le `structure` clôturé n'est jamais muté
     par le tick (l'overlay live est une couche séparée).
   - À la clôture suivante, le REST/cache reprend l'autorité ; l'overlay provisoire se résorbe dans
     l'état confirmé (ou disparaît si le prix s'était retiré).
6. **Discipline** : pas de nouvelle détection au tick ; pas d'appel TD REST supplémentaire (le WS est
   un flux séparé) ; build + tests verts ; 0 régression sur le mode clôture par défaut.

**Note clock-skew** : l'interaction de zones repose sur une **comparaison de prix** (le prix est-il
dans la bande ?), pas sur le temps → le décalage horloge ~10 h connu sur ce poste
(`LIVE_DATA_WIRING_2026_06_08.md` §6.1) **n'affecte pas** le rétrécissement FVG / la touche OB.

---

## 5. Free tier / dev only — note commerciale (à documenter dans le code)

- Prototype **DEV / non commercial**. La sonde valide le trial, **pas** un usage production.
- WS « officiellement » = plan **Pro (indiv.) / Venture-Business**. Le trial = accès de test limité
  (1 connexion, 8 crédits). **Le passage commercial exigera le plan Business Twelve Data** (WS multi-
  connexions, quotas, SLA). Ne rien câbler qui suppose un plan payant ; documenter ce prérequis.

---

## 6. ✅ IMPLÉMENTATION LIVRÉE (étape 2, après GO)

Tout est **additif et derrière flag** ; le mode par défaut (refresh à la clôture) est **inchangé**.

### Backend
- **`src/intelligence/data_providers/twelve_data_ws.py`** (nouveau) — `TwelveDataLiveTickBridge` :
  **1 connexion WS partagée** (thread daemon + event loop dédié), dernier tick par instrument
  thread-safe (`get_latest`), reconnexion backoff, parsing tolérant (`_handle_message` pur). Réutilise
  `_SYMBOL_MAP` du provider REST (XAU/USD↔XAUUSD). Clé serveur, jamais exposée.
- **`src/api/routes/live_price.py`** (nouveau) — `GET /api/live-price?instrument=…` en **SSE**
  (`text/event-stream`), payload **descriptif** `{instrument, price, ts}`, keepalive 15 s,
  `400` hors périmètre / `503` si bridge absent (flag off). Formatteur pur `format_price_event`.
- **`src/api/bootstrap.py`** — `is_live_tick_enabled()` (`LIVE_TICK_ENABLED`, défaut **off**) +
  `build_live_tick_bridge()` (lit `TWELVE_DATA_API_KEY`, `LIVE_TICK_INSTRUMENTS`).
- **`src/api/app.py`** — bootstrap gated dans le lifespan, **start** après enregistrement du
  shutdown, **stop** câblé dans le `GracefulShutdownCoordinator` (libère le slot 1-connexion).
  Router monté. **`dependencies.py`** : champ `live_tick_bridge`. **`.env.example`** documenté.

### Frontend (opt-in `NEXT_PUBLIC_LIVE_TICK`, défaut off)
- **`webapp/lib/market-reading/live-price.ts`** (nouveau) — `useLivePrice` (client EventSource SSE,
  flag), prix live `null` quand off → chart strictement identique au mode clôture.
- **`webapp/lib/chart/zoneLayout.ts`** — fonctions **pures** : `provisionalOpenFvgBand` (rétrécit le
  FVG vers la portion encore ouverte au prix courant ; **null sur retrait** → revient à la géométrie
  clôturée), `isObInTestLive` (OB **actif** touché en direct), `buildLiveOverlay`.
- **`webapp/components/app/ReadingChart.tsx`** — overlay **provisoire** distinct : front de
  comblement FVG (ambre chaud) rétrécissant en direct + badge OB « • en test » + pastille
  « EN DIRECT · provisoire ». Recalculé du prix courant chaque frame (rAF) → **réversible**.
  `ReadingColumn.tsx` câble `useLivePrice` (désactivé en mock).

### Honnêteté tenue
- **FVG / OB seulement.** **BOS / CHOCH : aucune mise à jour au tick** (validés à la clôture).
- Live = **provisoire/intra-bougie**, hue ambre **distincte** du palette clôturé (cool slate/blue) ;
  jamais affiché comme confirmé ; le `structure` clôturé n'est **jamais muté** ; retrait → revient proprement.
- Descriptif pur (prix réel), zéro champ prédictif dans le flux SSE.

### Vérifications
- Backend : `tests/test_live_tick_bridge.py` + `tests/test_live_price_endpoint.py` → **18 passés**.
- Frontend : suite vitest complète **190 passés / 26 fichiers**, `tsc --noEmit` **OK**, `next build` **vert**
  (`/app` 12.1 kB / 161 kB). `zoneLayout` **24 passés**.
- **0 régression** mode par défaut (2 échecs smoke health/scanner **préexistants**, vérifiés sur l'arbre sans la branche).
- **End-to-end réel** : bridge connecté au WS, ticks XAU **4319.09** + EUR **1.15229** peuplés via
  `get_latest`, `stop()` propre (`running=False`).

### Free tier / dev only
`LIVE_TICK_ENABLED=false` par défaut. WS trial = **1 connexion** (respectée par le bridge partagé).
**Le commercial exigera le plan Business Twelve Data** (documenté dans le code + `.env.example`).
