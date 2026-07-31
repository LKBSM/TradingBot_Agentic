# AUDIT REC-1 — Restauration du chargement des données

Date : 2026-07-30 · Branche : `fix/rec-1-restauration` (depuis main `f9cd22f`)

Symptôme : l'UI charge, la navigation fonctionne, mais **toutes** les pages
produit restent bloquées sur les squelettes de chargement, **sans aucun message
d'erreur**. Ce n'est pas un échec de requête : c'est un état de chargement qui ne
se termine jamais.

---

## 1. Cause racine — PLUSIEURS causes superposées

Le symptôme « toutes les pages, squelette perpétuel, aucune erreur » résulte de
la conjonction de trois éléments (deux latents pré-existants + une régression
aggravante récente).

### (a) DIRECT — `fetchAccess` sans timeout
`webapp/lib/access/api-client.ts` : `fetchAccess` (appelé par le
`SubscriptionGate` qui enveloppe **chaque** page produit) faisait un `fetch`
**sans AbortController ni timeout**. Tant que `/api/access/me` ne répond pas, le
gate reste `access === null` → **spinner perpétuel sur toutes les pages, sans
erreur** (il ne rejette jamais). C'est la cause immédiate du symptôme.
*Pré-existant (commit 8c2f410, beta lockdown) — latent jusqu'ici.*

### (b) MÉCANISME — l'event loop gelé par un endpoint `async` bloquant
`src/api/routes/market_reading.py` : `get_market_reading` était `async def` mais
appelait `assembler.get_or_generate()` de façon **synchrone bloquante**
(détection + fetch provider, ~1,7 s). Un endpoint `async` exécute son corps **sur
l'event loop** : pendant toute la construction d'une lecture, **aucune** autre
requête ne peut être servie — y compris `/api/access/me` (lui aussi `async`).
*Architectural, pré-existant (commit 4f88e7c, Chantier-2).*

**Preuve (vrai serveur uvicorn, mesurée) :**
| Endpoint lent (2 s) | Latence d'un `/probe` concurrent |
|---|---|
| `async def` (ancien comportement) | **1,85 s** — bloqué |
| `def` (nouveau, threadpoolé) | **0,22 s** — non bloqué |

### (c) AGGRAVANT — régression MT-D1 (`49a5408`, PR #105)
`src/intelligence/lookback_config.py`, `analysis_window_bars()` : la fenêtre
d'analyse **par requête** est passée de **500 fixe → jusqu'à 3000 bougies**
(M15≈2880, M5/M1=3000), + journal d'événements non plafonné. Chaque construction
de lecture fait donc un fetch provider et une détection **~6× plus lourds** →
l'event loop est gelé ~6× plus longtemps par requête.

**Multiplicateur — invalidation totale du cache.** Les PR #103/#104/#106
(LQ-D1, TR-1) ont bumpé `READING_LOGIC_VERSION 2→3`
(`market_reading_assembler.py:41`). Toutes les lectures stockées (avant ces
merges) sont invalidées → **chaque** requête reconstruit désormais (avant :
cache-hit instantané, sans blocage). À la première ouverture de /app après ces
merges, l'event loop est saturé de reconstructions lourdes → `access_me` ne passe
plus → squelette partout.

### Ce qui n'est PAS la cause (mesuré / vérifié)
- **Volume de réponse** : `/api/market-reading` XAUUSD M15 = **~1,7 s · 20 Ko · 24
  zones** (borne basse, provider stubé). Réponse **bornée**, pas volumineuse.
  L'hypothèse « historique profond non borné » est **infirmée**.
- **Bornage des requêtes** : `/api/candles` borné (front 400, serveur 500/1000) ;
  `/api/structure` et `/api/coverage` bornés côté serveur, sans détection.
- **Détection** : 416 ms sur 2233 bougies M15. Rapide.
- **Variable d'env manquante** : NON — toutes les clés `.env` sont présentes.

---

## 2. Corrections appliquées (minimales, une à la fois, vérifiées)

Aucune n'invente de donnée, n'avale d'erreur, ni ne touche aux règles de
détection.

### Correction 1 — `get_market_reading` : `async def` → `def`
`src/api/routes/market_reading.py`. FastAPI exécute un endpoint `def` dans son
threadpool → une lecture lente ne gèle plus l'event loop ; `/api/access/me` et
les autres pages sont servis en parallèle. **Zéro changement de donnée/logique.**
- *Vérifié* : test de concurrence (vrai uvicorn) — probe concurrent 0,22 s (vs
  1,85 s bloqué avant). Garde de non-régression :
  `test_endpoint_is_sync_so_it_never_blocks_the_event_loop`.

### Correction 2 — `fetchAccess` : timeout 8 s + abandon
`webapp/lib/access/api-client.ts`. Borne l'attente : à l'expiration, le `fetch`
est abandonné et rejette → le `SubscriptionGate` atteint une décision (fail-open
vers les enfants hors lockdown, selon son design existant) au lieu de tourner
indéfiniment. Répond au point 4 : une requête sans réponse ne laisse plus jamais
l'UI en chargement perpétuel.
- *Vérifié* : garde `REC-1: aborts on timeout so the gate can never spin forever`.

### Non touché (délibérément) — `analysis_window_bars` (MT-D1)
C'est une **fenêtre de détection** ; la mission interdit de modifier les règles de
détection. Signalé comme aggravant, à arbitrer par le propriétaire si l'on veut
réduire le coût par requête.

---

## 3. Surfaces vérifiées

**Limite d'honnêteté** : je n'ai pas pu faire la vérification LIVE complète car
(1) `.env` a `DATA_SOURCE=MT5` → le backend exige le terminal MetaTrader, absent
de mon environnement (chez le propriétaire il tourne) ; (2) le `candles.db` local
(3 juil.) n'est pas backfillé profond (pas de D1). J'ai vérifié ce qui l'était :

| Surface | Vérifié ? | Détail |
|---|---|---|
| `/api/market-reading` XAUUSD/EURUSD M15/H4/H1 | ✅ | 200, 8–18 Ko, zones réelles (TestClient + vraies bases) |
| `/api/candles` (400 bougies) | ✅ | 200, ~42 Ko |
| `/api/coverage`, `/api/structure`, `/api/market-status` | ✅ | 200 |
| Concurrence event-loop | ✅ | probe 0,22 s pendant lecture lente |
| D1 / M5 | ❌ | pas de données locales |
| Chart (bougies/zones/liquidité/BOS/CHOCH) rendu | ❌ | nécessite le front live |
| Panneau Régime, lecture narrée | ❌ | narration = Anthropic ; front live |
| M.I.A Agent, scanner, Zones, Actualités | ❌ | nécessitent le backend MT5 live |

**À valider en live par le propriétaire** : ouvrir /app (XAU/USD, EUR/USD, les 6
unités), confirmer que le contenu s'affiche au lieu du squelette, puis parcourir
Régime, lecture narrée, M.I.A Agent, scanner, Zones, Actualités.

### Point 5 — unité M1 manquante
**Délibéré.** `M1 ∈ supported_timeframes` mais **hors `enabled_combos`** (M5..D1) —
cohérent avec LB-1 « M1 off ». Pas une conséquence de la régression. Signalé, non
corrigé.

---

## 4. Section 6 — pourquoi la panne est passée, et comment l'empêcher

**Existe-t-il un test de fumée qui vérifie qu'une page charge avec de vraies
données ? Non.** Pourquoi 285 tests back + 35 front sont passés alors que le
produit est inutilisable :

1. **Aucun test ne mesure la CONCURRENCE.** Les tests backend utilisent
   `TestClient` (in-process, une requête à la fois) : le gel de l'event loop ne
   se manifeste que quand une requête lente et une requête légère s'exécutent
   **en même temps**. Aucun test ne lance `/api/access/me` concurremment avec un
   `/api/market-reading` lent. La preuve nécessite un vrai serveur + 2 requêtes
   simultanées (ce que fait ma vérification).
2. **Aucun test ne mesure la LATENCE sous volume réel.** Les fixtures sont
   petites ; MT-D1 (500→2880) n'a pas dégradé les tests.
3. **Le front mocke `fetch` (instantané).** Le gate sans timeout ne se manifeste
   jamais en test unitaire — il faut une réponse qui n'arrive pas.

**Ce qu'il faudrait pour détecter automatiquement un merge qui casse le
chargement (sans l'implémenter ici) :**

- **Test de fumée de concurrence** : vrai serveur (uvicorn) + un endpoint lent
  simulé + une sonde légère concurrente → échoue si la sonde dépasse un seuil
  (ex. 300 ms). Aurait attrapé l'event-loop gelé.
- **e2e « la page affiche des données »** : Playwright charge /app contre un
  backend réaliste et assure que le contenu (pas le squelette) apparaît sous N s.
  Aurait attrapé le symptôme directement.
- **Budget de latence sur `/api/market-reading`** avec un volume réaliste →
  échoue si une lecture dépasse un seuil. Aurait signalé l'aggravation MT-D1.
- **Garde « endpoint async non bloquant »** : lint/test interdisant un appel
  synchrone lourd (détection, fetch provider) dans un `async def`. La garde
  `test_endpoint_is_sync_so_it_never_blocks_the_event_loop` en est une amorce.
- **Timeout obligatoire sur tout fetch de gate** : garde front que tout appel
  bloquant le rendu porte un timeout (la garde ajoutée sur `fetchAccess` en est
  une amorce).

Le vrai enseignement : les tests actuels valident la **correction fonctionnelle**
d'une unité isolée, jamais le **comportement du système sous charge concurrente
avec des données réelles** — exactement la dimension où cette panne vit.
