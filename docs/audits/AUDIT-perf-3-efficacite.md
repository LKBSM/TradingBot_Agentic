# AUDIT PERF-3 — Efficacité : appels fournisseur de données & tokens Anthropic

**Nature** : diagnostic en lecture seule. Aucune ligne de code de production modifiée.
**Branche** : `docs/perf-3-audit-efficacite` (worktree dédié `wt-perf-3`, depuis `origin/main` = `da8bbe9`).
**Date** : 2026-09-09.

> **Discipline d'audit** — le HEAD du dépôt principal était **73 commits derrière `origin/main`**
> (`e0dc69c` vs `da8bbe9`). Tout ce rapport est lu contre `origin/main` à jour, jamais contre ce HEAD.

---

## 0bis. Vérification de dépendance — localisation des fichiers

| Élément | Fichier exact |
|---|---|
| `build_chatbot` | `src/api/bootstrap.py:200` → `src/intelligence/chatbot/chatbot.py` (61 688 o) |
| `build_scanner_translator` | `src/api/bootstrap.py:253` → `src/intelligence/scanner_translator.py` (563 l.) |
| Agent démo vitrine (3ᵉ appelant LLM) | `src/api/bootstrap.py:227` → `src/intelligence/chatbot/demo_agent.py` |
| Polling des données | `src/intelligence/scheduler.py` (255 l.) + `src/intelligence/market_reading_assembler.py` (1014 l.) |
| Fournisseur REST | `src/intelligence/data_providers/twelve_data_provider.py` (343 l.) |
| Pont WebSocket (existant) | `src/intelligence/data_providers/twelve_data_ws.py` (273 l.) → `src/api/routes/live_price.py` |
| `candles.db` | `src/storage/candles_cache_store.py` ; chemin prod `/app/data/candles.db` (`CANDLES_DB_PATH`, disque persistant Render 10 Go) |
| Périmètre marchés / unités | `config/markets.json` (**2 marchés**), `config/lookback_depths.json`, `src/intelligence/lookback_config.py` |
| Config de production réelle | `render.yaml` — backend `uvicorn src.api.asgi:app` |

**Point d'entrée de production** : `render.yaml` lance `uvicorn src.api.asgi:app`. Le module
`src/intelligence/main.py` (ancien pipeline Sentinel + `LLMNarrativeEngine`) **n'est pas déployé** ;
les `CMD ["python","-m","src.intelligence.main"]` des deux Dockerfile racine sont des vestiges.
Conséquence directe pour la partie B : **les seuls appelants Anthropic en production sont le chatbot
M.I.A, le traducteur du scanner et l'agent démo de l'accueil.**

### Journaux et métriques disponibles

| Source | Disponible ? |
|---|---|
| Compteur de crédits Twelve Data | **Non.** Aucun compteur, aucun log par appel. Le seul signal est un `logger.info` du limiteur *quand il dort déjà* (`twelve_data_provider.py:104`). |
| `usage.cache_read_input_tokens` / `cache_creation_input_tokens` | **Non**, sur aucun des deux appelants de production. Seul `llm_narrative_engine.py:515` le lit — et ce module n'est pas déployé. |
| Latence par route | Oui — `LatencyTracker` + `/api/v1/metrics/latency` (HMAC admin). Ne dit rien des crédits ni des tokens. |
| `/health`, `/health/deep` | Circuit breakers + `cost_quota` par palier. **Rien sur Twelve Data.** |
| `logs/` | Vide (`.gitkeep` seul). Logs prod = flux Render, non capturés ici. |

**Les chiffres de la partie A sont donc dérivés du code + de la configuration réelle de production
(`render.yaml`), pas de compteurs.** En revanche, les tailles de charge utile de la partie B sont
des **mesures directes** sur les bases locales (`data/market_readings.db`, `data/candles.db`,
rafraîchies le 2026-09-08), et les tailles de prompt sont des **comptes de caractères exacts**.

---

# PARTIE A — Appels au fournisseur de données (Twelve Data)

## A.0 État mesuré

### Périmètre réel

`config/markets.json` ne contient que **deux marchés** : `XAUUSD` et `EURUSD`. Le mapping du
fournisseur est d'ailleurs codé en dur à ces deux symboles (`twelve_data_provider.py:31`,
`_SYMBOL_MAP`) : un 3ᵉ marché lèverait `ValueError` **avant** toute dépense de crédit.
L'hypothèse « 80 marchés » de la section 14 est donc aujourd'hui une projection de plan, pas un
état du code.

Unités activées : `M5, M15, H1, H4, D1` (M1 fermé par la porte `LB1_ENABLE_M1`).
Combos entretenus en continu (`live_warm_combos()`, M5 exclu) : **8**.

### Budget REST en régime permanent (marché ouvert)

Le scheduler tourne toutes les 60 s (`SCHEDULER_TICK_INTERVAL_SECONDS` non surchargé dans
`render.yaml` → défaut 60) et ne régénère un combo **que si une bougie s'est réellement fermée**
(`_needs_regeneration`, verrou MC-1 market-aware : un week-end coûte **zéro** appel sortant).
Une régénération = **un** `fetch_candles`.

| Unité | Fermetures / jour | Appels / marché / jour |
|---|---|---|
| M15 | 96 | 96 |
| H1 | 24 | 24 |
| H4 | 6 | 6 |
| D1 | 1 | 1 |
| **Total** | | **127** |

→ **254 appels / jour** pour les 2 marchés, soit **31,8 % du plafond gratuit de 800/jour**.

**Pic minute** : à 00:00 UTC, M15 + H1 + H4 + D1 ferment simultanément sur les deux marchés →
**8 appels dans un seul tick**, soit exactement le plafond du limiteur (8/min). À chaque heure ronde :
4 appels/min.

**Le cache TTL du fournisseur ne dédoublonne jamais ce chemin.** `TWELVE_DATA_CACHE_TTL_S`
vaut 300 s par défaut (`twelve_data_provider.py:163`) et n'est pas surchargé en prod ; l'intervalle
le plus court entre deux fermetures sur un combo *warm* est de 900 s (M15). Le TTL ne sert donc
qu'au chemin interactif (démarrage à froid, bascule de `_logic_version`).

### A.1 — Le jitter n'est pas implémenté

**Vérifié** : `grep -rn "jitter\|random\." src/intelligence src/api src/storage` → **zéro occurrence**
(hors une phrase de documentation dans `session_auth.py`). Le scheduler parcourt ses combos
**séquentiellement dans le même tick**, sans décalage.

Le seul étalement qui existe est un effet de bord du **limiteur bloquant** : quand les 8 créneaux
de la minute sont pris, `TwelveDataRateLimiter.acquire()` **dort** jusqu'à libération. Ce n'est pas
du jitter — c'est une file d'attente qui allonge le tick au lieu de lisser la charge.

**Effet réel sur les pics observés, à 2 marchés** : le pic est de 8 appels/minute une fois par jour
(00:00 UTC). Le jitter n'apporterait **rien de mesurable aujourd'hui**. Il devient nécessaire au-delà
de ~8 marchés.

**Vérification du chiffre « ~16 crédits/min pour 80 marchés × 5 unités avec jitter »** — cohérent
comme *moyenne*, à condition de bien voir ce que le jitter fait et ne fait pas :

| Scénario 80 marchés × 5 unités | Sans jitter (code actuel) | Avec jitter |
|---|---|---|
| Pic à la fermeture M15 | **80 appels dans la même minute** | ~5,3/min étalés sur 15 min |
| Pic à l'heure ronde (M15+H1) | **160 appels/min** | ~11/min |
| Pic à 00:00 UTC (4 unités) | **320 appels/min** | ~21/min |
| **Total / jour** | **10 160** | **10 160 — inchangé** |

> **Le jitter aplatit le pic, il ne réduit pas la facture.** C'est un correctif de *forme*
> d'appel, pas de *volume*. À ne pas confondre avec les leviers A-1/A-2 ci-dessous, qui eux
> retirent réellement des appels.

### A.2 — Plusieurs sessions simultanées ne dupliquent PAS les appels

**C'est la bonne nouvelle de l'audit, et elle est solide.**

Un seul `TwelveDataProvider` est construit au démarrage (`bootstrap.py:110`) et stocké dans
`app_state` : un cache TTL partagé, un limiteur partagé, pour tout le processus. En amont,
**PERF-2** (`market_reading_assembler.py:395-410`) fait que le chemin interactif ne touche
**jamais** le fournisseur quand une lecture de la version de logique courante est déjà stockée :
il sert la lecture SQLite instantanément et se contente de marquer le combo actif pour que le
scheduler l'avance hors du chemin de la requête.

Conséquence chiffrée : **N utilisateurs sur le même marché = N lectures SQLite, 0 appel Twelve Data.**
Le front interroge `/api/market-reading` toutes les 60 s par page ouverte (`AppWorkspace.tsx:24`,
`ZonesWorkspace.tsx:39`, `CalendarWorkspace.tsx:38`) — cela ne produit aucun trafic sortant.

**La faille la plus coûteuse que la mission redoutait n'existe pas.** Deux réserves honnêtes :

1. **Démarrage à froid / bascule de `_logic_version`** : là, PERF-2 laisse volontairement passer la
   reconstruction. N requêtes concurrentes sur un combo vide déclenchent N `fetch_candles`
   concurrents — le cache TTL n'est renseigné qu'*après* retour, il n'y a pas de verrou par combo.
   Borné dans les faits par `_FETCH_POOL_WORKERS = 4` et par le limiteur, et le TTL de 300 s
   referme la fenêtre ensuite. Risque réel mais court et rare.
2. **`/api/candles` sur D1/W1** : seul chemin interactif qui peut encore appeler le fournisseur
   (`candles.py:143` → `warm_candles`), quand le cache est vide sur une série de référence. Idempotent
   et market-aware : au plus un appel par bougie D1/W1 fermée.

### A.3 — Une unité de temps = un appel REST séparé (confirmé), et la dérivation depuis M5 est bien perdante

**Confirmé par le code** : `_build_fresh` → `_fetch_candles_for_build(instrument, timeframe, window)`
→ `provider.fetch_candles(instrument, timeframe, window)`. Aucune agrégation. Chaque unité paie
son propre crédit, avec sa propre fenêtre :

| Unité | `analysis_window_bars` (`outputsize` envoyé) |
|---|---|
| M15 | 2 880 |
| H1 | 720 |
| H4 | 500 |
| D1 | 500 |

*(Twelve Data facture à la requête, pas à la barre : une fenêtre large ne coûte pas plus cher.)*

Une seule dérivation existe déjà et ne coûte rien : `build_cache_mtf_provider` alimente
`regime.mtf_confluence` **en lisant `candles.db`**, sans appel supplémentaire.

**Vérification de la conclusion de la section 14 contre le code réel — elle tient :**

| Stratégie | Appels / marché / jour |
|---|---|
| Aujourd'hui : 4 unités interrogées séparément | **127** |
| Tout dériver d'un flux M5 interrogé en REST | **288** (M5 ferme 288 ×/jour) |

Dériver depuis M5 **par REST coûte 2,3 × plus cher**. Le commentaire de `lookback_config.py:216`
le disait déjà (« ~288 req/day per symbol … over the 800/day free cap ») ; le code confirme.

**Mais la conclusion mérite une correction, et elle est favorable.** La bonne base de dérivation
n'est pas M5, c'est **M15** — l'unité déjà interrogée, et déjà stockée en profondeur.
Mesure sur `candles.db` (local, 2026-09-08) :

```
XAUUSD M15  n= 177 986   2019-01-02 -> 2026-09-08
EURUSD M15  n= 179 618   2019-01-01 -> 2026-09-08
XAUUSD H1   n=     933   2026-05-25 -> 2026-07-03   (figé depuis juillet)
XAUUSD H4   n=     806   2026-03-24 -> 2026-07-03   (figé depuis juillet)
```

Les ingrédients de H1 (4 × M15), H4 (16 × M15) et D1 sont **déjà dans la base**, sept ans en arrière.
Les dériver supprimerait 31 appels/marché/jour :

| | Appels/jour (2 marchés) |
|---|---|
| Aujourd'hui | 254 |
| M15 seul + dérivation H1/H4/D1 | **192 (−24 %)** |
| À 80 marchés | 10 160 → **7 680 (−24 %)** |

Une fonction `resample_ohlcv` existe déjà (`src/intelligence/volatility_forecaster.py:179`).
**Réserve à lever avant de décider** : la frontière D1. Une D1 dérivée de M15 se ferme à minuit UTC ;
la D1 de Twelve Data peut suivre une convention de place. À réconcilier sur données réelles,
jamais à supposer.

### A.4 — Appels redondants identifiés

| # | Redondance | Fichier | Coût |
|---|---|---|---|
| R1 | **Fuite de quota M5 par l'ensemble actif.** M5 est exclu du *warm* précisément parce qu'il coûterait ~830/j > 800 (`lookback_config.py:216`). Mais le tick régénère `active ∪ always_warm`, et un combo visité **une fois** reste actif 24 h (`SCHEDULER_AUTO_STOP_HOURS=24`). Une seule visite sur M5 → 288 appels/marché/jour pendant 24 h. `active_combinations` contient **M5 pour les deux marchés** aujourd'hui. | `scheduler.py:165-180` + `lookback_config.py:230` | **254 → 830/j : dépassement du plafond gratuit.** Le pire de la partie A. |
| R2 | **Fenêtres désalignées → clés de cache TTL distinctes.** `warm_candles` et `refresh_if_reopened` appellent `fetch_candles(..., self._lookback)` = **500**, alors que les constructions utilisent `analysis_window_bars(tf)` (M15 = 2 880, H1 = 720). Clé `(symbol, tf, lookback)` différente ⇒ **le cache TTL ne peut jamais être partagé entre ces chemins** : appel REST garanti. | `market_reading_assembler.py:523` et `:586` | 1 crédit gaspillé par sonde/warm sur M15 et H1 |
| R3 | **Second `TwelveDataProvider` instancié dans une route.** Le backfill des mesures de publication construit sa propre instance → **second limiteur 8/min indépendant + cache TTL vide**. Les deux limiteurs additionnés peuvent dépasser le plafond réel du plan. | `src/api/routes/calendar.py:251` | Déclenché par opérateur, mono-vol (garde `_BACKFILL_STATE`), mais casse le compteur global |
| R4 | **Le retry reconsomme un créneau.** `self._rate_limiter.acquire()` est **dans** la boucle de retry : une requête logique qui échoue 4 fois consomme 4 créneaux du budget local. Le backoff exponentiel lui-même est correct (1→2→4→8 s, `MAX_RETRIES=4`), et 401/403 échoue vite sans retry. | `twelve_data_provider.py:263` | Amplification × 4 sous incident |
| R5 | **Le limiteur dort en tenant son verrou.** `acquire()` fait `with self._lock:` puis `sleep()` **à l'intérieur**. Sur le plafond journalier, `sleep_s` peut atteindre ~86 400 s : tout appelant du fournisseur dans le processus est gelé. La dégradation reste honnête (le fetch interactif expire à 5 s et lit `candles.db`), mais le mode de panne est brutal. | `twelve_data_provider.py:81-115` | Risque de disponibilité, pas de crédit |
| R6 | **Front : `useMtfTrends` retélécharge des lectures entières pour un seul champ.** Pour afficher les flèches de tendance des unités supérieures, le hook fait un `fetchMarketReading` complet par unité et n'en lit que `regime.trend`. Charge utile mesurée : **10 001 à 11 471 caractères en moyenne** par lecture. Deux implémentations indépendantes (`RegimeCard.tsx:240` sur /app, `RegimeSection.tsx:90` sur les surfaces carte) — **pas** co-rendues, donc pas un double appel sur une même page, mais deux fois le même gaspillage à maintenir. | `webapp/lib/market-reading/hooks.ts:221` | Serveur + bande passante uniquement (0 crédit Twelve Data) |

**Ce qui n'est PAS redondant** (vérifié, à ne pas « corriger ») :
`prewarm_publication_measures` lit `candles.db` seul ; le démon de backfill profond est **désactivé
par défaut** et cadencé (2 pages/marché/15 min) ; la sonde de réouverture est limitée aux fériés,
1 fois/30 min/combo ; un week-end produit zéro appel sortant ; le SSE `/api/live-price` multiplexe
**une** connexion WS partagée vers N clients.

### A.5 — Cartographie d'un passage au flux WebSocket

**Le pont existe déjà** — c'est l'élément décisif de cette cartographie.
`twelve_data_ws.py` : une connexion unique au niveau du processus, souscription multi-symboles,
reconnexion à backoff plafonné (1 → 30 s), dernier tick par instrument en mémoire, lecture
thread-safe `get_latest`. Il ne transporte **que le dernier prix** et **n'écrit pas dans `candles.db`**.

**La couture d'insertion existe aussi, et elle est déjà écrite.** `_fetch_candles_for_build`, sur le
chemin de fond (`bound_provider=False`), **saute déjà l'appel REST** quand `candles.db` est à la fois
assez profond et à jour vis-à-vis de la fermeture market-aware
(`market_reading_assembler.py:691-716`). Le commentaire du code nomme lui-même le chemin :

> « *the live WebSocket feeds only the last PRICE (/api/live-price), NOT candles.db — wiring it to
> build M15 bars into candles.db (then resampling H1/H4/D1 from them) is the path to ~zero
> steady-state REST.* »

**Ce qu'il faudrait écrire, concrètement :**

| Où | Quoi | Complexité |
|---|---|---|
| `twelve_data_ws.py` | Émettre les ticks vers un consommateur (rappel), au lieu de ne garder que le dernier | Faible |
| **Nouveau module** `live_bar_aggregator.py` | Tick → barre M15 OHLCV ; fermeture d'une barre sur frontière temporelle ; `upsert_candles` dans `candles.db` | **Le vrai travail.** C'est là que vit l'agrégation. |
| **Nouveau** — dérivation | H1/H4/D1 par ré-échantillonnage depuis M15 ; `resample_ohlcv` existe déjà (`volatility_forecaster.py:179`) | Moyenne — la frontière D1 est le piège |
| `bootstrap.build_live_tick_bridge` | Injecter `CandlesCacheStore` dans le pont | Faible |
| **Nouveau** — réconciliation REST | Une passe périodique qui compare les barres agrégées aux barres officielles Twelve Data et corrige les écarts | **Non négociable** — un trou WS ne doit jamais produire silencieusement une fausse bougie |
| `scheduler.py`, `market_reading_assembler.py` | **Rien à changer.** La lecture-traversante de fond éteint le REST d'elle-même dès que le cache est à jour. | Nulle |

**Gain** : de 254 appels/jour à ~0 en régime permanent (hors réconciliation), et surtout un coût qui
cesse de croître avec le nombre d'unités de temps.

**Trois contraintes à porter au dossier, pas à contourner** :
1. Le fichier le dit : le WS Twelve Data est officiellement Pro/Business ; le palier d'essai gratuit
   n'autorise **qu'une connexion** et sert au test. **Un lancement commercial exige le plan Business.**
2. Une barre agrégée depuis des ticks n'est pas bit-à-bit la barre officielle du fournisseur
   (ticks manqués, reconnexions, révisions). Sans réconciliation, on remplace une facture par une
   dette de justesse — inacceptable pour un produit qui affiche du BOS/CHOCH.
3. La loi SMC du projet reste intacte : le tick ne sert jamais à détecter de la structure, seulement
   à construire la bougie. La détection reste à la clôture.

## A.6 — Recommandations partie A (classées par gain, aucune appliquée ici)

| # | Recommandation | Gain | Nature |
|---|---|---|---|
| **A-1** | **Corriger la fuite M5 (R1)** : exclure M5 de la régénération planifiée même quand il est dans l'ensemble actif, ou raccourcir `auto_stop_hours` pour M5. | Évite un dépassement dur 830 > 800/j | **mécanique** |
| **A-2** | **Dériver H1/H4/D1 depuis M15 stocké** au lieu de les interroger. | −24 % d'appels (254 → 192 ; 10 160 → 7 680 à 80 marchés) | **architecture** (frontière D1 à réconcilier) |
| **A-3** | **Agrégateur de bougies depuis le WebSocket** vers `candles.db` + réconciliation REST. | 254 → ~0 en régime permanent ; coût découplé du nombre d'unités | **architecture** (+ plan Business) |
| **A-4** | Aligner le `lookback` de `warm_candles` / `refresh_if_reopened` sur `analysis_window_bars` (R2). | 1 crédit par sonde | **mécanique** |
| **A-5** | Injecter le fournisseur partagé dans le backfill de `calendar.py` (R3). | Restaure un compteur unique | **mécanique** |
| **A-6** | Ne pas reprendre de créneau de limitation à chaque retry (R4). | ×4 sous incident | **mécanique** |
| **A-7** | Étalement (jitter) du tick du scheduler. | **Nul aujourd'hui.** Indispensable au-delà de ~8 marchés. | **mécanique** |
| **A-8** | Ne pas dormir en tenant le verrou du limiteur (R5). | Disponibilité | **architecture** (légère) |
| **A-9** | Compteur de crédits Twelve Data exposé sur `/health`. | Rend tout le reste mesurable | **mécanique** |
| **A-10** | Servir les tendances des unités supérieures dans la charge utile de la lecture (R6). | −N requêtes serveur de ~11 ko par changement de combo | **mécanique** |

---

# PARTIE B — Tokens Anthropic

## B.0 État mesuré

Tous les chiffres ci-dessous sont des **comptes de caractères exacts** obtenus en important les
modules réels ; les charges utiles d'outil sont mesurées sur `data/market_readings.db` (données
réelles, 2026-09-08).

> **Limite de mesure, déclarée** : il n'y a **aucune clé Anthropic dans cet environnement**
> (`ANTHROPIC_API_KEY` non défini, `ant` absent). Les comptes de **tokens** exacts exigent
> `client.messages.count_tokens` et ne peuvent pas être produits ici. Les conversions sont donc
> encadrées par deux bornes (4 car./token, la règle Anthropic ; 3,3 car./token, plus réaliste pour du
> français dense). Le script de vérification exacte est fourni en annexe — **à exécuter dans
> l'environnement du fondateur**, c'est le préalable de toute décision B.

### Chatbot M.I.A — `chatbot.py`

| Élément | Caractères | ≈ tokens (4 c/t) | ≈ tokens (3,3 c/t) |
|---|---|---|---|
| `SYSTEM_PROMPT_STATIC` | **9 588** | 2 397 | 2 905 |
| Schémas des 7 outils (JSON) | **7 460** | 1 865 | 2 261 |
| **Préfixe mis en cache (outils + statique)** | **17 048** | **4 262** | **5 166** |
| Bloc `signal_summary` (10 combos) — *hors cache, par tour* | 2 871 | 718 | 870 |

Modèle `claude-haiku-4-5-20251001` · `max_tokens=768` · `timeout=20 s` · historique plafonné
serveur à 12 messages (le client peut en envoyer 20 × 2 000 car.) · `MAX_TOOL_TURNS=3`.
Les 7 outils : `get_market_reading`, `get_signal_summary`, `get_ob_diagnostic`, `apply_chart_view`,
`list_markets`, `get_economic_calendar`, `get_publication`.

### Traducteur du scanner — `scanner_translator.py`

| Élément | Caractères | ≈ tokens (4 c/t) | ≈ tokens (3,3 c/t) |
|---|---|---|---|
| `SYSTEM_PROMPT` | 2 138 | 534 | 648 |
| Schéma d'outil (palette de 22 conditions) | 4 402 | 1 100 | 1 334 |
| **Préfixe total** | **6 540** | **1 635** | **1 982** |

Modèle `claude-haiku-4-5-20251001` · `max_tokens=1024` · `timeout=20 s` · `tool_choice` forcé ·
`CircuitBreaker(3, 60 s)` · déclenché par bouton (`ConversationalScanner.tsx:81`), pas à la frappe —
il n'y a pas de traduction « au fil de la saisie » (le `live` de SC-4 est le compteur de combinaisons,
`useLiveComboCount`, qui n'appelle aucun LLM).

### B.1 — Le prompt caching : présent côté chatbot, **absent côté traducteur**, et **non vérifiable** des deux côtés

**Chatbot** : `cache_control:{type:"ephemeral"}` **est** émis, sur le bloc système statique
(`chatbot.py:765` → `llm_cost_policy.cache_block_for`). MIA-2 a correctement placé le bloc variable
`signal_summary` **après** le point de rupture. L'intention est juste. Mais le garde-fou qui décide
d'émettre le marqueur comporte **deux erreurs qui se compensent aujourd'hui par accident** :

```python
# src/intelligence/llm_cost_policy.py
CACHE_MIN_TOKENS = 1024          # ← erreur 1
CHARS_PER_TOKEN  = 4

def cache_block_for(system_prompt):
    est_tokens = len(system_prompt) // CHARS_PER_TOKEN   # ← erreur 2
    if est_tokens < CACHE_MIN_TOKENS: return None
```

1. **Le seuil est faux pour ce modèle.** Le préfixe minimal réellement mis en cache dépend du
   modèle : **Haiku 4.5 exige 4 096 tokens** — le plus élevé de toute la gamme (Opus 5 : 512 ;
   Sonnet 5 / Sonnet 4.6 : 1 024). En dessous, le marqueur est **silencieusement ignoré** :
   pas d'erreur, `cache_creation_input_tokens: 0`. Le code contrôle 1 024.
2. **Le garde-fou mesure le mauvais périmètre.** L'ordre de rendu de la requête est
   `tools → system → messages` : un point de rupture sur le bloc système met en cache
   **les définitions d'outils ET le système**. `cache_block_for` ne regarde que
   `SYSTEM_PROMPT_STATIC` seul (≈ 2 397 tokens estimés) et ignore les 7 460 caractères d'outils
   qui font pourtant partie du même préfixe.

**Où cela nous laisse-t-il en pratique ?** Le vrai préfixe fait 17 048 caractères, soit
**4 262 tokens dans l'hypothèse la plus défavorable** contre un seuil de 4 096 : **le cache
fonctionne, avec une marge de ~4 %.** En français réel (~3,3 c/t) la marge est confortable
(~5 200 tokens). Mais :

- **rien ne le mesure** — aucun des deux appelants ne lit `usage.cache_read_input_tokens` ;
- **la marge n'est protégée par rien.** Une mission d'allègement de texte qui raboterait
  `SYSTEM_PROMPT_STATIC`, ou le retrait d'un outil, ferait passer le préfixe sous 4 096 tokens et
  **désactiverait le cache sans le moindre signal** : pas d'erreur, juste une facture qui monte et
  une latence qui grimpe. Le projet a déjà eu trois missions de réduction de texte (TXT-1, UI-3,
  MIA-5) — le risque n'est pas théorique.

**Traducteur du scanner** : **aucun `cache_control`, nulle part.** `system=SYSTEM_PROMPT` est une
chaîne simple et l'outil est repassé à chaque appel. Chaque traduction repaie l'intégralité du
préfixe au prix plein.

**Mais — et c'est le point qui change la recommandation — ajouter `cache_control` ici ne servirait
à rien.** Le préfixe fait 1 635 à 1 982 tokens, **sous le minimum de 4 096 de Haiku 4.5**. Le
marqueur serait accepté par l'API et silencieusement inopérant. Mettre le traducteur en cache exige
**de changer de modèle**, pas d'ajouter une ligne. Voir B-4.

**Agent démo de l'accueil** (`demo_agent.py`, 9 langues) : réutilise la classe `Chatbot`. Son premier
point de rupture porte sur outils (3 362 car.) + statique (9 588) = 12 950 car. ≈ **3 238–3 924 tokens**,
donc **très probablement inerte** sous les 4 096 de Haiku 4.5. Le **second** point de rupture, posé sur
le dernier bloc supplémentaire, couvre un préfixe de 26 927 car. ≈ 6 700–8 200 tokens et **fonctionne**.
Le premier marqueur est donc gaspillé, sans dommage.

### B.2 — Redondance de contexte : ce qui est propre, et ce qui coûte

**Ce qui est déjà bien fait, à ne pas défaire** :
- historique **tronqué côté serveur** à 12 messages, coupé sur un tour `user` pour rester une
  transcription valide (`_truncate_history`) ;
- **déduplication d'outil intra-tour** : un outil rappelé avec les mêmes arguments dans le même tour
  réutilise le résultat enregistré (`tool_cache`, `chatbot.py:567`) ;
- `signal_summary` **mis en cache 60 s** côté serveur, donc l'assembleur n'est pas rejoué par message ;
- JSON compacté (`ensure_ascii=False`, sans indentation) ;
- l'historique transporté est **du texte seul** (`ConversationMessage.content: str`) : les résultats
  d'outils ne survivent pas au tour. Excellent — c'est le poste de coût que ce choix supprime.

**Le gaspillage réel, mesuré** : `get_market_reading` renvoie **la charge utile entière**, sans
condensation :

```python
# chatbot.py:899
reading = self._assembler.get_or_generate(instrument, timeframe)
return reading.model_dump(mode="json")
```

Tailles réelles, mesurées sur `data/market_readings.db` :

| Combo | Lignes | Moyenne (car.) | Max (car.) |
|---|---|---|---|
| XAUUSD M15 | 196 | 11 329 | **35 646** |
| EURUSD M15 | 187 | 10 313 | **38 913** |
| XAUUSD H1 | 68 | 11 338 | 23 744 |
| XAUUSD H4 | 29 | 11 471 | 20 442 |
| XAUUSD D1 | 1 | 23 637 | 23 637 |
| XAUUSD M5 | 1 | 30 640 | 30 640 |

→ **un seul appel d'outil injecte en moyenne ~11 000 caractères (≈ 2 750–3 400 tokens), et jusqu'à
38 913 caractères (≈ 9 700–11 800 tokens)** dans la conversation. À comparer au préfixe statique
complet (~4 300 tokens) : **un appel d'outil peut coûter plus cher que tout le prompt système** —
et il est facturé **plein tarif**, puisqu'il se place après le point de rupture et change à chaque tour.
Avec `MAX_TOOL_TURNS=3`, il est renvoyé à chaque ronde suivante du même tour.

Le modèle ne consomme qu'une fraction de cette charge (tendance, phase, quelques niveaux, le journal
structurel récent). Le reste — historiques de zones, identifiants, métadonnées destinées au graphique —
est du remplissage payé au prix fort.

Second poste, mineur et **délibéré** : le bloc `signal_summary` (~718–870 tokens) est réémis à chaque
tour hors cache. C'est un arbitrage assumé et documenté (« most questions are answered without any
tool call ») : il échange des tokens contre une latence évitée. À conserver tel quel.

### B.3 — Choix de modèle

Les deux tâches tournent sur **Haiku 4.5**. Prix réels au 2026-06 : **1,00 $ / 5,00 $ par MTok**
(entrée/sortie), lecture de cache à 0,10 $/MTok.

**Aucune des deux tâches n'appelle un modèle trop cher.** Haiku est déjà le palier le moins cher.
La question intéressante est inversée : **pour le traducteur, un modèle plus cher serait moins cher**,
à cause du seuil de cache.

Ordre de grandeur, par appel de traduction (préfixe ≈ 1 800 tokens) :

| Option | Coût entrée par appel |
|---|---|
| Haiku 4.5, sans cache (aujourd'hui) | 1 800 × 1,00 $/MTok = **0,0018 $** |
| Sonnet 5, cache **froid** (écriture, ×1,25) | 1 800 × 2,00 × 1,25 = **0,0045 $** |
| Sonnet 5, cache **chaud** (lecture, ×0,1) | 1 800 × 0,20 $/MTok = **0,00036 $** |

→ Sonnet 5 avec cache chaud est **~5 × moins cher** que Haiku sans cache. **Mais** le TTL est de
5 minutes : le gain n'existe qu'en trafic soutenu. Un appel isolé toutes les 10 minutes paie
l'écriture et **coûte 2,5 × plus cher** qu'aujourd'hui.

**Recommandation honnête, compte tenu de la phase du produit (test personnel, trafic faible) :
ne rien changer maintenant.** Le traducteur reste sur Haiku 4.5 sans cache — c'est le bon choix à
faible volume. Ce levier se rouvre le jour où le scanner est utilisé de façon continue, et il se
décide alors **sur des mesures d'`usage`**, pas sur ce tableau.

Pour le chatbot, Haiku 4.5 reste adapté : la tâche est de la reformulation encadrée par outils et par
quatre couches déterministes, pas du raisonnement ouvert. Le rehausser dégraderait la latence sans
gain de qualité perçue.

**Défaut annexe à signaler : la table de prix du dépôt est fausse et morte.**
`src/intelligence/llm_cost_policy.py:35` :

| Modèle | Table du dépôt | Prix réel |
|---|---|---|
| `claude-haiku-4-5` | 0,50 $ / 2,50 $ | **1,00 $ / 5,00 $** (sous-estimé × 2) |
| `claude-opus-4-7` | 15 $ / 75 $ | **5,00 $ / 25,00 $** (surestimé × 3) |
| `claude-sonnet-5`, `claude-opus-5` | absents | 2 $/10 $ et 5 $/25 $ |

Et `pick_model`, `should_batch`, `MODEL_PRICING` **ne sont appelés nulle part** : seul
`cache_block_for` est importé (`chatbot.py:37`). Une seconde table, distincte, vit dans
`src/intelligence/rag/cost_tracker.py:33`. Toute décision de coût prise en lisant ce fichier serait
fausse d'un facteur 2 à 3.

### B.4 — Les couches de sécurité : **aucune redondance coûteuse en tokens**

Vérification directe : `adversarial_filter.py`, `output_filter.py` et `view_action_filter.py` ne
contiennent **ni `messages.create`, ni import `anthropic`**. Ce sont des filtres Python déterministes
(motifs, listes fermées, validation d'identifiants).

**Elles coûtent exactement zéro token.** Il n'y a donc **aucune redondance à signaler entre les
couches**, et rien à arbitrer. La seule duplication existante est de nature différente et parfaitement
assumée : les règles sont **énoncées** dans `SYSTEM_PROMPT_STATIC` *et* **appliquées** par les filtres.
C'est de la défense en profondeur — le modèle est instruit, puis le serveur vérifie sans lui faire
confiance. Le coût de cette duplication est celui du texte des règles dans le prompt statique, lequel
est **dans le préfixe mis en cache** : payé une fois, relu à 10 % ensuite.

**Aucune recommandation n'est formulée sur ces couches. Elles sont efficaces telles quelles.**
Il faut même noter l'inverse : la marge de cache de ~4 % identifiée en B.1 signifie qu'**alléger le
texte des règles pour économiser des tokens serait contre-productif** — cela ferait passer le préfixe
sous le seuil et coûterait plus cher qu'il n'économiserait.

## B.5 — Recommandations partie B (classées par gain, aucune appliquée ici)

| # | Recommandation | Gain | Nature |
|---|---|---|---|
| **B-1** | **Condenser le résultat de `get_market_reading`** au sous-ensemble que le modèle exploite réellement (en-tête, régime, journal structurel récent, zones actives) au lieu du `model_dump` complet. | Le plus gros poste mesuré : ~11 000 car. en moyenne, jusqu'à 38 913, plein tarif, à chaque appel d'outil | **mécanique** |
| **B-2** | **Instrumenter `usage`** (`cache_read_input_tokens`, `cache_creation_input_tokens`, `input_tokens`, `output_tokens`) sur le chatbot **et** le traducteur, en log structuré JSON. | Rend B-1/B-3/B-4 décidables sur mesure au lieu d'estimation. **Préalable à tout le reste.** | **mécanique** |
| **B-3** | **Corriger `cache_block_for`** : seuil **par modèle** (Haiku 4.5 = 4 096), et compter **outils + système** dans le préfixe, pas le système seul. Ajouter un test qui échoue si le préfixe passe sous le seuil. | Aucun gain immédiat ; **protège** un cache qui tient à 4 % près et qu'aucune garde ne défend | **mécanique** |
| **B-4** | **Traducteur** : acter que le cache est impossible sous Haiku 4.5 (préfixe ~1 800 < 4 096 tokens). Décision à rouvrir *au volume* : rester Haiku sans cache, ou passer à Sonnet 5 + cache (~5 × moins cher à chaud, 2,5 × plus cher à froid). | Nul aujourd'hui ; significatif en usage soutenu | **architecture** (décision fondateur, sur les mesures de B-2) |
| **B-5** | **Point de rupture incrémental sur l'historique** : marquer le dernier message du tour précédent pour que l'historique déjà envoyé soit relu à 10 % au lieu d'être repayé. Jusqu'à 12 × 2 000 car. aujourd'hui plein tarif. | Croît avec la longueur de conversation | **mécanique** |
| **B-6** | **Corriger ou supprimer `llm_cost_policy.MODEL_PRICING` / `pick_model` / `should_batch`** : prix faux d'un facteur 2 à 3, code mort, table dupliquée dans `rag/cost_tracker.py`. | Évite une future décision de coût prise sur des chiffres faux | **mécanique** |
| — | **Couches de sécurité** | **Aucune action.** Zéro token, aucune redondance. | — |

---

## Annexe — Script de vérification des comptes de tokens exacts

À exécuter **dans l'environnement du fondateur** (celui qui possède `ANTHROPIC_API_KEY`).
Il transforme les bornes de ce rapport en chiffres exacts et tranche définitivement la marge de 4 %
identifiée en B.1.

```python
# scripts/audit/perf_3/count_prompt_tokens.py  (à créer si la décision est prise)
import anthropic
from src.intelligence.chatbot import chatbot as cb
from src.intelligence import scanner_translator as st

client = anthropic.Anthropic()
MODEL = cb.DEFAULT_MODEL   # claude-haiku-4-5-20251001

# 1) Préfixe réellement mis en cache par le chatbot = outils + système statique.
#    (récupérer _tool_schemas depuis l'instance chatbot déjà construite par le bootstrap)
r = client.messages.count_tokens(
    model=MODEL,
    system=[{"type": "text", "text": cb.SYSTEM_PROMPT_STATIC}],
    tools=chatbot_instance._tool_schemas,
    messages=[{"role": "user", "content": "."}],
)
print("chatbot: prefixe outils+systeme =", r.input_tokens, "tokens  (seuil Haiku 4.5 = 4096)")

# 2) Préfixe du traducteur
r = client.messages.count_tokens(
    model=MODEL, system=st.SYSTEM_PROMPT, tools=[st.build_tool_schema()],
    messages=[{"role": "user", "content": "."}],
)
print("traducteur: prefixe =", r.input_tokens, "tokens  (< 4096 => cache impossible)")
```

Puis, en production, une seule ligne de log par appel suffit à clore le sujet :
`usage.cache_read_input_tokens` **non nul et stable** ⇒ le cache tient ; **à zéro de façon répétée**
⇒ le préfixe est passé sous le seuil.

---

## Synthèse pour décision

**Partie A.** L'architecture est meilleure que ce que la mission redoutait : **le multi-utilisateur ne
duplique aucun appel** (PERF-2 + `candles.db` partagés), le week-end coûte zéro, le TTL et le
verrou market-aware sont en place. Le jitter n'est **pas** implémenté, mais à 2 marchés il ne
rapporterait rien — et il n'aurait de toute façon jamais réduit le volume, seulement le pic. Les vrais
leviers sont, dans l'ordre : **fermer la fuite de quota M5** (dépassement dur du plafond gratuit dès
qu'un utilisateur ouvre M5), **dériver H1/H4/D1 depuis le M15 déjà stocké** (−24 %), puis **agréger
les bougies depuis le WebSocket déjà en place** (→ ~0), ce dernier point exigeant le plan Business et
une passe de réconciliation.

**Partie B.** Le caching **existe** côté chatbot et il est correctement placé — mais il tient à ~4 %
d'un seuil (4 096 tokens sur Haiku 4.5) que le code croit être à 1 024, en mesurant qui plus est le
mauvais périmètre, **et rien ne le mesure en production**. Côté traducteur il est absent, et l'ajouter
ne servirait à rien sans changer de modèle. Le gaspillage le plus lourd et le plus facile à corriger
n'est pas le prompt : c'est **le résultat d'outil `get_market_reading`, ~11 000 caractères en moyenne
et jusqu'à 38 913, renvoyé intégralement au prix plein**. Les quatre couches de sécurité ne coûtent
rien et ne dupliquent rien.

**Aucune de ces recommandations n'est appliquée dans cette mission.** Chacune devient, ou non, une
mission séparée — un changement par mission.

---

## Suites données à cet audit (mise à jour 2026-09-10)

Les correctifs ci-dessous ont été livrés sur `fix/perf-3-quota-m5-cache-haiku`.
Le reste de ce document est le diagnostic d'origine, laissé tel quel.

| Reco | État | Commit | Gain constaté |
|---|---|---|---|
| **A-1** fuite de quota M5 | ✅ livré | `ef4c9d8` | 830 → 446 req/j : retour sous le plafond gratuit |
| **A-4** fenêtres alignées | ✅ livré | `1446591` | 1 requête économisée par sonde/warm sur M15 et H1 |
| **A-5** fournisseur unique | ✅ livré | `1446591` | un seul limiteur, compteur global rétabli |
| **A-8** verrou du limiteur | ✅ livré | `1446591` | plus de gel du processus sur le plafond journalier |
| **A-9** compteur de crédits | ✅ livré | `1446591` | `data_provider_credits` sur `/health` |
| **B-1** condensation du résultat d'outil | ✅ livré | `2fbd2b2` | **−34,2 %** (jusqu'à −51,7 %) sur les 10 lectures stockées |
| **B-2** instrumentation `usage` | ✅ livré | `1446591` | `cache_read_input_tokens` désormais journalisé |
| **B-3** seuil de cache par modèle | ✅ livré | `97cae0c` | marge de 4,1 % désormais tenue par un test |
| **B-6** table de prix | ✅ livré | `1446591` | erreurs ×2 et ×3 corrigées |
| — blocage `test_bootstrap_runtime` | ✅ livré | `8254081` | suite débloquée (16/16 en 6,5 s) |

### Recommandations NON appliquées, et pourquoi

**A-2 (dériver H1/H4/D1 depuis M15) et A-3 (agrégateur WebSocket).** Décision
fondateur assumée. A-3 exige en plus le plan Business, et les deux exigent la
passe de réconciliation décrite au §A.5 : sans elle on remplace une facture par
une dette de justesse, inacceptable sur un produit qui affiche du BOS/CHOCH.
La frontière D1 (minuit UTC vs convention de place) reste à trancher **sur
données réelles**.

**A-6 (ne pas reprendre un créneau de limitation à chaque retry).** *Je retire
cette recommandation.* À l'examen du code, le comportement actuel est le bon
côté de l'erreur : sur-compter localement nous fait nous auto-freiner, ce qui
est sûr ; sous-compter nous ferait dépasser le plan et récolter des 429. La
« correction » aurait rendu le client plus agressif pendant un incident.

**A-7 (jitter).** L'audit lui-même conclut à un gain nul à 2 marchés, et il
n'aurait de toute façon jamais réduit le volume — seulement le pic. L'implanter
aujourd'hui échangerait de la fraîcheur réelle contre un bénéfice théorique.
À rouvrir au-delà de ~8 marchés.

**A-10 (tendances des unités supérieures dans la charge utile).** Gain serveur
uniquement (0 crédit fournisseur), mais la moitié du changement est du frontend
et sa vérification demande la chaîne Playwright/vitest. Tout le reste de ce lot
est couvert par des tests exécutés ; je n'ai pas voulu livrer non vérifié la
seule partie qui ne l'aurait pas été.

**B-4 (modèle du traducteur).** Ma propre analyse §B.3 conclut « ne rien changer
maintenant » : à faible trafic Haiku sans cache est le choix le moins cher. À
rouvrir sur les mesures que B-2 produit désormais.

**B-5 (point de rupture incrémental sur l'historique) — neutralisé par
l'architecture actuelle.** Vérification faite : le cache est un préfixe, et
« toute différence d'un octet en position N invalide le cache pour tous les
points de rupture à partir de N ». L'ordre de rendu est `tools → system →
messages` : le bloc `signal_summary`, qui vit dans `system` et se rafraîchit
toutes les 60 s, se trouve **avant** les messages et invalide donc leur cache à
chaque rafraîchissement. Poser un point de rupture sur l'historique ne
rapporterait quasi rien tant que ce bloc reste où il est.

Le vrai préalable est de sortir le bloc volatil de `system` pour l'injecter dans
`messages` — c'est d'ailleurs la règle générale (« keep the system prompt
frozen ; inject dynamic context later in messages »). Mais Haiku 4.5 n'accepte
pas les messages `role: "system"` en cours de conversation, il faudrait donc le
porter dans le tour utilisateur, ce qui change la forme de la conversation et
peut peser sur la qualité des réponses. C'est une décision d'architecture, à
prendre **sur mesures** — que l'instrumentation B-2 rend enfin possibles.
