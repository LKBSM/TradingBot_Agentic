# AUDIT DATA-3 — 55 crédits/minute suffisent-ils pour 100 marchés ?

> **Diagnostic + simulation (§1-8) — puis correctifs (§9).**
> Les §1 à 8 ont été produits SANS toucher au code de production ; §9 documente
> les correctifs qui en découlent, mesurés avec le même harnais.
> Date : 2026-09-13 · Fournisseur : Twelve Data (REST `time_series`) · Plan envisagé : **Grow, 55 crédits/min**
> Base : `origin/main` @ `9de5138`, worktree dédié `wt-data-3`, branche `docs/data-3-budget-55`.
> *(Le HEAD local du dépôt principal était 122 commits en retard — tout ce rapport est produit contre `origin/main` à jour, conformément à la leçon process de DATA-1.)*

---

## RÉPONSE EN PREMIÈRE LIGNE

**55 crédits/min suffisent largement pour 100 marchés × 5 unités — mais PAS avec le code actuel.**

| | Pointe mesurée/simulée à 100 marchés | vs 55 |
|---|---|---|
| **S1 — code actuel tel quel** | **500 crédits/min** | ❌ **9,1× au-dessus**, 96 minutes/jour en dépassement |
| **S2 — étalement seul** | **31 crédits/min** | ✅ passe, 56 % du plafond |
| **S3 — étalement + dérivation** | **21 crédits/min** | ✅ passe, 38 % du plafond |
| **S4 — S3 + amorçage 1 an d'historique** | **51 crédits/min** (amorçage à débit fixe 30/min) | ✅ passe, marge faible |

Le levier décisif n'est **pas** la dérivation M15/H1/H4 depuis M5 (§ b) : c'est **l'étalement**.
Avec l'étalement seul, 55 crédits/min tiennent jusqu'à **180 marchés**. La dérivation
pousse à 270 — elle achète de la marge, elle ne débloque rien à 100 marchés.

**Mais trois défauts mesurés du code rendent l'ajout de marchés impossible aujourd'hui, quel que soit le plan payé** :
1. le limiteur est **figé à 8 crédits/min / 800 par jour**, non pilotable par variable d'environnement (mesuré, § a.6) — payer Grow ne changerait rien ;
2. le fournisseur ne sait traduire que **2 symboles** (`_SYMBOL_MAP` en dur) — tout autre marché lève `ValueError` ;
3. le tick du scheduler est **séquentiel et mono-instance**, ~0,8 à 1,6 s par combinaison : **38 à 71 combinaisons tiennent dans un tick de 60 s**, contre 500 à câbler (100 marchés × 5 unités).

La liste complète des correctifs à faire **avant tout ajout de marché** est en § 7.

---

## 1. Méthode — ce qui est mesuré, ce qui est simulé

| Question | Comment elle a été tranchée |
|---|---|
| Cadence réelle du code en production | **Données réelles** : `data/market_readings.db`, journée complète du 2026-07-04 (192 lectures générées, horodatées à la seconde) |
| Nombre d'appels par déclencheur | **Harnais instrumenté** : vraies classes de production (`TwelveDataProvider`, `MarketReadingAssembler`, `MarketReadingScheduler`, vrais magasins SQLite), seule la couche HTTP remplacée par une `requests.Session` qui compte et horodate — `tools/data_budget/measure_current.py` |
| Comportement du fournisseur au dépassement | **Appel réel unique** sur la clé gratuite — `tools/data_budget/probe_429.py` |
| Projection à 84 / 100 / 150 marchés | **Simulateur** rejouant 1 440 minutes de clôtures — `tools/data_budget/simulate_credits.py` |

Rien dans ce rapport ne repose sur une lecture « à l'œil » du code : chaque affirmation
chiffrée renvoie soit à `docs/audits/data-3/mesures-avant.json`, soit à la base de production,
soit à `docs/audits/data-3/test-429-reel.json`.

> **Note d'honnêteté** : aucun backend ne tournait pendant la mission (dernier écriture
> des bases : 2026-07-31 ; aucun processus `python -m src.intelligence.main` / `uvicorn src.api.asgi`
> actif). Il n'a donc pas été possible d'observer passivement 24 h de trafic *en direct* :
> les chiffres de (a) viennent de la **dernière journée réellement enregistrée** plus du
> **rejeu instrumenté du code d'aujourd'hui**. Les deux concordent exactement.

---

## 2. (a) Comportement ACTUEL, mesuré

### a.1 — Mesure sur données de production réelles (2 marchés)

`data/market_readings.db`, journée du **2026-07-04** (la dernière journée complète enregistrée) :

```
192 lectures générées entre 00:00:40 et 18:00:48
par unité :  M15 146 · H1 36 · H4 10 · D1 0 (régénéré la veille)
minutes distinctes ayant déclenché au moins un appel : 74  (sur 1 080)
PIC mesuré sur une minute : 6
distribution : 56 minutes à 2 appels · 14 minutes à 4 · 4 minutes à 6
minutes de pointe : 04:00, 08:00, 12:00, 16:00 (bords H4)
```

La forme est exactement celle du modèle « une requête par (marché, unité) à chaque clôture,
toutes émises dans la même minute » :

| Instant | Unités qui clôturent | Appels mesurés | = unités × 2 marchés |
|---|---|---|---|
| :15 / :30 / :45 | M15 | **2** | 1 × 2 |
| heure pleine | M15 + H1 | **4** | 2 × 2 |
| bord 4 h | M15 + H1 + H4 | **6** | 3 × 2 |
| minuit UTC | M15 + H1 + H4 + D1 | 8 (non observé ce jour-là) | 4 × 2 |

**Sur 24 h : 192 (M15) + 48 (H1) + 12 (H4) + 2 (D1) = 254 crédits/jour, pointe 8/min.**
94,9 % des minutes de la journée consomment **zéro** crédit : toute la consommation est
concentrée sur les bords d'horloge. C'est précisément ce que l'étalement corrige.

### a.2 — Chaque déclencheur d'appel sortant (vérifié dans `origin/main`, pas supposé)

| # | Déclencheur | Chemin | Appels | Mesure |
|---|---|---|---|---|
| 1 | **Tick planifié** (cœur) | `scheduler.tick()` → `assembler.get_or_generate(bound_provider=False)` | 1 par combinaison échue, **toutes dans le même tick** | 8 combos → **8 appels en un tick**, 2ᵉ tick → **0** |
| 2 | **Ouverture de page / changement de marché ou d'unité** | `routes/market_reading.py` → `get_or_generate(bound_provider=True)` | **0 en régime permanent** (PERF-2 sert la lecture stockée) ; 1 au démarrage à froid | **0** en régime permanent (mesuré) |
| 3 | **Graphique** | `routes/candles.py` → `assembler.warm_candles` | seulement D1/W1 absents du cache, 1 par bougie close | 1/marché/semaine pour W1 |
| 4 | **M.I.A (chat)** | `chatbot.py:1062` → `get_or_generate` (défaut `bound_provider=True`) | même chemin que #2 ⇒ **0 en régime permanent** | idem #2 |
| 5 | **Scanner de conditions** | `routes/conditions_scan.py` | **0** — `readings_store.get_latest_reading`, pur SELECT | confirmé |
| 6 | **Sonde de réouverture jour férié** | `assembler.refresh_if_reopened` | **1 à 2 par marché par 30 min, toutes dans le même tick** | mesuré (§ a.5) |
| 7 | **M5 à la demande** | tick, hors périmètre warm | plancher PERF-3 : 1 par 900 s par combo ouverte | code + §3 |
| 8 | **Amorçage d'historique** | `scripts/backfill_history.py` | 1 par page de 5 000 bougies | mesuré : 5 000 bougies = **1 crédit** |
| 9 | **Flux temps réel** | `twelve_data_ws.py` | WebSocket, facturé à part, `LIVE_TICK_ENABLED=false` | hors budget REST |

> ⚠️ **Consommateur non listé par DATA-1** : `src/intelligence/main.py` (l'entrée que le
> `Dockerfile` lance encore, `CMD ["python","-m","src.intelligence.main"]`) démarre en plus
> un `SentinelScanner`/`MultiSymbolScanner` qui appelle `get_ohlcv` **toutes les 60 s par
> symbole**. Le cache TTL de 300 s ramène cela à ~1 appel / 5 min / symbole, soit
> **288 crédits/jour/marché en plus** — 28 800/jour à 100 marchés. Le produit V2 réel
> (`src/api/asgi.py`) ne démarre PAS ce scanner. **Déployer l'image Docker telle quelle
> rallumerait ce consommateur.** À trancher en DATA-2.

### a.3 — Un utilisateur déclenche-t-il des appels ? *(3 onglets = 3 appels ou 1 ?)*

Mesuré sur le vrai assembleur, 3 requêtes **simultanées** sur la même combinaison :

| Situation | Appels fournisseur |
|---|---|
| Régime permanent (une lecture est stockée) | **0** |
| Démarrage à froid (rien en base pour cette combinaison) | **3** — un par onglet |

**Réponse chiffrée : 3 onglets = 0 appel en régime permanent, 3 appels au démarrage à froid.**
Le découplage coût/utilisateurs de PERF-2 fonctionne, **sauf** sur le premier accès à une
combinaison jamais générée : il n'y a **aucun verrou single-flight**, les N requêtes
concurrentes manquent le cache TTL ensemble et partent toutes. À 100 marchés × 5 unités,
un démarrage à froid, c'est 500 combinaisons à générer — et chacune peut être multipliée
par le nombre d'onglets ouverts à cet instant.

### a.4 — Étalement et dérivation : existent-ils ?

| | Mesuré | Verdict |
|---|---|---|
| **Étalement (jitter)** | 0 occurrence de `jitter` / `random.` / `stagger` dans `scheduler.py` | **N'EXISTE PAS.** Le tick parcourt les combinaisons en séquence et émet tout ce qui est échu dans la même minute. |
| **Dérivation M15/H1/H4 depuis M5** | `resample_ohlcv` existe (`volatility_forecaster.py:179`) mais `resample_ohlcv(` **n'apparaît nulle part** dans `market_reading_assembler.py` ni `scheduler.py` — seulement dans `scripts/audit/*` et `scripts/generate_validation_dataset.py` | **DORMANTE, confirmée.** Chaque unité est demandée séparément au fournisseur. Un commentaire de `bootstrap.py:309` la désigne explicitement comme travail futur (« until the live M5-base resample path is wired »). |

### a.5 — Deux consommations que personne n'avait chiffrées

**Les retries coûtent des crédits.** Mesuré : une seule récupération logique qui échoue en 429
produit **4 requêtes HTTP** (`MAX_RETRIES = 4`, `_rate_limiter.acquire()` est appelé **à
chaque tentative**). Et le test réel (§5) montre qu'**une requête refusée consomme quand même
un crédit** (`Api-Credits-Used` passe de 8 à 9 sur le 429). Une fenêtre dégradée coûte donc
jusqu'à **×4** le budget nominal.

**La sonde jour férié tire en rafale.** Mesuré sur le 2026-12-25 (dans `config/market_holidays.json`),
10 combinaisons, ticks espacés de 31 min :

```
tick 1 (12:00) : 10 appels — XAU/USD 5min,15min,1h,4h,1day + EUR/USD idem   (démarrage à froid)
tick 2 (12:31) :  4 appels — XAU/USD 5min,4h + EUR/USD 5min,4h              (sonde)
tick 3 (13:02) :  2 appels — XAU/USD 4h + EUR/USD 4h                        (sonde)
```

Toutes les combinaisons sont horodatées dans le **même** tick, donc elles redeviennent
échues dans le **même** tick 30 min plus tard : **1 à 2 appels par marché, groupés sur une
seule minute, toutes les 30 minutes**. À 100 marchés cela fait **100 à 200 crédits dans une
minute** un jour férié — au-dessus de 55 à lui seul, alors même que le marché est fermé.

**Un flux en retard ne part pas en boucle.** Mesuré : 10 ticks consécutifs sur une
combinaison dont la bougie n'atteint jamais la clôture attendue → **2 appels au total**,
le cache TTL de 300 s absorbant les 9 ticks suivants. Le plancher est donc de
**1 appel / 300 s / combinaison en retard** — soit, à 500 combinaisons toutes en retard,
**100 crédits/min en régime permanent** (frein : le TTL, pas l'ordonnanceur).

### a.6 — Les trois verrous durs, mesurés

| Fait mesuré | Valeur | Conséquence |
|---|---|---|
| `TwelveDataProvider.__init__` défauts | `per_minute=8`, `per_day=800` | plan gratuit |
| Ces plafonds sont-ils pilotables par variable d'environnement ? | **NON** (test fonctionnel : 4 noms de variables plausibles positionnés à 55 → limiteur toujours à **8/min, 800/jour**) | **payer Grow ne changerait rien** : le code s'auto-limiterait à 8/min |
| `bootstrap.py` / `main.py` passent-ils des plafonds explicites ? | **NON** | idem |
| Symboles que le fournisseur sait traduire | `['XAUUSD', 'EURUSD']` (`_SYMBOL_MAP` en dur) | tout autre marché ⇒ `ValueError: Unsupported symbol` |
| Registre de marchés (MKT-1) | `('XAUUSD', 'EURUSD')` | 2 marchés, pas 84 |
| Durée d'un tick | **0,8 à 1,6 s par combinaison** (deux exécutions : 0,836 s et 1,551 s) | **38 à 71 combinaisons tiennent dans un tick de 60 s** ; `max_instances=1` + `coalesce=True` ⇒ au-delà, les ticks sont fusionnés et la donnée décroche |
| Cache TTL du fournisseur | **300 s**, clé `(symbole, unité, lookback)` | déduplique deux lectures identiques ; **aucune** déduplication entre unités ou entre tailles de fenêtre |
| Compteur de crédits | `credit_snapshot()` exposé par `/health` (PERF-3) | l'observabilité agrégée existe ; **pas** d'attribution par déclencheur, et les en-têtes serveur ne sont pas lus |

---

## 3. (b) Projection à 100 marchés — quatre scénarios

Hypothèses communes : 5 unités (M5, M15, H1, H4, D1), **1 requête = 1 symbole = 1 crédit**
(vérifié réellement, § 5), marchés cotant 24 h (pire cas ; la variante « mix » est en § 4).
M1 reste fermé par son garde-fou (`LB1_ENABLE_M1`) — l'inclure ajouterait un plancher
incompressible de 100 crédits/min et rendrait la question sans objet.

| Sc. | Stratégie | Total/jour | Moy./min | **Pointe/min** | vs **55** | vs **144** | vs **377** |
|---|---|---|---|---|---|---|---|
| **S1** | Code actuel tel quel | 22 300 | 15,5 | **500** | ❌ +445, 96 min/j | ❌ +356, 96 min/j | ❌ +123, 6 min/j |
| **S2** | Étalement seul | 41 500 | 28,8 | **31** | ✅ | ✅ | ✅ |
| **S3** | Étalement + dérivation | 28 900 | 20,1 | **21** | ✅ | ✅ | ✅ |
| **S4** | S3 + amorçage 1 an de M5 | 31 120 | 21,6 | **51** | ✅ (marge 7 %) | ✅ | ✅ |

**Comment se forme la pointe de S1** : à minuit UTC, M5 + M15 + H1 + H4 + D1 clôturent
ensemble ; sans étalement les 5 unités × 100 marchés partent dans la même minute = **500**.
Aux 96 bords M15 de la journée : 200/min (M15 + le plancher M5 à 900 s, tous deux alignés
sur les multiples de 15 min) — déjà 3,6× le plafond. *Variante* : si aucun utilisateur
n'ouvre jamais M5 (M5 n'est pas dans le périmètre warm aujourd'hui), la pointe S1 tombe à
**400/min** — elle ne passe toujours pas.

**Pourquoi S2 consomme plus par jour mais passe** : l'étalement ne réduit pas le nombre de
requêtes (41 500/jour, puisque M5 est désormais réellement suivi), il en **change la forme**.
Twelve Data facture au **pic par minute**, illimité par jour sur les plans payants : seule la
pointe décide. Aplatir 500 crédits sur les 5 minutes qui séparent deux clôtures M5 donne
31/min soutenus.

**Ce que S3 ajoute** : dériver M15/H1/H4 d'une base M5 économise 126 requêtes/marché/jour
(12 600/jour à 100 marchés) et descend la pointe de 31 à 21. **Utile, pas nécessaire à
100 marchés.** C'est aussi le changement le plus risqué (fidélité O/H/L/C, bornes H4, trous
M5) — cf. DATA-1 §6 et son protocole de validation bit-pour-bit.

**S4 — amorçage** : mesuré réellement, une requête `outputsize=5000` sur M5 renvoie
**5 000 bougies pour 1 crédit**, couvrant **18 jours calendaires**. Un an de M5 = **~21-22
requêtes/marché**, soit **2 200 requêtes ponctuelles** pour 100 marchés. À débit fixe de
30/min en parallèle du direct, la pointe monte à 51/min (passe de justesse). En laissant
l'amorçage ne consommer **que la marge** laissée par le direct, il se termine en :

| Plafond | Durée de l'amorçage de 100 marchés (2 200 requêtes) sur la seule marge de S3 |
|---|---|
| 55 | **1,1 h** |
| 144 | 0,3 h |
| 377 | 0,1 h |

### Nombre maximal de marchés tenable, par stratégie et par plafond

*(pire cas 24 h continu, obtenu par recherche dichotomique sur le simulateur)*

| Plafond | S1 | S2 | S3 | S4 (amorçage 30/min en parallèle) |
|---|---|---|---|---|
| **40** *(cible de sécurité recommandée, § 6)* | 8 | 130 | 195 | 45 |
| **55** *(Grow)* | **11** | **180** | **270** | 120 |
| **144** | 28 | 490 | 715 | 565 |
| **377** | 75 | 1 305 | 1 875 | 1 725 |

> Lecture : **le code actuel plafonne à 11 marchés sur un plan à 55 crédits/min.** Ce n'est
> pas un problème de plan, c'est un problème d'ordonnancement.

---

## 4. (c) Simulateur — sensibilité

`tools/data_budget/simulate_credits.py`, sortie complète dans
`docs/audits/data-3/simulation.txt` et `simulation.json`.

**Sessions de marché.** Avec un panier réaliste (80 % cotant 24 h, 20 % d'indices à ~6 h 30
de séance), 100 marchés donnent : S1 400/min ❌ · S2 30/min ✅ · S3 20/min ✅ · S4 47/min ✅.
Les sessions réduisent le total journalier (−15 %) mais **pas la pointe** : elle se produit
quand tous les marchés sont ouverts.

**Retries.** En appliquant un multiplicateur de 1,25 (une requête sur quatre retentée une
fois), 100 marchés donnent : S2 39/min ✅ · S3 26/min ✅ · **S4 64/min ❌** (74 minutes/jour
au-dessus de 55). C'est le premier scénario où la marge se referme : **l'amorçage à débit
fixe pendant une fenêtre dégradée sort du plafond.** L'amorçage doit se caler sur la marge
disponible, pas sur un débit constant.

**Escalade.** À 150 marchés : S2 45/min ✅, S3 31/min ✅, S4 61/min ❌ contre 55.

---

## 5. (d) Test réel unique — ce que renvoie vraiment un dépassement

Exécuté **une seule fois** sur la clé gratuite (8 crédits/min), 8 requêtes, puis une sonde
de reprise. Rapport brut : `docs/audits/data-3/test-429-reel.json`. Coût : ~10 crédits sur 800.

```
#1..#7  HTTP 200   Api-Credits-Used: 2..8   Api-Credits-Left: 6..0
#8      HTTP 429   Api-Credits-Used: 9      Api-Credits-Left: 0
        {"status":"error","code":429,
         "message":"You have run out of API credits for the current minute.
                    9 API credits were used, with the current limit being 8."}
reprise : succès après 70 s d'attente
```

**Cinq faits à câbler dans le limiteur de DATA-2 :**

1. **C'est un vrai HTTP 429**, *et* l'enveloppe JSON porte `status:"error"`, `code:429`.
   Il faut traiter les deux : le code actuel gère bien le 429 HTTP, mais son traitement de
   l'enveloppe (`body.get("status") == "error"`) lève `TwelveDataError` **sans distinguer un
   429 d'une erreur de symbole** — un dépassement remonté par l'enveloppe seule serait
   confondu avec un bug de configuration.
2. **La requête refusée consomme quand même un crédit** (`Api-Credits-Used` : 8 → 9). Un 429
   n'est pas gratuit ; re-tenter aggrave le dépassement. Le code actuel re-tente **jusqu'à
   4 fois** → **4 crédits brûlés** pour une récupération qui échoue.
3. **Chaque réponse — succès comme échec — porte le compteur serveur** :
   `Api-Credits-Used`, `Api-Credits-Left`, `Api-Credits-Request`. C'est la source de vérité :
   elle rend inutile la fenêtre glissante côté client, qui ne peut pas être alignée sur celle
   du serveur (dès la 1ʳᵉ requête de la rafale le serveur annonçait déjà `used: 2`).
4. **Aucun en-tête `Retry-After` ni `RateLimit-Reset`.** Le délai de réinitialisation doit
   être déduit : reprise confirmée à **70 s**, cohérent avec une fenêtre glissante d'une minute.
5. **`Api-Credits-Request: 1`** sur une requête de 1 bougie **et** sur une requête de
   5 000 bougies (vérifié) : le coût est bien **1 crédit par requête, indépendant de la
   taille**. Télécharger 5 000 bougies plutôt que 1 est gratuit — cela ne change que la
   latence et la bande passante.

---

## 6. (e) Marge, risques, et conclusion chiffrée

### Consommation « invisible » qui partagera les 55

| Poste | Coût mesuré | À 100 marchés |
|---|---|---|
| Retries (jusqu'à 4 tentatives, chacune facturée) | ×1 à **×4** sur la fenêtre concernée | jusqu'à +60/min sur S2 en incident |
| Requête refusée (429) | **1 crédit**, non nul | s'ajoute au dépassement qu'elle signale |
| Sonde jour férié, groupée sans étalement | 1–2 par marché / 30 min, **dans une seule minute** | **100–200 crédits en une minute** |
| Démarrage à froid d'une combinaison, sans single-flight | 1 par onglet ouvert simultanément | 500 combinaisons × nombre d'onglets |
| Flux en retard (plancher TTL 300 s) | 1 / 300 s / combinaison | jusqu'à **100/min** si tout est en retard |
| Séries de référence W1 (`warm_candles`) | 1 / marché / semaine | ~14/jour |
| Tests manuels du fondateur pendant que le suivi tourne | 0 en régime permanent, 1 par combinaison jamais générée | ponctuel mais non nul |
| Scanner hérité si l'image Docker est déployée telle quelle | 1 / 300 s / marché | **+288/jour/marché** |

### Marge de sécurité recommandée

**Viser ≤ 40 crédits/min en régime permanent** (73 % de 55), en réservant les 15 crédits/min
restants aux rafales non étalées (sonde jour férié, démarrages à froid, retries). La cible
de 40 est atteinte confortablement à 100 marchés avec l'étalement seul (**31/min**, il reste
9/min de marge) et très confortablement avec la dérivation (**21/min**, 19/min de marge).

### Conclusion chiffrée

> **55 crédits/min suffisent pour 100 marchés × 5 unités SI, et seulement si :**
> 1. le limiteur devient **pilotable** et est effectivement réglé sur 55 (aujourd'hui : figé à 8) ;
> 2. les requêtes sont **étalées** sur la fenêtre entre deux clôtures (aujourd'hui : aucune) ;
> 3. la sonde jour férié et le démarrage à froid sont **étalés eux aussi** (sinon la rafale seule dépasse 55) ;
> 4. les retries sont **bornés et comptés** comme des crédits, et un 429 n'est **jamais** re-tenté immédiatement ;
> 5. l'amorçage d'historique consomme **la marge disponible**, pas un débit fixe ;
> 6. l'usage reste **interne** — afficher de la donnée Twelve Data à des abonnés payants impose la grille Business (DATA-1 §7), indépendamment du volume.

**Nombre maximal raisonnable de marchés** (avec étalement, cible ≤ 40/min soutenus) :

| Plafond | Sans dérivation (S2) | Avec dérivation (S3) |
|---|---|---|
| **55 (Grow)** | **130 marchés** *(180 à saturation)* | **195 marchés** *(270 à saturation)* |
| **144** | 360 *(490)* | 520 *(715)* |
| **377** | 950 *(1 305)* | 1 370 *(1 875)* |

La dérivation M5→M15/H1/H4 **n'est pas nécessaire pour atteindre 100 marchés**. Elle le
devient au-delà de ~130. Vu son risque (fidélité O/H/L/C, bornes H4, trous M5 — cf. DATA-1 §6),
elle devrait être traitée **après** l'étalement, pas avant.

---

## 7. Ce que DATA-2 doit corriger AVANT tout ajout de marché

S1 ne passe pas : les cinq premiers points sont **bloquants**, dans cet ordre.

| # | Correctif | Pourquoi (fait mesuré) | Portée |
|---|---|---|---|
| **B1** | **Rendre le plafond du limiteur configurable** (env `TWELVE_DATA_PER_MINUTE` / `_PER_DAY`, passé depuis `bootstrap.py` **et** `main.py`) | Mesuré : aucune variable d'environnement ne change les 8/min, 800/jour. Payer Grow ne débloquerait rien, et 500 combinaisons échues derrière un limiteur à 8/min mettent **62 minutes** à s'écouler | `twelve_data_provider.py`, `bootstrap.py`, `main.py` |
| **B2** | **Sortir `_SYMBOL_MAP` du code** et le dériver du registre de marchés (MKT-1), comme `_TIMEFRAME_MAP` l'est déjà du registre d'unités (TF-1) | Mesuré : le fournisseur ne connaît que 2 symboles ; tout autre lève `ValueError`. Même correctif à faire dans `twelve_data_ws.py` | `twelve_data_provider.py`, `twelve_data_ws.py`, `config/markets.json` |
| **B3** | **Étaler les émissions** : décalage déterministe par (marché, unité) sur la fenêtre entre deux clôtures | Mesuré : 0 étalement. C'est **le** levier — il fait passer la pointe de 500 à 31 à 100 marchés | `scheduler.py` |
| **B4** | **Paralléliser / borner le tick** (file de travail à N ouvriers, ou découpage du périmètre), et faire du plafond crédits le seul régulateur | Mesuré : 0,8–1,6 s/combinaison, **38 à 71 combinaisons par tick de 60 s**, `max_instances=1` + `coalesce=True` ⇒ à 500 combinaisons la donnée décroche silencieusement | `scheduler.py`, `bootstrap.py` |
| **B5** | **Limiteur fondé sur les en-têtes serveur** (`Api-Credits-Left`), pas sur une fenêtre glissante locale ; **ne jamais re-tenter un 429** (attendre la réinitialisation, ~70 s) ; distinguer le `code:429` de l'enveloppe des vraies erreurs | Mesuré : un 429 consomme un crédit ; `MAX_RETRIES=4` en brûle 4 ; le compteur local était déjà désaligné du serveur dès la 1ʳᵉ requête | `twelve_data_provider.py` |
| **B6** | **Verrou single-flight par (marché, unité)** sur le chemin interactif et sur `warm_candles` | Mesuré : 3 onglets simultanés au démarrage à froid = **3 appels** | `market_reading_assembler.py` |
| **B7** | **Étaler la sonde jour férié** (décalage par combinaison, pas un horodatage commun) | Mesuré : toutes les combinaisons redeviennent échues dans le même tick ⇒ 100–200 crédits en une minute à 100 marchés | `scheduler.py` |
| **B8** | **Amorçage d'historique piloté par la marge** (consommer `Api-Credits-Left`), pas à débit fixe | Simulé : à débit fixe 30/min avec retries, S4 sort du plafond 74 min/jour | `history_backfill.py`, `scripts/backfill_history.py` |
| **B9** | **Trancher l'entrée Docker** : `CMD` lance encore `src.intelligence.main`, qui démarre le `SentinelScanner` hérité (+288 crédits/jour/marché) que le produit V2 (`src/api/asgi.py`) n'utilise pas | Mesuré dans le code ; non listé par DATA-1 | `Dockerfile`, `infrastructure/Dockerfile` |
| **B10** | **Attribution de la consommation par déclencheur** dans `/health` (aujourd'hui : un total agrégé seulement) | Sans elle, une dérive est invisible jusqu'au 429 | `twelve_data_provider.py`, `routes/health.py` |

**Non bloquant, à traiter après** : la dérivation M5→M15/H1/H4 (gain de marge, pas de
déblocage à 100 marchés ; exige le protocole de validation bit-pour-bit de DATA-1 §6).

---

## 8. Outils livrés (réutilisables)

`tools/data_budget/` — trois scripts hors production, aucun secret dans le dépôt.

```
measure_current.py    Compte les appels fournisseur des VRAIES classes de production
                      (provider, assembleur, scheduler, magasins SQLite temporaires) en
                      remplaçant seulement la couche HTTP. Aucun accès réseau.
                        python tools/data_budget/measure_current.py

simulate_credits.py   Rejoue 24 h de clôtures pour N marchés × 5 unités selon S1..S4 ;
                      sort total/jour, moyenne, pointe, minutes en dépassement par plafond.
                        python tools/data_budget/simulate_credits.py --markets 84 100 150
                        python tools/data_budget/simulate_credits.py --markets 100 --sessions mix
                        python tools/data_budget/simulate_credits.py --retry-multiplier 1.25

probe_429.py          Dépasse le plafond UNE fois sur la clé réelle et enregistre code HTTP,
                      en-têtes de crédits, corps et délai de reprise. Exige --confirm.
                        python tools/data_budget/probe_429.py --confirm --out rapport.json
```

Sorties archivées : `docs/audits/data-3/mesures-avant.json` (état diagnostiqué),
`mesures-apres.json` (même harnais après les correctifs), `simulation.txt`,
`simulation.json`, `test-429-reel.json`.

---

## 9. Correctifs appliqués (branche `fix/data-2-conso-requetes`)

Les cinq points bloquants de §7 sont corrigés, plus quatre des cinq non bloquants.
Mesuré avec le MÊME harnais avant et après (`mesures-avant.json` / `mesures-apres.json`) :

| Mesure | Avant | Après |
|---|---|---|
| Plafond crédits/min pilotable par variable d'environnement | ❌ non | ✅ oui |
| Requêtes HTTP facturées pour une récupération refusée (429) | **4** | **1** |
| 3 onglets, démarrage à froid | **3** appels | **1** appel |
| 3 onglets, régime permanent | 0 | 0 *(inchangé)* |
| Sonde jour férié : combinaisons échues dans le même tick | toutes | étalées |
| Symboles résolubles par le fournisseur | 2, en dur | tout le registre MKT-1 |
| Pointe simulée à 100 marchés (périmètre warm) | **500/min** | **29/min** |
| Total simulé à 100 marchés | 22 300/jour | **12 700/jour** |

### Ce qui a changé, et pourquoi

| # | Correctif | Effet mesuré |
|---|---|---|
| **B1** | `TWELVE_DATA_PER_MINUTE` / `TWELVE_DATA_PER_DAY` pilotent le limiteur (défauts inchangés = palier gratuit) | payer Grow débloque enfin 55/min ; sans cela le code se serait auto-limité à 8 |
| **B2** | Le tableau des symboles dérive du registre de marchés (`providerSymbol` optionnel, sinon `XAUUSD → XAU/USD`) ; même correctif sur le pont WebSocket | ajouter un marché redevient **une entrée JSON**, plus un `ValueError` |
| **B3** | Étalement déterministe par (marché, unité) dans la fenêtre qui suit la clôture | **pointe 500 → 29/min** à 100 marchés |
| **B4** | `SENTINEL_SCHEDULER_WORKERS` (défaut 1 = comportement d'origine) parallélise le tick | lève le mur des 38–71 combinaisons par tick de 60 s |
| **B5** | Un 429 est terminal (`TwelveDataRateLimited`), le compteur serveur `Api-Credits-Left` fait foi, mise en attente quand il tombe à zéro | **4 crédits → 1** par échec ; plus aucune requête émise en sachant qu'elle sera refusée |
| **B6** | Verrou single-flight par (marché, unité) + mémo du dernier build | **3 → 1** crédit sur un démarrage à froid concurrent ; ferme aussi le cas « flux en retard », où les suiveurs reconstruisaient chacun de leur côté |
| **B7** | La sonde jour férié est semée avec un décalage stable par combinaison | plus de rafale de 100–200 crédits toutes les 30 min, marché fermé |
| **B9** | Les deux `Dockerfile` servent `src.api.asgi:app` au lieu de `src.intelligence.main` | supprime le scanner hérité : **−288 crédits/jour/marché** (28 800/jour à 100 marchés) |
| **B10** | Crédits comptés par déclencheur (`scheduler`, `interactive`, `backfill`, `holiday_probe`, `reference_series`, `legacy_scanner`), exposés par `/health` | une dérive devient attribuable avant le 429 ; le scanner hérité, s'il est lancé à la main via `python -m src.intelligence.main`, apparaît nommément au lieu de se fondre dans le total |
| *(bonus)* | Une combinaison ouverte une fois cesse de coûter des crédits au bout de 2 h (`SENTINEL_ON_DEMAND_ACTIVE_HOURS`) au lieu de 24 h ; elle reste disponible à la demande | −88 % sur le plancher M5 à la demande (96 → ~8 requêtes/marché/jour) |

**B8 (amorçage piloté par la marge) est traité par B5** : le limiteur respectant désormais
le compteur du serveur, l'amorçage — étiqueté `backfill` — ne peut plus déborder le
budget par minute ; il s'étire au lieu de dépasser. Aucun ordonnanceur dédié n'a été ajouté.

**Non fait, volontairement** : la dérivation M15/H1/H4 depuis M5. La mesure montre qu'elle
serait **contre-productive dans le périmètre actuel** — le produit ne suit pas M5 en direct,
donc dériver depuis M5 imposerait de le poller (289 requêtes/marché/jour) là où les quatre
unités suivies en coûtent **127**. Elle ne devient intéressante qu'à partir de ~130 marchés
avec M5 en direct, et exige au préalable la validation bit-pour-bit de §6.

### Vérification

Le simulateur lit désormais les décalages **réels** du scheduler (scénarios S5 et S6), donc il
valide le code livré et pas une idéalisation :

| Sc. | Configuration à 100 marchés | Total/jour | Pointe/min | vs 55 |
|---|---|---|---|---|
| **S5** | code corrigé, périmètre warm actuel (M15→D1) | **12 700** | **29** | ✅ 47 % de marge |
| **S6** | code corrigé + M5 en direct (`LB1_WARM_M5=1`) | 41 500 | **55** | ⚠️ exactement au plafond |

Autrement dit : **55 crédits/min suffisent pour 100 marchés dans le périmètre du produit**,
avec de la marge. Ajouter M5 en direct pour 100 marchés consomme exactement le plafond —
c'est le seul cas qui justifierait la dérivation ou un palier supérieur.

29 nouveaux tests verrouillent chacun de ces comportements
(`tests/test_data3_request_budget.py`). Deux tests existants ont été mis à jour parce qu'ils
décrivaient l'ancien contrat : le retry sur 429 (qui coûtait 4 crédits) et la fenêtre
d'activité unique. Deux tests de scheduler qui portent sur l'ORDRE demandent maintenant
explicitement `spread=False`, pour continuer à vérifier ce qu'ils vérifiaient.

---

## Annexe — invariants respectés

- **Le diagnostic (§1-8) n'a modifié aucun code de production** : il n'a ajouté que
  `tools/data_budget/` et `docs/audits/`. Les correctifs de §9 sont une étape distincte,
  chacun justifié par une mesure de ce rapport et verrouillé par un test.
- **Aucun secret dans le dépôt** : la clé est lue depuis `.env` (gitignoré) ou l'environnement ; le rapport de test a été vérifié comme ne la contenant pas avant d'être archivé.
- **Un seul dépassement réel**, non répété, ~10 crédits sur 800.
- **Aucune donnée inventée** : chaque chiffre vient de la base de production, du harnais instrumenté, ou de l'appel réel — les extrapolations sont signalées comme telles.
