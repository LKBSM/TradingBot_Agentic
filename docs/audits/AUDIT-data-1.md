# AUDIT DATA-1 — Consommation du fournisseur de données (Twelve Data)

> **Diagnostic en lecture seule. Aucun code modifié.** Objectif : chiffrer avant de décider.
> Date : 2026-08-16 · Fournisseur : **Twelve Data** (REST `time_series`).
>
> **⚠️ CORRECTIF 2026-08-16 (après revue) :** la première rédaction a été faite sur un **HEAD
> détaché 90 commits en retard** sur `origin/main`. Les faits data-path (2 symboles, M1 gated,
> M5/M1 hors live-warm, tick 60 s) sont identiques et l'analyse chiffrée tient — **mais** le
> drapeau PERF-2 `SENTINEL_INTERACTIVE_SERVE_STORED` **existe bien dans `origin/main`** (défaut
> `"1"` = activé). Voir §0-bis et §9 corrigés. Tout audit doit repartir de `origin/main`.

---

## RÉPONSE À LA QUESTION N°1 (en première ligne)

**OUI — un appel au fournisseur PEUT être déclenché par une action client.** Trois chemins :
`GET /api/market-reading`, `GET /api/candles` (séries de référence D1/W1 uniquement), et l'outil
`get_market_reading` du chatbot M.I.A. Tous passent par `MarketReadingAssembler.get_or_generate(...)`
avec `bound_provider=True` (défaut), qui, **sur défaut de cache**, appelle
`TwelveDataProvider.fetch_candles(...)`.

**MAIS ce n'est PAS proportionnel au nombre d'utilisateurs aujourd'hui**, grâce à trois garde-fous
déjà en place :
1. Le test de cache est fondé sur la **lecture stockée partagée** (`readings_store`, SQLite) : si le
   scheduler a déjà régénéré la combinaison pour la bougie courante, 100 clients obtiennent un *cache
   hit* et **zéro appel**.
2. Le provider a un **cache TTL 60 s** en mémoire (`(symbol, timeframe, lookback)`) : des misses
   simultanés identiques se rabattent sur la même réponse.
3. Sur timeout/échec, le chemin interactif **lit à travers `candles.db`** (PERF-1) au lieu de
   réappeler.

**Correctif (origin/main) — la fenêtre est encore plus étroite que décrit ci-dessous.** PERF-2
(`SENTINEL_INTERACTIVE_SERVE_STORED`, défaut activé) fait que le chemin interactif **sert la lecture
stockée sans appeler le fournisseur**, SAUF deux cas : **cold-start** (aucune lecture en base pour la
combo) ou **après un bump de `READING_LOGIC_VERSION`** (le stock d'ancienne version est reconstruit
une fois). En **régime permanent, un client ne déclenche aucun appel fournisseur** — seul le
scheduler le fait. Restent : le premier accès à une combo jamais générée, et `/api/candles` D1/W1
(`warm_candles`) au 1ᵉʳ affichage. Il n'y a toujours **pas de verrou single-flight** par `(symbole,
unité)` sur ces cas résiduels.

> ### §0-bis — Statut du drapeau PERF-2 (corrigé)
> - Commit de travail : **`944b401`** — *"perf(perf-2): fluidité — le client n'attend plus le fournisseur"*.
> - Merge : **`a464f80`** — *"Merge pull request #136 from LKBSM/perf/perf-2-fluidite"*.
> - **Présent dans `origin/main`** (`market_reading_assembler.py`, défaut `"1"`). Ni reverté, ni
>   jamais mergé : bien vivant.
> - Mon rapport initial le disait « absent » car il a été produit sur un **HEAD détaché à −90 commits**
>   (`6cfe8cf`, ~PR #120) au lieu de `origin/main` (`771ccc0`, PR #156). **Leçon process** : auditer
>   `origin/main`, pas le checkout local.

---

## 1. Cartographie complète des appels sortants

Un seul point d'appel réseau vers Twelve Data REST : `TwelveDataProvider._fetch_dataframe` →
`GET https://api.twelvedata.com/time_series` (**1 symbole/appel, 1 crédit/appel**, `outputsize` =
fenêtre demandée, `timezone=UTC`). Tout le reste transite par lui.

| # | Déclencheur | Fichier / fonction | Chemin | Symboles/appel | Crédits/appel | Fréquence |
|---|---|---|---|---|---|---|
| 1 | **Tâche planifiée** (cœur) | `scheduler.py::tick` → `assembler.get_or_generate(..., bound_provider=False)` | par combinaison *warm*, **si** une nouvelle bougie a clôturé | 1 | 1 | tick 60 s ; **appel seulement à la clôture** d'une bougie |
| 2 | **Requête client** (lecture) | `routes/market_reading.py::get_market_reading` → `get_or_generate(bound_provider=True)` | sur miss de cache uniquement, fetch **borné 5 s** puis read-through `candles.db` | 1 | 1 | par ouverture /app, changement de marché/unité — dédupliqué (voir §1 réponse) |
| 3 | **Requête client** (graphique) | `routes/candles.py::get_candles` → `assembler.warm_candles` | **seulement D1/W1** absents du cache (idempotent, market-aware) | 1 | 1 | 1 appel / bougie D1 (ou W1) close, par 1ʳᵉ consultation |
| 4 | **Requête client** (chat) | `chatbot.py` / `signal_summary_provider.py` → `get_or_generate` | outil `get_market_reading` de M.I.A | 1 | 1 | par tour de conversation nécessitant la lecture |
| 5 | **Sonde de réouverture** | `assembler.refresh_if_reopened` | jours fériés uniquement, `safety_poll_seconds=1800` | 1 | 1 | ≤ 1 / 30 min / combo, **et seulement en fermeture FÉRIÉ** (0 le week-end) |
| 6 | **Amorçage historique** | `history_backfill.py::backfill_all` + `scripts/backfill_history.py`, `scripts/seed_twelve_data.py` | **manuel/script**, PAS au boot | 1/combo | 1/combo | ponctuel, idempotent, resumable |
| 7 | **Flux temps réel (proto)** | `twelve_data_ws.py` (`LIVE_TICK_ENABLED`, défaut off) | 1 connexion WS partagée, prix courant | N/A (WS) | facturé séparément | continu si activé — **prix seul, jamais de bougie** |

**Ce qui NE consomme PAS le fournisseur** (lectures pures SQLite, vérifiées) :
- `routes/conditions_scan.py` (le **scanner**) — `readings_store.get_latest_reading` (SELECT). Jamais de fetch.
- `routes/candles.py` pour M1..H4 et D1/W1 **déjà en cache** — lecture `candles.db`.
- `routes/market_reading.py` sur *cache hit* — lecture stockée.
- `market_status`, `reference_levels`, MTF bias (`build_cache_mtf_provider`) — lectures cache.

**Absence de dérivation et de batch** : chaque combinaison est **téléchargée indépendamment**
(pas de M15 dérivé du M5), et **un symbole par requête** (pas de multi-symboles).

---

## 2. Consommation mesurée aujourd'hui (2 marchés : XAUUSD, EURUSD)

Périmètre *live-warm* actuel = `live_warm_combos()` = combinaisons activées **moins M5** et
**moins M1** (M1 gated off) = **{M15, H1, H4, D1} × 2 symboles = 8 combinaisons**.
Le fournisseur n'est appelé qu'à la **clôture** d'une bougie (`_needs_regeneration`).

Clôtures par jour (marché ouvert ~24 h, densité continue) :

| Unité | Clôtures/jour | ×2 symboles |
|---|---|---|
| M15 | 96 | 192 |
| H1 | 24 | 48 |
| H4 | 6 | 12 |
| D1 | 1 | 2 |
| **Total warm/jour** | **127/symbole** | **≈ 254 crédits/jour** |

- **Crédits/jour** : **≈ 254** (+ quelques appels D1/W1 de référence et misses interactifs résiduels ⇒ ~255–260). Marge confortable sous le plafond 800/jour du plan **Free**.
- **Crédits/heure** (marché ouvert) : ~10,6/h en moyenne (M15 4/h + H1 1/h + H4 0,25/h + D1 0,04/h, ×2).
- **PIC sur une minute** : **8 crédits/min, à minuit UTC** — quand M15+H1+H4+D1 clôturent *ensemble* pour les 2 symboles (8 combos régénérées dans le même tick). **C'est exactement le plafond Free (8/min)** : le tarif marche aujourd'hui **avec zéro marge**. Aux autres heures pleines : 4/min (M15+H1). Aux :15/:30/:45 : 2/min.
- **Part planifiée vs client** : **~100 % planifiée**. Le client ne tire un appel que dans la fenêtre ≤ 60 s post-clôture ou sur D1/W1 1ʳᵉ vue — dédupliqué par le stock partagé + TTL 60 s. Non proportionnel aux utilisateurs.
- **Appels retournant des données déjà en base** : dans le chemin **planifié, ~0** (on n'appelle que si une bougie a réellement clôturé). En revanche **chaque appel re-télécharge toute la fenêtre** (500–3000 bougies) pour **1 seule** bougie neuve : la *donnée* est ~99 % redondante, mais **Twelve Data facture par requête, pas par bougie** → coût crédit de cette redondance = **0**. Seuls les misses interactifs qui refetchent une fenêtre déjà présente dans `candles.db` sont de vrais **appels gaspillés** (faibles, bornés).

---

## 3. Extrapolation à 80 marchés × 6 unités — **architecture ACTUELLE**

6 unités = M1, M5, M15, H1, H4, D1 (les `perimeter=true`). Architecture actuelle = 1 requête par
combinaison à **chaque clôture**, aucune dérivation, aucun batch.

Clôtures/jour/marché (24 h continu) : M1 1440 · M5 288 · M15 96 · H1 24 · H4 6 · D1 1 = **1855/marché/jour**.

- **Crédits/jour** : 1855 × 80 = **≈ 148 400 crédits/jour**.
- **PIC sur une minute** (tout tire au bord, sans étalement) :
  - Chaque minute : M1 clôture pour 80 marchés ⇒ **plancher 80/min**.
  - :05, :10… : M1+M5 = 160/min.
  - Heure pleine : M1+M5+M15+H1 = 320/min.
  - Bord 4 h : +H4 = 400/min.
  - **Minuit UTC : M1+M5+M15+H1+H4+D1 = 480 crédits en une minute** ← **pic absolu**.
- **Palier requis (actuel)** : le plancher M1 seul (80/min) dépasse **Grow (55/min)**. Le pic 480/min impose **Pro ($99, 610/min)**. Le total 148 k/jour est sans effet (illimité/jour sur plans payants).

> Le facteur de coût dominant est **M1 en live pour 80 marchés** (1440 appels/marché/jour, plancher 80/min *incompressible* car M1 clôture chaque minute). C'est LA variable qui décide du palier.

---

## 4. Extrapolation à 80 marchés × 6 unités — **architecture OPTIMISÉE**

Leviers qui coupent réellement des **crédits** (= des requêtes) :
- **Dériver M15/H1/H4 d'une base M5 polled** (agrégation exacte) → ne plus poller ces 3 unités.
- **Ne pas poller M1 en live pour les 80** (M1 non dérivable, 1440/jour/marché) → M1 en **historique + rafraîchissement paresseux** des marchés consultés, âge affiché (le scanner sait déjà afficher `bars_behind`/`stale`).
- **Étaler (jitter)** les requêtes sur la minute → coupe le **pic** (le vrai driver de facture sur plan payant).
- **Sauter les marchés fermés** (déjà fait pour week-end/férié ; à étendre aux sessions d'indices ~6,5 h).

Ensemble polled/marché = **M5 (base) + D1**, le reste dérivé, M1 paresseux :

- Crédits/jour/marché (24 h) = 288 (M5) + 1 (D1) = **289**.
- **Crédits/jour** : 289 × 80 = **≈ 23 120/jour** en pire cas 24 h ; mix réel (FX 24×5, indices ~6,5 h, crypto 24×7) ⇒ **~15–20 k/jour**.
- **PIC sur une minute avec étalement** : M5 = 80 requêtes toutes les 5 min, étalées sur ~300 s ⇒ **≈ 16/min soutenu** (D1 négligeable, jitté). Sans étalement : 80/min au bord des 5 min.
- **Palier requis (optimisé)** : **Grow ($29, 55/min)** suffit, avec marge pour le rafraîchissement paresseux M1 des marchés consultés.

**Réduction : de 480 → ~16–27 crédits/min de pic (~18–30×), et de 148 k → ~15–23 k crédits/jour (~6–10×).**
Décalage de palier : **Pro $99 → Grow $29**.

> Si l'exigence est **M1 en live pour les 80 marchés** (pas de paresseux) : +115 200/jour et **plancher 80/min incompressible** ⇒ retour à **Pro $99** quoi qu'on fasse. Le choix M1-live-pour-tous vs M1-paresseux est **la** décision tarifaire.

---

## 5. Optimisations classées par gain décroissant

| Rang | Optimisation | Gain crédits | Effort | Risque |
|---|---|---|---|---|
| 1 | **M1 non live pour les 80** (historique + paresseux consultés, âge affiché) | ÉNORME — supprime 1440/marché/j **et** le plancher 80/min ; fait passer Pro→Grow | Moyen | Moyen — le scanner voit M1 potentiellement âgé ; **exige** l'affichage d'âge (déjà supporté) sinon régression scanner |
| 2 | **Étalement (jitter) sur la minute** | Coupe le PIC ~3–5× **sans changer le total** ; c'est le levier direct sur la facture (plan payant = facturé au pic/min) | Faible | Faible — pur ordonnancement, aucune donnée changée |
| 3 | **Dériver M15/H1/H4 d'une base M5** (agrégation exacte) | −126/marché/j (M15 96 + H1 24 + H4 6) = −10 080/j à 80 marchés | Moyen-élevé | **Élevé si mal fait** (bornes/fuseaux) — voir §6 ; à activer *par (marché,unité) validé* |
| 4 | **Sessions marché fermé** (indices ~6,5 h vs 24 h) | Selon composition ; ~−60 % sur les combos d'indices | Faible-moyen | Faible — étend une logique déjà présente (`market_calendar`) |
| 5 | **Fermer le chemin interactif→fournisseur** (servir stock, single-flight) | Supprime la fenêtre de miss client + rafales à la clôture ; **découple coût du nombre d'utilisateurs** | Moyen | Faible — recoupe la mission Performance (§9) |
| 6 | **WebSocket pour le prix courant** (Pro+) | Remplace le polling de quote par 1 connexion (500 symboles Pro) | Moyen | Faible — **prix seul, jamais de bougie/détection** |
| 7 | **Batch multi-symboles** | **0 crédit** (facturé 1/symbole) — gain latence/connexions seulement | Faible | Faible — n'aide pas la facture ; utile en latence |
| 8 | **Téléchargement incrémental** | **0 crédit** (1 crédit/requête quel que soit `outputsize`) — gain latence/bande passante | Moyen | Faible — **ne résout pas** le coût ; prérequis technique de la dérivation (fetch borné propre) |

> **À retenir** : contrairement à l'intuition, **B (incrémental) et A (batch) n'économisent aucun crédit** — Twelve Data facture *par requête*, pas par bougie ni par symbole. Les vrais leviers de facture sont **#1 (M1), #2 (étalement) et #3 (dérivation)**.

---

## 6. Recommandation sur la dérivation des unités supérieures + test de validation

**Faisable proprement — mais uniquement M15/H1/H4 à partir d'une base M5, et seulement après validation par marché.** Ne **jamais** dériver D1 (ancrage session/jour variable ; il ne coûte qu'1 appel/jour, on le polle). Ne pas dériver M1 (rien de plus fin).

**Pourquoi c'est plausiblement exact ici** : le provider est appelé avec `timeframe=UTC` et labellise les bougies sur des bornes d'horloge UTC. M15 (:00/:15/:30/:45), H1 (:00), H4 (00/04/08/12/16/20 UTC) sont des seaux d'horloge alignés sur M5. L'agrégation open=1ᵉʳ, close=dernier, high=max, low=min est **exacte** si les seaux coïncident.

**Pièges exacts à couvrir** :
- **Ancre H4** : vérifier que Twelve Data ancre bien H4 à 00 UTC (et non à l'ouverture de session). C'est le seul seau intraday non trivial.
- **Bougies M5 manquantes** (minute illiquide, trou de flux, halte) → la bougie dérivée aurait un high/low/volume faux. **Détecter le trou et refuser de dériver** (repli poll direct) plutôt qu'émettre une bougie approximée.
- **Week-end / férié / bords de session** : seaux partiels → ne dériver que des seaux complets.
- **Fuseaux / heure d'été** : neutralisés par l'ancrage UTC des seaux d'horloge — à re-vérifier pour H4.
- **Volume** : FX/métal renvoient souvent volume 0 ; l'agrégat de somme reste cohérent (0).

**Test de validation proposé (à implémenter avant activation, hors de cette mission)** :
1. Pour chaque marché candidat, sur **une semaine complète** : télécharger M5 **et** les M15/H1/H4 **du fournisseur**.
2. Dériver M15/H1/H4 depuis M5.
3. Assertion **bit-pour-bit** (tolérance flottante) sur **Open, High, Low, Close** de **chaque** bougie dérivée vs fournisseur ; logguer tout écart par (marché, unité, borne).
4. **N'activer la dérivation que pour les couples (marché, unité) à 100 % de correspondance.** Tout le reste reste polled directement. Réexécuter périodiquement (garde anti-régression).

**Conditions de recommandation** :
- ✅ **Recommandé** si le test passe à 100 % par marché et si la détection de trous M5 est câblée (repli poll). Le SMC vivant sur les mèches, la fidélité O/H/L/C est non négociable — le test la garantit.
- ❌ **À ne pas faire** en aveugle (sans test par marché), ni pour D1, ni sans repli sur trou M5. Un décalage de borne invisible produirait des zones SMC différentes = régression grave et silencieuse.

---

## 7. Palier tarifaire nécessaire — avant / après optimisation

**Correctif de grille (2026-08-16) :** afficher de la donnée à des **abonnés payants** exige la
grille **BUSINESS** (les plans individuels Grow/Pro sont réservés à l'usage personnel non affiché).
Le plancher n'est donc pas Grow/Pro mais **Venture**. Twelve Data facture au **crédit/minute**
(illimité/jour) → **seul le PIC/minute décide du palier.**

| Grille | Plan | Prix/mois | **Crédits/min** |
|---|---|---|---|
| Business | **Venture** | ~149 $ (à confirmer)¹ | **610** |
| Business | **Enterprise** | ~1 099 $ | **10 946** |
| Business | Enterprise+ | sur devis | 10 000+ (custom) |

¹ *La page business publique affiche Venture à ~499 $/mois (414 $ annuel) pour 610 cr/min — écart
avec le chiffre de 149 $ à revalider côté compte/sales. Le **plafond 610 cr/min est confirmé** ; les
calculs ne dépendent que de lui.*

- **Aujourd'hui (2 marchés)** : pic 8/min ⇒ Venture couvre trivialement (mais Venture est le
  **plancher obligatoire** dès qu'on affiche à des abonnés payants, quel que soit le volume).
- **80×6 archi actuelle (M1 live)** : pic **480/min** ⇒ **Venture (610)** — **marge 130/min = 21 %**.
- **80×6 optimisée (M1 paresseux + base M5 dérivée + jitter)** : pic ~16–27/min ⇒ **Venture**, marge énorme.

**À quel nombre de marchés Enterprise devient-il obligatoire ?** À l'archi actuelle le pic = **6 × N**
(6 unités coïncident à minuit). `6N ≤ 610 ⇒ N ≤ 101`. **À 102 marchés, Enterprise devient
obligatoire.** Avec étalement (plancher M1 = N/min) Venture tiendrait jusqu'à ~610 marchés ; en
architecture optimisée, Venture couvre des milliers de marchés — Enterprise jamais atteint à cette échelle.

**Conclusion tarifaire : Venture (~149 $, 610 cr/min) est le plancher, et il suffit pour 80 marchés
dans les deux architectures. L'optimisation ne fait pas économiser un palier ici — elle achète de la
MARGE (de 21 % à ~95 %) et repousse le mur Enterprise de 101 marchés à plusieurs centaines/milliers.**

---

## 8. Plan d'amorçage de l'historique (78 nouveaux marchés × 6 unités)

**Bonne nouvelle : ce n'est PAS un risque de saturation.** `backfill_combo` fait **1 requête par
combinaison** (`outputsize ≤ 5000` couvre toute la profondeur configurée ; la plus profonde,
H1 6mo ≈ 4384 bougies, tient en une requête). Profondeurs (`config/lookback_depths.json`) :
M1 1d · M5 1w · M15 1mo · H1 6mo · H4 2y · D1 5y — **toutes < 5000 bougies ⇒ 1 requête chacune.**

- **Coût total** : 78 marchés × 6 unités = **468 requêtes ponctuelles** (+ 2 marchés existants si re-seed = 480).
- **Durée** (limiteur intégré) : **Free** 8/min ⇒ ~59 min (et tient dans le plafond 800/j) ; **Grow** 55/min ⇒ ~8,5 min ; **Pro** 610/min ⇒ < 1 min.
- **Propriétés** : `backfill_all` est **idempotent** (INSERT OR REPLACE), **resumable** (combo « deep enough » + à jour = skip), **market-aware** (0 appel pour période fermée).

**Plan recommandé** :
1. Provisionner le plan cible (Grow/Pro) **avant** l'amorçage.
2. Lancer `scripts/backfill_history.py` (→ `backfill_all` sur `enabled_combos()`) en **fenêtre creuse**, ou laisser le limiteur entrelacer avec le live.
3. Le relancer une fois pour absorber les échecs isolés (skip des combos déjà pleines).
4. Puis régime permanent (scheduler). Aucune pagination nécessaire tant que les profondeurs restent < 5000 bougies (garde déjà loguée si dépassement).

> Nuance : si un futur approfondissement porte une unité au-delà de 5000 bougies (ex. D1 20 ans),
> il faudra la **pagination date-fenêtrée** (non implémentée, avertissement déjà en place).

---

## 9. Recoupements avec la mission Performance

**Même cause racine, même correctif.** Le chemin interactif `get_or_generate(bound_provider=True)`
appelle le fournisseur **de façon synchrone (bornée 5 s)** sur un miss de cache : c'est à la fois

- une source de **latence** (le client attend le réseau) → problème **Performance** ; et
- une source de **coût/duplication** (appel client-déclenché, rafales à la clôture) → problème **Data**.

Le correctif est unique : **ne servir que le stock partagé (`candles.db`/`readings_store`) sur le
chemin interactif, et laisser le SEUL scheduler toucher le fournisseur** (direction « serve stored »
de PERF-2). Cela :
1. supprime la fenêtre de miss client et les rafales simultanées (Data) ;
2. rend chaque requête client purement locale, ~0,03 s (Performance) ;
3. **découple définitivement le coût fournisseur du nombre d'utilisateurs.**

**Corrigé (origin/main) :** le drapeau `SENTINEL_INTERACTIVE_SERVE_STORED` **est présent et activé par
défaut** (PERF-2, PR #136, commit `944b401`) — le chemin interactif **sert déjà le stock** sans
appeler le fournisseur, sauf cold-start / bump de version. Le gros du correctif est donc **déjà en
place**. Reste à durcir : un **verrou single-flight** par `(symbole, unité)` pour les cas résiduels
(cold-start, `warm_candles` D1/W1), en gardant l'affichage d'âge (`market_status`, `bars_behind`)
comme filet d'honnêteté.

---

## Annexe — invariants respectés (aucune optimisation ne les viole)

- Précision O/H/L/C : la dérivation n'est activée **qu'après** validation bit-pour-bit par marché ; sinon poll direct.
- Aucune réduction d'historique : profondeurs `lookback_depths.json` inchangées ; amorçage complet.
- Donnée périmée : toujours servie **avec âge affiché** (`market_status` / `bars_behind` / `freshness`).
- Aucun marché retiré, aucune unité supprimée : M1 reste **disponible** (historique + paresseux), pas amputé.
- Aucune donnée approximée : dérivation = agrégation **exacte et vérifiée**, avec repli sur trou M5.
