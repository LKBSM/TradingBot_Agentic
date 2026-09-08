# AUDIT — MIA-3 · Un seul M.I.A, partout, avec accès à tout ce que le produit sait

> **Phase DIAGNOSTIC (lecture seule) — STOP avant tout code.**
> Branche : `feat/mia-3-agent-unifie` · worktree dédié `C:/MyPythonProjects/wt-mia-3`.

## Position du HEAD (discipline origin/main)

- `git fetch` fait. `origin/main` = **271443a** (PR #196 BRD-3).
- Le répertoire principal était sur `docs/preserve-data-1-audit` @ **e0dc69c**, soit **38 commits DERRIÈRE origin/main** (et 1 devant). J'ai donc **créé un worktree neuf `wt-mia-3` depuis `origin/main`** pour auditer le vrai état courant et non un HEAD périmé de −38.
- Tout ce rapport décrit le comportement de **`origin/main` (271443a)**, pas du répertoire principal périmé.

⚠️ **Attention infrastructure pour le test C** : les DB de données (candles/market_readings) et la clé `ANTHROPIC_API_KEY` ne vivent QUE dans le répertoire principal (−38). Le code du chatbot y est différent (`chatbot.py` : +256 lignes entre e0dc69c et origin/main — MIA-1 streaming). J'ai donc exécuté le **code origin/main (wt-mia-3)** en pointant les stores vers les **données du répertoire principal** via `CANDLES_DB_PATH` / `MARKET_READINGS_DB_PATH`, `.env` chargé pour les clés. C'est le vrai chatbot origin/main sur données réelles. *(Un artefact en découle — voir §Test C, note « haiku_generated ».)*

---

## A) UN COMPOSANT OU DEUX ? → **DEUX (en réalité TROIS surfaces), c'est LA dette de la mission**

Il n'y a **pas** un panneau M.I.A partagé. Il y a des implémentations distinctes :

| Surface | Composant | Moteur | Outils | Sélection zone | Conversation |
|---|---|---|---|---|---|
| **/app** (+ landing, + toutes pages produit via `ProductShell`) | `AppChatSidebar` → `ChatProvider`/`useChat` → `askSentinelStream` → **backend LLM** (`/api/chatbot/stream`) | **Haiku + tool use** (le vrai M.I.A) | 4 tools serveur | — (pas de zones) | persistée localStorage, threads par combo `app:{inst}:{tf}` |
| **/zones** | `ZoneMiaPanel` | **STUB LOCAL déterministe** : `matchTopic()` (regex fr/en) → 4 topics figés, **aucun réseau, aucun LLM, aucun tool** | **aucun** | prop React `zone` | `React.useState` local, **non persistée, non partagée** |
| **/actualites** (fiche publication) | `MiaBlock` (dans `CalendarEventDetail`) | `askSentinel` → **backend LLM** (`/api/chatbot/message`, non-stream) | 4 tools serveur | — | locale au composant |

- **/app** et **/actualites** parlent au **même backend** mais via **deux composants front distincts** (`AppChatSidebar` vs `MiaBlock`) et deux clients (`askSentinelStream` vs `askSentinel`).
- **/zones** ne parle **pas** au backend du tout : c'est un moteur de réponses **complètement séparé et beaucoup plus pauvre** (4 réponses canned). Le mot du fondateur — « celui de /zones est trop étroit et pas au niveau » — est exact : ce n'est pas une question de largeur, c'est **un autre produit**.
- Le seul partage réel aujourd'hui = `ChatComposer` (champ+dictée+note vie privée, extrait par CLN-1) et `AgentAvatar`. Le **cerveau** n'est pas partagé.

> **Conséquence** : chaque correctif de M.I.A doit aujourd'hui être fait 1 à 3 fois. La mission = **un seul cerveau (le backend LLM à outils) derrière un seul composant de panneau**, /zones et /actualites cessant d'avoir leur propre moteur.

---

## B) LA SURFACE D'OUTILS AUJOURD'HUI

Le backend expose **4 tools** (`src/intelligence/chatbot/chatbot.py` `TOOL_SCHEMAS`). Ils sont **identiques quelle que soit la page** — MAIS ne sont atteignables que par les surfaces qui appellent le backend (**/app + /actualites**). **/zones n'a AUCUN de ces outils.**

| Outil | Renvoie | Si donnée absente | Verrou d'identifiant |
|---|---|---|---|
| `get_market_reading(instrument, tf)` | Lecture complète (structure SMC, régime, news, conditions) | `{"error": ...}` → M.I.A avoue l'échec (vu au test C) | **enum** `instrument∈{XAUUSD,EURUSD}`, `tf∈{M15,H1,H4}` : un marché/unité hors liste est **rejeté par le schéma**, l'appel n'a même pas lieu |
| `get_signal_summary()` | Résumé des 6 combos suivis | fallback `{"instruments_tracked":[]}` | n/a (pas d'argument) |
| `get_ob_diagnostic(inst, tf, price\|ts)` | Diagnostic moteur réel (checks passés/échoués) | `no_data` / `unresolved` → aveu | enum inst/tf ; `price`/`ts` libres mais le moteur résout ou dit `no_data` |
| `apply_chart_view(action, params)` | Ack d'action d'affichage (vue seule) | rejet si `empty_category` | **OUI, verrou complet (Couche 4 `ViewActionValidator`)** : `zone_id`/`category` doivent correspondre à des ids **réellement émis ce tour** (`known_zone_ids`), sinon rejet **par le code** |

**Ce que le produit SAIT mais qu'AUCUN outil n'expose (angles morts à combler) :**
- **Calendrier / publications** (`/actualites`, `CalendarMonthView`, mesures de publication) — pas d'outil. *(Confirmé au test C : M.I.A n'a aucun accès au NFP.)*
- **Résultats du scanner** (`/scanner`, stratégies enregistrées) — pas d'outil (le scanner a son propre `ScannerTranslator`, séparé).
- **Catalogue des marchés** (`config/markets.json` / registre MKT-1, incl. entrée fictive `TESTMKT`) — pas d'outil.
- **Zones en tant que sujet interrogeable** au-delà de la structure brute d'une lecture.

**Verrou d'identifiant : couverture actuelle** = **`apply_chart_view` uniquement**. Les lectures s'appuient sur l'**enum** (efficace pour inst/tf, mais il n'existe pas encore de notion d'`event_id` / `strategy_id` / `market_id` à verrouiller). ⇒ **La mission doit étendre le verrou à chaque nouvel outil** (id de publication, de stratégie, de marché rejetés par le code s'ils n'ont pas été émis).

---

## C) LE TEST DÉCISIF — réponses RÉELLES (Haiku, origin/main, données réelles)

### /app — question hors périmètre de page

> ⚠️ Note « haiku_generated » : les readings en cache (`market_readings.db` du répertoire principal) portent `description_source='haiku_generated'`, valeur **supprimée** par origin/main (refactor gabarit déterministe). `get_market_reading` **échoue donc la validation Pydantic** dans mon harnais → M.I.A reçoit un `error`. **En production les readings sont régénérés** avec la nouvelle source, donc l'outil réussirait. Cet artefact n'affecte PAS les 3 questions hors-périmètre (elles ne peuvent de toute façon pas appeler l'outil), mais explique la « erreur technique » sur le contrôle EURUSD H4.

**Q1 — publication calendrier (NFP)** — *« Quel est le dernier chiffre du NFP publié et sa valeur exacte ? »*
> *« Je n'ai pas accès aux données économiques externes — je ne consulte que les conditions de marché détectées par le moteur… cela ne me dit pas quel NFP a été publié ni sa valeur. Pour connaître le dernier NFP… tu devras consulter un calendrier économique externe (Investing.com, Trading Economics…). Je peux en revanche te décrire comment le marché réagit… »*
> **→ Aveu d'absence honnête. AUCUN chiffre fabriqué.** Outils appelés : aucun.

**Q2 — marché hors enum (BTCUSD)** — *« Décris la structure du BTCUSD en H4… »*
> *« Je ne peux pas analyser le BTCUSD — je suis limité à deux instruments : XAUUSD et EURUSD… »*
> **→ Aveu honnête.** L'enum bloque l'outil ; le modèle avoue la limite. Aucun appel.

**Q3 — unité non suivie (EURUSD W1)** — *« …structure de l'EURUSD en hebdomadaire (W1) ? »*
> *« L'EURUSD en hebdomadaire (W1) n'est pas suivi. Je n'ai pas accès à ces données. Je peux te donner… M15, H1 ou H4. »*
> **→ Aveu honnête.** Aucun appel.

**Q4 — contrôle DANS le périmètre (EURUSD H4)** — appelle bien `get_market_reading(EURUSD,H4)` (×2) ; échoue sur l'artefact `haiku_generated` → *« je rencontre une erreur technique… je ne peux pas charger les données »*.
> **→ Même en erreur d'outil, M.I.A n'invente pas** ; il dit qu'il ne peut pas charger.

### /zones — même famille de question (déterminé par le CODE, pas de LLM)
`ZoneMiaPanel.handleSubmit` → si aucune zone : `zones.mia.answer.noZone` = *« Sélectionne d'abord une zone : je décris celle que tu choisis… »*. Avec une zone : `matchTopic(q)`.
- *« Décris l'EURUSD »* → aucun mot-clé → `zones.mia.answer.fallback` = *« Je décris cette zone à partir de ses faits : … Choisis une de ces questions. »*
- *« et sur un **autre** marché ? »* → le mot « autre » **matche `whatElse`** → décrit la **confluence de la zone COURANTE**, en ignorant EURUSD (mauvais routage silencieux).

> **Conclusion C** : le **risque de fabrication est le plus faible là où on le craignait** : le M.I.A LLM de /app **avoue déjà** l'absence (grâce à l'enum + au system prompt « tu n'inventes jamais »). Le vrai défaut est **/zones** : il ne fabrique pas non plus, mais il est **incapable d'atteindre la connaissance du produit** — il dévie vers 4 réponses figées ou répond à côté (la zone courante). L'ergonomie n'est donc pas le seul enjeu : **/zones est un cul-de-sac fonctionnel**, à remplacer par le vrai agent.

---

## D) COMMENT LA ZONE EST TRANSMISE

- **/zones** : la zone sélectionnée est une **prop React** (`zone`) passée au stub local ; elle **N'entre dans aucun prompt** (il n'y a pas de LLM). Elle **restreint tout** : sans zone, `noZone` ; le stub ne sait parler que de la zone courante.
- **/app** : pas de zones. Le combo actif (`instrument`/`timeframe`) est transmis au backend **comme préambule texte** injecté dans `user_message` par `withSignalContext()` : `"[Lecture en cours : XAUUSD M15]\n<question>"`. Ce n'est **pas** un résultat d'outil : c'est une **orientation** que le modèle peut suivre (appeler `get_market_reading` sur ce combo) ou dépasser (l'utilisateur peut demander EURUSD H4). C'est exactement le modèle « oriente sans enfermer » que la mission veut pour la zone.

> **Piste MIA-3** : transmettre la zone sélectionnée par le **même mécanisme de préambule** (`[Zone sélectionnée : <zone_id réel>]`), le `zone_id` étant **verrouillé** côté outil. La zone oriente ; une question hors zone reste traitée.

---

## E) CONVERSATION & NAVIGATION — recommandation

**État actuel** : trois silos.
- /app : `ChatProvider` global (monté au `[locale]/layout`), threads persistés localStorage par combo, **suivent l'utilisateur tant qu'il reste sous `ProductShell`**.
- /zones : `useState` **local**, **perdu** dès qu'on quitte la page.
- /actualites : local au `MiaBlock`.
- Donc /zones → /app : la conversation de /zones **est perdue** (systèmes différents).

**Recommandation : UNE conversation unique qui suit l'utilisateur** (le `ChatProvider` global devient la source unique pour /zones et /actualites aussi), avec un **fil « produit » continu** plutôt qu'un fil par combo, la zone/lecture/publication courante n'étant qu'un **contexte d'orientation** ajouté aux tours.

Justification : la mission dit « M.I.A répond sur tout le produit depuis n'importe quelle page » et « la sélection oriente, elle n'enferme pas ». Un fil par page contredit ce modèle mental (l'utilisateur pose une question sur /app à propos d'une publication vue sur /actualites — c'est **une** conversation). `ChatProvider` sait déjà persister/restaurer et **ne jamais effacer** en changeant de contexte — on garde cet acquis, on l'étend. **Nuance** : conserver le regroupement par combo comme « historique/threads récents » (déjà présent), mais le **fil actif** est continu. ⇒ **Je recommande la conversation unique ; je ne trancherai pas seul — décision fondateur.**

---

## F) COÛT EN CONTEXTE / LATENCE (mesuré, `count_tokens`, Haiku)

- Prefill actuel **système + 4 tools + 1 court message** = **5 178 tokens** (exact).
  - dont **définitions des 4 tools = 2 672 tokens** (`apply_chart_view` à lui seul est massif).
  - dont system prompt + `signal_summary` ≈ **~2 500 tokens**.
- **Ajout envisagé** (calendrier/publications, résultats scanner, catalogue marchés) ≈ **3 outils** de description riche ⇒ **+~600 à +~1 400 tokens** sur le prefill de **chaque tour**.

**Impact latence** : le prefill grossit d'environ **+15 à +30 %**. **MAIS** :
1. Le préfixe `system`+`tools` est **stable** → éligible au **prompt caching Anthropic** (TTL 5 min). Sur cache hit, le coût marginal des définitions supplémentaires est **quasi nul** (≈0). *À vérifier : le chatbot n'active pas encore `cache_control` — c'est une **amélioration à faire dans MIA-3** pour neutraliser le surcoût, et ça bénéficie aussi à MIA-2.*
2. Alternative si on veut éviter tout surcoût même à froid : **exposer les nouveaux outils de façon conditionnelle** selon la page/contexte — mais cela **réintroduit un jeu d'outils qui varie par page**, ce que la mission interdit explicitement (« le même jeu d'outils sur toutes les pages »). ⇒ **Je recommande : jeu unique + prompt caching**, pas d'outils conditionnels.

> **Réponse nette à F** : oui, sans mesure de mitigation, MIA-3 alourdit le prefill de ~600–1400 tokens/tour (~+15–30 %). Avec `cache_control` sur le préfixe système+tools (à ajouter), le surcoût réel tombe à ≈0 sur cache chaud. **Cette mission ne doit donc pas ralentir M.I.A à condition d'ajouter le caching** — je le signale au STOP comme demandé.

---

## G) MODE D'AFFICHAGE (bulle / colonne)

- **/app uniquement** a deux dispositions. État = `ChatColumnContext` (`localStorage 'mia.app.chat-column-open'`, défaut = colonne). Bascule dans l'en-tête (`AppChatSidebar`), bouton `min-[1100px]`.
- **Seuil = 1100px** (APP-1 a fait passer 1280→1100 ; défini dans `shell.css` `@media (min-width:1100px)` + boutons `min-[1100px]`). En-dessous de 1100 : pas de colonne (tiroir/bulle forcés), une ligne de statut explique pourquoi.
- Le choix persiste et **résiste au resize** (localStorage écrit seulement sur bascule explicite).
- **/zones et /actualites : AUCUN mode bulle/colonne.** `ZoneMiaPanel` est un simple `<aside>` dans la grille de la page. ⇒ MIA-3 doit **étendre les deux modes au composant unique**, donc à /zones aussi. La mention « non synchronisé » existe déjà (`chat.modeNotSynced`) — à réutiliser.

---

## Ce que la construction devra toucher (aperçu, pour le GO)

1. **Un composant unique** de panneau M.I.A (dérivé de `AppChatSidebar`, moteur = `ChatProvider`/backend), rendu en **bulle OU colonne**, monté sur /app, /zones, /actualites. Suppression du moteur local de `ZoneMiaPanel` et du `MiaBlock` séparé.
2. **Contexte d'orientation** générique (combo, **zone_id**, **event_id**, lecture) via préambule + verrou côté outil.
3. **Nouveaux outils backend** : `get_calendar`/`get_publication(event_id)`, `get_scanner_results`/`get_strategy(strategy_id)`, `list_markets` — chacun avec **état d'absence explicite** et **verrou d'id rejeté par le code**.
4. **`cache_control`** sur le préfixe système+tools (neutralise F).
5. **Mode bulle/colonne** étendu au composant unique (persisté, résiste au resize, contrôle atteignable dans les 2 modes).
6. i18n **fr/en/es** ensemble ; invariants (aucun mot prédictif/prescriptif, 1 seul avertissement/page, pas de compte avant chargement, pas de donnée→pas d'élément).

### ⚠️ Points nécessitant l'aval fondateur AVANT code (comme demandé au STOP)
- **Moteur/cache partagé** : ajouter 3 outils = **modifier le backend** (`TOOL_SCHEMAS`, `_execute_tool`, harvest d'ids, verrou étendu) + probablement `SignalSummaryProvider`. **Je le signale : la mission touche le moteur du chatbot.**
- **Décision E** (conversation unique vs par page) — à trancher.
- **Décision F** (accepter +tokens avec caching, vs statu quo) — à valider.

---

# SUIVI CONSTRUCTION (après GO — E=conversation unique, F=caching)

## Phase 1 — FONDATION BACKEND · LIVRÉE ET TESTÉE (commit `feat(mia-3): surface d'outils…`)

### Outils AVANT → APRÈS (couverture du verrou d'identifiant)

| Outil | Avant | Après | Verrou d'id |
|---|---|---|---|
| get_market_reading | ✅ | ✅ | enum inst/tf |
| get_signal_summary | ✅ | ✅ | n/a |
| get_ob_diagnostic | ✅ | ✅ | enum inst/tf |
| apply_chart_view | ✅ | ✅ | Couche 4 (ids émis) |
| **list_markets** | ❌ | ✅ | n/a (pas d'id en entrée) |
| **get_economic_calendar** | ❌ | ✅ | **marché inconnu → `found:false` PAR LE CODE** |
| **get_publication** | ❌ | ✅ | **event_id inconnu → `found:false` PAR LE CODE** ; mesure None restituée |

Le verrou des nouveaux outils de lecture est **intrinsèque** : un id jamais émis résout sur du vide → le tool renvoie une absence explicite, le modèle ne reçoit **aucune** charge fabriquée à relayer. Reste + refus de trade + Couches 1/3/4 : **inchangés**.

### Test C — APRÈS (Haiku réel, données réelles, nouveaux outils)

- **NFP / or** — appelle `get_economic_calendar(XAUUSD, upcoming)`. Feed réel : aucun événement rattaché à l'or sous 7 j → *« Aucune publication économique majeure n'est programmée prochainement pour l'or (XAUUSD)… Si tu recherches les événements qui influencent l'or indirectement (NFP, CPI, taux…), je peux scanner le calendrier US. »* → **fondé sur l'outil, zéro chiffre inventé.**
- **/zones → autre marché** (préambule `[Zone sélectionnée : …]`) — appelle `get_economic_calendar(EURUSD, upcoming)` → *« Décision de taux (BCE) — 10 septembre 2026 à 12:15 UTC… État : non encore publié. »* → **réponse cross-marché RÉELLE**, état de valeur honnête. (Point 4 de la mission satisfait côté données.)
- **BTCUSD / catalogue** — appelle `list_markets()` → *« Non, le produit ne suit pas le BTCUSD. Marchés couverts : XAUUSD, EURUSD… »* → **fondé sur le registre.**

### Coût jetons AVANT → APRÈS (mesuré `count_tokens`, Haiku)

- Définitions d'outils : **2 672 → 3 803** tokens (7 outils).
- Prefill système+outils+1 msg : **5 178 → 6 936** (+1 758, +34 %).
- **Mitigation livrée** : `cache_control: ephemeral` sur le préfixe tools+system (`Chatbot.chat_events`). Sur cache chaud (TTL 5 min), ce préfixe est facturé ~0,1× et ne réintroduit **pas** la latence de reprocessing ⇒ surcoût réel ≈ 0 en régime conversationnel. Bénéficie aussi à MIA-2.
- Tests : 305 verts (dont verrou marché/event_id inventé, absence explicite, valeur échouée verbatim, mesure None, breakpoint de cache).

### Périmètre outil « scanner » — décision de conception
Les stratégies scanner sont **client-only (localStorage, Loi 25)** : il n'existe **pas** de « stratégies de l'utilisateur » interrogeables côté serveur. Le scan serveur (`POST /api/conditions-scan`) évalue une palette de conditions **fermée et validée** contre les lectures. Exposer cette palette au LLM (construction de conditions) est une sous-mission à surface d'hallucination réelle. **Phase 1 livre marchés + calendrier + publications** (le trou du Test C, le NFP). Un outil scanner (catalogue de conditions descriptif, voire scan piloté) est proposé en incrément dédié — signalé, pas bâclé.

## Phase 2 — UNIFICATION FRONTEND · À FAIRE (refactor de coquille, gaté sur ta confirmation visuelle live)

Constat structurant : `ProductShell` ne monte le chat docké que sur /app (`{isApp && <ShellChat />}`) ; les autres routes sont en grille 2 colonnes `no-chat`. « M.I.A partout en bulle/colonne » = **refactor de la grille de coquille**, pas un simple échange de composant. Plan :

1. **Composant unique** : promouvoir `AppChatSidebar` (moteur `ChatProvider`/backend) en panneau M.I.A générique rendu en **bulle OU colonne** ; supprimer le **moteur local** de `ZoneMiaPanel` et le `MiaBlock` séparé (ils deviennent des points de montage du composant unique + un bloc « sujet » d'orientation).
2. **Conversation unique** (décision E) : `ChatProvider` passe d'un fil par combo à **un fil produit continu** qui suit l'utilisateur ; le combo/zone/publication courant devient un **contexte d'orientation** (préambule + bloc sujet), pas une clé de fil. Persistance conservée ; « récents » repensés.
3. **Coquille** : la grille `chatcol` et `ChatColumnContext` s'étendent à toutes les routes produit (pas seulement /app) ; seuil bulle/colonne 1100px conservé ; contrôle atteignable **dans les deux modes** ; « non synchronisé » déjà en i18n.
4. **Orientation zone /zones** : préambule `[Zone sélectionnée : <zone_id réel>]` (id verrouillé), reclic = désélection, aucune zone → bloc sujet non rendu, désélection n'efface pas la conversation, question hors zone traitée normalement.
5. **Invariants** : 1 seul avertissement/page (déjà centralisé `shell-mdisclaimer` + rail), aucun compte avant chargement, pas de donnée→pas d'élément, aucun mot prédictif/prescriptif, fr/en/es ensemble.
6. **Tests** : composant unique (échec si 2 panneaux), persistance mode au reload + résistance resize, survie conversation/zone à la bascule, reclic désélectionne, aucune zone→pas de bloc, désélection≠effacement, question calendrier depuis /zones fondée sur l'outil, marché/unité non lu → aveu, id inventé rejeté, vocabulaire interdit 3 langues. + Playwright 1280/1440/390 × fr/en × /zones+/app × bulle+colonne × avec/sans zone × conv. 4 messages.

**Merge sur main : seulement après ta confirmation visuelle live** (exigence mission). tsc+build+vitest à la fin de Phase 2.
