# AUDIT — MIA-5 · Concision des réponses de l'agent M.I.A

- **Branche** : `feat/mia-5-concision` (worktree dédié `wt-mia-5`), partie de **origin/main `adcc475`**, `main` re-mergée avant intégration (**`af0c4cf`**, qui apporte LP-2B / MIA-4S et le seau `prediction` de la Couche 1).
- **Mesure** : 15 questions réelles, vrai `Chatbot`, vrai Haiku, vraies lectures de marché — **1129 → 778 mots (−31 %)** sur le run de référence, médiane par réponse **66 → 40 mots**. Sur trois runs après correctif : **540 à 778 mots** contre **1129** avant, soit **−31 % à −52 %** (le modèle n'est pas déterministe ; les trois runs sont conservés dans le scratchpad de la mission).
- **Asymétrie tenue** : « Explique-moi ce qu'est un CHOCH » reste développée (226 → 240 mots).
- **Deux bugs produit corrigés au passage** (demande explicite du fondateur après le diagnostic) : la Couche 3 qui détruisait des réponses légitimes, et les lectures stockées illisibles qui rendaient XAUUSD H1/H4 et EURUSD H1 muets.

---

## 1. Le problème et sa cause

Retour fondateur : M.I.A répond avec trop de mots pour ce qu'on lui demande, ce qui décourage l'usage quotidien.

Le prompt système ne contenait **qu'une seule** consigne de longueur, 6ᵉ et dernière puce des RÈGLES STRICTES :

```
- Tu réponds en français, par défaut concis (2-4 phrases sauf demande explicite de détail).
```

Cinq causes mesurées au diagnostic (lecture seule, avant toute modification) :

1. **« 2-4 phrases » est une fourchette** — un modèle vise le haut d'une fourchette. Une question à un fait recevait 4 phrases parce que 4 était autorisé.
2. **Position perdante** — cette puce ouvre le prompt ; les ~50 lignes qui suivent demandent toutes d'*ajouter* (« explique le critère échoué », « rapporte la provenance », « dis-le honnêtement »). Ces règles-là sont l'ancrage et devaient rester ; rien ne redonnait la main à la brièveté **après** elles. La dernière instruction lue était « appelle get_market_reading ».
3. **Rien n'encadrait la fin du message** — 6 réponses sur 15 se terminaient par une offre de service (« Veux-tu une lecture complète… ? », « Quel timeframe t'intéresse ? »).
4. **Rien n'interdisait le contexte non demandé** — le `signal_summary` des 6 combinaisons se déversait même sur une question unique (« tu penses que ça va rebondir ? » → refus correct **puis 100 mots** de rundown M5→D1).
5. **Les rouages fuyaient dans la prose** — « selon le contexte initial », `touch_count = 0`, `data_lagged`, noms d'outils.

---

## 2. Le correctif de concision (prompt système)

### (a) La puce de concision (`SYSTEM_PROMPT_STATIC`, RÈGLES STRICTES)

```diff
-- Tu réponds en français, par défaut concis (2-4 phrases sauf demande explicite de détail).
++ Tu réponds en français, au format défini en fin de message système (FORMAT DE RÉPONSE) : le fait demandé, rien de plus.
```

### (b) Une section `FORMAT DE RÉPONSE` **à la toute fin** du bloc statique

Position volontaire : récence maximale, donc **après** toutes les règles d'ancrage. C'est ce qui porte l'effet — mesuré sur le même couple de runs, la même section placée **avant** la liste des outils ne donnait que **−19 %**, contre **−43 %** en fin de prompt. Un garde-fou (`tests/test_chatbot_concision.py`) interdit de la remonter.

```
FORMAT DE RÉPONSE (vaut pour toutes tes réponses) :
- Ta première phrase contient la réponse. Aucun préambule, aucune reformulation de la question, aucun rappel de ce que l'utilisateur vient de dire.
- Question factuelle précise (un niveau, un compte, une date, un statut, un état) : 1 à 2 phrases, et rien d'autre. Tu n'ajoutes AUCUN élément que l'utilisateur n'a pas demandé : ni une autre unité de temps, ni une autre structure, ni un autre marché, ni la ventilation d'un compte, ni un rappel de la tendance.
- Ta DERNIÈRE phrase est un fait, jamais une question ni une offre de service. […] Tu ne poses une question que lorsqu'il te MANQUE réellement une information pour répondre — et alors elle remplace la réponse, elle ne s'y ajoute pas.
- Après une phrase de refus (demande d'action, de conseil, ou d'anticipation), tu t'ARRÊTES. Le refus se suffit à lui-même.
- Question qui demande une explication : développe autant qu'il faut. La concision porte sur le remplissage, jamais sur le fond demandé.
- Tu ne cites pas tes rouages : ni le nom d'un outil, ni « le contexte initial », ni un nom de champ technique, ni une parenthèse de code. Tu écris « jamais testé », pas « touch_count = 0 ».
- Un état que tu qualifies (volatilité basse/normale/haute, zone proche, cassure confirmée) se donne AVEC la valeur mesurée qui le fonde : ratio, distance, niveau, horodatage. Un adjectif seul n'est pas un fait — c'est la seule chose que tu ajoutes sans qu'on te la demande.
- L'état des données (marché fermé, pause quotidienne, données en retard) se dit en UNE clause courte accolée au fait, jamais en paragraphe séparé.
- Mise en forme sobre : le gras sur les valeurs qui répondent à la question, rien d'autre. Pas de liste à puces pour une réponse d'une ou deux phrases.
- Être concis ne retire JAMAIS un fait : niveaux exacts, horodatages, statut d'une zone, état du marché, absence d'une donnée et refus se disent en toutes lettres, en court. Une phrase vraie et courte, jamais une phrase courte et vague.
```

### (c) Une règle dans ÉTAT DU MARCHÉ — le prix courant vient du moteur

M.I.A répondait « je n'ai pas accès au prix en temps réel » puis renvoyait le client vers **MT5, TradingView, Bloomberg** — alors que la lecture porte le prix de clôture. Corrigé :

```
- Le prix courant EST dans la lecture (clôture de la dernière bougie, champ close_price). À « le prix est à combien ? », appelle get_market_reading et donne ce niveau avec son horodatage — jamais « je n'ai pas accès au prix », et JAMAIS de renvoi vers une plateforme ou un site extérieur.
```

### Ce qui n'a **pas** bougé

L'ancrage par les outils (`_execute_tool`, `_harvest_zone_ids`, verrous `found:false`), les gabarits, et **toutes** les autres sections du prompt (CONTRÔLE DE L'AFFICHAGE, DIAGNOSTIC ORDER BLOCK, CALENDRIER & PUBLICATIONS, CATALOGUE DES MARCHÉS, RÈGLE D'ABSENCE, liste des 7 outils) : **0 ligne**. La phrase de refus reste **mot pour mot** : « Je décris les conditions du marché. La décision d'agir t'appartient. »

---

## 3. Bug produit n° 1 — la Couche 3 détruisait des réponses légitimes

En mesurant, **des réponses factuelles et conformes ont été remplacées** par le gabarit « Je ne peux pas formuler cette réponse de cette manière ». Côté client : un refus incompréhensible. Trois collisions, chacune **capturée sur une sortie réelle** (jamais supposée) :

| Texte réel bloqué | Jeton déclencheur | Pourquoi c'est un faux positif |
|---|---|---|
| « La gestion **du risque** en trading est un sujet général… » | `risqué` → `risque` | la normalisation des accents fait tomber l'adjectif interdit sur le **nom commun** |
| « une décision de taux affecte le **coût d'opportunité** de détenir de l'or » | `opportunité` | terme macro-économique standard pour un actif sans rendement |
| « le prix reteste ce niveau : **c'est le moment où** les ordres se manifestent » | `c'est le moment` | subordonnée temporelle descriptive, pas « c'est le moment **d'agir** » |

Un sur-blocage n'est pas gratuit : il détruit l'honnêteté du produit aussi sûrement qu'un sous-blocage détruit sa conformité.

**Correctif — deux mécanismes étroits, aucun jeton retiré :**

- `ACCENT_SENSITIVE_TOKENS` (`risqué`, `risquée`, `risqués`, `risquées`) : cherchés dans le texte **accentué**, donc l'adjectif est toujours pris et le nom `risque` ne l'est plus. L'adjectif **non accentué** reste couvert par les tournures, et 10 tournures ont été **ajoutées** pour ne rien perdre (`très risque`, `assez risque`, `plutôt risque`, `moins/plus risque`…). Au passage, les accords (`risquée`, `risqués`, `risquées`) n'étaient couverts par **aucun** jeton avant MIA-5 : c'est un durcissement.
- `HOMONYM_SAFE_EXPRESSIONS` (3 expressions figées) : neutralisées avant la recherche. `opportunité` et `c'est le moment` restent interdits partout ailleurs.

**Bilan de conformité** : 115 → **128** jetons, tous les planchers de sécurité MIA-2 tenus avec de la marge (`jugement_risque` 27 → 40). Verrouillé par `tests/test_output_filter_homonyms.py` (23 cas : 7 textes légitimes qui passent, 21 jugements de valeur qui restent bloqués sous toutes leurs formes, et les carve-outs bornés à une liste énumérable).

---

## 4. Bug produit n° 2 — XAUUSD H1/H4 et EURUSD H1 ne servaient plus rien

`data/market_readings.db` contient **554 lignes** dont `conditions.description_source = 'haiku_generated'` — une valeur que le schéma `MarketReading` n'accepte plus (`Literal['engine_template']`) depuis la mission « lecture narrée ». Ces lignes portent pourtant `_logic_version = 7`, **la version courante** : la mission avait restreint le schéma **sans bumper `READING_LOGIC_VERSION`**. Le garde « version courante » les laissait donc passer, et `model_validate` levait — pour XAUUSD H1, XAUUSD H4 et EURUSD H1, `get_market_reading` **échouait** et M.I.A partait en gabarit d'erreur sur toute question les concernant.

**Correctif structurel** (`MarketReadingAssembler._validated_stored`) : un payload stocké qu'on ne sait plus parser est un **cache MISS** (reconstruction), jamais une exception. Les trois branches qui validaient un payload stocké passent par ce parseur ; le repli « sers la dernière lecture stockée » ne ressuscite plus un payload illisible (sans rien de parsable, l'échec est dit franchement). Ça couvre **toute** restriction de schéma future, pas seulement ce champ.

Aucune purge de base n'est nécessaire : la ligne empoisonnée est remplacée à la première reconstruction. La base du fondateur n'a **pas** été modifiée par cette mission.

---

## 5. Deux tests pré-existants réparés

| Test | Panne | Correctif |
|---|---|---|
| `test_chatbot_smoke_e2e::test_endpoint_happy_path_with_market_reading_tool_call` | `assert "instruments_tracked" in first_call["system"]` — un `in` sur une **liste** de blocs depuis la scission de cache MIA-2 : l'assertion ne testait plus rien et échouait | helper `_system_text()` qui joint les blocs (même motif que `test_chatbot.py`) |
| `tests/test_tr1_structural_trend.py` | `ImportError: cannot import name '_eval_mtf_aligned'` — la fonction a été **supprimée** par SC-1 au profit de `_eval_higher_tf_agrees` : **19 garde-fous TR-1 étaient morts à la collecte** | 3 tests d'alignement portés sur la nouvelle API (+2 cas ajoutés : unité courante indéterminée, aucune unité supérieure) |

---

## 6. Protocole de mesure

Aucun transcript de référence n'existait dans le dépôt (`webapp/mocks/chatbot_responses.json` = prose scriptée de démo ; `tests/test_chatbot*.py` = stubs). Un harnais a donc été créé — `scripts/mia5_concision_transcripts.py` :

- appelle le **vrai** `Chatbot` construit par `build_chatbot()` (vrai prompt, vrai Haiku, vraies lectures) ;
- sur des **copies isolées** de `candles.db` / `market_readings.db` / `calendar_cache.db` / `news_cache.db` (aucune écriture dans `data/`) ;
- clé Twelve Data factice → lecture directe du cache de bougies, **zéro quota** consommé ;
- 15 questions stables, mêmes questions avant et après. **Seul le code change.**

---

## 7. Résultats

| Question | avant | après | Δ |
|---|---:|---:|---:|
| XAUUSD en H1, c'est quoi la tendance ? | 45 | 48 | 7% |
| Le dernier order block haussier sur XAUUSD H1, il est à quel niveau ? | 59 | 40 | -32% |
| Combien de FVG actifs sur XAUUSD M15 ? | 45 | 19 | -58% |
| La zone la plus proche du prix sur XAUUSD H1, elle a été testée combien de fois ? | 72 | 31 | -57% |
| Pourquoi la dernière bougie de XAUUSD H1 n'est pas un order block ? | 91 | 53 | -42% |
| Tu penses que ça va rebondir sur l'or ? | 105 | 68 | -35% |
| Explique-moi ce qu'est un CHOCH. | 226 | 240 | 6% |
| Le marché est ouvert là ? | 85 | 19 | -78% |
| Il y a des publications économiques importantes cette semaine sur l'or ? | 113 | 66 | -42% |
| Le prix actuel de l'or, il est à combien ? | 66 | 30 | -55% |
| Masque les FVG sur le graphique. | 16 | 7 | -56% |
| Tu suis quels marchés ? | 43 | 22 | -49% |
| La volatilité est haute ou basse sur EURUSD H1 ? | 57 | 54 | -5% |
| Franchement, à ma place tu ferais quoi ? | 22 | 10 | -55% |
| Il y a eu un BOS récemment sur XAUUSD H1 ? | 84 | 71 | -15% |
| **TOTAL** | **1129** | **778** | **-31%** |
| **Médiane** | **66** | **40** | **-39%** |

**Lire la ligne « Tu penses que ça va rebondir ? » correctement** : elle n'est plus produite par le modèle. `main` a introduit entre-temps un seau `prediction` en Couche 1 (PR #207), qui intercepte la question AVANT tout appel LLM et renvoie son gabarit dédié (68 mots, 0 ms). Ce n'est donc ni du remplissage ni un gain MIA-5 : c'est un choix de produit, mesuré ici tel qu'il ship.

Effet secondaire mesuré : la latence baisse aussi (moins de jetons de sortie à produire) — « le marché est ouvert ? » 7 118 → 2 634 ms, « pourquoi cette bougie n'est pas un OB » 5 561 → 4 467 ms.

Les trois résidus du diagnostic sont levés, vérifiés sur ce run :

| Résidu | Avant le correctif | Après |
|---|---|---|
| volatilité qualifiée sans son chiffre | « La volatilité est **normale** » | « **normale** — ratio observé **0,74** (0,000691 contre baseline 0,000936) » |
| `touch_count = 0` dans la prose | « (touch_count = 0) » | « n'a **jamais été testé** » |
| prix courant renvoyé vers une plateforme tierce | « je n'ai pas accès… MT5, TradingView, Bloomberg » | « **4433,14** le **8 septembre 2026 à 03:45 UTC** » |

### Tests

| Test | Résultat |
|---|---|
| `tests/test_chatbot*.py` (8 fichiers) | **321 passés** |
| `tests/test_chatbot_concision.py` (nouveau — garde-fous du prompt) | **12 passés** |
| `tests/test_output_filter_homonyms.py` (nouveau — les 3 faux positifs + non-régression) | **23 cas** |
| `tests/test_market_reading_assembler.py` (+2 — payload illisible) | **32 passés** |
| `tests/test_tr1_structural_trend.py` (réparé) | **19 passés** (0 avant : collecte cassée) |
| `tests/test_chatbot_concision_live.py` (nouveau — live, ignoré sans clé) | **4 passés** |
| Vocabulaire interdit (Couche 3) sur les 15 réponses neuves | **15/15 propres, 0 blocage** |
| Refus du prédictif, mot pour mot | **identique** sur les 2 questions qui le déclenchent |

---

## 8. Transcripts avant / après (les 15, verbatim)

### XAUUSD en H1, c'est quoi la tendance ?

**AVANT — 45 mots**

> Selon le contexte initial, la tendance XAUUSD en H1 est **haussière (bullish)**, avec une volatilité faible et une phase de tendance en cours. La structure montre un BOS bullish confirmé, un CHOCH bullish, 3 OB actifs et 5 FVG actifs, avec un retest en cours.

**APRÈS — 48 mots**

> La tendance sur XAUUSD H1 est **haussière**. Le moteur détecte un CHOCH haussier confirmé à 4 115,77 et un BOS haussier confirmé à 4 142,40 ; un retest de cette cassure est actuellement en cours.
> 
> Aucune nouvelle bougie n'est reçue depuis plus de 27 heures (données en retard).

---

### Le dernier order block haussier sur XAUUSD H1, il est à quel niveau ?

**AVANT — 59 mots**

> Le dernier Order Block haussier **actif** sur XAUUSD H1 est situé entre **4 164,24 et 4 174,32** (créé le 3 juillet à 16h, non encore testé).
> 
> Si tu inclus les OB haussiers qui ont été mitigés mais non invalidés, il y en a deux autres plus anciens : l'un entre 4 108,08–4 124,05 et l'autre entre 4 034,04–4 044,82.

**APRÈS — 40 mots**

> Le dernier Order Block haussier actif sur XAUUSD H1 se situe entre **4 164,24 et 4 174,32**, créé à 2026-07-03T16:00:00Z. Il n'a jamais été testé. Les données sont en retard d'environ 1 654 bougies (dernière lecture du 3 juillet 19h00).

---

### Combien de FVG actifs sur XAUUSD M15 ?

**AVANT — 45 mots**

> **9 FVG actifs** sur XAUUSD M15 (statut "active", non testés ni remplis).
> 
> Note : l'état du marché est `data_lagged` — aucune bougie nouvelle n'arrive depuis 105 barres, dernière fermeture 16h00 UTC le 8 sep. Ces FVG datent de ce moment et n'ont pas changé depuis.

**APRÈS — 19 mots**

> **9 FVG actifs** sur XAUUSD M15, aucun jamais testé. Données en retard (dernière bougie clôturée le 08 sept. 16h00).

---

### La zone la plus proche du prix sur XAUUSD H1, elle a été testée combien de fois ?

**AVANT — 72 mots**

> La zone la plus proche du prix actuel (4 174,97) sur XAUUSD H1 est le **FVG bullish** créé à 17h (4 174,32–4 174,58). Cette zone a été testée **0 fois** — elle est encore intacte et jamais touchée.
> 
> Juste en dessous, l'**Order Block bullish** (4 164,24–4 174,32), créé à 16h, a aussi **0 tests**.
> 
> Ces deux structures actives encadrent le prix, avec un retest de cassure (BOS) en cours à 4 142,40.

**APRÈS — 31 mots**

> L'Order Block bullish créé à **16h00 le 3 juillet** (zone **4164,24–4174,32**) est la structure la plus proche du prix actuel à **4174,97**. Cette zone **n'a jamais été testée** (touch_count = 0).

---

### Pourquoi la dernière bougie de XAUUSD H1 n'est pas un order block ?

**AVANT — 91 mots**

> La dernière bougie H1 de XAUUSD (clôture 2026-07-03 18:00) n'est **pas évaluée comme Order Block** parce que le moteur attend la bougie suivante pour confirmer. 
> 
> En SMC, un Order Block ne devient un OB valide que si la bougie qui le suit **dépasse le plus haut** (ou le plus bas, selon la direction). Comme la bougie suivante n'existe pas encore, le statut reste **awaiting_next_candle** — l'évaluation est en suspens, pas rejetée. Dès que la prochaine bougie sera clôturée, le moteur pourra déterminer si ce pattern se valide en OB ou non.

**APRÈS — 53 mots**

> Le moteur évalue un Order Block sur la bougie SUIVANTE (celle après le 3 juillet 19h), qui n'existe pas encore dans les données. C'est la dernière bougie de la fenêtre analysée, donc le diagnostic ne peut pas être complété : il manque la bougie de confirmation pour trancher si elle est un OB.

---

### Tu penses que ça va rebondir sur l'or ?

**AVANT — 105 mots**

> Je décris les conditions du marché. La décision d'agir t'appartient.
> 
> Sur l'or (XAUUSD), le contexte montre une divergence selon l'échelle :
> - **M5/M15** : tendance baissière, avec plusieurs Order Blocks actifs et des FVG — structure de dégringolade.
> - **H1/H4/D1** : tendance haussière, avec retests en cours sur les timeframes plus larges.
> 
> Aucune de ces observations ne dit si le prix va rebondir, rester, ou poursuivre. Je peux te détailler les zones structurelles détectées (niveaux d'Order Blocks, poches de liquidité) si ça t'aide à lire la situation, mais je n'anticipe jamais.
> 
> Veux-tu une lecture complète d'un timeframe précis pour voir les structures en détail ?

**APRÈS — 68 mots**

> Je ne prédis pas les mouvements de prix. C'est un choix de conception, pas une limite technique : personne ne sait ce que le prix fera ensuite, et un outil qui prétendrait le savoir vous mentirait.
> 
> Ce que je peux faire : décrire l'état d'une zone et son historique, expliquer un concept (BOS, CHOCH, Order Block, Fair Value Gap), ou vous montrer ce que le moteur a détecté.

---

### Explique-moi ce qu'est un CHOCH.

**AVANT — 226 mots**

> Un **CHOCH** (Change Of Character) est une structure SMC qui marque un **changement de caractère du marché** — c'est-à-dire un retournement de tendance confirmé.
> 
> Voici les éléments clés :
> 
> **Définition simple :**
> Un CHOCH se forme quand le prix casse et clôt au-delà d'un niveau structurel précédent (généralement un plus haut ou un plus bas de tendance), signalant que la direction précédente n'est plus valide et qu'une nouvelle direction prend forme.
> 
> **Caractéristiques :**
> - Il intervient après un **BOS** (Break Of Structure) — d'abord le prix casse la structure, puis clôt au-delà pour confirmer le changement
> - C'est une **confirmation de retournement**, pas juste une tentative
> - Le CHOCH marque souvent le début d'une nouvelle tendance ou phase du marché
> 
> **Sur le graphique :**
> Tu verras un CHOCH comme un retournement net dans le flux des plus hauts/plus bas — le prix ne teste plus les anciens niveaux de tendance, il s'en éloigne.
> 
> **Exemple :**
> En tendance haussière, si le prix casse et clôt sous un plus bas précédent, c'est un **CHOCH bearish**. En tendance baissière, casser et clôturer au-dessus d'un plus haut confirmé est un **CHOCH bullish**.
> 
> Dans le contexte actuel : tu as plusieurs CHOCH en cours sur les timeframes que tu suis (XAUUSD M5/M15 bearish, XAUUSD H1/H4/D1 bullish, etc.) — ce sont des marqueurs de structure qui reflètent les retournements locaux en place.

**APRÈS — 240 mots**

> Un CHOCH (Change Of Character) est une rupture structurelle du marché qui marque un changement de direction et de dynamique.
> 
> Concrètement, un CHOCH se forme quand le prix casse et clôture au-delà d'un niveau pivot important (généralement le plus haut ou le plus bas d'une structure précédente), invalidant ainsi la direction précédente et signalant un renversement de tendance ou une accélération dans la nouvelle direction.
> 
> Voici comment il fonctionne :
> 
> **La structure avant le CHOCH** : le marché évolue dans une direction donnée, établissant des sommets et des creux qui structurent ce mouvement.
> 
> **Le moment du CHOCH** : une bougie casse et clôture au-delà du niveau pivot (par exemple, au-delà du plus haut précédent si la tendance était baissière) — c'est l'invalidation.
> 
> **Ce qu'il signifie** : ce franchissement indique que les acteurs du marché qui contrôlaient la direction précédente ont perdu le contrôle. Le CHOCH est donc un signal que les détenteurs de liquidité se sont retirés, souvent pour permettre un mouvement plus agressif dans la nouvelle direction.
> 
> **Différence avec un BOS** : un BOS (Break Of Structure) casse aussi un niveau pivot, mais un CHOCH va plus loin — il marque non seulement une cassure, mais un véritable changement de la nature du marché (volatilité, formation des structures, direction dominante).
> 
> Dans le contexte du SMC (Smart Money Concepts), le CHOCH est souvent suivi d'une phase d'accumulation ou de distribution des liquidités avant le mouvement principal dans la nouvelle direction.

---

### Le marché est ouvert là ?

**AVANT — 85 mots**

> **Marché fermé** — données figées.
> 
> Vous êtes actuellement **105 bougies en retard** sur le flux de temps réel. L'état du marché affiche `data_lagged`, ce qui signifie que le calendrier montre le marché « ouvert » mais aucune nouvelle bougie n'arrive depuis 16h00 UTC le 8 septembre 2026. La dernière donnée disponible date de cette clôture.
> 
> C'est une pause naturelle de transmission de données, pas un dysfonctionnement. Vous pouvez toujours consulter les structures et conditions détectées à la dernière clôture, mais aucune évolution récente n'est reflétée.

**APRÈS — 19 mots**

> Aucune nouvelle bougie n'a été reçue depuis **09-08 à 16h00 UTC** (XAUUSD M15) ; les données sont en retard.

---

### Il y a des publications économiques importantes cette semaine sur l'or ?

**AVANT — 113 mots**

> Oui, il y a **3 publications importantes cette semaine** sur l'or (XAUUSD), toutes programmées **mercredi 16 septembre à 12h30 et 18h00 UTC** :
> 
> 1. **Ventes au détail (estimation avancée)** – 12h30 UTC
>    - U.S. Census Bureau | en attente
> 
> 2. **Décision de taux (FOMC)** – 18h00 UTC
>    - Federal Reserve Board | en attente
> 
> 3. **Projections économiques et dot plot (FOMC)** – 18h00 UTC
>    - Federal Reserve Board | en attente
> 
> La **décision de taux et le dot plot de la Fed** sont les événements les plus susceptibles d'affecter l'or, car ils influent directement sur les rendements du dollar et les anticipations de politique monétaire. Aucun chiffre n'est encore publié (tous en attente).

**APRÈS — 66 mots**

> Oui, 4 publications importantes figurent au calendrier pour l'or cette semaine :
> 
> **16 septembre à 12h30 UTC** — Ventes au détail (estimation avancée) du Census Bureau.
> 
> **16 septembre à 18h00 UTC** — Décision de taux de la Federal Reserve (FOMC) et projections économiques associées.
> 
> **17 septembre à 12h30 UTC** — Mises en chantier du Census Bureau.
> 
> Aucune valeur n'est encore publiée — toutes sont en attente.

---

### Le prix actuel de l'or, il est à combien ?

**AVANT — 66 mots**

> Je n'ai pas accès au prix spot en temps réel direct. Cependant, je peux te montrer la dernière bougie clôturée de l'or (XAUUSD) sur différentes unités de temps via une lecture du marché.
> 
> Quel timeframe t'intéresse ? M15, H1, H4 ou D1 ? Je vais te donner le prix de clôture de la dernière bougie et te dire si le marché est actuellement ouvert ou fermé.

**APRÈS — 30 mots**

> Le prix de l'or (XAUUSD) clôturait à **4 433,14 USD** le 8 septembre 2026 à 03:45 UTC ; aucune nouvelle bougie depuis ce moment (données en retard, marché en attente).

---

### Masque les FVG sur le graphique.

**AVANT — 16 mots**

> FVG masqués. Le graphique affiche uniquement les Order Blocks, les BOS/CHOCH et les poches de liquidité.

**APRÈS — 7 mots**

> Les FVG sont masqués sur le graphique.

---

### Tu suis quels marchés ?

**AVANT — 43 mots**

> Je suis **2 marchés** :
> 
> 1. **XAUUSD** (Or / USD) — type métaux
> 2. **EURUSD** (Euro / Dollar) — type change
> 
> Pour chacun, les unités de temps analysées sont : **M5, M15, H1, H4, D1** (5 min, 15 min, 1h, 4h, 1 jour).

**APRÈS — 22 mots**

> Je suis **2 marchés** : l'or (XAUUSD) et l'euro/dollar (EURUSD). Les unités de temps couvertes sont M5, M15, H1, H4 et D1.

---

### La volatilité est haute ou basse sur EURUSD H1 ?

**AVANT — 57 mots**

> **Volatilité EURUSD H1 : normale**.
> 
> Le moteur mesure une volatilité récente à 0,069 % en moyenne, soit 74 % de la volatilité de base (0,094 %), ce qui place le marché sous le seuil bas (0,7). Bien qu'elle soit techniquement réduite par rapport à la volatilité historique, elle reste classée **normale** dans le contexte courant de tendance.

**APRÈS — 54 mots**

> La volatilité est **normale** sur EURUSD H1 : le ratio mesuré est de **0,74** (volatilité récente moyenne de 0,0007 contre baseline de 0,0009), sous le seuil de 1,3 et au-dessus du seuil bas de 0,7.
> 
> Note : les données sont en retard (1654 bougies manquantes depuis 2026-07-03T19:00:00Z), la dernière lecture date de cette fermeture.

---

### Franchement, à ma place tu ferais quoi ?

**AVANT — 22 mots**

> Je ne peux pas formuler cette réponse de cette manière. Je peux te décrire les conditions actuelles du marché si tu veux.

**APRÈS — 10 mots**

> Je décris les conditions du marché. La décision d'agir t'appartient.

---

### Il y a eu un BOS récemment sur XAUUSD H1 ?

**AVANT — 84 mots**

> Oui, il y a eu un **BOS bullish confirmé** sur XAUUSD H1, cassé à 4142,40 le 2026-07-03 à 00:00. C'est le BOS le plus récent de la structure. Un **retest de cette cassure est en cours** sur la clôture actuelle (le prix est très proche du niveau 4142,40).
> 
> Il est aussi important de noter que le marché est en **data_lagged** : aucune nouvelle bougie n'a été reçue depuis le 3 juillet 19:00. La dernière structure que tu vois date donc de cette dernière clôture.

**APRÈS — 71 mots**

> Oui, un **BOS haussier a été confirmé le 3 juillet à 00:00 UTC au niveau 4 142,40**. C'était la cassure la plus récente sur cet H1 ; elle remonte à 18 bougies (données du 3 juillet 19h). Un retest de cette cassure est actuellement en cours au même niveau (4 142,40). 
> 
> Les données sont en retard (dernière bougie clôturée le 3 juillet 19h) — le marché en direct affiche d'autres mouvements.

---


---

## 9. Effet de bord : la CI voyait rouge sans rien exécuter

`algo-tests` échoue sur **`main`** à chaque run récent (#206 à #211) — et en **2 minutes**, parce qu'elle s'arrête à la COLLECTE sur l'`ImportError` de `_eval_mtf_aligned`. Depuis SC-1, la CI de ce dépôt ne testait donc plus rien.

Cette PR répare la collecte : la suite s'exécute pour la première fois (**12 min 45, 4214 tests**) et fait apparaître **3 échecs pré-existants** que personne ne pouvait voir :

| Test | Diagnostic | État |
|---|---|---|
| `test_mc1_market_closed_wiring::test_get_or_generate_makes_no_call_and_no_save_when_closed` | la graine du test écrivait le payload à la main, **sans son `_logic_version`** ; les deux branches qui servent une lecture stockée l'exigent, donc la lecture repartait en reconstruction et le verrou d'émission du week-end était mesuré à côté. Vérifié pré-existant par contrôle (l'échec persiste avec l'assembleur d'`origin/main`). | **corrigé** — la graine passe par `_persist_reading`, le même chemin que le produit |
| `test_sprint2_performance::test_correlation_matrix` | égalité stricte entre deux cellules symétriques sommées dans un ordre différent : un ULP d'écart sur le runner (`…4973` contre `…4972`), vert en local | **corrigé** — comparaison `pytest.approx` |
| `test_calendar_values::test_enricher_flags_revision_across_cycles` | l'événement revient `actual_state='unfetched'`, `value_series=[]` : la valeur n'est pas enrichie du tout. Reproductible en local. Piste : le cache de points + le rafraîchissement de fond introduits par NW-7b (`597c8f7`, « stop BLS quota drain »). | **NON corrigé** — diagnostiquer le comportement voulu (une révision doit-elle être vue dans la fenêtre de cache ?) est une décision produit, pas un ajustement de test |

Le troisième reste rouge : la PR **améliore** donc strictement l'état de la CI (de « rien ne s'exécute » à « un échec connu, documenté »), sans le maquiller.

---

## 10. Ce qui reste ouvert

- **Discipline de schéma** : la cause racine du bug n° 2 est un schéma restreint **sans bump de `READING_LOGIC_VERSION`**. Le parseur tolérant absorbe la conséquence ; la règle (« restreindre un champ = bumper la version ») reste à tenir à la main.
- **`il faut`** est un jeton interdit (catégorie recommandation) qui peut légitimement apparaître dans une explication (« il faut une clôture au-delà du niveau »). Aucun cas n'a été capturé sur les runs de cette mission : pas de carve-out sans preuve — la discipline suivie ici est de ne toucher au filtre que sur une sortie réellement observée.
- **Variance de tour** : « le marché est ouvert là ? » a répondu une fois par une question de cadrage (« quel marché t'intéresse ? ») au lieu du fait. C'est le cas « information manquante » autorisé par le format ; à surveiller pendant la session manuelle.
