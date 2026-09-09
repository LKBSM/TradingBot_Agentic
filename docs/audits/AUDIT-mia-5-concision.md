# AUDIT — MIA-5 · Concision des réponses de l'agent M.I.A

- **Branche** : `feat/mia-5-concision` (worktree dédié `wt-mia-5`), depuis **origin/main `adcc475`**.
- **Périmètre touché** : `src/intelligence/chatbot/chatbot.py` — **la seule constante `SYSTEM_PROMPT_STATIC`**. Aucune couche de sécurité, aucun validateur d'ancrage, aucun moteur.
- **Mesure** : 15 questions réelles, vrai `Chatbot`, vrai Haiku, vraies lectures de marché — **1129 → 685 mots (−39 %)**, médiane **66 → 39 mots**.
- **Asymétrie tenue** : « Explique-moi ce qu'est un CHOCH » reste développée (226 → 237 mots).

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

## 2. Ce qui a été changé

### (a) La puce de concision (`SYSTEM_PROMPT_STATIC`, RÈGLES STRICTES)

```diff
-- Tu réponds en français, par défaut concis (2-4 phrases sauf demande explicite de détail).
++ Tu réponds en français, au format défini en fin de message système (FORMAT DE RÉPONSE) : le fait demandé, rien de plus.
```

### (b) Une section `FORMAT DE RÉPONSE` **à la toute fin** du bloc statique

Position volontaire : récence maximale, donc **après** toutes les règles d'ancrage. C'est ce qui porte l'effet mesuré ; un garde-fou (`tests/test_chatbot_concision.py`) interdit de la remonter dans le prompt.

```
FORMAT DE RÉPONSE (vaut pour toutes tes réponses) :
- Ta première phrase contient la réponse. Aucun préambule, aucune reformulation de la question, aucun rappel de ce que l'utilisateur vient de dire.
- Question factuelle précise (un niveau, un compte, une date, un statut, un état) : 1 à 2 phrases, et rien d'autre. Tu n'ajoutes AUCUN élément que l'utilisateur n'a pas demandé : ni une autre unité de temps, ni une autre structure, ni un autre marché, ni la ventilation d'un compte, ni un rappel de la tendance.
- Ta DERNIÈRE phrase est un fait, jamais une question ni une offre de service. Interdits en fin de message : « veux-tu que… », « dis-moi si… », « je peux aussi… », « n'hésite pas… », « quelle unité de temps préfères-tu ? ». Tu ne poses une question que lorsqu'il te MANQUE réellement une information pour répondre (bougie non identifiée, combinaison non précisée) — et alors elle remplace la réponse, elle ne s'y ajoute pas.
- Après une phrase de refus (demande d'action, de conseil, ou d'anticipation), tu t'ARRÊTES. Tu n'enchaînes ni sur une description du marché qu'on ne t'a pas demandée, ni sur une offre d'approfondir. Le refus se suffit à lui-même.
- Question qui demande une explication (« explique-moi… », « c'est quoi… », « comment… », « pourquoi… ») : développe autant qu'il faut. La concision porte sur le remplissage, jamais sur le fond demandé.
- Tu ne cites pas tes rouages : ni le nom d'un outil, ni « le contexte initial », ni un nom de champ technique. Tu écris « jamais testé », pas « touch_count = 0 » ; « valeur pas encore publiée », pas « actual_state=pending ».
- L'état des données (marché fermé, pause quotidienne, données en retard) se dit en UNE clause courte accolée au fait, jamais en paragraphe séparé.
- Mise en forme sobre : le gras sur les valeurs qui répondent à la question, rien d'autre. Pas de liste à puces pour une réponse d'une ou deux phrases.
- Être concis ne retire JAMAIS un fait : niveaux exacts, horodatages, statut d'une zone, état du marché, absence d'une donnée et refus se disent en toutes lettres, en court. Une phrase vraie et courte, jamais une phrase courte et vague.
```

### Ce qui n'a **pas** bougé

`adversarial_filter.py` (Couche 1), `output_filter.py` (Couche 3), `view_action_filter.py` (Couche 4), `constants.py` (35 motifs adverses, 115 jetons interdits, tous les gabarits), l'ancrage par les outils (`_execute_tool`, `_harvest_zone_ids`, verrous `found:false`), et **toutes** les autres sections du prompt (CONTRÔLE DE L'AFFICHAGE, DIAGNOSTIC ORDER BLOCK, ÉTAT DU MARCHÉ, CALENDRIER & PUBLICATIONS, CATALOGUE DES MARCHÉS, RÈGLE D'ABSENCE, liste des 7 outils) : **0 ligne**.

La phrase de refus reste **mot pour mot** : « Je décris les conditions du marché. La décision d'agir t'appartient. »

---

## 3. Protocole de mesure

Aucun transcript de référence n'existait dans le dépôt (`webapp/mocks/chatbot_responses.json` = prose scriptée de démo ; `tests/test_chatbot*.py` = stubs). Un harnais a donc été créé :

- appelle le **vrai** `Chatbot` construit par `build_chatbot()` (vrai prompt, vrai Haiku, vraies lectures) ;
- sur des **copies isolées** de `candles.db` / `market_readings.db` / `calendar_cache.db` / `news_cache.db` (aucune écriture dans `data/`) ;
- clé Twelve Data factice → lecture directe du cache de bougies, **zéro quota** consommé ;
- 15 questions, mêmes questions avant et après, cache d'assembleur chaud dans les deux cas. **Seul le prompt change.**

---

## 4. Résultats

| Question | avant | après | Δ |
|---|---:|---:|---:|
| XAUUSD en H1, c'est quoi la tendance ? | 45 | 33 | -27% |
| Le dernier order block haussier sur XAUUSD H1, il est à quel niveau ? | 59 | 52 | -12% |
| Combien de FVG actifs sur XAUUSD M15 ? | 45 | 22 | -51% |
| La zone la plus proche du prix sur XAUUSD H1, elle a été testée combien de fois ? | 72 | 43 | -40% |
| Pourquoi la dernière bougie de XAUUSD H1 n'est pas un order block ? | 91 | 41 | -55% |
| Tu penses que ça va rebondir sur l'or ? | 105 | 10 | -90% |
| Explique-moi ce qu'est un CHOCH. | 226 | 237 | 5% |
| Le marché est ouvert là ? | 85 | 48 | -44% |
| Il y a des publications économiques importantes cette semaine sur l'or ? | 113 | 56 | -50% |
| Le prix actuel de l'or, il est à combien ? | 66 | 39 | -41% |
| Masque les FVG sur le graphique. | 16 | 4 | -75% |
| Tu suis quels marchés ? | 43 | 21 | -51% |
| La volatilité est haute ou basse sur EURUSD H1 ? | 57 | 19 | -67% |
| Franchement, à ma place tu ferais quoi ? | 22 | 10 | -55% |
| Il y a eu un BOS récemment sur XAUUSD H1 ? | 84 | 50 | -40% |
| **TOTAL** | **1129** | **685** | **-39%** |
| **Médiane** | **66** | **39** | **-41%** |

Effet secondaire mesuré : la latence baisse aussi (moins de jetons de sortie à produire) — p. ex. « pourquoi cette bougie n'est pas un OB » 5 561 → 3 444 ms, « le marché est ouvert ? » 7 118 → 3 109 ms.

### Tests

| Test | Résultat |
|---|---|
| `tests/test_chatbot.py`, `test_chatbot_latency.py`, `test_chatbot_constants.py`, `test_chatbot_adversarial_stress.py`, `test_chatbot_output_stress.py`, `test_chatbot_view_actions.py`, `test_chatbot_streaming.py`, `test_chatbot_endpoint.py` | **321 passés** |
| `tests/test_chatbot_concision.py` (nouveau — garde-fous du prompt) | **10 passés** |
| `tests/test_chatbot_concision_live.py` (nouveau — non-régression live, ignoré sans clé) | **4 passés** dans l'environnement de mesure |
| Vocabulaire interdit (Couche 3) sur les 15 réponses neuves | **15/15 propres** |
| Refus du prédictif, mot pour mot | **identique** sur les 2 questions qui le déclenchent |
| Sélection `-k "chatbot or prompt"` | 406 passés, 5 ignorés, **1 échec pré-existant** |

Échecs **pré-existants** (vérifiés par `git stash` sur origin/main, sans rapport avec MIA-5) :
- `test_chatbot_smoke_e2e.py::test_endpoint_happy_path_with_market_reading_tool_call` — assertion `"instruments_tracked" in first_call["system"]` faite sur une **liste** de blocs depuis la scission de cache MIA-2.
- `tests/test_tr1_structural_trend.py` — `ImportError: cannot import name '_eval_mtf_aligned'` (collecte).

---

## 5. Transcripts avant / après (les 15, verbatim)

### XAUUSD en H1, c'est quoi la tendance ?

**AVANT — 45 mots**

> Selon le contexte initial, la tendance XAUUSD en H1 est **haussière (bullish)**, avec une volatilité faible et une phase de tendance en cours. La structure montre un BOS bullish confirmé, un CHOCH bullish, 3 OB actifs et 5 FVG actifs, avec un retest en cours.

**APRÈS — 33 mots**

> La tendance est **haussière** en H1 (données jusqu'au 3 juillet 19h, en retard depuis). Un retest de BOS haussier (4142,40) est en cours, avec volatilité basse et alignement bullish sur H4 et D1.

---

### Le dernier order block haussier sur XAUUSD H1, il est à quel niveau ?

**AVANT — 59 mots**

> Le dernier Order Block haussier **actif** sur XAUUSD H1 est situé entre **4 164,24 et 4 174,32** (créé le 3 juillet à 16h, non encore testé).
> 
> Si tu inclus les OB haussiers qui ont été mitigés mais non invalidés, il y en a deux autres plus anciens : l'un entre 4 108,08–4 124,05 et l'autre entre 4 034,04–4 044,82.

**APRÈS — 52 mots**

> L'Order Block haussier le plus récent sur XAUUSD H1 est l'**OB_bullish_20260703160000**, situé entre **4164,24 et 4174,32** (créé le 3 juillet à 16:00 UTC), actuellement actif et jamais testé. Il y a aussi un FVG haussier immédiatement adjacent plus haut, entre 4174,32 et 4174,58 (données en retard depuis le 3 juillet à 19h).

---

### Combien de FVG actifs sur XAUUSD M15 ?

**AVANT — 45 mots**

> **9 FVG actifs** sur XAUUSD M15 (statut "active", non testés ni remplis).
> 
> Note : l'état du marché est `data_lagged` — aucune bougie nouvelle n'arrive depuis 105 barres, dernière fermeture 16h00 UTC le 8 sep. Ces FVG datent de ce moment et n'ont pas changé depuis.

**APRÈS — 22 mots**

> **9 FVG actifs** sur XAUUSD M15, mais données en retard (107 bougies en arrière ; dernière clôture le 8 sept 16:00 UTC).

---

### La zone la plus proche du prix sur XAUUSD H1, elle a été testée combien de fois ?

**AVANT — 72 mots**

> La zone la plus proche du prix actuel (4 174,97) sur XAUUSD H1 est le **FVG bullish** créé à 17h (4 174,32–4 174,58). Cette zone a été testée **0 fois** — elle est encore intacte et jamais touchée.
> 
> Juste en dessous, l'**Order Block bullish** (4 164,24–4 174,32), créé à 16h, a aussi **0 tests**.
> 
> Ces deux structures actives encadrent le prix, avec un retest de cassure (BOS) en cours à 4 142,40.

**APRÈS — 43 mots**

> La zone la plus proche du prix (**4 174,97**) est l'Order Block haussier actif (4 164,24–4 174,32), créé à 16h00 le 3 juillet : il n'a **jamais été testé** (touch_count = 0). Données en retard, dernière bougie clôturée à 19h00 le 3 juillet.

---

### Pourquoi la dernière bougie de XAUUSD H1 n'est pas un order block ?

**AVANT — 91 mots**

> La dernière bougie H1 de XAUUSD (clôture 2026-07-03 18:00) n'est **pas évaluée comme Order Block** parce que le moteur attend la bougie suivante pour confirmer. 
> 
> En SMC, un Order Block ne devient un OB valide que si la bougie qui le suit **dépasse le plus haut** (ou le plus bas, selon la direction). Comme la bougie suivante n'existe pas encore, le statut reste **awaiting_next_candle** — l'évaluation est en suspens, pas rejetée. Dès que la prochaine bougie sera clôturée, le moteur pourra déterminer si ce pattern se valide en OB ou non.

**APRÈS — 41 mots**

> Le moteur évalue un Order Block sur la bougie suivante, qui n'existe pas encore. Cette dernière bougie (clôture à **4174.97**) sert de base pour la détection, mais la confirmation requiert la bougie d'après, qu'on n'a pas en direct pour le moment.

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

**APRÈS — 10 mots**

> Je décris les conditions du marché. La décision d'agir t'appartient.

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

**APRÈS — 237 mots**

> Un CHOCH (Change Of Character) est une rupture de structure qui marque un changement du caractère directif du marché.
> 
> **Mécanisme :**
> Le CHOCH se forme quand le prix casse un niveau clé dans une direction contraire à la tendance en place, générant une impulsion qui renverse le contrôle du marché. Contrairement à un simple retest, le CHOCH est une rupture franche qui crée une nouvelle phase directrice.
> 
> **Caractéristiques :**
> - Il succède généralement à un BOS (Break Of Structure) : le BOS établit une première rupture, le CHOCH en confirme l'invalidation ultérieure.
> - Il s'accompagne d'une impulsion significative, souvent plus forte que le mouvement précédent.
> - Il redéfinit les niveaux de contrôle : après un CHOCH bullish, les vendeurs perdent le contrôle au profit des acheteurs ; après un CHOCH bearish, l'inverse se produit.
> 
> **Différence avec le BOS :**
> - **BOS** : rupture du dernier plus haut (ou plus bas) confirmé, marque le début d'un retournement.
> - **CHOCH** : invalidation d'une structure plus ancienne et plus large (souvent un sommet ou creux majeur antérieur), marque le caractère definitif du retournement et l'établissement d'une nouvelle direction.
> 
> **Utilité dans le contexte SMC :**
> Le CHOCH signale une perte de contrôle structurelle et sert souvent d'amorce à une phase tendancielle nouvelle. Il est observé aux différentes unités de temps pour valider des cycles de marché imbriqués — par exemple, un CHOCH H1 peut précéder un retest d'Order Block sur M15.

---

### Le marché est ouvert là ?

**AVANT — 85 mots**

> **Marché fermé** — données figées.
> 
> Vous êtes actuellement **105 bougies en retard** sur le flux de temps réel. L'état du marché affiche `data_lagged`, ce qui signifie que le calendrier montre le marché « ouvert » mais aucune nouvelle bougie n'arrive depuis 16h00 UTC le 8 septembre 2026. La dernière donnée disponible date de cette clôture.
> 
> C'est une pause naturelle de transmission de données, pas un dysfonctionnement. Vous pouvez toujours consulter les structures et conditions détectées à la dernière clôture, mais aucune évolution récente n'est reflétée.

**APRÈS — 48 mots**

> Aucune nouvelle bougie n'arrive depuis le **8 septembre 2026** — les données sont en retard. Les dernières lectures datent du M5 (03:45 UTC) et M15 (16:00 UTC) ce jour-là, tandis que l'H1 et H4 ne montrent aucune activité depuis le **3 juillet 2026**. Le marché ne communique plus.

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

**APRÈS — 56 mots**

> Trois publications importantes sont prévues sur l'or cette semaine. Mercredi 16 septembre :
> 
> - **Ventes au détail (estimation avancée)** à 12h30 UTC — Census Bureau
> - **Décision de taux (FOMC)** à 18h00 UTC — Federal Reserve
> - **Projections économiques et dot plot (FOMC)** à 18h00 UTC — Federal Reserve
> 
> Aucune de ces valeurs n'est encore publiée.

---

### Le prix actuel de l'or, il est à combien ?

**AVANT — 66 mots**

> Je n'ai pas accès au prix spot en temps réel direct. Cependant, je peux te montrer la dernière bougie clôturée de l'or (XAUUSD) sur différentes unités de temps via une lecture du marché.
> 
> Quel timeframe t'intéresse ? M15, H1, H4 ou D1 ? Je vais te donner le prix de clôture de la dernière bougie et te dire si le marché est actuellement ouvert ou fermé.

**APRÈS — 39 mots**

> Je n'ai pas accès au prix en temps réel en dehors d'une lecture de marché. Quelle unité de temps te permet de lire le prix que tu cherches — M5, M15, H1, H4 ou D1 sur l'or (XAUUSD) ?

---

### Masque les FVG sur le graphique.

**AVANT — 16 mots**

> FVG masqués. Le graphique affiche uniquement les Order Blocks, les BOS/CHOCH et les poches de liquidité.

**APRÈS — 4 mots**

> Les FVG sont masqués.

---

### Tu suis quels marchés ?

**AVANT — 43 mots**

> Je suis **2 marchés** :
> 
> 1. **XAUUSD** (Or / USD) — type métaux
> 2. **EURUSD** (Euro / Dollar) — type change
> 
> Pour chacun, les unités de temps analysées sont : **M5, M15, H1, H4, D1** (5 min, 15 min, 1h, 4h, 1 jour).

**APRÈS — 21 mots**

> Je suis **deux marchés** : **Or (XAU/USD)** et **Euro/Dollar (EUR/USD)**. Pour chacun, j'analyse les timeframes **M5, M15, H1, H4 et D1**.

---

### La volatilité est haute ou basse sur EURUSD H1 ?

**AVANT — 57 mots**

> **Volatilité EURUSD H1 : normale**.
> 
> Le moteur mesure une volatilité récente à 0,069 % en moyenne, soit 74 % de la volatilité de base (0,094 %), ce qui place le marché sous le seuil bas (0,7). Bien qu'elle soit techniquement réduite par rapport à la volatilité historique, elle reste classée **normale** dans le contexte courant de tendance.

**APRÈS — 19 mots**

> La volatilité sur EURUSD H1 est **normale** (marché en pause, dernière bougie clôturée à 19:00 UTC le 3 juillet).

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

**APRÈS — 50 mots**

> Oui, il y a un **BOS haussier confirmé à 4142,40** cassé le **3 juillet à 00:00 UTC** (18 bougies avant la fermeture à 19:00 UTC ce même jour). Actuellement, un retest de cette cassure est en cours au même niveau (4142,40), données en retard depuis la clôture de cette bougie.

---


---

## 6. Résidus assumés

1. **Volatilité** — « avant » donnait le ratio (0,74) et le seuil (0,70) ; « après » répond « normale » à une question binaire (« haute ou basse ? ») sans les chiffres. Conforme à la consigne (rien qui n'ait été demandé), mais c'est le seul cas où un chiffre disparaît. Une question explicite sur le ratio le fait revenir.
2. **`touch_count = 0`** — le modèle l'écrit encore parfois *en plus* de « jamais testé », malgré la consigne sur les rouages. 1 cas sur 15, fait juste, formulation imparfaite.
3. **« Le prix actuel »** — M.I.A n'a pas de flux temps réel et le dit ; la réponse se termine par une question de cadrage (« quelle unité de temps ? »), qui est le cas autorisé « information manquante » et non une offre de service.

## 7. Constats HORS périmètre relevés pendant la mesure

1. **La Couche 3 casse des réponses légitimes.** Sur la question calendrier (« des publications importantes cette semaine ? »), **2 runs sur 3** ont vu une réponse factuelle correcte remplacée par « Je ne peux pas formuler cette réponse de cette manière » — déclencheurs `risque` / `sans risque` (nom commun), puis catégorie `jugement_moment`. C'est le sur-blocage assumé documenté dans `output_filter.py` (« Homonym note ») ; côté client, c'est un refus incompréhensible. **Mission séparée.**
2. **554 lignes périmées dans `data/market_readings.db`** portent `description_source='haiku_generated'`, valeur que le schéma `MarketReading` n'accepte plus (`Literal['engine_template']`). Conséquence observée sur la base réelle : `get_market_reading` **échoue** pour XAUUSD/EURUSD en H1 et H4, et M.I.A part en fail-safe ou en réponse dégradée. Les lignes n'ont été purgées que dans la copie de scratchpad — **la base du fondateur n'a pas été touchée**. **À corriger à part** (migration ou purge du cache).

## 8. Avant merge

Session de test manuel du fondateur sur **au moins 15 questions réelles** (la mission remplace Playwright par cette validation : le changement porte sur la génération, pas sur la mise en page). Le merge se fait sur la validation des transcripts, pas seulement du code.
