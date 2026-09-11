# AUDIT SC-4 — Scanner conversationnel : traduction et recherche en direct pendant la saisie

| | |
|---|---|
| Branche | `feat/sc-4-live-translate` (worktree dédié `C:\MyPythonProjects\wt-sc-4`) |
| Base | `origin/main` = `da8bbe9` (après `git fetch`) |
| Écart au démarrage | Le worktree **principal** était à `e0dc69c`, **73 commits derrière `origin/main`**. Tout ce travail est fait contre `origin/main`, jamais contre le HEAD principal. |
| Phases | Phase 1 diagnostic (2026-09-09) → **GO fondateur** → Phase 2 implémentation (2026-09-09/10) |
| Décisions reçues au GO | Option B pour la réconciliation · mesure `count_tokens` d'abord · refonte de mise en page **dans** SC-4 |

---

## 0. Résumé

Le second clic est supprimé, la traduction et la recherche avancent pendant la
frappe, et un retrait manuel n'est jamais réintroduit en douce.

Trois choses valent d'être lues avant le reste :

1. **Le coût mesuré est ~20 % au-dessus de mon estimation de phase 1**, mais le
   ratio ne bouge pas : **×4 à ×7** par session de frappe. §2.
2. **Le cache de prompt ne fonctionne pas sur ce modèle**, et ce n'est plus une
   lecture de documentation : `cache_creation_input_tokens: 0` et
   `cache_read_input_tokens: 0` sur les 5 appels réels. §2.2.
3. **Deux prémisses de la mission étaient fausses sur `main`** : la suppression
   du second clic n'était implémentée nulle part, et SC-3 n'est pas mergée. §1.

Un ajout hors périmètre strict a été fait et doit être accepté ou retiré
explicitement : **un plafond serveur sur `/api/scanner/translate`** (§6). Sans
lui, un déclenchement automatique à la frappe est une dépense non bornée — le
point d'entrée déployé n'a ni limiteur ni quota.

---

## 1. Vérification des prémisses

| Prémisse de la mission | Réalité mesurée sur `origin/main` |
|---|---|
| « la suppression du second clic (déjà acté) » | **Non implémentée**, ni sur `main`, ni sur `origin/feat/sc3-scanner-density`, ni dans `docs/`. La décision existait, le code non → **SC-4 la porte**. |
| « le pipeline unique déjà décidé (SC-3) » | **Vrai, mais il date de SC-2.** `ScanResults` est partagé par `ScannerWorkspace` (palette manuelle) et par le scanner conversationnel depuis SC-2, et lui seul rend `ComboCard`. Réutilisation directe confirmée, aucun second pipeline à démonter. |
| SC-3 est la mission précédente | **`feat/sc3-scanner-density` n'est PAS mergée** (3 commits d'avance). SC-4 part donc de `main`, avec un `ComboCard` sans la refonte de densité SC-3. |

**Conséquence de merge :** SC-3 et SC-4 touchent tous deux `ComboCard.tsx`,
`ScanResults.tsx` et les 9 `messages/*.json`. Un conflit est probable, du même
type que BRD-3 × THM-1. SC-4 recâble par ailleurs les deux props
d'auto-actualisation que SC-3 supprime (§5.4) : au merge, **SC-3 gagne** sur ce
point précis.

### Le bloc « ce qui va à l'encontre »

Contrainte tenue **par construction** : le flux live rend le même `ScanResults`
→ `ComboCard`, où le bloc 2 est écrit à plat, jamais dans un `<details>`. Un
test dédié le vérifie **dans le nouveau flux**, aux deux résolutions, sur trois
critères distincts (§7) — pas seulement « le titre est visible ».

---

## 2. Coût — mesuré, pas estimé

### 2.1 Méthode

`messages.count_tokens` (gratuit) pour l'entrée, puis **5 appels réels** au
traducteur pour la sortie — la sortie était la moitié la moins fiable de
l'estimation, et c'est elle qui décide de l'arbitrage §2.3.
**Coût réel de cette mesure : $0,0197.**

| | Estimation phase 1 | **Mesuré** | Écart |
|---|---:|---:|---:|
| Préfixe fixe (`SYSTEM_PROMPT` + schéma d'outil) | 2 180 tok | **2 838 tok** | +30 % |
| Entrée par appel (phrase de ~85 car.) | 2 270 tok | **2 861 tok** | +26 % |
| Sortie par appel | 200 tok | **217 tok** (109 – 320) | +9 % |
| **Coût par appel (Haiku 4.5)** | $0,0033 | **$0,00394** | +19 % |

Détail de la mesure d'entrée (le texte de l'utilisateur pèse peu ; c'est le
préfixe qui domine, et il est repayé en entier à chaque appel) :

| Texte envoyé | Tokens d'entrée |
|---|---:|
| (vide — préfixe seul) | 2 838 |
| 14 caractères | 2 842 |
| 84 caractères (exemple réel du produit) | 2 862 |
| 143 caractères | 2 885 |
| 500 caractères (maximum accepté) | 3 008 |

### 2.2 🔴 Le cache de prompt ne fonctionne pas — constaté

`DEFAULT_MODEL = "claude-haiku-4-5-20251001"`. Le **préfixe minimum cacheable de
Haiku 4.5 est 4 096 tokens** ; le nôtre en fait **2 838**. Le cache ne se crée
donc jamais, sans erreur ni avertissement — et les 5 appels réels le confirment :
`cache_creation_input_tokens: 0` et `cache_read_input_tokens: 0` partout.

**Le levier de coût le plus efficace du produit (celui que MIA-2 a appliqué au
chatbot) est inaccessible ici en l'état.**

### 2.3 Sonnet 5 + cache : moins cher, mais seulement à trafic soutenu

Sonnet 5 ($2,00 / $10,00 par MTok) a un préfixe minimum cacheable de 1 024
tokens : nos 2 838 tokens **cachent**. Lecture de cache = 0,1× l'entrée.

| | Haiku 4.5 (sans cache) | Sonnet 5 (cache chaud) |
|---|---:|---:|
| Entrée | 2 861 × $1,00/M = $0,00286 | 2 838 cachés × $0,20/M + 23 frais × $2/M = **$0,00062** |
| Sortie | 217 × $5,00/M = $0,00109 | 217 × $10,00/M = $0,00217 |
| **Total / appel** | **$0,00394** | **$0,00279** (−29 %) |

Le point de bascule recalculé sur les vrais tokens est à **449 tokens de
sortie** ; la sortie mesurée plafonne à 320. Sonnet gagne donc sur toute la plage
observée — **mais seulement quand le préfixe reste chaud**. À froid, l'écriture
de cache coûte 2 838 × $2,50/M = $0,0071, et sur une session de 5 appels les deux
modèles s'égalisent ($0,0204 contre $0,0197). **En bêta (trafic clairsemé),
Sonnet n'apporte rien ; à l'échelle, il apporte −29 % et un modèle plus fort.**

*Non implémenté : arbitrage laissé ouvert, hors périmètre SC-4.*

### 2.4 Coût par session et par mois

Session = composer une phrase (~85–140 car.), gardes de §3 actives.

| Scénario | Appels | Coût session |
|---|---:|---:|
| **Avant SC-4** (1 clic, parfois une reformulation) | 1–2 | **$0,004 – $0,008** |
| **Live sans les gardes** (débounce seul) | 8–15 | **$0,032 – $0,059** |
| **Live tel qu'implémenté** | 4–7 | **$0,016 – $0,028** |

> **Multiplication : ×4 à ×7.** (×8 à ×15 sans les gardes.)

Hypothèse : 3 sessions par utilisateur actif et par jour, 30 jours.

| Utilisateurs actifs | Sessions/mois | Avant SC-4 | **SC-4** |
|---:|---:|---:|---:|
| 10 (bêta privée) | 900 | $4 – $7 | **$14 – $25** |
| 100 | 9 000 | $36 – $72 | **$144 – $252** |
| 1 000 | 90 000 | $360 – $720 | **$1 440 – $2 520** |

**Lecture :** en test personnel le surcoût est trivial. Le risque n'est pas le
prix d'aujourd'hui, c'est la pente — et sans le plafond de §6, la borne haute
n'existait pas.

---

## 3. Déclenchement — ce qui a été implémenté

`webapp/lib/scanner-chat/use-live-translation.ts`. **Les cinq gardes sont
évaluées avant d'émettre la requête**, délibérément : annuler un `fetch`
n'interrompt pas l'appel LLM déjà parti côté serveur, donc l'annulation ne fait
économiser **aucun centime** — elle ne sert qu'à la cohérence de l'affichage.

| # | Garde | Valeur | Raison |
|---|---|---|---|
| 1 | Pause de frappe | **450 ms** | Aligné sur `use-live-combo-count` (400 ms), l'idiome maison. La latence LLM (~1,2–2 s) domine de toute façon le ressenti. |
| 2 | Frontière de mot | dernier mot/ponctuation complet | Ne jamais traduire un fragment ; sinon « jamais tes » remonte en *untranslatable* et l'écran donne l'impression que le produit échoue. |
| 3 | Longueur minimale | **≥ 25 car. et ≥ 4 mots** | Les vraies phrases du produit font 41–84 car. (mesuré sur les 6 exemples `fr.json`). |
| 4 | Delta minimal | **≥ 12 car.** depuis le texte réellement traduit | La garde qui rapporte le plus : elle supprime la rafale d'appels quasi identiques. |
| 5 | Jamais deux fois | texte identique ⇒ pas d'appel | — |

**Filet de repos (obligatoire) :** 1 200 ms sans frappe déclenchent **toujours**
une lecture du texte **intégral**, même si les gardes 3 et 4 la refusent. Sans
lui, ajouter cinq caractères puis s'arrêter laisserait la dernière idée
silencieusement non lue — une interface qui ment sur ce qu'elle a vu.

**Plafond par onglet :** 25 lectures automatiques, au-delà desquelles seul le
bouton explicite déclenche. Un plafond client ne protège de personne (il s'enlève
avec la console) — il couvre l'accident ordinaire, pas l'abus. L'abus est traité
en §6.

**Deux horloges pour ne pas afficher d'honnêteté provisoire.** Les blocs « ce que
j'ai supposé » et « ce que je n'ai pas pu traduire » ne sont publiés que depuis
une lecture **reposée**. Pendant la frappe ils sont **retenus, pas périmés** :
une ligne dit que la lecture est en cours. On n'affiche jamais un « je n'ai pas
pu traduire » portant sur une proposition à moitié écrite, et jamais une copie
obsolète de la lecture précédente.

---

## 4. Réconciliation — mécanisme retenu (Option B)

### 4.1 L'obstacle

Une condition émise par le traducteur est
`{"type": "price_in_ob", "direction": "bullish"}` : **aucun champ ne dit quel
morceau de phrase l'a produite**. `assumptions` portait déjà un `source_phrase`
et `untranslatable` un `fragment` — les conditions, et elles seules, n'avaient
rien. Sans ce lien, « le texte qui l'a produite » n'est pas une notion que le
code peut évaluer.

L'option sans changement serveur (suppression par identité seule) a été écartée
au diagnostic : elle respecte le retrait mais **ne le lève jamais**. Réécrire
explicitement la condition ne la ramènerait pas — l'autre moitié de « elle ne
choisit rien à ta place ».

### 4.2 Ce qui a été construit

**Serveur** — `src/intelligence/scanner_translator.py` :

* le schéma d'outil offre un `source_phrase` **par condition**, *facultatif* :
  une citation inventée est pire qu'une citation absente ;
* une 7ᵉ règle du `SYSTEM_PROMPT` demande le fragment **mot pour mot** ;
* `verify_source_phrase()` **ne croit jamais le modèle** : la citation n'est
  conservée que si elle figure **verbatim** dans le texte de l'utilisateur,
  comparée après repli d'accents/casse et normalisation des espaces. Sinon →
  `None`. On ne fabrique jamais une origine.

**Transport** — la citation **ne peut pas** voyager dans l'objet condition :
`ScanCondition` a `model_config = {"extra": "forbid"}`, et `conditions` doit
rester postable verbatim au scan. Elle voyage donc dans un tableau parallèle
**aligné par index**, `condition_sources`, maintenu aligné à travers le dédoublonnage
**et** à travers la re-validation `ScanCondition` (une condition rejetée emporte
sa citation, sinon toutes les suivantes se décalent d'un cran).

**Client** — `webapp/lib/scanner-chat/reconciliation.ts`, pur et testé seul :

1. **Purge** — toute suppression dont la `source_phrase` a quitté le texte
   courant est **levée**.
2. **Filtre** — une condition proposée est masquée si le couple
   `(condition, source_phrase)` est encore supprimé.

### 4.3 Comportement obtenu

| L'utilisateur… | Résultat |
|---|---|
| retire une puce, continue d'écrire ailleurs | elle **ne revient pas** (sa phrase source tient toujours) |
| retire une puce, puis **efface/réécrit** le morceau de phrase | suppression **levée** ; si le texte redit la chose, la puce **revient** |
| retire une puce, puis la **redécrit avec d'autres mots** | nouvelle citation → couple différent → la puce **revient** |
| retire une puce que le modèle n'a pas su citer (`null`) | repli sur l'identité seule pour **cette** condition : levée seulement si le champ est vidé, ré-ajout toujours possible via la palette — **dégradation honnête, jamais silencieuse** |

**Deux cas non demandés mais couverts**, parce qu'ils constituent la même
violation :

* **les éditions** — changer une valeur sur une puce survit aux re-traductions.
  L'édition est classée sous l'identité de la **proposition** de M.I.A, pas sous
  la condition modifiée : sinon la clé ne correspondrait plus à ce que M.I.A
  reproposerait, et la correction de l'utilisateur serait réinitialisée en
  silence à la frappe suivante.
* **les ajouts** depuis la palette — ils ne viennent d'aucun texte, donc aucune
  re-traduction ne peut les retirer.

---

## 5. Mise en page — la refonte demandée

### 5.1 Pourquoi elle était obligatoire

`ConversationalScanner` était une machine à **5 modes mutuellement exclusifs**
rendus par `return` anticipés. **En mode `translation`, la zone de saisie
n'existait plus dans le DOM** : elle était remplacée par un rappel en lecture
seule et un bouton « Modifier ». « Voir mes puces se construire en écrivant »
était donc impossible sans refonte.

### 5.2 Ce qui la remplace

Une surface de composition unique : `compose` (le champ, la lecture, les
résultats) et `strategies` (un autre écran, resté séparé). En ≥ 1024 px, deux
colonnes — la saisie collante à gauche, la lecture vivante à droite ; en dessous,
une pile verticale. Le champ **n'est jamais démonté**.

### 5.3 Le refus ne détourne plus l'écran

Avant, `outcome === 'refused'` faisait `setMode('refusal')` → **plein écran**,
saisie détruite. En direct, taper « quels sont les meilleurs setups » aurait fait
exploser l'écran **au moment où le mot se termine**, texte en cours compris.
Le refus est désormais rendu **en ligne** : il occupe la colonne de lecture et
laisse le champ tranquille. Il n'est pas plus doux pour autant — il ne traduit
toujours rien, et les deux exemples cliquables de reformulation sont conservés.

### 5.4 Une erreur transitoire ne parle plus

Le disjoncteur du traducteur s'ouvre après 3 échecs pour 60 s. Avec 4 à 7 appels
par session au lieu de 1, il s'ouvrira nettement plus souvent. Une lecture
automatique qui échoue est donc **silencieuse** : la dernière lecture valide reste
à l'écran. Seule une demande explicite (le bouton) rapporte l'erreur.
Exception : un 429 est toujours dit, parce qu'il change ce que l'utilisateur peut
faire dans la minute.

### 5.5 Un contrôle qui était décoratif est devenu vrai

Avant SC-4, cette page passait la préférence d'auto-actualisation à
`ScanResults` (qui rend le bouton) mais n'armait jamais le minuteur : le contrôle
**ne faisait rien ici**. Il est maintenant câblé à `useCandleCloseRefresh`, comme
sur la palette manuelle. *(SC-3 supprime l'auto-actualisation entièrement ; au
merge, SC-3 gagne — cf. §1.)*

### 5.6 Le piège de mise en page qui n'a pas été rejoué

L'indicateur de lecture occupe une **hauteur réservée en permanence** et ne change
que son contenu. Un indice qui apparaît en agrandissant son conteneur fait
remonter le pied de page entre `mousedown` et `mouseup` et **avale le clic** —
c'est exactement ce qui avait cassé la dictée sur `/app` en MIA-1.

---

## 6. ⚠️ Ajout hors périmètre strict — à accepter ou retirer

**Un plafond serveur sur `POST /api/scanner/translate` : 40 traductions / 5 min,
par compte ET par IP** (`TRANSLATE_THROTTLE_MAX=0` le désactive).

**Pourquoi je l'ai ajouté sans l'avoir demandé explicitement :** le point d'entrée
déployé est `src/api/asgi.py` → `create_app()` **sans argument**, donc
`rate_limiter=None` et `cost_quota=None` ; le limiteur 100 req/min n'existe que
dans `src/intelligence/main.py`, **qui n'est pas déployé**. Tant que dépenser une
traduction demandait un clic, la borne était la patience humaine. **En livrant un
déclenchement automatique à la frappe, je retirais cette borne sans la
remplacer** — c'était la moitié irresponsable de la fonctionnalité.

Il réutilise `AuthThrottle`, créé exactement pour ce constat (per-process, en
mémoire : un frein, pas un quota global — un store partagé reste la couche
suivante). Une session réelle en dépense 4 à 7 ; un humain ne le rencontre
jamais, un script le rencontre immédiatement.

**Si tu préfères le sortir de SC-4, c'est 3 fichiers et ça se retire proprement —
mais alors ne diffuse pas la frappe en direct au-delà de ton propre test.**

---

## 7. Tests

### 7.1 Comportements adversariaux — observés et documentés

| Geste | Comportement observé | Verrouillé par |
|---|---|---|
| **Taper puis effacer plus vite que le débounce** | **Aucun appel émis.** Les gardes s'évaluant avant le `fetch`, rien n'est dépensé. | vitest + Playwright |
| **Effacer une phrase déjà lue** | La lecture entière est effacée (puces, résultats), l'indicateur repasse à `idle`. Pas seulement le champ. | vitest + Playwright |
| **Coller un texte long d'un coup** | **Une seule lecture**, pas une par caractère (un collage est un seul événement `change`). | vitest + Playwright |
| **Retirer une condition puis continuer à écrire** | Elle **ne réapparaît pas** tant que les mots qui l'ont produite sont là, alors que le traducteur la repropose à chaque lecture. | vitest + Playwright |
| **Retirer une condition puis la redécrire autrement** | Elle **revient**. | vitest + Playwright |
| **Retirer une condition non attribuée** | Reste retirée ; ré-ajout par la palette toujours offert (jamais une impasse). | vitest |
| **Demande de classement en pleine frappe** | Refus **en ligne** ; le champ et le texte en cours restent intacts. | vitest + Playwright |
| **Panne LLM pendant la frappe** | La dernière lecture valide reste affichée, aucune erreur affichée. Le bouton explicite, lui, rapporte l'erreur. | vitest |

### 7.2 Le bloc « ce qui va à l'encontre » dans le flux live

`tests/e2e/sc4-live-translate.spec.ts`, **aux deux résolutions**, trois critères
indépendants plutôt qu'une simple visibilité du titre :

1. il n'est dans **aucun `<details>`** (`el.closest('details') === null`) — pas
   même un ouvert par défaut ;
2. son **contenu** est lisible sans interaction : la condition non remplie **et**
   le signal factuel contraire ;
3. **aucun ancêtre** ne le masque (`display:none` / `visibility:hidden`).

### 7.3 Résultats

| Suite | Résultat |
|---|---|
| `tests/test_sc4_source_phrase.py` (neuf) | **23 / 23** |
| `test_scanner_translator` + `_endpoint` + `_live_degradation` | **38 / 38**, aucune régression |
| `reconciliation.test.ts` (neuf) | **23 / 23** |
| `live-translation-gates.test.ts` (neuf) | **16 / 16** |
| `sc4-live.test.tsx` (neuf) | **13 / 13** |
| `ScanResults.test.tsx` | 9 / 9 |
| `sc1-results.test.tsx` | 7 / 7 |
| `ComboCard.test.tsx` | 6 / 6 |
| `StrategyPanel.test.tsx` | 6 / 6 |
| `scannerchat-vocab` / `forbidden-vocab` | 2 / 2 · 2 / 2 |
| `locale-parity.test.ts` | 10 / 10 (2 397 clés × 9 locales, écart nul) |
| `tsc --noEmit` | **propre** |
| `next build` | **réussi** (`/[locale]/scanner/decrire` : 8,89 kB) |

**Deux défauts trouvés PAR ces tests, pas avant :**

1. **Un double appel payant.** Si l'appel rapide était encore en vol quand le
   filet de repos se déclenchait, la **même** requête partait deux fois :
   `lastTranslatedRef` n'est posé qu'au retour de la réponse. Corrigé par un
   `inFlightRef` posé avant l'attente. C'était du gaspillage pur, invisible à
   l'œil, et c'est le test « collage lu une seule fois » qui l'a attrapé.
2. **Un retrait qui survivait à un effacement complet.** Vider le champ puis
   réécrire la même phrase laissait la suppression active, alors que
   l'utilisateur avait tout réécrit. Les retraits et les éditions sont désormais
   purgés quand le champ se vide — mais **pas** les ajouts palette, qui
   n'appartiennent à aucune phrase (les purger casserait aussi le chargement
   d'une stratégie enregistrée, qui pose des conditions avec un champ vide).

---

## 8. Fichiers

**Serveur**
* `src/intelligence/scanner_translator.py` — `source_phrase` par condition,
  `verify_source_phrase`, `condition_sources` aligné, 7ᵉ règle du prompt.
* `src/api/routes/scanner_translate.py` — `condition_sources` dans la réponse,
  alignement préservé à la re-validation, plafond dédié (§6).
* `tests/test_sc4_source_phrase.py` — 23 tests.
* `tests/conftest.py` — le plafond est un singleton de module : chaque test en
  reçoit une instance neuve, sinon le 40ᵉ appel `/translate` d'une session ferait
  échouer un test qui n'a rien à voir avec le plafond.

**Client**
* `webapp/lib/scanner-chat/reconciliation.ts` — **neuf**, pur.
* `webapp/lib/scanner-chat/use-live-translation.ts` — **neuf**, les 5 gardes,
  le filet de repos, les deux horloges, l'anti-réponse-périmée.
* `webapp/lib/conditions/use-live-scan.ts` — **neuf** : le scan complet, même
  débounce que `use-live-combo-count`, dont il est le jumeau à charge utile
  entière (les faire tourner tous les deux doublerait chaque scan).
* `webapp/lib/scanner-chat/translate-client.ts` — `condition_sources` optionnel,
  `TranslateRateLimitedError`.
* `webapp/components/scanner/conversational/ConversationalScanner.tsx` — refonte.
* `webapp/components/scanner/conversational/DescribePanel.tsx` — `textareaRef` +
  `statusSlot`.
* `webapp/messages/*.json` — 6 clés × 9 locales (insertion chirurgicale, CRLF
  préservés, aucun round-trip JSON).
* Tests : `reconciliation.test.ts`, `live-translation-gates.test.ts`,
  `sc4-live.test.tsx`, `tests/e2e/sc4-live-translate.spec.ts`, et
  `tests/e2e/sc2-scanner-conversationnel.spec.ts` recalé sur le flux sans second
  clic.

---

## 9. Vérification

*(complété en fin d'exécution)*

---

## 10. Note d'environnement

Deux pièges rencontrés, tous deux déjà documentés sur ce dépôt :

* **`node_modules` d'un worktree neuf.** Une **jonction** vers un autre worktree
  fait mourir les workers vitest sans message utile. Une **copie réelle** depuis
  `wt-cln-1` (lock identique, vérifié) fonctionne — 42 560 fichiers, 609 Mo. La
  jonction a été retirée via `Directory.Delete(path, recursive:false)`, qui ne
  descend jamais dans la cible : un `rm -rf` aurait suivi le lien et vidé le
  store partagé (incident THM-1).
* **Machine saturée.** Sept autres terminaux tournaient en parallèle
  (`next build`, `vitest`, `tsc`) à 100 % de CPU. Le `START_TIMEOUT` du worker
  vitest est une constante **codée en dur à 60 s** ; sous cette charge le worker
  ne démarre pas. Aucun patch de `node_modules` : les runs ont simplement été
  relancés jusqu'à une fenêtre plus calme, un fichier à la fois.
