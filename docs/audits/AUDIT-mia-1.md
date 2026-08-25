# AUDIT MIA-1 — M.I.A Agent : latence perçue & place donnée à la conversation

> **Phase actuelle : DIAGNOSTIC (lecture seule). Aucun code modifié. En attente du GO.**

## 0. Position du HEAD (discipline d'audit)

- Worktree dédié créé : `C:\MyPythonProjects\wt-mia-1`, branche `fix/mia-1-agent-latence-espace`.
- Base : `b4f4e75` = **origin/main à jour** (`git rev-list --left-right --count origin/main...HEAD` → `0	0`).
- ⚠️ Le checkout principal (`C:\MyPythonProjects\TradingBOT_Agentic`) était sur `docs/preserve-data-1-audit` **−12 commits derrière origin/main** : diagnostiquer là aurait lu du code périmé (incident DATA-1). Tout ce rapport lit le worktree à jour.
- Coordination : **aucune modification du moteur ni du cache partagé** n'est nécessaire (voir §7) → pas de collision avec PERF-2 / CHART-2.

---

## 1. LE CONSTAT (mots du trader)
1. M.I.A met trop de temps à répondre.
2. Pas assez de place pour la conversation ; les suggestions du bas sont trop grosses et mangent le panneau.

---

## 2. OÙ PART LE TEMPS (A) — découpage par composant

### Chaîne de bout en bout (code réel)
`ChatInput.handleSubmit` → `ChatProvider.askFreeForm` → `askSentinel` (`fetch` **synchrone**, `POST /api/chatbot/message`) → route `chatbot.py` → `Chatbot.chat()` :

1. **Couche 1** — filtre adverse sur le message (regex) — **avant** tout appel LLM.
2. **Couche 2** — boucle tool-use Haiku (`claude-haiku-4-5-20251001`, `max_tokens=1024`, `MAX_TOOL_TURNS=3`), `client.messages.create()` **non-stream**.
3. Exécution des tools (`get_market_reading` / `get_signal_summary` / `get_ob_diagnostic` / `apply_chart_view`).
4. **Couche 3** — filtre de sortie (regex) sur le **texte final complet**.
5. Retour **JSON en un bloc** → `res.json()` → une seule bulle rendue d'un coup.

### Découpage par cas — AVANT

> Les segments dérivés du code (nombre d'allers-retours, cache, validation, rendu) sont **structurellement certains**. Les durées LLM sont **modélisées** (Haiku 4.5, TTFT ~0,4–0,8 s, génération ~2–4 phrases) faute de clé Anthropic live dans cet environnement de diagnostic ; elles seront **mesurées** à l'instrumentation post-GO. L'ordre de grandeur et surtout **la structure des coûts** ne changent pas.

| Segment | Cas 1 — simple, sans outil | Cas 2 — lecture de marché | Cas 3 — zone précise / OB |
|---|---|---|---|
| Signal d'activité perçu | **0 texte** (3 points muets, cf. §3) | idem | idem |
| Couche 1 (filtre adverse, regex) | < 1 ms | < 1 ms | < 1 ms |
| Aller-retour Haiku #1 (réseau + gén.) | 1,3–2,2 s | 0,5–1,0 s (décision d'outil) | 0,5–1,0 s |
| Exécution outil(s) | — | cache-hit 5–50 ms / cache-miss ≤ ~1 s (build local) | idem, ×1–2 outils **en série** |
| Aller-retour Haiku #2 (gros JSON injecté) | — | 1,5–3,0 s | 1,5–3,0 s |
| Aller-retour Haiku #3 (si besoin) | — | — | 0–2,5 s (plafond `MAX_TOOL_TURNS`) |
| Couche 3 (filtre sortie, regex) | < 1 ms | < 1 ms | < 1 ms |
| Réseau proxy `/api` → FastAPI | ~10–30 ms | ~10–30 ms | ~10–30 ms |
| Rendu navigateur (markdown, une bulle) | ~10–30 ms | ~10–30 ms | ~10–30 ms |
| **Premier caractère affiché** | **= réponse complète** (rien avant) | **= réponse complète** | **= réponse complète** |
| **Réponse complète** | **≈ 1,3–2,2 s** | **≈ 3–5 s** (jusqu'à ~7 s à froid) | **≈ 4–7 s** |

### Constats-clés
- **Le coût dominant = allers-retours Haiku en série**, pas les outils. La lecture de marché est déjà protégée par le cache de l'assembleur (clé = bougie clôturée attendue, PERF-1/PERF-2) → un `get_market_reading` répété dans la même fenêtre de bougie est un **cache-hit** (5–50 ms). Le moteur n'est pas le goulot.
- **Validation = microsecondes** (regex). Elle n'est pas le problème de vitesse.
- **Rendu navigateur = négligeable.**

---

## 3. DIFFUSION AU FIL DE L'EAU ? (B) — **NON**

- Frontend : `askSentinel` fait un `fetch` puis `await res.json()` — **réponse complète en un bloc** (commentaire explicite du code : « no streaming »).
- Backend : `client.messages.create()` (pas `.stream()`) → JSON unique.
- Pendant l'attente : `ThinkingIndicator` = **trois points animés muets** ; le seul texte (`chat.thinking`) est `sr-only` (lecteurs d'écran uniquement). **Aucun signal visible ne dit ce qui se passe.**

➡️ **C'est la première cause de l'impression de lenteur** : l'utilisateur fixe un écran immobile 1,3 à 7 s, sans savoir qu'une lecture de marché est en cours.

---

## 4. QUESTION DÉCISIVE (C) — VALIDATION vs DIFFUSION

**Décomposition des couches et de leur fenêtre de vérification :**

| Couche | Ce qu'elle vérifie | Sur quoi | Quand | Diffusable au fil de l'eau ? |
|---|---|---|---|---|
| **Couche 1 — filtre adverse** (`adversarial_filter.py`) | jailbreak, demande de trade, hijack de persona, conseil financier | le **message utilisateur** | **AVANT** toute génération | ✅ déjà en amont — n'entrave rien |
| **Couche 4 — liste blanche / ancrage d'ID** (`view_action_filter.py`) | action ∈ liste blanche ; **aucun id de zone inventé** (uniquement les ids réellement émis par le moteur ce tour, `known_zone_ids`) ; pas de géométrie/prix | les **tool calls** (blocs `tool_use`), pas le texte | **PENDANT** la boucle d'outils | ✅ porte sur des appels structurés, pas sur le texte affiché |
| **Couche 3 — filtre de sortie** (`output_filter.py`) | tokens interdits : verbes d'action, recommandation, jugement de moment/risque | le **texte final produit** | **APRÈS** génération complète ; **remplace tout le message** par un template si contaminé | ❌ **incompatible** avec une diffusion progressive conservée |

### Réponse directe
- **Ce qui est vérifiable AVANT la génération** (déjà le cas) : la forme et l'intention de la requête (Couche 1), et — indépendamment du texte — **le rejet d'un identifiant inventé** se fait sur les *tool calls* (Couche 4), pas sur la prose. Ces deux couches **ne bloquent pas** une diffusion.
- **Ce qui n'est vérifiable qu'APRÈS** : **uniquement** le filtre de tokens interdits sur le langage naturel produit (Couche 3). Il ne peut pas remonter en amont : il inspecte des mots que le modèle n'a pas encore écrits.

➡️ **La tension est réelle et localisée à la Couche 3.** Diffuser le texte du modèle token-par-token, puis découvrir un token interdit en fin de phrase, obligerait à **retirer un message déjà affiché** — exactement ce que la mission interdit (« un texte affiché puis retiré est pire qu'une seconde d'attente »).

### Proposition (ta décision — non appliquée)
On **ne diffuse pas** le texte du modèle. On gagne le temps ailleurs, sans toucher à aucune vérification :
1. **Signal d'activité honnête < 200 ms** : dès l'envoi, une ligne fixe (chaîne système, **pas** du texte LLM → aucune validation requise) — p. ex. « M.I.A réfléchit… ».
2. **Statut d'outil honnête, en SSE** : quand — et seulement quand — un `get_market_reading` part réellement, pousser « Lecture de XAUUSD M15 en cours… » (chaîne fixe, vraie, non-LLM → hors périmètre de validation). L'utilisateur voit l'activité progresser sans qu'aucun texte non validé n'apparaisse.
3. **Le texte final** reste généré puis **validé en entier par la Couche 3** avant d'être poussé en une fois (comportement actuel, intact).
4. Couper les **allers-retours redondants** (§5) et **paralléliser** les outils indépendants (§7) réduit la durée totale — sans affaiblir la validation.

### ⚠️ Tension à arbitrer avec le budget « premier caractère < 1 s »
Le texte de réponse validé ne peut pas apparaître avant la fin de génération (≈ 1,3–2,2 s même sur un cas simple). Deux lectures possibles du budget, **à trancher par toi** :
- **(recommandé)** « premier caractère » = **premier retour visible** → tenu **< 200 ms** via le signal d'activité honnête ; le **premier caractère de la réponse validée** reste borné par la génération (qu'on minimise : moins d'allers-retours, prompt resserré).
- **(refusé par défaut)** diffuser la prose du modèle pour tenir < 1 s sur le texte lui-même → viole la règle « pas de retrait ». Non retenu sauf accord explicite de ta part.

---

## 5. OUTILS RAPPELÉS INUTILEMENT ? (D) — **OUI, un aller-retour LLM redondant**

- L'historique envoyé au backend (`historyForApi`) = **6 derniers tours en texte seul** (`{role, content}`). **Les résultats d'outils n'y sont jamais.**
- Conséquence : une **question de suivi sur la même combinaison** (« et la structure H1 ? » après une 1re lecture) **relance `get_market_reading`** → le modèle ne dispose plus du JSON du tour précédent. Le **moteur** est protégé par le cache (cache-hit), mais on paie **un aller-retour Haiku de plus** (+1,5–3 s) et un **gonflement de tokens d'entrée** à chaque message.
- Les tool calls d'un même tour sont exécutés **en série** (`for block in response.content`) → deux lectures demandées ensemble ne partent pas en parallèle.

➡️ Levier réel : **éviter la relecture quand le contexte du tour précédent suffit**, et **paralléliser** les outils indépendants. **Ces deux corrections vivent dans `chatbot.py` (périmètre MIA), sans toucher `market_reading_assembler.py` (périmètre PERF-2).**

---

## 6. RÉPARTITION DE LA HAUTEUR DU PANNEAU (E) — 1280 × 800

Panneau docké = 3ᵉ colonne de la grille `.app-shell` : **338 px de large × 800 px de haut** (`grid-template-columns: 232px minmax(0,1fr) 338px`, `height: 100dvh`). Composant `AppChatSidebar` (identique en desktop docké **et** en onglet mobile).

> Valeurs **calculées** depuis les classes Tailwind/CSS (assises sur les tailles de police et paddings du code). Les valeurs **mesurées** exactes seront capturées dans les Playwright avant/après (bounding boxes).

| Zone | Composition | Hauteur (px) AVANT |
|---|---|---|
| En-tête (`<header>` `px-4 py-3`, border-b) | avatar md + titre 1 ligne + note pédagogique | **≈ 78** |
| Bloc « combinaison » | **inline dans le titre** (`· XAUUSD · M15`, `truncate`) — ne se réduit pas | (inclus en-tête) |
| Zone conversation (`flex-1 overflow-y-auto px-4 py-4`) | messages | **≈ 537** |
| Pied — `ChatInput` + ligne conformité (`space-y-2 border-t px-4 py-3`) | formulaire ~54 + indice clavier ~23 + **note transcription ~34** + conformité ~45 + paddings | **≈ 185** (≈ 155 sans micro) |
| dont champ de saisie (form) | textarea + micro + envoyer | ~54 |

**Le pied engloutit ~23 % du panneau** (indice clavier + note de transcription vocale + ligne de conformité, tous empilés en permanence). En état vide, la zone conversation affiche en plus un **héros** (avatar lg + titre + sous-titre + **3 suggestions pleine largeur** `px-3.5 py-2.5`, ~142 px).

### Messages tenant à l'écran — AVANT
- Message assistant typique (avatar + label + 2–4 phrases markdown à 306 px de large) ≈ **130–170 px** ; message utilisateur ≈ **50–60 px** ; `gap-4` = 16 px.
- Zone conversation ≈ 537 px ⇒ **≈ 1 échange complet + le début du suivant** ; avec l'indicateur d'attente et une réponse longue, souvent **1 seul échange** visible.
- **AVANT ≈ 2–3 messages** visibles sans défiler → **cible ≥ 4**.

### 390 × 844
Même `AppChatSidebar` (onglet Chat, `h-full`). Le pied ~185 px et l'en-tête ~78 px pèsent proportionnellement **plus lourd** (hauteur utile ~760 px moins la nav) → mêmes leviers.

---

## 7. CYCLE DE VIE DES SUGGESTIONS (F)

- Les **STARTERS** (3, `ChatWelcome`) ne s'affichent que si `turns.length === 0`. **Après le 1er échange, ils disparaissent** (bien) — mais tant qu'ils sont là ils sont **empilés pleine largeur** (blocs), pas compacts.
- `SuggestedQuestions` (post-message, chips consommables) n'est câblé **que dans le slide-over `ChatPanel`**, **pas** dans le panneau docké `/app`.
- Aucun recalcul par message dans le docké ; état vide binaire.

➡️ Travail visé (après GO) : suggestions **compactes 1–2 lignes** en état vide, **repliables/effaçables** après le 1er échange (déjà le cas pour disparaître, à rendre plus léger), et surtout **dégonfler le pied** (fusionner/alléger indice + note transcription + conformité) pour rendre ~100 px à la conversation.

---

## 8. CE QUI NE POURRA PAS ÊTRE ACCÉLÉRÉ SANS TOUCHER À LA VALIDATION

- **Le premier caractère du texte *validé*** : borné par la fin de génération, car la **Couche 3** ne peut vérifier que du texte déjà produit. On ne diffusera pas la prose du modèle.
- **Le rejet d'un identifiant inventé** (Couche 4) reste sur les tool calls, avant le texte — **non négociable, conservé tel quel**.
- Ce qu'on **peut** accélérer sans y toucher : signal d'activité honnête (< 200 ms), statut d'outil honnête en SSE, suppression de l'aller-retour redondant, parallélisation des outils, prompt resserré / `max_tokens` ajusté. **Le cache de l'assembleur (PERF) n'est pas modifié.**

---

## 9. PLAN PROPOSÉ (après GO) — résumé
1. **SSE** sur `/api/chatbot/message` : événements d'orchestration **honnêtes** (activité < 200 ms → statut d'outil réel → **réponse finale validée en un bloc**). Aucune couche retirée.
2. **Zéro relecture redondante** + **outils parallèles** (dans `chatbot.py`).
3. **Dégonfler le pied** et **compacter en-tête / bloc combinaison** → rendre ≥ 100 px à la conversation ; **≥ 4 messages** visibles en 1280×800.
4. Suggestions compactes, effacées après le 1er échange, accessibles derrière un geste.
5. Tests : 4 budgets mesurés ; garde « nombre de vérifications non diminué » ; id inventé toujours rejeté ; pas de double appel outil pour un même contexte ; ≥ 4 messages sans défiler ; champ de saisie toujours visible ; aucun mot interdit (fr/en) dans suggestions et messages d'attente. Playwright 1280×800 + 390×844, fr/en, avant/après.

---

## 10. APRÈS — CE QUI A ÉTÉ IMPLÉMENTÉ (GO reçu)

### Vitesse perçue — SSE honnête, validation intacte
- **Nouveau `Chatbot.chat_events()`** (source unique) : `chat()` le draine, l'endpoint SSE le diffuse → **impossible de diverger** sur la validation. Événements : `activity` (dès que Couche 1 laisse passer, avant le modèle) → `tool` (juste avant une lecture RÉELLE, jamais pour un appel dédupliqué/caché, jamais pour une action d'affichage) → `answer` (texte **complet, validé Couche 3**, poussé d'un bloc). `chatbot.py:295-540`.
- **Nouvel endpoint `POST /api/chatbot/stream`** (`StreamingResponse` `text/event-stream`). Le `/message` JSON reste pour le fallback et la carte /actualites (`askSentinel`). `src/api/routes/chatbot.py`.
- **Frontend** : `askSentinelStream` (SSE, parse `data:` frames) ; `ChatProvider` expose `activity` ; `ThinkingIndicator` affiche une légende **honnête et visible** (« Lecture de XAU/USD M15… ») bâtie côté client depuis les arguments de l'outil (i18n 9 locales), sinon des points muets (`sr-only`). Le signal générique s'affiche **immédiatement** (avant tout réseau) via `isLoading`/`activity='thinking'`.
- **Zéro relecture redondante** : dédup par tour (clé nom+arguments) sur toutes les rounds → le moteur n'est touché qu'une fois pour un même contexte, et le flash de statut n'apparaît qu'une fois.
- **Outils indépendants en parallèle** : `ThreadPoolExecutor` quand un tour demande ≥2 lectures (sinon série) — vérifié par un test de temps mural.
- **Prompt** : `signal_summary` injecté en **JSON compact** (moins de tokens de prefill, contenu identique).

### La contrainte tenue (réponse à C, appliquée)
La prose du modèle **n'est jamais diffusée** : seul l'événement `answer` porte du texte, et il passe par la Couche 3 **avant** émission. Décision fondateur : « premier caractère » = **premier retour visible** (< 200 ms, signal honnête) ; le premier caractère de la **réponse validée** reste borné par la génération. Couche 1 (avant modèle) et Couche 4 (ancrage d'ID sur les tool calls) restent en amont/hors de la prose — un identifiant inventé reste rejeté par le code.

### Place donnée à la conversation
- **Suggestions** : blocs pleine largeur → **chips compacts qui s'enroulent** (`ChatWelcome`), **effacés après le 1er échange** (état vide uniquement). `data-testid` pour les assertions.
- **En-tête** : note pédagogique repliée à l'**état vide** → l'en-tête se compacte dès que la conversation démarre (la ligne de conformité sous le champ garde la posture visible en permanence).
- **Pied** : l'indice clavier passe en **`peer-focus-within`** (visible seulement pendant la saisie) → rend ~1 ligne permanente à la conversation. Champ de saisie **toujours visible** (flex, jamais défilé).
- **Bloc combinaison** : déjà sur une seule ligne inline (`· XAU/USD · M15`).

### Tests ajoutés / mis à jour
- **Backend** : `tests/test_chatbot_streaming.py` (11) — ordre activité→outil→réponse ; **seul `answer` porte du texte** ; dédup = 1 appel moteur ; **lectures parallèles** (temps mural) ; Couche 1/3/4 tirent dans le flux ; **garde « nombre de vérifications non diminué »** (adversarial ≥35, forbidden ≥115, whitelist == 11) ; overhead d'orchestration < 200 ms. `tests/test_chatbot_endpoint.py` +4 (SSE happy/tool-status/adversarial/503). Suites chatbot : **302 passent**.
- **Frontend** : `mia1-vocabulary.test.ts` (9 locales — aucun mot interdit dans suggestions & messages d'attente) ; `ThinkingIndicator.test.tsx` (3 — légende honnête + points muets) ; smoke + `ChatProvider.test` + integration e2e repointés SSE ; nouveau `tests/e2e/mia-1-latence-espace.spec.ts` (signal d'activité avant réponse retardée ; **≥4 messages visibles** 1280×800 fr+en ; suggestions effacées après 1er échange ; champ toujours visible ; onglet mobile 390×844).

### Régression attrapée en test (et corrigée)
Première tentative de récupération de la ligne du pied : révéler l'indice clavier **au focus** (`peer-focus-within`). Les tests Playwright de dictée `/app` ont cassé (2/2). Diagnostic : l'indice apparaissant **au focus** agrandissait le pied ancré en bas → le micro **remontait entre le `mousedown` et le `mouseup`** → aucun `click` émis (le clic natif Playwright ratait, un `.click()` programmatique passait). `/app` seul touché car seul `/app` utilise le `ChatInput` partagé (/zones, /actualites ont leur propre champ). **Correctif** : l'indice clavier passe en `title` du champ (infobulle, coût de mise en page nul, aucun décalage au focus) — la ligne est rendue à la conversation sans casser la dictée. Preuve : `voice-input-mia` `/app` 2/2 reverts au vert.

### Ce qui n'a PAS pu être accéléré sans toucher à la validation
- Le **premier caractère du texte validé** : borné par la fin de génération (Couche 3 = post-hoc sur le texte). On ne diffuse pas la prose.
- Le **rejet d'identifiant inventé** (Couche 4) reste sur les tool calls, avant la prose — conservé tel quel.
- **Non touché** : moteur de détection, `market_reading_assembler.py`, cache PERF (PERF-1/PERF-2). Le gain anti-redondance vit dans `chatbot.py`.

### Découpage du temps — APRÈS (structure)
La durée de génération du modèle est inchangée (Haiku, même tier), mais l'**écran n'est plus immobile** : signal honnête < 200 ms, statut de lecture réel pendant l'attente, et **un aller-retour LLM en moins** sur les suivis de même contexte. Les valeurs ms précises seront confirmées à l'instrumentation live (clé Anthropic requise).

*(Répartition pixels « après » et comptage de messages : confirmés par les captures Playwright avant/après, à joindre.)*
