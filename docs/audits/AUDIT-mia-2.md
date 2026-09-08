# AUDIT MIA-2 — Latence M.I.A (DIAGNOSTIC — phase STOP, avant tout code)

> **État git au démarrage.** Répertoire principal `C:\MyPythonProjects\TradingBOT_Agentic`
> sur `docs/preserve-data-1-audit` @ `e0dc69c` = **38 commits DERRIÈRE `origin/main`
> (`271443a`)**, 1 devant. Diagnostic mené dans un **worktree dédié**
> `C:\MyPythonProjects\wt-mia-2` créé sur `origin/main` à jour (`271443a`), branche
> `fix/mia-2-latence`. Toutes les mesures ci-dessous sont prises sur `271443a`.

> **Périmètre de mesure honnête.** Ce worktree n'a **ni `ANTHROPIC_API_KEY` ni base
> de marché** (`data/` vide, `.gitkeep` seul). Les composants **déterministes**
> (tailles de contexte, coût des couches regex, sérialisation) sont **mesurés
> réellement** sur cette machine (section B, D). Les composants **réseau + génération
> LLM + lecture moteur** sont **modélisés** à partir des comptes de jetons mesurés et
> des caractéristiques connues de Haiku 4.5, et **explicitement étiquetés « modélisé »**.
> Les millisecondes bout-en-bout définitives (3 cas × 3 exécutions) seront capturées
> **en live dans l'environnement du fondateur** (clé API + DB) — qui est de toute façon
> requis par la mission pour la validation visuelle avant merge. Aucune optimisation
> n'a été appliquée : ceci est le livrable du STOP.

---

## 0. Carte du chemin (une réponse M.I.A)

```
Frontend (webapp)                         Backend (src/intelligence/chatbot)
─────────────────                         ──────────────────────────────────
ChatComposer.handleSubmit
  └─ vide le champ (feedback immédiat)     POST /api/chatbot/stream (SSE)
ChatProvider.askFreeForm
  ├─ push msg user en state  ◀── < 16 ms   chat_events(user_message, history)
  ├─ setActivity('thinking') ◀── < 200 ms   ├─ Couche 1  adversarial.check()  [AVANT LLM]
  └─ askSentinelStream (ReadableStream)      │     → si match: REFUSAL, 0 appel LLM
        ◀── event {activity}                 ├─ yield {activity}
        ◀── event {tool,…}                   ├─ boucle ≤ 3 tours:
        ◀── event {answer} (bloc complet)    │     ├─ messages.create(Haiku)  ← DOMINANT
                                             │     ├─ si tool_use: yield {tool}; exécute
                                             │     │     (dédup in-turn + parallèle ≤4)
                                             │     │     Couche 4 view-action whitelist
                                             │     └─ sinon: Couche 3 output filter [APRÈS LLM]
                                             └─ yield {answer} (texte validé, 1 bloc)
```

**Déjà livré par MIA-1 (PR #188, sur `main`) :** l'événement `{activity}` < 200 ms, la
narration d'outil honnête `{tool}`, la **déduplication d'outil in-turn**, les **lectures
parallèles** (`ThreadPoolExecutor` ≤ 4), le SSE `askSentinelStream`, le `ThinkingIndicator`.
Donc les leviers **3 (signal d'activité)** et **4 (appels d'outils)** de la mission sont
**en place sur `main`**. Le net-restant de MIA-2 est surtout le **levier 1 (cache de prompt)**.

---

## 1. Réponse à E — QUAND VALIDE-T-ON ? (la question décisive)

Quatre défenses. **Trois sur quatre sont vérifiables AVANT / indépendamment de la prose
générée.** Une seule exige le texte complet.

| Couche | Rôle | Fichier | Quand | Sur quoi | Vérifiable en amont ? |
|---|---|---|---|---|---|
| **1** Adversarial | 35 patterns regex (jailbreak/trade/persona/advice) | `adversarial_filter.py` | **AVANT** l'appel LLM | le **message utilisateur** | ✅ oui — sur la requête, pas la sortie |
| **2** Orchestration | Haiku + tool-use borné à 3 tours | `chatbot.py:399-522` | pendant | — | n/a (c'est la génération) |
| **4** View whitelist | 11 actions listées ; rejette tout id/prix inventé | `view_action_filter.py` | **PENDANT** le tool-use, avant `{answer}` | l'**entrée structurée** de l'outil (ids vs ids réellement émis ce tour) | ✅ oui — sur des arguments structurés, pas de la prose |
| **3** Output filter | 115 jetons interdits (4 catégories) | `output_filter.py` | **APRÈS** génération complète | le **texte produit** | ❌ non — n'existe qu'une fois généré |

**Ce qui est vérifiable AVANT génération** (identifiants demandés, marché, unité de temps,
forme de la requête) : Couche 1 (forme/intention de la requête) et Couche 4 (les identifiants
de zone que le modèle cite sont validés contre l'ensemble réellement émis par le moteur, au
moment du tool-use — un id inventé est rejeté **par le code**, indépendamment de la prose
finale).

**Ce qui n'est vérifiable qu'APRÈS, sur le texte produit** : uniquement **Couche 3** (balayage
des 115 jetons interdits sur le texte complet).

**Conséquence directe (tension réelle).** On ne peut pas diffuser au fil de l'eau le **texte
final** token par token sans risquer d'afficher un jeton interdit **avant** que Couche 3 l'ait
vu — donc afficher puis retirer. La mission l'interdit. **Donc on ne diffuse pas la prose
finale non validée.** C'est cohérent avec `chatbot.py:342-347` (contrat déjà documenté).

**Proposition à décider par le fondateur (section 6-P) :** une validation *incrémentale par
phrase* pourrait révéler progressivement **seulement le texte déjà validé**, sans jamais
montrer d'invalidé — un vrai gain sur « premier caractère » sans affaiblir Couche 3. Détaillée
plus bas. **Non appliquée de ma propre initiative.**

---

## 2. Découpage du temps (par cas)

Composants **mesurés** (déterministes, cette machine) vs **modélisés** (réseau+LLM+moteur).

### Composants déterministes — MESURÉS (médiane / pire)
| Étape | Médiane | Pire |
|---|---|---|
| Couche 1 `adversarial.check()` (msg propre) | **0,18 ms** | 3,7 ms |
| Couche 3 `output_filter.check()` (réponse ~600 c) | **1,98 ms** | 19,5 ms |
| `SYSTEM_PROMPT.format(signal_summary)` | 0,04 ms | 0,56 ms |
| `json.dumps(tool_result ~8 Ko)` | 0,04 ms | 1,16 ms |
| Feedback UI (msg user affiché, `setState`) | < 16 ms (1 frame) | — |
| Signal d'activité `{activity}` (MIA-1) | < 200 ms | — |

→ **Les couches de validation ne sont PAS le goulot** : ensemble < 25 ms au pire. Le coût
est ~100 % dans l'appel LLM (préremplissage + génération) et la lecture moteur.

### Modèle de latence LLM — MODÉLISÉ (Haiku 4.5, ancré sur les jetons mesurés en §3)
Hypothèses étiquetées : préremplissage ~4 k jetons non caché → TTFT ≈ 0,4–0,9 s ; lecture
cachée (~3,3 k jetons) ≈ 10× moins cher → TTFT ≈ 0,3–0,5 s ; génération 2–4 phrases
≈ 90–180 jetons @ ~70–110 j/s ≈ 1,0–2,0 s ; lecture moteur `get_or_generate` ≈ 0,1–0,4 s.

| Cas | Appels LLM | Lecture moteur | Total AVANT (modélisé) | Total APRÈS cache (modélisé) |
|---|---|---|---|---|
| **A — simple, sans outil** | 1 | 0 | ~1,5–2,9 s (méd. ~2,1 s) | ~1,3–2,4 s (méd. ~1,7 s) |
| **B — lecture de marché** | 2 (tool_use → texte) | 1× (~0,1–0,4 s) | ~3,2–5,6 s (méd. ~4,2 s) | ~2,4–4,3 s (méd. ~3,2 s) |
| **C — zone précise (OB diag)** | 2 | 1× diagnostic | ~3,0–5,2 s (méd. ~4,0 s) | ~2,3–4,1 s (méd. ~3,1 s) |

> La ligne « premier caractère » n'apparaît pas ici volontairement : **le texte final n'étant
> pas diffusé au fil de l'eau, le premier caractère de la RÉPONSE = la réponse complète.** Ce
> qui apparaît < 200 ms/< 1 s est le **signal d'activité honnête** (thinking → narration
> d'outil), pas la prose. Voir §5 budgets.

---

## 3. Taille du contexte envoyé à CHAQUE tour — MESURÉ

| Bloc | Chars | ~Jetons (÷3,7) | Stable d'un tour à l'autre ? |
|---|---|---|---|
| Prompt système (rendu complet) | 9 197 | ~2 486 | partiellement |
| — partie statique du système | 7 508 | **~2 029** | ✅ **stable** |
| — `signal_summary` injecté | 1 695 | ~458 | 🔶 varie (TTL 60 s) |
| Schémas d'outils (4 tools) | 4 844 | **~1 309** | ✅ **stable** |
| **PRÉFIXE STABLE (tools + système statique)** | **12 352** | **~3 338** | ✅ **cacheable** |
| **TOTAL FIXE / tour (système + tools)** | **14 041** | **~3 795** | envoyé **intégralement à chaque message** |

**+ variable non compté ci-dessus :** l'historique (jusqu'à 20 messages × 2 000 chars côté
client, **passé tel quel, aucune troncature serveur** — `chatbot.py:356,380`), le message
utilisateur, et sur les tours d'outil le **`tool_result` = `MarketReading.model_dump()` complet**
(plusieurs Ko → ~1 000–2 500 jetons) ré-injecté dans `messages`.

**Où placer la frontière stable/variable (réponse au point C de la mission) :**
- **Stable, à cacher** : les **schémas d'outils** (~1 309 j) + la **partie statique du système**
  (~2 029 j) = **~3 338 j** identiques à tous les tours.
- **Variable** : `signal_summary` (~458 j, change au plus toutes les 60 s), l'historique, le
  message. **Défaut structurel actuel** : `signal_summary` est injecté **au MILIEU** du prompt
  système (`chatbot.py:236-237`), suivi du bloc stable « Tu as accès à 4 tools » (l.239-245).
  Un contenu qui varie **avant** du contenu stable **empêche** de cacher ce bloc stable de fin.
  → Il faut **déplacer `signal_summary` en FIN** de système (ou hors système) pour que tout le
  préfixe stable soit un bloc cacheable propre.

---

## 4. État de la mise en cache du prompt — MESURÉ : **AUCUN**

- `grep cache_control|anthropic-beta|prompt-caching` dans `src/` : **aucune occurrence dans le
  chatbot ni le scanner_translator**. L'appel `chatbot.py:401-407` passe `model, max_tokens,
  system, messages, tools` — **sans `cache_control`**.
- **Mais le motif existe déjà, testé, dans le dépôt** : `src/intelligence/llm_cost_policy.py`
  (`cache_block_for()`, seuil `CACHE_MIN_TOKENS=1024`, bloc `{"type":"text","cache_control":
  {"type":"ephemeral"}}`) et `llm_narrative_engine.py:504` l'utilise déjà en production. Le
  chatbot est **le seul appelant qui ne l'a jamais adopté**.
- Conséquence : à **chaque** message, ~3 338 jetons parfaitement stables sont **retraités
  intégralement** par le modèle (préremplissage plein tarif, pleine latence). C'est **le
  levier le moins cher et le plus large** de la mission.

---

## 5. Budgets visés vs réalité modélisée

| Budget mission | Cible | État |
|---|---|---|
| Signal d'activité visible | < 200 ms | ✅ **atteint** (MIA-1 `{activity}`) |
| Premier caractère de la réponse | < 1 s | ⚠️ **en tension avec Couche 3** — voir ci-dessous |
| Réponse complète, sans outil | < 3 s | ✅ atteignable (cache → méd. ~1,7 s) |
| Réponse complète, avec outil | < 5 s | ✅ atteignable, plus serré (cache → méd. ~3,2 s) |

**« Premier caractère < 1 s ».** Deux lectures :
1. *Premier retour visible* (thinking / narration d'outil) : **déjà < 200 ms**. ✅
2. *Premier caractère de la PROSE de réponse* : impossible sans diffuser du texte **avant**
   Couche 3 → **interdit** par la contrainte. Sauf adoption de la validation incrémentale par
   phrase (proposition §6-P). **Décision fondateur requise sur l'interprétation.**

---

## 6. Leviers classés par rapport GAIN / EFFORT / RISQUE

| # | Levier | Gain | Effort | Risque validation | Touche moteur/cache partagé ? |
|---|---|---|---|---|---|
| **1** | **Cache de prompt** (tools + système statique ~3,3 k j) + déplacer `signal_summary` en fin | **Élevé** (préremplissage −~90 % sur le bloc stable, chaque appel) | Faible | **Nul** (métadonnée d'appel ; motif déjà testé) | **Non** — uniquement construction de requête dans `chatbot.py` |
| **5a** | **Tronquer l'historique côté serveur** (ex. N derniers tours) | Moyen (croît avec la conversation) | Faible | Nul | Non |
| **5b** | **Condenser le `tool_result`** ré-injecté (MarketReading complet → champs utiles) | Moyen-élevé sur cas B/C | Moyen | ⚠️ **doit préserver les ids** que Couche 4 exige | Non (mais délicat) |
| **6** | **Longueur de réponse** : `max_tokens` 1024 → ~600, resserrer la consigne « 2-4 phrases » | Faible-moyen | Faible | Nul (vocab inchangé) | Non |
| **G** | **Timeout explicite sur l'appel chatbot** (le scanner_translator a 20 s + breaker ; le chatbot **n'a aucun timeout** → défaut SDK ~600 s sur un appel bloqué) | Faible en médiane, **élevé en pire cas** | Faible | Nul | Non |
| **2** | Diffusion token-par-token de la prose finale | — | — | **Incompatible** Couche 3 → **NE PAS FAIRE** | — |
| **P** | *(proposition)* Validation Couche 3 **incrémentale par phrase** + diffusion du seul texte validé | Élevé sur « premier caractère » | **Élevé** | Nul **si** bien fait (ne révèle que le validé) | Non |

**Ordre d'implémentation recommandé (après GO) :** 1 → 5a → 6 → G, puis discuter P et 5b.
Le levier **1 seul** devrait rentrer les budgets « réponse complète ». P est la seule voie
honnête vers « premier caractère < 1 s » et mérite une décision explicite.

### 6-P. Proposition — validation incrémentale par phrase (À DÉCIDER, non appliquée)
Aujourd'hui Couche 3 balaye le texte **complet** puis on émet `{answer}` d'un bloc. Alternative
**sans affaiblir Couche 3** : activer `stream=True` sur l'appel Haiku, **bufferiser** la sortie,
et à chaque **frontière de phrase** faire passer la phrase par le **même** `output_filter.check`
(115 jetons, inchangé) ; **ne révéler la phrase que si elle est propre**. Une phrase contaminée
déclenche le fallback et **rien d'invalidé n'a jamais été affiché**. Gain : le premier segment
validé peut apparaître ~0,5–1,2 s au lieu d'attendre toute la génération. Coût : complexité
réelle (gestion du flux, frontières de phrase robustes, fallback en cours de flux), et le
« premier caractère < 1 s » resterait **best-effort**, pas garanti. **Invariant préservé** :
on ne diffuse **que** du texte déjà passé par Couche 3. **Je ne l'implémente pas sans ton feu vert.**

---

## 7. Question G — Modèles & réglages (MESURÉ)

| | Chatbot M.I.A | Scanner translator |
|---|---|---|
| Modèle | `claude-haiku-4-5-20251001` | `claude-haiku-4-5-20251001` (**identique**) |
| `max_tokens` | 1024 | 1024 |
| `temperature` | non défini (défaut) | non défini |
| `timeout` | **aucun** (défaut SDK) | **20 s** + circuit breaker |
| `cache_control` | **non** | **non** |
| Fichier | `chatbot.py:44-45,401-407` | `scanner_translator.py:43-44,520-527` |

Le chatbot utilise déjà le modèle le plus rapide (Haiku 4.5) : **ce n'est pas le modèle le
problème**. `max_tokens=1024` n'allonge pas la génération en soi ; la consigne « 2-4 phrases »
la borne déjà. Point d'attention : **absence de timeout côté chatbot** (le scanner en a un).

---

## 8. Comptes de vérifications de sécurité (socle du test de non-régression)

| Couche | Unité | Compte (mesuré `271443a`) |
|---|---|---|
| Couche 1 adversarial | patterns regex | **35** (jailbreak 8 · trade_request 9 · persona_hijack 9 · financial_advice 9) |
| Couche 3 output filter | jetons interdits (déclarés) | **115** (action 26 · reco 41 · moment 21 · risque 27) |
| Couche 4 view whitelist | actions autorisées | **11** |

→ Le test « échoue si le NOMBRE de vérifications diminue » verrouillera ces comptes
(par catégorie), plus le fait que chaque couche s'exécute (Couche 1 avant LLM, Couche 4 sur
id inventé rejeté, Couche 3 sur texte complet).

---

## 9. Ce qui NE pourra PAS être accéléré sans toucher à la validation

- **Diffusion token-par-token de la prose finale** — Couche 3 exige le texte complet ; on
  garde l'émission en un bloc validé (sauf adoption de 6-P).
- **Couche 3 elle-même** — ne peut pas remonter en amont : elle inspecte une prose qui
  n'existe qu'après génération. On peut la rendre *incrémentale* (6-P) mais pas *antérieure*.
- **La lecture moteur `get_or_generate`** sur les cas B/C reste une E/S réelle ; on peut la
  cacher/paralléliser (déjà fait in-turn par MIA-1) mais pas la supprimer sans changer le
  moteur (= code PERF-2, à signaler avant de toucher).

---

## 10. Décisions demandées au fondateur (STOP)

1. **GO cache de prompt** (levier 1) + **déplacement de `signal_summary` en fin de système** ?
   (aucun impact validation ; motif déjà testé dans le dépôt)
2. **Interprétation de « premier caractère < 1 s »** : signal d'activité (déjà atteint) suffit-il,
   ou veux-tu la **proposition 6-P** (validation incrémentale par phrase, effort élevé) ?
3. **Levier 5b** (condenser le `tool_result`) : autorisé **à condition** de préserver les ids
   requis par Couche 4 ? (sinon je m'en tiens à 5a + 6 + G, zéro risque)
4. Confirmer que la capture **live** des 4 budgets se fera dans ton environnement (clé API +
   DB) lors de la validation visuelle avant merge.

_(Fin de la partie DIAGNOSTIC — le STOP a été validé.)_

---
---

# PARTIE 2 — APRÈS (implémentation, post-GO)

**Décisions du fondateur au STOP :** (1) GO cache + déplacement de `signal_summary` ;
(2) « signal d'activité suffit » → **pas de diffusion de la prose** (proposition 6-P
écartée) ; (3) **5a + 6 + G seulement** — pas de condensation du `tool_result` ;
(4) capture **live** des 4 budgets dans l'environnement du fondateur avant merge.

## A. Ce qui a été implémenté (4 leviers, aucune couche affaiblie)

| Levier | Changement | Fichier |
|---|---|---|
| **1 — cache** | Prompt système scindé : bloc **statique** `SYSTEM_PROMPT_STATIC` marqué `cache_control:ephemeral` (via `cache_block_for`, motif existant) + bloc **variable** `SIGNAL_CONTEXT_TEMPLATE` (signal_summary) placé **APRÈS** la frontière de cache. `system` devient une liste de 2 blocs. Contenu identique, summary déplacé du milieu vers la fin. | `chatbot.py` (`_build_system`, `SYSTEM_PROMPT_STATIC`, `SIGNAL_CONTEXT_TEMPLATE`) |
| **5a — historique** | Troncature serveur à `MAX_MODEL_HISTORY = 12` messages, tranche **ouvrant sur un tour `user`** (transcript alternant valide). | `chatbot.py` (`_truncate_history`) |
| **6 — longueur** | `DEFAULT_MAX_TOKENS` 1024 → **768** (plafond de sécurité ; ne rallonge jamais une réponse normale). | `chatbot.py` |
| **G — timeout** | `timeout=DEFAULT_TIMEOUT_S (20 s)` passé à **chaque** `messages.create` (le scanner en avait un, pas le chatbot). Sur timeout → `LLM_ERROR_TEMPLATE` (branche existante, inchangée). | `chatbot.py` |

**Invariants préservés (vérifiés par tests) :** Couche 1 avant LLM, Couche 4 rejette
tout id inventé, Couche 3 sur le **texte complet** avant `{answer}`. Aucun compte de
vérification n'a baissé (35 / 115 / 11, gardés par `>=`). Dédup in-turn et lectures
parallèles (MIA-1) intacts. La prose finale n'est **pas** diffusée token par token.

## B. Taille du contexte — AVANT / APRÈS (mesuré)

| | AVANT (271443a) | APRÈS (cache-hit) |
|---|---|---|
| Préfixe fixe **retraité** à chaque tour | **~3 795 j** (système+tools, plein tarif) | **~468 j** (bloc signal_summary seul) |
| dont **caché** (lu, non retraité) | 0 | **~3 357 j** (tools ~1 309 + système statique ~2 048) |
| Historique envoyé | jusqu'à 20 × 2 000 c, **non tronqué** | **≤ 12 messages** |
| Plafond de réponse | 1024 j | 768 j |

→ **−~88 % de jetons de préremplissage fixes retraités par tour** sur un cache-hit
(~3 795 → ~468). Le préremplissage est le premier poste de latence par tour : c'est
le gain central. **Caveat honnête** : le cache Anthropic (ephemeral) a un TTL de ~5 min
— le **1ᵉʳ** message d'une session paie le plein préremplissage ; les tours suivants
dans les 5 min lisent le cache. Un chat de trader est typiquement en rafale → cache chaud.

## C. Découpage du temps — AVANT / APRÈS (modélisé ; live à venir)

Les composants déterministes restent négligeables et **inchangés** (Couche 1 ~0,18 ms,
Couche 3 ~2 ms). Le gain porte sur le préremplissage LLM. Estimations modélisées
(mêmes hypothèses qu'en Partie 1), **cache chaud** :

| Cas | AVANT (méd. modélisé) | APRÈS (méd. modélisé) | Budget | Gain attribué |
|---|---|---|---|---|
| A — sans outil | ~2,1 s | **~1,6 s** | < 3 s | cache (préremplissage ↓) |
| B — lecture marché (2 appels) | ~4,2 s | **~3,0 s** | < 5 s | cache ×2 appels |
| C — zone précise (2 appels) | ~4,0 s | **~2,9 s** | < 5 s | cache ×2 appels |
| Signal d'activité | < 200 ms | **< 200 ms** (inchangé, MIA-1) | < 200 ms | — |

**Gain par levier, séparément :**
- **Levier 1 (cache)** — le gros du gain : ~3,3 k jetons/appel non retraités ; effet
  doublé sur les cas à 2 appels. C'est le levier qui rentre les budgets « réponse complète ».
- **Levier 5a (historique)** — gain croissant avec la longueur de conversation ; borne le
  préremplissage variable (au-delà de 12 messages, plus d'inflation).
- **Levier 6 (max_tokens 768)** — gain **médian ~nul** (réponses typiques ~120-180 j) ;
  protège les cas pathologiques de génération longue. Assumé faible (cf. classement STOP).
- **Levier G (timeout 20 s)** — gain **médian nul, pire-cas majeur** : un appel bloqué
  tombait sur le défaut SDK (~600 s) ; plafonné à 20 s → bascule sur le template.

> **Les millisecondes bout-en-bout définitives (AVANT/APRÈS, 3 cas × 3 exécutions,
> médiane + pire cas) seront produites par `tests/test_chatbot_budgets_live.py`** dans
> ton environnement (clé API + DB) lors de la validation visuelle avant merge. Ce
> tableau est un modèle, pas une mesure live — je le remplacerai par les chiffres réels
> après ta session.

## D. Tests livrés

- `tests/test_chatbot_latency.py` (16 tests) : préfixe caché + `cache_control` ; summary
  **après** la frontière (non caché) ; préfixe **byte-stable** entre tours ; aucune info
  perdue ; historique tronqué + ouverture sur `user` ; `max_tokens==768` ; `timeout` sur
  **chaque** appel ; **non-régression sécurité** (35/115/11 en `>=`) ; **id inventé rejeté** ;
  **même outil pas 2× même contexte** ; signal d'activité **avant** tout appel modèle ;
  overhead non-LLM < 150 ms.
- `tests/test_chatbot_budgets_live.py` : **les 4 budgets end-to-end**, skip sans clé+DB,
  tourne dans ton env (3 cas × 3 exécutions, médiane/pire, activité < 200 ms).
- Front `ChatProvider.test.tsx` : **message utilisateur affiché avant** la résolution réseau
  (stream jamais résolu → message + état loading visibles).
- Front `mia1-vocabulary.test.ts` (existant, étendu au périmètre) : **aucun mot interdit**
  dans les messages d'attente/activité, **9 locales** (fr/en/es inclus).
- E2E `tests/e2e/mia-2-latence.spec.ts` : **3 états** (attente → réponse affichée →
  terminée) à **1280×800 et 390×844**. 4/4 verts.
- Suite chatbot backend : **356 passed, 1 skipped (budgets live), 2 échecs PRÉ-EXISTANTS**
  (`test_smoke_e2e` scanner 503 = pas de DB dans le worktree ; confirmés sur `271443a` vierge).
- `tsc` : 3 erreurs **pré-existantes** (`dictation-copy-honesty.test.ts`), **0 nouvelle**.
- `npm run build` : **vert**. Vitest chat : **32/32**.

## E. Ce qui n'a PAS pu être accéléré sans toucher à la validation

- **Diffusion token-par-token de la prose finale** — bloquée par Couche 3 (texte complet).
  Écartée sur décision fondateur (« signal d'activité suffit »). Le premier retour visible
  reste le signal d'activité honnête (< 200 ms), pas la prose.
- **Couche 3 remontée en amont** — impossible : elle inspecte une prose post-génération.
- **Condensation du `tool_result`** (levier 5b) — **non faite** (décision fondateur), pour
  ne prendre aucun risque sur les ids requis par Couche 4.
- **Lecture moteur `get_or_generate`** (cas B/C) — E/S réelle ; la toucher = code PERF-2
  (cache partagé), **hors périmètre** — signalé, non modifié.

## F. Discipline

Worktree dédié `wt-mia-2` (jonction `node_modules` ← `wt-run-main`, **à retirer via
`cmd /c rmdir` avant tout teardown**). Staging **explicite** au commit (jamais `git add -A`).
**Pas de merge avant ta confirmation visuelle live.**
