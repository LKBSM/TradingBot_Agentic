# M.I.A Agent — masquage sélectif multi-zones + défilement chat

Date : 2026-06-29
Branche : `feat/agent-multihide-chat-scroll` (depuis `main` consolidé, worktree dédié)
Portée : **affichage / UX uniquement**. Détection, sécurité, honnêteté : INCHANGÉES.

---

## TL;DR

- **(A) Masquage multi-zones** : la branche `feat/hide-specific-zone` (jamais mergée dans
  `main`) implémentait DÉJÀ le masquage/isolation **multi-zones** par liste d'ids — malgré son
  nom singulier. Elle a été **mergée** dans la branche de travail (décision utilisateur :
  « merger puis étendre »), puis **étendue** par la **résolution d'un groupe factuel**
  (« masque les FVG touchés ») vers les ids réels correspondants.
- **(B) Défilement chat** : **déjà livré dans `main`** (`c8bc8d7`, hook `useChatAnchorScroll`).
  Conforme à la spec (ancre la question en haut, fluide, streaming-safe, désengagement au
  geste). Décision utilisateur : **vérif live seulement**, aucun changement de code.

---

## Diagnostic (lecture seule)

### (A) État du code existant

`feat/hide-specific-zone` = 1 commit (`dc33b1d`), **NON mergé** dans `main`. Contenu déjà
présent et déjà multi-zones :

| Couche | Existant |
|---|---|
| Verrou serveur `view_action_filter.py` | `hide_zones / isolate_zones / show_zones` ; `_ZONE_REFS_ACTIONS` + `_zone_refs()` valide une **LISTE** `zone_ids`, dédup, **rejette toute l'action si un seul id est inventé/vide** (même verrou que `focus_zone`). `show_zones` sans ids = tout restaurer. |
| Verrou front `viewActions.ts` | types union, `coerceZoneIdList()`, reducer `hiddenZoneIds[]` + `isolatedZoneIds[]\|null`, garde géométrie inchangée. |
| Rendu `zoneLayout.ts` | `applyZoneVisibility()` — filtre d'affichage pur. |
| Wiring | `ReadingChart.tsx` + `ReadingColumn.tsx`. |
| Agent `chatbot.py` | schémas d'outils + prompt, **résolution par prix** (« l'OB à 4160 »). |

**Delta réel** : (1) intégrer la branche ; (2) ajouter la **résolution par état/groupe** —
la donnée existe déjà (chaque OB/FVG du `get_market_reading` porte `id` **et** `status`), seule
manquait la consigne d'agent + un test.

### (B) Défilement

`feat/chat-scroll-ux` mergée dans `main` (`c8bc8d7`). `useChatAnchorScroll` ancre la dernière
question en haut, défilement `smooth`, re-ancrage no-op (streaming-safe), 1er geste
molette/tactile désengage le suivi. **= spec (B) déjà satisfaite.**

---

## Implémentation (après GO)

### (A) Extension : résolution de groupe par état

1. **Merge** `feat/hide-specific-zone` → `feat/agent-multihide-chat-scroll` (`--no-ff`).
2. **Prompt agent** (`src/intelligence/chatbot/chatbot.py`) : nouvelle consigne — pour un
   GROUPE désigné par un critère factuel (« masque les FVG touchés », « n'affiche que les OB
   actifs », « cache les zones mitigées »), l'agent lit `get_market_reading`, sélectionne les
   zones réelles via leur champ `status` (`active / mitigated / partially_filled / filled /
   invalidated` ; « touché » = `mitigated` ou `partially_filled`), rassemble TOUS leurs ids et
   les passe en une seule fois dans `zone_ids`. Si aucune ne correspond → ne masque rien.

Aucun nouveau code déterministe : la résolution est faite par le LLM à partir de données
réelles, et la liste d'ids passe par le **même verrou** que `focus_zone` (un id inventé rejette
toute l'action). Aucune action `create/move/resize` n'est représentable (vocabulaire fermé +
garde géométrie inchangés).

### (B) Aucun changement de code (vérif live).

---

## Tests

Python (`tests/test_chatbot_view_actions.py`) — **34 passants**, dont 3 neufs :
- `test_reading_exposes_per_zone_status_for_group_resolution` — contrat : chaque zone porte
  `id` + `status` (sans quoi le groupe ne serait pas résoluble).
- `test_hide_touched_fvg_group_targets_the_right_ids` — 3 FVG (2 `partially_filled`), l'agent
  masque le groupe « touchés » en UN appel multi-id ; le FVG actif reste.
- `test_group_with_one_invented_id_rejects_whole_action` — un id halluciné dans le groupe →
  toute l'action rejetée, rien masqué.

Tests existants (re-validés) : liste de zones réelles masquée/réversible, isolate/show
roundtrip, id inexistant rejeté, géométrie interdite, détection jamais mutée.

Front : `webapp/lib/chart/__tests__/viewActions.test.ts` — **37 passants** (multi-id
hide/isolate/show + `applyZoneVisibility` composé déjà couverts).

Vérifs : `tsc --noEmit` = 0 · `next build` = OK · `pytest test_chatbot* ` = 48 passants.

---

## Discipline

Affichage/UX uniquement. Le moteur de détection n'est ni lu en écriture ni muté. Staging
explicite (pas de `git add -A`), pas de force push. **Pas de merge sur `main` avant
confirmation explicite du rendu live par le founder.**
