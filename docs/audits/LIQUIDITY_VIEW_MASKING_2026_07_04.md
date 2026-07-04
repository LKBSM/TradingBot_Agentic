# Poches de liquidité (SSL/BSL) masquables par M.I.A Agent — 2026-07-04

**Branche** : `feat/liquidity-view-masking` (worktree dédié, depuis `origin/main` 1526049)
**Statut** : prêt — merge sur `main` UNIQUEMENT après confirmation live du fondateur.

## Problème

L'agent refusait « masque les SSL/BSL » en prétendant que la liquidité n'était pas
contrôlable. **Faux refus** : masquer une poche est un filtre d'affichage réversible,
au même titre qu'un OB/FVG. Deux causes racines prouvées au diagnostic :

1. **Omission prompt/tool** — la description de `apply_chart_view` énumérait
   `layer: 'fvg'|'ob'|'breaks'|'all'` et le prompt système ne citait que
   « FVG, OB, BOS/CHOCH » comme couches masquables, alors que le validateur
   acceptait déjà `layer: 'liquidity'` (test existant vert).
2. **Trou du verrou d'id** — `Chatbot._harvest_zone_ids` ne récoltait que
   `order_blocks` + `fair_value_gaps`. Les ids `LIQ_*` (stables, émis par
   `_liquidity_to_models` sous la forme `LIQ_{side}_{kind}_{created}`) n'entraient
   jamais dans `known_zone_ids` → tout `hide_zones` sur une poche RÉELLE était
   rejeté `unknown_zone_id`.

Côté frontend, le rendu des segments liquidité (`buildLiquidityLines`) ne passait
pas par `applyZoneVisibility` — le masquage-par-id ne touchait que les boîtes OB/FVG,
alors que chaque segment portait déjà son id.

## Ce qui a été construit (option b + isolate uniforme, GO fondateur)

### Backend (Couche 4 + orchestrateur)

- **Résolveur de catégorie serveur** (`view_action_filter.py::_zone_category`) :
  `hide_zones` / `isolate_zones` / `show_zones` acceptent
  `{category: 'fvg'|'ob'|'bsl'|'ssl'|'liquidity'}` (enum fermé
  `ALLOWED_ZONE_CATEGORIES`). La catégorie est résolue vers les ids **réellement
  émis ce tour** (index `known_category_ids` construit par la récolte), jamais
  inventés — défense en profondeur : chaque id résolu est re-vérifié contre
  `known_zone_ids`. Le même mécanisme générique sert « enlève les FVG / les OB »
  par ids. `category` + `zone_ids` simultanés → rejet `ambiguous_target`.
- **Catégorie vide = honnêteté** : résolution vide → rejet `empty_category` avec
  message dédié `VIEW_ACTION_EMPTY_CATEGORY_TEMPLATE` (« le moteur n'émet aucune
  structure de cette catégorie… ») remis au modèle — rien n'est masqué, rien
  n'est inventé.
- **Récolte élargie** (`chatbot.py::_harvest_zone_ids`) : les ids des
  `liquidity_pools` entrent dans `known_zone_ids` + buckets `bsl`/`ssl`/`liquidity`
  (et `ob`/`fvg` pour les zones). Le verrou d'id est INTACT : seuls des ids émis
  passent ; un id inventé est rejeté par le code.
- **Multi-couches** : `ALLOWED_MULTI_LAYERS` inclut `liquidity`
  (« enlève les FVG et la liquidité » en un appel).
- **Prompt + description du tool corrigés** : la liquidité EST masquable (couche
  et poche par id/catégorie) ; consigne lecture-d'abord (`get_market_reading`),
  honnêteté si aucune poche émise, et rappel « décrire, jamais “le prix va la
  chercher” ».

### Frontend (mécanisme existant réutilisé, aucun doublon)

- `applyZoneVisibility` généralisé (`<T extends {id: string}>`) — **le même**
  filtre par id sert boîtes OB/FVG et segments liquidité.
- `ReadingChart.tsx` : les segments liquidité passent par
  `applyZoneVisibility(lines, hiddenZoneIds, isolatedZoneIds)`.
  **Isolation uniforme** (décision fondateur) : `isolate_zones` sur des ids
  quelconques masque toute autre structure, poches comprises.
- `AppWorkspace.tsx` : les ids de poches rejoignent `validZoneIds` (verrou
  frontend). Un `focus_zone` sur un id de poche est un no-op gracieux (vérifié).
- `viewActions.ts` : `MultiChartLayer` += `liquidity` (miroir du backend).

### Contrat inchangé

Le frontend ne reçoit **jamais** de `category` : le serveur normalise toujours en
`zone_ids` concrets. Aucun nouveau verbe d'action ; create/place/move/resize
restent irreprésentables et rejetés. **Détection intouchée** (zéro modification
de `SmartMoneyEngine` / mappers / assembler).

## Ligne inviolable — preuve par les tests

| Cas d'honnêteté | Test |
|---|---|
| « masque les SSL » masque TOUTES les SSL émises et rien d'autre | `test_hide_ssl_category_masks_all_emitted_ssl_and_nothing_else` |
| « masque la liquidité » (2 côtés) puis ré-affichage restaure | `test_hide_liquidity_category_then_show_restores` |
| Id de poche INVENTÉ rejeté par le code | `test_hide_invented_pool_id_is_rejected`, `test_hide_zones_rejects_invented_pool_id` |
| Aucune poche de la catégorie → dit honnêtement, ne masque rien | `test_hide_ssl_with_no_ssl_emitted_reports_honestly`, `test_empty_category_rejected_honestly` |
| create/move/resize toujours refusés | `test_create_move_resize_still_rejected_for_pockets`, `test_off_list_action_rejected` |
| Détection jamais mutée par un masquage | `test_detection_never_mutated_by_pool_masking` |
| Résolveur ne peut pas surfacer un id hors verrou | `test_category_resolver_drops_ids_missing_from_known` |
| Pas de géométrie sur une action catégorie | `test_category_mask_rejects_geometry_param` |
| Masquage front par id + isolate uniforme sur les segments | `liquidityLines.test.ts` (« through applyZoneVisibility ») |

## Vérifications

- Backend : `pytest -k "chatbot or view_action or market_reading"` → **437 passed**
  (dont 58 dans `test_chatbot_view_actions.py`, 19 nouveaux). 0 régression.
- Frontend : `vitest run` → **395 passed** (43 fichiers). `tsc --noEmit` → 0 erreur.
  `next build` → vert.

## Fichiers touchés

- `src/intelligence/chatbot/view_action_filter.py` — catégories + multi-couches liquidité
- `src/intelligence/chatbot/constants.py` — template catégorie vide
- `src/intelligence/chatbot/chatbot.py` — récolte poches, index catégories, prompt, description tool
- `tests/test_chatbot_view_actions.py` — 19 tests ajoutés
- `webapp/lib/chart/viewActions.ts`, `webapp/lib/chart/zoneLayout.ts`
- `webapp/components/app/ReadingChart.tsx`, `webapp/components/app/AppWorkspace.tsx`
- `webapp/lib/chart/__tests__/viewActions.test.ts`, `webapp/lib/chart/__tests__/liquidityLines.test.ts`

## Caveat connu (hors périmètre)

L'id d'une poche retombe sur le timestamp de la barre courante quand l'index du
frame n'est pas datetime (chemin tests uniquement — en production les frames sont
datetime, l'id est stable entre lectures, condition du masquage persistant).
