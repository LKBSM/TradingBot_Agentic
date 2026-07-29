# Décision DG-1 (b) — Un seul journal de structure à l'écran

- **Date** : 2026-07-29
- **Contexte** : mission DG-1 (divergence tendance / journal), point 6.
- **Statut** : adoptée.

## Problème

Deux systèmes de journal BOS/CHOCH coexistent dans le code :

1. **Assembler live** (`MarketReadingAssembler.get_or_generate` → `/api/market-reading`) :
   recalcule la structure à chaque accès sur une fenêtre de ~500 bougies fraîches.
   C'est ce que lit l'App (tuile Régime, maturité, dernier événement, chart).
2. **Store profond LB-1** (`structure_store.py` → `/api/structure`, `/api/coverage`) :
   journal persistant, **seedé par le backfill uniquement** — le câblage
   `scheduler → IncrementalDetector.refresh` est différé (cf.
   `docs/audits/AUDIT-lb-1-lookback-profond.md` §4.2). Son journal est donc **gelé**
   entre deux backfills.

L'état **inacceptable** serait que ces deux journaux, couvrant des périodes
différentes, soient tous deux atteignables depuis le même écran sans que rien
n'indique qu'ils ne recouvrent pas la même période.

## Constat

Aujourd'hui, **aucune surface de l'app ne consomme `/api/structure` ni
`/api/coverage`** : la tuile et le chart lisent tous deux le payload de
l'assembler live. L'écran est donc **déjà mono-source**. Le piège n'est pas
encore refermé, mais l'infrastructure (LB-1) permettrait de le refermer par
inadvertance.

## Décision : option (b)

- **La source unique du journal à l'écran est l'assembler live** (`/api/market-reading`).
- **`/api/structure` et `/api/coverage` ne sont PAS branchés à l'UI** en l'état.
- La **fenêtre est nommée à l'écran** (DG-1 point 5 : Concept de la tuile Tendance
  + étendue calendaire par unité).
- Un **test-garde** verrouille l'invariant « aucune surface ne consomme le journal
  profond » :
  `webapp/lib/market-reading/__tests__/single-journal-guard.test.ts`.

## Condition de réouverture (vers l'option a)

Brancher `/api/structure` à l'UI n'est autorisé qu'après avoir, dans le même
chantier :

1. câblé `scheduler → IncrementalDetector.refresh` pour que le store cesse d'être
   gelé ;
2. réconcilié la fenêtre (DG-1 point 5) et **libellé la période couverte** à
   l'écran, afin que jamais deux journaux de périodes différentes ne cohabitent
   sans mention.

Le test-garde devra alors être mis à jour dans ce même chantier (il documente la
condition), jamais désactivé isolément.
