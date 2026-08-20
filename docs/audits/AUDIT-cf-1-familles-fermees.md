# CF-1 — État initial fermé de toutes les familles de conditions

**Date :** 2026-08-20
**Branche :** `fix/conditions-families-collapsed-by-default` (depuis `origin/main` @ 8894bbf)
**Worktree :** `C:/MyPythonProjects/wt-cond-families`
**Statut :** implémenté, tests verts — **en attente de confirmation visuelle live du fondateur avant merge**

## Demande

Au premier chargement de la page de choix des conditions du scanner (`/scanner`,
onglet « Choisir mes conditions »), les **4 familles** (01 Structure, 02 Zones,
03 Liquidité, 04 Contexte) doivent toutes être **fermées**. Aucune famille dépliée
automatiquement — le client déplie lui-même via le chevron.

## Diagnostic (lecture seule)

**Discipline git :** `git fetch` d'abord. Repo primaire en HEAD détaché, **23 commits
derrière `origin/main`** (main locale elle-même 176 derrière). Diagnostic mené contre
`origin/main`, source de vérité.

**Composant :** `webapp/components/scanner/ConditionsBuilder.tsx` rend les 4 familles.
L'état d'ouverture est un `Set<Family>` :

```tsx
const [open, setOpen] = React.useState<Set<Family>>(() => new Set(DEFAULT_OPEN_FAMILIES));
```

**Cause de l'ouverture automatique :** valeur **codée en dur** dans la palette —

```ts
// webapp/lib/conditions/palette.ts
export const DEFAULT_OPEN_FAMILIES: readonly Family[] = ['structure', 'zones'] as const;
```

→ au chargement, 01 Structure et 02 Zones s'ouvraient d'office.

**Persistance ? NON.** `DEFAULT_OPEN_FAMILIES` n'a qu'un seul consommateur (cet état
initial). Aucun `localStorage`/`sessionStorage` ne mémorise le pli des familles (le seul
`localStorage` du dossier scanner est dans `StrategyPanel.tsx`, pour la liste « Mes
stratégies enregistrées » — sans rapport). Rien à trancher côté persistance : l'état est
recalculé à chaque montage depuis la constante.

**Comportement de pli préservé :** `toggleFamily` ajoute/retire du `Set` — **pas un
accordéon exclusif**. Plusieurs familles peuvent être ouvertes simultanément. Rendre
l'initial vide ne change pas ce comportement.

**Compteurs indépendants :** `activeInFamily()` / `selectedCount` dérivent de `rows`
(sélections), totalement indépendants de `open`. Fermer une famille ne masque ni le
compteur ni les sélections.

## Implémentation

Changement **d'une ligne** (self-documenté, réversible) :

```ts
// webapp/lib/conditions/palette.ts
export const DEFAULT_OPEN_FAMILIES: readonly Family[] = [] as const;
```

La constante (le « seam » déclarant quelles familles sont ouvertes par défaut = aucune)
est conservée plutôt que supprimée — minimal, sans toucher l'import du builder. **Aucun
autre changement** : compteurs, contenu, animation de dépliage, logique de sélection,
`toggleFamily` — tous intacts.

## Fichiers touchés

| Fichier | Nature |
|---|---|
| `webapp/lib/conditions/palette.ts` | `DEFAULT_OPEN_FAMILIES` → `[]` (1 ligne + commentaire) |
| `webapp/lib/conditions/__tests__/palette.test.ts` | verrou : `DEFAULT_OPEN_FAMILIES` est vide |
| `webapp/tests/e2e/cf1-families-collapsed.spec.ts` | **nouveau** — 3 cas × 2 viewports |
| `webapp/tests/e2e/sc1-scanner.spec.ts` | déplie Structure avant de cocher (le défaut est désormais fermé) |

## Tests d'honnêteté

- **tsc `--noEmit`** : 0 erreur.
- **vitest** `palette.test.ts` : **7/7** (dont le nouveau verrou « n'ouvre aucune famille par défaut »).
- **build** `next build` : OK.
- **Playwright** `cf1` + `sc1`, `chromium-desktop`, viewports 1280×800 **et** 390×844 pilotés par le spec : **14/14**.
  - Au chargement : les 4 en-têtes de famille visibles, toutes `aria-expanded=false`, **0 checkbox** dans le DOM (contenu replié).
  - Clic sur un chevron : déplie **cette** famille seule ; ouvrir une 2ᵉ ne referme pas la 1ʳᵉ (multi-ouverture préservée).
  - Replier une famille conserve son compteur « 2 actives » **et** ses coches (ré-ouverture : conditions toujours cochées).
- Captures : `test-results/cf1-collapsed-1280.png`, `test-results/cf1-collapsed-390.png` — conformes à la cible (tout replié, titre + compteur + chevron).

## Reste

Merge sur `main` **seulement après confirmation visuelle live du fondateur**.
