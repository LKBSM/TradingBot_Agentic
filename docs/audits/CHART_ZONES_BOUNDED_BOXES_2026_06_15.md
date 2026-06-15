# Rendu des zones OB/FVG — boîtes localisées et bornées (front/chart)

**Date :** 2026-06-15
**Branche :** `feat/chart-zones-bounded-boxes` (depuis le tip Direction 1 + `mitigated_at`)
**Périmètre :** front/chart **uniquement**. Aucune modif moteur / détection / schéma.

---

## 1. Problème

Le chart dessinait chaque OB/FVG comme une **bande horizontale pleine largeur**
(`<div left-0 right-[64px]>`), bornée uniquement en prix. L'axe temps était ignoré :
aucune localisation à la bougie de formation, labels empilés en haut-gauche →
chevauchement illisible. `mitigated_at` n'était pas consommé.

## 2. Solution livrée

### A. Boîtes localisées
Chaque zone = une boîte ancrée à sa **bougie de formation** (`x-début =
created_at`). Plus aucune bande edge-to-edge. La conversion temps→x passe par
`timeScale.timeToCoordinate()` après **snap sur la bougie la plus proche** (garantit
une coordonnée résoluble), puis clamp dans la zone de tracé (jamais au-delà de la
gouttière de prix).

### B. Bornage via `mitigated_at` (lecture seule)
- **Active** (`mitigated_at` = null) → boîte jusqu'à la **bougie courante**.
- **Testée/mitigée** (`mitigated_at` présent) → boîte bornée au **point de
  mitigation**. Aucune projection vers le futur.

### C. Hiérarchie active / testée
| État | Remplissage | Bordure | Label |
|---|---|---|---|
| Active | `rgba(rgb, 0.12)` | pointillé `0.45` | visible (10px) |
| Testée | `rgba(rgb, 0.05)` | pointillé fantôme `0.18` | **aucun** |

Une active et une testée ne se disputent jamais l'attention.

### D. Style Direction 1 (conservé)
Chandeliers `#2F9E78` / `#C2693E`, OB `#8B95A7`, FVG `#6E84B0`, bordures fines
pointillées, labels muted 10px mono, grille horizontale seule. Pas de
néon/glow/dégradé. Clair + sombre vérifiés.

### E. Curation anti-encombrement (mécanique, documentée)
`lib/chart/zoneLayout.ts` → `curateZones()` : au plus **4 actives + 3 testées**,
choisies par **récence + proximité au prix** via une somme de rangs
(`proximityRank + recencyRank`, plus bas = gardé ; tie-break sur `id` pour
déterminisme). Le reste est masqué. **Ce n'est PAS un tri par importance
prédictive** (qui dépendra de l'annotation) — juste « quoi montrer sans saturer ».

### F. Labels
Un seul petit label par boîte **active**, au bord gauche (`whitespace-nowrap`,
pas d'empilement car chaque boîte est localisée en x). Testées : aucun label.

## 3. Fichiers

- `webapp/lib/chart/zoneLayout.ts` *(nouveau)* — logique **pure** : `buildZoneModels`
  (drop défensif des zones consommées), `curateZones`, `isoToSec`.
- `webapp/components/app/ReadingChart.tsx` — rendu localisé + bornage temporel.
- `webapp/lib/market-reading/fixtures.ts` — 2 zones **testées de démo** ajoutées à
  `FIXTURE_XAU_M15` (OB mitigé + FVG partiellement comblé, avec `mitigated_at`) pour
  exercer le bornage. Front-only, pas de moteur/schéma.
- `webapp/lib/chart/__tests__/zoneLayout.test.ts` *(nouveau)* — 9 tests.

## 4. Vérifications

- `npm run typecheck` (tsc --noEmit) : **OK**.
- `npx vitest run` : **155/155** (23 fichiers), 0 régression. +9 tests zoneLayout
  (caps actif/testé indépendants, proximité, récence en tie-break, déterminisme,
  drop zones consommées, `mitigated_at` lecture seule).
- `npm run build` : **vert**.
- **Vérification visuelle** (Edge, locale fr-FR, `/app` live XAU/USD) : 7 boîtes
  rendues — **4 actives labellisées + 3 testées sans label** (cap respecté), toutes
  **localisées** (left ~200-264px, largeurs 2-62px, plus aucune pleine largeur),
  **clair + sombre** corrects, 0 erreur page.

## 5. Discipline respectée

`mitigated_at` consommé en lecture seule ; aucune zone recalculée ; descriptif, aucune
projection ; pas de néon/glow ; clair + sombre. Pas de `git add -A`, pas de force push.

## 6. Note environnement (hors scope)

Boucle de redirection navigateur sur `/` et `/en/*` quand le navigateur annonce
`Accept-Language: en` : interaction pré-existante entre la détection de locale
next-intl et le strip `/en → /` (locale FR par défaut sans préfixe). Sans rapport
avec ce changement. Contournée en forçant la locale `fr-FR` pour la capture. À
traiter séparément si besoin.
