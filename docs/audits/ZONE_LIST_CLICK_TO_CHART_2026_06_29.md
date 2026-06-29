# Zone list → click-to-chart — implémentation

**Date** : 2026-06-29
**Branche** : `feat/zone-list-click-to-chart` (worktree dédié, depuis `feat/chat-ui-redesign`)
**Périmètre** : Affichage / navigation uniquement. Détection, sécurité, honnêteté **inchangées**.

---

## 1. Objectif

Dans `/app`, la liste des zones OB / FVG (niveau · importance · état) était purement
descriptive : non cliquable, et l'`id` réel émis par le moteur était **jeté** au profit
d'une clé synthétique de dédup. On relie désormais la liste au graphe :

> **Cliquer une entrée → le graphe se centre sur la zone et la met en évidence**, via le
> `zone_id` **réel**. Aucune zone inventée, aucune mutation de détection.

## 2. Décision de base de branche

Le composant liste cliquable (`ZoneList.tsx`) n'existe **que** sur `feat/chat-ui-redesign`
(sur `main`, les zones sont une simple ligne texte jointe par `|`). Le mécanisme de
focus/highlight cible (canal `apply_chart_view`) est présent sur les deux. Sur décision
utilisateur, la branche part de **`feat/chat-ui-redesign`** pour câbler dans la liste
existante plutôt que reconstruire un `ZoneList` divergent sur `main`.

## 3. Réutilisation (rien réinventé)

Le clic réutilise **exactement** le canal de vue que M.I.A Agent pilote déjà :

| Élément réutilisé | Fichier |
|---|---|
| Contrat d'actions `focus_zone` / `highlight_zone` + verrou d'id `coerceViewActions` | `webapp/lib/chart/viewActions.ts` |
| État de vue partagé `ChartViewProvider` / `applyActions` | `webapp/lib/chart/viewState.tsx` |
| Effet de centrage (`setVisibleRange` par span de zone) + surbrillance (anneau bleu) | `webapp/components/app/ReadingChart.tsx` |

Le clic dans la liste émet le **même couple d'actions** que le chatbot
(`focus_zone` + `highlight_zone`), validé par la **même** coercion. Aucun nouveau chemin de
rendu, aucune géométrie envoyée (le contrat interdit déjà les clés géométriques).

## 4. Changements

### `webapp/components/market-reading/sections/ZoneList.tsx`
- 3 props **optionnelles** : `idOf(zone) => string` (l'`id` moteur réel), `onSelect(zoneId)`,
  `selectedZoneId`.
- Quand `idOf` **et** `onSelect` sont fournis, chaque entrée devient un `<button>` accessible
  (clic + clavier, `aria-pressed`, anneau de focus), titré « Localiser cette zone sur le
  graphique » ; l'entrée sélectionnée est surlignée.
- Sinon : rendu **strictement identique** à l'existant (ligne non interactive) → la liste
  reste lisible sans graphe câblé, et les tests isolés ne régressent pas.

### `webapp/components/market-reading/sections/StructureSection.tsx`
- Devient client (`'use client'`), consomme `useChartViewOptional()` (tolérant : no-op hors
  `/app`).
- **Verrou d'id** : `validZoneIds` construit depuis `structure.order_blocks` +
  `fair_value_gaps` — identique au set d'`AppWorkspace`. Un clic ne peut référencer qu'une
  zone réellement émise.
- `selectZone(zoneId)` repasse par `coerceViewActions([focus_zone, highlight_zone], validZoneIds)`
  (défense en profondeur) puis `applyActions`. Id inconnu → action droppée.
- L'entrée sélectionnée **reflète `chartView.highlightZoneId`** (source de vérité unique) :
  surbrillance **persistante-sélection** — elle reste jusqu'au clic suivant.
- `idOf` / `onSelect` / `selectedZoneId` câblés sur les deux `ZoneList` (OB et FVG).

## 5. Comportements clés

- **Hors vue** : pris en charge par l'effet focus existant (`setVisibleRange` encadre le span
  temps/prix de la zone) → une zone hors fenêtre est ramenée à l'écran. Un re-clic sur la
  même zone **bump le nonce** → le centrage se re-déclenche (toujours montrée).
- **Zone non dessinable** (OB `invalidated` / FVG `filled` — que le moteur ne pousse en
  pratique jamais) : no-op gracieux, comme le chemin chatbot.
- **Cohabitation chatbot** : même état de vue ; clic et chatbot écrivent la même cible —
  dernier acteur gagne, pas de désynchronisation.

## 6. Vérifications

| Contrôle | Résultat |
|---|---|
| `tsc --noEmit` | ✅ 0 erreur |
| Vitest (suite complète) | ✅ **277 / 277** (35 fichiers) |
| Tests dédiés click-to-chart | ✅ **7 / 7** (`zone-click-to-chart.test.tsx`) |
| `next build` | ✅ vert (`/app` 12.7 kB) |
| Vérif visuelle live | ⏳ à confirmer par le founder (source `/app` = live backend) |

### Tests ajoutés (`zone-click-to-chart.test.tsx`)
1. Clic OB → centre + surbrillance de **la bonne zone par son id réel**, entrée marquée.
2. Idem pour une entrée FVG (id FVG réel).
3. Sélectionner une autre zone déplace la sélection (une seule entrée `aria-pressed`).
4. Re-clic même zone → **nouvelle commande focus** (nonce incrémenté) = re-cadrage hors vue.
5. **Aucune zone inventée** — seuls les ids émis par le moteur sont focusables.
6. La navigation **ne mute jamais** la structure détectée (égalité JSON avant/après).
7. Reste lisible **sans provider** (entrées rendues, clic = no-op gracieux).

## 7. Discipline

Affichage / navigation uniquement. Aucune création de zone, aucune géométrie transmise,
détection intacte. Staging explicite (pas de `git add -A`), pas de force push. **Push + merge
sur `main` uniquement après confirmation live du founder.**
