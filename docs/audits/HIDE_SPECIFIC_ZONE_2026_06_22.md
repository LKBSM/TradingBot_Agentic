# View-control — masquer / isoler une zone SPÉCIFIQUE par son id (réversible)

**Date** : 2026-06-22
**Branche** : `feat/hide-specific-zone` (worktree isolé `C:/MyPythonProjects/TradingBOT_hide_zone`)
**Périmètre** : AFFICHAGE uniquement. Détection INCHANGÉE.

---

## 1. Objet

M.I.A Agent pouvait masquer une **couche** entière (tous les OB / FVG) ou filtrer par
critères (actives seules / proximité / taille), mais pas masquer ni isoler **UNE zone précise**.
On ajoute trois actions à la liste blanche fermée :

| Action | Params | Effet (vue seule) |
|---|---|---|
| `hide_zones` | `{zone_ids: string[]}` | retire ces zones réelles de l'affichage |
| `isolate_zones` | `{zone_ids: string[]}` | n'affiche QUE ces zones réelles (masque le reste) |
| `show_zones` | `{zone_ids?: string[]}` | ré-affiche ; sans `zone_ids` → tout restaurer |

## 2. Ligne inviolable — respectée

- **Zones réelles uniquement.** Chaque id est revalidé contre les zones réellement émises par
  le moteur ce tour (`known_zone_ids` backend / `validZoneIds` frontend) — **le même verrou que
  `focus_zone` / `highlight_zone`, réutilisé tel quel**, généralisé à une liste. Si **un seul**
  id est inventé, l'action entière est rejetée (`unknown_zone_id`) et **rien n'est masqué**.
- **« l'OB à 4160 ».** L'agent lit `get_market_reading`, trouve la zone réelle dont la bande
  contient 4160, masque/isole **son id**. Si aucune zone réelle ne correspond → refus on-brand,
  rien masqué (guidance prompt + verrou d'id en backstop).
- **Masquer = retirer de l'AFFICHAGE**, pas fabriquer ni supprimer une structure. La zone reste
  dans le moteur ; l'état de masquage vit dans l'état de **VUE**.
- **Vocabulaire fermé + garde géométrie INCHANGÉS.** Aucun verbe create/place/move/resize n'est
  représentable ; `zone_ids` n'est pas une clé géométrique → la garde `GEOMETRY_KEYS` continue de
  rejeter tout `price/level/high/low/...` qui voyagerait avec l'action.
- **Réversible.** `show_zones` (par ids ou global) et `reset_view` restaurent. Rien masqué par
  défaut (`hiddenZoneIds: []`, `isolatedZoneIds: null`).

## 3. Implémentation

### Backend (`src/intelligence/chatbot/`)
- `view_action_filter.py` : 3 actions ajoutées à `ALLOWED_ACTIONS` ; nouveau `_ZONE_REFS_ACTIONS`
  et helper `_zone_refs()` (liste d'ids, dédup, rejet si un id inconnu, `allow_empty` pour
  `show_zones`). `_v_hide_zones` / `_v_isolate_zones` / `_v_show_zones`.
- `chatbot.py` : description de l'outil `apply_chart_view` + guidance système (résolution
  « l'OB à 4160 » → id réel, refus si aucune correspondance). Dispatcher inchangé : la même
  validation `validate(..., known_zone_ids=...)` couvre les nouvelles actions.

### Frontend (`webapp/lib/chart/`, `webapp/components/app/`)
- `viewActions.ts` : types `ViewAction` étendus ; `coerceZoneIdList()` (miroir de `_zone_refs`) ;
  cas `hide_zones`/`isolate_zones`/`show_zones` dans `coerceViewAction` ; `ChartViewState` gagne
  `hiddenZoneIds: string[]` et `isolatedZoneIds: string[] | null` ; réducteur
  `applyChartViewAction` (union/replace/restore) ; `reset_view` restaure déjà tout.
- `zoneLayout.ts` : `applyZoneVisibility(zones, hiddenZoneIds, isolatedZoneIds)` — passe pure,
  appliquée après `filterZoneModels`, avant `curateZones`.
- `ReadingChart.tsx` / `ReadingColumn.tsx` : propagation des deux nouveaux champs de l'état de vue.

## 4. Tests (verts)

- **Backend** `tests/test_chatbot_view_actions.py` : **31 passed**. Couvre : id réel accepté ;
  un id inventé rejette toute l'action ; dédup ; liste vide rejetée (hide/isolate) ;
  `show_zones` sans ids = restaurer ; garde géométrie sur les actions de masquage ;
  intégration chatbot (masquer une zone réelle, refus on-brand sur id inventé, roundtrip
  isolate→show) ; détection jamais mutée.
- **Frontend** `webapp/lib/chart/__tests__/viewActions.test.ts` : **37 passed**. Couvre :
  coercion hide/isolate/show (dédup, rejet d'id inventé, non-array, restore) ; réducteur
  (union/replace/restore + ré-ajout sous isolation) ; `applyZoneVisibility` (hide réversible,
  isolate, composition hide∩isolate, non-mutation).
- **TypeScript** : `tsc --noEmit` (voir statut en pied de PR).

## 5. Discipline

Travail réalisé dans un **worktree git isolé** (`feat/hide-specific-zone`) après constat que le
répertoire principal était partagé par plusieurs terminaux actifs qui basculaient de branche et
écrasaient les éditions non commitées. Staging explicite (pas de `git add -A`), pas de force push,
pas de PR.
