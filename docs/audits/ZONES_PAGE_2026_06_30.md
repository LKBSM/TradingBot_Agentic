# Page « Zones » — cycle de vie des zones détectées

**Date** : 2026-06-30
**Branche** : `feat/zones-page` (worktree dédié, depuis `main` consolidé `62e368c`)
**Statut** : implémenté, tsc + 377 tests verts, build OK, preuve visuelle capturée. En attente de confirmation live avant push/merge.

---

## 0. Objectif

Nouvelle surface d'usage quotidien `/[locale]/zones` : la VIE de chaque zone détectée
(Order Block / Fair Value Gap) — formation, tests, mitigation, comblement, état courant —
présentée en **timeline de cycle de vie** + une **phrase de narration factuelle**, avec deux
actions par carte : **« Analyser → »** (focalise la zone sur le graphe) et **« Masquer du
graphique »** (masque cette zone précise).

Ligne inviolable respectée : **strictement descriptif**, passé/présent factuel, **aucune
prédiction / cible / conviction**. Lecture seule sur la détection — on affiche le cycle déjà
produit, on n'invente jamais un événement, et on dégrade proprement toute donnée absente.

## 1. Diagnostic (lecture seule) — résultats

### Données de cycle de vie réellement disponibles (par zone)
Source = `GET /api/market-reading` (`response_model=MarketReading`), **même source que l'App**
(hook `useMarketReading`). Zones synthétisées dans `market_reading_mappers.py`, schéma
`market_reading_schema.py`.

| Donnée | OB | FVG | Note |
|---|---|---|---|
| id réel (déterministe) | ✅ | ✅ | `OB/FVG_{dir}_{created:%Y%m%d%H%M%S}` — stable par combo/TF, sert de verrou d'id |
| direction | ✅ | ✅ | optionnelle |
| range (level_high/low) | ✅ | ✅ | |
| importance | ✅ 3 paliers | ❌ **absente** | FVG n'a aucun champ importance |
| état | ✅ active/mitigated | ✅ active/partially_filled/filled | `invalidated` OB droppé en amont |
| `created_at` (formation) | ✅ | ✅ | |
| **compte de tests « ×N » + historique** | ❌ | ❌ | **booléen `tested` + 1er contact uniquement** — jamais affiché « ×N » |
| `mitigated_at` | ✅ (1er contact) | ✅ (1re entrée) | seul horodatage hors `created_at` |
| `fill_level` | — | ✅ **prix** (pas un %) | barre dérivée géométriquement |
| `broken_at` | ❌ | ❌ | n'existe que pour BOS/CHOCH/Liquidity |

### Masquage-par-id & focus
- **Déjà mergés dans `main`** (`feat/hide-specific-zone`) : `hide_zones`/`show_zones` +
  verrou d'id `coerceZoneIdList` (id inexistant → action entièrement rejetée), et `focus_zone`
  (id validé contre les zones à l'écran). **Aucune dépendance de branche bloquante.**
- **État de vue NON partagé entre pages** : `ChartViewProvider` n'était monté que dans
  `AppWorkspace`. → décision : **le hisser dans `[locale]/layout.tsx`** pour qu'une action
  prise sur `/zones` se reflète sur le graphe de `/app`.

## 2. Implémentation

### Cœur descriptif — `webapp/lib/zones/lifecycle.ts` (pur, testable)
- `collectZones(structure)` : projette OB+FVG en `ZoneLifecycle` (champs moteur 1:1 + booléens
  dérivés, **aucun fait nouveau**).
- `buildTimeline(zone)` : n'émet QUE les événements réellement tracés. `Formé` toujours présent ;
  `Testé`/`Pénétré` seulement si `tested` (1 seul, jamais « ×N ») ; `Mitigé`/`Comblé` terminal
  selon le statut ; `Suivi en cours` si actif. Un événement sans horodatage (ex. FVG `filled`
  sans « filled_at ») s'affiche **sans date inventée**.
- `fillFraction(zone)` : fraction comblée **dérivée géométriquement** du prix `fill_level` réel
  et des bornes — direction-aware (haussier comble par le haut, baissier par le bas), bornée
  [0,1], `null` si pas de `fill_level` (jamais de pourcentage inventé).
- `narrateZone(zone, instrument)` : une phrase factuelle (présent/passé), composée des seuls
  faits moteur. Zéro lexique prédictif/directif (testé).
- `matchesFilter` / `sortZones` : filtres Toutes/Actives/Mitigées, tris importance/récence/proximité.

### UI — `webapp/components/zones/`
- `ZonesWorkspace.tsx` : sélecteur combo/TF (`perimeter.ts`), filtres + tri, `useMarketReading`
  (poll 60 s, **lecture seule**), `useChartView()` pour le masquage partagé.
- `ZoneLifecycleCard.tsx` : badge type+sens, range, importance (OB), état, `ZoneTimeline`, barre
  de remplissage FVG (si `partially_filled`), phrase factuelle, boutons **Analyser** / **Masquer**.
- `ZoneTimeline.tsx` : timeline verticale, n'affiche que les événements reçus.
- `webapp/app/[locale]/zones/page.tsx` : route + `SubscriptionGate` (comme `/app`).

### Câblage (réutilisation, pas de 2ᵉ implémentation)
- **État partagé** : `ChartViewProvider` hissé dans `[locale]/layout.tsx`, retiré d'`AppWorkspace`
  (2 tests de `AppWorkspace` enveloppés en conséquence). `/zones` et `/app` partagent
  `hiddenZoneIds` ; un masquage sur `/zones` se reflète sur le graphe.
- **Analyser →** : `buildAppHref(locale, combo, zoneId)` ajoute `?focus=<id>` ; `/app` lit le
  param et dispatch `focus_zone` une fois la lecture chargée, **re-validé par le verrou d'id**
  (id inconnu/périmé → no-op).
- **Masquer** : route par `coerceViewActions` contre l'ensemble des ids à l'écran → un id
  inexistant masque **rien**. Réversible (le bouton bascule Masquer/Afficher). Affichage seul.

## 3. Tests (Vitest)

- `lib/zones/__tests__/lifecycle.test.ts` — collectZones, timeline (événements réels + dégradation
  propre, jamais « ×N »), fillFraction (direction-aware + null sans data), filtres/tri, narration
  **factuelle et sans lexique prédictif**.
- `components/zones/__tests__/ZonesWorkspace.test.tsx` — (a) cartes avec vraies données ; (b)
  donnée absente → dégradation (pas de barre `progressbar` sans `fill_level`, pas d'étape `Testé`
  inventée) ; (c) « Analyser » → bon `focus=<id>` ; (d) « Masquer » reflété dans l'état partagé +
  réversible ; (e) id inexistant rejeté ; filtre Mitigées.
- `components/app/__tests__/zone-focus-deeplink.test.tsx` — « Analyser » focalise bout-en-bout la
  bonne zone ; id inconnu = no-op gracieux.

**Résultat** : `tsc --noEmit` 0 erreur · **43 fichiers / 377 tests verts** (0 régression) · `next build` OK
(route `/[locale]/zones` 6.36 kB / 131 kB) · preuve visuelle (`zones_shot.png`, `zones_card_shot.png`).

## 4. Discipline
Affichage / navigation / état de vue uniquement. **Détection et cycle de vie inchangés** (lecture
seule). Aucune sortie prédictive. Staging explicite (pas de `git add -A`), pas de force push.
Push/merge sur `main` **uniquement après confirmation utilisateur du rendu live**.
