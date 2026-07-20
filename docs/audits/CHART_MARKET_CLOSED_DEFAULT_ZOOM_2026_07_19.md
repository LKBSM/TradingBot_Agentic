# Graphique — état « Marché fermé » + plancher de zoom par défaut

**Date** : 2026-07-19
**Branche** : `feat/chart-market-closed-default-zoom` (worktree dédié, depuis `origin/main` = b2516f9, i18n consolidé)
**Périmètre** : affichage / viewport / état de session UNIQUEMENT. Détection SMC intacte.

---

## 1. Problème

1. **Zoom « extrême » à l'ouverture le week-end / marché fermé.** Les chandeliers
   apparaissent démesurés, « collés aux micro-bougies », quand l'amplitude de prix
   de la fenêtre visible est minuscule.
2. **Badge trompeur « EN DIRECT · provisoire ».** L'application ne doit pas prétendre
   être en direct quand le marché au comptant (FX / or) est fermé.

## 2. Diagnostic (lecture seule)

- **Viewport horizontal (temps)** : déjà stable et indépendant du prix. Au 1er
  chargement, `ReadingChart` fixe une **plage logique** de `DEFAULT_VISIBLE_BARS = 90`
  bougies (`setVisibleLogicalRange`), pas un `fitContent()` sur le prix. Week-end
  comme semaine : ~90 bougies. **Ce n'était pas la cause.**
- **Axe des prix (vertical)** : la série chandelier tournait avec l'**auto-échelle
  par défaut** de lightweight-charts (ajustement au min/max des bougies visibles),
  **sans aucun plancher**. Le week-end, les dernières bougies ne bougent quasi plus
  → l'écart de quelques ticks est étiré sur toute la hauteur → chandeliers
  magnifiés = le « zoom extrême » perçu. **Cause racine.**
- **Prix & session** : le prix header vient de `useLatestPrice` → `DailyChange`
  qui porte `priceTs` (epoch de la dernière bougie M15, indépendant du TF affiché).
  **Aucune notion ouvert/fermé** n'existait côté webapp.
- **Badge live** : `ReadingChart` affichait « EN DIRECT · provisoire » dès qu'un
  `livePrice` arrivait (`liveActive`), sans considérer l'état de session.

## 3. Correctifs livrés

### (a) Plancher d'auto-échelle vertical — `ReadingChart.tsx`
- `autoscaleInfoProvider` ajouté sur la série. On récupère la plage ajustée par
  la librairie et, **uniquement** quand elle est plus petite que
  `MIN_VISIBLE_RANGE_FRAC = 0.3 %` du prix médian, on l'élargit symétriquement à
  ce minimum. Une session normale dépasse déjà le plancher → **no-op en semaine**.
- Viewport horizontal 90 bougies **inchangé**. Zoom / pan manuel **inchangés**.
- **Données jamais modifiées** — présentation seule.

### (b) Détection de session — `lib/market-reading/session.ts` (nouveau, pur)
- `isMarketClosed(instrument, { now, priceTs, staleThresholdSec })` :
  - **Crypto** (BTC/ETH…) → toujours ouvert (24/7).
  - **Feed frais** (âge du prix < seuil) → **ouvert**, prioritaire sur le calendrier
    (protège les bords DST autour de la réouverture du dimanche).
  - Sinon **fermé** si **week-end FX** (vendredi ~22:00 → dimanche ~22:00 UTC) **OU**
    **feed périmé** (`MARKET_STALE_THRESHOLD_SEC = 3 h`, capte les jours fériés).
- `isForexWeekend(now)` + `isTwentyFourSevenMarket(instrument)` exportés/testés.
- `useMarketClosed(instrument, priceTs)` : hook client, recalcule chaque ~60 s,
  SSR-safe (retourne `false` avant montage). **Purement descriptif — jamais de
  pronostic de réouverture.**

### (c) Badge « Marché fermé » près du prix — header + chart
- `MarketReadingHeader` : chip sobre neutre « Marché fermé » à côté du prix quand
  fermé (gaté par la prop `marketClosed`, absent sur les samples landing).
- `ReadingChart` : quand `marketClosed`, la pastille neutre « Marché fermé » **remplace**
  « EN DIRECT · provisoire » (jamais les deux ; l'app ne prétend pas être live).
- Câblage : `ReadingColumn` calcule `marketClosed` (via `useMarketClosed`) et le
  passe à `MarketReadingCard` → `MarketReadingHeader` **et** au chart slot.

### (d) i18n — 9 locales
- 2 clés ajoutées sous `app.chart` dans les 9 fichiers (`fr/en/de/es/it/pt/nl/pl/ar`) :
  `marketClosed` + `marketClosedTitle`. Parité stricte vérifiée (parse + présence).
- Titre descriptif générique (valable week-end ET férié), **sans mention de réouverture**.
- ⚠️ Relecture native `ar` recommandée avant prod (traduction ajoutée de bonne foi).

## 4. Tests

- **`session.test.ts`** (nouveau) : crypto 24/7, fenêtre week-end (bords Fri 22:00 /
  Sun 22:00), feed frais qui prime sur le calendrier (DST), férié via garde d'âge,
  gap sous-seuil ne bascule pas.
- **`market-reading-components.test.tsx`** : badge « Marché fermé » présent quand
  fermé / absent quand ouvert.
- **tsc** : 0 erreur. **Suite** : 496/496 verts (le fail unique `claims-cleanup`
  du run parallèle complet = timeout de charge 5 s ; **passe isolé 18/18**).
- **Build** : voir §5.

## 5. Discipline

- Détection SMC **inchangée** (moteur intact) — seuls affichage / viewport / session.
- Staging explicite (pas de `git add -A`), rapport ci-présent, pas de force push.
- Merge sur `main` **seulement après confirmation live** du fondateur.

## 6. Fichiers touchés

| Fichier | Nature |
|---|---|
| `webapp/lib/market-reading/session.ts` | **nouveau** — détection + hook |
| `webapp/lib/market-reading/session.test.ts` | **nouveau** — tests unitaires |
| `webapp/components/app/ReadingChart.tsx` | plancher auto-échelle + prop `marketClosed` + badge |
| `webapp/components/app/ReadingColumn.tsx` | calcul + câblage `marketClosed` |
| `webapp/components/market-reading/MarketReadingCard.tsx` | forward `marketClosed` |
| `webapp/components/market-reading/MarketReadingHeader.tsx` | badge près du prix |
| `webapp/components/market-reading/__tests__/market-reading-components.test.tsx` | 2 tests badge |
| `webapp/messages/{fr,en,de,es,it,pt,nl,pl,ar}.json` | clés `app.chart.marketClosed(+Title)` |

## 7. Hors périmètre (non modifié)

- `components/landing/HeroLive.tsx` (landing marketing) contient aussi « EN DIRECT » —
  laissé tel quel (surface promotionnelle, hors app). À traiter séparément si voulu.
- Chemin prix-live réel (`NEXT_PUBLIC_LIVE_TICK`) : le badge « Marché fermé » prime
  déjà et supprime « EN DIRECT » même si un tick arrivait pendant la fermeture.
