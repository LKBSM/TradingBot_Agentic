# Fix surfaçage BOS/CHOCH + câblage mtf_provider

**Date :** 2026-06-17
**Branche :** `fix/bos-choch-surfacing-and-mtf-provider` (depuis `main`)
**Périmètre :** lecture seule des colonnes/caches déjà calculés + front. **Aucune
détection touchée, aucun recompute.** Clair + sombre.

Deux corrections autorisées explicitement par le founder (gate levé) :

---

## A — Surfaçage BOS/CHOCH (fixe le « sous-surfaçage » du 2026-06-16)

**Problème (diagnostic Mission A) :** le moteur détecte 88 BOS / 40 CHOCH sur 6 combos,
mais le front n'en affichait que ≤1 — il n'existait **pas** de collecteur d'événements BOS/CHOCH
(contrairement à OB/FVG via `collect_zones`) ; seul l'état du **dernier bar** était mappé.

**Correctif (plomberie read-only, symétrique à `collect_zones`) :**
- `collect_structure_events(enriched, idx, max_per_type)` — parcourt la fenêtre et renvoie les
  listes `bos_events` / `choch_events`, **plus récents d'abord**, cappées (`MAX_STRUCTURE_EVENTS=8`,
  env-overridable). Lit UNIQUEMENT les colonnes moteur `BOS_EVENT` (±1 sur les vrais bars de
  cassure), `CHOCH_SIGNAL`, `BOS_BREAK_LEVEL` (niveau réel ; fallback `close`). **Jamais** le
  `BOS_SIGNAL` propagé (que le fix F6 avait prouvé bruyant à ~100 %).
- `MarketReadingStructure.bos_events` / `choch_events` ajoutés au schéma (listes, défaut vide →
  rétro-compatible ; `bos`/`choch` singuliers conservés).
- `_structure_events_to_models` + consommation dans `confluence_signal_to_structure` (les 2
  chemins de retour). Câblé dans l'assembleur (`smc_features["_structure_events"]`), comme `_zones`.
- **Front** : `buildStructureMarkers(structure)` (module pur testable) → markers
  lightweight-charts (flèche ↑ sous le bar pour un break haussier, ↓ au-dessus pour baissier ;
  texte BOS/CHOCH). Un CHOCH **prime** sur un BOS au même bar (un CHOCH est une cassure de
  retournement sur le même bar) → pas de doublon. `ReadingChart` crée le plugin markers à la
  création du chart et `setMarkers(...)` à chaque mise à jour. LWC ignore les markers hors
  fenêtre affichée → dégradation gracieuse. La ligne de prix « BOS/CHOCH actuelle » reste.

Vérif réelle : sur XAUUSD/H1, le collecteur surface **8 BOS** (cap) et **6 CHOCH** vs ≤1 avant.

## B — Câblage `mtf_provider` (mtf_confluence vide en live)

**Problème :** `regime.mtf_confluence` (biais multi-timeframe) était **toujours vide en live** —
aucun `mtf_provider` n'était injecté dans l'assembleur (`bootstrap.py`). C'est pourquoi le
panneau MTF du front lit 3 reads séparés.

**Correctif (lecture cache pure, aucun appel Twelve Data, aucune détection) :**
- `build_cache_mtf_provider(candles_store, lookback)` (module assembleur, testable) → callable
  `(instrument, timeframe)` qui lit les **timeframes supérieurs** au TF courant **depuis le
  cache SQLite** (`get_last_n_candles`, lecture pure) et renvoie leurs OHLC en dicts.
  `candles_to_regime` dérive ensuite le biais via la logique de tendance **existante** du moteur.
  Les TF sans bougies en cache sont omis (dégradation gracieuse).
- Injecté dans `bootstrap.py` : `mtf_provider=build_cache_mtf_provider(candles_store, MTF_BIAS_LOOKBACK)`.

→ `mtf_confluence` est désormais peuplé en live pour tous ses consommateurs (tags, description,
haiku). Le front MTF n'est pas modifié (il continue de marcher) ; il pourra être simplifié plus
tard pour lire ce champ au lieu des 3 reads.

---

## Discipline & vérification
- **Détection jamais touchée** : seules les couches mapper/lifecycle (lecture de colonnes
  produites) + le cache (lecture) + le front sont modifiées.
- **Exposition** : 2 listes descriptives read-only (`bos_events`/`choch_events`) + peuplement
  d'un champ existant (`mtf_confluence`).
- Tests : **+10 Python** (collecteur events ×5, mtf_provider ×3, + mapper exposition ×2),
  **+5 vitest** (markers). Suites : Python 124 verts (zone détection/mappers/candles/news),
  vitest **180 verts**, `tsc` clean, `next build` vert. **0 régression.**
- Pas de `git add -A`.

### Fichiers touchés
```
src/intelligence/market_reading_schema.py            (A : bos_events/choch_events)
src/intelligence/market_reading_mappers.py           (A : collect_structure_events + helper + consume)
src/intelligence/market_reading_assembler.py         (A : wiring _structure_events ; B : build_cache_mtf_provider)
src/api/bootstrap.py                                 (B : injection mtf_provider)
tests/test_market_reading_mappers.py                 (A : +5)
tests/test_market_reading_assembler.py               (B : +3)
webapp/types/market-reading.ts                       (A : bos_events/choch_events)
webapp/lib/chart/structureMarkers.ts                 (A : builder pur, nouveau)
webapp/lib/chart/__tests__/structureMarkers.test.ts  (A : nouveau)
webapp/components/app/ReadingChart.tsx               (A : plugin markers)
```
