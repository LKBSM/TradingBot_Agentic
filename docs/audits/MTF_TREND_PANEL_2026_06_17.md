# Panneau « Régime de marché » → Alignement de tendance multi-TF

**Date :** 2026-06-17
**Branche :** `feat/mtf-trend-panel` (basée sur `feat/chart-refinements`, tip front portant `/app`)
**Périmètre :** front uniquement + **lecture read-only** des reads existants des 3 TF.
Aucune détection touchée, aucune tendance recalculée. Clair + sombre.

---

## Pourquoi

Le panneau collapsible « 🌊 Régime de marché » (`RegimeSection`) répétait la même info que
le hero (`MarketPhasePanel`) : badge « Tendance baissière » + badge « Phase de tendance »
disent la même chose. Remplacé par un résumé d'**alignement de tendance multi-timeframe**
(M15 / H1 / H4) + une **ligne descriptive** caractérisant la relation entre les TF.

> Le hero `MarketPhasePanel` (tendance/vol/phase) reste **hors périmètre** — la mission visait
> nommément le panneau « Régime de marché ».

## Confirmation (étape 1) — rappel

- Le panneau vit dans `webapp/components/market-reading/sections/RegimeSection.tsx`.
- La tendance du **TF affiché** est dans `reading.regime.trend`. Mais les **3 TF M15/H1/H4
  d'un coup ne sont PAS disponibles dans un seul read en live** : `regime.mtf_confluence`
  existe mais **aucun `mtf_provider` n'est câblé** (`bootstrap.py`) → vide en prod
  (`READING_DATA_SOURCE = 'live'`) ; seuls les mocks le peuplent, et partiellement.
- Source retenue : **lecture des 3 reads existants** de l'instrument courant
  (`fetchMarketReading(instr, 'M15'|'H1'|'H4')`), chacun fournissant son propre
  `regime.trend`. Lecture seule de reads déjà calculés — **aucune nouvelle détection**.

## Implémentation

**A. Affichage (3 TF d'un coup d'œil).** Trois badges colorés par tonalité dans l'ordre
descendant : `H4 ↗ · H1 ↗ · M15 ↘` (↗ haussier, ↘ baissier, → neutre/range, · indisponible).
Valeurs issues des reads existants, jamais recalculées.

**B. Ligne descriptive (classification pure des valeurs existantes).** Fonction
`describeMtfAlignment` — **présent strict**, décrit ce qui EST observé :
- 3 identiques non-plats → « Les 3 TF sont alignés (haussiers). »
- tous neutres → « Les 3 TF sont neutres. »
- H4+H1 d'accord, M15 opposé → « M15 se replie contre la tendance H4 haussière. »
- sinon → « Les TF divergent : H4 haussier, H1 neutre et M15 baissier. »
- gère la disponibilité partielle (n<3) et l'absence totale (ligne masquée).

Aucune formulation interdite : pas de futur/probabilité (« va », « a tendance à »,
« attends-toi »), pas de score (« 70 % », « setup fort »), pas de verdict d'action
(« évite », « bon moment », « risqué »). Un test verrouille l'absence de ce vocabulaire.

**C. Disclaimer.** Conservé, équivalent à l'actuel : « Cet alignement décrit l'état observé
des timeframes. Il ne constitue pas une instruction adressée au trader. »

**Volatilité.** Badge **conservé** (info distincte, non redondante — on superpose, on
n'ampute pas la sortie algo).

### Architecture (lecture seule, fetch paresseux)
- `lib/market-reading/mtf-trend.ts` — **module pur** : `MtfTrendMap`, `mtfTrendGlyph`,
  `describeMtfAlignment`. Aucune dépendance réseau → unit-testable.
- `lib/market-reading/hooks.ts` — `useMtfTrends(instrument)` : fetch en parallèle des 3
  reads M15/H1/H4 (live `fetchMarketReading`, ou `getMockReading` en mock), prend chaque
  `regime.trend`, dégrade en `null` si un TF manque. Aucune détection, aucun recompute.
- `RegimeSection.tsx` — le corps fetché vit dans un sous-composant **interne au
  `AccordionContent`** : Radix démonte le contenu replié, donc le fetch ne part **qu'à
  l'ouverture** du panneau (et pas dans les tests/rendus où il est replié).
- `MarketReadingSections.tsx` — passe `instrument={reading.header.instrument}`.

## Discipline & vérification
- **Front uniquement** + lecture read-only de 3 reads existants. **Détection jamais touchée.**
- Tests : **+15** (`mtf-trend` 8, `RegimeSection` 7) ; suite vitest **175 verte** (25 fichiers),
  **0 régression**. `tsc --noEmit` clean. `next build` vert (`/[locale]/app` inchangé côté poids).
- Clair + sombre : tons via le système `Badge` (variants bull/bear/neutral) résolus par thème.
- Pas de `git add -A`.

### Fichiers touchés
```
webapp/lib/market-reading/mtf-trend.ts                         (nouveau, pur)
webapp/lib/market-reading/hooks.ts                             (+ useMtfTrends)
webapp/lib/market-reading/__tests__/mtf-trend.test.ts          (nouveau)
webapp/components/market-reading/sections/RegimeSection.tsx    (réécrit)
webapp/components/market-reading/sections/__tests__/RegimeSection.test.tsx (nouveau)
webapp/components/market-reading/MarketReadingSections.tsx     (+ instrument)
```

### Note de suivi (non bloquante)
`mtf_confluence` n'est pas alimenté en live (pas de `mtf_provider` câblé). Le panneau lit
donc 3 reads séparés. Si un `mtf_provider` est câblé un jour côté backend, `useMtfTrends`
pourra être simplifié pour lire un seul read — sans changer l'UI ni la détection.
