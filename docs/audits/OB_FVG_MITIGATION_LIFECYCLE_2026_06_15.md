# Cycle de vie / mitigation des OB & FVG — couche additive

**Date :** 2026-06-15
**Branche :** `feat/ob-fvg-mitigation-lifecycle`
**Type :** additif — la DÉTECTION/FORMATION des OB/FVG reste **strictement inchangée**
(`src/environment/strategy_features.py` non touché).

---

## 1. Contexte

Une couche de cycle de vie *partielle* existait déjà (commit `12ac3db`,
`market_reading_mappers.py`) : chaque OB/FVG formé sur la fenêtre était classé et
les zones consommées (OB invalidé, FVG comblé) étaient retirées. Trois manques vs
la mission :

1. la règle de mitigation était codée **inline**, sans source de vérité nommée ;
2. aucun **timestamp de consommation** n'était exposé → le front ne pouvait borner
   la boîte qu'à `formation → prix courant`, jamais à `formation → point de mitigation` ;
3. les défauts n'étaient pas formalisés comme « à valider par annotation » ni leur
   biais conservateur.

Cette livraison comble ces trois points **sans modifier la géométrie des zones**.

---

## 2. Définitions de mitigation RETENUES

> ⚠️ **DÉFAUTS À VALIDER PAR ANNOTATION** — ce sont des règles de *surfaçage*
> provisoires, pas une définition de détection. La géométrie de chaque zone vient
> du moteur et n'est pas touchée ; on ne décide ici que **quand** une zone formée
> est considérée touchée (mitigée / partiellement comblée) ou consommée
> (invalidée / comblée, donc retirée). Biais conservateur : en cas de doute, on
> déclare mitigé **plus tôt**, et on n'affiche **jamais** une zone consommée comme active.

Validé par le founder le 2026-06-15 :

### Order Block
| Statut | Règle retenue (défaut) | Exposition |
|---|---|---|
| **invalidated** | une bougie **clôture à travers** le bloc (close < zlow bullish ; close > zhigh bearish) | **retiré** (jamais exposé) |
| **mitigated** | le prix **touche** le bloc (overlap mèche) sans clôture à travers | **gardé, tagué** `mitigated` + `mitigated_at` = 1er tap |
| **active** | le prix n'est jamais revenu dans le bloc | exposé, `mitigated_at = None` |

- **Alternative annotation-gated :** exiger une **pénétration ≥ X %** de la hauteur
  du bloc (`ob_mitigation_penetration`, défaut `0.0` = toute touche, le plus
  conservateur) ; ou **retirer** les OB mitigés (`ob_drop_when_mitigated`, défaut
  `False`).

### Fair Value Gap
| Statut | Règle retenue (défaut) | Exposition |
|---|---|---|
| **filled** | comblement **100 %** (le prix atteint l'arête lointaine) | **retiré** (jamais exposé) |
| **partially_filled** | le prix entre par l'arête proche sans atteindre l'arête lointaine | **gardé, tagué** + `mitigated_at` = 1ère entrée |
| **active** | le prix n'a pas touché le gap | exposé, `mitigated_at = None` |

- **Alternative annotation-gated :** abaisser `fvg_fill_fraction` (défaut `1.0` =
  arête lointaine) pour retirer les gaps plus tôt ; ou **retirer** dès la première
  entrée (`fvg_drop_when_partial`, défaut `False`).

---

## 3. Mécanisme

- **Source de vérité unique :** `MitigationPolicy` (dataclass frozen) +
  l'instance `MITIGATION_POLICY` dans `src/intelligence/market_reading_mappers.py`.
  Tous les seuils (pénétration OB, fraction de comblement FVG, retirer-vs-taguer)
  y vivent et sont documentés. Aucun seuil épars.
- À chaque bougie, `collect_zones()` rejoue chaque OB/FVG formé sur la fenêtre via
  `_ob_lifecycle()` / `_fvg_lifecycle()`, qui consomment la policy et renvoient
  désormais l'**index de première interaction** → converti en `mitigated_at`.
- **Garde-fou honnêteté :** les zones consommées (invalidated / filled) sont
  retirées ; en option stricte, les zones mitigées/partielles aussi. Une zone
  consommée n'est **jamais** exposée comme active.

## 4. Ce qui est exposé au front

- Schéma `OrderBlock` / `FairValueGap` (`market_reading_schema.py`) : nouveau champ
  **`mitigated_at: Optional[datetime] = None`** (rétro-compatible, optionnel).
- Type TS miroir `webapp/types/market-reading.ts` : `mitigated_at?: string | null`.
- Le front peut désormais dessiner des **boîtes bornées** :
  `created_at → mitigated_at` si la zone a été touchée, sinon `created_at → prix courant`.
- **Contrat strictement descriptif** : aucun champ prédictif ajouté.

## 5. Tests & vérifications

- `tests/test_market_reading_mappers.py` : +9 tests (mitigated_at actif/mitigé,
  bias conservateur, knob de pénétration, drop-policy, défauts founder, schéma).
  49/49 verts.
- Suites liées : 148/148 verts (mappers, schema, assembler, pipeline, scanner,
  insight_v2). 0 régression.
- Webapp : `npm run build` vert (type-check inclus), vitest market-reading 27/27.

## 6. Hors scope (volontaire)

- Détection/formation OB/FVG : intouchée.
- Rendu des boîtes côté chart (branche `feat/chart-direction1-ux`) : la donnée est
  exposée, le dessin reste un travail front séparé.
- Calibration finale des seuils : **gated annotation** (audit §4/§5).
