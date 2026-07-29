# AUDIT — TF-1 : Parité complète des six unités de temps

Date : 2026-07-28 · Branche : `feat/tf-1-parite-unites-temps` (depuis `main` @ f4c7252)

Ligne : mission de cohérence. Aucune règle de détection modifiée. Aucun libellé
prédictif introduit. Une unité sans donnée affiche l'absence, jamais un repli
silencieux.

---

## 1. Cause supprimée : les énumérations dispersées

La liste des unités de temps était recopiée à ~15 endroits. Chaque nouvelle unité
rouvrait les mêmes trous. **Source unique désormais : `config/timeframes.json`**
(lue par `src/intelligence/timeframe_registry.py` au backend ; le front consomme
un module GÉNÉRÉ `webapp/lib/timeframes.generated.ts` via
`scripts/gen_timeframes.mjs`, plus `webapp/lib/timeframes.ts` pour les helpers).

### Tableau — avant → après

| Fichier | Avant (énuméré en dur) | Après |
|---|---|---|
| `lookback_config._TF_MINUTES` / `supported_timeframes` | dict + clés depuis depths | `timeframe_registry.minutes_map()` / `perimeter_ids()` |
| `market_reading_assembler._TIMEFRAME_MINUTES` | dict 8 clés | `registry.minutes_map()` |
| `market_calendar._timeframe_minutes` | import assembler | `registry.minutes()` |
| `volatility_forecaster.TIMEFRAME_MINUTES` | dict 8 clés | `registry.minutes_map()` |
| `twelve_data_provider._TIMEFRAME_MAP` | dict → strings TD | `registry.provider_map()` |
| `conditions_scan._TF_MINUTES` (route) | dict | `registry.minutes_map()` |
| `conditions_scanner._TF_MINUTES` (moteur) | `{M15,H1,H4}` | `registry.minutes_map()` |
| `data_quality.TIMEFRAME_MINUTES` | dict 8 clés | `registry.minutes_map()` |
| `chatbot.SUPPORTED_TIMEFRAMES` / `view_action_filter` | `(M15,H1,H4)` | `lookback_config.enabled_timeframes()` |
| `market_reading_schema.VALID_MTF_KEYS` | `{m15,h1,h4,d1,w1}` | dérivé du registre (périmètre∪référence) |
| `ReadingChart.TF_SECONDS` | **`{M15,H1,H4}`** | `lib/timeframes.TF_SECONDS` (les 6) |
| `mockReadings.INTERVAL_SECONDS` | `{M15,H1,H4}` | `TF_SECONDS` |
| `formatters.TIMEFRAME_LABEL` | dict FR | `TF_LABEL_LONG` |
| `candle-clock` / `regime-facts` `TIMEFRAME_MINUTES` | dicts | `TF_MINUTES` |
| `mtf-trend.MTF_TREND_ORDER` | `[h4,h1,m15]` fixe | `mtfOrderFor(tf)` (relatif) |
| `ComboCard` (scanner) triplet MTF | `[h4,h1,m15]` | `mtfOrderFor(match.timeframe)` |

**Garde anti-régression** : `tests/test_timeframe_guard.py` (backend) +
`webapp/lib/__tests__/timeframes-guard.test.ts` (front) échouent si une carte
d'unités réapparaît hors du registre, et vérifient la synchro du module généré.
Les gardes ont démasqué 3 oublis (conditions_scanner, data_quality, ComboCard),
depuis corrigés.

---

## 2. Cause exacte du symptôme (clic zone → cadrage échoue sur M5/D1)

`components/app/ReadingChart.tsx` : `TF_SECONDS = { M15, H1, H4 }` seulement.
`barSec = TF_SECONDS[tf] ?? 0` → **0 pour M5/D1** → **early return** avant
`frameZone` (l'effet de cadrage retournait sur `barSec <= 0`). Sur M15/H1 `barSec>0`
→ cadrage OK. La même table cassait aussi le bucketing de la bougie live et les
bougies mock. **Corrigé** : `TF_SECONDS` vient du registre → les 6 unités ont une
valeur > 0. Garde dédiée : `TF_SECONDS[tf] > 0` pour chaque unité du périmètre.

Changement d'unité avec zone sélectionnée : la sélection est par id de zone (verrou
d'id) ; l'unité changée charge d'autres zones, l'id obsolète ne correspond plus →
la surbrillance n'est pas appliquée (abandon naturel, jamais mal-appliquée).

---

## 3. Règle d'alignement retenue (décision C)

**Unités AU-DESSUS de celle regardée**, relatives : M5→M15/H1/H4/D1 ; H1→H4/D1 ;
H4→D1 ; D1→W1 (unité de référence au-dessus du périmètre) ; au-dessus de D1, aucune
→ « aucune unité supérieure ». Implémenté par `timeframe_registry.alignment_timeframes`
(backend) et `alignmentTimeframes` (front). Les unités comparées sont **nommées** ;
une unité sans donnée est comptée **indisponible** (dénominateur ajusté, glyphe ·),
**jamais un accord**. Libellés strictement descriptifs (aucun « fort/favorable »).

---

## 4. Décisions de pertinence par unité (décision D)

Drapeaux dans le registre (`sessionRelevant`, `prevLevelsRelevant`) :
- **Session** et **niveaux « veille »** : `false` sur **D1/W1** (la bougie couvre
  toute la journée) → tuiles **masquées AVEC mention** (« non applicable sur D1 :
  la bougie couvre toutes les sessions » / « la veille est la bougie précédente »).
  Jamais de disparition silencieuse.
- **Maturité** : inchangée — bougies + temps écoulé affichés (déjà le cas), lisible
  sur M1 (55 bougies = « ≈ 55 min » en parallèle).

---

## 5. Quatre points du fondateur

1. **Pourcentages signés — corrigé.** Les deux blocs `dist` (Structure + Liquidité,
   fr+en) rendaient `+/−{pct} %` (signés et redondants). Mots seuls partout ;
   **test de garde** `no-signed-percent` échoue si un % signé revient sur une
   surface de lecture (scopé pour ne pas attraper une remise commerciale légitime).
2. **Deux comptes de zones — rapprochés.** En-tête « N sur M zones · K actives »,
   Densité « … FVG actifs ». Total = actives + consommées, désormais lisible.
3. **Panneau Structure — porte son unité.** `StructureCard` reçoit `timeframe` et
   l'affiche (badge), comme Régime.
4. **Événements vs tendance — rendu lisible.** Les événements portent l'unité
   regardée (via le badge Structure) ; l'alignement porte les unités supérieures
   nommées (via C). Aucune touche détection — la divergence était légitime (unités
   différentes), elle est maintenant explicite à l'écran.

---

## 6. Ajouter une septième unité de temps

Critère de réussite de la mission — **une seule édition** :
1. Ajouter l'entrée dans `config/timeframes.json` (id, minutes, provider, libellé,
   format date, `perimeter`/`reference`, `sessionRelevant`, `prevLevelsRelevant`).
2. Régénérer le module front : `node scripts/gen_timeframes.mjs`.
3. Le cas échéant, ajouter sa profondeur dans `config/lookback_depths.json`.
Tout le reste (minutes, secondes, provider, libellés, échelle d'alignement,
pertinence, périmètres scanner/chatbot/API) en dérive automatiquement. Les tests
de garde vérifient qu'aucune énumération n'a été recopiée ailleurs.

---

## 7. Tests

- Backend : `test_timeframe_registry.py` (24), `test_timeframe_guard.py` (garde),
  suites lookback/calendar/scanner/chatbot/volatility/candles/market_reading vertes.
- Front : `mtf-trend` + `RegimeSection` (modèle relatif), `no-signed-percent`
  (garde E1), `timeframes-guard` (sync + symptôme + anti-carte), ui2c/ui3a mis à jour.
- Playwright 1280×800 / 390×844 sur M1/M5/D1 : voir le run de clôture (nécessite un
  backend/mock couvrant M5/D1 — sinon validé live par le fondateur avant merge).
- `tsc` + `build` verts.
