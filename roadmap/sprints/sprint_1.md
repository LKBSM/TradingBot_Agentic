# Sprint 1 — Data Layer Hardening

**Période** : Semaines 3-4 (S3-S4, ~2026-05-30 → 2026-06-13)
**Charge estimée totale** : **104 h** productives (~52 h/sem) + buffer 12 h = 116 h
**Objectif** : transformer le data layer d'un pipeline "CSV + glue code" vers un contrat strict (Pydantic v2), prouver l'absence de look-ahead MTF par property-based testing, livrer un pipeline calendrier économique end-to-end, et faire passer le coverage CSV des 4 actifs MVP supplémentaires à ≥ 90 %. Extraction physique `smart_money/` reportée Sprint 6 (façade créée en parallèle).
**Gate de sortie** : 6 actifs avec coverage ≥ 90 %, contrat Pydantic v2 actif, 0 look-ahead MTF prouvé, pipeline calendrier 2026 chargeable, suite tests verte (≥ 1 366 + ~80 nouveaux tests).

---

## 0. Vue d'ensemble — 6 batches

| Batch | Titre                                                        | Heures | Critique chemin |
| ----- | ------------------------------------------------------------ | ------ | --------------- |
| 1.0   | Façade `smart_money/` + fix 5 P0/P1 ICT (extraction physique Sprint 6) | 0 h ✅ Done parallèle | — |
| 1.1   | DataProvider Pydantic v2 + ingestion contractuelle           | 26 h   | ✅              |
| 1.2   | Resampler MTF property-based no-look-ahead (fix `<=`→`<` déjà appliqué) | 14 h   | ✅              |
| 1.3   | Pipeline calendrier économique 2026 + blackouts end-to-end   | 22 h   | ✅              |
| 1.4   | Décision sources licenciées (différée — go/no-go documentaire) | 6 h  |                 |
| 1.5   | Étendre CSV propres BTCUSD / US500 / GBPUSD / USDJPY         | 36 h   | ✅              |
| —     | Buffer (debug Windows, dépendances, review)                  | 12 h   |                 |
| **TOTAL** |                                                          | **116 h** |             |

Note : batch 1.0 (façade smart_money + bugs RSI Div + magic number retest) est livré **en parallèle** par la session précédente. L'extraction physique du package est repoussée Sprint 6.6.

---

## Batch 1.0 — Façade `smart_money/` ✅ Done en parallèle

### Statut
- Façade `src/intelligence/smart_money/` créée avec `__init__.py` ré-exportant les fonctions critiques depuis `strategy_features.py`.
- Bug **RSI Divergence indexage** (`strategy_features.py:849-857`) corrigé + test régression.
- Bug **magic number retest** (`armed_window=5` vs `RETEST_ARMED_WINDOW=30`) normalisé sur la constante module.
- 4 tests dédiés ajoutés (`tests/test_smart_money_facade.py`).

### Findings audit adressés
- **P0-9** (Smart Money pas extrait) — partiel (façade, pas extraction physique → P0-9 reste ouvert pour Sprint 6.6).
- **P0-15** (Bug RSI Divergence) — ✅ closed.
- **P1-1** (Magic number retest incohérent) — ✅ closed.

### Restant
- Extraction physique BOS/CHOCH/OB/FVG/retest depuis `strategy_features.py` vers `smart_money/{bos.py, choch.py, ob.py, fvg.py, retest.py}` → reportée **Sprint 6 batch 6.5** (audit 1700 LOC, risque régression élevé, on attend les annotations Sprint 2 pour valider l'équivalence).

---

## Batch 1.1 — DataProvider Pydantic v2 + ingestion contractuelle (26 h)

### Objectif
Imposer un contrat unique d'ingestion OHLCV à travers le pipeline. Toute donnée qui entre dans `intelligence/` passe par un `DataFrameContract` Pydantic v2 (timezone, dtype, monotone, NaN, gaps) avant d'atteindre les détecteurs.

### Steps
1. **Audit usages actuels** (3 h)
   - `Grep "pd.read_csv|read_parquet|read_feather"` dans `src/intelligence/`, `src/environment/`, `src/agents/`.
   - Lister les 15-20 call-sites, classer par criticité (prod scanner / backtest / training / R&D).
   - Sortie : `audits/2026-Q2/data_provider_usages.md`.

2. **Schéma Pydantic v2** (4 h)
   - `src/intelligence/data_layer/contracts.py` :
     - `OHLCVBar` (open, high, low, close, volume, timestamp UTC tz-aware) avec `model_config = ConfigDict(strict=True, frozen=True)`.
     - `OHLCVFrame` wrapper avec validators : monotone strict, fréquence cohérente, pas de NaN, low ≤ open/close ≤ high.
     - `InstrumentMetadata` (symbol, tf, pip_value, price_decimals, sessions).

3. **`DataProvider` interface unifiée** (5 h)
   - `src/intelligence/data_layer/provider.py` (Protocol + impls).
   - 3 implémentations :
     - `CsvDataProvider` (actuel, refondu).
     - `MemoryDataProvider` (tests).
     - `StubProvider` (placeholder pour Polygon/Databento Sprint 1.4).
   - API : `provider.get_bars(symbol, tf, start, end) -> OHLCVFrame`.

4. **Migration call-sites prod** (6 h)
   - `sentinel_scanner.py`, `data_providers.py` (legacy), `signal_replay`, `audit_backtest.py`.
   - Conserver compat ascendante 1 sprint via wrapper `to_legacy_dataframe()`.

5. **Tests contractuels** (4 h)
   - `tests/test_data_layer_contracts.py` :
     - Reject tz-naive timestamps.
     - Reject low > high.
     - Reject duplicates / non-monotone.
     - Reject NaN volume.
     - Accept frame XAU M15 complet.
   - Cible : 25-30 tests, coverage 95 % du module.

6. **Réconciliation MTF intelligence/ vs environment/** (P1-13) (2 h)
   - Identifier les 2 impls (`multi_timeframe_features.py` vs autres).
   - Désigner `intelligence/data_layer/` comme canonique, marquer l'autre legacy.

7. **Documentation** (2 h)
   - `docs/algo/data_layer.md` : contrat, exemples, migration guide.

### Critères d'acceptation
- ✅ `OHLCVFrame` accepté par 100 % du chemin prod scanner.
- ✅ 25+ tests contractuels verts.
- ✅ Coverage `intelligence/data_layer/` ≥ 90 %.
- ✅ Aucune régression sur suite existante.

### Findings audit adressés
- **P0-8** (Pas de contrat Pydantic v2) — ✅ closed.
- **P1-13** (Réconciliation MTF intelligence/ vs environment/) — ✅ closed.

### Dépendances
- Sprint 0 batch 0.4 (CSV XAU propre).

### Risques
- Refactor large → migration cassante. Mitigation : wrapper compat 1 sprint.
- Pydantic v2 strict peut rejeter données réelles avec edge cases (1 bar volume=0). Mitigation : config `strict=False` par défaut, opt-in `strict=True` pour prod.

---

## Batch 1.2 — Resampler MTF property-based no-look-ahead (14 h)

### Objectif
Prouver formellement que le resampler M1→M15→H1→H4→D1→W1 ne fuit aucune info future. Le fix `<=`→`<` à `multi_timeframe_features.py:269` est déjà appliqué (P0-7 closed) — il faut prouver l'absence de régression.

### Steps
1. **Audit code resampler** (2 h)
   - `Read multi_timeframe_features.py` complet + `resample_ohlcv()`.
   - Documenter règles d'agrégation (open=first, close=last, high=max, low=min, volume=sum).
   - Sortie : `docs/algo/mtf_resampling.md`.

2. **Property-based tests avec Hypothesis** (5 h)
   - `tests/test_mtf_no_look_ahead.py` :
     - Property : `resample(df[:t], 'H1').iloc[-1].close == resample(df[:t+1], 'H1').iloc[-1].close` SI bar H1 fermée à t.
     - Property : aucune bar agrégée ne contient de timestamp > son timestamp d'ouverture + freq.
     - Property : monotonie préservée.
     - Strategy Hypothesis : génère OHLCV synthétiques contraints (low ≤ close ≤ high).
   - 5-8 properties, 200 cas chacune.

3. **Test régression `<=`→`<`** (1 h)
   - Test explicite : bar H1 fermée à 14:00 ne contient PAS la M15 14:00-14:15.
   - Si on revert le fix, le test tombe rouge.

4. **Détection look-ahead par fuzzing** (3 h)
   - Script `scripts/audit_mtf_lookahead.py` :
     - Charge XAU M15 réel.
     - Resample en H1 progressivement (rolling window).
     - Pour chaque bar H1, compare valeurs à un resample "oracle" (full dataset).
     - Différences → look-ahead détecté.

5. **Tests dans CI** (1 h)
   - Marquer property-based comme `@pytest.mark.slow` (≥ 10 s).
   - CI lance subset rapide + nightly full.

6. **Documentation** (2 h)
   - `docs/algo/mtf_resampling.md` enrichi avec proofs.

### Critères d'acceptation
- ✅ 5+ property-based tests, 200+ cas chacun, 0 échec.
- ✅ Fuzzing 100 000 bars XAU + EURUSD → 0 look-ahead détecté.
- ✅ Test régression `<=`→`<` rouge sur revert volontaire.

### Findings audit adressés
- **P0-7** (Look-ahead MTF `<=`) — déjà ✅ closed, ce batch ajoute le garde-fou.

### Dépendances
- Batch 1.1 (`OHLCVFrame` pour fixtures).

### Risques
- Hypothesis génère des cas pathologiques (volume=0, OHLC=identiques). Si le resampler crashe sur ces cas → bug à fixer. Mitigation : OK c'est ce qu'on cherche.

---

## Batch 1.3 — Pipeline calendrier économique 2026 + blackouts (22 h)

### Objectif
Le calendrier économique actuel est figé à 2025-12-31 (P1-14). Pipeline end-to-end de fetch + parse + blackout pour 2026, avec fallback si source indisponible.

### Steps
1. **Audit source actuelle** (2 h)
   - Lire `src/agents/news/economic_calendar.py`, `fetch_forexfactory_live.py`.
   - Documenter format CSV attendu, champs (date, time, currency, impact, event).
   - Sortie : `docs/algo/calendar_pipeline.md`.

2. **Fetcher 2026 ForexFactory** (4 h)
   - Script `scripts/fetch_ff_2026.py` (basé sur `fetch_forexfactory_live.py`).
   - Récupère 2026-01-01 → 2026-12-31.
   - Output : `data/news/ff_calendar_2026.csv`.
   - Validation : ≥ 1500 events Mid+High, dates monotones, currencies USD/EUR/GBP/JPY/CAD/AUD/NZD/CHF.

3. **Cross-check MT5** (2 h)
   - Adapter `crosscheck_mt5_calendar.py` pour 2026.
   - Diff FF vs MT5 sur les events High → idéalement ≤ 5 % de discrepancy.

4. **Contrat Pydantic `EconomicEvent`** (3 h)
   - `src/intelligence/data_layer/calendar.py` :
     - `EconomicEvent` (timestamp, currency, impact, event_name, actual, forecast, previous).
     - `EconomicCalendar` wrapper avec methods `get_blackouts(start, end, impact_filter)`.

5. **Intégration `ConfluenceDetector`** (3 h)
   - Vérifier que `NewsAnalysisAgent` consomme bien `EconomicCalendar` v2.
   - Ajouter blackout pre-event (T-15min) et post-event (T+15min) configurable par `InstrumentConfig`.

6. **Tests** (4 h)
   - `tests/test_calendar_pipeline.py` :
     - Fetch CSV → validate → query blackout.
     - Edge cases (event à minuit UTC, event sans actual).
     - Régression : NFP 2024-12-06 doit déclencher blackout USD ±15min.
   - 15-20 tests.

7. **Documentation + retraining script** (2 h)
   - `docs/algo/calendar_pipeline.md` enrichi.
   - `Makefile` target `make refresh-calendar` qui fetch + valide + replace.

### Critères d'acceptation
- ✅ `data/news/ff_calendar_2026.csv` chargé, ≥ 1500 events.
- ✅ Cross-check MT5 ≤ 5 % discrepancy events High.
- ✅ `ConfluenceDetector` blackout 2026 fonctionne sur replay XAU 2026-Q1.
- ✅ 15+ tests verts.

### Findings audit adressés
- **P1-14** (Calendrier 2025-12-31) — ✅ closed.

### Dépendances
- Batch 1.1 (`OHLCVFrame` + `data_layer/`).

### Risques
- FF peut bloquer scraping → fallback Investing.com / TradingEconomics scraper. Mitigation : abstraire fetcher derrière interface.
- 2026 events forecasts non finalisés en mai → re-fetch trimestriel automatique.

---

## Batch 1.4 — Décision sources licenciées (6 h, différée go/no-go)

### Objectif
Acter formellement le statut zone grise Dukascopy et préparer la migration commerciale Sprint 7+.

### Steps
1. **Lecture licences** (2 h)
   - Dukascopy ToS section "commercial use".
   - Polygon.io tiers ($29/mo Starter, $79/mo Developer).
   - Databento tiers (~$200/mo pour FX intraday).
   - Sortie : `audits/2026-Q2/data_licensing.md`.

2. **Coûts vs MVP** (1 h)
   - Polygon : XAU n'existe pas, FX OK.
   - Databento : XAU OUI (CME futures GC), FX OK.
   - Estimation $200-500/mo pour MVP 6 actifs commercial.

3. **Décision documentaire** (1 h)
   - **Tranche** : Dukascopy → R&D / personal testing only. **Pas commercial sans license**.
   - Migration commercial = Sprint 7+ (post-PF gate Sprint 3).
   - Si Sprint 3 gate échoue → décision reportée indéfiniment.

4. **Mockup wiring** (2 h)
   - `src/intelligence/data_layer/provider.py` : `PolygonProvider` (stub) + `DatabentoProvider` (stub).
   - Pas d'implémentation réelle Sprint 1, juste interface.

### Critères d'acceptation
- ✅ `data_licensing.md` signé.
- ✅ Stubs Polygon/Databento présents.
- ✅ Aucun fetch live commercial Sprint 1-6.

### Findings audit adressés
- **P0-14 partiel** (sources licenciées différées).

### Dépendances
- Décision A Sprint 0 (Dukascopy R&D-only acté).

### Risques
- Auteur des stubs peut être tenté de wire trop tôt. Mitigation : commentaire `# NOT FOR PROD pre-Sprint-7`.

---

## Batch 1.5 — CSV propres BTCUSD / US500 / GBPUSD / USDJPY (36 h)

### Objectif
Faire passer les 4 actifs MVP restants à coverage ≥ 90 % sur 2019-2025 M15. Dukascopy reste la source R&D.

### Steps
1. **Audit existant** (2 h)
   - `Grep` sur les 4 symboles dans `data/`.
   - Lister fichiers présents, coverage estimée.

2. **Fetcher Dukascopy par actif** (12 h, 3h/actif)
   - Adapter `scripts/download_dukascopy_xau.py` pour BTCUSD, US500, GBPUSD, USDJPY.
   - **Attention BTCUSD** : Dukascopy n'a que BTC/USD spot, OK pour R&D.
   - **Attention US500** : symbole `USA500.IDXUSD` chez Dukascopy.
   - Output : `data/BTCUSD_15MIN_2019_2025.csv` etc.

3. **Audit coverage par actif** (4 h)
   - Script `scripts/audit_csv_coverage.py` (générique, basé sur `audit_xau_coverage.py`).
   - Pour chaque symbole : bars expected (sessions weekday), bars actual, gaps > 30min, coverage %.
   - Sortie : `audits/2026-Q2/csv_coverage_mvp_4assets.md`.

4. **Tests régression data quality** (4 h)
   - `tests/test_data_quality_multi_asset.py` :
     - Pour chaque symbole : BOS firing rate < 5 %.
     - OHLC consistency (low ≤ close ≤ high).
     - Volume > 0 sur ≥ 95 % bars.
     - Coverage ≥ 90 %.

5. **Intégration `config.py` / `instruments.py`** (4 h)
   - Mettre à jour les 4 `InstrumentConfig` avec paths CSV.
   - Vérifier `price_decimals` (BTC=2, US500=1, GBP=5, JPY=3).

6. **Resampling H1/H4/D1** (4 h)
   - Via `multi_timeframe_features.resample_ohlcv()`.
   - Coverage H1 ≥ 95 % (moins de bars donc plus tolérant).

7. **Quick backtest smoke** (4 h)
   - Sur chaque actif, lancer `audit_backtest.py` rapide (subset 2024).
   - Pas de gate, juste smoke test : engine ne crashe pas, trades > 0.
   - Sortie : `reports/baseline/smoke_4assets.md`.

8. **Documentation** (2 h)
   - `docs/algo/mvp_assets.md` : 6 actifs status (XAU ✅, EUR ✅, BTC/US500/GBP/JPY new).

### Critères d'acceptation
- ✅ 4 CSV avec coverage ≥ 90 % sur 2019-2025.
- ✅ Tests régression data quality verts pour 4 actifs.
- ✅ Smoke backtest non-crash sur chaque actif.

### Findings audit adressés
- **P0-14** (5/6 presets sans CSV) — ✅ closed pour 4/5 (USOIL toujours absent — P2).

### Dépendances
- Batch 1.1 (DataProvider contractuel pour ingestion).

### Risques
- Dukascopy BTC pre-2019 inexistant → coverage 2019+ seulement (OK pour MVP).
- US500 sessions différentes (NYSE 9:30-16:00 ET) → audit doit ajuster bars expected.

---

## Gate de sortie du Sprint 1 (checklist 12 items)

1. ✅ Batch 1.0 façade smart_money livrée (RSI div + magic retest fixed) — déjà fait.
2. ✅ `OHLCVFrame` Pydantic v2 actif sur chemin prod scanner.
3. ✅ 25+ tests contractuels data layer verts.
4. ✅ Property-based tests MTF resampling (5+ properties, 0 fail).
5. ✅ Calendrier économique 2026 chargé, ≥ 1500 events Mid+High.
6. ✅ Cross-check FF/MT5 ≤ 5 % discrepancy.
7. ✅ 4 CSV MVP (BTC/US500/GBP/JPY) avec coverage ≥ 90 %.
8. ✅ Tests régression data quality 4 actifs verts.
9. ✅ Smoke backtest 4 actifs non-crash.
10. ✅ Doc licences sources (`data_licensing.md`) signée.
11. ✅ Suite tests complète verte (1 366 + ~80 nouveaux = ~1 446).
12. ✅ `sprint_1_retrospective.md` rédigé + diff stat vs `v0.9.0-pre-institutional`.

---

## Livrables Sprint 1 (arborescence)

```
src/intelligence/data_layer/         # nouveau package
  ├── __init__.py
  ├── contracts.py                   # OHLCVBar, OHLCVFrame, InstrumentMetadata
  ├── provider.py                    # DataProvider Protocol + impls
  └── calendar.py                    # EconomicEvent, EconomicCalendar

src/intelligence/smart_money/        # façade (déjà créée batch 1.0)
  └── __init__.py                    # ré-exports

data/
  ├── BTCUSD_15MIN_2019_2025.csv
  ├── US500_15MIN_2019_2025.csv
  ├── GBPUSD_15MIN_2019_2025.csv
  ├── USDJPY_15MIN_2019_2025.csv
  └── news/ff_calendar_2026.csv

scripts/
  ├── fetch_ff_2026.py
  ├── audit_csv_coverage.py
  └── audit_mtf_lookahead.py

tests/
  ├── test_data_layer_contracts.py        (25-30 tests)
  ├── test_mtf_no_look_ahead.py           (5-8 properties)
  ├── test_calendar_pipeline.py           (15-20 tests)
  ├── test_data_quality_multi_asset.py    (20+ tests)
  └── test_smart_money_facade.py          (déjà créé batch 1.0)

audits/2026-Q2/
  ├── data_provider_usages.md
  ├── csv_coverage_mvp_4assets.md
  └── data_licensing.md

docs/algo/
  ├── data_layer.md
  ├── mtf_resampling.md
  ├── calendar_pipeline.md
  └── mvp_assets.md

reports/baseline/
  └── smoke_4assets.md

roadmap/sprints/
  ├── sprint_1.md                    # ce fichier
  ├── sprint_1_progress.md
  └── sprint_1_retrospective.md
```

---

## Décisions ouvertes pour user

1. **Validation Dukascopy R&D-only** : confirmer que l'usage personal testing reste OK jusqu'à Sprint 7 gate commercial.
2. **Budget Polygon/Databento Sprint 7+** : ordre de grandeur $200-500/mo acceptable ? Si non → pivot data strategy.
3. **USOIL inclusion MVP** : si reportée P2 → confirmé. Sinon → ajouter +8h au batch 1.5.

---

**Signé** : Claude, 2026-05-15
