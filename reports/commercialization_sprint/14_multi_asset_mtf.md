# Plan de Commercialisation — Catégorie 14 : Multi-Asset & Multi-Timeframe

**Date** : 2026-05-21
**Périmètre** : 6 presets (XAUUSD, EURUSD, BTCUSD, US500, GBPUSD, USDJPY), `InstrumentConfig`, `resample_ohlcv`, `MultiTimeframeFeatures`, `MultiSymbolScanner`, alignement HTF/LTF, sessions, calendrier events.
**Objectif** : passer de 1 instrument ground-truth (XAU M15) à **3 instruments commercialisables** (XAU + EUR + USOIL) sur **3 timeframes utilisables** (M15 primaire + H1/H4 secondaires) en GA.
**Auteur** : Catégorie 14 — auto-mode.
**Conv. parallèle (MTF lead)** : Phase 1.1 → 1.7 (lookback bump 200→800, integration tests), Phase 2 (empirical 6-yr XAU replay), Phase 3 (activation poids `htf_alignment`). Tout ce plan **complète** sans dupliquer.

---

## 1. État actuel (Audit)

### 1.1 Multi-asset : 1/6 vraiment shippable
- **Registry propre** : `src/intelligence/volatility_forecaster.py:38-153` — 6 presets typés avec `session_hours`, `calendar_events`, `sl_atr_mult`, `tp_atr_mult`, `price_decimals`.
- **Data réellement disponible** (`ls data/`) :
  - `XAU_15MIN_2019_2024.csv` (97.6 % coverage) — backup figé.
  - `XAU_15MIN_2019_2026.csv` (98.72 % coverage 2019-2025) — source primaire (`config.py:43`).
  - `EURUSD_15MIN_2019_2025.csv` (~99.6 %) — onboardé entre eval 20 et aujourd'hui (cf. memory eval_05_09_refresh).
  - `XAU_15MIN_2019_2025.csv` (63 %) — DEPRECATED, garde-fou actif (`tests/test_data_quality_bos_regression.py`).
  - Zéro fichier OHLCV pour BTCUSD / US500 / GBPUSD / USDJPY.
- **MultiSymbolScanner** : implémenté (`src/intelligence/sentinel_scanner.py:990-1100+`), tests verts (`tests/test_multi_instrument.py:216-401`), wired dans `src/intelligence/main.py:353-366` derrière `SYMBOLS` env var.
- **`InstrumentConfig` manquants critiques** (cf. eval 20 §M3) : `pip_value`, `weekend_behavior`, `news_relevance_weight`, `atr_baseline_usd` — non posés sur aucun preset → impossible de calculer le notional, de gérer le gap weekend FX vs continuous crypto, ou de pondérer la blackout news par instrument.

### 1.2 Multi-timeframe : wiring en cours (autre conv.)
- **`resample_ohlcv`** : OK (`src/intelligence/volatility_forecaster.py:179-231`), couvre M1→W1.
- **`MultiTimeframeFeatures`** : `src/environment/multi_timeframe_features.py:41-100+` — produit `HTF_TREND_1H`, `HTF_TREND_4H`, `HTF_STRENGTH_1H/4H`, `HTF_RSI_4H`, `PRICE_VS_SMA20_1H/4H`, `PRICE_VS_SMA50_1H/4H`. Possède un *look-ahead-safe causal mask* (l.272 d'après le commentaire du scanner).
- **Phase 1.1 done par conv. parallèle** : `lookback_bars` bumpé 200→800 (`src/intelligence/sentinel_scanner.py:90`) avec warm-up guard (`_mtf_warmup_bars = 200`, ligne 182).
- **Scoring htf_alignment câblé à poids 0** : `src/intelligence/confluence_detector.py:131` (`"htf_alignment": 0.0`) et `:626-689` (`_score_htf_alignment`). Composant calculé et exposé (readout descriptif) mais ne touche pas le score tant que Phase 2 n'a pas validé.
- **`_compute_htf_features_safe`** : `src/intelligence/sentinel_scanner.py:785-` — résolve les HTF features par scan, fallback `None` propre.

### 1.3 Sessions, calendrier, corrélations
- **Sessions UTC** : présentes par preset mais conjecturales pour BTC/US500 (jamais testées sur data réelle). XAU/EUR validées par 7 ans de replay.
- **Calendrier events** : `EconomicCalendarFetcher` (`src/agents/news/economic_calendar.py:130+`) honore un `CURRENCY_MAP`. BTC = **3 events seulement** (FOMC, FFR, CPI), pas de BOJ pour USDJPY dans le fixture de test.
- **Cross-asset correlation** : `src/intelligence/cross_asset_correlation.py` existe — rolling Pearson XAU vs (DXY, SPX, US10Y, BTC) pour narratives, **jamais** consommé par `ConfluenceDetector` aujourd'hui. Aucun bucket de cap de portefeuille (eval 20 §M5).

### 1.4 Conclusion audit
- **Ready GA** : XAU M15 (PF 1.572 → 1.598 avec RegimeFilter, cf. commentaire `volatility_forecaster.py:58`).
- **Ready post-validation** : EURUSD M15 (data ok, sweep non publié, asymétrie long/short non testée).
- **Pas ready** : BTC / US500 / GBP / JPY — pas de CSV, pas de baseline, sessions/news conjecturales.
- **MTF** : pipeline en place, **observability only** (poids 0). Décision empirique attendue en Phase 2/3 de l'autre conv.

---

## 2. Vision cible (GA)

### 2.1 Coverage commercialisable
- **3 instruments en GA** : XAUUSD, EURUSD, **USOIL** (recommandation eval 20 §M7). GBP/JPY/BTC/US500 → backlog post-GA.
- **3 timeframes** par instrument :
  - **M15** : timeframe principal (≥1 signal/jour, sweet spot retail).
  - **H1** : signal swing intraday (~3/jour XAU, ANALYST+).
  - **H4** : signal swing low-volume (FREE teaser) + **utilisé comme HTF context** pour M15.
- **M5** = optionnel STRATEGIST+ (à shipper si compute le permet, dépend du sweep tier).
- **D1/W1** = nice-to-have, pas bloquant GA.

### 2.2 Architecture cible
1. Chaque scanner consomme un `InstrumentConfig` **enrichi** (pip_value, weekend_behavior, news_relevance_weight, atr_baseline_usd, htf_lookback_bars).
2. `MultiSymbolScanner` instancie un scanner par (symbol × timeframe primaire) avec partage du `LLMNarrativeEngine`, `SemanticCache`, `SignalStore`.
3. `_compute_htf_features_safe` renvoie un dict consommé par `ConfluenceDetector._score_htf_alignment` avec poids ≠ 0 (cible **13** après Phase 3 — cf. commentaire `confluence_detector.py:126-131`).
4. **Cross-asset correlation gate** : avant publication, si ρ_30j(symbol_a, symbol_b) > 0.7 et qu'un signal actif existe sur l'autre, on retient le moins scoré (cap 1 signal par bucket corrélé).
5. **Session-aware scoring** : multiplicateur de score (ou de position) dépendant de la session active vs `session_hours` du preset. Hors session "regular" pour US500 → blackout.

### 2.3 KPIs go/no-go GA
- ≥ **2 symbols** PF OOS > 1.20 (XAU + EUR), 1 symbol PF OOS > 1.0 (USOIL).
- 0 doublon ρ > 0.7 actif simultanément (cap portfolio).
- `htf_alignment` poids ≥ 10 dans `DEFAULT_WEIGHTS` (validé Phase 3 autre conv.).
- 100 % presets ont `pip_value`, `weekend_behavior`, `news_relevance_weight`, `atr_baseline_usd`.
- Suite tests verte sur **les 3 instruments shippés** (golden replays).

---

## 3. Gap analysis

| # | Domaine | Gap | Sévérité | Effort | Dépend MTF Phase ? |
|---|---------|-----|----------|--------|--------------------|
| G1 | InstrumentConfig | Manque `pip_value`, `weekend_behavior`, `news_relevance_weight`, `atr_baseline_usd` | **P0** | 4 h | Non |
| G2 | Décision GA | Quels 3-4 instruments shipper en GA ? Décision non actée | **P0** | 3 h | Non |
| G3 | Sweep tiers | Tiers (`PREMIUM`/`STANDARD`/`WEAK` à 55/40/25) basés sur replay **AVANT** `htf_alignment` activé — à re-sweep | **P0** | 6 h | **OUI Phase 3** |
| G4 | USOIL onboarding | Pas de preset USOIL, pas de CSV Dukascopy, pas de calendar events spécifiques (OPEC, EIA) | P1 | 12 h | Non |
| G5 | Tests per-instrument | `tests/test_multi_instrument.py` couvre uniquement le scaffolding (mock). Pas de golden replay par instrument | **P0** | 8 h | Partiel |
| G6 | Cross-asset gate | `cross_asset_correlation.py` non branché sur `MultiSymbolScanner` pour cap portefeuille | P1 | 6 h | Non |
| G7 | Session-aware scoring | `session_hours` présent mais non consommé pour gating (XAU asian session = bruit) | P1 | 5 h | Non |
| G8 | Calendar coverage | BTC = 3 events (insuffisant), USDJPY sans BOJ fixture, USOIL absent | P1 | 4 h | Non |
| G9 | Decimal precision | `price_decimals` honoré dans `ConfluenceDetector` (`:180`) mais pas systématique côté `_publish_signal` / `Telegram` / `Insight V2` | P1 | 3 h | Non |
| G10 | Symbol whitelist | `SYMBOLS` env var parsé brut (`main.py:534`) — pas de whitelist anti-injection | P1 | 1 h | Non |
| G11 | Per-instrument metrics | Pas de PF/signal_count/narrative_quality breakdown par symbole dans `/health` ou `/metrics` | P1 | 4 h | Non |
| G12 | Multi-TF entry | Confluence M15 ne sait pas qu'un signal H1 vient d'être publié sur le même symbole (risque duplicates inter-TF) | P2 | 6 h | OUI Phase 3 |
| G13 | Regime divergence | RegimeFilter calibré sur XAU 7-yr — non re-fit pour EUR/USOIL | P1 | 6 h | Non |
| G14 | Data gap handling | `data_quality.py` strict refuse les feeds mais aucun fallback graceful pour weekend FX (jeudi 22h ASCII) | P2 | 3 h | Non |

---

## 4. Plan d'exécution (COMPLÉMENTAIRE à Phase 1-3 MTF de l'autre conv.)

> **Règle d'or** : tant que la conv. MTF n'a pas livré Phase 3 (activation `htf_alignment` ≠ 0 + re-sweep), **NE PAS** re-publier les seuils de tier (`min_score`, `enter_threshold`). Tout travail tier-related doit attendre `weights["htf_alignment"] > 0`.

### P0 — Décision : quels 3-4 instruments shipper en GA ?
- **Fichiers** : `docs/decisions/ga_instruments_2026_q2.md` (à créer), `config.py` (`SYMBOLS` default), `infrastructure/docker-compose.yml` (env var).
- **Action** :
  1. Recapituler PF/Sharpe XAU sur dernière baseline (réf. `reports/sweep_sl_tp.md`).
  2. Lancer sweep EURUSD M15 réel via `reports/eval_20_sweep_runner.py --symbols EURUSD --walk-forward` sur `data/EURUSD_15MIN_2019_2025.csv`.
  3. Décider : **GA = XAU + EUR**, **USOIL P1 post-onboarding**, **drop temporaire BTC/US500/GBP/JPY** (cohérent eval 20 §M12).
  4. Documenter rationale (ρ EUR-GBP 0.75 = cannibalisation, BTC 24/7 = coût LLM 2x, US500 RTH = bars_per_day=28 jamais testé).
- **Heures** : 3 h (incl. exécution sweep EUR).
- **Acceptance** :
  - Doc `ga_instruments_2026_q2.md` posté avec PF OOS chiffré XAU + EUR.
  - `config.py` ou env `SYMBOLS` par défaut limité aux instruments choisis.
  - Note `MEMORY.md` actée.
- **Dépendances** : aucune (peut tourner en parallèle de Phase 1 MTF).

### P0 — Enrichir InstrumentConfig (champs commerciaux manquants)
- **Fichiers** : `src/intelligence/volatility_forecaster.py:238-300` (dataclass + 6 presets), `tests/test_multi_instrument.py` (ajouter assertions).
- **Action** :
  1. Ajouter à `InstrumentConfig` : `pip_value: float = 1.0`, `weekend_behavior: str = "gap"`, `news_relevance_weight: float = 1.0`, `atr_baseline_usd: float = 0.0`, `htf_lookback_bars: int = 800`.
  2. Patcher les 6 presets avec les valeurs eval 20 §M3 (XAU pip_value=10, BTC weekend_behavior="continuous", etc.).
  3. Adapter `ConfluenceDetector.__init__` (`confluence_detector.py:155-186`) pour propager les nouveaux champs si pertinent (news_relevance_weight pondère la blackout NewsAnalysisAgent).
  4. Mettre à jour les tests existants `test_multi_instrument.py::TestInstrumentRegistry` avec assertions sur les nouveaux champs.
- **Heures** : 4 h.
- **Acceptance** :
  - `assert reg["BTCUSD"].weekend_behavior == "continuous"` passe.
  - `assert reg["XAUUSD"].atr_baseline_usd == 4.5` passe.
  - Aucune régression dans `tests/test_multi_instrument.py` (33 tests existants verts).
- **Dépendances** : aucune.

### P0 — Tier re-sweep POST `htf_alignment` activation
- **⚠️ ATTENDRE Phase 3 autre conv.** (poids `htf_alignment` lifté ≠ 0).
- **Fichiers** : `reports/eval_20_sweep_runner.py` (étendre), `scripts/sweep_state_machine.py`, `src/intelligence/confluence_detector.py:35-44` (SignalTier cutpoints), `src/intelligence/main.py:307-308` (STATE_MACHINE_ENTER/EXIT_THRESHOLD).
- **Action** :
  1. Une fois Phase 3 livrée, re-runner `reports/eval_20_sweep_runner.py --walk-forward --instruments XAU,EUR` (et USOIL si onboardé) pour mesurer la nouvelle distribution de scores avec `htf_alignment` activé.
  2. Recalibrer `SignalTier` (`confluence_detector.py:41-44`) sur p90/p50/p25 de la nouvelle distribution OOS.
  3. Recalibrer `state_machine_enter/exit` (`main.py:307-308`) sur p75/p25 de la post-filter distribution.
  4. Mettre à jour le commentaire historique l.36-40 avec la date et le n des trades.
- **Heures** : 6 h (sweep + analyse + commit).
- **Acceptance** :
  - Distribution `confluence_score` documentée dans `reports/sweep/tier_recalibration_post_mtf.md`.
  - Tiers et thresholds défendables empiriquement.
  - Suite de tests verte (notamment `tests/test_signal_state_machine.py`, `test_sprint4_graduated_scoring.py`).
- **Dépendances** : **Phase 3 MTF (autre conv.)**.

### P0 — Instrument config validation tests
- **Fichiers** : nouveau `tests/test_instrument_config_validation.py`, fixtures dans `tests/fixtures/golden_replays/`.
- **Action** :
  1. Test par preset : decimal precision round-trip (XAU price 2050.567 → arrondi à 2050.57 ; EUR 1.10532 reste à 5 décimales ; JPY 152.345 → 152.345).
  2. Test sessions : générer une bar par session pour chaque preset, vérifier `_session_of(bar)` cohérent.
  3. Test HAR windows auto-computed (déjà partiel dans `test_multi_instrument.py:84-95`) étendu aux 6 presets.
  4. Test calendar_events non vide et au moins 1 event par juridiction (US/EU/UK/JP/CH/OPEC selon symbol).
  5. **Golden replay** : pour XAU et EUR, charger 100 bars de référence depuis fixtures, faire tourner `_scan_once()` mock-isolé, vérifier le `signal_id`, `tier`, `score` produits sont stables (snapshot test).
- **Heures** : 8 h (le golden replay est le gros morceau).
- **Acceptance** :
  - 12+ nouveaux tests verts.
  - 0 régression sur les 33 tests `test_multi_instrument.py`.
  - Golden replay XAU et EUR commit dans `tests/fixtures/`.
- **Dépendances** : G1 (InstrumentConfig enrichi).

### P0 — Cross-asset correlation handling (XAU-DXY, XAU-VIX, EUR-GBP) pour confluence
- **Fichiers** : `src/intelligence/cross_asset_correlation.py:79+` (corr_table), `src/intelligence/confluence_detector.py:117-132` (ajouter composant `cross_asset` à `DEFAULT_WEIGHTS` à poids 0, observability-only Phase 1), `src/intelligence/sentinel_scanner.py` (`_compute_cross_asset_safe` à l'image de `_compute_htf_features_safe`), nouveau `data/macro/dxy_15min.csv` (Dukascopy), `data/macro/spx_15min.csv`.
- **Action** :
  1. Onboarder DXY + SPX via `scripts/download_dukascopy_xau.py --symbol DXYINDEX` (et SP500 si dispo Dukascopy ; sinon fallback Polygon).
  2. Pré-calculer la matrice ρ rolling 30j (XAU vs DXY, XAU vs SPX, EUR vs GBP, EUR vs DXY) au démarrage du scanner ; refit toutes les 96 bars (1 jour M15).
  3. Brancher `_compute_cross_asset_safe` dans `_scan_once` (juste après `_compute_htf_features_safe`) → renseigne `signal.cross_asset_context` (nouveau champ `ConfluenceSignal`).
  4. Ajouter composant `cross_asset_confirm` à `DEFAULT_WEIGHTS` à poids **0** pour Phase 1 (observability), avec scoring logique : XAU LONG + DXY < 30j SMA → +quality, XAU LONG + ρ XAU-DXY < -0.6 → +quality.
  5. À Phase 2, mesurer si ce composant améliore le PF (sweep `--with-cross-asset`). Si oui, activer poids ≥5 en Phase 3 (re-sweep tiers).
- **Heures** : 10 h (data + module + brachange + Phase 1 wiring).
- **Acceptance** :
  - `data/macro/dxy_15min.csv` ≥ 95 % coverage 2019-2025.
  - `tests/test_cross_asset_correlation.py` (existe ou créer) avec 8+ tests verts.
  - `signal.cross_asset_context` peuplé dans 100 % des scans avec data dispo, fallback `None` quand absent.
  - Poids 0 garanti = aucune régression du score actuel.
- **Dépendances** : G1, et **synchronisation soft avec Phase 1 MTF** (même architecture safe-compute, ne pas dupliquer `lookback_bars`).

### P1 — Onboarding USOIL (si décision G2 supporte)
- **Fichiers** : `src/intelligence/volatility_forecaster.py:38+` (ajouter preset USOIL), `scripts/download_dukascopy_xau.py` (paramétrer pour USOIL), `data/USOIL_15MIN_2019_2025.csv` (à générer), `tests/test_multi_instrument.py` (ajouter cas USOIL).
- **Action** :
  1. Préset USOIL : `symbol="USOIL"`, `timeframe="M15"`, `bars_per_day=96` (extended hours), `price_decimals=2`, `sl_atr_mult=2.5`, `tp_atr_mult=5.0`, `pip_value=10.0`, `weekend_behavior="gap"`, `news_relevance_weight=1.0` (OPEC = haut impact), `calendar_events=["FOMC", "CPI", "EIA Crude Oil Inventories", "OPEC Meeting", "API Weekly Crude Oil Stock"]`.
  2. Sessions UTC : `{"asian":(0,8), "london":(8,13), "ny_overlap":(13,17), "ny_afternoon":(17,21), "after_hours":(21,24)}`.
  3. Onboarder data via Dukascopy (script paramétré `--symbol USOIL`), vérifier coverage > 95 %.
  4. Calibrer RegimeFilter pour USOIL (n'utilise pas XAU-fit) — cf. G13.
  5. Sweep walk-forward sur 6 ans USOIL via `reports/eval_20_sweep_runner.py` → décision keep/drop sur PF OOS > 1.0.
  6. Ajouter dans `mockups/pricing_bundles.md` le Metal Pack incluant USOIL.
- **Heures** : 12 h (data download = 4 h, preset + tests = 4 h, sweep + analyse = 4 h).
- **Acceptance** :
  - Preset committed et listé dans `get_instrument_registry()` (7 presets).
  - Data file `data/USOIL_15MIN_2019_2025.csv` ≥ 95 % coverage.
  - Sweep report `reports/usoil_baseline_2026_q2.md` avec PF OOS chiffré.
  - Verdict keep / drop documenté.
- **Dépendances** : G2, G1.

### P1 — Session-aware vol/scoring (London / NY / Asia)
- **Fichiers** : `src/intelligence/confluence_detector.py` (ajouter méthode `_session_modifier(bar_ts, instrument_config)`), `src/intelligence/regime_filter.py` (existe — étendre pour blacklister sessions hors marché).
- **Action** :
  1. Implémenter `_session_modifier` qui retourne un multiplicateur `[0.0, 1.0]` selon `session_hours` du preset (XAU asian = 0.7 — bruit, XAU london+ny = 1.0).
  2. Pour US500 : multiplicateur 0.0 (= drop) hors `(regular, after_hours)` — la conjecture eval 20 sur `bars_per_day=28` est non testée, mais ça active le drop au minimum.
  3. Brancher dans `_scan_once` post-scoring : `signal.confluence_score *= session_modifier`.
  4. Mesurer impact via `reports/eval_20_sweep_runner.py --with-session-aware` sur XAU + EUR.
  5. Si lift > +5 % PF OOS → activer en GA ; sinon parquer en backlog.
- **Heures** : 5 h.
- **Acceptance** :
  - Test unitaire `test_session_modifier_xau_asian == 0.7`.
  - Rapport empirique `reports/session_aware_lift_2026_q2.md` avec décision activer/parquer.
- **Dépendances** : G1 (sessions présent), G13 (RegimeFilter per-instrument à terme).

### P1 — Per-instrument metrics dans /health et /metrics
- **Fichiers** : `src/api/routes/health.py`, `src/intelligence/sentinel_scanner.py:1100+` (`MultiSymbolScanner.get_stats`).
- **Action** :
  1. Étendre `MultiSymbolScanner.get_stats()` (déjà partiel cf. `test_multi_instrument.py:332-351`) pour exposer `pf_30d`, `signal_count_30d`, `narrative_failure_rate` **par symbole**.
  2. Brancher `/health` (`src/api/routes/health.py`) pour exposer ces stats par symbole.
  3. Ajouter compteurs Prometheus `signals_generated{symbol}`, `pf_30d{symbol}`, `narrative_fallback_uses_total{symbol}` (utilise registry déjà initialisée — cf. eval 16 sur `/metrics` payload vide).
- **Heures** : 4 h.
- **Acceptance** :
  - `GET /health` retourne `per_symbol: {XAUUSD: {pf_30d: 1.45, signals_30d: 124}, EURUSD: {…}}`.
  - `GET /metrics` retourne `signals_generated{symbol="XAUUSD"} 1234`.
  - Test `test_health_per_symbol` ajouté.
- **Dépendances** : aucune.

### P1 — Symbol whitelist (sécurité)
- **Fichiers** : `src/intelligence/main.py:533-534` (parse SYMBOLS).
- **Action** :
  1. Définir `ALLOWED_SYMBOLS = set(get_instrument_registry().keys())` au boot.
  2. Refuser tout symbol absent (`if any(s not in ALLOWED_SYMBOLS for s in symbols): raise ValueError`).
  3. Logger un warning explicite.
- **Heures** : 1 h.
- **Acceptance** :
  - `SYMBOLS=XAUUSD,FAKEPAIR python -m src.intelligence.main` → erreur claire au boot, pas de crash plus tard.
- **Dépendances** : aucune.

### P1 — Calendar coverage complet (BTC, JPY, USOIL)
- **Fichiers** : `src/intelligence/volatility_forecaster.py:91-93` (BTC), `:145-148` (USDJPY), preset USOIL.
- **Action** :
  1. BTC : ajouter `"Coinbase Quarterly Earnings", "SEC ETF Decision", "Halving Event"` — pas dans ForexFactory mais marqués via fixture custom.
  2. USDJPY : vérifier BOJ fixture présente (eval 20 §M4 a noté l'absence). Compléter `scripts/fetch_forexfactory_live.py` pour inclure JPY events.
  3. USOIL : ajouter EIA Crude Oil Inventories (hebdo mercredi 14:30 UTC), API Weekly (mardi 20:30 UTC), OPEC meetings (Q2/Q4).
  4. Cross-check via `scripts/crosscheck_mt5_calendar.py`.
- **Heures** : 4 h.
- **Acceptance** :
  - Chaque preset a ≥ 5 calendar_events.
  - Fixture `tests/fixtures/economic_calendar_sample.csv` couvre USD/EUR/GBP/JPY/CH/OPEC.
- **Dépendances** : G4 si USOIL en GA.

### P1 — RegimeFilter per-instrument (XAU vs EUR vs USOIL)
- **Fichiers** : `src/intelligence/regime_filter.py` (existe), `src/intelligence/main.py:291-300` (instantiation).
- **Action** :
  1. RegimeFilter actuel fitte sur le replay XAU 7-yr. Refit nécessaire sur EUR/USOIL avant activation par instrument.
  2. Soit (a) ranger un fit par instrument dans `data/regime_filter_{symbol}.pkl`, soit (b) reparameter `from_env()` pour accepter un override par symbole.
  3. Mesurer lift PF par instrument sans regime filter (baseline) puis avec regime filter fitté sur cet instrument.
  4. Garder le regime filter actif uniquement sur les instruments où PF lift ≥ +10 %.
- **Heures** : 6 h.
- **Acceptance** :
  - `RegimeFilter(symbol="EURUSD").evaluate(...)` retourne décisions distinctes de l'XAU-fit.
  - Lift PF documenté par instrument dans `reports/regime_filter_lift_per_instrument.md`.
- **Dépendances** : G2.

### P1 — Decimal precision end-to-end (publish + Telegram + Insight V2)
- **Fichiers** : `src/intelligence/sentinel_scanner.py:718-775` (`_publish_signal`), `src/delivery/telegram_notifier.py`, `src/api/insight_signal_v2.py`.
- **Action** :
  1. Auditer le formattage prix dans `Telegram` notifier : utilise-t-il bien `instrument_config.price_decimals` ?
  2. Idem dans `Insight V2` (`entry_price`, `stop_loss`, `take_profit` doivent être arrondis selon le preset).
  3. Test unitaire : envoyer un signal EUR avec entry=1.10532 → notification montre 1.10532, pas 1.11.
- **Heures** : 3 h.
- **Acceptance** :
  - Aucun signal n'affiche plus de décimales que `price_decimals`.
  - Test `test_decimal_precision_per_instrument` ajouté.
- **Dépendances** : G1.

### P2 — Cross-asset arb / multi-leg signals
- **Fichiers** : nouveau `src/intelligence/cross_asset_arb.py`.
- **Action** :
  1. Détecter des setups inter-asset (ex. XAU LONG + DXY SHORT confirmé sur même bar) — relevant pour Tier INSTITUTIONAL.
  2. Compute un `arb_score` distinct du `confluence_score`.
  3. Backtest sur 7 ans (XAU + EUR + DXY synthétique).
  4. Si PF > 1.2 et n_trades > 100 → activer pour INSTITUTIONAL only.
- **Heures** : 20 h (gros morceau, P2 → post-GA).
- **Acceptance** :
  - `arb_score` exposé dans `InsightSignalV2.cross_asset_arb`.
  - Backtest 7 ans avec walk-forward documenté.
- **Dépendances** : P0 cross-asset correlation (G6), data multi-asset complet.

### P2 — Multi-TF same-symbol coordination (anti-doublon M15 vs H1 vs H4)
- **Fichiers** : `src/api/signal_store.py`, `src/intelligence/sentinel_scanner.py`.
- **Action** :
  1. Si un signal XAU H1 LONG est actif (state machine = LONG) et qu'un scan M15 produit XAU M15 LONG → ne pas publier (sinon le client voit 2 signaux contradictoires-en-direction-mais-redondants).
  2. Si M15 SHORT pendant qu'H1 LONG → publier mais flagger `conflict_with_higher_tf=True` (le narrateur peut prévenir).
  3. Cap : max 1 entrée active par (symbol × direction) toutes timeframes confondues.
- **Heures** : 6 h.
- **Acceptance** :
  - Test `test_no_duplicate_m15_h1_same_direction` vert.
  - Test `test_conflict_flag_when_tf_disagree` vert.
- **Dépendances** : Phase 3 MTF + état multi-scanner partagé (extension `SignalStore`).

### P2 — Data gap graceful fallback (weekend FX)
- **Fichiers** : `src/intelligence/data_quality.py`, `src/intelligence/sentinel_scanner.py:334-340`.
- **Action** :
  1. Le `validate_ohlcv(strict=True)` actuel coupe le scan dès qu'il y a un gap. En weekend FX (vendredi 21h → dimanche 22h UTC), c'est normal — ne pas alerter.
  2. Étendre `data_quality.py` pour consommer `instrument_config.weekend_behavior` et accepter le gap weekend en FX/Metal, exiger continuous sur BTC.
- **Heures** : 3 h.
- **Acceptance** :
  - Test `test_weekend_gap_accepted_for_xau` vert, `test_weekend_gap_rejected_for_btc` vert.
- **Dépendances** : G1.

---

## 5. Tests & validation

### 5.1 Tests unitaires à ajouter
- `tests/test_instrument_config_validation.py` — 12+ tests (G5).
- `tests/test_cross_asset_correlation_wiring.py` — composant à poids 0 produit `cross_asset_context`.
- `tests/test_session_modifier.py` — 1 test par preset × session.
- `tests/test_decimal_precision_per_instrument.py` — round-trip prix.
- `tests/test_no_duplicate_m15_h1_same_direction.py` — P2.

### 5.2 Golden replays
- Snapshot 200 bars XAU + 200 bars EUR depuis fixtures réelles → enregistrer le `signal_id`, `tier`, `score`, `entry_price` produits par `SentinelScanner._scan_once()` en mode déterministe (mock LLM).
- Toute modif du scoring doit casser le snapshot → review obligatoire.
- Re-générer snapshots **uniquement** quand `htf_alignment` est activé (Phase 3 done) ou quand un patch volontaire change le scoring.

### 5.3 Tests d'intégration multi-asset
- `tests/test_multi_symbol_scanner_e2e.py` — XAU + EUR scanners coexistent, partagent `LLMNarrativeEngine` et `SemanticCache`, n'interfèrent pas (mock data feed).
- Vérifier `scan_all_once()` ne crash pas si un symbole a une data corrompue (isolation per-scanner).

### 5.4 Suite complète à faire passer avant GA
- 1366 + ~25 nouveaux tests verts (cf. memory section Test Suite).
- Coverage `src/intelligence/sentinel_scanner.py` ≥ 85 %.
- Coverage `src/intelligence/cross_asset_correlation.py` ≥ 90 %.

---

## 6. Sécurité

- **G10 Symbol whitelist** : voir P1. Anti-injection via env var.
- **Instrument config injection** : `InstrumentConfig` dataclass — déjà typé. Refuser tout `sl_atr_mult > 10`, `tp_atr_mult > 20`, `bars_per_day > 1440` au load via `__post_init__`.
- **Cross-asset data ingestion** (`data/macro/dxy_15min.csv`) : valider via `data_quality.py` au même titre que XAU/EUR. Refuser CSV avec < 95 % coverage.
- **Telegram price formatting** : si `price_decimals` mal configuré, on peut envoyer "Entry 1.10" pour EUR au lieu de "1.10532" → risque commercial (signal incompréhensible). Test bloquant.

---

## 7. Métriques

### 7.1 Par instrument (à exposer dans `/health` et `/metrics`)
- `signals_generated{symbol}`, `signals_dropped_by_regime_filter{symbol}`, `signals_blocked_by_kill_switch{symbol}`.
- `pf_30d{symbol}`, `sharpe_30d{symbol}` (calculé sur les signaux clos par state machine).
- `narrative_fallback_uses_total{symbol}`, `llm_failures_total{symbol}`.
- `htf_features_computed_total{symbol}` (cf. `sentinel_scanner.py:184`).

### 7.2 Per timeframe
- Une fois multi-TF entry coordination active (P2) : `signals_generated{symbol, tf}`, `conflict_inter_tf_total{symbol}`.

### 7.3 Cross-asset
- `cross_asset_corr{pair="XAU_vs_DXY"}` (gauge, refit 1 jour).
- `cross_asset_confirm_score_avg{symbol}`.

### 7.4 Per-bundle (commercial)
- `subscribers_per_bundle{bundle="metal_pack"}` (post-tier wiring, cf. eval 24).
- `mrr_per_bundle` — externe à ce périmètre.

---

## 8. Risques & mitigations

| # | Risque | Sévérité | Mitigation |
|---|--------|----------|------------|
| R1 | EURUSD se dégrade post-2024 (β-capture XAU, cf. memory decision_matrix) → PF EUR OOS < 1.0 | **HAUTE** | Sweep walk-forward avant ship. Si EUR fail → GA XAU only + repli BTC ou GBP ; ne PAS shipper EUR juste pour "multi-asset" claim. |
| R2 | Activation `htf_alignment` ≠ 0 (Phase 3 MTF) dégrade le PF XAU au lieu de l'améliorer (htf_features biaisées) | **MOY** | Phase 2 valide avant. Si dégradation → garder poids 0 (observability only). Mon plan ne s'engage à re-sweep tiers QUE si lift confirmé. |
| R3 | USOIL onboarding coûteux (12 h) et finit PF < 1.0 → dette technique sans valeur | MOY | Sweep OOS obligatoire avant pricing du Metal Pack. Decision-gate après G4 sweep. |
| R4 | Cross-asset data feeds (DXY/SPX) trop chers ou pas dispo Dukascopy → bloquant correlation handling | MOY | Fallback Polygon ($29/mo) ou Twelve Data ($79/mo). Acceptable budget pour 1 feed institutional. |
| R5 | Sessions session-aware multiplicateur fait baisser PF XAU asian (élimine setups valides) | FAIBLE | Sweep avec/sans, ne pas activer si lift < 5 %. |
| R6 | RegimeFilter XAU-fit appliqué à EUR/USOIL crée des faux positifs (regime fit non transférable) | HAUTE | G13 oblige fit per-instrument. Pas d'activation per-instrument tant que pas fitté. |
| R7 | Decimal precision bug → Telegram envoie "Entry 1.11" au lieu de "1.10532" → client se sent floué | HAUTE | Test bloquant G9. Snapshot fixtures. |
| R8 | MTF Phase 3 ne se concrétise pas → tier re-sweep bloqué indéfiniment | FAIBLE | Plan B : recalibrer tiers sans htf_alignment sur la dist actuelle (post-Phase 2 even). Documenter le "tier provisoire" en GA. |
| R9 | Conv. parallèle MTF modifie `DEFAULT_WEIGHTS` ou `_score_htf_alignment` pendant que je sweep → conflits merge | MOY | Coord sync : ne PAS toucher à `confluence_detector.py:117-132` et `:626-689` tant que Phase 3 pas merged. Mon plan touche `weights` ailleurs (ajout `cross_asset_confirm` à poids 0 — pas de collision tant qu'on garde la somme=100). |
| R10 | Data gap weekend FX bloque le scanner en strict mode samedi/dimanche | FAIBLE | P2/G14. Workaround court terme : scheduler ne lance pas le scan en weekend pour FX/Metal. |

---

## 9. Dépendances

### 9.1 Dépendances vers d'autres catégories
- **Catégorie 1 — Data infrastructure** : G4 (USOIL Dukascopy), G6 (DXY/SPX feeds) dépendent du pipeline Dukascopy stable et de la licence (eval 08 = zone grise commerciale).
- **Catégorie 2 — Algo / ConfluenceDetector** : G3 tier re-sweep dépend de la convergence ConfluenceDetector (memory `confluence_calibration.md` = Pearson −0.023). Si autre conv. remplace la scoring fn, tout mon plan tier doit re-tourner.
- **Catégorie 3 — Backtest engine** : G3, G4, G7 utilisent `reports/eval_20_sweep_runner.py` + walk-forward. Doit être maintenu.
- **Catégorie 4 — MTF wiring (conv. parallèle)** : G3, G12 dépendent strictement de Phase 3 done (htf_alignment ≠ 0).
- **Catégorie 5 — News / Calendar** : G8 dépend de `fetch_forexfactory_live.py` + JPY/OPEC events sourcing.

### 9.2 Ordre d'exécution recommandé (Gantt court)
```
S0 (J1-J3)   : G2 décision GA + G1 InstrumentConfig enrichi + G10 symbol whitelist
S1 (J4-J7)   : G5 instrument validation tests + golden replays XAU/EUR + G9 decimal
S1 (J4-J7)   : G6 cross-asset correlation (poids 0, observability-only) [parallèle]
S2 (J8-J14)  : G4 USOIL onboarding (data + preset + sweep) [si G2 supporte]
S2 (J8-J14)  : G7 session-aware + G13 RegimeFilter per-instrument
S2 (J8-J14)  : G11 per-instrument metrics + G8 calendar
S3 (J15-J18) : attendre Phase 3 MTF, puis G3 tier re-sweep
S4 (J19-J25) : tests d'intégration, golden replay refresh, doc commerciale
S4 (J19-J25) : P2 cross-asset arb, multi-TF coord (si temps)
```

---

## 10. Estimation totale & timeline

### 10.1 Heures par phase
| Phase | Tâches | Heures |
|-------|--------|--------|
| **P0 must-have GA** | G1+G2+G3+G5+G6 | 31 h |
| **P1 amélioration GA** | G4+G7+G8+G9+G10+G11+G13 | 36 h |
| **P2 post-GA** | Cross-asset arb + multi-TF coord + weekend gap | 29 h |
| **TOTAL** | | **96 h** |

### 10.2 Timeline (solo, 8 h/jour, blocages MTF compris)
- **Sprint A (3 jours)** : P0 hors G3 = 22 h → S0+S1 (G1+G2+G5+G6+G10 + tests).
- **Sprint B (2-3 jours)** : P1 onboarding + per-instrument = 18 h → G4+G7+G11.
- **Sprint C (1 jour)** : P1 sécurité/qualité = 13 h → G8+G9+G13.
- **Attente MTF Phase 3 (J15-J17)** : pas d'attente bloquante en pratique, P1 tient parallèle.
- **Sprint D (1 jour)** : G3 tier re-sweep post-MTF = 6 h.
- **Sprint E (3-4 jours, post-GA)** : P2 = 29 h.

**GA-ready estimé : 7-9 jours dev solo** (P0+P1, sans P2). Post-GA enrichissement P2 sur 3-4 jours supplémentaires.

### 10.3 Conditions de GA (go/no-go)
- ≥ 2 instruments avec PF OOS > 1.2 (XAU + EUR ou XAU + USOIL).
- 100 % presets enrichis (G1).
- Suite tests verte (1366 + ~25 nouveaux).
- `/health` et `/metrics` exposent per-symbol stats.
- Tiers et thresholds documentés, défendables empiriquement (G3 done OU plan B "tier provisoire" assumé).
- Décision GA actée dans `docs/decisions/ga_instruments_2026_q2.md`.

---

**Livrable** : `C:\MyPythonProjects\TradingBOT_Agentic\reports\commercialization_sprint\14_multi_asset_mtf.md`
**Top 3 P0** : (1) G2 décision GA instruments + G1 InstrumentConfig enrichi (7h) ; (2) G5 instrument validation tests + golden replays XAU/EUR (8h) ; (3) G3 tier re-sweep POST-htf_alignment (6h, dépend conv. parallèle Phase 3).
**Heures totales** : 96 h (P0=31, P1=36, P2=29). GA-ready en **7-9 jours solo** sur P0+P1 (sans P2).
