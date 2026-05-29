# Plan de Commercialisation — Catégorie 6 : Backtest & Validation Framework

**Auteur** : Sprint commercialisation — track Backtest & Validation
**Date** : 2026-05-21
**Périmètre** : `src/backtest/`, `src/research/`, `scripts/run_backtest.py`,
`scripts/audit_backtest.py`, `scripts/forensics_walkforward_purged.py`,
`scripts/walk_forward_factor_model.py`, harnais CPCV `src/research/cpcv_harness.py`,
gates `src/research/strategy_gates.py`.
**Statut courant** : Note crédibilité backtest **2/10**
(`reports/eval_18_backtest.md:17`). **Aucune stratégie ne franchit
CI 95 % PF lo > 1.0** sur full sample
(`reports/decision_matrix_2026_04_30.md:29-34`). 0/3 piliers passent les
gates institutionnelles `reports/3_pillars_implementation_2026_05_13.md:14-19`.
**Verdict commercial** : **NON commercialisable** tant que P0 non livré.

---

## 0. Pourquoi cette catégorie est bloquante pour la commercialisation

1. **Risque légal** : tout chiffre publié actuellement est in-sample,
   sans IC, sans coûts. Diffuser ces chiffres = FTC §5 / AMF / MiFID II
   2024/2811 finfluencer-rule = action en classe-action possible
   (`reports/eval_29_compliance_findings.md`).
2. **Risque commercial** : XAU M15 v2 PF point 1.04, CI 95 % lo 0.92
   (`reports/decision_matrix_2026_04_30.md:11-17`) — sans IC lo > 1.0,
   on ne peut **rien** affirmer (« edge » indistinguable de noise après
   bootstrap).
3. **Risque produit** : pas de séparation train/val/test sur l'ensemble
   des sweeps actuels (`scripts/audit_backtest.py:76-85` =
   best-of-7 in-sample, biais Hansen SPA non corrigé).
4. **Risque scientifique** : 1 look-ahead confirmé non-causal
   (`src/environment/multi_timeframe_features.py:554-566`, voir
   `reports/eval_18_backtest.md:128-136`), 1 leak léger via `bfill`
   (`src/environment/environment.py:802`).
5. **Risque opérationnel** : sizing désaligné backtest vs prod
   (`reports/eval_18_backtest.md:402-440`) — le client ne sait pas si
   `pos_mult=1.2` veut dire 1.2 lots, 1.2 % equity ou 1.2 R.

---

## 1. État actuel (Audit)

### 1.1 Ce qui existe — l'os est sain

| Composant | Fichier | Statut |
|---|---|---|
| Harnais event-driven | `src/backtest/state_machine_replay.py` | Déterministe, 18 tests, classify_regime/vol expanding-window causale (`:145-150`) |
| Cost-model wired (Sprint 3) | `src/backtest/state_machine_replay.py:760-868` (`_build_trade`) avec `spread_model` / `slippage_model` injectés | **Branché**, fields `spread_cost/slippage_cost/commission` exposés (`:187-191`). À durcir : commission $7/lot non encore par défaut, pas de news-window flag. |
| Cost-model lib | `src/environment/execution_model.py:18-119` (DynamicSlippageModel + DynamicSpreadModel) | Disponible, défauts XAU `:75-81` (3-8 bps par session, ×3 news) — **conservateur** vs IC Markets RAW (`reports/eval_18_backtest.md:325-339`) |
| Gates institutionnelles | `src/research/strategy_gates.py:189-365` (`evaluate_gates`) | DSR>=1.5, PBO<=0.35, PF lo>1.0, DM p<0.05, n_trades>=30 — contrat unique source of truth |
| CPCV harness | `src/research/cpcv_harness.py:382-507` (`run_cpcv`) | 8 folds × 2 test = 28 paths, purge + embargo, DSR Bailey-LdP 2014, PBO rank-logit |
| Validation bridge | `src/backtest/validation.py:55-163` | `validate_trades_dataframe`, `validate_backtest_artifact` |
| Stress tests | `src/backtest/stress_tests.py` | présent, à auditer |
| Walk-forward skeleton | `reports/eval_18_walkforward_skeleton.py` (PR-ready 320 lignes) | Spec embargo 5j (480 bars) + Train 2019-22 / Val 2023 / Test 2024-25 (`reports/eval_18_backtest.md:255-282`) |
| Pillar suite (Sprint 0-3) | `src/strategies/event_driven_macro.py`, `src/intelligence/conformal_wrapper.py`, `src/intelligence/regime_gate.py` | 46/46 tests passent (`reports/3_pillars_implementation_2026_05_13.md:14-19`) |

### 1.2 Ce qui manque ou est cassé

#### Look-ahead (`reports/eval_18_backtest.md:101-251`)

| Site | Statut | Action |
|---|---|---|
| B2 — `iloc[i+1/+2]` swing detector (`src/environment/multi_timeframe_features.py:554-566`) | **NON-CAUSAL confirmé** | Patch émettre swing à `i+2` (shift(+2)) — **bloque tout backtest MTF** |
| B3 — `bfill()` indicateurs lents (`src/environment/environment.py:802`, `multi_timeframe_features.py:184`) | Faible leak warmup | `WARMUP=500` ou masque NaN explicite |
| B1 — `rolling(center=True)` fractals (`src/environment/strategy_features.py:617-618`) | Mitigé par shift(N) `:637-638` | Test différentiel `tests/test_smart_money_engine_causality.py` |

#### Coûts (`reports/eval_18_backtest.md:303-396`)

- `_build_trade` accepte spread/slippage models mais **ne calcule pas commission $7/lot** (1 lot = 100 oz XAU)
- Pas de drapeau `is_news_window` issu de `BacktestNewsProvider` dans la branche cost (présent en signal-gating mais pas en pricing — à fact-check)
- Spread session Asia 8 bps `execution_model.py:76` ≈ $2 à $2 500 vs broker raw 28 cts ≈ 11 bps → **optimiste de 30 %** en Asia
- Slippage news flagré, multiplier ×3 par défaut `:83` — vs broker raw NFP/FOMC 30-100 cts (`reports/eval_18_backtest.md:334`) ≈ ×8-25 réel
- Aucun **swap/overnight** modélisé (trades XAU H1 peuvent traverser 17h NY = swap point négatif)

#### Walk-forward et CPCV

- `scripts/forensics_walkforward_purged.py` existe (single-split Train 2019-22 / Test 2023-24 / Holdout 2025-26, voir `reports/forensics/forensics_2026_04_30.md` L2) — **pas anchored expanding**, pas multi-window
- **0 exécution du squelette `reports/eval_18_walkforward_skeleton.py`** sur le baseline (`reports/eval_18_backtest.md:286-294`)
- CPCV harness `src/research/cpcv_harness.py:run_cpcv` est ML-flavoured (model_factory + X, y + label_horizon) — **pas rebranché sur la stratégie event-driven** (`state_machine_replay.py`)
- Pas de **block bootstrap** stationary (Politis-Romano) pour respecter l'autocorrélation des R-multiples (corr lag-1 ~0.1 sur baseline)

#### Tests statistiques

- IC bootstrap PF présent (`src/research/strategy_gates.py:152-181` `profit_factor_bootstrap_ci`) — n=1000 par défaut → **passer à n=10000** pour CI 95 % stable
- DSR implémenté (`src/research/cpcv_harness.py:200-246`) — convention prob→z dans `strategy_gates.py:278-284`, à documenter
- Diebold-Mariano implémenté (`src/research/cpcv_harness.py:345-374`) — HAC h=1 OK pour non-overlap, à étendre à h>1 si overlap
- **PBO** : current implementation = single-strategy proxy via median-split (`src/research/cpcv_harness.py:_pbo_from_path_returns:295-307`). Le PBO Bailey-Borwein-LdP-Zhu 2014 strict requiert grille hyperparam → **pas calibrée** sur audit_backtest 7-config sweep
- **Hansen 2005 SPA** : pseudocode fourni `reports/eval_18_backtest.md:572-615` — **non implémenté**
- **Reality Check White 2000** : non implémenté

#### Reproductibilité

- Pas de **manifest JSON** des runs (git_sha, data_hash, config hash, seed) embarqué dans chaque `_summary.json`
- 19 fichiers `replay_*.json/csv` à la racine (`reports/eval_18_backtest.md:62-86`) — confusion mémoire interne vs publiable
- `backtest_trades.csv` à la racine : 1 jumeau orphelin

### 1.3 Verdict synthétique

| Critère | Cible | Actuel | Gap |
|---|---|---|---|
| Walk-forward strict | Anchored expanding + 3+ splits OOS | 1 split single-shot | manque 2-3 windows |
| Coûts | Spread+slip+commission+swap, news-conditional | spread+slip wired, commission absente | +commission +swap +news flag |
| Look-ahead | 0 site non-causal | 1 confirmé (B2 swing detector) | patch shift(+2) |
| CPCV | 28 paths + DSR + PBO sur stratégie principale | CPCV existe (ML), pas branché event-driven | adapter `cpcv_harness` → `event_driven_replay` |
| IC bootstrap | n=10 000, block stationary | n=1 000, iid | bump n + Politis-Romano |
| Hansen SPA | p<0.05 sur sweep | non calculé | implémenter |
| Reproductibilité | manifest + lock files | pas de manifest | écrire schema |
| Decision gates wired | run_backtest.py invoque automatiquement les gates | bridge `validation.py` existe, non câblé CLI | wire automatic gates check at end of run |

---

## 2. Vision cible — Institutional-Grade Backtest Framework

### 2.1 Principes directeurs

1. **Hard-coded gates** : aucun chiffre ne sort de `run_backtest.py`
   sans passer par `evaluate_gates` (`src/research/strategy_gates.py:189`).
   Si `GateResult.all_passed=False`, le JSON est tagué
   `"commercial_status": "REJECTED"` et le rendu texte affiche en
   en-tête `❌ NOT COMMERCIALISABLE`.
2. **Reproducibilité bit-à-bit** : chaque artefact embarque
   `{git_sha, data_hash_sha256, config_hash, seed, framework_version}`.
   Une réexécution avec les mêmes 4 doit produire le même fichier.
3. **Aucun look-ahead** : test différentiel `tests/test_no_lookahead.py`
   qui re-run le replay sur séries tronquées `[0..i]` pour chaque `i` et
   compare aux valeurs émises sur série complète à l'index `i` — la
   stratégie échoue le build si la moindre divergence.
4. **Trois portes statistiques** :
   - Porte 1 — IC bootstrap : `PF_lo_95 > 1.00`
   - Porte 2 — Multiple testing : `DSR_z > 1.5` (sous n_trials =
     |sweep_grid|) ET `Hansen_SPA_p < 0.05`
   - Porte 3 — Out-of-sample : walk-forward 3+ splits, **chaque** split
     OOS doit valider PF_lo > 1.0 OU rejet automatique
5. **Coûts hostiles** : par défaut les coûts sont calibrés
   *worst-of-broker-retail*, pas le RAW IC Markets. Spread Asia 11 bps,
   slip news ×8, commission $7/lot, swap ½-point négatif. Si la
   stratégie résiste, elle résiste partout.
6. **Multi-régime** : chaque chiffre publié est **décomposé** par
   régime (bull/bear/range × low/high-vol) — pas de chiffre globaux qui
   masquent le « long-only sur bull XAU » de
   `reports/forensics/forensics_2026_04_30.md` (L3).
7. **Monte Carlo path-dependent** : pas seulement IC sur PF/Sharpe —
   simulation 10 000 chemins de capital, on publie probabilité
   `P(DD > 25 %)`, `P(ruin)`, et `time-to-recovery_p50`.

### 2.2 Architecture cible

```
scripts/run_backtest.py
    ├─► src/backtest/state_machine_replay.py        (event-driven)
    │       ├─► src/environment/execution_model.py  (coûts hostiles)
    │       ├─► src/backtest/news_replay.py         (news_window flag → cost ×3)
    │       └─► src/backtest/snapshot_store.py      (entry/exit immutability)
    │
    ├─► src/backtest/metrics.py                     (PF, Sharpe ann, Sortino, Calmar, MaxDD)
    │
    ├─► src/backtest/validation.py                  (BRIDGE → gates)
    │       └─► src/research/strategy_gates.py      (DSR + PBO + DM + PF_lo + n_trades)
    │              └─► src/research/cpcv_harness.py (28 paths + bootstrap)
    │
    ├─► src/backtest/walk_forward.py        [NEW]   (anchored expanding, K splits)
    ├─► src/backtest/monte_carlo.py         [NEW]   (10k path-dependent sims)
    ├─► src/backtest/hansen_spa.py          [NEW]   (Hansen 2005 SPA + Reality Check)
    ├─► src/backtest/regime_decompose.py    [NEW]   (heatmap régime × année)
    └─► src/backtest/manifest.py            [NEW]   (git_sha + data_hash + config_hash)
```

---

## 3. Gap Analysis (synthèse pré-plan)

| # | Gap | Sévérité | Bloquant commercial |
|---|---|---|---|
| G1 | Walk-forward anchored multi-split non exécuté sur baseline 6 ans | P0 | OUI |
| G2 | Commission $7/lot + swap + news flag dans cost-model | P0 | OUI |
| G3 | CPCV harness non rebranché stratégie event-driven (seulement ML) | P0 | OUI |
| G4 | IC bootstrap n=1 000 iid → besoin n=10 000 + block stationary | P0 | OUI |
| G5 | B2 swing detector look-ahead à patcher | P0 | OUI (si MTF activé) |
| G6 | Hansen SPA non implémenté | P1 | OUI dès qu'on sweep |
| G7 | `run_backtest.py` ne câble pas automatiquement les gates | P0 | OUI (process) |
| G8 | Pas de manifest reproductibilité | P0 | OUI (audit légal) |
| G9 | Pas de Monte Carlo path-dependent (P(DD>X), P(ruin)) | P1 | OUI (marketing risque) |
| G10 | Pas de regime-conditional metrics dans rapport | P1 | OUI (transparence) |
| G11 | Pas de pipeline multi-asset (XAU + EUR + USOIL) | P1 | NON (mais nécessaire ICP) |
| G12 | Sizing désaligné replay vs prod | P1 | OUI (UX client) |
| G13 | Tests anti-leak (data quarantine) absents | P1 | OUI (audit) |
| G14 | Tier system invalide (`reports/eval_02_confluence.md` Pearson −0.023) | P1 | NON (délégué eval 02) |

---

## 4. Plan d'exécution

### Notation

- **Sévérité** : P0 (commercialisation bloquée), P1 (commercialisation
  dégradée), P2 (nice-to-have)
- **Estimation** : h (heures effectives, hors revue)
- **Acceptance** : critère booléen testable
- **Dépendances** : tickets internes + autres conversations

---

### P0 — Walk-Forward strict sans look-ahead

#### P0.1 — Anchored expanding walk-forward 6 ans (3+ splits) `[12 h]`

- **Fichiers**
  - **NEW** `src/backtest/walk_forward.py` (~250 lignes)
  - **MOD** `scripts/run_backtest.py` (ajouter flag `--walk-forward`)
  - **NEW** `tests/test_walk_forward_anchored.py` (~150 lignes, 10 tests)
- **Contenu**
  - Adapter `reports/eval_18_walkforward_skeleton.py` en module testable
    (suppression des chemins en dur, injection `csv_path` / `calendar_path`)
  - 3 splits anchored expanding :
    - Split 1 : Train 2019-01→2021-12, Embargo 5j, Val 2022-01→2022-12, Test 2023-01→2023-12
    - Split 2 : Train 2019-01→2022-12, Embargo 5j, Val 2023-01→2023-12, Test 2024-01→2024-12
    - Split 3 : Train 2019-01→2023-12, Embargo 5j, Val 2024-01→2024-12, Test 2025-01→2026-04
  - Search space : `enter ∈ {40, 45, 50, 55, 60}`, `exit ∈ {25, 30, 35}`,
    `max_lifetime_bars ∈ {24, 36, 48, 64}` = **60 candidates** (vs 7 actuels
    `scripts/audit_backtest.py:76-85`)
  - Sélection sur VAL via score `log(PF) + 0.5·Sharpe − 0.1·MaxDD_R / total_R`
    avec hard-gate `trades ≥ 50, PF > 1.0, Sharpe > 0.5` (`reports/eval_18_backtest.md:278-285`)
  - **Enrichment per-fold** — `SmartMoneyEngine.analyze()` exécuté sur
    chaque slice indépendamment → empêche `rolling(center=True).shift(N)`
    de traverser la frontière train/val
- **Acceptance**
  - `pytest tests/test_walk_forward_anchored.py -q` = 10/10
  - `python scripts/run_backtest.py --csv ... --walk-forward --out wf.json`
    écrit `wf.json` contenant champs `splits[].train`, `splits[].val`,
    `splits[].test`, `winner_per_split`, `aggregate_test_metrics`
  - Chaque `splits[i].test.profit_factor_lo_95 > 0` mesuré et reporté
    (publication OK ssi tous > 1.0)
- **Dépendances** : aucune (lib `state_machine_replay` + `metrics` + `gates`
  déjà en place)

#### P0.2 — Patch look-ahead B2 (swing detector causal) `[4 h]`

- **Fichiers**
  - **MOD** `src/environment/multi_timeframe_features.py:554-566`
    (émettre swing à `i+2`, pas à `i` — patch `reports/eval_18_backtest.md:148-156`)
  - **NEW** `tests/test_mtf_swing_causality.py` (5 tests)
  - **NEW** `tests/test_no_lookahead.py` (différentiel sur 1 000 bars
    aléatoires — replay sur série tronquée vs complète)
- **Acceptance**
  - Tous les nouveaux tests verts
  - Le replay full 6 ans avec MTF activé produit la même métrique PF à
    `±1e-9` qu'avec MTF désactivé (proxy de causalité)
- **Dépendances** : P0.1 (le test différentiel utilise `walk_forward.py`)

#### P0.3 — Test anti-leak global (différentiel time-travel) `[6 h]`

- **Fichiers**
  - **NEW** `tests/test_backtest_no_lookahead_differential.py` (~200 lignes)
  - **NEW** `src/backtest/leakage_audit.py` (~150 lignes, helper qui
    re-run le pipeline complet sur séries tronquées `[0..i]` pour chaque
    i ∈ échantillon de 100 bars répartis sur 6 ans)
- **Contenu**
  - Pour chaque bar `i` testé : compare valeur signal à `t=i` sur série
    complète vs valeur à `t=i` sur série tronquée à `[0..i]`. Toute
    divergence > tolérance numérique (1e-6) → FAIL
  - Bonus : audit `WARMUP=500` (`reports/eval_18_backtest.md:179-180`)
    pour absorber `bfill` (B3)
- **Acceptance** : `pytest tests/test_backtest_no_lookahead_differential.py
  -q` vert sur 100 bars
- **Dépendances** : P0.2

---

### P0 — Transaction Cost Model durci

#### P0.4 — Commission $7/lot par défaut + swap + news flag `[6 h]`

- **Fichiers**
  - **MOD** `src/backtest/state_machine_replay.py:760-868` (`_build_trade`) :
    ajouter `commission_per_lot` (default $7) et `swap_per_night_pts`
    (default −0.5 point XAU)
  - **MOD** `src/environment/execution_model.py` : ajouter
    `is_news_window: bool` paramètre déjà supporté ligne 102 — câbler
    via `BacktestNewsProvider` (`src/backtest/news_replay.py`)
  - **MOD** `src/backtest/state_machine_replay.py` — propager `news_provider`
    (déjà signature ligne 767) jusqu'à `cost_per_round_trip`
  - **NEW** `tests/test_cost_model_calibration.py` (~150 lignes)
- **Calibration cible**
  - Spread Asia 11 bps (vs 8 bps actuel) — calibration IC Markets RAW
    `reports/eval_18_backtest.md:331`
  - Slip news ×5 (au lieu de ×3 défaut `execution_model.py:83`)
  - Commission $7/lot round-trip, 1 lot XAU = 100 oz → $0.07/oz
  - Swap court XAU long −0.5 pt par nuit (≈ $0.05/oz)
- **Acceptance**
  - Tests verts (5 sessions × 2 news_flag × 2 sides = 20 cas)
  - Re-run baseline 6 ans : PF point estimé attendu **0.94-0.96**
    (était 1.086 sans coûts, `reports/baseline_full.json`)
  - PF lo CI 95 % logé dans `_summary.json` sous `cost_calibration: "hostile_retail_2026Q2"`
- **Dépendances** : aucune (cost model lib déjà présente)

#### P0.5 — Calibration coûts cross-broker (IC Markets + Pepperstone + Oanda) `[4 h]`

- **Fichiers**
  - **NEW** `src/backtest/cost_calibrations.py` (~120 lignes — dict de
    profils brokers)
  - **NEW** `tests/test_cost_calibrations.py`
- **Contenu** : 3 profils — `oanda_retail` (worst, spread 12-30 bps),
  `pepperstone_razor` (médian), `ic_markets_raw` (best). `run_backtest.py`
  accepte `--cost-profile NAME`, défaut `hostile_retail_2026Q2 =
  oanda_retail × 0.8` (légèrement optimiste sur le pire)
- **Acceptance** : `python scripts/run_backtest.py --cost-profile ic_markets_raw`
  produit un PF de 5-8 points supérieur au profil hostile (sanity check
  direction)
- **Dépendances** : P0.4

---

### P0 — CPCV propre + DSR + PBO + DM tests

#### P0.6 — Brancher CPCV harness sur la stratégie event-driven `[12 h]`

- **Fichiers**
  - **NEW** `src/backtest/cpcv_event_driven.py` (~250 lignes)
  - **NEW** `tests/test_cpcv_event_driven.py` (~200 lignes)
- **Contenu**
  - Le `run_cpcv` actuel (`src/research/cpcv_harness.py:382-507`) prend
    `model_factory + X + y` (ML-flavoured). Adapter pour le replay :
    `run_cpcv_event_driven(df, smc, config, news_provider, n_folds=8, k=2, embargo=480_bars)`
  - Émettre 28 paths, chaque path = (train_idx, test_idx) ; sur test_idx
    on exécute `SignalReplay.replay(df.iloc[test_idx])` → `r_series`
  - Embargo `480 bars = 5 jours` couvre `ZSCORE_WINDOW=200` +
    fractal `2N+1=5` + retest ~50 bars (`reports/eval_18_backtest.md:269-275`)
  - Renvoie `CPCVResult` avec DSR, PBO, p25/p75 PF + Sharpe par path
- **Acceptance**
  - 28 paths produits, aucun « skipped » sur dataset 6 ans
  - DSR z-score reporté dans `_summary.json`
  - `tests/test_cpcv_event_driven.py` vérifie reproductibilité
    (seed=42 → mêmes 28 paths) et purge (aucun train index dans
    `[a-embargo-horizon, b+embargo)`)
- **Dépendances** : P0.1 (réutilise `walk_forward.py` pour le slicing)

#### P0.7 — IC bootstrap n=10 000 + block stationary (Politis-Romano) `[8 h]`

- **Fichiers**
  - **MOD** `src/research/strategy_gates.py:152-181`
    (`profit_factor_bootstrap_ci`) : bump `DEFAULT_BOOTSTRAP_N = 10000`,
    ajouter `block_mean_size: int | None = None` paramètre
  - **NEW** `src/backtest/block_bootstrap.py` (~100 lignes — stationary
    block bootstrap Politis-Romano 1994, taille moyenne `~sqrt(n)`)
  - **MOD** `src/research/strategy_gates.py:profit_factor_bootstrap_ci`
    appelle `block_bootstrap` quand `block_mean_size > 0`
  - **NEW** `tests/test_block_bootstrap.py`
- **Contenu**
  - L'iid bootstrap actuel ignore l'autocorrélation des R-multiples
    (cluster de gagnants en bull, cluster de perdants en range)
  - Politis-Romano : on tire des blocs de taille `Geom(1/block_size)`
    avec wrap-around → préserve la structure temporelle
  - Block size cible = `int(sqrt(n_trades))` ; pour n=1 753 → block=42
  - Exposer `pf_lo_95_iid`, `pf_lo_95_block`, `pf_lo_95_block_p10`
    (le plus défensif des trois pour gate)
- **Acceptance**
  - Test : sur séquence AR(1) phi=0.3 simulée n=1 000, le CI block est
    plus large que iid de ≥ 15 %
  - Sur baseline 6 ans : PF lo iid 0.92 → PF lo block ≈ 0.88 attendu
    (élargissement crédible)
- **Dépendances** : aucune

#### P0.8 — Diebold-Mariano vs **multi-baselines** `[4 h]`

- **Fichiers**
  - **MOD** `src/research/strategy_gates.py:189-365` : DM vs 3 baselines
    au lieu d'un seul (zero) — `(zero, buy_and_hold, random_50_50_2R_1R)`
  - **MOD** `src/research/cpcv_harness.py:diebold_mariano` : exposer
    HAC lag dynamique `h = floor(n^{1/3})`
  - **NEW** `tests/test_dm_multi_baseline.py`
- **Acceptance**
  - GateResult expose `dm_pvalue_vs_zero`, `dm_pvalue_vs_bh`,
    `dm_pvalue_vs_random` — gate passe ssi **les 3 < 0.05**
  - Sur baseline 6 ans XAU : `dm_pvalue_vs_bh` attendu **>> 0.05**
    (preuve β-capture, `reports/decision_matrix_2026_04_30.md:96-100`)
- **Dépendances** : P0.4 (coûts ON pour fair-comparison)

#### P0.9 — Câbler les gates dans `run_backtest.py` (CLI hard-fail) `[3 h]`

- **Fichiers**
  - **MOD** `scripts/run_backtest.py` : à la fin, appel
    `validate_trades_dataframe(trades_df, n_trials=GRID_SIZE)` ; injecter
    le résultat dans `_summary.json` sous clé `admission_gates`
  - **MOD** `scripts/run_backtest.py` : exit code 1 si
    `--fail-on-gate-reject` et `result.all_passed=False`
  - **NEW** `tests/test_run_backtest_cli_gates.py` (4 tests)
- **Acceptance**
  - `python scripts/run_backtest.py ... --fail-on-gate-reject` retourne
    exit 1 sur baseline (PF lo 0.92 < 1.00)
  - Rendu texte ajoute en-tête :
    `❌ NOT COMMERCIALISABLE — PF lo 0.917 < 1.00`
    `   (DSR z 0.34 < 1.5, PBO 0.52 > 0.35, DM_vs_zero p 0.12)`
- **Dépendances** : P0.4, P0.7

---

### P0 — Reproductibilité, manifest, anti-leak

#### P0.10 — Manifest reproductibilité (git_sha + data_hash) `[4 h]`

- **Fichiers**
  - **NEW** `src/backtest/manifest.py` (~120 lignes)
  - **MOD** `scripts/run_backtest.py` : inclure `manifest` dans
    `_summary.json`
- **Contenu**
  - `manifest = {git_sha, framework_version="6.0.0", data_hash_sha256,
    config_hash_sha256, seed, python_version, numpy_version,
    pandas_version, n_cores, ts_start, ts_end, duration_sec}`
  - `data_hash` = sha256 du fichier CSV (sans dépendre du timestamp file)
  - `config_hash` = sha256 du JSON canonicalisé `StateMachineConfig`
- **Acceptance**
  - Re-run identique = mêmes hash, même `summary.json` bit-à-bit
  - Sous-tâche : ajouter `--check-manifest path/to/old.json` pour
    valider l'identité de deux runs

#### P0.11 — Archiver les 19 replay_*.json à la racine `[1 h]`

- **Action** : `git mv replay_*.json replay_*.csv backtest_trades.csv
  reports/replays_archive/2026-04-23-iteration/` (chemin recommandé
  `reports/eval_18_backtest.md:96-98`)
- **Acceptance** : `git status` clean, aucune référence cassée dans
  `tests/` ni `scripts/`
- **Dépendances** : aucune

---

### P1 — Multi-asset backtest pipeline

#### P1.1 — Extension multi-asset (XAU + EUR + USOIL) `[16 h]`

- **Fichiers**
  - **NEW** `src/backtest/multi_asset_runner.py` (~250 lignes)
  - **MOD** `scripts/run_backtest.py` : accepter
    `--symbols XAUUSD,EURUSD,USOIL`
  - **NEW** `tests/test_multi_asset_runner.py`
- **Contenu**
  - Pour chaque symbol : ré-instancier `SmartMoneyEngine` (pas de partage
    d'état), exécuter `walk_forward.run`, agréger résultats
  - Calibration `cost_profile` par symbol (FX = 1 pip vs XAU 3 bps)
  - **Important** : pas d'optim hyperparam cross-asset (sinon biais
    multiple-testing × N_symbols) — on prend la config **gagnante sur
    XAU OOS** et on l'applique aux autres assets en pure OOS
- **Acceptance**
  - Rapport `_summary.json` contient `per_symbol_metrics: {XAUUSD: {...},
    EURUSD: {...}, USOIL: {...}}`
  - **Gate cross-asset** : au moins 2/3 symbols PF lo > 1.0 OOS — sinon
    `commercial_status="REJECTED:NOT_CROSS_ASSET"`. Ce gate confirme/infirme
    « β-capture XAU » (`reports/decision_matrix_2026_04_30.md:96-100`)
- **Dépendances** : P0.1, P0.4, P0.6

#### P1.2 — Regime-conditional metrics (heatmap régime × année) `[8 h]`

- **Fichiers**
  - **NEW** `src/backtest/regime_decompose.py` (~180 lignes)
    (basé pseudocode K8 `reports/eval_18_backtest.md:629-678`)
  - **MOD** `src/backtest/report.py` : ajouter section
    `regime_breakdown_table` dans `render_text`
  - **NEW** `tests/test_regime_decompose.py`
- **Contenu**
  - Pour chaque trade : tag `(regime, vol_regime, year, session)` à
    entry. Calculer PF, win-rate, expectancy par cellule
  - Heatmap (Markdown table) régime × année — révèle « long-only sur
    bull XAU 2024-2026 » de `reports/forensics/forensics_2026_04_30.md` L3
  - Gate complémentaire : « si **toute** la profitabilité concentrée
    sur 1 régime → `commercial_status="REJECTED:REGIME_DEPENDENT"` »
    (seuil : 80 % des R sur 1 régime / année)
- **Acceptance**
  - Sur baseline 6 ans XAU : heatmap montre PF 1.55 long-2024-26 vs
    0.81 long-2019-23 — déjà connu (`reports/forensics/forensics_2026_04_30.md` L3) ;
    le pipeline doit le reproduire automatiquement
- **Dépendances** : P0.1

#### P1.3 — Hansen 2005 SPA + White Reality Check `[10 h]`

- **Fichiers**
  - **NEW** `src/backtest/hansen_spa.py` (~200 lignes)
    (basé pseudocode K7 `reports/eval_18_backtest.md:572-615`)
  - **NEW** `src/backtest/white_reality_check.py` (~100 lignes)
  - **NEW** `tests/test_hansen_spa.py`
- **Contenu**
  - Hansen SPA stationary block bootstrap n_boot=5 000, block
    `~sqrt(T)` bars
  - White RC plus simple (variante moins puissante) en backup
  - GateResult propage `spa_p` ; gate global : pass ssi `spa_p < 0.05`
- **Acceptance**
  - Sur le sweep 60-candidates de P0.1 : SPA p-value retournée et
    intégrée à `_summary.json`. Si p > 0.05, marketing must use
    qualitative language only
- **Dépendances** : P0.1, P0.6

#### P1.4 — Sizing alignment (replay vs prod) `[6 h]`

- **Fichiers**
  - **MOD** `src/intelligence/signal_state_machine.py` (StateMachineConfig) :
    ajouter `position_sizing_pct: float = 0.005` (0.5 % equity/trade)
  - **MOD** `src/backtest/state_machine_replay.py:_build_trade` :
    calculer `size = equity * pct / (initial_risk * contract_size)`,
    pondérer `r_mult_weighted = r_mult * pos_mult` (régime × news)
  - **NEW** `tests/test_sizing_alignment.py`
- **Acceptance**
  - Replay sur baseline avec/sans `position_sizing_pct` produit chiffres
    identiques en R-space, divergents en $-space (cohérence comptable)
  - Sortie `_summary.json` : champ `equity_curve_usd` reflète sizing
- **Dépendances** : aucune (mais lit `pos_mult` qui dépend de
  `confluence_detector.py:317-334`)

---

### P2 — Monte Carlo path-dependent + stress tests

#### P2.1 — Monte Carlo path-dependent (10 000 chemins) `[10 h]`

- **Fichiers**
  - **NEW** `src/backtest/monte_carlo.py` (~250 lignes)
  - **NEW** `tests/test_monte_carlo.py`
- **Contenu** (étend pseudocode K6 `reports/eval_18_backtest.md:454-533`)
  - Bootstrap **path-dependent** : tirer n_sims=10 000 chemins de
    n_trades=N R-multiples (block stationary), simuler courbe d'equity
    cumulée, calculer pour chaque chemin : maxDD, time-to-recovery,
    final_equity
  - Renvoie distributions IC 95 % de `P(maxDD > 0.25)`, `P(ruin =
    equity ≤ 0)`, `time-to-recovery_p50`, `final_equity_p25/p50/p75`
  - Test H0 random-walk (50 % win, 2:1 RR) : p-value Mann-Whitney sur
    distribution final_equity observée vs H0
- **Acceptance**
  - Sur baseline 6 ans : reporting `P(maxDD > 25 %) = 0.74` (typique
    pour PF 1.04) — chiffre utilisable en disclaimers marketing
  - p-value vs random walk reportée
- **Dépendances** : P0.7

#### P2.2 — Stress tests scénarios « what-if » `[6 h]`

- **Fichiers**
  - **MOD** `src/backtest/stress_tests.py` (présent, à étendre)
  - **NEW** `tests/test_stress_tests.py`
- **Contenu**
  - Scénario 1 — « coûts ×1.5 » (spread + slip + commission tous ×1.5)
  - Scénario 2 — « retour au régime range 2019-2022 only » (sous-sample)
  - Scénario 3 — « 30 % des trades arbitrairement supprimés » (data outage)
  - Scénario 4 — « slippage news ×3 supplémentaire » (flash crash NFP)
  - Chaque scénario : pass/fail gates, rapport delta vs baseline
- **Acceptance** : tableau `stress_summary` dans `_summary.json` avec
  PF lo, MaxDD, DSR par scénario
- **Dépendances** : P0.4, P0.7

#### P2.3 — OOS live tracker (paper-trade nightly + alerte 1σ) `[12 h]`

- Reporter à l'eval 18 (`reports/eval_18_backtest.md:686-770`, spec K9
  déjà rédigée) — exécution post-MVP
- Schema SQLite `oos_trades` + `oos_metrics_daily` + `published_baseline`
- Cron nightly `scripts/oos_live_tracker.py --alert-email …`
- **Hors scope sprint 6 immédiat — listé pour traçabilité**

---

## 5. Tests & validation

### 5.1 Unit tests sur les gates

- `tests/test_strategy_gates.py` (existant, 12 tests) — étendre :
  - Test 13 : sur série iid Bernoulli 50 %, PBO ≈ 0.50 ± 0.05 (sanity)
  - Test 14 : sur série positive expectancy, DSR z > 0
  - Test 15 : sur n_trades < min_trades, `all_passed = False`
  - Test 16 : avec block bootstrap, CI plus large que iid sur série AR(1)
- `tests/test_validation_bridge.py` (nouveau) : vérifier que
  `validate_backtest_artifact(json)` rend même résultat que
  `evaluate_gates(returns_extracted_from_json)`

### 5.2 Regression tests sur backtest

- `tests/test_backtest_regression_baseline.py` (nouveau)
  - Snapshot `_summary.json` baseline 6 ans (post-coûts) check-summé
  - Tout PR qui change `state_machine_replay.py`, `execution_model.py`,
    `confluence_detector.py` doit régénérer ce snapshot
- `tests/test_backtest_regression_walkforward.py` (nouveau)
  - Snapshot des 3 splits walk-forward, métriques par split

### 5.3 Differential tests anti-leak

- `tests/test_backtest_no_lookahead_differential.py` (P0.3)
- `tests/test_mtf_swing_causality.py` (P0.2)
- `tests/test_smart_money_engine_causality.py`
  (`reports/eval_18_backtest.md:117-124` — test différentiel B1)

### 5.4 Property-based tests

- Hypothesis sur `profit_factor_bootstrap_ci` : pour toute série
  injectée, `lo <= point <= hi`, `lo >= 0`
- Hypothesis sur `cpcv_path_indices` : aucun train_idx ∈ test_idx,
  `|train ∪ test ∪ forbidden| == n_samples`

### 5.5 CI/CD

- **NEW** `.github/workflows/backtest_validation.yml`
  - Lancer suite `pytest tests/test_strategy_gates.py
    tests/test_cpcv_event_driven.py tests/test_walk_forward_anchored.py
    tests/test_backtest_no_lookahead_differential.py
    tests/test_block_bootstrap.py tests/test_hansen_spa.py -q` à chaque PR
  - Job nightly : `python scripts/run_backtest.py --csv data/...
    --walk-forward --fail-on-gate-reject` sur baseline → si gate FAIL,
    issue automatique « marketing FROZEN until fix »

---

## 6. Sécurité (data leakage detection, time-travel guards)

### 6.1 Garde-fous au runtime

- **Decorator** `@no_lookahead` sur fonctions critiques
  (`state_machine_replay.classify_*`, `cost_per_round_trip`) qui
  vérifie en debug-mode que le DataFrame passé en argument est tronqué
  à un index ≤ celui du bar courant.
- **Snapshot store** (`src/backtest/snapshot_store.py` existant) : à
  auditer — doit garantir que chaque transition observe uniquement les
  données disponibles à `bar_ts`.

### 6.2 Data quarantine

- **NEW** `src/backtest/data_quarantine.py` (~100 lignes)
  - Au chargement CSV : split anchored 80/20 train/holdout d'office.
    Holdout = TOUCH-ONLY-ONCE — toute lecture loggée et compteurée.
    Plus de 1 lecture sur la fenêtre vie projet → exception
    `HoldoutBurnedError`.
  - **À justifier** : sur le baseline existant, le holdout 2025+ est
    déjà fortement biaisé (lu N fois). On le marque
    `holdout_status: "burned"` dans manifest et on n'utilise que les
    splits 2019-2024 pour la décision Go/No-Go.

### 6.3 Embargo enforcement

- Test de régression : pour chaque CPCV path, vérifier `min(test_idx) -
  max(train_idx) >= embargo + label_horizon` et symétrique. Le harness
  actuel le code (`src/research/cpcv_harness.py:108-134`) mais aucun
  test n'audite le contrat.

### 6.4 Random seed pinning

- Tous les bootstraps ont `seed=42` par défaut
  (`src/research/strategy_gates.py:79`). Doc : un seed différent ne
  doit PAS changer la décision pass/fail — si oui, sample trop petit.

---

## 7. Métriques cibles (commercialisation)

### 7.1 Grille de gates (hard-coded, `src/research/strategy_gates.py`)

| Gate | Seuil | Source |
|---|---|---|
| n_trades | ≥ 30 OOS | López de Prado 2018 |
| DSR z-score | ≥ 1.5 | Bailey & López de Prado 2014 |
| PBO | ≤ 0.35 | Bailey-Borwein-LdP-Zhu 2014 |
| PF lower CI 95 % (block bootstrap n=10 000) | > 1.00 | Politis-Romano 1994 |
| DM p-value vs zero | < 0.05 | Diebold-Mariano 1995 |
| DM p-value vs buy-and-hold | < 0.05 | Identifier alpha vs β |
| Hansen SPA p | < 0.05 | Hansen 2005 |
| Cross-asset transfer (2/3 symbols PF lo > 1.0) | TRUE | Anti β-capture |

### 7.2 Métriques de performance reportées

| Métrique | Définition | Statut |
|---|---|---|
| PF (point, CI 95 % block, CI 95 % iid) | Gross win / Gross loss | EXISTANT, extend |
| Sharpe annualisé (M15 = √(252×96)) | μ/σ × √f | EXISTANT `cpcv_harness.py:163` |
| Sortino annualisé | μ/σ_negative × √f | EXISTANT `metrics.py` |
| Calmar | CAGR / |MaxDD| | EXISTANT `metrics.py` |
| MaxDD (% equity + R-space) | drawdown peak-to-trough | EXISTANT |
| Expectancy (R/trade) | mean(R) | EXISTANT |
| **MAR / Recovery Factor** | total_R / |MaxDD_R| | À ajouter |
| **Ulcer Index** | RMS drawdown | À ajouter |
| **P(maxDD > 25 %)** | Monte Carlo | NEW P2.1 |
| **P(ruin)** | Monte Carlo P(equity ≤ 0) | NEW P2.1 |
| **Pearson(score, R)** | corrélation score → trade outcome | EXISTANT (cf. eval 02) |

### 7.3 Reporting commercial cible (post-implémentation P0)

Si `all_passed = True` sur baseline corrigé (peu probable au vu de
`reports/decision_matrix_2026_04_30.md`), reporting publiable :

```
Smart Sentinel AI — Backtest XAU M15 (2019-2025, 6 ans)
[manifest] git=4f7e8a2, data_sha=ab12…, framework=6.0.0, seed=42
Coûts : spread 11 bps Asia / 3 bps London, slippage news ×5,
        commission $7/lot, swap −0.5 pt/nuit (profile hostile_retail_2026Q2)

OOS metrics (walk-forward 3 splits, anchored expanding) :
  - Profit Factor : 1.42 [CI95 block 1.18, 1.69]            ✅ PF lo > 1.0
  - Sharpe ann.   : 0.84                                     ✅ ≥ 0.8
  - MaxDD         : -19 %                                    ✅ < 25 %
  - DSR z-score   : 1.78  (sous 60 trials)                   ✅ ≥ 1.5
  - PBO           : 0.18                                     ✅ ≤ 0.35
  - DM vs zero    : p = 0.014                                ✅ < 0.05
  - DM vs B&H     : p = 0.041                                ✅ < 0.05  (alpha ≠ β)
  - Hansen SPA    : p = 0.038                                ✅ < 0.05

Cross-asset : XAUUSD ✅, EURUSD ⚠️ PF lo 0.94, USOIL ✅
  → status : COMMERCIALISABLE (sous-réserve disclaimer EURUSD)
```

Si **un seul** gate fail → marketing **interdit** au-delà du langage
qualitatif §4 de `BACKTEST_LEGAL_GUARDRAILS.md`.

---

## 8. Risques & mitigations

| # | Risque | Impact | Mitigation |
|---|---|---|---|
| R1 | Baseline 6 ans XAU continue à fail après P0 (probable) | Marketing FROZEN | Plan B `reports/decision_matrix_2026_04_30.md` (pivot B2B-API) + maintien framework pour évals futures |
| R2 | Adapter CPCV event-driven > 12h | Retard P0 | Réutiliser `walk_forward.py` pour le slicing, ne pas réinventer |
| R3 | Block bootstrap n=10 000 trop lent (>30s) | UX dev | Vectoriser via `np.random.choice(replace=True)` + numba si besoin ; cap à n=5 000 si > 60s |
| R4 | Holdout 2025-26 déjà brûlé | Audit légal | Marquer `burned` dans manifest, n'utiliser que splits 2019-24 pour décision |
| R5 | Sweep 60 candidats × 3 splits = 180 runs × 30s = 90 min | Patience dev | Parallel `joblib.Parallel(n_jobs=-1)` — 4 cores → 22 min |
| R6 | Patch B2 (P0.2) casse comportement actuel de l'agent MTF | Régressions | Inactif en runtime actuel `reports/eval_18_backtest.md:142-146`, juste ajouter test |
| R7 | Coûts hostiles tuent toute profitabilité (`reports/eval_18_backtest.md:380`) | Confirmation NO-GO | Acceptable — preuve scientifique nécessaire pour décision pivot |
| R8 | Confusion utilisateur sur 3 cost-profiles | UX | `--cost-profile` par défaut = hostile, autres profils opt-in et flaggés "advisory" |
| R9 | Hansen SPA implémentation buggy | Faux verdict | 4 tests : null=Bernoulli (p uniform), null=ones×0 (skip), test paper-replication Hansen 2005 §5.1 |
| R10 | Sizing alignement P1.4 casse contrats Telegram (`pos_mult` semantics changent) | Régressions clients | Maintenir backward-compat sous flag `legacy_unit_R`, deprecation 1 sprint |

---

## 9. Dépendances inter-catégories

| De | À | Nature |
|---|---|---|
| **6. Backtest** | 1. Data | Données 6 ans propres XAU + EUR + USOIL — `reports/eval_08_data_providers.md` flagge 5/6 presets sans CSV. EUR 99.6 % coverage acquis (Sprint W3). USOIL à acquérir. |
| **6. Backtest** | 2. Algo / SMC | Détecteurs OB ICT + FVG 0.4 ATR + retest 0.25 ATR (`reports/eval_03_smart_money.md`). Aucun edge mesurable jusqu'à patch P0 de cette eval. |
| **6. Backtest** | 3. ML / Score | Tier system invalide (`reports/eval_02_confluence.md` Pearson −0.023). Remplacement par LightGBM (audit P0-1 sprint 4 scoring). Le backtest doit lire le nouveau score sans changement d'API. |
| **6. Backtest** | 4. Vol forecaster | HAR défaut (`VOL_MODE=har`). Le replay utilise un naïf ATR — alignement non-critique. |
| **6. Backtest** | 5. Risk | `position_sizing_pct` dans `StateMachineConfig` (P1.4) doit être consommé par `risk_manager.py` côté prod (eval 19). |
| **6. Backtest** | 7. News | `BacktestNewsProvider` (`src/backtest/news_replay.py`) doit exposer `is_news_window(bar_ts) -> bool` pour le cost flag P0.4. Présent. |
| **6. Backtest** | 9. Compliance | Tout chiffre commercialisable franchit cette eval. Disclaimers `BACKTEST_LEGAL_GUARDRAILS.md` § 2.1-2.2 préfaçant tout reporting. |
| **6. Backtest** | 10. GTM | Marketing FROZEN tant qu'aucune config ne passe TOUS les gates. Si pivot B2B-API, le framework reste utilisable comme « certification context layer » pour brokers partenaires. |
| **Autre conversation** | 6. Backtest | Phase 2 6-yr XAU replay + Phase 3 tier sweep → ne dupliquer ni audit_backtest sweep ni replay full. **Coordination** : cette catégorie livre l'**infrastructure** (walk-forward, gates, manifest), l'autre conv exécute les **runs** + interprète. Interface commune = `_summary.json` schema. |

---

## 10. Estimation totale & timeline

### 10.1 Estimation par phase

| Phase | Tickets | Heures dev | Heures revue (×0.3) | Total | Bloquant |
|---|---|---:|---:|---:|---|
| **P0 — Walk-forward sans look-ahead** | P0.1 + P0.2 + P0.3 | 22 | 7 | **29 h** | OUI |
| **P0 — Cost model** | P0.4 + P0.5 | 10 | 3 | **13 h** | OUI |
| **P0 — CPCV + DSR + PBO + DM** | P0.6 + P0.7 + P0.8 + P0.9 | 27 | 8 | **35 h** | OUI |
| **P0 — Reproductibilité** | P0.10 + P0.11 | 5 | 2 | **7 h** | OUI |
| **Sous-total P0** | 10 tickets | **64** | **20** | **84 h** | |
| **P1 — Multi-asset + régime + SPA + sizing** | P1.1 + P1.2 + P1.3 + P1.4 | 40 | 12 | **52 h** | partiel |
| **P2 — Monte Carlo + stress + OOS tracker** | P2.1 + P2.2 + P2.3 | 28 | 8 | **36 h** | NON (post-MVP) |
| **TOTAL FRAMEWORK INSTITUTIONNEL** | 17 tickets | **132** | **40** | **172 h** | |

### 10.2 Timeline propose (1 dev solo, 8 h/sem)

- **Sem 1-3 (24 h)** : P0.1 walk-forward + P0.2 patch B2 + P0.3 anti-leak diff
- **Sem 4 (8 h)** : P0.4 commission/swap + P0.5 cost profiles
- **Sem 5-7 (24 h)** : P0.6 CPCV event-driven + P0.7 block bootstrap
- **Sem 8 (8 h)** : P0.8 DM multi-baseline + P0.9 wire gates CLI
- **Sem 9 (8 h)** : P0.10 manifest + P0.11 archive
- **GATE Sprint 0** — re-run baseline 6 ans avec P0 complet. Verdict
  pass/fail des admission gates. Si pass : continuer P1. Si fail
  (probable) : décision **pivot B2B-API** ou « FREE-only dégradé »
  (`reports/decision_matrix_2026_04_30.md:90-110`)
- **Sem 10-12 (24 h)** : P1.1 multi-asset + P1.2 regime decompose
- **Sem 13-14 (16 h)** : P1.3 Hansen SPA + P1.4 sizing alignment
- **Sem 15-17 (24 h)** : P2.1 Monte Carlo + P2.2 stress tests
- **Sem 18+ (12 h)** : P2.3 OOS tracker post-MVP

**Délai total** : 12-17 semaines à 8 h/sem (= 3-4 mois solo). En
parallèle d'autres catégories (eval 02, eval 19), c'est **le sprint
infrastructurel** qui débloque la commercialisation.

### 10.3 Critère Go/No-Go à fin P0

Après livraison P0 (8 sem, 84 h) :

1. **Re-run** `python scripts/run_backtest.py --csv data/XAU_15MIN_2019_2026.csv
   --walk-forward --cost-profile hostile_retail_2026Q2 --fail-on-gate-reject`
2. Lire `admission_gates.all_passed` dans `_summary.json`
3. **Si TRUE** : continuer P1 + P2, préparer landing avec chiffres OOS
   et disclaimers
4. **Si FALSE** : décision-arbre `reports/decision_matrix_2026_04_30.md`
   - Option A : Pivot B2B-API (recommandé par décision matrix)
   - Option B : FREE-only XAU M15 v2 avec disclaimers + forward-test
     90 j paper avant monétisation
   - Option C : Stopper TradingBOT_Agentic (cap dur 8 semaines —
     `reports/decision_matrix_2026_04_30.md:135`)

---

## Annexes

### A. Schema cible `_summary.json`

```json
{
  "manifest": {
    "git_sha": "4f7e8a2…",
    "framework_version": "6.0.0",
    "data_hash_sha256": "ab12…",
    "config_hash_sha256": "cd34…",
    "seed": 42,
    "ts_start": "2026-05-21T08:00:00Z",
    "duration_sec": 47.3
  },
  "cost_calibration": "hostile_retail_2026Q2",
  "splits": [
    {"split": 1, "train": "...", "val": "...", "test": "...", "metrics": {...}},
    {"split": 2, ...}, {"split": 3, ...}
  ],
  "aggregate_test_metrics": {
    "profit_factor": 1.42,
    "profit_factor_lo_block": 1.18,
    "profit_factor_lo_iid": 1.22,
    "sharpe_ann": 0.84,
    "max_dd_pct": -19.2,
    "calmar": 0.51,
    "n_trades": 1187
  },
  "admission_gates": {
    "all_passed": true,
    "n_trades": true,
    "dsr": {"z": 1.78, "pass": true},
    "pbo": {"value": 0.18, "pass": true},
    "pf_lo": {"block": 1.18, "iid": 1.22, "pass": true},
    "dm": {"vs_zero": 0.014, "vs_bh": 0.041, "vs_random": 0.022, "pass": true},
    "hansen_spa": {"p": 0.038, "pass": true}
  },
  "cross_asset_transfer": {
    "XAUUSD": {"pf_lo": 1.18, "pass": true},
    "EURUSD": {"pf_lo": 0.94, "pass": false},
    "USOIL":  {"pf_lo": 1.05, "pass": true},
    "transferred_2_of_3": true
  },
  "regime_breakdown": {...},
  "monte_carlo": {
    "p_maxdd_gt_25pct": 0.21,
    "p_ruin": 0.03,
    "time_to_recovery_p50_days": 47
  },
  "commercial_status": "COMMERCIALISABLE",
  "trades": [...]
}
```

### B. Files modifiés/créés (récapitulatif)

**Créés (P0)**
- `src/backtest/walk_forward.py`
- `src/backtest/cpcv_event_driven.py`
- `src/backtest/block_bootstrap.py`
- `src/backtest/manifest.py`
- `src/backtest/leakage_audit.py`
- `src/backtest/cost_calibrations.py`
- `tests/test_walk_forward_anchored.py`
- `tests/test_cpcv_event_driven.py`
- `tests/test_block_bootstrap.py`
- `tests/test_mtf_swing_causality.py`
- `tests/test_backtest_no_lookahead_differential.py`
- `tests/test_cost_model_calibration.py`
- `tests/test_cost_calibrations.py`
- `tests/test_dm_multi_baseline.py`
- `tests/test_run_backtest_cli_gates.py`

**Modifiés (P0)**
- `src/research/strategy_gates.py:79` (DEFAULT_BOOTSTRAP_N 1000→10000) ;
  `:152-181` (block bootstrap parameter)
- `src/research/cpcv_harness.py:345-374` (DM HAC dynamique)
- `src/backtest/state_machine_replay.py:760-868` (`_build_trade`
  commission + swap)
- `src/environment/execution_model.py:75-119` (calibration cross-broker)
- `src/environment/multi_timeframe_features.py:554-566` (patch swing
  shift(+2))
- `src/environment/environment.py:802` (WARMUP=500 ou masque NaN)
- `scripts/run_backtest.py` (flag `--walk-forward`, `--cost-profile`,
  `--fail-on-gate-reject`, manifest)

**Créés (P1-P2)**
- `src/backtest/multi_asset_runner.py`
- `src/backtest/regime_decompose.py`
- `src/backtest/hansen_spa.py`
- `src/backtest/white_reality_check.py`
- `src/backtest/monte_carlo.py`
- `tests/test_multi_asset_runner.py`
- `tests/test_regime_decompose.py`
- `tests/test_hansen_spa.py`
- `tests/test_monte_carlo.py`
- `tests/test_stress_tests.py`
- `tests/test_sizing_alignment.py`

### C. Références scientifiques

- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. — CPCV ch. 7, purging, embargo.
- Bailey, D.H., & López de Prado, M. (2014). *The Deflated Sharpe
  Ratio*. — DSR.
- Bailey, Borwein, López de Prado, Zhu (2014). *The Probability of
  Backtest Overfitting*. — PBO rank-logit.
- Diebold, F.X., & Mariano, R.S. (1995). *Comparing Predictive
  Accuracy*. — DM test.
- Hansen, P.R. (2005). *A Test for Superior Predictive Ability*. — SPA.
- White, H. (2000). *A Reality Check for Data Snooping*. — RC test.
- Politis, D.N., & Romano, J.P. (1994). *The Stationary Bootstrap*. —
  Block bootstrap.

---

## Synthèse 5 lignes

- **Chemin** : `C:\MyPythonProjects\TradingBOT_Agentic\reports\commercialization_sprint\06_backtest_validation.md`
- **Top P0-1** : Walk-forward anchored 3-splits + patch B2 swing detector + tests anti-leak différentiels (P0.1 + P0.2 + P0.3 = **22 h**)
- **Top P0-2** : Coûts hostiles (commission $7/lot + swap + news flag) + cost-profiles cross-broker (P0.4 + P0.5 = **10 h**)
- **Top P0-3** : CPCV event-driven branché + block bootstrap n=10k + DM multi-baseline + wire gates CLI (P0.6+P0.7+P0.8+P0.9 = **27 h**) — débloque le verdict commercial automatique
- **Heures totales** : **84 h P0 (bloquant) + 52 h P1 + 36 h P2 = 172 h total**, soit ~12-17 sem solo à 8 h/sem ; **gate Go/No-Go à fin P0 (sem 9)** = re-run baseline avec gates wired → décision pivot B2B-API ou commercialisation FREE-tier dégradée
