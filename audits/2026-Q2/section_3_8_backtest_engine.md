# Audit Phase 1 — Section 3.8 : Backtest Engine & Statistical Validation

**Date** : 2026-05-15
**Auditeur** : Claude (Lead Quant Architect — Sprint 0 institutional overhaul)
**Périmètre** : `src/backtest/*.py` (4 fichiers, 1 714 LOC), `src/environment/execution_model.py` (118 LOC), `src/research/cpcv_harness.py` (507 LOC), `src/research/strategy_gates.py` (389 LOC), `scripts/run_backtest.py`, `scripts/run_baseline_sprint0.py`, `scripts/audit_backtest.py`, `scripts/forensics_walkforward_purged.py`.
**Référence amont** : `reports/eval_18_backtest.md` (note 2/10 ❌), `reports/3_pillars_implementation_2026_05_13.md`, `reports/a1_verdict_2026.md`.

---

## Score : **3.5 / 10** ❌ (vs eval_18 = 2/10)

Le score remonte légèrement vs eval_18 (2/10) car la **machinerie statistique d'admission a été construite** dans `src/research/cpcv_harness.py` et `src/research/strategy_gates.py` (Sprint 0 / Pilier 1, mai 2026) et **a déjà recalé deux candidats** (A1 stack, naive event-driven). C'est une avancée majeure.

Mais le score reste **< 5/10** parce que :

1. La machinerie CPCV+DSR+PBO vit dans `src/research/` et n'est **pas appelable depuis `src/backtest/`** — couplage zéro entre l'engine production et l'admission gates. Le runner principal `scripts/run_backtest.py` n'invoque ni CPCV, ni DSR, ni PBO, ni les gates `strategy_gates.evaluate_gates`.
2. **Coûts transactionnels** (`DynamicSpreadModel`, `DynamicSlippageModel` dans `src/environment/execution_model.py`) **existent mais ne sont jamais branchés** dans `run_backtest.py` ni `run_baseline_sprint0.py`. Commission est hardcodée à `0.0` (state_machine_replay.py:834). Le code path `if spread_model is not None` (ligne 791) est mort sur tous les pipelines.
3. **Aucun walk-forward** dans `src/backtest/state_machine_replay.py`. C'est un single-pass forward-only sur 100 % des bars. Le seul walk-forward du repo (`scripts/forensics_walkforward_purged.py`) est un script ad-hoc qui dépend de `quant_audit_2026_04_30.py` (pas de l'engine `state_machine_replay`), donc 0 réutilisabilité.
4. **Reproductibilité bit-à-bit cassée par `uuid.uuid4()`** dans `confluence_detector.py:343` — chaque run produit des `signal_id` différents (mais les trades sont sinon identiques, voir §10).
5. **Calmar formule incorrecte** (`metrics.py:254` : `total_r / max_drawdown_r` au lieu de `annualised_return / max_drawdown` — confusion d'unités).
6. **Sharpe calculé avec deux formules différentes** dans le même engine : `state_machine_replay._compute_metrics:716-718` utilise `statistics.stdev` (ddof=1, sample) ; `metrics.compute_performance:213` utilise `statistics.pstdev` (ddof=0, population). Sur 30 trades l'écart est ~3 %, sur petits échantillons davantage.

---

## 0. Repères inventaire

| Fichier                                          | LOC  | Statut         | Rôle                                             |
| ------------------------------------------------ | ---- | -------------- | ------------------------------------------------ |
| `src/backtest/state_machine_replay.py`           | 914  | Production     | Pipeline replay deterministique principal        |
| `src/backtest/metrics.py`                        | 383  | Production     | PerformanceMetrics institutionnelles             |
| `src/backtest/report.py`                         | 225  | Production     | Rendu text + JSON                                |
| `src/backtest/news_replay.py`                    | 192  | Production     | Provider news offline (blackout ±30min)          |
| `src/environment/execution_model.py`             | 118  | **Orphelin**   | Spread + slippage models — non-câblés en backtest |
| `src/research/cpcv_harness.py`                   | 507  | Sprint 0       | CPCV 28 paths + DSR + PBO + DM + Holm            |
| `src/research/strategy_gates.py`                 | 389  | Sprint 0       | Gates DSR>=1.5, PBO<=0.35, PF_lo>1.0, DM_p<0.05  |
| `scripts/run_backtest.py`                        | 244  | Production CLI | Wrapper principal `python scripts/run_backtest.py` |
| `scripts/run_baseline_sprint0.py`                | 415  | Sprint 0       | Orchestrateur baseline XAU+EURUSD avec bootstrap |
| `scripts/audit_backtest.py`                      | 993  | Audit interne  | Sweep 7 ans, 7 configs, stratification per-year/hour/tier |
| `scripts/forensics_walkforward_purged.py`        | 100+ | Forensics      | Walk-forward train/embargo/test/embargo/holdout sur 7 ans |

---

## 1. Walk-forward — **finding P0**

### 1.1 État actuel

`SignalReplay.run()` (`state_machine_replay.py:445-611`) fait **un seul forward-pass linéaire** :

```python
# state_machine_replay.py:482
for i in range(self.warmup_bars, len(enriched_df)):
    row = enriched_df.iloc[i]
    ...
```

Pas de rolling window. Pas de refit. Pas d'IS/OOS split. Toutes les 100 k bars sont traitées comme **une seule fenêtre**. Conséquence : tous les `replay_*.json` de la racine (19 fichiers inventoriés dans eval_18 §K1) ainsi que le `reports/baseline/baseline_report.json` (Sprint 0 Batch 0.2) sont **in-sample single-fold**.

### 1.2 Le walk-forward du repo

Un walk-forward existe dans `scripts/forensics_walkforward_purged.py` (~250 LOC) :

- Segments : Train (2019-2022) → Embargo (5 jours) → Test (2023-2024) → Embargo → Holdout (2025-2026)
- Réinitialise la state machine entre segments (commentaire L91-92 : « la sequence d'etats StateMachine se reset entre segments, ce qui est correct »)
- Bootstrap PF par segment

**Problème** : ce walk-forward dépend de `quant_audit_2026_04_30.py` (35 k LOC, ad-hoc audit script), pas de `src/backtest/state_machine_replay`. Donc deux engines différents = deux codes de signal, deux codes de PnL. Aucune garantie de cohérence.

### 1.3 Refit aux fenêtres

Aucun composant du backtest ne nécessite actuellement de refit (rule-based 8-comp scoring + state-machine déterministe). MAIS :

- Si **HMM** est ré-activé (eval_04 vol findings recommande HAR pur), il faudra refitter par segment.
- Si **LightGBM volatility** est rebranché (VOL_MODE=lgbm), idem.
- Le **conformal wrapper** (`src/intelligence/conformal_wrapper.py`, mémoire 3 piliers) nécessite calibration par fenêtre.

→ Toute extension ML/stat requerra un walk-forward + refit. **Aucune scaffolding actuel pour ça dans `src/backtest/`.**

### Recommandation P0 (Sprint 3)

Créer `src/backtest/walk_forward.py` avec :

```python
class WalkForwardEngine:
    def __init__(self, train_bars, test_bars, embargo_bars, step_bars): ...
    def run(self, df, fit_fn, eval_fn) -> List[FoldResult]: ...
```

Réutilise `SignalReplay.run()` comme `eval_fn`. Documente refit policy pour chaque composant (HMM, LGBM, conformal).

---

## 2. Coûts transactionnels — **finding P0**

### 2.1 Modèles existants

`src/environment/execution_model.py:1-118` définit :

- **DynamicSlippageModel** (lignes 18-58) : ATR-proportional, `slip = base * max(1, (atr/median_atr)^scale)`.
- **DynamicSpreadModel** (lignes 61-118) : Session-dependent + news multiplier. Tableau `SESSION_SPREADS` Asian=8bp, London=3bp, etc.

### 2.2 Câblage replay

`SignalReplay.__init__` accepte `spread_model` et `slippage_model` (state_machine_replay.py:404-405) et `_build_trade` (lignes 749-836) sait les utiliser.

**MAIS** : aucun runner ne les fournit :

- `scripts/run_backtest.py` : aucune mention (grep 0 match).
- `scripts/run_baseline_sprint0.py` : aucune mention (grep 0 match).
- `scripts/audit_backtest.py` : aucune mention (grep 0 match).

Conséquence : `_build_trade:812` retourne `pnl_net = pnl_raw - 0 - 0 = pnl_raw`. **TOUS les `replay_*.json` et `baseline_report.json` actuels publient un PF brut.**

### 2.3 Commission

Hardcodée à `0.0` (state_machine_replay.py:834). Pas de paramètre. Pas même un modèle stub.

Pour XAU M15 chez les brokers retail, commission round-turn ~$0.7/lot. Sur ~362 trades/an et $30/lot exposure moyen, ça grignote ~5-8 R/an. Pas négligeable.

### 2.4 Impact estimé

Eval_18 §1.2 estime que brancher spread + slippage divise le PF par 1.10 à 1.20 (XAU M15). Pour la baseline publique actuelle (XAU M15 PF~1.08 sur 6 ans), c'est passer **PF 1.08 → 0.95-1.00** = **basculer du côté non-profitable**.

### Recommandation P0 (Sprint 3 ou avant)

1. Câbler dans `scripts/run_backtest.py` :

```python
from src.environment.execution_model import DynamicSlippageModel, DynamicSpreadModel
spread = DynamicSpreadModel(news_multiplier=3.0)
slippage = DynamicSlippageModel(base_slippage=0.0001, atr_scale_factor=1.0)
replay = SignalReplay(..., spread_model=spread, slippage_model=slippage)
```

2. Ajouter `CommissionModel` dans `execution_model.py` (round-turn fixe + per-lot).
3. Re-runner les baselines Sprint 0 avec costs ON → nouveau snapshot.
4. **Tag tous les `replay_*.json` actuels comme `cost_model=NONE`** et créer le set parallèle `cost_model=v1`.

---

## 3. Look-ahead MTF — **finding P0 latent** (pas exposé en prod actuellement)

### 3.1 État actuel du backtest

`state_machine_replay.py:445-611` n'utilise **aucun MTF** (M15 only). Les features sont calculées par `SmartMoneyEngine` sur la seule TF de l'input. Donc **pas de look-ahead MTF en prod backtest**.

### 3.2 Le piège quand MTF sera ré-activé

`src/environment/multi_timeframe_features.py:269` :

```python
mask = htf_df.index <= current_ts
htf_bar = htf_df.loc[mask].iloc[-1]
```

`pd.resample('1h')` labellise par défaut par **début de bar** (e.g. 09:00 couvre 09:00→09:59). Pour `current_ts=09:15`, la barre H1 09:00 (encore en formation, **incluant 09:15**) est retournée. C'est du look-ahead direct.

Le swing detector 4H (`multi_timeframe_features.py:559-575`) a été causally corrigé (`.shift(2)`), mais le path principal `_get_htf_features` (lignes 258-310) n'a **pas** la correction symétrique.

**Verdict** : si quelqu'un câble `MultiTimeframeFeatures.fit()` en amont de `SignalReplay.run()`, look-ahead immédiat sans warning. Eval_18 §1.3 déjà flaggé.

### Recommandation P0 (avant ré-activation MTF)

Patcher `multi_timeframe_features.py:269` :

```python
# Use STRICT < to exclude the still-forming HTF bar
mask = htf_df.index < current_ts
# OR: label='right' on resample, then mask <=
```

Ajouter test :

```python
def test_htf_features_no_lookahead():
    df_m15 = make_df_with_known_h1_close_at_09_00()
    mtf = MultiTimeframeFeatures().fit(df_m15)
    feats_at_09_15 = mtf.get_features(idx=at("09:15"))
    feats_at_09_45 = mtf.get_features(idx=at("09:45"))
    # The H1 bar 09:00→10:00 only closes at 10:00, so 09:15 and 09:45
    # must both report the H1 bar 08:00→09:00 (i.e. same value).
    assert feats_at_09_15 == feats_at_09_45
```

---

## 4. Refit aux fenêtres — non applicable (actuellement)

Vu §1.3 : la version actuelle du backtest est purement rule-based (pas de modèle entraîné). Donc pas de risque de leak « fit on all data, eval on subset ».

**MAIS** :

- Quand A1 v2 ou successeur sera tenté, refit per-window est obligatoire.
- Le **HMM** dans `volatility_forecaster.py` est entraîné une seule fois et appliqué sur tout l'historique → si on l'inclut dans le backtest, leak.
- Le **calibrateur conformal** (`src/intelligence/conformal_wrapper.py`) nécessite par construction un split calibration / test.

**P1 Sprint 3** : intégrer dans `WalkForwardEngine` (cf. §1) une API `fit_fn(train_df) -> fitted_model` appelée à chaque fold. Tester sur HMM puis LGBM.

---

## 5. CPCV — **finding P1 (existe, mais isolé)**

### 5.1 Code trouvé

`src/research/cpcv_harness.py:1-507`. Méthodologie complète López de Prado AFML ch. 7 :

- `split_into_n_folds` (L92-105) — contiguous folds
- `purged_train_indices` (L108-134) — purge + embargo
- `cpcv_path_indices` (L137-155) — générateur (n_folds=8, k=2 → C(8,2)=28 paths)
- `run_cpcv` (L382-507) — end-to-end avec model_factory et returns synthétiques

### 5.2 Code-review

- **Purge correct** : `forbidden = [a - embargo - label_horizon, b + embargo)` (L121-122). Cohérent avec AFML §7.4.
- **Embargo respecté** : test couvre déjà (`tests/test_cpcv_harness.py:36-50`).
- **28 paths** confirmés par `itertools.combinations(range(8), 2)`.
- **Génération returns** (L447-453) :

  ```python
  positions = np.where(preds > threshold, 1.0, np.where(preds < -threshold, -1.0, 0.0))
  returns = positions * y.iloc[test_idx].to_numpy()
  ```

  Standard mais **assume returns IID et symétriques** (long/short). Pas d'option `cost_model=` pour ré-injecter spread/slippage dans les CPCV returns. → P2.

### 5.3 Couplage avec `src/backtest/`

**ZERO**. `cpcv_harness.run_cpcv` prend `(model_factory, X, y)`, pas un `SignalReplay`. Donc impossible de fait passer la stratégie state-machine + 8-comp scoring + retest dans CPCV sans réécrire la couche d'évaluation.

Conséquence directe : **les `replay_*.json` et `baseline_report.json` ne peuvent pas être passés au CPCV harness en l'état**. La machinerie qui a recalé A1 ne peut pas recaler la stratégie commerciale principale.

### Recommandation P1 (Sprint 3)

Créer `src/backtest/cpcv_replay_adapter.py` :

```python
def replay_as_cpcv_strategy(df: pd.DataFrame, replay_config: dict) -> Callable:
    """Wraps SignalReplay into the (model_factory, X, y) contract of run_cpcv.
    
    Each `path` becomes a (train_idx, test_idx) subset of df. SignalReplay
    is recreated for the test slice (no fit needed for rule-based scoring).
    Returns trade R-multiples as the OOS metric series.
    """
```

Brancher dans `scripts/run_backtest.py` avec flag `--cpcv N=8 k=2`. Output : tableau 28 paths × {sharpe, PF, hit_rate, DSR, PBO} + gate verdict via `strategy_gates.evaluate_gates`.

---

## 6. DSR (Deflated Sharpe Ratio) — **conforme**

### 6.1 Formule

`cpcv_harness.deflated_sharpe_ratio` (L216-246) :

```python
denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr)
z = (sr - sr0) * np.sqrt(len(r) - 1) / denom
return float(stats.norm.cdf(z))
```

Conforme à Bailey & López de Prado 2014 équation (6). ✅

### 6.2 Multi-testing inflation

`expected_max_sharpe(n_trials)` (L200-213) implémente la correction Euler-Mascheroni de Bailey-LdP :

```
E[max SR] ≈ (1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N·e))
```

✅ Conforme.

### 6.3 Caveat

Le DSR renvoie une **probabilité** (CDF de la stat normalisée), pas un z-score direct. `strategy_gates.evaluate_gates:275-284` inverse via `stats.norm.ppf(dsr_prob)` pour matcher le gate `DSR_z >= 1.5`. Conceptuellement correct, mais propage erreurs de précision quand `dsr_prob ≈ 1.0` (z saturé à 8.0).

### Recommandation P2

Exposer **deux** APIs distinctes : `dsr_probability()` et `dsr_zscore()`, pour éviter la double conversion. Documenter explicitement laquelle est utilisée par les gates.

---

## 7. PBO (Probability of Backtest Overfitting) — **partiellement conforme**

### 7.1 Code

`cpcv_harness.probability_backtest_overfitting` (L254-292) implémente la **vraie** version Bailey-Borwein-LdP-Zhu 2014 (rank-logit method) — mais nécessite une grille de stratégies/hyperparams `(n_paths, n_strategies)`.

`_pbo_from_path_returns` (L295-307) est un **proxy single-strategy** : « fraction des paths où Sharpe < médiane des paths ». Documenté en commentaire L295-300 : « This is NOT the multi-strategy Bailey-Borwein-LdP-Zhu PBO ».

### 7.2 Usage actuel

`run_cpcv:491` appelle `_pbo_from_path_returns` (proxy). `strategy_gates.evaluate_gates:291-309` appelle aussi le proxy (fraction de paths/chunks à Sharpe ≤ 0).

**Problème** : le proxy `< median` donne ~0.5 sur du bruit (50% des paths sous la médiane par définition), donc la valeur n'a pas le même sens que le PBO Bailey 0.5 = noise. La traduction proxy-vers-vrai-PBO n'est pas calibrée.

### 7.3 Diagnostic empirique

L'A1 verdict reporte PBO = 0.5000 (`reports/a1_verdict_2026.md`). C'est précisément la valeur attendue si proxy = fraction sous médiane (mécaniquement 0.5). Donc PBO = 0.5 doit s'interpréter comme « pas d'information » plutôt que « overfit catastrophique ». À documenter.

### Recommandation P1

1. Implémenter le vrai PBO multi-strategy en générant une **grille de hyperparams** (e.g. enter ∈ {65, 70, 75, 80}, exit ∈ {40, 45, 50, 55}, cooldown ∈ {1, 2, 3}) = 48 stratégies → CPCV par path → matrice IS×OOS → vrai PBO Bailey.
2. Renommer le proxy : `pbo_proxy_below_median` pour éviter confusion.

---

## 8. IC bootstrap sur PF — **conforme + identifie une faiblesse**

### 8.1 Code

Deux implémentations parallèles :

1. **`run_baseline_sprint0.bootstrap_pf_ci`** (L133-190) : 10 000 resamples, percentile method, alpha=2.5%/97.5%. ✅ Bootstrap simple correctement implémenté.
2. **`strategy_gates.profit_factor_bootstrap_ci`** (L152-181) : 1 000 resamples (default), `seed=42`. ✅ Idem.

### 8.2 Faiblesse : méthode percentile, pas BCa

Aucune des deux n'implémente **BCa (Bias-Corrected and accelerated)** d'Efron — la méthode standard pour les distributions skewed (le PF est notoirement skewed à droite). Sur petit échantillon (n < 100), la CI percentile sous-estime la borne basse de ~5-15 %.

Pour XAU baseline Sprint 0, n_trades est probablement 30-200 selon le seuil. → BCa souhaitable.

### 8.3 Reproductibilité bootstrap

- `run_baseline_sprint0.bootstrap_pf_ci` : `seed=42` (L134). ✅
- `strategy_gates.profit_factor_bootstrap_ci` : `seed=DEFAULT_BOOTSTRAP_SEED=42` (L156). ✅

### Recommandation P2

1. Remplacer percentile par BCa (utiliser `scipy.stats.bootstrap` avec `method='BCa'` ; disponible scipy >= 1.7).
2. Ajouter bootstrap CI sur Sharpe et Sortino (actuellement seul PF est bootstrappé).

---

## 9. Reproductibilité bit-à-bit — **finding P1 ID-level**

### 9.1 Test empirique

Test conduit (15/05/2026) : `python scripts/run_backtest.py --csv data/XAU_15MIN_2019_2026.csv --symbol XAUUSD --timeframe M15 --last-n 5000 --enter 50 --exit 35` exécuté **2 fois** sur la même branche / même data.

Résultat :

| Métrique                | Run 1 | Run 2 | Égalité bit-à-bit ? |
| ----------------------- | ----- | ----- | --------------------|
| Total trades            | 46    | 46    | ✅                  |
| Entry price, exit price | …     | …     | ✅                  |
| PnL, R-multiple         | …     | …     | ✅                  |
| Score percentiles       | p50=42.4, p99=67.1 | identique | ✅ |
| `signal_id`             | `45a6d9ed-2da` | `334422ff-c81` | ❌ |

### 9.2 Cause racine

`src/intelligence/confluence_detector.py:343` :

```python
signal_id=str(uuid.uuid4())[:12],
```

`uuid.uuid4()` consomme `os.urandom(16)` → non-deterministe. Tous les autres champs sont reproductibles bit-à-bit.

### 9.3 Impact

- Métriques agrégées (PF, Sharpe, etc.) bit-identiques.
- Trade-level CSV diffère uniquement par la colonne `signal_id`.
- Checksums SHA256 différents entre runs → casse `reports/baseline/checksums.txt` (qui hashe les JSON contenant les signal_id).
- Audit ledger (`scripts/audit_ledger_snapshot.py`) ne peut pas vérifier qu'un re-run produit la même chaîne — chaque snapshot a des hashes différents au niveau LedgerEntry.

### Recommandation P1 (Sprint 3 ou avant)

1. Remplacer `uuid.uuid4()` par hash déterministe :

```python
signal_id=hashlib.blake2b(
    f"{bar_timestamp}|{symbol}|{signal_type.value}|{round(score, 2)}".encode(),
    digest_size=6,
).hexdigest()
```

2. Ajouter test : `test_replay_bit_for_bit_reproducibility`.

### 9.4 Threads / parallélisme

`SignalReplay.run` est mono-thread. Aucune source de race condition dans le replay lui-même. ✅

---

## 10. Snapshot d'état au signal — **finding P1 (Sprint 6)**

### 10.1 État actuel

`TradeRecord` (`state_machine_replay.py:163-213`) capture :

- IDs (signal_id, direction)
- Bars (entry/exit timestamp)
- Prix (entry, exit, SL, TP)
- PnL (raw, net, R)
- Cost decomposition (spread_cost, slippage_cost, commission)
- exit_reason, bars_held, confluence_score

**Absents** pour audit reconstructible :

- ❌ Décomposition du score 8-composants (BOS, FVG, OB, RSI, MACD, CHOCH, regime, news contributions individuelles)
- ❌ Snapshot features d'entrée (ATR, vol_regime, regime tag, news status à la barre d'entrée)
- ❌ Version du code (git SHA)
- ❌ Version des données (data SHA256)
- ❌ Version config (hyperparams state machine + detector)
- ❌ Random seed effectif
- ❌ Snapshot du detector state interne (cooldown counter, last_signal_age, etc.)

### 10.2 Importance pour audit institutionnel

Si un prospect demande « explique-moi pourquoi ce trade du 12/03/2026 a été pris », on doit pouvoir **rejouer exactement les conditions** : features de la barre, état du detector, scoring composante par composante. Actuellement c'est **impossible sans re-runner tout le backtest** (et même là, les signal_ids ne matcheront pas — cf. §9).

### 10.3 Couplage existant : audit ledger

`scripts/audit_ledger_snapshot.py:1-193` + `src/audit/hash_chain_ledger.py` (mentionné) implémentent une chaîne hash SQLite WAL pour les insights livrés en **production** (Sprint DATA-2B.9). Mais le backtest ne tape PAS dedans — c'est purement live-delivery.

### Recommandation P1 (Sprint 6 - production hardening)

Étendre `TradeRecord` ou créer `TradeRecordExtended` :

```python
@dataclass(frozen=True)
class TradeRecordExtended(TradeRecord):
    score_breakdown: dict[str, float]   # {'bos': 15.0, 'fvg': 10.0, ...}
    entry_features: dict[str, float]    # snapshot SMC + regime + news
    detector_state: dict[str, Any]      # cooldown, signal_age, etc.
    config_hash: str                    # SHA256 of (state_machine_cfg + detector_cfg)
    code_commit: str                    # git rev-parse HEAD
    data_hash: str                      # SHA256 of input CSV file
```

Plus une `BacktestSnapshot` (replay-level) :

```python
@dataclass
class BacktestSnapshot:
    run_id: str                         # blake2b(config+data+code)
    started_utc: str
    finished_utc: str
    code_commit: str
    data_hashes: dict[str, str]
    config_snapshot: dict[str, Any]
    metrics: PerformanceMetrics
    trades_csv_sha256: str
    lib_versions: dict[str, str]        # pandas, numpy, lightgbm, etc.
```

→ Persistence dans `reports/backtest_snapshots/{run_id}.json` (Sprint 6).

---

## 11. Calcul des métriques — table conformité

### 11.1 Métriques calculées + validation

| Métrique             | Lieu                                                  | Formule actuelle                                        | Statut          | Bug détecté |
| -------------------- | ----------------------------------------------------- | ------------------------------------------------------- | --------------- | ----------- |
| Win rate             | `metrics.py:207` + `state_machine_replay.py:694`     | wins / total                                            | ✅              | —           |
| Loss rate            | `metrics.py:208`                                      | losses / total                                          | ✅              | —           |
| Average R            | `metrics.py:211`                                      | `np.mean(r_series)`                                     | ✅              | —           |
| Median R             | `metrics.py:212`                                      | `np.median(r_series)`                                   | ✅              | —           |
| Stdev R              | `metrics.py:213`                                      | `statistics.pstdev` (ddof=0, **population**)            | ⚠️              | Voir #B1     |
| Total R              | `metrics.py:214`                                      | `sum(r_series)`                                         | ✅              | —           |
| Profit factor        | `metrics.py:222-226` + `state_machine_replay.py:709`  | `sum(>0) / abs(sum(<0))`, inf if no losses              | ✅              | —           |
| Payoff ratio         | `metrics.py:230-233`                                  | `avg_win / abs(avg_loss)`                               | ✅              | —           |
| Sharpe per trade     | `metrics.py:244` + `state_machine_replay.py:716-718`  | `mean / stdev` (ddof inconsistent !)                    | ⚠️              | Voir #B1     |
| Sortino per trade    | `metrics.py:246-252`                                  | `mean / sqrt(sum(min(0, r)²) / N)`                      | ✅              | —           |
| Calmar               | `metrics.py:253-256`                                  | `total_r / max_drawdown_r`                              | ❌              | Voir #B2     |
| Max drawdown R       | `metrics.py:236, 302-309`                             | `(cumsum - cummax).max()`                               | ✅              | —           |
| Max consec losses    | `metrics.py:237, 312-321`                             | longest run of `r <= 0`                                 | ⚠️              | Voir #B3     |
| Sharpe annualised    | `metrics.py:267`                                      | `sharpe_per_trade * sqrt(trades_per_year)`              | ⚠️              | Voir #B4     |
| Sortino annualised   | `metrics.py:269`                                      | idem sortino                                            | ⚠️              | Voir #B4     |
| Expectancy R         | `metrics.py:215`                                      | `avg_r` (duplicate field)                               | ✅              | —           |
| Trades per year      | `metrics.py:264`                                      | `total_trades / years`                                  | ✅              | —           |
| Tier breakdown       | `metrics.py:324-383`                                  | groupby score >= {40, 60, 80}                           | ✅              | Doc obsolete (eval_02 : zero predictive power) |
| PF bootstrap CI      | `run_baseline_sprint0.py:133-190`                     | percentile method, 10 000 resamples                     | ⚠️              | Voir §8     |
| DSR                  | `cpcv_harness.py:216-246`                             | Bailey-LdP 2014 eq. (6)                                 | ✅              | —           |
| PBO (proxy)          | `cpcv_harness.py:295-307`                             | fraction paths < median Sharpe                          | ⚠️              | Voir §7     |
| PBO (full)           | `cpcv_harness.py:254-292`                             | Bailey-Borwein-LdP-Zhu 2014 rank-logit                  | ✅              | Pas utilisé en prod |
| Diebold-Mariano      | `cpcv_harness.py:345-374`                             | HAC-adjusted DM test                                    | ✅              | —           |
| Holm-Bonferroni      | `cpcv_harness.py:315-342`                             | sequentially-rejective                                  | ✅              | —           |

### 11.2 Bugs identifiés (B1-B4)

#### **B1** — Sharpe stdev inconsistent (P1)

- `state_machine_replay.py:716-718` : `statistics.stdev(r_series)` = `ddof=1` (sample)
- `metrics.py:213, 244` : `statistics.pstdev(r_series)` = `ddof=0` (population)

Sur 30 trades : `pstdev ≈ stdev × sqrt(29/30) = stdev × 0.983`. Donc Sharpe `metrics.py` surestime de 1.7 % vs Sharpe `state_machine_replay.py`. Sur 1000 trades : 0.05 %. Pas catastrophique, mais **deux nombres différents** pour la même métrique dans le même `ReplayResults` est inacceptable.

**Fix** : choisir une convention (population pour l'estimateur point ou sample pour l'IID assumption). Recommandation : `pstdev` (cohérent avec textbook financial Sharpe).

#### **B2** — Calmar formule incorrecte (P1)

`metrics.py:254` : `m.calmar = m.total_r / m.max_drawdown_r`.

Calmar ratio est **CAGR / max_drawdown** (Young 1991), où CAGR = compound annual growth rate. Ici on a total_r (R-multiples cumulés) / max_drawdown_r (max DD en R). Les unités sont cohérentes, mais ce n'est pas annualisé. Sur 7 ans XAU avec total_r=39R et max_dd=10R, formule actuelle donne 3.9 ; vrai Calmar (annualisé) donnerait 39/7/10 = 0.56.

**Fix** : `calmar = (total_r / years) / max_drawdown_r`, en supposant `years` calculable depuis `bars_processed / bars_per_year`.

#### **B3** — Max consec losses pénalise les R=0 (P2)

`metrics.py:237` :

```python
m.max_consecutive_losses = _max_consecutive_run(r_series, lambda r: r <= 0)
```

Les trades à R=0 (breakeven exits, e.g. SL/TP à entry) sont comptés comme losses. Définition non standard. Devrait être `r < 0`.

**Fix** : `lambda r: r < 0`. Documenter le breakeven counting comme métrique séparée.

#### **B4** — Annualisation Sharpe par sqrt(trades_per_year) (P2)

`metrics.py:267` :

```python
m.sharpe_annualised = m.sharpe_per_trade * math.sqrt(trades_per_year)
```

Formule textbook sous hypothèse **IID returns**. Pour des trades :

1. Returns souvent autocorrélés (drawdown clusters, momentum runs).
2. Trade durations hétérogènes (bars_held varie 1-50).

Annualisation correcte = `sharpe_per_trade * sqrt(N)` où N = # observations ANNUALISÉES, donc bien `trades_per_year`. MAIS : assume returns IID, sinon overstate de 15-30 %.

**Fix** : ajouter test Ljung-Box d'autocorrélation sur `r_series`. Si rejet H0 à 5 %, déflate Sharpe annualisé par √(1 + 2·ρ) (Lo 2002 formule) ou logger un warning.

---

## 12. Tests existants — couverture

| Fichier test                          | LOC  | Couvre                                          | Couverture |
| ------------------------------------- | ---- | ----------------------------------------------- | ---------- |
| `tests/test_state_machine_replay.py`  | 284  | _build_trade, _pair_trades, _compute_metrics, _max_drawdown_r, _max_consecutive_losses, _count_bars_between, run() | 18 tests |
| `tests/test_backtest_metrics.py`      | 228  | compute_performance, tier_breakdown, annualisation | ~12 tests |
| `tests/test_backtest_report.py`       | 121  | render_text, render_json                        | ~6 tests  |
| `tests/test_news_replay.py`           | 109  | from_csv, blackout window, BLOCK assessment     | ~5 tests  |
| `tests/test_cpcv_harness.py`          | 275  | purge, embargo, DSR formule, PBO, Holm, split   | 10+ tests |

**Manquants** :

- ❌ Aucun test pour `DynamicSpreadModel` / `DynamicSlippageModel` (orphans).
- ❌ Aucun test bout-à-bout `replay → metrics → report` avec **costs ON**.
- ❌ Aucun test reproductibilité bit-à-bit.
- ❌ Aucun test look-ahead MTF (§3.2).
- ❌ Aucun test wiring `SignalReplay` → `strategy_gates.evaluate_gates` (cf. §5.3).

---

## 13. Synthèse findings P0/P1/P2

### P0 (bloquants commercialisation)

| #  | Finding                                                                                      | Fichier:ligne                                              | Recommandation |
| -- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | -------------- |
| P0-1 | Aucun walk-forward dans l'engine production                                                | `state_machine_replay.py:445-611`                          | Sprint 3 : créer `src/backtest/walk_forward.py` |
| P0-2 | Coûts transactionnels = 0 sur tous les runs publiés                                          | `state_machine_replay.py:834` (commission), `run_backtest.py` (no spread/slip wire) | Sprint 3 : câbler `DynamicSpreadModel` + `DynamicSlippageModel` partout + ajouter `CommissionModel` |
| P0-3 | Look-ahead MTF latent — pas exposé en prod aujourd'hui, mais bloque toute ré-activation MTF  | `multi_timeframe_features.py:269`                          | Sprint 3 : patch `<=` → `<` + test no-lookahead |
| P0-4 | CPCV + DSR + PBO **non couplés** à `src/backtest/` — impossible de gater la stratégie commerciale | `src/research/cpcv_harness.py` standalone                | Sprint 3 : créer `cpcv_replay_adapter.py` |

### P1 (sérieux mais workaround-able)

| #  | Finding                                                                                  | Fichier:ligne                                  | Recommandation |
| -- | ---------------------------------------------------------------------------------------- | ---------------------------------------------- | -------------- |
| P1-1 | `uuid.uuid4()` casse reproductibilité bit-à-bit (signal_id différent à chaque run)    | `confluence_detector.py:343`                   | Sprint 3 : hash déterministe blake2b |
| P1-2 | Sharpe stdev inconsistent (pstdev vs stdev) entre `metrics.py` et `state_machine_replay.py` | `metrics.py:213` vs `state_machine_replay.py:716-718` | Sprint 3 : consolidation pstdev |
| P1-3 | Calmar formule sans annualisation                                                       | `metrics.py:254`                               | Sprint 3 : `(total_r / years) / max_dd` |
| P1-4 | PBO actuellement proxy single-strategy — pas le vrai Bailey-Borwein-LdP-Zhu             | `cpcv_harness.py:295-307` (proxy par défaut)   | Sprint 3 : implémenter grille hyperparams pour vrai PBO |
| P1-5 | Aucun snapshot reproductible (features, score breakdown, code SHA, data SHA, lib versions) | `state_machine_replay.py:163-213` (TradeRecord) | Sprint 6 : `TradeRecordExtended` + `BacktestSnapshot` |
| P1-6 | `news_provider` pas wiré dans `run_backtest.py` ni `run_baseline_sprint0.py`            | `run_backtest.py`, `run_baseline_sprint0.py`   | Sprint 3 : câbler `BacktestNewsProvider.from_csv` |
| P1-7 | Aucun test reproductibilité bit-à-bit, ni test look-ahead MTF, ni test wiring `strategy_gates` | `tests/`                                       | Sprint 3 : 3 tests à ajouter |

### P2 (cosmétique / dette technique)

| #  | Finding                                                              | Fichier:ligne                          | Recommandation |
| -- | -------------------------------------------------------------------- | -------------------------------------- | -------------- |
| P2-1 | Bootstrap PF en percentile method (pas BCa)                       | `run_baseline_sprint0.py:178-181`      | Sprint 5 : BCa via `scipy.stats.bootstrap(method='BCa')` |
| P2-2 | Max consec losses compte breakeven (R=0) comme losses             | `metrics.py:237`                       | Sprint 5 : `lambda r: r < 0` |
| P2-3 | Annualisation Sharpe sans correction autocorrélation Lo 2002      | `metrics.py:267`                       | Sprint 5 : ajouter test Ljung-Box + correction |
| P2-4 | `DSR` retourne probabilité, gates utilisent z-score — double conversion fragile | `cpcv_harness.py:246` + `strategy_gates.py:281-284` | Sprint 5 : 2 API distinctes |
| P2-5 | Tier breakdown publié alors que `confluence_score` non-prédictif (eval_02) | `metrics.py:282-294`               | Sprint 5 : flag obsolète ou retirer (Sprint 3 fait la calibration) |
| P2-6 | CPCV `run_cpcv` n'accepte pas cost_model (returns sans frais)     | `cpcv_harness.py:447-453`              | Sprint 5 : ajouter cost_fn=Callable |
| P2-7 | Pas de bootstrap CI sur Sharpe / Sortino                          | `metrics.py`, `run_baseline_sprint0.py`| Sprint 5 |

---

## 14. Roadmap d'intégration

### Sprint 3 (Statistical Edge Discovery)

**Goal** : Industrialiser CPCV + DSR + PBO sur la stratégie production (state-machine + 8-comp scoring), pas seulement sur les expérimentations.

1. **Walk-forward** : `src/backtest/walk_forward.py` — rolling train+test+embargo, refit hooks pour modèles fittés (HMM, LGBM, conformal).
2. **CPCV adapter** : `src/backtest/cpcv_replay_adapter.py` — wraps `SignalReplay.run()` dans le contrat `(model_factory, X, y)` de `cpcv_harness.run_cpcv`.
3. **Cost models câblés** : modifier `run_backtest.py` et `run_baseline_sprint0.py` pour instancier `DynamicSpreadModel` + `DynamicSlippageModel` par défaut. Ajouter flag `--no-costs` (réservé aux tests). Ajouter `CommissionModel` (round-turn).
4. **News provider câblé** : idem pour `BacktestNewsProvider.from_csv(data/economic_calendar_HIGH_IMPACT_2019_2025.csv)`.
5. **Reproductibilité ID** : remplacer `uuid.uuid4()` par hash blake2b déterministe dans `confluence_detector.py:343`.
6. **Fix B1, B2, B3** : pstdev partout, Calmar annualisé, max_consec_losses sur r<0.
7. **Tests** : reproductibilité bit-à-bit + no-lookahead MTF + wiring strategy_gates.
8. **Vrai PBO multi-strategy** : grille hyperparams (enter, exit, cooldown, confirm) × CPCV → vrai PBO Bailey.

### Sprint 5 (Stress tests)

1. **BCa bootstrap** sur PF, Sharpe, Sortino, expectancy.
2. **Ljung-Box autocorrélation** sur r_series + correction Lo 2002.
3. **Cost sensitivity sweep** : run baseline à 0.5× / 1× / 2× / 3× costs ; report degradation.
4. **Regime-stratified stress** : 2019 (calm) vs 2020 (COVID) vs 2022 (Fed tightening) vs 2023+ (range). PF per regime.
5. **Drawdown distribution** : empirique vs Brownian benchmark.
6. **Outlier impact** : Sharpe avec/sans top-5 % et bottom-5 % trades.
7. **Tier breakdown revalidation** post-Sprint-3 calibration (eval_02 PMC).

### Sprint 6 (Production hardening — snapshot store)

1. **`TradeRecordExtended`** + score_breakdown + entry_features + detector_state.
2. **`BacktestSnapshot`** + code SHA + data SHA + config hash + lib versions.
3. **Persistence** : `reports/backtest_snapshots/{run_id}.json` + index `snapshots_index.csv` (run_id, date, symbol, PF, gates_passed).
4. **CI hook** : `pytest tests/test_smoke_e2e.py::test_baseline_snapshot_runs_bitwise_identical` qui re-run et compare SHA256.
5. **Couplage audit ledger** : étendre `HashChainLedger` pour ingérer aussi des `BacktestSnapshot` (pas seulement des delivered insights).
6. **API** : `GET /api/v1/backtest/snapshots/{run_id}` retourne le snapshot complet.

---

## 15. Comparaison avec la note eval_18

| Critère                       | Eval_18 (2026-04-26) | Audit Sprint 0 (2026-05-15) | Évolution |
| ----------------------------- | -------------------- | --------------------------- | --------- |
| **Note globale**              | 2/10                 | 3.5/10                      | +1.5      |
| **Walk-forward**              | ❌ aucun             | ❌ aucun (mais script ad-hoc identifié) | = |
| **Coûts transactionnels**     | ❌ $0                | ❌ modèles existent mais déconnectés | + 0.5 (modèles présents) |
| **Look-ahead MTF**            | ❌ flagged           | ❌ confirmé, swing 4H corrigé partiellement | + 0.5 |
| **CPCV + DSR + PBO**          | ❌ inexistant        | ✅ implémenté (mai 2026) mais standalone | + 2.5 |
| **IC bootstrap**              | ❌ aucun             | ✅ percentile method (pas BCa) | + 1.0 |
| **Strategy gates**            | ❌ inexistant        | ✅ DSR+PBO+PF_lo+DM en place | + 2.0 |
| **Reproductibilité**          | non évalué           | ⚠️ trades bit-identiques, IDs aléatoires | + 0.5 |
| **Couplage engine ↔ gates**   | non évalué           | ❌ zéro                     | -1.0 vs attente |

Net : **+1.5 points**, principalement grâce à `src/research/cpcv_harness.py` (mai 2026, Sprint 0 / Pilier 1) et `src/research/strategy_gates.py` (Sprint 0 / Pilier 1). Le verdict eval_18 « NON commercialisable » **tient toujours** tant que P0-1 à P0-4 ne sont pas adressés.

---

## 16. Ce que cet audit ne couvre PAS

Pour transparence, voici les zones du périmètre 3.8 que je n'ai pas couvertes (raisons et follow-up suggéré) :

1. **Performance / latence** du backtest. Sortie de scope (Section 3.9 ou 3.12 Performance). Le profiling du replay 7 ans n'a pas été exécuté ; un audit performance séparé devrait mesurer P99 latence par bar + memory footprint à 1M bars.

2. **Stress tests opérationnels**. Pas de simulation de panne (CSV corrompu mid-replay, ATR division par zéro, etc.). Couvert en Sprint 5.

3. **Comparaison empirique vs Quantopian/Backtrader/vectorbt**. L'audit n'a pas vérifié que `SignalReplay` reproduit les chiffres d'une lib établie sur un même dataset. Recommandé Sprint 5.

4. **Equity curve plotting**. `report.py` ne génère pas de graphique d'equity. Pas critique mais attendu par prospects institutionnels.

5. **News blackout cross-instrument**. `BacktestNewsProvider.DEFAULT_AFFECTING_CURRENCIES` (lignes 36-43) est statique. Pas d'audit sur la qualité du calendrier économique CSV (couvert par Section 3.1).

6. **Live-replay parity**. L'audit ne teste pas si `SignalReplay.run()` produit les mêmes signaux que `SentinelScanner` en live mode sur les mêmes bars. P0 fonctionnel séparé.

7. **Coverage data quality** au moment du backtest. Eval_18 / Section 3.1 ont couvert que `XAU_15MIN_2019_2026.csv` = 98.72 % coverage. Si bars manquantes affectent les warmup_bars=100 ou les ATR rolling, le test n'a pas été reproduit ici.

8. **Spread / slippage calibration empirique**. `DynamicSpreadModel.SESSION_SPREADS` hardcode Asian=8bp, London=3bp etc. Pas d'audit sur la source de ces nombres ni leur validité historique 2019-2026 chez les brokers retail (IC Markets, Pepperstone). À calibrer en Sprint 5.

9. **Liquidité / market impact**. Aucun modèle d'impact prix (rare en backtest M15 mais critique en M1 ou crypto). Hors scope MVP.

10. **PF distribution sur sous-périodes < 1 mois**. La granularité actuelle de `audit_backtest.py` est per-year. Pas de PF mensuel/hebdo. À ajouter Sprint 5 pour test de robustesse.

11. **`hash_chain_ledger.py`**. Le code de `src/audit/hash_chain_ledger.py` (référencé par `audit_ledger_snapshot.py`) n'a pas été lu — seulement le CLI. Audit séparé recommandé.

12. **`backtest_combo_E.py`** + autres scripts ad-hoc à la racine. ~10 scripts (`forensics_*.py`, `falsification_*.py`, `comparatif_*.py`) génèrent des `replay_*.json` non listés dans eval_18. Audit forensique séparé pour identifier lesquels ont un walk-forward correct.

---

## 17. Tableau récap pour décision Sprint 3

| P0/P1/P2 | Effort estimé | Bloqueur commercialisation ? | Sprint cible |
| -------- | ------------- | ---------------------------- | ------------ |
| P0-1 walk-forward    | 12-20h       | OUI                          | Sprint 3 obligatoire |
| P0-2 cost models     | 6-8h         | OUI                          | Sprint 3 obligatoire |
| P0-3 MTF lookahead   | 4-6h         | OUI (avant ré-activation MTF) | Sprint 3 conditionnel |
| P0-4 CPCV adapter    | 16-24h       | OUI (sans gates, pas de PF audit-grade) | Sprint 3 obligatoire |
| P1-1 ID determinism  | 1h           | NON mais audit trail cassé    | Sprint 3 |
| P1-2 stdev unify     | 1h           | NON mais cohérence brisée    | Sprint 3 |
| P1-3 Calmar formula  | 2h           | NON                          | Sprint 3 |
| P1-4 PBO grille      | 6-10h        | NON                          | Sprint 3 |
| P1-5 snapshot store  | 20-30h       | NON (mais P0 pour audit institutionnel) | Sprint 6 |
| P1-6 news wiring     | 1h           | NON                          | Sprint 3 |
| P1-7 tests           | 8-12h        | NON                          | Sprint 3 |
| P2 (bouquet)         | 16-24h       | NON                          | Sprint 5 |

**Total Sprint 3 obligatoire** : 49-71h (~8-10 jours homme).
**Total Sprint 3 + 5 + 6** : 96-140h (~15-20 jours homme).

---

**Fin section 3.8 audit.**
