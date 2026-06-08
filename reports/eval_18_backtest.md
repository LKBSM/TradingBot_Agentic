# Eval 18 — Backtest & Replay Harness (crédibilité marketing)

> **Synthèse** : Eval 18, prompt « les chiffres affichés = la conversion ».
> Mission : auditer la chaîne `state_machine_replay → audit_backtest → replay_*.json`
> avant toute publication marketing. PhD-level = walk-forward + IC bootstrap +
> SPA + correction multiple-testing.
>
> **Auteur** : Eval 18 K1-K11 (Synthesis Lead).
> **Date** : 2026-04-26.
> **Périmètre** : `src/backtest/`, `scripts/audit_backtest.py`,
> `scripts/run_backtest.py`, 19 fichiers `replay_*.json/csv` à la racine.

---

## Executive Summary

**Verdict global : ❌ NON commercialisable. Note crédibilité backtest = 2/10.**

Cinq problèmes structurels rendent **tout chiffre actuellement publiable
inadmissible** sur une page marketing :

1. **Aucun walk-forward n'a été exécuté.** Les 19 `replay_*.json` sont
   tous des in-sample single-fold runs. Le sweep `audit_backtest.py`
   sélectionne le meilleur PF parmi 7 configs sans correction
   multiple-testing → biais Hansen SPA non mesuré, inflation attendue
   +15-25 %.
2. **Coûts de transaction = $0.** `_build_trade()` calcule `pnl =
   exit_price - entry_price` brut. `src/environment/execution_model.py`
   (DynamicSlippageModel + DynamicSpreadModel) existe mais n'est PAS
   branché dans le replay. Sur XAU M15 ~362 trades/an avec spread
   moyen 5 bps + slippage ATR-prop, l'impact estimé sur PF est
   **−0.10 à −0.20**.
3. **Look-ahead non-mitigé sur le sous-système MTF.**
   `multi_timeframe_features.py:554-566` itère `iloc[i+1]`, `iloc[i+2]`
   pour détecter les swings 4H — fuite causale flagrante. Heureusement
   pas utilisé par le replay actuel, mais bloque toute extension
   multi-timeframe sans patch.
4. **`confluence_score` empiriquement non-prédictif** (Pearson −0.023,
   Brier worse-than-baseline) — déjà documenté dans
   `confluence_calibration.md`. Tier system (PREMIUM/STANDARD/WEAK)
   actuellement invalide.
5. **Aucune simulation Monte Carlo / bootstrap.** Les chiffres de PF
   actuels (0.96, 1.086, 0.39, 1.08…) circulent sans IC 95 %. Sans IC,
   on ne peut pas affirmer qu'ils sont statistiquement différents de
   1.0.

**Plan d'action P1** : exécuter `reports/eval_18_walkforward_skeleton.py`
(livré), brancher `execution_model.py` dans le replay, écrire
`scripts/montecarlo_bootstrap.py`. Tant que ces 3 livrables ne sont pas
faits, la communication marketing doit utiliser le langage qualitatif
**§4 de `BACKTEST_LEGAL_GUARDRAILS.md`** uniquement.

---

## K1 — Inventaire des artefacts (publiable / interne)

19 fichiers `replay_*.json/csv` à la racine du repo, tous générés entre
2026-04-23 et 2026-04-24 pendant les itérations Sprint-1/2/3. Aucun
n'est issu d'un walk-forward.

### Matrice des replays

| Fichier | Date | Window | Config | Trades | PF | Statut |
|---|---|---|---|---:|---:|---|
| `replay_report.json` | 23/04 00:49 | 20k bars | enter=40 / exit=25 | — | — | INTERNE — corrompu (0 trades visible) |
| `replay_report_20k.json` | 23/04 00:44 | 2024-09→2025-12 (19 875 bars) | enter=50 / exit=30 | 63 | 1.034 | INTERNE — single-window, no costs |
| `replay_report_20k_nv.json` | 23/04 00:45 | 20k bars no-vol | — | — | — | INTERNE — diagnostic |
| `replay_trades_20k.csv` | 23/04 00:44 | jumeau de report_20k | 8 947 B | — | INTERNE |
| `replay_diagnostic.json` | 23/04 08:54 | 20k bars | enter=40 / exit=25 / cooldown=3 | 7 | **0.385** | INTERNE — petit échantillon |
| `replay_diagnostic_trades.csv` | 23/04 08:54 | jumeau diagnostic | 1 143 B | — | INTERNE |
| `replay_noregime.json` | 23/04 08:56 | 20k bars regime off | — | — | INTERNE — diagnostic |
| `replay_post_asymmetry_fix.json` | 23/04 14:44 | 19 875 bars | enter=40 / exit=25 / retest | 13 | **0.631** | INTERNE — post-fix retest, n=13 non-significatif |
| `replay_post_detector_fix.json` | 23/04 14:50 | 19 875 bars | enter=40 / exit=25 | 65 | **0.944** | INTERNE — post-fix detector |
| `replay_no_vol_exit.json` | 23/04 14:52 | 19 875 bars vol exit off | — | — | INTERNE |
| `replay_exit15.json` | 23/04 14:54 | 19 875 bars exit=15 | — | — | INTERNE |
| `replay_trades.csv` | 23/04 14:54 | jumeau exit15 | 8 887 B | — | INTERNE |
| `replay_noretest.json` | 24/04 01:00 | 20k bars (sub-window) | retest off | — | — | INTERNE — A/B avec retest |
| `replay_noretest_trades.csv` | 24/04 01:00 | jumeau | 304 B | — | INTERNE |
| `replay_retest.json` | 24/04 01:00 | jumeau noretest avec retest on | — | — | INTERNE |
| `replay_retest_trades.csv` | 24/04 01:00 | jumeau | 300 B | — | INTERNE |
| `replay_noretest_2025.json` | 24/04 01:01 | 2025 noretest | — | — | INTERNE |
| `replay_noretest_2025_trades.csv` | 24/04 01:01 | jumeau | 66.7 KB | — | INTERNE |
| `replay_retest_2025.json` | 24/04 01:01 | 2025 retest | — | — | INTERNE |
| `replay_retest_2025_trades.csv` | 24/04 01:01 | jumeau | 37.8 KB | — | INTERNE |
| `replay_retest_2025_v2.json` | 24/04 01:03 | 19 875 bars (2024-09→2025-12) | enter=40 / exit=25 / retest_v2 | 409 | **1.080** | INTERNE — 1.4 an, no costs, no walk-forward |
| `replay_retest_2025_v2_trades.csv` | 24/04 01:03 | jumeau | 58.5 KB | — | INTERNE |

### Verdict K1

**0 fichier publiable.** Tous sont des single-fold in-sample sur 1.4 an
ou moins. Le seul qui couvre la fenêtre 6 ans complète est
`reports/baseline_full.json` (PF 1.086, Sharpe 0.59) — déjà flaggé
**INTERNE UNIQUEMENT** dans `BACKTEST_LEGAL_GUARDRAILS.md` (§1).

**Action recommandée** : déplacer tous les `replay_*.json/csv` à la
racine vers `reports/replays_archive/2026-04-23-iteration/` pour
clarifier qu'aucun n'est utilisé en production.

---

## K2 — Look-Ahead Auditor (verdict par site)

Grep exhaustif sur `src/`. **8 sites suspects**, **2 confirmés
non-causaux**, **6 mitigés** mais à documenter.

### B1 — `rolling(center=True)` sur fractals (MITIGÉ)

`src/environment/strategy_features.py:617-618`
```python
rolling_max = self.df['high'].rolling(window=window_size, center=True).max()
rolling_min = self.df['low'].rolling(window=window_size, center=True).min()
```

**Verdict** : `center=True` lit jusqu'à `i+N` valeurs futures. **Mais**
ligne 637-638 applique `shift(N)` qui restaure la causalité (un fractal
détecté à `t` n'est connu qu'à `t+N`). Lignes 641-644 forcent les `2N`
premiers et derniers bars à NaN.

**Test différentiel suggéré** : remplacer le `rolling(center=True).max()`
par un `rolling(window).max().shift(-N)` puis `shift(N)` pour
prouver l'équivalence causale. À ajouter à
`tests/test_smart_money_engine.py`. **Statut** : OK fonctionnellement,
fragile architecturalement (un futur ré-écrivant ces lignes peut
casser la causalité).

### B2 — `iloc[i+1]`, `iloc[i+2]` swing detector (CONFIRMÉ NON-CAUSAL)

`src/environment/multi_timeframe_features.py:554-566`
```python
for i in range(2, len(df) - 2):
    if (df['High'].iloc[i] > df['High'].iloc[i-1] and
        df['High'].iloc[i] > df['High'].iloc[i-2] and
        df['High'].iloc[i] > df['High'].iloc[i+1] and  # FUTURE
        df['High'].iloc[i] > df['High'].iloc[i+2]):    # FUTURE
        swing_highs.append(...)
```

**Verdict** : Look-ahead flagrant. À l'instant `t = i`, on consulte
`H[i+1]` et `H[i+2]`. La logique cohérente serait d'attendre `i+2` pour
confirmer un swing à `i`, c.-à-d. itérer sur `i` mais émettre le swing
sous l'index `i+2` (équivalent shift(+2)).

**Impact dans le replay actuel** : NUL — `multi_timeframe_features.py`
n'est pas appelé par `state_machine_replay.py`. Mais c'est un blocker
absolu si Eval 19 ou 20 décide d'introduire un filtre 4H. **À PATCHER
avant tout backtest MTF.**

**Patch proposé** :
```python
for i in range(2, len(df) - 2):
    if (...) :
        # Emit swing AT i+2 (not i) so detection is causal
        swing_highs.append({'price': df['High'].iloc[i],
                            'idx': i + 2,
                            'timestamp': df.index[i + 2]})
```

### B3 — `bfill()` sur indicateurs lents (FAIBLE LEAK)

`src/environment/environment.py:802` et `multi_timeframe_features.py:184`
```python
df_processed[col] = df_processed[col].bfill()
df.bfill(inplace=True)
```

**Verdict** : `bfill` propage des valeurs futures vers les NaN passés.
Sur un timeframe 15min avec rolling(200), les 0-200 premiers bars
récupèrent leur valeur depuis le futur. **Impact** : confiné aux
warmup bars. Le replay impose `WARMUP=100` (audit_backtest.py:66) qui
évite que ces bars génèrent des signaux **mais** les indicateurs sont
calculés en amont sur la trame complète, donc une `bfill` au bar 0-200
peut influencer les valeurs à `WARMUP+1` via les fenêtres glissantes
qui les agrègent.

**Patch proposé** : remplacer `bfill()` par un masque NaN explicite ou
augmenter `WARMUP` à 500 pour absorber.

### B4 — `expanding().quantile()` (CAUSAL OK)

`src/backtest/state_machine_replay.py:145-146`
```python
q_low = atr.expanding(min_periods=100).quantile(low_quantile)
q_high = atr.expanding(min_periods=100).quantile(high_quantile)
```

**Verdict** : `expanding` ne lit que le passé (de 0 à `t`). **OK
strictement**. Bien commenté ligne 138 *« no look-ahead bias »*.

### B5 — `shift(-cfg.pred_horizon)` (TARGET, pas FEATURE)

`src/intelligence/volatility_forecaster.py:669`
```python
df["future_atr"] = (
    df["tr"].rolling(cfg.pred_horizon).mean().shift(-cfg.pred_horizon)
)
```

**Verdict** : C'est la *cible* d'entraînement du LightGBM volatility
forecaster, pas un feature. Pas de leak dans le replay (qui utilise un
ATR naif et n'appelle pas le forecaster). **OK** — mais il faut
**garantir** que `future_atr` n'est jamais consommé en inférence : le
contrat doit être que la pipeline de training drop cette colonne avant
de fitter `X`. À auditer dans Eval 04 (volatility forecaster).

### B6 — `rolling(...)` sans `center=True` (CAUSAL OK)

`src/intelligence/volatility_lgbm.py:127, 133, 134, 206, 207, 214, 215`
`src/intelligence/volatility_forecaster.py:497, 647, 660, 662, 664`
`src/environment/multi_timeframe_features.py:155, 156, 160, 161, 180`
`src/backtest/state_machine_replay.py:104, 105, 438, 439`
`src/environment/environment.py:831, 832`

**Verdict** : `rolling(N)` sans `center=True` est causal par défaut
(uses `[t-N+1, t]`). **OK pour tous ces sites.**

### B7 — `iloc[-1]` sur enriched / 4H (BORDER CASE)

`src/intelligence/sentinel_scanner.py:269, 425`, `multi_timeframe_features.py:280`
```python
latest = enriched.iloc[-1]
htf_bar = htf_df.loc[mask].iloc[-1]
```

**Verdict** : `iloc[-1]` = dernière ligne disponible. En mode live OK
(c'est le bar courant). En backtest, le replay parcourt `i` de
`warmup` à `len-1` et passe `enriched` complète à chaque itération —
si le scanner reçoit `enriched.iloc[-1]` au lieu de `enriched.iloc[i]`,
c'est un leak. Le replay actuel **n'utilise pas** `sentinel_scanner`
(il appelle directement `confluence.analyze`), donc OK en pratique.
**Mais** si le scanner est intégré au replay un jour, il faudra que
`run_scan` accepte un index courant et slice avant.

### B8 — `tr[i+1]`, `tr[i+2]` Yang-Zhang (PAS DE LEAK)

`src/intelligence/volatility_forecaster.py:689-702` (`_compute_yang_zhang_rv`)
N'utilise que `df["high"], df["low"], df["close"], df["open"]` du même
bar — c'est le RV par-bar, pas une fenêtre. **OK strictement**.

### Récapitulatif K2

| # | Site | Verdict | Action |
|---|---|---|---|
| B1 | strategy_features.py:617-618 | OK (mitigé par shift) | Test différentiel à ajouter |
| B2 | multi_timeframe_features.py:554-566 | **NON-CAUSAL** | **Patch shift(+2) avant tout MTF backtest** |
| B3 | environment.py:802 | Faible leak | `WARMUP=500` ou masquer NaN |
| B4 | state_machine_replay.py:145-146 | OK | — |
| B5 | volatility_forecaster.py:669 | Target, OK | Auditer pipeline training |
| B6 | rolling(N) sites multiples | OK | — |
| B7 | sentinel_scanner.py:269 | OK pour replay actuel | Refactor si intégré au replay |
| B8 | volatility_forecaster._compute_yang_zhang_rv | OK | — |

---

## K3 — Walk-Forward Designer

**Livré** : `reports/eval_18_walkforward_skeleton.py` (PR-ready, 320 lignes).

### Spec

| Fold | Période | Bars (M15) | Usage |
|---|---|---:|---|
| Train | 2019-01-03 → 2022-12-31 | ~70 000 | Tuning des paramètres (advisory only) |
| Embargo | 2023-01-01 → 2023-01-07 | 480 | Purge — gap obligatoire |
| Val | 2023-01-08 → 2023-12-31 | ~17 500 | Sélection du gagnant |
| Embargo | 2024-01-01 → 2024-01-07 | 480 | Purge — gap obligatoire |
| **Test** | 2024-01-08 → 2025-12-31 | ~35 000 | **OOS — chiffres publiables** |

### Choix de design (justifications)

- **Anchored expanding-train** plutôt que rolling : la stratégie n'a
  pas de saisonnalité forte sur Gold M15 ; agrandir la fenêtre train
  augmente la robustesse.
- **5-jours d'embargo** (480 bars) > tous les rolling features
  (`ZSCORE_WINDOW=200`, fractal `2N+1=5`, BOS retest ~50 bars).
- **Enrichment per-fold** : `SmartMoneyEngine.analyze()` est appelé
  séparément sur chaque slice. Empêche `rolling(center=True)` de
  traverser la frontière train/val.
- **Search space minimal** : `enter ∈ {40, 45, 50, 55, 60}`,
  `exit ∈ {25, 30, 35, 40}`, `max_age ∈ {8, 12, 16, 24}` — 80 combos
  candidats (vs 7 du sweep audit_backtest qui ne respectait aucune
  contrainte d'OOS).
- **Critère de sélection sur VAL** : log(PF) + 0.5·Sharpe − 0.1·DD/total_R,
  avec gates `trades ≥ 50`, `PF > 1.0`, `Sharpe > 0.5`. Si aucun
  candidat ne passe le gate → conclusion = NON commercialisable.

### Comment le brancher

```bash
python reports/eval_18_walkforward_skeleton.py \
    --csv data/XAU_15MIN_2019_2025.csv \
    --calendar data/economic_calendar_HIGH_IMPACT_2019_2025.csv \
    --out reports/walkforward_xau_m15.json
```

→ Génère `reports/walkforward_xau_m15.json` avec :
- `candidates[]` : 80 lignes (paramètres × train/val metrics)
- `winner` : config gagnante sur VAL
- `test` : métriques OOS — **les seuls chiffres légalement publiables**

---

## K4 — Cost Model Validator

### Coûts actuels dans le replay

`src/backtest/state_machine_replay.py:_build_trade` (ligne 691-733) :
```python
if direction == "LONG":
    pnl = exit_price - entry_price
else:
    pnl = entry_price - exit_price
r_mult = pnl / initial_risk if initial_risk > 0 else 0.0
```

**Verdict** : **0 spread, 0 slippage, 0 commission.** Un PF 1.086
publié sur cette base est gonflé.

### Coûts existants (non-branchés)

`src/environment/execution_model.py` fournit déjà `DynamicSlippageModel`
et `DynamicSpreadModel`. **Pas wired dans le replay.**

### Calibration XAU IC Markets / Pepperstone

Sources : tarifs IC Markets RAW Spread, Pepperstone Razor (2025-Q4) :

| Session UTC | Spread XAUUSD (raw) | Slippage typique | Comm. round-trip |
|---|---:|---:|---:|
| 0-8h (Asia) | 28 cts | 5-10 cts | $7 / lot |
| 8-13h (London) | 12 cts | 3-5 cts | $7 / lot |
| 13-17h (NY overlap) | 12 cts | 3-5 cts | $7 / lot |
| 17-21h (NY pm) | 18 cts | 5-8 cts | $7 / lot |
| 21-24h (After-hours) | 28 cts | 8-15 cts | $7 / lot |
| **NFP / FOMC ±15min** | **80-150 cts** | **30-100 cts** | $7 / lot |

DynamicSpreadModel (execution_model.py:74-80) emploie 8/3/3/5/8 *bps*
(en fraction). Sur Gold à $2 500, 3 bps = $0.75, 8 bps = $2. C'est
**conservateur** par rapport aux chiffres broker raw mais légèrement
optimiste sur les sessions Asia (réel ~28 cts ≈ 11 bps à $2 500).

### Fonction `cost(symbol, session, size, side)`

```python
def cost_per_round_trip(
    symbol: str, hour_utc: int, atr: float, median_atr: float,
    is_news_window: bool = False, lot_size: float = 1.0,
) -> float:
    """Round-trip cost in $ for a `lot_size`-lot trade.

    XAU/USD assumed: 1 lot = 100 oz. $7 commission/lot round-trip,
    spread per get_spread() applied at entry+exit (×2 = round-trip),
    slippage applied at entry+exit on 0.5×ATR per side.
    """
    spread_model = DynamicSpreadModel(news_multiplier=3.0)
    slip_model = DynamicSlippageModel(base_slippage=0.0001)
    spread_frac = spread_model.get_spread(hour_utc, is_news_window)
    slip_frac = slip_model.get_slippage(atr, median_atr)
    # Translate fractions to $ on Gold (price ~ $2500)
    avg_price = 2500.0  # placeholder; pass real bar.close in production
    spread_usd = 2 * spread_frac * avg_price * 100 * lot_size
    slip_usd = 2 * slip_frac * avg_price * 100 * lot_size
    commission_usd = 7.0 * lot_size  # round-trip
    return spread_usd + slip_usd + commission_usd
```

### Impact estimé sur PF

- 362 trades/an (cf baseline_2019_2025.md), prix moyen Gold ~$2 500.
- Cost moyen par trade = (spread 5 bps + slip 5 bps) × 2 × 2 500 × 100 +
  $7 = $50 + $7 = **~$57 par lot**.
- Initial risk ATR ~ $20 → **$57 / $20 = 2.85R de coût ?**
  ❌ Non — les R sont calculés sur prix par-once, pas par-lot.
  Recalcul : 5 bps de spread × $2500 = $1.25 par once par trade-side.
  Round-trip = $2.50 par once. ATR ~ $20 → coût en R = 2.5 / 20
  = **0.125 R par trade**.
- Avec expectancy actuelle de +0.024 R (baseline), **les coûts
  effacent 5× l'edge**. PF projeté post-coûts ≈ **0.93** (de 1.086
  à 0.93 environ).

→ Le baseline actuel est **non-rentable une fois coûts inclus**.

### Action

Brancher `execution_model.py` dans `_build_trade()` :

```python
def _build_trade(entry, exit_t, atr_at_entry, median_atr, hour_utc, ...):
    ...
    cost_usd = cost_per_round_trip("XAUUSD", hour_utc, atr_at_entry,
                                    median_atr, is_news_window=False)
    cost_per_oz = cost_usd / 100.0  # 1 lot = 100 oz Gold
    pnl_with_costs = pnl - cost_per_oz
    r_mult = pnl_with_costs / initial_risk
```

**À écrire en P1.** Refaire le sweep et le walk-forward avec coûts.

---

## K5 — Sizing Alignment Auditor

### Sizing dans le replay

`_build_trade` calcule `r_multiple = pnl / initial_risk` où
`initial_risk = |entry - stop_loss|` (en prix). **Le replay ignore le
position_size**. Chaque trade vaut **1 R**, indépendamment de la taille.

### Sizing prévu en prod

`src/intelligence/confluence_detector.py:317-334` :
```python
regime_mult = float(getattr(regime, "position_size_multiplier", 1.0))
news_mult = float(getattr(news, "position_multiplier", 1.0))
pos_mult = max(0.0, min(1.5, regime_mult * news_mult))
```

→ Multiplicateur de taille basé sur (regime × news), capé à [0, 1.5].
Mais **aucun calcul de la taille de base** (% equity, kelly,
vol-target). C'est laissé au consommateur (Telegram embed,
sentinel_scanner ne l'utilise pas).

### Verdict K5

**Désaligné.** Le replay traite chaque trade comme 1R fixe ; la prod
applique un multiplicateur 0-1.5×, mais aucune base de sizing
% equity n'est définie. Conséquences :
- En backtest, le PnL en $ (price-space) suppose 1 once / trade.
- En prod, un client recevant `pos_mult=1.2` ne sait pas si c'est
  1.2 lots, 1.2 % equity, ou 1.2 R.

**Recommandation P2** :
1. Ajouter `position_sizing_pct: float` à `StateMachineConfig` (default
   0.5 % equity per trade).
2. Calculer `size = (equity × pct) / (initial_risk × contract_size)`
   dans `_build_trade`.
3. Pondérer le R-multiple par `pos_mult` dans le replay :
   `r_mult_weighted = r_mult × pos_mult`.

Cf Eval 19 (Risk Management) — délégué.

---

## K6 — Monte Carlo Simulator (spec + pseudocode)

### Objectif

Sur chaque liste de trades (train_metrics, val_metrics, test_metrics
issus du walk-forward), produire un IC 95 % bootstrap pour PF, Sharpe
ann., max DD, expectancy. P-value vs random walk (H0 : trades ~
Bernoulli(0.5) avec payoff 2:1).

### Pseudocode `scripts/montecarlo_bootstrap.py` (à écrire)

```python
"""Monte Carlo bootstrap on trade list — outputs IC95% for PF/Sharpe/DD."""
import numpy as np
import pandas as pd
import argparse, json, math
from pathlib import Path

N_BOOT = 10_000

def bootstrap_metric(r_series: np.ndarray, metric_fn, rng) -> float:
    sample = rng.choice(r_series, size=len(r_series), replace=True)
    return metric_fn(sample)

def profit_factor(r):
    gw = r[r > 0].sum()
    gl = -r[r < 0].sum()
    return gw / gl if gl > 0 else float('inf')

def sharpe_per_trade(r):
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return r.mean() / r.std()

def max_drawdown(r):
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max())

def random_walk_null(n: int, rr: float, n_sim: int, rng) -> np.ndarray:
    """Generate n_sim bootstrap samples under H0: 50% wins, 2R/-1R."""
    # n trades, each +rr or -1.0 with p=0.5 (no edge)
    pf_dist = np.empty(n_sim)
    for i in range(n_sim):
        wins = rng.random(n) < 0.5
        sample = np.where(wins, rr, -1.0)
        pf_dist[i] = profit_factor(sample)
    return pf_dist

def main(trades_csv: str, out_json: str, rr_target: float = 2.0):
    df = pd.read_csv(trades_csv)
    r = df["r_multiple"].to_numpy()
    rng = np.random.default_rng(42)

    # 1. Bootstrap distributions
    pf_boot = np.array([bootstrap_metric(r, profit_factor, rng) for _ in range(N_BOOT)])
    sh_boot = np.array([bootstrap_metric(r, sharpe_per_trade, rng) for _ in range(N_BOOT)])
    dd_boot = np.array([bootstrap_metric(r, max_drawdown, rng) for _ in range(N_BOOT)])
    exp_boot = np.array([bootstrap_metric(r, np.mean, rng) for _ in range(N_BOOT)])

    # 2. IC 95%
    def ci(x): return (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))

    # 3. P-value vs H0 random walk
    pf_h0 = random_walk_null(len(r), rr_target, 5000, rng)
    pf_observed = profit_factor(r)
    p_value = (pf_h0 >= pf_observed).mean()

    result = {
        "n_trades": len(r),
        "observed": {
            "profit_factor": pf_observed,
            "sharpe_per_trade": float(sharpe_per_trade(r)),
            "max_drawdown_r": float(max_drawdown(r)),
            "expectancy_r": float(r.mean()),
        },
        "bootstrap_ci_95": {
            "profit_factor": ci(pf_boot),
            "sharpe_per_trade": ci(sh_boot),
            "max_drawdown_r": ci(dd_boot),
            "expectancy_r": ci(exp_boot),
        },
        "p_value_vs_random_walk": float(p_value),
        "verdict": (
            "edge significant (p<0.05) and CI(PF) > 1.0"
            if p_value < 0.05 and ci(pf_boot)[0] > 1.0
            else "edge NOT significant"
        ),
    }
    Path(out_json).write_text(json.dumps(result, indent=2))
```

### IC bootstrap chiffrés actuels

**Non calculables depuis ce poste** sans exécuter le script ci-dessus.
Les chiffres `replay_*.json` actuels n'incluent ni IC ni p-value.

→ **Action P1** : écrire le script et le lancer sur
`reports/baseline_full_trades.csv`. Si l'IC PF inclut 1.0,
**enterrer le baseline**.

---

## K7 — SPA / Reality Check

### Multiple-testing dans `audit_backtest.py`

`scripts/audit_backtest.py:76-85` itère 7 configs :
```python
SWEEP_CONFIGS = [
    ("production_default",  75, 55, ...),
    ("relaxed_55",          55, 40, ...),
    ...
    ("relaxed_30",          30, 15, ...),
]
```

Le rapport headline le **meilleur PF** (`relaxed_30 = 0.96`). Sous H0
(aucune stratégie n'a d'edge), l'espérance du best-of-7 PF est
significativement supérieure à 1.0 simplement par chance. Pour 7
tests indépendants avec PF distribué selon F, `E[max_7]` >
quantile(F, 1 − 1/7) ≈ quantile-86 %.

### Recommandation : Hansen SPA (2005)

**Hansen's Superior Predictive Ability test** (Hansen, 2005, *J.
Bus. Econ. Stat.*) corrige ce biais en testant H0 : « la meilleure
config n'a pas d'edge significatif vs benchmark random walk ».

### Pseudocode `scripts/hansen_spa.py` (à écrire)

```python
"""Hansen 2005 SPA test — corrects for multiple-testing in sweeps."""
import numpy as np
from scipy import stats

def hansen_spa(loss_matrix: np.ndarray, n_boot: int = 5000) -> dict:
    """
    loss_matrix: shape (T, M) where T = bootstrap blocks, M = strategies.
        loss[t, k] = -PnL of strategy k at block t (so lower = better).
    Returns p-value for H0: best strategy has NO edge over benchmark.
    """
    T, M = loss_matrix.shape
    # 1. Block bootstrap (stationary, mean block size = sqrt(T))
    block_size = int(np.sqrt(T))
    rng = np.random.default_rng(42)

    # 2. Test statistic: max over k of standardised (mean loss diff vs bench)
    bench = np.zeros(T)  # assume benchmark = no-trade
    diffs = bench[:, None] - loss_matrix  # +ve = strategy beats bench
    means = diffs.mean(axis=0)
    stds = diffs.std(axis=0, ddof=1) + 1e-9
    t_stats = means / (stds / np.sqrt(T))
    t_obs = float(t_stats.max())

    # 3. Recentered bootstrap distribution of max-stat under H0
    boot_max = np.zeros(n_boot)
    for b in range(n_boot):
        # Stationary block bootstrap indices
        idx = []
        while len(idx) < T:
            start = rng.integers(0, T)
            blen = rng.geometric(1 / block_size)
            idx.extend(range((start + i) % T for i in range(blen)))
        idx = np.array(idx[:T])
        d_b = diffs[idx] - means[None, :]  # recentered
        m_b = d_b.mean(axis=0)
        s_b = d_b.std(axis=0, ddof=1) + 1e-9
        boot_max[b] = float((m_b / (s_b / np.sqrt(T))).max())

    p_value = float((boot_max >= t_obs).mean())
    return {"t_obs": t_obs, "p_value_spa": p_value, "M_strategies": M}
```

### Action

1. Refactor `audit_backtest.py` pour exporter trades par config dans
   un `loss_matrix.npy`.
2. Lancer `hansen_spa.py` sur cette matrice.
3. Si p_value > 0.05 → **aucune config testée n'a d'edge significatif
   après correction multiple-testing**.

**À écrire en P2.**

---

## K8 — Regime Decomposer (spec)

### Idée

Pour chaque trade, tagger (regime_at_entry, vol_regime_at_entry, year)
et calculer PF × régime × année. Permet d'identifier le « cheval de
bataille » du système.

### Stack disponible

- `src/agents/market_regime_agent.py` — HMM à 3 états (uptrend/downtrend/ranging).
- `src/backtest/state_machine_replay.classify_regime_series` — heuristique SMA-slope
  (déjà utilisée pour le replay).

### Tableau attendu (à remplir après walk-forward)

```
              | 2019  | 2020  | 2021  | 2022  | 2023  | 2024  | 2025  |
--------------|-------|-------|-------|-------|-------|-------|-------|
trending_up   |  PF1  |  PF2  | ...                                   |
trending_dn   |  ...                                                  |
ranging       |  ...                                                  |
high_vol      |  ...                                                  |
```

### Spec script `scripts/regime_decompose.py`

```python
import pandas as pd
def regime_decompose(trades_csv: str, enriched_pickle: str) -> pd.DataFrame:
    trades = pd.read_csv(trades_csv, parse_dates=["entry_bar", "exit_bar"])
    enriched = pd.read_pickle(enriched_pickle)
    # tag each trade with regime at entry
    trades["regime"] = trades["entry_bar"].map(
        lambda ts: classify_regime_series(enriched).loc[ts]
    )
    trades["year"] = trades["entry_bar"].dt.year
    # compute per-bucket PF
    grouped = trades.groupby(["regime", "year"]).apply(
        lambda g: pd.Series({
            "trades": len(g),
            "win_rate": (g["r_multiple"] > 0).mean(),
            "pf": g[g["r_multiple"] > 0]["r_multiple"].sum() /
                  abs(g[g["r_multiple"] < 0]["r_multiple"].sum() or 1e-9),
            "expectancy": g["r_multiple"].mean(),
        })
    )
    return grouped
```

**Heatmap attendue** : c'est la pièce maîtresse de Eval 18 pour
identifier *« notre système est rentable en trending bull, perdant en
ranging »* — info qui transforme un produit générique en produit
ciblé avec discipline d'entrée.

---

## K9 — OOS Live Tracker (spec)

### Objectif

Job nightly qui paper-trade en live et compare aux stats publiées sur
landing. Si déviation > 1σ par rapport à l'IC bootstrap → alerte
automatique → marketing/landing à corriger.

### Schéma DB minimal (SQLite)

```sql
CREATE TABLE oos_trades (
    trade_id    TEXT PRIMARY KEY,
    symbol      TEXT NOT NULL,
    direction   TEXT CHECK(direction IN ('LONG', 'SHORT')),
    entry_ts    TEXT NOT NULL,           -- ISO 8601 UTC
    exit_ts     TEXT,                    -- nullable while open
    entry_price REAL NOT NULL,
    exit_price  REAL,
    stop_loss   REAL,
    take_profit REAL,
    confluence_score REAL,
    r_multiple  REAL,
    cost_usd    REAL DEFAULT 0,
    exit_reason TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE oos_metrics_daily (
    date          TEXT PRIMARY KEY,      -- YYYY-MM-DD
    n_trades      INT NOT NULL,
    win_rate      REAL,
    profit_factor REAL,
    sharpe_per_t  REAL,
    expectancy_r  REAL,
    max_dd_r      REAL,
    deviation_sigma_pf REAL,             -- vs published baseline IC
    deviation_sigma_sharpe REAL,
    alert_triggered INT DEFAULT 0
);

CREATE TABLE published_baseline (
    -- Snapshot of what's currently on landing
    version       TEXT PRIMARY KEY,
    published_at  TEXT NOT NULL,
    pf_point      REAL,
    pf_ci_low     REAL,
    pf_ci_high    REAL,
    sharpe_point  REAL,
    sharpe_ci_low REAL,
    sharpe_ci_high REAL,
    git_sha       TEXT
);
```

### Spec cron job

```python
# scripts/oos_live_tracker.py — run via cron @ 03:00 UTC daily
def daily_check():
    # 1. Pull latest paper-trade fills from sentinel_scanner output
    # 2. Aggregate: PF/Sharpe/DD on rolling 30-trade window
    # 3. Load current `published_baseline`
    # 4. Compute Z = (live_pf - baseline_pf) / baseline_pf_sigma
    # 5. If |Z| > 1.0 → alert email + freeze landing
    # 6. Append daily metrics to oos_metrics_daily
```

### Cron entry (Linux/Docker)

```cron
0 3 * * * cd /app && python -m scripts.oos_live_tracker --alert-email founder@smartsentinel.ai
```

### Alerte (template)

```
SUBJECT: [SMART SENTINEL ALERT] OOS deviation > 1σ on PF

Date: 2026-XX-XX
Live PF (last 30 trades): 0.84
Baseline PF (published):  1.42 [IC95: 1.18-1.69]
Z-score:                  -2.1σ
Action required: review landing page within 24h.
```

---

## K10 — Marketing Risk Reviewer

→ Voir `BACKTEST_LEGAL_GUARDRAILS.md` (livré). Synthèse :

- **0 chiffre actuel publiable** sans walk-forward.
- `BUSINESS_PLAN_SMART_SENTINEL.md` lignes 39-42 (anciennes métriques
  RL) : à reformuler en *« phase de validation »*.
- `COMMERCIALIZATION_REPORT.md` : aucun chiffre PnL trouvé — OK.
- `README.md` : aucun chiffre PnL trouvé — OK, ne pas en ajouter.
- Template de citation OOS obligatoire fourni au §2.1 du guardrails.
- Disclaimers FTC/AMF prêts-à-coller fournis au §2.2.
- Process d'approbation 8 étapes au §6 — **aucune étape ne peut être
  sautée**.

---

## K11 — Synthèse finale

### Verdict

**❌ NON COMMERCIALISABLE en l'état.** Note crédibilité = **2/10**.

Détail :
- (+1) Harnais `state_machine_replay` propre, 18 tests, déterministe.
- (+1) `BacktestNewsProvider` correctement branchée (CSV → blackout).
- (0) `audit_backtest.py` exhaustif **mais** in-sample, pas de SPA.
- (−1) Aucun walk-forward exécuté.
- (−1) Aucun coût modélisé dans `_build_trade`.
- (−1) Look-ahead confirmé sur sous-système MTF (B2).
- (−2) Tier system invalide (Pearson −0.023).
- (−1) 19 fichiers `replay_*.json` à la racine, tous in-sample,
  facilement confondus avec des résultats publiables.
- (−1) Pas d'IC bootstrap. Pas de p-value. Aucune assertion
  statistiquement défendable.
- (−1) Sizing désaligné replay vs prod.

### Top-3 biais

1. **B2 — `iloc[i+1/+2]` swing detector** (multi_timeframe_features.py:554-566)
   = look-ahead confirmé.
2. **Coûts à zéro** (state_machine_replay.py:_build_trade)
   = PnL gonflé de 0.10-0.20 PF.
3. **Multiple-testing non-corrigé** (audit_backtest.py:76-85)
   = best-of-7 inflate +15-25 %.

### Plan d'action

#### P1 — Bloquant avant tout marketing chiffré (Sprint en cours)

- [P1.1] Exécuter `reports/eval_18_walkforward_skeleton.py` sur 6 ans.
- [P1.2] Brancher `execution_model.DynamicSpreadModel` +
  `DynamicSlippageModel` + commission $7/lot dans
  `state_machine_replay._build_trade`. Refaire walk-forward.
- [P1.3] Écrire `scripts/montecarlo_bootstrap.py` (pseudocode K6
  fourni). Lancer sur les trades OOS. **Aucun chiffre ne sort de
  l'entreprise tant qu'IC PF ne contient pas > 1.0.**
- [P1.4] Migrer le wording landing/business plan vers le langage
  qualitatif §4 du guardrails en attendant.

#### P2 — Avant scaling commercial

- [P2.1] Patcher B2 (swing detector causal) — bloque tout backtest MTF.
- [P2.2] Écrire `scripts/hansen_spa.py` (pseudocode K7 fourni). Re-tester
  le sweep `audit_backtest.py` avec correction Hansen 2005. Si p > 0.05,
  abandonner ce sweep et n'utiliser que le walk-forward.
- [P2.3] Aligner sizing replay vs prod (Eval 19) :
  `position_sizing_pct` dans `StateMachineConfig`, R-multiple pondéré
  par `pos_mult`.
- [P2.4] Implémenter le calibrage Confluence Score (LightGBM
  classifier — voir `confluence_calibration.md`). Tier system actuel
  est non-prédictif.
- [P2.5] Écrire `scripts/regime_decompose.py` (spec K8 fournie). Heatmap
  régime × année — cheval de bataille du système.

#### P3 — Hardening continu

- [P3.1] Déployer `oos_live_tracker.py` (spec K9 fournie) — cron
  nightly + alerte 1σ + freeze landing.
- [P3.2] Migrer les 19 `replay_*.json` à la racine vers
  `reports/replays_archive/2026-04-23-iteration/`.
- [P3.3] Ajouter test différentiel `tests/test_smart_money_engine_causality.py`
  vérifiant que `rolling(center=True).max().shift(N)` ≡ pure causal max.
- [P3.4] Augmenter `WARMUP` à 500 dans `audit_backtest.py` pour
  absorber bfill leakage (B3).
- [P3.5] Exporter le walk-forward + bootstrap result dans la signature
  du landing : *« code disponible : github.com/[org]/[repo],
  commit hash : XXX »*.

### KPIs cibles (post-corrections)

| Métrique | Confiance | Borderline | Inadmissible (état actuel) |
|---|---:|---:|---:|
| PF OOS | ≥ 1.5 | 1.2-1.5 | 1.086 in-sample |
| Sharpe ann. OOS | ≥ 0.8 | 0.5-0.8 | 0.59 in-sample |
| Max DD OOS | < 25 % equity | 25-35 % | 23.14 R unscaled |
| IC95% PF lower bound | > 1.0 | > 0.9 | inconnu |
| p-value vs random walk | < 0.05 | < 0.10 | inconnu |
| p-value Hansen SPA | < 0.05 | < 0.10 | inconnu |

### Note finale

**2/10**. Le harnais est techniquement sain (parsing déterministe,
event-driven, news provider propre), mais la méthodologie qui en
découle (single-fold sweep + cherry-pick) ne supporte aucune assertion
marketing. Tant que P1 n'est pas fait, communiquer **uniquement** en
mode qualitatif (langage §4 du guardrails). Une fois P1 fait, la note
peut bondir à 7-8/10 si OOS PF > 1.5 ; sinon à 4/10 (« en bêta paper-
trade ») ou rester à 2/10 (« le système ne s'est pas montré rentable
hors-échantillon, à pivoter »).

---

_Fichiers livrés :_
- `reports/eval_18_backtest.md` (ce document)
- `reports/eval_18_walkforward_skeleton.py` (script walk-forward PR-ready)
- `BACKTEST_LEGAL_GUARDRAILS.md` (chiffres autorisés / à enterrer)
