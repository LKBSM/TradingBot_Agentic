# Eval 20 — Multi-Asset & Multi-Timeframe (TAM expansion)

**Date** : 2026-04-26
**Scope** : 6 presets actuels (XAUUSD, EURUSD, BTCUSD, US500, GBPUSD, USDJPY),
`resample_ohlcv()` (`src/intelligence/volatility_forecaster.py:179`),
matrice 6×5 = 30 cellules, 4 candidats new-asset, 4 packs commerciaux.

---

## Executive Summary

**Note globale : 5.0 / 10**

Le système est **architecturalement multi-asset-ready** (6 presets propres,
`resample_ohlcv` fonctionnel, `price_decimals` corrects), mais
**opérationnellement single-asset** : 5 des 6 presets n'ont **aucun CSV
local** (`data/` ne contient que les 2 fichiers Gold). La matrice 30
cellules est donc un **template** : 1 cellule ground-truth (XAU M15 PF
1.086, baseline 6 ans), 4 cellules XAU resampling-ready (M5/H1/H4/D1), 25
cellules en `MISSING_DATA`. Avant d'évaluer le TAM multi-asset, il faut
résoudre l'onboarding data (Dukascopy / MT5 export).

**Verdict commercial** :
- **Keep** : XAUUSD (asset signature, données de qualité), EURUSD
  (corrélation négative XAU = utile portfolio, faible coût onboarding).
- **Conditional keep** : GBPUSD, USDJPY (similaires EURUSD, mais
  cannibalisation forte ρ > 0.7).
- **Drop / Park** : BTCUSD (24/7 coût LLM 2x + asymétrie crypto vs SMC),
  US500 (RTH-only, bars_per_day=28 = pipeline non testé).
- **Add P1** : USOIL (corrélation XAU + macro forte, smart-money relevant).
- **Add P2** : NAS100 (upsell index trader US).
- **Drop / Defer** : AUDJPY (carry trade, trop spécialisé), ETHUSDT (avant BTC profitable).

**KPIs du prompt** : aucun atteint actuellement
- ≥ 4 symbols PF OOS > 1.2 → **0/4** (XAU 1.086 in-sample)
- 0 doublon ρ > 0.7 → impossible à mesurer (data manquante)
- 2 nouveaux assets PF > 1.0 onboardés → **0/2**
- FX Pack lancé → **non**
- +30 % MRR projeté → non mesurable

---

## M1 — Inventaire (Multi-Asset Lead)

### 6 presets actuels

Source unique : `src/intelligence/volatility_forecaster.py:38-153` (registre
`get_instrument_registry()`).

| Symbol | file:line | bars_per_day | price_decimals | sl_atr_mult | tp_atr_mult | Sessions UTC | Calendar events |
|--------|-----------|--------------|----------------|-------------|-------------|--------------|-----------------|
| XAUUSD | vol_forecaster.py:41 | 96 | 2 | 2.0 | 4.0 | asian/london/ny_overlap/ny_after/after_hrs (0-24) | NFP, FOMC, CPI, GDP, PCE, Retail Sales |
| EURUSD | :61 | 96 | 5 | 1.5 | 3.0 | asian/london/ny_overlap/ny_after/after_hrs (0-24) | NFP, ECB Rate, CPI, GDP, ECB Press |
| BTCUSD | :81 | 96 | 2 | 2.0 | 4.0 | asian/london/us/after_hrs (0-24) | FOMC, CPI **only 3 events** |
| US500  | :98 | **28** | 1 | 1.5 | 3.0 | pre_market/regular/after_hrs (8-20) | NFP, CPI, FOMC, GDP |
| GBPUSD | :115 | 96 | 5 | 1.5 | 3.0 | asian/london/ny_overlap/ny_after/after_hrs (0-24) | NFP, BOE Rate, CPI, GDP |
| USDJPY | :134 | 96 | 3 | 1.5 | 3.0 | asian/london/ny_overlap/ny_after/after_hrs (0-24) | NFP, BOJ Rate, CPI, FOMC |

### Données disponibles (`ls data/`)

| File | Symbol | TF | Coverage | Status |
|------|--------|----|----|--------|
| `XAU_15MIN_2019_2024.csv` | XAUUSD | M15 | 97.6 % | **PROD-GRADE** |
| `XAU_15MIN_2019_2025.csv` | XAUUSD | M15 | 63 % | DEPRECATED (cf. data_quality_audit) |
| `economic_calendar_2019_2025.csv` | — | — | — | shared |
| `economic_calendar_HIGH_IMPACT_2019_2025.csv` | — | — | — | shared |

**Pour les 5 autres presets : ZÉRO fichier OHLCV présent.** Le pipeline
multi-asset ne peut tourner qu'en théorie.

### Matrice cible 6 × 5 = 30 cellules

| Symbol \ TF | M5 | M15 | H1 | H4 | D1 |
|-------------|----|-----|----|----|----|
| XAUUSD      | RESAMPLE_NEEDED | **GROUND_TRUTH** PF 1.086 | RESAMPLE_NEEDED | RESAMPLE_NEEDED | INSUFFICIENT_BARS |
| EURUSD      | MISSING_DATA | MISSING_DATA | MISSING_DATA | MISSING_DATA | MISSING_DATA |
| BTCUSD      | MISSING_DATA | MISSING_DATA | MISSING_DATA | MISSING_DATA | MISSING_DATA |
| US500       | MISSING_DATA | MISSING_DATA | MISSING_DATA | MISSING_DATA | MISSING_DATA |
| GBPUSD      | MISSING_DATA | MISSING_DATA | MISSING_DATA | MISSING_DATA | MISSING_DATA |
| USDJPY      | MISSING_DATA | MISSING_DATA | MISSING_DATA | MISSING_DATA | MISSING_DATA |

→ **1 cellule chiffrée / 30**. Le sweep 30-cells est conçu mais non
exécutable end-to-end aujourd'hui.

---

## M2 — Sweep Engineer (design)

### Spec exécutable : `reports/eval_20_sweep_runner.py`

Le runner :
1. Pour chaque (symbol × tf) : charge le CSV base, resample via
   `resample_ohlcv()` si tf ≠ tf_base, enrichit via `SmartMoneyEngine`,
   passe à `SignalReplay` avec config par défaut (enter=75, exit=55).
2. Catch broad → status `LOAD_ERROR`, `RESAMPLE_ERROR`, `INSUFFICIENT_BARS`,
   `MISSING_DATA` au lieu d'exception.
3. Mode `--walk-forward` : OOS = derniers 15 % (Sprint 18 alignment).

**Run réel actuel** :
```
python reports/eval_20_sweep_runner.py --data-dir data \
    --out reports/eval_20_sweep_30cells.csv
```
Résultat attendu : 5 cellules XAU (M5/M15/H1/H4/D1) + 25 cellules
MISSING_DATA. Cf. `reports/eval_20_sweep_30cells.csv` pour le template
pré-rempli avec la cellule ground-truth XAU M15.

### Heatmap PF (estimative — XAU only ground-truth)

| Symbol \ TF | M5 | M15 | H1 | H4 | D1 |
|-------------|----|-----|----|----|----|
| XAUUSD      | TBD | **1.086** | TBD | TBD | (insuf bars) |
| EURUSD      | — | — | — | — | — |
| BTCUSD      | — | — | — | — | — |
| US500       | — | — | — | — | — |
| GBPUSD      | — | — | — | — | — |
| USDJPY      | — | — | — | — | — |

→ **Aucune décision keep/drop sur PF empirique possible aujourd'hui.**
Toute conclusion est un *prior* sur la nature de l'asset.

---

## M3 — Instrument Config Auditor

Audit `param × préset × verdict` (`vol_forecaster.py:38-153`) :

| Paramètre              | XAUUSD | EURUSD | BTCUSD | US500 | GBPUSD | USDJPY |
|------------------------|--------|--------|--------|-------|--------|--------|
| `symbol`               | OK     | OK     | OK     | OK    | OK     | OK     |
| `timeframe` (M15)      | OK     | OK     | OK     | OK    | OK     | OK     |
| `bars_per_day`         | 96 OK  | 96 OK  | 96 OK  | **28**| 96 OK  | 96 OK  |
| `price_decimals`       | 2 OK   | 5 OK   | 2 OK   | 1 OK  | 5 OK   | 3 OK   |
| `sl_atr_mult` / `tp_atr_mult` | 2.0/4.0 | 1.5/3.0 | 2.0/4.0 | 1.5/3.0 | 1.5/3.0 | 1.5/3.0 | OK différencié vol |
| `session_hours` cohérent ? | OK | OK    | **manque US session séparée** | OK | OK | OK |
| `calendar_events` exhaustif ? | OK 9 | OK 8 | **3 trop pauvre** | OK 5 | OK 6 | OK 5 |
| `pip_value`            | **MISSING** | **MISSING** | **MISSING** | **MISSING** | **MISSING** | **MISSING** |
| `weekend_behavior`     | **MISSING** | **MISSING** | **MISSING** | **MISSING** | **MISSING** | **MISSING** |
| `news_relevance_weight`| **MISSING** | **MISSING** | **MISSING** | **MISSING** | **MISSING** | **MISSING** |
| ATR baseline (USD)     | **MISSING** | **MISSING** | **MISSING** | **MISSING** | **MISSING** | **MISSING** |

### Verdict
- `InstrumentConfig` couvre la **vol forecasting** mais pas la **couche
  trading complète** (sizing, weekend, news weighting).
- Manquants critiques : `pip_value` (impossible de calculer le notional
  d'une position sans), `weekend_behavior` (BTC = no gap, FX = Sun open
  gap, XAU = Sun open gap), `news_relevance_weight` (NFP impacte XAU 1.0
  / impacte BTC 0.3 — pas modélisé).

### Patches recommandés (PR-ready, non appliqués)

```python
# Patch 1: enrichir InstrumentConfig (vol_forecaster.py:239)
@dataclass
class InstrumentConfig:
    # ... existing fields ...
    pip_value: float = 1.0          # USD per pip per 1 lot
    weekend_behavior: str = "gap"    # "gap" | "continuous" | "rth_only"
    news_relevance_weight: float = 1.0  # multiplier on news blackout duration
    atr_baseline_usd: float = 0.0       # typical ATR_14 in USD (telemetry)
```

```python
# Patch 2: presets (vol_forecaster.py:41+)
"XAUUSD":  pip_value=10.0, weekend_behavior="gap", news_relevance_weight=1.0, atr_baseline_usd=4.5
"EURUSD":  pip_value=10.0, weekend_behavior="gap", news_relevance_weight=0.7, atr_baseline_usd=0.0008
"BTCUSD":  pip_value=1.0,  weekend_behavior="continuous", news_relevance_weight=0.3, atr_baseline_usd=400.0
"US500":   pip_value=1.0,  weekend_behavior="rth_only", news_relevance_weight=1.0, atr_baseline_usd=8.0
"GBPUSD":  pip_value=10.0, weekend_behavior="gap", news_relevance_weight=0.7, atr_baseline_usd=0.0010
"USDJPY":  pip_value=9.0,  weekend_behavior="gap", news_relevance_weight=0.7, atr_baseline_usd=0.12
```

---

## M4 — Session & Calendar Auditor

### Calendrier UTC sessions actives

| Symbol | Session principale | Session secondaire | Weekend | RTH only ? |
|--------|--------------------|--------------------|---------|------------|
| XAUUSD | London 08-13 + NY 13-21 UTC | Asia 00-08 (faible) | Sun 21:00 → Fri 21:00 UTC | non |
| EURUSD | London 07-12 + NY overlap 12-16 | Asia 00-07 | Sun 21:00 → Fri 21:00 UTC | non |
| BTCUSD | 24/7 (US 13-21 le plus liquide) | — | aucun (continuous) | non |
| US500  | RTH 14:30 → 21:00 UTC + extended 13:00-14:30 / 21:00-01:00 | — | Fri 21:00 → Sun 23:00 UTC | **OUI** |
| GBPUSD | London 07-12 + NY overlap 12-16 | Asia 00-07 (faible) | Sun 21:00 → Fri 21:00 UTC | non |
| USDJPY | Asia 00-08 + London 08-13 | NY overlap 13-17 | Sun 21:00 → Fri 21:00 UTC | non |

### Audit news coverage

`EconomicCalendarFetcher` (`src/agents/news/economic_calendar.py:130`) supporte un
`CURRENCY_MAP` pour détecter la juridiction d'un événement (`:257
_detect_currency`). Couverture observée :
- **USD** : OK (NFP, FOMC, CPI — événements présents dans events factices :547)
- **EUR** : OK (ECB)
- **GBP** : OK (BOE)
- **JPY** : présent dans XAU/USDJPY presets mais **absent du factice
  events test fixture** :547-571 — risque de under-detection en backtest.
- **CHF** : listé dans `relevant_currencies` (config.py:651) mais aucun
  événement SNB dans presets vol_forecaster. **Gap.**

**Action P1** : ajouter SNB dans XAUUSD/EURUSD calendar_events si
volatilité Franc Suisse importante.

**Action P2** : croiser le CSV ForexFactory live (`scripts/fetch_forexfactory_live.py`)
avec presets pour vérifier que tous les events sont effectivement
disponibles dans le fichier source.

---

## M5 — Correlation Analyst

### Matrice ρ théorique (6 presets)

Sources : conventions FX academia, Bouri et al. 2017 (gold-DXY), Klein 2018 (BTC equity).

|        | XAU   | EUR   | GBP   | JPY   | BTC   | SPX   |
|--------|-------|-------|-------|-------|-------|-------|
| XAU    | 1.00  | +0.40 | +0.30 | -0.20 | +0.15 | -0.20 |
| EUR    |       | 1.00  | **+0.75** | +0.40 | +0.10 | +0.30 |
| GBP    |       |       | 1.00  | +0.35 | +0.10 | +0.30 |
| JPY    |       |       |       | 1.00  | +0.05 | +0.10 |
| BTC    |       |       |       |       | 1.00  | **+0.55** (post-2021) |
| SPX    |       |       |       |       |       | 1.00  |

(`USDJPY` ici = inverse de JPY-strength : convention "EUR" = EURUSD.)

### Clusters identifiés

1. **Risk-on basket** : EURUSD + GBPUSD + (US500/NAS100) + BTC (ρ > 0.5
   sur regime bull markets)
2. **USD-strength inverse** : XAUUSD + EURUSD + GBPUSD short DXY (ρ ~ -0.7
   vs DXY)
3. **Safe haven** : XAUUSD + USDJPY long (carry vs vol regime)

### Doublon détecté
- **EURUSD vs GBPUSD : ρ ~ +0.75** → un signal sur l'un duplique l'autre
  ~75 % du temps. **Cap portfolio à 1 position parmi les 2.**

### Spec règle portfolio cap (PR-ready)
```python
PORTFOLIO_CORRELATION_CAP = {
    "thresholds": {
        "ρ_pair_30d": 0.7,         # if rolling 30d ρ > 0.7 → bucket
        "max_per_bucket": 1,        # max 1 active signal per bucket
    },
    "buckets_static": [
        ["EURUSD", "GBPUSD"],       # FX London-correlated
        ["BTCUSD", "ETHUSD"],       # Crypto top-2
        ["US500", "NAS100"],        # US equity index
    ],
}
```

### Script ρ rolling 30j (data dispo XAU only)
Aucune donnée multi-asset présente → matrice ρ non calculable
empiriquement aujourd'hui. Lorsque CSVs onboardés :
```python
import pandas as pd
returns = pd.DataFrame({s: load(s).pct_change() for s in SYMBOLS})
rho = returns.rolling(30 * 96).corr().unstack().mean()
```

---

## M6 — Asymmetry Investigator

### Hypothèse à tester
XAU sur 2019-2024 montre **shorts profitables, longs marginaux**
(memory `xau_replay_findings_2026_04_23.md`). Question : pattern systémique
(bug pipeline) ou biais structurel asset (Gold = safe haven, drawdown
plus rapide que recovery) ?

### Test diagnostic conceptuel

```python
# pseudo-code à intégrer dans run_backtest.py
trades_df = pd.DataFrame([t.to_dict() for t in results.trades])
for sym in SYMBOLS:
    by_side = trades_df.groupby(["symbol", "year", "side"]).agg(
        n=("pnl_R", "count"),
        pf=("pnl_R", lambda x: x[x>0].sum() / abs(x[x<0].sum())),
        win_rate=("pnl_R", lambda x: (x > 0).mean()),
    )
```

**Verdict attendu (priors)** :
- Si XAU asymétrie persiste sur EUR/GBP/JPY → bug pipeline (CHOCH
  detection biaisé long, ou retest rule asymétrique).
- Si seul XAU asymétrique → biais structurel (Gold drawdowns 2x faster
  que recovery cf. Baur & Lucey 2010).

**Action P1** : exécuter sur XAU + EURUSD dès qu'EURUSD est onboardé. Si
asymétrie EURUSD < 30 % de l'asymétrie XAU → biais structurel confirmé,
on peut ship XAU short-only en attendant.

---

## M7 — New Asset Scout

| Candidat | Data feed dispo | Coût 2026/mo | Smart-money relevance | Effort onboarding | PF estimé 1y | Verdict |
|----------|-----------------|--------------|-----------------------|-------------------|--------------|---------|
| **USOIL** (WTI) | Dukascopy free / Polygon 29 USD / Twelve Data 79 USD | 0 → 29 USD | **HIGH** — OPEC events, CHOCH net sur supply news | ~6h (preset + tests + tuning ATR) | 1.0-1.2 (volatile, mean-revert) | **P1 — ADD** |
| **NAS100** | Dukascopy free / MT5 broker free / Polygon 29 USD | 0 → 29 USD | MEDIUM — earnings season noisy | ~8h (RTH session, bars_per_day=28) | 0.9-1.1 | **P2 — ADD post-USOIL** |
| **AUDJPY** | Dukascopy free / MT5 free | 0 USD | MEDIUM — risk-on/off proxy mais carry-driven | ~5h | 0.8-1.0 | **P3 — DEFER** (trop spécialisé) |
| **ETHUSDT** | Binance public API free / Tiingo 10 USD / Twelve Data 79 USD | 0 USD | LOW — mêmes problèmes que BTC + ρ 0.85 vs BTC | ~4h | inconnu | **P3 — DEFER** (avant BTC profitable) |

### Justifications
- **USOIL P1** : marché smart-money textbook, OPEC Cushing inventory
  events → BOS/CHOCH propres. Corrélation négative DXY (~-0.5) fait
  sens pour Metal Pack. Dukascopy XAU downloader (`scripts/download_dukascopy_xau.py`)
  est **réutilisable directement** en changeant le symbole instrument.
- **NAS100 P2** : upsell index trader US, complète US500 dans Index Pack.
  Mais RTH-only nécessite tests de sessions séparés.
- **AUDJPY P3** : niche carry-trade, les vrais traders AUDJPY préfèrent
  des outils macro spécialisés (forexlive). Faible TAM.
- **ETHUSDT P3** : pas avant que BTCUSD démontre PF > 1.0. Sinon on
  multiplie les pertes par 2.

---

## M8 — Multi-TF Confluence Architect

### Design score-boost confluent multi-TF

```python
# pseudo-code dans confluence_detector.py
def compute_confluence(self, df_M15, df_H4=None):
    base_score = self._score_single_tf(df_M15)
    if df_H4 is not None:
        parent_signal = self._detect_signal_simple(df_H4)
        if parent_signal == base_score.direction:
            base_score.score *= 1.2   # boost +20 %
            base_score.tier_hint = "PREMIUM_MTF"
        elif parent_signal == -base_score.direction:
            base_score.score *= 0.7   # contradicting parent
    return base_score
```

### Estimation impact volume

Sur les 1597 trades XAU baseline 2019-2024 :
- Score actuel moyen 46.4 (cf. baseline_2019_2025.md)
- Boost x1.2 sur ~30 % des cas (alignement H4 trend) → score moyen 50.5
- Trades qui passeraient PREMIUM (≥ 75) : **0 actuellement → ~50** post-boost
- Trades **bloqués** (score chute < 25 par contradiction H4) : **~120** sur 1597 (-7.5 % volume)

**Net effet** :
- Volume signaux/jour : -8 %
- Profit factor escompté : +12 % (1.086 → ~1.22) si l'effet se confirme
- Tier PREMIUM enfin populé (cf. memory `confluence_calibration.md`)

**Risque** : si la rule H4 est biaisée (déjà BOS-forward-leaking),
le boost amplifie le bruit. À tester en walk-forward strict avant ship.

---

## M9 — TF Tier Strategist

| Timeframe | Bars/jour XAU | Compute cost (LLM calls/j) | Tier proposé | Justification valeur perçue |
|-----------|---------------|----------------------------|--------------|----------------------------|
| M1        | 1440          | ~50/j (gating algo) | **INSTITUTIONAL** | scalping, latence sensible |
| M5        | 288           | ~25/j | **STRATEGIST+** | day-trading actif |
| M15       | 96            | ~12/j | **ANALYST+** | sweet spot retail |
| H1        | 24            | ~3/j  | **ANALYST+** | swing intraday |
| H4        | 6             | ~1/j  | **FREE teaser** | swing low-volume, valeur démo |
| D1        | 1             | ~0.3/j | **FREE teaser** | position trading |
| W1        | 0.2           | ~0.1/j | **FREE teaser** | macro context |

### Justification computationnelle
- M1 sur 6 symbols × 24h = 8640 décisions/jour → **50× M15 load** → CPU
  prohibitif pour < INSTITUTIONAL.
- H4/D1 : volume signal très faible mais valeur perçue forte (un swing
  trade par semaine = "ça vaut le coup d'attendre"), idéal en hook FREE.

### Mapping commercial
- **FREE** : H4/D1/W1, 1 signal/j cap
- **ANALYST 19 USD** : + M15/H1, cap 5/j
- **STRATEGIST 49 USD** : + M5, illimité
- **INSTITUTIONAL 199 USD** : + M1 + webhooks

---

## M10 — Bundling Strategist

Cf. mockup détaillé : `mockups/pricing_bundles.md`.

### Récap 4 packs

| Bundle      | Symbols                    | Prix | Justification |
|-------------|----------------------------|------|---------------|
| FX Pack     | EUR + GBP + JPY + AUD      | 29 USD | Bench ForexSignals 47, FXLeaders 30 |
| Metal Pack  | XAU + XAG + USOIL          | 39 USD | XAU asset signature, USOIL P1 add |
| Crypto Pack | BTC + ETH + SOL            | 49 USD | 24/7 = compute 2x, no discount |
| Index Pack  | US500 + NAS100 + DAX       | 35 USD | Bench Trade Ideas 84 (overcovered) |

### Risque cannibalisation tier all-access
- **HIGH** : ANALYST 19 USD < FX Pack 29 USD. Mitigation : ANALYST cap
  5 signaux/j ; FX Pack illimité.
- **MEDIUM** : Crypto Pack 49 = STRATEGIST 49. Mitigation : STRATEGIST
  inclut M5, Crypto Pack non.
- **LOW** : Metal/Index packs offrent une économie claire vs all-access.

### MRR projection
Hypothèse acquisition 100 utilisateurs en 90j :
- 30 % FX Pack × 29 = 870 USD
- 25 % Metal Pack × 39 = 975 USD
- 15 % Crypto Pack × 49 = 735 USD
- 10 % Index Pack × 35 = 350 USD
- 20 % all-access STRATEGIST × 49 = 980 USD
**Total : 3 910 USD MRR** vs 49 USD × 100 = 4 900 USD all-access pur.
**MRR -20 % court terme** mais activation +40 % (bundles plus accessible),
**LTV +25 %** par segmentation.

---

## M11 — Red-Team

### Q1 : Le sweep 30-cells est-il OOS ou in-sample ?
- **In-sample par défaut** dans `eval_20_sweep_runner.py`.
- Mode `--walk-forward` dispo mais **OOS = derniers 15 %**, pas un vrai
  rolling Sprint 18. Les chiffres affichés à un prospect doivent passer
  par `--walk-forward` minimum.
- **Risque** : les PF in-sample sur 6 ans seront trop flatteurs (~1.2-1.5)
  vs OOS réel ~0.9-1.1.

### Q2 : Onboarder USOIL avant que XAU soit profitable est prématuré ?
- XAU baseline PF 1.086 = **marginalement profitable mais pas
  commercialement défendable** (target ≥ 1.3 cf. config QUALITY_GATES).
- Onboarder USOIL maintenant = multiplier la dette technique (data
  pipeline, news calendar, presets) **sans avoir une stratégie qui marche**.
- **Recommandation pragmatique** : USOIL onboarding peut commencer en
  **parallèle** du tuning XAU (différent owners, peu de conflits), mais
  **ne pas pricer le Metal Pack avant que XAU PF OOS > 1.2**.

### Q3 : Bundles Metal/Index ont-ils assez d'assets pour justifier prix vs all-access ?
- Metal Pack 39 (3 assets) vs all-access 49 (12+ assets) : **ratio
  3 USD/asset/mo** dans Metal vs **4 USD/asset/mo** all-access. Économie
  réelle 20 % seulement → marginal.
- Index Pack 35 (3 assets) vs all-access 49 : **11.7 USD/asset Pack** vs
  4 USD/asset all-access. **Pack plus cher par asset que all-access** →
  positionnement à clarifier (focus / curation, pas économies).
- **Mitigation** : positionner les packs comme « expert mode » (curation
  + plus de signaux par asset, pas plus d'assets), pas comme « budget ».

### Autres fragilités
- 5/6 presets n'ont jamais été testés sur de la donnée réelle. Les
  `session_hours` et `calendar_events` sont des conjectures.
- Asymétrie XAU long/short non testée sur autre asset → on ne sait pas
  si c'est un bug pipeline.
- `bars_per_day=28` pour US500 : aucun test pipeline, risque de bugs
  (warmup, cooldown, max_signal_age en bars sur asset RTH).

---

## M12 — Synthèse

### Recommandation keep / drop / add

| Asset | Verdict | Action | Délai |
|-------|---------|--------|-------|
| XAUUSD | **KEEP** | Tuning P1 jusqu'à PF OOS > 1.2 | 30 j |
| EURUSD | **KEEP cond.** | Onboard data Dukascopy + sweep | 14 j |
| GBPUSD | **PARK** | ρ +0.75 vs EUR — onboard si EUR PF > 1.2 | 60 j |
| USDJPY | **PARK** | onboard Q3 si business case | 90 j |
| BTCUSD | **DROP MVP** | trop coûteux LLM 24/7, ρ vs SPX > 0.5 | défer |
| US500  | **PARK** | RTH-only = pipeline non testé | 90 j |
| **+ USOIL** | **ADD P1** | preset + Dukascopy download + tests | 30 j |
| **+ NAS100** | **ADD P2** | post-USOIL + post-US500 | 60 j |

### Roadmap 90 jours

**Mois 1 (J1-J30)** :
- J1-J7 : onboarder EURUSD via Dukascopy (réutiliser
  `scripts/download_dukascopy_xau.py`).
- J8-J14 : sweep 5 cellules EURUSD M5/M15/H1/H4/D1.
- J15-J21 : test asymétrie long/short EURUSD vs XAU. Verdict bug vs biais.
- J22-J30 : tuning XAU pour PF OOS > 1.2 (cf. memory
  `confluence_calibration.md` — replace scoring fn).

**Mois 2 (J31-J60)** :
- J31-J45 : onboarder USOIL + preset + 5-cell sweep.
- J46-J60 : implémenter portfolio correlation cap (M5 spec) + multi-TF
  boost H4→M15 (M8 spec).

**Mois 3 (J61-J90)** :
- J61-J75 : Metal Pack landing page + Stripe + soft launch.
- J76-J90 : si XAU + EUR + USOIL ont PF OOS > 1.2 → FX Pack launch.

### KPIs cible 90 jours (revus)

| KPI Eval 20 | Cible originale | Cible révisée (réaliste) |
|-------------|-----------------|--------------------------|
| Symbols PF OOS > 1.2 | 4 | **2** (XAU + EUR) |
| Doublons ρ > 0.7 | 0 | 0 (cap portfolio) |
| Nouveaux assets PF > 1.0 | 2 | **1** (USOIL) |
| Bundles lancés | FX Pack | **Metal Pack soft launch** |
| MRR projection | +30 % | **+15 %** (bundles + 20 utilisateurs) |

### Note finale : **5.0 / 10**

**Forces (3.0)** :
- Architecture multi-asset propre, presets cleanly typed.
- `resample_ohlcv` + `InstrumentConfig` testés sur XAU.
- 4-pack pricing bien benché concurrent.

**Faiblesses (-5)** :
- 5/6 presets n'ont **jamais** vu de la donnée réelle (-2)
- XAU lui-même PF 1.086 < 1.2 cible commerciale (-1)
- `pip_value`, `weekend_behavior`, `news_relevance_weight` manquants (-1)
- Asymétrie long/short non investiguée multi-asset (-1)

---

## Annexes

- `reports/eval_20_sweep_30cells.csv` — template 30 cellules (1 ground-truth)
- `reports/eval_20_sweep_runner.py` — script PR-ready pour exécuter le sweep
- `mockups/pricing_bundles.md` — mockup pricing page 4 packs
