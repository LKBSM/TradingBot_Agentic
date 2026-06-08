# Plan de transformation institutionnelle — Smart Sentinel AI

**Date** : 2026-05-13
**Auteur** : Quant senior review (4 agents parallèles : sell-side, buy-side, cartographie code, revue académique)
**Statut** : DRAFT — à valider avant Sprint Phase 2B
**Contexte décisif** : A1 verdict 2026-05-01 (DSR=0.000, PBO=0.50) confirme que P(edge prédictif ML pur sur M15 OHLCV XAU) est faible structurellement. Le projet pivote vers narrative-first.

---

## 0. TL;DR — 1 page

**Question posée** : quels algorithmes utilisent les banques et top hedge funds, et comment les transposer au projet ?

**Réponse honnête** :
- **~85% des algos sell-side/HFT sont INAPPLICABLES** (Avellaneda-Stoikov, OFI L2/L3, queue imbalance, latency arb, Almgren-Chriss) faute de Level 2/3 LOB, FIX feed, co-location, et taille de book.
- **~70% des techniques ML buy-side sont du HYPE** sur ce setup (TSFM Chronos/Moirai/TimesFM/Lag-Llama avec data-leakage documenté 47-184%, Mamba pour finance non-replicated, deep RL).
- **L'A1 a déjà testé la stack "Renaissance-like"** (LightGBM 2 niveaux + 19 features + CPCV 28 paths) et c'est précisément cette approche qui a échoué (DSR=0.000). La méthodologie n'est pas en cause — c'est le signal/bruit M15 retail mono-actif.

**Les 3 piliers institutionnels RÉELLEMENT transférables**, ordonnés par P(succès)×ROI :

| # | Pilier | P(PF > 1.20 OOS) | Effort | Référence académique |
|---|---|---|---|---|
| 1 | **Event-driven macro** (NFP/CPI/FOMC ±30min) | 50-60% | 80h | Andersen-Bollerslev-Diebold (2003, 2007), Faust-Wright (2018) |
| 2 | **Conformal Prediction wrapper** + reject-option sur SMC existant | 40-50% | 30-50h | Angelopoulos-Bates (2024), Gibbs-Candès (2021) |
| 3 | **HAR-RV + bipower jumps + BOCPD régime gate** | 30-40% | 50-70h | Corsi (2009), Barndorff-Nielsen (2004), Adams-MacKay (2007), Tsaknaki-Lillo (2024) |

**Quick wins risk layer (à faire en parallèle, 20-30h)** : Fractional Kelly + CVaR + drawdown cap + lead-lag cross-asset features (XAU vs DXY/UST10Y/SPX).

**Anti-recommandations dures** :
- NE PAS implémenter Avellaneda-Stoikov / OFI / queue imbalance (pas de LOB → cargo-culting).
- NE PAS investir dev sur TSFM (Chronos/TimesFM/Moirai/Kronos) — data leakage 47-184% documenté arxiv:2511.18578.
- NE PAS empiler de nouveaux indicateurs sans gate CPCV+DSR+PBO obligatoire (sinon factor zoo → PBO 0.5 garanti).

---

## 1. Diagnostic actuel — ce qui existe déjà

Source : agent Explore complet sur src/ (2026-05-13).

### 1.1 Maturité par sous-système

| Aspect | Maturité | Détail |
|---|---|---|
| Data pipeline (OHLCV CSV/MT5) | 5/5 | `src/intelligence/data_providers.py:51-147` |
| Rule-based signals (BOS/CHOCH/FVG/OB) | 5/5 | `src/environment/strategy_features.py:597-850` (Numba JIT 50-100×) |
| Volatility forecasting (HAR-RV / HMM 3-state / TCP sketch / Hybrid) | 4/5 | `src/intelligence/volatility_forecaster.py:468-1500` — pas de daily refit |
| Regime detection (ADX rule-based + HMM predictor) | 3.5/5 | `src/agents/market_regime_agent.py`, `src/agents/regime_predictor.py` |
| News/macro integration | 2.5/5 | ForexFactory CSV, FinBERT optionnel — pas de live feed |
| Confluence scoring (7 components weighted) | 4/5 | `src/intelligence/confluence_detector.py:136-627` — poids fixes |
| SignalStateMachine (hysteresis + cooldown) | 5/5 | `src/intelligence/signal_state_machine.py:298-720` |
| Walk-forward validation (CPCV + DSR + PBO) | 4.5/5 | `src/research/cpcv_harness.py:1-100`, 28 paths |
| Kelly sizing | 2/5 | `src/environment/risk_manager.py:1-100` — floor conditionnel, pas d'optim dynamique |
| Production delivery | 4/5 | Telegram + Discord + Webhook async |

**Verdict global** : **3.8/5**. Core robuste, faiblesses sur Kelly/sizing, macro live, online learning, et features cross-asset.

### 1.2 Gaps institutionnels identifiés (à combler)

1. Kelly fractionnaire sans calibration dynamique (cf. MacLean-Thorp-Ziemba 2011).
2. Pas de Conformal Prediction complet (TCP sketch existe, jamais wrappé en reject-option).
3. Pas de DoubleML / orthogonal ML (Chernozhukov 2018) pour mesurer effet causal d'événements macro.
4. Pas de cross-asset features (XAU vs DXY, vs UST10Y real-yields, vs SPX, vs VIX) → composant lead-lag manquant.
5. Pas de regime-aware feature normalization (RSI/MACD bruts utilisés sans Z-score par régime vol).
6. Pas de transaction cost stochastic (ATR-prop fixe, pas de bid-ask spread aléatoire).
7. Pas de online learning / non-stationarity tracking (HMM/GARCH batch-fit une fois).
8. Pas de CVaR-constrained sizing / drawdown cap (Magdon-Ismail 2004).
9. Pas de BOCPD régime (HMM seul lag 5-10 bars).
10. Pas de portfolio-level correlation tracking (multi-symbol existe mais pas de hedge sizing).

---

## 2. Algos sell-side / market makers — verdict transposabilité

Source : agent recherche sell-side. Pour chaque algo : (a) nom, (b) référence, (c) data requirements, (d) verdict.

### 2.1 Market Making
| Algo | Référence | Data requise | Verdict projet |
|---|---|---|---|
| Avellaneda-Stoikov | Avellaneda & Stoikov 2008 | LOB + inventaire temps réel | ❌ Pas de LOB. Sentinel n'est pas MM. |
| Glosten-Milgrom | 1985 | Trades signés informés | ❌ Cadre théorique. |
| Cartea-Jaimungal | 2015+ (Cambridge book + SIAM 2017) | LOB + Hawkes intensités | ❌ Stochastic control sur LOB. |
| ISAC (RL + Stoikov) | im1235/ISAC 2023 | Simulateur LOB | ❌ |

**Cargo-culting alert** : implémenter A-S "pour faire sérieux" est précisément le piège. Sans LOB c'est inutile.

### 2.2 Execution Algos
| Algo | Référence | Verdict retail XAU M15 |
|---|---|---|
| VWAP/TWAP/POV | Berkowitz 1988, Konishi 2002 | ⚠️ Utile seulement si Sprint "exécution broker" ajouté |
| Almgren-Chriss IS | 2000 | ❌ Pertinent pour blocs >0.1% ADV, pas trade retail |
| Arrival Price | Perold 1988 | ✅ Comme **benchmark de slippage**, pas comme algo |

### 2.3 Microstructure HFT
| Signal | Référence | Verdict OHLCV M15 |
|---|---|---|
| OFI (Order Flow Imbalance) | Cont-Kukanov-Stoikov 2014, arXiv:2408.03594 (2024) | ❌ LOB requis. Proxy tick-rule M15 bruité, ROI marginal |
| Queue Imbalance | Gould-Bonart 2016 | ❌ LOB |
| **Kyle's Lambda** | Kyle 1985, Hasbrouck 2009 | ✅ Estimable depuis OHLCV : `λ = slope(r ~ sign(r)·sqrt(VolUSD))`. **Feature illiquidité utile** |
| **VPIN (Bulk Volume Class.)** | Easley-LdP-O'Hara 2012 | ⚠️ Approximable par BVC sur M15. Feature toxicité régime |
| Hasbrouck Info Share | 1995 | ❌ Multi-venue inaccessible XAU OTC |
| Tick-rule Lee-Ready | 1991 | ✅ Proxy signed volume utilisable |

### 2.4 Stat Arb intra-day
| Stratégie | Verdict |
|---|---|
| Pair trading cointégration (Engle-Granger, Johansen) | ⚠️ XAU vs XAG/GDX/DXY possible, edge arbitré |
| **Lead-lag inter-asset** (Hayashi-Yoshida 2005) | ✅ **MEILLEUR créneau retail** : XAU lagging DXY/UST10Y/SPX/VIX. À M15 l'arb institutionnel est moins serré |
| Futures-spot basis (GC-XAU) | ⚠️ Spread négligeable retail |
| ETF arb (GLD/IAU) | ❌ Inaccessible (AP only) |
| Latency arb cross-venue | ❌ Impossible retail |

**Synthèse sell-side : 10-15% transposable, et UNIQUEMENT comme features de confluence** :
- ✅ Kyle's Lambda (illiquidité feature)
- ⚠️ VPIN-BVC (toxicité régime feature)
- ✅ Lead-lag cross-asset (XAU vs DXY/UST/SPX)
- ✅ Arrival Price benchmark slippage

---

## 3. Algos buy-side hedge funds — verdict transposabilité

Source : agent recherche buy-side (Renaissance, Two Sigma, DE Shaw, Citadel, AQR, Man AHL, Winton, Bridgewater).

### 3.1 Statistical Arbitrage
- Avellaneda-Lee PCA stat arb (NYU 2010) : Sharpe 1.44 (1997-2007) → **0.54 post-2013** par over-crowding. ❌ standalone, ⚠️ comme feature (z-score XAU/DXY, XAU/silver).
- Cointégration Engle-Granger / Johansen : pair-trading valid cross-asset, **inapplicable intra-XAU seul**.
- OU mean reversion : couche timing intra-régime seulement.

### 3.2 Factor Models pour XAU/commodités
Pas Fama-French equity. Le cadre pertinent :
- **Lustig-Stathopoulos-Verdelhan 2019 (AER)** : carry term-structure FX
- **Brennan-Schwartz / Schwartz 1997** : commodity term structure
- **Erb-Harvey 2006, Bhardwaj 2015** : carry/momentum/value commos
- XAU-specific : **TIPS 10Y real-rates, DXY, GLD/IAU flows, central-bank purchases (WGC), CFTC COT positioning**

✅ **GO** : construire panel 6-8 facteurs macro-XAU comme features narrative.

### 3.3 ML Finance 2024-2026 — réalité vs hype

| Technique | Statut empirique |
|---|---|
| **GBM + Purged/Embargo CV + CPCV** (LdP 2018) | ✅ Standard académique. A1 a appliqué, verdict honnête : pas d'edge sur OHLCV M15 pur. |
| **TSFM** Chronos/Moirai/Lag-Llama/TimesFM/Kronos | ❌ **Data leakage 47-184% documenté** (arxiv:2510.13654, 2511.18578). HAR/LGBM/ARIMA matchent ou battent TSFM sur 2/3 tâches. |
| **PatchTST / TimeMixer / TSMixer** | ⚠️ Aucune preuve gold M15. |
| **NGBoost / Conformal Prediction** (Gibbs-Candès 2021, Angelopoulos-Bates 2024) | ✅ Validé pour incertitude. Variantes online (ACI, EnbPI) pour série temporelle. |
| **DoubleML / CausalForestDML** (Chernozhukov 2018, EconML 0.16) | ✅ Mature. Useable pour **mesurer effet causal NFP/CPI/FOMC sur return XAU**. |
| **PyMC / Stan HMM régime** | ✅ Solide mais lent (déjà chez vous via HAR-HMM). |
| **BOCPD** (Adams-MacKay 2007, Tsaknaki-Lillo 2024) | ✅ Supérieur HMM pour shifts brusques (FOMC, war shocks). |
| **Mamba / SSM finance** (2024-2025) | ❌ Non-replicated robustly. |

### 3.4 Risk & sizing
- **Fractional Kelly (half-Kelly)** : −75% vol pour −25% growth (MacLean-Thorp-Ziemba 2011) ✅
- **HRP** (LdP 2016, Antonov-Lipton-LdP 2024) : ⚠️ étude 2025 montre 1/N le bat sur certains setups → prudence
- **CVaR Rockafellar-Uryasev 2000** : ✅ standard moderne
- **Magdon-Ismail drawdown-controlled Kelly** : ✅ formule `f* = (μ-r)/(σ²·(1+D_max))`

### 3.5 Régime detection
- HMM 3-state : ✅ existant
- BOCPD score-driven : ✅ extension naturelle (Tsaknaki 2024 Quant Finance T&F)
- HAR-RV + bipower jumps (Barndorff-Nielsen-Shephard 2004, Al Rababaa 2025) : ✅ amélioration validée

### 3.6 NLP/News
- FinBERT (Yang 2020) : ⚠️ **battu par Logistic Regression TF-IDF** (Singh 2024 NGX, 81.83% vs FinBERT)
- GPT-4o / Claude pour event classification : ✅ pertinent, attention data-leakage post-cutoff
- **RAG embeddings (E5, BGE, voyage-finance-2)** pour event-matching historique : ✅ **CLAIR GO** — aligné narrative-first

---

## 4. Les 3 piliers institutionnels prioritaires (plan détaillé)

### Pilier 1 — Event-Driven Macro (XAU autour de NFP/CPI/FOMC/ECB) ★ Priorité #1

**Référence académique** :
- Andersen, Bollerslev, Diebold (2003 AER, 2007 ReStud) : jumps significatifs ±30min post-release sur FX/Gold
- Faust & Wright (2018) : surprise macro index → response systematique
- Andersen, Bollerslev, Diebold, Vega (2003 JIE) : "Micro effects of macro announcements"

**Edge crédible** : Sharpe 0.8-1.4 net après costs, ~40-60 trades/an, P(succès) 50-60%.

**Setup proposé** :
1. **Universe events** : NFP, CPI, Core CPI, PCE, FOMC decision + minutes, Fed Chair speeches, ECB rate, US Retail Sales, ISM Manufacturing, Powell pressers.
2. **Fenêtres** : T-5min (entry preparation) → T (release) → T+30min (event window) → T+2h (post-event drift).
3. **Signal** : volatility breakout straddle synthétique (long ATR-based) avec direction décidée par :
   - Surprise direction (actual vs consensus Bloomberg/Reuters) — feature primaire
   - Pre-event positioning (CFTC COT delta last week)
   - News LLM sentiment (Claude classifier sur headline + first paragraph)
4. **Risk** : taille ½ Kelly conditionnel à surprise magnitude, stop = 1.5 ATR(M15) au release, target = 3 ATR ou T+2h close.
5. **Validation gate** : CPCV 28 paths sur 6 ans, PF > 1.30 IS, PF lo > 1.05 CI 95%, DSR > 2.0, PBO < 0.30, Diebold-Mariano p<0.01 vs constant baseline. **Tout résultat sub-gate → abandon, pas de cherry-picking.**

**Effort** : 80h (Aisha lead, infra news déjà en place via `src/agents/news/economic_calendar.py`).

**Livrables** :
- `src/strategies/event_driven_macro.py` : module isolé, prend output `EconomicCalendarFetcher` + `news_analysis_agent`
- `scripts/eval_event_driven_macro.py` : sweep + CPCV harness
- `reports/sprint_event_driven_macro_results.md` : verdict GO/NO-GO chiffré

**Gate de décision** : si Sharpe net OOS < 0.5 ou PF lo < 1.05 après CPCV, → pivot Pilier 2/3 ou pivot B2B-API (decision_matrix_2026_04_30).

---

### Pilier 2 — Conformal Prediction Wrapper + Reject-Option ★ Priorité #2

**Référence académique** :
- Angelopoulos & Bates (2024) "Theoretical Foundations of Conformal Prediction" arxiv:2411.11824
- Gibbs & Candès (2021) "Adaptive Conformal Inference" arxiv:2106.00170
- Kato (2024) "Conformal Predictive Portfolio Selection" arxiv:2410.16333
- "Conformal Risk Control" ICLR 2024

**Edge crédible** : ne crée pas de nouveau signal, mais **transforme SMC actuel (PF 0.94) en stratégie sélective**. P(succès) 40-50%.

**Setup proposé** :
1. Wrapper sur confluence score 0-100 actuel : pour chaque candidat trade, calculer interval conforme P95 de `P(target_hit | features, regime)`.
2. **Reject-option** : si lower-bound interval < seuil (ex: 0.55), ne PAS trader.
3. **Variants** :
   - Split Conformal (offline calibration set 30% data)
   - Adaptive Conformal Inference (ACI, Gibbs-Candès) pour non-stationarity
   - EnbPI (Ensemble Prediction Interval) pour bootstrap robustness
4. **Calibration set** : trades historiques replay state machine 2019-2023, test set 2024-2025.

**Effort** : 30-50h.

**Livrables** :
- `src/intelligence/conformal_wrapper.py` : ConformalScorer class avec `predict_interval()` et `should_reject()`
- Integration dans `SignalStateMachine` comme gate supplémentaire après hysteresis
- `tests/test_conformal_wrapper.py` + replay comparatif PF avant/après wrapper

**Gate** : amélioration absolue PF (after - before) > 0.15 sur OOS test, et nombre de trades >= 50% du baseline (sinon trop sélectif).

---

### Pilier 3 — HAR-RV + Bipower Jumps + BOCPD Régime Gate ★ Priorité #3

**Référence académique** :
- Corsi (2009) "A Simple Approximate Long-Memory Model of Realized Volatility" JoFEM
- Barndorff-Nielsen & Shephard (2004 JoFEM) : bipower variation pour décomposition continuous/jumps
- Adams & MacKay (2007) "Bayesian Online Changepoint Detection" arxiv:0710.3742
- Tsaknaki, Lillo, Mazzarisi (2024) "Online score-driven CPD" Quant Finance T&F
- Al Rababaa (2025) : HAR + asymmetric jumps + spillovers

**Edge crédible** : pas predictor direct mais **filtre de régime temporel** (gate les signaux en régime jump-heavy / régime-shift). P(succès) 30-40%.

**Setup proposé** :
1. Remplacer `VOL_MODE=har` simple par **HAR-RV-J** (Corsi 2010 extension avec jump component).
2. Calculer bipower variation `BV_t` et jumps `J_t = max(0, RV_t - BV_t)` sur M5 base, agrégé M15.
3. **BOCPD layer** : online changepoint detection sur returns + RV → probabilité de régime shift à chaque bar.
4. **Régime gate** : si P(regime shift) > 0.40 dans les 5 prochains bars → bloquer nouveaux trades (juste exits).
5. **HMM existant garde**, BOCPD ajoute layer reactive (HMM lag 5-10 bars, BOCPD lag ~1-2 bars).

**Effort** : 50-70h.

**Livrables** :
- `src/intelligence/volatility_forecaster.py` extension : `HAR_RV_J` mode + bipower variation calc
- `src/intelligence/bocpd_regime.py` : Adams-MacKay BOCPD implementation (PyMC ou custom)
- Integration `ConfluenceDetector` : régime_score component poids dynamique selon BOCPD posterior
- `tests/test_bocpd_regime.py` + comparatif IS/OOS

**Gate** : réduction drawdown >20% à PF baseline équivalent, ou amélioration Sharpe >0.3.

---

## 5. Quick wins risk layer (parallèle, 20-30h)

### 5.1 Fractional Kelly + CVaR + drawdown cap
**Référence** : MacLean-Thorp-Ziemba 2011, Rockafellar-Uryasev 2000, Magdon-Ismail-Atiya 2004.

**Implémentation** :
1. `src/intelligence/sizing.py` (NOUVEAU) :
   - `kelly_fraction(win_rate, rr_ratio, fraction=0.5)` → half-Kelly base
   - `cvar_constraint(returns, alpha=0.05, target_cvar=-0.03)` → clamp si CVaR < target
   - `drawdown_cap(current_dd, max_dd=0.10)` → réduit size linéaire si dd>5%, kill si dd>10%
2. Final size = `min(half_kelly, cvar_max, dd_cap)`
3. Refit weekly via in-sample win-rate / RR du SignalStateMachine replay.

### 5.2 Cross-asset lead-lag features (Hayashi-Yoshida)
**Référence** : Hayashi-Yoshida (2005), de Jong-Nijman (1997).

**Implémentation** :
1. Ajouter feeds dans `data/macro/` : DXY M15, UST10Y_TIPS daily, SPX M15, VIX M15 (déjà partiellement présents `data/macro/fred_*.csv`).
2. `src/features/cross_asset.py` (NOUVEAU) : compute rolling cross-correlation à plusieurs lags (1, 3, 5, 10 bars M15) pour XAU vs {DXY, UST10Y, SPX, VIX}.
3. Features dérivées : `XAU_DXY_DIVERGENCE_5BAR`, `XAU_LEADING_UST_3BAR`, `RISK_OFF_REGIME` (composite VIX↑ + SPX↓ + XAU↑).
4. Inject dans `ConfluenceDetector` comme component (poids initial 10%, à recalibrer).

### 5.3 Kyle's Lambda + VPIN-BVC comme features de régime
**Référence** : Kyle (1985), Easley-LdP-O'Hara (2012).

**Implémentation** :
1. `src/features/microstructure_proxy.py` (NOUVEAU)
2. Kyle's λ rolling : `regress |r_t| ~ sign(r_t) * sqrt(volume_USD_t)` sur fenêtre 50 bars.
3. VPIN via Bulk Volume Classification (LdP method) sur buckets volume M15.
4. Inject comme **features de régime**, pas de prédicteur directionnel.

---

## 6. Plan d'exécution par sprints (6 mois)

### Sprint 0 (Semaine 1, 10h) — Guards & gates
- [ ] Hardcoder CPCV+DSR+PBO comme gate obligatoire dans CI pour tout nouveau signal (`tests/test_strategy_gates.py`)
- [ ] Bloquer merge si `DSR < 1.5 || PBO > 0.35 || PF_lo < 1.0`
- [ ] Documenter dans CLAUDE.md : "no new feature/strategy without CPCV gate"

### Sprint 1 (Semaines 2-5, 80h) — Pilier 1 Event-Driven Macro
- [ ] S1.1 (20h) : EventWindowDetector autour FF calendar (T-5/T+30/T+2h)
- [ ] S1.2 (20h) : Surprise score calculator (actual vs consensus, normalized)
- [ ] S1.3 (20h) : Event strategy backtest harness + CPCV
- [ ] S1.4 (20h) : LLM event classifier (Claude Haiku, surprise direction + sentiment)
- [ ] **Gate Sprint 1** : Sharpe net OOS > 0.5, PF lo > 1.05 → GO Sprint 2. Sinon → Sprint 4 directement.

### Sprint 2 (Semaines 6-7, 40h) — Pilier 2 Conformal Wrapper
- [ ] S2.1 (15h) : ConformalScorer class (Split Conformal baseline)
- [ ] S2.2 (15h) : ACI variant pour adaptive
- [ ] S2.3 (10h) : Integration SignalStateMachine + replay comparatif
- [ ] **Gate Sprint 2** : ΔPF > +0.15 OOS, trades ≥ 50% baseline.

### Sprint 3 (Semaines 8-10, 60h) — Pilier 3 HAR-J + BOCPD
- [ ] S3.1 (20h) : HAR-RV-J avec bipower jumps
- [ ] S3.2 (25h) : BOCPD Adams-MacKay (PyMC ou custom Python)
- [ ] S3.3 (15h) : Integration ConfluenceDetector régime gate
- [ ] **Gate Sprint 3** : ΔDrawdown > −20% OU ΔSharpe > +0.3.

### Sprint 4 (Semaines 11-12, 30h) — Risk layer quick wins
- [ ] S4.1 (10h) : Fractional Kelly + CVaR + drawdown cap
- [ ] S4.2 (15h) : Cross-asset lead-lag features
- [ ] S4.3 (5h) : Kyle's λ + VPIN-BVC features de régime

### Sprint 5 (Semaine 13-14, 30h) — Synthèse + production
- [ ] S5.1 (10h) : Full pipeline integration + smoke tests
- [ ] S5.2 (10h) : OOS sur 2025+ unseen (2026 H1)
- [ ] S5.3 (10h) : Doc + déploiement Railway

**Total effort estimé : 250h** sur 14 semaines (~3.5 mois).

**Gate global mi-parcours (fin Sprint 3)** : si aucun pilier n'a passé son gate, **trigger pivot B2B-API** (decision_matrix_2026_04_30).

---

## 7. Anti-recommandations explicites (à NE PAS faire)

| ❌ Ne pas | Pourquoi |
|---|---|
| Implémenter Avellaneda-Stoikov / market making | Pas de LOB, Sentinel n'est pas MM. Cargo-culting. |
| Investir dev sur TSFM (Chronos/TimesFM/Moirai/Lag-Llama/Kronos) | Data leakage 47-184% documenté arxiv:2511.18578. Specialistes classiques (HAR/LGBM/ARIMA) matchent ou battent. |
| Refaire stack ML "Renaissance-style" sur OHLCV M15 pur | A1 vient de prouver : DSR=0, PBO=0.5. P(edge net) < 5% post-costs. |
| Ajouter indicateurs/features sans CPCV+DSR+PBO gate | Factor zoo (Harvey-Liu) → t-stat seuil réel ~3.0, pas 2.0. PBO 0.5 garanti sinon. |
| Cascade Haiku→Sonnet→Opus LLM | Déjà tranché (eval_05_llm_implementation) : tier-routing simple suffit. |
| FinBERT brut sur news | Logistic regression TF-IDF le bat (Singh 2024 NGX 81.83%). |
| Mamba / SSM finance | Non-replicated robustly. |
| HRP standalone | 1/N le bat parfois (étude 2025). Utiliser seulement si portfolio multi-asset >5 actifs. |
| OFI / Queue imbalance sur OHLCV | LOB requis. Proxy tick-rule M15 trop bruité, ROI marginal. |
| Almgren-Chriss exécution | Block size retail invisible au marché. |

---

## 8. Décision finale et conditions

### Critères GO (continuer Phase 2B narrative-first + 3 piliers)
- Au moins **1 pilier sur 3** passe son gate avec Sharpe net OOS > 0.5 ET PF lo > 1.05 CI 95%.
- Compliance W4 légal validée d'ici fin Sprint 3.
- ANTHROPIC_API_KEY costs < 5% MRR projeté.

### Critères PIVOT B2B-API (decision_matrix_2026_04_30)
- 0/3 piliers passent gate après 14 semaines.
- OU : un pilier passe mais retention payée < 25% à M+2 après live trial.
- Pivot acté : cible $310k ARR via IC Markets, Exness, Pepperstone (80h dev MVP).

### Critères STOP complet
- Compliance bloquante non résolvable (US ban + EU MiFID II tightening 2026-03 sur "finfluencer").
- Coûts LLM > 30% MRR.

---

## 9. Sources & références (consolidées)

### Sell-side / HFT
- Avellaneda & Stoikov (2008) "High-frequency trading in a limit order book"
- Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
- Cont, Kukanov, Stoikov (2014) "OFI in HFT"
- Kyle (1985) ; Hasbrouck (2009) — Kyle's Lambda
- Easley, López de Prado, O'Hara (2012) — VPIN
- Hayashi & Yoshida (2005) — Lead-lag estimator

### Buy-side / hedge funds
- Bailey & López de Prado (2025) "How to Use the Sharpe Ratio" SSRN 5520741
- López de Prado (2018) "Advances in Financial Machine Learning" + CPCV/PBO
- Harvey, Liu, Zhu (2016) ; Harvey & Liu (2019) — Factor zoo
- Andersen, Bollerslev, Diebold (2003, 2007) — Macro announcements
- Faust & Wright (2018) — Surprise index
- Lustig-Stathopoulos-Verdelhan (2019 AER) — FX carry term structure
- Chernozhukov et al. (2018) — DoubleML
- MacLean-Thorp-Ziemba (2011) — Fractional Kelly
- Rockafellar-Uryasev (2000) — CVaR optimization

### ML / forecasting
- Angelopoulos & Bates (2024) "Foundations of Conformal Prediction" arxiv:2411.11824
- Gibbs & Candès (2021) "Adaptive Conformal Inference" arxiv:2106.00170
- Kato (2024) "Conformal Predictive Portfolio Selection" arxiv:2410.16333
- Adams & MacKay (2007) "Bayesian Online Changepoint Detection" arxiv:0710.3742
- Tsaknaki, Lillo, Mazzarisi (2024) — Score-driven CPD, Quant Finance
- Corsi (2009) — HAR-RV
- Barndorff-Nielsen & Shephard (2004) — Bipower variation
- Singh et al. (2024) — FinBERT vs LogReg arxiv:2412.06837
- Aksu (2024) — TSFM data leakage finance arxiv:2511.18578
- "Backtest overfitting in ML era" DSS 2024 — S0950705124011110

### Anti-references (hype documenté)
- Chronos (Amazon 2024) — TSFM
- TimesFM (Google 2024) — TSFM
- Moirai 2.0 (Salesforce 2024) arxiv:2511.11698
- Lag-Llama (2024)
- Kronos (2025) arxiv:2508.02739
- Mamba-2 / S-Mamba pour finance — non-replicated

---

## 10. Mise à jour CLAUDE.md / MEMORY proposée

Ajouter dans `MEMORY.md` :
```
- [Institutional Quant Transformation Plan 2026-05-13](institutional_quant_plan_2026_05_13.md) — 3 piliers (event-driven macro / conformal wrapper / HAR-J+BOCPD), 250h, gates CPCV+DSR+PBO obligatoires. Anti-reco : pas TSFM, pas Avellaneda, pas refaire A1. Cf. `reports/institutional_quant_transformation_plan.md`.
```

Et dans `CLAUDE.md` (à créer si absent) :
```
## Quant strategy gates (OBLIGATOIRE)
Avant tout merge d'une nouvelle stratégie ou feature signal :
- CPCV 28 paths minimum
- DSR > 1.5
- PBO < 0.35
- PF lo > 1.00 CI 95%
- Diebold-Mariano p<0.05 vs baseline constant
Sources : `reports/institutional_quant_transformation_plan.md` §0 et §7
```

---

**Fin du plan. Version 1.0, draft à valider.**
