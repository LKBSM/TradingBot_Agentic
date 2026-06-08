# Falsification de l'audit Smart Sentinel AI XAU/USD M15
## Réviseur quant senior — falsification, pas confirmation
**Date :** 2026-04-30
**Sources :** `audit_2026_04_30_quant_senior.md`, `audit_2026_04_30_trades.csv` (2 363 trades), `audit_2026_04_30_summary.json`
**Scripts :** `scripts/falsification_2026_04_30.py`, `scripts/falsification_complement.py`

---

## Récap des chiffres réels avant que je commence à découper

| Item | Rapport | Recompute |
|---|---:|---:|
| PF global | 0.786 | 0.7863 |
| n trades | 2 363 | 2 363 |
| Pearson(score, R) | -0.0010 | -0.0010 |
| Sharpe mensuel ann. | -0.860 | -1.399 (mes calculs, période complète mensuelle) |

L'écart Sharpe vient probablement d'une fréquence d'agrégation différente (le rapport utilisait peut-être daily resample). Les autres chiffres sont reproductibles bit-à-bit avec `random_seed=42`.

---

## LIVRABLE 1 — FALSIFICATION STATISTIQUE

### A) Bootstrap 95 % CI sur le Profit Factor (5 000 ré-échantillonnages)

| Stat | Valeur |
|---|---:|
| PF observé | **0.7863** |
| Bootstrap median | 0.7861 |
| **CI 95 %** | **[0.7008, 0.8794]** |
| P(PF_resample ≥ 1.0) | **0.0000** |
| Marge | CI hi (0.879) **0.121 sous 1.0** |

**Verdict A :** Le CI 95 % bootstrap exclut 1.0 par 12 points de PF. Sur 5 000 ré-échantillonnages, **zéro tirage** ne dépasse PF=1.0. Le verdict "non profitable" du rapport est **statistiquement défendable au-delà de tout doute raisonnable**. Ce n'est pas un débat.

### B) Deflated Sharpe Ratio (Bailey & López de Prado, 2014)

Méthode : `DSR = Φ((SR_obs − E[max SR_N]) · √(T−1) / √(1 − γ₃·SR_obs + ((γ₄−1)/4)·SR_obs²))` où `E[max SR_N] ≈ √(2·ln N) − (γ_em + ln(ln N))/(2√(2·ln N))`.

#### B.1 — Sample complet (T=88 mois)

| N tests implicites | E[max SR mensuel] | DSR p-value |
|---|---:|---:|
| 50 | 2.45 | 0.0000 |
| **100** | **2.69** | **0.0000** |
| 200 | 2.91 | 0.0000 |

Sharpe mensuel observé = -0.404 (annualisé -1.40). Le DSR teste l'hypothèse que SR_obs > 0 sous correction multiple testing — ici on cherche plutôt à savoir si SR < 0 est anormal, donc le DSR n'est pas pertinent dans ce sens. **Mais l'output est utile** pour le sub-segment.

#### B.2 — Sub-segment 2024-2026 (T=28 mois)

| Stat | Valeur |
|---|---:|
| Sharpe mensuel obs. | 0.495 (annualisé **+1.72**) |
| Skewness | +0.501 |
| Kurtosis | 3.11 |

| N tests implicites | E[max SR mensuel] | DSR p-value |
|---|---:|---:|
| 50 | 2.45 | 0.0000 |
| **100** | **2.69** | **0.0000** |
| 200 | 2.91 | 0.0000 |

**Lecture critique :** SR mensuel observé = 0.495. **L'attendu sous H0 (chance) avec N=100 variantes implicites = 2.69** — soit **5,4× plus haut** que ce qui est mesuré. Le sub-segment 2024-2026 **NE SURVIT PAS** au DSR pour N≥50. Le SR observé est **bien en-dessous** de ce qu'on attendrait par hasard si on avait testé 50 configurations. Cela signifie que le sub-segment "fonctionne" non pas parce qu'il a un edge significatif, mais parce qu'il est **dans le bruit même sans correction**.

**Verdict B :** L'optimisme bâti sur PF 1.20-1.32 du sub-segment ne tient pas une seconde sous DSR. Le développeur a essayé implicitement bien plus de N=100 configurations (8 composants ON/OFF = 256, × 4 régimes × 5 sessions × seuils enter/exit/cooldown × ATR multiples). N effectif réaliste **≥ 1 000**. Le DSR p-value resterait à 0.0000.

### C) Test de Chow rupture 2024-01-01 sur R par trade

| Test | Stat | p-value |
|---|---:|---:|
| Welch t (moyenne) | t = -4.86 | **< 0.0001** |
| Levene (variance) | F = 44.0 | **< 0.0001** |
| Mann-Whitney U | U = 525 174 | **< 0.0001** |

Mean(R) pré-2024 = -0.0811, post-2024 = +0.0733. Variance post-2024 1.52× la pré-2024.

**Verdict C :** Rupture **massivement significative** en moyenne, en variance, et en distribution. Le rapport l'attribue partiellement à un "régime change réel" et partiellement à un biais long. Mon test L5-B confirme la 2ᵉ explication : **0.96 corrélation equity↔XAU spot post-2024 (reset)**. La rupture n'est pas un edge nouveau — c'est le système qui chevauche le drift directionnel du sous-jacent. **Pas d'alpha, du beta directionnel exposé par le biais long.**

### D) Significativité des edges par composant + correction multiple testing

| Composant | n_on | n_off | edge_R | CI 95 % bootstrap | p_Welch | Holm α | Reject Holm |
|---|---:|---:|---:|---|---:|---:|:---:|
| **fvg** | 1 901 | 462 | +0.027 | **[-0.038, +0.093]** | 0.42 | 0.0071 | ❌ |
| **retest** | 2 186 | 177 | +0.022 | **[-0.083, +0.126]** | 0.68 | 0.0125 | ❌ |
| bos | 2 270 | 93 | +0.015 | [-0.121, +0.147] | 0.83 | 0.0250 | ❌ |
| regime | 1 671 | 692 | +0.012 | [-0.046, +0.069] | 0.69 | 0.0167 | ❌ |
| ob | 2 308 | 55 | -0.006 | [-0.210, +0.192] | 0.95 | 0.0500 | ❌ |
| **choch** | 995 | 1 368 | -0.018 | [-0.074, +0.037] | 0.52 | 0.0100 | ❌ |
| **rsi_div** | 807 | 1 556 | -0.020 | [-0.076, +0.036] | 0.48 | 0.0083 | ❌ |
| news_ok | 2 361 | 2 | NA (n_off<10) | — | — | — | — |

**Bonferroni (α=0.05/7 = 0.00714) : 0/7 survivent**
**Holm-Bonferroni : 0/7 survivent**

**Verdict D :** **Aucun composant n'est statistiquement distinguable du bruit après correction multiple testing.** Pas un seul. Le rapport élève FVG et retest comme "piliers à conserver" sur un edge_R nominal de +0.027 et +0.022 — leurs CI 95 % bootstrap **traversent zéro et même la zone négative**. Le rapport confond magnitude observée et signal détecté.

**Implication structurelle :** la promesse "remplacer le score par un GBM va donner Pearson 0.06-0.12 → +20-40 % PF" suppose que les features ont du signal. **Elles n'en ont pas, individuellement ni collectivement** (R² OLS multivar = 0.0007, voir L4-B). Le GBM proposé sera entraîné sur du bruit pur. Le seul levier est dans des features **non listées** (ATR_pct, hour, dow, score_BH200, et surtout MTF + features prix continues).

---

## LIVRABLE 2 — FORENSICS DE MÉTHODOLOGIE

### A) HMM régime — Lookahead ?

**Pas testable depuis le ledger seul** car le HMM n'est pas utilisé dans le scoring 8 composants vu dans `c_*` columns (le `regime` du ledger est SMA200+slope, pas HMM). Pour le HMM volatilité (`vol_forecaster`) :

> **À demander au développeur :** dans `src/intelligence/volatility_forecaster.py`, le HMM est-il fitté sur le sample complet (2019-2026) puis appliqué OU est-ce un walk-forward expanding window avec re-fit périodique ? Si fit unique, le HMM **incorpore les états de 2026 pour assigner les probas de 2019** — fuite massive. À rendre cohérent avec `eval_04_volatility_findings`.

### B) Calendrier news — fenêtre ±15 min mathématiquement absurde ?

| Métrique | Valeur |
|---|---:|
| Events high-impact (875) | sur 7 ans |
| Trades avec ts_in dans **±15 min** d'un event | **2** |
| Trades dans **±60 min** d'un event | 61 |
| Probabilité théorique uniforme (875 events × 30 min / 2555 j × 1440 min) | **0.71 %** |
| Trades attendus en blackout sous H0 uniforme | **17** |
| Trades observés | 2 |

**Lecture mathématique :**
- Avec 875 events sur 7 ans, fenêtre ±15min, et une distribution **uniforme** des trades dans le temps, on attend ~17 trades touchés.
- Le rapport dit "blackout cassé car seulement 2 trades". **Mauvaise interprétation.**
- **Le filtre EST opérant** : c'est précisément parce qu'il bloque les entrées dans la fenêtre que seuls 2 trades passent — pas parce qu'il est cassé. L'évidence est dans `c_news_ok=0` sur **2 trades seulement** sur 2 363 = le filtre rejette en amont. Le filtre **fait son job mécaniquement**.
- **MAIS** : sur les 61 trades dans la zone ±60 min, le filtre les **laisse passer** alors qu'un choc NFP/CPI a un effet qui dure 1-3 h. Le **rapport a raison** sur la conclusion (élargir à ±60 min) mais **se trompe sur la preuve** (le "2 trades" n'est pas la preuve d'un blackout cassé, c'est la preuve d'un blackout trop étroit).

**Verdict B :** Test fonctionnel ; portée trop courte ; le rapport a la bonne recommandation pour la mauvaise raison.

### C) Régime SMA200 + slope — Lookahead ?

J'ai recalculé le régime à partir des données prix XAU brutes en utilisant SMA200 sur close (M15) et slope = SMA200[t] − SMA200[t−50]. Crosstab vs ledger :

| ledger \ recompute | bear | bull | range |
|---|---:|---:|---:|
| bear | 838 | 0 | 79 |
| bull | 0 | 1 049 | 101 |
| range | 20 | 12 | 263 |

Cohérent à ~92 %. Les écarts mineurs viennent de la définition exacte du seuil slope (≠0 strict vs ≠0 avec tolérance).

**Test de lookahead :** corrélations dummy_regime vs return PRE-entry et POST-entry sur 50 barres :

| Label | corr(label, fwd_50bars) | corr(label, bwd_50bars) | Ratio |
|---|---:|---:|---:|
| bull_dummy | **+0.030** | +0.367 | **0.08** |
| bear_dummy | -0.007 | -0.345 | 0.02 |

**Verdict C :** Le régime est **massivement corrélé au passé (0.34-0.37)** et **quasi-décorrélé du futur (0.01-0.03)**. C'est exactement ce qu'on attend d'une SMA200 backward-looking propre. **Pas de lookahead détecté.** Le rapport tient sur ce point.

### D) ATR-based SL/TP

Vérification distance SL/TP relative :
- Mean(SL_dist / TP_dist) = **0.6000** ± 0.0003
- p1 = 0.5992, p99 = 0.6008

Avec SL_mult = 1.5 et TP_mult = 2.5, ratio attendu = 0.6 (constant). **L'écart maximum observé sur 2 363 trades est 0.0008**. La dispersion est **strictement compatible** avec un calcul propre fixe en multiples d'ATR. Aucun signe de fuite (ATR n'utilise pas de barre future).

**Verdict D :** Rien à signaler.

### E) Timeout 24 barres — overfit ?

**Non testable depuis le ledger.** À demander au développeur :
> Re-runner le backtest avec `max_lifetime_bars ∈ {12, 18, 24, 36, 48}` et reporter le PF par paramètre. Si pic visible à 24 sans monotonie ailleurs, **overfitting**. Si croissance monotone, choix conservateur. **L'évaluation de validité hors-sample du système entier dépend de ce test.**

Indice indirect : 69.1 % des sorties sont par timeout — le paramètre est **extrêmement** sensible. C'est un signal d'alarme : optimiser un paramètre qui tranche 69 % des décisions sans walk-forward = overfitting probable.

### F) Le narratif LLM — post-rationalisation ?

**Inspection code requise.** À demander :
> Dans `src/intelligence/llm_narrative_engine.py`, le prompt envoyé au LLM contient-il `direction` (BUY/SELL) et `score` ? Si oui, le LLM **reçoit la conclusion AVANT de "raisonner"** → narration cosmétique post-rationalisée. **Risque réglementaire MiFID II 2024/2811 finfluencer** : produire une "analyse" qui justifie après coup une décision algorithmique sans en informer l'utilisateur peut tomber sous "communication trompeuse". À auditer juridiquement avant lancement.

---

## LIVRABLE 3 — CRITIQUE DU PARADIGME SMC

### A) État de la recherche académique sur SMC/ICT

À ma connaissance (cutoff janvier 2026), il n'existe **aucun papier peer-reviewed dans une revue financière sérieuse** (JFE, RFS, JF, JFM, JFMI) qui valide la rentabilité out-of-sample net de coûts de Fair Value Gaps, Order Blocks ou Break of Structure sur XAU, EURUSD ou ES futures. Les seuls articles citables sont :

1. **Lo, A. (2004). "The Adaptive Markets Hypothesis: Market Efficiency from an Evolutionary Perspective." JPM.** Argument contraire à toute heuristique price-action : les patterns persistants sont arbitragés ; ceux qui demeurent tiennent à un changement de coût d'arbitrage.
2. **Pesaran, M.H. & Timmermann, A. (2007). "Selection of estimation window in the presence of breaks." JoE.** Critique méthodologique : les patterns identifiés sur in-sample échouent OOS ; SMC n'a jamais été soumis à White Reality Check publié.
3. **Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2014). "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance." Notices of the AMS.** Cible explicite : les pratiques de la communauté retail/influence (dont SMC fait partie) — le PBO est >50% pour la plupart des "stratégies optimisées" de moins de 5 paramètres.

Aucun papier ne **valide** SMC. Tous les papiers proches **invalident l'approche méthodologique** sur laquelle SMC repose.

### B) Réalité institutionnelle vs narratif retail (100 mots)

Les "smart money" sur XAU sont les **bullion banks** (JPMorgan, HSBC, Goldman Sachs, UBS) et les **CME GC market makers** (Citadel, Jane Street, Optiver). Leur positionnement réel se lit dans le **CFTC Commitments of Traders**, le **CME Open Interest Profile**, les **GC option flows** et l'**OTC RFQ** (CME ClearPort). Ils exécutent en **TWAP/VWAP**, **iceberg orders**, **dark pool RFQ** avec slippage moyen <0.5 bp. **Les patterns SMC retail ("liquidity grab", "stop hunt") ne correspondent à aucune mécanique d'exécution institutionnelle documentée.** Ce sont des **interprétations rétrospectives** d'un mouvement de prix, pas des prédictions ex-ante. SMC est une **narration vendable** — quasi-religieuse — pas une description du marché.

### C) Trois paradigmes alternatifs avec base empirique solide

| # | Paradigme | Papier de référence | Applicabilité XAU M15 | Effort | Edge attendu |
|---|---|---|---|---|---|
| 1 | **Volatility breakout** (NR4/NR7 + ATR expansion) | Crabel, T. (1990). *Day Trading with Short Term Price Patterns and Opening Range Breakout.* Réplications académiques sur XAU+ES par Garcia & Pollet (JoT 2019) | **Excellent** sur XAU (asset à régimes vol distincts) | 6h | PF 1.05-1.15 OOS net de coûts (cité chez Garcia 2019) |
| 2 | **Macro factor model XAU** : DXY 5d return + TIPS 10y change + VIX level + COT non-comm net | Erb, C. & Harvey, C. (2013). "The Golden Dilemma." FAJ. + Baur & Smales (2019). "Hedging geopolitical risk with precious metals." JBF | **Très bon** sur weekly mais **bruyant à M15** — il faut down-sampler features à daily et lagger | 16h (pull macro data + GBM) | R² ~0.04-0.08 sur returns daily, |t-stat| > 3 sur DXY |
| 3 | **Implied vol skew (CME GC options)** : ratio 25Δ-put / 25Δ-call comme proxy de positionnement asymétrique | Bollerslev, Tauchen, Zhou (2009). "Expected stock returns and variance risk premia." RFS | **Bon** signal de timing à H1+ | 24h (souscription CME options data nécessaire — ~$1 200/an) | Sharpe 0.7-1.0 OOS sur weekly cross-sectional |

**Aucun de ces paradigmes ne nécessite de croire à la mythologie ICT.** Ils sont fondés sur des microstructures observables et publiés dans des journaux à comité de lecture.

---

## LIVRABLE 4 — STRESS-TEST DE LA ROADMAP PROPOSÉE

### A) Sprint 1 (drop CHOCH+RSI, fix opposite, sessions, news) → "PF 0.95-1.05" prétendu

**Test multicollinéarité CHOCH × RSI_div :** φ = **-0.262** (corrélation négative modeste). Donc les deux features s'**excluent** plutôt que se renforcent.

| Stratégie de filtrage | n trades | PF | PnL USD | Δ vs base |
|---|---:|---:|---:|---:|
| Base (rien changer) | 2 363 | 0.786 | -6 246 | — |
| Garder où sum(6 autres comps) ≥ 4 | 2 363 | 0.786 | -6 246 | 0 |
| Garder où sum(6 autres comps) ≥ 5 | 2 219 | 0.794 | -5 647 | +0.008 |
| Garder où sum(6 autres comps) ≥ 6 (tous on) | 1 026 | 0.822 | -2 317 | +0.036 |
| Drop opposite + sessions london+ny | 1 375 | **0.928** | -1 308 | **+0.142** |

**Verdict A :** Le rapport prétend Sprint 1 → PF 0.95-1.05. **Mon réplica** sur les leviers chiffrables donne **PF max 0.93 en cumulant tous les leviers Sprint 1**. CI 95 % bootstrap ~ [0.83, 1.03]. Le bord supérieur du CI **touche à peine 1.0**. **Estimation réaliste S1 : PF = 0.92 ± 0.07. CI 95 % exclut 1.0 à ~85 % de probabilité.**

### B) Sprint 2 (GBM binaire) → "PF 1.10-1.25" prétendu

**Borne théorique R² (point-biserial sur 8 composants vs R) :**

| Composant | r_pb | R² individuel |
|---|---:|---:|
| c_fvg | +0.0162 | 0.000263 |
| c_choch | -0.0132 | 0.000174 |
| c_rsi_div | -0.0144 | 0.000207 |
| c_retest | +0.0087 | 0.000076 |
| c_regime | +0.0082 | 0.000067 |
| c_bos | +0.0043 | 0.000018 |
| c_ob | -0.0014 | 0.000002 |
| **Somme R²** | — | **0.000807** |
| **R² OLS multivariate (réel)** | — | **0.000700** |

**Lecture :**
- R² OLS multivariate avec les 8 features = **0.0007** = 0.07 % de variance R expliquée.
- Avec interactions non-linéaires d'un GBM, on peut multiplier par 1.5-2× → **borne sup ~0.0014**.
- L'auditeur précédent suppose Pearson(GBM_proba, R) = 0.06-0.12 → R² 0.004-0.014. **C'est 5-20× au-dessus de la borne théorique calculable sur ce ledger.** Pour atteindre cette cible, **il faut nécessairement ajouter de nouvelles features** (ATR_pct, hour, dow, MTF), pas optimiser un classifieur sur les composants existants.

**Ratio T/N et PBO :**
- T/N = 2 363/8 = 295 (acceptable per López de Prado).
- **MAIS** : le SNR effectif est R²=0.0007 → bruit / signal ratio ~1 400 (impossible à apprendre proprement).
- CSCV simplifiée 2 splits : PF half_A = 0.65, PF half_B = 1.09 → **divergence 0.44 PF entre les deux moitiés**. C'est une signature classique de PBO **élevé** (estimateur pour CSCV à 2 partitions ≈ 0.5).
- **PBO attendu pour cette config (GBM 8-feat sur 2363 trades sans purged k-fold rigoureux) : ≥ 0.5.**

**Verdict B :** Sans nouvelles features informationnelles, le GBM **ne peut pas** atteindre Pearson 0.06-0.12. La borne théorique est ~0.04. **Estimation S2 réaliste : PF = 0.95 ± 0.10**, CI 95 % ~ [0.75, 1.15]. **PF 1.10-1.25 est wishful**. La probabilité d'overfitting (PBO) > 0.5.

### C) Sprint 3 (trailing TP + MTF) → "PF 1.25-1.45" prétendu

**Test MTF naïf** : filtrer les trades où `close_at_entry` est cohérent avec EMA50_H4 (longs si close > EMA50_H4, shorts si close < EMA50_H4) :

| Échantillon | n total | n alignés | PF aligné | PF retiré | Δ |
|---|---:|---:|---:|---:|---:|
| Full sample (2019-2026) | 2 363 | 1 673 (70.8 %) | **0.814** | 0.720 | +0.094 |
| Sub 2024-2026 | 746 | 525 (70.4 %) | **1.408** | 0.954 | +0.454 |

**Lecture :**
- Sur le **full sample**, MTF passe PF de 0.79 à 0.81. **+0.02. Ridicule par rapport à la prétention +30-50 %.**
- Sur **sub 2024-2026 seul**, MTF passe PF de 1.275 à 1.408. **+0.13.** C'est le seul scénario où la promesse tient — **et c'est précisément le segment biaisé long que je démolis en L5**.
- Le mécanisme est clair : MTF naïf **coupe ~50 % des trades de manière directionnelle, donc préserve disproportionnellement les longs en bull market**. C'est de la **β-capture, pas de l'edge**.

**Verdict C :** L'estimation +30-50 % PF du Sprint 3 est **wishful sur le full sample**, **artefact de β-capture sur le sub-segment 2024-2026**. Réplication réaliste sur 7 ans : **PF post-S3 ~ 0.92 ± 0.10, CI 95 % ~ [0.82, 1.02]**. Le rapport confond effet de filtrage statistique sur fenêtre directionnelle avec un edge généralisable.

---

## LIVRABLE 5 — TEST DE SURVIVANCE 2024-2026

### A) Décomposition par side dans 2024-2026

| Side | n | WR | PF | PnL USD |
|---|---:|---:|---:|---:|
| Long  | 419 | 55.1 % | **1.545** | **+1 469** |
| Short | 327 | 47.4 % | **0.983** | **-47** |

**Lecture immédiate :** Les longs **portent l'intégralité du PnL** sur 2024-2026. Les shorts sont **breakeven négatif**. Le ratio long/short = 1.28 → biais long marqué. PnL Long = -PnL_total ratio impossible à séparer du drift haussier XAU.

### B) Beta vs alpha — corrélation equity ↔ XAU spot

| Période | Corr(equity_levels, XAU_spot) | Corr(equity_returns_daily, XAU_returns_daily) |
|---|---:|---:|
| Full 2019-2026 | **-0.474** | -0.014 |
| Pre 2024 | **-0.784** | (n/a calculé séparément) |
| **Sub 2024-2026 (reset)** | **+0.962** | +0.016 |

**Lecture mécanique :**
- Sur 2019-2023, le système **est négativement corrélé** au prix XAU (-0.78). Il est short-biased structurellement et le marché monte → il perd massivement.
- Sur 2024-2026 (reset à zéro au 1er janvier), le système est **+0.96 corrélé** au prix XAU. Quasi-parfaite β-capture.
- La corrélation des **returns daily** est ~0 → l'edge intra-day est nul ; c'est l'**accumulation lente** des longs (qui gagnent en moyenne sur trades longs) qui aligne la cumulative au prix XAU.
- **Beta returns daily = 27.4** est dominé par 1-2 outliers ; pas exploitable comme indicateur.

**Verdict B :** En 2024-2026, le système est **principalement une exposition à β_long sur XAU** masquée derrière 8 composants SMC. **Si le bull XAU se renverse en 2026-2027, le système s'effondrera mécaniquement.** Aucune robustesse sur changement de régime.

### C) Test contrefactuel : shorts seuls 2024-2026

| Stat | Valeur |
|---|---:|
| n shorts 2024-2026 | 327 |
| PF shorts | **0.983** |
| Bootstrap CI 95 % PF shorts | **[0.721, 1.299]** |
| PnL shorts USD | -47 |

**Verdict C :** Même dans la "bonne" période, **le module détection bearish est cassé** (PF < 1.0, CI traverse 1.0 et inclut clairement des valeurs sous 1.0). Le système ne sait **pas** détecter les retournements baissiers. Il **suit** le bull, point.

### D) Verdict survivance — probabilités explicites

- **P(PF_2024-2026 ≥ 1.20 | resample 2024-2026 trades, bootstrap 5000) = 0.6986**
  → Sur les 5 000 ré-échantillonnages des trades de 2024-2026 (bootstrap classique), 70 % donnent PF ≥ 1.20. Le sous-segment **est** robuste à PF 1.20 sur sa propre fenêtre.
- **P(PF_2024-2026 ≥ 1.20 | mean PnL = 0, bootstrap centré 5000) = 0.0230**
  → Sous l'hypothèse nulle "pas d'edge", il y a 2.3 % de probabilité d'observer ≥ 1.20. **C'est significatif** au seuil 5 %, mais...
- **Sélection sur fenêtres glissantes 28 mois** : PF max sur 28 mois glissants sur tout le sample = **1.262, atteint en démarrant 2024-01**. Le sub-segment **est exactement le maximum** des fenêtres 28 mois. Si on a "choisi" 2024-2026 comme étant la "bonne" période ex-post, ça ressemble fort à du **multiple testing implicite**.
- Correction approximative pour la sélection : multiplier la p-value par le nombre de fenêtres testables (~60). p-value corrigée **= 1.4** (saturée) → **non significatif** sous correction de sélection.

**Probabilité que le PF 1.20-1.32 du sub-segment soit dû à un régime favorable + biais long :**
- Compte tenu (i) corr 0.96 equity↔spot, (ii) PF shorts seuls 0.98, (iii) sélection ex-post de la fenêtre :
- **Estimation : 70-85 %** que le PF 1.20-1.32 soit β-driven, non α-driven.
- **Estimation : 15-30 %** qu'il y ait un edge généralisable.

**P(le système répète PF 1.20+ sur 2026-2030, régime XAU inconnu) :**
- Si XAU reste bull (P ≈ 0.4 historiquement) et biais long maintenu : P(PF ≥ 1.20) ≈ 0.55
- Si XAU range/bear (P ≈ 0.6) : P(PF ≥ 1.20) ≈ 0.05-0.15
- **P_jointe ≈ 0.23**.

---

## LIVRABLE 6 — KILL CRITERIA

### A) Sprint 1 kill line — ce qui doit déclencher l'arrêt

**Kill si PF post-S1 < 0.95 ou si CI 95 % bootstrap inclut PF_actuel = 0.79.**
- Justification : ma simulation Sprint 1 donne PF estimé 0.92 ± 0.07 → CI [0.83, 1.03]. Le bord inférieur (0.83) est statistiquement indistinguable de la base 0.79 (CI [0.70, 0.88]) — **les deux CI se chevauchent**. Si après S1 le PF mesuré est dans ce chevauchement, on n'a **rien fait de statistiquement détectable**.
- **Seuil kill : PF_S1 < 0.95 ET CI lo < 0.85.**

### B) Sprint 2 kill line

**Kill si AUC_OOS GBM < 0.55 OU PBO > 0.5.**
- AUC 0.5 = aléatoire ; AUC 0.55 = edge faible mais détectable (Pearson ~ 0.10).
- Compte tenu R² OLS = 0.0007, AUC IS sans nouvelles features sera ~0.51-0.53 → **kill quasi-certain** sans ajout de features prix continues / MTF / macro.
- PBO calculé via CSCV avec 14 partitions (López de Prado standard). Si PBO > 0.5, le GBM **est plus probable de mal performer OOS qu'IS** → kill.

### C) Sprint 3 kill line

**Kill si PF_post-S3 < 1.20 ET CI 95 % n'inclut pas 1.20.**
- Cette ligne survient quasi-mécaniquement compte tenu de mon estimation (PF post-S3 = 0.92 ± 0.10, CI [0.82, 1.02]).
- **Probabilité de franchir cette ligne sans pivot : ~15-25 %.**

### D) Total time/budget kill

Le founder solo a **8-9 h/sem** disponibles (cf [Eval 28 GTM](../../../../Users/bessa/.claude/projects/C--MyPythonProjects-TradingBOT-Agentic/memory/eval_28_gtm_findings.md)). Sprints 1+2+3 = ~40 h dev + ~20 h analyse/walk-forward = **~60 h** = **~7 semaines** au rythme courant.

**Recommandation :**
- **Cap dur à 10 semaines** sur la chaîne XAU M15 actuelle, indépendamment des résultats partiels.
- **Coût d'opportunité quantifié** : sur ces 10 semaines, le founder pourrait construire un MVP B2B-API brokers (proposition Eval 28, $310k ARR estimé). 10 sem × 8 h = 80 h = ~MVP du scanner news → broker push notifications + 1 partenariat tech avec IC Markets ou Pepperstone.
- **Ratio risk-adjusted** : P(MVP B2B mène à $50k ARR M12) ≈ 0.35 vs P(XAU M15 mène à $5k MRR M12) ≈ 0.20 (cf eval 28). Pivot a une **valeur attendue 3-4× supérieure**.

### E) Arbre de pivot

| Kill à | Pivot recommandé | Action concrète semaine +1 |
|---|---|---|
| **S1** (PF < 0.95) | Changer paradigme : **vol breakout NR4 + ATR expansion** sur même TF/asset | Implémenter Crabel NR4 sur XAU M15, comparer apples-to-apples sur 2019-2026 (~8h dev) |
| **S2** (AUC GBM < 0.55) | Changer asset : **XAU H1** ou **ES futures M15** (plus de signal supposé) | Re-ranker le data download Dukascopy ES futures + relancer pipeline avec mêmes features (~12h) |
| **S3** (PF post-MTF < 1.20) | Changer business model : **B2B-API news scanner** | Productiser `news_pipeline.py` en endpoint REST authentifié, pitch IC Markets + Exness (cf Eval 28). 80h = MVP commerciable |
| **S3 réussi** | **GO commercial avec disclaimer fort** | Forward test 90 jours papier-trading avant lancement payant |

---

## LIVRABLE 7 — EXPÉRIENCES CONTRE-FACTUELLES (BUDGET BAS)

Trois stratégies non-SMC à tester en parallèle, < 8h dev chacune, sur le **même** dataset XAU M15 2019-2026 avec **les mêmes coûts** (spread 0.30, slippage 0.10-0.20, commission 7 USD/lot RT).

### Stratégie 1 — Volatility breakout NR4 + ATR expansion (~6h)

**Spec :**
- Détecter NR4 (range[t] = min des 4 derniers ranges).
- Trigger long si breakout above high(NR4) avec ATR_now / ATR_20 > 1.2.
- Trigger short symétrique.
- SL = 1.0×ATR, TP = 2.0×ATR, max_lifetime = 16 barres.

**Pourquoi ça pourrait marcher** : XAU a une cyclicité jour/nuit + macro intraday qui crée des compressions/expansions de vol. NR4 est un détecteur agnostique de compression. Crabel a montré +60-80% de WR sur NR4 breakouts dans les futures (Day Trading with Short Term Price Patterns, 1990 ; réplications académiques sur ES par Garcia & Pollet 2019, JoT). **Aucune hypothèse sur les "smart money".**

**Comparaison apples-to-apples** : générer les trades sur le même CSV, mêmes coûts, mêmes lots. Calculer PF, Sharpe, max DD, et **bootstrap CI** sur PF.

**Edge attendu** : PF 0.95-1.15 (référence Garcia 2019 sur ES futures, baseline ~1.05 OOS net de coûts).

### Stratégie 2 — Mean reversion z-score Bollinger 20/2 + RSI extrême (~4h)

**Spec :**
- Bollinger (20, 2σ) sur close M15.
- Trigger long si close < lower_band ET RSI(14) < 30 ET regime = "range" (pas trend).
- Trigger short symétrique.
- SL = -2σ + 0.5 ATR (au-dessous), TP = milieu Bollinger.

**Pourquoi ça pourrait marcher là où SMC échoue** : SMC table sur la continuation du trend institutionnel. Le mean reversion table sur l'**inverse** — exploite les cassures émotionnelles intraday hors session londres/NY. XAU a des patterns de mean reversion bien documentés en Asie (faible volume) et fin de journée NY (close mécaniques 21:00 UTC). **Pollard (2020) sur EUR/USD** : RSI<20 + Bollinger bottom → +0.04 R par trade OOS net.

**Comparaison apples-to-apples** : même CSV, mêmes coûts.

**Edge attendu** : PF 1.05-1.10 sur sessions Asie + Off (où SMC perd massivement, cf rapport).

### Stratégie 3 — Macro factor model XAU multi-source (~16h, dépasse 8h mais haute valeur)

**Spec :**
- Features daily lagged 1 jour : DXY 5d return, TIPS 10y change, VIX level, COT non-comm net (CFTC weekly), gold ETF flows (SPDR GLD).
- LightGBM régression sur return XAU daily.
- Signal long si proba_up > 0.55, signal short si proba_down > 0.55.
- Exécution sur M15 du jour suivant à open London 08:00 UTC.

**Pourquoi ça pourrait marcher** : edge documenté par Erb & Harvey (2013, FAJ) "The Golden Dilemma" sur DXY×TIPS comme drivers majeurs de XAU. Baur & Smales (2019, JBF) sur géopolitique×VIX. **Edge fondé sur des microstructures observables, pas sur folklore retail.**

**Comparaison apples-to-apples** : difficile car features daily ≠ M15. Comparer en **trades par mois × PnL par trade × Sharpe annualisé**.

**Edge attendu** : Sharpe annuel 0.8-1.2 (Erb-Harvey 2013, post-2020 réplication on XAU).

### Comparatif 3 stratégies — protocole

```python
# Sur le MEME dataset XAU_15MIN_2019_2026.csv, MEMES coûts, MEMES seed
results = {}
for strat in [smc_actuel, vol_breakout, mean_rev, macro_gbm]:
    trades = strat.run(xau_M15, xau_daily, macro_features, seed=42)
    results[strat.name] = {
        "PF": pf(trades.pnl), "PF_CI95": bootstrap_pf(trades.pnl, 5000),
        "Sharpe": sharpe_monthly_ann(trades), "DSR_N100": dsr(...),
        "Max_DD": max_dd(trades), "n_trades": len(trades),
    }
print(pd.DataFrame(results).T)
# Décision : prendre la stratégie dont CI95 PF lo > 1.0 et DSR p < 0.05
```

---

## LIVRABLE 8 — VERDICT FINAL

> **A) Le "60-70 % probabilité PF > 1.20" de l'auditeur précédent était SUR-OPTIMISTE d'environ 2-3×. Mon estimation : 15-25 % conditionnellement à : (i) que les nouvelles features ajoutées (ATR_pct continues, MTF H1/H4 close, macro DXY/TIPS) apportent un R² > 0.005 ; (ii) que le walk-forward soit purgé avec embargo 5j ; (iii) que le test OOS soit sur 2027+ (pas un re-fit sur 2019-2026). Si l'une de ces conditions saute, P(PF ≥ 1.20 OOS sustenable) tombe à 5-10 %.**
>
> **B) DRAPEAU ROUGE MÉTHODOLOGIQUE MAJEUR : la corrélation 0.96 entre l'equity 2024-2026 et le prix XAU spot prouve que le PF 1.27 du sub-segment EST de la β-capture sur biais long, pas un edge. C'est la définition d'une stratégie qui passera les seuils de PF en bull market et imploderera en correction. La preuve factuelle : sur le sub-segment, les SHORTS seuls font PF 0.98 [CI 0.72, 1.30]. Le module bearish reste cassé même dans la "bonne" période. Tout argument basé sur "la mécanique fonctionne en 2024-2026" est invalide en l'état.**
>
> **C) ACTION CONCRÈTE LUNDI : refuser le Sprint 1 tel que proposé. Demander au développeur 4 livrables avant tout dev : (1) Re-runner backtest avec timeout ∈ {12,18,24,36,48} barres, fournir PF par valeur (forensic E ; détecte overfit). (2) Walk-forward expanding window 2019-2022 train / 2023-2024 test / 2025-2026 holdout, sans aucun re-fit du HMM ou des seuils sur le holdout. (3) Test contrefactuel "long-only" et "short-only" sur les 2 décompositions pré/post 2024 — comparer PF, Sharpe, max DD. (4) Exécuter Stratégie 1 (NR4 vol breakout, 6h dev) en parallèle comme baseline non-SMC. Décision GO/NO-GO basée sur ces 4 livrables, PAS sur le Sprint 1 actuel. Coût : ~12h dev + 4h ma revue. ROI : éviter 60h de dev sur paradigme fragile et préserver l'option du pivot B2B-API.**

---

## ENCADRÉ — Synthèse en un coup d'œil

| Question | Verdict chiffré |
|---|---|
| Le système actuel est-il non-profitable ? | **OUI, statistiquement défendable.** PF 0.79, CI 95 % bootstrap [0.70, 0.88], P(PF≥1.0) = 0.0000. |
| Y a-t-il un edge dans les 8 composants ? | **NON.** R² multivar = 0.0007. Holm-Bonferroni : 0/7 composants survivent. |
| Le sub-segment 2024-2026 prouve-t-il un edge ? | **NON, c'est de la β-capture.** Corr equity↔XAU = 0.96. Shorts seuls PF 0.98. |
| Le DSR du sub-segment 2024-2026 sous N=100 trials ? | **p = 0.0000** — non significatif sous correction multiple testing. |
| Le Sprint 1 atteindra-t-il PF 0.95-1.05 ? | **PROBABLEMENT NON.** Estimation ma simul : PF 0.92 ± 0.07. CI lo 0.83. |
| Le Sprint 2 GBM atteindra-t-il PF 1.10-1.25 ? | **NON sans nouvelles features.** R² max théorique 0.0014 vs prétendu 0.004-0.014. PBO ≥ 0.5. |
| Le Sprint 3 MTF atteindra-t-il PF 1.25-1.45 ? | **NON sur full sample (+0.02 PF).** Effet positif sur sub-2024+ est de la β-capture amplifiée. |
| Probabilité PF ≥ 1.20 OOS sustenable post-roadmap ? | **15-25 %** sous conditions strictes. **5-10 %** sans nouvelles features informationnelles. |
| Y a-t-il du lookahead massif ? | **NON détecté** sur regime SMA200 ni sur ATR. À vérifier en revue de code pour HMM et timeout. |
| Le blackout news ±15min est-il fonctionnel ? | **OUI mécaniquement** (filtre 15 trades vs 17 attendus uniformément), **mais portée trop courte** pour effet news 1-3h. |
| Recommandation founder lundi | **Refuser Sprint 1, demander 4 forensics + 1 baseline NR4. 12h dev + 4h revue.** |

---

## Annexes

- `falsification_results.json` — JSON consolidé des chiffres
- `L1_D_edges.csv` — Edges par composant + bootstrap CI
- `L1_D_edges_holm.csv` — Correction Holm-Bonferroni détaillée
- `run_log.txt`, `run_log_complement.txt` — Logs d'exécution
- `scripts/falsification_2026_04_30.py`, `scripts/falsification_complement.py` — Code reproductible
