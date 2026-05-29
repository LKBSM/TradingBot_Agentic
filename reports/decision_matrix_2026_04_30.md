# Décision Matrix Finale — Smart Sentinel AI
## Synthèse 4-runs : XAU M15 v2 / XAU H1 / EURUSD M15 / NR4 baseline
**Date :** 2026-04-30
**Période d'analyse :** 2019-2026 (XAU M15/H1 et NR4) ; 2019-2025 (EURUSD M15)
**Coûts :** spread, slippage, commission identiques (adapté FX), capital 10 000 USD, risk 1 %/trade, seed=42

---

## A. TABLEAU MAÎTRE — toutes métriques côte à côte

| Métrique | **XAU M15 v2** (timeout 64) | **XAU H1** | **EURUSD M15** | **NR4 baseline** (XAU M15) |
|---|---:|---:|---:|---:|
| Période | 2019-01 → 2026-04 | 2019-01 → 2026-04 | 2019-01 → 2025-12 | 2019-01 → 2026-04 |
| n_trades | 1 753 | 532 | 1 805 | 6 538 |
| Win rate | 41.0 % | 45.7 % | 40.4 % | 31.3 % |
| **Profit Factor** | **1.037** | **0.946** | **0.854** | **0.603** |
| **CI 95 % PF** | **[0.917, 1.170]** | **[0.758, 1.166]** | **[0.748, 0.969]** | **[0.540, 0.679]** |
| Sharpe (mensuel ann.) | +0.27 | -0.17 | -0.96 | +0.08 |
| Sortino | +0.45 | -0.28 | -1.42 | +0.08 |
| Max DD | -48.0 % | **-16.2 %** ✅ | -53.5 % | -133 % (bust) |
| Return total | +14.8 % | -8.3 % | -50.2 % | -129 % |
| Pearson(score, R) | -0.008 | **+0.016** | -0.009 | n/a (pas de score) |
| Buy & Hold benchmark | +255.6 % | +255.3 % | +2.5 % | +255.6 % |

### Critère GO/NO-GO : **CI 95 % PF lo > 1.0 sur full sample**

| Strat | CI 95 % lo | Verdict |
|---|---:|:---:|
| XAU M15 v2 | 0.917 | ❌ NO-GO (manque 0.083) |
| XAU H1 | 0.758 | ❌ NO-GO |
| EURUSD M15 | 0.748 | ❌ NO-GO (CI hi 0.969 exclut 1.0) |
| NR4 baseline | 0.540 | ❌ NO-GO |

**Aucune stratégie ne franchit le critère.** Le plus proche est XAU M15 v2 avec un CI lo à 0.917 — il manquerait ~9 points de PF pour passer.

---

## B. LECTURE TRANSVERSE — qu'est-ce qui se confirme ?

### B.1 Le système n'a aucun edge généralisable

- **EURUSD M15 PF 0.854** alors que **buy & hold EURUSD = +2.46 %** sur 7 ans. Le système perd 50 % du capital sur un asset range-bound. Si le pipeline avait un edge structurel, il ne pourrait pas faire pire que le BH d'un asset sans drift.
- **EURUSD se DÉGRADE post-2024** (PF 0.886 → 0.736), inverse exact de XAU M15 (0.68 → 1.28). Cela **prouve** que l'amélioration 2024-2026 sur XAU est **β-capture du bull XAU**, pas un edge structurel.
- **NR4 (paradigme radicalement différent) sur même XAU M15 fait PF 0.60** — pire que SMC. Donc le problème n'est pas le paradigme SMC vs vol-breakout — **c'est XAU M15 qui est noisy**, **et le pipeline qui sous-performe le passive holding sur asset stationnaire (EURUSD)**.

### B.2 H1 résout le problème de drawdown mais pas l'edge

- **MaxDD H1 = -16 %** vs M15 v2 = -48 %. C'est un gain colossal en risk-adjusted.
- Mais PF reste sous 1.0, return négatif, Sharpe négatif. La symétrie long/short revient (PF 1.01 vs 0.86 — pas de biais), mais **rien n'est profitable**.
- **L'amélioration de drawdown vient de la baisse de fréquence** (532 trades H1 vs 1753 M15 v2 = 30 %), pas d'un edge.

### B.3 Le pipeline est XAU-spécifique par construction

Les seuils détecteurs (atr_mult=0.4 pour FVG, atr_tol=0.25 pour retest, lookback=14 pour RSI div, SMA200 pour régime, etc.) ont été tunés implicitement sur XAU. Sans aucun re-tuning :
- **XAU M15** : PF 1.04 (avec bull XAU)
- **XAU H1** : PF 0.95 (sans bull intraday capture)
- **EURUSD M15** : PF 0.85 (asset range-bound)
- **NR4 sur XAU M15** : PF 0.60 (paradigme alternatif noisy)

Pattern clair : **le pipeline a une dégradation monotone de la profitabilité quand on s'éloigne de la configuration XAU M15 + bull regime**.

---

## C. RECOMMANDATIONS — décision finale

### Critère strict (CI 95 % PF lo > 1.0)

**Aucune stratégie ne passe.** Décision strictement appliquée : **(d) Kill total → pivot B2B-API brokers**.

### Critère relaxé (PF point > 1.0 ET MaxDD < 50 %)

| Strat | PF | MaxDD | Pass relaxé ? |
|---|---:|---:|:---:|
| XAU M15 v2 | 1.037 | -48.0 % | ✓ marginal |
| XAU H1 | 0.946 | -16.2 % | ✗ (PF) |
| EURUSD M15 | 0.854 | -53.5 % | ✗ |
| NR4 | 0.603 | -133 % | ✗ |

Sous critère relaxé, **XAU M15 v2 marginalement passe** mais avec MaxDD à la limite acceptable et CI lo qui exclut 1.0.

### Critère production-grade (commercialisable à 29-149 USD/mois)

Cible eval_27 : **PF > 1.20, MaxDD < 25 %, Sharpe > 1.0, 3 années consécutives profitables OOS**.

**Aucune stratégie testée ne s'en approche.** Le plus proche est XAU H1 (MaxDD -16 % ✓, mais PF 0.95 et Sharpe -0.17). XAU M15 v2 a PF 1.04 mais MaxDD -48 % et Sharpe +0.27.

---

## D. RECOMMANDATION FINALE

**Décision : (d) Kill total du paradigme SMC sur XAU M15/H1 + EURUSD M15. Pivot B2B-API brokers.**

### Justification chiffrée

1. **CI 95 % PF lo > 1.0 : 0/4 strats passent.** Critère défini ex-ante par le user, appliqué strictement.
2. **EURUSD résultat décisif** : si le pipeline avait un edge structurel, il ne pourrait pas perdre 50 % sur asset range-bound. Cela invalide la thèse "edge généralisable cross-asset".
3. **XAU M15 v2 PF 1.04 = β-capture pure**, prouvée par l'inversion EURUSD (PF se dégrade post-2024 alors qu'il s'améliore sur XAU). Le système est **non-robuste au régime de marché**.
4. **XAU H1 ne sauve pas la situation** : PF 0.95, return -8 %. Le passage TF n'est pas le pivot.
5. **NR4 sur XAU M15 fait pire** (PF 0.60). Le problème n'est pas le paradigme SMC.

### Alternative tolérée — "GO commercial dégradé XAU M15 v2 avec disclaimers"

Si le founder refuse le kill, l'option de moins-pire est :
- **Lancer XAU M15 v2 (timeout=64) en TIER FREE only**, avec disclaimers explicites :
  - "Performance 2019-2026 PF 1.04, return +14.8 %, MaxDD -48 % — non-garantie, β-capture sur bull XAU 2024-2026"
  - "Pour usage éducatif uniquement — non un conseil financier"
  - **Géoblock US/QC/UK/OFAC** déjà actif (Sprint W1)
- **Ne PAS facturer ANALYST/STRATEGIST/INSTITUTIONAL** tant que CI 95 % PF lo > 1.0 OOS.
- Forward-test 90 jours papier-trading **avant** toute monétisation.

C'est techniquement légal mais commercialement fragile. **Coût d'opportunité** : 1-2 mois de "prouver" un système β-driven, vs même temps pour MVP B2B-API ($310k ARR cible).

### Alternative préférée — "Pivot B2B-API brokers"

Capitaliser les actifs déjà construits :
1. `news_pipeline.py` (FF JSON → CSV → calendrier API) — **endpoint REST authentifié pour brokers retail**.
2. **Score confluence comme "context layer"** sur signaux fournis par les brokers (pas comme générateur de signal autonome).
3. **Forward test 90 jours** en parallèle avec 1-2 broker-partners (IC Markets, Pepperstone) pour mesurer la valeur pratique du context.
4. **Sortir du mode "predict & deliver"** vers "explain & contextualise" — bien plus défendable juridiquement et techniquement.

**ROI attendu** : MVP en 80h dev. Si 1 broker-partner signe un POC à $5k/mois, ARR M12 réaliste $30-60k. Au-delà, $310k ARR cible Eval 28 (3-5 broker-partners + context layer multi-asset).

---

## E. KILL CRITERION POST-RECOMMANDATION

**Pour l'option "GO XAU M15 v2 dégradé"** (si retenue) :
- **Forward test 90j paper-trading**, mesuré en temps réel sur Telegram (logs immutable).
- **Kill si PF forward < 0.95 OU MaxDD > 35 %.**

**Pour l'option "Pivot B2B-API"** :
- **Kill si 0 broker-partner signé après 60 jours de pitch.** Re-évaluer alors XAU H1 long-only filtré + sortie B2C analyst-only.

### Time/budget cap

**Cap dur : 8 semaines** (8h/sem × 8 = 64h total). Au-delà, indépendamment de l'option choisie, **arrêt obligatoire et bilan honnête : abandon du business model TradingBOT_Agentic ou pivot complet vers une autre vertical (e.g. AI prompts marketplace, no-code app, etc.).**

---

## F. ENCADRÉ — Synthèse 1 ligne

> **Aucune des 4 stratégies ne franchit CI 95 % PF lo > 1.0 sur full sample. EURUSD M15 PF 0.85 [CI 0.75, 0.97] sur asset range-bound prouve que le pipeline ne se transfère pas hors XAU. Recommandation finale : (d) kill total + pivot B2B-API brokers (80h dev MVP, $310k ARR cible). Option dégradée acceptable : GO XAU M15 v2 FREE-tier only avec disclaimers, 90j forward-test obligatoire avant monétisation.**

---

## ANNEXES — fichiers produits dans cette session

| Action | Script | Rapport | Données |
|---|---|---|---|
| 1. Quick-win | `scripts/quant_audit_2026_04_30.py` (CFG modif) | `reports/audit_2026_04_30_v2_timeout64.md` | `_v2_timeout64_*.{csv,json}` |
| 2. XAU H1 | `scripts/comparatif_xau_h1.py` | `reports/comparatif_xau_h1.md` | `comparatif_xau_h1_*.{csv,json}` |
| 3. EURUSD M15 | `scripts/comparatif_eurusd_m15.py` | `reports/comparatif_eurusd_m15.md` | `comparatif_eurusd_m15_*.{csv,json}` |
| 4. Décision matrix | (synthèse) | `reports/decision_matrix_2026_04_30.md` | (ce fichier) |
| Forensics base | `scripts/forensics_*.py`, `baseline_nr4_*.py` | `reports/forensics/forensics_2026_04_30.md` | `reports/forensics/L1-L4_*` |
| Falsification base | `scripts/falsification_*.py` | `reports/falsification/falsification_2026_04_30.md` | `reports/falsification/*` |

Production change committed in **44bc8bd** (timeout 24→64 in audit script + state machine config).
