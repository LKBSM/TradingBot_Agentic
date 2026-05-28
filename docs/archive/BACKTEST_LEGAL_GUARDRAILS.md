# Smart Sentinel AI — Backtest Legal Guardrails

> **Auteur** : Eval 18 (K10 + K11) — 2026-04-26
> **Statut** : NORMATIF. Toute publication marketing (landing, social, ads,
> pitch deck, sales call, mail prospect) doit passer ce filtre AVANT envoi.

## 0. Pourquoi ce document existe

Une statistique de PnL gonflée par look-ahead bias, cherry-picking ou
multiple-testing publiée sur une page commerciale n'est pas une « marge
d'optimisme » : c'est une **fausse représentation matérielle** au sens
de la **FTC Endorsement Guides** (16 CFR §255 — *substantiation
required*) et de l'**article L121-1 du Code de la consommation**
français (« pratiques commerciales trompeuses »), avec en prime une
violation potentielle du **règlement AMF n°2016-08** sur la
« sollicitation d'investissement ». Pénalité encourue côté FTC : jusqu'à
$50 120 par violation ; AMF/DGCCRF : amendes administratives + sanctions
pénales (jusqu'à 2 ans + 300 k€).

Sources :
- FTC, *Guides Concerning the Use of Endorsements and Testimonials*, 16 CFR §255.0–§255.5 (révision 2023).
- FTC v. *Triangle Media Corp.*, No. 18-cv-2660 (S.D. Cal. 2018) — sanctions $1.32 M sur publicité « before/after ».
- AMF, *Position-recommandation 2014-12* (commercialisation des CFD/forex), §3.2.1.
- Code de la consommation, art. L121-1 à L121-5 (pratiques trompeuses).
- ESMA Q&A on CFD restrictions, §15 (« past performance ≠ future »).

---

## 1. Verdict actuel sur les chiffres déjà cités en interne

| Source | Métrique citée | Chiffre | Statut | Raison |
|---|---|---|---|---|
| `audit_backtest_2026_04_24.md` | PF 7-ans config prod | 0 trades | **À ENTERRER** | Aucune statistique. Le seuil 75 est mathématiquement inatteignable en backtest (cf §2.1 du rapport eval_18). |
| `audit_backtest_2026_04_24.md` | PF max sweep (relaxed_30) | 0.96 | **À ENTERRER** | < 1.0 = perdant. Surtout : best-of-7 sans correction Hansen SPA → biais multiple-testing (≈ +15-25 % d'inflation attendue). |
| `baseline_2019_2025.md` | PF 6-ans baseline (enter=40) | 1.086 | **INTERNE UNIQUEMENT** | Single-fold in-sample, sans walk-forward, sans coûts, sans correction multiple-testing. **NE JAMAIS publier.** |
| `baseline_2019_2025.md` | Sharpe annualisé baseline | 0.59 | **INTERNE UNIQUEMENT** | Idem. Calculé via `trades_per_year` extrapolation (annualisation par √trades), pas par √252 jours — convention non-standard. |
| `confluence_calibration.md` | Pearson(score, r_multiple) | −0.023 | **PUBLIABLE — comme aveu** | C'est un constat technique négatif. Peut servir de transparence (« nous avons audité notre propre score, voici le résultat »). |
| `replay_post_asymmetry_fix.json` (filtre retest) | PF 1.4-an | 0.631 | **À ENTERRER** | < 1.0, échantillon 13 trades non-significatif. |
| `replay_retest_2025_v2.json` (config relâchée) | PF 1.4-an | 1.08 | **À ENTERRER** | Cherry-pick d'une fenêtre + d'une config, sans walk-forward. |

### Verdict global

**Aucun chiffre actuel n'est publiable** sur une page marketing en
l'état. Le seul chemin légalement défendable :

1. Exécuter `reports/eval_18_walkforward_skeleton.py` (livré).
2. Exécuter `scripts/montecarlo_bootstrap.py` (à écrire — voir K6).
3. Si IC 95 % du PF OOS > 1.0 ET Sharpe OOS ≥ 0.8 → publier avec les
   reformulations §3.
4. Sinon → publier UNIQUEMENT le langage qualitatif §4.

---

## 2. Chiffres autorisés à la publication APRÈS exécution walk-forward

### 2.1 Format obligatoire de citation

Tout chiffre publié doit suivre ce template :

> *« Sur XAU/USD M15 entre [TEST_START] et [TEST_END] (out-of-sample,
> jamais utilisé pour optimiser le système), profit factor [PF] [IC95 %
> bootstrap = (PF_low, PF_high)] sur N trades, drawdown maximum [DD_MAX]
> R, Sharpe annualisé [S]. Coûts inclus : spread session moyen [SPREAD]
> bps + slippage ATR-proportionnel + commission $7/lot. Méthodologie :
> walk-forward avec 5-jours d'embargo, sélection de paramètres sur
> 2019-2023 uniquement, test 2024-2025 jamais vu pendant le tuning. Code
> reproductible : `reports/eval_18_walkforward_skeleton.py`,
> commit hash [GIT_SHA]. La performance passée ne préjuge pas des
> performances futures. »*

### 2.2 Disclaimers obligatoires (FTC + AMF)

Sur toute page contenant un PnL :

```
[Avertissement] Les performances historiques sont basées sur une
simulation walk-forward et ne garantissent aucun résultat futur. Le
trading sur or, devises ou indices comporte un risque substantiel de
perte en capital. Smart Sentinel AI ne fournit pas de conseil en
investissement personnalisé au sens de l'article L321-1-4 du Code
monétaire et financier. Les utilisateurs sont seuls responsables de
leurs décisions de trading.
```

### 2.3 Interdictions explicites

| Interdit | Pourquoi | Alternative légale |
|---|---|---|
| « +X % de gain en N mois » | Implique une promesse de rendement (AMF Position 2014-12 §3.2). | « Profit factor X.XX sur N trades OOS » |
| « X % de réussite » sans n_trades + IC | Win-rate isolé est un Brier-incomplete claim (FTC *substantiation*). | « X % de win-rate (n=N, IC95% [a, b]) sur la fenêtre OOS [date1, date2]. » |
| « Stratégie testée sur 7 ans » | Faux : on a test sur 2024-2025 uniquement (2 ans), 2019-2023 = train+val. | « Méthodologie walk-forward : sélection sur 2019-2023, test out-of-sample sur 2024-2025. » |
| Capture d'écran d'un trade gagnant isolé | Cherry-picking → trompeur (16 CFR §255.2(b)). | Capture d'écran de la distribution complète des trades sur OOS. |
| « Notre IA prédit … » | Falsifiable + suggère une garantie. | « Notre système identifie statistiquement … » |
| « Profit factor 1.086 » (chiffre baseline IS) | In-sample, aucune valeur prédictive. | Ne pas citer. |

### 2.4 Cas particuliers

- **Backtest sans frais** → INTERDIT en publication. Tout PnL public
  doit inclure spread + slippage + commission, idéalement chiffrés à
  côté.
- **Volatilité forecasting** → Publier le **MAE** ou **RMSE** out-of-sample,
  jamais des « +X % de précision » qui n'ont pas de définition stable.
- **Témoignages clients** → 16 CFR §255.5 : un cas atypique doit être
  flaggé « results not typical » de façon proéminente.

---

## 3. Reformulations légales prêtes-à-coller

### 3.1 Si OOS PF ≥ 1.5 et Sharpe ≥ 0.8

> Sur la fenêtre out-of-sample 2024–2025 (jamais utilisée pour
> calibrer le système), Smart Sentinel AI affiche un profit factor de
> [X.XX] (intervalle de confiance bootstrap 95% : [a.aa, b.bb]) et un
> Sharpe annualisé de [S.SS] sur [N] signaux XAU/USD M15, après
> intégration des coûts de transaction réels (spread session moyen
> [SP] bps, slippage ATR-proportionnel, commission $7 par lot
> standard). Drawdown maximum observé : [DD] R. *Méthodologie complète
> et code source disponibles sur demande.*

### 3.2 Si OOS PF entre 1.2 et 1.5 (borderline)

> Smart Sentinel AI est en phase de bêta de paper-trading sur la
> période 2024–2025. Les statistiques préliminaires (profit factor
> [X.XX], n=[N] trades) sont publiées à titre transparent ; nous
> recommandons aux premiers utilisateurs de tester le système en mode
> *paper-trade* avant tout engagement de capital.

### 3.3 Si OOS PF < 1.2 (état actuel)

> Smart Sentinel AI fournit des **alertes contextuelles** sur les
> mouvements de marché — Break-of-Structure, retests, fenêtres de news
> high-impact — accompagnées de narratives explicatives générées par
> IA. Le service ne vend pas de signaux d'achat-vente garantis et ne
> publie pas de profit factor pour cette raison : nous considérons
> qu'aucune méthodologie de backtest publiable ne peut substantiver
> aujourd'hui une promesse de gain. **Le produit se positionne sur la
> qualité de l'analyse, pas sur le PnL.**

→ **C'est la formulation à utiliser tant que walk-forward + bootstrap
n'ont pas été exécutés.** Elle est défendable juridiquement et reste
commercialement honnête.

---

## 4. Langage qualitatif autorisé sans backtest publiable

Ces formulations ne nécessitent pas de chiffres et restent légales tant
qu'elles ne suggèrent ni rendement, ni garantie, ni performance
historique implicite :

- « Détection automatique des Break-of-Structure et Fair Value Gaps »
- « Narratives multi-paragraphes générées par Claude pour chaque signal »
- « Filtrage par calendrier économique HIGH-impact (NFP, FOMC, CPI) »
- « Couverture multi-instruments (XAU, EUR/USD, BTC, US500) »
- « Cadre de scoring 0–100 transparent (8 composants documentés) »

→ **Tout langage marketing actuel doit être migré vers ce registre tant
que walk-forward n'a pas validé un OOS exploitable.**

---

## 5. Audit de cohérence — fichiers à modifier

### 5.1 `BUSINESS_PLAN_SMART_SENTINEL.md`

Lignes 39-42 — tableau « v1-v3 Sharpe -32 to -26 » : c'est l'ancien
système RL, **pas** le pivot Smart Sentinel actuel. Reformuler en :

> *Le pipeline actuel (Smart Sentinel AI, post-pivot 2026-04) est en
> phase de validation walk-forward. Les métriques de performance
> finales seront publiées après exécution OOS sur 2024-2025.*

Ligne 291 — *« Tier 4 Institutional, backtesting data »* : OK, *backtesting
data* veut dire *historical OHLCV access*, pas *promesse PnL*. Garder.

### 5.2 `COMMERCIALIZATION_REPORT.md`

À auditer ligne par ligne dès qu'un chiffre PF/Sharpe/win-rate y
apparaît. Le grep K10 a trouvé : aucun chiffre « PF/Sharpe/expectancy »
explicite — le document est en règle pour l'instant. **Vérifier à
chaque mise à jour.**

### 5.3 `README.md`

Le grep K10 ne trouve aucune métrique chiffrée. **OK** — laisser tel
quel, NE PAS ajouter de chiffres tant que §3 n'est pas validé.

### 5.4 Landing page (à venir)

Devra reproduire le template §2.1 textuellement, **sans
paraphrase qui pourrait altérer les qualifications**.

---

## 6. Process d'approbation avant publication

```
[1] Modifier le wording marketing               (developer/founder)
[2] Lancer reports/eval_18_walkforward_skeleton.py  (Bash)
[3] Lancer scripts/montecarlo_bootstrap.py      (à écrire)
[4] Comparer le wording aux §2.1 / §3.x         (founder)
[5] Si chiffre cité : vérifier IC 95% > 1.0     (founder)
[6] Vérifier disclaimers §2.2 présents          (founder)
[7] Commit + git tag legal-review-YYYY-MM-DD    (developer)
[8] Publier                                     (marketing)
```

**Aucune étape ne peut être sautée.** En cas de doute, défaut = §4
(qualitatif sans chiffres).

---

## 7. Annexe — biais identifiés en eval_18

Ces biais sont la raison pour laquelle on impose ce process. Détail
complet : `reports/eval_18_backtest.md`.

| # | Biais | Fichier | Impact estimé |
|---|---|---|---|
| B1 | Look-ahead via `rolling(center=True)` (fractals) | `src/environment/strategy_features.py:617-618` | Mitigé par `shift(N)` ligne 637-638 — **vérifié OK** mais fragile, à noter dans CHANGELOG |
| B2 | Look-ahead `iloc[i+1]`, `iloc[i+2]` (swing) | `src/environment/multi_timeframe_features.py:554-566` | **CONFIRMÉ leak** dans le sous-système 4H — non utilisé par le replay actuel mais à neutraliser avant tout MTF backtest |
| B3 | `bfill()` sur indicateurs lents | `src/environment/environment.py:802`, `multi_timeframe_features.py:184` | Leak théorique sur les 1-50 premiers bars de chaque session — **mitigé par WARMUP=100** dans le replay |
| B4 | `expanding().quantile()` pour vol-regime | `src/backtest/state_machine_replay.py:145-146` | Causal (`expanding` = passé seulement) — **OK** |
| B5 | `shift(-cfg.pred_horizon)` pour `future_atr` | `src/intelligence/volatility_forecaster.py:669` | **TARGET de training** — pas un feature, ne fuit pas dans le replay |
| B6 | Multiple-testing sur 7 configs sweep | `scripts/audit_backtest.py:76-85` | Inflation PF estimée +15-25 % vs vraie distribution — **À CORRIGER avec Hansen SPA** |
| B7 | Pas de coûts (spread/slippage/commission) | `src/backtest/state_machine_replay.py:_build_trade` | Erreur sur PF estimée −0.10 à −0.20 sur XAU M15 — **À AJOUTER** |
| B8 | `confluence_score` non calibré (Pearson −0.023) | `src/intelligence/confluence_detector.py` | Tier system invalide — **déjà acté dans confluence_calibration.md, REPLACE prévu** |

---

## 8. Engagement signé

**Le founder s'engage à ne publier aucune statistique de PnL avant que
le walk-forward et le Monte Carlo bootstrap aient été exécutés et
documentés. Toute violation expose l'entreprise à des actions FTC/AMF.**

Date : __________  Signature : __________
