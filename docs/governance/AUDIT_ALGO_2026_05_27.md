# Audit rigoureux de l'algorithme d'analyse du marché

**Date** : 2026-05-27
**Auteur** : second instance Claude Code (audit indépendant via agent Explore)
**Périmètre** : moteur algorithmique uniquement (pas le frontend, pas le packaging)
**Mission** : verdict honnête sur la commerciabilité actuelle de l'algorithme

---

## TL;DR brutal

**Note système globale : 3/10**

Le moteur algorithmique est **techniquement complet mais empiriquement non-rentable**. Le scoring 0-100 affiché au client a **zéro pouvoir prédictif** (Pearson −0.023 sur 924 trades 7 ans). Le backtest 2019-2025 montre **−62 % return cumulé, sous-performance −318 pp vs Buy & Hold XAU**. Les tiers PREMIUM/STANDARD sont cosmétiques (PREMIUM = 1 trade en 7 ans, STANDARD = 0).

**Verdict commerciabilité** :
- ❌ **NON vendable comme "outil de trading profitable"** — risque réputationnel + légal majeur
- ✅ **Vendable comme "outil éducatif d'analyse SMC/macro"** — 9-19 €/mois max, disclaimer renforcé
- ⏸️ **4 blockers critiques** (~11j dev) avant pouvoir réviser ce verdict

---

## 1. Notes par dimension

| Dimension | Statut | Note /10 |
|---|---|---|
| **Architecture pipeline** | Complet (5 briques OK : SMC, Vol, Régime, BOCPD, State Machine) | **7/10** |
| **Pouvoir prédictif scoring** | Pearson −0.023 sur 924 trades, pire qu'une constante | **1/10** |
| **Calibration tiers PREMIUM/STANDARD** | Cosmétique (0-1 trade en 7 ans) | **1/10** |
| **Robustesse tests** | 46 tests Pillar 1-3, tous verts | **8/10** |
| **Backtests rigoureux** | In-sample single-fold, coûts zéro, pas d'IC bootstrap | **2/10** |
| **Qualité données** | Dukascopy 98.4 % coverage XAU + EUR, propre | **8/10** |
| **Commercialisabilité** | Trompeuse si vendue comme "trading", OK si "éducatif" | **2/10** |
| **Code quality** | Propre, modulaire, pas de dépendances cycliques | **7/10** |

---

## 2. Les 4 problèmes critiques chiffrés

### 🔴 Problème #1 — Scoring rule-based sans pouvoir prédictif

**Mode utilisé en prod** : Legacy rule-based (somme pondérée 8 composants calibrés à la main).
**Mode B disponible mais NON WIRED** : LightGBM model `scoring_v3_lgbm.pkl` (81 KB, daté 16/05) existe mais n'est pas chargé par `ConfluenceDetector`.

**Validation empirique du scoring actuel (924 trades, 7 ans relaxed_40)** :

| Métrique | Observé | Cible | Verdict |
|---|---|---|---|
| Pearson(score, r_multiple) | **−0.0232** | > +0.10 | ❌ Zéro corrélation |
| Spearman rank | −0.0162 | > +0.10 | ❌ Idem |
| Brier score | 0.2551 | < 0.225 | ❌ Pire que baseline 0.2456 |
| Win-rate trend par tier | **Décroît** avec score | Croissant | ❌ INVERSE attendu |
| PREMIUM (≥55) | 0 trades / 7 ans | 50-100/an | ❌ Impossible |
| STANDARD (≥40) | 0 trades / 7 ans | 300-1200/an | ❌ Impossible |

**Diagnostic** : la note 0-100 affichée au client est **du bruit déguisé en signal**. Le LLM narrative fait croire à une cohérence inexistante.

### 🔴 Problème #2 — Backtest réel catastrophique

**Backtest 2019-2025 complet** (`reports/audit_2026_04_30_quant_senior.md`) :

| Métrique | XAU M15 | Cible |
|---|---|---|
| Profit Factor | **0.786** | > 1.20 ❌ |
| Win rate | 44.18 % | > 50 % ❌ |
| Sharpe | −0.860 | > 1.0 ❌ |
| Max DD | −77.92 % | < 25 % ❌ |
| Return total | **−62.46 %** | > +50 % ❌ |
| **vs Buy & Hold XAU** | **−318 pp** (BH +256 %, système −62 %) | — | ❌❌ |
| Expectancy | −0.032 R | > +0.1 R | ❌ |

**Décomposition** :
- Longs PF 0.95 (faible)
- Shorts PF 0.636 (89 % des pertes nettes)
- SL hit (13.3 %) : −1.024 R/trade (catastrophe)
- Exit "opposite" cassé : −0.30 R quand déclenché (12.8 % du temps)
- Composants individuels : FVG+retest +0.027R (léger positif), CHOCH/RSI div **anti-prédictifs**

**Les commits récents "VICTORY 5/5 markets" (`e7c9e95`, `05c76fb`)** : pseudo-trades signal-contiguous, **coûts transactionnels NON inclus**, DSR échoue sur 3/5 marchés selon les cavetas honnêtes des commits eux-mêmes.

### 🔴 Problème #3 — Look-ahead bug + coûts à zéro

**Bug B2 (`multi_timeframe_features.py:554-566`)** : swing detector utilise `iloc[i+1], iloc[i+2]` = lookahead. Toute extension MTF est invalidée.

**Coûts transactionnels à zéro** : `state_machine_replay.py` `_build_trade()` ignore le `DynamicSpreadModel` et `DynamicSlippageModel` existants. Coûts réalistes XAU = ~−0.125 R/trade. Quand on les inclut : PF baseline 1.086 → **0.93** (non rentable).

### 🔴 Problème #4 — Pas de validation statistique

- Aucun walk-forward exécuté en prod (les 19 `replay_*.json` sont in-sample single-fold)
- Aucun IC bootstrap calculé sur PF
- Aucun PBO (Probability of Backtest Overfitting)
- Aucun DSR (Deflated Sharpe Ratio)
- Aucune p-value vs random walk

**Conséquence** : aucun chiffre publiable comme "PF 1.30" n'est statistiquement défendable aujourd'hui.

---

## 3. Pourquoi l'architecture est techniquement complète

C'est important de le dire — le travail n'est pas perdu :

✅ **5 briques implémentées et testées (46 tests verts)** :
- `confluence_detector.py` — 625 lignes, 8 composants
- `volatility_forecaster.py` — HAR-RV + diurnal + calendar + HMM (validé 4/5)
- `regime_classifier.py` — HMM 3-état GaussianHMM
- `bocpd.py` — Adams & MacKay 2007, run-length pruning ~80k steps/s
- `signal_state_machine.py` — hysteresis + cooldown + lifetime

✅ **Données propres** : Dukascopy 98.4 % coverage 172 874 bars M15

✅ **InsightSignalV2 v2.1.0** : contrat Pydantic propre, multi-surface

✅ **LLM narrative engine** : Claude Haiku/Sonnet/Opus tier-routed, refus pédagogique scripté

**Le problème n'est pas l'architecture. C'est la fonction de scoring qui doit être remplacée.**

---

## 4. Les 4 blockers critiques (priorité absolue Sprint 1)

### Blocker #1 — Wire scoring v2 LightGBM en prod (5j)

- LightGBM scorer existe (`models/scoring_v3_lgbm.pkl`) mais pas chargé
- Re-entraîner sur target = `r_multiple > 0` avec walk-forward strict
- Isotonic recalibration sur out-of-fold predictions
- ACI wrapper (Gibbs & Candès 2021) déjà implémenté → wrapper actif
- Cible : Brier 0.255 → 0.215 (+16 % info prédictive)
- **Cible Brier skill ≥ +5 % vs baseline naïf**

### Blocker #2 — Brancher coûts transactionnels (3j)

- Wire `DynamicSpreadModel` + `DynamicSlippageModel` dans `_build_trade`
- Coûts réalistes XAU : 5 bps spread + 5 bps slip + ~$7 commission
- Re-run backtest → PF ajusté
- **Sans cela, tous les chiffres affichés au client sont gonflés de 10-20 %**

### Blocker #3 — Bootstrap IC 95 % + walk-forward CPCV (2j)

- 1000 itérations resample avec replacement
- Walk-forward CPCV (López de Prado 2018) k=5 folds purged
- Calculer PBO (Probability of Backtest Overfitting)
- **Aucun chiffre PF affiché tant que pas d'IC publiable**

### Blocker #4 — Patcher look-ahead B2 (1j)

- `multi_timeframe_features.py:554-566` swing detector
- Patch : `shift(+2)` pour rendre causal
- **Sans cela, toute extension MTF est pourrie**

**Total effort blockers : ~11 jours dev**.

---

## 5. Implications stratégiques

### Scénario A — Lancer maintenant (NON recommandé)

- Risque réputationnel majeur : client paie 29 €/mois, perd 62 % en 18 mois
- Risque légal : claims "PF 1.30" + tiers PREMIUM/STANDARD non substanciables = potentielle action AMF Québec / CNIL / Stripe suspension
- Coût : remboursements + procès + marque détruite

### Scénario B — Pivoter en "outil pédagogique" honest (RECOMMANDÉ court terme)

- Repositionner : "Apprenez à lire les marchés SMC + macro" et NON "système de trading profitable"
- Prix : **9-19 €/mois** au lieu de 29 €/mois (cohérent avec valeur réelle)
- Wording : retirer tout "PF 1.30", "profit factor 1.30" tant que pas validé
- Garder : architecture progressive uniforme + chatbot + sources RAG (valeur éducative réelle)
- Disclaimer renforcé : "backtest in-sample non validé OOS, à but pédagogique uniquement"

### Scénario C — Réparer les 4 blockers PUIS lancer (RECOMMANDÉ stratégique)

- Allouer ~11 jours dev sur Sprint 1 (cœur algorithmique du MASTER_PLAN)
- Re-valider empiriquement : Brier skill, IC bootstrap, walk-forward CPCV
- Si Brier skill ≥ +5 % validé : claim 29 €/mois substanciable
- Si Brier skill < +5 % : retour Scénario B (positionnement éducatif)

### Scénario D — Pivot B2B-API (option de secours)

- Cf. `memory/decision_matrix_2026_04_30.md` : pivot B2B-API brokers
- Cible ARR $310k, 80h dev MVP
- Le client B2B (broker, prop shop) accepte un signal "outil de recherche" même sans edge prouvé, parce qu'il l'utilise comme input parmi 50
- Vendre l'output JSON, pas la promesse de profit

---

## 6. Recommandation honnête

**À court terme (cette semaine)** :

1. **Briefer l'autre instance Claude Code en priorité absolue Sprint 1** sur les 4 blockers (Wire LGBM + Costs + Bootstrap + Lookahead) — c'est exactement ce que MASTER_PLAN.md Sprint 1 prévoit. ~11 jours dev.

2. **Mettre en pause toute communication "PF 1.30" / "profit factor"** dans le webapp/mockup/copies en attendant validation Brier skill. Garder les copies actuelles MAIS marquer les chiffres avec tag "in-sample, OOS pending".

3. **Audit final wording produit** : retirer tout claim de performance non validé.

**À moyen terme (M2-M3)** :

4. **Si blockers résolus + Brier skill ≥ +5 %** : continuer Phase 1 vers Gate Final perfection
5. **Si blockers résolus mais Brier skill < +5 %** : pivot positionnement "outil éducatif" à 9-19 €/mois, retirer tiers PREMIUM/STANDARD ou les renommer
6. **Si bloqué techniquement** : pivot B2B-API (Scénario D)

---

## 7. Ce que ce rapport ne dit pas (limites de l'audit)

- L'audit n'a pas pu tester runtime sur prod (just code review + rapports historiques)
- Les commits `e7c9e95 + 05c76fb` annoncent un edge — l'audit suspecte (pseudo-trades, coûts zéro) mais n'a pas pu reproduire intégralement
- L'audit n'a pas évalué le **chatbot conversationnel** comme produit standalone (qui a sa propre valeur éducative)
- L'audit n'a pas évalué la **valeur perçue par client** (UX, narrative LLM, sources RAG cliquables) — ces composants peuvent justifier un prix même sans edge prouvé

**Question ouverte stratégique** : un client paye-t-il 29 €/mois pour "comprendre les marchés via un chatbot et un dashboard pédagogique riches", même si le moteur n'a pas d'edge ? Réponse probable : oui à 9-19 €/mois, peut-être à 29 €/mois si bien marketé "éducation", probablement non à 29 €/mois si marketé "trading profitable".

---

## 8. Décision attendue de l'utilisateur

Choisis ton scénario stratégique :

- [ ] **A** — Lancer maintenant tel quel (déconseillé, risque réputationnel)
- [ ] **B** — Pivoter "éducatif honest" à 9-19 €/mois (rapide, faible risque, ARR plus modeste)
- [ ] **C** — Réparer les 4 blockers (~11j) puis re-évaluer (recommandé stratégique)
- [ ] **D** — Pivot B2B-API brokers (option secondaire si C échoue)

**Le MASTER_PLAN.md Sprint 1 est déjà aligné sur le Scénario C.** L'autre instance Claude Code travaille déjà dessus. Donc la décision concrète attendue est :
- Veux-tu **renforcer la pression sur Sprint 1 blockers** (#1, #2, #3, #4 ci-dessus) ?
- Veux-tu **ajuster les copies/marketing** en attendant les résultats Brier ?

---

**FIN AUDIT.**
