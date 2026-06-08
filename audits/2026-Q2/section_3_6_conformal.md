# Audit Phase 1 — Section 3.6 : ConformalWrapper

**Date** : 2026-05-15
**Auditeur** : Claude
**Périmètre** : `src/intelligence/conformal_wrapper.py` (384 LOC).

---

## Score : **7.0 / 10**

Implémentation institutionnelle propre (Split Conformal + ACI), références académiques solides (Angelopoulos & Bates 2024, Gibbs & Candès 2021, Kato 2024). Le bémol vient de l'utilisation actuelle : le wrapper rejette **TOUT** sur le stack actuel (correct sur weak edge — A1 verdict) ce qui est un constat scientifique honnête mais limite l'impact opérationnel.

---

## 1. Architecture

`src/intelligence/conformal_wrapper.py:1-50` (header) documente :
- **Split Conformal** vanilla (exchangeable assumption).
- **ACI** (Adaptive Conformal Inference) — gère la distribution drift en time series.
- **Reject-option filter** : on trade uniquement si `lo > threshold` (default 0.0 = "non-negative expected return").

Classes :
- `ConformalInterval` — point, lower, upper, alpha, n_calibration.
- `CalibrationSet`
- `TCPForecaster` (Transductive Conformal Prediction)

---

## 2. Strengths

| # | Strength                                                                              |
| - | ------------------------------------------------------------------------------------- |
| S1 | **References académiques explicites** dans le header — auditabilité externe forte.   |
| S2 | **2 variantes** (Split + ACI) — gère l'exchangeabilité et la distribution drift.    |
| S3 | **Reject-option clear** : pas un nouveau signal mais un filtre — bonne séparation des responsabilités. |
| S4 | Tests existants : `test_conformal_wrapper.py`.                                       |
| S5 | Type hints solides (snapshot inventory : 75 %).                                      |

---

## 3. Findings

| # | Finding                                                                                | Sévérité | Action                                |
| - | -------------------------------------------------------------------------------------- | -------- | ------------------------------------- |
| F1 | **PICP (Prediction Interval Coverage Probability) marginale + conditionnelle non mesurée empiriquement** sur le baseline Sprint 0. | P0       | Sprint 4 batch 4.1                    |
| F2 | **MPIW (Mean Prediction Interval Width)** non rapporté.                               | P1       | Sprint 4                              |
| F3 | **Exchangeabilité** non testée formellement (KS test, runs test).                     | P0       | Sprint 4                              |
| F4 | **Sur stack actuel : conformal rejette TOUT** (A1 verdict : "weak edge, conformal rejette tout (correct)"). C'est scientifiquement honnête mais opérationnellement : 0 trade. | P0       | Sprint 3 (edge discovery) doit précéder |
| F5 | **Pas de variante Mondrian** (conformal stratifié par régime). Avec 2 régimes vol distincts, exchangeabilité est plus crédible intra-régime. | P1       | Sprint 4 batch 4.1                    |
| F6 | **ACI hazard / window size** : magic numbers à calibrer.                              | P2       | Sprint 4                              |
| F7 | **Quel `outcome` est utilisé** pour calibrer ? "Réalisation post-signal_lifetime_bars" (header line 33). Si signal_lifetime_bars dérive (config change), la calibration historique devient invalide. | P1 | Sprint 4 |

---

## 4. Gain réel quantifié

**A1 verdict 2026-05-01** : conformal applied to stack RL → tout rejeté car edge insuffisant.

**3-pillars implementation 2026-05-13** : conformal sur event-driven macro → idem, conforme à un weak edge.

**Sprint 0 baseline (cette mission)** : 0 trades naturellement (score plafonne < 75) → conformal n'a même pas matière à filtrer.

**Conclusion** : le ConformalWrapper est **prêt à fonctionner**, mais il a besoin d'un edge prédictif source pour démontrer sa valeur. Sprint 3 (edge discovery) doit précéder Sprint 4 (calibration & confidence).

---

## 5. Couverture conformelle cible (gate Sprint 4)

| Métrique               | Cible       | Statut actuel |
| ---------------------- | ----------- | ------------- |
| PICP marginale         | nominal ± 2 % sur OOS | ❓ non mesuré |
| PICP conditionnelle (par régime) | nominal ± 5 % sur OOS | ❓ non mesuré |
| MPIW                   | minimisé sous contrainte de coverage | ❓ non mesuré |
| Exchangeabilité (KS test) | p > 0.05 | ❓ non testé |

---

## 6. Recommandations

| Sprint | Action                                                                       | Priorité |
| ------ | ---------------------------------------------------------------------------- | -------- |
| 4.1    | Refonte avec **Mondrian conformal** stratifié par régime (HMM ou BOCPD)    | P0       |
| 4.2    | Mesure empirique PICP + MPIW out-of-sample                                  | P0       |
| 4.3    | Test exchangeabilité formelle (KS + runs)                                   | P1       |
| 4.4    | Documentation client-facing (ce que le conformal garantit, ce qu'il ne dit pas) | P0       |

---

## 7. Ce que cet audit ne couvre pas

- **Validation empirique** des garanties théoriques sur les données Sprint 0 (impossible sans trades).
- **Comparaison vs autres approches** (quantile regression, isotonic conformal).
- **Performance / latence** (probable négligeable mais à mesurer Sprint 4).

---

**Signé** : 2026-05-15, Claude
