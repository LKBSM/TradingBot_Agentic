# Audit Phase 1 — Section 3.5 : Régime Stack

**Date** : 2026-05-15
**Auditeur** : Claude
**Périmètre** : `src/intelligence/regime_classifier.py` (210), `regime_filter.py` (189), `regime_gate.py` (335), `bocpd.py` (275) + composantes côté `agents/` (legacy).

---

## Score : **6.5 / 10**

Stack riche, deux générations bien identifiées (legacy RL + institutional pilier 3), mais fragmentation = dette technique non négligeable.

---

## 1. Inventaire des 6 implémentations parallèles

| # | Fichier                                       | LOC   | Rôle                                                 | Statut Sprint 0 (décision D) |
| - | --------------------------------------------- | ----- | ---------------------------------------------------- | ----------------------------- |
| 1 | `src/intelligence/regime_filter.py`           | 189   | Gate session NY + ATR percentile (empirique +0.22 PF) | ✅ **CANONIQUE**              |
| 2 | `src/intelligence/regime_gate.py`             | 335   | BOCPD + Bipower jumps (pilier 3 institutionnel)      | ✅ **CANONIQUE**              |
| 3 | `src/intelligence/bocpd.py`                   | 275   | Bayesian Online Changepoint Detection                | ✅ **CANONIQUE**              |
| 4 | `src/intelligence/regime_classifier.py`       | 210   | HMM 3-state (low/normal/high vol)                    | 🟡 Utilitaire bas-niveau (gardé) |
| 5 | `src/agents/market_regime_agent.py`           | 887   | Détection régime technique (trend/vol/S-R)           | 🔒 LEGACY figé (RL era)       |
| 6 | `src/agents/regime_predictor.py`              | 1 051 | HMM + regime switching                               | 🔒 LEGACY figé (RL era)       |

---

## 2. Audit `regime_filter.py` (canonique)

### Strengths
- **Empiriquement validé** : commentaires de header (`reports/feature_filter_audit.md` chantier 1+3, 2026-04-29) — XAU M15 7 ans, PF 1.13 → 1.30 OOS avec `ny_mode="high_vol"` + `vol_pctl_max=0.75`.
- **3 modes documentés** : "off", "all", "high_vol" — le default `high_vol` est le choix pareto.
- **Tunables env-var** : `vol_pctl_max`, `vol_window_bars`, `vol_min_periods`.
- **Test coverage** : `test_regime_filter.py` (à valider précisément Sprint 1).

### Findings

| # | Finding                                                                             | Sévérité |
| - | ----------------------------------------------------------------------------------- | -------- |
| F1 | **Session NY hardcodée UTC 13-21** (lines 34-35). Devrait être InstrumentConfig param (FX vs Crypto diffèrent). | P1 |
| F2 | **Pas de gate Tokyo / London** symétrique. Asymétrie justifiée pour XAU mais à confirmer EURUSD/USDJPY. | P1 |
| F3 | Empirique XAU uniquement. Pas de validation cross-actifs documentée. | P2 |
| F4 | `vol_pctl_max=0.75` est un magic number. Pas de CV documenté. | P2 |

---

## 3. Audit `regime_gate.py` (canonique — pilier 3)

### Strengths
- **Architecture sain** : 2 détecteurs indépendants (BOCPD changepoint + Bipower jumps) qui votent.
- **3-state decision** : TRADE / REDUCE / BLOCK — déclaratif et auditable.
- **Références académiques** : Adams & MacKay 2007 (BOCPD), Barndorff-Nielsen & Shephard 2004 (Bipower), Corsi 2009 (HAR-RV).
- **Complémentaire HMM** : HMM = "we ARE in high vol", BOCPD = "we are ENTERING a new regime".

### Findings

| # | Finding                                                                          | Sévérité |
| - | -------------------------------------------------------------------------------- | -------- |
| F5 | **Seuils défaut hardcodés** : `regime_block_threshold=0.30`, `jump_block_threshold=0.40`. Pas de justification empirique (pilier 3 récent — 2026-05-13). | P0 |
| F6 | Impact réel sur PF non quantifié. Le `3_pillars_implementation_2026_05_13.md` dit "Regime gate +0.16 DSR mais insuffisant" — c'est faible. | P0 |
| F7 | Pas de `REDUCE` côté state machine (le gate émet REDUCE mais qui consomme l'info ?). | P1 |
| F8 | BOCPD hazard rate (`DEFAULT_HAZARD_INV`) — valeur à auditer. | P2 |

---

## 4. Audit `bocpd.py` (utilitaire pur)

### Strengths
- Code core BOCPD propre, vectorisé. Tests : `test_bocpd.py`.

### Findings

| # | Finding                                                                          | Sévérité |
| - | -------------------------------------------------------------------------------- | -------- |
| F9 | Pas d'audit de **stabilité numérique** sur séries longues (overflow log-probs). | P2 |
| F10 | Pas de comparaison vs scipy reference implementation.                            | P3       |

---

## 5. Audit `regime_classifier.py` (HMM)

### Findings

| # | Finding                                                                          | Sévérité |
| - | -------------------------------------------------------------------------------- | -------- |
| F11 | HMM `predict()` typique : refit at every call ? (bug B1 mentionné eval_04). À vérifier précisément. | P0 |
| F12 | 3 états (low/normal/high vol) hardcoded. Pas de model selection (BIC) documenté. | P2 |

---

## 6. Audit agents/regime_* (LEGACY)

### Décision D actée
Ces 2 fichiers (887 + 1 051 LOC) sont **figés** pendant Sprint 0-7. Pas d'audit profond pendant Sprint 0.

### Findings flag-only

| # | Finding                                                                                      | Sévérité |
| - | -------------------------------------------------------------------------------------------- | -------- |
| F13 | **2 000 LOC dead code** potentiellement — à confirmer Sprint 1.                              | P1       |
| F14 | Potentielle réutilisation (features HMM, predictor switching) que la nouvelle stack pourrait absorber. Audit Sprint 6 (production hardening). | P2 |

---

## 7. Stabilité des états HMM (test empirique requis)

**Non mesuré en Sprint 0** — à exécuter Sprint 1 :
- Nombre de transitions HMM sur 7 ans XAU.
- Durée moyenne des régimes (régime stable doit durer ≥ 50 bars).
- Cohérence sémantique vs labels manuels (crise 2020, LDI 2022, SVB 2023, yen 2024).

---

## 8. Impact réel du RegimeGate sur PF

D'après mémoire (3-pillars-implementation_2026_05_13) :
- DSR delta : +0.16 (insuffisant pour passer gate DSR=0.65).
- Sur 329 trades event-driven, le RegimeGate filtre une partie mais l'edge underlying est trop faible.

**Conclusion** : le RegimeGate **fait son travail** (filtre les régimes pathologiques) mais ne peut pas créer un edge à partir du néant. Si le ConfluenceDetector (eval_02) n'a pas de pouvoir prédictif, le gate ne sauve pas.

---

## 9. Recommandations

| Sprint | Action                                                                | Priorité |
| ------ | --------------------------------------------------------------------- | -------- |
| 1      | Acter formellement décision D (canonique vs legacy)                   | P0       |
| 1      | Calibration empirique des seuils RegimeGate (F5, F6)                  | P0       |
| 1      | Cross-actifs validation `regime_filter` (F3)                          | P1       |
| 2      | Stabilité HMM mesurée (transitions, durée régimes)                    | P0       |
| 5      | REDUCE state consumé par state machine (F7)                           | P1       |
| 6      | Audit / archive du legacy agents/regime_* (F13)                       | P2       |

---

## 10. Ce que cet audit ne couvre pas

- Audit empirique **stabilité HMM** sur 7 ans.
- **Comparaison vs labels manuels** d'expert macro.
- **Sensibilité** des seuils à ±20 %.

---

**Signé** : 2026-05-15, Claude
