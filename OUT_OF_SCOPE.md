# OUT OF SCOPE — Findings hors périmètre algo

Ce fichier consigne les bugs, anomalies, points d'amélioration **détectés pendant la mission Algo Institutional Overhaul** mais qui tombent hors du périmètre algorithmique (LLM, distribution, compliance, API, infra, etc.).

**Règle** : on log, on continue, on ne corrige pas. Le user décide quoi en faire séparément.

---

## Format d'entrée

```
### YYYY-MM-DD — [zone] — Titre

- **Zone** : api / delivery / llm / compliance / infra / autre
- **Sévérité** : info / mineur / majeur / bloquant
- **Source** : où/quand détecté
- **Description** : 2-3 phrases
- **Suggestion** : (optionnel) action recommandée
```

---

## Log

### 2026-05-15 — [data] — Référence au CSV XAU 2019-2025 (63 % broken) dans scripts hors-scope

- **Zone** : Colab POC + RL legacy + examples
- **Sévérité** : mineur (hors périmètre algo Sprint 0)
- **Source** : audit grep batch 0.4
- **Description** : 14 fichiers hors périmètre algo référencent encore `XAU_15MIN_2019_2025.csv` (CSV à 63 % de coverage). Ces fichiers seront utilisés pour du training Colab ou des démos — pas pour la baseline Sprint 0. À fixer dans un sprint dédié RL/training.
  - `colab_setup.py` (Colab init)
  - `examples/agentic_trading_demo.py` (démo RL)
  - `notebooks/Colab_Full_Training_Script.py` (training Colab)
  - `parallel_training.py:1046, 1320` (RL legacy)
  - `scripts/colab_egarch_tcp_poc.py`, `colab_har_rv_poc.py`, `colab_hybrid_vol_poc.py`, `colab_kronos_poc.py`, `colab_lgbm_vol_poc.py`, `colab_training_full.py` (5 POC scripts)
  - `scripts/download_xau_data.py:44` (downloader, output name — pas critique)
- **Suggestion** : sprint séparé pour migrer les scripts Colab vers `XAU_15MIN_2019_2026.csv` (release v1.1-data sur GitHub Releases nécessaire si on garde l'URL pattern).

### 2026-05-15 — [risk] — `arch` library non installée

- **Zone** : `src/environment/risk_manager.py:14`
- **Sévérité** : mineur
- **Source** : preflight env Batch 0.0
- **Description** : `arch` (GARCH/EGARCH) non installé, fallback vol activé silencieusement avec UserWarning. Le fallback est probablement OK pour la baseline mais à valider Sprint 1/2 quand on auditera le risk engine (eval_19 dit 3 moteurs concurrents).
- **Suggestion** : décider Sprint 5 (refonte unifiée risk) si on installe `arch` ou si on retire le fallback.

### 2026-05-15 — [scripts] — `scripts/audit_data_quality.py` et `scripts/audit_xau_coverage.py` listent encore le CSV cassé

- **Zone** : scripts audit (volontaire)
- **Sévérité** : info
- **Source** : Batch 0.4 grep
- **Description** : Les deux scripts d'audit gardent la référence au CSV 2019_2025 pour permettre la **comparaison** entre coverages (audit utility). C'est intentionnel — pas un bug.

---

**Initialisé** : 2026-05-15
