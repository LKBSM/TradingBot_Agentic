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

### 2026-05-27 — [docs] — `PROJET_VISION_INDICATEUR_CHATBOT.md` contient 2 références obsolètes (nom + pricing)

- **Zone** : doc fondatrice racine
- **Sévérité** : mineur (cohérence éditoriale, pas bloquant fonctionnel)
- **Source** : vérification A pré-Lot 1 cleanup 2026-05-27 (cf. `docs/architecture/CLEANUP_AUDIT_LOT1.md`)
- **Description** : Le document est vivant côté **substance** (Vision B narrative-first, dualité indicateur+chatbot, RAG, B2B porte de sortie) et reste à la racine intouché par le Lot 1. Mais il porte 2 références textuelles obsolètes :
  1. Ancien nom **« Smart Sentinel »** (à rebrand → **« M.I.A. Markets »** per `rebrand_mia_markets_2026_05_26.md`)
  2. Ancienne grille tarifaire **$29 / $79 / $1990 decoy** (devenue **FREE / 9€ / 19€** post-pivot 2026-05-27 per `decisions/2026-05-27_pivot_positioning_audit.md`)
- **Suggestion** : Lot séparé à planifier — `docs(vision): align PROJET_VISION with pivot positioning + M.I.A. rebrand`. Effort ~30 min. À faire **après** les Lot 1 PR1 + PR2 mergés, pour éviter de mélanger les scopes.

---

**Initialisé** : 2026-05-15
