# Audit pré-archivage — Lot 1 Cleanup

**Date** : 2026-05-27
**Statut** : audit livré, attente validation utilisateur avant tout `git mv`
**Scope** : RL legacy stack + pollution racine .md
**Méthode** : `git log -1` par fichier · `wc -l` · `grep -r` sur l'arborescence pour les imports croisés.

---

## 1. Liste exhaustive — fichiers à archiver (code)

### 1.1 Stack RL pure (zéro doute, tous traçables RL-era)

| Fichier | Lignes | Dernier commit | Description courte |
|---|---:|---|---|
| `parallel_training.py` | 1 634 | 2026-03-06 `2c545aa` | Orchestrateur RL multi-process : entraîne N bots PPO en parallèle, walk-forward, sélection champion. Entry-point legacy `python parallel_training.py`. |
| `src/training/__init__.py` | 36 | 2026-01-30 `ccfd3f8` | Package init du sous-système d'entraînement RL. |
| `src/training/advanced_reward_shaper.py` | 563 | 2026-02-27 `936907b` | Multi-objective reward shaping (Sharpe + drawdown + WR + flat penalty + entropy alignment). |
| `src/training/checkpoint_manager.py` | 421 | 2026-02-27 `936907b` | Sauvegarde/reprise checkpoints PPO (Sprint 4). |
| `src/training/curriculum_trainer.py` | 697 | 2026-03-27 `390086f` | Curriculum progressif par phases de difficulté. |
| `src/training/ensemble_trainer.py` | 666 | 2026-01-30 `ccfd3f8` | Diversité de modèles + sélection ensemble. |
| `src/training/ewc_regularization.py` | 247 | 2026-03-16 `4c4377f` | Elastic Weight Consolidation (Sprint 14) anti-oubli catastrophique. |
| `src/training/meta_learner.py` | 742 | 2026-01-30 `ccfd3f8` | Meta-learning + adaptation rapide aux régimes. |
| `src/training/sophisticated_trainer.py` | 1 123 | 2026-02-27 `936907b` | Orchestrateur maître RL (curriculum + ensemble + meta + EWC). |
| `src/training/unified_agentic_env.py` | 731 | 2026-03-21 `ce848b7` | Env Gymnasium unifié pour PPO (observation constante, multi-mode). |
| `src/agents/integration.py` | 768 | 2026-02-27 `936907b` | Wrapper Gymnasium des agents → `AgenticTradingEnv` PPO. |
| `src/agents/intelligent_integration.py` | 721 | 2026-02-27 `936907b` | Variante ML-powered (RiskPredictor + AdaptiveSizing). |
| `src/agents/orchestrated_integration.py` | 919 | 2026-02-27 `936907b` | Multi-agent coordination en environnement RL. |
| `src/agents/risk_integration.py` | 1 101 | 2026-03-16 `4c4377f` | `IntegratedRiskManager` v1 (Sprint 1 RL : Portfolio + KillSwitch + Audit + Sentinel). |
| **Sous-total RL stack pure** | **10 369** | | |

### 1.2 Modules dérivés (importent la stack RL — à archiver en cascade)

| Fichier | Lignes | Dernier commit | Description courte |
|---|---:|---|---|
| `src/agent_trainer.py` | 863 | 2026-03-27 `390086f` | Wrapper `AgentTrainer` qui instancie `SophisticatedTrainer`. Entry-point legacy `python -m src.agent_trainer`. |
| `src/weekly_adaptation.py` | 743 | 2025-12-13 `a863bc1` | Job hebdomadaire fine-tuning PPO sur dernière semaine de data. Importe `agent_trainer`. |
| `colab_setup.py` | 406 | 2026-03-06 `2c545aa` | Bootstrap Google Colab pour `parallel_training.py`. |
| `notebooks/Colab_Full_Training_Script.py` | 374 | 2026-03-06 `2c545aa` | Notebook Colab full training (importe `sophisticated_trainer`). |
| `scripts/colab_training_full.py` | 1 088 | 2026-03-27 `e1b052f` | Script Colab v2 (importe `unified_agentic_env`, `curriculum_trainer`, etc.). |
| **Sous-total dérivés** | **3 474** | | |

### 1.3 Tests RL-era (cassent à la collection si on archive sans eux)

| Fichier | Lignes | Dernier commit | Description courte |
|---|---:|---|---|
| `tests/test_sprint1_risk.py` | 1 066 | 2026-03-16 `4c4377f` | Tests `IntegratedRiskManager` (importe `src.agents.risk_integration`). |
| `tests/test_sprint13_entropy_alignment.py` | 118 | 2026-03-16 `4c4377f` | Tests entropy alignment RL (importe `src.training.curriculum_trainer/unified_agentic_env/advanced_reward_shaper`). |
| `tests/test_sprint14_ewc.py` | 220 | 2026-03-16 `4c4377f` | Tests EWC (importe `src.training.ewc_regularization`). |
| `tests/test_training_pipeline.py` | 456 | 2026-02-27 `936907b` | Tests pipeline sophistiqué (stubs en MagicMock pour éviter coût SB3). |
| `tests/test_v4_dsr_changes.py` | 513 | 2026-03-27 `390086f` | Tests DSR sur courbe RL v4. |
| `tests/test_v5_action_masking.py` | 411 | 2026-03-27 `390086f` | Tests action-masking RL v5. |
| `tests/test_v6_flat_penalty.py` | 511 | 2026-03-27 `390086f` | Tests flat-penalty RL v6. |
| **Sous-total tests** | **3 295** | | |

### 1.4 Total code à archiver

**17 138 lignes** réparties sur **26 fichiers**, dernier commit le plus récent : 2026-03-27 (≥ 2 mois sans toucher, parfaitement gelé).

---

## 2. Cartographie des imports croisés

### 2.1 Production canale (à conserver) → modules archivés : **ZÉRO import**

Vérifié par `grep` exhaustif :
- `src/intelligence/` → **aucun import** vers `parallel_training`, `src.training`, `src.agents.*_integration`, `src.agent_trainer`, `src.weekly_adaptation`. ✅
- `src/api/` → **aucun import**. ✅
- `src/delivery/` → **aucun import**. ✅

**Conclusion** : le pipeline de production (Sentinel + API + delivery) est entièrement découplé de la stack RL. Archivage 100 % safe sur ces canaux.

### 2.2 Modules à conserver qui touchent la stack RL — à patcher avant `git mv`

| Fichier conservé | Ligne | Type d'import | Traitement requis |
|---|---:|---|---|
| `src/agents/__init__.py` | 55-63 | `try: from src.agents.integration import ...` (wrappé) | Retirer le bloc try/except — déjà silencieux, mais réduit le bruit logger. **Patch obligatoire**. |
| `src/agents/__init__.py` | 116-… | `try: from src.agents.intelligent_integration import ...` | Idem — retirer bloc. |
| `src/agents/__init__.py` | 174-… | `try: from src.agents.orchestrated_integration import ...` | Idem — retirer bloc. |
| `src/agents/__init__.py` | 296-… | `try: from src.agents.risk_integration import ...` | Idem — retirer bloc. |
| `src/environment/strategy_features.py` | 1134 | `from src.agent_trainer import AgentTrainer` (dans `__main__` only) | **No-op** : exécuté uniquement si `python strategy_features.py`. Retirer le bloc `__main__` entier (script de démo RL-era, plus utile). |

**Total patches** : 2 fichiers à éditer après le `git mv` (5 changements localisés).

### 2.3 Imports entre fichiers archivés (cohérence intra-archive)

Confirmé : `agent_trainer.py` → `src.training.*`, `weekly_adaptation.py` → `agent_trainer`, `colab_*` → `src.training.*`, `parallel_training.py` autonome. Tous les imports restent **internes** au scope archivé → cohérent.

### 2.4 Référence string non-import (no-op pour archivage)

`src/weekly_adaptation.py:182` : message d'erreur `"Run parallel_training.py first."` — string only, archivé avec le reste donc OK.

---

## 3. Tri des 24 fichiers .md racine en 3 catégories

### 3.1 Racine (à garder visible — justifié)

| Fichier | Lignes | Justification |
|---|---:|---|
| `README.md` | 15 | Repo entry-point. À mettre à jour Vision B mais reste visible. |
| `CHANGELOG.md` | 38 | Algo changelog actif, dernière entrée 2026-05-15. |
| `PROJET_VISION_INDICATEUR_CHATBOT.md` | 45 | **Document fondateur Vision B narrative-first**. Source canonique de la dualité indicateur+chatbot. Doit rester visible. |

**3 fichiers, 98 lignes.**

### 3.2 `docs/archive/` (historique précieux à conserver classifié)

Documents à valeur historique ou méthodologique réutilisable. Préservés mais sortis de la racine pour ne pas polluer.

| Fichier | Lignes | Dernier commit | Pourquoi préserver |
|---|---:|---|---|
| `BUSINESS_PLAN_SMART_SENTINEL.md` | 533 | 2026-04-23 | Business plan pré-pivot positioning du 2026-05-27. Trail historique du repositionnement. |
| `MISSION_ACK.md` | 137 | 2026-05-15 | Mission Algo Institutional Overhaul (institutional-overhaul branch). Contexte branche active. |
| `MISSION_ACK_NAMING.md` | 33 | 2026-05-26 | Rebrand M.I.A. Markets ack. Historique récent du rename. |
| `OUT_OF_SCOPE.md` | 56 | 2026-05-15 | Seed du out-of-scope (versions plus complètes dans `docs/governance/` et `docs/audits/`). Préserver comme premier jet. |
| `EVALUATION_PROMPTS.md` | 706 | non-tracked | Source-of-truth des 29 prompts d'évaluation sectorielle. Réutilisable pour futurs audits. |
| `GO_NO_GO_PROMPT.md` | 300 | non-tracked | Template de prompt audit go/no-go. Réutilisable. |
| `BACKTEST_LEGAL_GUARDRAILS.md` | 238 | non-tracked | Garde-fous légaux backtest (Eval 18 K10+K11). Référence active. |
| `FORWARD_TEST_AND_GTM.md` | 212 | 2026-04-29 | Playbook forward-test + GTM. Contenu utile pour Phase D commerciale. |
| `PLAN.md` | 320 | 2026-04-23 | Plan "Claude as Primary Analyst" — concepts repris dans Vision B. |
| `COMPLETE_PROJECT_DOCUMENTATION.md` | 1 073 | 2026-04-23 | Snapshot architecture/runtime à date. Trail de la transition. |
| `COMPREHENSIVE_EVALUATION_REPORT.md` | 356 | 2026-04-23 | Rapport profitabilité XAU M15 — utile pour benchmarks comparatifs. |
| `INSTITUTIONAL_AUDIT_REVIEW.md` | 1 055 | 2026-02-27 | Audit institutionnel due-diligence — méthodologie réutilisable (DSR/PBO/CPCV). |

**12 fichiers, 5 019 lignes.**

### 3.3 `_archive/2026-05-27_pivot_vision_b/` (obsolète RL-era — à enterrer)

Contenu factuellement obsolète post-pivot Vision B. Aucune valeur réutilisable côté Vision B sauf comme contexte historique brut.

| Fichier | Lignes | Dernier commit | Raison |
|---|---:|---|---|
| `AGENTS_SYSTEM_ANALYSIS.md` | 864 | 2026-01-22 | Analyse agentic multi-agent RL. Plus pertinent post-pivot narrative-first. |
| `ANALYSE_COMPLETE_SYSTEME.md` | 1 211 | 2026-01-30 | Audit production capitaux institutionnels RL FR. |
| `ANALYSE_TRAINING_SYSTEM.md` | 337 | 2026-01-30 | Analyse système d'entraînement PPO. |
| `COMMERCIALIZATION_REPORT.md` | 7 411 | 2026-02-27 | Rapport commercialisation TradingBOT Agentic — concepts dépassés. |
| `COMPREHENSIVE_PROJECT_GUIDE.md` | 4 833 | 2026-01-22 | Guide complet TradingBOT_Agentic — obsolète. |
| `SPRINT_PLAN.md` | 619 | 2026-01-22 | Plan sophistication RL (sprints 1-15). |
| `SPRINT_ROADMAP.md` | 1 354 | 2026-02-27 | Roadmap 15-sprints RL (6.0→9.0). |
| `SPRINT3_READINESS_AUDIT.md` | 460 | 2026-01-22 | Audit Sprint 3 RL. |
| `TRAINING_ARCHITECTURE.md` | 543 | 2026-02-27 | Architecture technique training bot RL. |

**9 fichiers, 17 632 lignes.**

### 3.4 Total .md à déplacer

**21 fichiers / 22 651 lignes** sortent de la racine. **3 fichiers / 98 lignes** restent visibles. Réduction pollution racine : **~99,6 % en lignes**.

---

## 4. Note de réutilisabilité — scénario D B2B-API (M+6)

Sur les **17 138 lignes de code archivées**, voici ce qui pourrait servir si pivot B2B-API broker :

| Module archivé | Réutilisable B2B ? | Détails |
|---|---|---|
| `src/agents/risk_integration.py` | ⚠️ **PARTIELLEMENT** (~300 L) | L'API unifiée `IntegratedRiskManager` (PortfolioRiskManager + KillSwitch + AuditLogger) reste pertinente pour exposer `risk_score` + `kill_decision` à des partenaires. **MAIS** : les composants amont (`portfolio_risk`, `kill_switch`, `audit_logger`) sont déjà en dehors du scope archivé (dans `src/agents/`). Donc seule la couche d'orchestration de 300 L est concernée — refaisable propre en < 4 h. Verdict : **ne pas tenter de récupérer, refaire**. |
| `src/training/meta_learner.py` | ⚠️ **CONCEPTS** seulement | `MetaLearnerConfig`, notion de "regime adaptation rapide". Les concepts (HMM regime + adaptive sizing) sont déjà ré-implémentés ailleurs (`src/intelligence/regime_gate.py`). **Pas de code à récupérer.** |
| `src/training/curriculum_trainer.py` | ❌ | Curriculum RL spécifique PPO. Hors-scope B2B. |
| `src/training/ensemble_trainer.py` | ❌ | Ensemble RL → remplacé par calibrated conviction (LGBM+Isotonic). |
| `src/training/advanced_reward_shaper.py` | ❌ | Reward shaping PPO. N'a aucun sens hors RL. |
| `src/training/ewc_regularization.py` | ❌ | EWC PPO anti-oubli. N/A. |
| `src/training/sophisticated_trainer.py` | ❌ | Orchestrateur RL. N/A. |
| `src/training/unified_agentic_env.py` | ❌ | Env Gymnasium. N/A. |
| `src/training/checkpoint_manager.py` | ❌ | Checkpoint PPO. Remplacé par MLflow / artefacts standard. |
| `src/agents/integration.py` | ❌ | Wrapper Gymnasium. N/A. |
| `src/agents/intelligent_integration.py` | ❌ | Variante RL. N/A. |
| `src/agents/orchestrated_integration.py` | ❌ | Multi-agent RL. N/A. |
| `src/agent_trainer.py` | ❌ | Wrapper RL. N/A. |
| `src/weekly_adaptation.py` | ❌ | Fine-tuning PPO. N/A. |
| `parallel_training.py` | ❌ | Orchestrateur multi-process PPO. N/A. |
| `colab_setup.py` + `notebooks/Colab_*` + `scripts/colab_training_full.py` | ❌ | Bootstrap Colab RL. N/A. |
| `tests/test_v{4,5,6}_*.py`, `test_sprint{1,13,14}_*.py`, `test_training_pipeline.py` | ❌ | Tests RL. N/A. |

### Verdict réutilisabilité

**Aucun module à exfiltrer avant archivage.** Le seul concept partiellement réutilisable (couche d'orchestration risk dans `risk_integration.py`) est triviale à refaire propre — moins de coût que de la maintenir vivante.

**Mémo pour future-Claude** (consigné dans `_archive/README.md`) : si pivot B2B-API à M+6, ne pas chercher à dépoussiérer la stack RL. Les concepts utiles (VaR, kill-switch, audit log) sont dans `src/agents/portfolio_risk.py`, `src/agents/kill_switch.py`, `src/agents/audit_logger.py` — qui restent dans le repo actif.

---

## 5. Récapitulatif pré-`git mv`

### 5.1 Volume final

| Catégorie | Fichiers | Lignes |
|---|---:|---:|
| Code RL → `_archive/2026-05-27_pivot_vision_b/` | 26 | 17 138 |
| .md RL-era → `_archive/2026-05-27_pivot_vision_b/` | 9 | 17 632 |
| .md historiques → `docs/archive/` | 12 | 5 019 |
| .md gardés racine | 3 | 98 |
| **Total déplacé** | **47** | **39 789** |

### 5.2 Patches collatéraux requis (post-`git mv`)

1. `src/agents/__init__.py` — retirer 4 blocs try/except vers `*_integration` archivés.
2. `src/environment/strategy_features.py` — retirer le bloc `if __name__ == '__main__'` (script démo RL).
3. **Pas d'autre changement** dans le code de prod.

### 5.3 Tests à valider après le mv + patches

```bash
pytest tests/ --ignore=tests/test_sprint1_risk.py \
              --ignore=tests/test_sprint13_entropy_alignment.py \
              --ignore=tests/test_sprint14_ewc.py \
              --ignore=tests/test_training_pipeline.py \
              --ignore=tests/test_v4_dsr_changes.py \
              --ignore=tests/test_v5_action_masking.py \
              --ignore=tests/test_v6_flat_penalty.py \
              -x --tb=short
```

**Gate** : ≥ 1 200 tests verts, 0 ImportError nouveau, 0 régression sur les suites Intelligence/API/Delivery.

### 5.4 Convention d'archivage

Voir `_archive/README.md` (créé en bonus). Convention :
- Sous-dossier daté : `_archive/<YYYY-MM-DD>_<motif-court>/`
- Préserve structure d'origine (`src/training/...` reste `src/training/...` sous le sous-dossier).
- `.md` companion en tête : raison de l'archivage + contexte pivot.

---

## 6. Décision attendue de l'utilisateur

1. **Valider la liste 1.1+1.2+1.3** : 17 138 L code RL — OK pour archivage en cascade ?
2. **Valider la classification 3.1/3.2/3.3** : 3 fichiers racine / 12 → `docs/archive/` / 9 → `_archive/`. OK ?
3. **Valider l'absence d'exfiltration B2B** (Note §4) : confirmer que `risk_integration.py` n'est pas à récupérer.
4. **OK pour patcher** `src/agents/__init__.py` et `src/environment/strategy_features.py` dans le même PR ?

Une fois ces 4 points validés, exécution du `git mv` + patches + tests + PR `chore(repo): archive RL legacy stack + clean root pollution (Lot 1)`.
