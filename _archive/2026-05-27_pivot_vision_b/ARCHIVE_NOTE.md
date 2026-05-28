# Archive — 2026-05-27 · Pivot Vision A → B

**Date d'archivage** : 2026-05-27
**Branche** : `chore/archive-rl-legacy` (sera mergée dans `main` après PR1)
**PR de référence** : `chore(repo): archive RL legacy stack (Lot 1 PR1)`
**Audit de pré-archivage** : `docs/architecture/CLEANUP_AUDIT_LOT1.md`
**Convention** : `_archive/README.md`

## Contexte du pivot

Vision A (origine 2025-Q4 → 2026-Q1) : **bot de trading RL autonome**.
- Stack : PPO + multi-agent + curriculum + ensemble + meta-learning + EWC.
- Repo : `TradingBOT_Agentic`.
- Verdict A1 du 2026-05-01 : DSR=0, PBO=0.50, CPCV PF=1.008, score 1/6. **Aucun edge prédictif**.

Vision B (formalisée 2026-05-27) : **M.I.A. Markets — indicateur de marché conversationnel**.
- Pipeline déterministe `SmartMoneyEngine → ConfluenceDetector → CalibratedConviction → InsightAssembler` + chatbot Sentinel.
- L'IA explique, **ne décide jamais**. Aucune exécution.
- Compliance UE 2024/2811 + posture éducative.

Documents de référence : `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md`, `docs/governance/AUDIT_ALGO_2026_05_27.md`, `docs/architecture/MIA_MARKETS_ARCHITECTURE.md`, `PROJET_VISION_INDICATEUR_CHATBOT.md`.

## Contenu de cet archive (32 fichiers code/tests, ~19 316 lignes)

**Note d'audit (transparence)** : l'audit pré-archivage initial (`docs/architecture/CLEANUP_AUDIT_LOT1.md` §1.3) listait **7 tests RL-era** (3 295 L). 6 fichiers supplémentaires ont été détectés en 2 vagues car ils utilisent des patterns non-standard que mon `grep "from src.training.X import"` initial ne matche pas :

- `tests/test_checkpoint_manager.py` (410 L) — `importlib.util.spec_from_file_location()` au top-level → ImportError à la collection
- `tests/test_walk_forward.py` (433 L) — `from parallel_training import ...` au top-level → ImportError à la collection (le grep a anormalement omis ce match)
- `tests/test_feature_reducer.py` (378 L) — lit `src/training/sophisticated_trainer.py` via `open().read()` dans `TestSourceVerification`
- `tests/test_hyperparameters.py` (337 L) — idem (`sophisticated_trainer.py` + `curriculum_trainer.py`)
- `tests/test_reward_function.py` (523 L) — idem (`sophisticated_trainer.py` dans `TestSourceCodeVerification`)

Tous ces 6 tests **ne testent QUE du code archivé** (source-verification de constantes RL/Colab ou exécution PPO). Aucune valeur résiduelle post-pivot Vision B → archivés avec le reste. Décision et expansion de scope validées par l'utilisateur (2 approbations successives le 2026-05-27 puis 2026-05-28).

## Contenu détaillé (32 fichiers, ~19 316 lignes)

### Stack RL pure (10 369 L)

- `parallel_training.py` — orchestrateur multi-process PPO walk-forward.
- `src/training/__init__.py` — package init.
- `src/training/advanced_reward_shaper.py` — multi-objective reward shaping.
- `src/training/checkpoint_manager.py` — checkpoints PPO (Sprint 4).
- `src/training/curriculum_trainer.py` — curriculum progressif phases difficulté.
- `src/training/ensemble_trainer.py` — diversité de modèles + sélection ensemble.
- `src/training/ewc_regularization.py` — EWC anti-oubli (Sprint 14).
- `src/training/meta_learner.py` — meta-learning + adaptation régime.
- `src/training/sophisticated_trainer.py` — orchestrateur maître RL.
- `src/training/unified_agentic_env.py` — env Gymnasium unifié PPO.
- `src/agents/integration.py` — `AgenticTradingEnv` PPO wrapper.
- `src/agents/intelligent_integration.py` — variante ML-powered.
- `src/agents/orchestrated_integration.py` — multi-agent coordination RL.
- `src/agents/risk_integration.py` — `IntegratedRiskManager` v1 Sprint 1 RL.

### Dérivés (3 474 L)

- `src/agent_trainer.py` — wrapper `AgentTrainer` qui instancie `SophisticatedTrainer`.
- `src/weekly_adaptation.py` — job hebdo fine-tuning PPO.
- `colab_setup.py` — bootstrap Colab pour `parallel_training.py`.
- `notebooks/Colab_Full_Training_Script.py` — notebook training Colab.
- `scripts/colab_training_full.py` — script Colab training v2.

### Tests RL-era (13 fichiers, 5 473 L · 304 tests collectés)

**Bloc initial (audité pré-PR1, 7 fichiers, 3 295 L, 159 tests)**

- `tests/test_sprint1_risk.py` (1 066 L, 48 tests)
- `tests/test_sprint13_entropy_alignment.py` (118 L, 5)
- `tests/test_sprint14_ewc.py` (220 L, 7)
- `tests/test_training_pipeline.py` (456 L, 22)
- `tests/test_v4_dsr_changes.py` (513 L, 30)
- `tests/test_v5_action_masking.py` (411 L, 19)
- `tests/test_v6_flat_penalty.py` (511 L, 28)

**Bloc additionnel #1 (détecté post-`git mv` à la collection pytest, 5 fichiers, 2 081 L, 138 tests)**

- `tests/test_checkpoint_manager.py` (410 L, ~25 tests) — chargeait `src/training/checkpoint_manager.py` via `spec_from_file_location` au top-level → ImportError bloquant à la collection
- `tests/test_walk_forward.py` (433 L, ~19 tests) — `from parallel_training import ...` au top-level → ImportError bloquant
- `tests/test_feature_reducer.py` (378 L, 31 tests dont 1 `TestSourceVerification` qui lit `sophisticated_trainer.py`)
- `tests/test_hyperparameters.py` (337 L, 34 tests dont `TestSourceVerification` qui lit `sophisticated_trainer.py` + `curriculum_trainer.py`)
- `tests/test_reward_function.py` (523 L, 29 tests dont `TestSourceCodeVerification::test_ent_coef_is_001` qui lit `sophisticated_trainer.py`)

**Bloc additionnel #2 (détecté à l'exécution pytest gate, 1 fichier, 97 L, 7 tests)**

- `tests/test_sprint15_colab_pinning.py` (97 L, 7 tests SEC-1 — VERIFIED_COMMIT, shallow clone, checksum verification, hashlib) — lit `scripts/colab_training_full.py` (archivé). Mon sweep grep initial cherchait `src/training/*` mais omettait les paths `scripts/colab_*`. Toutes 7 failures résolues par cet archivage. Pattern identique au bloc #1 (source-verification d'un fichier archivé → valeur résiduelle nulle).

**Garantie (vérifiée 2026-05-27)** : ZÉRO de ces 12 fichiers n'importe `src/intelligence/`, `src/api/`, `src/delivery/`. Aucune perte de couverture Vision B. Les 67 fichiers de tests touchant `api/scanner/llm/confluence`, 69 touchant `intelligence/`, et 21 touchant `delivery/` restent tous en place.

## Patches collatéraux appliqués dans cette PR

### `src/agents/__init__.py` — 4 blocs `try/except` supprimés

Les 4 blocs qui tentaient d'importer `integration`, `intelligent_integration`, `orchestrated_integration`, `risk_integration` ont été remplacés par des commentaires d'archivage + assignations `None` (préserve l'API publique pour les modules importants). Aucun bruit logger au démarrage.

### `src/environment/strategy_features.py` — patch **différé**

Le fichier contient un bloc `if __name__ == '__main__':` qui fait `from src.agent_trainer import AgentTrainer`. Module-import fonctionne ; seul `python src/environment/strategy_features.py` est cassé après cet archivage.

Le patch (suppression du bloc `__main__`) **n'a pas été appliqué dans PR1** car le fichier est en M-state dans une session Sprint 1 parallèle (LGBM wire-up + 4-tier LLM cascade + auth DG-057). Toucher ce fichier mélangerait les scopes.

À traiter dans un Lot futur — soit par la session Sprint 1 quand elle commit, soit dans un PR de cleanup post-merge. Effort : 2 minutes.

## Ce qui n'est PAS dans cet archive

### Ce qui reste actif dans le repo

Les composants Sprint 1 risk **encore utiles** restent dans `src/agents/` :
- `src/agents/portfolio_risk.py` (VaR, CVaR, exposure, correlations) — toujours actif.
- `src/agents/kill_switch.py` (circuit breakers, halt levels) — toujours actif.
- `src/agents/audit_logger.py` (audit trail, export formats) — toujours actif.
- `src/agents/intelligent_risk_sentinel.py` (ML risk prediction) — toujours actif.
- `src/agents/news_analysis_agent.py`, `market_regime_agent.py`, `risk_sentinel.py`, `base_agent.py`, `events.py`, `config.py`, `orchestrator.py` — tous actifs.

Seule la **couche d'orchestration** `risk_integration.py` (≈300 L de glue code) est archivée. Elle est triviale à refaire propre (< 4h) si un pivot B2B-API réactive une surface unifiée `IntegratedRiskManager`.

### Pourquoi pas d'exfiltration vers l'actif

Audit complet dans `docs/architecture/CLEANUP_AUDIT_LOT1.md` §4. Verdict : **rien à exfiltrer**. Concepts utiles (HMM regime, VaR, kill-switch) sont déjà ré-implémentés ailleurs (`src/intelligence/regime_gate.py`, composants risk listés ci-dessus).

## Restauration (procédure)

Voir `_archive/README.md`. Résumé : `git mv _archive/2026-05-27_pivot_vision_b/<chemin>/<fichier> <chemin>/<fichier>` préserve l'historique via `git log --follow`.

**Avertissement** : restaurer un fichier RL implique de reconstruire toute sa chaîne d'imports (le code archive importe d'autres fichiers archivés). Voir la cartographie des imports croisés dans l'audit pré-archivage §2.

## Tests post-archive

**Compte exact à la collection** (`pytest tests/ --collect-only`) :
- Pré-archive : **2 894 tests** collectés (avec test_checkpoint_manager + test_walk_forward qui plantaient déjà à l'ImportError sur leurs cibles maintenant archivées — mais étaient comptabilisés grâce à `--collect-only`)
- Post-archive (13 fichiers RL) : **~2 590 tests** collectés, **0 collection error**
- Différentiel : ~304 tests archivés (159 du bloc initial + 138 du bloc #1 + 7 du bloc #2)

**Gate de PR1** : `pytest tests/` retourne 2 580 verts, 13 failures **toutes WIP-préexistantes** (pas causées par cet archivage) :
- 1 failure sur `tests/test_template_narrative_engine.py::test_validator_accepts_strong_signal` — appartient au refactor 4-tier LLM cascade en cours dans session Sprint 1 (modifs M sur `src/intelligence/llm_narrative_engine.py`).
- 12 failures sur `tests/test_webapp_preview.py` — appartiennent au redesign landing en cours dans session frontend (modifs M sur `src/api/app.py`, `src/api/routes/qa.py`, mocks webapp).

Confirmation : ZERO de ces 13 fichiers de test n'importe `src/intelligence/`, `src/api/`, `src/delivery/`. Aucune perte de couverture Vision B. Les 67 fichiers de tests touchant `api/scanner/llm/confluence`, 69 touchant `intelligence/`, et 21 touchant `delivery/` restent tous en place. Les 13 failures pré-existantes seront résolues par les sessions Sprint 1 et frontend respectivement ; PR1 attend leur résolution avant merge (discipline « jamais merger en rouge »).
