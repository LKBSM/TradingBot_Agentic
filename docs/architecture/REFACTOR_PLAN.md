# REFACTOR PLAN — Détail actionnable

**Référence** : `docs/architecture/MIA_MARKETS_ARCHITECTURE.md` Partie 5
**Statut** : à exécuter avant ou en parallèle de la Vague 1 V1
**Effort total** : 82-122 h (≈ 10-15 sem solo à 8-9 h/sem)
**Principe** : **chaque étape commit + tests verts avant la suivante**. Pas de big-bang.

## Checklist par lot

### Lot 1 — Cleanup (3-4 h) · 🔴 P0 immédiat

Objectif : enlever le bruit visuel et mental sans rien casser fonctionnellement.

- [ ] Créer le dossier `_archive/2026-05-XX_pivot_b/`.
- [ ] `git mv parallel_training.py _archive/2026-05-XX_pivot_b/`
- [ ] `git mv src/training _archive/2026-05-XX_pivot_b/training`
- [ ] `git mv src/agents/integration.py _archive/2026-05-XX_pivot_b/agents/`
- [ ] `git mv src/agents/intelligent_integration.py _archive/2026-05-XX_pivot_b/agents/`
- [ ] `git mv src/agents/orchestrated_integration.py _archive/2026-05-XX_pivot_b/agents/`
- [ ] `git mv src/agents/risk_integration.py _archive/2026-05-XX_pivot_b/agents/`
- [ ] `git mv src/agents/regime_predictor.py _archive/2026-05-XX_pivot_b/agents/`
- [ ] `git mv src/agents/sprint2_intelligence.py _archive/2026-05-XX_pivot_b/agents/`
- [ ] Audit `src/agents/market_regime_agent.py` — vérifier appelants vivants. Si zéro → archive.
- [ ] `git rm tests/test_long_short_trading.py` (DG-002).
- [ ] `git rm Procfile railway.toml` (DG-012).
- [ ] Créer `docs/archive/` et `git mv` les 15+ .md de pollution racine :
  - [ ] `BUSINESS_PLAN_SMART_SENTINEL.md`
  - [ ] `COMMERCIALIZATION_REPORT.md`
  - [ ] `COMPLETE_PROJECT_DOCUMENTATION.md`
  - [ ] `COMPREHENSIVE_*.md` (×2)
  - [ ] `INSTITUTIONAL_AUDIT_REVIEW.md`
  - [ ] `SPRINT_PLAN.md` · `SPRINT_ROADMAP.md` · `SPRINT3_READINESS_AUDIT.md`
  - [ ] `AGENTS_SYSTEM_ANALYSIS.md`
  - [ ] `ANALYSE_COMPLETE_SYSTEME.md` · `ANALYSE_TRAINING_SYSTEM.md`
  - [ ] `TRAINING_ARCHITECTURE.md`
  - [ ] `PLAN.md` · `MISSION_ACK.md` (versions obsolètes — ne pas archiver MISSION_ACK_NAMING.md récent)
  - [ ] `EVALUATION_PROMPTS.md` · `GO_NO_GO_PROMPT.md` · `BACKTEST_LEGAL_GUARDRAILS.md`
  - [ ] `FORWARD_TEST_AND_GTM.md`
  - [ ] `PROJET_VISION_INDICATEUR_CHATBOT.md` (consolidé dans governance/)
- [ ] `git rm` ou archive les vieux `replay_*.json` et `replay_*.csv` racine.
- [ ] `nul`, `Script collab`, `test_env_debug.py` racine → archive.
- [ ] Lancer la suite de tests complète : **doit rester verte**.
- [ ] PR titre : `chore(repo): archive legacy RL stack + clean root pollution`.

**Gate de sortie Lot 1** : 1366+ tests verts (hors `test_long_short_trading`), `git log` propre, `ls /` plus lisible.

### Lot 2 — Unification contrats + merges (13-20 h) · 🔴 P0

Objectif : un seul `InsightSignalV2`, un seul `regime_gate`, SmartMoney promu.

- [ ] **Merge regime** :
  - [ ] Diff `regime_filter.py` vs `regime_gate.py` ligne à ligne.
  - [ ] Identifier appelants des deux modules (grep).
  - [ ] Créer `src/intelligence/regime/gate.py` qui combine les responsabilités.
  - [ ] Migrer appelants un par un.
  - [ ] Archiver l'ancien module orphelin.
  - [ ] Tests régression régime/jump_ratio.
- [ ] **Audit `bocpd.py`** : si zéro appelant après merge → archive.
- [ ] **Audit `volatility_lgbm.py`** : si VOL_MODE=har figé en prod → archive (DG MISC).
- [ ] **Audit `semantic_cache.py` vs `rag/cache.py`** :
  - [ ] Identifier responsabilités exactes.
  - [ ] Si distinctes → renommer `llm_response_cache.py` + `rag_embedding_cache.py`.
  - [ ] Si overlap → fusionner.
- [ ] **Unifier `InsightSignalV2`** :
  - [ ] Source unique = `src/intelligence/insight_v2/`.
  - [ ] Lire `src/api/insight_signal_v2.py` ; déplacer toute logique de projection (`to_telegram_b2c`, `to_b2b_dict`) vers `src/delivery/renderers/`.
  - [ ] `git rm src/api/insight_signal_v2.py`.
  - [ ] Dans `src/api/schemas/insight.py` (nouveau), faire `from src.intelligence.insight_v2 import InsightSignalV2`.
  - [ ] Mettre à jour tous les imports.
  - [ ] Tests : round-trip API ↔ pipeline doit produire byte-identique.
- [ ] **Promouvoir SmartMoneyEngine** :
  - [ ] `git mv src/environment/strategy_features.py src/intelligence/smart_money/engine.py`.
  - [ ] Search-replace imports : `from src.environment.strategy_features import SmartMoneyEngine` → `from src.intelligence.smart_money.engine import SmartMoneyEngine`.
  - [ ] Tests SMC restent verts.
- [ ] PR par étape, tests verts entre chaque.

**Gate de sortie Lot 2** : `git grep "from src.api.insight_signal_v2"` retourne 0 résultat. `regime_filter.py` ET `regime_gate.py` n'existent plus séparément.

### Lot 3 — Ports / adapters (16-22 h) · 🔴 P0

Objectif : injection de dépendances ; ajouter un canal V1+ ne touche que `delivery/`.

- [ ] Créer `src/interfaces/__init__.py`.
- [ ] Créer `src/interfaces/delivery.py` avec `IDeliveryChannel` Protocol.
- [ ] Créer `src/interfaces/data.py` avec `IDataProvider` Protocol.
- [ ] Créer `src/interfaces/narrative.py` avec `INarrativeEngine` Protocol.
- [ ] Créer `src/interfaces/signal_store.py` avec `ISignalStore` Protocol.
- [ ] Créer `src/interfaces/llm.py` avec `ILLMRouter` Protocol.
- [ ] Refactor `src/delivery/` :
  - [ ] Créer `src/delivery/adapters/` (sous-dossier).
  - [ ] Déplacer `telegram_notifier.py` → `adapters/telegram_adapter.py` qui implémente `IDeliveryChannel`.
  - [ ] Idem `discord_notifier.py` → `adapters/discord_adapter.py`.
  - [ ] Créer `src/delivery/renderers/` :
    - [ ] `renderers/telegram.py` (logique `to_telegram_b2c`).
    - [ ] `renderers/discord.py`.
    - [ ] `renderers/focus.py`, `renderers/copilot.py`, `renderers/expert.py` (pour API).
  - [ ] Créer `src/delivery/factory.py` : `build_delivery_engine(config) → DeliveryEngine`.
- [ ] Refactor `src/intelligence/main.py` (à renommer `scanner_runtime.py`) :
  - [ ] Remplacer les imports directs des notifiers par injection via factory.
  - [ ] Tests build_system passent.
- [ ] Créer un `StdoutAdapter` (adapter dummy qui print) **pour valider le découplage** : son ajout ne doit toucher AUCUN fichier hors `delivery/`.
- [ ] PR `feat(arch): introduce ports/adapters pattern in delivery layer`.

**Gate de sortie Lot 3** : `git diff` lors d'ajout d'un nouveau canal n'affecte QUE `src/delivery/`.

### Lot 4 — Splits god modules (18-28 h) · 🟡 P1

Objectif : réduire les fichiers > 1 200 LOC en sous-modules cohérents.

- [ ] **Splitter `src/environment/environment.py`** (2 423 L) :
  - [ ] Extraire features actives → `src/intelligence/smart_money/multi_timeframe_features.py` (si pas déjà dans `multi_timeframe_features.py`).
  - [ ] Archiver `_legacy/rl_env.py` (TradingEnv RL).
  - [ ] Archiver `_legacy/reward_shaper.py`.
  - [ ] Tests régressions strategy_features.
- [ ] **Splitter `src/intelligence/volatility_forecaster.py`** (1 561 L) :
  - [ ] `volatility/har.py` (HAR-RV core)
  - [ ] `volatility/hmm_mult.py` (multiplicateur HMM)
  - [ ] `volatility/calendar_mult.py`
  - [ ] `volatility/diurnal_mult.py`
  - [ ] `volatility/blender.py` (compose forecast final)
  - [ ] `volatility/tcp_intervals.py` (Transductive Conformal Prediction)
  - [ ] `volatility/__init__.py` re-export `VolatilityForecaster`.
- [ ] **Splitter `src/api/models.py`** (~800 L) en `src/api/schemas/` :
  - [ ] `schemas/auth.py` (LoginRequest, TokenResponse, UserMe, …)
  - [ ] `schemas/billing.py` (CheckoutRequest, Subscription, Usage, …)
  - [ ] `schemas/insight.py` (re-export `InsightSignalV2`, query filters, …)
  - [ ] `schemas/chat.py` (AskRequest, SSEFrame, Suggestion, …)
  - [ ] `schemas/account.py`
  - [ ] `schemas/track_record.py`
  - [ ] `schemas/webhooks.py`
  - [ ] `schemas/__init__.py` re-export pour compat imports.
- [ ] PR par split, tests verts.

**Gate de sortie Lot 4** : aucun fichier Python ne dépasse 1 200 LOC dans `src/intelligence/` ou `src/api/`.

### Lot 5 — Risk consolidation (12-18 h) · 🟡 P1 (DG-039 MODIFY)

Objectif : un seul `RiskService` qui expose ce qui est utile à `InsightSignalV2` mode EXPERT.

- [ ] Inventaire de tous les "RiskManager" :
  - [ ] `src/agents/risk_sentinel.py`
  - [ ] `src/agents/intelligent_risk_sentinel.py`
  - [ ] `src/agents/portfolio_risk.py`
  - [ ] `src/agents/ensemble_risk_model.py`
  - [ ] `src/environment/risk_manager.py`
  - [ ] (et tout `src/risk/`, `src/interfaces/risk.py`)
- [ ] Identifier appelants vivants pour chacun.
- [ ] Créer `src/services/risk/risk_service.py` :
  - [ ] `risk_score(insight: InsightSignalV2) → int (0-100)`
  - [ ] `kill_decision(insight) → Literal["TRADE","REDUCE","BLOCK"]`
  - [ ] Internement : vote pondéré des stratégies de risk existantes (ensemble).
- [ ] Migrer appelants un par un vers `RiskService`.
- [ ] Archiver les anciennes classes orphelines après migration complète.
- [ ] Ajouter `risk_readout` (optionnel) à `InsightSignalV2` mode EXPERT.
- [ ] PR `feat(arch): consolidate risk managers into single RiskService`.

**Gate de sortie Lot 5** : `grep "class.*RiskManager"` retourne ≤ 2 résultats (le service + 1 wrapper compat éventuel).

### Lot 6 — Wire LGBM (20-30 h) · 🔴 P0 chemin critique algo

Objectif : remplacer le scoring rule-based cosmétique par le pipeline calibré.

- [ ] Lire `src/intelligence/scoring/` actuel :
  - [ ] Vérifier `LGBMScorer` charge bien `models/scoring_v3_lgbm.pkl`.
  - [ ] Vérifier `IsotonicRecalibrator` fitté disponible.
  - [ ] Vérifier `AdaptiveConformalScorer` initialisé avec buffer.
- [ ] Audit `CalibratedConvictionPipeline.score_one()` :
  - [ ] Test sur un sample d'insights historiques que mode B produit P(win) calibrée.
  - [ ] Mesurer Brier OOS sur fenêtre out-of-fold.
- [ ] Comparer mode A (rule-based, Pearson −0.023) vs mode B (LGBM calibré) sur 7 ans walk-forward :
  - [ ] Brier OOS amélioration ≥ +5 % (gate de migration).
  - [ ] PF rolling 12 mois > 1.20 ?
  - [ ] DSR > 1.0 ?
- [ ] Si gates passent :
  - [ ] Activer `SCORING_VERSION=v3` (LGBM) par défaut dans `config.py`.
  - [ ] Mettre à jour `conviction_mode` à `calibrated` dans le contrat.
  - [ ] Tests E2E confirment.
- [ ] Si gates ne passent pas :
  - [ ] Documenter dans `reports/governance/kill_criteria_board.md`.
  - [ ] Maintenir mode A en attendant amélioration du modèle.
  - [ ] Faire remonter à l'utilisateur pour décision.
- [ ] PR `feat(scoring): wire calibrated LGBM pipeline as default`.

**Gate de sortie Lot 6** : `conviction_mode == "calibrated"` par défaut en prod, Brier skill OOS > +2 % ET DSR > 1.0 ET PBO < 0.5 (cf. `pivot_positioning_2026_05_27`). Tant que la Gate de promotion premium n'est pas franchie, **aucun chiffre de performance (PF, IC, win-rate, drawdown) ne doit apparaître en surface client-facing** — claims retirés depuis 2026-05-27.

## Ordre d'exécution recommandé

```
Lot 1 (Cleanup)
  └─▶ Lot 2 (Unification contrats)
        └─▶ Lot 3 (Ports/adapters)
              ├─▶ Lot 4 (Splits god modules)        [parallèle possible]
              ├─▶ Lot 5 (Risk consolidation)        [parallèle possible]
              └─▶ Lot 6 (Wire LGBM)                 [chemin critique algo]
```

**Lot 1 + Lot 2 + Lot 3** sont **séquentiels et bloquants** pour la Vague 1.

**Lots 4, 5, 6** peuvent être parallélisés une fois Lot 3 fini.

**Lot 6 est le seul qui nécessite validation empirique (Brier OOS, gates kill_criteria_board)** avant merge.

## Effort cumulé par scénario

| Scénario | Lots | Heures | Semaines solo |
|---|---|---|---|
| Minimum viable refactor (Lots 1+2+3) | Cleanup + Contrats + Ports | 32-46 h | 4-6 sem |
| Refactor complet (Lots 1-6) | + Splits + Risk + LGBM | 82-122 h | 10-15 sem |

## Tests de non-régression à exécuter à chaque PR

```bash
# Suite complète
pytest tests/ -x --tb=short

# Smoke E2E
pytest tests/test_smoke_e2e.py -v

# Contrat InsightSignalV2
pytest tests/test_insight_signal_v2.py tests/test_insight_signal_v2_enrichment.py

# Pipeline integration
pytest tests/test_pipeline_integration.py tests/test_sprint1_e2e_integration.py

# State machine
pytest tests/test_state_machine_replay.py

# Confluence detector
pytest tests/test_confluence_detector.py
```

Tous doivent rester verts (hors `test_long_short_trading.py` qui sera supprimé en Lot 1).

## Liens

- Plan complet 8 parties : `docs/architecture/MIA_MARKETS_ARCHITECTURE.md`
- Plan dev-focus en cours : `docs/governance/dev_focus_plan_2026_05_27.md`
- Decision Gate Review V2 : `docs/governance/decision_gate_review_v2.md`
- Audit algo 2026-05-27 (justifie Lot 6 P0) : `docs/governance/AUDIT_ALGO_2026_05_27.md`
