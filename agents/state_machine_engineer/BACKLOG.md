# Backlog — State Machine Engineer

**Date** : 2026-05-15
**Owner** : State Machine Engineer

## Sprint 1 (S3-S4) — Data Layer (support)

- [ ] Audit `signal_state_machine.py` ligne par ligne : documenter la machine (graphe états + transitions + guards) dans `docs/algo/state_machine_states.md` — 4 h
- [ ] Identifier les paramètres TF-dépendants candidats pour P1-10 : `confirm_bars`, `max_age`, `cooldown_bars` ; proposer le schéma `Dict[Timeframe, StateMachineParams]` — 3 h
- [ ] Coordination Data Quality : adapter le contrat `Signal` Pydantic v2 si nouveau champ `regime_state` ajouté (préparation P1-8) — 1 h

## Sprint 2 (S5-S6) — Detection (support)

- [ ] **Fix P1-10** — Paramétrer `confirm_bars` / `max_age` par TF dans `InstrumentConfig` (ou dataclass dédié) ; refactor pour lookup par TF — 5 h
- [ ] Tests régression `tests/test_state_machine_tf_params.py` (3 TF testés : M15, H1, H4) — 3 h
- [ ] Default values par TF (à valider Sprint 3 sweep) : indicatives M15 confirm=2/max=64, H1 confirm=2/max=24, H4 confirm=1/max=12 — 1 h

## Sprint 3 (S7-S8) — Edge Discovery (rôle pivot)

- [ ] **P0-12 — Sweep paramétrique state machine** : 432 cellules × 4 actifs (XAU, EURUSD, BTCUSD si dispo, USOIL si dispo) × 4 TF (M15, H1, H4, D1) = 6 912 backtests — 16 h
  - Grille : `enter_threshold ∈ {55, 60, 65, 70, 75}`, `exit_threshold ∈ {45, 50, 55}`, `hysteresis ∈ {5, 10, 15}`, `confirm_bars ∈ {1, 2, 3}`, `cooldown_bars ∈ {0, 5, 10}`, `max_age ∈ {24, 48, 64}`.
  - Output : `audits/2026-Q3/state_machine_sweep_<asset>_<tf>.md` × 16 fichiers + JSON consolidé.
- [ ] Sélection des defaults empiriques par actif/TF basée sur PF + Sharpe + nb trades > 30 (input Stat Validator) — 4 h
- [ ] Refactor `config.py` : centraliser les defaults state machine par actif/TF — 3 h
- [ ] Coordination Backtest Infra : la grille doit être runnée en parallèle (joblib) sur le harness Sprint 3 — 2 h
- [ ] Tests régression `tests/test_state_machine_defaults_empirical.py` : assert que `enter_threshold` provient du fichier de defaults empiriques (pas hardcoded 75) — 2 h

## Sprint 4 (S9-S10) — Calibration (frozen)

- [ ] Pas de livrable nouveau. Stand-by pour adapter state machine si Stat Validator change l'API du score (logistic L1).

## Sprint 5 (S11-S12) — Robustness & Stress Testing (rôle pivot)

- [ ] **Fix P1-11** — Tests property-based avec Hypothesis : `tests/property/test_state_machine_invariants.py` — 8 h
  - Invariants à tester : (1) après cooldown, no transition possible ; (2) opposing-lockout symétrique ; (3) max_age force EXIT_TIMEOUT ; (4) hysteresis empêche oscillations ; (5) state hash identique sur replay.
  - Cible : 10⁶ tirages, 0 oscillation parasite, 0 transition non documentée.
- [ ] Chaos testing : injecter inputs adversariaux (score=NaN, score=Inf, bar_ts décroissant, simultanéité score+regime_flip) — 5 h
- [ ] **Fix P1-8** — Consommer REDUCE state du RegimeGate (coordination Regime Scientist) : ajouter `position_multiplier ∈ {1.0, 0.5, 0.0}` selon ALLOW/REDUCE/BLOCK ; propager dans `Signal` Pydantic v2 — 6 h
- [ ] Tests `tests/test_state_machine_reduce_state.py` — 3 h
- [ ] Mutation testing avec `mutmut` sur `signal_state_machine.py` ; cible ≥ 70 % mutation score (input QA) — 4 h

## Sprint 6 (S13-S14) — Production Hardening (rôle pivot)

- [ ] **Fix P0-16** — Snapshot store API per-signal : `src/intelligence/signal_snapshot_store.py` + endpoint `GET /signals/{id}/snapshot` retournant l'état machine complet au `bar_ts` du signal — 8 h
- [ ] **Fix P1-12** — Versioning JSON state : ajout `"version": "2.0"` dans le format `state_persistence.py` ; script migration `scripts/migrate_state_v1_to_v2.py` ; raise `IncompatibleVersionError` à load si version inconnue — 5 h
- [ ] Tests `tests/test_state_persistence_versioning.py` (v1→v2 migration + version unknown) — 3 h
- [ ] Profiling state machine : cible < 1 ms / tick (probable déjà OK) — 2 h
- [ ] Documentation `docs/algo/state_machine.md` v1 — 4 h

## Sprint 7 (S15-S16) — Commercial Readiness

- [ ] Tear sheet state machine par actif : nb arms_started, transitions matrix, exit reasons distribution, hold time distribution (batch 7.2 input) — 4 h
- [ ] Documentation finale `docs/algo/state_machine.md` v2 (avec graphe Mermaid des états + références théoriques sur hysteresis) — 4 h
- [ ] Fiche transparence client : "comment notre machine d'état évite les signaux contradictoires" — 2 h

## Inbox (non priorisé)
- Multi-symbol state coupling (e.g. XAU-DXY anti-corrélation) — feature additionnelle.
- Adaptive thresholds : ajuster `enter_threshold` par volatility régime (input Vol Modeler).
- Replay debugger UI (visualisation step-by-step pour audit).
- WebSocket state stream pour delivery temps-réel.
- Backup/restore state machine across server restarts (déjà partiellement via state_persistence).
- Persistent queue pour les signaux en pending confirmation (vs in-memory actuel).
