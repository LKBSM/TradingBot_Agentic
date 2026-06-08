# Charter — State Machine Engineer

**Slug** : `state_machine_engineer`
**Date** : 2026-05-15
**Sponsor** : Lead Quant Architect

## 1. Mission
Posséder `src/intelligence/signal_state_machine.py` (HOLD/BUY/SELL trust layer avec hystérésis, confirmation, cooldown, lifetime, opposing-lockout) et `state_persistence.py`. À l'issue de Sprint 0, le code est excellent (8.0/10, déterministe, thread-safe, persistence-ready, 54 tests) mais les defaults ne sont pas empiriques (P0-12 — sweep 432 cellules pending depuis eval_07) et les tests chaos / property-based manquent (P1-11). Le rôle livre Sprint 3 le sweep paramétrique sur 4 actifs × 4 TF, Sprint 5 la batterie property-based, Sprint 6 le versioning JSON state.

## 2. Périmètre
- **Inclus** :
  - `src/intelligence/signal_state_machine.py`.
  - `src/intelligence/state_persistence.py`.
  - Sweep paramétrique (`enter_threshold`, `exit_threshold`, `hysteresis_band`, `confirm_bars`, `cooldown_bars`, `max_age`, `opposing_lockout_bars`).
  - Property-based / chaos / mutation tests.
  - Snapshot store API per-signal (Sprint 6).
  - Versioning JSON state.
- **Exclu** :
  - Scoring confluence (Stat Validator).
  - Régime gate consumption (REDUCE state — coordination Regime Sci P1-8).
  - Backtest engine (Backtest Infrastructure).
  - Signal delivery (Telegram / API — out of algo scope).

## 3. KPI principal et métriques
- **KPI** : 0 oscillation parasite sur 10⁶ tirages property-based.
- **Sous-métriques** :
  - 0 oscillation HOLD↔BUY↔HOLD↔BUY dans cooldown window sur Hypothesis 10⁶ tirages.
  - 0 transition non documentée sur l'ensemble du sweep paramétrique.
  - Sweep paramétrique : 432 cellules × 4 actifs × 4 TF = 6 912 backtests reproductibles.
  - Reproductibilité bit-à-bit (state hash identique sur 3 runs successifs).
  - Mutation score `mutmut` ≥ 70 % sur `signal_state_machine.py` (Sprint 6).
  - Stabilité persistence : load après save = state identique sur 100 % des cas.
  - Versioning JSON : downgrade lecture v2 par code v1 raise `IncompatibleVersionError` (pas de silent fail).
- **Cadence de mesure** : continue (CI sur chaque PR) + recalc complet end of Sprint 3 + Sprint 5.

## 4. RACI sur les sprints

| Sprint | Responsable | Accountable | Consulted | Informed |
| ------ | ----------- | ----------- | --------- | -------- |
| Sprint 1 | State Machine Eng (P1-10) | LQA | Data Quality | — |
| Sprint 2 | State Machine Eng (P1-10) | LQA | SMC, Vol Modeler | — |
| Sprint 3 | State Machine Eng (sweep P0-12) | LQA | Backtest Infra, Stat Validator | Tous |
| Sprint 4 | — | LQA | Stat Validator (consume score) | — |
| Sprint 5 | State Machine Eng (chaos P1-11) | LQA | QA | — |
| Sprint 6 | State Machine Eng (P0-16, P1-12) | LQA | Backtest Infra | — |
| Sprint 7 | — | LQA | — | — |

## 5. Findings prioritaires de l'audit Phase 1 (P0/P1)
- **P0-12** — Defaults state machine non empiriques (sweep 432 cellules pending depuis eval_07). Sprint 3.
- **P0-16** — Snapshot store API per-signal manquant. Sprint 6.
- **P1-8** — `REDUCE` state du RegimeGate non consommé par state machine. Sprint 5 (coordination Regime Sci).
- **P1-10** — `confirm_bars` / `max_age` non paramétrés par TF. Sprint 1-2.
- **P1-11** — Tests chaos / property-based manquants. Sprint 5.
- **P1-12** — Versioning JSON state absent (`state_persistence.py`). Sprint 6.
- Empirique baseline : 0 arms_started sur 7 ans XAU + 7 ans EURUSD (lié à P0-3 — score plafonne sous enter=75 — résolution Sprint 3-4 par autres owners).

(Liens : [audit §3.7](../../audits/2026-Q2/section_3_7_state_machine.md))

## 6. Inputs / Outputs
- **Inputs** :
  - Confluence score (depuis Stat Validator / ConfluenceDetector).
  - Régime state ALLOW/REDUCE/BLOCK (depuis Regime Gate).
  - Conformal probability (depuis Conformal Wrapper).
  - Time-of-day filter (depuis Regime Scientist sessions).
- **Outputs** :
  - `src/intelligence/signal_state_machine.py` (paramètres empiriques, REDUCE state consommé).
  - `src/intelligence/state_persistence.py` (versionné).
  - `tests/test_signal_state_machine_*.py` + `tests/property/test_state_machine_invariants.py`.
  - `audits/2026-Q3/state_machine_sweep_<asset>_<tf>.md` × 16 fichiers (4×4 grid).
  - Snapshot store API : `src/intelligence/signal_snapshot_store.py` (Sprint 6).
  - `docs/algo/state_machine.md`.

## 7. Critères de "done"
- Sweep 432 cellules × 4 actifs × 4 TF complété ; defaults empiriques choisis par actif/TF documentés dans `config.py`.
- 0 oscillation parasite sur Hypothesis 10⁶ tirages (`tests/property/test_state_machine_invariants.py`).
- Versioning JSON state : `version` field obligatoire, migration scripts pour v1→v2.
- Snapshot store API : `POST /signals/{id}/snapshot` retourne état machine complet à `bar_ts`.
- REDUCE state consommé : position multiplier ∈ {1.0, 0.5, 0.0} selon ALLOW/REDUCE/BLOCK.
- Mutation score `mutmut` ≥ 70 % sur state_machine.py.
