# Audit Phase 1 — Section 3.7 : SignalStateMachine

**Date** : 2026-05-15
**Auditeur** : Claude
**Périmètre** : `src/intelligence/signal_state_machine.py` (922 LOC), `state_persistence.py` (137 LOC), `state_machine_replay.py` (914 LOC), `notification_queue.py` (267 LOC).

---

## Score : **8.0 / 10** (confirme eval_07)

Code excellent : déterministe, thread-safe (re-entrant lock), persistence-ready, riche en transitions documentées. Le seul bémol majeur : **les paramètres par défaut ne sont pas empiriquement calibrés** — résultat = 0 trade sur 7 ans de baseline Sprint 0 (cf. ce sprint).

---

## 1. Architecture

| Classe                       | Rôle                                                  |
| ---------------------------- | ----------------------------------------------------- |
| `PublicState`                | HOLD / BUY / SELL — visible client                    |
| `Direction`                  | LONG / SHORT                                          |
| `ExitReason`                 | 6 types (signal_change, max_age, stop_loss, take_profit, opposing_lockout, manual) |
| `_Phase`                     | ARMING / CONFIRMED / COOLDOWN / LOCKOUT (internal)    |
| `BarInput`                   | Container pour un tick (price, score, ts, etc.)       |
| `SignalStateMachine`         | Cœur de la trust layer                                |
| `StateMachineConfig`         | Tunables (enter, exit, confirm_bars, cooldown_bars, max_signal_age_bars, ...) |

---

## 2. Strengths

| # | Strength                                                                                       |
| - | ---------------------------------------------------------------------------------------------- |
| S1 | **Déterministe** : seed-free, dépend uniquement de la séquence d'inputs.                       |
| S2 | **Thread-safe** : re-entrant lock (RLock) au niveau de la classe.                              |
| S3 | **Persistence ready** : `state_persistence.py` sérialise/restore atomique avec staleness guard. |
| S4 | **6 exit reasons** explicites et auditables.                                                   |
| S5 | **Hystérésis** : `enter_threshold` ≠ `exit_threshold` (default 75 vs 55) — évite oscillation. |
| S6 | **Couverture tests : 54 tests** (`test_signal_state_machine.py`) + tests connexes (`test_bos_no_repeated_fire`, `test_bos_retest`, `test_sprint2_churning`). |
| S7 | **Lockout opposé** : empêche le flip-flop LONG↔SHORT immédiat (mémoire `opposing_lockout`). |

---

## 3. Findings

| # | Finding                                                                                       | Sévérité | Action                            |
| - | --------------------------------------------------------------------------------------------- | -------- | --------------------------------- |
| F1 | **`enter_threshold=75` ne produit AUCUN trade** sur 7 ans XAU + EURUSD (Sprint 0 baseline). Score plafonne à 72.61 / 74.97. **Defaults non empiriques**, eval_07 le disait déjà ("sweep 432 cellules à exécuter"). | P0 | Sprint 3 — sweep + recalibration  |
| F2 | **`exit_threshold=55`** : pas de justification empirique. Symmétrie 75/55 = +20/−20 — pourquoi pas 70/40 ou 80/60 ? | P0 | Sprint 3                          |
| F3 | **`confirm_bars=2`** : 2 bars × 15 min = 30 min de latence avant entrée. Sur M15 = OK ; sur H1 = 2 h de retard. À paramétrer par TF. | P1 | Sprint 2 / 3                      |
| F4 | **`max_signal_age_bars=12`** : 12 × 15 min = 3 h. Cohérent avec horizon trade XAU M15. Mais hardcoded — pas adapté H1 ou D1. | P1 | Sprint 2                          |
| F5 | **`silent_bars_before_score_exit=2`** : si le score "rate" 2 bars consécutives < exit_threshold, exit forcé. Magic number. | P2 | Sprint 3                          |
| F6 | **`high_vol_forces_exit=True`** : par défaut, un régime high-vol expulse les trades en cours. Politique discutable (peut-être on veut juste réduire la taille). | P1 | Sprint 3                          |
| F7 | 1 TODO + `state_machine_replay.py:390` 1 `print()` debug — mineur.                            | P3       | Cleanup Sprint 6                  |
| F8 | **Tests chaos / propriété-based manquants** : que se passe-t-il si les inputs oscillent à haute fréquence ? Hystérésis robuste à du noise ? | P0 | Sprint 5 batch 5.1                |
| F9 | **Cooldown post-exit** : `cooldown_bars=2` = 30 min. Bloque ré-entrée immédiate. Empirique ? | P2       | Sprint 3                          |
| F10 | **`signals_per_day`** rapporté en sortie summary — à mettre en garde-fou (alerte si > 5 sig/jour = bug). | P3 | Sprint 6                          |

---

## 4. Test coverage transitions

Tests existants vérifient :
- HOLD → BUY (signal LONG passe enter)
- HOLD → SELL (signal SHORT passe enter)
- BUY → HOLD via score < exit
- BUY → HOLD via max_age
- BUY → HOLD via stop_loss / take_profit
- BUY → SELL bloqué par opposing_lockout
- COOLDOWN après HOLD bloque ré-entrée

**Manque** :
- Chaos test : oscillation rapide score 70-80-70-80 sur 100 bars (devrait pas générer N trades).
- Property-based : pour toute séquence d'inputs valides, jamais 2 ouvertures simultanées.
- Concurrence : 2 threads appelant `on_bar()` simultanément (re-entrant lock testé ?).

---

## 5. Persistence (`state_persistence.py`)

| # | Finding                                                                              | Sévérité |
| - | ------------------------------------------------------------------------------------ | -------- |
| F11 | **Atomic write OK** (write tmp → rename) mais staleness guard à valider sur reboot crash. | P1 |
| F12 | **JSON format** — pas de versioning du schéma. Si on ajoute un champ, l'ancien dump cassera. | P1 |

---

## 6. Replay engine (`state_machine_replay.py`)

| # | Finding                                                                                       | Sévérité |
| - | --------------------------------------------------------------------------------------------- | -------- |
| F13 | **Reproductibilité bit-à-bit** : seed numpy seedé dans `run_baseline_sprint0.py`, mais le replay lui-même n'a pas besoin de RNG (déterministe). Confirmé : 2× quick run = mêmes SHA256. | ✅ OK |
| F14 | `print()` ligne 390 (`results.pretty()`) — debug, à transformer en `log.info`. | P3 |
| F15 | Pas d'API pour rejouer un signal **individuel** à un timestamp donné (Sprint 6 snapshot store). | P0 | Sprint 6 |

---

## 7. Notification queue (`notification_queue.py`)

| # | Finding                                                                              | Sévérité |
| - | ------------------------------------------------------------------------------------ | -------- |
| F16 | File événements pour le delivery (Telegram/Discord). Pas vraiment dans le périmètre algo Sprint 0. Hors scope. | INFO |

---

## 8. Empirique Sprint 0 baseline

- **XAU M15** 172 749 bars → 192 signals_produced_by_detector → **0 arms_started** (aucun signal franchit enter=75).
- **EURUSD M15** 174 381 bars → 13 signals → **0 arms_started**.
- Score `p99 = 69.5` (XAU) / `73.82` (EURUSD) → **plafond du detector actuel**.

**Implication architecturale** :
- Soit le ConfluenceDetector doit être recalibré pour produire des scores > 75 plus fréquemment (Sprint 4).
- Soit le `enter_threshold` doit baisser à 65-70 (Sprint 3 sweep décide).
- Soit les composantes News + Vol doivent être branchées en replay (Sprint 2 — pipeline calendrier économique).

---

## 9. Recommandations

| Sprint | Action                                                                              | Priorité |
| ------ | ----------------------------------------------------------------------------------- | -------- |
| 2      | Pipeline calendrier + vol branché en replay (eliminate score plafond)               | P0       |
| 3      | Sweep paramétrique `(enter, exit, confirm, cooldown, max_age)` × 4 TF × 4 actifs    | P0       |
| 3      | Calibration empirique des defaults par instrument / TF                              | P0       |
| 5      | Tests chaos + property-based (F8)                                                   | P0       |
| 6      | Snapshot store API per-signal (F15)                                                 | P0       |
| 6      | Versioning JSON state (F12)                                                         | P1       |

---

## 10. Ce que cet audit ne couvre pas

- **Property-based testing exhaustif** (hypothesis lib) — Sprint 5.
- **Concurrency stress test** (re-entrant lock sous charge) — Sprint 6.
- **Snapshot store design** complet — Sprint 6.

---

**Signé** : 2026-05-15, Claude
