# Circuit-breaker thresholds — audit (2026-05-28)

**Scope** : Sprint 4 close-out task — validate the thresholds we ship for the LLM and Telegram circuit breakers.
**Sources** : `src/intelligence/main.py:222-232`, `src/intelligence/circuit_breaker.py`, production stats surfaced by `HealthChecker` (`/health/deep`).

---

## Configured breakers

| Name | failure_threshold | recovery_timeout (s) | Wrapped subsystem |
|---|---|---|---|
| `llm_api` | 3 | 60 | Anthropic Claude API calls (narrative engine) |
| `telegram` | 5 | 120 | python-telegram-bot ``send_message`` |

Both are configured at boot in `_build_circuit_breakers()` and pinned onto `AppState.circuit_breakers`, then aggregated by `HealthChecker.health_status()` so a tripped breaker surfaces in `/health`.

## Field-data window

The brief asked for a 30-day audit, but the current deployment is pre-production:

- No public traffic before 2026-05-27 (M.I.A. branding cutover).
- Only internal smoke + Vitest + pytest exercise the code paths since then.
- No real Telegram outage and no real Anthropic 5xx events have been observed in this window.

We therefore can only validate the thresholds **against the spec and against known incident behaviour from comparable systems**, not against a 30-day production log. The check goes back to live traffic once Phase 2 launches and a full month is recorded.

## Threshold rationale (current values)

### `llm_api` — threshold = 3, recovery = 60 s

- Anthropic's published SLO is < 0.5% 5xx + transient rate-limit retries handled in-SDK.
- A 3-failure threshold is roughly P(3 consecutive failures | independent 0.5% baseline) ≈ 1.3 × 10⁻⁷ — well below noise.
- Recovery 60 s: matches the upper bound of the SDK's internal retry-after on transient 429 / 5xx (≤ 30 s typical); doubling that gives the upstream time to settle.
- DG-052 cost-alert watcher fires *before* the breaker if cost runs away, so a wedged budget never reaches "3 consecutive failures" because every call still returns 200 with a $5 invoice.

**Verdict** : ✅ keep as-is until a 30-day prod sample disagrees.

### `telegram` — threshold = 5, recovery = 120 s

- Telegram's documented rate limit is 30 msg/sec/bot global and 1 msg/sec/chat. With DG-054 retry + jittered exponential backoff (Sprint 4 — `src/delivery/telegram_notifier.py`), a single `RetryAfter` event no longer fires the breaker because the notifier handles it.
- Threshold = 5 therefore represents *consecutive non-retryable failures* (e.g. bot token revoked, chat_id invalid, Telegram global outage). 5 is conservative — any production incident generating five consecutive *hard* failures is real.
- Recovery 120 s: Telegram's longest documented `Retry-After` advice is 60 s; doubling gives headroom for cascade recovery.

**Verdict** : ✅ keep as-is. DG-054 added retry around this so the breaker only sees the residual hard failures.

## Recommended actions

1. **Re-audit at +30 days post-launch.** When real traffic has logged the breakers' counters in `/health/deep`, snapshot the counts and the open/half-open transitions. Targets:
   - `llm_api`: 0–1 *legitimate* opens / 30 d. If > 5, the threshold needs raising or the upstream is genuinely unstable.
   - `telegram`: 0 opens / 30 d. If > 0, investigate the failure mode (token rotation? blocked region?) before adjusting.
2. **Add Sentry breadcrumbs on every state transition** (already wired via the standard logging integration — confirm via `tags={"breaker.name": "llm_api", "breaker.state": "OPEN"}` on the next event).
3. **Wire the cost-alert watcher (DG-052) to call ``llm_breaker.force_open()``** when the budget alert fires. This converts a runaway-cost incident into an immediate visible breaker state instead of waiting for organic failures. (Filed as Sprint 5 polish — out of scope for this audit.)

## Out-of-scope but worth noting

The brief mentions a 3rd breaker for *external data sources* (Dukascopy / FRED / ForexFactory). None is currently configured because:

- Dukascopy reads are file-based (CSV cache), not network.
- FRED has its own internal retry policy via `pandas_datareader`.
- ForexFactory CSV is also cached locally.

If we move to live-streaming providers in Sprint 5, a third breaker is justified.

---

## Sign-off

Sprint 4 circuit-breaker audit — **closed for the pre-launch window**. Re-open at M+1 with real traffic to validate the thresholds empirically.
