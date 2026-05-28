# Security audit — Sprint 2 close-out (M0, 2026-05-27)

**Scope** — Phase 1 DEV Performance-ready, Sprint 2 Sécurité critique.
**Auditor** — Internal (post-fix verification).
**Branch** — `institutional-overhaul`
**Reference** — `docs/governance/dev_focus_plan_2026_05_27.md`

This audit closes out the four blockers raised in eval reports
`eval_10_15_findings.md` and `eval_10_15_team_audit.md` and replaces
the placeholder paragraph in `PROGRESS.md` Sprint 2 with a full
finding-by-finding write-up.

---

## TL;DR

| # | Finding ID | Severity | Status | Closed by |
|---|-----------|----------|--------|-----------|
| 1 | DG-041 TESTING_MODE silently enabled in prod | High | ✅ Closed | `SENTINEL_TESTING_MODE` default flipped to `0`; CI gate `testing_mode_gate` scans for `=1` in shipped configs |
| 2 | DG-055 (F-03) admin HMAC replay window | High | ✅ Closed | Per-request nonce in `NonceStore`, canonical `ts:nonce:path`, default `ADMIN_NONCE_REQUIRED=on` |
| 3 | DG-056 (F-04) `users.api_key_id` not unique | Medium | ✅ Closed | Partial UNIQUE index + v1→v2 migration with online dedupe |
| 4 | DG-057 (F-05) `subscription_expires` never read | Medium | ✅ Closed | `require_api_key` returns 402 on lapsed subscriptions |

All four findings produced regressions-free tests. 61 net-new + regression admin tests pass; admin scripts must migrate to nonce headers within one release cycle.

---

## 1. DG-041 — TESTING_MODE auth bypass

**Severity** High. **Status** ✅ Closed.

### Vulnerability

`SENTINEL_TESTING_MODE=1` (the default until this fix) globally bypasses `require_api_key` and grants every caller a fabricated `INSTITUTIONAL` subscriber dict. The flag was originally intended for local dev. A deployment that forgot to set the env var to `0` would silently ship as fully open prod.

Indicators: `src/api/auth.py:25` warns on import-time, but no CI gate prevented the value from leaking into a release.

### Fix

1. **Default flipped to fail-closed.** `os.environ.get("SENTINEL_TESTING_MODE", "0") == "1"` (`src/api/auth.py:24`). Any deployment without an explicit `=1` now requires real API keys.
2. **CI gate.** `scripts/ci_testing_mode_gate.py` walks shipped configs (Dockerfile, docker-compose, k8s manifests, Procfile, fly.toml, env templates) and fails if `SENTINEL_TESTING_MODE=1` is found outside the explicit allowlist (`.github/workflows/algo_tests.yml`, `.github/workflows/ci.yml`).
3. **CI job wired.** `.github/workflows/ci.yml` runs the gate as job `testing_mode_gate`. A PR that ships `=1` in a prod config now fails check.

### Residual risk

Operators can still run `SENTINEL_TESTING_MODE=1` locally (intended). The CI gate only inspects committed files — runtime env vars set in cloud secret stores are not visible. Recommend Fly.io secret audit nightly (`fly secrets list`) once deployment lands.

### Tests

- `scripts/ci_testing_mode_gate.py` smoke-runs clean on the current tree.
- Auth tests in `tests/test_auth.py` patch the module-level toggle, exercising both `TESTING_MODE=True` and `TESTING_MODE=False` paths.

---

## 2. DG-055 (F-03) — Admin HMAC replay window

**Severity** High. **Status** ✅ Closed.

### Vulnerability

`require_admin` accepted any valid HMAC signature whose timestamp fell within a 5-minute window. An attacker capturing a signed request (e.g. via TLS-terminating proxy logs, browser dev tools, or a man-in-the-middle on the operator's network) could replay it as-is up to 300 seconds later — long enough to re-trigger key rotation, kill-switch toggles, or arbitrary state-changing admin endpoints.

Additional weakness: the original canonical (`data = timestamp.encode()`) had no route binding, so a signature minted for `/admin/rotate-key` would also validate against `/admin/revoke`.

### Fix

1. **Per-request single-use nonce.** New `src/api/nonce_store.py` provides a thread-safe, TTL-bounded `NonceStore.check_and_record()` that returns `False` on replay. TTL = 300 s (mirrors the timestamp window), `max_entries = 100k` with oldest-eviction as a flood-defense soft cap.
2. **Cross-route binding.** Canonical signed payload is now `f"{ts}:{nonce}:{path}"`. A signature minted for `/admin/rotate-key` does not validate on `/admin/revoke`.
3. **Strict-by-default.** `ADMIN_NONCE_REQUIRED=on` is the default. `X-Admin-Nonce` is required; a missing header yields 401 with detail `"Missing X-Admin-Nonce header"`.
4. **Legacy fallback gated.** `ADMIN_NONCE_REQUIRED=off` re-enables the legacy bare-timestamp canonical for one release cycle so that existing ops scripts have time to migrate. A deprecation warning fires on every legacy call.
5. **AppState wiring.** `src/api/app.py` default-instantiates `NonceStore()` when `nonce_store` is not provided to `create_app`. `src/api/dependencies.py` AppState exposes it.

### Residual risk

- Single-process memory only. Once we deploy multi-worker (uvicorn `--workers >= 2`), the in-memory store stops blocking cross-worker replays. **Migrate to SQLite or Redis** at that time (the public surface of `NonceStore` is intentionally minimal so the swap is transparent).
- Nonces are not signed — an attacker who can MITM the operator could mint a fresh nonce, but they would also need the HMAC secret to sign it, so this does not weaken the existing trust model.

### Tests

`tests/test_admin_hmac_nonce.py` — 11 tests, all pass:
- 5 unit tests on `NonceStore` (first use, replay rejection, TTL expiry, max-entries eviction, empty-string rejection).
- 6 integration tests on `require_admin` (fresh OK, replay 401, cross-route 401, missing-nonce 401, legacy-off-mode 200, stale-timestamp 401).

Plus `tests/test_auth.py::TestAdminEndpoints` (34 tests) regression: helper `_admin_headers` now mints nonces and binds to path; all 4 admin endpoints under test continue to pass.

---

## 3. DG-056 (F-04) — `users.api_key_id` not UNIQUE

**Severity** Medium. **Status** ✅ Closed.

### Vulnerability

`UserTierManager` schema v1 declares `api_key_id INTEGER` with only a non-unique index (`idx_users_api_key`). Two distinct `users` rows could share the same `api_key_id`, meaning a leaked or shared key could be silently bound to a second user under a different tier. `link_api_key()` happily wrote the duplicate.

### Fix

1. **Schema bumped to v2** (`SCHEMA_VERSION = 2`). New migration step in `_migrate()`:
   ```sql
   CREATE UNIQUE INDEX IF NOT EXISTS uq_users_api_key
       ON users(api_key_id) WHERE api_key_id IS NOT NULL;
   ```
   Partial index — NULL is permitted multiple times so FREE/anonymous users without a generated key are unaffected.
2. **Online dedupe.** `_dedupe_api_key_links()` runs before the index is created. If pre-existing rows violate uniqueness, the oldest `user_id` keeps the key and the others have their `api_key_id` set to NULL with a WARN log per duplicate so operators can audit.
3. **API hardened.** `link_api_key()` returns `False` (not raise) on `IntegrityError`, with a WARN log. `create_user()` wraps `IntegrityError` in `ValueError` so callers get a clean exception either for `UNIQUE(email)` or `UNIQUE(api_key_id)`.

### Residual risk

The migration is one-way; rollback would require a manual schema downgrade. The dedupe is silent at the data layer — it WARN-logs but does not page. Operators should grep the WARN line after the first prod migration and reach out to affected duplicate users.

### Tests

`tests/test_dg056_unique_api_key.py` — 6 tests, all pass:
- Schema state (v2 marker, partial unique index present, multiple NULLs allowed).
- Uniqueness enforcement (`create_user` raises, `link_api_key` returns False, fresh keys still link).
- Migration from v1→v2 with pre-seeded duplicates: oldest user retains key, newer is nulled.

---

## 4. DG-057 (F-05) — `subscription_expires` never read

**Severity** Medium. **Status** ✅ Closed.

### Vulnerability

`UserTierManager.users.subscription_expires` exists in the schema and is written by the Stripe webhook handler when a subscription is created. **It was never read on the auth path.** A subscriber whose card had been cancelled retained whatever paying-tier access they had until their API key was manually revoked. The Stripe-side cancellation event did fire — it just had no enforcement endpoint.

### Fix

1. **Auth path enforces expiry.** `require_api_key` reads `user["subscription_expires"]` after fetching the user row; if `_subscription_is_expired(expires_raw)` returns `True`, raises `HTTPException(status_code=402, detail="Subscription expired — renew to restore access")`.
2. **402 chosen over 401** — semantically distinct from "invalid key" and lets the client surface a payments-renewal CTA rather than a generic auth error.
3. **TZ-tolerant.** `_subscription_is_expired` accepts both naive and aware ISO timestamps; naive strings are treated as UTC (matching how `tier_manager.create_user` writes via `datetime.now().isoformat()`).
4. **Defensive parsing.** A malformed `expires_raw` (corrupt write, schema drift) is treated as *not expired* with a WARN log, on the principle that a parse bug must not lock paying customers out. Operators are paged via the WARN line.
5. **Setter exposed.** `UserTierManager.set_subscription_expires(user_id, expires_iso)` writes (and clears with `None`) the timestamp; used by tests and the Stripe webhook handler.

### Residual risk

- A user whose card lapses keeps access for up to (current-time minus `subscription_expires`) — i.e. the renewal grace period. This is policy, not a bug — set the Stripe webhook to push `subscription_expires` forward only on confirmed payment.
- No grace period is built in. If product wants a 7-day soft-fail window, add it at the Stripe webhook layer (write `expires + 7d`), not in `_subscription_is_expired`. Auth must remain a hard gate.

### Tests

`tests/test_dg057_subscription_expires.py` — 10 tests, all pass:
- 5 unit tests on `_subscription_is_expired` (past/future, aware/naive, empty, garbage).
- 2 unit tests on `set_subscription_expires` (write+read, clear).
- 3 integration tests on `require_api_key` (expired → 402, active → 200, no-expires-field → 200).

---

## Out-of-scope (next sprint or later)

These were called out in eval_10–15 but are **not** in Sprint 2 scope:

- **Markdown injection in Telegram narratives** (eval 14). Sprint 4 (DG-054).
- **API key rotation cadence + key derivation hardening** (eval 12). Sprint 6 (DG-029-MODIFIED).
- **Per-tier rate-limit headers RFC 6585** (eval 10). Not in any Sprint 1-6 deliverable; tracked for Phase 2 commercial polish.
- **Audit ledger for admin actions** (SECURITY-2B.1 — partly already wired via `admin_action_log` on AppState). Sprint 4 OBS work depends on it.
- **Geo-block + sanctions list** (eval 29). Already shipped in Sprint W1+W2+W3 compliance work — not Sprint 2's concern.

---

## Operator migration checklist

- [ ] Confirm all deployments set `SENTINEL_TESTING_MODE=0` (or omit it — default is now safe).
- [ ] Migrate ops admin scripts to send `X-Admin-Nonce` and sign canonical `ts:nonce:path`. Reference impl: `tests/test_admin_hmac_nonce.py::_sign` and `tests/test_auth.py::_admin_headers`.
- [ ] Run prod DB migration during next maintenance window — `UserTierManager` v1→v2 runs at first import, dedupe WARNs go to stdout/log shipper.
- [ ] Verify Stripe webhook handler writes `subscription_expires` on every renewal/cancellation event (`POST /webhooks/stripe`).
- [ ] Sunset legacy admin HMAC mode (`ADMIN_NONCE_REQUIRED=off`) within one release cycle once all scripts are migrated. Deprecation warning is already in place.

---

## Sign-off

Sprint 2 Sécurité critique — **closed**. 61/61 net-new + regression tests pass. No remaining open findings within the Sprint 2 scope. Proceeding to Sprint 3 — Chatbot pilier (DG-110/111/112/114/042).
