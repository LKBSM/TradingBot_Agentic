# Private-Beta Lockdown — Audit & Runbook (2026-07-06)

Branch: `feat/private-beta-lockdown` · Base: `origin/main` (0a69bed)

Mission: open the product to ~10 closed-beta testers with (a) server-side access
control on every endpoint, (b) 10 real hashed tester accounts, (c) working
navigation, (d) a 24/7 Docker deployment. Detection engine & product line
unchanged.

---

## 1. State of auth (before)

Two independent auth systems coexist:

| System | Module | Mechanism | Used by |
|---|---|---|---|
| A — Session/account (webapp) | `session_auth.py` + `account_store.py` | Signed `mia_session` cookie (itsdangerous) → opaque token (SHA-256 at rest) → Argon2id passwords | V2 product routes |
| B — API-key / HMAC (legacy) | `auth.py` | `X-API-Key` header / HMAC admin signature | `/api/v1/*` + revenue surfaces |

**The hole (confirmed):** in the running configuration (`SENTINEL_TESTING_MODE=1`,
`SUBSCRIPTION_GATE_ENFORCED=0` — the defaults), **every** data / AI / scanner /
view-control endpoint served an anonymous caller:

* **Session routes** (`market_reading`, `candles`, `chatbot`, `conditions_scan`,
  `live_price`) use `optional_account` + `enforce_*` guards that are **no-ops**
  while the subscription gate is OFF.
* **API-key routes** (`signals`, `state` = view-control, `dashboard`,
  `narratives`, `enrich`, `insight_history`, `operator`, `qa`, `webapp`,
  `webhook_ack`, `audit`) have `require_api_key` **bypassed** by
  `SENTINEL_TESTING_MODE=1` (`auth.py:423`).

Only HMAC-admin routes (`admin`, `admin_audit`, `metrics_latency`) were
protected in all cases.

Frontend: `middleware.ts` did locale routing only; `SubscriptionGate` redirected
to login **only** when `gate_enforced===true` and **failed open** on fetch error
→ the whole product (`/app`, `/zones`, `/scanner`) was reachable logged-out.

---

## 2. What was implemented

### A. Server-side gate — one blanket middleware (`middleware/beta_auth.py`)
`BetaAuthMiddleware`, wired in `app.py` right after `GeoBlockMiddleware`.
* No-op unless `BETA_LOCKDOWN=1` (so all existing tests & the current deployment
  are unaffected — enforcement is one deliberate env flip).
* When on: any request whose path is **not** on the public allowlist must carry
  a **valid** `mia_session` cookie resolving to an **active** account → else
  `401` with `{"error":"authentication_required"}` and **zero data**.
* Fronts BOTH auth systems at once → impossible to forget a route.
* Optional `BETA_ALLOWED_ROLES` (e.g. `tester,owner`) further restricts by role.
* Cookie crypto reused from `session_auth` (no new crypto). Fails **closed** if
  the account store is somehow unavailable.

Public allowlist (still reachable anonymously): `/health*`, `/api/access/me`,
`/api/auth/login|logout|me|register|password-reset/*`, `/api/v1/terms`,
`/api/v1/privacy`, `/api/v1/legal/*`, plus CORS preflight (`OPTIONS`).

`register` is reachable but the route itself returns **403** under lockdown
(public self-registration disabled — only seeded accounts exist).

`/api/access/me` now also returns `beta_lockdown` and `must_login` so the UI can
route to login even while the freemium gate stays OFF.

### B. Tester role + seed (`scripts/seed_testers.py`)
* New role `tester` added to `VALID_ROLES` (distinct from `owner`).
* Seed creates N (default 10) accounts `betaNN@<domain>` with **128-bit random
  passwords**, **Argon2id-hashed** in the real accounts DB. The 10 credentials
  print to the console **once**; only the hash is stored (irrecoverable).
* **Idempotent** (deterministic ids → re-run reports `exists`, no duplicates).
  Optional `--reset` rotates a lost password (and revokes sessions).

### C. Frontend lockdown (defence in depth)
* `middleware.ts` (edge): under `BETA_LOCKDOWN`, a cookieless visitor to a
  protected route (`/app`, `/zones`, `/scanner`, `/compte`, `/abonnement`) is
  307-redirected to `/connexion?next=…` before render.
* `SubscriptionGate`: honors `must_login`; under `NEXT_PUBLIC_BETA_LOCKDOWN=1`
  it **fails closed** (redirects to login) on transport error instead of open.
* `AccessSummary` type extended with `beta_lockdown` / `must_login`.

### D. Dockerization 24/7
* `infrastructure/Dockerfile.api` — FastAPI via **`uvicorn src.api.asgi:app`**
  (the correct V2 entrypoint, not the legacy `src.intelligence.main`), non-root,
  `curl` healthcheck on `/health`.
* `webapp/Dockerfile` — Next 15 **standalone**, non-root, healthcheck on `/`.
* `docker-compose.beta.yml` — `backend` + `frontend` + persistent named volume
  `mia-beta-data` (all SQLite DBs). `restart: unless-stopped`, healthchecks,
  `frontend depends_on backend: service_healthy`. Secrets via `.env` (never
  baked). `.env.beta.example` documents every var. `next.config.js` gains
  `output: 'standalone'`; `.dockerignore` added for the webapp.

No separate DB container: SQLite lives on the volume. The legacy
`docker-compose.yml` / `infrastructure/Dockerfile` are left untouched.

---

## 3. Endpoints — after lockdown (proof)

`tests/test_beta_lockdown.py` (9 tests, all green) asserts, with `BETA_LOCKDOWN=1`:

| Check | Result |
|---|---|
| Anonymous `GET` on `market-reading`, `candles`, `conditions-scan`, `dashboard/overview`, `v1/signals/state`, `v1/signals/current` | **401** (zero data) |
| `/health`, `/api/access/me` anonymous | **200** (public) |
| `/api/access/me` anon | `beta_lockdown:true`, `must_login:true` |
| Tester login → product endpoint | not 401 (past the wall) |
| Public registration under lockdown | **403**, no account created |
| Junk/forged session cookie | **401** |
| `BETA_ALLOWED_ROLES=owner` + tester session | **403** (role gate) |
| Flag OFF → anonymous call | **not** 401 (no-op, existing behavior intact) |
| Seed idempotent, Argon2id, no plaintext, login works | ✔ |

---

## 4. Navigation checklist (route-by-route)

Verified: the production build compiles every route; backend gating verified by
tests. Under lockdown, protected routes redirect to `/connexion` when
logged-out; after tester login they render.

| Route | Type | Status |
|---|---|---|
| `/` (landing) | public | OK (compiles, public) |
| `/methodology` | public | OK |
| `/conditions` | public (legal) | OK |
| `/confidentialite` | public (legal) | OK |
| `/connexion` | public (login) | OK — login form posts `/api/auth/login` |
| `/inscription` | public | OK — self-register blocked server-side (403) in beta |
| `/mot-de-passe-oublie` | public | OK |
| `/compte` | protected | OK — edge-redirect if logged-out; renders when authed |
| `/abonnement` | protected | OK — edge-redirect if logged-out |
| `/app` (chart/reading) | protected | OK — SubscriptionGate + edge wall |
| `/zones` (OB/FVG lifecycle) | protected | OK — SubscriptionGate + edge wall |
| `/scanner` | protected (paid) | OK — SubscriptionGate + edge wall |

No dead links or crashing pages were found; the change adds the login wall and
does not alter page content. Final live click-through is part of the operator's
`docker compose up` validation before merge.

---

## 5. Docker runbook

```bash
# 1) Configure secrets (once)
cp .env.beta.example .env
#    edit .env → SESSION_SECRET (openssl rand -hex 32), OWNER_*, and optionally
#    ANTHROPIC_API_KEY / TWELVE_DATA_API_KEY for live readings/chat.

# 2) Build + launch the whole stack (single command)
docker compose -f docker-compose.beta.yml up -d --build

# 3) Seed the 10 tester accounts ON the persistent volume (prints creds once)
docker compose -f docker-compose.beta.yml exec backend python scripts/seed_testers.py

# 4) Use it
#    Frontend  → http://localhost:3000   (redirects to /connexion until login)
#    Health    → docker compose -f docker-compose.beta.yml ps   (healthy)
```

Persistence: `accounts.db` / `signals.db` / `candles.db` / `market_readings.db`
live on the `mia-beta-data` volume → testers and readings survive
`docker compose restart` / `up -d --build`. `restart: unless-stopped` keeps both
containers up across host reboots.

Security posture: `SESSION_COOKIE_SECURE=0` for local http — set to `1` (and
front with HTTPS) for any public host. `BETA_LOCKDOWN=1` on both services.
Secrets only ever come from the non-committed `.env`.

---

## 6. Verification summary

* Backend: `tests/test_beta_lockdown.py` 9/9 green; touched-area regression
  (`-k "auth or access or entitlement or subscription_gate or account"`) 121
  passed (1 pre-existing order-dependent smoke test, passes in isolation).
* Frontend: `tsc --noEmit` clean; `SubscriptionGate` vitest 11/11 (incl. new
  lockdown redirect); production build green (standalone).
* No secret is committed or baked into an image; no plaintext password anywhere.
