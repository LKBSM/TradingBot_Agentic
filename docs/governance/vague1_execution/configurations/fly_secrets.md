# DG-022 + DG-029-MOD — Fly.io secrets manifest

**Scope** : Sprint 6 deploy runbook. Defines the exact `fly secrets set`
commands the operator runs once after `fly launch`. This file is the
single source of truth for what *must* be set before `fly deploy`
succeeds against the prod app `mia-markets-api`.

Plain-env values are in `fly.toml` (`[env]` block). **Secrets** are
everything that should never enter a git repo: API keys, signing keys,
database passwords, webhook URLs.

---

## 1. Required secrets (fail-closed without them)

| Secret name | Purpose | Where to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | LLM narrative + chatbot calls | console.anthropic.com → API keys |
| `ADMIN_HMAC_SECRET` | Signs admin requests (DG-055 nonce mode) | Generate locally: `python -c "import secrets; print(secrets.token_hex(32))"` |

Operator command:

```bash
fly secrets set \
  ANTHROPIC_API_KEY=sk-ant-… \
  ADMIN_HMAC_SECRET=<from generator>
```

---

## 2. Optional but recommended

| Secret name | Purpose | Default behaviour if unset |
|---|---|---|
| `SENTRY_DSN` | Server-side error capture (DG-033) | Sentry disabled silently |
| `SENTRY_RELEASE` | Release tag for error grouping | "unknown" |
| `TELEGRAM_BOT_TOKEN` | Telegram delivery (DG-054) | Telegram delivery skipped |
| `TELEGRAM_CHAT_ID` | Default broadcast channel | None — caller must pass per-message |
| `STRIPE_SECRET_KEY` | Subscription webhooks (when Stripe live) | Subscription tier stays FREE |
| `STRIPE_WEBHOOK_SECRET` | Verifies Stripe webhook signatures | Webhooks rejected |
| `DISCORD_COST_ALERT_WEBHOOK_URL` | Cost-alert notifications (DG-052) | LoggingNotifier only |
| `LLM_DAILY_COST_ALERT_USD` | Threshold for the cost-alert watcher | $5 / day |
| `FRED_API_KEY` | Macroeconomic data via FRED | Macro fields stay null |

Operator command (run after the required ones):

```bash
fly secrets set \
  SENTRY_DSN=https://…@…ingest.sentry.io/… \
  TELEGRAM_BOT_TOKEN=… \
  TELEGRAM_CHAT_ID=-100… \
  STRIPE_SECRET_KEY=sk_live_… \
  STRIPE_WEBHOOK_SECRET=whsec_… \
  DISCORD_COST_ALERT_WEBHOOK_URL=https://discord.com/api/webhooks/… \
  LLM_DAILY_COST_ALERT_USD=5 \
  FRED_API_KEY=…
```

---

## 3. DO-NOT-SET in prod

- `SENTINEL_TESTING_MODE` — already pinned to `"0"` in `[env]`. Never override to `1` in prod (the DG-041 CI gate would catch it, but defence-in-depth).
- `ADMIN_NONCE_REQUIRED=off` — never set in prod. Legacy mode is for the one-release-cycle migration window only.
- `DATA_QUALITY_STRICT=off` — never set in prod. Disabling the gate masks corrupt feeds (DG-053).

---

## 4. Verifying the set

```bash
fly secrets list
```

The output should include every name in §1 and §2 that you intended to
set, with timestamps. Values are never shown — verify by hitting
`/health/deep` after deploy (returns the wired components but no values).

---

## 5. Rotation runbook

Rotation policy: every 90 days for keys, immediately on any incident.

```bash
# 1. Generate new value
NEW=$(python -c "import secrets; print(secrets.token_hex(32))")

# 2. Set as secondary (both valid during overlap window)
fly secrets set ADMIN_HMAC_SECRET_NEW="$NEW"

# 3. Update the loader to read SECRET_NEW first, fall back to SECRET
# (this is filed for Sprint 7+ — V1 ships single-key with manual cutover)

# 4. After 24h of clean operation, deprecate old:
fly secrets unset ADMIN_HMAC_SECRET
fly secrets set ADMIN_HMAC_SECRET="$NEW"
fly secrets unset ADMIN_HMAC_SECRET_NEW
```

---

## 6. CI / preview environments

Use `fly secrets set --app <preview-app-name>` against a **different**
Fly app for preview deploys. Never share prod secrets with preview —
the `ANTHROPIC_API_KEY` should be a separate sandboxed key with a
hard cost cap.

For CI runs that don't need real Anthropic calls, leave
`ANTHROPIC_API_KEY` unset; the chatbot route falls back to the 503
"no_api_key" response which the frontend handles gracefully (scripted
fallback messages).
