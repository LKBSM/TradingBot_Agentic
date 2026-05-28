# Variables d'environnement — Vague 1

Liste exhaustive des env vars à configurer.

Convention : `MAJUSCULES_AVEC_UNDERSCORES`. Secrets via Fly.io `fly secrets set`. Variables non-secrètes dans `fly.toml [env]`.

---

## 🔒 Secrets (via `fly secrets set`)

### Anthropic / LLM
- `ANTHROPIC_API_KEY` — clé API Claude (déjà en place)

### Telegram
- `TELEGRAM_BOT_TOKEN` — token bot principal
- `TELEGRAM_PUBLIC_CHANNEL_ID` — chat_id du channel public track record (DG-072)

### Stripe
- `STRIPE_API_KEY` — clé secrète Stripe (live mode S6, test mode avant)
- `STRIPE_WEBHOOK_SECRET` — secret de signature des webhooks

### Database / persistence
- `SIGNAL_DB_PATH` — défaut `./data/signals.db`
- `USER_DB_PATH` — défaut `./data/users.db`

### Analytics / Observability
- `PLAUSIBLE_URL` — défaut `https://analytics.mia.markets`
- `PLAUSIBLE_DOMAIN` — défaut `mia.markets`
- `SENTRY_DSN` — DSN Sentry (DG-033-MODIFIED)

### Email transactionnel (S5+)
- `EMAIL_PROVIDER` — `postmark` / `resend` / `brevo`
- `EMAIL_API_KEY` — clé API du provider
- `EMAIL_FROM` — `M.I.A. Markets <support@mia.markets>`

### Trading Economics (M2+)
- `TRADING_ECONOMICS_API_KEY` — clé API TE (DG-027)

---

## 🌐 Configuration produit (non-secrètes, `fly.toml [env]`)

### Mode et environnement
- `SENTINEL_TESTING_MODE=0` — **STRICT en prod** (DG-041)
- `LOG_LEVEL=INFO` — `INFO` prod, `DEBUG` dev
- `LOG_FORMAT=json` — JSON prod, `text` dev
- `ENV=production` — `production` / `staging` / `development`

### Géographie et compliance
- `GEO_BLOCK_ENABLED=true` — bloquant en prod
- `GEO_BLOCK_MODE=allowlist` — `allowlist` (V0 bootstrap) | `denylist` (V2 quand legacy)
- `GEO_BLOCK_ALLOWED_COUNTRIES=FR,BE,CH,LU` — V0 bootstrap (DG-045)
- `DISCLOSURE_MODE=qualitative` — DG-047 MiFID
- `JURISDICTION_BLOCKED=US,QC,UK,GB,OFAC` — info dans payload InsightSignalV2

### Scoring + algo
- `SCORING_VERSION=v2` — v2 = CalibratedConvictionPipeline (DG-025), fallback v1 si modèle absent
- `VOL_MODE=har` — `har` (default 2026-04-29), `lgbm`, `hybrid`
- `NARRATIVE_MODE=llm` — `llm` (DG-042 défaut prod), `template` (fallback)
- `EDGE_CLAIM=false` — STRICT false jusqu'à PF live > 1.20 sur 12 mois (DG-081)

### Instruments GA
- `SYMBOLS=XAUUSD,EURUSD` — DG-074 (drop BTC/US500/JPY/GBP marketing)

### Tier rate-limit + quotas
- `TIER_RATE_LIMIT_ENFORCEMENT=warn` — `warn` S3-S4, `enforce` S5 (DG-006)
- `HARD_CAPS_ENABLED=true` — DG-046
- `GLOBAL_PAID_USERS_CAP=50` — cap stratégie bootstrap M1-M3
- `FREE_DAILY_SIGNALS_CAP=3`
- `STARTER_MONTHLY_SIGNALS_CAP=200`
- `PRO_MONTHLY_SIGNALS_CAP=800`
- `INSTITUTIONAL_MONTHLY_SIGNALS_CAP=2000`
- `FREE_DAILY_CHAT_QUESTIONS=5`
- `STARTER_DAILY_CHAT_QUESTIONS=100`

### Stripe
- `STRIPE_MODE=test` — `test` jusqu'à S6, puis `live`
- `STRIPE_TAX_ENABLED=true` — DG-044
- `STRIPE_PUBLIC_KEY` — clé publique (front-end)

### Compliance UX
- `COOKIE_BANNER_ENABLED=false` — non requis si Plausible self-hosted seul (DG-048 modifié)
- `MEDIATION_PLATFORM_URL` — à compléter post-DG-082 adhésion CM2C/MEDICYS

### Cost monitoring (DG-052)
- `ANTHROPIC_DAILY_COST_ALERT_USD=20` — seuil alerte Discord/email
- `ANTHROPIC_MONTHLY_COST_HARD_CAP_USD=500` — hard cap fail-closed

### Cache + perf
- `CACHE_BACKEND=local` — `local` V0, `redis` V3 (DG-020 DEFER MAU > 200)
- `SCORE_BUCKET_PTS=10` — bump cache hit rate (eval_05_09)
- `STRICT_DATA_QUALITY=true` — DG-053 fail-fast au boot

### Notifications
- `CIRCUIT_BREAKER_LLM_THRESHOLD=3`
- `CIRCUIT_BREAKER_TELEGRAM_THRESHOLD=5`
- `TELEGRAM_RETRY_MAX_ATTEMPTS=5`
- `TELEGRAM_DEDUP_TTL_SECONDS=3600`

### Frontend / Domain
- `NEXT_PUBLIC_API_URL=https://api.mia.markets`
- `NEXT_PUBLIC_PLAUSIBLE_DOMAIN=mia.markets`
- `NEXT_PUBLIC_CALENDLY_INSTITUTIONAL_URL` — lien Calendly demo INSTITUTIONAL (DG-080)

### CORS
- `CORS_ALLOWED_ORIGINS=https://mia.markets,https://www.mia.markets`

---

## 📋 Récap critique — variables à NE PAS oublier

| Var | Valeur prod | Pourquoi critique |
|---|---|---|
| `SENTINEL_TESTING_MODE` | `0` | Auth bypass si 1 → data leak |
| `EDGE_CLAIM` | `false` | Sinon claim non substanciable → risque légal |
| `GEO_BLOCK_ENABLED` | `true` | Sinon US/UK accessible → MiFID + OFAC |
| `STRIPE_MODE` | `live` (S6 only) | Test avant S6 obligatoire |
| `HARD_CAPS_ENABLED` | `true` | Sinon abuse free → OPEX LLM exploding |
| `STRICT_DATA_QUALITY` | `true` | Sinon feed 63% consommé silencieusement |
| `DISCLOSURE_MODE` | `qualitative` | MiFID finfluencer mars 2026 |
| `ANTHROPIC_MONTHLY_COST_HARD_CAP_USD` | `500` | Fail-closed protection bill explosion |

---

## Setup local dev

`.env.local` (dev only, **ne PAS commiter**) :

```bash
# === Anthropic ===
ANTHROPIC_API_KEY=sk-ant-...

# === Telegram ===
TELEGRAM_BOT_TOKEN=...
TELEGRAM_PUBLIC_CHANNEL_ID=-100...

# === Stripe (test mode) ===
STRIPE_MODE=test
STRIPE_API_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PUBLIC_KEY=pk_test_...

# === Mode dev ===
SENTINEL_TESTING_MODE=1
LOG_LEVEL=DEBUG
LOG_FORMAT=text
ENV=development
GEO_BLOCK_ENABLED=false
EDGE_CLAIM=false

# === Frontend ===
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_PLAUSIBLE_DOMAIN=localhost
```

`.gitignore` doit contenir : `.env`, `.env.local`, `.env.production`.
