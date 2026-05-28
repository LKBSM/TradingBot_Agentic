# Quotas et limites par tier

Référence : DG-046 hard caps + DG-006 tier rate-limit + DG-070 pricing v1.

---

## Tableau master des quotas

| Quota | FREE | STARTER | PRO | INSTITUTIONAL |
|---|---|---|---|---|
| **Actifs autorisés** | 1 (XAU) | 4 (XAU, EUR + 2) | 6 (XAU, EUR + 4) | illimité |
| **Lectures/jour** | 3 | 30 (proxy mensuel : 200) | illimité (proxy 800/mois soft) | illimité (proxy 2 000/mois soft) |
| **Lectures/mois (hard cap)** | 90 | 200 | 800 | 2 000 |
| **Chatbot questions/jour** | 5 | 100 | illimité | illimité |
| **Chatbot questions/mois (hard cap)** | 150 | 3 000 | 30 000 | 100 000 |
| **Sections débloquées** | Hero + verbale | + Conviction + Régime+Vol + Structure | + Détail technique (waterfall, conformal, RAG) | tout |
| **Alertes event imminent** | ❌ | ✓ | ✓ | ✓ |
| **Exports CSV** | ❌ | ❌ | ✓ | ✓ + API |
| **Email digest hebdo** | ❌ | ❌ | ✓ | ✓ |
| **API B2B JSON** | ❌ | ❌ | ❌ | ✓ |
| **Webhooks** | ❌ | ❌ | ❌ | ✓ HMAC signed |
| **SLA** | aucun | best effort | 99 % | 99,9 % |
| **Trial** | — | 14j sans CB | 14j avec CB | aucun (book demo) |
| **Refund 30j** | — | ✓ | ✓ | non (engagement 12 mois) |

---

## Cap global stratégie bootstrap

**Variable** : `GLOBAL_PAID_USERS_CAP=50`

Pendant M1-M3 (bootstrap légal), limite globale **50 abonnés payants** (STARTER + PRO + INSTITUTIONAL cumulés).

Si cap atteint :
- Inscription STARTER/PRO → message "Liste d'attente — nous reviendrons sous 7 jours"
- Capture email pour waitlist
- Notification automatique quand spot libre (resiliation, downgrade)

```python
# backend/src/api/middleware/global_cap.py
async def check_global_paid_cap():
    if not HARD_CAPS_ENABLED:
        return True
    current = await UserStore.count_paid_users()
    return current < GLOBAL_PAID_USERS_CAP
```

---

## Implémentation hard caps

```python
# backend/src/api/quota_manager.py
from datetime import datetime, timedelta
import sqlite3

TIER_QUOTAS = {
    "FREE":          {"signals_day": 3,   "signals_month": 90,    "chat_day": 5,   "chat_month": 150,    "assets_max": 1},
    "STARTER":       {"signals_day": 30,  "signals_month": 200,   "chat_day": 100, "chat_month": 3000,   "assets_max": 4},
    "PRO":           {"signals_day": 999, "signals_month": 800,   "chat_day": 999, "chat_month": 30000,  "assets_max": 6},
    "INSTITUTIONAL": {"signals_day": 9999,"signals_month": 2000,  "chat_day": 9999,"chat_month": 100000, "assets_max": 999},
}

async def can_view_signal(user) -> tuple[bool, str | None]:
    quotas = TIER_QUOTAS[user.tier]
    daily = await count_signal_views(user.id, since=datetime.utcnow().replace(hour=0, minute=0))
    monthly = await count_signal_views(user.id, since=datetime.utcnow().replace(day=1))

    if daily >= quotas["signals_day"]:
        return False, f"Daily signal limit reached ({quotas['signals_day']}/day). Upgrade to view more."
    if monthly >= quotas["signals_month"]:
        return False, f"Monthly signal limit reached ({quotas['signals_month']}/month). Upgrade for higher cap."
    return True, None


async def can_ask_chatbot(user) -> tuple[bool, str | None]:
    quotas = TIER_QUOTAS[user.tier]
    daily = await count_chat_questions(user.id, since=datetime.utcnow().replace(hour=0, minute=0))
    if daily >= quotas["chat_day"]:
        return False, f"Daily chatbot limit ({quotas['chat_day']}). Resets at midnight UTC."
    return True, None
```

---

## Mode `warn` vs `enforce` (DG-006)

Variable : `TIER_RATE_LIMIT_ENFORCEMENT=warn|enforce`

### Mode `warn` (S3-S4)
- Toutes les requêtes passent
- Si quota dépassé : header `X-Quota-Warning: 80%-of-monthly-limit-reached`
- Email user envoyé à 80 % + 100 % (mais service continue)
- Métrique Prometheus `quota_warn_count`

### Mode `enforce` (S5+)
- Requête refusée si quota dépassé : HTTP 429 + payload :
  ```json
  {
    "error": "quota_exceeded",
    "tier": "STARTER",
    "limit": 200,
    "used": 201,
    "reset_at": "2026-09-01T00:00:00Z",
    "upgrade_url": "/pricing"
  }
  ```
- Notification client UI claire avec CTA upgrade

---

## Soft cap UX 80 %

```tsx
// frontend/components/QuotaBanner.tsx
import { useQuotaUsage } from '@/lib/api';

export default function QuotaBanner() {
  const { used, limit, resetAt } = useQuotaUsage();
  const pct = (used / limit) * 100;

  if (pct < 80) return null;

  return (
    <div className={`px-4 py-2 text-sm rounded-md ${pct >= 100 ? 'bg-bearish/10 text-bearish border border-bearish/30' : 'bg-warning/10 text-warning border border-warning/30'}`}>
      {pct >= 100 ? (
        <>
          ⚠ Quota mensuel atteint ({used}/{limit}).
          <a href="/pricing" className="ml-2 underline">Débloquer plus →</a>
        </>
      ) : (
        <>
          ℹ Vous approchez votre quota mensuel ({used}/{limit}, {Math.round(pct)} %).
          <a href="/pricing" className="ml-2 underline">Upgrader →</a>
        </>
      )}
    </div>
  );
}
```

---

## Calcul "lectures consommées"

**Définition** : une "lecture" = un signal `InsightSignalV2` consulté par un user (event `signal_view`).

- Le même `signal_id` consulté plusieurs fois par le même user = **1 seule lecture** (idempotent par tuple `user_id × signal_id`)
- Cette idempotence évite que le user soit pénalisé pour avoir rafraîchi la page
- Implémentation : table `user_signal_views(user_id, signal_id, first_viewed_at)` avec unique constraint

```sql
CREATE TABLE IF NOT EXISTS user_signal_views (
  user_id TEXT NOT NULL,
  signal_id TEXT NOT NULL,
  first_viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (user_id, signal_id)
);

CREATE INDEX IF NOT EXISTS idx_usv_user_date ON user_signal_views(user_id, first_viewed_at);
```

---

## Évolution quotas (V2+)

| Période | Trigger | Action |
|---|---|---|
| M0-M3 (bootstrap) | — | `GLOBAL_PAID_USERS_CAP=50` strict |
| M3 (avocat V1) | MRR ≥ $1500 stable 60j | Lever cap à 200 |
| V2 (MAU > 200) | Redis branché (DG-020) | Pas de changement quotas mais perf cap counter |
| V2 (MAU > 500) | Stable | Réviser quotas à la hausse selon usage observé |
| V3+ | PMF B2C confirmé | Repenser tiers et quotas selon data réelle |

---

## Tests

```python
# backend/tests/test_quota_manager.py
@pytest.mark.asyncio
async def test_free_user_blocked_after_3_signals_per_day():
    user = await create_test_user(tier="FREE")
    for i in range(3):
        await record_signal_view(user.id, f"signal_{i}")
        assert (await can_view_signal(user))[0] is True

    await record_signal_view(user.id, "signal_4")
    can, reason = await can_view_signal(user)
    assert can is False
    assert "Daily signal limit" in reason


@pytest.mark.asyncio
async def test_global_cap_blocks_signup_at_50_paid():
    # Crée 50 paid users
    for i in range(50):
        await create_test_user(tier="STARTER")

    # 51ème tentative
    can_create = await check_global_paid_cap()
    assert can_create is False
```
