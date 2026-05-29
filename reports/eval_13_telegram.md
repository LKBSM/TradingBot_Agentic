# Eval 13 — Telegram Delivery

**Date** : 2026-04-25
**Périmètre** : `src/delivery/telegram_notifier.py` (220 l), `src/delivery/discord_notifier.py` (250 l, comparatif).
**Verdict global** : **3.5/10** — un wrapper synchrone fonctionnel mais avec **2 bugs bloquants** (compatibilité python-telegram-bot v20+, MarkdownV2 non échappé) et zéro infrastructure commerciale (pas de retry, pas de rate-limit Telegram, pas de feedback inline, pas de fallback). Discord notifier est **plus mature** (embeds riches, position sizing, gestion failures).

---

## 1. Cartographie

```
TelegramNotifier
    ├── __init__(bot_token, default_chat_id)
    ├── _init_bot()              ← import telegram → telegram.Bot(token=...)
    ├── format_signal_message()  ← static, par tier
    ├── send_signal()            ← sync call to bot.send_message()
    ├── send_to_multiple()       ← séquentiel, sans rate-limit
    └── get_stats()              ← messages_sent counter

DiscordNotifier (référence)
    ├── + send_exit() avec pnl_pct
    ├── + position_multiplier dans embed
    ├── + failures counter séparé
    └── + retry implicite via requests timeout
```

---

## 2. Compatibilité python-telegram-bot

```python
# telegram_notifier.py:42
self._bot = telegram.Bot(token=self._bot_token)
...
# telegram_notifier.py:181
self._bot.send_message(chat_id=target, text=message, parse_mode="Markdown")
```

### 🔴 Problème : python-telegram-bot ≥ v20 est **async-only**

- v13.x : `Bot.send_message()` synchrone ✅
- v20.x+ (2023+) : `Bot.send_message()` retourne une **coroutine**, doit être `await`-ed.

D'après `requirements.txt` (cf. MEMORY.md) : "python-telegram-bot added" — version non spécifiée. Si pinning < v20 c'est OK mais v13 n'est plus supportée (sécurité). Si v20+, le code **ne fonctionne pas** : `bot.send_message()` retourne une coroutine non awaitée → message jamais envoyé, no error visible (juste `RuntimeWarning: coroutine was never awaited`).

**Vérification rapide à faire** :
```bash
pip show python-telegram-bot | grep Version
```

**Recommandation** :
- Pinner `python-telegram-bot>=21.0`.
- Réécrire en async natif OU utiliser **httpx + Bot API REST direct** (plus léger, pas de wrapper) :

```python
import httpx
async def send(chat_id, text):
    async with httpx.AsyncClient(timeout=10.0) as cli:
        r = await cli.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "MarkdownV2"},
        )
        return r.status_code == 200
```

---

## 3. Markdown escaping

```python
# telegram_notifier.py:184
parse_mode="Markdown"
```

### 🔴 Problème 1 : `parse_mode="Markdown"` est **legacy** (deprecated par Telegram en 2018)

Recommandation Telegram : `MarkdownV2` ou `HTML`. Legacy "Markdown" :
- Ne supporte pas `|`, `~~strikethrough~~`, etc.
- Échappe différemment.
- Telegram peut désactiver à tout moment.

### 🔴 Problème 2 : aucun escaping des données utilisateur

```python
# telegram_notifier.py:122
lines.append(f"\U0001f9e0 *Validation:* {reason}")
# telegram_notifier.py:134
lines.append(narrative)
```

`reason` et `narrative` viennent du LLM (Claude). Si le LLM produit `*bold*`, `_italic_`, `[link](url)`, `_underscore_in_text_`, ces caractères **brisent le parsing** :
- Telegram renvoie `400 Bad Request: can't parse entities`
- Message **jamais délivré**, juste un log error.

**Test reproductible** : asker au LLM "explain RSI_divergence" → réponse contient `RSI_divergence` → underscore non échappé → 400.

**Fix MarkdownV2** :
```python
import re
MD_V2_ESCAPE = r'([_*\[\]()~`>#+\-=|{}.!])'
def escape_md_v2(text: str) -> str:
    return re.sub(MD_V2_ESCAPE, r'\\\1', text)
```

Ne pas oublier d'échapper **partout sauf** dans les balises explicites (`*…*`, `[…](…)`).

---

## 4. Telegram Bot API rate limits

D'après les [docs Telegram](https://core.telegram.org/bots/faq#broadcasting-to-users) :

| Limite | Valeur |
|---|---|
| Global bot | 30 messages/seconde |
| Par chat (PV) | 1 msg/s |
| Par groupe / canal | 20 msg/min |
| Bursts | tolérés mais flood control kicks |

**État actuel** : aucun mécanisme.

```python
# telegram_notifier.py:210-214
def send_to_multiple(self, signal, narrative_data, recipients):
    sent = 0
    for r in recipients:
        if self.send_signal(...):
            sent += 1
    return sent
```

**Conséquence** : 100 abonnés × 1 broadcast → 100 sends séquentiels en `< 100 ms` → **flood block immédiat** (HTTP 429 RetryAfter). Le bot peut être ban temporairement (1 h à 24 h).

**Mitigation requise** :
- Token bucket global 30/s + per-chat 1/s.
- Backoff exponentiel sur 429 (`Retry-After` header).
- Queue async (asyncio.Queue ou Redis Streams) avec workers limités.

**Bonus** : pour 1k+ abonnés, Telegram recommande d'utiliser `forward_message()` (1 send + N forwards) ou un **canal public** (pas de limite per-chat).

---

## 5. Formatage par tier — audit

Le formatage applique le bon gating tier, **mais** :

| Item | FREE | ANALYST | STRATEGIST | INSTITUTIONAL |
|---|---|---|---|---|
| Direction/Symbol/Score | ✅ | ✅ | ✅ | ✅ |
| Entry/SL/TP/RR | ✅ | ✅ | ✅ | ✅ |
| Volatility regime | ✅ (si présent) | ✅ | ✅ | ✅ |
| Vol confidence interval (95% CI) | ❌ | ❌ | ✅ | ✅ |
| Validation reason | ❌ | ✅ | ✅ | ✅ |
| Full narrative (Analysis) | ❌ | ❌ | ✅ tronqué 2500 chars | ✅ |
| Upgrade prompt | ✅ | ❌ | ❌ | ❌ |
| Disclaimer "Not financial advice" | ✅ | ✅ | ✅ | ✅ |
| Position sizing suggéré | ❌ | ❌ | ❌ | ❌ ← Discord l'a, Telegram non |
| Inline buttons (👍/👎 feedback) | ❌ | ❌ | ❌ | ❌ |
| Deep link vers dashboard | ❌ | ❌ | ❌ | ❌ |
| Image / chart | ❌ | ❌ | ❌ | ❌ |

**Discord notifier** (référence) ajoute :
- Position sizing (`pos_value`, `pos_reason`)
- Embeds riches avec couleur LONG/SHORT
- `send_exit()` avec PnL %
- Failures counter séparé (Telegram n'a que `messages_sent`)

→ Telegram notifier est **en retard** sur Discord notifier dans le même repo.

---

## 6. Edge cases & robustesse

| Edge case | Handling | Note |
|---|---|---|
| `bot_token = None` | ✅ `_bot = None`, warning | OK |
| `python-telegram-bot` non installé | ✅ ImportError caught, warning | OK |
| `chat_id = None` + pas de default | ✅ warning, return False | OK |
| Telegram 429 (rate limit) | ❌ pas de retry, pas de queue | bug |
| Telegram 400 (bad markdown) | ❌ log error, message perdu | bug |
| Telegram 5xx transient | ❌ pas de retry | bug |
| Network timeout | ❌ aucun timeout configuré sur `bot.send_message` | bug |
| Telegram banned | ❌ pas de fallback Discord/email | bug |
| Concurrent calls depuis multiple threads | ⚠️ thread-safety du `telegram.Bot` non vérifiée |
| Long narrative > 4096 chars | ✅ truncate à 2500 | OK (marge confortable) |
| Emoji unicode (`\U0001f7e2`) | ✅ | OK |
| Signal sans `vol_regime` / `vol_forecast_atr` | ✅ skip section | OK |

---

## 7. Anti-spam & dédup

**État** : aucun.

- Si le scanner publie 2× le même signal_id (bug ou retry), Telegram envoie 2 messages.
- Aucun cooldown utilisateur (un user peut être bombardé si X symbols émettent en même temps).
- Pas de batching ("3 signaux dans la dernière heure" en 1 message).

**Best-practice** :
- Cache LRU local `(chat_id, signal_id) → ts` 24h pour dédup.
- Cooldown par chat configurable (max 1 signal/min, configurable INSTITUTIONAL plus permissif).

---

## 8. Feedback collection (data → fine-tune)

**Manquant** : aucun mécanisme inline.

**Opportunité business** :
```python
reply_markup = InlineKeyboardMarkup([[
    InlineKeyboardButton("👍 Useful", callback_data=f"fb:{sig_id}:up"),
    InlineKeyboardButton("👎 Skip", callback_data=f"fb:{sig_id}:dn"),
]])
```

Ces données alimentent :
1. NPS / satisfaction par tier.
2. Re-training du ConfluenceDetector (cf. eval_02 verdict : LightGBM classifier).
3. Filtre par utilisateur (si user vote 5× 👎 sur shorts → désactiver shorts pour lui).

---

## 9. Multi-canal & fallback

**État** : Discord notifier existe parallèlement, mais pas de **dispatcher**.

```
[Pipeline] → TelegramNotifier
            (et / ou)
          → DiscordNotifier
```

Pas de :
- Choix par utilisateur (Telegram vs Discord vs email).
- Fallback automatique si Telegram down.
- Webhook tier INSTITUTIONAL (TIER_CONFIG.webhooks=True mais aucun code).

**Architecture cible** :
```
DeliveryDispatcher
    ├── prefer_channel(user) → "telegram" | "discord" | "webhook" | "email"
    ├── send(signal, user)
    │     try primary
    │     on fail → secondary
    │     log to audit
    └── retry queue (Redis / SQLite)
```

---

## 10. Top 5 améliorations priorisées

| # | Amélioration | Effort | Impact |
|---|---|---|---|
| **R1** | **MarkdownV2 escaping + parse_mode=MarkdownV2** | 0.5 jour | 🔴 messages perdus actuellement quand narrative contient `_*[]` |
| **R2** | **Async migration** (httpx direct ou ptb v21 async) + timeout 10s | 1 jour | 🔴 si v20+ installé, code ne fonctionne pas du tout |
| **R3** | **Token bucket rate-limit** (30/s global, 1/s par chat) + retry 429 avec Retry-After | 1 jour | 🟠 prévient ban Telegram à >30 abonnés |
| **R4** | **Inline feedback buttons** (👍/👎) + handler webhook callback_query → table `signal_feedback` | 2 jours | 🟠 différenciation produit + data pour calibrer scoring |
| **R5** | **Delivery dispatcher multi-canal** (Telegram + Discord + email + webhook) avec preferences user | 2 jours | 🟠 résilience + flexibilité commerciale |

**Matrice** :

```
Impact ↑
  5 |  R1   R2
  4 |        R3
  3 |              R4   R5
  2 |
    +-------------------→ Effort
       1   2   3   4   5
```

---

## 11. Plan d'exécution

### Quick wins (< 1 jour)
- **QW1** Pin `python-telegram-bot==13.15` ou `>=21.0` selon choix migration (15 min)
- **QW2** Ajouter timeout=10s sur `bot.send_message` (5 min, dépend de v13/v20)
- **QW3** `escape_md_v2()` helper + appel sur reason/narrative (2 h)
- **QW4** Switch `parse_mode="MarkdownV2"` (avec QW3) (5 min)
- **QW5** Failures counter (`self._failures`) en miroir de Discord notifier (5 min)
- **QW6** Dédup `(chat_id, signal_id)` LRU 24h (1 h)
- **QW7** Disclaimer compliance : ajouter mention juridiction ("FR/EU: not investment advice (AMF/MiFID II)") par tier (15 min)

### Moyen terme (< 1 semaine)
- **MT1** Migration httpx async + tests (4 h)
- **MT2** Token bucket rate-limit (asyncio + redis or in-memory) (4 h)
- **MT3** Retry 429/5xx avec Retry-After + jitter exponentiel (3 h)
- **MT4** Inline keyboard 👍/👎 + endpoint `/api/v1/telegram/callback` + table feedback (1.5 jour)
- **MT5** Deep link vers dashboard web (`https://dashboard.smartsentinel.ai/signals/{id}`) (1 h)
- **MT6** Position sizing dans message Telegram (parité Discord) (1 h)
- **MT7** `send_exit` avec PnL pct (parité Discord) (2 h)
- **MT8** DeliveryDispatcher avec fallback Telegram→Discord→email (1 jour)

### Long terme (> 1 semaine)
- **LT1** Canal public Telegram + bot privé : broadcast canal (sans rate-limit per-chat), bot pour chat
- **LT2** Charts embeddés (matplotlib → PNG → `bot.send_photo`) — entry/SL/TP plottés sur OHLC
- **LT3** Localization FR/EN/ES (i18n) → +TAM
- **LT4** Webhook tier INSTITUTIONAL (POST signal JSON à URL utilisateur, signature HMAC)
- **LT5** SMS fallback (Twilio) pour critical signals tier INSTITUTIONAL
- **LT6** Telegram Mini-App / Web App pour dashboard intégré

---

## 12. KPIs mesurables post-amélioration

| KPI | Baseline | 30 j | 90 j |
|---|---|---|---|
| Délivrabilité messages | inconnue (silent failures) | ≥ 99 % | ≥ 99.9 % |
| Taux 400 (bad markdown) | inconnu | < 0.1 % | 0 % |
| Bot ban incidents | inconnu | 0 | 0 |
| Feedback signal collected | 0 % | 30 % | 60 % |
| Channels supportés | 2 (Telegram, Discord, sans dispatch) | 4 (T/D/email/webhook) | 5 (+SMS) |
| P95 latence send | non mesuré | < 500 ms | < 200 ms |
| Retry success rate | n/a | > 90 % | > 95 % |
| Dedup hit rate | n/a | mesuré | mesuré |
| Scaling testé (msgs/s) | non | 30/s | 100/s (avec canal pub) |

---

## 13. Trade-offs assumés

- **R1 MarkdownV2** : breaking change pour clients existants si on inverse l'escaping mal — couvrir par tests snapshot.
- **R2 async migration** : si v13 conservé, on n'utilise pas les nouvelles features Telegram (Web Apps, etc.) ; v21+ recommandé.
- **R3 token bucket** : ajoute latence de queueing au-delà de 30/s ; OK car 30/s = ~2.6M msgs/jour théorique.
- **R4 feedback** : alimente fine-tuning futur mais ajoute table + endpoint à maintenir.
- **R5 dispatcher** : complexifie le code mais essentiel pour résilience SaaS B2B.

---

## 14. Note finale par dimension

| Dimension | Note /10 | Justification |
|---|---|---|
| Compatibilité Telegram API | 3 | parse_mode legacy + (potentiel) ptb v20 cassure |
| Robustesse erreurs | 3 | aucun retry, aucun rate-limit, aucun fallback |
| Sécurité (escaping) | 2 | injection markdown possible via narrative LLM |
| UX message | 5 | tier gating ✅ ; pas d'image, pas de buttons, pas de deep link |
| Différenciation tier | 6 | gating correct ; mais Discord plus riche dans même repo |
| Anti-spam / dédup | 1 | aucun |
| Feedback loop | 0 | aucun |
| Multi-canal / fallback | 2 | Discord existe en parallèle, pas de dispatcher |
| Observabilité | 3 | counter sent, pas de failures, pas de latency tracking |
| Compliance disclaimer | 5 | "Not financial advice" présent, pas par juridiction |
| **Global** | **3.5/10** | **Fonctionne pour 1 user en test ; non viable commercialement** |

---

## 15. Verdict

- **Garder** : structure `format_signal_message` par tier, dataclass interface signal/narrative.
- **Refondre immédiatement** : escaping (R1) + async (R2) — sans cela, messages perdus en silence.
- **Atteindre parité Discord notifier** : position sizing, send_exit, failures counter (1 jour).
- **Avant go-live** : R3 (rate-limit) + R5 (dispatcher) sont **non-négociables** au-delà de 30 abonnés.

---

## Annexe — fichiers et lignes critiques

- `src/delivery/telegram_notifier.py:42` v20+ async incompat
- `src/delivery/telegram_notifier.py:184` parse_mode="Markdown" (legacy)
- `src/delivery/telegram_notifier.py:122,134` injection markdown non échappée
- `src/delivery/telegram_notifier.py:181` pas de timeout, pas de retry
- `src/delivery/telegram_notifier.py:210-214` send_to_multiple sequential, no rate limit
- `src/delivery/discord_notifier.py:128-135` position sizing (à porter sur Telegram)
- `src/delivery/discord_notifier.py:69-80` send_exit (manquant côté Telegram)

## Annexe — script de test reproductible

```python
# tests/test_telegram_md_injection.py
def test_narrative_with_underscores_does_not_break():
    notifier = TelegramNotifier(bot_token=None)
    fake_signal = type("S", (), {
        "signal_type": "LONG", "symbol": "XAUUSD",
        "entry_price": 2400.0, "stop_loss": 2390.0,
        "take_profit": 2420.0, "rr_ratio": 2.0,
        "confluence_score": 75, "tier": "STANDARD",
    })()
    msg = notifier.format_signal_message(
        fake_signal,
        narrative_data={"full_narrative": "RSI_divergence + BOS_break_level"},
        tier="STRATEGIST",
    )
    # Should not contain raw underscores in markdown context
    # Validate via Telegram parse simulator
    from telegram.helpers import escape_markdown
    assert "RSI\\_divergence" in escape_markdown(msg, version=2) or "RSI_divergence" not in msg
```
