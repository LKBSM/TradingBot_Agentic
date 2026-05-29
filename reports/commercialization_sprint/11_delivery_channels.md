# Plan de Commercialisation — Catégorie 11 : Delivery Channels

**Périmètre** : Telegram (B2C), Discord (B2C+broker), Webhooks B2B (HMAC), Email,
Push mobile, In-app inbox. Deduplication, retry/backoff, rate-limit providers,
multilingue (FR/EN/DE/ES), formatting Markdown/HTML, signature webhook,
DLQ et ack.

**Date** : 2026-05-21
**Auteur** : Sentinel commercialization sprint — agent #11
**Documents source** : `src/delivery/*`, `reports/eval_13_telegram.md`,
`reports/eval_10_15_team_audit.md`, `reports/architecture/dual_b2c_b2b_design.md`,
`mockups/*`, `src/api/disclaimers.py`.

---

## 1. État actuel (Audit)

### 1.1 Inventaire des composants livrés

| Module | Fichier | Statut | Note |
|---|---|---|---|
| Telegram notifier | `src/delivery/telegram_notifier.py` (1-299) | Partiellement durci | HTML parse_mode OK, mais sync-only, pas de retry, pas de rate-limit |
| Discord notifier | `src/delivery/discord_notifier.py` (1-265) | Plus mature que Telegram | embed riche, failures counter, mais pas de retry |
| Telegram lang store | `src/delivery/telegram_lang_store.py` (1-149) | Production-ready | SQLite + cache mémoire, FR/EN/DE/ES gated |
| Webhook signer HMAC | `src/delivery/webhook_signer.py` (1-190) | Production-ready | `t=…,v1=…` Stripe-style + replay guard 300s |
| Webhook delivery queue | `src/delivery/webhook_queue.py` (1-312) | Production-ready | In-process, retry exp backoff, DLQ, ack idempotent |
| Webhook drain worker | `src/delivery/webhook_drain_worker.py` (1-202) | Production-ready | asyncio loop avec stop event |
| Webhook ack endpoint | `src/api/routes/webhook_ack.py` (1-119) | Production-ready | POST `/api/v1/webhooks/deliveries/{id}/ack` + GET inspect |
| Disclaimers FR/EN/DE/ES | `src/api/disclaimers.py` (1-60) | Production-ready | Footer ≤ 280 chars par locale |
| Email | (absent) | NON LIVRÉ | Pas de module, pas de provider, pas de digest |
| Push mobile (FCM/APNs) | (absent) | NON LIVRÉ | Hors scope MVP |
| In-app inbox | (absent) | NON LIVRÉ | Hors scope MVP |
| Delivery dispatcher | (absent) | NON LIVRÉ | Pas de routage user-pref / fallback |

### 1.2 Constats critiques

**C1 — Telegram bot client SYNCHRONE et `python-telegram-bot` v20+ async-only**
`telegram_notifier.py:97-105` instancie `telegram.Bot(token=…)` puis appelle
`self._bot.send_message(...)` **synchrone** (ligne 252-257). Si la version
installée est ≥ v20 (cas par défaut de `pip install python-telegram-bot`
depuis 2023), `send_message` retourne une coroutine **jamais awaitée**.
Pénalité : `RuntimeWarning: coroutine was never awaited`, message
silencieusement perdu, `try/except` global masque la fuite. Aucun
test ne couvre ce chemin (`test_telegram_notifier.py` n'existe pas, glob vide).

**C2 — Aucun retry, aucun backoff sur Telegram**
`telegram_notifier.py:261-263` capture toute exception, log, return False.
Pas de retry sur 429 (`flood_wait`), pas de respect de `Retry-After`,
pas de circuit-breaker delivery, pas de DLQ. Idem Discord (`discord_notifier.py:246-265`).

**C3 — Aucune déduplication signal_id → chat_id**
Si le scanner ré-émet le même `signal_id` (relance, race condition replay,
double drain), Telegram et Discord enverront le même push N fois.
Aucun cache `(chat_id, signal_id) → ts` n'existe.
Conséquence commerciale : « spam » utilisateur → unsubscribe massif.

**C4 — Aucun rate-limit côté Telegram**
`send_to_multiple()` (lignes 265-292) itère séquentiellement et appelle
`send_signal` en boucle. À > 30 abonnés ou > 1 msg/s par chat, le bot
est flood-banni 1-24h (cf. eval_13 §4). Conséquence commerciale : 100
abonnés × 1 broadcast → ban Telegram et **toute la base perdue**.

**C5 — Markdown injection mitigée mais pas testée**
`telegram_notifier.py:34-44` escape `& < >` via `html.escape` et utilise
`parse_mode="HTML"` (ligne 255), ce qui ferme la faille `_*[]` du
legacy Markdown. **Mais** :
- aucune assertion de sécurité dans la base de tests (cherche `escape_html` → 0 match dans tests/).
- l'attaquant peut injecter `<script>` neutre côté Telegram, mais un user
  malveillant pourrait soumettre une narrative LLM contenant `<a href="javascript:…">`
  qui passe par `html.escape` (correctement) mais qui n'est testé nulle part.
- Discord embed (`discord_notifier.py:122-130`) n'échappe rien : `getattr(signal,…)`
  poussé directement dans `fields[].value`. Discord neutralise `<>` automatiquement
  côté client, **mais** mentions @everyone / @here ne sont pas filtrées
  (`allowed_mentions` jamais déclaré dans le payload `_post`).

**C6 — Aucun module Email**
Le mockup `mockups/telegram_b2c.txt` ligne 58 promet « digest hebdo
($14/mo) » au tier Analyst. Aucun module n'existe (`src/**/email*` → 0 résultats).
Conséquence : promesse marketing fausse, churn payant immédiat.

**C7 — Test coverage Telegram = 10 %** (eval 17 §3)
Seul `tests/test_telegram_lang.py` existe (12 tests). Aucun
`test_telegram_notifier.py`. Aucun `test_discord_notifier.py`. Zéro
test d'intégration delivery → assertion d'idempotence, escape, retry,
multilingue, fallback.

**C8 — Multilingue partiellement câblé**
`TelegramLangStore` (`telegram_lang_store.py`) résout
`chat_id → fr|en|de|es` correctement, le notifier appelle bien
`_resolve_lang` (ligne 84-92). **Mais** :
- la **narrative LLM** (`narrative_data["full_narrative"]`) reste en
  anglais 100 % du temps (cf. eval 5 LLM, prompt système fixed en EN).
- seul le **disclaimer footer** est traduit (4 locales).
- Discord notifier accepte le paramètre `lang` (ligne 72) mais il n'est
  jamais résolu via store, toujours `"en"` par défaut.

**C9 — Webhook B2B opérationnel mais sans publisher pipeline**
`webhook_queue.py` + `webhook_signer.py` + `webhook_drain_worker.py` +
`webhook_ack.py` sont livrés. **Mais** :
- aucun `WebhookPublisher` qui souscrit aux insights et enqueue.
- aucune table `webhook_subscriptions` (broker_url, secret_hash, tier, events).
- aucune endpoint `POST /api/v1/webhooks/subscribe` côté broker (cf. mockup
  `b2b_webhook_payload.json:25-29` qui référence un `subscription` object
  inexistant en base).
- aucun « auto-disable after 10 consecutive failures » (promesse `b2b_webhook_payload.json:105-108`).

**C10 — Discord webhook unique, pas de routage par utilisateur**
`discord_notifier.py:50` prend **une seule** URL webhook globale. Pas de
mapping `user_id → discord_webhook_url`. Donc inutilisable pour un
SaaS B2C multi-tenant : utilisable seulement comme canal admin
ou « broadcast to one channel ».

**C11 — Pas de queue pour Telegram, juste pour webhooks B2B**
La belle queue `WebhookDeliveryQueue` n'est utilisée que pour les push
broker. Telegram + Discord n'ont aucune queue intermédiaire — un crash
du scanner pendant `send_signal` perd définitivement le message.

**C12 — Pas d'observabilité delivery**
`TelegramNotifier.get_stats()` retourne `{messages_sent, bot_initialized}` —
aucun compteur de failures, aucun histogramme de latence, aucun
breakdown par status code, aucun export Prometheus. `WebhookDrainWorker.stats()`
expose des compteurs (`webhook_drain_worker.py:184-195`) mais ils ne
sont publiés sur **aucun endpoint** `/metrics`.

**C13 — Pas de tracing PII en logs**
Le `chat_id` Telegram est loggé en clair (`telegram_notifier.py:259`).
Sous RGPD, un `chat_id` est un identifiant personnel indirect. Logs
non-cleansés → risque P29 compliance.

### 1.3 Score audit par dimension

| Dimension | Note /10 | Justification |
|---|---|---|
| Compatibilité Telegram API | 4 | HTML parse_mode OK (eval_13 R1 corrigé). PTB v20 async cassure non vérifiée. |
| Robustesse erreurs | 3 | Pas de retry, pas de queue Telegram/Discord, exception générique. |
| Sécurité (escaping, signature) | 6 | HTML escape OK, HMAC OK, mais Discord mentions non filtrées, pas de tests injection. |
| UX message | 6 | Tier gating OK, vol regime OK, footers traduits, pas d'inline buttons, pas de deep-link. |
| Différenciation tier | 7 | Telegram + Discord OK pour 4 tiers, mais Discord plus riche (position sizing, exit embed). |
| Anti-spam / dédup | 1 | Aucun mécanisme. |
| Multilingue | 5 | Footer traduit, narrative non traduite, Discord lang non câblé via store. |
| Multi-canal / dispatcher | 2 | Pas de routage par préférence user, pas de fallback. |
| Webhooks B2B | 7 | Queue + signer + ack + drain LIVRÉS, mais publisher + table subscriptions absents. |
| Email / Push / Inbox | 0 | Non implémentés. |
| Observabilité delivery | 3 | Counters basiques, pas d'export Prom, pas de p95. |
| Compliance PII / logs | 4 | chat_id en clair, pas d'audit purge logs. |
| Tests | 2 | 10 % coverage Telegram, 0 % Discord, webhooks bien couverts. |
| **Global** | **4.0/10** | **Fondations livrées côté B2B, B2C fragile, email/dispatcher absents.** |

---

## 2. Vision cible

> 99.9 % SLA delivery sur 4 canaux (Telegram, Discord, Email, Webhooks B2B),
> idempotence garantie par `signal_id × subscriber_id`, retry avec backoff
> exponentiel + DLQ, multilingue FR/EN/DE/ES sur l'intégralité du payload
> (footer + narrative + scenarios), markdown-safe par construction,
> observable via `/metrics` (delivery_rate, p95_latency, dedupe_hit_rate),
> et conforme RGPD (logs sans PII).

### Objectifs quantitatifs go-live

| KPI | Cible 30 j | Cible 90 j |
|---|---|---|
| Delivery success rate (toutes channels) | ≥ 99.0 % | ≥ 99.9 % |
| p95 latence enqueue → ack provider | < 1 s | < 500 ms |
| Dedup hit rate (replay scanner) | mesuré ≥ 99 % | ≥ 99.5 % |
| Telegram 400 (bad markdown) | 0 % | 0 % |
| Telegram 429 (flood) incidents | 0 | 0 |
| Webhook DLQ size persistant | < 10 deliveries | 0 (drained ≤ 1h) |
| Multilang coverage (FR+EN+DE+ES sur payload complet) | 100 % footer, 100 % narrative | idem |
| Email open rate (digest hebdo) | mesuré | ≥ 25 % |
| Unsubscribe compliance < 24 h | 100 % | 100 % |

### Architecture cible (vue logique)

```
   SignalStateMachine            ┌────────────────────┐
   → SignalPublished event ───▶  │  DeliveryDispatcher │
                                 │  (NEW, src/delivery/dispatcher.py) │
                                 └──────────┬─────────┘
                                            │
                  ┌──────────────┬──────────┼──────────┬──────────────┐
                  ▼              ▼          ▼          ▼              ▼
        TelegramChannel  DiscordChannel  EmailChannel WebhookChannel  InAppChannel
                  │              │          │          │              │
                  ▼              ▼          ▼          ▼              ▼
            NotificationQueue (Redis Streams ou SQLite-backed)
                  │   item = {sub_id, signal_id, channel, payload_lang, retry_count}
                  │   idempotency_key = sha256(sub_id + signal_id + channel)
                  ▼
            DeliveryWorker pool (asyncio)
                  ├── token-bucket per provider (Telegram 30/s global, 1/s/chat ; Discord 5/s/webhook)
                  ├── retry: exp backoff jitter, max_attempts=5, DLQ après
                  ├── circuit-breaker par provider (déjà existant)
                  └── per-delivery audit row (provider_status, latency_ms, attempts)
```

---

## 3. Gap analysis

| Capacité | État actuel | Cible | Gap |
|---|---|---|---|
| Queue notifications globale (pas seulement webhooks) | webhook only | unifiée 4 canaux | **NEW NotificationQueue + DeliveryWorker** |
| Retry + backoff Telegram/Discord | aucun | exp backoff + DLQ | Réutiliser `WebhookDeliveryQueue` ou élargir |
| Dedup `signal_id × subscriber_id` | aucun | LRU 24h + audit | **NEW DeliveryDedupStore** (SQLite) |
| Telegram async v21+ | sync (cassure potentielle) | `httpx` direct Bot API | Refactor `_init_bot` + `send_signal` |
| Token bucket Telegram | aucun | 30/s global + 1/s/chat | **NEW TelegramRateLimiter** |
| Markdown safety | HTML escape OK | + tests fuzz + Discord mentions filter | Tests + `allowed_mentions={"parse": []}` |
| Multilang narrative (pas seulement footer) | EN only | FR/EN/DE/ES via LLM ou template | Hook `LLMNarrativeEngine.generate(lang=…)` (eval 5) |
| DeliveryDispatcher par préférence user | absent | table `user_channel_prefs` | **NEW** route module |
| Email | absent | Resend ou SendGrid + unsubscribe one-click | **NEW EmailChannel + table subscribers** |
| Webhook publisher (côté pipeline → enqueue) | queue OK, publisher absent | listener → `WebhookDeliveryQueue.enqueue()` | **NEW WebhookPublisher service** |
| Webhook subscriptions table | absente | (sub_id, broker_id, url, secret_hash, events, status) | **NEW migration + endpoint** |
| Webhook auto-disable | promis dans mockup | 10 fails consécutifs → status=disabled | Hook dans `WebhookDeliveryQueue.drain()` |
| Observabilité delivery | counters in-memory | Prom histogrammes + endpoint `/metrics` | Instrumenter |
| Logs sans PII | chat_id en clair | hash chat_id, last 4 chars | Refactor logger |
| Tests | 10 % | ≥ 85 % | **NEW** suites pytest delivery |

---

## 4. Plan d'exécution

### P0 — Queue notifications + retry (15 h)

**P0.1** — `src/delivery/notification_queue.py` (NEW, ~250 l, 5 h)
- Généraliser `WebhookDeliveryQueue` en `NotificationQueue` polymorphe
  qui accepte un `transport_factory: channel_name → Transport`.
- Schema d'item : `{delivery_id, channel, subscriber_id, payload, lang,
  enqueued_at, attempts, next_attempt_at, idempotency_key}`.
- Persistance SQLite (table `delivery_queue`) pour survivre aux restarts
  scanner — différence majeure avec la version in-memory actuelle.
- Acceptance :
  - 5 transports plugins (telegram/discord/email/webhook/inapp)
  - Persistance restart-safe : redémarrer le scanner ne perd pas la queue.
  - `pending_size` + `dead_letter_size` + `next_due_at` API conservées.
- Dépendances : `WebhookDeliveryQueue` existant (porter retry policy).

**P0.2** — `src/delivery/delivery_worker.py` (NEW, ~200 l, 4 h)
- Pool asyncio (`N=4` configurable) qui draine `NotificationQueue`.
- Token bucket per-provider (Telegram 30/s global, 1/s/chat ; Discord 5/s/webhook ;
  webhooks broker 10/s/url).
- Circuit-breaker par provider réutilisant `src/intelligence/circuit_breaker.py`.
- Acceptance :
  - 100 msgs Telegram → respect 30/s sans 429.
  - Provider down (mocked 5xx) → DLQ après 5 essais.
  - p95 enqueue → send < 500 ms idle.
- Dépendances : P0.1 + `circuit_breaker.py`.

**P0.3** — Refacto `TelegramNotifier` async + `httpx` direct (3 h)
- Remplacer `telegram.Bot(token).send_message` (sync, cassé v20+) par
  `httpx.AsyncClient.post(f"https://api.telegram.org/bot{token}/sendMessage", json=…)`.
- Conserver `format_signal_message` (statique, déjà testable).
- Garder API publique `send_signal()` mais devenir async (et le wrapper
  sync existant via `asyncio.run` pour compat) — cf. `telegram_notifier.py:213-263`.
- Acceptance :
  - Tests passent avec `python-telegram-bot` désinstallé (plus de dépendance).
  - Latence < 200 ms warm.
- Dépendances : aucune.

**P0.4** — Refacto `DiscordNotifier` retry + `allowed_mentions` (1 h)
- Wrapper `_post` (`discord_notifier.py:238-265`) dans retry exp backoff
  (réutiliser `_backoff_seconds` de `webhook_queue.py:97`).
- Forcer `allowed_mentions={"parse": []}` dans tout payload pour bloquer
  `@everyone`/`@here` injection depuis narrative LLM.
- Acceptance :
  - Test : narrative `@everyone` ne pingue pas.
  - 503 mocké → retry 3× avant échec.
- Dépendances : aucune.

**P0.5** — Brancher pipeline scanner → `NotificationQueue` (2 h)
- Dans `src/intelligence/main.py:202-220`, remplacer l'appel direct
  `notifier.send_signal(...)` par `notification_queue.enqueue(channel="telegram"|"discord", …)`.
- Démarrer le `DeliveryWorker` dans le lifespan FastAPI (`src/api/app.py`).
- Acceptance :
  - Scanner ne bloque plus sur send → tick scanner reste < 100 ms.
  - Restart scanner → queue persiste, livre les msgs en attente.
- Dépendances : P0.1 + P0.2 + P0.3.

### P0 — Markdown injection & escape hardening (3 h)

**P0.6** — Tests injection Telegram + Discord (2 h)
- `tests/test_telegram_md_injection.py` (NEW) :
  - narrative avec `<script>alert(1)</script>` → présent escapé `&lt;script&gt;…`.
  - narrative avec `<a href="javascript:…">` → escape complet.
  - chat_id non numérique → reject.
- `tests/test_discord_mentions_filter.py` (NEW) :
  - narrative avec `@everyone @here <@123>` → payload contient `allowed_mentions={"parse": []}`.
- Acceptance : 6+ tests verts.
- Dépendances : P0.4.

**P0.7** — Fuzz LLM output → format_signal_message (1 h)
- `tests/test_telegram_format_fuzz.py` (NEW) — `hypothesis` library :
  - String aléatoire injectée comme `full_narrative` → message reste valide
    HTML (parse via `html.parser` en assertion).
- Acceptance : 100 itérations Hypothesis passent.
- Dépendances : P0.6.

### P0 — Deduplication idempotente (5 h)

**P0.8** — `src/delivery/dedup_store.py` (NEW, ~120 l, 3 h)
- SQLite table `delivery_dedup (idempotency_key TEXT PRIMARY KEY, channel TEXT, subscriber_id TEXT, signal_id TEXT, sent_at REAL)`.
- `idempotency_key = sha256(f"{channel}|{subscriber_id}|{signal_id}")`.
- TTL configurable (default 7 jours).
- API : `seen_or_mark(channel, subscriber_id, signal_id) -> bool`.
- Acceptance :
  - Double enqueue même `signal_id` + même subscriber → second envoi skipped.
  - Test concurrence 10 threads × 100 inserts → 0 doublons.
- Dépendances : aucune.

**P0.9** — Intégrer dedup_store dans `DeliveryWorker` (2 h)
- Avant chaque `transport(...)` call, `if dedup_store.seen_or_mark(...): skip`.
- Compteur `deduped_count` exposé via `worker.stats()`.
- Acceptance :
  - Test : enqueue identique 5× → 1 seul send.
- Dépendances : P0.8.

### P0 — Multilang Telegram complet (8 h)

**P0.10** — `format_signal_message(lang=…)` traduit (3 h)
- Étendre `telegram_notifier.py:111-207` avec dict `LABELS[lang]` pour les
  intitulés (`Setup`, `Symbol`, `Score`, `Entry zone`, `Invalidation`, `Target`,
  `R:R Ratio`, `Volatility`, `Validation`, `Analysis`, `Upgrade to Analyst…`).
- Idem `discord_notifier.py:122-130` (fields names).
- Acceptance : 4 snapshots (fr/en/de/es) committés dans `tests/snapshots/`.
- Dépendances : aucune.

**P0.11** — Narrative LLM multilang (3 h, dépend agent #5 LLM)
- Hook `LLMNarrativeEngine.generate(insight, lang=…)` qui passe `lang`
  dans le prompt système Claude.
- Fallback : si LLM `lang=…` indisponible, template traduit dans
  `src/intelligence/template_narrative_engine.py` (4 locales).
- Acceptance : 4 narratives sample traduites, validation par lang detect.
- Dépendances : agent #5 LLM (template engine multilang).

**P0.12** — TelegramLangStore renforcement (2 h)
- Ajouter méthode `set_from_telegram_update(update)` qui extrait
  `update.effective_user.language_code` + `update.effective_chat.id`.
- Migration : commande `/lang fr|en|de|es` côté bot pour override.
- Acceptance : 5 tests `tests/test_telegram_lang.py` couvrent override.
- Dépendances : aucune.

### P0 — Webhook B2B publisher + auto-disable (10 h)

**P0.13** — Table `webhook_subscriptions` (SQLite migration) (2 h)
- Schema : `(sub_id TEXT PK, broker_id TEXT, url TEXT, secret_hash TEXT,
  events TEXT JSON, tier TEXT, status TEXT, consecutive_failures INT, created_at REAL)`.
- Module `src/delivery/webhook_subscriptions.py` (NEW).
- Acceptance : CRUD complet, secret jamais stocké en clair (sha256 + salt).
- Dépendances : aucune.

**P0.14** — Endpoint `POST /api/v1/webhooks/subscribe` (3 h)
- Body : `{url, events: ["insight.created", "insight.exit"]}`.
- Génère secret via `generate_webhook_secret()` (`webhook_signer.py:168`),
  retourne **une seule fois** au broker.
- Auth : `require_api_key` (broker tier ≥ BROKER_STANDARD).
- Acceptance : endpoint exposé via OpenAPI, 4 tests (success, dup url,
  invalid tier, secret non-relisible).
- Dépendances : P0.13.

**P0.15** — `WebhookPublisher` service (3 h)
- Listener sur l'event `InsightSignal` produit par pipeline.
- Pour chaque subscription `status=active` × event matching, enqueue
  dans `WebhookDeliveryQueue` (déjà existante).
- Payload conforme `mockups/b2b_webhook_payload.json` (event, delivery_id,
  subscription, data, compliance).
- Acceptance : 3 brokers mocked, 1 insight → 3 webhooks enqueués.
- Dépendances : P0.13 + `webhook_queue.py` existant.

**P0.16** — Auto-disable after N failures (2 h)
- Hook dans `WebhookDeliveryQueue.drain()` (`webhook_queue.py:155-213`) :
  quand `dead_lettered += 1` pour une `sub_id`, incrémenter
  `consecutive_failures` ; si ≥ 10, passer `status="disabled"` et envoyer
  email admin broker.
- Reset à `0` sur premier succès suivant.
- Acceptance : test mock 10 fails consécutifs → status flip.
- Dépendances : P0.13 + P0.15.

### P1 — Email delivery (16 h)

**P1.1** — Choix provider + module `src/delivery/email_channel.py` (NEW, 6 h)
- Provider : **Resend** (free tier 100/jour, 3000/mois — adapté MVP) OU
  SendGrid (free 100/jour). Trade-off : Resend a meilleure API, SendGrid
  meilleur deliverability international. **Reco Resend** pour le MVP.
- Templates HTML+text en `templates/email/` (signal_alert, digest_weekly,
  unsubscribe_confirm).
- API : `send(to, subject, html, text, lang, tags)`.
- Acceptance : envoi réel test box, tests mocked.
- Dépendances : env var `RESEND_API_KEY`.

**P1.2** — Table `email_subscribers` + `email_preferences` (2 h)
- `(user_id, email, lang, status, unsubscribe_token, confirmed_at, …)`.
- Double opt-in obligatoire (RGPD).
- Acceptance : confirmation email envoyé, click → `confirmed_at` set.

**P1.3** — Unsubscribe one-click compliance (3 h)
- Endpoint `GET /api/v1/email/unsubscribe?token=…` (pas d'auth user).
- Header `List-Unsubscribe: <https://…/unsubscribe?token=…>` (CAN-SPAM /
  RGPD Article 21).
- Header `List-Unsubscribe-Post: List-Unsubscribe=One-Click` (RFC 8058).
- Acceptance : Gmail client affiche bouton « Unsubscribe ».

**P1.4** — Digest hebdomadaire Analyst+ (3 h)
- Cron `every Monday 09:00 UTC` enqueue 1 digest par subscriber avec
  les N meilleurs signaux 7j (filtré par lang).
- Template `digest_weekly.html` (mockup à créer).
- Acceptance : 1 user test reçoit 1 email par semaine.

**P1.5** — Brancher dans `DeliveryDispatcher` (2 h)
- Channel `email` ajouté à `NotificationQueue`.
- Préférence user : `user_channel_prefs.email = true`.
- Acceptance : flag user → email reçu en plus du Telegram.

### P1 — Push mobile FCM/APNs (12 h, OPT pour MVP)

**P1.6** — Module `src/delivery/push_channel.py` (NEW, 8 h)
- Provider : **Firebase Cloud Messaging** (gratuit, iOS+Android).
- Table `device_tokens (user_id, platform, token, last_seen)`.
- Refresh tokens auto-expirés (28 j FCM).
- Acceptance : test device réel reçoit notif.

**P1.7** — Brancher Dispatcher (2 h)
- Channel `push` ajouté.
- Acceptance : préférence user → push reçu.

**P1.8** — Tests + observabilité (2 h)
- Mock FCM API, tests retry + token-expired.
- Counter `push_sent`, `push_token_expired`, `push_failed`.

### P2 — In-app inbox + read receipts (10 h)

**P2.1** — Table `inbox_messages` (3 h)
- `(msg_id, user_id, signal_id, body_html, body_lang, sent_at, read_at)`.
- Endpoint `GET /api/v1/inbox?lang=…&unread_only=true`.

**P2.2** — Mark-as-read endpoint (2 h)
- `POST /api/v1/inbox/{msg_id}/read` → emit event `MessageRead`.
- Idempotent.

**P2.3** — Brancher Dispatcher + tests (5 h)
- Channel `inapp` toujours actif (fallback ultime).
- Acceptance : signal émis → 1 row inbox créée.

### Total chiffré

| Phase | Total h |
|---|---|
| P0 Queue + retry (P0.1-P0.5) | 15 |
| P0 Markdown safety (P0.6-P0.7) | 3 |
| P0 Dedup (P0.8-P0.9) | 5 |
| P0 Multilang (P0.10-P0.12) | 8 |
| P0 Webhook B2B publisher (P0.13-P0.16) | 10 |
| **P0 total** | **41 h** |
| P1 Email (P1.1-P1.5) | 16 |
| P1 Push mobile (P1.6-P1.8) | 12 |
| **P1 total** | **28 h** |
| P2 In-app inbox | 10 |
| **TOTAL** | **79 h** |

---

## 5. Tests & validation

### 5.1 Cible coverage

| Module | Baseline | Cible P0 | Cible P2 |
|---|---|---|---|
| `telegram_notifier.py` | 10 % | 85 % | 95 % |
| `discord_notifier.py` | 0 % | 80 % | 90 % |
| `notification_queue.py` | n/a | 95 % | 95 % |
| `delivery_worker.py` | n/a | 90 % | 95 % |
| `dedup_store.py` | n/a | 95 % | 95 % |
| `email_channel.py` | n/a | n/a | 85 % |
| `webhook_*` | ≥ 90 % | ≥ 90 % | ≥ 95 % |

### 5.2 Suites tests à créer

- `tests/test_telegram_notifier.py` — formatage 4 tiers × 4 lang × 2 directions = 32 snapshots.
- `tests/test_telegram_md_injection.py` — 6 cas injection (HTML, JS, mentions, balises Telegram non supportées).
- `tests/test_telegram_format_fuzz.py` — Hypothesis 200 itérations.
- `tests/test_discord_notifier.py` — embed structure, exit embed PnL, allowed_mentions.
- `tests/test_discord_mentions_filter.py` — `@everyone`, `@here`, `<@123>`.
- `tests/test_notification_queue.py` — persistance, retry, DLQ, multi-channel.
- `tests/test_delivery_worker.py` — rate limit, circuit breaker, concurrence.
- `tests/test_dedup_store.py` — idempotence, TTL, concurrence.
- `tests/test_webhook_publisher.py` — broker × event matching.
- `tests/test_webhook_auto_disable.py` — 10 fails consécutifs → status flip.
- `tests/test_email_channel.py` — provider mock, unsubscribe, double opt-in.
- `tests/test_delivery_dispatcher.py` — préférences user, fallback.

### 5.3 Tests d'intégration end-to-end

- `tests/test_smoke_delivery_e2e.py` — pipeline scanner → queue → 4 channels → audit DB.
- `tests/test_delivery_restart_persistence.py` — kill scanner mid-drain, redémarrer, vérifier remise.
- `tests/test_delivery_multilang_e2e.py` — 1 signal × 4 chats × 4 lang → 4 messages traduits.

### 5.4 Tests de charge

- Locust scenario : 1000 signaux × 100 abonnés × 4 channels en 60 s.
- Validation : p95 < 1 s, 0 message perdu, dedup hit rate > 99 %.
- Script : `scripts/load_delivery.py` (NEW).

### 5.5 Validation manuelle pré-go-live

- [ ] Envoyer 100 signaux test à 5 chats Telegram réels, 4 lang → 100 % reçus.
- [ ] Couper API Telegram (mock) → DLQ se remplit, drain reprend à reconnect.
- [ ] Inscrire 1 broker test, envoyer 5 signaux, vérifier HMAC côté broker.
- [ ] Tester unsubscribe email Gmail → bouton natif s'affiche.

---

## 6. Sécurité

### 6.1 Escape user content

- **Telegram HTML** (`telegram_notifier.py:34-44`) — `html.escape(str(s), quote=False)`
  appliqué partout : OK. Ajouter `quote=True` pour les attrs si jamais
  on génère `<a href="…">` (pas le cas actuellement, mais defensive).
- **Discord embeds** (`discord_notifier.py:122-130`) — forcer
  `allowed_mentions={"parse": []}` (P0.4). Discord neutralise auto les
  balises HTML mais pas les markdown mentions.
- **Tests fuzz** Hypothesis 200 itérations (P0.7) pour valider invariants.

### 6.2 Webhook signature

- HMAC-SHA256 sur `f"{timestamp}.{body}"` (`webhook_signer.py:60-83`) — déjà OK.
- Tolérance 300 s anti-replay (`webhook_signer.py:39`) — OK.
- `hmac.compare_digest` (`webhook_signer.py:157`) — anti timing-oracle OK.
- Rotation secret broker : ajouter endpoint `POST /api/v1/webhooks/{sub_id}/rotate-secret`
  (P0+, hors scope MVP mais à documenter).

### 6.3 Secrets storage

- Secret broker stocké en **sha256+salt** (`webhook_subscriptions` schema P0.13).
- Token Telegram bot via env `TELEGRAM_BOT_TOKEN` — jamais loggé.
- Audit log : `webhook_ack.py:47-62` masque déjà `subscriber["key_id"]`.

### 6.4 PII compliance

- **chat_id Telegram** — logger uniquement les 4 derniers chars
  (`*****1234`). Refacto `telegram_notifier.py:259` :
  `logger.info("Signal sent to Telegram chat ****%s", target[-4:])`.
- **Email** — jamais en clair dans logs. Logger `email_hash = sha256(email)[:8]`.
- **discord_webhook_url** — jamais loggé (contient le token dans le path).
- Audit purge : cron mensuel qui purge `delivery_dedup` + audit rows > 90 j.

### 6.5 Anti-spam / abuse

- Rate-limit par subscriber : max 50 msgs/jour/canal (configurable par tier).
- Cooldown : 1 signal max par symbol par chat par minute.
- Liste de blocage : permettre user de `/mute XAUUSD` côté Telegram.

### 6.6 RGPD / opt-out

- Endpoint `DELETE /api/v1/user/me` (right to be forgotten) purge :
  `telegram_lang`, `email_subscribers`, `device_tokens`, `inbox_messages`,
  `delivery_dedup`, `user_channel_prefs`.
- Acceptance : SLA 30 j max (RGPD).

---

## 7. Métriques

### 7.1 Exposées sur `/metrics` (Prometheus format)

| Métrique | Type | Labels | Description |
|---|---|---|---|
| `delivery_sent_total` | counter | `channel`, `lang`, `tier` | Messages envoyés avec succès |
| `delivery_failed_total` | counter | `channel`, `error_class` | Messages en DLQ |
| `delivery_retried_total` | counter | `channel`, `attempt` | Retries (par attempt #) |
| `delivery_deduped_total` | counter | `channel` | Skipped pour dedup |
| `delivery_latency_seconds` | histogram | `channel` | enqueue→provider_ack |
| `delivery_queue_size` | gauge | `channel`, `state` | pending / dead-letter |
| `webhook_subscription_active` | gauge | `tier` | Brokers actifs |
| `webhook_consecutive_failures` | gauge | `sub_id` | Compteur par sub |
| `telegram_rate_limit_hits_total` | counter | - | Hits 429 |
| `email_open_total` | counter | `template` | Open tracking (digest) |

### 7.2 Dashboards Grafana cibles

- **Delivery health** : delivery_rate (sum success / sum total), p95 latency, queue depth, DLQ size.
- **Per-channel breakdown** : split par Telegram/Discord/Email/Webhook.
- **Multilang split** : delivery_rate par lang (alerter si fr/de/es < 95 % et en ≥ 99 %).
- **Webhook brokers** : top 10 sub_id par failures.

### 7.3 Alertes (PagerDuty / email admin)

- `delivery_rate < 99 % sur 5 min` → P1.
- `delivery_queue_size > 1000 sur 10 min` → P1.
- `dead_letter_size > 50` → P2.
- `webhook_consecutive_failures > 5 sur 1 sub` → P3 (informer broker).

---

## 8. Risques & mitigations

| # | Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Telegram bot flood-ban (>30 msgs/s) | Élevée sans rate-limit | Toute la base perdue 1-24h | Token bucket P0.2 + canal public Telegram en plan B |
| R2 | Discord webhook URL rate-limit (5/s) | Moyenne | Messages perdus broker | Rate limit per-URL P0.2 |
| R3 | python-telegram-bot v20+ async cassure silencieuse | Élevée si non testé | Messages silently dropped | Migration `httpx` direct P0.3 + removal dependency |
| R4 | LLM narrative contient `<script>` ou markdown | Faible (Claude curated) | XSS Telegram preview | HTML escape OK + tests fuzz P0.7 |
| R5 | Resend / SendGrid outage | Moyenne | Email digest perdu | Fallback secondary provider OU best-effort (digest pas critique) |
| R6 | FCM token expiré silencieusement | Élevée (28j) | Push non délivré | Refresh tokens via /api/v1/push/register au startup app mobile |
| R7 | Webhook broker secret leaké | Faible | Replay attack | Tolérance 300s + endpoint rotate-secret P1 |
| R8 | Multilang LLM hallucine traduction inexacte | Faible-Moyenne | Confusion user FR/DE/ES | Template fallback déterministe (P0.11) si LLM down |
| R9 | Scanner crash mid-drain → message perdu | Moyenne | Confiance utilisateur | NotificationQueue persistant SQLite P0.1 |
| R10 | RGPD violation (chat_id en logs) | Moyenne | Amende CNIL | Refacto logs masqués P0 |
| R11 | Spam unsubscribe massif (UX too pushy) | Moyenne | Churn | Cooldown 1 signal/symbol/chat/min P0 |
| R12 | Broker auto-disabled à tort (10 fails transients) | Moyenne | Relation broker B2B | Notification email avant flip + manual re-enable endpoint |

---

## 9. Dépendances

### 9.1 Internes (autres agents commercialization)

- **Agent #5 LLM** — narrative multilang (P0.11) dépend du template engine
  traduit côté `src/intelligence/template_narrative_engine.py`.
- **Agent #signal_store** — schema `delivery_dedup` + `webhook_subscriptions`
  + `email_subscribers` doivent cohabiter avec migration unique.
- **Agent #observability** — endpoint `/metrics` doit accepter les
  métriques `delivery_*` (Prometheus collector).
- **Agent #compliance** — wording footers FR/EN/DE/ES déjà validé
  (`src/api/disclaimers.py`). Tout nouveau template email doit re-passer
  la revue compliance avant production.
- **Agent #auth** — `require_api_key` (`src/api/auth.py`) utilisé sur
  `POST /api/v1/webhooks/subscribe`. Tier `BROKER_STANDARD` doit être
  défini dans `tier_manager.py`.

### 9.2 Externes

- **Resend API key** (ou SendGrid) — provider email.
- **Firebase project** (FCM service account JSON) — push P1.
- **Telegram Bot Token** (déjà en env).
- **Redis** (optionnel) — si on veut élargir `NotificationQueue` vers
  Redis Streams au-delà de 10k msgs/jour. SQLite suffit pour MVP.

### 9.3 Tooling dev

- `httpx>=0.27` (HTTP async client) — déjà dans requirements.
- `hypothesis>=6.0` (fuzz testing) — à ajouter `requirements-dev.txt`.
- `python-resend>=0.7` ou `sendgrid>=6.10` (P1).
- `firebase-admin>=6.5` (P1.6).

---

## 10. Estimation totale & timeline

### 10.1 Récapitulatif effort

| Phase | Effort h | Calendar (1 dev FT) | Calendar (1 dev 4h/j) |
|---|---|---|---|
| **P0 (go-live blocker)** | **41 h** | 5-6 jours | 10-11 jours |
| P1 (Email + Push) | 28 h | 4 jours | 7 jours |
| P2 (Inbox) | 10 h | 1.5 jours | 2.5 jours |
| **TOTAL** | **79 h** | **~10-12 jours** | **~20 jours** |

### 10.2 Séquencement recommandé

**Sprint 1 (semaine 1)** — Foundations
- P0.1 NotificationQueue
- P0.2 DeliveryWorker + rate limit
- P0.3 Telegram async httpx
- P0.5 Brancher pipeline

**Sprint 2 (semaine 2)** — Hardening + B2B
- P0.4 Discord retry + mentions
- P0.6-P0.7 Tests injection + fuzz
- P0.8-P0.9 Dedup
- P0.13-P0.16 Webhook subs + publisher + auto-disable

**Sprint 3 (semaine 3)** — Multilang
- P0.10 format_signal_message lang
- P0.11 LLM narrative multilang (en coordination agent #5)
- P0.12 TelegramLangStore renforcement
- Tests + load test + go-live P0

**Sprint 4 (semaine 4)** — P1 Email
- P1.1-P1.5 Resend + double opt-in + digest + dispatcher

**Sprint 5+** — Push + Inbox (post go-live)

### 10.3 Critère go/no-go P0

- ☐ 4 channels (Telegram + Discord + Webhook + InApp placeholder) opérationnels.
- ☐ Delivery rate ≥ 99 % sur 24 h de scénario test (1 000 signaux).
- ☐ Dedup hit rate ≥ 99 % sur replay scanner.
- ☐ p95 latence enqueue → ack < 1 s.
- ☐ 0 message perdu sur restart scanner.
- ☐ 4 langues couvertes (footer + narrative).
- ☐ Coverage tests ≥ 85 % sur `src/delivery/*`.
- ☐ Audit sécurité passé (fuzz + injection + mentions filter).
- ☐ Aucun chat_id / email en clair dans logs.

---

## Annexe — Références fichiers + lignes

- `src/delivery/telegram_notifier.py:34-44` — escape HTML (eval_13 R1 corrigé).
- `src/delivery/telegram_notifier.py:97-105` — `_init_bot()` python-telegram-bot sync (à refacto P0.3).
- `src/delivery/telegram_notifier.py:213-263` — `send_signal()` à rendre async via `httpx`.
- `src/delivery/telegram_notifier.py:265-292` — `send_to_multiple()` séquentiel sans rate-limit (P0.2).
- `src/delivery/discord_notifier.py:50-58` — webhook URL unique, à étendre vers mapping user.
- `src/delivery/discord_notifier.py:122-130` — fields embed sans `allowed_mentions` (P0.4).
- `src/delivery/discord_notifier.py:238-265` — `_post` sans retry (P0.4).
- `src/delivery/telegram_lang_store.py:42-149` — store production-ready, à étendre `set_from_telegram_update` (P0.12).
- `src/delivery/webhook_signer.py:60-83` — `sign_payload` HMAC v1 conforme Stripe-style.
- `src/delivery/webhook_signer.py:124-160` — `verify_payload` replay-guard 300 s.
- `src/delivery/webhook_queue.py:106-313` — Queue in-process avec retry + DLQ + ack idempotent.
- `src/delivery/webhook_drain_worker.py:40-202` — Worker asyncio production-ready.
- `src/api/routes/webhook_ack.py:65-119` — ack + inspect endpoints.
- `src/api/disclaimers.py:23-56` — 4 langues footer ≤ 280 chars.
- `src/intelligence/main.py:202-220` — point d'intégration scanner → notifier (à brancher P0.5).
- `reports/eval_13_telegram.md:252-273` — Top 5 améliorations Telegram (alignées avec P0).
- `reports/architecture/dual_b2c_b2b_design.md:34-83` — Architecture dual cible.
- `mockups/b2b_webhook_payload.json:100-109` — Contrat retry policy + auto-disable broker.
- `mockups/telegram_b2c.txt:1-95` — UX cible Telegram FR/EN par tier.
- `mockups/risk_score_telegram.md:1-104` — Risk score 0-100 (extension future Telegram).

---

**Fin du plan — Catégorie 11 Delivery Channels**
