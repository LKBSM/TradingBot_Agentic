# Audit produit — ce que le client voit AUJOURD'HUI

**Date :** 2026-04-30 — **Périmètre :** photo de l'existant, pas une refonte. Tout ce qui est marqué « non implémenté » ne tourne pas en prod.

**Sources principales lues :**
- `src/delivery/telegram_notifier.py`
- `src/delivery/discord_notifier.py`
- `src/intelligence/sentinel_scanner.py`
- `src/intelligence/signal_state_machine.py`
- `src/intelligence/confluence_detector.py`
- `src/intelligence/template_narrative_engine.py`
- `src/intelligence/main.py`
- `src/api/app.py`, `src/api/routes/{signals,narratives,dashboard}.py`, `src/api/models.py`, `src/api/signal_store.py`

---

## TL;DR

| Question | Réponse |
|---|---|
| Combien de types de messages le client peut-il recevoir ? | **Telegram : 1.** Discord : 2. (+1 ping admin Discord uniquement.) |
| Reçoit-il une notification de clôture (TP/SL/timeout) ? | **Telegram : NON. Discord : OUI.** |
| Le client a-t-il un dashboard web ? | **Non.** Il existe une API JSON (`/api/v1/...`), pas d'UI HTML servie par l'app. |
| Reçoit-il SL/TP en chiffres à l'ouverture ? | **Oui** (Telegram + Discord + API). |
| Reçoit-il une taille de position recommandée ? | **Discord : oui** (`position_multiplier`). **Telegram : non.** **API : non** (champ existe en interne, pas exposé dans `SignalResponse`). |
| Reçoit-il des updates en cours de trade ? | **Non, dans aucun canal.** |
| La narrative est générée à quel moment ? | **À l'ouverture uniquement.** Jamais sur exit. |

---

## Livrable 1 — Inventaire des messages Telegram

### Le notifier Telegram n'a qu'**une seule** méthode d'envoi de signal

`src/delivery/telegram_notifier.py:213` — `send_signal(signal, narrative_data, chat_id, tier, lang)`

Le format est produit par `format_signal_message` à `src/delivery/telegram_notifier.py:111`. Le contenu varie par tier (FREE / ANALYST / STRATEGIST / INSTITUTIONAL), pas par état de la state machine.

| Méthode | Fichier:ligne | Trigger | Actionnable ? |
|---|---|---|---|
| `send_signal` | `src/delivery/telegram_notifier.py:213` | Scanner — transition state machine `HOLD → BUY` ou `HOLD → SELL` (`src/intelligence/sentinel_scanner.py:634-655`) | Non, purement informatif. Aucun bouton inline, aucun callback. |
| `send_to_multiple` | `src/delivery/telegram_notifier.py:265` | Helper interne, broadcast du même signal à N chats. Pas un nouveau type. | — |

**Pas de méthode `send_exit` sur le `TelegramNotifier`.** Le scanner essaie via `getattr(self._notifier, "send_exit", None)` (`src/intelligence/sentinel_scanner.py:554`) — sur Telegram, ça retourne `None` et la notification de sortie est **silencieusement absente**.

**Pas de message dédié pour :** lockout, cooldown, news blackout, regime shift, score decayed, time expired, kill-switch trip, circuit ouvert. Ces événements existent dans la state machine mais ne sont pas push.

### Template du message d'ouverture (Telegram)

Format (HTML mode, `src/delivery/telegram_notifier.py:144-205`) :

```
🟢 <b>Smart Sentinel — Algorithmic Analysis</b>

<b>Setup:</b> BULLISH SETUP
<b>Symbol:</b> XAUUSD
<b>Score:</b> 78/100 (PREMIUM)

<b>Entry zone:</b> 2350.42
<b>Invalidation:</b> 2345.10
<b>Target:</b> 2362.50
<b>R:R Ratio:</b> 2.3:1

🟡 <b>Volatility:</b> Normal (ATR forecast: 5.32)
  <i>95% CI: [4.18 — 6.50]</i>      ← STRATEGIST+ uniquement

🧠 <b>Validation:</b> Long setup validated at score 78, R:R 2.30 — top
confluences: BOS (90%), OrderBlock (72%), FVG (60%)
                                          ← ANALYST+ uniquement

📊 <b>Analysis:</b>                       ← STRATEGIST+ uniquement
[paragraphe Market Setup + paragraphe Confluences + paragraphe Risk,
généré par Template ou LLM selon NARRATIVE_MODE — tronqué à 2500 chars]

<i>⚠️ Algorithmic analysis — not personalised investment advice...</i>
```

**Différences par tier :**
- `FREE` : pas de validation, pas de narrative, pied de page « 🔒 Upgrade to Analyst for AI validation ».
- `ANALYST` : ajoute le bloc « Validation ».
- `STRATEGIST` / `INSTITUTIONAL` : ajoute la narrative + 95% CI vol.

### Discord — pour comparaison

`src/delivery/discord_notifier.py` (rich embeds, pas du tout le même canal — `main.py:200-215` choisit Discord SI `DISCORD_WEBHOOK_URL` est set, sinon Telegram, jamais les deux).

| Méthode | Fichier:ligne | Trigger | Actionnable ? |
|---|---|---|---|
| `send_signal` | `discord_notifier.py:67` | HOLD→BUY/SELL — embed couleur vert/rouge | Non (webhook, pas de boutons). |
| `send_exit` | `discord_notifier.py:82` | ACTIVE→HOLD via state machine | Non. |
| `send_raw` | `discord_notifier.py:63` | Pings admin (startup, kill-switch trip) — `main.py:593-647`. Pas pour clients finaux. | — |

Discord ajoute aussi un champ **« Suggested Size »** (`discord_notifier.py:143-150`) qui n'apparaît PAS sur Telegram :
```
Suggested Size: 75% of baseline risk
                regime×news = 0.83 × 0.90 = 0.75
```

---

## Livrable 2 — Inventaire des niveaux livrés à l'ouverture

Pour un signal BUY type sur XAU/USD M15 (Telegram, payload réel) :

| Niveau | Livré ? | Preuve |
|---|---|---|
| **Prix d'entrée chiffré** | ✅ OUI | `Entry zone: {entry_price:.2f}` — `telegram_notifier.py:151` |
| **Stop-loss chiffré** | ✅ OUI | `Invalidation: {stop_loss:.2f}` — `telegram_notifier.py:152` |
| **Take-profit chiffré** | ✅ OUI | `Target: {take_profit:.2f}` — `telegram_notifier.py:153` |
| **R:R** | ✅ OUI | `R:R Ratio: {rr_ratio:.1f}:1` — `telegram_notifier.py:154` |
| **Taille de position recommandée** | ❌ NON sur Telegram, ✅ OUI sur Discord | `position_multiplier` est calculé `confluence_detector.py:337` mais Telegram **ne l'affiche pas**. Discord oui (`discord_notifier.py:143-150`). L'API JSON ne l'expose pas non plus dans `SignalResponse` (`api/models.py:51-62`). |
| **Durée de validité du signal** | ❌ NON | `max_signal_age_bars=64` (M15 ⇒ 16h) existe `signal_state_machine.py:136` mais n'est **jamais rendu** dans le message. Le client ne sait pas combien de temps le setup tient. |
| **Volatilité prévue (ATR forecast)** | ✅ OUI | `Volatility: Normal (ATR forecast: 5.32)` — `telegram_notifier.py:166-170`. **95% CI réservé STRATEGIST+** (`telegram_notifier.py:172-178`). |

**Calcul SL/TP** — `src/intelligence/confluence_detector.py:305-319` :
- `sl_distance = sl_atr_mult × ATR` (par défaut 2× ATR forecast)
- `tp_distance = tp_atr_mult × ATR` (par défaut 4× ATR forecast)
- En régime `high` vol, `sl_distance × 1.5` (TP inchangé pour ne pas dégrader hit rate)
- Arrondi au `_price_decimals` de l'instrument (XAU=2, FX=5, JPY=3)

---

## Livrable 3 — Lifecycle post-envoi

Une fois le message d'ouverture envoyé, voici ce que le client reçoit :

| Update | Telegram | Discord | Preuve |
|---|---|---|---|
| Mid-trade sur mouvement significatif | ❌ NON | ❌ NON | Aucun code n'envoie de message hors transition state-machine. Le scanner ne notifie qu'à `HOLD → BUY/SELL` et `ACTIVE → HOLD` (`sentinel_scanner.py:419-437`, `522-566`). |
| Re-évaluation (score change) | ❌ NON | ❌ NON | Le score est recalculé à chaque bar mais **rien n'est envoyé** tant que la state machine ne transitionne pas. Bars en `ARMING` (avant confirmation) sont silencieux. |
| Approche TP/SL | ❌ NON | ❌ NON | Aucun « almost there » / « TP within 0.5×ATR ». La sortie n'est annoncée qu'au moment où high/low touche TP/SL. |
| Clôture avec résultat | ❌ NON sur Telegram | ⚠️ PARTIEL sur Discord | **Telegram : aucune notification de clôture.** Discord envoie un embed « Analysis Closed » avec `Entry`, `Exit`, `Reason`, et `P&L %` (`discord_notifier.py:183-227`). Pas de R-multiple, pas de USD, pas de pips. La PnL est calculée `(exit-entry)/entry × 100`. |

**Ce qui est persisté en base mais NON push au client** (Telegram) :
- `signal_store.update_outcome(signal_id, outcome, pnl_pips)` — `src/api/signal_store.py:277-297`. Le résultat est dans la SQLite et accessible via `GET /api/v1/signals/history` (le client doit poller pour le voir).
- `outcome` ∈ {target_reached, invalidated, time_expired, score_decayed, regime_shifted, opposing_signal} — exposé dans `SignalHistoryItem` (`api/models.py:65-77`).

**Cas concret côté Telegram** : un utilisateur reçoit le signal BUY à 14h00, le SL est touché à 17h30. Aucun nouveau message. Il doit aller voir lui-même ou interroger l'API.

---

## Livrable 4 — Narrative LLM

| Question | Réponse | Preuve |
|---|---|---|
| Moment de génération ? | À l'ouverture uniquement, jamais sur exit. | `sentinel_scanner.py:445-468` — narrative générée **après** confirmation state machine, avant `_send_notification_safe`. Pas d'appel LLM dans `_publish_exit_transition`. |
| Une seule fois ou plusieurs ? | Une seule fois par signal. Si cache hit, aucun appel LLM ; sinon un appel. | `sentinel_scanner.py:450-468` |
| Score : chiffre ou label ? | **Les deux**, juxtaposés. | `Score: 78/100 (PREMIUM)` — chiffre brut + label tier (PREMIUM/STANDARD/WEAK) — `telegram_notifier.py:149` |
| La narrative explique-t-elle SL/TP ? | OUI sur le tier NARRATOR (Strategist+). | `template_narrative_engine.py:316-345` (`_paragraph_risk`) cite explicitement entry/SL/TP, le multiple ATR, et le R:R. Exemple rendu : *« A close below 2345.10 invalidates the long setup. Target at 2362.50 (4.0×ATR) for a 2.30:1 reward-to-risk. »* |
| Sur LLM ou Template ? | **Template par défaut** (`NARRATIVE_MODE=template` en `main.py:186`). LLM activé uniquement si `NARRATIVE_MODE=llm` + `ANTHROPIC_API_KEY`. | `src/intelligence/main.py:185-192` |
| Fallback ? | LLM circuit-breaker → Template (`fallback_used=true`). Le client ne sait pas qu'il a eu le fallback. | `sentinel_scanner.py:580-632` |

---

## Livrable 5 — Gap analysis

Comparaison à un produit « lifecycle complet » : niveaux clairs à l'ouverture + updates + clôture R/USD + narrative à chaque étape.

| Gap | État actuel | Effort dev (h) | Impact client | Priorité |
|---|---|---|---|---|
| **Notification de clôture sur Telegram** | Inexistante. Signal envoyé, jamais de suivi. Le client ne sait pas si son trade a touché TP, SL ou timeout. | 4-6h (méthode `send_exit` symétrique au Discord, branchement déjà en place côté scanner) | **FORT** — c'est le « moment de vérité » d'un service signaux. Sans ça, le produit paraît cassé. | **P0** |
| **PnL en R-multiple et USD à la clôture** | Discord a `(exit-entry)/entry × 100` (un %), pas un R, pas un $. Telegram n'a rien. Pas de notion de risk size côté client (pas de slot pour son équity). | 3-4h pour R-multiple (dispo via `pnl_pips / risk_distance`) ; 8-12h pour USD (besoin d'une saisie d'équity côté client + persistence) | **MOYEN-FORT** — R-multiple est l'unité pro. USD demande de récolter le capital de l'utilisateur. | P0 (R), P1 (USD) |
| **Taille de position dans le message Telegram** | `position_multiplier` calculé, livré sur Discord, **omis sur Telegram et omis dans `SignalResponse`**. | 1-2h (ajouter une ligne au template + un champ Pydantic) | **MOYEN** — gain immédiat sans nouveau calcul. | **P0** |
| **Durée de validité affichée** | `max_signal_age_bars=64` connue mais jamais montrée. | 1-2h (calculer ETA d'expiration à partir de la TF + l'ajouter au message) | **MOYEN** — réduit le « combien de temps ça reste valable ? ». | P1 |
| **Update mid-trade quand le score se dégrade au-dessus de l'exit_threshold** | Aucun message. Le client ne voit ni le « score à 60, fragile » ni le « score remonté à 80 ». | 8-12h (debounce, throttle, format alerté, branchement state machine en `ARMING` ou bar-level) | **FAIBLE-MOYEN** — risque de spam ; valeur perçue dépend de la qualité du signal de seconde dérivée. | P2 |
| **Approche TP/SL (ex : « 70% du chemin vers TP »)** | Aucune notification. | 4-6h | **FAIBLE** — le client peut le voir sur son broker. Effet « engagement » plus que « valeur ». | P2 |
| **Re-narrative sur exit (post-mortem court : « SL touché parce que… »)** | Aucune. La narrative s'arrête à l'entrée. | 6-10h pour Template ; +cost LLM si `NARRATIVE_MODE=llm`. | **MOYEN-FORT** — différenciateur majeur vs concurrents. Justifie la perte aux yeux du client. | P1 |
| **Dashboard web** | Inexistant. Pas de `StaticFiles`, pas de `Jinja`, pas de HTML servi. Juste API JSON. Le client n'a aucune surface visuelle officielle hors Telegram/Discord. | 40-80h MVP | **FORT** (pour un produit SaaS perçu comme sérieux) | P1 |
| **News blackout / kill-switch / circuit ouvert visibles** | Le scanner peut bloquer un signal (kill-switch, news blackout, regime filter, circuit) — silencieux côté client. | 2-3h pour Telegram via `send_raw`-like + 4h pour template par état. | **FAIBLE** — surtout admin. Mais utile pour la transparence en SaaS payant. | P2 |

---

## Annexes

### A. Chaîne complète de l'envoi (Telegram, signal d'ouverture)

```
ConfluenceDetector.analyze()                    confluence_detector.py:280
  └─ ConfluenceSignal(entry, sl, tp, rr, score, vol_*, position_mult)
       │
       ▼
SignalStateMachine.on_bar()                     signal_state_machine.py:408
  └─ HOLD → ARMING → ARMING (×confirm_bars=2) → ACTIVE_LONG/ACTIVE_SHORT
       │ retourne StateTransition(to_state=BUY|SELL, active_signal)
       ▼
SentinelScanner._scan_once()                    sentinel_scanner.py:419-444
  ├─ TemplateNarrativeEngine.generate_narrative() OU LLM            :458-468
  ├─ SignalStore.publish(SignalRecord)                               :695-708
  └─ TelegramNotifier.send_signal()                                  :634-655
       └─ format_signal_message() → bot.send_message(parse_mode=HTML)
```

### B. Chaîne complète de l'exit

```
SentinelScanner._step_state_machine()           sentinel_scanner.py:485-520
  └─ SignalStateMachine.on_bar() détecte:
       TP hit / SL hit / timeout / score decay / regime / opposing
       └─ retourne StateTransition(to_state=HOLD, exit_reason, exit_price)
            │
            ▼
SentinelScanner._publish_exit_transition()      sentinel_scanner.py:522-574
  ├─ SignalStore.update_outcome(signal_id, outcome, pnl_pips)    [PERSISTÉ]
  └─ getattr(notifier, "send_exit", None)
        ├─ Discord  : DiscordNotifier.send_exit() → embed         [ENVOYÉ]
        └─ Telegram : None → silently skipped                     [PERDU]
```

### C. Surface API côté client

| Endpoint | Renvoie | Inclut SL/TP ? | Inclut narrative ? | Inclut outcome ? |
|---|---|---|---|---|
| `GET /api/v1/signals/current` | Dernier signal publié | Oui | Non (lean) | Non |
| `GET /api/v1/signals/history` | Historique paginé | Oui | Non | **Oui** (`outcome`, `pnl_pips`, `closed_at`) |
| `GET /api/v1/narratives/{signal_id}` | Narrative gated par tier | Oui | Oui (FREE: levels-only ; ANALYST+: validation ; STRATEGIST+: full) | Non |
| `POST /api/v1/narratives/chat` | Réponse LLM contextuelle | — | — | INSTITUTIONAL only |
| `GET /api/v1/dashboard/summary` | Stats agrégées (PF, win-rate, Sharpe…) | Non (agrégé) | Non | — |
| `GET /api/v1/dashboard/equity-curve` | Courbe d'équity en pips | Non | Non | — |

Aucune route HTML — le client doit construire son propre front, ou utiliser uniquement Telegram/Discord.

### D. Ce qui n'est PAS exposé au client final aujourd'hui

- `position_multiplier` (existe interne, surface Discord seulement)
- Bars restantes avant `time_expired` (snapshot dispo via `state_machine.snapshot()` mais aucune route ne le sert au client)
- Raison de blocage d'un signal (regime filter, kill-switch, news blackout) — purement loggué
- Score historique (séries des bars en `ARMING`) — non persisté
- Statut « cache hit » ou « fallback template utilisé » — interne
