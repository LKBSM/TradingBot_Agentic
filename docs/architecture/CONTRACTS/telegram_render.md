# Contrat — Telegram render (`to_telegram_b2c`)

**Statut** : production V1
**Renderer** : `src/delivery/renderers/telegram.py` (à créer en Phase B)
**Source** : `InsightSignalV2` canonical
**Cible** : Telegram Bot API `sendMessage`

## Contraintes Telegram

- Longueur message max : **4 096 chars** (limite API).
- Cible UX : **≤ 800 chars** (lisibilité mobile + densité).
- Markdown ou HTML (privilégier MarkdownV2 pour escape consistant).
- Inline keyboard pour CTAs (boutons sous le message).

## Format de sortie

```python
class TelegramPayload(BaseModel):
    chat_id: int                       # depuis TelegramSubscription
    text: str                          # MarkdownV2, ≤ 800 chars
    parse_mode: Literal["MarkdownV2"]
    reply_markup: InlineKeyboardMarkup  # 2 boutons
    disable_web_page_preview: bool = True
```

## Template (français — `narrative_language="fr"`)

```
{emoji_direction} *Lecture {direction_fr} {instrument} · {timeframe}*

Conviction : *{conviction_label_uc}* ({conviction_0_100}/100)
Structure : {bos_part} · {fvg_part} · {retest_part}
Invalidation : *{structural_invalidation_fmt}*
Régime : {hmm_label_fr} · gate *{regime_gate_decision_fr}*
Volatilité : {vol_regime_fr} · forecast {forecast_vs_naive_pct:+.0f}% vs naïf
{event_line}

_« {narrative_short_truncated} »_

_Lecture algorithmique éducative. Ne constitue ni un signal de trading ni un conseil en investissement._
```

### Variables dérivées

| Variable | Source | Mapping |
|---|---|---|
| `emoji_direction` | `direction` | 🟢 BULLISH_SETUP / 🔴 BEARISH_SETUP / ⚪ NEUTRAL |
| `direction_fr` | `direction` | haussière / baissière / neutre |
| `conviction_label_uc` | `conviction_label` | FAIBLE / MODÉRÉE / STRONG / INSTITUTIONAL |
| `bos_part` | `structure_readout` | "BOS {bos_level_fmt}" si présent, sinon "" |
| `fvg_part` | `structure_readout` | "FVG {fvg_low_fmt}-{fvg_high_fmt}" si présent |
| `retest_part` | `structure_readout.retest_state` | "retest armé" / "retest en attente" / "retest passé" / "" |
| `structural_invalidation_fmt` | `structural_invalidation` | formaté avec `price_decimals` instrument |
| `hmm_label_fr` | `regime_readout.hmm_label` | trend_bullish→"trend haussier", range_low_vol→"range bas-vol", stress→"stress" |
| `regime_gate_decision_fr` | `regime_readout.regime_gate_decision` | TRADE→"FAVORABLE", REDUCE→"DÉGRADÉ", BLOCK→"HOSTILE" |
| `vol_regime_fr` | `volatility_readout.regime` | low→"faible", normal→"normale", high→"élevée" |
| `event_line` | `event_readout` | Si blackout actif : "⚠ Blackout news actif"; sinon si next_event_in_minutes ≤ 240 : "⚠ {next_event_label} dans {format_duration(next_event_in_minutes)}"; sinon : "Session {session_fr}" |
| `narrative_short_truncated` | `narrative_short` | tronqué à 220 chars + "…" si dépassé |

### Boutons inline

```python
keyboard = InlineKeyboardMarkup([
    [
        InlineKeyboardButton(
            text="📊 Voir détails",
            url=f"https://mia.markets/{locale}/insight/{signal_id}?utm_source=telegram&utm_medium=push"
        ),
        InlineKeyboardButton(
            text="💬 Demander à Sentinel",
            url=f"https://mia.markets/{locale}/chat?signal_id={signal_id}&utm_source=telegram&utm_medium=push"
        ),
    ]
])
```

## Mapping anglais (`narrative_language="en"`)

| FR | EN |
|---|---|
| Lecture haussière | Bullish reading |
| Lecture baissière | Bearish reading |
| Conviction | Conviction |
| Structure | Structure |
| Invalidation | Invalidation |
| Régime | Regime |
| Volatilité | Volatility |
| forecast +N% vs naïf | forecast +N% vs naive |
| trend haussier | bullish trend |
| range bas-vol | low-vol range |
| stress | stress |
| FAVORABLE / DÉGRADÉ / HOSTILE | FAVORABLE / DEGRADED / HOSTILE |
| Blackout news actif | News blackout active |
| {label} dans X min | {label} in X min |
| Lecture algorithmique éducative. Ne constitue ni un signal de trading ni un conseil en investissement. | Educational algorithmic reading. Does not constitute a trading signal or investment advice. |

Le mapping vit dans `messages/{fr,en}/telegram.json`.

## Règles d'invariants

1. **Length cap** : si rendu > 800 chars, raccourcir `narrative_short` avant tout autre champ.
2. **Hard cap** : si > 4 096 chars (limite API), tronquer brutalement avec lien "voir tout" et alerter Sentry.
3. **No prescriptive tokens** : disclaimer obligatoire en fin, garde-fou `contains_forbidden_token()` côté backend.
4. **Idempotence** : `(signal_id, chat_id)` ne produit qu'un seul send. Re-render = no-op.
5. **Compliance markdown injection** : escape strict des `_*[]()~`>#+-=|{}.!`.
6. **Geo-block** : si `chat_id` géolocalisé US/QC/UK/OFAC (sur opt-in déclaré) → renderer retourne `None`, pas d'envoi.
7. **edge_claim=False** : aucune surface ne peut afficher "edge prouvé". Mot interdit dans le template.

## Tests obligatoires

- `to_telegram_b2c(sample_insight, "fr")` ≤ 800 chars.
- Tous les boutons inline contiennent UTM `?utm_source=telegram`.
- Si `direction == NEUTRAL` : pas d'envoi (skip).
- Si `event_readout.news_blackout_active` : ligne `event_line` commence par "⚠ Blackout news".
- Si `narrative_short` contient un forbidden token, le test fail-fast.
