# Mockup — Telegram Signal avec Risk Score 0-100

## Wire-frame ASCII (largeur Telegram mobile ~ 36 col)

### Tier ANALYST+ (Risk Score visible)

```
🟢 Smart Sentinel Signal
────────────────────────────
Direction : LONG
Symbol    : XAUUSD
Score     : 72/100 (STANDARD)

Entry     : 4 217.40
Stop Loss : 4 198.10  (-19.30)
Take Prof.: 4 256.00  (+38.60)
R:R Ratio : 2.0 : 1

Risk Score      : 38/100 🟢 LOW
  ├ Confluence       : 72
  ├ Vol Forecast     : NORMAL (ATR 18.4)
  ├ News Proximity   : safe (>4h)
  ├ Regime Alignment : trend↑ matched
  └ Kill-Switch      : ARMED

Position Sizing (vol-target)
  Risk Budget   : 1.0 % equity
  Suggested Lot : 0.04 (1 K eq.)

🧠 Validation : BOS + retest at OB,
   regime trending, calm vol band.

────────────────────────────
⚠️  Smart Sentinel AI — analyse
algorithmique uniquement, ne
constitue PAS un conseil en
investissement. Risque de perte
en capital.  /disclaimer
```

### Tier FREE (Risk Score caché — gating commercial)

```
🟢 Smart Sentinel Signal
────────────────────────────
Direction : LONG
Symbol    : XAUUSD
Score     : 72/100 (STANDARD)

Entry     : 4 217.40
Stop Loss : 4 198.10
Take Prof.: 4 256.00
R:R Ratio : 2.0 : 1

🔒 Risk Score, Vol Forecast et
   Position Sizing — réservés
   Analyst (49 $/mois) /upgrade
────────────────────────────
⚠️  Not financial advice.
```

### Kill-Switch tripped — admin DM

```
🔴 KILL-SWITCH TRIPPED
────────────────────────────
Reason : DAILY_DRAWDOWN
Detail : daily DD 5.2 % >= 5 %
At     : 2026-04-26 14:32 UTC

Auto-reset in 23h57.
To override now :
  /resume I-ACCEPT-RISK
(operator identity logged)
```

## Composantes Risk Score 0-100

`risk_score = clip(0..100, 100 − weighted_safety)`

| Composante         | Poids | Contribution si "safe"      |
|--------------------|------:|-----------------------------|
| Confluence ≥ 60    |  25 % | −25 (réduit le risque)      |
| Vol Forecast normal|  20 % | −20                         |
| News proximity OK  |  20 % | −20                         |
| Regime alignment   |  20 % | −20                         |
| Kill-switch armed  |  10 % | −10                         |
| Kelly bucket OK    |   5 % | −5                          |

Tiers visuels :
- 0-30 : 🟢 LOW
- 31-60: 🟡 MODERATE
- 61-80: 🟠 ELEVATED
- 81-100: 🔴 EXTREME (signal supprimé côté serveur, alerte admin)

## Conditions d'affichage

- **FREE** : score caché → upsell (`/upgrade`).
- **ANALYST** : score + composante dominante.
- **STRATEGIST/INSTITUTIONAL** : score + breakdown complet + sizing
  vol-target en lots.
- **Toutes tiers** : disclaimer FR/EN footer + lien
  `/disclaimer` vers la page pleine.
