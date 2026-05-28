# Smart Sentinel AI — Pricing Bundles (Mockup)

> Page mockup destinée à l'équipe produit / web. Le HTML final reprendra ce
> contenu en cards 4-colonnes (cf. `mockups/tradingview_dashboard_mockup.html`
> pour le style global).

---

## Hero

# Trade smart. Pay only for what you need.

Smart Sentinel AI propose **4 packs thématiques** en plus des tiers
all-access. Vous ne tradez que les majors FX ? Prenez le **FX Pack** et
économisez 40 % vs STRATEGIST.

| Bundle             | Symbols                       | Plan       | Prix mensuel | Économie vs all-access |
|--------------------|-------------------------------|------------|--------------|------------------------|
| **FX Pack**        | EURUSD · GBPUSD · USDJPY · AUDUSD | Single-asset | **29 USD/mo** | -41 %                  |
| **Metal Pack**     | XAUUSD · XAGUSD · USOIL       | Single-asset | **39 USD/mo** | -20 %                  |
| **Crypto Pack**    | BTCUSD · ETHUSD · SOLUSD      | Single-asset | **49 USD/mo** | 0 % (vol premium)      |
| **Index Pack**     | US500 · NAS100 · DAX40        | Single-asset | **35 USD/mo** | -28 %                  |
| All-access STRATEGIST | All 12 + future assets     | Cross-asset | **49 USD/mo** | reference              |

> Tous les packs incluent : signaux M15/H1/H4 illimités, Telegram,
> dashboard, narratives Claude Sonnet, alertes risque. M5/M1 et
> webhooks API restent réservés au tier INSTITUTIONAL.

---

## FX Pack — 29 USD/mo

> **For traders who live London-NY overlap.**

- 4 majors corrélés : EUR/USD, GBP/USD, USD/JPY, AUD/USD
- Couverture session London + NY overlap (07:00 → 21:00 UTC)
- Calendar : NFP, FOMC, ECB, BoE, BoJ
- Scoring multi-TF M15 → H4
- **Cibles** : day-traders FX, prop-firm challengers (FTMO/MyForexFunds)

**Pourquoi 29 USD ?** Benchmark concurrent : ForexSignals.com 47 USD,
LuxAlgo basic 39.99 USD, FXLeaders 30 USD. On entre par le bas pour
acquisition, marge brute > 80 % (coût LLM ~3 USD/user/mo).

---

## Metal Pack — 39 USD/mo

> **Smart-money meets safe-haven.**

- XAU/USD, XAG/USD, USOIL (WTI)
- Corrélation XAU vs DXY (-0.7) + WTI vs USD (-0.5) : un rapport DXY couvre
  les 3 assets
- Calendar : NFP, CPI, FOMC, EIA Crude Inventory
- **Cibles** : commodity macro traders, gold bugs, retail XAU-only.

**Pourquoi 39 USD ?** XAU notre asset historique le mieux audité (PF
1.086 sur 6 ans baseline, cf. `baseline_2019_2025.md`). USOIL comme
prio P1 dans la roadmap (M7). Marge brute ~75 %.

---

## Crypto Pack — 49 USD/mo

> **24/7 markets need 24/7 alpha.**

- BTC, ETH, SOL — top-3 par capitalisation
- Marché 24/7 = LLM cost double (4 sessions au lieu de 2)
- Volatilité élevée → SL/TP multiplicateurs adaptés (sl_atr_mult=2.0)
- **Cibles** : crypto natives, Coinbase Advanced / Bybit retail.

**Pourquoi 49 USD ?** Pas de remise vs all-access — la couverture 24/7
double les coûts compute + LLM. Benchmark : CryptoQuant 39 USD (basic),
TradingView Premium 59.95 USD. Au pricing du STRATEGIST tier sans la
contrainte d'apprendre les autres assets.

---

## Index Pack — 35 USD/mo

> **For the SPX-NDX-DAX swing trader.**

- US500, NAS100, DAX40
- Sessions RTH-only (14:30 → 21:00 UTC pour US, 07:00 → 15:30 pour DAX)
- Calendar : FOMC, NFP, ECB, ZEW
- **Cibles** : E-mini retail, swing traders, proprietary traders DAX.

**Pourquoi 35 USD ?** US500 est l'asset n°2 derrière XAU en volume signal
estimé. DAX = upsell européen. Benchmark : Trade Ideas 84 USD (mais
covers actions individuelles). Marge brute ~78 %.

---

## All-access tiers (rappel)

| Tier            | Prix      | Description                            | Bundle savings ?      |
|-----------------|-----------|----------------------------------------|-----------------------|
| FREE            | 0 USD     | 1 signal/jour H4-D1, no narrative      | n/a                   |
| ANALYST         | 19 USD/mo | All packs M15+H1, no API               | -33 % vs FX Pack      |
| STRATEGIST      | 49 USD/mo | All packs + M5 + Multi-TF boost        | reference             |
| INSTITUTIONAL   | 199 USD/mo| All + M1 + webhooks + dedicated ops    | enterprise            |

> ANALYST 19 USD < FX Pack 29 USD → **risque cannibalisation** si
> ANALYST inclut tous les majors. Solution : ANALYST = M15+H1 cap 5
> signaux/jour, packs = illimités. Couverture restreinte vs profondeur.

---

## Cannibalisation matrix (audit interne)

| Combinaison       | Risque cannibalisation | Mitigation                                   |
|-------------------|------------------------|----------------------------------------------|
| FX Pack 29 vs ANALYST 19 | HIGH               | Cap ANALYST à 5 signaux/j ; FX Pack illimité |
| Metal Pack 39 vs STRATEGIST 49 | LOW          | 10 USD diff = clear value gap                |
| Crypto Pack 49 vs STRATEGIST 49 | MEDIUM      | Crypto Pack = 24/7 coverage, STRATEGIST=M5   |
| Index Pack 35 vs STRATEGIST 49 | LOW           | 14 USD diff acceptable                       |

---

## CTA

[Start FX Pack — 29 USD/mo →]   [Start Metal Pack — 39 USD/mo →]
[Start Crypto Pack — 49 USD/mo →]   [Start Index Pack — 35 USD/mo →]

> 7-day free trial · Annulable · Stripe / Crypto

---

## Notes design

- 4 cards en grid CSS, dark theme cohérent avec le dashboard mockup.
- Badge « Most popular » sur Metal Pack (XAU = vitrine technique).
- Badge « Best value » sur FX Pack (entrée gamme).
- Toggle mensuel/annuel : -20 % en annuel (psychologie LuxAlgo).
- Logos brokers compatibles (MT5, cTrader, Binance) en footer trust.
