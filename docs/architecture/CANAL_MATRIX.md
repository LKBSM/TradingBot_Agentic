# CANAL_MATRIX — Matrice de tous les canaux de distribution

**Référence** : `docs/architecture/MIA_MARKETS_ARCHITECTURE.md` Partie 3
**Format** : exportable en CSV (une ligne = un canal, attributs en colonnes)
**Version** : 1.0 — 2026-05-27 (post-corrections utilisateur)

## Légende

- **Effort init** : TF (très faible, <8 h), F (faible, 8-40 h), M (moyen, 40-100 h), E (élevé, 100-250 h), TE (très élevé, >250 h)
- **Coût/an** : USD, ordre de grandeur ($0 = gratuit ou marginal sur stack existante)
- **Tier client** : FREE / STARTER / PRO / INSTITUTIONAL / B2B
- **Forme** : push / pull / interactive / autonome
- **Latence acc.** : RT (<1s) / NRT (<30s) / batch (asynchrone) / N/A
- **Compliance** : F (faible) / M (moyen) / E (élevé)
- **Diff. & ROI** : score 1-5 (5 = killer feature ou ROI maximal)
- **Verdict** : V1 / V2 / V3 / V4 / NEVER

## Matrice complète (21 canaux + email transactionnel)

| Canal | Effort init | Coût/an | Tier client | Forme | Latence acc. | Risque compliance | Diff. (/5) | ROI (/5) | Dépendances | Verdict | Justification 1 phrase |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Webapp SaaS (Next.js) | F (140-198h, ~30% livré) | $0-240 | FREE→INST | pull + SSE | RT | M | 5 | 5 | API REST, Vercel, Cloudflare | **V1** | Surface principale, tout pointe ici. |
| PWA installable | TF (déjà livré V2.3) | $0 | FREE→INST | push + UI | RT | M | 3 | 4 | manifest + Service Worker | **V1** | Couvre 70-80% UX mobile native sans dev natif. |
| Bot Telegram | F (40-50h, ~40% livré) | $0 | FREE→PRO | push + chat | NRT | M | 4 | 5 | Bot API, queue | **V1** | Wedge FR-first, conversion vers paid. |
| TradingView Pine showcase | F (26-36h) | $0 | FREE→PRO (acquisition) | pull statique | N/A | F | 5 | 5 | Compte TV, Pine v5, profil soigné | **V1** | Hub où vit la cible — acquisition gratuite. |
| Email transactionnel | TF (8-12h infra) | $0-60 | FREE→INST (infra) | push | NRT | F | 1 | 5 (bloquant) | Resend free, templates | **V1 (infra)** | Bloquant Stripe + DSAR + funnel signup. |
| Email digest quotidien | F (12-20h) | $60-300 | STARTER+ | push + lien | batch | F | 3 | 4 | Audience ≥ 100 MAU | **V2** | Canal de rétention, pas d'acquisition. |
| TradingView webhook receiver | M (30-40h) | $0 + TV Pro user $15/mo | PRO+ | push alert | NRT | M | 4 | 4 | Webhook endpoint, normalizer | **V2** | Pont alerts TV → flux MIA cross-canal. |
| Bot Discord | F (16-24h) | $0 | FREE→PRO | push + chat | NRT | M | 3 | 3 | Bot API | **V2** | EN/community, complète Telegram FR. |
| Mobile cross-platform (React Native + Expo) | E (200-300h) | $99 Apple + $25 Google one-time | STARTER+ | push + UI | RT | M | 4 | 4 | Client API typé, FCM/APNs | **V2 (conditionnel)** | Si ≥ 2 conditions §3.2 atteintes. |
| Push web (Web Push API) | TF (8-12h) | $0 | STARTER+ | push | NRT | F | 2 | 3 | SW PWA déjà en place | **V2** | Complète PWA, alerte event imminent. |
| API REST publique B2B | M (40-60h) | $0 | INSTITUTIONAL | pull | RT (p95<200ms) | F | 4 | 3 (cond.) | OpenAPI public, tier auth | **V3** | Pré-mature avant DG-071 (MRR B2C > $5k 3 mois). |
| Webhooks B2B (push) | F (20-30h, infra signée existe) | $0 | INSTITUTIONAL | push | NRT | F | 4 | 3 | DeliveryAdapter + queue Redis | **V3** | Infra HMAC déjà prête, activer post-B2B. |
| Extension Chrome/Firefox | E (80-120h) | $5 Chrome one-time | PRO+ | pull + overlay | NRT | M | 5 | 3 | API REST, manifest v3, CSP | **V3** | Différenciation forte mais maintenance multi-navigateurs lourde. |
| Widget embeddable (script JS) | M (40-60h) | $0 | INSTITUTIONAL B2B | pull | RT | M | 4 | 3 | API REST, CSP, iframe | **V3** | Couplé B2B-API, brokers/éducateurs. |
| SMS premium | F (12-20h) | $0,03-0,06/SMS | PRO+ | push | NRT | E | 2 | 2 | Twilio/Vonage | **V3** | Coût/msg tue marge FREE. Event-critique payant. |
| Plugin WordPress | M (40-60h) | $0 | INSTITUTIONAL B2B | pull | NRT | M | 3 | 2 | API REST | **V3** | Niche B2B (brokers/éducateurs WP). |
| ChatGPT plugin / Claude tool | M (30-50h) | $0 | STARTER+ | conv. pull | NRT | E | 5 | 3 | OpenAPI public, OAuth | **V3** | Reach énorme mais perte de contrôle compliance wording. |
| MetaTrader 4/5 EA/indicateur | E (100-150h MQL5) | $0 | PRO+ | push alert | NRT | E | 3 | 2 | MQL5 dev, webhook receiver, avocat review | **V3 (conditionnel)** | Reach énorme mais écosystème MT5 = vente signaux régulée. |
| NinjaTrader / cTrader / Sierra | E (par plateforme 80-120h) | $0 | PRO+ | push alert | NRT | E | 2 | 1 | SDK propriétaire ×N | **V4** | Audience trop fragmentée, effort multiplié. |
| Application desktop (Electron) | E (200-300h) | $0 + cert code-signing $200/an | INSTITUTIONAL | UI complète | RT | M | 3 | 1 | API REST, Electron build | **V4** | Web suffit, desktop = re-package faible valeur. |
| Apple Watch / wearables | M (40-60h) | $99 Apple Dev | PRO+ | push alert | NRT | F | 2 | 1 | Mobile natif iOS | **NEVER** | Gadget marketing, pas cas d'usage trader. |
| Alexa / Google Home (briefing vocal) | E (80-120h) | $0 | STARTER+ | pull vocal | NRT | M | 4 | 1 | Skill/Action dev, TTS | **V4** | Brand-play, audience résiduelle < 1 % traders. |
| Audio briefing podcast quotidien (TTS) | F (20-30h) | $0 + ElevenLabs $20/mo | STARTER+ | pull | batch | F | 4 | 2 | TTS service, RSS hosting | **V4** | Différenciant brand, niche, à tester si bande passante. |

## Synthèse par vague

| Vague | Canaux retenus | Effort cumulé V1-V3 | Coût récurrent |
|---|---|---|---|
| **V1 (0-3 mois)** | Webapp + PWA + Telegram + TradingView Pine showcase + Email transactionnel | ~220-300 h | ~$50/mo |
| **V2 (3-9 mois)** | Email digest + TV webhook + Discord + Mobile RN + Push web | ~270-410 h (+conditionnel mobile 200-300h) | +$150-250/mo |
| **V3 (9-18 mois)** | API B2B + Webhooks B2B + Extension nav + Widget + SMS + WordPress + ChatGPT tool + MT4/5 (cond.) | ~360-540 h | +$200-300/mo |
| **V4 (18+ mois)** | Desktop Electron + NinjaTrader/cTrader + Alexa/Google + Audio podcast | ~400-600 h | variable |
| **NEVER** | Apple Watch | — | — |

## Filtres utiles

### Canaux à effort < 50 h en V1 (quick wins)

- Email transactionnel (TF, 8-12h, P0 infra)
- PWA installable (déjà livré V2.3)
- TradingView Pine showcase (F, 26-36h)
- Bot Telegram (F, 40-50h, ~40% livré)

### Canaux à ROI 5/5

- Webapp SaaS V1
- Bot Telegram V1
- TradingView Pine showcase V1
- Email transactionnel V1 (bloquant)

### Canaux à différenciation 5/5

- Webapp SaaS (chatbot pilier)
- TradingView Pine showcase (hub cible)
- Extension navigateur (V3, overlay)
- ChatGPT/Claude tool (V3, reach LLM marketplaces)

### Canaux à risque compliance élevé

- SMS premium (opt-in CNIL strict) — V3
- ChatGPT/Claude tool (perte de contrôle wording) — V3
- MT4/5 (écosystème = signaux régulés) — V3 cond.
- NinjaTrader/cTrader (idem) — V4
- Alexa/Google (TTS du disclaimer obligatoire) — V4

## Export CSV

Pour exporter en CSV brut (séparation Tab) : copier le tableau "Matrice complète" et coller dans Excel/Google Sheets, ou utiliser un script `pandoc` :

```bash
pandoc docs/architecture/CANAL_MATRIX.md -o canals.csv --to=csv
```
