# Positioning Brief 2A — Edge Confirmed

> **Sprint COMM-1.1 (Karim, Phase 1).** À activer SI verdict A1 ✅
> (DSR > 1.0, PBO < 0.3, ≥3 Holm-significant, CPCV PF > 1.20).
>
> Référence : `reports/roadmap_2026_2027/PLAN_12_MOIS.md` Partie III.
> Écrit AVANT le verdict A1 pour neutraliser le biais ex-post.
> **Validation Sofia (compliance) requise avant publication.**

---

## 1. Audience cible

### Primary B2C — ICP A "Marc, scalper XAU FR-first" (eval_25)

- 28-45 ans (médiane 34), France/Belgique/Québec/Maghreb FR
- Salarié cadre IT/finance ou freelance, trade 2-4h/jour
- Capital $5k-50k (médiane $12k), broker IC Markets/Pepperstone/FTMO
- 2-5 ans XP, ICT/SMC YouTube, connaît BOS/CHOCH/FVG/OB
- Stack actuelle : TradingView Premium $14-60 + LuxAlgo $39 + Discord trading $30-50 = $80-150/mo
- WTP sustained $20-49/mo, pic $79 si live results 3+ mois consécutifs

### Secondary B2C — Persona C "James, prop firm trader" (eval_25 ICP score 19)

- FTMO / MyForexFunds challenge en cours
- WTP $79-99 si edge réduit son risque de bust
- ⚠️ **Activation conditionnelle** : seulement après 60+ jours forward-test live PF≥1.10
  (eval_28 finding 3 : un prop firm trader qui blow up = brand damage 12-24 mois)

### B2B — Brokers / IBs

- Cible wave 1 (5 prospects, COMM-2A.3) : IC Markets, Pepperstone, Exness, FP Markets, Tickmill
- WTP $1500-3000/mo pour white-label XAU intraday signals API avec SLA 99.5% + audit trail signable
- Use case : enrichir leur offre IB/PAMM, attirer scalpers retail XAU sans dev interne

---

## 2. Claims autorisés (Edge confirmé)

Une fois A1 validé, on peut affirmer en toute honnêteté :

- "Edge backtested CPCV-validated sur 6 ans XAU"
- "Deflated Sharpe Ratio > 1.0, Probability of Backtest Overfitting < 0.3"
- "Features Holm-Bonferroni-significant (pas d'illusion multiple-testing)"
- "Forward-test paper transparent, equity curve publiée live"
- "Audit trail signable bar-par-bar et signal-par-signal" (B2B)
- "Modèle drift-monitored PSI < 0.20" (Phase 2A QUANT-2A.1)

## 3. Claims interdits (compliance MiFID II 2024/2811, eval_29)

- ❌ "100% sûr", "garantie de gain", "achetez maintenant"
- ❌ "Achetez / Vendez" en messages Telegram → utiliser "BULLISH SETUP / BEARISH SETUP"
- ❌ "Recommended exits" → utiliser "levels observed by algorithm"
- ❌ Mots "signal" / "signaux" en marketing landing → utiliser "analyses algorithmiques"
- ❌ Pricing INSTITUTIONAL avec promesse "edge institutionnel garanti"
- ⚠️ Diffusion publique brute SL/TP chiffrés : tolérée à utilisateurs payants logués,
  **interdite** sur Telegram broadcast public > 500 abonnés (requalification CIF AMF probable)

---

## 4. Proof points (différenciation 2A)

| # | Proof point | Origine | Visibilité |
|---|---|---|---|
| 1 | DSR + PBO publiables | QUANT-1.3 verdict | Landing + technical whitepaper |
| 2 | CPCV walk-forward 28 paths | QUANT-1.2 harness | Whitepaper + sample report |
| 3 | Forward-test live equity curve | INFRA-2A.2 | Landing temps réel |
| 4 | Hash-chained audit trail | DATA-2A.7 | B2B whitepaper |
| 5 | Multi-asset transfer EURUSD | QUANT-2A.5 | Blog post technique |
| 6 | Drift PSI monitoring < 0.20 | RISK-2A.2 | B2B SLA section |
| 7 | Eval LLM ≥ 0.78 / 100 prompts | LLM-2A.2 | Landing FAQ |

**Aucun concurrent du panel benchmarké (eval_27 bonus 6) ne propose narration LLM auditée par signal + audit-trail signable.** Window of opportunity = lock content moat (FR-first SEO + dataset SMC) avant Q4 2026.

---

## 5. Pricing recommandé (eval_27 grille v1)

| Tier | Prix | Cible | Inclus |
|---|---|---|---|
| **FREE** | 0€ | Acquisition top-funnel | 1 signal/jour delayed 24h, narrative court template |
| **ANALYST** | **29€/mo** | Marc primary | XAU+EURUSD live, narrative Sonnet, Telegram + webapp |
| **STRATEGIST** | **79€/mo** | James prop firm | + multi-TF + 6 instruments + Telegram alerts personnalisées + API key |
| INSTITUTIONAL (decoy) | 199€/mo affiché | Anchoring | + accès historique 6 ans + email support 24h |
| **B2B-API** | **1 500-3 000€/mo** | Brokers wave 1 | White-label, SLA 99.5%, 1k-10k req/mois, audit trail |

**Decoy stratégie eval_27 finding 2** : afficher "INSTITUTIONAL : Custom enterprise pricing — sur devis" (sans révéler 199 ou 1990) sur landing pour anchoring +25-40% conversion STRATEGIST. ROI projeté = +632 à +2 622€ MRR / 1k visiteurs/mois.

**Trial 14j sans carte sur ANALYST + STRATEGIST** (eval_27 finding 4 : +$1 168 MRR M+6 vs freemium-only). Marc refuse de donner CB sans social proof live.

**Annual -16.7% ("Save 2 months")** affiché par défaut (eval_27 finding 3 : LTV +150% réel, churn -66%).

---

## 6. GTM channel mix

### M3-M6 — Setup & SEO foundation (eval_28 wedge FR-first)

- **5 cornerstone pages SEO FR** (COMM-2A.1) : "analyse XAU/USD intraday", "trading or M15 SMC", "signaux trading or vs forex", "robot trading or open source", "comprendre les COT or"
- **KD<20 cible** (eval_28 wedge 3 780 vol/mo, KD moyen 14 — quasi-vide en SERP FR)
- **Telegram canal public broadcast** + Combot automod gratuit (eval_28 finding 4)
- **Newsletter Substack hebdo** (owned audience nurture)
- **Pas de Discord public** (modération 4-6h/sem inacceptable solo, eval_28 finding 4)
- **Pas de paid ads** (eval_28 finding 3 : 6 triggers verts requis ; PF live ≥ 1.10 sur 60+ jours pour ICP James)

### M6-M9 — Forward-test gate franchie, B2B activation

- **Discord privé paid uniquement** (gating M3+, retention M3 67% gated vs 28% ungated)
- **Outbound brokers wave 1** (COMM-2A.3) : 5 prospects via LinkedIn IB liaison + email
- **1 LOI signée à M9** = KPI CP-2A.2

### M9-M12 — Scaling

- 1 article SEO / 2 sem (12 articles annuel)
- 1 vidéo YouTube market wrap / sem (différenciation FR XAU intraday)
- Pricing experiments (annual -20% bundle, trial 7j vs 14j)

### Cadence non-négociable (eval_28 finding 5)

- **Batch dimanche 14-18h** : article + tournage + queue Buffer + draft newsletter
- Backup mardi 19-23h compressé toléré 1× / 6 sem max
- Si 3 dimanches loupés en 6 sem → audit + couper YouTube en premier (sacrifice acceptable)

### MRR targets

- M6 : 2 000€ (P=55%) — 30-50 ANALYST + 5-10 STRATEGIST
- M9 : 6 500€ (P=40%) — + 1 LOI B2B 1 500€
- M12 : 11 000-14 000€ (P=30%) — 150-200 users payants + 2 contrats B2B

---

## 7. Analyse concurrentielle 5 acteurs

| Concurrent | Prix | Forces | Faiblesses vs 2A | Gap exploitable |
|---|---|---|---|---|
| **TradingView Premium** | 14-60€/mo | UX + charting + community 60M users | Pas de signal prédictif edge-validated, généraliste | Hyper-spé XAU + audit-trail |
| **Trade Ideas** | 168$/mo | Scanner AI stocks US, OddsMaker | Pas FX/commodities, pas FR, $168 = hors WTP Marc | Niche XAU + pricing $29-79 |
| **LuxAlgo** | 39$/mo | Indicateur TradingView, marketing | Pas de backtest CPCV publié, signal "vu après coup" | DSR/PBO transparent, audit-trail |
| **Tickeron** | 60$/mo | AI signals multi-asset | No audit trail, claims marketing flous | Compliance MiFID + audit-trail B2B |
| **FXPremiere** | 99$/mo | Telegram-native, brand FX | Perf non auditée, accusations passées hyperbole | Forward-test live + DSR publiable |

**Moat 2A** (eval_26 competitive 3.5/10 → 7.5/10 18 mois) :
1. **Track-record auditable** (hash-chain DATA-2A.7) — 0 concurrent fait ça B2C+B2B
2. **Hyper-spé XAU + macro features** (FRED+COT+GLD au lieu d'OHLCV-only)
3. **Rubric LLM open-sourced** (LLM-2A.2 eval harness public)

**Window** : 6 mois pour lock content moat avant que Sonnet/Opus permettent à concurrents de répliquer le narrative LLM (eval_27 bonus 6).

---

## 8. Risques de claim et mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| AMF requalifie en CIF (>500 abonnés payants) | Modérée | 🔴 critique | Tier-gate SL/TP chiffrés, never broadcast public, INFRA-2A.4 legal review |
| MiFID II 2024/2811 finfluencer mars 2026 | Élevée | 🟡 modérée | Reformuler "signaux" → "analyses", "BUY" → "BULLISH SETUP" (eval_29) |
| Drift live > backtest (PF < 0.85) | Modérée | 🔴 critique | RISK-2A.1 forward-test gate, kill 2A si fail M4 |
| Concurrent réplique LLM narrative | Élevée 12 mois | 🟡 modérée | Lock content moat M3-M9 (eval_27 window) |
| 1 vidéo influencer "this AI is terrible" | Faible si forward-test | 🔴 brand damage | Pas de paid/influencer avant 6 triggers verts (eval_28) |

---

## 9. Métriques succès (CP-2A)

- **CP-2A.1 (M4)** : forward-test paper PF 30j ≥ 1.10 → Stripe ouverture
- **CP-2A.2 (M9)** : 1 LOI B2B signée
- **M12** : 150+ users payants, 2 contrats B2B, 11k+ MRR (P=30% honnête)

---

**Sofia review checklist (avant publication)** :
- [ ] Aucun "achetez", "vendez", "100% sûr", "garantie" dans le texte
- [ ] Disclaimer eval_29 multi-langue présent en footer landing
- [ ] CGU/Privacy publiées et liées (INFRA-2A.4 closed)
- [ ] Forward-test live publié avec disclaimer "résultats hypothétiques passés ≠ futurs"
- [ ] Geo-block US/QC/UK/OFAC actif (sprint W1+W2+W3 livré)
- [ ] Pricing INSTITUTIONAL ne promet pas "edge institutionnel garanti"

Sofia signature : ___________________  Date : ___________________
