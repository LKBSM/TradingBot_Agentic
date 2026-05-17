# Smart Sentinel AI — Diagnostic & Roadmap d'amélioration

> Document de travail. Date : 2026-05-16. Branche : `institutional-overhaul`.
> Compagnon de `docs/value/information_value_map.md` (inventaire) et des rapports `reports/eval_*`.

---

## §0 — Verdict synthétique

**L'indicateur n'est PAS à son potentiel maximum.** Trois constats objectifs :

1. **~40 % de l'edge technique calculé est invisible client** (conformal, BOCPD, jump ratio, vol CI, breakdown 8-composantes en B2C, multiplicateur composé) — cf. inventaire Partie A.
2. **Le score de conviction n'a aucun pouvoir prédictif empirique** (Pearson −0,023 sur 7 ans XAU M15, eval_02). On vend une note 0-100 qui *mesure la confluence, pas la rentabilité* — c'est intellectuellement honnête mais commercialement faible tant que rien ne « cale » le score à un outcome.
3. **La défendabilité commerciale est 3,5/10 aujourd'hui** (eval_26) — niveau « commodity SMC Pine script à $19/mo » plutôt que « intelligence institutionnelle à $79/mo ».

**Le travail principal n'est pas de la R&D technique supplémentaire** — c'est de **l'exposition** (ce qui est calculé mais caché) et du **trust-building** (track-record auditable, sources citées, calibration empirique).

---

## §1 — Diagnostic actuel : où on en est

### 1.1 Ce qui est solide ✓

- Pipeline 9 étages clean, séparation des responsabilités correcte (`src/intelligence/`).
- 73 informations cartographiées, 10 différenciateurs 🟢 défendables identifiés.
- Compliance UE 2024/2811 implémentée (geo-block, disclaimers FR/EN/DE/ES, `edge_claim=False` honnêteté codifiée).
- Repositionnement « indicateur, pas signal » verbalisé et appliqué côté UI.
- Stack technique académique-grade : HMM 3 états, HAR-RV, conformal prediction (split + ACI Gibbs-Candès), BOCPD (Adams & MacKay 2007), Barndorff-Nielsen bipower variation.
- Tests : 1366+, 0 régressions critiques.
- Documentation interne riche : 29 rapports d'évaluation, eval_00 synthèse maîtresse.

### 1.2 Ce qui ne va pas ✗

#### Performance technique

- **Score non calibré** : `confluence_score` 0-100 ↔ outcome réel ⇒ Pearson −0,023 mesuré, Brier worse than baseline (eval_02). Les composantes existent (`src/intelligence/scoring/isotonic_recalibration.py`) mais pas wirées en prod.
- **`edge_claim=False`** par défaut — on déclare honnêtement n'avoir pas validé d'edge prédictif (verdict A1 du 2026-05-01 : DSR=0, PBO=0,5, CPCV PF=1,008).
- **Data quality** : XAU M15 2019-2025 à 63 % de couverture (eval_08 + data_quality_audit_2026_04_23). 5/6 presets sans CSV propre. Dukascopy + ForexFactory en usage commercial **zone grise** licensingly (eval_29).
- **Latence** : LGBM/Hybrid 1,6-5 s/forecast = ×30-100 hors cible 50 ms (eval_04 verdict). Doit rester HAR-only en prod.
- **Walk-forward non publié** : aucun rolling PF visible client. Backtest 7 ans existe mais n'est pas live-tracked.
- **Real-time delivery** : polling 60 s (sentinel scanner), single-worker, single-process. SLA 30 s inatteignable (eval_09).
- **News pipeline** : ForexFactory CSV scraping, pas de WebSocket événements, sentiment basique.

#### Qualité UX / narrative / trust

- **Lecture monosignal** : le client voit une lecture isolée. Pas de **time-series de conviction** (conviction y a 1h, 4h, 1j → où en est la dynamique ?).
- **Pas de cross-asset readout** : XAU + DXY + US10Y + SPX + WTI = la confluence macro intuition basique d'un trader Or. On ne livre rien.
- **Phase 2B RAG pas démarrée** : `sources_cited` vide en prod. Sans sources, le narratif est de la prose LLM non-auditée — risque hallucination + zero proof-of-method.
- **Track-record paper-demo non publié** : flag `is_paper_demo=True` mais aucun track-record live démontré publiquement. C'est l'élément n°1 de trust qui manque.
- **Pas de mobile-native** : webapp desktop-first. Trader retail = mobile-first.
- **Pas d'alerts customisables** : crossing de seuil conviction, changement de régime, blackout news entrant — rien.
- **Pas de glossaire / onboarding** : « BOS », « CHOCH », « FVG », « OB », « jump ratio » — un nouveau ne comprend rien.
- **Internationalisation** : 4 langues codées, QA réel uniquement FR + EN. DE et ES = templates non-relus par natifs.

#### Commercialité

- **Pricing non-testé** : grille recommandée eval_27 (FREE / $29 / $79 / $1990 decoy) jamais déployée.
- **Pas de trial 14 jours sans carte** : eval_27 estime +$1 168 MRR vs freemium pur.
- **GTM solo bloqué** : eval_28 dit MRR M12 réaliste $5-7 k vs business plan $39 k. Paid ads gated par PF > 1,20 (qu'on n'a pas).
- **Défendabilité 3,5/10 today** (eval_26) — niveau commodity SMC. Cible 7,5/10 dans 18 mois conditionnée à 3 moats (track-record auditable, hyper-spé XAU + macro, rubric LLM open-sourced).
- **Plan B B2B brokers** ($310 k ARR cible, IC Markets + Exness + Pepperstone) **non démarré**. 80 h dev MVP.
- **Compliance ForexFactory/Dukascopy** : licence floue commercial — bloquant à terme.
- **0 trust signals externes** : pas de testimonial, pas d'audit tiers, pas de backtest publié reviewable, pas de presse.

---

## §2 — Recherche : ce que les « best-in-class » offrent et qu'on ne livre pas

### 2.1 Indicateurs de référence à observer

| Concurrent | Tarif | Ce qu'ils font de plus | Applicable à Smart Sentinel ? |
|---|---|---|---|
| **Bloomberg Terminal** | $24 k/an | Cross-asset matrix, COT positioning, news heat, smart-money flow indicators | 🟢 Cross-asset matrix oui ; flow indicators non (data licensing) |
| **Refinitiv Eikon** | $22 k/an | News sentiment time series par asset, économie macro consensus | 🟢 Sentiment time-series ; ⚠️ consensus data payante |
| **Trade Ideas Pro** | $228/mo | Alertes live customisables, AI confidence tracker over time, watchlist multi-asset | 🟢 Time-series conviction + alerts custom |
| **TradingView Premium+** | $60/mo | Backtest visuel, multi-chart, alertes complexes, community ideas | 🟡 Backtest visuel applicable ; community = compliance-risky |
| **TrendSpider** | $79/mo | Multi-timeframe matrix, pattern recognition, backtest automatique | 🟢 Multi-TF matrix très applicable |
| **Tickeron** | $100/mo | AI confidence avec time-to-target stats, pattern library | 🟡 Stats time-to-target déjà partiellement ; library applicable |
| **LuxAlgo Premium** | $59/mo | Détection SMC overlay visuel direct sur chart, alerts customisables | 🟡 Overlay visuel = bonne UX, alerts oui |
| **AltIndex** | $30-50/mo | Alt data (Reddit, Wikipedia, web traffic) sentiment | 🔴 Pas applicable au XAU/FX |

### 2.2 Top 8 features que les concurrents livrent et qu'on ne livre pas (encore)

1. **Time-series de la conviction** (Trade Ideas, Tickeron) — sparkline 24 h / 7 j sur le score.
2. **Cross-asset readout matrix** (Bloomberg, TrendSpider) — XAU + DXY + US10Y + SPX + WTI dans une vue unique.
3. **Alertes custom** (Trade Ideas, TradingView, LuxAlgo) — seuil conviction crossing, changement de régime, blackout news entrant.
4. **Pattern library + analogues historiques** (Tickeron) — « cette configuration ressemble à 47 setups passés, hit rate 38 % ».
5. **Backtest interactif** (TradingView, TrendSpider) — le client peut lancer un backtest sur ses critères.
6. **Visualisation overlay direct sur chart** (LuxAlgo) — niveaux BOS / FVG / OB dessinés sur TradingView.
7. **Mobile-native expérience** (Trade Ideas mobile, LuxAlgo mobile) — push notifications, lecture rapide.
8. **News sentiment time series** (Refinitiv, Bloomberg) — courbe de sentiment XAU sur 7 j, événements annotés.

### 2.3 Ce que personne ne fait bien en retail et qu'on PEUT faire

C'est notre opportunité moat-building :

- ⭐ **Conformal coverage exposée** — aucun concurrent retail. Garantie mathématique distribution-free.
- ⭐ **Régime BOCPD + jump ratio académiques** — academic-grade, retail n'a pas l'outillage.
- ⭐ **Sources RAG citées dans le narratif** — Phase 2B. Aucun ChatGPT-wrapper ne fait propre.
- ⭐ **Multi-langue compliance UE 2024/2811** — quasi-personne ne respecte le wording proprement.
- ⭐ **FR-first XAU SMC** — wedge ICP eval_25, concurrence FR vide.
- ⭐ **Track-record auditable hash-chained** (`src/audit/hash_chain_ledger.py`) — l'infrastructure existe, manque le rendu public.

---

## §3 — Axe 1 : Performance technique

### 3.1 Tableau initiatives × impact × effort

| # | Initiative | Impact | Effort | Statut code |
|---|---|---|---|---|
| P-1 | **Calibrer score via isotonic regression** (`scoring/isotonic_recalibration.py` exists) → `conviction_0_100` devient une vraie proba calibrée | 🟢🟢🟢 | 🔵 4 h | Code prêt non wiré |
| P-2 | **Exposer `ConformalInterval` dans `InsightSignalV2`** (sous-modèle `UncertaintyContext`) | 🟢🟢🟢 | 🔵 8 h | Calculé, non exposé |
| P-3 | **Wirer LGBM scorer** (`scoring/lgbm_scorer.py`) comme alternative au logistic L1 | 🟢🟢 | 🟡 12 h | Code prêt, non bench |
| P-4 | **Exposer `regime_readout` complet** (HMM posterior + BOCPD cp_prob + jump_ratio + regime_gate decision) en v2 | 🟢🟢🟢 | 🔵 6 h | Calculé, non exposé |
| P-5 | **Exposer `vol_uncertainty` complet** (forecast + naïf + CI conformel TCP) en v2 | 🟢🟢 | 🔵 4 h | Calculé, non exposé |
| P-6 | **Exposer breakdown 8-composantes en v2** (B2C aussi) avec arbitrage IP sur les poids | 🟢🟢🟢 | 🟡 8 h | Calculé, B2B-only |
| P-7 | **Data quality refresh XAU** : feed propre Dukascopy v2 ou Polygon.io / Tiingo (license clean) | 🟢🟢 | 🟡 16 h + $40-200/mo licence | À refaire |
| P-8 | **Multi-asset CSV ingest live** : EUR/USD H1, USOIL, US500 propres + tests data quality | 🟢🟢 | 🟡 24 h | 5/6 presets sans CSV |
| P-9 | **Walk-forward rolling published** : exposer PF rolling 6m / 12m / 24m sur le track-record page | 🟢🟢🟢 | 🟡 16 h | Existe en backtest, pas public |
| P-10 | **Streaming bars (WebSocket) au lieu polling 60s** → latence 30 s SLA atteignable | 🟢 | 🔴 40 h | À architecturer |
| P-11 | **News pipeline propre** : remplacer ForexFactory scraping par TradingEconomics API ($XX/mo) ou EconDB | 🟢🟢 | 🟡 12 h + $50-200/mo | Compliance latente |

**Quick wins P0 (sub-8h chacun, total ~30 h)** : P-1, P-2, P-4, P-5. Ces 4 initiatives **débloquent immédiatement la moitié de l'edge buried** sans nouvelle R&D.

---

## §4 — Axe 2 : Qualité (UX, narrative, trust)

### 4.1 Tableau initiatives × impact × effort

| # | Initiative | Impact | Effort | Statut |
|---|---|---|---|---|
| Q-1 | **Sparkline time-series de conviction** sur 24 h / 7 j dans la webapp + heatmap conviction par heure | 🟢🟢🟢 | 🟡 16 h | Données dispo via SignalStore |
| Q-2 | **Cross-asset readout matrix** : page récap XAU + EURUSD + DXY + US10Y + SPX + WTI — lectures synchronisées + corrélations | 🟢🟢🟢 | 🔴 32 h | Module `cross_asset_correlation.py` partiel |
| Q-3 | **Track-record paper-demo public** : page `/track-record` avec liste signaux + outcomes + sparkline PF rolling, downloadable CSV | 🟢🟢🟢 | 🟡 40 h | SignalStore prêt, page UI à faire |
| Q-4 | **Alertes custom** : conviction crossing X, changement régime, blackout news entrant — webhook + Telegram + email | 🟢🟢 | 🔴 32 h | Notification queue existe |
| Q-5 | **Phase 2B RAG launch** : RAG sourcé (papers, LBMA, CFTC COT) + `sources_cited` rempli en prod | 🟢🟢🟢 | 🔴 80 h | Plan agent Aisha (roadmap_2026_2027) |
| Q-6 | **Mobile-responsive webapp** + push notifications (PWA) | 🟢🟢 | 🟡 24 h | Webapp desktop only |
| Q-7 | **Glossaire interactif + tooltips contextuelles** sur tous les termes techniques (BOS, CHOCH, FVG, OB, jump ratio, conformal, etc.) | 🟢🟢 | 🟡 12 h | Absent |
| Q-8 | **QA multi-langue DE/ES** par natifs (audit + correction des templates LLM) | 🟢 | 🟡 8 h freelance ($300-500) | DE/ES non revus |
| Q-9 | **Backtesting interactif** sur la page track-record : le client choisit dates, instrument, TF, voit PF | 🟢🟢 | 🔴 60 h | Backtest engine existe, UI non |
| Q-10 | **Pattern library + analogues historiques** : « cette lecture ressemble à 47 setups passés, distribution outcomes » | 🟢🟢 | 🔴 50 h | Nouvelle feature |
| Q-11 | **News sentiment time series** : courbe sentiment XAU 7 j avec événements annotés | 🟢 | 🟡 16 h | Sentiment dispo |

**Quick wins Q0** : Q-1 (sparkline conviction, 16 h), Q-7 (glossaire, 12 h), Q-8 (QA DE/ES, 8 h freelance).
**Coup de tonnerre commercial** : Q-3 (track-record public 40 h) — c'est *littéralement* le moat #1 selon eval_26.

---

## §5 — Axe 3 : Commercialité (pricing, positioning, conversion, défendabilité)

### 5.1 Tableau initiatives × impact × effort

| # | Initiative | Impact | Effort | Statut |
|---|---|---|---|---|
| C-1 | **Déployer grille tarifaire eval_27** : FREE / $29 ANALYST / $79 STRATEGIST / $1990 decoy INSTITUTIONAL | 🟢🟢🟢 | 🔵 décision + 8 h config Stripe | Recommandé non déployé |
| C-2 | **Trial 14 j sans carte** sur tier STRATEGIST | 🟢🟢 | 🔵 4 h Stripe config | Non implémenté |
| C-3 | **Landing page wedge XAU SMC FR-first** : SEO, contenu fondateur, démo interactive | 🟢🟢🟢 | 🟡 40 h dev + content | Absent |
| C-4 | **Public track-record landing** (dépend Q-3) : preuve sociale chiffrée | 🟢🟢🟢 | inclus dans Q-3 | Bloqué par Q-3 |
| C-5 | **B2B broker MVP** : API minimale (auth + endpoint `/api/v2/insights/latest`) + page partner-onboarding | 🟢🟢🟢 | 🔴 80 h MVP | Plan B eval_26 |
| C-6 | **Compliance data licences** : remplacer ForexFactory par TradingEconomics ($50-200/mo) ou EconDB ; idem Dukascopy → Tiingo / Polygon ($30-200/mo) | 🟢🟢 | 🟡 12 h + abonnements | Risque latent |
| C-7 | **Content marketing fondateur FR** : 1 article/sem expliquant un setup XAU, sources citées | 🟢🟢🟢 | hors dev — 6 h/sem | Pas démarré |
| C-8 | **Affiliate / community-driven distribution** | 🟢 | 🟡 — gated par PF > 1,20 | Bloqué eval_28 |
| C-9 | **Audit indépendant du track-record** (firme quant tierce) — trust seal | 🟢🟢 | 🔴 $5-15 k + 3-6 mois | Premier seal de l'industrie retail XAU |
| C-10 | **Open-source la rubric de scoring** (poids + composantes documentés, sous license MIT) — moat paradoxal selon eval_26 | 🟢🟢 | 🟡 16 h + decision | Différenciateur selon eval_26 |
| C-11 | **Webhook B2B + SDK Python/JS** pour intégration broker | 🟢🟢 | 🟡 24 h | Dépend C-5 |

**Quick wins C0** : C-1 (pricing déploiement, 8 h), C-2 (trial 14 j, 4 h), C-6 (data licences propres, 12 h + abos).
**Moat-builder** : C-5 (B2B brokers), C-9 (audit tiers), C-10 (open-source rubric).

---

## §6 — Matrice priorité globale (impact × effort)

Critères : impact (proxy revenue / défendabilité / fondation pour le reste) × effort (h dev) × dépendances.

### 6.1 P0 — À faire dans les 14 prochains jours (≤ 40 h dev cumulés)

| ID | Initiative | Effort | Pourquoi P0 |
|---|---|---|---|
| **P-1** | Calibrer score via isotonic | 4 h | Code prêt, débloque la crédibilité du score 0-100 |
| **P-2** | Exposer `ConformalInterval` v2 | 8 h | Levier #1 différenciation, déjà calculé |
| **P-4** | Exposer `regime_readout` complet v2 | 6 h | BOCPD + jump ratio = academic-grade visible |
| **P-5** | Exposer `vol_uncertainty` complet v2 | 4 h | Confidence interval forecast vol |
| **C-1** | Déployer grille tarifaire eval_27 | 8 h | Test commercial sans dev majeur |
| **C-2** | Trial 14 j sans carte | 4 h | +$1 168 MRR estimé |
| **Q-8** | QA DE/ES par natifs (freelance) | 8 h ($400) | Compliance multi-langue propre |
| **TOTAL P0** | **~42 h dev + ~$400 freelance** | Débloquage immédiat de 50 % de l'edge buried |

### 6.2 P1 — À faire dans les 30 prochains jours (≤ 120 h dev cumulés)

| ID | Initiative | Effort | Pourquoi P1 |
|---|---|---|---|
| **P-6** | Exposer breakdown 8-composantes en v2 B2C | 8 h | Concurrence n'a pas |
| **Q-1** | Sparkline time-series conviction | 16 h | UX différenciante low-cost |
| **Q-3** | Track-record paper-demo public | 40 h | Moat #1 selon eval_26 |
| **Q-7** | Glossaire interactif | 12 h | Onboarding crucial |
| **P-9** | Walk-forward rolling published | 16 h | Trust signal |
| **C-6** | Data licences propres (TradingEconomics + Tiingo) | 12 h + $80-400/mo | Compliance latente résolue |
| **C-3** | Landing wedge XAU SMC FR-first | 40 h | SEO + conversion |
| **TOTAL P1** | **~144 h dev + ~$300/mo abos** | Trust + différenciation visible |

### 6.3 P2 — À faire dans les 90 prochains jours (≤ 320 h dev cumulés)

| ID | Initiative | Effort | Pourquoi P2 |
|---|---|---|---|
| **Q-5** | Phase 2B RAG launch (sources citées) | 80 h | Plan Aisha, moat #2 |
| **C-5** | B2B broker MVP | 80 h | Plan B, $310 k ARR cible eval_26 |
| **Q-2** | Cross-asset readout matrix | 32 h | Feature concurrentielle Bloomberg-like |
| **Q-6** | Mobile-responsive + PWA | 24 h | Trader retail = mobile-first |
| **Q-4** | Alertes custom | 32 h | Feature attendue (Trade Ideas, LuxAlgo) |
| **P-7** | Data quality refresh XAU feed propre | 16 h | Élimine le 63 % coverage problem |
| **P-8** | Multi-asset CSV ingest live (USOIL, US500, GBPUSD) | 24 h | Justifie 6 presets |
| **TOTAL P2** | **~288 h dev + abonnements** | Moat-building + B2B |

### 6.4 Reportés / conditionnels

- **P-10** Streaming WebSocket (gated par traffic réel — pas avant 100+ utilisateurs actifs)
- **Q-9** Backtesting interactif UI (gated par PF rolling validé)
- **Q-10** Pattern library / analogues historiques (gated par data quality refresh P-7)
- **C-8** Affiliate / community (gated par PF > 1,20)
- **C-9** Audit indépendant track-record (à faire après 6+ mois de track-record paper public)

---

## §7 — Plan d'exécution 90 jours

### Sprint 1 (Jours 1-14) — Quick wins exposition
**Objectif** : débloquer 50 % de l'edge buried sans nouvelle R&D + tester pricing.

Livraisons :
- `InsightSignalV2` enrichi avec `uncertainty`, `regime_readout`, `volatility_readout`, `breakdown_components` (P-2, P-4, P-5, P-6)
- Score calibré isotonic en prod (P-1)
- Grille tarifaire eval_27 déployée + trial 14 j (C-1, C-2)
- QA DE/ES freelance (Q-8)

Effort : ~50 h dev + $400 freelance. **Aucun risque technique.**

### Sprint 2 (Jours 15-44) — Trust-building + UX différenciante
**Objectif** : publier le premier track-record auditable + premier visuel temporel.

Livraisons :
- Page `/track-record` paper-demo publique avec sparkline PF rolling + CSV downloadable (Q-3, P-9)
- Sparkline time-series conviction sur la webapp (Q-1)
- Glossaire interactif (Q-7)
- Landing wedge XAU SMC FR-first (C-3)
- Data licences propres (TradingEconomics + Tiingo, C-6)

Effort : ~144 h dev + setup abos $80-400/mo. **Le sprint qui crée le moat principal.**

### Sprint 3 (Jours 45-90) — Moat-building Phase 2B + B2B
**Objectif** : exécuter le pari narrative-first (sources RAG) + ouvrir le canal B2B brokers.

Livraisons :
- Phase 2B RAG complet : `sources_cited` rempli en prod (Q-5)
- B2B broker MVP : API minimale + onboarding partenaire (C-5)
- Cross-asset readout matrix (Q-2)
- Mobile-responsive + PWA (Q-6)
- Alertes custom (Q-4)
- Data quality refresh (P-7) + multi-asset propre (P-8)

Effort : ~288 h dev. **Le sprint stratégique qui ouvre les deux moats commerciaux.**

---

## §8 — Risques et anti-patterns à éviter

### 8.1 Ce qu'il NE faut PAS faire

❌ **Ajouter de la R&D nouvelle** (autres modèles vol, nouveau régime detector, TSFM, neural nets, etc.) — interdit par les évaluations passées (eval_04 sur les TSFMs, verdict A1 sur la R&D pure). Le travail principal est d'**exposer ce qui existe**.

❌ **Survendre** dans le marketing : tant que `edge_claim=False`, ne JAMAIS dire « edge prouvé », « 90 % winrate », « battre le marché ». La compliance UE 2024/2811 + l'honnêteté codifiée sont notre moat — les casser c'est se transformer en énième signal Telegram.

❌ **Pivoter le score 0-100 en proba sans calibration empirique sérieuse** — risque de promettre « 72 % de chances ». L'isotonic regression sur outcome backtest doit être la base, et toujours présenté comme « probabilité empirique sur setups similaires historiques », pas « probabilité de gain ».

❌ **Lancer paid ads avant PF > 1,20** — eval_28 verdict.

❌ **B2C uniquement** — l'unit economics eval_24 dépend de 3 hypothèses non vérifiées (cache hit 60 %, dedup 95 %, NARRATIVE_MODE=llm). Le canal B2B brokers est le hedge revenue.

### 8.2 Points de vigilance

⚠️ **Exposer le breakdown 8-composantes (P-6)** révèle la méthode de scoring → arbitrage IP. Compromis : exposer `name`, `score_pct`, `reasoning` mais masquer `weight`.

⚠️ **Phase 2B RAG (Q-5)** dépend d'une bonne curation des sources (LBMA, CFTC COT, BIS, papers). Mauvaises sources = narratif crédible mais creux.

⚠️ **B2B broker MVP (C-5)** demande SLA, support, contractuel. Pas plug-and-play à 80 h — c'est un MVP, pas la prod V1.

⚠️ **Audit indépendant (C-9)** = $5-15 k. À budgéter en Sprint 4+ après 6 mois de track-record paper.

---

## §9 — Synthèse stratégique en 5 lignes

1. L'indicateur **n'est pas à son max** — ~40 % de l'edge technique est calculé mais caché client.
2. Le **score 0-100 doit être calibré empiriquement** avant tout autre travail (P-1, 4 h) — sinon on vend une métrique sans pouvoir prédictif.
3. **Le moat #1 commercial** est le track-record paper-demo public auditable (Q-3, 40 h) — c'est la condition unique pour passer de défendabilité 3,5/10 à 7,5/10.
4. **Le quick win technique #1** est d'exposer `ConformalInterval` + `regime_readout` + `vol_uncertainty` + breakdown 8-composantes en `InsightSignalV2` (P-2/4/5/6, total ~26 h) — débloque la moitié des différenciateurs 🟢 forts identifiés.
5. **Le plan B revenue** est le B2B brokers (C-5, 80 h MVP, $310 k ARR cible eval_26) — à ouvrir parallèlement au B2C, pas après.

**Effort total des 3 sprints (90 jours)** : ~482 h dev + ~$80-400/mo abonnements data + ~$400 freelance one-shot. Soit ~12 semaines × 40 h pour un dev solo, ou 6 semaines × 80 h en mode intensif.

**Verdict** : on a un produit techniquement riche au niveau institutional, commercialisé au niveau commodity. Le gap se ferme par 90 jours d'**exposition + trust + B2B** — pas par de la R&D.
