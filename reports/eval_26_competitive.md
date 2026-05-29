# Eval 26 — Competitive Analysis & Différenciation (Smart Sentinel AI)

> **Périmètre audité** : marché 2026 des signaux trading IA pour traders retail/semi-pro (XAU, FX, indices, crypto). Identification de 10 concurrents directs, matrice features × prix × user-base × funding, analyse défendabilité, plan B copycat, partnerships brokers.
>
> **Date** : 2026-04-26 · **Branch** : `main` · **Snapshot** : 632e9dd + Sprint 2 uncommitted.
>
> **Sources** : WebSearch (avril 2026, voir Annexe 17 pour URLs et dates d'accès), MEMORY.md, `reports/eval_05_llm.md`, `reports/eval_25_pmf_icp.md`, `src/api/tier_manager.py:36-67`.

---

## 0. TL;DR

| Dimension | Note /10 | Justification chiffrée |
|-----------|----------|------------------------|
| Différenciation produit *brute* (vs top-3) | 5 | TradingView 100M+ users, LuxAlgo 1M+ followers, Trade Ideas 22 ans d'existence — Sentinel n'a aucune feature unique, juste une combinaison potentiellement nouvelle. |
| Défendabilité technique (moat sustaintable) | **3** | LLM narrative + SMC + multi-TF est *trivialement copiable* en 6-12 semaines par un acteur funded ; aucun moat-data, aucun network effect. |
| Différenciation revendicable *aujourd'hui* (audit-tested) | 2 | Backtest XAU PF 0.94 (cf. `xau_replay_findings`), narrative LLM en mode `template` par défaut (cf. `eval_05`), score Confluence Pearson −0.023 (cf. `confluence_calibration`). Aucune preuve de surperformance vs concurrents. |
| Risque copycat 6 mois | 2 (faible défense) | Bloomberg ASKB déjà lancé (févr. 2026), TradingView AI Chart Copilot déjà en beta (mars 2026), LuxAlgo positionné "AI Algorithmic Trading Platform" — la fenêtre se ferme. |
| Données & moat-data potentiel | 4 | SignalStore + audit trail + tracker PnL = infrastructure d'un futur moat-data, mais pas encore exploité. |
| Partnerships/distribution leverage | 3 | Aucun deal broker, aucun deal influenceur, aucune communauté. Exness CPA $1850/client est l'une des rares vraies routes. |
| Multi-langue (TAM emerging markets) | 1 | EN-only ; aucun concurrent EN-only ne dépasse 30% LATAM/MENA. |
| Exécution rapidité-vs-funded | 5 | Solo-dev 1 mois SMC + LLM = avantage vitesse mais aucun budget marketing. Composer ($16.7M Series A) et TradingView ($339M raised) sur la même proposition. |
| **GLOBAL — défendabilité** | **3.5 / 10** | « Stratégiquement attaquable. Le moat n'existe pas encore — il faut le construire en 90 jours sur la donnée propriétaire (track-record auditable + dataset SMC labellisé) ou pivoter vers un wedge ultra-vertical (XAU only, FR market). » |

---

## 1. Cartographie du marché 2026

### 1.1 Segmentation marché signaux trading IA

```
                            MARCHÉ SIGNAUX TRADING IA 2026
                                       │
        ┌──────────────┬───────────────┼─────────────────┬──────────────────┐
        ▼              ▼               ▼                 ▼                  ▼
   PLATFORMS     INDICATORS/       AUTOMATED           SIGNALS           PRO/INSTI
   (analyst)     OVERLAYS         BOTS               SUBSCRIPTIONS     (Bloomberg)
        │              │               │                 │                  │
   TradingView    LuxAlgo         3Commas           AltSignals         Bloomberg ASKB
   TrendSpider    FluxCharts      StockHero         1000pip Builder    Refinitiv
   Trade Ideas    QuantZee        Composer          MQL5 Signals       FactSet
   Tickeron                       SignalStack       BlackBoxStocks
                                                    ForexSignals.io
                                                    [Telegram channels]
```

**Smart Sentinel AI** se positionne à l'**intersection** "Indicators + Automated Bots + Signals Subscriptions" avec un angle *narration LLM explicable*. C'est un positionnement valide mais **occupé** : LuxAlgo, FluxCharts, Trade Ideas (Holly AI), Tickeron tirent tous une corde "AI signals + explanation" en avril 2026.

### 1.2 Taille de marché et tendances

- TradingView annonce **15 M MAU** (2025, Crunchbase) à **100 M+ utilisateurs cumulés** revendiqués site (2026).
- TradingView levée totale : **$339M**, valuation $3B (Series C 2021, Tiger Global lead).
- Le wedge "AI trading signals" connaît une saturation rapide : 10+ launches en 2025-2026 (Holly AI Money Machine, ASKB, AI Chart Copilot, Sidekick…).
- ICT/SMC en particulier : `joshyattridge/smart-money-concepts` (Python OSS) packagé MIT, LuxAlgo SMC indicator est gratuit sur TradingView (>1M utilisateurs revendiqués). **L'algorithme SMC n'est plus un moat technique.**

---

## 2. 10 Concurrents directs — Fiches détaillées

### 2.1 TradingView (Plateforme + AI Chart Copilot)

| Champ | Valeur |
|-------|--------|
| URL | https://www.tradingview.com/pricing/ |
| Pricing 2026 | Plus, Premium (+$120/an depuis 2026), Ultimate. Range ~$15/mo (Plus) → ~$60/mo (Ultimate annuel). |
| User base | **15M MAU** (2025), revendique 100M+ utilisateurs cumulés (2026, [TradingView Pricing](https://www.tradingview.com/pricing/)). |
| Funding | **$339M total raised**, valuation $3B (Series C oct. 2021, Tiger Global lead) ([Crunchbase](https://www.crunchbase.com/organization/tradingview/company_financials)). |
| Feature gaps vs Sentinel | (1) Pas de narration LLM par défaut (AI Chart Copilot Chrome ext. en beta mars 2026, free, pas natif Telegram). (2) Pas de SMC formalisé natif (offert via overlays tiers). (3) Pas de routing multi-modèle Claude/GPT. (4) Pas de 4-tier subscription orienté signals. |
| Forces | Distribution massive, Pine Script ecosystem, marque ultra-établie, AI Chart Copilot **gratuit** en beta = menace prix directe sur narratives. |
| Faiblesses | Lourdeur produit, alerts payantes par tier (800 alerts max Premium), pas de focus "explanation AI". |
| Source date | 2026-04-26 (WebSearch). |

### 2.2 LuxAlgo (Indicators + AI Algorithmic Platform)

| Champ | Valeur |
|-------|--------|
| URL | https://www.luxalgo.com/pricing/ |
| Pricing 2026 | Premium **$27.99/mo** (annual) ou $39.99 monthly · Ultimate $32.99/mo (annual). Premium annual year 1 = $335.92, renouvel. $479.88. |
| User base | **100 000+ traders** revendiqués · profil TradingView **1M+ followers** ("Pine Wizard" 2024) · Trustpilot 4.7/5 sur 1 293 reviews ([Trustpilot LuxAlgo](https://www.trustpilot.com/review/luxalgo.com)). |
| Funding | **Bootstrap (aucune levée publique)** — pas de profil Crunchbase/PitchBook visible. Estimation revenue $30-50M/an basée sur user-base × ARPU $300 ⇒ ne nécessite pas levée. |
| Feature gaps vs Sentinel | (1) Indicateurs *visuels* uniquement, pas de narration textuelle structurée. (2) Repaint signalé en reviews (pivothigh/pivotlow look-ahead). (3) Pas d'API/Telegram natif (utilisateur câble via SignalStack). (4) Pas de track-record verifié MyFXBook. |
| Forces | Marque dominante TradingView, communauté Discord large, déploiement rapide. |
| Faiblesses | Plaintes répétées : repainting, auto-renewal abusif ($479.88 surprise), inconsistance des signaux. |

### 2.3 TrendSpider (Charts + AI Sidekick)

| Champ | Valeur |
|-------|--------|
| URL | https://trendspider.com/pricing/ |
| Pricing 2026 | Standard $54/mo · Premium $91/mo · Enhanced $122/mo · Business $214/mo · Advanced $399/mo. Trial 14j à $19. Signals automated $27 (50 sig) → $340 (1000 sig). |
| User base | **20 000+ customers**, 4.6/5 stars ([TrendSpider Pricing](https://trendspider.com/pricing/)). |
| Funding | Pas de round public récent (bootstrap/PE selon CrunchBase). |
| Feature gaps vs Sentinel | (1) US equities-centric (pas XAU spot). (2) Sidekick = chat AI générique, pas de narration *attachée à un signal validé*. (3) 25 messages AI/mois inclus, paywall au-delà — friction. (4) Pas de SMC focus, plutôt patterns techniques classiques. |
| Forces | Backtester puissant, automated trading via SignalStack, Sidekick LLM dans plateforme. |
| Faiblesses | Cher ($91-122 sweet-spot), courbe d'apprentissage haute. |

### 2.4 Trade Ideas (Holly AI scanner)

| Champ | Valeur |
|-------|--------|
| URL | https://www.trade-ideas.com/pricing/ |
| Pricing 2026 | Standard $89/mo ($1 068/an) · Premium **$178/mo ($2 136/an)** avec Holly AI. |
| User base | Pas de chiffre public (estimation 5-15k subs basé sur ARR ~$10-30M). |
| Funding | Société de 2003, profitable, **non-funded externe** (estimation propriétaire). |
| Feature gaps vs Sentinel | (1) US equities only. (2) Holly = ML "black-box", aucune narration explicable. (3) Pas de Telegram. (4) Money Machine v2 = exécution, pas analyse. |
| Forces | 22 ans d'historique, Holly AI revendique 60-65% win-rate verified, communauté pro-trader. |
| Faiblesses | UX old-school, prix élevé (3-12× Sentinel target tier). |

### 2.5 StockHero (No-code stock bots)

| Champ | Valeur |
|-------|--------|
| URL | https://www.stockhero.ai/pricing/ |
| Pricing 2026 | Lite $29.99/mo (1 bot) · Premium $49.99 (15 bots) · Pro $99.99 (50 bots, 1min TF). V4 launch janv. 2026 → hausses prix annoncées. |
| User base | « Thousands of traders » revendiqués (pas de chiffre précis). |
| Funding | Aucune info publique trouvée — boutique ops. |
| Feature gaps vs Sentinel | (1) Stocks only (pas XAU spot, pas FX). (2) Bot-first (auto-execute), pas signals lisibles. (3) Pas de LLM narrative. |
| Forces | Trial gratuit 14j, prix bas, simple à setup. |
| Faiblesses | Pas de différenciation IA (label marketing), V4 peut introduire churn. |

### 2.6 3Commas (Crypto bots)

| Champ | Valeur |
|-------|--------|
| URL | https://3commas.io/pricing |
| Pricing 2026 | Free · Starter $29/mo · Pro · Expert (annual discount). |
| User base | Estimation 100k+ accounts, processed "billions in trading volume". 15+ exchanges. ([3Commas](https://3commas.io/)). |
| Funding | Pas de profil Crunchbase clair (Estonia HQ, sans round notable annoncé). |
| Feature gaps vs Sentinel | (1) Crypto only. (2) DCA/grid bots, pas signals-first. (3) Pas LLM narrative. (4) Image dégradée par hack 2022 (DB leak). |
| Forces | Mature, exchanges intégrés, AI Trading Bot beta 2024+. |
| Faiblesses | Trust issues post-incident, satisfaction split (4.1-4.3/5 CryptoCompare). |

### 2.7 Composer (No-code automated trading)

| Champ | Valeur |
|-------|--------|
| URL | https://www.composer.trade/ |
| Pricing 2026 | Crypto = free + 0.2% commission · Stocks/IRA = $40/mo (trial 14j) · Business custom. |
| User base | 29 employés (janv. 2026), pas de chiffre subscribers public. ([PitchBook Composer](https://pitchbook.com/profiles/company/455170-33)). |
| Funding | **$16.7M raised** (Series A sept. 2022), investisseurs : First Round Capital, Golden Ventures, Left Lane Capital. |
| Feature gaps vs Sentinel | (1) US ETFs/stocks centric. (2) Strategy *building* > signals delivery. (3) Pas SMC. (4) Pas Telegram. |
| Forces | UX excellente, AI strategy generation natif (2024+). |
| Faiblesses | TAM US (broker integration), pas de narrative explicable par signal. |

### 2.8 SignalStack (Order automation gateway)

| Champ | Valeur |
|-------|--------|
| URL | https://signalstack.com/pricing/ |
| Pricing 2026 | Pay-as-you-go : 25 signaux gratuits puis $0.59-$1.49/signal. Subs : Basic $27 (50 sig) · Premium $97 (250 sig) · Pro $250 (1000 sig). |
| User base | Pas de chiffre public ; 33+ brokers intégrés. |
| Funding | Pas trouvé. |
| Feature gaps vs Sentinel | (1) **N'émet PAS de signaux** — ne fait que router des signaux d'origine externe vers brokers. (2) Pas d'IA, pas de SMC, pas de narrative. |
| Forces | Latence 0.45s, ecosystem large, partenariats LuxAlgo/TrendSpider. |
| Faiblesses | Commodity infrastructure ; risque dépendance broker. **Pas un concurrent direct mais un canal de distribution potentiel pour Sentinel.** |

### 2.9 Tickeron (AI patterns + Pattern Bots)

| Champ | Valeur |
|-------|--------|
| URL | https://tickeron.com/ |
| Pricing 2026 | Investor $60/mo · Swing $80 · Day Trader $90 · Expert $250 ($1 250/an). PSE/RTP/TPE add-ons $20-$30/mo chacun. |
| User base | « Substantial » mais pas de chiffre public 2026. |
| Funding | Pas de levée publique majeure trouvée. |
| Feature gaps vs Sentinel | (1) US equities centric. (2) Patterns classiques, pas SMC/ICT. (3) Pas de narration explicable LLM (juste descriptions canned). (4) Claims "87% accuracy" non auditable. |
| Forces | 300M patterns historiques analysés (revendication), AI bot ecosystem. |
| Faiblesses | UX surchargée, claims accuracy contestables (typique secteur). |

### 2.10 AltSignals (Telegram signals provider)

| Champ | Valeur |
|-------|--------|
| URL | https://altsignals.io/ |
| Pricing 2026 | Forex $40/mo · Crypto AI (ActualizeAI) $48/mo · Quarterly -15% · Lifetime $400/$480. |
| User base | Pas de chiffre public, depuis 2017. |
| Funding | A levé en token (ASI presale, $3M+, 2023) — token-based, pas equity. |
| Feature gaps vs Sentinel | (1) Signals broadcast 1-to-N sans personnalisation. (2) Pas de narration structurée par signal (juste "BUY entry/SL/TP"). (3) Pas d'API natif. (4) ActualizeAI = ML black-box, pas LLM. |
| Forces | Communauté Telegram active, prix bas, multi-asset. |
| Faiblesses | Performances claimées non verified MyFXBook, churn élevé sur signal-resellers Telegram. |

---

## 3. Matrice features × prix × user-base × funding

> Légende : ✅ feature présente · ⚠️ partielle · ❌ absente · `?` non-vérifiable publiquement.

| Concurrent | Prix entry $/mo | Prix top $/mo | Users | Funding raised | XAU/FX | SMC/ICT | LLM narrative | Multi-TF | Telegram natif | Track-record vérifié | Multi-langue |
|------------|----------------|---------------|-------|----------------|--------|---------|---------------|----------|----------------|---------------------|--------------|
| **TradingView** | ~$15 | ~$60 | 15M MAU | $339M | ✅ | ⚠️ (overlays) | ⚠️ (Copilot beta) | ✅ | ❌ | ❌ | ✅ (33+ langues) |
| **LuxAlgo** | $27.99 | $32.99 | 100k+ | Bootstrap | ✅ | ✅ (SMC indicator) | ❌ | ✅ | ❌ | ❌ (repaint flagged) | ⚠️ (limité) |
| **TrendSpider** | $54 | $399 | 20k+ | ? | ⚠️ (forex partiel) | ❌ | ⚠️ (Sidekick chat) | ✅ | ❌ | ✅ (backtester) | ❌ |
| **Trade Ideas** | $89 | $178 | 5-15k est. | Bootstrap | ❌ (US equities) | ❌ | ❌ (Holly = ML) | ⚠️ | ❌ | ⚠️ (claims 60-65%) | ❌ |
| **StockHero** | $29.99 | $99.99 | « thousands » | ? | ❌ | ❌ | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| **3Commas** | $0/$29 | ~$99 | 100k+ | ? | ❌ (crypto) | ❌ | ❌ | ✅ | ❌ | ⚠️ | ⚠️ |
| **Composer** | $0 | $40 | ? (29 emp.) | $16.7M | ❌ (US stocks) | ❌ | ❌ | ⚠️ | ❌ | ✅ (live exec) | ❌ |
| **SignalStack** | $0/$27 | $250 | 33+ brokers | ? | ✅ (relai) | ❌ | ❌ | n/a | ❌ | n/a | ❌ |
| **Tickeron** | $60 | $250 | ? | ? | ❌ (US equities) | ❌ | ❌ | ✅ | ❌ | ⚠️ (claims 87%) | ❌ |
| **AltSignals** | $40 | $48 | ? | Token raise $3M+ | ✅ | ❌ | ❌ | ⚠️ | ✅ | ⚠️ (claims 1000pip/mo) | ⚠️ (EN/RU) |
| **Smart Sentinel AI** | $0 | $149 | 0 (alpha) | $0 | ✅ (XAU first) | ✅ (formalisé) | ✅ (Claude 4.6/4.7) | ✅ | ✅ | ⚠️ (en cours, voir baseline_2019_2025) | ❌ (EN-only) |

### 3.1 Lectures clés de la matrice

1. **Sentinel n'a aucune feature 100% unique** : chaque case ✅ est aussi cochée par au moins 1 concurrent. Le moat n'est PAS dans une feature isolée mais dans la *combinaison* (XAU spot + SMC + LLM narrative + Telegram + multi-TF + audit trail). Cette combinaison n'existe nulle part en 2026 — c'est *un wedge*.
2. **Sweet-spot prix concurrent = $27-$99/mo**. Les tiers ANALYST $49 / STRATEGIST $99 (`tier_manager.py:45,53`) sont pile dans la fenêtre. INSTITUTIONAL $149 vs Trade Ideas Premium $178 = défendable.
3. **Aucun concurrent ne sert une narration LLM 4.6/4.7 par signal validé**. AI Chart Copilot (TradingView) et Sidekick (TrendSpider) sont *chats généralistes*, pas des narrations *attachées* au signal. C'est l'angle le plus différenciant — mais voir §6 pour la défendabilité réelle.
4. **TAM hors-EN = ouverture massive**. AltSignals couvre EN/RU. Aucun acteur US ne sert FR/ES/PT correctement. Cf. eval_05 §8 : ajout `lang` quasi-zéro coût.
5. **Track-record vérifié MyFXBook = absent partout**. Si Sentinel publie un track-record audited (3rd-party, MyFXBook ou equiv.), c'est un *différenciateur trust* immédiat — surtout post-eval_25 (PMF/ICP wedge gold-trader-épuisé).

---

## 4. Trois différenciateurs uniques *défendables* (le cœur du verdict)

> Critère de "défendable" en 2026 : (1) coût/temps de copie ≥ 6 mois ; (2) requiert assets ou data que le copieur n'a pas ; (3) protégé par network effect, switching cost, ou compliance ; (4) *l'IA seule n'est plus défendable* — c'est commodity API.

### Différenciateur #1 — **Track-record auditable signal-by-signal sur Gold spot M15 (proof-of-edge)**

- **Description** : chaque signal généré (BUY/SELL/HOLD), son score, son contexte, la narrative LLM, et l'outcome (TP hit / SL hit / vol_realized vs vol_forecast) sont écrits dans `SignalStore` et exportables en CSV signé. Publication mensuelle 3rd-party verified (MyFXBook, FX Blue, ou équivalent on-chain pour transparence).
- **Pourquoi défendable** :
  - Concurrents ne peuvent pas *reconstruire le passé* : il faut accumuler 6-12 mois de signaux live pour avoir un track-record statistiquement significatif (≥ 200 trades).
  - Switching cost émotionnel : un trader qui voit "Sentinel: 247 signals depuis 6 mois, PF 1.32, hit-rate 51%, max DD -8R" ne switche pas vers un concurrent zéro-historique.
  - SignalStore + audit trail = *infra de moat-data* déjà construite (`src/api/signal_store.py`, voir `eval_12_signal_store.md`).
- **Pourquoi pas un concurrent ne l'a fait** : LuxAlgo refuse car repaint l'oblige à exposer signaux ex-post (litige). TradingView/Trade Ideas servent des plateformes, pas un produit de signaux singulier. AltSignals/Telegram resellers ont des intérêts économiques opposés à la transparence.
- **Coût de copie** : ≥ 6 mois calendaires + risque réputationnel si le track est mauvais. Forte barrière.
- **À faire pour activer** : Sprint 4 — fix `confluence_detector` (cf. `confluence_calibration.md`, score Pearson −0.023 = à reconstruire), brancher SignalStore.export → MyFXBook (via SignalStack ou MT5 EA), publier dashboard public read-only `/track-record`.

### Différenciateur #2 — **Hyper-spécialisation Gold (XAU/USD) M15 + macro-événements catalyseurs**

- **Description** : tous les concurrents listés sont *généralistes* (multi-asset). Sentinel mise tout sur XAU — un seul actif, une seule timeframe primaire (M15), avec calendrier macro (FOMC, NFP, CPI, jobless claims) intégré au scoring (`src/agents/news/economic_calendar.py` + ConfluenceDetector blackout — voir `news_pipeline.md`).
- **Pourquoi défendable** :
  - Le "smart-money flow gold" est *statistiquement différent* des FX/equities : sensibilité USD, real yields, geopolitical risk premium, sessions Asia-London-NY. Un modèle généraliste perd cette asymétrie.
  - SMC/ICT a été popularisé sur l'or (Michael Huddleston, "Gold Trading" cours), il y a une *demande active concentrée* (eval_25 ICP).
  - Couplage news/calendrier *par actif* est un gros chantier : LuxAlgo/TradingView ne le font pas (un overlay ne sait pas qu'il est devant CPI).
- **Coût de copie** : un concurrent généraliste devrait *spécialiser* — perd son TAM multi-asset. Personne ne le fera.
- **Risque** : c'est aussi un anti-moat si le marché gold se fait dépasser par crypto/equities en attention retail. Mitigation : XAU = $13T d'AUM 2026, marché institutionnel résiliant, retail XAU encore 4-5× plus petit que crypto.
- **À faire pour activer** : Sprint 5 — landing page `/gold-trading-ai` SEO-targeted, glossary ICT-gold, 5 articles cornerstone (cf. eval_25 wedge).

### Différenciateur #3 — **Narration LLM "audit-grade" avec rubric anti-hallucination + fallback déterministe**

- **Description** : combinaison (a) Sonnet 4.6 / Opus 4.7 routing par tier (cf. `eval_05_llm_implementation`), (b) rubric 25pts CI-bloquante (factuelle, actionnable, non-générique, anti-hallucination, vol-aware), (c) fallback `TemplateNarrativeEngine` automatique si circuit ouvert ou hallucination détectée, (d) audit trail "narrative_id → signal_id → model_used → cost_usd → quality_score" exportable.
- **Pourquoi défendable** :
  - Tous les concurrents qui font "AI explanation" en 2026 (TradingView Copilot, TrendSpider Sidekick) sont des *chats* — l'utilisateur tape sa question. Sentinel livre **proactivement** une narrative *contractuelle* (3 paragraphes, format machine-parsable, audited).
  - La rubric CI-bloquante est un *processus*, pas un feature ; reproduire = re-construire 4-6 semaines d'eval pipeline + dataset annoté.
  - Anti-hallucination certifiée = couverture marketing/légale (vs un Copilot qui dit "I might be wrong, verify with your broker"). Dans un marché secoué par scams (cf. §5), c'est un trust premium.
- **Coût de copie** : 4-6 semaines de chantier *strictement éval*, plus dataset propriétaire pour fine-tuning ou few-shots. Modéré.
- **Limite honnête** : à terme (12-18 mois), Anthropic / OpenAI publieront probablement des "evals templates" rendant cette barrière plus basse. Donc c'est un *moat de 12 mois max*, à transformer en moat-data (Diff #1) ou moat-marque (cf. plan B §5).
- **À faire pour activer** : finir le script `scripts/eval_narratives.py` (cf. `eval_05` top-5 #5), publier la rubric & 200 narratives notées en open-source whitelisted (geste de trust + SEO).

### 4.1 Différenciateurs *non-retenus* (et pourquoi)

| Candidat | Pourquoi PAS défendable |
|----------|------------------------|
| « SMC/ICT formalisé » | OSS Python `joshyattridge/smart-money-concepts` (MIT), LuxAlgo SMC indicator gratuit TradingView 1M+ users. Commodity. |
| « Multi-TF confluent » | TrendSpider, TradingView, LuxAlgo le font tous. Standard de marché. |
| « Volatility forecasting HAR-RV+LightGBM » | Académique connu (Corsi 2009 HAR), packages OSS dispo (`arch`, `garch_models`). Différenciateur *interne* (qualité moteur) mais pas *visible* utilisateur. |
| « Telegram delivery natif » | AltSignals + 1000s Telegram channels. Commodity. Le canal n'est pas un moat. |
| « 4 tiers FREE/ANALYST/STRATEGIST/INSTITUTIONAL » | Tier-design = Patrick Campbell 101, copiable en 1h. Pas un moat. |
| « Cache sémantique LLM » | Optimisation interne, pas user-visible. Non-utilisable en marketing. |
| « Open-source/transparency » (si choisi) | Defensible UNIQUEMENT si la communauté contribue effectivement. Historique solo-dev = peu probable sans investissement DevRel. |

---

## 5. Analyse avis clients — top 5 plaintes récurrentes (= opportunités)

> Sources : Trustpilot LuxAlgo (4.7/5 sur 1 293 reviews), Trustpilot ForexSignals.io & Prosignalsfx, Reddit r/algotrading (mentions LuxAlgo "repaint"), Quora, Medium ("Is LuxAlgo a Scam?"), avril 2026.

### Plainte #1 — **Repainting / look-ahead bias** (LuxAlgo, FluxCharts, indicateurs en général)

- **Verbatim agrégé** : « *Every single one of their indicators I tried repainted* » · « *LuxAlgo's code uses pivothigh — waits 21 candles before giving a signal — essentially knowing the future* » ([Medium analysis](https://medium.com/coinmonks/is-luxalgo-a-scam-ad50f39cb6d7)).
- **Fréquence** : très élevée — top 1 plainte technique LuxAlgo. Reddit r/TradingView 2024-2026 récurrent.
- **Opportunité Sentinel** : marketing direct "**No-repaint guarantee — every signal logged at the bar it fires, audit trail public**". Notre `SignalStore` capture `created_at` au timestamp exact ; il suffit de l'exposer publiquement. Coût marketing + 0.5j tech.

### Plainte #2 — **Performance claims non-vérifiées / fake screenshots**

- **Verbatim** : « *Providers post fake screenshots in groups, pretending signals had ended in profit even though some had actually closed in loss* » ([SureshotFX scam guide](https://sureshotfx.com/forex-trading-scam/)) · « *70% of positions hit stop loss, results shared are mostly fake* ».
- **Fréquence** : pandémique sur signal providers Telegram, élevée sur Tickeron ("87% accuracy" non-auditable).
- **Opportunité Sentinel** : MyFXBook integration + `/track-record` public dashboard (cf. Diff #1). Tagline : « *Every signal published is auditable. We publish losers too.* »

### Plainte #3 — **Auto-renewal abusif / surprise charges**

- **Verbatim** : « *Charged $479.88 for a yearly auto-renewal I was not expecting* » (LuxAlgo Trustpilot complaint).
- **Fréquence** : récurrente sur LuxAlgo (jump $335→$479 année 2), Tickeron ($250 auto-renew).
- **Opportunité Sentinel** : pricing transparency + email J-7 avant renouvel + flat pricing à vie clause (au moins année 2). Coût implémentation = 1j Stripe + 0.5j email template.

### Plainte #4 — **Latence signal / Telegram delivery missed**

- **Verbatim** : « *Frequent delays in receiving forex trading signals, which caused missed opportunities* ».
- **Fréquence** : top 3 sur signal subscribers tous segments.
- **Opportunité Sentinel** : SLA "signal delivered < 30s after bar close" exposé sur landing page + `/health` public endpoint montrant P95 latency. **Pré-requis** : fix circuit breaker timeout (eval_14), fix Telegram async (eval_13), monitoring Prometheus. Effort ~3j ingé déjà budgétisé.

### Plainte #5 — **Indicateurs visuels sans explanation / "why this signal?"**

- **Verbatim** : « *Got the signal but no idea why it fired* » (récurrent r/algotrading sur indicators-only platforms).
- **Fréquence** : structurelle sur LuxAlgo, Trade Ideas (Holly black-box), Tickeron.
- **Opportunité Sentinel** : c'est le **wedge marketing #1** — narration LLM rubric-tested. Slogan : « *Every signal explained in 3 paragraphs, by Claude Sonnet 4.6, audit-tested for hallucinations.* »

### 5.1 Synthèse opportunités → roadmap marketing

Les 5 plaintes ci-dessus sont **toutes adressables** par features ou messaging déjà *partiellement* implémentés chez Sentinel. Le risque est de communiquer ces avantages **avant** que les bugs `eval_10-15` (Telegram markdown injection, Sharpe incorrect, no backup, etc.) ne soient résolus — un acheteur trompé écrit la même Trustpilot review.

**Decision rule** : pas de claim marketing tant que :
- (a) `confluence_detector` reconstruit (Pearson > +0.10),
- (b) Telegram delivery 0 message perdu (eval_13 R1-R5),
- (c) `/track-record` public publié avec ≥ 30j de live signals.

---

## 6. Risque copycat 6 mois — Bloomberg / TradingView / LuxAlgo intègrent Claude

### 6.1 État de la menace en avril 2026 (ce qui existe DÉJÀ)

| Acteur | Move récent | Date | Menace pour Sentinel |
|--------|------------|------|---------------------|
| **Bloomberg** | ASKB (conversational AI in Terminal, beta) | févr. 2026 | Limité retail (Terminal $24k/an) mais signal directionnel : Bloomberg active sur LLM. ([Markets Media](https://www.marketsmedia.com/bloomberg-introduces-agentic-ai-to-the-terminal/)) |
| **TradingView** | AI Chart Copilot (Chrome extension, free public beta) | mars 2026 | **Critique** : free + 100M users + Pine Script ecosystem. Si TradingView fait passer Copilot en natif Telegram, écrase tout signal-provider sub-$50/mo. |
| **LuxAlgo** | Repositionnement "AI Algorithmic Trading Platform" + Discord AI bot | 2025-2026 | Élevé : LuxAlgo a 1M followers TradingView, peut router cette base vers tout nouveau produit. Si LuxAlgo embarque Anthropic API : moat #3 de Sentinel (LLM narrative) tombe à 0. |
| **Anthropic Claude for Financial Services** | Lancement début 2026 ([eWeek article](https://www.eweek.com/news/ai-finance-tools-anthropic-claude-perpexity/)) | janv. 2026 | Indirect : ouvre la voie à *toute* startup pour faire LLM-on-trading-data. Réduit le coût d'entrée. |

### 6.2 Scénarios à 6 mois et plan B chiffré

#### Scénario A — TradingView intègre LLM dans Alerts natif (proba 35%)

- **Ce qui arrive** : TradingView lance "Alert Copilot" : chaque alert Pine Script déclenche une narration LLM (4o-mini ou Claude Haiku, free dans Premium). Distribution = 100M users.
- **Impact Sentinel** : 60-80% du wedge "explication LLM par signal" évaporé en 90 jours. Stratégie tier ANALYST $49 = invendable.
- **Plan B chiffré** :
  1. **Pivoter sur le wedge XAU-only audit-grade institutionnel** (Diff #1 + #2). Cible : prop-shops, family offices, signaux INSTITUTIONAL $499-$999/mo.
  2. **Sortir du B2C et attaquer B2B-API** : licence l'engine SMC+vol+LLM à des brokers (IC Markets, Exness) qui veulent enrichir leur webtrader. Pricing $0.05-0.20 par signal généré, contrat 12 mois min, $30-60k ARR par broker.
  3. **CapEx supplémentaire estimé** : 0 (pivot pas re-build). OpEx marketing redirigé landing institutional + sales LinkedIn outbound. **Runway minimal nécessaire** : 6 mois × $3k/mo Anthropic + $2k/mo infra = $30k.

#### Scénario B — LuxAlgo embarque Claude/GPT pour narration (proba 50%)

- **Ce qui arrive** : LuxAlgo ajoute "Signal Insights AI" payant ($15/mo add-on) au-dessus de Premium. ARR LuxAlgo +$10M sans effort.
- **Impact Sentinel** : moat #3 perdu. Mais Diff #1 (track-record) et #2 (XAU-spe) tiennent.
- **Plan B chiffré** :
  1. Accélérer Diff #1 — publier track-record auditable verified MyFXBook **avant** que LuxAlgo ait stabilisé son AI add-on (LuxAlgo n'a pas track-record car indicators repaint, ils ne pourront *jamais* publier un track propre).
  2. Sortir un comparatif honnête : « *LuxAlgo Insights vs Smart Sentinel : same Claude API, but who shows you the trades that lost?* »
  3. Coût : 1 ingé × 2 semaines pour MyFXBook integration + landing page comparatif + 5 articles SEO. **$2-3k OpEx max.**

#### Scénario C — Bloomberg lance retail tier ASKB ($99-$199/mo) (proba 15%)

- **Ce qui arrive** : Bloomberg compresse ASKB en offre retail (challenge Composer/TradingView).
- **Impact Sentinel** : très limité court-terme (Bloomberg n'aime pas le retail < $5k/an), mais signal long-terme : la convergence devient réalité.
- **Plan B chiffré** :
  1. Conserver le focus XAU + SMC (Bloomberg ne fera *jamais* SMC formalisé, c'est culturellement incompatible — ils servent CIO macro, pas day-trader ICT).
  2. Pas d'action immédiate. Surveillance trimestrielle.

#### Scénario D — Anthropic lance "Claude Trading Copilot" managed service (proba 10%)

- **Ce qui arrive** : Anthropic rentre verticalement (Apple-style) avec un copilote trading.
- **Impact Sentinel** : catastrophique B2C (concurrence sur tarif unitaire infaisable). Mais B2B-API pivot reste possible.
- **Plan B chiffré** :
  1. Acquisition acqui-hire : positionner Sentinel comme talent SMC/ICT + pipeline opérationnel auprès Anthropic (ou OpenAI / Perplexity).
  2. Maintenir la qualité de l'eval pipeline et la documentation comme dossier acqui-hire.
  3. Coût : 0 — c'est l'option "exit" implicite.

### 6.3 Plan B principal (à activer si A ou B se matérialisent)

**Pivot vers B2B-API pour brokers + B2C ultra-vertical XAU institutionnel.**

- 6 semaines de re-positioning landing : `smartsentinel.ai/for-brokers` + `smartsentinel.ai/gold-pro`.
- 3 deals broker (IC Markets, Exness, Pepperstone) à **$30-60k ARR/broker** = $90-180k ARR plancher.
- 50 utilisateurs INSTITUTIONAL B2C × $499/mo = $300k ARR.
- **ARR plan B total cible** : $400-500k 12 mois post-pivot. Marge brute > 70% (LLM cost amorti par cache + tier routing).

---

## 7. Partnerships brokers — modalités réalistes

### 7.1 IC Markets

- **Programme** : Introducing Broker (IB) program, Affiliate program (white-label optional).
- **Modalités typiques** (non-publiques 2026 mais standard industrie) :
  - **Revshare** : 10-25% du spread/commission revenue par client référé, à vie compte.
  - **CPA** : $300-$600 par client actif (≥ $200 dépôt + ≥ 1 lot tradé).
  - **Volume tier-up** : >100 actives clients/mois → custom CPA jusqu'à $800.
- **Fit Sentinel** : excellent — IC Markets attire prop-trader / SMC traders (institutional ECN). Cible XAU 100% match. Webhook signals → MT4/MT5 EA pré-built ⇒ friction zéro.
- **Estimation 12 mois** : 200 leads convertis @ $400 CPA = **$80k**. Deuxième couche revshare ≈ $20-50k/an.

### 7.2 Exness

- **Programme** ([Exness Affiliates](https://www.exnessaffiliates.com/)) : CPA ($1 850 max/client) + RevShare (40% lifetime).
- **Modalités précises** :
  - **CPA** : up to $1 850 par client actif (variable selon pays + dépôt initial + plateforme).
  - **RevShare** : up to 40% of Exness revenue par trade, à vie. Daily payout.
- **Fit Sentinel** : bon mais Exness moins "pro" image que IC Markets. Audience XAU plutôt grand-public/Asie/Afrique = match audience eval_25 wedge LATAM/MENA si Sentinel multi-langue.
- **Estimation 12 mois** : 150 leads convertis @ moyenne $600 CPA effectif (pas tous au max) = **$90k**. RevShare lifetime = annuité $30-80k/an stable.

### 7.3 Pepperstone

- **Programme** : Partnership + API integration (cTrader / MT5).
- **Modalités** : moins publiques, négociation custom. Standard : revshare 20% spread, CPA $200-$500.
- **Fit Sentinel** : moyen — Pepperstone très AU/UK retail focus, moins XAU-pro que IC Markets. Mais cTrader API permet exécution native.
- **Estimation 12 mois** : 100 leads convertis @ $300 CPA = **$30k**. Revshare $10-30k/an.

### 7.4 Total potentiel brokers 12 mois

| Broker | CPA gross | RevShare/an | 12-mois total |
|--------|-----------|-------------|---------------|
| IC Markets | $80k | $35k | **$115k** |
| Exness | $90k | $55k | **$145k** |
| Pepperstone | $30k | $20k | **$50k** |
| **TOTAL** | **$200k** | **$110k** | **$310k ARR add-on** |

**Conditions** :
- Minimum 500-1500 utilisateurs Sentinel (FREE compris) pour produire 200-400 conversions broker.
- Build : webhook MT4/MT5 EA + dashboard partner + tracking link infrastructure. 4-6 semaines ingé. 
- Risque : conflit avec marketing direct subscription (un user broker-converted ne paye pas Sentinel sub). Mitigation : tier ANALYST gratuit pour clients broker liés (deal RevShare > $49/mo de toute façon).

### 7.5 Partnerships secondaires non-broker

- **TradingView Partner Program** : intégrer Sentinel comme overlay/indicator officiel. Coût : 0, exposition large.
- **MetaTrader Market** : publier EA/Indicator Sentinel-branded. Coût : licence MT5 dev + 50% revshare MQ.
- **Discord/Telegram influencers ICT-Gold** (5-10 micro-comm 10-50k abos) : revshare 30% premier mois, CPA optionnel. Budget pilote $5k.

---

## 8. Verdict commercial & note finale

### 8.1 Verdict

**Smart Sentinel AI n'a pas de moat aujourd'hui (avril 2026)**. La combinaison features est pertinente mais aucune barrière à l'entrée n'est érigée. Le projet est en *fenêtre de 6-9 mois* pour construire 1 ou 2 moats avant que TradingView ou LuxAlgo n'intègrent Claude/GPT en natif.

**Note défendabilité globale : 3.5 / 10** — projetable à **6.5 / 10 sous 9 mois** si :
1. Track-record auditable publié et alimenté en continu (Diff #1).
2. Hyper-spé XAU + macro-events monétisée (landing + SEO + 50 INSTITUTIONAL users).
3. Évaluation rubric narratives publiée open-source + 200 narratives notées (Diff #3 transformé en proof matérielle).

**Note défendabilité projetée 18 mois** : **7.5 / 10** si pivot B2B-API broker + 3 deals signés (revenu indépendant de la subscription B2C).

### 8.2 Top 3 moats RETENUS (ordre de priorité)

| # | Moat | Effort | Impact défendabilité |
|---|------|--------|----------------------|
| 1 | **Track-record auditable signal-by-signal Gold** (MyFXBook + `/track-record` public) | MT 4-6 semaines | +3.0 pts (de 3.5 → 6.5) |
| 2 | **Hyper-spé XAU/USD M15 + macro-calendar integration** (déjà 70% built) | QW 2 semaines (landing + SEO) | +1.5 pts |
| 3 | **Rubric narratives anti-hallucination open-sourced** (CI-bloquante + 200 narratives notées publiques) | MT 3 semaines | +1.0 pt + signal trust |

### 8.3 Plan B principal

**Pivot B2B-API vers brokers (IC Markets, Exness, Pepperstone) + B2C ultra-vertical XAU institutionnel ($499-$999/mo)** si :
- TradingView lance Copilot natif Telegram (proba 35% à 6 mois), OU
- LuxAlgo lance "Signal Insights AI" add-on (proba 50% à 6 mois).

ARR plancher cible plan B : **$400-500k 12 mois post-pivot** ($310k brokers + $300k 50 INSTITUTIONAL B2C, marge brute > 70%).

---

## 9. Top 5 actions prioritaires (effort × impact × délai)

| # | Action | Effort | Impact défendabilité | KPI 30j | Source/justification |
|---|--------|--------|----------------------|---------|----------------------|
| **1** | **Brancher MyFXBook export depuis SignalStore + publier `/track-record` public read-only** | MT 3 sem (1 ingé) | +3.0 pts (Diff #1) | ≥ 30j live signals visibles, ≥ 100 trades, PF > 1.0 OU plan d'action si non | `eval_12_signal_store.md` infra prête, manque connector + dashboard frontend |
| **2** | **Landing wedge `/gold-trading-ai` + 5 articles cornerstone SEO H1 2026** ("smart money concepts gold AI signals", "ICT order block AI detector XAU", etc.) | QW 1.5 sem (no-code Webflow + Claude-assisted writing) | +1.5 pts (Diff #2) | 500 visiteurs organiques/mois mois 3, ≥ 20 emails captured | Wedge eval_25 PMF, cf. §5 plaintes #5 explanation |
| **3** | **Publier rubric eval narrative (25pts) + 200 narratives notées open-source (Github + landing)** | MT 3 sem | +1.0 pt (Diff #3 trust signal) | 1 article HN/Reddit r/algotrading, ≥ 50 GitHub stars | Pré-requis : `scripts/eval_narratives.py` (eval_05 top-5 #5) |
| **4** | **Sign deal IB Exness + IC Markets** (negociation + integration MT5 EA + `/partner` page) | MT 4-5 sem (1 ingé + 0.5 BD) | $145k+ ARR potentiel + canal alternatif si copycat | 1 deal signé J60, 1 deal J90, premier paiement J120 | §7.1, §7.2 |
| **5** | **Multi-langue FR/ES/PT** (système prompt Sentinel + landing localisée) | QW 2 sem (Claude-translation + review) | +0.5 pt + ouverture LATAM/MENA TAM | ≥ 5% trafic landing FR/ES/PT mois 3 | eval_05 §8 cost ≈ 0, gain TAM 30%+ |

### 9.1 Backlog (impact moindre)

6. SLA latency public `/health` exposant P95 (réponse plainte #4) — QW 1j.
7. Comparatif "Sentinel vs LuxAlgo" page SEO honest comparison — QW 2j.
8. Affiliate program micro-influencers ICT-Gold (5 comm) — pilote $5k budget.
9. White-label MetaTrader Market EA — LT 6 sem.
10. Open-source SMC indicator Pine Script (community building) — MT 3 sem.

---

## 10. KPIs mesurables baseline → 30j → 90j → 12 mois

| KPI | Baseline avril 2026 | Cible 30j | Cible 90j | Cible 12 mois | Méthode mesure |
|-----|----------------------|-----------|-----------|---------------|----------------|
| Track-record signals live publiés | 0 | 30j × 100 trades | 90j × 300 trades | 12 mois × 1 200 trades | `/track-record` public + MyFXBook export |
| Profit Factor live (XAU M15) | 0.94 (replay, biais data) | ≥ 1.05 | ≥ 1.15 | ≥ 1.25 (audited) | `signal_tracker._compute_pf()` (à fixer cf. eval_12) |
| Subscribers payants total | 0 | 5 (alpha) | 30 (early adopters) | 200 (validation PMF) | `tier_manager.list_users()` |
| ARR | $0 | $200 | $2 000 | $35 000 ($150 ARPU × 200 + brokers) | Stripe MRR × 12 |
| Trafic organique landing/mois | 0 | 200 | 800 | 5 000 | GA4 / Plausible |
| Gold-related backlinks earned | 0 | 3 | 10 | 30 | Ahrefs/Backlinko |
| Couverture multi-langue (% trafic non-EN) | 0% | 5% | 15% | 30% | GA4 segment lang |
| Deals broker signés | 0 | 0 (negotiation) | 1 | 3 | CRM partnerships |
| Trustpilot rating | n/a | n/a | 4.5+ (5 reviews) | 4.7+ (50 reviews) | Trustpilot widget |
| Defensibility score interne (auto) | 3.5/10 | 4.0/10 | 5.5/10 | 7.5/10 | Re-eval 26 trimestriel |

---

## 11. Trade-offs assumés

| Décision recommandée | Trade-off explicite |
|----------------------|---------------------|
| Hyper-spé XAU (Diff #2) | TAM + ARPU plus petit qu'un produit multi-asset. **Mitigation** : XAU = $13T AUM 2026, ICT/SMC community ultra-active, LTV élevé (gold-trader fidèle). |
| Track-record public auditable (Diff #1) | Si premières 30j sont mauvaises, signal kill marketing. **Mitigation** : ne pas publier tant que `confluence_detector` reconstruit (Pearson > +0.10). Repousser publication de 4-6 sem si nécessaire. **Critère d'arrêt** : si après 90j de fix PF live < 1.0, pivoter vers "tool d'analyse SMC" (no-signal) — abandonner le claim signals. |
| Open-sourcing rubric eval (Diff #3) | Donne à concurrents le template pour rattraper. **Mitigation** : on sera 6 mois en avance et c'est le geste de trust qui paye le marketing (HN, papier de référence). Le moat passe de "secret" à "rapidité d'exécution". |
| Pivot B2B brokers en plan B | Cycle vente 6-9 mois, sale lourde, pas SaaS pur. **Mitigation** : 3 deals = $300k ARR = couvre 5x burn-rate en cas perte B2C complète. |
| Multi-langue (FR/ES/PT) | Risque qualité narrative LLM en non-EN. **Mitigation** : tester d'abord FR (proximité linguistique LLM), monitorer rubric score / langue. |
| Pas de différenciation sur "vol forecasting HAR-RV" | Capacité technique non-monétisée. **Mitigation** : intégrer dans narratives tier INSTITUTIONAL ("position sized to forecast vol of 1.4× ATR_14") = visible utilisateur, mais non-affiché comme moat externe. |

---

## 12. Benchmarks sectoriels (référencements)

- **TradingView pricing & user base** : [Pricing 2026](https://www.tradingview.com/pricing/), [Crunchbase financials](https://www.crunchbase.com/organization/tradingview/company_financials), [PitchBook](https://pitchbook.com/profiles/company/58597-66).
- **LuxAlgo user-base & Trustpilot** : [LuxAlgo Pricing](https://www.luxalgo.com/pricing/), [Trustpilot 4.7/5](https://www.trustpilot.com/review/luxalgo.com).
- **TrendSpider** : [Pricing](https://trendspider.com/pricing/), [Capterra reviews](https://www.capterra.com/p/10001451/TrendSpider/).
- **Trade Ideas Holly AI** : [Pricing](https://www.trade-ideas.com/pricing/), [Stockbrokers.com review 2026](https://www.stockbrokers.com/review/tools/trade-ideas).
- **StockHero V4 launch** : [Stockhero blog janv. 2026](https://blog.stockhero.ai/stockhero-version-4-launching-january-26-2026-the-future-of-automated-trading-is-here/).
- **3Commas** : [Pricing 2026](https://3commas.io/pricing).
- **Composer Series A $16.7M** : [PitchBook](https://pitchbook.com/profiles/company/455170-33), [Tracxn](https://tracxn.com/d/companies/composer/__pA20885_v_ub5A3IFCCMiivy6N4pQnnCtLikLqGC7s8).
- **SignalStack pricing 2026** : [SignalStack pricing](https://signalstack.com/pricing/).
- **Tickeron pricing & 87% claim** : [Tickeron 2026 review](https://www.findmymoat.com/tools/tickeron).
- **AltSignals pricing & lifetime** : [AltSignals top-20 forex providers](https://altsignals.io/post/top-20-best-forex-signals-providers).
- **Bloomberg ASKB launch** : [Markets Media févr. 2026](https://www.marketsmedia.com/bloomberg-introduces-agentic-ai-to-the-terminal/), [The Trade News](https://www.thetradenews.com/bloomberg-embeds-agentic-ai-into-the-terminal/).
- **Anthropic Claude for Financial Services** : [eWeek janv. 2026](https://www.eweek.com/news/ai-finance-tools-anthropic-claude-perpexity/).
- **TradingView AI Chart Copilot** : [Chartwisehub Q1 2026 update](https://chartwisehub.com/tradingview-updates-q1-2026/).
- **SMC OSS Python** : [joshyattridge/smart-money-concepts (MIT)](https://github.com/joshyattridge/smart-money-concepts).
- **Forex signal scams Trustpilot** : [SureshotFX scam guide 2026](https://sureshotfx.com/forex-trading-scam/), [Trustpilot ForexSignals.io](https://www.trustpilot.com/review/forexsignals.io).
- **Exness affiliate CPA $1 850** : [Exness Affiliates official](https://www.exnessaffiliates.com/affiliate-program/).
- **IC Markets IB program** : [WR Trading 13 best forex affiliates 2026](https://wrtrading.com/broker/forex-affiliate-programs/).

---

## 13. Ce qui manque à l'analyse (limites)

1. **Données précises user-base TrendSpider/Trade Ideas/Tickeron 2026** — sources publiques vagues. Estimations basées sur ARR proxy. À valider via SimilarWeb / SemRush si décision majeure dépend.
2. **Pas d'enquête utilisateur primary** — tous les "complaints" §5 viennent de Trustpilot/Reddit publics. Une 20-prospect interview campaign (cf. eval_25) validerait le ranking des plaintes.
3. **Pas d'analyse Telegram channels** privés — c'est le segment le plus opaque (AltSignals, Wolfx, etc. ont 50-200k abonnés combinés invisibles à WebSearch).
4. **Pas d'estimate ARR concurrents** précise — LuxAlgo bootstrap revenue $30-50M/an = guess. Si critique pour pricing strategy : payer abonnement Crunchbase Pro pour valider.
5. **Bloomberg ASKB pricing retail futur** : pure spéculation. Réévaluer Q3 2026.

---

## 14. Annexe — Actions concrètes file_path:line

1. `src/api/signal_store.py:247-275` — fix `_row_to_record` désérialisation `vol_forecast_atr/regime/confidence` (cf. eval_10_15 #1) — pré-requis Diff #1 track-record.
2. `src/api/signal_tracker.py:_compute_sharpe` — corriger le calcul Sharpe avant tout claim public (eval_10_15 #4).
3. `src/api/signal_store.py` — NEW route `/track-record` read-only public + endpoint export CSV signed pour MyFXBook integration (≈ 200 LOC).
4. `src/intelligence/confluence_detector.py:*` — refonte scoring fn (Pearson actuel −0.023, cf. `confluence_calibration.md`) — sans cela, track-record sera perdant.
5. `scripts/eval_narratives.py` — finir implémentation (cf. eval_05 top-5 #5), publier outputs en `reports/eval_05/narratives_scored.csv`.
6. `src/intelligence/llm_narrative_engine.py:25-26` — multi-langue : ajouter param `lang` au routing, system prompt FR/ES dispatch.
7. `infrastructure/landing/` (NEW) — pages `/gold-trading-ai`, `/track-record`, `/for-brokers`, `/comparison-luxalgo`.
8. `src/api/routes/partners.py` (NEW) — webhook partners, tracking link, dashboard partner read-only.
9. `tests/test_partner_routing.py` (NEW) — coverage IB/CPA tracking + idempotency.
10. Migration Stripe : ajouter clause "12-month price-lock guarantee" dans Terms (réponse plainte #3 auto-renew).

---

## 15. Verdict final & mantra

**Mantra du sprint compétitif Q2-Q3 2026** :

> « *On ne gagne pas en faisant la même chose que TradingView/LuxAlgo en mieux — on perd. On gagne en ÉCRIVANT publiquement le track-record qu'aucun d'eux ne peut publier (à cause du repaint, du scope généraliste, du conflit d'intérêt commission/signal). Le moat est dans la transparence radicale, pas dans l'IA.* »

Le moat technique (LLM + SMC + vol forecasting) tombe à 0 en 12 mois. Le moat-data (track-record auditable) commence à 0 et grandit chaque jour de signal live publié — c'est l'inverse de la courbe LLM. **Décision** : tout investir 90 jours sur Diff #1 + Diff #2.

---

## 16. Mise à jour suggérée du `MEMORY.md`

```
- [Eval 26 Competitive 2026-04-26](eval_26_competitive_findings.md) — Defensibility 3.5/10 today → 6.5/10 in 9 months. Top 3 moats: track-record auditable, hyper-spé XAU+macro, rubric LLM open-sourced. Plan B: pivot B2B-API brokers ($310k ARR, IC Markets+Exness+Pepperstone).
```

---

## 17. Annexe — Sources web (URLs et dates d'accès)

> Toutes URLs accédées le **2026-04-26** via WebSearch. Marquées "estimation" si non sourcées par chiffre public direct.

### Concurrents pricing & user-base
- TradingView Pricing 2026 — https://www.tradingview.com/pricing/ · https://checkthat.ai/brands/tradingview/pricing
- TradingView Crunchbase financials — https://www.crunchbase.com/organization/tradingview/company_financials
- TradingView PitchBook — https://pitchbook.com/profiles/company/58597-66
- TradingView AI Chart Copilot — https://chartwisehub.com/tradingview-updates-q1-2026/
- LuxAlgo Pricing — https://www.luxalgo.com/pricing/
- LuxAlgo Trustpilot — https://www.trustpilot.com/review/luxalgo.com
- LuxAlgo repaint critique — https://medium.com/coinmonks/is-luxalgo-a-scam-ad50f39cb6d7
- TrendSpider Pricing — https://trendspider.com/pricing/
- TrendSpider StockBrokers — https://www.stockbrokers.com/review/tools/trendspider
- Trade Ideas Pricing — https://www.trade-ideas.com/pricing/
- Trade Ideas StockBrokers — https://www.stockbrokers.com/review/tools/trade-ideas
- StockHero Pricing — https://www.stockhero.ai/pricing/
- StockHero V4 launch — https://blog.stockhero.ai/stockhero-version-4-launching-january-26-2026-the-future-of-automated-trading-is-here/
- 3Commas Pricing — https://3commas.io/pricing
- Composer PitchBook — https://pitchbook.com/profiles/company/455170-33
- Composer Tracxn — https://tracxn.com/d/companies/composer/__pA20885_v_ub5A3IFCCMiivy6N4pQnnCtLikLqGC7s8
- SignalStack Pricing — https://signalstack.com/pricing/
- Tickeron — https://tickeron.com/ · https://www.findmymoat.com/tools/tickeron · https://www.wallstreetzen.com/blog/tickeron-review/
- AltSignals — https://altsignals.io/post/top-20-best-forex-signals-providers
- 8 Best Forex Signals 2026 — https://www.valuewalk.com/investing/best-forex-signals-providers/
- FluxCharts — https://www.fluxcharts.com/ · https://pineify.app/resources/blog/fluxcharts-reviews-comprehensive-analysis-premium-trading-platform-2025

### Bloomberg / Anthropic / threats
- Bloomberg ASKB Markets Media — https://www.marketsmedia.com/bloomberg-introduces-agentic-ai-to-the-terminal/
- Bloomberg ASKB The Trade News — https://www.thetradenews.com/bloomberg-embeds-agentic-ai-into-the-terminal/
- Anthropic Claude Financial Services — https://www.eweek.com/news/ai-finance-tools-anthropic-claude-perpexity/
- Bloomberg Terminal alternatives — https://helmterminal.dev/blog/best-bloomberg-terminal-alternatives

### SMC / ICT context
- joshyattridge/smart-money-concepts (MIT) — https://github.com/joshyattridge/smart-money-concepts
- SMC ICT Guide 2026 ChartingLens — https://chartinglens.com/blog/smart-money-concepts-guide
- ICT Gold 2026 Mind Math Money — https://www.mindmathmoney.com/articles/smart-money-concepts-the-ultimate-guide-to-trading-like-institutional-investors-in-2025

### Plaintes / complaints
- Forex Signals scams 2026 — https://sureshotfx.com/forex-trading-scam/ · https://syntiumalgo.com/forex-signal-provider-scams/
- ForexSignals.io Trustpilot — https://www.trustpilot.com/review/forexsignals.io
- WolfxSignals Trustpilot — https://www.trustpilot.com/review/wolfxsignals.com
- ForexGDP signal Trustpilot reviews — https://www.forexgdp.com/signals-blog/trustpilot-reviews/

### Brokers partnerships
- Exness Affiliates official — https://www.exnessaffiliates.com/affiliate-program/
- Exness CPA up to $1850 (Dollarbreak) — https://dollarbreak.co.ke/exness-affiliate-program/
- 13 Best Forex Affiliates 2026 — https://wrtrading.com/broker/forex-affiliate-programs/
- 8 Best Forex Affiliate Programs (AffMaven) — https://affmaven.com/forex-affiliate-programs/
- Exness CPA vs RevShare guide — https://www.exnessaffiliates.com/blog/tips/revshare-vs-cpa-exness-partners/

— *Fin du rapport. Note défendabilité globale : 3.5 / 10 (court terme) → 7.5 / 10 (projeté 18 mois si Top 5 actions exécutées). Plan B activable sous 6 semaines si copycat se matérialise.*
