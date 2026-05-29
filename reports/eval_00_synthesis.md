# Eval 00 — Synthèse maîtresse 29 évaluations

> **Date** : 2026-04-28 · **Branch** : `main` · **Snapshot** : 632e9dd + uncommitted Sprint 2.
>
> **Mission** : consolider les 29 rapports `eval_01..29` en un document unique avec roadmap priorisée et critères go/no-go.

---

## TL;DR — Verdict global

**Note moyenne pondérée** : **5.0 / 10** (29 rapports).

**Verdict commercial** : ❌ **NON commercialisable en l'état**. 4 bloqueurs absolus go-live identifiés :
1. **Data feed XAU à 63 % couverture** (eval 08) — falsifie tout backtest publié
2. **Aucun walk-forward / IC bootstrap** (eval 18) — chiffres PF non-publiables
3. **`Procfile`/railway.toml lance `parallel_training.py`** (eval 22) — la prod ne run probablement PAS Sentinel
4. **`TESTING_MODE=1` par défaut + auth bypass silencieux** (eval 11) — API ouverte en prod si env var non explicite

**Note logiciel pure** : 7-8/10 (state machine, circuit breaker, persistence) — la partie ingénierie est solide.
**Note pipeline analytique** : 4-5/10 — score 0-100 sans pouvoir prédictif (Pearson −0.023), OB non-ICT, retest tol trop laxe.
**Note business / GTM** : 4-5/10 — ICP identifié (XAU SMC FR-first), mais blocked par PF < 1.20.

---

## 1. Tableau récapitulatif des 29 rapports

### 1.1 Couche technique (12 prompts)

| # | Module | Note | Verdict | Bloqueur ? |
|---|--------|:----:|---------|:----------:|
| 01 | Architecture pipeline | — | Single-process, 2 daemon threads, shared state sans lock | non |
| 02 | ConfluenceDetector | — | Score Pearson −0.023, double-gating empilé, tier system invalide | **OUI** scoring |
| 03 | SmartMoneyEngine | **4.5** | OB non-ICT (engulfing seule), retest tol 0.5 ATR ≈ spread, edge nul (PF 0.94 armed) | **OUI** scoring |
| 04 | Volatility Forecasting | **5.5** | Leakage in-sample blend-CV/résidu LGBM, claims RMSE non reproduits walk-forward | non |
| 05 | LLM Narrative | (livré) | `cache_control` no-op, NARRATIVE_MODE=template par défaut, cascade redondante | non (Sprint 1 livré) |
| 06 | SemanticCache | **5.0** | Pas sémantique malgré le nom (hash strict), hit rate plafond 30-45 %, multi-worker non-safe | non |
| 07 | SignalStateMachine | **8.0** | Code excellent (déterministe, thread-safe), defaults non empiriques | non |
| 08 | Data Providers | **3.5** | Feed XAU 63 % cov, 5/6 presets sans CSV, licence Dukascopy zone grise | **OUI** ❌ |
| 09 | SentinelScanner | **6.5** | Polling 60s fixe, mono-symbole, pas de queue notifications | non |
| 21 | Performance | **5.5** | Sync I/O bloque event loop, single-worker, pas scalable >1k MAU | non |
| 22 | Deployment | **4.5** | Procfile lance parallel_training.py PAS Sentinel, restart=never, port mismatch | **OUI** ❌ |
| 23 | MLOps | **4.5** | Maturity Level 1, skews train/serve = saignement 5-15% RMSE invisible | non |

### 1.2 API / Auth / Sécurité (6 prompts)

| # | Module | Note | Verdict | Bloqueur ? |
|---|--------|:----:|---------|:----------:|
| 10 | API FastAPI | **6.0** | RFC 7807 absent, sync SQLite bloque event loop, /metrics public, /operator no tier check | non |
| 11 | Auth & Tier | **4.5** | TESTING_MODE=1 default catastrophique, tier rate-limit dead code, HMAC partial signature | **OUI** ❌ |
| 12 | Signal Store | **5.5** | v3 deserialization bug, Sharpe incorrect, no backup, _current not multi-worker safe | non |
| 13 | Telegram | **3.5** | Markdown injection, ptb v20 async incompat, no rate limit, no feedback buttons | non |
| 14 | Circuit Breaker | **6.5** | Pas de timeout sur func(), pas de window failure rate, pas de persistence | non |
| 15 | Security | **5.0** | Pas de security headers, prompt injection /chat undefended, no CVE scan | non |

### 1.3 Backtest / Risk / Observability (4 prompts)

| # | Module | Note | Verdict | Bloqueur ? |
|---|--------|:----:|---------|:----------:|
| 16 | Observability | **3.2** | /metrics payload vide en prod, 109 print() dans 23 fichiers src/, aucun trace_id E2E | non |
| 17 | Testing | **5.5** | 1673 tests mais zones revenue sous-protégées (telegram 10%), 0 GitHub Actions | non |
| 18 | Backtest | **2.0** | ❌ Aucun walk-forward, coûts transaction $0, look-ahead MTF, pas d'IC bootstrap | **OUI** ❌ |
| 19 | Risk Mgmt | **4.5** | Pas de kill-switch op, pas de position-sizing live, 3 moteurs risk concurrents | non |

### 1.4 Business / GTM (7 prompts)

| # | Module | Note | Verdict | Bloqueur ? |
|---|--------|:----:|---------|:----------:|
| 20 | Multi-Asset | **5.0** | 6 presets ready mais 5/6 sans CSV. Keep XAU+EUR, drop BTC+US500 | non (post PF>1.2) |
| 24 | Unit Economics | **5.5** | Marges 78-98% reposent sur 3 hypothèses non vérifiées (cache 60%, dedup 95%, NARRATIVE=llm) | non |
| 25 | PMF/ICP | **4.5** | ICP gagnant = "XAU SMC FR-first $20-49/mo". GTM bloqué tant que PF<1.20 | non |
| 26 | Competitive | **3.5** today → 7.5/10 in 18 months. Top 3 moats: track-record auditable, hyper-spé XAU+macro, rubric LLM open-sourced | non |
| 27 | Pricing | (5/10) | Grille recommandée FREE/$29/$79/$1990. INSTITUTIONAL $149 sous-prixé ×13 | non |
| 28 | GTM | **5.8** | Plan solo 8-9h/sem, wedge FR-first KD 14, MRR M12 réaliste $5-7k vs BP $39k | non |
| 29 | Compliance | **3.5** | P0: geo-block US/QC/UK + disclaimer multi-langue + endpoints /terms/privacy (bloquant Stripe) | non (post PF) |

---

## 2. 4 bloqueurs absolus go-live (croisement)

### Bloqueur 1 — Data feed XAU à 63 % couverture (eval 08)

- **Fichier** : `data/XAU_15MIN_2019_2025.csv` à 63 % vs `XAU_15MIN_2019_2024.csv` à 97.6 %
- **Conséquence** : BOS firait sur 100 % des bars (root cause documenté dans `data_quality_audit_2026_04_23.md`)
- **Fix** : re-télécharger Dukascopy 2025 via `scripts/download_dukascopy_xau.py` ; coverage gate au boot (5 lignes)
- **Effort** : 2 h
- **Bloque** : tout chiffre de backtest publiable (eval 18)

### Bloqueur 2 — Aucun walk-forward / IC bootstrap (eval 18)

- **État** : 19 `replay_*.json` à la racine, tous in-sample single-fold. PF rapportés (0.96, 1.086, 0.39…) sans IC 95 %.
- **Inflation Hansen SPA estimée** : +15-25 %.
- **Fix** : exécuter `reports/eval_18_walkforward_skeleton.py` + brancher `execution_model.py` (DynamicSlippage + DynamicSpread) dans replay + écrire `scripts/montecarlo_bootstrap.py`
- **Effort** : 3-5 jours
- **Bloque** : tout marketing chiffré (`BACKTEST_LEGAL_GUARDRAILS.md` §4 = langage qualitatif obligatoire en attendant)

### Bloqueur 3 — Procfile / railway.toml lancent `parallel_training.py` (eval 22)

- **État** : la prod déployée Railway run probablement encore l'ancien pipeline RL, **pas** `python -m src.intelligence.main`
- **Conséquence** : tout déploiement actuel n'est pas le SaaS Smart Sentinel AI documenté
- **Fix** : 3 lignes — éditer `Procfile`, `railway.toml`, vérifier `SENTINEL_TESTING_MODE=0` set explicit
- **Effort** : 30 min
- **Bloque** : tout déploiement réel

### Bloqueur 4 — `TESTING_MODE=1` par défaut + auth bypass (eval 11)

- **État** : `auth.py:22` `TESTING_MODE=os.environ.get("SENTINEL_TESTING_MODE","1")=="1"` — défaut **OPEN**
- **Conséquence** : déploiement sans config explicite = API ouverte avec INSTITUTIONAL pour tout le monde
- **Fix** : changer le défaut à `"0"`, logger.warning au boot si `TESTING_MODE=1`, fail-closed
- **Effort** : 15 min
- **Bloque** : tout paid launch / B2B / Stripe

**Total bloqueurs : ~5 j cumulés**. Sans ces 4 fixes, **toute mise en prod commerciale est dangereuse**.

---

## 3. Roadmap Sprint 1 / 2 / 3

### Sprint 1 — "Unblock Go-Live" (1 semaine, ~30 h)

**Objectif** : lever les 4 bloqueurs absolus + 8 quick-wins triviaux cross-cutting.

| Action | Source | Effort | Impact |
|--------|--------|--------|--------|
| Fix Procfile + railway.toml | eval 22 | 30 min | déblocage déploiement |
| `TESTING_MODE=0` par défaut + log warning | eval 11 | 15 min | auth fail-closed |
| Re-download XAU 2025 propre + coverage gate boot | eval 08 | 2 h | déblocage backtest |
| Walk-forward 6-ans + IC bootstrap | eval 18 | 4 j | déblocage marketing chiffré |
| `MetricsRegistry` instanciée dans main.py | eval 16 | 1 h | /metrics fonctionnel |
| `cleanup_expired` cache branché au scanner | eval 06 | 15 min | DB ne grossit plus |
| `cache.get_stats()` exposé `/health` | eval 06 | 15 min | observabilité |
| Connection pool SQLite partagé | eval 10-15 audit | 2 h | -80 % latence auth |
| `lru_cache` sur `verify_key` | eval 11 | 30 min | -70 % SQL |
| Module-level `re.compile` `sanitize_string` | eval 15 | 15 min | -30 % CPU |
| `GZipMiddleware` FastAPI | eval 10 | 15 min | -60 % bandwidth |
| Inconsistency check `from_dict` state machine | eval 07 | 1 h | robustesse persistence |

**Sortie Sprint 1** : produit déployable en interne (FREE-only soft launch), backtest publiable avec IC.

### Sprint 2 — "Trust Layer" (2 semaines, ~60 h)

**Objectif** : crédibiliser le scoring + sécuriser le pipeline live.

| Action | Source | Effort |
|--------|--------|--------|
| OB ICT-compliant (ancrer BOS + body/ATR≥1.5 sur impulse) | eval 03 | 3 j |
| FVG threshold 0.4 ATR + tracking remplissage | eval 03 | 1 j |
| Retest tol 0.25 ATR + touch strict | eval 03 | 1 j |
| Sweep 432 cellules state machine + recalibrage defaults | eval 07 | 2-3 j |
| Notification queue + replay sur circuit close | eval 09 | 2 j |
| Polling event-driven + bar-aligned `Event.wait` | eval 09 | 1 j |
| Kill-switch opérationnel + position-sizing live | eval 19 | 4 j |
| Fix telegram parse_mode Markdown V2 + escape | eval 13 | 1 j |
| Tier rate-limit câblé (`tier_manager.check_rate_limit`) | eval 11 | 1 j |
| `/operator/*` tier-gated INSTITUTIONAL only | eval 10 | 4 h |
| Security headers middleware (HSTS, CSP, X-Frame) | eval 15 | 30 min |
| Prompt injection defense `/chat` | eval 15 | 1 j |

**Sortie Sprint 2** : produit défendable techniquement face à un audit externe + ready pour soft-launch FREE Discord.

### Sprint 3 — "Commercial-grade" (3-4 semaines, ~90 h)

**Objectif** : ouvrir Stripe, support 100+ MAU, multi-asset.

| Action | Source | Effort |
|--------|--------|--------|
| Souscrire Tiingo $30/mo + migrer historique | eval 08 | 1 j |
| Pipeline ingestion live WebSocket Polygon.io | eval 08 | 1 sem |
| MultiSymbolScanner unifié + agents thread-safe | eval 09 | 3-4 j |
| Compteurs cache multi-worker safe (Redis) | eval 06 | 2 j |
| Onboarder data EURUSD M15 + valider PF > 1.0 OOS | eval 20 | 3 j |
| Replacer scoring fn ConfluenceDetector (PF Pearson nul) | eval 02 | 1 sem |
| OpenTelemetry trace_id E2E | eval 16 | 1 sem |
| Geo-block US/QC/UK + disclaimer multi-langue | eval 29 | 3 j |
| Endpoints /terms /privacy (bloquant Stripe) | eval 29 | 1 j |
| Stripe integration + 4 tiers + grille $29/$79/$1990 | eval 27 | 3 j |
| Trial 14j sans carte | eval 27 | 1 j |
| Wedge GTM FR-first (KD 14, vol 3 780/mo) | eval 28 | continu |

**Sortie Sprint 3** : Stripe live, FR-first wedge, 50 paid abonnés cible M9.

---

## 4. KPIs go/no-go phasés

### Avant Soft-launch FREE (Sprint 1+2)
- ✅ `Procfile` corrigé + `TESTING_MODE=0` explicit
- ✅ XAU coverage ≥ 95 %
- ✅ Walk-forward 6-ans + IC bootstrap publié
- ✅ Kill-switch opérationnel
- ✅ Markdown V2 fix Telegram

### Avant Paid launch (Sprint 3 P0)
- ✅ PF walk-forward OOS > 1.20 net coûts (eval 18)
- ✅ Tiingo souscrit (licence commerciale data)
- ✅ Geo-block + disclaimers multi-langue
- ✅ Endpoints /terms /privacy live
- ✅ SemanticCache renommé OU vrai layer sémantique
- ✅ Tier rate-limit câblé
- ✅ Security headers + prompt injection defense

### Avant Scale 1k MAU (Sprint 3 P1)
- ✅ MultiSymbolScanner unifié
- ✅ WebSocket live Polygon.io
- ✅ Compteurs Redis multi-worker
- ✅ OpenTelemetry trace_id E2E
- ✅ MLflow / drift monitor

---

## 5. Notes finales par couche

### Logiciel pure (note moyenne 7/10)

Le code applicatif (state machine, circuit breaker, persistence, semantic_cache structure) est **bien écrit** — déterministe, thread-safe, fail-fast, testé. C'est la couche qui passe le mieux un audit externe.

**À mettre en avant marketing** : "anti-spam déterministe à 5 règles documentées, persistence atomique, fallback algorithmique, circuit breaker double couche".

### Pipeline analytique (note moyenne 5/10)

Le scoring 0-100 ne prédit rien (Pearson −0.023). Les détecteurs SMC sont approximatifs (OB engulfing, FVG laxe, retest 0.5 ATR). Le replay backtest est in-sample sans IC.

**À fixer prioritairement** : OB ICT-compliant + remplacer fn scoring + walk-forward + Monte Carlo bootstrap. Sans ça, **toute promesse marketing chiffrée est mensongère**.

### Business / GTM (note moyenne 5/10)

ICP clair (Persona Marc, scalper XAU FR/EN, $20-49/mo). Wedge GTM identifié (FR-first SEO KD 14). Pricing grille recommandée. Compliance roadmap claire.

**Bloqué par PF < 1.20** — toutes les actions GTM doivent être subordonnées à la fix produit (Sprint 1+2+3 P0).

---

## 6. Recommandation finale (1 paragraphe)

Smart Sentinel AI a un **socle ingénierie solide** mais **3 bloqueurs scientifiques** (data feed, scoring non-prédictif, backtest sans IC) et **2 bloqueurs ops** (Procfile cassé, TESTING_MODE défaut). Investir 1 semaine sur Sprint 1 (lever bloqueurs absolus + 8 quick-wins cross-cutting) pour passer en interne deployable. Investir 2 semaines sur Sprint 2 (OB ICT, retest, sweep state machine, kill-switch) pour défendre le scoring face à un audit. Investir 3-4 semaines sur Sprint 3 (Tiingo, WebSocket, multi-asset, Stripe, geo-block) pour ouvrir au paiement. **Total ~6-7 semaines de dev solo pour go-live commercial défendable**, vs essayer de lancer marketing avant la fix produit (catastrophique).

Avant tout cela : **statuer empiriquement sur le PF walk-forward** (Sprint 1 P0). Si PF < 1.20 OOS net coûts → pas de paid launch ; pivot vers le scoring v2 (Sprint 3 P1) ou plan B-API B2B brokers (eval 26).

---

## 7. Index des rapports

| Rapport | Fichier |
|---------|---------|
| 01 Architecture | `reports/eval_01_architecture.md` |
| 02 ConfluenceDetector | `reports/eval_02_confluence.md` |
| 03 Smart Money | `reports/eval_03/eval_03_smart_money.md` |
| 04 Volatility | `reports/eval_04_volatility.md` |
| 05 LLM Narrative | `reports/eval_05_llm.md` |
| 06 SemanticCache | `reports/eval_06_semantic_cache.md` |
| 07 SignalStateMachine | `reports/eval_07_signal_state_machine.md` |
| 08 Data Providers | `reports/eval_08_data_providers.md` |
| 09 SentinelScanner | `reports/eval_09_sentinel_scanner.md` |
| 10 API FastAPI | `reports/eval_10_api.md` |
| 11 Auth & Tier | `reports/eval_11_auth.md` |
| 12 Signal Store | `reports/eval_12_signal_store.md` |
| 13 Telegram | `reports/eval_13_telegram.md` |
| 14 Circuit Breaker | `reports/eval_14_circuit_breaker.md` |
| 15 Security | `reports/eval_15_security.md` |
| 16 Observability | `reports/eval_16_observability.md` |
| 17 Testing Suite | `reports/eval_17_testing.md` |
| 18 Backtest | `reports/eval_18_backtest.md` |
| 19 Risk | `reports/eval_19_risk.md` |
| 20 Multi-Asset | `reports/eval_20_multi_asset.md` |
| 21 Performance | `reports/eval_21_performance.md` |
| 22 Deployment | `reports/eval_22_deployment.md` |
| 23 MLOps | `reports/eval_23_mlops.md` |
| 24 Unit Economics | `reports/eval_24_unit_economics.md` |
| 25 PMF/ICP | `reports/eval_25_pmf_icp.md` |
| 26 Competitive | `reports/eval_26_competitive.md` |
| 27 Pricing | `reports/eval_27_pricing.md` |
| 28 GTM | `reports/eval_28_gtm.md` |
| 29 Compliance | `reports/eval_29_compliance.md` |
| **Audit team 10-15** | `reports/eval_10_15_team_audit.md` (57 deltas additionnels) |

---

## 8. Index mémoire (`.claude/memory/`)

`MEMORY.md` mis à jour avec 26 entrées eval pointant vers les rapports. Voir `## Eval reports index (29 prompts)` dans MEMORY.md.

---

**Date génération** : 2026-04-28
**Prochaine action recommandée** : exécuter Sprint 1 (4 bloqueurs absolus + 8 quick-wins, ~30 h cumulées) avant toute autre work. Voir §3.
