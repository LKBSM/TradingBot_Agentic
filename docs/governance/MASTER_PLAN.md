# MASTER PLAN — M.I.A. Markets · PRE-LAUNCH

**Source unique de vérité · Phase Dev pure** · Date : 2026-05-27 (révisé post-audit algo) · Statut : actif
**Objectif** : produit **fonctionnel à la perfection** avant tout launching commercial.
**Hors-scope** : Stripe live, marketing, légal commercial, RC Pro, acquisition. Tout ça sera planifié quand le produit sera prêt.

## ⚠️ Mise à jour 2026-05-27 — Pivot positioning + pricing révisé

Suite à l'audit `AUDIT_ALGO_2026_05_27.md` (note 3/10, scoring Pearson −0.023, backtest 7 ans PF 0.786) :

- **Positionnement** : "outil de compréhension augmentée" (PLUS de "système de trading")
- **Pricing révisé** : FREE / **9 €** / **19 €** (au lieu de FREE / $29 / $79)
- **INSTITUTIONAL retiré de la grille publique** (devient "Contact us" / Calendly)
- **Tiers PREMIUM/STANDARD/WEAK supprimés du visible** (cosmétiques, empiriquement vides)
- **Tous claims chiffrés (PF 1.30, IC, win rate, 329 setups) retirés** tant que pas validé OOS
- **Sprint 1 augmenté** : Blockers #2 (coûts transactionnels), #3 (bootstrap CPCV), #4 (look-ahead) ajoutés
- **Gate de promotion premium** : Brier OOS > +2 % AND DSR > 1.0 AND PBO < 0.5 (cf. `decisions/2026-05-27_pivot_positioning_audit.md`)

---

## 📍 Où on en est

### Vision produit

**M.I.A. Markets** (Multi-asset Intelligence Assistant for Markets) = **indicateur de marché conversationnel** Or + FX. Bilingue EN + FR. Compréhension augmentée, jamais promesse de profit.

3 couches d'information :
- **Hero permanent** : verdict + PF historique + alerte event
- **Sections collapsibles** : régime, vol, structure, conviction
- **Détail technique** : waterfall 8 composantes + conformal viz + sources RAG (tier-gated)

**Chatbot Sentinel** = moat conversationnel.

### Contexte

- Loukmane Bessam · résident Québec, citoyen canadien
- Premier business
- Marché cible monde anglo + franco (Phase 1 = 9 pays, sans US)
- Bilingue EN + FR égalité dès J1
- Domain : `mia.markets`

### État actuel du code (commits récents)

| Commit | Item | État |
|---|---|---|
| `066eff0` | Chatbot Claude SSE streaming (/api/chat) | ✅ Base fonctionnelle |
| `c8407f5` | Rebrand M.I.A. Markets | ✅ Fait |
| `bf425a5` | Landing + how-it-works + pricing teaser | ✅ V0 visible |
| `902f260` | ChatPanel + Q&A scriptés + refus pédagogique | ✅ V0 fonctionnel |
| `6aec3b9` | Sections collapsibles | ✅ Fait |
| `505eb72` | MarketReadingCard hero | ✅ Fait |
| `e61c3e4` | Next.js 15 + shadcn/ui | ✅ Bootstrap fait |
| `f40d641` | InsightSignalV2 + LGBM-Isotonic-Conformal pipeline | ✅ Base existe (à valider prod) |
| `5895f2e` | LLM narrative generator | ✅ Fait |

---

## 🎯 Définition "fonctionnel à la perfection"

C'est plus exigeant que "performance-ready". Ça couvre :

### A. Solidité technique
- Zéro bug critique ou bloquant connu
- Tests couverture ≥ 85 % zones revenue, ≥ 70 % autres
- Toutes les vulnérabilités sécurité fermées
- Pas de scoring cosmétique (Brier skill validé empiriquement)

### B. Robustesse opérationnelle
- Aucun message Telegram perdu silencieusement
- Cost monitoring Anthropic actif avec hard cap fail-closed
- Sentry capture toutes exceptions
- Circuit breakers calibrés sur 30j de logs réels
- Backups quotidiens testés restaurabilité

### C. Qualité UX
- Mobile-first responsive (Lighthouse ≥ 95 mobile)
- Accessibility WCAG AA (a11y)
- Loading states + error states + offline detection
- Animations + transitions fluides (60fps)
- i18n FR + EN parité totale (zéro `[MISSING_TRANSLATION]`)

### D. Cohérence produit
- Chatbot répond à 30+ questions sans hallucination
- Aucun vocabulaire interdit dans tout le produit
- Refus pédagogique 100 % patterns adversariaux
- Sources RAG cliquables (12 papers académiques)
- Track-record public mis à jour quotidiennement avec IC bootstrap

### E. Maintenabilité
- Documentation interne complète (runbooks, ADRs, architecture)
- Tests E2E complets sur parcours critiques
- Onboarding développeur < 1h (si tu reviens dans 6 mois)
- Logs structurés + monitoring dashboard

**Quand ces 5 piliers sont validés → le produit est launching-ready.**

---

# 🏗️ PLAN PRE-LAUNCH — 8 sprints dev (~14-18 sem)

---

## Sprint 1 — Cœur algorithmique (3-4 sem)

**Objectif** : scoring VRAIMENT calibré, validé empiriquement, sweep state machine livré.

### Items détaillés

#### 1.1 — Audit DG-025 existant
- Vérifier `models/calibrated_conviction_v1.pkl` chargé en prod
- Vérifier que `CalibratedConvictionPipeline` est utilisé par `ConfluenceDetector`
- Logs : confirmer que la conviction renvoyée passe bien par LGB → Isotonic → ACI
- **Sortie** : rapport `docs/audits/scoring_v2_audit_M0.md` avec verdict GO/NO-GO

#### 1.2 — Finition DG-025 si gaps
- Si modèle absent : entraînement Colab walk-forward CPCV sur XAU 7 ans (López de Prado 2018, k=5 folds purged)
- Features : 8 composantes (BOS, FVG, OB, regime, vol_forecast, news, momentum, RSI_div)
- Hyperparams LGB : 200 trees, num_leaves=31, learning_rate=0.05, early stopping
- Isotonic recalibration sur out-of-fold predictions
- ACI (Gibbs & Candès 2021) wrapper, γ=0.05, buffer_size=500
- Export `models/calibrated_conviction_v1.pkl`
- Deploy via feature flag `SCORING_VERSION=v2`, fallback v1 si absent

#### 1.3 — Validation empirique scoring v2
- Métriques sur test set walk-forward :
  - Brier skill ≥ +5 % vs baseline naïf
  - DM test (Diebold-Mariano) p < 0.05
  - Calibration : reliability diagram OK
  - Coverage conformel observée ≥ 90 %
- Rapport `reports/scoring_v2_validation.md` avec graphs

#### 1.4 — DG-034 sweep state machine 432 cellules
- Sweep : `enter_threshold × exit_threshold × confirm_bars × cooldown_bars × max_age_bars × silent_period`
- Cellules : 6 × 4 × 3 × 3 × 2 × 2 = 432
- Sur 7 ans XAU + EUR (Colab GPU si dispo)
- Métrique : PF moyenne × IC 95 % × n_trades / cellule
- Choisir defaults Pareto-optimaux
- Mettre à jour `src/intelligence/signal_state_machine.py` defaults
- Tests régression sur defaults

#### 1.5 — DG-004 archive feed corrompu
- `data/XAU_15MIN_2019_2025.csv` (63 % coverage) → `data/_archived/XAU_15MIN_2019_2025_FY63_CORRUPTED.csv`
- README dans `_archived/` expliquant root cause
- Vérifier grep zero référence en prod

#### 1.6 — DG-053 verify_data_quality boot fail-fast
- Au boot du scanner, vérifier chaque feed :
  - Coverage ≥ 95 %
  - Pas de gaps > 1h
  - Pas de doublons timestamp
  - Bid-ask spread cohérent (XAU < 0.5 $, FX < 5 pips)
- Si fail : log ERROR + exit code 1
- Env var override `STRICT_DATA_QUALITY=false` pour urgence (mais log WARNING)

#### 1.7 — Tests Sprint 1
- Couverture ≥ 85 % sur `src/intelligence/scoring/`
- Couverture ≥ 80 % sur `src/intelligence/signal_state_machine.py`
- Tests unitaires LGB, Isotonic, ACI individuellement
- Tests intégration pipeline complet sur 100 signals mock
- Tests régression scoring v2 vs v1 (différentiel attendu documenté)

#### 1.8 — BLOCKER #2 (audit 2026-05-27) — Brancher coûts transactionnels
- Wire `DynamicSpreadModel` et `DynamicSlippageModel` (existent) dans `state_machine_replay.py::_build_trade()`
- Coûts réalistes XAU : 5 bps spread + 5 bps slippage + ~$7 commission ≈ −0.125 R/trade
- Re-run backtest 2019-2025 avec coûts intégrés → mesurer PF ajusté
- **Acceptance** : tous les rapports replay incluent désormais la ligne "costs included: YES"

#### 1.9 — BLOCKER #3 (audit 2026-05-27) — Bootstrap IC 95 % + walk-forward CPCV + PBO + DSR
- Implémenter `scripts/bootstrap_ic95.py` : 1000 itérations resample avec replacement sur trade returns
- Calculer IC 95 % du Profit Factor
- Walk-forward CPCV (Combinatorial Purged CV) k=5 folds purged (López de Prado 2018)
- PBO (Probability of Backtest Overfitting) selon Bailey-López de Prado 2014
- DSR (Deflated Sharpe Ratio) selon Bailey-López de Prado 2014
- Rapport `reports/scoring_v2_OOS_validation.md`
- **Acceptance** : Brier skill OOS, DSR, PBO, IC publiables avec méthodologie reproductible

#### 1.10 — BLOCKER #4 (audit 2026-05-27) — Patcher look-ahead bug B2
- Bug : `src/environment/multi_timeframe_features.py:554-566` swing detector utilise `iloc[i+1], iloc[i+2]` → lookahead
- Patch : ajouter `.shift(+2)` pour rendre causal
- Re-run backtest pour comparer impact avant/après
- Test régression : `tests/test_multi_timeframe_causal.py` valide qu'aucune feature n'utilise le futur
- **Acceptance** : zéro feature lookahead, backtest invalidé puis re-validé causal

### Gate Sprint 1 (AUGMENTÉ post-audit)
- [ ] Scoring v2 chargé prod, Brier skill ≥ +5 % vs baseline OOS
- [ ] Sweep 432 cellules livré, defaults empiriques mis à jour
- [ ] Feed corrompu archivé, verify_data_quality boot fail-fast actif
- [ ] Couverture tests ≥ 85 % scoring
- [ ] **Coûts transactionnels intégrés** (Blocker #2) ✓
- [ ] **Bootstrap IC + CPCV + PBO + DSR calculés** (Blocker #3) ✓
- [ ] **Look-ahead B2 patché** (Blocker #4) ✓
- [ ] **Rapport `scoring_v2_OOS_validation.md` publié** avec méthodologie

### Gate de promotion pricing premium (post Sprint 1)

Pour passer au-delà du pricing FREE / 9 € / 19 € :
- Brier skill OOS > +2 %
- DSR > 1.0
- PBO < 0.5

Si tous validés simultanément → claims publiables, pricing premium réactivable, INSTITUTIONAL réintroduit grille publique.

### Révision pricing à M+3 (décision conditionnelle)

**À évaluer M+3** (≈ 2026-08-27) : ajout d'un **tier "Pro" intermédiaire 39-49 €/mois** pour effet d'ancrage commercial.

**Logique** : pricing actuel FREE/9€/19€ est correct pour démarrer (faible risque, posture honnête) mais **sans effet d'ancrage** — le 19 € apparaît comme "tier d'élite" alors qu'il devrait être "tier d'entrée pour utilisateurs sérieux".

**Conditions de déclenchement à M+3** :
- Traction observée : ≥ 30 abonnés Approfondie 19 € actifs sur 60 j glissants
- OU Gate de promotion premium validée (Brier > +2 % AND DSR > 1.0 AND PBO < 0.5) → ouvre la possibilité d'un tier supérieur substancié
- Stabilité produit : aucune régression critique sur 30 j

**Si conditions remplies** : introduire **tier Pro 39-49 €/mo** avec features additionnelles :
- 4 actifs (XAU + EUR + 2 autres si Polygon souscrit)
- Chatbot illimité + cache prioritaire (latence < 1s)
- Accès anticipé features V2 (notifications proactives, mode comparaison 2 actifs, replay chart annoté)
- Email digest hebdomadaire premium
- Support prioritaire

**Si conditions NON remplies à M+3** : maintien FREE/9€/19€, ré-évaluation à M+6.

**Pourquoi ne pas le faire maintenant** : sans traction validée, ajouter un tier intermédiaire dilue le funnel + crée confusion + suggère qu'on a "trouvé un edge premium" alors qu'on est en pleine validation OOS. Décision conditionnelle ferme.

### Cap abonnés bootstrap — Condition de retrait

**Cap actuel** : 50 abonnés payants total (Découverte + Approfondie cumulés) pendant phase bootstrap (M0-M3 selon stratégie légale `legal_bootstrap_strategy_2026_05_26.md`).

**⚠️ Risque auto-plafonnement** : revenue projeté max au cap ≈ 50 × 14 €/mo moyen = ~700 €/mo, **en dessous de l'objectif budget avocat fintech 3-5 k€/an** (~250-420 €/mo cumulé sur 12 mois). Si le cap n'est pas retiré post-légal, on se plafonne soi-même sous le budget avocat.

**Condition de retrait** :

> **Le cap 50 abonnés DOIT être retiré au plus tard 30 jours après livraison du terminal légal low-cost (Iubenda Pro souscrit + RC Pro Freelance souscrite + CGU/Privacy V0 publiées + médiation conso adhérée).**

Cible : **S6-S7 du plan dev** (≈ M+4-M+5).

**Pourquoi 30j max** : laisser un buffer raisonnable pour observation post-livraison légale (incidents éventuels, ajustements wording, validation Stripe). Au-delà = procrastination = auto-sabotage commercial.

**Action concrète à T+30j post-légal** :
1. Augmenter `GLOBAL_PAID_USERS_CAP` env var de 50 → 200 (ou retirer le cap si Stripe + RC Pro + CGU permettent ouverture sans risque)
2. Documenter dans `docs/governance/decisions/YYYY-MM-DD_cap_removal.md` la décision + justification
3. Mettre à jour landing/pricing pour retirer la mention "limité à 50 abonnés"

**Si incident légal majeur entre T0 et T+30j** : reporter retrait cap de 30 j supplémentaires, mais documenter pourquoi.

---

## Sprint 2 — Sécurité critique (1-2 sem)

**Objectif** : zéro vulnérabilité ouverte connue. Bloquant absolu avant tout deploy prod.

### Items détaillés

#### 2.1 — DG-041 TESTING_MODE=0 défaut prod + gate CI
- Vérifier `src/api/auth.py` : `SENTINEL_TESTING_MODE` default = `"0"` (fail-closed)
- Warning explicit au boot si `=1` détecté
- Ajouter gate CI : workflow GitHub Actions qui fail si `SENTINEL_TESTING_MODE=1` dans `.env.production` ou `fly.toml`

#### 2.2 — DG-055 HMAC admin replay nonce-based (F-03)
- Avant : signature TS-only → vulnérabilité cross-route replay 5 min
- Après : signature inclut `(route + body + ts + nonce)`
- Nonce : UUID v4 généré côté client, stocké en Redis/SQLite avec TTL 5 min
- Refuser si nonce déjà vu
- Tests adversariaux : replay même request 2x → 403, cross-route avec même signature → 403

#### 2.3 — DG-056 UNIQUE constraint api_key_id (F-04)
- Avant : 2 users peuvent avoir le même `api_key_id` → account hijack possible
- Migration : `ALTER TABLE users ADD CONSTRAINT uq_api_key_id UNIQUE (api_key_id)`
- Avant migration : dump SQLite, vérifier zéro doublon, fix manuel si nécessaire
- Tests : insertion doublon → IntegrityError

#### 2.4 — DG-057 lecture subscription_expires (F-05)
- Avant : `require_api_key` ne vérifie pas `subscription_expires` → user qui annule garde accès premium
- Après : vérifier `expires_at < now() → downgrade FREE + log
- Tests : fixture user `expires=yesterday` → tier = FREE après auth

#### 2.5 — Audit sécurité complet
- Grep secrets en clair : `grep -rn "sk-ant-\|sk_live_\|whsec_" .` → doit être vide hors `.env.local`
- Vérifier CORS : `Access-Control-Allow-Origin` est config par env, pas `*`
- Rate limiter middleware : 100 req/min per-IP, 1MB body limit
- Sanitize input : signal_id regex `[a-f0-9]{12}`, sanitize_string sur tous les param API
- Error detail leakage : `str(exc)` → "Internal server error" (déjà fait, vérifier)
- LLM circuit check actif
- **Sortie** : rapport `docs/audits/security_pre_launch.md`

#### 2.6 — Sécurité frontend
- CSP (Content Security Policy) headers configurés Next.js
- XSS protection (Next.js le fait nativement, vérifier)
- HTTPS strict (HSTS header)
- Cookies : `HttpOnly`, `Secure`, `SameSite=Strict` sur session
- Bcrypt pour password (si signup avec password)
- 2FA optionnel sur compte (V2 nice-to-have)

### Gate Sprint 2
- [ ] F-03 fermé : tests replay adversariaux passent
- [ ] F-04 fermé : UNIQUE constraint appliqué
- [ ] F-05 fermé : tests expiry passent
- [ ] TESTING_MODE=0 défaut + gate CI active
- [ ] Audit sécurité rapport produit, zéro CRITICAL ouvert
- [ ] CSP + cookies + HTTPS strict frontend

**⚠ Sprint 2 peut être parallélisé avec Sprint 1**.

---

## Sprint 3 — Chatbot pilier solidifié (2-3 sem)

**Objectif** : le moat fonctionne. Chatbot anti-hallucination, contextualisé, robuste.

### Items détaillés

#### 3.1 — Audit DG-110 existant
- Vérifier `/api/chat` (commit `066eff0`) injecte bien le contexte InsightSignalV2 complet
- Logs : confirmer présence dans prompt système des 8 composantes + uncertainty + structure + regime + vol + event + history + breakdown
- **Sortie** : rapport `docs/audits/chatbot_context_audit.md`

#### 3.2 — Finition DG-110 si gaps : context_builder
- Créer / étendre `src/intelligence/chatbot/context_builder.py`
- Fonction `build_context_payload(signal: InsightSignalV2) -> dict` filtrant tous les champs critiques
- Test : prompt généré contient bien les valeurs réelles du signal
- Question "Pourquoi 72 ?" → réponse cite les 8 contributions avec valeurs chiffrées

#### 3.3 — DG-111 chatbot conformal + stats J.*
- Capacité à expliquer intervalle conformel [54-82] en langage humain
- Capacité à expliquer empirical coverage (94.3 % vs nominal 95 %)
- Capacité à comparer PF + IC bootstrap (1.30 [1.12-1.49]) aux setups historiques
- 4 réponses scriptées de référence (`copies/chatbot_scripted_responses.md` Q3, Q4, Q6)

#### 3.4 — DG-112 tests adversariaux refus pédagogique
- Implémenter `src/intelligence/chatbot/refusal_detector.py` (regex + LLM-as-classifier fallback)
- 30+ patterns FR (`tests/adversarial_chatbot_tests.md`)
- 18+ patterns EN
- Cible : Recall ≥ 98 %, FP < 5 %
- 5 templates de refus rotated random
- Refusal inclut contexte du signal (conviction, regime, event)
- Visual tag UI `REFUS PÉDAGOGIQUE · compliance UE 2024/2811`
- Métrique Prometheus `chatbot_refusals_total`

#### 3.5 — DG-114 questions suggérées dynamiques
- Endpoint `/api/v1/chat/suggestions/{signal_id}`
- 3 questions par défaut, dynamiques :
  - Q1 : "Pourquoi la conviction n'est que de {X} ?"
  - Q2 : "C'est quoi {terme_dominant}, en simple ?"
  - Q3 : event ≤ 4h OU historical si conviction ≥ 70 OU uncertainty fallback
- Multilingue FR + EN
- Event analytics `chatbot_question` avec `question_source: suggested|free`

#### 3.6 — DG-042 NARRATIVE_MODE=llm tier-routed
- FREE : `claude-haiku-4-5-20251001` (cheap)
- STARTER : `claude-haiku-4-5-20251001`
- PRO : `claude-sonnet-4-6`
- INSTITUTIONAL : `claude-opus-4-7`
- Logs montrent le bon modèle utilisé par tier
- Fallback template si LLM circuit ouvert

#### 3.7 — Forbidden tokens post-processing
- `contains_forbidden_token()` validation après chaque réponse Claude
- Liste : signal, achetez, vendez, garanti, profit X%, gagnez, recommandation, conseil, opportunité, va monter/descendre + EN equivalents
- Si match : remplacer par fallback safe + logger incident + incrémenter `chatbot_forbidden_blocked_total`

#### 3.8 — Session memory 5-turn
- Stocker 5 derniers échanges par `session_id` (Redis ou SQLite TTL 1h)
- Injecter dans prompt système
- Test : 3 questions consécutives → réponses cohérentes contextuellement

#### 3.9 — Cache sémantique
- Hash bucketé : `(question_normalized, conviction_bucket_5, instrument, direction)`
- TTL 1h
- Hit rate cible 30 %+ après warm-up
- Métriques Prometheus

#### 3.10 — Streaming SSE optimization
- Already done (commit `066eff0`)
- Vérifier : first byte < 500ms, total latency < 3s pour réponse 200 tokens

#### 3.11 — Tests E2E chatbot
- Test 6 questions de référence (`scripted_responses.md`)
- Test prompt injection malicieux → fallback safe
- Test session memory cohérence 3-turn
- Test tier routing (FREE → Haiku, PRO → Sonnet)
- Test rate limiting per tier
- Test cache sémantique hit/miss

### Gate Sprint 3
- [ ] Chatbot répond aux 6 questions types avec contexte complet
- [ ] 30+ tests adversariaux passent (recall ≥ 98 %, FP < 5 %)
- [ ] Tier routing actif avec logs vérifiables
- [ ] Forbidden tokens post-processing bloque 100 % slip-throughs en test
- [ ] Session memory 5-turn fonctionnelle
- [ ] Cache sémantique hit rate ≥ 30 % observable

**Dépendance** : Sprint 1 stable (besoin scoring v2 pour décomposition cohérente).

---

## Sprint 4 — Delivery + Observability (1-2 sem)

**Objectif** : la prod ne plante pas silencieusement, coûts contrôlés, bugs visibles.

### Items détaillés

#### 4.1 — DG-054 Telegram retry + dedup
- Fix python-telegram-bot v20+ : remplacer `send_message` sync coroutine non awaitée par `await`
- Retry exp backoff sur erreur 429 (Telegram rate limit) avec respect `Retry-After` header
- Max 5 retries, jitter random
- Dedup `(chat_id, signal_id)` TTL 1h → un même signal jamais envoyé 2× au même chat
- Tests intégration : ≥ 30 abonnés simultanés sans flood ban
- Métriques : `telegram_messages_sent_total`, `telegram_retries_total`, `telegram_dedup_skipped_total`

#### 4.2 — DG-052 cost monitoring Anthropic
- Prometheus gauge `llm_cost_usd_total` (cumulative)
- Counter `llm_tokens_used_total` par modèle (Haiku/Sonnet/Opus)
- Alerte Discord webhook ou email à seuil journalier (env var `ANTHROPIC_DAILY_COST_ALERT_USD=20`)
- Hard cap fail-closed (env var `ANTHROPIC_MONTHLY_COST_HARD_CAP_USD=500`) : si dépassé, narrative_mode fallback template

#### 4.3 — DG-033 Sentry activé
- DSN configuré via `SENTRY_DSN` env var
- Backend : sentry-sdk Python
- Frontend : sentry-react Next.js
- Source maps uploadés au build
- PII scrubbing actif (email, IP non envoyés)
- Test exception backend → apparaît dans dashboard
- Test exception frontend → apparaît avec source map

#### 4.4 — Circuit breaker thresholds validation
- Analyse 30j logs production (test) :
  - LLM circuit : false-positive rate sur threshold=3
  - Telegram circuit : false-positive rate sur threshold=5
- Ajustement si nécessaire
- Documentation runbook : quand modifier les seuils, qui valide
- Sortie : `docs/runbooks/circuit_breaker_tuning.md`

#### 4.5 — Logs structurés JSON
- Vérifier `LOG_FORMAT=json` en prod
- Tous les logs ont : timestamp, level, module, function, request_id (si applicable)
- Pas de PII dans logs (email, IP scrubbed)
- Rotation logs : 30j retention max

#### 4.6 — Health checks robustes
- `/health` retourne :
  - Backend status (DB connection, Anthropic reachable, Telegram reachable)
  - Scanner running status
  - Signals generated last 24h
  - Cost spend last 24h
  - Cache hit rate
- Endpoint utilisable par Fly.io health check + monitoring externe (UptimeRobot, Pingdom)

#### 4.7 — Backup automatique
- `signals.db` backup daily 02:00 UTC → Cloudflare R2
- Retention 90 jours
- Test restauration mensuelle (manuel ou cron)
- Documentation runbook `docs/runbooks/disaster_recovery.md`

### Gate Sprint 4
- [ ] Telegram async + retry + dedup actif, 0 message perdu sur 30 abonnés
- [ ] Cost monitoring + alerte spend actif
- [ ] Sentry capture backend + frontend exceptions
- [ ] Circuit breaker thresholds validés sur 30j logs
- [ ] Logs structurés JSON sans PII
- [ ] Health check endpoint complet
- [ ] Backup quotidien testé

**Parallélisable avec Sprint 3**.

---

## Sprint 5 — Sources RAG + Track Record (2 sem)

**Objectif** : substancier la promesse "honest confidence" par sources et transparence vérifiable.

### Items détaillés

#### 5.1 — DG-058a 12 papers académiques RAG curés
- Sélection finale 12 papers (cf. `dev_focus_plan_2026_05_27.md`) :
  1. López de Prado (2018) — Advances in Financial Machine Learning
  2. Corsi (2009) — HAR-RV
  3. Gibbs & Candès (2021) — Adaptive Conformal Inference
  4. Barndorff-Nielsen & Shephard (2004) — Bipower Variation
  5. Adams & MacKay (2007) — BOCPD
  6. Angelopoulos & Bates (2024) — Conformal Prediction Gentle Introduction
  7. Avellaneda & Lee (2010) — Statistical Arbitrage
  8. Cont (2001) — Empirical Properties of Asset Returns
  9. Engle (2002) — Dynamic Conditional Correlation
  10. Hasbrouck (2007) — Empirical Market Microstructure
  11. Lo (2004) — Adaptive Markets Hypothesis
  12. Pedersen (2015) — Efficiently Inefficient

- Mini-fiches inline : pour chaque paper, 3 phrases vulgarisées + DOI/arXiv ID + lien
- Composant frontend : `<SourcesList>` dans mode EXPERT
- Validation post-génération : si LLM mentionne une source, elle DOIT être dans cette liste (sinon rejet)
- Test : génération narrative_long mode EXPERT → sources cliquables visibles

#### 5.2 — DG-142 tableau performance public `/track-record`
- Backend :
  - `src/intelligence/track_record/aggregator.py` : `compute_stats(period_days)`
  - `bootstrap_profit_factor_ci(trades, n_iter=1000, alpha=0.05)`
  - `compute_equity_curve(trades)` + `compute_max_drawdown`
  - Endpoint `/api/v1/track-record/?period=1m|3m|6m|all`
  - Pas d'auth (page publique)
- Frontend :
  - Page `/track-record`
  - `<PerformanceHeader>` : stats agrégées (n_trades, win_rate, PF avec IC, DD max, exposure_time)
  - `<EquityChart>` : SVG inline minimal (line chart cumulative R)
  - `<TradesTable>` : trades closed (date, instrument, direction, entry, exit, R-multiple)
  - `<PeriodFilter>` : 1m / 3m / 6m / all
- Disclaimer footer : "Paper-trading uniquement. Performances passées ne préjugent pas des performances futures."

#### 5.3 — Cron nightly stats agrégées
- Job cron J+1 23:59 UTC : recompute stats périodes 1m/3m/6m/all
- Snapshot dans `data/track_record_snapshots/{period}_{YYYY-MM-DD}.json`
- Backup sur Cloudflare R2
- Test cron manuel

#### 5.4 — Bootstrap CI validation
- Test unitaire : 100 trades simulés PF=1.3 → IC bootstrap inclut bien 1.3
- Test unitaire : 4 trades connus (3 wins, 1 loss) → PF=4.0 attendu
- 1000 itérations resample reproductibles (seed fixe pour test)

#### 5.5 — Frontend mode EXPERT
- Vérifier que mode EXPERT (tier PRO+) affiche bien :
  - Waterfall 8 composantes avec contributions chiffrées
  - Conformal viz (bande + point + ticks)
  - Sources RAG cliquables (DG-058a)
  - Stats J.* enrichies (skew, kurtosis, exposure time)
- Mockup référence : `mockups/v3/best_concept_demo.html`

### Gate Sprint 5
- [ ] 12 sources RAG curées + mini-fiches inline visibles mode EXPERT
- [ ] `/track-record` public accessible, stats + equity chart + trades table fonctionnels
- [ ] Cron nightly opérationnel
- [ ] Bootstrap CI tests validés
- [ ] Mode EXPERT complet (waterfall + conformal viz + sources + stats J.*)

**Dépendance** : Sprint 1 stable.

---

## Sprint 6 — Infra deploy + Analytique (1-2 sem)

**Objectif** : prod déployée Fly.io + Vercel, métriques observables, mobile-first finalisé.

### Items détaillés

#### 6.1 — DG-022 Fly.io cdg deploy backend
- `fly.toml` configuré région cdg (Paris)
- Deploy via `fly deploy`
- Test cold-start < 2s
- Test latence Paris→endpoint < 30ms p50 depuis VPN FR
- DNS `api.mia.markets` → Fly.io IP
- HTTPS Let's Encrypt auto
- Monitoring Fly.io metrics

#### 6.2 — Vercel deploy frontend
- `vercel.json` configuré
- Deploy production sur `mia.markets` + `www.mia.markets`
- HTTPS auto
- ISR (Incremental Static Regeneration) pour landing
- Server-side rendering pour lectures

#### 6.3 — DG-029-MODIFIED Fly.io secrets natifs
- Tous les secrets via `fly secrets set` :
  - `ANTHROPIC_API_KEY`
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_PUBLIC_CHANNEL_ID`
  - `STRIPE_API_KEY` (test mode pour l'instant)
  - `STRIPE_WEBHOOK_SECRET`
  - `SENTRY_DSN`
- Vérifier zéro secret en clair dans `fly.toml`, `.env.production`, git history
- Audit `git log --all -p -- .env*` pour vérifier qu'aucun secret n'a fuité historiquement

#### 6.4 — DG-103 mobile-first responsive audit complet
- Test viewports : 375 (iPhone SE), 393 (iPhone 14), 768 (iPad Mini), 1024 (laptop small), 1440 (desktop), 1920 (desktop XL)
- Lighthouse mobile ≥ 95 (au-delà du seuil ≥ 90)
- Touch targets ≥ 44×44px
- FAB chatbot visible mobile < 1024px, sidebar ≥ 1024px
- Aucun scroll horizontal forcé
- Texte minimum 14px body
- Test physique réel iPhone + Android (au moins 1 device de chaque)

#### 6.5 — DG-160 Plausible self-hosted
- Image Docker `plausible/community-edition` sur Fly.io
- Domain `analytics.mia.markets` configuré
- `DISABLE_REGISTRATION=invite_only`
- Backup ClickHouse daily R2
- Aucun cookie tiers posé (vérification DevTools)

#### 6.6 — DG-161 event tracking core 6 events
- Wrapper `frontend/lib/analytics.ts` typé
- Events :
  - `signal_view` (signal_id, instrument, direction, tier_user, surface)
  - `chatbot_question` (question_category, question_source, tier_user, session_id_hash)
  - `section_expanded` (section_id, tier_user, was_locked)
  - `upgrade_clicked` (from_tier, to_tier, cta_location)
  - `signup` (source, campaign)
  - `paid_conversion` (tier, price_id, amount_cents, currency, billing_cycle, trial_converted)
- Aucun PII dans props (vérification test)
- `paid_conversion` server-side via webhook Stripe (pas client-side ad-blockable)
- Dashboard Plausible : 6 events visibles

#### 6.7 — Performance optimization
- Code splitting Next.js (App Router natif)
- Image optimization (`<Image>` Next.js)
- Lazy loading sections collapsibles
- Prefetch hover sur liens internes
- Service worker basique pour offline detection
- Bundle analyzer audit : first-load JS < 150 kB target

#### 6.8 — Browser compatibility test
- Chrome / Firefox / Safari / Edge (versions actuelles)
- iOS Safari, Android Chrome
- Pas de polyfills exotiques requis (cible ES2022)

### Gate Sprint 6
- [ ] Backend Fly.io deployed, `api.mia.markets` opérationnel
- [ ] Frontend Vercel deployed, `mia.markets` opérationnel
- [ ] Secrets via Fly.io secrets (zéro clair)
- [ ] Lighthouse mobile ≥ 95, tests physiques iPhone + Android OK
- [ ] Plausible self-hosted opérationnel
- [ ] 6 events arrivent dans dashboard
- [ ] First-load JS < 150 kB
- [ ] Cross-browser tested

---

## Sprint 7 — UX polish + edge cases (2 sem)

**Objectif** : passer de "fonctionne" à "fonctionne à la perfection". UX polish, edge cases, accessibility.

### Items détaillés

#### 7.1 — Loading states partout
- Chaque action async a un loading state visible
- Skeletons pour sections collapsibles en cours de fetch
- Spinners discrets (≤ 200ms) ou progress bars (> 200ms)
- Pas de "écran blanc" jamais

#### 7.2 — Error states gracieux
- Erreur API → toast informatif + retry option
- Network offline → bandeau "Mode hors ligne, certaines fonctionnalités indisponibles"
- 404 page custom (cohérent design)
- 500 page custom (cohérent design)
- Timeout (>30s) → message clair + suggestion (refresh, contact support)

#### 7.3 — Empty states
- Si aucun signal récent : message + suggestion d'attendre
- Si chatbot 0 question : invitation aux 3 questions suggérées
- Si track-record vide : "Backtest 7 ans en cours d'agrégation"
- Si signup réussi mais pas encore d'activité : welcome onboarding

#### 7.4 — Animations + transitions
- Transitions sections collapsibles : 200ms ease-out
- Animations chatbot messages : slide-in 150ms
- Hover states sur boutons (couleur + élévation subtle)
- Aucune animation > 400ms (sauf onboarding tour)
- `prefers-reduced-motion` respecté (a11y)

#### 7.5 — Accessibility WCAG AA
- Contraste : tous textes ≥ 4.5:1 (gris muted vérifier)
- Navigation clavier complète : Tab order logique, focus visible
- ARIA labels : sections collapsibles, FAB chatbot, modals
- Screen reader friendly : `aria-live` pour notifications dynamiques
- Skip-to-content link
- Form labels associés
- Test avec NVDA / VoiceOver (au moins 1 fois)

#### 7.6 — i18n FR + EN parité totale
- Audit : grep `[MISSING_TRANSLATION]` → doit être vide
- Toutes les copies dans `messages/fr.json` + `messages/en.json`
- Aucun texte hardcodé dans composants
- Dates / nombres formatés selon locale (1,234.56 EN vs 1 234,56 FR)
- Devise formatée selon locale ($29 USD partout, mais peut-être 29 $US en FR)

#### 7.7 — Form validation
- Validation client-side ET server-side
- Messages d'erreur clairs et localisés
- Aucun "Internal server error" remonté au user
- Champs requis marqués clairement

#### 7.8 — Onboarding contextuel discret
- À la première connexion : tour 3 steps optionnel
- Hero card spotlight → chatbot spotlight → upgrade hint
- Skippable, mémorisé (`localStorage` ou user prefs DB)

### Gate Sprint 7
- [ ] Loading states présents partout
- [ ] Error states gracieux (404, 500, network)
- [ ] Empty states informatifs
- [ ] Animations fluides 60fps
- [ ] WCAG AA validé (contraste + clavier + ARIA)
- [ ] i18n FR + EN parité 100 %
- [ ] Form validation server + client
- [ ] Onboarding 3-step fonctionnel

**Dépendance** : Sprints 1-6 stables.

---

## Sprint 8 — Tests E2E + Documentation interne (1-2 sem)

**Objectif** : tests end-to-end complets, documentation maintenable.

### Items détaillés

#### 8.1 — Tests E2E Playwright
- Suite Playwright couvrant les parcours critiques :
  - Visiteur → landing → /pricing → /signup FREE → première lecture
  - User FREE → consultation signal → chatbot 3 questions → upgrade clicked
  - User PRO → mode EXPERT → waterfall + conformal viz + sources cliquables
  - User STARTER → trial 14j expiration → revert FREE auto
  - Test mobile viewport iPhone SE
  - Test responsive tablet iPad Mini
- Tests passent en CI sur Chrome + Firefox + Safari (Playwright)
- Screenshots regression : snapshots diff
- Couverture parcours : 100 % parcours critiques

#### 8.2 — Tests load (performance under load)
- Simuler 100 users concurrents : `signal_view` + `chatbot_question`
- p99 latency < 2s
- Aucune erreur 5xx
- Cost Anthropic projetable avec confiance

#### 8.3 — Documentation interne
- `docs/architecture/` :
  - `system_overview.md` : diagramme pipeline 7 étages
  - `data_flow.md` : flux InsightSignalV2 de DataProvider à Telegram
  - `chatbot_architecture.md` : context_builder + refusal_detector + cache
  - `scoring_pipeline.md` : LGB → Isotonic → ACI
- `docs/runbooks/` :
  - `disaster_recovery.md` : restauration backup, rollback
  - `circuit_breaker_tuning.md` (Sprint 4)
  - `incident_response.md` (déjà créé `legal_templates/`)
  - `deploy.md` : `fly deploy` procedure
  - `monitoring.md` : dashboards Plausible + Sentry + Fly metrics
- `docs/adr/` (Architecture Decision Records) :
  - ADR-001 : choix architecture progressive uniforme (vs toggle 3 modes)
  - ADR-002 : choix LGBM + Isotonic + ACI (vs autres)
  - ADR-003 : choix Fly.io + Vercel (vs Railway/AWS)
  - ADR-004 : choix Plausible self-hosted (vs GA4)
  - ADR-005 : choix bilingue FR+EN dès J1 (vs FR-first)

#### 8.4 — README projet
- `README.md` root : vision + setup dev + tests + deploy
- `webapp/README.md` : setup Next.js, env vars frontend
- `backend/README.md` (ou root) : setup Python, run scanner

#### 8.5 — Onboarding développeur < 1h
- Tester : si tu reviens dans 6 mois, peux-tu re-build le projet from scratch en < 1h ?
- `git clone` → install deps → setup `.env.local` → `npm run dev` + `python -m src.intelligence.main` → premier signal généré
- Critère : OUI sans demander aide

#### 8.6 — Changelog
- `CHANGELOG.md` à la racine
- Format Keep a Changelog
- Versions sémantiques (`2.1.0` = InsightSignalV2 v2.1.0)
- Entrées datées avec features / fixes / breaking changes

#### 8.7 — Coverage finale
- Backend : ≥ 85 % zones revenue (auth, quota_manager, chatbot, webhooks Stripe quand activé)
- Backend : ≥ 70 % autres
- Frontend : ≥ 75 % composants critiques (LectureView, ChatbotPanel)
- Frontend : ≥ 50 % autres
- Couverture mesurée + visualisée (Codecov ou équivalent)

### Gate Sprint 8
- [ ] Tests E2E Playwright : 100 % parcours critiques passent
- [ ] Tests load : p99 < 2s sur 100 users concurrents
- [ ] Documentation architecture complète
- [ ] Runbooks ops complets
- [ ] ADRs documentés (5+ décisions clés)
- [ ] README projet permet onboarding < 1h
- [ ] Coverage final : 85 % revenue, 70 % autres backend, 75 % composants critiques front

---

## ═══ GATE FINAL — "Fonctionnel à la perfection" ═══

Quand ces 5 piliers sont tous validés simultanément, le produit est **launching-ready**.

### Pilier A — Solidité technique
- [ ] Scoring v2 prod avec Brier skill ≥ +5 %
- [ ] Sweep state machine 432 cellules livré
- [ ] 4 vulnérabilités sécurité fermées (F-03/04/05 + TESTING_MODE gate CI)
- [ ] Couverture tests ≥ 85 % zones revenue

### Pilier B — Robustesse opérationnelle
- [ ] Telegram retry + dedup, 0 message perdu sur 30 abonnés
- [ ] Cost monitoring Anthropic + alerte spend + hard cap fail-closed
- [ ] Sentry backend + frontend opérationnel
- [ ] Circuit breakers calibrés sur 30j logs
- [ ] Backup quotidien testé restaurabilité

### Pilier C — Qualité UX
- [ ] Lighthouse mobile ≥ 95
- [ ] WCAG AA validé
- [ ] i18n FR + EN parité 100 %
- [ ] Loading + error + empty states partout
- [ ] First-load JS < 150 kB

### Pilier D — Cohérence produit
- [ ] Chatbot répond à 30+ questions sans hallucination
- [ ] 30+ tests adversariaux refus pédagogique passent
- [ ] Aucun vocabulaire interdit nulle part (audit final)
- [ ] 12 sources RAG cliquables mode EXPERT
- [ ] Track-record public quotidien avec IC bootstrap

### Pilier E — Maintenabilité
- [ ] Documentation architecture complète
- [ ] Runbooks ops + ADRs
- [ ] Tests E2E Playwright couvrant parcours critiques
- [ ] Onboarding développeur < 1h depuis README

**Si les 5 piliers cochés → tu décides quand passer au commercial.** À ce moment-là, je réactive le plan commercial.

---

# 📊 Effort total Pre-launch

| Sprint | Items | Effort | Cumul |
|---|---|---|---|
| 1 — Cœur algorithmique | DG-025 + DG-034 + DG-004 + DG-053 + tests | 47-69h | 3-4 sem |
| 2 — Sécurité critique | DG-041 + DG-055 + DG-056 + DG-057 + audit + frontend sécu | 17-25h | 1-2 sem |
| 3 — Chatbot pilier | DG-110 + DG-111 + DG-112 + DG-114 + DG-042 + memory + cache + tests E2E | 50-75h | 2-3 sem |
| 4 — Delivery + obs | DG-054 + DG-052 + DG-033 + CB + logs + health + backup | 22-32h | 1-2 sem |
| 5 — Sources + Track Record | DG-058a + DG-142 + cron + bootstrap CI + mode EXPERT | 40-55h | 2 sem |
| 6 — Infra + analytique | DG-022 + DG-029 + DG-103 + DG-160 + DG-161 + perf + cross-browser | 35-50h | 1-2 sem |
| **7 — UX polish + edge cases** | Loading + error + empty + animations + WCAG + i18n + onboarding | 40-55h | 2 sem |
| **8 — Tests E2E + doc interne** | Playwright + load + architecture + runbooks + ADRs + README + changelog | 30-45h | 1-2 sem |
| **TOTAL** | **8 sprints** | **~280-405h** | **~14-18 sem** |

**Effort estimé honnête** : ~3,5 à 4,5 mois de dev focus pour atteindre "fonctionnel à la perfection".

C'est ~6-8 semaines de plus que "performance-ready" (10-12 sem). L'écart vient des sprints 7-8 (polish UX + tests E2E + documentation) qui sont la différence entre "ça marche" et "ça marche à la perfection".

---

# 🚀 Brief autre instance Claude Code

```
Mission : produit M.I.A. Markets "fonctionnel à la perfection" avant tout commercial.

Lis docs/governance/MASTER_PLAN.md puis docs/governance/vague1_execution/briefs/
pour les détails ready-to-code.

Exécute les 8 sprints dans l'ordre (sprints parallélisables marqués).

Mets à jour docs/governance/vague1_execution/PROGRESS.md à chaque fin de sprint
avec items livrés, items reportés, métriques gate validées.

Ignore TOUT item commercial :
- Stripe live, pricing UI, page pricing (DG-132)
- RC Pro, Iubenda, avocat
- Marketing copies actives, onboarding production
- Email automation cycle de vie
- Geo-block prod activation (peut rester en mode test pendant dev)
- Track-record Telegram public ouverture
- Email lifecycle automation

Tous ces items sont "post-launching" et seront réactivés quand le user décide.

Ouvre PR "feat: pre-launch perfection gate validated" quand les 5 piliers
(A solidité, B robustesse, C UX, D cohérence, E maintenabilité) sont tous cochés.

À ce moment-là, l'utilisateur reprendra la main pour décider du timing
et du séquençage commercial.
```

---

# 📝 Ce qui est HORS du plan pre-launch

Tous ces items existent dans `decision_gate_review_v2.md` et `vague1_execution/`, mais sont **mis en pause**. Ils seront re-planifiés au moment décidé par le user :

- Stripe live activation (DG-043)
- Stripe Tax + reverse charge (DG-044)
- Page pricing UI complète (DG-132)
- Hard caps signaux/mois Stripe (DG-046) — peut être actif pour bootstrap dev test
- Geo-block prod activation (DG-045)
- Tier rate-limit enforce (DG-006)
- DSAR endpoints (DG-038) — sera fait avec V0 légal post-launch
- Avocat fintech canadien
- Statut entreprise REQ Québec
- Domain `mia.markets` achat
- Email pro ProtonMail
- RC Pro Hiscox/Intact
- Iubenda Pro souscription
- Marketing copies actives, témoignages, cas d'usage
- Onboarding tour avec capture analytique
- Email automation cycle de vie
- Trading Economics souscription
- Telegram public channel ouverture
- Pédagogie EXPERT (DG-170-173) — sera fait en partie Sprint 5 mais polish post-launch
- B2B, INSTITUTIONAL tier
- Médiation conso adhésion

**Tous ces items sont "ready-to-do" mais déclenchés post-validation Gate Final.**

---

# 🔗 Documents de référence

| Document | Statut | Rôle |
|---|---|---|
| **`MASTER_PLAN.md`** (ce fichier) | ✅ ACTIF | Source unique pre-launch |
| `decision_gate_review_v2.md` | 📚 référence | Audit 69 items DG |
| `vague1_execution/00_BATTLE_PLAN.md` | 📚 référence | Séquencement initial (obsolète) |
| `dev_focus_plan_2026_05_27.md` | 📚 référence | Pivot focus dev (obsolète, remplacé par MASTER_PLAN) |
| `decisions/2026-05-26_5_political_locks.md` | 📚 ref | Décisions politiques signées |
| `vague1_execution/briefs/` | ✅ ACTIF | 10 briefs ready-to-code détaillés |
| `vague1_execution/configurations/` | ✅ ACTIF | Configs prod (env vars, geo-block, tier quotas) |
| `vague1_execution/copies/` | ✅ ACTIF | Copies finales (chatbot prompt, scripted responses) |
| `vague1_execution/tests/` | ✅ ACTIF | Acceptance criteria + tests adversariaux |
| `mockups/v3/best_concept_demo.html` | ✅ ACTIF | Mockup référence architecture progressive uniforme |

---

# ⏱️ Timeline réaliste

```
M0 (aujourd'hui, 2026-05-27)
  ╠ Sprint 1 démarre (cœur algo)
  ╠ Sprint 2 démarre parallèle (sécu)

M1
  ╠ Sprint 1 + 2 stables
  ╠ Sprint 3 démarre (chatbot)
  ╠ Sprint 4 démarre parallèle (delivery+obs)

M2
  ╠ Sprint 3 + 4 stables
  ╠ Sprint 5 démarre (sources+track-record)

M3
  ╠ Sprint 5 stable
  ╠ Sprint 6 démarre (infra+analytique)

M4
  ╠ Sprint 6 stable
  ╠ Sprint 7 démarre (UX polish)

M4 fin
  ╠ Sprint 7 stable
  ╠ Sprint 8 démarre (tests E2E + doc)

M4-M5 frontière
  ╠ GATE FINAL "fonctionnel à la perfection" validée
  ╠ Décision utilisateur sur timing launching commercial
```

---

# ✅ Action immédiate

**Aujourd'hui** :

1. **Toi** : valide ce MASTER_PLAN (réponds "GO") et brief l'autre instance Claude Code avec le brief ci-dessus.

2. **Autre instance Claude Code** : démarre Sprint 1 (audit DG-025 + finition modèle si gap + DG-034 sweep + DG-004 archive + DG-053 boot fail-fast).

3. **Cette instance (moi)** : passe en mode "support à la demande". Je redeviens actif si :
   - Tu veux ajuster un sprint
   - Tu hésites sur une décision technique
   - Gate Final approche → préparer le plan commercial post-launching
   - Sinon, je laisse le code avancer.

**Plus de planification.** Tout est ici. Reste à exécuter.

---

**FIN MASTER_PLAN.** Pre-launch focus pur. Le commercial sera ré-injecté quand tu décides.
