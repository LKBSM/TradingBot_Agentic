# Les 5 Piliers de la Perfection — Checklist détaillée

**Statut** : ✅ Document de référence (compagnon du `MASTER_PLAN.md`)
**Audience** : autre instance Claude Code (en cours d'exécution Phase 1) + utilisateur
**Usage** : checklist exhaustive de validation des 5 piliers du Gate Final "fonctionnel à la perfection"

**Cohabitation** : ce document ne change PAS le séquencement des sprints actuels. Il fournit une vue **thématique par pilier** complémentaire à la vue **chronologique par sprint** du MASTER_PLAN. Les deux convergent vers le même Gate Final.

---

## 📋 Vue d'ensemble

| Pilier | Sous-critères | Sprints concernés MASTER_PLAN |
|---|---|---|
| **A — Solidité technique** | 6 sous-critères | Sprints 1 + 2 |
| **B — Robustesse opérationnelle** | 7 sous-critères | Sprint 4 |
| **C — Qualité UX** | 8 sous-critères | Sprints 6 + 7 |
| **D — Cohérence produit** | 7 sous-critères | Sprints 3 + 5 |
| **E — Maintenabilité** | 7 sous-critères | Sprint 8 |
| **TOTAL** | **35 sous-critères** | — |

Quand les 35 sous-critères sont cochés → Gate Final validé.

---

# 🛡️ PILIER A — Solidité technique

> Le code marche, calculé correctement, sécurisé. Zéro bug critique ouvert connu.

## A.1 — Scoring v2 calibré empiriquement

**Sous-critères** :
- [ ] `models/calibrated_conviction_v1.pkl` présent et chargé en prod
- [ ] Pipeline LGB → Isotonic → ACI utilisé par `ConfluenceDetector` (vérifiable via logs)
- [ ] Modèle entraîné en walk-forward CPCV (López de Prado 2018) sur XAU 7 ans (2019-2025) **avec coverage > 95 %**
- [ ] Hyperparams documentés (n_trees=200, num_leaves=31, lr=0.05, early stopping)
- [ ] Isotonic recalibration sur out-of-fold predictions
- [ ] ACI wrapper (Gibbs & Candès 2021) avec γ=0.05, buffer_size=500
- [ ] Feature flag `SCORING_VERSION=v2` actif en prod, fallback v1 si modèle absent

**Validation empirique** :
- [ ] Brier skill score ≥ +5 % vs baseline naïf
- [ ] DM test (Diebold-Mariano) p < 0.05
- [ ] Reliability diagram visualisable (calibration OK)
- [ ] Coverage conformel observée ≥ 90 % out-of-sample
- [ ] Rapport `reports/scoring_v2_validation.md` avec graphs

**Comment valider** :
```bash
# Test pipeline en local
pytest tests/test_calibrated_conviction.py -v
# Vérifier modèle chargé en prod
curl https://api.mia.markets/health | jq '.scoring_version'
# Vérifier Brier validation
cat reports/scoring_v2_validation.md
```

## A.2 — Sweep state machine 432 cellules

**Sous-critères** :
- [ ] Sweep `enter × exit × confirm × cooldown × max_age × silent` (6×4×3×3×2×2 = 432) exécuté
- [ ] Sur 7 ans XAU + EUR
- [ ] Métrique : PF moyenne × IC 95 % × n_trades par cellule
- [ ] Choix Pareto-optimal documenté
- [ ] Defaults `src/intelligence/signal_state_machine.py` mis à jour empiriquement
- [ ] Tests régression sur defaults

**Validation** :
- [ ] Rapport `reports/state_machine_sweep_432.md` avec tableau Pareto

## A.3 — Data quality fail-fast

**Sous-critères** :
- [ ] Feed `XAU_15MIN_2019_2025.csv` (63 %) archivé dans `data/_archived/` avec README
- [ ] `verify_data_quality` au boot vérifie : coverage ≥ 95 %, no gaps > 1h, no doublons timestamp, bid-ask cohérent
- [ ] Scanner refuse de booter si feed corrompu (exit code 1)
- [ ] Env var override `STRICT_DATA_QUALITY=false` (urgence uniquement, log WARNING)
- [ ] Test boot avec feed corrompu → fail attendu

**Validation** :
```bash
# Forcer feed corrompu
mv data/XAU_15MIN_2019_2024.csv data/.bak && cp data/_archived/XAU_15MIN_2019_2025_FY63_CORRUPTED.csv data/XAU_15MIN_2019_2024.csv
python -m src.intelligence.main  # doit exit code 1 avec log ERROR
# Restaurer
mv data/.bak data/XAU_15MIN_2019_2024.csv
```

## A.4 — Sécurité critique : 4 vulnérabilités fermées

**F-03 HMAC admin replay nonce-based** :
- [ ] Signature inclut `(route + body + ts + nonce)`
- [ ] Nonce UUID v4 stocké TTL 5 min (rejet si déjà vu)
- [ ] Tests adversariaux : replay 2× même request → 403
- [ ] Tests adversariaux : cross-route même signature → 403

**F-04 UNIQUE constraint api_key_id** :
- [ ] Migration `ALTER TABLE users ADD CONSTRAINT uq_api_key_id UNIQUE` appliquée
- [ ] Dump SQLite vérifié zéro doublon préexistant
- [ ] Test insertion doublon → IntegrityError

**F-05 subscription_expires lu dans require_api_key** :
- [ ] Vérification `expires_at < now() → downgrade FREE + log`
- [ ] Fixture user `expires=yesterday` → tier=FREE après auth

**TESTING_MODE défaut prod** :
- [ ] `SENTINEL_TESTING_MODE` default = `"0"` (fail-closed)
- [ ] Warning au boot si `=1`
- [ ] Gate CI fail si `=1` détecté dans `.env.production` ou `fly.toml`

**Validation** :
```bash
pytest tests/test_security_adversarial.py -v
# Doit passer : test_hmac_replay_rejected, test_api_key_unique, test_expired_subscription_downgrade, test_testing_mode_off_in_prod
```

## A.5 — Audit sécurité complet

**Sous-critères** :
- [ ] Grep secrets en clair : `grep -rn "sk-ant-\|sk_live_\|whsec_" .` → vide hors `.env.local` et `git history`
- [ ] CORS : `Access-Control-Allow-Origin` configuré par env, pas `*`
- [ ] Rate limiter middleware : 100 req/min per-IP, 1 MB body limit
- [ ] Input sanitization : signal_id regex `[a-f0-9]{12}`, sanitize_string sur params API
- [ ] Error detail leakage : `str(exc)` → "Internal server error"
- [ ] LLM circuit check actif sur endpoints chatbot
- [ ] CSP (Content Security Policy) headers Next.js configurés
- [ ] HSTS header strict
- [ ] Cookies : `HttpOnly`, `Secure`, `SameSite=Strict` sur session
- [ ] Bcrypt pour password (si signup avec password)

**Validation** :
- [ ] Rapport `docs/audits/security_pre_launch.md` complet
- [ ] Zéro CRITICAL/HIGH ouvert

## A.6 — Tests couverture ≥ 85 % zones revenue

**Sous-critères** :
- [ ] Backend `src/intelligence/scoring/` : ≥ 85 %
- [ ] Backend `src/api/auth.py` : ≥ 85 %
- [ ] Backend `src/api/quota_manager.py` : ≥ 85 %
- [ ] Backend `src/intelligence/chatbot/` : ≥ 85 %
- [ ] Backend `src/api/routes/webhooks/` : ≥ 85 %
- [ ] Backend autres : ≥ 70 %
- [ ] Frontend composants critiques (LectureView, ChatbotPanel) : ≥ 75 %
- [ ] Frontend autres : ≥ 50 %

**Validation** :
```bash
# Backend
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Frontend
cd webapp && npm run test:coverage
```

---

# 🛡️ PILIER B — Robustesse opérationnelle

> La prod ne plante pas silencieusement, les coûts sont sous contrôle, les bugs sont visibles.

## B.1 — Telegram delivery zéro perte

**Sous-critères** :
- [ ] `python-telegram-bot v20+` async correctement awaité (plus de sync coroutine perdue)
- [ ] Retry exp backoff sur erreur 429 avec respect `Retry-After` header
- [ ] Max 5 retries avec jitter random
- [ ] Dedup `(chat_id, signal_id)` TTL 1h (un signal jamais envoyé 2× au même chat)
- [ ] Métriques Prometheus : `telegram_messages_sent_total`, `telegram_retries_total`, `telegram_dedup_skipped_total`

**Validation** :
- [ ] Test intégration ≥ 30 abonnés simultanés sans flood ban
- [ ] Test simulation 429 → retry réussi
- [ ] Test envoi 2× même signal → 1 seul envoi effectif

## B.2 — Cost monitoring Anthropic

**Sous-critères** :
- [ ] Prometheus gauge `llm_cost_usd_total` cumulative
- [ ] Counter `llm_tokens_used_total` par modèle (Haiku/Sonnet/Opus)
- [ ] Alerte Discord webhook OU email à seuil journalier (`ANTHROPIC_DAILY_COST_ALERT_USD=20`)
- [ ] Hard cap fail-closed (`ANTHROPIC_MONTHLY_COST_HARD_CAP_USD=500`)
- [ ] Si hard cap dépassé : `NARRATIVE_MODE` fallback `template`, alerte critique envoyée

**Validation** :
- [ ] Test seuil journalier déclenche alerte
- [ ] Test hard cap déclenche fallback + alerte critique

## B.3 — Sentry backend + frontend

**Sous-critères** :
- [ ] DSN configuré via `SENTRY_DSN` env var
- [ ] Backend sentry-sdk Python avec breadcrumbs
- [ ] Frontend sentry-react Next.js
- [ ] Source maps uploadés au build (production builds)
- [ ] PII scrubbing actif (email, IP non envoyés)
- [ ] Release tags configurés (corrélation commits)

**Validation** :
- [ ] Test exception backend `raise Exception("test")` → apparaît dashboard Sentry
- [ ] Test exception frontend → apparaît avec source map résolvable

## B.4 — Circuit breakers calibrés

**Sous-critères** :
- [ ] LLM circuit : threshold=3, timeout=60s validé sur 30j logs réels
- [ ] Telegram circuit : threshold=5, timeout=120s validé sur 30j logs réels
- [ ] False-positive rate analyse documentée
- [ ] Ajustement si FP > 5 %
- [ ] Runbook `docs/runbooks/circuit_breaker_tuning.md` créé

## B.5 — Logs structurés JSON

**Sous-critères** :
- [ ] `LOG_FORMAT=json` en prod
- [ ] Tous les logs ont : `timestamp, level, module, function, request_id`
- [ ] Pas de PII dans logs (email/IP scrubbed)
- [ ] Rotation logs : 30j retention max
- [ ] Logs accessibles via Fly.io logs

**Validation** :
```bash
fly logs | head -20 | jq '.'  # doit être JSON valide
fly logs | grep -i "email\|password" | grep -v "***"  # doit être vide
```

## B.6 — Health checks complets

**Sous-critères** :
- [ ] Endpoint `/health` retourne :
  - Backend status (DB connection, Anthropic reachable, Telegram reachable)
  - Scanner running status
  - Signals generated last 24h
  - Cost spend last 24h
  - Cache hit rate
  - Circuit breakers states
- [ ] Endpoint utilisable par Fly.io health check natif
- [ ] Endpoint utilisable par monitoring externe (UptimeRobot, Pingdom)
- [ ] Réponse < 200ms p95

**Validation** :
```bash
curl https://api.mia.markets/health | jq '.'
# Tous les status doivent être "healthy"
```

## B.7 — Backup automatique

**Sous-critères** :
- [ ] `signals.db` backup daily 02:00 UTC → Cloudflare R2
- [ ] `users.db` backup daily
- [ ] Retention 90 jours
- [ ] Test restauration mensuelle (manuel ou cron)
- [ ] Runbook `docs/runbooks/disaster_recovery.md`

**Validation** :
- [ ] Backup test réussi (download depuis R2, restauration sur env staging)
- [ ] Documentation procédure restauration

---

# 🎨 PILIER C — Qualité UX

> L'expérience utilisateur est polish, mobile-first, accessible, performante.

## C.1 — Mobile-first responsive

**Sous-critères** :
- [ ] Test viewports : 375 (iPhone SE), 393 (iPhone 14), 768 (iPad Mini), 1024 (laptop), 1440 (desktop), 1920 (XL)
- [ ] Lighthouse mobile ≥ 95 (au-delà du seuil ≥ 90 standard)
- [ ] Touch targets ≥ 44×44px (Apple HIG / Material)
- [ ] FAB chatbot < 1024px, sidebar ≥ 1024px
- [ ] Aucun scroll horizontal forcé
- [ ] Texte minimum 14px body (12px footer accepté)
- [ ] Test physique iPhone + Android (au moins 1 device chaque)

**Validation** :
```bash
npm run lighthouse -- --preset=mobile
# Score ≥ 95
```

## C.2 — Accessibility WCAG AA

**Sous-critères** :
- [ ] Contraste : tous textes ≥ 4.5:1 (gris muted vérifier)
- [ ] Contraste large text (≥ 18px) ≥ 3:1
- [ ] Navigation clavier complète : Tab order logique, focus visible
- [ ] ARIA labels : sections collapsibles, FAB chatbot, modals
- [ ] `aria-live` pour notifications dynamiques
- [ ] `aria-expanded`, `aria-disabled` corrects
- [ ] Skip-to-content link
- [ ] Form labels associés (label `for` ↔ input `id`)
- [ ] Test screen reader : NVDA (Windows) OU VoiceOver (Mac/iOS)
- [ ] `prefers-reduced-motion` respecté

**Validation** :
```bash
# axe-core CI
npm run test:a11y
# Manuel : ouvrir avec NVDA / VoiceOver
```

## C.3 — Loading + error + empty states

**Loading** :
- [ ] Chaque action async a un loading state visible
- [ ] Skeletons pour sections collapsibles en fetch
- [ ] Spinners ≤ 200ms, progress bars > 200ms
- [ ] Jamais d'écran blanc

**Error** :
- [ ] Erreur API → toast informatif + retry option
- [ ] Network offline → bandeau "Mode hors ligne"
- [ ] 404 page custom cohérent design
- [ ] 500 page custom cohérent design
- [ ] Timeout > 30s → message clair + suggestion

**Empty** :
- [ ] Aucun signal récent → message + suggestion
- [ ] Chatbot 0 question → invitation aux 3 questions suggérées
- [ ] Track-record vide → "Backtest en cours d'agrégation"
- [ ] Signup réussi sans activité → welcome onboarding

**Validation** :
- [ ] Tests E2E couvrent tous les états (loading + error + empty + happy path)

## C.4 — Animations fluides 60fps

**Sous-critères** :
- [ ] Transitions sections collapsibles : 200ms ease-out
- [ ] Animations chatbot messages : slide-in 150ms
- [ ] Hover states sur boutons (couleur + élévation subtle)
- [ ] Aucune animation > 400ms (sauf onboarding tour)
- [ ] `prefers-reduced-motion` respecté
- [ ] Pas de jank visible (60fps maintenu)

**Validation** :
- [ ] Chrome DevTools Performance tab : 60fps lors interactions
- [ ] Pas de "long tasks" > 50ms

## C.5 — i18n FR + EN parité 100 %

**Sous-critères** :
- [ ] Toutes les copies dans `webapp/messages/fr.json` + `webapp/messages/en.json`
- [ ] Aucun texte hardcodé dans composants
- [ ] Grep `[MISSING_TRANSLATION]` → vide
- [ ] Dates formatées selon locale (1,234.56 EN vs 1 234,56 FR)
- [ ] Devises formatées selon locale ($29 USD universel, $29 US en FR)
- [ ] Switching langue préservé en session
- [ ] `next-intl` configuré pour SSR

**Validation** :
```bash
cd webapp
npm run i18n:audit  # script à créer : compare clés FR vs EN
grep -r "MISSING_TRANSLATION" app/ components/
```

## C.6 — Form validation client + server

**Sous-critères** :
- [ ] Validation client-side (zod ou yup)
- [ ] Validation server-side (Pydantic backend)
- [ ] Messages d'erreur clairs et localisés
- [ ] Aucun "Internal server error" remonté
- [ ] Champs requis marqués clairement (étoile *)

## C.7 — Performance optimization

**Sous-critères** :
- [ ] Code splitting Next.js App Router actif
- [ ] Image optimization (`<Image>` Next.js partout)
- [ ] Lazy loading sections collapsibles (suspense)
- [ ] Prefetch hover sur liens internes
- [ ] Service worker basique pour offline detection
- [ ] First-load JS < 150 kB (target stricte)
- [ ] LCP < 2.5s, CLS < 0.1, INP < 200ms (Core Web Vitals)

**Validation** :
```bash
npm run analyze  # @next/bundle-analyzer
# /[locale] first-load JS doit être < 150 kB
```

## C.8 — Cross-browser tested

**Sous-critères** :
- [ ] Chrome (latest 2 versions)
- [ ] Firefox (latest 2 versions)
- [ ] Safari (latest 2 versions)
- [ ] Edge (latest 2 versions)
- [ ] iOS Safari (iOS 15+)
- [ ] Android Chrome (latest)
- [ ] Aucun polyfill exotique requis (cible ES2022)

**Validation** :
- [ ] Playwright tests passent sur Chromium + Firefox + WebKit
- [ ] Test manuel iPhone + Android device réel

---

# 🎯 PILIER D — Cohérence produit

> Le produit est cohérent avec sa vision : compréhension augmentée, jamais promesse de profit.

## D.1 — Chatbot anti-hallucination

**Sous-critères** :
- [ ] Contexte InsightSignalV2 complet injecté dans prompt (8 composantes + uncertainty + structure + regime + vol + event + history + breakdown)
- [ ] `context_builder.py` sérialise tous les champs critiques
- [ ] Système prompt enforce "tu n'inventes pas de chiffres"
- [ ] Test : question "Pourquoi 72 ?" → réponse cite les 8 contributions chiffrées exactes du signal en cours
- [ ] Test prompt injection malicieux → réponse cohérente, pas de hallucination chiffres

**Validation** :
- [ ] `tests/test_chatbot_no_hallucination.py` : 20 questions de validation, aucun chiffre inventé

## D.2 — Refus pédagogique scripté

**Sous-critères** :
- [ ] `refusal_detector.py` regex + LLM-as-classifier fallback
- [ ] 30+ patterns FR couverts (cf. `vague1_execution/tests/adversarial_chatbot_tests.md`)
- [ ] 18+ patterns EN couverts
- [ ] Recall ≥ 98 %, FP < 5 %
- [ ] 5 templates de refus rotated random
- [ ] Refus inclut contexte du signal (conviction, regime, event imminent)
- [ ] Tag UI visible `REFUS PÉDAGOGIQUE · compliance UE 2024/2811`
- [ ] Métrique Prometheus `chatbot_refusals_total`

**Validation** :
```bash
pytest tests/test_chatbot_adversarial.py -v
# Recall ≥ 98 %, FP < 5 %
```

## D.3 — Vocabulaire interdit zéro fuite

**Liste interdite FR** : signal de trading, acheter, vendre, garanti, profit X%, gagnez, recommandation, conseil, opportunité, va monter/descendre, stop-loss prescriptif, take-profit prescriptif

**Liste interdite EN** : trading signal, buy, sell, guaranteed, X% profit, earn, recommendation, advice, opportunity, will go up/down

**Sous-critères** :
- [ ] Audit grep complet du frontend → zéro occurrence
- [ ] Audit grep du backend → zéro occurrence (sauf dans liste de mots interdits eux-mêmes)
- [ ] `contains_forbidden_token()` validation post-LLM
- [ ] Si match : remplacer par fallback safe + log incident + incrémenter `chatbot_forbidden_blocked_total`

**Validation** :
```bash
# Audit complet
grep -rn "signal de trading\|achetez\|vendez\|garanti" webapp/ src/ docs/value/
# Doit être vide ou seulement dans contextes "interdit"
```

## D.4 — Sources RAG 12 papers cliquables

**Sous-critères** :
- [ ] 12 papers académiques sélectionnés (López de Prado, Corsi, Gibbs & Candès, Barndorff-Nielsen, Adams & MacKay, Angelopoulos & Bates, Avellaneda, Cont, Engle, Hasbrouck, Lo, Pedersen)
- [ ] Mini-fiches inline : pour chaque, 3 phrases vulgarisées + DOI/arXiv ID + lien
- [ ] Composant frontend `<SourcesList>` dans mode EXPERT
- [ ] Validation post-génération : si LLM mentionne une source, elle DOIT être dans cette liste

**Validation** :
- [ ] Test E2E : ouvrir lecture en tier PRO → mode EXPERT → 12 sources visibles cliquables

## D.5 — Track-record public quotidien

**Sous-critères** :
- [ ] `/track-record` accessible sans auth
- [ ] Stats agrégées : n_trades, win_rate, PF avec IC bootstrap, DD max, exposure_time
- [ ] Equity chart SVG inline minimal
- [ ] Tableau trades closed (date, instrument, direction, R-multiple)
- [ ] Filtre période 1m/3m/6m/all
- [ ] Cron nightly J+1 23:59 UTC met à jour
- [ ] Disclaimer "Paper-trading uniquement" visible
- [ ] IC bootstrap 1000 itérations reproductible (seed fixe pour test)
- [ ] Snapshots dans `data/track_record_snapshots/` + R2 backup

**Validation** :
- [ ] Visite `https://mia.markets/track-record` sans auth → page chargée < 2s
- [ ] Test cron manuel → snapshot créé

## D.6 — Architecture progressive uniforme

**Sous-critères** :
- [ ] Layout unique responsive (PAS de toggle 3 modes)
- [ ] Hero card permanent (visible toujours, non collapsible)
- [ ] 4-5 sections collapsibles dépliables au clic
- [ ] Gating tier = disponibilité contenu, PAS layout
- [ ] Section "Détail technique" verrouillée 🔒 STRATEGIST pour FREE/STARTER
- [ ] Bouton "Tout déplier" pour STRATEGIST+
- [ ] Cohérent avec mockup `mockups/v3/best_concept_demo.html`

**Validation** :
- [ ] Test E2E : user FREE voit hero + sections collapsed + lock sur Détail technique
- [ ] Test E2E : user PRO peut "Tout déplier"

## D.7 — Posture éducative + compliance UE 2024/2811

**Sous-critères** :
- [ ] `edge_claim = false` en prod
- [ ] Disclaimer "Paper-trading · éducatif" footer permanent
- [ ] Page méthodologie publie l'algorithme + sources + limites
- [ ] `is_paper_demo = true` partout tant que pas validé live forward
- [ ] Aucun claim de gain ni de performance future
- [ ] Wording "lecture" / "analyse" / "setup" / "calibré" — pas "signal" / "recommandation"

**Validation** :
- [ ] Audit final wording : aucun vocabulaire interdit nulle part
- [ ] Page `/methodologie` accessible et complète

---

# 🔧 PILIER E — Maintenabilité

> Le code est compréhensible, documenté, testable, maintenable par toi dans 6 mois.

## E.1 — Documentation architecture

**Sous-critères** :
- [ ] `docs/architecture/system_overview.md` : diagramme pipeline 7 étages
- [ ] `docs/architecture/data_flow.md` : flux InsightSignalV2 de DataProvider à Telegram
- [ ] `docs/architecture/chatbot_architecture.md` : context_builder + refusal_detector + cache
- [ ] `docs/architecture/scoring_pipeline.md` : LGB → Isotonic → ACI
- [ ] `docs/architecture/frontend_architecture.md` : Next.js App Router + composants critiques

**Validation** :
- [ ] Si tu ne connais pas le code, peux-tu comprendre l'architecture en 30 min en lisant ces 5 docs ?

## E.2 — Runbooks ops

**Sous-critères** :
- [ ] `docs/runbooks/disaster_recovery.md` : restauration backup, rollback
- [ ] `docs/runbooks/circuit_breaker_tuning.md` : quand modifier les seuils
- [ ] `docs/runbooks/incident_response.md` : déjà créé dans `legal_templates/`
- [ ] `docs/runbooks/deploy.md` : procédure `fly deploy` step-by-step
- [ ] `docs/runbooks/monitoring.md` : dashboards Plausible + Sentry + Fly metrics
- [ ] `docs/runbooks/scoring_retraining.md` : comment ré-entraîner le scoring v2

**Validation** :
- [ ] Si urgence à 3h du matin, peux-tu rollback en suivant le runbook sans réfléchir ?

## E.3 — ADRs (Architecture Decision Records)

**5 ADRs minimum** :
- [ ] ADR-001 : choix architecture progressive uniforme (vs toggle 3 modes)
- [ ] ADR-002 : choix LGBM + Isotonic + ACI (vs alternatives)
- [ ] ADR-003 : choix Fly.io + Vercel (vs Railway/AWS/autres)
- [ ] ADR-004 : choix Plausible self-hosted (vs GA4)
- [ ] ADR-005 : choix bilingue FR+EN dès J1 (vs FR-first)

**Format** :
```
# ADR-XXX : Titre court
## Context : problème, contraintes
## Decision : ce qu'on a choisi
## Consequences : conséquences positives + négatives
## Alternatives considered : autres options écartées + pourquoi
## Date : YYYY-MM-DD
## Status : Accepted | Superseded | Deprecated
```

## E.4 — Tests E2E Playwright

**Suite Playwright couvrant** :
- [ ] Visiteur landing → /pricing → /signup FREE → première lecture
- [ ] User FREE → signal_view → chatbot 3 questions → upgrade_clicked
- [ ] User PRO → mode EXPERT → waterfall + conformal + sources cliquables
- [ ] User STARTER → trial 14j expiration → revert FREE auto
- [ ] Test mobile viewport iPhone SE
- [ ] Test responsive tablet iPad Mini
- [ ] Test cross-browser (Chromium + Firefox + WebKit)

**Sous-critères** :
- [ ] Screenshots regression : snapshots diff stockés
- [ ] CI bloque PR si E2E échoue
- [ ] Coverage parcours critiques : 100 %

**Validation** :
```bash
npx playwright test --project=chromium --project=firefox --project=webkit
```

## E.5 — Tests load / performance

**Sous-critères** :
- [ ] 100 users concurrents simulés (`signal_view` + `chatbot_question`)
- [ ] p99 latency < 2s
- [ ] Aucune erreur 5xx
- [ ] Cost Anthropic projetable avec confiance ($X / N users)

**Validation** :
- [ ] `k6` ou `locust` test load réussi

## E.6 — README projet + onboarding < 1h

**Sous-critères** :
- [ ] `README.md` root : vision + setup dev + tests + deploy
- [ ] `webapp/README.md` : setup Next.js, env vars frontend
- [ ] `backend/README.md` (ou root) : setup Python, run scanner
- [ ] Test "onboarding 6 mois plus tard" :
  - `git clone`
  - `cp .env.local.example .env.local`
  - `npm install && pip install -r requirements.txt`
  - `npm run dev` + `python -m src.intelligence.main`
  - Premier signal généré en console
- [ ] Critère : si tu reviens dans 6 mois, peux-tu re-build en < 1h sans demander aide ?

## E.7 — Changelog + versioning

**Sous-critères** :
- [ ] `CHANGELOG.md` à la racine
- [ ] Format Keep a Changelog
- [ ] Versions sémantiques (`2.1.0` = InsightSignalV2 v2.1.0)
- [ ] Entrées datées avec : Added / Changed / Deprecated / Removed / Fixed / Security
- [ ] Versioning du schema InsightSignal (déjà v2.1.0)

---

# ✅ Synthèse Gate Final

**Quand les 35 sous-critères sont validés, le produit est launching-ready.**

## Tableau récap des 35 sous-critères

| Pilier | Sous-critères | Statut (à cocher au fil de l'avancement) |
|---|---|---|
| **A — Solidité technique** | | |
| A.1 Scoring v2 calibré | 13 items | [ ] |
| A.2 Sweep state machine | 6 items | [ ] |
| A.3 Data quality fail-fast | 5 items | [ ] |
| A.4 Sécurité 4 vulnérabilités | 11 items | [ ] |
| A.5 Audit sécurité complet | 11 items | [ ] |
| A.6 Tests couverture | 8 items | [ ] |
| **B — Robustesse opérationnelle** | | |
| B.1 Telegram zéro perte | 5 items | [ ] |
| B.2 Cost monitoring Anthropic | 5 items | [ ] |
| B.3 Sentry backend + frontend | 6 items | [ ] |
| B.4 Circuit breakers | 5 items | [ ] |
| B.5 Logs structurés JSON | 5 items | [ ] |
| B.6 Health checks | 4 items | [ ] |
| B.7 Backup automatique | 5 items | [ ] |
| **C — Qualité UX** | | |
| C.1 Mobile-first | 7 items | [ ] |
| C.2 Accessibility WCAG AA | 10 items | [ ] |
| C.3 Loading/error/empty states | 14 items | [ ] |
| C.4 Animations 60fps | 6 items | [ ] |
| C.5 i18n FR+EN | 7 items | [ ] |
| C.6 Form validation | 5 items | [ ] |
| C.7 Performance | 7 items | [ ] |
| C.8 Cross-browser | 7 items | [ ] |
| **D — Cohérence produit** | | |
| D.1 Chatbot anti-hallucination | 5 items | [ ] |
| D.2 Refus pédagogique | 8 items | [ ] |
| D.3 Vocabulaire interdit | 4 items | [ ] |
| D.4 Sources RAG 12 papers | 4 items | [ ] |
| D.5 Track-record public | 9 items | [ ] |
| D.6 Architecture progressive uniforme | 7 items | [ ] |
| D.7 Posture éducative | 6 items | [ ] |
| **E — Maintenabilité** | | |
| E.1 Documentation architecture | 5 items | [ ] |
| E.2 Runbooks ops | 6 items | [ ] |
| E.3 ADRs | 5 items | [ ] |
| E.4 Tests E2E Playwright | 9 items | [ ] |
| E.5 Tests load | 4 items | [ ] |
| E.6 README onboarding | 5 items | [ ] |
| E.7 Changelog versioning | 5 items | [ ] |

**Total items granulaires** : ~220 items cochés cumulatifs.

---

# 🤝 Comment l'autre instance utilise ce document

**Sans changer son séquencement actuel** :

1. À chaque PR, **vérifier** quels sous-critères sont cochés par ce changement
2. Mettre à jour ce fichier ou `docs/governance/vague1_execution/PROGRESS.md` avec les items validés
3. **Ne pas** abandonner son ordre de sprints pour suivre l'ordre des piliers
4. Quand un sprint termine, lister les sous-critères cochés
5. Quand le total atteint 35/35 (avec items granulaires ≥ 95 % cochés) → **Gate Final atteint**

**Ouvre alors une PR** `feat: pre-launch perfection gate validated` avec ce fichier complété + lien vers PRs où chaque pilier a été validé.

---

# 📚 Documents de référence

- `MASTER_PLAN.md` — vue sprints chronologique
- `PILLARS_PERFECTION.md` (ce fichier) — vue piliers thématique
- `vague1_execution/briefs/` — détails ready-to-code par item DG-XXX
- `vague1_execution/tests/` — acceptance criteria + tests adversariaux
- `vague1_execution/copies/` — copies finales (chatbot prompts, scripted responses)
- `vague1_execution/configurations/` — configs prod (env vars, geo-block, quotas)
- `mockups/v3/best_concept_demo.html` — référence visuelle

---

**FIN PILLARS_PERFECTION.md** — checklist exhaustive pour atteindre "fonctionnel à la perfection" sans changer le séquencement de l'autre instance.
