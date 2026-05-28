# PROGRESS — Phase 1 DEV Performance-ready (Sprints 1-6)

**Démarré** : 2026-05-27
**Référence plan** : `docs/governance/dev_focus_plan_2026_05_27.md`
**Branche** : `institutional-overhaul`

---

## Sprint 1 — Cœur algorithmique (DG-025/034/004/053)

**Statut** : ✅ LIVRÉ (avec une réserve documentée sur Brier skill)
**Démarré** : 2026-05-27
**Terminé** : 2026-05-27

### Items
- [x] Audit DG-025 : pipeline LGBM-Isotonic-Conformal entraîné + chargé prod
- [x] DG-025 finition : entraînement sur 133 trades XAU 2019-2026, feature flag `SCORING_VERSION` câblé (default v1)
- [x] DG-034 sweep state machine **harnais 432 cellules** livré (`scripts/sweep_state_machine_432.py`), smoke run 8/8 OK ; full run handoff Colab documenté
- [x] DG-004 feed XAU 63 % déplacé vers `data/_archived/` + README
- [x] DG-053 `verify_data_quality_or_abort` au boot — coverage + structural strict, exit codes 3/4
- [x] Tests scoring : **91 %** coverage sur `src/intelligence/scoring/` (76 tests verts)

### Gate de sortie
- [x] Scoring v2 chargé en prod via env var (default v1 — pipeline prêt à flip)
- [ ] ⚠️ **Brier skill ≥ +5 %** : **NON ATTEINT** (Brier skill OOS = −0.251 sur 133 trades). Voir `reports/scoring_v2_brier_validation.md` pour le verdict empirique honnête. Pipeline et wiring prêts, mais le gate empirique requiert Sprint 4+ ou pivot pilier institutionnel.
- [x] Sweep state machine — harnais 432 cellules livré, full run XAU + EUR à exécuter Colab (handoff documenté)
- [x] Data quality boot fail-fast actif (gates coverage 95 % + structural NaN/H/L/dup)

### Artefacts livrés
- `models/calibrated_conviction_v1.pkl` (27.9 kB)
- `scripts/train_calibrated_conviction_real.py` (adapter trades CSV → format trainer)
- `scripts/sweep_state_machine_432.py` (harnais 432 cellules)
- `reports/scoring_v2_brier_validation.md` (rapport honnêteté Brier)
- `reports/sweep_432/README.md` (handoff Colab)
- `reports/calibration/trades_xau_2019_2026.csv` (133 trades enrichis avec composantes)
- `data/_archived/README.md` + feed déplacé
- `tests/test_dg025_wiring.py`, `tests/test_logistic_l1_scorer.py`, `tests/test_lgbm_scoring_engine.py`

### Décisions ad-hoc
- **`SCORING_VERSION=v1` par défaut en prod**. Le pipeline v2 est chargé uniquement quand explicitement activé via env, parce que Brier skill OOS < 0 → le calibrer expose un faux signal calibré. Réversible en flippant l'env var dès qu'un edge est démontré.
- **`--silent` exposé dans `scripts/run_backtest.py`** (nouvel arg) pour permettre le sweep 6-dimensionnel.

---

## Sprint 2 — Sécurité critique (DG-041/055/056/057)

**Statut** : ✅ LIVRÉ
**Démarré** : 2026-05-27
**Terminé** : 2026-05-27

### Items
- [x] DG-041 TESTING_MODE=0 défaut prod + gate CI fail si =1 (`scripts/ci_testing_mode_gate.py` + job `.github/workflows/ci.yml`)
- [x] DG-055 HMAC admin replay nonce-based (`src/api/nonce_store.py`, canonical `ts:nonce:path`, default `ADMIN_NONCE_REQUIRED=on`)
- [x] DG-056 UNIQUE constraint `api_key_id` + migration v1→v2 (`uq_users_api_key` partial unique index + dedup on upgrade)
- [x] DG-057 lecture `subscription_expires` dans `require_api_key` (402 si expiré) + setter `tier_manager.set_subscription_expires`
- [x] Audit sécurité global → `docs/audits/security_M0_2026.md`

### Gate de sortie
- [x] 4 vulnérabilités fermées (DG-041 + F-03/04/05)
- [x] Audit sécurité écrit
- [x] Gate CI bloque mode test en prod (job `testing_mode_gate`)

### Tests
- `tests/test_admin_hmac_nonce.py` — 11/11 ✅
- `tests/test_dg056_unique_api_key.py` — 6/6 ✅
- `tests/test_dg057_subscription_expires.py` — 10/10 ✅
- `tests/test_auth.py` régression — 34/34 ✅ (helper `_admin_headers` upgradé nonce-aware)

### Décisions ad-hoc
- **Mode legacy admin HMAC conservé** via `ADMIN_NONCE_REQUIRED=off` pour cohabitation avec scripts ops existants jusqu'à migration coordonnée. Default est strict.
- **`subscription_expires` malformé → utilisateur non-bloqué** (warn log + treat as active) pour éviter qu'une mauvaise écriture Stripe verrouille les comptes. Audit row-level recommandé.
- **DG-056 dedupe** : sur migration v1→v2 avec doublons préexistants, on garde l'utilisateur le plus ancien (min user_id), nulle le link sur les autres, log WARN.

---

## Sprint 3 — Chatbot pilier (DG-110/111/112/114/042)

**Statut** : ⏳ EN ATTENTE

### Items
- [ ] Audit DG-110 : contexte InsightSignalV2 complet injecté ?
- [ ] DG-110 finition si gaps (context_builder.py)
- [ ] DG-111 chatbot conformal + stats J.* (ACI Gibbs-Candès en langage humain)
- [ ] DG-112 tests adversariaux refus pédagogique (≥30 FR + 18 EN, recall ≥ 98 %)
- [ ] DG-114-REDUCED 3 questions suggérées contextuelles dynamiques
- [ ] DG-042 NARRATIVE_MODE=llm tier-routed (FREE/STARTER/PRO/INSTITUTIONAL)
- [ ] Validation forbidden tokens post-processing
- [ ] Session memory 5-turn

### Gate de sortie
- [ ] Chatbot répond aux 6 questions types
- [ ] 30+ tests adversariaux passent
- [ ] Aucun vocabulaire interdit ne fuite

---

## Sprint 4 — Delivery + Observability (DG-054/052/033)

**Statut** : ⏳ EN ATTENTE

### Items
- [ ] DG-054 Telegram retry exp backoff + dedup (chat_id, signal_id)
- [ ] DG-052 cost monitoring Anthropic + alerte Discord/email
- [ ] DG-033 Sentry activé + source maps front
- [ ] Circuit breaker thresholds validation 30j logs

### Gate de sortie
- [ ] Telegram robuste 30+ abonnés sans flood
- [ ] Cost monitoring actif
- [ ] Sentry capture exceptions
- [ ] CB thresholds calibrés

---

## Sprint 5 — Sources RAG + Track Record (DG-058a/142)

**Statut** : ⏳ EN ATTENTE

### Items
- [ ] DG-058a 12 papers RAG curés (López de Prado, Corsi, Gibbs & Candès, Barndorff-Nielsen, Adams & MacKay, Angelopoulos & Bates, Avellaneda, Cont, Engle, Hasbrouck, Lo, Pedersen)
- [ ] DG-142 `/track-record` page : PF + IC bootstrap + equity curve
- [ ] Nightly cron stats agrégées + snapshot R2
- [ ] Bootstrap CI 95 % (1000 itérations) — IC reproductible

### Gate de sortie
- [ ] 12 sources RAG cliquables (mode EXPERT)
- [ ] Track-record public quotidien
- [ ] IC bootstrap validé

---

## Sprint 6 — Infra deploy + Analytique (DG-022/029/103/160/161)

**Statut** : ⏳ EN ATTENTE

### Items
- [ ] DG-022 Fly.io cdg deploy + DNS `api.mia.markets`
- [ ] DG-029-MODIFIED secrets natifs `fly secrets set`
- [ ] DG-103 mobile-first responsive audit Lighthouse ≥ 90
- [ ] DG-160 Plausible self-hosted sur Fly.io
- [ ] DG-161 event tracking core 6 events

### Gate de sortie
- [ ] Backend Fly.io + Vercel en prod
- [ ] Plausible reçoit events
- [ ] Mobile-first opérationnel

---

## Performance-ready Gate (10 critères)

- [ ] Scoring v2 chargé en prod avec Brier skill ≥ +5 % vs baseline
- [ ] Sweep state machine 432 cellules livré, defaults empiriques
- [ ] Feed XAU 100 % coverage, verify_data_quality fail-fast actif
- [ ] 4 vulnérabilités sécurité fermées (F-03/04/05 + TESTING_MODE gate CI)
- [ ] Chatbot répond aux 6 questions types avec contexte InsightSignalV2 injecté
- [ ] 30+ tests adversariaux refus pédagogique passent
- [ ] Telegram retry + dedup activé, 0 message perdu sur 30 abonnés
- [ ] Cost monitoring Anthropic actif avec alerte spend
- [ ] Track-record public `/track-record` opérationnel avec PF + IC bootstrap quotidien
- [ ] 12 sources RAG cliquables dans mode EXPERT
- [ ] Plausible self-hosted + 6 events core opérationnels
- [ ] Mobile-first Lighthouse ≥ 90

---

## Journal de décisions ad-hoc

(à remplir au fil)
