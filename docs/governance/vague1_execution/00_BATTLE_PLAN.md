# Battle Plan — Vague 1 sprint-by-sprint

**Période** : S1-S6 (≈ 6 semaines, 2026-06-01 → 2026-07-15)
**Effort total** : ~230-290h dev + ~120h compliance/marketing/setup utilisateur
**Gate de sortie** : 1er paiement Stripe live encaissé

---

## Légende

- 🔧 **Dev** = code à écrire
- ⚙️ **Config** = configuration produit (env vars, DB migration, etc.)
- 📝 **Markdown** = doc/copy/template
- 💼 **User** = action utilisateur (souscription, signature, paiement)
- 🧪 **Test** = test à écrire + valider
- 🚀 **Deploy** = déploiement prod

---

## 🟢 Sprint 1 (semaine 1) — Décisions + cleanup + setup statut juridique

**Objectif** : socle propre, code mort éliminé, statut juridique en place pour facturer.

### S1.1 — Décisions politiques actées (LUNDI MATIN)

| Tâche | Type | Effort | Owner | Deps |
|---|---|---|---|---|
| Acter Vision B dans `reports/governance/kill_criteria_board.md` | 📝 | 30 min | toi | — |
| Acter Pricing v1 + Instruments XAU+EUR + Vocab dans `kill_criteria_board.md` | 📝 | 30 min | toi | — |
| Communiquer décisions à l'autre instance Claude Code | 📝 | 5 min | toi | — |

### S1.2 — Setup statut juridique (cette semaine)

| Tâche | Type | Effort | Owner | Deps |
|---|---|---|---|---|
| Créer auto-entreprise sur `autoentrepreneur.urssaf.fr` | 💼 | 15 min | toi | — |
| Acheter domain `mia.markets` (Namecheap/OVH) | 💼 | 10 min | toi | — |
| Souscrire ProtonMail Custom Domain (`contact@`, `privacy@`, `security@`, `support@`) | 💼 | 30 min | toi | domain |
| Devis + souscription RC Pro Freelance Hiscox/Wemind | 💼 | 1h | toi | AE créée |
| Compte bancaire pro Shine/Qonto | 💼 | 30 min | toi | AE créée |
| Récupérer SIRET (7-14j après création AE) | 💼 | passif | toi | AE créée |

### S1.3 — Cleanup code mort (parallèle de S1.2)

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-001 supprimer `parallel_training.py` | 🔧 | 30 min | code | — | trivial |
| DG-002 supprimer `tests/test_long_short_trading.py` | 🔧 | 5 min | code | — | déjà D dans git |
| DG-008 supprimer `models/scoring_v2.lgb` si non référencé | 🔧 | 15 min | code | — | grep avant drop |
| DG-009 supprimer `test_env_debug.py` racine | 🔧 | 5 min | code | — | trivial |
| DG-011 dédup `data/macro/` vs `data/research/` | 🔧 | 1h | code | — | trivial |
| DG-012 drop `Procfile` + `railway.toml` legacy | 🔧 | 5 min | code | — | trivial |
| DG-014 drop `scripts/export_mt5_history.py` | 🔧 | 5 min | code | — | trivial |
| DG-041 vérification `SENTINEL_TESTING_MODE=0` défaut prod + gate CI | 🔧 | 1h | code | — | brief Sécu |

**Gate S1** : statut juridique en place, code mort purgé, gate CI bloquante warn active.

---

## 🟡 Sprint 2 (semaine 2) — Data quality + infra deploy + sécurité

**Objectif** : infra prod déployée, sécurité critique fixée, Telegram robuste.

### S2.1 — Data quality (LUNDI-MARDI)

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-004 archive `XAU_15MIN_2019_2025.csv` 63% → `data/_archived/` + README | 🔧 | 1h | code | — | trivial |
| DG-053 implémenter `verify_data_quality` au boot fail-fast | 🔧 | 4h | code | DG-004 | brief Data Quality |
| Tests boot avec feed corrompu → doit fail | 🧪 | 1h | code | DG-053 | — |

### S2.2 — Infra deploy (MARDI-VENDREDI)

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-022 setup Fly.io app + `fly.toml` cdg region | ⚙️ | 4h | code | — | brief Fly.io setup |
| DG-022 deploy initial backend Sentinel | 🚀 | 2h | code | fly.toml | — |
| DG-023 init Next.js 15 + Tailwind + shadcn project dans `frontend/` | 🔧 | 4h | code | — | brief Next.js init |
| Deploy frontend initial sur Vercel (hello world) | 🚀 | 2h | code | DG-023 | — |
| Configurer DNS `mia.markets` → Fly.io + Vercel | ⚙️ | 1h | toi | domain acheté | — |
| DG-029-MODIFIED Fly.io secrets natifs (env ANTHROPIC_API_KEY, TELEGRAM_*, etc.) | ⚙️ | 1h | code | Fly.io déployé | brief Secrets |

### S2.3 — Sécurité critique (parallèle)

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-055 HMAC admin replay protection nonce-based | 🔧 | 4h | code | — | brief Sécu HMAC |
| DG-056 UNIQUE constraint sur `users.api_key_id` + migration SQLite | 🔧 | 2h | code | — | brief Sécu UNIQUE |
| DG-057 Lecture `subscription_expires` dans `require_api_key` | 🔧 | 2h | code | — | brief Sécu Expiry |
| Tests adversariaux replay + hijack + expiry | 🧪 | 3h | code | DG-055/56/57 | — |

### S2.4 — Telegram delivery robuste

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-054 fix python-telegram-bot v20+ async + retry exp backoff 429 + dedup `(chat_id, signal_id)` | 🔧 | 8h | code | — | brief Telegram retry |
| Test integration ≥ 30 abonnés simultanés | 🧪 | 2h | code | DG-054 | — |

**Gate S2** : prod déployée Fly.io + Vercel, sécurité OUVERTE F-03/F-04/F-05 fermées, Telegram async stable.

---

## 🟠 Sprint 3 (semaine 3) — Scoring + frontend MVP core

**Objectif** : scoring calibré qui donne sens au "PF 1.30", landing hero card visible, analytique opérationnelle.

### S3.1 — Refonte scoring (LUNDI-MERCREDI, gros morceau)

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-025 implémenter `LogisticL1Scorer` (sklearn) | 🔧 | 6h | code | — | brief DG-025 |
| DG-025 implémenter `IsotonicRecalibrator` | 🔧 | 4h | code | LGB en place | — |
| DG-025 wire `CalibratedConvictionPipeline` (LGB → Isotonic → ACI) | 🔧 | 4h | code | — | — |
| DG-025 fallback gracieux si modèle non chargé (mode A legacy) | 🔧 | 2h | code | — | — |
| DG-025 entraîner v1 sur XAU 7 ans walk-forward | 🔧 | 4h | code (Colab) | — | — |
| DG-025 export `models/calibrated_conviction_v1.pkl` | ⚙️ | 1h | code | training | — |
| DG-025 tests unitaires + intégration | 🧪 | 4h | code | — | — |
| DG-025 feature flag `SCORING_VERSION=v1|v2`, défaut v2 en prod | ⚙️ | 1h | code | — | — |

### S3.2 — Frontend MVP core (MARDI-VENDREDI parallèle)

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| **DG-101-MODIFIED** un seul renderer + sections collapsibles tier-gated | 🔧 | 16-20h | code | DG-023 Next.js init | brief DG-101 |
| **DG-120** landing hero card track-record honnête | 🔧 | 8-12h | code | DG-023 | brief DG-120 |
| Audit visuel mockup HTML vs implémentation Next.js (qualité égale) | 🧪 | 2h | code | — | — |

### S3.3 — Analytique core P0-strict V1

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| **DG-160** Plausible self-hosted sur Fly.io app | ⚙️ | 6h | code | Fly.io déployé | brief DG-160 |
| **DG-161** event tracking core (6 events) intégration front + back | 🔧 | 10-14h | code | DG-160 | brief DG-161 |
| Dashboard Plausible vérifié (events arrivent) | 🧪 | 1h | code | DG-161 | — |

### S3.4 — Préparation monetization

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-006 tier rate-limit mode `warn` (préparation enforce S5) | 🔧 | 2h | code | — | — |

**Gate S3** : scoring v2 chargé, landing visible avec hero card, Plausible reçoit events, mode `warn` rate-limit actif.

---

## 🔵 Sprint 4 (semaine 4) — Chatbot pilier + compliance UX

**Objectif** : chatbot fonctionne comme pilier, refus pédagogique scripté, compliance UX en place.

### S4.1 — Chatbot wiring (LUNDI-JEUDI, gros morceau)

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-042 wire `NARRATIVE_MODE=llm` tier-routed (FREE=template, STARTER=Haiku, PRO=Sonnet) | 🔧 | 4h | code | — | brief DG-042 |
| **DG-110** wire chatbot context-injection 8 composantes (décomposition "Pourquoi 72 ?") | 🔧 | 20-30h | code | DG-025 scoring v2 | brief DG-110 |
| **DG-112** tests adversariaux refus pédagogique (10+ patterns) | 🔧 | 6-10h | code | DG-110 | brief DG-112 |
| **DG-114-REDUCED** 3 questions suggérées contextuelles | 🔧 | 6-8h | code | DG-110 | brief DG-114 |
| Tests chatbot E2E 6 dialogues types | 🧪 | 4h | code | DG-110/112/114 | — |

### S4.2 — Compliance UX (MERCREDI-VENDREDI parallèle)

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-038 endpoints DSAR `/me/data` (GET) + `/me` (DELETE) | 🔧 | 8h | code | — | brief DSAR |
| DG-047 param API `disclosure_mode=qualitative|numeric` défaut qualitative | 🔧 | 4h | code | — | brief MiFID |
| Audit wording compliance UE 2024/2811 (vocabulaire interdit) | 📝 | 4h | toi+code | `legal_templates/disclaimer_compliance.md` | — |
| DG-048 cookie banner CNIL — pas nécessaire si Plausible self-hosted seul (à vérifier) | ⚙️ | 1h | code | DG-160 | — |
| DG-005-FUSED migration providers : cut FF/Dukascopy live + delete scrapers + archive backtest | 🔧 | 6h | code | — | brief Migration providers |
| DG-103 mobile-first responsive layout | 🔧 | 16h | code | DG-101 | brief DG-103 |

### S4.3 — Cost monitoring

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-052 Prometheus gauge `llm_cost_usd_total` + alerte Discord/email seuil | 🔧 | 4h | code | — | brief DG-052 |

**Gate S4** : chatbot répond aux 6 questions types incl. refus pédagogique, compliance UE auditée, mobile-first OK, cost monitoring actif.

---

## 🟣 Sprint 5 (semaine 5) — Monetization + lancement marketing

**Objectif** : monetization fonctionnelle (hors Stripe live), pricing page complète, track-record public ouvert.

### S5.1 — Hard caps + tier enforcement

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-046 hard caps signaux/mois par tier (DB counter SQLite — Redis defer) | 🔧 | 6h | code | — | brief DG-046 |
| DG-046 soft-cap warn 80% UX + email warning | 🔧 | 4h | code | DG-046 caps | — |
| Cap global abonnés payants = 50 (anti-exposition stratégie bootstrap) | 🔧 | 2h | code | DG-046 | — |
| DG-006 tier rate-limit mode `enforce` après 7j warn | ⚙️ | 1h | code | DG-046 | — |

### S5.2 — Geo-block strict + fiscal

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-045 geo-block prod allow-list FR+BE+CH+LU (au lieu de deny-list US/QC/UK/OFAC) | 🔧 | 4h | code | — | brief Geo-block |
| DG-044 Stripe Tax UE + reverse charge B2B + EU OSS (config seulement, activation S6) | ⚙️ | 4h | toi+code | Stripe compte | brief Stripe Tax |
| Test VPN US/UK/JP → 451 Unavailable | 🧪 | 1h | code | DG-045 | — |

### S5.3 — Page pricing + marketing landing

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| **DG-132** page pricing avec decoy $1990 + dual trial 14j+14j | 🔧 | 10-16h | code | — | brief DG-132 |
| DG-077 USP "honest confidence" en hero secondaire landing | 🔧 | 2h | code | DG-120 | copy fournie |
| DG-080 INSTITUTIONAL = book demo Calendly intégré | 🔧 | 2h | code | DG-132 | — |
| DG-083 decoy $1990 toujours visible pricing (4 cards) | 🔧 | inclus DG-132 | code | DG-132 | — |
| DG-084 dual trial 14j sans CB + 14j avec CB | 🔧 | 4h | code | DG-132 | — |
| DG-079 refund 30j codifié dans CGU V0 + Stripe Customer Portal | 📝 | 1h | toi | CGU V0 | — |

### S5.4 — Sources RAG curées + track record public

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-058a 12 papers académiques curés + mini-fiches inline | 🔧 | 16-20h | code | — | brief DG-058a |
| **DG-142** tableau performance public mensuel forme simple | 🔧 | 14-20h | code | DG-025 + signals.db | brief DG-142 |
| DG-072 ouverture Telegram public channel "M.I.A. Markets — Public Tape" forward only | 💼 | 1h | toi | — | — |

### S5.5 — Légal + assurance

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| RC Pro Freelance souscrite (rappel S1) | 💼 | — | toi | — | — |
| Adhésion médiation conso (CM2C/MEDICYS, optionnel S5 ou S6) | 💼 | 1h | toi | — | — |
| Publication CGU V0 + Privacy V0 + Mentions légales (templates customisés) | 📝 | 8h | toi | SIRET reçu | `legal_templates/` |

### S5.6 — Process / qualité

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-035 CI bloquante mode warn (sera enforce S6) | ⚙️ | 2h | code | — | — |
| DG-032 process versioning schema InsightSignal écrit (`docs/api/schema_versioning.md`) | 📝 | 1h | code | — | — |

**Gate S5** : page pricing fonctionnelle, geo-block strict, hard caps actifs, track record public visible, CGU V0 publiée.

---

## 🔴 Sprint 6 (semaine 6) — Lancement Stripe live

**Objectif** : 1er paiement Stripe live encaissé.

### S6.1 — Stripe live activation

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| Stripe : créer 4 products (STARTER / PRO / INSTITUTIONAL) + 8 price IDs (mensuel + annuel) | ⚙️ | 2h | toi+code | DG-070 grille | brief Stripe products |
| Stripe : activer Customer Portal (config refund 30j) | ⚙️ | 1h | toi | products créés | — |
| Stripe : webhook signed configuration | ⚙️ | 2h | code | products créés | — |
| Stripe : tests Checkout end-to-end (carte 4242) | 🧪 | 2h | code | webhook | — |
| Stripe : tests refund automation | 🧪 | 1h | code | Customer Portal | — |
| Stripe : soumission compte pour review fintech (avant activation live) | 💼 | 30 min | toi | — | — |
| **DG-043 activation Stripe live mode** | 🚀 | 1h | toi | review passée | — |
| Tests Checkout en mode LIVE (vraie CB perso, refund immédiat) | 🧪 | 1h | toi | live activé | — |

### S6.2 — Optimisations finales

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| DG-035 CI bloquante mode `enforce` (warn → enforce après 7j) | ⚙️ | 1h | code | DG-035 warn | — |
| DG-034 sweep state machine 432 cellules (post-DG-025) — peut être différé S7 si retard | 🔧 | 16h (Colab) | code | DG-025 stable | brief DG-034 |
| Audit final compliance UE + RGPD + MiFID | 📝 | 4h | toi+code | — | checklist `tests/` |
| Documentation déploiement runbook | 📝 | 2h | code | — | — |

### S6.3 — Lancement contrôlé

| Tâche | Type | Effort | Owner | Deps | Brief |
|---|---|---|---|---|---|
| Annonce soft launch Twitter / Telegram / mailing perso (≤ 20 prospects beta) | 💼 | 4h | toi | tout livré | — |
| Suivi 1er paiement encaissé | 💼 | passif | toi | — | — |
| Audit J+7 : taux conversion, plaintes, refunds, bugs | 📝 | 4h | toi+code | live | — |

**Gate de sortie Vague 1** :
- ✅ 1er paiement Stripe live encaissé
- ✅ CGU V0 publiée
- ✅ Hero card mobile + desktop opérationnel
- ✅ Chatbot répond aux 6 questions types
- ✅ Architecture progressive uniforme opérationnelle
- ✅ Track-record Telegram public ≥ 30 trades clôturés

---

## 📊 Vue parallélisation

```
                     S1 ─────── S2 ─────── S3 ─────── S4 ─────── S5 ─────── S6
                     │
[Code path 1]        cleanup ── data quality ── scoring DG-025 ── chatbot DG-110 ── monetize ── Stripe live
[Code path 2]                   infra deploy ── frontend MVP ──── compliance UX ── pricing ──── CI enforce
[Code path 3]                   sécu critique ── Plausible ────── DSAR ──────────── geo-block ── refund
[User path]          AE+RC Pro ── domain+email ── (passif) ────── audit wording ── CGU V0 ── Stripe live
```

3 voies de code parallélisables (si 3 dev OU 1 dev qui jongle bien). Sinon 6 sem en série compacte.

---

## ⚠️ Risques principaux Vague 1 et mitigation

| Risque | Probabilité | Mitigation |
|---|---|---|
| DG-025 scoring ne converge pas / Brier skill < +5 % | Moyenne | Kill criterion S8 (DG-081) — pivot B2B si confirmé. Mais en V1, fallback v1 legacy reste actif. |
| Suspension Stripe pour wording | Moyenne | Audit wording strict S4 + soumission compte review S6 + posture Early Access |
| Délai création AE > 14j | Faible | Démarrer S1 immédiat (lead time URSSAF ~7-14j) |
| Bug critique chatbot (hallucination, refus pédagogique cassé) | Moyenne | Tests adversariaux DG-112 obligatoires + cost monitoring DG-052 + circuit breaker |
| Coût Anthropic dépasse budget | Moyenne | Cache sémantique + cost monitoring DG-052 + hard caps DG-046 + tier-routed (FREE=template) |
| Sweep state machine DG-034 retarde S6 | Moyenne | Différable S7 sans casser Vague 1, defaults `enter=65` acceptables temporairement |

---

## 🎓 Checklist d'exécution journalière (pour l'instance qui exécute)

### Chaque jour
- [ ] Lire `00_BATTLE_PLAN.md` section du sprint en cours
- [ ] Choisir 1-3 tâches du sprint (selon deps satisfaites)
- [ ] Pour chaque tâche : lire le brief (`briefs/DG-XXX_*.md`)
- [ ] Coder → tester → PR avec acceptance criteria coché
- [ ] Mise à jour MEMORY.md si décision non-évidente prise

### Chaque fin de sprint
- [ ] Vérifier gate de sortie sprint
- [ ] Mettre à jour `docs/governance/vague1_execution/PROGRESS.md` (à créer en cours d'exécution)
- [ ] Audit lessons learned

---

**FIN DU BATTLE PLAN.** Lire ensuite `briefs/README.md` pour entrer dans les détails P0.
