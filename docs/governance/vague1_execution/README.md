# Execution Kit — Vague 1 (MVP commercialisable)

**Date** : 2026-05-26
**Cible** : démarrage S1 (~2026-06-01), gate de sortie ≈ S6 (~2026-07-15)
**Référence** : `docs/governance/decision_gate_review_v2.md` Partie 4
**Audience** : autre instance Claude Code chargée de l'exécution (OU toi en mode terminal)

---

## 🎯 Objectif Vague 1

Livrer un **MVP commercialisable** : un premier client B2C paie en confiance, hero card visible mobile + desktop, chatbot répond aux 6 questions types (incl. refus pédagogique), audit compliance UE 2024/2811 passé, track-record Telegram public ≥ 30 trades clôturés visibles.

**Stripe live activé S6.**

---

## 📂 Structure du dossier

```
docs/governance/vague1_execution/
├── README.md                           ← tu es ici
├── 00_BATTLE_PLAN.md                   ← plan sprint-by-sprint S1-S6
├── briefs/                             ← 10 specs ready-to-code (P0-strict-MVP)
│   ├── README.md                       ← index briefs
│   ├── DG-101_renderer_unique.md
│   ├── DG-103_mobile_first.md
│   ├── DG-110_chatbot_wiring.md
│   ├── DG-112_refus_pedagogique.md
│   ├── DG-114_questions_suggerees.md
│   ├── DG-120_landing_hero.md
│   ├── DG-132_page_pricing.md
│   ├── DG-142_track_record_public.md
│   ├── DG-160_plausible.md
│   └── DG-161_event_tracking.md
├── copies/                             ← textes finaux prêts à coller
│   ├── README.md
│   ├── landing_copy.md
│   ├── pricing_copy.md
│   ├── methodologie_copy.md
│   ├── emails_lifecycle.md
│   ├── chatbot_system_prompt.md
│   └── chatbot_scripted_responses.md
├── configurations/                     ← configs prêtes à appliquer
│   ├── README.md
│   ├── env_vars.md
│   ├── geo_block_allowlist.md
│   ├── stripe_products.md
│   ├── plausible_events.md
│   └── tier_quotas.md
├── tests/                              ← acceptance criteria
│   ├── README.md
│   ├── p0_acceptance_criteria.md
│   └── adversarial_chatbot_tests.md
└── scripts/                            ← commandes M0 setup
    ├── README.md
    └── m0_setup_commands.md
```

---

## 🚀 Ordre d'exécution recommandé

### Pour une instance Claude Code chargée d'exécuter

```
1. Lis 00_BATTLE_PLAN.md d'un bout à l'autre (~15 min)
2. Lis briefs/README.md (vue d'ensemble des 10 P0)
3. Pour chaque sprint S1→S6, exécute les items dans l'ordre du battle plan
4. Avant chaque coding session :
   - Lire le brief correspondant (briefs/DG-XXX_*.md)
   - Lire les copies associées (copies/)
   - Lire la config associée (configurations/)
5. À chaque PR : vérifier acceptance criteria (tests/p0_acceptance_criteria.md)
```

### Pour l'utilisateur (Loukmane)

```
1. Cette semaine : exécuter scripts/m0_setup_commands.md (auto-entreprise, domain, RC Pro, etc.)
2. Souscriptions différées :
   - Iubenda : optionnel V0 (templates V0 dans docs/governance/legal_templates/ suffisent au démarrage)
   - Trading Economics : à activer M1-M2 quand bannière event ≤4h passe en P1
   - Stripe live : S6, après CGU avocat (DEFER M3 selon stratégie bootstrap)
3. Surveiller progression de l'autre instance via PRs
4. Décisions ad-hoc qui émergent : trancher rapidement, ne pas bloquer l'exécution
```

---

## ⚠️ Décisions actées qui orientent toute la Vague 1

Référence : `docs/governance/decisions/2026-05-26_5_political_locks.md`

1. **Vision B narrative-first** confirmée → toute la roadmap technique sert narrative + chatbot, pas RL
2. **DG-075 avocat DEFER M3** → CGU V0 via templates fournis, Stripe live S6 conditionné à V0 publiée
3. **Pricing FREE / $29 / $79 / $1990** + dual trial 14j+14j + refund 30j
4. **Instruments XAU + EUR seuls** → drop dev BTC/US500/JPY/GBP
5. **Vocabulaire "signaux" → "analyses"** partout + USP "honest confidence" landing

---

## 🎯 Gate de sortie Vague 1 (à valider avant Vague 2)

- [ ] 1er paiement Stripe live encaissé
- [ ] CGU V0 publiée (templates customisés)
- [ ] Hero card visible mobile + desktop sur landing + webapp
- [ ] Chatbot répond aux 6 questions types (incl. refus pédagogique scripté)
- [ ] Audit compliance UE 2024/2811 + RGPD passé (auto-audit, pas avocat)
- [ ] Architecture progressive uniforme opérationnelle (hero + sections collapsibles tier-gated)
- [ ] Track-record Telegram public ≥ 30 trades clôturés visibles
- [ ] Plausible self-hosted opérationnel, 6 events trackés
- [ ] Geo-block FR+BE+CH+LU strict
- [ ] Cap 50 abonnés payants enforce

---

## 📊 Effort estimé Vague 1

| Sprint | Période | Items principaux | Effort dev |
|---|---|---|---|
| S1 | sem 1 | Décisions + cleanup code mort + setup AE + email | ~20-30h |
| S2 | sem 2 | Data quality + Fly.io deploy + sécurité + Telegram | ~40-50h |
| S3 | sem 3 | Scoring DG-025 + renderer unique + landing hero + Plausible | ~50-60h |
| S4 | sem 4 | Chatbot wiring + compliance UX + cost monitoring | ~40-50h |
| S5 | sem 5 | Monetization core + page pricing + track record | ~50-60h |
| S6 | sem 6 | Stripe live + CI bloquante + sweep state machine | ~30-40h |
| **Total** | **6 sem** | **~40 items** | **~230-290h** |

---

## 🔗 Documents connexes

- Stratégie globale : `../decision_gate_review_v2.md`
- Décisions politiques signées : `../decisions/2026-05-26_5_political_locks.md`
- Stratégie bootstrap légal : `../legal_bootstrap_strategy_2026_05_26.md`
- Templates légaux V0 : `../legal_templates/`
- Plan migration avocat M3 : `../legal_migration_plan_to_lawyer.md`
- Mockup HTML de référence : `../../../mockups/v3/best_concept_demo.html`
- Vérité terrain produit : `../../value/client_information_explained.txt`
- Concept B retenu : `../../value/best_product_concept.md`
- 24 recos enrichissement : `../../value/information_enrichment_recommendations.md`
