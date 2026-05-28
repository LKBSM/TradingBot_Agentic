# `docs/governance/` — Documents de gouvernance M.I.A. Markets

Index des documents produits lors de la revue critique du plan d'exécution (2026-05-26).

---

## 🎯 SOURCES UNIQUES DE VÉRITÉ

**🚨 [AUDIT_ALGO_2026_05_27.md](AUDIT_ALGO_2026_05_27.md)** — Audit algorithmique brutal (note 3/10). **Lecture obligatoire pour comprendre l'état actuel.**

**🔄 [decisions/2026-05-27_pivot_positioning_audit.md](decisions/2026-05-27_pivot_positioning_audit.md)** — Pivot positioning officiel post-audit (B + C en parallèle, pricing 9€/19€).

**📨 [PROPAGATION_BRIEF_2026_05_27.md](PROPAGATION_BRIEF_2026_05_27.md)** — Brief à transmettre aux autres terminaux pour aligner copies, pricing, wording.

**📕 [MASTER_PLAN.md](MASTER_PLAN.md)** — Plan pre-launch consolidé (8 sprints, Sprint 1 augmenté avec Blockers #2/#3/#4 post-audit). **Vue chronologique sprint-by-sprint.**

**📋 [PILLARS_PERFECTION.md](PILLARS_PERFECTION.md)** — Checklist 5 piliers du Gate Final (35 sous-critères, ~220 items). **Vue thématique pilier-by-pilier.**

Ces documents convergent vers le même Gate Final "fonctionnel à la perfection" + nouveau gate de promotion premium (Brier OOS > +2 % AND DSR > 1.0 AND PBO < 0.5).

---

## 📚 Documents principaux

### 1. `decision_gate_review_v2.md` — Rapport principal (1100+ lignes)

**Le document central.** Revue critique impitoyable du plan d'exécution `reports/commercialization_sprint/00_DANGEROUS_CHANGES.md` (69 items DG-001 à DG-085).

**Contenu** :
- Partie 1 — Cartographie d'alignement des 69 items (KEEP/MODIFY/DEFER/DROP)
- Partie 2 — Les 28 angles morts du plan (items manquants DG-100+)
- Partie 3 — Sur-dimensionnements, mal séquencés, doublons, dette légale déguisée
- Partie 4 — Re-séquençage en 3 vagues B2C (MVP / Activation / Solidification)
- Partie 5 — Les 5 décisions politiques à trancher maintenant
- Partie 6 — Synthèse 1 page pour décision

**Distribution finale plan original 69 items** : KEEP 50 / MODIFY 6 / DEFER 10 / DROP 3
**Items ajoutés effectifs** : 27 (28 − 1 DG-100 DROP)
**P0-strict-MVP** : 10 items
**DROP totaux** (plan + ajoutés) : 4 (DG-030, DG-049, DG-078, DG-100)

### 2. `MISSION_ACK_V2.md` — Acquittement de mission

Vision produit comprise + principes d'évaluation + corrections actées 2026-05-26.

### 3. `decisions/2026-05-26_5_political_locks.md` — 5 décisions SIGNÉES ✅

**Statut 2026-05-26** : 4/5 approuvées + 1/5 reformulée en stratégie bootstrap :
1. ✅ Vision B (narrative-first) — engagement 90 jours
2. 🔄 RFQ avocat fintech FR — **DEFER M3** avec stratégie provisoire (cf. ci-dessous)
3. ✅ Pricing v1 lock FREE/$29/$79/$1990 + dual trial + refund 30j
4. ✅ Instruments GA = XAU + EUR seuls
5. ✅ Reformulation "signaux"→"analyses" + USP "honest confidence"

### 4. `legal_bootstrap_strategy_2026_05_26.md` — Stratégie protection légale provisoire

Stratégie en 7 piliers pour lancer Vague 1 sans budget avocat (3-5 k€ DEFER M3) :
- Auto-entrepreneur FR (statut le plus simple)
- Géo-restriction FR + BE + CH + LU (4 juridictions au lieu de 30+)
- Posture produit "Early Access · Educational Use"
- Stack templates Iubenda Pro + suppléments V0
- Cap 50 abonnés payants M1-M3
- Refund 30j systématique (bouclier juridique principal)
- RC Pro Freelance basique 300-500€/an

**Coût bootstrap** : ~750-1640€/an vs 3-5 k€ avocat one-shot.
**Exposition résiduelle estimée** : ~$1 200-2 400/an pendant 3 mois max.

### 5. `legal_templates/` — 6 templates V0 prêts à customiser

- `disclaimer_compliance.md` — wording FR/EN multilingue
- `cgu_cgv_v0_template.md` — clauses fintech-specific à ajouter au template Iubenda
- `privacy_policy_v0_template.md` — Privacy Policy RGPD V0
- `mentions_legales_auto_entrepreneur.md` — mentions légales AE
- `cookie_notice_minimal.md` — sans bandeau intrusif (Plausible self-hosted)
- `incident_response_runbook.md` — 6 types d'incidents + templates emails
- `README.md` — guide d'usage du dossier

**Effort customisation** : ~8-10h sur 1-2 jours.

### 6. `legal_migration_plan_to_lawyer.md` — Plan migration V0 → V1 à M3

Déclencheurs, RFQ process (3 cabinets), brief avocat, séquencement migration sans rupture, souscription RC Pro complète + médiateur conso.

### 7. `OUT_OF_SCOPE.md` — Sujets identifiés hors mission

Liste des sujets soulevés pendant l'audit mais hors scope strict "revue critique" :
- 🚨 **PRIORITÉ P0** : réécriture mockup HTML `mockups/v3/best_concept_demo.html` (encore basé sur toggle 3 modes abandonné, doit passer en architecture progressive uniforme)
- Exécution Vague 1 S1-S6 (autre instance Claude Code)
- SOP `docs/runbooks/circuit_breaker_tuning.md` (DG-049 DROP)

---

## 📋 État des décisions politiques (snapshot 2026-05-26)

| # | Décision | Statut | Action |
|---|---|---|---|
| 1 | Vision B narrative-first | ✅ APPROUVÉ | Acter écrit dans `kill_criteria_board.md` |
| 2 | RFQ avocat fintech (3-5 k€) | 🔄 DEFER M3 + bootstrap | Stratégie provisoire active |
| 3 | Pricing v1 FREE/$29/$79/$1990 | ✅ APPROUVÉ | Implémenter en Vague 1 S5 |
| 4 | Instruments GA XAU+EUR | ✅ APPROUVÉ | Drop marketing "6 instruments" |
| 5 | Reformulation + USP "honest confidence" | ✅ APPROUVÉ | Audit wording + landing copy |

**Démarrage Vague 1 = débloqué.** Prochaine action : exécution des 10 P0-strict-MVP par l'instance qui touche au code.

---

## 🗺 Comment utiliser ces documents

### Si tu es l'utilisateur revenant 3 mois plus tard

1. Lis `decision_gate_review_v2.md` Partie 6 (synthèse 1 page) — 5 min
2. Vérifie l'état des 5 décisions politiques dans `decisions/2026-05-26_5_political_locks.md`
3. Si toujours pas démarrée Vague 1 : signe les décisions, brief autre instance Claude Code sur Partie 4

### Si tu es une future instance Claude Code

1. Lis `MISSION_ACK_V2.md` pour la vision produit en 5 lignes + principes d'évaluation
2. Lis `decision_gate_review_v2.md` Partie 4 (re-séquençage 3 vagues) pour la priorité d'exécution
3. Vérifie `OUT_OF_SCOPE.md` pour les sujets parallèles qui dépendent peut-être de toi
4. Note la mémoire persistante dans `~/.claude/projects/.../memory/MEMORY.md` (référence : `[Decision Gate Review V2]`)

### Si tu démarres l'exécution

**Prérequis obligatoire** : les 5 décisions politiques signées dans `decisions/2026-05-26_5_political_locks.md`. Sans signature, ne pas démarrer Vague 1.

**Premiers items à attaquer Vague 1 S1** :
- Décisions politiques tranchées + actées
- Cleanup code mort (DG-001/002/008/009/011/012/014 = 7 quick wins parallèles)
- DG-041 TESTING_MODE=0 + gate CI
- DG-027-CONSOLIDATED Trading Economics souscrit (lead time critique)
- DG-075 RFQ avocat fintech démarré

---

## 🎯 Décisions structurantes actées 2026-05-26

Trois corrections post-validation utilisateur :

1. **Toggle 3 modes (DG-100) DROP** → architecture progressive uniforme
   Un seul layout responsive, sections collapsibles tier-gated, hero card permanent. Économie ~30-40h dev.

2. **Filtre P0-strict-MVP : 10 items** (vs 13 P0 brut)
   Distinction stricte "bloque le 1er paiement" vs "accélère conversion/rétention".

3. **Analytique core (DG-160 + DG-161) basculée V1**
   Sans Plausible + event tracking en V1, les 10 DEFER deviennent invisibles donc inopérants. ~16-20h supplémentaires V1.

---

## 📊 Effort total plan révisé

| Vague | Période | Items | Effort |
|---|---|---|---|
| **Vague 1 — MVP commercialisable** | S1-S6 (6 sem) | 40 items dont 10 P0-strict-MVP | ~240-280h |
| **Vague 2 — Activation acquisition** | S7-S14 (8 sem) | 20 items | ~180h |
| **Vague 3 — Solidification** | S15+ (conditionnel) | 17 items | conditionnel |
| **Hors-vague (DROP)** | — | 4 items | — |
| **TOTAL** | 14+ semaines | **93 items** | **~420-460h Vague 1+2** |

---

## 🔗 Documents externes référencés

- `reports/commercialization_sprint/00_DANGEROUS_CHANGES.md` — plan audité (69 items DG)
- `docs/value/client_information_explained.txt` — vérité terrain algo (InsightSignalV2)
- `docs/value/client_relevance_review.md` — Livrable 1 (pertinence client)
- `docs/value/best_product_concept.md` — Livrable 2 (concept produit B retenu)
- `docs/value/information_enrichment_recommendations.md` — Livrable 4 (24 recos enrichissement)
- `mockups/v3/best_concept_demo.html` — Livrable 3 (démo HTML, à refaire)

---

**Auteur** : second instance Claude Code, revue critique 2026-05-26.
**État** : ✅ Mission complète. En attente signature 5 décisions politiques pour démarrage Vague 1.
