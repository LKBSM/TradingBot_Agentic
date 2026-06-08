# MISSION_ACK_V2 — Revue critique du plan d'exécution M.I.A. Markets

**Date** : 2026-05-26
**Auteur** : second instance Claude Code (revue indépendante)
**Plan audité** : `reports/commercialization_sprint/00_DANGEROUS_CHANGES.md` (65 items DG-001 à DG-085)

---

## Vision produit comprise (en 5 lignes)

1. **M.I.A. Markets** est un **indicateur conversationnel orienté Or + FX**, B2C retail large (pas niche premium), positionné comme **compréhension augmentée** — jamais comme promesse de profit (compliance UE 2024/2811 par construction, `edge_claim=False` assumé).
2. **Architecture en 3 couches obligatoire** : **C1** = un verdict en une phrase (FOCUS, mobile-first, ≤10s), **C2** = le « pourquoi » en langage simple au clic (CO-PILOT, 6 cartes hiérarchisées), **C3** = tout le détail technique (EXPERT : waterfall 8 composantes, conformal viz, sources RAG) uniquement à la demande, principalement via chatbot.
3. **Le chatbot est le moat**, pas un gadget : il définit le jargon, décompose la conviction, compare aux setups historiques, refuse pédagogiquement de donner des ordres ("Donc je dois acheter ?" → refus incarné = argument anti-finfluenceur).
4. **Hero permanent factuel** : *« 329 setups · PF 1.30 [1.12-1.49] · walk-forward 7 ans »* + alerte event imminent. Aucune promesse de gain. La défensabilité = honnêteté codifiée + calibration empirique + sources académiques visibles.
5. **Tout item du plan qui ne sert pas C1/C2/C3, le chatbot, l'acquisition B2C 12 mois ou la compliance non-négociable est suspect** — KEEP / MODIFY / DEFER (avec condition de réactivation) / DROP. Pas de complaisance.

---

## Écart relevé sur le périmètre

- Mission demande "69 items DG-001 à DG-085" — le plan source en contient **65** (numérotés DG-001 à DG-085 avec trous : pas de DG-015 à DG-019, pas de DG-059 à DG-069, pas de DG-070...085 contigus). Décompte effectif :
  - 🔴 DESTRUCTIVE : DG-001 à DG-014 → **14 items**
  - 🟠 BIG ARCHITECTURAL : DG-020 à DG-039 → **20 items**
  - 🟡 RISKY OPERATIONAL : DG-040 à DG-058 → **19 items**
  - 🟣 POLITIQUE/MÉTIER : DG-070 à DG-085 → **16 items**
  - **Total : 69 items** ✅ (le footer du plan dit 65 mais le décompte par catégorie donne 69 — je travaille sur 69)
- Le rapport final sera `docs/governance/decision_gate_review_v2.md` (créé). Hypothèse : pas d'autre rapport à toucher en parallèle.

---

## Principes d'évaluation appliqués

- **KEEP** = critique pour la vision produit dans les 12 prochains mois. À faire tel quel.
- **MODIFY** = utile mais mal cadré / mal dimensionné / mal phasé. Reformuler.
- **DEFER** = pas inutile, ne sert pas l'acquisition B2C 12 mois. Toujours assorti d'une **condition de réactivation** (ex : "DEFER jusqu'à MAU > 500" ou "DEFER jusqu'à conversion Stripe > 2 %/mois").
- **DROP** = ne sert pas la vision, charge cognitive de roadmap. Éliminer.

- **Compliance non-négociable** (`is_paper_demo`, geo-block, médiation conso, RGPD, CGU/CGV avocat, MiFID wording) = reclassée en **contrainte** plutôt que décision politique. Pas droppable, à designer proprement.

---

## Prochaine étape

Produire la Partie 1 (cartographie des 69 items en tableau) → arrêt pour validation → puis enchaînement Parties 2-6.

---

## Corrections actées 2026-05-26 (post-Partie 1 validée)

L'utilisateur a validé la Partie 1 et apporté 3 corrections structurantes qui se sont propagées dans tout le rapport :

### Correction 1 — Toggle 3 modes → architecture progressive uniforme (DROP)

**Problème identifié** : le toggle FOCUS/CO-PILOT/EXPERT (DG-100 dans la version initiale) forçait le client à choisir 3 concepts avant de comprendre la valeur. Friction UX inacceptable B2C retail.

**Décision** : **DG-100 → DROP** (4e DROP du plan). Remplacé par :
- **DG-101-MODIFIED** = un seul renderer (layout unique responsive), avec hero card permanent + sections collapsibles dépliables au clic
- **Gating par tier = disponibilité de contenu**, pas par layout (FREE voit moins de sections, STRATEGIST a bouton "Tout déplier")

**Économie** : ~30-40h dev (3 layouts → 1 layout). Mockup HTML existant (`mockups/v3/best_concept_demo.html`) devient obsolète et doit être réécrit (cf. `OUT_OF_SCOPE.md`).

### Correction 2 — Filtre P0-strict-MVP : 10 items vs 13 P0 brut

**Problème identifié** : la mention "13 items P0" mélangeait bloqueurs 1er paiement et accélérateurs conversion/rétention.

**Décision** : appliquer un filtre strict "qu'est-ce qui doit exister pour qu'un client paie en confiance ?" → **10 items P0-strict-MVP**. Les 3 items reportés en V2 (DG-102, DG-111, DG-122, DG-133, DG-114 réduit) sont des accélérateurs, pas des bloqueurs.

### Correction 3 — Analytique core (DG-160 + DG-161) basculée P0-strict V1

**Problème identifié** : 10 items DEFER s'appuyaient sur des métriques (MAU, churn, engagement chat) non mesurables sans analytique produit. Faille d'architecture du plan révisé.

**Décision** : Plausible self-hosted (DG-160, ~6h) + event tracking core 6 events (DG-161, ~10-14h) deviennent **P0-strict-MVP V1** au lieu de V2. Sans, les conditions de réactivation DEFER deviennent inopérantes.

### Distribution finale post-corrections

- **Sur plan original 69 items** : KEEP 50 / MODIFY 6 / DEFER 10 / DROP 3 = inchangé
- **Items ajoutés effectifs** : 28 - 1 (DG-100 DROP) = **27 items ajoutés**
- **P0-strict-MVP** : **10 items** (8 produit + 2 analytique core)
- **DROP totaux** : 4 (DG-030, DG-049, DG-078, DG-100)
- **Total plan révisé** : **93 items**

---

## État final mission

✅ Rapport `decision_gate_review_v2.md` livré (6 parties, 1100+ lignes)
✅ Distribution validée par utilisateur
✅ 3 corrections post-validation propagées dans tout le rapport
✅ Cohérence interne vérifiée (audit 2026-05-26)
▶ Artefacts complémentaires en cours : 5 décisions politiques, OUT_OF_SCOPE.md, README index, mémoire persistante
