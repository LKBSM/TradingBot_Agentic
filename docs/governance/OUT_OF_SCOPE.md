# OUT_OF_SCOPE — Sujets identifiés mais hors mission "revue critique"

**Date** : 2026-05-26
**Contexte** : mission `decision_gate_review_v2.md` = revue critique du plan d'exécution, **zéro modification de code**. Les sujets ci-dessous ont été identifiés pendant l'audit mais sortent du périmètre strict de la mission. Ils sont **listés ici pour qu'aucun ne soit oublié** lors de l'exécution du plan révisé par l'autre instance Claude Code (ou par toi).

---

## ✅ RÉSOLU (2026-05-26) — Mockup HTML refait en architecture progressive uniforme

Le mockup `mockups/v3/best_concept_demo.html` a été **réécrit en single-file HTML (1489 lignes, 60 KB) le 2026-05-26** avec l'architecture progressive uniforme. Hero card permanent + 4 sections collapsibles tier-gated + chatbot sidebar permanent / FAB mobile + footer compliance permanent + tier view switcher démo (FREE / Analyst / Strategist) + 6 dialogues chatbot scriptés incl. refus pédagogique.

**Ne plus traiter cet item.**

---

## 🗂️ Archive — Mockup HTML obsolète (texte original conservé pour traçabilité)

### Le problème

`mockups/v3/best_concept_demo.html` est **le support commercial visuel principal** du concept M.I.A. Markets. C'est ce qui sera montré :
- aux prospects en démo
- aux avocats fintech pour brief CGU
- aux partenaires éventuels
- aux candidats employés / freelance

**Il est aujourd'hui basé sur l'architecture toggle FOCUS/CO-PILOT/EXPERT qui a été ABANDONNÉE le 2026-05-26** au profit d'une architecture progressive uniforme (cf. `decision_gate_review_v2.md` Partie 2 Angle mort #1, post-révision).

**Le laisser tel quel = présenter une vision produit qui n'existe plus.**

### Ce qu'il faut faire

**Réécrire le mockup HTML avec l'architecture progressive uniforme** :

```
┌─────────────────────────────────────────────┐
│  HEADER                                     │
│  M.I.A. Markets · XAUUSD M15      │
├─────────────────────────────────────────────┤
│  HERO CARD permanent (toujours visible)     │
│  🟢 LECTURE HAUSSIÈRE · STRONG              │
│  329 setups · PF 1.30 [1.12-1.49] · 7 ans  │
│  ⚠ FOMC dans 2h47                          │
│  « Cassure haussière confirmée... »         │
│  [💬 Demander à Sentinel] [📊 Tout déplier] │
├─────────────────────────────────────────────┤
│  ⌄ Pourquoi cette conviction ? (collapsed)  │
│     ↳ Conviction calibrée + jauge          │
│     ↳ Marge d'erreur honnête               │
├─────────────────────────────────────────────┤
│  ⌄ Régime + volatilité (collapsed)          │
├─────────────────────────────────────────────┤
│  ⌄ Structure SMC (collapsed)                │
├─────────────────────────────────────────────┤
│  ⌄ Détail technique (collapsed, locked 🔒)  │
│     STRATEGIST+ : waterfall + conformal     │
│     viz + sources RAG                       │
├─────────────────────────────────────────────┤
│  CHATBOT SIDEBAR (desktop) / FAB (mobile)   │
│  Suggestions : 3 questions contextuelles    │
├─────────────────────────────────────────────┤
│  FOOTER compliance permanent                │
└─────────────────────────────────────────────┘
```

**Principes UX** :
- Hero card toujours visible, jamais masqué
- Sections collapsibles dépliables au clic (état initial : collapsed sauf hero)
- Gating tier = section "Détail technique" verrouillée (🔒) pour FREE/STARTER, accessible PRO/STRATEGIST
- Bouton "Tout déplier" pour les STRATEGIST+ (1 clic = tout ouvert d'un coup)
- Mobile : même structure, sections verticales scroll (pas de sidebar parallèle, FAB chatbot)
- Chatbot : sidebar permanente desktop ≥ 1024px, FAB mobile < 1024px

**Inspirations design** : Bloomberg Terminal (sobriété), Linear (densité élégante), Stripe (clarté), Pitchbook (track record honnête en hero).

**Single-file HTML** : pas de CDN, pas de framework. Tailwind inliné OK. JavaScript vanilla pour les collapse + chatbot scripté (6 questions/réponses pré-écrites).

### Pourquoi c'est hors scope de MA mission

Ma mission = revue critique du plan d'exécution, livraison de markdown analytique. Toucher au code/HTML = autre mission, autre instance Claude Code, ou toi.

### Priorité

**P0 — à faire avant toute démo commerciale.** Sinon, tu présenteras une vision obsolète à des prospects ou des avocats. Estimation effort : 8-12h (single-file HTML + 6 dialogues chatbot scriptés).

### Brief pour l'instance qui le fera

> "Réécris `mockups/v3/best_concept_demo.html` en architecture progressive uniforme. Lis `docs/governance/decision_gate_review_v2.md` Partie 2 Angle mort #1 (post-révision 2026-05-26) pour le détail. Garde le contenu fonctionnel (BOS 2391.5, FVG 2378-2381, conviction 72, régime trend bullish, FOMC dans 2h47, 6 dialogues chatbot incl. refus pédagogique). Change la structure : 1 layout responsive avec hero permanent + sections collapsibles tier-gated, plus de toggle 3 modes. Sortie attendue : single-file HTML sans CDN, Tailwind inliné OK, JS vanilla."

---

## 📄 SOP `docs/runbooks/circuit_breaker_tuning.md` (DG-049 DROP)

### Le problème

DG-049 a été DROP du plan d'exécution parce que ce n'est pas un livrable — c'est un process opérationnel continu (chaque modif circuit-breaker = revue analyse historique false-positive rate). Mais le process **doit exister quelque part**, sinon les seuils CircuitBreaker vont dériver sans contrôle.

### Ce qu'il faut faire

Créer `docs/runbooks/circuit_breaker_tuning.md` avec :
- Quand modifier (déclencheurs : incident, false positive observé, baseline shift)
- Comment modifier (procédure : sortir historique 30j, analyser FP rate, proposer seuil, peer review, déploiement progressif)
- Qui valide (responsable infra)
- Logs à conserver

Effort : ~2-3h, markdown pur.

### Pourquoi hors scope

Runbook ops = production de documentation d'exploitation. Différent de la revue critique stratégique.

### Priorité

**P2 — à faire avant la première modif circuit-breaker post-launch.** Pas bloquant Vague 1, mais bloquant scaling Vague 3.

---

## 🛠 Exécution du plan révisé v2

### Le problème

Le rapport `decision_gate_review_v2.md` livre **la cartographie + le séquencement + les décisions politiques**. Il ne livre **pas l'exécution** :
- Pas de PR créées
- Pas de tests écrits
- Pas de code touché
- Pas de Stripe configuré

L'exécution est la mission de l'autre instance Claude Code (qui travaille en parallèle dans un autre terminal) ou de toi.

### Brief pour l'instance qui exécutera

> "Lis `docs/governance/decision_gate_review_v2.md` Partie 4 (re-séquençage 3 vagues) et `docs/governance/decisions/2026-05-26_5_political_locks.md` (5 décisions politiques signées). Démarre Vague 1 S1 selon l'ordre décrit. Les 10 P0-strict-MVP sont :
> 1. DG-101-MODIFIED renderer unique sections collapsibles
> 2. DG-103 mobile-first responsive
> 3. DG-110 wire chatbot 8 composantes
> 4. DG-112 tests adversariaux refus pédagogique
> 5. DG-114-REDUCED 3 questions suggérées
> 6. DG-120 landing hero card track-record
> 7. DG-132 page pricing decoy + dual trial
> 8. DG-142 tableau performance public mensuel
> 9. DG-160 Plausible self-hosted
> 10. DG-161 event tracking core 6 events
>
> Effort cumulé P0-strict-MVP : ~112-160h. Vague 1 totale : ~240-280h sur 6 semaines."

### Priorité

**Démarrage S1 (2026-06-01) conditionné à la signature des 5 décisions politiques.**

---

## 🔍 Vérification suivi mémoire persistante

### Sujet

Une mémoire produit doit être sauvegardée dans `~/.claude/projects/.../memory/MEMORY.md` pour que :
- Une future instance Claude Code retrouve le contexte (4 DROP, 10 P0-strict, architecture progressive uniforme, etc.)
- 3 mois plus tard, toi-même puisses comprendre l'arborescence en 30 secondes

### Statut

✅ Tâche prévue dans le plan de finalisation (Étape 6). Sera créée dans la même session.

### Priorité

P1 — sans cela, perte de contexte garantie à la prochaine session.

---

## 📋 Récap priorités

| Sujet | Priorité | Effort | Type | Responsable |
|---|---|---|---|---|
| **🚨 Réécriture mockup HTML** | **P0** | 8-12h | Code/HTML | Autre instance Claude Code OU toi |
| Exécution Vague 1 S1-S6 | P0 (post-signature) | ~240-280h | Code + ops | Autre instance Claude Code |
| Mémoire persistante | P1 | 5 min | Memory file | Cette session (en cours) |
| SOP CircuitBreaker | P2 | 2-3h | Markdown | À planifier |

---

**Aucun de ces sujets ne doit être oublié. Le plus critique : le mockup HTML doit être refait AVANT toute démo commerciale, sinon tu vendras une vision obsolète.**
