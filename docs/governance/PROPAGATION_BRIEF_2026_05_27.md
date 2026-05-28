# Brief de propagation — Pivot positioning post-audit

**Date** : 2026-05-27
**Destinataires** : autres terminaux Claude Code (frontend webapp, légal low-cost, autres)
**Référence** : `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md`
**Statut** : actions à appliquer cette semaine

---

## 🚨 Contexte — En 30 secondes

L'audit `AUDIT_ALGO_2026_05_27.md` du 2026-05-27 a révélé que l'algorithme actuel (mode rule-based) a **zéro pouvoir prédictif** (Pearson −0.023, backtest 7 ans PF 0.786, return −62 %, sous-perf −318 pp vs Buy & Hold). Les tiers PREMIUM/STANDARD/WEAK sont empiriquement vides (1-0 trades en 7 ans).

**Décision utilisateur Loukmane Bessam 2026-05-27** : B + C en parallèle :
- **B immédiat** = pivot positioning + pricing + retrait claims (CE BRIEF)
- **C en arrière-plan** = réparation algo Sprint 1 augmenté (briefs déjà à jour dans `vague1_execution/briefs/`)

**Toi (autre terminal) tu dois appliquer le volet B au code que tu touches.**

---

## ✅ Actions à appliquer

### 1. Repositionnement du produit

**Avant** : système d'analyse / signaux de trading / "PF 1.30"
**Après** : **outil de compréhension augmentée**

Cherche dans tout le code/copies que tu touches :
- "système de trading" / "trading system" → **outil de compréhension augmentée** / "educational analysis tool"
- "signaux" / "signals" → **lectures** / "analyses" / "readings"
- "outil d'analyse algorithmique" → **outil pédagogique d'analyse algorithmique**

### 2. Pricing révisé : FREE / 9 € / 19 €

Remplace partout :
| Avant | Après |
|---|---|
| FREE / $29 / $79 / $1990 | **FREE / 9 € / 19 €** (INSTITUTIONAL retiré grille publique) |
| STARTER ($29/mo) | **Découverte (9 €/mo)** |
| PRO ($79/mo) | **Approfondie (19 €/mo)** |
| INSTITUTIONAL ($1990/mo) grille publique | Retiré → **"Contact us" / Calendly link** |

Variables d'env à mettre à jour (frontend) :
```bash
# Anciens (à retirer)
NEXT_PUBLIC_PRICE_STARTER_MONTHLY=2900
NEXT_PUBLIC_PRICE_PRO_MONTHLY=7900
NEXT_PUBLIC_PRICE_INSTITUTIONAL_MONTHLY=199000

# Nouveaux
NEXT_PUBLIC_PRICE_DECOUVERTE_MONTHLY_EUR=900
NEXT_PUBLIC_PRICE_DECOUVERTE_YEARLY_EUR=9000
NEXT_PUBLIC_PRICE_APPROFONDIE_MONTHLY_EUR=1900
NEXT_PUBLIC_PRICE_APPROFONDIE_YEARLY_EUR=19000
# INSTITUTIONAL : pas de price ID public, lien Calendly
NEXT_PUBLIC_CALENDLY_INSTITUTIONAL_URL=https://calendly.com/mia-markets/demo
```

### 3. Suppression des tiers algorithmiques PREMIUM/STANDARD/WEAK

Les **tiers de qualité de signal** PREMIUM/STANDARD/WEAK sont **retirés du visible client**.

Dans le webapp :
- Retire les composants `<PremiumBadge>` / `<StandardBadge>` / `<WeakBadge>` ou équivalents
- Retire le mapping conviction → tier dans l'affichage utilisateur
- Le score 0-100 reste visible MAIS sans étiquette tier
- En interne backend : tier peut rester pour debug/monitoring (champ caché)

Pour le contrat InsightSignalV2 :
- `signal.conviction_label` (weak/moderate/strong/institutional) → reste dans le contrat backend
- Mais ne plus l'afficher en UI au client
- Mockup HTML : `<span class="hero-strength">Strong</span>` → retirer ou neutraliser

### 4. Retrait claims de performance non substanciés

**À RETIRER ABSOLUMENT** dans tout le code/copies que tu touches :
- "Profit factor 1.30"
- "1.30 [1.12-1.49]"
- "1.30 € gagnés pour 1 € perdu"
- "IC 95 %" lié à des chiffres de performance
- "329 setups historiques"
- "Win rate 31.9 %" / "41.6 %"
- "Drawdown max 8.4 %"
- "Track record honnête : PF 1.30"
- "Backtest 7 ans validé walk-forward" (qualifier en "in-sample, OOS pending")

**À CONSERVER** (factuel non-performance) :
- "8 facteurs analysés"
- "12 papers académiques sources"
- "7 années de données historiques"
- "2 actifs : XAU + EUR/USD"
- "Pipeline 5 briques scientifiques"
- "Méthodologie publique"
- "Phase d'accès anticipé"

**À AJOUTER** sur les pages méthodologie / track-record :
- *"Validation statistique OOS en cours. Sprint 1 du plan dev. Métriques détaillées publiables après validation Brier skill > +2 % AND DSR > 1.0 AND PBO < 0.5."*

### 5. Nouvelles taglines / wording

**Hero principal landing** :
> **Comprenez les marchés Or et FX.**
> **Décidez en autonomie.**

**Sub-tagline** :
> M.I.A. Markets est un outil pédagogique d'analyse algorithmique des marchés Or et FX. Il décompose la lecture du marché en couches, vous laisse poser toutes vos questions via le chatbot Sentinel, et refuse pédagogiquement de vous donner des ordres. Vous restez seul décideur.

**4 stats hero (révisé)** :
- **8** facteurs analysés
- **12** papers académiques sources
- **7 ans** de données historiques
- **2 actifs** XAU + EUR/USD

**Footer compliance** :
> Outil pédagogique d'analyse algorithmique · Phase d'accès anticipé · Ne constitue ni un signal de trading, ni un conseil en investissement, ni une recommandation.

### 6. Chatbot Q&A scriptées

Si tu touches le code des Q&A scriptées du chatbot (`/api/chat/responses` ou similaire), regarde le fichier référence à jour : `docs/governance/vague1_execution/copies/chatbot_scripted_responses.md`.

En particulier la **Q4 "Ça ressemble à quoi historiquement ?"** a été **entièrement réécrite** pour retirer PF 1.30 / IC / 329 setups / 41.6 % win rate.

Le nouveau ton :
- Honnêteté sur la non-validation OOS
- Renvoi vers Sprint 1 / page méthodologie
- Pas de chiffre de performance

### 7. Pages à mettre à jour

| Page | Action |
|---|---|
| `/` (landing) | Hero + 4 stats + différenciateurs + footer (cf. `landing_copy.md`) |
| `/pricing` | 3 tiers FREE/9€/19€ + INSTITUTIONAL → Calendly (cf. `pricing_copy.md`) |
| `/methodologie` | Section "Métriques agrégées" : claims retirés, replaced par "OOS pending" |
| `/track-record` | Disclaimer fort en haut + retrait PF/IC affichés tant que pas OOS validé |
| `/signup` | "Découvrir gratuitement" pas "S'inscrire" |
| Footer toutes pages | Compliance + lien `/methodologie` + lien `/cgu` |

### 8. Composants UI à toucher

| Composant | Action |
|---|---|
| `<HeroCard>` | Retirer `<TrackRecordBadge>` montrant "PF 1.30" |
| `<TierBadge>` (PREMIUM/STANDARD/WEAK) | Retirer du visible client (rester backend uniquement) |
| `<PricingGrid>` | 3 cards au lieu de 4, prix en EUR |
| `<DiffCard #1>` (track record) | Changer pour "Méthodologie publique transparente" |
| `<MetricsSection>` page méthodologie | Retirer 5 lignes PF/IC/Win/DD/Skew, replacer par disclaimer OOS |

---

## 🚫 Ce que tu NE FAIS PAS

- ❌ Ne touche pas au code du moteur algorithmique backend (`src/intelligence/`) — c'est le terminal qui fait Sprint 1
- ❌ Ne change pas la structure InsightSignalV2 (le contrat reste v2.1.0)
- ❌ Ne supprime pas les sources RAG (12 papers académiques) — ils sont l'argument différenciateur
- ❌ Ne change pas la structure progressive uniforme (hero + sections collapsibles)
- ❌ Ne touche pas aux briefs Sprint 1 (déjà à jour avec Blockers #2/#3/#4)

---

## 📚 Fichiers de référence

Tu peux consulter pour les nouvelles copies exactes :

| Fichier | Contenu |
|---|---|
| `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md` | Décision officielle + justification |
| `docs/governance/AUDIT_ALGO_2026_05_27.md` | Audit complet de l'algorithme |
| `docs/governance/vague1_execution/copies/landing_copy.md` | Nouvelle landing copy |
| `docs/governance/vague1_execution/copies/pricing_copy.md` | Nouvelle pricing copy + 3 tiers |
| `docs/governance/vague1_execution/copies/methodologie_copy.md` | Méthodologie révisée |
| `docs/governance/vague1_execution/copies/chatbot_scripted_responses.md` | Q4 réécrite |
| `docs/governance/vague1_execution/copies/emails_lifecycle.md` | Email D+7 réécrit |
| `docs/governance/MASTER_PLAN.md` | Plan dev avec Sprint 1 augmenté Blockers #2/#3/#4 |

---

## ⏱️ Délai

**Application Volet B : cette semaine (S0-S1).**

Dans 4 semaines (~2026-06-24), si Sprint 1 réussit la validation OOS (Brier > +2 %, DSR > 1.0, PBO < 0.5), il y aura un **second brief de propagation** pour rétablir les claims chiffrés et possiblement augmenter le pricing.

D'ici là : posture pédagogique honnête, prix faibles, pas de claim non substancié.

---

## ✅ Confirmation attendue

Quand tu as terminé l'application de ce brief sur ton terminal :

1. Mets à jour `docs/governance/vague1_execution/PROGRESS.md` avec l'entrée `2026-05-27_propagation_brief_applied`
2. Liste les fichiers/composants touchés
3. Liste ce qui reste à valider (ex : si attente d'env vars Stripe pour finaliser pricing config)

**FIN DU BRIEF.**
