# HANDOFF FRONTEND — Pivot positioning post-audit (2026-05-27)

**De** : terminal backend/governance (Claude Code instance #1)
**Pour** : terminal frontend (Claude Code instance #2 — celle qui touche `webapp/`)
**Émis le** : 2026-05-27
**Statut** : à appliquer cette semaine (S0-S1)

---

## 🎯 Ta mission en 1 phrase

Applique intégralement le brief `docs/governance/PROPAGATION_BRIEF_2026_05_27.md` sur `webapp/` (et uniquement `webapp/`).

---

## ⚡ TL;DR — 30 secondes

L'audit algo 2026-05-27 a révélé que le scoring rule-based a **zéro pouvoir prédictif** (Pearson −0.023, backtest 7 ans PF 0.786). Décision Loukmane = pivot B+C parallèle :

- **Volet B (toi)** : repositionner le produit en "outil de compréhension augmentée" + pricing FREE/9€/19€ + retirer TOUS les claims chiffrés de performance + supprimer les badges tier PREMIUM/STANDARD/WEAK de l'UI.
- **Volet C (autre terminal)** : Sprint 1 algo augmenté Blockers #2/#3/#4 → **NE PAS toucher à `src/intelligence/`**.

---

## ✅ Actions concrètes (résumé)

1. **Repositionnement** : "outil de compréhension augmentée" (pédagogique 100 %). Remplace "système de trading" / "signaux" → "lectures" / "analyses pédagogiques".

2. **Pricing 3 tiers** : `FREE / Découverte 9 €/mo / Approfondie 19 €/mo`. INSTITUTIONAL retiré de la grille publique → lien Calendly "Contact us".

3. **Env vars Stripe** à mettre à jour :
   ```bash
   # Retirer
   NEXT_PUBLIC_PRICE_STARTER_MONTHLY=2900
   NEXT_PUBLIC_PRICE_PRO_MONTHLY=7900
   NEXT_PUBLIC_PRICE_INSTITUTIONAL_MONTHLY=199000

   # Ajouter
   NEXT_PUBLIC_PRICE_DECOUVERTE_MONTHLY_EUR=900
   NEXT_PUBLIC_PRICE_DECOUVERTE_YEARLY_EUR=9000
   NEXT_PUBLIC_PRICE_APPROFONDIE_MONTHLY_EUR=1900
   NEXT_PUBLIC_PRICE_APPROFONDIE_YEARLY_EUR=19000
   NEXT_PUBLIC_CALENDLY_INSTITUTIONAL_URL=https://calendly.com/mia-markets/demo
   ```

4. **Composants UI** à retirer/neutraliser :
   - `<PremiumBadge>` / `<StandardBadge>` / `<WeakBadge>` ou équivalents → retirer du visible client (peuvent rester en backend pour debug)
   - `<TrackRecordBadge>` "PF 1.30" sur `<HeroCard>` → remplacer par badge "Méthodologie publique · OOS validation pending"
   - `<PricingGrid>` → 3 cards au lieu de 4, prix en EUR
   - `<MetricsSection>` page méthodologie → retirer les 5 lignes PF/IC/Win/DD/Skew, replacer par disclaimer "Validation OOS en cours"

5. **Claims à retirer absolument** dans tout code/copies que tu touches :
   - "Profit factor 1.30" / "1.30 [1.12-1.49]"
   - "IC 95 %" lié à des chiffres de performance
   - "329 setups historiques"
   - "Win rate 31.9 %" / "41.6 %"
   - "Drawdown max 8.4 %"
   - "$29 / $79 / $1990"
   - "Backtest 7 ans validé walk-forward" → qualifier "in-sample, OOS pending"

6. **Claims à conserver** (factuel non-performance) :
   - 8 facteurs analysés
   - 12 papers académiques sources
   - 7 années de données historiques
   - 2 actifs (XAU + EUR/USD)
   - Pipeline 5 briques scientifiques
   - Méthodologie publique
   - Phase d'accès anticipé

7. **Hero principal landing** (nouveau wording) :
   ```
   Comprenez les marchés Or et FX.
   Décidez en autonomie.
   ```

8. **Footer compliance** (toutes pages) :
   ```
   Outil pédagogique d'analyse algorithmique · Phase d'accès anticipé ·
   Ne constitue ni un signal de trading, ni un conseil en investissement,
   ni une recommandation.
   ```

---

## 🚫 Périmètre exclu — NE PAS TOUCHER

- ❌ `src/intelligence/` (autre terminal y est, volet C Sprint 1)
- ❌ Structure du contrat `InsightSignalV2` (v2.1.0 stable)
- ❌ Architecture progressive uniforme (hero permanent + sections collapsibles)
- ❌ Briefs Sprint 1 dans `docs/governance/vague1_execution/briefs/` (déjà à jour avec Blockers #2/#3/#4)
- ❌ Tests unitaires backend (`tests/test_insight_*.py` — fixtures internes)

---

## 📚 Sources de vérité (copies à jour 2026-05-27)

| Fichier | Contenu |
|---|---|
| `docs/governance/PROPAGATION_BRIEF_2026_05_27.md` | Brief complet détaillé (lis-le en entier avant de coder) |
| `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md` | Décision officielle + justification |
| `docs/governance/AUDIT_ALGO_2026_05_27.md` | Audit complet de l'algorithme (verdict 3/10) |
| `docs/governance/vague1_execution/copies/landing_copy.md` | Nouvelle landing copy |
| `docs/governance/vague1_execution/copies/pricing_copy.md` | Nouvelle pricing copy + 3 tiers |
| `docs/governance/vague1_execution/copies/methodologie_copy.md` | Méthodologie révisée |
| `docs/governance/vague1_execution/copies/chatbot_scripted_responses.md` | Q4 réécrite (chiffres OOS pending) |
| `docs/governance/vague1_execution/copies/emails_lifecycle.md` | Email D+7 réécrit, pricing 9€/19€ |
| `mockups/v3/best_concept_demo.html` | Mockup HTML à jour (référence visuelle) |

⚠️ Le mockup `mockups/v2/client_view_full.html` est OBSOLÈTE (figé pré-pivot) — ne pas s'en servir.

---

## ✅ Confirmation attendue quand tu as fini

1. Mets à jour `docs/governance/vague1_execution/PROGRESS.md` avec une entrée :
   ```
   ## 2026-05-27_propagation_brief_applied
   - Status: ✅ Done
   - Files touched: [liste]
   - Components changed: [liste]
   - Pending validation: [env vars Stripe live, etc.]
   ```

2. Liste explicitement les fichiers/composants touchés.

3. Liste ce qui reste à valider (par exemple : attente d'env vars Stripe pour finaliser pricing config, ou nouvelle URL Calendly à connecter).

---

## 📅 Délai et horizon

**Application volet B : cette semaine (S0-S1).**

Re-évaluation prévue le **2026-06-24** (4 semaines après pivot). Si Sprint 1 réussit la validation OOS (Brier skill > +2 % ET DSR > 1.0 ET PBO < 0.5), un second brief de propagation rétablira possiblement les claims chiffrés et augmentera le pricing premium.

D'ici là : posture pédagogique honnête, prix faibles, aucun claim non substancié.

---

**FIN DU HANDOFF.**

Référence complète si tu veux les détails exhaustifs : `docs/governance/PROPAGATION_BRIEF_2026_05_27.md`.
