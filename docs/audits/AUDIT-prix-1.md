# AUDIT PRIX-1 — Tarification unique 39 $ US / 348 $ US (29 $ US par mois)

**Branche :** `chore/prix-1-tarification` (worktree dédié `C:/MyPythonProjects/wt-prix-1`, depuis `origin/main` d90cbd3)
**Date :** 2026-08-01

---

## 0. Tarification retenue

| Cadence | Prix facturé | Équivalent mensuel | Devise |
|---------|--------------|--------------------|--------|
| Mensuel | **39,00 $ US / mois** | 39 $ US | USD |
| Annuel  | **348,00 $ US / an** | **29 $ US / mois** | USD |

Un seul palier payant. FREE conservé côté backend (les démos de la landing sont la surface gratuite ; aucune carte FREE affichée — décision fondateur). Devise **USD partout**, y compris clients canadiens. **Aucune taxe.** Aucune remise, aucun prix barré, aucun compte à rebours.

L'affichage annuel montre **le total ET l'équivalent mensuel** (« 348 $ US par an, soit 29 $ US par mois ») — jamais 29 $ seul.

---

## 1. Diagnostic — le vrai problème : 3 systèmes de prix incohérents

Avant PRIX-1, trois définitions de prix coexistaient, sur **deux devises** et des **clés de plan divergentes** :

| Système | Emplacement | Prix | Devise | Clés |
|---------|-------------|------|--------|------|
| A — Vitrine + SEO + panneau abonnement | front | 49,99 mensuel / 39,99 annuel (479,88/an) −20 % | **USD** ($) | plan unique |
| B — Grille backend `src/billing/pricing.py` | `/api/v1/billing` | FREE €0 · LITE €19 · PRO €39 · PRO+ €99 · B2B €499/1500/3000 | **EUR** | LITE/PRO/PRO_PLUS/B2B_* |
| C — Route compte `account_billing.py` | `/api/billing` (utilisée par le webapp) | montants non codés (lus de Stripe) | — | STANDARD / PREMIUM |

De plus le front `SubscriptionPanel` attendait des clés `MONTHLY`/`ANNUAL` que **aucun** backend ne servait (fallback silencieux sur la clé brute).

Un 4ᵉ jeu de montants existait dans `src/api/tier_manager.py` (`price_usd` 49/99/149) — **métadonnée morte** (jamais lue ni sérialisée) du système d'accès RL hérité, jamais affichée. Laissée en place (hors périmètre PRIX-1, aucun impact d'affichage) — signalée ici.

---

## 2. Source unique retenue

**`config/pricing.json`** est désormais l'UNIQUE endroit où vivent les montants.

```
config/pricing.json          ← montants (39, 348) + devise (USD) + noms des variables Stripe
  ├─ backend : src/billing/pricing.py lit le JSON directement
  └─ frontend : scripts/gen_pricing.mjs → webapp/lib/pricing.generated.ts (généré, committé)
```

- `annualPerMonth` (29) est **dérivé** (`348 / 12`), jamais écrit à la main ; le générateur refuse un total non divisible par 12.
- Patron identique à celui des unités de temps (`config/timeframes.json` → `gen_timeframes.mjs`).
- Test-garde : `webapp/components/landing/__tests__/pricing-prix-1.test.ts` vérifie la synchro JSON↔généré et l'absence de tout montant codé en dur.

---

## 3. Emplacements modifiés

### Source & génération
- `config/pricing.json` — **NOUVEAU** (source unique).
- `scripts/gen_pricing.mjs` — **NOUVEAU** (générateur + `--check`).
- `webapp/lib/pricing.generated.ts` — **NOUVEAU** (généré ; `PRICING = {currency:'USD', monthly:39, annualPerYear:348, annualPerMonth:29}`).

### Backend
- `src/billing/pricing.py` — réécrit : grille 4-tiers EUR → **un plan USD** (FREE + MONTHLY + ANNUAL), lit `config/pricing.json`.
- `src/billing/__init__.py` — exports mis à jour (PLAN_*, PricingPlan, get_plan, list_plans, list_paid_plans, currency).
- `src/billing/stripe_client.py` — `price→plan` (au lieu de `price→tier`) ; `StripeWebhookEvent.tier_key` → `plan_key` ; `allow_promotion_codes` **False** (aucune remise) ; docstring tax nettoyée.
- `src/api/routes/billing.py` (`/api/v1/billing`) — `tier_key`→`plan_key`, `/pricing` renvoie `{plans:[…]}`.
- `src/api/routes/account_billing.py` (`/api/billing`, utilisé par le webapp) — plans dérivés de la source unique (`list_paid_plans`), clés **MONTHLY/ANNUAL** (au lieu de STANDARD/PREMIUM) ; **taxe retirée** (`_tax_enabled`, `tax_enabled` du `/pricing`, `automatic_tax` du checkout).

### Frontend
- `webapp/components/landing/PricingSection.tsx` — consomme `PRICING` ; **badge −20 % et « 120 $ d'économie » retirés** ; annuel affiche le **total (348) + équivalent (29)** ; **5 mentions** ajoutées à côté du prix.
- `webapp/components/seo/JsonLd.tsx` — offres SEO depuis `PRICING` (39/348 USD).
- `webapp/components/billing/SubscriptionPanel.tsx` — libellés de plan depuis `PRICING` + devise explicite.
- `webapp/lib/billing/api-client.ts` — champ `tax_enabled` retiré de l'interface `Pricing`.
- `webapp/app/[locale]/(site)/page.tsx` — commentaire de sommaire mis à jour.

### i18n (9 locales : fr, en, de, es, it, nl, pl, pt, ar)
- `landing.pricing` : `currency` `$`→`$ US`/`USD`/`دولار أمريكي` ; `discountBadge` **supprimé** ; `annualBilling` réécrit (total + équivalent, sans remise) ; **ajout** `perYear`, `mentionCancel`, `mentionCurrency`, `mentionRenewal`, `mentionEducational`, `mentionRisk`.
- `billing` : ajout `currency` ; `planMonthly`/`planAnnual` réécrits en gabarits `{amount}/{total}/{perMonth} {currency}` (plus aucun 49,99/39,99).

### Config & env
- `.env.example` — `STRIPE_PRICE_STANDARD/PREMIUM` → **`STRIPE_PRICE_MONTHLY` / `STRIPE_PRICE_ANNUAL`** ; **bloc `STRIPE_TAX_ENABLED` supprimé** (taxe interdite).

### Tests
- `tests/test_billing.py` — réécrit pour le modèle plan unique + assertions source unique (39/348/29, USD) + « aucune taxe ».
- `tests/test_account_billing.py` — STANDARD → MONTHLY ; assertion « pas de `tax_enabled` ».
- `webapp/components/landing/__tests__/pricing-prix-1.test.ts` — **NOUVEAU** garde.
- `webapp/tests/e2e/landing.spec.ts` — assertions prix mises à jour (348/29/39 $ US, absence 49,99/39,99/−20 %, mentions visibles).

---

## 4. Variables d'environnement à créer (tableau de bord Stripe → puis env de déploiement)

Crée **deux prix récurrents USD** dans Stripe et renseigne :

| Variable | Valeur | Prix Stripe à créer |
|----------|--------|---------------------|
| `STRIPE_PRICE_MONTHLY` | `price_…` | Récurrent **mensuel**, **39,00 USD** |
| `STRIPE_PRICE_ANNUAL`  | `price_…` | Récurrent **annuel**, **348,00 USD** |

Rappels :
- **Ne pas** activer Stripe Tax sur ces prix (aucune taxe). `STRIPE_TAX_ENABLED` n'existe plus.
- Les codes promo sont désactivés au Checkout (`allow_promotion_codes=False`).
- `STRIPE_SECRET_KEY` / `STRIPE_WEBHOOK_SECRET` restent requis (déjà documentés).
- Anciennes variables `STRIPE_PRICE_STANDARD` / `STRIPE_PRICE_PREMIUM` : **à supprimer** de l'environnement de déploiement (plus lues).

---

## 5. Endroits où un ancien prix subsistait (tous corrigés)

- `PricingSection.tsx` : `49,99` / `39,99` / `479,88` codés en dur → source unique.
- `JsonLd.tsx` (SEO) : `49.99` / `39.99` USD → source unique.
- i18n × 9 : `landing.pricing.annualBilling` (479,88 + −20 % + « 120 $ d'économie »), `billing.planMonthly/planAnnual` (49,99 / 39,99).
- `src/billing/pricing.py` : grille EUR 19/39/99/499/1500/3000 → plan unique USD.
- `.env.example` : bloc taxe + variables STANDARD/PREMIUM.
- Playwright `landing.spec.ts` : assertions 39,99 / 49,99.

**Hors périmètre (signalé, non touché) :** `src/api/tier_manager.py` `price_usd` 49/99/149 — métadonnée morte du système d'accès hérité, jamais affichée.

---

## 6. Vérifications

- Backend : `pytest tests/test_billing.py tests/test_account_billing.py` → **34 passed**.
- `node scripts/gen_pricing.mjs --check` → **up to date** (généré synchro avec la source).
- `tsc --noEmit` → **0 erreur**.
- `vitest` garde PRIX-1 (`pricing-prix-1.test.ts`) → **58 passed** (synchro source, aucun prix périmé, devise explicite × 9 locales, aucune taxe, aucune remise, 4 mentions fr+en).
- `vitest` `claims-cleanup.test.ts` → **18 passed** (aucun claim interdit réintroduit ; timeout 5 s pré-existant sous charge, vert avec `--testTimeout=30000`).
- `next build` → **exit 0**.
- Playwright `landing.spec.ts -g pricing`, projets **chromium-desktop (1280×720)** et **mobile-iphone-12 (390×844)**, **fr + en** → **4 passed** : annuel = total 348 $ US + équivalent 29 $ US ; mensuel = 39 $ US ; aucun 49,99/39,99/479 ; aucun −20 % ; mentions « Annulable à tout moment » / « Prix en dollars américains » (et EN) visibles sans clic.

> Note environnement : `next.config` est `output: 'standalone'` → `next start` inutilisable ; e2e via `next dev` (préchauffé). Machine chargée → e2e lancés `--workers=1 --timeout=120000`.

## 7. Reste avant merge (confirmation live founder)

- Créer les 2 prix Stripe USD (39/mois, 348/an) et poser `STRIPE_PRICE_MONTHLY` / `STRIPE_PRICE_ANNUAL` dans l'environnement ; retirer `STRIPE_PRICE_STANDARD/PREMIUM` et `STRIPE_TAX_ENABLED` du déploiement.
- Vérifier en live la section tarifs (/#tarifs) fr + en et le panneau /abonnement.
- **MERGE sur `main` uniquement après confirmation.**
