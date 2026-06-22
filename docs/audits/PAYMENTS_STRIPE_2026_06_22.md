# Paiements & abonnements (Stripe) — Mission ② — 2026-06-22

Branche : `feat/payments-stripe` (worktree dédié `wt-payments-stripe`), basée sur
`feat/account-auth-legal-shell` (Mission ① — pas encore mergée dans `main`).

## 1. Cadre & sécurité (rappel)
- Clés Stripe **en `.env` uniquement**, jamais en dur.
- Webhooks à **signature vérifiée** avant toute mutation d'état.
- **Aucune donnée de carte stockée** : Checkout + Customer Portal sont hébergés
  par Stripe ; seules les **IDs Stripe** et l'**état d'abonnement** sont persistés.
- Essai gratuit **en option, désactivé par défaut** (`STRIPE_TRIAL_DAYS=0`).
- Renouvellement automatique géré par Stripe (mode `subscription`).

## 2. Décisions (validées avant implémentation)
1. **Base de branche** : sur `feat/account-auth-legal-shell` (rebase final sur
   `main` après le merge de ①).
2. **Grille tarifaire** : prix **configurables via env** (`STRIPE_PRICE_*`) —
   libellés/montants décidés au dashboard Stripe, rien de figé en code.
3. **Legacy `/api/v1/billing`** (tier-keyed, `UserTierManager`, B2B) : **laissé
   coexister**. La nouvelle surface compte est séparée (`/api/billing/*`).

## 3. État des lieux (diagnostic)
Un scaffold Stripe legacy existait déjà mais **inerte** et branché sur le mauvais
système d'identité (`UserTierManager` + clés API, join par email) :
- `src/billing/stripe_client.py` (`StripeClient`, `verify_webhook`) — **solide,
  réutilisé**.
- `src/api/routes/billing.py` `/api/v1/billing/*` — laissé tel quel (B2B).
- `stripe_client` n'était **jamais instancié** → 503. `stripe` **absent** de
  requirements. Aucune page webapp d'abonnement.

## 4. Implémentation livrée

### Backend
| Fichier | Changement |
|---|---|
| `requirements.txt` | + `stripe>=9.0.0` |
| `.env.example` | section Stripe : `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_STANDARD/PREMIUM`, `STRIPE_TRIAL_DAYS`, `STRIPE_TAX_ENABLED`, `STRIPE_SUCCESS_URL`, `STRIPE_CANCEL_URL`, `STRIPE_PORTAL_RETURN_URL` |
| `src/api/account_store.py` | schema **v1→v2** : tables `subscriptions` (1 ligne/compte) + `processed_webhooks` (idempotence). Méthodes : `get_subscription`, `link_stripe_customer`, `get_account_by_stripe_customer`, `upsert_subscription`, `mark_webhook_processed` |
| `src/billing/stripe_client.py` | + `create_customer` (metadata `account_id`), `create_billing_portal_session`, `automatic_tax` au Checkout, `create_checkout_session(customer=…, account_id=…)` ; + parseur **account-centric** `parse_account_event` + `AccountSubscriptionEvent` |
| `src/api/routes/account_billing.py` | **NOUVEAU** routeur `/api/billing/*` lié à la session compte |
| `src/api/subscription_gate.py` | `account_has_access(account, store)` lit l'état réel (owner toujours OK ; sinon `status ∈ {active, trialing}` ET non expiré). Garde-fou env `SUBSCRIPTION_GATE_ENFORCED` (défaut OFF = phase test perso) |
| `src/api/app.py` | enregistre le routeur + instancie `StripeClient` quand `STRIPE_SECRET_KEY` est défini |
| `src/api/routes/accounts.py` | `/api/auth/access` passe le store à `account_has_access` |

### Endpoints (`/api/billing/*`, surface compte)
- `GET  /pricing` — public ; plans dont le price id est configuré + `trial_days` + `tax_enabled`.
- `POST /checkout` — auth ; crée/réutilise le customer Stripe du compte, session Checkout, renvoie l'URL hébergée.
- `POST /portal` — auth ; session Customer Portal (gérer/annuler/changer de carte).
- `GET  /subscription` — auth ; état courant + accès résolu.
- `POST /webhook` — Stripe ; **signature vérifiée**, **idempotent** (1 event id = 1 application), met à jour `subscriptions`.

### Flux d'événements webhook → compte
`checkout.session.completed` (lie customer↔compte) ·
`customer.subscription.created/updated/deleted` (état complet ; deleted→`canceled`) ·
`invoice.paid` / `invoice.payment_succeeded` (→`active`) ·
`invoice.payment_failed` (→`past_due`).
Résolution du compte par `metadata.account_id`, sinon par `stripe_customer_id`.
Un event **non résoluble n'est pas "consommé"** → le retry Stripe peut aboutir
une fois le customer lié.

### Taxes TPS/TVQ
**Stripe Tax** via `automatic_tax` au Checkout (drapeau `STRIPE_TAX_ENABLED`,
défaut OFF). Requiert l'enregistrement des taxes au dashboard Stripe (config, pas
code). Quand activé, Checkout collecte l'adresse et la sauvegarde sur le customer
(`customer_update.address=auto`).

### Frontend (webapp)
- `webapp/lib/billing/api-client.ts` — client (pricing, subscription, checkout, portal).
- `webapp/components/billing/SubscriptionPanel.tsx` — état, souscription, portail, retours `?status=success|cancel`.
- `webapp/app/[locale]/abonnement/page.tsx` — page (Suspense pour `useSearchParams`).
- `webapp/components/app/AccountMenu.tsx` — lien « Abonnement ».

## 5. Tests & build
- **`tests/test_account_billing.py` (14 tests, tous verts)** — Stripe simulé en
  mémoire (aucune clé/réseau) :
  - checkout crée/réutilise le customer + renvoie l'URL ; auth requise ; prix
    non configuré refusé (400) ; portal 409 sans customer puis 200 ;
  - `subscription.created` → **état actif** ; annulation → `canceled` ;
    `payment_failed` → `past_due` ; gate enforced n'accorde que `active`/`trialing`
    non expiré ;
  - **signature invalide → 400** (aucune écriture) ; signature manquante → 400 ;
    **idempotence** (même event id appliqué une seule fois) ; event non résolu
    non consommé.
- **Régression** : `tests/test_account_auth.py` + `tests/test_billing.py` =
  42 verts. Webapp : `tsc --noEmit` OK, `next build` **vert** (route `/abonnement`
  présente), `vitest` 257/257.
- **Pré-existant (hors périmètre)** : `test_smoke_e2e.py::test_api_health…` et
  `…test_api_narratives_no_auth…` échouent **aussi sur la base** (scanner non câblé
  dans `create_app`) — vérifié par `git stash`.

## 6. Mise en service (rappel opérateur)
1. `STRIPE_SECRET_KEY` (test : `sk_test_…`) + `STRIPE_WEBHOOK_SECRET` (`whsec_…`).
2. Créer les prix au dashboard → renseigner `STRIPE_PRICE_STANDARD` / `_PREMIUM`.
3. (Option) `STRIPE_TRIAL_DAYS=14`, `STRIPE_TAX_ENABLED=1` (+ enregistrements taxe).
4. Pour activer le paywall : `SUBSCRIPTION_GATE_ENFORCED=1` (sinon ouvert, phase
   de test perso ; l'owner passe toujours).
5. Endpoint webhook à déclarer côté Stripe : `POST /api/billing/webhook`.
