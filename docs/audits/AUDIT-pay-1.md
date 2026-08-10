# AUDIT — PAY-1 · Authentification & abonnement Stripe

_Branche `feat/pay-1-auth-abonnement` (worktree dédié, depuis `origin/main`)._
_Date : 2026-08-07 · Décision produit : **abonnement payant unique, aucun palier gratuit**._

> **Merge sur `main` uniquement après ta confirmation live ET un paiement de test
> réussi de bout en bout que tu auras constaté toi-même.** Ce document est le
> rapport demandé au STOP + la référence de déploiement.

---

## 0. Constat de départ : ce qui existait déjà

Contrairement à la prémisse « terrain vierge », l'auth ET Stripe étaient **déjà
bâtis et fusionnés sur `main`** par 3 missions antérieures (auth-hardening,
payments-stripe, subscription-gate). PAY-1 a donc été **de la finition + un
pivot**, pas une création :

- Auth maison **durcie** : Argon2id, sessions opaques révocables, cookie signé
  `HttpOnly`/`SameSite=Lax`/`Secure`, reset 1 h usage-unique, throttle login/reset,
  anti-énumération, `SESSION_SECRET` fail-fast en prod.
- Stripe : Checkout hébergé + Customer Portal + webhooks signés & idempotents.
- Point d'autorisation unique déjà présent (`subscription_gate`).

---

## 1. Fournisseur d'authentification — décision & justification

**Retenu : conserver et ÉTENDRE l'authentification maison** (pas de migration vers
Clerk / Supabase / Auth.js).

**Pourquoi.** La consigne « n'écris pas l'auth toi-même » vise la partie risquée —
hachage de mot de passe, sessions, réinitialisation. Or **cette partie est déjà
écrite, revue et durcie** (Argon2id, qui est plus fort que le bcrypt des
fournisseurs managés ; sessions opaques révocables = révocation instantanée, le
gold standard). Migrer signifierait **jeter du code qui marche** et introduire une
dépendance fournisseur + une vérification inter-services, pour un seul gain réel :
**Google OAuth + vérification e-mail** — que l'on peut ajouter à l'existant (fait
ici). 

**Comparaison (résumé).** Si l'on partait de zéro, **Supabase Auth** serait le
meilleur choix (Google + vérif e-mail intégrés, région **Montréal `ca-central-1`**
pour la Loi 25, vérification JWKS rapide côté FastAPI, ~0 $ à faible volume).
**Clerk** = US-only sans choix de région → transfert hors Québec permanent.
**Auth.js** = disqualifié pour un backend FastAPI séparé (session JWE non
vérifiable, on ré-écrit hachage/reset/vérif soi-même). Mais l'auth maison étant
déjà là et solide, l'étendre est le choix le plus sûr et le moins coûteux pour un
mainteneur solo.

**Nuance résidence de données :** garder l'auth « en interne » ne met PAS les
données au Québec — voir §6 (l'hébergement Render est en Oregon, US).

---

## 2. Schéma des données (SQLite `accounts.db`)

Une seule base, migrée par `SCHEMA_VERSION` (maintenant **5**).

```
accounts(
  id, username, username_lower UNIQUE, email, email_lower UNIQUE,
  password_hash (Argon2id), role ('user'|'owner'|…), age_confirmed,
  is_active, email_verified,            -- v5 : vérification e-mail obligatoire
  created_at )

account_consents( id, account_id→accounts, doc('terms'|'privacy'),
  version, accepted_at )                 -- Loi 25 : consentement horodaté + version

sessions( token_hash PK, account_id→accounts, created_at, expires_at )   -- opaques
password_resets( token_hash PK, account_id, created_at, expires_at, used_at )  -- 1 h
email_verifications( token_hash PK, account_id, created_at, expires_at, used_at ) -- 24 h, v5

subscriptions(                           -- ÉTAT d'abonnement, jamais de carte
  account_id PK →accounts, stripe_customer_id, stripe_subscription_id,
  status, price_id, current_period_end, cancel_at_period_end, trial_end,
  last_event_created,                    -- v4 : garde anti-désordre des webhooks
  updated_at )
processed_webhooks( event_id PK, event_type, processed_at )   -- idempotence
chat_usage( account_id, day, count )     -- vestige freemium, non lu (paid-only)
```

**Aucune donnée de carte** n'est stockée — uniquement des identifiants Stripe
opaques + le statut. Tokens (session/reset/vérif) hachés SHA-256 au repos.

---

## 3. Webhooks Stripe & leur traitement

Endpoint : `POST /api/billing/webhook`. **Signature vérifiée** (corps brut) avant
tout changement d'état → 400 sinon. **Idempotent** : chaque `event.id` appliqué une
seule fois (`processed_webhooks`). **Ordre non garanti** géré par `last_event_created`
(un événement plus ancien que le dernier appliqué est ignoré). Événement non
résoluble (compte introuvable) → **non réclamé**, pour qu'un retry Stripe aboutisse
une fois le client lié.

| Événement Stripe | Effet sur l'état |
|---|---|
| `checkout.session.completed` | Lie `customer`↔compte (l'état complet arrive via `subscription.*`). Pas d'accès accordé sur cette base seule. |
| `customer.subscription.created` / `updated` | Écrit statut + `current_period_end` + `cancel_at_period_end` + `price_id` (source de vérité). |
| `customer.subscription.deleted` | Statut `canceled` → accès coupé. |
| `invoice.paid` / `payment_succeeded` | Renouvellement réussi → `active`. |
| `invoice.payment_failed` | `past_due` → **période de grâce** (accès conservé), pas de coupure immédiate. |
| `charge.refunded` (total) | `suspended` → accès coupé (remboursement partiel ignoré). |
| `charge.dispute.created` | `suspended` → accès coupé (litige). |

**États app** dérivés du statut Stripe + `cancel_at_period_end` : actif · en grâce
(past_due, fenêtre `SUBSCRIPTION_GRACE_DAYS`) · résilié-mais-actif-jusqu'à-la-fin ·
expiré · suspendu — chacun avec un affichage clair (date, montant, devise) sur la
page d'abonnement.

**Test local** : `stripe listen --forward-to localhost:8000/api/billing/webhook`
(fournit le `whsec_` local) + `stripe trigger …` + cartes de test (`4242…`) +
**test clocks** pour simuler renouvellement/dunning.

---

## 4. Point d'autorisation — UNIQUE

`src/api/subscription_gate.py` :

- **`enforce_access(request, account)`** = la SEULE couture pour les routes de
  données (market-reading, candles, live-price, conditions-scan, scanner-translate,
  chatbot). Ordre : gate OFF → passe (test) ; non authentifié → **401** ; e-mail non
  vérifié → **403** ; sans abonnement actif → **402** ; owner & abonné actif passent.
- `account_has_access()` = décision d'entitlement (owner / abonnement actif /
  past_due en grâce). `has_active_subscription()` lit l'état persistant — **jamais**
  d'appel Stripe à chaud (latence = une lecture SQLite indexée, conforme PERF-1/2).
- **Accès décidé uniquement à partir de l'état reçu par webhook**, jamais de la
  redirection de succès, d'une valeur client ou d'un paramètre d'URL.

Le périmètre freemium (`entitlements.py`) a été **supprimé** : plus de palier
gratuit, plus de second point de décision.

---

## 5. Préavis de renouvellement (LPC Québec) — délai retenu

**Structure :** l'abonnement est à **durée INDÉTERMINÉE, résiliable à tout moment**
(et non un terme fixe qui se reverrouille) — c'est exactement le modèle des
subscriptions Stripe, et c'est ce qu'impose la LPC (interdiction de reconduire
automatiquement un contrat à durée déterminée > 60 j vers une nouvelle période
déterminée). À formuler ainsi partout : « sans engagement, résiliable en un clic ».

**Délai retenu :** **rappel courriel 30 jours avant le renouvellement annuel** +
**reçu à chaque cycle mensuel**. La LPC n'impose pas de « rappel avant chaque
renouvellement » chiffré pour un abonnement à durée indéterminée, mais :
- **30 jours** de préavis sont **obligatoires** avant toute **modification** (ex.
  hausse de prix) ;
- **60 jours** si c'est le commerçant qui résilie.

Résiliation **aussi simple que l'abonnement** (LPC) : bouton → Customer Portal,
**sans parcours de rétention** imposé.

---

## 6. Transferts de données hors Québec à documenter (Loi 25)

Deux transferts **hors Québec** (à consigner : ÉFVP + clause de protection
comparable + mention dans la politique de confidentialité) :

1. **Hébergement Render — région Oregon (US)** *(confirmé)*. La base `accounts.db`
   (données personnelles) réside donc aux États-Unis, **même avec l'auth maison** —
   garder l'auth en interne ne règle pas la résidence. Exposition CLOUD Act à noter.
2. **Stripe (US)** — données de paiement (Stripe reste responsable PCI ; nous ne
   stockons aucune carte).

Loi 25 respectée par ailleurs : consentement explicite horodaté + version
(`account_consents`), politique de confidentialité accessible avant inscription,
droit d'accès/rectification (page compte), **suppression fonctionnelle**
(`DELETE /api/auth/account`, annule l'abonnement Stripe puis efface en cascade),
mention **18+** à l'inscription (et à l'étape de consentement Google).

**À NE PAS copier d'un modèle US :** aucune clause d'exclusion totale de
responsabilité, aucun arbitrage obligatoire, aucune renonciation à l'action
collective (inopposables au Québec).

**Taxes :** sous 30 000 $ de ventes → **aucune taxe** facturée, le prix affiché est
le prix payé. Stripe Tax **préparé mais OFF** (code taxe/adresse à activer plus tard
sans migration douloureuse ; aucun `STRIPE_TAX_ENABLED`).

---

## 7. Variables d'environnement à poser dans Render

**Obligatoires (production) :**

| Variable | Rôle |
|---|---|
| `SESSION_SECRET` | Signe les cookies de session (long aléatoire ; fail-fast en prod si absent). |
| `SESSION_COOKIE_SECURE=1` | Cookie `Secure` (HTTPS). |
| `ENVIRONMENT=production` | Active les contrôles fail-fast. |
| `SENTINEL_TESTING_MODE=0` | Auth réellement appliquée (pas de bypass clé API). |
| `SUBSCRIPTION_GATE_ENFORCED=1` | **Active le mur payant** (sinon tout est ouvert). |
| `SUBSCRIPTION_GRACE_DAYS=7` | Fenêtre de grâce après échec de paiement. |
| `STRIPE_SECRET_KEY` | `sk_live_…` (serveur uniquement). |
| `STRIPE_WEBHOOK_SECRET` | `whsec_…` de l'endpoint webhook **de production** (≠ celui du CLI). |
| `STRIPE_PRICE_MONTHLY` / `STRIPE_PRICE_ANNUAL` | `price_…` USD (39 $/mo, 348 $/an ; montants = `config/pricing.json`). |
| `STRIPE_SUCCESS_URL` / `STRIPE_CANCEL_URL` / `STRIPE_PORTAL_RETURN_URL` | Retours Checkout/Portal (URLs publiques). |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_FROM` | Envoi des e-mails de **vérification** et de reset. |
| `APP_URL` | Base du site Next.js (liens e-mail + retours OAuth). |
| `OWNER_USERNAME` / `OWNER_EMAIL` / `OWNER_PASSWORD` | Compte propriétaire (semé vérifié, accès complet). |

**Optionnelles :**

| Variable | Rôle |
|---|---|
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Activent le bouton « Continuer avec Google ». Absentes → bouton masqué, endpoints 404. |
| `GOOGLE_REDIRECT_URI` | Callback OAuth (sinon `{API_PUBLIC_URL}/api/auth/google/callback`). |
| `API_PUBLIC_URL` | Origine publique de l'API (callback Google). |
| `STRIPE_TRIAL_DAYS=0` | **Laisser 0** — PAY-1 interdit l'essai gratuit. |
| `CORS_ALLOWED_ORIGINS` | Origines autorisées. |

**Interdits dans le dépôt :** aucune clé Stripe, aucun `whsec_`, aucun secret
d'auth/OAuth — variables d'environnement uniquement ; `.env.example` à jour avec
des valeurs vides.

---

## 8. État de livraison (au moment de ce rapport)

| Lot | État | Tests |
|---|---|---|
| 1 — Pivot payant seul (point d'autorisation unique) | ✅ commité | vitest 872, gate 14, tsc 0 |
| 3 — Stripe : grâce / ordre / remboursement / 5 états | ✅ commité | billing 22, i18n/pricing 68 |
| 2 — Vérif e-mail + mdp + suppression compte (backend) | ✅ commité | +14, suite PAY-1 backend 159 |
| 2 — Google OAuth (env-gated, backend) | ✅ commité | 9 (round-trip live à valider) |
| 2 — Frontend (page vérif, bannière, page compte mdp/suppression) | ✅ commité | vitest + tsc 0 + build ✓ |
| 2 — Frontend Google (bouton + page finalisation) | ⏳ reste (nécessite identifiants Google) | — |
| 4 — Textes légaux (durée indéterminée, aucune clause US) | ✅ vérifié (mentions PRIX-1 conformes) | — |
| 4 — Préavis de renouvellement annuel 30 j (job Loi 25) | ✅ commité | +8 |
| 5 — `tsc` + `next build` | ✅ verts | build ✓ |
| 5 — Playwright (inscription→vérif→paiement→états→résiliation) + e2e Stripe CLI | ⏳ reste (paiement de test = validation live) | — |

**Job Loi 25 à câbler au déploiement :** appeler
`src.billing.renewal_notices.send_due_renewal_notices(store)` une fois par jour
(cron/scheduler Render). Idempotent ; no-op sans `STRIPE_PRICE_ANNUAL`/SMTP.

**À valider en live par toi (nécessite tes accès) :** paiement de test Stripe de
bout en bout (clés test + `stripe listen`) et connexion Google réelle (identifiants
Google Cloud). Le merge est conditionné à ta constatation de ces deux flux.

**Pré-existants non liés à PAY-1 :** `tests/test_tr1_structural_trend.py`
(ImportError `_eval_mtf_aligned`, symbole retiré avant PAY-1) et 2 smoke e2e
`/api/v1/scanner/status` 503 (API B2B legacy).
