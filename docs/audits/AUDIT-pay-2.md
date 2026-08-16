# AUDIT — PAY-2 : payer est la condition d'entrée

Branche : `fix/pay-2-acces-abonnement` (depuis `origin/main` = PAY-1, PR #156).
Worktree dédié : `C:\MyPythonProjects\wt-pay-2`.
Périmètre : corriger les 4 défauts de commercialisation — redirections localhost, essai
gratuit, « se connecter suffit », impossibilité de payer. **Aucune règle de détection touchée.**

---

## 0. Résumé exécutif

| Défaut constaté en prod | Cause racine | Correction |
|---|---|---|
| **3. Se connecter suffit pour entrer** | `render.yaml` posait `SUBSCRIPTION_GATE_ENFORCED=0` → `enforce_access` est un no-op → tout compte connecté a tout | Gate activé en prod + toutes les routes de données passées derrière `enforce_access`, y compris celles qui n'y étaient pas |
| **1. Redirections vers localhost** | Stripe (`STRIPE_*_URL`) et Google (`API_PUBLIC_URL`/`APP_URL`) défaut `localhost`, non posés dans Render | Source unique `src/api/public_urls.py` + **refus de démarrage** si base absente/localhost en prod |
| **2. Essai gratuit / palier gratuit** | Backend déjà payant-seul (PAY-1) mais **vitrine + i18n + catalogue** gardaient « Découverte / gratuit / sans carte » | Palier gratuit supprimé partout (vitrine, i18n fr+en+7 locales, catalogue `pricing.py`) + test-garde |
| **4. On n'arrive pas à payer** | Stripe **pas branché en prod** (aucune clé dans `render.yaml`) + funnel sans redirection auto vers le paiement | Clés Stripe déclarées dans `render.yaml` (à renseigner) + funnel inscription→vérif→**choix formule**→Checkout→attente webhook |

---

## A. URL codées en dur — trouvées et corrigées

### Trouvées (toutes avec un repli `localhost`, aucune posée dans Render)
| Emplacement | Variable | Ancien défaut |
|---|---|---|
| `src/api/routes/account_billing.py` `_success_url/_cancel_url/_portal_return_url` | `STRIPE_SUCCESS_URL` / `STRIPE_CANCEL_URL` / `STRIPE_PORTAL_RETURN_URL` | `http://localhost:3000/abonnement…` |
| `src/api/routes/google_auth.py` `_redirect_uri` | `API_PUBLIC_URL` | `http://localhost:8000` + `/api/auth/google/callback` |
| `src/api/routes/google_auth.py` `_app_url` | `APP_URL` | `http://localhost:3000` |
| `src/api/routes/accounts.py` `_reset_base_url` (vérif e-mail + reset mdp) | `FRONTEND_BASE_URL` | `https://mia.markets` (correct, mais non déclaré) |

### Corrigées
- **Source unique** : `src/api/public_urls.py`
  - `app_public_url()` — base **frontend** (Stripe success/cancel/portal, retour OAuth, liens e-mail). Lit `APP_PUBLIC_URL` (replis `APP_URL`, `FRONTEND_BASE_URL`).
  - `api_public_url()` — base **backend** (callback OAuth). Lit `API_PUBLIC_URL` (repli `API_BASE_URL`).
- Les trois constructeurs Stripe, les deux Google et les liens e-mail passent désormais **par cette source** ; plus aucun host en dur (les `STRIPE_*_URL`/`GOOGLE_REDIRECT_URI` restent des surcharges optionnelles).
- **Refus de démarrage** (`assert_public_urls_configured`, appelé au boot dans `app.py`) : si `ENVIRONMENT=production` et qu'une base est absente **ou** pointe sur localhost → l'app **ne démarre pas** (message explicite). No-op en dev/CI/tests.
- `.env.example` mis à jour (nouvelle section `APP_PUBLIC_URL`/`API_PUBLIC_URL`, `STRIPE_*_URL` laissés vides).

Tests : `tests/test_pay2_access.py::TestPublicUrlGuard` (no-op dev, lève sans URL en prod, lève sur localhost, passe avec https) + `TestNoLocalhostRedirectInProduction` (les 6 redirections résolvent en `https://…`, jamais localhost).

---

## B. Routes non protégées — trouvées et protégées (le point central)

Décision d'accès **unique** : `src/api/subscription_gate.py::enforce_access(request, account)`
(paid-only, tout-ou-rien ; no-op tant que `SUBSCRIPTION_GATE_ENFORCED≠1`, 401 anonyme /
403 e-mail non vérifié / 402 sans abonnement quand le gate est ON ; owner + abonnés passent).

### Déjà protégées avant PAY-2 (6)
`/api/market-reading`, `/api/candles`, `/api/live-price`, `/api/conditions-scan`,
`/api/scanner/translate`, `/api/chatbot/message`.

### Trouvées NON protégées → protégées par PAY-2 (données de marché / intelligence)
| Route | Fichier | Alimente |
|---|---|---|
| `GET /api/calendar` | `routes/calendar.py` | — |
| `GET /api/calendar/month` | `routes/calendar.py` | **/actualites** |
| `GET /api/calendar/event/{id}` | `routes/calendar.py` | /actualites (détail) |
| `GET /api/publications/{key}/measures` | `routes/calendar.py` | /actualites (mesures) |
| `GET /api/publications/{key}/measures/debug` | `routes/calendar.py` | diag |
| `GET /api/publications/{key}/values/debug` | `routes/calendar.py` | diag |
| `GET /api/publications/{key}/measures/backfill` (+ `/status`) | `routes/calendar.py` | ops |
| `GET /api/publications/{key}/measures/import-csv` (+ `/status`) | `routes/calendar.py` | ops |
| `GET /api/structure` | `routes/structure.py` | zones/events persistés |
| `GET /api/coverage` | `routes/structure.py` | métadonnées d'historique |

Chacune reçoit maintenant `account = Depends(optional_account)` + `enforce_access(request, account)`.

### Frontend — `/actualites` remis derrière le mur
- `webapp/middleware.ts` : `/actualites` ajouté à `PROTECTED_PREFIXES` (mur d'authentification au bord).
- La page `/actualites` (et `/actualites/[eventId]`) est déjà enveloppée de `<SubscriptionGate>` (paid-only par défaut) → un compte sans abonnement voit le **Paywall**, pas les données.

### Hors périmètre (documenté) — B2B par clé d'API
Les routes `/api/v1/*` (narratives, enrich, qa, insights, audit, operator, signals, state)
sont un **système d'accès distinct** (clé d'API partenaire, tiers `tier_manager`), sans rapport
avec l'abonnement web grand public. Elles restent gouvernées par `require_api_key`. À revoir
séparément si/quand elles sont commercialisées.

Tests : `tests/test_pay2_access.py` (401 visiteur / 402 sans-abo / passe pour abonné / ouvert
gate OFF, sur les 6 routes nouvellement fermées) + suite existante
`tests/test_subscription_gate_paid_only.py` (reading/candles/scanner/chat).

---

## C. État d'abonnement & webhooks — inchangé (déjà sain, PAY-1)

- `POST /api/billing/webhook` : **signature Stripe vérifiée** avant tout changement d'état
  (400 sinon) ; **idempotent** (`processed_webhooks`, chaque `event.id` appliqué une fois) ;
  **désordre géré** (`event_created` — un événement plus ancien n'écrase pas) ; événement non
  rattachable **non « claimé »** (retry Stripe possible). L'accès lit **la base**
  (`subscriptions`), jamais le client.
- Grâce `past_due` : conservée `SUBSCRIPTION_GRACE_DAYS` jours (défaut 7) avant coupure.
- Tests existants : `tests/test_account_billing.py::test_invalid_signature_rejected`,
  `test_missing_signature_rejected`, `test_duplicate_event_applied_once`.

---

## D. Traces d'essai gratuit — supprimées

### i18n (fr + en natifs, + de/es/it/pt/nl/pl/ar traduits — parité de clés maintenue)
- **Bloc plan gratuit `home.pricing.free`** (name/sub/bill/f1-4/cta/note) **supprimé** dans les 9 locales ; la carte « 0 $ / Découverte » retirée de `HomeLanding.tsx`.
- Reformulé partout vers **« accès complet dès l'abonnement, résiliable à tout moment »** :
  `home.hero.ctaPrimary` (« Essayer gratuitement » → « S'abonner »), `home.hero.micro`,
  `home.pricing.subtitle`, `home.faq.a4/a6`, `home.final.p/cta1`, `home.demoSection.cta`,
  `nav.tryFree`, top-level `pricing.subtitle/free_title`, `app.header.planBadge`
  (« Accès libre » → « Abonné »).
- Section « Honnêteté » : « démos gratuites » → « démos ouvertes, sans compte ».
- Attribution Twelve Data : « (palier gratuit, 800 req/j) » → « (800 req/j) ».
- Copie de vérification e-mail corrigée : « ton accès est activé » → « choisis ta formule pour activer ton accès » (les 9 locales).

### Logique
- `src/billing/pricing.py` : plan `FREE` **retiré du catalogue** (`PLAN_FREE` conservé comme alias legacy, jamais dans les plans). `list_paid_plans`/`list_plans` ne renvoient que MONTHLY + ANNUAL.
- Stripe : `STRIPE_TRIAL_DAYS=0` (aucune période d'essai sur aucun prix) ; `trial_period_days` n'est envoyé que si `>0`.
- Hors périmètre (documenté) : `tier_manager.UserTier.FREE` et `telegram_notifier` (canal Telegram/B2B) — système séparé, non commercialisé ici.

### Garde
- `webapp/lib/i18n/__tests__/no-free-tier.test.ts` : échoue si un mot d'offre gratuite
  (gratuit / gratis / free / free trial / try for free / no credit card / kostenlos / darmowy /
  مجاني …) réapparaît dans les namespaces d'offre (`home`, `nav`, `pricing`), dans **les 9 locales** ;
  vérifie aussi que `home.pricing.free` n'existe plus nulle part.

---

## Le parcours attendu — état après PAY-2

- **Visiteur non connecté** : accueil, démos, tarif, FAQ, légal, connexion, inscription.
  Accès direct à une surface de données → Paywall (CTA « Voir les abonnements » → `/abonnement`) ;
  anonyme + gate ON → redirection `/connexion?next=…`.
- **Inscription** → e-mail/mdp ou Google → **vérification e-mail obligatoire** → **redirection
  automatique vers `/abonnement`** (choix mensuel/annuel) → Stripe Checkout hébergé → retour,
  **accès accordé par le webhook** (écran d'attente qui relance et bascule vers `/app`).
  - `EmailVerifier` : succès → auto-redirection `/abonnement`.
  - `GoogleFinalizeForm` (nouveau compte Google) : → `/abonnement`.
  - `SubscriptionPanel` : `?status=success` → écran « confirmation Stripe » qui **poll** l'abonnement puis entre dans `/app` (même si l'onglet a été fermé avant, le webhook a déjà accordé l'accès côté serveur).
- **Compte sans abonnement** : page de compte + invitation à s'abonner ; **aucune donnée**.
- **Résiliation** : accès conservé jusqu'à la fin de la période (date affichée).
- **Fin de période** : accès retiré. **Échec de renouvellement** : grâce puis suspension.

---

## Résultat des neuf scénarios (§4)

Automatisés dans ce dépôt. La **preuve de bout en bout sur le domaine de production avec un
vrai paiement de test Stripe** reste à ta charge (câblage Stripe live jamais constaté — cf.
mémoire PAY-1) : c'est la condition de merge que tu as posée.

| # | Scénario | Couverture automatisée | Résultat |
|---|---|---|---|
| 1 | Inscription e-mail → vérif → paiement → accès | Playwright `S1` (+`S1b`) : vérif→`/abonnement`, plan payant offert, Checkout→confirm→`/app` | ✅ |
| 2 | Inscription Google → paiement → accès | Redirection `GoogleFinalizeForm`→`/abonnement` (code) ; suite du funnel = S1b. OAuth Google = **check live** | ⚠️ live |
| 3 | Compte sans payer → aucun accès | Playwright `S3` (Paywall sur `/app` **et** `/actualites`) + `tests/test_pay2_access.py` | ✅ |
| 4 | Accès direct par URL sans abo → redirection abonnement | Playwright `S4` (Paywall→`/abonnement`) + `S4b` (anonyme→`/connexion`) | ✅ |
| 5 | Paiement puis fermeture onglet avant redirection → accès par webhook | Playwright `S5` (écran d'attente → poll → `/app`) ; côté serveur l'accès vient de la base | ✅ |
| 6 | Résiliation → accès jusqu'à fin de période | Playwright `S6` (pas de Paywall + date de fin) | ✅ |
| 7 | Après fin de période → accès retiré | Playwright `S7` (Paywall + plans) + `tests/test_subscription_gate_paid_only.py` (expiry) | ✅ |
| 8 | Échec renouvellement → grâce puis suspension | Playwright `S8a` (grâce = accès) / `S8b` (suspendu = Paywall) | ✅ |
| 9 | Appel direct route API sans abonnement → refusé côté serveur | `tests/test_pay2_access.py` + `tests/test_subscription_gate_paid_only.py` (401/402) | ✅ |

Playwright tourne sur **les deux viewports** (1280×800 desktop + 390×844 mobile) via les deux
projets de `playwright.config.ts`.

---

## Ce que TU dois configurer (Render / Stripe / Google)

### Render — service backend `mia-backend` (déjà encodé dans `render.yaml`, valeurs à renseigner)
| Variable | Valeur |
|---|---|
| `ENVIRONMENT` | `production` *(active les gardes fail-fast)* |
| `SUBSCRIPTION_GATE_ENFORCED` | `1` *(le mur de paiement mord)* |
| `BETA_LOCKDOWN` | `0` *(inscriptions ouvertes)* |
| `SENTINEL_TESTING_MODE` | `0` |
| `APP_PUBLIC_URL` | l'URL publique du **frontend** (ex. `https://mia.markets`) |
| `API_PUBLIC_URL` | l'URL publique de **ce backend** (ex. `https://mia-backend-xxxx.onrender.com`) |
| `STRIPE_SECRET_KEY` | clé secrète Stripe (test d'abord) |
| `STRIPE_WEBHOOK_SECRET` | secret de signature du endpoint webhook |
| `STRIPE_PRICE_MONTHLY` | price id du plan mensuel (39 USD, **sans essai**) |
| `STRIPE_PRICE_ANNUAL` | price id du plan annuel (348 USD, **sans essai**) |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | si connexion Google (sinon bouton masqué) |
| `CORS_ALLOWED_ORIGINS` | l'origine du frontend |
| `SMTP_HOST/PORT/USER/PASSWORD/FROM` | pour l'envoi réel des e-mails (vérif + reset) |

Service frontend `mia-frontend` : `BETA_LOCKDOWN=0`, `NEXT_PUBLIC_BETA_LOCKDOWN=0`,
`NEXT_PUBLIC_API_BASE` = URL du backend (baked au build → redéployer le front après l'avoir posée).

### Stripe (dashboard)
1. Produit + **deux prix récurrents** USD : 39,00/mois et 348,00/an — **aucune période d'essai** sur aucun des deux (Stripe Tax **désactivé**, aucun code promo).
2. Copier les deux price ids → `STRIPE_PRICE_MONTHLY` / `STRIPE_PRICE_ANNUAL`.
3. **Webhook** : endpoint `https://<API_PUBLIC_URL>/api/billing/webhook`, événements
   `checkout.session.completed`, `customer.subscription.created/updated/deleted`,
   `invoice.paid`, `invoice.payment_succeeded`, `invoice.payment_failed`. Copier le secret → `STRIPE_WEBHOOK_SECRET`.
4. **Customer Portal** activé (résiliation) — l'URL de retour est fournie par le backend (`APP_PUBLIC_URL/abonnement`).

### Google Cloud Console (si connexion Google)
- OAuth 2.0 Web client. **Authorized redirect URI** : `https://<API_PUBLIC_URL>/api/auth/google/callback` (exactement).
- Authorized JavaScript origins : l'origine du frontend.
- `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` → Render.

---

## Validation locale (au moment de l'audit)

- Backend : `tests/test_pay2_access.py` (9) + suites affectées (billing, gate paid-only,
  calendar, structure, google, comptes, sessions, pricing) = **162 passés**.
- Frontend : `tsc --noEmit` **0 erreur** ; vitest **904 passés** (dont garde no-free-tier + parité 9 locales) ; `next build` vert.
- Playwright : `tests/e2e/pay2-access.spec.ts` — 9 scénarios × 2 viewports.

## Reste à faire (toi, sur la prod)
1. Renseigner les variables Render/Stripe/Google ci-dessus (clés **de test** d'abord).
2. Déployer la branche, **constater un paiement de test Stripe réussi de bout en bout** sur le
   domaine de production (inscription → vérif → Checkout → accès), + un webhook reçu.
3. Vérifier la connexion Google réelle (scénario 2) et un cycle de résiliation.
4. Sur ta confirmation live → merge sur `main`.
