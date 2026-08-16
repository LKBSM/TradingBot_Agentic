# AUDIT PAY-3 — Parcours d'accès complet : authentifier, faire payer, laisser entrer

**Branche** : `fix/pay-3-parcours-acces` (worktree dédié, depuis `origin/main` = `eddcfbe`).
**Posture** : rien ne se commercialise tant que ce parcours n'est pas irréprochable.

Ce document livre : la cause exacte de l'échec Google, le comportement constaté et
corrigé sur la double inscription, la recommandation sur le nom d'utilisateur, les
routes protégées, **un bug de production critique découvert au passage**, et la liste
exacte de ce que tu dois configurer toi-même dans Render, Stripe et la Google Console.

---

## 0. Résumé exécutif

| Défaut | Cause racine | Correctif |
|---|---|---|
| **A — Google échoue en silence** | Le `redirect_uri` du callback pointait sur l'origine **backend** (`API_PUBLIC_URL`) alors que le cookie CSRF `g_oauth_state` est posé sur l'origine **front** (le bouton passe par le proxy `/api/*`). Le cookie n'est jamais présenté au callback → échec systématique. Et la page `/connexion` ne lisait jamais le paramètre `?error`. | Callback routé par l'origine **front** (`APP_PUBLIC_URL`), même origine que `/start` → le cookie circule. Sous-codes d'erreur distincts (`state`/`expired`/`exchange`/`email`) + bannière visible fr/en sur `/connexion`. |
| **B — La page d'inscription ne vend rien** | Un simple formulaire centré, sans prix, sans valeur, sans bouton Google. | Refonte : proposition de valeur, prix + devise (source unique), 3 étapes, mentions obligatoires, **bouton Google à l'inscription** (parité avec la connexion). Aucun « gratuit/essai ». |
| **C — Le parcours n'en est pas un** | La connexion (e-mail **et** Google d'un compte existant) allait droit à `/app` ; l'inscription e-mail allait à `/abonnement` sans passer par la vérification. | Routage par les **6 états** : connexion → `/abonnement` si non abonné (serveur pour Google, client pour e-mail). Inscription e-mail → écran « confirme ton e-mail » (avec **renvoi**) → `/abonnement`. |
| **🔴 Bug prod critique** (découvert) | `StripeClient.verify_webhook` renvoyait un objet `stripe.Event` dont `.get()` lève `AttributeError` ; les parseurs de webhook appellent `.get()`. **Chaque vrai webhook aurait planté en 500 → abonnement jamais persisté → « payé mais pas d'accès ».** Jamais détecté : les tests utilisaient un faux client renvoyant un `dict`. | `verify_webhook` renvoie désormais le corps JSON décodé (dict pur) **après** vérification de signature. Couvert par un test qui exerce la **vraie** signature Stripe. |
| **Faille d'accès** | `GET /api/market-status` servait une donnée de marché **sans aucune authentification** (anonyme total). | Gaté par `enforce_access` (401/402 sous le mur). |

**Tests** : backend `tests/test_pay3_payment_journey.py` (**8/8**, dont la vraie signature
Stripe) + suite backend **3903 passés** (4 échecs prouvés pré-existants sur `origin/main`,
cf. §7) ; front **tsc 0**, **vitest 913**, **build vert**, Playwright **pay3-parcours 18/18**
+ **pay2-access 20/20** (fr + en × 2 viewports 1280×800 / 390×844).

---

## 1. La cause EXACTE de l'échec Google

### Le flux réel
1. Le bouton « Continuer avec Google » est un `<a href="/api/auth/google/start">` :
   une navigation pleine page vers l'**origine front** (`mia.markets`), que Next.js
   réécrit (`/api/:path*` → backend, `next.config.js`).
2. `/start` (backend) pose le cookie CSRF `g_oauth_state`. Comme la réponse revient
   par l'origine front, **le cookie appartient à `mia.markets`** (path `/api/auth/google`).
3. Google renvoie le navigateur vers `redirect_uri`.

### Le défaut
`_redirect_uri()` valait `{API_PUBLIC_URL}/api/auth/google/callback` — l'**origine
backend directe** (`api.mia.markets` / `onrender.com`). Le navigateur y envoie les
cookies de `api.mia.markets`, **pas** ceux de `mia.markets`. Résultat : `g_oauth_state`
absent au callback → `state_cookie is None` → branche « state » → redirection
`/connexion?error=google` **sans jamais appeler Google** (donc aucun log de la branche
« exchange »). Corollaire : même en cas de succès, le cookie de **session** aurait été
posé sur la mauvaise origine — l'app front ne l'aurait jamais lu.

Et côté écran : `/connexion` (`LoginForm.tsx`) **ne lisait jamais** `?error` — le
paramètre était reçu puis ignoré → formulaire vierge, échec silencieux.

### Le correctif
- `src/api/routes/google_auth.py::_redirect_uri()` → défaut `{APP_PUBLIC_URL}/api/auth/google/callback`
  (origine **front**, proxifiée vers le backend). `start` et `callback` partagent
  désormais l'origine qui détient le cookie → le state circule, la session atterrit au
  bon endroit. `GOOGLE_REDIRECT_URI` reste un override.
- Sous-codes d'erreur : `_err_redirect(reason)` émet `?error=google&reason=state|expired|exchange|email`,
  chacun **loggué** côté serveur (la branche « state » loggue enfin, avec l'indice
  cross-origin).
- Front : `LoginForm.tsx` lit `?error=google&reason=…` au montage et affiche une
  bannière `role="alert"` claire fr/en (namespace `auth.google.error.*`), puis nettoie
  l'URL. Propose l'autre chemin (e-mail/mot de passe).

---

## 2. Double inscription même e-mail — constaté et corrigé

### Constaté (avant)
Le pire défaut redouté **n'existait pas** : le callback Google cherche déjà un compte
existant **par e-mail** (`get_account_by_identifier`) **avant** toute création. La table
`accounts` a `email_lower NOT NULL UNIQUE`. Donc :
- inscription e-mail puis Google même adresse → **le même compte** (avec son abonnement) ;
- inscription Google puis mot de passe → 401 propre (pas de doublon), remède = « mot de
  passe oublié » ;
- deux comptes pour une même adresse → **physiquement impossible**.

### Corrigé / renforcé (PAY-3)
- Un **test** verrouille l'invariant : `TestOneEmailOneAccount` prouve que la même adresse
  sur les deux chemins ne produit **qu'un seul** compte (`create_account_auto` lève
  `email_taken`, `get_account_by_identifier` renvoie le même id).
- La liaison reposait entièrement sur l'e-mail vérifié Google (pas de `google_sub` stocké).
  C'est acceptable (Google ne renvoie qu'une adresse vérifiée) et documenté ; un
  `google_sub` durable serait une amélioration future, hors périmètre PAY-3.

---

## 3. Recommandation sur le nom d'utilisateur — RETIRÉ des formulaires

**Décision appliquée : retiré des deux chemins, dérivé de l'e-mail côté serveur.**

Usages réels constatés : aucune valeur produit (un seul affichage marginal, le menu
avatar), absent du profil ; mais **seconde surface d'unicité** (`username_lower UNIQUE`)
qui pouvait refuser un utilisateur Google légitime (pseudo pris) alors que son e-mail
était libre — friction pure sur un parcours censé être sans couture.

Mise en œuvre **non destructive** (zéro migration) :
- Colonne `username` conservée. Nouvelle méthode `AccountStore.create_account_auto(email, …)`
  qui **dérive** un username unique de l'e-mail (nettoyage + retry anti-collision).
- Champ retiré du formulaire d'inscription **et** de la finalisation Google.
- Connexion : le backend continue d'accepter « username **ou** e-mail » (aucun compte
  existant cassé) ; l'identité présentée à l'utilisateur est l'e-mail.

---

## 4. Contrôle d'accès — routes de données

Point de décision **unique** : `src/api/subscription_gate.py::enforce_access` (lit
l'abonnement **depuis la base**, alimentée par webhook — jamais Stripe/redirection/valeur
client). Activé en prod via `SUBSCRIPTION_GATE_ENFORCED=1` (`render.yaml`).

### Faille fermée
- **`GET /api/market-status`** (`market_reading.py`) — servait le statut marché
  **sans aucune auth** (anonyme). Désormais `enforce_access` (401 anonyme / 402 non abonné).

### Laissé ouvert, à dessein (documenté)
- **`GET /api/conditions-scan/palette`** — ne renvoie que la **palette statique** (vocabulaire
  fermé du scanner), aucune donnée de marché. La page scanner est de toute façon derrière
  le mur front ; la palette ne fuit rien d'exploitable.

### Surface legacy `/api/v1/*` (X-API-Key)
Les routes `/api/v1/*` (signals, narratives, dashboard, insights, enrich) sont gardées
par `require_api_key` (clé X-API-Key + tier), **pas** par le mur d'abonnement compte.
Elles appartiennent à un **autre domaine d'auth** : un compte MIA (session cookie) n'a
**pas** de clé API, donc **ne peut pas** les appeler. Aucune clé n'est émise pour des
clients MIA. Un compte sans abonnement ne peut donc tirer aucune donnée par ces routes.
Elles restent une dette B2B distincte, à retirer ou rattacher plus tard.

### Webhook — désambiguïsation (le piège « payé mais pas d'accès »)
Deux routes existent :
- **`POST /api/billing/webhook`** (compte) — **LA bonne** : alimente la table `subscriptions`
  que le gate lit.
- **`POST /api/v1/billing/webhook`** (legacy) — alimente `tier_manager`, **jamais lu** par
  le gate.
Correctif : le webhook legacy **loggue désormais une ERREUR bruyante** si un événement y
arrive (« Stripe pointe le mauvais endpoint »). **Action requise : voir §6 — Stripe doit
pointer `/api/billing/webhook`.**

---

## 5. 🔴 Bug de production critique découvert (et corrigé)

En écrivant le test qui exerce la **vraie** intégration Stripe, j'ai découvert que
`StripeClient.verify_webhook` renvoyait l'objet `stripe.Event` de
`stripe.Webhook.construct_event`. Or cet objet lève `AttributeError` sur `.get()`, et
**tous** les parseurs (`parse_account_event`, `parse_webhook_event`) appellent `.get()`.

**Conséquence en production** : chaque vrai webhook Stripe aurait renvoyé **500** →
l'abonnement n'aurait **jamais** été persisté → le client paie, Stripe enregistre, mais
l'accès n'est jamais accordé → il retombe sur le mur et part. **Exactement le défaut le
plus coûteux décrit dans la mission (§7).** Jamais détecté car toute la suite de tests
utilisait un faux client renvoyant un `dict`.

**Correctif** (`src/billing/stripe_client.py::verify_webhook`) : on vérifie la signature
via `construct_event` (lève sur mauvaise signature) puis on renvoie
`json.loads(body)` — un dict pur, identique à ce que le faux client renvoie. Prod et
tests sont désormais alignés. Couvert par `TestRealStripeSignature` (signature Stripe
authentique ; **skip bruyant** si le SDK `stripe` est absent — jamais un skip silencieux).

---

## 6. Ce que TU dois configurer toi-même

### Google Cloud Console (OAuth 2.0 Web client)
1. **Authorized redirect URI** — ajoute **exactement** :
   `https://<APP_PUBLIC_URL>/api/auth/google/callback`
   (l'origine **front**, ex. `https://mia.markets/api/auth/google/callback`).
   ⚠️ **Pas** l'origine backend. C'était la cause de l'échec.
2. **Authorized JavaScript origins** — ajoute `https://<APP_PUBLIC_URL>`.
3. Récupère le **Client ID** et le **Client secret**.

### Render — service **backend**
| Variable | Valeur |
|---|---|
| `APP_PUBLIC_URL` | l'origine front publique, ex. `https://mia.markets` (⚠️ pas localhost) |
| `API_PUBLIC_URL` | l'origine backend publique |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | depuis la Google Console |
| `STRIPE_SECRET_KEY` | clé secrète Stripe (test puis live) |
| `STRIPE_WEBHOOK_SECRET` | `whsec_…` du **endpoint** créé ci-dessous |
| `STRIPE_PRICE_MONTHLY` / `STRIPE_PRICE_ANNUAL` | les price IDs Stripe (39 $/mois, 348 $/an) |
| `SUBSCRIPTION_GATE_ENFORCED` | `1` (déjà dans `render.yaml`) |
| `SESSION_SECRET` | une longue valeur aléatoire (identique = le state Google reste vérifiable) |

`GOOGLE_REDIRECT_URI` : **laisse vide** (le défaut front est correct). Ne le pose que
pour override, et alors enregistre-le verbatim dans la Console.

### Render — service **frontend**
| Variable | Valeur |
|---|---|
| `NEXT_PUBLIC_API_BASE` | l'origine backend (pour le proxy `/api/*`) |

### Stripe
1. **Webhook endpoint** → URL = `https://<APP_PUBLIC_URL>/api/billing/webhook`
   **(surtout PAS `/api/v1/billing/webhook`)**. Copie son `whsec_…` dans
   `STRIPE_WEBHOOK_SECRET`.
2. Événements à envoyer : `checkout.session.completed`, `customer.subscription.created`,
   `customer.subscription.updated`, `customer.subscription.deleted`,
   `invoice.payment_failed`, `charge.refunded`, `charge.dispute.created`.
3. Crée les **deux prix récurrents** (mensuel 39 $, annuel 348 $) → renseigne les price IDs.
4. **Le test qui compte** : après paiement de test réel, vérifie que
   `TestRealStripeSignature` / le parcours accorde bien l'accès. En prod, constate
   toi-même les deux parcours complets (§8).

---

## 7. Le point qui coûte le plus cher — la défense en place

La mission (§7) : le client paie, Stripe enregistre, mais l'app ne met pas à jour son
état → il repart sans réclamer, et la baisse de conversions est attribuée à autre chose.

Défense livrée : `tests/test_pay3_payment_journey.py` pousse le parcours complet à travers
les **vraies** routes / le **vrai** gate / le **vrai** store et **échoue bruyamment**
(assertion `has_access is True` avec message explicite) si l'accès n'est pas accordé après
le webhook. `TestRealStripeSignature` exerce en plus la **vraie** vérification de signature
Stripe. Ce test s'exécute à chaque `pytest` (donc à chaque déploiement). C'est lui qui a
déjà attrapé le bug §5.

*Note* : un module pré-existant `tests/test_tr1_structural_trend.py` échoue à la collecte
sur `origin/main` (import `_eval_mtf_aligned` supprimé par un refactor TR-1 antérieur) —
**sans aucun rapport avec PAY-3**. Désélectionné pour obtenir un signal propre ; à réparer
hors périmètre.

---

## 8. Ce qu'il reste — à constater sur le domaine de production

Le code est livré et testé. Avant fusion sur `main`, **tu** dois :
1. Poser les variables Render / Stripe / Google du §6.
2. Constater **toi-même**, sur le domaine de prod, les **deux parcours complets** :
   - inscription e-mail → vérification → choix formule → paiement test → accès ;
   - inscription/connexion Google → (choix formule) → paiement test → accès.
3. Vérifier qu'un échec Google affiche bien un message à l'écran.

**La fusion sur `main` n'intervient qu'après ta confirmation live et ton constat des deux
parcours.**
