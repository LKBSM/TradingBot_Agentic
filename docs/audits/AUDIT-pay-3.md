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

---
---

# PAY-3 — REPRISE DU 2026-09-13 : les trous que l'audit ci-dessus n'avait pas fermés

**Branche** : `feat/pay-3-acces` (worktree dédié `C:\MyPythonProjects\wt-pay-3-acces`,
depuis `origin/main` = `9de5138`).
**Posture** : le parcours d'accès était déjà bâti (PAY-1, PAY-2, PRIX-1, PAY-3a→e, tous
fusionnés). Cette reprise ne le refait pas : elle ferme ce qui manquait encore, et
énonce ce qui a été décidé de ne pas faire.

---

## 0. Résumé exécutif

| Sujet | Constat à l'ouverture | Livré |
|---|---|---|
| **Le test vivant (G1)** | `4242` n'apparaissait dans **aucun test**. Pire : `ci.yml` lançait une liste curée d'une trentaine de fichiers où **aucun test de paiement** ne figurait. Le scénario le plus coûteux n'avait **aucune défense automatisée**. | Test Playwright qui pousse une vraie carte de test à travers le Checkout hébergé, attend le webhook, échoue bruyamment à 90 s. Workflow dédié non contournable + les 7 fichiers de tests paiement enfin lancés par la CI. |
| **Géo CA/US (G2)** | Aucun `allowed_countries`, aucun `billing_address_collection` dans le dépôt. Restriction inexistante. | Adresse de facturation obligatoire au Checkout + filet webhook (annulation, remboursement, statut `blocked_region`) + écran dédié + règle Radar documentée. |
| **Webhook mort (G4)** | `/api/billing/sync` sauvait le client **et masquait la panne** : rien n'alertait. | Chaque sauvetage compté, journalisé en ERROR, poussé sur le canal d'alerte. |
| **Préavis annuel (G5)** | Conditionné à la seule présence de `SMTP_HOST`. « Peut-on envoyer » valait « a-t-on le droit ». | Verrou explicite `RENEWAL_NOTICE_TEXT_APPROVED` (défaut : rien ne part). |
| **Clerk** | Demandé par la mission. | **Non fait — décision fondateur.** Voir §5. |
| **39,99 $** | Mission 39,99 $, dépôt 39 $. | **39 $ maintenu — décision fondateur.** |

**Tests** : backend **220 passés** (7 suites paiement + 6 suites auth), dont **24 nouveaux**
(`tests/test_pay3_zone_and_health.py`) ; front **tsc 0**, **build vert**, **vitest 9/9**
(SubscriptionGate), **Playwright 16/16** (`pay3-etats.spec.ts`, fr × 1280×800 + 390×844).

---

## 1. G1 — le test vivant, et pourquoi il n'en existait pas

### Le constat
`git grep 4242` ne trouvait le numéro de carte que dans des `.md` et des CSV de données.
Les 8 tests PAY-3 existants utilisent un `FakeStripeClient` — utile, mais structurellement
incapable d'attraper la classe de panne qui compte : c'est exactement ainsi qu'un
`verify_webhook` renvoyant un objet `stripe.Event` (dont `.get()` lève `AttributeError`) est
passé en production, **le faux client, lui, renvoyant un `dict`**.

Et `ci.yml` ne lançait **aucun** de ces tests. La suite paiement existait sans jamais tourner.

### Ce qui est livré
`webapp/tests/live/stripe-live-journey.spec.ts` + `webapp/playwright.live.config.ts` :

1. compte jetable créé par la vraie route d'inscription ;
2. **on vérifie d'abord que le mur MORD** (402) — sans quoi un vert ne prouverait rien :
   un `SUBSCRIPTION_GATE_ENFORCED` oublié passerait pour un succès ;
3. vraie session Checkout, vraie page hébergée, carte `4242 4242 4242 4242`,
   pays **CA** (la zone est respectée : c'est le chemin nominal qu'on mesure ici) ;
4. attente de l'accès, 90 s maximum, 2 s entre deux sondages ;
5. échec **bruyant** avec la liste ordonnée de ce qu'il faut vérifier ;
6. puis contrôle que le mur s'est vraiment ouvert sur la route de données.

**Ce qu'il ne fait surtout pas** : appeler `POST /api/billing/sync`. La réconciliation
directe sauverait le client et rendrait le test vert **alors même que le webhook est mort** —
elle masquerait précisément la panne que ce test existe pour détecter. Seul le chemin webhook
est mesuré.

**Ce qui l'empêche d'être désactivé** :
- une clé `sk_live_` fait **échouer** le test (il pousse un vrai numéro de carte) ;
- `PAY3_LIVE_REQUIRED=1` (posé par le workflow) fait d'une configuration manquante un
  **échec**, pas un skip silencieux ;
- `retries: 0` : une réussite au second essai masquerait la latence anormale recherchée ;
- dans `.github/workflows/stripe-live.yml` : aucun `continue-on-error`, aucun `if:` qui
  puisse sauter le job ; le seul `if: always()` ne sert qu'à récupérer les journaux d'un échec.

**Quand il tourne** : push sur `main`, PR touchant la facturation, **et une fois par jour** —
un webhook peut mourir le mardi après un déploiement vert du lundi.

### Ce qui reste à faire avant que ce filet soit réellement armé
Deux secrets GitHub, en **mode test** : `STRIPE_SECRET_KEY` (`sk_test_…`) et
`STRIPE_PRICE_MONTHLY`. Tant qu'ils sont absents, le job échoue avec un message explicite —
c'est voulu : un filet non armé doit se voir.

> **Honnêteté sur ce qui n'a pas pu être vérifié ici** : ce workflow n'a jamais été exécuté.
> Aucune clé Stripe n'est présente dans cet environnement, et GitHub Actions ne tourne pas
> en local. Le YAML est validé (`yaml.safe_load`), les sélecteurs de la page Checkout sont
> ceux de la page hébergée actuelle, mais **le premier vrai passage est à faire par toi** —
> vraisemblablement en `workflow_dispatch` une fois les secrets posés.

---

## 2. G2 — la zone de vente Canada + États-Unis

### La contrainte technique, d'abord
**Stripe Checkout n'offre aucune liste blanche de pays de facturation** pour un abonnement
(`allowed_countries` ne concerne qu'une adresse de *livraison*). Il n'existe donc pas de
« cocher CA et US » — d'où deux couches, et il faut les deux.

| Couche | Où | Ce qu'elle fait |
|---|---|---|
| 1 — **Radar** | Tableau de bord Stripe (**action fondateur**) | `Block if :card_country: not in ('CA','US')` — empêche le paiement d'aboutir. |
| 2 — **filet webhook** | `src/billing/geo.py` + `account_billing.py` | Une session hors zone qui aboutit quand même est **annulée**, **remboursée**, et persistée en `blocked_region`. |

La couche 2 seule laisserait l'argent arriver avant d'être rendu ; la couche 1 seule est un
réglage de tableau de bord qu'aucun test ne voit. Ensemble, elles échouent fermé.

### Décisions de conception
- `billing_address_collection: "required"` au Checkout : **sans adresse, il n'y a pas de pays
  à lire**, donc pas de filet. C'est ce paramètre qui rend la couche 2 possible ; il est
  verrouillé par un test qui inspecte les paramètres envoyés à Stripe.
- **Un pays inconnu n'est jamais un motif de refus.** On ne bloque que sur un pays réellement
  lu. Refuser sur une absence d'adresse enfermerait dehors un client légitime pour une
  variation de forme d'événement — et Radar, lui, bloque la carte sans avoir besoin d'adresse.
- `BILLING_ALLOWED_COUNTRIES` permet d'élargir la zone par configuration. **Une valeur vide
  retombe sur `CA,US`** : une coquille dans une variable d'environnement ne doit jamais ouvrir
  le monde entier.
- **Le remboursement est automatique.** Annuler un abonnement ne rend pas l'argent ; garder
  un paiement pour un service qu'on refuse de rendre n'est pas défendable, encore moins sous
  la LPC. Si le remboursement échoue, il est journalisé « REFUND BY HAND » — et le refus
  d'accès, lui, passe quand même.
- **Une panne Stripe pendant l'annulation ne peut pas ouvrir l'accès** : le statut
  `blocked_region` est persisté même si l'appel d'annulation lève (test dédié).

### L'écran
`blocked_region` retombait sur l'écran « expiré », donc sur le choix de formule : on
réinvitait à payer un client dont le paiement venait d'être annulé et remboursé. Il a
maintenant son écran — motif nommé, remboursement dit explicitement, **aucune formule,
aucun bouton de gestion**. Corollaire corrigé : au retour de Checkout, le spinner de
confirmation aurait tourné 24 tentatives sur un refus qui ne changera jamais ; un refus est
une réponse **arrivée**, pas une réponse en attente.

---

## 3. G4 — un webhook mort ne peut plus se cacher derrière son propre contournement

`POST /api/billing/sync` (PAY-3e) réconcilie depuis l'API Stripe quand aucun webhook n'arrive.
C'est une bonne défense — mais elle **masquait** la panne : vu de l'extérieur tout allait
bien pendant que le webhook pouvait être mort depuis une semaine, tous les autres clients
enfermés dehors.

Désormais (`src/billing/webhook_health.py`), quand la réconciliation trouve un abonnement
**actif** que notre base ignorait : compteur incrémenté, ERROR journalisé avec la marche à
suivre, alerte poussée sur `DISCORD_WEBHOOK_URL` si configuré. **Un sauvetage isolé n'est pas
un incident** (le premier appel après Checkout peut devancer la livraison Stripe) ; un
compteur qui monte, si.

*Réserve assumée* : l'état est en mémoire de processus, donc remis à zéro à chaque
redéploiement. C'est un **signal**, pas une piste d'audit — la piste d'audit est la table
`processed_webhooks`. Si tu veux une métrique durable, c'est une mission séparée.

---

## 4. G5 — le préavis de renouvellement, verrouillé

Le préavis annuel à 30 jours existait depuis PAY-1 mais ne dépendait que de `SMTP_HOST`.
« Peut-on envoyer » et « a-t-on le droit d'envoyer **ce texte-là** » sont deux questions
différentes ; les confondre est la façon dont un texte juridique non relu part par accident.

`RENEWAL_NOTICE_TEXT_APPROVED` (défaut `0`) : rien ne part, et un WARNING nomme la variable
pour que l'arrêt ne soit jamais silencieux. Le gabarit est annoté des points que le texte doit
couvrir (date et montant du prélèvement, résiliation aussi simple que la souscription, aucune
clause de vente finale, lien direct vers la gestion).

**Le texte reste à faire valider par l'avocat. Ne pose `=1` qu'après.**

---

## 5. Ce qui a été décidé de NE PAS faire (et pourquoi)

### Clerk — écarté
La mission demandait Clerk. Constat : **aucune clé Clerk n'existe**, et surtout
`AUDIT-pay-1.md` §1 avait déjà tranché par écrit, sur un motif juridique —
*« Clerk = US-only sans choix de région → transfert hors Québec permanent »* (Loi 25).
L'auth maison en place est Argon2id + sessions opaques révocables + Google OAuth +
vérification e-mail : migrer signifierait jeter ce code, ajouter une dépendance, et
contredire une analyse déjà rendue. **Décision fondateur : on garde l'auth maison.**
Si un fournisseur managé devient souhaitable, le candidat documenté est Supabase Auth
(région Montréal `ca-central-1`), pas Clerk.

Conséquence : la partie A de la mission (identité, fusion par courriel, écran « connecté
jamais payé ») **était déjà livrée** — `email_lower NOT NULL UNIQUE`, recherche par e-mail
avant toute création côté Google, test `TestOneEmailOneAccount`. Rien à refaire.

### 39,99 $ — écarté
`config/pricing.json` (source unique lue par le backend **et** le front généré) porte **39 $**.
L'annuel 348 $ correspond à la mission. **Décision fondateur : 39 $ maintenu.** Note pour
plus tard : un prix Stripe ne se modifie pas — changer le montant obligerait à créer deux
nouveaux objets Price et à repointer `STRIPE_PRICE_*`.

### La table d'accès — rien à créer
Elle existe : `accounts.db`, `SCHEMA_VERSION 7`, tables `accounts` / `subscriptions` /
`processed_webhooks` / `sessions` / `email_verifications` / `renewal_notices` /
`account_consents`. `subscriptions` ne contient que des identifiants Stripe opaques, un
statut et des dates — **aucune donnée de carte, jamais**. C'est exactement la minimisation
demandée (Loi 25). Aucune migration n'a été nécessaire : `blocked_region` est une **valeur**
de la colonne `status`, pas une colonne de plus.

---

## 6. Tests — ce qui est verrouillé, et par quoi

| Exigence de la mission | Où |
|---|---|
| Signature de webhook invalide rejetée | `test_pay3_payment_journey.py::TestRealStripeSignature` (vraie signature Stripe) + 400 dur sur signature absente |
| Doublon de courriel impossible | `test_pay3_payment_journey.py::TestOneEmailOneAccount` + `email_lower UNIQUE` |
| Chaque état affiche son écran **et aucun autre** | `webapp/tests/e2e/pay3-etats.spec.ts` — 8 cas × 2 gabarits, chacun asserte la **présence** de son écran et l'**absence** des marqueurs des autres |
| Pays hors CA/US refusé | `test_pay3_zone_and_health.py::TestOutOfZoneIsRefused` — prouvé par le **refus de données (402)**, pas par un drapeau écrit |
| « Payé mais pas d'accès » impossible | `test_pay3_payment_journey.py` (chemin faux Stripe) **+** `stripe-live-journey.spec.ts` (chemin réel) |
| Zone paramétrable sans ouvrir le monde | `TestZoneHelpers::test_blank_env_falls_back_to_the_default_zone` |
| Préavis bloqué tant que non validé | `TestRenewalNoticeLegalLock` (3 cas, dont « SMTP seul ne vaut pas approbation ») |

Résultats : backend **220/220**, `tsc` **0**, `build` vert, vitest **9/9**,
Playwright **16/16**.

---

## 7. Ce que tu dois faire, toi

1. **Secrets GitHub** (mode test) : `STRIPE_SECRET_KEY` = `sk_test_…`,
   `STRIPE_PRICE_MONTHLY`. Puis lancer `Stripe live journey` en `workflow_dispatch` et
   constater qu'il passe au vert — **c'est le premier vrai passage du filet**.
2. **Règle Radar** dans le tableau de bord Stripe :
   `Block if :card_country: not in ('CA','US')`. Le code est le filet, Radar est le mur.
3. **Render** : rien d'obligatoire à ajouter (`BILLING_ALLOWED_COUNTRIES` non posée = zone
   CA/US par défaut ; `RENEWAL_NOTICE_TEXT_APPROVED` reste à `0`).
4. **Avocat** : faire valider le texte du préavis annuel, puis seulement
   `RENEWAL_NOTICE_TEXT_APPROVED=1`.
5. **Le paiement de test complet et l'annulation complète, en direct, par toi** — la
   condition de fusion posée par la mission. Le test vivant prouve la mécanique ;
   ton passage prouve le parcours.
6. **Le passage des clés test aux clés réelles reste ton action**, jamais celle d'une mission.

---

## 8. Ce qui reste ouvert (dit, non fait)

- **Écrans `none` et `expired`** : ils partagent toujours la même vue (choix de formule),
  seule la phrase d'accroche change. Décision fondateur du 2026-09-13 : **c'est acceptable** —
  dans les deux cas l'action attendue est la même (choisir une formule), contrairement à
  `blocked_region` où proposer de payer serait faux. Noté ici pour que ce soit un choix et
  non un oubli.
- **`EMAIL_VERIFICATION_ENFORCED=0`** (PAY-3e, pour un environnement sans SMTP) crée en
  pratique un état de plus. Non refermé : c'est la soupape qui permet au test vivant de
  tourner dans un runner. À reconsidérer quand SMTP sera posé en production.
- **Le compteur de sauvetages** est en mémoire de processus (cf. §3).

---

## 9. ÉCART DE PRIX CONSTATÉ DANS STRIPE (2026-09-14) — à corriger avant toute vente

Constat fait en ouvrant le tableau de bord Stripe (mode test, compte
``acct_1U2xW9FiM5Kf1kQc``), produit **M.I.A MARKETS** (``prod_V3OVamPC9Neh3f``) :

| | Stripe (créé le 11 août) | Dépôt (PRIX-1, 1ᵉʳ août) |
|---|---|---|
| Mensuel | **39,99 $US** (``price_1U3HibFiM5Kf1kQcsjkxBzbS``) | **39 $US** |
| Annuel | **348,99 $US** | **348 $US** |

**Stripe est périmé, pas le dépôt.** PRIX-1 a tranché 39 $/348 $ le 1ᵉʳ août
(commit ``4814179``) ; les objets Price ont été créés dix jours plus tard avec des
montants qui ne correspondent à aucune décision. Trois éléments le confirment :

1. ``webapp/components/landing/__tests__/pricing-prix-1.test.ts`` liste ``39,99``
   parmi les **prix hérités interdits** (``STALE``) — aligner le dépôt sur Stripe
   ferait rougir la suite immédiatement ;
2. ``scripts/gen_pricing.mjs`` **refuse** un annuel non divisible par 12, pour que
   l'affichage « soit 29 $ par mois » soit exact. 348 ÷ 12 = 29 pile ;
   348,99 ÷ 12 = 29,0825, et afficher « 29,08 » serait un mensonge d'arrondi
   (29,08 × 12 = 348,96) ;
3. le commentaire de ``config/pricing.json`` dit « Amounts are whole USD ».

**Conséquence si rien n'est fait** : le site annonce 39 $ et Stripe prélève
39,99 $. Annoncer un prix et en facturer un autre n'est pas défendable sous la LPC.

**Correction décidée (fondateur, 2026-09-14)** : créer deux NOUVEAUX tarifs Stripe
à 39,00 $ et 348,00 $ (un prix Stripe ne se modifie pas), archiver les anciens, et
repointer ``STRIPE_PRICE_MONTHLY`` / ``STRIPE_PRICE_ANNUAL``. **Le code ne bouge
pas.** À refaire en mode **live** avant la mise en vente réelle : les objets Price
sont séparés entre les deux modes.

État au moment où ces lignes sont écrites : **non fait** — le tableau de bord Stripe
a cessé de répondre pendant l'opération.

### Radar — même visite

Les **8 règles** Radar du compte sont toutes des règles Stripe par défaut :
**aucune règle de pays**. Le trou G2 est donc confirmé côté Stripe aussi, pas
seulement dans le code. La condition ``Bloquer si :card_country: not in ('CA','US')``
est acceptée par l'éditeur Stripe (syntaxe valide) mais n'a pas pu être enregistrée.
⚠️ **Les règles Radar sont séparées entre mode test et mode live** — une règle posée
en test ne protège PAS la production.

---

## 10. LE FILET EST ARMÉ ET VÉRIFIÉ (2026-09-14)

La réserve du §1 — « ce workflow n'a jamais été exécuté » — est levée.

**Run vert : `34855383764`, 1 test passé en 7,8 s.** Trois webhooks Stripe réels
livrés et acceptés (`POST /api/billing/webhook` → 200), accès ouvert, mur de
données franchi. C'est la première preuve de bout en bout que la chaîne
paiement → événement signé → webhook → accès fonctionne.

### Ce qu'il a fallu corriger pour y arriver

Quatre passages rouges, quatre trouvailles — toutes dans le TEST, aucune dans le
produit :

1. `npm ci` nu échouait en ERESOLVE avant même d'installer Playwright.
   `webapp-ci.yml` utilisait déjà `--legacy-peer-deps` partout : alignement.
2. **Le test ne distinguait pas « paiement refusé » de « webhook mort ».** Il
   cliquait « payer » puis attendait l'accès ; un clic raté donnait donc le même
   verdict qu'un webhook mort, et envoyait chercher la panne au mauvais endroit.
   Corrigé : on vérifie que le paiement a abouti AVANT de mesurer quoi que ce soit.
3. Le bouton visé n'était pas le bon : « Apple Pay » et « Payer avec Link » sont
   AVANT « S'abonner » dans le DOM, et un sélecteur CSS à virgules se résout dans
   l'ordre du DOM, pas dans l'ordre écrit. `.first()` cliquait un portefeuille.
4. **Stripe refuse la soumission de Checkout depuis un navigateur automatisé.**
   La page présente une case « I am an AI agent acting on behalf of someone
   else » ; même cochée, la session interrogée côté Stripe restait
   `status: open`, `payment_status: unpaid`, `subscription: null`.

### Décision d'architecture qui en découle

La page Checkout appartient à Stripe, et Stripe la protège délibérément contre
l'automatisation. Ce qui nous appartient, c'est ce qui se passe APRÈS le
paiement — et c'est là que se loge la panne que cette mission existe pour
empêcher. Le test crée donc l'abonnement par l'**API Stripe réelle** (jeton de
test `tok_visa`, prix récurrent, mode test) et mesure la chaîne réelle : vrais
événements signés → `stripe listen` → notre vérification de signature → accès.

Invariants conservés : aucun faux client Stripe, aucun appel à `/sync` (qui
rendrait le test vert sur un webhook mort), mur vérifié à 402 AVANT paiement
(sans quoi un vert ne prouverait rien), `error_if_incomplete` pour qu'un paiement
raté ne puisse jamais ressembler à un webhook mort.

Non couvert : le rendu de la page Checkout elle-même. C'est la surface de Stripe,
et le passage manuel du fondateur — condition de fusion — la couvre.

### Reste en attente

| | État |
|---|---|
| Prix Stripe 39 $/348 $ (cf. §9) | **non fait** — tableau de bord gelé pendant l'opération |
| Règle Radar CA/US | **non fait** — bouton d'enregistrement sans réaction |
| Secrets GitHub | ✅ posés (`STRIPE_SECRET_KEY`, `STRIPE_PRICE_MONTHLY`) |

⚠️ `STRIPE_PRICE_MONTHLY` pointe aujourd'hui sur le prix à **39,99 $**
(`price_1U3HibFiM5Kf1kQcsjkxBzbS`), le seul qui existe. À repointer sur le
nouveau prix à 39 $ dès qu'il sera créé. Le montant n'affecte pas ce que le test
mesure — la mécanique — mais il affecte ce qu'un vrai client paierait.
