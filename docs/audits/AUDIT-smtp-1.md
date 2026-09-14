# AUDIT SMTP-1 — Envoi de courriels transactionnels

**Date** : 2026-09-13
**Branche** : `fix/smtp-1-envoi-courriels`, worktree dédié, depuis `origin/main`
(9de5138).
**Mode d'emploi** : [`docs/ops/envoi-courriels.md`](../ops/envoi-courriels.md).

Trouvé en marge de BKP-1, traité séparément pour ne pas le retenir derrière la
condition de fusion de la PR #221.

---

## Le défaut

Il ne s'agissait pas de « des courriels ne partent pas ». Trois faits qui, pris
ensemble, forment autre chose :

1. `EMAIL_VERIFICATION_ENFORCED` vaut **`1` par défaut**
   (`src/api/subscription_gate.py:157`) et **n'était déclaré nulle part** dans
   `render.yaml` → le mur de confirmation est **debout** en production.
2. `src/api/routes/access.py:52` : `has_access = False` pour tout compte non
   confirmé. `enforce_access` bloque les routes de données de la même façon.
3. **`SMTP_HOST` n'était déclaré nulle part** dans `render.yaml` → la clé de ce
   mur n'est jamais envoyée.

**Conséquence en production : tout client qui s'inscrivait était enfermé dehors
définitivement.** Avec `BETA_LOCKDOWN=0` (inscriptions ouvertes) et
`SUBSCRIPTION_GATE_ENFORCED=1` (payer est la condition d'entrée), un client
pouvait payer et ne jamais obtenir l'accès.

Et la raison pour laquelle personne ne l'a vu : `email_verified()` **exempte le
propriétaire** (`role == "owner"`, semé déjà vérifié). Le compte du fondateur
fonctionnait parfaitement. Le produit avait l'air sain depuis le seul poste
d'où on le regardait.

Les deux réglages avaient pourtant été conçus comme une paire — le commentaire
d'origine dit « Turn it back ON once SMTP works ». Le mur a reçu son défaut à
`1`, SMTP n'a jamais été configuré, et les deux ont dérivé.

### Pourquoi c'est resté silencieux

Les mêmes vingt lignes de `smtplib` étaient **copiées-collées trois fois** :
`_send_reset_email` et `_send_verification_email` dans `routes/accounts.py`,
`_send_email` dans `billing/renewal_notices.py`. Chaque copie décidait seule quoi
faire sans SMTP, et chacune répondait « retourner `False`, discrètement ». La
seule trace était une ligne **`INFO`** — dans un flux de journaux JSON de
production, indiscernable de « rien ne se passe ».

---

## Décisions

Deux arbitrages tranchés par le fondateur.

### Le backend refuse de démarrer

Même famille que les trois gardes existants du `lifespan`
(`assert_stable_secret_configured`, `assert_public_urls_configured`, semis du
compte propriétaire). Un déploiement raté vaut mieux qu'un client débité pour un
compte qui ne s'activera jamais.

La condition est **volontairement étroite** : pas « SMTP doit exister », mais
**« le mur de confirmation ne doit pas se dresser sans moyen d'envoyer la clé
qui l'ouvre »**. C'est la contradiction précise qui enfermait les clients, et
c'est elle qui mérite un refus de déploiement.

Mur baissé + pas de SMTP → le déploiement **passe** (les nouveaux clients
atteignent le produit) mais la réinitialisation de mot de passe et les préavis
restent morts → `ERROR` au démarrage, pas un haussement d'épaules.

Hors production (`ENVIRONMENT` non posé), le garde est un **no-op** : le
développement local et toute la suite de tests continuent sans SMTP. C'est la
seule raison pour laquelle un refus de démarrage est acceptable ici.

### Brevo dans la procédure

300 courriels/jour gratuits, SMTP standard, authentification de domaine
classique. Le code n'a **rien** de spécifique à Brevo : tout relais convient.

---

## Ce qui a été fait

| Fichier | Changement |
|---|---|
| `src/api/mailer.py` | **nouveau** — l'unique expéditeur, le garde de démarrage, la vue `/health` |
| `src/api/routes/accounts.py` | les deux copies `smtplib` remplacées ; les deux `INFO` passés en `ERROR` |
| `src/billing/renewal_notices.py` | la troisième copie remplacée ; `ERROR` seulement si quelqu'un est réellement dû |
| `src/api/app.py` | `assert_email_delivery_configured()` dans le `lifespan` |
| `src/api/models.py`, `src/api/routes/health.py` | bloc `email` + dégradation du statut |
| `render.yaml` | `SMTP_*` déclarées (`sync: false`) + `EMAIL_VERIFICATION_ENFORCED` **explicite** |
| `docs/ops/envoi-courriels.md` | la procédure |
| `tests/test_smtp1_mailer.py` | 20 tests |

### Les quatre changements de comportement

1. **Un seul expéditeur.** `smtplib` n'apparaît plus qu'à un endroit pour le
   courrier transactionnel. Un test le verrouille
   (`test_the_three_transactional_mails_go_through_the_one_mailer`) : si une
   quatrième copie apparaît, il échoue.
2. **Bruyant au démarrage.** Le refus, avec un message qui nomme les deux
   sorties — configurer SMTP, ou baisser le mur sciemment.
3. **Bruyant à l'événement.** `ERROR` au lieu d'`INFO`, et le message dit ce que
   ça coûte au client (« a new account can never be confirmed, so it never gains
   access »), pas seulement que l'envoi a échoué. La ligne suivante nomme le
   compte bloqué.
4. **Visible.** `/health` porte `email.healthy`, et le service passe en
   `degraded` quand le mur se dresse sans clé — parce que de l'extérieur l'API a
   l'air parfaite pendant que personne ne peut entrer.

Le contrat des appelants est **inchangé** : `send_email` retourne `False` sans
lever quand SMTP est absent, et **lève** sur un échec réel d'envoi — exactement
ce que les trois sites d'appel attendaient déjà. Aucun `try/except` existant n'a
eu à bouger.

### Un défaut trouvé par le test qui devait le trouver

`test_wall_flag_reading_matches_the_subscription_gate` épingle la lecture du
drapeau par le mailer sur celle de la passerelle. Il a immédiatement échoué :

- la passerelle fait `os.environ.get(nom, "1")` → une valeur **vide** donne
  `""` → mur **baissé** ;
- ma première version faisait `(env.get(nom) or "1")` → une valeur vide retombait
  sur `"1"` → mur **debout**.

Avec `EMAIL_VERIFICATION_ENFORCED=""`, le garde aurait **refusé de démarrer pour
un mur qui n'était pas dressé**. Corrigé : seule une variable *absente* retombe
sur le défaut. C'est la même classe de dérive que celle qui a causé le défaut
d'origine — d'où le test.

---

## Tests

### 20 tests, tous verts

Ceux qui portent le raisonnement :

- `test_production_refuses_to_boot_with_the_wall_up_and_no_way_to_send` — et le
  message doit contenir `SMTP_HOST`, `EMAIL_VERIFICATION_ENFORCED`, « locked
  out » et le chemin de la procédure, sinon le test échoue.
- `test_the_wall_is_up_by_default_which_is_why_this_bites` — sans la variable du
  tout, c'est-à-dire l'état exact de la production avant ce correctif.
- `test_the_guard_never_touches_dev_ci_or_tests` — la contrepartie du
  fail-fast.
- `test_an_unconfigured_send_is_logged_at_error_not_info` — le cœur de la
  correction de silence.
- `test_a_real_delivery_failure_still_raises` — le contrat des appelants est
  préservé.
- `test_no_credential_is_ever_logged`.
- `test_renewal_notices_shout_only_when_someone_is_actually_due` — et le faux
  magasin **lève** si quoi que ce soit est marqué comme envoyé sans SMTP : un
  préavis dû reste dû.
- `test_health_endpoint_degrades_the_service_and_carries_the_block`.

### Démarrage simulé en production, les trois cas

| Environnement | Résultat |
|---|---|
| prod, mur debout (défaut), **aucun SMTP** — *l'état actuel* | **REFUSE** — `EmailDeliveryNotConfigured`, message complet |
| prod, mur baissé volontairement, aucun SMTP | **DÉMARRE**, avec un `ERROR` nommant ce qui reste cassé |
| prod, mur debout, SMTP configuré | **DÉMARRE** |

### Non-régression

- `test_smtp1_mailer.py` : **20 passés**
- `test_auth_reliability`, `test_renewal_notices_pay1`, `test_account_auth`,
  `test_auth`, `test_auth_completions_pay1` : **96 passés**
- `test_api`, `test_health_deep_endpoint`, `test_shutdown_lifespan`,
  `test_bootstrap_runtime` : **82 passés**

Aucun secret dans le dépôt : les valeurs sont `sync: false` dans `render.yaml`.

---

## Conséquence à accepter avant de fusionner

**Le prochain déploiement de `main` échouera** tant que vous n'aurez pas posé
soit les variables `SMTP_*`, soit `EMAIL_VERIFICATION_ENFORCED=0`. C'est le but
du garde ; ce n'est pas une surprise à découvrir en production. La marche à
suivre tient en §1 de `docs/ops/envoi-courriels.md`.

Le service tourne aujourd'hui dans l'état que le garde refusera. Autrement dit :
il fonctionne pour vous et pour les comptes déjà vérifiés, et pour personne
d'autre.

---

## Ce qui n'est pas traité

- **Aucune file ni réessai.** Un envoi raté est journalisé, pas repris. Le client
  peut redemander un code (`/verify-email/resend`).
- **Aucun suivi des rebonds** côté produit.
- **Aucun déblocage administrateur** des comptes coincés : il faut passer par la
  base (`mark_email_verified`) ou baisser le mur. Si des comptes ont déjà été
  créés en production depuis l'ouverture des inscriptions, **ils sont coincés** —
  à vérifier (`SELECT id, email FROM accounts WHERE email_verified = 0`) et à
  débloquer à la main une fois SMTP en place.
- **Un seul relais.** Brevo indisponible = aucun courriel, et le mur reste debout.
- **`src/live_trading/alerting.py` et `src/security/alert_manager.py`** gardent
  leur propre `smtplib` : ce sont des alertes d'exploitation internes, pas du
  courrier client, et elles sont hors du périmètre de ce correctif.
