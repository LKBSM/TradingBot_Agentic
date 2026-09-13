# Envoi de courriels — mise en place et diagnostic

**SMTP-1.** Ce document est le mode d'emploi. Le raisonnement et la preuve sont dans
[`docs/audits/AUDIT-smtp-1.md`](../audits/AUDIT-smtp-1.md).

---

## Pourquoi ce n'est pas optionnel

Trois courriels transactionnels partent du backend :

| Courriel | Sans lui |
|---|---|
| **Code de confirmation d'adresse** | un nouveau compte n'est **jamais** confirmé, donc n'obtient **jamais** l'accès |
| **Réinitialisation de mot de passe** | un client enfermé dehors n'a aucun retour possible |
| **Préavis 30 jours avant un prélèvement annuel** | le client est débité sans l'avertissement que la loi attend |

Le premier est le grave. `EMAIL_VERIFICATION_ENFORCED` vaut **1 par défaut**, et
`src/api/routes/access.py` refuse `has_access` à tout compte non confirmé. Le code
qui lève ce mur **arrive par courriel**. Sans SMTP, le mur est debout et sa clé
n'est jamais envoyée : **tout client qui s'inscrit est enfermé dehors
définitivement.**

Le compte propriétaire est exempté (`role == "owner"`, semé déjà vérifié) — c'est
précisément pour ça que la panne est invisible depuis le compte du fondateur.

**Le backend refuse maintenant de démarrer dans cet état.** Le message d'erreur
nomme les deux sorties.

---

## 1. Mise en place avec Brevo

Palier gratuit : 300 courriels/jour, ce qui est très au-dessus d'un lancement.
Le code n'a rien de spécifique à Brevo : tout relais SMTP convient, seules les
valeurs changent.

### 1.1 Compte et authentification du domaine

1. Créer un compte sur brevo.com.
2. **Senders, Domains & Dedicated IPs** → **Domains** → *Add a domain* :
   `mia.markets`.
3. Brevo affiche des enregistrements DNS (DKIM, `brevo-code`, et un DMARC
   conseillé). Les ajouter chez le registrar du domaine, puis revenir cliquer
   *Authenticate*.

> Cette étape n'est pas décorative. Un expéditeur non authentifié part en
> indésirable ou est refusé — un code de confirmation qui atterrit dans les
> indésirables enferme le client dehors exactement comme un courriel non envoyé.
> Attendre que le domaine soit **vert** chez Brevo avant de considérer la mise en
> place terminée.

### 1.2 Créer une clé SMTP

**SMTP & API** → onglet **SMTP** → *Generate a new SMTP key*.

⚠️ C'est une **clé SMTP**, pas la clé d'API v3. C'est l'erreur classique : les
deux existent, et la clé d'API ne fonctionne pas en SMTP.

La page affiche aussi le **SMTP login** (votre adresse de connexion Brevo).

### 1.3 Renseigner Render

Render → service `mia-backend` → **Environment**. Les clés sont déjà déclarées
dans `render.yaml` avec `sync: false` ; il ne reste qu'à poser les valeurs :

```
SMTP_HOST      smtp-relay.brevo.com
SMTP_PORT      587                      (déjà posé par le blueprint)
SMTP_USER      <le SMTP login affiché par Brevo>
SMTP_PASSWORD  <la clé SMTP>
SMTP_FROM      no-reply@mia.markets     ← sur le domaine authentifié à l'étape 1.1
```

`EMAIL_VERIFICATION_ENFORCED=1` est déjà posé explicitement par le blueprint.

### 1.4 Vérifier

Redéployer, puis :

```bash
curl -s https://<backend>/health | python -m json.tool | grep -A8 '"email"'
```

Attendu : `"configured": true`, `"healthy": true`.

Puis le vrai test, celui qui compte — **créer un compte de bout en bout avec une
adresse que vous relevez** et confirmer qu'il reçoit le code, qu'il ne tombe pas
dans les indésirables, et que la confirmation donne bien l'accès.

---

## 2. Variables

| Variable | Défaut | Rôle |
|---|---|---|
| `SMTP_HOST` | *(vide)* | **l'interrupteur** — vide = aucun envoi |
| `SMTP_PORT` | `587` | 587 STARTTLS ; 465 implique `SMTP_STARTTLS=0` |
| `SMTP_USER` | *(vide)* | identifiant — **jamais dans le dépôt** |
| `SMTP_PASSWORD` | *(vide)* | clé SMTP — **jamais dans le dépôt** |
| `SMTP_FROM` | `SMTP_USER`, sinon `no-reply@mia.markets` | expéditeur, sur un domaine authentifié |
| `SMTP_STARTTLS` | `1` | mettre `0` seulement pour un port SSL implicite (465) |
| `SMTP_TIMEOUT_S` | `10` | délai d'attente de la connexion |
| `EMAIL_VERIFICATION_ENFORCED` | `1` | le mur de confirmation. **Ne pas baisser pour contourner un SMTP cassé sans le savoir.** |

Les identifiants ne vivent **que** dans l'environnement.

---

## 3. Diagnostic

### Le backend refuse de démarrer : « the email-verification wall is up but no email can be sent »

C'est le garde, et il a raison : le déploiement enfermerait dehors tout nouveau
client. Deux sorties, au choix :

- **configurer SMTP** (§1) — la bonne ;
- **baisser le mur en connaissance de cause** : `EMAIL_VERIFICATION_ENFORCED=0`.
  Les nouveaux comptes atteignent alors le produit sans confirmer d'adresse, et
  la réinitialisation de mot de passe reste indisponible. À ne faire que
  temporairement, et à remettre à `1` dès que SMTP fonctionne.

### `/health` dit `"healthy": false` dans le bloc `email`

Le mur est debout sans moyen d'envoyer. Le service répond normalement par
ailleurs — de l'extérieur tout a l'air parfait pendant que personne ne peut
entrer. C'est précisément ce que ce champ existe pour rendre visible.

### Journaux : `email NOT DELIVERED (verification)`

Un client précis vient d'être bloqué. La ligne suivante nomme le compte :
`account id=… is now stuck behind the email-verification wall`.

Pour le débloquer à la main, en attendant : confirmer son adresse côté base
(`mark_email_verified`), ou baisser le mur.

### `SMTPAuthenticationError` / `535`

Presque toujours la **clé d'API v3 utilisée à la place de la clé SMTP** (§1.2).
Sinon : `SMTP_USER` doit être le *SMTP login* Brevo, pas l'adresse d'expédition.

### Les courriels partent mais n'arrivent pas

Domaine non authentifié, ou `SMTP_FROM` sur un autre domaine que celui
authentifié. Vérifier que le domaine est **vert** chez Brevo et que SPF/DKIM sont
en place. Le journal Brevo (**Transactional → Logs**) dit ce qui a été accepté,
différé ou rejeté, et pourquoi.

### Le préavis de renouvellement ne part pas

`STRIPE_PRICE_ANNUAL` doit être posé, sinon le travail s'arrête avant.
Si des clients sont dus et que SMTP manque, le journal porte un `ERROR` nommant
le nombre de comptes concernés — et **rien n'est marqué comme envoyé**, donc le
préavis reste dû et repartira une fois SMTP branché.

---

## 4. Ce que ce dispositif ne fait pas

- **Pas de file d'attente ni de réessai.** Un envoi qui échoue est journalisé,
  pas repris. Le client peut redemander un code (`/verify-email/resend`), mais
  rien ne le refait tout seul.
- **Pas de suivi des rebonds.** Une adresse invalide est visible dans les
  journaux Brevo, pas dans le produit.
- **Pas de limite d'envoi côté application.** Le palier de 300/jour de Brevo
  n'est pas surveillé ici ; un dépassement se verra dans les journaux comme un
  échec d'envoi.
- **Un seul relais.** Brevo indisponible = aucun courriel, et le mur reste
  debout.
