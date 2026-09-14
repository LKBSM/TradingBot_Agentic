# Sauvegardes SQLite — mise en place, vérification, restauration

**BKP-1.** Ce document est le mode d'emploi. Le raisonnement (pourquoi ce
fournisseur, pourquoi ce mécanisme, ce qui a été testé) est dans
`docs/audits/AUDIT-bkp-1.md`.

---

## Ce qui est en place, en une page

| | |
|---|---|
| **Ce qui est sauvegardé** | toutes les `*.db` de `DATA_DIR` (`/app/data` sur Render), découvertes à l'exécution — un magasin ajouté demain est couvert sans rien changer |
| **Comment** | `sqlite3.Connection.backup()`, l'API de sauvegarde en ligne de SQLite — **jamais** une copie de fichier |
| **Où** | Cloudflare R2, un objet `.tar.gz` par jour |
| **Quand** | tous les jours à **03:17 UTC**, depuis le processus `mia-backend` lui-même |
| **Combien de temps** | 30 jours, purge automatique après chaque sauvegarde réussie |
| **Qui crie** | un workflow GitHub Actions à **05:40 UTC** → courriel de GitHub si ça a raté ; journal `ERROR` côté Render ; champ `backup` sur `/health` |
| **Coût** | 0 $/mois sous le palier gratuit de 10 Go de R2 (≈ 0,11 $/mois même à 17 Go) |

**Une sauvegarde de 89 Mo de bases pèse 15,8 Mo et prend 14 s** (mesuré, cf. l'audit).

### Pourquoi la sauvegarde tourne *dans* le backend

Parce que Render l'impose : *« You can't add a disk to a cron job service »*, et
*« A persistent disk is accessible by only a single service instance »*. Un Cron
Job Render tournerait dans un conteneur voisin et ne verrait jamais `/app/data`.
Le seul processus qui peut lire ces bases est `mia-backend`. Le planificateur y
vit donc aussi (APScheduler, déjà utilisé par `src/intelligence/scheduler.py`).

Conséquence assumée : si le backend est mort, la sauvegarde ne tourne pas. C'est
exactement pour ça que le vérificateur, lui, tourne **chez GitHub** — à
l'extérieur.

---

## 1. Mise en place (à faire une fois)

### 1.1 Créer le bucket R2

1. Cloudflare → **R2** → **Create bucket**. Nom : `mia-backups`. Région :
   *Automatic*. Pas d'accès public — ce bucket ne doit jamais être lisible depuis
   le web.
2. Noter l'**endpoint S3** affiché sur la page du bucket :
   `https://<account-id>.r2.cloudflarestorage.com`

### 1.2 Créer DEUX jetons, pas un

Cloudflare → R2 → **Manage API tokens** → *Create API token*.

| Jeton | Permission | Portée | Pour qui |
|---|---|---|---|
| `mia-backup-write` | **Object Read & Write** | le seul bucket `mia-backups` | le backend Render |
| `mia-backup-verify` | **Object Read only** | le seul bucket `mia-backups` | GitHub Actions |

Deux jetons parce que le vérificateur n'a aucune raison de pouvoir supprimer une
sauvegarde. Si son jeton fuite par les journaux d'un workflow public, il ne
permet que de lire.

Chaque jeton donne un *Access Key ID* et un *Secret Access Key*. Le secret n'est
affiché **qu'une fois**.

### 1.3 Renseigner Render

Render → service `mia-backend` → **Environment**. Les clés sont déjà déclarées
dans `render.yaml` avec `sync: false` ; il ne reste qu'à poser les valeurs :

```
BACKUP_S3_ENDPOINT           https://<account-id>.r2.cloudflarestorage.com
BACKUP_S3_BUCKET             mia-backups
BACKUP_S3_ACCESS_KEY_ID      <Access Key ID du jeton mia-backup-write>
BACKUP_S3_SECRET_ACCESS_KEY  <Secret du jeton mia-backup-write>
```

Déjà posées par le blueprint, à ne pas retoucher : `BACKUP_ENABLED=1`,
`BACKUP_HOUR_UTC=3`, `BACKUP_MINUTE_UTC=17`, `BACKUP_RETENTION_DAYS=30`,
`BACKUP_STAGING_DIR=/tmp`, `BACKUP_S3_REGION=auto`.

> `BACKUP_STAGING_DIR=/tmp` n'est pas un détail : les instantanés sont fabriqués
> **hors** du disque de données, sur le système de fichiers éphémère du
> conteneur. Une sauvegarde ne doit jamais pouvoir remplir le volume qu'elle
> protège.

Au redéploiement, le démon constate qu'aucune sauvegarde n'existe et en lance une
**tout de suite** (rattrapage au démarrage) — inutile d'attendre 03:17.

### 1.4 Renseigner GitHub

Dépôt → **Settings** → *Secrets and variables* → **Actions** → *New repository
secret* :

```
BACKUP_S3_ENDPOINT           https://<account-id>.r2.cloudflarestorage.com
BACKUP_S3_BUCKET             mia-backups
BACKUP_S3_ACCESS_KEY_ID      <Access Key ID du jeton mia-backup-verify>
BACKUP_S3_SECRET_ACCESS_KEY  <Secret du jeton mia-backup-verify>
```

Puis, pour ne pas attendre le lendemain : onglet **Actions** → *Backup verify* →
**Run workflow**. Il doit passer au vert.

Vérifier aussi, une fois, que GitHub a bien le droit d'envoyer le courriel
d'échec : *Settings* du compte → *Notifications* → **Actions** → « Send
notifications for failed workflows only » coché.

### 1.5 Vérifier de vos yeux

```bash
# Dans l'onglet Shell de mia-backend, sur Render :
python scripts/backup_sqlite.py list
```

Une ligne doit apparaître. Sinon, voir « Quand l'alerte se déclenche » plus bas.

---

## 2. Restaurer

C'est le seul chapitre qui compte vraiment. **À lire avant d'en avoir besoin.**

### 2.1 Répétition (sans rien toucher à la production)

À faire au moins une fois par trimestre, dans le Shell Render :

```bash
python scripts/backup_sqlite.py restore --target /tmp/repetition
```

Chaque base restaurée est vérifiée deux fois : SHA-256 contre le manifeste, puis
`PRAGMA integrity_check`. La sortie doit montrer `integrity_check=ok` sur chaque
ligne. Puis :

```bash
rm -rf /tmp/repetition
```

### 2.2 Restauration réelle après sinistre

**Ordre à respecter.** Restaurer par-dessus des bases vivantes pendant que le
backend écrit produit exactement le désastre qu'on veut éviter.

```bash
# 1. ARRÊTER les écritures. Render → mia-backend → Settings → Suspend Service.
#    (Ou, si le service ne démarre plus, il est déjà arrêté.)

# 2. Restaurer dans un répertoire NEUF, jamais directement sur /app/data.
python scripts/backup_sqlite.py restore --target /app/data/_restore

# 3. Lire la sortie. Chaque base doit dire integrity_check=ok.
#    Une seule ligne qui ne dit pas "ok" → STOP, voir §2.3.

# 4. Mettre les anciennes de côté (ne pas les supprimer : elles peuvent
#    contenir des heures que la sauvegarde n'a pas).
mkdir -p /app/data/_avant_restauration
mv /app/data/*.db /app/data/*.db-wal /app/data/*.db-shm /app/data/_avant_restauration/ 2>/dev/null

# 5. Mettre les restaurées en place.
mv /app/data/_restore/*.db /app/data/

# 6. Redémarrer. Render → Resume Service.

# 7. Contrôler.
curl -s https://<backend>/health | python -m json.tool | grep -A8 '"backup"'
```

Les sidecars `-wal`/`-shm` ne sont volontairement **pas** dans l'archive : l'API
de sauvegarde en ligne replie déjà le WAL dans l'instantané. SQLite les recrée
tout seul au premier accès.

### 2.3 Restaurer une seule base

Perdre `accounts.db` n'oblige pas à remonter les bougies de 2019 :

```bash
python - <<'PY'
from src.persistence.backup_service import restore_latest
print(restore_latest("/app/data/_restore", force=False))
PY
# puis ne déplacer que le fichier voulu
```

Ou, plus simple, restaurer tout dans `_restore` et ne déplacer que celui-là.

### 2.4 Revenir à une date précise

```bash
python scripts/backup_sqlite.py list          # choisir la ligne voulue
python scripts/backup_sqlite.py restore --key mia-backup-20260910T031700Z.tar.gz \
    --target /tmp/le-10-septembre
```

---

## 3. Quand l'alerte se déclenche

Le workflow *Backup verify* a échoué. Son résumé nomme le problème. Les trois
cas, et quoi faire :

### « no backup found in the store » / « cannot list the backup store »

Rien n'arrive dans le bucket, ou on ne peut même pas le lire.

1. `/health` du backend : le bloc `backup` dit-il quelque chose ?
   `last_error` porte la raison.
2. Journaux Render, filtrer sur `backup:` — le démon écrit une ligne `ERROR` avec
   la cause exacte à chaque échec.
3. Causes les plus courantes, dans l'ordre : une des quatre variables
   `BACKUP_S3_*` absente ou mal copiée côté Render ; le jeton R2 révoqué ou
   expiré ; le jeton créé en *Read only* au lieu de *Read & Write* ; le bucket
   renommé.
4. Test immédiat depuis le Shell Render : `python scripts/backup_sqlite.py run`.

### « the most recent backup is Nh old »

La sauvegarde ne tourne plus. Le backend est-il vivant ? A-t-il redémarré en
boucle ? Le démon ne démarre que si `BACKUP_ENABLED=1` **et** qu'une destination
est configurée — s'il manque la destination, il écrit
`NOTHING WILL BE BACKED UP` au niveau `ERROR` au démarrage. Chercher cette ligne.

Relancer à la main, tout de suite : `python scripts/backup_sqlite.py run`.

### « weighs N bytes, X% away from the median »

Le plus insidieux : la sauvegarde a « réussi » et ne contient presque rien. Le
plus souvent, `DATA_DIR` pointe ailleurs qu'avant (une base a disparu du
répertoire), ou un magasin a été purgé.

```bash
python scripts/backup_sqlite.py databases   # ce qui serait sauvegardé, et le poids
```

Comparer à la liste de la veille (dans le résumé du workflow, ou
`last_databases` de `/health`). **Ne pas ignorer cette alerte** en se disant que
la taille bougera bien. Si la variation est légitime (une purge volontaire), la
tolérance se règle avec `BACKUP_SIZE_TOLERANCE` (0,40 par défaut).

---

## 4. Utiliser la ligne de commande

Toutes les sous-commandes sortent en `0` si tout va bien, `1` sinon.

```bash
python scripts/backup_sqlite.py run          # sauvegarder maintenant + purger
python scripts/backup_sqlite.py list         # ce que contient le stockage
python scripts/backup_sqlite.py verify       # le verdict (sortie 1 si malade)
python scripts/backup_sqlite.py databases    # ce qui serait sauvegardé, et le poids
python scripts/backup_sqlite.py restore --target <rep>
python scripts/backup_sqlite.py snapshot --out ./backups   # local, sans envoi
```

`--json` sur n'importe laquelle pour une sortie exploitable par une machine.

### Essayer sans toucher à R2

`BACKUP_DESTINATION=file:///chemin` remplace le stockage objet par un
répertoire. C'est le même code, la même archive, la même vérification — c'est ce
que la suite de tests utilise.

```bash
DATA_DIR=./data BACKUP_ENABLED=1 BACKUP_DESTINATION=file:///tmp/r2-simule \
  python scripts/backup_sqlite.py run
```

---

## 5. Toutes les variables

| Variable | Défaut | Rôle |
|---|---|---|
| `BACKUP_ENABLED` | *(vide = off)* | `1` arme le démon quotidien |
| `BACKUP_S3_ENDPOINT` | — | `https://<account-id>.r2.cloudflarestorage.com` |
| `BACKUP_S3_BUCKET` | — | nom du bucket |
| `BACKUP_S3_ACCESS_KEY_ID` | — | jeton R2 — **jamais dans le dépôt** |
| `BACKUP_S3_SECRET_ACCESS_KEY` | — | jeton R2 — **jamais dans le dépôt** |
| `BACKUP_S3_REGION` | `auto` | R2 l'ignore ; `auto` est sa convention |
| `BACKUP_S3_PREFIX` | *(vide)* | préfixe de clé si le bucket est partagé |
| `BACKUP_DESTINATION` | *(vide)* | `file:///chemin` → répertoire local au lieu de S3 |
| `BACKUP_HOUR_UTC` / `BACKUP_MINUTE_UTC` | `3` / `17` | heure UTC du déclenchement |
| `BACKUP_RETENTION_DAYS` | `30` | fenêtre de rétention |
| `BACKUP_STAGING_DIR` | temp du système | où sont fabriqués les instantanés |
| `BACKUP_MAX_AGE_HOURS` | `26` | au-delà, la sauvegarde est déclarée périmée |
| `BACKUP_SIZE_TOLERANCE` | `0.40` | écart toléré à la médiane des 7 dernières |
| `DATA_DIR` | `./data` | répertoire des bases |

Les identifiants ne vivent **que** dans l'environnement : le dépôt n'en contient
aucun, et un test (`test_no_backup_credential_is_committed`) le vérifie à chaque
exécution de la suite.

---

## 6. Ce que ce dispositif ne fait pas

Écrit ici pour que personne ne s'en découvre protégé par erreur.

- **Point de reprise : 24 h.** Une panne à 03:16 UTC coûte presque une journée
  d'écritures. Réduire ce chiffre demande une réplication continue (Litestream),
  pas une sauvegarde quotidienne.
- **Pas de chiffrement de notre côté.** R2 chiffre au repos ; nous n'ajoutons pas
  de couche. Quiconque tient le jeton lit les données — d'où le jeton scopé à un
  seul bucket, et le jeton de lecture séparé.
- **Pas de copie hors R2.** Si le compte Cloudflare est perdu, les sauvegardes le
  sont. Une seconde destination est un chantier à part.
- **Le contenu n'est pas jugé.** `integrity_check` prouve que la base est saine,
  pas que la logique métier y a écrit ce qu'il fallait.
- **Une seule instance.** Le disque Render interdit le passage à plusieurs
  instances ; si cela change un jour, ce démon devra apprendre à ne s'exécuter
  que sur une seule d'entre elles.
