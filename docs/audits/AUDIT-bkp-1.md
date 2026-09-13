# AUDIT BKP-1 — Sauvegarde quotidienne des bases SQLite

**Date** : 2026-09-13
**Branche** : `feat/bkp-1-sauvegarde`, worktree dédié, dérivée de `origin/main`
(9de5138). Le HEAD du dépôt principal était 122 commits en retard (e0dc69c) ;
rien n'a été audité contre lui.
**Mode d'emploi** (mise en place, restauration, que faire quand ça crie) :
[`docs/ops/sauvegardes-sqlite.md`](../ops/sauvegardes-sqlite.md).

---

## Le problème, tel qu'il était

`/app/data` sur Render était la copie **unique** de tout ce que le produit vend :
l'histoire de chaque zone, ses contacts datés, les comptes et leurs liens Stripe.
Render ne propose aucun instantané qu'un client puisse restaurer lui-même. Un
disque perdu, et le produit n'a plus rien à vendre.

Il existait bien un `scripts/backup_dr.py` (234 lignes, Sprint INFRA-2B.6). Il ne
protégeait rien :

1. Il faisait `tar.add()` sur `*.db`, `*.db-wal` et `*.db-shm` — **une copie de
   fichier à cru pendant que le processus écrit**. En WAL, les pages fraîches
   sont dans le `-wal` ; trois fichiers lus à trois instants différents donnent
   une base déchirée. Pire : les sommes SHA-256 de son manifeste auraient
   déclaré cette base déchirée « intacte ».
2. Sa docstring annonçait « optionally pushed to Backblaze B2 / S3 ». **Aucun
   code d'envoi n'existait.** Rien ne quittait le disque.
3. Il n'était planifié nulle part (unique référence :
   `tests/test_phase_2b_final_batch.py`).
4. Sa liste de bases était celle d'un sprint mort (`api_keys.db`,
   `audit_ledger.db`, `admin_action_log.db`, `idempotency.db`).

Il est laissé en place, intact : sa suite de tests s'appuie dessus et le
supprimer dépasse le périmètre. Le nouveau dispositif ne le réutilise pas.

---

## Décisions, et pourquoi

Les quatre choix ont été arbitrés par le fondateur après le diagnostic.

### Stockage : Cloudflare R2

Le coût n'a pas départagé — les deux candidats sont gratuits ou dérisoires :

| Hypothèse (bases sur le disque) | Archive/jour | Stocké à 30 j | R2 | B2 |
|---|---|---|---|---|
| ~92 Mo (taille mesurée ici) | 16 Mo | 0,49 Go | 0 $ | 0 $ |
| ~1 Go (niveau atteint le 2026-08-05) | ~190 Mo | 5,6 Go | 0 $ | 0 $ |
| ~3 Go | ~570 Mo | 17 Go | 0,11 $/mois | 0,05 $/mois |

Ce qui a départagé :

1. **Sortie de données gratuite et sans plafond.** Elle ne compte qu'un seul
   jour — celui de la restauration. On ne veut pas, ce jour-là, calculer un
   quota. B2 plafonne la sortie gratuite à 3× le stockage.
2. **`boto3>=1.28` était déjà en ligne 74 de `requirements.txt`** et déjà dans
   l'image `Dockerfile.api`. R2 parle S3. **Zéro nouvelle dépendance ajoutée.**
3. Jetons scopés à un seul bucket, et séparables en lecture seule.

Rien dans le code n'est spécifique à R2 : `S3Destination` marche contre B2 ou S3
en changeant l'endpoint.

### Planification : dans le processus backend — parce que Render l'impose

Ce n'est pas un arbitrage de confort. La documentation Render est catégorique :
*« You can't add a disk to a cron job service »* et *« A persistent disk is
accessible by only a single service instance »*. Un Cron Job Render tourne dans
un conteneur voisin et **ne verra jamais `/app/data`**. Le seul processus qui
peut lire ces bases est `mia-backend`.

Le démon est donc dans le backend, sur APScheduler — déjà une dépendance
(`requirements.txt` ligne 57), déjà utilisée par
`src/intelligence/scheduler.py`. **Aucun second mécanisme de planification n'a
été introduit**, conformément au §0bis de la mission. Le scheduler du démon lui
est propre pour qu'un raté de sauvegarde ne puisse pas perturber la cadence des
lectures de marché.

Conséquence assumée et compensée : si le backend est mort, la sauvegarde ne
tourne pas — d'où le vérificateur externe ci-dessous.

### Alerte : trois couches, la première hors de Render

| Couche | Où | Attrape « le backend est mort » ? |
|---|---|---|
| Workflow GitHub Actions quotidien (05:40 UTC) | **chez GitHub** | **oui** |
| Journal `ERROR` structuré | Render | non |
| Champ `backup` sur `/health` (+ `status: degraded`) | Render | non |

La couche qui compte est la première, et sa vertu principale est d'être
**ailleurs**. Une vérification qui tourne dans le processus qu'elle surveille
meurt avec lui. GitHub envoie automatiquement un courriel au propriétaire du
dépôt à chaque workflow en échec : pas de nouveau fournisseur, pas de coût.

Le courriel SMTP direct a été écarté : **`SMTP_HOST` n'est configuré nulle part
dans `render.yaml`** — le code existe (`src/billing/renewal_notices.py`) mais
aucun courriel ne part de la production aujourd'hui (cf. « observations de
bord »).

Les trois couches lisent la **même** fonction, `verify_backups()` : elles ne
peuvent pas se contredire.

### Périmètre : toutes les `*.db`, découvertes à l'exécution

La mission en nommait deux. `accounts.db` — comptes, empreintes de mots de passe,
liens d'abonnement Stripe — est la plus irremplaçable des dix et pèse 80 Ko.
Sauvegarder le répertoire entier coûte quelques centaines de kilo-octets par jour
et supprime le mode de défaillance « on avait sauvegardé les deux mauvais
fichiers ».

Découverte plutôt que liste en dur : un magasin ajouté dans six mois est couvert
sans que personne ait à y penser. Verrouillé par
`test_discovery_is_not_a_hard_coded_list`.

---

## Ce qui a été construit

| Fichier | Rôle |
|---|---|
| `src/persistence/sqlite_backup.py` | instantané cohérent, archive + manifeste, vérification, restauration. Aucun réseau. |
| `src/persistence/backup_storage.py` | destinations : R2/S3 (`boto3`) et répertoire local |
| `src/persistence/backup_service.py` | une exécution complète, rétention, verdict, état, vue `/health` |
| `src/persistence/backup_daemon.py` | le déclencheur quotidien + le rattrapage au démarrage |
| `scripts/backup_sqlite.py` | ligne de commande : `run`, `list`, `verify`, `restore`, `snapshot`, `databases` |
| `.github/workflows/backup-verify.yml` | le vérificateur externe |
| `docs/ops/sauvegardes-sqlite.md` | la procédure |
| `tests/test_bkp1_backup.py` | 50 tests |

Câblage : `src/api/app.py` (démarrage + arrêt propre), `src/api/dependencies.py`,
`src/api/models.py`, `src/api/routes/health.py`, `render.yaml`.
**3 045 lignes ajoutées, 0 supprimée.**

### Les trois décisions de conception qui portent tout

**L'API de sauvegarde en ligne, pas une copie.**
`sqlite3.Connection.backup()` copie page par page sous le verrouillage de la
source et **recommence** si un écrivain concurrent modifie une page déjà copiée.
La destination est toujours transactionnellement cohérente, pendant que le
backend écrit. L'instantané passe ensuite `PRAGMA integrity_check` : un
instantané qui échoue est une **erreur**, pas un avertissement — l'expédier
fabriquerait l'illusion d'une sauvegarde. Les sidecars `-wal`/`-shm` ne sont
délibérément pas dans l'archive : le WAL est déjà replié dedans.

**La purge après l'envoi, jamais avant, et jamais jusqu'au vide.**
Un envoi raté ne peut pas être la cause de la disparition de la veille. Et un
plancher de deux sauvegardes survit à tout : si l'horloge dérape ou que la
fenêtre est mal réglée, la rétention ne peut pas vider le stockage.

**Le verdict est rendu contre le stockage, pas contre notre souvenir.**
`verify_backups()` liste le bucket et répond à trois questions : y a-t-il quelque
chose ? la plus récente a-t-elle moins de 26 h ? sa taille tient-elle dans ±40 %
de la médiane des précédentes ? Ce dernier point vise le mode de défaillance
classique : le travail « réussit » et expédie une base tronquée. La médiane
exclut la sauvegarde du jour, pour qu'une mauvaise exécution ne traîne pas sa
propre référence avec elle. Et l'horodatage fait foi par le **nom** de l'objet,
pas par le `last_modified` du stockage, qu'un réenvoi déplacerait.

---

## Tests

### Suite automatisée — 50 tests, tous verts

`tests/test_bkp1_backup.py`. Le cycle complet (instantané → archive → envoi →
listage → purge → téléchargement → restauration → `integrity_check`) tourne
contre `LocalDirDestination`, **le même chemin de code** que la production avec
`S3Destination` substituée. Le côté S3 est couvert par un faux client : gestion
des préfixes de clé, pagination du listage, objets étrangers ignorés.

Les tests qui portent le raisonnement, pas la couverture :

- `test_snapshot_is_consistent_while_a_writer_is_hammering` — un fil d'exécution
  insère en boucle pendant l'instantané ; `integrity_check` doit dire `ok`. C'est
  la raison d'être de la mission, vérifiée.
- `test_verify_catches_a_backup_that_shrank` — quatre jours à 1 000 octets, puis
  40 : doit être déclaré malade.
- `test_verify_waits_for_enough_history_before_judging_size` — deux sauvegardes
  ne font pas une médiane ; refuser de juger vaut mieux qu'une fausse alerte au
  deuxième jour.
- `test_prune_can_never_empty_the_store` — horloge avancée de dix ans, fenêtre à
  un jour : deux sauvegardes survivent quand même.
- `test_run_backup_refuses_when_there_is_no_room` — et le message doit nommer
  `BACKUP_STAGING_DIR`, sinon le test échoue.
- `test_daemon_refuses_to_pretend_without_a_destination` — armé sans destination,
  le démon doit écrire `NOTHING WILL BE BACKED UP` au niveau `ERROR`.
- `test_the_app_boots_backs_up_and_says_so_on_health` — toute la chaîne à travers
  `create_app()`.
- `test_no_backup_credential_is_committed` — garde-fou permanent.

### Test réel — vraies bases, 89 Mo

Sur le répertoire `data/` réel de ce poste, 10 bases, 89,1 Mo :

```
snapshot accounts.db (81 920 octets) ok
snapshot calendar_cache.db (36 864) ok
snapshot candles.db (84 156 416) ok
snapshot kill_switch.db (40 960) ok
snapshot market_readings.db (7 602 176) ok
snapshot market_readings_live.db (376 832) ok
snapshot market_readings_viewcontrol.db (1 044 480) ok
snapshot narrative_cache.db (20 480) ok
snapshot news_cache.db (36 864) ok
snapshot signals.db (20 480) ok
archive mia-backup-20260913T215447Z.tar.gz
  10 bases, 93 417 472 octets source, 16 521 119 compressés (ratio 0,177) en 13,6 s
verify — 10 fichiers intacts
```

**89,1 Mo → 15,8 Mo en 13,6 s.**

### Restauration réelle

```
restored accounts.db                     integrity_check=ok
restored calendar_cache.db               integrity_check=ok
restored candles.db (84 156 416 octets)  integrity_check=ok
restored kill_switch.db                  integrity_check=ok
restored market_readings.db              integrity_check=ok
restored market_readings_live.db         integrity_check=ok
restored market_readings_viewcontrol.db  integrity_check=ok
restored narrative_cache.db              integrity_check=ok
restored news_cache.db                   integrity_check=ok
restored signals.db                      integrity_check=ok
```

`integrity_check=ok` prouve que le fichier est une base saine. Pour prouver que
c'est **la même** base, les 35 tables ont été recomptées des deux côtés :

| Base | Table | Source | Restauré |
|---|---|---|---|
| candles.db | candles_cache | 370 885 | 370 885 |
| market_readings.db | market_readings | 589 | 589 |
| market_readings.db | haiku_description_cache | 262 | 262 |
| news_cache.db | news_cache | 108 | 108 |
| … | *(35 tables au total)* | | |

**Tables divergentes : 0.**

### Les trois alertes, déclenchées pour de vrai

| Situation simulée | Verdict | Code de sortie |
|---|---|---|
| Stockage sain | `HEALTHY` | 0 |
| Sauvegarde du jour à 400 Ko contre une médiane de 16,5 Mo | `98% away from the median … over the 40% tolerance` | **1** |
| Plus rien depuis 3 jours | `is 72.0h old, over the 26h limit` | **1** |

Le code de sortie est ce sur quoi GitHub Actions s'appuie pour envoyer le
courriel.

### Non-régression

- `tests/test_bkp1_backup.py` : **50 passés**
- `test_health_deep_endpoint`, `test_health_deep_cache`,
  `test_scheduler_app_wiring`, `test_phase_2b_final_batch` : **100 passés**
- `test_bootstrap_runtime`, `test_shutdown_coordinator`,
  `test_shutdown_lifespan`, `test_admin_action_log`, `test_auth`,
  `test_account_auth` : **112 passés**
- `test_api.py` : **passé**
- `tests/test_smoke_e2e.py` : 2 échecs
  (`test_api_health_endpoint_in_testing_mode`,
  `test_api_narratives_no_auth_in_testing_mode`) — **vérifiés PRÉEXISTANTS** :
  les mêmes deux échouent sur `origin/main` avec mes modifications remisées.

Un cas a mis un défaut au jour pendant l'intégration :
`test_auto_register_picks_up_default_handlers` vérifie la liste **exacte** des
gestionnaires d'arrêt. Ma première version en enregistrait un inconditionnel.
Corrigé en n'enregistrant que si `BACKUP_ENABLED` est armé — meilleur choix de
toute façon — et deux tests ajoutés pour figer les deux cas.

### Aucun secret dans le dépôt

`grep` avant commit sur `AKIA`, `aws_secret`, endpoints R2 littéraux
(`[0-9a-f]{32}.r2.cloudflarestorage.com`) et toute valeur affectée à
`BACKUP_S3_ACCESS_KEY_ID` / `BACKUP_S3_SECRET_ACCESS_KEY` : **aucune
correspondance**. Les quatre variables sont `sync: false` dans `render.yaml`
(valeurs posées au tableau de bord) et `${{ secrets.* }}` dans le workflow.

---

## Ce qui reste à faire — côté fondateur

Rien de tout cela ne peut être fait depuis ce poste : il faut un compte
Cloudflare et les droits du dépôt.

1. **Créer le bucket R2** `mia-backups` (§1.1 de la procédure).
2. **Créer deux jetons** : `mia-backup-write` (Object Read & Write) pour Render,
   `mia-backup-verify` (Object **Read only**) pour GitHub. Deux, parce que le
   vérificateur n'a aucune raison de pouvoir supprimer une sauvegarde.
3. **Poser les quatre valeurs sur Render**, puis redéployer. Le rattrapage au
   démarrage lance une sauvegarde immédiatement — inutile d'attendre 03:17.
4. **Poser les quatre secrets sur GitHub**, puis lancer *Backup verify* à la main
   une fois.
5. **Voir la sauvegarde apparaître** dans le bucket. C'est la condition de merge
   fixée par la mission.

Tant que l'étape 3 n'est pas faite, `BACKUP_ENABLED=1` sans destination fait
écrire au démarrage, au niveau `ERROR` : `NOTHING WILL BE BACKED UP`. C'est
voulu — « armé mais aveugle » est le pire des états, il devait être bruyant.

---

## Deux observations de bord (hors périmètre, signalées)

1. **`SMTP_HOST` n'est défini nulle part dans `render.yaml`.** En production,
   `_send_email` retourne `False` en silence : ni les courriels de vérification
   d'adresse (`src/api/routes/accounts.py`), ni les préavis de renouvellement
   (`src/billing/renewal_notices.py`) ne partent. À traiter avant d'encaisser —
   c'est une obligation d'information client, pas un confort.
2. **Le disque C: de ce poste était plein** (0 octet libre sur 476 Go), ce qui
   bloquait le test de restauration. Sur accord du fondateur, quatre répertoires
   de compilation `webapp/.next` ont été purgés (`wt-lp-3`, `wt-vz-4`, `wt-vz-5`,
   `wt-sc-4`), libérant 644 Mo. Ce sont des artefacts reconstructibles. Le disque
   reste à 99,9 % : le problème de fond n'est pas réglé.

---

## Les limites, écrites noir sur blanc

Pour que personne ne se découvre protégé par erreur :

- **Point de reprise : 24 h.** Une panne à 03:16 UTC coûte presque une journée
  d'écritures. Descendre sous ce seuil demande une réplication continue
  (Litestream), pas une sauvegarde quotidienne.
- **Pas de chiffrement de notre côté.** R2 chiffre au repos ; nous n'ajoutons
  rien. Qui tient le jeton lit les données — d'où le scope à un seul bucket.
- **Une seule destination.** Compte Cloudflare perdu, sauvegardes perdues.
- **Le contenu n'est pas jugé.** `integrity_check` prouve que la base est saine,
  pas que la logique métier y a écrit ce qu'il fallait.
- **Une seule instance.** Le disque Render interdit l'horizontal ; si cela change,
  ce démon devra apprendre à ne tourner que sur une instance.
- **Le vérificateur externe dépend de GitHub Actions.** Si le dépôt est archivé
  ou les workflows désactivés, l'alerte se tait. Le silence prolongé de *Backup
  verify* est lui-même un signal à surveiller.
