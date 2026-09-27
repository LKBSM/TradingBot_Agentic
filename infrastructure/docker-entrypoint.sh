#!/bin/sh
# Point d'entrée du conteneur API — répare la propriété du disque monté, puis
# redescend les privilèges avant de lancer l'application.
#
# POURQUOI CE FICHIER EXISTE
# --------------------------
# Le Dockerfile crée /app/data et le donne à l'utilisateur `sentinel` (uid
# 10001) — mais ce `chown` a lieu À LA CONSTRUCTION de l'image. Au démarrage,
# Render monte le disque persistant PAR-DESSUS /app/data, ce qui remplace le
# répertoire construit par un volume neuf appartenant à root. L'application,
# qui tourne en `sentinel`, ne peut alors plus y écrire : la toute première
# écriture SQLite échoue avec
#
#     sqlite3.OperationalError: disk I/O error
#
# levée par `PRAGMA synchronous=NORMAL` dans SignalStore, AVANT que le serveur
# n'ouvre un port. Render voit des plantages répétés et finit par suspendre le
# service — c'est exactement ce qui est arrivé le 2026-09-14.
#
# La correction doit avoir lieu À L'EXÉCUTION, après le montage, et exige root :
# d'où ce script, lancé en root, qui `chown` le volume puis abandonne ses
# privilèges pour exécuter la commande passée en arguments.
#
# ⚠️ Render REMPLACE l'ENTRYPOINT du Dockerfile par son `dockerCommand`. Ce
# script doit donc être appelé EXPLICITEMENT depuis `dockerCommand` dans
# render.yaml — le déclarer en ENTRYPOINT ne suffirait pas, il serait ignoré.
set -e

DATA_DIR="${DATA_DIR:-/app/data}"
LOG_DIR="/app/logs"
APP_UID=10001
APP_GID=10001

# Diagnostic imprimé À CHAQUE démarrage. Si un jour la cause est ailleurs (disque
# plein, montage absent), ces trois lignes le disent immédiatement dans les logs
# au lieu de laisser deviner.
echo "[entrypoint] uid=$(id -u) gid=$(id -g) data_dir=${DATA_DIR}"
mkdir -p "${DATA_DIR}" "${LOG_DIR}" 2>/dev/null || true
ls -ldn "${DATA_DIR}" 2>/dev/null || echo "[entrypoint] ${DATA_DIR} illisible"
df -h "${DATA_DIR}" 2>/dev/null | tail -1 || true

if [ "$(id -u)" = "0" ]; then
  # Root : on rend le volume écrivable par l'utilisateur applicatif.
  chown -R "${APP_UID}:${APP_GID}" "${DATA_DIR}" "${LOG_DIR}" 2>/dev/null \
    || echo "[entrypoint] AVERTISSEMENT : chown de ${DATA_DIR} impossible"

  # Vérification franche : si l'écriture échoue ENCORE, le problème n'est pas la
  # propriété (disque plein, volume en lecture seule…). On le dit ici plutôt que
  # de laisser SQLite lever une erreur illisible trois appels plus loin.
  if su -s /bin/sh -c "touch '${DATA_DIR}/.write-probe' && rm -f '${DATA_DIR}/.write-probe'" sentinel 2>/dev/null; then
    echo "[entrypoint] ${DATA_DIR} est écrivable par sentinel — OK"
  else
    echo "[entrypoint] ERREUR : ${DATA_DIR} reste NON écrivable par sentinel."
    echo "[entrypoint] Causes possibles : disque plein, volume en lecture seule,"
    echo "[entrypoint] ou disque non monté. Voir le 'df' ci-dessus."
  fi

  # Abandon des privilèges. setpriv (util-linux) est présent sur les images
  # Debian ; `su` sert de repli. On n'exécute JAMAIS l'application en root si
  # l'un des deux est disponible.
  if command -v setpriv >/dev/null 2>&1; then
    echo "[entrypoint] démarrage en sentinel (setpriv)"
    exec setpriv --reuid="${APP_UID}" --regid="${APP_GID}" --clear-groups "$@"
  fi
  if command -v su >/dev/null 2>&1; then
    echo "[entrypoint] démarrage en sentinel (su)"
    exec su -s /bin/sh sentinel -c 'exec "$0" "$@"' -- "$@"
  fi
  echo "[entrypoint] AVERTISSEMENT : aucun outil d'abandon de privilèges, démarrage en root"
  exec "$@"
fi

# Déjà non-root (exécution locale, docker compose…) : rien à réparer.
echo "[entrypoint] déjà non-root, démarrage direct"
exec "$@"
