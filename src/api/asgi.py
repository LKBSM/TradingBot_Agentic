"""Production ASGI entry point for the MIA Markets API.

Run with::

    uvicorn src.api.asgi:app --host 0.0.0.0 --port 8000

Unlike ``src.intelligence.main`` (which also spins up the legacy Sentinel
scanner thread + MT5 data source), this entry point boots ONLY the FastAPI
application. The MarketReading engine, hybrid scheduler and the niveau-1.5
chatbot are wired by ``create_app``'s lifespan from the environment
(``BOOTSTRAP_ENABLED`` / ``SCHEDULER_ENABLED`` / ``CHATBOT_ENABLED`` /
``NEWS_PIPELINE_ENABLED``), so this is the lean way to serve the V2 product
(webapp + chatbot) that talks to Twelve Data + Anthropic.

``.env`` is loaded before the app is built so those env-gated flags and the
API keys are visible to the lifespan bootstrap.
"""

from __future__ import annotations

import os
import traceback

from dotenv import load_dotenv

# Load .env into the process environment BEFORE create_app() so the lifespan
# bootstrap (which reads os.environ) sees TWELVE_DATA_API_KEY, ANTHROPIC_API_KEY
# and the *_ENABLED flags. override=False keeps any var already exported in the
# shell authoritative over the file.
load_dotenv(override=False)

def _report_data_dir_state() -> None:
    """Dire, AVANT toute ouverture SQLite, dans quel état est le disque.

    Le service est resté 13 jours hors ligne sur un
    ``sqlite3.OperationalError: disk I/O error`` levé à la première écriture,
    sans qu'aucun chiffre ne permette de trancher entre « disque plein »,
    « disque en lecture seule » et « mauvaise propriété ». Les métriques de
    l'hébergeur ne servent à rien ici : le conteneur meurt avant d'en produire,
    et un script d'entrée peut être court-circuité par la commande de démarrage.

    Ce contrôle-ci vit DANS l'application : rien dans la chaîne de démarrage ne
    peut l'empêcher de s'exécuter. Il écrit sur stdout (et pas via ``logging``,
    non encore configuré à cet instant), et n'échoue JAMAIS — diagnostiquer ne
    doit pas devenir une nouvelle cause de panne.
    """
    import shutil
    import tempfile

    data_dir = os.environ.get("DATA_DIR", "./data")
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as exc:
        print(f"[disk] création de {data_dir} impossible : {exc}", flush=True)

    try:
        st = os.stat(data_dir)
        # getuid/getgid n'existent pas sous Windows : le diagnostic doit rester
        # inoffensif partout, y compris en développement local.
        uid = getattr(os, "getuid", lambda: -1)()
        gid = getattr(os, "getgid", lambda: -1)()
        print(
            f"[disk] {data_dir} — uid_processus={uid} gid={gid} "
            f"propriétaire={st.st_uid}:{st.st_gid} mode={oct(st.st_mode & 0o777)}",
            flush=True,
        )
    except OSError as exc:
        print(f"[disk] stat({data_dir}) impossible : {exc}", flush=True)

    try:
        usage = shutil.disk_usage(data_dir)
        gio = 1024 ** 3
        pct = 100.0 * usage.used / usage.total if usage.total else 0.0
        print(
            f"[disk] espace — total={usage.total / gio:.2f} Gio "
            f"utilisé={usage.used / gio:.2f} Gio "
            f"libre={usage.free / gio:.2f} Gio ({pct:.1f} % occupé)",
            flush=True,
        )
        if usage.free < 64 * 1024 * 1024:
            print("[disk] ALERTE : moins de 64 Mio libres — SQLite ne pourra pas écrire.", flush=True)
    except OSError as exc:
        print(f"[disk] mesure de l'espace impossible : {exc}", flush=True)

    try:
        with tempfile.NamedTemporaryFile(dir=data_dir, prefix=".write-probe-"):
            pass
        print(f"[disk] écriture dans {data_dir} : OK", flush=True)
    except OSError as exc:
        print(
            f"[disk] ÉCRITURE IMPOSSIBLE dans {data_dir} : {exc} "
            "— c'est la cause du plantage SQLite qui suit.",
            flush=True,
        )

    # Les fichiers les plus gros : si le disque est plein, ils disent QUI le remplit.
    try:
        tailles = []
        for nom in os.listdir(data_dir):
            chemin = os.path.join(data_dir, nom)
            if os.path.isfile(chemin):
                tailles.append((os.path.getsize(chemin), nom))
        tailles.sort(reverse=True)
        if tailles:
            apercu = ", ".join(f"{n}={t / (1024 ** 2):.0f} Mio" for t, n in tailles[:8])
            print(f"[disk] plus gros fichiers — {apercu}", flush=True)
        else:
            print(f"[disk] {data_dir} est vide", flush=True)
    except OSError as exc:
        print(f"[disk] listage de {data_dir} impossible : {exc}", flush=True)


def _disk_summary() -> str:
    """Résumé d'une ligne de l'état du disque, pour l'attacher à une erreur.

    Les lignes ``[disk]`` imprimées sur stdout se perdent dans l'interface de
    journaux de l'hébergeur (liste virtualisée, recherche capricieuse, fenêtre
    de temps). La DERNIÈRE ligne d'une trace, elle, est toujours lisible. On
    embarque donc les chiffres dans le message d'erreur lui-même : quand le
    démarrage échoue, la cause probable est écrite là où l'œil tombe.
    """
    import shutil

    data_dir = os.environ.get("DATA_DIR", "./data")
    morceaux = [f"dir={data_dir}"]
    try:
        u = shutil.disk_usage(data_dir)
        gio = 1024 ** 3
        morceaux.append(
            f"total={u.total / gio:.2f}Gio utilisé={u.used / gio:.2f}Gio "
            f"libre={u.free / gio:.3f}Gio ({100.0 * u.used / u.total:.1f}%)"
            if u.total
            else "espace=illisible"
        )
    except OSError as exc:
        morceaux.append(f"espace=erreur({exc})")
    try:
        st = os.stat(data_dir)
        uid = getattr(os, "getuid", lambda: -1)()
        morceaux.append(f"uid={uid} owner={st.st_uid}:{st.st_gid} mode={oct(st.st_mode & 0o777)}")
    except OSError as exc:
        morceaux.append(f"stat=erreur({exc})")
    try:
        sonde = os.path.join(data_dir, ".write-probe")
        with open(sonde, "w"):
            pass
        os.unlink(sonde)
        morceaux.append("écriture=OK")
    except OSError as exc:
        morceaux.append(f"écriture=IMPOSSIBLE({exc.__class__.__name__}: {exc})")
    return " | ".join(morceaux)


def _largest_files(limit: int = 12) -> dict:
    """Où passe la place du volume — fichiers ET sous-dossiers, récursivement.

    Première version de cette fonction : elle ne listait que les fichiers posés
    DIRECTEMENT dans le répertoire, en sautant les sous-dossiers. Résultat :
    571 Mio recensés alors que le volume en déclarait 9 790 — plus de 9 Go
    invisibles, et un faux coupable désigné (market_readings.db, 479 Mio).
    Une mesure qui ignore une partie du terrain ne vaut rien.

    Elle parcourt donc tout l'arbre : le poids cumulé de chaque entrée de
    premier niveau (dossier compris), les plus gros fichiers où qu'ils soient,
    et le total réellement recensé — que l'on peut comparer au `df` pour
    vérifier qu'il ne reste plus de zone d'ombre.

    Sans SQLite, sans sous-processus : cela doit fonctionner précisément quand
    plus rien d'autre ne fonctionne.
    """
    data_dir = os.environ.get("DATA_DIR", "./data")
    mio = 1024 * 1024
    par_entree: dict = {}
    fichiers: list = []
    total = 0
    erreurs = 0

    try:
        for racine, _dossiers, noms in os.walk(data_dir, onerror=lambda _e: None):
            for nom in noms:
                chemin = os.path.join(racine, nom)
                try:
                    if os.path.islink(chemin):
                        continue
                    taille = os.path.getsize(chemin)
                except OSError:
                    erreurs += 1
                    continue
                total += taille
                relatif = os.path.relpath(chemin, data_dir)
                sommet = relatif.split(os.sep)[0]
                par_entree[sommet] = par_entree.get(sommet, 0) + taille
                fichiers.append((taille, relatif))
    except OSError as exc:
        return {"erreur": f"{exc.__class__.__name__}: {exc}"}

    fichiers.sort(reverse=True)
    sommets = sorted(par_entree.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "total_recense_mio": round(total / mio, 1),
        "fichiers_illisibles": erreurs,
        "par_entree": [
            {"entree": nom, "mio": round(taille / mio, 1)} for nom, taille in sommets[:limit]
        ],
        "plus_gros_fichiers": [
            {"chemin": rel, "mio": round(taille / mio, 1)} for taille, rel in fichiers[:limit]
        ],
    }


_report_data_dir_state()

from src.api.app import create_app  # noqa: E402 — must follow load_dotenv

# Module-level ASGI app uvicorn can import directly. No subsystem injection:
# everything the V2 product needs is built by the lifespan from env.
def _degraded_app(exc: BaseException, diagnostic: str):
    """Une application minimale qui SERT le diagnostic au lieu de mourir.

    Pourquoi ce filet existe
    ------------------------
    Quand le démarrage échoue, le processus sortait en erreur, Render ne voyait
    aucun port s'ouvrir, relançait, et finissait par SUSPENDRE le service — ce
    qui a tenu le produit hors ligne 13 jours. Pire : sans port ouvert, plus
    aucun moyen d'interroger la machine, et les journaux de l'hébergeur se sont
    révélés illisibles (liste virtualisée, recherche inopérante). On se
    retrouvait aveugle exactement au moment où il fallait voir.

    Une application qui démarre en disant « je suis cassée, voici pourquoi »
    vaut mieux qu'une application qui meurt en silence. Elle ouvre un port
    (donc plus de boucle de plantage ni de suspension), et elle rend le
    diagnostic lisible d'un simple appel HTTP.

    Elle ne sert AUCUNE donnée de marché : toute autre route répond 503. Aucun
    risque de laisser croire que le produit fonctionne.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    detail = {
        "status": "degraded",
        "reason": "startup_failed",
        "error": f"{type(exc).__name__}: {exc}",
        "disk": diagnostic,
        # Qui occupe la place : indispensable pour choisir quoi purger.
        "largest_files": _largest_files(),
    }
    print(f"[boot] MODE DÉGRADÉ — {detail['error']} | {diagnostic}", flush=True)

    degraded = FastAPI(title="M.I.A Markets (mode dégradé)")

    @degraded.get("/health")
    def _health():  # noqa: ANN202
        # 200 volontaire : le contrôle de santé de l'hébergeur doit passer pour
        # que le service RESTE debout et reste interrogeable. La charge utile,
        # elle, ne ment pas — « degraded » y est en toutes lettres.
        return detail

    @degraded.api_route("/{_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def _everything_else(_path: str):  # noqa: ANN202
        return JSONResponse(status_code=503, content=detail)

    return degraded


try:
    app = create_app()
except Exception as _exc:  # noqa: BLE001 — on dégrade au lieu de mourir
    # La trace complète part quand même dans les journaux : on ne masque rien.
    traceback.print_exc()
    app = _degraded_app(_exc, _disk_summary())


__all__ = ["app"]
