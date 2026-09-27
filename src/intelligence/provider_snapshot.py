"""Raw provider-response snapshots — reproducibility for MarketReading audits.

Audit DETECTION_QUALITY_REVIEW_2026_06_12 §T3: the feed revises forming bars
between fetches, so a published reading could not be replayed from the final
candles stored in ``candles_cache`` (one reading's close_price existed in no
stored candle). Persisting the raw response per generation makes every reading
replayable bit-for-bit.

Format: one JSON line per generation, appended to a daily-rotated file
``<dir>/<instrument>_<timeframe>_<YYYYMMDD>.jsonl``::

    {"fetched_at": "...", "instrument": "XAUUSD", "timeframe": "M15",
     "candles": [{"ts": "...", "open": ..., "high": ..., "low": ...,
                  "close": ..., "volume": ...}, ...]}

⚠️ COÛT RÉEL — corrigé le 2026-09-27 après incident
Cet en-tête annonçait « ~20 KB per generation, daily files are trivial to
prune ». Les deux affirmations étaient fausses, et elles ont mis le produit
hors ligne 13 jours : chaque ligne réécrit l'INTÉGRALITÉ du tableau de bougies
(jusqu'à 70 132 barres pour XAUUSD M15), soit des fichiers de 90 à 134 Mio ;
et rien n'a jamais purgé quoi que ce soit. Mesuré en production : 9 452 Mio
sur un volume de 9 810 — 94 % du disque pour un outil de débogage, jusqu'au
``disk I/O error`` au démarrage.
Depuis : rétention CODÉE (``prune_snapshots``, appelée au démarrage) et
capture DÉSACTIVÉE en production (``PROVIDER_SNAPSHOT_ENABLED=0``).

Config (env):
  - ``PROVIDER_SNAPSHOT_ENABLED`` — truthy/falsy, default ON. À laisser à 0 en
    production : c'est un outil d'audit, pas une fonction produit.
  - ``PROVIDER_SNAPSHOT_DIR``     — default ``./data/provider_snapshots``.
  - ``PROVIDER_SNAPSHOT_RETENTION_DAYS`` — défaut 7. Au-delà, les fichiers
    quotidiens sont supprimés au démarrage.

Failure policy: best-effort. A snapshot write must never break reading
generation — errors are logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_ENABLED_ENV_VAR = "PROVIDER_SNAPSHOT_ENABLED"
_DIR_ENV_VAR = "PROVIDER_SNAPSHOT_DIR"
_DEFAULT_DIR = "./data/provider_snapshots"


def _enabled() -> bool:
    raw = os.environ.get(_ENABLED_ENV_VAR)
    if raw is None or raw == "":
        return True
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _candle_to_dict(candle: Any) -> dict[str, Any]:
    ts = getattr(candle, "ts", None)
    return {
        "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        "open": float(candle.open),
        "high": float(candle.high),
        "low": float(candle.low),
        "close": float(candle.close),
        "volume": float(getattr(candle, "volume", 0.0) or 0.0),
    }


def snapshot_provider_response(
    instrument: str,
    timeframe: str,
    candles: Sequence[Any],
    fetched_at: datetime,
) -> None:
    """Append the raw candle list to today's snapshot file (best-effort)."""
    if not _enabled() or not candles:
        return
    try:
        ts = (
            fetched_at.astimezone(timezone.utc)
            if fetched_at.tzinfo
            else fetched_at.replace(tzinfo=timezone.utc)
        )
        snapshot_dir = Path(os.environ.get(_DIR_ENV_VAR) or _DEFAULT_DIR)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / f"{instrument}_{timeframe}_{ts.strftime('%Y%m%d')}.jsonl"
        line = json.dumps(
            {
                "fetched_at": ts.isoformat(),
                "instrument": instrument,
                "timeframe": timeframe,
                "candles": [_candle_to_dict(c) for c in candles],
            },
            separators=(",", ":"),
        )
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:  # never break reading generation for observability
        logger.warning(
            "provider snapshot write failed for %s/%s: %s", instrument, timeframe, exc
        )


def prune_snapshots(retention_days: int | None = None, directory: str | None = None) -> dict:
    """Supprimer les instantanés plus vieux que la fenêtre de rétention.

    POURQUOI CETTE FONCTION EXISTE
    Ce module a rempli le disque de production et tenu le produit hors ligne
    13 jours. L'en-tête annonçait « ~20 KB per generation, daily files are
    trivial to prune » : les deux moitiés de la phrase étaient fausses. Chaque
    ligne réécrit l'INTÉGRALITÉ du tableau de bougies (jusqu'à 70 132 barres
    pour XAUUSD M15), d'où des fichiers de 90 à 134 Mio — et rien n'a jamais
    « pruned » quoi que ce soit. Mesuré le 2026-09-27 : 9 452 Mio sur un volume
    de 9 810, soit 94 % du disque occupés par un outil de débogage.

    La rétention est donc CODÉE, plus seulement souhaitée dans un commentaire.
    Réglable par ``PROVIDER_SNAPSHOT_RETENTION_DAYS`` (défaut 7 jours).

    Ne lève jamais : appelée au démarrage, elle ne doit en aucun cas empêcher
    le service de monter. Renvoie un compte-rendu destiné aux journaux.
    """
    if retention_days is None:
        try:
            retention_days = int(os.environ.get("PROVIDER_SNAPSHOT_RETENTION_DAYS", "7"))
        except ValueError:
            retention_days = 7
    retention_days = max(0, retention_days)

    snapshot_dir = Path(directory or os.environ.get(_DIR_ENV_VAR) or _DEFAULT_DIR)
    rapport = {"supprimes": 0, "mio_liberes": 0.0, "conserves": 0, "erreurs": 0}
    if not snapshot_dir.is_dir():
        return rapport

    limite = datetime.now(timezone.utc).timestamp() - retention_days * 86400
    octets = 0
    try:
        fichiers = list(snapshot_dir.glob("*.jsonl"))
    except OSError:
        return rapport

    for chemin in fichiers:
        try:
            st = chemin.stat()
            if st.st_mtime < limite:
                taille = st.st_size
                chemin.unlink()
                rapport["supprimes"] += 1
                octets += taille
            else:
                rapport["conserves"] += 1
        except OSError:
            rapport["erreurs"] += 1

    rapport["mio_liberes"] = round(octets / (1024 * 1024), 1)
    return rapport


__all__ = ["snapshot_provider_response", "prune_snapshots"]
