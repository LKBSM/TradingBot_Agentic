"""Banc d'essai fournisseurs — orchestrateur.

Usage (depuis tools/data-benchmark/) :
    python runner.py                       # tous les fournisseurs ayant une cle
    python runner.py --providers twelve_data,oanda --days 30
    python runner.py --symbols EURUSD,XAUUSD --tfs M15,H4 --days 5
    python runner.py --list                # etat des cles / testabilite

Regles :
- Cle absente  -> fournisseur "no_key" (jamais simule).
- Cache data/raw/<provider>/<SYM>_<TF>.csv : relance = reprise (sauf --refresh).
- La reference (twelve_data) est toujours telechargee en premier si presente.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from symbols import ALL_SYMBOLS, SYM_BY_NAME, TIMEFRAMES  # noqa: E402
from providers import build_registry, ENV_ALIASES, REFERENCE  # noqa: E402

UTC = timezone.utc
RAW_DIR = HERE / "data" / "raw"
RESULTS_DIR = HERE / "results"


def load_env():
    """Charge tools/data-benchmark/.env puis le .env racine du repo (sans
    ecraser les variables deja definies). Parser minimal, pas de dependance."""
    for path in [HERE / ".env", HERE.parent.parent / ".env"]:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def get_key(cls):
    if cls.env_key is None:
        return "keyless"
    for name in [cls.env_key] + ENV_ALIASES.get(cls.env_key, []):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return None


def meta_path(provider_name):
    return RESULTS_DIR / f"fetch_meta_{provider_name}.json"


def load_meta(provider_name):
    p = meta_path(provider_name)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def save_meta(provider_name, meta):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path(provider_name).write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def run_provider(name, cls, key, syms, tfs, start, end, refresh):
    prov = cls(key)
    meta = load_meta(name)
    raw = RAW_DIR / name
    raw.mkdir(parents=True, exist_ok=True)
    done = 0
    for sym_name in syms:
        sym = SYM_BY_NAME[sym_name]
        for tf in tfs:
            cell = f"{sym_name}_{tf}"
            csv_path = raw / f"{cell}.csv"
            if not refresh and cell in meta and (
                    meta[cell]["status"] != "ok" or csv_path.is_file()):
                continue  # reprise : deja fait
            res = prov.fetch(sym, tf, start, end)
            meta[cell] = {
                "status": res.status, "error": res.error,
                "requests": res.requests_made, "errors_seen": res.errors_seen,
                "latency_ms": round(res.latency_ms, 1), "derived": res.derived,
                "fetched_at": res.fetched_at,
                "last_ts": res.df.index[-1].isoformat() if res.df is not None and len(res.df) else None,
                "rows": int(len(res.df)) if res.df is not None else 0,
            }
            if res.status == "ok":
                res.df.to_csv(csv_path, index_label="ts")
            done += 1
            if done % 20 == 0:
                save_meta(name, meta)
                print(f"  [{name}] {done} cellules traitees...", flush=True)
    save_meta(name, meta)
    counts = {}
    for v in meta.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    print(f"  [{name}] termine: {counts} | requetes HTTP: {prov.stats.requests}, "
          f"erreurs: {prov.stats.errors}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Banc d'essai qualite fournisseurs OHLC")
    ap.add_argument("--providers", default="", help="liste csv (defaut: tous avec cle)")
    ap.add_argument("--symbols", default="", help="liste csv (defaut: les 80)")
    ap.add_argument("--tfs", default="", help="liste csv parmi M5,M15,H1,H4,D1")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--refresh", action="store_true", help="ignore le cache")
    ap.add_argument("--list", action="store_true", help="etat des cles et sortie")
    args = ap.parse_args()

    load_env()
    registry = build_registry()

    keys = {name: get_key(cls) for name, cls in registry.items()}
    if args.list:
        for name in registry:
            print(f"  {name:16s} {'CLE OK' if keys[name] else 'no_key (non teste)'}")
        return

    wanted = [p.strip() for p in args.providers.split(",") if p.strip()] or list(registry)
    unknown = [p for p in wanted if p not in registry]
    if unknown:
        sys.exit(f"fournisseurs inconnus: {unknown} (dispo: {list(registry)})")
    # reference d'abord
    wanted.sort(key=lambda p: (p != REFERENCE, p))

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or \
        [s.name for s in ALL_SYMBOLS]
    bad = [s for s in syms if s not in SYM_BY_NAME]
    if bad:
        sys.exit(f"symboles inconnus: {bad}")
    tfs = [t.strip().upper() for t in args.tfs.split(",") if t.strip()] or TIMEFRAMES

    end = datetime.now(UTC).replace(microsecond=0)
    start = end - timedelta(days=args.days)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_meta = {"start": start.isoformat(), "end": end.isoformat(),
                "days": args.days, "symbols": len(syms), "tfs": tfs,
                "launched_at": datetime.now(UTC).isoformat()}
    (RESULTS_DIR / "run_meta.json").write_text(json.dumps(run_meta, indent=1),
                                               encoding="utf-8")

    for name in wanted:
        if not keys[name]:
            meta = load_meta(name)
            for s in syms:
                for tf in tfs:
                    meta.setdefault(f"{s}_{tf}", {"status": "no_key"})
            save_meta(name, meta)
            print(f"  [{name}] no_key -> marque non teste ({len(syms)*len(tfs)} cellules)")
            continue
        print(f"[{name}] demarrage ({len(syms)} symboles x {tfs})", flush=True)
        try:
            run_provider(name, registry[name], keys[name], syms, tfs, start, end,
                         args.refresh)
        except KeyboardInterrupt:
            print(f"[{name}] interrompu — le cache permet la reprise.")
            raise
        except Exception as exc:  # un fournisseur qui casse n'arrete pas le banc
            print(f"[{name}] ECHEC GLOBAL: {type(exc).__name__}: {exc}")

    print("Telechargements termines. Etape suivante: python metrics.py && "
          "python scoring.py && python report.py")


if __name__ == "__main__":
    main()
