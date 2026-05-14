"""Snapshot / restore / verify CLI for the hash-chained audit ledger.

Sprint DATA-2B.9. The audit ledger is the canonical record of every
delivered insight (DATA-2B.4) and lives in a SQLite WAL file. This CLI
gives ops three primitives without needing direct SQL access:

  ./scripts/audit_ledger_snapshot.py snapshot DB.sqlite OUT.jsonl
  ./scripts/audit_ledger_snapshot.py restore  IN.jsonl  NEW_DB.sqlite
  ./scripts/audit_ledger_snapshot.py verify   DB.sqlite

snapshot   verifies the chain end-to-end before exporting, then writes
           one entry per line as compact JSON. Aborts on broken chain.

restore    expects a fresh / non-existent target DB. Reads the JSONL
           preserving every (seq, ts, hash) field verbatim, re-inserts,
           then re-verifies — so a tampered archive fails loudly.

verify     run-anywhere chain audit; prints OK or the offending seq.

JSONL format
------------
Each line is the LedgerEntry dataclass serialised by ``to_dict()`` plus
a header line ``{"_meta": {"snapshot_v": 1, "n_entries": N,
"head_hash": "...", "created_utc": "..."}}`` so a reader can validate
the archive *before* it starts inserting.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.audit.hash_chain_ledger import HashChainLedger, LedgerEntry

logger = logging.getLogger("ledger-snapshot")

SNAPSHOT_VERSION = 1


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_snapshot(db_path: str, out_path: str) -> int:
    led = HashChainLedger(db_path=db_path)
    try:
        result = led.verify()
        if not result.ok:
            print(
                f"ABORT: source chain broken at seq {result.broken_at_seq}: "
                f"{result.reason}",
                file=sys.stderr,
            )
            return 2

        meta = {
            "_meta": {
                "snapshot_v": SNAPSHOT_VERSION,
                "n_entries": result.n_entries,
                "head_hash": led.head_hash,
                "created_utc": datetime.now(timezone.utc).isoformat() + "Z",
                "source_db": str(Path(db_path).resolve()),
            }
        }
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            n = 0
            for entry in led.iter_entries():
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
                n += 1
        print(f"snapshot OK: {n} entries → {out_path}")
        return 0
    finally:
        led.close()


def _iter_entries_from_jsonl(path: str) -> Iterable[LedgerEntry]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        if not first:
            raise ValueError("empty snapshot file")
        header = json.loads(first)
        if "_meta" not in header:
            raise ValueError("missing _meta header — not a snapshot file")
        if header["_meta"].get("snapshot_v") != SNAPSHOT_VERSION:
            raise ValueError(
                f"unsupported snapshot version "
                f"{header['_meta'].get('snapshot_v')}"
            )
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield LedgerEntry(
                seq=int(row["seq"]),
                inserted_at_utc=row["inserted_at_utc"],
                insight_id=row["insight_id"],
                canonical_json=row["canonical_json"],
                prev_hash=row["prev_hash"],
                entry_hash=row["entry_hash"],
            )


def cmd_restore(in_path: str, new_db_path: str) -> int:
    if Path(new_db_path).exists():
        print(
            f"ABORT: target {new_db_path} already exists. Restore "
            "refuses to merge into an existing DB.",
            file=sys.stderr,
        )
        return 2

    led = HashChainLedger(db_path=new_db_path)
    try:
        try:
            n = led.restore_from_entries(_iter_entries_from_jsonl(in_path))
        except ValueError as exc:
            print(f"ABORT: {exc}", file=sys.stderr)
            return 2
        print(f"restore OK: {n} entries → {new_db_path}")
        return 0
    finally:
        led.close()


def cmd_verify(db_path: str) -> int:
    led = HashChainLedger(db_path=db_path)
    try:
        result = led.verify()
        if result.ok:
            print(f"verify OK: {result.n_entries} entries, head={led.head_hash}")
            return 0
        print(
            f"verify FAIL at seq {result.broken_at_seq}: {result.reason}",
            file=sys.stderr,
        )
        return 2
    finally:
        led.close()


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audit_ledger_snapshot",
        description="Snapshot / restore / verify the hash-chained audit ledger.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="Export ledger to JSONL.")
    s.add_argument("db_path")
    s.add_argument("out_path")

    r = sub.add_parser("restore", help="Re-ingest a JSONL into a fresh DB.")
    r.add_argument("in_path")
    r.add_argument("new_db_path")

    v = sub.add_parser("verify", help="Walk the chain and report integrity.")
    v.add_argument("db_path")

    return p


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    if args.cmd == "snapshot":
        return cmd_snapshot(args.db_path, args.out_path)
    if args.cmd == "restore":
        return cmd_restore(args.in_path, args.new_db_path)
    if args.cmd == "verify":
        return cmd_verify(args.db_path)
    return 1


if __name__ == "__main__":
    sys.exit(main())
