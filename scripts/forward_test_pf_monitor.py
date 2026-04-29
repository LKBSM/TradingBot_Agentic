"""Rolling PF watchdog — emits a one-line alert if rolling-N PF drops.

Designed for cron / Monitor: prints exactly one line per run, machine-readable:
  STATE PF=1.23 N=45 WIN=0.51 SINCE=2026-04-01

States:
  OK     — PF ≥ pause_threshold (default 1.20)
  WATCH  — pause < PF < ok (default [1.0, 1.20])
  PAUSE  — PF < pause (default 1.0)  → triggers Telegram alert
  WAIT   — n_closed < min_sample (default 10) — no alert

Usage:
    python scripts/forward_test_pf_monitor.py
        [--db data/signals.db] [--days 30] [--ok 1.20] [--pause 1.0]
        [--min-sample 10] [--alert]

The --alert flag posts to Telegram on PAUSE (and only on PAUSE — silent
when OK / WATCH so cron output stays quiet).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.api.signal_store import SignalStore
from scripts.forward_test_daily import metrics, parse_iso, filter_window  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("SIGNAL_DB_PATH", "./data/signals.db"))
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--ok", type=float, default=1.20, help="OK threshold (PF ≥ this)")
    ap.add_argument("--pause", type=float, default=1.0, help="PAUSE threshold (PF < this)")
    ap.add_argument("--min-sample", type=int, default=10)
    ap.add_argument("--alert", action="store_true", help="Telegram alert on PAUSE")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"WAIT PF=nan N=0 SINCE=- (db missing)")
        return 0

    store = SignalStore(db_path=args.db)
    records, _ = store.get_history(page=1, page_size=1000)
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    sub = filter_window(records, cutoff)
    m = metrics(sub)

    if m["n_closed"] < args.min_sample:
        print(f"WAIT PF=nan N={m['n_closed']} WIN=nan SINCE={cutoff.date()}")
        return 0

    pf = m["PF"]
    win = m["win%"]
    state = "OK" if pf >= args.ok else "WATCH" if pf >= args.pause else "PAUSE"
    print(f"{state} PF={pf:.2f} N={m['n_closed']} WIN={win:.2f} SINCE={cutoff.date()}")

    if state == "PAUSE" and args.alert:
        from scripts.forward_test_daily import maybe_send_telegram
        msg = (
            f"⚠️ Smart Sentinel rolling-{args.days}d PF dropped below {args.pause}.\n"
            f"PF={pf:.2f} on {m['n_closed']} closed trades since {cutoff.date()}.\n"
            f"Recommended action: pause Telegram emission, investigate."
        )
        maybe_send_telegram(msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
