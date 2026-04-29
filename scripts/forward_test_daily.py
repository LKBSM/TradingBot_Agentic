"""Daily summary of forward-test signals and outcomes.

Reads the SignalStore SQLite, computes:
  * Counts: total / today / yesterday / 7d / 30d
  * Rolling PF on closed trades (7d, 30d)
  * Win rate (closed only), expectancy in pips, rr_ratio achieved
  * Filter stats (would need scanner-process introspection for live counters;
    here we report only signal_store-derived metrics)
  * Tier breakdown

Usage:
    python scripts/forward_test_daily.py [--db data/signals.db]
        [--symbol XAUUSD] [--telegram]

Run via cron at 23:55 UTC for a "yesterday closed" summary.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.api.signal_store import SignalRecord, SignalStore


def parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        # Fallback: assume UTC, drop microseconds
        return datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def metrics(rs: List[SignalRecord]) -> Dict[str, float]:
    closed = [r for r in rs if r.outcome in ("WIN", "LOSS")]
    if not closed:
        return {"n": len(rs), "n_closed": 0, "open": len(rs), "win%": float("nan"),
                "PF": float("nan"), "exp_pips": 0.0, "tot_pips": 0.0}
    wins = [r for r in closed if r.outcome == "WIN"]
    losses = [r for r in closed if r.outcome == "LOSS"]
    gross_win = sum((r.pnl_pips or 0) for r in wins)
    gross_loss = -sum((r.pnl_pips or 0) for r in losses)
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    return {
        "n": len(rs), "n_closed": len(closed), "open": len(rs) - len(closed),
        "win%": len(wins) / len(closed),
        "PF": pf if pf != float("inf") else 999.0,
        "exp_pips": sum((r.pnl_pips or 0) for r in closed) / len(closed),
        "tot_pips": sum((r.pnl_pips or 0) for r in closed),
    }


def filter_window(rs: List[SignalRecord], since: datetime) -> List[SignalRecord]:
    return [r for r in rs if parse_iso(r.created_at) >= since]


def fmt_metrics(m: Dict[str, float], label: str) -> str:
    if m["n_closed"] == 0:
        return f"  {label:>10s}: n={m['n']} (all open)"
    return (
        f"  {label:>10s}: n={m['n']} closed={m['n_closed']} "
        f"PF={m['PF']:.2f} win={m['win%']:.0%} "
        f"exp={m['exp_pips']:+.1f}p tot={m['tot_pips']:+.1f}p"
    )


def render(records: List[SignalRecord], symbol: str = None) -> str:
    if symbol:
        records = [r for r in records if r.symbol == symbol]
    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yest = today - timedelta(days=1)

    lines = []
    lines.append("=" * 60)
    lines.append(f" Smart Sentinel — Forward Test Daily Summary")
    lines.append(f" Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    if symbol:
        lines.append(f" Symbol: {symbol}")
    lines.append("=" * 60)

    if not records:
        lines.append(" No signals recorded yet.")
        return "\n".join(lines)

    # Window slices
    windows = [
        ("today",     filter_window(records, today)),
        ("yesterday", [r for r in records if today > parse_iso(r.created_at) >= yest]),
        ("7d",        filter_window(records, today - timedelta(days=7))),
        ("30d",       filter_window(records, today - timedelta(days=30))),
        ("all",       records),
    ]
    lines.append("\n COUNTS & PERFORMANCE")
    lines.append("-" * 60)
    for label, sub in windows:
        lines.append(fmt_metrics(metrics(sub), label))

    # Tier breakdown (all-time)
    by_tier: Dict[str, List[SignalRecord]] = defaultdict(list)
    for r in records:
        # Tier is not stored as a column — derive from confluence_score using
        # the recalibrated cutpoints. Aligns with confluence_detector.py.
        s = r.confluence_score or 0
        tier = "PREMIUM" if s >= 55 else "STANDARD" if s >= 40 else "WEAK" if s >= 25 else "INVALID"
        by_tier[tier].append(r)

    lines.append("\n PER-TIER (all-time)")
    lines.append("-" * 60)
    for tier in ("PREMIUM", "STANDARD", "WEAK"):
        sub = by_tier.get(tier, [])
        if not sub:
            continue
        m = metrics(sub)
        lines.append(fmt_metrics(m, tier))

    # KPI gates (from MEMORY)
    lines.append("\n KPI GATES (from forward-test plan)")
    lines.append("-" * 60)
    m30 = metrics(filter_window(records, today - timedelta(days=30)))
    if m30["n_closed"] >= 5:
        pf_30d = m30["PF"]
        gate = "🟢 OK" if pf_30d >= 1.20 else "🟡 WATCH" if pf_30d >= 1.0 else "🔴 PAUSE"
        lines.append(f"  Rolling 30d PF: {pf_30d:.2f}  → {gate}  (target ≥ 1.20)")
    else:
        lines.append(f"  Rolling 30d PF: insufficient sample ({m30['n_closed']} closed)")

    sigs_30d = m30["n"]
    daily_rate = sigs_30d / 30
    target_daily = 0.27   # ~100/year
    gate = "🟢 OK" if daily_rate >= target_daily else "🟡 LOW"
    lines.append(f"  Signals/day (30d): {daily_rate:.2f}  → {gate}  (target ≥ {target_daily:.2f})")

    return "\n".join(lines)


def maybe_send_telegram(report: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[telegram] skipped (TELEGRAM_BOT_TOKEN/CHAT_ID not set)")
        return
    import urllib.request
    import urllib.parse
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat,
        "text": "```\n" + report + "\n```",
        "parse_mode": "MarkdownV2",
    }).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
        print("[telegram] sent")
    except Exception as e:
        print(f"[telegram] failed: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("SIGNAL_DB_PATH", "./data/signals.db"))
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--telegram", action="store_true", help="POST report to Telegram")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"DB not found: {args.db}")
        return 2

    store = SignalStore(db_path=args.db)
    # Pull a generous window — 1000 signals covers ~12 yrs at our rate
    records, _ = store.get_history(page=1, page_size=1000)
    report = render(records, symbol=args.symbol)
    print(report)

    if args.telegram:
        maybe_send_telegram(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
