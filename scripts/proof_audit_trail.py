"""Proof dashboard generator — audit trail (hash-chain ledger).

Run::

    python scripts/proof_audit_trail.py

Outputs::

    reports/proof/audit_trail.json     # machine-readable
    reports/proof/audit_trail.html     # client-facing dashboard

What this proves
----------------
Client question: "How do I know you haven't rewritten old signals to
make your indicator look better than it actually was?"

Answer: every delivered InsightSignalV2 is appended to a SHA-256
hash-chained ledger (src/audit/hash_chain_ledger.py). Each entry's hash
incorporates the previous entry's hash, so any mutation cascades and
is detectable in O(N) at audit time. The client (or a regulator) can
re-run verify() at any point and either get ``ok=True`` or the exact
seq where the chain was broken.

This script:
  1. Builds a fresh demo ledger with 250 realistic synthetic insights
     spanning the last 90 days (XAU M15, mixed bull / bear / hold).
  2. Runs verify() on the clean chain -> must pass.
  3. Forks a tampered copy of the DB, mutates a body in-place, then
     runs verify() on it -> must catch it and report the seq.
  4. Emits a JSON summary + a self-contained HTML viewer with
       - chain head hash + entry count + first/last timestamp
       - paginated entries table (newest 50 by default)
       - tamper-demo block: which seq was mutated, what verify() returned
       - chain visualization (entry_hash -> prev_hash links, sample of 20)

Honesty policy
--------------
This is a structural integrity proof. It does NOT certify the *truth*
of the insight bodies — only that whatever was written at delivery time
is what remains in the archive today. Edge claims live in
detection_accuracy.html and calibration.html.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import sqlite3
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.audit.hash_chain_ledger import (  # noqa: E402
    HashChainLedger,
    LedgerEntry,
    canonical_json,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("proof.audit_trail")

REPORTS_DIR = REPO / "reports" / "proof"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DB_PATH = REPORTS_DIR / "_audit_demo.sqlite"
TAMPERED_DB_PATH = REPORTS_DIR / "_audit_demo_tampered.sqlite"

RNG_SEED = 42
N_INSIGHTS = 250
DAYS_BACK = 90


# ---------------------------------------------------------------------------
# Synthetic insight generator — kept narrow so we don't depend on the full
# InsightAssembler pipeline. Shapes match InsightSignalV2 v2.1.0 enough that
# downstream code (export, paginate, verify) sees something realistic.
# ---------------------------------------------------------------------------


def _gen_insight(seq: int, ts: datetime, rng: random.Random) -> Dict[str, Any]:
    direction = rng.choices(
        ["BULLISH", "BEARISH", "NEUTRAL"], weights=[0.42, 0.40, 0.18]
    )[0]
    base_price = 2050.0 + rng.gauss(0, 35.0)
    conviction = max(0.05, min(0.95, rng.gauss(0.52, 0.18)))
    return {
        "id": f"insight_{seq:06d}_{ts.strftime('%Y%m%dT%H%M%S')}",
        "schema_version": "2.1.0",
        "instrument": "XAUUSD",
        "timeframe": "M15",
        "emitted_at": ts.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "direction": direction,
        "conviction": round(conviction, 4),
        "conviction_band": (
            "high" if conviction >= 0.70 else ("medium" if conviction >= 0.50 else "low")
        ),
        "price_ref": round(base_price, 2),
        "smc": {
            "bos_event": rng.random() < 0.18,
            "bos_armed": rng.random() < 0.08,
            "fvg_active": rng.random() < 0.22,
            "ob_active": rng.random() < 0.14,
            "choch": rng.random() < 0.05,
        },
        "volatility": {
            "atr_pct": round(0.18 + rng.expovariate(8.0), 4),
            "regime": rng.choice(["calm", "normal", "elevated"]),
        },
        "narrative_short": (
            "Structural break confirmed; momentum aligned with daily bias."
            if direction == "BULLISH"
            else (
                "Distribution detected at premium; sellers in control."
                if direction == "BEARISH"
                else "Range conditions; awaiting confirmation."
            )
        ),
        "narrative_long": None,
        "edge_claim": False,
        "tier_visible_for": ["FREE", "ANALYST", "STRATEGIST", "INSTITUTIONAL"],
    }


def build_demo_ledger() -> Dict[str, Any]:
    """Build a fresh demo ledger and return its statistics."""
    if DEMO_DB_PATH.exists():
        DEMO_DB_PATH.unlink()
    if TAMPERED_DB_PATH.exists():
        TAMPERED_DB_PATH.unlink()

    rng = random.Random(RNG_SEED)
    now = datetime(2026, 5, 23, 12, 0, 0)  # frozen for reproducibility
    start = now - timedelta(days=DAYS_BACK)
    span_seconds = (now - start).total_seconds()

    # Generate timestamps in ascending order so seq matches chronology.
    timestamps = sorted(
        [start + timedelta(seconds=rng.uniform(0, span_seconds)) for _ in range(N_INSIGHTS)]
    )

    ledger = HashChainLedger(DEMO_DB_PATH)
    t0 = time.perf_counter()
    written: List[Dict[str, Any]] = []
    for i, ts in enumerate(timestamps, start=1):
        body = _gen_insight(i, ts, rng)
        ledger.append(body)
        written.append(body)
    build_ms = (time.perf_counter() - t0) * 1000.0

    # Verify the clean chain.
    t1 = time.perf_counter()
    clean_result = ledger.verify()
    verify_ms = (time.perf_counter() - t1) * 1000.0

    head_hash = ledger.head_hash
    size = ledger.size

    # Sample entries (newest first) for the table.
    page, _ = ledger.paginate(limit=50)
    page_dicts = [e.to_dict() for e in page]

    # Sample for visualization (oldest 5 + middle 10 + newest 5).
    all_entries = list(ledger.iter_entries())
    sample_idx = list(range(min(5, size))) + \
                 list(range(max(0, size // 2 - 5), min(size, size // 2 + 5))) + \
                 list(range(max(0, size - 5), size))
    sample_idx = sorted(set(sample_idx))
    chain_sample = [all_entries[i].to_dict() for i in sample_idx]

    # Frequency of insights per day (for histogram). We aggregate by
    # the body's ``emitted_at`` rather than ``inserted_at_utc`` — in the
    # demo, all 250 appends happen in the same wall-clock second, but the
    # synthetic insights span the full 90-day window. Prod systems would
    # see the two timestamps coincide because deliveries are real-time.
    per_day: Dict[str, int] = {}
    for e in all_entries:
        try:
            body = json.loads(e.canonical_json)
            day = body.get("emitted_at", e.inserted_at_utc)[:10]
        except Exception:
            day = e.inserted_at_utc[:10]
        per_day[day] = per_day.get(day, 0) + 1
    daily_series = sorted(per_day.items())

    first_entry = all_entries[0].to_dict() if all_entries else None
    last_entry = all_entries[-1].to_dict() if all_entries else None

    ledger.close()

    return {
        "db_path": str(DEMO_DB_PATH),
        "build_ms": round(build_ms, 1),
        "verify_ms": round(verify_ms, 1),
        "size": size,
        "head_hash": head_hash,
        "first_entry": first_entry,
        "last_entry": last_entry,
        "clean_verification": {
            "ok": clean_result.ok,
            "n_entries": clean_result.n_entries,
            "broken_at_seq": clean_result.broken_at_seq,
            "reason": clean_result.reason,
        },
        "page_sample": page_dicts,
        "chain_sample": chain_sample,
        "daily_series": daily_series,
    }


def tamper_and_verify() -> Dict[str, Any]:
    """Fork the demo DB, mutate one body in-place, verify -> must fail."""
    shutil.copy2(DEMO_DB_PATH, TAMPERED_DB_PATH)

    # Mutate a specific row outside the ledger API. We pick seq=137 — a
    # mid-chain row — and twiddle the body inside canonical_json. The
    # entry_hash stored on disk stays untouched, so verify() will detect
    # the body mismatch when it recomputes the hash.
    target_seq = 137
    conn = sqlite3.connect(str(TAMPERED_DB_PATH), isolation_level=None)
    row = conn.execute(
        "SELECT canonical_json FROM ledger WHERE seq = ?", (target_seq,)
    ).fetchone()
    if not row:
        conn.close()
        return {"available": False, "reason": "target seq not in ledger"}

    original_body = row[0]
    mutated = json.loads(original_body)
    # Flip direction BULLISH<->BEARISH and bump conviction by 0.20 — the kind
    # of "make the call look better in hindsight" rewrite a bad actor would
    # attempt. ensure_ascii=False / separators to keep canonical shape.
    mutated["direction"] = "BULLISH" if mutated.get("direction") != "BULLISH" else "BEARISH"
    mutated["conviction"] = round(min(0.99, float(mutated.get("conviction", 0.5)) + 0.20), 4)
    mutated_canonical = canonical_json(mutated)

    conn.execute(
        "UPDATE ledger SET canonical_json = ? WHERE seq = ?",
        (mutated_canonical, target_seq),
    )
    conn.close()

    # Now verify via the public API.
    tampered_ledger = HashChainLedger(TAMPERED_DB_PATH)
    t0 = time.perf_counter()
    result = tampered_ledger.verify()
    verify_ms = (time.perf_counter() - t0) * 1000.0

    # Pull the surrounding rows (seq-2 .. seq+2) for the dashboard.
    neighbours = []
    for s in range(max(1, target_seq - 2), target_seq + 3):
        e = tampered_ledger.get(s)
        if e is not None:
            d = e.to_dict()
            d["is_tampered"] = (s == target_seq)
            neighbours.append(d)

    tampered_ledger.close()

    return {
        "available": True,
        "target_seq": target_seq,
        "original_body": original_body,
        "mutated_body": mutated_canonical,
        "verification": {
            "ok": result.ok,
            "n_entries": result.n_entries,
            "broken_at_seq": result.broken_at_seq,
            "reason": result.reason,
        },
        "verify_ms": round(verify_ms, 1),
        "neighbours": neighbours,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>Audit Trail — Smart Sentinel AI</title>
<script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js\"></script>
<style>
:root {
  --bg: #0f1419;
  --panel: #1a1f2e;
  --border: #2a3142;
  --text: #e6e9ef;
  --muted: #8b94a8;
  --green: #4ade80;
  --red: #f87171;
  --amber: #fbbf24;
  --blue: #60a5fa;
  --accent: #a78bfa;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  margin: 0; padding: 24px; line-height: 1.5;
}
h1 { font-size: 24px; margin: 0 0 4px 0; }
h2 { font-size: 18px; margin: 24px 0 12px 0; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
.subtitle { color: var(--muted); font-size: 14px; margin-bottom: 24px; }
.panel {
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  padding: 20px; margin-bottom: 20px;
}
.banner {
  background: linear-gradient(90deg, rgba(167,139,250,0.10), rgba(96,165,250,0.05));
  border-left: 3px solid var(--accent);
}
.grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
.kpi { text-align: left; }
.kpi-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
.kpi-value { font-size: 22px; font-weight: 600; margin-top: 4px; word-break: break-all; }
.kpi-sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
.badge {
  display: inline-block; padding: 3px 10px; border-radius: 4px;
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
}
.b-green { background: rgba(74,222,128,0.15); color: var(--green); border: 1px solid rgba(74,222,128,0.30); }
.b-red   { background: rgba(248,113,113,0.15); color: var(--red); border: 1px solid rgba(248,113,113,0.30); }
.b-amber { background: rgba(251,191,36,0.15); color: var(--amber); border: 1px solid rgba(251,191,36,0.30); }
.mono { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }
.hash { color: var(--blue); word-break: break-all; }
.hash-short { color: var(--blue); }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; font-size: 11px; }
tr.tampered { background: rgba(248,113,113,0.08); }
.chain-link {
  display: flex; align-items: center; gap: 8px; padding: 10px 12px;
  background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 6px;
  margin-bottom: 6px;
}
.chain-link .arrow { color: var(--muted); }
.chain-link .seq { color: var(--accent); font-weight: 600; width: 50px; }
.body-diff {
  background: #0a0d12; border: 1px solid var(--border); border-radius: 4px;
  padding: 10px; font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 11px;
  white-space: pre-wrap; word-break: break-all; max-height: 180px; overflow-y: auto;
}
.body-original { border-left: 3px solid var(--green); }
.body-mutated  { border-left: 3px solid var(--red); }
.footer { color: var(--muted); font-size: 11px; margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border); }
.section-intro { color: var(--muted); font-size: 13px; margin-bottom: 14px; max-width: 850px; }
canvas { max-height: 220px; }
</style>
</head>
<body>

<h1>Audit Trail — Hash-Chained Insight Ledger</h1>
<div class=\"subtitle\">
  Tamper-evident proof that every delivered insight remains exactly as it was
  emitted — Smart Sentinel AI · XAUUSD M15 · generated {GEN_TS}
</div>

<div class=\"panel banner\">
  <strong>What this dashboard proves.</strong> Every <code>InsightSignalV2</code>
  we deliver is appended to a SHA-256 hash-chained SQLite ledger. Each entry's
  hash includes the previous entry's hash, so any post-hoc mutation cascades
  forward and is caught in O(N) at audit time. The client can re-run
  <code>verify()</code> at any point — either the chain is intact end-to-end,
  or it returns the exact seq where the tamper occurred. <em>Note: this is a
  structural integrity proof — it does not certify the truth of insight bodies,
  only that the archive matches what was delivered.</em>
</div>

<!-- ============================================================ -->
<h2>Chain Status — Clean Demo Ledger</h2>
<div class=\"panel\">
  <div class=\"section-intro\">
    A fresh demo ledger built with {N_INSIGHTS} synthetic InsightSignalV2 entries
    spanning the last {DAYS_BACK} days. Statistics below are read directly from
    SQLite after the build.
  </div>
  <div class=\"grid\">
    <div class=\"kpi\">
      <div class=\"kpi-label\">Verification</div>
      <div class=\"kpi-value\"><span class=\"badge {CLEAN_BADGE_CLASS}\">{CLEAN_BADGE_TEXT}</span></div>
      <div class=\"kpi-sub\">{CLEAN_VERIFY_MS} ms · {SIZE} entries</div>
    </div>
    <div class=\"kpi\">
      <div class=\"kpi-label\">Chain head</div>
      <div class=\"kpi-value mono hash-short\" title=\"{HEAD_HASH}\">{HEAD_HASH_SHORT}…</div>
      <div class=\"kpi-sub\">SHA-256 · last entry</div>
    </div>
    <div class=\"kpi\">
      <div class=\"kpi-label\">First entry</div>
      <div class=\"kpi-value mono\">{FIRST_TS}</div>
      <div class=\"kpi-sub\">seq=1</div>
    </div>
    <div class=\"kpi\">
      <div class=\"kpi-label\">Last entry</div>
      <div class=\"kpi-value mono\">{LAST_TS}</div>
      <div class=\"kpi-sub\">seq={SIZE}</div>
    </div>
  </div>
</div>

<!-- ============================================================ -->
<h2>Daily Delivery Volume</h2>
<div class=\"panel\">
  <div class=\"section-intro\">
    Histogram of insights appended per UTC day in the demo window. A sudden
    gap or spike would itself be a forensic signal — but the chain primitive
    only guarantees that the entries you see here have not been altered.
  </div>
  <canvas id=\"chartDaily\"></canvas>
</div>

<!-- ============================================================ -->
<h2>Tamper-Detection Demo</h2>
<div class=\"panel\">
  <div class=\"section-intro\">
    We forked the clean DB, mutated the body of <strong>seq={TAMPER_SEQ}</strong>
    in-place (flipped direction + bumped conviction — the kind of "make the
    call look better in hindsight" rewrite a bad actor would attempt), then
    re-ran <code>verify()</code>. The chain caught the mutation at the
    expected seq.
  </div>

  <div class=\"grid\" style=\"margin-bottom: 16px;\">
    <div class=\"kpi\">
      <div class=\"kpi-label\">Result</div>
      <div class=\"kpi-value\"><span class=\"badge {TAMPER_BADGE_CLASS}\">{TAMPER_BADGE_TEXT}</span></div>
      <div class=\"kpi-sub\">{TAMPER_VERIFY_MS} ms</div>
    </div>
    <div class=\"kpi\">
      <div class=\"kpi-label\">Broken at seq</div>
      <div class=\"kpi-value\">{TAMPER_BROKEN_SEQ}</div>
      <div class=\"kpi-sub\">expected: {TAMPER_SEQ}</div>
    </div>
    <div class=\"kpi\" style=\"grid-column: span 2;\">
      <div class=\"kpi-label\">Reason</div>
      <div class=\"kpi-value\" style=\"font-size: 14px;\">{TAMPER_REASON}</div>
    </div>
  </div>

  <h3 style=\"font-size: 13px; color: var(--muted); margin-top: 18px;\">ORIGINAL BODY (seq={TAMPER_SEQ})</h3>
  <div class=\"body-diff body-original\">{ORIGINAL_BODY}</div>

  <h3 style=\"font-size: 13px; color: var(--muted); margin-top: 14px;\">MUTATED BODY — what an attacker wrote</h3>
  <div class=\"body-diff body-mutated\">{MUTATED_BODY}</div>

  <h3 style=\"font-size: 13px; color: var(--muted); margin-top: 18px;\">SURROUNDING CHAIN (seq={TAMPER_SEQ}±2)</h3>
  <table>
    <thead><tr><th>seq</th><th>inserted_at_utc</th><th>insight_id</th><th>entry_hash</th></tr></thead>
    <tbody>
      {NEIGHBOURS_ROWS}
    </tbody>
  </table>
</div>

<!-- ============================================================ -->
<h2>Chain Visualization — entry_hash → prev_hash</h2>
<div class=\"panel\">
  <div class=\"section-intro\">
    Sample of 20 entries (oldest 5 + middle 10 + newest 5). Each row shows
    how the entry's hash depends on the previous one. If any body in this
    sequence was altered, the chain would unzip starting at that seq.
  </div>
  {CHAIN_LINKS}
</div>

<!-- ============================================================ -->
<h2>Latest 50 Entries</h2>
<div class=\"panel\">
  <div class=\"section-intro\">
    Paginated read from the live ledger (newest-first). Each row is auditable
    end-to-end by recomputing <code>SHA-256(seq | ts | body | prev_hash)</code>.
  </div>
  <div style=\"overflow-x: auto;\">
    <table>
      <thead>
        <tr>
          <th>seq</th>
          <th>inserted_at_utc</th>
          <th>insight_id</th>
          <th>entry_hash</th>
          <th>prev_hash</th>
        </tr>
      </thead>
      <tbody>
        {PAGE_ROWS}
      </tbody>
    </table>
  </div>
</div>

<div class=\"footer\">
  Generated by <code>scripts/proof_audit_trail.py</code> ·
  Ledger source: <code>src/audit/hash_chain_ledger.py</code> ·
  Raw JSON: <code>reports/proof/audit_trail.json</code> ·
  Smart Sentinel AI · {GEN_TS}
</div>

<script>
const daily = {DAILY_JSON};
const labels = daily.map(d => d[0]);
const counts = daily.map(d => d[1]);
new Chart(document.getElementById('chartDaily'), {
  type: 'bar',
  data: { labels, datasets: [{
    label: 'Insights per day',
    data: counts,
    backgroundColor: 'rgba(96, 165, 250, 0.6)',
    borderColor: '#60a5fa',
    borderWidth: 1,
  }]},
  options: {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#8b94a8', maxTicksLimit: 12 }, grid: { color: '#2a3142' } },
      y: { ticks: { color: '#8b94a8' }, grid: { color: '#2a3142' }, beginAtZero: true }
    }
  }
});
</script>

</body>
</html>
"""


def _short(h: str) -> str:
    return (h[:16] if h else "") + ("…" if h and len(h) > 16 else "")


def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _pretty_body(canonical: str) -> str:
    try:
        return _esc(json.dumps(json.loads(canonical), indent=2, ensure_ascii=False))
    except Exception:
        return _esc(canonical)


def render_html(clean: Dict[str, Any], tamper: Dict[str, Any]) -> str:
    out = HTML_TEMPLATE

    out = out.replace("{GEN_TS}", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    out = out.replace("{N_INSIGHTS}", str(N_INSIGHTS))
    out = out.replace("{DAYS_BACK}", str(DAYS_BACK))

    out = out.replace("{SIZE}", str(clean["size"]))
    out = out.replace("{HEAD_HASH}", clean["head_hash"])
    out = out.replace("{HEAD_HASH_SHORT}", clean["head_hash"][:16])
    out = out.replace("{FIRST_TS}", clean["first_entry"]["inserted_at_utc"][:19] + "Z")
    out = out.replace("{LAST_TS}", clean["last_entry"]["inserted_at_utc"][:19] + "Z")
    out = out.replace("{CLEAN_VERIFY_MS}", str(clean["verify_ms"]))

    clean_ok = clean["clean_verification"]["ok"]
    out = out.replace("{CLEAN_BADGE_CLASS}", "b-green" if clean_ok else "b-red")
    out = out.replace("{CLEAN_BADGE_TEXT}", "OK · INTACT" if clean_ok else "FAILED")

    # Tamper demo
    tv = tamper["verification"]
    tampered_ok = tv["ok"]
    # We WANT verify() to fail here — that's the proof.
    out = out.replace("{TAMPER_BADGE_CLASS}", "b-red" if not tampered_ok else "b-amber")
    out = out.replace(
        "{TAMPER_BADGE_TEXT}",
        "TAMPER DETECTED" if not tampered_ok else "UNEXPECTED OK",
    )
    out = out.replace("{TAMPER_SEQ}", str(tamper["target_seq"]))
    out = out.replace(
        "{TAMPER_BROKEN_SEQ}", str(tv["broken_at_seq"] if tv["broken_at_seq"] is not None else "—")
    )
    out = out.replace("{TAMPER_REASON}", _esc(tv["reason"] or "—"))
    out = out.replace("{TAMPER_VERIFY_MS}", str(tamper["verify_ms"]))
    out = out.replace("{ORIGINAL_BODY}", _pretty_body(tamper["original_body"]))
    out = out.replace("{MUTATED_BODY}", _pretty_body(tamper["mutated_body"]))

    # Neighbours table
    n_rows = []
    for nb in tamper["neighbours"]:
        cls = ' class=\"tampered\"' if nb["is_tampered"] else ""
        n_rows.append(
            f"<tr{cls}>"
            f"<td class=mono>{nb['seq']}</td>"
            f"<td class=mono>{_esc(nb['inserted_at_utc'][:19])}Z</td>"
            f"<td class=mono>{_esc(nb['insight_id'])}</td>"
            f"<td class='mono hash'>{_esc(_short(nb['entry_hash']))}</td>"
            f"</tr>"
        )
    out = out.replace("{NEIGHBOURS_ROWS}", "\n      ".join(n_rows))

    # Page rows
    p_rows = []
    for e in clean["page_sample"]:
        p_rows.append(
            f"<tr>"
            f"<td class=mono>{e['seq']}</td>"
            f"<td class=mono>{_esc(e['inserted_at_utc'][:19])}Z</td>"
            f"<td class=mono>{_esc(e['insight_id'])}</td>"
            f"<td class='mono hash' title='{_esc(e['entry_hash'])}'>{_esc(_short(e['entry_hash']))}</td>"
            f"<td class='mono hash' title='{_esc(e['prev_hash'])}'>{_esc(_short(e['prev_hash']))}</td>"
            f"</tr>"
        )
    out = out.replace("{PAGE_ROWS}", "\n        ".join(p_rows))

    # Chain visualization
    links = []
    for e in clean["chain_sample"]:
        links.append(
            f"<div class='chain-link'>"
            f"<span class='seq'>#{e['seq']}</span>"
            f"<span class='mono hash' title='{_esc(e['prev_hash'])}'>{_esc(_short(e['prev_hash']))}</span>"
            f"<span class='arrow'>→</span>"
            f"<span class='mono hash' title='{_esc(e['entry_hash'])}'>{_esc(_short(e['entry_hash']))}</span>"
            f"<span style='margin-left:auto;color:var(--muted);font-size:11px;'>{_esc(e['inserted_at_utc'][:19])}Z</span>"
            f"</div>"
        )
    out = out.replace("{CHAIN_LINKS}", "\n  ".join(links))

    out = out.replace("{DAILY_JSON}", json.dumps(clean["daily_series"]))

    return out


def main() -> None:
    logger.warning("Building demo ledger (n=%d, span=%d days)...", N_INSIGHTS, DAYS_BACK)
    clean = build_demo_ledger()
    logger.warning(
        "Built %d entries in %.1f ms, verified in %.1f ms (ok=%s)",
        clean["size"], clean["build_ms"], clean["verify_ms"], clean["clean_verification"]["ok"],
    )

    logger.warning("Running tamper demo...")
    tamper = tamper_and_verify()
    logger.warning(
        "Tampered seq=%s -> verify ok=%s, broken_at_seq=%s",
        tamper["target_seq"], tamper["verification"]["ok"], tamper["verification"]["broken_at_seq"],
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_insights": N_INSIGHTS,
        "days_back": DAYS_BACK,
        "clean": {
            "size": clean["size"],
            "build_ms": clean["build_ms"],
            "verify_ms": clean["verify_ms"],
            "head_hash": clean["head_hash"],
            "first_entry_ts": clean["first_entry"]["inserted_at_utc"],
            "last_entry_ts": clean["last_entry"]["inserted_at_utc"],
            "verification": clean["clean_verification"],
        },
        "tamper": {
            "target_seq": tamper["target_seq"],
            "verification": tamper["verification"],
            "verify_ms": tamper["verify_ms"],
        },
    }

    (REPORTS_DIR / "audit_trail.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    html = render_html(clean, tamper)
    (REPORTS_DIR / "audit_trail.html").write_text(html, encoding="utf-8")

    logger.warning("Wrote %s", REPORTS_DIR / "audit_trail.json")
    logger.warning("Wrote %s", REPORTS_DIR / "audit_trail.html")

    # Clean up demo DBs (they're regeneratable).
    try:
        DEMO_DB_PATH.unlink(missing_ok=True)
        TAMPERED_DB_PATH.unlink(missing_ok=True)
        # Also wal/shm files SQLite leaves around.
        for suffix in ("-wal", "-shm"):
            for p in (DEMO_DB_PATH, TAMPERED_DB_PATH):
                Path(str(p) + suffix).unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
