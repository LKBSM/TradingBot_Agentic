"""Generate the 4 v2 mockups from a canonical InsightSignalV2 instance.

Sprint UX-1.1 (Inès). Running this script (re-)produces:
  mockups/v2/insight_signal_b2c_telegram.txt
  mockups/v2/insight_signal_b2c_webapp.html
  mockups/v2/insight_signal_b2b_rest.json
  mockups/v2/insight_signal_b2b_webhook.json

This keeps the mockups in sync with the model: any field rename, validator
change, or schema-version bump is reflected by re-running this script.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.api.insight_signal_v2 import (
    ComplianceMeta,
    InsightSignalV2,
    NarrativeLanguage,
    SetupDirection,
    SignalLevels,
    Source,
    SourceType,
    Timeframe,
    VolatilityContext,
    to_b2b_dict,
    to_telegram_b2c,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MOCKUPS_DIR = REPO_ROOT / "mockups" / "v2"


def _canonical_signal() -> InsightSignalV2:
    """Bullish XAU M15 setup with full narrative + 3 sources cited."""
    return InsightSignalV2(
        id="ins_2026_05_01_xau_m15_001",
        instrument="XAUUSD",
        timeframe=Timeframe.M15,
        direction=SetupDirection.BULLISH_SETUP,
        conviction_0_100=72,
        levels=SignalLevels(
            entry=2350.00,
            stop=2340.00,
            target_1=2370.00,
            target_2=2390.00,
            invalidation=2335.00,
        ),
        volatility=VolatilityContext(
            regime="normal",
            forecast_atr_pct=0.42,
            naive_atr_pct=0.38,
        ),
        narrative_short=(
            "Setup haussier XAU M15. Cassure de structure confirmée + retest "
            "du FVG 2348-2350. Régime vol normal (forecast +10% vs naïve). "
            "Pas de news bloquante. Analyse algorithmique éducative."
        ),
        narrative_long=(
            "Lecture de marché : Smart Money structure haussière sur XAU/USD M15. "
            "La cassure de structure au-dessus de 2348 a été confirmée par un retest "
            "du Fair Value Gap entre 2348 et 2350. Le forecast HAR-RV anticipe une "
            "volatilité +10% vs la moyenne ATR14, conforme au régime normal. Le "
            "niveau d'invalidation structurel se situe à 2335 (sous le FVG inférieur). "
            "Première zone de prise partielle 2370, extension 2390 si la session NY "
            "soutient le momentum. Pas de blackout news actif sur les 30 prochaines "
            "minutes. Cette analyse est algorithmique et éducative ; elle ne "
            "constitue pas un conseil en investissement."
        ),
        narrative_language=NarrativeLanguage.FR,
        sources_cited=[
            Source(
                type=SourceType.PAPER,
                ref="https://onlinelibrary.wiley.com/doi/10.1002/9781119482086.ch7",
                label="López de Prado, AFML ch. 7 (CPCV)",
                quoted_excerpt="Combinatorial purged cross-validation generates 28 paths for N=8, k=2.",
            ),
            Source(
                type=SourceType.DATA,
                ref="https://www.cftc.gov/dea/newcot/deafutu.txt",
                label="CFTC COT Gold (088691) 2026-04-29 release",
            ),
            Source(
                type=SourceType.REPORT,
                ref="https://www.lbma.org.uk/quarterly-report-2026-q1",
                label="LBMA Q1 2026 quarterly review",
            ),
        ],
        compliance=ComplianceMeta(
            disclaimer_lang=NarrativeLanguage.FR,
            jurisdiction_blocked=["US", "CA-QC", "GB"],
            edge_claim=False,
            is_paper_demo=True,
        ),
        created_at_utc=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
        valid_until_utc=datetime(2026, 5, 1, 16, 0, tzinfo=timezone.utc),
        extras={"session": "london", "rr_ratio_displayed": True},
    )


def _telegram_mockup(signal: InsightSignalV2) -> str:
    body = to_telegram_b2c(signal)
    header = (
        "=" * 64
        + "\n"
        + "MOCKUP v2.0.0 — Telegram B2C — generated from InsightSignalV2\n"
        + "parse_mode=HTML, ≤ 800 chars, UE 2024/2811 compliant\n"
        + "=" * 64
        + "\n\n"
    )
    return header + body + "\n"


def _b2b_rest_mockup(signal: InsightSignalV2) -> str:
    payload = {
        "_comment": "MOCKUP v2.0.0 — B2B REST GET /api/v2/insights/{id}",
        "_endpoint": "GET /api/v2/insights/{id}",
        "_auth": "Authorization: Bearer <api_key>",
        **to_b2b_dict(signal),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _b2b_webhook_mockup(signal: InsightSignalV2) -> str:
    webhook = {
        "_comment": (
            "MOCKUP v2.0.0 — B2B Webhook POST. Broker registers a callback URL "
            "via /api/v2/webhooks/subscribe; we POST every new insight."
        ),
        "_http_request": {
            "method": "POST",
            "url": "<broker_callback_url_registered_at_subscribe>",
            "headers": {
                "Content-Type": "application/json",
                "User-Agent": "SmartSentinel-Webhook/2.0",
                "X-Sentinel-Insight-Id": signal.id,
                "X-Sentinel-Schema-Version": signal.schema_version,
                "X-Sentinel-Event": "insight.created",
                "X-Sentinel-Timestamp": "<unix_ts>",
                "X-Sentinel-Signature": "sha256=<HMAC_SHA256(secret, ts.body)>",
            },
            "_signature_recipe": (
                "Verify: sha256= + HMAC_SHA256(broker_webhook_secret, "
                "X-Sentinel-Timestamp + '.' + raw_request_body). "
                "Replay protection: reject if abs(now - X-Sentinel-Timestamp) > 300s."
            ),
        },
        "event": "insight.created",
        "delivery_id": "wh_dlv_01HKL3M2N6P9R7V8X4Y0Z1A2B",
        "delivery_attempt": 1,
        "occurred_at": signal.created_at_utc.isoformat(),
        "subscription": {
            "webhook_id": "wh_01HKL2ABCDEF3456",
            "tier": "BROKER_STANDARD",
            "white_label": True,
        },
        "insight": to_b2b_dict(signal),
    }
    return json.dumps(webhook, indent=2, ensure_ascii=False) + "\n"


def _webapp_mockup(signal: InsightSignalV2) -> str:
    sources_items = "\n    ".join(
        f'<li><a href="{s.ref}" rel="noopener">[{s.type.value}] {s.label}</a></li>'
        for s in signal.sources_cited
    )
    direction_label = {
        SetupDirection.BULLISH_SETUP: "SETUP HAUSSIER",
        SetupDirection.BEARISH_SETUP: "SETUP BAISSIER",
        SetupDirection.NEUTRAL: "NEUTRE",
    }[signal.direction]
    badge_class = {
        SetupDirection.BULLISH_SETUP: "bull",
        SetupDirection.BEARISH_SETUP: "bear",
        SetupDirection.NEUTRAL: "neutral",
    }[signal.direction]
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Smart Sentinel — {signal.instrument} {signal.timeframe.value}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 680px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
    .mockup-tag {{ font-size: 0.75rem; color: #6b7280; }}
    .badge {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 0.4rem; font-size: 0.85rem; font-weight: 600; }}
    .badge.bull {{ background: #dcfce7; color: #166534; }}
    .badge.bear {{ background: #fee2e2; color: #991b1b; }}
    .badge.neutral {{ background: #e5e7eb; color: #374151; }}
    .levels-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin: 1rem 0; font-variant-numeric: tabular-nums; }}
    .levels-grid div {{ padding: 0.5rem; background: #f9fafb; border-radius: 0.3rem; }}
    .levels-grid b {{ display: block; color: #6b7280; font-size: 0.75rem; text-transform: uppercase; }}
    .sources li {{ margin: 0.3rem 0; }}
    .disclaimer {{ margin-top: 2rem; padding: 0.8rem; background: #fef3c7; border-left: 4px solid #f59e0b; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <p class="mockup-tag">MOCKUP v2.0.0 — rendered from InsightSignalV2 + UI adapter</p>
  <h1>{signal.instrument} — {signal.timeframe.value}</h1>
  <p>
    <span class="badge {badge_class}">{direction_label}</span>
    Conviction <b>{signal.conviction_label.value.upper()}</b>
    ({signal.conviction_0_100}/100)
  </p>
  <div class="levels-grid">
    <div><b>Entrée</b>{signal.levels.entry}</div>
    <div><b>Stop</b>{signal.levels.stop}</div>
    <div><b>Cible 1</b>{signal.levels.target_1}</div>
    <div><b>Cible 2</b>{signal.levels.target_2}</div>
    <div><b>Invalidation</b>{signal.levels.invalidation}</div>
    <div><b>R:R</b>{signal.rr_ratio}</div>
  </div>
  <h2>Lecture du marché</h2>
  <p>{signal.narrative_long}</p>
  <h2>Sources citées</h2>
  <ul class="sources">
    {sources_items}
  </ul>
  <div class="disclaimer">
    <strong>Démonstration paper-trading.</strong> Smart Sentinel ne prétend pas
    posséder un edge prédictif. Cette analyse algorithmique est éducative et ne
    constitue pas un conseil en investissement. Performances passées n'indiquent
    pas les performances futures.
  </div>
</body>
</html>
"""


def main() -> None:
    MOCKUPS_DIR.mkdir(parents=True, exist_ok=True)
    signal = _canonical_signal()

    files = {
        "insight_signal_b2c_telegram.txt": _telegram_mockup(signal),
        "insight_signal_b2b_rest.json": _b2b_rest_mockup(signal),
        "insight_signal_b2b_webhook.json": _b2b_webhook_mockup(signal),
        "insight_signal_b2c_webapp.html": _webapp_mockup(signal),
    }
    for name, content in files.items():
        path = MOCKUPS_DIR / name
        path.write_text(content, encoding="utf-8")
        print(f"  Wrote {path} ({len(content)} bytes)")
    print(f"\nGenerated {len(files)} v2 mockups from canonical InsightSignalV2.")


if __name__ == "__main__":
    main()
