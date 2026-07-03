"""Diagnostics de rejet OB — mission 2026-07-02.

Couvre les 5 exigences de la mission :
  (a) la raison rapportée pour un candidat rejeté = le critère réellement échoué ;
  (b) non-régression — complété par tests/test_ob_golden_nonregression.py ;
      ici : le flag ``with_rejects`` ne change pas d'un octet les zones surfacées,
      et les Series combinées == l'expression inline legacy (oracle) ;
  (c) SOURCE UNIQUE — changer un seuil de décision (OB_REQUIRE_FVG) change la
      décision du moteur ET la raison rapportée, dans le même run ;
  (d) l'IA ne relaie que des raisons réellement produites par le moteur, et le
      payload est honnête quand il n'y a pas de diagnostic (aucune fabrication) ;
  (e) aucune sortie prédictive — labels/disclaimer passent le filtre de tokens
      interdits et ne contiennent aucun vocabulaire d'anticipation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd
import pytest

from src.environment.strategy_features import (
    OB_CRITERIA,
    OB_FVG_CRITERION,
    SMCConfig,
    SmartMoneyEngine,
    combine_ob_conditions,
    ob_candidate_conditions,
)
from src.intelligence.market_reading_mappers import collect_zones, ob_zone_id
from src.intelligence.ob_diagnostics import (
    CRITERIA_LABELS_FR,
    DISCLAIMER_FR,
    REJECT_LABELS_FR,
    diagnose_ob,
    resolve_bar,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ob_golden" / "XAUUSD_M15.csv"


def _load_fixture_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE, index_col="ts", parse_dates=["ts"])
    df.index = pd.to_datetime(df.index, utc=True)
    return df


@pytest.fixture(scope="module")
def enriched_cfg() -> tuple[Any, Any]:
    engine = SmartMoneyEngine(data=_load_fixture_df(), config={}, verbose=False)
    return engine.analyze(compute_divergence=False), engine.config


# --------------------------------------------------------------------------- #
# (b) Oracle — the named/combined Series equal the legacy inline expression
# --------------------------------------------------------------------------- #


def test_combined_conditions_equal_legacy_inline_expression(enriched_cfg) -> None:
    """The pre-refactor expressions, copied verbatim as test oracle: the shared
    condition builder must reproduce them bit-for-bit (default AND legacy FVG
    mode)."""
    df, _ = enriched_cfg
    legacy_bull = (
        (df['close'].shift(1) < df['open'].shift(1))
        & (df['close'] > df['open'])
        & (df['high'] > df['high'].shift(1))
    )
    legacy_bear = (
        (df['close'].shift(1) > df['open'].shift(1))
        & (df['close'] < df['open'])
        & (df['low'] < df['low'].shift(1))
    )
    legacy_fvg = (df['FVG_SIGNAL'] != 0).shift(1).fillna(False)

    conds = ob_candidate_conditions(df, SMCConfig())
    assert combine_ob_conditions(conds, "bullish").equals(legacy_bull)
    assert combine_ob_conditions(conds, "bearish").equals(legacy_bear)

    conds_req = ob_candidate_conditions(df, SMCConfig(OB_REQUIRE_FVG=True))
    assert combine_ob_conditions(conds_req, "bullish").equals(legacy_bull & legacy_fvg)
    assert combine_ob_conditions(conds_req, "bearish").equals(legacy_bear & legacy_fvg)


def test_with_rejects_flag_never_changes_surfaced_zones(enriched_cfg) -> None:
    df, _ = enriched_cfg
    last = len(df) - 1
    off = collect_zones(df, idx=last)
    on = collect_zones(df, idx=last, with_rejects=True)
    assert on["order_blocks"] == off["order_blocks"]
    assert on["fair_value_gaps"] == off["fair_value_gaps"]
    assert "rejected_order_blocks" not in off
    assert len(on["rejected_order_blocks"]) > 0  # the fixture does exercise rejects


def test_every_engine_criterion_has_a_french_label() -> None:
    """Adding a criterion to the engine without a diagnostic label must fail
    loudly — the two views cannot drift apart silently."""
    keys = {name for names in OB_CRITERIA.values() for name in names}
    keys.add(OB_FVG_CRITERION)
    assert keys <= set(CRITERIA_LABELS_FR)


# --------------------------------------------------------------------------- #
# (a) Reported reason == the branch that actually dropped/failed the candidate
# --------------------------------------------------------------------------- #


def test_invalidated_reject_reason_matches_lifecycle_reality(enriched_cfg) -> None:
    df, cfg = enriched_cfg
    last = len(df) - 1
    zones = collect_zones(df, idx=last, with_rejects=True)
    invalidated = [
        z for z in zones["rejected_order_blocks"]
        if z["reject_reason"] == "invalidated_close_through"
    ]
    assert invalidated, "fixture must contain at least one invalidated OB"
    z = invalidated[0]

    candle_ts = df.index[z["_k"] - 1]  # the zone candle = the user's candle
    diag = diagnose_ob(df, cfg, ts=candle_ts)
    assert diag["status"] == "was_rejected"
    assert diag["side"] == z["direction"]
    assert diag["zone"]["reject_reason"] == "invalidated_close_through"
    assert diag["zone"]["reject_label_fr"] == REJECT_LABELS_FR["invalidated_close_through"]
    assert diag["zone"]["invalidated_at"] is not None

    # The reported invalidation bar REALLY closed through the zone (the reason
    # is a fact of the engine's data, not a narrative).
    inv_ts = pd.Timestamp(diag["zone"]["invalidated_at"])
    j = df.index.get_loc(inv_ts)
    if z["direction"] == "bullish":
        assert float(df["close"].iloc[j]) < z["level_low"]
    else:
        assert float(df["close"].iloc[j]) > z["level_high"]


def test_capped_reject_reason_reports_rank_and_cap(enriched_cfg) -> None:
    df, cfg = enriched_cfg
    last = len(df) - 1
    zones = collect_zones(df, idx=last, max_per_type=3, with_rejects=True)
    capped = [
        z for z in zones["rejected_order_blocks"]
        if z["reject_reason"] == "capped_max_zones"
    ]
    assert capped, "with cap=3 the fixture must overflow"
    z = capped[0]

    diag = diagnose_ob(df, cfg, ts=df.index[z["_k"] - 1], max_per_type=3)
    assert diag["status"] == "was_rejected"
    assert diag["zone"]["reject_reason"] == "capped_max_zones"
    assert diag["zone"]["cap_max"] == 3
    assert diag["zone"]["cap_rank"] >= 3


def test_surfaced_zone_reports_is_order_block_with_reading_id(enriched_cfg) -> None:
    df, cfg = enriched_cfg
    last = len(df) - 1
    surfaced = collect_zones(df, idx=last)["order_blocks"][0]
    diag = diagnose_ob(df, cfg, ts=df.index[surfaced["_k"] - 1])
    assert diag["status"] == "is_order_block"
    assert diag["zone"]["lifecycle_status"] == surfaced["status"]
    # Same id format as the MarketReading models — the agent can cross-reference.
    assert diag["zone"]["id"] == ob_zone_id(surfaced["direction"], surfaced["created_at"])


def _mk_frame(rows: list[dict]) -> pd.DataFrame:
    index = pd.date_range("2026-06-29 10:00:00+00:00", periods=len(rows), freq="15min", name="ts")
    df = pd.DataFrame(rows, index=index)
    if "FVG_SIGNAL" not in df.columns:
        df["FVG_SIGNAL"] = 0.0
    return df


def test_not_candidate_reports_the_exact_failing_criterion() -> None:
    """Crafted case: bearish candle followed by a bullish candle that does NOT
    break its high — the only failing bullish criterion must be breaks_prev_high."""
    flat = {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0}
    rows = [dict(flat) for _ in range(6)]
    rows[3] = {"open": 101.0, "high": 101.5, "low": 99.5, "close": 100.0}  # bearish, high 101.5
    rows[4] = {"open": 100.0, "high": 101.0, "low": 99.8, "close": 100.8}  # bullish, no break
    df = _mk_frame(rows)

    diag = diagnose_ob(df, SMCConfig(), ts=df.index[3])
    assert diag["status"] == "not_candidate"
    checks = {c["criterion"]: c for c in diag["sides"]["bullish"]["checks"]}
    assert checks["prev_candle_bearish"]["passed"] is True
    assert checks["confirm_candle_bullish"]["passed"] is True
    assert checks["breaks_prev_high"]["passed"] is False
    # Observed facts anchor the explanation to real numbers.
    assert checks["breaks_prev_high"]["observed"] == {"next_high": 101.0, "candle_high": 101.5}
    assert checks["breaks_prev_high"]["label_fr"] == CRITERIA_LABELS_FR["breaks_prev_high"]


def test_last_bar_is_awaiting_next_candle(enriched_cfg) -> None:
    df, cfg = enriched_cfg
    diag = diagnose_ob(df, cfg, ts=df.index[-1])
    assert diag["status"] == "awaiting_next_candle"
    assert "SUIVANTE" in diag["note_fr"]


def test_diagnostic_payload_is_json_serializable(enriched_cfg) -> None:
    df, cfg = enriched_cfg
    diag = diagnose_ob(df, cfg, ts=df.index[100])
    json.dumps(diag, default=str)


# --------------------------------------------------------------------------- #
# Bar resolution — deterministic, honest when it cannot resolve
# --------------------------------------------------------------------------- #


def test_price_resolution_picks_most_recent_containing_bar() -> None:
    flat = {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0}
    rows = [dict(flat) for _ in range(6)]
    rows[2] = {"open": 100.0, "high": 100.5, "low": 95.0, "close": 100.0}
    rows[4] = {"open": 100.0, "high": 100.5, "low": 95.0, "close": 100.0}
    df = _mk_frame(rows)
    bar, info = resolve_bar(df, price=95.2)
    assert bar == 4  # most recent of the two containing bars
    assert info["matched_by"] == "price"
    assert info["matched_bars"] == 2


def test_unresolved_price_and_ts_are_honest(enriched_cfg) -> None:
    df, cfg = enriched_cfg
    diag = diagnose_ob(df, cfg, price=50.0)  # far outside the XAU window
    assert diag["status"] == "unresolved"
    assert diag["resolution"]["reason"] == "price_not_touched"
    assert "checks" not in json.dumps(diag)  # nothing fabricated

    diag2 = diagnose_ob(df, cfg, ts="2020-01-01T00:00:00Z")
    assert diag2["status"] == "unresolved"
    assert diag2["resolution"]["reason"] == "out_of_window"

    diag3 = diagnose_ob(df, cfg)
    assert diag3["status"] == "unresolved"
    assert diag3["resolution"]["reason"] == "no_reference"


# --------------------------------------------------------------------------- #
# (c) SINGLE SOURCE — a decision-threshold change moves decision AND reason
# --------------------------------------------------------------------------- #


def test_threshold_change_moves_decision_and_reason_together(enriched_cfg) -> None:
    """Flip OB_REQUIRE_FVG on the SAME candles: the engine's accept/reject
    columns change AND the diagnostic's failing criterion is exactly the flipped
    clause — proof there is no parallel logic that could tell a different story."""
    enriched_default, cfg_default = enriched_cfg
    engine_fvg = SmartMoneyEngine(
        data=_load_fixture_df(), config={"OB_REQUIRE_FVG": True}, verbose=False
    )
    enriched_fvg = engine_fvg.analyze(compute_divergence=False)
    cfg_fvg = engine_fvg.config
    assert cfg_fvg.OB_REQUIRE_FVG is True

    conds = ob_candidate_conditions(enriched_default, cfg_default)
    bull = combine_ob_conditions(conds, "bullish")
    fvg = conds[OB_FVG_CRITERION]
    flipped = [
        d for d in range(1, len(enriched_default) - 1)
        if bool(bull.iloc[d]) and not bool(fvg.iloc[d])
    ]
    assert flipped, "fixture must contain a bullish OB without adjacent FVG"
    d = flipped[0]
    candle_ts = enriched_default.index[d - 1]

    # Decision changed at the ENGINE level (output columns)...
    assert not pd.isna(enriched_default["BULLISH_OB_HIGH"].iloc[d])
    assert pd.isna(enriched_fvg["BULLISH_OB_HIGH"].iloc[d])

    # ...and the DIAGNOSTIC follows, blaming exactly the flipped clause.
    diag_default = diagnose_ob(enriched_default, cfg_default, ts=candle_ts)
    assert diag_default["status"] in {"is_order_block", "was_rejected"}
    diag_fvg = diagnose_ob(enriched_fvg, cfg_fvg, ts=candle_ts)
    assert diag_fvg["status"] == "not_candidate"
    checks = {c["criterion"]: c["passed"] for c in diag_fvg["sides"]["bullish"]["checks"]}
    assert checks[OB_FVG_CRITERION] is False
    assert all(checks[name] for name in OB_CRITERIA["bullish"])


# --------------------------------------------------------------------------- #
# Assembler access — same candle cache, honest no_data
# --------------------------------------------------------------------------- #


class _FakeCandlesStore:
    def __init__(self, candles: list) -> None:
        self._candles = candles

    def get_last_n_candles(self, instrument: str, timeframe: str, n: int) -> list:
        return self._candles[-n:]


def _fixture_candles(limit: int = 200) -> list:
    df = _load_fixture_df().tail(limit)
    return [
        SimpleNamespace(
            ts=ts.to_pydatetime(),
            open=float(r["open"]), high=float(r["high"]),
            low=float(r["low"]), close=float(r["close"]),
            volume=float(r["volume"]),
        )
        for ts, r in df.iterrows()
    ]


def _make_assembler(candles: list):
    from src.intelligence.market_reading_assembler import MarketReadingAssembler

    return MarketReadingAssembler(
        data_provider=None,
        readings_store=None,
        candles_store=_FakeCandlesStore(candles),
        clock=lambda: datetime(2026, 6, 30, 20, 30, tzinfo=timezone.utc),
    )


def test_assembler_diagnostic_end_to_end() -> None:
    candles = _fixture_candles()
    asm = _make_assembler(candles)
    diag = asm.get_ob_diagnostic("XAUUSD", "M15", ts=candles[-40].ts.isoformat())
    assert diag["instrument"] == "XAUUSD"
    assert diag["timeframe"] == "M15"
    assert diag["status"] in {
        "is_order_block", "was_rejected", "not_candidate",
        "awaiting_next_candle", "unresolved",
    }
    assert diag["resolution"]["resolved"] is True


def test_assembler_diagnostic_no_data_is_honest() -> None:
    asm = _make_assembler([])
    diag = asm.get_ob_diagnostic("XAUUSD", "M15", price=4114.0)
    assert diag["status"] == "no_data"
    assert diag["resolution"]["resolved"] is False


# --------------------------------------------------------------------------- #
# (d) Chatbot — relays engine reasons only, honest without a diagnostic
# --------------------------------------------------------------------------- #


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _ToolUseBlock:
    name: str
    input: dict
    id: str = "tu_1"
    type: str = "tool_use"


@dataclass
class _StubResponse:
    content: list
    stop_reason: str


class _StubMessages:
    def __init__(self, parent: "_StubClient") -> None:
        self._p = parent

    def create(self, **kwargs: Any) -> Any:
        self._p.calls.append(kwargs)
        return self._p.responses.pop(0)


class _StubClient:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.messages = _StubMessages(self)


class _DiagAssembler:
    """Assembler stub: canned get_ob_diagnostic, unused get_or_generate."""

    def __init__(self, diag: dict) -> None:
        self.diag = diag
        self.diag_calls: list[dict] = []

    def get_or_generate(self, instrument: str, timeframe: str):  # summary path
        raise RuntimeError("not used in this test")

    def get_ob_diagnostic(
        self, instrument: str, timeframe: str,
        ts: Optional[Any] = None, price: Optional[float] = None,
    ) -> dict:
        self.diag_calls.append(
            {"instrument": instrument, "timeframe": timeframe, "ts": ts, "price": price}
        )
        return self.diag


def _make_bot(responses: list, diag: dict):
    from src.intelligence.chatbot.chatbot import Chatbot
    from src.intelligence.chatbot.signal_summary_provider import SignalSummaryProvider

    assembler = _DiagAssembler(diag)
    provider = SignalSummaryProvider(assembler)
    client = _StubClient(responses)
    return Chatbot(client, provider, assembler), client, assembler


def test_tool_schema_and_prompt_wire_the_diagnostic() -> None:
    from src.intelligence.chatbot.chatbot import SYSTEM_PROMPT_TEMPLATE, TOOL_SCHEMAS

    names = [t["name"] for t in TOOL_SCHEMAS]
    assert "get_ob_diagnostic" in names
    schema = next(t for t in TOOL_SCHEMAS if t["name"] == "get_ob_diagnostic")
    assert set(schema["input_schema"]["required"]) == {"instrument", "timeframe"}
    assert {"price", "ts"} <= set(schema["input_schema"]["properties"])
    # The prompt forbids fabricated reasons and mandates the tool.
    assert "get_ob_diagnostic" in SYSTEM_PROMPT_TEMPLATE
    assert "Jamais une raison de ton cru" in SYSTEM_PROMPT_TEMPLATE
    assert "tu ne devines pas" in SYSTEM_PROMPT_TEMPLATE


def test_chatbot_relays_engine_reject_reason() -> None:
    diag = {
        "status": "was_rejected",
        "side": "bullish",
        "zone": {
            "reject_reason": "invalidated_close_through",
            "reject_label_fr": REJECT_LABELS_FR["invalidated_close_through"],
            "invalidated_at": "2026-06-30T14:00:00+00:00",
        },
        "resolution": {"resolved": True, "matched_by": "price", "input_price": 4114.0},
        "disclaimer_fr": DISCLAIMER_FR,
    }
    r1 = _StubResponse(
        [_ToolUseBlock("get_ob_diagnostic", {"instrument": "XAUUSD", "timeframe": "M15", "price": 4114})],
        "tool_use",
    )
    r2 = _StubResponse(
        [_TextBlock("Un Order Block s'était formé ici, mais une bougie a clôturé à travers la zone le 30/06 : le moteur l'a invalidé.")],
        "end_turn",
    )
    bot, client, assembler = _make_bot([r1, r2], diag)
    out = bot.chat("Pourquoi la bougie à 4114 n'est pas un OB ?")
    assert out.blocked_reason is None
    assert assembler.diag_calls == [
        {"instrument": "XAUUSD", "timeframe": "M15", "ts": None, "price": 4114}
    ]
    # The tool_result handed to the model is the engine payload verbatim.
    tool_result = client.calls[1]["messages"][-1]["content"][0]["content"]
    assert "invalidated_close_through" in tool_result
    assert "invalidé" in out.content


def test_chatbot_honest_payload_when_no_diagnostic_exists() -> None:
    diag = {
        "status": "unresolved",
        "resolution": {"resolved": False, "reason": "price_not_touched",
                       "input_price": 9999.0, "window_low": 4050.0, "window_high": 4180.0},
        "disclaimer_fr": DISCLAIMER_FR,
    }
    r1 = _StubResponse(
        [_ToolUseBlock("get_ob_diagnostic", {"instrument": "XAUUSD", "timeframe": "M15", "price": 9999})],
        "tool_use",
    )
    r2 = _StubResponse(
        [_TextBlock("Je n'ai pas le détail pour cette bougie : ce prix n'a pas été touché sur la période analysée.")],
        "end_turn",
    )
    bot, client, _ = _make_bot([r1, r2], diag)
    out = bot.chat("Pourquoi 9999 n'est pas un OB ?")
    assert out.blocked_reason is None
    tool_result = client.calls[1]["messages"][-1]["content"][0]["content"]
    # No criteria, no reasons — nothing available to fabricate from.
    assert "price_not_touched" in tool_result
    assert "checks" not in tool_result
    assert "reject_reason" not in tool_result


# --------------------------------------------------------------------------- #
# (e) No predictive output — labels pass the forbidden-token layers
# --------------------------------------------------------------------------- #


def test_labels_pass_output_filter_and_forbidden_tokens() -> None:
    from src.intelligence.chatbot.output_filter import OutputFilter
    from src.intelligence.market_reading_mappers import FORBIDDEN_TOKENS

    corpus = " ".join(
        [*CRITERIA_LABELS_FR.values(), *REJECT_LABELS_FR.values(), DISCLAIMER_FR]
    )
    low = corpus.lower()
    for token in FORBIDDEN_TOKENS:
        assert token not in low, f"forbidden token in diagnostic labels: {token}"
    check = OutputFilter().check(corpus)
    assert not check.contaminated, f"output filter flags labels: {check.matched_tokens}"
    # Descriptive past/present only — no anticipation vocabulary.
    for word in ("va ", "devrait", "anticip", "probabl", "cible", "objectif", "signal"):
        assert word not in low, f"predictive vocabulary in labels: {word}"
