"""InsightV2 narrative generator — Claude LLM + deterministic template fallback.

Strict contract : NEVER mentions entry / stop-loss / take-profit / R:R.
The product is an indicator, not a trading bot. The narrative describes
the **structural state of the market** and the algorithm's **conviction
about that state** — never an action to take.

Two tiers :
- ``short`` : ≤ 400 chars, Telegram-friendly (FR default).
- ``long``  : ≤ 2000 chars, webapp / B2B (FR default).

LLM is used when an Anthropic API key is configured. Otherwise the
template generator produces a deterministic narrative from the
InsightSignalV2 contract (zero cost, zero latency, reproducible).

Both paths obey the EU 2024/2811 wording :
- BULLISH_SETUP / BEARISH_SETUP / NEUTRAL (not "BUY" / "SELL")
- "Lecture de marché" / "Market reading" (not "trade signal")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from src.intelligence.insight_v2.contract import InsightSignalV2

logger = logging.getLogger(__name__)


_SHORT_MAX_CHARS = 400
_LONG_MAX_CHARS = 2000


@dataclass
class NarrativeOutput:
    short: str
    long: str
    lang: str
    backend: str    # "claude" | "template"
    cost_usd: float = 0.0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------- #
# Template (deterministic, no LLM, no cost)
# ---------------------------------------------------------------------- #


_REGIME_FR = {
    "trend_bullish": "trend haussier",
    "trend_bearish": "trend baissier",
    "low_vol_trending": "trend faible volatilité",
    "low_vol_ranging": "range faible volatilité",
    "high_vol_stress": "stress haute volatilité",
    "normal": "régime normal",
    "unknown": "régime non classifié",
}


_REGIME_EN = {
    "trend_bullish": "bullish trend",
    "trend_bearish": "bearish trend",
    "low_vol_trending": "low-vol trending",
    "low_vol_ranging": "low-vol ranging",
    "high_vol_stress": "high-vol stress",
    "normal": "normal regime",
    "unknown": "unclassified",
}


def _setup_phrase_fr(direction: str) -> str:
    return {
        "bullish": "structure haussière",
        "bearish": "structure baissière",
        "neutral": "structure indéterminée",
    }.get(direction, "structure indéterminée")


def _setup_phrase_en(direction: str) -> str:
    return {
        "bullish": "bullish structure",
        "bearish": "bearish structure",
        "neutral": "indeterminate structure",
    }.get(direction, "indeterminate structure")


def _label_phrase_fr(label: str) -> str:
    return {
        "institutional": "conviction institutionnelle",
        "strong": "conviction forte",
        "moderate": "conviction modérée",
        "weak": "conviction faible",
    }.get(label, "conviction non classée")


def _label_phrase_en(label: str) -> str:
    return {
        "institutional": "institutional conviction",
        "strong": "strong conviction",
        "moderate": "moderate conviction",
        "weak": "weak conviction",
    }.get(label, "unclassified conviction")


def _fmt_zone(z: Optional[list | tuple]) -> str:
    if z is None or len(z) < 2:
        return ""
    return f"[{float(z[0]):,.2f} ; {float(z[1]):,.2f}]"


def _generate_template(insight: InsightSignalV2, lang: str = "fr") -> NarrativeOutput:
    start = time.time()
    s = insight.structure_readout
    r = insight.regime_readout
    v = insight.volatility_readout
    e = insight.event_readout
    is_fr = (lang == "fr")
    bias = s.direction
    fvg = _fmt_zone(s.fvg_zone)
    inval = (f"{s.structural_invalidation:.2f}"
             if s.structural_invalidation is not None else "n/d")

    # ---- SHORT ----
    if is_fr:
        setup = _setup_phrase_fr(bias)
        conviction = _label_phrase_fr(insight.conviction_label)
        regime_lbl = _REGIME_FR.get(r.hmm_label, r.hmm_label)
        short_parts = [
            f"{insight.asset} {insight.timeframe} : {setup} ({conviction}, {insight.conviction_0_100:.0f}/100).",
        ]
        if fvg:
            short_parts.append(f"FVG actif {fvg}.")
        if r.hmm_label != "unknown":
            short_parts.append(f"Régime : {regime_lbl}.")
        if e.news_blackout_active:
            short_parts.append("Blackout news actif.")
        elif e.next_event and e.next_event_in_minutes is not None and e.next_event_in_minutes < 240:
            short_parts.append(f"Prochain événement : {e.next_event} dans {e.next_event_in_minutes} min.")
        short_parts.append(f"Invalidation sous {inval}.")
        short = " ".join(short_parts)
    else:
        setup = _setup_phrase_en(bias)
        conviction = _label_phrase_en(insight.conviction_label)
        regime_lbl = _REGIME_EN.get(r.hmm_label, r.hmm_label)
        short_parts = [
            f"{insight.asset} {insight.timeframe}: {setup} ({conviction}, {insight.conviction_0_100:.0f}/100).",
        ]
        if fvg:
            short_parts.append(f"Active FVG {fvg}.")
        if r.hmm_label != "unknown":
            short_parts.append(f"Regime: {regime_lbl}.")
        if e.news_blackout_active:
            short_parts.append("News blackout active.")
        elif e.next_event and e.next_event_in_minutes is not None and e.next_event_in_minutes < 240:
            short_parts.append(f"Next event: {e.next_event} in {e.next_event_in_minutes} min.")
        short_parts.append(f"Structural invalidation below {inval}.")
        short = " ".join(short_parts)

    short = short[:_SHORT_MAX_CHARS]

    # ---- LONG ----
    if is_fr:
        lines = [
            f"La lecture algorithmique sur {insight.asset} {insight.timeframe} indique une "
            f"{_setup_phrase_fr(bias)} avec une {_label_phrase_fr(insight.conviction_label)} "
            f"({insight.conviction_0_100:.0f}/100, intervalle conformel "
            f"[{insight.conviction_interval['lower'] or 0:.0f}, "
            f"{insight.conviction_interval['upper'] or 0:.0f}]).",
        ]
        if s.bos_level is not None:
            lines.append(
                f"Cassure de structure détectée au niveau {s.bos_level:.2f}, "
                f"{'continuation haussière' if bias == 'bullish' else 'continuation baissière' if bias == 'bearish' else 'structure ambiguë'}."
            )
        if fvg:
            lines.append(
                f"Un Fair Value Gap actif est positionné en {fvg}"
                + (f" ({s.fvg_size_atr:.2f}×ATR)" if s.fvg_size_atr else "") + "."
            )
        if s.ob_zone is not None:
            ob_str = _fmt_zone(s.ob_zone)
            lines.append(f"Order Block ICT en {ob_str}"
                         + (f", force {s.ob_strength:.2f}" if s.ob_strength is not None else "") + ".")
        if s.retest_state != "none":
            lines.append(f"Statut de retest : {s.retest_state}.")
        if r.hmm_label != "unknown":
            lines.append(
                f"Régime HMM : {_REGIME_FR.get(r.hmm_label, r.hmm_label)}"
                + (f" (postérieur {r.hmm_posterior:.2f})" if r.hmm_posterior is not None else "")
                + ". "
                + (f"Probabilité de changepoint imminent : {r.bocpd_changepoint_prob:.0%}"
                    if r.bocpd_changepoint_prob is not None else "")
            )
        if v.forecast_atr is not None:
            naive = v.naive_atr or 0
            delta = v.forecast_vs_naive_pct
            lines.append(
                f"Forecast ATR (HAR-RV) : {v.forecast_atr:.2f}, ATR naïf : {naive:.2f}"
                + (f", écart {delta:+.1f}%" if delta is not None else "") + "."
            )
        if r.regime_gate_decision != "TRADE":
            lines.append(f"Le régime gate signale : {r.regime_gate_decision}.")
        if e.news_blackout_active:
            lines.append("Un blackout news est actuellement actif : prudence accrue recommandée.")
        elif e.next_event:
            lines.append(f"Prochain événement high-impact : {e.next_event}"
                         + (f" dans {e.next_event_in_minutes} minutes" if e.next_event_in_minutes else "") + ".")
        lines.append(f"Le niveau d'invalidation structurelle se situe à {inval}.")
        lines.append(
            "Cette lecture est descriptive et algorithmique. Elle ne constitue ni un signal de "
            "trading ni un conseil en investissement. La composition du trade revient au trader."
        )
    else:
        lines = [
            f"The algorithmic reading on {insight.asset} {insight.timeframe} indicates a "
            f"{_setup_phrase_en(bias)} with {_label_phrase_en(insight.conviction_label)} "
            f"({insight.conviction_0_100:.0f}/100, conformal interval "
            f"[{insight.conviction_interval['lower'] or 0:.0f}, "
            f"{insight.conviction_interval['upper'] or 0:.0f}]).",
        ]
        if s.bos_level is not None:
            lines.append(f"Break of structure detected at level {s.bos_level:.2f}.")
        if fvg:
            lines.append(f"Active Fair Value Gap at {fvg}"
                         + (f" ({s.fvg_size_atr:.2f}×ATR)" if s.fvg_size_atr else "") + ".")
        if s.ob_zone is not None:
            lines.append(f"ICT Order Block at {_fmt_zone(s.ob_zone)}"
                         + (f", strength {s.ob_strength:.2f}" if s.ob_strength is not None else "") + ".")
        if r.hmm_label != "unknown":
            lines.append(
                f"HMM regime: {_REGIME_EN.get(r.hmm_label, r.hmm_label)}"
                + (f" (posterior {r.hmm_posterior:.2f})" if r.hmm_posterior is not None else "")
            )
        if v.forecast_atr is not None:
            lines.append(
                f"HAR-RV ATR forecast: {v.forecast_atr:.2f} "
                f"(naive {v.naive_atr or 0:.2f}, delta {v.forecast_vs_naive_pct or 0:+.1f}%)."
            )
        if e.news_blackout_active:
            lines.append("News blackout is currently active — heightened caution recommended.")
        elif e.next_event:
            lines.append(f"Next high-impact event: {e.next_event}"
                         + (f" in {e.next_event_in_minutes} minutes" if e.next_event_in_minutes else "") + ".")
        lines.append(f"Structural invalidation level: {inval}.")
        lines.append(
            "This reading is descriptive and algorithmic. It does not constitute a trading signal "
            "nor investment advice. Trade composition is at the trader's discretion."
        )

    long = " ".join(lines)[:_LONG_MAX_CHARS]
    return NarrativeOutput(
        short=short,
        long=long,
        lang=lang,
        backend="template",
        cost_usd=0.0,
        latency_ms=(time.time() - start) * 1000,
    )


# ---------------------------------------------------------------------- #
# Claude LLM backend
# ---------------------------------------------------------------------- #


_LLM_SYSTEM_FR = (
    "Tu es un analyste de marché institutionnel rigoureux. Tu rédiges des LECTURES descriptives "
    "(pas des recommandations de trading). Wording compliance UE 2024/2811 : utilise « structure "
    "haussière / baissière », JAMAIS « achetez / vendez ». N'invente AUCUN prix d'entrée, stop-loss, "
    "take-profit, ratio R/R, ni instruction temporelle. Tu décris la situation, le trader décide. "
    "Style : phrases courtes, faits structurels, jargon ICT/HMM/conformel sans pédanterie."
)

_LLM_SYSTEM_EN = (
    "You are a rigorous institutional market analyst. You write descriptive READINGS (not trade "
    "recommendations). EU 2024/2811 compliance : use 'bullish / bearish structure', NEVER 'buy / "
    "sell'. NEVER invent entry prices, stop-loss, take-profit, R:R ratios, or timing instructions. "
    "Style : short sentences, structural facts, ICT/HMM/conformal jargon without pedantry."
)


def _generate_llm(
    insight: InsightSignalV2,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    lang: str = "fr",
) -> NarrativeOutput:
    """Generate via Claude API. Falls back to template on error."""
    start = time.time()
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as exc:
        logger.warning("anthropic init failed (%s) — falling back to template", exc)
        return _generate_template(insight, lang=lang)

    system = _LLM_SYSTEM_FR if lang == "fr" else _LLM_SYSTEM_EN
    payload = insight.to_dict()
    # Strip narrative/sources (avoid feedback loop) and trade fields
    for k in ("narrative_short", "narrative_long", "sources_cited"):
        payload.pop(k, None)
    import json as _json
    facts_block = _json.dumps(payload, ensure_ascii=False, indent=2)

    user_prompt = (
        (f"Voici les FAITS structurels d'une lecture de marché Smart Sentinel AI sur "
         f"{insight.asset} {insight.timeframe} :\n\n{facts_block}\n\n"
         f"Produis UN narratif court (≤ {_SHORT_MAX_CHARS} chars, Telegram) puis UN narratif long "
         f"(≤ {_LONG_MAX_CHARS} chars, webapp).\n\n"
         f"Format de sortie strict (3 lignes, pas de markdown bold) :\n"
         f"SHORT: <le narratif court>\n"
         f"LONG: <le narratif long>\n")
        if lang == "fr" else
        (f"Here are the structural FACTS of a Smart Sentinel AI market reading on "
         f"{insight.asset} {insight.timeframe}:\n\n{facts_block}\n\n"
         f"Produce ONE short narrative (≤ {_SHORT_MAX_CHARS} chars, Telegram) then ONE long narrative "
         f"(≤ {_LONG_MAX_CHARS} chars, webapp).\n\n"
         f"Strict output format (3 lines, no markdown bold):\n"
         f"SHORT: <short narrative>\n"
         f"LONG: <long narrative>\n")
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        cost_usd = 0.0  # cost tracking optional
    except Exception as exc:
        logger.warning("Claude call failed (%s) — falling back to template", exc)
        return _generate_template(insight, lang=lang)

    short, long = _parse_short_long(text)
    if not short or not long:
        logger.warning("LLM output unparseable, falling back to template. Raw: %r", text[:200])
        return _generate_template(insight, lang=lang)

    return NarrativeOutput(
        short=short[:_SHORT_MAX_CHARS],
        long=long[:_LONG_MAX_CHARS],
        lang=lang,
        backend="claude",
        cost_usd=cost_usd,
        latency_ms=(time.time() - start) * 1000,
    )


def _parse_short_long(text: str) -> tuple[str, str]:
    short = ""
    long = ""
    lines = text.splitlines()
    in_long = False
    long_buf: list[str] = []
    for line in lines:
        ls = line.strip()
        if ls.upper().startswith("SHORT:"):
            short = ls.split(":", 1)[1].strip()
            in_long = False
        elif ls.upper().startswith("LONG:"):
            long = ls.split(":", 1)[1].strip()
            long_buf = [long] if long else []
            in_long = True
        elif in_long and ls:
            long_buf.append(ls)
    if long_buf:
        long = " ".join(long_buf).strip()
    return short, long


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #


class InsightV2NarrativeGenerator:
    """Drop-in narrative generator for InsightV2Builder."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        lang: str = "fr",
        force_template: bool = False,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model
        self._lang = lang
        self._force_template = bool(force_template)

    def generate(self, insight: InsightSignalV2, lang: Optional[str] = None) -> NarrativeOutput:
        lang = lang or self._lang
        if self._force_template or not self._api_key:
            return _generate_template(insight, lang=lang)
        return _generate_llm(insight, api_key=self._api_key, model=self._model, lang=lang)


__all__ = ["InsightV2NarrativeGenerator", "NarrativeOutput"]
