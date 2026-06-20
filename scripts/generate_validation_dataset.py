"""Génère le dataset de validation algorithmique (Phase 2 de l'audit).

Produit, à partir des CSV OHLCV historiques réels :
  - data/validation/marketreadings_2026_06_06.json    (60 MarketReadings)
  - docs/audits/VALIDATION_DATASET_2026_06_06.md       (liste des candles)
  - docs/audits/MANUAL_ANNOTATION_TEMPLATE_2026_06_06.md (template founder + résultats algo)
  - docs/audits/SCORING_TEMPLATE_2026_06_06.csv         (tableau de scoring)

Fidélité production
-------------------
Réutilise les fonctions de production sans modification :
  - SmartMoneyEngine.analyze()                    (détection BOS/CHOCH/OB/FVG/régime)
  - confluence_signal_to_structure()              (mapping → MarketReadingStructure)
  - candles_to_regime()                           (mapping → MarketReadingRegime)
  - tags_and_description()                         (tags + description template)
  - HaikuDescriptionEngine.generate()             (description LLM Haiku réelle)

Le confluence_signal est None (comme l'assembler par défaut, market_reading_assembler.py:84).

Échantillonnage
---------------
Le moteur tourne UNE fois sur l'historique H1 complet (warmup maximal, causal,
déterministe). Les bougies sont stratifiées par ÉTAT-ALGO (bos bull / bos bear /
choch / range / vol élevée / ordinaire) pour garantir la diversité du jeu à
annoter. La stratification utilise l'état détecté par l'algo lui-même — ce qui est
exactement ce qu'on veut faire valider (couvrir des conditions algo variées).
Sélection déterministe (pas de RNG) : bougies équi-espacées dans chaque strate.

Écart data documenté
---------------------
XAUUSD : période cible jan-mars 2026 (données disponibles jusqu'au 2026-04-29). OK.
EURUSD : CSV s'arrête au 2025-12-31 → période oct-déc 2025 (dernier trimestre
disponible). Écart assumé et signalé dans le rapport.

Usage : python scripts/generate_validation_dataset.py
Aucune clé API n'est jamais affichée (chargée depuis .env, masquée).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.intelligence.smart_money import SmartMoneyEngine  # noqa: E402
from src.intelligence.volatility_forecaster import resample_ohlcv  # noqa: E402
from src.intelligence.market_reading_mappers import (  # noqa: E402
    candles_to_regime,
    confluence_signal_to_structure,
    realized_levels,
    tags_and_description,
    _derive_trend,
    _derive_volatility,
)
from src.intelligence.market_reading_schema import (  # noqa: E402
    MarketReading,
    MarketReadingConditions,
    MarketReadingHeader,
)
from src.intelligence.market_reading_assembler import empty_events  # noqa: E402
from src.intelligence.haiku_description_engine import HaikuDescriptionEngine  # noqa: E402

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
LOOKBACK = 200          # fenêtre régime (= MarketReadingAssembler.DEFAULT_LOOKBACK)
PER_INSTRUMENT = 30
TARGETS = [
    {
        "instrument": "XAUUSD",
        "csv": ROOT / "data" / "XAU_15MIN_2019_2026.csv",
        "period_start": "2026-01-01",
        "period_end": "2026-03-31",
    },
    {
        "instrument": "EURUSD",
        "csv": ROOT / "data" / "EURUSD_15MIN_2019_2025.csv",
        "period_start": "2025-10-01",   # EUR CSV s'arrête 2025-12-31 (écart documenté)
        "period_end": "2025-12-31",
    },
]
OUT_JSON = ROOT / "data" / "validation" / "marketreadings_2026_06_06.json"


# --------------------------------------------------------------------------- #
# Cache Haiku minimal (interface .get/.put attendue par le moteur)
# --------------------------------------------------------------------------- #
class _MemCache:
    def __init__(self):
        self._d = {}

    def get(self, k):
        return self._d.get(k)

    def put(self, k, desc, source):
        self._d[k] = (desc, source)


def _load_env(root: Path) -> dict[str, str]:
    env = {}
    f = root / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _load_h1(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    h1 = resample_ohlcv(df.reset_index().rename(columns={"Date": "timestamp"}),
                        "M15", "H1")
    h1["timestamp"] = pd.to_datetime(h1["timestamp"])
    h1 = h1.set_index("timestamp").sort_index()
    return h1


def _resample_tf(df_m15: pd.DataFrame, target: str) -> pd.DataFrame:
    r = resample_ohlcv(df_m15.reset_index().rename(columns={"timestamp": "timestamp"}),
                       "M15", target)
    r["timestamp"] = pd.to_datetime(r["timestamp"])
    return r.set_index("timestamp").sort_index()


def _features_row_to_dict(row: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in row.items():
        if isinstance(v, bool):
            out[str(k)] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)) and not pd.isna(v):
            out[str(k)] = float(v)
    return out


def _pick_evenly(items: list, n: int) -> list:
    """Sélection déterministe équi-espacée de n éléments dans items (triés)."""
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _stratify_and_select(h1: pd.DataFrame, enriched: pd.DataFrame,
                        period_start: str, period_end: str) -> tuple[list, dict]:
    """Retourne (liste d'index positionnels sélectionnés, composition par strate)."""
    closes = h1["close"].tolist()
    highs = h1["high"].tolist()
    lows = h1["low"].tolist()
    ts = h1.index.tolist()

    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end) + timedelta(days=1)

    buckets = {"bos_bull": [], "bos_bear": [], "choch": [],
              "range": [], "high_vol": [], "ordinary": []}

    for i in range(LOOKBACK, len(h1)):
        t = ts[i]
        if not (start <= t < end):
            continue
        row = enriched.iloc[i]
        bos_event = float(row.get("BOS_EVENT", 0) or 0)
        choch = float(row.get("CHOCH_SIGNAL", 0) or 0)
        fvg = float(row.get("FVG_SIGNAL", 0) or 0)
        ob = float(row.get("OB_STRENGTH_NORM", 0) or 0)

        win_closes = closes[i - LOOKBACK + 1:i + 1]
        win_rows = [{"high": highs[j], "low": lows[j]}
                    for j in range(i - LOOKBACK + 1, i + 1)]
        trend = _derive_trend(win_closes)
        vol = _derive_volatility(win_rows)

        if bos_event > 0:
            buckets["bos_bull"].append(i)
        if bos_event < 0:
            buckets["bos_bear"].append(i)
        if choch != 0:
            buckets["choch"].append(i)
        if trend == "ranging":
            buckets["range"].append(i)
        if vol == "elevated":
            buckets["high_vol"].append(i)
        if (bos_event == 0 and choch == 0 and fvg == 0 and ob == 0
                and trend in ("bullish", "bearish") and vol == "normal"):
            buckets["ordinary"].append(i)

    # 5 par strate, déterministe
    selected: list[int] = []
    composition: dict[str, int] = {}
    for name, idxs in buckets.items():
        picks = _pick_evenly(sorted(idxs), 5)
        composition[name] = len(picks)
        for p in picks:
            if p not in selected:
                selected.append(p)

    # Top-up déterministe jusqu'à PER_INSTRUMENT depuis toute la période
    if len(selected) < PER_INSTRUMENT:
        all_period = [i for i in range(LOOKBACK, len(h1))
                      if start <= ts[i] < end and i not in selected]
        for p in _pick_evenly(sorted(all_period), PER_INSTRUMENT - len(selected)):
            selected.append(p)
    selected = sorted(set(selected))[:PER_INSTRUMENT]
    return selected, composition


def _build_reading(instrument: str, h1: pd.DataFrame, enriched: pd.DataFrame,
                  h4: pd.DataFrame, d1: pd.DataFrame, i: int,
                  haiku: HaikuDescriptionEngine) -> dict:
    ts = h1.index[i]
    bar_close_ts = ts.to_pydatetime() + timedelta(hours=1)  # close = open + 1h (H1)
    close_price = float(h1["close"].iloc[i])

    smc_features = _features_row_to_dict(enriched.iloc[i])
    # Merge real structural levels (F1/F2/F3) for THIS bar from the enriched window.
    smc_features.update(realized_levels(enriched, idx=i))

    structure = confluence_signal_to_structure(
        confluence_signal=None,
        smc_features=smc_features,
        bar_ts=bar_close_ts,
        current_price=close_price,
    )

    win = h1.iloc[max(0, i - LOOKBACK + 1):i + 1]
    candle_dicts = [{"open": float(r.open), "high": float(r.high),
                     "low": float(r.low), "close": float(r.close)}
                    for r in win.itertuples()]
    mtf = {
        "h4": [{"open": float(r.open), "high": float(r.high), "low": float(r.low),
                "close": float(r.close)} for r in h4[h4.index <= ts].tail(LOOKBACK).itertuples()],
        "d1": [{"open": float(r.open), "high": float(r.high), "low": float(r.low),
                "close": float(r.close)} for r in d1[d1.index <= ts].tail(LOOKBACK).itertuples()],
    }
    regime = candles_to_regime(candle_dicts, mtf_candles_above=mtf)

    tags, template_desc = tags_and_description(structure, regime)
    haiku_desc, source = haiku.generate(tags, regime, structure, close_price, instrument)

    reading = MarketReading(
        header=MarketReadingHeader(
            instrument=instrument, timeframe="H1",
            candle_close_ts=bar_close_ts, close_price=close_price,
        ),
        structure=structure, regime=regime, events=empty_events(),
        conditions=MarketReadingConditions(
            tags=tags, description=haiku_desc, description_source=source,
        ),
    )

    return {
        "instrument": instrument,
        "timeframe": "H1",
        "bar_open_ts": ts.isoformat(),
        "candle_close_ts": bar_close_ts.isoformat(),
        "close_price": close_price,
        "template_description": template_desc,
        "haiku_description": haiku_desc,
        "description_source": source,
        "market_reading": reading.model_dump(mode="json"),
    }


def main() -> int:
    env = _load_env(ROOT)
    api_key = env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    client = None
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            print("Anthropic client : OK (clé chargée, masquée)")
        except Exception as exc:
            print(f"Anthropic client indisponible ({type(exc).__name__}) — fallback template")
    else:
        print("ANTHROPIC_API_KEY absente — descriptions en fallback template")

    haiku = HaikuDescriptionEngine(anthropic_client=client, cache_store=_MemCache())

    all_readings: list[dict] = []
    meta_strata: dict[str, dict] = {}
    errors = 0

    for tgt in TARGETS:
        inst = tgt["instrument"]
        print(f"\n=== {inst} ({tgt['period_start']} → {tgt['period_end']}) ===")
        df_m15 = pd.read_csv(tgt["csv"])
        df_m15 = df_m15.rename(columns={c: c.capitalize() for c in df_m15.columns})
        df_m15["Date"] = pd.to_datetime(df_m15["Date"])
        df_m15 = df_m15.set_index("Date").sort_index()
        df_m15.columns = [c.lower() for c in df_m15.columns]
        df_m15 = df_m15[["open", "high", "low", "close", "volume"]]
        df_m15.index.name = "timestamp"

        h1 = _resample_tf(df_m15, "H1")
        h4 = _resample_tf(df_m15, "H4")
        d1 = _resample_tf(df_m15, "D1")
        print(f"  H1 bars: {len(h1)} | H4: {len(h4)} | D1: {len(d1)}")

        engine = SmartMoneyEngine(data=h1[["open", "high", "low", "close", "volume"]],
                                 config={}, verbose=False)
        enriched = engine.analyze()
        # L'engine peut dropper des lignes de warmup → réaligner par timestamp
        # sur l'index H1 complet pour que les positions iloc correspondent.
        enriched = enriched.reindex(h1.index)

        selected, composition = _stratify_and_select(
            h1, enriched, tgt["period_start"], tgt["period_end"])
        meta_strata[inst] = {"composition": composition, "n_selected": len(selected)}
        print(f"  strates: {composition} | sélectionnés: {len(selected)}")

        for cid, i in enumerate(selected, 1):
            try:
                rec = _build_reading(inst, h1, enriched, h4, d1, i, haiku)
                rec["candle_id"] = len(all_readings) + 1
                all_readings.append(rec)
            except Exception as exc:
                errors += 1
                print(f"  [WARN] bar {i} ({inst}) erreur: {type(exc).__name__}: {exc}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "generated_for": "2026-06-06 validation audit",
            "instruments": [t["instrument"] for t in TARGETS],
            "timeframe": "H1",
            "lookback": LOOKBACK,
            "total_candles": len(all_readings),
            "errors": errors,
            "strata": meta_strata,
            "periods": {t["instrument"]: f"{t['period_start']}..{t['period_end']}"
                        for t in TARGETS},
            "data_note": "EURUSD utilise oct-déc 2025 (CSV s'arrête 2025-12-31).",
            "haiku_live": client is not None,
        },
        "readings": all_readings,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ {len(all_readings)} readings → {OUT_JSON.relative_to(ROOT)} ({errors} erreurs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
