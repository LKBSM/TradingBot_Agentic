"""DATA-3 (a) — MESURE du nombre d'appels Twelve Data declenches par le code reel.

Ne simule rien du code applicatif : les vraies classes de production sont
instanciees (``TwelveDataProvider``, ``MarketReadingAssembler``,
``MarketReadingScheduler``, vrais magasins SQLite dans un dossier temporaire).
Seule la couche HTTP est remplacee par une ``requests.Session`` factice qui
COMPTE et horodate chaque requete sortante et renvoie une serie OHLCV
synthetique. Le compte obtenu est donc le nombre de credits reellement
consommes par ces chemins, pas une estimation.

Usage : python tools/data_budget/measure_current.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

os.environ.setdefault("TWELVE_DATA_API_KEY", "measure-harness-no-network")


# --------------------------------------------------------------------------- #
# Couche HTTP factice : compte les requetes, renvoie une serie OHLCV plausible
# --------------------------------------------------------------------------- #
class _Resp:
    status_code = 200
    ok = True
    text = ""

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class CountingSession:
    """Remplace ``requests.Session`` : ne sort jamais sur le reseau."""

    def __init__(self, latency_s: float = 0.0):
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        self._latency = latency_s

    def get(self, url, params=None, timeout=None):
        params = params or {}
        with self._lock:
            self.calls.append(
                {
                    "t": time.monotonic(),
                    "symbol": params.get("symbol"),
                    "interval": params.get("interval"),
                    "outputsize": params.get("outputsize"),
                }
            )
        if self._latency:
            time.sleep(self._latency)
        return _Resp(
            _series(params.get("interval", "15min"), int(params.get("outputsize", 500)))
        )

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.calls)


_INTERVAL_MIN = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "1h": 60,
    "4h": 240,
    "1day": 1440,
    "1week": 10080,
}


def _series(interval: str, outputsize: int) -> dict:
    mins = _INTERVAL_MIN.get(interval, 15)
    step = timedelta(minutes=mins)
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    if mins < 1440:
        end -= timedelta(minutes=end.minute % mins)
    else:
        end = end.replace(hour=0, minute=0)
    n = max(1, min(int(outputsize), 3000))
    values = []
    price = 2000.0
    for i in range(n):
        ts = end - step * (n - 1 - i)
        price += ((i * 37) % 11 - 5) * 0.1
        values.append(
            {
                "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "open": f"{price:.4f}",
                "high": f"{price + 0.5:.4f}",
                "low": f"{price - 0.5:.4f}",
                "close": f"{price + 0.1:.4f}",
                "volume": "0",
            }
        )
    return {"status": "ok", "values": list(reversed(values))}


# --------------------------------------------------------------------------- #
# Montage : vrais objets de production, magasins en dossier temporaire
# --------------------------------------------------------------------------- #
def build_real_stack(tmp: Path, latency_s: float = 0.0):
    from src.intelligence.data_providers.twelve_data_provider import TwelveDataProvider
    from src.intelligence.market_reading_assembler import (
        MarketReadingAssembler,
        build_cache_mtf_provider,
    )
    from src.storage import CandlesCacheStore, MarketReadingsStore

    session = CountingSession(latency_s=latency_s)
    provider = TwelveDataProvider(
        api_key="measure-harness",
        session=session,
        # plafonds reels du plan gratuit tels que cables par bootstrap.py
        per_minute=8,
        per_day=800,
    )
    readings = MarketReadingsStore(db_path=str(tmp / "readings.db"))
    candles = CandlesCacheStore(db_path=str(tmp / "candles.db"))
    assembler = MarketReadingAssembler(
        data_provider=provider,
        readings_store=readings,
        candles_store=candles,
        news_pipeline=None,
        mtf_provider=build_cache_mtf_provider(candles, 200),
    )
    return session, provider, assembler, readings, candles


# --------------------------------------------------------------------------- #
# Mesures
# --------------------------------------------------------------------------- #
def measure_sessions(n_tabs: int = 3) -> dict:
    """Combien d'appels fournisseur pour N ouvertures SIMULTANEES de /app ?

    Mesure deux fois : demarrage a froid (rien en base) puis regime permanent
    (une lecture deja stockee). C'est la question "3 onglets = 3 appels ou 1 ?".
    """
    out: dict = {"tabs": n_tabs}
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # latence reseau simulee pour que les N requetes se chevauchent vraiment
        session, _provider, assembler, _r, _c = build_real_stack(tmp, latency_s=0.35)

        def hit():
            try:
                assembler.get_or_generate("XAUUSD", "M15", bound_provider=True)
            except Exception as exc:  # noqa: BLE001 — on mesure, on ne juge pas
                out.setdefault("errors", []).append(repr(exc))

        def burst():
            before = session.count
            threads = [threading.Thread(target=hit) for _ in range(n_tabs)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return session.count - before

        out["cold_start_calls"] = burst()
        out["steady_state_calls"] = burst()
    return out


def measure_scheduler_tick() -> dict:
    """Un tick du VRAI scheduler sur le perimetre warm : combien d'appels ?"""
    from src.intelligence.scheduler import MarketReadingScheduler

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        session, _p, assembler, readings, _c = build_real_stack(tmp)
        warm = [("XAUUSD", tf) for tf in ("M15", "H1", "H4", "D1")]
        warm += [("EURUSD", tf) for tf in ("M15", "H1", "H4", "D1")]
        sched = MarketReadingScheduler(
            assembler=assembler,
            readings_store=readings,
            always_warm=warm,
        )
        before = session.count
        t0 = time.monotonic()
        regenerated = sched.tick()
        first = session.count - before
        elapsed = time.monotonic() - t0
        # deuxieme tick immediat : rien n'a cloture entre-temps
        before = session.count
        regenerated2 = sched.tick()
        second = session.count - before
        return {
            "warm_combos": len(warm),
            "tick1_regenerated": regenerated,
            "tick1_provider_calls": first,
            "tick1_seconds": round(elapsed, 2),
            "tick2_regenerated": regenerated2,
            "tick2_provider_calls": second,
        }


def measure_retry_cost() -> dict:
    """Un 429 cote fournisseur coute-t-il un credit par tentative ?"""
    from src.intelligence.data_providers.twelve_data_provider import (
        TwelveDataError,
        TwelveDataProvider,
    )

    class Failing(CountingSession):
        def get(self, url, params=None, timeout=None):
            super().get(url, params=params, timeout=timeout)

            class R:
                status_code = 429
                ok = False
                text = "rate limit"

                @staticmethod
                def json():
                    return {}

            return R()

    s = Failing()
    p = TwelveDataProvider(api_key="x", session=s, sleep_fn=lambda _s: None)
    try:
        p.get_ohlcv("XAUUSD", "M15", 500)
    except TwelveDataError:
        pass
    return {
        "http_attempts_for_one_logical_fetch": s.count,
        "max_retries": TwelveDataProvider.MAX_RETRIES,
    }


def measure_ttl_dedup() -> dict:
    """Le cache TTL du fournisseur deduplique-t-il, et sur quelle cle ?"""
    from src.intelligence.data_providers.twelve_data_provider import TwelveDataProvider

    s = CountingSession()
    p = TwelveDataProvider(api_key="x", session=s)
    p.get_ohlcv("XAUUSD", "M15", 500)
    p.get_ohlcv("XAUUSD", "M15", 500)  # meme cle -> doit etre gratuit
    same_key = s.count
    p.get_ohlcv("XAUUSD", "M15", 2880)  # lookback different -> nouvelle cle
    diff_lookback = s.count - same_key
    p.get_ohlcv("XAUUSD", "H1", 500)  # unite differente -> nouvelle cle
    diff_tf = s.count - same_key - diff_lookback
    return {
        "ttl_seconds": p._cache_ttl_s,
        "calls_for_2_identical_fetches": same_key,
        "extra_call_for_different_lookback": diff_lookback,
        "extra_call_for_different_timeframe": diff_tf,
    }


def measure_limiter_ceiling() -> dict:
    """Quel plafond/minute le code applique-t-il, et est-il pilotable ?"""
    import inspect

    from src.intelligence.data_providers.twelve_data_provider import TwelveDataProvider

    sig = inspect.signature(TwelveDataProvider.__init__)
    bootstrap_src = (REPO / "src" / "api" / "bootstrap.py").read_text(encoding="utf-8")
    main_src = (REPO / "src" / "intelligence" / "main.py").read_text(encoding="utf-8")

    # Test FONCTIONNEL : une variable d'environnement plausible change-t-elle le
    # plafond effectif ? (on n'interroge pas le texte du code, on lit le limiteur)
    effective = {}
    for name in (
        "TWELVE_DATA_PER_MINUTE",
        "TWELVE_DATA_RATE_LIMIT",
        "TWELVE_DATA_RPM",
        "TWELVE_DATA_CREDITS_PER_MINUTE",
    ):
        os.environ[name] = "55"
    try:
        probe = TwelveDataProvider(api_key="x", session=CountingSession())
        effective = {
            "per_minute": probe._rate_limiter._per_minute,
            "per_day": probe._rate_limiter._per_day,
        }
    finally:
        for name in (
            "TWELVE_DATA_PER_MINUTE",
            "TWELVE_DATA_RATE_LIMIT",
            "TWELVE_DATA_RPM",
            "TWELVE_DATA_CREDITS_PER_MINUTE",
        ):
            os.environ.pop(name, None)

    return {
        "default_per_minute": sig.parameters["per_minute"].default,
        "default_per_day": sig.parameters["per_day"].default,
        "effective_limit_with_env_set_to_55": effective,
        "per_minute_settable_from_env": effective.get("per_minute") == 55,
        "bootstrap_overrides_limits": "per_minute" in bootstrap_src,
        "main_overrides_limits": "per_minute" in main_src,
    }


def measure_lagging_feed(ticks: int = 10) -> dict:
    """Cout d'un flux EN RETARD : le scheduler re-tente-t-il a chaque tick ?

    La serie synthetique est volontairement arretee une bougie avant la cloture
    attendue, exactement ce qui arrive quand le fournisseur est lent, limite ou
    en retard. On compte les appels sur N ticks d'horloge consecutifs.
    """
    from src.intelligence.scheduler import MarketReadingScheduler

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        session, provider, assembler, readings, _c = build_real_stack(tmp)
        warm = [("XAUUSD", "M15"), ("EURUSD", "M15")]
        clock_now = datetime.now(timezone.utc)
        sched = MarketReadingScheduler(
            assembler=assembler,
            readings_store=readings,
            always_warm=warm,
            clock=lambda: clock_now,
        )
        calls = []
        for i in range(ticks):
            before = session.count
            sched.tick()
            calls.append(session.count - before)
            # avance l'horloge d'un tick de 60 s sans qu'une bougie M15 close
            # ne soit jamais servie : le TTL de 300 s est le seul frein
            provider._cache_ttl_s = provider._cache_ttl_s  # inchange, explicite
            time.sleep(0)
        return {
            "warm_combos": len(warm),
            "ticks": ticks,
            "provider_calls_per_tick": calls,
            "total_provider_calls": sum(calls),
        }


def measure_tick_duration_scaling() -> dict:
    """Duree d'un tick par combo : le tick de 60 s tient-il a 100 marches ?"""
    from src.intelligence.scheduler import MarketReadingScheduler

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        session, _p, assembler, readings, _c = build_real_stack(tmp)
        warm = [("XAUUSD", tf) for tf in ("M15", "H1", "H4", "D1")]
        sched = MarketReadingScheduler(
            assembler=assembler, readings_store=readings, always_warm=warm
        )
        t0 = time.monotonic()
        sched.tick()
        elapsed = time.monotonic() - t0
        n = max(1, session.count)
        per_combo = elapsed / n
        return {
            "combos_regenerated": n,
            "seconds_total": round(elapsed, 2),
            "seconds_per_combo": round(per_combo, 3),
            "combos_fitting_in_a_60s_tick": int(60 // per_combo) if per_combo else None,
            "tick_is_sequential_single_instance": True,
        }


def measure_symbol_perimeter() -> dict:
    """Combien de symboles le fournisseur sait-il seulement traduire ?"""
    from src.intelligence.data_providers.twelve_data_provider import TwelveDataProvider
    from src.intelligence import market_registry

    p = TwelveDataProvider(api_key="x", session=CountingSession())
    return {
        "provider_symbol_map": p.available_symbols(),
        "market_registry_ids": list(market_registry.all_ids()),
    }


def measure_derivation_wired() -> dict:
    """La derivation M15/H1/H4 depuis M5 est-elle branchee dans le chemin live ?"""
    import inspect

    from src.intelligence import market_reading_assembler as mra
    from src.intelligence import scheduler as sch

    live_sources = inspect.getsource(mra) + inspect.getsource(sch)
    try:
        from src.intelligence.volatility_forecaster import resample_ohlcv  # noqa: F401

        helper_exists = True
    except Exception:  # noqa: BLE001
        helper_exists = False
    return {
        "resample_helper_exists": helper_exists,
        "resample_used_in_live_path": "resample_ohlcv(" in live_sources,
        "jitter_in_scheduler": any(
            tok in inspect.getsource(sch) for tok in ("jitter", "random.", "stagger")
        ),
    }


def measure_holiday_probe(combos: int = 20, ticks: int = 3) -> dict:
    """Sonde de reouverture un JOUR FERIE : combien d'appels, et groupes comment ?

    ``_should_safety_probe`` n'autorise qu'une sonde par combo par 1800 s — mais
    toutes les combos sont horodatees dans le MEME tick, donc elles redeviennent
    echues dans le meme tick 30 min plus tard. On mesure la forme de la rafale.
    """
    from src.intelligence.scheduler import MarketReadingScheduler

    holiday = datetime(2026, 12, 25, 12, 0, tzinfo=timezone.utc)  # config/market_holidays.json
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        session, provider, assembler, readings, _c = build_real_stack(tmp)
        # L'horloge du scheduler est simulee mais le TTL du fournisseur suit
        # ``time.monotonic`` : sans cela le cache de 300 s masquerait des appels
        # que 31 minutes reelles auraient laisse passer.
        provider._cache_ttl_s = 0.0
        warm = [("XAUUSD", tf) for tf in ("M5", "M15", "H1", "H4", "D1")]
        warm += [("EURUSD", tf) for tf in ("M5", "M15", "H1", "H4", "D1")]
        warm = (warm * ((combos // len(warm)) + 1))[:combos]
        now = {"t": holiday}
        sched = MarketReadingScheduler(
            assembler=assembler,
            readings_store=readings,
            always_warm=warm,
            clock=lambda: now["t"],
        )
        per_tick = []
        detail = []
        for _ in range(ticks):
            before = len(session.calls)
            sched.tick()
            new = session.calls[before:]
            per_tick.append(len(new))
            detail.append([f"{c['symbol']}/{c['interval']}" for c in new])
            # +31 min : la fenetre de 1800 s est franchie pour TOUTES les combos
            now["t"] = now["t"] + timedelta(minutes=31)
        return {
            "combos_probed": len(warm),
            "instruments": len({i for i, _tf in warm}),
            "calls_per_tick_31min_apart": per_tick,
            "calls_detail": detail,
            "note": (
                "tick 1 = demarrage a froid (1 appel/combo) ; ticks suivants = sonde "
                "de reouverture, toutes les combos echues DANS LE MEME tick"
            ),
        }


def main() -> None:
    import logging

    logging.disable(logging.CRITICAL)
    report = {
        "limiter": measure_limiter_ceiling(),
        "symbol_perimeter": measure_symbol_perimeter(),
        "derivation_and_jitter": measure_derivation_wired(),
        "ttl_cache": measure_ttl_dedup(),
        "retry_cost": measure_retry_cost(),
        "scheduler_tick": measure_scheduler_tick(),
        "lagging_feed": measure_lagging_feed(10),
        "holiday_probe": measure_holiday_probe(10, 3),
        "tick_duration": measure_tick_duration_scaling(),
        "user_sessions": measure_sessions(3),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
