"""DATA-3 — garde-fous sur la consommation de requetes Twelve Data.

Chaque test ici verrouille une fuite de credits MESUREE par l'audit
(``docs/audits/AUDIT-data-3-budget-55.md``) : sans eux, une regression se
verrait uniquement sur la facture, ou par un 429 en production.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.intelligence.data_providers.twelve_data_provider import (
    DEFAULT_PER_DAY,
    DEFAULT_PER_MINUTE,
    TwelveDataProvider,
    TwelveDataRateLimited,
    TwelveDataRateLimiter,
    _derive_provider_symbol,
    credit_purpose,
)
from src.intelligence.scheduler import (
    MarketReadingScheduler,
    spread_offset_minutes,
    stable_slot,
)


# --------------------------------------------------------------------------- #
# Outils
# --------------------------------------------------------------------------- #
def _response(status=200, body=None, headers=None):
    resp = MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 400
    resp.text = ""
    resp.headers = headers or {}
    resp.json.return_value = body if body is not None else {"status": "ok", "values": []}
    return resp


class _Spec:
    def __init__(self, symbol, mtype, provider_symbol=None):
        self.symbol = symbol
        self.type = mtype
        self.provider_symbol = provider_symbol


# =========================================================================== #
# B1 — le plafond du plan doit etre pilotable SANS toucher au code
# =========================================================================== #
class TestPlanLimitsAreConfigurable:
    """Mesure de l'audit : aucune variable d'environnement ne changeait les
    8 credits/min. Payer un plan superieur n'aurait rien debloque — le code se
    serait auto-limite au palier gratuit."""

    def test_defaults_are_the_free_tier(self, monkeypatch):
        monkeypatch.delenv("TWELVE_DATA_PER_MINUTE", raising=False)
        monkeypatch.delenv("TWELVE_DATA_PER_DAY", raising=False)
        limiter = TwelveDataRateLimiter()
        assert limiter._per_minute == DEFAULT_PER_MINUTE == 8
        assert limiter._per_day == DEFAULT_PER_DAY == 800

    def test_env_raises_the_ceiling(self, monkeypatch):
        monkeypatch.setenv("TWELVE_DATA_PER_MINUTE", "55")
        monkeypatch.setenv("TWELVE_DATA_PER_DAY", "100000")
        provider = TwelveDataProvider(api_key="k", session=MagicMock())
        assert provider._rate_limiter._per_minute == 55
        assert provider._rate_limiter._per_day == 100000

    def test_explicit_argument_still_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("TWELVE_DATA_PER_MINUTE", "55")
        provider = TwelveDataProvider(api_key="k", session=MagicMock(), per_minute=8)
        assert provider._rate_limiter._per_minute == 8

    def test_garbage_env_falls_back_instead_of_crashing(self, monkeypatch):
        monkeypatch.setenv("TWELVE_DATA_PER_MINUTE", "beaucoup")
        assert TwelveDataRateLimiter()._per_minute == DEFAULT_PER_MINUTE


# =========================================================================== #
# B5 — un 429 est TERMINAL : il est facture, le re-tenter aggrave le depassement
# =========================================================================== #
class TestRateLimitIsTerminal:
    def test_http_429_costs_exactly_one_request(self):
        session = MagicMock()
        session.get.return_value = _response(429, headers={"Api-Credits-Left": "0"})
        provider = TwelveDataProvider(
            api_key="k", session=session, sleep_fn=lambda _: None
        )
        with pytest.raises(TwelveDataRateLimited):
            provider.get_ohlcv("XAUUSD", "M15", 10)
        assert session.get.call_count == 1

    def test_envelope_429_is_a_rate_limit_not_a_config_error(self):
        """Twelve Data peut signaler le depassement dans le CORPS (HTTP 200).
        Confondre ce cas avec une erreur de configuration masquait la vraie cause."""
        session = MagicMock()
        session.get.return_value = _response(
            200, body={"status": "error", "code": 429, "message": "out of credits"}
        )
        provider = TwelveDataProvider(
            api_key="k", session=session, sleep_fn=lambda _: None
        )
        with pytest.raises(TwelveDataRateLimited):
            provider.get_ohlcv("XAUUSD", "M15", 10)
        assert session.get.call_count == 1

    def test_server_credit_counter_is_adopted(self):
        """Chaque reponse porte le compteur du serveur ; c'est lui qui fait foi."""
        session = MagicMock()
        session.get.return_value = _response(
            200, headers={"Api-Credits-Used": "7", "Api-Credits-Left": "1"}
        )
        provider = TwelveDataProvider(api_key="k", session=session)
        provider.get_ohlcv("XAUUSD", "M15", 10)
        snap = provider.credit_snapshot()
        assert snap["server_minute_used"] == 7
        assert snap["server_minute_left"] == 1

    def test_zero_credits_left_stands_down_without_spending_more(self):
        """Quand le serveur annonce 0 credit restant, on attend : emettre coute un
        credit ET echoue de toute facon."""
        session = MagicMock()
        session.get.return_value = _response(
            200, headers={"Api-Credits-Used": "8", "Api-Credits-Left": "0"}
        )
        provider = TwelveDataProvider(api_key="k", session=session)
        provider.get_ohlcv("XAUUSD", "M15", 10)
        assert provider.credit_snapshot()["cooling_down_s"] > 0

    def test_5xx_is_still_retried(self):
        """Un 5xx n'est pas un depassement de budget : le retry reste legitime."""
        session = MagicMock()
        session.get.side_effect = [
            _response(503),
            _response(200, body={"status": "ok", "values": []}),
        ]
        provider = TwelveDataProvider(
            api_key="k", session=session, sleep_fn=lambda _: None
        )
        provider.get_ohlcv("XAUUSD", "M15", 10)
        assert session.get.call_count == 2


# =========================================================================== #
# B10 — attribution : quel declencheur a depense le budget
# =========================================================================== #
class TestCreditAttribution:
    def test_credits_are_counted_per_trigger(self):
        session = MagicMock()
        session.get.return_value = _response(200)
        provider = TwelveDataProvider(api_key="k", session=session)
        with credit_purpose("scheduler"):
            provider.get_ohlcv("XAUUSD", "M15", 10)
        with credit_purpose("backfill"):
            provider.get_ohlcv("XAUUSD", "H1", 10)
            provider.get_ohlcv("XAUUSD", "H4", 10)
        assert provider.credit_snapshot()["by_purpose"] == {
            "scheduler": 1,
            "backfill": 2,
        }


# =========================================================================== #
# B2 — le perimetre de symboles vient du registre, plus du code
# =========================================================================== #
class TestSymbolPerimeterComesFromTheRegistry:
    def test_registry_markets_are_all_resolvable(self):
        from src.intelligence import market_registry

        provider = TwelveDataProvider(api_key="k", session=MagicMock())
        assert set(provider.available_symbols()) == set(market_registry.all_ids())
        for market in market_registry.all_ids():
            assert provider._map_symbol(market)

    def test_explicit_provider_symbol_wins(self):
        assert _derive_provider_symbol(_Spec("XAUUSD", "metal", "XAU/USD"), "XAUUSD") == "XAU/USD"

    def test_six_letter_pair_is_split_in_the_middle(self):
        assert _derive_provider_symbol(_Spec("BTCUSD", "crypto"), "BTCUSD") == "BTC/USD"
        assert _derive_provider_symbol(_Spec("GBPJPY", "fx"), "GBPJPY") == "GBP/JPY"

    def test_index_ticker_passes_through_untouched(self):
        assert _derive_provider_symbol(_Spec("US500", "index"), "US500") == "US500"

    def test_unknown_market_names_the_supported_ones(self):
        provider = TwelveDataProvider(api_key="k", session=MagicMock())
        with pytest.raises(ValueError, match="Unsupported symbol"):
            provider._map_symbol("NOPE")


# =========================================================================== #
# B3 — etalement : le levier sur la POINTE, qui est ce que le plan facture
# =========================================================================== #
class TestSpreading:
    def test_offset_is_stable_across_calls(self):
        first = spread_offset_minutes("XAUUSD", "H1", 60)
        assert first == spread_offset_minutes("XAUUSD", "H1", 60)

    def test_offset_never_reaches_the_next_close(self):
        """Un decalage >= la periode ferait sauter une bougie entiere."""
        for tf, period in (("M5", 5), ("M15", 15), ("H1", 60), ("H4", 240), ("D1", 1440)):
            for i in range(50):
                assert 0 <= spread_offset_minutes(f"MKT{i}", tf, period) < period

    def test_daily_candle_is_never_held_back_for_hours(self):
        assert spread_offset_minutes("XAUUSD", "D1", 1440) <= 60

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("SENTINEL_SPREAD_FRACTION", "0")
        assert spread_offset_minutes("XAUUSD", "H1", 60) == 0

    def test_hundred_markets_fit_under_a_55_credit_minute(self):
        """Le chiffre qui decide du plan. Sans etalement, les 4 unites du
        perimetre warm tirent ensemble a minuit : 400 credits dans la minute."""
        markets = [f"MKT{i:03d}" for i in range(100)]
        units = {"M15": 15, "H1": 60, "H4": 240, "D1": 1440}
        per_minute: dict[int, int] = {}
        for market in markets:
            for tf, period in units.items():
                # minuit : les 4 unites cloturent ensemble, pire minute de la journee
                slot = spread_offset_minutes(market, tf, period)
                per_minute[slot] = per_minute.get(slot, 0) + 1
        assert max(per_minute.values()) <= 55

    def test_stable_slot_is_uniform_enough(self):
        counts: dict[int, int] = {}
        for i in range(600):
            slot = stable_slot(f"seed-{i}", 10)
            counts[slot] = counts.get(slot, 0) + 1
        assert len(counts) == 10
        # aucun creneau ne doit concentrer plus du double de la part moyenne
        assert max(counts.values()) < 2 * (600 / 10)


# =========================================================================== #
# B3/B7 — comportement du scheduler : creneau respecte, rien perdu
# =========================================================================== #
class _Store:
    def __init__(self, active=(), readings=None):
        self._active = list(active)
        self._readings = readings or {}
        self.active_calls = []

    def get_active_combinations(self, since):
        self.active_calls.append(since)
        return list(self._active)

    def get_latest_reading(self, instrument, timeframe):
        return self._readings.get((instrument, timeframe))

    def mark_combination_active(self, instrument, timeframe):
        pass


class _Assembler:
    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def get_or_generate(self, instrument, timeframe, bound_provider=True):
        with self._lock:
            self.calls.append((instrument, timeframe))
        return {}

    def refresh_if_reopened(self, instrument, timeframe):
        return False


def _payload(ts):
    return {"header": {"candle_close_ts": ts.isoformat()}, "_logic_version": 7}


class TestSchedulerSpendsLess:
    def test_combo_waits_for_its_slot_then_regenerates(self):
        """Rien n'est perdu : la combinaison est seulement decalee dans la fenetre."""
        stale = datetime(2026, 5, 28, 8, 0, tzinfo=timezone.utc)
        combo = ("XAUUSD", "H1")
        offset = spread_offset_minutes(*combo, 60)
        if offset == 0:  # pragma: no cover — ce marche tombe sur le creneau 0
            pytest.skip("ce couple tombe sur le creneau 0, rien a retarder")

        close = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        store = _Store(active=[combo], readings={combo: _payload(stale)})
        assembler = _Assembler()

        before = close + timedelta(minutes=offset - 1)
        sched = MarketReadingScheduler(assembler, store, clock=lambda: before)
        assert sched.tick() == 0, "la combinaison doit attendre son creneau"

        after = close + timedelta(minutes=offset)
        sched_after = MarketReadingScheduler(assembler, store, clock=lambda: after)
        assert sched_after.tick() == 1, "et regenerer une fois son creneau atteint"

    def test_on_demand_combo_stops_costing_credits_once_nobody_looks(self):
        """Un coup d'oeil sur M5 achetait 24 h de rafraichissements (~96 requetes
        par marche et par jour) pour un ecran que plus personne ne regarde."""
        now = datetime(2026, 5, 28, 14, 23, tzinfo=timezone.utc)
        store = _Store(active=[("XAUUSD", "M5")])
        sched = MarketReadingScheduler(
            _Assembler(), store, clock=lambda: now, auto_stop_hours=24
        )
        sched.tick()
        assert store.active_calls[0] == now - timedelta(hours=24)
        assert store.active_calls[1] == now - timedelta(
            hours=MarketReadingScheduler.DEFAULT_ON_DEMAND_ACTIVE_HOURS
        )

    def test_warm_perimeter_is_never_subject_to_the_short_window(self):
        """Le perimetre warm est le produit, pas un artefact d'acces : il ne doit
        jamais disparaitre parce que personne n'a ouvert la page."""
        now = datetime(2026, 5, 28, 14, 23, tzinfo=timezone.utc)
        store = _Store(active=[])  # aucun acces recent
        assembler = _Assembler()
        sched = MarketReadingScheduler(
            assembler,
            store,
            always_warm=[("XAUUSD", "M15")],
            clock=lambda: now,
            spread=False,
        )
        sched.tick()
        assert assembler.calls == [("XAUUSD", "M15")]

    def test_holiday_probes_do_not_all_come_due_in_the_same_tick(self):
        """Mesure de l'audit : toutes les combinaisons etaient horodatees dans le
        meme tick, donc toutes re-echues dans le meme tick 30 min plus tard —
        100 a 200 credits dans une minute a 100 marches, marche FERME."""
        sched = MarketReadingScheduler(
            _Assembler(), _Store(), clock=lambda: datetime.now(timezone.utc)
        )
        seeds = {
            sched._last_safety_probe.get(("XAUUSD", tf))
            for tf in ("M5", "M15", "H1", "H4", "D1")
        }
        # premiere vue : chaque combinaison est semee, aucune n'est sondee
        base = datetime(2026, 12, 25, 12, 0, tzinfo=timezone.utc)
        sched2 = MarketReadingScheduler(_Assembler(), _Store(), clock=lambda: base)
        for tf in ("M5", "M15", "H1", "H4", "D1"):
            assert sched2._should_safety_probe("XAUUSD", tf, base) is False
        stamps = {
            sched2._last_safety_probe[("XAUUSD", tf)]
            for tf in ("M5", "M15", "H1", "H4", "D1")
        }
        assert len(stamps) > 1, "les combinaisons doivent etre semees a des instants differents"
        assert seeds == {None}


# =========================================================================== #
# B6 — single-flight : N onglets sur une combinaison froide = UN credit
# =========================================================================== #
class TestSingleFlight:
    def test_concurrent_cold_start_spends_one_credit(self, tmp_path):
        from src.intelligence.market_reading_assembler import MarketReadingAssembler

        calls: list = []
        lock = threading.Lock()

        class _SlowProvider:
            def fetch_candles(self, instrument, timeframe, count):
                import time

                with lock:
                    calls.append((instrument, timeframe))
                time.sleep(0.2)  # laisse les autres fils arriver pendant l'appel
                base = datetime(2026, 5, 28, 0, 0, tzinfo=timezone.utc)
                return [
                    type(
                        "C",
                        (),
                        {
                            "ts": base + timedelta(minutes=15 * i),
                            "open": 1.0,
                            "high": 1.5,
                            "low": 0.5,
                            "close": 1.2,
                            "volume": 0.0,
                        },
                    )()
                    for i in range(200)
                ]

            def get_ohlcv(self, *a, **k):  # pragma: no cover — interface
                raise NotImplementedError

        class _Readings:
            def __init__(self):
                self.rows = {}
                self.lock = threading.Lock()

            def get_latest_reading(self, i, tf):
                with self.lock:
                    return self.rows.get((i, tf))

            def save_reading(self, i, tf, ts, payload):
                with self.lock:
                    self.rows[(i, tf)] = payload
                return 1

            def mark_combination_active(self, i, tf):
                pass

        class _Candles:
            def get_last_n_candles(self, *a, **k):
                return []

            def upsert_candles(self, *a, **k):
                return 0

        assembler = MarketReadingAssembler(
            data_provider=_SlowProvider(),
            readings_store=_Readings(),
            candles_store=_Candles(),
            news_pipeline=None,
        )

        errors: list = []

        def hit():
            try:
                assembler.get_or_generate("XAUUSD", "M15", bound_provider=False)
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        threads = [threading.Thread(target=hit) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert len(calls) == 1, f"3 onglets ont depense {len(calls)} credits au lieu d'un"

    def test_memo_does_not_suppress_the_next_tick(self, monkeypatch):
        """Le memo single-flight ne doit PAS empecher un rattrapage.

        S'il durait toute la bougie, une combinaison dont le flux a brievement
        pris du retard ne pourrait plus se rattraper avant la bougie suivante :
        on aurait echange un credit contre de la peremption silencieuse.
        """
        from src.intelligence import market_reading_assembler as mra

        calls: list = []

        class _Provider:
            def fetch_candles(self, instrument, timeframe, count):
                calls.append((instrument, timeframe))
                base = datetime(2026, 5, 28, 0, 0, tzinfo=timezone.utc)
                return [
                    type("C", (), {"ts": base + timedelta(minutes=15 * i), "open": 1.0,
                                   "high": 1.5, "low": 0.5, "close": 1.2, "volume": 0.0})()
                    for i in range(200)
                ]

            def get_ohlcv(self, *a, **k):  # pragma: no cover — interface
                raise NotImplementedError

        class _Readings:
            def get_latest_reading(self, i, tf):
                return None  # le flux est en retard : le stock ne rattrape jamais

            def save_reading(self, i, tf, ts, payload):
                return 1

            def mark_combination_active(self, i, tf):
                pass

        class _Candles:
            def get_last_n_candles(self, *a, **k):
                return []

            def upsert_candles(self, *a, **k):
                return 0

        assembler = mra.MarketReadingAssembler(
            data_provider=_Provider(),
            readings_store=_Readings(),
            candles_store=_Candles(),
            news_pipeline=None,
        )
        assembler.get_or_generate("XAUUSD", "M15", bound_provider=False)
        assert len(calls) == 1

        # memo encore chaud : un appel immediat ne redemande rien
        assembler.get_or_generate("XAUUSD", "M15", bound_provider=False)
        assert len(calls) == 1

        # memo expire (le tick suivant arrive une minute plus tard) : on re-tente
        monkeypatch.setattr(mra, "_BUILD_MEMO_TTL_S", 0.0)
        assembler.get_or_generate("XAUUSD", "M15", bound_provider=False)
        assert len(calls) == 2


# =========================================================================== #
# B9 — l'image deployee ne doit pas rallumer le scanner herite
# =========================================================================== #
class TestDeployedEntrypoint:
    @pytest.mark.parametrize("path", ["Dockerfile", "infrastructure/Dockerfile"])
    def test_docker_serves_the_v2_entrypoint(self, path):
        """``src.intelligence.main`` demarre en plus le scanner herite, qui
        interroge Twelve Data toutes les 60 s par symbole — ~288 credits/jour/marche
        pour une donnee que le produit V2 ne lit jamais."""
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        content = open(os.path.join(repo, path), encoding="utf-8").read()
        runtime_cmds = [
            line
            for line in content.splitlines()
            if line.startswith("CMD [") and "pytest" not in line
        ]
        assert runtime_cmds, f"{path}: aucune commande de demarrage"
        for cmd in runtime_cmds:
            assert "src.intelligence.main" not in cmd
            assert "src.api.asgi" in cmd


# =========================================================================== #
# Le registre reste la source unique
# =========================================================================== #
class TestRegistryStaysTheSingleSource:
    def test_provider_symbol_is_declared_for_every_market(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo, "config", "markets.json"), encoding="utf-8") as fh:
            raw = json.load(fh)
        assert "providerSymbol" in raw["_fields"], "le champ doit rester documente"
        for market in raw["markets"]:
            assert "/" in market.get("providerSymbol", ""), (
                f"{market['id']}: donner le ticker Twelve Data exact, ou retirer le "
                "champ pour laisser la derivation faire le travail"
            )
