"""DATA-4 — garde-fous du perimetre de marches.

Le registre est passe de 2 a 80 marches. Ces tests verrouillent ce qui rend un
marche REELLEMENT servi, plutot que simplement declare : un symbole que le
fournisseur resout, une classe d'actif qui donne les bonnes heures de seance et
un prereglage de volatilite, une regle de rattachement des actualites, et une
consommation de credits qui tient dans le forfait.

Sans eux, ajouter un marche produit une page vide sans message utile — le mode
de defaillance que l'audit DATA-3 a mesure.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.intelligence import market_registry as mr
from src.intelligence.data_providers.twelve_data_provider import _symbol_map
from src.intelligence.market_calendar import OPEN, calendar_state, hours_for

REPO = Path(__file__).resolve().parents[1]


# =========================================================================== #
# Un marche declare doit etre un marche SERVI
# =========================================================================== #
class TestEveryMarketIsActuallyServable:
    def test_every_market_resolves_to_a_provider_symbol(self):
        mapping = _symbol_map()
        missing = [m for m in mr.all_ids() if not mapping.get(m)]
        assert not missing, f"sans symbole fournisseur : {missing}"

    def test_provider_symbols_are_unique(self):
        """Deux marches sur le meme ticker seraient le meme marche affiche deux
        fois — et deux fois le cout en credits."""
        mapping = _symbol_map()
        seen: dict = {}
        for market, symbol in mapping.items():
            assert symbol not in seen, f"{market} et {seen[symbol]} partagent {symbol}"
            seen[symbol] = market

    def test_every_market_has_a_volatility_preset(self):
        """L'invariant : registre inclus dans prereglages. Un marche servi sans
        configuration de prevision n'aurait aucune volatilite a annoncer."""
        from src.intelligence.volatility_forecaster import get_instrument_registry

        presets = set(get_instrument_registry())
        missing = sorted(set(mr.all_ids()) - presets)
        assert not missing, f"sans prereglage de volatilite : {missing}"

    def test_every_market_has_a_news_attachment_rule(self):
        """Sans devise motrice, le calendrier d'un marche est vide — la page
        existe mais ne dit rien."""
        missing = [m for m in mr.all_ids() if not mr.driver_currencies(m)]
        assert not missing, f"sans devise motrice : {missing}"

    def test_news_drivers_are_plausible_currencies(self):
        for market in mr.all_ids():
            for currency in mr.driver_currencies(market):
                assert len(currency) == 3 and currency.isalpha(), (
                    f"{market} : devise motrice invalide {currency!r}"
                )


# =========================================================================== #
# Les heures de seance viennent de la CLASSE, pas d'une table recopiee
# =========================================================================== #
class TestTradingHoursDeriveFromAssetClass:
    #: un samedi : le FX et les metaux sont fermes, la crypto cote
    SATURDAY = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)

    def test_crypto_trades_at_the_weekend(self):
        """Avant DATA-4 la table ne listait que BTC et ETH : toute autre crypto
        etait traitee comme une paire FX et gelee tout le week-end."""
        cryptos = [s.id for s in mr.all_specs() if s.type == "crypto"]
        assert len(cryptos) > 2, "ce test n'a de sens qu'avec plus que BTC/ETH"
        for market in cryptos:
            assert hours_for(market).always_open, f"{market} devrait coter 24/7"
            assert calendar_state(market, self.SATURDAY) == OPEN

    def test_metals_keep_their_rollover_pause(self):
        """Idem : seuls l'or et l'argent l'avaient ; le platine ne l'aurait pas eu."""
        metals = [s.id for s in mr.all_specs() if s.type == "metal"]
        for market in metals:
            assert hours_for(market).daily_break is not None, (
                f"{market} : pause de rollover manquante"
            )

    def test_forex_is_closed_at_the_weekend(self):
        fx = [s.id for s in mr.all_specs() if s.type == "fx"]
        for market in fx:
            assert not hours_for(market).always_open
            assert calendar_state(market, self.SATURDAY) != OPEN

    def test_unknown_symbol_stays_conservative(self):
        """Un symbole hors registre est suppose FERME le week-end, jamais 24/7 :
        annoncer un marche ouvert a tort est le pire des deux."""
        assert not hours_for("NOTAMARKET").always_open


# =========================================================================== #
# Le budget de credits doit tenir dans le forfait
# =========================================================================== #
class TestCreditBudgetHolds:
    def test_live_perimeter_fits_the_grow_plan(self):
        """Le chiffre qui decide du forfait : la POINTE par minute, calculee avec
        les decalages REELS du scheduler sur les marches REELLEMENT au registre."""
        import collections

        from src.intelligence.lookback_config import live_warm_combos
        from src.intelligence.scheduler import spread_offset_minutes
        from src.intelligence.timeframe_registry import minutes_map

        minutes = minutes_map()
        per_minute: collections.Counter = collections.Counter()
        for instrument, timeframe in live_warm_combos():
            period = minutes[timeframe]
            offset = spread_offset_minutes(instrument, timeframe, period)
            for close_min in range(0, 1440, period):
                per_minute[(close_min + offset) % 1440] += 1

        peak = max(per_minute.values())
        # 55 = forfait Grow. La cible de securite de l'audit est 40/min soutenus,
        # les 15 restants absorbant les rafales non etalees (demarrage a froid,
        # sonde jour ferie, re-tentatives).
        assert peak <= 40, (
            f"pointe {peak} credits/min : au-dessus de la cible de securite de 40. "
            "Reduire le perimetre warm, ou re-mesurer avant d'ajouter des marches."
        )

    def test_daily_total_is_reported_not_capped(self):
        """Twelve Data facture au pic/minute ; le total/jour est illimite sur un
        plan paye. On le mesure quand meme, pour qu'une derive soit visible."""
        from src.intelligence.lookback_config import live_warm_combos
        from src.intelligence.timeframe_registry import minutes_map

        minutes = minutes_map()
        total = sum(1440 // minutes[tf] for _i, tf in live_warm_combos())
        assert total > 0
        # garde-fou large : un ordre de grandeur au-dessus signalerait que M1 ou
        # M5 est repasse dans le perimetre warm sans qu'on l'ait decide.
        assert total < 100_000, f"{total} requetes/jour : perimetre warm inattendu"


# =========================================================================== #
# Ce qui est injecte dans CHAQUE message de M.I.A doit rester borne
# =========================================================================== #
class TestPromptStaysBounded:
    def test_signal_summary_is_capped(self):
        """Ce bloc part dans le prompt systeme a chaque message : son cout est paye
        par tour. A 2 marches il faisait 10 combinaisons ; a 80 il en ferait 400."""
        from src.intelligence.chatbot.signal_summary_provider import (
            DEFAULT_MAX_COMBOS,
            SignalSummaryProvider,
        )

        class _Assembler:
            def get_or_generate(self, instrument, timeframe, **kw):
                raise RuntimeError("aucune lecture : on ne teste que le cadrage")

        provider = SignalSummaryProvider(_Assembler())
        assert len(provider._combinations) <= DEFAULT_MAX_COMBOS

    def test_cap_keeps_the_founding_markets_first(self):
        """Quand le plafond mord, ce sont les marches qui ouvrent la colonne qui
        restent en ligne — pas un sous-ensemble arbitraire."""
        from src.intelligence.chatbot.signal_summary_provider import SignalSummaryProvider

        class _Assembler:
            def get_or_generate(self, instrument, timeframe, **kw):
                raise RuntimeError("non utilise")

        combos = SignalSummaryProvider(_Assembler())._combinations
        assert combos[0][0] == mr.all_ids()[0]


# =========================================================================== #
# Le registre reste la source unique
# =========================================================================== #
class TestRegistryIsTheSingleSource:
    def test_frontend_module_matches_the_registry(self):
        """``markets.generated.ts`` est genere depuis le JSON : s'il a derive,
        c'est qu'on a oublie `node scripts/gen_markets.mjs`."""
        generated = (REPO / "webapp" / "lib" / "markets.generated.ts").read_text(
            encoding="utf-8"
        )
        for market in mr.all_ids():
            assert f'id: "{market}"' in generated, (
                f"{market} absent du module frontend — regenerer avec "
                "`node scripts/gen_markets.mjs`"
            )

    def test_catalogue_ux_is_not_the_product_perimeter(self):
        """Le catalogue d'affichage reste distinct du registre : il sert a tester
        la tenue de l'interface, il ne decide pas de ce que le moteur suit."""
        catalogue = json.loads(
            (REPO / "config" / "market_catalog_ux_test.json").read_text(encoding="utf-8")
        )
        assert "_comment" in catalogue
        for market in catalogue["markets"]:
            assert "timeframes" not in market, (
                f"{market['id']} : le catalogue d'affichage ne doit porter ni unite "
                "de temps ni precision de prix (cf. _no_data_rule)"
            )

    @pytest.mark.parametrize("field", ["id", "label", "symbol", "type", "priceDecimals", "glyph"])
    def test_every_registry_entry_is_complete(self, field):
        raw = json.loads((REPO / "config" / "markets.json").read_text(encoding="utf-8"))
        for market in raw["markets"]:
            assert market.get(field) not in (None, ""), f"{market.get('id')}: {field} manquant"
