"""Groupe A — corrections hygiène / défensif / observabilité (audit R&D).

Un bloc de tests par finding. Aucun touch aux heuristiques de détection :
ces tests vérifient des garde-fous, du nettoyage et de l'observabilité.

Findings couverts (ajoutés commit par commit) :
- D1-5 : pas de logging.basicConfig à l'import
- D1-4 : plus de __main__ d'entraînement RL legacy
- D1-3 : defaults train/serve unifiés (DEFAULT_SMC_CONFIG)
- D2-9 : divergence RSI désactivable dans le chemin produit
- D4-2 : sanity checks OHLC en entrée
- D4-3 : flag bougies aberrantes (>m×ATR) sans altération
- D4-6 : monitoring firing-rate / NaN-rate
"""

from __future__ import annotations

import inspect
import logging

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n=120, base=1900.0, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    closes = np.zeros(n)
    closes[0] = base
    for i in range(1, n):
        closes[i] = closes[i - 1] + np.random.randn() * 3.0
    opens = closes + np.random.randn(n) * 1.5
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n)) * 2.0
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n)) * 2.0
    vols = np.random.uniform(500, 5000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


# ---------------------------------------------------------------------------
# D1-5 — pas de logging.basicConfig à l'import (effet de bord global)
# ---------------------------------------------------------------------------
class TestD1_5_NoBasicConfigAtImport:
    def test_source_has_no_basicconfig(self):
        import src.environment.strategy_features as sf

        src = inspect.getsource(sf)
        # On cible l'APPEL (avec parenthèse), pas une simple mention en commentaire.
        assert "basicConfig(" not in src, (
            "strategy_features ne doit pas appeler logging.basicConfig() "
            "(reconfigure le root logger du process entier)."
        )

    def test_module_logger_is_named_not_root(self):
        import src.environment.strategy_features as sf

        assert sf.logger.name == "src.environment.strategy_features"
        assert sf.logger is not logging.getLogger()  # pas le root logger

    def test_import_does_not_add_root_handlers(self):
        import importlib

        root = logging.getLogger()
        before = list(root.handlers)
        import src.environment.strategy_features as sf

        importlib.reload(sf)
        assert list(root.handlers) == before, (
            "Recharger le module ne doit ajouter aucun handler au root logger."
        )


# ---------------------------------------------------------------------------
# D1-4 — plus de __main__ d'entraînement RL legacy
# ---------------------------------------------------------------------------
class TestD1_4_NoLegacyRLMain:
    def test_source_has_no_agent_trainer_import(self):
        import src.environment.strategy_features as sf

        src = inspect.getsource(sf)
        # On cible les patterns de CODE (import / appel), pas une mention en commentaire.
        assert "import AgentTrainer" not in src, (
            "Le module détection ne doit plus importer AgentTrainer (RL legacy)."
        )
        assert "train_offline(" not in src, "Plus d'appel d'entraînement RL ici."
        assert "continue_training(" not in src

    def test_only_one_main_guard_remains(self):
        import src.environment.strategy_features as sf

        src = inspect.getsource(sf)
        # Seul le bloc benchmark doit subsister (1 guard exécutable).
        assert src.count('if __name__ ==') == 1

    def test_import_module_is_side_effect_free(self):
        # Importer ne doit pas lancer d'entraînement (sinon l'import lèverait /
        # consommerait du temps). On vérifie juste qu'il s'importe proprement.
        import importlib

        import src.environment.strategy_features as sf

        importlib.reload(sf)
        assert hasattr(sf, "SmartMoneyEngine")
        assert hasattr(sf, "run_benchmark")
