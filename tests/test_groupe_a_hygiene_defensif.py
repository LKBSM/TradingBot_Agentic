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


# ---------------------------------------------------------------------------
# D1-3 — defaults train/serve unifiés (DEFAULT_SMC_CONFIG)
# ---------------------------------------------------------------------------
class TestD1_3_UnifiedDefaults:
    def test_default_config_matches_smcconfig(self):
        from src.environment.strategy_features import DEFAULT_SMC_CONFIG, SMCConfig

        assert DEFAULT_SMC_CONFIG == SMCConfig().model_dump(), (
            "DEFAULT_SMC_CONFIG doit être l'unique source de vérité = SMCConfig()."
        )

    def test_default_config_uses_product_periods_not_legacy_7(self):
        from src.environment.strategy_features import DEFAULT_SMC_CONFIG

        # Le produit sert 14 / 0.1 ; plus de skew 7 / 0.0.
        assert DEFAULT_SMC_CONFIG["RSI_WINDOW"] == 14
        assert DEFAULT_SMC_CONFIG["ATR_WINDOW"] == 14
        assert DEFAULT_SMC_CONFIG["FVG_THRESHOLD"] == 0.1

    def test_no_legacy_hardcoded_default_dicts_in_source(self):
        import src.environment.strategy_features as sf

        src = inspect.getsource(sf)
        # Les anciens defaults divergents (ATR=7 / FVG=0.0) ne doivent plus être
        # codés en dur dans les helpers de preprocessing.
        assert '"ATR_WINDOW": 7' not in src
        assert '"FVG_THRESHOLD": 0.0' not in src

    def test_preprocess_none_config_runs_with_product_defaults(self):
        from src.environment.strategy_features import preprocess_dataframe

        df = _make_ohlcv(120)
        out = preprocess_dataframe(df.copy(), config=None)
        # Pipeline complet OK avec les defaults unifiés (colonnes clés présentes).
        for col in ("RSI", "ATR", "BOS_SIGNAL", "FVG_SIGNAL"):
            assert col in out.columns


# ---------------------------------------------------------------------------
# D3-3 — docstrings perf honnêtes (numba conditionnel) + NUMBA_AVAILABLE exposé
# ---------------------------------------------------------------------------
class TestD3_3_NumbaHonestDocs:
    def test_numba_available_flag_is_bool(self):
        from src.environment.strategy_features import NUMBA_AVAILABLE

        assert isinstance(NUMBA_AVAILABLE, bool)

    def test_class_docstring_states_fallback_caveat(self):
        from src.environment.strategy_features import SmartMoneyEngine

        doc = SmartMoneyEngine.__doc__ or ""
        assert "fallback" in doc.lower()
        assert "NUMBA_AVAILABLE" in doc
        # L'ancienne promesse absolue trompeuse ne doit plus apparaître seule.
        assert "30-100x improvement" not in doc

    def test_timing_report_exposes_path(self):
        from src.environment.strategy_features import SmartMoneyEngine

        eng = SmartMoneyEngine(_make_ohlcv(120), {})
        eng.analyze()
        report = eng.get_timing_report()
        assert isinstance(report, dict)
        assert "total" in report


# ---------------------------------------------------------------------------
# D2-9 — divergence RSI désactivable dans le chemin produit (sans toucher la détection)
# ---------------------------------------------------------------------------
class TestD2_9_DivergenceOptOut:
    def test_default_still_computes_divergence(self):
        from src.environment.strategy_features import SmartMoneyEngine

        out = SmartMoneyEngine(_make_ohlcv(200), {}).analyze()  # default True
        assert "CHOCH_DIVERGENCE" in out.columns, (
            "Le flux legacy (ConfluenceDetector) dépend de CHOCH_DIVERGENCE."
        )

    def test_opt_out_skips_divergence_column(self):
        from src.environment.strategy_features import SmartMoneyEngine

        out = SmartMoneyEngine(_make_ohlcv(200), {}).analyze(compute_divergence=False)
        assert "CHOCH_DIVERGENCE" not in out.columns

    def test_detection_columns_identical_with_or_without_divergence(self):
        """Skipper la divergence ne doit RIEN changer à la détection structurelle."""
        from src.environment.strategy_features import SmartMoneyEngine

        a = SmartMoneyEngine(_make_ohlcv(200, seed=7), {}).analyze(compute_divergence=True)
        b = SmartMoneyEngine(_make_ohlcv(200, seed=7), {}).analyze(compute_divergence=False)
        for col in (
            "BOS_SIGNAL", "BOS_EVENT", "CHOCH_SIGNAL", "BOS_BREAK_LEVEL",
            "FVG_SIGNAL", "OB_STRENGTH_NORM", "BOS_RETEST_ARMED",
        ):
            # NaN-safe equality
            assert a[col].fillna(-999).tolist() == b[col].fillna(-999).tolist(), (
                f"La colonne {col} diffère selon compute_divergence — la détection "
                f"ne doit pas dépendre du calcul de divergence."
            )

    def test_assembler_pipeline_opts_out(self):
        import inspect

        import src.intelligence.market_reading_assembler as asm

        src = inspect.getsource(asm._default_smc_pipeline)
        assert "compute_divergence=False" in src, (
            "Le pipeline produit doit appeler analyze(compute_divergence=False)."
        )


# ---------------------------------------------------------------------------
# D4-2 — sanity checks OHLC en entrée (non-fatals)
# ---------------------------------------------------------------------------
class TestD4_2_OHLCSanityChecks:
    def test_clean_data_reports_no_issue(self):
        from src.environment.strategy_features import SmartMoneyEngine

        eng = SmartMoneyEngine(_make_ohlcv(120), {})
        assert eng.get_data_quality_report() == {}

    def test_high_lt_low_detected_without_raising(self):
        from src.environment.strategy_features import SmartMoneyEngine

        df = _make_ohlcv(120)
        df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["low"] - 5  # high < low
        eng = SmartMoneyEngine(df, {})  # must NOT raise
        rep = eng.get_data_quality_report()
        assert rep.get("high_lt_low", 0) >= 1

    def test_nan_ohlc_detected(self):
        from src.environment.strategy_features import SmartMoneyEngine

        df = _make_ohlcv(120)
        df.iloc[5, df.columns.get_loc("close")] = np.nan
        eng = SmartMoneyEngine(df, {})
        assert eng.get_data_quality_report().get("nan_ohlc_rows", 0) >= 1

    def test_non_positive_price_detected(self):
        from src.environment.strategy_features import SmartMoneyEngine

        df = _make_ohlcv(120)
        df.iloc[7, df.columns.get_loc("low")] = -1.0
        eng = SmartMoneyEngine(df, {})
        assert eng.get_data_quality_report().get("non_positive_price", 0) >= 1

    def test_engine_still_analyzes_dirty_data(self):
        """Données sales = warning, pas crash : analyze() doit aboutir."""
        from src.environment.strategy_features import SmartMoneyEngine

        df = _make_ohlcv(120)
        df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["low"] - 5
        out = SmartMoneyEngine(df, {}).analyze()
        assert "BOS_SIGNAL" in out.columns

    def test_duplicate_timestamps_detected(self):
        from src.environment.strategy_features import SmartMoneyEngine

        df = _make_ohlcv(120)
        idx = df.index.tolist()
        idx[50] = idx[49]  # duplicate timestamp
        df.index = pd.DatetimeIndex(idx)
        eng = SmartMoneyEngine(df, {})
        assert eng.get_data_quality_report().get("duplicate_timestamps", 0) >= 1
