"""Tests for scripts/seed_twelve_data.py (Chantier 1 — Étape 5)."""

from __future__ import annotations

import pytest

from scripts import seed_twelve_data


class TestDryRun:
    def test_iterates_default_6_combos(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("CANDLES_DB_PATH", str(tmp_path / "candles.db"))
        monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)
        rc = seed_twelve_data.main(["--dry-run"])
        out = capsys.readouterr().out
        assert rc == 0
        # 6 combos = 2 instruments × 3 timeframes
        for inst in ["XAUUSD", "EURUSD"]:
            for tf in ["M15", "H1", "H4"]:
                assert f"{inst} {tf}: dry-run" in out
        assert "DONE: 6 combinations" in out

    def test_dry_run_does_not_require_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CANDLES_DB_PATH", str(tmp_path / "candles.db"))
        monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)
        # Should not raise / not exit 1
        assert seed_twelve_data.main(["--dry-run"]) == 0


class TestCliArgs:
    def test_single_instrument_single_timeframe(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("CANDLES_DB_PATH", str(tmp_path / "candles.db"))
        rc = seed_twelve_data.main(
            ["--dry-run", "--instrument", "XAUUSD", "--timeframe", "M15"]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "XAUUSD M15: dry-run" in out
        assert "EURUSD" not in out
        assert " H1:" not in out and " H4:" not in out
        assert "DONE: 1 combinations" in out

    def test_lookback_arg_parsed(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("CANDLES_DB_PATH", str(tmp_path / "candles.db"))
        rc = seed_twelve_data.main(
            ["--dry-run", "--instrument", "XAUUSD", "--timeframe", "M15", "--lookback", "50"]
        )
        # Dry-run doesn't actually fetch, but argparse must not reject the flag
        assert rc == 0


class TestExitOnMissingApiKey:
    def test_exits_1_when_no_api_key_and_not_dry_run(self, monkeypatch, capsys):
        monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)
        rc = seed_twelve_data.main([])
        err = capsys.readouterr().err
        assert rc == 1
        assert "TWELVE_DATA_API_KEY" in err

    def test_no_exit_when_dry_run_and_no_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CANDLES_DB_PATH", str(tmp_path / "candles.db"))
        monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)
        assert seed_twelve_data.main(["--dry-run"]) == 0
