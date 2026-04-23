@echo off
REM Smart Sentinel AI — MT5 Live Data Startup Script
REM
REM Usage:
REM   1. Edit this file and fill in your MT5 credentials below
REM   2. Double-click to run, or: scripts\run_mt5_live.bat
REM
REM Prerequisites:
REM   - MetaTrader 5 terminal installed and running
REM   - Demo account logged in
REM   - pip install MetaTrader5 pandas numpy scikit-learn

echo ============================================================
echo   Smart Sentinel AI — MT5 Live Mode
echo ============================================================
echo.

REM === MT5 Credentials (fill these in) ===
set DATA_SOURCE=mt5
set MT5_LOGIN=
set MT5_PASSWORD=
set MT5_SERVER=

REM === Scanning Configuration ===
set SYMBOLS=XAUUSD
set VOL_MODE=hybrid

REM === Narrative Engine ===
set NARRATIVE_MODE=template

REM === API Server ===
set API_PORT=8000

REM === Auth (testing mode = all features unlocked) ===
set SENTINEL_TESTING_MODE=1

REM === Logging ===
set LOG_LEVEL=INFO
set LOG_FORMAT=text

REM === Signal Database ===
set SIGNAL_DB_PATH=./data/signals.db

REM === Check credentials ===
if "%MT5_LOGIN%"=="" (
    echo   ERROR: MT5_LOGIN is not set.
    echo   Edit this file and fill in your MT5 credentials.
    echo   Or run: python scripts/mt5_setup.py
    echo.
    pause
    exit /b 1
)

REM === Verify MT5 connection first ===
echo   Step 1: Verifying MT5 connection...
python scripts/mt5_setup.py --check-only
if errorlevel 1 (
    echo.
    echo   MT5 connection check failed. Fix the issue above and retry.
    pause
    exit /b 1
)

echo.
echo   Step 2: Starting Smart Sentinel AI...
echo   API will be available at: http://localhost:%API_PORT%
echo   Press Ctrl+C to stop.
echo.

python -m src.intelligence.main

pause
