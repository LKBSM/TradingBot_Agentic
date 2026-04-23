"""MT5 Setup & Verification Script for Smart Sentinel AI.

Run this script to:
  1. Detect MetaTrader 5 installation
  2. Check if the terminal is running
  3. Connect to MT5 (auto-detects logged-in account or uses env vars)
  4. Test data feed for supported symbols
  5. Report data quality (gaps, format, recency)

Usage:
    python scripts/mt5_setup.py                    # Auto-detect + test all symbols
    python scripts/mt5_setup.py --symbols XAUUSD   # Test specific symbol
    python scripts/mt5_setup.py --check-only        # Just check connection, no data test

Environment variables (optional):
    MT5_LOGIN    — Account number
    MT5_PASSWORD — Account password
    MT5_SERVER   — Broker server name
    MT5_PATH     — Path to terminal64.exe (auto-detected if not set)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Default symbols to test
DEFAULT_SYMBOLS = ["XAUUSD", "EURUSD", "BTCUSD", "US500", "GBPUSD", "USDJPY"]

# Timeframes to verify
TIMEFRAMES = ["M1", "M5", "M15", "H1", "H4", "D1"]


def find_mt5_terminal() -> Optional[str]:
    """Search common installation paths for terminal64.exe."""
    candidates = [
        os.environ.get("MT5_PATH", ""),
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        r"C:\Program Files\FTMO MetaTrader 5\terminal64.exe",
    ]

    # Also check user's home for portable installs
    home = Path.home()
    candidates.extend([
        str(home / "AppData" / "Roaming" / "MetaQuotes" / "Terminal"),
        str(home / "Desktop" / "MetaTrader 5" / "terminal64.exe"),
    ])

    for path in candidates:
        if path and Path(path).exists():
            return path

    # Try globbing Program Files for any MT5 install
    for base in [r"C:\Program Files", r"C:\Program Files (x86)"]:
        base_path = Path(base)
        if base_path.exists():
            for d in base_path.iterdir():
                if "metatrader" in d.name.lower() and d.is_dir():
                    exe = d / "terminal64.exe"
                    if exe.exists():
                        return str(exe)

    return None


def is_mt5_running() -> bool:
    """Check if MT5 terminal process is running."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq terminal64.exe"],
            capture_output=True, text=True, timeout=5,
        )
        return "terminal64.exe" in result.stdout
    except Exception:
        return False


def launch_mt5(path: str) -> bool:
    """Launch MT5 terminal and wait for it to start."""
    print(f"  Launching MT5 terminal: {path}")
    try:
        subprocess.Popen([path], shell=False)
        # Wait for terminal to initialize
        for i in range(15):
            time.sleep(2)
            if is_mt5_running():
                print("  MT5 terminal is now running.")
                return True
            print(f"  Waiting for MT5 to start... ({(i+1)*2}s)")
        print("  WARNING: MT5 terminal did not start within 30 seconds.")
        return False
    except Exception as e:
        print(f"  ERROR: Failed to launch MT5: {e}")
        return False


def connect_mt5(
    login: Optional[int] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
    path: Optional[str] = None,
) -> Tuple[bool, Any]:
    """Connect to MT5 terminal. Returns (success, mt5_module)."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("  ERROR: MetaTrader5 package not installed.")
        print("  Fix: pip install MetaTrader5")
        return False, None

    kwargs: Dict[str, Any] = {}
    if login is not None:
        kwargs["login"] = login
    if password is not None:
        kwargs["password"] = password
    if server is not None:
        kwargs["server"] = server
    if path is not None:
        kwargs["path"] = path

    if not mt5.initialize(**kwargs):
        error = mt5.last_error()
        error_code = error[0] if error else "unknown"
        error_msg = error[1] if error and len(error) > 1 else "unknown"
        print(f"  ERROR: MT5 initialization failed (code={error_code}): {error_msg}")

        if error_code == -6:
            print()
            print("  This means the MT5 terminal has no account logged in.")
            print("  To fix this:")
            print("    1. Open the MT5 terminal window")
            print("    2. File > Open an Account")
            print("    3. Search for a broker (e.g., 'MetaQuotes', 'ICMarkets', 'Exness')")
            print("    4. Select 'Open a demo account'")
            print("    5. Fill in details and click 'Finish'")
            print("    6. Re-run this script after logging in")
        elif error_code == -2:
            print()
            print("  This means the MT5 terminal is not running.")
            print("  The script will try to launch it...")

        return False, mt5

    return True, mt5


def print_account_info(mt5: Any) -> None:
    """Print connected account details."""
    info = mt5.account_info()
    if info is None:
        print("  WARNING: Could not retrieve account info.")
        return

    print(f"  Account: {info.login}")
    print(f"  Name:    {info.name}")
    print(f"  Server:  {info.server}")
    print(f"  Balance: {info.balance:.2f} {info.currency}")
    print(f"  Company: {info.company}")
    print(f"  Mode:    {'Demo' if info.trade_mode == 0 else 'Live' if info.trade_mode == 2 else 'Contest'}")


def print_terminal_info(mt5: Any) -> None:
    """Print terminal details."""
    info = mt5.terminal_info()
    if info is None:
        return

    print(f"  Terminal: {info.name}")
    print(f"  Build:    {info.build}")
    print(f"  Company:  {info.company}")
    print(f"  Path:     {info.path}")
    print(f"  Data dir: {info.data_path}")
    print(f"  Trade OK: {bool(info.trade_allowed)}")


def test_symbol(mt5: Any, symbol: str, timeframe: str = "M15", bars: int = 200) -> Dict[str, Any]:
    """Test data feed for a single symbol + timeframe.

    Returns dict with test results.
    """
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
    }

    result: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "available": False,
        "bars_received": 0,
        "latest_bar": None,
        "bar_age_minutes": None,
        "has_gaps": False,
        "price_range": None,
        "error": None,
    }

    # Check if symbol exists
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        result["error"] = "Symbol not found on this broker"
        return result

    # Enable symbol if needed
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            result["error"] = "Symbol exists but could not be enabled"
            return result

    result["available"] = True

    # Fetch data
    mt5_tf = tf_map.get(timeframe)
    if mt5_tf is None:
        result["error"] = f"Unknown timeframe: {timeframe}"
        return result

    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        result["error"] = f"No data returned: {error}"
        return result

    import pandas as pd
    import numpy as np

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    result["bars_received"] = len(df)
    result["latest_bar"] = str(df["time"].iloc[-1])

    # Check data recency
    now = datetime.utcnow()
    latest = df["time"].iloc[-1].to_pydatetime()
    age = now - latest
    result["bar_age_minutes"] = round(age.total_seconds() / 60, 1)

    # Check for gaps (expected interval for the timeframe)
    tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
    expected_interval = timedelta(minutes=tf_minutes.get(timeframe, 15))
    diffs = df["time"].diff().dropna()
    # Allow 2x expected interval (weekends, holidays will still flag)
    gaps = diffs[diffs > expected_interval * 3]
    result["has_gaps"] = len(gaps) > 0
    if len(gaps) > 0:
        result["gap_count"] = len(gaps)

    # Price range
    result["price_range"] = f"{df['low'].min():.2f} - {df['high'].max():.2f}"

    # Volume check
    zero_vol = (df["tick_volume"] == 0).sum()
    if zero_vol > len(df) * 0.1:
        result["warning"] = f"{zero_vol} bars with zero volume ({zero_vol/len(df)*100:.0f}%)"

    return result


def test_all_symbols(mt5: Any, symbols: List[str]) -> None:
    """Test data feed for all symbols."""
    print()
    print("=" * 70)
    print("  DATA FEED VERIFICATION")
    print("=" * 70)

    passed = 0
    failed = 0
    warnings = 0

    for symbol in symbols:
        result = test_symbol(mt5, symbol, "M15", 200)
        status = "PASS" if result["available"] and result["error"] is None else "FAIL"

        if status == "PASS":
            passed += 1
            icon = "OK"
        else:
            failed += 1
            icon = "XX"

        print(f"\n  [{icon}] {symbol}")

        if result["error"]:
            print(f"       Error: {result['error']}")
            continue

        print(f"       Bars: {result['bars_received']}")
        print(f"       Latest: {result['latest_bar']}")
        print(f"       Age: {result['bar_age_minutes']} min")
        print(f"       Range: {result['price_range']}")

        if result.get("has_gaps"):
            print(f"       WARNING: {result.get('gap_count', '?')} gaps in data (weekends/holidays expected)")
            warnings += 1

        if result.get("warning"):
            print(f"       WARNING: {result['warning']}")
            warnings += 1

    print()
    print("-" * 70)
    print(f"  Results: {passed} passed, {failed} failed, {warnings} warnings")
    print("-" * 70)


def test_multi_timeframe(mt5: Any, symbol: str) -> None:
    """Test multiple timeframes for a single symbol."""
    print(f"\n  Multi-timeframe test for {symbol}:")
    for tf in TIMEFRAMES:
        result = test_symbol(mt5, symbol, tf, 50)
        status = "OK" if result["available"] and not result.get("error") else "FAIL"
        bars = result.get("bars_received", 0)
        error = result.get("error", "")
        if status == "OK":
            print(f"    {tf:>4s}: {bars:>4d} bars | Latest: {result.get('latest_bar', 'N/A')}")
        else:
            print(f"    {tf:>4s}: FAIL | {error}")


def write_env_template(account_info: Any = None) -> None:
    """Write .env.example if it doesn't exist."""
    env_path = Path(__file__).parent.parent / ".env.example"
    if env_path.exists():
        return

    login = getattr(account_info, "login", "") if account_info else ""
    server = getattr(account_info, "server", "") if account_info else ""

    content = f"""# Smart Sentinel AI — Environment Configuration
# Copy this file to .env and fill in your values.

# === Data Source ===
DATA_SOURCE=mt5              # csv or mt5
DATA_DIR=./data              # Path to CSV files (when DATA_SOURCE=csv)

# === MT5 Connection (when DATA_SOURCE=mt5) ===
MT5_LOGIN={login}
MT5_PASSWORD=
MT5_SERVER={server}
# MT5_PATH=                  # Auto-detected; override if needed

# === Symbols to scan ===
SYMBOLS=XAUUSD               # Comma-separated: XAUUSD,EURUSD,BTCUSD

# === Volatility Forecaster ===
VOL_MODE=hybrid              # har, lgbm, or hybrid

# === Narrative Engine ===
NARRATIVE_MODE=template      # template ($0) or llm (uses Claude API)
# ANTHROPIC_API_KEY=         # Required only when NARRATIVE_MODE=llm

# === Telegram Delivery (optional) ===
# TELEGRAM_BOT_TOKEN=
# TELEGRAM_CHAT_ID=

# === API ===
API_PORT=8000

# === Auth ===
SENTINEL_TESTING_MODE=1      # 1=all features unlocked (for personal testing)

# === Logging ===
LOG_LEVEL=INFO
LOG_FORMAT=text              # text or json

# === Economic Calendar (optional) ===
# CALENDAR_PATH=./data/calendar.csv

# === Signal Database ===
SIGNAL_DB_PATH=./data/signals.db
"""
    env_path.write_text(content)
    print(f"\n  Created: {env_path}")


def main():
    parser = argparse.ArgumentParser(description="Smart Sentinel AI — MT5 Setup & Verification")
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbols to test (default: all)")
    parser.add_argument("--check-only", action="store_true", help="Just check connection, skip data tests")
    parser.add_argument("--multi-tf", action="store_true", help="Test multiple timeframes per symbol")
    parser.add_argument("--launch", action="store_true", help="Launch MT5 terminal if not running")
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS

    print()
    print("=" * 70)
    print("  Smart Sentinel AI — MT5 Setup & Verification")
    print("=" * 70)

    # Step 1: Find MT5 installation
    print("\n[1/5] Checking MT5 installation...")
    mt5_path = find_mt5_terminal()
    if mt5_path:
        print(f"  Found: {mt5_path}")
    else:
        print("  ERROR: MetaTrader 5 not found.")
        print("  Download from: https://www.metatrader5.com/en/download")
        sys.exit(1)

    # Step 2: Check if running
    print("\n[2/5] Checking if MT5 terminal is running...")
    if is_mt5_running():
        print("  MT5 terminal is running.")
    else:
        print("  MT5 terminal is NOT running.")
        if args.launch or mt5_path:
            if not launch_mt5(mt5_path):
                print("  Please start MetaTrader 5 manually and re-run this script.")
                sys.exit(1)
        else:
            print("  Start MT5 and re-run, or use --launch flag.")
            sys.exit(1)

    # Step 3: Check Python package
    print("\n[3/5] Checking MetaTrader5 Python package...")
    try:
        import MetaTrader5 as mt5
        print(f"  Installed: MetaTrader5 v{mt5.__version__}")
    except ImportError:
        print("  ERROR: MetaTrader5 package not installed.")
        print("  Fix: pip install MetaTrader5")
        sys.exit(1)

    # Step 4: Connect
    print("\n[4/5] Connecting to MT5...")
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")

    success, mt5_mod = connect_mt5(
        login=int(login) if login else None,
        password=password,
        server=server,
        path=mt5_path if mt5_path.endswith(".exe") else None,
    )

    if not success:
        print("\n  Connection FAILED. See instructions above.")
        if mt5_mod:
            mt5_mod.shutdown()
        sys.exit(1)

    print("  Connected successfully!")
    print()
    print_terminal_info(mt5_mod)
    print()
    print_account_info(mt5_mod)

    # Write .env.example with detected values
    account_info = mt5_mod.account_info()
    write_env_template(account_info)

    if args.check_only:
        print("\n  --check-only: skipping data tests.")
        mt5_mod.shutdown()
        print("\n  All checks passed!")
        return

    # Step 5: Test data feed
    print("\n[5/5] Testing data feed...")
    test_all_symbols(mt5_mod, symbols)

    if args.multi_tf:
        primary = symbols[0] if symbols else "XAUUSD"
        test_multi_timeframe(mt5_mod, primary)

    # Summary
    print()
    print("=" * 70)
    print("  SETUP COMPLETE")
    print("=" * 70)
    print()

    if account_info:
        print(f"  Your MT5 account: {account_info.login} @ {account_info.server}")
        print()
        print("  To start Smart Sentinel AI with live data:")
        print()
        print(f"    set DATA_SOURCE=mt5")
        print(f"    set MT5_LOGIN={account_info.login}")
        print(f"    set MT5_PASSWORD=<your_password>")
        print(f"    set MT5_SERVER={account_info.server}")
        print(f"    python -m src.intelligence.main")
        print()
        print("  Or use the batch file: scripts\\run_mt5_live.bat")

    mt5_mod.shutdown()
    print("\n  Done.")


if __name__ == "__main__":
    main()
