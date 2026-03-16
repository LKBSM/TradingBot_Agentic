# =============================================================================
# LIVE TRADING LOOP — M15 Bar-by-Bar Execution (Sprint 16)
# =============================================================================
# Connects the trained PPO model to MT5 via ExecutionBridge.
#
# Flow (each M15 bar close):
#   1. Fetch latest OHLCV from MT5
#   2. Compute features (TA, SMC, MTF)
#   3. Build observation vector
#   4. Model predicts action
#   5. ExecutionBridge sends to MT5
#   6. Feed return to VaREngine → update KillSwitch
#   7. Log & alert
#
# Usage:
#   loop = LiveTradingLoop(
#       model_path="trained_models/best_model.zip",
#       mt5_account=12345678,
#       mt5_password="secret",
#       mt5_server="Broker-Server",
#       paper_mode=True,  # Start with paper trading!
#   )
#   loop.run()
# =============================================================================

import os
import sys
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class LiveTradingLoop:
    """
    Main live trading loop connecting PPO model to MT5 broker.

    Args:
        model_path: Path to trained SB3 PPO model (.zip)
        mt5_account: MT5 account number
        mt5_password: MT5 account password
        mt5_server: MT5 broker server name
        symbol: Trading symbol (default "XAUUSD")
        timeframe_minutes: Bar timeframe in minutes (default 15)
        paper_mode: If True, simulate trades without broker execution
        kill_switch_db: Path to kill switch SQLite database
    """

    def __init__(
        self,
        model_path: str,
        mt5_account: int = 0,
        mt5_password: str = "",
        mt5_server: str = "",
        symbol: str = "XAUUSD",
        timeframe_minutes: int = 15,
        paper_mode: bool = True,
        kill_switch_db: str = None,
    ):
        self.model_path = model_path
        self.symbol = symbol
        self.timeframe_minutes = timeframe_minutes
        self.paper_mode = paper_mode

        # State
        self._running = False
        self._step_count = 0
        self._last_bar_time = None

        # Initialize components (lazy — only when run() is called)
        self._model = None
        self._connector = None
        self._bridge = None
        self._var_engine = None
        self._kill_switch = None
        self._spread_model = None
        self._slippage_model = None

        # MT5 credentials
        self._mt5_account = mt5_account
        self._mt5_password = mt5_password
        self._mt5_server = mt5_server
        self._kill_switch_db = kill_switch_db

        logger.info(
            f"LiveTradingLoop initialized: {symbol} M{timeframe_minutes} "
            f"{'PAPER' if paper_mode else 'LIVE'} mode"
        )

    def _initialize_components(self) -> None:
        """Lazy-initialize all components."""
        # 1. Load trained model
        from stable_baselines3 import PPO
        logger.info(f"Loading model from {self.model_path}...")
        self._model = PPO.load(self.model_path)
        logger.info("Model loaded successfully")

        # 2. Initialize MT5 connector (if live)
        if not self.paper_mode:
            from src.live_trading.mt5_connector import MT5Connector
            self._connector = MT5Connector(
                account=self._mt5_account,
                password=self._mt5_password,
                server=self._mt5_server,
            )
            self._connector.connect()
            logger.info("MT5 connected")

        # 3. Initialize execution bridge
        from src.live_trading.execution_bridge import ExecutionBridge
        self._bridge = ExecutionBridge(
            connector=self._connector,
            symbol=self.symbol,
            paper_mode=self.paper_mode,
        )

        # 4. Initialize VaR engine
        from src.risk.var_engine import VaREngine
        from config import VAR_CONFIDENCE_LEVEL, VAR_ROLLING_WINDOW, VAR_METHOD
        self._var_engine = VaREngine(
            confidence=VAR_CONFIDENCE_LEVEL,
            window=VAR_ROLLING_WINDOW,
            method=VAR_METHOD,
        )

        # 5. Initialize dynamic spread & slippage
        from src.environment.execution_model import DynamicSpreadModel, DynamicSlippageModel
        from config import SLIPPAGE_PERCENTAGE, SPREAD_NEWS_MULTIPLIER
        self._spread_model = DynamicSpreadModel(news_multiplier=SPREAD_NEWS_MULTIPLIER)
        self._slippage_model = DynamicSlippageModel(base_slippage=SLIPPAGE_PERCENTAGE)

        # 6. Initialize kill switch (optional)
        if self._kill_switch_db:
            try:
                from src.agents.kill_switch import KillSwitch
                self._kill_switch = KillSwitch(db_path=self._kill_switch_db)
                logger.info("Kill switch initialized")
            except ImportError:
                logger.warning("KillSwitch not available — proceeding without")

        logger.info("All components initialized")

    def run(self, max_steps: int = None) -> None:
        """
        Start the live trading loop.

        Args:
            max_steps: Maximum number of bars to process (None = run forever)
        """
        self._initialize_components()
        self._running = True

        logger.info(f"Starting live trading loop (max_steps={max_steps})...")

        try:
            while self._running:
                if max_steps is not None and self._step_count >= max_steps:
                    logger.info(f"Reached max_steps={max_steps}, stopping")
                    break

                # Wait for next bar close
                self._wait_for_bar_close()

                # Process the bar
                self._process_bar()

                self._step_count += 1

        except KeyboardInterrupt:
            logger.info("Trading loop stopped by user")
        except Exception as e:
            logger.error(f"Trading loop error: {e}", exc_info=True)
        finally:
            self._shutdown()

    def _wait_for_bar_close(self) -> None:
        """Wait until the next M15 bar closes."""
        now = datetime.now(timezone.utc)
        minutes = now.minute
        # Next bar close at next multiple of timeframe_minutes
        next_close_minute = ((minutes // self.timeframe_minutes) + 1) * self.timeframe_minutes
        if next_close_minute >= 60:
            # Roll over to next hour
            wait_seconds = (60 - minutes) * 60 - now.second + 5  # +5s buffer
        else:
            wait_seconds = (next_close_minute - minutes) * 60 - now.second + 5

        if wait_seconds > 0:
            logger.debug(f"Waiting {wait_seconds:.0f}s for next bar close...")
            time.sleep(max(0, wait_seconds))

    def _process_bar(self) -> None:
        """Process a single bar: observe → predict → execute."""
        # This is a scaffolding implementation — the actual observation
        # construction requires the full feature pipeline from environment.py
        logger.info(f"Processing bar #{self._step_count}")

        # In a full implementation, this would:
        # 1. Fetch OHLCV data from MT5 or data feed
        # 2. Compute technical indicators (RSI, MACD, BB, ATR, etc.)
        # 3. Compute SMC features (FVG, BOS, CHOCH, OB)
        # 4. Compute MTF features (1H, 4H)
        # 5. Build lookback window observation
        # 6. Run model.predict(obs)
        # 7. Execute action via bridge
        # 8. Update VaR & kill switch

        # For now, log that the loop is running correctly
        logger.info(f"Bar #{self._step_count} processed (scaffolding)")

    def stop(self) -> None:
        """Signal the loop to stop."""
        self._running = False
        logger.info("Stop signal received")

    def _shutdown(self) -> None:
        """Clean shutdown of all components."""
        self._running = False

        # Close any open positions (safety)
        if self._bridge and self._bridge.has_position:
            logger.warning("Shutting down with open position — closing...")
            direction = self._bridge.position_direction
            self._bridge.execute_action(
                action=2 if direction == "BUY" else 4,  # CLOSE_LONG or CLOSE_SHORT
            )

        # Disconnect MT5
        if self._connector:
            try:
                self._connector.disconnect()
            except Exception:
                pass

        logger.info(f"Trading loop shut down after {self._step_count} bars")

    def get_status(self) -> Dict[str, Any]:
        """Get current loop status."""
        return {
            'running': self._running,
            'step_count': self._step_count,
            'paper_mode': self.paper_mode,
            'has_position': self._bridge.has_position if self._bridge else False,
            'var_ready': self._var_engine.is_ready if self._var_engine else False,
            'kill_switch_ok': True,  # Placeholder
        }
