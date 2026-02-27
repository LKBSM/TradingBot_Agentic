"""Signal performance tracker — reads closed signals from SignalStore's SQLite DB."""

from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from config import RISK_FREE_RATE, TRADING_DAYS_YEAR

logger = logging.getLogger(__name__)


class SignalTracker:
    """
    Computes live performance stats from the signals table.

    Uses the same DB as ``SignalStore`` (read-only queries).
    Thread-safe: opens its own connection per call.
    """

    def __init__(self, db_path: str = "./data/signals.db"):
        self._db_path = Path(db_path)

    # --------------------------------------------------------------------- #
    # SQLite helpers (same pattern as SignalStore)
    # --------------------------------------------------------------------- #
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path), timeout=30.0, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _table_exists(self, conn: sqlite3.Connection) -> bool:
        """Check whether the signals table has been created."""
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
        )
        return cur.fetchone() is not None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    _EMPTY_SUMMARY: Dict[str, Any] = {
        "total": 0,
        "won": 0,
        "lost": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_rr": 0.0,
        "cumulative_pnl": 0.0,
        "sharpe_30d": 0.0,
        "max_drawdown_pct": 0.0,
    }

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Aggregate performance for closed signals within the last N days.

        Returns dict with: total, won, lost, win_rate, profit_factor,
        avg_rr, cumulative_pnl, sharpe_30d, max_drawdown_pct.
        """
        conn = self._get_connection()
        try:
            if not self._table_exists(conn):
                return dict(self._EMPTY_SUMMARY)
            rows = conn.execute(
                "SELECT pnl_pips, rr_ratio FROM signals "
                "WHERE outcome IS NOT NULL "
                "AND closed_at >= datetime('now', ?)",
                (f"-{days} days",),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {
                "total": 0,
                "won": 0,
                "lost": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_rr": 0.0,
                "cumulative_pnl": 0.0,
                "sharpe_30d": 0.0,
                "max_drawdown_pct": 0.0,
            }

        pnls = [r["pnl_pips"] for r in rows]
        rrs = [r["rr_ratio"] for r in rows]

        total = len(pnls)
        won = sum(1 for p in pnls if p > 0)
        lost = total - won
        win_rate = won / total if total else 0.0

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        avg_rr = sum(rrs) / len(rrs) if rrs else 0.0
        cumulative_pnl = sum(pnls)

        sharpe = self._compute_sharpe(pnls)
        max_dd = self._compute_max_drawdown_pct(pnls)

        return {
            "total": total,
            "won": won,
            "lost": lost,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.99,
            "avg_rr": round(avg_rr, 4),
            "cumulative_pnl": round(cumulative_pnl, 2),
            "sharpe_30d": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 4),
        }

    def get_equity_curve(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Return cumulative PnL curve for closed signals in the last N days.

        Ordered by closed_at ascending.
        """
        conn = self._get_connection()
        try:
            if not self._table_exists(conn):
                return []
            rows = conn.execute(
                "SELECT signal_id, closed_at, pnl_pips FROM signals "
                "WHERE outcome IS NOT NULL "
                "AND closed_at >= datetime('now', ?) "
                "ORDER BY closed_at ASC",
                (f"-{days} days",),
            ).fetchall()
        finally:
            conn.close()

        curve: List[Dict[str, Any]] = []
        cumulative = 0.0
        for r in rows:
            cumulative += r["pnl_pips"]
            curve.append({
                "signal_id": r["signal_id"],
                "closed_at": r["closed_at"],
                "pnl_pips": r["pnl_pips"],
                "cumulative_pnl": round(cumulative, 2),
            })
        return curve

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    @staticmethod
    def _compute_sharpe(pnls: List[float]) -> float:
        """Annualised Sharpe ratio from a list of PnL values."""
        n = len(pnls)
        if n < 2:
            return 0.0

        mean = sum(pnls) / n
        variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0:
            return 0.0

        daily_rf = RISK_FREE_RATE / TRADING_DAYS_YEAR
        sharpe = (mean - daily_rf) / std
        return sharpe * math.sqrt(TRADING_DAYS_YEAR)

    @staticmethod
    def _compute_max_drawdown_pct(pnls: List[float]) -> float:
        """Max drawdown as percentage of peak cumulative PnL."""
        if not pnls:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        if peak <= 0:
            return 0.0

        return round((max_dd / peak) * 100, 4)
