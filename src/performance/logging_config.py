# =============================================================================
# STRUCTURED LOGGING - JSON Logging Configuration
# =============================================================================
# Production-grade structured logging for the trading system.
#
# Features:
# - JSON-formatted log output (machine-readable)
# - Context injection (trade_id, agent_id, session_id)
# - Performance timing
# - Log aggregation friendly (ELK, Datadog, CloudWatch)
# - Human-readable fallback for development
#
# Usage:
#   from src.performance.logging_config import setup_structured_logging, get_trading_logger
#
#   setup_structured_logging(level="INFO", json_format=True)
#
#   logger = get_trading_logger("agent.risk")
#   logger.info("Trade evaluated", extra={
#       'trade_id': 'abc123',
#       'action': 'OPEN_LONG',
#       'position_size': 0.15,
#       'decision': 'APPROVE'
#   })
#
#   # Output:
#   # {"timestamp":"2024-01-15T10:30:00Z","level":"INFO","logger":"agent.risk",
#   #  "message":"Trade evaluated","trade_id":"abc123","action":"OPEN_LONG",
#   #  "position_size":0.15,"decision":"APPROVE"}
#
# =============================================================================

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# =============================================================================
# CONTEXT VARIABLES (Thread-safe context propagation)
# =============================================================================

# These propagate automatically through async/sync call chains
_session_id: ContextVar[str] = ContextVar('session_id', default='')
_trade_id: ContextVar[str] = ContextVar('trade_id', default='')
_agent_id: ContextVar[str] = ContextVar('agent_id', default='')
_request_id: ContextVar[str] = ContextVar('request_id', default='')


class LogContext:
    """
    Context manager for injecting fields into all log messages.

    Usage:
        with LogContext(trade_id='abc123', agent_id='risk_sentinel'):
            logger.info("Processing trade")  # Automatically includes trade_id and agent_id
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        trade_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        self._tokens = []
        self._values = {
            'session_id': (session_id, _session_id),
            'trade_id': (trade_id, _trade_id),
            'agent_id': (agent_id, _agent_id),
            'request_id': (request_id, _request_id),
        }

    def __enter__(self):
        for name, (value, ctx_var) in self._values.items():
            if value is not None:
                token = ctx_var.set(value)
                self._tokens.append((ctx_var, token))
        return self

    def __exit__(self, *args):
        for ctx_var, token in reversed(self._tokens):
            ctx_var.reset(token)


def get_log_context() -> Dict[str, str]:
    """Get current log context values."""
    ctx = {}
    if _session_id.get():
        ctx['session_id'] = _session_id.get()
    if _trade_id.get():
        ctx['trade_id'] = _trade_id.get()
    if _agent_id.get():
        ctx['agent_id'] = _agent_id.get()
    if _request_id.get():
        ctx['request_id'] = _request_id.get()
    return ctx


# =============================================================================
# JSON FORMATTER
# =============================================================================

class StructuredJsonFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Produces one JSON object per log line, suitable for
    log aggregation systems (ELK, Datadog, CloudWatch).
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_context: bool = True,
        include_source: bool = True,
        include_exception: bool = True,
    ):
        super().__init__()
        self._include_timestamp = include_timestamp
        self._include_context = include_context
        self._include_source = include_source
        self._include_exception = include_exception

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {}

        # Timestamp (ISO 8601 UTC)
        if self._include_timestamp:
            log_entry['timestamp'] = datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat()

        # Core fields
        log_entry['level'] = record.levelname
        log_entry['logger'] = record.name
        log_entry['message'] = record.getMessage()

        # Source location
        if self._include_source:
            log_entry['source'] = {
                'file': record.pathname,
                'line': record.lineno,
                'function': record.funcName,
            }

        # Context variables
        if self._include_context:
            ctx = get_log_context()
            if ctx:
                log_entry['context'] = ctx

        # Extra fields (passed via extra={} in log calls)
        reserved = {
            'name', 'msg', 'args', 'created', 'filename',
            'funcName', 'levelname', 'levelno', 'lineno',
            'module', 'msecs', 'pathname', 'process',
            'processName', 'relativeCreated', 'stack_info',
            'thread', 'threadName', 'exc_info', 'exc_text',
            'message', 'taskName',
        }
        extra = {
            k: v for k, v in record.__dict__.items()
            if k not in reserved and not k.startswith('_')
        }
        if extra:
            log_entry['data'] = extra

        # Exception info
        if self._include_exception and record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info) if record.exc_info[2] else None,
            }

        try:
            return json.dumps(log_entry, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            # Fallback for non-serializable data
            log_entry.pop('data', None)
            return json.dumps(log_entry, default=str, ensure_ascii=False)


# =============================================================================
# HUMAN-READABLE FORMATTER (for development)
# =============================================================================

class DevelopmentFormatter(logging.Formatter):
    """
    Human-readable formatter for development use.

    Color-coded output with context information.
    """

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET

        # Timestamp
        ts = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]

        # Context
        ctx = get_log_context()
        ctx_str = ""
        if ctx:
            parts = [f"{k}={v}" for k, v in ctx.items()]
            ctx_str = f" [{', '.join(parts)}]"

        # Base message
        msg = f"{color}{ts} {record.levelname:8}{reset} {record.name}: {record.getMessage()}{ctx_str}"

        # Exception
        if record.exc_info and record.exc_info[2]:
            msg += f"\n{self.formatException(record.exc_info)}"

        return msg


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_structured_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
    include_source: bool = False,
) -> None:
    """
    Set up structured logging for the trading system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON format (True for production, False for dev)
        log_file: Optional file path for log output
        include_source: Include source file/line in logs
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if json_format:
        formatter = StructuredJsonFormatter(
            include_source=include_source,
        )
    else:
        formatter = DevelopmentFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (always JSON for machine processing)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(StructuredJsonFormatter(
            include_source=True,
        ))
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_trading_logger(name: str) -> logging.Logger:
    """
    Get a logger for a trading component.

    Args:
        name: Logger name (e.g., "agent.risk", "orchestrator", "environment")

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"trading.{name}")


# =============================================================================
# PERFORMANCE TIMING
# =============================================================================

class TimingContext:
    """
    Context manager for timing code blocks and logging results.

    Usage:
        with TimingContext("decision_coordination", logger) as timer:
            result = orchestrator.coordinate(proposal)
        # Automatically logs: "decision_coordination completed in 12.5ms"
    """

    def __init__(
        self,
        operation: str,
        logger: Optional[logging.Logger] = None,
        level: int = logging.DEBUG,
        warn_threshold_ms: Optional[float] = None,
    ):
        self._operation = operation
        self._logger = logger or logging.getLogger("trading.timing")
        self._level = level
        self._warn_threshold_ms = warn_threshold_ms
        self._start: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

        if (self._warn_threshold_ms and
                self.elapsed_ms > self._warn_threshold_ms):
            self._logger.warning(
                f"{self._operation} slow: {self.elapsed_ms:.1f}ms "
                f"(threshold: {self._warn_threshold_ms}ms)",
                extra={
                    'operation': self._operation,
                    'elapsed_ms': self.elapsed_ms,
                    'threshold_ms': self._warn_threshold_ms,
                }
            )
        else:
            self._logger.log(
                self._level,
                f"{self._operation}: {self.elapsed_ms:.1f}ms",
                extra={
                    'operation': self._operation,
                    'elapsed_ms': self.elapsed_ms,
                }
            )
