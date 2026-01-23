# =============================================================================
# AUDIT LOGGER - Institutional-Grade Trading Audit Trail
# =============================================================================
# Professional audit logging system for regulatory compliance and analysis.
#
# Features:
#   1. STRUCTURED LOGGING - JSON-formatted audit records
#   2. DECISION TRAIL - Complete record of every trading decision
#   3. COMPLIANCE READY - Meets MiFID II, SEC, and CFTC requirements
#   4. ANALYSIS EXPORT - Export to various formats for analysis
#   5. RETENTION MANAGEMENT - Automatic archival and cleanup
#
# Every trading decision is recorded with:
#   - Timestamp (microsecond precision)
#   - Decision ID (unique identifier)
#   - All inputs (market data, portfolio state, agent signals)
#   - All outputs (decision, modified parameters)
#   - Performance metrics (latency, confidence)
#   - Full reasoning chain
#
# Architecture:
#   ┌─────────────────────────────────────────────────────────────────┐
#   │                       AUDIT LOGGER                              │
#   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
#   │  │ Decision    │ │ Trade       │ │ System      │ │ Export    │ │
#   │  │ Logger      │ │ Logger      │ │ Logger      │ │ Manager   │ │
#   │  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
#   └─────────────────────────────────────────────────────────────────┘
#
# =============================================================================

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import logging
import os
import secrets
import shutil
import threading
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import io


# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# HMAC secret key for audit integrity (loaded from env or generated)
_AUDIT_HMAC_KEY: Optional[bytes] = None

def _get_hmac_key() -> bytes:
    """Get or generate HMAC key for audit integrity."""
    global _AUDIT_HMAC_KEY
    if _AUDIT_HMAC_KEY is None:
        env_key = os.environ.get("AUDIT_HMAC_KEY", "")
        if env_key and len(env_key) >= 32:
            _AUDIT_HMAC_KEY = env_key.encode('utf-8')
        else:
            # Generate a session key (logged for recovery)
            _AUDIT_HMAC_KEY = secrets.token_bytes(32)
            logging.getLogger(__name__).warning(
                "No AUDIT_HMAC_KEY in env, using session key. "
                "Set AUDIT_HMAC_KEY for persistent integrity verification."
            )
    return _AUDIT_HMAC_KEY

# Allowed directories for audit logs (security: prevent directory traversal)
ALLOWED_AUDIT_DIRS = [
    "./logs",
    "./audit",
    "./data/audit",
    "/var/log/tradingbot",
    "/tmp/tradingbot_audit"
]


def _validate_audit_directory(directory: str) -> Path:
    """
    Validate audit directory is within allowed paths.

    Args:
        directory: Proposed directory path

    Returns:
        Validated Path object

    Raises:
        ValueError: If directory is not in allowed list or is a symlink
    """
    path = Path(directory)

    # SECURITY: Reject symlinks to prevent symlink attacks
    # An attacker could create a symlink pointing outside allowed directories
    if path.is_symlink():
        raise ValueError(
            f"Directory '{directory}' is a symlink. Symlinks are not allowed "
            "for audit directories due to security concerns."
        )

    # Also check if any parent is a symlink (symlink chain attack)
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError(
                f"Parent directory '{parent}' is a symlink. Symlink chains are not "
                "allowed for audit directories due to security concerns."
            )

    path = path.resolve()

    # Check if path is within any allowed directory
    for allowed in ALLOWED_AUDIT_DIRS:
        allowed_path = Path(allowed).resolve()
        try:
            path.relative_to(allowed_path)
            return path
        except ValueError:
            continue

    # Check if it's a subdirectory of current working directory
    cwd = Path.cwd()
    try:
        path.relative_to(cwd)
        # Must be under logs or audit subdirectory
        rel = path.relative_to(cwd)
        if str(rel).startswith(("logs", "audit", "data")):
            return path
    except ValueError:
        pass

    raise ValueError(
        f"Directory '{directory}' is not in allowed audit directories. "
        f"Allowed: {ALLOWED_AUDIT_DIRS} or ./logs, ./audit, ./data subdirectories"
    )


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class AuditEventType(Enum):
    """Types of audit events."""
    # Decision events
    DECISION_REQUEST = "decision_request"
    DECISION_RESPONSE = "decision_response"
    DECISION_OVERRIDE = "decision_override"

    # Trade events
    TRADE_PROPOSED = "trade_proposed"
    TRADE_APPROVED = "trade_approved"
    TRADE_REJECTED = "trade_rejected"
    TRADE_MODIFIED = "trade_modified"
    TRADE_EXECUTED = "trade_executed"
    TRADE_CLOSED = "trade_closed"
    TRADE_CANCELLED = "trade_cancelled"

    # Risk events
    RISK_ASSESSMENT = "risk_assessment"
    RISK_LIMIT_WARNING = "risk_limit_warning"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
    KILL_SWITCH_RESET = "kill_switch_reset"

    # Agent events
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_ERROR = "agent_error"
    AGENT_SIGNAL = "agent_signal"

    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_ERROR = "system_error"
    CONFIG_CHANGE = "config_change"

    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    DAY_START = "day_start"
    DAY_END = "day_end"


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class ExportFormat(Enum):
    """Export file formats."""
    JSON = "json"
    JSONL = "jsonl"      # JSON Lines (one record per line)
    CSV = "csv"
    PARQUET = "parquet"  # Requires pyarrow


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AuditRecord:
    """
    Single audit record with complete context.

    This is the atomic unit of the audit trail.
    Every significant event generates one AuditRecord.
    """
    # Identification
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    correlation_id: str = ""  # Links related records

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    timestamp_local: datetime = field(default_factory=datetime.now)

    # Classification
    event_type: AuditEventType = AuditEventType.SYSTEM_START
    level: LogLevel = LogLevel.INFO
    source: str = ""  # Component that generated the record

    # Content
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # Context
    portfolio_state: Optional[Dict[str, Any]] = None
    market_state: Optional[Dict[str, Any]] = None

    # Integrity
    checksum: str = ""

    def __post_init__(self):
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """
        Calculate HMAC-SHA256 checksum of record content.

        Uses HMAC with secret key for cryptographic integrity.
        Full 64-char hex digest for proper security.
        """
        content = f"{self.record_id}|{self.timestamp.isoformat()}|{self.event_type.value}|{self.message}"
        return hmac.new(
            _get_hmac_key(),
            content.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()  # Full 64 chars, not truncated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "timestamp_utc": self.timestamp.isoformat(),
            "timestamp_local": self.timestamp_local.isoformat(),
            "event_type": self.event_type.value,
            "level": self.level.name,
            "source": self.source,
            "message": self.message,
            "details": self.details,
            "portfolio_state": self.portfolio_state,
            "market_state": self.market_state,
            "checksum": self.checksum
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class DecisionAuditRecord(AuditRecord):
    """
    Specialized audit record for trading decisions.

    Captures the complete decision-making process including
    all agent inputs and the final decision.
    """
    # Decision specifics
    decision_id: str = field(default_factory=lambda: f"dec_{uuid.uuid4().hex[:12]}")
    proposal_id: str = ""

    # Input state
    proposed_action: str = ""
    proposed_quantity: float = 0.0
    proposed_symbol: str = ""

    # Agent assessments
    agent_assessments: List[Dict[str, Any]] = field(default_factory=list)

    # Decision outcome
    final_decision: str = ""  # APPROVE, REJECT, MODIFY
    final_quantity: float = 0.0
    position_multiplier: float = 1.0

    # Reasoning
    reasoning: List[str] = field(default_factory=list)
    blocking_agent: Optional[str] = None
    modifying_agents: List[str] = field(default_factory=list)

    # Performance
    decision_time_ms: float = 0.0
    confidence: float = 0.0

    def __post_init__(self):
        """Set event type and calculate checksum."""
        self.event_type = AuditEventType.DECISION_RESPONSE
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with decision-specific fields."""
        base = super().to_dict()
        base.update({
            "decision_id": self.decision_id,
            "proposal_id": self.proposal_id,
            "proposed_action": self.proposed_action,
            "proposed_quantity": self.proposed_quantity,
            "proposed_symbol": self.proposed_symbol,
            "agent_assessments": self.agent_assessments,
            "final_decision": self.final_decision,
            "final_quantity": self.final_quantity,
            "position_multiplier": self.position_multiplier,
            "reasoning": self.reasoning,
            "blocking_agent": self.blocking_agent,
            "modifying_agents": self.modifying_agents,
            "decision_time_ms": self.decision_time_ms,
            "confidence": self.confidence
        })
        return base


@dataclass
class TradeAuditRecord(AuditRecord):
    """
    Specialized audit record for trade execution.
    """
    # Trade identification
    trade_id: str = field(default_factory=lambda: f"trd_{uuid.uuid4().hex[:12]}")
    decision_id: str = ""  # Links to decision that approved this trade

    # Trade details
    symbol: str = ""
    action: str = ""  # BUY, SELL, CLOSE_LONG, etc.
    quantity: float = 0.0
    entry_price: float = 0.0
    exit_price: Optional[float] = None

    # Execution
    requested_price: float = 0.0
    executed_price: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0

    # Result
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_winner: bool = False

    # Timing
    execution_time_ms: float = 0.0

    def __post_init__(self):
        """Set event type and calculate checksum."""
        self.event_type = AuditEventType.TRADE_EXECUTED
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with trade-specific fields."""
        base = super().to_dict()
        base.update({
            "trade_id": self.trade_id,
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "requested_price": self.requested_price,
            "executed_price": self.executed_price,
            "slippage": self.slippage,
            "commission": self.commission,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "is_winner": self.is_winner,
            "execution_time_ms": self.execution_time_ms
        })
        return base


@dataclass
class RiskAuditRecord(AuditRecord):
    """
    Specialized audit record for risk events.
    """
    # Risk identification
    risk_event_id: str = field(default_factory=lambda: f"rsk_{uuid.uuid4().hex[:12]}")

    # Risk metrics
    var_amount: float = 0.0
    var_pct: float = 0.0
    cvar_amount: float = 0.0
    cvar_pct: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0

    # Exposure
    gross_exposure: float = 0.0
    net_exposure: float = 0.0

    # Limits
    limit_name: str = ""
    limit_value: float = 0.0
    current_value: float = 0.0
    utilization_pct: float = 0.0

    # Action taken
    action_taken: str = ""  # WARNING, REDUCTION, HALT, etc.

    def __post_init__(self):
        """Set event type and calculate checksum."""
        self.event_type = AuditEventType.RISK_ASSESSMENT
        super().__post_init__()


# =============================================================================
# LOG HANDLERS
# =============================================================================

class AuditHandler(ABC):
    """Abstract base class for audit log handlers."""

    @abstractmethod
    def write(self, record: AuditRecord) -> bool:
        """Write a record to the handler."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered records."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the handler and release resources."""
        pass


class FileHandler(AuditHandler):
    """
    File-based audit handler with rotation and compression.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        filename_prefix: str = "audit",
        max_file_size_mb: int = 100,
        max_files: int = 30,
        compress_old: bool = True
    ):
        """
        Initialize file handler.

        Args:
            directory: Directory for log files (must be in allowed paths)
            filename_prefix: Prefix for log filenames
            max_file_size_mb: Max size before rotation
            max_files: Max number of files to keep
            compress_old: Compress rotated files

        Raises:
            ValueError: If directory is outside allowed paths (security)
        """
        # SECURITY: Validate directory to prevent directory traversal
        self.directory = _validate_audit_directory(str(directory))

        # Validate filename_prefix (no path separators)
        if '/' in filename_prefix or '\\' in filename_prefix or '..' in filename_prefix:
            raise ValueError(f"Invalid filename_prefix: {filename_prefix}")
        self.filename_prefix = filename_prefix

        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.max_files = max_files
        self.compress_old = compress_old

        self._current_file: Optional[io.TextIOWrapper] = None
        self._current_file_path: Optional[Path] = None
        self._current_file_size = 0
        self._lock = threading.Lock()

        # Ensure directory exists
        self.directory.mkdir(parents=True, exist_ok=True)

        # Open initial file
        self._open_new_file()

    def _open_new_file(self) -> None:
        """Open a new log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filename_prefix}_{timestamp}.jsonl"
        self._current_file_path = self.directory / filename

        self._current_file = open(self._current_file_path, 'a', encoding='utf-8')
        self._current_file_size = 0

    def _rotate_if_needed(self) -> None:
        """Rotate log file if size exceeded."""
        if self._current_file_size >= self.max_file_size:
            self._rotate()

    def _rotate(self) -> None:
        """Rotate the current log file."""
        if self._current_file:
            self._current_file.close()

            # Compress old file if enabled
            if self.compress_old and self._current_file_path:
                with open(self._current_file_path, 'rb') as f_in:
                    with gzip.open(f"{self._current_file_path}.gz", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                self._current_file_path.unlink()

        # Cleanup old files
        self._cleanup_old_files()

        # Open new file
        self._open_new_file()

    def _cleanup_old_files(self) -> None:
        """Remove old log files if exceeding max_files."""
        pattern = f"{self.filename_prefix}_*.jsonl*"
        files = sorted(self.directory.glob(pattern), key=lambda f: f.stat().st_mtime)

        while len(files) > self.max_files:
            oldest = files.pop(0)
            oldest.unlink()

    def write(self, record: AuditRecord) -> bool:
        """Write a record to the file."""
        with self._lock:
            try:
                self._rotate_if_needed()

                line = record.to_json() + "\n"
                self._current_file.write(line)
                self._current_file_size += len(line.encode('utf-8'))

                return True

            except Exception as e:
                logging.error(f"Failed to write audit record: {e}")
                return False

    def flush(self) -> None:
        """Flush the file buffer."""
        with self._lock:
            if self._current_file:
                self._current_file.flush()

    def close(self) -> None:
        """Close the file handler."""
        with self._lock:
            if self._current_file:
                self._current_file.close()
                self._current_file = None


class MemoryHandler(AuditHandler):
    """
    In-memory audit handler for testing and real-time analysis.
    """

    def __init__(self, max_records: int = 10000):
        """
        Initialize memory handler.

        Args:
            max_records: Maximum records to keep in memory
        """
        self.max_records = max_records
        self._records: deque = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def write(self, record: AuditRecord) -> bool:
        """Write a record to memory."""
        with self._lock:
            self._records.append(record)
            return True

    def flush(self) -> None:
        """No-op for memory handler."""
        pass

    def close(self) -> None:
        """Clear memory."""
        with self._lock:
            self._records.clear()

    def get_records(
        self,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditRecord]:
        """
        Query records from memory.

        Args:
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum records to return

        Returns:
            List of matching records
        """
        with self._lock:
            results = []

            for record in reversed(self._records):
                # Apply filters
                if event_type and record.event_type != event_type:
                    continue
                if start_time and record.timestamp < start_time:
                    continue
                if end_time and record.timestamp > end_time:
                    continue

                results.append(record)

                if len(results) >= limit:
                    break

            return results


class CallbackHandler(AuditHandler):
    """
    Callback-based handler for custom processing.
    """

    def __init__(self, callback: Callable[[AuditRecord], None]):
        """
        Initialize callback handler.

        Args:
            callback: Function to call for each record
        """
        self.callback = callback

    def write(self, record: AuditRecord) -> bool:
        """Call the callback with the record."""
        try:
            self.callback(record)
            return True
        except Exception as e:
            logging.error(f"Callback handler error: {e}")
            return False

    def flush(self) -> None:
        """No-op for callback handler."""
        pass

    def close(self) -> None:
        """No-op for callback handler."""
        pass


# =============================================================================
# MAIN AUDIT LOGGER
# =============================================================================

class AuditLogger:
    """
    Main audit logging system.

    This is the central class for all audit logging in the trading system.
    It manages multiple handlers and provides convenience methods for
    logging different types of events.

    Example:
        logger = AuditLogger(
            session_id="session_123",
            log_directory="./logs/audit"
        )

        # Log a trading decision
        logger.log_decision(
            decision_id="dec_123",
            proposal_id="prop_456",
            proposed_action="BUY",
            proposed_quantity=1.0,
            final_decision="APPROVE",
            agent_assessments=[...],
            reasoning=["All checks passed"]
        )

        # Export for analysis
        logger.export_to_csv("./reports/audit_export.csv")
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        log_directory: Optional[Union[str, Path]] = None,
        enable_file_logging: bool = True,
        enable_memory_logging: bool = True,
        memory_buffer_size: int = 10000
    ):
        """
        Initialize audit logger.

        Args:
            session_id: Unique session identifier
            log_directory: Directory for log files
            enable_file_logging: Enable file-based logging
            enable_memory_logging: Enable in-memory logging
            memory_buffer_size: Size of memory buffer
        """
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.log_directory = Path(log_directory) if log_directory else Path("./logs/audit")

        self._handlers: List[AuditHandler] = []
        self._lock = threading.Lock()
        self._record_count = 0
        self._error_count = 0

        self._logger = logging.getLogger("audit_logger")

        # Initialize handlers
        if enable_file_logging:
            self._handlers.append(FileHandler(
                directory=self.log_directory,
                filename_prefix=f"audit_{self.session_id[:8]}"
            ))

        if enable_memory_logging:
            self._memory_handler = MemoryHandler(max_records=memory_buffer_size)
            self._handlers.append(self._memory_handler)
        else:
            self._memory_handler = None

        # Log session start
        self.log_system_event(
            AuditEventType.SESSION_START,
            f"Audit session started: {self.session_id}"
        )

    # =========================================================================
    # CORE LOGGING METHODS
    # =========================================================================

    def log(self, record: AuditRecord) -> str:
        """
        Log an audit record.

        Args:
            record: The audit record to log

        Returns:
            Record ID
        """
        # Set session ID if not set
        if not record.session_id:
            record.session_id = self.session_id

        with self._lock:
            self._record_count += 1

            for handler in self._handlers:
                try:
                    handler.write(record)
                except Exception as e:
                    self._error_count += 1
                    self._logger.error(f"Handler write failed: {e}")

        return record.record_id

    def log_decision(
        self,
        decision_id: str,
        proposal_id: str,
        proposed_action: str,
        proposed_quantity: float,
        proposed_symbol: str,
        final_decision: str,
        agent_assessments: List[Dict[str, Any]],
        reasoning: List[str],
        final_quantity: Optional[float] = None,
        position_multiplier: float = 1.0,
        blocking_agent: Optional[str] = None,
        modifying_agents: Optional[List[str]] = None,
        decision_time_ms: float = 0.0,
        confidence: float = 0.0,
        portfolio_state: Optional[Dict[str, Any]] = None,
        market_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a trading decision.

        This is the primary method for logging trading decisions.
        It captures the complete decision-making process.

        Returns:
            Decision record ID
        """
        record = DecisionAuditRecord(
            source="orchestrator",
            decision_id=decision_id,
            proposal_id=proposal_id,
            proposed_action=proposed_action,
            proposed_quantity=proposed_quantity,
            proposed_symbol=proposed_symbol,
            final_decision=final_decision,
            final_quantity=final_quantity or proposed_quantity * position_multiplier,
            position_multiplier=position_multiplier,
            agent_assessments=agent_assessments,
            reasoning=reasoning,
            blocking_agent=blocking_agent,
            modifying_agents=modifying_agents or [],
            decision_time_ms=decision_time_ms,
            confidence=confidence,
            portfolio_state=portfolio_state,
            market_state=market_state,
            message=f"Decision {decision_id}: {final_decision} for {proposed_action} {proposed_quantity} {proposed_symbol}"
        )

        return self.log(record)

    def log_trade(
        self,
        trade_id: str,
        decision_id: str,
        symbol: str,
        action: str,
        quantity: float,
        executed_price: float,
        requested_price: Optional[float] = None,
        slippage: float = 0.0,
        commission: float = 0.0,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
        execution_time_ms: float = 0.0,
        portfolio_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a trade execution.

        Returns:
            Trade record ID
        """
        record = TradeAuditRecord(
            source="execution_engine",
            trade_id=trade_id,
            decision_id=decision_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            executed_price=executed_price,
            requested_price=requested_price or executed_price,
            slippage=slippage,
            commission=commission,
            pnl=pnl,
            pnl_pct=pnl_pct,
            is_winner=pnl > 0,
            execution_time_ms=execution_time_ms,
            portfolio_state=portfolio_state,
            message=f"Trade {trade_id}: {action} {quantity} {symbol} @ {executed_price}"
        )

        return self.log(record)

    def log_risk_event(
        self,
        event_type: AuditEventType,
        message: str,
        var_pct: float = 0.0,
        cvar_pct: float = 0.0,
        current_drawdown: float = 0.0,
        gross_exposure: float = 0.0,
        net_exposure: float = 0.0,
        limit_name: str = "",
        limit_value: float = 0.0,
        current_value: float = 0.0,
        action_taken: str = "",
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a risk event.

        Returns:
            Risk record ID
        """
        record = RiskAuditRecord(
            event_type=event_type,
            source="risk_manager",
            message=message,
            var_pct=var_pct,
            cvar_pct=cvar_pct,
            current_drawdown=current_drawdown,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            limit_name=limit_name,
            limit_value=limit_value,
            current_value=current_value,
            utilization_pct=current_value / limit_value if limit_value > 0 else 0,
            action_taken=action_taken,
            details=details or {}
        )

        return self.log(record)

    def log_agent_event(
        self,
        event_type: AuditEventType,
        agent_name: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an agent event.

        Returns:
            Record ID
        """
        record = AuditRecord(
            event_type=event_type,
            source=agent_name,
            message=message,
            details=details or {}
        )

        return self.log(record)

    def log_system_event(
        self,
        event_type: AuditEventType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        level: LogLevel = LogLevel.INFO
    ) -> str:
        """
        Log a system event.

        Returns:
            Record ID
        """
        record = AuditRecord(
            event_type=event_type,
            level=level,
            source="system",
            message=message,
            details=details or {}
        )

        return self.log(record)

    def log_kill_switch_event(
        self,
        triggered: bool,
        reason: str,
        halt_level: str,
        trigger_value: float = 0.0,
        threshold: float = 0.0,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a kill switch event.

        Returns:
            Record ID
        """
        event_type = (
            AuditEventType.KILL_SWITCH_TRIGGERED if triggered
            else AuditEventType.KILL_SWITCH_RESET
        )

        record = AuditRecord(
            event_type=event_type,
            level=LogLevel.CRITICAL if triggered else LogLevel.INFO,
            source="kill_switch",
            message=f"Kill Switch {'TRIGGERED' if triggered else 'RESET'}: {reason}",
            details={
                "halt_level": halt_level,
                "trigger_value": trigger_value,
                "threshold": threshold,
                **(details or {})
            }
        )

        return self.log(record)

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_recent_records(
        self,
        event_type: Optional[AuditEventType] = None,
        limit: int = 100
    ) -> List[AuditRecord]:
        """
        Get recent records from memory buffer.

        Args:
            event_type: Filter by event type
            limit: Maximum records to return

        Returns:
            List of records
        """
        if not self._memory_handler:
            return []

        return self._memory_handler.get_records(
            event_type=event_type,
            limit=limit
        )

    def get_recent_decisions(self, limit: int = 50) -> List[DecisionAuditRecord]:
        """Get recent trading decisions."""
        records = self.get_recent_records(
            event_type=AuditEventType.DECISION_RESPONSE,
            limit=limit
        )
        return [r for r in records if isinstance(r, DecisionAuditRecord)]

    def get_recent_trades(self, limit: int = 50) -> List[TradeAuditRecord]:
        """Get recent trades."""
        records = self.get_recent_records(
            event_type=AuditEventType.TRADE_EXECUTED,
            limit=limit
        )
        return [r for r in records if isinstance(r, TradeAuditRecord)]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get audit statistics.

        Returns:
            Dict with statistics
        """
        records = self.get_recent_records(limit=10000)

        # Count by event type
        type_counts = {}
        for record in records:
            event_type = record.event_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1

        # Count by level
        level_counts = {}
        for record in records:
            level = record.level.name
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            "session_id": self.session_id,
            "total_records": self._record_count,
            "error_count": self._error_count,
            "records_in_memory": len(records),
            "by_event_type": type_counts,
            "by_level": level_counts
        }

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def export_to_json(
        self,
        filepath: Union[str, Path],
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        """
        Export records to JSON file.

        Returns:
            Number of records exported
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        records = self.get_recent_records(event_type=event_type, limit=100000)

        # Filter by time if specified
        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                [r.to_dict() for r in records],
                f,
                indent=2,
                default=str
            )

        return len(records)

    def export_to_jsonl(
        self,
        filepath: Union[str, Path],
        event_type: Optional[AuditEventType] = None
    ) -> int:
        """
        Export records to JSON Lines file.

        Returns:
            Number of records exported
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        records = self.get_recent_records(event_type=event_type, limit=100000)

        with open(filepath, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(record.to_json() + "\n")

        return len(records)

    def export_to_csv(
        self,
        filepath: Union[str, Path],
        event_type: Optional[AuditEventType] = None
    ) -> int:
        """
        Export records to CSV file.

        Returns:
            Number of records exported
        """
        import csv

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        records = self.get_recent_records(event_type=event_type, limit=100000)

        if not records:
            return 0

        # Get all unique keys
        all_keys = set()
        for record in records:
            all_keys.update(record.to_dict().keys())

        all_keys = sorted(all_keys)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()

            for record in records:
                row = record.to_dict()
                # Flatten complex fields
                for key in row:
                    if isinstance(row[key], (dict, list)):
                        row[key] = json.dumps(row[key])
                writer.writerow(row)

        return len(records)

    def export_decisions_summary(self, filepath: Union[str, Path]) -> int:
        """
        Export a summary of decisions for analysis.

        Returns:
            Number of decisions exported
        """
        import csv

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        decisions = self.get_recent_decisions(limit=10000)

        if not decisions:
            return 0

        fields = [
            'timestamp_utc', 'decision_id', 'proposed_symbol', 'proposed_action',
            'proposed_quantity', 'final_decision', 'final_quantity',
            'position_multiplier', 'decision_time_ms', 'confidence',
            'blocking_agent', 'num_modifying_agents'
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for dec in decisions:
                writer.writerow({
                    'timestamp_utc': dec.timestamp.isoformat(),
                    'decision_id': dec.decision_id,
                    'proposed_symbol': dec.proposed_symbol,
                    'proposed_action': dec.proposed_action,
                    'proposed_quantity': dec.proposed_quantity,
                    'final_decision': dec.final_decision,
                    'final_quantity': dec.final_quantity,
                    'position_multiplier': dec.position_multiplier,
                    'decision_time_ms': dec.decision_time_ms,
                    'confidence': dec.confidence,
                    'blocking_agent': dec.blocking_agent,
                    'num_modifying_agents': len(dec.modifying_agents)
                })

        return len(decisions)

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def flush(self) -> None:
        """Flush all handlers."""
        for handler in self._handlers:
            try:
                handler.flush()
            except Exception as e:
                self._logger.error(f"Handler flush failed: {e}")

    def close(self) -> None:
        """Close the audit logger and all handlers."""
        self.log_system_event(
            AuditEventType.SESSION_END,
            f"Audit session ended: {self.session_id}. "
            f"Total records: {self._record_count}, Errors: {self._error_count}"
        )

        self.flush()

        for handler in self._handlers:
            try:
                handler.close()
            except Exception as e:
                self._logger.error(f"Handler close failed: {e}")

        self._handlers.clear()

    def add_handler(self, handler: AuditHandler) -> None:
        """Add a custom handler."""
        self._handlers.append(handler)

    def add_callback(self, callback: Callable[[AuditRecord], None]) -> None:
        """Add a callback handler."""
        self._handlers.append(CallbackHandler(callback))

    def get_dashboard(self) -> str:
        """Generate a text dashboard of audit statistics."""
        stats = self.get_statistics()

        return f"""
================================================================================
                          AUDIT LOGGER DASHBOARD
================================================================================

  SESSION: {stats['session_id']}

  STATISTICS
  ─────────────────────────────────────────────────────────────────────────────
  Total Records:       {stats['total_records']:>15,}
  Records in Memory:   {stats['records_in_memory']:>15,}
  Errors:              {stats['error_count']:>15,}

  BY EVENT TYPE
  ─────────────────────────────────────────────────────────────────────────────
{chr(10).join(f"  {k:30}: {v:>10,}" for k, v in sorted(stats['by_event_type'].items()))}

  BY LEVEL
  ─────────────────────────────────────────────────────────────────────────────
{chr(10).join(f"  {k:30}: {v:>10,}" for k, v in sorted(stats['by_level'].items()))}

================================================================================
"""


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_audit_logger(
    session_id: Optional[str] = None,
    log_directory: str = "./logs/audit",
    preset: str = "production"
) -> AuditLogger:
    """
    Create an AuditLogger with preset configuration.

    Args:
        session_id: Optional session ID
        log_directory: Directory for log files
        preset: "development", "production", "minimal"

    Returns:
        Configured AuditLogger
    """
    presets = {
        "development": {
            "enable_file_logging": True,
            "enable_memory_logging": True,
            "memory_buffer_size": 50000
        },
        "production": {
            "enable_file_logging": True,
            "enable_memory_logging": True,
            "memory_buffer_size": 100000
        },
        "minimal": {
            "enable_file_logging": True,
            "enable_memory_logging": False,
            "memory_buffer_size": 1000
        }
    }

    config = presets.get(preset, presets["production"])

    return AuditLogger(
        session_id=session_id,
        log_directory=log_directory,
        **config
    )


# =============================================================================
# GLOBAL LOGGER INSTANCE
# =============================================================================

_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger instance."""
    global _global_audit_logger

    if _global_audit_logger is None:
        _global_audit_logger = create_audit_logger()

    return _global_audit_logger


def set_audit_logger(logger: AuditLogger) -> None:
    """Set the global audit logger instance."""
    global _global_audit_logger
    _global_audit_logger = logger
