# =============================================================================
# METRICS - Prometheus-Compatible Metrics Collection
# =============================================================================
# Lightweight metrics collection compatible with Prometheus exposition format.
#
# No external dependencies required. The registry can:
# - Export to Prometheus text format (for scraping)
# - Export to JSON (for REST APIs / dashboards)
# - Log periodic summaries
#
# Metric Types:
# - Counter: Monotonically increasing value (e.g., total_trades)
# - Gauge: Value that goes up and down (e.g., current_drawdown)
# - Histogram: Distribution of values (e.g., trade_pnl_distribution)
#
# Usage:
#   registry = get_registry()
#
#   trades_total = registry.counter('trades_total', 'Total trades executed')
#   trades_total.inc()
#   trades_total.inc(labels={'action': 'OPEN_LONG'})
#
#   drawdown = registry.gauge('current_drawdown_pct', 'Current drawdown %')
#   drawdown.set(5.2)
#
#   pnl_hist = registry.histogram('trade_pnl', 'Trade PnL distribution',
#       buckets=[-100, -50, -10, 0, 10, 50, 100, 500])
#   pnl_hist.observe(42.50)
#
#   # Export
#   print(registry.to_prometheus())
#   data = registry.to_json()
#
# =============================================================================

import logging
import math
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# LABEL HELPERS
# =============================================================================

def _labels_key(labels: Optional[Dict[str, str]]) -> str:
    """Convert labels dict to a hashable key."""
    if not labels:
        return ""
    return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))


def _labels_prometheus(labels: Optional[Dict[str, str]]) -> str:
    """Format labels for Prometheus text format."""
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}"


# =============================================================================
# COUNTER
# =============================================================================

class Counter:
    """
    Monotonically increasing counter.

    Use for counting events: trades, errors, requests, etc.
    """

    def __init__(self, name: str, description: str = ""):
        self._name = name
        self._description = description
        self._values: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def inc(
        self,
        amount: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment the counter.

        Args:
            amount: Amount to increment (must be positive)
            labels: Optional labels dict
        """
        if amount < 0:
            raise ValueError("Counter can only be incremented")
        key = _labels_key(labels)
        with self._lock:
            self._values[key] += amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        key = _labels_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def reset(self) -> None:
        """Reset all values (use with care)."""
        with self._lock:
            self._values.clear()

    def to_prometheus(self) -> str:
        """Export to Prometheus text format."""
        lines = [
            f"# HELP {self._name} {self._description}",
            f"# TYPE {self._name} counter",
        ]
        with self._lock:
            for key, value in sorted(self._values.items()):
                label_str = f"{{{key}}}" if key else ""
                lines.append(f"{self._name}{label_str} {value}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        with self._lock:
            return {
                'name': self._name,
                'type': 'counter',
                'description': self._description,
                'values': dict(self._values),
                'total': sum(self._values.values()),
            }


# =============================================================================
# GAUGE
# =============================================================================

class Gauge:
    """
    Gauge metric that can go up and down.

    Use for current state values: balance, drawdown, position count, etc.
    """

    def __init__(self, name: str, description: str = ""):
        self._name = name
        self._description = description
        self._values: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def set(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set gauge to a specific value."""
        key = _labels_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(
        self,
        amount: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment gauge."""
        key = _labels_key(labels)
        with self._lock:
            self._values[key] += amount

    def dec(
        self,
        amount: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Decrement gauge."""
        key = _labels_key(labels)
        with self._lock:
            self._values[key] -= amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        key = _labels_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def to_prometheus(self) -> str:
        """Export to Prometheus text format."""
        lines = [
            f"# HELP {self._name} {self._description}",
            f"# TYPE {self._name} gauge",
        ]
        with self._lock:
            for key, value in sorted(self._values.items()):
                label_str = f"{{{key}}}" if key else ""
                lines.append(f"{self._name}{label_str} {value}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        with self._lock:
            return {
                'name': self._name,
                'type': 'gauge',
                'description': self._description,
                'values': dict(self._values),
            }


# =============================================================================
# HISTOGRAM
# =============================================================================

# Default buckets similar to Prometheus
DEFAULT_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
    0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf')
)

# Trading-specific buckets
TRADE_PNL_BUCKETS = (
    -500, -200, -100, -50, -20, -10, -5, 0,
    5, 10, 20, 50, 100, 200, 500, float('inf')
)

LATENCY_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
    0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')
)


class Histogram:
    """
    Histogram for value distributions.

    Use for measuring distributions: trade PnL, latency, position sizes.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Sequence[float] = DEFAULT_BUCKETS
    ):
        self._name = name
        self._description = description
        self._upper_bounds = sorted(set(list(buckets) + [float('inf')]))

        # Per-label storage
        self._data: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def _get_data(self, key: str) -> Dict:
        """Get or create data for a label key."""
        if key not in self._data:
            self._data[key] = {
                'buckets': {b: 0 for b in self._upper_bounds},
                'sum': 0.0,
                'count': 0,
                'min': float('inf'),
                'max': float('-inf'),
            }
        return self._data[key]

    def observe(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record an observation.

        Args:
            value: The value to observe
            labels: Optional labels
        """
        key = _labels_key(labels)
        with self._lock:
            data = self._get_data(key)
            data['sum'] += value
            data['count'] += 1
            data['min'] = min(data['min'], value)
            data['max'] = max(data['max'], value)

            # Increment appropriate buckets
            for bound in self._upper_bounds:
                if value <= bound:
                    data['buckets'][bound] += 1

    def get_summary(
        self,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get summary statistics for a label set."""
        key = _labels_key(labels)
        with self._lock:
            if key not in self._data:
                return {'count': 0, 'sum': 0, 'mean': 0, 'min': 0, 'max': 0}

            data = self._data[key]
            count = data['count']
            return {
                'count': count,
                'sum': data['sum'],
                'mean': data['sum'] / count if count > 0 else 0,
                'min': data['min'] if count > 0 else 0,
                'max': data['max'] if count > 0 else 0,
            }

    def to_prometheus(self) -> str:
        """Export to Prometheus text format."""
        lines = [
            f"# HELP {self._name} {self._description}",
            f"# TYPE {self._name} histogram",
        ]
        with self._lock:
            for key, data in sorted(self._data.items()):
                label_prefix = f"{{{key}," if key else "{"
                # Cumulative bucket counts
                cumulative = 0
                for bound in self._upper_bounds:
                    cumulative += data['buckets'].get(bound, 0)
                    if bound == float('inf'):
                        le_val = "+Inf"
                    else:
                        le_val = str(bound)
                    lines.append(
                        f'{self._name}_bucket{label_prefix}le="{le_val}"}} {cumulative}'
                    )
                label_str = f"{{{key}}}" if key else ""
                lines.append(f"{self._name}_sum{label_str} {data['sum']}")
                lines.append(f"{self._name}_count{label_str} {data['count']}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        with self._lock:
            result = {
                'name': self._name,
                'type': 'histogram',
                'description': self._description,
                'series': {},
            }
            for key, data in self._data.items():
                result['series'][key or 'default'] = {
                    'count': data['count'],
                    'sum': data['sum'],
                    'mean': data['sum'] / data['count'] if data['count'] > 0 else 0,
                    'min': data['min'] if data['count'] > 0 else 0,
                    'max': data['max'] if data['count'] > 0 else 0,
                    'buckets': {
                        str(b): c for b, c in data['buckets'].items()
                    },
                }
            return result


# =============================================================================
# METRICS REGISTRY
# =============================================================================

class MetricsRegistry:
    """
    Central registry for all metrics.

    Thread-safe singleton for collecting and exporting all trading metrics.
    """

    def __init__(self, prefix: str = "trading"):
        self._prefix = prefix
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()
        self._created_at = time.time()

    def counter(
        self,
        name: str,
        description: str = ""
    ) -> Counter:
        """Get or create a counter metric."""
        full_name = f"{self._prefix}_{name}"
        with self._lock:
            if full_name not in self._counters:
                self._counters[full_name] = Counter(full_name, description)
            return self._counters[full_name]

    def gauge(
        self,
        name: str,
        description: str = ""
    ) -> Gauge:
        """Get or create a gauge metric."""
        full_name = f"{self._prefix}_{name}"
        with self._lock:
            if full_name not in self._gauges:
                self._gauges[full_name] = Gauge(full_name, description)
            return self._gauges[full_name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: Sequence[float] = DEFAULT_BUCKETS
    ) -> Histogram:
        """Get or create a histogram metric."""
        full_name = f"{self._prefix}_{name}"
        with self._lock:
            if full_name not in self._histograms:
                self._histograms[full_name] = Histogram(
                    full_name, description, buckets
                )
            return self._histograms[full_name]

    def to_prometheus(self) -> str:
        """Export all metrics to Prometheus text format."""
        with self._lock:
            sections = []
            for metric in self._counters.values():
                sections.append(metric.to_prometheus())
            for metric in self._gauges.values():
                sections.append(metric.to_prometheus())
            for metric in self._histograms.values():
                sections.append(metric.to_prometheus())
            return "\n\n".join(sections) + "\n"

    def to_json(self) -> Dict[str, Any]:
        """Export all metrics as JSON-friendly dict."""
        with self._lock:
            return {
                'prefix': self._prefix,
                'uptime_seconds': time.time() - self._created_at,
                'counters': {
                    name: m.to_dict() for name, m in self._counters.items()
                },
                'gauges': {
                    name: m.to_dict() for name, m in self._gauges.items()
                },
                'histograms': {
                    name: m.to_dict() for name, m in self._histograms.items()
                },
            }

    def get_all_names(self) -> List[str]:
        """Get all registered metric names."""
        with self._lock:
            return (
                list(self._counters.keys()) +
                list(self._gauges.keys()) +
                list(self._histograms.keys())
            )

    def reset_all(self) -> None:
        """Reset all metrics (use for testing)."""
        with self._lock:
            for c in self._counters.values():
                c.reset()
            self._gauges.clear()
            self._histograms.clear()


# =============================================================================
# GLOBAL REGISTRY SINGLETON
# =============================================================================

_global_registry: Optional[MetricsRegistry] = None
_registry_lock = threading.Lock()


def get_registry(prefix: str = "trading") -> MetricsRegistry:
    """Get the global metrics registry singleton."""
    global _global_registry
    with _registry_lock:
        if _global_registry is None:
            _global_registry = MetricsRegistry(prefix)
        return _global_registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    with _registry_lock:
        _global_registry = None


# =============================================================================
# PRE-DEFINED TRADING METRICS
# =============================================================================

def create_trading_metrics(
    registry: Optional[MetricsRegistry] = None
) -> Dict[str, Any]:
    """
    Create standard trading metrics.

    Returns a dict of pre-configured metrics for common trading operations.
    """
    reg = registry or get_registry()

    return {
        # Trade execution
        'trades_total': reg.counter(
            'trades_total', 'Total number of trades executed'
        ),
        'trades_won': reg.counter(
            'trades_won', 'Number of winning trades'
        ),
        'trades_lost': reg.counter(
            'trades_lost', 'Number of losing trades'
        ),

        # Decisions
        'decisions_total': reg.counter(
            'decisions_total', 'Total orchestrated decisions'
        ),
        'decisions_approved': reg.counter(
            'decisions_approved', 'Approved trade proposals'
        ),
        'decisions_rejected': reg.counter(
            'decisions_rejected', 'Rejected trade proposals'
        ),

        # Financial
        'current_balance': reg.gauge(
            'current_balance_usd', 'Current account balance (USD)'
        ),
        'current_equity': reg.gauge(
            'current_equity_usd', 'Current account equity (USD)'
        ),
        'current_drawdown_pct': reg.gauge(
            'current_drawdown_pct', 'Current drawdown percentage'
        ),
        'position_count': reg.gauge(
            'open_positions', 'Number of open positions'
        ),

        # Performance distributions
        'trade_pnl': reg.histogram(
            'trade_pnl_usd', 'Trade PnL distribution (USD)',
            buckets=TRADE_PNL_BUCKETS
        ),
        'trade_duration': reg.histogram(
            'trade_duration_steps', 'Trade duration in steps',
            buckets=(1, 5, 10, 20, 40, 80, 160, 320, float('inf'))
        ),

        # Latency
        'decision_latency': reg.histogram(
            'decision_latency_seconds', 'Decision coordination latency',
            buckets=LATENCY_BUCKETS
        ),
        'agent_query_latency': reg.histogram(
            'agent_query_latency_seconds', 'Per-agent query latency',
            buckets=LATENCY_BUCKETS
        ),

        # Risk events
        'risk_breaches': reg.counter(
            'risk_breaches_total', 'Total risk limit breaches'
        ),
        'kill_switch_triggers': reg.counter(
            'kill_switch_triggers_total', 'Kill switch activations'
        ),
        'news_blocks': reg.counter(
            'news_blocks_total', 'Trades blocked by news events'
        ),
    }
