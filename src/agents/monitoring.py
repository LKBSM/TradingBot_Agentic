# =============================================================================
# AGENTIC MONITORING - Real-time Agent Dashboard and Analytics
# =============================================================================
# This module provides professional monitoring capabilities for the Agentic
# trading system. It includes:
#
#   1. Real-time CLI dashboard with Rich formatting
#   2. Statistics collection and aggregation
#   3. Alert system for critical events
#   4. Export capabilities (JSON, CSV)
#
# === USAGE ===
#
#   from src.agents.monitoring import AgentMonitor, print_live_dashboard
#
#   # Create monitor
#   monitor = AgentMonitor()
#   monitor.attach(risk_sentinel)
#
#   # Print dashboard
#   print_live_dashboard(monitor)
#
# =============================================================================

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import json
import csv
import io

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class DecisionRecord:
    """
    Record of a single agent decision for historical tracking.

    Used for:
        - Audit trails
        - Performance analysis
        - Pattern detection
    """
    timestamp: datetime
    agent_id: str
    action_proposed: str
    action_approved: str
    decision: str  # APPROVE, REJECT, MODIFY
    risk_score: float
    risk_level: str
    rejection_reason: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class AgentSnapshot:
    """
    Point-in-time snapshot of agent state for monitoring.
    """
    timestamp: datetime
    agent_id: str
    state: str
    total_decisions: int
    approvals: int
    rejections: int
    avg_risk_score: float
    current_drawdown: float
    uptime_seconds: float


# =============================================================================
# AGENT MONITOR
# =============================================================================


class AgentMonitor:
    """
    Comprehensive monitoring system for trading agents.

    Collects metrics, tracks decisions, and provides dashboards.

    === FEATURES ===
    1. Real-time metrics collection
    2. Decision history with rolling window
    3. Alert system for critical events
    4. Export to JSON/CSV
    5. Rich CLI dashboard (if Rich installed)

    === USAGE ===
    ```python
    monitor = AgentMonitor()
    monitor.attach(risk_sentinel)

    # Get current stats
    stats = monitor.get_stats()

    # Print dashboard
    monitor.print_dashboard()

    # Export history
    monitor.export_to_json("decisions.json")
    ```
    """

    def __init__(self, history_size: int = 1000):
        """
        Initialize the agent monitor.

        Args:
            history_size: Maximum number of decisions to keep in history
        """
        self._agents: Dict[str, Any] = {}
        self._decision_history: deque = deque(maxlen=history_size)
        self._snapshots: deque = deque(maxlen=100)
        self._alerts: List[Dict[str, Any]] = []
        self._start_time = datetime.now()

        # Alert thresholds
        self._drawdown_alert_threshold = 0.07  # 7%
        self._rejection_rate_alert_threshold = 0.8  # 80%
        self._risk_score_alert_threshold = 75  # Out of 100

        # Console for Rich output
        if RICH_AVAILABLE:
            self._console = Console()

    def attach(self, agent: 'BaseAgent') -> None:
        """
        Attach an agent to the monitor.

        Args:
            agent: Agent instance to monitor
        """
        self._agents[agent.full_id] = agent

    def detach(self, agent_id: str) -> None:
        """
        Detach an agent from the monitor.

        Args:
            agent_id: ID of agent to detach
        """
        if agent_id in self._agents:
            del self._agents[agent_id]

    def record_decision(
        self,
        agent_id: str,
        action_proposed: int,
        action_approved: int,
        decision: str,
        risk_score: float,
        risk_level: str,
        rejection_reason: Optional[str] = None,
        processing_time_ms: float = 0.0
    ) -> None:
        """
        Record a decision for historical tracking.

        Args:
            agent_id: ID of the agent that made the decision
            action_proposed: Original action from RL agent
            action_approved: Action after risk gate
            decision: Decision type (APPROVE, REJECT, etc.)
            risk_score: Risk score assigned
            risk_level: Risk level category
            rejection_reason: Reason if rejected
            processing_time_ms: Time to process
        """
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}

        record = DecisionRecord(
            timestamp=datetime.now(),
            agent_id=agent_id,
            action_proposed=action_map.get(action_proposed, str(action_proposed)),
            action_approved=action_map.get(action_approved, str(action_approved)),
            decision=decision,
            risk_score=risk_score,
            risk_level=risk_level,
            rejection_reason=rejection_reason,
            processing_time_ms=processing_time_ms
        )

        self._decision_history.append(record)

        # Check for alerts
        self._check_alerts(record)

    def take_snapshot(self) -> None:
        """
        Take a point-in-time snapshot of all agents.

        Call periodically for trend analysis.
        """
        for agent_id, agent in self._agents.items():
            if hasattr(agent, 'get_statistics'):
                stats = agent.get_statistics()
                snapshot = AgentSnapshot(
                    timestamp=datetime.now(),
                    agent_id=agent_id,
                    state=agent.state.name,
                    total_decisions=stats.get('total_assessments', 0),
                    approvals=stats.get('total_approvals', 0),
                    rejections=stats.get('total_rejections', 0),
                    avg_risk_score=0.0,  # Calculate from history if needed
                    current_drawdown=float(stats.get('current_drawdown', '0%').rstrip('%')) / 100,
                    uptime_seconds=(datetime.now() - self._start_time).total_seconds()
                )
                self._snapshots.append(snapshot)

    def _check_alerts(self, record: DecisionRecord) -> None:
        """Check if a decision triggers any alerts."""
        # High risk score alert
        if record.risk_score >= self._risk_score_alert_threshold:
            self._alerts.append({
                'type': 'HIGH_RISK_SCORE',
                'timestamp': datetime.now().isoformat(),
                'message': f"High risk score: {record.risk_score:.0f}/100",
                'severity': 'WARNING'
            })

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics from all monitored agents.

        Returns:
            Dictionary with comprehensive statistics
        """
        stats = {
            'monitor_uptime': str(datetime.now() - self._start_time),
            'agents_monitored': len(self._agents),
            'total_decisions_recorded': len(self._decision_history),
            'active_alerts': len(self._alerts),
            'agents': {}
        }

        # Collect per-agent stats
        for agent_id, agent in self._agents.items():
            if hasattr(agent, 'get_statistics'):
                stats['agents'][agent_id] = agent.get_statistics()
            else:
                stats['agents'][agent_id] = {'status': agent.state.name}

        # Calculate decision statistics from history
        if self._decision_history:
            recent = list(self._decision_history)[-100:]  # Last 100
            approvals = sum(1 for d in recent if d.decision == 'APPROVE')
            rejections = sum(1 for d in recent if d.decision == 'REJECT')
            total = approvals + rejections

            stats['recent_approval_rate'] = f"{approvals / total * 100:.1f}%" if total > 0 else "N/A"
            stats['avg_risk_score'] = sum(d.risk_score for d in recent) / len(recent)
            stats['avg_processing_time_ms'] = sum(d.processing_time_ms for d in recent) / len(recent)

            # Top rejection reasons
            reasons = [d.rejection_reason for d in recent if d.rejection_reason]
            from collections import Counter
            stats['top_rejection_reasons'] = dict(Counter(reasons).most_common(5))

        return stats

    # =========================================================================
    # DASHBOARD
    # =========================================================================

    def print_dashboard(self) -> None:
        """
        Print a formatted dashboard to the console.

        Uses Rich library for beautiful formatting if available,
        falls back to plain text otherwise.
        """
        if RICH_AVAILABLE:
            self._print_rich_dashboard()
        else:
            self._print_plain_dashboard()

    def _print_rich_dashboard(self) -> None:
        """Print dashboard using Rich library."""
        console = self._console
        stats = self.get_stats()

        # Clear screen
        console.clear()

        # Header
        console.print(Panel.fit(
            "[bold cyan]AGENTIC AI TRADING SYSTEM - RISK MONITOR[/bold cyan]",
            border_style="cyan"
        ))

        # System Status Table
        system_table = Table(title="System Status", box=box.ROUNDED)
        system_table.add_column("Metric", style="cyan")
        system_table.add_column("Value", style="green")

        system_table.add_row("Uptime", str(stats.get('monitor_uptime', 'N/A')))
        system_table.add_row("Agents Active", str(stats.get('agents_monitored', 0)))
        system_table.add_row("Decisions Tracked", str(stats.get('total_decisions_recorded', 0)))
        system_table.add_row("Active Alerts", str(stats.get('active_alerts', 0)))

        console.print(system_table)
        console.print()

        # Agent Details
        for agent_id, agent_stats in stats.get('agents', {}).items():
            self._print_agent_panel(agent_id, agent_stats)

        # Recent Decision Stats
        if 'recent_approval_rate' in stats:
            decision_table = Table(title="Recent Decisions (Last 100)", box=box.ROUNDED)
            decision_table.add_column("Metric", style="cyan")
            decision_table.add_column("Value", style="yellow")

            decision_table.add_row("Approval Rate", stats.get('recent_approval_rate', 'N/A'))
            decision_table.add_row("Avg Risk Score", f"{stats.get('avg_risk_score', 0):.1f}/100")
            decision_table.add_row("Avg Processing Time", f"{stats.get('avg_processing_time_ms', 0):.2f}ms")

            console.print(decision_table)
            console.print()

            # Top Rejection Reasons
            if stats.get('top_rejection_reasons'):
                reason_table = Table(title="Top Rejection Reasons", box=box.ROUNDED)
                reason_table.add_column("Reason", style="red")
                reason_table.add_column("Count", style="yellow")

                for reason, count in stats['top_rejection_reasons'].items():
                    reason_table.add_row(reason[:50] + "..." if len(reason) > 50 else reason, str(count))

                console.print(reason_table)

        # Alerts
        if self._alerts:
            console.print()
            console.print(Panel(
                "\n".join([f"[red]{a['type']}[/red]: {a['message']}" for a in self._alerts[-5:]]),
                title="[bold red]Recent Alerts[/bold red]",
                border_style="red"
            ))

    def _print_agent_panel(self, agent_id: str, stats: Dict[str, Any]) -> None:
        """Print a panel for a single agent."""
        console = self._console

        # Create content
        content = []
        content.append(f"[cyan]State:[/cyan] {stats.get('state', 'UNKNOWN')}")

        if 'total_assessments' in stats:
            content.append(f"[cyan]Total Assessments:[/cyan] {stats['total_assessments']}")
            content.append(f"[green]Approved:[/green] {stats.get('total_approvals', 0)}")
            content.append(f"[red]Rejected:[/red] {stats.get('total_rejections', 0)}")
            content.append(f"[cyan]Approval Rate:[/cyan] {stats.get('approval_rate', 'N/A')}")

        if 'current_drawdown' in stats:
            dd = stats['current_drawdown']
            dd_color = "red" if float(dd.rstrip('%')) > 5 else "yellow" if float(dd.rstrip('%')) > 3 else "green"
            content.append(f"[cyan]Drawdown:[/cyan] [{dd_color}]{dd}[/{dd_color}]")

        if 'current_regime' in stats:
            regime = stats['current_regime']
            regime_color = "red" if regime == 'VOLATILE' else "green"
            content.append(f"[cyan]Market Regime:[/cyan] [{regime_color}]{regime}[/{regime_color}]")

        console.print(Panel(
            "\n".join(content),
            title=f"[bold blue]{agent_id}[/bold blue]",
            border_style="blue"
        ))

    def _print_plain_dashboard(self) -> None:
        """Print dashboard using plain text (no Rich)."""
        stats = self.get_stats()

        print("\n" + "=" * 70)
        print("         AGENTIC AI TRADING SYSTEM - RISK MONITOR")
        print("=" * 70)

        print(f"\nUptime: {stats.get('monitor_uptime', 'N/A')}")
        print(f"Agents Active: {stats.get('agents_monitored', 0)}")
        print(f"Decisions Tracked: {stats.get('total_decisions_recorded', 0)}")

        for agent_id, agent_stats in stats.get('agents', {}).items():
            print(f"\n--- {agent_id} ---")
            for key, value in agent_stats.items():
                print(f"  {key}: {value}")

        if 'recent_approval_rate' in stats:
            print(f"\nRecent Approval Rate: {stats['recent_approval_rate']}")
            print(f"Avg Risk Score: {stats.get('avg_risk_score', 0):.1f}/100")

        print("\n" + "=" * 70)

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_to_json(self, filepath: str) -> None:
        """
        Export decision history to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'stats': self.get_stats(),
            'decisions': [
                {
                    'timestamp': d.timestamp.isoformat(),
                    'agent_id': d.agent_id,
                    'action_proposed': d.action_proposed,
                    'action_approved': d.action_approved,
                    'decision': d.decision,
                    'risk_score': d.risk_score,
                    'risk_level': d.risk_level,
                    'rejection_reason': d.rejection_reason,
                    'processing_time_ms': d.processing_time_ms
                }
                for d in self._decision_history
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def export_to_csv(self, filepath: str) -> None:
        """
        Export decision history to CSV file.

        Args:
            filepath: Path to output CSV file
        """
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'agent_id', 'action_proposed', 'action_approved',
                'decision', 'risk_score', 'risk_level', 'rejection_reason',
                'processing_time_ms'
            ])

            for d in self._decision_history:
                writer.writerow([
                    d.timestamp.isoformat(),
                    d.agent_id,
                    d.action_proposed,
                    d.action_approved,
                    d.decision,
                    d.risk_score,
                    d.risk_level,
                    d.rejection_reason or '',
                    d.processing_time_ms
                ])

    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent decision history as list of dicts.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of decision records
        """
        return [
            {
                'timestamp': d.timestamp.isoformat(),
                'agent_id': d.agent_id,
                'action_proposed': d.action_proposed,
                'action_approved': d.action_approved,
                'decision': d.decision,
                'risk_score': d.risk_score,
                'risk_level': d.risk_level,
                'rejection_reason': d.rejection_reason
            }
            for d in list(self._decision_history)[-limit:]
        ]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def print_live_dashboard(monitor: AgentMonitor) -> None:
    """
    Print a one-time dashboard snapshot.

    Args:
        monitor: AgentMonitor instance
    """
    monitor.print_dashboard()


def create_monitor_for_env(env: 'AgenticTradingEnv') -> AgentMonitor:
    """
    Create a monitor attached to an AgenticTradingEnv.

    Args:
        env: AgenticTradingEnv instance

    Returns:
        Configured AgentMonitor
    """
    monitor = AgentMonitor()
    monitor.attach(env.risk_sentinel)
    return monitor
