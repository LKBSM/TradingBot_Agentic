# =============================================================================
# AGENTIC TRADING DEMO - Phase 1: Risk Sentinel Integration
# =============================================================================
# This script demonstrates how to use the new Agentic AI system with the
# existing trading bot. It shows:
#
#   1. Creating an AgenticTradingEnv (environment with Risk Sentinel)
#   2. Running a training session with risk gating
#   3. Monitoring agent decisions
#   4. Viewing the risk dashboard
#
# Run with: python examples/agentic_trading_demo.py
#
# =============================================================================

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the new Agentic system
from src.agents import (
    AgenticTradingEnv,
    create_agentic_env,
    RiskSentinelConfig,
    ConfigPreset
)
from src.agents.monitoring import AgentMonitor, create_monitor_for_env

# Import config
try:
    from config import HISTORICAL_DATA_FILE, FEATURES
except ImportError:
    HISTORICAL_DATA_FILE = r"C:\MyPythonProjects\TradingBotNew\data\XAU_15MIN_2019_2024.csv"
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']


def create_sample_data(n_rows: int = 1000) -> pd.DataFrame:
    """
    Create sample OHLCV data for demonstration.

    In production, you would load your actual historical data.
    """
    print("Creating sample data for demonstration...")

    np.random.seed(42)

    # Generate realistic price movements
    returns = np.random.normal(0.0001, 0.002, n_rows)
    prices = 2000 * np.cumprod(1 + returns)

    # Create OHLCV data
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=n_rows, freq='15min'),
        'Open': prices * (1 + np.random.uniform(-0.001, 0.001, n_rows)),
        'High': prices * (1 + np.random.uniform(0, 0.003, n_rows)),
        'Low': prices * (1 - np.random.uniform(0, 0.003, n_rows)),
        'Close': prices,
        'Volume': np.random.randint(100, 1000, n_rows)
    }

    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)

    print(f"Created {len(df)} rows of sample data")
    return df


def demo_basic_usage():
    """
    Demonstrate basic usage of AgenticTradingEnv.

    Shows how to:
    - Create an agentic environment
    - Step through with random actions
    - Check if actions were approved/rejected
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Basic AgenticTradingEnv Usage")
    print("=" * 70)

    # Create sample data
    df = create_sample_data(500)

    # Create agentic environment with moderate risk settings
    print("\nCreating AgenticTradingEnv with 'backtesting' preset...")
    env = create_agentic_env(
        df=df,
        risk_preset="backtesting",  # Less strict for backtesting
        initial_balance=10000.0
    )

    # Reset environment
    obs, info = env.reset()
    print(f"Environment reset. Initial info: {info.get('risk_sentinel_active')}")

    # Run a few steps with random actions
    print("\nRunning 20 steps with random actions...")
    total_approved = 0
    total_rejected = 0

    for step in range(20):
        # Random action: 0=HOLD, 1=BUY, 2=SELL
        action = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])

        obs, reward, done, truncated, info = env.step(action)

        # Check if action was approved
        if info.get('risk_approved'):
            total_approved += 1
            status = "APPROVED"
        else:
            total_rejected += 1
            status = f"REJECTED ({info.get('risk_reason', 'unknown')})"

        if action != 0:  # Only print non-HOLD actions
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            print(f"  Step {step+1}: {action_names[action]} -> {status}")

        if done:
            break

    # Print summary
    print(f"\nSummary:")
    print(f"  Approved actions: {total_approved}")
    print(f"  Rejected actions: {total_rejected}")

    # Print risk dashboard
    print("\n" + env.get_risk_dashboard())

    env.close()


def demo_custom_config():
    """
    Demonstrate using custom risk configuration.

    Shows how to:
    - Create custom RiskSentinelConfig
    - Pass it to AgenticTradingEnv
    - Compare with different presets
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Custom Risk Configuration")
    print("=" * 70)

    df = create_sample_data(300)

    # Create a very conservative config
    conservative_config = RiskSentinelConfig(
        max_drawdown_pct=0.05,         # Only 5% max drawdown
        max_risk_per_trade_pct=0.005,  # 0.5% risk per trade
        max_position_size_pct=0.10,    # 10% max position
        strict_mode=True,              # Reject on any violation
        enable_rule_explanations=True
    )

    env = AgenticTradingEnv(
        df=df,
        risk_config=conservative_config,
        initial_balance=10000.0
    )

    obs, _ = env.reset()

    # Try several trades
    print("\nTesting with conservative config (5% max drawdown)...")

    for i in range(10):
        action = 1  # Always try to BUY

        obs, reward, done, truncated, info = env.step(action)

        decision = info.get('risk_decision', 'N/A')
        reason = info.get('risk_reason', 'N/A')
        score = info.get('risk_score', 0)

        print(f"  Trade {i+1}: {decision} | Score: {score:.0f} | Reason: {reason}")

        if done:
            break

    env.close()


def demo_monitoring():
    """
    Demonstrate the monitoring system.

    Shows how to:
    - Attach a monitor to the environment
    - Collect statistics
    - Export decision history
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Agent Monitoring")
    print("=" * 70)

    df = create_sample_data(500)

    # Create environment
    env = create_agentic_env(df=df, risk_preset="moderate")

    # Create and attach monitor
    monitor = create_monitor_for_env(env)

    obs, _ = env.reset()

    # Run simulation
    print("\nRunning 50-step simulation with monitoring...")

    for step in range(50):
        action = np.random.choice([0, 1, 2], p=[0.4, 0.35, 0.25])

        obs, reward, done, truncated, info = env.step(action)

        # Record decision in monitor
        if action != 0:
            monitor.record_decision(
                agent_id=env.risk_sentinel.full_id,
                action_proposed=action,
                action_approved=info.get('risk_approved_action', action),
                decision=info.get('risk_decision', 'N/A'),
                risk_score=info.get('risk_score', 0),
                risk_level=info.get('risk_level', 'LOW'),
                rejection_reason=info.get('risk_reason')
            )

        if done:
            break

    # Print statistics
    stats = monitor.get_stats()
    print(f"\nMonitor Statistics:")
    print(f"  Total Decisions: {stats.get('total_decisions_recorded', 0)}")
    print(f"  Recent Approval Rate: {stats.get('recent_approval_rate', 'N/A')}")
    print(f"  Avg Risk Score: {stats.get('avg_risk_score', 0):.1f}")

    # Print dashboard
    monitor.print_dashboard()

    # Export to JSON (optional)
    # monitor.export_to_json("decision_history.json")

    env.close()


def demo_with_real_data():
    """
    Demonstrate with real historical data (if available).

    This shows production-like usage.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Real Data Integration (if available)")
    print("=" * 70)

    # Try to load real data
    if os.path.exists(HISTORICAL_DATA_FILE):
        print(f"Loading real data from: {HISTORICAL_DATA_FILE}")
        try:
            df = pd.read_csv(HISTORICAL_DATA_FILE)
            print(f"Loaded {len(df)} rows")

            # Take a subset for demo
            df_subset = df.iloc[:2000].copy()

            # Create agentic environment
            env = create_agentic_env(
                df=df_subset,
                risk_preset="backtesting"
            )

            obs, _ = env.reset()

            # Quick simulation
            print("\nRunning 100-step simulation with real data...")

            for step in range(100):
                action = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
                obs, reward, done, truncated, info = env.step(action)
                if done:
                    break

            # Print summary
            env.print_risk_summary()

            env.close()

        except Exception as e:
            print(f"Error loading real data: {e}")
            print("Using sample data instead...")
            demo_basic_usage()
    else:
        print(f"Real data file not found: {HISTORICAL_DATA_FILE}")
        print("Using sample data instead...")


def demo_risk_sentinel_standalone():
    """
    Demonstrate using RiskSentinel without environment wrapper.

    Useful for:
    - Testing rules in isolation
    - Integrating with custom environments
    - Research and analysis
    """
    print("\n" + "=" * 70)
    print("DEMO 5: Standalone Risk Sentinel")
    print("=" * 70)

    from src.agents import RiskSentinelAgent, create_risk_sentinel
    from src.agents.events import TradeProposal

    # Create sentinel
    sentinel = create_risk_sentinel(preset="moderate")
    sentinel.start()

    # Create some test proposals
    proposals = [
        # Normal trade
        TradeProposal(
            action="BUY",
            quantity=0.1,
            entry_price=2000.0,
            current_equity=10000.0,
            current_balance=10000.0,
            market_data={'ATR': 20.0}
        ),
        # High leverage trade
        TradeProposal(
            action="BUY",
            quantity=10.0,  # Would be 2x leverage
            entry_price=2000.0,
            current_equity=10000.0,
            current_balance=10000.0,
            market_data={'ATR': 20.0}
        ),
        # Low balance trade
        TradeProposal(
            action="BUY",
            quantity=0.01,
            entry_price=2000.0,
            current_equity=50.0,
            current_balance=50.0,
            market_data={'ATR': 20.0}
        ),
    ]

    print("\nEvaluating trade proposals:")
    print("-" * 60)

    for i, proposal in enumerate(proposals, 1):
        assessment = sentinel.evaluate_trade(proposal)
        print(f"\nProposal {i}: {proposal.action} qty={proposal.quantity}")
        print(f"  Decision: {assessment.decision.name}")
        print(f"  Risk Score: {assessment.risk_score:.0f}/100")
        print(f"  Risk Level: {assessment.risk_level.name}")

        if assessment.violations:
            print(f"  Violations:")
            for v in assessment.violations[:3]:  # Show first 3
                print(f"    - {v.rule_name}: {v.rule_description[:60]}...")

    # Print dashboard
    print("\n" + sentinel.get_risk_dashboard())

    sentinel.stop()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("      AGENTIC AI TRADING SYSTEM - PHASE 1 DEMONSTRATION")
    print("             Risk Sentinel Agent Integration")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run demos
    demo_basic_usage()
    demo_custom_config()
    demo_monitoring()
    demo_risk_sentinel_standalone()

    # Optionally run with real data
    # demo_with_real_data()

    print("\n" + "=" * 70)
    print("All demonstrations completed!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Integrate AgenticTradingEnv into your training pipeline")
    print("  2. Customize RiskSentinelConfig for your risk tolerance")
    print("  3. Use the monitor to track agent decisions")
    print("  4. Export decision history for analysis")
    print("\nFor production, use:")
    print("  env = create_agentic_env(df, risk_preset='moderate')")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
