# =============================================================================
# AGENTIC AI MODULE - Trading Bot Agent Framework
# =============================================================================
# This module implements a professional multi-agent architecture for autonomous
# trading decisions. Each agent specializes in a specific domain (risk, execution,
# research, etc.) and communicates through a standardized event system.
#
# Architecture Overview:
# ┌─────────────────────────────────────────────────────────────────┐
# │                    AGENT ORCHESTRATOR                           │
# │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
# │  │ Risk         │  │ Execution    │  │ Research     │          │
# │  │ Sentinel     │  │ Agent        │  │ Agent        │          │
# │  └──────────────┘  └──────────────┘  └──────────────┘          │
# └─────────────────────────────────────────────────────────────────┘
#
# Version: 1.0.0
# Author: TradingBot Team
# License: Proprietary - Commercial Use
# =============================================================================

from src.agents.base_agent import BaseAgent, AgentState, AgentCapability
from src.agents.events import (
    AgentEvent,
    EventType,
    EventBus,
    TradeProposal,
    RiskAssessment,
    AgentDecision
)
from src.agents.risk_sentinel import RiskSentinelAgent, create_risk_sentinel
from src.agents.config import AgentConfig, RiskSentinelConfig, ConfigPreset
from src.agents.integration import (
    AgenticTradingEnv,
    create_agentic_env,
    wrap_existing_env,
    AgentOrchestrator
)

__all__ = [
    # Base classes
    'BaseAgent',
    'AgentState',
    'AgentCapability',

    # Event system
    'AgentEvent',
    'EventType',
    'EventBus',
    'TradeProposal',
    'RiskAssessment',
    'AgentDecision',

    # Agents
    'RiskSentinelAgent',
    'create_risk_sentinel',

    # Integration
    'AgenticTradingEnv',
    'create_agentic_env',
    'wrap_existing_env',
    'AgentOrchestrator',

    # Configuration
    'AgentConfig',
    'RiskSentinelConfig',
    'ConfigPreset',
]

__version__ = '1.0.0'
__author__ = 'TradingBot Team'
