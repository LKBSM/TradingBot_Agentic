# =============================================================================
# AGENTIC AI MODULE - Trading Bot Agent Framework v4.1
# =============================================================================
# This module implements a professional multi-agent architecture for autonomous
# trading decisions. Each agent specializes in a specific domain and
# communicates through a standardized event system.
#
# === INTELLIGENT AGENTS (v4.0) ===
# - IntelligentRiskSentinel: ML-powered risk prediction and adaptive sizing
# - MarketRegimeAgent: Real-time market regime detection
# - RiskSentinelAgent: Rule-based risk management (legacy, still available)
# - Sprint 1: Portfolio Risk, Kill Switch, Audit Logger
# - Sprint 2: Sentiment Analysis, Regime Prediction, Multi-Timeframe, Ensemble ML
#
# Version: 4.1.0
# Author: TradingBot Team
# License: Proprietary - Commercial Use
# =============================================================================

import logging
import warnings

logger = logging.getLogger(__name__)

# =============================================================================
# CORE COMPONENTS (Always Available)
# =============================================================================

from src.agents.base_agent import BaseAgent, AgentState, AgentCapability
from src.agents.events import (
    AgentEvent,
    EventType,
    EventBus,
    TradeProposal,
    RiskAssessment,
    AgentDecision,
    RiskViolation,
    RiskLevel,
    DecisionType
)
from src.agents.risk_sentinel import RiskSentinelAgent, create_risk_sentinel
from src.agents.config import AgentConfig, RiskSentinelConfig, ConfigPreset

# =============================================================================
# GYMNASIUM-DEPENDENT COMPONENTS (Optional)
# =============================================================================

_HAS_GYMNASIUM = False
AgenticTradingEnv = None
create_agentic_env = None
wrap_existing_env = None
AgentOrchestrator = None

try:
    from src.agents.integration import (
        AgenticTradingEnv,
        create_agentic_env,
        wrap_existing_env,
        AgentOrchestrator
    )
    _HAS_GYMNASIUM = True
except ImportError as e:
    logger.warning(f"Gymnasium not available, integration module disabled: {e}")


# =============================================================================
# INTELLIGENT AGENTS (v2.0)
# =============================================================================

try:
    from src.agents.intelligent_risk_sentinel import (
        IntelligentRiskSentinel,
        create_intelligent_risk_sentinel,
        RiskPredictor,
        AdaptivePositionSizer,
        MarketRegimeDetector,
        MarketRegime
    )
except ImportError as e:
    logger.warning(f"IntelligentRiskSentinel not available: {e}")
    IntelligentRiskSentinel = None
    create_intelligent_risk_sentinel = None
    RiskPredictor = None
    AdaptivePositionSizer = None
    MarketRegimeDetector = None
    MarketRegime = None

try:
    from src.agents.market_regime_agent import (
        MarketRegimeAgent,
        create_market_regime_agent,
        RegimeType,
        RegimeAnalysis,
        TrendDirection,
        VolatilityState
    )
except ImportError as e:
    logger.warning(f"MarketRegimeAgent not available: {e}")
    MarketRegimeAgent = None
    create_market_regime_agent = None
    RegimeType = None
    RegimeAnalysis = None
    TrendDirection = None
    VolatilityState = None

# =============================================================================
# INTELLIGENT INTEGRATION (Gymnasium-dependent)
# =============================================================================

IntelligentAgenticEnv = None
create_intelligent_env = None
upgrade_to_intelligent = None

if _HAS_GYMNASIUM:
    try:
        from src.agents.intelligent_integration import (
            IntelligentAgenticEnv,
            create_intelligent_env,
            upgrade_to_intelligent
        )
    except ImportError as e:
        logger.warning(f"IntelligentIntegration not available: {e}")

# =============================================================================
# NEWS ANALYSIS AGENT (v2.1)
# =============================================================================

try:
    from src.agents.news_analysis_agent import (
        NewsAnalysisAgent,
        NewsAgentConfig,
        NewsAssessment,
        NewsDecision,
        create_news_analysis_agent
    )
except ImportError as e:
    logger.warning(f"NewsAnalysisAgent not available: {e}")
    NewsAnalysisAgent = None
    NewsAgentConfig = None
    NewsAssessment = None
    NewsDecision = None
    create_news_analysis_agent = None

# =============================================================================
# TRADING ORCHESTRATOR (v2.1)
# =============================================================================

try:
    from src.agents.orchestrator import (
        TradingOrchestrator,
        OrchestratorConfig,
        AgentPriority,
        OrchestratedDecision,
        create_trading_orchestrator
    )
except ImportError as e:
    logger.warning(f"TradingOrchestrator not available: {e}")
    TradingOrchestrator = None
    OrchestratorConfig = None
    AgentPriority = None
    OrchestratedDecision = None
    create_trading_orchestrator = None

# =============================================================================
# ORCHESTRATED INTEGRATION (Gymnasium-dependent)
# =============================================================================

OrchestratedTradingEnv = None
create_orchestrated_env = None
upgrade_to_orchestrated = None

if _HAS_GYMNASIUM:
    try:
        from src.agents.orchestrated_integration import (
            OrchestratedTradingEnv,
            create_orchestrated_env,
            upgrade_to_orchestrated
        )
    except ImportError as e:
        logger.warning(f"OrchestratedIntegration not available: {e}")

# =============================================================================
# NEWS MODULE COMPONENTS
# =============================================================================

SentimentAnalyzer = None
EconomicCalendarFetcher = None
EconomicEvent = None
NewsHeadlineFetcher = None
NewsItem = None
NewsImpact = None

try:
    from src.agents.news import (
        SentimentAnalyzer,
        EconomicCalendarFetcher,
        EconomicEvent,
        NewsHeadlineFetcher,
        NewsItem
    )
    from src.agents.news.economic_calendar import NewsImpact
except ImportError as e:
    logger.debug(f"News module components not available: {e}")

# =============================================================================
# SPRINT 1: INSTITUTIONAL-GRADE RISK MANAGEMENT (v3.0)
# =============================================================================

try:
    from src.agents.portfolio_risk import (
        PortfolioRiskManager,
        VaRCalculator,
        VaRMethod,
        VaRResult,
        CorrelationEngine,
        ExposureManager,
        StressTester,
        Position,
        RiskLimits,
        ExposureReport,
        CorrelationAlert,
        create_portfolio_risk_manager
    )
except ImportError as e:
    logger.warning(f"PortfolioRisk not available: {e}")
    PortfolioRiskManager = None
    VaRCalculator = None
    VaRMethod = None
    VaRResult = None
    CorrelationEngine = None
    ExposureManager = None
    StressTester = None
    Position = None
    RiskLimits = None
    ExposureReport = None
    CorrelationAlert = None
    create_portfolio_risk_manager = None

try:
    from src.agents.kill_switch import (
        KillSwitch,
        KillSwitchConfig,
        CircuitBreaker,
        CircuitBreakerConfig,
        HaltLevel,
        HaltReason,
        BreakerState,
        RecoveryManager,
        RecoveryState,
        AlertManager,
        create_kill_switch
    )
except ImportError as e:
    logger.warning(f"KillSwitch not available: {e}")
    KillSwitch = None
    KillSwitchConfig = None
    CircuitBreaker = None
    CircuitBreakerConfig = None
    HaltLevel = None
    HaltReason = None
    BreakerState = None
    RecoveryManager = None
    RecoveryState = None
    AlertManager = None
    create_kill_switch = None

try:
    from src.agents.audit_logger import (
        AuditLogger,
        AuditRecord,
        DecisionAuditRecord,
        TradeAuditRecord,
        RiskAuditRecord,
        AuditEventType,
        LogLevel,
        ExportFormat,
        create_audit_logger,
        get_audit_logger,
        set_audit_logger
    )
except ImportError as e:
    logger.warning(f"AuditLogger not available: {e}")
    AuditLogger = None
    AuditRecord = None
    DecisionAuditRecord = None
    TradeAuditRecord = None
    RiskAuditRecord = None
    AuditEventType = None
    LogLevel = None
    ExportFormat = None
    create_audit_logger = None
    get_audit_logger = None
    set_audit_logger = None

try:
    from src.agents.risk_integration import (
        IntegratedRiskManager,
        IntegratedRiskConfig,
        IntegratedRiskResult,
        RiskDecision,
        RiskSentinelAdapter,
        create_integrated_risk_manager
    )
except ImportError as e:
    logger.warning(f"RiskIntegration not available: {e}")
    IntegratedRiskManager = None
    IntegratedRiskConfig = None
    IntegratedRiskResult = None
    RiskDecision = None
    RiskSentinelAdapter = None
    create_integrated_risk_manager = None

# =============================================================================
# SPRINT 2: INTELLIGENCE ENHANCEMENT (v4.0)
# =============================================================================

# Sentiment Analyzer (FinBERT NLP)
FinBERTSentimentAnalyzer = None
SentimentResult = None
AggregatedSentiment = None
SentimentCategory = None
SentimentConfig = None
create_sentiment_analyzer = None

try:
    from src.agents.sentiment_analyzer import (
        SentimentAnalyzer as FinBERTSentimentAnalyzer,
        SentimentResult,
        AggregatedSentiment,
        SentimentCategory,
        SentimentConfig,
        create_sentiment_analyzer
    )
except ImportError as e:
    logger.debug(f"SentimentAnalyzer not available: {e}")

# Regime Predictor (HMM)
RegimePredictor = None
RegimePrediction = None
HMMMarketRegime = None
RegimeConfig = None
create_regime_predictor = None

try:
    from src.agents.regime_predictor import (
        RegimePredictor,
        RegimePrediction,
        MarketRegime as HMMMarketRegime,
        RegimeConfig,
        create_regime_predictor
    )
except ImportError as e:
    logger.debug(f"RegimePredictor not available: {e}")

# Multi-Timeframe Engine
MultiTimeframeEngine = None
TimeframeAnalyzer = None
AlignmentResult = None
TimeframeAnalysis = None
Timeframe = None
SignalType = None
SignalStrength = None
TrendState = None
MomentumState = None
OHLCV = None
MultiTimeframeConfig = None
TechnicalIndicators = None
create_multi_timeframe_engine = None

try:
    from src.agents.multi_timeframe import (
        MultiTimeframeEngine,
        TimeframeAnalyzer,
        AlignmentResult,
        TimeframeAnalysis,
        Timeframe,
        SignalType,
        SignalStrength,
        TrendState,
        MomentumState,
        OHLCV,
        MultiTimeframeConfig,
        TechnicalIndicators,
        create_multi_timeframe_engine
    )
except ImportError as e:
    logger.debug(f"MultiTimeframeEngine not available: {e}")

# Ensemble Risk Model (XGBoost/LSTM/MLP)
EnsembleRiskModel = None
EnsemblePrediction = None
EnsembleConfig = None
ModelType = None
RiskCategory = None
PredictionType = None
GradientBoostRegressor = None
NumpyLSTM = None
NumpyMLP = None
FeatureNormalizer = None
create_ensemble_risk_model = None

try:
    from src.agents.ensemble_risk_model import (
        EnsembleRiskModel,
        EnsemblePrediction,
        EnsembleConfig,
        ModelType,
        RiskCategory,
        PredictionType,
        GradientBoostRegressor,
        NumpyLSTM,
        NumpyMLP,
        FeatureNormalizer,
        create_ensemble_risk_model
    )
except ImportError as e:
    logger.debug(f"EnsembleRiskModel not available: {e}")

# Sprint 2 Intelligence (Unified Integration)
Sprint2Intelligence = None
Sprint2Config = None
IntelligenceReport = None
ComponentSignal = None
TradingAction = None
ConfidenceLevel = None
MarketCondition = None
Sprint2RiskLevel = None
create_sprint2_intelligence = None

try:
    from src.agents.sprint2_intelligence import (
        Sprint2Intelligence,
        Sprint2Config,
        IntelligenceReport,
        ComponentSignal,
        TradingAction,
        ConfidenceLevel,
        MarketCondition,
        RiskLevel as Sprint2RiskLevel,
        create_sprint2_intelligence
    )
except ImportError as e:
    logger.debug(f"Sprint2Intelligence not available: {e}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_dependencies() -> dict:
    """Check which optional dependencies are available."""
    return {
        'gymnasium': _HAS_GYMNASIUM,
        'portfolio_risk': PortfolioRiskManager is not None,
        'kill_switch': KillSwitch is not None,
        'audit_logger': AuditLogger is not None,
        'sentiment_analyzer': FinBERTSentimentAnalyzer is not None,
        'regime_predictor': RegimePredictor is not None,
        'multi_timeframe': MultiTimeframeEngine is not None,
        'ensemble_model': EnsembleRiskModel is not None,
        'sprint2_intelligence': Sprint2Intelligence is not None,
        'orchestrator': TradingOrchestrator is not None,
    }


def get_available_agents() -> list:
    """Return list of available agent classes."""
    agents = [
        ('BaseAgent', BaseAgent),
        ('RiskSentinelAgent', RiskSentinelAgent),
    ]

    if IntelligentRiskSentinel:
        agents.append(('IntelligentRiskSentinel', IntelligentRiskSentinel))
    if MarketRegimeAgent:
        agents.append(('MarketRegimeAgent', MarketRegimeAgent))
    if NewsAnalysisAgent:
        agents.append(('NewsAnalysisAgent', NewsAnalysisAgent))
    if TradingOrchestrator:
        agents.append(('TradingOrchestrator', TradingOrchestrator))

    return agents


# =============================================================================
# __all__ EXPORTS
# =============================================================================

__all__ = [
    # Core (Always available)
    'BaseAgent',
    'AgentState',
    'AgentCapability',
    'AgentEvent',
    'EventType',
    'EventBus',
    'TradeProposal',
    'RiskAssessment',
    'AgentDecision',
    'RiskViolation',
    'RiskLevel',
    'DecisionType',
    'RiskSentinelAgent',
    'create_risk_sentinel',
    'AgentConfig',
    'RiskSentinelConfig',
    'ConfigPreset',

    # Gymnasium-dependent (may be None)
    'AgenticTradingEnv',
    'create_agentic_env',
    'wrap_existing_env',
    'AgentOrchestrator',
    'IntelligentAgenticEnv',
    'create_intelligent_env',
    'upgrade_to_intelligent',
    'OrchestratedTradingEnv',
    'create_orchestrated_env',
    'upgrade_to_orchestrated',

    # Intelligent Agents
    'IntelligentRiskSentinel',
    'create_intelligent_risk_sentinel',
    'RiskPredictor',
    'AdaptivePositionSizer',
    'MarketRegimeDetector',
    'MarketRegime',
    'MarketRegimeAgent',
    'create_market_regime_agent',
    'RegimeType',
    'RegimeAnalysis',
    'TrendDirection',
    'VolatilityState',

    # News Analysis
    'NewsAnalysisAgent',
    'NewsAgentConfig',
    'NewsAssessment',
    'NewsDecision',
    'create_news_analysis_agent',
    'SentimentAnalyzer',
    'EconomicCalendarFetcher',
    'EconomicEvent',
    'NewsHeadlineFetcher',
    'NewsItem',
    'NewsImpact',

    # Orchestrator
    'TradingOrchestrator',
    'OrchestratorConfig',
    'AgentPriority',
    'OrchestratedDecision',
    'create_trading_orchestrator',

    # Sprint 1: Risk Management
    'PortfolioRiskManager',
    'VaRCalculator',
    'VaRMethod',
    'VaRResult',
    'CorrelationEngine',
    'ExposureManager',
    'StressTester',
    'Position',
    'RiskLimits',
    'ExposureReport',
    'CorrelationAlert',
    'create_portfolio_risk_manager',
    'KillSwitch',
    'KillSwitchConfig',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'HaltLevel',
    'HaltReason',
    'BreakerState',
    'RecoveryManager',
    'RecoveryState',
    'AlertManager',
    'create_kill_switch',
    'AuditLogger',
    'AuditRecord',
    'DecisionAuditRecord',
    'TradeAuditRecord',
    'RiskAuditRecord',
    'AuditEventType',
    'LogLevel',
    'ExportFormat',
    'create_audit_logger',
    'get_audit_logger',
    'set_audit_logger',
    'IntegratedRiskManager',
    'IntegratedRiskConfig',
    'IntegratedRiskResult',
    'RiskDecision',
    'RiskSentinelAdapter',
    'create_integrated_risk_manager',

    # Sprint 2: Intelligence
    'FinBERTSentimentAnalyzer',
    'SentimentResult',
    'AggregatedSentiment',
    'SentimentCategory',
    'SentimentConfig',
    'create_sentiment_analyzer',
    'RegimePredictor',
    'RegimePrediction',
    'HMMMarketRegime',
    'RegimeConfig',
    'create_regime_predictor',
    'MultiTimeframeEngine',
    'TimeframeAnalyzer',
    'AlignmentResult',
    'TimeframeAnalysis',
    'Timeframe',
    'SignalType',
    'SignalStrength',
    'TrendState',
    'MomentumState',
    'OHLCV',
    'MultiTimeframeConfig',
    'TechnicalIndicators',
    'create_multi_timeframe_engine',
    'EnsembleRiskModel',
    'EnsemblePrediction',
    'EnsembleConfig',
    'ModelType',
    'RiskCategory',
    'PredictionType',
    'GradientBoostRegressor',
    'NumpyLSTM',
    'NumpyMLP',
    'FeatureNormalizer',
    'create_ensemble_risk_model',
    'Sprint2Intelligence',
    'Sprint2Config',
    'IntelligenceReport',
    'ComponentSignal',
    'TradingAction',
    'ConfidenceLevel',
    'MarketCondition',
    'Sprint2RiskLevel',
    'create_sprint2_intelligence',

    # Helper functions
    'check_dependencies',
    'get_available_agents',
]

__version__ = '4.1.0'  # Pre-Sprint 3 - Robust imports
__author__ = 'TradingBot Team'
