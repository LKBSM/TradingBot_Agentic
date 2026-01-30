# =============================================================================
# CORE MODULE
# =============================================================================
# Core infrastructure components for the trading bot.
#
# Components:
# - exceptions: Custom exception hierarchy
# - retry: Retry patterns with exponential backoff
# - config_loader: Lazy configuration loading
# - resource_pool: Connection pooling
#
# =============================================================================

from src.core.exceptions import (
    TradingError,
    TransientError,
    PermanentError,
    ConfigurationError,
    ValidationError,
    RiskLimitError,
    DataFeedError,
    BrokerConnectionError,
    ExecutionError,
    AgentError,
    TimeoutError as TradingTimeoutError,
)

from src.core.retry import (
    retry_with_backoff,
    RetryConfig,
    CircuitBreaker,
)

from src.core.config_loader import (
    TradingConfig,
    get_config,
    initialize_config,
    reset_config,
    is_initialized,
    get_action_name,
    get_features_list,
    get_smc_config,
    get_hyperparameter_search_space,
    get_model_hyperparameters,
    ConfigurationError as ConfigLoaderError,
)

__all__ = [
    # Exceptions
    'TradingError',
    'TransientError',
    'PermanentError',
    'ConfigurationError',
    'ValidationError',
    'RiskLimitError',
    'DataFeedError',
    'BrokerConnectionError',
    'ExecutionError',
    'AgentError',
    'TradingTimeoutError',
    # Retry
    'retry_with_backoff',
    'RetryConfig',
    'CircuitBreaker',
    # Config Loader
    'TradingConfig',
    'get_config',
    'initialize_config',
    'reset_config',
    'is_initialized',
    'get_action_name',
    'get_features_list',
    'get_smc_config',
    'get_hyperparameter_search_space',
    'get_model_hyperparameters',
    'ConfigLoaderError',
]
