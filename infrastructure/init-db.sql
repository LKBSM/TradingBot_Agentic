-- =============================================================================
-- DATABASE INITIALIZATION - Trading Bot PostgreSQL
-- =============================================================================
-- Creates tables for:
-- - Audit logs (trade decisions, agent reasoning)
-- - Trade history
-- - Performance metrics
-- - System events
-- =============================================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- =============================================================================
-- AUDIT LOGS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decision_id UUID NOT NULL,

    -- Action details
    action_type VARCHAR(50) NOT NULL,  -- TRADE, RISK_CHECK, NEWS_BLOCK, etc.
    action_proposed TEXT,
    action_executed TEXT,

    -- Agent decisions
    agents_consulted JSONB,  -- Array of {agent, decision, confidence, reason}
    final_decision VARCHAR(50),
    final_reasoning TEXT,

    -- Context
    symbol VARCHAR(20),
    direction VARCHAR(10),  -- BUY, SELL, HOLD
    size_proposed DECIMAL(18, 8),
    size_executed DECIMAL(18, 8),
    price DECIMAL(18, 8),

    -- Portfolio state at decision time
    portfolio_state JSONB,

    -- Metadata
    session_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for audit_logs
CREATE INDEX idx_audit_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX idx_audit_decision_id ON audit_logs(decision_id);
CREATE INDEX idx_audit_symbol ON audit_logs(symbol);
CREATE INDEX idx_audit_action_type ON audit_logs(action_type);

-- =============================================================================
-- TRADES TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id VARCHAR(100) UNIQUE NOT NULL,

    -- Trade details
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL,  -- BUY, SELL
    entry_price DECIMAL(18, 8) NOT NULL,
    exit_price DECIMAL(18, 8),
    size DECIMAL(18, 8) NOT NULL,

    -- Timestamps
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,

    -- P&L
    realized_pnl DECIMAL(18, 4),
    unrealized_pnl DECIMAL(18, 4),
    commission DECIMAL(18, 4),
    swap DECIMAL(18, 4),

    -- Trade metadata
    strategy VARCHAR(100),
    signal_source VARCHAR(100),
    entry_reason TEXT,
    exit_reason TEXT,

    -- Risk at entry
    stop_loss DECIMAL(18, 8),
    take_profit DECIMAL(18, 8),
    risk_percent DECIMAL(5, 2),

    -- Market context
    market_regime VARCHAR(50),
    volatility_percentile DECIMAL(5, 2),
    news_sentiment DECIMAL(5, 2),

    -- Status
    status VARCHAR(20) DEFAULT 'OPEN',  -- OPEN, CLOSED, CANCELLED
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for trades
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_entry_time ON trades(entry_time DESC);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_strategy ON trades(strategy);

-- =============================================================================
-- PERFORMANCE METRICS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Time period
    period_type VARCHAR(20) NOT NULL,  -- HOURLY, DAILY, WEEKLY, MONTHLY
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,

    -- Performance
    total_pnl DECIMAL(18, 4),
    win_rate DECIMAL(5, 2),
    profit_factor DECIMAL(8, 4),
    sharpe_ratio DECIMAL(8, 4),
    sortino_ratio DECIMAL(8, 4),

    -- Risk
    max_drawdown_percent DECIMAL(5, 2),
    var_95 DECIMAL(18, 4),
    cvar_95 DECIMAL(18, 4),

    -- Activity
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_win DECIMAL(18, 4),
    avg_loss DECIMAL(18, 4),

    -- Exposure
    avg_exposure_percent DECIMAL(5, 2),
    max_exposure_percent DECIMAL(5, 2),

    -- By symbol breakdown
    metrics_by_symbol JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance_metrics
CREATE INDEX idx_perf_timestamp ON performance_metrics(timestamp DESC);
CREATE INDEX idx_perf_period ON performance_metrics(period_type, period_start);

-- =============================================================================
-- SYSTEM EVENTS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Event details
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,  -- INFO, WARNING, ERROR, CRITICAL
    source VARCHAR(100),
    message TEXT,

    -- Context
    details JSONB,

    -- Metadata
    session_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for system_events
CREATE INDEX idx_events_timestamp ON system_events(timestamp DESC);
CREATE INDEX idx_events_type ON system_events(event_type);
CREATE INDEX idx_events_severity ON system_events(severity);

-- =============================================================================
-- NEWS ARTICLES TABLE (for caching/analysis)
-- =============================================================================
CREATE TABLE IF NOT EXISTS news_articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id VARCHAR(100) UNIQUE NOT NULL,

    -- Content
    source_name VARCHAR(100),
    source_type VARCHAR(50),
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    url TEXT,

    -- Classification
    category VARCHAR(50),
    importance VARCHAR(20),
    assets TEXT[],  -- Array of related assets
    keywords TEXT[],

    -- Sentiment
    sentiment_score DECIMAL(5, 4),
    sentiment_label VARCHAR(20),

    -- Timestamps
    published_at TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for news_articles
CREATE INDEX idx_news_published ON news_articles(published_at DESC);
CREATE INDEX idx_news_source ON news_articles(source_name);
CREATE INDEX idx_news_assets ON news_articles USING GIN(assets);
CREATE INDEX idx_news_title_search ON news_articles USING GIN(to_tsvector('english', title));

-- =============================================================================
-- VIEWS
-- =============================================================================

-- Daily performance summary view
CREATE OR REPLACE VIEW daily_performance AS
SELECT
    DATE(entry_time) as trade_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)::DECIMAL / NULLIF(COUNT(*), 0) * 100, 2) as win_rate,
    SUM(realized_pnl) as total_pnl,
    AVG(realized_pnl) as avg_pnl,
    MAX(realized_pnl) as best_trade,
    MIN(realized_pnl) as worst_trade
FROM trades
WHERE status = 'CLOSED'
GROUP BY DATE(entry_time)
ORDER BY trade_date DESC;

-- Recent audit log summary
CREATE OR REPLACE VIEW recent_decisions AS
SELECT
    timestamp,
    action_type,
    symbol,
    direction,
    final_decision,
    final_reasoning,
    portfolio_state->>'var_95' as var_95,
    portfolio_state->>'current_drawdown_pct' as drawdown_pct
FROM audit_logs
ORDER BY timestamp DESC
LIMIT 100;

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to calculate rolling Sharpe ratio
CREATE OR REPLACE FUNCTION calculate_sharpe(
    lookback_days INTEGER DEFAULT 30
)
RETURNS DECIMAL AS $$
DECLARE
    avg_return DECIMAL;
    std_return DECIMAL;
    risk_free_rate DECIMAL := 0.0001;  -- Daily risk-free rate
BEGIN
    SELECT
        AVG(realized_pnl),
        STDDEV(realized_pnl)
    INTO avg_return, std_return
    FROM trades
    WHERE status = 'CLOSED'
      AND exit_time >= NOW() - (lookback_days || ' days')::INTERVAL;

    IF std_return IS NULL OR std_return = 0 THEN
        RETURN 0;
    END IF;

    RETURN (avg_return - risk_free_rate) / std_return * SQRT(252);
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- GRANTS
-- =============================================================================
-- (Uncomment and modify for production)
-- GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO tradingbot_app;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO tradingbot_app;
