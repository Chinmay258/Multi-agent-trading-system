-- =============================================================================
-- infrastructure/postgres/init.sql
-- Runs once when the PostgreSQL container is first created.
-- =============================================================================

-- Enable TimescaleDB extension (must be first)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- OHLCV candles — TimescaleDB hypertable for time-series performance
-- =============================================================================
CREATE TABLE IF NOT EXISTS ohlcv_candles (
    id          BIGSERIAL,
    symbol      VARCHAR(20)     NOT NULL,
    timeframe   VARCHAR(5)      NOT NULL,
    timestamp   TIMESTAMPTZ     NOT NULL,
    open        NUMERIC(20, 8)  NOT NULL,
    high        NUMERIC(20, 8)  NOT NULL,
    low         NUMERIC(20, 8)  NOT NULL,
    close       NUMERIC(20, 8)  NOT NULL,
    volume      NUMERIC(30, 8)  NOT NULL,
    quote_volume NUMERIC(30, 8),
    received_at TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, timestamp)
);

-- Convert to hypertable (partitioned by timestamp)
SELECT create_hypertable(
    'ohlcv_candles',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Compression policy: compress chunks older than 7 days
SELECT add_compression_policy('ohlcv_candles', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention policy: drop data older than 90 days (adjust per strategy)
-- SELECT add_retention_policy('ohlcv_candles', INTERVAL '90 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe ON ohlcv_candles (symbol, timeframe, timestamp DESC);

-- =============================================================================
-- Trade proposals
-- =============================================================================
CREATE TABLE IF NOT EXISTS trade_proposals (
    proposal_id     UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol          VARCHAR(20)     NOT NULL,
    side            VARCHAR(10)     NOT NULL,   -- buy | sell
    order_type      VARCHAR(20)     NOT NULL,
    requested_size_usd NUMERIC(20, 2) NOT NULL,
    suggested_stop_loss_pct NUMERIC(10, 6),
    suggested_take_profit_pct NUMERIC(10, 6),
    signal_direction VARCHAR(20)    NOT NULL,
    signal_confidence NUMERIC(5, 4) NOT NULL,
    reasoning       TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Risk assessments
-- =============================================================================
CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id   UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    proposal_id     UUID            NOT NULL REFERENCES trade_proposals(proposal_id),
    decision        VARCHAR(20)     NOT NULL,   -- approved | modified | rejected
    rejection_reason VARCHAR(50),
    rejection_detail TEXT,
    approved_size_usd NUMERIC(20, 2),
    approved_stop_loss_pct NUMERIC(10, 6),
    portfolio_value_usd NUMERIC(20, 2),
    current_daily_loss_pct NUMERIC(10, 6),
    open_positions_count INT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Executions
-- =============================================================================
CREATE TABLE IF NOT EXISTS executions (
    result_id           UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    proposal_id         UUID            NOT NULL REFERENCES trade_proposals(proposal_id),
    assessment_id       UUID            NOT NULL REFERENCES risk_assessments(assessment_id),
    exchange_order_id   VARCHAR(100),
    symbol              VARCHAR(20)     NOT NULL,
    side                VARCHAR(10)     NOT NULL,
    order_type          VARCHAR(20)     NOT NULL,
    status              VARCHAR(30)     NOT NULL,
    requested_quantity  NUMERIC(30, 8)  NOT NULL,
    filled_quantity     NUMERIC(30, 8)  NOT NULL DEFAULT 0,
    average_fill_price  NUMERIC(20, 8),
    total_cost_usd      NUMERIC(20, 2),
    fee_usd             NUMERIC(20, 8),
    fee_currency        VARCHAR(10),
    is_paper            BOOLEAN         NOT NULL DEFAULT TRUE,
    error_message       TEXT,
    retry_count         INT             NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Positions (current open positions)
-- =============================================================================
CREATE TABLE IF NOT EXISTS positions (
    position_id         UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol              VARCHAR(20)     NOT NULL UNIQUE,
    side                VARCHAR(10)     NOT NULL,
    quantity            NUMERIC(30, 8)  NOT NULL,
    entry_price         NUMERIC(20, 8)  NOT NULL,
    current_price       NUMERIC(20, 8),
    unrealised_pnl_usd  NUMERIC(20, 2),
    stop_loss_price     NUMERIC(20, 8),
    take_profit_price   NUMERIC(20, 8),
    is_paper            BOOLEAN         NOT NULL DEFAULT TRUE,
    opened_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Agent heartbeats (for monitoring / dashboards)
-- =============================================================================
CREATE TABLE IF NOT EXISTS agent_heartbeats (
    id              BIGSERIAL       PRIMARY KEY,
    agent_name      VARCHAR(50)     NOT NULL,
    status          VARCHAR(20)     NOT NULL,
    messages_processed BIGINT       NOT NULL DEFAULT 0,
    errors_since_start BIGINT       NOT NULL DEFAULT 0,
    uptime_seconds  NUMERIC(15, 2),
    recorded_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'agent_heartbeats',
    'recorded_at',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_heartbeats_agent ON agent_heartbeats (agent_name, recorded_at DESC);

-- =============================================================================
-- System alerts
-- =============================================================================
CREATE TABLE IF NOT EXISTS system_alerts (
    alert_id        UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type      VARCHAR(50)     NOT NULL,
    severity        VARCHAR(20)     NOT NULL,
    message         TEXT            NOT NULL,
    agent_name      VARCHAR(50),
    resolved        BOOLEAN         NOT NULL DEFAULT FALSE,
    resolved_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
