-- feature_store_schema.sql
-- Feature Store Schema for Breakout Prediction System

-- Feature Group Registry
CREATE TABLE feature_groups (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    version VARCHAR(20) NOT NULL,
    schema_definition JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Feature definitions and metadata
CREATE TABLE feature_definitions (
    id SERIAL PRIMARY KEY,
    feature_group_id INTEGER REFERENCES feature_groups(id),
    feature_name VARCHAR(100) NOT NULL,
    feature_type VARCHAR(50) NOT NULL, -- 'float', 'int', 'boolean', 'string'
    description TEXT,
    calculation_logic TEXT,
    dependencies JSONB, -- Other features this depends on
    validation_rules JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(feature_group_id, feature_name)
);

-- Technical indicator features (optimized for fast retrieval)
CREATE TABLE technical_features (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Price-based features
    price_close FLOAT NOT NULL,
    price_open FLOAT,
    price_high FLOAT,
    price_low FLOAT,
    volume BIGINT,
    
    -- Moving averages
    sma_10 FLOAT,
    sma_20 FLOAT,
    sma_50 FLOAT,
    ema_10 FLOAT,
    ema_20 FLOAT,
    ema_50 FLOAT,
    
    -- Volatility indicators
    atr_14 FLOAT,
    atr_normalized FLOAT,
    std_dev_20 FLOAT,
    volatility_1d FLOAT,
    volatility_10d FLOAT,
    
    -- Momentum indicators
    rsi_14 FLOAT,
    macd FLOAT,
    macd_signal FLOAT,
    macd_histogram FLOAT,
    stochastic_k FLOAT,
    stochastic_d FLOAT,
    
    -- Volume indicators
    volume_sma_20 FLOAT,
    volume_ratio FLOAT,
    obv FLOAT,
    cmf_20 FLOAT,
    mfi_14 FLOAT,
    vwap FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Consolidation-specific features
CREATE TABLE consolidation_features (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Keltner Channel features
    kc_upper FLOAT,
    kc_middle FLOAT,
    kc_lower FLOAT,
    kc_width FLOAT,
    kc_position FLOAT,
    
    -- Bollinger Band features
    bb_upper FLOAT,
    bb_middle FLOAT,
    bb_lower FLOAT,
    bb_width FLOAT,
    bb_position FLOAT,
    
    -- Squeeze indicators
    keltner_squeeze BOOLEAN,
    ttm_squeeze BOOLEAN,
    squeeze_momentum FLOAT,
    
    -- Consolidation metrics
    range_high_10d FLOAT,
    range_low_10d FLOAT,
    range_pct_10d FLOAT,
    consolidation_days INTEGER,
    breakout_readiness FLOAT,
    
    -- Efficiency metrics
    price_efficiency_10 FLOAT,
    price_efficiency_20 FLOAT,
    trend_strength FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Market structure features
CREATE TABLE market_features (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Market cap and fundamentals
    market_cap BIGINT,
    float_shares BIGINT,
    avg_volume_30d BIGINT,
    relative_volume FLOAT,
    
    -- Support/resistance levels
    support_level_1 FLOAT,
    support_level_2 FLOAT,
    resistance_level_1 FLOAT,
    resistance_level_2 FLOAT,
    support_strength FLOAT,
    resistance_strength FLOAT,
    
    -- Pattern recognition
    higher_highs_5d INTEGER,
    higher_lows_5d INTEGER,
    breakout_pattern_score FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Historical feature snapshots for training
CREATE TABLE feature_snapshots (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    feature_group VARCHAR(100) NOT NULL,
    features JSONB NOT NULL, -- All features as JSON for flexibility
    target_label INTEGER, -- For supervised learning (0=no breakout, 1=breakout)
    target_magnitude FLOAT, -- Actual breakout magnitude if occurred
    target_days_to_breakout INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature serving cache (for real-time inference)
CREATE TABLE feature_cache (
    ticker VARCHAR(10) PRIMARY KEY,
    last_updated TIMESTAMP NOT NULL,
    technical_features JSONB,
    consolidation_features JSONB,
    market_features JSONB,
    combined_features JSONB, -- Pre-computed feature vector for ML
    feature_hash VARCHAR(64), -- For change detection
    expires_at TIMESTAMP
);

-- Feature computation jobs tracking
CREATE TABLE feature_jobs (
    id SERIAL PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL, -- 'batch', 'streaming', 'backfill'
    status VARCHAR(20) NOT NULL, -- 'running', 'completed', 'failed'
    ticker VARCHAR(10),
    start_date DATE,
    end_date DATE,
    features_computed INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_technical_features_ticker_ts ON technical_features(ticker, timestamp DESC);
CREATE INDEX idx_consolidation_features_ticker_ts ON consolidation_features(ticker, timestamp DESC);
CREATE INDEX idx_market_features_ticker_ts ON market_features(ticker, timestamp DESC);
CREATE INDEX idx_feature_snapshots_ticker_ts ON feature_snapshots(ticker, timestamp DESC);
CREATE INDEX idx_feature_snapshots_target ON feature_snapshots(target_label, timestamp DESC);
CREATE INDEX idx_feature_cache_updated ON feature_cache(last_updated);

-- Partitioning for large datasets (optional)
-- CREATE TABLE technical_features_2024 PARTITION OF technical_features
-- FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');