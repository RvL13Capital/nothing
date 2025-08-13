-- schema.sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    confidence FLOAT NOT NULL,
    breakout_probability FLOAT NOT NULL,
    expected_magnitude FLOAT NOT NULL,
    expected_days FLOAT NOT NULL,
    signal_strength VARCHAR(20),
    risk_level VARCHAR(20),
    entry_price FLOAT,
    stop_loss FLOAT,
    target_1 FLOAT,
    target_2 FLOAT,
    target_3 FLOAT,
    actual_outcome FLOAT,
    outcome_date DATE,
    success BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE watchlist (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    market_cap BIGINT,
    sector VARCHAR(50),
    active BOOLEAN DEFAULT TRUE,
    added_date DATE DEFAULT CURRENT_DATE
);

CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    total_predictions INTEGER,
    successful_predictions INTEGER,
    average_confidence FLOAT,
    average_return FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_ticker ON predictions(ticker);
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
