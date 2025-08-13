# retail_feature_store.py
"""
Lightweight Feature Store for Retail 40%+ Breakout Prediction System
Cost-effective, EOD-only, optimized for alpha generation on limited budget
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import talib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Simple feature configuration"""
    name: str
    description: str
    lookback_days: int
    feature_type: str  # 'consolidation', 'volume', 'momentum', 'pattern'
    enabled: bool = True

@dataclass
class ConsolidationMetrics:
    """Core consolidation metrics for breakout prediction"""
    ticker: str
    date: datetime
    in_consolidation: bool
    consolidation_days: int
    range_tightness: float  # Higher = tighter consolidation
    keltner_squeeze: bool
    bollinger_squeeze: bool
    support_level: float
    resistance_level: float
    position_in_range: float  # 0-1, where 0.5 = middle
    breakout_readiness: float  # 0-1 composite score
    quality_score: float  # Pattern quality 0-1

@dataclass
class VolumeBreakoutSignal:
    """Volume analysis for breakout timing (separate from consolidation detection)"""
    ticker: str
    date: datetime
    volume_breakout_score: float  # 0-1, higher = more likely volume breakout
    relative_volume: float  # Current vs average
    volume_accumulation: float  # Volume building during consolidation
    volume_dry_up: float  # Volume contraction indicator
    institutional_activity: float  # Proxy for institutional accumulation

class RetailFeatureStore:
    """Lightweight feature store optimized for retail trading"""
    
    def __init__(self, data_dir: str = "./feature_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # SQLite database for features
        self.db_path = self.data_dir / "features.db"
        self.conn = None
        
        # Feature cache (in-memory for session)
        self.feature_cache = {}
        self.cache_size_limit = 10000  # Keep it reasonable for memory
        
        # Feature configurations
        self.feature_configs = self._define_feature_configs()
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Retail Feature Store initialized at {self.data_dir}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Create tables
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_features (
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            price_close REAL,
            price_high REAL,
            price_low REAL,
            price_open REAL,
            volume BIGINT,
            
            -- Technical indicators
            sma_10 REAL, sma_20 REAL, sma_50 REAL,
            ema_10 REAL, ema_20 REAL,
            rsi_14 REAL,
            atr_14 REAL,
            
            -- Keltner Channels (primary for consolidation)
            kc_upper REAL, kc_middle REAL, kc_lower REAL,
            kc_width REAL, kc_position REAL,
            
            -- Bollinger Bands (for squeeze detection)
            bb_upper REAL, bb_middle REAL, bb_lower REAL,
            bb_width REAL,
            
            -- Consolidation features (NO volume filtering)
            consolidation_days INTEGER DEFAULT 0,
            range_tightness REAL DEFAULT 0,
            breakout_readiness REAL DEFAULT 0,
            pattern_quality REAL DEFAULT 0,
            
            -- Volume features (for breakout timing only)
            volume_ma_20 REAL,
            relative_volume REAL,
            volume_accumulation REAL,
            volume_dry_up REAL,
            
            -- Squeeze indicators
            keltner_squeeze BOOLEAN DEFAULT 0,
            bollinger_squeeze BOOLEAN DEFAULT 0,
            ttm_squeeze BOOLEAN DEFAULT 0,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, date)
        );
        """)
        
        # Labels table for training
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS breakout_labels (
            ticker TEXT NOT NULL,
            consolidation_end_date DATE NOT NULL,
            breakout_occurred BOOLEAN NOT NULL,
            breakout_magnitude REAL,
            days_to_breakout INTEGER,
            max_volume_spike REAL,
            label_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, consolidation_end_date)
        );
        """)
        
        # Feature consistency tracking
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_versions (
            version_hash TEXT PRIMARY KEY,
            feature_list TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        self.conn.commit()
        logger.info("Database tables initialized")
    
    def _define_feature_configs(self) -> List[FeatureConfig]:
        """Define core features for breakout prediction"""
        return [
            # Price-based consolidation features (NO volume)
            FeatureConfig("keltner_channels", "Primary consolidation indicator", 20, "consolidation"),
            FeatureConfig("bollinger_bands", "Squeeze detection", 20, "consolidation"),
            FeatureConfig("range_analysis", "Price range tightness", 20, "consolidation"),
            FeatureConfig("support_resistance", "Key levels identification", 50, "consolidation"),
            
            # Volume features (for breakout timing prediction)
            FeatureConfig("volume_patterns", "Volume analysis for timing", 20, "volume"),
            FeatureConfig("accumulation_distribution", "Smart money tracking", 30, "volume"),
            
            # Momentum (confirmation)
            FeatureConfig("rsi_momentum", "Momentum confirmation", 14, "momentum"),
            FeatureConfig("price_efficiency", "Trend strength", 10, "momentum"),
        ]
    
    def compute_eod_features(self, ticker: str, price_data: pd.DataFrame) -> bool:
        """Compute all EOD features for a ticker and store in database"""
        
        if len(price_data) < 100:  # Need sufficient history
            logger.warning(f"Insufficient data for {ticker}: {len(price_data)} days")
            return False
        
        try:
            # Compute technical indicators
            features_df = self._compute_technical_indicators(price_data)
            
            # Compute consolidation features (pure price action)
            consolidation_features = self._compute_consolidation_features(features_df)
            features_df = pd.concat([features_df, consolidation_features], axis=1)
            
            # Compute volume features (separate from consolidation detection)
            volume_features = self._compute_volume_features(features_df)
            features_df = pd.concat([features_df, volume_features], axis=1)
            
            # Add ticker and date
            features_df['ticker'] = ticker
            features_df['date'] = features_df.index.date
            
            # Store in database
            self._store_features(features_df)
            
            logger.info(f"Computed and stored features for {ticker}: {len(features_df)} days")
            return True
            
        except Exception as e:
            logger.error(f"Error computing features for {ticker}: {e}")
            return False
    
    def _compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute core technical indicators"""
        
        result = pd.DataFrame(index=df.index)
        
        # Price data
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        volume = df['volume'].values
        
        result['price_close'] = close
        result['price_high'] = high
        result['price_low'] = low
        result['price_open'] = open_price
        result['volume'] = volume
        
        # Moving averages
        result['sma_10'] = talib.SMA(close, 10)
        result['sma_20'] = talib.SMA(close, 20)
        result['sma_50'] = talib.SMA(close, 50)
        result['ema_10'] = talib.EMA(close, 10)
        result['ema_20'] = talib.EMA(close, 20)
        
        # Momentum
        result['rsi_14'] = talib.RSI(close, 14)
        
        # Volatility
        result['atr_14'] = talib.ATR(high, low, close, 14)
        
        # Keltner Channels (PRIMARY consolidation indicator)
        ema_20 = talib.EMA(close, 20)
        atr_20 = talib.ATR(high, low, close, 20)
        result['kc_upper'] = ema_20 + (2.0 * atr_20)
        result['kc_middle'] = ema_20
        result['kc_lower'] = ema_20 - (2.0 * atr_20)
        result['kc_width'] = (result['kc_upper'] - result['kc_lower']) / result['kc_middle']
        result['kc_position'] = (close - result['kc_lower']) / (result['kc_upper'] - result['kc_lower'])
        
        # Bollinger Bands (for squeeze detection)
        result['bb_upper'], result['bb_middle'], result['bb_lower'] = talib.BBANDS(close, 20, 2, 2)
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        # Squeeze indicators
        result['keltner_squeeze'] = result['kc_width'] < result['kc_width'].rolling(50).mean() * 0.7
        result['bollinger_squeeze'] = (result['bb_upper'] <= result['kc_upper']) & (result['bb_lower'] >= result['kc_lower'])
        result['ttm_squeeze'] = result['bollinger_squeeze']  # Same for simplicity
        
        return result.fillna(0)
    
    def _compute_consolidation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute consolidation features WITHOUT volume filtering"""
        
        result = pd.DataFrame(index=df.index)
        
        close = df['price_close']
        high = df['price_high']
        low = df['price_low']
        
        # Range analysis
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        range_pct = (rolling_high - rolling_low) / rolling_low
        
        # Range tightness (higher = tighter consolidation)
        result['range_tightness'] = 1 / (range_pct + 0.01)  # Inverse of range
        
        # Consolidation detection (pure price action)
        in_consolidation = range_pct < 0.10  # 10% range threshold for $300M-$2B caps
        
        # Count consolidation days
        consolidation_streak = pd.Series(0, index=df.index)
        current_streak = 0
        
        for i in range(len(in_consolidation)):
            if in_consolidation.iloc[i]:
                current_streak += 1
            else:
                current_streak = 0
            consolidation_streak.iloc[i] = current_streak
        
        result['consolidation_days'] = consolidation_streak
        
        # Support and resistance levels
        result['support_level'] = low.rolling(50, center=True).min()
        result['resistance_level'] = high.rolling(50, center=True).max()
        
        # Position within range
        result['position_in_range'] = ((close - result['support_level']) / 
                                     (result['resistance_level'] - result['support_level'])).fillna(0.5)
        
        # Pattern quality score
        quality_factors = []
        
        # Factor 1: Range tightness
        quality_factors.append(np.clip(result['range_tightness'] / 10, 0, 1))
        
        # Factor 2: Consolidation duration (optimal 15-45 days)
        duration_score = np.where(
            result['consolidation_days'] < 15, result['consolidation_days'] / 15,
            np.where(result['consolidation_days'] > 45, 
                    np.maximum(0, 1 - (result['consolidation_days'] - 45) / 45), 1)
        )
        quality_factors.append(duration_score)
        
        # Factor 3: Squeeze presence
        quality_factors.append(df['keltner_squeeze'].astype(float))
        
        result['pattern_quality'] = np.mean(quality_factors, axis=0)
        
        # Breakout readiness (composite score)
        readiness_factors = []
        readiness_factors.append(result['pattern_quality'])  # Base quality
        readiness_factors.append(np.clip(result['consolidation_days'] / 30, 0, 1))  # Duration factor
        readiness_factors.append(df['keltner_squeeze'].astype(float))  # Squeeze factor
        
        # Position factor (closer to resistance = more ready)
        position_factor = np.where(result['position_in_range'] > 0.7, 1.0,
                                 np.where(result['position_in_range'] > 0.5, 0.5, 0.2))
        readiness_factors.append(position_factor)
        
        result['breakout_readiness'] = np.clip(np.mean(readiness_factors, axis=0), 0, 1)
        
        return result.fillna(0)
    
    def _compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume features for breakout TIMING prediction (not consolidation detection)"""
        
        result = pd.DataFrame(index=df.index)
        
        volume = df['volume']
        close = df['price_close']
        
        # Basic volume metrics
        result['volume_ma_20'] = volume.rolling(20).mean()
        result['relative_volume'] = volume / result['volume_ma_20']
        
        # Volume accumulation during consolidation
        # Higher volume on up days vs down days during consolidation
        price_change = close.pct_change()
        up_volume = np.where(price_change > 0, volume, 0)
        down_volume = np.where(price_change < 0, volume, 0)
        
        up_vol_ma = pd.Series(up_volume).rolling(20).mean()
        down_vol_ma = pd.Series(down_volume).rolling(20).mean()
        result['volume_accumulation'] = up_vol_ma / (down_vol_ma + 1)  # Avoid division by zero
        
        # Volume dry-up (decreasing volume during consolidation)
        recent_volume = volume.rolling(10).mean()
        historical_volume = volume.rolling(50).mean()
        result['volume_dry_up'] = 1 - (recent_volume / historical_volume)
        
        return result.fillna(0)
    
    def _store_features(self, features_df: pd.DataFrame):
        """Store features in SQLite database"""
        
        # Convert boolean columns to int for SQLite
        bool_columns = ['keltner_squeeze', 'bollinger_squeeze', 'ttm_squeeze']
        for col in bool_columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype(int)
        
        # Store in database (replace existing)
        features_df.to_sql('daily_features', self.conn, if_exists='replace', index=False)
        self.conn.commit()
    
    def get_features_for_inference(self, ticker: str, date: datetime = None) -> Optional[Dict[str, float]]:
        """Get features for inference with training consistency"""
        
        if date is None:
            date = datetime.now().date()
        
        # Try cache first
        cache_key = f"{ticker}_{date}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Query database
        query = """
        SELECT * FROM daily_features 
        WHERE ticker = ? AND date = ?
        """
        
        cursor = self.conn.execute(query, (ticker, str(date)))
        row = cursor.fetchone()
        
        if row:
            # Convert to dict
            columns = [desc[0] for desc in cursor.description]
            features = dict(zip(columns, row))
            
            # Remove non-feature columns
            for col in ['ticker', 'date', 'created_at']:
                features.pop(col, None)
            
            # Cache result
            if len(self.feature_cache) < self.cache_size_limit:
                self.feature_cache[cache_key] = features
            
            return features
        
        return None
    
    def get_training_data(self, 
                         tickers: List[str], 
                         start_date: datetime, 
                         end_date: datetime,
                         include_labels: bool = True) -> pd.DataFrame:
        """Get training dataset with consistent features"""
        
        ticker_list = "','".join(tickers)
        query = f"""
        SELECT f.*, 
               {'l.breakout_occurred, l.breakout_magnitude, l.days_to_breakout' if include_labels else ''}
        FROM daily_features f
        {'LEFT JOIN breakout_labels l ON f.ticker = l.ticker AND f.date = l.consolidation_end_date' if include_labels else ''}
        WHERE f.ticker IN ('{ticker_list}')
        AND f.date BETWEEN ? AND ?
        ORDER BY f.ticker, f.date
        """
        
        df = pd.read_sql(query, self.conn, params=[str(start_date), str(end_date)])
        return df
    
    def store_breakout_label(self, 
                           ticker: str, 
                           consolidation_end_date: datetime,
                           breakout_occurred: bool,
                           breakout_magnitude: float = None,
                           days_to_breakout: int = None):
        """Store breakout labels for training"""
        
        query = """
        INSERT OR REPLACE INTO breakout_labels 
        (ticker, consolidation_end_date, breakout_occurred, breakout_magnitude, days_to_breakout)
        VALUES (?, ?, ?, ?, ?)
        """
        
        self.conn.execute(query, (
            ticker, str(consolidation_end_date), breakout_occurred,
            breakout_magnitude, days_to_breakout
        ))
        self.conn.commit()
    
    def analyze_consolidation_for_ticker(self, ticker: str, current_date: datetime = None) -> ConsolidationMetrics:
        """Analyze current consolidation status for a ticker"""
        
        if current_date is None:
            current_date = datetime.now()
        
        features = self.get_features_for_inference(ticker, current_date.date())
        
        if not features:
            return ConsolidationMetrics(
                ticker=ticker, date=current_date, in_consolidation=False,
                consolidation_days=0, range_tightness=0, keltner_squeeze=False,
                bollinger_squeeze=False, support_level=0, resistance_level=0,
                position_in_range=0.5, breakout_readiness=0, quality_score=0
            )
        
        return ConsolidationMetrics(
            ticker=ticker,
            date=current_date,
            in_consolidation=features['consolidation_days'] > 10,
            consolidation_days=int(features['consolidation_days']),
            range_tightness=features['range_tightness'],
            keltner_squeeze=bool(features['keltner_squeeze']),
            bollinger_squeeze=bool(features['bollinger_squeeze']),
            support_level=features.get('support_level', 0),
            resistance_level=features.get('resistance_level', 0),
            position_in_range=features.get('position_in_range', 0.5),
            breakout_readiness=features['breakout_readiness'],
            quality_score=features['pattern_quality']
        )
    
    def analyze_volume_breakout_potential(self, ticker: str, current_date: datetime = None) -> VolumeBreakoutSignal:
        """Analyze volume patterns for breakout timing (separate from consolidation)"""
        
        if current_date is None:
            current_date = datetime.now()
        
        features = self.get_features_for_inference(ticker, current_date.date())
        
        if not features:
            return VolumeBreakoutSignal(
                ticker=ticker, date=current_date, volume_breakout_score=0,
                relative_volume=1.0, volume_accumulation=1.0, volume_dry_up=0,
                institutional_activity=0
            )
        
        # Calculate volume breakout score
        volume_factors = []
        
        # Factor 1: Relative volume (higher = more likely breakout)
        rel_vol = features.get('relative_volume', 1.0)
        volume_factors.append(min(rel_vol / 2.0, 1.0))  # Cap at 2x volume
        
        # Factor 2: Volume accumulation (smart money accumulating)
        vol_acc = features.get('volume_accumulation', 1.0)
        volume_factors.append(min(vol_acc / 2.0, 1.0))
        
        # Factor 3: Volume dry-up (spring being loaded)
        vol_dry = features.get('volume_dry_up', 0)
        volume_factors.append(vol_dry)
        
        volume_breakout_score = np.mean(volume_factors)
        
        return VolumeBreakoutSignal(
            ticker=ticker,
            date=current_date,
            volume_breakout_score=volume_breakout_score,
            relative_volume=rel_vol,
            volume_accumulation=vol_acc,
            volume_dry_up=vol_dry,
            institutional_activity=vol_acc * (1 + vol_dry)  # Proxy calculation
        )
    
    def get_feature_consistency_hash(self) -> str:
        """Generate hash for feature consistency between training and inference"""
        
        feature_names = []
        query = "PRAGMA table_info(daily_features)"
        cursor = self.conn.execute(query)
        
        for row in cursor.fetchall():
            column_name = row[1]
            if column_name not in ['ticker', 'date', 'created_at']:
                feature_names.append(column_name)
        
        feature_list = sorted(feature_names)
        feature_string = ','.join(feature_list)
        
        return hashlib.md5(feature_string.encode()).hexdigest()
    
    def validate_inference_consistency(self, training_hash: str) -> bool:
        """Validate that inference features match training features"""
        
        current_hash = self.get_feature_consistency_hash()
        return current_hash == training_hash
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics"""
        
        stats = {}
        
        # Count records
        cursor = self.conn.execute("SELECT COUNT(*) FROM daily_features")
        stats['total_feature_records'] = cursor.fetchone()[0]
        
        # Count unique tickers
        cursor = self.conn.execute("SELECT COUNT(DISTINCT ticker) FROM daily_features")
        stats['unique_tickers'] = cursor.fetchone()[0]
        
        # Date range
        cursor = self.conn.execute("SELECT MIN(date), MAX(date) FROM daily_features")
        min_date, max_date = cursor.fetchone()
        stats['date_range'] = {'min': min_date, 'max': max_date}
        
        # Cache statistics
        stats['cache_size'] = len(self.feature_cache)
        stats['cache_limit'] = self.cache_size_limit
        
        return stats
    
    def cleanup_old_data(self, days_to_keep: int = 1000):
        """Clean up old data to manage disk space"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        self.conn.execute("DELETE FROM daily_features WHERE date < ?", (str(cutoff_date.date()),))
        self.conn.execute("DELETE FROM breakout_labels WHERE consolidation_end_date < ?", (str(cutoff_date.date()),))
        self.conn.execute("VACUUM")  # Reclaim space
        self.conn.commit()
        
        logger.info(f"Cleaned up data older than {days_to_keep} days")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def create_retail_feature_store(data_dir: str = "./feature_data") -> RetailFeatureStore:
    """Create and initialize retail feature store"""
    
    store = RetailFeatureStore(data_dir)
    logger.info("Retail Feature Store ready for alpha generation!")
    
    return store