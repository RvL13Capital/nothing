# enterprise_feature_store.py
"""
Enterprise-Grade Feature Store for Breakout Prediction System
Supports real-time streaming, GPU acceleration, and ML Ops automation
"""

import asyncio
import hashlib
import numpy as np
import pandas as pd
import redis.asyncio as redis
import clickhouse_connect
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from abc import ABC, abstractmethod
import logging
from enum import Enum
import pickle
import joblib
from pathlib import Path

# Optional GPU acceleration
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# ML drift detection
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_suite import MetricSuite
    from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
    DRIFT_DETECTION_AVAILABLE = True
except ImportError:
    DRIFT_DETECTION_AVAILABLE = False

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Feature data types with validation"""
    FLOAT = "float64"
    INT = "int64"
    BOOL = "bool"
    CATEGORICAL = "category"
    TIMESTAMP = "datetime64[ns]"
    JSON = "object"

class ComputeMode(Enum):
    """Feature computation modes"""
    CPU = "cpu"
    GPU = "gpu"
    DISTRIBUTED = "distributed"
    STREAMING = "streaming"

@dataclass
class FeatureSchema:
    """Enhanced feature schema with validation and metadata"""
    name: str
    feature_type: FeatureType
    description: str
    tags: List[str]
    dependencies: List[str] = None
    validation_rules: Dict = None
    compute_mode: ComputeMode = ComputeMode.CPU
    ttl_seconds: int = 3600  # 1 hour default TTL
    version: str = "1.0.0"
    is_label: bool = False
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.validation_rules is None:
            self.validation_rules = {}

@dataclass
class FeatureGroup:
    """Feature group with enhanced metadata and lineage"""
    name: str
    features: List[FeatureSchema]
    description: str
    version: str
    tags: List[str]
    source_tables: List[str]
    compute_schedule: str = None  # Cron expression
    retention_days: int = 365
    materialization_mode: str = "batch"  # batch, streaming, on_demand
    
    def get_feature_names(self) -> List[str]:
        return [f.name for f in self.features]
    
    def get_feature_types(self) -> Dict[str, FeatureType]:
        return {f.name: f.feature_type for f in self.features}

class FeatureValidator:
    """Advanced feature validation and quality checks"""
    
    def __init__(self):
        self.validation_cache = {}
        
    def validate_features(self, features: pd.DataFrame, schema: FeatureGroup) -> Dict[str, Any]:
        """Comprehensive feature validation"""
        validation_results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        # Schema validation
        expected_features = set(schema.get_feature_names())
        actual_features = set(features.columns)
        
        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features
        
        if missing_features:
            validation_results["errors"].append(f"Missing features: {missing_features}")
            validation_results["passed"] = False
            
        if extra_features:
            validation_results["warnings"].append(f"Unexpected features: {extra_features}")
        
        # Data type validation
        feature_types = schema.get_feature_types()
        for feature_name, expected_type in feature_types.items():
            if feature_name in features.columns:
                actual_dtype = str(features[feature_name].dtype)
                if not self._is_compatible_type(actual_dtype, expected_type.value):
                    validation_results["errors"].append(
                        f"Type mismatch for {feature_name}: expected {expected_type.value}, got {actual_dtype}"
                    )
                    validation_results["passed"] = False
        
        # Data quality checks
        validation_results["metrics"]["null_percentage"] = features.isnull().mean().to_dict()
        validation_results["metrics"]["feature_stats"] = features.describe().to_dict()
        
        # Custom validation rules
        for feature_schema in schema.features:
            if feature_schema.validation_rules and feature_schema.name in features.columns:
                feature_data = features[feature_schema.name]
                rule_results = self._apply_validation_rules(feature_data, feature_schema.validation_rules)
                if not rule_results["passed"]:
                    validation_results["errors"].extend(rule_results["errors"])
                    validation_results["passed"] = False
        
        return validation_results
    
    def _is_compatible_type(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible"""
        type_mappings = {
            "float64": ["float32", "float64", "int64", "int32"],
            "int64": ["int32", "int64"],
            "bool": ["bool"],
            "category": ["object", "category"],
            "datetime64[ns]": ["datetime64[ns]", "object"],
            "object": ["object"]
        }
        return actual in type_mappings.get(expected, [expected])
    
    def _apply_validation_rules(self, data: pd.Series, rules: Dict) -> Dict[str, Any]:
        """Apply custom validation rules to feature data"""
        results = {"passed": True, "errors": []}
        
        if "min_value" in rules and data.min() < rules["min_value"]:
            results["errors"].append(f"Minimum value {data.min()} below threshold {rules['min_value']}")
            results["passed"] = False
            
        if "max_value" in rules and data.max() > rules["max_value"]:
            results["errors"].append(f"Maximum value {data.max()} above threshold {rules['max_value']}")
            results["passed"] = False
            
        if "allowed_values" in rules:
            invalid_values = set(data.unique()) - set(rules["allowed_values"])
            if invalid_values:
                results["errors"].append(f"Invalid values found: {invalid_values}")
                results["passed"] = False
        
        return results

class AdvancedFeatureComputer:
    """GPU-accelerated feature computation engine"""
    
    def __init__(self, use_gpu: bool = GPU_AVAILABLE):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.compute_cache = {}
        
        if self.use_gpu:
            logger.info("GPU acceleration enabled for feature computation")
        else:
            logger.info("Using CPU for feature computation")
    
    def compute_technical_features(self, price_data: Union[pd.DataFrame, 'cudf.DataFrame']) -> Dict[str, np.ndarray]:
        """Compute technical indicators with GPU acceleration"""
        
        if self.use_gpu and isinstance(price_data, pd.DataFrame):
            price_data = cudf.from_pandas(price_data)
        
        features = {}
        
        # Price features
        features['price_close'] = self._safe_array(price_data['close'])
        features['price_open'] = self._safe_array(price_data['open'])
        features['price_high'] = self._safe_array(price_data['high'])
        features['price_low'] = self._safe_array(price_data['low'])
        features['volume'] = self._safe_array(price_data['volume'])
        
        # Returns and volatility (vectorized)
        close_prices = features['price_close']
        returns = self._compute_returns(close_prices)
        
        features['returns_1d'] = returns
        features['volatility_10d'] = self._rolling_std(returns, 10)
        features['volatility_20d'] = self._rolling_std(returns, 20)
        
        # Moving averages (optimized)
        features['sma_10'] = self._sma(close_prices, 10)
        features['sma_20'] = self._sma(close_prices, 20)
        features['sma_50'] = self._sma(close_prices, 50)
        features['ema_10'] = self._ema(close_prices, 10)
        features['ema_20'] = self._ema(close_prices, 20)
        features['ema_50'] = self._ema(close_prices, 50)
        
        # Advanced technical indicators
        features['rsi_14'] = self._rsi(close_prices, 14)
        features['atr_14'] = self._atr(features['price_high'], features['price_low'], close_prices, 14)
        features['bb_upper'], features['bb_lower'] = self._bollinger_bands(close_prices, 20, 2)
        
        # Keltner Channels (critical for consolidation detection)
        features['kc_upper'], features['kc_middle'], features['kc_lower'] = self._keltner_channels(
            close_prices, features['price_high'], features['price_low'], 20, 2.0
        )
        
        # Volume features
        features['volume_sma_20'] = self._sma(features['volume'], 20)
        features['volume_ratio'] = features['volume'] / features['volume_sma_20']
        features['obv'] = self._obv(close_prices, features['volume'])
        
        # Consolidation-specific features
        features['range_pct_10d'] = self._range_percentage(features['price_high'], features['price_low'], 10)
        features['kc_position'] = self._channel_position(close_prices, features['kc_upper'], features['kc_lower'])
        features['squeeze_indicator'] = self._squeeze_detector(features['bb_upper'], features['bb_lower'], 
                                                             features['kc_upper'], features['kc_lower'])
        
        # Convert back to numpy if using GPU
        if self.use_gpu:
            features = {k: cp.asnumpy(v) if hasattr(v, 'values') else v for k, v in features.items()}
        
        return features
    
    def _safe_array(self, series):
        """Safely convert series to array with GPU support"""
        if self.use_gpu:
            return series.values if hasattr(series, 'values') else series
        return series.values if hasattr(series, 'values') else np.array(series)
    
    def _compute_returns(self, prices):
        """Compute returns with GPU acceleration"""
        if self.use_gpu:
            return (prices[1:] - prices[:-1]) / prices[:-1]
        return np.diff(prices) / prices[:-1]
    
    def _rolling_std(self, data, window):
        """Rolling standard deviation"""
        if self.use_gpu:
            return data.rolling(window).std()
        return pd.Series(data).rolling(window).std().values
    
    def _sma(self, data, period):
        """Simple Moving Average"""
        if self.use_gpu:
            return data.rolling(period).mean()
        return pd.Series(data).rolling(period).mean().values
    
    def _ema(self, data, period):
        """Exponential Moving Average"""
        if self.use_gpu:
            return data.ewm(span=period).mean()
        return pd.Series(data).ewm(span=period).mean().values
    
    def _rsi(self, prices, period=14):
        """Relative Strength Index"""
        if self.use_gpu:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        else:
            prices = pd.Series(prices)
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values if hasattr(rsi, 'values') else rsi
    
    def _atr(self, high, low, close, period=14):
        """Average True Range"""
        if self.use_gpu:
            high_low = high - low
            high_close = abs(high - close.shift(1))
            low_close = abs(low - close.shift(1))
            true_range = cudf.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        else:
            high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
            high_low = high - low
            high_close = abs(high - close.shift(1))
            low_close = abs(low - close.shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        atr = true_range.rolling(period).mean()
        return atr.values if hasattr(atr, 'values') else atr
    
    def _bollinger_bands(self, prices, period=20, std_dev=2):
        """Bollinger Bands"""
        if self.use_gpu:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
        else:
            prices = pd.Series(prices)
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return (upper.values if hasattr(upper, 'values') else upper,
                lower.values if hasattr(lower, 'values') else lower)
    
    def _keltner_channels(self, close, high, low, period=20, multiplier=2.0):
        """Keltner Channels - critical for consolidation detection"""
        if self.use_gpu:
            ema = close.ewm(span=period).mean()
            atr = self._atr(high, low, close, period)
        else:
            close = pd.Series(close)
            ema = close.ewm(span=period).mean()
            atr = self._atr(high, low, close, period)
        
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)
        return (upper.values if hasattr(upper, 'values') else upper,
                ema.values if hasattr(ema, 'values') else ema,
                lower.values if hasattr(lower, 'values') else lower)
    
    def _obv(self, close, volume):
        """On-Balance Volume"""
        if self.use_gpu:
            price_change = close.diff()
            obv = (volume * price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum()
        else:
            close, volume = pd.Series(close), pd.Series(volume)
            price_change = close.diff()
            obv = (volume * price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum()
        
        return obv.values if hasattr(obv, 'values') else obv
    
    def _range_percentage(self, high, low, period=10):
        """Rolling range percentage"""
        if self.use_gpu:
            rolling_high = high.rolling(period).max()
            rolling_low = low.rolling(period).min()
        else:
            high, low = pd.Series(high), pd.Series(low)
            rolling_high = high.rolling(period).max()
            rolling_low = low.rolling(period).min()
        
        range_pct = (rolling_high - rolling_low) / rolling_low
        return range_pct.values if hasattr(range_pct, 'values') else range_pct
    
    def _channel_position(self, close, upper, lower):
        """Position within channel (0 = bottom, 1 = top)"""
        position = (close - lower) / (upper - lower)
        if self.use_gpu:
            return position.fillna(0.5)
        return pd.Series(position).fillna(0.5).values
    
    def _squeeze_detector(self, bb_upper, bb_lower, kc_upper, kc_lower):
        """TTM Squeeze detector (Bollinger Bands inside Keltner Channels)"""
        squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
        return squeeze.values if hasattr(squeeze, 'values') else squeeze

class EnterpriseFeatureStore:
    """Enterprise-grade feature store with advanced capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize storage backends
        self.redis_client = None
        self.clickhouse_client = None
        self.postgres_client = None
        
        # Feature computation engine
        self.feature_computer = AdvancedFeatureComputer(config.get('use_gpu', False))
        self.validator = FeatureValidator()
        
        # Feature registry
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self.feature_schemas: Dict[str, FeatureSchema] = {}
        
        # Performance monitoring
        self.metrics = {
            "features_served": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "computation_time": [],
            "validation_failures": 0
        }
        
        # Thread pools for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=config.get('max_workers', 10))
        self.process_pool = ProcessPoolExecutor(max_workers=config.get('max_processes', 4))
    
    async def initialize(self):
        """Initialize all storage backends and connections"""
        # Initialize Redis cluster
        redis_config = self.config.get('redis', {})
        self.redis_client = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            password=redis_config.get('password'),
            decode_responses=False  # Handle binary data
        )
        
        # Initialize ClickHouse for time-series data
        ch_config = self.config.get('clickhouse', {})
        self.clickhouse_client = clickhouse_connect.get_client(
            host=ch_config.get('host', 'localhost'),
            port=ch_config.get('port', 8123),
            username=ch_config.get('username', 'default'),
            password=ch_config.get('password', '')
        )
        
        logger.info("Enterprise Feature Store initialized successfully")
    
    def register_feature_group(self, feature_group: FeatureGroup):
        """Register a new feature group with the store"""
        self.feature_groups[feature_group.name] = feature_group
        
        # Register individual features
        for feature in feature_group.features:
            feature_key = f"{feature_group.name}.{feature.name}"
            self.feature_schemas[feature_key] = feature
        
        logger.info(f"Registered feature group: {feature_group.name} with {len(feature_group.features)} features")
    
    async def compute_and_store_features(self, 
                                       ticker: str, 
                                       price_data: pd.DataFrame, 
                                       feature_group_name: str) -> Dict[str, Any]:
        """Compute and store features for a ticker"""
        start_time = datetime.now()
        
        # Get feature group
        if feature_group_name not in self.feature_groups:
            raise ValueError(f"Feature group {feature_group_name} not found")
        
        feature_group = self.feature_groups[feature_group_name]
        
        # Compute features
        if feature_group_name == "technical_indicators":
            computed_features = self.feature_computer.compute_technical_features(price_data)
        else:
            # Add other feature group computations here
            computed_features = {}
        
        # Create DataFrame for validation
        feature_df = pd.DataFrame(computed_features)
        
        # Validate features
        validation_results = self.validator.validate_features(feature_df, feature_group)
        if not validation_results["passed"]:
            self.metrics["validation_failures"] += 1
            logger.error(f"Feature validation failed for {ticker}: {validation_results['errors']}")
            return {"success": False, "errors": validation_results["errors"]}
        
        # Store in ClickHouse (time-series optimized)
        await self._store_features_clickhouse(ticker, feature_df, feature_group_name)
        
        # Cache latest features in Redis
        await self._cache_features_redis(ticker, computed_features, feature_group_name)
        
        # Update metrics
        computation_time = (datetime.now() - start_time).total_seconds()
        self.metrics["computation_time"].append(computation_time)
        
        logger.info(f"Computed and stored {len(computed_features)} features for {ticker} in {computation_time:.2f}s")
        
        return {
            "success": True,
            "features_computed": len(computed_features),
            "computation_time": computation_time,
            "validation_results": validation_results
        }
    
    async def get_features(self, 
                          ticker: str, 
                          feature_names: List[str], 
                          timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Retrieve features with multi-tier caching"""
        
        # Try Redis cache first (sub-millisecond retrieval)
        cached_features = await self._get_cached_features(ticker, feature_names)
        if cached_features:
            self.metrics["cache_hits"] += 1
            return cached_features
        
        self.metrics["cache_misses"] += 1
        
        # Fallback to ClickHouse
        features = await self._get_features_clickhouse(ticker, feature_names, timestamp)
        
        # Cache for next time
        if features:
            await self._cache_features_redis(ticker, features, "mixed")
        
        self.metrics["features_served"] += 1
        return features
    
    async def get_training_dataset(self, 
                                 tickers: List[str], 
                                 feature_groups: List[str],
                                 start_date: datetime,
                                 end_date: datetime,
                                 include_labels: bool = True) -> pd.DataFrame:
        """Generate training dataset with consistent features"""
        
        query = f"""
        SELECT 
            ticker,
            timestamp,
            {','.join([f"feature_data['{fg}'] as {fg}" for fg in feature_groups])}
            {', target_label, target_magnitude' if include_labels else ''}
        FROM feature_snapshots 
        WHERE ticker IN ({','.join([f"'{t}'" for t in tickers])})
        AND timestamp BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY ticker, timestamp
        """
        
        training_data = self.clickhouse_client.query_df(query)
        
        logger.info(f"Generated training dataset with {len(training_data)} samples")
        return training_data
    
    async def _store_features_clickhouse(self, ticker: str, features_df: pd.DataFrame, feature_group: str):
        """Store features in ClickHouse for time-series optimization"""
        
        # Prepare data for insertion
        data_to_insert = []
        for idx, row in features_df.iterrows():
            record = {
                'ticker': ticker,
                'timestamp': datetime.now(),
                'feature_group': feature_group,
                'feature_data': json.dumps(row.to_dict())
            }
            data_to_insert.append(record)
        
        # Insert data
        self.clickhouse_client.insert('feature_store', data_to_insert)
    
    async def _cache_features_redis(self, ticker: str, features: Dict[str, Any], feature_group: str):
        """Cache features in Redis with optimized serialization"""
        
        # Create cache key
        cache_key = f"features:{ticker}:{feature_group}"
        
        # Serialize features (use pickle for numpy arrays)
        serialized_features = pickle.dumps(features)
        
        # Store with TTL
        await self.redis_client.setex(cache_key, 3600, serialized_features)  # 1 hour TTL
    
    async def _get_cached_features(self, ticker: str, feature_names: List[str]) -> Optional[Dict[str, Any]]:
        """Retrieve cached features from Redis"""
        
        # Try to get from cache
        cache_keys = [f"features:{ticker}:*"]
        
        # Get all cached feature groups for this ticker
        cached_data = {}
        for pattern in cache_keys:
            keys = await self.redis_client.keys(pattern)
            for key in keys:
                serialized_data = await self.redis_client.get(key)
                if serialized_data:
                    features = pickle.loads(serialized_data)
                    cached_data.update(features)
        
        # Return requested features if available
        if cached_data and all(fn in cached_data for fn in feature_names):
            return {fn: cached_data[fn] for fn in feature_names}
        
        return None
    
    async def _get_features_clickhouse(self, 
                                     ticker: str, 
                                     feature_names: List[str], 
                                     timestamp: Optional[datetime]) -> Dict[str, Any]:
        """Retrieve features from ClickHouse"""
        
        timestamp_clause = f"AND timestamp = '{timestamp}'" if timestamp else ""
        
        query = f"""
        SELECT feature_data
        FROM feature_store 
        WHERE ticker = '{ticker}' 
        {timestamp_clause}
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        
        result = self.clickhouse_client.query(query)
        if result:
            feature_data = json.loads(result[0][0])
            return {fn: feature_data.get(fn) for fn in feature_names if fn in feature_data}
        
        return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.metrics.copy()
        if metrics["computation_time"]:
            metrics["avg_computation_time"] = np.mean(metrics["computation_time"])
            metrics["p95_computation_time"] = np.percentile(metrics["computation_time"], 95)
        
        cache_total = metrics["cache_hits"] + metrics["cache_misses"]
        metrics["cache_hit_rate"] = metrics["cache_hits"] / cache_total if cache_total > 0 else 0
        
        return metrics

# Feature group definitions for breakout prediction
TECHNICAL_INDICATORS_GROUP = FeatureGroup(
    name="technical_indicators",
    description="Core technical indicators for breakout detection",
    version="2.0.0",
    tags=["technical", "price", "volume"],
    source_tables=["market_data"],
    features=[
        FeatureSchema("price_close", FeatureType.FLOAT, "Closing price", ["price"]),
        FeatureSchema("sma_10", FeatureType.FLOAT, "10-day Simple Moving Average", ["trend"]),
        FeatureSchema("sma_20", FeatureType.FLOAT, "20-day Simple Moving Average", ["trend"]),
        FeatureSchema("ema_10", FeatureType.FLOAT, "10-day Exponential Moving Average", ["trend"]),
        FeatureSchema("rsi_14", FeatureType.FLOAT, "14-day RSI", ["momentum"], validation_rules={"min_value": 0, "max_value": 100}),
        FeatureSchema("atr_14", FeatureType.FLOAT, "14-day Average True Range", ["volatility"]),
        FeatureSchema("volume_ratio", FeatureType.FLOAT, "Volume relative to 20-day average", ["volume"]),
        FeatureSchema("volatility_10d", FeatureType.FLOAT, "10-day rolling volatility", ["volatility"]),
    ]
)

CONSOLIDATION_GROUP = FeatureGroup(
    name="consolidation_patterns",
    description="Features specific to consolidation pattern detection",
    version="2.0.0",
    tags=["consolidation", "breakout", "pattern"],
    source_tables=["market_data"],
    features=[
        FeatureSchema("kc_upper", FeatureType.FLOAT, "Keltner Channel Upper Band", ["keltner"]),
        FeatureSchema("kc_middle", FeatureType.FLOAT, "Keltner Channel Middle Line", ["keltner"]),
        FeatureSchema("kc_lower", FeatureType.FLOAT, "Keltner Channel Lower Band", ["keltner"]),
        FeatureSchema("kc_position", FeatureType.FLOAT, "Position within Keltner Channel", ["keltner"], validation_rules={"min_value": 0, "max_value": 1}),
        FeatureSchema("squeeze_indicator", FeatureType.BOOL, "TTM Squeeze indicator", ["squeeze"]),
        FeatureSchema("range_pct_10d", FeatureType.FLOAT, "10-day range percentage", ["consolidation"]),
    ]
)

async def initialize_feature_store(config: Dict[str, Any]) -> EnterpriseFeatureStore:
    """Initialize and configure the enterprise feature store"""
    
    feature_store = EnterpriseFeatureStore(config)
    await feature_store.initialize()
    
    # Register feature groups
    feature_store.register_feature_group(TECHNICAL_INDICATORS_GROUP)
    feature_store.register_feature_group(CONSOLIDATION_GROUP)
    
    logger.info("Enterprise Feature Store ready for production")
    return feature_store