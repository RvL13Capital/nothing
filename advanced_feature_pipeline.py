# advanced_feature_pipeline.py
"""
Advanced Feature Engineering Pipeline with ML Drift Detection
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import logging
from abc import ABC, abstractmethod
import dask.dataframe as dd
from dask.distributed import Client
import joblib

# Advanced ML libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Feature engineering libraries
try:
    import ta  # Technical Analysis library
    import talib
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class FeatureDriftReport:
    """Feature drift detection results"""
    ticker: str
    timestamp: datetime
    drift_detected: bool
    drift_score: float
    affected_features: List[str]
    drift_details: Dict[str, Any]
    recommendation: str

class AdvancedFeatureEngineer:
    """Advanced feature engineering with ML-driven feature selection and drift detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_scalers = {}
        self.feature_selectors = {}
        self.drift_detectors = {}
        self.feature_importance_cache = {}
        
        # Dask client for distributed processing
        self.dask_client = None
        if config.get('use_distributed', False):
            try:
                self.dask_client = Client(config.get('dask_scheduler', 'localhost:8786'))
                logger.info("Connected to Dask cluster for distributed processing")
            except Exception as e:
                logger.warning(f"Could not connect to Dask cluster: {e}")
    
    def create_advanced_features(self, price_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Create advanced features using multiple techniques"""
        
        features_df = pd.DataFrame(index=price_data.index)
        
        # Basic price features
        features_df = self._add_price_features(features_df, price_data)
        
        # Advanced technical indicators
        features_df = self._add_advanced_technical_indicators(features_df, price_data)
        
        # Statistical features
        features_df = self._add_statistical_features(features_df, price_data)
        
        # Pattern recognition features
        features_df = self._add_pattern_features(features_df, price_data)
        
        # Market microstructure features
        features_df = self._add_microstructure_features(features_df, price_data)
        
        # Time-based features
        features_df = self._add_temporal_features(features_df, price_data)
        
        # Interaction features
        features_df = self._add_interaction_features(features_df)
        
        logger.info(f"Created {len(features_df.columns)} advanced features for {ticker}")
        return features_df
    
    def _add_price_features(self, features_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced price-based features"""
        
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        open_price = price_data['open']
        volume = price_data['volume']
        
        # Price transformations
        features_df['log_price'] = np.log(close)
        features_df['price_zscore'] = (close - close.rolling(50).mean()) / close.rolling(50).std()
        
        # Gap analysis
        features_df['gap_up'] = (open_price - close.shift(1)) / close.shift(1)
        features_df['gap_size'] = abs(features_df['gap_up'])
        features_df['gap_direction'] = np.sign(features_df['gap_up'])
        
        # Intraday features
        features_df['high_low_ratio'] = high / low
        features_df['close_to_high'] = close / high
        features_df['close_to_low'] = close / low
        features_df['body_size'] = abs(close - open_price) / close
        features_df['upper_shadow'] = (high - np.maximum(close, open_price)) / close
        features_df['lower_shadow'] = (np.minimum(close, open_price) - low) / close
        
        # Price velocity and acceleration
        returns = close.pct_change()
        features_df['returns'] = returns
        features_df['price_velocity'] = returns.rolling(5).mean()
        features_df['price_acceleration'] = returns.diff()
        
        return features_df
    
    def _add_advanced_technical_indicators(self, features_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Advanced technical indicators beyond basic ones"""
        
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        volume = price_data['volume']
        
        if TA_AVAILABLE:
            # TA-Lib indicators
            features_df['adx'] = talib.ADX(high.values, low.values, close.values, timeperiod=14)
            features_df['cci'] = talib.CCI(high.values, low.values, close.values, timeperiod=14)
            features_df['williams_r'] = talib.WILLR(high.values, low.values, close.values, timeperiod=14)
            features_df['ultimate_oscillator'] = talib.ULTOSC(high.values, low.values, close.values)
            features_df['trix'] = talib.TRIX(close.values, timeperiod=14)
            
            # Cycle indicators
            features_df['ht_dcperiod'] = talib.HT_DCPERIOD(close.values)
            features_df['ht_dcphase'] = talib.HT_DCPHASE(close.values)
            features_df['ht_trendmode'] = talib.HT_TRENDMODE(close.values)
            
            # Volume indicators
            features_df['ad'] = talib.AD(high.values, low.values, close.values, volume.values)
            features_df['adosc'] = talib.ADOSC(high.values, low.values, close.values, volume.values)
        
        # Custom advanced indicators
        features_df['kaufman_efficiency'] = self._kaufman_efficiency_ratio(close)
        features_df['fractal_dimension'] = self._fractal_dimension(close)
        features_df['hurst_exponent'] = self._hurst_exponent(close)
        
        # Multi-timeframe features
        for period in [5, 10, 20, 50]:
            features_df[f'rsi_{period}'] = self._rsi(close, period)
            features_df[f'stoch_k_{period}'] = self._stochastic_k(high, low, close, period)
            
            # Momentum oscillators
            features_df[f'momentum_{period}'] = close / close.shift(period) - 1
            features_df[f'roc_{period}'] = close.pct_change(period)
        
        return features_df
    
    def _add_statistical_features(self, features_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Statistical and distributional features"""
        
        close = price_data['close']
        returns = close.pct_change()
        
        # Rolling statistics
        for window in [10, 20, 50]:
            features_df[f'skewness_{window}'] = returns.rolling(window).skew()
            features_df[f'kurtosis_{window}'] = returns.rolling(window).kurt()
            features_df[f'var_{window}'] = returns.rolling(window).var()
            features_df[f'std_{window}'] = returns.rolling(window).std()
            
            # Percentile features
            features_df[f'percentile_rank_{window}'] = close.rolling(window).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
            )
        
        # Distribution features
        features_df['jarque_bera'] = returns.rolling(50).apply(self._jarque_bera_stat)
        features_df['normality_test'] = returns.rolling(50).apply(self._shapiro_stat)
        
        # Entropy and complexity
        features_df['sample_entropy'] = returns.rolling(50).apply(self._sample_entropy)
        features_df['approximate_entropy'] = returns.rolling(50).apply(self._approximate_entropy)
        
        return features_df
    
    def _add_pattern_features(self, features_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Pattern recognition features"""
        
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        open_price = price_data['open']
        
        # Candlestick patterns (simplified)
        features_df['doji'] = self._is_doji(open_price, close, high, low)
        features_df['hammer'] = self._is_hammer(open_price, close, high, low)
        features_df['shooting_star'] = self._is_shooting_star(open_price, close, high, low)
        
        # Support/Resistance levels
        features_df['support_level'] = self._find_support_level(close, window=20)
        features_df['resistance_level'] = self._find_resistance_level(close, window=20)
        features_df['distance_to_support'] = (close - features_df['support_level']) / close
        features_df['distance_to_resistance'] = (features_df['resistance_level'] - close) / close
        
        # Trend patterns
        features_df['higher_highs'] = self._count_higher_highs(high, window=10)
        features_df['lower_lows'] = self._count_lower_lows(low, window=10)
        features_df['trend_consistency'] = features_df['higher_highs'] - features_df['lower_lows']
        
        # Wave analysis
        features_df['wave_amplitude'] = self._calculate_wave_amplitude(close)
        features_df['wave_frequency'] = self._calculate_wave_frequency(close)
        
        return features_df
    
    def _add_microstructure_features(self, features_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features"""
        
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        volume = price_data['volume']
        
        # Price impact features
        features_df['price_impact'] = abs(close.pct_change()) / (volume.rolling(5).mean() / 1000)
        features_df['volume_weighted_price'] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # Liquidity indicators
        features_df['bid_ask_spread_proxy'] = (high - low) / close
        features_df['depth_proxy'] = volume / abs(close.pct_change()).replace(0, np.nan)
        
        # Order flow imbalance proxy
        price_change = close.diff()
        volume_change = volume.diff()
        features_df['flow_imbalance'] = np.where(
            price_change > 0, volume_change, 
            np.where(price_change < 0, -volume_change, 0)
        )
        
        return features_df
    
    def _add_temporal_features(self, features_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""
        
        # Day of week effects
        features_df['day_of_week'] = price_data.index.dayofweek
        features_df['is_monday'] = (features_df['day_of_week'] == 0).astype(int)
        features_df['is_friday'] = (features_df['day_of_week'] == 4).astype(int)
        
        # Month effects
        features_df['month'] = price_data.index.month
        features_df['quarter'] = price_data.index.quarter
        
        # Seasonal decomposition
        close = price_data['close']
        features_df['seasonal_component'] = self._seasonal_decomposition(close)
        features_df['trend_component'] = close.rolling(50).mean()
        features_df['residual_component'] = close - features_df['trend_component']
        
        return features_df
    
    def _add_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Feature interactions and combinations"""
        
        # Select key features for interactions
        key_features = ['rsi_14', 'atr_14', 'volume_ratio', 'price_velocity', 'momentum_10']
        
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i+1:]:
                if feat1 in features_df.columns and feat2 in features_df.columns:
                    # Multiplicative interactions
                    features_df[f'{feat1}_x_{feat2}'] = features_df[feat1] * features_df[feat2]
                    
                    # Ratio interactions
                    features_df[f'{feat1}_ratio_{feat2}'] = features_df[feat1] / (features_df[feat2] + 1e-8)
        
        return features_df
    
    # Helper methods for advanced calculations
    def _kaufman_efficiency_ratio(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Kaufman's Adaptive Moving Average Efficiency Ratio"""
        direction = abs(prices - prices.shift(period))
        volatility = abs(prices.diff()).rolling(period).sum()
        return direction / volatility
    
    def _fractal_dimension(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """Fractal dimension calculation"""
        def calculate_fd(series):
            if len(series) < 10:
                return np.nan
            # Simplified fractal dimension calculation
            diffs = np.diff(series)
            return 1 + np.log(np.sum(np.abs(diffs))) / np.log(len(series))
        
        return prices.rolling(window).apply(calculate_fd)
    
    def _hurst_exponent(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """Hurst exponent for trend persistence"""
        def calculate_hurst(series):
            if len(series) < 20:
                return np.nan
            
            lags = range(2, 20)
            tau = []
            
            for lag in lags:
                pp = np.sum((series[lag:] - series[:-lag])**2)
                tau.append(pp / len(series))
            
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        return prices.rolling(window).apply(calculate_hurst)
    
    def _rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _stochastic_k(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Stochastic %K"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        return 100 * (close - lowest_low) / (highest_high - lowest_low)
    
    def _jarque_bera_stat(self, returns: pd.Series) -> float:
        """Jarque-Bera test statistic for normality"""
        if len(returns) < 10:
            return np.nan
        n = len(returns)
        skew = returns.skew()
        kurt = returns.kurt()
        return n / 6 * (skew**2 + (kurt**2) / 4)
    
    def _shapiro_stat(self, returns: pd.Series) -> float:
        """Simplified Shapiro-Wilk statistic proxy"""
        if len(returns) < 10:
            return np.nan
        # Simplified version - actual implementation would use scipy.stats
        return abs(returns.skew()) + abs(returns.kurt())
    
    def _sample_entropy(self, series: pd.Series) -> float:
        """Sample entropy calculation"""
        if len(series) < 20:
            return np.nan
        # Simplified entropy calculation
        return -np.sum(pd.value_counts(pd.cut(series, bins=10), normalize=True) * 
                      np.log(pd.value_counts(pd.cut(series, bins=10), normalize=True) + 1e-8))
    
    def _approximate_entropy(self, series: pd.Series) -> float:
        """Approximate entropy calculation"""
        if len(series) < 20:
            return np.nan
        # Simplified approximate entropy
        return np.std(series) / (np.mean(series) + 1e-8)
    
    def _is_doji(self, open_price: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Detect Doji candlestick pattern"""
        body_size = abs(close - open_price)
        total_range = high - low
        return (body_size / total_range < 0.1).astype(int)
    
    def _is_hammer(self, open_price: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Detect Hammer candlestick pattern"""
        body_top = np.maximum(open_price, close)
        body_bottom = np.minimum(open_price, close)
        upper_shadow = high - body_top
        lower_shadow = body_bottom - low
        body_size = body_top - body_bottom
        
        return ((lower_shadow > 2 * body_size) & (upper_shadow < 0.1 * body_size)).astype(int)
    
    def _is_shooting_star(self, open_price: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Detect Shooting Star candlestick pattern"""
        body_top = np.maximum(open_price, close)
        body_bottom = np.minimum(open_price, close)
        upper_shadow = high - body_top
        lower_shadow = body_bottom - low
        body_size = body_top - body_bottom
        
        return ((upper_shadow > 2 * body_size) & (lower_shadow < 0.1 * body_size)).astype(int)
    
    def _find_support_level(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Find support levels using rolling minimums"""
        return prices.rolling(window, center=True).min()
    
    def _find_resistance_level(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Find resistance levels using rolling maximums"""
        return prices.rolling(window, center=True).max()
    
    def _count_higher_highs(self, high: pd.Series, window: int = 10) -> pd.Series:
        """Count higher highs in rolling window"""
        return high.rolling(window).apply(lambda x: sum(x[i] > x[i-1] for i in range(1, len(x))))
    
    def _count_lower_lows(self, low: pd.Series, window: int = 10) -> pd.Series:
        """Count lower lows in rolling window"""
        return low.rolling(window).apply(lambda x: sum(x[i] < x[i-1] for i in range(1, len(x))))
    
    def _calculate_wave_amplitude(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate wave amplitude"""
        return (prices.rolling(window).max() - prices.rolling(window).min()) / prices.rolling(window).mean()
    
    def _calculate_wave_frequency(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate wave frequency (zero crossings)"""
        detrended = prices - prices.rolling(window).mean()
        return detrended.rolling(window).apply(lambda x: len(np.where(np.diff(np.sign(x)))[0]))
    
    def _seasonal_decomposition(self, prices: pd.Series, period: int = 252) -> pd.Series:
        """Simple seasonal component extraction"""
        if len(prices) < period * 2:
            return pd.Series(0, index=prices.index)
        
        # Simple seasonal decomposition using moving averages
        trend = prices.rolling(period).mean()
        detrended = prices - trend
        seasonal = detrended.rolling(period).mean()
        
        return seasonal.fillna(0)

class FeatureDriftMonitor:
    """ML-based feature drift detection and monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reference_data = {}
        self.drift_thresholds = config.get('drift_thresholds', {
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        })
    
    def set_reference_data(self, ticker: str, features: pd.DataFrame):
        """Set reference data for drift detection"""
        self.reference_data[ticker] = features.copy()
        logger.info(f"Set reference data for {ticker} with {len(features)} samples")
    
    def detect_drift(self, ticker: str, current_features: pd.DataFrame) -> FeatureDriftReport:
        """Detect feature drift using statistical tests and ML methods"""
        
        if ticker not in self.reference_data:
            return FeatureDriftReport(
                ticker=ticker,
                timestamp=datetime.now(),
                drift_detected=False,
                drift_score=0.0,
                affected_features=[],
                drift_details={},
                recommendation="No reference data available"
            )
        
        reference_data = self.reference_data[ticker]
        
        # Statistical drift detection
        drift_results = {}
        affected_features = []
        
        for feature in current_features.columns:
            if feature in reference_data.columns:
                # KS test for distribution drift
                drift_score = self._calculate_drift_score(
                    reference_data[feature].dropna(), 
                    current_features[feature].dropna()
                )
                
                drift_results[feature] = drift_score
                
                if drift_score > self.drift_thresholds['medium']:
                    affected_features.append(feature)
        
        # Overall drift score
        overall_drift_score = np.mean(list(drift_results.values())) if drift_results else 0.0
        drift_detected = overall_drift_score > self.drift_thresholds['medium']
        
        # Generate recommendation
        recommendation = self._generate_drift_recommendation(overall_drift_score, affected_features)
        
        return FeatureDriftReport(
            ticker=ticker,
            timestamp=datetime.now(),
            drift_detected=drift_detected,
            drift_score=overall_drift_score,
            affected_features=affected_features,
            drift_details=drift_results,
            recommendation=recommendation
        )
    
    def _calculate_drift_score(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate drift score between reference and current data"""
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        
        # Statistical distance measures
        from scipy.stats import ks_2samp
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(reference, current)
        
        # Jensen-Shannon divergence (simplified)
        ref_hist, _ = np.histogram(reference, bins=20, density=True)
        cur_hist, _ = np.histogram(current, bins=20, density=True)
        
        # Add small epsilon to avoid log(0)
        ref_hist += 1e-10
        cur_hist += 1e-10
        
        # Normalize
        ref_hist /= ref_hist.sum()
        cur_hist /= cur_hist.sum()
        
        # JS divergence
        m = 0.5 * (ref_hist + cur_hist)
        js_div = 0.5 * np.sum(ref_hist * np.log(ref_hist / m)) + 0.5 * np.sum(cur_hist * np.log(cur_hist / m))
        
        # Combine scores
        drift_score = 0.7 * ks_stat + 0.3 * js_div
        return min(drift_score, 1.0)  # Cap at 1.0
    
    def _generate_drift_recommendation(self, drift_score: float, affected_features: List[str]) -> str:
        """Generate actionable recommendations based on drift analysis"""
        
        if drift_score < self.drift_thresholds['low']:
            return "No action required. Feature distributions are stable."
        elif drift_score < self.drift_thresholds['medium']:
            return f"Minor drift detected in {len(affected_features)} features. Monitor closely."
        elif drift_score < self.drift_thresholds['high']:
            return f"Moderate drift detected. Consider retraining models. Affected features: {affected_features[:5]}"
        else:
            return f"Significant drift detected! Immediate model retraining required. Critical features: {affected_features[:3]}"

# Integration with existing consolidation detector
async def create_enhanced_consolidation_features(ticker: str, price_data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Create enhanced consolidation features using the advanced pipeline"""
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer(config)
    
    # Create comprehensive features
    advanced_features = feature_engineer.create_advanced_features(price_data, ticker)
    
    # Focus on consolidation-specific features
    consolidation_features = advanced_features.filter(regex='kc_|bb_|squeeze_|range_|support_|resistance_|trend_')
    
    logger.info(f"Created {len(consolidation_features.columns)} enhanced consolidation features for {ticker}")
    
    return consolidation_features