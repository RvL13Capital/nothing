# feature_store_integration.py
"""
Integration module connecting enterprise feature store with breakout prediction system
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from enterprise_feature_store import (
    EnterpriseFeatureStore, FeatureGroup, FeatureSchema, FeatureType, 
    TECHNICAL_INDICATORS_GROUP, CONSOLIDATION_GROUP
)
from advanced_feature_pipeline import AdvancedFeatureEngineer, FeatureDriftMonitor
from feature_serving import EnterpriseFeatureServer, FeatureRequest
from config_loader import load_config

logger = logging.getLogger(__name__)

class BreakoutFeatureManager:
    """Manages features specifically for breakout prediction system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_store = None
        self.feature_server = None
        self.feature_engineer = None
        self.drift_monitor = None
        
        # Breakout-specific feature groups
        self.breakout_features = self._define_breakout_feature_groups()
        
    async def initialize(self):
        """Initialize all feature management components"""
        
        # Initialize feature store
        self.feature_store = EnterpriseFeatureStore(self.config.get('feature_store', {}))
        await self.feature_store.initialize()
        
        # Register breakout-specific feature groups
        for feature_group in self.breakout_features:
            self.feature_store.register_feature_group(feature_group)
        
        # Initialize feature server
        self.feature_server = EnterpriseFeatureServer(self.config.get('feature_serving', {}))
        await self.feature_server.initialize()
        
        # Initialize feature engineering pipeline
        self.feature_engineer = AdvancedFeatureEngineer(self.config.get('feature_engineering', {}))
        
        # Initialize drift monitoring
        self.drift_monitor = FeatureDriftMonitor(self.config.get('drift_monitoring', {}))
        
        logger.info("Breakout Feature Manager initialized successfully")
    
    def _define_breakout_feature_groups(self) -> List[FeatureGroup]:
        """Define feature groups specific to breakout prediction"""
        
        # Enhanced consolidation features
        consolidation_enhanced = FeatureGroup(
            name="consolidation_enhanced",
            description="Enhanced consolidation detection features for 40%+ breakout prediction",
            version="2.0.0",
            tags=["consolidation", "breakout", "enhanced"],
            source_tables=["market_data"],
            features=[
                FeatureSchema("consolidation_days", FeatureType.INT, "Days in current consolidation", ["consolidation"]),
                FeatureSchema("range_tightness", FeatureType.FLOAT, "Price range tightness indicator", ["consolidation"]),
                FeatureSchema("volume_contraction", FeatureType.FLOAT, "Volume contraction during consolidation", ["volume"]),
                FeatureSchema("squeeze_intensity", FeatureType.FLOAT, "Intensity of price squeeze", ["squeeze"]),
                FeatureSchema("breakout_readiness", FeatureType.FLOAT, "Overall breakout readiness score", ["prediction"]),
                FeatureSchema("support_strength", FeatureType.FLOAT, "Strength of support level", ["support_resistance"]),
                FeatureSchema("resistance_strength", FeatureType.FLOAT, "Strength of resistance level", ["support_resistance"]),
                FeatureSchema("volatility_contraction", FeatureType.FLOAT, "Volatility contraction ratio", ["volatility"]),
                FeatureSchema("pattern_quality", FeatureType.FLOAT, "Quality score of consolidation pattern", ["pattern"]),
            ]
        )
        
        # Micro-cap specific features
        microcap_features = FeatureGroup(
            name="microcap_indicators",
            description="Features specific to micro and small cap stock behavior",
            version="2.0.0",
            tags=["microcap", "small_cap", "volume"],
            source_tables=["market_data", "fundamental_data"],
            features=[
                FeatureSchema("float_turnover", FeatureType.FLOAT, "Daily volume as % of float", ["volume"]),
                FeatureSchema("relative_volume_spike", FeatureType.FLOAT, "Volume spike relative to average", ["volume"]),
                FeatureSchema("insider_accumulation", FeatureType.FLOAT, "Insider accumulation indicator", ["insider"]),
                FeatureSchema("catalyst_proximity", FeatureType.FLOAT, "Days to next known catalyst", ["catalyst"]),
                FeatureSchema("short_interest_ratio", FeatureType.FLOAT, "Short interest as % of float", ["short_interest"]),
                FeatureSchema("market_cap_category", FeatureType.CATEGORICAL, "Market cap size category", ["fundamental"]),
                FeatureSchema("sector_momentum", FeatureType.FLOAT, "Sector relative momentum", ["sector"]),
                FeatureSchema("news_sentiment", FeatureType.FLOAT, "Recent news sentiment score", ["sentiment"]),
            ]
        )
        
        # Advanced momentum features
        momentum_advanced = FeatureGroup(
            name="momentum_advanced",
            description="Advanced momentum indicators for breakout prediction",
            version="2.0.0",
            tags=["momentum", "advanced", "prediction"],
            source_tables=["market_data"],
            features=[
                FeatureSchema("momentum_divergence", FeatureType.FLOAT, "Price vs indicator divergence", ["divergence"]),
                FeatureSchema("momentum_acceleration", FeatureType.FLOAT, "Rate of momentum change", ["momentum"]),
                FeatureSchema("trend_exhaustion", FeatureType.FLOAT, "Trend exhaustion indicator", ["trend"]),
                FeatureSchema("momentum_thrust", FeatureType.FLOAT, "Momentum thrust indicator", ["momentum"]),
                FeatureSchema("price_momentum_combo", FeatureType.FLOAT, "Combined price-momentum score", ["combo"]),
                FeatureSchema("breakout_momentum", FeatureType.FLOAT, "Momentum specific to breakouts", ["breakout"]),
            ]
        )
        
        return [consolidation_enhanced, microcap_features, momentum_advanced]
    
    async def compute_breakout_features(self, ticker: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive features for breakout prediction"""
        
        start_time = datetime.now()
        
        # Compute base technical features
        technical_result = await self.feature_store.compute_and_store_features(
            ticker, price_data, "technical_indicators"
        )
        
        # Compute consolidation features
        consolidation_result = await self.feature_store.compute_and_store_features(
            ticker, price_data, "consolidation_patterns"
        )
        
        # Compute enhanced features using advanced pipeline
        enhanced_features = self.feature_engineer.create_advanced_features(price_data, ticker)
        
        # Extract breakout-specific features
        breakout_specific = self._extract_breakout_specific_features(enhanced_features, price_data)
        
        # Store enhanced features
        enhanced_result = await self.feature_store.compute_and_store_features(
            ticker, breakout_specific, "consolidation_enhanced"
        )
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Computed breakout features for {ticker} in {computation_time:.2f}s")
        
        return {
            'ticker': ticker,
            'computation_time': computation_time,
            'technical_features': technical_result.get('success', False),
            'consolidation_features': consolidation_result.get('success', False),
            'enhanced_features': enhanced_result.get('success', False),
            'total_features': len(enhanced_features.columns)
        }
    
    def _extract_breakout_specific_features(self, enhanced_features: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract and compute breakout-specific features"""
        
        breakout_df = pd.DataFrame(index=enhanced_features.index)
        
        # Consolidation days (simplified calculation)
        close_prices = price_data['close']
        rolling_high = close_prices.rolling(20).max()
        rolling_low = close_prices.rolling(20).min()
        range_pct = (rolling_high - rolling_low) / rolling_low
        
        # Consolidation detection
        in_consolidation = range_pct < 0.15  # 15% range threshold
        consolidation_periods = in_consolidation.astype(int).rolling(50).sum()
        
        breakout_df['consolidation_days'] = consolidation_periods
        breakout_df['range_tightness'] = 1 / (range_pct + 0.01)  # Higher value = tighter range
        
        # Volume features
        volume = price_data['volume']
        volume_ma = volume.rolling(20).mean()
        breakout_df['volume_contraction'] = 1 - (volume / volume_ma).rolling(10).mean()
        
        # Squeeze intensity
        if 'kc_width' in enhanced_features.columns and 'bb_width' in enhanced_features.columns:
            breakout_df['squeeze_intensity'] = enhanced_features['bb_width'] / (enhanced_features['kc_width'] + 0.001)
        else:
            breakout_df['squeeze_intensity'] = 0.5
        
        # Volatility contraction
        returns = close_prices.pct_change()
        recent_vol = returns.rolling(10).std()
        historical_vol = returns.rolling(50).std()
        breakout_df['volatility_contraction'] = 1 - (recent_vol / historical_vol)
        
        # Support/Resistance strength (simplified)
        support_level = rolling_low
        resistance_level = rolling_high
        
        # Count touches of support/resistance
        support_touches = (close_prices <= support_level * 1.02).rolling(20).sum()
        resistance_touches = (close_prices >= resistance_level * 0.98).rolling(20).sum()
        
        breakout_df['support_strength'] = support_touches / 20
        breakout_df['resistance_strength'] = resistance_touches / 20
        
        # Pattern quality score
        breakout_df['pattern_quality'] = (
            breakout_df['range_tightness'] * 0.3 +
            breakout_df['volume_contraction'] * 0.2 +
            breakout_df['squeeze_intensity'] * 0.2 +
            breakout_df['volatility_contraction'] * 0.2 +
            (breakout_df['support_strength'] + breakout_df['resistance_strength']) * 0.1
        )
        
        # Breakout readiness (composite score)
        breakout_df['breakout_readiness'] = np.clip(
            breakout_df['pattern_quality'] * 
            (breakout_df['consolidation_days'] / 50) *  # Longer consolidation = higher readiness
            (1 + breakout_df['volume_contraction']),  # Volume drying up = higher readiness
            0, 1
        )
        
        return breakout_df.fillna(0)
    
    async def get_inference_features(self, ticker: str, feature_names: List[str] = None) -> Dict[str, Any]:
        """Get features for real-time inference with consistency guarantees"""
        
        if feature_names is None:
            # Default breakout prediction features
            feature_names = [
                'price_close', 'rsi_14', 'atr_14', 'volume_ratio',
                'kc_position', 'squeeze_indicator', 'consolidation_days',
                'range_tightness', 'breakout_readiness', 'pattern_quality'
            ]
        
        # Create feature request
        request = FeatureRequest(
            ticker=ticker,
            feature_names=feature_names,
            max_age_seconds=300,  # 5 minutes max age for real-time inference
            include_metadata=True
        )
        
        # Serve features through high-performance serving layer
        response = await self.feature_server.serve_features(request)
        
        return {
            'ticker': response.ticker,
            'features': response.features,
            'cache_hit': response.cache_hit,
            'latency_ms': response.latency_ms,
            'timestamp': response.timestamp
        }
    
    async def monitor_feature_drift(self, ticker: str) -> Dict[str, Any]:
        """Monitor feature drift for a specific ticker"""
        
        # Get recent features for comparison
        recent_features = await self.get_inference_features(ticker)
        
        if not recent_features['features']:
            return {'drift_detected': False, 'message': 'No features available for drift monitoring'}
        
        # Convert to DataFrame for drift detection
        current_df = pd.DataFrame([recent_features['features']])
        
        # Perform drift detection
        drift_report = self.drift_monitor.detect_drift(ticker, current_df)
        
        return {
            'ticker': ticker,
            'drift_detected': drift_report.drift_detected,
            'drift_score': drift_report.drift_score,
            'affected_features': drift_report.affected_features,
            'recommendation': drift_report.recommendation,
            'timestamp': drift_report.timestamp
        }
    
    async def batch_compute_features(self, tickers: List[str], price_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Compute features for multiple tickers in parallel"""
        
        tasks = []
        for ticker in tickers:
            if ticker in price_data_dict:
                task = self.compute_breakout_features(ticker, price_data_dict[ticker])
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        success_count = 0
        error_count = 0
        results_dict = {}
        
        for i, result in enumerate(results):
            ticker = tickers[i]
            if isinstance(result, Exception):
                logger.error(f"Error computing features for {ticker}: {result}")
                error_count += 1
                results_dict[ticker] = {'error': str(result)}
            else:
                success_count += 1
                results_dict[ticker] = result
        
        return {
            'total_tickers': len(tickers),
            'successful': success_count,
            'errors': error_count,
            'results': results_dict
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        metrics = {}
        
        # Feature store metrics
        if self.feature_store:
            metrics['feature_store'] = self.feature_store.get_metrics()
        
        # Feature server metrics
        if self.feature_server:
            metrics['feature_server'] = self.feature_server.get_performance_metrics()
        
        return metrics

# Integration with existing API
async def integrate_with_breakout_api(app, feature_manager: BreakoutFeatureManager):
    """Integrate feature store with existing breakout API"""
    
    @app.route('/features/<ticker>', methods=['GET'])
    async def get_ticker_features(ticker: str):
        """Get features for a specific ticker"""
        
        try:
            features = await feature_manager.get_inference_features(ticker.upper())
            return {
                'success': True,
                'data': features
            }
        except Exception as e:
            logger.error(f"Error getting features for {ticker}: {e}")
            return {
                'success': False,
                'error': str(e)
            }, 500
    
    @app.route('/features/<ticker>/drift', methods=['GET'])
    async def check_feature_drift(ticker: str):
        """Check for feature drift"""
        
        try:
            drift_report = await feature_manager.monitor_feature_drift(ticker.upper())
            return {
                'success': True,
                'data': drift_report
            }
        except Exception as e:
            logger.error(f"Error checking drift for {ticker}: {e}")
            return {
                'success': False,
                'error': str(e)
            }, 500
    
    @app.route('/features/metrics', methods=['GET'])
    async def get_feature_metrics():
        """Get feature store performance metrics"""
        
        try:
            metrics = feature_manager.get_performance_metrics()
            return {
                'success': True,
                'data': metrics
            }
        except Exception as e:
            logger.error(f"Error getting feature metrics: {e}")
            return {
                'success': False,
                'error': str(e)
            }, 500

# Initialization function
async def initialize_breakout_feature_store(config: Dict[str, Any] = None) -> BreakoutFeatureManager:
    """Initialize the complete breakout feature store system"""
    
    if config is None:
        config = load_config()
    
    # Create and initialize feature manager
    feature_manager = BreakoutFeatureManager(config)
    await feature_manager.initialize()
    
    logger.info("Breakout Feature Store system ready for production")
    return feature_manager