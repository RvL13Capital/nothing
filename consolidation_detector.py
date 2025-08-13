# consolidation_detector.py
"""
Advanced Consolidation Detection Module
Optimized for Micro/Small Cap Stocks
Focus on 40%+ Breakout Potential
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConsolidationPattern:
    """Detailed consolidation pattern data"""
    ticker: str
    start_date: datetime
    end_date: Optional[datetime]
    duration_days: int
    lower_bound: float
    upper_bound: float
    range_percent: float
    position_in_range: float
    boundary_tests: Dict
    volume_characteristics: Dict
    keltner_squeeze: bool
    ttm_squeeze: bool
    quality_score: float
    breakout_readiness: float

class ConsolidationDetector:
    """
    Robust Consolidation Detection with Keltner Channels
    NO volume filtering in detection - pure price action
    Volume data collected for ML models
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Core parameters
        self.MIN_HISTORY_DAYS = 252
        self.ROLLING_THRESHOLD_WINDOW = 126
        self.STD_DEV_WINDOW = 20
        self.ATR_WINDOW = 14
        self.STD_DEV_QUANTILE = 0.25
        self.ATR_QUANTILE = 0.25
        self.CLIP_PERCENTILE = 0.95
        self.START_LONG_CONSOLIDATION_STREAK = 10
        self.SET_BOUNDS_STREAK = 20
        self.ADJUST_BOUNDS_STREAKS = [40, 60, 100]
        
        # Keltner Channel parameters
        self.KC_PERIOD = 20
        self.KC_ATR_MULTIPLIER = 2.0
        
        # Override with config if provided
        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def detect_consolidations(self, ticker: str, df: pd.DataFrame) -> Dict:
        """
        Main consolidation detection with comprehensive analysis
        """
        
        if len(df) < self.MIN_HISTORY_DAYS:
            return {'ticker': ticker, 'in_consolidation': False, 'error': 'Insufficient data'}
        
        # Ensure proper column names
        df = self._normalize_dataframe(df)
        
        # Calculate core indicators
        df = self._calculate_indicators(df)
        
        # Detect consolidation periods
        consolidation_data = self._detect_consolidation_periods(df)
        
        # Add current market state
        current_state = self._analyze_current_state(df, consolidation_data)
        
        # Calculate volume profile (for ML, not for detection)
        volume_profile = self._analyze_volume_profile(df, consolidation_data)
        
        # Assess breakout readiness
        if current_state['in_consolidation']:
            readiness = self._assess_breakout_readiness(df, current_state)
            current_state['breakout_readiness'] = readiness
        
        return {
            'ticker': ticker,
            'in_consolidation': current_state['in_consolidation'],
            'current_consolidation': current_state.get('current_pattern'),
            'consolidation_history': consolidation_data['periods'],
            'volume_profile': volume_profile,
            'indicators': {
                'keltner_squeeze': current_state.get('keltner_squeeze', False),
                'ttm_squeeze': current_state.get('ttm_squeeze', False),
                'atr_contraction': current_state.get('atr_contraction', 0),
                'range_contraction': current_state.get('range_contraction', 0)
            },
            'df': df  # Return enhanced dataframe for ML models
        }
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent column naming"""
        df = df.copy()
        df.columns = df.columns.str.lower()
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        
        # ATR and volatility
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, 
                              timeperiod=self.ATR_WINDOW)
        df['atr_normalized'] = df['atr'] / df['close']
        df['std_dev'] = df['close'].rolling(window=self.STD_DEV_WINDOW).std()
        
        # Keltner Channels (PRIMARY INDICATOR)
        df = self._calculate_keltner_channels(df)
        
        # Bollinger Bands (for TTM Squeeze)
        upper_bb, middle_bb, lower_bb = talib.BBANDS(
            df['close'].values, 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2
        )
        df['bb_upper'] = upper_bb
        df['bb_middle'] = middle_bb
        df['bb_lower'] = lower_bb
        df['bb_width'] = (upper_bb - lower_bb) / middle_bb
        
        # TTM Squeeze (Bollinger inside Keltner)
        df['ttm_squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
        
        # Keltner Squeeze (Price compressing within Keltner)
        kc_width = df['kc_upper'] - df['kc_lower']
        historical_width = kc_width.rolling(50).mean()
        df['keltner_squeeze'] = kc_width < (historical_width * 0.7)
        
        # Price efficiency
        df['price_efficiency'] = self._calculate_price_efficiency(df)
        
        # Volume metrics (collected but not used for detection)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
        df['cmf'] = self._calculate_cmf(df)
        df['mfi'] = talib.MFI(df['high'].values, df['low'].values, 
                              df['close'].values, df['volume'].values, timeperiod=14)
        
        return df
    
    def _calculate_keltner_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channels - Better for micro caps"""
        
        # EMA of close
        ema = talib.EMA(df['close'].values, timeperiod=self.KC_PERIOD)
        
        # ATR for channel width
        atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, 
                       timeperiod=self.KC_PERIOD)
        
        # Calculate channels
        df['kc_middle'] = ema
        df['kc_upper'] = ema + (self.KC_ATR_MULTIPLIER * atr)
        df['kc_lower'] = ema - (self.KC_ATR_MULTIPLIER * atr)
        df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
        
        # Position within Keltner Channel
        df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
        df['kc_position'] = df['kc_position'].clip(0, 1)
        
        return df
    
    def _calculate_price_efficiency(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Kaufman's Efficiency Ratio - measures trend strength"""
        direction = abs(df['close'] - df['close'].shift(period))
        volatility = abs(df['close'].diff()).rolling(period).sum()
        efficiency = direction / volatility
        return efficiency.fillna(0)
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Chaikin Money Flow - measures accumulation/distribution"""
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
                       (df['high'] - df['low'])
        mf_multiplier = mf_multiplier.fillna(0)
        mf_volume = mf_multiplier * df['volume']
        cmf = mf_volume.rolling(period).sum() / df['volume'].rolling(period).sum()
        return cmf.fillna(0)
    
    def _detect_consolidation_periods(self, df: pd.DataFrame) -> Dict:
        """Core consolidation detection logic"""
        
        consolidation_periods = []
        consolidation_streak = 0
        in_long_consolidation = False
        lower_bound, upper_bound = None, None
        current_period = {}
        boundary_history = []
        
        for i in range(self.ROLLING_THRESHOLD_WINDOW, len(df)):
            # Calculate adaptive thresholds
            history_slice = df.iloc[i - self.ROLLING_THRESHOLD_WINDOW:i]
            
            std_dev_threshold = history_slice['std_dev'].clip(
                upper=history_slice['std_dev'].quantile(self.CLIP_PERCENTILE)
            ).quantile(self.STD_DEV_QUANTILE)
            
            atr_threshold = history_slice['atr_normalized'].clip(
                upper=history_slice['atr_normalized'].quantile(self.CLIP_PERCENTILE)
            ).quantile(self.ATR_QUANTILE)
            
            # Check consolidation conditions
            is_consolidating = (
                (df['std_dev'].iloc[i] < std_dev_threshold) and 
                (df['atr_normalized'].iloc[i] < atr_threshold)
            )
            
            # Additional Keltner-based confirmation
            if is_consolidating and i > 20:
                # Check if price is contained within contracting Keltner Channels
                recent_kc_width = df['kc_width'].iloc[i-10:i].mean()
                prior_kc_width = df['kc_width'].iloc[i-30:i-10].mean()
                is_consolidating = is_consolidating and (recent_kc_width < prior_kc_width)
            
            previous_in_long_consolidation = in_long_consolidation
            
            # Check for breakout if in consolidation
            if in_long_consolidation and lower_bound is not None:
                breakout_detected = self._check_breakout(
                    df.iloc[i], lower_bound, upper_bound
                )
                
                if breakout_detected:
                    in_long_consolidation = False
                    consolidation_streak = 0
                    current_period['end'] = df.index[i]
                    current_period['end_idx'] = i
                    current_period['breakout'] = breakout_detected
                    
                    if 'start' in current_period:
                        # Calculate pattern quality
                        current_period['quality_score'] = self._calculate_quality_score(
                            df.iloc[current_period['start_idx']:i+1],
                            current_period
                        )
                        consolidation_periods.append(current_period.copy())
                    
                    current_period = {}
                    lower_bound, upper_bound = None, None
                    boundary_history = []
            
            # Update consolidation streak
            if is_consolidating:
                consolidation_streak += 1
            else:
                if in_long_consolidation:
                    current_period['end'] = df.index[i]
                    current_period['end_idx'] = i
                    
                    if 'start' in current_period:
                        current_period['quality_score'] = self._calculate_quality_score(
                            df.iloc[current_period['start_idx']:i+1],
                            current_period
                        )
                        consolidation_periods.append(current_period.copy())
                    
                    current_period = {}
                    lower_bound, upper_bound = None, None
                    boundary_history = []
                
                in_long_consolidation = False
                consolidation_streak = 0
            
            # Check for new consolidation start
            if not previous_in_long_consolidation and consolidation_streak >= self.START_LONG_CONSOLIDATION_STREAK:
                in_long_consolidation = True
                current_period = {
                    'start': df.index[i - consolidation_streak + 1],
                    'start_idx': i - consolidation_streak + 1,
                    'initial_price': df['close'].iloc[i - consolidation_streak + 1],
                    'boundaries': {'dates': [], 'lower': [], 'upper': []},
                    'volume_profile': {}
                }
            
            # Set and adjust boundaries
            if in_long_consolidation:
                if consolidation_streak == self.SET_BOUNDS_STREAK:
                    bounds_slice = df.iloc[i - self.SET_BOUNDS_STREAK + 1:i + 1]
                    lower_bound = bounds_slice['low'].min()
                    upper_bound = bounds_slice['high'].max()
                    
                    boundary_history.append({
                        'day': consolidation_streak,
                        'lower': lower_bound,
                        'upper': upper_bound,
                        'range_pct': ((upper_bound - lower_bound) / lower_bound) * 100,
                        'kc_width': df['kc_width'].iloc[i]
                    })
                    
                elif consolidation_streak in self.ADJUST_BOUNDS_STREAKS and lower_bound is not None:
                    bounds_slice = df.iloc[i - consolidation_streak + 1:i + 1]
                    new_lower = bounds_slice['low'].min()
                    new_upper = bounds_slice['high'].max()
                    
                    # Tighten boundaries (never expand)
                    lower_bound = max(lower_bound, new_lower)
                    upper_bound = min(upper_bound, new_upper)
                    
                    boundary_history.append({
                        'day': consolidation_streak,
                        'lower': lower_bound,
                        'upper': upper_bound,
                        'range_pct': ((upper_bound - lower_bound) / lower_bound) * 100,
                        'kc_width': df['kc_width'].iloc[i]
                    })
                
                # Record boundaries
                if upper_bound is not None:
                    current_period['boundaries']['dates'].append(df.index[i])
                    current_period['boundaries']['upper'].append(upper_bound)
                    current_period['boundaries']['lower'].append(lower_bound)
                    current_period['boundary_history'] = boundary_history
        
        # Handle ongoing consolidation
        if in_long_consolidation and 'start' in current_period:
            current_period['end'] = df.index[-1]
            current_period['end_idx'] = len(df) - 1
            current_period['ongoing'] = True
            current_period['days'] = consolidation_streak
            current_period['lower_bound'] = lower_bound
            current_period['upper_bound'] = upper_bound
            
            if lower_bound and upper_bound:
                current_period['range_percent'] = ((upper_bound - lower_bound) / lower_bound) * 100
                current_period['position_in_range'] = ((df['close'].iloc[-1] - lower_bound) / 
                                                       (upper_bound - lower_bound))
            
            current_period['quality_score'] = self._calculate_quality_score(
                df.iloc[current_period['start_idx']:],
                current_period
            )
            consolidation_periods.append(current_period)
        
        return {
            'periods': consolidation_periods,
            'current_streak': consolidation_streak if in_long_consolidation else 0,
            'in_consolidation': in_long_consolidation
        }
    
    def _check_breakout(self, current_row: pd.Series, lower_bound: float, 
                       upper_bound: float) -> Optional[str]:
        """Check for breakout from consolidation"""
        
        # Upside breakout
        if current_row['close'] > upper_bound * 1.02:  # 2% above resistance
            return 'bullish'
        
        # Downside breakdown
        if current_row['close'] < lower_bound * 0.98:  # 2% below support
            return 'bearish'
        
        return None
    
    def _calculate_quality_score(self, window_df: pd.DataFrame, period_data: Dict) -> float:
        """Calculate consolidation quality score"""
        
        scores = []
        
        # 1. Range tightness
        if 'range_percent' in period_data:
            range_score = max(0, 1 - period_data['range_percent'] / 30)
            scores.append(range_score)
        
        # 2. Keltner Channel contraction
        if 'kc_width' in window_df.columns:
            kc_contraction = 1 - (window_df['kc_width'].iloc[-1] / window_df['kc_width'].iloc[0])
            scores.append(max(0, kc_contraction))
        
        # 3. TTM Squeeze presence
        if 'ttm_squeeze' in window_df.columns:
            squeeze_ratio = window_df['ttm_squeeze'].sum() / len(window_df)
            scores.append(squeeze_ratio)
        
        # 4. Price efficiency (low = consolidating)
        if 'price_efficiency' in window_df.columns:
            efficiency_score = 1 - window_df['price_efficiency'].mean()
            scores.append(max(0, efficiency_score))
        
        # 5. Support/Resistance tests
        if period_data.get('lower_bound') and period_data.get('upper_bound'):
            support_tests = (window_df['low'] <= period_data['lower_bound'] * 1.02).sum()
            resistance_tests = (window_df['high'] >= period_data['upper_bound'] * 0.98).sum()
            test_score = min((support_tests + resistance_tests) / 20, 1.0)
            scores.append(test_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _analyze_current_state(self, df: pd.DataFrame, consolidation_data: Dict) -> Dict:
        """Analyze current market state"""
        
        state = {
            'in_consolidation': consolidation_data['in_consolidation'],
            'consolidation_days': consolidation_data['current_streak']
        }
        
        if consolidation_data['in_consolidation'] and consolidation_data['periods']:
            current = consolidation_data['periods'][-1]
            
            state['current_pattern'] = ConsolidationPattern(
                ticker='',  # Will be filled by caller
                start_date=current['start'],
                end_date=current.get('end'),
                duration_days=current.get('days', consolidation_data['current_streak']),
                lower_bound=current.get('lower_bound', 0),
                upper_bound=current.get('upper_bound', 0),
                range_percent=current.get('range_percent', 0),
                position_in_range=current.get('position_in_range', 0.5),
                boundary_tests=self._count_boundary_tests(df, current),
                volume_characteristics=self._analyze_volume_characteristics(df, current),
                keltner_squeeze=df['keltner_squeeze'].iloc[-1] if 'keltner_squeeze' in df else False,
                ttm_squeeze=df['ttm_squeeze'].iloc[-1] if 'ttm_squeeze' in df else False,
                quality_score=current.get('quality_score', 0.5),
                breakout_readiness=0  # Will be calculated
            )
            
            # Current indicators
            state['keltner_squeeze'] = df['keltner_squeeze'].iloc[-1]
            state['ttm_squeeze'] = df['ttm_squeeze'].iloc[-1]
            state['atr_contraction'] = 1 - (df['atr'].iloc[-1] / df['atr'].rolling(50).mean().iloc[-1])
            state['range_contraction'] = 1 - (current.get('range_percent', 20) / 20)
        
        return state
    
    def _count_boundary_tests(self, df: pd.DataFrame, period: Dict) -> Dict:
        """Count support and resistance tests"""
        
        if not period.get('lower_bound') or not period.get('upper_bound'):
            return {'support_tests': 0, 'resistance_tests': 0}
        
        window = df.iloc[period['start_idx']:period.get('end_idx', len(df))]
        
        support_tests = (window['low'] <= period['lower_bound'] * 1.02).sum()
        resistance_tests = (window['high'] >= period['upper_bound'] * 0.98).sum()
        
        return {
            'support_tests': int(support_tests),
            'resistance_tests': int(resistance_tests),
            'total_tests': int(support_tests + resistance_tests)
        }
    
    def _analyze_volume_characteristics(self, df: pd.DataFrame, period: Dict) -> Dict:
        """Analyze volume patterns during consolidation (for ML features)"""
        
        window = df.iloc[period['start_idx']:period.get('end_idx', len(df))]
        
        return {
            'avg_volume': float(window['volume'].mean()),
            'volume_trend': float(np.polyfit(range(len(window)), window['volume'].values, 1)[0]),
            'volume_volatility': float(window['volume'].std() / window['volume'].mean()),
            'obv_trend': float(np.polyfit(range(len(window)), window['obv'].values, 1)[0]) if 'obv' in window else 0,
            'cmf_avg': float(window['cmf'].mean()) if 'cmf' in window else 0,
            'mfi_avg': float(window['mfi'].mean()) if 'mfi' in window else 50
        }
    
    def _analyze_volume_profile(self, df: pd.DataFrame, consolidation_data: Dict) -> Dict:
        """Create volume profile for ML models"""
        
        if not consolidation_data['in_consolidation'] or not consolidation_data['periods']:
            return {}
        
        current = consolidation_data['periods'][-1]
        window = df.iloc[current['start_idx']:current.get('end_idx', len(df))]
        
        # Volume at price levels
        price_bins = pd.qcut(window['close'], q=10, duplicates='drop')
        volume_by_price = window.groupby(price_bins)['volume'].agg(['sum', 'mean', 'std'])
        
        # Find high volume nodes
        poc_idx = volume_by_price['sum'].idxmax()  # Point of Control
        
        profile = {
            'poc_price': float(poc_idx.mid) if hasattr(poc_idx, 'mid') else float(window['close'].median()),
            'poc_volume': float(volume_by_price.loc[poc_idx, 'sum']),
            'volume_skew': float(window['volume'].skew()),
            'volume_kurtosis': float(window['volume'].kurtosis()),
            'accumulation_days': int((window['close'] > window['close'].shift(1)).sum()),
            'distribution_days': int((window['close'] < window['close'].shift(1)).sum())
        }
        
        return profile
    
    def _assess_breakout_readiness(self, df: pd.DataFrame, current_state: Dict) -> float:
        """Assess how ready the consolidation is for breakout"""
        
        readiness_score = 0
        weights = []
        
        pattern = current_state.get('current_pattern')
        if not pattern:
            return 0
        
        # 1. Duration (optimal: 20-40 days)
        if 20 <= pattern.duration_days <= 40:
            readiness_score += 1.0
            weights.append(1.0)
        elif pattern.duration_days < 20:
            readiness_score += pattern.duration_days / 20
            weights.append(1.0)
        else:
            readiness_score += max(0, 1 - (pattern.duration_days - 40) / 40)
            weights.append(1.0)
        
        # 2. Range contraction
        if pattern.range_percent < 10:
            readiness_score += 1.0
            weights.append(1.2)  # Higher weight
        else:
            readiness_score += max(0, 1 - pattern.range_percent / 30)
            weights.append(1.2)
        
        # 3. Squeeze indicators
        if pattern.ttm_squeeze:
            readiness_score += 1.0
            weights.append(1.5)  # Highest weight
        if pattern.keltner_squeeze:
            readiness_score += 1.0
            weights.append(1.3)
        
        # 4. Position in range (near resistance is ready)
        if pattern.position_in_range > 0.7:
            readiness_score += 1.0
            weights.append(1.0)
        elif pattern.position_in_range > 0.5:
            readiness_score += 0.5
            weights.append(1.0)
        
        # 5. Volume characteristics
        vol_chars = pattern.volume_characteristics
        if vol_chars.get('cmf_avg', 0) > 0:  # Positive money flow
            readiness_score += 1.0
            weights.append(1.1)
        
        # Calculate weighted average
        if weights:
            total_score = sum(s * w for s, w in zip([readiness_score], weights))
            total_weight = sum(weights)
            return min(1.0, total_score / total_weight)
        
        return 0.5
