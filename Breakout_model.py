# breakout_system.py
"""
Integrated Breakout Detection System
Combines all modules for production use
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

# Import our modules
from consolidation_detector import ConsolidationDetector
from lstm_breakout_model import LSTMTrainer, BreakoutLSTMSeq2Seq
from xgboost_breakout_model import XGBoostBreakoutModel, MicroCapFeatureEngineer

logger = logging.getLogger(__name__)

@dataclass
class BreakoutSignal:
    """Complete breakout signal with all information"""
    ticker: str
    timestamp: datetime
    confidence: float
    breakout_probability: float
    expected_magnitude: float
    expected_days: float
    volume_surge_expected: float
    signal_strength: str  # 'STRONG', 'MODERATE', 'WEAK'
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    key_factors: List[str]
    model_scores: Dict[str, float]
    consolidation_info: Dict
    entry_price: float
    stop_loss: float
    target_prices: List[float]
    
    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp.isoformat(),
            'confidence': round(self.confidence, 2),
            'breakout_probability': round(self.breakout_probability, 4),
            'expected_magnitude': round(self.expected_magnitude, 4),
            'expected_days': round(self.expected_days, 1),
            'volume_surge_expected': round(self.volume_surge_expected, 2),
            'signal_strength': self.signal_strength,
            'risk_level': self.risk_level,
            'key_factors': self.key_factors,
            'model_scores': {k: round(v, 4) for k, v in self.model_scores.items()},
            'entry_price': round(self.entry_price, 2),
            'stop_loss': round(self.stop_loss, 2),
            'target_prices': [round(p, 2) for p in self.target_prices]
        }

class BreakoutDetectionSystem:
    """
    Complete integrated system for breakout detection
    """
    
    def __init__(self, model_path: str = "models"):
        """Initialize all components"""
        
        # Core components
        self.consolidation_detector = ConsolidationDetector()
        self.feature_engineer = MicroCapFeatureEngineer()
        self.xgboost_model = XGBoostBreakoutModel()
        self.lstm_trainer = None  # Will be loaded
        
        # Configuration
        self.min_confidence = 0.60  # 60% minimum confidence
        self.market_cap_range = (10e6, 2e9)  # $10M - $2B
        
        # Load models
        self.model_path = model_path
        self._load_models()
        
        # Performance tracking
        self.predictions_history = []
        self.performance_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'pending_predictions': []
        }
        
        logger.info("Breakout Detection System initialized")
    
    def _load_models(self):
        """Load trained models"""
        try:
            # Load XGBoost
            self.xgboost_model.load_model(f"{self.model_path}/xgboost")
            logger.info("XGBoost model loaded")
            
            # Load LSTM
            self.lstm_trainer = LSTMTrainer(self.xgboost_model.selected_features)
            self.lstm_trainer.load_model(f"{self.model_path}/lstm")
            logger.info("LSTM model loaded")
            
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            logger.info("System will need training before use")
    
    async def scan_watchlist(self, tickers: List[str]) -> List[BreakoutSignal]:
        """
        Scan a watchlist for breakout opportunities
        """
        
        signals = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for ticker in tickers:
                future = executor.submit(self._analyze_ticker, ticker)
                futures.append((ticker, future))
            
            for ticker, future in futures:
                try:
                    signal = future.result(timeout=30)
                    if signal and signal.confidence >= self.min_confidence:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error analyzing {ticker}: {e}")
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Found {len(signals)} high-confidence signals from {len(tickers)} tickers")
        
        return signals
    
    def _analyze_ticker(self, ticker: str) -> Optional[BreakoutSignal]:
        """Analyze single ticker through complete pipeline"""
        
        try:
            # Get data
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            
            if len(df) < 252:
                return None
            
            # Check market cap
            info = stock.info
            market_cap = info.get('marketCap', 0)
            
            if not (self.market_cap_range[0] <= market_cap <= self.market_cap_range[1]):
                logger.debug(f"{ticker}: Outside market cap range ({market_cap/1e6:.1f}M)")
                return None
            
            # Step 1: Detect consolidation
            consolidation_data = self.consolidation_detector.detect_consolidations(ticker, df)
            
            if not consolidation_data.get('in_consolidation'):
                logger.debug(f"{ticker}: Not in consolidation")
                return None
            
            # Step 2: Create features
            features_df = self.feature_engineer.create_features(
                consolidation_data['df'],
                consolidation_data
            )
            
            # Step 3: Get XGBoost prediction
            xgb_prob = self.xgboost_model.predict(features_df)[0]
            
            # Step 4: Get LSTM prediction
            # Prepare sequence data (simplified - in production, maintain historical features)
            feature_history = self._prepare_feature_history(consolidation_data['df'], consolidation_data)
            lstm_output, attention = self.lstm_trainer.predict(feature_history)
            
            lstm_prob = lstm_output[0]
            lstm_magnitude = lstm_output[1]
            lstm_days = lstm_output[2]
            lstm_volume = lstm_output[3]
            
            # Step 5: Ensemble predictions
            ensemble_prob = (xgb_prob * 0.6 + lstm_prob * 0.4)  # XGBoost weighted higher
            
            # Step 6: Calculate confidence
            confidence = self._calculate_confidence(
                ensemble_prob, xgb_prob, lstm_prob,
                consolidation_data
            )
            
            # Step 7: Determine signal characteristics
            signal_strength = self._determine_signal_strength(confidence)
            risk_level = self._assess_risk(consolidation_data, ensemble_prob)
            
            # Step 8: Calculate entry and targets
            current_price = df['close'].iloc[-1]
            entry, stop_loss, targets = self._calculate_levels(
                current_price,
                consolidation_data,
                lstm_magnitude
            )
            
            # Step 9: Identify key factors
            key_factors = self._identify_key_factors(
                features_df,
                consolidation_data,
                xgb_prob,
                lstm_prob
            )
            
            # Create signal
            signal = BreakoutSignal(
                ticker=ticker,
                timestamp=datetime.now(),
                confidence=confidence,
                breakout_probability=ensemble_prob,
                expected_magnitude=lstm_magnitude,
                expected_days=lstm_days,
                volume_surge_expected=lstm_volume,
                signal_strength=signal_strength,
                risk_level=risk_level,
                key_factors=key_factors,
                model_scores={
                    'xgboost': xgb_prob,
                    'lstm': lstm_prob,
                    'ensemble': ensemble_prob
                },
                consolidation_info={
                    'days': consolidation_data['current_consolidation'].duration_days,
                    'range_percent': consolidation_data['current_consolidation'].range_percent,
                    'quality_score': consolidation_data['current_consolidation'].quality_score
                },
                entry_price=entry,
                stop_loss=stop_loss,
                target_prices=targets
            )
            
            # Store for tracking
            self._store_prediction(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None
    
    def _prepare_feature_history(self, df: pd.DataFrame, 
                                consolidation_data: Dict) -> np.ndarray:
        """Prepare feature history for LSTM"""
        
        # Simplified - create features for last N days
        sequence_length = self.lstm_trainer.best_config.get('sequence_length', 30)
        
        features_list = []
        
        for i in range(max(0, len(df) - sequence_length), len(df)):
            # Create point-in-time features
            temp_df = df.iloc[:i+1]
            temp_cons_data = consolidation_data.copy()  # Simplified
            
            features = self.feature_engineer.create_features(temp_df, temp_cons_data)
            features_list.append(features.values[0])
        
        # Pad if necessary
        while len(features_list) < sequence_length:
            features_list.insert(0, np.zeros_like(features_list[0] if features_list else np.zeros(50)))
        
        return np.array(features_list)
    
    def _calculate_confidence(self, ensemble_prob: float, xgb_prob: float,
                            lstm_prob: float, consolidation_data: Dict) -> float:
        """Calculate overall confidence score"""
        
        # Base confidence from ensemble
        confidence = ensemble_prob * 100
        
        # Boost if models agree
        model_agreement = 1 - abs(xgb_prob - lstm_prob)
        if model_agreement > 0.8:
            confidence *= 1.1
        
        # Factor in consolidation quality
        quality = consolidation_data['current_consolidation'].quality_score
        confidence *= (0.7 + 0.3 * quality)
        
        # Factor in breakout readiness
        readiness = consolidation_data['current_consolidation'].breakout_readiness
        confidence *= (0.8 + 0.2 * readiness)
        
        return min(100, max(0, confidence))
    
    def _determine_signal_strength(self, confidence: float) -> str:
        """Determine signal strength category"""
        
        if confidence >= 80:
            return 'STRONG'
        elif confidence >= 65:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def _assess_risk(self, consolidation_data: Dict, probability: float) -> str:
        """Assess risk level of the signal"""
        
        risk_score = 0
        
        # Position in range
        position = consolidation_data['current_consolidation'].position_in_range
        if position > 0.8:
            risk_score += 1  # Near resistance
        elif position < 0.3:
            risk_score -= 1  # Near support (lower risk)
        
        # Consolidation duration
        days = consolidation_data['current_consolidation'].duration_days
        if days > 80:
            risk_score += 1  # Old consolidation
        elif 20 <= days <= 40:
            risk_score -= 1  # Optimal duration
        
        # Probability confidence
        if probability < 0.65:
            risk_score += 1
        elif probability > 0.80:
            risk_score -= 1
        
        # Determine risk level
        if risk_score <= -1:
            return 'LOW'
        elif risk_score >= 1:
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def _calculate_levels(self, current_price: float, 
                         consolidation_data: Dict,
                         expected_magnitude: float) -> Tuple[float, float, List[float]]:
        """Calculate entry, stop loss, and target prices"""
        
        cons = consolidation_data['current_consolidation']
        
        # Entry (current price or slightly above for confirmation)
        entry = current_price * 1.01  # 1% above for confirmation
        
        # Stop loss (below support)
        stop_loss = cons.lower_bound * 0.97  # 3% below support
        
        # Targets based on expected magnitude
        target1 = entry * (1 + expected_magnitude * 0.4)  # 40% of expected move
        target2 = entry * (1 + expected_magnitude * 0.7)  # 70% of expected move
        target3 = entry * (1 + expected_magnitude)  # Full expected move
        
        return entry, stop_loss, [target1, target2, target3]
    
    def _identify_key_factors(self, features_df: pd.DataFrame,
                             consolidation_data: Dict,
                             xgb_prob: float, lstm_prob: float) -> List[str]:
        """Identify key factors driving the signal"""
        
        factors = []
        
        # Consolidation factors
        cons = consolidation_data['current_consolidation']
        
        if cons.range_percent < 10:
            factors.append(f"Very tight range ({cons.range_percent:.1f}%)")
        
        if cons.ttm_squeeze:
            factors.append("TTM Squeeze active")
        
        if cons.keltner_squeeze:
            factors.append("Keltner Channel squeeze")
        
        # Volume factors
        if features_df['cmf_avg'].iloc[0] > 0.1:
            factors.append("Positive money flow (accumulation)")
        
        if features_df['volume_ratio_5d'].iloc[0] > 1.2:
            factors.append("Rising volume pattern")
        
        # Technical factors
        if features_df['rsi'].iloc[0] > 50 and features_df['rsi'].iloc[0] < 70:
            factors.append(f"Bullish RSI ({features_df['rsi'].iloc[0]:.0f})")
        
        # Model agreement
        if abs(xgb_prob - lstm_prob) < 0.1:
            factors.append("Strong model consensus")
        
        return factors[:5]  # Top 5 factors
    
    def _store_prediction(self, signal: BreakoutSignal):
        """Store prediction for tracking"""
        
        self.predictions_history.append(signal.to_dict())
        self.performance_metrics['total_predictions'] += 1
        self.performance_metrics['pending_predictions'].append({
            'ticker': signal.ticker,
            'date': signal.timestamp,
            'expected_days': signal.expected_days,
            'expected_magnitude': signal.expected_magnitude
        })
        
        # Keep only last 1000 predictions
        if len(self.predictions_history) > 1000:
            self.predictions_history = self.predictions_history[-1000:]
    
    def validate_prediction(self, ticker: str, actual_move: float, days_taken: int):
        """Validate a past prediction"""
        
        # Find pending prediction
        for pred in self.performance_metrics['pending_predictions']:
            if pred['ticker'] == ticker:
                # Check if prediction was correct
                if actual_move >= pred['expected_magnitude'] * 0.7:  # 70% of expected
                    self.performance_metrics['successful_predictions'] += 1
                
                # Remove from pending
                self.performance_metrics['pending_predictions'].remove(pred)
                break
        
        # Trigger self-optimization if needed
        success_rate = (self.performance_metrics['successful_predictions'] / 
                       max(self.performance_metrics['total_predictions'], 1))
        
        if success_rate < 0.5 and self.performance_metrics['total_predictions'] > 20:
            logger.warning(f"Success rate low ({success_rate:.1%}), consider retraining")
    
    def generate_report(self, signals: List[BreakoutSignal]) -> str:
        """Generate formatted report of signals"""
        
        if not signals:
            return "No breakout signals found."
        
        report = []
        report.append("="*80)
        report.append("MICRO/SMALL CAP BREAKOUT SIGNALS")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("="*80)
        
        for i, signal in enumerate(signals[:10], 1):
            report.append(f"\n{i}. {signal.ticker} - {signal.signal_strength} SIGNAL")
            report.append(f"   Confidence: {signal.confidence:.1f}%")
            report.append(f"   Expected: {signal.expected_magnitude*100:.1f}% in {signal.expected_days:.0f} days")
            report.append(f"   Entry: ${signal.entry_price:.2f} | Stop: ${signal.stop_loss:.2f}")
            report.append(f"   Targets: ${signal.target_prices[0]:.2f} / ${signal.target_prices[1]:.2f} / ${signal.target_prices[2]:.2f}")
            report.append(f"   Risk Level: {signal.risk_level}")
            report.append(f"   Key Factors: {', '.join(signal.key_factors[:3])}")
        
        return "\n".join(report)

# ======================== MAIN EXECUTION ========================

async def main():
    """Example usage"""
    
    # Initialize system
    system = BreakoutDetectionSystem(model_path="models")
    
    # Example watchlist
    watchlist = [
        'PLTR', 'SOFI', 'HOOD', 'RIVN', 'LCID',
        'CHPT', 'STEM', 'OPEN', 'IONQ', 'DNA'
    ]
    
    # Scan for opportunities
    logger.info("Scanning watchlist for breakout opportunities...")
    signals = await system.scan_watchlist(watchlist)
    
    # Generate report
    report = system.generate_report(signals)
    print(report)
    
    # Save signals
    if signals:
        import json
        with open(f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump([s.to_dict() for s in signals], f, indent=2)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())
