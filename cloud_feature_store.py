# cloud_feature_store.py
"""
Cloud-Optimized Feature Store for 40%+ Breakout Prediction AI
Designed for Google Cloud Storage + Website deployment on minimal budget
"""

import json
import pandas as pd
import numpy as np
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import talib
from google.cloud import storage
import sqlite3
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class BreakoutSignal:
    """Core breakout prediction signal"""
    ticker: str
    date: datetime
    confidence: float  # 0-1, probability of 40%+ breakout
    consolidation_days: int
    range_tightness: float
    keltner_squeeze: bool
    breakout_readiness: float
    expected_magnitude: float  # Expected % gain if breakout occurs
    expected_days: int  # Expected days to reach target
    support_level: float
    resistance_level: float
    key_factors: List[str]  # Main factors driving the signal

class CloudFeatureStore:
    """Lightweight feature store optimized for cloud deployment and web serving"""
    
    def __init__(self, 
                 gcs_bucket_name: str = None,
                 local_cache_dir: str = "./cache"):
        
        self.gcs_bucket_name = gcs_bucket_name
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(exist_ok=True)
        
        # Initialize Google Cloud Storage client (if bucket provided)
        self.gcs_client = None
        self.bucket = None
        if gcs_bucket_name:
            try:
                self.gcs_client = storage.Client()
                self.bucket = self.gcs_client.bucket(gcs_bucket_name)
                logger.info(f"Connected to GCS bucket: {gcs_bucket_name}")
            except Exception as e:
                logger.warning(f"Could not connect to GCS: {e}")
        
        # Local SQLite for fast queries
        self.local_db = self.local_cache_dir / "features.db"
        self.conn = sqlite3.connect(self.local_db, check_same_thread=False)
        self._init_local_db()
        
        # Feature definitions (focused on consolidation only)
        self.core_features = [
            'price_close', 'price_high', 'price_low',
            'kc_upper', 'kc_middle', 'kc_lower', 'kc_position',
            'bb_upper', 'bb_lower', 'bb_width',
            'consolidation_days', 'range_tightness', 'breakout_readiness',
            'pattern_quality', 'rsi_14', 'atr_14'
        ]
        
        logger.info("Cloud Feature Store initialized for web deployment")
    
    def _init_local_db(self):
        """Initialize local SQLite cache"""
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS ticker_features (
            ticker TEXT PRIMARY KEY,
            last_updated DATE,
            features_json TEXT,
            signal_json TEXT
        );
        """)
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        self.conn.commit()
    
    def compute_consolidation_features(self, ticker: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Compute ONLY consolidation features (no volume) for alpha generation"""
        
        if len(price_data) < 100:
            return None
        
        # Core price data
        close = price_data['close'].values
        high = price_data['high'].values
        low = price_data['low'].values
        
        features = {}
        
        # Price levels
        features['price_close'] = close[-1]
        features['price_high'] = high[-1]
        features['price_low'] = low[-1]
        
        # Keltner Channels (PRIMARY consolidation indicator)
        ema_20 = talib.EMA(close, 20)
        atr_20 = talib.ATR(high, low, close, 20)
        kc_upper = ema_20 + (2.0 * atr_20)
        kc_middle = ema_20
        kc_lower = ema_20 - (2.0 * atr_20)
        
        features['kc_upper'] = kc_upper[-1]
        features['kc_middle'] = kc_middle[-1]
        features['kc_lower'] = kc_lower[-1]
        features['kc_position'] = (close[-1] - kc_lower[-1]) / (kc_upper[-1] - kc_lower[-1])
        
        # Bollinger Bands (for squeeze detection)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 20, 2, 2)
        features['bb_upper'] = bb_upper[-1]
        features['bb_lower'] = bb_lower[-1]
        features['bb_width'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
        
        # Consolidation analysis (pure price action)
        rolling_high = pd.Series(high).rolling(20).max()
        rolling_low = pd.Series(low).rolling(20).min()
        range_pct = (rolling_high - rolling_low) / rolling_low
        
        # Count consolidation days
        in_consolidation = range_pct < 0.10  # 10% range for $300M-$2B caps
        consolidation_days = 0
        for i in range(len(in_consolidation)-1, -1, -1):
            if in_consolidation.iloc[i]:
                consolidation_days += 1
            else:
                break
        
        features['consolidation_days'] = consolidation_days
        features['range_tightness'] = 1 / (range_pct.iloc[-1] + 0.01)
        
        # Support/Resistance
        features['support_level'] = pd.Series(low).rolling(50).min().iloc[-1]
        features['resistance_level'] = pd.Series(high).rolling(50).max().iloc[-1]
        
        # Pattern quality
        kc_width = (kc_upper - kc_lower) / kc_middle
        keltner_squeeze = kc_width[-1] < np.mean(kc_width[-50:]) * 0.7
        
        quality_score = (
            min(features['range_tightness'] / 10, 1.0) * 0.4 +  # Range tightness
            min(consolidation_days / 30, 1.0) * 0.3 +  # Duration
            float(keltner_squeeze) * 0.3  # Squeeze
        )
        features['pattern_quality'] = quality_score
        
        # Breakout readiness
        position_factor = 1.0 if features['kc_position'] > 0.7 else 0.5
        readiness = quality_score * position_factor * min(consolidation_days / 20, 1.0)
        features['breakout_readiness'] = min(readiness, 1.0)
        
        # Additional indicators
        features['rsi_14'] = talib.RSI(close, 14)[-1]
        features['atr_14'] = talib.ATR(high, low, close, 14)[-1]
        
        return features
    
    def generate_breakout_signal(self, ticker: str, features: Dict[str, Any]) -> BreakoutSignal:
        """Generate final breakout signal for the AI system"""
        
        # Calculate confidence based on consolidation strength
        confidence_factors = []
        
        # Factor 1: Pattern quality
        confidence_factors.append(features['pattern_quality'])
        
        # Factor 2: Consolidation duration (optimal 15-45 days)
        days = features['consolidation_days']
        if 15 <= days <= 45:
            duration_score = 1.0
        elif days < 15:
            duration_score = days / 15
        else:
            duration_score = max(0, 1 - (days - 45) / 45)
        confidence_factors.append(duration_score)
        
        # Factor 3: Range tightness
        tightness_score = min(features['range_tightness'] / 8, 1.0)
        confidence_factors.append(tightness_score)
        
        # Factor 4: Position in range (closer to resistance = higher confidence)
        position_score = features['kc_position'] if features['kc_position'] > 0.5 else 0.3
        confidence_factors.append(position_score)
        
        # Factor 5: RSI not overbought
        rsi_score = 1.0 if features['rsi_14'] < 70 else 0.5
        confidence_factors.append(rsi_score)
        
        confidence = np.mean(confidence_factors)
        
        # Expected magnitude (based on historical micro-cap breakouts)
        base_magnitude = 0.45  # 45% base expectation
        magnitude_multiplier = 1.0 + (features['pattern_quality'] * 0.5)  # Up to 67% for perfect patterns
        expected_magnitude = base_magnitude * magnitude_multiplier
        
        # Expected timeframe (better patterns = faster breakouts)
        base_days = 30
        quality_factor = 1 - (features['pattern_quality'] * 0.4)  # Higher quality = faster
        expected_days = int(base_days * quality_factor)
        
        # Key factors driving the signal
        key_factors = []
        if features['consolidation_days'] > 20:
            key_factors.append(f"{features['consolidation_days']}-day consolidation")
        if features['pattern_quality'] > 0.7:
            key_factors.append("High-quality pattern")
        if features['kc_position'] > 0.7:
            key_factors.append("Near resistance")
        if features['range_tightness'] > 5:
            key_factors.append("Tight range")
        
        return BreakoutSignal(
            ticker=ticker,
            date=datetime.now(),
            confidence=confidence,
            consolidation_days=features['consolidation_days'],
            range_tightness=features['range_tightness'],
            keltner_squeeze=features.get('keltner_squeeze', False),
            breakout_readiness=features['breakout_readiness'],
            expected_magnitude=expected_magnitude,
            expected_days=expected_days,
            support_level=features['support_level'],
            resistance_level=features['resistance_level'],
            key_factors=key_factors
        )
    
    def process_ticker_for_web(self, ticker: str, price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Process ticker and prepare data for web serving"""
        
        # Compute features
        features = self.compute_consolidation_features(ticker, price_data)
        if not features:
            return None
        
        # Generate signal
        signal = self.generate_breakout_signal(ticker, features)
        
        # Only return signals with reasonable confidence
        if signal.confidence < 0.4:  # 40% minimum confidence
            return None
        
        # Prepare web-ready data
        web_data = {
            'ticker': ticker,
            'signal': asdict(signal),
            'features': {k: features[k] for k in self.core_features if k in features},
            'last_updated': datetime.now().isoformat(),
            'data_hash': self._calculate_data_hash(features)
        }
        
        # Store locally
        self._store_local(ticker, web_data)
        
        # Upload to cloud if available
        if self.bucket:
            self._upload_to_gcs(ticker, web_data)
        
        return web_data
    
    def _store_local(self, ticker: str, data: Dict[str, Any]):
        """Store in local SQLite cache"""
        
        self.conn.execute("""
        INSERT OR REPLACE INTO ticker_features 
        (ticker, last_updated, features_json, signal_json)
        VALUES (?, ?, ?, ?)
        """, (
            ticker,
            datetime.now().date(),
            json.dumps(data['features']),
            json.dumps(data['signal'])
        ))
        self.conn.commit()
    
    def _upload_to_gcs(self, ticker: str, data: Dict[str, Any]):
        """Upload to Google Cloud Storage for web serving"""
        
        try:
            # Individual ticker file
            blob_name = f"signals/{ticker}.json"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(data, indent=2, default=str),
                content_type='application/json'
            )
            
            # Compressed version for bandwidth
            compressed_data = gzip.compress(json.dumps(data, default=str).encode())
            blob_compressed = self.bucket.blob(f"signals/compressed/{ticker}.json.gz")
            blob_compressed.upload_from_string(compressed_data, content_type='application/gzip')
            
            logger.info(f"Uploaded {ticker} data to GCS")
            
        except Exception as e:
            logger.error(f"Failed to upload {ticker} to GCS: {e}")
    
    def get_signal_for_web(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get signal data optimized for web display"""
        
        # Try local cache first
        cursor = self.conn.execute("""
        SELECT features_json, signal_json, last_updated 
        FROM ticker_features 
        WHERE ticker = ?
        """, (ticker,))
        
        row = cursor.fetchone()
        if row:
            features_json, signal_json, last_updated = row
            
            return {
                'ticker': ticker,
                'features': json.loads(features_json),
                'signal': json.loads(signal_json),
                'last_updated': last_updated
            }
        
        return None
    
    def get_all_signals_for_web(self, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Get all signals above confidence threshold for web dashboard"""
        
        cursor = self.conn.execute("""
        SELECT ticker, features_json, signal_json, last_updated 
        FROM ticker_features 
        ORDER BY ticker
        """)
        
        signals = []
        for row in cursor.fetchall():
            ticker, features_json, signal_json, last_updated = row
            signal_data = json.loads(signal_json)
            
            if signal_data.get('confidence', 0) >= min_confidence:
                signals.append({
                    'ticker': ticker,
                    'confidence': signal_data['confidence'],
                    'expected_magnitude': signal_data['expected_magnitude'],
                    'consolidation_days': signal_data['consolidation_days'],
                    'key_factors': signal_data.get('key_factors', []),
                    'last_updated': last_updated
                })
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def create_web_summary(self) -> Dict[str, Any]:
        """Create summary data for web dashboard"""
        
        all_signals = self.get_all_signals_for_web(min_confidence=0.4)
        
        if not all_signals:
            return {
                'total_signals': 0,
                'high_confidence_signals': 0,
                'avg_confidence': 0,
                'top_signals': [],
                'last_updated': datetime.now().isoformat()
            }
        
        high_confidence = [s for s in all_signals if s['confidence'] > 0.7]
        avg_confidence = np.mean([s['confidence'] for s in all_signals])
        
        summary = {
            'total_signals': len(all_signals),
            'high_confidence_signals': len(high_confidence),
            'avg_confidence': round(avg_confidence, 3),
            'top_signals': all_signals[:10],  # Top 10 for dashboard
            'last_updated': datetime.now().isoformat(),
            'consolidation_stats': {
                'avg_days': int(np.mean([s['consolidation_days'] for s in all_signals])),
                'max_days': max([s['consolidation_days'] for s in all_signals]),
                'ready_for_breakout': len([s for s in all_signals if s['consolidation_days'] > 15])
            }
        }
        
        # Upload summary to GCS for web access
        if self.bucket:
            try:
                blob = self.bucket.blob("dashboard/summary.json")
                blob.upload_from_string(
                    json.dumps(summary, indent=2),
                    content_type='application/json'
                )
            except Exception as e:
                logger.error(f"Failed to upload summary: {e}")
        
        return summary
    
    def _calculate_data_hash(self, features: Dict[str, Any]) -> str:
        """Calculate hash for data integrity"""
        feature_string = json.dumps(features, sort_keys=True)
        return hashlib.md5(feature_string.encode()).hexdigest()[:8]
    
    def batch_process_for_web(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Process multiple tickers for web deployment"""
        
        results = {
            'processed': 0,
            'signals_generated': 0,
            'errors': 0,
            'top_signals': []
        }
        
        all_signals = []
        
        for ticker, price_data in ticker_data.items():
            try:
                web_data = self.process_ticker_for_web(ticker, price_data)
                if web_data:
                    results['signals_generated'] += 1
                    all_signals.append(web_data['signal'])
                
                results['processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                results['errors'] += 1
        
        # Sort signals by confidence
        all_signals.sort(key=lambda x: x['confidence'], reverse=True)
        results['top_signals'] = all_signals[:20]  # Top 20
        
        # Create and upload summary
        summary = self.create_web_summary()
        results['summary'] = summary
        
        return results
    
    def get_feature_consistency_hash(self) -> str:
        """Get hash for training/inference consistency"""
        feature_list = sorted(self.core_features)
        return hashlib.md5(','.join(feature_list).encode()).hexdigest()
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old cached data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        self.conn.execute(
            "DELETE FROM ticker_features WHERE last_updated < ?", 
            (cutoff_date.date(),)
        )
        self.conn.commit()
        
        # Clean up GCS old files if needed
        if self.bucket:
            try:
                blobs = self.bucket.list_blobs(prefix="signals/")
                for blob in blobs:
                    if blob.time_created < cutoff_date.replace(tzinfo=blob.time_created.tzinfo):
                        blob.delete()
            except Exception as e:
                logger.error(f"Error cleaning GCS: {e}")

def create_cloud_feature_store(gcs_bucket_name: str = None) -> CloudFeatureStore:
    """Create cloud-optimized feature store for web deployment"""
    
    store = CloudFeatureStore(gcs_bucket_name)
    logger.info("Cloud Feature Store ready for web deployment!")
    
    return store