# gcs_breakout_system.py
"""
Complete 40%+ Breakout Prediction System running entirely on Google Cloud Storage
Ultra-low cost, serverless architecture for retail trading
"""

import json
import pandas as pd
import numpy as np
import pickle
import gzip
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import hashlib
import io
import talib
from google.cloud import storage
from google.cloud import functions_v1
import requests

logger = logging.getLogger(__name__)

@dataclass
class BreakoutPrediction:
    """Complete breakout prediction"""
    ticker: str
    prediction_date: datetime
    confidence: float  # 0-1
    expected_magnitude: float  # Expected % gain
    expected_timeframe: int  # Days to target
    consolidation_days: int
    key_indicators: List[str]
    support_level: float
    resistance_level: float
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float

class GCSBreakoutSystem:
    """Complete breakout prediction system using only Google Cloud Storage"""
    
    def __init__(self, 
                 bucket_name: str,
                 twelvedata_api_key: str = None,
                 alphavantage_api_key: str = None):
        
        self.bucket_name = bucket_name
        self.twelvedata_api_key = twelvedata_api_key
        self.alphavantage_api_key = alphavantage_api_key
        
        # Initialize GCS client
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # GCS folder structure
        self.folders = {
            'raw_data': 'data/raw/',
            'processed_data': 'data/processed/',
            'models': 'models/',
            'predictions': 'predictions/',
            'web_data': 'web/',
            'configs': 'config/',
            'logs': 'logs/'
        }
        
        logger.info(f"GCS Breakout System initialized with bucket: {bucket_name}")
    
    def fetch_eod_data(self, ticker: str, days: int = 300) -> Optional[pd.DataFrame]:
        """Fetch EOD data using free APIs and cache in GCS"""
        
        # Check if we have recent cached data
        cached_data = self._get_cached_price_data(ticker)
        if cached_data is not None:
            last_date = pd.to_datetime(cached_data.index[-1]).date()
            if (datetime.now().date() - last_date).days <= 1:
                return cached_data  # Return cached if recent
        
        # Fetch fresh data
        price_data = None
        
        # Try TwelveData first (free tier: 800 requests/day)
        if self.twelvedata_api_key:
            price_data = self._fetch_from_twelvedata(ticker, days)
        
        # Fallback to Alpha Vantage (free tier: 500 requests/day)
        if price_data is None and self.alphavantage_api_key:
            price_data = self._fetch_from_alphavantage(ticker)
        
        # Fallback to yfinance (free but rate limited)
        if price_data is None:
            price_data = self._fetch_from_yfinance(ticker, days)
        
        if price_data is not None:
            # Cache the data in GCS
            self._cache_price_data(ticker, price_data)
        
        return price_data
    
    def _fetch_from_twelvedata(self, ticker: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch from TwelveData API"""
        try:
            url = f"https://api.twelvedata.com/time_series"
            params = {
                'symbol': ticker,
                'interval': '1day',
                'outputsize': days,
                'apikey': self.twelvedata_api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'values' in data:
                df = pd.DataFrame(data['values'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                df = df.astype(float)
                df.rename(columns={'1. open': 'open', '2. high': 'high', 
                                 '3. low': 'low', '4. close': 'close', 
                                 '5. volume': 'volume'}, inplace=True)
                return df.sort_index()
                
        except Exception as e:
            logger.error(f"TwelveData error for {ticker}: {e}")
        
        return None
    
    def _fetch_from_alphavantage(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch from Alpha Vantage API"""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker,
                'apikey': self.alphavantage_api_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                df = pd.DataFrame(data['Time Series (Daily)']).T
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                return df.sort_index()
                
        except Exception as e:
            logger.error(f"Alpha Vantage error for {ticker}: {e}")
        
        return None
    
    def _fetch_from_yfinance(self, ticker: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch from yfinance as fallback"""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            df = stock.history(period=f"{days}d")
            if not df.empty:
                df.columns = df.columns.str.lower()
                return df
        except Exception as e:
            logger.error(f"yfinance error for {ticker}: {e}")
        
        return None
    
    def _get_cached_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get cached price data from GCS"""
        try:
            blob_name = f"{self.folders['raw_data']}{ticker}.pkl.gz"
            blob = self.bucket.blob(blob_name)
            
            if blob.exists():
                # Check if data is recent (within 24 hours)
                if blob.time_created and (datetime.now() - blob.time_created.replace(tzinfo=None)).days > 1:
                    return None
                
                data = blob.download_as_bytes()
                decompressed = gzip.decompress(data)
                return pickle.loads(decompressed)
                
        except Exception as e:
            logger.debug(f"No cached data for {ticker}: {e}")
        
        return None
    
    def _cache_price_data(self, ticker: str, df: pd.DataFrame):
        """Cache price data to GCS"""
        try:
            blob_name = f"{self.folders['raw_data']}{ticker}.pkl.gz"
            blob = self.bucket.blob(blob_name)
            
            # Compress and upload
            pickled_data = pickle.dumps(df)
            compressed_data = gzip.compress(pickled_data)
            blob.upload_from_string(compressed_data, content_type='application/gzip')
            
        except Exception as e:
            logger.error(f"Failed to cache data for {ticker}: {e}")
    
    def analyze_consolidation(self, ticker: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consolidation pattern (pure price action, no volume)"""
        
        if len(price_data) < 100:
            return None
        
        close = price_data['close'].values
        high = price_data['high'].values
        low = price_data['low'].values
        
        # Keltner Channels (primary consolidation detector)
        ema_20 = talib.EMA(close, 20)
        atr_20 = talib.ATR(high, low, close, 20)
        kc_upper = ema_20 + (2.0 * atr_20)
        kc_middle = ema_20
        kc_lower = ema_20 - (2.0 * atr_20)
        
        # Current position
        current_position = (close[-1] - kc_lower[-1]) / (kc_upper[-1] - kc_lower[-1])
        
        # Range analysis
        rolling_high = pd.Series(high).rolling(20).max()
        rolling_low = pd.Series(low).rolling(20).min()
        range_pct = (rolling_high - rolling_low) / rolling_low
        
        # Consolidation detection (adjusted for $300M-$2B market cap stocks)
        current_range = range_pct.iloc[-1]
        in_consolidation = current_range < 0.10  # 10% range threshold for larger caps
        
        # Count consolidation days
        consolidation_days = 0
        for i in range(len(range_pct)-1, -1, -1):
            if range_pct.iloc[i] < 0.10:  # Consistent with tighter range for larger caps
                consolidation_days += 1
            else:
                break
        
        # Support and resistance
        support_level = pd.Series(low).rolling(50).min().iloc[-1]
        resistance_level = pd.Series(high).rolling(50).max().iloc[-1]
        
        # Squeeze detection
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 20, 2, 2)
        kc_width = (kc_upper - kc_lower) / kc_middle
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        keltner_squeeze = kc_width[-1] < np.mean(kc_width[-50:]) * 0.7
        bollinger_squeeze = bb_upper[-1] <= kc_upper[-1] and bb_lower[-1] >= kc_lower[-1]
        
        # Pattern quality
        quality_factors = []
        quality_factors.append(min(10 / (current_range + 0.01), 1.0))  # Range tightness
        quality_factors.append(min(consolidation_days / 30, 1.0))  # Duration
        quality_factors.append(float(keltner_squeeze))  # Squeeze
        
        pattern_quality = np.mean(quality_factors)
        
        # Breakout readiness
        readiness_factors = []
        readiness_factors.append(pattern_quality)
        readiness_factors.append(1.0 if current_position > 0.7 else 0.5)  # Near resistance
        readiness_factors.append(float(consolidation_days > 15))  # Sufficient duration
        
        breakout_readiness = np.mean(readiness_factors)
        
        return {
            'ticker': ticker,
            'date': datetime.now().isoformat(),
            'in_consolidation': in_consolidation,
            'consolidation_days': consolidation_days,
            'current_range_pct': current_range,
            'kc_position': current_position,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'keltner_squeeze': keltner_squeeze,
            'bollinger_squeeze': bollinger_squeeze,
            'pattern_quality': pattern_quality,
            'breakout_readiness': breakout_readiness,
            'current_price': close[-1],
            'rsi_14': talib.RSI(close, 14)[-1],
            'atr_14': talib.ATR(high, low, close, 14)[-1]
        }
    
    def generate_breakout_prediction(self, analysis: Dict[str, Any]) -> Optional[BreakoutPrediction]:
        """Generate 40%+ breakout prediction based on consolidation analysis"""
        
        if not analysis or not analysis['in_consolidation']:
            return None
        
        # Calculate confidence
        confidence_factors = []
        
        # Pattern quality (40% weight)
        confidence_factors.extend([analysis['pattern_quality']] * 4)
        
        # Duration factor (20% weight)
        days = analysis['consolidation_days']
        if 15 <= days <= 45:
            duration_score = 1.0
        elif days < 15:
            duration_score = days / 15
        else:
            duration_score = max(0.2, 1 - (days - 45) / 60)
        confidence_factors.extend([duration_score] * 2)
        
        # Position factor (20% weight)
        position_score = analysis['kc_position'] if analysis['kc_position'] > 0.5 else 0.3
        confidence_factors.extend([position_score] * 2)
        
        # Squeeze factor (20% weight)
        squeeze_score = 1.0 if analysis['keltner_squeeze'] else 0.5
        confidence_factors.extend([squeeze_score] * 2)
        
        confidence = np.mean(confidence_factors)
        
        # Only generate predictions with confidence > 50%
        if confidence < 0.5:
            return None
        
        # Calculate targets
        current_price = analysis['current_price']
        resistance = analysis['resistance_level']
        support = analysis['support_level']
        
        # Expected magnitude (40%+ breakout system)
        base_magnitude = 0.40  # 40% minimum
        quality_bonus = analysis['pattern_quality'] * 0.30  # Up to 30% bonus
        expected_magnitude = base_magnitude + quality_bonus
        
        # Entry, stop loss, and target
        entry_price = current_price
        stop_loss = support * 0.95  # 5% below support
        target_price = entry_price * (1 + expected_magnitude)
        
        # Risk-reward ratio
        potential_loss = entry_price - stop_loss
        potential_gain = target_price - entry_price
        risk_reward_ratio = potential_gain / potential_loss if potential_loss > 0 else 0
        
        # Expected timeframe (higher quality = faster breakout)
        base_days = 45
        quality_factor = 1 - (analysis['pattern_quality'] * 0.4)
        expected_timeframe = int(base_days * quality_factor)
        
        # Key indicators
        key_indicators = []
        if analysis['consolidation_days'] > 20:
            key_indicators.append(f"{analysis['consolidation_days']}-day consolidation")
        if analysis['pattern_quality'] > 0.7:
            key_indicators.append("High-quality pattern")
        if analysis['keltner_squeeze']:
            key_indicators.append("Keltner squeeze")
        if analysis['kc_position'] > 0.7:
            key_indicators.append("Near resistance breakout")
        if risk_reward_ratio > 3:
            key_indicators.append("Excellent risk/reward")
        
        return BreakoutPrediction(
            ticker=analysis['ticker'],
            prediction_date=datetime.now(),
            confidence=confidence,
            expected_magnitude=expected_magnitude,
            expected_timeframe=expected_timeframe,
            consolidation_days=analysis['consolidation_days'],
            key_indicators=key_indicators,
            support_level=support,
            resistance_level=resistance,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def process_ticker(self, ticker: str) -> Optional[BreakoutPrediction]:
        """Complete pipeline: fetch data -> analyze -> predict"""
        
        # Fetch EOD data
        price_data = self.fetch_eod_data(ticker)
        if price_data is None:
            logger.error(f"Could not fetch data for {ticker}")
            return None
        
        # Analyze consolidation
        analysis = self.analyze_consolidation(ticker, price_data)
        if analysis is None:
            return None
        
        # Generate prediction
        prediction = self.generate_breakout_prediction(analysis)
        
        if prediction:
            # Store prediction in GCS
            self._store_prediction(prediction)
            # Store for web display
            self._store_for_web(prediction, analysis)
        
        return prediction
    
    def batch_process_watchlist(self, tickers: List[str]) -> Dict[str, Any]:
        """Process entire watchlist and generate web data"""
        
        results = {
            'processed': 0,
            'predictions': 0,
            'errors': 0,
            'high_confidence': 0,
            'predictions_list': []
        }
        
        for ticker in tickers:
            try:
                prediction = self.process_ticker(ticker)
                results['processed'] += 1
                
                if prediction:
                    results['predictions'] += 1
                    results['predictions_list'].append(asdict(prediction))
                    
                    if prediction.confidence > 0.7:
                        results['high_confidence'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                results['errors'] += 1
        
        # Sort by confidence
        results['predictions_list'].sort(key=lambda x: x['confidence'], reverse=True)
        
        # Create web dashboard data
        self._create_web_dashboard(results)
        
        return results
    
    def _store_prediction(self, prediction: BreakoutPrediction):
        """Store prediction in GCS"""
        try:
            blob_name = f"{self.folders['predictions']}{prediction.ticker}_{datetime.now().strftime('%Y%m%d')}.json"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(asdict(prediction), indent=2, default=str),
                content_type='application/json'
            )
        except Exception as e:
            logger.error(f"Failed to store prediction for {prediction.ticker}: {e}")
    
    def _store_for_web(self, prediction: BreakoutPrediction, analysis: Dict[str, Any]):
        """Store data optimized for web display"""
        try:
            web_data = {
                'ticker': prediction.ticker,
                'confidence': round(prediction.confidence, 3),
                'expected_magnitude': round(prediction.expected_magnitude, 3),
                'expected_timeframe': prediction.expected_timeframe,
                'consolidation_days': prediction.consolidation_days,
                'key_indicators': prediction.key_indicators,
                'entry_price': round(prediction.entry_price, 2),
                'target_price': round(prediction.target_price, 2),
                'stop_loss': round(prediction.stop_loss, 2),
                'risk_reward': round(prediction.risk_reward_ratio, 1),
                'pattern_quality': round(analysis['pattern_quality'], 3),
                'last_updated': datetime.now().isoformat()
            }
            
            # Individual ticker file
            blob_name = f"{self.folders['web_data']}tickers/{prediction.ticker}.json"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(web_data, indent=2),
                content_type='application/json'
            )
            
            # Make it publicly readable for website
            blob.make_public()
            
        except Exception as e:
            logger.error(f"Failed to store web data for {prediction.ticker}: {e}")
    
    def _create_web_dashboard(self, results: Dict[str, Any]):
        """Create dashboard data for website"""
        try:
            dashboard_data = {
                'summary': {
                    'total_processed': results['processed'],
                    'total_predictions': results['predictions'],
                    'high_confidence_count': results['high_confidence'],
                    'error_count': results['errors'],
                    'last_updated': datetime.now().isoformat()
                },
                'top_signals': results['predictions_list'][:20],  # Top 20 for website
                'statistics': {
                    'avg_confidence': round(np.mean([p['confidence'] for p in results['predictions_list']]), 3) if results['predictions_list'] else 0,
                    'avg_expected_return': round(np.mean([p['expected_magnitude'] for p in results['predictions_list']]), 3) if results['predictions_list'] else 0,
                    'avg_consolidation_days': int(np.mean([p['consolidation_days'] for p in results['predictions_list']])) if results['predictions_list'] else 0
                }
            }
            
            # Store dashboard data
            blob_name = f"{self.folders['web_data']}dashboard.json"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(dashboard_data, indent=2, default=str),
                content_type='application/json'
            )
            blob.make_public()
            
            # Create a simple HTML page
            self._create_simple_webpage(dashboard_data)
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
    
    def _create_simple_webpage(self, dashboard_data: Dict[str, Any]):
        """Create a simple HTML page for viewing results"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>40%+ Breakout Predictions</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .signal {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .high-confidence {{ border-left: 5px solid green; }}
                .medium-confidence {{ border-left: 5px solid orange; }}
                .confidence {{ font-weight: bold; color: green; }}
                .ticker {{ font-size: 18px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>🚀 40%+ Breakout Predictions</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Signals:</strong> {dashboard_data['summary']['total_predictions']}</p>
                <p><strong>High Confidence (>70%):</strong> {dashboard_data['summary']['high_confidence_count']}</p>
                <p><strong>Average Confidence:</strong> {dashboard_data['statistics']['avg_confidence']:.1%}</p>
                <p><strong>Average Expected Return:</strong> {dashboard_data['statistics']['avg_expected_return']:.1%}</p>
                <p><strong>Last Updated:</strong> {dashboard_data['summary']['last_updated'][:19]}</p>
            </div>
            
            <h2>Top Signals</h2>
        """
        
        for signal in dashboard_data['top_signals'][:10]:
            confidence_class = "high-confidence" if signal['confidence'] > 0.7 else "medium-confidence"
            html_content += f"""
            <div class="signal {confidence_class}">
                <div class="ticker">{signal['ticker']}</div>
                <p><span class="confidence">Confidence: {signal['confidence']:.1%}</span></p>
                <p><strong>Expected Return:</strong> {signal['expected_magnitude']:.1%}</p>
                <p><strong>Consolidation:</strong> {signal['consolidation_days']} days</p>
                <p><strong>Entry:</strong> ${signal['entry_price']:.2f} | <strong>Target:</strong> ${signal['target_price']:.2f} | <strong>Stop:</strong> ${signal['stop_loss']:.2f}</p>
                <p><strong>Risk/Reward:</strong> {signal['risk_reward']:.1f}:1</p>
                <p><strong>Key Factors:</strong> {', '.join(signal['key_indicators'])}</p>
            </div>
            """
        
        html_content += """
            </body>
        </html>
        """
        
        # Upload HTML page
        try:
            blob_name = f"{self.folders['web_data']}index.html"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(html_content, content_type='text/html')
            blob.make_public()
            
            logger.info(f"Web page created: https://storage.googleapis.com/{self.bucket_name}/{blob_name}")
            
        except Exception as e:
            logger.error(f"Failed to create webpage: {e}")
    
    def get_public_url(self, file_path: str) -> str:
        """Get public URL for a file in GCS"""
        return f"https://storage.googleapis.com/{self.bucket_name}/{file_path}"

def create_gcs_breakout_system(bucket_name: str, 
                              twelvedata_api_key: str = None,
                              alphavantage_api_key: str = None) -> GCSBreakoutSystem:
    """Create complete GCS-based breakout system"""
    
    system = GCSBreakoutSystem(bucket_name, twelvedata_api_key, alphavantage_api_key)
    logger.info("GCS Breakout System ready for cloud deployment!")
    
    return system

# Example usage for Cloud Function
def cloud_function_entry_point(request):
    """Entry point for Google Cloud Function"""
    
    import os
    
    # Get environment variables
    bucket_name = os.environ.get('GCS_BUCKET_NAME')
    twelvedata_key = os.environ.get('TWELVEDATA_API_KEY')
    alphavantage_key = os.environ.get('ALPHAVANTAGE_API_KEY')
    
    # Get watchlist from request or use default
    if request.method == 'POST':
        request_json = request.get_json()
        watchlist = request_json.get('tickers', ['PLTR', 'SOFI', 'HOOD'])  # Default $300M-$2B watchlist
    else:
        watchlist = ['PLTR', 'SOFI', 'HOOD']  # Default for GET requests
    
    # Create system and process
    system = create_gcs_breakout_system(bucket_name, twelvedata_key, alphavantage_key)
    results = system.batch_process_watchlist(watchlist)
    
    return {
        'status': 'success',
        'results': results,
        'web_url': system.get_public_url('web/index.html')
    }