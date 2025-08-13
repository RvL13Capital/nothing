# training_pipeline.py
"""
Vollständige Training Pipeline für beide Modelle
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self):
        self.training_tickers = []
        self.labeled_data = []
        
    def collect_historical_data(self, tickers: List[str], years: int = 3):
        """
        Sammelt historische Daten und labelt Breakouts
        """
        all_data = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period=f"{years}y")
                
                # Label breakouts (40%+ moves)
                labeled = self.label_breakouts(ticker, df)
                all_data.extend(labeled)
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
        
        return all_data
    
    def label_breakouts(self, ticker: str, df: pd.DataFrame) -> List[Dict]:
        """
        Automatisches Labeling von Breakouts
        Sucht nach 40%+ Moves innerhalb von 60 Tagen nach Consolidation
        """
        labeled_examples = []
        
        # Detect consolidations
        detector = ConsolidationDetector()
        consolidations = detector.detect_consolidations(ticker, df)
        
        for period in consolidations['consolidation_periods']:
            end_idx = period['end_idx']
            
            # Look forward 60 days
            future_window = df.iloc[end_idx:end_idx+60]
            
            if len(future_window) > 0:
                start_price = df.iloc[end_idx]['close']
                max_price = future_window['high'].max()
                
                # Check for 40%+ move
                max_return = (max_price / start_price) - 1
                
                if max_return >= 0.40:
                    # Successful breakout found
                    days_to_peak = future_window['high'].idxmax()
                    days_taken = (days_to_peak - df.index[end_idx]).days
                    
                    labeled_examples.append({
                        'ticker': ticker,
                        'consolidation_start': period['start'],
                        'consolidation_end': period['end'],
                        'breakout': True,
                        'magnitude': max_return,
                        'days_to_breakout': days_taken,
                        'df': df.iloc[period['start_idx']:end_idx+1]
                    })
                else:
                    # No breakout
                    labeled_examples.append({
                        'ticker': ticker,
                        'consolidation_start': period['start'],
                        'consolidation_end': period['end'],
                        'breakout': False,
                        'magnitude': 0,
                        'days_to_breakout': 0,
                        'df': df.iloc[period['start_idx']:end_idx+1]
                    })
        
        return labeled_examples
    
    def prepare_training_data(self, labeled_data: List[Dict]):
        """
        Bereitet Features und Labels für Training vor
        """
        X_list = []
        y_list = []
        
        feature_engineer = MicroCapFeatureEngineer()
        
        for example in labeled_data:
            # Create features
            consolidation_data = {
                'in_consolidation': True,
                'current_consolidation': {...}  # Fill with actual data
            }
            
            features = feature_engineer.create_features(
                example['df'], 
                consolidation_data
            )
            
            X_list.append(features)
            
            # Create labels [breakout, magnitude, days, volume_surge]
            y_list.append([
                1 if example['breakout'] else 0,
                example['magnitude'],
                example['days_to_breakout'],
                1.5  # Placeholder for volume surge
            ])
        
        return pd.concat(X_list), np.array(y_list)
    
    def train_models(self, X: pd.DataFrame, y: np.ndarray):
        """
        Trainiert beide Modelle mit Optuna Optimization
        """
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train XGBoost
        xgb_model = XGBoostBreakoutModel()
        xgb_model.optimize_hyperparameters(X_train, y_train[:, 0], n_trials=100)
        xgb_model.train(X_train, y_train[:, 0], X_val, y_val[:, 0])
        xgb_model.save_model("models/xgboost")
        
        # Train LSTM
        lstm_trainer = LSTMTrainer(X.columns.tolist())
        lstm_trainer.optimize_hyperparameters(
            X_train.values, y_train,
            X_val.values, y_val,
            n_trials=50
        )
        lstm_trainer.train_final_model(
            X_train.values, y_train,
            X_val.values, y_val,
            epochs=100
        )
        lstm_trainer.save_model("models/lstm")
        
        return xgb_model, lstm_trainer

# Verwendung:
pipeline = TrainingPipeline()
training_tickers = ['PLTR', 'SOFI', 'HOOD', ...]  # Ihre Micro/Small Caps
labeled_data = pipeline.collect_historical_data(training_tickers, years=3)
X, y = pipeline.prepare_training_data(labeled_data)
xgb_model, lstm_model = pipeline.train_models(X, y)
