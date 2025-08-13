# xgboost_breakout_model.py
"""
XGBoost Model with Self-Optimization for Breakout Detection
Specialized for Micro/Small Cap 40%+ Moves
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.preprocessing import RobustScaler
import optuna
from optuna.integration import XGBoostPruningCallback
import shap
import joblib
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json

logger = logging.getLogger(__name__)

class MicroCapFeatureEngineer:
    """
    Feature engineering specialized for micro/small cap breakouts
    Focus on volume patterns and consolidation characteristics
    """
    
    def __init__(self):
        self.feature_importance = {}
        self.selected_features = []
        
    def create_features(self, df: pd.DataFrame, consolidation_data: Dict) -> pd.DataFrame:
        """
        Create comprehensive features for XGBoost
        """
        
        features = pd.DataFrame(index=[len(df)-1])  # Single row for current state
        
        # ========== CONSOLIDATION FEATURES ==========
        if consolidation_data.get('in_consolidation'):
            current = consolidation_data.get('current_consolidation')
            
            # Core consolidation metrics
            features['consolidation_days'] = current.duration_days
            features['range_percent'] = current.range_percent
            features['position_in_range'] = current.position_in_range
            features['consolidation_quality'] = current.quality_score
            features['breakout_readiness'] = current.breakout_readiness
            
            # Boundary dynamics
            features['support_tests'] = current.boundary_tests['support_tests']
            features['resistance_tests'] = current.boundary_tests['resistance_tests']
            features['boundary_test_ratio'] = features['support_tests'] / (features['resistance_tests'] + 1)
            
            # Squeeze indicators
            features['keltner_squeeze'] = int(current.keltner_squeeze)
            features['ttm_squeeze'] = int(current.ttm_squeeze)
            
            # Volume characteristics during consolidation
            vol_chars = current.volume_characteristics
            features['volume_trend'] = vol_chars.get('volume_trend', 0)
            features['volume_volatility'] = vol_chars.get('volume_volatility', 0)
            features['obv_trend'] = vol_chars.get('obv_trend', 0)
            features['cmf_avg'] = vol_chars.get('cmf_avg', 0)
            features['mfi_avg'] = vol_chars.get('mfi_avg', 50)
            
            # Volume profile
            if 'volume_profile' in consolidation_data:
                profile = consolidation_data['volume_profile']
                features['poc_distance'] = abs(df['close'].iloc[-1] - profile.get('poc_price', df['close'].iloc[-1]))
                features['volume_skew'] = profile.get('volume_skew', 0)
                features['accumulation_ratio'] = profile.get('accumulation_days', 0) / (profile.get('distribution_days', 1) + 1)
        else:
            # Not in consolidation - set defaults
            for key in ['consolidation_days', 'range_percent', 'position_in_range',
                       'consolidation_quality', 'breakout_readiness', 'support_tests',
                       'resistance_tests', 'boundary_test_ratio', 'keltner_squeeze',
                       'ttm_squeeze', 'volume_trend', 'volume_volatility', 'obv_trend',
                       'cmf_avg', 'mfi_avg', 'poc_distance', 'volume_skew', 'accumulation_ratio']:
                features[key] = 0
        
        # ========== KELTNER CHANNEL FEATURES ==========
        if 'kc_width' in df.columns:
            features['kc_width_current'] = df['kc_width'].iloc[-1]
            features['kc_width_ma20'] = df['kc_width'].rolling(20).mean().iloc[-1]
            features['kc_width_ratio'] = features['kc_width_current'] / features['kc_width_ma20']
            features['kc_position'] = df['kc_position'].iloc[-1]
            
            # Keltner channel momentum
            features['kc_position_change_5d'] = df['kc_position'].iloc[-1] - df['kc_position'].iloc[-6]
            features['kc_width_change_10d'] = df['kc_width'].iloc[-1] - df['kc_width'].iloc[-11]
        
        # ========== VOLUME FEATURES (Critical for micro caps) ==========
        
        # Volume patterns
        features['volume_ratio_5d'] = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-20:].mean()
        features['volume_ratio_1d'] = df['volume'].iloc[-1] / df['volume'].iloc[-20:].mean()
        
        # Volume momentum
        features['volume_acceleration'] = (df['volume'].iloc[-5:].mean() - df['volume'].iloc[-10:-5].mean()) / df['volume'].iloc[-20:].mean()
        
        # OBV features
        if 'obv' in df.columns:
            features['obv_ma_ratio'] = df['obv'].iloc[-1] / df['obv'].rolling(20).mean().iloc[-1]
            features['obv_acceleration'] = (df['obv'].iloc[-5:].mean() - df['obv'].iloc[-10:-5].mean()) / abs(df['obv'].iloc[-20:].mean() + 1)
        
        # Money flow
        if 'mfi' in df.columns:
            features['mfi_current'] = df['mfi'].iloc[-1]
            features['mfi_ma20'] = df['mfi'].rolling(20).mean().iloc[-1]
            features['mfi_trending_up'] = int(df['mfi'].iloc[-1] > df['mfi'].iloc[-6])
        
        if 'cmf' in df.columns:
            features['cmf_current'] = df['cmf'].iloc[-1]
            features['cmf_positive_days'] = (df['cmf'].iloc[-20:] > 0).sum()
        
        # Volume at price extremes
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        
        high_volume_days = df.iloc[-20:][(df['high'].iloc[-20:] >= recent_high * 0.98)]
        low_volume_days = df.iloc[-20:][(df['low'].iloc[-20:] <= recent_low * 1.02)]
        
        features['volume_at_highs'] = high_volume_days['volume'].mean() / df['volume'].iloc[-20:].mean() if len(high_volume_days) > 0 else 0
        features['volume_at_lows'] = low_volume_days['volume'].mean() / df['volume'].iloc[-20:].mean() if len(low_volume_days) > 0 else 0
        
        # ========== PRICE ACTION FEATURES ==========
        
        # Returns
        features['return_5d'] = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1)
        features['return_10d'] = (df['close'].iloc[-1] / df['close'].iloc[-11] - 1)
        features['return_20d'] = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1)
        
        # Volatility
        returns = df['close'].pct_change()
        features['volatility_5d'] = returns.iloc[-5:].std()
        features['volatility_20d'] = returns.iloc[-20:].std()
        features['volatility_ratio'] = features['volatility_5d'] / features['volatility_20d'] if features['volatility_20d'] > 0 else 1
        
        # Price efficiency
        if 'price_efficiency' in df.columns:
            features['efficiency_current'] = df['price_efficiency'].iloc[-1]
            features['efficiency_ma10'] = df['price_efficiency'].rolling(10).mean().iloc[-1]
        
        # Distance from moving averages
        features['dist_from_ma20'] = (df['close'].iloc[-1] - df['close'].rolling(20).mean().iloc[-1]) / df['close'].rolling(20).mean().iloc[-1]
        features['dist_from_ma50'] = (df['close'].iloc[-1] - df['close'].rolling(50).mean().iloc[-1]) / df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else 0
        
        # ========== MOMENTUM INDICATORS ==========
        
        # RSI
        features['rsi'] = talib.RSI(df['close'].values)[-1]
        features['rsi_ma'] = talib.RSI(df['close'].values)[-10:].mean()
        features['rsi_trending_up'] = int(features['rsi'] > features['rsi_ma'])
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'].values)
        features['macd_histogram'] = hist[-1] if len(hist) > 0 else 0
        features['macd_histogram_ma'] = hist[-5:].mean() if len(hist) >= 5 else 0
        features['macd_positive'] = int(features['macd_histogram'] > 0)
        
        # ADX for trend strength
        features['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values)[-1]
        
        # ========== PATTERN FEATURES ==========
        
        # Candlestick patterns
        features['bullish_engulfing'] = talib.CDLENGULFING(df['open'].values, df['high'].values, 
                                                          df['low'].values, df['close'].values)[-1] / 100
        features['hammer'] = talib.CDLHAMMER(df['open'].values, df['high'].values,
                                            df['low'].values, df['close'].values)[-1] / 100
        
        # Price pattern
        features['higher_highs'] = int(df['high'].iloc[-1] > df['high'].iloc[-6:-1].max())
        features['higher_lows'] = int(df['low'].iloc[-1] > df['low'].iloc[-6:-1].min())
        
        # ========== MICRO CAP SPECIFIC ==========
        
        # Daily range (volatility proxy)
        features['daily_range_pct'] = ((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]) * 100
        features['avg_daily_range_20d'] = ((df['high'].iloc[-20:] - df['low'].iloc[-20:]) / df['close'].iloc[-20:]).mean() * 100
        
        # Close location in daily range
        features['close_location'] = (df['close'].iloc[-1] - df['low'].iloc[-1]) / (df['high'].iloc[-1] - df['low'].iloc[-1]) if df['high'].iloc[-1] > df['low'].iloc[-1] else 0.5
        
        # Gap analysis
        features['gap_up_today'] = int(df['open'].iloc[-1] > df['high'].iloc[-2]) if len(df) > 1 else 0
        features['gaps_last_10d'] = sum((df['open'].iloc[-10:] > df['high'].iloc[-11:-1]) | 
                                       (df['open'].iloc[-10:] < df['low'].iloc[-11:-1]))
        
        # Fill NaN and inf values
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features

class XGBoostBreakoutModel:
    """
    XGBoost model with self-optimization for breakout detection
    """
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_engineer = MicroCapFeatureEngineer()
        self.scaler = RobustScaler()
        self.feature_importance = {}
        self.selected_features = []
        self.performance_history = []
        
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                n_trials: int = 100, cv_folds: int = 5) -> Dict:
        """
        Optimize XGBoost hyperparameters using Optuna with TimeSeriesSplit
        """
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),  # For imbalanced data
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10)
            }
            
            # Add fixed parameters
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',  # Faster
                'random_state': 42,
                'n_jobs': -1
            })
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                # Scale features
                X_fold_train_scaled = self.scaler.fit_transform(X_fold_train)
                X_fold_val_scaled = self.scaler.transform(X_fold_val)
                
                # Create DMatrix
                dtrain = xgb.DMatrix(X_fold_train_scaled, label=y_fold_train)
                dval = xgb.DMatrix(X_fold_val_scaled, label=y_fold_val)
                
                # Train with early stopping
                pruning_callback = XGBoostPruningCallback(trial, f'validation-auc')
                
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=params['n_estimators'],
                    evals=[(dval, 'validation')],
                    early_stopping_rounds=50,
                    callbacks=[pruning_callback],
                    verbose_eval=False
                )
                
                # Predict and evaluate
                y_pred = model.predict(dval)
                score = roc_auc_score(y_fold_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Create study with pruning
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best XGBoost AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        self.best_params = study.best_params
        self.best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        })
        
        # Store optimization history
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'best_score': study.best_value,
            'n_trials': n_trials,
            'best_params': self.best_params
        })
        
        return self.best_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame, y_val: pd.Series,
             params: Optional[Dict] = None) -> xgb.Booster:
        """
        Train XGBoost model with given or optimized parameters
        """
        
        if params is None:
            params = self.best_params
        
        if params is None:
            # Use default parameters
            params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'random_state': 42
            }
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=X_train.columns.tolist())
        dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=X_val.columns.tolist())
        
        # Train model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 500),
            evals=[(dtrain, 'train'), (dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=True
        )
        
        # Calculate feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')
        
        # Select top features
        importance_df = pd.DataFrame.from_dict(
            self.feature_importance, 
            orient='index', 
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        # Select top 75% features by cumulative importance
        cumsum = importance_df['importance'].cumsum() / importance_df['importance'].sum()
        self.selected_features = cumsum[cumsum <= 0.95].index.tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features from {len(X_train.columns)}")
        
        # Evaluate
        y_pred_val = self.model.predict(dval)
        val_auc = roc_auc_score(y_val, y_pred_val)
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_val)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        logger.info(f"Validation AUC: {val_auc:.4f}")
        logger.info(f"Optimal threshold: {self.optimal_threshold:.4f}")
        
        return self.model
    
    def predict(self, X: pd.DataFrame, return_proba: bool = True) -> np.ndarray:
        """Make predictions"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(X_scaled, feature_names=X.columns.tolist())
        
        # Predict
        predictions = self.model.predict(dmatrix)
        
        if return_proba:
            return predictions
        else:
            return (predictions > self.optimal_threshold).astype(int)
    
    def explain_prediction(self, X: pd.DataFrame) -> Dict:
        """
        Explain prediction using SHAP values
        """
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)
        
        # Get feature contributions
        feature_contributions = {}
        for i, feature in enumerate(X.columns):
            feature_contributions[feature] = float(shap_values[0, i])
        
        # Sort by absolute contribution
        sorted_contributions = dict(sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return {
            'base_value': float(explainer.expected_value),
            'prediction': float(self.predict(X)[0]),
            'feature_contributions': sorted_contributions,
            'top_positive_factors': {k: v for k, v in sorted_contributions.items() if v > 0}[:5],
            'top_negative_factors': {k: v for k, v in sorted_contributions.items() if v < 0}[:5]
        }
    
    def self_optimize(self, recent_performance: List[Dict]) -> bool:
        """
        Self-optimization based on recent performance
        Retrain if performance degrades
        """
        
        if len(recent_performance) < 10:
            return False
        
        # Calculate recent metrics
        recent_accuracy = np.mean([p['correct'] for p in recent_performance[-20:]])
        historical_accuracy = np.mean([p['correct'] for p in recent_performance[:-20]]) if len(recent_performance) > 20 else recent_accuracy
        
        # Check if performance degraded
        performance_degraded = recent_accuracy < historical_accuracy * 0.9
        
        if performance_degraded:
            logger.warning(f"Performance degraded: Recent {recent_accuracy:.2%} vs Historical {historical_accuracy:.2%}")
            logger.info("Triggering self-optimization...")
            return True
        
        return False
    
    def save_model(self, path: str):
        """Save model and associated files"""
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(save_path / "xgboost_model.json"))
        
        # Save scaler
        joblib.dump(self.scaler, save_path / "scaler.pkl")
        
        # Save parameters and metadata
        metadata = {
            'best_params': self.best_params,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'optimal_threshold': self.optimal_threshold,
            'performance_history': self.performance_history
        }
        
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"XGBoost model saved to {save_path}")
    
    def load_model(self, path: str):
        """Load model and associated files"""
        
        load_path = Path(path)
        
        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(str(load_path / "xgboost_model.json"))
        
        # Load scaler
        self.scaler = joblib.load(load_path / "scaler.pkl")
        
        # Load metadata
        with open(load_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.best_params = metadata['best_params']
        self.selected_features = metadata['selected_features']
        self.feature_importance = metadata['feature_importance']
        self.optimal_threshold = metadata.get('optimal_threshold', 0.5)
        self.performance_history = metadata.get('performance_history', [])
        
        logger.info(f"XGBoost model loaded from {load_path}")
