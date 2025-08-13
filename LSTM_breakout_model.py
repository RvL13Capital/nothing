# lstm_breakout_model.py
"""
Advanced LSTM Seq2Seq Model with Self-Optimization
Specialized for Micro/Small Cap 40%+ Breakout Detection
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import optuna
from optuna.trial import Trial
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

class BreakoutDataset(Dataset):
    """Dataset for LSTM training with proper sequence handling"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 sequence_length: int = 30):
        """
        Args:
            features: Feature array (n_samples, n_features)
            labels: Labels array (n_samples, n_outputs)
            sequence_length: Length of input sequences
        """
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        
        # Create valid indices for sequences
        self.valid_indices = []
        for i in range(len(features) - sequence_length):
            self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Get sequence
        X = self.features[start_idx:end_idx]
        
        # Get target (next day after sequence)
        y = self.labels[end_idx]
        
        return torch.FloatTensor(X), torch.FloatTensor(y)

class AttentionModule(nn.Module):
    """Self-attention module for important time points"""
    
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        attn_out, attn_weights = self.attention(x, x, x)
        return self.norm(x + attn_out), attn_weights

class BreakoutLSTMSeq2Seq(nn.Module):
    """
    Advanced LSTM for breakout prediction
    Seq2Seq architecture with attention mechanism
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.use_attention = config.get('use_attention', True)
        
        # Encoder LSTM (Bidirectional)
        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        if self.use_attention:
            self.attention = AttentionModule(self.hidden_size * 2)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Output heads for different predictions
        self.breakout_head = nn.Sequential(
            nn.Linear(self.hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Breakout probability [0, 1]
        )
        
        self.magnitude_head = nn.Sequential(
            nn.Linear(self.hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Magnitude (positive)
        )
        
        self.timing_head = nn.Sequential(
            nn.Linear(self.hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Days to breakout (positive)
        )
        
        # Volume surge prediction (important for micro caps)
        self.volume_head = nn.Sequential(
            nn.Linear(self.hidden_size // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # Volume multiplier (positive)
        )
    
    def forward(self, x):
        # Encode sequence
        lstm_out, (hidden, cell) = self.encoder(x)
        
        # Apply attention if enabled
        if self.use_attention:
            lstm_out, attention_weights = self.attention(lstm_out)
        else:
            attention_weights = None
        
        # Use last hidden state
        if self.encoder.bidirectional:
            # Concatenate forward and backward hidden states
            last_hidden = lstm_out[:, -1, :]
        else:
            last_hidden = hidden[-1]
        
        # Extract features
        features = self.feature_extractor(last_hidden)
        
        # Generate predictions
        breakout_prob = self.breakout_head(features)
        magnitude = self.magnitude_head(features)
        timing = self.timing_head(features)
        volume_surge = self.volume_head(features)
        
        # Combine outputs
        output = torch.cat([breakout_prob, magnitude, timing, volume_surge], dim=1)
        
        return output, attention_weights

class LSTMTrainer:
    """Training and optimization manager for LSTM"""
    
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self.input_size = len(feature_columns)
        self.best_config = None
        self.best_model = None
        self.scaler = None
        
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                n_trials: int = 50) -> Dict:
        """
        Optimize hyperparameters using Optuna
        """
        
        def objective(trial: Trial) -> float:
            # Suggest hyperparameters
            config = {
                'input_size': self.input_size,
                'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'sequence_length': trial.suggest_categorical('sequence_length', [20, 30, 40, 50]),
                'use_attention': trial.suggest_categorical('use_attention', [True, False]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
            }
            
            # Create datasets
            train_dataset = BreakoutDataset(X_train, y_train, config['sequence_length'])
            val_dataset = BreakoutDataset(X_val, y_val, config['sequence_length'])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False
            )
            
            # Create model
            model = BreakoutLSTMSeq2Seq(config)
            
            # Setup training
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
            
            # Custom loss function for multi-task learning
            criterion = self._create_loss_function()
            
            # Train for limited epochs (for speed)
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(30):  # Limited epochs for optimization
                # Training
                model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    output, _ = model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        output, _ = model(X_batch)
                        loss = criterion(output, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        break
                
                # Pruning
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return best_val_loss
        
        # Run optimization
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best LSTM validation loss: {study.best_value:.4f}")
        self.best_config = study.best_params
        self.best_config['input_size'] = self.input_size
        
        return self.best_config
    
    def train_final_model(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         config: Optional[Dict] = None,
                         epochs: int = 100) -> BreakoutLSTMSeq2Seq:
        """
        Train final model with best configuration
        """
        
        if config is None:
            config = self.best_config
        
        # Create datasets
        train_dataset = BreakoutDataset(X_train, y_train, config['sequence_length'])
        val_dataset = BreakoutDataset(X_val, y_val, config['sequence_length'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        # Create model
        model = BreakoutLSTMSeq2Seq(config)
        
        # Setup training
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        criterion = self._create_loss_function()
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_breakout_acc = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output, _ = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate breakout accuracy
                breakout_pred = (output[:, 0] > 0.5).float()
                breakout_true = (y_batch[:, 0] > 0.5).float()
                train_breakout_acc += (breakout_pred == breakout_true).float().mean().item()
            
            train_loss /= len(train_loader)
            train_breakout_acc /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            val_breakout_acc = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output, _ = model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()
                    
                    breakout_pred = (output[:, 0] > 0.5).float()
                    breakout_true = (y_batch[:, 0] > 0.5).float()
                    val_breakout_acc += (breakout_pred == breakout_true).float().mean().item()
            
            val_loss /= len(val_loader)
            val_breakout_acc /= len(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                          f"Train Acc: {train_breakout_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_breakout_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        self.best_model = model
        
        return model
    
    def _create_loss_function(self):
        """Create multi-task loss function"""
        
        class BreakoutLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.bce = nn.BCELoss()
                self.mse = nn.MSELoss()
                self.huber = nn.HuberLoss()
            
            def forward(self, output, target):
                # Split outputs
                breakout_pred = output[:, 0]
                magnitude_pred = output[:, 1]
                timing_pred = output[:, 2]
                volume_pred = output[:, 3]
                
                # Split targets
                breakout_true = target[:, 0]
                magnitude_true = target[:, 1]
                timing_true = target[:, 2]
                volume_true = target[:, 3] if target.shape[1] > 3 else torch.ones_like(timing_true)
                
                # Calculate losses
                breakout_loss = self.bce(breakout_pred, breakout_true)
                
                # Only calculate magnitude/timing loss for positive examples
                positive_mask = breakout_true > 0.5
                
                if positive_mask.any():
                    magnitude_loss = self.huber(
                        magnitude_pred[positive_mask],
                        magnitude_true[positive_mask]
                    )
                    timing_loss = self.huber(
                        timing_pred[positive_mask],
                        timing_true[positive_mask]
                    )
                    volume_loss = self.mse(
                        volume_pred[positive_mask],
                        volume_true[positive_mask]
                    )
                else:
                    magnitude_loss = torch.tensor(0.0)
                    timing_loss = torch.tensor(0.0)
                    volume_loss = torch.tensor(0.0)
                
                # Weighted combination
                total_loss = (
                    breakout_loss * 1.0 +
                    magnitude_loss * 0.3 +
                    timing_loss * 0.2 +
                    volume_loss * 0.1
                )
                
                return total_loss
        
        return BreakoutLoss()
    
    def predict(self, X: np.ndarray, model: Optional[BreakoutLSTMSeq2Seq] = None) -> np.ndarray:
        """Make predictions"""
        
        if model is None:
            model = self.best_model
        
        if model is None:
            raise ValueError("No trained model available")
        
        model.eval()
        
        # Create dataset
        # For prediction, we need at least sequence_length samples
        sequence_length = self.best_config['sequence_length']
        
        if len(X) < sequence_length:
            # Pad with zeros if necessary
            padding = np.zeros((sequence_length - len(X), X.shape[1]))
            X = np.vstack([padding, X])
        
        # Create batch
        X_tensor = torch.FloatTensor(X[-sequence_length:]).unsqueeze(0)
        
        with torch.no_grad():
            output, attention = model(X_tensor)
        
        return output.numpy()[0], attention
    
    def save_model(self, path: str):
        """Save model and configuration"""
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.best_model.state_dict(),
            'config': self.best_config,
            'feature_columns': self.feature_columns
        }, save_path / "lstm_model.pt")
        
        # Save scaler if exists
        if self.scaler:
            joblib.dump(self.scaler, save_path / "lstm_scaler.pkl")
        
        logger.info(f"LSTM model saved to {save_path}")
    
    def load_model(self, path: str):
        """Load model and configuration"""
        
        load_path = Path(path)
        
        # Load model
        checkpoint = torch.load(load_path / "lstm_model.pt")
        self.best_config = checkpoint['config']
        self.feature_columns = checkpoint['feature_columns']
        
        # Create model
        self.best_model = BreakoutLSTMSeq2Seq(self.best_config)
        self.best_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load scaler if exists
        scaler_path = load_path / "lstm_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        logger.info(f"LSTM model loaded from {load_path}")
