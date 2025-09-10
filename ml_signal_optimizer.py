"""
AI/ML Signal Quality Optimizer
Learns from historical signal performance to improve future signal quality
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import streamlit as st


class MLSignalOptimizer:
    """
    AI-powered signal quality optimizer that learns from historical performance
    """
    
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.performance_model = None
        self.quality_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.training_data = []
        
        # Create models directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Load existing models if available
        self.load_models()
    
    def extract_features(self, signal: Dict, market_data: pd.DataFrame) -> Dict:
        """
        Extract features from signal and market data for ML training
        
        Args:
            signal: Trading signal dictionary
            market_data: Historical OHLCV data
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic signal features
        features['confluences'] = signal.get('confluences', 0)
        features['rr_ratio'] = signal.get('rr', 0)
        features['entry_price'] = signal.get('entry', 0)
        features['stop_loss'] = signal.get('sl', 0)
        features['take_profit'] = signal.get('tp', 0)
        
        # Market context features
        current_price = market_data['close'].iloc[-1]
        features['price_distance'] = abs(signal.get('entry', 0) - current_price) / current_price
        features['atr_ratio'] = self._calculate_atr_ratio(signal, market_data)
        features['volume_ratio'] = self._calculate_volume_ratio(market_data)
        features['volatility'] = self._calculate_volatility(market_data)
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(market_data)
        features['macd_signal'] = self._calculate_macd_signal(market_data)
        features['bollinger_position'] = self._calculate_bollinger_position(signal, market_data)
        
        # Time-based features
        features['hour_of_day'] = datetime.now().hour
        features['day_of_week'] = datetime.now().weekday()
        
        # Market type encoding
        market_type = signal.get('market_type', 'forex')
        features['is_forex'] = 1 if market_type == 'forex' else 0
        features['is_crypto'] = 1 if market_type == 'crypto' else 0
        features['is_stocks'] = 1 if market_type == 'stocks' else 0
        
        return features
    
    def _calculate_atr_ratio(self, signal: Dict, market_data: pd.DataFrame) -> float:
        """Calculate ATR ratio for risk assessment"""
        try:
            high = market_data['high'].values
            low = market_data['low'].values
            close = market_data['close'].values
            
            # Calculate ATR
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(tr[-14:])  # 14-period ATR
            
            price_risk = abs(signal.get('entry', 0) - signal.get('sl', 0))
            return price_risk / atr if atr > 0 else 0
        except:
            return 0
    
    def _calculate_volume_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate recent volume ratio"""
        try:
            recent_volume = market_data['volume'].tail(5).mean()
            avg_volume = market_data['volume'].mean()
            return recent_volume / avg_volume if avg_volume > 0 else 1
        except:
            return 1
    
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate price volatility"""
        try:
            returns = market_data['close'].pct_change().dropna()
            return returns.std() * np.sqrt(252)  # Annualized volatility
        except:
            return 0
    
    def _calculate_rsi(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = market_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50
    
    def _calculate_macd_signal(self, market_data: pd.DataFrame) -> float:
        """Calculate MACD signal strength"""
        try:
            exp1 = market_data['close'].ewm(span=12).mean()
            exp2 = market_data['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9).mean()
            return (macd.iloc[-1] - signal_line.iloc[-1]) / market_data['close'].iloc[-1]
        except:
            return 0
    
    def _calculate_bollinger_position(self, signal: Dict, market_data: pd.DataFrame) -> float:
        """Calculate position within Bollinger Bands"""
        try:
            sma = market_data['close'].rolling(20).mean()
            std = market_data['close'].rolling(20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            entry_price = signal.get('entry', 0)
            if entry_price == 0:
                return 0.5
            
            position = (entry_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            return max(0, min(1, position))
        except:
            return 0.5
    
    def add_training_sample(self, signal: Dict, market_data: pd.DataFrame, 
                          actual_outcome: str, actual_pnl: float = None):
        """
        Add a training sample for ML model
        
        Args:
            signal: Original signal
            market_data: Market data at signal time
            actual_outcome: 'win', 'loss', or 'breakeven'
            actual_pnl: Actual profit/loss (optional)
        """
        features = self.extract_features(signal, market_data)
        
        # Add outcome labels
        features['outcome'] = 1 if actual_outcome == 'win' else 0
        features['pnl'] = actual_pnl if actual_pnl is not None else 0
        
        # Add signal metadata
        features['signal_id'] = f"{signal.get('symbol', '')}_{datetime.now().timestamp()}"
        features['timestamp'] = datetime.now()
        
        self.training_data.append(features)
        
        # Save training data periodically
        if len(self.training_data) % 100 == 0:
            self.save_training_data()
    
    def train_models(self):
        """Train ML models on accumulated data"""
        if len(self.training_data) < 50:
            st.warning("⚠️ Need at least 50 training samples to train models")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(self.training_data)
        
        # Prepare features and targets
        feature_columns = [col for col in df.columns if col not in 
                          ['outcome', 'pnl', 'signal_id', 'timestamp']]
        
        X = df[feature_columns].fillna(0)
        y_outcome = df['outcome']
        y_pnl = df['pnl']
        
        # Split data
        X_train, X_test, y_outcome_train, y_outcome_test = train_test_split(
            X, y_outcome, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train outcome prediction model
        self.performance_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        self.performance_model.fit(X_train_scaled, y_outcome_train)
        
        # Train P&L prediction model
        self.quality_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        self.quality_model.fit(X_train_scaled, y_outcome_train)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            feature_columns, 
            self.performance_model.feature_importances_
        ))
        
        # Evaluate models
        y_pred = self.performance_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_outcome_test, y_pred)
        precision = precision_score(y_outcome_test, y_pred)
        recall = recall_score(y_outcome_test, y_pred)
        
        st.success(f"✅ Models trained successfully!")
        st.write(f"**Accuracy:** {accuracy:.2%}")
        st.write(f"**Precision:** {precision:.2%}")
        st.write(f"**Recall:** {recall:.2%}")
        
        # Save models
        self.save_models()
        
        return True
    
    def predict_signal_quality(self, signal: Dict, market_data: pd.DataFrame) -> Dict:
        """
        Predict signal quality using trained ML models
        
        Args:
            signal: Trading signal
            market_data: Current market data
            
        Returns:
            Dictionary with quality predictions
        """
        if self.performance_model is None:
            # When no model is trained, provide reasonable defaults
            # This allows signals to pass through for initial data collection
            return {
                'win_probability': 0.6,  # Slightly optimistic default
                'expected_pnl': 0.1,     # Small positive expectation
                'quality_score': 50,     # Neutral quality score
                'confidence': 0.8        # High confidence to allow signals through
            }
        
        # Extract features
        features = self.extract_features(signal, market_data)
        feature_columns = [col for col in features.keys() if col not in 
                          ['outcome', 'pnl', 'signal_id', 'timestamp']]
        
        # Prepare feature vector
        X = np.array([features[col] for col in feature_columns]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        win_probability = self.performance_model.predict_proba(X_scaled)[0][1]
        expected_pnl = self.quality_model.predict(X_scaled)[0]
        
        # Calculate quality score
        quality_score = (win_probability * 100) + (expected_pnl * 10)
        confidence = min(win_probability, 1 - win_probability) * 2
        
        return {
            'win_probability': win_probability,
            'expected_pnl': expected_pnl,
            'quality_score': quality_score,
            'confidence': confidence
        }
    
    def optimize_signal_parameters(self, signal: Dict, market_data: pd.DataFrame) -> Dict:
        """
        Optimize signal parameters using ML insights
        
        Args:
            signal: Original signal
            market_data: Market data
            
        Returns:
            Optimized signal with improved parameters
        """
        optimized_signal = signal.copy()
        
        if self.performance_model is None:
            return optimized_signal
        
        # Get current quality prediction
        quality_pred = self.predict_signal_quality(signal, market_data)
        
        # Adjust parameters based on ML insights
        if quality_pred['win_probability'] < 0.6:
            # Increase stop loss distance for lower probability signals
            current_sl = signal.get('sl', 0)
            entry = signal.get('entry', 0)
            if current_sl > 0:
                sl_adjustment = abs(entry - current_sl) * 0.1
                if signal.get('direction') == 'Buy':
                    optimized_signal['sl'] = current_sl - sl_adjustment
                else:
                    optimized_signal['sl'] = current_sl + sl_adjustment
        
        # Adjust take profit based on expected P&L
        if quality_pred['expected_pnl'] > 0:
            current_tp = signal.get('tp', 0)
            entry = signal.get('entry', 0)
            if current_tp > 0:
                tp_adjustment = abs(entry - current_tp) * 0.05
                if signal.get('direction') == 'Buy':
                    optimized_signal['tp'] = current_tp + tp_adjustment
                else:
                    optimized_signal['tp'] = current_tp - tp_adjustment
        
        # Add ML predictions to signal
        optimized_signal['ml_win_probability'] = quality_pred['win_probability']
        optimized_signal['ml_expected_pnl'] = quality_pred['expected_pnl']
        optimized_signal['ml_quality_score'] = quality_pred['quality_score']
        optimized_signal['ml_confidence'] = quality_pred['confidence']
        
        return optimized_signal
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        return self.feature_importance
    
    def save_models(self):
        """Save trained models to disk"""
        if self.performance_model is not None:
            joblib.dump(self.performance_model, 
                       os.path.join(self.model_path, 'performance_model.pkl'))
        
        if self.quality_model is not None:
            joblib.dump(self.quality_model, 
                       os.path.join(self.model_path, 'quality_model.pkl'))
        
        joblib.dump(self.scaler, 
                   os.path.join(self.model_path, 'scaler.pkl'))
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            self.performance_model = joblib.load(
                os.path.join(self.model_path, 'performance_model.pkl'))
            self.quality_model = joblib.load(
                os.path.join(self.model_path, 'quality_model.pkl'))
            self.scaler = joblib.load(
                os.path.join(self.model_path, 'scaler.pkl'))
        except FileNotFoundError:
            # Models don't exist yet, will be created on first training
            pass
    
    def save_training_data(self):
        """Save training data to CSV"""
        if self.training_data:
            df = pd.DataFrame(self.training_data)
            df.to_csv(os.path.join(self.model_path, 'training_data.csv'), 
                     index=False)
    
    def load_training_data(self):
        """Load existing training data"""
        try:
            df = pd.read_csv(os.path.join(self.model_path, 'training_data.csv'))
            self.training_data = df.to_dict('records')
        except FileNotFoundError:
            self.training_data = []
