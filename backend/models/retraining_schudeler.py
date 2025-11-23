import schedule
import time
import threading
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class ModelRetrainer:
    """Automatic model retraining scheduler"""
    
    def __init__(self, retrain_interval_hours: int = 24):
        self.retrain_interval = retrain_interval_hours
        self.models = {}
        self.retraining_history = []
        
    def schedule_retraining(self):
        """Schedule periodic model retraining"""
        schedule.every(self.retrain_interval).hours.do(self.retrain_all_models)
        
        # Start scheduler in background thread
        scheduler_thread = threading.Thread(target=self._run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
    def _run_scheduler(self):
        """Run the scheduling loop"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def retrain_all_models(self):
        """Retrain all trading models"""
        print(f"Starting model retraining at {datetime.now()}")
        
        try:
            # Retrain signal prediction model
            self.retrain_signal_model()
            
            # Retrain price prediction model
            self.retrain_price_model()
            
            # Update retraining history
            self.retraining_history.append({
                'timestamp': datetime.now(),
                'status': 'success',
                'models_retrained': list(self.models.keys())
            })
            
            print("Model retraining completed successfully")
            
        except Exception as e:
            print(f"Model retraining failed: {e}")
            self.retraining_history.append({
                'timestamp': datetime.now(),
                'status': 'failed',
                'error': str(e)
            })
    
    def retrain_signal_model(self, symbol: str = 'AAPL'):
        """Retrain buy/sell signal prediction model"""
        try:
            # Fetch latest data (in production, this would come from your database)
            data = self._fetch_training_data(symbol)
            
            if len(data) < 100:  # Minimum data points required
                print(f"Insufficient data for {symbol}")
                return
            
            # Prepare features and target
            X, y = self._prepare_signal_features(data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model
            self.models[f'signal_{symbol}'] = {
                'model': model,
                'accuracy': accuracy,
                'last_trained': datetime.now(),
                'feature_names': list(X.columns)
            }
            
            # Save model to file
            joblib.dump(model, f'models/signal_model_{symbol}.pkl')
            
            print(f"Signal model for {symbol} retrained. Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            print(f"Error retraining signal model for {symbol}: {e}")
    
    def _fetch_training_data(self, symbol: str) -> pd.DataFrame:
        """Fetch training data for model retraining"""
        # This would typically come from your database
        # For demo purposes, we'll use yfinance
        import yfinance as yf
        
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period='2y', interval='1d')
            return data
        except Exception as e:
            print(f"Error fetching training data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _prepare_signal_features(self, data: pd.DataFrame) -> tuple:
        """Prepare features for signal prediction"""
        if len(data) < 50:
            return pd.DataFrame(), pd.Series()
        
        # Calculate features
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['momentum'] = data['Close'] / data['Close'].shift(5) - 1
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['Close'])
        features['macd'] = self._calculate_macd(data['Close'])
        
        # Volume features
        features['volume_sma_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        # Create target (1 if next day return positive, else 0)
        features['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        # Drop NaN values
        features = features.dropna()
        
        X = features.drop('target', axis=1)
        y = features['target']
        
        return X, y
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        return macd
