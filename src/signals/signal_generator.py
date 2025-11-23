import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import ta
from transformers import pipeline

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class AISignalGenerator:
    """AI-powered trading signal generator"""
    
    def __init__(self, model_config: Dict):
        self.technical_weights = model_config.get('technical_weights', {})
        self.sentiment_weights = model_config.get('sentiment_weights', {})
        self.thresholds = model_config.get('thresholds', {})
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis"
        )
        
    def generate_signals(self, 
                        data: pd.DataFrame,
                        news_data: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate comprehensive trading signals"""
        
        signals = pd.DataFrame(index=data.index)
        
        # Technical signals
        technical_signals = self._generate_technical_signals(data)
        
        # AI/ML signals
        ml_signals = self._generate_ml_signals(data)
        
        # Sentiment signals
        sentiment_signals = self._generate_sentiment_signals(news_data) if news_data else 0
        
        # Combine signals
        combined_signals = self._combine_signals(
            technical_signals, ml_signals, sentiment_signals
        )
        
        signals['final_signal'] = combined_signals
        signals['action'] = signals['final_signal'].apply(self._signal_to_action)
        
        return signals
    
    def _generate_technical_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate technical analysis signals"""
        signals = []
        
        # RSI
        rsi = ta.momentum.RSIIndicator(data['close']).rsi()
        rsi_signal = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        
        # MACD
        macd = ta.trend.MACD(data['close'])
        macd_signal = np.where(macd.macd_diff() > 0, 1, -1)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['close'])
        bb_position = (data['close'] - bollinger.bollinger_lband()) / \
                     (bollinger.bollinger_hband() - bollinger.bollinger_lband())
        bb_signal = np.where(bb_position < 0.2, 1, np.where(bb_position > 0.8, -1, 0))
        
        # Combine technical signals
        for i in range(len(data)):
            tech_score = (
                self.technical_weights.get('rsi', 0.3) * rsi_signal[i] +
                self.technical_weights.get('macd', 0.4) * macd_signal[i] +
                self.technical_weights.get('bollinger', 0.3) * bb_signal[i]
            )
            signals.append(tech_score)
            
        return pd.Series(signals, index=data.index)
    
    def _generate_ml_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate machine learning based signals"""
        # Feature engineering
        features = self._create_features(data)
        
        # Simple ensemble model (replace with actual trained models)
        predictions = self._ensemble_prediction(features)
        
        return predictions
    
    def _generate_sentiment_signals(self, news_data: List[str]) -> float:
        """Generate sentiment-based signals from news"""
        if not news_data:
            return 0.0
            
        sentiments = []
        for news in news_data[:10]:  # Analyze top 10 news
            try:
                result = self.sentiment_analyzer(news[:512])[0]  # Truncate for model
                score = 1 if result['label'] == 'POS' else -1 if result['label'] == 'NEG' else 0
                sentiments.append(score * result['score'])
            except:
                continue
                
        return np.mean(sentiments) if sentiments else 0.0
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML models"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['momentum'] = data['close'] / data['close'].shift(5) - 1
        
        # Volume features
        features['volume_sma'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Technical indicators as features
        features['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
        features['macd'] = ta.trend.MACD(data['close']).macd_diff()
        
        return features.fillna(0)
    
    def _ensemble_prediction(self, features: pd.DataFrame) -> pd.Series:
        """Simple ensemble prediction (replace with actual model inference)"""
        # This would typically load pre-trained models
        prediction = (
            0.6 * np.tanh(features['returns'].rolling(5).mean()) +
            0.4 * np.tanh(features['rsi'] / 50 - 1)
        )
        return prediction
    
    def _combine_signals(self, 
                        technical: pd.Series, 
                        ml: pd.Series, 
                        sentiment: float) -> pd.Series:
        """Combine different signal types"""
        combined = (
            self.technical_weights.get('total', 0.4) * technical +
            self.technical_weights.get('ml', 0.4) * ml +
            self.technical_weights.get('sentiment', 0.2) * sentiment
        )
        return combined
    
    def _signal_to_action(self, signal: float) -> SignalType:
        """Convert signal strength to action"""
        if signal > self.thresholds.get('buy', 0.3):
            return SignalType.BUY
        elif signal < self.thresholds.get('sell', -0.3):
            return SignalType.SELL
        else:
            return SignalType.HOLD
