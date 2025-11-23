import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import deque
import json

class MemoryAgent:
    """Agent that remembers past predictions and learns from them"""
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.prediction_memory = deque(maxlen=memory_size)
        self.pattern_memory = {}
        self.performance_history = []
        
    def record_prediction(self, prediction: Dict[str, Any]):
        """Record a prediction and its outcome"""
        # Store prediction
        self.prediction_memory.append(prediction)
        
        # Update outcome when available
        if prediction.get('actual_result') is not None:
            self._update_pattern_memory(prediction)
            
    def _update_pattern_memory(self, prediction: Dict):
        """Update pattern memory based on prediction outcomes"""
        symbol = prediction['symbol']
        signal = prediction['signal']
        is_correct = prediction['is_correct']
        
        key = f"{symbol}_{signal}"
        
        if key not in self.pattern_memory:
            self.pattern_memory[key] = {
                'total': 0,
                'correct': 0,
                'confidence_sum': 0,
                'recent_outcomes': deque(maxlen=50)
            }
        
        memory = self.pattern_memory[key]
        memory['total'] += 1
        memory['confidence_sum'] += prediction.get('confidence', 0.5)
        
        if is_correct:
            memory['correct'] += 1
            
        memory['recent_outcomes'].append(is_correct)
        
    def calculate_recent_accuracy(self, lookback_days: int = 7) -> float:
        """Calculate accuracy for recent predictions"""
        recent_time = datetime.now() - timedelta(days=lookback_days)
        recent_predictions = [
            p for p in self.prediction_memory 
            if datetime.fromisoformat(p['timestamp']) > recent_time
            and p.get('is_correct') is not None
        ]
        
        if not recent_predictions:
            return 0.0
            
        correct = sum(1 for p in recent_predictions if p['is_correct'])
        return correct / len(recent_predictions)
    
    def get_trading_insights(self) -> Dict[str, Any]:
        """Get insights from memory for improving trading"""
        insights = {
            'best_performing_symbols': self._get_best_symbols(),
            'optimal_confidence_threshold': self._calculate_optimal_confidence(),
            'pattern_success_rates': self._get_pattern_success_rates(),
            'time_based_insights': self._get_time_insights(),
            'recommendations': self._generate_recommendations()
        }
        return insights
    
    def _get_best_symbols(self, min_trades: int = 5) -> List[Dict]:
        """Get best performing trading symbols"""
        symbol_stats = {}
        
        for prediction in self.prediction_memory:
            if prediction.get('is_correct') is not None:
                symbol = prediction['symbol']
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {'total': 0, 'correct': 0}
                
                symbol_stats[symbol]['total'] += 1
                if prediction['is_correct']:
                    symbol_stats[symbol]['correct'] += 1
        
        # Calculate accuracy
        best_symbols = []
        for symbol, stats in symbol_stats.items():
            if stats['total'] >= min_trades:
                accuracy = stats['correct'] / stats['total']
                best_symbols.append({
                    'symbol': symbol,
                    'accuracy': accuracy,
                    'total_trades': stats['total']
                })
        
        return sorted(best_symbols, key=lambda x: x['accuracy'], reverse=True)[:5]
    
    def _calculate_optimal_confidence(self) -> float:
        """Calculate optimal confidence threshold"""
        if not self.prediction_memory:
            return 0.7
            
        # Analyze relationship between confidence and accuracy
        confidence_buckets = {}
        
        for prediction in self.prediction_memory:
            if prediction.get('is_correct') is not None:
                confidence = prediction.get('confidence', 0.5)
                bucket = round(confidence * 10) / 10  # Bucket by 0.1
                
                if bucket not in confidence_buckets:
                    confidence_buckets[bucket] = {'total': 0, 'correct': 0}
                
                confidence_buckets[bucket]['total'] += 1
                if prediction['is_correct']:
                    confidence_buckets[bucket]['correct'] += 1
        
        # Find bucket with highest accuracy
        best_bucket = 0.7
        best_accuracy = 0
        
        for bucket, stats in confidence_buckets.items():
            if stats['total'] > 10:  # Minimum samples
                accuracy = stats['correct'] / stats['total']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_bucket = bucket
        
        return best_bucket
    
    def _get_pattern_success_rates(self) -> Dict[str, float]:
        """Get success rates for different trading patterns"""
        pattern_rates = {}
        
        for pattern_key, memory in self.pattern_memory.items():
            if memory['total'] >= 5:  # Minimum samples
                success_rate = memory['correct'] / memory['total']
                pattern_rates[pattern_key] = success_rate
        
        return dict(sorted(pattern_rates.items(), key=lambda x: x[1], reverse=True))
    
    def _get_time_insights(self) -> Dict[str, Any]:
        """Get time-based trading insights"""
        hour_performance = {}
        
        for prediction in self.prediction_memory:
            if prediction.get('is_correct') is not None:
                hour = datetime.fromisoformat(prediction['timestamp']).hour
                if hour not in hour_performance:
                    hour_performance[hour] = {'total': 0, 'correct': 0}
                
                hour_performance[hour]['total'] += 1
                if prediction['is_correct']:
                    hour_performance[hour]['correct'] += 1
        
        # Calculate best trading hours
        best_hours = []
        for hour, stats in hour_performance.items():
            if stats['total'] >= 3:
                accuracy = stats['correct'] / stats['total']
                best_hours.append({'hour': hour, 'accuracy': accuracy})
        
        return {
            'best_trading_hours': sorted(best_hours, key=lambda x: x['accuracy'], reverse=True)[:3],
            'worst_trading_hours': sorted(best_hours, key=lambda x: x['accuracy'])[:3]
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate trading recommendations based on memory"""
        recommendations = []
        
        insights = self.get_trading_insights()
        
        # Best symbols recommendation
        best_symbols = insights['best_performing_symbols']
        if best_symbols:
            top_symbol = best_symbols[0]
            if top_symbol['accuracy'] > 0.7:
                recommendations.append(f"Focus on {top_symbol['symbol']} - {top_symbol['accuracy']:.1%} accuracy")
        
        # Confidence threshold recommendation
        optimal_conf = insights['optimal_confidence_threshold']
        recommendations.append(f"Use confidence threshold of {optimal_conf:.2f} for better accuracy")
        
        # Trading hours recommendation
        best_hours = insights['time_based_insights']['best_trading_hours']
        if best_hours:
            best_hour = best_hours[0]
            recommendations.append(f"Best trading time: {best_hour['hour']}:00 UTC ({best_hour['accuracy']:.1%} accuracy)")
        
        return recommendations
