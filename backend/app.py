from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

class TradingBackend:
    def __init__(self):
        self.live_predictions = []
        self.performance_history = []
        self.memory_agent = MemoryAgent()
        self.is_live = False
        
    def start_live_trading(self, demo_account=True):
        """Start live trading with demo account"""
        self.is_live = True
        # Start background thread for live predictions
        thread = threading.Thread(target=self._live_prediction_loop)
        thread.daemon = True
        thread.start()
        
    def _live_prediction_loop(self):
        """Background loop for live predictions"""
        while self.is_live:
            try:
                prediction = self._generate_live_prediction()
                self.live_predictions.append(prediction)
                
                # Store in memory agent
                self.memory_agent.record_prediction(prediction)
                
                # Wait before next prediction
                time.sleep(60)  # Every minute
            except Exception as e:
                print(f"Prediction error: {e}")
                time.sleep(10)

    def _generate_live_prediction(self):
        """Generate live prediction using real market data"""
        # Use real API data here (example with yfinance)
        import yfinance as yf
        
        # Get real market data
        ticker = "AAPL"
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", interval="1m")
        
        if len(hist) > 0:
            current_price = hist['Close'].iloc[-1]
            
            # Generate AI prediction
            prediction = {
                'timestamp': datetime.now().isoformat(),
                'symbol': ticker,
                'predicted_price': current_price * (1 + np.random.normal(0, 0.02)),  # Replace with real AI model
                'current_price': current_price,
                'confidence': np.random.uniform(0.5, 0.95),
                'signal': 'BUY' if np.random.random() > 0.5 else 'SELL',
                'actual_result': None,  # Will be updated later
                'is_correct': None
            }
            
            return prediction
        return None

backend = TradingBackend()

@app.route('/')
def serve_frontend():
    return render_template('index.html')

@app.route('/api/start_trading', methods=['POST'])
def start_trading():
    demo_mode = request.json.get('demo_mode', True)
    backend.start_live_trading(demo_account=demo_mode)
    return jsonify({'status': 'success', 'message': 'Trading started'})

@app.route('/api/predictions')
def get_predictions():
    predictions = backend.live_predictions[-50:]  # Last 50 predictions
    return jsonify(predictions)

@app.route('/api/performance')
def get_performance():
    if backend.live_predictions:
        # Calculate performance metrics
        correct_predictions = [p for p in backend.live_predictions if p.get('is_correct')]
        accuracy = len(correct_predictions) / len(backend.live_predictions) if backend.live_predictions else 0
        
        performance = {
            'total_predictions': len(backend.live_predictions),
            'correct_predictions': len(correct_predictions),
            'accuracy': accuracy,
            'recent_accuracy': backend.memory_agent.calculate_recent_accuracy()
        }
        return jsonify(performance)
    return jsonify({'total_predictions': 0, 'correct_predictions': 0, 'accuracy': 0})

@app.route('/api/memory_insights')
def get_memory_insights():
    insights = backend.memory_agent.get_trading_insights()
    return jsonify(insights)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
