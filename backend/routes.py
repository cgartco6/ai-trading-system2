from flask import Blueprint, jsonify, request, render_template
from backend.app import backend, TradingBackend
from models.model_manager import ModelManager
from live_trading.trade_executor import TradeExecutor
from backtesting.backtest_engine import BacktestEngine
from backtesting.performance_analyzer import PerformanceAnalyzer
from backtesting.walk_forward import WalkForwardAnalyzer
import pandas as pd
import json
from datetime import datetime, timedelta

# Create blueprint
routes = Blueprint('routes', __name__)

# Initialize components
trading_backend = TradingBackend()
model_manager = ModelManager()
trade_executor = TradeExecutor()
backtest_engine = BacktestEngine()
performance_analyzer = PerformanceAnalyzer()
walk_forward = WalkForwardAnalyzer()

@routes.route('/')
def index():
    return render_template('index.html')

@routes.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@routes.route('/api/trading/start', methods=['POST'])
def start_trading():
    try:
        data = request.get_json()
        demo_mode = data.get('demo_mode', True)
        
        trading_backend.start_live_trading(demo_mode)
        
        return jsonify({
            'status': 'success',
            'message': 'Trading started successfully',
            'demo_mode': demo_mode
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to start trading: {str(e)}'
        }), 500

@routes.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    try:
        trading_backend.stop_trading()
        return jsonify({
            'status': 'success',
            'message': 'Trading stopped successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to stop trading: {str(e)}'
        }), 500

@routes.route('/api/trading/status')
def trading_status():
    status = trading_backend.get_trading_status()
    return jsonify(status)

@routes.route('/api/predictions')
def get_predictions():
    try:
        predictions = trading_backend.get_recent_predictions(limit=50)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get predictions: {str(e)}'
        }), 500

@routes.route('/api/performance')
def get_performance():
    try:
        performance = trading_backend.get_performance_metrics()
        return jsonify(performance)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get performance: {str(e)}'
        }), 500

@routes.route('/api/memory/insights')
def get_memory_insights():
    try:
        insights = trading_backend.get_memory_insights()
        return jsonify(insights)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get memory insights: {str(e)}'
        }), 500

@routes.route('/api/models/retrain', methods=['POST'])
def retrain_models():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        
        success = model_manager.retrain_models(symbol)
        
        return jsonify({
            'status': 'success' if success else 'error',
            'message': 'Models retrained successfully' if success else 'Failed to retrain models'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to retrain models: {str(e)}'
        }), 500

@routes.route('/api/backtest/run', methods=['POST'])
def run_backtest():
    try:
        data = request.get_json()
        
        symbol = data.get('symbol', 'AAPL')
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2023-12-31')
        strategy_config = data.get('strategy', {})
        initial_capital = data.get('initial_capital', 10000)
        
        result = backtest_engine.run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy_config,
            initial_capital=initial_capital
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to run backtest: {str(e)}'
        }), 500

@routes.route('/api/backtest/performance', methods=['POST'])
def analyze_performance():
    try:
        data = request.get_json()
        trades = data.get('trades', [])
        portfolio_values = data.get('portfolio_values', [])
        
        analysis = performance_analyzer.analyze_performance(trades, portfolio_values)
        
        return jsonify(analysis)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to analyze performance: {str(e)}'
        }), 500

@routes.route('/api/backtest/walk_forward', methods=['POST'])
def run_walk_forward():
    try:
        data = request.get_json()
        
        symbol = data.get('symbol', 'AAPL')
        start_date = data.get('start_date', '2022-01-01')
        end_date = data.get('end_date', '2023-12-31')
        window_size = data.get('window_size', 252)  # 1 year
        step_size = data.get('step_size', 63)       # 3 months
        
        result = walk_forward.run_walk_forward_analysis(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            step_size=step_size
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to run walk-forward analysis: {str(e)}'
        }), 500

@routes.route('/api/risk/metrics')
def get_risk_metrics():
    try:
        from live_trading.risk_monitor import RiskMonitor
        risk_monitor = RiskMonitor()
        metrics = risk_monitor.calculate_current_risk()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get risk metrics: {str(e)}'
        }), 500

@routes.route('/api/data/historical')
def get_historical_data():
    try:
        symbol = request.args.get('symbol', 'AAPL')
        period = request.args.get('period', '1y')
        interval = request.args.get('interval', '1d')
        
        import yfinance as yf
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        
        # Convert to list of dicts for JSON serialization
        historical_data = []
        for index, row in data.iterrows():
            historical_data.append({
                'timestamp': index.isoformat(),
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume']
            })
        
        return jsonify(historical_data)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get historical data: {str(e)}'
        }), 500

@routes.route('/api/signals/generate', methods=['POST'])
def generate_signals():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        
        from signals.signal_generator import AISignalGenerator
        signal_generator = AISignalGenerator({})
        
        # Get recent data
        import yfinance as yf
        stock = yf.Ticker(symbol)
        market_data = stock.history(period='3mo', interval='1d')
        
        signals = signal_generator.generate_signals(market_data)
        
        return jsonify({
            'symbol': symbol,
            'signals': signals.to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to generate signals: {str(e)}'
        }), 500

# Register blueprint
backend.register_blueprint(routes)
