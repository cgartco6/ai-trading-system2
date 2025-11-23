import threading
import time
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from .broker_api import DemoBroker, RealTimeData
from signals.signal_generator import AISignalGenerator, SignalType
from models.model_manager import ModelManager
from src.agents.memory_agent import MemoryAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeExecutor:
    """Executes trades based on AI signals"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.broker = DemoBroker(initial_capital)
        self.real_time_data = RealTimeData()
        self.signal_generator = AISignalGenerator({})
        self.model_manager = ModelManager()
        self.memory_agent = MemoryAgent()
        
        self.is_running = False
        self.trading_thread = None
        self.execution_history: List[Dict[str, Any]] = []
        
        # Trading parameters
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        self.position_sizes = {symbol: 0.1 for symbol in self.symbols}  # 10% per symbol
        self.max_total_position = 0.5  # 50% maximum total position
        
    def start_trading(self, demo_mode: bool = True):
        """Start the trading execution"""
        if self.is_running:
            logger.warning("Trading is already running")
            return
        
        self.is_running = True
        self.broker.demo_mode = demo_mode
        
        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
        logger.info(f"Trading started in {'demo' if demo_mode else 'live'} mode")
    
    def stop_trading(self):
        """Stop the trading execution"""
        self.is_running = False
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        logger.info("Trading stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                # Process each symbol
                for symbol in self.symbols:
                    self._process_symbol(symbol)
                
                # Wait before next iteration
                time.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(30)  # Wait 30 seconds on error
    
    def _process_symbol(self, symbol: str):
        """Process trading for a single symbol"""
        try:
            # Get real-time data
            price_data = self.real_time_data.get_real_time_price(symbol)
            if not price_data:
                return
            
            current_price = price_data['price']
            
            # Get historical data for signal generation
            hist_data = self._get_historical_data(symbol, days=30)
            if hist_data.empty:
                return
            
            # Generate trading signal
            signals = self.signal_generator.generate_signals(hist_data)
            if signals.empty:
                return
            
            latest_signal = signals.iloc[-1]
            signal_action = latest_signal['action']
            confidence = latest_signal.get('confidence', 0.5)
            
            # Only trade if confidence is high enough
            if confidence < 0.6:
                return
            
            # Get current position
            current_position = self.broker.positions.get(symbol, 0)
            
            # Execute trade based on signal
            if signal_action == SignalType.BUY and current_position == 0:
                self._execute_buy(symbol, current_price, confidence)
            elif signal_action == SignalType.SELL and current_position > 0:
                self._execute_sell(symbol, current_price, confidence)
            elif signal_action == SignalType.HOLD and current_position > 0:
                # Consider partial profit taking
                self._consider_profit_taking(symbol, current_price, confidence)
                
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
    
    def _execute_buy(self, symbol: str, price: float, confidence: float):
        """Execute a buy order"""
        try:
            # Calculate position size
            portfolio_value = self.broker.get_portfolio_value(
                {sym: self.real_time_data.get_real_time_price(sym)['price'] 
                 for sym in self.symbols if sym in self.broker.positions}
            )
            
            position_value = portfolio_value * self.position_sizes[symbol] * confidence
            quantity = int(position_value / price)
            
            if quantity > 0:
                # Place buy order
                order = self.broker.place_order(symbol, 'BUY', quantity, price)
                
                # Record execution
                execution_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'confidence': confidence,
                    'order_id': order.get('order_id'),
                    'status': order.get('status')
                }
                
                self.execution_history.append(execution_record)
                self.memory_agent.record_trade(execution_record)
                
                logger.info(f"BUY order executed: {symbol} {quantity} @ ${price:.2f}")
                
        except Exception as e:
            logger.error(f"Error executing BUY for {symbol}: {e}")
    
    def _execute_sell(self, symbol: str, price: float, confidence: float):
        """Execute a sell order"""
        try:
            current_position = self.broker.positions.get(symbol, 0)
            
            if current_position > 0:
                # Place sell order
                order = self.broker.place_order(symbol, 'SELL', current_position, price)
                
                # Record execution
                execution_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': current_position,
                    'price': price,
                    'confidence': confidence,
                    'order_id': order.get('order_id'),
                    'status': order.get('status')
                }
                
                self.execution_history.append(execution_record)
                self.memory_agent.record_trade(execution_record)
                
                logger.info(f"SELL order executed: {symbol} {current_position} @ ${price:.2f}")
                
        except Exception as e:
            logger.error(f"Error executing SELL for {symbol}: {e}")
    
    def _consider_profit_taking(self, symbol: str, current_price: float, confidence: float):
        """Consider taking profits on existing positions"""
        try:
            current_position = self.broker.positions.get(symbol, 0)
            
            if current_position > 0:
                # Get average purchase price
                avg_price = self._get_average_purchase_price(symbol)
                
                if avg_price:
                    profit_percentage = (current_price - avg_price) / avg_price
                    
                    # Take profits if significant gain
                    if profit_percentage > 0.1:  # 10% profit
                        sell_quantity = int(current_position * 0.5)  # Sell half
                        
                        if sell_quantity > 0:
                            order = self.broker.place_order(symbol, 'SELL', sell_quantity, current_price)
                            
                            execution_record = {
                                'timestamp': datetime.now(),
                                'symbol': symbol,
                                'action': 'SELL_PARTIAL',
                                'quantity': sell_quantity,
                                'price': current_price,
                                'profit_percentage': profit_percentage,
                                'confidence': confidence,
                                'order_id': order.get('order_id'),
                                'status': order.get('status')
                            }
                            
                            self.execution_history.append(execution_record)
                            self.memory_agent.record_trade(execution_record)
                            
                            logger.info(f"Partial profit taking: {symbol} {sell_quantity} @ ${current_price:.2f} "
                                      f"({profit_percentage:.2%} profit)")
                            
        except Exception as e:
            logger.error(f"Error in profit taking for {symbol}: {e}")
    
    def _get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data for signal generation"""
        try:
            import yfinance as yf
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days * 2)  # Get extra data for indicators
            
            data = stock.history(start=start_date, end=end_date, interval='1d')
            return data
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_average_purchase_price(self, symbol: str) -> Optional[float]:
        """Calculate average purchase price for a symbol"""
        try:
            buy_trades = [
                trade for trade in self.execution_history
                if trade['symbol'] == symbol and trade['action'] in ['BUY', 'SELL']
            ]
            
            if not buy_trades:
                return None
            
            total_cost = 0
            total_shares = 0
            
            for trade in buy_trades:
                if trade['action'] == 'BUY':
                    total_cost += trade['quantity'] * trade['price']
                    total_shares += trade['quantity']
                elif trade['action'] == 'SELL':
                    total_shares -= trade['quantity']
                    # For simplicity, we don't adjust cost basis on partial sells
                    # In a real system, you'd use FIFO or other accounting method
            
            return total_cost / total_shares if total_shares > 0 else None
            
        except Exception as e:
            logger.error(f"Error calculating average price for {symbol}: {e}")
            return None
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        current_prices = {}
        for symbol in self.symbols:
            price_data = self.real_time_data.get_real_time_price(symbol)
            if price_data:
                current_prices[symbol] = price_data['price']
        
        portfolio_value = self.broker.get_portfolio_value(current_prices)
        cash = self.broker.balance
        
        positions = {}
        for symbol, quantity in self.broker.positions.items():
            if quantity > 0 and symbol in current_prices:
                current_value = quantity * current_prices[symbol]
                avg_price = self._get_average_purchase_price(symbol)
                unrealized_pnl = (current_prices[symbol] - avg_price) * quantity if avg_price else 0
                
                positions[symbol] = {
                    'quantity': quantity,
                    'current_price': current_prices[symbol],
                    'current_value': current_value,
                    'average_price': avg_price,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_percent': unrealized_pnl / (avg_price * quantity) if avg_price else 0
                }
        
        return {
            'portfolio_value': portfolio_value,
            'cash': cash,
            'positions': positions,
            'total_unrealized_pnl': sum(pos['unrealized_pnl'] for pos in positions.values()),
            'timestamp': datetime.now()
        }
    
    def get_trading_performance(self) -> Dict[str, Any]:
        """Get trading performance metrics"""
        if not self.execution_history:
            return {}
        
        # Calculate basic metrics
        total_trades = len(self.execution_history)
        winning_trades = len([t for t in self.execution_history if t.get('profit', 0) > 0])
        losing_trades = len([t for t in self.execution_history if t.get('profit', 0) < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.get('profit', 0) for t in self.execution_history)
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit_per_trade': avg_profit_per_trade,
            'last_trade_time': self.execution_history[-1]['timestamp'] if self.execution_history else None
        }
    
    def get_recent_executions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trade executions"""
        return self.execution_history[-limit:] if self.execution_history else []
