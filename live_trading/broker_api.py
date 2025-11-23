import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
import requests
import time

class RealTimeData:
    """Real-time market data using Yahoo Finance and Alpha Vantage"""
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        
    def get_real_time_price(self, symbol: str) -> Dict[str, float]:
        """Get real-time price data"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period='1d', interval='1m')
            
            if not hist.empty:
                return {
                    'symbol': symbol,
                    'price': hist['Close'].iloc[-1],
                    'volume': hist['Volume'].iloc[-1],
                    'timestamp': pd.Timestamp.now(),
                    'change': info.get('regularMarketChangePercent', 0)
                }
        except Exception as e:
            print(f"Error getting real-time data for {symbol}: {e}")
        
        return {}
    
    def get_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """Get real-time technical indicators"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='2mo', interval='1d')
            
            if len(hist) > 20:
                # Calculate basic technical indicators
                closes = hist['Close']
                
                # RSI
                delta = closes.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                # Moving averages
                sma_20 = closes.rolling(20).mean()
                sma_50 = closes.rolling(50).mean()
                
                return {
                    'rsi': rsi.iloc[-1],
                    'sma_20': sma_20.iloc[-1],
                    'sma_50': sma_50.iloc[-1],
                    'price_vs_sma20': (closes.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1],
                    'volume_sma': hist['Volume'].rolling(20).mean().iloc[-1]
                }
        except Exception as e:
            print(f"Error calculating indicators for {symbol}: {e}")
        
        return {}

class DemoBroker:
    """Demo trading broker for testing predictions"""
    
    def __init__(self, initial_balance: float = 10000):
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        
    def place_order(self, symbol: str, signal: str, quantity: int, price: float) -> Dict:
        """Place a demo trade order"""
        order = {
            'symbol': symbol,
            'signal': signal,
            'quantity': quantity,
            'price': price,
            'timestamp': pd.Timestamp.now(),
            'status': 'filled'
        }
        
        if signal.upper() == 'BUY':
            cost = quantity * price
            if cost <= self.balance:
                self.balance -= cost
                if symbol in self.positions:
                    self.positions[symbol] += quantity
                else:
                    self.positions[symbol] = quantity
                order['type'] = 'BUY'
                order['cost'] = cost
            else:
                order['status'] = 'rejected'
                
        elif signal.upper() == 'SELL':
            if symbol in self.positions and self.positions[symbol] >= quantity:
                self.balance += quantity * price
                self.positions[symbol] -= quantity
                if self.positions[symbol] == 0:
                    del self.positions[symbol]
                order['type'] = 'SELL'
                order['proceeds'] = quantity * price
            else:
                order['status'] = 'rejected'
        
        self.trade_history.append(order)
        return order
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        stock_value = sum(
            quantity * current_prices.get(symbol, 0) 
            for symbol, quantity in self.positions.items()
        )
        return self.balance + stock_value
    
    def get_performance(self) -> Dict[str, float]:
        """Get trading performance metrics"""
        if not self.trade_history:
            return {}
            
        profitable_trades = [
            t for t in self.trade_history 
            if t['status'] == 'filled' and t.get('profit', 0) > 0
        ]
        
        total_trades = len([t for t in self.trade_history if t['status'] == 'filled'])
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'current_balance': self.balance,
            'positions': len(self.positions)
        }
