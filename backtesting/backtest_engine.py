import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import yfinance as yf

class BacktestEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def run_backtest(self, 
                    symbol: str, 
                    start_date: str, 
                    end_date: str,
                    strategy: Any) -> Dict[str, Any]:
        """Run backtest for a given symbol and strategy"""
        
        # Fetch historical data
        data = self._fetch_historical_data(symbol, start_date, end_date)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0
        trades = []
        portfolio_values = []
        
        for i in range(1, len(data)):
            current_data = data.iloc[:i]
            current_price = data['Close'].iloc[i]
            
            # Get strategy signal
            signal = strategy.generate_signal(current_data)
            
            # Execute trading logic
            if signal == 'BUY' and position == 0:
                # Buy signal
                position_size = capital // current_price
                if position_size > 0:
                    capital -= position_size * current_price
                    position = position_size
                    trades.append({
                        'timestamp': data.index[i],
                        'action': 'BUY',
                        'price': current_price,
                        'shares': position_size
                    })
                    
            elif signal == 'SELL' and position > 0:
                # Sell signal
                capital += position * current_price
                trades.append({
                    'timestamp': data.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position
                })
                position = 0
            
            # Calculate portfolio value
            portfolio_value = capital + (position * current_price)
            portfolio_values.append(portfolio_value)
        
        # Calculate performance metrics
        performance = self._calculate_performance(portfolio_values, trades)
        
        return {
            'symbol': symbol,
            'period': f"{start_date} to {end_date}",
            'initial_capital': self.initial_capital,
            'final_portfolio': portfolio_values[-1] if portfolio_values else self.initial_capital,
            'total_return': (portfolio_values[-1] - self.initial_capital) / self.initial_capital if portfolio_values else 0,
            'trades': trades,
            'performance_metrics': performance
        }
    
    def _fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_performance(self, portfolio_values: List[float], trades: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not portfolio_values:
            return {}
        
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Basic metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Drawdown calculation
        rolling_max = pd.Series(portfolio_values).cummax()
        drawdown = (pd.Series(portfolio_values) - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        winning_trades = len([t for t in trades if t.get('profit', 0) > 0])
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': total_return * 252 / len(portfolio_values),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': returns.std() * np.sqrt(252),
            'win_rate': win_rate,
            'total_trades': total_trades,
            'profit_factor': self._calculate_profit_factor(trades)
        }
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor from trades"""
        gross_profit = sum(t.get('profit', 0) for t in trades if t.get('profit', 0) > 0)
        gross_loss = abs(sum(t.get('profit', 0) for t in trades if t.get('profit', 0) < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
