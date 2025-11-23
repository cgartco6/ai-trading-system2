import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Analyzes trading performance and calculates metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def analyze_performance(self, trades: List[Dict], portfolio_values: List[float]) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        if not trades or not portfolio_values:
            return self._get_empty_metrics()
        
        try:
            # Convert to DataFrames for easier analysis
            trades_df = self._trades_to_dataframe(trades)
            portfolio_df = self._portfolio_to_dataframe(portfolio_values)
            
            # Calculate metrics
            basic_metrics = self._calculate_basic_metrics(trades_df, portfolio_df)
            risk_metrics = self._calculate_risk_metrics(portfolio_df)
            trade_metrics = self._calculate_trade_metrics(trades_df)
            timing_metrics = self._calculate_timing_metrics(trades_df)
            
            # Combine all metrics
            performance_metrics = {
                **basic_metrics,
                **risk_metrics,
                **trade_metrics,
                **timing_metrics,
                'analysis_timestamp': datetime.now()
            }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return self._get_empty_metrics()
    
    def _trades_to_dataframe(self, trades: List[Dict]) -> pd.DataFrame:
        """Convert trades list to DataFrame"""
        if not trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(trades)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def _portfolio_to_dataframe(self, portfolio_values: List[float]) -> pd.DataFrame:
        """Convert portfolio values to DataFrame with timestamps"""
        if not portfolio_values:
            return pd.DataFrame()
        
        # Create synthetic timestamps if not provided
        if isinstance(portfolio_values[0], dict) and 'timestamp' in portfolio_values[0]:
            df = pd.DataFrame(portfolio_values)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            # Create daily timestamps
            start_date = datetime.now() - timedelta(days=len(portfolio_values))
            dates = pd.date_range(start=start_date, periods=len(portfolio_values), freq='D')
            df = pd.DataFrame({
                'timestamp': dates,
                'portfolio_value': portfolio_values
            })
        
        df = df.sort_values('timestamp')
        return df
    
    def _calculate_basic_metrics(self, trades_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        if portfolio_df.empty:
            return {}
        
        initial_value = portfolio_df['portfolio_value'].iloc[0]
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        daily_returns = portfolio_df['daily_return'].dropna()
        
        if len(daily_returns) == 0:
            return {
                'total_return': total_return,
                'annualized_return': 0,
                'cagr': 0
            }
        
        # Annualized return
        days = (portfolio_df['timestamp'].iloc[-1] - portfolio_df['timestamp'].iloc[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # CAGR
        cagr = (final_value / initial_value) ** (365 / days) - 1 if days > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cagr': cagr,
            'initial_value': initial_value,
            'final_value': final_value,
            'total_days': days
        }
    
    def _calculate_risk_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-adjusted metrics"""
        if portfolio_df.empty or 'daily_return' not in portfolio_df.columns:
            return {}
        
        daily_returns = portfolio_df['daily_return'].dropna()
        
        if len(daily_returns) == 0:
            return {}
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        excess_returns = daily_returns - (self.risk_free_rate / 252)
        sharpe_ratio = (excess_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Sortino Ratio (only downside risk)
        downside_returns = daily_returns[daily_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (daily_returns.mean() * 252 - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum Drawdown
        portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Calmar Ratio
        calmar_ratio = -portfolio_df['portfolio_value'].pct_change().mean() * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (95%)
        var_95 = np.percentile(daily_returns, 5)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _calculate_trade_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-based metrics"""
        if trades_df.empty:
            return {}
        
        # Filter only completed trades
        completed_trades = trades_df[trades_df['status'] == 'filled']
        
        if completed_trades.empty:
            return {}
        
        # Calculate win rate
        winning_trades = completed_trades[completed_trades['profit'] > 0]
        losing_trades = completed_trades[completed_trades['profit'] < 0]
        
        total_trades = len(completed_trades)
        win_rate = len(winning_trades) / total_trades
        
        # Average profit/loss
        avg_profit = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['profit'].sum()
        gross_loss = abs(losing_trades['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate * avg_profit) + ((1 - win_rate) * avg_loss)
        
        # Largest winning and losing trades
        largest_win = winning_trades['profit'].max() if len(winning_trades) > 0 else 0
        largest_loss = losing_trades['profit'].min() if len(losing_trades) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_trade_return': completed_trades['profit'].mean()
        }
    
    def _calculate_timing_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate timing-related metrics"""
        if trades_df.empty or 'timestamp' not in trades_df.columns:
            return {}
        
        completed_trades = trades_df[trades_df['status'] == 'filled']
        
        if len(completed_trades) < 2:
            return {}
        
        # Calculate holding periods
        completed_trades = completed_trades.sort_values('timestamp')
        completed_trades['next_timestamp'] = completed_trades['timestamp'].shift(-1)
        completed_trades['holding_period'] = (
            completed_trades['next_timestamp'] - completed_trades['timestamp']
        ).dt.total_seconds() / 3600  # Convert to hours
        
        avg_holding_period = completed_trades['holding_period'].mean()
        
        # Trade frequency
        first_trade = completed_trades['timestamp'].min()
        last_trade = completed_trades['timestamp'].max()
        total_hours = (last_trade - first_trade).total_seconds() / 3600
        
        trades_per_hour = len(completed_trades) / total_hours if total_hours > 0 else 0
        
        # Consistency metrics
        monthly_trades = completed_trades.groupby(
            completed_trades['timestamp'].dt.to_period('M')
        ).size()
        
        trade_consistency = monthly_trades.std() / monthly_trades.mean() if len(monthly_trades) > 0 else 0
        
        return {
            'avg_holding_period_hours': avg_holding_period,
            'trades_per_hour': trades_per_hour,
            'trade_consistency': trade_consistency,
            'first_trade': first_trade.isoformat(),
            'last_trade': last_trade.isoformat()
        }
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'total_return': 0,
            'annualized_return': 0,
            'cagr': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0,
            'var_95': 0,
            'cvar_95': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_trade_return': 0,
            'avg_holding_period_hours': 0,
            'trades_per_hour': 0,
            'trade_consistency': 0,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def compare_strategies(self, strategy_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare multiple trading strategies"""
        comparison = {}
        
        for strategy_name, results in strategy_results.items():
            metrics = self.analyze_performance(
                results.get('trades', []),
                results.get('portfolio_values', [])
            )
            comparison[strategy_name] = metrics
        
        # Calculate rankings
        strategies = list(comparison.keys())
        
        # Rank by Sharpe Ratio
        sharpe_ratios = {s: comparison[s].get('sharpe_ratio', -999) for s in strategies}
        sharpe_ranking = sorted(strategies, key=lambda x: sharpe_ratios[x], reverse=True)
        
        # Rank by Total Return
        total_returns = {s: comparison[s].get('total_return', -999) for s in strategies}
        return_ranking = sorted(strategies, key=lambda x: total_returns[x], reverse=True)
        
        # Rank by Max Drawdown (lower is better)
        max_drawdowns = {s: comparison[s].get('max_drawdown', 999) for s in strategies}
        drawdown_ranking = sorted(strategies, key=lambda x: max_drawdowns[x])
        
        comparison['rankings'] = {
            'by_sharpe_ratio': sharpe_ranking,
            'by_total_return': return_ranking,
            'by_max_drawdown': drawdown_ranking
        }
        
        return comparison
    
    def generate_performance_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable performance report"""
        report = []
        report.append("TRADING PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {metrics.get('analysis_timestamp', 'N/A')}")
        report.append("")
        
        # Basic Performance
        report.append("BASIC PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
        report.append(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        report.append(f"CAGR: {metrics.get('cagr', 0):.2%}")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-" * 30)
        report.append(f"Volatility: {metrics.get('volatility', 0):.2%}")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        report.append("")
        
        # Trade Analysis
        report.append("TRADE ANALYSIS")
        report.append("-" * 30)
        report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
        report.append(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        report.append(f"Expectancy: ${metrics.get('expectancy', 0):.2f}")
        
        return "\n".join(report)
