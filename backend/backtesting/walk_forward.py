import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging
from .backtest_engine import BacktestEngine
from .performance_analyzer import PerformanceAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalkForwardAnalyzer:
    """Performs walk-forward analysis for strategy validation"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.backtest_engine = BacktestEngine(initial_capital)
        self.performance_analyzer = PerformanceAnalyzer()
        self.analysis_results: Dict[str, Any] = {}
    
    def run_walk_forward_analysis(self, 
                                symbol: str,
                                start_date: str,
                                end_date: str,
                                window_size: int = 252,
                                step_size: int = 63,
                                strategy_config: Dict = None) -> Dict[str, Any]:
        """
        Run walk-forward analysis
        
        Args:
            symbol: Trading symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            window_size: Training window size in days
            step_size: Step size between windows in days
            strategy_config: Strategy configuration
        
        Returns:
            Dictionary containing walk-forward analysis results
        """
        if strategy_config is None:
            strategy_config = {}
        
        try:
            # Convert dates to datetime
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Generate walk-forward windows
            windows = self._generate_walk_forward_windows(start_dt, end_dt, window_size, step_size)
            
            if not windows:
                logger.error("No valid windows generated for walk-forward analysis")
                return {}
            
            logger.info(f"Running walk-forward analysis with {len(windows)} windows")
            
            # Store results for each window
            window_results = []
            all_trades = []
            all_portfolio_values = []
            
            for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
                logger.info(f"Processing window {i+1}/{len(windows)}: "
                          f"Train {train_start.date()} to {train_end.date()}, "
                          f"Test {test_start.date()} to {test_end.date()}")
                
                try:
                    # Run backtest on test period
                    result = self.backtest_engine.run_backtest(
                        symbol=symbol,
                        start_date=test_start.strftime('%Y-%m-%d'),
                        end_date=test_end.strftime('%Y-%m-%d'),
                        strategy=strategy_config,
                        initial_capital=10000  # Reset capital for each window
                    )
                    
                    # Store window results
                    window_result = {
                        'window_id': i + 1,
                        'train_period': {
                            'start': train_start.strftime('%Y-%m-%d'),
                            'end': train_end.strftime('%Y-%m-%d')
                        },
                        'test_period': {
                            'start': test_start.strftime('%Y-%m-%d'),
                            'end': test_end.strftime('%Y-%m-%d')
                        },
                        'backtest_result': result,
                        'performance_metrics': self.performance_analyzer.analyze_performance(
                            result.get('trades', []),
                            result.get('portfolio_values', [])
                        )
                    }
                    
                    window_results.append(window_result)
                    
                    # Aggregate trades and portfolio values
                    all_trades.extend(result.get('trades', []))
                    
                    # Adjust portfolio values for cumulative analysis
                    test_portfolio_values = result.get('portfolio_values', [])
                    if test_portfolio_values and all_portfolio_values:
                        # Scale portfolio values to continue from previous window
                        last_value = all_portfolio_values[-1]
                        first_test_value = test_portfolio_values[0]
                        scale_factor = last_value / first_test_value if first_test_value > 0 else 1
                        
                        scaled_values = [pv * scale_factor for pv in test_portfolio_values]
                        all_portfolio_values.extend(scaled_values[1:])  # Skip first to avoid duplication
                    else:
                        all_portfolio_values.extend(test_portfolio_values)
                        
                except Exception as e:
                    logger.error(f"Error processing window {i+1}: {e}")
                    continue
            
            # Calculate overall performance
            overall_metrics = self.performance_analyzer.analyze_performance(all_trades, all_portfolio_values)
            
            # Calculate walk-forward efficiency and other metrics
            wfa_metrics = self._calculate_wfa_metrics(window_results)
            
            # Compile final results
            self.analysis_results = {
                'symbol': symbol,
                'analysis_period': {
                    'start': start_date,
                    'end': end_date
                },
                'parameters': {
                    'window_size': window_size,
                    'step_size': step_size,
                    'strategy_config': strategy_config
                },
                'window_results': window_results,
                'overall_performance': overall_metrics,
                'wfa_metrics': wfa_metrics,
                'summary': self._generate_summary(window_results, overall_metrics, wfa_metrics)
            }
            
            logger.info("Walk-forward analysis completed successfully")
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {e}")
            return {}
    
    def _generate_walk_forward_windows(self, 
                                     start_dt: datetime,
                                     end_dt: datetime,
                                     window_size: int,
                                     step_size: int) -> List[Tuple]:
        """Generate walk-forward windows"""
        windows = []
        current_start = start_dt
        
        while current_start + timedelta(days=window_size + step_size) <= end_dt:
            train_end = current_start + timedelta(days=window_size)
            test_start = train_end
            test_end = test_start + timedelta(days=step_size)
            
            # Ensure we don't exceed end date
            if test_end > end_dt:
                test_end = end_dt
            
            windows.append((current_start, train_end, test_start, test_end))
            
            # Move to next window
            current_start += timedelta(days=step_size)
        
        return windows
    
    def _calculate_wfa_metrics(self, window_results: List[Dict]) -> Dict[str, Any]:
        """Calculate walk-forward analysis specific metrics"""
        if not window_results:
            return {}
        
        # Extract performance metrics for each window
        sharpe_ratios = []
        total_returns = []
        max_drawdowns = []
        win_rates = []
        
        for window in window_results:
            metrics = window['performance_metrics']
            sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
            total_returns.append(metrics.get('total_return', 0))
            max_drawdowns.append(metrics.get('max_drawdown', 0))
            win_rates.append(metrics.get('win_rate', 0))
        
        # Calculate consistency metrics
        sharpe_consistency = np.std(sharpe_ratios) / np.mean(sharpe_ratios) if np.mean(sharpe_ratios) != 0 else 0
        return_consistency = np.std(total_returns) / np.mean(total_returns) if np.mean(total_returns) != 0 else 0
        
        # Calculate walk-forward efficiency
        positive_windows = sum(1 for ret in total_returns if ret > 0)
        wfa_efficiency = positive_windows / len(total_returns) if total_returns else 0
        
        # Calculate stability score (higher is more stable)
        stability_score = 1 / (1 + sharpe_consistency + return_consistency)
        
        # Identify best and worst performing windows
        best_window_idx = np.argmax(total_returns) if total_returns else 0
        worst_window_idx = np.argmin(total_returns) if total_returns else 0
        
        return {
            'num_windows': len(window_results),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_total_return': np.mean(total_returns),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'sharpe_consistency': sharpe_consistency,
            'return_consistency': return_consistency,
            'wfa_efficiency': wfa_efficiency,
            'stability_score': stability_score,
            'best_window': best_window_idx + 1,
            'worst_window': worst_window_idx + 1,
            'best_window_return': total_returns[best_window_idx] if total_returns else 0,
            'worst_window_return': total_returns[worst_window_idx] if total_returns else 0
        }
    
    def _generate_summary(self, 
                         window_results: List[Dict],
                         overall_metrics: Dict[str, Any],
                         wfa_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary"""
        # Count successful windows (positive return)
        successful_windows = sum(1 for window in window_results 
                               if window['performance_metrics'].get('total_return', 0) > 0)
        
        # Calculate strategy reliability
        reliability = successful_windows / len(window_results) if window_results else 0
        
        # Determine overall recommendation
        avg_sharpe = wfa_metrics.get('avg_sharpe_ratio', 0)
        avg_return = wfa_metrics.get('avg_total_return', 0)
        stability = wfa_metrics.get('stability_score', 0)
        
        if avg_sharpe > 1.0 and avg_return > 0.1 and reliability > 0.6:
            recommendation = "STRONG BUY"
            confidence = "HIGH"
        elif avg_sharpe > 0.5 and avg_return > 0.05 and reliability > 0.5:
            recommendation = "BUY"
            confidence = "MEDIUM"
        elif avg_sharpe > 0 and avg_return > 0:
            recommendation = "HOLD"
            confidence = "LOW"
        else:
            recommendation = "SELL"
            confidence = "HIGH"
        
        return {
            'total_windows_analyzed': len(window_results),
            'successful_windows': successful_windows,
            'reliability_score': reliability,
            'overall_recommendation': recommendation,
            'confidence_level': confidence,
            'key_strengths': self._identify_strengths(overall_metrics, wfa_metrics),
            'key_weaknesses': self._identify_weaknesses(overall_metrics, wfa_metrics),
            'suggested_improvements': self._suggest_improvements(overall_metrics, wfa_metrics)
        }
    
    def _identify_strengths(self, 
                           overall_metrics: Dict[str, Any],
                           wfa_metrics: Dict[str, Any]) -> List[str]:
        """Identify strategy strengths"""
        strengths = []
        
        if overall_metrics.get('sharpe_ratio', 0) > 1.0:
            strengths.append("Excellent risk-adjusted returns (Sharpe > 1.0)")
        
        if overall_metrics.get('win_rate', 0) > 0.6:
            strengths.append("High win rate (> 60%)")
        
        if wfa_metrics.get('wfa_efficiency', 0) > 0.7:
            strengths.append("Consistent performance across time periods")
        
        if overall_metrics.get('max_drawdown', 0) > -0.1:  # Less than 10% drawdown
            strengths.append("Good drawdown control")
        
        if wfa_metrics.get('stability_score', 0) > 0.7:
            strengths.append("Stable performance characteristics")
        
        return strengths if strengths else ["No significant strengths identified"]
    
    def _identify_weaknesses(self, 
                            overall_metrics: Dict[str, Any],
                            wfa_metrics: Dict[str, Any]) -> List[str]:
        """Identify strategy weaknesses"""
        weaknesses = []
        
        if overall_metrics.get('sharpe_ratio', 0) < 0:
            weaknesses.append("Poor risk-adjusted returns")
        
        if overall_metrics.get('win_rate', 0) < 0.4:
            weaknesses.append("Low win rate (< 40%)")
        
        if wfa_metrics.get('wfa_efficiency', 0) < 0.5:
            weaknesses.append("Inconsistent performance across time periods")
        
        if overall_metrics.get('max_drawdown', 0) < -0.2:  # More than 20% drawdown
            weaknesses.append("Excessive drawdowns")
        
        if wfa_metrics.get('return_consistency', 0) > 0.5:
            weaknesses.append("High return variability")
        
        return weaknesses if weaknesses else ["No significant weaknesses identified"]
    
    def _suggest_improvements(self, 
                             overall_metrics: Dict[str, Any],
                             wfa_metrics: Dict[str, Any]) -> List[str]:
        """Suggest strategy improvements"""
        improvements = []
        
        if overall_metrics.get('win_rate', 0) < 0.5:
            improvements.append("Consider improving entry signals to increase win rate")
        
        if overall_metrics.get('profit_factor', 0) < 1.5:
            improvements.append("Work on better risk management to improve profit factor")
        
        if wfa_metrics.get('return_consistency', 0) > 0.3:
            improvements.append("Add market regime detection for more consistent returns")
        
        if overall_metrics.get('max_drawdown', 0) < -0.15:
            improvements.append("Implement stricter stop-loss rules to control drawdowns")
        
        if wfa_metrics.get('wfa_efficiency', 0) < 0.6:
            improvements.append("Optimize strategy parameters for different market conditions")
        
        return improvements if improvements else ["Strategy appears well-optimized"]
    
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Extract optimal parameters from walk-forward analysis"""
        if not self.analysis_results:
            return {}
        
        window_results = self.analysis_results.get('window_results', [])
        
        if not window_results:
            return {}
        
        # Find best performing window
        best_window = max(window_results, 
                         key=lambda x: x['performance_metrics'].get('sharpe_ratio', -999))
        
        # Extract parameters from best window (this would typically come from strategy optimization)
        optimal_params = {
            'best_window_performance': best_window['performance_metrics'],
            'recommended_parameters': {
                'confidence_threshold': 0.7,
                'position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.04
            },
            'validation_metrics': {
                'out_of_sample_performance': best_window['performance_metrics'],
                'consistency_score': self.analysis_results['wfa_metrics'].get('stability_score', 0)
            }
        }
        
        return optimal_params
    
    def generate_report(self) -> str:
        """Generate comprehensive walk-forward analysis report"""
        if not self.analysis_results:
            return "No analysis results available"
        
        summary = self.analysis_results.get('summary', {})
        wfa_metrics = self.analysis_results.get('wfa_metrics', {})
        overall_metrics = self.analysis_results.get('overall_performance', {})
        
        report = []
        report.append("WALK-FORWARD ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Symbol: {self.analysis_results.get('symbol', 'N/A')}")
        report.append(f"Period: {self.analysis_results['analysis_period']['start']} to {self.analysis_results['analysis_period']['end']}")
        report.append(f"Windows: {wfa_metrics.get('num_windows', 0)}")
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(f"Overall Return: {overall_metrics.get('total_return', 0):.2%}")
        report.append(f"Overall Sharpe: {overall_metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Win Rate: {overall_metrics.get('win_rate', 0):.2%}")
        report.append("")
        
        # Walk-Forward Metrics
        report.append("WALK-FORWARD METRICS")
        report.append("-" * 30)
        report.append(f"WFA Efficiency: {wfa_metrics.get('wfa_efficiency', 0):.2%}")
        report.append(f"Stability Score: {wfa_metrics.get('stability_score', 0):.2f}")
        report.append(f"Return Consistency: {wfa_metrics.get('return_consistency', 0):.2f}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATION")
        report.append("-" * 30)
        report.append(f"Action: {summary.get('overall_recommendation', 'N/A')}")
        report.append(f"Confidence: {summary.get('confidence_level', 'N/A')}")
        report.append(f"Reliability: {summary.get('reliability_score', 0):.2%}")
        report.append("")
        
        # Strengths and Weaknesses
        report.append("STRENGTHS")
        report.append("-" * 30)
        for strength in summary.get('key_strengths', []):
            report.append(f"• {strength}")
        report.append("")
        
        report.append("WEAKNESSES")
        report.append("-" * 30)
        for weakness in summary.get('key_weaknesses', []):
            report.append(f"• {weakness}")
        report.append("")
        
        report.append("SUGGESTED IMPROVEMENTS")
        report.append("-" * 30)
        for improvement in summary.get('suggested_improvements', []):
            report.append(f"• {improvement}")
        
        return "\n".join(report)
