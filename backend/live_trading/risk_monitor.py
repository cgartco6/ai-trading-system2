import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from .trade_executor import TradeExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskMonitor:
    """Monitors and manages trading risk"""
    
    def __init__(self, trade_executor: TradeExecutor):
        self.trade_executor = trade_executor
        self.risk_metrics_history: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        # Risk parameters
        self.max_drawdown_limit = 0.10  # 10% maximum drawdown
        self.max_position_limit = 0.20  # 20% maximum position size
        self.max_daily_loss_limit = 0.05  # 5% maximum daily loss
        self.volatility_limit = 0.30  # 30% maximum annualized volatility
        self.var_confidence = 0.95  # 95% VaR confidence level
        
    def calculate_current_risk(self) -> Dict[str, Any]:
        """Calculate current risk metrics"""
        try:
            portfolio_status = self.trade_executor.get_portfolio_status()
            performance = self.trade_executor.get_trading_performance()
            
            # Basic metrics
            portfolio_value = portfolio_status['portfolio_value']
            cash = portfolio_status['cash']
            positions = portfolio_status['positions']
            
            # Calculate position concentration
            position_concentration = {}
            for symbol, position_data in positions.items():
                concentration = position_data['current_value'] / portfolio_value
                position_concentration[symbol] = concentration
            
            # Calculate drawdown
            drawdown = self._calculate_drawdown(portfolio_value)
            
            # Calculate volatility (simplified)
            volatility = self._calculate_volatility()
            
            # Calculate Value at Risk (VaR)
            var = self._calculate_var(portfolio_value)
            
            # Check risk limits
            risk_breaches = self._check_risk_limits({
                'drawdown': drawdown,
                'position_concentration': position_concentration,
                'volatility': volatility,
                'var': var
            })
            
            risk_metrics = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'cash_ratio': cash / portfolio_value,
                'drawdown': drawdown,
                'volatility': volatility,
                'var_95': var,
                'position_concentration': position_concentration,
                'max_position_concentration': max(position_concentration.values()) if position_concentration else 0,
                'risk_breaches': risk_breaches,
                'win_rate': performance.get('win_rate', 0),
                'total_trades': performance.get('total_trades', 0)
            }
            
            # Store in history
            self.risk_metrics_history.append(risk_metrics)
            
            # Keep only last 1000 records
            if len(self.risk_metrics_history) > 1000:
                self.risk_metrics_history = self.risk_metrics_history[-1000:]
            
            # Generate alerts if necessary
            self._generate_alerts(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_drawdown(self, current_value: float) -> float:
        """Calculate current drawdown from peak"""
        if not self.risk_metrics_history:
            return 0.0
        
        # Get peak portfolio value from history
        peak_value = max([metrics['portfolio_value'] for metrics in self.risk_metrics_history])
        
        if peak_value == 0:
            return 0.0
        
        drawdown = (peak_value - current_value) / peak_value
        return max(0.0, drawdown)  # Ensure non-negative
    
    def _calculate_volatility(self, window: int = 30) -> float:
        """Calculate portfolio volatility"""
        if len(self.risk_metrics_history) < window:
            return 0.0
        
        # Get recent portfolio values
        recent_values = [metrics['portfolio_value'] for metrics in self.risk_metrics_history[-window:]]
        
        if len(recent_values) < 2:
            return 0.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(recent_values)):
            ret = (recent_values[i] - recent_values[i-1]) / recent_values[i-1]
            returns.append(ret)
        
        if not returns:
            return 0.0
        
        # Annualize volatility (assuming daily data)
        daily_volatility = np.std(returns)
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility
    
    def _calculate_var(self, portfolio_value: float, horizon: int = 1) -> float:
        """Calculate Value at Risk"""
        if len(self.risk_metrics_history) < 30:
            return portfolio_value * 0.05  # Default 5% VaR
        
        # Get recent portfolio returns
        recent_values = [metrics['portfolio_value'] for metrics in self.risk_metrics_history[-30:]]
        
        returns = []
        for i in range(1, len(recent_values)):
            ret = (recent_values[i] - recent_values[i-1]) / recent_values[i-1]
            returns.append(ret)
        
        if not returns:
            return portfolio_value * 0.05
        
        # Calculate VaR using historical simulation
        var = np.percentile(returns, (1 - self.var_confidence) * 100)
        var_amount = portfolio_value * abs(var)
        
        return var_amount
    
    def _check_risk_limits(self, risk_metrics: Dict[str, Any]) -> List[str]:
        """Check for risk limit breaches"""
        breaches = []
        
        # Drawdown check
        if risk_metrics['drawdown'] > self.max_drawdown_limit:
            breaches.append(f"Drawdown {risk_metrics['drawdown']:.2%} exceeds limit {self.max_drawdown_limit:.2%}")
        
        # Position concentration check
        if risk_metrics['max_position_concentration'] > self.max_position_limit:
            breaches.append(f"Position concentration {risk_metrics['max_position_concentration']:.2%} exceeds limit {self.max_position_limit:.2%}")
        
        # Volatility check
        if risk_metrics['volatility'] > self.volatility_limit:
            breaches.append(f"Volatility {risk_metrics['volatility']:.2%} exceeds limit {self.volatility_limit:.2%}")
        
        # Daily loss check (simplified)
        if len(self.risk_metrics_history) >= 2:
            current_value = risk_metrics['portfolio_value']
            previous_value = self.risk_metrics_history[-2]['portfolio_value']
            daily_return = (current_value - previous_value) / previous_value
            
            if daily_return < -self.max_daily_loss_limit:
                breaches.append(f"Daily loss {daily_return:.2%} exceeds limit {-self.max_daily_loss_limit:.2%}")
        
        return breaches
    
    def _generate_alerts(self, risk_metrics: Dict[str, Any]):
        """Generate risk alerts"""
        for breach in risk_metrics['risk_breaches']:
            alert = {
                'timestamp': datetime.now(),
                'level': 'HIGH',
                'message': breach,
                'metric': risk_metrics
            }
            
            self.alert_history.append(alert)
            logger.warning(f"RISK ALERT: {breach}")
        
        # Generate warning for high risk levels
        if risk_metrics['drawdown'] > self.max_drawdown_limit * 0.8:
            alert = {
                'timestamp': datetime.now(),
                'level': 'MEDIUM',
                'message': f"Drawdown approaching limit: {risk_metrics['drawdown']:.2%}",
                'metric': risk_metrics
            }
            self.alert_history.append(alert)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary"""
        if not self.risk_metrics_history:
            return {}
        
        current_risk = self.risk_metrics_history[-1]
        
        # Calculate historical risk statistics
        drawdowns = [metrics['drawdown'] for metrics in self.risk_metrics_history]
        volatilities = [metrics['volatility'] for metrics in self.risk_metrics_history]
        
        summary = {
            'current_risk': current_risk,
            'historical_max_drawdown': max(drawdowns) if drawdowns else 0,
            'historical_avg_volatility': np.mean(volatilities) if volatilities else 0,
            'active_alerts': len([a for a in self.alert_history[-24:] if a['level'] == 'HIGH']),
            'total_alerts_24h': len(self.alert_history[-24:]),
            'risk_score': self._calculate_risk_score(current_risk)
        }
        
        return summary
    
    def _calculate_risk_score(self, risk_metrics: Dict[str, Any]) -> float:
        """Calculate overall risk score (0-100, higher is riskier)"""
        score = 0
        
        # Drawdown contribution (0-40 points)
        drawdown_score = min(40, (risk_metrics['drawdown'] / self.max_drawdown_limit) * 40)
        score += drawdown_score
        
        # Volatility contribution (0-30 points)
        volatility_score = min(30, (risk_metrics['volatility'] / self.volatility_limit) * 30)
        score += volatility_score
        
        # Position concentration contribution (0-20 points)
        concentration_score = min(20, (risk_metrics['max_position_concentration'] / self.max_position_limit) * 20)
        score += concentration_score
        
        # VaR contribution (0-10 points)
        var_ratio = risk_metrics['var_95'] / risk_metrics['portfolio_value']
        var_score = min(10, (var_ratio / 0.05) * 10)  # Compare to 5% benchmark
        score += var_score
        
        return min(100, score)
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent risk alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert['timestamp'] >= cutoff_time
        ]
    
    def should_stop_trading(self) -> bool:
        """Determine if trading should be stopped due to excessive risk"""
        if not self.risk_metrics_history:
            return False
        
        current_risk = self.risk_metrics_history[-1]
        
        # Stop trading if multiple high-level breaches
        high_breaches = len(current_risk['risk_breaches'])
        
        # Stop trading if drawdown exceeds emergency limit
        emergency_drawdown = self.max_drawdown_limit * 1.5  # 50% beyond normal limit
        
        if current_risk['drawdown'] > emergency_drawdown:
            logger.critical(f"EMERGENCY: Drawdown {current_risk['drawdown']:.2%} exceeds emergency limit")
            return True
        
        if high_breaches >= 3:
            logger.critical(f"EMERGENCY: {high_breaches} risk breaches detected")
            return True
        
        return False
