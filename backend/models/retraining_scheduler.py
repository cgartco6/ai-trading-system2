import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from .model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrainingScheduler:
    """Schedules and manages model retraining"""
    
    def __init__(self, model_manager: ModelManager, retrain_interval_hours: int = 24):
        self.model_manager = model_manager
        self.retrain_interval = retrain_interval_hours
        self.is_running = False
        self.scheduler_thread = None
        self.retraining_history: List[Dict[str, Any]] = []
        
    def start(self):
        """Start the retraining scheduler"""
        if self.is_running:
            logger.warning("Retraining scheduler is already running")
            return
        
        self.is_running = True
        
        # Schedule retraining
        schedule.every(self.retrain_interval).hours.do(self._retrain_all_models)
        
        # Start scheduler in background thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"Retraining scheduler started with {self.retrain_interval} hour interval")
    
    def stop(self):
        """Stop the retraining scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Retraining scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduling loop"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _retrain_all_models(self):
        """Retrain all models"""
        logger.info("Starting scheduled model retraining...")
        
        try:
            # Symbols to retrain models for
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
            
            successful_retrains = 0
            for symbol in symbols:
                try:
                    success = self.model_manager.retrain_models(symbol)
                    if success:
                        successful_retrains += 1
                        logger.info(f"Successfully retrained models for {symbol}")
                    else:
                        logger.warning(f"Failed to retrain models for {symbol}")
                    
                    # Add delay to avoid rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error retraining models for {symbol}: {e}")
            
            # Record retraining history
            retraining_record = {
                'timestamp': datetime.now(),
                'successful_retrains': successful_retrains,
                'total_symbols': len(symbols),
                'status': 'success' if successful_retrains > 0 else 'partial_failure'
            }
            
            self.retraining_history.append(retraining_record)
            
            # Keep only last 100 records
            if len(self.retraining_history) > 100:
                self.retraining_history = self.retraining_history[-100:]
            
            logger.info(f"Model retraining completed: {successful_retrains}/{len(symbols)} successful")
            
        except Exception as e:
            logger.error(f"Error in model retraining: {e}")
            
            # Record failure
            self.retraining_history.append({
                'timestamp': datetime.now(),
                'successful_retrains': 0,
                'total_symbols': 0,
                'status': 'failure',
                'error': str(e)
            })
    
    def manual_retrain(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Manually trigger model retraining"""
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        logger.info(f"Starting manual retraining for symbols: {symbols}")
        
        results = {}
        for symbol in symbols:
            try:
                success = self.model_manager.retrain_models(symbol)
                results[symbol] = {
                    'status': 'success' if success else 'failed',
                    'timestamp': datetime.now()
                }
                
                if success:
                    logger.info(f"Manual retraining successful for {symbol}")
                else:
                    logger.warning(f"Manual retraining failed for {symbol}")
                
            except Exception as e:
                results[symbol] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now()
                }
                logger.error(f"Error in manual retraining for {symbol}: {e}")
        
        return results
    
    def get_retraining_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get retraining history for the specified number of days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            record for record in self.retraining_history
            if record['timestamp'] >= cutoff_date
        ]
    
    def get_retraining_stats(self) -> Dict[str, Any]:
        """Get retraining statistics"""
        if not self.retraining_history:
            return {}
        
        recent_history = self.get_retraining_history(days=30)
        
        if not recent_history:
            return {}
        
        total_retrains = len(recent_history)
        successful_retrains = sum(1 for r in recent_history if r['status'] == 'success')
        failed_retrains = sum(1 for r in recent_history if r['status'] == 'failure')
        
        success_rate = successful_retrains / total_retrains if total_retrains > 0 else 0
        
        return {
            'total_retrains_30d': total_retrains,
            'successful_retrains_30d': successful_retrains,
            'failed_retrains_30d': failed_retrains,
            'success_rate_30d': success_rate,
            'last_retraining': self.retraining_history[-1] if self.retraining_history else None
        }
    
    def update_retrain_interval(self, new_interval_hours: int):
        """Update the retraining interval"""
        self.retrain_interval = new_interval_hours
        
        # Clear existing schedule
        schedule.clear()
        
        # Set new schedule
        schedule.every(self.retrain_interval).hours.do(self._retrain_all_models)
        
        logger.info(f"Retraining interval updated to {new_interval_hours} hours")
