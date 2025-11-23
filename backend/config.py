import os
from dataclasses import dataclass
from typing import Dict, Any, List
import yaml

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_system"
    username: str = "postgres"
    password: str = "password"

@dataclass
class APIConfig:
    alpha_vantage_key: str = "demo"
    finnhub_key: str = ""
    polygon_key: str = ""
    newsapi_key: str = ""
    rate_limit_per_minute: int = 5

@dataclass
class TradingConfig:
    initial_capital: float = 10000.0
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.02   # 2% max daily loss
    risk_free_rate: float = 0.02   # 2% risk free rate
    commission: float = 0.001      # 0.1% commission

@dataclass
class ModelConfig:
    retrain_interval_hours: int = 24
    model_save_path: str = "models/"
    ensemble_weights: Dict[str, float] = None

@dataclass
class AgentConfig:
    memory_size: int = 1000
    learning_rate: float = 0.001
    exploration_rate: float = 0.1

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.trading = TradingConfig()
        self.model = ModelConfig()
        self.agent = AgentConfig()
        
        # Load from YAML if exists
        self._load_from_yaml()
        
    def _load_from_yaml(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self._update_from_dict(config_data)
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        # Update database config
        if 'database' in config_dict:
            db_config = config_dict['database']
            self.database = DatabaseConfig(**db_config)
        
        # Update API config
        if 'api' in config_dict:
            api_config = config_dict['api']
            self.api = APIConfig(**api_config)
        
        # Update trading config
        if 'trading' in config_dict:
            trading_config = config_dict['trading']
            self.trading = TradingConfig(**trading_config)
        
        # Update model config
        if 'model' in config_dict:
            model_config = config_dict['model']
            self.model = ModelConfig(**model_config)
        
        # Update agent config
        if 'agent' in config_dict:
            agent_config = config_dict['agent']
            self.agent = AgentConfig(**agent_config)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'database': self.database.database,
                'username': self.database.username,
                'password': self.database.password
            },
            'api': {
                'alpha_vantage_key': self.api.alpha_vantage_key,
                'finnhub_key': self.api.finnhub_key,
                'polygon_key': self.api.polygon_key,
                'newsapi_key': self.api.newsapi_key,
                'rate_limit_per_minute': self.api.rate_limit_per_minute
            },
            'trading': {
                'initial_capital': self.trading.initial_capital,
                'max_position_size': self.trading.max_position_size,
                'max_daily_loss': self.trading.max_daily_loss,
                'risk_free_rate': self.trading.risk_free_rate,
                'commission': self.trading.commission
            },
            'model': {
                'retrain_interval_hours': self.model.retrain_interval_hours,
                'model_save_path': self.model.model_save_path,
                'ensemble_weights': self.model.ensemble_weights
            },
            'agent': {
                'memory_size': self.agent.memory_size,
                'learning_rate': self.agent.learning_rate,
                'exploration_rate': self.agent.exploration_rate
            }
        }
    
    def save(self, path: str = None):
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

# Global config instance
config = Config()
