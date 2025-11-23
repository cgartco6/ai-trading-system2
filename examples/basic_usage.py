#!/usr/bin/env python3
"""
Basic usage example for the AI Trading System
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.agent_factory import SpecializedAgentFactory
from signals.signal_generator import AISignalGenerator
from synthetic_intelligence.data_synthesizer import MarketDataSynthesizer
import pandas as pd
import numpy as np

def main():
    print("=== AI Trading System - Basic Usage Example ===")
    
    # 1. Initialize agent factory
    print("\n1. Initializing Agent Factory...")
    factory = SpecializedAgentFactory()
    
    # 2. Create a trading team
    print("\n2. Creating Trading Team...")
    market_conditions = {
        'volatility': 'high',
        'trend': 'bullish',
        'volume': 'above_average'
    }
    
    team = factory.create_trading_team(market_conditions)
    print(f"Created team: {list(team.keys())}")
    
    # 3. Generate sample data
    print("\n3. Generating Sample Data...")
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 2, 100),
        'high': np.random.normal(102, 2, 100),
        'low': np.random.normal(98, 2, 100),
        'close': np.random.normal(101, 2, 100),
        'volume': np.random.normal(1000000, 100000, 100)
    }, index=dates)
    
    # 4. Initialize signal generator
    print("\n4. Initializing Signal Generator...")
    signal_config = {
        'technical_weights': {
            'rsi': 0.3,
            'macd': 0.4,
            'bollinger': 0.3,
            'total': 0.4,
            'ml': 0.4,
            'sentiment': 0.2
        },
        'thresholds': {
            'buy': 0.3,
            'sell': -0.3
        }
    }
    
    signal_generator = AISignalGenerator(signal_config)
    
    # 5. Generate signals
    print("\n5. Generating Trading Signals...")
    news_samples = [
        "Company reports strong earnings growth",
        "Market shows bullish momentum",
        "Economic indicators positive"
    ]
    
    signals = signal_generator.generate_signals(sample_data, news_samples)
    print(f"Generated {len(signals)} signals")
    print(f"Signal distribution:\n{signals['action'].value_counts()}")
    
    # 6. Initialize data synthesizer
    print("\n6. Initializing Data Synthesizer...")
    synthesizer = MarketDataSynthesizer(seq_len=30, feature_dim=5)
    synthesizer.build_models()
    
    # Generate synthetic data for stress testing
    synthetic_data = synthesizer.generate_stress_scenarios(
        sample_data, scenario_type="crash"
    )
    print(f"Generated synthetic data with {len(synthetic_data)} points")
    
    print("\n=== Example Completed Successfully ===")

if __name__ == "__main__":
    main()
