from core.agent_factory import SpecializedAgentFactory
from signals.signal_generator import AISignalGenerator

# Create trading agents
factory = SpecializedAgentFactory()
team = factory.create_trading_team({
    'volatility': 'high',
    'trend': 'bullish'
})

# Generate trading signals
signal_generator = AISignalGenerator(config)
signals = signal_generator.generate_signals(market_data, news_feeds)
