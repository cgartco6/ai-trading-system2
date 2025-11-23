from typing import Dict, Type, Any, List
from abc import ABC, abstractmethod
import inspect

class AgentFactory:
    """Factory for creating and managing AI agents"""
    
    def __init__(self):
        self._agents = {}
        self._agent_types = {}
        
    def register_agent_type(self, agent_type: str, agent_class: Type):
        """Register a new type of agent"""
        self._agent_types[agent_type] = agent_class
        
    def create_agent(self, agent_type: str, config: Dict[str, Any], **kwargs) -> Any:
        """Create a new agent instance"""
        if agent_type not in self._agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        agent_class = self._agent_types[agent_type]
        agent_id = f"{agent_type}_{len(self._agents) + 1}"
        
        # Create agent instance
        agent = agent_class(**config, **kwargs)
        self._agents[agent_id] = agent
        
        return agent_id, agent
    
    def create_agent_swarm(self, swarm_config: List[Dict]) -> List[str]:
        """Create a swarm of coordinated agents"""
        swarm_agents = []
        
        for agent_config in swarm_config:
            agent_type = agent_config['type']
            config = agent_config.get('config', {})
            agent_id, agent = self.create_agent(agent_type, config)
            swarm_agents.append(agent_id)
            
        return swarm_agents
    
    def get_agent(self, agent_id: str) -> Any:
        """Retrieve an agent by ID"""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """List all created agents"""
        return list(self._agents.keys())

class SpecializedAgentFactory(AgentFactory):
    """Factory for creating specialized trading agents"""
    
    def __init__(self):
        super().__init__()
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize with default trading agent types"""
        from src.agents.trading_agent import TradingAgent
        from src.agents.research_agent import ResearchAgent
        from src.agents.risk_agent import RiskAgent
        
        self.register_agent_type("trading", TradingAgent)
        self.register_agent_type("research", ResearchAgent)
        self.register_agent_type("risk", RiskAgent)
    
    def create_trading_team(self, market_conditions: Dict) -> Dict[str, str]:
        """Create a coordinated team of agents for specific market conditions"""
        team = {}
        
        # Market analysis agent
        analysis_agent_id, _ = self.create_agent(
            "research", 
            {"specialization": "market_analysis"}
        )
        team["analyst"] = analysis_agent_id
        
        # Primary trading agent
        trading_agent_id, _ = self.create_agent(
            "trading",
            {"strategy": "adaptive", "risk_tolerance": market_conditions.get('volatility', 'medium')}
        )
        team["trader"] = trading_agent_id
        
        # Risk management agent
        risk_agent_id, _ = self.create_agent(
            "risk",
            {"max_drawdown": 0.02, "position_sizing": "dynamic"}
        )
        team["risk_manager"] = risk_agent_id
        
        return team
