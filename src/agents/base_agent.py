from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum
import uuid

class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    COMPLETED = "completed"

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.config = config
        self.status = AgentStatus.IDLE
        self.memory = {}
        self.capabilities = self._initialize_capabilities()
        
    @abstractmethod
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific task"""
        pass
    
    def _initialize_capabilities(self) -> List[str]:
        """Initialize agent capabilities"""
        return []
    
    def get_status(self) -> AgentStatus:
        """Get current agent status"""
        return self.status
    
    def update_memory(self, key: str, value: Any):
        """Update agent memory"""
        self.memory[key] = value
    
    def retrieve_memory(self, key: str) -> Optional[Any]:
        """Retrieve from agent memory"""
        return self.memory.get(key)
    
    def collaborate(self, other_agents: List['BaseAgent'], task: Dict) -> Dict:
        """Collaborate with other agents"""
        results = {}
        for agent in other_agents:
            if self._can_collaborate(agent):
                result = agent.execute_task(task)
                results[agent.agent_id] = result
        return results
    
    def _can_collaborate(self, other_agent: 'BaseAgent') -> bool:
        """Check if collaboration is possible"""
        return any(
            capability in other_agent.capabilities 
            for capability in self.config.get('required_capabilities', [])
        )

class AdaptiveAgent(BaseAgent):
    """Agent that can adapt its behavior based on environment"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.adaptation_history = []
        
    def adapt_strategy(self, performance_metrics: Dict[str, float]):
        """Adapt strategy based on performance"""
        self.adaptation_history.append(performance_metrics)
        
        # Simple adaptation logic (replace with more sophisticated RL)
        if performance_metrics.get('success_rate', 0) < 0.6:
            self.config['aggressiveness'] *= 0.9  # Become more conservative
        elif performance_metrics.get('success_rate', 0) > 0.8:
            self.config['aggressiveness'] *= 1.1  # Become more aggressive
            
    def transfer_learning(self, source_task: str, target_task: str):
        """Transfer learning from one task to another"""
        # Implement transfer learning logic
        pass
