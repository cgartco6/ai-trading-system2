import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional
import numpy as np

class DeepAgent(nn.Module):
    """Base deep reinforcement learning agent for trading"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DeepAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state: torch.Tensor) -> tuple:
        features = self.feature_net(state)
        value = self.value_net(features)
        policy = self.policy_net(features)
        return policy, value

class MultiTaskDeepAgent(DeepAgent):
    """Deep agent capable of handling multiple trading tasks"""
    
    def __init__(self, state_dim: int, action_dim: int, task_count: int, hidden_dim: int = 512):
        super().__init__(state_dim, action_dim, hidden_dim)
        self.task_count = task_count
        
        # Task-specific layers
        self.task_embeddings = nn.Embedding(task_count, 64)
        self.task_net = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
    def forward(self, state: torch.Tensor, task_id: torch.Tensor) -> tuple:
        features = self.feature_net(state)
        task_emb = self.task_embeddings(task_id)
        combined = torch.cat([features, task_emb], dim=-1)
        task_features = self.task_net(combined)
        
        value = self.value_net(task_features)
        policy = self.policy_net(task_features)
        return policy, value

class MetaLearningAgent:
    """Agent capable of meta-learning and rapid adaptation"""
    
    def __init__(self, base_agent: DeepAgent, learning_rate: float = 0.001):
        self.base_agent = base_agent
        self.optimizer = optim.Adam(base_agent.parameters(), lr=learning_rate)
        
    def meta_train(self, tasks: List[Any], meta_batch_size: int = 16):
        """Meta-training across multiple tasks"""
        for task_batch in self._create_task_batches(tasks, meta_batch_size):
            meta_gradients = []
            
            for task in task_batch:
                # Inner loop adaptation
                adapted_agent = self._fast_adapt(task)
                # Compute meta-gradient
                meta_gradients.append(self._compute_meta_gradient(adapted_agent, task))
            
            # Meta-update
            self._meta_update(meta_gradients)
    
    def _fast_adapt(self, task: Any) -> DeepAgent:
        """Fast adaptation to a new task"""
        adapted_agent = DeepAgent(
            self.base_agent.state_dim,
            self.base_agent.action_dim
        )
        adapted_agent.load_state_dict(self.base_agent.state_dict())
        
        # Few-shot learning on the new task
        for _ in range(5):  # 5 gradient steps
            loss = self._compute_task_loss(adapted_agent, task)
            grad = torch.autograd.grad(loss, adapted_agent.parameters())
            # Update adapted agent
            for param, g in zip(adapted_agent.parameters(), grad):
                param.data -= 0.01 * g
                
        return adapted_agent
