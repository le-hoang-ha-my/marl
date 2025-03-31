import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
    """
    Neural network architecture that combines actor (policy) and critic (value) functions.
    Uses shared layers for feature extraction and separate heads for policy and value outputs.
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor (policy) head
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic (value) head
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            policy: Action probability distribution
            value: Estimated state value
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor output (action probabilities)
        policy = F.softmax(self.actor(x), dim=-1)
        
        # Critic output (state value)
        value = self.critic(x)
        
        return policy, value
    
    def get_action(self, state, deterministic=False):
        """
        Select an action based on the current policy.
        
        Args:
            state: Current state
            deterministic: If True, select the action with highest probability
                          If False, sample from the probability distribution
                          
        Returns:
            action: Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        policy, _ = self.forward(state)
        
        if deterministic:
            action = torch.argmax(policy).item()
        else:
            m = Categorical(policy)
            action = m.sample().item()
            
        return action