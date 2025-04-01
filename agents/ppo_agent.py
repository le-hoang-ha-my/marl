import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from agents.networks import ActorCriticNetwork

class PPOAgent:
    """
    Agent implementation using Proximal Policy Optimization (PPO) algorithm.
    """
    def __init__(self, state_size, action_size, hidden_size=128, lr=0.001, gamma=0.99, 
                 epsilon_clip=0.2, value_coef=0.5, entropy_coef=0.01, agent_id=0,
                 update_steps=4, batch_size=32):
        """
        Initialize a PPO agent.
        
        Args:
            state_size: Dimension of the observation space
            action_size: Dimension of the action space
            hidden_size: Number of neurons in hidden layers
            lr: Learning rate
            gamma: Discount factor
            epsilon_clip: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            agent_id: Identifier for the agent
            update_steps: Number of epochs to update policy
            batch_size: Mini-batch size for updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.agent_id = agent_id
        self.update_steps = update_steps
        self.batch_size = batch_size
        
        # Policy network
        self.policy = ActorCriticNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Old policy for PPO update
        self.old_policy = ActorCriticNetwork(state_size, action_size, hidden_size)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Memory for storing trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
    def select_action(self, state, deterministic=False):
        """
        Select an action based on the current policy.
        
        Args:
            state: Current state
            deterministic: Whether to select actions deterministically
            
        Returns:
            action: Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = self.old_policy(state_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs).item()
            else:
                dist = Categorical(action_probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action))
                self.log_probs.append(log_prob.item())
                
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the agent's memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def update(self):
        """
        Update the policy using the PPO algorithm.
        
        Returns:
            loss: Average loss value during the update
        """
        # If not enough data, skip update
        if len(self.states) < self.batch_size:
            return 0.0
            
        # Convert lists to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        next_states = torch.FloatTensor(self.next_states)
        dones = torch.FloatTensor(self.dones)
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        value = 0
        
        with torch.no_grad():
            for i in reversed(range(len(self.states))):
                if dones[i]:
                    value = 0
                
                _, next_value = self.old_policy(next_states[i].unsqueeze(0))
                _, curr_value = self.old_policy(states[i].unsqueeze(0))
                
                # TD error as advantage
                advantage = rewards[i] + self.gamma * next_value * (1 - dones[i]) - curr_value
                value = rewards[i] + self.gamma * value * (1 - dones[i])
                
                returns.insert(0, value.item())
                advantages.insert(0, advantage.item())
                
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_loss = 0
        
        for _ in range(self.update_steps):
            # Generate random mini-batches
            indices = torch.randperm(len(states))[:self.batch_size]
            
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_advantages = advantages[indices]
            batch_returns = returns[indices]
            batch_old_log_probs = old_log_probs[indices]
            
            # Get current policy and values
            action_probs, values = self.policy(batch_states)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            # Compute PPO loss
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * batch_advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), batch_returns)
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
        return total_loss / self.update_steps

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)
        
    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))
        self.old_policy.load_state_dict(torch.load(filename))