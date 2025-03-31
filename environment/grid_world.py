import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

class MultiAgentGridWorld(gym.Env):
    """
    A multi-agent grid world environment where agents can either compete or cooperate.
    
    Environment features:
    - Multiple agents navigate a grid
    - Resources scattered around that agents can collect
    - Agents can choose to share resources (cooperate) or steal from others (compete)
    - Rewards based on resource collection and agent interactions
    """
    
    def __init__(self, grid_size=10, num_agents=3, num_resources=15, max_steps=100, coop_factor=0.3):
        super(MultiAgentGridWorld, self).__init__()
        
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_resources = num_resources
        self.max_steps = max_steps
        self.coop_factor = coop_factor  # Determines reward for cooperation
        
        # Define action and observation spaces
        # Actions: 0=up, 1=right, 2=down, 3=left, 4=share, 5=steal
        self.action_space = [Discrete(6) for _ in range(num_agents)]
        
        # Observation: agent sees a 5x5 grid around itself (25 cells) + positions of other agents (2*num_agents)
        # + own resource count (1) + other agents' resource counts (num_agents-1)
        obs_size = 25 + 2*num_agents + num_agents
        self.observation_space = [Box(low=0, high=1, shape=(obs_size,), dtype=np.float32) 
                                 for _ in range(num_agents)]
        
        # Initialize grid, agent positions, and resources
        self.grid = np.zeros((grid_size, grid_size))
        self.agent_positions = []
        self.agent_resources = np.zeros(num_agents)
        self.resource_positions = []
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        if seed is not None:
            np.random.seed(seed)
            
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.agent_positions = []
        self.agent_resources = np.zeros(self.num_agents)
        self.resource_positions = []
        self.current_step = 0
        
        # Place agents randomly on the grid
        positions = set()
        for _ in range(self.num_agents):
            while True:
                pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                if pos not in positions:
                    self.agent_positions.append(pos)
                    positions.add(pos)
                    self.grid[pos] = 1  # Mark agent position on grid
                    break
        
        # Place resources randomly on the grid
        for _ in range(self.num_resources):
            while True:
                pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                if pos not in positions:
                    self.resource_positions.append(pos)
                    positions.add(pos)
                    self.grid[pos] = 2  # Mark resource position on grid
                    break
        
        observations = self._get_observations()
        return observations, {}
    
    def step(self, actions):
        """
        Process one step in the environment.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            observations, rewards, dones, truncated, info
        """
        self.current_step += 1
        rewards = np.zeros(self.num_agents)
        
        # Process movement actions first
        new_positions = []
        for i, action in enumerate(actions):
            pos = self.agent_positions[i]
            
            # Handle movement (0=up, 1=right, 2=down, 3=left)
            if action < 4:
                new_pos = list(pos)
                if action == 0 and pos[0] > 0:
                    new_pos[0] -= 1
                elif action == 1 and pos[1] < self.grid_size - 1:
                    new_pos[1] += 1
                elif action == 2 and pos[0] < self.grid_size - 1:
                    new_pos[0] += 1
                elif action == 3 and pos[1] > 0:
                    new_pos[1] -= 1
                
                new_positions.append(tuple(new_pos))
            else:
                new_positions.append(pos)
        
        # Update grid with new positions
        self.grid = np.zeros((self.grid_size, self.grid_size))
        for pos in self.resource_positions:
            self.grid[pos] = 2
            
        # Check for resource collection
        for i, pos in enumerate(new_positions):
            for j, res_pos in enumerate(self.resource_positions):
                if pos == res_pos:
                    self.agent_resources[i] += 1
                    rewards[i] += 1
                    self.resource_positions.pop(j)
                    break
        
        # Process social actions (4=share, 5=steal)
        for i, action in enumerate(actions):
            if action == 4:  # Share
                # Find nearby agents
                nearby_agents = []
                for j, pos in enumerate(new_positions):
                    if i != j and self._is_nearby(new_positions[i], pos):
                        nearby_agents.append(j)
                
                if nearby_agents and self.agent_resources[i] > 0:
                    share_amount = min(1, self.agent_resources[i])
                    self.agent_resources[i] -= share_amount
                    
                    # Distribute evenly among nearby agents
                    for j in nearby_agents:
                        self.agent_resources[j] += share_amount / len(nearby_agents)
                        rewards[j] += share_amount / len(nearby_agents)
                    
                    # Cooperative bonus
                    coop_reward = self.coop_factor * share_amount
                    rewards[i] += coop_reward
            
            elif action == 5:  # Steal
                # Find nearby agents
                for j, pos in enumerate(new_positions):
                    if i != j and self._is_nearby(new_positions[i], pos):
                        if self.agent_resources[j] > 0:
                            steal_amount = min(0.5, self.agent_resources[j])
                            self.agent_resources[j] -= steal_amount
                            self.agent_resources[i] += steal_amount
                            rewards[i] += steal_amount
                            rewards[j] -= steal_amount * 2  # Penalty for being stolen from
        
        # Update agent positions
        self.agent_positions = new_positions
        for pos in self.agent_positions:
            self.grid[pos] = 1
        
        # Check if episode is done
        dones = [False] * self.num_agents
        truncated = [False] * self.num_agents
        
        if not self.resource_positions or self.current_step >= self.max_steps:
            dones = [True] * self.num_agents
            truncated = [True] * self.num_agents
        
        observations = self._get_observations()
        
        info = {
            'agent_resources': self.agent_resources,
            'remaining_resources': len(self.resource_positions)
        }
        
        return observations, rewards, dones, truncated, info
    
    def _get_observations(self):
        """Return observations for all agents."""
        observations = []
        
        for i, pos in enumerate(self.agent_positions):
            # Get 5x5 grid around agent
            local_view = np.zeros((5, 5))
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    x, y = pos[0] + dx, pos[1] + dy
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        local_view[dx+2, dy+2] = self.grid[x, y]
            
            # Flatten grid view
            grid_obs = local_view.flatten()
            
            # Add positions of all agents relative to this agent
            agent_pos_obs = []
            for other_pos in self.agent_positions:
                rel_x = (other_pos[0] - pos[0]) / self.grid_size
                rel_y = (other_pos[1] - pos[1]) / self.grid_size
                agent_pos_obs.extend([rel_x, rel_y])
            
            # Add resource information
            resource_obs = list(self.agent_resources / max(1, self.num_resources))
            
            # Combine all observations
            full_obs = np.concatenate([grid_obs, agent_pos_obs, resource_obs])
            observations.append(full_obs)
        
        return observations
    
    def _is_nearby(self, pos1, pos2):
        """Check if two positions are adjacent to each other."""
        return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1
    
    def render(self):
        """Render the environment."""
        grid_display = np.zeros((self.grid_size, self.grid_size, 3))
        
        # Draw resources
        for pos in self.resource_positions:
            grid_display[pos] = [0.0, 1.0, 0.0]  # Green for resources
        
        # Draw agents with different colors
        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
        ]
        
        for i, pos in enumerate(self.agent_positions):
            grid_display[pos] = colors[i % len(colors)]
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_display)
        plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        plt.xticks(np.arange(-.5, self.grid_size, 1), [])
        plt.yticks(np.arange(-.5, self.grid_size, 1), [])
        
        # Display resource counts
        for i, res in enumerate(self.agent_resources):
            plt.text(0, i, f"Agent {i} resources: {res:.1f}", fontsize=12)
        
        plt.show()