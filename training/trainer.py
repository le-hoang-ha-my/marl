import numpy as np
import matplotlib.pyplot as plt
import os

class MultiAgentTrainer:
    """
    Framework for training multiple agents in a shared environment.
    """
    def __init__(self, env, agents, max_episodes=1000, max_steps=100, update_interval=128, 
                 log_interval=10, save_interval=100, eval_interval=20, save_dir='saved_models'):
        """
        Initialize the trainer.
        
        Args:
            env: Multi-agent environment
            agents: List of agent objects
            max_episodes: Maximum number of episodes to train
            max_steps: Maximum steps per episode
            update_interval: Frequency of policy updates (in steps)
            log_interval: Frequency of logging (in episodes)
            save_interval: Frequency of model saving (in episodes)
            eval_interval: Frequency of evaluation (in episodes)
            save_dir: Directory to save trained models
        """
        self.env = env
        self.agents = agents
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.update_interval = update_interval
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        
        self.num_agents = len(agents)
        self.step_counter = 0
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Metrics for logging
        self.episode_rewards = [[] for _ in range(self.num_agents)]
        self.resource_counts = [[] for _ in range(self.num_agents)]
        self.cooperation_counts = [0] * self.num_agents
        self.theft_counts = [0] * self.num_agents
        
    def train(self):
        """
        Main training loop.
        
        Returns:
            episode_rewards: List of rewards for each agent across episodes
            resource_counts: List of resources collected by each agent across episodes
        """
        for episode in range(1, self.max_episodes + 1):
            observations, _ = self.env.reset()
            episode_rewards = np.zeros(self.num_agents)
            
            for step in range(self.max_steps):
                actions = []
                
                # Select actions for each agent
                for i, agent in enumerate(self.agents):
                    action = agent.select_action(observations[i])
                    actions.append(action)
                    
                    # Track social actions
                    if action == 4:  # Share
                        self.cooperation_counts[i] += 1
                    elif action == 5:  # Steal
                        self.theft_counts[i] += 1
                
                # Execute actions in the environment
                next_observations, rewards, dones, truncated, info = self.env.step(actions)
                
                # Store transitions for each agent
                for i, agent in enumerate(self.agents):
                    agent.store_transition(observations[i], actions[i], rewards[i], 
                                          next_observations[i], dones[i])
                    episode_rewards[i] += rewards[i]
                
                observations = next_observations
                self.step_counter += 1
                
                # Check if it's time to update
                if self.step_counter % self.update_interval == 0:
                    for i, agent in enumerate(self.agents):
                        agent.update()
                
                # Check if episode is done
                if all(dones):
                    break
            
            for i in range(self.num_agents):
                self.episode_rewards[i].append(episode_rewards[i])
                self.resource_counts[i].append(self.env.agent_resources[i])
            
            if episode % self.log_interval == 0:
                avg_rewards = [np.mean(rewards[-self.log_interval:]) for rewards in self.episode_rewards]
                avg_resources = [np.mean(resources[-self.log_interval:]) for resources in self.resource_counts]
                
                print(f"Episode {episode}/{self.max_episodes}")
                print(f"Average Rewards: {avg_rewards}")
                print(f"Average Resources: {avg_resources}")
                print(f"Cooperation Actions: {self.cooperation_counts}")
                print(f"Theft Actions: {self.theft_counts}")
                print("-" * 50)
            
            # Save models
            if episode % self.save_interval == 0:
                for i, agent in enumerate(self.agents):
                    save_path = os.path.join(self.save_dir, f"agent_{i}_episode_{episode}.pth")
                    agent.save(save_path)
            
            # Evaluation
            if episode % self.eval_interval == 0:
                self.evaluate(5, render=False)
                
        return self.episode_rewards, self.resource_counts
    
    def evaluate(self, num_episodes=10, render=True):
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            avg_rewards: Average rewards for each agent
            avg_resources: Average resources collected by each agent
        """
        total_rewards = np.zeros(self.num_agents)
        total_resources = np.zeros(self.num_agents)
        
        for episode in range(num_episodes):
            observations, _ = self.env.reset()
            episode_rewards = np.zeros(self.num_agents)
            done = False
            
            while not done:
                actions = []
                
                # Select deterministic actions for evaluation
                for i, agent in enumerate(self.agents):
                    action = agent.select_action(observations[i], deterministic=True)
                    actions.append(action)
                
                # Execute actions in the environment
                next_observations, rewards, dones, truncated, info = self.env.step(actions)
                
                for i in range(self.num_agents):
                    episode_rewards[i] += rewards[i]
                
                observations = next_observations
                
                if render and episode == 0:
                    self.env.render()
                
                if all(dones):
                    done = True
            
            total_rewards += episode_rewards
            total_resources += self.env.agent_resources
        
        avg_rewards = total_rewards / num_episodes
        avg_resources = total_resources / num_episodes
        
        print("\nEVALUATION")
        print(f"Average Rewards: {avg_rewards}")
        print(f"Average Resources: {avg_resources}")
        print("---\n")
        
        return avg_rewards, avg_resources
    
    def plot_metrics(self, save_fig=True):
        """
        Plot training metrics.
        
        Args:
            save_fig: Whether to save the figure
        """
        # Plot rewards
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        for i in range(self.num_agents):
            plt.plot(self.episode_rewards[i], label=f"Agent {i}")
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        
        # Plot smoothed rewards
        plt.subplot(2, 2, 2)
        window_size = min(10, len(self.episode_rewards[0]))
        for i in range(self.num_agents):
            if len(self.episode_rewards[i]) > window_size:
                smoothed_rewards = np.convolve(self.episode_rewards[i], 
                                            np.ones(window_size)/window_size, 
                                            mode='valid')
                plt.plot(smoothed_rewards, label=f"Agent {i}")
        plt.title(f"Smoothed Rewards (Window={window_size})")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        
        # Plot resources
        plt.subplot(2, 2, 3)
        for i in range(self.num_agents):
            plt.plot(self.resource_counts[i], label=f"Agent {i}")
        plt.title("Resources Collected")
        plt.xlabel("Episode")
        plt.ylabel("Resources")
        plt.legend()
        
        # Plot social actions
        plt.subplot(2, 2, 4)
        agents = range(self.num_agents)
        width = 0.35
        plt.bar(agents, self.cooperation_counts, width, label='Cooperation')
        plt.bar([p + width for p in agents], self.theft_counts, width, label='Theft')
        plt.title("Social Actions")
        plt.xlabel("Agent")
        plt.ylabel("Count")
        plt.xticks([p + width/2 for p in agents], [f"Agent {i}" for i in agents])
        plt.legend()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.save_dir, "training_metrics.png"))
            
        plt.show()