import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

def plot_training_history(rewards_history, resource_history, save_dir='figures'):
    """
    Plot the training history metrics.
    
    Args:
        rewards_history: List of rewards for each agent across episodes
        resource_history: List of resources collected by each agent across episodes
        save_dir: Directory to save the figures
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_agents = len(rewards_history)
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    for i in range(num_agents):
        plt.plot(rewards_history[i], label=f"Agent {i}")
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "rewards.png"))
    plt.close()
    
    # Plot resources
    plt.figure(figsize=(10, 6))
    for i in range(num_agents):
        plt.plot(resource_history[i], label=f"Agent {i}")
    plt.title("Resources Collected per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Resources")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "resources.png"))
    plt.close()
    
    # Plot learning curves (moving average)
    window_size = min(10, len(rewards_history[0]))
    if len(rewards_history[0]) > window_size:
        plt.figure(figsize=(10, 6))
        for i in range(num_agents):
            smoothed_rewards = np.convolve(rewards_history[i], 
                                        np.ones(window_size)/window_size, 
                                        mode='valid')
            plt.plot(smoothed_rewards, label=f"Agent {i}")
        plt.title(f"Smoothed Rewards (Window={window_size})")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "learning_curve.png"))
        plt.close()

def visualize_episode(env, agents, save_path=None, max_steps=100):
    """
    Run and visualize a full episode with the trained agents.
    
    Args:
        env: Environment
        agents: List of trained agents
        save_path: Path to save the animation (optional)
        max_steps: Maximum steps per episode
    """
    observations, _ = env.reset()
    done = False
    step = 0
    
    # Store episode data for visualization
    episode_data = []
    episode_data.append({
        'grid': env.grid.copy(),
        'agent_positions': env.agent_positions.copy(),
        'resource_positions': env.resource_positions.copy(),
        'agent_resources': env.agent_resources.copy()
    })
    
    while not done and step < max_steps:
        actions = []
        for i, agent in enumerate(agents):
            action = agent.select_action(observations[i], deterministic=True)
            actions.append(action)
        
        next_observations, rewards, dones, truncated, info = env.step(actions)
        
        # Store step data
        episode_data.append({
            'grid': env.grid.copy(),
            'agent_positions': env.agent_positions.copy(),
            'resource_positions': env.resource_positions.copy(),
            'agent_resources': env.agent_resources.copy(),
            'actions': actions,
            'rewards': rewards
        })
        
        observations = next_observations
        step += 1
        
        if all(dones):
            done = True
    
    # Visualize the episode as an animation
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def init():
        ax.clear()
        return []
    
    def animate(i):
        ax.clear()
        data = episode_data[i]
        
        # Create grid
        grid_size = env.grid_size
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        
        # Draw grid lines
        for x in range(grid_size):
            ax.axvline(x - 0.5, color='black', linewidth=1)
        for y in range(grid_size):
            ax.axhline(y - 0.5, color='black', linewidth=1)
            
        # Draw resources
        for pos in data['resource_positions']:
            rect = patches.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, 
                                    linewidth=1, edgecolor='none', facecolor='green', alpha=0.5)
            ax.add_patch(rect)
            
        # Draw agents
        colors = ['red', 'blue', 'yellow', 'magenta', 'cyan']
        for j, pos in enumerate(data['agent_positions']):
            circle = plt.Circle((pos[1], pos[0]), 0.4, color=colors[j % len(colors)])
            ax.add_patch(circle)
            ax.text(pos[1], pos[0], str(j), ha='center', va='center', color='white')
            
        # Add step information
        if i > 0:
            actions_map = {0: '↑', 1: '→', 2: '↓', 3: '←', 4: 'Share', 5: 'Steal'}
            action_text = [f"Agent {j}: {actions_map[act]}" for j, act in enumerate(data['actions'])]
            reward_text = [f"Reward {j}: {r:.2f}" for j, r in enumerate(data['rewards'])]
            
            ax.set_title(f"Step {i}\n" + "\n".join(action_text) + "\n" + "\n".join(reward_text))
        else:
            ax.set_title("Initial State")
            
        # Add resource counts
        for j, res in enumerate(data['agent_resources']):
            ax.text(grid_size * 0.8, grid_size - 1 - j, f"Agent {j}: {res:.1f} resources", 
                   fontsize=10, bbox=dict(facecolor=colors[j % len(colors)], alpha=0.3))
                   
        return []
    
    ani = FuncAnimation(fig, animate, frames=len(episode_data), init_func=init, blit=True, interval=500)
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=2)
    
    plt.tight_layout()
    plt.show()
    
    return episode_data