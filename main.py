"""
Main script to run the multi-agent reinforcement learning project.
"""

import os
import torch
import numpy as np
import random
import argparse
from environment import MultiAgentGridWorld
from agents import PPOAgent
from training import MultiAgentTrainer
from utils import visualize_episode, load_training_state
from config import (
    RANDOM_SEED, ENV_CONFIG, AGENT_CONFIG, 
    TRAIN_CONFIG, EVAL_CONFIG, VIZ_CONFIG
)

def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set torch to deterministic mode if available
    if hasattr(torch, 'set_deterministic'):
        torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_environment():
    """Create the multi-agent environment."""
    return MultiAgentGridWorld(
        grid_size=ENV_CONFIG['grid_size'],
        num_agents=ENV_CONFIG['num_agents'],
        num_resources=ENV_CONFIG['num_resources'],
        max_steps=ENV_CONFIG['max_steps'],
        coop_factor=ENV_CONFIG['coop_factor']
    )

def create_agents(env):
    """Create agents for the environment."""
    agents = []
    for i in range(env.num_agents):
        state_size = env.observation_space[i].shape[0]
        action_size = env.action_space[i].n
        
        agent = PPOAgent(
            state_size=state_size, 
            action_size=action_size,
            hidden_size=AGENT_CONFIG['hidden_size'],
            lr=AGENT_CONFIG['lr'],
            gamma=AGENT_CONFIG['gamma'],
            epsilon_clip=AGENT_CONFIG['epsilon_clip'],
            value_coef=AGENT_CONFIG['value_coef'],
            entropy_coef=AGENT_CONFIG['entropy_coef'],
            agent_id=i,
            update_steps=AGENT_CONFIG['update_steps'],
            batch_size=AGENT_CONFIG['batch_size']
        )
        agents.append(agent)
    
    return agents

def train(env, agents, start_episode=1):
    """Train the agents in the environment."""
    trainer = MultiAgentTrainer(
        env=env,
        agents=agents,
        max_episodes=TRAIN_CONFIG['max_episodes'],
        max_steps=ENV_CONFIG['max_steps'],
        update_interval=TRAIN_CONFIG['update_interval'],
        log_interval=TRAIN_CONFIG['log_interval'],
        save_interval=TRAIN_CONFIG['save_interval'],
        eval_interval=TRAIN_CONFIG['eval_interval'],
        save_dir=TRAIN_CONFIG['save_dir']
    )
    
    if start_episode > 1:
        state = load_training_state(TRAIN_CONFIG['save_dir'])
        if state:
            trainer.episode_rewards = state['episode_rewards']
            trainer.resource_counts = state['resource_counts']
            trainer.cooperation_counts = state['cooperation_counts']
            trainer.theft_counts = state['theft_counts']
            trainer.step_counter = state['step_counter']
            print(f"Resumed training from episode {start_episode}")
    
    rewards_history, resource_history = trainer.train(start_episode)
    trainer.plot_metrics(save_fig=True)
    
    return trainer

def evaluate(env, agents, auto_close=True):
    """Evaluate the trained agents."""
    os.makedirs(EVAL_CONFIG['video_dir'], exist_ok=True)
    
    for episode in range(EVAL_CONFIG['num_episodes']):
        print(f"\nEvaluation Episode {episode+1}/{EVAL_CONFIG['num_episodes']}")
        
        save_path = None
        if EVAL_CONFIG['save_video']:
            save_path = os.path.join(EVAL_CONFIG['video_dir'], f"episode_{episode}.gif")
            
        episode_data = visualize_episode(
            env=env,
            agents=agents,
            save_path=save_path,
            max_steps=ENV_CONFIG['max_steps'],
            auto_close=auto_close
        )
        
        final_data = episode_data[-1]
        print("Final resources:", final_data['agent_resources'])
        print("Total steps:", len(episode_data) - 1)  # Subtract initial state

def main():
    parser = argparse.ArgumentParser(description='Multi-Agent Reinforcement Learning')
    parser.add_argument('--train', action='store_true', help='Train the agents')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the agents')
    parser.add_argument('--load_dir', type=str, default=None, help='Directory to load trained models')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes for training')
    parser.add_argument('--resume', action='store_true', help='Resume training from last saved state')
    parser.add_argument('--auto_close', action='store_true', help='Automatically close animations')
    
    args = parser.parse_args()
    
    set_random_seeds(RANDOM_SEED)
    
    env = create_environment()
    agents = create_agents(env)
    
    start_episode = 1
    if args.resume or args.load_dir:
        load_dir = args.load_dir if args.load_dir else TRAIN_CONFIG['save_dir']
        # Find the latest episode
        state = load_training_state(load_dir)
        if state:
            start_episode = state['episode'] + 1
            
        for i, agent in enumerate(agents):
            # Find the latest model for each agent
            model_files = [f for f in os.listdir(load_dir) if f.startswith(f"agent_{i}_")]
            if model_files:
                latest_model = sorted(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
                model_path = os.path.join(load_dir, latest_model)
                print(f"Loading agent {i} model from {model_path}")
                agent.load(model_path)
    
    # Update episodes if specified
    if args.episodes:
        TRAIN_CONFIG['max_episodes'] = args.episodes
    
    if args.train:
        print(f"Starting training from episode {start_episode}...")
        trainer = train(env, agents, start_episode)
        print("Training completed!")
    
    if args.evaluate:
        print("Starting evaluation...")
        evaluate(env, agents, auto_close=args.auto_close)
        print("Evaluation completed!")
    
    if not args.train and not args.evaluate:
        print(f"Starting training from episode {start_episode} and evaluation...")
        trainer = train(env, agents, start_episode)
        evaluate(env, agents)
        print("All tasks completed!")

if __name__ == "__main__":
    main()