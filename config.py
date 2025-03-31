"""
Configuration settings for the multi-agent reinforcement learning project.
"""

# Random seed for reproducibility
RANDOM_SEED = 42

# Environment settings
ENV_CONFIG = {
    'grid_size': 10,
    'num_agents': 3,
    'num_resources': 15,
    'max_steps': 100,
    'coop_factor': 0.3,  # Reward bonus for cooperation
}

# Agent settings
AGENT_CONFIG = {
    'hidden_size': 128,
    'lr': 0.001,
    'gamma': 0.99,
    'epsilon_clip': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'update_steps': 4,
    'batch_size': 32,
}

# Training settings
TRAIN_CONFIG = {
    'max_episodes': 500,
    'update_interval': 128,
    'log_interval': 10,
    'save_interval': 100,
    'eval_interval': 20,
    'save_dir': 'saved_models',
}

# Evaluation settings
EVAL_CONFIG = {
    'num_episodes': 10,
    'render': True,
    'save_video': True,
    'video_dir': 'videos',
}

# Visualization settings
VIZ_CONFIG = {
    'fig_size': (12, 8),
    'save_dir': 'figures',
}