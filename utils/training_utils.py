import os
import torch

def save_training_state(episode, trainer, save_dir):
    """Save training state to resume later"""
    state_path = os.path.join(save_dir, "training_state.pt")
    state = {
        'episode': episode,
        'episode_rewards': trainer.episode_rewards,
        'resource_counts': trainer.resource_counts,
        'cooperation_counts': trainer.cooperation_counts,
        'theft_counts': trainer.theft_counts,
        'step_counter': trainer.step_counter
    }
    torch.save(state, state_path)
    print(f"Training state saved at episode {episode}")

def load_training_state(save_dir):
    """Load training state to resume training"""
    state_path = os.path.join(save_dir, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        return state
    return None