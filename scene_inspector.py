import time
import os
import datetime
import numpy as np
import tensorflow as tf
from src.dqn_agent import DQNAgent
from src.carla_environment import CarlaEnv
from config import get_args
from tqdm import tqdm


def train_dqn_agent(config):
    """
    Function to train a DQN agent in the CARLA environment.

    Args:
    - config (dict): Configuration parameters for training.

    Returns:
    - None
    """

    # Extract knob value from configuration
    knob_value = config['knob_value']
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + f"-knob-{knob_value}")
    model_dir = os.path.join("models", f"{knob_value}")

    # Create directories for logging and model saving
    # os.makedirs(log_dir, exist_ok=True)
    # os.makedirs(model_dir, exist_ok=True)

    # Initialize CARLA environment and extract state and action dimensions
    env = CarlaEnv(**config)
    state_shape = env.observation_space.shape
    action_dim = env.action_space.n

    # Configure DQN agent
    agent_config = {
        'input_shape': state_shape,
        'num_actions': action_dim,
        'replay_buffer_capacity': config['replay_buffer_capacity'],
        'batch_size': config['batch_size'],
        'gamma': config['gamma'],
        'lr': config['lr'],
        'epsilon_start': config['epsilon_start'],
        'epsilon_end': config['epsilon_end'],
        'epsilon_decay': config['epsilon_decay'],
        'target_update': config['target_update'],
        'save_model_freq': config['save_model_freq'],
        'knob_value': knob_value,
        'log_dir': log_dir
    }
    agent = DQNAgent(**agent_config)



    # Get current world and enable synchronous mode
    world = env.client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True  # You control time
    settings.fixed_delta_seconds = 0.05  # Optional: controls time step per tick
    world.apply_settings(settings)

    # Spawn your scene
    while True:
        print("Spawning new scene...")
        env.reset(evaluate=0)

        # Tick once to render the first frame
        world.tick()

        input("Scene ready. Press Enter to reset...")

        env.destroy_actors()





if __name__ == "__main__":
    config = get_args()  # Get configuration parameters
    train_dqn_agent(config)  # Train the DQN agent
