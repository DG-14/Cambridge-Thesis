
from config import get_args
from train import train_dqn_agent_baseline,train_dqn_agent_fixed,train_dqn_agent_adaptive,train_dqn_agent_tscl




if __name__ == "__main__":
    
    config = get_args()  # Get configuration parameters

    if config['mode'] == "train":

    
        if config['curriculum_mode'] == 'baseline':
            train_dqn_agent_baseline(config)  # Train the DQN agent (baseline)
        elif config['curriculum_mode'] == 'fixed':
            train_dqn_agent_fixed(config)  # Train the DQN agent (fixed)
        elif config['curriculum_mode'] == 'adaptive':
            train_dqn_agent_adaptive(config) # Train the DQN agent (Teacher Student Learning)
        elif config['curriculum_mode'] == 'tscl':
            train_dqn_agent_tscl(config) # Train the DQN agent (Teacher Student Learning)

