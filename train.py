import time
import os
import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import json
import csv
import math
import jsonpickle
import random

from src.dqn_agent import DQNAgent
from src.carla_environment import CarlaEnv
from src.utilies import intersection_ID_to_int, safe_for_json
from src.uncertainty import build_cnn_model_with_dropout, calculate_total_uncertainty, calculate_nll, calculate_rmse, calculate_entropy


# Baseline RL
def train_dqn_agent_baseline(config):
    """
    Function to train a DQN agent in the CARLA environment.

    Args:
    - config (dict): Configuration parameters for training.

    Returns:
    - None
    """

    print("[TRAINING MODE] Baseline")

    # Extract knob value from configuration
    knob_value = config['knob_value']

    log_dir = os.path.join("training_logs", "baseline", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    model_dir = os.path.join("models", "baseline", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))

    # Create directories for logging and model saving
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save config arguements to file
    config_save_path = os.path.join(log_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Initialize CARLA environment and extract state and action dimensions
    env = CarlaEnv(log_dir,**config)

    max_difficulty_level = 5  # Assuming 6 levels: 0–5

    # defaults to max difficulty
    # difficulty = 5
    # env.set_difficulty(difficulty)
    
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
        'log_dir': log_dir,
        'use_GPU': config['use_GPU']
    }
    
    agent = DQNAgent(**agent_config)

    env.get_summary_writer_from_agent(agent.get_summary_writer())

    td_error = 0
    uncertainity_weight = config['uncertainity_weight']
    td_weight = config['td_weight']

    # # Save agent arguements to file
    # agent_config_save_path = os.path.join(log_dir, "agent_config.json")
    # with open(agent_config_save_path, 'w') as f:
    #     json.dump(agent_config, f, indent=4)

    # Training loop
    for e in tqdm(range(config['episodes']), desc="Training Episodes"): # Total training episodes
            
        try:
            difficulty = np.random.randint(0, 6)
            print("Difficulty: " + str(difficulty))
            env.set_difficulty(difficulty)
            state = env.reset(evaluate=0,episode=e)
            done = False
            total_reward = 0
            finished = False


            # Training steps
            while not finished:  
                action = agent.act(state)
                next_state, reward, done, episode_result_dict = env.step(action)

                raw_reward = reward  # preserve for logging

                # normalized_reward = np.clip(raw_reward, -1, 1)  # keep reward in safe range for DQN stability
                normalized_reward = np.tanh(raw_reward)

                # Clip reward to [-1, 1]
                # clipped_reward = min(1, max(-1, reward))
        
                # Optional: differentiate step vs. terminal for clarity
                # if done:
                    # clipped_reward = np.clip(reward, -1, 1)

                agent.store_experience(state, action, normalized_reward, next_state, done)
                state = next_state
                total_reward += normalized_reward

                if done:
                    try:
                        env.destroy_actors()
                    except:
                        pass
                    finished = True
                    break

                td_error = agent.train()

            if config['track_training']:
                train_util, train_util_path = env.get_train_util()
                train_util['td_error'] = td_error

                # get uncertainity
                unc_scalar = get_mc_dropout_uncertainty(agent.policy_net, state, K=config['sample_nets'])
                train_util['uncertainty'] = unc_scalar

                # # Penalty for bad episodes
                # if total_reward < -0.5:
                    # # print(f"Penalty applied: reward = {total_reward:.3f}")
                    # td_error *= 0.1
                    # unc_scalar *= 0.1
                
                # get utility

                utility = uncertainity_weight*unc_scalar + td_weight*td_error

                utility = gaussian_difficulty_prior(utility, difficulty, e, total_episodes=config['episodes'], sigma=2.0)

                train_util['utility'] = utility


                # save to json
                with open(train_util_path, 'a') as f:
                    json.dump(safe_for_json(train_util), f, indent=4) 

        except RuntimeError:
            print("[DEBUG] Episode Failed. Resetting and trying Again")

        agent.train_episode += 1

        
        if agent.train_episode % agent.save_checkpoint == 0:
            model_path = os.path.join(log_dir, f"model_checkpoint_{agent.train_episode}.h5")
            buffer_path = os.path.join(log_dir, f"replay_buffer_{agent.train_episode}.pkl")
            agent.save_model(model_path)
            agent.save_replay_buffer(buffer_path)


        agent.rewards.insert(0, total_reward)
        if len(agent.rewards) > 100:
            agent.rewards = agent.rewards[:-1]
        avg_rewards = np.mean(agent.rewards)
        
        print(f"Episode: {e + 1}/{config['episodes']}, Episode reward: {total_reward}, Average reward (100 Episodes): {avg_rewards}")

        collision_type = env.get_collision_summary()
        print("Collision type: ")
        print(collision_type)

        # Logging average reward
        with agent.summary_writer.as_default():
            tf.summary.scalar('Episode Reward', total_reward, step=e)
            tf.summary.scalar('Average Reward (100 Episodes)', avg_rewards, step=e)
            tf.summary.scalar('Episode to Training Index', e, step=agent.train_step)
            tf.summary.scalar('Episode to Simulation Step', e, step=env.simulation_step_count)
            tf.summary.scalar('Difficulty',difficulty,step=e)
            tf.summary.scalar('intersection_id', intersection_ID_to_int(getattr(env, 'location', 'unknown')), step=e)
            
            # if collision_type is not None:
                # tf.summary.scalar('Total Collisions', collision_type['total_collisions'], step=e)

            tf.summary.scalar("collision/total_collisions", collision_type['total_collisions'], step=e)
            tf.summary.scalar("collision/average_speed", collision_type['average_speed'], step=e)

            if bool(collision_type['actor_IDs']):

                for actor_id, count in collision_type['actor_IDs'].items():
                    tf.summary.scalar(f"collision/actor_IDs/{actor_id}", count, step=e)

                for actor_type, count in collision_type['actor_types'].items():
                    tf.summary.scalar(f"collision/actor_types/{actor_type}", count, step=e)

            if episode_result_dict['termination_type'] is not None:
                tf.summary.scalar('Termination Type', episode_result_dict['termination_type'], step=e)


        # Evaluation phase
        print(f"train step: {agent.train_step}")

        
        if (agent.train_step // config['evaluation_interval']) != agent.evaluation_checkpoint:
            print("\n******Evaluation*********\n")
            # evaluate_agent(agent, env, e,max_difficulty_level)
            _,_,utility_scores = evaluate_agent(agent, env, e,max_difficulty_level,log_dir,config['episodes'],td_weight, uncertainity_weight)
            print("\n******Training*********\n")

    # Save final model and replay buffer
    final_model_path = os.path.join(model_dir, 'final_model.h5')
    final_replay_buffer_path = os.path.join(model_dir, 'final_replay_buffer.pkl')
    agent.save_model(final_model_path)
    agent.save_replay_buffer(final_replay_buffer_path)

# Fixed CL
def train_dqn_agent_fixed(config):
    """
    Function to train a DQN agent in the CARLA environment.

    Args:
    - config (dict): Configuration parameters for training.

    Returns:
    - None
    """

    print("[TRAINING MODE] Fixed")

    # Extract knob value from configuration
    knob_value = config['knob_value']

    log_dir = os.path.join("training_logs", "fixed", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    model_dir = os.path.join("models", "fixed", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))

    # Create directories for logging and model saving
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save config arguements to file
    config_save_path = os.path.join(log_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Initialize CARLA environment and extract state and action dimensions
    env = CarlaEnv(log_dir,**config)
    # rl_wrapper = RL_wrapper(env)
    
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
        'log_dir': log_dir,
        'use_GPU': config['use_GPU']
    }
    
    agent = DQNAgent(**agent_config)

    env.get_summary_writer_from_agent(agent.get_summary_writer())

    episodes_per_difficulty = config.get('episodes_per_level', 1000)
    max_difficulty_level = 5  # Assuming 6 levels: 0–5

    # defaults to max difficulty
    difficulty = 0
    env.set_difficulty(difficulty)

    td_error = 0
    uncertainity_weight = config['uncertainity_weight']
    td_weight = config['td_weight']


    # # Save agent arguements to file
    # agent_config_save_path = os.path.join(log_dir, "agent_config.json")
    # with open(agent_config_save_path, 'w') as f:
    #     json.dump(agent_config, f, indent=4)

    # Training loop
    for e in tqdm(range(config['episodes']), desc="Training Episodes"): # Total training episodes
            
        try:
            difficulty = e//episodes_per_difficulty

            # block = e // episodes_per_difficulty
            # within_block = e % episodes_per_difficulty

            # if within_block <= 50:
            #     difficulty = 0
            # elif within_block <= 100:
            #     difficulty = 1
            # else:
            #     difficulty = block





            # initial_deterministic_cutoff = 100
            # gaussian_std = 1.5
            # curriculum_interval = 1000  # how often the mean increases
            # max_difficulty = 5


            # if e < 50:
            #     difficulty = 0
            # elif e < 100:
            #     difficulty = 1
            # else:
            #     # Calculate "mean difficulty" for current phase
            #     phase_index = (e - 100) // curriculum_interval
            #     mean_difficulty = min(phase_index, max_difficulty)

            #     # Sample from Gaussian and clip to valid difficulty range
            #     sampled = np.random.normal(loc=mean_difficulty, scale=gaussian_std)
            #     difficulty = int(np.clip(round(sampled), 0, max_difficulty))

            
            if e%episodes_per_difficulty == 0:
                agent.reset_epsilion()
            env.set_difficulty(difficulty)
            state = env.reset(evaluate=0,episode=e)
            
            done = False
            total_reward = 0
            finished = False


            # Training steps
            while not finished:  
                action = agent.act(state)
                next_state, reward, done, episode_result_dict = env.step(action)

                raw_reward = reward  # preserve for logging

                # normalized_reward = np.clip(raw_reward, -1, 1)  # keep reward in safe range for DQN stability
                normalized_reward = np.tanh(raw_reward)

                # Clip reward to [-1, 1]
                # clipped_reward = min(1, max(-1, reward))
        
                # Optional: differentiate step vs. terminal for clarity
                # if done:
                    # clipped_reward = np.clip(reward, -1, 1)

                agent.store_experience(state, action, normalized_reward, next_state, done)
                state = next_state
                total_reward += normalized_reward


                if done:
                    try:
                        env.destroy_actors()
                    except:
                        pass
                    finished = True
                    break

                td_error = agent.train()

            if config['track_training']:
                train_util, train_util_path = env.get_train_util()
                train_util['td_error'] = td_error

                # get uncertainity
                unc_scalar = get_mc_dropout_uncertainty(agent.policy_net, state, K=config['sample_nets'])
                train_util['uncertainty'] = unc_scalar

                 # Penalty for bad episodes
                # if total_reward < -0.5:
                #     # print(f"Penalty applied: reward = {total_reward:.3f}")
                #     td_error *= 0.1
                #     unc_scalar *= 0.1
                
                # get utility

                utility = uncertainity_weight*unc_scalar + td_weight*td_error

                utility = gaussian_difficulty_prior(utility, difficulty, e, total_episodes=config['episodes'], sigma=2.0)

                train_util['utility'] = utility


                # save to json
                with open(train_util_path, 'a') as f:
                    json.dump(safe_for_json(train_util), f, indent=4) 

        except RuntimeError:
            print("[DEBUG] Episode Failed. Resetting and trying Again")

        agent.train_episode += 1

        
        if agent.train_episode % agent.save_checkpoint == 0:
            model_path = os.path.join(log_dir, f"model_checkpoint_{agent.train_episode}.h5")
            buffer_path = os.path.join(log_dir, f"replay_buffer_{agent.train_episode}.pkl")
            agent.save_model(model_path)
            agent.save_replay_buffer(buffer_path)


        agent.rewards.insert(0, total_reward)
        if len(agent.rewards) > 100:
            agent.rewards = agent.rewards[:-1]
        avg_rewards = np.mean(agent.rewards)
        
        print(f"Episode: {e + 1}/{config['episodes']}, Episode reward: {total_reward}, Average reward (100 Episodes): {avg_rewards}")

        collision_type = env.get_collision_summary()
        print("Collision type: ")
        print(collision_type)

        # Logging average reward
        with agent.summary_writer.as_default():
            tf.summary.scalar('Episode Reward', total_reward, step=e)
            tf.summary.scalar('Average Reward (100 Episodes)', avg_rewards, step=e)
            tf.summary.scalar('Episode to Training Index', e, step=agent.train_step)
            tf.summary.scalar('Episode to Simulation Step', e, step=env.simulation_step_count)
            tf.summary.scalar('Difficulty',difficulty,step=e)
            tf.summary.scalar('intersection_id', intersection_ID_to_int(getattr(env, 'location', 'unknown')), step=e)
            
            # if collision_type is not None:
                # tf.summary.scalar('Total Collisions', collision_type['total_collisions'], step=e)

            tf.summary.scalar("collision/total_collisions", collision_type['total_collisions'], step=e)
            tf.summary.scalar("collision/average_speed", collision_type['average_speed'], step=e)

            if bool(collision_type['actor_IDs']):

                for actor_id, count in collision_type['actor_IDs'].items():
                    tf.summary.scalar(f"collision/actor_IDs/{actor_id}", count, step=e)

                for actor_type, count in collision_type['actor_types'].items():
                    tf.summary.scalar(f"collision/actor_types/{actor_type}", count, step=e)

            if episode_result_dict['termination_type'] is not None:
                tf.summary.scalar('Termination Type', episode_result_dict['termination_type'], step=e)


        # Evaluation phase
        print(f"train step: {agent.train_step}")

        
        if (agent.train_step // config['evaluation_interval']) != agent.evaluation_checkpoint:
            print("\n******Evaluation*********\n")
            # evaluate_agent(agent, env, e,max_difficulty_level)
            _,_,utility_scores = evaluate_agent(agent, env, e,max_difficulty_level,log_dir,config['episodes'],td_weight, uncertainity_weight)
            print("\n******Training*********\n")

    # Save final model and replay buffer
    final_model_path = os.path.join(model_dir, 'final_model.h5')
    final_replay_buffer_path = os.path.join(model_dir, 'final_replay_buffer.pkl')
    agent.save_model(final_model_path)
    agent.save_replay_buffer(final_replay_buffer_path)


# UGCS
def train_dqn_agent_adaptive(config):
    """
    Function to train a DQN agent in the CARLA environment.

    Args:
    - config (dict): Configuration parameters for training.

    Returns:
    - None
    """

    print("[TRAINING MODE] Adaptive")

    # Extract knob value from configuration
    knob_value = config['knob_value']

    log_dir = os.path.join("training_logs", "adaptive", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    model_dir = os.path.join("models", "adaptive", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))

    # Create directories for logging and model saving
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save config arguements to file
    config_save_path = os.path.join(log_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Initialize CARLA environment and extract state and action dimensions
    env = CarlaEnv(log_dir,**config)
    # rl_wrapper = RL_wrapper(env)
    
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
        'log_dir': log_dir,
        'use_GPU': config['use_GPU']
    }
    
    agent = DQNAgent(**agent_config)

    env.get_summary_writer_from_agent(agent.get_summary_writer())

    episodes_per_difficulty = config.get('episodes_per_level', 1000)
    max_difficulty_level = 5  # Assuming 6 levels: 0–5

    # defaults to max difficulty
    difficulty = 0
    env.set_difficulty(difficulty)

    episode_td_errors = []

    utility_scores = [1,0,0,0,0,0]
    probabilities = softmax(np.array(utility_scores),temperature=0.2)
    # print(probabilities)

    td_error = 0
    uncertainity_weight = config['uncertainity_weight']
    td_weight = config['td_weight']

    # # Save agent arguements to file
    # agent_config_save_path = os.path.join(log_dir, "agent_config.json")
    # with open(agent_config_save_path, 'w') as f:
    #     json.dump(agent_config, f, indent=4)

    # Training loop
    for e in tqdm(range(config['episodes']), desc="Training Episodes"): # Total training episodes
            
        try:
            difficulty = np.random.choice(len(utility_scores), p=probabilities)
            env.set_difficulty(difficulty)
            print("Difficulty: " +str(difficulty))
            # difficulty = e//episodes_per_difficulty
            if e%episodes_per_difficulty == 0:
                agent.reset_epsilion()
            state = env.reset(evaluate=0,episode=e)
            done = False
            total_reward = 0
            finished = False


            # Training steps
            while not finished:  
                action = agent.act(state)
                next_state, reward, done, episode_result_dict = env.step(action)

                raw_reward = reward  # preserve for logging

                # normalized_reward = np.clip(raw_reward, -1, 1)  # keep reward in safe range for DQN stability
                normalized_reward = np.tanh(raw_reward)

                # Clip reward to [-1, 1]
                # clipped_reward = min(1, max(-1, reward))
        
                # Optional: differentiate step vs. terminal for clarity
                # if done:
                    # clipped_reward = np.clip(reward, -1, 1)

                agent.store_experience(state, action, normalized_reward, next_state, done)
                state = next_state
                total_reward += normalized_reward


                if done:
                    try:
                        env.destroy_actors()
                    except:
                        pass
                    finished = True
                    break

                td_error = agent.train()

            if config['track_training']:
                train_util, train_util_path = env.get_train_util()
                train_util['td_error'] = td_error

                # get uncertainity
                unc_scalar = get_mc_dropout_uncertainty(agent.policy_net, state, K=config['sample_nets'])
                train_util['uncertainty'] = unc_scalar

                #  # Penalty for bad episodes
                # if total_reward < -0.5:
                #     # print(f"Penalty applied: reward = {total_reward:.3f}")
                #     td_error *= 0.1
                #     unc_scalar *= 0.1
                
                # get utility

                utility = uncertainity_weight*unc_scalar + td_weight*td_error

                utility = gaussian_difficulty_prior(utility, difficulty, e, total_episodes=config['episodes'], sigma=2.0)

                train_util['utility'] = utility


                # save to json
                with open(train_util_path, 'a') as f:
                    json.dump(safe_for_json(train_util), f, indent=4) 

        except RuntimeError:
            print("[DEBUG] Episode Failed. Resetting and trying Again")

        agent.train_episode += 1

        
        if agent.train_episode % agent.save_checkpoint == 0:
            model_path = os.path.join(log_dir, f"model_checkpoint_{agent.train_episode}.h5")
            buffer_path = os.path.join(log_dir, f"replay_buffer_{agent.train_episode}.pkl")
            agent.save_model(model_path)
            agent.save_replay_buffer(buffer_path)


        agent.rewards.insert(0, total_reward)
        if len(agent.rewards) > 100:
            agent.rewards = agent.rewards[:-1]
        avg_rewards = np.mean(agent.rewards)
        
        print(f"Episode: {e + 1}/{config['episodes']}, Episode reward: {total_reward}, Average reward (100 Episodes): {avg_rewards}")

        collision_type = env.get_collision_summary()
        print("Collision type: ")
        print(collision_type)

        # Logging average reward
        with agent.summary_writer.as_default():
            tf.summary.scalar('Episode Reward', total_reward, step=e)
            tf.summary.scalar('Average Reward (100 Episodes)', avg_rewards, step=e)
            tf.summary.scalar('Episode to Training Index', e, step=agent.train_step)
            tf.summary.scalar('Episode to Simulation Step', e, step=env.simulation_step_count)
            tf.summary.scalar('Difficulty',difficulty,step=e)
            tf.summary.scalar('intersection_id', intersection_ID_to_int(getattr(env, 'location', 'unknown')), step=e)
            
            # if collision_type is not None:
                # tf.summary.scalar('Total Collisions', collision_type['total_collisions'], step=e)

            tf.summary.scalar("collision/total_collisions", collision_type['total_collisions'], step=e)
            tf.summary.scalar("collision/average_speed", collision_type['average_speed'], step=e)

            if bool(collision_type['actor_IDs']):

                for actor_id, count in collision_type['actor_IDs'].items():
                    tf.summary.scalar(f"collision/actor_IDs/{actor_id}", count, step=e)

                for actor_type, count in collision_type['actor_types'].items():
                    tf.summary.scalar(f"collision/actor_types/{actor_type}", count, step=e)

            if episode_result_dict['termination_type'] is not None:
                tf.summary.scalar('Termination Type', episode_result_dict['termination_type'], step=e)


        # Evaluation phase
        print(f"train step: {agent.train_step}")

        
        if (agent.train_step // config['evaluation_interval']) != agent.evaluation_checkpoint:
            print("\n******Evaluation*********\n")
            difficulty,_,utility_scores = evaluate_agent(agent, env, e,max_difficulty_level,log_dir,config['episodes'],td_weight, uncertainity_weight)
            probabilities = softmax(np.array(utility_scores))
            # env.set_difficulty(difficulty)
            print("\n******Training*********\n")

    # Save final model and replay buffer
    final_model_path = os.path.join(model_dir, 'final_model.h5')
    final_replay_buffer_path = os.path.join(model_dir, 'final_replay_buffer.pkl')
    agent.save_model(final_model_path)
    agent.save_replay_buffer(final_replay_buffer_path)

def evaluate_agent(agent, env, e, max_difficulty_level, log_dir,total_training_episodes,td_weight=1, uncertainity_weight=100):

    eps_per_difficulty = 5
    gamma = 0.99

    results = {}  # difficulty → list of utility scores

    # Initialize reward history if not already done
    if not hasattr(agent, "reward_history"):
        agent.reward_history = {d: [] for d in range(max_difficulty_level + 1)}

    for d in range(0, max_difficulty_level + 1):
        env.set_difficulty(d)
        td_errors_all = []
        uncertainties_all = []
        total_rewards_all = []

        for eval_eps in range(eps_per_difficulty):

            print("---------------------------")
            print("[Eval] Diff: " + str(d) + ". Episode: " + str(eval_eps))
            print("---------------------------")

            state = env.reset(evaluate=1, episode=e)
            done = False
            episode_td_errors = []
            episode_uncertainties = []
            total_reward = 0

            while not done:
                q_values = agent.policy_net(tf.convert_to_tensor([state], dtype=tf.float32)).numpy()[0]
                action = np.argmax(q_values)
                uncertainty = get_mc_dropout_uncertainty(agent.policy_net, state, K=10)

                next_state, reward, done, episode_result_dict = env.step(action)
                total_reward += reward

                next_q_values = agent.target_net(tf.convert_to_tensor([next_state], dtype=tf.float32)).numpy()[0]
                next_action = np.argmax(agent.policy_net(tf.convert_to_tensor([next_state], dtype=tf.float32)).numpy()[0])
                td_target = reward + gamma * next_q_values[next_action] * (1 - int(done))
                td_error = td_target - q_values[action]

                episode_td_errors.append(abs(td_error))
                episode_uncertainties.append(uncertainty)
                total_rewards_all.append(total_reward)

                state = next_state

            episode_mean_td_error = np.mean(episode_td_errors)
            episode_mean_uncertainty = np.mean(episode_uncertainties)

            # Penalty for bad episodes
            if total_reward < -5:
                print(f"Penalty applied: reward = {total_reward:.3f}")
                episode_mean_td_error *= 0.5
                episode_mean_uncertainty *= 0.5

            # Continuous sigmoid-based penalty for low-reward episodes
            reward_threshold = -2.5     # Center of the sigmoid curve
            sharpness = 2.0           # Higher = steeper penalty near the threshold

            # Compute penalty factor (smoothly ranges from ~0 to 1)
            penalty_factor = 1 / (1 + np.exp(-(total_reward - reward_threshold) / sharpness))

            # Apply penalty to TD-error and uncertainty
            episode_mean_td_error *= penalty_factor
            episode_mean_uncertainty *= penalty_factor



            # scale = soft_reward_penalty(total_reward)
            # td_error *= scale
            # uncertainty *= scale

            td_errors_all.append(episode_mean_td_error)
            uncertainties_all.append(episode_mean_uncertainty)

            collision_type = env.get_collision_summary()
            print("Eval Ep Reward: " + str(total_reward))
            time.sleep(0.5)

            with agent.summary_writer.as_default():
                tf.summary.scalar(f"Eval/{e}/{d}/collision_history_eval", collision_type['total_collisions'], step=eval_eps)
                tf.summary.scalar(f'Eval/{e}/{d}/total_reward_eval', total_reward, step=eval_eps)
                tf.summary.scalar(f'Eval/{e}/{d}/td_error', episode_mean_td_error, step=eval_eps)
                tf.summary.scalar(f'Eval/{e}/{d}/mean_uncertainity', episode_mean_uncertainty, step=eval_eps)
                # tf.summary.scalar(f'Eval/{e}/{d}/reward_gradient', reward_gradient, step=eval_eps)

        # Compute unadjusted utility
        mean_td_error = np.mean(td_errors_all)
        mean_uncertainty = np.mean(uncertainties_all)
        utility = td_weight* mean_td_error + uncertainity_weight * mean_uncertainty
        # utility = mean_uncertainty
        # utility = mean_td_error

        # Update reward history and compute forgetting boost
        mean_reward = np.mean(total_rewards_all)
        agent.reward_history[d].append(mean_reward)
        if len(agent.reward_history[d]) > 5:
            agent.reward_history[d].pop(0)

        recent_rewards = agent.reward_history[d]
        if len(recent_rewards) >= 2:
            reward_gradient = recent_rewards[-1] - recent_rewards[0]
            forgetting_boost = -0.2 * reward_gradient if reward_gradient < 0 else 0
            # -0.2
        else:
            forgetting_boost = 0
            reward_gradient = 0



        # --- Apply curriculum prior ---
        utility = gaussian_difficulty_prior(utility, d, e, total_episodes=total_training_episodes, sigma=2.0)
        utility += forgetting_boost


        results[d] = {
            "Episode": e,
            "Difficulty": d,
            "utility": utility,
            "mean_td_error": mean_td_error,
            "mean_uncertainty": mean_uncertainty
        }

        eval_results_save_path = os.path.join(log_dir, "eval_results.csv")
        file_exists = os.path.exists(eval_results_save_path)

        with open(eval_results_save_path, 'a', newline='') as csvfile:
            fieldnames = ["Episode", "Difficulty", "Utility", "Mean_TD_Error", "Mean_Uncertainty", "forgetting_boost", "reward_gradient"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "Episode": e,
                "Difficulty": d,
                "Utility": utility,
                "Mean_TD_Error": mean_td_error,
                "Mean_Uncertainty": mean_uncertainty,
                "forgetting_boost": forgetting_boost,
                "reward_gradient": reward_gradient
            })

        print("-----------------------------------")
        print(f"Difficulty {d} → Utility: {utility:.4f}, TD-Error: {mean_td_error:.4f}, Uncertainty: {mean_uncertainty:.4f}, Reward Gradient: {reward_gradient:.4f}")
        print("-----------------------------------")

    best_difficulty = max(results, key=lambda k: results[k]["utility"])
    print(f"\nSwitching to difficulty level: {best_difficulty}\n")

    utilities = [results[d]["utility"] for d in sorted(results.keys())]

    agent.evaluation_checkpoint += 1
    return best_difficulty, results, utilities

# Epistemic (model) uncertainty
# predictive standard deviation
def get_mc_dropout_uncertainty(model, state, K=10):
    preds = []
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

    for _ in range(K):
        q_vals = model(state_tensor, training=True)  # Keep dropout active
        preds.append(q_vals.numpy()[0])  # shape: (num_actions,)

    preds = np.array(preds)  # shape: (K, num_actions)
    uncertainty = np.mean(np.std(preds, axis=0))  # mean std over actions
    return uncertainty


def gaussian_difficulty_prior(utility, difficulty, episode, total_episodes, sigma=1.0):
    num_stages = 6
    stage_duration = total_episodes / num_stages

    # Map episode to curriculum stage (center of Gaussian)
    current_stage = min(int(episode / stage_duration), num_stages - 1)
    center_difficulty = current_stage  # center around stage index

    # Compute Gaussian weight
    weight = math.exp(-0.5 * ((difficulty - center_difficulty) / sigma) ** 2)

    return utility * weight

# Compute softmax
def softmax(x, temperature=0.3):
    x = np.array(x)
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


def soft_reward_penalty(reward, r0=-10, k=5.0):
    return 1.0 / (1.0 + np.exp(k * (reward - r0)))


def softmax_temperature(x, temperature=0.3, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    t = max(temperature, eps)
    x = x / t
    x = x - np.max(x)  # stabilize
    ex = np.exp(x)
    s = ex.sum()
    if s <= eps:
        return np.ones_like(x) / len(x)  # uniform fallback
    return ex / s

# Gets TSCL slopes and LP
def evaluate_agent_tscl_distribution(
    agent,
    env,
    e,
    max_difficulty_level,
    log_dir,
    total_training_episodes,
    window=2,
    temperature=0.5,
    eps_per_difficulty=5
):
    """
    Compute a *distribution* over difficulties using TSCL's learning progress (reward change).

    For each difficulty d:
      - run eps_per_difficulty eval episodes → mean_reward[d]
      - update a short history buffer per difficulty
      - progress[d] = last_mean - prev_mean  (0 if not enough history)

    Then convert progress → probabilities via softmax_temperature(progress, T=temperature).

    Returns: (probs, progresses, mean_rewards), also appends rows to CSV and TB.
    """

    # ensure short reward history exists (kept on the agent so it persists across evals)
    if not hasattr(agent, "tscl_reward_history"):
        agent.tscl_reward_history = {d: [] for d in range(max_difficulty_level + 1)}

    mean_rewards = []
    progresses   = []

    for d in range(0, max_difficulty_level + 1):
        env.set_difficulty(d)
        rewards = []

        for eval_eps in range(eps_per_difficulty):
            print("---------------------------")
            print(f"[TSCL Eval] Diff: {d}. Episode: {eval_eps}")
            print("---------------------------")

            state = env.reset(evaluate=1, episode=e)
            done = False
            total_reward = 0.0

            while not done:
                q_values = agent.policy_net(tf.convert_to_tensor([state], dtype=tf.float32)).numpy()[0]
                action = int(np.argmax(q_values))
                next_state, reward, done, episode_result_dict = env.step(action)
                total_reward += reward
                state = next_state

            rewards.append(total_reward)

            with agent.summary_writer.as_default():
                tf.summary.scalar(f"TSCL_Eval/{e}/{d}/total_reward_eval", total_reward, step=eval_eps)

        mean_reward = float(np.mean(rewards))
        agent.tscl_reward_history[d].append(mean_reward)
        if len(agent.tscl_reward_history[d]) > window:
            agent.tscl_reward_history[d].pop(0)

        hist = agent.tscl_reward_history[d]
        progress = float(hist[-1] - hist[-2]) if len(hist) >= 2 else 0.0

        mean_rewards.append(mean_reward)
        progresses.append(progress)

    # softmax over progress (learning progress / slope)
    probs = softmax_temperature(np.array(progresses, dtype=np.float64), temperature=temperature)

    # ---- CSV logging (per difficulty) ----
    eval_csv = os.path.join(log_dir, "tscl_eval_results.csv")
    file_exists = os.path.exists(eval_csv)
    with open(eval_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Episode", "Difficulty", "Mean_Reward", "Progress", "Probability", "Temperature"
        ])
        if not file_exists:
            writer.writeheader()
        for d in range(0, max_difficulty_level + 1):
            writer.writerow({
                "Episode": e,
                "Difficulty": d,
                "Mean_Reward": mean_rewards[d],
                "Progress":   progresses[d],
                "Probability": probs[d],
                "Temperature": temperature
            })

    # ---- TB scalars ----
    with agent.summary_writer.as_default():
        tf.summary.scalar("TSCL/Temperature", temperature, step=e)
        for d in range(0, max_difficulty_level + 1):
            tf.summary.scalar(f"TSCL/MeanReward/d{d}", mean_rewards[d], step=e)
            tf.summary.scalar(f"TSCL/Progress/d{d}",   progresses[d],   step=e)
            tf.summary.scalar(f"TSCL/Prob/d{d}",       probs[d],        step=e)

    # advance evaluation checkpoint to throttle
    # agent.evaluation_checkpoint += 1

    print("\n[TSCL] Distribution over difficulties:")
    for d in range(0, max_difficulty_level + 1):
        print(f"  d={d}: progress={progresses[d]:+.4f}  prob={probs[d]:.3f}")
    print()

    return probs, progresses, mean_rewards

# TSCL
def train_dqn_agent_tscl(config):
    """
    Train with TSCL *episode-by-episode* sampling:
      - At eval checkpoints: compute TSCL probs over difficulties from reward-change (progress).
      - Between checkpoints: for *each episode*, sample difficulty ~ Categorical(probs).
      - Also run your legacy evaluator to compute utilities (td-error + uncertainty) for logging only.
    """

    print("[TRAINING MODE] Adaptive-TSCL (per-episode sampling)")

    knob_value = config['knob_value']

    log_dir   = os.path.join("training_logs", "adaptive_tscl_ep", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    model_dir = os.path.join("models",        "adaptive_tscl_ep", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(log_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    env = CarlaEnv(log_dir, **config)

    state_shape = env.observation_space.shape
    action_dim  = env.action_space.n

    agent_config = {
        'input_shape': state_shape,
        'num_actions': action_dim,
        'replay_buffer_capacity': config['replay_buffer_capacity'],
        'batch_size':   config['batch_size'],
        'gamma':        config['gamma'],
        'lr':           config['lr'],
        'epsilon_start':config['epsilon_start'],
        'epsilon_end':  config['epsilon_end'],
        'epsilon_decay':config['epsilon_decay'],
        'target_update':config['target_update'],
        'save_model_freq': config['save_model_freq'],
        'knob_value':   knob_value,
        'log_dir':      log_dir,
        'use_GPU':      config['use_GPU']
    }
    agent = DQNAgent(**agent_config)
    env.get_summary_writer_from_agent(agent.get_summary_writer())

    max_difficulty_level = 5  # 0..5
    episodes_per_difficulty = config.get('episodes_per_level', 1000)  # still used for epsilon reset cadence

    # TSCL params
    tscl_window        = config.get("tscl_window", 2)
    tscl_temperature   = config.get("tscl_temperature", 0.5)
    tscl_eps_per_diff  = config.get("tscl_eps_per_difficulty", 5)

    # legacy utility weights (for telemetry only)
    uncertainity_weight = config['uncertainity_weight']
    td_weight           = config['td_weight']

    # initialize TSCL distribution (uniform to start)
    tscl_probs = np.ones(max_difficulty_level + 1, dtype=np.float64) / (max_difficulty_level + 1)

    td_error = 0.0

    for e in tqdm(range(config['episodes']), desc="Training Episodes"):
        try:
            # re-evaluate TSCL distribution at checkpoints (mirrors your adaptive gate)
            # if (agent.train_step // config['evaluation_interval']) != agent.evaluation_checkpoint:
            current_bucket = agent.train_step // config['evaluation_interval']
            if current_bucket > agent.evaluation_checkpoint:
                print("\n******Evaluation (TSCL distribution + Legacy util)*********\n")

                # TSCL distribution from reward-change (progress)
                tscl_probs, tscl_progresses, tscl_mean_rewards = evaluate_agent_tscl_distribution(
                    agent, env, e, max_difficulty_level, log_dir,
                    total_training_episodes=config['episodes'],
                    window=tscl_window,
                    temperature=tscl_temperature,
                    eps_per_difficulty=tscl_eps_per_diff
                )

                # Legacy evaluation (for logging only)
                try:
                    _, legacy_results, legacy_utilities = evaluate_agent(
                        agent, env, e, max_difficulty_level, log_dir,
                        total_training_episodes=config['episodes'],
                        td_weight=td_weight,
                        uncertainity_weight=uncertainity_weight
                    )
                    with agent.summary_writer.as_default():
                        for d in range(0, max_difficulty_level + 1):
                            tf.summary.scalar(f"LegacyEval/Utility/d{d}", legacy_results[d]["utility"], step=e)
                            tf.summary.scalar(f"LegacyEval/TD_Error/d{d}", legacy_results[d]["mean_td_error"], step=e)
                            tf.summary.scalar(f"LegacyEval/Uncertainty/d{d}", legacy_results[d]["mean_uncertainty"], step=e)
                except NameError:
                    print("[WARN] evaluate_agent(...) not found; skipping legacy utility logging.")

            # per-episode sampling from current TSCL distribution
            difficulty = int(np.random.choice(len(tscl_probs), p=tscl_probs))
            env.set_difficulty(difficulty)
            print(f"[TSCL] Episode {e} → sampled difficulty: {difficulty}")

            if e % episodes_per_difficulty == 0:
                agent.reset_epsilion()

            state   = env.reset(evaluate=0, episode=e)
            done    = False
            total_reward = 0.0
            finished = False

            while not finished:
                action = agent.act(state)
                next_state, reward, done, episode_result_dict = env.step(action)

                normalized_reward = np.tanh(reward)  # like your adaptive
                agent.store_experience(state, action, normalized_reward, next_state, done)
                state = next_state
                total_reward += normalized_reward

                if done:
                    try:
                        env.destroy_actors()
                    except:
                        pass
                    finished = True
                    break

                td_error = agent.train()

            # optional per-episode telemetry (kept minimal; safe fallbacks)
            if config['track_training']:
                try:
                    train_util, train_util_path = env.get_train_util()
                    train_util['td_error'] = td_error
                    # If you have MC-dropout util available, log it; else 0
                    try:
                        unc_scalar = get_mc_dropout_uncertainty(agent.policy_net, state, K=config.get('sample_nets', 5))
                    except Exception:
                        unc_scalar = 0.0
                    train_util['uncertainty'] = unc_scalar

                    # keep "utility" for continuity of plots (not used to pick difficulty here)
                    try:
                        utility = uncertainity_weight * unc_scalar + td_weight * td_error
                        utility = gaussian_difficulty_prior(utility, difficulty, e,
                                                            total_episodes=config['episodes'], sigma=2.0)
                        train_util['utility'] = utility
                        with open(train_util_path, 'a') as f:
                            json.dump(safe_for_json(train_util), f, indent=4)
                    except Exception:
                        pass
                except Exception:
                    pass

        except RuntimeError:
            print("[DEBUG] Episode Failed. Resetting and trying again")

        agent.train_episode += 1

        # checkpoints
        if agent.train_episode % agent.save_checkpoint == 0:
            model_path  = os.path.join(log_dir, f"model_checkpoint_{agent.train_episode}.h5")
            buffer_path = os.path.join(log_dir, f"replay_buffer_{agent.train_episode}.pkl")
            agent.save_model(model_path)
            agent.save_replay_buffer(buffer_path)

        # rolling reward stats
        agent.rewards.insert(0, total_reward)
        if len(agent.rewards) > 100:
            agent.rewards = agent.rewards[:-1]
        avg_rewards = np.mean(agent.rewards)

        collision_type = env.get_collision_summary()
        print(f"Episode: {e + 1}/{config['episodes']}, "
              f"Episode reward: {total_reward:.3f}, "
              f"Average reward (100 Episodes): {avg_rewards:.3f}")
        print("Collision type:")
        print(collision_type)

        with agent.summary_writer.as_default():
            tf.summary.scalar('Episode Reward', total_reward, step=e)
            tf.summary.scalar('Average Reward (100 Episodes)', avg_rewards, step=e)
            tf.summary.scalar('Episode to Training Index', e, step=agent.train_step)
            tf.summary.scalar('Episode to Simulation Step', e, step=env.simulation_step_count)
            tf.summary.scalar('Difficulty', difficulty, step=e)
            tf.summary.scalar('intersection_id', intersection_ID_to_int(getattr(env, 'location', 'unknown')), step=e)
            tf.summary.scalar("collision/total_collisions", collision_type['total_collisions'], step=e)
            tf.summary.scalar("collision/average_speed",   collision_type['average_speed'],   step=e)
            if bool(collision_type['actor_IDs']):
                for actor_id, count in collision_type['actor_IDs'].items():
                    tf.summary.scalar(f"collision/actor_IDs/{actor_id}", count, step=e)
                for actor_type, count in collision_type['actor_types'].items():
                    tf.summary.scalar(f"collision/actor_types/{actor_type}", count, step=e)
            if episode_result_dict['termination_type'] is not None:
                tf.summary.scalar('Termination Type', episode_result_dict['termination_type'], step=e)

    # save final artifacts
    final_model_path          = os.path.join(model_dir, 'final_model.h5')
    final_replay_buffer_path  = os.path.join(model_dir, 'final_replay_buffer.pkl')
    agent.save_model(final_model_path)
    agent.save_replay_buffer(final_replay_buffer_path)
