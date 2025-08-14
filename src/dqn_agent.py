import random
import numpy as np
import tensorflow as tf
from src.uncertainty import build_cnn_model_with_dropout
from tensorflow.keras import layers, initializers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from collections import deque
import pickle

tf.keras.utils.disable_interactive_logging()


class ReplayBuffer:
    """
    Replay buffer to store and sample experience tuples for training.
    """
    def __init__(self, capacity):
        """
        Initialize the replay buffer.

        Parameters:
        capacity (int): Maximum number of experiences to store in the buffer.
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def insert(self, state, action, reward, next_state, done):
        """
        Insert a new experience into the buffer.

        Parameters:
        state (np.array): The current state.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (np.array): The next state.
        done (bool): Whether the episode is done.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.buffer = pickle.load(f)

# CNN Model
def build_cnn_model(input_shape, num_actions, learning_rate, dropout_rate=0.2):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Lambda(lambda x: x / 255.0)(inputs)

    # Convolutional base
    x = layers.Conv2D(32, (8, 8), strides=4, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_actions, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)  # or clipvalue=0.5
    model.compile(optimizer=optimizer, loss='mse')

    return model

# DDQN Agent.
class DQNAgent:
    """
    Deep Q-Network agent for training and interacting with the environment.
    """
    def __init__(self, **kwargs):
        """
        Initialize the DQN agent with the given parameters.

        Parameters:
        kwargs (dict): Dictionary of configuration parameters.
        """
        if not kwargs.get('use_GPU', False):
            tf.config.set_visible_devices([], 'GPU')

        self.input_shape = kwargs['input_shape']
        self.num_actions = kwargs['num_actions']
        self.gamma = kwargs['gamma']
        self.batch_size = kwargs['batch_size']
        self.epsilon_start = kwargs['epsilon_start']
        self.epsilon_end = kwargs['epsilon_end']
        self.epsilon_decay = kwargs['epsilon_decay']
        self.target_update = kwargs['target_update']
        self.save_checkpoint = kwargs['save_model_freq']
        self.log_dir = kwargs['log_dir']
        self.knob = kwargs['knob_value']

        self.epsilon = self.epsilon_start
        self.replay_buffer = ReplayBuffer(kwargs['replay_buffer_capacity'])
        # print("Input shape")
        # print(self.input_shape)
        # print("Num Actions")
        # print(self.num_actions)

        # self.policy_net = build_cnn_model(self.input_shape, self.num_actions, kwargs['lr'])
        # self.target_net = build_cnn_model(self.input_shape, self.num_actions, kwargs['lr'])
        
        self.policy_net = build_cnn_model_with_dropout(self.input_shape, self.num_actions, kwargs['lr'], dropout_rate=0.2)
        self.target_net = build_cnn_model_with_dropout(self.input_shape, self.num_actions, kwargs['lr'], dropout_rate=0.2)


        self.update_target_network()

        self.train_step = 0
        self.evaluation_checkpoint = 0

        self.train_episode = 0
        self.evaluation_episode_1 = 0
        self.evaluation_episode_2 = 0

        self.rewards = []


        if bool(self.log_dir):
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        #     # self.summary_writer.set_as_default()

    def get_summary_writer(self):

        return self.summary_writer

    def update_target_network(self):
        """
        Update the target network with the weights from the policy network.
        """
        self.target_net.set_weights(self.policy_net.get_weights())


    # return an action 
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.policy_net(np.expand_dims(state, axis=0), training=False)
        return tf.argmax(q_values[0]).numpy()


    def act_trained(self, state):  # No exploration
        """
        Choose an action without exploration (for evaluation).

        Parameters:
        state (np.array): The current state.

        Returns:
        int: The action to take.
        """
        state = np.expand_dims(state, axis=0)
        act_values = self.policy_net.predict(state)
        return np.argmax(act_values[0])

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # print("TRAINING")

        # Predict next actions using policy network (Double DQN step 1)
        next_action_indices = tf.argmax(self.policy_net(next_states), axis=1)

        # Predict Q-values using target network (Double DQN step 2)
        target_q_values = self.target_net(next_states)
        max_next_q = tf.gather(target_q_values, next_action_indices, batch_dims=1)

        # Compute target values
        target = rewards + (1 - dones) * self.gamma * max_next_q

        with tf.GradientTape() as tape:
            # Forward pass through policy network
            q_values = self.policy_net(states, training=True)

            # Get Q-values for the taken actions
            indices = tf.stack([tf.range(self.batch_size), actions], axis=1)
            q_action = tf.gather_nd(q_values, indices)

            td_error = target - q_action  # shape: (batch_size,)
            
            # positive_td_error = tf.maximum(td_error, 0.0)

            # Optional: average learning potential for the whole batch
            # learning_potential = tf.reduce_mean(positive_td_error)                

            # Compute Huber loss for better stability
            loss_fn = tf.keras.losses.Huber()
            loss = loss_fn(target, q_action)

            # Max Q-value for logging
            max_q = tf.reduce_max(q_values).numpy()

        # Backpropagation and optimization
        gradients = tape.gradient(loss, self.policy_net.trainable_variables)
        self.policy_net.optimizer.apply_gradients(zip(gradients, self.policy_net.trainable_variables))

        # Logging
        with self.summary_writer.as_default():
            tf.summary.scalar('training_loss', loss.numpy(), step=self.train_step)
            tf.summary.scalar('epsilon', self.epsilon, step=self.train_step)
            tf.summary.scalar('max_q_value', max_q, step=self.train_step)
            # tf.summary.scalar('learning_potential', learning_potential.numpy(), step=self.train_step)

        # Update counters
        self.train_step += 1
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Periodically update target network
        if self.train_step % self.target_update == 0:
            self.update_target_network()

        # Save checkpoints
        if self.train_episode % self.save_checkpoint == 0:
            self.save_model(f'{self.log_dir}/model_checkpoint_{self.train_episode}.h5')
            self.save_replay_buffer(f'{self.log_dir}/replay_buffer_{self.train_episode}.pkl')

        return td_error


    def reset_epsilion(self):
        self.epsilon = self.epsilon_start
        return
    
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.insert(state, action, reward, next_state, done)

    def save_model(self, filename):
        print("Model saving...")
        # self.policy_net.save(filename)
        print("Model saved.")

    def load_model(self, filename):
        self.policy_net = tf.keras.models.load_model(filename)
        self.update_target_network()

    def save_replay_buffer(self, filename):
        pass
        # print("Buffer saving...")
        # self.replay_buffer.save(filename)
        # print("Buffer saved.")

    def load_replay_buffer(self, filename):
        self.replay_buffer.load(filename)
