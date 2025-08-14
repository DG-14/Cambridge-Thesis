import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers, optimizers
import random

# Randomly generated weather parameters for uncertainty evaluation
# It leads to a variance of approximately 0.6 in the input images.
weather_params = {
    'cloudiness': random.randint(0, 20),
    'sun_altitude_angle': 90
}

def build_cnn_model_with_dropout(input_shape, num_actions, learning_rate, dropout_rate=0.2):
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
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


# Calculate Total Uncertainty (σ_tot)
def calculate_total_uncertainty(model, noisy_images_list, samples):
    predictions = []
    for i in range(samples):
        noisy_images = noisy_images_list[i]
        noisy_images = np.expand_dims(noisy_images, axis=0)
        # pred = model.predict(noisy_images)
        pred = model(noisy_images, training=True).numpy()
        preds = pred[0]
        predictions.append(preds)
    predictions = np.array(predictions)

    mu_bar = np.mean(predictions, axis=0)

    sigma_tot = 2 + np.mean((predictions - mu_bar) ** 2, axis=0)
    # 2 ~= 0.6 (is resulted from changing Carla weather) + 1.4 (noise added to input images)

    return sigma_tot, mu_bar, predictions


# Calculate Negative Log-Likelihood (NLL)
def calculate_nll(sigma_tot, y_gt, y_pred):
    nll = np.where(sigma_tot != 0,
                   0.5 * np.log(sigma_tot) + 0.5 / sigma_tot * (y_gt - y_pred) ** 2,
                   0)
    return float(np.mean(nll))


# Calculate RMSE
def calculate_rmse(y_gt, y_predict):
    y_gt = [y_gt]
    y_predict = np.array(y_predict)
    y_gt = np.array(y_gt)
    rmse = np.sqrt(np.mean((y_gt - y_predict) ** 2, axis=0))
    return float(np.mean(rmse))  # or np.mean(rmse).item()


# Optimize dropout_rate (Φ) using Grid Search.
def optimize_phi(state_shape, num_actions, learning_rate, num_samples, y_gt, noisy_image_list):
    # log-range of 20 possible rates in the range [0, 1]:
    # (see more details at https://ieeexplore.ieee.org/document/9001195):
    rates = [0.01, 0.012, 0.016, 0.02, 0.026, 0.033, 0.042, 0.054, 0.069, 0.088,
             0.112, 0.143, 0.183, 0.233, 0.297, 0.379, 0.483, 0.615, 0.784, 1.]
    best_nll = float('inf')
    best_rmse = float('inf')
    best_phi = None

    position = np.argmax(y_gt)
    y_gt = y_gt[position]

    for rate in rates:
        bayesian_model = build_cnn_model_with_dropout(state_shape, num_actions, learning_rate, rate)
        sigma_tot, mu_bar, y_pred = calculate_total_uncertainty(bayesian_model, noisy_image_list, num_samples)

        mu_bar = mu_bar[position]
        y_predict = [y[position] for y in y_pred]
        sigma_tot = sigma_tot[position]

        nll = calculate_nll(sigma_tot, y_gt, mu_bar)
        rmse = calculate_rmse(y_gt, y_predict)

        if nll < best_nll:
            best_nll = nll
            best_phi = rate
            best_rmse = rmse

    return best_phi, best_nll, best_rmse

def calculate_entropy(q_values):
    """
    Compute entropy over the softmax-normalized Q-values.

    Args:
    - q_values (np.array): 1D array of Q-values for actions.

    Returns:
    - float: Entropy measure (higher = more uncertainty).
    """
    probs = tf.nn.softmax(q_values).numpy()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)
