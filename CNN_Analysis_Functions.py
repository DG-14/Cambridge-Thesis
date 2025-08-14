import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def compute_saliency_map(model, state, action_index):
    """
    Generate a saliency map for a given input state and action index.

    Parameters:
    - model: the trained DQN policy network.
    - state: np.array of shape (84, 84, 4), input image stack.
    - action_index: int, action whose influence to visualize.

    Returns:
    - saliency_map: np.array of shape (84, 84), normalized gradients.
    """
    # Expand dims for batch format
    state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(state_tensor)
        q_values = model(state_tensor, training=False)
        selected_q = q_values[:, action_index]

    # Compute gradients of the selected Q-value wrt input
    grads = tape.gradient(selected_q, state_tensor)[0]  # shape: (84, 84, 4)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()  # shape: (84, 84)

    # Normalize
    saliency -= saliency.min()
    saliency /= saliency.max() + 1e-8
    return saliency

def plot_saliency(state, saliency):
    """
    Plot the original grayscale input and the saliency map side by side.
    """
    input_image = np.mean(state, axis=-1)  # collapse 4-frame stack to 2D
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Input State")
    plt.imshow(input_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Saliency Map")
    plt.imshow(saliency, cmap='hot')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Useage Example:
# state = env.reset(evaluate=1)  # or load a saved frame
# action = agent.act(state)
# saliency_map = compute_saliency_map(agent.policy_net, state, action)
# plot_saliency(state, saliency_map)
