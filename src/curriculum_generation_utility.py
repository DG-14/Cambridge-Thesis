import numpy as np

def compute_gae_utility(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute the learning potential (positive value loss) for a trajectory.
    Inputs:
        rewards: np.array of shape (T,)
        values: np.array of shape (T+1,)  # V(s_0)...V(s_T)
        gamma: discount factor
        lam: GAE lambda
    Returns:
        utility_score: scalar
    """
    T = len(rewards)
    deltas = np.zeros(T)
    advantages = np.zeros(T)

    for t in range(T):
        deltas[t] = rewards[t] + gamma * values[t + 1] - values[t]

    # GAE computation (discounted sum of deltas)
    gae = 0.0
    for t in reversed(range(T)):
        gae = deltas[t] + gamma * lam * gae
        advantages[t] = max(gae, 0)  # clip to positive part only

    utility_score = np.mean(advantages)
    return utility_score
