import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

def compute_actor_risks_in_cone(ego_pos, ego_heading, actors, 
                                cone_angle_deg=60, max_range=20, 
                                w1=1.0, w2=0.5):
    risks = {}

    for actor in actors:
        actor_id = actor['id']
        ax, ay = actor['pos']
        dx = ax - ego_pos[0]
        dy = ay - ego_pos[1]
        dist = np.hypot(dx, dy)
        if dist == 0 or dist > max_range:
            continue

        angle = np.arctan2(dy, dx)
        rel_angle = np.abs((angle - ego_heading + np.pi) % (2 * np.pi) - np.pi)
        if rel_angle > np.radians(cone_angle_deg / 2):
            continue

        dist_score = max(0, 1 - (dist / max_range))
        align_score = np.cos(rel_angle)
        risk = w1 * dist_score + w2 * align_score
        risks[actor_id] = float(risk)

    return risks

def visualize_risk_cone(ego_pos, ego_heading, actors, cone_angle_deg=60, max_range=20):
    risks = compute_actor_risks_in_cone(ego_pos, ego_heading, actors, cone_angle_deg, max_range)
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(ego_pos[0] - max_range, ego_pos[0] + max_range)
    ax.set_ylim(ego_pos[1] - max_range, ego_pos[1] + max_range)

    # Plot the ego vehicle
    ax.plot(ego_pos[0], ego_pos[1], 'bo', label='Ego Vehicle')

    # Plot risk cone
    angle_deg = np.degrees(ego_heading)
    wedge = Wedge(ego_pos, max_range, angle_deg - cone_angle_deg / 2, angle_deg + cone_angle_deg / 2, 
                  alpha=0.2, color='blue')
    ax.add_patch(wedge)

    # Plot actors
    for actor in actors:
        actor_id = actor['id']
        ax_, ay_ = actor['pos']
        if actor_id in risks:
            color = plt.cm.viridis(risks[actor_id])  # Map risk to color
            ax.plot(ax_, ay_, 'o', color=color)
            ax.text(ax_ + 0.5, ay_ + 0.5, f"{risks[actor_id]:.2f}", fontsize=8)

    plt.title('Risk Cone Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
ego_position = (0, 0)
ego_heading = np.radians(0)  # Facing right
sample_actors = [
    {'id': 'A1', 'pos': (10, 2)},
    {'id': 'A2', 'pos': (15, 5)},
    {'id': 'A3', 'pos': (5, -1)},
    {'id': 'A4', 'pos': (22, 0)},  # Outside range
    {'id': 'A5', 'pos': (-5, 0)},  # Behind the vehicle
]

visualize_risk_cone(ego_position, ego_heading, sample_actors)
