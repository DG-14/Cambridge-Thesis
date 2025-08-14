from src.waypoints import *
import numpy as np
import pandas as pd
import os
import numpy as np
import tensorflow as tf

# intersection
def intersection_ID_to_int(input):

    intersection_map = {
        'intersection1': 1,
        'intersection2': 2,
        'intersection3': 3,
        'intersection4': 4,
        'intersection5': 5,
        'intersection6': 6,
        'intersection7': 7,
        'intersection8': 8,
        'intersection9': 9,
        'intersection10': 10,
        'intersection11': 11
    }

    return intersection_map.get(input, -1)  # -1 = unknown

def classify_actor_type(actor_id):
    """
    Classifies a CARLA actor blueprint ID into a high-level semantic category.

    Args:
        actor_id (str): The blueprint ID of the actor.

    Returns:
        str: One of ['static', 'car', 'motorbike', 'bicycle', 'child', 'regular_adult', 'old_adult']
    """

    # print("Actor ID: " + str(actor_id))
    
    # Lowercase to ensure consistency
    actor_id = actor_id.lower()

    # Vehicle groups
    if any(vehicle in actor_id for vehicle in blueprints_dict['cars']):
        return "car"
    if any(bike in actor_id for bike in blueprints_dict['motorcycles']):
        return "motorbike"
    if any(bicycle in actor_id for bicycle in blueprints_dict['bicycles']):
        return "bicycle"

    # Pedestrian groups
    if any(child_id in actor_id for child_id in pedestrians_age_gp['child']):
        return "child"
    if any(old_id in actor_id for old_id in pedestrians_age_gp['old']):
        return "old_adult"
    if any(ped_id in actor_id for ped_id in blueprints_dict['pedestrians1'] + blueprints_dict['pedestrians2']):
        return "regular_adult"

    # Fallback
    return "static"

# def convert_collisiong_log_to_actor_type_list():

#     return output

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

def get_at_risk_actors(ego_pos, ego_heading, actors, cone_angle_deg=180, max_range=60):
    """
    Returns a set of actor IDs inside a broad risk cone.
    """
    risk_dict = compute_actor_risks_in_cone(ego_pos, ego_heading, actors, 
                                            cone_angle_deg=cone_angle_deg, 
                                            max_range=max_range)
    
    print(f"[Debug] Risk Cone Actors {risk_dict}")

    return set(risk_dict.keys())


def log_episode_metrics(step_logs, episode_id,evaluation,fair_collision_rate,terminal_reward,ego,other,difficulty):
    """
    Compute aggregated ethical metrics over an episode.

    Parameters:
    - step_logs: list of per-step dictionaries from self.last_harm_metrics
    - episode_id: integer episode number

    Returns:
    - dict containing episode summary
    """
    if not step_logs:
        return {'episode': episode_id, 'error': 'No steps recorded'}
    
    # print("Step Logs")
    # print(step_logs)

    harm_avgs = [s['harm_avg'] for s in step_logs]
    harm_vars = [s['harm_var'] for s in step_logs]
    harm_maxs = [s['harm_max'] for s in step_logs]
    step_rewards = [s['step_reward'] for s in step_logs]

    summary = {
        'episode': episode_id,
        'difficulty': difficulty,
        'steps': len(step_logs),
        'evaluation':evaluation,
        'mean_harm_avg': float(np.mean(harm_avgs)),
        'mean_harm_var': float(np.mean(harm_vars)),
        'max_harm_max': float(np.max(harm_maxs)),
        'harm_max_90p': float(np.percentile(harm_maxs, 90)),
        'total_step_reward': float(np.sum(step_rewards)),
        'mean_step_reward': float(np.mean(step_rewards)),
        'fair_collision_rate':fair_collision_rate,
        'terminal_reward':terminal_reward,
        'ego':sum(ego),
        'other':sum(other),
    }

    return summary

def is_actor_in_group(actor, group):
    """
    Check if a given actor is part of a group of dicts by matching actor.id with 'id' field.

    Parameters:
    - actor: carla.Actor object
    - group: list of dicts with structure {'id': int, 'bp': blueprint}

    Returns:
    - True if actor.id matches any 'id' in the group
    """
    return any(actor.id == a['id'] for a in group if 'id' in a)

def log_episode_summary_stats_csv(input,filepath):

    # print("LOGGING")

    # pd.DataFrame(input).to_csv("harm_summary.csv", index=False)

    filepath = os.path.join(filepath, "harm.csv")

    # Convert summary dict to DataFrame
    df_new = pd.DataFrame([input])

    # If file exists, append; else, write with header
    if os.path.exists(filepath):
        df_new.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df_new.to_csv(filepath, mode='w', header=True, index=False)
    
    return



def safe_for_json(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tf.Tensor):
        try:
            return obj.numpy().item() if obj.shape == () else obj.numpy().tolist()
        except:
            return str(obj)
    elif isinstance(obj, dict):
        return {k: safe_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_for_json(i) for i in obj]
    return str(obj)