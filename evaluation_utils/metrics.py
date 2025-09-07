import numpy as np
from typing import List, Callable, Union, Sequence

def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculates the Euclidean distance between two points."""
    if point1.shape != point2.shape:
        raise ValueError(f"Points must have the same shape. Got {point1.shape} and {point2.shape}")
    return np.linalg.norm(point1 - point2)   

def calculate_task_completion(
    distances_to_goal: List[float],
    success_threshold: float = 3.0
) -> float:
    """
    Calculates Task Completion (TC).
    TC = % of episodes where the agent stops within success_threshold (e.g., 3m) of the goal. [cite: 98]

    Args:
        distances_to_goal (List[float]): A list where each element is the final distance (d_i)
                                         of the agent to the goal for an episode.
                                         This distance is typically Euclidean for the 3m check.
        success_threshold (float): The maximum distance (in meters) to be considered a success.

    Returns:
        float: Task Completion rate (between 0.0 and 1.0).
    """
    if not distances_to_goal:
        return 0.0
    
    num_episodes = len(distances_to_goal)
    successful_episodes = sum(1 for d_i in distances_to_goal if d_i <= success_threshold)
    
    return successful_episodes / num_episodes


def calculate_shortest_path_distance(
    final_geodesic_distances_to_goal: List[float]
) -> float:
    """
    Calculates Shortest-Path Distance (SPD).
    SPD = average geodesic distance (in meters) from the agent's final position to the goal. [cite: 98]

    Args:
        final_geodesic_distances_to_goal (List[float]): A list where each element is the
                                                        geodesic distance (d_i) from the agent's
                                                        final stop position to the goal for an episode.

    Returns:
        float: Average Shortest-Path Distance. Returns float('inf') if list is empty or contains inf.
    """
    if not final_geodesic_distances_to_goal:
        return float('inf')
        
    # Handle cases where some distances might be infinity (e.g., goal unreachable)
    # The sum will be inf if any element is inf.
    total_distance = sum(final_geodesic_distances_to_goal)
    num_episodes = len(final_geodesic_distances_to_goal)

    if total_distance == float('inf'):
        return float('inf')
    return total_distance / num_episodes


def dynamic_time_warping_cost(
    path1: Union[np.ndarray, List[np.ndarray]], # Sequence of N-dim points or feature vectors
    path2: Union[np.ndarray, List[np.ndarray]], # Sequence of N-dim points or feature vectors
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance
) -> float:
    """
    Computes the Dynamic Time Warping (DTW) cost between two paths (sequences of points/vectors).

    Args:
        path1 (Union[np.ndarray, List[np.ndarray]]): The first path, as a list of NumPy arrays
                                                     or a 2D NumPy array (num_points1, dim).
        path2 (Union[np.ndarray, List[np.ndarray]]): The second path, similar format.
        distance_metric (Callable): A function that takes two points (NumPy arrays)
                                    and returns their distance (float). Defaults to Euclidean.

    Returns:
        float: The DTW cost.
    """
    p1 = np.asarray(path1, dtype=np.float32)
    p2 = np.asarray(path2, dtype=np.float32)

    if p1.ndim == 1: p1 = p1.reshape(-1, 1) # Handle 1D paths (e.g. sequence of scalars)
    if p2.ndim == 1: p2 = p2.reshape(-1, 1)
    if p1.ndim != 2 or p2.ndim != 2:
        raise ValueError("Paths must be convertible to 2D NumPy arrays (num_points, dim).")
    if p1.shape[1] != p2.shape[1] and p1.size > 0 and p2.size > 0: # Allow empty paths
        raise ValueError(f"Points in paths must have the same dimension. Got {p1.shape[1]} and {p2.shape[1]}")

    len1, len2 = len(p1), len(p2)
    if len1 == 0 or len2 == 0: # If one path is empty, cost is sum of distances from empty to other points (effectively infinite if not handled by metric)
                               # For VLN, often an empty path implies failure; DTW might not be meaningful.
                               # Standard DTW cost for empty path is often considered infinite or very large.
                               # Let's return inf if one is empty and other isn't. If both empty, 0.
        if len1 == 0 and len2 == 0: return 0.0
        return float('inf')


    # Initialize DP table with infinities
    dtw_matrix = np.full((len1 + 1, len2 + 1), float('inf'))
    dtw_matrix[0, 0] = 0.0 # Cost of aligning two empty paths is 0

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = distance_metric(p1[i-1], p2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # Insertion
                                          dtw_matrix[i, j-1],    # Deletion
                                          dtw_matrix[i-1, j-1])  # Match/Substitution
    
    return dtw_matrix[len1, len2]


def calculate_ndtw(
    agent_paths: List[Union[np.ndarray, List[np.ndarray]]], # List of agent trajectories
    expert_paths: List[Union[np.ndarray, List[np.ndarray]]], # List of expert trajectories
    expert_path_lengths: List[float], # Precomputed lengths (L_i) of expert paths
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance
) -> float:
    """
    Calculates normalized Dynamic Time Warping (nDTW).
    nDTW = (1/N) * sum(exp(-DTW_i / L_i)) [cite: 98, 99]
    where DTW_i is the DTW cost between agent path i and expert path i,
    and L_i is the length of the expert path i.

    Args:
        agent_paths (List[Union[np.ndarray, List[np.ndarray]]]): List of agent trajectories.
            Each trajectory is a sequence of N-dim points.
        expert_paths (List[Union[np.ndarray, List[np.ndarray]]]): List of expert trajectories.
        expert_path_lengths (List[float]): List of precomputed lengths (L_i) of the expert paths.
                                          Length calculation depends on path type (e.g., sum of
                                          Euclidean distances for coordinate paths, or sum of edge
                                          weights for graph node paths).
        distance_metric (Callable): Metric used for DTW cost calculation between points.

    Returns:
        float: Normalized Dynamic Time Warping score (between 0.0 and 1.0).
    """
    if not (len(agent_paths) == len(expert_paths) == len(expert_path_lengths)):
        raise ValueError("agent_paths, expert_paths, and expert_path_lengths must have the same number of elements.")
    if not agent_paths:
        return 0.0

    num_episodes = len(agent_paths)
    total_ndtw_score = 0.0

    for i in range(num_episodes):
        agent_path_i = np.asarray(agent_paths[i], dtype=np.float32)
        expert_path_i = np.asarray(expert_paths[i], dtype=np.float32)
        expert_len_L_i = expert_path_lengths[i]

        if expert_len_L_i == 0: # Avoid division by zero; if expert path has no length, nDTW is ill-defined or 0.
            # If agent path also has no length (e.g. start=goal), could be 1.0.
            # If agent path has length, and expert has 0, this is problematic.
            # For VLN, L_i > 0 is generally expected if start != goal.
            # Let's assume nDTW is 0 if L_i is 0 and agent path has content, or 1 if both are empty/trivial.
            if len(agent_path_i) == 0 or (len(agent_path_i) == 1 and len(expert_path_i) == 1 and np.allclose(agent_path_i[0], expert_path_i[0])): # Both trivial/same start
                ndtw_i = 1.0
            else:
                ndtw_i = 0.0 # Or handle as error, or skip episode
        else:
            dtw_cost_i = dynamic_time_warping_cost(agent_path_i, expert_path_i, distance_metric)
            if dtw_cost_i == float('inf'): # If DTW cost is infinite (e.g., one path empty)
                ndtw_i = 0.0
            else:
                ndtw_i = np.exp(-dtw_cost_i / expert_len_L_i)
        
        total_ndtw_score += ndtw_i
    
    return total_ndtw_score / num_episodes


# --- Optional: Success weighted by Path Length (SPL) ---
# While not explicitly requested as TC, SPD, nDTW from Mem4Nav's main metrics,
# SPL is a very common and important VLN metric.
def calculate_spl(
    distances_to_goal: List[float], # Euclidean distances for success check
    agent_path_lengths: List[float],
    expert_path_lengths: List[float], # Geodesic lengths of shortest paths to goal
    success_threshold: float = 3.0
) -> float:
    """
    Calculates Success weighted by Path Length (SPL).
    SPL = (1/N) * sum(S_i * (L_i / max(P_i, L_i)))
    where S_i is 1 if successful, 0 otherwise.
    L_i is the length of the expert (shortest) path.
    P_i is the length of the agent's path.

    Args:
        distances_to_goal (List[float]): Euclidean distance from agent's stop to goal.
        agent_path_lengths (List[float]): Lengths of the paths taken by the agent.
        expert_path_lengths (List[float]): Lengths of the shortest paths (expert paths).
        success_threshold (float): Threshold for task completion success.

    Returns:
        float: SPL score.
    """
    if not (len(distances_to_goal) == len(agent_path_lengths) == len(expert_path_lengths)):
        raise ValueError("All input lists must have the same number of elements for SPL.")
    if not distances_to_goal:
        return 0.0

    num_episodes = len(distances_to_goal)
    total_spl_score = 0.0

    for i in range(num_episodes):
        S_i = 1.0 if distances_to_goal[i] <= success_threshold else 0.0
        L_i = expert_path_lengths[i]
        P_i = agent_path_lengths[i]

        if L_i == 0: # Goal is at start or path is trivial
            # If agent also took zero length path and S_i=1, SPL term is 1.
            # If agent took non-zero path P_i > 0 and L_i=0, then max(P_i,L_i)=P_i. L_i/P_i = 0. SPL term is 0 unless S_i=1 AND P_i=0.
            # Standard definition: if L_i=0, and P_i=0 and S_i=1, SPL_i = 1. Otherwise SPL_i = 0 if L_i=0.
            spl_i = S_i if P_i == 0 else 0.0
        else:
            spl_i = S_i * (L_i / max(P_i, L_i))
        
        total_spl_score += spl_i
        
    return total_spl_score / num_episodes


if __name__ == '__main__':
    print("--- Testing VLN Metrics ---")

    # Test Data
    # Episode 1: Success, good path
    dist_1 = 2.0
    geodesic_dist_1 = 2.5
    agent_path_1 = np.array([[0,0], [1,0], [2,0], [2,1]])
    expert_path_1 = np.array([[0,0], [1,0], [2,0], [2,1]])
    expert_len_1 = 3.0 # (1+1+1)
    agent_len_1 = 3.0

    # Episode 2: Failure, far from goal
    dist_2 = 10.0
    geodesic_dist_2 = 10.0
    agent_path_2 = np.array([[0,0], [1,1]])
    expert_path_2 = np.array([[0,0], [3,0], [3,3], [0,3], [0,0]]) # Longer expert path
    expert_len_2 = 12.0 # (3+3+3+3)
    agent_len_2 = np.sqrt(2)

    # Episode 3: Success, but inefficient path
    dist_3 = 1.0
    geodesic_dist_3 = 1.5
    agent_path_3 = np.array([[0,0], [0,1], [0,2], [1,2], [2,2], [2,1], [2,0]]) # Long agent path
    expert_path_3 = np.array([[0,0], [1,0], [2,0]]) # Short expert path
    expert_len_3 = 2.0
    agent_len_3 = 6.0

    # Episode 4: Expert path length is 0 (start is goal)
    dist_4_succ = 0.1 # Successful stop at start
    geodesic_dist_4_succ = 0.0
    agent_path_4_succ = np.array([[0,0]])
    expert_path_4 = np.array([[0,0]])
    expert_len_4 = 0.0
    agent_len_4_succ = 0.0

    dist_4_fail = 5.0 # Agent moved away
    geodesic_dist_4_fail = 5.0
    agent_path_4_fail = np.array([[0,0], [1,0]])
    agent_len_4_fail = 1.0


    all_distances_to_goal = [dist_1, dist_2, dist_3, dist_4_succ, dist_4_fail]
    all_geodesic_distances = [geodesic_dist_1, geodesic_dist_2, geodesic_dist_3, geodesic_dist_4_succ, geodesic_dist_4_fail]
    all_agent_paths = [agent_path_1, agent_path_2, agent_path_3, agent_path_4_succ, agent_path_4_fail]
    all_expert_paths = [expert_path_1, expert_path_2, expert_path_3, expert_path_4, expert_path_4]
    all_expert_lengths = [expert_len_1, expert_len_2, expert_len_3, expert_len_4, expert_len_4]
    all_agent_lengths = [agent_len_1, agent_len_2, agent_len_3, agent_len_4_succ, agent_len_4_fail]

    # Test TC
    tc = calculate_task_completion(all_distances_to_goal, success_threshold=3.0)
    print(f"\nTask Completion (TC): {tc:.4f}") # Expected: (1+0+1+1+0)/5 = 3/5 = 0.6
    assert np.isclose(tc, 3.0/5.0)

    # Test SPD
    spd = calculate_shortest_path_distance(all_geodesic_distances)
    print(f"Shortest-Path Distance (SPD): {spd:.4f}") # Expected: (2.5+10+1.5+0+5)/5 = 19/5 = 3.8
    assert np.isclose(spd, 19.0/5.0)

    # Test DTW cost (helper)
    dtw_1 = dynamic_time_warping_cost(agent_path_1, expert_path_1)
    print(f"DTW cost for ep1 (identical paths): {dtw_1:.4f}")
    assert np.isclose(dtw_1, 0.0)
    
    dtw_2 = dynamic_time_warping_cost(agent_path_2, expert_path_2)
    print(f"DTW cost for ep2 (different paths): {dtw_2:.4f}")
    assert dtw_2 > 0

    # Test nDTW
    ndtw = calculate_ndtw(all_agent_paths, all_expert_paths, all_expert_lengths)
    print(f"Normalized Dynamic Time Warping (nDTW): {ndtw:.4f}")
    # For ep1: exp(0/3) = 1.0
    # For ep4_succ: exp(0/0) -> special handling in func, should be 1.0
    # For ep4_fail: DTW(agent_moved, expert_at_start) / 0 -> special handling, should be 0.0
    assert 0.0 <= ndtw <= 1.0

    # Test SPL
    spl = calculate_spl(all_distances_to_goal, all_agent_lengths, all_expert_lengths, success_threshold=3.0)
    print(f"Success weighted by Path Length (SPL): {spl:.4f}")
    # Ep1: S=1, L=3, P=3 -> 1 * (3/3) = 1
    # Ep2: S=0 -> 0
    # Ep3: S=1, L=2, P=6 -> 1 * (2/6) = 1/3
    # Ep4_succ: S=1, L=0, P=0 -> 1
    # Ep4_fail: S=0, L=0, P=1 -> 0
    # Total SPL = (1 + 0 + 1/3 + 1 + 0) / 5 = (2 + 1/3) / 5 = (7/3) / 5 = 7/15
    assert np.isclose(spl, 7.0/15.0)
    
    print("\nmetrics.py tests completed.")
