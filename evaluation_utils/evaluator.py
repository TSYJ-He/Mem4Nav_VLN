import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from tqdm import tqdm
import networkx as nx 
import os 
from PIL import Image 
try:
    from .metrics import (
        calculate_task_completion,
        calculate_shortest_path_distance,
        calculate_ndtw,
        calculate_spl,
        euclidean_distance
    )
    from ..agents.base_vln_agent import BaseVLNAgent 
except ImportError:
    print("Warning: Evaluator using placeholders for some imported metric/agent modules due to import error.")

    def calculate_task_completion(*args, **kwargs) -> float: return 0.0
    def calculate_shortest_path_distance(*args, **kwargs) -> float: return float('inf')
    def calculate_ndtw(*args, **kwargs) -> float: return 0.0
    def calculate_spl(*args, **kwargs) -> float: return 0.0
    def euclidean_distance(p1,p2): return np.linalg.norm(np.array(p1)-np.array(p2))   
    class BaseVLNAgent(torch.nn.Module):   
        def __init__(self, *args, **kwargs): super().__init__()
        def reset(self, *args, **kwargs): pass
        def step(self, *args, **kwargs) -> str: return "stop"


class EnvironmentGraph:
    """
    Wraps the navigation graph for an environment (e.g., Touchdown, Map2Seq).
    It uses a networkx graph internally for pathfinding and connectivity.
    Node IDs are expected to be panoids (strings).
    Edge weights should represent distances between connected panoids.
    Node attributes should include 'position' (np.ndarray for [x,y,z]).
    """
    def __init__(self,
                 nx_graph: Optional[nx.Graph] = None,
                 node_positions_xyz: Optional[Dict[str, np.ndarray]] = None):
        """
        Args:
            nx_graph (Optional[nx.Graph]): A pre-loaded networkx graph.
                Nodes are panoids. Edges should have a 'weight' attribute for distance.
                Nodes should ideally have a 'position' attribute [x,y,z].
            node_positions_xyz (Optional[Dict[str, np.ndarray]]):
                A dictionary mapping panoid (str) to its [x,y,z] coordinates (np.ndarray).
                This is used if positions are not directly attributes of nx_graph nodes.
        """
        self.graph = nx_graph if nx_graph is not None else nx.Graph()
        self.node_positions = node_positions_xyz if node_positions_xyz is not None else {}

        if not self.node_positions and self.graph.nodes():
            # Try to populate from graph node attributes if available
            for node_id, data in self.graph.nodes(data=True):
                if 'position' in data and isinstance(data['position'], (np.ndarray, list, tuple)):
                    self.node_positions[str(node_id)] = np.array(data['position'])
                elif 'x' in data and 'y' in data and 'z' in data: # Common alternative
                    self.node_positions[str(node_id)] = np.array([data['x'], data['y'], data['z']])


    def get_node_position(self, panoid: str) -> Optional[np.ndarray]:
        """Retrieves the XYZ position of a given panoid."""
        pos = self.node_positions.get(panoid)
        if pos is None and panoid in self.graph: # Try graph attributes directly
            node_data = self.graph.nodes[panoid]
            if 'position' in node_data:
                pos = np.array(node_data['position'])
                self.node_positions[panoid] = pos # Cache it
        return pos

    def get_valid_transitions(self, panoid: str) -> Dict[float, str]:
        """
        Returns a dictionary of possible transitions from a panoid.
        Keys are headings (degrees, relative to current view if applicable, or absolute world yaw)
        to take to reach the Value (neighbor panoid).
        This is highly specific to how the graph encodes links (e.g., StreetLearn links).
        Needs to be implemented based on the actual graph format.
        """
        if panoid not in self.graph:
            return {}
        
        transitions = {}
        # Example: If graph edges have 'action_heading_degrees' attribute
        for u, v, data in self.graph.edges(panoid, data=True):   
            neighbor = v if u == panoid else u
            #
            if data.get('action') == 'forward': # This is hypothetical edge data
                transitions[data.get('heading_required', 0.0)] = neighbor   
        
        if not transitions and list(self.graph.neighbors(panoid)): # Fallback if no heading info
             # This fallback is poor, just takes first neighbor for 'forward'
             transitions[0.0] = list(self.graph.neighbors(panoid))[0]   
        return transitions


    def shortest_path_distance(self, panoid_start: str, panoid_goal: str) -> float:
        """Calculates geodesic distance between two panoids using Dijkstra on the graph."""
        if not self.graph or panoid_start not in self.graph or panoid_goal not in self.graph:   
            return float('inf')
        try:
            # Assumes edges have a 'weight' attribute representing distance
            return float(nx.dijkstra_path_length(self.graph, panoid_start, panoid_goal, weight='weight'))   
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')

    def shortest_path_panoids(self, panoid_start: str, panoid_goal: str) -> List[str]:
        """Returns the sequence of panoids in the shortest path."""
        if not self.graph or panoid_start not in self.graph or panoid_goal not in self.graph:   
            return []
        try:
            return nx.dijkstra_path(self.graph, panoid_start, panoid_goal, weight='weight')   
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []


class Evaluator:
    def __init__(self,
                 agent: BaseVLNAgent,
                 dataloader: torch.utils.data.DataLoader,
                 env_graph: EnvironmentGraph,
                 eval_config: Dict[str, Any],
                 device: torch.device,

                 panorama_store_path: Optional[str] = None):
        self.agent = agent
        self.dataloader = dataloader
        self.env_graph = env_graph
        self.config = eval_config
        self.device = device
        self.panorama_store_path = panorama_store_path 

        self.max_episode_steps = self.config.get('max_episode_steps', 100)
        self.success_threshold_m = self.config.get('success_threshold_m', 3.0)

        control_config = self.config.get('control_params', {})
        self.default_step_distance = control_config.get('default_step_distance', 1.0) # Used if simulating XYZ from panos
        self.turn_angle_rad = control_config.get('default_turn_angle_rad', np.deg2rad(30.0))
        self.logger = self.config.get('logger', None) # Optional logger, can be None


    def _simulate_action_on_graph(self, current_panoid: str, current_heading_rad: float, action: str) \
            -> Tuple[str, float, np.ndarray]:
        """
        Simulates taking a discrete action using the EnvironmentGraph.
        Returns new (panoid, heading_rad, pose_xyz).
        This needs to be implemented robustly based on how the graph encodes transitions.
        """
        next_panoid = current_panoid
        next_heading_rad = current_heading_rad
        current_pos_xyz = self.env_graph.get_node_position(current_panoid)
        if current_pos_xyz is None: # Should not happen if graph is well-formed
            self.logger.error(f"Position for current panoid {current_panoid} not found in env_graph! Using [0,0,0].")
            current_pos_xyz = np.array([0.,0.,0.])
        next_pos_xyz = current_pos_xyz.copy()


        
        valid_transitions = self.env_graph.get_valid_transitions(current_panoid) # This method needs to be robust

        if action == "forward":
            # This needs to find the neighbor in the current_heading_rad direction.
            # The `get_valid_transitions` should ideally return this mapping.
            # For simplicity, if `valid_transitions` just gives one forward option:
            target_panoid_for_fwd = valid_transitions.get(np.rad2deg(current_heading_rad), # Exact heading match
                                                        valid_transitions.get(0.0, # Fallback for simple 'forward'
                                                                            current_panoid)) 
            # This logic is highly dependent on how your graph stores connections and headings.
            # If it's a grid or fixed step:
            # dx = self.default_step_distance * np.cos(current_heading_rad)
            # dy = self.default_step_distance * np.sin(current_heading_rad)
            # next_pos_xyz = current_pos_xyz + np.array([dx, dy, 0])
            # next_panoid would then be found by finding the closest graph node to next_pos_xyz.
            # This is complex. For discrete pano graphs, "forward" usually moves to a specific linked pano.
            
            if target_panoid_for_fwd != current_panoid:
                next_panoid = target_panoid_for_fwd
                new_pos = self.env_graph.get_node_position(next_panoid)
                if new_pos is not None: next_pos_xyz = new_pos
                # Heading might also change if moving to a new pano that has a default entry heading
        elif action == "turn_left":
            next_heading_rad = (current_heading_rad + self.turn_angle_rad + 2 * np.pi) % (2 * np.pi)
        elif action == "turn_right":
            next_heading_rad = (current_heading_rad - self.turn_angle_rad + 2 * np.pi) % (2 * np.pi)
        elif action == "stop":
            pass # State remains the same
        # elif action == "turn_around": # Mem4Nav paper backbones use 4-way actions
        #     next_heading_rad = (current_heading_rad + np.pi + 2 * np.pi) % (2 * np.pi)
        else:
            self.logger.warning(f"Unknown action '{action}' received in simulation. No state change.")

        # Ensure heading is wrapped
        next_heading_rad = (next_heading_rad + np.pi) % (2 * np.pi) - np.pi
        
        # If next_panoid changed, update position. Otherwise, position is same (only heading changed).
        if next_panoid != current_panoid:
            pos = self.env_graph.get_node_position(next_panoid)
            if pos is not None:
                next_pos_xyz = pos
            else: # Should not happen if graph is valid
                self.logger.error(f"Position for next panoid {next_panoid} not found! Using previous position.")
                next_panoid = current_panoid # Revert panoid change if no position found
                next_pos_xyz = current_pos_xyz


        return next_panoid, next_heading_rad, next_pos_xyz


    def _get_observation_for_agent(self, current_panoid: str, current_heading_rad: float,
                                   current_pose_xyz: np.ndarray, episode_data: Dict) -> Dict[str, Any]:
        """
        Constructs the observation dictionary for the agent's step method.
        MUST load actual RGB image/features for the current_panoid.
        """
        rgb_image_tensor: Optional[torch.Tensor] = None
        if self.panorama_store_path:
            # Example: panoramas are JPEGs named {panoid}.jpg
            # This is highly dataset-specific. For Touchdown, images are from StreetLearn.
            # Path construction and loading needs to be robust.
            # E.g. for StreetLearn, panoids can be complex.
            # This is a MAJOR placeholder.
            image_path = os.path.join(self.panorama_store_path, f"{current_panoid}.jpg") # Simplistic
            try:
                if os.path.exists(image_path):
                    pil_image = Image.open(image_path).convert('RGB')

                    # rgb_image_tensor = T.ToTensor()(pil_image) # Example, needs correct transform for agent
                    rgb_image_for_agent = pil_image # Pass PIL to PerceptionModule
                else:
                    self.logger.warning(f"Image for panoid {current_panoid} not found at {image_path}. Using mock.")
                    rgb_image_for_agent = Image.new('RGB', (320,240), 'grey') # Mock PIL
            except Exception as e:
                self.logger.error(f"Error loading image {image_path}: {e}. Using mock.")
                rgb_image_for_agent = Image.new('RGB', (320,240), 'grey') # Mock PIL
        else:
            self.logger.warning("Panorama store path not configured. Using mock RGB for agent observation.")
            rgb_image_for_agent = Image.new('RGB', (320,240), 'grey') # Mock PIL

        return {
            'rgb': rgb_image_for_agent,
            'current_pose': current_pose_xyz.copy(), # Current [x,y,z]
            'current_panoid': current_panoid,
            'current_heading_rad': current_heading_rad,
            'instruction_text': episode_data.get('instruction_text'),
            'episode_id': episode_data.get('id')

        }

    def run_single_episode(self, episode_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Runs the agent for a single episode and collects results."""
        self.agent.reset(instruction_data=episode_data)
        episode_id = episode_data.get('id', 'unknown_episode')

        initial_panoid = episode_data.get('initial_panoid')
        if not initial_panoid:
            self.logger.error(f"Episode {episode_id} missing 'initial_panoid'. Skipping episode.")
            return None
            
        current_heading_rad = np.deg2rad(float(episode_data.get('initial_heading_deg', 0.0)))
        current_panoid: str = initial_panoid
        
        current_pose_xyz: Optional[np.ndarray] = episode_data.get('initial_pose_xyz')
        if current_pose_xyz is None:
            current_pose_xyz = self.env_graph.get_node_position(current_panoid)
        if current_pose_xyz is None:
            self.logger.error(f"Episode {episode_id}: Cannot get initial XYZ for panoid {current_panoid}. Skipping.")
            return None
        current_pose_xyz = np.array(current_pose_xyz)


        agent_trajectory_xyz: List[np.ndarray] = [current_pose_xyz.copy()]
        agent_trajectory_panoids: List[str] = [current_panoid]

        for step_count in range(self.max_episode_steps):
            observation = self._get_observation_for_agent(
                current_panoid, current_heading_rad, current_pose_xyz, episode_data
            )
            
            action_str = self.agent.step(observation)

            if action_str == "stop":
                break

            current_panoid, current_heading_rad, current_pose_xyz = \
                self._simulate_action_on_graph(current_panoid, current_heading_rad, action_str)
            
            agent_trajectory_xyz.append(current_pose_xyz.copy())
            if not agent_trajectory_panoids or agent_trajectory_panoids[-1] != current_panoid:
                 agent_trajectory_panoids.append(current_panoid)
        
        final_agent_pose_xyz = agent_trajectory_xyz[-1]
        final_agent_panoid = agent_trajectory_panoids[-1]
        
        goal_panoid = episode_data.get('target_panoid', episode_data.get('goal_panoid')) # Target from dataset
        goal_pose_xyz_gt = episode_data.get('goal_pose_xyz') # If dataset provides explicit XYZ goal
        if goal_pose_xyz_gt is None and goal_panoid:
            goal_pose_xyz_gt = self.env_graph.get_node_position(goal_panoid)
        
        if goal_pose_xyz_gt is None:
            self.logger.error(f"Episode {episode_id}: Goal position (XYZ or from panoid) could not be determined. Metrics might be incorrect.")
            # Fallback: use agent's final position as goal, resulting in 0 distance (bad for metrics)
            # Or, this episode should be marked as error. For now, to avoid crash:
            distance_to_goal_euclidean = 0.0 
            geodesic_dist_to_goal = 0.0
            # This indicates a data or graph problem.
        else:
            goal_pose_xyz_gt = np.array(goal_pose_xyz_gt)
            distance_to_goal_euclidean = euclidean_distance(final_agent_pose_xyz, goal_pose_xyz_gt)
            if goal_panoid: # Ensure goal_panoid corresponds to goal_pose_xyz_gt if both exist
                geodesic_dist_to_goal = self.env_graph.shortest_path_distance(final_agent_panoid, goal_panoid)
            else: # No goal panoid, try to find closest graph node to goal_pose_xyz_gt for SPD
   
                self.logger.warning(f"Episode {episode_id}: No goal_panoid for SPD. Using Euclidean distance as fallback for geodesic.")
                geodesic_dist_to_goal = distance_to_goal_euclidean


        agent_path_len_xyz = calculate_euclidean_path_length(agent_trajectory_xyz) # # type: ignore
        
        expert_path_panoids = episode_data.get('route_panoids', []) # Expert path from dataset
        expert_path_len_geodesic = 0.0
        if expert_path_panoids and len(expert_path_panoids) > 0:
            # Calculate true geodesic length of expert path
            current_expert_pano = expert_path_panoids[0]
            if self.env_graph.get_node_position(current_expert_pano) is None and episode_data.get('initial_pose_xyz') is not None:
                # If start pano has no pos, but episode has initial_pose_xyz, try to find closest node
                # This is for robustness; ideally, all panos in routes exist in graph with positions.
                pass # Advanced: map initial_pose_xyz to a graph node

            for i in range(len(expert_path_panoids) - 1):
                dist = self.env_graph.shortest_path_distance(expert_path_panoids[i], expert_path_panoids[i+1])
                if dist == float('inf'):
                    # Handle unreachable segments in expert path (should ideally not happen in clean data)
                    # Fallback: estimate with Euclidean if positions available, or add large penalty
                    p1 = self.env_graph.get_node_position(expert_path_panoids[i])
                    p2 = self.env_graph.get_node_position(expert_path_panoids[i+1])
                    if p1 is not None and p2 is not None:
                        dist = euclidean_distance(p1, p2)
                    else:
                        dist = self.config.get('large_dist_for_inf_path_segment', 50.0) # Large penalty
                    self.logger.warning(f"Segment {expert_path_panoids[i]}->{expert_path_panoids[i+1]} in expert path for {episode_id} is unreachable on graph. Using fallback dist: {dist:.2f}")
                expert_path_len_geodesic += dist
        
        # For nDTW, expert path needs to be in XYZ for comparison with agent's XYZ path
        expert_path_xyz_for_dtw: List[np.ndarray] = []
        if expert_path_panoids:
            for panoid in expert_path_panoids:
                pos = self.env_graph.get_node_position(panoid)
                if pos is not None:
                    expert_path_xyz_for_dtw.append(pos)
        if not expert_path_xyz_for_dtw and expert_path_panoids: # If conversion failed but panoids exist
            self.logger.warning(f"Could not convert expert panoid path to XYZ for nDTW for ep {episode_id}. nDTW might be 0.")
            expert_path_xyz_for_dtw.append(agent_trajectory_xyz[0]) # Avoid crash, use start


        return {
            "id": episode_id,
            "agent_trajectory_xyz": [p.tolist() for p in agent_trajectory_xyz],
            "agent_trajectory_panoids": agent_trajectory_panoids,
            "final_agent_pose_xyz": final_agent_pose_xyz.tolist(),
            "goal_pose_xyz": goal_pose_xyz_gt.tolist() if goal_pose_xyz_gt is not None else None,
            "expert_path_panoids": expert_path_panoids,
            "expert_path_xyz_for_dtw": [p.tolist() for p in expert_path_xyz_for_dtw],
            "distance_to_goal_euclidean": float(distance_to_goal_euclidean),
            "geodesic_dist_to_goal": float(geodesic_dist_to_goal),
            "agent_path_length": float(agent_path_len_xyz),
            "expert_path_length_geodesic": float(expert_path_len_geodesic) # Used for SPL and as L_i for nDTW
        }

    def evaluate(self) -> Dict[str, float]:
        self.agent.eval()
        all_episode_results: List[Dict[str, Any]] = []
        
        self.logger.info(f"Starting evaluation on {len(self.dataloader.dataset)} episodes using {self.device}...")
        for episode_data_batch in tqdm(self.dataloader, desc="Evaluating Episodes"):
            # Assuming dataloader batch_size=1 for evaluation, so episode_data_batch is one episode.
            # If dataloader has batch_size > 1, need to iterate through the batch here.
            # For now, assume episode_data_batch is a single episode dict.
            # If it's a dict of lists/tensors (batched), unbatch it first.
            # This check is a bit fragile. Proper batch handling is better.
            is_batched = isinstance(episode_data_batch, dict) and \
                         len(episode_data_batch) > 0 and \
                         isinstance(next(iter(episode_data_batch.values())), (list, torch.Tensor)) and \
                         len(next(iter(episode_data_batch.values()))) > 1

            if is_batched: # This is a rough check
                num_in_batch = len(episode_data_batch[next(iter(episode_data_batch))])
                for i in range(num_in_batch):
                    single_episode_data = {k: v[i] for k, v in episode_data_batch.items()}
                    ep_result = self.run_single_episode(single_episode_data)
                    if ep_result: all_episode_results.append(ep_result)
            else: # Assumed to be a single episode dictionary
                ep_result = self.run_single_episode(episode_data_batch)
                if ep_result: all_episode_results.append(ep_result)


        if not all_episode_results:
            self.logger.warning("No episodes were successfully processed during evaluation.")
            return {"TC": 0.0, "SPD": float('inf'), "nDTW": 0.0, "SPL": 0.0, "num_episodes": 0.0}

        # Collect data for metric functions
        distances_for_tc_spl = [res['distance_to_goal_euclidean'] for res in all_episode_results]
        geodesic_distances_for_spd = [res['geodesic_dist_to_goal'] for res in all_episode_results]
        
        agent_paths_xyz_for_ndtw = [np.array(res['agent_trajectory_xyz']) for res in all_episode_results]
        expert_paths_xyz_for_ndtw = [np.array(res['expert_path_xyz_for_dtw']) for res in all_episode_results]
        
        expert_lengths_geodesic = [res['expert_path_length_geodesic'] for res in all_episode_results]
        agent_lengths_xyz = [res['agent_path_length'] for res in all_episode_results]

        metrics_results: Dict[str, float] = {}
        metrics_results["TC"] = calculate_task_completion(distances_for_tc_spl, self.success_threshold_m)
        metrics_results["SPD"] = calculate_shortest_path_distance(geodesic_distances_for_spd)
        # For nDTW, expert_path_lengths should be the length of the actual path used for DTW comparison
        # If expert_path_xyz_for_dtw is used, its length should be used.
        # The paper uses L_i as expert path length for nDTW.
        expert_lengths_for_ndtw_calc = [calculate_euclidean_path_length(p) for p in expert_paths_xyz_for_ndtw]
        metrics_results["nDTW"] = calculate_ndtw(agent_paths_xyz_for_ndtw, expert_paths_xyz_for_ndtw, expert_lengths_for_ndtw_calc)
        metrics_results["SPL"] = calculate_spl(distances_for_tc_spl, agent_lengths_xyz, expert_lengths_geodesic, self.success_threshold_m)
        
        metrics_results["num_episodes"] = float(len(all_episode_results))
        
        self.logger.info("\n--- Evaluation Summary ---")
        for metric_name, value in metrics_results.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")
        
        return metrics_results

if __name__ == '__main__':
    print("--- Conceptual Test for Evaluator (Revised) ---")
    device = torch.device('cpu')
    
    graph_nx = nx.Graph()
    nodes_data = {
        "start_node": {"position": np.array([0.,0.,0.]), "name": "start"},
        "waypoint1": {"position": np.array([1.,0.,0.]), "name": "waypoint1"},
        "waypoint2": {"position": np.array([2.,0.,0.]), "name": "waypoint2"},
        "goal_node": {"position": np.array([3.,0.,0.]), "name": "goal"},
        "other_node": {"position": np.array([0.,1.,0.]), "name": "other"}
    }
    for node_id, data in nodes_data.items():
        graph_nx.add_node(node_id, **data)
    
    edges_data = [
        ("start_node", "waypoint1", {"weight": 1.0, "action": "forward", "heading_required": 0.0}),
        ("waypoint1", "waypoint2", {"weight": 1.0, "action": "forward", "heading_required": 0.0}),
        ("waypoint2", "goal_node", {"weight": 1.0, "action": "forward", "heading_required": 0.0}),
        ("start_node", "other_node", {"weight": 1.0, "action": "forward", "heading_required": 90.0}) # Example other path
    ]
    for u,v,data in edges_data:
        graph_nx.add_edge(u,v,**data)

    mock_env_graph_revised = EnvironmentGraph(nx_graph=graph_nx)

    # Mock Agent
    class PathFollowAgent(BaseVLNAgent):
        def __init__(self, config, device): super().__init__(config,device); self.path_to_follow = []; self.path_idx = 0
        def reset(self, instruction_data: Optional[Dict[str, Any]] = None):
            self.path_to_follow = instruction_data.get('expert_path_panoids', [])   
            self.path_idx = 0
            print(f"PathFollowAgent: Reset. Path: {self.path_to_follow}")
        def step(self, observation: Dict[str, Any]) -> str:
            current_pano = observation['current_panoid']
            if not self.path_to_follow or self.path_idx >= len(self.path_to_follow): return "stop"
            
            target_pano_in_path = self.path_to_follow[self.path_idx]
            if current_pano == target_pano_in_path:
                self.path_idx += 1 # Reached current target in path, advance
                if self.path_idx >= len(self.path_to_follow): return "stop" # Reached end of path
            

            if self.path_idx < len(self.path_to_follow) and current_pano != self.path_to_follow[-1]:
                 # This naive "forward" relies on _simulate_action_on_graph being smart or graph simple.
                 return "forward" 
            return "stop"

    mock_agent_instance = PathFollowAgent({}, device)

    # Mock Dataloader
    mock_episodes_for_eval = [
        {
            'id': 'eval_ep1', 'instruction_text': 'Follow path to goal.', 
            'initial_panoid': 'start_node', 'initial_heading_deg': 0.0, 
            'goal_panoid': 'goal_node', 
            'expert_path_panoids': ['start_node', 'waypoint1', 'waypoint2', 'goal_node']
            # initial_pose_xyz and goal_pose_xyz will be looked up from graph
        },
         { # Episode where agent might stop early or path is shorter
            'id': 'eval_ep2', 'instruction_text': 'Go to waypoint1.', 
            'initial_panoid': 'start_node', 'initial_heading_deg': 0.0,
            'goal_panoid': 'waypoint1',
            'expert_path_panoids': ['start_node', 'waypoint1']
        }
    ]
    class MockEvalDataLoader(torch.utils.data.DataLoader):   
        def __init__(self, data, **kwargs): self.dataset = data # Dataset should be a list of dicts
        def __iter__(self): return iter(self.dataset)
        def __len__(self): return len(self.dataset)


    mock_eval_dataloader = MockEvalDataLoader(mock_episodes_for_eval)
    
    eval_config_dict = {
        'max_episode_steps': 5, 'success_threshold_m': 1.0,
        'control_params': {'default_step_distance': 1.0, 'default_turn_angle_rad': np.deg2rad(90)}
    }


    def mock_get_neighbors_for_test(panoid: str, heading: float, path_context: List[str]):

        if panoid in path_context:
            current_idx_in_path = path_context.index(panoid)
            if current_idx_in_path + 1 < len(path_context):
                return {"forward": path_context[current_idx_in_path + 1], "stop": panoid}
        return {"forward": panoid, "stop": panoid} # Stay if not on path or at end


    evaluator_instance_revised = Evaluator(
        mock_agent_instance, mock_eval_dataloader, mock_env_graph_revised, 
        eval_config_dict, device, panorama_store_path="./dummy_pano_store" # Mock path
    )

    os.makedirs("./dummy_pano_store", exist_ok=True)
    for node_id in nodes_data.keys():
        try: Image.new('RGB',(64,64)).save(f"./dummy_pano_store/{node_id}.jpg")
        except: pass


    print("\nRunning evaluation with revised Evaluator...")
    try:
        results_revised = evaluator_instance_revised.evaluate()
        print("\nFinal Evaluation Metrics from Revised Mock Run:")
        for k, v in results_revised.items():
            print(f"  {k}: {v:.4f}")

        assert 'TC' in results_revised
    except Exception as e:
        print(f"Error during revised Evaluator test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import shutil
        if os.path.exists("./dummy_pano_store"): shutil.rmtree("./dummy_pano_store")


    print("\nEvaluator (revised) conceptual test finished.")
