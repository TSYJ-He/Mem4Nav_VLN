import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, List, Union
class ControlModule(nn.Module):
    """
    Control Module for the Modular Pipeline Agent.
    It defines the effects of discrete actions on the agent's pose
    and can predict the next pose based on a chosen action.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.module_config = config.get('control', {}) # Specific config for this module
        self.device = device

        # Define parameters for discrete actions
        self.default_step_distance: float = self.module_config.get('default_step_distance', 1.0)  # meters for 'forward'
        self.default_turn_angle_rad: float = self.module_config.get('default_turn_angle_rad', np.deg2rad(30.0)) # radians for turns

        # Action space - should match the policy network's output interpretation
        self.action_list: List[str] = self.module_config.get('action_list', ["forward", "turn_left", "turn_right", "stop"])

    def get_action_list(self) -> List[str]:
        """Returns the list of recognized discrete actions."""
        return self.action_list

    def _wrap_angle_rad(self, angle_rad: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Wraps an angle in radians to the [-pi, pi] range."""
        return (angle_rad + np.pi) % (2 * np.pi) - np.pi

    def predict_next_pose(self, 
                            current_pose_tensor: torch.Tensor, # Expected [x, y, yaw_rad] on self.device
                            action_str: str
                           ) -> torch.Tensor:
        """
        Predicts the next pose given the current pose and a discrete action string.
        This is a kinematic prediction and does not consider collisions or environment constraints.

        Args:
            current_pose_tensor (torch.Tensor): Agent's current pose [x, y, yaw_in_radians],
                                                shape (3,).
            action_str (str): The discrete action to take (e.g., "forward", "turn_left").

        Returns:
            torch.Tensor: The predicted next pose [x_next, y_next, yaw_next_rad], shape (3,).
        """
        if current_pose_tensor.shape[0] < 3:
            raise ValueError("current_pose_tensor must have at least x, y, yaw components.")

        x, y, yaw_rad = current_pose_tensor[0], current_pose_tensor[1], current_pose_tensor[2]
        next_x, next_y, next_yaw_rad = x.clone(), y.clone(), yaw_rad.clone()

        if action_str == "forward":
            next_x += self.default_step_distance * torch.cos(yaw_rad)
            next_y += self.default_step_distance * torch.sin(yaw_rad)
        elif action_str == "turn_left":
            next_yaw_rad = self._wrap_angle_rad(yaw_rad + self.default_turn_angle_rad)
        elif action_str == "turn_right":
            next_yaw_rad = self._wrap_angle_rad(yaw_rad - self.default_turn_angle_rad)
        elif action_str == "stop":
            # Pose does not change
            pass
        # Could add other actions like "backward", "strafe_left", "strafe_right" if needed
        else:
            print(f"Warning: ControlModule received unknown action_str '{action_str}'. Pose unchanged.")

        return torch.stack([next_x, next_y, next_yaw_rad])

    def get_action_parameters(self, action_str: str) -> Dict[str, float]:
        """Returns parameters associated with a given action."""
        if action_str == "forward":
            return {'distance': self.default_step_distance}
        elif action_str in ["turn_left", "turn_right"]:
            return {'angle_rad': self.default_turn_angle_rad}
        elif action_str == "stop":
            return {}
        return {}


if __name__ == '__main__':
    print("--- Conceptual Test for ControlModule ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mock_control_config = {
        'control': {
            'default_step_distance': 0.5,      # meters
            'default_turn_angle_rad': np.pi / 6 # 30 degrees
        }
    }
    control_module = ControlModule(mock_control_config, device)

    # Initial pose: at origin, facing positive x-axis (0 radians yaw)
    initial_pose = torch.tensor([0.0, 0.0, 0.0], device=device) # [x, y, yaw_rad]
    print(f"Initial Pose: {initial_pose.cpu().numpy()}")

    # Test "forward"
    action = "forward"
    next_pose_fwd = control_module.predict_next_pose(initial_pose, action)
    print(f"Action: '{action}', Next Pose: {next_pose_fwd.cpu().numpy().round(3)}")
    assert torch.allclose(next_pose_fwd, torch.tensor([0.5, 0.0, 0.0], device=device))

    # Test "turn_left"
    action = "turn_left"
    # Pose after forward: [0.5, 0.0, 0.0]
    next_pose_tl = control_module.predict_next_pose(next_pose_fwd, action)
    print(f"Action: '{action}', Next Pose: {next_pose_tl.cpu().numpy().round(3)}")
    expected_yaw_tl = (0.0 + np.pi / 6)
    assert torch.allclose(next_pose_tl, torch.tensor([0.5, 0.0, expected_yaw_tl], device=device))

    # Test "turn_right" from the new orientation
    action = "turn_right"
    # Pose after turn_left: [0.5, 0.0, np.pi/6]
    next_pose_tr = control_module.predict_next_pose(next_pose_tl, action)
    print(f"Action: '{action}', Next Pose: {next_pose_tr.cpu().numpy().round(3)}")
    expected_yaw_tr = control_module._wrap_angle_rad(expected_yaw_tl - np.pi / 6) # Should be close to 0
    assert torch.allclose(next_pose_tr, torch.tensor([0.5, 0.0, expected_yaw_tr], device=device), atol=1e-6)

    # Test "stop"
    action = "stop"
    next_pose_stop = control_module.predict_next_pose(next_pose_tr, action)
    print(f"Action: '{action}', Next Pose: {next_pose_stop.cpu().numpy().round(3)}")
    assert torch.allclose(next_pose_stop, next_pose_tr)
    
    # Test unknown action
    action = "fly"
    next_pose_unknown = control_module.predict_next_pose(initial_pose, action)
    print(f"Action: '{action}', Next Pose: {next_pose_unknown.cpu().numpy().round(3)}")
    assert torch.allclose(next_pose_unknown, initial_pose) # Pose should be unchanged

    print("\nControlModule conceptual test finished.")
