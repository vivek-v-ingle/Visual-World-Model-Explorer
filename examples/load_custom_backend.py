import sys
import os
import torch
from typing import Dict, Any

# Adjust path to import abstract base classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.types import BaseModelBackend

class CustomWorldModelBackend(BaseModelBackend):
    """
    Example custom backend implementation (e.g. for Demo-JEPA or FastWAM).
    """
    def get_supported_checkpoints(self) -> Dict[str, str]:
        return {"MyModel Default": "/path/to/my/model.pt"}
        
    def get_default_trajectories(self) -> Dict[str, str]:
        return {"MyModel Trajectory": "/path/to/my/trajectory.pkl"}
        
    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        print(f"Loading custom model weights from {checkpoint_path}")
        # Initialize your model here
        
    def load_trajectory(self, traj_path: str) -> Dict[str, Any]:
        print(f"Loading custom trajectory from {traj_path}")
        # Parse and return images and context tensors
        return {
            "images": torch.zeros((1, 1, 3, 240, 320)),
            "context": torch.zeros((1, 10, 3, 240, 320)),
            "projection_matrix": torch.eye(4)[:3, :4].unsqueeze(0)
        }
        
    def run_inference(self, images: Any, context: Any, T_tot: int = 16) -> Dict[str, Any]:
        print("Running custom model inference step-by-step")
        # Populate and return intermediate tensors registry
        return {
            "images": images,
            "context": context,
            "resnet_features": torch.zeros((1, 11, 512, 8, 10)),
            "raw_waypoints": torch.zeros((1, 15, 4))
        }

if __name__ == "__main__":
    backend = CustomWorldModelBackend()
    print("Custom model backend instantiated successfully!")
