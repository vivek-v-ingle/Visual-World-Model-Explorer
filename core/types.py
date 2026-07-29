from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseModelBackend(ABC):
    """
    Abstract base class defining the contract for all World Model Backends
    (e.g., OSVI-WM, FastWAM, Demo-JEPA, Dreamer).
    """

    @abstractmethod
    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        """
        Load the model architecture and configure it using checkpoint weights.
        """
        pass

    @abstractmethod
    def load_trajectory(self, traj_path: str) -> Dict[str, Any]:
        """
        Load demonstration and initial observation frames from a file.
        Returns a dictionary containing preprocessed tensors and metadata:
        - "images": agent initial frame tensor [B, T_pair, C, H, W]
        - "context": teacher context frame tensor [B, T_context, C, H, W]
        - "projection_matrix": camera intrinsics/projection matrix
        - "raw_data": original raw pickle or dictionary payload
        """
        pass

    @abstractmethod
    def run_inference(self, images: Any, context: Any, T_tot: int = 16) -> Dict[str, Any]:
        """
        Run inference step-by-step and capture intermediate states, returning
        a registry dictionary mapping tensor names to their values.
        """
        pass

    @abstractmethod
    def get_supported_checkpoints(self) -> Dict[str, str]:
        """
        Return a mapping of friendly names to default checkpoint paths on the system.
        """
        pass

    @abstractmethod
    def get_default_trajectories(self) -> Dict[str, str]:
        """
        Return a mapping of friendly names to default trajectories on the system.
        """
        pass
