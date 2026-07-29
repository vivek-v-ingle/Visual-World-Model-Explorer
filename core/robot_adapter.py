import os
import base64
import numpy as np
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

class BaseRobotAdapter(ABC):
    """
    Abstract Base Class for Robot Adapters.
    Decouples Cartesian waypoint predictions from specific robot visualizers.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the robot."""
        pass

    @abstractmethod
    def get_urdf_content(self) -> str:
        """Return the URDF XML contents as a base64 encoded data URI."""
        pass

    @abstractmethod
    def get_default_rotation(self) -> List[float]:
        """Return default rotation angles [roll, pitch, yaw] in degrees."""
        pass

    @abstractmethod
    def get_default_position(self) -> List[float]:
        """Return default position translation [x, y, z] in meters."""
        pass

    @abstractmethod
    def cartesian_to_joint_trajectory(self, cartesian_wps: np.ndarray, grasp_scores: np.ndarray) -> List[Dict[str, Any]]:
        """
        Solve Inverse Kinematics for a Cartesian path (N x 3) and return
        a trajectory list of joint configurations for visualization.
        """
        pass


class FrankaPandaAdapter(BaseRobotAdapter):
    """
    Franka Emika Panda Robot Adapter.
    Uses ikpy to solve numerical IK on Cartesian waypoints.
    """
    def __init__(self, urdf_path: str = None):
        if urdf_path is None:
            # Fallback path inside standard pybullet package
            urdf_path = os.path.expanduser("~/osvi-wm/.venv/lib/python3.10/site-packages/pybullet_data/franka_panda/panda.urdf")
        self.urdf_path = urdf_path
        self.chain = None
        
    def get_name(self) -> str:
        return "Franka Panda (7-DOF)"
        
    def get_default_rotation(self) -> List[float]:
        return [0.0, 0.0, 0.0]
        
    def get_default_position(self) -> List[float]:
        return [0.0, 0.0, 0.0]

    def _detect_root_link(self) -> str:
        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()
            all_links = {l.get("name") for l in root.findall("link") if l.get("name")}
            child_links = set()
            for joint in root.findall("joint"):
                child = joint.find("child")
                if child is not None:
                    child_links.add(child.get("link"))
            root_links = all_links - child_links
            if root_links:
                if "base_link" in root_links:
                    return "base_link"
                return sorted(root_links)[0]
        except Exception:
            pass
        return "base_link"

    def _load_chain(self):
        if self.chain is None:
            from ikpy.chain import Chain
            root_link = self._detect_root_link()
            self.chain = Chain.from_urdf_file(self.urdf_path, base_elements=[root_link])
        return self.chain

    def get_urdf_content(self) -> str:
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"Panda URDF not found at {self.urdf_path}")
            
        with open(self.urdf_path, "r") as f:
            urdf_xml = f.read()
            
        # We base64 encode the URDF XML so the WebGL frontend index.html custom loader can parse it
        urdf_b64 = base64.b64encode(urdf_xml.encode("utf-8")).decode("utf-8")
        return f"data:text/xml;base64,{urdf_b64}"

    def cartesian_to_joint_trajectory(self, cartesian_wps: np.ndarray, grasp_scores: np.ndarray) -> List[Dict[str, Any]]:
        chain = self._load_chain()
        trajectory = []
        
        # Base seed for IK solver
        seed = np.zeros(len(chain.links))
        
        for i, pt in enumerate(cartesian_wps):
            # Solve IK using ikpy
            q_solved = chain.inverse_kinematics(target_position=pt, initial_position=seed)
            # Update seed with previous solved joints to keep trajectories smooth
            seed = q_solved
            
            # Map joints based on Panda chain link structure
            # Franka Panda has joints: panda_joint1, panda_joint2, ..., panda_joint7
            # Grasp value drives fingers: panda_finger_joint1, panda_finger_joint2
            q_dict = {}
            for j_idx, link in enumerate(chain.links):
                if "panda_joint" in link.name:
                    q_dict[link.name] = float(q_solved[j_idx])
                    
            # Inject gripper finger joint values depending on grasp score
            # Score > 0.5 indicates closing (value 0.0), else open (value 0.04)
            finger_val = 0.0 if grasp_scores[i] > 0.5 else 0.04
            q_dict["panda_finger_joint1"] = finger_val
            q_dict["panda_finger_joint2"] = finger_val
            
            trajectory.append({
                "time": float(i * 0.1),
                "q": q_dict
            })
            
        return trajectory
