import unittest
import sys
import os

# Adjust path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.manager import VisualExplorerManager
from backends.osvi.osvi_backend import MockFastWAMBackend, MockDemoJepaBackend

class TestModelBackends(unittest.TestCase):
    """
    Unit tests to verify backend registration and abstract interfaces.
    """
    def setUp(self):
        self.manager = VisualExplorerManager()
        
    def test_backend_registration(self):
        self.manager.register_backend("FastWAM", MockFastWAMBackend())
        self.manager.register_backend("Demo-JEPA", MockDemoJepaBackend())
        
        self.assertEqual(len(self.manager.backends), 2)
        self.assertIn("FastWAM", self.manager.backends)
        self.assertIn("Demo-JEPA", self.manager.backends)
        
    def test_mock_backend_interface(self):
        backend = MockFastWAMBackend()
        trajs = backend.get_default_trajectories()
        self.assertIsNotNone(trajs)
        
        # Load trajectory mock
        traj_data = backend.load_trajectory("dummy.pkl")
        self.assertIn("images", traj_data)
        self.assertIn("context", traj_data)
        
        # Run inference mock
        tensors = backend.run_inference(traj_data["images"], traj_data["context"])
        self.assertIn("raw_waypoints", tensors)
        self.assertEqual(list(tensors["raw_waypoints"].shape), [1, 15, 4])


class TestRobotAdapter(unittest.TestCase):
    """
    Unit tests to verify RobotAdapter functionality and IK solvers.
    """
    def test_panda_adapter(self):
        from core.robot_adapter import FrankaPandaAdapter
        import numpy as np
        
        adapter = FrankaPandaAdapter()
        self.assertEqual(adapter.get_name(), "Franka Panda (7-DOF)")
        
        # Test default settings
        self.assertEqual(adapter.get_default_rotation(), [0.0, 0.0, 0.0])
        self.assertEqual(adapter.get_default_position(), [0.0, 0.0, 0.0])
        
        # Test URDF loading
        urdf_content = adapter.get_urdf_content()
        self.assertTrue(urdf_content.startswith("data:text/xml;base64,"))
        
        # Test IK conversion
        # Simulate 5 waypoints representing a straight path
        wps = np.array([
            [0.4, 0.0, 0.5],
            [0.42, 0.02, 0.51],
            [0.44, 0.04, 0.52],
            [0.46, 0.06, 0.53],
            [0.48, 0.08, 0.54]
        ])
        grasp = np.array([0.1, 0.2, 0.8, 0.9, 0.1])
        
        traj = adapter.cartesian_to_joint_trajectory(wps, grasp)
        self.assertEqual(len(traj), 5)
        self.assertIn("q", traj[0])
        self.assertIn("panda_joint1", traj[0]["q"])
        self.assertIn("panda_finger_joint1", traj[0]["q"])

if __name__ == "__main__":
    unittest.main()
