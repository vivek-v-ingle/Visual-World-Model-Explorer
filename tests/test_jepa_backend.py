import sys
import os
import torch

sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from backends.jepa.jepa_backend import DemoJEPABackend

def test_jepa_backend():
    print("Testing DemoJEPABackend initialization...")
    backend = DemoJEPABackend(device="cpu")
    
    ckpts = backend.get_supported_checkpoints()
    print("Supported Checkpoints:", list(ckpts.keys()))
    assert len(ckpts) > 0
    
    trajs = backend.get_default_trajectories()
    print("Default Trajectories:", list(trajs.keys()))
    assert len(trajs) > 0
    
    # Test loading sample trajectory
    sample_key = "Packaged Sample Trajectory (.pkl)"
    sample_path = trajs[sample_key]
    print(f"Loading trajectory: {sample_key} ({sample_path})")
    traj_data = backend.load_trajectory(sample_path)
    
    assert "images" in traj_data
    assert "context" in traj_data
    print(f"  Context shape: {traj_data['context'].shape}")
    print(f"  Images shape: {traj_data['images'].shape}")
    
    # Test model loading
    ckpt_path = list(ckpts.values())[0]
    backend.load_model(ckpt_path)
    
    # Test inference execution
    print("Running Demo-JEPA latent inference and CEM rollout...")
    tensors = backend.run_inference(traj_data["images"], traj_data["context"])
    
    assert "action_7d_deltas" in tensors
    assert "vjepa_full_grid" in tensors
    assert "dreamer_subgoal" in tensors
    assert "raw_waypoints" in tensors
    assert "predicted_qpos" in tensors
    
    print("  Action 7D shape:", tensors["action_7d_deltas"].shape)
    print("  V-JEPA grid shape:", tensors["vjepa_full_grid"].shape)
    print("  Waypoints shape:", tensors["raw_waypoints"].shape)
    print("  Predicted qpos shape:", tensors["predicted_qpos"].shape)
    print("  Latent L1 distance:", tensors["latent_l1_distance"])
    print("✓ All Demo-JEPA backend tests passed successfully!")

if __name__ == "__main__":
    test_jepa_backend()
