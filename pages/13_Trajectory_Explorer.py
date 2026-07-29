import streamlit as st
import sys
import os
import numpy as np
import torch
import tempfile
import cv2
import subprocess
import base64

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
sys.path.insert(0, '/home/vvijaykumar/osvi-wm')
from core.manager import render_debugger_navigation
from visualizers.plots import plot_final_predictions_3d, render_2d_trajectory_overlay
from utils.projection_utils import image_coords_to_3d
from dataset.agent_dataset import traj_to_base_matrix

# Import sync_viewer widget
from widgets.sync_viewer import sync_viewer

st.set_page_config(page_title="Trajectory Explorer - World Model Explorer", layout="wide")

st.markdown("## 🏁 Stage 9: Trajectory Explorer")
st.write("Visualizes the final trajectory overlayed in 3D robot workspace coordinates, fully synchronized with the expert demonstration video.")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to the **03 Input Explorer** page and click **Run Inference** first.")
else:
    tensors = st.session_state['captured_tensors']
    traj_data = st.session_state['loaded_trajectory']
    
    raw_waypoints = tensors['raw_waypoints'] # [1, 15, 4]
    proj_mats = traj_data['projection_matrix'] # [1, 3, 4]
    
    # 1. Project 2D coordinates to 3D world space
    with st.spinner("Projecting coordinates to 3D workspace..."):
        try:
            world_waypoints = image_coords_to_3d(raw_waypoints.cpu(), proj_mats.cpu())
            world_waypoints[..., 3:] = raw_waypoints[..., 3:]
            world_wps_np = world_waypoints[0].numpy()
        except Exception as e:
            st.error(f"Failed to project waypoints to 3D: {e}")
            world_wps_np = np.zeros((15, 4))
            
    # 2. Extract ground truth and camera origin
    raw_traj = traj_data["raw_data"]["traj"]
    try:
        if 'world_to_image_transform' not in raw_traj.get(0)['obs']:
            raw_traj.setting_name = "ost"
            raw_traj.fname = "traj_zed.pkl"
        base_matrix = traj_to_base_matrix(raw_traj)
        camera_pos = base_matrix[:3, 3]
    except Exception:
        camera_pos = np.array([0.5, 0.0, 0.8])
        
    gt_coords = []
    for t in range(len(raw_traj)):
        gt_coords.append(raw_traj.get(t)['obs']['ee_aa'][:3])
    gt_coords = np.array(gt_coords)
    
    # Compile expert video to base64 Data URI
    if "video_b64" not in st.session_state:
        with st.spinner("Compiling expert video for synchronized viewer..."):
            try:
                temp_dir = tempfile.mkdtemp()
                for t in range(len(raw_traj)):
                    frame = raw_traj.get(t, decompress=True)
                    cv2.imwrite(os.path.join(temp_dir, f"frame_{t:06d}.png"), frame["obs"]["image"])
                
                video_out = os.path.join(temp_dir, "demo.mp4")
                cmd = ["ffmpeg", "-y", "-framerate", "10", "-i", os.path.join(temp_dir, "frame_%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", video_out]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                with open(video_out, "rb") as f:
                    v_bytes = f.read()
                st.session_state["video_b64"] = "data:video/mp4;base64," + base64.b64encode(v_bytes).decode()
            except Exception as ve:
                st.session_state["video_b64"] = ""
                st.error(f"Failed to compile demo video: {ve}")

    # Tabs
    tab_sync, tab_plots = st.tabs(["🔗 Synchronized Video & 3D Viewer", "📊 Static Overlays & Controls"])
    
    with tab_sync:
        st.markdown("### 🔗 Synchronized Viewer Options")
        vis_mode = st.radio("Choose 3D Visualization Mode:", ["3D Path Plotly Viewer", "3D Robot Arm Three.js Viewer"])
        
        if vis_mode == "3D Path Plotly Viewer":
            st.write("Play the video in the left panel. The yellow marker in the 3D plot tracks the end-effector location over time.")
            times = list(np.linspace(0, len(raw_traj)/10.0, len(world_wps_np)))
            viewer_data = {
                "mode": "dmp",
                "dmpTrajectory": {
                    "x": list(world_wps_np[:, 0].astype(float)),
                    "y": list(world_wps_np[:, 1].astype(float)),
                    "z": list(world_wps_np[:, 2].astype(float)),
                    "t": times,
                    "markers": [
                        {"label": "Start", "index": 0, "color": "#00cc96"},
                        {"label": "Grasp Point", "index": int(len(world_wps_np)/2), "color": "#ffaa00"},
                        {"label": "End", "index": len(world_wps_np)-1, "color": "#ef553b"}
                    ]
                }
            }
        else:
            st.write("Play the video in the left panel. The 3D robot model is animated step-by-step using joint angles resolved by the RobotAdapter IK.")
            
            from core.robot_adapter import FrankaPandaAdapter
            adapter = FrankaPandaAdapter()
            
            with st.spinner("Solving Inverse Kinematics for Franka Panda joints..."):
                try:
                    # Run IK on Cartesians
                    robot_trajectory = adapter.cartesian_to_joint_trajectory(world_wps_np[:, :3], world_wps_np[:, 3])
                    urdf_uri = adapter.get_urdf_content()
                    
                    viewer_data = {
                        "mode": "robot",
                        "models": [
                            {
                                "path": urdf_uri,
                                "trajectory": robot_trajectory,
                                "rotation": [-90, 0, 0], # default rotation
                                "position": [0, 0.1, -0.4] # default position
                            }
                        ]
                    }
                except Exception as e:
                    st.error(f"IK solver error: {e}")
                    viewer_data = None
                    
        if viewer_data is not None and st.session_state["video_b64"]:
            sync_viewer(viewer_data, video_path=st.session_state["video_b64"])
        elif not st.session_state["video_b64"]:
            st.warning("Video compiling failed. Use the charts tab below instead.")
            
    with tab_plots:
        st.markdown("### 📊 Interactive Trajectory Plots")
        col_c, col_d = st.columns([1, 2])
        
        with col_c:
            playback_limit = st.slider("Playback Index Limit", 1, len(world_wps_np), len(world_wps_np))
            wp_data = world_wps_np[playback_limit - 1]
            st.markdown(f"**Waypoint {playback_limit} Positions:**")
            st.write(f"- X: {wp_data[0]:.4f} m")
            st.write(f"- Y: {wp_data[1]:.4f} m")
            st.write(f"- Z: {wp_data[2]:.4f} m")
            st.write(f"- Grasp Score: {wp_data[3]:.4f} ({'CLOSE' if wp_data[3] > 0.5 else 'OPEN'})")
            
        with col_d:
            base_frame = traj_data['images'][0, 0].cpu().numpy()
            overlay_img = render_2d_trajectory_overlay(base_frame, raw_waypoints[0, :playback_limit].numpy())
            st.image(overlay_img, caption="Predicted Coordinates Overlaid on Observation", use_container_width=True)

# Render debug timeline
render_debugger_navigation("pages/13_Trajectory_Explorer.py")
