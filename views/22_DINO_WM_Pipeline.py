import streamlit as st
import sys
import os
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.manager import get_manager

manager = get_manager()

st.markdown("## 🦕 Stage 22: DINO-WM End-to-End Pipeline Explorer")
st.markdown("Examine Meta FAIR's **DINO-WM** (DINOv2 ViT-S/14) closed-loop visual trajectory planning architecture.")

col_opt1, col_opt2 = st.columns([1, 1])
with col_opt1:
    checkpoints = manager.get_active_backend().get_supported_checkpoints()
    selected_ckpt_name = st.selectbox("Select DINO-WM Checkpoint", list(checkpoints.keys()), key="dino_wm_ckpt_select")
    ckpt_path = checkpoints[selected_ckpt_name]

with col_opt2:
    trajectories = manager.get_active_backend().get_default_trajectories()
    selected_traj_name = st.selectbox("Select Trajectory Episode", list(trajectories.keys()), key="dino_wm_traj_select")
    traj_path = trajectories[selected_traj_name]

if st.button("🚀 Load Checkpoint & Run DINO-WM Rollout", use_container_width=True):
    with st.spinner("Running 6-depth prediction rollout & DINOv2 latent patch attention..."):
        try:
            backend = manager.get_active_backend()
            backend.load_model(ckpt_path)
            traj_data = backend.load_trajectory(traj_path)
            tensors = backend.run_inference(traj_data["images"], traj_data["context"])
            
            st.session_state["captured_tensors"] = tensors
            st.session_state["loaded_trajectory"] = traj_data
            st.success("✅ DINO-WM rollout completed successfully!")
        except Exception as e:
            st.error(f"Error executing DINO-WM pipeline: {e}")

st.divider()

if "captured_tensors" in st.session_state and st.session_state["captured_tensors"] is not None:
    tensors = st.session_state["captured_tensors"]
    
    st.markdown("### 📊 DINO-WM Captured Latent Tensors & Patch Heatmaps")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric("Input Resolution", "224 x 224")
        st.metric("Encoder Backbone", "DINOv2 ViT-S/14")
    with col_t2:
        st.metric("Latent Token Grid", "256 Tokens x 384 Dim")
        st.metric("Predictor Depth", "6 Rollout Steps")
    with col_t3:
        st.metric("Planner Objective", "CEM Latent MPC (L1 Threshold = 0.70)")
        st.metric("Action Output", "7-DoF Delta Trajectory")

    if "predicted_latent_states" in tensors:
        rollout_tensor = tensors["predicted_latent_states"]
        st.markdown(f"**6-Step Depth Latent Rollout Shape:** `{list(rollout_tensor.shape)}`")
        
        # Display step slider
        step_idx = st.slider("Select Prediction Rollout Depth Step", 0, rollout_tensor.shape[1] - 1, 0)
        step_feat = rollout_tensor[0, step_idx].numpy()
        
        st.info(f"💡 Currently inspecting Latent Step {step_idx + 1} / 6. Mean activation magnitude: `{float(np.mean(np.abs(step_feat))):.4f}`")
