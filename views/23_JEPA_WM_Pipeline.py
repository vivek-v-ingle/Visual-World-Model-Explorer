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

st.markdown("## 🧠 Stage 23: JEPA-WM (DINOv3) Deep Latent Pipeline Explorer")
st.markdown("Examine Meta FAIR's **JEPA-WM** (DINOv3 ViT-L/16) 12-layer deep latent trajectory planning architecture.")

col_opt1, col_opt2 = st.columns([1, 1])
with col_opt1:
    checkpoints = manager.get_active_backend().get_supported_checkpoints()
    selected_ckpt_name = st.selectbox("Select JEPA-WM Checkpoint", list(checkpoints.keys()), key="jepa_wm_ckpt_select")
    ckpt_path = checkpoints[selected_ckpt_name]

with col_opt2:
    trajectories = manager.get_active_backend().get_default_trajectories()
    selected_traj_name = st.selectbox("Select Trajectory Episode", list(trajectories.keys()), key="jepa_wm_traj_select")
    traj_path = trajectories[selected_traj_name]

if st.button("🚀 Load Checkpoint & Run 12-Layer JEPA-WM Rollout", use_container_width=True):
    with st.spinner("Running 12-depth prediction rollout & Dreamer cross-attention subgoal synthesis..."):
        try:
            backend = manager.get_active_backend()
            backend.load_model(ckpt_path)
            traj_data = backend.load_trajectory(traj_path)
            tensors = backend.run_inference(traj_data["images"], traj_data["context"])
            
            st.session_state["captured_tensors"] = tensors
            st.session_state["loaded_trajectory"] = traj_data
            st.success("✅ JEPA-WM 12-layer rollout completed successfully!")
        except Exception as e:
            st.error(f"Error executing JEPA-WM pipeline: {e}")

st.divider()

if "captured_tensors" in st.session_state and st.session_state["captured_tensors"] is not None:
    tensors = st.session_state["captured_tensors"]
    
    st.markdown("### 📊 JEPA-WM Captured Deep Latent Tensors & Subgoal Tokens")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric("Input Resolution", "256 x 256")
        st.metric("Encoder Backbone", "DINOv3 ViT-L/16")
    with col_t2:
        st.metric("Latent Token Grid", "256 Tokens x 1024 Dim")
        st.metric("Predictor Depth", "12 Rollout Steps")
    with col_t3:
        st.metric("Subgoal Module", "Dreamer Cross-Attention Predictor")
        st.metric("Action Output", "7-DoF Delta Trajectory")

    if "predicted_latent_states" in tensors:
        rollout_tensor = tensors["predicted_latent_states"]
        st.markdown(f"**12-Step Depth Latent Rollout Shape:** `{list(rollout_tensor.shape)}`")
        
        step_idx = st.slider("Select Prediction Rollout Depth Step", 0, rollout_tensor.shape[1] - 1, 0)
        step_feat = rollout_tensor[0, step_idx].numpy()
        
        st.info(f"💡 Currently inspecting Latent Step {step_idx + 1} / 12. Mean activation magnitude: `{float(np.mean(np.abs(step_feat))):.4f}`")
