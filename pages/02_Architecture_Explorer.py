import streamlit as st
import sys
import os

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from core.manager import get_manager

st.set_page_config(page_title="Architecture Explorer - World Model Explorer", layout="wide")

manager = get_manager()
active_model = manager.active_backend_name

st.markdown(f"## 🏗️ Architecture Explorer ({active_model})")
st.write(f"Explore the dedicated neural network architecture for **{active_model}**.")

if active_model == "OSVI-WM":
    tabs = st.tabs([
        "1. ResNet Encoder",
        "2. Latent Bottleneck",
        "3. Action Model",
        "4. Forward Transition Model",
        "5. Spatial Softmax",
        "6. Temporal Pooler",
        "7. Waypoint Decoder Head"
    ])

    with tabs[0]:
        st.markdown("### 🧠 Shared Encoder (ResNet-18/50)")
        st.write("**Role:** Extracts spatial visual activations from monocular RGB image inputs.")
        st.write("**Input:** RGB Video/Frame tensors of shape `[B, T, 3, H, W]`.")
        st.write("**Output:** Convolutional feature maps of shape `[B, T, 512, H/32, W/32]`.")
        st.write("**Implementation:** ResNet convolutional layers with final pooling removed.")

    with tabs[1]:
        st.markdown("### 🌌 Latent Bottleneck ($Z$)")
        st.write("**Role:** Collapses high-dimensional pixel spaces into dense task-specific feature coordinates.")
        st.write("**Details:** Slices representations into $Z_E$ (expert demonstration features) and $Z_R$ (agent current feature state).")

    with tabs[2]:
        st.markdown("### 🎬 Action Model (Temporal Self-Attention)")
        st.write("**Role:** Models temporal causal dependencies across the expert and agent steps.")
        st.write("**Input:** Concatenation of $Z_E$ and $Z_R$.")
        st.write("**Features:** Employs Multi-Head Non-Local Self-Attention with causal masking.")

    with tabs[3]:
        st.markdown("### 🔮 Forward Transition Model")
        st.write("**Role:** Simulates the environment's dynamics internally (latent imagination).")
        st.write(r"**Rollout:** Predicts the subsequent latent state $\hat{z}_{t+1}$ given the action features and current state $\hat{z}_t$ autoregressively.")
        st.write("**Implementation:** A causal GPT-style Transformer network.")

    with tabs[4]:
        st.markdown("### 📍 Spatial Softmax (Differentiable Coords)")
        st.write("**Role:** Maps feature maps directly into 2D coordinate centers $(x_c, y_c)$ for planning.")
        st.write("**Math:** Computes expected index positions scaled by a 2D Softmax activation map.")

    with tabs[5]:
        st.markdown("### ⚡ Temporal Pooler (Attentive Cross-Attention)")
        st.write("**Role:** Summarizes the entire predicted future trajectory sequence into a single context embedding.")
        st.write("**Implementation:** Learnable query tokens cross-attending over temporal rollout steps.")

    with tabs[6]:
        st.markdown("### 📊 Waypoint Decoder MLP")
        st.write("**Role:** Generates the final 3D robot trajectory coordinates.")
        st.write("**Output:** Sequence of 15 waypoints, each defined as `(u_norm, v_norm, depth, grasp)`.")

elif active_model == "Demo-JEPA":
    tabs = st.tabs([
        "1. V-JEPA 2.1 ViT-Giant Encoder",
        "2. Spatio-Temporal Patch Tokens",
        "3. Dreamer Predictor Subgoal",
        "4. Action-Conditioned Dynamics (F_wm)",
        "5. CEM Latent Space MPC",
        "6. 7-DoF Controller Interface"
    ])

    with tabs[0]:
        st.markdown("### 👁️ V-JEPA 2.1 ViT-Giant Vision Backbone")
        st.write("**Role:** Extracts semantic spatial-temporal patch tokens from RGB inputs with RoPE positional embeddings.")
        st.write("**Input:** RGB video/observation frames `[B, 3, 256, 256]`.")
        st.write("**Output:** 256 latent patch tokens `[B, 256, 1408]` ($16 \\times 16$ spatial grid).")
        st.write("**Paper Link:** Yann LeCun et al., *V-JEPA 2.1 Joint-Embedding Predictive Architecture*.")

    with tabs[1]:
        st.markdown("### 🧱 Spatio-Temporal Patch Token Space ($Z$)")
        st.write("**Role:** Represents the visual scene as high-level abstract geometrical tokens, discarding pixel-level noise and background distractions.")
        st.write("**Details:** Slices representations into $S_{\\text{demo}}$ (reference tokens) and $S_{\\text{obs}}$ (current agent tokens).")

    with tabs[2]:
        st.markdown("### 🔮 Dreamer Predictor (Cross-Attention Subgoal)")
        st.write("**Role:** Cross-attends agent observation queries against demonstration reference keys/values to synthesize the target future latent subgoal $\\hat{s}_{\\text{target}}$.")
        st.write("**Implementation:** Multi-head cross-attention blocks (16 heads, 1024/1408 hidden dims).")

    with tabs[3]:
        st.markdown("### 🎯 Action-Conditioned Dynamics Model ($F_{wm}$)")
        st.write("**Role:** Simulates the causal forward transition $\\hat{s}_{t+1} = F_{wm}(s_t, a_t)$ inside latent space.")
        st.write("**Conditioning:** Conditioned on continuous 7-DoF robot action tokens ($[\\Delta x, \\Delta y, \\Delta z, \\Delta r_x, \\Delta r_y, \\Delta r_z, \\text{gripper}]$).")

    with tabs[4]:
        st.markdown("### ⚡ CEM Latent Space Trajectory Planner")
        st.write("**Role:** Optimizes candidate action sequences using the Cross-Entropy Method (CEM) to minimize latent L1 distance to the goal.")
        st.write("**Parameters:** 100 candidate samples, 10 elite selections, 15 CEM iterations, 5 rollout steps.")

    with tabs[5]:
        st.markdown("### 🦾 7-DoF Robot Controller Interface")
        st.write("**Role:** Directly streams planned continuous action commands to physical robot controllers (Fairino FR10 / Sawyer).")
        st.write("**Safety:** Cartesian delta clamping (max 100mm/step) and adaptive goal advancement ($D_k < 1.0$).")
