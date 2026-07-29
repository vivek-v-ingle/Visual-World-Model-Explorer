import streamlit as st

st.set_page_config(page_title="Architecture Explorer - World Model Explorer", layout="wide")

st.markdown("## 🏗️ World Model Architecture Explorer")
st.write("Click on any module tab below to explore its structural role, dimensions, and operational logic.")

tabs = st.tabs([
    "1. Shared Encoder",
    "2. Latent Bottleneck",
    "3. Action Model",
    "4. Forward Transition Model",
    "5. Spatial Softmax",
    "6. Temporal Pooler",
    "7. Planner Head"
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
    st.write("**Rollout:** Predicts the subsequent latent state $\hat{z}_{t+1}$ given the action features and current state $\hat{z}_t$ autoregressively.")
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
