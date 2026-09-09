import streamlit as st
import sys
import os

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from core.manager import init_shared_state, get_manager

# Configure page settings
st.set_page_config(
    page_title="Visual World Model Explorer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global premium theme and styles
st.markdown("""
<style>
    div.stButton > button {
        background-color: #6366f1 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background-color: #4f46e5 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    .premium-header {
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .premium-subheader {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .model-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .model-badge {
        background-color: #312e81;
        color: #c7d2fe;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State variables
init_shared_state()
manager = get_manager()

# Sidebar Model Switcher
st.sidebar.markdown("### 🔌 Select Active World Model")
backend_options = ["OSVI-WM", "Demo-JEPA"]
current_idx = backend_options.index(manager.active_backend_name) if manager.active_backend_name in backend_options else 0
selected_backend = st.sidebar.radio(
    "Active World Model:",
    backend_options,
    index=current_idx,
    help="Switching models tailors the entire dashboard to display only the selected model's pipeline."
)

if selected_backend != manager.active_backend_name:
    manager.set_active_backend(selected_backend)
    st.session_state["captured_tensors"] = None
    st.session_state["loaded_trajectory"] = None
    st.rerun()

# Layout Header
st.markdown('<div class="premium-header">🔮 Visual World Model Explorer</div>', unsafe_allow_html=True)
st.markdown(f'<div class="premium-subheader">Interactive Pipeline & Tensor Debugger — Currently Viewing: <b>{manager.active_backend_name}</b></div>', unsafe_allow_html=True)

st.divider()

# Model Specific Dashboard View
if manager.active_backend_name == "OSVI-WM":
    col_main, col_stats = st.columns([2, 1])
    with col_main:
        st.markdown('<div class="model-badge">OSVI-WM Active</div>', unsafe_allow_html=True)
        st.markdown("""
        ### 📍 OSVI-WM: One-Shot Visual Imitation World Model
        **OSVI-WM** simulates future robotic trajectories directly in **spatial-feature coordinate space** using differentiable spatial softmax:
        
        1. **ResNet-18/50 Shared Encoder:** Extracts high-resolution spatial feature maps `[B, 512, 8, 10]`.
        2. **Causal Action Model:** Encodes cross-embodiment demonstration and current observation.
        3. **Autoregressive GPT Forward Model:** Rolls out future latent states step-by-step.
        4. **Differentiable Spatial Softmax:** Maps feature heatmaps directly to continuous 2D/3D coordinates without decoding pixels.
        5. **Waypoint Decoder Head:** Dispatches 15 3D Cartesian waypoints $(x, y, z, \text{grasp})$ to the robot.
        """)
        
        st.info("💡 **Recommended Next Step:** Open **`20 End to End Pipeline`** or **`03 Input Explorer`** to inspect the 9-stage OSVI-WM dataflow.")

    with col_stats:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ OSVI-WM Parameters")
        st.write("- **Backbone:** ResNet-18 / ResNet-50")
        st.write("- **Latent Dim ($Z$):** `512 x 8 x 10`")
        st.write("- **Planning Decoder:** Attentive Pooling + MLP Head")
        st.write("- **Output Space:** 3D Cartesian Waypoints `(u, v, d, grasp)`")
        st.write("- **Target Robots:** Franka Emika Panda, UR5")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🗺️ OSVI-WM Pipeline Flowchart")
    st.markdown("""
    ```mermaid
    graph LR
        V[🎥 RGB Video Demo] --> E[🔎 ResNet Encoder]
        E --> L[🌌 Latent Space Z]
        L --> AM[🎬 Action Model]
        AM --> FM[🔮 Autoregressive Forward Model]
        FM --> SS[📍 Spatial Softmax Coords]
        SS --> TP[⚡ Attentive Pooler]
        TP --> WD[📊 Waypoint Decoder]
        WD --> R[🤖 3D Cartesian Path]
    ```
    """)

elif manager.active_backend_name == "Demo-JEPA":
    col_main, col_stats = st.columns([2, 1])
    with col_main:
        st.markdown('<div class="model-badge">Demo-JEPA Active</div>', unsafe_allow_html=True)
        st.markdown(r"""
        ### 🧠 Demo-JEPA: Joint-Embedding Predictive Architecture
        **Demo-JEPA** operates entirely in **abstract latent embedding space**, discarding raw pixel noise and planning directly in latent representations:
        
        1. **V-JEPA 2.1 ViT-Giant Encoder:** Encodes frames into 256 spatio-temporal patch tokens `[B, 256, 1408]`.
        2. **Dreamer Predictor:** Cross-attends demonstration frames to synthesize latent subgoals $\hat{s}_{target}$.
        3. **Action-Conditioned World Model ($F_{wm}$):** Simulates prospective latent rollouts conditioned on candidate actions.
        4. **CEM Latent Trajectory Planner:** Samples continuous candidate actions and optimizes 7-DoF deltas via Cross-Entropy Method.
        5. **7-DoF Joint / Controller Stream:** Dispatches $[\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z, \text{gripper}]$ commands directly to physical robots (Fairino FR10 / Sawyer).
        """)
        
        st.info("💡 **Recommended Next Step:** Open **`21 Demo JEPA Pipeline`** to run real latent rollouts and CEM optimization.")

    with col_stats:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Demo-JEPA Parameters")
        st.write("- **Backbone:** V-JEPA 2.1 ViT-Giant (RoPE)")
        st.write("- **Latent Dim ($Z$):** `256 x 1408` (16x16 patch grid)")
        st.write("- **Subgoal Module:** Dreamer Cross-Attention Predictor")
        st.write("- **Planner:** Cross-Entropy Method (CEM) Latent MPC")
        st.write("- **Output Space:** 7-DoF Robot Actions / Joint angles ($qpos$)")
        st.write("- **Target Robots:** Fairino FR10, Sawyer, Franka")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🗺️ Demo-JEPA Pipeline Flowchart")
    st.markdown("""
    ```mermaid
    graph LR
        V[🎥 Demonstration Episode] --> E[🔎 V-JEPA 2.1 ViT-Giant]
        O[📷 Current Observation] --> E
        E --> DP[🔮 Dreamer Subgoal]
        DP --> CEM[🎯 CEM Latent Planner]
        CEM --> FR[🦾 Fairino FR10 7-DoF Stream]
    ```
    """)
