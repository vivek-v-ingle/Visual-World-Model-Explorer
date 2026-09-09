import streamlit as st
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from core.manager import get_manager

manager = get_manager()

# -------------------------------------------------------------
# Main View Header & Dashboards
# -------------------------------------------------------------
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
        2. **Causal Action Model:** Encodes cross-embodiment demonstration context and agent current observation.
        3. **Autoregressive GPT Forward Model:** Rolls out future latent states step-by-step in imagination.
        4. **Differentiable Spatial Softmax:** Maps feature heatmaps directly to continuous 2D/3D coordinates without decoding pixels.
        5. **Waypoint Decoder Head:** Dispatches 15 3D Cartesian waypoints $(x, y, z, \\text{grasp})$ to the robot.
        """)
        
        st.info("💡 **Next Step:** Use the sidebar to open **`End-to-End Pipeline`** or **`Input & Trajectory`** to step through the OSVI-WM dataflow.")

    with col_stats:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ OSVI-WM Specifications")
        st.write("- **Backbone:** ResNet-18 / ResNet-50")
        st.write("- **Latent Dimension ($Z$):** `512 x 8 x 10`")
        st.write("- **Planning Decoder:** Attentive Pooling + MLP Head")
        st.write("- **Output Space:** 15 3D Cartesian Waypoints `(u, v, d, grasp)`")
        st.write("- **Coordinate Mapping:** Camera frame $\\rightarrow$ Robot Base ($T_{\\text{cam2base}}$)")
        st.write("- **Target Robots:** Franka Emika Panda, Fairino FR10, UR5")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🗺️ OSVI-WM Pipeline Flowchart")
    st.markdown("""
    ```mermaid
    graph TD
        classDef stage fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
        classDef tensor fill:#064e3b,stroke:#34d399,stroke-width:1px,color:#f8fafc;

        V[🎥 1. Demonstration Video: 10 Context Frames] -->|Video E_1:10| E[🔎 2. ResNet-18 Shared Encoder]
        O[📷 3. Agent Current Workspace Observation] -->|Frame R_1| E
        E -->|Context Features Z_E: 10x512x8x10| AM[🎬 4. Causal Action Model]
        E -->|Agent Features Z_R: 1x512x8x10| AM
        AM -->|Action Tokens: 1x512x8x10| FM[🔮 5. Autoregressive GPT Forward Model]
        FM -->|Foreseen Latent Futures z_hat_t+1: 5x512x8x10| SS[📍 6. Differentiable Spatial Softmax]
        SS -->|2D Spatial Coords: 5x1024| TP[⚡ 7. Attentive Temporal Pooler]
        TP -->|Pooled Embedding: 1x1024| WD[📊 8. Waypoint Decoder MLP]
        WD -->|15 Waypoints u,v,depth,grasp| TC[📐 9. Cam-to-Base Calibration T_cam2base]
        TC -->|"Physical (X, Y, Z) Trajectory in mm"| R[🦾 10. Robot Arm Execution / Fairino FR10]

        class V,E,AM,FM,SS,TP,WD,TC,R stage;
        class O tensor;
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
        
        st.info("💡 **Next Step:** Use the sidebar to open **`Real Demo-JEPA Pipeline`** to run live latent rollouts and CEM optimization.")

    with col_stats:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Demo-JEPA Specifications")
        st.write("- **Backbone:** V-JEPA 2.1 ViT-Giant (RoPE)")
        st.write("- **Latent Dimension ($Z$):** `256 x 1408` (16x16 patch grid)")
        st.write("- **Subgoal Module:** Dreamer Cross-Attention Predictor")
        st.write("- **Planner:** Cross-Entropy Method (CEM) Latent MPC")
        st.write("- **Output Space:** 7-DoF Robot Actions / Joint angles ($qpos$)")
        st.write("- **Target Robots:** Fairino FR10, Sawyer, Franka")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🗺️ Demo-JEPA Pipeline Flowchart")
    st.markdown("""
    ```mermaid
    graph TD
        classDef stage fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
        classDef tensor fill:#064e3b,stroke:#34d399,stroke-width:1px,color:#f8fafc;

        V[🎥 1. Reference Video / Episode Demonstration] -->|Frames x_1:T| E[🔎 2. V-JEPA 2.1 ViT-Giant Encoder]
        O[📷 3. Agent Current Observation Frame] -->|Frame x_obs| E
        E -->|Demo Tokens s_demo: 256x1408| DP[🔮 4. Dreamer Predictor Cross-Attention]
        E -->|Obs Tokens s_obs: 256x1408| DP
        DP -->|Synthesized Latent Subgoal s_hat_target: 256x1408| CEM[🎯 5. CEM Latent Space Trajectory Planner]
        R[🤖 6. Robot TCP Pose State] -->|Conditioning| CEM
        CEM -->|Rollout Evaluation via F_wm Latent Dynamics| CEM
        CEM -->|7-DoF Deltas: dx,dy,dz,drx,dry,drz,gripper| FR[🦾 7. Fairino FR10 Robot Controller Driver]
        FR -->|Adaptive Subgoal Tracker: D_k < 1.0| V

        class V,E,DP,CEM,FR stage;
        class O,R tensor;
    ```
    """)
