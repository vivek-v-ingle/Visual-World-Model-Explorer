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

elif manager.active_backend_name == "DINO-WM":
    col_main, col_stats = st.columns([2, 1])
    with col_main:
        st.markdown('<div class="model-badge">DINO-WM Active</div>', unsafe_allow_html=True)
        st.markdown(r"""
        ### 🦕 DINO-WM: DINOv2 Visual World Model
        **DINO-WM** (Meta FAIR) employs frozen **DINOv2 ViT-S/14** spatial patch embeddings paired with a 6-layer action-conditioned ViT predictor:
        
        1. **DINOv2 ViT-S/14 Encoder:** Encodes 224x224 RGB frames into 256 spatial patch tokens `[B, 256, 384]`.
        2. **6-Depth ViTPredictor:** Action-conditioned forward transformer predicts future latent patch representations.
        3. **Latent Patch Attention Retargeting:** Computes patch-wise attention maps for object-centric tracking.
        4. **CEM / MPPI Latent Planner:** Evaluates candidate 7-DoF actions against target subgoal representations using $L_1$ objective ($L_{1,\text{thresh}} = 0.70$).
        """)
        
        st.info("💡 **Next Step:** Use the sidebar to open **`DINO-WM Pipeline`** to run live 6-step depth rollouts.")

    with col_stats:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ DINO-WM Specifications")
        st.write("- **Backbone:** DINOv2 ViT-S/14 ($224 \\times 224$)")
        st.write("- **Latent Dimension ($Z$):** `256 x 384` (16x16 patch grid)")
        st.write("- **Predictor Depth:** 6-Layer Action-Conditioned Transformer")
        st.write("- **Planner:** CEM Latent MPC ($L_1$ threshold = 0.70)")
        st.write("- **Target Robots:** DROID benchmark, Fairino FR10, RoboCasa")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🗺️ DINO-WM Pipeline Flowchart")
    st.markdown("""
    ```mermaid
    graph TD
        classDef stage fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
        classDef tensor fill:#064e3b,stroke:#34d399,stroke-width:1px,color:#f8fafc;

        O[📷 1. 224x224 RGB Workspace Frame] --> E[🔎 2. DINOv2 ViT-S/14 Encoder]
        E -->|256 Spatial Patch Tokens: 256x384| VP[🔮 3. 6-Depth ViTPredictor]
        A[🦾 4. Candidate 7-DoF Actions] -->|Conditioning| VP
        VP -->|6-Step Latent Rollout z_hat_1:6| CEM[🎯 5. CEM Latent Space Planner]
        CEM -->|Optimal 7-DoF Actions dx,dy,dz,drx,dry,drz,gripper| R[🦾 6. Fairino FR10 Execution]

        class O,E,VP,A,CEM,R stage;
    ```
    """)

elif manager.active_backend_name == "JEPA-WM":
    col_main, col_stats = st.columns([2, 1])
    with col_main:
        st.markdown('<div class="model-badge">JEPA-WM Active</div>', unsafe_allow_html=True)
        st.markdown(r"""
        ### 🧠 JEPA-WM: DINOv3 12-Layer Deep Latent World Model
        **JEPA-WM** combines Meta FAIR's **DINOv3 ViT-L/16** vision backbone with a deep 12-layer action-conditioned predictor for long-horizon latent planning:
        
        1. **DINOv3 ViT-L/16 Encoder:** Extracts high-capacity 1024-dimensional patch representations `[B, 256, 1024]`.
        2. **12-Depth Action-Conditioned Predictor:** Deep transformer dynamics model capable of long-horizon temporal rollouts.
        3. **Dreamer Cross-Attention Subgoal Module:** Synthesizes prospective goal representations $\hat{s}_{\text{target}}$.
        4. **7-DoF Joint & Gripper Controller Stream:** Streams continuous Cartesian deltas directly to physical robot arms.
        """)
        
        st.info("💡 **Next Step:** Use the sidebar to open **`JEPA-WM 12-Layer Pipeline`** to run live 12-step depth rollouts.")

    with col_stats:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ JEPA-WM Specifications")
        st.write("- **Backbone:** DINOv3 ViT-L/16 ($256 \\times 256$)")
        st.write("- **Latent Dimension ($Z$):** `256 x 1024` (16x16 patch grid)")
        st.write("- **Predictor Depth:** 12-Layer Deep Transformer")
        st.write("- **Subgoal Module:** Dreamer Cross-Attention Predictor")
        st.write("- **Target Robots:** DROID, Fairino FR10, Sawyer")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🗺️ JEPA-WM Pipeline Flowchart")
    st.markdown("""
    ```mermaid
    graph TD
        classDef stage fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
        classDef tensor fill:#064e3b,stroke:#34d399,stroke-width:1px,color:#f8fafc;

        O[📷 1. 256x256 RGB Frame] --> E[🔎 2. DINOv3 ViT-L/16 Encoder]
        E -->|Patch Tokens: 256x1024| DP[🔮 3. Dreamer Subgoal Cross-Attention]
        DP -->|Subgoal s_hat_target: 256x1024| VP[🔮 4. 12-Depth Deep Action-Conditioned Predictor]
        A[🦾 5. Candidate 7-DoF Actions] -->|Conditioning| VP
        VP -->|12-Step Deep Latent Rollout| CEM[🎯 6. CEM Latent Space MPC]
        CEM -->|7-DoF Deltas| R[🦾 7. Fairino FR10 Execution]

        class O,E,DP,VP,A,CEM,R stage;
    ```
    """)
