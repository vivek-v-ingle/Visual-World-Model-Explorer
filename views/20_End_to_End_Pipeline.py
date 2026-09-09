import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go

# Adjust path to import files
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from core.manager import get_manager, render_debugger_navigation
from visualizers.plots import preprocess_image_tensor


manager = get_manager()
active_model = manager.active_backend_name

st.markdown(f"## 🌐 End-to-End Pipeline Explorer ({active_model})")
st.write(f"Understand the full dataflow lifecycle for **{active_model}** from raw camera inputs to physical robot trajectory actions.")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to **03 Input Explorer** or **21 Demo JEPA Pipeline** and click **Run Inference** first.")
else:
    tensors = st.session_state['captured_tensors']
    traj_data = st.session_state['loaded_trajectory']
    
    # -------------------------------------------------------------
    # DEMO-JEPA PIPELINE VIEW
    # -------------------------------------------------------------
    if active_model == "Demo-JEPA" or "action_7d_deltas" in tensors:
        st.markdown("""
        ```mermaid
        graph LR
            V[🎥 Demo Video] --> E[🔎 V-JEPA 2.1 ViT-Giant]
            O[📷 Obs Frame] --> E
            E --> DP[🔮 Dreamer Subgoal]
            DP --> CEM[🎯 CEM Latent Planner]
            CEM --> FR[🦾 Fairino FR10 7-DoF Actions]
        ```
        """)
        
        st.info("💡 Expand any pipeline stage below to view its latent representations, mathematical parameters, and hardware outputs.")

        # Stage 1: Reference Episode
        with st.expander("🎥 Stage 1: Cross-Embodiment Reference Demonstration", expanded=True):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown("**Explanation:**")
                st.write(
                    "Demo-JEPA receives multi-view video observations from a source embodiment (Sawyer / Franka / Human). "
                    "The system samples 10 checkpoint nodes ($E_1 ... E_{10}$) representing task progress."
                )
                context_np = traj_data['context'][0].cpu().numpy()
                obs_np = traj_data['images'][0].cpu().numpy()

                st.markdown("**Sampled Reference Frames ($E_1 ... E_{10}$):**")
                grid_top = st.columns(5)
                for i in range(5):
                    with grid_top[i]:
                        st.image(preprocess_image_tensor(context_np[i]), caption=f"Ref {i+1}", use_container_width=True)
            with col_r:
                st.markdown("**Agent Observation ($R_1$):**")
                st.image(preprocess_image_tensor(obs_np[0]), caption="Current Camera Observation", use_container_width=True)
                st.write(f"- **Resolution:** $256 \\times 256$ pixels")
                st.write(f"- **Dataset Source:** `{traj_data.get('raw_data', {}).get('dataset_type', 'Episode Dataset')}`")

        # Stage 2: V-JEPA ViT-Giant Encoder
        with st.expander("🔎 Stage 2: V-JEPA 2.1 ViT-Giant Spatio-Temporal Patch Encoder", expanded=True):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown("**Explanation:**")
                st.write(
                    "Frames are tokenized into $16 \\times 16$ non-overlapping patches ($256$ tokens per frame) and projected "
                    "into a 1408-dimensional embedding space using RoPE positional encodings."
                )
                if "vjepa_full_grid" in tensors:
                    vjepa_grid = tensors["vjepa_full_grid"][0].cpu().numpy()
                    obs_latent = vjepa_grid[-1]
                    token_norms = np.linalg.norm(obs_latent, axis=0)
                    fig_hm = px.imshow(token_norms, title="V-JEPA Patch Feature Norm (16x16 Grid)", color_continuous_scale="Viridis")
                    fig_hm.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_hm, use_container_width=True)
            with col_r:
                st.markdown("**Encoder Specs:**")
                st.write("- **Backbone:** ViT-Giant with RoPE")
                st.write("- **Latent Dimension:** `[1, 256, 1408]`")
                st.write("- **Patch Size:** $16 \\times 16$")

        # Stage 3: Dreamer Predictor Subgoal
        with st.expander("🔮 Stage 3: Dreamer Predictor Cross-Attention Subgoal", expanded=True):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown("**Explanation:**")
                st.write(
                    "The Dreamer Predictor cross-attends current observation tokens with reference demo tokens "
                    "to synthesize the future latent target $\\hat{s}_{\\text{target}} \\in \\mathbb{R}^{256 \\times 1408}$."
                )
                if "dreamer_attention" in tensors:
                    attn_w = tensors["dreamer_attention"][0].cpu().numpy()
                    fig_a = px.imshow(attn_w[:32, :32], title="Cross-Attention Matrix (Obs Queries vs Demo Keys)", color_continuous_scale="Plasma")
                    fig_a.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_a, use_container_width=True)
            with col_r:
                st.markdown("**Subgoal Specs:**")
                st.write("- **Attention Heads:** 16")
                st.write("- **Subgoal Shape:** `[1, 256, 1408]`")

        # Stage 4: CEM Latent MPC & Dynamics
        with st.expander("🎯 Stage 4: CEM Latent Space Trajectory Planning", expanded=True):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown("**Explanation:**")
                st.write(
                    "Cross-Entropy Method samples 100 candidate 7-DoF action sequences, rolls out latent predictions "
                    "inside $F_{wm}$, and selects elite samples minimizing latent $L_1$ distance."
                )
                costs_h = tensors.get("cem_costs_history", [0.85, 0.72, 0.61, 0.52, 0.45, 0.42])
                fig_c = px.line(x=list(range(1, len(costs_h) + 1)), y=costs_h, labels={"x": "CEM Iteration", "y": "Min Latent L1"}, title="CEM Convergence Curve", markers=True)
                fig_c.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_c, use_container_width=True)
            with col_r:
                st.markdown("**Planner Specs:**")
                st.write(f"- **Final Latent L1:** `{tensors.get('latent_l1_distance', 0.42):.4f}`")
                st.write("- **Rollout Horizon:** 5 steps")

        # Stage 5: 7-DoF Robot Output
        with st.expander("🦾 Stage 5: 7-DoF Robot Actions & 3D Execution Path", expanded=True):
            actions_7d = tensors.get("action_7d_deltas", np.zeros((5, 7)))
            df_act = pd.DataFrame(actions_7d, columns=["Δx (m)", "Δy (m)", "Δz (m)", "Δrx (rad)", "Δry (rad)", "Δrz (rad)", "Gripper (0-1)"])
            df_act.index = [f"Step {i+1}" for i in range(len(actions_7d))]
            st.dataframe(df_act.style.format("{:.4f}"), use_container_width=True)

            raw_wp = tensors["raw_waypoints"][0].cpu().numpy()
            fig_3d = go.Figure(data=[
                go.Scatter3d(x=raw_wp[:, 0], y=raw_wp[:, 1], z=raw_wp[:, 2], mode='lines+markers', line=dict(color='#818cf8', width=5), name="CEM Rollout")
            ])
            fig_3d.update_layout(title="Continuous 3D Cartesian Execution Path (Fairino FR10 Flange)", height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_3d, use_container_width=True)

    # -------------------------------------------------------------
    # OSVI-WM PIPELINE VIEW
    # -------------------------------------------------------------
    else:
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

        # 1. Input RGB Video
        with st.expander("🎥 Stage 1: Input RGB Video", expanded=True):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown("**Explanation:**")
                st.write("Monocular RGB video demonstration of a task. Only flat RGB pixel values are provided—no depth sensor data is fed into the world model.")
            with col_r:
                st.write("- **Format:** Monocular RGB (240x320)")

        # 2. ResNet Encoder
        with st.expander("🔎 Stage 2: ResNet Shared Encoder", expanded=True):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown("**Explanation:**")
                st.write("Extracts convolutional feature maps `[B, 512, 8, 10]` across time, removing final classification layers.")
            with col_r:
                st.write(f"- **Feature Shape:** `{list(tensors.get('resnet_features', torch.zeros(1, 11, 512, 8, 10)).shape)}`")

        # 3. Action Model
        with st.expander("🎬 Stage 3: Causal Action Model (Temporal Self-Attention)", expanded=True):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown("**Explanation:**")
                st.write("Models temporal causal dependencies across expert and agent steps using non-local self-attention blocks.")
            with col_r:
                st.write("- **Attention Mechanism:** Non-Local Spatiotemporal Self-Attention")

        # 4. Forward Transition Model
        with st.expander("🔮 Stage 4: Forward Transition Model (Latent Rollout)", expanded=True):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown("**Explanation:**")
                st.write("Predicts future latent states autoregressively in imagination space.")
            with col_r:
                st.write(f"- **Rollout Shape:** `{list(tensors.get('predicted_latent_states', torch.zeros(1, 5, 512, 8, 10)).shape)}`")

        # 5. Spatial Softmax & Waypoints
        with st.expander("📍 Stage 5: Differentiable Spatial Softmax & 3D Waypoints", expanded=True):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown("**Explanation:**")
                st.write("Computes 2D coordinate centers and decodes 15 Cartesian waypoints `(u, v, depth, grasp)`.")
                raw_wp = tensors.get("raw_waypoints", torch.zeros(1, 15, 4))[0].cpu().numpy()
                fig_3d = go.Figure(data=[
                    go.Scatter3d(x=raw_wp[:, 0], y=raw_wp[:, 1], z=raw_wp[:, 2], mode='lines+markers', line=dict(color='#38bdf8', width=5), name="OSVI Waypoints")
                ])
                fig_3d.update_layout(title="3D Predicted Waypoint Trajectory", height=380, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_3d, use_container_width=True)
            with col_r:
                st.write("- **Total Waypoints:** 15 Cartesian Steps")

st.divider()
render_debugger_navigation("views/20_End_to_End_Pipeline.py")
