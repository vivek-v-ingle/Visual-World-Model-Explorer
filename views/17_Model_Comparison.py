import streamlit as st
import pandas as pd
import plotly.express as px

st.markdown("## ⚖️ World Model Comparison Mode")
st.write("Compare the architecture, latent representations, and predictive rollouts of OSVI-WM, Demo-JEPA, DINO-WM, and JEPA-WM side-by-side.")

all_models = ["OSVI-WM", "Demo-JEPA", "DINO-WM", "JEPA-WM", "FastWAM", "Dreamer (RSSM)"]
default_models = ["OSVI-WM", "Demo-JEPA", "DINO-WM", "JEPA-WM"]

selected_compare_models = st.multiselect(
    "Select World Models to Compare:",
    all_models,
    default=default_models
)

model_data = {
    "Model Name": ["OSVI-WM", "Demo-JEPA", "DINO-WM", "JEPA-WM", "FastWAM", "Dreamer (RSSM)"],
    "Encoder Type": [
        "ResNet-18 / ResNet-50",
        "V-JEPA 2.1 ViT-Giant (RoPE)",
        "DINOv2 ViT-S/14 (224x224)",
        "DINOv3 ViT-L/16 (256x256)",
        "DINOv2 (Frozen)",
        "CNN Encoder"
    ],
    "Latent Dimension ($Z$)": [
        "512 x 8 x 10 (Spatial Maps)",
        "256 x 1408 (Spatio-Temporal Tokens)",
        "256 x 384 (Patch Grid)",
        "256 x 1024 (High-Capacity Patch Grid)",
        "1024 (Dense Feature Vector)",
        "1024 (Stochastic + Deterministic)"
    ],
    "Predictive Transition (Rollout)": [
        "Autoregressive Transformer",
        "Action-Conditioned ViT Predictor (F_wm)",
        "6-Layer Action-Conditioned ViTPredictor",
        "12-Layer Deep Transformer Predictor",
        "Non-autoregressive MLP Block",
        "Recurrent SSM (RSSM)"
    ],
    "Planning Decoder": [
        "Attentive Pooling + MLP Head",
        "Dreamer Predictor + CEM Latent MPC",
        "CEM / MPPI Latent MPC (L1 = 0.70)",
        "Dreamer Subgoal + CEM Latent MPC",
        "MLP Head",
        "Pixel Reconstruction + Policy Head"
    ],
    "Primary Use-Case": [
        "One-Shot Trajectory Imitation",
        "Cross-Embodiment Goal Imitation",
        "DROID Benchmark Object-Centric Planning",
        "Deep Long-Horizon Subgoal Tracking",
        "Real-time High-frequency Control",
        "Model-based Reinforcement Learning"
    ],
    "Inference Latency (ms)": [14.5, 19.8, 11.2, 16.4, 4.2, 32.5]
}

df_all = pd.DataFrame(model_data)

if len(selected_compare_models) > 0:
    df_filtered = df_all[df_all["Model Name"].isin(selected_compare_models)]
else:
    df_filtered = df_all

st.markdown("### 📊 Architecture Feature Comparison")
st.dataframe(df_filtered.set_index("Model Name"), use_container_width=True)

st.markdown("### ⚡ Inference Latency Comparison")
st.write("A critical factor in deploying world models for physical robotics is control loop latency. Compare the typical forward-pass times:")

fig_lat = px.bar(
    df_filtered,
    x="Model Name",
    y="Inference Latency (ms)",
    color="Model Name",
    title="Inference Latency (Lower is Better)",
    labels={"Inference Latency (ms)": "Latency (milliseconds)"},
    color_discrete_sequence=px.colors.qualitative.Plotly
)
st.plotly_chart(fig_lat, use_container_width=True)

st.divider()

st.markdown("### 🏗️ Deep Dive: Understanding the Architectures")

cols_deep = st.columns(len(selected_compare_models) if len(selected_compare_models) > 0 else 1)

for idx, model in enumerate(selected_compare_models):
    with cols_deep[idx]:
        st.markdown(f"#### 🌟 {model}")
        if model == "OSVI-WM":
            st.markdown("""
            - **How it works:** Encodes the expert video into a sequence of spatiotemporal feature maps, then predicts the agent's future states in feature-coordinate space.
            - **Key Advantage:** Differentiable spatial softmax allows mapping to precise 3D coords without decoding images.
            """)
        elif model == "Demo-JEPA":
            st.markdown("""
            - **How it works:** Uses V-JEPA 2.1 ViT-Giant to extract 256 spatio-temporal tokens ($256 \\times 1408$). The Dreamer Predictor cross-attends demonstration frames to synthesize latent subgoals, and a CEM planner optimizes continuous 7-DoF robot actions in latent feature space.
            - **Key Advantage:** Operates entirely in abstract embedding space without pixel reconstruction artifacts, achieving natural cross-embodiment generalization.
            """)
        elif model == "DINO-WM":
            st.markdown("""
            - **How it works:** Meta FAIR's model combining frozen DINOv2 ViT-S/14 visual patch tokens ($224 \\times 224$) with a 6-layer action-conditioned ViTPredictor.
            - **Key Advantage:** High spatial resolution for object-centric manipulation and fast CEM/MPPI trajectory optimization.
            """)
        elif model == "JEPA-WM":
            st.markdown("""
            - **How it works:** Combines DINOv3 ViT-L/16 patch tokens ($256 \\times 256$) with a deep 12-layer action-conditioned predictor and Dreamer cross-attention subgoal synthesis.
            - **Key Advantage:** Deep transformer layers enable stable, long-horizon multi-step sub-goal rollouts.
            """)
        elif model == "FastWAM":
            st.markdown("""
            - **How it works:** Replaces the heavy autoregressive transformer rollout with a feedforward network block.
            - **Key Advantage:** Reduces latency below 5ms for direct 250Hz real-time robot control.
            """)
        elif model == "Dreamer (RSSM)":
            st.markdown("""
            - **How it works:** Learns a Recurrent State Space Model (RSSM) containing both deterministic (GRU) and stochastic components.
            - **Key Advantage:** Reconstructs RGB frames to predict reward signals for model-based RL.
            """)

st.write("---")

st.markdown("### ⚖️ Real-World Benchmarking: VILMA vs. OSVI-WM vs. JEPA World Models")
st.write("This table presents objective comparative metrics across tracking frameworks and visual world models:")

comparison_payload = {
    "Metric Description": [
        "Primary Input Type",
        "Trajectory Length",
        "Number of Waypoints",
        "Average Inference/Solve Time",
        "Depth Sensor Requirement",
        "Coordinate Range X (meters)",
        "Coordinate Range Y (meters)",
        "Coordinate Range Z (meters)"
    ],
    "VILMA (Tracking-Centric)": [
        "Stereo RGB-D Video",
        "30 frames (dense sampling)",
        "30 tracking points",
        "~1.2 ms (CPU OpenCV + MediaPipe)",
        "Mandatory (ZED Stereo Camera)",
        "[-0.5, 0.5]",
        "[-0.5, 0.5]",
        "[0.1, 1.2]"
    ],
    "OSVI-WM (Spatial World Model)": [
        "Monocular RGB Video",
        "10 context + 5 rollout steps",
        "15 continuous 3D waypoints",
        "~14.5 ms (PyTorch GPU)",
        "Optional (Monocular Depth Prediction)",
        "[-0.4, 0.4]",
        "[-0.4, 0.4]",
        "[0.0, 0.8]"
    ],
    "JEPA-WM / DINO-WM (Latent World Models)": [
        "Monocular RGB / Video",
        "6-12 depth rollout steps",
        "Continuous 7-DoF Deltas (dx,dy,dz,drx,dry,drz,gripper)",
        "~11.2 - 16.4 ms (PyTorch GPU)",
        "None (Abstract Latent MPC)",
        "[-0.4, 0.4]",
        "[-0.4, 0.4]",
        "[0.0, 0.8]"
    ]
}

st.dataframe(pd.DataFrame(comparison_payload).set_index("Metric Description"), use_container_width=True)
