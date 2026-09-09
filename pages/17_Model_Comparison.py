import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Comparison - World Model Explorer", layout="wide")

st.markdown("## ⚖️ World Model Comparison Mode")
st.write("Compare the architecture, latent representations, and predictive rollouts of OSVI-WM against other predictive latent world models.")

selected_compare_models = st.multiselect(
    "Select World Models to Compare:",
    ["OSVI-WM", "FastWAM", "Demo-JEPA", "Dreamer (RSSM)"],
    default=["OSVI-WM", "Demo-JEPA"]
)

model_data = {
    "Model Name": ["OSVI-WM", "FastWAM", "Demo-JEPA", "Dreamer (RSSM)"],
    "Encoder Type": ["ResNet-18 / ResNet-50", "Dino-v2 (Frozen)", "V-JEPA 2.1 ViT-Giant (RoPE)", "CNN Encoder"],
    "Latent Dimension ($Z$)": ["512 x 8 x 10 (Spatial Maps)", "1024 (Dense Feature Vector)", "256 x 1408 (Spatio-Temporal Tokens)", "1024 (Stochastic + Deterministic)"],
    "Predictive Transition (Rollout)": ["Autoregressive Transformer", "Non-autoregressive MLP Block", "Action-Conditioned ViT Predictor (F_wm)", "Recurrent SSM (RSSM)"],
    "Planning Decoder": ["Attentive Pooling + MLP Head", "MLP Head", "Dreamer Predictor + CEM Latent MPC", "Pixel Reconstruction + Policy Head"],
    "Primary Use-Case": ["One-Shot Trajectory Imitation", "Real-time High-frequency Control", "Cross-Embodiment Goal Imitation", "Model-based Reinforcement Learning"],
    "Inference Latency (ms)": [14.5, 4.2, 19.8, 32.5]
}

df_all = pd.DataFrame(model_data)

if len(selected_compare_models) > 0:
    df_filtered = df_all[df_all["Model Name"].isin(selected_compare_models)]
else:
    df_filtered = df_all

st.markdown("### 📊 Architecture Feature Comparison")
st.dataframe(df_filtered.set_index("Model Name"))

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
        elif model == "FastWAM":
            st.markdown("""
            - **How it works:** A high-speed variation that replaces the heavy autoregressive transformer rollout with a feedforward network block.
            - **Key Advantage:** Reduces latency below 5ms, allowing direct integration inside 250Hz real-time robot controllers.
            """)
        elif model == "Demo-JEPA":
            st.markdown("""
            - **How it works:** Uses Meta's V-JEPA 2.1 ViT-Giant encoder to extract 256 spatio-temporal tokens ($256 \\times 1408$). The Dreamer Predictor cross-attends demonstration frames to synthesize latent subgoals, and a CEM planner optimizes continuous 7-DoF robot actions in latent feature space.
            - **Key Advantage:** Operates entirely in abstract embedding space without pixel reconstruction artifacts, achieving natural cross-embodiment generalization (Franka $\\leftrightarrow$ Sawyer $\\leftrightarrow$ Fairino).
            """)
        elif model == "Dreamer (RSSM)":
            st.markdown("""
            - **How it works:** Learns a Recurrent State Space Model (RSSM) containing both deterministic (GRU) and stochastic (sampled Gaussian) components.
            - **Key Advantage:** Reconstructs actual RGB frames to predict the environment reward signals, enabling model-based reinforcement learning inside imagination.
            """)

st.write("---")

st.markdown("### ⚖️ Real-World Benchmarking: VILMA vs. OSVI-WM")
st.write("This table presents objective comparative metrics between **VILMA** (our tracking framework) and the **OSVI-WM** world model:")

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
        "[0.35, 0.70] m",
        "[-0.20, 0.20] m",
        "[0.00, 0.50] m"
    ],
    "OSVI-WM (Predictive Latent World Model)": [
        "Monocular RGB Video",
        "15 frames (linear downsampling)",
        "15 waypoints (5 main execution stages)",
        "~14.5 ms (GPU PyTorch Forward Pass)",
        "Optional (resolves depth via latent regression)",
        "[0.40, 0.68] m",
        "[-0.18, 0.18] m",
        "[0.02, 0.48] m"
    ],
    "Demo-JEPA (Joint-Space Policy)": [
        "Multi-View RGB Video (Front + Wrist)",
        "Continuous (Denoised horizon)",
        "5 joint actions (qpos sequences)",
        "~22.4 ms (GPU ViT-Giant + Diffusion Denoising)",
        "None (Fully latent representations)",
        "N/A (Direct Joint 1 Radian)",
        "N/A (Direct Joint 2 Radian)",
        "N/A (Direct Joint 3 Radian)"
    ]
}

st.dataframe(pd.DataFrame(comparison_payload).set_index("Metric Description"), use_container_width=True)

st.info("💡 **Methodology Note:** "
        "VILMA provides high-speed reactive coordinate tracking from stereo depth cameras but requires active physical observation. "
        "OSVI-WM operates on a single flat monocular camera video to infer latent subgoals and predict trajectories under a learned forward world model dynamics.")
