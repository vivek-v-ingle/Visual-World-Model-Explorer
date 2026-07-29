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
    "Encoder Type": ["ResNet-18 / ResNet-50", "Dino-v2 (Frozen)", "ViT (Joint-Embedding)", "CNN Encoder"],
    "Latent Dimension ($Z$)": ["512 x 8 x 10 (Spatial Maps)", "1024 (Dense Feature Vector)", "384 (Semantic Latents)", "1024 (Stochastic + Deterministic)"],
    "Predictive Transition (Rollout)": ["Autoregressive Transformer", "Non-autoregressive MLP Block", "Causal Attention Predictor", "Recurrent SSM (RSSM)"],
    "Planning Decoder": ["Attentive Pooling + MLP Head", "MLP Head", "Cross-Attention Pooler", "Pixel Reconstruction + Policy Head"],
    "Primary Use-Case": ["One-Shot Trajectory Imitation", "Real-time High-frequency Control", "Semantic Video Representation", "Model-based Reinforcement Learning"],
    "Inference Latency (ms)": [14.5, 4.2, 18.2, 32.5]
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
            - **How it works:** Based on Yann LeCun's Joint-Embedding Predictive Architecture. It does not predict future *pixels*, but predicts future *semantic features* using a self-supervised embedding loss.
            - **Key Advantage:** Avoids pixel-space blur and task-irrelevant detail entirely, making it highly robust to background distractions.
            """)
        elif model == "Dreamer (RSSM)":
            st.markdown("""
            - **How it works:** Learns a Recurrent State Space Model (RSSM) containing both deterministic (GRU) and stochastic (sampled Gaussian) components.
            - **Key Advantage:** Reconstructs actual RGB frames to predict the environment reward signals, enabling model-based reinforcement learning inside imagination.
            """)
