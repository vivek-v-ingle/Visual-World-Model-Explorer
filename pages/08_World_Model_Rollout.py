import streamlit as st
import sys
import numpy as np
import time

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from visualizers.plots import render_feature_heatmap, preprocess_image_tensor

st.set_page_config(page_title="World Model Rollout - World Model Explorer", layout="wide")

st.markdown("## 🎞️ World Model Rollout: Imagined Latent Simulation")
st.write("Play or scrub through the world model's mental rollout sequence over time.")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to the **03 Input Explorer** page and click **Run Inference** first.")
else:
    tensors = st.session_state['captured_tensors']
    images = tensors['images'] # [1, 1, 3, 240, 320]
    predicted_states = tensors['predicted_latent_states'] # [1, T_pred, 512, 8, 10]
    
    T_pred = predicted_states.shape[1]
    
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.markdown("### 🎛️ Rollout Controls")
        autoplay = st.checkbox("Auto-play Rollout Timeline", value=False)
        selected_step = st.slider("Select Rollout Timeline Step", 1, T_pred, 1)
        selected_channel = st.slider("Select Channel", 0, predicted_states.shape[2] - 1, 128)
        alpha = st.slider("Heatmap Overlay Alpha", 0.0, 1.0, 0.7)
        
        st.metric("Imagined Horizon Length", f"{T_pred} steps")
        
    with col_r:
        st.markdown("### 📺 Rollout Playback Screen")
        
        if autoplay:
            placeholder = st.empty()
            while autoplay:
                for t in range(T_pred):
                    img = images[0, 0].cpu().numpy()
                    feat = predicted_states[0, t]
                    blended, _ = render_feature_heatmap(img, feat, selected_channel, alpha=alpha)
                    
                    with placeholder.container():
                        st.image(blended, caption=f"Imagined Future step {t+1} of {T_pred} (Channel {selected_channel})", use_container_width=True)
                    time.sleep(0.6)
        else:
            img = images[0, 0].cpu().numpy()
            feat = predicted_states[0, selected_step-1]
            blended, _ = render_feature_heatmap(img, feat, selected_channel, alpha=alpha)
            st.image(blended, caption=f"Imagined Future step {selected_step} (Channel {selected_channel})", use_container_width=True)
