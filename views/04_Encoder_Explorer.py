import streamlit as st
import sys
import os
import numpy as np

# Adjust path to import files
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from core.manager import render_debugger_navigation
from visualizers.plots import render_feature_heatmap, preprocess_image_tensor


st.markdown("## 🧠 Stage 2: Shared ResNet Encoder")
st.write("Visualizes how the convolutional encoder filters raw RGB frames into deep feature coordinate maps.")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to the **03 Input Explorer** page and click **Run Inference** first.")
else:
    tensors = st.session_state['captured_tensors']
    images = tensors['images'] # [1, 1, 3, 240, 320]
    context = tensors['context'] # [1, 10, 3, 240, 320]
    resnet_features = tensors['resnet_features_raw'] # [B, 11, 512, 8, 10]
    
    T_context = context.shape[1]
    T_pair = images.shape[1]
    
    input_frames = []
    frame_labels = []
    for t in range(T_context):
        input_frames.append(context[0, t].cpu().numpy())
        frame_labels.append(f"Context Frame E_{t+1}")
    for t in range(T_pair):
        input_frames.append(images[0, t].cpu().numpy())
        frame_labels.append(f"Agent Frame R_1")
        
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Inspection Parameters")
        selected_frame = st.selectbox("Select Frame to Probe", range(len(frame_labels)), format_func=lambda idx: frame_labels[idx])
        selected_channel = st.slider("Select ResNet Channel", 0, resnet_features.shape[2] - 1, 128)
        alpha = st.slider("Heatmap Transparency", 0.0, 1.0, 0.6)
        
        st.metric("Raw Features Shape", str(list(resnet_features.shape)))
        
    with col2:
        img_raw = input_frames[selected_frame]
        feat_raw = resnet_features[0, selected_frame]
        
        blended, _ = render_feature_heatmap(img_raw, feat_raw, selected_channel, alpha=alpha)
        
        c_vis1, c_vis2 = st.columns(2)
        with c_vis1:
            st.image(preprocess_image_tensor(img_raw), caption="Original RGB Frame", use_container_width=True)
        with c_vis2:
            st.image(blended, caption=f"Channel {selected_channel} Activation Overlaid", use_container_width=True)
            
        st.markdown("**Feature Activation Grid Values (8 x 10):**")
        st.dataframe(feat_raw[selected_channel].cpu().numpy())

# Render debug timeline
render_debugger_navigation("views/04_Encoder_Explorer.py")
