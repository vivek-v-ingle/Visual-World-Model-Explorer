import streamlit as st
import sys
import os
import numpy as np

# Adjust path to import files
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from core.manager import render_debugger_navigation
from visualizers.plots import plot_spatial_softmax_distribution


st.markdown("## 📍 Stage 6: Spatial Embedding (Spatial Softmax)")
st.write("Visualizes how a 2D probability distribution collapses the convolutional feature map to a single $(x, y)$ expected coordinate node.")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to the **03 Input Explorer** page and click **Run Inference** first.")
else:
    tensors = st.session_state['captured_tensors']
    softmax_mask = tensors['spatial_softmax_mask'] # [B, T_rollout, 512, 8, 10]
    spatial_coords = tensors['spatial_coords'] # [B, T_rollout, 1024]
    
    T_rollout = softmax_mask.shape[1]
    
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.markdown("### 🎛️ Parameters")
        selected_step = st.slider("Select Rollout Step", 1, T_rollout, 1)
        selected_channel = st.slider("Select Channel", 0, softmax_mask.shape[2] - 1, 128)
        
        h_coord = spatial_coords[0, selected_step-1, selected_channel].item()
        w_coord = spatial_coords[0, selected_step-1, 512 + selected_channel].item()
        
        st.write("---")
        st.markdown("**Expected Coordinate Center Output:**")
        st.metric("Y Coordinate (Height)", f"{h_coord:.4f}")
        st.metric("X Coordinate (Width)", f"{w_coord:.4f}")
        
    with col_r:
        softmax_slice = softmax_mask[0, selected_step-1]
        fig = plot_spatial_softmax_distribution(
            softmax_slice,
            selected_channel,
            expected_h=h_coord,
            expected_w=w_coord
        )
        st.plotly_chart(fig, use_container_width=True)

# Render debug timeline
render_debugger_navigation("views/09_Spatial_Embedding.py")
