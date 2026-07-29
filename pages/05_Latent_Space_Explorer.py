import streamlit as st
import sys
import numpy as np

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from core.manager import render_debugger_navigation
from visualizers.plots import preprocess_image_tensor, render_feature_heatmap

st.set_page_config(page_title="Latent Space Explorer - World Model Explorer", layout="wide")

st.markdown("## 🌌 Stage 3: Latent Space Explorer")
st.write("Examine the latent representations $Z_E$ (expert demonstration features) and $Z_R$ (agent current state features).")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to the **03 Input Explorer** page and click **Run Inference** first.")
else:
    tensors = st.session_state['captured_tensors']
    resnet_features = tensors['resnet_features'] # [B, 11, 512, 8, 10]
    images = tensors['images'] # [1, 1, 3, 240, 320]
    context = tensors['context'] # [1, 10, 3, 240, 320]
    
    T_context = context.shape[1]
    
    ze = resnet_features[:, :T_context]
    zr = resnet_features[:, T_context:]
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown("### 🏷️ Teacher Latent ($Z_E$)")
        st.metric("ZE Shape", str(list(ze.shape)))
        
        selected_step = st.slider("Select Context Frame E_t", 1, T_context, 1)
        selected_channel_e = st.slider("Select ZE Channel", 0, ze.shape[2] - 1, 128, key="ze_c")
        
        img_e = context[0, selected_step-1].cpu().numpy()
        feat_e = ze[0, selected_step-1]
        blended_e, _ = render_feature_heatmap(img_e, feat_e, selected_channel_e, alpha=0.5)
        st.image(blended_e, caption=f"ZE Frame {selected_step} Channel {selected_channel_e}", use_container_width=True)
        
    with col_r:
        st.markdown("### 🤖 Agent Latent ($Z_R$)")
        st.metric("ZR Shape", str(list(zr.shape)))
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        selected_channel_r = st.slider("Select ZR Channel", 0, zr.shape[2] - 1, 128, key="zr_c")
        
        img_r = images[0, 0].cpu().numpy()
        feat_r = zr[0, 0]
        blended_r, _ = render_feature_heatmap(img_r, feat_r, selected_channel_r, alpha=0.5)
        st.image(blended_r, caption=f"ZR Frame 1 Channel {selected_channel_r}", use_container_width=True)

# Render debug timeline
render_debugger_navigation("pages/05_Latent_Space_Explorer.py")
