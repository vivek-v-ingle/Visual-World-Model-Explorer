import streamlit as st
import sys
import os
import numpy as np

# Adjust path to import files
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from visualizers.plots import plot_interactive_attention

st.markdown("## 🔍 Central Attention Explorer")
st.write("Compare the attention distributions between temporal self-attention (Action Model), predictive transition attention (Forward Model), and cross-attention pooling layers.")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to the **03 Input Explorer** or **Real Demo-JEPA Pipeline** page and run inference first.")
else:
    tensors = st.session_state['captured_tensors']
    
    # Extract all attention tensors
    attn_keys = [k for k in tensors.keys() if "attn" in k or "attention" in k]
    
    if len(attn_keys) == 0:
        st.error("No attention matrices found in captured session state.")
    else:
        col_l, col_r = st.columns([1, 2])
        
        with col_l:
            st.markdown("### 🎛️ Parameters")
            selected_key = st.selectbox("Select Attention Map to Inspect", attn_keys)
            attn_val = tensors[selected_key]
            
            st.metric("Attention Tensor Shape", str(list(attn_val.shape)))
            
            ndim = attn_val.ndim
            if ndim == 4:
                # [B, heads, T_q, T_k]
                B, H, Tq, Tk = attn_val.shape
                if H > 1:
                    selected_head = st.slider("Select Head Index", 0, H - 1, 0)
                else:
                    selected_head = 0
                    st.info("Single-head attention map.")
                attn_matrix = attn_val[0, selected_head].detach().cpu().numpy()
            elif ndim == 3:
                # [B, T_q, T_k] or [heads, T_q, T_k]
                if attn_val.shape[0] == 1:
                    st.info("Single-head attention map [1, T_q, T_k].")
                    attn_matrix = attn_val[0].detach().cpu().numpy()
                else:
                    H = attn_val.shape[0]
                    selected_head = st.slider("Select Head Index", 0, H - 1, 0)
                    attn_matrix = attn_val[selected_head].detach().cpu().numpy()
            elif ndim == 2:
                st.info("2D attention matrix.")
                attn_matrix = attn_val.detach().cpu().numpy()
            else:
                attn_matrix = attn_val.reshape(-1, attn_val.shape[-1]).detach().cpu().numpy()
            
            if attn_matrix.ndim == 1:
                attn_matrix = attn_matrix.reshape(1, -1)
                
        with col_r:
            # Simple downscaling or slicing if too large
            if attn_matrix.shape[0] > 120 or attn_matrix.shape[1] > 120:
                st.info("💡 Attention matrix is large. Displaying top-left 100x100 entries.")
                attn_matrix = attn_matrix[:100, :100]
                
            fig = plot_interactive_attention(attn_matrix, title=f"Attention Distribution: {selected_key}")
            st.plotly_chart(fig, use_container_width=True)
