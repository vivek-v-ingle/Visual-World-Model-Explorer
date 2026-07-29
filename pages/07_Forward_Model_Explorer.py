import streamlit as st
import sys
import numpy as np

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from core.manager import render_debugger_navigation
from visualizers.plots import plot_interactive_attention

st.set_page_config(page_title="Forward Model Explorer - World Model Explorer", layout="wide")

st.markdown("## 🔮 Stage 5: Forward Model Explorer")
st.write("Examine the self-attention weights inside the causal Transformer blocks of the Forward Transition Model.")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to the **03 Input Explorer** page and click **Run Inference** first.")
else:
    tensors = st.session_state['captured_tensors']
    
    forward_keys = [k for k in tensors.keys() if "forward_model_attn_block" in k]
    
    if len(forward_keys) == 0:
        st.error("No forward model attention maps captured.")
    else:
        col_l, col_r = st.columns([1, 2])
        
        with col_l:
            st.markdown("### 🎛️ Parameters")
            selected_block = st.selectbox("Select Forward Block", forward_keys)
            block_tensor = tensors[selected_block] # [B, heads, T_seq, T_seq]
            
            st.metric("Attention Shape", str(list(block_tensor.shape)))
            
            num_heads = block_tensor.shape[1]
            selected_head = st.slider("Select Head", 0, num_heads - 1, 0)
            
        with col_r:
            attn_matrix = block_tensor[0, selected_head].cpu().numpy()
            
            fig = plot_interactive_attention(
                attn_matrix,
                title=f"{selected_block} (Head {selected_head}) Self-Attention Matrix",
                x_labels=[f"Step {t+1}" for t in range(attn_matrix.shape[1])],
                y_labels=[f"Step {t+1}" for t in range(attn_matrix.shape[0])]
            )
            st.plotly_chart(fig, use_container_width=True)

# Render debug timeline
render_debugger_navigation("pages/07_Forward_Model_Explorer.py")
