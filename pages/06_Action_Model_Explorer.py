import streamlit as st
import sys
import numpy as np

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from core.manager import render_debugger_navigation
from visualizers.plots import plot_interactive_attention

st.set_page_config(page_title="Action Model Explorer - World Model Explorer", layout="wide")

st.markdown("## 🎬 Stage 4: Action Model Explorer")
st.write("Examine spatiotemporal causal self-attention weights inside the Action Model network block.")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to the **03 Input Explorer** page and click **Run Inference** first.")
else:
    tensors = st.session_state['captured_tensors']
    
    available_layers = [k for k in tensors.keys() if "action_attn_layer" in k]
    
    if len(available_layers) == 0:
        st.error("No action attention maps captured.")
    else:
        col_l, col_r = st.columns([1, 2])
        
        with col_l:
            st.markdown("### 🎛️ Parameters")
            selected_layer = st.selectbox("Select Attention Layer", available_layers)
            attn_tensor = tensors[selected_layer] # [B, heads, T*80, T*80]
            
            st.metric("Raw Attention Shape", str(list(attn_tensor.shape)))
            
            num_heads = attn_tensor.shape[1]
            selected_head = st.slider("Select Head", 0, num_heads - 1, 0)
            
            view_mode = st.radio("Visualization Mode", [
                "Temporal Attention Map (Pooled over Space)",
                "Full Spatiotemporal Attention (First 100 tokens)"
            ])
            
        with col_r:
            # Safe numpy CPU array conversion
            attn_head = attn_tensor[0, selected_head].cpu().numpy()
            
            T = int(attn_head.shape[0] / 80)
            
            if view_mode == "Temporal Attention Map (Pooled over Space)":
                attn_temp = attn_head.reshape(T, 80, T, 80).mean(axis=(1, 3))
                row_sums = attn_temp.sum(axis=1, keepdims=True)
                attn_temp_norm = np.divide(attn_temp, row_sums, out=np.zeros_like(attn_temp), where=row_sums!=0)
                
                x_labels = [f"Step {t+1}" for t in range(T)]
                y_labels = [f"Step {t+1}" for t in range(T)]
                
                fig = plot_interactive_attention(
                    attn_temp_norm,
                    title=f"Temporal Self-Attention - {selected_layer} (Head {selected_head})",
                    x_labels=x_labels,
                    y_labels=y_labels
                )
            else:
                slice_size = min(100, attn_head.shape[0])
                fig = plot_interactive_attention(
                    attn_head[:slice_size, :slice_size],
                    title=f"Spatiotemporal Attention Slice - {selected_layer} (Head {selected_head})"
                )
                
            st.plotly_chart(fig, use_container_width=True)

# Render debug timeline
render_debugger_navigation("pages/06_Action_Model_Explorer.py")
