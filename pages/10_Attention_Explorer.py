import streamlit as st
import sys
import numpy as np

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from visualizers.plots import plot_interactive_attention

st.set_page_config(page_title="Attention Explorer - World Model Explorer", layout="wide")

st.markdown("## 🔍 Central Attention Explorer")
st.write("Compare the attention distributions between temporal self-attention (Action Model), predictive transition attention (Forward Model), and cross-attention pooling layers.")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to the **03 Input Explorer** page and click **Run Inference** first.")
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
            
            num_heads = attn_val.shape[1] if len(attn_val.shape) > 1 else 1
            selected_head = st.slider("Select Head Index", 0, num_heads - 1, 0)
            
        with col_r:
            attn_np = attn_val[0, selected_head].cpu().numpy() if len(attn_val.shape) > 2 else attn_val.cpu().numpy()
            
            # Simple downscaling or slicing if too large
            if attn_np.shape[0] > 120:
                st.info("💡 Attention size is large. Displaying first 100 entries.")
                attn_np = attn_np[:100, :100]
                
            fig = plot_interactive_attention(attn_np, title=f"Attention Distribution: {selected_key}")
            st.plotly_chart(fig, use_container_width=True)
