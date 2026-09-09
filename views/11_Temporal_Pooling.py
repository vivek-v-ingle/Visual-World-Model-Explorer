import streamlit as st
import sys
import os
import numpy as np
import plotly.express as px

# Adjust path to import files
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from core.manager import render_debugger_navigation


st.markdown("## ⚡ Stage 7: Temporal Attentive Pooling")
st.write("Understand how learned query token cross-attends over all temporal rollout steps to capture the essence of the path.")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to the **03 Input Explorer** page and click **Run Inference** first.")
else:
    tensors = st.session_state['captured_tensors']
    pooler_attn = tensors['pooler_attention'] # [B, heads, 1, T_rollout]
    pooled_states = tensors['pooled_states'] # [B, 1, 1024]
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown("### 🔍 Attention Allocation Over Future Rollout Steps")
        
        weights = pooler_attn[0, 0, 0].cpu().numpy()
        
        fig = px.bar(
            x=[f"Imagined Step {t+1}" for t in range(len(weights))],
            y=weights,
            labels={'x': 'Rollout Steps', 'y': 'Attention Weight'},
            title="Attentive Pooler Cross-Attention Weights",
            color=weights,
            color_continuous_scale='Purples'
        )
        fig.update_layout(yaxis_range=[0, 1.0])
        st.plotly_chart(fig, use_container_width=True)
        
    with col_r:
        st.markdown("### 🧬 Pooled State Vector")
        st.write("The pooler aggregates the coordinate features across time into a single vector:")
        st.metric("Pooled State Vector Shape", str(list(pooled_states.shape)))
        
        st.write("Below is a slice of values inside the 1024-dimensional pooled vector:")
        pooled_slice = pooled_states[0, 0, :100].cpu().numpy()
        st.line_chart(pooled_slice)
        st.caption("First 100 features of the Pooled State Vector")

# Render debug timeline
render_debugger_navigation("views/11_Temporal_Pooling.py")
