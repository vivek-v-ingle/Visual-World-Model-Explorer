import streamlit as st
import sys
import os
import numpy as np
import torch
import plotly.express as px

# Adjust path to import files
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from core.explanations import TENSOR_DESCRIPTIONS


st.markdown("## 🔍 Central Tensor Explorer")
st.write("Inspect and probe any intermediate activation tensor in the model's computation graph.")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to the **03 Input Explorer** page and click **Run Inference** first.")
else:
    tensors = st.session_state['captured_tensors']
    
    available_tensors = list(TENSOR_DESCRIPTIONS.keys())
    available_tensors = [t for t in available_tensors if t in tensors]
    
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        selected_tensor_name = st.selectbox("Select Tensor to Probe", available_tensors)
        
        desc = TENSOR_DESCRIPTIONS[selected_tensor_name]
        st.markdown(f"**Description:** {desc['meaning']}")
        st.markdown(f"**Expected Shape:** `{desc['shape']}`")
        st.markdown(f"**Role in pipeline:** {desc['why']}")
        
        t_data = tensors[selected_tensor_name]
        
        st.divider()
        st.markdown("### 📊 Tensor Statistics")
        if isinstance(t_data, torch.Tensor):
            t_np = t_data.detach().cpu().float().numpy()
        elif isinstance(t_data, list):
            t_np = np.stack([t.detach().cpu().numpy() for t in t_data])
        else:
            t_np = np.array(t_data)
            
        st.metric("Actual Shape", str(list(t_np.shape)))
        st.metric("Minimum Value", f"{t_np.min():.5f}")
        st.metric("Maximum Value", f"{t_np.max():.5f}")
        st.metric("Mean Value", f"{t_np.mean():.5f}")
        st.metric("Std Deviation", f"{t_np.std():.5f}")
        
    with col_r:
        st.markdown("### 👁️ Tensor Visualization")
        
        dims = len(t_np.shape)
        
        if dims == 1:
            st.write("Visualized as a line chart:")
            st.line_chart(t_np)
            st.dataframe(t_np)
            
        elif dims == 2:
            st.write("Visualized as a 2D Heatmap:")
            fig = px.imshow(t_np, color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(t_np)
            
        elif dims == 3:
            st.write("Visualized as slices of a 3D Tensor:")
            max_v = t_np.shape[0] - 1
            if max_v > 0:
                slice_dim = st.slider("Select Slice Index", 0, max_v, 0)
            else:
                slice_dim = 0
                st.write("Slice Index: 0 (Size is 1)")
            slice_data = t_np[slice_dim]
            fig = px.imshow(slice_data, color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(slice_data)
            
        elif dims == 4:
            st.write("Visualized as slices of a 4D Tensor:")
            max1 = t_np.shape[0] - 1
            max2 = t_np.shape[1] - 1
            
            if max1 > 0:
                dim1 = st.slider("Select Dim 1 Index (e.g., Head/Batch)", 0, max1, 0)
            else:
                dim1 = 0
                st.write("Dim 1 Index: 0 (Size is 1)")
                
            if max2 > 0:
                dim2 = st.slider("Select Dim 2 Index (e.g., Step/Channel)", 0, max2, 0)
            else:
                dim2 = 0
                st.write("Dim 2 Index: 0 (Size is 1)")
                
            slice_data = t_np[dim1, dim2]
            fig = px.imshow(slice_data, color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(slice_data)
            
        elif dims == 5:
            st.write("Visualized as slices of a 5D Tensor:")
            max1 = t_np.shape[0] - 1
            max2 = t_np.shape[1] - 1
            max3 = t_np.shape[2] - 1
            
            if max1 > 0:
                dim1 = st.slider("Select Dim 1 Index (Batch)", 0, max1, 0)
            else:
                dim1 = 0
                st.write("Dim 1 Index: 0 (Size is 1)")
                
            if max2 > 0:
                dim2 = st.slider("Select Dim 2 Index (Time Step)", 0, max2, 0)
            else:
                dim2 = 0
                st.write("Dim 2 Index: 0 (Size is 1)")
                
            if max3 > 0:
                dim3 = st.slider("Select Dim 3 Index (Channel/RGB)", 0, max3, 0)
            else:
                dim3 = 0
                st.write("Dim 3 Index: 0 (Size is 1)")
                
            slice_data = t_np[dim1, dim2, dim3]
            fig = px.imshow(slice_data, color_continuous_scale='gray')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(slice_data)
            
        else:
            st.write(f"Cannot render tensor of dimension {dims} automatically.")
            st.write(t_np.flatten()[:100])
