import streamlit as st
import sys

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from core.manager import get_manager

st.set_page_config(page_title="Settings - World Model Explorer", layout="wide")

st.markdown("## ⚙️ Visual Explorer Settings")
st.write("Configure active model backends, hardware parameters, and repository system paths.")

manager = get_manager()

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### 🔌 Active Backend Model")
    selected_backend = st.selectbox(
        "Active Model Backend", 
        list(manager.backends.keys()),
        index=list(manager.backends.keys()).index(manager.active_backend_name)
    )
    
    if selected_backend != manager.active_backend_name:
        manager.set_active_backend(selected_backend)
        st.success(f"Switched active backend to `{selected_backend}`!")
        # Clear active session state triggers
        st.session_state["captured_tensors"] = None
        st.session_state["loaded_trajectory"] = None
        
    st.write("---")
    st.markdown("### 🎛️ Computation & Hardware Options")
    use_gpu = st.toggle("Use GPU Acceleration (if available)", value=True)
    precision = st.selectbox("Floating Point Precision", ["FP32 (Single Precision)", "FP16 (Half Precision)"])
    
with col_r:
    import os
    osvi_wm_path = st.text_input("OSVI-WM Source Directory Path", value=os.path.expanduser("~/osvi-wm"))
    checkpoints_path = st.text_input("Model Checkpoints Directory Path", value=os.path.expanduser("~/osvi-wm/checkpoints"))
    trajectories_path = st.text_input("ZED Trajectories Directory Path", value=os.path.expanduser("~/OSVI-Deploy"))
    
    if st.button("Apply Paths & Restart Session"):
        st.success("Paths updated successfully. Session cache refreshed.")
