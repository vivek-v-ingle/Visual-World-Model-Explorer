import streamlit as st
import sys
import numpy as np
import pandas as pd

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from core.manager import render_debugger_navigation

st.set_page_config(page_title="Waypoint Decoder - World Model Explorer", layout="wide")

st.markdown("## 📊 Stage 8: Waypoint Decoder (MLP Planner Head)")
st.write("Examine the final sequence of predicted image-space spatial waypoints before world-coordinate projection.")

if 'captured_tensors' not in st.session_state or st.session_state['captured_tensors'] is None:
    st.warning("⚠️ No active inference data. Please go to the **03 Input Explorer** page and click **Run Inference** first.")
else:
    tensors = st.session_state['captured_tensors']
    raw_waypoints = tensors['raw_waypoints'] # [B, waypoints, 4]
    
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.markdown("### 🎛️ Waypoints Details")
        st.metric("Raw Waypoints Shape", str(list(raw_waypoints.shape)))
        st.write(f"The model predicted a sequence of **{raw_waypoints.shape[1]} waypoints**.")
        
        st.write("---")
        st.markdown("#### Physical Meaning of Waypoint Channels:")
        st.markdown("""
        - **Channel 1 (u):** Normalized image coordinate $u$ along the width axis (range $[-1, 1]$).
        - **Channel 2 (v):** Normalized image coordinate $v$ along the height axis (range $[-1, 1]$).
        - **Channel 3 (depth):** Projected depth coordinate relative to the camera frame (meters).
        - **Channel 4 (grasp):** Grasp probability/score. A value close to `1.0` means close the gripper; close to `-1.0` or `0.0` means open the gripper.
        """)
        
    with col_r:
        st.markdown("### 📋 Predicted Waypoint Values Sequence")
        st.write("Below are the exact values predicted by the model:")
        
        waypoints_np = raw_waypoints[0].cpu().numpy()
        
        df_wps = pd.DataFrame(
            waypoints_np,
            columns=["u (Width Coord)", "v (Height Coord)", "Depth (meters)", "Grasp Score"],
            index=[f"Waypoint {i+1}" for i in range(len(waypoints_np))]
        )
        
        st.dataframe(df_wps.style.background_gradient(cmap="coolwarm", subset=["Grasp Score"]))
        
        st.markdown("**Grasp Transition Sequence Chart:**")
        st.line_chart(df_wps["Grasp Score"])

# Render debug timeline
render_debugger_navigation("pages/12_Waypoint_Decoder.py")
