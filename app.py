import streamlit as st
import sys
import os

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from core.manager import init_shared_state, get_manager

# Configure page settings
st.set_page_config(
    page_title="Visual World Model Explorer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global premium theme and styles
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #1a1f2c !important;
        border-right: 1px solid #2e3748;
    }
    div.stButton > button {
        background-color: #6366f1 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background-color: #4f46e5 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    .premium-header {
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .premium-subheader {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State variables
init_shared_state()
manager = get_manager()

# Layout
st.markdown('<div class="premium-header">🔮 Visual World Model Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="premium-subheader">Interactive Debugger & Educational Platform for Robotics World Models</div>', unsafe_allow_html=True)

st.divider()

# Welcome and Features Card
col_intro, col_side = st.columns([2, 1])

with col_intro:
    st.markdown("""
    ### Welcome to the World Model Explorer!
    This application is designed to help researchers, students, and engineers understand how **latent world models** think. 
    Rather than acting as a deployment black-box, this platform behaves like a **neural network debugger** to step through every stage of inference:
    
    1. **Deconstruct intermediate tensors:** Inspect feature maps, activation heatmaps, and spatial coordinate grids.
    2. **Visualize spatiotemporal self-attention:** Understand Q, K, V matching over time and space.
    3. **Compare architectures side-by-side:** Examine structural differences between OSVI-WM, FastWAM, and Demo-JEPA.
    4. **Differentiable Spatial Softmax Sandbox:** Click on grid nodes or edit Query/Key sliders to see math calculations update live!
    """)
    
    # Large Merging Flow Diagram
    st.info("💡 **Debugger Workflow:** Use the sidebar to navigate to the pages, or start by configuring inputs in **03 Input Explorer**!")

with col_side:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Quick Status")
    st.write(f"Active Backend: `{manager.active_backend_name}`")
    
    if st.session_state["captured_tensors"] is None:
        st.warning("⚠️ No active inference session. Go to **03 Input Explorer** to load a trajectory and run model inference.")
    else:
        st.success("✓ Inference Active. Tensors ready for visualization.")
        
    st.write(f"Available Backends: `{list(manager.backends.keys())}`")
    st.markdown('</div>', unsafe_allow_html=True)
