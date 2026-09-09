import streamlit as st
import sys
import os

# Adjust path to import files
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from core.manager import init_shared_state, get_manager

# Configure page settings once for the application
st.set_page_config(
    page_title="Visual World Model Explorer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State variables
init_shared_state()
manager = get_manager()

# Apply global premium theme and styles
st.markdown("""
<style>
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
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .premium-subheader {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .model-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .model-badge {
        background-color: #312e81;
        color: #c7d2fe;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .mermaid {
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Sidebar Model Switcher
# -------------------------------------------------------------
st.sidebar.markdown("## 🔮 World Model Hub")
backend_options = ["OSVI-WM", "Demo-JEPA"]
current_idx = backend_options.index(manager.active_backend_name) if manager.active_backend_name in backend_options else 0
selected_backend = st.sidebar.radio(
    "Active World Model:",
    backend_options,
    index=current_idx,
    help="Switching models tailors the entire sidebar and dashboard to display only the selected model's pipeline."
)

if selected_backend != manager.active_backend_name:
    manager.set_active_backend(selected_backend)
    st.session_state["captured_tensors"] = None
    st.session_state["loaded_trajectory"] = None
    st.rerun()

st.sidebar.divider()

# -------------------------------------------------------------
# Dynamic Page Definitions per Active Model
# -------------------------------------------------------------
home_page = st.Page("views/00_Home.py", title="Home Overview", icon="🏠", default=True)

if manager.active_backend_name == "OSVI-WM":
    pages = {
        "Overview": [home_page],
        "📍 OSVI-WM Pipeline": [
            st.Page("views/20_End_to_End_Pipeline.py", title="End-to-End Pipeline", icon="🌐"),
            st.Page("views/02_Architecture_Explorer.py", title="Architecture Explorer", icon="🏗️"),
            st.Page("views/03_Input_Explorer.py", title="Input & Trajectory", icon="📥"),
            st.Page("views/04_Encoder_Explorer.py", title="ResNet Shared Encoder", icon="🔎"),
            st.Page("views/05_Latent_Space_Explorer.py", title="Latent Space (Z)", icon="🌌"),
            st.Page("views/06_Action_Model_Explorer.py", title="Action Model", icon="🎬"),
            st.Page("views/07_Forward_Model_Explorer.py", title="Forward Transition Model", icon="🔮"),
            st.Page("views/09_Spatial_Embedding.py", title="Spatial Softmax", icon="📍"),
            st.Page("views/10_Attention_Explorer.py", title="Attention Heatmaps", icon="🔥"),
            st.Page("views/11_Temporal_Pooling.py", title="Attentive Pooler", icon="⚡"),
            st.Page("views/12_Waypoint_Decoder.py", title="Waypoint Decoder Head", icon="📊"),
            st.Page("views/13_Trajectory_Explorer.py", title="3D Trajectory Viewer", icon="🤖"),
        ],
        "🛠️ Tools & Comparison": [
            st.Page("views/14_Tensor_Explorer.py", title="Tensor Registry", icon="📐"),
            st.Page("views/15_Equation_Explorer.py", title="Equation Sandbox", icon="📝"),
            st.Page("views/16_Paper_Reader.py", title="Paper Reader", icon="📖"),
            st.Page("views/17_Model_Comparison.py", title="Model Comparison", icon="⚖️"),
            st.Page("views/18_Settings.py", title="Settings", icon="⚙️"),
            st.Page("views/19_About.py", title="About", icon="ℹ️"),
        ]
    }
else:
    pages = {
        "Overview": [home_page],
        "🧠 Demo-JEPA Pipeline": [
            st.Page("views/21_Demo_JEPA_Pipeline.py", title="Real Demo-JEPA Pipeline", icon="🧠"),
            st.Page("views/02_Architecture_Explorer.py", title="Architecture Explorer", icon="🏗️"),
            st.Page("views/10_Attention_Explorer.py", title="Cross-Attention Heatmaps", icon="🔥"),
        ],
        "🛠️ Tools & Comparison": [
            st.Page("views/14_Tensor_Explorer.py", title="Tensor Registry", icon="📐"),
            st.Page("views/15_Equation_Explorer.py", title="Equation Sandbox", icon="📝"),
            st.Page("views/16_Paper_Reader.py", title="Paper Reader", icon="📖"),
            st.Page("views/17_Model_Comparison.py", title="Model Comparison", icon="⚖️"),
            st.Page("views/18_Settings.py", title="Settings", icon="⚙️"),
            st.Page("views/19_About.py", title="About", icon="ℹ️"),
        ]
    }

pg = st.navigation(pages)
pg.run()
