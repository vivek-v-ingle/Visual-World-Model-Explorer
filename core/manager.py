import streamlit as st
from typing import Dict, List, Any, Optional
from core.types import BaseModelBackend

DEBUGGER_STAGES = [
    "Input Module",
    "Shared ResNet Encoder",
    "Latent Space Explorer",
    "Action Model",
    "Forward Model",
    "Spatial Embedding",
    "Temporal Pooling",
    "Waypoint Decoder",
    "Final Predictions"
]

class VisualExplorerManager:
    """
    Central manager that orchestrates loaded backends, runs inference,
    manages debugger steps, and handles app state.
    """
    def __init__(self):
        self.backends: Dict[str, BaseModelBackend] = {}
        self.active_backend_name: Optional[str] = None
        
    def register_backend(self, name: str, backend: BaseModelBackend) -> None:
        """Register a new world model backend plugin."""
        self.backends[name] = backend
        if self.active_backend_name is None:
            self.active_backend_name = name
            
    def get_active_backend(self) -> Optional[BaseModelBackend]:
        """Get the active backend instance."""
        if self.active_backend_name is None or self.active_backend_name not in self.backends:
            return None
        return self.backends[self.active_backend_name]
        
    def set_active_backend(self, name: str) -> None:
        """Switch the active backend plugin."""
        if name in self.backends:
            self.active_backend_name = name
            
    def get_stages(self) -> List[str]:
        """Return the list of neural network debugger stages."""
        return DEBUGGER_STAGES

# Helper to retrieve/initialize the manager inside Streamlit Session State
def get_manager() -> VisualExplorerManager:
    if "manager" not in st.session_state:
        manager = VisualExplorerManager()
        # Lazily register backends
        from backends.osvi.osvi_backend import OSVIWorldModelBackend
        manager.register_backend("OSVI-WM", OSVIWorldModelBackend())
        
        # We can add backends
        from backends.osvi.osvi_backend import MockFastWAMBackend
        from backends.jepa.jepa_backend import DemoJEPABackend
        manager.register_backend("FastWAM (Mock)", MockFastWAMBackend())
        manager.register_backend("Demo-JEPA", DemoJEPABackend())
        
        st.session_state["manager"] = manager
        
    return st.session_state["manager"]

# Initialize shared states in Session State
def init_shared_state():
    if "active_stage_idx" not in st.session_state:
        st.session_state["active_stage_idx"] = 0
    if "loaded_trajectory" not in st.session_state:
        st.session_state["loaded_trajectory"] = None
    if "captured_tensors" not in st.session_state:
        st.session_state["captured_tensors"] = None
    if "selected_checkpoint" not in st.session_state:
        st.session_state["selected_checkpoint"] = None
    if "selected_trajectory_file" not in st.session_state:
        st.session_state["selected_trajectory_file"] = None

DEBUGGER_PAGES = [
    ("Input Module", "views/03_Input_Explorer.py"),
    ("Shared ResNet Encoder", "views/04_Encoder_Explorer.py"),
    ("Latent Space Explorer", "views/05_Latent_Space_Explorer.py"),
    ("Action Model", "views/06_Action_Model_Explorer.py"),
    ("Forward Model", "views/07_Forward_Model_Explorer.py"),
    ("Spatial Embedding", "views/09_Spatial_Embedding.py"),
    ("Temporal Pooling", "views/11_Temporal_Pooling.py"),
    ("Waypoint Decoder", "views/12_Waypoint_Decoder.py"),
    ("Final Predictions", "views/13_Trajectory_Explorer.py")
]

def render_debugger_navigation(current_page_name: str):
    """
    Renders step-by-step navigation buttons and a progress track at the bottom of pages.
    """
    # Find current page index
    current_idx = -1
    for i, (name, path) in enumerate(DEBUGGER_PAGES):
        if current_page_name in path or path in current_page_name:
            current_idx = i
            break
            
    if current_idx == -1:
        return
        
    st.write("---")
    st.write(f"**⚡ Neural Network Debugger Timeline:**")
    
    # Render steps indicator
    step_items = []
    for i, (name, _) in enumerate(DEBUGGER_PAGES):
        if i == current_idx:
            step_items.append(f"**[{name}]**")
        elif i < current_idx:
            step_items.append(f"~~{name}~~")
        else:
            step_items.append(name)
            
    st.markdown(" → ".join(step_items))
    
    col_prev, col_status, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if current_idx > 0:
            if st.button("⬅ Previous Stage", use_container_width=True):
                st.session_state["active_stage_idx"] = current_idx - 1
                st.switch_page(DEBUGGER_PAGES[current_idx - 1][1])
        else:
            st.button("⬅ Previous Stage", disabled=True, use_container_width=True)
            
    with col_status:
        st.markdown(f"<div style='text-align: center; font-weight: bold; color: #a5b4fc;'>Stage {current_idx+1} of {len(DEBUGGER_PAGES)}</div>", unsafe_allow_html=True)
        
    with col_next:
        if current_idx < len(DEBUGGER_PAGES) - 1:
            if st.button("Next Stage ➡", use_container_width=True):
                st.session_state["active_stage_idx"] = current_idx + 1
                st.switch_page(DEBUGGER_PAGES[current_idx + 1][1])
        else:
            st.button("Next Stage ➡", disabled=True, use_container_width=True)

