import streamlit as st
import sys
import os
import re

# Adjust path to import files
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from core.manager import get_manager
from core.explanations import OSVI_PAPER_SECTIONS, JEPA_PAPER_SECTIONS

manager = get_manager()
active_model = manager.active_backend_name

st.markdown(f"## 📖 Interactive Paper Reader ({active_model})")
st.write(f"Cross-reference **{active_model}** paper formulations directly with active codebase implementation on the server.")

sections = OSVI_PAPER_SECTIONS if active_model == "OSVI-WM" else JEPA_PAPER_SECTIONS

# Helper to read code lines directly from server files
def load_code_lines(code_link):
    try:
        parts = code_link.split("#")
        rel_path = parts[0]
        
        # Check repository root first, then fallback to ~/osvi-wm
        full_path = os.path.join(ROOT_DIR, rel_path)
        if not os.path.exists(full_path):
            full_path = os.path.expanduser(os.path.join("~/osvi-wm", rel_path))
            
        if not os.path.exists(full_path):
            return f"Code file not found at: {full_path}"
            
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if len(parts) > 1:
            range_str = parts[1]
            # Strip non-digits except hyphen
            clean_range = re.sub(r'[^0-9\-]', '', range_str)
            range_parts = clean_range.split("-")
            
            if len(range_parts) >= 1 and range_parts[0].isdigit():
                start = max(0, int(range_parts[0]) - 1)
                end = int(range_parts[1]) if (len(range_parts) > 1 and range_parts[1].isdigit()) else start + 25
                selected_lines = lines[start:end]
                numbered = [f"{start + i + 1:4d}: {line}" for i, line in enumerate(selected_lines)]
                return "".join(numbered)
        
        return "".join([f"{i + 1:4d}: {line}" for i, line in enumerate(lines[:100])])
    except Exception as e:
        return f"Error reading code lines: {e}"

col_paper, col_code = st.columns([1, 1])

with col_paper:
    st.markdown("### 📝 Paper Sections")
    st.write("Click a section of the paper to highlight the corresponding implementation details:")
    
    session_key = f"selected_section_idx_{active_model}"
    if session_key not in st.session_state or st.session_state[session_key] >= len(sections):
        st.session_state[session_key] = 0
        
    for idx, sec in enumerate(sections):
        is_selected = st.session_state[session_key] == idx
        label = f"✨ {sec['title']}" if is_selected else sec['title']
        
        if st.button(label, key=f"sec_btn_{active_model}_{idx}", use_container_width=True):
            st.session_state[session_key] = idx
            
    active_sec = sections[st.session_state[session_key]]
    
    st.write("---")
    st.markdown(f"#### 📖 Section Details: {active_sec['title']}")
    st.info(active_sec['summary'])
    
    st.markdown("**Mapped Visualization Page:**")
    st.markdown(f"👉 Visit page `{active_sec['visual_page']}` in the sidebar to visualize this component.")
    
    st.markdown("**Relevant Tensor Output:**")
    st.markdown(f"🔍 Inspect tensor `{active_sec['tensor']}` in the `Tensor Explorer` tab.")

with col_code:
    st.markdown("### 💻 Live Implementation Code")
    st.write(f"Displaying implementation file: `{active_sec['code_link']}`")
    
    code_text = load_code_lines(active_sec['code_link'])
    st.code(code_text, language="python")
