import streamlit as st
import sys
import os

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from core.explanations import PAPER_SECTIONS

st.set_page_config(page_title="Paper Reader - World Model Explorer", layout="wide")

st.markdown("## 📖 Interactive Paper Reader")
st.write("Cross-reference paper formulations directly with active codebase implementation on the server.")

# Helper to read code lines directly from server files
def load_code_lines(code_link):
    try:
        parts = code_link.split("#")
        rel_path = parts[0]
        full_path = os.path.join('/home/vvijaykumar/osvi-wm', rel_path)
        
        if not os.path.exists(full_path):
            return f"Code file not found at: {full_path}"
            
        with open(full_path, 'r') as f:
            lines = f.readlines()
            
        if len(parts) > 1 and parts[1].startswith("L"):
            range_str = parts[1][1:] # strip 'L'
            range_parts = range_str.split("-")
            start = int(range_parts[0]) - 1
            end = int(range_parts[1]) if len(range_parts) > 1 else start + 20
            
            selected_lines = lines[start:end]
            numbered = [f"{start + i + 1:4d}: {line}" for i, line in enumerate(selected_lines)]
            return "".join(numbered)
        else:
            return "".join(lines[:100])
    except Exception as e:
        return f"Error reading code lines: {e}"

col_paper, col_code = st.columns([1, 1])

with col_paper:
    st.markdown("### 📝 Paper Sections")
    st.write("Click a section of the paper to highlight the corresponding implementation details:")
    
    if "selected_section_idx" not in st.session_state:
        st.session_state["selected_section_idx"] = 0
        
    for idx, sec in enumerate(PAPER_SECTIONS):
        is_selected = st.session_state["selected_section_idx"] == idx
        label = f"✨ {sec['title']}" if is_selected else sec['title']
        
        if st.button(label, key=f"sec_btn_{idx}", use_container_width=True):
            st.session_state["selected_section_idx"] = idx
            
    active_sec = PAPER_SECTIONS[st.session_state["selected_section_idx"]]
    
    st.write("---")
    st.markdown(f"#### 📖 Section Details: {active_sec['title']}")
    st.info(active_sec['summary'])
    
    st.markdown("**Mapped Visualization Page:**")
    st.markdown(f"👉 Visit page `{active_sec['visual_page']}` in the sidebar to visualize this component.")
    
    st.markdown("**Relevant Tensor Output:**")
    st.markdown(f"🔍 Inspect tensor `{active_sec['tensor']}` in the `Tensor Explorer` tab.")

with col_code:
    st.markdown("### 💻 Live Implementation Code")
    st.write(f"Displaying implementation file: `osvi-wm/{active_sec['code_link'].split('#')[0]}`")
    
    code_text = load_code_lines(active_sec['code_link'])
    st.code(code_text, language="python")
