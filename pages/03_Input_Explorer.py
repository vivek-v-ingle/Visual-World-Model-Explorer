import streamlit as st
import sys
import os

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from core.manager import get_manager, render_debugger_navigation
from visualizers.plots import preprocess_image_tensor

st.set_page_config(page_title="Input Explorer - World Model Explorer", layout="wide")

st.markdown("## 📥 Stage 1: Input Explorer")
st.write("Load demonstration trajectories and trigger step-by-step tensor debugger compilation.")

manager = get_manager()
backend = manager.get_active_backend()

if backend is None:
    st.error("No active backend configured in state manager.")
else:
    # Left: Controls
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.markdown("### ⚙️ Pipeline Configuration")
        
        # Load trajectories
        trajs = backend.get_default_trajectories()
        sel_traj = st.selectbox("Select Trajectory File", list(trajs.keys()))
        traj_path = trajs[sel_traj]
        
        # Checkpoints
        ckpts = backend.get_supported_checkpoints()
        sel_ckpt = st.selectbox("Select Pretrained Model", list(ckpts.keys()))
        ckpt_path = ckpts[sel_ckpt]
        is_metaworld = "metaworld" in sel_ckpt.lower()
        
        # Upload
        uploaded_file = st.file_uploader("Or Upload Custom Episode/Trajectory (.h5, .pkl)", type=["h5", "hdf5", "pkl"])
        if uploaded_file is not None:
            scratch_dir = "/home/vvijaykumar/.gemini/antigravity-ide/brain/25458c0f-e26c-44f2-9682-23adefaf5468/scratch"
            os.makedirs(scratch_dir, exist_ok=True)
            traj_path = os.path.join(scratch_dir, uploaded_file.name)
            with open(traj_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Uploaded custom trajectory.")
            
        st.divider()
        
        if st.button("🚀 Run Inference & Extract Tensors"):
            with st.spinner("Executing pipeline and capturing intermediate activations..."):
                try:
                    # 1. Load data
                    traj_data = backend.load_trajectory(traj_path)
                    st.session_state["loaded_trajectory"] = traj_data
                    
                    # 2. Load model
                    backend.load_model(ckpt_path, metaworld=is_metaworld)
                    
                    # 3. Run step-by-step inference
                    tensors = backend.run_inference(traj_data["images"], traj_data["context"])
                    st.session_state["captured_tensors"] = tensors
                    
                    st.success("Inference completed! Intermediate states captured.")
                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    
    with col_r:
        st.markdown("### 🔍 Input Data Inspector")
        traj_data = st.session_state["loaded_trajectory"]
        
        if traj_data is None:
            st.info("👈 Configure options and click **Run Inference** to begin debugger exploration.")
        else:
            col_sh = st.columns(3)
            with col_sh[0]:
                st.metric("Context Tensors Shape", str(list(traj_data['context'].shape)))
            with col_sh[1]:
                st.metric("Observation Frame Shape", str(list(traj_data['images'].shape)))
            with col_sh[2]:
                st.metric("Projection Matrix Shape", str(list(traj_data['projection_matrix'].shape)))
                
            # Render context frames E1...EN
            st.markdown("**Teacher Context Frames ($E_1 ... E_{10}$):**")
            context_np = traj_data['context'][0].cpu().numpy() # safe CPU copy
            
            c_grid = st.columns(5)
            for i in range(5):
                with c_grid[i]:
                    img = preprocess_image_tensor(context_np[i])
                    st.image(img, caption=f"E {i+1}", use_container_width=True)
            c_grid2 = st.columns(5)
            for i in range(5):
                with c_grid2[i]:
                    img = preprocess_image_tensor(context_np[5+i])
                    st.image(img, caption=f"E {6+i}", use_container_width=True)
                    
            st.markdown("**Agent Initial Observation Frame ($R_1$):**")
            obs_img = preprocess_image_tensor(traj_data['images'][0, 0].cpu().numpy())
            st.image(obs_img, caption="Agent Frame R1", width=320)

    # Render debug controls
    render_debugger_navigation("pages/03_Input_Explorer.py")
