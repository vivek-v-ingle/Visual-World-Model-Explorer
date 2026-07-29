import streamlit as st
import sys
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Adjust path to import files
sys.path.insert(0, '/home/vvijaykumar/Visual-World-Model-Explorer')
from core.explanations import EQUATIONS

st.set_page_config(page_title="Equation Explorer - World Model Explorer", layout="wide")

st.markdown("## 🧮 Equation Explorer & Mathematical Sandbox")
st.write("Understand the core mathematical formulations of World Models through interactive playgrounds.")

selected_eq = st.selectbox("Select Equation to Explore", list(EQUATIONS.keys()))

st.divider()

if selected_eq == "Spatial Softmax":
    eq = EQUATIONS["Spatial Softmax"]
    st.markdown(f"### 📍 {selected_eq}")
    st.latex(eq["latex"])
    st.write(eq["explanation"])
    
    st.latex(eq["expected_coords"])
    st.write(eq["expected_coords_explanation"])
    
    st.subheader("🎮 Spatial Softmax Sandbox")
    st.write("Set values in a $3 \\times 3$ feature activation grid to see how the Softmax probability distribution and expected center coordinate $(x_{center}, y_{center})$ change in real-time!")
    
    cols = st.columns(3)
    grid_inputs = np.zeros((3, 3))
    
    for r in range(3):
        for c in range(3):
            with cols[c]:
                grid_inputs[r, c] = st.slider(
                    f"Activation at ({r}, {c})",
                    -10.0, 10.0, 0.0, 0.5,
                    key=f"grid_{r}_{c}"
                )
                
    exp_grid = np.exp(grid_inputs)
    sum_exp = np.sum(exp_grid)
    softmax_grid = exp_grid / sum_exp
    
    coords_y = np.array([-1.0, 0.0, 1.0])
    coords_x = np.array([-1.0, 0.0, 1.0])
    
    expected_y = 0.0
    expected_x = 0.0
    for r in range(3):
        for c in range(3):
            expected_y += coords_y[r] * softmax_grid[r, c]
            expected_x += coords_x[c] * softmax_grid[r, c]
            
    c_heat1, c_heat2 = st.columns(2)
    
    with c_heat1:
        st.markdown("**Softmax Probability Output:**")
        fig = go.Figure(data=go.Heatmap(
            z=softmax_grid,
            x=["X=-1", "X=0", "X=1"],
            y=["Y=-1", "Y=0", "Y=1"],
            colorscale='Hot',
            zmin=0.0,
            zmax=1.0
        ))
        
        plot_x = expected_x + 1.0
        plot_y = expected_y + 1.0
        
        fig.add_trace(go.Scatter(
            x=[plot_x],
            y=[plot_y],
            mode='markers',
            marker=dict(size=18, color='cyan', line=dict(color='black', width=2)),
            name="Expected Center"
        ))
        st.plotly_chart(fig, use_container_width=True)
        
    with c_heat2:
        st.markdown("**Computed Values:**")
        st.metric("Expected X Coordinate Center", f"{expected_x:.4f}")
        st.metric("Expected Y Coordinate Center", f"{expected_y:.4f}")
        st.info("💡 **Notice:** Slide one activation slider to high values (e.g. `10.0`). The probability heatmap will concentrate completely around that point, pulling the expected center directly to that coordinate center!")

elif selected_eq == "Self-Attention":
    eq = EQUATIONS["Self-Attention"]
    st.markdown(f"### 🤝 {selected_eq}")
    st.latex(eq["latex"])
    st.write(eq["explanation"])
    
    st.subheader("🎮 Attention Sandbox")
    st.write("Configure Query ($Q$) and Key ($K$) vectors to observe how temperature ($\sqrt{d_k}$ or $\\tau$) impacts the attention weight distribution.")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.markdown("**Query Vector (1 x 3):**")
        q1 = st.slider("Q1", -5.0, 5.0, 1.0, 0.5)
        q2 = st.slider("Q2", -5.0, 5.0, 0.0, 0.5)
        q3 = st.slider("Q3", -5.0, 5.0, -1.0, 0.5)
        q_vec = np.array([q1, q2, q3])
        
        temperature = st.slider("Scaling Factor Temperature (Tau)", 0.1, 10.0, 1.0, 0.1)
        
    with col_v2:
        st.markdown("**Key Vectors (3 x 3):**")
        st.write("Three keys representing three different past items to compare against.")
        
        k1_1 = st.slider("Key 1 (Feature 1)", -5.0, 5.0, 1.0, 0.5)
        k2_1 = st.slider("Key 2 (Feature 1)", -5.0, 5.0, 0.0, 0.5)
        k3_1 = st.slider("Key 3 (Feature 1)", -5.0, 5.0, 2.0, 0.5)
        
        keys = np.array([
            [k1_1, 0.0, 0.0],
            [0.0, k2_1, 0.0],
            [0.0, 0.0, k3_1]
        ])
        
    raw_scores = q_vec @ keys.T
    scaled_scores = raw_scores / temperature
    
    attn_weights = np.exp(scaled_scores) / np.sum(np.exp(scaled_scores))
    
    st.write("---")
    st.markdown("### 📊 Attention Allocation Result")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        fig_bar = px.bar(
            x=["Key 1", "Key 2", "Key 3"],
            y=attn_weights,
            labels={'x': 'Keys', 'y': 'Attention Weight'},
            title="Resulting Attention Weights",
            color=attn_weights,
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(yaxis_range=[0, 1.0])
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_res2:
        st.metric("Attention to Key 1", f"{attn_weights[0]*100:.2f}%")
        st.metric("Attention to Key 2", f"{attn_weights[1]*100:.2f}%")
        st.metric("Attention to Key 3", f"{attn_weights[2]*100:.2f}%")
        st.info("💡 **Notice:** Lowering the Temperature scaling factor acts as a sharpening filter, forcing attention onto the single key with the highest similarity score. Increasing the temperature distributes attention evenly.")

else:
    st.markdown(f"### ⛓️ {selected_eq}")
    st.latex(EQUATIONS[selected_eq]["latex"])
    st.write(EQUATIONS[selected_eq]["explanation"])
    st.info("These equations are visualized dynamically inside stages **07 Forward Model Explorer** and **11 Temporal Pooling** respectively using actual runtime tensors!")
