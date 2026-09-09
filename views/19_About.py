import streamlit as st


st.markdown("## 🔮 About Visual World Model Explorer")
st.write("Visual World Model Explorer is an open-source interactive textbook and debugger for robotic latent world models.")

st.markdown("""
### 🌟 Project Mission
To demystify neural network representations for robotic decision-making. By visualising intermediate features, self-attention, causal temporal rollouts, and coordinate transformations, we help researchers and students build a clear mental model of how world models learn.

### 🔌 How to add a new Model Backend
The application features a modular plugin architecture. To add a new world model:
1. Create a backend class inside `backends/` that inherits from `core.types.BaseModelBackend`.
2. Implement the required abstract methods: `load_model`, `load_trajectory`, and `run_inference` (which captures your model's spatiotemporal tensors).
3. Register the backend in `core/manager.py` using `manager.register_backend("YourModel", YourModelBackend())`.
4. The Streamlit pages will dynamically populate all debugger sections with your backend's tensors, heatmaps, and equations automatically!

### 📚 Citations & References
If you use this educational platform or base research on it, please cite the original papers:
```bibtex
@inproceedings{osviwm2024,
  title={One-Shot Visual Imitation via Latent World Models},
  author={OSVI-WM Contributors},
  booktitle={Robotics and Automation Conference},
  year={2024}
}
```
""")
