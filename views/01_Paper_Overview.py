import streamlit as st
import os
import sys

# Ensure repository root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from core.explanations import PAPER_OVERVIEW_CONTENT


st.markdown("## 📖 Paper Overview & One-Shot Learning Concepts")
st.write("Understand the core foundations of predictive latent world models before diving into spatiotemporal debugging.")

st.markdown(PAPER_OVERVIEW_CONTENT)

st.markdown("### 🏗️ Global Data Flow & Representations")
st.markdown("""
```mermaid
graph TD
    subgraph Input Module [1. Input Explorer]
        E["Teacher Context (E1...E10)"]
        R["Agent Observation (R1)"]
    end

    subgraph Encoder [2. Encoder Explorer]
        RGB["RGB Frames [B, 11, 3, 240, 320]"]
        Feats["Feature Maps [B, 11, 512, 8, 10]"]
        RGB --> Feats
    end
    E --> RGB
    R --> RGB

    subgraph Latent Space [3. Latent Space Explorer]
        ZE["ZE (Context Latent)"]
        ZR["ZR (Agent Observation Latent)"]
    end
    Feats -->|Split| ZE
    Feats -->|Split| ZR

    subgraph Action [4. Action Model]
        PE1["Temporal Positional Encodings"]
        SelfAttn["Non-Local Causal Self-Attention"]
        ActEmbed["Action Embeddings [B, 11, 512, 8, 10]"]
        ZE --> PE1
        ZR --> PE1
        PE1 --> SelfAttn --> ActEmbed
    end

    subgraph World Model [5. Forward World Model]
        Rollout["Autoregressive Latent Transition Loop"]
        PredictedStates["Imagined Future Latents (Z1...Z5)"]
        ActEmbed --> Rollout
        ZR --> Rollout
        Rollout --> PredictedStates
    end

    subgraph Spatial Softmax [6. Spatial Embedding]
        Softmax["2D Softmax Map"]
        Coords["Expected Coords Vector [B, 5, 1024]"]
        PredictedStates --> Softmax --> Coords
    end

    subgraph Pooling [7. Temporal Attentive Pooling]
        CrossAttn["Attentive Pooler Cross-Attention"]
        Pooled["Pooled Embedding [B, 1, 1024]"]
        Coords --> CrossAttn --> Pooled
    end

    subgraph Decoder [8. Waypoint Decoder]
        MLP["Fully Connected Layer MLP"]
        Waypoints["Predicted Waypoints [B, 15, 4]"]
        Pooled --> MLP --> Waypoints
    end

    classDef default fill:#1e293b,stroke:#2e3748,color:#e2e8f0;
    classDef highlight fill:#312e81,stroke:#6366f1,color:#c7d2fe;
    class E,R highlight;
    class Waypoints highlight;
```
""", unsafe_allow_html=True)
