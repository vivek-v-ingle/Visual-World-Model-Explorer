PAPER_OVERVIEW_CONTENT = """
## OSVI-WM (One-Shot Visual Imitation World Model)

One-Shot Visual Imitation (OSVI) is a paradigm where an agent learns to perform a task after observing only **a single expert demonstration** of that task. Instead of requiring hundreds of demonstrations, the agent must generalize from a single video.

OSVI-WM achieves this by training a **Predictive Latent World Model**. 
Rather than directly mapping pixels to actions, the model learns:
1. **To encode images** into a compact latent space ($z$).
2. **To predict** how the latent state changes over time using a transition model.
3. **To plan** a sequence of 3D spatial waypoints that imitate the expert demonstration.

### Why use a World Model?
A World Model acts as a simulator of the environment's dynamics. By training the model to predict the future latent states autoregressively, the agent can **imagine the consequences of its choices** without executing them on a physical robot. This provides a robust foundation for trajectory planning and enables transferring the model to new systems (FastWAM, Demo-JEPA, etc.) by swapping the encoder or planning head.
"""

TENSOR_DESCRIPTIONS = {
    "images": {
        "shape": "[B, T_pair, C, H, W] - e.g., [1, 1, 3, 240, 320]",
        "meaning": "The agent's current workspace observation frame (the start frame of its attempt).",
        "why": "This acts as the query state $R_1$ from which the model rolls out predictions."
    },
    "context": {
        "shape": "[B, T_context, C, H, W] - e.g., [1, 10, 3, 240, 320]",
        "meaning": "The context frames extracted from the expert demonstration video.",
        "why": "These frames ($E_1 ... E_{10}$) define the task reference. The model compares the current state to these to understand the goal."
    },
    "resnet_features_raw": {
        "shape": "[B, T_context + T_pair, 512, H_feat, W_feat] - e.g., [1, 11, 512, 8, 10]",
        "meaning": "Raw feature maps extracted by the Shared ResNet18 encoder before coordinate pooling.",
        "why": "It represents local spatial and semantic features of both context and agent frames."
    },
    "resnet_features": {
        "shape": "[B, T_context + T_pair, 512, 8, 10]",
        "meaning": "Encoder latent state $z$.",
        "why": "It represents the downsampled embedding of the frames in the latent space."
    },
    "action_attn_layer_0": {
        "shape": "[B, heads, T * H * W, T * H * W] - e.g., [1, 4, 880, 880]",
        "meaning": "Causal self-attention weight matrix for the Action Model's first attention layer.",
        "why": "It maps how much the model attends from each spatiotemporal token (at time step $t$, channel $c$) to prior tokens."
    },
    "action_attn_layer_1": {
        "shape": "[B, heads, T * H * W, T * H * W]",
        "meaning": "Causal self-attention weight matrix for the Action Model's second attention layer.",
        "why": "Refines self-attention over the sequence history to model actions."
    },
    "predicted_latent_states": {
        "shape": "[B, T_rollout, 512, 8, 10] - e.g., [1, 5, 512, 8, 10]",
        "meaning": "Autoregressively predicted future latent states $\\hat{z}_1 ... \\hat{z}_T$ representing the future trajectory.",
        "why": "These are the world model's internal imagination of how the workspace features evolve."
    },
    "spatial_softmax_mask": {
        "shape": "[B, T_rollout, 512, 8, 10] - e.g., [1, 5, 512, 8, 10]",
        "meaning": "The normalized 2D probability distributions output by the Spatial Softmax layer.",
        "why": "Softmax converts raw feature maps into coordinate probability grids, isolating key points of interest."
    },
    "spatial_coords": {
        "shape": "[B, T_rollout, 1024] - e.g., [1, 5, 1024]",
        "meaning": "Expected coordinates vector: 512 channels $\\times$ $(h, w)$ pairs.",
        "why": "Reduces the spatiotemporal latent states to raw coordinate coordinates for planning."
    },
    "waypoint_states_pe": {
        "shape": "[B, T_rollout, 1024]",
        "meaning": "Expected coordinates with positional encodings added.",
        "why": "Provides temporal sequence information to the pooling layer."
    },
    "pooler_attention": {
        "shape": "[B, heads, num_queries, T_rollout] - e.g., [1, 1, 1, 5]",
        "meaning": "Cross-attention weights between the query token and the temporal rollout states.",
        "why": "Models how much each predicted future frame contributes to the final pooled trajectory representation."
    },
    "pooled_states": {
        "shape": "[B, 1, 1024]",
        "meaning": "The temporally pooled coordinate feature representation.",
        "why": "Aggregates the entire imagined trajectory sequence into a single context embedding."
    },
    "act_embed": {
        "shape": "[B, 1280] (latent_dim * 5)",
        "meaning": "Fully connected feature projection ready for decoding.",
        "why": "Matches the input dimensions of the waypoint decoder head."
    },
    "raw_waypoints": {
        "shape": "[B, waypoints, 4] - e.g., [1, 15, 4]",
        "meaning": "Raw predicted waypoint sequence in normalized image coordinates (u, v, depth, grasp).",
        "why": "This is the final output of the neural network pipeline before physical projection."
    }
}

EQUATIONS = {
    "Spatial Softmax": {
        "latex": r"s(x, y, c) = \frac{e^{f(x, y, c)}}{\sum_{x', y'} e^{f(x', y', c)}}",
        "explanation": "Applies a 2D Softmax function over each channel $c$ of the latent feature map $f$. This normalizes the feature activations into a spatial probability distribution.",
        "expected_coords": r"h_c = \sum_{x, y} y \cdot s(x, y, c), \quad w_c = \sum_{x, y} x \cdot s(x, y, c)",
        "expected_coords_explanation": "Computes the expected coordinate center $(h_c, w_c)$ by taking the weighted sum of pixel indices scaled by the probability map $s(x,y,c)$."
    },
    "Self-Attention": {
        "latex": r"\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V",
        "explanation": "Calculates query-key similarity scaled by key dimension $d_k$, normalized by Softmax to form attention weights, and aggregates the values $V$. Used in the Action Model and Forward Model."
    },
    "Autoregressive Rollout": {
        "latex": r"\hat{z}_{t+1} = \text{Forward}(\text{concat}(a_t, s_t))",
        "explanation": "The World Model's forward model is causal. It takes the action features $a_t$ and the current state representation $s_t$, and autoregressively predicts the subsequent state $\\hat{z}_{t+1}$."
    },
    "Temporal Attentive Pooling": {
        "latex": r"\text{Pool}(z_{1:T}) = \text{CrossAttention}(Q_{query}, K_{z}, V_{z})",
        "explanation": "A learned query token $Q_{query}$ pools temporal representations $z_{1:T}$ via cross-attention, allowing the model to focus dynamically on specific frames of the rollout."
    }
}

PAPER_SECTIONS = [
    {
        "id": "intro",
        "title": "1. One-Shot Visual Imitation",
        "summary": "Imitating actions from a single demonstration requires learning a dynamic representation of tasks. OSVI-WM achieves this through a visual world model, avoiding direct pixel reconstruction.",
        "code_link": "models/model.py#L88-L130",
        "visual_page": "01_Paper_Overview",
        "tensor": "context"
    },
    {
        "id": "encoder",
        "title": "2. Shared ResNet Encoder",
        "summary": "Converts RGB video frames into high-dimensional feature maps using a ResNet-18 convnet backend. The encoder is shared between teacher demonstration and agent frames.",
        "code_link": "models/basic_embeddings.py#L79-L109",
        "visual_page": "04_Encoder_Explorer",
        "tensor": "resnet_features_raw"
    },
    {
        "id": "action",
        "title": "3. Action Model (Non-Local Temporal Attention)",
        "summary": "Models spatiotemporal dependencies. Employs a sequence of causal Non-Local layers that calculate self-attention across time and space tokens.",
        "code_link": "models/model.py#L61-L87",
        "visual_page": "06_Action_Model_Explorer",
        "tensor": "action_attn_layer_0"
    },
    {
        "id": "forward",
        "title": "4. Forward World Model (Autoregressive Rollout)",
        "summary": "Predicts future state representations. Uses a causal GPT-style Transformer to imagine future latent frames step-by-step.",
        "code_link": "models/system_model.py#L23-L97",
        "visual_page": "07_Forward_Model_Explorer",
        "tensor": "predicted_latent_states"
    },
    {
        "id": "spatial",
        "title": "5. Spatial Softmax (Feature Coordinates)",
        "summary": "Extracts exact expected 2D coordinates from predicted feature maps. Bypasses decoder reconstruction by mapping activations directly to coordinate spaces.",
        "code_link": "models/model.py#L169-L177",
        "visual_page": "09_Spatial_Embedding",
        "tensor": "spatial_softmax_mask"
    },
    {
        "id": "pooling",
        "title": "6. Temporal Attentive Pooling & Waypoints",
        "summary": "Gathers information from all imagined frames. A learned query token cross-attends over all temporal steps, and maps the output to a multi-waypoint 3D sequence.",
        "code_link": "models/attentive_pooler.py#L21-L103",
        "visual_page": "11_Temporal_Pooling",
        "tensor": "pooler_attention"
    }
]
