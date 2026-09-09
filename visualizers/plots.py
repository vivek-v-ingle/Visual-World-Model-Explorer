import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import torch

def preprocess_image_tensor(tensor_img):
    """
    Convert a PyTorch image tensor [3, H, W] in [-1, 1] or [0, 1] range
    to a standard numpy RGB image [H, W, 3] in [0, 255] range.
    Ensures transfer to CPU first.
    """
    if isinstance(tensor_img, torch.Tensor):
        img = tensor_img.detach().cpu().numpy()
    else:
        img = tensor_img
        
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
        
    # Un-normalize if in standard ImageNet format (approximate)
    if img.min() < 0:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def render_feature_heatmap(img, feat_map, channel_idx, alpha=0.6):
    """
    Interpolates a feature map slice [H_feat, W_feat] to match [H, W],
    converts it to a Jet colormap heatmap, and blends it with the original RGB image.
    """
    img_rgb = preprocess_image_tensor(img)
    H, W, _ = img_rgb.shape
    
    # Extract channel slice
    slice_map = feat_map[channel_idx]
    if isinstance(slice_map, torch.Tensor):
        slice_map = slice_map.detach().cpu().numpy()
        
    # Normalize slice to [0, 1]
    min_val, max_val = slice_map.min(), slice_map.max()
    if max_val - min_val > 1e-5:
        slice_norm = (slice_map - min_val) / (max_val - min_val)
    else:
        slice_norm = np.zeros_like(slice_map)
        
    # Resize to original resolution
    resized = cv2.resize(slice_norm, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Apply colormap
    heatmap = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Alpha blend
    blended = cv2.addWeighted(img_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return blended, resized

def plot_interactive_attention(attn_matrix, title="Attention Weights", x_labels=None, y_labels=None):
    """
    Plot attention weight matrices using Plotly.
    attn_matrix has shape [seq_len, seq_len] or similar.
    """
    if isinstance(attn_matrix, torch.Tensor):
        attn = attn_matrix.detach().cpu().numpy()
    else:
        attn = attn_matrix
        
    # If heads dimension is present, average over heads or select head 0
    if len(attn.shape) == 3:
        attn = attn[0] # select first head
    elif len(attn.shape) == 4:
        attn = attn[0, 0] # batch 0, head 0
        
    fig = go.Figure(data=go.Heatmap(
        z=attn,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        zmin=0.0,
        zmax=attn.max()
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Key / Source Tokens",
        yaxis_title="Query / Target Tokens",
        width=700,
        height=600,
        margin=dict(l=50, r=50, b=50, t=50)
    )
    return fig

def plot_spatial_softmax_distribution(softmax_prob_map, channel_idx, expected_h, expected_w):
    """
    Visualize the spatial softmax distribution map for a single channel,
    with expected coordinate center overlaid.
    """
    prob_slice = softmax_prob_map[channel_idx]
    if isinstance(prob_slice, torch.Tensor):
        prob_slice = prob_slice.detach().cpu().numpy()
        
    H_feat, W_feat = prob_slice.shape
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=prob_slice,
        colorscale='Hot',
        showscale=True,
        name='Activation Probability'
    ))
    
    pixel_w = ((expected_w + 1.0) / 2.0) * (W_feat - 1)
    pixel_h = ((expected_h + 1.0) / 2.0) * (H_feat - 1)
    
    fig.add_trace(go.Scatter(
        x=[pixel_w],
        y=[pixel_h],
        mode='markers',
        marker=dict(size=15, color='lime', symbol='circle-dot', line=dict(color='black', width=2)),
        name=f'Expected Center ({expected_w:.2f}, {expected_h:.2f})'
    ))
    
    fig.update_layout(
        title=f"Channel {channel_idx} Softmax Probability Distribution",
        xaxis=dict(title="X (width index)", range=[-0.5, W_feat-0.5]),
        yaxis=dict(title="Y (height index)", range=[-0.5, H_feat-0.5]),
        width=600,
        height=500
    )
    return fig

def plot_final_predictions_3d(pred_world_waypoints, gt_trajectory=None, camera_pos=None):
    """
    Generate interactive 3D trajectory plot.
    """
    pred_x = pred_world_waypoints[:, 0]
    pred_y = pred_world_waypoints[:, 1]
    pred_z = pred_world_waypoints[:, 2]
    
    fig = go.Figure()
    
    if gt_trajectory is not None:
        gt_x = gt_trajectory[:, 0]
        gt_y = gt_trajectory[:, 1]
        gt_z = gt_trajectory[:, 2]
        fig.add_trace(go.Scatter3d(
            x=gt_x, y=gt_y, z=gt_z,
            mode='lines',
            line=dict(color='gray', width=3, dash='dash'),
            name='Ground-Truth Trajectory'
        ))
        
    fig.add_trace(go.Scatter3d(
        x=pred_x, y=pred_y, z=pred_z,
        mode='lines',
        line=dict(color='yellow', width=5),
        name='Predicted Trajectory Line'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=pred_x[1:-1], y=pred_y[1:-1], z=pred_z[1:-1],
        mode='markers+text',
        marker=dict(size=6, color='blue', opacity=0.8),
        text=[f"WP {i+2}" for i in range(len(pred_x)-2)],
        textposition="top center",
        name='Intermediate Waypoints'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[pred_x[0]], y=[pred_y[0]], z=[pred_z[0]],
        mode='markers+text',
        marker=dict(size=10, color='green', symbol='circle'),
        text=["START (WP 1)"],
        textposition="top center",
        name='Start Position'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[pred_x[-1]], y=[pred_y[-1]], z=[pred_z[-1]],
        mode='markers+text',
        marker=dict(size=10, color='red', symbol='circle'),
        text=[f"END (WP {len(pred_x)})"],
        textposition="top center",
        name='End Goal Position'
    ))
    
    if camera_pos is not None:
        fig.add_trace(go.Scatter3d(
            x=[camera_pos[0]], y=[camera_pos[1]], z=[camera_pos[2]],
            mode='markers+text',
            marker=dict(size=12, color='purple', symbol='diamond'),
            text=["Camera origin"],
            textposition="top center",
            name='Camera Center'
        ))
        
    axis_len = 0.15
    fig.add_trace(go.Scatter3d(x=[0, axis_len], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=3), name='Robot X-Axis (Red)'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, axis_len], z=[0, 0], mode='lines', line=dict(color='green', width=3), name='Robot Y-Axis (Green)'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_len], mode='lines', line=dict(color='blue', width=3), name='Robot Z-Axis (Blue)'))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def render_2d_trajectory_overlay(base_img, waypoints_raw, crop_offset=0):
    """
    Project waypoints on the 2D image coordinates and draw them.
    Ensures correct raw pixel calculations mapping [-1, 1] to width/height.
    """
    img_rgb = preprocess_image_tensor(base_img).copy()
    H, W, _ = img_rgb.shape
    
    # Calculate pixel coords
    pts = []
    for i in range(len(waypoints_raw)):
        u_norm, v_norm = waypoints_raw[i, 0], waypoints_raw[i, 1]
        x_px = int(((u_norm + 1.0) / 2.0) * W)
        y_px = int(((v_norm + 1.0) / 2.0) * (H - crop_offset)) + crop_offset
        pts.append((x_px, y_px))
        
    for i in range(len(pts) - 1):
        cv2.line(img_rgb, pts[i], pts[i+1], (255, 255, 0), 2)
        
    for i, pt in enumerate(pts):
        if i == 0:
            color = (0, 255, 0)
            size = 8
        elif i == len(pts) - 1:
            color = (255, 0, 0)
            size = 8
        else:
            color = (0, 0, 255)
            size = 5
        cv2.circle(img_rgb, pt, size, color, -1)
        cv2.putText(img_rgb, str(i+1), (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
    return img_rgb
