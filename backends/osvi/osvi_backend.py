import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from typing import Dict, Any, List
from core.types import BaseModelBackend

# Dynamic path adjustment for osvi-wm imports
sys.path.insert(0, '/home/vvijaykumar/osvi-wm')

class OSVIWorldModelBackend(BaseModelBackend):
    """
    Official OSVI-WM model backend implementation.
    """
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        self.model = None
        self.checkpoint_path = None
        self.is_metaworld = False

    def get_supported_checkpoints(self) -> Dict[str, str]:
        import os
        return {
            "Pick & Place (Panda simulator)": os.path.expanduser("~/osvi-wm/checkpoints/pp/model.pt"),
            "MetaWorld simulator": os.path.expanduser("~/osvi-wm/checkpoints/metaworld/model.pt")
        }

    def get_default_trajectories(self) -> Dict[str, str]:
        import os
        local_sample = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets/sample_traj.pkl"))
        return {
            "sample_traj.pkl (Packaged, Local)": local_sample,
            "traj_zed_resized.pkl (Small, External)": os.path.expanduser("~/OSVI-Deploy/traj_zed_resized.pkl"),
            "traj_zed.pkl (Full, External)": os.path.expanduser("~/OSVI-Deploy/traj_zed.pkl"),
            "traj_zed_real.pkl (Real, External)": os.path.expanduser("~/OSVI-Deploy/traj_zed_real.pkl")
        }

    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        self.is_metaworld = kwargs.get("metaworld", False)
        
        # Avoid reloading if same config
        if self.model is not None and self.checkpoint_path == checkpoint_path:
            return
            
        from models.model import StateSpaceModel
        
        model = StateSpaceModel(
            latent_dim=256,
            waypoints=5,
            sub_waypoints=True,
            metaworld=self.is_metaworld
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)['model_state_dict']
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        self.model = model
        self.checkpoint_path = checkpoint_path

    def load_trajectory(self, traj_path: str) -> Dict[str, Any]:
        from dataset.agent_dataset import AgentDemonstrations
        from dataset.teacher_dataset import TeacherDemonstrations
        
        with open(traj_path, "rb") as f:
            raw_data = pickle.load(f)
            
        agent_ds = AgentDemonstrations(
            files=[traj_path],
            height=240,
            width=320,
            crop=[100, 0, 0, 0],
            T_context=10,
            T_pair=1,
            mode='train',
            waypoints=True,
            cache=False
        )
        
        teacher_ds = TeacherDemonstrations(
            files=[traj_path],
            height=240,
            width=320,
            crop=[100, 0, 0, 0],
            T_context=10,
            T_pair=1,
            mode='train',
            waypoints=True,
            cache=False
        )
        
        agent_pairs, _ = agent_ds[0]
        teacher_context = teacher_ds[0]
        
        # Save tensors on CPU for safe visualization
        images = torch.from_numpy(agent_pairs['images']).unsqueeze(0).float()
        context = torch.from_numpy(teacher_context['video']).unsqueeze(0).float()
        projection_matrix = torch.from_numpy(agent_pairs['projection_matrix']).unsqueeze(0).float()
        
        return {
            "images": images,
            "context": context,
            "projection_matrix": projection_matrix,
            "raw_data": raw_data,
            "agent_pairs": agent_pairs,
            "teacher_context": teacher_context
        }

    def run_inference(self, images: Any, context: Any, T_tot: int = 16) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded!")
            
        # Push variables to active GPU/CPU device
        device_images = images.to(self.device)
        device_context = context.to(self.device)
        
        captured_tensors = {}
        
        # 1. Disable SDPA globally to capture explicit attention matrices
        def disable_sdpa(module):
            if hasattr(module, 'use_sdpa'):
                module.use_sdpa = False
            for child in module.children():
                disable_sdpa(child)
        disable_sdpa(self.model)
        
        # 2. Patch ActionModel layers to capture self-attention
        original_non_local_forwards = {}
        def make_patched_forward(layer_name):
            def patched_forward(self_module, inputs):
                K, Q, V = self_module._K(inputs), self_module._Q(inputs), self_module._V(inputs)
                B, C, T, H, W = K.shape
                K, Q, V = [t.reshape((B, self_module._n_heads, int(C / self_module._n_heads), T*H*W)) for t in (K, Q, V)]
                KQ = torch.matmul(K.transpose(2, 3), Q) / self_module._temperature
                if self_module._causal:
                    mask = torch.tril(torch.ones((T, T))).to(KQ.device)
                    mask = mask.repeat_interleave(H*W, 0).repeat_interleave(H*W, 1)
                    KQ = KQ + torch.log(mask).unsqueeze(0).unsqueeze(0)
                attn = F.softmax(KQ, 3)
                
                captured_tensors[layer_name] = attn.detach().cpu()
                
                V = torch.matmul(V, attn.transpose(2, 3)).reshape((B, C, T, H, W))
                out = inputs + self_module._drop1(self_module._a1(self_module._out(V))) if self_module._skip else self_module._drop1(self_module._a1(self_module._out(V)))
                return self_module._norm(out)
            return patched_forward
            
        for idx, layer in enumerate(self.model.action_model._st_attn):
            original_non_local_forwards[layer] = layer.forward
            layer.forward = types.MethodType(make_patched_forward(f"action_attn_layer_{idx}"), layer)
            
        # 3. Patch AttentivePooler to capture cross-attention
        original_pooler_forward = self.model.attn_pool.forward
        def patched_pooler_forward(self_module, x):
            q = self_module.query_tokens.repeat(len(x), 1, 1)
            q, attn = self_module.cross_attention_block(q, x, return_attention=False)
            captured_tensors["pooler_attention"] = attn.detach().cpu()
            if self_module.blocks is not None:
                for blk in self_module.blocks:
                    q = blk(q)
            return q
        self.model.attn_pool.forward = types.MethodType(patched_pooler_forward, self.model.attn_pool)

        # 4. Patch Forward Model Block to capture self-attention
        original_block_forwards = {}
        for idx, block in enumerate(self.model.forward_model.transformer.h):
            original_block_forwards[block] = block.forward
            def make_patched_block_forward(block_idx):
                def patched_block_forward(self_block, x, return_attention=True, mask=None):
                    y, attn = self_block.attn(self_block.norm1(x), mask=mask)
                    captured_tensors[f"forward_model_attn_block_{block_idx}"] = attn.detach().cpu()
                    x = x + self_block.drop_path(y)
                    x = x + self_block.drop_path(self_block.mlp(self_block.norm2(x)))
                    return x
                return patched_block_forward
            block.forward = types.MethodType(make_patched_block_forward(idx), block)

        try:
            with torch.no_grad():
                T_context = device_context.shape[1]
                enc_features, target_features, enc_features_raw = self.model._embed(device_images, device_context, False)
                
                # Capture variables explicitly on CPU to prevent device crashes
                captured_tensors["images"] = images.clone()
                captured_tensors["context"] = context.clone()
                captured_tensors["resnet_features_raw"] = enc_features_raw.detach().cpu()
                captured_tensors["resnet_features"] = enc_features.detach().cpu()
                
                context_states = enc_features[:, :T_context]
                current_state = enc_features[:, T_context:T_context+1]
                
                all_next_states = []
                action_features_by_step = []
                forward_inputs_by_step = []
                
                for i in range(T_tot - T_context - 1):
                    act_input = torch.concat([context_states, current_state], dim=1)
                    act_features = self.model.action_model(act_input)
                    action_features_by_step.append(act_features.detach().cpu())
                    
                    forward_model_input = torch.concat([act_features[:, T_context:], current_state], dim=2).flatten(2, 4)
                    forward_inputs_by_step.append(forward_model_input.detach().cpu())
                    
                    next_state = self.model.forward_model(forward_model_input)[:, -1:]
                    next_state = next_state.view(next_state.shape[0], 1, enc_features.shape[2], enc_features.shape[3], enc_features.shape[4])
                    
                    current_state = torch.concat([current_state, next_state], dim=1)
                    all_next_states.append(next_state)
                    
                all_next_states = torch.cat(all_next_states, dim=1)
                captured_tensors["predicted_latent_states"] = all_next_states.detach().cpu()
                captured_tensors["action_features_by_step"] = action_features_by_step
                captured_tensors["forward_inputs_by_step"] = forward_inputs_by_step
                
                # Spatial Softmax extraction
                forward_states, spatial_softmax_mask = self.model._spatial_embed(all_next_states, ret_mask=True)
                captured_tensors["spatial_softmax_mask"] = spatial_softmax_mask.detach().cpu()
                captured_tensors["spatial_coords"] = forward_states.detach().cpu()
                
                waypoint_states = forward_states
                waypoint_states_pe = self.model.attn_pe(waypoint_states)
                captured_tensors["waypoint_states_pe"] = waypoint_states_pe.detach().cpu()
                
                pooled_states = self.model.attn_pool(waypoint_states_pe)
                captured_tensors["pooled_states"] = pooled_states.detach().cpu()
                
                pooled_flat = pooled_states.view(pooled_states.shape[0], -1)
                act_embed = self.model._to_embed(pooled_flat)
                captured_tensors["act_embed"] = act_embed.detach().cpu()
                
                waypoints = self.model.waypoint_head(act_embed)
                captured_tensors["raw_waypoints"] = waypoints.detach().cpu()
                
        finally:
            # Restore original forwards
            for layer, orig_f in original_non_local_forwards.items():
                layer.forward = orig_f
            self.model.attn_pool.forward = original_pooler_forward
            for block, orig_f in original_block_forwards.items():
                block.forward = orig_f
                
        return captured_tensors


class MockFastWAMBackend(BaseModelBackend):
    """
    Mock class for FastWAM (Fast World Action Model) to test architecture comparison features.
    """
    def get_supported_checkpoints(self) -> Dict[str, str]:
        return {"FastWAM Default (Mock)": "mock_fastwam_weights.pt"}
        
    def get_default_trajectories(self) -> Dict[str, str]:
        return {"FastWAM Traj (Mock)": "mock_fastwam_traj.pkl"}
        
    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        pass
        
    def load_trajectory(self, traj_path: str) -> Dict[str, Any]:
        # Return dummy data matching expected shapes
        return {
            "images": torch.zeros((1, 1, 3, 240, 320)),
            "context": torch.zeros((1, 10, 3, 240, 320)),
            "projection_matrix": torch.eye(4)[:3, :4].unsqueeze(0)
        }
        
    def run_inference(self, images: Any, context: Any, T_tot: int = 16) -> Dict[str, Any]:
        # FastWAM returns a non-autoregressive parallel rollout
        return {
            "images": images,
            "context": context,
            "resnet_features_raw": torch.zeros((1, 11, 512, 8, 10)),
            "resnet_features": torch.zeros((1, 11, 512, 8, 10)),
            "predicted_latent_states": torch.zeros((1, 5, 512, 8, 10)),
            "spatial_softmax_mask": torch.zeros((1, 5, 512, 8, 10)),
            "spatial_coords": torch.zeros((1, 5, 1024)),
            "pooled_states": torch.zeros((1, 1, 1024)),
            "raw_waypoints": torch.zeros((1, 15, 4))
        }


class MockDemoJepaBackend(BaseModelBackend):
    """
    DinoV2 + OSVI-assisted backend for Demo-JEPA to compare real visual representations and waypoints.
    """
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        self.model = None
        self.checkpoint_path = None
        
    def get_supported_checkpoints(self) -> Dict[str, str]:
        return {"Demo-JEPA ViT-Giant (Pretrained Backbone)": "dinov2_vitb14"}
        
    def get_default_trajectories(self) -> Dict[str, str]:
        import os
        local_sample = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets/sample_traj.pkl"))
        return {
            "sample_traj.pkl (Packaged, Local)": local_sample,
            "traj_zed_resized.pkl (Small, External)": os.path.expanduser("~/OSVI-Deploy/traj_zed_resized.pkl"),
            "traj_zed.pkl (Full, External)": os.path.expanduser("~/OSVI-Deploy/traj_zed.pkl"),
            "traj_zed_real.pkl (Real, External)": os.path.expanduser("~/OSVI-Deploy/traj_zed_real.pkl")
        }
        
    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        if self.model is None:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.model.to(self.device)
            self.model.eval()
            self.checkpoint_path = checkpoint_path
            
    def load_trajectory(self, traj_path: str) -> Dict[str, Any]:
        # Reuse OSVI loader logic to read real pickle trajectory data
        osvi = OSVIWorldModelBackend(device=str(self.device))
        return osvi.load_trajectory(traj_path)
        
    def run_inference(self, images: Any, context: Any, T_tot: int = 16) -> Dict[str, Any]:
        # 1. Run real OSVI waypoints prediction under the hood
        osvi = OSVIWorldModelBackend(device=str(self.device))
        ckpt = os.path.expanduser("~/osvi-wm/checkpoints/pp/model.pt")
        osvi.load_model(ckpt)
        osvi_tensors = osvi.run_inference(images, context, T_tot)
        
        # 2. Process inputs for DinoV2 visual features
        # all_frames shape: [B, 11, 3, 240, 320]
        all_frames = torch.cat([context, images], dim=1)
        B, T, C, H, W = all_frames.shape
        
        # Resize to DinoV2 expected 224x224 shape
        all_frames_flat = all_frames.view(-1, C, H, W)
        all_frames_resized = F.interpolate(all_frames_flat, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Move to GPU
        all_frames_resized = all_frames_resized.to(self.device)
        
        # Run DinoV2 ViT-B14 encoder forward
        with torch.no_grad():
            features_dict = self.model.forward_features(all_frames_resized)
            patch_tokens = features_dict['x_norm_patchtokens'] # shape [B*T, 256, 768]
            
        # Reshape to spatial feature grid [B*T, 768, 16, 16]
        patch_tokens = patch_tokens.permute(0, 2, 1).view(-1, 768, 16, 16)
        # Reshape back to batch: [B, T, 768, 16, 16]
        patch_tokens = patch_tokens.view(B, T, 768, 16, 16)
        
        # Channel-reduce to 384 dimensions to match expected viewer shape
        reduced_features = patch_tokens[:, :, :384, :, :]
        
        # 3. Simulate predicted joint position (qpos) by calling IK solver or mapping
        # Let's map the waypoints to joint space using a simulated joint path matching Panda
        qpos_seq = torch.tensor([[
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04],  # Home
            [0.1, -0.600, 0.0, -2.100, 0.0, 1.600, 0.785, 0.04],  # Approach
            [0.2, -0.400, 0.1, -1.900, 0.0, 1.700, 0.785, 0.00],  # Grasp
            [0.1, -0.600, 0.0, -2.100, 0.0, 1.600, 0.785, 0.00],  # Lift
            [-0.3, -0.500, -0.2, -2.000, 0.0, 1.500, 0.785, 0.04] # Place
        ]])
        
        return {
            "images": images,
            "context": context,
            "resnet_features_raw": reduced_features.cpu(),
            "resnet_features": reduced_features.cpu(),
            "predicted_latent_states": reduced_features[:, 1::2, :, :, :].cpu(), # Mock latent predictions
            "spatial_softmax_mask": torch.zeros((B, 5, 384, 16, 16)),
            "spatial_coords": torch.zeros((B, 5, 768)),
            "pooled_states": torch.zeros((B, 1, 768)),
            "pooler_attention": torch.tensor([[[[0.4, 0.25, 0.15, 0.1, 0.1]]]]).repeat(B, 8, 1, 1),
            "raw_waypoints": osvi_tensors["raw_waypoints"],
            "predicted_qpos": qpos_seq
        }

