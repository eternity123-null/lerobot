#!/usr/bin/env python

"""CARP Policy for Stage 2: Coarse-to-Fine Autoregressive Prediction training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Tuple

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.carp.configuration_carp import CARPConfig
from lerobot.policies.carp.MSAT.vqvae import MultiScaleActionTokenizer
from lerobot.policies.carp.CFAP.autoreg import Coarse2FineAutoRegressor
from lerobot.policies.carp.obs_encoder_builder import build_carp_obs_encoder


class CARPPolicy(PreTrainedPolicy):
    """
    CARP Stage 2: Coarse-to-Fine Autoregressive Policy.

    This policy trains the Transformer to predict multi-scale action tokens
    from observations and task IDs, then decodes them using a frozen VQ-VAE.

    Training: Cross-entropy loss on predicted tokens (teacher forcing)
    Inference: Autoregressive token generation + VAE decoding
    """

    config_class = CARPConfig
    name = "carp"

    def __init__(self, config: CARPConfig):
        super().__init__(config)
        config.validate_features()

        assert config.training_stage == "ar", "CARPPolicy requires training_stage='ar'"
        assert config.vae_checkpoint_path is not None, \
            "vae_checkpoint_path is required for AR training"

        self.config = config

        # Extract action dimension
        action_feature = config.output_features.get("action")
        assert action_feature is not None, "action feature is required"
        self.action_dim = action_feature.shape[0]

        # Build observation encoder
        self.obs_encoder = build_carp_obs_encoder(config)
        self.obs_dim = self.obs_encoder.output_shape()[0]

        # Load pretrained VQ-VAE and freeze
        self.vae = MultiScaleActionTokenizer(
            vocab_size=config.vocab_size,
            z_channels=config.vocab_ch,
            ch=config.vch,
            action_dim=self.action_dim,
            num_actions=config.action_horizon,
            dropout=config.vae_dropout,
            beta=config.vqbeta,
            using_znorm=config.vqnorm,
            quant_conv_ks=3,
            quant_resi=config.vqresi,
            share_quant_resi=4,
            v_patch_nums=config.patch_nums,
            test_mode=True,  # Inference mode
        )
        self._load_vae_checkpoint(config.vae_checkpoint_path)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # Build Coarse-to-Fine Autoregressive Transformer
        self.ar_model = Coarse2FineAutoRegressor(
            vae_proxy=self.vae,
            obs_encoder=self.obs_encoder,
            action_dim=self.action_dim,
            task_num=config.task_num,
            task_embed_dim=config.task_embed_dim,
            depth=config.ar_depth,
            embed_dim=config.ar_embed_dim,
            num_heads=config.ar_num_heads,
            mlp_ratio=config.ar_mlp_ratio,
            drop_rate=config.ar_dropout,
            attn_drop_rate=config.ar_attn_dropout,
            drop_path_rate=config.ar_drop_path_rate,
            norm_eps=1e-6,
            shared_aln=False,  # Use separate AdaLN for each layer
            attn_l2_norm=True,  # Use L2 normalization in attention
            patch_nums=config.patch_nums,
            n_obs_steps=config.n_obs_steps,
        )

        # Initialize AR model weights
        self.ar_model.init_weights(
            init_adaln=0.5,
            init_adaln_gamma=1e-5,
            init_head=0.02,
            init_std=-1,  # Auto-scale
        )

        self.vae.to(config.device)
        self.obs_encoder.to(config.device)
        self.ar_model.to(config.device)

        self.reset()

    def _load_vae_checkpoint(self, checkpoint_path: str):
        """Load pretrained VAE weights."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove "vae." prefix if present
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("vae."):
                cleaned_state_dict[key[4:]] = value
            else:
                cleaned_state_dict[key] = value

        self.vae.load_state_dict(cleaned_state_dict, strict=True)

    def get_optim_params(self):
        """Return AR model + obs encoder parameters (VAE is frozen)."""
        return list(self.obs_encoder.parameters()) + list(self.ar_model.parameters())

    def reset(self):
        """Reset action queue for temporal action chunking."""
        self._action_queue = deque(maxlen=self.config.action_horizon)

    def forward(self, batch: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass for AR training.

        Args:
            batch: {
                "observation.images.*": (B, n_obs_steps, C, H, W),
                "observation.state": (B, n_obs_steps, D),
                "action": (B, action_horizon, action_dim),
                "task_id": (B,) - optional, defaults to 0
            }

        Returns:
            loss: Cross-entropy loss on predicted tokens
            loss_dict: {"loss": float, "accuracy": float, ...}
        """
        self.train()
        self.obs_encoder.train()
        self.ar_model.train()
        self.vae.eval()  # Keep VAE frozen

        # Extract observations
        obs_dict = {k: v for k, v in batch.items() if k.startswith("observation.")}

        # Extract task IDs (default to 0 if not provided)
        task_ids = batch.get("task_id", torch.zeros(len(batch["action"]), dtype=torch.long, device=self.config.device))

        # Extract and encode actions to tokens (ground truth)
        actions = batch["action"]  # (B, T, A)
        B = actions.shape[0]

        with torch.no_grad():
            # Encode actions to multi-scale tokens
            actions_per_dim = actions.transpose(1, 2).unsqueeze(-1)  # (B, A, T, 1)
            latent = self.vae.encoder(actions_per_dim)
            latent = self.vae.quant_conv(latent)
            gt_token_indices = self.vae.quantize.f_to_idxBl_or_fhat(
                latent,
                to_fhat=False,
                v_patch_nums=self.config.patch_nums
            )  # List of [B, pn*1] for each scale

        # Forward through AR model with teacher forcing
        logits_list = self.ar_model(
            obs_dict=obs_dict,
            task_ids=task_ids,
            gt_indices=gt_token_indices if self.training else None,
        )  # List of [B, pn*1, vocab_size] for each scale

        # Compute cross-entropy loss for each scale
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        scale_losses = []

        for scale_idx, (logits, gt_indices) in enumerate(zip(logits_list, gt_token_indices)):
            # logits: (B, num_patches, vocab_size)
            # gt_indices: (B, num_patches)

            logits_flat = logits.reshape(-1, self.config.vocab_size)  # (B*num_patches, V)
            gt_flat = gt_indices.reshape(-1)  # (B*num_patches,)

            # Cross-entropy loss with optional label smoothing
            scale_loss = F.cross_entropy(
                logits_flat,
                gt_flat,
                label_smoothing=self.config.ar_label_smoothing,
            )

            total_loss += scale_loss
            scale_losses.append(scale_loss.item())

            # Compute accuracy
            pred_indices = logits_flat.argmax(dim=-1)
            correct = (pred_indices == gt_flat).sum().item()
            total_correct += correct
            total_tokens += gt_flat.numel()

        # Average loss across scales
        total_loss = total_loss / len(logits_list)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        loss_dict = {
            "loss": total_loss.item(),
            "accuracy": accuracy,
        }

        # Add per-scale losses
        for i, scale_loss in enumerate(scale_losses):
            loss_dict[f"loss_scale{i+1}"] = scale_loss

        return total_loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Predict a full action chunk using autoregressive generation.

        Args:
            batch: {
                "observation.images.*": (B, n_obs_steps, C, H, W),
                "observation.state": (B, n_obs_steps, D),
                "task_id": (B,) - optional
            }

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        self.eval()
        self.obs_encoder.eval()
        self.ar_model.eval()
        self.vae.eval()

        # Extract observations
        obs_dict = {k: v for k, v in batch.items() if k.startswith("observation.")}

        # Extract task IDs
        task_ids = batch.get("task_id", torch.zeros(1, dtype=torch.long, device=self.config.device))

        # Autoregressive generation (no teacher forcing)
        logits_list = self.ar_model(
            obs_dict=obs_dict,
            task_ids=task_ids,
            gt_indices=None,  # Autoregressive mode
        )

        # Convert logits to token indices (greedy sampling)
        pred_token_indices = [logits.argmax(dim=-1) for logits in logits_list]
        # Each: (B, num_patches)

        # Decode tokens to actions using frozen VAE decoder
        actions = self._decode_tokens_to_actions(pred_token_indices)

        return actions

    def _decode_tokens_to_actions(self, token_indices: list[torch.Tensor]) -> torch.Tensor:
        """
        Decode multi-scale token indices to actions using VAE decoder.

        Args:
            token_indices: List of (B, num_patches) tensors for each scale

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        B = token_indices[0].shape[0]

        # Reconstruct latent from tokens (multi-scale)
        # This uses the VQ-VAE's quantizer to map indices back to continuous latent
        C = self.config.vocab_ch
        H = self.config.action_horizon
        W = 1

        # Initialize with zeros
        latent_reconstructed = torch.zeros(B, C, H, W, device=self.config.device)

        # Accumulate contributions from each scale
        for scale_idx, (pn, indices) in enumerate(zip(self.config.patch_nums, token_indices)):
            # indices: (B, pn*1)
            indices_hw = indices.view(B, pn, 1)  # (B, pn, 1)

            # Embed indices to get quantized features
            h_BChw = self.vae.quantize.embedding(indices_hw).permute(0, 3, 1, 2)  # (B, C, pn, 1)

            # Upsample to target size
            if pn != H:
                h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')

            # Apply residual function
            SN = len(self.config.patch_nums)
            h_BChw = self.vae.quantize.quant_resi[scale_idx / (SN - 1)](h_BChw)

            # Accumulate
            latent_reconstructed += h_BChw

        # Decode latent to actions
        # latent_reconstructed: (B, C, H, 1) where H=action_horizon
        # Need to reshape to (B, action_dim, action_horizon, 1) for decoder

        # Post-quantization conv
        decoded_latent = self.vae.post_quant_conv(latent_reconstructed)

        # Decoder expects (B, action_dim, T, 1)
        actions_per_dim = self.vae.decoder(decoded_latent)  # (B, action_dim, T, 1)

        # Reshape to (B, T, action_dim)
        actions = actions_per_dim.squeeze(-1).transpose(1, 2)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Select a single action for real-time control.

        Implements temporal action chunking: predict a full chunk, then
        execute actions one-by-one from the queue.

        Args:
            batch: Observation batch (same as predict_action_chunk)

        Returns:
            action: (action_dim,) - single action to execute
        """
        self.eval()

        # If action queue is empty, generate a new chunk
        if len(self._action_queue) == 0:
            action_chunk = self.predict_action_chunk(batch)  # (1, horizon, action_dim)
            action_chunk = action_chunk.squeeze(0)  # (horizon, action_dim)

            # Add all actions to queue
            for action in action_chunk:
                self._action_queue.append(action)

        # Pop and return the next action
        return self._action_queue.popleft()
