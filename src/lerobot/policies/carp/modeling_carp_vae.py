#!/usr/bin/env python

"""CARP VAE Policy for Stage 1: Multi-Scale Action Tokenization training."""

import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.carp.configuration_carp import CARPConfig
from lerobot.policies.carp.MSAT.vqvae import MultiScaleActionTokenizer


class CARPVAEPolicy(PreTrainedPolicy):
    """
    CARP Stage 1: VQ-VAE Training Policy.

    This policy trains the Multi-Scale Action Tokenizer (MSAT) which learns to
    tokenize action sequences into multi-scale discrete codes.

    Training objective: Reconstruction loss + VQ commitment loss
    """

    config_class = CARPConfig
    name = "carp_vae"

    def __init__(self, config: CARPConfig, **kwargs):
        super().__init__(config)
        config.validate_features()

        assert config.training_stage == "vae", "CARPVAEPolicy requires training_stage='vae'"

        self.config = config

        # Extract action dimension from config
        action_feature = config.output_features.get("action")
        assert action_feature is not None, "action feature is required"
        self.action_dim = action_feature.shape[0]

        # Build VQ-VAE tokenizer
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
            share_quant_resi=4,  # Partially-shared residual
            v_patch_nums=config.patch_nums,
            test_mode=False,  # Training mode
        )

        self.vae.to(config.device)
        self.reset()

    def get_optim_params(self):
        """Return VAE parameters for optimization."""
        return self.vae.parameters()

    def reset(self):
        """Reset internal state (no state for VAE training)."""
        pass

    def forward(self, batch: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass for VAE training.

        Args:
            batch: {
                "action": (B, action_horizon, action_dim) - normalized actions
            }

        Returns:
            loss: Total loss (reconstruction + VQ commitment)
            loss_dict: {"loss": float, "recon_loss": float, "vq_loss": float, "usages": list}
        """
        self.train()

        # Extract actions: (B, T, A)
        actions = batch["action"]
        B, T, A = actions.shape

        assert T == self.config.action_horizon, \
            f"Expected action_horizon={self.config.action_horizon}, got {T}"

        # Reshape to VAE expected format: (B, 1, T, A)
        # CARP uses per-dimension VQ-VAE where each action dimension is processed independently
        actions_reshaped = actions.unsqueeze(1)  # (B, 1, T, A)

        # Forward through VQ-VAE
        # Returns: (recon, usages, vq_loss)
        recon_actions, usages, vq_loss = self.vae(actions_reshaped, ret_usages=True)

        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon_actions, actions_reshaped)

        # Total loss
        total_loss = recon_loss + vq_loss

        loss_dict = {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "vq_loss": vq_loss.item(),
        }

        # Add codebook usage statistics
        if usages is not None:
            for i, usage in enumerate(usages):
                if i < len(usages) - 1:
                    loss_dict[f"usage_scale{i+1}"] = usage
                else:
                    loss_dict["usage_total"] = usage

        return total_loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Not used during VAE training stage.
        Raises NotImplementedError.
        """
        raise NotImplementedError(
            "CARPVAEPolicy is for training only. Use CARPPolicy for inference."
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Not used during VAE training stage.
        Raises NotImplementedError.
        """
        raise NotImplementedError(
            "CARPVAEPolicy is for training only. Use CARPPolicy for inference."
        )

    @torch.no_grad()
    def encode_actions(self, actions: torch.Tensor) -> list[torch.Tensor]:
        """
        Encode actions to multi-scale token indices.

        Args:
            actions: (B, action_horizon, action_dim)

        Returns:
            token_indices: List of (B, num_patches) tensors for each scale
        """
        self.eval()

        # Reshape to VAE expected format: (B, 1, T, A)
        actions_reshaped = actions.unsqueeze(1)

        # Encode through VAE
        latent = self.vae.encoder(actions_reshaped)
        latent = self.vae.quant_conv(latent)

        # Quantize to get indices
        token_indices = self.vae.quantize.f_to_idxBl_or_fhat(
            latent,
            to_fhat=False,
            v_patch_nums=self.config.patch_nums
        )

        return token_indices

    @torch.no_grad()
    def decode_tokens(self, token_indices: list[torch.Tensor]) -> torch.Tensor:
        """
        Decode multi-scale token indices to actions.

        Args:
            token_indices: List of (B, num_patches) tensors for each scale

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        self.eval()

        # Reconstruct latent from tokens
        # This is a simplified version - actual implementation would need to
        # match the VQ-VAE's multi-scale decoding process

        # For now, just forward through the full VAE to get reconstruction
        # In practice, this should be replaced with direct token-to-action decoding
        raise NotImplementedError(
            "Direct token decoding not yet implemented. "
            "Use the full VAE forward pass for reconstruction."
        )

    def save_pretrained(
        self,
        save_directory: str | Path,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save VAE model and config.

        Additionally saves a standalone VAE checkpoint for Stage 2 AR training.
        """
        # Call parent class save_pretrained to save full model
        super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)

        # Save standalone VAE checkpoint for AR training
        save_directory = Path(save_directory)
        vae_checkpoint_path = save_directory / "vae_model.pt"
        torch.save({
            "model_state_dict": self.vae.state_dict(),
            "config": {
                "vocab_size": self.config.vocab_size,
                "z_channels": self.config.vocab_ch,
                "ch": self.config.vch,
                "action_dim": self.action_dim,
                "num_actions": self.config.action_horizon,
                "patch_nums": self.config.patch_nums,
            }
        }, vae_checkpoint_path)
        print(f"✓ Saved VAE checkpoint to: {vae_checkpoint_path}")
